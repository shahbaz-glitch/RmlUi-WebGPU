#include "RmlUi_Renderer_WebGPU.h"
#include "RmlUi_Shaders_WebGPU.h"

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/MeshUtilities.h>
#include <RmlUi/Core/SystemInterface.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace {

using namespace RmlWebGPU;

constexpr uint32_t kMaxGradientStops = 16;
constexpr uint32_t kStencilMask = 0xffu;

static uint32_t Align256(uint32_t value)
{
    return (value + 255u) & ~255u;
}

static bool ValidDimensions(int w, int h)
{
    return w > 0 && h > 0;
}

static WGPUTextureView CreateView(WGPUTexture texture, WGPUTextureFormat format)
{
    if (!texture)
        return nullptr;
    WGPUTextureViewDescriptor d = {};
    d.format = format;
    d.dimension = WGPUTextureViewDimension_2D;
    d.baseMipLevel = 0;
    d.mipLevelCount = 1;
    d.baseArrayLayer = 0;
    d.arrayLayerCount = 1;
    return wgpuTextureCreateView(texture, &d);
}

static WGPUTexture CreateColorTexture(WGPUDevice device, uint32_t width, uint32_t height, WGPUTextureFormat format)
{
    WGPUTextureDescriptor d = {};
    d.dimension = WGPUTextureDimension_2D;
    d.size = {width, height, 1};
    d.format = format;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    d.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding |
              WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    return wgpuDeviceCreateTexture(device, &d);
}

static WGPUTexture CreateDepthStencilTexture(WGPUDevice device, uint32_t width, uint32_t height, WGPUTextureFormat format)
{
    WGPUTextureDescriptor d = {};
    d.dimension = WGPUTextureDimension_2D;
    d.size = {width, height, 1};
    d.format = format;
    d.mipLevelCount = 1;
    d.sampleCount = 1;
    d.usage = WGPUTextureUsage_RenderAttachment;
    return wgpuDeviceCreateTexture(device, &d);
}

static WGPUTextureView CreateDepthView(WGPUTexture texture, WGPUTextureFormat format)
{
    if (!texture)
        return nullptr;
    WGPUTextureViewDescriptor d = {};
    d.format = format;
    d.dimension = WGPUTextureViewDimension_2D;
    d.baseMipLevel = 0;
    d.mipLevelCount = 1;
    d.baseArrayLayer = 0;
    d.arrayLayerCount = 1;
    return wgpuTextureCreateView(texture, &d);
}

static WGPUBuffer CreateUniformBuffer(WGPUDevice device, uint64_t size)
{
    WGPUBufferDescriptor d = {};
    d.size = size;
    d.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    return wgpuDeviceCreateBuffer(device, &d);
}

static WGPUBlendState PremultipliedBlend()
{
    WGPUBlendState b = {};
    b.color.srcFactor = WGPUBlendFactor_One;
    b.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    b.color.operation = WGPUBlendOperation_Add;
    b.alpha.srcFactor = WGPUBlendFactor_One;
    b.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    b.alpha.operation = WGPUBlendOperation_Add;
    return b;
}

static WGPUVertexBufferLayout VertexLayout()
{
    static WGPUVertexAttribute attributes[3] = {};
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = offsetof(Rml::Vertex, position);
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Unorm8x4;
    attributes[1].offset = offsetof(Rml::Vertex, colour);
    attributes[1].shaderLocation = 1;
    attributes[2].format = WGPUVertexFormat_Float32x2;
    attributes[2].offset = offsetof(Rml::Vertex, tex_coord);
    attributes[2].shaderLocation = 2;

    WGPUVertexBufferLayout layout = {};
    layout.arrayStride = sizeof(Rml::Vertex);
    layout.attributeCount = 3;
    layout.attributes = attributes;
    layout.stepMode = WGPUVertexStepMode_Vertex;
    return layout;
}

static WGPUStencilFaceState StencilFace(WGPUCompareFunction compare, WGPUStencilOperation pass)
{
    WGPUStencilFaceState s = {};
    s.compare = compare;
    s.failOp = WGPUStencilOperation_Keep;
    s.depthFailOp = WGPUStencilOperation_Keep;
    s.passOp = pass;
    return s;
}

static WGPUColorTargetState ColorTarget(WGPUTextureFormat format, const WGPUBlendState* blend)
{
    WGPUColorTargetState t = {};
    t.format = format;
    t.blend = blend;
    t.writeMask = WGPUColorWriteMask_All;
    return t;
}

static Rml::Colourf Colorf(Rml::ColourbPremultiplied c)
{
    Rml::Colourf r;
    for (int i = 0; i < 4; ++i)
        r[i] = float(c[i]) / 255.0f;
    return r;
}

static void MatrixToArray(const Rml::Matrix4f& m, float* out)
{
    const float* p = m.data();
    std::memcpy(out, p, sizeof(float) * 16);
}

static void ReportUnsupported(const char* what)
{
    Rml::Log::Message(Rml::Log::LT_WARNING, "WebGPU backend: unsupported %s.", what);
}

static Rml::CompiledGeometryHandle CreateFullscreenGeometry(RenderInterface_WebGPU* renderer, bool flip_y)
{
    Rml::Mesh mesh;
    Rml::MeshUtilities::GenerateQuad(mesh, Rml::Vector2f(-1), Rml::Vector2f(2), {});
    if (flip_y) {
        for (Rml::Vertex& vertex : mesh.vertices)
            vertex.tex_coord.y = 1.0f - vertex.tex_coord.y;
    }
    return renderer->CompileGeometry(mesh.vertices, mesh.indices);
}

} // namespace

namespace {

enum class FilterType { Invalid = 0, Passthrough, Blur, DropShadow, ColorMatrix, MaskImage };

struct CompiledFilter {
    FilterType type = FilterType::Invalid;
    float blend_factor = 1.0f;
    float sigma = 0.0f;
    Rml::Vector2f offset;
    Rml::ColourbPremultiplied color;
    Rml::Matrix4f color_matrix;
    Gfx::WebGPUTexture* mask_texture = nullptr;
};

enum class ShaderType { Invalid = 0, Gradient, Creation };

struct CompiledShader {
    ShaderType type = ShaderType::Invalid;
    int func = 0;
    Rml::Vector2f p;
    Rml::Vector2f v;
    std::vector<float> stop_positions;
    std::vector<Rml::Colourf> stop_colors;
    Rml::Vector2f dimensions;
};

static void DestroyTextureResource(Gfx::WebGPUTexture& t)
{
    if (t.bind_group) wgpuBindGroupRelease(t.bind_group);
    if (t.view) wgpuTextureViewRelease(t.view);
    if (t.texture) wgpuTextureRelease(t.texture);
    t = {};
}

static Gfx::WebGPUTexture* CreateTextureResource(WGPUDevice device, WGPUBindGroupLayout bgl,
    WGPUSampler sampler, WGPUTextureFormat format, uint32_t width, uint32_t height)
{
    if (!ValidDimensions((int)width, (int)height))
        return nullptr;
    auto* t = new Gfx::WebGPUTexture();
    t->texture = CreateColorTexture(device, width, height, format);
    if (!t->texture) { delete t; return nullptr; }
    t->view = CreateView(t->texture, format);
    if (!t->view) { DestroyTextureResource(*t); delete t; return nullptr; }

    WGPUBindGroupEntry entries[2] = {};
    entries[0].binding = 0;
    entries[0].sampler = sampler;
    entries[1].binding = 1;
    entries[1].textureView = t->view;

    WGPUBindGroupDescriptor bg = {};
    bg.layout = bgl;
    bg.entryCount = 2;
    bg.entries = entries;
    t->bind_group = wgpuDeviceCreateBindGroup(device, &bg);
    if (!t->bind_group) { DestroyTextureResource(*t); delete t; return nullptr; }
    t->width = width;
    t->height = height;
    return t;
}

} // namespace

// ============================================================================
// RenderLayerStack
// ============================================================================

void RenderInterface_WebGPU::RenderLayerStack::Initialize(
    WGPUDevice device, int width, int height, WGPUTextureFormat color_format,
    WGPUTextureFormat depth_stencil_format, WGPUBindGroupLayout texture_bgl,
    WGPUSampler sampler)
{
    Shutdown();
    m_device = device;
    m_color_format = color_format;
    m_depth_stencil_format = depth_stencil_format;
    m_texture_bgl = texture_bgl;
    m_sampler = sampler;
    m_width = std::max(width, 1);
    m_height = std::max(height, 1);

    m_shared_depth_stencil = CreateDepthStencilTexture(m_device, uint32_t(m_width), uint32_t(m_height), m_depth_stencil_format);
    m_shared_depth_stencil_view = CreateDepthView(m_shared_depth_stencil, m_depth_stencil_format);

    m_fb_layers.resize(1); // layer 0 is the permanent base layer.
    CreateFramebuffer(m_fb_layers[0], m_width, m_height, true);
    m_layers_size = 1;

    for (auto& fb : m_fb_postprocess)
        CreateFramebuffer(fb, m_width, m_height, false);
}

void RenderInterface_WebGPU::RenderLayerStack::Shutdown()
{
    DestroyFramebuffers();
    m_fb_layers.clear();
    m_layers_size = 0;
    m_device = nullptr;
    m_texture_bgl = nullptr;
    m_sampler = nullptr;
    m_width = m_height = 0;
}

void RenderInterface_WebGPU::RenderLayerStack::CreateFramebuffer(
    Gfx::WebGPUFramebuffer& fb, int width, int height, bool has_depth_stencil)
{
    DestroyFramebuffer(fb);
    fb.width = uint32_t(std::max(width, 1));
    fb.height = uint32_t(std::max(height, 1));
    fb.color_texture = CreateColorTexture(m_device, fb.width, fb.height, m_color_format);
    if (!fb.color_texture) return;
    fb.color_view = CreateView(fb.color_texture, m_color_format);

    if (has_depth_stencil) {
        fb.depth_stencil_view = m_shared_depth_stencil_view;
    }

    if (fb.color_view) {
        WGPUBindGroupEntry e[2] = {};
        e[0].binding = 0;
        e[0].sampler = m_sampler;
        e[1].binding = 1;
        e[1].textureView = fb.color_view;
        WGPUBindGroupDescriptor d = {};
        d.layout = m_texture_bgl;
        d.entryCount = 2;
        d.entries = e;
        fb.texture_bind_group = wgpuDeviceCreateBindGroup(m_device, &d);
    }
}

void RenderInterface_WebGPU::RenderLayerStack::DestroyFramebuffer(Gfx::WebGPUFramebuffer& fb)
{
    if (fb.texture_bind_group) wgpuBindGroupRelease(fb.texture_bind_group);
    if (fb.depth_stencil_view && fb.depth_stencil_view != m_shared_depth_stencil_view)
        wgpuTextureViewRelease(fb.depth_stencil_view);
    if (fb.color_view) wgpuTextureViewRelease(fb.color_view);
    if (fb.color_texture) wgpuTextureRelease(fb.color_texture);
    fb = {};
}

void RenderInterface_WebGPU::RenderLayerStack::DestroyFramebuffers()
{
    for (auto& fb : m_fb_layers)
        DestroyFramebuffer(fb);
    for (auto& fb : m_fb_postprocess)
        DestroyFramebuffer(fb);

    if (m_shared_depth_stencil_view) wgpuTextureViewRelease(m_shared_depth_stencil_view);
    if (m_shared_depth_stencil) wgpuTextureRelease(m_shared_depth_stencil);
    m_shared_depth_stencil_view = nullptr;
    m_shared_depth_stencil = nullptr;
}

void RenderInterface_WebGPU::RenderLayerStack::Resize(int width, int height)
{
    if (!m_device || width <= 0 || height <= 0) return;
    m_width = width;
    m_height = height;

    const int active_layers = std::max(m_layers_size, 1);
    m_layers_size = 0;
    for (auto& fb : m_fb_layers) DestroyFramebuffer(fb);
    for (auto& fb : m_fb_postprocess) DestroyFramebuffer(fb);
    if (m_shared_depth_stencil_view) wgpuTextureViewRelease(m_shared_depth_stencil_view);
    if (m_shared_depth_stencil) wgpuTextureRelease(m_shared_depth_stencil);
    m_shared_depth_stencil_view = nullptr;
    m_shared_depth_stencil = CreateDepthStencilTexture(m_device, uint32_t(width), uint32_t(height), m_depth_stencil_format);
    m_shared_depth_stencil_view = CreateDepthView(m_shared_depth_stencil, m_depth_stencil_format);
    m_fb_layers.clear();
    m_fb_layers.resize(std::max(active_layers, 1));
    for (auto& fb : m_fb_layers) CreateFramebuffer(fb, width, height, true);
    m_layers_size = active_layers;
    for (auto& fb : m_fb_postprocess) CreateFramebuffer(fb, width, height, false);
}

Rml::LayerHandle RenderInterface_WebGPU::RenderLayerStack::PushLayer()
{
    const int index = m_layers_size;
    if (index >= (int)m_fb_layers.size()) {
        m_fb_layers.emplace_back();
        CreateFramebuffer(m_fb_layers.back(), m_width, m_height, true);
    }
    m_layers_size = index + 1;
    return Rml::LayerHandle(index);
}

void RenderInterface_WebGPU::RenderLayerStack::PopLayer()
{
    if (m_layers_size > 1)
        --m_layers_size;
}

const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetLayer(Rml::LayerHandle layer) const
{
    const size_t i = size_t(layer);
    RMLUI_ASSERT(i < size_t(m_layers_size));
    return m_fb_layers[i];
}

const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetTopLayer() const
{
    return GetLayer(GetTopLayerHandle());
}

Rml::LayerHandle RenderInterface_WebGPU::RenderLayerStack::GetTopLayerHandle() const
{
    return Rml::LayerHandle(std::max(m_layers_size - 1, 0));
}

const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetPostprocessPrimary() { return m_fb_postprocess[0]; }
const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetPostprocessSecondary() { return m_fb_postprocess[1]; }
const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetPostprocessTertiary() { return m_fb_postprocess[2]; }
const Gfx::WebGPUFramebuffer& RenderInterface_WebGPU::RenderLayerStack::GetBlendMask() { return m_fb_postprocess[3]; }
void RenderInterface_WebGPU::RenderLayerStack::SwapPostprocessPrimarySecondary() { std::swap(m_fb_postprocess[0], m_fb_postprocess[1]); }

// ============================================================================
// Main renderer
// ============================================================================

RenderInterface_WebGPU::RenderInterface_WebGPU()
{
    m_transform = Rml::Matrix4f::Identity();
    m_projection = Rml::Matrix4f::Identity();
}

RenderInterface_WebGPU::~RenderInterface_WebGPU()
{
    Shutdown();
}

bool RenderInterface_WebGPU::Initialize(WGPUDevice device, WGPUQueue queue,
    WGPUTextureFormat render_target_format, WGPUTextureFormat depth_stencil_format)
{
    if (!device || !queue || render_target_format == WGPUTextureFormat_Undefined)
        return false;

    Shutdown();
    m_device = device;
    m_queue = queue;
    m_render_format = render_target_format;
    m_depth_stencil_format = depth_stencil_format;

    CreateBindGroupLayouts();
    CreateDefaultSamplers();
    if (!m_bgl_transform || !m_bgl_texture || !m_bgl_blend_mask || !m_bgl_filter_uniform || !m_sampler_linear) {
        Shutdown();
        return false;
    }

    m_transform_uniform_buffer = CreateUniformBuffer(m_device, kTransformUniformBufferSize);
    m_filter_uniform_buffer = CreateUniformBuffer(m_device, kFilterUniformBufferSize);
    if (!m_transform_uniform_buffer || !m_filter_uniform_buffer) {
        Shutdown();
        return false;
    }

    WGPUBindGroupEntry te = {};
    te.binding = 0;
    te.buffer = m_transform_uniform_buffer;
    te.size = sizeof(TransformUniform);
    WGPUBindGroupDescriptor td = {};
    td.layout = m_bgl_transform;
    td.entryCount = 1;
    td.entries = &te;
    m_transform_bind_group = wgpuDeviceCreateBindGroup(m_device, &td);

    WGPUBindGroupEntry fe = {};
    fe.binding = 0;
    fe.buffer = m_filter_uniform_buffer;
    fe.size = kFilterUniformBufferSize;
    WGPUBindGroupDescriptor fd = {};
    fd.layout = m_bgl_filter_uniform;
    fd.entryCount = 1;
    fd.entries = &fe;
    m_filter_bind_group = wgpuDeviceCreateBindGroup(m_device, &fd);
    if (!m_transform_bind_group || !m_filter_bind_group) {
        Shutdown();
        return false;
    }

    CreatePipelines();
    bool all_pipelines_valid = true;
    for (size_t i = 0; i < static_cast<size_t>(WebGPUPipelineId::Count); ++i)
        all_pipelines_valid = all_pipelines_valid && (m_pipelines[i] != nullptr);
    if (!all_pipelines_valid) {
        Shutdown();
        return false;
    }

    const int layer_width = std::max(m_viewport_width, 1);
    const int layer_height = std::max(m_viewport_height, 1);
    m_layer_stack.Initialize(m_device, layer_width, layer_height, m_render_format,
        m_depth_stencil_format, m_bgl_texture, m_sampler_linear);
    if (!m_layer_stack.GetLayer(Rml::LayerHandle(0)).color_view) {
        Shutdown();
        return false;
    }

    m_fullscreen_quad_geometry = CreateFullscreenGeometry(this, true);
    if (!m_fullscreen_quad_geometry) {
        Shutdown();
        return false;
    }

    m_initialized = true;
    return true;
}

void RenderInterface_WebGPU::Shutdown()
{
    EndCurrentRenderPass();
    m_layer_stack.Shutdown();

    if (m_fullscreen_quad_geometry) {
        ReleaseGeometry(m_fullscreen_quad_geometry);
        m_fullscreen_quad_geometry = {};
    }

    for (auto& p : m_pipelines) {
        if (p) wgpuRenderPipelineRelease(p);
        p = nullptr;
    }

    if (m_pipeline_layout_filter) wgpuPipelineLayoutRelease(m_pipeline_layout_filter);
    if (m_pipeline_layout_blend_mask) wgpuPipelineLayoutRelease(m_pipeline_layout_blend_mask);
    if (m_pipeline_layout_transform_texture) wgpuPipelineLayoutRelease(m_pipeline_layout_transform_texture);
    if (m_pipeline_layout_transform) wgpuPipelineLayoutRelease(m_pipeline_layout_transform);
    m_pipeline_layout_filter = nullptr;
    m_pipeline_layout_blend_mask = nullptr;
    m_pipeline_layout_transform_texture = nullptr;
    m_pipeline_layout_transform = nullptr;

    if (m_transform_bind_group) wgpuBindGroupRelease(m_transform_bind_group);
    if (m_filter_bind_group) wgpuBindGroupRelease(m_filter_bind_group);
    if (m_transform_uniform_buffer) wgpuBufferRelease(m_transform_uniform_buffer);
    if (m_filter_uniform_buffer) wgpuBufferRelease(m_filter_uniform_buffer);
    m_transform_bind_group = nullptr;
    m_filter_bind_group = nullptr;
    m_transform_uniform_buffer = nullptr;
    m_filter_uniform_buffer = nullptr;

    if (m_sampler_clamp) wgpuSamplerRelease(m_sampler_clamp);
    if (m_sampler_linear) wgpuSamplerRelease(m_sampler_linear);
    m_sampler_clamp = nullptr;
    m_sampler_linear = nullptr;

    if (m_bgl_filter_uniform) wgpuBindGroupLayoutRelease(m_bgl_filter_uniform);
    if (m_bgl_blend_mask) wgpuBindGroupLayoutRelease(m_bgl_blend_mask);
    if (m_bgl_texture) wgpuBindGroupLayoutRelease(m_bgl_texture);
    if (m_bgl_transform) wgpuBindGroupLayoutRelease(m_bgl_transform);
    m_bgl_filter_uniform = nullptr;
    m_bgl_blend_mask = nullptr;
    m_bgl_texture = nullptr;
    m_bgl_transform = nullptr;

    if (m_command_encoder) wgpuCommandEncoderRelease(m_command_encoder);
    m_command_encoder = nullptr;
    m_render_pass = nullptr;
    m_target_view = nullptr;
    m_owns_render_pass = false;
    m_initialized = false;
    m_device = nullptr;
    m_queue = nullptr;
}

void RenderInterface_WebGPU::SetViewport(int width, int height)
{
    m_viewport_width = std::max(width, 1);
    m_viewport_height = std::max(height, 1);

    // WebGPU framebuffer Y grows downward, while RmlUi's coordinates start at top-left.
    // Reverse the Y interval compared with the OpenGL backend.
    m_projection = Rml::Matrix4f::ProjectOrtho(
        0.0f, float(m_viewport_width),
        0.0f, float(m_viewport_height),
        -10000.0f, 10000.0f);

    if (m_initialized)
        m_layer_stack.Resize(m_viewport_width, m_viewport_height);
}

void RenderInterface_WebGPU::BeginFrame(WGPUTextureView target_view)
{
    if (!m_initialized || !target_view || m_viewport_width <= 0 || m_viewport_height <= 0)
        return;

    EndCurrentRenderPass();
    if (m_command_encoder) {
        wgpuCommandEncoderRelease(m_command_encoder);
        m_command_encoder = nullptr;
    }

    WGPUCommandEncoderDescriptor ed = {};
    m_command_encoder = wgpuDeviceCreateCommandEncoder(m_device, &ed);
    m_target_view = target_view;
    m_current_uniform_offset = 0;
    m_scissor_enabled = false;
    m_scissor_state = Rml::Rectanglei::MakeInvalid();
    m_stencil_enabled = false;
    m_stencil_equal = false;
    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_current_clip_operation = -1;
    if (m_layer_stack.GetTopLayerHandle() == Rml::LayerHandle(0) && m_viewport_width > 0 && m_viewport_height > 0) {
        // The stack is initialized during Initialize/SetViewport; this guard keeps the base layer valid.
    }
    SetTransform(nullptr);

    BeginLayerRenderPass(m_layer_stack.GetLayer(0), true, true);
}

void RenderInterface_WebGPU::EndFrame()
{
    if (!m_command_encoder)
        return;

    EndCurrentRenderPass();

    const auto& base = m_layer_stack.GetLayer(0);
    const auto& primary = m_layer_stack.GetPostprocessPrimary();

    WGPUExtent3D extent = {base.width, base.height, 1};
    WGPUTexelCopyTextureInfo src = {};
    src.texture = base.color_texture;
    WGPUTexelCopyTextureInfo dst = {};
    dst.texture = primary.color_texture;
    wgpuCommandEncoderCopyTextureToTexture(m_command_encoder, &src, &dst, &extent);

    WGPUColor clear = {0, 0, 0, 0};
    WGPURenderPassColorAttachment ca = {};
    ca.view = m_target_view;
    ca.loadOp = WGPULoadOp_Load;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = clear;

    WGPURenderPassDepthStencilAttachment ds = {};
    ds.view = base.depth_stencil_view;
    ds.depthLoadOp = WGPULoadOp_Clear;
    ds.depthStoreOp = WGPUStoreOp_Store;
    ds.depthClearValue = 1.0f;
    ds.stencilLoadOp = WGPULoadOp_Clear;
    ds.stencilStoreOp = WGPUStoreOp_Store;
    ds.stencilClearValue = 0;

    WGPURenderPassDescriptor rp = {};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    rp.depthStencilAttachment = base.depth_stencil_view ? &ds : nullptr;
    m_render_pass = wgpuCommandEncoderBeginRenderPass(m_command_encoder, &rp);
    m_owns_render_pass = true;

    UsePipeline(WebGPUPipelineId::Passthrough);
    auto& pf = const_cast<Gfx::WebGPUFramebuffer&>(primary);
    if (pf.texture_bind_group)
        wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, pf.texture_bind_group, 0, nullptr);
    DrawFullscreenQuad();
    EndCurrentRenderPass();

    WGPUCommandBufferDescriptor cbd = {};
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(m_command_encoder, &cbd);
    if (command_buffer) {
        wgpuQueueSubmit(m_queue, 1, &command_buffer);
        wgpuCommandBufferRelease(command_buffer);
    }
    wgpuCommandEncoderRelease(m_command_encoder);
    m_command_encoder = nullptr;
    m_target_view = nullptr;
}

void RenderInterface_WebGPU::SetActiveRenderPass(WGPURenderPassEncoder pass_encoder)
{
    EndCurrentRenderPass();
    m_render_pass = pass_encoder;
    m_owns_render_pass = false;
}

void RenderInterface_WebGPU::CreateBindGroupLayouts()
{
    WGPUBindGroupLayoutEntry transform = {};
    transform.binding = 0;
    transform.visibility = WGPUShaderStage_Vertex;
    transform.buffer.type = WGPUBufferBindingType_Uniform;
    transform.buffer.hasDynamicOffset = true;
    transform.buffer.minBindingSize = sizeof(TransformUniform);
    WGPUBindGroupLayoutDescriptor td = {};
    td.entryCount = 1;
    td.entries = &transform;
    m_bgl_transform = wgpuDeviceCreateBindGroupLayout(m_device, &td);

    WGPUBindGroupLayoutEntry tex[2] = {};
    tex[0].binding = 0;
    tex[0].visibility = WGPUShaderStage_Fragment;
    tex[0].sampler.type = WGPUSamplerBindingType_Filtering;
    tex[1].binding = 1;
    tex[1].visibility = WGPUShaderStage_Fragment;
    tex[1].texture.sampleType = WGPUTextureSampleType_Float;
    tex[1].texture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor txd = {};
    txd.entryCount = 2;
    txd.entries = tex;
    m_bgl_texture = wgpuDeviceCreateBindGroupLayout(m_device, &txd);

    WGPUBindGroupLayoutEntry mask[3] = {};
    mask[0] = tex[0];
    mask[1] = tex[1];
    mask[2] = tex[1];
    mask[2].binding = 2;
    WGPUBindGroupLayoutDescriptor md = {};
    md.entryCount = 3;
    md.entries = mask;
    m_bgl_blend_mask = wgpuDeviceCreateBindGroupLayout(m_device, &md);

    WGPUBindGroupLayoutEntry filter = {};
    filter.binding = 0;
    filter.visibility = WGPUShaderStage_Fragment;
    filter.buffer.type = WGPUBufferBindingType_Uniform;
    filter.buffer.minBindingSize = 16;
    WGPUBindGroupLayoutDescriptor fd = {};
    fd.entryCount = 1;
    fd.entries = &filter;
    m_bgl_filter_uniform = wgpuDeviceCreateBindGroupLayout(m_device, &fd);
}

void RenderInterface_WebGPU::CreateDefaultSamplers()
{
    WGPUSamplerDescriptor d = {};
    d.addressModeU = WGPUAddressMode_ClampToEdge;
    d.addressModeV = WGPUAddressMode_ClampToEdge;
    d.addressModeW = WGPUAddressMode_ClampToEdge;
    d.magFilter = WGPUFilterMode_Linear;
    d.minFilter = WGPUFilterMode_Linear;
    d.mipmapFilter = WGPUMipmapFilterMode_Linear;
    d.maxAnisotropy = 1;
    m_sampler_linear = wgpuDeviceCreateSampler(m_device, &d);
    m_sampler_clamp = m_sampler_linear;
    if (m_sampler_clamp) wgpuSamplerAddRef(m_sampler_clamp);
}

void RenderInterface_WebGPU::CreatePipelines()
{
    WGPUShaderModule vs_main = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_vert_main);
    WGPUShaderModule vs_pass = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_vert_passthrough);
    WGPUShaderModule fs_color = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_color);
    WGPUShaderModule fs_texture = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_texture);
    WGPUShaderModule fs_gradient = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_gradient);
    WGPUShaderModule fs_creation = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_creation);
    WGPUShaderModule fs_pass = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_passthrough);
    WGPUShaderModule fs_matrix = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_color_matrix);
    WGPUShaderModule fs_mask = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_blend_mask);
    WGPUShaderModule fs_blur = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_blur);
    WGPUShaderModule fs_shadow = RmlWebGPU::CreateShaderModule(m_device, RmlWebGPU::s_shader_frag_drop_shadow);

    WGPUBindGroupLayout filter_groups[] = {m_bgl_transform, m_bgl_texture, m_bgl_filter_uniform};
    WGPUPipelineLayoutDescriptor fld = {};
    fld.bindGroupLayoutCount = 3;
    fld.bindGroupLayouts = filter_groups;
    m_pipeline_layout_filter = wgpuDeviceCreatePipelineLayout(m_device, &fld);

    WGPUBindGroupLayout texture_groups[] = {m_bgl_transform, m_bgl_texture};
    WGPUPipelineLayoutDescriptor tld = {};
    tld.bindGroupLayoutCount = 2;
    tld.bindGroupLayouts = texture_groups;
    m_pipeline_layout_transform_texture = wgpuDeviceCreatePipelineLayout(m_device, &tld);

    WGPUBindGroupLayout mask_groups[] = {m_bgl_transform, m_bgl_blend_mask};
    WGPUPipelineLayoutDescriptor mld = {};
    mld.bindGroupLayoutCount = 2;
    mld.bindGroupLayouts = mask_groups;
    m_pipeline_layout_blend_mask = wgpuDeviceCreatePipelineLayout(m_device, &mld);

    WGPUBindGroupLayout only_transform[] = {m_bgl_transform};
    WGPUPipelineLayoutDescriptor old = {};
    old.bindGroupLayoutCount = 1;
    old.bindGroupLayouts = only_transform;
    m_pipeline_layout_transform = wgpuDeviceCreatePipelineLayout(m_device, &old);

    const WGPUBlendState blend = PremultipliedBlend();
    const WGPUVertexBufferLayout vl = VertexLayout();

    auto make = [&](WebGPUPipelineId id, WGPUPipelineLayout layout, WGPUShaderModule vs,
                    WGPUShaderModule fs, bool textured, WGPUCompareFunction stencil_compare,
                    WGPUStencilOperation stencil_pass, bool enable_blend, bool is_passthrough,
                    bool write_color = true, bool use_depth_stencil = true) {
        WGPUStencilFaceState sf = StencilFace(stencil_compare, stencil_pass);
        WGPUDepthStencilState ds = {};
        ds.format = m_depth_stencil_format;
        ds.depthWriteEnabled = false;
        ds.depthCompare = WGPUCompareFunction_Always;
        ds.stencilFront = sf;
        ds.stencilBack = sf;
        ds.stencilReadMask = kStencilMask;
        ds.stencilWriteMask = kStencilMask;

        WGPUColorTargetState ct = ColorTarget(m_render_format, enable_blend ? &blend : nullptr);
        if (!write_color) ct.writeMask = WGPUColorWriteMask_None;
        WGPUFragmentState frag = {};
        frag.module = fs;
        frag.entryPoint = {"main", 4};
        frag.targetCount = 1;
        frag.targets = &ct;

        WGPUVertexState vertex = {};
        vertex.module = vs;
        vertex.entryPoint = {"main", 4};
        vertex.bufferCount = 1;
        vertex.buffers = &vl;

        WGPUPrimitiveState primitive = {};
        primitive.topology = WGPUPrimitiveTopology_TriangleList;
        primitive.frontFace = WGPUFrontFace_CCW;
        primitive.cullMode = WGPUCullMode_None;

        WGPUMultisampleState ms = {};
        ms.count = 1;
        ms.mask = ~0u;

        WGPURenderPipelineDescriptor p = {};
        p.layout = layout;
        p.vertex = vertex;
        p.primitive = primitive;
        p.depthStencil = use_depth_stencil ? &ds : nullptr;
        p.multisample = ms;
        p.fragment = &frag;
        m_pipelines[(size_t)id] = wgpuDeviceCreateRenderPipeline(m_device, &p);
        (void)textured;
        (void)is_passthrough;
    };

    make(WebGPUPipelineId::Color_Stencil_Disabled, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, true, false);
    make(WebGPUPipelineId::Color_Stencil_Always, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, true, false);
    make(WebGPUPipelineId::Color_Stencil_Equal, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep, true, false);
    make(WebGPUPipelineId::Color_Stencil_Set, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Always, WGPUStencilOperation_Replace, false, false, false);
    make(WebGPUPipelineId::Color_Stencil_SetInverse, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Always, WGPUStencilOperation_Replace, false, false, false);
    make(WebGPUPipelineId::Color_Stencil_Intersect, m_pipeline_layout_transform, vs_main, fs_color, false, WGPUCompareFunction_Always, WGPUStencilOperation_IncrementClamp, false, false, false);

    make(WebGPUPipelineId::Texture_Stencil_Disabled, m_pipeline_layout_transform_texture, vs_main, fs_texture, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, true, false);
    make(WebGPUPipelineId::Texture_Stencil_Always, m_pipeline_layout_transform_texture, vs_main, fs_texture, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, true, false);
    make(WebGPUPipelineId::Texture_Stencil_Equal, m_pipeline_layout_transform_texture, vs_main, fs_texture, true, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep, true, false);

    make(WebGPUPipelineId::Gradient, m_pipeline_layout_filter, vs_main, fs_gradient, false, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep, true, false, true);
    make(WebGPUPipelineId::Creation, m_pipeline_layout_filter, vs_main, fs_creation, false, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep, true, false, true);
    make(WebGPUPipelineId::Passthrough, m_pipeline_layout_transform_texture, vs_pass, fs_pass, true, WGPUCompareFunction_Equal, WGPUStencilOperation_Keep, true, true, true);
    make(WebGPUPipelineId::Passthrough_NoBlend, m_pipeline_layout_transform_texture, vs_pass, fs_pass, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, false, true, true);
    make(WebGPUPipelineId::ColorMatrix, m_pipeline_layout_filter, vs_pass, fs_matrix, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, false, true, false);
    make(WebGPUPipelineId::BlendMask, m_pipeline_layout_blend_mask, vs_pass, fs_mask, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, false, true, false);
    make(WebGPUPipelineId::Blur, m_pipeline_layout_filter, vs_pass, fs_blur, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, false, true, false);
    make(WebGPUPipelineId::DropShadow, m_pipeline_layout_filter, vs_pass, fs_shadow, true, WGPUCompareFunction_Always, WGPUStencilOperation_Keep, false, true, false);

    if (vs_main) wgpuShaderModuleRelease(vs_main);
    if (vs_pass) wgpuShaderModuleRelease(vs_pass);
    if (fs_color) wgpuShaderModuleRelease(fs_color);
    if (fs_texture) wgpuShaderModuleRelease(fs_texture);
    if (fs_gradient) wgpuShaderModuleRelease(fs_gradient);
    if (fs_creation) wgpuShaderModuleRelease(fs_creation);
    if (fs_pass) wgpuShaderModuleRelease(fs_pass);
    if (fs_matrix) wgpuShaderModuleRelease(fs_matrix);
    if (fs_mask) wgpuShaderModuleRelease(fs_mask);
    if (fs_blur) wgpuShaderModuleRelease(fs_blur);
    if (fs_shadow) wgpuShaderModuleRelease(fs_shadow);
}

void RenderInterface_WebGPU::BeginLayerRenderPass(const Gfx::WebGPUFramebuffer& fb, bool clear_color, bool clear_stencil)
{
    if (!m_command_encoder || !fb.color_view) return;
    EndCurrentRenderPass();

    WGPUColorAttachment ca = {};
    ca.view = fb.color_view;
    ca.loadOp = clear_color ? WGPULoadOp_Clear : WGPULoadOp_Load;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = {0, 0, 0, 0};

    WGPURenderPassDepthStencilAttachment ds = {};
    if (fb.depth_stencil_view) {
        ds.view = fb.depth_stencil_view;
        ds.depthLoadOp = WGPULoadOp_Clear;
        ds.depthStoreOp = WGPUStoreOp_Store;
        ds.depthClearValue = 1.0f;
        ds.stencilLoadOp = clear_stencil ? WGPULoadOp_Clear : WGPULoadOp_Load;
        ds.stencilStoreOp = WGPUStoreOp_Store;
        ds.stencilClearValue = 0;
    }

    WGPURenderPassDescriptor rp = {};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    rp.depthStencilAttachment = fb.depth_stencil_view ? &ds : nullptr;
    m_render_pass = wgpuCommandEncoderBeginRenderPass(m_command_encoder, &rp);
    m_owns_render_pass = true;
    m_active_pipeline = WebGPUPipelineId::Count;
    ApplyScissor();
}

void RenderInterface_WebGPU::EndCurrentRenderPass()
{
    if (m_render_pass && m_owns_render_pass) {
        wgpuRenderPassEncoderEnd(m_render_pass);
        wgpuRenderPassEncoderRelease(m_render_pass);
    }
    m_render_pass = nullptr;
    m_owns_render_pass = false;
}

void RenderInterface_WebGPU::ApplyScissor()
{
    if (!m_render_pass) return;
    int x = 0, y = 0, w = m_viewport_width, h = m_viewport_height;
    if (m_scissor_enabled && m_scissor_state.Valid()) {
        const int left = std::max(m_scissor_state.Left(), 0);
        const int top = std::max(m_scissor_state.Top(), 0);
        const int right = std::min(m_scissor_state.Right(), m_viewport_width);
        const int bottom = std::min(m_scissor_state.Bottom(), m_viewport_height);
        x = std::min(left, m_viewport_width);
        y = std::min(top, m_viewport_height);
        w = std::max(right - x, 0);
        h = std::max(bottom - y, 0);
    }
    wgpuRenderPassEncoderSetScissorRect(m_render_pass, uint32_t(x), uint32_t(y), uint32_t(w), uint32_t(h));
}

void RenderInterface_WebGPU::UsePipeline(WebGPUPipelineId pipeline_id)
{
    if (!m_render_pass) return;
    if (m_active_pipeline == pipeline_id) return;
    WGPURenderPipeline p = m_pipelines[(size_t)pipeline_id];
    if (p) {
        wgpuRenderPassEncoderSetPipeline(m_render_pass, p);
        m_active_pipeline = pipeline_id;
        if (m_stencil_enabled)
            wgpuRenderPassEncoderSetStencilReference(m_render_pass, m_stencil_ref_value);
    }
}

void RenderInterface_WebGPU::SubmitTransform(const Rml::Vector2f& translation)
{
    if (!m_render_pass || !m_transform_bind_group) return;
    if (m_current_uniform_offset + kUniformAlignment > kTransformUniformBufferSize) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "WebGPU transform uniform ring exhausted for one frame.");
        return;
    }

    TransformUniform u = {};
    MatrixToArray(m_transform, u.transform);
    u.translate[0] = translation.x;
    u.translate[1] = translation.y;
    wgpuQueueWriteBuffer(m_queue, m_transform_uniform_buffer, m_current_uniform_offset, &u, sizeof(u));
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 0, m_transform_bind_group, 1, &m_current_uniform_offset);
    m_current_uniform_offset += kUniformAlignment;
}

void RenderInterface_WebGPU::DrawFullscreenQuad()
{
    if (!m_render_pass || !m_fullscreen_quad_geometry) return;
    auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(m_fullscreen_quad_geometry);
    wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, g->vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, g->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(m_render_pass, g->num_indices, 1, 0, 0, 0);
}

Rml::CompiledGeometryHandle RenderInterface_WebGPU::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices)
{
    if (!m_device || vertices.empty() || indices.empty()) return {};

    auto* g = new Gfx::WebGPUGeometry();
    g->num_indices = uint32_t(indices.size());

    WGPUBufferDescriptor vb = {};
    vb.size = uint64_t(vertices.size_bytes());
    vb.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
    g->vertex_buffer = wgpuDeviceCreateBuffer(m_device, &vb);

    WGPUBufferDescriptor ib = {};
    ib.size = uint64_t(indices.size()) * sizeof(uint32_t);
    ib.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
    g->index_buffer = wgpuDeviceCreateBuffer(m_device, &ib);

    if (!g->vertex_buffer || !g->index_buffer) {
        if (g->vertex_buffer) wgpuBufferRelease(g->vertex_buffer);
        if (g->index_buffer) wgpuBufferRelease(g->index_buffer);
        delete g;
        return {};
    }

    wgpuQueueWriteBuffer(m_queue, g->vertex_buffer, 0, vertices.data(), vertices.size_bytes());
    wgpuQueueWriteBuffer(m_queue, g->index_buffer, 0, indices.data(), uint64_t(indices.size()) * sizeof(uint32_t));
    return reinterpret_cast<Rml::CompiledGeometryHandle>(g);
}

void RenderInterface_WebGPU::RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture)
{
    if (!m_render_pass || !handle) return;
    auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(handle);
    if (!g->vertex_buffer || !g->index_buffer) return;

    if (texture == TexturePostprocess) {
        UsePipeline(WebGPUPipelineId::Passthrough);
        SubmitTransform({});
        if (m_layer_stack.GetPostprocessPrimary().texture_bind_group)
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, m_layer_stack.GetPostprocessPrimary().texture_bind_group, 0, nullptr);
    } else if (texture != 0 && texture != TextureEnableWithoutBinding) {
        const auto* t = reinterpret_cast<const Gfx::WebGPUTexture*>(texture);
        UsePipeline(m_stencil_enabled && m_stencil_equal ? WebGPUPipelineId::Texture_Stencil_Equal : WebGPUPipelineId::Texture_Stencil_Disabled);
        SubmitTransform(translation);
        if (t && t->bind_group)
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, t->bind_group, 0, nullptr);
    } else {
        UsePipeline(m_stencil_enabled && m_stencil_equal ? WebGPUPipelineId::Color_Stencil_Equal : WebGPUPipelineId::Color_Stencil_Disabled);
        SubmitTransform(translation);
    }

    wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, g->vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, g->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(m_render_pass, g->num_indices, 1, 0, 0, 0);
}

void RenderInterface_WebGPU::ReleaseGeometry(Rml::CompiledGeometryHandle handle)
{
    auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(handle);
    if (!g) return;
    if (g->vertex_buffer) wgpuBufferRelease(g->vertex_buffer);
    if (g->index_buffer) wgpuBufferRelease(g->index_buffer);
    delete g;
}

Rml::TextureHandle RenderInterface_WebGPU::GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions)
{
    if (!m_device || !ValidDimensions(source_dimensions.x, source_dimensions.y)) return {};
    const size_t required = size_t(source_dimensions.x) * size_t(source_dimensions.y) * 4;
    if (!source_data.data() || source_data.size() != required) return {};

    auto* t = CreateTextureResource(m_device, m_bgl_texture, m_sampler_linear, WGPUTextureFormat_RGBA8Unorm,
                                    uint32_t(source_dimensions.x), uint32_t(source_dimensions.y));
    if (!t) return {};

    const uint32_t row_bytes = uint32_t(source_dimensions.x * 4);
    const uint32_t padded = Align256(row_bytes);
    std::vector<uint8_t> staging(size_t(padded) * source_dimensions.y, 0);
    // RmlUi texture coordinates use a bottom-left origin. WebGPU texture coordinates
    // use a top-left origin, so upload the rows in reverse order.
    for (int y = 0; y < source_dimensions.y; ++y) {
        const int src_y = source_dimensions.y - 1 - y;
        std::memcpy(staging.data() + size_t(y) * padded,
                    source_data.data() + size_t(src_y) * row_bytes, row_bytes);
    }

    // QueueWriteTexture is the simplest path and does not require a staging buffer object.
    WGPUTexelCopyTextureInfo dst = {};
    dst.texture = t->texture;
    WGPUExtent3D extent = {uint32_t(source_dimensions.x), uint32_t(source_dimensions.y), 1};
    WGPUTexelCopyBufferLayout qlayout = {};
    qlayout.offset = 0;
    qlayout.bytesPerRow = padded;
    qlayout.rowsPerImage = uint32_t(source_dimensions.y);
    wgpuQueueWriteTexture(m_queue, &dst, staging.data(), staging.size(), &qlayout, &extent);

    return reinterpret_cast<Rml::TextureHandle>(t);
}

#pragma pack(push, 1)
struct TGAHeader {
    uint8_t idLength;
    uint8_t colorMapType;
    uint8_t dataType;
    uint16_t colorMapOrigin;
    uint16_t colorMapLength;
    uint8_t colorMapDepth;
    uint16_t xOrigin;
    uint16_t yOrigin;
    uint16_t width;
    uint16_t height;
    uint8_t bitsPerPixel;
    uint8_t imageDescriptor;
};
#pragma pack(pop)

Rml::TextureHandle RenderInterface_WebGPU::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source)
{
    auto* fi = Rml::GetFileInterface();
    if (!fi) return {};
    Rml::FileHandle fh = fi->Open(source);
    if (!fh) return {};
    fi->Seek(fh, 0, SEEK_END);
    const size_t size = fi->Tell(fh);
    fi->Seek(fh, 0, SEEK_SET);
    if (size < sizeof(TGAHeader)) { fi->Close(fh); return {}; }

    std::vector<Rml::byte> data(size);
    if (fi->Read(data.data(), size, fh) != size) { fi->Close(fh); return {}; }
    fi->Close(fh);

    TGAHeader h = {};
    std::memcpy(&h, data.data(), sizeof(h));
    if (h.dataType != 2 || h.colorMapType != 0 || (h.bitsPerPixel != 24 && h.bitsPerPixel != 32) || h.width == 0 || h.height == 0) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "WebGPU TGA loader: unsupported or invalid TGA '%s'.", source.c_str());
        return {};
    }

    const size_t pixel_offset = sizeof(TGAHeader) + h.idLength;
    const size_t pixel_bytes = size_t(h.width) * h.height * (h.bitsPerPixel / 8);
    if (pixel_offset > size || pixel_bytes > size - pixel_offset) return {};

    const uint8_t* src = reinterpret_cast<const uint8_t*>(data.data()) + pixel_offset;
    std::vector<Rml::byte> rgba(size_t(h.width) * h.height * 4);
    const bool top_to_bottom = (h.imageDescriptor & 0x20) != 0;
    const int channels = h.bitsPerPixel / 8;
    for (uint32_t y = 0; y < h.height; ++y) {
        const uint32_t dst_y = top_to_bottom ? y : (h.height - 1 - y);
        for (uint32_t x = 0; x < h.width; ++x) {
            const size_t si = (size_t(y) * h.width + x) * channels;
            const size_t di = (size_t(dst_y) * h.width + x) * 4;
            const uint8_t b = src[si + 0], g = src[si + 1], r = src[si + 2];
            const uint8_t a = channels == 4 ? src[si + 3] : 255;
            rgba[di + 0] = Rml::byte((uint32_t(r) * a) / 255);
            rgba[di + 1] = Rml::byte((uint32_t(g) * a) / 255);
            rgba[di + 2] = Rml::byte((uint32_t(b) * a) / 255);
            rgba[di + 3] = Rml::byte(a);
        }
    }
    texture_dimensions = {int(h.width), int(h.height)};
    return GenerateTexture(rgba, texture_dimensions);
}

void RenderInterface_WebGPU::ReleaseTexture(Rml::TextureHandle texture_handle)
{
    if (!texture_handle || texture_handle == TextureEnableWithoutBinding || texture_handle == TexturePostprocess) return;
    auto* t = reinterpret_cast<Gfx::WebGPUTexture*>(texture_handle);
    DestroyTextureResource(*t);
    delete t;
}

void RenderInterface_WebGPU::EnableScissorRegion(bool enable)
{
    m_scissor_enabled = enable;
    ApplyScissor();
}

void RenderInterface_WebGPU::SetScissorRegion(Rml::Rectanglei region)
{
    m_scissor_state = region;
    ApplyScissor();
}

void RenderInterface_WebGPU::EnableClipMask(bool enable)
{
    m_stencil_enabled = enable;
    m_stencil_equal = enable && m_clip_mask_depth > 0;
    if (m_render_pass && enable)
        wgpuRenderPassEncoderSetStencilReference(m_render_pass, m_stencil_ref_value);
}

void RenderInterface_WebGPU::RenderToClipMask(Rml::ClipMaskOperation operation,
    Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation)
{
    if (!m_render_pass || !geometry) return;

    switch (operation) {
    case Rml::ClipMaskOperation::Set:
        {
            const auto fb = m_layer_stack.GetTopLayer();
            EndCurrentRenderPass();
            BeginLayerRenderPass(fb, false, true);
            m_clip_mask_depth = 1;
            m_stencil_ref_value = 1;
            m_stencil_equal = true;
            UsePipeline(WebGPUPipelineId::Color_Stencil_Set);
        }
        break;
    case Rml::ClipMaskOperation::SetInverse:
        // Clear stencil to zero, fill it with one, then replace the supplied
        // geometry with zero. The remaining pixels are therefore the inverse.
        {
            const auto fb = m_layer_stack.GetTopLayer();
            EndCurrentRenderPass();
            BeginLayerRenderPass(fb, false, true);
            m_clip_mask_depth = 1;
            m_stencil_ref_value = 1;
            m_stencil_equal = true;
            UsePipeline(WebGPUPipelineId::Color_Stencil_Set);
            const Rml::Matrix4f saved_transform = m_transform;
            m_transform = Rml::Matrix4f::Identity();
            SubmitTransform({});
            if (m_fullscreen_quad_geometry) {
                auto* q = reinterpret_cast<Gfx::WebGPUGeometry*>(m_fullscreen_quad_geometry);
                wgpuRenderPassEncoderSetStencilReference(m_render_pass, 1);
                wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, q->vertex_buffer, 0, WGPU_WHOLE_SIZE);
                wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, q->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
                wgpuRenderPassEncoderDrawIndexed(m_render_pass, q->num_indices, 1, 0, 0, 0);
            }
            m_transform = saved_transform;
            m_stencil_ref_value = 0;
            wgpuRenderPassEncoderSetStencilReference(m_render_pass, 0);
            UsePipeline(WebGPUPipelineId::Color_Stencil_SetInverse);
        }
        break;
    case Rml::ClipMaskOperation::Intersect:
        if (m_clip_mask_depth < 254) ++m_clip_mask_depth;
        ++m_stencil_ref_value;
        m_stencil_equal = true;
        UsePipeline(WebGPUPipelineId::Color_Stencil_Intersect);
        break;
    }

    if (m_render_pass) {
        wgpuRenderPassEncoderSetStencilReference(m_render_pass, m_stencil_ref_value);
        auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(geometry);
        wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, g->vertex_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, g->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(m_render_pass, g->num_indices, 1, 0, 0, 0);
    }
}

void RenderInterface_WebGPU::SetTransform(const Rml::Matrix4f* transform)
{
    m_transform = transform ? (m_projection * (*transform)) : m_projection;
}

Rml::LayerHandle RenderInterface_WebGPU::PushLayer()
{
    EndCurrentRenderPass();
    const Rml::LayerHandle h = m_layer_stack.PushLayer();
    BeginLayerRenderPass(m_layer_stack.GetLayer(h), true, true);
    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_stencil_equal = false;
    return h;
}

void RenderInterface_WebGPU::BlitFramebuffer(const Gfx::WebGPUFramebuffer& source, const Gfx::WebGPUFramebuffer& destination)
{
    if (!m_command_encoder || !source.color_texture || !destination.color_texture) return;
    EndCurrentRenderPass();
    WGPUExtent3D extent = {std::min(source.width, destination.width), std::min(source.height, destination.height), 1};
    WGPUTexelCopyTextureInfo s = {};
    s.texture = source.color_texture;
    WGPUTexelCopyTextureInfo d = {};
    d.texture = destination.color_texture;
    wgpuCommandEncoderCopyTextureToTexture(m_command_encoder, &s, &d, &extent);
}

void RenderInterface_WebGPU::RenderBlur(float sigma, const Gfx::WebGPUFramebuffer& source_destination,
    const Gfx::WebGPUFramebuffer& temp, Rml::Rectanglei window)
{
    if (!window.Valid() || sigma <= 0.0f || !m_command_encoder)
        return;

    const auto& tertiary = m_layer_stack.GetPostprocessTertiary();

    // Pass 0: primary -> secondary.
    BlitFramebuffer(source_destination, temp);

    BlurUniformData u = {};
    const float s = std::max(sigma, 0.0f);
    float weights[4] = {};
    float norm = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (std::abs(s) < 0.1f)
            weights[i] = (i == 0) ? 1.0f : 0.0f;
        else
            weights[i] = std::exp(-float(i * i) / (2.0f * s * s));
        norm += (i == 0 ? 1.0f : 2.0f) * weights[i];
    }
    if (norm <= 0.0f)
        weights[0] = norm = 1.0f;
    for (float& w : weights) w /= norm;
    std::memcpy(u.weights, weights, sizeof(weights));

    const int left = std::clamp(window.Left(), 0, int(source_destination.width));
    const int top = std::clamp(window.Top(), 0, int(source_destination.height));
    const int right = std::clamp(window.Right(), left, int(source_destination.width));
    const int bottom = std::clamp(window.Bottom(), top, int(source_destination.height));
    u.tex_coord_min[0] = (float(left) + 0.5f) / float(source_destination.width);
    u.tex_coord_min[1] = (float(top) + 0.5f) / float(source_destination.height);
    u.tex_coord_max[0] = (float(right) - 0.5f) / float(source_destination.width);
    u.tex_coord_max[1] = (float(bottom) - 0.5f) / float(source_destination.height);

    // Pass 1: vertical blur, secondary -> tertiary.
    EndCurrentRenderPass();
    BeginLayerRenderPass(tertiary, true, false);
    UsePipeline(WebGPUPipelineId::Blur);
    u.texel_offset[0] = 0.0f;
    u.texel_offset[1] = 1.0f / float(std::max(temp.height, 1u));
    wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &u, sizeof(u));
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, temp.texture_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);
    DrawFullscreenQuad();
    EndCurrentRenderPass();

    // Pass 2: horizontal blur, tertiary -> secondary.
    BeginLayerRenderPass(temp, true, false);
    UsePipeline(WebGPUPipelineId::Blur);
    u.texel_offset[0] = 1.0f / float(std::max(tertiary.width, 1u));
    u.texel_offset[1] = 0.0f;
    wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &u, sizeof(u));
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, tertiary.texture_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);
    DrawFullscreenQuad();
    EndCurrentRenderPass();

    // Pass 3: copy the filtered result back to primary.
    BlitFramebuffer(temp, source_destination);
    BeginLayerRenderPass(source_destination, false, false);
}

void RenderInterface_WebGPU::RenderFilters(Rml::Span<const Rml::CompiledFilterHandle> filter_handles)
{
    if (filter_handles.empty())
        return;

    // RmlUi defines the scissor as a final-composition constraint. Filters operate
    // on the complete source layer, while blur/drop-shadow use the saved window
    // bounds to prevent sampling outside the intended region.
    const bool saved_scissor_enabled = m_scissor_enabled;
    m_scissor_enabled = false;
    ApplyScissor();

    for (auto fh : filter_handles) {
        auto* f = reinterpret_cast<CompiledFilter*>(fh);
        if (!f) continue;
        const auto& src = m_layer_stack.GetPostprocessPrimary();
        auto& dst = const_cast<Gfx::WebGPUFramebuffer&>(m_layer_stack.GetPostprocessSecondary());

        switch (f->type) {
        case FilterType::Passthrough:
        case FilterType::ColorMatrix: {
            EndCurrentRenderPass();
            BeginLayerRenderPass(dst, true, false);
            UsePipeline(f->type == FilterType::ColorMatrix ? WebGPUPipelineId::ColorMatrix : WebGPUPipelineId::Passthrough_NoBlend);
            if (f->type == FilterType::ColorMatrix) {
                ColorMatrixUniformData u = {};
                MatrixToArray(f->color_matrix, u.matrix);
                wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &u, sizeof(u));
                wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);
            }
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, src.texture_bind_group, 0, nullptr);
            DrawFullscreenQuad();
            EndCurrentRenderPass();
            m_layer_stack.SwapPostprocessPrimarySecondary();
        } break;

        case FilterType::Blur: {
            RenderBlur(f->sigma, src, dst, m_scissor_state.Valid() ? m_scissor_state : Rml::Rectanglei(Rml::Vector2i(0,0), Rml::Vector2i(int(src.width),int(src.height))));
        } break;

        case FilterType::DropShadow: {
            EndCurrentRenderPass();
            BeginLayerRenderPass(dst, true, false);
            UsePipeline(WebGPUPipelineId::DropShadow);

            DropShadowUniformData u = {};
            const auto c = Colorf(f->color);
            std::memcpy(u.color, &c[0], sizeof(float) * 4);
            const Rml::Rectanglei window = m_scissor_state.Valid() ? m_scissor_state
                : Rml::Rectanglei(Rml::Vector2i(0, 0), Rml::Vector2i(int(src.width), int(src.height)));
            u.tex_coord_min[0] = (float(window.Left()) + 0.5f) / float(src.width);
            u.tex_coord_min[1] = (float(window.Top()) + 0.5f) / float(src.height);
            u.tex_coord_max[0] = (float(window.Right()) - 0.5f) / float(src.width);
            u.tex_coord_max[1] = (float(window.Bottom()) - 0.5f) / float(src.height);
            u.offset[0] = -f->offset.x / float(std::max(m_viewport_width, 1));
            u.offset[1] = -f->offset.y / float(std::max(m_viewport_height, 1));
            u.sigma = f->sigma;
            wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &u, sizeof(u));
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, src.texture_bind_group, 0, nullptr);
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);
            DrawFullscreenQuad();
            EndCurrentRenderPass();

            if (f->sigma >= 0.5f) {
                RenderBlur(f->sigma, dst, const_cast<Gfx::WebGPUFramebuffer&>(m_layer_stack.GetPostprocessTertiary()), window);
            } else {
                BeginLayerRenderPass(dst, false, false);
            }

            // Draw the original source over the shadow. RenderBlur leaves dst active.
            UsePipeline(WebGPUPipelineId::Passthrough);
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, src.texture_bind_group, 0, nullptr);
            DrawFullscreenQuad();
            EndCurrentRenderPass();
            m_layer_stack.SwapPostprocessPrimarySecondary();
        } break;

        case FilterType::MaskImage: {
            if (!f->mask_texture) break;
            EndCurrentRenderPass();
            BeginLayerRenderPass(dst, true, false);
            UsePipeline(WebGPUPipelineId::BlendMask);
            WGPUBindGroupEntry e[3] = {};
            e[0].binding = 0; e[0].sampler = m_sampler_linear;
            e[1].binding = 1; e[1].textureView = src.color_view;
            e[2].binding = 2; e[2].textureView = f->mask_texture->view;
            WGPUBindGroupDescriptor d = {};
            d.layout = m_bgl_blend_mask; d.entryCount = 3; d.entries = e;
            WGPUBindGroup bg = wgpuDeviceCreateBindGroup(m_device, &d);
            wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, bg, 0, nullptr);
            DrawFullscreenQuad();
            if (bg) wgpuBindGroupRelease(bg);
            EndCurrentRenderPass();
            m_layer_stack.SwapPostprocessPrimarySecondary();
        } break;

        default: break;
        }
    }

    m_scissor_enabled = saved_scissor_enabled;
    ApplyScissor();
}

void RenderInterface_WebGPU::CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination,
    Rml::BlendMode blend_mode, Rml::Span<const Rml::CompiledFilterHandle> filters)
{
    EndCurrentRenderPass();
    BlitFramebuffer(m_layer_stack.GetLayer(source), m_layer_stack.GetPostprocessPrimary());
    RenderFilters(filters);

    BeginLayerRenderPass(m_layer_stack.GetLayer(destination), false, false);
    UsePipeline(blend_mode == Rml::BlendMode::Replace ? WebGPUPipelineId::Passthrough_NoBlend : WebGPUPipelineId::Passthrough);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, m_layer_stack.GetPostprocessPrimary().texture_bind_group, 0, nullptr);
    DrawFullscreenQuad();
    EndCurrentRenderPass();

    BeginLayerRenderPass(m_layer_stack.GetTopLayer(), false, false);
}

void RenderInterface_WebGPU::PopLayer()
{
    EndCurrentRenderPass();
    m_layer_stack.PopLayer();
    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_stencil_equal = false;
    BeginLayerRenderPass(m_layer_stack.GetTopLayer(), false, true);
}

Rml::TextureHandle RenderInterface_WebGPU::SaveLayerAsTexture()
{
    if (!m_scissor_state.Valid()) return {};
    const int w = m_scissor_state.Width();
    const int h = m_scissor_state.Height();
    if (w <= 0 || h <= 0) return {};

    auto* t = CreateTextureResource(m_device, m_bgl_texture, m_sampler_linear, m_render_format, uint32_t(w), uint32_t(h));
    if (!t) return {};

    EndCurrentRenderPass();
    const auto& src = m_layer_stack.GetTopLayer();
    auto& staging = const_cast<Gfx::WebGPUFramebuffer&>(m_layer_stack.GetBlendMask());

    ColorMatrixUniformData identity = {};
    identity.matrix[0] = identity.matrix[5] = identity.matrix[10] = identity.matrix[15] = 1.0f;
    wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &identity, sizeof(identity));

    WGPURenderPassColorAttachment ca = {};
    ca.view = staging.color_view;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = {0, 0, 0, 0};
    WGPURenderPassDescriptor rp = {};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    m_render_pass = wgpuCommandEncoderBeginRenderPass(m_command_encoder, &rp);
    m_owns_render_pass = true;
    m_active_pipeline = WebGPUPipelineId::Count;
    UsePipeline(WebGPUPipelineId::ColorMatrix);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, src.texture_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);

    Rml::CompiledGeometryHandle geometry = CreateFullscreenGeometry(this, false);
    if (geometry) {
        auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(geometry);
        wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, g->vertex_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, g->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(m_render_pass, g->num_indices, 1, 0, 0, 0);
        ReleaseGeometry(geometry);
    }
    EndCurrentRenderPass();

    WGPUTexelCopyTextureInfo s = {};
    s.texture = staging.color_texture;
    s.origin = {uint32_t(std::max(m_scissor_state.Left(), 0)), uint32_t(std::max(m_scissor_state.Top(), 0)), 0};
    WGPUTexelCopyTextureInfo d = {};
    d.texture = t->texture;
    WGPUExtent3D extent = {uint32_t(w), uint32_t(h), 1};
    wgpuCommandEncoderCopyTextureToTexture(m_command_encoder, &s, &d, &extent);

    BeginLayerRenderPass(src, false, false);
    return reinterpret_cast<Rml::TextureHandle>(t);
}

Rml::CompiledFilterHandle RenderInterface_WebGPU::SaveLayerAsMaskImage()
{
    auto* f = new CompiledFilter();
    f->type = FilterType::MaskImage;

    const auto& src = m_layer_stack.GetTopLayer();
    f->mask_texture = CreateTextureResource(m_device, m_bgl_texture, m_sampler_linear, m_render_format, src.width, src.height);
    if (!f->mask_texture)
        return reinterpret_cast<Rml::CompiledFilterHandle>(f);

    EndCurrentRenderPass();

    // Render the layer into the mask texture with the vertical convention expected
    // by RmlUi textures. ColorMatrix is used as an identity, no-depth, no-blend copy.
    ColorMatrixUniformData identity = {};
    identity.matrix[0] = identity.matrix[5] = identity.matrix[10] = identity.matrix[15] = 1.0f;
    wgpuQueueWriteBuffer(m_queue, m_filter_uniform_buffer, 0, &identity, sizeof(identity));

    WGPURenderPassColorAttachment ca = {};
    ca.view = f->mask_texture->view;
    ca.loadOp = WGPULoadOp_Clear;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = {0, 0, 0, 0};
    WGPURenderPassDescriptor rp = {};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    m_render_pass = wgpuCommandEncoderBeginRenderPass(m_command_encoder, &rp);
    m_owns_render_pass = true;
    m_active_pipeline = WebGPUPipelineId::Count;
    UsePipeline(WebGPUPipelineId::ColorMatrix);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 1, src.texture_bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass, 2, m_filter_bind_group, 0, nullptr);

    Rml::CompiledGeometryHandle geometry = CreateFullscreenGeometry(this, true);
    if (geometry) {
        auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(geometry);
        wgpuRenderPassEncoderSetVertexBuffer(m_render_pass, 0, g->vertex_buffer, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderSetIndexBuffer(m_render_pass, g->index_buffer, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
        wgpuRenderPassEncoderDrawIndexed(m_render_pass, g->num_indices, 1, 0, 0, 0);
        ReleaseGeometry(geometry);
    }
    EndCurrentRenderPass();
    BeginLayerRenderPass(src, false, false);
    return reinterpret_cast<Rml::CompiledFilterHandle>(f);
}

Rml::CompiledFilterHandle RenderInterface_WebGPU::CompileFilter(const Rml::String& name, const Rml::Dictionary& parameters)
{
    auto* f = new CompiledFilter();
    if (name == "opacity") {
        const float v = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.0f, 1.0f);
        f->type = FilterType::ColorMatrix;
        f->color_matrix = Rml::Matrix4f::Diag(v, v, v, v);
    } else if (name == "blur") {
        f->type = FilterType::Blur;
        f->sigma = std::max(Rml::Get(parameters, "sigma", 1.0f), 0.0f);
    } else if (name == "drop-shadow") {
        f->type = FilterType::DropShadow;
        f->sigma = std::max(Rml::Get(parameters, "sigma", 0.0f), 0.0f);
        f->offset = Rml::Get(parameters, "offset", Rml::Vector2f(0.0f));
        f->color = Rml::Get(parameters, "color", Rml::Colourb()).ToPremultiplied();
    } else if (name == "brightness") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Get(parameters, "value", 1.0f);
        f->color_matrix = Rml::Matrix4f::Diag(v, v, v, 1.0f);
    } else if (name == "contrast") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Get(parameters, "value", 1.0f);
        const float b = 0.5f - 0.5f*v;
        f->color_matrix = Rml::Matrix4f::Diag(v, v, v, 1.0f);
        f->color_matrix.SetColumn(3, Rml::Vector4f(b,b,b,1.0f));
    } else if (name == "invert") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Math::Clamp(Rml::Get(parameters, "value", 1.0f), 0.0f, 1.0f);
        const float inv = 1.0f - 2.0f*v;
        f->color_matrix = Rml::Matrix4f::Diag(inv, inv, inv, 1.0f);
        f->color_matrix.SetColumn(3, Rml::Vector4f(v,v,v,1.0f));
    } else if (name == "grayscale") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Get(parameters, "value", 1.0f), r = 0.2126f*v, g = 0.7152f*v, b = 0.0722f*v, rv = 1-v;
        f->color_matrix = Rml::Matrix4f::FromRows({r+rv,g,b,0},{r,g+rv,b,0},{r,g,b+rv,0},{0,0,0,1});
    } else if (name == "sepia") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Get(parameters, "value", 1.0f), rv = 1-v;
        f->color_matrix = Rml::Matrix4f::FromRows({0.393f*v+rv,0.769f*v,0.189f*v,0},{0.349f*v,0.686f*v+rv,0.168f*v,0},{0.272f*v,0.534f*v,0.131f*v+rv,0},{0,0,0,1});
    } else if (name == "hue-rotate") {
        f->type = FilterType::ColorMatrix;
        const float a = Rml::Get(parameters, "value", 0.0f), s = Rml::Math::Sin(a), c = Rml::Math::Cos(a);
        f->color_matrix = Rml::Matrix4f::FromRows(
            {0.213f+0.787f*c-0.213f*s, 0.715f-0.715f*c-0.715f*s, 0.072f-0.072f*c+0.928f*s,0},
            {0.213f-0.213f*c+0.143f*s, 0.715f+0.285f*c+0.140f*s, 0.072f-0.072f*c-0.283f*s,0},
            {0.213f-0.213f*c-0.787f*s, 0.715f-0.715f*c+0.715f*s, 0.072f+0.928f*c+0.072f*s,0},
            {0,0,0,1});
    } else if (name == "saturate") {
        f->type = FilterType::ColorMatrix;
        const float v = Rml::Get(parameters, "value", 1.0f);
        f->color_matrix = Rml::Matrix4f::FromRows(
            {0.213f+0.787f*v,0.715f-0.715f*v,0.072f-0.072f*v,0},
            {0.213f-0.213f*v,0.715f+0.285f*v,0.072f-0.072f*v,0},
            {0.213f-0.213f*v,0.715f-0.715f*v,0.072f+0.928f*v,0},
            {0,0,0,1});
    } else {
        delete f;
        ReportUnsupported(name.c_str());
        return {};
    }
    return reinterpret_cast<Rml::CompiledFilterHandle>(f);
}

void RenderInterface_WebGPU::ReleaseFilter(Rml::CompiledFilterHandle filter)
{
    auto* f = reinterpret_cast<CompiledFilter*>(filter);
    if (!f) return;
    if (f->mask_texture) {
        DestroyTextureResource(*f->mask_texture);
        delete f->mask_texture;
    }
    delete f;
}

Rml::CompiledShaderHandle RenderInterface_WebGPU::CompileShader(const Rml::String& name, const Rml::Dictionary& parameters)
{
    auto* s = new CompiledShader();
    auto ApplyStops = [&](const Rml::Dictionary& p) {
        auto it = p.find("color_stop_list");
        if (it == p.end() || it->second.GetType() != Rml::Variant::COLORSTOPLIST) return;
        const auto& list = it->second.GetReference<Rml::ColorStopList>();
        const int n = std::min<int>(int(list.size()), int(kMaxGradientStops));
        s->stop_positions.resize(n);
        s->stop_colors.resize(n);
        for (int i=0;i<n;++i) { s->stop_positions[i]=list[i].position.number; s->stop_colors[i]=Colorf(list[i].color); }
    };
    if (name == "linear-gradient") {
        const bool repeating = Rml::Get(parameters, "repeating", false);
        s->type = ShaderType::Gradient;
        s->func = repeating ? 1 : 0;
        s->p = Rml::Get(parameters, "p0", Rml::Vector2f(0));
        s->v = Rml::Get(parameters, "p1", Rml::Vector2f(0)) - s->p;
        ApplyStops(parameters);
    } else if (name == "radial-gradient") {
        const bool repeating = Rml::Get(parameters, "repeating", false);
        s->type = ShaderType::Gradient;
        s->func = repeating ? 3 : 2;
        s->p = Rml::Get(parameters, "center", Rml::Vector2f(0));
        const auto r = Rml::Get(parameters, "radius", Rml::Vector2f(1));
        s->v = {1.0f / std::max(r.x, 0.0001f), 1.0f / std::max(r.y, 0.0001f)};
        ApplyStops(parameters);
    } else if (name == "conic-gradient") {
        const bool repeating = Rml::Get(parameters, "repeating", false);
        s->type = ShaderType::Gradient;
        s->func = repeating ? 5 : 4;
        s->p = Rml::Get(parameters, "center", Rml::Vector2f(0));
        const float a = Rml::Get(parameters, "angle", 0.0f);
        s->v = {Rml::Math::Cos(a), Rml::Math::Sin(a)};
        ApplyStops(parameters);
    } else if (name == "shader" && Rml::Get(parameters,"value",Rml::String()) == "creation") {
        s->type=ShaderType::Creation; s->dimensions=Rml::Get(parameters,"dimensions",Rml::Vector2f(0));
    } else {
        delete s; ReportUnsupported(name.c_str()); return {};
    }
    return reinterpret_cast<Rml::CompiledShaderHandle>(s);
}

void RenderInterface_WebGPU::RenderShader(Rml::CompiledShaderHandle shader_handle,
    Rml::CompiledGeometryHandle geometry_handle, Rml::Vector2f translation, Rml::TextureHandle)
{
    auto* s = reinterpret_cast<CompiledShader*>(shader_handle);
    auto* g = reinterpret_cast<Gfx::WebGPUGeometry*>(geometry_handle);
    if (!s || !g || !m_render_pass) return;

    if (s->type == ShaderType::Gradient) {
        GradientUniformData u = {};
        u.func=s->func; u.num_stops=int32_t(std::min(s->stop_positions.size(), size_t(kMaxGradientStops)));
        u.p[0]=s->p.x; u.p[1]=s->p.y; u.v[0]=s->v.x; u.v[1]=s->v.y;
        for (size_t i=0;i<s->stop_positions.size() && i<kMaxGradientStops;++i) u.stop_positions[i]=s->stop_positions[i];
        for (size_t i=0;i<s->stop_colors.size() && i<kMaxGradientStops;++i) for(int c=0;c<4;++c) u.stop_colors[i*4+c]=s->stop_colors[i][c];
        wgpuQueueWriteBuffer(m_queue,m_filter_uniform_buffer,0,&u,sizeof(u));
        UsePipeline(WebGPUPipelineId::Gradient);
    } else if (s->type == ShaderType::Creation) {
        CreationUniformData u = {};
        u.dimensions[0]=s->dimensions.x; u.dimensions[1]=s->dimensions.y;
        u.value=float(Rml::GetSystemInterface()->GetElapsedTime());
        wgpuQueueWriteBuffer(m_queue,m_filter_uniform_buffer,0,&u,sizeof(u));
        UsePipeline(WebGPUPipelineId::Creation);
    } else return;

    SubmitTransform(translation);
    wgpuRenderPassEncoderSetBindGroup(m_render_pass,2,m_filter_bind_group,0,nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(m_render_pass,0,g->vertex_buffer,0,WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(m_render_pass,g->index_buffer,WGPUIndexFormat_Uint32,0,WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndexed(m_render_pass,g->num_indices,1,0,0,0);
}

void RenderInterface_WebGPU::ReleaseShader(Rml::CompiledShaderHandle shader_handle)
{
    delete reinterpret_cast<CompiledShader*>(shader_handle);
}
