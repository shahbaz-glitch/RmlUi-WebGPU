#include "RmlUi_Renderer_WebGPU.h"
#include "RmlUi_Renderer_shader.h"

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/MeshUtilities.h>
#include <RmlUi/Core/SystemInterface.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace {

using namespace RmlWebGPU;

constexpr uint32_t kMaxGradientStops = 16;
constexpr uint32_t kStencilMask = 0xffu;

static uint32_t Align256(uint32_t value)
{
    return (value + 255u) & ~255u;
}

static bool ValidDimensions(int width, int height)
{
    return width > 0 && height > 0;
}

static WGPUTextureView CreateView(WGPUTexture texture, WGPUTextureFormat format)
{
    if (!texture)
        return nullptr;

    WGPUTextureViewDescriptor desc = {};
    desc.format = format;
    desc.dimension = WGPUTextureViewDimension_2D;
    desc.baseMipLevel = 0;
    desc.mipLevelCount = 1;
    desc.baseArrayLayer = 0;
    desc.arrayLayerCount = 1;

    return wgpuTextureCreateView(texture, &desc);
}

static WGPUTexture CreateColorTexture(
    WGPUDevice device,
    uint32_t width,
    uint32_t height,
    WGPUTextureFormat format)
{
    if (!device || width == 0 || height == 0 || format == WGPUTextureFormat_Undefined)
        return nullptr;

    WGPUTextureDescriptor desc = {};
    desc.dimension = WGPUTextureDimension_2D;
    desc.size = {width, height, 1};
    desc.format = format;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.usage =
        WGPUTextureUsage_RenderAttachment |
        WGPUTextureUsage_TextureBinding |
        WGPUTextureUsage_CopySrc |
        WGPUTextureUsage_CopyDst;

    return wgpuDeviceCreateTexture(device, &desc);
}

static WGPUTexture CreateDepthStencilTexture(
    WGPUDevice device,
    uint32_t width,
    uint32_t height,
    WGPUTextureFormat format)
{
    if (!device || width == 0 || height == 0 || format == WGPUTextureFormat_Undefined)
        return nullptr;

    WGPUTextureDescriptor desc = {};
    desc.dimension = WGPUTextureDimension_2D;
    desc.size = {width, height, 1};
    desc.format = format;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.usage = WGPUTextureUsage_RenderAttachment;

    return wgpuDeviceCreateTexture(device, &desc);
}

static WGPUTextureView CreateDepthView(
    WGPUTexture texture,
    WGPUTextureFormat format)
{
    if (!texture)
        return nullptr;

    WGPUTextureViewDescriptor desc = {};
    desc.format = format;
    desc.dimension = WGPUTextureViewDimension_2D;
    desc.baseMipLevel = 0;
    desc.mipLevelCount = 1;
    desc.baseArrayLayer = 0;
    desc.arrayLayerCount = 1;

    return wgpuTextureCreateView(texture, &desc);
}

static WGPUBuffer CreateUniformBuffer(WGPUDevice device, uint64_t size)
{
    if (!device || size == 0)
        return nullptr;

    WGPUBufferDescriptor desc = {};
    desc.size = size;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    return wgpuDeviceCreateBuffer(device, &desc);
}

static WGPUBlendState PremultipliedBlend()
{
    WGPUBlendState blend = {};
    blend.color.srcFactor = WGPUBlendFactor_One;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.color.operation = WGPUBlendOperation_Add;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.operation = WGPUBlendOperation_Add;
    return blend;
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

static WGPUStencilFaceState StencilFace(
    WGPUCompareFunction compare,
    WGPUStencilOperation pass)
{
    WGPUStencilFaceState state = {};
    state.compare = compare;
    state.failOp = WGPUStencilOperation_Keep;
    state.depthFailOp = WGPUStencilOperation_Keep;
    state.passOp = pass;
    return state;
}

static WGPUColorTargetState ColorTarget(
    WGPUTextureFormat format,
    const WGPUBlendState* blend)
{
    WGPUColorTargetState target = {};
    target.format = format;
    target.blend = blend;
    target.writeMask = WGPUColorWriteMask_All;
    return target;
}

static Rml::Colourf Colorf(Rml::ColourbPremultiplied colour)
{
    Rml::Colourf result;
    for (int i = 0; i < 4; ++i)
        result[i] = float(colour[i]) / 255.0f;
    return result;
}

static void MatrixToArray(const Rml::Matrix4f& matrix, float* out)
{
    std::memcpy(out, matrix.data(), sizeof(float) * 16);
}

static void ReportUnsupported(const char* what)
{
    Rml::Log::Message(
        Rml::Log::LT_WARNING,
        "WebGPU backend: unsupported %s.",
        what);
}

static Rml::CompiledGeometryHandle CreateFullscreenGeometry(
    RenderInterface_WebGPU* renderer,
    bool flip_y)
{
    if (!renderer)
        return {};

    Rml::Mesh mesh;
    Rml::MeshUtilities::GenerateQuad(
        mesh,
        Rml::Vector2f(-1.0f),
        Rml::Vector2f(2.0f),
        {});

    if (flip_y)
    {
        for (Rml::Vertex& vertex : mesh.vertices)
            vertex.tex_coord.y = 1.0f - vertex.tex_coord.y;
    }

    return renderer->CompileGeometry(mesh.vertices, mesh.indices);
}

} // namespace

namespace {

enum class FilterType
{
    Invalid = 0,
    Passthrough,
    Blur,
    DropShadow,
    ColorMatrix,
    MaskImage
};

struct CompiledFilter
{
    FilterType type = FilterType::Invalid;

    float blend_factor = 1.0f;
    float sigma = 0.0f;

    Rml::Vector2f offset;
    Rml::ColourbPremultiplied color;

    Rml::Matrix4f color_matrix;

    Gfx::WebGPUTexture* mask_texture = nullptr;
};

enum class ShaderType
{
    Invalid = 0,
    Gradient,
    Creation
};

struct CompiledShader
{
    ShaderType type = ShaderType::Invalid;

    int func = 0;

    Rml::Vector2f p;
    Rml::Vector2f v;

    std::vector<float> stop_positions;
    std::vector<Rml::Colourf> stop_colors;

    Rml::Vector2f dimensions;
};

static void DestroyTextureResource(Gfx::WebGPUTexture& texture)
{
    if (texture.bind_group)
        wgpuBindGroupRelease(texture.bind_group);

    if (texture.view)
        wgpuTextureViewRelease(texture.view);

    if (texture.texture)
        wgpuTextureRelease(texture.texture);

    texture = {};
}

static Gfx::WebGPUTexture* CreateTextureResource(
    WGPUDevice device,
    WGPUBindGroupLayout texture_layout,
    WGPUSampler sampler,
    WGPUTextureFormat format,
    uint32_t width,
    uint32_t height)
{
    if (!device || !texture_layout || !sampler ||
        format == WGPUTextureFormat_Undefined ||
        !ValidDimensions(int(width), int(height)))
        return nullptr;

    auto* texture = new Gfx::WebGPUTexture();

    texture->texture =
        CreateColorTexture(device, width, height, format);

    if (!texture->texture)
    {
        delete texture;
        return nullptr;
    }

    texture->view =
        CreateView(texture->texture, format);

    if (!texture->view)
    {
        DestroyTextureResource(*texture);
        delete texture;
        return nullptr;
    }

    WGPUBindGroupEntry entries[2] = {};

    entries[0].binding = 0;
    entries[0].sampler = sampler;

    entries[1].binding = 1;
    entries[1].textureView = texture->view;

    WGPUBindGroupDescriptor desc = {};
    desc.layout = texture_layout;
    desc.entryCount = 2;
    desc.entries = entries;

    texture->bind_group =
        wgpuDeviceCreateBindGroup(device, &desc);

    if (!texture->bind_group)
    {
        DestroyTextureResource(*texture);
        delete texture;
        return nullptr;
    }

    texture->width = width;
    texture->height = height;

    return texture;
}

} // namespace

// ============================================================================
// RenderLayerStack
// ============================================================================

void RenderInterface_WebGPU::RenderLayerStack::Initialize(
    WGPUDevice device,
    int width,
    int height,
    WGPUTextureFormat color_format,
    WGPUTextureFormat depth_stencil_format,
    WGPUBindGroupLayout texture_bgl,
    WGPUSampler sampler)
{
    Shutdown();

    if (!device || !texture_bgl || !sampler)
        return;

    m_device = device;
    m_color_format = color_format;
    m_depth_stencil_format = depth_stencil_format;
    m_texture_bgl = texture_bgl;
    m_sampler = sampler;

    m_width = std::max(width, 1);
    m_height = std::max(height, 1);

    m_shared_depth_stencil =
        CreateDepthStencilTexture(
            m_device,
            uint32_t(m_width),
            uint32_t(m_height),
            m_depth_stencil_format);

    m_shared_depth_stencil_view =
        CreateDepthView(
            m_shared_depth_stencil,
            m_depth_stencil_format);

    m_fb_layers.resize(1);

    CreateFramebuffer(
        m_fb_layers[0],
        m_width,
        m_height,
        true);

    m_layers_size = 1;

    for (auto& framebuffer : m_fb_postprocess)
    {
        CreateFramebuffer(
            framebuffer,
            m_width,
            m_height,
            false);
    }
}

void RenderInterface_WebGPU::RenderLayerStack::Shutdown()
{
    DestroyFramebuffers();

    m_fb_layers.clear();
    m_layers_size = 0;

    m_device = nullptr;
    m_texture_bgl = nullptr;
    m_sampler = nullptr;

    m_color_format = WGPUTextureFormat_Undefined;
    m_depth_stencil_format = WGPUTextureFormat_Undefined;

    m_width = 0;
    m_height = 0;
}

void RenderInterface_WebGPU::RenderLayerStack::CreateFramebuffer(
    Gfx::WebGPUFramebuffer& framebuffer,
    int width,
    int height,
    bool has_depth_stencil)
{
    DestroyFramebuffer(framebuffer);

    if (!m_device ||
        m_color_format == WGPUTextureFormat_Undefined ||
        width <= 0 ||
        height <= 0)
        return;

    framebuffer.width = uint32_t(width);
    framebuffer.height = uint32_t(height);

    framebuffer.color_texture =
        CreateColorTexture(
            m_device,
            framebuffer.width,
            framebuffer.height,
            m_color_format);

    if (!framebuffer.color_texture)
    {
        framebuffer = {};
        return;
    }

    framebuffer.color_view =
        CreateView(
            framebuffer.color_texture,
            m_color_format);

    if (!framebuffer.color_view)
    {
        DestroyFramebuffer(framebuffer);
        return;
    }

    if (has_depth_stencil)
        framebuffer.depth_stencil_view = m_shared_depth_stencil_view;

    WGPUBindGroupEntry entries[2] = {};

    entries[0].binding = 0;
    entries[0].sampler = m_sampler;

    entries[1].binding = 1;
    entries[1].textureView = framebuffer.color_view;

    WGPUBindGroupDescriptor desc = {};
    desc.layout = m_texture_bgl;
    desc.entryCount = 2;
    desc.entries = entries;

    framebuffer.texture_bind_group =
        wgpuDeviceCreateBindGroup(
            m_device,
            &desc);

    if (!framebuffer.texture_bind_group)
    {
        DestroyFramebuffer(framebuffer);
        return;
    }
}

void RenderInterface_WebGPU::RenderLayerStack::DestroyFramebuffer(
    Gfx::WebGPUFramebuffer& framebuffer)
{
    if (framebuffer.texture_bind_group)
        wgpuBindGroupRelease(framebuffer.texture_bind_group);

    if (framebuffer.depth_stencil_view &&
        framebuffer.depth_stencil_view != m_shared_depth_stencil_view)
    {
        wgpuTextureViewRelease(framebuffer.depth_stencil_view);
    }

    if (framebuffer.color_view)
        wgpuTextureViewRelease(framebuffer.color_view);

    if (framebuffer.color_texture)
        wgpuTextureRelease(framebuffer.color_texture);

    framebuffer = {};
}

void RenderInterface_WebGPU::RenderLayerStack::DestroyFramebuffers()
{
    for (auto& framebuffer : m_fb_layers)
        DestroyFramebuffer(framebuffer);

    for (auto& framebuffer : m_fb_postprocess)
        DestroyFramebuffer(framebuffer);

    if (m_shared_depth_stencil_view)
        wgpuTextureViewRelease(m_shared_depth_stencil_view);

    if (m_shared_depth_stencil)
        wgpuTextureRelease(m_shared_depth_stencil);

    m_shared_depth_stencil_view = nullptr;
    m_shared_depth_stencil = nullptr;
}

void RenderInterface_WebGPU::RenderLayerStack::Resize(
    int width,
    int height)
{
    if (!m_device || width <= 0 || height <= 0)
        return;

    const int active_layers =
        std::max(m_layers_size, 1);

    m_width = width;
    m_height = height;

    m_layers_size = 0;

    for (auto& framebuffer : m_fb_layers)
        DestroyFramebuffer(framebuffer);

    for (auto& framebuffer : m_fb_postprocess)
        DestroyFramebuffer(framebuffer);

    if (m_shared_depth_stencil_view)
        wgpuTextureViewRelease(m_shared_depth_stencil_view);

    if (m_shared_depth_stencil)
        wgpuTextureRelease(m_shared_depth_stencil);

    m_shared_depth_stencil_view = nullptr;
    m_shared_depth_stencil = nullptr;

    m_shared_depth_stencil =
        CreateDepthStencilTexture(
            m_device,
            uint32_t(width),
            uint32_t(height),
            m_depth_stencil_format);

    m_shared_depth_stencil_view =
        CreateDepthView(
            m_shared_depth_stencil,
            m_depth_stencil_format);

    m_fb_layers.clear();
    m_fb_layers.resize(active_layers);

    for (auto& framebuffer : m_fb_layers)
    {
        CreateFramebuffer(
            framebuffer,
            width,
            height,
            true);
    }

    m_layers_size = active_layers;

    for (auto& framebuffer : m_fb_postprocess)
    {
        CreateFramebuffer(
            framebuffer,
            width,
            height,
            false);
    }
}

Rml::LayerHandle
RenderInterface_WebGPU::RenderLayerStack::PushLayer()
{
    const int index = m_layers_size;

    if (index >= int(m_fb_layers.size()))
    {
        m_fb_layers.emplace_back();

        CreateFramebuffer(
            m_fb_layers.back(),
            m_width,
            m_height,
            true);
    }

    m_layers_size = index + 1;

    return Rml::LayerHandle(index);
}

void RenderInterface_WebGPU::RenderLayerStack::PopLayer()
{
    if (m_layers_size > 1)
        --m_layers_size;
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetLayer(
    Rml::LayerHandle layer) const
{
    const size_t index = size_t(layer);

    RMLUI_ASSERT(index < size_t(m_layers_size));

    return m_fb_layers[index];
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetTopLayer() const
{
    return GetLayer(GetTopLayerHandle());
}

Rml::LayerHandle
RenderInterface_WebGPU::RenderLayerStack::GetTopLayerHandle() const
{
    RMLUI_ASSERT(m_layers_size > 0);

    return Rml::LayerHandle(m_layers_size - 1);
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetPostprocessPrimary()
{
    return m_fb_postprocess[0];
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetPostprocessSecondary()
{
    return m_fb_postprocess[1];
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetPostprocessTertiary()
{
    return m_fb_postprocess[2];
}

const Gfx::WebGPUFramebuffer&
RenderInterface_WebGPU::RenderLayerStack::GetBlendMask()
{
    return m_fb_postprocess[3];
}

void RenderInterface_WebGPU::RenderLayerStack::
SwapPostprocessPrimarySecondary()
{
    std::swap(
        m_fb_postprocess[0],
        m_fb_postprocess[1]);
}

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

bool RenderInterface_WebGPU::Initialize(
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTextureFormat render_target_format,
    WGPUTextureFormat depth_stencil_format)
{
    if (!device ||
        !queue ||
        render_target_format == WGPUTextureFormat_Undefined ||
        depth_stencil_format == WGPUTextureFormat_Undefined)
    {
        return false;
    }

    Shutdown();

    m_device = device;
    m_queue = queue;
    m_render_format = render_target_format;
    m_depth_stencil_format = depth_stencil_format;

    CreateBindGroupLayouts();
    CreateDefaultSamplers();

    if (!m_bgl_transform ||
        !m_bgl_texture ||
        !m_bgl_blend_mask ||
        !m_bgl_filter_uniform ||
        !m_sampler_linear)
    {
        Shutdown();
        return false;
    }

    m_transform_uniform_buffer =
        CreateUniformBuffer(
            m_device,
            kTransformUniformBufferSize);

    m_filter_uniform_buffer =
        CreateUniformBuffer(
            m_device,
            kFilterUniformBufferSize);

    if (!m_transform_uniform_buffer ||
        !m_filter_uniform_buffer)
    {
        Shutdown();
        return false;
    }

    WGPUBindGroupEntry transform_entry = {};
    transform_entry.binding = 0;
    transform_entry.buffer = m_transform_uniform_buffer;
    transform_entry.size = sizeof(TransformUniform);

    WGPUBindGroupDescriptor transform_desc = {};
    transform_desc.layout = m_bgl_transform;
    transform_desc.entryCount = 1;
    transform_desc.entries = &transform_entry;

    m_transform_bind_group =
        wgpuDeviceCreateBindGroup(
            m_device,
            &transform_desc);

    WGPUBindGroupEntry filter_entry = {};
    filter_entry.binding = 0;
    filter_entry.buffer = m_filter_uniform_buffer;
    filter_entry.size = kFilterUniformBufferSize;

    WGPUBindGroupDescriptor filter_desc = {};
    filter_desc.layout = m_bgl_filter_uniform;
    filter_desc.entryCount = 1;
    filter_desc.entries = &filter_entry;

    m_filter_bind_group =
        wgpuDeviceCreateBindGroup(
            m_device,
            &filter_desc);

    if (!m_transform_bind_group ||
        !m_filter_bind_group)
    {
        Shutdown();
        return false;
    }

    CreatePipelines();

    bool all_pipelines_valid = true;

    for (size_t i = 0;
         i < static_cast<size_t>(WebGPUPipelineId::Count);
         ++i)
    {
        if (!m_pipelines[i])
            all_pipelines_valid = false;
    }

    if (!all_pipelines_valid)
    {
        Shutdown();
        return false;
    }

    const int layer_width =
        std::max(m_viewport_width, 1);

    const int layer_height =
        std::max(m_viewport_height, 1);

    m_layer_stack.Initialize(
        m_device,
        layer_width,
        layer_height,
        m_render_format,
        m_depth_stencil_format,
        m_bgl_texture,
        m_sampler_linear);

    const auto& base =
        m_layer_stack.GetLayer(Rml::LayerHandle(0));

    if (!base.color_view ||
        !base.texture_bind_group)
    {
        Shutdown();
        return false;
    }

    m_fullscreen_quad_geometry =
        CreateFullscreenGeometry(
            this,
            true);

    if (!m_fullscreen_quad_geometry)
    {
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

    if (m_fullscreen_quad_geometry)
    {
        ReleaseGeometry(
            m_fullscreen_quad_geometry);

        m_fullscreen_quad_geometry = {};
    }

    for (auto& pipeline : m_pipelines)
    {
        if (pipeline)
            wgpuRenderPipelineRelease(pipeline);

        pipeline = nullptr;
    }

    if (m_pipeline_layout_filter)
        wgpuPipelineLayoutRelease(m_pipeline_layout_filter);

    if (m_pipeline_layout_blend_mask)
        wgpuPipelineLayoutRelease(m_pipeline_layout_blend_mask);

    if (m_pipeline_layout_transform_texture)
        wgpuPipelineLayoutRelease(m_pipeline_layout_transform_texture);

    if (m_pipeline_layout_transform)
        wgpuPipelineLayoutRelease(m_pipeline_layout_transform);

    m_pipeline_layout_filter = nullptr;
    m_pipeline_layout_blend_mask = nullptr;
    m_pipeline_layout_transform_texture = nullptr;
    m_pipeline_layout_transform = nullptr;

    if (m_transform_bind_group)
        wgpuBindGroupRelease(m_transform_bind_group);

    if (m_filter_bind_group)
        wgpuBindGroupRelease(m_filter_bind_group);

    if (m_transform_uniform_buffer)
        wgpuBufferRelease(m_transform_uniform_buffer);

    if (m_filter_uniform_buffer)
        wgpuBufferRelease(m_filter_uniform_buffer);

    m_transform_bind_group = nullptr;
    m_filter_bind_group = nullptr;
    m_transform_uniform_buffer = nullptr;
    m_filter_uniform_buffer = nullptr;

    if (m_sampler_clamp)
        wgpuSamplerRelease(m_sampler_clamp);

    if (m_sampler_linear)
        wgpuSamplerRelease(m_sampler_linear);

    m_sampler_clamp = nullptr;
    m_sampler_linear = nullptr;

    if (m_bgl_filter_uniform)
        wgpuBindGroupLayoutRelease(m_bgl_filter_uniform);

    if (m_bgl_blend_mask)
        wgpuBindGroupLayoutRelease(m_bgl_blend_mask);

    if (m_bgl_texture)
        wgpuBindGroupLayoutRelease(m_bgl_texture);

    if (m_bgl_transform)
        wgpuBindGroupLayoutRelease(m_bgl_transform);

    m_bgl_filter_uniform = nullptr;
    m_bgl_blend_mask = nullptr;
    m_bgl_texture = nullptr;
    m_bgl_transform = nullptr;

    if (m_command_encoder)
        wgpuCommandEncoderRelease(m_command_encoder);

    m_command_encoder = nullptr;
    m_render_pass = nullptr;
    m_target_view = nullptr;
    m_owns_render_pass = false;

    m_active_pipeline = WebGPUPipelineId::Count;

    m_initialized = false;
    m_device = nullptr;
    m_queue = nullptr;

    m_render_format = WGPUTextureFormat_Undefined;
    m_depth_stencil_format = WGPUTextureFormat_Depth24PlusStencil8;
}

// ============================================================================
// Viewport / frame
// ============================================================================

void RenderInterface_WebGPU::SetViewport(int width, int height)
{
    m_viewport_width = std::max(width, 1);
    m_viewport_height = std::max(height, 1);

    // RmlUi uses a top-left origin. The reversed Y interval matches the GL3
    // renderer's orthographic projection while targeting WebGPU coordinates.
    m_projection = Rml::Matrix4f::ProjectOrtho(
        0.0f,
        float(m_viewport_width),
        float(m_viewport_height),
        0.0f,
        -10000.0f,
        10000.0f);

    if (m_initialized)
    {
        EndCurrentRenderPass();

        m_layer_stack.Resize(
            m_viewport_width,
            m_viewport_height);
    }
}

void RenderInterface_WebGPU::BeginFrame(WGPUTextureView target_view)
{
    if (!m_initialized ||
        !target_view ||
        !m_command_encoder == false)
    {
        // Intentionally handled below; the command encoder is recreated per frame.
    }

    if (!m_initialized ||
        !target_view ||
        m_viewport_width <= 0 ||
        m_viewport_height <= 0)
    {
        return;
    }

    EndCurrentRenderPass();

    if (m_command_encoder)
    {
        wgpuCommandEncoderRelease(m_command_encoder);
        m_command_encoder = nullptr;
    }

    WGPUCommandEncoderDescriptor encoder_desc = {};
    m_command_encoder =
        wgpuDeviceCreateCommandEncoder(
            m_device,
            &encoder_desc);

    if (!m_command_encoder)
        return;

    m_target_view = target_view;

    m_current_uniform_offset = 0;

    m_scissor_enabled = false;
    m_scissor_state = Rml::Rectanglei::MakeInvalid();

    m_stencil_enabled = false;
    m_stencil_equal = false;
    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_current_clip_operation = -1;

    m_active_pipeline = WebGPUPipelineId::Count;

    SetTransform(nullptr);

    BeginLayerRenderPass(
        m_layer_stack.GetLayer(Rml::LayerHandle(0)),
        true,
        true);
}

void RenderInterface_WebGPU::EndFrame()
{
    if (!m_command_encoder ||
        !m_target_view)
    {
        return;
    }

    EndCurrentRenderPass();

    const auto& base =
        m_layer_stack.GetLayer(Rml::LayerHandle(0));

    const auto& primary =
        m_layer_stack.GetPostprocessPrimary();

    if (!base.color_texture ||
        !primary.color_texture)
    {
        wgpuCommandEncoderRelease(m_command_encoder);
        m_command_encoder = nullptr;
        m_target_view = nullptr;
        return;
    }

    WGPUExtent3D extent = {
        std::min(base.width, primary.width),
        std::min(base.height, primary.height),
        1
    };

    WGPUTexelCopyTextureInfo source = {};
    source.texture = base.color_texture;

    WGPUTexelCopyTextureInfo destination = {};
    destination.texture = primary.color_texture;

    wgpuCommandEncoderCopyTextureToTexture(
        m_command_encoder,
        &source,
        &destination,
        &extent);

    WGPUColorAttachment color_attachment = {};
    color_attachment.view = m_target_view;
    color_attachment.loadOp = WGPULoadOp_Load;
    color_attachment.storeOp = WGPUStoreOp_Store;
    color_attachment.clearValue = {0, 0, 0, 0};

    WGPURenderPassDescriptor render_pass_desc = {};
    render_pass_desc.colorAttachmentCount = 1;
    render_pass_desc.colorAttachments = &color_attachment;
    render_pass_desc.depthStencilAttachment = nullptr;

    m_render_pass =
        wgpuCommandEncoderBeginRenderPass(
            m_command_encoder,
            &render_pass_desc);

    if (!m_render_pass)
    {
        wgpuCommandEncoderRelease(m_command_encoder);
        m_command_encoder = nullptr;
        m_target_view = nullptr;
        return;
    }

    m_owns_render_pass = true;
    m_active_pipeline = WebGPUPipelineId::Count;

    UsePipeline(WebGPUPipelineId::Passthrough);

    if (primary.texture_bind_group)
    {
        wgpuRenderPassEncoderSetBindGroup(
            m_render_pass,
            1,
            primary.texture_bind_group,
            0,
            nullptr);
    }

    DrawFullscreenQuad();

    EndCurrentRenderPass();

    WGPUCommandBufferDescriptor command_buffer_desc = {};

    WGPUCommandBuffer command_buffer =
        wgpuCommandEncoderFinish(
            m_command_encoder,
            &command_buffer_desc);

    if (command_buffer)
    {
        wgpuQueueSubmit(
            m_queue,
            1,
            &command_buffer);

        wgpuCommandBufferRelease(command_buffer);
    }

    wgpuCommandEncoderRelease(m_command_encoder);

    m_command_encoder = nullptr;
    m_target_view = nullptr;
    m_active_pipeline = WebGPUPipelineId::Count;
}

void RenderInterface_WebGPU::SetActiveRenderPass(
    WGPURenderPassEncoder pass_encoder)
{
    EndCurrentRenderPass();

    m_render_pass = pass_encoder;
    m_owns_render_pass = false;
    m_active_pipeline = WebGPUPipelineId::Count;

    ApplyScissor();
}

// ============================================================================
// Bind groups / samplers / pipelines
// ============================================================================

void RenderInterface_WebGPU::CreateBindGroupLayouts()
{
    WGPUBindGroupLayoutEntry transform = {};
    transform.binding = 0;
    transform.visibility = WGPUShaderStage_Vertex;
    transform.buffer.type = WGPUBufferBindingType_Uniform;
    transform.buffer.hasDynamicOffset = true;
    transform.buffer.minBindingSize = sizeof(TransformUniform);

    WGPUBindGroupLayoutDescriptor transform_desc = {};
    transform_desc.entryCount = 1;
    transform_desc.entries = &transform;

    m_bgl_transform =
        wgpuDeviceCreateBindGroupLayout(
            m_device,
            &transform_desc);

    WGPUBindGroupLayoutEntry texture_entries[2] = {};

    texture_entries[0].binding = 0;
    texture_entries[0].visibility = WGPUShaderStage_Fragment;
    texture_entries[0].sampler.type = WGPUSamplerBindingType_Filtering;

    texture_entries[1].binding = 1;
    texture_entries[1].visibility = WGPUShaderStage_Fragment;
    texture_entries[1].texture.sampleType = WGPUTextureSampleType_Float;
    texture_entries[1].texture.viewDimension =
        WGPUTextureViewDimension_2D;

    WGPUBindGroupLayoutDescriptor texture_desc = {};
    texture_desc.entryCount = 2;
    texture_desc.entries = texture_entries;

    m_bgl_texture =
        wgpuDeviceCreateBindGroupLayout(
            m_device,
            &texture_desc);

    WGPUBindGroupLayoutEntry mask_entries[3] = {};

    mask_entries[0] = texture_entries[0];
    mask_entries[1] = texture_entries[1];

    mask_entries[2] = texture_entries[1];
    mask_entries[2].binding = 2;

    WGPUBindGroupLayoutDescriptor mask_desc = {};
    mask_desc.entryCount = 3;
    mask_desc.entries = mask_entries;

    m_bgl_blend_mask =
        wgpuDeviceCreateBindGroupLayout(
            m_device,
            &mask_desc);

    WGPUBindGroupLayoutEntry filter = {};
    filter.binding = 0;
    filter.visibility = WGPUShaderStage_Fragment;
    filter.buffer.type = WGPUBufferBindingType_Uniform;
    filter.buffer.minBindingSize = 16;

    WGPUBindGroupLayoutDescriptor filter_desc = {};
    filter_desc.entryCount = 1;
    filter_desc.entries = &filter;

    m_bgl_filter_uniform =
        wgpuDeviceCreateBindGroupLayout(
            m_device,
            &filter_desc);
}

void RenderInterface_WebGPU::CreateDefaultSamplers()
{
    WGPUSamplerDescriptor desc = {};

    desc.addressModeU = WGPUAddressMode_ClampToEdge;
    desc.addressModeV = WGPUAddressMode_ClampToEdge;
    desc.addressModeW = WGPUAddressMode_ClampToEdge;

    desc.magFilter = WGPUFilterMode_Linear;
    desc.minFilter = WGPUFilterMode_Linear;
    desc.mipmapFilter = WGPUMipmapFilterMode_Linear;

    desc.maxAnisotropy = 1;

    m_sampler_linear =
        wgpuDeviceCreateSampler(
            m_device,
            &desc);

    if (m_sampler_linear)
    {
        m_sampler_clamp = m_sampler_linear;
        wgpuSamplerAddRef(m_sampler_clamp);
    }
}

void RenderInterface_WebGPU::CreatePipelines()
{
    WGPUShaderModule vs_main =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_vert_main);

    WGPUShaderModule vs_pass =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_vert_passthrough);

    WGPUShaderModule fs_color =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_color);

    WGPUShaderModule fs_texture =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_texture);

    WGPUShaderModule fs_gradient =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_gradient);

    WGPUShaderModule fs_creation =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_creation);

    WGPUShaderModule fs_pass =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_passthrough);

    WGPUShaderModule fs_matrix =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_color_matrix);

    WGPUShaderModule fs_mask =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_blend_mask);

    WGPUShaderModule fs_blur =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_blur);

    WGPUShaderModule fs_shadow =
        RmlWebGPU::CreateShaderModule(
            m_device,
            RmlWebGPU::s_shader_frag_drop_shadow);

    WGPUBindGroupLayout filter_groups[] = {
        m_bgl_transform,
        m_bgl_texture,
        m_bgl_filter_uniform
    };

    WGPUPipelineLayoutDescriptor filter_layout_desc = {};
    filter_layout_desc.bindGroupLayoutCount = 3;
    filter_layout_desc.bindGroupLayouts = filter_groups;

    m_pipeline_layout_filter =
        wgpuDeviceCreatePipelineLayout(
            m_device,
            &filter_layout_desc);

    WGPUBindGroupLayout texture_groups[] = {
        m_bgl_transform,
        m_bgl_texture
    };

    WGPUPipelineLayoutDescriptor texture_layout_desc = {};
    texture_layout_desc.bindGroupLayoutCount = 2;
    texture_layout_desc.bindGroupLayouts = texture_groups;

    m_pipeline_layout_transform_texture =
        wgpuDeviceCreatePipelineLayout(
            m_device,
            &texture_layout_desc);

    WGPUBindGroupLayout mask_groups[] = {
        m_bgl_transform,
        m_bgl_blend_mask
    };

    WGPUPipelineLayoutDescriptor mask_layout_desc = {};
    mask_layout_desc.bindGroupLayoutCount = 2;
    mask_layout_desc.bindGroupLayouts = mask_groups;

    m_pipeline_layout_blend_mask =
        wgpuDeviceCreatePipelineLayout(
            m_device,
            &mask_layout_desc);

    WGPUBindGroupLayout transform_groups[] = {
        m_bgl_transform
    };

    WGPUPipelineLayoutDescriptor transform_layout_desc = {};
    transform_layout_desc.bindGroupLayoutCount = 1;
    transform_layout_desc.bindGroupLayouts = transform_groups;

    m_pipeline_layout_transform =
        wgpuDeviceCreatePipelineLayout(
            m_device,
            &transform_layout_desc);

    const WGPUBlendState blend =
        PremultipliedBlend();

    const WGPUVertexBufferLayout vertex_layout =
        VertexLayout();

    auto MakePipeline =
        [&](WebGPUPipelineId id,
            WGPUPipelineLayout layout,
            WGPUShaderModule vertex_shader,
            WGPUShaderModule fragment_shader,
            WGPUCompareFunction stencil_compare,
            WGPUStencilOperation stencil_pass,
            bool enable_blend,
            bool write_color,
            bool use_depth_stencil)
    {
        if (!layout || !vertex_shader || !fragment_shader)
            return;

        WGPUStencilFaceState stencil =
            StencilFace(
                stencil_compare,
                stencil_pass);

        WGPUDepthStencilState depth_stencil = {};
        depth_stencil.format = m_depth_stencil_format;
        depth_stencil.depthWriteEnabled = false;
        depth_stencil.depthCompare = WGPUCompareFunction_Always;

        depth_stencil.stencilFront = stencil;
        depth_stencil.stencilBack = stencil;

        depth_stencil.stencilReadMask = kStencilMask;
        depth_stencil.stencilWriteMask = kStencilMask;

        WGPUColorTargetState color_target =
            ColorTarget(
                m_render_format,
                enable_blend ? &blend : nullptr);

        if (!write_color)
            color_target.writeMask =
                WGPUColorWriteMask_None;

        WGPUFragmentState fragment = {};
        fragment.module = fragment_shader;
        fragment.entryPoint = {"main", 4};
        fragment.targetCount = 1;
        fragment.targets = &color_target;

        WGPUVertexState vertex = {};
        vertex.module = vertex_shader;
        vertex.entryPoint = {"main", 4};
        vertex.bufferCount = 1;
        vertex.buffers = &vertex_layout;

        WGPUPrimitiveState primitive = {};
        primitive.topology =
            WGPUPrimitiveTopology_TriangleList;
        primitive.frontFace = WGPUFrontFace_CCW;
        primitive.cullMode = WGPUCullMode_None;

        WGPUMultisampleState multisample = {};
        multisample.count = 1;
        multisample.mask = ~0u;

        WGPURenderPipelineDescriptor pipeline = {};
        pipeline.layout = layout;
        pipeline.vertex = vertex;
        pipeline.primitive = primitive;
        pipeline.depthStencil =
            use_depth_stencil ? &depth_stencil : nullptr;
        pipeline.multisample = multisample;
        pipeline.fragment = &fragment;

        m_pipelines[size_t(id)] =
            wgpuDeviceCreateRenderPipeline(
                m_device,
                &pipeline);
    };

    // ------------------------------------------------------------------------
    // Normal color rendering
    // ------------------------------------------------------------------------

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_Disabled,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_Always,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_Equal,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    // ------------------------------------------------------------------------
    // Clip-mask writing
    // ------------------------------------------------------------------------

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_Set,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Replace,
        false,
        false,
        true);

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_SetInverse,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Replace,
        false,
        false,
        true);

    MakePipeline(
        WebGPUPipelineId::Color_Stencil_Intersect,
        m_pipeline_layout_transform,
        vs_main,
        fs_color,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_IncrementClamp,
        false,
        false,
        true);

    // ------------------------------------------------------------------------
    // Normal textured rendering
    // ------------------------------------------------------------------------

    MakePipeline(
        WebGPUPipelineId::Texture_Stencil_Disabled,
        m_pipeline_layout_transform_texture,
        vs_main,
        fs_texture,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Texture_Stencil_Always,
        m_pipeline_layout_transform_texture,
        vs_main,
        fs_texture,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Texture_Stencil_Equal,
        m_pipeline_layout_transform_texture,
        vs_main,
        fs_texture,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    // ------------------------------------------------------------------------
    // RmlUi shaders
    // ------------------------------------------------------------------------

    MakePipeline(
        WebGPUPipelineId::Gradient,
        m_pipeline_layout_filter,
        vs_main,
        fs_gradient,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Creation,
        m_pipeline_layout_filter,
        vs_main,
        fs_creation,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    // ------------------------------------------------------------------------
    // Fullscreen passes
    // ------------------------------------------------------------------------

    MakePipeline(
        WebGPUPipelineId::Passthrough,
        m_pipeline_layout_transform_texture,
        vs_pass,
        fs_pass,
        WGPUCompareFunction_Equal,
        WGPUStencilOperation_Keep,
        true,
        true,
        true);

    MakePipeline(
        WebGPUPipelineId::Passthrough_NoBlend,
        m_pipeline_layout_transform_texture,
        vs_pass,
        fs_pass,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        false,
        true,
        false);

    MakePipeline(
        WebGPUPipelineId::ColorMatrix,
        m_pipeline_layout_filter,
        vs_pass,
        fs_matrix,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        false,
        true,
        false);

    MakePipeline(
        WebGPUPipelineId::BlendMask,
        m_pipeline_layout_blend_mask,
        vs_pass,
        fs_mask,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        false,
        true,
        false);

    MakePipeline(
        WebGPUPipelineId::Blur,
        m_pipeline_layout_filter,
        vs_pass,
        fs_blur,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        false,
        true,
        false);

    MakePipeline(
        WebGPUPipelineId::DropShadow,
        m_pipeline_layout_filter,
        vs_pass,
        fs_shadow,
        WGPUCompareFunction_Always,
        WGPUStencilOperation_Keep,
        false,
        true,
        false);

    if (vs_main)
        wgpuShaderModuleRelease(vs_main);

    if (vs_pass)
        wgpuShaderModuleRelease(vs_pass);

    if (fs_color)
        wgpuShaderModuleRelease(fs_color);

    if (fs_texture)
        wgpuShaderModuleRelease(fs_texture);

    if (fs_gradient)
        wgpuShaderModuleRelease(fs_gradient);

    if (fs_creation)
        wgpuShaderModuleRelease(fs_creation);

    if (fs_pass)
        wgpuShaderModuleRelease(fs_pass);

    if (fs_matrix)
        wgpuShaderModuleRelease(fs_matrix);

    if (fs_mask)
        wgpuShaderModuleRelease(fs_mask);

    if (fs_blur)
        wgpuShaderModuleRelease(fs_blur);

    if (fs_shadow)
        wgpuShaderModuleRelease(fs_shadow);
}

// ============================================================================
// Render-pass state
// ============================================================================

void RenderInterface_WebGPU::BeginLayerRenderPass(
    const Gfx::WebGPUFramebuffer& framebuffer,
    bool clear_color,
    bool clear_stencil)
{
    if (!m_command_encoder ||
        !framebuffer.color_view)
    {
        return;
    }

    EndCurrentRenderPass();

    WGPUColorAttachment color_attachment = {};
    color_attachment.view = framebuffer.color_view;
    color_attachment.loadOp =
        clear_color
            ? WGPULoadOp_Clear
            : WGPULoadOp_Load;
    color_attachment.storeOp =
        WGPUStoreOp_Store;
    color_attachment.clearValue = {
        0, 0, 0, 0
    };

    WGPURenderPassDepthStencilAttachment depth_stencil = {};

    if (framebuffer.depth_stencil_view)
    {
        depth_stencil.view =
            framebuffer.depth_stencil_view;

        depth_stencil.depthLoadOp =
            WGPULoadOp_Clear;
        depth_stencil.depthStoreOp =
            WGPUStoreOp_Store;
        depth_stencil.depthClearValue = 1.0f;

        depth_stencil.stencilLoadOp =
            clear_stencil
                ? WGPULoadOp_Clear
                : WGPULoadOp_Load;

        depth_stencil.stencilStoreOp =
            WGPUStoreOp_Store;

        depth_stencil.stencilClearValue = 0;
    }

    WGPURenderPassDescriptor render_pass_desc = {};
    render_pass_desc.colorAttachmentCount = 1;
    render_pass_desc.colorAttachments = &color_attachment;
    render_pass_desc.depthStencilAttachment =
        framebuffer.depth_stencil_view
            ? &depth_stencil
            : nullptr;

    m_render_pass =
        wgpuCommandEncoderBeginRenderPass(
            m_command_encoder,
            &render_pass_desc);

    if (!m_render_pass)
        return;

    m_owns_render_pass = true;
    m_active_pipeline = WebGPUPipelineId::Count;

    ApplyScissor();
}

void RenderInterface_WebGPU::EndCurrentRenderPass()
{
    if (m_render_pass &&
        m_owns_render_pass)
    {
        wgpuRenderPassEncoderEnd(
            m_render_pass);

        wgpuRenderPassEncoderRelease(
            m_render_pass);
    }

    m_render_pass = nullptr;
    m_owns_render_pass = false;
    m_active_pipeline = WebGPUPipelineId::Count;
}

void RenderInterface_WebGPU::ApplyScissor()
{
    if (!m_render_pass)
        return;

    int x = 0;
    int y = 0;
    int width = m_viewport_width;
    int height = m_viewport_height;

    if (m_scissor_enabled &&
        m_scissor_state.Valid())
    {
        const int left =
            std::max(
                m_scissor_state.Left(),
                0);

        const int top =
            std::max(
                m_scissor_state.Top(),
                0);

        const int right =
            std::min(
                m_scissor_state.Right(),
                m_viewport_width);

        const int bottom =
            std::min(
                m_scissor_state.Bottom(),
                m_viewport_height);

        x = std::min(
            left,
            m_viewport_width);

        y = std::min(
            top,
            m_viewport_height);

        width =
            std::max(
                right - x,
                0);

        height =
            std::max(
                bottom - y,
                0);
    }

    wgpuRenderPassEncoderSetScissorRect(
        m_render_pass,
        uint32_t(x),
        uint32_t(y),
        uint32_t(width),
        uint32_t(height));
}

void RenderInterface_WebGPU::UsePipeline(
    WebGPUPipelineId pipeline_id)
{
    if (!m_render_pass ||
        pipeline_id >= WebGPUPipelineId::Count)
    {
        return;
    }

    if (m_active_pipeline == pipeline_id)
    {
        if (m_stencil_enabled)
        {
            wgpuRenderPassEncoderSetStencilReference(
                m_render_pass,
                m_stencil_ref_value);
        }

        return;
    }

    WGPURenderPipeline pipeline =
        m_pipelines[size_t(pipeline_id)];

    if (!pipeline)
        return;

    wgpuRenderPassEncoderSetPipeline(
        m_render_pass,
        pipeline);

    m_active_pipeline = pipeline_id;

    if (m_stencil_enabled)
    {
        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            m_stencil_ref_value);
    }
}

void RenderInterface_WebGPU::SubmitTransform(
    const Rml::Vector2f& translation)
{
    if (!m_render_pass ||
        !m_transform_bind_group ||
        !m_queue ||
        !m_transform_uniform_buffer)
    {
        return;
    }

    if (m_current_uniform_offset +
            kUniformAlignment >
        kTransformUniformBufferSize)
    {
        Rml::Log::Message(
            Rml::Log::LT_ERROR,
            "WebGPU transform uniform ring exhausted for one frame.");

        return;
    }

    TransformUniform uniform = {};

    MatrixToArray(
        m_transform,
        uniform.transform);

    uniform.translate[0] =
        translation.x;

    uniform.translate[1] =
        translation.y;

    wgpuQueueWriteBuffer(
        m_queue,
        m_transform_uniform_buffer,
        m_current_uniform_offset,
        &uniform,
        sizeof(uniform));

    const uint32_t dynamic_offset =
        m_current_uniform_offset;

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        0,
        m_transform_bind_group,
        1,
        &dynamic_offset);

    m_current_uniform_offset +=
        kUniformAlignment;
}

void RenderInterface_WebGPU::DrawFullscreenQuad()
{
    if (!m_render_pass ||
        !m_fullscreen_quad_geometry)
    {
        return;
    }

    auto* geometry =
        reinterpret_cast<Gfx::WebGPUGeometry*>(
            m_fullscreen_quad_geometry);

    if (!geometry ||
        !geometry->vertex_buffer ||
        !geometry->index_buffer)
    {
        return;
    }

    wgpuRenderPassEncoderSetVertexBuffer(
        m_render_pass,
        0,
        geometry->vertex_buffer,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderSetIndexBuffer(
        m_render_pass,
        geometry->index_buffer,
        WGPUIndexFormat_Uint32,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderDrawIndexed(
        m_render_pass,
        geometry->num_indices,
        1,
        0,
        0,
        0);
}

// ============================================================================
// Geometry
// ============================================================================

Rml::CompiledGeometryHandle
RenderInterface_WebGPU::CompileGeometry(
    Rml::Span<const Rml::Vertex> vertices,
    Rml::Span<const int> indices)
{
    if (!m_device ||
        !m_queue ||
        vertices.empty() ||
        indices.empty())
    {
        return {};
    }

    auto* geometry =
        new Gfx::WebGPUGeometry();

    geometry->num_indices =
        uint32_t(indices.size());

    WGPUBufferDescriptor vertex_desc = {};
    vertex_desc.size =
        uint64_t(vertices.size_bytes());

    vertex_desc.usage =
        WGPUBufferUsage_Vertex |
        WGPUBufferUsage_CopyDst;

    geometry->vertex_buffer =
        wgpuDeviceCreateBuffer(
            m_device,
            &vertex_desc);

    WGPUBufferDescriptor index_desc = {};
    index_desc.size =
        uint64_t(indices.size()) *
        sizeof(uint32_t);

    index_desc.usage =
        WGPUBufferUsage_Index |
        WGPUBufferUsage_CopyDst;

    geometry->index_buffer =
        wgpuDeviceCreateBuffer(
            m_device,
            &index_desc);

    if (!geometry->vertex_buffer ||
        !geometry->index_buffer)
    {
        if (geometry->vertex_buffer)
            wgpuBufferRelease(
                geometry->vertex_buffer);

        if (geometry->index_buffer)
            wgpuBufferRelease(
                geometry->index_buffer);

        delete geometry;
        return {};
    }

    wgpuQueueWriteBuffer(
        m_queue,
        geometry->vertex_buffer,
        0,
        vertices.data(),
        vertices.size_bytes());

    wgpuQueueWriteBuffer(
        m_queue,
        geometry->index_buffer,
        0,
        indices.data(),
        uint64_t(indices.size()) *
            sizeof(uint32_t));

    return reinterpret_cast<
        Rml::CompiledGeometryHandle>(
            geometry);
}

void RenderInterface_WebGPU::RenderGeometry(
    Rml::CompiledGeometryHandle handle,
    Rml::Vector2f translation,
    Rml::TextureHandle texture)
{
    if (!m_render_pass ||
        !handle)
    {
        return;
    }

    auto* geometry =
        reinterpret_cast<Gfx::WebGPUGeometry*>(
            handle);

    if (!geometry ||
        !geometry->vertex_buffer ||
        !geometry->index_buffer)
    {
        return;
    }

    if (texture == TexturePostprocess)
    {
        UsePipeline(
            WebGPUPipelineId::Passthrough);

        if (m_layer_stack.GetPostprocessPrimary()
                .texture_bind_group)
        {
            wgpuRenderPassEncoderSetBindGroup(
                m_render_pass,
                1,
                m_layer_stack
                    .GetPostprocessPrimary()
                    .texture_bind_group,
                0,
                nullptr);
        }

        // Passthrough has no transform group in
        // the vertex shader.
    }
    else if (texture != 0 &&
             texture != TextureEnableWithoutBinding)
    {
        WebGPUPipelineId pipeline =
            WebGPUPipelineId::Texture_Stencil_Disabled;

        if (m_stencil_enabled)
        {
            pipeline =
                m_stencil_equal
                    ? WebGPUPipelineId::Texture_Stencil_Equal
                    : WebGPUPipelineId::Texture_Stencil_Always;
        }

        UsePipeline(pipeline);
        SubmitTransform(translation);

        const auto* gpu_texture =
            reinterpret_cast<
                const Gfx::WebGPUTexture*>(
                    texture);

        if (gpu_texture &&
            gpu_texture->bind_group)
        {
            wgpuRenderPassEncoderSetBindGroup(
                m_render_pass,
                1,
                gpu_texture->bind_group,
                0,
                nullptr);
        }
    }
    else
    {
        WebGPUPipelineId pipeline =
            WebGPUPipelineId::Color_Stencil_Disabled;

        if (m_stencil_enabled)
        {
            pipeline =
                m_stencil_equal
                    ? WebGPUPipelineId::Color_Stencil_Equal
                    : WebGPUPipelineId::Color_Stencil_Always;
        }

        UsePipeline(pipeline);
        SubmitTransform(translation);
    }

    wgpuRenderPassEncoderSetVertexBuffer(
        m_render_pass,
        0,
        geometry->vertex_buffer,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderSetIndexBuffer(
        m_render_pass,
        geometry->index_buffer,
        WGPUIndexFormat_Uint32,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderDrawIndexed(
        m_render_pass,
        geometry->num_indices,
        1,
        0,
        0,
        0);
}

void RenderInterface_WebGPU::ReleaseGeometry(
    Rml::CompiledGeometryHandle handle)
{
    auto* geometry =
        reinterpret_cast<Gfx::WebGPUGeometry*>(
            handle);

    if (!geometry)
        return;

    if (geometry->vertex_buffer)
        wgpuBufferRelease(
            geometry->vertex_buffer);

    if (geometry->index_buffer)
        wgpuBufferRelease(
            geometry->index_buffer);

    delete geometry;
}

// ============================================================================
// Textures
// ============================================================================

Rml::TextureHandle
RenderInterface_WebGPU::GenerateTexture(
    Rml::Span<const Rml::byte> source_data,
    Rml::Vector2i source_dimensions)
{
    if (!m_device ||
        !m_queue ||
        !ValidDimensions(
            source_dimensions.x,
            source_dimensions.y))
    {
        return {};
    }

    const size_t required_size =
        size_t(source_dimensions.x) *
        size_t(source_dimensions.y) *
        4u;

    if (!source_data.data() ||
        source_data.size() != required_size)
    {
        return {};
    }

    auto* texture =
        CreateTextureResource(
            m_device,
            m_bgl_texture,
            m_sampler_linear,
            WGPUTextureFormat_RGBA8Unorm,
            uint32_t(source_dimensions.x),
            uint32_t(source_dimensions.y));

    if (!texture)
        return {};

    const uint32_t row_bytes =
        uint32_t(source_dimensions.x) * 4u;

    const uint32_t padded_row_bytes =
        Align256(row_bytes);

    std::vector<uint8_t> upload(
        size_t(padded_row_bytes) *
        size_t(source_dimensions.y),
        0);

    // RmlUi's GenerateTexture input is already prepared
    // RGBA8 premultiplied texture data. Keep its row order.
    for (int y = 0;
         y < source_dimensions.y;
         ++y)
    {
        std::memcpy(
            upload.data() +
                size_t(y) *
                padded_row_bytes,
            source_data.data() +
                size_t(y) *
                row_bytes,
            row_bytes);
    }

    WGPUTexelCopyTextureInfo destination = {};
    destination.texture =
        texture->texture;

    WGPUExtent3D extent = {
        uint32_t(source_dimensions.x),
        uint32_t(source_dimensions.y),
        1
    };

    WGPUTexelCopyBufferLayout layout = {};
    layout.offset = 0;
    layout.bytesPerRow =
        padded_row_bytes;
    layout.rowsPerImage =
        uint32_t(source_dimensions.y);

    wgpuQueueWriteTexture(
        m_queue,
        &destination,
        upload.data(),
        upload.size(),
        &layout,
        &extent);

    return reinterpret_cast<
        Rml::TextureHandle>(
            texture);
}

#pragma pack(push, 1)
struct TGAHeader
{
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

Rml::TextureHandle
RenderInterface_WebGPU::LoadTexture(
    Rml::Vector2i& texture_dimensions,
    const Rml::String& source)
{
    auto* file_interface =
        Rml::GetFileInterface();

    if (!file_interface)
        return {};

    Rml::FileHandle file =
        file_interface->Open(source);

    if (!file)
        return {};

    file_interface->Seek(
        file,
        0,
        SEEK_END);

    const size_t file_size =
        file_interface->Tell(file);

    file_interface->Seek(
        file,
        0,
        SEEK_SET);

    if (file_size < sizeof(TGAHeader))
    {
        Rml::Log::Message(
            Rml::Log::LT_ERROR,
            "WebGPU TGA loader: file is too small for '%s'.",
            source.c_str());

        file_interface->Close(file);
        return {};
    }

    std::vector<Rml::byte> file_data(
        file_size);

    if (file_interface->Read(
            file_data.data(),
            file_size,
            file) != file_size)
    {
        file_interface->Close(file);
        return {};
    }

    file_interface->Close(file);

    TGAHeader header = {};

    std::memcpy(
        &header,
        file_data.data(),
        sizeof(header));

    if (header.dataType != 2 ||
        header.colorMapType != 0 ||
        (header.bitsPerPixel != 24 &&
         header.bitsPerPixel != 32) ||
        header.width == 0 ||
        header.height == 0)
    {
        Rml::Log::Message(
            Rml::Log::LT_ERROR,
            "WebGPU TGA loader: unsupported or invalid TGA '%s'.",
            source.c_str());

        return {};
    }

    const int channels =
        header.bitsPerPixel / 8;

    const size_t pixel_offset =
        sizeof(TGAHeader) +
        size_t(header.idLength);

    const size_t pixel_size =
        size_t(header.width) *
        size_t(header.height) *
        size_t(channels);

    if (pixel_offset > file_size ||
        pixel_size > file_size - pixel_offset)
    {
        Rml::Log::Message(
            Rml::Log::LT_ERROR,
            "WebGPU TGA loader: truncated image '%s'.",
            source.c_str());

        return {};
    }

    const uint8_t* source_pixels =
        reinterpret_cast<const uint8_t*>(
            file_data.data() + pixel_offset);

    const size_t rgba_size =
        size_t(header.width) *
        size_t(header.height) *
        4u;

    std::vector<Rml::byte> rgba(
        rgba_size);

    const bool top_to_bottom =
        (header.imageDescriptor & 0x20) != 0;

    for (uint32_t y = 0;
         y < header.height;
         ++y)
    {
        const uint32_t destination_y =
            top_to_bottom
                ? y
                : (header.height - 1u - y);

        for (uint32_t x = 0;
             x < header.width;
             ++x)
        {
            const size_t source_index =
                (size_t(y) *
                 size_t(header.width) +
                 size_t(x)) *
                size_t(channels);

            const size_t destination_index =
                (size_t(destination_y) *
                 size_t(header.width) +
                 size_t(x)) *
                4u;

            const uint8_t b =
                source_pixels[source_index + 0];

            const uint8_t g =
                source_pixels[source_index + 1];

            const uint8_t r =
                source_pixels[source_index + 2];

            const uint8_t alpha =
                channels == 4
                    ? source_pixels[source_index + 3]
                    : 255;

            rgba[destination_index + 0] =
                Rml::byte(
                    (uint32_t(r) *
                     uint32_t(alpha)) /
                    255u);

            rgba[destination_index + 1] =
                Rml::byte(
                    (uint32_t(g) *
                     uint32_t(alpha)) /
                    255u);

            rgba[destination_index + 2] =
                Rml::byte(
                    (uint32_t(b) *
                     uint32_t(alpha)) /
                    255u);

            rgba[destination_index + 3] =
                Rml::byte(alpha);
        }
    }

    texture_dimensions = {
        int(header.width),
        int(header.height)
    };

    return GenerateTexture(
        rgba,
        texture_dimensions);
}

void RenderInterface_WebGPU::ReleaseTexture(
    Rml::TextureHandle texture_handle)
{
    if (!texture_handle ||
        texture_handle == TextureEnableWithoutBinding ||
        texture_handle == TexturePostprocess)
    {
        return;
    }

    auto* texture =
        reinterpret_cast<Gfx::WebGPUTexture*>(
            texture_handle);

    if (!texture)
        return;

    DestroyTextureResource(*texture);
    delete texture;
}

// ============================================================================
// Scissor
// ============================================================================

void RenderInterface_WebGPU::EnableScissorRegion(
    bool enable)
{
    m_scissor_enabled = enable;

    ApplyScissor();
}

void RenderInterface_WebGPU::SetScissorRegion(
    Rml::Rectanglei region)
{
    m_scissor_state = region;

    ApplyScissor();
}

// ============================================================================
// Clip mask
// ============================================================================

void RenderInterface_WebGPU::EnableClipMask(
    bool enable)
{
    m_stencil_enabled = enable;

    m_stencil_equal =
        enable &&
        (m_clip_mask_depth > 0);

    if (m_render_pass)
    {
        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            m_stencil_ref_value);
    }
}

void RenderInterface_WebGPU::RenderToClipMask(
    Rml::ClipMaskOperation operation,
    Rml::CompiledGeometryHandle geometry,
    Rml::Vector2f translation)
{
    if (!m_render_pass ||
        !geometry)
    {
        return;
    }

    auto* gpu_geometry =
        reinterpret_cast<Gfx::WebGPUGeometry*>(
            geometry);

    if (!gpu_geometry ||
        !gpu_geometry->vertex_buffer ||
        !gpu_geometry->index_buffer)
    {
        return;
    }

    switch (operation)
    {
    case Rml::ClipMaskOperation::Set:
    {
        const auto& framebuffer =
            m_layer_stack.GetTopLayer();

        EndCurrentRenderPass();

        BeginLayerRenderPass(
            framebuffer,
            false,
            true);

        m_clip_mask_depth = 1;
        m_stencil_ref_value = 1;
        m_stencil_equal = true;

        UsePipeline(
            WebGPUPipelineId::Color_Stencil_Set);

        SubmitTransform(translation);

        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            m_stencil_ref_value);

        wgpuRenderPassEncoderSetVertexBuffer(
            m_render_pass,
            0,
            gpu_geometry->vertex_buffer,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderSetIndexBuffer(
            m_render_pass,
            gpu_geometry->index_buffer,
            WGPUIndexFormat_Uint32,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderDrawIndexed(
            m_render_pass,
            gpu_geometry->num_indices,
            1,
            0,
            0,
            0);
    }
    break;

    case Rml::ClipMaskOperation::SetInverse:
    {
        const auto& framebuffer =
            m_layer_stack.GetTopLayer();

        EndCurrentRenderPass();

        // GL3 clears stencil to 1 for the inverse case,
        // then writes 0 where the supplied geometry exists.
        WGPUColorAttachment color_attachment = {};
        color_attachment.view =
            framebuffer.color_view;
        color_attachment.loadOp =
            WGPULoadOp_Load;
        color_attachment.storeOp =
            WGPUStoreOp_Store;
        color_attachment.clearValue =
            {0, 0, 0, 0};

        WGPURenderPassDepthStencilAttachment depth_stencil = {};

        if (framebuffer.depth_stencil_view)
        {
            depth_stencil.view =
                framebuffer.depth_stencil_view;

            depth_stencil.depthLoadOp =
                WGPULoadOp_Clear;

            depth_stencil.depthStoreOp =
                WGPUStoreOp_Store;

            depth_stencil.depthClearValue =
                1.0f;

            depth_stencil.stencilLoadOp =
                WGPULoadOp_Clear;

            depth_stencil.stencilStoreOp =
                WGPUStoreOp_Store;

            depth_stencil.stencilClearValue =
                1;
        }

        WGPURenderPassDescriptor render_pass_desc = {};
        render_pass_desc.colorAttachmentCount = 1;
        render_pass_desc.colorAttachments =
            &color_attachment;

        render_pass_desc.depthStencilAttachment =
            framebuffer.depth_stencil_view
                ? &depth_stencil
                : nullptr;

        m_render_pass =
            wgpuCommandEncoderBeginRenderPass(
                m_command_encoder,
                &render_pass_desc);

        if (!m_render_pass)
            break;

        m_owns_render_pass = true;
        m_active_pipeline =
            WebGPUPipelineId::Count;

        ApplyScissor();

        m_clip_mask_depth = 1;
        m_stencil_ref_value = 0;
        m_stencil_equal = false;

        UsePipeline(
            WebGPUPipelineId::Color_Stencil_SetInverse);

        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            0);

        SubmitTransform(translation);

        wgpuRenderPassEncoderSetVertexBuffer(
            m_render_pass,
            0,
            gpu_geometry->vertex_buffer,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderSetIndexBuffer(
            m_render_pass,
            gpu_geometry->index_buffer,
            WGPUIndexFormat_Uint32,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderDrawIndexed(
            m_render_pass,
            gpu_geometry->num_indices,
            1,
            0,
            0,
            0);

        // The pixels outside the mask remain 1.
        // The pixels inside the mask became 0.
        m_stencil_ref_value = 1;
        m_stencil_equal = true;

        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            m_stencil_ref_value);
    }
    break;

    case Rml::ClipMaskOperation::Intersect:
    {
        const uint32_t old_reference =
            std::min(
                m_stencil_ref_value,
                kStencilMask - 1u);

        const uint32_t new_reference =
            old_reference + 1u;

        m_stencil_equal = true;

        // Existing mask pixels must compare equal
        // before they are incremented.
        UsePipeline(
            WebGPUPipelineId::Color_Stencil_Intersect);

        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            old_reference);

        SubmitTransform(translation);

        wgpuRenderPassEncoderSetVertexBuffer(
            m_render_pass,
            0,
            gpu_geometry->vertex_buffer,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderSetIndexBuffer(
            m_render_pass,
            gpu_geometry->index_buffer,
            WGPUIndexFormat_Uint32,
            0,
            WGPU_WHOLE_SIZE);

        wgpuRenderPassEncoderDrawIndexed(
            m_render_pass,
            gpu_geometry->num_indices,
            1,
            0,
            0,
            0);

        m_stencil_ref_value =
            std::min(
                new_reference,
                kStencilMask);

        if (m_clip_mask_depth < kStencilMask)
            ++m_clip_mask_depth;

        m_stencil_equal = true;

        wgpuRenderPassEncoderSetStencilReference(
            m_render_pass,
            m_stencil_ref_value);
    }
    break;
    }
}

// ============================================================================
// Transform
// ============================================================================

void RenderInterface_WebGPU::SetTransform(
    const Rml::Matrix4f* transform)
{
    m_transform =
        transform
            ? (m_projection * (*transform))
            : m_projection;
}

// ============================================================================
// Layers
// ============================================================================

Rml::LayerHandle
RenderInterface_WebGPU::PushLayer()
{
    EndCurrentRenderPass();

    const Rml::LayerHandle layer =
        m_layer_stack.PushLayer();

    BeginLayerRenderPass(
        m_layer_stack.GetLayer(layer),
        true,
        true);

    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_stencil_equal = false;

    return layer;
}

void RenderInterface_WebGPU::BlitFramebuffer(
    const Gfx::WebGPUFramebuffer& source,
    const Gfx::WebGPUFramebuffer& destination)
{
    if (!m_command_encoder ||
        !source.color_texture ||
        !destination.color_texture)
    {
        return;
    }

    EndCurrentRenderPass();

    WGPUExtent3D extent = {
        std::min(
            source.width,
            destination.width),
        std::min(
            source.height,
            destination.height),
        1
    };

    WGPUTexelCopyTextureInfo source_info = {};
    source_info.texture =
        source.color_texture;

    WGPUTexelCopyTextureInfo destination_info = {};
    destination_info.texture =
        destination.color_texture;

    wgpuCommandEncoderCopyTextureToTexture(
        m_command_encoder,
        &source_info,
        &destination_info,
        &extent);
}

// ============================================================================
// Blur / filters
// ============================================================================

void RenderInterface_WebGPU::RenderBlur(
    float sigma,
    const Gfx::WebGPUFramebuffer& source_destination,
    const Gfx::WebGPUFramebuffer& temp,
    Rml::Rectanglei window)
{
    if (!m_command_encoder ||
        !window.Valid() ||
        sigma <= 0.0f)
    {
        return;
    }

    const auto& tertiary =
        m_layer_stack.GetPostprocessTertiary();

    // Copy source into the temporary buffer.
    BlitFramebuffer(
        source_destination,
        temp);

    BlurUniformData uniform = {};

    const float clamped_sigma =
        std::max(sigma, 0.0f);

    float weights[4] = {};
    float normalization = 0.0f;

    for (int i = 0; i < 4; ++i)
    {
        if (std::abs(clamped_sigma) < 0.1f)
        {
            weights[i] =
                (i == 0)
                    ? 1.0f
                    : 0.0f;
        }
        else
        {
            weights[i] =
                std::exp(
                    -float(i * i) /
                    (2.0f *
                     clamped_sigma *
                     clamped_sigma));
        }

        normalization +=
            (i == 0 ? 1.0f : 2.0f) *
            weights[i];
    }

    if (normalization <= 0.0f)
    {
        weights[0] = 1.0f;
        normalization = 1.0f;
    }

    for (float& weight : weights)
        weight /= normalization;

    std::memcpy(
        uniform.weights,
        weights,
        sizeof(weights));

    const int left =
        std::clamp(
            window.Left(),
            0,
            int(source_destination.width));

    const int top =
        std::clamp(
            window.Top(),
            0,
            int(source_destination.height));

    const int right =
        std::clamp(
            window.Right(),
            left,
            int(source_destination.width));

    const int bottom =
        std::clamp(
            window.Bottom(),
            top,
            int(source_destination.height));

    uniform.tex_coord_min[0] =
        (float(left) + 0.5f) /
        float(source_destination.width);

    uniform.tex_coord_min[1] =
        (float(top) + 0.5f) /
        float(source_destination.height);

    uniform.tex_coord_max[0] =
        (float(right) - 0.5f) /
        float(source_destination.width);

    uniform.tex_coord_max[1] =
        (float(bottom) - 0.5f) /
        float(source_destination.height);

    // Vertical pass.
    EndCurrentRenderPass();

    BeginLayerRenderPass(
        tertiary,
        true,
        false);

    UsePipeline(
        WebGPUPipelineId::Blur);

    uniform.texel_offset[0] = 0.0f;
    uniform.texel_offset[1] =
        1.0f /
        float(std::max(
            temp.height,
            1u));

    wgpuQueueWriteBuffer(
        m_queue,
        m_filter_uniform_buffer,
        0,
        &uniform,
        sizeof(uniform));

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        1,
        temp.texture_bind_group,
        0,
        nullptr);

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        2,
        m_filter_bind_group,
        0,
        nullptr);

    DrawFullscreenQuad();

    EndCurrentRenderPass();

    // Horizontal pass.
    BeginLayerRenderPass(
        temp,
        true,
        false);

    UsePipeline(
        WebGPUPipelineId::Blur);

    uniform.texel_offset[0] =
        1.0f /
        float(std::max(
            tertiary.width,
            1u));

    uniform.texel_offset[1] = 0.0f;

    wgpuQueueWriteBuffer(
        m_queue,
        m_filter_uniform_buffer,
        0,
        &uniform,
        sizeof(uniform));

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        1,
        tertiary.texture_bind_group,
        0,
        nullptr);

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        2,
        m_filter_bind_group,
        0,
        nullptr);

    DrawFullscreenQuad();

    EndCurrentRenderPass();

    // Copy result back into source/destination.
    BlitFramebuffer(
        temp,
        source_destination);

    BeginLayerRenderPass(
        source_destination,
        false,
        false);
}

void RenderInterface_WebGPU::RenderFilters(
    Rml::Span<const Rml::CompiledFilterHandle> filter_handles)
{
    if (filter_handles.empty())
        return;

    const bool saved_scissor_enabled =
        m_scissor_enabled;

    m_scissor_enabled = false;
    ApplyScissor();

    for (auto filter_handle : filter_handles)
    {
        auto* filter =
            reinterpret_cast<CompiledFilter*>(
                filter_handle);

        if (!filter)
            continue;

        const auto& source =
            m_layer_stack.GetPostprocessPrimary();

        auto& destination =
            const_cast<Gfx::WebGPUFramebuffer&>(
                m_layer_stack.GetPostprocessSecondary());

        switch (filter->type)
        {
        case FilterType::Passthrough:
        case FilterType::ColorMatrix:
        {
            EndCurrentRenderPass();

            BeginLayerRenderPass(
                destination,
                true,
                false);

            UsePipeline(
                filter->type ==
                        FilterType::ColorMatrix
                    ? WebGPUPipelineId::ColorMatrix
                    : WebGPUPipelineId::Passthrough_NoBlend);

            if (filter->type ==
                FilterType::ColorMatrix)
            {
                ColorMatrixUniformData uniform = {};

                MatrixToArray(
                    filter->color_matrix,
                    uniform.matrix);

                wgpuQueueWriteBuffer(
                    m_queue,
                    m_filter_uniform_buffer,
                    0,
                    &uniform,
                    sizeof(uniform));

                wgpuRenderPassEncoderSetBindGroup(
                    m_render_pass,
                    2,
                    m_filter_bind_group,
                    0,
                    nullptr);
            }

            if (source.texture_bind_group)
            {
                wgpuRenderPassEncoderSetBindGroup(
                    m_render_pass,
                    1,
                    source.texture_bind_group,
                    0,
                    nullptr);
            }

            DrawFullscreenQuad();

            EndCurrentRenderPass();

            m_layer_stack
                .SwapPostprocessPrimarySecondary();
        }
        break;

        case FilterType::Blur:
        {
            const Rml::Rectanglei window =
                m_scissor_state.Valid()
                    ? m_scissor_state
                    : Rml::Rectanglei(
                          Rml::Vector2i(0, 0),
                          Rml::Vector2i(
                              int(source.width),
                              int(source.height)));

            RenderBlur(
                filter->sigma,
                source,
                destination,
                window);
        }
        break;

        case FilterType::DropShadow:
        {
            EndCurrentRenderPass();

            BeginLayerRenderPass(
                destination,
                true,
                false);

            UsePipeline(
                WebGPUPipelineId::DropShadow);

            DropShadowUniformData uniform = {};

            const Rml::Colourf colour =
                Colorf(filter->color);

            std::memcpy(
                uniform.color,
                colour.data(),
                sizeof(float) * 4);

            const Rml::Rectanglei window =
                m_scissor_state.Valid()
                    ? m_scissor_state
                    : Rml::Rectanglei(
                          Rml::Vector2i(0, 0),
                          Rml::Vector2i(
                              int(source.width),
                              int(source.height)));

            uniform.tex_coord_min[0] =
                (float(window.Left()) + 0.5f) /
                float(source.width);

            uniform.tex_coord_min[1] =
                (float(window.Top()) + 0.5f) /
                float(source.height);

            uniform.tex_coord_max[0] =
                (float(window.Right()) - 0.5f) /
                float(source.width);

            uniform.tex_coord_max[1] =
                (float(window.Bottom()) - 0.5f) /
                float(source.height);

            // The fragment shader consumes this in normalized
            // texture space. Keep pixel-space direction consistent
            // with RmlUi's top-left coordinates.
            uniform.offset[0] =
                -filter->offset.x /
                float(std::max(
                    m_viewport_width,
                    1));

            uniform.offset[1] =
                -filter->offset.y /
                float(std::max(
                    m_viewport_height,
                    1));

            uniform.sigma =
                filter->sigma;

            wgpuQueueWriteBuffer(
                m_queue,
                m_filter_uniform_buffer,
                0,
                &uniform,
                sizeof(uniform));

            if (source.texture_bind_group)
            {
                wgpuRenderPassEncoderSetBindGroup(
                    m_render_pass,
                    1,
                    source.texture_bind_group,
                    0,
                    nullptr);
            }

            wgpuRenderPassEncoderSetBindGroup(
                m_render_pass,
                2,
                m_filter_bind_group,
                0,
                nullptr);

            DrawFullscreenQuad();

            EndCurrentRenderPass();

            if (filter->sigma >= 0.5f)
            {
                RenderBlur(
                    filter->sigma,
                    destination,
                    const_cast<
                        Gfx::WebGPUFramebuffer&>(
                        m_layer_stack
                            .GetPostprocessTertiary()),
                    window);
            }
            else
            {
                BeginLayerRenderPass(
                    destination,
                    false,
                    false);
            }

            // Composite the original image over the shadow.
            UsePipeline(
                WebGPUPipelineId::Passthrough);

            if (source.texture_bind_group)
            {
                wgpuRenderPassEncoderSetBindGroup(
                    m_render_pass,
                    1,
                    source.texture_bind_group,
                    0,
                    nullptr);
            }

            DrawFullscreenQuad();

            EndCurrentRenderPass();

            m_layer_stack
                .SwapPostprocessPrimarySecondary();
        }
        break;

        case FilterType::MaskImage:
        {
            if (!filter->mask_texture)
                break;

            EndCurrentRenderPass();

            BeginLayerRenderPass(
                destination,
                true,
                false);

            UsePipeline(
                WebGPUPipelineId::BlendMask);

            WGPUBindGroupEntry entries[3] = {};

            entries[0].binding = 0;
            entries[0].sampler =
                m_sampler_linear;

            entries[1].binding = 1;
            entries[1].textureView =
                source.color_view;

            entries[2].binding = 2;
            entries[2].textureView =
                filter->mask_texture->view;

            WGPUBindGroupDescriptor desc = {};
            desc.layout =
                m_bgl_blend_mask;
            desc.entryCount = 3;
            desc.entries = entries;

            WGPUBindGroup bind_group =
                wgpuDeviceCreateBindGroup(
                    m_device,
                    &desc);

            if (bind_group)
            {
                wgpuRenderPassEncoderSetBindGroup(
                    m_render_pass,
                    1,
                    bind_group,
                    0,
                    nullptr);

                DrawFullscreenQuad();

                wgpuBindGroupRelease(
                    bind_group);
            }

            EndCurrentRenderPass();

            m_layer_stack
                .SwapPostprocessPrimarySecondary();
        }
        break;

        case FilterType::Invalid:
        default:
            break;
        }
    }

    m_scissor_enabled =
        saved_scissor_enabled;

    ApplyScissor();
}

// ============================================================================
// Layer compositing
// ============================================================================

void RenderInterface_WebGPU::CompositeLayers(
    Rml::LayerHandle source,
    Rml::LayerHandle destination,
    Rml::BlendMode blend_mode,
    Rml::Span<const Rml::CompiledFilterHandle> filters)
{
    EndCurrentRenderPass();

    BlitFramebuffer(
        m_layer_stack.GetLayer(source),
        m_layer_stack.GetPostprocessPrimary());

    RenderFilters(filters);

    BeginLayerRenderPass(
        m_layer_stack.GetLayer(destination),
        false,
        false);

    UsePipeline(
        blend_mode == Rml::BlendMode::Replace
            ? WebGPUPipelineId::Passthrough_NoBlend
            : WebGPUPipelineId::Passthrough);

    const auto& primary =
        m_layer_stack.GetPostprocessPrimary();

    if (primary.texture_bind_group)
    {
        wgpuRenderPassEncoderSetBindGroup(
            m_render_pass,
            1,
            primary.texture_bind_group,
            0,
            nullptr);
    }

    DrawFullscreenQuad();

    EndCurrentRenderPass();

    BeginLayerRenderPass(
        m_layer_stack.GetTopLayer(),
        false,
        false);
}

void RenderInterface_WebGPU::PopLayer()
{
    EndCurrentRenderPass();

    m_layer_stack.PopLayer();

    m_stencil_ref_value = 0;
    m_clip_mask_depth = 0;
    m_stencil_equal = false;

    BeginLayerRenderPass(
        m_layer_stack.GetTopLayer(),
        false,
        true);
}

// ============================================================================
// SaveLayerAsTexture
// ============================================================================

Rml::TextureHandle
RenderInterface_WebGPU::SaveLayerAsTexture()
{
    if (!m_device ||
        !m_command_encoder ||
        !m_scissor_state.Valid())
    {
        return {};
    }

    const int width =
        m_scissor_state.Width();

    const int height =
        m_scissor_state.Height();

    if (width <= 0 ||
        height <= 0)
    {
        return {};
    }

    auto* texture =
        CreateTextureResource(
            m_device,
            m_bgl_texture,
            m_sampler_linear,
            WGPUTextureFormat_RGBA8Unorm,
            uint32_t(width),
            uint32_t(height));

    if (!texture)
        return {};

    const auto& source =
        m_layer_stack.GetTopLayer();

    EndCurrentRenderPass();

    // Render the selected source rectangle directly into the new texture.
    // WebGPU's texture coordinate origin and texture memory row order are
    // both top-left based, so there is no second CPU-side Y flip here.
    WGPURenderPassColorAttachment color_attachment = {};
    color_attachment.view =
        texture->view;
    color_attachment.loadOp =
        WGPULoadOp_Clear;
    color_attachment.storeOp =
        WGPUStoreOp_Store;
    color_attachment.clearValue =
        {0, 0, 0, 0};

    WGPURenderPassDescriptor render_pass_desc = {};
    render_pass_desc.colorAttachmentCount = 1;
    render_pass_desc.colorAttachments =
        &color_attachment;

    m_render_pass =
        wgpuCommandEncoderBeginRenderPass(
            m_command_encoder,
            &render_pass_desc);

    if (!m_render_pass)
    {
        DestroyTextureResource(*texture);
        delete texture;
        return {};
    }

    m_owns_render_pass = true;
    m_active_pipeline =
        WebGPUPipelineId::Count;

    ApplyScissor();

    UsePipeline(
        WebGPUPipelineId::ColorMatrix);

    ColorMatrixUniformData identity = {};
    identity.matrix[0] =
        identity.matrix[5] =
        identity.matrix[10] =
        identity.matrix[15] = 1.0f;

    wgpuQueueWriteBuffer(
        m_queue,
        m_filter_uniform_buffer,
        0,
        &identity,
        sizeof(identity));

    if (source.texture_bind_group)
    {
        wgpuRenderPassEncoderSetBindGroup(
            m_render_pass,
            1,
            source.texture_bind_group,
            0,
            nullptr);
    }

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        2,
        m_filter_bind_group,
        0,
        nullptr);

    // Build a temporary quad whose UVs sample only the requested layer region.
    Rml::Mesh mesh;

    Rml::MeshUtilities::GenerateQuad(
        mesh,
        Rml::Vector2f(-1.0f),
        Rml::Vector2f(2.0f),
        {});

    const float source_width =
        float(std::max(source.width, 1u));

    const float source_height =
        float(std::max(source.height, 1u));

    const float u0 =
        float(m_scissor_state.Left()) /
        source_width;

    const float u1 =
        float(m_scissor_state.Right()) /
        source_width;

    const float v0 =
        float(m_scissor_state.Top()) /
        source_height;

    const float v1 =
        float(m_scissor_state.Bottom()) /
        source_height;

    for (Rml::Vertex& vertex : mesh.vertices)
    {
        // The fullscreen quad's input UV is flipped in Y so that
        // destination top maps to source top in WebGPU.
        const float base_u =
            vertex.tex_coord.x;

        const float base_v =
            1.0f - vertex.tex_coord.y;

        vertex.tex_coord.x =
            u0 + base_u * (u1 - u0);

        vertex.tex_coord.y =
            v0 + base_v * (v1 - v0);

        // Saved texture destination is full size and uses the
        // temporary quad's clip-space coverage.
    }

    const auto temporary_geometry =
        CompileGeometry(
            mesh.vertices,
            mesh.indices);

    if (temporary_geometry)
    {
        auto* geometry =
            reinterpret_cast<
                Gfx::WebGPUGeometry*>(
                temporary_geometry);

        if (geometry &&
            geometry->vertex_buffer &&
            geometry->index_buffer)
        {
            wgpuRenderPassEncoderSetVertexBuffer(
                m_render_pass,
                0,
                geometry->vertex_buffer,
                0,
                WGPU_WHOLE_SIZE);

            wgpuRenderPassEncoderSetIndexBuffer(
                m_render_pass,
                geometry->index_buffer,
                WGPUIndexFormat_Uint32,
                0,
                WGPU_WHOLE_SIZE);

            wgpuRenderPassEncoderDrawIndexed(
                m_render_pass,
                geometry->num_indices,
                1,
                0,
                0,
                0);
        }

        ReleaseGeometry(
            temporary_geometry);
    }

    EndCurrentRenderPass();

    BeginLayerRenderPass(
        source,
        false,
        false);

    return reinterpret_cast<
        Rml::TextureHandle>(
        texture);
}

// ============================================================================
// SaveLayerAsMaskImage
// ============================================================================

Rml::CompiledFilterHandle
RenderInterface_WebGPU::SaveLayerAsMaskImage()
{
    auto* filter =
        new CompiledFilter();

    filter->type =
        FilterType::MaskImage;

    const auto& source =
        m_layer_stack.GetTopLayer();

    filter->mask_texture =
        CreateTextureResource(
            m_device,
            m_bgl_texture,
            m_sampler_linear,
            m_render_format,
            source.width,
            source.height);

    if (!filter->mask_texture)
        return reinterpret_cast<
            Rml::CompiledFilterHandle>(
            filter);

    EndCurrentRenderPass();

    WGPURenderPassColorAttachment color_attachment = {};
    color_attachment.view =
        filter->mask_texture->view;
    color_attachment.loadOp =
        WGPULoadOp_Clear;
    color_attachment.storeOp =
        WGPUStoreOp_Store;
    color_attachment.clearValue =
        {0, 0, 0, 0};

    WGPURenderPassDescriptor render_pass_desc = {};
    render_pass_desc.colorAttachmentCount = 1;
    render_pass_desc.colorAttachments =
        &color_attachment;

    m_render_pass =
        wgpuCommandEncoderBeginRenderPass(
            m_command_encoder,
            &render_pass_desc);

    if (!m_render_pass)
    {
        DestroyTextureResource(
            *filter->mask_texture);

        delete filter->mask_texture;
        filter->mask_texture = nullptr;

        return reinterpret_cast<
            Rml::CompiledFilterHandle>(
            filter);
    }

    m_owns_render_pass = true;
    m_active_pipeline =
        WebGPUPipelineId::Count;

    UsePipeline(
        WebGPUPipelineId::ColorMatrix);

    ColorMatrixUniformData identity = {};
    identity.matrix[0] =
        identity.matrix[5] =
        identity.matrix[10] =
        identity.matrix[15] = 1.0f;

    wgpuQueueWriteBuffer(
        m_queue,
        m_filter_uniform_buffer,
        0,
        &identity,
        sizeof(identity));

    if (source.texture_bind_group)
    {
        wgpuRenderPassEncoderSetBindGroup(
            m_render_pass,
            1,
            source.texture_bind_group,
            0,
            nullptr);
    }

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        2,
        m_filter_bind_group,
        0,
        nullptr);

    // Shared fullscreen geometry already has the WebGPU Y orientation.
    DrawFullscreenQuad();

    EndCurrentRenderPass();

    BeginLayerRenderPass(
        source,
        false,
        false);

    return reinterpret_cast<
        Rml::CompiledFilterHandle>(
        filter);
}

// ============================================================================
// Filters
// ============================================================================

Rml::CompiledFilterHandle
RenderInterface_WebGPU::CompileFilter(
    const Rml::String& name,
    const Rml::Dictionary& parameters)
{
    auto* filter =
        new CompiledFilter();

    if (name == "opacity")
    {
        const float value =
            Rml::Math::Clamp(
                Rml::Get(
                    parameters,
                    "value",
                    1.0f),
                0.0f,
                1.0f);

        filter->type =
            FilterType::ColorMatrix;

        // Premultiplied color + alpha are both scaled.
        filter->color_matrix =
            Rml::Matrix4f::Diag(
                value,
                value,
                value,
                value);
    }
    else if (name == "blur")
    {
        filter->type =
            FilterType::Blur;

        filter->sigma =
            std::max(
                Rml::Get(
                    parameters,
                    "sigma",
                    1.0f),
                0.0f);
    }
    else if (name == "drop-shadow")
    {
        filter->type =
            FilterType::DropShadow;

        filter->sigma =
            std::max(
                Rml::Get(
                    parameters,
                    "sigma",
                    0.0f),
                0.0f);

        filter->offset =
            Rml::Get(
                parameters,
                "offset",
                Rml::Vector2f(0.0f));

        filter->color =
            Rml::Get(
                parameters,
                "color",
                Rml::Colourb())
                .ToPremultiplied();
    }
    else if (name == "brightness")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Get(
                parameters,
                "value",
                1.0f);

        filter->color_matrix =
            Rml::Matrix4f::Diag(
                value,
                value,
                value,
                1.0f);
    }
    else if (name == "contrast")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Get(
                parameters,
                "value",
                1.0f);

        const float grayness =
            0.5f - 0.5f * value;

        filter->color_matrix =
            Rml::Matrix4f::Diag(
                value,
                value,
                value,
                1.0f);

        filter->color_matrix.SetColumn(
            3,
            Rml::Vector4f(
                grayness,
                grayness,
                grayness,
                1.0f));
    }
    else if (name == "invert")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Math::Clamp(
                Rml::Get(
                    parameters,
                    "value",
                    1.0f),
                0.0f,
                1.0f);

        const float inverted =
            1.0f - 2.0f * value;

        filter->color_matrix =
            Rml::Matrix4f::Diag(
                inverted,
                inverted,
                inverted,
                1.0f);

        filter->color_matrix.SetColumn(
            3,
            Rml::Vector4f(
                value,
                value,
                value,
                1.0f));
    }
    else if (name == "grayscale")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Get(
                parameters,
                "value",
                1.0f);

        const float reverse =
            1.0f - value;

        const float red =
            0.2126f * value;

        const float green =
            0.7152f * value;

        const float blue =
            0.0722f * value;

        filter->color_matrix =
            Rml::Matrix4f::FromRows(
                {
                    red + reverse,
                    green,
                    blue,
                    0.0f
                },
                {
                    red,
                    green + reverse,
                    blue,
                    0.0f
                },
                {
                    red,
                    green,
                    blue + reverse,
                    0.0f
                },
                {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f
                });
    }
    else if (name == "sepia")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Get(
                parameters,
                "value",
                1.0f);

        const float reverse =
            1.0f - value;

        filter->color_matrix =
            Rml::Matrix4f::FromRows(
                {
                    0.393f * value + reverse,
                    0.769f * value,
                    0.189f * value,
                    0.0f
                },
                {
                    0.349f * value,
                    0.686f * value + reverse,
                    0.168f * value,
                    0.0f
                },
                {
                    0.272f * value,
                    0.534f * value,
                    0.131f * value + reverse,
                    0.0f
                },
                {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f
                });
    }
    else if (name == "hue-rotate")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float angle =
            Rml::Get(
                parameters,
                "value",
                0.0f);

        const float s =
            Rml::Math::Sin(angle);

        const float c =
            Rml::Math::Cos(angle);

        filter->color_matrix =
            Rml::Matrix4f::FromRows(
                {
                    0.213f + 0.787f * c - 0.213f * s,
                    0.715f - 0.715f * c - 0.715f * s,
                    0.072f - 0.072f * c + 0.928f * s,
                    0.0f
                },
                {
                    0.213f - 0.213f * c + 0.143f * s,
                    0.715f + 0.285f * c + 0.140f * s,
                    0.072f - 0.072f * c - 0.283f * s,
                    0.0f
                },
                {
                    0.213f - 0.213f * c - 0.787f * s,
                    0.715f - 0.715f * c + 0.715f * s,
                    0.072f + 0.928f * c + 0.072f * s,
                    0.0f
                },
                {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f
                });
    }
    else if (name == "saturate")
    {
        filter->type =
            FilterType::ColorMatrix;

        const float value =
            Rml::Get(
                parameters,
                "value",
                1.0f);

        filter->color_matrix =
            Rml::Matrix4f::FromRows(
                {
                    0.213f + 0.787f * value,
                    0.715f - 0.715f * value,
                    0.072f - 0.072f * value,
                    0.0f
                },
                {
                    0.213f - 0.213f * value,
                    0.715f + 0.285f * value,
                    0.072f - 0.072f * value,
                    0.0f
                },
                {
                    0.213f - 0.213f * value,
                    0.715f - 0.715f * value,
                    0.072f + 0.928f * value,
                    0.0f
                },
                {
                    0.0f,
                    0.0f,
                    0.0f,
                    1.0f
                });
    }
    else
    {
        delete filter;

        ReportUnsupported(
            name.c_str());

        return {};
    }

    return reinterpret_cast<
        Rml::CompiledFilterHandle>(
        filter);
}

void RenderInterface_WebGPU::ReleaseFilter(
    Rml::CompiledFilterHandle filter_handle)
{
    auto* filter =
        reinterpret_cast<CompiledFilter*>(
            filter_handle);

    if (!filter)
        return;

    if (filter->mask_texture)
    {
        DestroyTextureResource(
            *filter->mask_texture);

        delete filter->mask_texture;
        filter->mask_texture = nullptr;
    }

    delete filter;
}

// ============================================================================
// Custom shaders
// ============================================================================

Rml::CompiledShaderHandle
RenderInterface_WebGPU::CompileShader(
    const Rml::String& name,
    const Rml::Dictionary& parameters)
{
    auto* shader =
        new CompiledShader();

    auto ApplyColorStops =
        [&](const Rml::Dictionary& shader_parameters)
    {
        auto it =
            shader_parameters.find(
                "color_stop_list");

        if (it ==
                shader_parameters.end() ||
            it->second.GetType() !=
                Rml::Variant::COLORSTOPLIST)
        {
            return;
        }

        const auto& color_stop_list =
            it->second.GetReference<
                Rml::ColorStopList>();

        const size_t count =
            std::min(
                color_stop_list.size(),
                size_t(kMaxGradientStops));

        shader->stop_positions.resize(count);
        shader->stop_colors.resize(count);

        for (size_t i = 0; i < count; ++i)
        {
            shader->stop_positions[i] =
                color_stop_list[i]
                    .position.number;

            shader->stop_colors[i] =
                Colorf(
                    color_stop_list[i]
                        .color);
        }
    };

    if (name == "linear-gradient")
    {
        shader->type =
            ShaderType::Gradient;

        const bool repeating =
            Rml::Get(
                parameters,
                "repeating",
                false);

        shader->func =
            repeating ? 1 : 0;

        shader->p =
            Rml::Get(
                parameters,
                "p0",
                Rml::Vector2f(0.0f));

        const Rml::Vector2f p1 =
            Rml::Get(
                parameters,
                "p1",
                Rml::Vector2f(0.0f));

        shader->v =
            p1 - shader->p;

        ApplyColorStops(
            parameters);
    }
    else if (name == "radial-gradient")
    {
        shader->type =
            ShaderType::Gradient;

        const bool repeating =
            Rml::Get(
                parameters,
                "repeating",
                false);

        shader->func =
            repeating ? 3 : 2;

        shader->p =
            Rml::Get(
                parameters,
                "center",
                Rml::Vector2f(0.0f));

        const Rml::Vector2f radius =
            Rml::Get(
                parameters,
                "radius",
                Rml::Vector2f(1.0f));

        shader->v = {
            1.0f / std::max(radius.x, 0.0001f),
            1.0f / std::max(radius.y, 0.0001f)
        };

        ApplyColorStops(
            parameters);
    }
    else if (name == "conic-gradient")
    {
        shader->type =
            ShaderType::Gradient;

        const bool repeating =
            Rml::Get(
                parameters,
                "repeating",
                false);

        shader->func =
            repeating ? 5 : 4;

        shader->p =
            Rml::Get(
                parameters,
                "center",
                Rml::Vector2f(0.0f));

        const float angle =
            Rml::Get(
                parameters,
                "angle",
                0.0f);

        shader->v = {
            Rml::Math::Cos(angle),
            Rml::Math::Sin(angle)
        };

        ApplyColorStops(
            parameters);
    }
    else if (
        name == "shader" &&
        Rml::Get(
            parameters,
            "value",
            Rml::String()) == "creation")
    {
        shader->type =
            ShaderType::Creation;

        shader->dimensions =
            Rml::Get(
                parameters,
                "dimensions",
                Rml::Vector2f(0.0f));
    }
    else
    {
        delete shader;

        ReportUnsupported(
            name.c_str());

        return {};
    }

    return reinterpret_cast<
        Rml::CompiledShaderHandle>(
        shader);
}

void RenderInterface_WebGPU::RenderShader(
    Rml::CompiledShaderHandle shader_handle,
    Rml::CompiledGeometryHandle geometry_handle,
    Rml::Vector2f translation,
    Rml::TextureHandle)
{
    auto* shader =
        reinterpret_cast<CompiledShader*>(
            shader_handle);

    auto* geometry =
        reinterpret_cast<
            Gfx::WebGPUGeometry*>(
            geometry_handle);

    if (!shader ||
        !geometry ||
        !geometry->vertex_buffer ||
        !geometry->index_buffer ||
        !m_render_pass)
    {
        return;
    }

    switch (shader->type)
    {
    case ShaderType::Gradient:
    {
        GradientUniformData uniform = {};

        uniform.func =
            shader->func;

        uniform.num_stops =
            int32_t(
                std::min(
                    shader->stop_positions.size(),
                    size_t(kMaxGradientStops)));

        uniform.p[0] =
            shader->p.x;

        uniform.p[1] =
            shader->p.y;

        uniform.v[0] =
            shader->v.x;

        uniform.v[1] =
            shader->v.y;

        for (size_t i = 0;
             i < shader->stop_positions.size() &&
             i < kMaxGradientStops;
             ++i)
        {
            uniform.stop_positions[i] =
                shader->stop_positions[i];

            for (int c = 0; c < 4; ++c)
            {
                uniform.stop_colors[
                    i * 4 + c] =
                    shader->stop_colors[i][c];
            }
        }

        wgpuQueueWriteBuffer(
            m_queue,
            m_filter_uniform_buffer,
            0,
            &uniform,
            sizeof(uniform));

        UsePipeline(
            WebGPUPipelineId::Gradient);
    }
    break;

    case ShaderType::Creation:
    {
        CreationUniformData uniform = {};

        uniform.dimensions[0] =
            shader->dimensions.x;

        uniform.dimensions[1] =
            shader->dimensions.y;

        auto* system_interface =
            Rml::GetSystemInterface();

        uniform.value =
            system_interface
                ? float(system_interface
                             ->GetElapsedTime())
                : 0.0f;

        wgpuQueueWriteBuffer(
            m_queue,
            m_filter_uniform_buffer,
            0,
            &uniform,
            sizeof(uniform));

        UsePipeline(
            WebGPUPipelineId::Creation);
    }
    break;

    case ShaderType::Invalid:
    default:
        Rml::Log::Message(
            Rml::Log::LT_WARNING,
            "WebGPU backend: invalid render shader.");

        return;
    }

    SubmitTransform(
        translation);

    wgpuRenderPassEncoderSetBindGroup(
        m_render_pass,
        2,
        m_filter_bind_group,
        0,
        nullptr);

    wgpuRenderPassEncoderSetVertexBuffer(
        m_render_pass,
        0,
        geometry->vertex_buffer,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderSetIndexBuffer(
        m_render_pass,
        geometry->index_buffer,
        WGPUIndexFormat_Uint32,
        0,
        WGPU_WHOLE_SIZE);

    wgpuRenderPassEncoderDrawIndexed(
        m_render_pass,
        geometry->num_indices,
        1,
        0,
        0,
        0);
}

void RenderInterface_WebGPU::ReleaseShader(
    Rml::CompiledShaderHandle shader_handle)
{
    delete reinterpret_cast<
        CompiledShader*>(
            shader_handle);
}
