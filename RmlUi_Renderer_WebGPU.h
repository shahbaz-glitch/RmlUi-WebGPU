#pragma once

#include <RmlUi/Core/RenderInterface.h>
#include <RmlUi/Core/Types.h>

#include <webgpu/webgpu.h>

#include <array>
#include <cstdint>
#include <vector>

namespace Gfx {

struct WebGPUGeometry {
    WGPUBuffer vertex_buffer = nullptr;
    WGPUBuffer index_buffer = nullptr;
    uint32_t num_indices = 0;
};

struct WebGPUTexture {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    WGPUBindGroup bind_group = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct WebGPUFramebuffer {
    WGPUTexture color_texture = nullptr;
    WGPUTextureView color_view = nullptr;
    WGPUTextureView depth_stencil_view = nullptr;
    WGPUBindGroup texture_bind_group = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

} // namespace Gfx


// ============================================================================
// WebGPU Pipeline IDs
// ============================================================================

enum class WebGPUPipelineId {
    // Flat color + stencil
    Color_Stencil_Disabled = 0,
    Color_Stencil_Always,
    Color_Stencil_Equal,
    Color_Stencil_Set,
    Color_Stencil_SetInverse,
    Color_Stencil_Intersect,

    // Textured geometry + stencil
    Texture_Stencil_Disabled,
    Texture_Stencil_Always,
    Texture_Stencil_Equal,

    // RmlUi custom shaders
    Gradient,
    Creation,

    // Framebuffer / postprocess
    Passthrough,
    Passthrough_NoBlend,

    // Filters
    ColorMatrix,
    BlendMask,
    Blur,
    DropShadow,

    Count
};


// ============================================================================
// Main RmlUi WebGPU Render Interface
// ============================================================================

class RenderInterface_WebGPU : public Rml::RenderInterface {
public:
    RenderInterface_WebGPU();
    ~RenderInterface_WebGPU() override;

    // ------------------------------------------------------------------------
    // WebGPU initialization
    // ------------------------------------------------------------------------

    bool Initialize(
        WGPUDevice device,
        WGPUQueue queue,
        WGPUTextureFormat render_target_format,
        WGPUTextureFormat depth_stencil_format =
            WGPUTextureFormat_Depth24PlusStencil8);

    void Shutdown();

    // ------------------------------------------------------------------------
    // Viewport
    // ------------------------------------------------------------------------

    void SetViewport(int width, int height);

    // ------------------------------------------------------------------------
    // Frame lifecycle
    // ------------------------------------------------------------------------

    void BeginFrame(WGPUTextureView target_view);
    void EndFrame();

    // Optional integration with an externally managed render pass.
    void SetActiveRenderPass(WGPURenderPassEncoder pass_encoder);

    // ========================================================================
    // RmlUi RenderInterface overrides
    // ========================================================================

    // ------------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------------

    Rml::CompiledGeometryHandle CompileGeometry(
        Rml::Span<const Rml::Vertex> vertices,
        Rml::Span<const int> indices) override;

    void RenderGeometry(
        Rml::CompiledGeometryHandle handle,
        Rml::Vector2f translation,
        Rml::TextureHandle texture) override;

    void ReleaseGeometry(
        Rml::CompiledGeometryHandle handle) override;

    // ------------------------------------------------------------------------
    // Textures
    // ------------------------------------------------------------------------

    Rml::TextureHandle LoadTexture(
        Rml::Vector2i& texture_dimensions,
        const Rml::String& source) override;

    Rml::TextureHandle GenerateTexture(
        Rml::Span<const Rml::byte> source_data,
        Rml::Vector2i source_dimensions) override;

    void ReleaseTexture(
        Rml::TextureHandle texture_handle) override;

    // ------------------------------------------------------------------------
    // Scissor
    // ------------------------------------------------------------------------

    void EnableScissorRegion(bool enable) override;

    void SetScissorRegion(
        Rml::Rectanglei region) override;

    // ------------------------------------------------------------------------
    // Clip mask
    // ------------------------------------------------------------------------

    void EnableClipMask(bool enable) override;

    void RenderToClipMask(
        Rml::ClipMaskOperation mask_operation,
        Rml::CompiledGeometryHandle geometry,
        Rml::Vector2f translation) override;

    // ------------------------------------------------------------------------
    // Transform
    // ------------------------------------------------------------------------

    void SetTransform(
        const Rml::Matrix4f* transform) override;

    // ------------------------------------------------------------------------
    // Layers / compositing
    // ------------------------------------------------------------------------

    Rml::LayerHandle PushLayer() override;

    void CompositeLayers(
        Rml::LayerHandle source,
        Rml::LayerHandle destination,
        Rml::BlendMode blend_mode,
        Rml::Span<const Rml::CompiledFilterHandle> filters) override;

    void PopLayer() override;

    Rml::TextureHandle SaveLayerAsTexture() override;

    Rml::CompiledFilterHandle SaveLayerAsMaskImage() override;

    // ------------------------------------------------------------------------
    // CSS filters
    // ------------------------------------------------------------------------

    Rml::CompiledFilterHandle CompileFilter(
        const Rml::String& name,
        const Rml::Dictionary& parameters) override;

    void ReleaseFilter(
        Rml::CompiledFilterHandle filter) override;

    // ------------------------------------------------------------------------
    // Custom shaders
    // ------------------------------------------------------------------------

    Rml::CompiledShaderHandle CompileShader(
        const Rml::String& name,
        const Rml::Dictionary& parameters) override;

    void RenderShader(
        Rml::CompiledShaderHandle shader_handle,
        Rml::CompiledGeometryHandle geometry_handle,
        Rml::Vector2f translation,
        Rml::TextureHandle texture) override;

    void ReleaseShader(
        Rml::CompiledShaderHandle shader_handle) override;

    // ------------------------------------------------------------------------
    // Special RmlUi texture handles
    // ------------------------------------------------------------------------

    static constexpr Rml::TextureHandle TextureEnableWithoutBinding =
        Rml::TextureHandle(-1);

    static constexpr Rml::TextureHandle TexturePostprocess =
        Rml::TextureHandle(-2);

private:

    // ========================================================================
    // Resource creation
    // ========================================================================

    void CreateBindGroupLayouts();
    void CreateDefaultSamplers();
    void CreatePipelines();

    // ========================================================================
    // Render pass helpers
    // ========================================================================

    void BeginLayerRenderPass(
        const Gfx::WebGPUFramebuffer& framebuffer,
        bool clear_color,
        bool clear_stencil);

    void EndCurrentRenderPass();

    void ApplyScissor();

    void UsePipeline(
        WebGPUPipelineId pipeline_id);

    void SubmitTransform(
        const Rml::Vector2f& translation);

    void DrawFullscreenQuad();

    // ========================================================================
    // Framebuffer / postprocess helpers
    // ========================================================================

    void BlitFramebuffer(
        const Gfx::WebGPUFramebuffer& source,
        const Gfx::WebGPUFramebuffer& destination);

    void RenderFilters(
        Rml::Span<const Rml::CompiledFilterHandle> filter_handles);

    void RenderBlur(
        float sigma,
        const Gfx::WebGPUFramebuffer& source_destination,
        const Gfx::WebGPUFramebuffer& temp,
        Rml::Rectanglei window);

    // ========================================================================
    // WebGPU device state
    // ========================================================================

    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;

    WGPUTextureFormat m_render_format =
        WGPUTextureFormat_Undefined;

    WGPUTextureFormat m_depth_stencil_format =
        WGPUTextureFormat_Depth24PlusStencil8;

    bool m_initialized = false;

    // ========================================================================
    // Active frame / render pass
    // ========================================================================

    WGPUTextureView m_target_view = nullptr;

    WGPUCommandEncoder m_command_encoder = nullptr;
    WGPURenderPassEncoder m_render_pass = nullptr;

    // True when this class created the active render pass.
    bool m_owns_render_pass = false;

    // ========================================================================
    // Bind group layouts
    //
    // WGSL contract:
    //
    // Group 0 = Transform
    // Group 1 = Texture / Mask
    // Group 2 = Effect uniform
    // ========================================================================

    WGPUBindGroupLayout m_bgl_transform = nullptr;

    WGPUBindGroupLayout m_bgl_texture = nullptr;

    WGPUBindGroupLayout m_bgl_blend_mask = nullptr;

    WGPUBindGroupLayout m_bgl_filter_uniform = nullptr;

    // ========================================================================
    // Pipeline layouts
    // ========================================================================

    // Group 0
    WGPUPipelineLayout m_pipeline_layout_transform = nullptr;

    // Group 0 + Group 1
    WGPUPipelineLayout m_pipeline_layout_transform_texture = nullptr;

    // Group 0 + Group 1 (blend-mask layout)
    WGPUPipelineLayout m_pipeline_layout_blend_mask = nullptr;

    // Group 0 + Group 1 + Group 2
    //
    // Used by:
    //   ColorMatrix
    //   Blur
    //   DropShadow
    //   Gradient
    //   Creation
    //
    // Gradient/Creation do not need to bind Group 1, but the pipeline
    // layout preserves the fixed WGSL group numbering.
    WGPUPipelineLayout m_pipeline_layout_filter = nullptr;

    // ========================================================================
    // Render pipelines
    // ========================================================================

    WGPURenderPipeline
        m_pipelines[static_cast<size_t>(WebGPUPipelineId::Count)] = {};

    WebGPUPipelineId m_active_pipeline =
        WebGPUPipelineId::Count;

    // ========================================================================
    // Samplers
    // ========================================================================

    WGPUSampler m_sampler_linear = nullptr;
    WGPUSampler m_sampler_clamp = nullptr;

    // ========================================================================
    // Transform uniform
    //
    // WGSL:
    //
    // mat4x4<f32> = 64 bytes
    // vec2<f32>  = 8 bytes
    // padding    = 8 bytes
    // Total      = 80 bytes
    //
    // Dynamic uniform offsets are aligned to 256 bytes.
    // ========================================================================

    struct alignas(16) TransformUniform {
        float transform[16];
        float translate[2];
        float padding[2];
    };

    static constexpr uint32_t kUniformAlignment = 256;

    static constexpr uint32_t kMaxUniformDrawsPerFrame = 2048;

    static constexpr uint32_t kTransformUniformBufferSize =
        kUniformAlignment * kMaxUniformDrawsPerFrame;

    WGPUBuffer m_transform_uniform_buffer = nullptr;

    WGPUBindGroup m_transform_bind_group = nullptr;

    uint32_t m_current_uniform_offset = 0;

    // ========================================================================
    // Filter / shader uniform buffer
    // ========================================================================

    static constexpr uint32_t kFilterUniformBufferSize = 1024;

    WGPUBuffer m_filter_uniform_buffer = nullptr;

    WGPUBindGroup m_filter_bind_group = nullptr;

    // ========================================================================
    // Projection / transform state
    // ========================================================================

    Rml::Matrix4f m_transform;
    Rml::Matrix4f m_projection;

    int m_viewport_width = 0;
    int m_viewport_height = 0;

    // ========================================================================
    // Scissor state
    // ========================================================================

    bool m_scissor_enabled = false;

    Rml::Rectanglei m_scissor_state;

    // ========================================================================
    // Clip-mask state
    // ========================================================================

    bool m_stencil_enabled = false;

    // True when normal geometry must pass the current stencil mask.
    bool m_stencil_equal = false;

    // Current stencil reference value.
    uint32_t m_stencil_ref_value = 0;

    // Current clip-mask operation while rendering the mask geometry.
    int m_current_clip_operation = -1;

    // Current clip-mask nesting depth.
    uint32_t m_clip_mask_depth = 0;

    // ========================================================================
    // Fullscreen quad
    // ========================================================================

    Rml::CompiledGeometryHandle m_fullscreen_quad_geometry = {};

    // ========================================================================
    // Layer stack
    // ========================================================================

    class RenderLayerStack {
    public:

        void Initialize(
            WGPUDevice device,
            int width,
            int height,
            WGPUTextureFormat color_format,
            WGPUTextureFormat depth_stencil_format,
            WGPUBindGroupLayout texture_bgl,
            WGPUSampler sampler);

        void Shutdown();

        Rml::LayerHandle PushLayer();
        void PopLayer();

        const Gfx::WebGPUFramebuffer& GetLayer(
            Rml::LayerHandle layer) const;

        const Gfx::WebGPUFramebuffer& GetTopLayer() const;

        Rml::LayerHandle GetTopLayerHandle() const;

        // Postprocess ping-pong buffers.
        const Gfx::WebGPUFramebuffer& GetPostprocessPrimary();
        const Gfx::WebGPUFramebuffer& GetPostprocessSecondary();

        // Additional framebuffer for mask / shadow operations.
        const Gfx::WebGPUFramebuffer& GetPostprocessTertiary();

        const Gfx::WebGPUFramebuffer& GetBlendMask();

        void SwapPostprocessPrimarySecondary();

        void Resize(
            int width,
            int height);

    private:

        void CreateFramebuffer(
            Gfx::WebGPUFramebuffer& framebuffer,
            int width,
            int height,
            bool has_depth_stencil);

        void DestroyFramebuffer(
            Gfx::WebGPUFramebuffer& framebuffer);

        void DestroyFramebuffers();

        // --------------------------------------------------------------------
        // WebGPU resources
        // --------------------------------------------------------------------

        WGPUDevice m_device = nullptr;

        WGPUTextureFormat m_color_format =
            WGPUTextureFormat_Undefined;

        WGPUTextureFormat m_depth_stencil_format =
            WGPUTextureFormat_Undefined;

        WGPUBindGroupLayout m_texture_bgl = nullptr;

        WGPUSampler m_sampler = nullptr;

        // --------------------------------------------------------------------
        // Dimensions
        // --------------------------------------------------------------------

        int m_width = 0;
        int m_height = 0;

        // Number of currently active layers.
        //
        // Layer 0 is always the RmlUi base layer.
        int m_layers_size = 0;

        // --------------------------------------------------------------------
        // Shared depth/stencil
        // --------------------------------------------------------------------

        WGPUTexture m_shared_depth_stencil = nullptr;

        WGPUTextureView m_shared_depth_stencil_view = nullptr;

        // --------------------------------------------------------------------
        // Layer framebuffers
        //
        // Index 0 is reserved for the initial/base RmlUi layer.
        // --------------------------------------------------------------------

        std::vector<Gfx::WebGPUFramebuffer> m_fb_layers;

        // --------------------------------------------------------------------
        // Postprocess buffers
        //
        // 0 = primary
        // 1 = secondary
        // 2 = tertiary
        // 3 = blend/mask
        // --------------------------------------------------------------------

        std::array<Gfx::WebGPUFramebuffer, 4> m_fb_postprocess = {};
    };

    RenderLayerStack m_layer_stack;
};
