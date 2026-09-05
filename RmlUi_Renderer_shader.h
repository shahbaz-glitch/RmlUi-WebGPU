#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <cstring>

namespace RmlWebGPU {

// ============================================================================
// Transform Uniform
//
// WGSL layout:
//   mat4x4<f32> = 64 bytes
//   vec2<f32>  = 8 bytes
//   vec2<f32>  = 8 bytes
//
// Total = 80 bytes
// ============================================================================

struct alignas(16) TransformUniformData {
    float transform[16];
    float translate[2];
    float padding[2];
};


// ============================================================================
// Gradient Uniform
//
// WGSL layout:
//
//   func             : i32
//   num_stops        : i32
//   p                : vec2<f32>
//   v                : vec2<f32>
//   padding0         : vec2<f32>
//
//   stop_positions   : array<vec4<f32>, 4>
//   stop_colors      : array<vec4<f32>, 16>
//
// Maximum:
//   16 gradient stops
// ============================================================================

struct alignas(16) GradientUniformData {
    int32_t func;
    int32_t num_stops;

    float p[2];
    float v[2];

    float padding0[2];

    float stop_positions[16];

    float stop_colors[64];
};


// ============================================================================
// Creation Shader Uniform
// ============================================================================

struct alignas(16) CreationUniformData {
    float dimensions[2];
    float value;
    float padding;
};


// ============================================================================
// CSS Color Matrix
//
// CSS color matrix is effectively a 4x5 matrix:
//
//   R' = m00 R + m01 G + m02 B + m03 A + m04
//   G' = m10 R + m11 G + m12 B + m13 A + m14
//   B' = m20 R + m21 G + m22 B + m23 A + m24
//   A' = m30 R + m31 G + m32 B + m33 A + m34
//
// Stored as four vec4 rows plus one vec4 containing the bias.
// The last component of the bias vector is unused/padding.
// ============================================================================

struct alignas(16) ColorMatrixUniformData {
    float matrix[16];
    float bias[4];
};


// ============================================================================
// Blur Uniform
//
// Seven-tap Gaussian blur.
//
// weights[0..3]:
//   center + three symmetric pairs
//
// texel_offset:
//   Direction and size of one blur step.
//
// tex_coord_min/max:
//   Region used by post-processing.
// ============================================================================

struct alignas(16) BlurUniformData {
    float weights[4];

    float texel_offset[2];

    float tex_coord_min[2];
    float tex_coord_max[2];

    float padding[2];
};


// ============================================================================
// Drop Shadow Uniform
// ============================================================================

struct alignas(16) DropShadowUniformData {
    float color[4];

    float tex_coord_min[2];
    float tex_coord_max[2];
};


// ============================================================================
// Fullscreen / Geometry Vertex Shader
//
// Group 0:
//   binding 0 = TransformUniform
//
// RmlUi geometry uses:
//   location 0 = position
//   location 1 = color
//   location 2 = tex_coord
//
// WebGPU depth range is [0, 1].
// ============================================================================

static const char* s_shader_vert_main = R"WGSL(

struct TransformUniform {
    transform: mat4x4<f32>,
    translate: vec2<f32>,
    padding: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> u_data: TransformUniform;


struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) tex_coord: vec2<f32>,
};


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.color = input.color;
    output.tex_coord = input.tex_coord;

    let translated =
        input.position + u_data.translate;

    let position =
        u_data.transform *
        vec4<f32>(translated, 0.0, 1.0);

    // RmlUi's projection uses the OpenGL-style [-1,+1]
    // depth convention. Convert Z to WebGPU [0,1].
    output.position =
        vec4<f32>(
            position.xy,
            position.z * 0.5 + 0.5,
            position.w
        );

    return output;
}

)WGSL";


// ============================================================================
// Passthrough Vertex Shader
//
// Used for framebuffer/postprocess rendering.
//
// No transform uniform is required by the shader itself.
// The pipeline layout can still contain Group 0 for the common
// renderer contract.
// ============================================================================

static const char* s_shader_vert_passthrough = R"WGSL(

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) tex_coord: vec2<f32>,
};


struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    output.position =
        vec4<f32>(
            input.position,
            0.0,
            1.0
        );

    output.color = input.color;
    output.tex_coord = input.tex_coord;

    return output;
}

)WGSL";


// ============================================================================
// Flat Color Fragment Shader
// ============================================================================

static const char* s_shader_frag_color = R"WGSL(

struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    return input.color;
}

)WGSL";


// ============================================================================
// Texture Fragment Shader
//
// Group 1:
//   binding 0 = sampler
//   binding 1 = texture
//
// RmlUi colors/textures are premultiplied-alpha data.
// ============================================================================

static const char* s_shader_frag_texture = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let tex_color =
        textureSample(
            t_texture,
            s_sampler,
            input.tex_coord
        );

    return input.color * tex_color;
}

)WGSL";


// ============================================================================
// Gradient Fragment Shader
//
// Group 2:
//   binding 0 = GradientUniform
//
// Group 1 intentionally unused.
// Pipeline layout must preserve group numbering.
// ============================================================================

static const char* s_shader_frag_gradient = R"WGSL(

struct GradientUniform {
    func: i32,
    num_stops: i32,

    p: vec2<f32>,
    v: vec2<f32>,

    padding0: vec2<f32>,

    stop_positions: array<vec4<f32>, 4>,
    stop_colors: array<vec4<f32>, 16>,
};


@group(2) @binding(0)
var<uniform> u_gradient: GradientUniform;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


fn get_stop_position(index: i32) -> f32 {
    let group_index = index / 4;
    let component = index % 4;

    let values =
        u_gradient.stop_positions[group_index];

    if (component == 0) {
        return values.x;
    }

    if (component == 1) {
        return values.y;
    }

    if (component == 2) {
        return values.z;
    }

    return values.w;
}


fn get_stop_color(index: i32) -> vec4<f32> {
    return u_gradient.stop_colors[index];
}


fn sample_gradient(t: f32) -> vec4<f32> {
    let count =
        max(u_gradient.num_stops, 1);

    if (count <= 1) {
        return get_stop_color(0);
    }

    let clamped_t =
        clamp(t, 0.0, 1.0);

    if (clamped_t <= get_stop_position(0)) {
        return get_stop_color(0);
    }

    for (var i: i32 = 0; i < 15; i = i + 1) {
        if (i >= count - 1) {
            break;
        }

        let p0 =
            get_stop_position(i);

        let p1 =
            get_stop_position(i + 1);

        if (clamped_t <= p1) {
            let range =
                max(p1 - p0, 0.000001);

            let factor =
                clamp(
                    (clamped_t - p0) / range,
                    0.0,
                    1.0
                );

            return mix(
                get_stop_color(i),
                get_stop_color(i + 1),
                factor
            );
        }
    }

    return get_stop_color(count - 1);
}


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    var t = 0.0;

    if (u_gradient.func == 0) {
        // Linear gradient.
        let relative =
            input.tex_coord - u_gradient.p;

        let denominator =
            max(
                dot(u_gradient.v, u_gradient.v),
                0.000001
            );

        t =
            dot(relative, u_gradient.v) /
            denominator;
    }
    else {
        // Radial-style fallback.
        let relative =
            input.tex_coord - u_gradient.p;

        let radius =
            max(length(u_gradient.v), 0.000001);

        t =
            length(relative) /
            radius;
    }

    return
        input.color *
        sample_gradient(t);
}

)WGSL";


// ============================================================================
// Creation Fragment Shader
// ============================================================================

static const char* s_shader_frag_creation = R"WGSL(

struct CreationUniform {
    dimensions: vec2<f32>,
    value: f32,
    padding: f32,
};


@group(2) @binding(0)
var<uniform> u_creation: CreationUniform;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    let dimensions =
        max(
            u_creation.dimensions,
            vec2<f32>(0.000001, 0.000001)
        );

    let uv =
        input.tex_coord;

    let centered =
        uv - vec2<f32>(0.5, 0.5);

    let normalized =
        centered * dimensions;

    let distance_value =
        length(normalized);

    let radius =
        max(
            u_creation.value,
            0.000001
        );

    let alpha =
        1.0 -
        clamp(
            distance_value / radius,
            0.0,
            1.0
        );

    return
        input.color *
        vec4<f32>(
            1.0,
            1.0,
            1.0,
            alpha
        );
}

)WGSL";


// ============================================================================
// Passthrough Fragment Shader
//
// Used for framebuffer copies and postprocess stages.
// ============================================================================

static const char* s_shader_frag_passthrough = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    return textureSample(
        t_texture,
        s_sampler,
        input.tex_coord
    );
}

)WGSL";


// ============================================================================
// CSS Color Matrix Fragment Shader
//
// Group 1:
//   texture
//
// Group 2:
//   4x4 matrix + bias
// ============================================================================

static const char* s_shader_frag_color_matrix = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;


struct ColorMatrixUniform {
    matrix: mat4x4<f32>,
    bias: vec4<f32>,
};


@group(2) @binding(0)
var<uniform> u_matrix: ColorMatrixUniform;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    let source =
        textureSample(
            t_texture,
            s_sampler,
            input.tex_coord
        );

    let transformed =
        u_matrix.matrix *
        source;

    let result =
        transformed +
        u_matrix.bias;

    return
        vec4<f32>(
            clamp(result.rgb, 0.0, 1.0),
            clamp(result.a, 0.0, 1.0)
        );
}

)WGSL";


// ============================================================================
// Blend Mask Fragment Shader
//
// Group 1:
//   binding 0 = sampler
//   binding 1 = source texture
//   binding 2 = mask texture
//
// Both source and mask are sampled at the same UV.
// ============================================================================

static const char* s_shader_frag_blend_mask = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;

@group(1) @binding(2)
var t_mask: texture_2d<f32>;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    let source =
        textureSample(
            t_texture,
            s_sampler,
            input.tex_coord
        );

    let mask =
        textureSample(
            t_mask,
            s_sampler,
            input.tex_coord
        );

    let mask_alpha =
        clamp(mask.a, 0.0, 1.0);

    return
        source *
        input.color *
        mask_alpha;
}

)WGSL";


// ============================================================================
// Blur Fragment Shader
//
// Seven taps:
//   -3, -2, -1, 0, +1, +2, +3
//
// weights:
//   x = center
//   y = +/-1
//   z = +/-2
//   w = +/-3
// ============================================================================

static const char* s_shader_frag_blur = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;


struct BlurUniform {
    weights: vec4<f32>,
    texel_offset: vec2<f32>,
    tex_coord_min: vec2<f32>,
    tex_coord_max: vec2<f32>,
    padding: vec2<f32>,
};


@group(2) @binding(0)
var<uniform> u_blur: BlurUniform;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


fn clamp_uv(uv: vec2<f32>) -> vec2<f32> {
    return clamp(
        uv,
        u_blur.tex_coord_min,
        u_blur.tex_coord_max
    );
}


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    let center =
        clamp_uv(input.tex_coord);

    var result =
        textureSample(
            t_texture,
            s_sampler,
            center
        ) * u_blur.weights.x;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center +
                u_blur.texel_offset
            )
        ) * u_blur.weights.y;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center -
                u_blur.texel_offset
            )
        ) * u_blur.weights.y;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center +
                u_blur.texel_offset * 2.0
            )
        ) * u_blur.weights.z;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center -
                u_blur.texel_offset * 2.0
            )
        ) * u_blur.weights.z;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center +
                u_blur.texel_offset * 3.0
            )
        ) * u_blur.weights.w;

    result +=
        textureSample(
            t_texture,
            s_sampler,
            clamp_uv(
                center -
                u_blur.texel_offset * 3.0
            )
        ) * u_blur.weights.w;

    return result * input.color;
}

)WGSL";


// ============================================================================
// Drop Shadow Fragment Shader
//
// The source texture is expected to contain alpha information.
//
// tex_coord_min/max restrict the valid source region.
// ============================================================================

static const char* s_shader_frag_drop_shadow = R"WGSL(

@group(1) @binding(0)
var s_sampler: sampler;

@group(1) @binding(1)
var t_texture: texture_2d<f32>;


struct DropShadowUniform {
    color: vec4<f32>,
    tex_coord_min: vec2<f32>,
    tex_coord_max: vec2<f32>,
};


@group(2) @binding(0)
var<uniform> u_shadow: DropShadowUniform;


struct FragmentInput {
    @location(0) color: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
};


@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {

    let uv =
        clamp(
            input.tex_coord,
            u_shadow.tex_coord_min,
            u_shadow.tex_coord_max
        );

    let source =
        textureSample(
            t_texture,
            s_sampler,
            uv
        );

    let alpha =
        clamp(source.a, 0.0, 1.0);

    return
        vec4<f32>(
            u_shadow.color.rgb * alpha,
            u_shadow.color.a * alpha
        );
}

)WGSL";


// ============================================================================
// Shader Module Creation
// ============================================================================
//
// Uses the WebGPU C API exactly as requested.
// ============================================================================

inline WGPUShaderModule CreateShaderModule(
    WGPUDevice device,
    const char* wgsl_source)
{
    if (!device || !wgsl_source) {
        return nullptr;
    }

    WGPUShaderSourceWGSL wgsl_desc = {};
    wgsl_desc.chain.next = nullptr;
    wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_desc.code = {
        wgsl_source,
        std::strlen(wgsl_source)
    };

    WGPUShaderModuleDescriptor descriptor = {};
    descriptor.nextInChain =
        reinterpret_cast<const WGPUChainedStruct*>(
            &wgsl_desc);

    return wgpuDeviceCreateShaderModule(
        device,
        &descriptor);
}

} // namespace RmlWebGPU
