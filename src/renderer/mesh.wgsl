@group(0) @binding(0)
var raycast_image: texture_2d<f32>;

struct Camera {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(1)
var<uniform> camera: Camera;

struct CompositeVertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct DebugVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct DebugVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn composite_vs(@builtin(vertex_index) vertex_index: u32) -> CompositeVertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: CompositeVertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

@fragment
fn composite_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(raycast_image, vec2<i32>(position.xy), 0);
}

@vertex
fn debug_vs(input: DebugVertexInput) -> DebugVertexOutput {
    var output: DebugVertexOutput;
    output.position = camera.view_projection * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}

@fragment
fn debug_fs(input: DebugVertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 0.85);
}
