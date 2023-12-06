// Vertex shader

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_id: u32,
) -> VertexOutput {
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0),
        vec2<f32>(0.0, 0.0),
    );

    var out: VertexOutput;
    out.clip_position = vec4<f32>(uvs[vertex_id] * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Fragment shader

@group(0) @binding(0)
var gbuffer: texture_storage_2d<rgba8unorm, read>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(gbuffer, vec2<i32>(in.clip_position.xy));
}
