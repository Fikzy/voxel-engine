@group(0) @binding(0)
var gbuffer: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<i32>(textureDimensions(gbuffer));
    let coords = vec2<i32>(global_id.xy);

    if coords.x >= dimensions.x || coords.y >= dimensions.y {
        return;
    }

    let color = vec4<f32>(f32(coords.x) / f32(dimensions.x), f32(coords.y) / f32(dimensions.y), 0.0, 1.0);
    textureStore(gbuffer, coords, color);
}
