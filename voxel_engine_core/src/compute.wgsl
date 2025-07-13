const PI = 3.1415926535897932384626433832795;

const RAY_EPSILON = 0.0001;
const MAX_ITERATION = 50u;

const S_MAX = 23u;

const CHUNK_SIZE = 8u;
// const MAX_DEPTH = log2(CHUNK_SIZE) - 1;

// const CHUNK_LOAD_RADIUS = 8.0;
// const CHUNK_LOAD_DIAMETER = CHUNK_LOAD_RADIUS * 2.0;

// const CHUNK_SIZE_VEC3 = vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
// const CHUNK_LOAD_VEC3 = vec3<f32>(CHUNK_LOAD_RADIUS, CHUNK_LOAD_RADIUS, CHUNK_LOAD_RADIUS);

struct CameraUniform {
    position: vec4<f32>,
    projection: mat4x4<f32>,
    camera_to_world: mat4x4<f32>,
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var gbuffer: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1)
var<storage, read> chunk_data: array<u32>;

struct Intersection {
    tmin: f32,
    tmax: f32,
};

fn intersect_box(ray_origin: vec3<f32>, inv_ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> Intersection {

    let tx1 = (box_min.x - ray_origin.x) * inv_ray_dir.x;
    let tx2 = (box_max.x - ray_origin.x) * inv_ray_dir.x;

    let txmin = min(tx1, tx2);
    let txmax = max(tx1, tx2);

    let ty1 = (box_min.y - ray_origin.y) * inv_ray_dir.y;
    let ty2 = (box_max.y - ray_origin.y) * inv_ray_dir.y;

    let tymin = min(ty1, ty2);
    let tymax = max(ty1, ty2);

    let tz1 = (box_min.z - ray_origin.z) * inv_ray_dir.z;
    let tz2 = (box_max.z - ray_origin.z) * inv_ray_dir.z;

    let tzmin = min(tz1, tz2);
    let tzmax = max(tz1, tz2);

    let tmin = max(max(txmin, tymin), tzmin); // nearest intersection point
    let tmax = min(min(txmax, tymax), tzmax); // farthest intersection point

    return Intersection(tmin, tmax);
}

fn rem_euclid(a: f32, b: f32) -> f32 {
    var m = a % b;
    if m < 0.0 {
        if b < 0.0 {
            m -= b;
        } else {
            m += b;
        }
    }
    return m;
}

// fn chunk_coord_to_index(coord: vec3<f32>) -> u32 {
//     return u32(
//         rem_euclid(coord.x, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER * CHUNK_LOAD_DIAMETER +
//         rem_euclid(coord.y, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER +
//         rem_euclid(coord.z, CHUNK_LOAD_DIAMETER)
//     );
// }

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = vec2<i32>(textureDimensions(gbuffer));
    let coords = vec2<i32>(global_id.xy);

    if coords.x >= dimensions.x || coords.y >= dimensions.y {
        return;
    }

    // https://stackoverflow.com/a/46195462/15873956
    let fov = 2.0 * f32(atan(1.0 / camera.projection[1][1])) * 180.0 / PI;
    let aspect_ratio = camera.projection[1][1] / camera.projection[0][0];

    let tan_view_angle = tan(fov / 2.0 * PI / 180.0);

    // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html
    // Raster space -> NDC space -> Screen space
    let px = (2.0 * ((f32(coords.x) + 0.5) / f32(dimensions.x)) - 1.0) * aspect_ratio * tan_view_angle;
    let py = (1.0 - 2.0 * ((f32(coords.y) + 0.5) / f32(dimensions.y))) * tan_view_angle;

    let pixel_pos = vec3<f32>(px, py, -1.0); // in camera space
    let pixel_pos_world = camera.camera_to_world * vec4<f32>(pixel_pos, 1.0); // in world space

    let ray_dir = normalize(pixel_pos_world.xyz - camera.position.xyz);
    let inv_ray_dir = 1.0 / ray_dir;

    let ray_sign = sign(ray_dir);
    // let t_delta = CHUNK_SIZE * abs(inv_ray_dir);

    let player_pos = camera.position.xyz;
    // let player_pos = vec3<f32>(camera.position.x, 0.0, camera.position.z);
    // let player_chunk = floor(player_pos / CHUNK_SIZE);
    let player_chunk = vec3<f32>(0.0);

    // var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var color = vec4<f32>(0.53, 0.81, 0.92, 1.0); // sky blue

    let chunk_min = player_chunk * f32(CHUNK_SIZE) + RAY_EPSILON;
    let chunk_max = (player_chunk + 1.0) * f32(CHUNK_SIZE) - RAY_EPSILON;

    var intersection = intersect_box(camera.position.xyz, inv_ray_dir, chunk_min, chunk_max);

    var t_coef = vec3<f32>(-inv_ray_dir.x, -inv_ray_dir.y, -inv_ray_dir.z);
    var t_bias = vec3<f32>(camera.position.x * t_coef.x, camera.position.y * t_coef.y, camera.position.z * t_coef.z);

    if intersection.tmax >= max(intersection.tmin, 0.0) {

        var intersection_point = camera.position.xyz;

        if intersection.tmin >= 0.0 { // if the ray starts outside the grid
            intersection_point += ray_dir * intersection.tmin;
        }

        intersection_point = clamp(intersection_point, chunk_min, chunk_max);

        let chunk_middle = (chunk_min + chunk_max) / 2.0;

        var parent = chunk_data[0]; // root node
        var child_descriptor = 0u;
    
        var x = u32(intersection_point.x < chunk_middle.x);
        var y = u32(intersection_point.y < chunk_middle.y);
        var z = u32(intersection_point.z < chunk_middle.z);
        var octant_size = f32(CHUNK_SIZE / 2u);
        var pos = vec3<f32>(f32(x), f32(y), f32(z)) * octant_size;
        var idx = (x << 2u) | (y << 1u) | z;

        // var t_corner = vec3<f32>(camera.position.x * t_coef.x - t_bias.x,
        //                          camera.position.y * t_coef.y - t_bias.y,
        //                          camera.position.z * t_coef.z - t_bias.z);
        // var tc_max = min(t_corner.x, t_corner.y, t_corner.z);

        let stack = array<u32, S_MAX>();
        var scale = S_MAX - 1u;
        var scale_exp2 = 0.5;

        while true {
            var box_max = vec3<f32>(pos.x + octant_size, pos.y + octant_size, pos.z + octant_size);
            intersection = intersect_box(camera.position.xyz, inv_ray_dir, pos, box_max);

            var child_descriptor = parent;
            var child_masks = (parent & 0xff00u) >> 8u;

            // If voxel exists and the ray intersects the box
            if (child_masks & (1u << idx)) != 0u && intersection.tmin <= intersection.tmax {

                // Terminate if voxel is small enough
                // TODO: not sure how to do this?

                color = vec4<f32>(f32(idx) / 8.0, 0.0, 0.0, 1.0);
                break;
            }
        }

        // let valid_mask = (parent & 0xff00u) >> 8u;
        // if (valid_mask & (1u << idx)) != 0u {
        //     color = vec4<f32>(f32(idx) / 8.0, 0.0, 0.0, 1.0);
        // }
    }

    // Notes:
    // - use popcount to count the number of bits set to 1

    textureStore(gbuffer, coords, color);
}
