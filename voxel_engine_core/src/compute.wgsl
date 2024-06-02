const PI = 3.1415926535897932384626433832795;

const RAY_EPSILON = 0.0001;
const MAX_ITERATION = 50u;

const CHUNK_SIZE = 4.0;
const CHUNK_LOAD_RADIUS = 8.0;
const CHUNK_LOAD_DIAMETER = CHUNK_LOAD_RADIUS * 2.0;

const CHUNK_SIZE_VEC3 = vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);
const CHUNK_LOAD_VEC3 = vec3<f32>(CHUNK_LOAD_RADIUS, CHUNK_LOAD_RADIUS, CHUNK_LOAD_RADIUS);

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
    t1: f32,
    t2: f32,
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

fn chunk_coord_to_index(coord: vec3<f32>) -> u32 {
    return u32(
        rem_euclid(coord.x, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER * CHUNK_LOAD_DIAMETER +
        rem_euclid(coord.y, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER +
        rem_euclid(coord.z, CHUNK_LOAD_DIAMETER)
    );
}

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
    let t_delta = CHUNK_SIZE * abs(inv_ray_dir);

    let player_pos = camera.position.xyz;
    // let player_pos = vec3<f32>(camera.position.x, 0.0, camera.position.z);
    // let player_chunk = floor(player_pos / CHUNK_SIZE);
    let player_chunk = vec3<f32>(0.0);

    let load_min = player_chunk * CHUNK_SIZE - CHUNK_LOAD_VEC3 * CHUNK_SIZE + RAY_EPSILON;
    let load_max = player_chunk * CHUNK_SIZE + CHUNK_LOAD_VEC3 * CHUNK_SIZE - RAY_EPSILON;

    let chunk_load_min = player_chunk - CHUNK_LOAD_VEC3;
    let chunk_load_max = player_chunk + CHUNK_LOAD_VEC3;
    
    // var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var color = vec4<f32>(0.53, 0.81, 0.92, 1.0); // sky blue

    var intersection = intersect_box(camera.position.xyz, inv_ray_dir, load_min, load_max);

    if intersection.t2 >= max(intersection.t1, 0.0) {

        var intersection_point = camera.position.xyz;

        if intersection.t1 >= 0.0 { // if the ray starts outside the grid
            intersection_point += ray_dir * intersection.t1;
        }

        intersection_point = clamp(intersection_point, load_min, load_max);

        var chunk_coord = floor(intersection_point / CHUNK_SIZE);
        var chunk_index = 0u;
        var c = 0.0;

        let chunk_min = chunk_coord * CHUNK_SIZE;
        let chunk_max = (chunk_coord + 1.0) * CHUNK_SIZE;

        let tx1 = (chunk_min.x - camera.position.x) * inv_ray_dir.x;
        let tx2 = (chunk_max.x - camera.position.x) * inv_ray_dir.x;

        let ty1 = (chunk_min.y - camera.position.y) * inv_ray_dir.y;
        let ty2 = (chunk_max.y - camera.position.y) * inv_ray_dir.y;

        let tz1 = (chunk_min.z - camera.position.z) * inv_ray_dir.z;
        let tz2 = (chunk_max.z - camera.position.z) * inv_ray_dir.z;

        var tmax = vec3<f32>(max(tx1, tx2), max(ty1, ty2), max(tz1, tz2));

        for (var i = 0u; i < MAX_ITERATION; i++) {

            chunk_index = chunk_coord_to_index(chunk_coord);
            c = f32(chunk_data[chunk_index]) / 255.0;

            if c > 0.0 {
                break;
            }

            // Normal DDA
            if tmax.x < tmax.y {
                if tmax.x < tmax.z {
                    tmax.x += t_delta.x;
                    chunk_coord.x += ray_sign.x;
                } else {
                    tmax.z += t_delta.z;
                    chunk_coord.z += ray_sign.z;
                }
            } else {
                if tmax.y < tmax.z {
                    tmax.y += t_delta.y;
                    chunk_coord.y += ray_sign.y;
                } else {
                    tmax.z += t_delta.z;
                    chunk_coord.z += ray_sign.z;
                }
            }

            if chunk_coord.x < chunk_load_min.x || chunk_coord.x >= chunk_load_max.x ||
                chunk_coord.y < chunk_load_min.y || chunk_coord.y >= chunk_load_max.y ||
                chunk_coord.z < chunk_load_min.z || chunk_coord.z >= chunk_load_max.z {
                break;
            }
        }

        if c > 0.0 {
            color = vec4<f32>(c, c, c, 1.0);
        }
    }

    textureStore(gbuffer, coords, color);
}
