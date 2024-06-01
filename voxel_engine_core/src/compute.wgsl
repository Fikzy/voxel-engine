const PI = 3.1415926535897932384626433832795;

const RAY_EPSILON = 0.0001;

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
    tmax: vec3<f32>,
    hit: bool,
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

    if tmax > max(tmin, 0.0) {
        return Intersection(tmin, tmax, vec3<f32>(txmax, tymax, tzmax), true);
    }

    return Intersection(0.0, 0.0, vec3<f32>(), false);
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

    let player_pos = camera.position.xyz;
    // let player_pos = vec3<f32>(camera.position.x, 0.0, camera.position.z);
    // let player_chunk = floor(player_pos / CHUNK_SIZE);
    let player_chunk = vec3<f32>(0.0);

    let load_min = player_chunk * CHUNK_SIZE - CHUNK_LOAD_VEC3 * CHUNK_SIZE + RAY_EPSILON;
    let load_max = player_chunk * CHUNK_SIZE + CHUNK_LOAD_VEC3 * CHUNK_SIZE - RAY_EPSILON;

    let chunk_min = player_chunk - CHUNK_LOAD_VEC3;
    let chunk_max = player_chunk + CHUNK_LOAD_VEC3;
    
    // var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var color = vec4<f32>(0.53, 0.81, 0.92, 1.0); // sky blue

    var intersection = intersect_box(camera.position.xyz, inv_ray_dir, load_min, load_max);

    // Inside load zone
    if intersection.hit {

        var intersection_point = camera.position.xyz;

        // color = vec4<f32>(floor(abs(intersection_point)) / CHUNK_LOAD_RADIUS, 1.0);

        if intersection.t1 > RAY_EPSILON {
            intersection_point += ray_dir * intersection.t1;
            intersection_point = clamp(intersection_point, load_min, load_max);
        }

        // var chunk_coord = vec3<f32>();
        var chunk_coord = floor(intersection_point / CHUNK_SIZE);
        var chunk_index = 0u;
        var c = 0.0;

        // DDA
        while true { // max iterations?

            // chunk_coord = floor(intersection_point / CHUNK_SIZE);

            chunk_index = u32(
                rem_euclid(chunk_coord.x, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER * CHUNK_LOAD_DIAMETER +
                rem_euclid(chunk_coord.y, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER +
                rem_euclid(chunk_coord.z, CHUNK_LOAD_DIAMETER)
            );

            c = f32(chunk_data[chunk_index]) / 255.0;

            if c > 0.0 {
                break;
            }

            intersection = intersect_box(
                intersection_point, inv_ray_dir,
                chunk_coord * CHUNK_SIZE,
                chunk_coord * CHUNK_SIZE + CHUNK_SIZE_VEC3
            );
            if !intersection.hit {
                break;
            }

            if intersection.tmax.x < intersection.tmax.y {
                if intersection.tmax.x < intersection.tmax.z {
                    chunk_coord.x += ray_sign.x;
                } else {
                    chunk_coord.z += ray_sign.z;
                }
            } else {
                if intersection.tmax.y < intersection.tmax.z {
                    chunk_coord.y += ray_sign.y;
                } else {
                    chunk_coord.z += ray_sign.z;
                }
            }

            if chunk_coord.x < chunk_min.x || chunk_coord.x >= chunk_max.x ||
                chunk_coord.y < chunk_min.y || chunk_coord.y >= chunk_max.y ||
                chunk_coord.z < chunk_min.z || chunk_coord.z >= chunk_max.z {
                break;
            }

            intersection_point += ray_dir * intersection.t2;
            // if intersection_point.x < load_min.x || intersection_point.x > load_max.x ||
            //     intersection_point.y < load_min.y || intersection_point.y > load_max.y ||
            //     intersection_point.z < load_min.z || intersection_point.z > load_max.z {
            //     break;
            // }
        }

        if c > 0.0 {
            color = vec4<f32>(c, c, c, c);
        }
    }

    textureStore(gbuffer, coords, color);
}
