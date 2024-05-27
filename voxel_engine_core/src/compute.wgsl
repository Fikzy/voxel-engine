const PI = 3.1415926535897932384626433832795;

const CHUNK_SIZE = 4.0;
const CHUNK_LOAD_RADIUS = 8.0;
const CHUNK_LOAD_DIAMETER = CHUNK_LOAD_RADIUS * 2.0;

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
    t: f32,
    hit: bool
};

fn intersect_box(ray_origin: vec3<f32>, ray_dir: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> Intersection {
    let inv_ray_dir = 1.0 / ray_dir;

    let tx1 = (box_min.x - ray_origin.x) * inv_ray_dir.x;
    let tx2 = (box_max.x - ray_origin.x) * inv_ray_dir.x;

    var tmin = min(tx1, tx2);
    var tmax = max(tx1, tx2);

    let ty1 = (box_min.y - ray_origin.y) * inv_ray_dir.y;
    let ty2 = (box_max.y - ray_origin.y) * inv_ray_dir.y;

    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    let tz1 = (box_min.z - ray_origin.z) * inv_ray_dir.z;
    let tz2 = (box_max.z - ray_origin.z) * inv_ray_dir.z;

    tmin = max(tmin, min(tz1, tz2)); // nearest intersection point
    tmax = min(tmax, max(tz1, tz2)); // farthest intersection point

    if tmax > max(tmin, 0.0) {
        return Intersection(tmin, true);
    }

    return Intersection(0.0, false);
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

    let box_min = vec3<f32>(-CHUNK_SIZE * CHUNK_LOAD_RADIUS, -CHUNK_SIZE * CHUNK_LOAD_RADIUS, -CHUNK_SIZE * CHUNK_LOAD_RADIUS);
    let box_max = vec3<f32>(CHUNK_SIZE * CHUNK_LOAD_RADIUS, CHUNK_SIZE * CHUNK_LOAD_RADIUS, CHUNK_SIZE * CHUNK_LOAD_RADIUS);

    // var color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var color = vec4<f32>(0.53, 0.81, 0.92, 1.0); // sky blue

    let intersection = intersect_box(camera.position.xyz, ray_dir, box_min, box_max);
    if intersection.hit && intersection.t > 0.0 {
        let intersection_point = camera.position.xyz + ray_dir * (intersection.t + 0.0001);

        // color = vec4<f32>(floor(abs(intersection_point)) / CHUNK_LOAD_RADIUS, 1.0);

        let chunk_coord = floor(intersection_point / CHUNK_SIZE);

        let chunk_index = u32(
            rem_euclid(chunk_coord.x, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER * CHUNK_LOAD_DIAMETER +
            rem_euclid(chunk_coord.y, CHUNK_LOAD_DIAMETER) * CHUNK_LOAD_DIAMETER +
            rem_euclid(chunk_coord.z, CHUNK_LOAD_DIAMETER)
        );

        let c = f32(chunk_data[chunk_index]) / 255.0;
        color = vec4<f32>(c, c, c, 1.0);
    }

    textureStore(gbuffer, coords, color);
}
