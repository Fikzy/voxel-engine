const PI = 3.1415926535897932384626433832795;

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

struct Insterction {
    t: f32,
    hit: bool,
};

fn intersect_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_center: vec3<f32>, sphere_radius: f32) -> Insterction {
    let oc = ray_origin - sphere_center;

    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;

    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 {
        return Insterction(0.0, false);
    }

    let sqrt_delta = sqrt(delta);
    let r1 = (-b - sqrt_delta) / (2.0 * a);
    let r2 = (-b + sqrt_delta) / (2.0 * a);

    if r1 > 0.0 {
        return Insterction(r1, true);
    }

    if r2 > 0.0 {
        return Insterction(r2, true);
    }

    return Insterction(0.0, false);
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

    var spheres = array<vec3<f32>, 7>(
        vec3<f32>(-4.0, 0.0, 0.0),
        vec3<f32>(0.0, -4.0, 0.0),
        vec3<f32>(0.0, 0.0, -4.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(4.0, 0.0, 0.0),
        vec3<f32>(0.0, 4.0, 0.0),
        vec3<f32>(0.0, 0.0, 4.0),
    );

    for (var i = 0; i < 7; i += 1) {
        let intersection = intersect_sphere(camera.position.xyz, ray_dir, spheres[i], 0.5);
        if intersection.hit {
            let color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
            textureStore(gbuffer, coords, color);
            return;
        }
    }

    let color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    textureStore(gbuffer, coords, color);
}
