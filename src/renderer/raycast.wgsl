struct Camera {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    camera_position: vec4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<storage, read> voxels: array<u32>;

@group(0) @binding(2)
var output_image: texture_storage_2d<rgba8unorm, write>;

struct RayProbe {
    origin: vec4<f32>,
    direction: vec4<f32>,
};

@group(1) @binding(0)
var<storage, read> ray_probes: array<RayProbe>;

@group(1) @binding(1)
var<storage, read_write> probe_results: array<vec4<u32>>;

struct TraceResult {
    voxel: vec3<i32>,
    material: u32,
    normal: vec3<f32>,
    distance: f32,
    hit: u32,
};

fn trace_chunk(origin: vec3<f32>, ray_direction: vec3<f32>) -> TraceResult {
    let direction = normalize(ray_direction);
    let edge = f32(16);
    var entry = 0.0;
    var exit = 128.0;

    for (var axis = 0; axis < 3; axis += 1) {
        if abs(direction[axis]) < 1.0e-7 {
            if origin[axis] < 0.0 || origin[axis] >= edge {
                return TraceResult(vec3<i32>(-1), 0u, vec3<f32>(0.0), 0.0, 0u);
            }
        } else {
            let first = (0.0 - origin[axis]) / direction[axis];
            let second = (edge - origin[axis]) / direction[axis];
            entry = max(entry, min(first, second));
            exit = min(exit, max(first, second));
        }
    }

    if entry > exit || exit < 0.0 {
        return TraceResult(vec3<i32>(-1), 0u, vec3<f32>(0.0), 0.0, 0u);
    }

    let start = origin + direction * (max(entry, 0.0) + 1.0e-4);
    var voxel = vec3<i32>(floor(start));
    var step = vec3<i32>(0);
    var next_boundary = vec3<f32>(1.0e30);
    var delta = vec3<f32>(1.0e30);
    for (var axis = 0; axis < 3; axis += 1) {
        if abs(direction[axis]) >= 1.0e-7 {
            step[axis] = select(-1, 1, direction[axis] > 0.0);
            let boundary = select(f32(voxel[axis]), f32(voxel[axis] + 1), step[axis] > 0);
            next_boundary[axis] = (boundary - origin[axis]) / direction[axis];
            delta[axis] = abs(1.0 / direction[axis]);
        }
    }

    var distance = max(entry, 0.0);
    var normal = vec3<f32>(0.0);
    for (var iteration = 0; iteration < 48; iteration += 1) {
        if distance > exit || distance > 128.0 {
            break;
        }
        if any(voxel < vec3<i32>(0)) || any(voxel >= vec3<i32>(16)) {
            return TraceResult(vec3<i32>(-1), 0u, vec3<f32>(0.0), 0.0, 0u);
        }

        let index = u32(voxel.x + 16 * (voxel.y + 16 * voxel.z));
        let material = voxels[index];
        if material != 0u {
            return TraceResult(voxel, material, normal, distance, 1u);
        }

        var axis = 0;
        if next_boundary.y < next_boundary.x {
            axis = 1;
        }
        if next_boundary.z < next_boundary[axis] {
            axis = 2;
        }
        distance = next_boundary[axis];
        voxel[axis] += step[axis];
        next_boundary[axis] += delta[axis];
        normal = vec3<f32>(0.0);
        normal[axis] = f32(-step[axis]);
    }

    return TraceResult(vec3<i32>(-1), 0u, vec3<f32>(0.0), 0.0, 0u);
}

fn shade_ray(origin: vec3<f32>, direction: vec3<f32>) -> vec4<f32> {
    let hit = trace_chunk(origin, direction);
    if hit.hit == 0u {
        return vec4<f32>(0.035, 0.045, 0.055, 1.0);
    }

    let palette = array<vec3<f32>, 3>(
        vec3<f32>(0.88, 0.24, 0.16),
        vec3<f32>(0.18, 0.72, 0.38),
        vec3<f32>(0.20, 0.44, 0.92),
    );
    let material_color = palette[min(hit.material - 1u, 2u)];
    let light = 0.62 + 0.38 * max(dot(hit.normal, normalize(vec3<f32>(-0.4, 0.8, 0.5))), 0.0);
    return vec4<f32>(material_color * light, 1.0);
}

@compute @workgroup_size(8, 8)
fn raycast_image(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if f32(invocation.x) >= camera.viewport.x || f32(invocation.y) >= camera.viewport.y {
        return;
    }

    let pixel = vec2<f32>(invocation.xy) + vec2<f32>(0.5);
    let ndc = vec2<f32>(
        pixel.x / camera.viewport.x * 2.0 - 1.0,
        1.0 - pixel.y / camera.viewport.y * 2.0,
    );
    let far_point = camera.inverse_view_projection * vec4<f32>(ndc, 1.0, 1.0);
    let world_point = far_point.xyz / far_point.w;
    let color = shade_ray(camera.camera_position.xyz, world_point - camera.camera_position.xyz);
    textureStore(output_image, vec2<i32>(invocation.xy), color);
}

@compute @workgroup_size(1)
fn validate_rays(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let probe = ray_probes[invocation.x];
    let hit = trace_chunk(probe.origin.xyz, probe.direction.xyz);
    if hit.hit == 0u {
        probe_results[invocation.x] = vec4<u32>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0u);
    } else {
        probe_results[invocation.x] = vec4<u32>(
            u32(hit.voxel.x), u32(hit.voxel.y), u32(hit.voxel.z), hit.material,
        );
    }
}
