use glam::Vec3;
use libnoise::Generator2D;

pub const CHUNK_EDGE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;

pub const SURFACE_RANGE: [usize; 2] = [0, 32];

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn from_voxel(x: i32, y: i32, z: i32) -> Self {
        let edge = CHUNK_EDGE as i32;
        Self {
            x: x.div_euclid(edge),
            y: y.div_euclid(edge),
            z: z.div_euclid(edge),
        }
    }

    pub fn local_voxel(x: i32, y: i32, z: i32) -> [i32; 3] {
        let edge = CHUNK_EDGE as i32;
        [x.rem_euclid(edge), y.rem_euclid(edge), z.rem_euclid(edge)]
    }
}

#[derive(Clone)]
pub struct Chunk {
    voxels: Box<[u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub voxel: [i32; 3],
    pub normal: [i32; 3],
    pub material: u8,
    pub distance: f32,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    pub fn new() -> Self {
        Self {
            voxels: vec![0; CHUNK_VOLUME].into_boxed_slice(),
        }
    }

    pub fn generate<T: Generator2D>(coord: ChunkCoord, noise: &T) -> Self {
        let mut chunk = Self::new();
        let edge = CHUNK_EDGE as i32;

        for z in 0..CHUNK_EDGE {
            for x in 0..CHUNK_EDGE {
                let world_x = coord.x * edge + x as i32;
                let world_z = coord.z * edge + z as i32;

                let surface_y = terrain_surface_height(world_x, world_z, noise);

                for y in 0..CHUNK_EDGE {
                    let world_y = coord.y * edge + y as i32;
                    if world_y <= surface_y {
                        let material = if world_y == surface_y {
                            1
                        } else if world_y >= surface_y - 3 {
                            2
                        } else {
                            3
                        };
                        chunk.set_voxel([x as i32, y as i32, z as i32], material);
                    }
                }
            }
        }
        chunk
    }

    pub fn voxel(&self, local: [i32; 3]) -> Option<u8> {
        Self::index(local).map(|index| self.voxels[index])
    }

    pub fn set_voxel(&mut self, local: [i32; 3], material: u8) -> bool {
        let Some(index) = Self::index(local) else {
            return false;
        };
        self.voxels[index] = material;
        true
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        let edge = CHUNK_EDGE as f32;
        let mut entry = 0.0_f32;
        let mut exit = max_distance;
        let mut entry_axis = None;

        for axis in 0..3 {
            let origin_axis = origin[axis];
            let direction_axis = direction[axis];
            if direction_axis.abs() < f32::EPSILON {
                if !(0.0..edge).contains(&origin_axis) {
                    return None;
                }
                continue;
            }

            let first = (0.0 - origin_axis) / direction_axis;
            let second = (edge - origin_axis) / direction_axis;
            let near = first.min(second);
            if near > entry || (entry_axis.is_none() && near == entry) {
                entry = near;
                entry_axis = Some(axis);
            }
            exit = exit.min(first.max(second));
        }

        if entry > exit || exit < 0.0 {
            return None;
        }

        let start = origin + direction * (entry.max(0.0) + 1.0e-4);
        let mut voxel = [0_i32; 3];
        let mut step = [0_i32; 3];
        let mut next_boundary = [f32::INFINITY; 3];
        let mut delta = [f32::INFINITY; 3];
        let mut normal = [0_i32; 3];
        if let Some(axis) = entry_axis {
            normal[axis] = -(direction[axis].signum() as i32);
        }

        for axis in 0..3 {
            voxel[axis] = start[axis].floor() as i32;
            step[axis] = direction[axis].signum() as i32;
            if step[axis] != 0 {
                let boundary = if step[axis] > 0 {
                    voxel[axis] as f32 + 1.0
                } else {
                    voxel[axis] as f32
                };
                next_boundary[axis] = (boundary - origin[axis]) / direction[axis];
                delta[axis] = direction[axis].recip().abs();
            }
        }

        let mut distance = entry.max(0.0);
        while distance <= exit && distance <= max_distance {
            if voxel
                .iter()
                .any(|coordinate| *coordinate < 0 || *coordinate >= CHUNK_EDGE as i32)
            {
                return None;
            }
            let material = self.voxel(voxel)?;
            if material != 0 {
                return Some(RayHit {
                    voxel,
                    normal,
                    material,
                    distance,
                });
            }

            let axis =
                if next_boundary[0] <= next_boundary[1] && next_boundary[0] <= next_boundary[2] {
                    0
                } else if next_boundary[1] <= next_boundary[2] {
                    1
                } else {
                    2
                };
            distance = next_boundary[axis];
            voxel[axis] += step[axis];
            next_boundary[axis] += delta[axis];
            normal = [0; 3];
            normal[axis] = -step[axis];
        }
        None
    }

    pub(crate) fn voxels(&self) -> &[u8] {
        &self.voxels
    }

    fn index([x, y, z]: [i32; 3]) -> Option<usize> {
        let edge = CHUNK_EDGE as i32;
        if x < 0 || y < 0 || z < 0 || x >= edge || y >= edge || z >= edge {
            return None;
        }
        Some((x + edge * (y + edge * z)) as usize)
    }
}

pub(super) fn terrain_surface_height<T: Generator2D>(world_x: i32, world_z: i32, noise: &T) -> i32 {
    let base_frequency = 0.0125;
    let nx = world_x as f64 * base_frequency;
    let nz = world_z as f64 * base_frequency;

    let sample = 1.0 * noise.sample([1.0 * nx, 1.0 * nz])
        + 0.5 * noise.sample([2.0 * nx, 2.0 * nz])
        + 0.25 * noise.sample([4.0 * nx, 4.0 * nz]);
    let sample_normalized = sample / (1.0 + 0.5 + 0.25);
    // [-1;1] -> [0;1]
    let sample_remaped = sample_normalized / 2.0 + 0.5;

    // [0;1] -> [SURFACE_RANGE[0];SURFACE_RANGE[1]]
    let height =
        SURFACE_RANGE[0] as f64 + sample_remaped * (SURFACE_RANGE[1] - SURFACE_RANGE[0]) as f64;

    height as i32
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use libnoise::Source;

    use super::{CHUNK_EDGE, Chunk, ChunkCoord, terrain_surface_height};

    #[test]
    fn maps_negative_voxels_to_euclidean_chunk_coordinates() {
        let edge = CHUNK_EDGE as i32;
        assert_eq!(
            ChunkCoord::from_voxel(-1, -edge, -edge - 1),
            ChunkCoord {
                x: -1,
                y: -1,
                z: -2
            }
        );
        assert_eq!(
            ChunkCoord::local_voxel(-1, -edge, -edge - 1),
            [edge - 1, 0, edge - 1]
        );
    }

    #[test]
    fn stores_materials_at_valid_local_coordinates() {
        let mut chunk = Chunk::new();
        let edge = CHUNK_EDGE as i32;
        let valid = [edge - 1, edge / 2, edge / 2];
        assert!(chunk.set_voxel(valid, 7));
        assert_eq!(chunk.voxel(valid), Some(7));
        assert_eq!(chunk.voxel([edge, 0, 0]), None);
        assert!(!chunk.set_voxel([0, 0, edge], 1));
    }

    #[test]
    fn rejects_local_coordinates_outside_the_chunk() {
        let mut chunk = Chunk::new();
        let edge = CHUNK_EDGE as i32;
        assert!(!chunk.set_voxel([-1, 0, 0], 1));
        assert!(!chunk.set_voxel([0, -1, 0], 1));
        assert!(!chunk.set_voxel([0, 0, -1], 1));
        assert_eq!(chunk.voxel([-1, 0, 0]), None);
        assert_eq!(chunk.voxel([0, -1, edge]), None);
        assert_eq!(chunk.voxel([edge - 1, edge - 1, edge - 1]), Some(0));
    }

    #[test]
    fn raycast_hits_known_voxels_and_respects_chunk_edges() {
        let edge = CHUNK_EDGE as f32;
        let middle = (CHUNK_EDGE / 2) as i32;
        let depth = (CHUNK_EDGE / 4) as i32;
        let mut chunk = Chunk::new();
        if CHUNK_EDGE < 3 {
            chunk.set_voxel([0, 0, 0], 1);
            let hit = chunk
                .raycast(Vec3::new(-1.0, 0.5, 0.5), Vec3::X, edge * 2.0)
                .unwrap();
            assert_eq!(hit.voxel, [0, 0, 0]);
            assert_eq!(hit.material, 1);
            return;
        }
        chunk.set_voxel([middle, middle, depth], 1);
        chunk.set_voxel([(CHUNK_EDGE - 1) as i32, middle, depth], 2);
        chunk.set_voxel([0, middle, depth], 3);

        let center_hit = chunk
            .raycast(
                Vec3::new(middle as f32 + 0.5, middle as f32 + 0.5, edge + 1.0),
                -Vec3::Z,
                edge * 2.0,
            )
            .unwrap();
        assert_eq!(center_hit.voxel, [middle, middle, depth]);
        assert_eq!(center_hit.normal, [0, 0, 1]);
        assert_eq!(center_hit.material, 1);

        let edge_hit = chunk
            .raycast(
                Vec3::new(edge + 4.0, middle as f32 + 0.5, depth as f32 + 0.5),
                -Vec3::X,
                edge * 2.0,
            )
            .unwrap();
        assert_eq!(edge_hit.voxel, [(CHUNK_EDGE - 1) as i32, middle, depth]);
        assert_eq!(edge_hit.normal, [1, 0, 0]);
        assert_eq!(edge_hit.material, 2);

        assert!(
            chunk
                .raycast(Vec3::new(edge - 0.5, 0.5, 0.5), Vec3::X, edge * 2.0,)
                .is_none()
        );
        assert!(
            chunk
                .raycast(
                    Vec3::new(-2.0, middle as f32 + 0.5, depth as f32 + 0.5),
                    -Vec3::X,
                    edge * 2.0,
                )
                .is_none()
        );

        let entry_hit = chunk
            .raycast(
                Vec3::new(-2.0, middle as f32 + 0.5, depth as f32 + 0.5),
                Vec3::X,
                edge * 2.0,
            )
            .unwrap();
        assert_eq!(entry_hit.voxel, [0, middle, depth]);
        assert_eq!(entry_hit.normal, [-1, 0, 0]);
        assert_eq!(entry_hit.material, 3);
    }

    #[test]
    fn terrain_height_is_deterministic() {
        let noise = Source::simplex(42);

        let heights: Vec<_> = (0..CHUNK_EDGE as i32)
            .map(|x| terrain_surface_height(x, 0, &noise))
            .collect();
        let repeated: Vec<_> = (0..CHUNK_EDGE as i32)
            .map(|x| terrain_surface_height(x, 0, &noise))
            .collect();

        assert_eq!(heights, repeated);
    }
}
