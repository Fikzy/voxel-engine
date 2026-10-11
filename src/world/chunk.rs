use glam::Vec3;

pub const CHUNK_EDGE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_EDGE * CHUNK_EDGE * CHUNK_EDGE;

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

    pub fn local_voxel(x: i32, y: i32, z: i32) -> [usize; 3] {
        let edge = CHUNK_EDGE as i32;
        [
            x.rem_euclid(edge) as usize,
            y.rem_euclid(edge) as usize,
            z.rem_euclid(edge) as usize,
        ]
    }
}

pub struct Chunk {
    voxels: Box<[u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub voxel: [usize; 3],
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

    pub fn generate(coord: ChunkCoord) -> Self {
        let mut chunk = Self::new();
        let edge = CHUNK_EDGE as i32;

        for z in 0..CHUNK_EDGE {
            for x in 0..CHUNK_EDGE {
                let world_x = coord.x * edge + x as i32;
                let world_z = coord.z * edge + z as i32;
                let surface_y = terrain_surface_height(world_x, world_z);

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
                        chunk.set_voxel([x, y, z], material);
                    }
                }
            }
        }
        chunk
    }

    pub fn voxel(&self, local: [usize; 3]) -> Option<u8> {
        Self::index(local).map(|index| self.voxels[index])
    }

    pub fn set_voxel(&mut self, local: [usize; 3], material: u8) -> bool {
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
            entry = entry.max(first.min(second));
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
            let local = [voxel[0] as usize, voxel[1] as usize, voxel[2] as usize];
            let material = self.voxel(local)?;
            if material != 0 {
                return Some(RayHit {
                    voxel: local,
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
        }
        None
    }

    pub(crate) fn voxels(&self) -> &[u8] {
        &self.voxels
    }

    fn index([x, y, z]: [usize; 3]) -> Option<usize> {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return None;
        }
        Some(x + CHUNK_EDGE * (y + CHUNK_EDGE * z))
    }
}

pub(super) fn terrain_surface_height(world_x: i32, world_z: i32) -> i32 {
    let edge = CHUNK_EDGE as i32;
    let base_height = edge / 2;
    let variation = (edge / 8).max(1);
    let hash = world_x.wrapping_mul(73_856_093) ^ world_z.wrapping_mul(19_349_663);
    base_height + hash.rem_euclid(variation * 2 + 1) - variation
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

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
            [CHUNK_EDGE - 1, 0, CHUNK_EDGE - 1]
        );
    }

    #[test]
    fn stores_materials_at_valid_local_coordinates() {
        let mut chunk = Chunk::new();
        let valid = [CHUNK_EDGE - 1, CHUNK_EDGE / 2, CHUNK_EDGE / 2];
        assert!(chunk.set_voxel(valid, 7));
        assert_eq!(chunk.voxel(valid), Some(7));
        assert_eq!(chunk.voxel([CHUNK_EDGE, 0, 0]), None);
        assert!(!chunk.set_voxel([0, 0, CHUNK_EDGE], 1));
    }

    #[test]
    fn raycast_hits_known_voxels_and_respects_chunk_edges() {
        let edge = CHUNK_EDGE as f32;
        let middle = CHUNK_EDGE / 2;
        let depth = CHUNK_EDGE / 4;
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
        chunk.set_voxel([CHUNK_EDGE - 1, middle, depth], 2);
        chunk.set_voxel([0, middle, depth], 3);

        let center_hit = chunk
            .raycast(
                Vec3::new(middle as f32 + 0.5, middle as f32 + 0.5, edge + 1.0),
                -Vec3::Z,
                edge * 2.0,
            )
            .unwrap();
        assert_eq!(center_hit.voxel, [middle, middle, depth]);
        assert_eq!(center_hit.material, 1);

        let edge_hit = chunk
            .raycast(
                Vec3::new(edge + 4.0, middle as f32 + 0.5, depth as f32 + 0.5),
                -Vec3::X,
                edge * 2.0,
            )
            .unwrap();
        assert_eq!(edge_hit.voxel, [CHUNK_EDGE - 1, middle, depth]);
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
        assert_eq!(entry_hit.material, 3);
    }

    #[test]
    fn terrain_height_is_deterministic_and_varies_per_column() {
        let heights: Vec<_> = (0..CHUNK_EDGE as i32)
            .map(|x| terrain_surface_height(x, 0))
            .collect();
        let repeated: Vec<_> = (0..CHUNK_EDGE as i32)
            .map(|x| terrain_surface_height(x, 0))
            .collect();

        assert_eq!(heights, repeated);
        assert!(heights.windows(2).any(|pair| pair[0] != pair[1]));
    }
}
