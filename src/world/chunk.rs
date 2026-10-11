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

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{CHUNK_EDGE, Chunk, ChunkCoord};

    #[test]
    fn maps_negative_voxels_to_euclidean_chunk_coordinates() {
        assert_eq!(
            ChunkCoord::from_voxel(-1, -16, -17),
            ChunkCoord {
                x: -1,
                y: -1,
                z: -2
            }
        );
        assert_eq!(ChunkCoord::local_voxel(-1, -16, -17), [15, 0, 15]);
    }

    #[test]
    fn stores_materials_at_valid_local_coordinates() {
        let mut chunk = Chunk::new();
        assert!(chunk.set_voxel([CHUNK_EDGE - 1, 2, 3], 7));
        assert_eq!(chunk.voxel([CHUNK_EDGE - 1, 2, 3]), Some(7));
        assert_eq!(chunk.voxel([CHUNK_EDGE, 2, 3]), None);
        assert!(!chunk.set_voxel([0, 0, CHUNK_EDGE], 1));
    }

    #[test]
    fn raycast_hits_known_voxels_and_respects_chunk_edges() {
        let mut chunk = Chunk::new();
        chunk.set_voxel([8, 8, 4], 1);
        chunk.set_voxel([15, 8, 4], 2);
        chunk.set_voxel([0, 8, 4], 3);

        let center_hit = chunk
            .raycast(Vec3::new(8.5, 8.5, 10.0), -Vec3::Z, 20.0)
            .unwrap();
        assert_eq!(center_hit.voxel, [8, 8, 4]);
        assert_eq!(center_hit.material, 1);

        let edge_hit = chunk
            .raycast(Vec3::new(20.0, 8.5, 4.5), -Vec3::X, 20.0)
            .unwrap();
        assert_eq!(edge_hit.voxel, [15, 8, 4]);
        assert_eq!(edge_hit.material, 2);

        assert!(
            chunk
                .raycast(Vec3::new(15.5, 9.5, 4.5), Vec3::X, 20.0)
                .is_none()
        );
        assert!(
            chunk
                .raycast(Vec3::new(-2.0, 8.5, 4.5), -Vec3::X, 20.0)
                .is_none()
        );

        let entry_hit = chunk
            .raycast(Vec3::new(-2.0, 8.5, 4.5), Vec3::X, 20.0)
            .unwrap();
        assert_eq!(entry_hit.voxel, [0, 8, 4]);
        assert_eq!(entry_hit.material, 3);
    }
}
