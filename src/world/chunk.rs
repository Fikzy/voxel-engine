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

    fn index([x, y, z]: [usize; 3]) -> Option<usize> {
        if x >= CHUNK_EDGE || y >= CHUNK_EDGE || z >= CHUNK_EDGE {
            return None;
        }
        Some(x + CHUNK_EDGE * (y + CHUNK_EDGE * z))
    }
}

#[cfg(test)]
mod tests {
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
}
