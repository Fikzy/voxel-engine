mod chunk;

pub use chunk::{CHUNK_EDGE, Chunk, ChunkCoord, RayHit};

use std::collections::{HashMap, HashSet};

use glam::Vec3;

pub const STREAM_RADIUS: i32 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRayHit {
    pub chunk: ChunkCoord,
    pub voxel: [i32; 3],
    pub material: u8,
    pub distance: f32,
}

pub struct World {
    chunks: HashMap<ChunkCoord, Chunk>,
    stream_origin: Option<ChunkCoord>,
    stream_dimensions: [u32; 3],
}

impl Default for World {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            stream_origin: None,
            stream_dimensions: [0; 3],
        }
    }
}

impl World {
    pub fn stream_around(&mut self, position: Vec3, radius: i32) -> bool {
        let center = ChunkCoord::from_voxel(
            position.x.floor() as i32,
            position.y.floor() as i32,
            position.z.floor() as i32,
        );
        let radius = radius.max(0);
        let diameter = (radius * 2 + 1) as u32;
        let origin = ChunkCoord {
            x: center.x - radius,
            y: center.y - radius,
            z: center.z - radius,
        };
        let desired: HashSet<_> = (0..diameter)
            .flat_map(|z| {
                (0..diameter).flat_map(move |y| {
                    (0..diameter).map(move |x| ChunkCoord {
                        x: origin.x + x as i32,
                        y: origin.y + y as i32,
                        z: origin.z + z as i32,
                    })
                })
            })
            .collect();

        let mut changed = self.chunks.len() != desired.len()
            || desired.iter().any(|coord| !self.chunks.contains_key(coord));
        self.chunks.retain(|coord, _| desired.contains(coord));
        for coord in desired {
            if let std::collections::hash_map::Entry::Vacant(entry) = self.chunks.entry(coord) {
                entry.insert(Chunk::generate(coord));
                changed = true;
            }
        }

        self.stream_origin = Some(origin);
        self.stream_dimensions = [diameter; 3];
        changed
    }

    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
    }

    pub fn stream_bounds(&self) -> Option<(ChunkCoord, [u32; 3])> {
        self.stream_origin
            .map(|origin| (origin, self.stream_dimensions))
    }

    pub fn streamed_chunks(&self) -> impl Iterator<Item = (ChunkCoord, &Chunk)> {
        self.chunks.iter().map(|(&coord, chunk)| (coord, chunk))
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<WorldRayHit> {
        let edge = CHUNK_EDGE as f32;
        self.chunks
            .iter()
            .filter_map(|(&coord, chunk)| {
                let chunk_origin = Vec3::new(coord.x as f32, coord.y as f32, coord.z as f32) * edge;
                let hit = chunk.raycast(origin - chunk_origin, direction, max_distance)?;
                Some(WorldRayHit {
                    chunk: coord,
                    voxel: [
                        coord.x * CHUNK_EDGE as i32 + hit.voxel[0] as i32,
                        coord.y * CHUNK_EDGE as i32 + hit.voxel[1] as i32,
                        coord.z * CHUNK_EDGE as i32 + hit.voxel[2] as i32,
                    ],
                    material: hit.material,
                    distance: hit.distance,
                })
            })
            .min_by(|left, right| left.distance.total_cmp(&right.distance))
    }

    pub fn insert_chunk(&mut self, coord: ChunkCoord, chunk: Chunk) -> Option<Chunk> {
        self.chunks.insert(coord, chunk)
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.chunks.remove(&coord)
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{CHUNK_EDGE, Chunk, ChunkCoord, STREAM_RADIUS, World};

    #[test]
    fn streaming_loads_and_unloads_a_chunk_size_aware_region() {
        let mut world = World::default();
        assert!(world.stream_around(Vec3::ZERO, 1));
        assert_eq!(world.loaded_chunk_count(), 27);
        assert_eq!(
            world.stream_bounds(),
            Some((
                ChunkCoord {
                    x: -1,
                    y: -1,
                    z: -1
                },
                [3; 3]
            ))
        );
        assert!(!world.stream_around(Vec3::splat(0.5), 1));

        let moved = Vec3::new((CHUNK_EDGE * 2) as f32, 0.0, 0.0);
        assert!(world.stream_around(moved, 1));
        assert_eq!(world.loaded_chunk_count(), 27);
        assert!(world.chunk(ChunkCoord { x: 1, y: 0, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: -1, y: 0, z: 0 }).is_none());
    }

    #[test]
    fn generated_surface_uses_global_coordinates_across_chunk_boundaries() {
        let mut world = World::default();
        world.stream_around(Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0), 1);
        let edge = CHUNK_EDGE as i32;
        for z in 0..CHUNK_EDGE {
            let global_z = z as i32;
            let left_surface = super::chunk::terrain_surface_height(edge - 1, global_z);
            let right_surface = super::chunk::terrain_surface_height(edge, global_z);
            let left_y = left_surface.rem_euclid(edge) as usize;
            let right_y = right_surface.rem_euclid(edge) as usize;
            let left = world
                .chunk(ChunkCoord {
                    x: 0,
                    y: left_surface.div_euclid(edge),
                    z: 0,
                })
                .unwrap();
            let right = world
                .chunk(ChunkCoord {
                    x: 1,
                    y: right_surface.div_euclid(edge),
                    z: 0,
                })
                .unwrap();
            assert_eq!(left.voxel([CHUNK_EDGE - 1, left_y, z]), Some(1));
            assert_eq!(right.voxel([0, right_y, z]), Some(1));
        }
    }
    #[test]
    fn default_stream_radius_is_nonempty() {
        let mut world = World::default();
        world.stream_around(Vec3::ZERO, STREAM_RADIUS);
        let diameter = (STREAM_RADIUS * 2 + 1) as usize;
        assert_eq!(world.loaded_chunk_count(), diameter.pow(3));
    }

    #[test]
    fn cpu_world_raycast_crosses_chunk_boundaries() {
        let edge = CHUNK_EDGE as i32;
        let middle = CHUNK_EDGE / 2;
        let mut world = World::default();
        world.stream_around(Vec3::splat(edge as f32), 1);

        world.insert_chunk(ChunkCoord { x: 0, y: 1, z: 1 }, Chunk::new());
        let mut next_chunk = Chunk::new();
        let target_x = 1.min(CHUNK_EDGE - 1);
        next_chunk.set_voxel([target_x, middle, middle], 2);
        world.insert_chunk(ChunkCoord { x: 1, y: 1, z: 1 }, next_chunk);

        let hit = world
            .raycast(
                Vec3::new(
                    edge as f32 - 0.5,
                    edge as f32 + middle as f32 + 0.5,
                    edge as f32 + middle as f32 + 0.5,
                ),
                Vec3::X,
                edge as f32 * 4.0,
            )
            .unwrap();
        assert_eq!(
            hit.voxel,
            [
                edge + target_x as i32,
                edge + middle as i32,
                edge + middle as i32
            ]
        );
        assert_eq!(hit.material, 2);
    }
}
