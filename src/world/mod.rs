mod chunk;

pub use chunk::{CHUNK_EDGE, Chunk, ChunkCoord, RayHit};

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::Vec3;

pub const STREAM_RADIUS: i32 = 5;
const MAX_CHUNKS_INTEGRATED_PER_FRAME: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRayHit {
    pub chunk: ChunkCoord,
    pub voxel: [i32; 3],
    pub material: u8,
    pub distance: f32,
}

#[derive(Default)]
pub struct WorldChunkChanges {
    pub loaded: Vec<ChunkCoord>,
    pub unloaded: Vec<ChunkCoord>,
}

pub struct World {
    chunks: HashMap<ChunkCoord, Chunk>,
    stream_origin: Option<ChunkCoord>,
    stream_dimensions: [u32; 3],
    generation_epoch: u64,
    pending_generation: HashSet<ChunkCoord>,
    chunk_changes: WorldChunkChanges,
    generated_chunks_since_report: u32,
    generation_time_since_report: Duration,
    generator: ChunkGenerator,
}

enum GenerationRequest {
    Generate { epoch: u64, chunks: Vec<ChunkCoord> },
    Shutdown,
}

struct GeneratedChunk {
    epoch: u64,
    coord: ChunkCoord,
    chunk: Chunk,
    generation_time: Duration,
}

struct ChunkGenerator {
    requests: Sender<GenerationRequest>,
    results: Receiver<GeneratedChunk>,
    worker: Option<JoinHandle<()>>,
}

impl ChunkGenerator {
    fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("voxel-chunk-generator".into())
            .spawn(move || {
                let mut request = request_rx.recv();
                'worker: loop {
                    let GenerationRequest::Generate { epoch, chunks } = (match request {
                        Ok(request) => request,
                        Err(_) => break,
                    }) else {
                        break;
                    };

                    for coord in chunks {
                        let start = Instant::now();
                        let chunk = Chunk::generate(coord);
                        let generated = GeneratedChunk {
                            epoch,
                            coord,
                            chunk,
                            generation_time: start.elapsed(),
                        };
                        if result_tx.send(generated).is_err() {
                            break 'worker;
                        }

                        let mut latest_request = None;
                        loop {
                            match request_rx.try_recv() {
                                Ok(GenerationRequest::Shutdown) => break 'worker,
                                Ok(next_request) => latest_request = Some(next_request),
                                Err(TryRecvError::Disconnected) => break 'worker,
                                Err(TryRecvError::Empty) => break,
                            }
                        }
                        if let Some(next_request) = latest_request {
                            request = Ok(next_request);
                            continue 'worker;
                        }
                    }
                    request = request_rx.recv();
                }
            })
            .expect("failed to start chunk generation worker");

        Self {
            requests: request_tx,
            results: result_rx,
            worker: Some(worker),
        }
    }
}

impl Drop for ChunkGenerator {
    fn drop(&mut self) {
        let _ = self.requests.send(GenerationRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Default for World {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            stream_origin: None,
            stream_dimensions: [0; 3],
            generation_epoch: 0,
            pending_generation: HashSet::new(),
            chunk_changes: WorldChunkChanges::default(),
            generated_chunks_since_report: 0,
            generation_time_since_report: Duration::ZERO,
            generator: ChunkGenerator::new(),
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
        let bounds_changed =
            self.stream_origin != Some(origin) || self.stream_dimensions != [diameter; 3];
        let mut changed = bounds_changed;
        if bounds_changed {
            let unloaded: Vec<_> = self
                .chunks
                .keys()
                .copied()
                .filter(|&coord| !is_inside_stream(coord, origin, [diameter; 3]))
                .collect();
            for coord in unloaded {
                self.chunks.remove(&coord);
                self.chunk_changes.unloaded.push(coord);
            }
            self.stream_origin = Some(origin);
            self.stream_dimensions = [diameter; 3];
            self.generation_epoch = self.generation_epoch.wrapping_add(1);

            let mut missing = Vec::new();
            for z in 0..diameter {
                for y in 0..diameter {
                    for x in 0..diameter {
                        let coord = ChunkCoord {
                            x: origin.x + x as i32,
                            y: origin.y + y as i32,
                            z: origin.z + z as i32,
                        };
                        if !self.chunks.contains_key(&coord) {
                            missing.push(coord);
                        }
                    }
                }
            }
            self.pending_generation = missing.iter().copied().collect();
            missing.sort_by_key(|coord| {
                let dx = i64::from(coord.x - center.x);
                let dy = i64::from(coord.y - center.y);
                let dz = i64::from(coord.z - center.z);
                (dx * dx + dy * dy + dz * dz, coord.z, coord.y, coord.x)
            });
            let _ = self.generator.requests.send(GenerationRequest::Generate {
                epoch: self.generation_epoch,
                chunks: missing,
            });
        }

        let mut integrated = 0;
        while integrated < MAX_CHUNKS_INTEGRATED_PER_FRAME {
            let Ok(generated) = self.generator.results.try_recv() else {
                break;
            };
            if generated.epoch == self.generation_epoch
                && is_inside_stream(generated.coord, origin, [diameter; 3])
            {
                self.pending_generation.remove(&generated.coord);
                let was_missing = !self.chunks.contains_key(&generated.coord);
                if was_missing {
                    self.chunks.insert(generated.coord, generated.chunk);
                    self.chunk_changes.loaded.push(generated.coord);
                    self.generated_chunks_since_report += 1;
                    self.generation_time_since_report += generated.generation_time;
                    integrated += 1;
                    changed = true;
                }
            }
        }
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

    pub fn take_chunk_changes(&mut self) -> WorldChunkChanges {
        std::mem::take(&mut self.chunk_changes)
    }

    pub fn pending_chunk_count(&self) -> usize {
        self.pending_generation.len()
    }

    pub fn take_generation_timings(&mut self) -> (u32, Duration) {
        (
            std::mem::take(&mut self.generated_chunks_since_report),
            std::mem::take(&mut self.generation_time_since_report),
        )
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
        let previous = self.chunks.insert(coord, chunk);
        self.pending_generation.remove(&coord);
        if previous.is_none() {
            self.chunk_changes.loaded.push(coord);
        } else if !self.chunk_changes.loaded.contains(&coord) {
            self.chunk_changes.loaded.push(coord);
        }
        previous
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        let removed = self.chunks.remove(&coord);
        if removed.is_some() {
            self.chunk_changes.unloaded.push(coord);
        }
        removed
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

fn is_inside_stream(coord: ChunkCoord, origin: ChunkCoord, dimensions: [u32; 3]) -> bool {
    let offset = [coord.x - origin.x, coord.y - origin.y, coord.z - origin.z];
    offset
        .iter()
        .zip(dimensions)
        .all(|(&axis, dimension)| axis >= 0 && axis < dimension as i32)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use glam::Vec3;

    use super::{CHUNK_EDGE, Chunk, ChunkCoord, STREAM_RADIUS, World};

    #[test]
    fn streaming_loads_and_unloads_a_chunk_size_aware_region() {
        let mut world = World::default();
        assert!(world.stream_around(Vec3::ZERO, 1));
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, 1, 27);
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
        wait_for_loaded_chunks(&mut world, moved, 1, 27);
        assert_eq!(world.loaded_chunk_count(), 27);
        assert!(world.chunk(ChunkCoord { x: 1, y: 0, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: -1, y: 0, z: 0 }).is_none());
    }

    #[test]
    fn one_chunk_window_shift_emits_only_entering_and_leaving_chunks() {
        let mut world = World::default();
        let radius = 1;
        world.stream_around(Vec3::ZERO, radius);
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, radius, 27);
        let initial = world.take_chunk_changes();
        assert_eq!(initial.loaded.len(), 27);

        let moved = Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0);
        world.stream_around(moved, radius);
        wait_for_loaded_chunks(&mut world, moved, radius, 27);
        let changes = world.take_chunk_changes();
        assert_eq!(changes.unloaded.len(), 9);
        assert_eq!(changes.loaded.len(), 9);
    }

    #[test]
    fn generated_surface_uses_global_coordinates_across_chunk_boundaries() {
        let mut world = World::default();
        world.stream_around(Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0), 1);
        wait_for_loaded_chunks(&mut world, Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0), 1, 27);
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
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, STREAM_RADIUS, diameter.pow(3));
        assert_eq!(world.loaded_chunk_count(), diameter.pow(3));
    }

    #[test]
    fn larger_stream_radius_generates_without_blocking_the_caller() {
        let mut world = World::default();
        let radius = 3;
        world.stream_around(Vec3::ZERO, radius);
        let diameter = (radius * 2 + 1) as usize;
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, radius, diameter.pow(3));
    }

    #[test]
    fn latest_stream_request_discards_stale_generation_results() {
        let mut world = World::default();
        world.stream_around(Vec3::ZERO, STREAM_RADIUS);
        let latest_position = Vec3::new((CHUNK_EDGE * 8) as f32, 0.0, 0.0);
        world.stream_around(latest_position, 1);

        wait_for_loaded_chunks(&mut world, latest_position, 1, 27);
        let (origin, dimensions) = world.stream_bounds().unwrap();
        assert_eq!(dimensions, [3; 3]);
        assert!(world.chunk(ChunkCoord { x: 7, y: 0, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: -2, y: 0, z: 0 }).is_none());
        assert!(origin.x <= 7);
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

    fn wait_for_loaded_chunks(world: &mut World, position: Vec3, radius: i32, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while world.loaded_chunk_count() < expected && Instant::now() < deadline {
            world.stream_around(position, radius);
            std::thread::yield_now();
        }
        assert_eq!(world.loaded_chunk_count(), expected);
    }
}
