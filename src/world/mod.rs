mod chunk;

pub use chunk::{CHUNK_EDGE, Chunk, ChunkCoord, RayHit};
use libnoise::Source;

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::Vec3;

pub const STREAM_RADIUS: i32 = 8;
const MAX_CHUNKS_INTEGRATED_PER_FRAME: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldRayHit {
    pub chunk: ChunkCoord,
    pub voxel: [i32; 3],
    pub material: u8,
    pub distance: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct WorldChunkChanges {
    pub loaded: Vec<ChunkCoord>,
    pub updated: Vec<ChunkCoord>,
    pub unloaded: Vec<ChunkCoord>,
}

pub struct World {
    #[allow(unused)]
    seed: u64,
    chunks: HashMap<ChunkCoord, Chunk>,
    stream_origin: Option<ChunkCoord>,
    stream_dimensions: [u32; 3],
    generation_epoch: u64,
    pending_chunks: HashSet<ChunkCoord>,
    chunk_changes: WorldChunkChanges,
    generated_chunks_since_report: u32,
    generation_time_since_report: Duration,
    chunk_manager: ChunkManager,
}

enum ChunkManagerRequest {
    Update {
        epoch: u64,
        unload: Vec<(ChunkCoord, Chunk)>,
        load: Vec<ChunkCoord>,
    },
    Discard {
        coord: ChunkCoord,
    },
    Shutdown,
}

struct LoadedChunk {
    epoch: u64,
    coord: ChunkCoord,
    chunk: Chunk,
    generation_time: Option<Duration>,
}

struct ChunkManager {
    requests: Sender<ChunkManagerRequest>,
    results: Receiver<LoadedChunk>,
    worker: Option<JoinHandle<()>>,
}

impl ChunkManager {
    fn new(seed: u64) -> Self {
        let (request_tx, request_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let noise = Source::simplex(seed);
        let worker = thread::Builder::new()
            .name("voxel-chunk-manager".into())
            .spawn(move || {
                let mut cached_chunks = HashMap::new();
                let mut request = request_rx.recv();
                'worker: loop {
                    let (epoch, unload, load) = match request {
                        Ok(ChunkManagerRequest::Update {
                            epoch,
                            unload,
                            load,
                        }) => (epoch, unload, load),
                        Ok(ChunkManagerRequest::Discard { coord }) => {
                            cached_chunks.remove(&coord);
                            request = request_rx.recv();
                            continue;
                        }
                        Ok(ChunkManagerRequest::Shutdown) | Err(_) => break,
                    };

                    cached_chunks.extend(unload);
                    for coord in load {
                        let (chunk, generation_time) =
                            if let Some(chunk) = cached_chunks.remove(&coord) {
                                (chunk, None)
                            } else {
                                let start = Instant::now();
                                let chunk = Chunk::generate(coord, &noise);
                                (chunk, Some(start.elapsed()))
                            };
                        let loaded = LoadedChunk {
                            epoch,
                            coord,
                            chunk,
                            generation_time,
                        };
                        if result_tx.send(loaded).is_err() {
                            break 'worker;
                        }

                        let mut latest_update = None;
                        loop {
                            match request_rx.try_recv() {
                                Ok(ChunkManagerRequest::Shutdown) => break 'worker,
                                Ok(ChunkManagerRequest::Discard { coord }) => {
                                    cached_chunks.remove(&coord);
                                }
                                Ok(ChunkManagerRequest::Update {
                                    epoch,
                                    unload,
                                    load,
                                }) => {
                                    cached_chunks.extend(unload);
                                    latest_update = Some((epoch, load));
                                }
                                Err(TryRecvError::Disconnected) => break 'worker,
                                Err(TryRecvError::Empty) => break,
                            }
                        }
                        if let Some((epoch, load)) = latest_update {
                            request = Ok(ChunkManagerRequest::Update {
                                epoch,
                                unload: Vec::new(),
                                load,
                            });
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

    fn update(&self, epoch: u64, unload: Vec<(ChunkCoord, Chunk)>, load: Vec<ChunkCoord>) {
        let _ = self.requests.send(ChunkManagerRequest::Update {
            epoch,
            unload,
            load,
        });
    }

    fn discard(&self, coord: ChunkCoord) {
        let _ = self.requests.send(ChunkManagerRequest::Discard { coord });
    }
}

impl Drop for ChunkManager {
    fn drop(&mut self) {
        let _ = self.requests.send(ChunkManagerRequest::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl World {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            chunks: HashMap::new(),
            stream_origin: None,
            stream_dimensions: [0; 3],
            generation_epoch: 0,
            pending_chunks: HashSet::new(),
            chunk_changes: WorldChunkChanges::default(),
            generated_chunks_since_report: 0,
            generation_time_since_report: Duration::ZERO,
            chunk_manager: ChunkManager::new(seed),
        }
    }

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
                .filter(|&coord| !is_inside_stream(coord, origin, [diameter; 3], center, radius))
                .collect();
            let mut chunks_to_cache = Vec::with_capacity(unloaded.len());
            for coord in unloaded {
                if let Some(chunk) = self.chunks.remove(&coord) {
                    chunks_to_cache.push((coord, chunk));
                    self.chunk_changes.unloaded.push(coord);
                }
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
                        if is_inside_stream(coord, origin, [diameter; 3], center, radius)
                            && !self.chunks.contains_key(&coord)
                        {
                            missing.push(coord);
                        }
                    }
                }
            }
            self.pending_chunks = missing.iter().copied().collect();
            // Sort chunks by distance from center to prioritize their loading
            missing.sort_by_key(|coord| {
                let dx = i64::from(coord.x - center.x);
                let dy = i64::from(coord.y - center.y);
                let dz = i64::from(coord.z - center.z);
                (dx * dx + dy * dy + dz * dz, coord.z, coord.y, coord.x)
            });
            self.chunk_manager
                .update(self.generation_epoch, chunks_to_cache, missing);
        }

        let mut integrated = 0;
        while integrated < MAX_CHUNKS_INTEGRATED_PER_FRAME {
            let Ok(loaded) = self.chunk_manager.results.try_recv() else {
                break;
            };
            if loaded.epoch == self.generation_epoch
                && is_inside_stream(loaded.coord, origin, [diameter; 3], center, radius)
                && self.pending_chunks.remove(&loaded.coord)
            {
                let was_missing = !self.chunks.contains_key(&loaded.coord);
                if was_missing {
                    self.chunks.insert(loaded.coord, loaded.chunk);
                    self.chunk_changes.loaded.push(loaded.coord);
                    if let Some(generation_time) = loaded.generation_time {
                        self.generated_chunks_since_report += 1;
                        self.generation_time_since_report += generation_time;
                    }
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
        self.pending_chunks.len()
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
        self.chunk_manager.discard(coord);
        let previous = self.chunks.insert(coord, chunk);
        self.pending_chunks.remove(&coord);
        if previous.is_none() {
            self.chunk_changes.loaded.push(coord);
        } else if !self.chunk_changes.loaded.contains(&coord) {
            self.chunk_changes.loaded.push(coord);
        }
        previous
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) -> Option<Chunk> {
        self.chunk_manager.discard(coord);
        self.pending_chunks.remove(&coord);
        let removed = self.chunks.remove(&coord);
        if removed.is_some() {
            self.chunk_changes.unloaded.push(coord);
        }
        removed
    }

    pub fn set_voxel(&mut self, voxel: [i32; 3], new_material: u8) {
        let chunk_coord = ChunkCoord::from_voxel(voxel[0], voxel[1], voxel[2]);
        let local_voxel = ChunkCoord::local_voxel(voxel[0], voxel[1], voxel[2]);
        if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
            if let Some(prev_material) = chunk.voxel(local_voxel) {
                println!(
                    "Changed voxel ({}, {}, {}) from {} to {}",
                    voxel[0], voxel[1], voxel[2], prev_material, new_material
                );
            } else {
                println!(
                    "Set undefined voxel ({}, {}, {}) to {}",
                    voxel[0], voxel[1], voxel[2], new_material
                );
            }
            chunk.set_voxel(local_voxel, new_material);
            self.chunk_changes.updated.push(chunk_coord);
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

fn is_inside_stream(
    coord: ChunkCoord,
    origin: ChunkCoord,
    dimensions: [u32; 3],
    center: ChunkCoord,
    radius: i32,
) -> bool {
    let offset = [coord.x - origin.x, coord.y - origin.y, coord.z - origin.z];
    let inside_bounds = offset
        .iter()
        .zip(dimensions)
        .all(|(&axis, dimension)| axis >= 0 && axis < dimension as i32);
    let dx = i64::from(coord.x) - i64::from(center.x);
    let dz = i64::from(coord.z) - i64::from(center.z);
    inside_bounds && dx * dx + dz * dz <= i64::from(radius) * i64::from(radius)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use glam::Vec3;
    use libnoise::Source;

    use super::{CHUNK_EDGE, Chunk, ChunkCoord, STREAM_RADIUS, World};

    #[test]
    fn streaming_loads_and_unloads_a_chunk_size_aware_region() {
        let mut world = World::new(42);
        let radius = 1;
        assert!(world.stream_around(Vec3::ZERO, radius));
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, radius, 15);
        assert_eq!(world.loaded_chunk_count(), 15);
        assert!(world.chunk(ChunkCoord { x: 1, y: 0, z: 1 }).is_none());
        assert!(world.chunk(ChunkCoord { x: 0, y: 1, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: 0, y: 2, z: 0 }).is_none());
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
        wait_for_loaded_chunks(&mut world, moved, 1, 15);
        assert_eq!(world.loaded_chunk_count(), 15);
        assert!(world.chunk(ChunkCoord { x: 1, y: 0, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: -1, y: 0, z: 0 }).is_none());
    }

    #[test]
    fn one_chunk_window_shift_emits_only_entering_and_leaving_chunks() {
        let mut world = World::new(42);
        let radius = 1;
        world.stream_around(Vec3::ZERO, radius);
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, radius, 15);
        let initial = world.take_chunk_changes();
        assert_eq!(initial.loaded.len(), 15);

        let moved = Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0);
        world.stream_around(moved, radius);
        wait_for_loaded_chunks(&mut world, moved, radius, 15);
        let changes = world.take_chunk_changes();
        assert_eq!(changes.unloaded.len(), 9);
        assert_eq!(changes.loaded.len(), 9);
    }

    #[test]
    fn streamed_chunks_are_restored_from_memory_after_unloading() {
        let mut world = World::new(42);
        let original_position = Vec3::ZERO;
        let original_coord = ChunkCoord::default();
        world.stream_around(original_position, 0);
        wait_for_loaded_chunks(&mut world, original_position, 0, 1);
        world.set_voxel([1, 2, 3], 7);

        let moved_position = Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0);
        world.stream_around(moved_position, 0);
        wait_for_loaded_chunks(&mut world, moved_position, 0, 1);
        world.take_chunk_changes();
        world.take_generation_timings();

        assert!(world.stream_around(original_position, 0));
        let deadline = Instant::now() + Duration::from_secs(10);
        while world.chunk(original_coord).is_none() && Instant::now() < deadline {
            world.stream_around(original_position, 0);
            std::thread::yield_now();
        }
        assert_eq!(
            world.chunk(original_coord).unwrap().voxel([1, 2, 3]),
            Some(7)
        );
        assert_eq!(world.pending_chunk_count(), 0);
        assert_eq!(world.take_generation_timings(), (0, Duration::ZERO));
        let changes = world.take_chunk_changes();
        assert_eq!(changes.loaded, vec![original_coord]);
        assert_eq!(changes.unloaded, vec![ChunkCoord { x: 1, y: 0, z: 0 }]);
    }

    #[test]
    fn generated_surface_uses_global_coordinates_across_chunk_boundaries() {
        let seed = 42;
        let radius = 3;
        let mut world = World::new(seed);
        world.stream_around(Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0), radius);
        wait_for_loaded_chunks(
            &mut world,
            Vec3::new(CHUNK_EDGE as f32, 0.0, 0.0),
            radius,
            203,
        );
        let edge = CHUNK_EDGE as i32;
        let noise = Source::simplex(seed);
        for z in 0..CHUNK_EDGE {
            let global_z = z as i32;
            let left_surface = super::chunk::terrain_surface_height(edge - 1, global_z, &noise);
            let right_surface = super::chunk::terrain_surface_height(edge, global_z, &noise);
            let left_y = left_surface.rem_euclid(edge);
            let right_y = right_surface.rem_euclid(edge);
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
            assert_eq!(
                left.voxel([(CHUNK_EDGE - 1) as i32, left_y, z as i32]),
                Some(1)
            );
            assert_eq!(right.voxel([0, right_y, z as i32]), Some(1));
        }
    }

    #[test]
    fn default_stream_radius_is_nonempty() {
        let mut world = World::new(42);
        world.stream_around(Vec3::ZERO, STREAM_RADIUS);
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, STREAM_RADIUS, 891);
        assert_eq!(world.loaded_chunk_count(), 891);
    }

    #[test]
    fn larger_stream_radius_generates_without_blocking_the_caller() {
        let mut world = World::new(42);
        let radius = 3;
        world.stream_around(Vec3::ZERO, radius);
        wait_for_loaded_chunks(&mut world, Vec3::ZERO, radius, 203);
    }

    #[test]
    fn latest_stream_request_discards_stale_generation_results() {
        let mut world = World::new(42);
        world.stream_around(Vec3::ZERO, STREAM_RADIUS);
        let latest_position = Vec3::new((CHUNK_EDGE * 8) as f32, 0.0, 0.0);
        world.stream_around(latest_position, 1);

        wait_for_loaded_chunks(&mut world, latest_position, 1, 15);
        let (origin, dimensions) = world.stream_bounds().unwrap();
        assert_eq!(dimensions, [3; 3]);
        assert!(world.chunk(ChunkCoord { x: 7, y: 0, z: 0 }).is_some());
        assert!(world.chunk(ChunkCoord { x: -2, y: 0, z: 0 }).is_none());
        assert!(origin.x <= 7);
    }

    #[test]
    fn cpu_world_raycast_crosses_chunk_boundaries() {
        let edge = CHUNK_EDGE as i32;
        let middle = (CHUNK_EDGE / 2) as i32;
        let mut world = World::new(42);
        world.stream_around(Vec3::splat(edge as f32), 1);

        world.insert_chunk(ChunkCoord { x: 0, y: 1, z: 1 }, Chunk::new());
        let mut next_chunk = Chunk::new();
        let target_x = 1.min(CHUNK_EDGE - 1) as i32;
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

    #[test]
    fn voxel_edits_report_the_affected_chunk_as_updated() {
        let edge = CHUNK_EDGE as i32;
        let mut world = World::new(42);
        let mut chunk = Chunk::new();
        chunk.set_voxel([edge - 1, 0, 0], 1);
        chunk.set_voxel([0, edge - 1, edge - 1], 3);
        world.insert_chunk(ChunkCoord { x: -1, y: 0, z: -1 }, chunk);
        world.take_chunk_changes();

        // Breaking the voxel at the chunk's highest x coordinate.
        world.set_voxel([-1, 0, -edge], 0);
        let changes = world.take_chunk_changes();
        assert_eq!(changes.updated, vec![ChunkCoord { x: -1, y: 0, z: -1 }]);
        assert!(changes.loaded.is_empty());
        assert!(changes.unloaded.is_empty());
        assert_eq!(
            world
                .chunk(ChunkCoord { x: -1, y: 0, z: -1 })
                .unwrap()
                .voxel([edge - 1, 0, 0]),
            Some(0)
        );

        // Placing a block in a previously empty voxel of a neighbouring chunk.
        world.set_voxel([-edge, edge - 1, -1], 4);
        let changes = world.take_chunk_changes();
        assert_eq!(changes.updated, vec![ChunkCoord { x: -1, y: 0, z: -1 }]);
        assert_eq!(
            world
                .chunk(ChunkCoord { x: -1, y: 0, z: -1 })
                .unwrap()
                .voxel([0, edge - 1, edge - 1]),
            Some(4)
        );
    }

    #[test]
    fn voxel_edits_to_unloaded_chunks_are_ignored() {
        let mut world = World::new(42);
        world.set_voxel([0, 0, 0], 1);
        world.set_voxel([-(CHUNK_EDGE as i32), 0, 0], 2);
        let changes = world.take_chunk_changes();
        assert!(changes.updated.is_empty());
        assert!(changes.loaded.is_empty());
        assert!(changes.unloaded.is_empty());
        assert_eq!(world.loaded_chunk_count(), 0);
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
