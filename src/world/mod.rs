mod chunk;

pub use chunk::{CHUNK_EDGE, Chunk, ChunkCoord, RayHit};

use std::collections::HashMap;

#[derive(Default)]
pub struct World {
    chunks: HashMap<ChunkCoord, Chunk>,
}

impl World {
    pub fn chunk(&self, coord: ChunkCoord) -> Option<&Chunk> {
        self.chunks.get(&coord)
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
