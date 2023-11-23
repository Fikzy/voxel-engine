use pollster;
use voxel_engine_core;

fn main() {
    pollster::block_on(voxel_engine_core::run());
}
