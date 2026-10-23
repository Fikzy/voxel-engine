use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    keyboard::{KeyCode, PhysicalKey},
};

use crate::{camera::Camera, world::World};

#[derive(Debug, PartialEq)]
pub enum Action {
    None,
    Break,
    Place,
}

impl Default for Action {
    fn default() -> Self {
        Action::None
    }
}

pub struct WorldInteractor {
    action: Action,
    block_type: u8,
}

impl Default for WorldInteractor {
    fn default() -> Self {
        Self {
            action: Action::None,
            block_type: 1,
        }
    }
}

impl WorldInteractor {
    pub fn process_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Other(_) => return,
            MouseButton::Left => {
                if state.is_pressed() {
                    self.action = Action::Break;
                } else if self.action == Action::Break {
                    self.action = Action::None;
                }
            }
            MouseButton::Right => {
                if state.is_pressed() {
                    self.action = Action::Place;
                } else if self.action == Action::Place {
                    self.action = Action::None;
                }
            }
            _ => (),
        }
    }

    pub fn process_key(&mut self, event: &KeyEvent) {
        self.process_physical_key(event.physical_key, event.state);
    }

    /// Select the block type for a physical key press, independently of
    /// `winit`'s opaque [`KeyEvent`] platform payload.
    pub fn process_physical_key(&mut self, physical_key: PhysicalKey, state: ElementState) {
        let PhysicalKey::Code(key) = physical_key else {
            return;
        };

        if !state.is_pressed() {
            return;
        }

        match key {
            KeyCode::Digit1 => self.block_type = 1,
            KeyCode::Digit2 => self.block_type = 2,
            KeyCode::Digit3 => self.block_type = 3,
            _ => (),
        }
    }

    pub fn update(&mut self, camera: &Camera, world: &mut World) {
        match self.action {
            Action::Break => {
                if let Some(hit) = world.raycast(camera.position, camera.forward(), 10.0) {
                    println!(
                        "Break voxel at ({}, {}, {})",
                        hit.voxel[0], hit.voxel[1], hit.voxel[2],
                    );
                    world.set_voxel(hit.voxel, 0);
                };
                self.action = Action::None;
            }
            Action::Place => {
                if let Some(hit) = world.raycast(camera.position, camera.forward(), 10.0) {
                    let voxel = [
                        hit.voxel[0] + hit.normal[0],
                        hit.voxel[1] + hit.normal[1],
                        hit.voxel[2] + hit.normal[2],
                    ];
                    println!("Place voxel at ({}, {}, {})", voxel[0], voxel[1], voxel[2],);
                    if world.voxel(voxel) == Some(0) {
                        world.set_voxel(voxel, self.block_type);
                    }
                };
                self.action = Action::None;
            }
            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use winit::event::{ElementState, MouseButton};
    use winit::keyboard::{KeyCode, NativeKeyCode, PhysicalKey};

    use super::{Action, WorldInteractor};
    use crate::camera::Camera;
    use crate::world::{CHUNK_EDGE, Chunk, ChunkCoord, World};

    /// A one-voxel world whose chunk covers voxels `[0..CHUNK_EDGE]`, with the
    /// voxel `(8, 8, 8)` filled with material 1.
    fn world_with_target_voxel() -> World {
        let mut world = World::new(42);
        let mut chunk = Chunk::new();
        chunk.set_voxel([8, 8, 8], 1);
        world.insert_chunk(ChunkCoord { x: 0, y: 0, z: 0 }, chunk);
        world.take_chunk_changes();
        world
    }

    /// Camera looking down the -Z axis at the `(8, 8, 8)` voxel.
    fn camera_at_target() -> Camera {
        Camera {
            position: Vec3::new(8.5, 8.5, 8.0 + 4.0),
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn code(key: KeyCode) -> PhysicalKey {
        PhysicalKey::Code(key)
    }

    #[test]
    fn defaults_idle_with_the_first_block_type() {
        let interactor = WorldInteractor::default();
        assert_eq!(interactor.action, Action::None);
        assert_eq!(interactor.block_type, 1);
    }

    #[test]
    fn mouse_presses_arm_break_and_place_then_release_clears_them() {
        let mut interactor = WorldInteractor::default();

        interactor.process_mouse_input(MouseButton::Left, ElementState::Pressed);
        assert_eq!(interactor.action, Action::Break);
        interactor.process_mouse_input(MouseButton::Left, ElementState::Released);
        assert_eq!(interactor.action, Action::None);

        interactor.process_mouse_input(MouseButton::Right, ElementState::Pressed);
        assert_eq!(interactor.action, Action::Place);
        interactor.process_mouse_input(MouseButton::Right, ElementState::Released);
        assert_eq!(interactor.action, Action::None);
    }

    #[test]
    fn unrelated_mouse_buttons_leave_the_action_alone() {
        let mut interactor = WorldInteractor::default();
        interactor.process_mouse_input(MouseButton::Middle, ElementState::Pressed);
        assert_eq!(interactor.action, Action::None);
        interactor.process_mouse_input(MouseButton::Other(4), ElementState::Pressed);
        assert_eq!(interactor.action, Action::None);

        interactor.process_mouse_input(MouseButton::Left, ElementState::Pressed);
        interactor.process_mouse_input(MouseButton::Middle, ElementState::Released);
        assert_eq!(interactor.action, Action::Break);

        interactor.process_mouse_input(MouseButton::Right, ElementState::Pressed);
        interactor.process_mouse_input(MouseButton::Left, ElementState::Released);
        assert_eq!(interactor.action, Action::Place);
    }

    #[test]
    fn digit_keys_select_the_block_type() {
        let mut interactor = WorldInteractor::default();

        interactor.process_physical_key(code(KeyCode::Digit2), ElementState::Pressed);
        assert_eq!(interactor.block_type, 2);
        interactor.process_physical_key(code(KeyCode::Digit3), ElementState::Pressed);
        assert_eq!(interactor.block_type, 3);
        interactor.process_physical_key(code(KeyCode::Digit1), ElementState::Pressed);
        assert_eq!(interactor.block_type, 1);
    }

    #[test]
    fn block_type_selectors_ignore_releases_and_unmapped_keys() {
        let mut interactor = WorldInteractor::default();
        interactor.process_physical_key(code(KeyCode::Digit3), ElementState::Pressed);
        assert_eq!(interactor.block_type, 3);

        interactor.process_physical_key(code(KeyCode::Digit2), ElementState::Released);
        assert_eq!(interactor.block_type, 3);

        interactor.process_physical_key(code(KeyCode::KeyW), ElementState::Pressed);
        assert_eq!(interactor.block_type, 3);

        interactor.process_physical_key(
            PhysicalKey::Unidentified(NativeKeyCode::Unidentified),
            ElementState::Pressed,
        );
        assert_eq!(interactor.block_type, 3);
    }

    #[test]
    fn break_removes_the_hit_voxel_and_reports_the_chunk_as_updated() {
        let mut interactor = WorldInteractor::default();
        let mut world = world_with_target_voxel();
        interactor.process_mouse_input(MouseButton::Left, ElementState::Pressed);

        interactor.update(&camera_at_target(), &mut world);

        assert_eq!(interactor.action, Action::None);
        assert_eq!(
            world
                .chunk(ChunkCoord { x: 0, y: 0, z: 0 })
                .unwrap()
                .voxel([8, 8, 8]),
            Some(0)
        );
        let changes = world.take_chunk_changes();
        assert_eq!(changes.updated, vec![ChunkCoord { x: 0, y: 0, z: 0 }]);

        // The action is consumed, so a second update must not edit anything.
        interactor.update(&camera_at_target(), &mut world);
        assert!(world.take_chunk_changes().updated.is_empty());
    }

    #[test]
    fn place_adds_the_selected_block_next_to_the_hit_voxel() {
        let mut interactor = WorldInteractor::default();
        let mut world = world_with_target_voxel();
        interactor.process_physical_key(code(KeyCode::Digit2), ElementState::Pressed);
        interactor.process_mouse_input(MouseButton::Right, ElementState::Pressed);

        interactor.update(&camera_at_target(), &mut world);

        assert_eq!(interactor.action, Action::None);
        assert_eq!(
            world
                .chunk(ChunkCoord { x: 0, y: 0, z: 0 })
                .unwrap()
                .voxel([8, 8, 8]),
            Some(1)
        );
        assert_eq!(
            world
                .chunk(ChunkCoord { x: 0, y: 0, z: 0 })
                .unwrap()
                .voxel([8, 8, 9]),
            Some(2)
        );
        assert_eq!(
            world.take_chunk_changes().updated,
            vec![ChunkCoord { x: 0, y: 0, z: 0 }]
        );
    }

    #[test]
    fn place_does_not_write_into_an_unloaded_adjacent_chunk() {
        let mut interactor = WorldInteractor::default();
        let mut world = World::new(42);
        let mut chunk = Chunk::new();
        chunk.set_voxel([8, 8, CHUNK_EDGE as i32 - 1], 1);
        world.insert_chunk(ChunkCoord { x: 0, y: 0, z: 0 }, chunk);
        world.take_chunk_changes();
        interactor.process_mouse_input(MouseButton::Right, ElementState::Pressed);

        let camera = Camera {
            position: Vec3::new(8.5, 8.5, CHUNK_EDGE as f32 + 4.0),
            yaw: 0.0,
            pitch: 0.0,
        };
        interactor.update(&camera, &mut world);

        assert_eq!(world.voxel([8, 8, CHUNK_EDGE as i32 - 1]), Some(1));
        assert_eq!(world.voxel([8, 8, CHUNK_EDGE as i32]), None);
        assert!(world.take_chunk_changes().updated.is_empty());
    }

    #[test]
    fn idle_and_unhit_updates_change_nothing() {
        let mut interactor = WorldInteractor::default();
        let mut world = world_with_target_voxel();

        interactor.update(&camera_at_target(), &mut world);
        assert!(world.take_chunk_changes().updated.is_empty());
        assert_eq!(
            world
                .chunk(ChunkCoord { x: 0, y: 0, z: 0 })
                .unwrap()
                .voxel([8, 8, 8]),
            Some(1)
        );

        // Looking away from the voxel (upwards from above it) finds no hit.
        interactor.process_mouse_input(MouseButton::Left, ElementState::Pressed);
        let camera_looking_away = Camera {
            position: Vec3::new(8.5, CHUNK_EDGE as f32 + 4.0, 8.5),
            yaw: 0.0,
            pitch: std::f32::consts::FRAC_PI_2,
        };
        interactor.update(&camera_looking_away, &mut world);
        assert_eq!(interactor.action, Action::None);
        assert!(world.take_chunk_changes().updated.is_empty());
        assert_eq!(
            world
                .chunk(ChunkCoord { x: 0, y: 0, z: 0 })
                .unwrap()
                .voxel([8, 8, 8]),
            Some(1)
        );
    }
}
