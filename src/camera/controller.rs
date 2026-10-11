use std::collections::HashSet;
use std::time::Duration;

use glam::Vec3;
use winit::{
    event::KeyEvent,
    keyboard::{KeyCode, PhysicalKey},
};

use crate::camera::Camera;

const MOUSE_SENSITIVITY: f32 = 0.002;
const NORMAL_SPEED: f32 = 5.0;
const FAST_SPEED: f32 = 50.0;
const MAX_PITCH: f32 = 1.5533;

#[derive(Default)]
pub struct CameraController {
    pressed_keys: HashSet<KeyCode>,
    mouse_delta: (f64, f64),
}

impl CameraController {
    pub fn process_key(&mut self, event: &KeyEvent) {
        let PhysicalKey::Code(key) = event.physical_key else {
            return;
        };

        if event.state.is_pressed() {
            self.pressed_keys.insert(key);
        } else {
            self.pressed_keys.remove(&key);
        }
    }

    pub fn process_mouse_motion(&mut self, delta: (f64, f64)) {
        self.mouse_delta.0 += delta.0;
        self.mouse_delta.1 += delta.1;
    }

    pub fn clear_input(&mut self) {
        self.pressed_keys.clear();
        self.mouse_delta = (0.0, 0.0);
    }

    pub fn update(&mut self, camera: &mut Camera, dt: Duration) {
        let (mouse_x, mouse_y) = std::mem::take(&mut self.mouse_delta);
        camera.yaw += mouse_x as f32 * MOUSE_SENSITIVITY;
        camera.pitch =
            (camera.pitch - mouse_y as f32 * MOUSE_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);

        // Cancel out up and down direction when moving forward/backward
        let mut forward = camera.forward();
        forward.y = 0.0;
        forward = forward.normalize();

        let right = forward.cross(Vec3::Y).normalize();
        let mut movement = Vec3::ZERO;
        if self.pressed_keys.contains(&KeyCode::KeyW) {
            movement += forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyS) {
            movement -= forward;
        }
        if self.pressed_keys.contains(&KeyCode::KeyD) {
            movement += right;
        }
        if self.pressed_keys.contains(&KeyCode::KeyA) {
            movement -= right;
        }
        if self.pressed_keys.contains(&KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.pressed_keys.contains(&KeyCode::ControlLeft) {
            movement -= Vec3::Y;
        }

        if movement != Vec3::ZERO {
            let speed = if self.pressed_keys.contains(&KeyCode::ShiftLeft) {
                FAST_SPEED
            } else {
                NORMAL_SPEED
            };
            camera.position += movement.normalize() * speed * dt.as_secs_f32();
        }
    }
}
