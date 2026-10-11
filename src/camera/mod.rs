pub mod controller;

use glam::{Mat4, Vec3};

const WORLD_UP: Vec3 = Vec3::Y;
const FIELD_OF_VIEW: f32 = 60.0_f32.to_radians();
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 500.0;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 2.0, 8.0),
            yaw: 0.0,
            pitch: -0.245,
        }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn view_projection(&self, aspect_ratio: f32) -> Mat4 {
        let forward = self.forward();
        let right = forward.cross(WORLD_UP).normalize();
        let up = right.cross(forward).normalize();
        let camera_world = Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            (-forward).extend(0.0),
            self.position.extend(1.0),
        );
        let view = camera_world.inverse();

        let focal_length = 1.0 / (FIELD_OF_VIEW * 0.5).tan();
        let projection = Mat4::from_cols(
            glam::Vec4::new(focal_length / aspect_ratio, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, focal_length, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, FAR_PLANE / (NEAR_PLANE - FAR_PLANE), -1.0),
            glam::Vec4::new(
                0.0,
                0.0,
                NEAR_PLANE * FAR_PLANE / (NEAR_PLANE - FAR_PLANE),
                0.0,
            ),
        );
        projection * view
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec4;

    use super::Camera;

    #[test]
    fn default_camera_projects_scene_center_to_view_center() {
        let camera = Camera::default();
        let clip = camera.view_projection(16.0 / 9.0) * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let ndc = clip.truncate() / clip.w;

        assert!(clip.w > 0.0);
        assert!(ndc.x.abs() < 0.001);
        assert!(ndc.y.abs() < 0.01);
        assert!((0.0..=1.0).contains(&ndc.z));
    }
}
