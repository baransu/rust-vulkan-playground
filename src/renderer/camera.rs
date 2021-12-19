use glam::{EulerRot, Mat3, Mat4, Quat, Vec3};

use super::shaders::CameraUniformBufferObject;

pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Camera {
    pub fn get_look_at_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), self.up())
    }

    pub fn forward(&self) -> Vec3 {
        (self.rotation * -Vec3::Z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        (self.rotation * Vec3::X).normalize()
    }

    pub fn up(&self) -> Vec3 {
        (self.rotation * Vec3::Y).normalize()
    }

    pub fn get_skybox_uniform_data(&self, dimensions: [f32; 2]) -> CameraUniformBufferObject {
        // NOTE: this strips translation from the view matrix which makes
        // the skybox static and around scene no matter what the camera is doing
        let view = Mat4::from_mat3(Mat3::from_mat4(self.get_look_at_matrix()));

        let proj = self.get_projection(dimensions);

        CameraUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: self.position.to_array(),
        }
    }

    pub fn get_projection(&self, dimensions: [f32; 2]) -> Mat4 {
        let mut proj = Mat4::perspective_rh(
            (45.0_f32).to_radians(),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            500.0,
        );

        proj.y_axis.y *= -1.0;

        proj
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::new(1.20, 0.0, 0.0),
            rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 90.0_f32.to_radians(), 0.0),
        }
    }
}
