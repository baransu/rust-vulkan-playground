use glam::{Mat3, Mat4, Vec3};

use super::shaders::SceneUniformBufferObject;

pub struct Camera {
    pub position: Vec3,
    pub rotation: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl Camera {
    pub fn get_look_at_matrix(&self) -> Mat4 {
        // let roll = self.rotation.x.to_radians();
        let pitch = self.rotation.y.to_radians();
        let yaw = self.rotation.z.to_radians();

        let mut forward = Vec3::new(self.forward.x, self.forward.y, self.forward.z);
        forward.x = pitch.cos() * yaw.cos();
        forward.y = pitch.sin();
        forward.z = pitch.cos() * yaw.sin();
        forward = forward.normalize();

        let up = Vec3::cross(self.right, self.forward).normalize();

        Mat4::look_at_rh(self.position, self.position + forward, up)
    }

    pub fn get_skybox_uniform_data(&self, dimensions: [f32; 2]) -> SceneUniformBufferObject {
        // NOTE: this strips translation from the view matrix which makes
        // the skybox static and around scene no matter what the camera is doing
        let view = Mat4::from_mat3(Mat3::from_mat4(self.get_look_at_matrix()));

        let proj = Self::get_projection(dimensions);

        SceneUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
        }
    }

    pub fn get_model_uniform_data(&self, dimensions: [f32; 2]) -> SceneUniformBufferObject {
        let view = self.get_look_at_matrix();

        let proj = Self::get_projection(dimensions);

        SceneUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
        }
    }

    fn get_projection(dimensions: [f32; 2]) -> Mat4 {
        let mut proj = Mat4::perspective_rh(
            (45.0_f32).to_radians(),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0,
        );

        proj.y_axis.y *= -1.0;

        proj
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::new(0.0, 0.0, 20.0),
            rotation: Vec3::new(0.0, 0.0, -90.0),
            forward: Vec3::new(0.0, 0.0, -1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        }
    }
}
