use glam::{Mat3, Mat4, Vec3};

use super::shaders::SceneUniformBufferObject;

pub struct Camera {
    pub position: Vec3,
    pub rotation: Vec3,
}

impl Camera {
    pub fn get_look_at_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), self.up())
    }

    pub fn forward(&self) -> Vec3 {
        let pitch = self.rotation.y.to_radians();
        let yaw = self.rotation.z.to_radians();

        let x = pitch.cos() * yaw.cos();
        let y = pitch.sin();
        let z = pitch.cos() * yaw.sin();

        Vec3::new(x, y, z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        let forward = self.forward();
        Vec3::new(forward.z, forward.y, -forward.x).normalize()
    }

    pub fn up(&self) -> Vec3 {
        let forward = self.forward().normalize();
        let right = self.right();
        Vec3::cross(-right, forward).normalize()
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
        }
    }
}
