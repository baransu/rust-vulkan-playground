use glam::{Mat3, Mat4, Vec3};

use super::shaders::SceneUniformBufferObject;

pub struct Camera {
    pub theta: f32,
    pub phi: f32,
    pub r: f32,
    target: Vec3,
}

impl Camera {
    pub fn position(&self) -> Vec3 {
        Vec3::new(
            self.target[0] + self.r * self.phi.sin() * self.theta.sin(),
            self.target[1] + self.r * self.phi.cos(),
            self.target[2] + self.r * self.phi.sin() * self.theta.cos(),
        )
    }

    pub fn target(&self) -> Vec3 {
        self.target
    }

    pub fn get_skybox_uniform_data(&self, dimensions: [f32; 2]) -> SceneUniformBufferObject {
        // NOTE: this strips translation from the view matrix which makes
        // the skybox static and around scene no matter what the camera is doing
        let view = Mat4::from_mat3(Mat3::from_mat4(Mat4::look_at_rh(
            self.position(),
            self.target(),
            Vec3::Y,
        )));

        let proj = Self::get_projection(dimensions);

        SceneUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
        }
    }

    pub fn get_model_uniform_data(&self, dimensions: [f32; 2]) -> SceneUniformBufferObject {
        let view = Mat4::look_at_rh(self.position(), self.target(), Vec3::Y);

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
            theta: 0.0_f32.to_radians(),
            phi: 90.0_f32.to_radians(),
            r: 5.0,
            target: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}
