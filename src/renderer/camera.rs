use glam::{Mat4, Vec3};

use super::shaders::MVPUniformBufferObject;

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

    pub fn get_mvp_ubo(&self, dimensions: [f32; 2]) -> MVPUniformBufferObject {
        let view = Mat4::look_at_rh(self.position(), self.target(), Vec3::Y);

        let mut proj = Mat4::perspective_rh(
            (45.0_f32).to_radians(),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0,
        );

        proj.y_axis.y *= -1.0;

        MVPUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            model: Mat4::IDENTITY.to_cols_array_2d(),
        }
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
