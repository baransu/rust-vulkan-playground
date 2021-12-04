use glam::Vec3;

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
