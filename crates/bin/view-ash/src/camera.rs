use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,

    near: f32,
    far: f32,
    fov_radians: f32,
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        (self.rotation * -Vec3::Z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        (self.rotation * Vec3::X).normalize()
    }

    pub fn up(&self) -> Vec3 {
        (self.rotation * Vec3::Y).normalize()
    }

    pub fn look_at_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), self.up())
    }

    pub fn projection(&self, aspect_ratio: f32) -> Mat4 {
        let mut proj = Mat4::perspective_rh(self.fov_radians, aspect_ratio, self.near, self.far);

        proj.y_axis.y *= -1.0;

        proj
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: Vec3::new(0.0, 0.0, 5.0),
            rotation: Quat::IDENTITY,
            near: 0.1,
            far: 250.0,
            fov_radians: 45.0_f32.to_radians(),
        }
    }
}
