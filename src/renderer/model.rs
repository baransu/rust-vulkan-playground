use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use vulkano::{
    buffer::{CpuAccessibleBuffer, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
};

use super::{camera::Camera, shaders::MVPUniformBufferObject, vertex::Vertex};

pub struct Model {
    pub index_count: u32,
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<MVPUniformBufferObject>>,
    pub descriptor_set: Arc<PersistentDescriptorSet>,

    pub transform: Transform,
}

#[derive(Clone)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn get_mvp_ubo(&self, camera: &Camera, dimensions: [f32; 2]) -> MVPUniformBufferObject {
        let view = Mat4::look_at_rh(camera.position(), camera.target(), Vec3::Y);

        let mut proj = Mat4::perspective_rh(
            (45.0_f32).to_radians(),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0,
        );

        proj.y_axis.y *= -1.0;

        // this is needed to fix model rotation
        let model = Mat4::from_rotation_x((90.0_f32).to_radians())
            * Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation);

        MVPUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        }
    }
}
