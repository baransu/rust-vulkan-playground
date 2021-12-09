use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use vulkano::{
    buffer::{CpuAccessibleBuffer, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
};

use super::{shaders::SceneUniformBufferObject, vertex::Vertex};

pub struct Model {
    pub index_count: u32,
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<SceneUniformBufferObject>>,
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
    pub fn get_model_matrix(&self, translation: Vec3) -> Mat4 {
        // this is needed to fix model rotation
        Mat4::from_rotation_x((90.0_f32).to_radians())
            * Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
            * Mat4::from_translation(translation)
    }
}
