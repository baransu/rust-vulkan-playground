use std::sync::Arc;

use glam::{Quat, Vec3};
use vulkano::{
    buffer::{CpuAccessibleBuffer, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
};

use super::{shaders::MVPUniformBufferObject, vertex::Vertex};

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
