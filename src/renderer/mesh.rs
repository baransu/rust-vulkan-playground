use std::{str::FromStr, sync::Arc};

use glam::{Mat4, Quat, Vec3};
use vulkano::{buffer::ImmutableBuffer, descriptor_set::PersistentDescriptorSet};

use super::vertex::Vertex;

pub struct Mesh {
    pub id: String,
    pub index_count: u32,
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,
    pub descriptor_set: Arc<PersistentDescriptorSet>,
}

pub struct GameObject {
    // it's string for convenience
    pub mesh_id: String,

    pub transform: Transform,
}

impl GameObject {
    pub fn new(mesh_id: &str, transform: Transform) -> GameObject {
        GameObject {
            mesh_id: String::from_str(mesh_id).unwrap(),
            transform,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
}

vulkano::impl_vertex!(InstanceData, model);

#[derive(Clone)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn from_translation(translation: Vec3) -> Transform {
        Transform {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn get_model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}
