use std::{str::FromStr, sync::Arc};

use glam::{Mat4, Quat, Vec3};
use vulkano::{
    buffer::{CpuAccessibleBuffer, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
};

use super::{shaders::SceneUniformBufferObject, vertex::Vertex};

pub struct Mesh {
    pub id: String,
    pub index_count: u32,
    pub vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    pub index_buffer: Arc<ImmutableBuffer<[u32]>>,

    // TODO: this should be per scene not per mesh
    pub uniform_buffer: Arc<CpuAccessibleBuffer<SceneUniformBufferObject>>,
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
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn get_model_matrix(&self) -> Mat4 {
        // this is needed to fix model rotation
        Mat4::from_rotation_x((90.0_f32).to_radians())
            * Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}
