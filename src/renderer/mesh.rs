use std::sync::Arc;

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

pub struct Material {
    pub ambient: Vec3,
    pub diffuse: Vec3,
    pub specular: Vec3,
    pub shininess: f32,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            ambient: Vec3::ONE,
            diffuse: Vec3::ONE,
            specular: Vec3::ONE,
            shininess: 32.0,
        }
    }
}

pub struct GameObject {
    // it's string for convenience
    pub mesh_id: String,

    pub transform: Transform,
    pub material: Material,
}

impl GameObject {
    pub fn new(mesh_id: &str, transform: Transform, material: Material) -> GameObject {
        GameObject {
            mesh_id: mesh_id.to_string(),
            transform,
            material,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub material_ambient: [f32; 3],
    pub material_diffuse: [f32; 3],
    pub material_specular: [f32; 3],
    pub material_shininess: f32,
}

vulkano::impl_vertex!(
    InstanceData,
    model,
    material_ambient,
    material_diffuse,
    material_specular,
    material_shininess
);

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
