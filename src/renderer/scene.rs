use std::{collections::HashMap, sync::Arc};

use glam::Mat4;
use gltf::Semantic;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::Queue,
    pipeline::GraphicsPipeline,
    sampler::Sampler,
    sync::GpuFuture,
};

use crate::renderer::texture::Texture;

use super::{
    context::Context,
    mesh::{GameObject, Mesh},
    shaders::SceneUniformBufferObject,
    vertex::Vertex,
};

pub struct Scene {
    pub meshes: HashMap<String, Mesh>,
    pub game_objects: Vec<GameObject>,
}

impl Scene {
    pub fn initialize(
        context: &Context,
        mesh_paths: Vec<&str>,
        graphics_pipeline: &Arc<GraphicsPipeline>,
    ) -> Self {
        let mut meshes = HashMap::new();

        for path in mesh_paths {
            let meshes_vec = Self::load_gltf(context, path, graphics_pipeline);

            for mesh in meshes_vec {
                meshes.insert(mesh.id.clone(), mesh);
            }
        }

        for mesh in meshes.values() {
            println!("Loaded {} mesh", mesh.id);
        }

        Scene {
            meshes,
            game_objects: vec![],
        }
    }

    pub fn add_game_object(&mut self, game_object: GameObject) -> &mut Self {
        self.game_objects.push(game_object);

        self
    }

    fn load_gltf(
        context: &Context,
        path: &str,
        graphics_pipeline: &Arc<GraphicsPipeline>,
    ) -> Vec<Mesh> {
        let mut meshes = Vec::new();

        let (document, buffers, _images) = gltf::import(path).unwrap();

        // let transforms = document
        //     .nodes()
        //     .map(|node| {
        //         let (translation, rotation, scale) = node.transform().decomposed();

        //         Transform {
        //             scale: Vec3::new(scale[0], scale[1], scale[2]),
        //             rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        //             translation: Vec3::new(translation[0], translation[1], translation[2]),
        //         }
        //     })
        //     .collect::<Vec<_>>();

        for node in document.nodes() {
            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let pbr = primitive.material().pbr_metallic_roughness();

                    // let base_color_factor = pbr.base_color_factor();

                    let (texture, tex_coord) = pbr
                        .base_color_texture()
                        .map(|color_info| {
                            let texture = Texture::from_gltf_texture(
                                &context,
                                path,
                                &color_info.texture(),
                                &buffers,
                            );

                            (texture, color_info.tex_coord())
                        })
                        .unwrap_or_else(|| {
                            // just a fillter image to make descriptor set happy
                            (Texture::empty(&context), 0)
                        });

                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    if let Some(_accessor) = primitive.get(&Semantic::Positions) {
                        let positions =
                            &reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();
                        let normals = &reader
                            .read_normals()
                            .map_or(vec![], |normals| normals.collect());

                        let color = &reader
                            .read_colors(0)
                            .map_or(vec![], |colors| colors.into_rgba_f32().collect());

                        // TODO: what gltf has more than one uv channel?
                        let tex_coords_0 = &reader
                            .read_tex_coords(0)
                            .map_or(vec![], |coords| coords.into_f32().collect());

                        let tex_coords_1 = &reader
                            .read_tex_coords(1)
                            .map_or(vec![], |coords| coords.into_f32().collect());

                        let vertices = positions
                            .iter()
                            .enumerate()
                            .map(|(index, position)| {
                                let position = *position;
                                let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                                let tex_coords_0 = *tex_coords_0.get(index).unwrap_or(&[0.0, 0.0]);
                                let tex_coords_1 = *tex_coords_1.get(index).unwrap_or(&[0.0, 0.0]);

                                let color = *color.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);

                                let uv = [
                                    [tex_coords_0[0], tex_coords_0[1]],
                                    [tex_coords_1[0], tex_coords_1[1]],
                                ][tex_coord as usize];

                                Vertex::new(position, normal, uv, color)
                            })
                            .collect::<Vec<_>>();

                        let indices = reader
                            .read_indices()
                            .map(|indices| indices.into_u32().collect::<Vec<_>>())
                            .unwrap();

                        let vertex_buffer =
                            Self::create_vertex_buffer(&context.graphics_queue, &vertices);
                        let index_buffer =
                            Self::create_index_buffer(&context.graphics_queue, &indices);

                        let uniform_buffer = Self::create_uniform_buffer(&context);

                        let descriptor_set = Self::create_descriptor_set(
                            &graphics_pipeline,
                            &uniform_buffer,
                            &texture,
                            &context.image_sampler,
                        );

                        let mesh = Mesh {
                            id: node.name().unwrap().to_string(),
                            vertex_buffer,
                            index_buffer,
                            index_count: indices.len() as u32,
                            uniform_buffer,
                            descriptor_set,
                        };

                        meshes.push(mesh);
                    }
                }
            }
        }

        meshes
    }

    fn create_vertex_buffer(
        graphics_queue: &Arc<Queue>,
        vertices: &Vec<Vertex>,
    ) -> Arc<ImmutableBuffer<[Vertex]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            BufferUsage::vertex_buffer(),
            // TODO: idealy it should be transfer queue?
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_index_buffer(
        graphics_queue: &Arc<Queue>,
        indices: &Vec<u32>,
    ) -> Arc<ImmutableBuffer<[u32]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            BufferUsage::index_buffer(),
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<SceneUniformBufferObject>> {
        let identity = Mat4::IDENTITY.to_cols_array_2d();

        let uniform_buffer_data = SceneUniformBufferObject {
            view: identity,
            proj: identity,
        };

        let buffer = CpuAccessibleBuffer::from_data(
            context.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer_data,
        )
        .unwrap();

        buffer
    }

    fn create_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<SceneUniformBufferObject>>,
        texture: &Texture,
        image_sampler: &Arc<Sampler>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        set_builder
            .add_sampled_image(texture.image.clone(), image_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }
}
