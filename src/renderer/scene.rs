use std::{collections::HashMap, sync::Arc};

use glam::{Mat4, Vec3};
use gltf::Semantic;
use rand::Rng;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::Queue,
    pipeline::GraphicsPipeline,
    sync::GpuFuture,
};

use crate::renderer::texture::Texture;

use super::{
    camera::Camera,
    context::Context,
    light_system::{LightUniformBufferObject, ShaderDirectionalLight, ShaderPointLight},
    mesh::{GameObject, Mesh},
    shaders::{CameraUniformBufferObject, LightSpaceUniformBufferObject},
    vertex::Vertex,
};

const MAX_POINT_LIGHTS: usize = 32;

#[derive(Clone, Copy)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
}

pub struct Scene {
    pub meshes: HashMap<String, Mesh>,
    pub game_objects: Vec<GameObject>,

    pub camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub light_uniform_buffer: Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,
    pub light_space_uniform_buffer: Arc<CpuAccessibleBuffer<LightSpaceUniformBufferObject>>,

    pub shadow_descriptor_set: Arc<PersistentDescriptorSet>,

    pub point_lights: Vec<PointLight>,
}

impl Scene {
    pub fn initialize(
        context: &Context,
        mesh_paths: Vec<&str>,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        shadow_graphics_pipeline: &Arc<GraphicsPipeline>,
    ) -> Self {
        let point_lights = Self::gen_point_lights();

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);
        let light_uniform_buffer = Self::create_light_uniform_buffer(context, &point_lights);
        let light_space_uniform_buffer = Self::create_light_space_uniform_buffer(context);

        let mut meshes = HashMap::new();

        for path in mesh_paths {
            let meshes_vec =
                Self::load_gltf(context, path, graphics_pipeline, &camera_uniform_buffer);

            for mesh in meshes_vec {
                meshes.insert(mesh.id.clone(), mesh);
            }
        }

        for mesh in meshes.values() {
            println!("Loaded {} mesh", mesh.id);
        }

        let shadow_descriptor_set = Self::create_shadow_descriptor_set(
            &shadow_graphics_pipeline,
            &light_space_uniform_buffer,
        );

        Scene {
            meshes,
            game_objects: vec![],
            camera_uniform_buffer,
            light_uniform_buffer,
            light_space_uniform_buffer,

            shadow_descriptor_set,

            point_lights,
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
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    ) -> Vec<Mesh> {
        let mut meshes = Vec::new();

        println!("Loading {}", path);
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
                for (idx, primitive) in mesh.primitives().enumerate() {
                    let material = primitive.material();
                    let pbr = material.pbr_metallic_roughness();

                    // let base_color_factor = pbr.base_color_factor();

                    let diffuse_texture = pbr
                        .base_color_texture()
                        .map(|info| {
                            Texture::from_gltf_texture(&context, path, &info.texture(), &buffers)
                        })
                        .unwrap_or_else(|| {
                            // just a fillter image to make descriptor set happy
                            Texture::empty(&context)
                        });

                    let normal_texture = material
                        .normal_texture()
                        .map(|info| {
                            Texture::from_gltf_texture(&context, path, &info.texture(), &buffers)
                        })
                        .unwrap_or_else(|| {
                            // just a fillter image to make descriptor set happy
                            Texture::empty(&context)
                        });

                    let metalic_roughness_texture = pbr
                        .metallic_roughness_texture()
                        .map(|info| {
                            Texture::from_gltf_texture(&context, path, &info.texture(), &buffers)
                        })
                        .unwrap_or_else(|| {
                            // just a fillter image to make descriptor set happy
                            Texture::empty(&context)
                        });

                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    if let Some(_accessor) = primitive.get(&Semantic::Positions) {
                        let positions =
                            &reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();
                        let normals = &reader
                            .read_normals()
                            .map_or(vec![], |normals| normals.collect());

                        let tangents = &reader
                            .read_tangents()
                            .map_or(vec![], |tangets| tangets.collect());

                        // TODO: why gltf has more than one uv channel?
                        let tex_coords = &reader
                            .read_tex_coords(0)
                            .map_or(vec![], |coords| coords.into_f32().collect());

                        let vertices = positions
                            .iter()
                            .enumerate()
                            .map(|(index, position)| {
                                let position = *position;
                                let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                                let uv = *tex_coords.get(index).unwrap_or(&[0.0, 0.0]);
                                let tangent = *tangents.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);

                                Vertex::new(position, normal, uv, tangent)
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

                        let descriptor_set = Self::create_descriptor_set(
                            context,
                            &graphics_pipeline,
                            &camera_uniform_buffer,
                            &diffuse_texture,
                            &normal_texture,
                            &metalic_roughness_texture,
                        );

                        let mesh = Mesh {
                            id: node
                                .name()
                                .unwrap_or(format!("sponza-{}", idx).as_str())
                                .to_string(),
                            vertex_buffer,
                            index_buffer,
                            index_count: indices.len() as u32,
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

    fn create_descriptor_set(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        diffuse_texture: &Texture,
        normal_texture: &Texture,
        metalic_roughness_texture: &Texture,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(camera_uniform_buffer.clone())
            .unwrap();

        set_builder
            .add_sampled_image(diffuse_texture.image.clone(), context.image_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(normal_texture.image.clone(), context.image_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(
                metalic_roughness_texture.image.clone(),
                context.image_sampler.clone(),
            )
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_shadow_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        light_space_uniform_buffer: &Arc<CpuAccessibleBuffer<LightSpaceUniformBufferObject>>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(light_space_uniform_buffer.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_camera_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<CameraUniformBufferObject>> {
        let identity = Mat4::IDENTITY.to_cols_array_2d();

        let uniform_buffer_data = CameraUniformBufferObject {
            view: identity,
            proj: identity,
            position: Vec3::ZERO.to_array(),
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

    fn create_light_space_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<LightSpaceUniformBufferObject>> {
        // NOTE: vulkan has Y flipped from OpenGL so I guess that's why bottom/top has to be reversed
        let mut light_projection = Mat4::orthographic_rh(-25.0, 25.0, 25.0, -25.0, 0.1, 1000.0);

        light_projection.y_axis.y *= -1.0;

        let position = Self::dir_light_position();
        let direction = Vec3::new(-0.2, -1.0, -0.3);

        let light_view = Mat4::look_at_rh(position, direction, Vec3::Y);

        let matrix = light_projection * light_view;

        let uniform_buffer_data = LightSpaceUniformBufferObject {
            matrix: matrix.to_cols_array_2d(),
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

    fn gen_point_lights() -> Vec<PointLight> {
        vec![PointLight {
            position: Vec3::new(0.0, 5.0, 0.0),
            color: Vec3::new(1.0, 1.0, 1.0),
        }]
    }

    pub fn dir_light_position() -> Vec3 {
        Vec3::new(0.0, 10.0, 0.0) * 5.0
    }

    fn create_light_uniform_buffer(
        context: &Context,
        point_lights: &Vec<PointLight>,
    ) -> Arc<CpuAccessibleBuffer<LightUniformBufferObject>> {
        let mut shader_point_lights: [ShaderPointLight; MAX_POINT_LIGHTS] = [ShaderPointLight {
            color: Vec3::ZERO.to_array(),
            _dummy0: [0, 0, 0, 0],
            position: Vec3::ZERO.to_array(),
            _dummy1: [0, 0, 0, 0, 0, 0, 0, 0],
            constant_: 0.0,
            linear: 0.0,
            quadratic: 0.0,
        };
            MAX_POINT_LIGHTS];

        for (index, light) in point_lights.iter().enumerate() {
            shader_point_lights[index] = ShaderPointLight {
                position: light.position.to_array(),
                _dummy0: [0, 0, 0, 0],
                color: light.color.to_array(),
                _dummy1: [0, 0, 0, 0, 0, 0, 0, 0],
                constant_: 1.0,
                linear: 0.09,
                quadratic: 0.032,
            }
        }

        let dir_light = ShaderDirectionalLight {
            direction: Vec3::new(-0.2, -1.0, -0.3).to_array(),
            _dummy0: [0, 0, 0, 0],
            color: Vec3::ZERO.to_array(),
        };

        let buffer_data = LightUniformBufferObject {
            point_lights: shader_point_lights,
            _dummy0: [0, 0, 0, 0],
            dir_light,
            point_lights_count: point_lights.len() as i32,
        };

        let buffer = CpuAccessibleBuffer::from_data(
            context.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            buffer_data,
        )
        .unwrap();

        buffer
    }

    fn get_camera_uniform_buffer_data(
        &self,
        camera: &Camera,
        dimensions: [f32; 2],
    ) -> Arc<CameraUniformBufferObject> {
        let view = camera.get_look_at_matrix();
        let proj = camera.get_projection(dimensions);

        Arc::new(CameraUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: camera.position.to_array(),
        })
    }

    pub fn update_uniform_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        camera: &Camera,
        dimensions: [f32; 2],
    ) {
        builder
            .update_buffer(
                self.camera_uniform_buffer.clone(),
                self.get_camera_uniform_buffer_data(&camera, dimensions),
            )
            .unwrap();
    }
}
