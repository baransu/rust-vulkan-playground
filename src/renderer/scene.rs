use std::{collections::HashMap, convert::TryInto, sync::Arc};

use glam::{Mat4, Vec3};
use gltf::Semantic;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::Queue,
    pipeline::GraphicsPipeline,
    sync::GpuFuture,
};

use crate::{renderer::texture::Texture, FramebufferWithAttachment};

use super::{
    camera::Camera,
    context::Context,
    gbuffer::GBuffer,
    light_system::{DirectionalLight, LightUniformBufferObject, PointLight},
    mesh::{GameObject, Mesh},
    shaders::{CameraUniformBufferObject, LightSpaceUniformBufferObject},
    vertex::Vertex,
};

const NR_POINT_LIGHTS: usize = 4;
const MAX_POINT_LIGHTS: usize = 32;

pub struct Scene {
    pub meshes: HashMap<String, Mesh>,
    pub game_objects: Vec<GameObject>,

    camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    light_uniform_buffer: Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,

    pub shadow_descriptor_set: Arc<PersistentDescriptorSet>,
    pub light_descriptor_set: Arc<PersistentDescriptorSet>,
}

impl Scene {
    pub fn initialize(
        context: &Context,
        mesh_paths: Vec<&str>,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        shadow_graphics_pipeline: &Arc<GraphicsPipeline>,
        light_graphics_pipeline: &Arc<GraphicsPipeline>,
        shadow_framebuffer: &FramebufferWithAttachment,
        gbuffer: &GBuffer,
    ) -> Self {
        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);
        let light_uniform_buffer = Self::create_light_uniform_buffer(context);
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

        let light_descriptor_set = Self::create_light_descriptor_set(
            context,
            &light_graphics_pipeline,
            &camera_uniform_buffer,
            &gbuffer,
            &light_uniform_buffer,
            &light_space_uniform_buffer,
            &shadow_framebuffer,
        );

        Scene {
            meshes,
            game_objects: vec![],
            camera_uniform_buffer,
            light_uniform_buffer,

            shadow_descriptor_set,
            light_descriptor_set,
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

                        let descriptor_set = Self::create_descriptor_set(
                            context,
                            &graphics_pipeline,
                            &camera_uniform_buffer,
                            &texture,
                        );

                        let mesh = Mesh {
                            id: node.name().unwrap().to_string(),
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
        texture: &Texture,
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
            .add_sampled_image(texture.image.clone(), context.image_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_light_descriptor_set(
        context: &Context,
        light_graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        gbuffer: &GBuffer,
        light_uniform_buffer: &Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,
        light_space_uniform_buffer: &Arc<CpuAccessibleBuffer<LightSpaceUniformBufferObject>>,
        shadow_framebuffer: &FramebufferWithAttachment,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = light_graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(camera_uniform_buffer.clone())
            .unwrap();

        set_builder
            .add_image(gbuffer.position_buffer.clone())
            .unwrap();

        set_builder
            .add_image(gbuffer.normals_buffer.clone())
            .unwrap();

        set_builder
            .add_image(gbuffer.albedo_buffer.clone())
            .unwrap();

        set_builder
            .add_sampled_image(
                shadow_framebuffer.attachment.clone(),
                context.depth_sampler.clone(),
            )
            .unwrap();

        set_builder
            .add_buffer(light_space_uniform_buffer.clone())
            .unwrap();

        set_builder
            .add_buffer(light_uniform_buffer.clone())
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

    pub fn light_positions() -> [Vec3; NR_POINT_LIGHTS] {
        [
            Vec3::new(0.7, 5.0, 2.0),
            Vec3::new(2.3, 5.0, -4.0),
            Vec3::new(-4.0, 5.0, -12.0),
            Vec3::new(0.0, 5.0, -3.0),
        ]
    }

    pub fn light_colors() -> [Vec3; NR_POINT_LIGHTS] {
        [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        ]
    }

    pub fn dir_light_position() -> Vec3 {
        Vec3::new(-0.2, -1.0, -0.3) * -10.0
    }

    fn create_light_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<LightUniformBufferObject>> {
        let mut point_lights: [PointLight; MAX_POINT_LIGHTS] = [PointLight {
            position: Vec3::ZERO.to_array(),
            _dummy0: [0, 0, 0, 0],
            ambient: Vec3::ZERO.to_array(),
            _dummy1: [0, 0, 0, 0],
            diffuse: Vec3::ZERO.to_array(),
            _dummy2: [0, 0, 0, 0],
            specular: Vec3::ZERO.to_array(),
            _dummy3: [0, 0, 0, 0, 0, 0, 0, 0],
            constant_: 0.0,
            linear: 0.0,
            quadratic: 0.0,
        }; MAX_POINT_LIGHTS];

        let colors = Self::light_colors();

        for (index, position) in Self::light_positions().iter().enumerate() {
            let color = colors.get(index).unwrap().clone();

            point_lights[index] = PointLight {
                position: position.to_array(),
                _dummy0: [0, 0, 0, 0],
                ambient: (color * 0.1).to_array(),
                _dummy1: [0, 0, 0, 0],
                diffuse: color.to_array(),
                _dummy2: [0, 0, 0, 0],
                specular: color.to_array(),
                _dummy3: [0, 0, 0, 0, 0, 0, 0, 0],
                constant_: 1.0,
                linear: 0.09,
                quadratic: 0.032,
            }
        }

        let dir_light = DirectionalLight {
            direction: Vec3::new(-0.2, -1.0, -0.3).to_array(),
            _dummy0: [0, 0, 0, 0],
            ambient: Vec3::ZERO.to_array(),
            _dummy1: [0, 0, 0, 0],
            diffuse: (Vec3::ONE * 0.25).to_array(),
            _dummy2: [0, 0, 0, 0],
            specular: Vec3::ZERO.to_array(),
        };

        let buffer_data = LightUniformBufferObject {
            point_lights,
            _dummy0: [0, 0, 0, 0],
            dir_light,
            point_lights_count: NR_POINT_LIGHTS as i32,
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
