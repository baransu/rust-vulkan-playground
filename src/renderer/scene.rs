use std::{collections::HashMap, sync::Arc};

use glam::{Mat4, Vec3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet},
    pipeline::{GraphicsPipeline, Pipeline},
};

use super::{
    camera::Camera,
    context::Context,
    entity::{Entity, InstanceData},
    light_system::{LightUniformBufferObject, ShaderDirectionalLight, ShaderPointLight},
    model::Model,
    shaders::{CameraUniformBufferObject, LightSpaceUniformBufferObject},
};

const MAX_POINT_LIGHTS: usize = 32;

#[derive(Clone, Copy)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
}

pub struct Scene {
    pub camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub camera_descriptor_set: Arc<PersistentDescriptorSet>,
    pub light_uniform_buffer: Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,
    pub light_space_uniform_buffer: Arc<CpuAccessibleBuffer<LightSpaceUniformBufferObject>>,

    pub shadow_descriptor_set: Arc<PersistentDescriptorSet>,

    pub point_lights: Vec<PointLight>,

    pub entities: Vec<Entity>,
    pub models: Vec<Model>,
}

impl Scene {
    pub fn initialize(
        context: &Context,
        mesh_paths: Vec<&str>,
        layout: &Arc<DescriptorSetLayout>,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        shadow_graphics_pipeline: &Arc<GraphicsPipeline>,
    ) -> Self {
        let point_lights = Self::gen_point_lights();

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);
        let light_uniform_buffer = Self::create_light_uniform_buffer(context, &point_lights);
        let light_space_uniform_buffer = Self::create_light_space_uniform_buffer(context);

        let queue = context.graphics_queue.clone();
        let models = mesh_paths
            .into_iter()
            .map(|path| Model::load_gltf(&queue, &layout, &path))
            .collect();

        let shadow_descriptor_set = Self::create_shadow_descriptor_set(
            &shadow_graphics_pipeline,
            &light_space_uniform_buffer,
        );

        let camera_descriptor_set =
            Self::create_camera_descriptor_set(graphics_pipeline, &camera_uniform_buffer);

        Scene {
            models,
            entities: vec![],

            camera_uniform_buffer,
            camera_descriptor_set,
            light_uniform_buffer,
            light_space_uniform_buffer,

            shadow_descriptor_set,

            point_lights,
        }
    }

    pub fn add_entity(&mut self, entity: Entity) -> &mut Self {
        self.entities.push(entity);

        self
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

        set_builder.build().unwrap()
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

    fn create_camera_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
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

        set_builder.build().unwrap()
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
            color: Vec3::new(0.0, 0.0, 100.0),
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

    pub fn get_instance_data_buffers(
        &self,
        context: &Context,
    ) -> HashMap<String, Arc<CpuAccessibleBuffer<[InstanceData]>>> {
        let mut instance_data: HashMap<String, Vec<InstanceData>> = HashMap::new();

        for entity in self.entities.iter() {
            let model = entity.transform.get_model_matrix();

            let instances = instance_data
                .entry(entity.model_id.clone())
                .or_insert(Vec::new());

            (*instances).push(InstanceData {
                model: model.to_cols_array_2d(),
            });
        }

        // TODO: we should create one buffer that we'll update with the data from the scene
        let mut instance_data_buffers: HashMap<String, Arc<CpuAccessibleBuffer<[InstanceData]>>> =
            HashMap::new();

        instance_data.iter().for_each(|(mesh_id, instances)| {
            let buffer = CpuAccessibleBuffer::from_iter(
                context.device.clone(),
                BufferUsage::all(),
                false,
                instances.iter().cloned(),
            )
            .unwrap();

            instance_data_buffers.insert(mesh_id.clone(), buffer);
        });

        instance_data_buffers
    }
}
