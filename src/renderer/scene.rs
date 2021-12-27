use std::{collections::HashMap, sync::Arc};

use glam::{Mat4, Vec3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        layout::DescriptorSetLayout, DescriptorSetsCollection, PersistentDescriptorSet,
    },
    pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint},
    sync::GpuFuture,
};

use super::{
    camera::Camera,
    context::Context,
    dir_light_shadows::DirLightShadows,
    entity::{Entity, InstanceData},
    light_system::{LightUniformBufferObject, ShaderPointLight},
    model::{Material, Model},
    shaders::CameraUniformBufferObject,
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

    pub point_lights: Vec<PointLight>,

    pub entities: Vec<Entity>,
    pub models: Vec<Model>,
}

impl Scene {
    pub fn initialize(
        context: &Context,
        mesh_paths: Vec<&str>,
        camera_layout: &Arc<DescriptorSetLayout>,
        materials_layout: &Arc<DescriptorSetLayout>,
    ) -> Self {
        let point_lights = Self::gen_point_lights();

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);
        let light_uniform_buffer = Self::create_light_uniform_buffer(context, &point_lights);

        let queue = context.graphics_queue.clone();
        let models = mesh_paths
            .into_iter()
            .map(|path| Model::load_gltf(&queue, &path, materials_layout))
            .collect();

        let camera_descriptor_set =
            Self::create_camera_descriptor_set(camera_layout, &camera_uniform_buffer);

        Scene {
            models,
            entities: vec![],

            camera_uniform_buffer,
            camera_descriptor_set,
            light_uniform_buffer,

            point_lights,
        }
    }

    pub fn add_entity(&mut self, entity: Entity) -> &mut Self {
        self.entities.push(entity);

        self
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
        layout: &Arc<DescriptorSetLayout>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    ) -> Arc<PersistentDescriptorSet> {
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(camera_uniform_buffer.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    fn gen_point_lights() -> Vec<PointLight> {
        vec![PointLight {
            position: Vec3::new(1.0, 4.0, -1.0),
            color: Vec3::new(10.0, 10.0, 10.0),
        }]
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

        let matrix = DirLightShadows::light_space_matrix();

        let buffer_data = LightUniformBufferObject {
            dir_light_space_matrix: matrix.to_cols_array_2d(),
            point_lights: shader_point_lights,
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
    ) -> HashMap<String, Arc<ImmutableBuffer<[InstanceData]>>> {
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

        // TODO: we should create one buffer that we'll update with the data from the scene if new entity is added
        let mut instance_data_buffers: HashMap<String, Arc<ImmutableBuffer<[InstanceData]>>> =
            HashMap::new();

        instance_data.iter().for_each(|(mesh_id, instances)| {
            let (buffer, future) = ImmutableBuffer::from_iter(
                instances.iter().cloned(),
                BufferUsage::all(),
                context.graphics_queue.clone(),
            )
            .unwrap();

            future
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            instance_data_buffers.insert(mesh_id.clone(), buffer);
        });

        instance_data_buffers
    }

    pub fn draw<S>(
        &self,
        context: &Context,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        pipeline: &Arc<GraphicsPipeline>,
        get_descriptor_sets: impl Fn(&Material) -> S,
    ) where
        S: DescriptorSetsCollection,
    {
        let instance_data_buffers = self.get_instance_data_buffers(&context);

        for model in self.models.iter() {
            // if there is no instance_data_buffer it means we have 0 instances for this mesh
            if let Some(instance_data_buffer) = instance_data_buffers.get(&model.id) {
                for node_index in model.root_nodes.iter() {
                    self.draw_model_node(
                        model,
                        &pipeline,
                        builder,
                        *node_index,
                        instance_data_buffer,
                        &get_descriptor_sets,
                    );
                }
            }
        }
    }

    fn draw_model_node<S>(
        &self,
        model: &Model,
        pipeline: &Arc<GraphicsPipeline>,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        node_index: usize,
        instance_data_buffer: &Arc<ImmutableBuffer<[InstanceData]>>,
        get_descriptor_sets: &impl Fn(&Material) -> S,
    ) where
        S: DescriptorSetsCollection,
    {
        let node = model.nodes.get(node_index).unwrap();

        if let Some(mesh_index) = node.mesh {
            let mesh = model.meshes.get(mesh_index).unwrap();

            for primitive_index in mesh.primitives.iter() {
                self.draw_model_primitive(
                    pipeline,
                    model,
                    builder,
                    *primitive_index,
                    instance_data_buffer,
                    get_descriptor_sets,
                );
            }
        }

        for child_index in node.children.iter() {
            self.draw_model_node(
                model,
                pipeline,
                builder,
                *child_index,
                instance_data_buffer,
                get_descriptor_sets,
            );
        }
    }

    fn draw_model_primitive<S>(
        &self,
        pipeline: &Arc<GraphicsPipeline>,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        primitive_index: usize,
        instance_data_buffer: &Arc<ImmutableBuffer<[InstanceData]>>,
        get_descriptor_sets: &impl Fn(&Material) -> S,
    ) where
        S: DescriptorSetsCollection,
    {
        let primitive = model.primitives.get(primitive_index).unwrap();
        let material = model.materials.get(primitive.material).unwrap();

        let descriptor_sets = get_descriptor_sets(material);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                descriptor_sets,
            )
            .bind_vertex_buffers(
                0,
                (
                    primitive.vertex_buffer.clone(),
                    instance_data_buffer.clone(),
                ),
            )
            .bind_index_buffer(primitive.index_buffer.clone())
            .draw_indexed(
                primitive.index_count,
                instance_data_buffer.len() as u32,
                0,
                0,
                0,
            )
            .unwrap();
    }
}
