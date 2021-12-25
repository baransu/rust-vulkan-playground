use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    format::ClearValue,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        graphics::{vertex_input::BuffersDefinition, viewport::Viewport},
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use super::{
    context::Context, entity::InstanceData, model::Model, scene::Scene,
    shaders::CameraUniformBufferObject, vertex::Vertex,
};

const DIM: f32 = 1024.0;

pub struct DirLightShadows {
    pipeline: Arc<GraphicsPipeline>,
    camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub target_attachment: Arc<ImageView<AttachmentImage>>,

    framebuffer: Arc<Framebuffer>,

    camera_descriptor_set: Arc<PersistentDescriptorSet>,
}

impl DirLightShadows {
    pub fn initialize(context: &Context) -> DirLightShadows {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);

        let target_attachment = Self::create_depth_attachment(context);
        let framebuffer = Self::create_framebuffer(&render_pass, &target_attachment);

        let camera_descriptor_set =
            Self::create_camera_descriptor_set(&pipeline, &camera_uniform_buffer);

        DirLightShadows {
            camera_uniform_buffer,
            pipeline,
            target_attachment,

            framebuffer,
            camera_descriptor_set,
        }
    }

    fn create_camera_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<CameraUniformBufferObject>> {
        let identity = Mat4::IDENTITY.to_cols_array_2d();

        let uniform_buffer_data = CameraUniformBufferObject {
            view: identity,
            proj: identity,
            position: Vec3::ONE.to_array(),
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

    pub fn light_space() -> (Mat4, Mat4, Vec3, Vec3) {
        let position = Vec3::new(30.0, 42.0, -13.0);
        let direction = -Vec3::new(30.0, 42.0, -13.0).normalize();

        let mut proj = Mat4::orthographic_rh(-25.0, 25.0, 25.0, -25.0, 0.1, 250.0);

        proj.y_axis.y *= -1.0;

        let view = Mat4::look_at_rh(position, direction, Vec3::Y);

        (proj, view, position, direction)
    }

    fn update_uniform_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let (proj, view, position, _) = Self::light_space();

        // camera buffer
        let camera_buffer_data = Arc::new(CameraUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: position.to_array(),
        });

        builder
            .update_buffer(self.camera_uniform_buffer.clone(), camera_buffer_data)
            .unwrap();
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

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        target: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<Framebuffer> {
        Framebuffer::start(render_pass.clone())
            .add(target.clone())
            .unwrap()
            .build()
            .unwrap()
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        single_pass_renderpass!(context.device.clone(),
                attachments: {
                    depth: {
                        load: Clear,
                        store: Store,
                        format: context.depth_format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [],
                    depth_stencil: {depth}
                }
        )
        .unwrap()
    }

    pub fn add_to_builder(
        &self,
        context: &Context,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scene: &Scene,
    ) {
        let instance_data_buffers = scene.get_instance_data_buffers(&context);

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM as f32, DIM as f32],
            depth_range: 0.0..1.0,
        };

        let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        secondary_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, [viewport.clone()]);

        for model in scene.models.iter() {
            // if there is no instance_data_buffer it means we have 0 instances for this mesh
            if let Some(instance_data_buffer) = instance_data_buffers.get(&model.id) {
                self.draw_model(model, &mut secondary_builder, instance_data_buffer);
            }
        }

        let model_command_buffer = Arc::new(secondary_builder.build().unwrap());

        self.update_uniform_buffers(builder);

        builder
            .begin_render_pass(
                self.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![
                    // ClearValue::Float([1.0, 1.0, 1.0, 1.0]),
                    ClearValue::Depth(1.0),
                ],
            )
            .unwrap();

        builder
            .execute_commands(model_command_buffer.clone())
            .unwrap();

        builder.end_render_pass().unwrap();
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<InstanceData>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_simple_depth()
            .depth_clamp(false)
            .cull_mode_front()
            .front_face_clockwise()
            .viewports_dynamic_scissors_irrelevant(1)
            .blend_pass_through()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_depth_attachment(context: &Context) -> Arc<ImageView<AttachmentImage>> {
        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                [DIM as u32, DIM as u32],
                context.depth_format,
                ImageUsage {
                    sampled: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn draw_model(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        instance_data_buffer: &Arc<ImmutableBuffer<[InstanceData]>>,
    ) {
        for node_index in model.root_nodes.iter() {
            self.draw_model_node(model, builder, *node_index, instance_data_buffer);
        }
    }

    fn draw_model_node(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        node_index: usize,
        instance_data_buffer: &Arc<ImmutableBuffer<[InstanceData]>>,
    ) {
        let node = model.nodes.get(node_index).unwrap();

        if let Some(mesh_index) = node.mesh {
            let mesh = model.meshes.get(mesh_index).unwrap();

            for primitive_index in mesh.primitives.iter() {
                self.draw_model_primitive(model, builder, *primitive_index, instance_data_buffer);
            }
        }

        for child_index in node.children.iter() {
            self.draw_model_node(model, builder, *child_index, instance_data_buffer);
        }
    }

    fn draw_model_primitive(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        primitive_index: usize,
        instance_data_buffer: &Arc<ImmutableBuffer<[InstanceData]>>,
    ) {
        let primitive = model.primitives.get(primitive_index).unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.camera_descriptor_set.clone(),
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

pub mod vs {
    vulkano_shaders::shader! {
                                                                    ty: "vertex",
                                                                    src: "
			#version 450

            layout(binding = 0) uniform CameraUniformBufferObject {
                mat4 view;
                mat4 proj;
                vec3 position;
            } camera;

			// per vertex
			layout(location = 1) in vec3 position;

			// per instance
			layout(location = 4) in mat4 model; 

			void main() {
				gl_Position = camera.proj * camera.view * model * vec4(position, 1.0);
			}									
"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
                                    ty: "fragment",
                                    src: "
			#version 450

			void main() {
			}
"
    }
}
