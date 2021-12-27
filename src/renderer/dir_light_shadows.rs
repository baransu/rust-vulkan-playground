use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    format::ClearValue,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use super::{context::Context, entity::InstanceData, scene::Scene, vertex::Vertex};

const DIM: f32 = 1024.0;

pub struct DirLightShadows {
    pipeline: Arc<GraphicsPipeline>,
    camera_uniform_buffer: Arc<CpuAccessibleBuffer<ShaderLightSpace>>,
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
    ) -> Arc<CpuAccessibleBuffer<ShaderLightSpace>> {
        let identity = Mat4::IDENTITY.to_cols_array_2d();

        let uniform_buffer_data = ShaderLightSpace { matrix: identity };

        let buffer = CpuAccessibleBuffer::from_data(
            context.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer_data,
        )
        .unwrap();

        buffer
    }

    pub fn light_space_matrix() -> Mat4 {
        let direction = -Vec3::new(30.0, 30.0, 10.0); // .normalize();
        let position = -direction;

        let mut proj = Mat4::orthographic_rh(-25.0, 25.0, 25.0, -25.0, -150.0, 150.0);

        proj.y_axis.y *= -1.0;

        let view = Mat4::look_at_rh(position, direction, Vec3::Y);

        proj * view
    }

    fn update_uniform_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let matrix = Self::light_space_matrix();

        // camera buffer
        let camera_buffer_data = Arc::new(ShaderLightSpace {
            matrix: matrix.to_cols_array_2d(),
        });

        builder
            .update_buffer(self.camera_uniform_buffer.clone(), camera_buffer_data)
            .unwrap();
    }

    fn create_camera_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<ShaderLightSpace>>,
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
        let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        secondary_builder.bind_pipeline_graphics(self.pipeline.clone());

        scene.draw(
            context,
            &mut secondary_builder,
            &self.pipeline,
            |_material| self.camera_descriptor_set.clone(),
        );

        let model_command_buffer = Arc::new(secondary_builder.build().unwrap());

        self.update_uniform_buffers(builder);

        builder
            .begin_render_pass(
                self.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Depth(1.0)],
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

        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<InstanceData>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [DIM, DIM],
                    depth_range: 0.0..1.0,
                },
            ]))
            .rasterization_state(
                RasterizationState::new()
                    .cull_mode(CullMode::Front)
                    .front_face(FrontFace::Clockwise),
            )
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
}

pub mod vs {
    vulkano_shaders::shader! {
                                                                    ty: "vertex",
                                                                    src: "
			#version 450

            layout(binding = 0) uniform LightSpace {
                mat4 matrix;
            } lightSpace;

			// per vertex
			layout(location = 1) in vec3 position;

			// per instance
			layout(location = 4) in mat4 model; 

			void main() {
				gl_Position = lightSpace.matrix * model * vec4(position, 1.0);
			}									
"
    }
}

type ShaderLightSpace = vs::ty::LightSpace;

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
