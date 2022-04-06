use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::layout::DescriptorSetLayout,
    format::ClearValue,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use super::{
    context::Context, entity::InstanceData, gbuffer::GBuffer, scene::Scene,
    shaders::CameraUniformBufferObject,
};

const DIM: f32 = 1024.0;

pub struct DirLightShadows {
    pipeline: Arc<GraphicsPipeline>,
    pub target_attachment: Arc<ImageView<AttachmentImage>>,

    framebuffer: Arc<Framebuffer>,
}

impl DirLightShadows {
    pub fn initialize(
        context: &Context,
        layouts: &Vec<Arc<DescriptorSetLayout>>,
    ) -> DirLightShadows {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, layouts, &render_pass);

        let target_attachment = Self::create_depth_attachment(context);
        let framebuffer = Self::create_framebuffer(&render_pass, &target_attachment);

        DirLightShadows {
            pipeline,
            target_attachment,

            framebuffer,
        }
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
        uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    ) {
        let view = Self::light_space_matrix().to_cols_array_2d();

        let camera_buffer_data = Arc::new(CameraUniformBufferObject {
            view,
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            position: Vec3::ZERO.to_array(),
        });

        builder
            .update_buffer(uniform_buffer.clone(), camera_buffer_data)
            .unwrap();
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
            |(primitive, _material)| {
                (
                    primitive.descriptor_set.clone(),
                    scene.descriptor_set.clone(),
                )
            },
        );

        let model_command_buffer = Arc::new(secondary_builder.build().unwrap());

        self.update_uniform_buffers(builder, &scene.camera_uniform_buffer);

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
        layouts: &Vec<Arc<DescriptorSetLayout>>,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        let pipeline_layout = GBuffer::create_pipeline_layout(
            context,
            vec![
                Arc::new(fs.entry_point("main").unwrap()).as_ref(),
                Arc::new(vs.entry_point("main").unwrap()).as_ref(),
            ],
            &layouts,
        );

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().instance::<InstanceData>())
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
            .with_pipeline_layout(context.device.clone(), pipeline_layout)
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
        path: "src/renderer/shaders/dir_light.vert",
        types_meta: {
            #[derive(Clone, Copy, Default)]
        }
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
