use std::sync::Arc;

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
};

use super::{
    context::Context,
    screen_frame::{ScreenFrameQuadBuffers, ScreenQuadVertex},
    shaders::screen_vertex_shader,
};

pub struct SsaoBlur {
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<Framebuffer>,
    pub pipeline: Arc<GraphicsPipeline>,

    pub target: Arc<ImageView<AttachmentImage>>,

    pub descriptor_set: Arc<PersistentDescriptorSet>,

    pub command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,
}

impl SsaoBlur {
    pub fn initialize(context: &Context, ssao: &Arc<ImageView<AttachmentImage>>) -> SsaoBlur {
        let screen_quad_buffers = ScreenFrameQuadBuffers::initialize(context);

        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_pipeline(context, &render_pass);

        let target = Self::create_attachment(context);

        let framebuffer = Self::create_framebuffer(&render_pass, &target);

        let descriptor_set = Self::create_descriptor_set(&context, &pipeline, &ssao);

        let command_buffers =
            Self::create_command_buffers(context, &pipeline, &descriptor_set, &screen_quad_buffers);

        SsaoBlur {
            render_pass,
            pipeline,
            framebuffer,

            target,

            descriptor_set,

            command_buffers,
        }
    }

    fn create_pipeline(context: &Context, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
        let vs = screen_vertex_shader::load(context.graphics_queue.device().clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        let dimensions = context.swap_chain.dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        GraphicsPipeline::start()
            .vertex_input_single_buffer::<ScreenQuadVertex>()
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .blend_pass_through()
            .viewports_dynamic_scissors_irrelevant(1)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_command_buffers(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        descriptor_set: &Arc<PersistentDescriptorSet>,
        screen_quad_buffers: &ScreenFrameQuadBuffers,
    ) -> Vec<Arc<SecondaryAutoCommandBuffer>> {
        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

        for _i in 0..context.swap_chain.num_images() as usize {
            let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                context.device.clone(),
                context.graphics_queue.family(),
                CommandBufferUsage::SimultaneousUse,
                graphics_pipeline.subpass().clone(),
            )
            .unwrap();

            builder
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(graphics_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    graphics_pipeline.layout().clone(),
                    0,
                    descriptor_set.clone(),
                )
                .bind_vertex_buffers(0, screen_quad_buffers.vertex_buffer.clone())
                .bind_index_buffer(screen_quad_buffers.index_buffer.clone())
                .draw_indexed(screen_quad_buffers.indices_length as u32, 1, 0, 0, 0)
                .unwrap();

            let command_buffer = Arc::new(builder.build().unwrap());

            command_buffers.push(command_buffer);
        }

        command_buffers
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(context.device.clone(),
                attachments: {
                                        color: {
                                                load: Clear,
                                                store: Store,
                                                format: Format::R16G16B16A16_SFLOAT,
                                                samples: 1,
                                        }

                                },
                pass: {
                                                        color: [color],
                                                        depth_stencil: {}
                        }


        )
        .unwrap()
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

    fn create_attachment(context: &Context) -> Arc<ImageView<AttachmentImage>> {
        let dimensions = context.swap_chain.dimensions();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                dimensions,
                Format::R16G16B16A16_SFLOAT,
                usage,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn create_descriptor_set(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        ssao: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_sampled_image(ssao.clone(), context.attachment_sampler.clone())
            .unwrap();

        set_builder.build().unwrap()
    }
}

mod fs {
    vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/renderer/shaders/ssao_blur.frag"
    }
}
