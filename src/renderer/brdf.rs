use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer, SubpassContents,
    },
    format::{ClearValue, Format},
    image::{view::ImageView, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage},
    pipeline::{viewport::Viewport, GraphicsPipeline},
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
    sync::GpuFuture,
};

use crate::FramebufferT;

use super::{
    context::Context,
    screen_frame::{ScreenFrameQuadBuffers, ScreenQuadVertex},
};

const DIM: f32 = 512.0;

pub struct BRDFPass {
    pipeline: Arc<GraphicsPipeline>,
    screen_quad_buffers: ScreenFrameQuadBuffers,
    pub color_attachment_view: Arc<ImageView<Arc<StorageImage>>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<FramebufferT>,
}

impl BRDFPass {
    pub fn initialize(context: &Context) -> BRDFPass {
        let screen_quad_buffers = ScreenFrameQuadBuffers::initialize(context);

        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let color_attachment = Self::create_color_attachment(context);
        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(&render_pass, &color_attachment_view);

        BRDFPass {
            pipeline,
            screen_quad_buffers,
            color_attachment_view,
            render_pass,
            framebuffer,
        }
    }

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        target: &Arc<ImageView<Arc<StorageImage>>>,
    ) -> Arc<FramebufferT> {
        let framebuffer = Framebuffer::start(render_pass.clone())
            .add(target.clone())
            .unwrap()
            .build()
            .unwrap();

        Arc::new(framebuffer)
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        Arc::new(
            single_pass_renderpass!(context.device.clone(),
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
            .unwrap(),
        )
    }

    pub fn execute(&self, context: &Context) {
        let mut builder = AutoCommandBufferBuilder::primary(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        builder
            .begin_render_pass(
                self.framebuffer.clone(),
                SubpassContents::Inline,
                vec![ClearValue::Float([0.0, 0.0, 1.0, 1.0])],
            )
            .unwrap();

        builder
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.screen_quad_buffers.vertex_buffer.clone())
            .bind_index_buffer(self.screen_quad_buffers.index_buffer.clone())
            .draw_indexed(self.screen_quad_buffers.indices_length as u32, 1, 0, 0, 0)
            .unwrap();

        builder.end_render_pass().unwrap();

        builder
            .build()
            .unwrap()
            .execute(context.graphics_queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module = vs::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module = fs::Shader::load(context.device.clone()).unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<ScreenQuadVertex>()
                .vertex_shader(vert_shader_module.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
                .fragment_shader(frag_shader_module.main_entry_point(), ())
                .front_face_counter_clockwise()
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
    }

    fn create_color_attachment(context: &Context) -> Arc<StorageImage> {
        StorageImage::with_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: DIM as u32,
                height: DIM as u32,
                array_layers: 1,
            },
            Format::R16G16B16A16_SFLOAT,
            ImageUsage {
                color_attachment: true,
                sampled: true,
                ..ImageUsage::none()
            },
            ImageCreateFlags::none(),
            [context.graphics_queue.family()].iter().cloned(),
        )
        .unwrap()
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/fullscreen.vert"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/brdf.frag"
    }
}