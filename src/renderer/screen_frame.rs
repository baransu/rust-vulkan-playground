use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    image::{view::ImageView, AttachmentImage},
    pipeline::{
        graphics::{
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
    sync::GpuFuture,
};

use super::context::Context;

#[derive(Default, Debug, Clone)]
pub struct ScreenQuadVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl ScreenQuadVertex {
    fn new(position: [f32; 2], uv: [f32; 2]) -> ScreenQuadVertex {
        ScreenQuadVertex { position, uv }
    }
}

vulkano::impl_vertex!(ScreenQuadVertex, position, uv);

pub struct ScreenFrameQuadBuffers {
    pub index_buffer: Arc<ImmutableBuffer<[u16]>>,
    pub vertex_buffer: Arc<ImmutableBuffer<[ScreenQuadVertex]>>,
    pub indices_length: u32,
}

impl ScreenFrameQuadBuffers {
    pub fn initialize(context: &Context) -> ScreenFrameQuadBuffers {
        let quad_vertices = screen_quad_vertices();
        let quad_indices = screen_quad_indices();

        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            quad_vertices.clone(),
            BufferUsage::vertex_buffer(),
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        let (index_buffer, future) = ImmutableBuffer::from_iter(
            quad_indices.clone(),
            BufferUsage::index_buffer(),
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ScreenFrameQuadBuffers {
            index_buffer,
            vertex_buffer,
            indices_length: quad_indices.len() as u32,
        }
    }
}

/**
 * System used to render offscreen frame to screen.
 */
pub struct ScreenFrame {
    graphics_pipeline: Arc<GraphicsPipeline>,
    pub framebuffers: Vec<Arc<Framebuffer>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    screen_quad_buffers: ScreenFrameQuadBuffers,
    render_pass: Arc<RenderPass>,
}

impl ScreenFrame {
    pub fn initialize(
        context: &Context,
        scene_frame: &Arc<ImageView<AttachmentImage>>,
        ui_frame: &Arc<ImageView<AttachmentImage>>,
    ) -> ScreenFrame {
        let screen_quad_buffers = ScreenFrameQuadBuffers::initialize(context);

        let render_pass = Self::create_render_pass(context);
        let graphics_pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let framebuffers = Self::create_framebuffers_from_swap_chain_images(context, &render_pass);

        let descriptor_set =
            Self::create_descriptor_set(&graphics_pipeline, &scene_frame, ui_frame);

        ScreenFrame {
            graphics_pipeline,
            framebuffers,
            descriptor_set,
            screen_quad_buffers,
            render_pass,
        }
    }

    pub fn recreate_swap_chain(&mut self, context: &Context) {
        self.framebuffers =
            Self::create_framebuffers_from_swap_chain_images(context, &self.render_pass);
    }

    pub fn create_command_buffer(&self, context: &Context) -> Arc<SecondaryAutoCommandBuffer> {
        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.graphics_pipeline.subpass().clone(),
        )
        .unwrap();

        builder
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.graphics_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.graphics_pipeline.layout().clone(),
                0,
                self.descriptor_set.clone(),
            )
            .bind_vertex_buffers(0, self.screen_quad_buffers.vertex_buffer.clone())
            .bind_index_buffer(self.screen_quad_buffers.index_buffer.clone())
            .draw_indexed(self.screen_quad_buffers.indices_length as u32, 1, 0, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    fn create_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        scene_frame: &Arc<ImageView<AttachmentImage>>,
        ui_frame: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_image(scene_frame.clone()).unwrap();
        set_builder.add_image(ui_frame.clone()).unwrap();

        set_builder.build().unwrap()
    }

    /**
     * This function created frame buffer for each swap chain image.
     * It contains 1 attachment which is our offscreen framebuffer containing rendered scene.
     */
    fn create_framebuffers_from_swap_chain_images(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        context
            .swap_chain_images
            .iter()
            .map(|swapchain_image| {
                let image = ImageView::new(swapchain_image.clone()).unwrap();

                Framebuffer::start(render_pass.clone())
                    .add(image.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    /**
     * Creates graphics pipeline which is used to render offscreen frame buffer to screen
     */
    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = super::shaders::screen_vertex_shader::load(context.device.clone()).unwrap();
        let fs = super::shaders::screen_fragment_shader::load(context.device.clone()).unwrap();

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<ScreenQuadVertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .with_auto_layout(context.device.clone(), |set_descs| {
                // Modify the auto-generated layout by setting an immutable sampler to
                // set 0 binding 0.
                set_descs[0].set_immutable_samplers(0, [context.attachment_sampler.clone()]);
                // set 0 binding 1.
                set_descs[0].set_immutable_samplers(1, [context.attachment_sampler.clone()]);
            })
            .unwrap()
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        let color_format = context.swap_chain.format();

        single_pass_renderpass!(context.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: color_format,
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
}

fn screen_quad_vertices() -> [ScreenQuadVertex; 4] {
    [
        ScreenQuadVertex::new([-1.0, -1.0], [0.0, 0.0]),
        ScreenQuadVertex::new([1.0, -1.0], [1.0, 0.0]),
        ScreenQuadVertex::new([1.0, 1.0], [1.0, 1.0]),
        ScreenQuadVertex::new([-1.0, 1.0], [0.0, 1.0]),
    ]
}

fn screen_quad_indices() -> [u16; 6] {
    [0, 1, 2, 2, 3, 0]
}
