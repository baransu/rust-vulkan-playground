use std::sync::Arc;

use glam::Vec3;
use rand::Rng;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount,
    },
    pipeline::{viewport::Viewport, GraphicsPipeline, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
    sync::GpuFuture,
};

use crate::FramebufferT;

use super::{
    context::Context,
    gbuffer::GBuffer,
    screen_frame::{ScreenFrameQuadBuffers, ScreenQuadVertex},
    shaders::{screen_vertex_shader, CameraUniformBufferObject},
};

const SAMPLES_SIZE: usize = 64;

pub struct Ssao {
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<FramebufferT>,
    pub pipeline: Arc<GraphicsPipeline>,

    pub target: Arc<ImageView<Arc<AttachmentImage>>>,

    pub descriptor_set: Arc<PersistentDescriptorSet>,

    pub command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,
}

impl Ssao {
    pub fn initialize(
        context: &Context,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        gbuffer: &GBuffer,
    ) -> Ssao {
        let screen_quad_buffers = ScreenFrameQuadBuffers::initialize(context);

        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_pipeline(context, &render_pass);

        let target = Self::create_attachment(context);

        let framebuffer = Self::create_framebuffer(&render_pass, &target);

        let noise_texture = Self::create_noise_texture(context);
        let uniform_buffer = Self::create_uniform_buffer(context);
        let descriptor_set = Self::create_descriptor_set(
            &context,
            &pipeline,
            &camera_uniform_buffer,
            &gbuffer,
            &uniform_buffer,
            &noise_texture,
        );

        let command_buffers =
            Self::create_command_buffers(context, &pipeline, &descriptor_set, &screen_quad_buffers);

        Ssao {
            render_pass,
            pipeline,
            framebuffer,

            target,

            descriptor_set,

            command_buffers,
        }
    }

    fn create_pipeline(context: &Context, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
        let vs =
            screen_vertex_shader::Shader::load(context.graphics_queue.device().clone()).unwrap();
        let fs = fs::Shader::load(context.device.clone()).unwrap();

        let dimensions = context.swap_chain.dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<ScreenQuadVertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
                .fragment_shader(fs.main_entry_point(), ())
                .blend_pass_through()
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
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
        Arc::new(
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
            .unwrap(),
        )
    }

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        target: &Arc<ImageView<Arc<AttachmentImage>>>,
    ) -> Arc<FramebufferT> {
        let framebuffer = Framebuffer::start(render_pass.clone())
            .add(target.clone())
            .unwrap()
            .build()
            .unwrap();

        Arc::new(framebuffer)
    }

    fn create_attachment(context: &Context) -> Arc<ImageView<Arc<AttachmentImage>>> {
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
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        gbuffer: &GBuffer,
        ssao_unfirm_buffer: &Arc<CpuAccessibleBuffer<SsaoUniformBufferObject>>,
        noise_texture: &Arc<ImageView<Arc<ImmutableImage>>>,
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
            .add_sampled_image(
                gbuffer.position_buffer.clone(),
                context.attachment_sampler.clone(),
            )
            .unwrap();

        set_builder
            .add_sampled_image(
                gbuffer.normals_buffer.clone(),
                context.attachment_sampler.clone(),
            )
            .unwrap();

        set_builder
            .add_sampled_image(noise_texture.clone(), context.attachment_sampler.clone())
            .unwrap();

        set_builder.add_buffer(ssao_unfirm_buffer.clone()).unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<SsaoUniformBufferObject>> {
        let mut rng = rand::thread_rng();

        let mut samples = [[0.0, 0.0, 0.0, 0.0]; SAMPLES_SIZE];

        for i in 0..SAMPLES_SIZE {
            // inlined lerp
            let mut scale = i as f32 / SAMPLES_SIZE as f32;
            scale = lerp(0.1, 1.0, scale * scale);

            let sample = Vec3::new(
                rng.gen_range(0.0..1.0) * 2.0 - 1.0,
                rng.gen_range(0.0..1.0) * 2.0 - 1.0,
                rng.gen_range(0.0..1.0),
            )
            .normalize()
                * rng.gen_range(0.0..1.0)
                * scale;

            samples[i][0] = sample.x;
            samples[i][1] = sample.y;
            samples[i][2] = sample.z;
            samples[i][3] = 0.0;
        }

        let buffer_data = SsaoUniformBufferObject { samples };

        let buffer = CpuAccessibleBuffer::from_data(
            context.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            buffer_data,
        )
        .unwrap();

        buffer
    }

    fn create_noise_texture(context: &Context) -> Arc<ImageView<Arc<ImmutableImage>>> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(4 * 4 * 3);

        for _ in 0..16 {
            data.push(rng.gen_range(0.0..1.0) * 2.0 - 1.0);
            data.push(rng.gen_range(0.0..1.0) * 2.0 - 1.0);
            data.push(0.0);
        }

        let (image, future) = ImmutableImage::from_iter(
            data.iter().cloned(),
            ImageDimensions::Dim2d {
                width: 4,
                height: 4,
                array_layers: 1,
            },
            MipmapsCount::One,
            Format::R32G32B32A32_SFLOAT,
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }
}

mod fs {
    vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/renderer/shaders/ssao.frag"
    }
}

pub type SsaoUniformBufferObject = fs::ty::SsaoUniformBufferObject;

fn lerp(a: f32, b: f32, f: f32) -> f32 {
    return a + f * (b - a);
}
