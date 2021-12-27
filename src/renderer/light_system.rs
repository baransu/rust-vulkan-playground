use std::sync::Arc;

use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImmutableImage, StorageImage},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    single_pass_renderpass,
};

use super::{
    context::Context,
    gbuffer::{GBuffer, GBufferTarget},
    screen_frame::{ScreenFrameQuadBuffers, ScreenQuadVertex},
    shaders::{screen_vertex_shader, CameraUniformBufferObject},
};

pub struct LightSystem {
    pub pipeline: Arc<GraphicsPipeline>,
    pub framebuffer: Arc<Framebuffer>,
    pub render_pass: Arc<RenderPass>,

    pub command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,
}

impl LightSystem {
    pub fn initialize(
        context: &Context,
        target: &GBufferTarget,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        light_uniform_buffer: &Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,
        gbuffer: &GBuffer,
        ssao_target: &Arc<ImageView<AttachmentImage>>,
        irradiance_map: &Arc<ImageView<StorageImage>>,
        prefilter_map: &Arc<ImageView<StorageImage>>,
        brdf: &Arc<ImageView<ImmutableImage>>,
        point_shadow_map: &Arc<ImageView<StorageImage>>,
        dir_shadow_map: &Arc<ImageView<AttachmentImage>>,
    ) -> LightSystem {
        let screen_quad_buffers = ScreenFrameQuadBuffers::initialize(context);

        let render_pass = Self::create_render_pass(context);
        let framebuffer = Self::create_framebuffer(&render_pass, target);
        let pipeline = Self::create_pipeline(context, &render_pass);

        let descriptor_set = Self::create_descriptor_set(
            context,
            &pipeline,
            &camera_uniform_buffer,
            &gbuffer,
            &light_uniform_buffer,
            &ssao_target,
            &irradiance_map,
            &prefilter_map,
            &brdf,
            &point_shadow_map,
            &dir_shadow_map,
        );

        let command_buffers =
            Self::create_command_buffers(context, &pipeline, &descriptor_set, &screen_quad_buffers);

        LightSystem {
            pipeline,
            render_pass,
            framebuffer,

            command_buffers,
        }
    }

    fn create_descriptor_set(
        context: &Context,
        light_graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        gbuffer: &GBuffer,
        light_uniform_buffer: &Arc<CpuAccessibleBuffer<LightUniformBufferObject>>,
        ssao_target: &Arc<ImageView<AttachmentImage>>,
        irradiance_map: &Arc<ImageView<StorageImage>>,
        prefilter_map: &Arc<ImageView<StorageImage>>,
        brdf: &Arc<ImageView<ImmutableImage>>,
        point_shadow_map: &Arc<ImageView<StorageImage>>,
        dir_shadow_map: &Arc<ImageView<AttachmentImage>>,
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
            .add_sampled_image(
                gbuffer.albedo_buffer.clone(),
                context.attachment_sampler.clone(),
            )
            .unwrap();

        set_builder
            .add_sampled_image(
                gbuffer.metalic_roughness_buffer.clone(),
                context.attachment_sampler.clone(),
            )
            .unwrap();

        set_builder
            .add_sampled_image(gbuffer.depth_buffer.clone(), context.depth_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(point_shadow_map.clone(), context.depth_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(dir_shadow_map.clone(), context.depth_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(ssao_target.clone(), context.attachment_sampler.clone())
            .unwrap();

        set_builder
            .add_sampled_image(irradiance_map.clone(), Self::create_sampler(context, 7.0))
            .unwrap();

        set_builder
            .add_sampled_image(prefilter_map.clone(), Self::create_sampler(context, 10.0))
            .unwrap();

        set_builder
            .add_sampled_image(brdf.clone(), Self::create_sampler(context, 1.0))
            .unwrap();

        set_builder
            .add_buffer(light_uniform_buffer.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    fn create_sampler(context: &Context, mip: f32) -> Arc<Sampler> {
        Sampler::new(
            context.device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0,
            1.0,
            0.0,
            mip,
        )
        .unwrap()
    }

    fn create_pipeline(context: &Context, render_pass: &Arc<RenderPass>) -> Arc<GraphicsPipeline> {
        let vs = screen_vertex_shader::load(context.graphics_queue.device().clone()).unwrap();
        let fs = fs::load(context.graphics_queue.device().clone()).unwrap();

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<ScreenQuadVertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .input_assembly_state(InputAssemblyState::new())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        target: &GBufferTarget,
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
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/pbr.frag"
    }
}

pub type LightUniformBufferObject = fs::ty::LightUniformBufferObject;
pub type ShaderPointLight = fs::ty::PointLight;
