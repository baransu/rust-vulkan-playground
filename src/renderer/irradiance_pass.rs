use std::sync::Arc;

use gltf::accessor::Dimensions;
use vulkano::{
    buffer::{CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{
        view::{ImageView, ImageViewType},
        AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage,
    },
    pipeline::{
        shader::GraphicsEntryPoint, viewport::Viewport, GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use crate::FramebufferT;

use super::{
    context::Context,
    shaders::CameraUniformBufferObject,
    skybox_pass::{SkyboxPass, SkyboxVertex},
    texture::Texture,
};

pub struct IrradiancePass {
    // graphics_pipeline: Arc<GraphicsPipeline>,
    // vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    // descriptor_set: Arc<PersistentDescriptorSet>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub command_buffer: Arc<SecondaryAutoCommandBuffer>,
    pub attachment: Arc<ImageView<Arc<StorageImage>>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<FramebufferT>,
}

impl IrradiancePass {
    pub fn initialize(context: &Context, texture: &Texture) -> IrradiancePass {
        let render_pass = Self::create_render_pass(context);
        let graphics_pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let vertex_buffer = SkyboxPass::create_vertex_buffer(context);
        let uniform_buffer = SkyboxPass::create_uniform_buffer(context);

        let descriptor_set =
            Self::create_descriptor_set(context, &graphics_pipeline, &uniform_buffer, &texture);

        let command_buffer = Self::create_command_buffer(
            context,
            &graphics_pipeline,
            &descriptor_set,
            &vertex_buffer,
        );

        let attachment = Self::create_attachment(context);

        let framebuffer = Self::create_framebuffer(&render_pass, &attachment);

        IrradiancePass {
            uniform_buffer,
            // graphics_pipeline,
            // vertex_buffer,
            // descriptor_set,
            command_buffer,
            attachment,
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

    fn create_command_buffer(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        descriptor_set: &Arc<PersistentDescriptorSet>,
        vertex_buffer: &Arc<ImmutableBuffer<[SkyboxVertex]>>,
    ) -> Arc<SecondaryAutoCommandBuffer> {
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
            graphics_pipeline.subpass().clone(),
        )
        .unwrap();

        builder.set_viewport(0, [viewport.clone()]);

        builder.bind_pipeline_graphics(graphics_pipeline.clone());

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                graphics_pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(36, 1, 0, 0)
            .unwrap();

        Arc::new(builder.build().unwrap())
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module = vs::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module =
            irradiance_convolution_fs::Shader::load(context.device.clone()).unwrap();

        // TODO: add that to context as util or something
        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<SkyboxVertex>()
                .vertex_shader(vert_shader_module.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
                .fragment_shader(frag_shader_module.main_entry_point(), ())
                .depth_clamp(false)
                // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
                .polygon_mode_fill() // = default
                .line_width(1.0) // = default
                .cull_mode_back()
                .front_face_counter_clockwise()
                // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
                .blend_pass_through()
                // .depth_stencil(DepthStencil::simple_depth_test())
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
    }

    fn create_descriptor_set(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        texture: &Texture,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let image = texture.image.clone();

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        set_builder
            .add_sampled_image(image, context.image_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_attachment(context: &Context) -> Arc<ImageView<Arc<StorageImage>>> {
        ImageView::start(
            StorageImage::with_usage(
                context.device.clone(),
                ImageDimensions::Dim2d {
                    width: 64,
                    height: 64,
                    // TODO: what are array_layers?
                    array_layers: 6,
                },
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transfer_source: true,
                    transfer_destination: true,
                    color_attachment: true,
                    sampled: true,
                    ..ImageUsage::none()
                },
                ImageCreateFlags {
                    cube_compatible: true,
                    ..ImageCreateFlags::none()
                },
                [context.graphics_queue.family()].iter().cloned(),
            )
            .unwrap(),
        )
        .with_type(ImageViewType::Cube)
        .build()
        .unwrap()
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/skybox.vert"
    }
}

pub mod irradiance_convolution_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/irradiance_convolution.frag"
    }
}
