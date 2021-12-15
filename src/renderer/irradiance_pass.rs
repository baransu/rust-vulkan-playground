use std::{f32::consts::PI, sync::Arc};

use glam::{Mat4, Vec3};
use gltf::accessor::Dimensions;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewType},
        AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, ImmutableImage,
        StorageImage,
    },
    pipeline::{
        shader::GraphicsEntryPoint, viewport::Viewport, GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
    sync::GpuFuture,
};

use crate::FramebufferT;

use super::{
    camera::Camera,
    context::Context,
    shaders::CameraUniformBufferObject,
    skybox_pass::{SkyboxPass, SkyboxVertex},
    texture::Texture,
};

const DIM: f32 = 64.0;

pub struct IrradiancePass {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    // pub command_buffer: Arc<SecondaryAutoCommandBuffer>,
    pub cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<Arc<StorageImage>>>,
    pub color_attachment: Arc<AttachmentImage>,
    pub color_attachment_view: Arc<ImageView<Arc<AttachmentImage>>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<FramebufferT>,
}

impl IrradiancePass {
    pub fn initialize(
        context: &Context,
        texture: &Arc<ImageView<Arc<ImmutableImage>>>,
    ) -> IrradiancePass {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let vertex_buffer = SkyboxPass::create_vertex_buffer(context);
        let uniform_buffer = Self::create_uniform_buffer(context);

        let descriptor_set =
            Self::create_descriptor_set(context, &pipeline, &uniform_buffer, &texture);

        let cube_attachment = Self::create_cube_attachment(context);
        let color_attachment = Self::create_color_attachment(context);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(&render_pass, &color_attachment_view);

        IrradiancePass {
            uniform_buffer,
            pipeline,
            vertex_buffer,
            descriptor_set,
            cube_attachment,
            cube_attachment_view,
            color_attachment,
            color_attachment_view,
            render_pass,
            framebuffer,
        }
    }

    fn create_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<CameraUniformBufferObject>> {
        let camera: Camera = Default::default();

        let dim = context.swap_chain.dimensions();
        let mut uniform_buffer_data =
            camera.get_skybox_uniform_data([dim[0] as f32, dim[1] as f32]);

        // TODO: this have to be set to matrix for correct face
        uniform_buffer_data.view = Mat4::IDENTITY.to_cols_array_2d();

        uniform_buffer_data.proj =
            Mat4::perspective_rh(PI / 2.0, 1.0, 0.1, 512.0).to_cols_array_2d();

        let buffer = CpuAccessibleBuffer::from_data(
            context.device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer_data,
        )
        .unwrap();

        buffer
    }

    fn update_uniform_buffer(
        uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        matrix: &Mat4,
    ) {
        let view = matrix.to_cols_array_2d();

        let proj = Mat4::perspective_rh(PI / 2.0, 1.0, 0.1, 512.0).to_cols_array_2d();

        let data = Arc::new(CameraUniformBufferObject {
            view,
            proj,
            position: Vec3::ZERO.to_array(),
        });

        builder.update_buffer(uniform_buffer.clone(), data).unwrap();
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

    pub fn draw(&self, context: &Context) -> PrimaryAutoCommandBuffer {
        let mut builder = AutoCommandBufferBuilder::primary(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let mats = matrices();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        for f in 0..6 {
            Self::update_uniform_buffer(&self.uniform_buffer, &mut builder, &mats[f]);

            builder
                .begin_render_pass(
                    self.framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
                )
                .unwrap();

            builder
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(self.pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    self.descriptor_set.clone(),
                )
                .bind_vertex_buffers(0, self.vertex_buffer.clone())
                .draw(36, 1, 0, 0)
                .unwrap();

            builder.end_render_pass().unwrap();

            let source = self.color_attachment.clone();

            let destination = self.cube_attachment.clone();

            builder
                .copy_image(
                    source,
                    [0, 0, 0],
                    0,
                    0,
                    destination,
                    [0, 0, 0],
                    f as u32,
                    0,
                    [DIM as u32, DIM as u32, 1],
                    1,
                )
                .unwrap();
        }

        builder.build().unwrap()
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module = vs::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module =
            irradiance_convolution_fs::Shader::load(context.device.clone()).unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
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
        image: &Arc<ImageView<Arc<ImmutableImage>>>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        set_builder
            .add_sampled_image(image.clone(), context.attachment_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_color_attachment(context: &Context) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            context.device.clone(),
            [DIM as u32, DIM as u32],
            Format::R16G16B16A16_SFLOAT,
            ImageUsage {
                color_attachment: true,
                transfer_source: true,
                transfer_destination: true,
                sampled: true,
                ..ImageUsage::none()
            },
        )
        .unwrap()
    }

    fn create_cube_attachment(context: &Context) -> Arc<StorageImage> {
        StorageImage::with_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: DIM as u32,
                height: DIM as u32,
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

fn matrices() -> [Mat4; 6] {
    [
        // POSITIVE_X
        Mat4::from_rotation_y(90.0_f32.to_radians())
            * Mat4::from_rotation_x(180.0_f32.to_radians()),
        // glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),

        // NEGATIVE_X
        Mat4::from_rotation_y(-90.0_f32.to_radians())
            * Mat4::from_rotation_x(180.0_f32.to_radians()),
        // glm::rotate(glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0.0f, 1.0f, 0.0f)), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),

        // POSITIVE_Y
        Mat4::from_rotation_x(-90.0_f32.to_radians()),
        // glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),

        // NEGATIVE_Y
        Mat4::from_rotation_x(90.0_f32.to_radians()),
        // glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)),

        // POSITIVE_Z
        Mat4::from_rotation_x(180.0_f32.to_radians()),
        // glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f)),

        // NEGATIVE_Z
        Mat4::from_rotation_z(180.0_f32.to_radians()),
        // glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    ]
}
