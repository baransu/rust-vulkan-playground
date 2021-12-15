use std::{f32::consts::PI, sync::Arc};

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewType},
        ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract, ImmutableImage,
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
    context::Context,
    shaders::CameraUniformBufferObject,
    skybox_pass::{SkyboxPass, SkyboxVertex},
};

const DIM: f32 = 64.0;

pub struct CubemapGenerationPass {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<Arc<StorageImage>>>,
    pub color_attachment: Arc<StorageImage>,
    pub color_attachment_view: Arc<ImageView<Arc<StorageImage>>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<FramebufferT>,
}

impl CubemapGenerationPass {
    pub fn initialize<T>(
        context: &Context,
        input_image: &Arc<T>,
        fragment_shader_entry_point: GraphicsEntryPoint,
    ) -> CubemapGenerationPass
    where
        T: ImageViewAbstract + 'static,
    {
        let render_pass = Self::create_render_pass(context);
        let pipeline =
            Self::create_graphics_pipeline(context, &render_pass, fragment_shader_entry_point);

        let vertex_buffer = SkyboxPass::create_vertex_buffer(context);
        let uniform_buffer = Self::create_uniform_buffer(context);

        let descriptor_set =
            Self::create_descriptor_set(context, &pipeline, &uniform_buffer, &input_image);

        let cube_attachment = Self::create_cube_attachment(context);
        let color_attachment = Self::create_color_attachment(context);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(&render_pass, &color_attachment_view);

        CubemapGenerationPass {
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
                            format: Format::R32G32B32A32_SFLOAT,
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

        let mats = matrices();

        // let num_mips = (DIM.floor().log2() + 1.0) as u32;

        // TODO: vulkano is not supporting mipmaps for StorageImage right now
        // for m in 0..num_mips {
        let m = 0;
        for f in 0..6 {
            let width = DIM * 0.5_f32.powf(m as f32);
            let height = DIM * 0.5_f32.powf(m as f32);
            let viewport = Viewport {
                origin: [0.0, 0.0],
                dimensions: [width, height],
                depth_range: 0.0..1.0,
            };

            Self::update_uniform_buffer(&self.uniform_buffer, &mut builder, &mats[f]);

            builder
                .begin_render_pass(
                    self.framebuffer.clone(),
                    SubpassContents::Inline,
                    vec![ClearValue::Float([1.0, 0.0, 0.0, 1.0])],
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
                    m,
                    [width as u32, height as u32, 1],
                    1,
                )
                .unwrap();
        }
        // }

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
        fragment_shader_entry_point: GraphicsEntryPoint,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module = vs::Shader::load(context.device.clone()).unwrap();

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
                .fragment_shader(fragment_shader_entry_point, ())
                .front_face_counter_clockwise()
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
    }

    fn create_descriptor_set<T>(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        input_image: &Arc<T>,
    ) -> Arc<PersistentDescriptorSet>
    where
        T: ImageViewAbstract + 'static,
    {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        set_builder
            .add_sampled_image(input_image.clone(), context.image_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_color_attachment(context: &Context) -> Arc<StorageImage> {
        StorageImage::with_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: DIM as u32,
                height: DIM as u32,
                // TODO: what are array_layers?
                array_layers: 1,
            },
            Format::R32G32B32A32_SFLOAT,
            ImageUsage {
                color_attachment: true,
                transfer_source: true,
                sampled: true,
                ..ImageUsage::none()
            },
            ImageCreateFlags::none(),
            [context.graphics_queue.family()].iter().cloned(),
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
            Format::R32G32B32A32_SFLOAT,
            ImageUsage {
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

pub mod prefilterenvmap_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/prefilterenvmap.frag"
    }
}

fn matrices() -> [Mat4; 6] {
    [
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), -Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    ]
}
