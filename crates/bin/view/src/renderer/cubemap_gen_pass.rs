use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, SubpassContents},
    descriptor_set::PersistentDescriptorSet,
    format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewType},
        AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, ImageViewAbstract,
        MipmapsCount, StorageImage,
    },
    pipeline::{
        graphics::{
            rasterization::{FrontFace, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    shader::EntryPoint,
    single_pass_renderpass,
};

use super::{
    context::Context,
    shaders::CameraUniformBufferObject,
    skybox_pass::{SkyboxPass, SkyboxVertex},
};

const NUM_SAMPLES: i32 = 1024;

pub struct CubemapGenPass {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    roughness_uniform_buffer: Arc<CpuAccessibleBuffer<RoughnessBufferObject>>,
    pub cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<StorageImage>>,
    pub color_attachment: Arc<AttachmentImage>,
    pub color_attachment_view: Arc<ImageView<AttachmentImage>>,
    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<Framebuffer>,
    dim: f32,
}

impl CubemapGenPass {
    pub fn initialize<T1>(
        context: &Context,
        skybox_cubemap: &Arc<T1>,
        fragment_shader_entry_point: EntryPoint,
        format: Format,
        dim: f32,
    ) -> CubemapGenPass
    where
        T1: ImageViewAbstract + 'static,
    {
        let render_pass = Self::create_render_pass(context, format);
        let pipeline =
            Self::create_graphics_pipeline(context, &render_pass, fragment_shader_entry_point);

        let vertex_buffer = SkyboxPass::create_vertex_buffer(context);
        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);
        let roughness_uniform_buffer = Self::create_roughness_uniform_buffer(context);

        let descriptor_set = Self::create_descriptor_set(
            context,
            &pipeline,
            &camera_uniform_buffer,
            &roughness_uniform_buffer,
            &skybox_cubemap,
        );

        let cube_attachment = Self::create_cube_attachment(context, format, dim);
        let color_attachment = Self::create_color_attachment(context, format, dim);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(&render_pass, &color_attachment_view);

        CubemapGenPass {
            camera_uniform_buffer,
            roughness_uniform_buffer,
            pipeline,
            vertex_buffer,
            descriptor_set,
            cube_attachment,
            cube_attachment_view,
            color_attachment,
            color_attachment_view,
            render_pass,
            framebuffer,
            dim,
        }
    }

    fn create_camera_uniform_buffer(
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

    fn create_roughness_uniform_buffer(
        context: &Context,
    ) -> Arc<CpuAccessibleBuffer<RoughnessBufferObject>> {
        let uniform_buffer_data = RoughnessBufferObject {
            numSamples: NUM_SAMPLES,
            roughness: 0.0,
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

    fn update_uniform_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        matrix: Mat4,
        mip_map_level: u32,
        num_mips: u32,
    ) {
        // camera buffer
        let camera_buffer_data = Arc::new(CameraUniformBufferObject {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: (Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 10.0) * matrix)
                .to_cols_array_2d(),
            position: Vec3::ZERO.to_array(),
        });

        builder
            .update_buffer(self.camera_uniform_buffer.clone(), camera_buffer_data)
            .unwrap();

        let roughness = (mip_map_level as f32) / ((num_mips - 1) as f32);

        // roughness buffer
        let roughness_buffer_data = Arc::new(RoughnessBufferObject {
            numSamples: NUM_SAMPLES,
            roughness,
        });

        builder
            .update_buffer(self.roughness_uniform_buffer.clone(), roughness_buffer_data)
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

    fn create_render_pass(context: &Context, format: Format) -> Arc<RenderPass> {
        single_pass_renderpass!(context.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: format,
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

    pub fn add_to_builder(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) {
        let mats = matrices();

        let num_mips = (self.dim.log2().floor() + 1.0) as u32;

        for m in 0..num_mips {
            for f in 0..6 {
                let width = self.dim * 0.5_f32.powf(m as f32);
                let height = self.dim * 0.5_f32.powf(m as f32);

                let viewport = Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [width, height],
                    depth_range: 0.0..1.0,
                };

                self.update_uniform_buffers(builder, mats[f], m, num_mips);

                builder
                    .begin_render_pass(
                        self.framebuffer.clone(),
                        SubpassContents::Inline,
                        vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
                    )
                    .unwrap();

                builder
                    .bind_pipeline_graphics(self.pipeline.clone())
                    .set_viewport(0, [viewport.clone()])
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

                builder
                    .copy_image(
                        self.color_attachment.clone(),
                        [0, 0, 0],
                        0,
                        0,
                        self.cube_attachment.clone(),
                        [0, 0, 0],
                        f as u32,
                        m,
                        [width as u32, height as u32, 1],
                        1,
                    )
                    .unwrap();
            }
        }
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        fragment_shader_entry_point: EntryPoint,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<SkyboxVertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fragment_shader_entry_point, ())
            .rasterization_state(
                RasterizationState::new()
                    // .cull_mode(CullMode::Back)
                    .front_face(FrontFace::CounterClockwise),
            )
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_descriptor_set<T1>(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        roughness_uniform_buffer: &Arc<CpuAccessibleBuffer<RoughnessBufferObject>>,
        skybox_cubemap: &Arc<T1>,
    ) -> Arc<PersistentDescriptorSet>
    where
        T1: ImageViewAbstract + 'static,
    {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(camera_uniform_buffer.clone())
            .unwrap();

        let max_lod = 1024.0_f32.log2().floor() + 1.0;

        set_builder
            .add_sampled_image(
                skybox_cubemap.clone(),
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
                    max_lod,
                )
                .unwrap(),
            )
            .unwrap();

        set_builder
            .add_buffer(roughness_uniform_buffer.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    fn create_color_attachment(
        context: &Context,
        format: Format,
        dim: f32,
    ) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            context.device.clone(),
            [dim as u32, dim as u32],
            format,
            ImageUsage {
                color_attachment: true,
                transfer_source: true,
                ..ImageUsage::none()
            },
        )
        .unwrap()
    }

    fn create_cube_attachment(context: &Context, format: Format, dim: f32) -> Arc<StorageImage> {
        let num_mips = (dim.log2().floor() + 1.0) as u32;

        StorageImage::with_mipmaps_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: dim as u32,
                height: dim as u32,
                array_layers: 6,
            },
            format,
            MipmapsCount::Specific(num_mips),
            ImageUsage {
                transfer_destination: true,
                transfer_source: true,
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

type RoughnessBufferObject = prefilterenvmap_fs::ty::RoughnessBufferObject;

fn matrices() -> [Mat4; 6] {
    [
        // POSITIVE_X
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // NEGATIVE_X
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // POSITIVE_Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        // NEGATIVE_Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0), -Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        // POSITIVE_Z
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // NEGATIVE_Z
        Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    ]
}
