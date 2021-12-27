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
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use super::{
    context::Context,
    skybox_pass::{SkyboxPass, SkyboxVertex},
};

const DIM: f32 = 1024.0;
const FORMAT: Format = Format::R16G16B16A16_SFLOAT;

pub struct GenHdrCubemap {
    pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    uniform_buffer: Arc<CpuAccessibleBuffer<Ubo>>,

    cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<StorageImage>>,
    color_attachment: Arc<AttachmentImage>,
    framebuffer: Arc<Framebuffer>,
}

impl GenHdrCubemap {
    pub fn initialize(context: &Context, path: &str) -> GenHdrCubemap {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let vertex_buffer = SkyboxPass::create_vertex_buffer(context);
        let uniform_buffer = Self::create_uniform_buffer(context);

        let hdr = SkyboxPass::load_skybox_texture(&context, path);
        let descriptor_set = Self::create_descriptor_set(context, &pipeline, &uniform_buffer, &hdr);

        let cube_attachment = Self::create_cube_attachment(context);
        let color_attachment = Self::create_color_attachment(context);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(&render_pass, &color_attachment_view);

        GenHdrCubemap {
            uniform_buffer,
            pipeline,
            vertex_buffer,
            descriptor_set,
            cube_attachment,
            cube_attachment_view,
            color_attachment,
            framebuffer,
        }
    }

    fn create_uniform_buffer(context: &Context) -> Arc<CpuAccessibleBuffer<Ubo>> {
        let identity = Mat4::IDENTITY.to_cols_array_2d();

        let uniform_buffer_data = Ubo {
            view: identity,
            projection: identity,
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
    ) {
        // camera buffer
        let camera_buffer_data = Arc::new(Ubo {
            view: matrix.to_cols_array_2d(),
            projection: Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 10.0)
                .to_cols_array_2d(),
        });

        builder
            .update_buffer(self.uniform_buffer.clone(), camera_buffer_data)
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
                    color: {
                        load: Clear,
                        store: Store,
                        format: FORMAT,
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

        let num_mips = (DIM.log2().floor() + 1.0) as u32;

        for m in 0..num_mips {
            for f in 0..6 {
                let width = DIM * 0.5_f32.powf(m as f32);
                let height = DIM * 0.5_f32.powf(m as f32);

                let viewport = Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [width, height],
                    depth_range: 0.0..1.0,
                };

                self.update_uniform_buffers(builder, mats[f]);

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
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        GraphicsPipeline::start()
            .vertex_input_single_buffer::<SkyboxVertex>()
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .front_face_counter_clockwise()
            .viewports_dynamic_scissors_irrelevant(1)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_descriptor_set<T>(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<Ubo>>,
        hdr: &Arc<T>,
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
            .add_sampled_image(hdr.clone(), context.attachment_sampler.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    fn create_color_attachment(context: &Context) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            context.device.clone(),
            [DIM as u32, DIM as u32],
            FORMAT,
            ImageUsage {
                color_attachment: true,
                transfer_source: true,
                ..ImageUsage::none()
            },
        )
        .unwrap()
    }

    fn create_cube_attachment(context: &Context) -> Arc<StorageImage> {
        let num_mips = (DIM.log2().floor() + 1.0) as u32;

        StorageImage::with_mipmaps_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: DIM as u32,
                height: DIM as u32,
                array_layers: 6,
            },
            FORMAT,
            MipmapsCount::Specific(num_mips),
            ImageUsage {
                transfer_destination: true,
                transfer_source: true,
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
        path: "src/renderer/shaders/gen_hdr_cubemap.vert"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/gen_hdr_cubemap.frag"
    }
}

type Ubo = vs::ty::UBO;

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
