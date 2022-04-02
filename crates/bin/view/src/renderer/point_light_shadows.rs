use std::sync::Arc;

use glam::{Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewType},
        AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, MipmapsCount, StorageImage,
    },
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    single_pass_renderpass,
};

use super::{
    context::Context, entity::InstanceData, scene::Scene, shaders::CameraUniformBufferObject,
    vertex::Vertex,
};

const DIM: f32 = 512.0;
const FORMAT: Format = Format::R16G16B16A16_SFLOAT;

pub struct PointLightShadows {
    pub pipeline: Arc<GraphicsPipeline>,
    camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<StorageImage>>,
    target_attachment: Arc<AttachmentImage>,

    framebuffer: Arc<Framebuffer>,

    camera_descriptor_set: Arc<PersistentDescriptorSet>,
}

impl PointLightShadows {
    pub fn initialize(context: &Context) -> PointLightShadows {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);

        let cube_attachment = Self::create_cube_attachment(context);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let target_attachment = Self::create_color_attachment(context);
        let target_attachment_view = ImageView::new(target_attachment.clone()).unwrap();
        let framebuffer = Self::create_framebuffer(&context, &render_pass, &target_attachment_view);

        let camera_descriptor_set =
            Self::create_camera_descriptor_set(&pipeline, &camera_uniform_buffer);

        PointLightShadows {
            camera_uniform_buffer,
            pipeline,
            cube_attachment,
            cube_attachment_view,
            target_attachment,

            framebuffer,
            camera_descriptor_set,
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

    fn update_uniform_buffers(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        f: usize,
    ) {
        // NOTE: this have to be the same as light position in scene.rs
        let position = Vec3::new(1.0, 4.0, -1.0);

        let view = matrices(position)[f];

        let far_plane = 100.0;
        let mut proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, far_plane);

        proj.y_axis.y *= -1.0;

        // camera buffer
        let camera_buffer_data = Arc::new(CameraUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position: position.to_array(),
        });

        builder
            .update_buffer(self.camera_uniform_buffer.clone(), camera_buffer_data)
            .unwrap();
    }

    fn create_camera_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        camera_uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
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

        set_builder.build().unwrap()
    }

    fn create_framebuffer(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        target: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<Framebuffer> {
        let depth = Self::create_depth_attachment(context);

        Framebuffer::start(render_pass.clone())
            .add(target.clone())
            .unwrap()
            .add(depth)
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
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: context.depth_format,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
        )
        .unwrap()
    }

    pub fn add_to_builder(
        &self,
        context: &Context,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scene: &Scene,
    ) {
        let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        secondary_builder.bind_pipeline_graphics(self.pipeline.clone());

        scene.draw(
            context,
            &mut secondary_builder,
            &self.pipeline,
            |_material| self.camera_descriptor_set.clone(),
        );

        let model_command_buffer = Arc::new(secondary_builder.build().unwrap());

        for f in 0..6 {
            self.update_uniform_buffers(builder, f);

            builder
                .begin_render_pass(
                    self.framebuffer.clone(),
                    SubpassContents::SecondaryCommandBuffers,
                    vec![
                        ClearValue::Float([1.0, 1.0, 1.0, 1.0]),
                        ClearValue::Depth(1.0),
                    ],
                )
                .unwrap();

            builder
                .execute_commands(model_command_buffer.clone())
                .unwrap();

            builder.end_render_pass().unwrap();

            builder
                .copy_image(
                    self.target_attachment.clone(),
                    [0, 0, 0],
                    0,
                    0,
                    self.cube_attachment.clone(),
                    [0, 0, 0],
                    f as u32,
                    0,
                    [DIM as u32, DIM as u32, 1],
                    1,
                )
                .unwrap();
        }
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<InstanceData>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [DIM, DIM],
                    depth_range: 0.0..1.0,
                },
            ]))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_color_attachment(context: &Context) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            context.graphics_queue.device().clone(),
            [DIM as u32, DIM as u32],
            FORMAT,
            ImageUsage {
                transfer_source: true,
                ..ImageUsage::none()
            },
        )
        .unwrap()
    }

    fn create_depth_attachment(context: &Context) -> Arc<ImageView<AttachmentImage>> {
        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                [DIM as u32, DIM as u32],
                context.depth_format,
                ImageUsage {
                    transfer_source: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn create_cube_attachment(context: &Context) -> Arc<StorageImage> {
        StorageImage::with_mipmaps_usage(
            context.device.clone(),
            ImageDimensions::Dim2d {
                width: DIM as u32,
                height: DIM as u32,
                array_layers: 6,
            },
            FORMAT,
            MipmapsCount::One,
            ImageUsage {
                transfer_destination: true,
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
                                                                    src: "
			#version 450

            layout(binding = 0) uniform CameraUniformBufferObject {
                mat4 view;
                mat4 proj;
                vec3 position;
            } camera;

			// per vertex
			layout(location = 1) in vec3 position;

			// per instance
			layout(location = 4) in mat4 model; 

            layout(location = 0) out vec4 FragPos;

			void main() {
				gl_Position = camera.proj * camera.view * model * vec4(position, 1.0);
                FragPos = model * vec4(position, 1.0);
			}									
"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
                                    ty: "fragment",
                                    src: "
			#version 450

            layout(binding = 0) uniform CameraUniformBufferObject {
                mat4 view;
                mat4 proj;
                // this is point light position as well
                vec3 position;
            } camera;

            layout(location = 0) in vec4 FragPos;

            layout(location = 0) out vec4 color;
            
			void main() {
                const float farPlane = 100.0;
                float l = length(FragPos.xyz - camera.position) / farPlane;
                color = vec4(l, l, l, 1.0);
			}
"
    }
}

fn matrices(eye: Vec3) -> [Mat4; 6] {
    [
        // POSITIVE_X
        Mat4::look_at_rh(eye, eye + Vec3::new(1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // NEGATIVE_X
        Mat4::look_at_rh(eye, eye + Vec3::new(-1.0, 0.0, 0.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // POSITIVE_Y
        Mat4::look_at_rh(eye, eye + Vec3::new(0.0, -1.0, 0.0), -Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        // NEGATIVE_Y
        Mat4::look_at_rh(eye, eye + Vec3::new(0.0, 1.0, 0.0), Vec3::Z),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        // POSITIVE_Z
        Mat4::look_at_rh(eye, eye + Vec3::new(0.0, 0.0, 1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        // NEGATIVE_Z
        Mat4::look_at_rh(eye, eye + Vec3::new(0.0, 0.0, -1.0), -Vec3::Y),
        // glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3( 0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    ]
}
