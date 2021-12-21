use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::{layout::DescriptorSetLayout, PersistentDescriptorSet},
    format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewType},
        AttachmentImage, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage,
    },
    pipeline::{
        graphics::{vertex_input::BuffersDefinition, viewport::Viewport},
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::Filter,
    single_pass_renderpass,
};

use super::{
    context::Context,
    entity::InstanceData,
    gbuffer::GBuffer,
    model::Model,
    scene::Scene,
    shaders::CameraUniformBufferObject,
    skybox_pass::{fs_local_probe, SkyboxPass},
    vertex::Vertex,
};

const DIM: f32 = 1024.0;
const FORMAT: Format = Format::R16G16B16A16_SFLOAT;

pub struct LocalProbe {
    pipeline: Arc<GraphicsPipeline>,
    camera_uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub cube_attachment: Arc<StorageImage>,
    pub cube_attachment_view: Arc<ImageView<StorageImage>>,
    color_attachment: Arc<AttachmentImage>,
    framebuffer: Arc<Framebuffer>,

    camera_descriptor_set: Arc<PersistentDescriptorSet>,

    skybox: SkyboxPass,
}

impl LocalProbe {
    pub fn initialize(
        context: &Context,
        layout: &Arc<DescriptorSetLayout>,
        skybox_texture: &Arc<ImageView<StorageImage>>,
    ) -> LocalProbe {
        let render_pass = Self::create_render_pass(context);
        let pipeline = Self::create_graphics_pipeline(context, &render_pass, &layout);

        let camera_uniform_buffer = Self::create_camera_uniform_buffer(context);

        let cube_attachment = Self::create_cube_attachment(context);
        let color_attachment = Self::create_color_attachment(context);

        let cube_attachment_view = ImageView::start(cube_attachment.clone())
            .with_type(ImageViewType::Cube)
            .build()
            .unwrap();

        let color_attachment_view = ImageView::new(color_attachment.clone()).unwrap();

        let framebuffer = Self::create_framebuffer(context, &render_pass, &color_attachment_view);

        let camera_descriptor_set =
            Self::create_camera_descriptor_set(&pipeline, &camera_uniform_buffer);

        let skybox = SkyboxPass::initialize(
            context,
            &render_pass,
            &skybox_texture,
            fs_local_probe::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
        );

        LocalProbe {
            camera_uniform_buffer,
            pipeline,
            cube_attachment,
            cube_attachment_view,
            color_attachment,
            framebuffer,
            camera_descriptor_set,

            skybox,
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
        matrix: Mat4,
    ) {
        let position = Vec3::ZERO.to_array();

        let mut proj = Mat4::perspective_rh(90.0_f32.to_radians(), 1.0, 0.1, 10.0);

        proj.y_axis.y *= -1.0;

        // camera buffer
        let camera_buffer_data = Arc::new(CameraUniformBufferObject {
            view: matrix.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position,
        });

        builder
            .update_buffer(self.camera_uniform_buffer.clone(), camera_buffer_data)
            .unwrap();

        let skybox_buffer_data = Arc::new(CameraUniformBufferObject {
            view: Mat4::from_mat3(Mat3::from_mat4(matrix)).to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            position,
        });

        builder
            .update_buffer(self.skybox.uniform_buffer.clone(), skybox_buffer_data)
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

    fn create_light_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        scene: &Scene,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(2)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(scene.light_space_uniform_buffer.clone())
            .unwrap();

        set_builder
            .add_buffer(scene.light_uniform_buffer.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    fn create_framebuffer(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        target: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<Framebuffer> {
        let depth_attachment = Self::create_depth_attachment(context);

        Framebuffer::start(render_pass.clone())
            .add(target.clone())
            .unwrap()
            .add(depth_attachment)
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

    pub fn execute(&self, context: &Context, scene: &Scene) -> PrimaryAutoCommandBuffer {
        let mats = matrices();

        let light_descriptor_set = Self::create_light_descriptor_set(&self.pipeline, scene);
        let instance_data_buffers = scene.get_instance_data_buffers(&context);

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        let mut secondary_builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        secondary_builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .set_viewport(0, [viewport.clone()]);

        for model in scene.models.iter() {
            // if there is no instance_data_buffer it means we have 0 instances for this mesh
            if let Some(instance_data_buffer) = instance_data_buffers.get(&model.id) {
                self.draw_model(
                    model,
                    &mut secondary_builder,
                    instance_data_buffer,
                    &light_descriptor_set,
                );
            }
        }

        let model_command_buffer = Arc::new(secondary_builder.build().unwrap());

        let mut builder = AutoCommandBufferBuilder::primary(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
        )
        .unwrap();

        for f in 0..6 {
            self.update_uniform_buffers(&mut builder, mats[f]);

            builder
                .begin_render_pass(
                    self.framebuffer.clone(),
                    SubpassContents::SecondaryCommandBuffers,
                    vec![
                        ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
                        ClearValue::Depth(1.0),
                    ],
                )
                .unwrap();

            builder
                .execute_commands(self.skybox.command_buffer.clone())
                .unwrap();

            builder
                .execute_commands(model_command_buffer.clone())
                .unwrap();

            builder.end_render_pass().unwrap();

            builder
                .blit_image(
                    self.color_attachment.clone(),
                    [0, 0, 0],
                    [DIM as i32, DIM as i32, 1],
                    0,
                    0,
                    self.cube_attachment.clone(),
                    [0, 0, 0],
                    [DIM as i32, DIM as i32, 1],
                    f as u32,
                    0,
                    1,
                    Filter::Linear,
                )
                .unwrap();
        }

        builder.build().unwrap()
    }

    fn create_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        layout: &Arc<DescriptorSetLayout>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();
        let fs = fs::load(context.device.clone()).unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [DIM, DIM],
            depth_range: 0.0..1.0,
        };

        let pipeline_layout = GBuffer::create_pipeline_layout(
            &context,
            vec![
                Arc::new(vs.entry_point("main").unwrap()).as_ref(),
                Arc::new(fs.entry_point("main").unwrap()).as_ref(),
            ],
            layout,
        );

        GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .instance::<InstanceData>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_simple_depth()
            .cull_mode_back()
            .viewports_dynamic_scissors_irrelevant(1)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .with_pipeline_layout(context.device.clone(), pipeline_layout)
            .unwrap()
    }

    fn create_depth_attachment(context: &Context) -> Arc<ImageView<AttachmentImage>> {
        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                [DIM as u32, DIM as u32],
                context.depth_format,
                ImageUsage::none(),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn create_color_attachment(context: &Context) -> Arc<AttachmentImage> {
        AttachmentImage::with_usage(
            context.device.clone(),
            [DIM as u32, DIM as u32],
            FORMAT,
            ImageUsage {
                color_attachment: true,
                transfer_destination: true,
                transfer_source: true,
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
                array_layers: 6,
            },
            FORMAT,
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

    fn draw_model(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        instance_data_buffer: &Arc<CpuAccessibleBuffer<[InstanceData]>>,
        light_descriptor_set: &Arc<PersistentDescriptorSet>,
    ) {
        for node_index in model.root_nodes.iter() {
            self.draw_model_node(
                model,
                builder,
                *node_index,
                instance_data_buffer,
                light_descriptor_set,
            );
        }
    }

    fn draw_model_node(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        node_index: usize,
        instance_data_buffer: &Arc<CpuAccessibleBuffer<[InstanceData]>>,
        light_descriptor_set: &Arc<PersistentDescriptorSet>,
    ) {
        let node = model.nodes.get(node_index).unwrap();

        if let Some(mesh_index) = node.mesh {
            let mesh = model.meshes.get(mesh_index).unwrap();

            for primitive_index in mesh.primitives.iter() {
                self.draw_model_primitive(
                    model,
                    builder,
                    *primitive_index,
                    instance_data_buffer,
                    light_descriptor_set,
                );
            }
        }

        for child_index in node.children.iter() {
            self.draw_model_node(
                model,
                builder,
                *child_index,
                instance_data_buffer,
                light_descriptor_set,
            );
        }
    }

    fn draw_model_primitive(
        &self,
        model: &Model,
        builder: &mut AutoCommandBufferBuilder<SecondaryAutoCommandBuffer>,
        primitive_index: usize,
        instance_data_buffer: &Arc<CpuAccessibleBuffer<[InstanceData]>>,
        light_descriptor_set: &Arc<PersistentDescriptorSet>,
    ) {
        let primitive = model.primitives.get(primitive_index).unwrap();
        let material = model.materials.get(primitive.material).unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                (
                    self.camera_descriptor_set.clone(),
                    material.descriptor_set.clone(),
                    light_descriptor_set.clone(),
                ),
            )
            .bind_vertex_buffers(
                0,
                (
                    primitive.vertex_buffer.clone(),
                    instance_data_buffer.clone(),
                ),
            )
            .bind_index_buffer(primitive.index_buffer.clone())
            .draw_indexed(
                primitive.index_count,
                instance_data_buffer.len() as u32,
                0,
                0,
                0,
            )
            .unwrap();
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/local_probe.vert"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/renderer/shaders/local_probe.frag"
    }
}

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
