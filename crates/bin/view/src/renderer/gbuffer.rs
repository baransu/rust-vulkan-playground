use std::{
    collections::{hash_map::Entry, HashMap},
    sync::Arc,
};

use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::layout::{DescriptorSetDesc, DescriptorSetLayout, DescriptorSetLayoutError},
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, PipelineLayout,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    shader::{DescriptorRequirements, EntryPoint},
};

use super::{context::Context, entity::InstanceData, scene::Scene, shaders::gbuffer_shader};

pub type GBufferTarget = Arc<ImageView<AttachmentImage>>;

pub struct GBuffer {
    pub position_buffer: Arc<ImageView<AttachmentImage>>,
    pub normals_buffer: Arc<ImageView<AttachmentImage>>,
    pub albedo_buffer: Arc<ImageView<AttachmentImage>>,
    pub depth_buffer: Arc<ImageView<AttachmentImage>>,
    pub metalic_roughness_buffer: Arc<ImageView<AttachmentImage>>,

    pub render_pass: Arc<RenderPass>,
    pub framebuffer: Arc<Framebuffer>,
    pub pipeline: Arc<GraphicsPipeline>,

    layouts: Vec<Arc<DescriptorSetLayout>>,
}

impl GBuffer {
    pub fn initialize(context: &Context, layouts: &Vec<Arc<DescriptorSetLayout>>) -> GBuffer {
        let position_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        let normals_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        let albedo_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        let metalic_roughness_buffer =
            Self::create_attachment_image(context, Format::R8G8B8A8_UNORM);
        let depth_buffer = Self::create_depth_attachment(context);

        let render_pass = Self::create_render_pass(context);
        let framebuffer = Self::create_framebuffer(
            &render_pass,
            &position_buffer,
            &normals_buffer,
            &albedo_buffer,
            &metalic_roughness_buffer,
            &depth_buffer,
        );

        let pipeline = Self::create_pipeline(context, &layouts, &render_pass);

        GBuffer {
            position_buffer,
            normals_buffer,
            albedo_buffer,
            depth_buffer,
            metalic_roughness_buffer,

            render_pass,
            framebuffer,
            pipeline,

            layouts: layouts.clone(),
        }
    }

    pub fn recreate_swapchain(&mut self, context: &Context) {
        self.pipeline = Self::create_pipeline(context, &self.layouts, &self.render_pass);

        self.position_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        self.normals_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        self.albedo_buffer = Self::create_attachment_image(context, Format::R16G16B16A16_SFLOAT);
        self.metalic_roughness_buffer =
            Self::create_attachment_image(context, Format::R8G8B8A8_UNORM);
        self.depth_buffer = Self::create_depth_attachment(context);

        self.framebuffer = Self::create_framebuffer(
            &self.render_pass,
            &self.position_buffer,
            &self.normals_buffer,
            &self.albedo_buffer,
            &self.metalic_roughness_buffer,
            &self.depth_buffer,
        );
    }

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        position_buffer: &Arc<ImageView<AttachmentImage>>,
        normals_buffer: &Arc<ImageView<AttachmentImage>>,
        albedo_buffer: &Arc<ImageView<AttachmentImage>>,
        metalic_roughness_buffer: &Arc<ImageView<AttachmentImage>>,
        depth_buffer: &Arc<ImageView<AttachmentImage>>,
    ) -> Arc<Framebuffer> {
        Framebuffer::start(render_pass.clone())
            .add(position_buffer.clone())
            .unwrap()
            .add(normals_buffer.clone())
            .unwrap()
            .add(albedo_buffer.clone())
            .unwrap()
            .add(metalic_roughness_buffer.clone())
            .unwrap()
            .add(depth_buffer.clone())
            .unwrap()
            .build()
            .unwrap()
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(context.device.clone(),
            attachments: {
                        position: {
                            load: Clear,
                            store: Store,
                            format: Format::R16G16B16A16_SFLOAT,
                            samples: 1,
                        },
                        normals: {
                            load: Clear,
                            store: Store,
                            format: Format::R16G16B16A16_SFLOAT,
                            samples: 1,
                        },
                        albedo: {
                            load: Clear,
                            store: Store,
                            format: Format::R16G16B16A16_SFLOAT,
                            samples: 1,
                        },
                        metalic_roughness: {
                            load: Clear,
                            store: Store,
                            format: Format::R8G8B8A8_UNORM,
                            samples: 1,
                        },
                        depth: {
                            load: Clear,
                            store: Store,
                            format: context.depth_format,
                            samples: 1,
                        }
                    },
            pass: {
                color: [position, normals, albedo, metalic_roughness],
                depth_stencil: {depth}
            }
        )
        .unwrap()
    }

    fn create_pipeline(
        context: &Context,
        layouts: &Vec<Arc<DescriptorSetLayout>>,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = gbuffer_shader::load_vertex(context.device.clone()).unwrap();
        let fs = gbuffer_shader::load_fragment(context.device.clone()).unwrap();

        let pipeline_layout = Self::create_pipeline_layout(
            context,
            vec![
                Arc::new(fs.entry_point("main").unwrap()).as_ref(),
                Arc::new(vs.entry_point("main").unwrap()).as_ref(),
            ],
            &layouts,
        );

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let width = context.swapchain.dimensions()[0] as f32;
        let height = context.swapchain.dimensions()[1] as f32;

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().instance::<InstanceData>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [width, height],
                    depth_range: 0.0..1.0,
                },
            ]))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(subpass)
            .with_pipeline_layout(context.device.clone(), pipeline_layout)
            .unwrap()
    }

    fn create_attachment_image(
        context: &Context,
        format: Format,
    ) -> Arc<ImageView<AttachmentImage>> {
        let (usage, dimensions) = Self::usage_dimensions(context);

        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                dimensions,
                format,
                usage,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn create_depth_attachment(context: &Context) -> Arc<ImageView<AttachmentImage>> {
        let (usage, dimensions) = Self::usage_dimensions(context);

        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                dimensions,
                context.depth_format,
                ImageUsage {
                    depth_stencil_attachment: true,
                    ..usage
                },
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn usage_dimensions(context: &Context) -> (ImageUsage, [u32; 2]) {
        let dimensions = context.swapchain.dimensions();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        (usage, dimensions)
    }

    pub fn create_command_buffer(
        &self,
        context: &Context,
        scene: &Scene,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = AutoCommandBufferBuilder::secondary_graphics(
            context.device.clone(),
            context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
            self.pipeline.subpass().clone(),
        )
        .unwrap();

        builder.bind_pipeline_graphics(self.pipeline.clone());

        scene.draw(
            context,
            &mut builder,
            &self.pipeline,
            |(primitive, material)| {
                (
                    // TODO: this should one per scene - uber storage buffer,
                    primitive.descriptor_set.clone(),
                    scene.descriptor_set.clone(),
                    material.descriptor_set.clone(),
                )
            },
        );

        Arc::new(builder.build().unwrap())
    }

    // NOTE: this is utility and should be abstracted into something like geometry pass or something like this maybe?
    pub fn create_pipeline_layout(
        context: &Context,
        entries: Vec<&EntryPoint>,
        layouts: &Vec<Arc<DescriptorSetLayout>>,
    ) -> Arc<PipelineLayout> {
        // Produce `DescriptorRequirements` for each binding, by iterating over all shaders
        // and adding the requirements of each.
        let mut descriptor_requirements: HashMap<(u32, u32), DescriptorRequirements> =
            HashMap::default();

        for (loc, reqs) in entries
            .iter()
            .map(|shader| shader.descriptor_requirements())
            .flatten()
        {
            match descriptor_requirements.entry(loc) {
                Entry::Occupied(entry) => {
                    // Previous shaders already added requirements, so we produce the
                    // intersection of the previous requirements and those of the
                    // current shader.
                    let previous = entry.into_mut();
                    *previous = previous.intersection(reqs).expect(
                        "Could not produce an intersection of the shader descriptor requirements",
                    );
                }
                Entry::Vacant(entry) => {
                    // No previous shader had this descriptor yet, so we just insert the
                    // requirements.
                    entry.insert(reqs.clone());
                }
            }
        }

        let descriptor_set_layout_descs = DescriptorSetDesc::from_requirements(
            descriptor_requirements
                .iter()
                .map(|(&loc, reqs)| (loc, reqs)),
        );

        let descriptor_set_layouts = descriptor_set_layout_descs
            .into_iter()
            .enumerate()
            .map(|(index, desc)| {
                if let Some(layout) = layouts.get(index) {
                    Ok(layout.clone())
                } else {
                    Ok(DescriptorSetLayout::new(context.device.clone(), desc.clone()).unwrap())
                }
            })
            .collect::<Result<Vec<_>, DescriptorSetLayoutError>>()
            .unwrap();

        PipelineLayout::new(context.device.clone(), descriptor_set_layouts, []).unwrap()
    }
}
