use std::sync::Arc;

use vulkano::{
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageUsage, ImageViewAbstract},
    pipeline::{vertex::BuffersDefinition, viewport::Viewport, GraphicsPipeline},
    render_pass::{Framebuffer, RenderPass, Subpass},
};

use crate::FramebufferT;

use super::{context::Context, mesh::InstanceData, vertex::Vertex};

pub type GBufferTarget = Arc<ImageView<Arc<AttachmentImage>>>;

pub struct GBuffer {
    pub position_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    pub normals_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    pub albedo_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    pub depth_buffer: Arc<ImageView<Arc<AttachmentImage>>>,

    pub render_pass: Arc<RenderPass>,
    // pub lighting_subpass: Subpass,
    pub framebuffer: Arc<FramebufferT>,
    pub pipeline: Arc<GraphicsPipeline>,
}

impl GBuffer {
    pub fn initialize(context: &Context, target: &GBufferTarget) -> GBuffer {
        let position_buffer = Self::create_position_buffer(context, &target);
        let normals_buffer = Self::create_normals_buffer(context, &target);
        let albedo_buffer = Self::create_albedo_buffer(context, &target);
        let depth_buffer = Self::create_depth_buffer(context, &target);

        let render_pass = Self::create_render_pass(context);
        let framebuffer = Self::create_framebuffer(
            &render_pass,
            &position_buffer,
            &normals_buffer,
            &albedo_buffer,
            &depth_buffer,
        );

        let pipeline = Self::create_pipeline(context, &render_pass, &target);

        // let lighting_subpass = Subpass::from(render_pass.clone(), 1).unwrap();

        GBuffer {
            position_buffer,
            normals_buffer,
            albedo_buffer,
            depth_buffer,

            render_pass,
            // lighting_subpass,
            framebuffer,
            pipeline,
        }
    }

    fn create_framebuffer(
        render_pass: &Arc<RenderPass>,
        position_buffer: &Arc<ImageView<Arc<AttachmentImage>>>,
        normals_buffer: &Arc<ImageView<Arc<AttachmentImage>>>,
        albedo_buffer: &Arc<ImageView<Arc<AttachmentImage>>>,
        depth_buffer: &Arc<ImageView<Arc<AttachmentImage>>>,
    ) -> Arc<FramebufferT> {
        let framebuffer = Framebuffer::start(render_pass.clone())
            .add(position_buffer.clone())
            .unwrap()
            .add(normals_buffer.clone())
            .unwrap()
            .add(albedo_buffer.clone())
            .unwrap()
            .add(depth_buffer.clone())
            .unwrap()
            .build()
            .unwrap();

        Arc::new(framebuffer)
    }

    fn create_render_pass(context: &Context) -> Arc<RenderPass> {
        Arc::new(
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
                            depth: {
                                load: Clear,
                                store: DontCare,
                                format: context.depth_format,
                                samples: 1,
                            }
                        },
                pass: {
                    color: [position, normals, albedo],
                    depth_stencil: {depth}
                }
            )
            .unwrap(),
        )
    }

    fn create_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        target: &GBufferTarget,
    ) -> Arc<GraphicsPipeline> {
        let vs = super::shaders::model_vertex_shader::Shader::load(context.device.clone()).unwrap();
        let fs =
            super::shaders::model_fragment_shader::Shader::load(context.device.clone()).unwrap();

        let dimensions = target.image().dimensions().width_height();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input(
                    BuffersDefinition::new()
                        .vertex::<Vertex>()
                        .instance::<InstanceData>(),
                )
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
    }

    fn create_position_buffer(
        context: &Context,
        target: &GBufferTarget,
    ) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let (usage, dimensions) = Self::usage_dimensions(target);

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

    fn create_normals_buffer(
        context: &Context,
        target: &GBufferTarget,
    ) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let (usage, dimensions) = Self::usage_dimensions(target);

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

    fn create_albedo_buffer(
        context: &Context,
        target: &GBufferTarget,
    ) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let (usage, dimensions) = Self::usage_dimensions(target);

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

    fn create_depth_buffer(
        context: &Context,
        target: &GBufferTarget,
    ) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let (usage, dimensions) = Self::usage_dimensions(target);

        ImageView::new(
            AttachmentImage::with_usage(
                context.graphics_queue.device().clone(),
                dimensions,
                context.depth_format,
                usage,
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn usage_dimensions(target: &GBufferTarget) -> (ImageUsage, [u32; 2]) {
        let dimensions = target.image().dimensions().width_height();

        let usage = ImageUsage {
            // transient_attachment: true,
            // input_attachment: true,
            // NOTE: for debugging
            sampled: true,
            ..ImageUsage::none()
        };

        (usage, dimensions)
    }
}
