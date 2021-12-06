use std::sync::Arc;

use ktx::include_ktx;
use vulkano::{
    buffer::{BufferUsage, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::{self, PersistentDescriptorSet},
    pipeline::{viewport::Viewport, GraphicsPipeline, PipelineBindPoint},
    render_pass::{self, RenderPass, Subpass},
    single_pass_renderpass,
    sync::GpuFuture,
};

use super::{context::Context, shaders::skybox_vertex_shader, texture::Texture};

#[derive(Default, Debug, Clone)]
struct SkyboxVertex {
    position: [f32; 3],
    // later we'll remove that
    // uv: [f32; 2],
}

impl SkyboxVertex {
    fn new(pos_x: f32, pos_y: f32, pos_z: f32, _uv_x: f32, _uv_y: f32) -> SkyboxVertex {
        SkyboxVertex {
            position: [pos_x, pos_y, pos_z],
            // uv: [uv_x, uv_y],
        }
    }
}

vulkano::impl_vertex!(SkyboxVertex, position);

pub struct SkyboxPass {
    graphics_pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    // texture: Texture,
    pub command_buffer: Arc<SecondaryAutoCommandBuffer>,
}

impl SkyboxPass {
    pub fn initialize(context: &Context, render_pass: &Arc<RenderPass>) -> SkyboxPass {
        // let render_pass = Self::create_render_pass(context);
        let graphics_pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let vertex_buffer = Self::create_vertex_buffer(context);

        let texture = Self::load_skybox_texture(context);

        let descriptor_set = Self::create_descriptor_set(&graphics_pipeline, &texture);

        let command_buffer = Self::create_command_buffer(
            context,
            &graphics_pipeline,
            &descriptor_set,
            &vertex_buffer,
        );

        SkyboxPass {
            graphics_pipeline,
            vertex_buffer,
            descriptor_set,
            // texture,
            command_buffer,
        }
    }

    pub fn recreate_swap_chain(&mut self, context: &Context) {
        self.command_buffer = Self::create_command_buffer(
            context,
            &self.graphics_pipeline,
            &self.descriptor_set,
            &self.vertex_buffer,
        );
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
        let vert_shader_module =
            super::shaders::skybox_vertex_shader::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module =
            super::shaders::skybox_fragment_shader::Shader::load(context.device.clone()).unwrap();

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
                .front_face_clockwise()
                // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
                .blend_pass_through()
                // .depth_stencil(DepthStencil::simple_depth_test())
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                // .build(context.device.clone())
                .with_auto_layout(context.device.clone(), |set_descs| {
                    // Modify the auto-generated layout by setting an immutable sampler to
                    // set 0 binding 0.
                    set_descs[0].set_immutable_samplers(0, [context.image_sampler.clone()]);
                })
                .unwrap(),
        );

        pipeline
    }

    fn create_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        texture: &Texture,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let image = texture.image.clone();

        // NOTE: this works because we're setting immutable sampler when creating GraphicsPipeline
        set_builder.add_image(image).unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    // fn create_render_pass(context: &Context) -> Arc<RenderPass> {
    //     let color_format = context.swap_chain.format();

    //     Arc::new(
    //         single_pass_renderpass!(context.device.clone(),
    //                         attachments: {
    //                                 color: {
    //                                         load: Clear,
    //                                         store: Store,
    //                                         format: color_format,
    //                                         samples: 1,
    //                                 }
    //                         },
    //                         pass: {
    //                                 color: [color],
    //                                 depth_stencil: {}
    //                         }
    //         )
    //         .unwrap(),
    //     )
    // }

    fn create_vertex_buffer(context: &Context) -> Arc<ImmutableBuffer<[SkyboxVertex]>> {
        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            skybox_vertices().clone(),
            BufferUsage::vertex_buffer(),
            // TODO: idealy it should be transfer queue?
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        vertex_buffer
    }

    fn load_skybox_texture(context: &Context) -> Texture {
        let image: ktx::Ktx<&[u8]> =
            include_ktx!("../../vulkan_asset_pack_gltf/textures/hdr/uffizi_cube.ktx");

        Texture::from_ktx(context, &image)
    }
}

fn skybox_vertices() -> [SkyboxVertex; 36] {
    [
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 0.0),
        SkyboxVertex::new(0.5, -0.5, -0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(0.5, 0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(-0.5, 0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 0.0),
        SkyboxVertex::new(-0.5, -0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(0.5, -0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 1.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 1.0),
        SkyboxVertex::new(-0.5, 0.5, 0.5, 0.0, 1.0),
        SkyboxVertex::new(-0.5, -0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(-0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(-0.5, 0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(-0.5, -0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(-0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(0.5, -0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(0.5, -0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(0.5, -0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, -0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(-0.5, -0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(-0.5, -0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(-0.5, 0.5, -0.5, 0.0, 1.0),
        SkyboxVertex::new(0.5, 0.5, -0.5, 1.0, 1.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(0.5, 0.5, 0.5, 1.0, 0.0),
        SkyboxVertex::new(-0.5, 0.5, 0.5, 0.0, 0.0),
        SkyboxVertex::new(-0.5, 0.5, -0.5, 0.0, 1.0),
    ]
}
