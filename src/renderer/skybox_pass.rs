use std::{fs::File, io::BufReader, sync::Arc};

use glam::{Mat4, Vec3};
use ktx::Decoder;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    image::ImageViewAbstract,
    pipeline::{graphics::viewport::Viewport, GraphicsPipeline, Pipeline, PipelineBindPoint},
    render_pass::{RenderPass, Subpass},
    sync::GpuFuture,
};

use super::{context::Context, shaders::CameraUniformBufferObject, texture::Texture};

#[derive(Default, Debug, Clone)]
pub struct SkyboxVertex {
    position: [f32; 3],
}

impl SkyboxVertex {
    fn new(pos_x: f32, pos_y: f32, pos_z: f32) -> SkyboxVertex {
        SkyboxVertex {
            position: [pos_x, pos_y, pos_z],
        }
    }
}

vulkano::impl_vertex!(SkyboxVertex, position);

pub struct SkyboxPass {
    graphics_pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub command_buffer: Arc<SecondaryAutoCommandBuffer>,
}

impl SkyboxPass {
    pub fn initialize<T>(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        texture: &Arc<T>,
    ) -> SkyboxPass
    where
        T: ImageViewAbstract + 'static,
    {
        let graphics_pipeline = Self::create_graphics_pipeline(context, &render_pass);

        let vertex_buffer = Self::create_vertex_buffer(context);

        let uniform_buffer = Self::create_uniform_buffer(context);

        let descriptor_set =
            Self::create_descriptor_set(context, &graphics_pipeline, &uniform_buffer, &texture);

        let command_buffer = Self::create_command_buffer(
            context,
            &graphics_pipeline,
            &descriptor_set,
            &vertex_buffer,
        );

        SkyboxPass {
            uniform_buffer,
            graphics_pipeline,
            vertex_buffer,
            descriptor_set,
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

    pub fn create_uniform_buffer(
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
        let vert_shader_module = vs::load(context.device.clone()).unwrap();
        let frag_shader_module = fs::load(context.device.clone()).unwrap();

        // TODO: add that to context as util or something
        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        GraphicsPipeline::start()
            .vertex_input_single_buffer::<SkyboxVertex>()
            .vertex_shader(vert_shader_module.entry_point("main").unwrap(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.entry_point("main").unwrap(), ())
            .depth_clamp(false)
            .polygon_mode_fill() // = default
            .line_width(1.0) // = default
            .cull_mode_back()
            .front_face_counter_clockwise()
            .blend_pass_through()
            .viewports_dynamic_scissors_irrelevant(1)
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(context.device.clone())
            .unwrap()
    }

    fn create_descriptor_set<T>(
        context: &Context,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
        image: &Arc<T>,
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
            .add_sampled_image(image.clone(), context.image_sampler.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    pub fn create_vertex_buffer(context: &Context) -> Arc<ImmutableBuffer<[SkyboxVertex]>> {
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

    pub fn load_skybox_texture(context: &Context, path: &str) -> Texture {
        let ktx_file = BufReader::new(File::open(path).unwrap());
        let ktx: Decoder<BufReader<File>> = ktx::Decoder::new(ktx_file).unwrap();

        Texture::from_ktx(context, ktx)
    }
}

fn skybox_vertices() -> [SkyboxVertex; 36] {
    [
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
    ]
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/skybox.vert"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
	#version 450

	layout (binding = 1) uniform samplerCube skybox_texture;
	
	layout (location = 0) in vec3 inUV;
	
	// it's gbuffer albedo
	layout (location = 2) out vec4 outFragColor;

	void main() {
		vec3 uv = vec3(inUV.x, -inUV.y, inUV.z);
		outFragColor = texture(skybox_texture, uv);
	}
"
    }
}
