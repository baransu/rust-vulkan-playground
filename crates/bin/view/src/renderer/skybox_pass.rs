use std::{fs::File, io::BufReader, sync::Arc};

use glam::{Mat4, Vec3};
use image::codecs::hdr::HdrDecoder;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImageViewAbstract, ImmutableImage, MipmapsCount},
    pipeline::{
        graphics::{
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    shader::EntryPoint,
    sync::GpuFuture,
};

use super::{context::Context, shaders::CameraUniformBufferObject};

#[derive(Default, Debug, Clone, Copy)]
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
    // graphics_pipeline: Arc<GraphicsPipeline>,
    // vertex_buffer: Arc<ImmutableBuffer<[SkyboxVertex]>>,
    // descriptor_set: Arc<PersistentDescriptorSet>,
    pub uniform_buffer: Arc<CpuAccessibleBuffer<CameraUniformBufferObject>>,
    pub command_buffer: Arc<SecondaryAutoCommandBuffer>,
}

impl SkyboxPass {
    pub fn initialize<T>(
        context: &Context,
        render_pass: &Arc<RenderPass>,
        texture: &Arc<T>,
        fragment_shader_entry_point: EntryPoint,
        dimensions: [f32; 2],
    ) -> SkyboxPass
    where
        T: ImageViewAbstract + 'static,
    {
        let graphics_pipeline =
            Self::create_graphics_pipeline(context, &render_pass, fragment_shader_entry_point);

        let vertex_buffer = Self::create_vertex_buffer(context);

        let uniform_buffer = Self::create_uniform_buffer(context);

        let descriptor_set =
            Self::create_descriptor_set(context, &graphics_pipeline, &uniform_buffer, &texture);

        let command_buffer = Self::create_command_buffer(
            context,
            &graphics_pipeline,
            &descriptor_set,
            &vertex_buffer,
            dimensions,
        );

        SkyboxPass {
            uniform_buffer,
            command_buffer,
            // graphics_pipeline,
            // vertex_buffer,
            // descriptor_set,
        }
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
        dimensions: [f32; 2],
    ) -> Arc<SecondaryAutoCommandBuffer> {
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
        fs: EntryPoint,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(context.device.clone()).unwrap();

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<SkyboxVertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs, ())
            .rasterization_state(
                RasterizationState::new()
                    .cull_mode(CullMode::Back)
                    .front_face(FrontFace::Clockwise),
            )
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
            .add_sampled_image(image.clone(), context.attachment_sampler.clone())
            .unwrap();

        set_builder.build().unwrap()
    }

    pub fn create_vertex_buffer(context: &Context) -> Arc<ImmutableBuffer<[SkyboxVertex]>> {
        let (vertex_buffer, future) = ImmutableBuffer::from_iter(
            skybox_vertices().clone(),
            BufferUsage::vertex_buffer(),
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        vertex_buffer
    }

    pub fn load_skybox_texture(context: &Context, path: &str) -> Arc<ImageView<ImmutableImage>> {
        let buf_reader = HdrDecoder::new(BufReader::new(File::open(&path).unwrap())).unwrap();

        let metadata = buf_reader.metadata();

        let hdr = buf_reader
            .read_image_hdr()
            .unwrap()
            .iter()
            .flat_map(|v| [v.0[0], v.0[1], v.0[2], 1.0])
            .collect::<Vec<_>>();

        let width = metadata.width;
        let height = metadata.height;

        log::debug!(
            "Loading HDR texture: {}x{}, exposure: {:?}",
            width,
            height,
            metadata.exposure
        );

        let format = Format::R32G32B32A32_SFLOAT;

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            array_layers: 1,
        };

        let (immutable_image, future) = ImmutableImage::from_iter(
            hdr.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            format,
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ImageView::new(immutable_image).unwrap()
    }
}

fn skybox_vertices() -> [SkyboxVertex; 36] {
    [
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        //
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        //
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
        //
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        //
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, -1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        SkyboxVertex::new(1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, 1.0),
        SkyboxVertex::new(-1.0, -1.0, -1.0),
        //
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(1.0, 1.0, -1.0),
        SkyboxVertex::new(1.0, 1.0, 1.0),
        SkyboxVertex::new(-1.0, 1.0, -1.0),
        SkyboxVertex::new(-1.0, 1.0, 1.0),
    ]
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/renderer/shaders/skybox.vert"
    }
}

pub mod fs_gbuffer {
    vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
	#version 450

	layout (binding = 1) uniform samplerCube skybox_texture;
	
	layout (location = 0) in vec3 inUV;
	
	// it's gbuffer albedo
    layout(location = 0) out vec4 out_position;
    layout(location = 1) out vec3 out_normal;
    layout(location = 2) out vec4 out_albedo;
    layout(location = 3) out vec4 out_metallic_roughness;

	void main() {
		out_albedo = texture(skybox_texture, inUV);
        // just to make vulkan happy that we're not using attachments
        out_position = vec4(0.0, 0.0, 0.0, 0.0);
        out_normal = vec3(0.0, 0.0, 0.0);
        out_metallic_roughness = vec4(0.0, 0.0, 0.0, 0.0);
	}
"
    }
}

pub mod fs_local_probe {
    vulkano_shaders::shader! {
                    ty: "fragment",
                    src: "
	#version 450

	layout (binding = 1) uniform samplerCube skybox_texture;
	
	layout (location = 0) in vec3 inUV;
	
	// it's gbuffer albedo
    layout(location = 0) out vec4 out_albedo;

	void main() {
		out_albedo = texture(skybox_texture, inUV);
	}
"
    }
}
