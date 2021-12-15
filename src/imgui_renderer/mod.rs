pub mod shaders;

use vulkano::buffer::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{BufferAccess, BufferUsage, CpuBufferPool},
    command_buffer::{PrimaryAutoCommandBuffer, SubpassContents},
    image::{view::ImageView, ImageDimensions, ImageViewAbstract},
};

use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageUsage, ImmutableImage, StorageImage};
use vulkano::pipeline::viewport::Scissor;
use vulkano::pipeline::viewport::Viewport;
use vulkano::render_pass::Subpass;
use vulkano::render_pass::{Framebuffer, FramebufferAbstract};
use vulkano::sampler::Sampler;

use std::fmt;
use std::sync::Arc;

use imgui::{internal::RawWrapper, DrawCmd, DrawCmdParams, DrawVert, TextureId, Textures};

use crate::renderer::context::Context;

use self::shaders::TextureUsage;

#[derive(Default, Debug, Clone)]
#[repr(C)]
struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub col: u32,
}

vulkano::impl_vertex!(Vertex, pos, uv, col);

impl From<DrawVert> for Vertex {
    fn from(v: DrawVert) -> Vertex {
        unsafe { std::mem::transmute(v) }
    }
}

#[derive(Debug)]
pub enum RendererError {
    BadTexture(TextureId),
    BadImageDimensions(ImageDimensions),
}

impl fmt::Display for RendererError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            &Self::BadTexture(ref t) => {
                write!(f, "The Texture ID could not be found: {:?}", t)
            }
            &Self::BadImageDimensions(d) => {
                write!(f, "Image Dimensions not supported (must be Dim2d): {:?}", d)
            }
        }
    }
}

impl std::error::Error for RendererError {}

pub type Texture = (
    Arc<dyn ImageViewAbstract + 'static>,
    Arc<Sampler>,
    Arc<ImmutableBuffer<TextureUsage>>,
);

pub struct Renderer {
    pipeline: Arc<GraphicsPipeline>,
    font_texture: Texture,
    textures: Textures<Texture>,
    vrt_buffer_pool: CpuBufferPool<Vertex>,
    idx_buffer_pool: CpuBufferPool<u16>,
    pub target: Arc<ImageView<Arc<AttachmentImage>>>,
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
}

impl Renderer {
    /// Initialize the renderer object, including vertex buffers, ImGui font textures,
    /// and the Vulkan graphics pipeline.
    ///
    /// ---
    ///
    /// `ctx`: the ImGui `Context` object
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on.
    ///
    /// `queue`: the Vulkano `Queue` object for the queue the font atlas texture will be created on.
    ///
    /// `format`: the Vulkano `Format` that the render pass will use when storing the frame in the target image.
    pub fn init(
        context: &Context,
        imgui: &mut imgui::Context,
    ) -> Result<Renderer, Box<dyn std::error::Error>> {
        let vs = shaders::vs::Shader::load(context.device.clone()).unwrap();
        let fs = shaders::fs::Shader::load(context.device.clone()).unwrap();

        let format = context.swap_chain.format();
        let render_pass = Arc::new(
            vulkano::single_pass_renderpass!(
                context.device.clone(),
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
            .unwrap(),
        );

        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports(vec![viewport])
                .viewports_scissors_dynamic(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())?,
        );

        let textures = Textures::new();

        let font_texture = Self::upload_font_texture(context, imgui.fonts())?;

        let vrt_buffer_pool = CpuBufferPool::new(
            context.device.clone(),
            BufferUsage::vertex_buffer_transfer_destination(),
        );
        let idx_buffer_pool = CpuBufferPool::new(
            context.device.clone(),
            BufferUsage::index_buffer_transfer_destination(),
        );

        let target = ImageView::new(
            AttachmentImage::with_usage(
                context.device.clone(),
                context.swap_chain.dimensions(),
                context.swap_chain.format(),
                ImageUsage {
                    sampled: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffer = Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(target.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        Ok(Renderer {
            pipeline,
            font_texture,
            textures,
            vrt_buffer_pool,
            idx_buffer_pool,
            target,
            framebuffer,
        })
    }

    /// Appends the draw commands for the UI frame to an `AutoCommandBufferBuilder`.
    ///
    /// ---
    ///
    /// `cmd_buf_builder`: An `AutoCommandBufferBuilder` from vulkano to add commands to
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on
    ///
    /// `queue`: the Vulkano `Queue` object for buffer creation
    ///
    /// `target`: the target image to render to
    ///
    /// `draw_data`: the ImGui `DrawData` that each UI frame creates
    pub fn draw_commands(
        &mut self,
        cmd_buf_builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &imgui::DrawData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fb_width = draw_data.display_size[0] * draw_data.framebuffer_scale[0];
        let fb_height = draw_data.display_size[1] * draw_data.framebuffer_scale[1];
        if !(fb_width > 0.0 && fb_height > 0.0) {
            return Ok(());
        }
        let left = draw_data.display_pos[0];
        let right = draw_data.display_pos[0] + draw_data.display_size[0];
        let top = draw_data.display_pos[1];
        let bottom = draw_data.display_pos[1] + draw_data.display_size[1];

        let pc = shaders::vs::ty::VertPC {
            matrix: [
                [(2.0 / (right - left)), 0.0, 0.0, 0.0],
                [0.0, (2.0 / (bottom - top)), 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [
                    (right + left) / (left - right),
                    (top + bottom) / (top - bottom),
                    0.0,
                    1.0,
                ],
            ],
        };

        let dimensions = self.framebuffer.dimensions();
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0..1.0,
        };

        let clip_off = draw_data.display_pos;
        let clip_scale = draw_data.framebuffer_scale;

        let layout = self
            .pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        cmd_buf_builder.begin_render_pass(
            self.framebuffer.clone(),
            SubpassContents::Inline,
            vec![[0.0, 0.0, 0.0, 0.0].into()],
        )?;

        cmd_buf_builder.set_viewport(0, [viewport.clone()]);

        cmd_buf_builder.bind_pipeline_graphics(self.pipeline.clone());

        for draw_list in draw_data.draw_lists() {
            let vertex_buffer = Arc::new(
                self.vrt_buffer_pool
                    .chunk(draw_list.vtx_buffer().iter().map(|&v| Vertex::from(v)))
                    .unwrap(),
            );

            let index_buffer = Arc::new(
                self.idx_buffer_pool
                    .chunk(draw_list.idx_buffer().iter().cloned())
                    .unwrap(),
            );

            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements {
                        count,
                        cmd_params:
                            DrawCmdParams {
                                clip_rect,
                                texture_id,
                                // vtx_offset,
                                idx_offset,
                                ..
                            },
                    } => {
                        let clip_rect = [
                            (clip_rect[0] - clip_off[0]) * clip_scale[0],
                            (clip_rect[1] - clip_off[1]) * clip_scale[1],
                            (clip_rect[2] - clip_off[0]) * clip_scale[0],
                            (clip_rect[3] - clip_off[1]) * clip_scale[1],
                        ];

                        if clip_rect[0] < fb_width
                            && clip_rect[1] < fb_height
                            && clip_rect[2] >= 0.0
                            && clip_rect[3] >= 0.0
                        {
                            let scissors = Scissor {
                                origin: [
                                    f32::max(0.0, clip_rect[0]).floor() as u32,
                                    f32::max(0.0, clip_rect[1]).floor() as u32,
                                ],
                                dimensions: [
                                    (clip_rect[2] - clip_rect[0]).abs().ceil() as u32,
                                    (clip_rect[3] - clip_rect[1]).abs().ceil() as u32,
                                ],
                            };

                            let (texture_image, texture_sampler, usage) =
                                self.lookup_texture(texture_id)?;

                            let mut set_builder = PersistentDescriptorSet::start(layout.clone());

                            set_builder.add_sampled_image(
                                texture_image.clone(),
                                texture_sampler.clone(),
                            )?;

                            set_builder.add_buffer(usage.clone())?;

                            let set = Arc::new(set_builder.build()?);

                            cmd_buf_builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Graphics,
                                    self.pipeline.layout().clone(),
                                    0,
                                    set,
                                )
                                .bind_vertex_buffers(0, vertex_buffer.clone())
                                .bind_index_buffer(
                                    index_buffer
                                        .clone()
                                        .into_buffer_slice()
                                        .slice(
                                            idx_offset as u64..(idx_offset as u64 + count as u64),
                                        )
                                        .unwrap(),
                                )
                                .push_constants(self.pipeline.layout().clone(), 0, pc)
                                .set_scissor(0, [scissors.clone()])
                                .draw_indexed(count as u32, 1, 0, 0, 0)?;
                        }
                    }
                    DrawCmd::ResetRenderState => (), // TODO
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd)
                    },
                }
            }
        }
        cmd_buf_builder.end_render_pass()?;

        Ok(())
    }

    /// Update the ImGui font atlas texture.
    ///
    /// ---
    ///
    /// `ctx`: the ImGui `Context` object
    ///
    /// `device`: the Vulkano `Device` object for the device you want to render the UI on.
    ///
    /// `queue`: the Vulkano `Queue` object for the queue the font atlas texture will be created on.
    pub fn reload_font_texture(
        &mut self,
        context: &Context,
        ctx: &mut imgui::Context,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.font_texture = Self::upload_font_texture(context, ctx.fonts())?;
        Ok(())
    }

    /// Get the texture library that the renderer uses
    pub fn textures(&mut self) -> &mut Textures<Texture> {
        &mut self.textures
    }

    pub fn register_texture<T>(
        &mut self,
        context: &Context,
        image: &Arc<T>,
        usage: TextureUsage,
    ) -> Result<TextureId, Box<dyn std::error::Error>>
    where
        T: ImageViewAbstract + 'static,
    {
        let sampler = Sampler::simple_repeat_linear_no_mipmap(context.device.clone());

        let usage = Self::create_texture_usage_buffer(context, usage);
        let texture_id = self.textures.insert((image.clone(), sampler, usage));

        Ok(texture_id)
    }

    fn upload_font_texture(
        context: &Context,
        mut fonts: imgui::FontAtlasRefMut,
    ) -> Result<Texture, Box<dyn std::error::Error>> {
        let texture = fonts.build_rgba32_texture();

        let (image, fut) = ImmutableImage::from_iter(
            texture.data.iter().cloned(),
            ImageDimensions::Dim2d {
                width: texture.width,
                height: texture.height,
                array_layers: 1,
            },
            vulkano::image::MipmapsCount::One,
            Format::R8G8B8A8_SRGB,
            context.graphics_queue.clone(),
        )?;

        fut.then_signal_fence_and_flush()?.wait(None)?;

        let sampler = Sampler::simple_repeat_linear(context.device.clone());

        fonts.tex_id = TextureId::from(usize::MAX);

        let ctx = Self::create_texture_usage_buffer(
            context,
            TextureUsage {
                depth: 0,
                normal: 0,
                position: 0,
                rgb: 0,
            },
        );

        Ok((ImageView::new(image)?, sampler, ctx))
    }

    fn lookup_texture(&self, texture_id: TextureId) -> Result<&Texture, RendererError> {
        if texture_id.id() == usize::MAX {
            Ok(&self.font_texture)
        } else if let Some(texture) = self.textures.get(texture_id) {
            Ok(texture)
        } else {
            Err(RendererError::BadTexture(texture_id))
        }
    }

    fn create_texture_usage_buffer(
        context: &Context,
        data: TextureUsage,
    ) -> Arc<ImmutableBuffer<TextureUsage>> {
        let (buffer, future) = ImmutableBuffer::from_data(
            data,
            BufferUsage::uniform_buffer_transfer_destination(),
            context.graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }
}
