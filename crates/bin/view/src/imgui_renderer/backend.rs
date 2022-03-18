use std::sync::Arc;

use imgui::TextureId;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    image::{view::ImageView, AttachmentImage, ImageViewAbstract},
};

use crate::renderer::context::Context;

use super::{shaders::TextureUsage, ImguiRenderer};

pub struct ImguiBackend {
    platform: WinitPlatform,
    renderer: ImguiRenderer,
}

impl ImguiBackend {
    pub fn new(context: &Context, imgui: &mut imgui::Context) -> ImguiBackend {
        let mut platform = WinitPlatform::init(imgui);

        platform.attach_window(
            imgui.io_mut(),
            &context.surface.window(),
            HiDpiMode::Rounded,
        );

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;

        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    size_pixels: font_size,
                    ..imgui::FontConfig::default()
                }),
            }]);

        let renderer = ImguiRenderer::initialize(&context, imgui).unwrap();

        ImguiBackend { platform, renderer }
    }

    pub fn handle_event(
        &mut self,
        context: &Context,
        imgui: &mut imgui::Context,
        event: &winit::event::Event<'_, ()>,
    ) {
        self.platform
            .handle_event(imgui.io_mut(), context.surface.window(), event);
    }

    pub fn frame<'a>(
        &mut self,
        context: &Context,
        imgui: &'a mut imgui::Context,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        callback: impl FnOnce(&imgui::Ui<'_>),
    ) {
        self.platform
            .prepare_frame(imgui.io_mut(), context.surface.window())
            .unwrap();

        let ui = imgui.frame();

        callback(&ui);

        self.platform.prepare_render(&ui, context.surface.window());

        let draw_data = ui.render();

        self.renderer.draw_commands(builder, draw_data).unwrap();
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
        self.renderer.register_texture(context, image, usage)
    }

    pub fn ui_frame(&self) -> Arc<ImageView<AttachmentImage>> {
        self.renderer.target.clone()
    }
}
