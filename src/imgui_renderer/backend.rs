use imgui::{DrawData, Ui};
use imgui_winit_support::{HiDpiMode, WinitPlatform};

use crate::renderer::context::Context;

pub struct ImguiBackend {
    platform: WinitPlatform,
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

        ImguiBackend { platform }
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

    pub fn prepare_frame<'a>(
        &mut self,
        context: &Context,
        imgui: &'a mut imgui::Context,
    ) -> imgui::Ui<'a> {
        self.platform
            .prepare_frame(imgui.io_mut(), context.surface.window())
            .unwrap();

        imgui.frame()
    }

    pub fn prepare_render(&mut self, context: &Context, ui: &imgui::Ui) {
        self.platform.prepare_render(ui, context.surface.window());
    }
}
