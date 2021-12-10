pub mod imgui_renderer;
pub mod renderer;

use std::{collections::HashMap, f32::consts::PI, sync::Arc, time::Instant};

use glam::Vec3;
use imgui_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use renderer::{
    camera::Camera, context::Context, scene::Scene, screen_frame::ScreenFrame,
    skybox_pass::SkyboxPass, vertex::Vertex,
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::Device,
    format::ClearValue,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        depth_stencil::DepthStencil, vertex::BuffersDefinition, viewport::Viewport,
        GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{acquire_next_image, AcquireError},
    sync::{self, GpuFuture},
};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta,
        VirtualKeyCode, WindowEvent,
    },
    event_loop::ControlFlow,
};

const MODEL_PATH: &str = "res/damaged_helmet/scene.gltf";
// const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/gcanyon_cube.ktx";
// const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/pisa_cube.ktx";
const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/uffizi_cube.ktx";

#[derive(Default, Copy, Clone)]
struct InstanceData {
    model: [[f32; 4]; 4],
}

vulkano::impl_vertex!(InstanceData, model);

pub struct OffscreenFramebuffer {
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
    resolve_image: Arc<ImageView<Arc<AttachmentImage>>>,
}

struct Application {
    context: Context,

    scene_graphics_pipeline: Arc<GraphicsPipeline>,
    scene_framebuffers: Vec<OffscreenFramebuffer>,
    scene_command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,

    screen_frame: ScreenFrame,

    skybox: SkyboxPass,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    last_time: Instant,

    scene: Scene,

    camera: Camera,

    imgui: imgui::Context,
    imgui_renderer: Renderer,
    platform: WinitPlatform,
}

impl Application {
    pub fn initialize() -> Self {
        let context = Context::initialize();

        let scene_render_pass = Self::create_scene_render_pass(&context);
        let scene_framebuffers = Self::create_scene_framebuffers(&context, &scene_render_pass);
        let scene_graphics_pipeline =
            Self::create_scene_graphics_pipeline(&context, &scene_render_pass);

        // TODO: we should load models without use of graphics_pipeline
        let scene = Scene::load(&context, MODEL_PATH, &scene_graphics_pipeline);

        let previous_frame_end = Some(Self::create_sync_objects(&context.device));

        let camera = Default::default();

        let skybox = SkyboxPass::initialize(&context, &scene_render_pass, SKYBOX_PATH);

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);

        imgui.io_mut().display_size = [
            context.swap_chain.dimensions()[0] as f32,
            context.swap_chain.dimensions()[1] as f32,
        ];

        let mut platform = WinitPlatform::init(&mut imgui);
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

        let imgui_renderer = Renderer::init(&context, &mut imgui).unwrap();

        let screen_frame =
            ScreenFrame::initialize(&context, &scene_framebuffers, &imgui_renderer.target);

        let mut app = Self {
            context,

            screen_frame,

            skybox,

            imgui,
            imgui_renderer,
            platform,

            scene_graphics_pipeline,
            scene_framebuffers,
            scene_command_buffers: vec![],

            previous_frame_end,
            recreate_swap_chain: false,

            last_time: Instant::now(),

            scene,

            camera,
        };

        app.create_scene_command_buffers();

        app
    }

    fn create_depth_image(context: &Context) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let image = AttachmentImage::multisampled_with_usage(
            context.device.clone(),
            context.swap_chain.dimensions(),
            context.sample_count,
            context.depth_format,
            ImageUsage {
                depth_stencil_attachment: true,
                ..ImageUsage::none()
            },
        )
        .unwrap();

        ImageView::new(image).unwrap()
    }

    fn create_color_image(context: &Context) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let image = AttachmentImage::multisampled_with_usage(
            context.device.clone(),
            context.swap_chain.dimensions(),
            context.sample_count,
            context.swap_chain.format(),
            ImageUsage {
                transient_attachment: true,
                ..ImageUsage::none()
            },
        )
        .unwrap();

        ImageView::new(image).unwrap()
    }

    /**
     * Creates render pass which has color and depth attachments.
     * Last attachment is resolve which can be attached to swap chain image used to output to screen.
     */
    fn create_scene_render_pass(context: &Context) -> Arc<RenderPass> {
        let color_format = context.swap_chain.format();
        let depth_format = context.depth_format;
        let sample_count = context.sample_count;

        Arc::new(
            single_pass_renderpass!(context.device.clone(),
                    attachments: {
                        multisample_color: {
                            load: Clear,
                            store: Store,
                            format: color_format,
                            samples: sample_count,
                        },
                        multisample_depth: {
                            load: Clear,
                            store: DontCare,
                            format: depth_format,
                            samples: sample_count,
                            initial_layout: ImageLayout::Undefined,
                            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                        },
                        resolve_color: {
                            load: DontCare,
                            store: Store,
                            format: color_format,
                            samples: 1,
                        }
                    },
                    pass: {
                        color: [multisample_color],
                        depth_stencil: {multisample_depth},
                        resolve: [resolve_color]
                    }
            )
            .unwrap(),
        )
    }

    fn recreate_swap_chain(&mut self) {
        self.context.recreate_swap_chain();

        let offscreen_render_pass = Self::create_scene_render_pass(&self.context);
        self.scene_framebuffers =
            Self::create_scene_framebuffers(&self.context, &offscreen_render_pass);
        self.scene_graphics_pipeline =
            Self::create_scene_graphics_pipeline(&self.context, &offscreen_render_pass);

        self.screen_frame.recreate_swap_chain(&self.context);

        self.create_scene_command_buffers();
    }

    /**
     * Creates graphics pipeline from the given render pass, and vertex/fragment shaders.
     */
    fn create_scene_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module =
            renderer::shaders::model_vertex_shader::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module =
            renderer::shaders::model_fragment_shader::Shader::load(context.device.clone()).unwrap();

        let dimensions_u32 = context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input(
                    BuffersDefinition::new()
                        .vertex::<Vertex>()
                        .instance::<InstanceData>(),
                )
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
                .front_face_counter_clockwise()
                // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
                .blend_pass_through()
                .depth_stencil(DepthStencil::simple_depth_test())
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device.clone())
                .unwrap(),
        );

        pipeline
    }

    /**
     * This function created frame buffer for each swap chain image.
     *
     * It contains 3 attachments (color, depth, resolve) where resolve is swap chain image which is used to output to screen.
     */
    fn create_scene_framebuffers(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Vec<OffscreenFramebuffer> {
        let depth_image = Self::create_depth_image(&context);
        let color_image = Self::create_color_image(&context);

        let mut framebuffers: Vec<OffscreenFramebuffer> = Vec::new();

        for _i in 0..context.swap_chain.num_images() {
            let resolve_image = ImageView::new(
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

            let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(color_image.clone())
                    .unwrap()
                    .add(depth_image.clone())
                    .unwrap()
                    .add(resolve_image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            );

            framebuffers.push(OffscreenFramebuffer {
                framebuffer,
                resolve_image,
            });
        }

        framebuffers
    }

    fn create_scene_command_buffers(&mut self) {
        let dimensions_u32 = self.context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let mut instance_data = Vec::new();
        let model = self.scene.models.get(0).unwrap();

        let count = 25;
        let start = -(count / 2);
        let end = count / 2;

        for x in start..end {
            for y in start..end {
                let position = Vec3::new(x as f32 * 2.0, y as f32 * 2.0, 1.5);
                let model = model.transform.get_model_matrix(position);

                instance_data.push(InstanceData {
                    model: model.to_cols_array_2d(),
                })
            }
        }

        let instance_data_buffer = CpuAccessibleBuffer::from_iter(
            self.context.device.clone(),
            BufferUsage::all(),
            false,
            instance_data.iter().cloned(),
        )
        .unwrap();

        let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

        for _i in 0..self.context.swap_chain.num_images() {
            let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                self.context.device.clone(),
                self.context.graphics_queue.family(),
                CommandBufferUsage::SimultaneousUse,
                self.scene_graphics_pipeline.subpass().clone(),
            )
            .unwrap();

            builder.set_viewport(0, [viewport.clone()]);

            builder.bind_pipeline_graphics(self.scene_graphics_pipeline.clone());

            for model in self.scene.models.iter() {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.scene_graphics_pipeline.layout().clone(),
                        0,
                        model.descriptor_set.clone(),
                    )
                    .bind_vertex_buffers(
                        0,
                        (model.vertex_buffer.clone(), instance_data_buffer.clone()),
                    )
                    .bind_index_buffer(model.index_buffer.clone())
                    .draw_indexed(
                        model.index_count,
                        instance_data_buffer.len() as u32,
                        0,
                        0,
                        0,
                    )
                    .unwrap();
            }

            let command_buffer = Arc::new(builder.build().unwrap());

            command_buffers.push(command_buffer);
        }

        self.scene_command_buffers = command_buffers;
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.context.swap_chain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swap_chain = true;
        }

        let offscreen_command_buffer = self.scene_command_buffers[image_index].clone();

        let command_buffer = self.screen_frame.command_buffers[image_index].clone();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.context.device.clone(),
            self.context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
        )
        .unwrap();

        let offscreen_framebuffer = self.scene_framebuffers[image_index].framebuffer.clone();
        let framebuffer = self.screen_frame.framebuffers[image_index].clone();

        let dimensions_u32 = self.context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        builder
            .update_buffer(
                self.skybox.uniform_buffer.clone(),
                Arc::new(self.camera.get_skybox_uniform_data(dimensions)),
            )
            .unwrap();

        // TODO: we don't need uniform buffer for each model - just one will be ok
        for model in &self.scene.models {
            let model_uniform_data = Arc::new(self.camera.get_model_uniform_data(dimensions));

            builder
                .update_buffer(model.uniform_buffer.clone(), model_uniform_data)
                .unwrap();
        }

        builder
            .begin_render_pass(
                offscreen_framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![
                    [0.0, 0.0, 0.0, 1.0].into(),
                    ClearValue::Depth(1.0),
                    ClearValue::None,
                ],
            )
            .unwrap()
            .execute_commands(self.skybox.command_buffer.clone())
            .unwrap()
            .execute_commands(offscreen_command_buffer)
            .unwrap()
            .end_render_pass()
            .unwrap();

        let ui = self.imgui.frame();
        let mut value = true;
        ui.show_demo_window(&mut value);

        let draw_data = ui.render();
        self.imgui_renderer
            .draw_commands(&mut builder, draw_data)
            .unwrap();

        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            )
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.context.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                // TODO: swap to present queue???
                self.context.graphics_queue.clone(),
                self.context.swap_chain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.context.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.context.device.clone())) as Box<_>);
            }
        }
    }

    fn update(&mut self, keys: &HashMap<VirtualKeyCode, ElementState>, dt: f32) {
        let camera_speed = 5.0 * dt;

        if is_pressed(keys, VirtualKeyCode::A) {
            self.camera.position += self.camera.right() * camera_speed
        }

        if is_pressed(keys, VirtualKeyCode::D) {
            self.camera.position -= self.camera.right() * camera_speed
        }

        if is_pressed(keys, VirtualKeyCode::W) {
            self.camera.position += self.camera.forward() * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::S) {
            self.camera.position -= self.camera.forward() * camera_speed;
        }
    }

    fn main_loop(mut self) {
        let mut mouse_buttons: HashMap<MouseButton, ElementState> = HashMap::new();
        let mut keyboard_buttons: HashMap<VirtualKeyCode, ElementState> = HashMap::new();

        let mut last_frame = Instant::now();

        self.context
            .event_loop
            .take()
            .unwrap()
            .run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;

                let imgui_io = self.imgui.io_mut();
                self.platform
                    .handle_event(imgui_io, self.context.surface.window(), &event);

                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        *control_flow = ControlFlow::Exit;
                    }

                    Event::WindowEvent {
                        event: WindowEvent::Resized(_),
                        ..
                    } => {
                        self.recreate_swap_chain = true;
                    }

                    Event::WindowEvent {
                        event: WindowEvent::MouseInput { state, button, .. },
                        ..
                    } if !imgui_io.want_capture_mouse => {
                        mouse_buttons.insert(button, state);
                    }

                    Event::WindowEvent {
                        event:
                            WindowEvent::MouseWheel {
                                delta: MouseScrollDelta::PixelDelta(position),
                                ..
                            },
                        ..
                    } if !imgui_io.want_capture_mouse => {
                        let y = position.y as f32;

                        for model in self.scene.models.iter_mut() {
                            model.transform.scale += y / 100.0;
                            model.transform.scale =
                                model.transform.scale.max(Vec3::new(0.1, 0.1, 0.1));
                        }
                    }

                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        state,
                                        virtual_keycode: Some(virtual_keycode),
                                        ..
                                    },
                                ..
                            },
                        ..
                    } if !imgui_io.want_capture_keyboard => {
                        keyboard_buttons.insert(virtual_keycode, state);
                    }

                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta, .. },
                        ..
                    } if !imgui_io.want_capture_mouse => {
                        match mouse_buttons.get(&MouseButton::Left) {
                            Some(&ElementState::Pressed) => {
                                let sensitivity = 0.1;
                                let (x, y) = delta;

                                self.camera.rotation.z += (x as f32) * sensitivity;
                                self.camera.rotation.y -= (y as f32) * sensitivity;

                                if self.camera.rotation.y > 89.0 {
                                    self.camera.rotation.y = 89.0;
                                } else if self.camera.rotation.y < -89.0 {
                                    self.camera.rotation.y = -89.0;
                                }
                            }

                            _ => {}
                        };
                    }

                    Event::NewEvents(_) => {
                        let now = Instant::now();
                        self.imgui.io_mut().update_delta_time(now - last_frame);
                        last_frame = now;
                    }

                    Event::MainEventsCleared => {
                        self.platform
                            .prepare_frame(self.imgui.io_mut(), &self.context.surface.window())
                            .expect("Failed to prepare frame");

                        self.context.surface.window().request_redraw();
                    }

                    Event::RedrawRequested { .. } => {
                        let now = Instant::now();
                        let delta_time = now.duration_since(self.last_time).as_secs_f32();

                        let fps = 1.0 / delta_time;

                        println!("fps: {}", fps);

                        self.last_time = now;

                        self.update(&keyboard_buttons, delta_time);

                        self.draw_frame();
                    }

                    _ => (),
                }
            })
    }
}

fn main() {
    let app = Application::initialize();
    app.main_loop();
}

fn is_pressed(keys: &HashMap<VirtualKeyCode, ElementState>, key: VirtualKeyCode) -> bool {
    match keys.get(&key) {
        Some(&ElementState::Pressed) => true,
        _ => false,
    }
}
