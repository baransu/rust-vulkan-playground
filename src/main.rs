pub mod renderer;

use std::{collections::HashMap, f32::consts::PI, sync::Arc, time::Instant};

use glam::Vec3;
use renderer::{
    camera::Camera, context::Context, scene::Scene, screen_frame::ScreenFrame,
    skybox_pass::SkyboxPass, vertex::Vertex,
};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::Device,
    format::ClearValue,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    pipeline::{
        depth_stencil::DepthStencil, viewport::Viewport, GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{acquire_next_image, AcquireError},
    sync::{self, GpuFuture},
};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ControlFlow,
};

const MODEL_PATH: &str = "res/damaged_helmet/scene.gltf";
// const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/gcanyon_cube.ktx";
// const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/pisa_cube.ktx";
const SKYBOX_PATH: &str = "vulkan_asset_pack_gltf/textures/hdr/uffizi_cube.ktx";

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

        let screen_frame = ScreenFrame::initialize(&context, &scene_framebuffers);

        let skybox = SkyboxPass::initialize(&context, &scene_render_pass, SKYBOX_PATH);

        let mut app = Self {
            context,

            screen_frame,

            skybox,

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
                .vertex_input_single_buffer::<Vertex>()
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
                    .bind_vertex_buffers(0, model.vertex_buffer.clone())
                    .bind_index_buffer(model.index_buffer.clone())
                    .draw_indexed(model.index_count, 1, 0, 0, 0)
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
                Arc::new(self.camera.get_skybox_mvp_ubo(dimensions)),
            )
            .unwrap();

        for model in &self.scene.models {
            let data = Arc::new(model.transform.get_mvp_ubo(&self.camera, dimensions));

            builder
                .update_buffer(model.uniform_buffer.clone(), data)
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

    fn main_loop(mut self) {
        let mut mouse_buttons: HashMap<MouseButton, ElementState> = HashMap::new();

        self.context
            .event_loop
            .take()
            .unwrap()
            .run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;

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
                    } => {
                        mouse_buttons.insert(button, state);
                    }

                    // // on key press
                    // Event::WindowEvent {
                    //     event:
                    //         WindowEvent::KeyboardInput {
                    //             input:
                    //                 KeyboardInput {
                    //                     virtual_keycode,
                    //                     state: ElementState::Pressed,
                    //                     ..
                    //                 },
                    //             ..
                    //         },
                    //     ..
                    // } => {
                    // }
                    Event::WindowEvent {
                        event:
                            WindowEvent::MouseWheel {
                                delta: MouseScrollDelta::PixelDelta(position),
                                ..
                            },
                        ..
                    } => {
                        let y = position.y as f32;

                        for model in self.scene.models.iter_mut() {
                            model.transform.scale += y / 100.0;
                            model.transform.scale =
                                model.transform.scale.max(Vec3::new(0.1, 0.1, 0.1));
                        }
                    }

                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta, .. },
                        ..
                    } => {
                        match mouse_buttons.get(&MouseButton::Left) {
                            Some(&ElementState::Pressed) => {
                                let screen_width = self.context.swap_chain.dimensions()[0] as f32;
                                let screen_height = self.context.swap_chain.dimensions()[1] as f32;

                                let theta = 2.0 * PI * (delta.0 as f32) / screen_width;
                                let phi = 2.0 * PI * (delta.1 as f32) / screen_height;

                                self.camera.theta -= theta;

                                self.camera.phi = (self.camera.phi - phi)
                                    .clamp(10.0_f32.to_radians(), 170.0_f32.to_radians());
                            }

                            _ => {}
                        };
                    }

                    Event::MainEventsCleared { .. } => {
                        let now = Instant::now();
                        let delta_time = now.duration_since(self.last_time).as_secs_f32();

                        let fps = 1.0 / delta_time;

                        println!("fps: {}", fps);

                        self.last_time = now;

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
