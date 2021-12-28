use std::{collections::HashMap, sync::Arc, time::Instant, vec};

use glam::{Quat, Vec3};
use imgui::{Condition, Image, TextureId, Window};
use puffin_imgui::ProfilerUi;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::layout::{
        DescriptorDesc, DescriptorSetDesc, DescriptorSetLayout, DescriptorType,
    },
    device::Device,
    format::{ClearValue, Format},
    image::{view::ImageView, ImmutableImage, MipmapsCount},
    shader::ShaderStages,
    swapchain::{acquire_next_image, AcquireError},
    sync::{self, GpuFuture},
};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

use crate::{
    imgui_renderer::{backend::ImguiBackend, shaders::TextureUsage},
    renderer::skybox_pass::fs_gbuffer,
};

use super::{
    camera::Camera,
    context::Context,
    cubemap_gen_pass::{irradiance_convolution_fs, prefilterenvmap_fs, CubemapGenPass},
    dir_light_shadows::DirLightShadows,
    entity::Entity,
    gbuffer::GBuffer,
    gen_hdr_cubemap::GenHdrCubemap,
    light_system::LightSystem,
    point_light_shadows::PointLightShadows,
    scene::Scene,
    screen_frame::ScreenFrame,
    skybox_pass::SkyboxPass,
    ssao::Ssao,
    ssao_blur::SsaoBlur,
    texture::Texture,
};

const RENDER_SKYBOX: bool = true;

const BRDF_PATH: &str = "res/ibl_brdf_lut.png";

pub struct RenderContext {
    context: Context,
    gbuffer: GBuffer,
    light_system: LightSystem,

    screen_frame: ScreenFrame,

    skybox: SkyboxPass,

    scene: Scene,

    camera: Camera,

    gbuffer_position_texture_id: TextureId,
    gbuffer_normals_texture_id: TextureId,
    gbuffer_albedo_texture_id: TextureId,
    gbuffer_metalic_texture_id: TextureId,
    ssao_texture_id: TextureId,
    dir_shadow_texture_id: TextureId,

    ssao: Ssao,
    ssao_blur: SsaoBlur,

    gen_hdr_cubemap: GenHdrCubemap,

    irradiance_convolution: CubemapGenPass,
    prefilterenvmap: CubemapGenPass,

    point_light_shadows: PointLightShadows,
    dir_light_shadows: DirLightShadows,

    brdf_texture: Arc<ImageView<ImmutableImage>>,
}

pub struct Renderer {
    rc: RenderContext,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    last_time: Instant,

    imgui: imgui::Context,
    imgui_backend: ImguiBackend,

    event_loop: EventLoop<()>,

    prebuild: bool,
    puffin_ui: ProfilerUi,
}

impl Renderer {
    pub fn initialize(mesh_paths: Vec<&str>, skybox_path: &str) -> Self {
        // let mut rng = rand::thread_rng();
        let (context, event_loop) = Context::initialize();

        // TODO: generate brdf texture instead of loading it - why I have black spots???
        let brdf_texture = Texture::create_image_view(
            &context.graphics_queue,
            &image::io::Reader::open(BRDF_PATH)
                .unwrap()
                .decode()
                .unwrap(),
            Format::R8G8B8A8_UNORM,
            MipmapsCount::One,
        );

        let camera_layout = DescriptorSetLayout::new(
            context.device.clone(),
            DescriptorSetDesc::new([
                // camera
                Some(DescriptorDesc {
                    ty: DescriptorType::UniformBuffer,
                    descriptor_count: 1,
                    variable_count: false,
                    stages: ShaderStages {
                        fragment: true,
                        vertex: true,
                        ..ShaderStages::none()
                    },
                    immutable_samplers: vec![],
                }),
            ]),
        )
        .unwrap();

        let materials_layout = DescriptorSetLayout::new(
            context.device.clone(),
            DescriptorSetDesc::new([
                // diffuse texture
                Some(DescriptorDesc {
                    ty: DescriptorType::CombinedImageSampler,
                    descriptor_count: 1,
                    variable_count: false,
                    stages: ShaderStages {
                        fragment: true,
                        ..ShaderStages::none()
                    },
                    immutable_samplers: vec![context.image_sampler.clone()],
                }),
                // normal texture
                Some(DescriptorDesc {
                    ty: DescriptorType::CombinedImageSampler,
                    descriptor_count: 1,
                    variable_count: false,
                    stages: ShaderStages {
                        fragment: true,
                        ..ShaderStages::none()
                    },
                    immutable_samplers: vec![context.image_sampler.clone()],
                }),
                // metallic roughness texture
                Some(DescriptorDesc {
                    ty: DescriptorType::CombinedImageSampler,
                    descriptor_count: 1,
                    variable_count: false,
                    stages: ShaderStages {
                        fragment: true,
                        ..ShaderStages::none()
                    },
                    immutable_samplers: vec![context.image_sampler.clone()],
                }),
            ]),
        )
        .unwrap();

        let gbuffer = GBuffer::initialize(&context, &materials_layout);

        let gen_hdr_cubemap = GenHdrCubemap::initialize(&context, skybox_path);

        let point_light_shadows = PointLightShadows::initialize(&context);
        let dir_light_shadows = DirLightShadows::initialize(&context);

        let scene = Scene::initialize(&context, mesh_paths, &camera_layout, &materials_layout);

        let ssao = Ssao::initialize(&context, &scene.camera_uniform_buffer, &gbuffer);
        let ssao_blur = SsaoBlur::initialize(&context, &ssao.target);

        let irradiance_convolution_fs_mod =
            irradiance_convolution_fs::load(context.device.clone()).unwrap();
        let irradiance_convolution = CubemapGenPass::initialize(
            &context,
            &gen_hdr_cubemap.cube_attachment_view,
            irradiance_convolution_fs_mod.entry_point("main").unwrap(),
            Format::R32G32B32A32_SFLOAT,
            64.0,
        );

        let prefilterenvmap_fs_mod = prefilterenvmap_fs::load(context.device.clone()).unwrap();
        let prefilterenvmap = CubemapGenPass::initialize(
            &context,
            &gen_hdr_cubemap.cube_attachment_view,
            prefilterenvmap_fs_mod.entry_point("main").unwrap(),
            Format::R16G16B16A16_SFLOAT,
            512.0,
        );

        let skybox = SkyboxPass::initialize(
            &context,
            &gbuffer.render_pass,
            &gen_hdr_cubemap.cube_attachment_view,
            fs_gbuffer::load(context.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
            [
                context.swap_chain.dimensions()[0] as f32,
                context.swap_chain.dimensions()[1] as f32,
            ],
        );

        let light_system = LightSystem::initialize(
            &context,
            &scene.camera_uniform_buffer,
            &scene.light_uniform_buffer,
            &gbuffer,
            &ssao_blur.target,
            &irradiance_convolution.cube_attachment_view,
            &prefilterenvmap.cube_attachment_view,
            &brdf_texture,
            &point_light_shadows.cube_attachment_view,
            &dir_light_shadows.target_attachment,
        );

        let previous_frame_end = Some(Self::create_sync_objects(&context.device));

        let camera = Default::default();

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);

        imgui.io_mut().display_size = [
            context.swap_chain.dimensions()[0] as f32,
            context.swap_chain.dimensions()[1] as f32,
        ];

        let mut imgui_backend = ImguiBackend::new(&context, &mut imgui);

        let screen_frame =
            ScreenFrame::initialize(&context, &light_system.target, &imgui_backend.ui_frame());

        let gbuffer_position_texture_id = imgui_backend
            .register_texture(
                &context,
                &gbuffer.position_buffer,
                TextureUsage {
                    depth: 0,
                    normal: 0,
                    position: 1,
                    rgb: 0,
                },
            )
            .unwrap();

        let gbuffer_normals_texture_id = imgui_backend
            .register_texture(
                &context,
                &gbuffer.normals_buffer,
                TextureUsage {
                    depth: 0,
                    // NOTE: they are not normals but also are in [-1, 1] space
                    // so we need to convert them to [0,1]
                    normal: 1,
                    position: 0,
                    rgb: 0,
                },
            )
            .unwrap();

        let gbuffer_albedo_texture_id = imgui_backend
            .register_texture(
                &context,
                &gbuffer.albedo_buffer,
                TextureUsage {
                    depth: 0,
                    normal: 0,
                    position: 0,
                    rgb: 1,
                },
            )
            .unwrap();

        let gbuffer_metalic_texture_id = imgui_backend
            .register_texture(
                &context,
                &gbuffer.metalic_roughness_buffer,
                TextureUsage {
                    depth: 0,
                    normal: 0,
                    position: 0,
                    rgb: 1,
                },
            )
            .unwrap();

        let ssao_texture_id = imgui_backend
            .register_texture(
                &context,
                &ssao_blur.target,
                TextureUsage {
                    depth: 1,
                    normal: 0,
                    position: 0,
                    rgb: 0,
                },
            )
            .unwrap();

        let dir_shadow_texture_id = imgui_backend
            .register_texture(
                &context,
                &dir_light_shadows.target_attachment,
                TextureUsage {
                    depth: 1,
                    normal: 0,
                    position: 0,
                    rgb: 0,
                },
            )
            .unwrap();

        let puffin_ui = puffin_imgui::ProfilerUi::default();

        Renderer {
            rc: RenderContext {
                context,
                gbuffer,
                light_system,
                screen_frame,
                skybox,
                scene,
                camera,
                gbuffer_position_texture_id,
                gbuffer_normals_texture_id,
                gbuffer_albedo_texture_id,
                gbuffer_metalic_texture_id,

                ssao_texture_id,
                dir_shadow_texture_id,
                ssao,
                ssao_blur,
                gen_hdr_cubemap,
                irradiance_convolution,
                prefilterenvmap,
                point_light_shadows,
                dir_light_shadows,

                brdf_texture,
            },

            imgui,
            imgui_backend,

            previous_frame_end,
            recreate_swap_chain: false,

            last_time: Instant::now(),

            event_loop,

            prebuild: true,
            puffin_ui,
        }
    }

    pub fn add_entity(&mut self, entity: Entity) {
        self.rc.scene.add_entity(entity);
    }

    fn recreate_swap_chain(rc: &mut RenderContext, imgui_backend: &ImguiBackend) {
        puffin::profile_scope!("recreate_swap_chain");

        rc.context.recreate_swap_chain();

        rc.gbuffer.recreate_swap_chain(&rc.context);

        rc.ssao = Ssao::initialize(&rc.context, &rc.scene.camera_uniform_buffer, &rc.gbuffer);
        rc.ssao_blur = SsaoBlur::initialize(&rc.context, &rc.ssao.target);

        rc.light_system = LightSystem::initialize(
            &rc.context,
            &rc.scene.camera_uniform_buffer,
            &rc.scene.light_uniform_buffer,
            &rc.gbuffer,
            &rc.ssao_blur.target,
            &rc.irradiance_convolution.cube_attachment_view,
            &rc.prefilterenvmap.cube_attachment_view,
            &rc.brdf_texture,
            &rc.point_light_shadows.cube_attachment_view,
            &rc.dir_light_shadows.target_attachment,
        );

        rc.screen_frame.recreate_swap_chain(
            &rc.context,
            &rc.light_system.target,
            &imgui_backend.ui_frame(),
        );

        println!("Recreating swap chain");
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>
    }

    fn draw_ui(
        rc: &RenderContext,
        frame_context: FrameContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        delta_time: &f64,
    ) {
        let FrameContext {
            puffin_ui, imgui, ..
        } = frame_context;

        frame_context
            .imgui_backend
            .frame(&rc.context, imgui, builder, |ui| {
                let gbuffer_position_texture_id = rc.gbuffer_position_texture_id;
                let gbuffer_albedo_texture_id = rc.gbuffer_albedo_texture_id;
                let gbuffer_normals_texture_id = rc.gbuffer_normals_texture_id;
                let gbuffer_metalic_texture_id = rc.gbuffer_metalic_texture_id;
                let ssao_texture_id = rc.ssao_texture_id;
                let dir_shadow_texture_id = rc.dir_shadow_texture_id;

                let camera_pos = rc.camera.position;

                puffin_ui.window(&ui);

                // Here we create a window with a specific size, and force it to always have a vertical scrollbar visible
                Window::new("Debug")
                    .position([0.0, 0.0], Condition::Always)
                    .size([350.0, 1080.0], Condition::Always)
                    .build(&ui, || {
                        let fps = 1.0 / delta_time;
                        ui.text(format!("FPS: {:.2}", fps));
                        ui.text(format!(
                            "Camera: ({:.2}, {:.2}, {:.2})",
                            camera_pos.x, camera_pos.y, camera_pos.z
                        ));
                        ui.separator();

                        Image::new(dir_shadow_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("Dir shadow map");

                        Image::new(ssao_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("SSAO with Blur");

                        Image::new(gbuffer_metalic_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("GBuffer metalic");

                        Image::new(gbuffer_position_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("GBuffer position");

                        Image::new(gbuffer_normals_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("GBuffer normals");

                        Image::new(gbuffer_albedo_texture_id, [300.0, 300.0]).build(&ui);
                        ui.text("GBuffer albedo");
                    });
            });
    }

    fn prebuild(
        rc: &RenderContext,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        puffin::profile_scope!("prebuild");

        rc.gen_hdr_cubemap.add_to_builder(builder);

        rc.irradiance_convolution.add_to_builder(builder);

        rc.prefilterenvmap.add_to_builder(builder);
    }

    fn draw_frame(rc: &mut RenderContext, frame_context: FrameContext, delta_time: &f64) {
        let FrameContext {
            previous_frame_end,
            prebuild,
            recreate_swap_chain,
            imgui,
            imgui_backend,
            puffin_ui,
            ..
        } = frame_context;

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if *recreate_swap_chain {
            Self::recreate_swap_chain(rc, &imgui_backend);
            *recreate_swap_chain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rc.context.swap_chain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    *recreate_swap_chain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            *recreate_swap_chain = true;
        }

        let offscreen_framebuffer = rc.gbuffer.framebuffer.clone();
        let framebuffer = rc.screen_frame.framebuffers[image_index].clone();

        let dimensions_u32 = rc.context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let mut builder = AutoCommandBufferBuilder::primary(
            rc.context.device.clone(),
            rc.context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
        )
        .unwrap();

        if *prebuild {
            Self::prebuild(rc, &mut builder);
            println!("Prebuild done");
            *prebuild = false;
        }

        rc.point_light_shadows
            .add_to_builder(&rc.context, &mut builder, &rc.scene);

        rc.dir_light_shadows
            .add_to_builder(&rc.context, &mut builder, &rc.scene);

        builder
            .update_buffer(
                rc.skybox.uniform_buffer.clone(),
                Arc::new(rc.camera.get_skybox_uniform_data(dimensions)),
            )
            .unwrap();

        rc.scene
            .update_uniform_buffers(&mut builder, &rc.camera, dimensions);

        builder
            .begin_render_pass(
                offscreen_framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![
                    // position
                    ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
                    // normals
                    ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
                    // albedo
                    ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
                    // metalic_roughness
                    ClearValue::Float([0.0, 0.0, 0.0, 0.0]),
                    // depth
                    ClearValue::Depth(1.0),
                ],
            )
            .unwrap();

        if RENDER_SKYBOX {
            builder
                .execute_commands(rc.skybox.command_buffer.clone())
                .unwrap();
        }

        builder
            .execute_commands(rc.gbuffer.create_command_buffer(&rc.context, &rc.scene))
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                rc.ssao.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(rc.ssao.command_buffers[image_index].clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                rc.ssao_blur.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(rc.ssao_blur.command_buffers[image_index].clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                rc.light_system.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(rc.light_system.create_command_buffers(&rc.context).clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        Self::draw_ui(
            rc,
            FrameContext {
                prebuild,
                previous_frame_end,
                recreate_swap_chain,
                imgui_backend,
                imgui,
                puffin_ui,
            },
            &mut builder,
            delta_time,
        );

        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 1.0])],
            )
            .unwrap()
            .execute_commands(rc.screen_frame.create_command_buffer(&rc.context))
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(rc.context.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                rc.context.graphics_queue.clone(),
                rc.context.swap_chain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                *recreate_swap_chain = true;
                *previous_frame_end =
                    Some(Box::new(vulkano::sync::now(rc.context.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                *previous_frame_end =
                    Some(Box::new(vulkano::sync::now(rc.context.device.clone())) as Box<_>);
            }
        }
    }

    fn update(camera: &mut Camera, keys: &HashMap<VirtualKeyCode, ElementState>, dt: f64) {
        let camera_speed = (10.0 * dt) as f32;

        if is_pressed(keys, VirtualKeyCode::Q) {
            camera.position += Vec3::Y * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::E) {
            camera.position -= Vec3::Y * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::A) {
            camera.position -= camera.right() * camera_speed
        }

        if is_pressed(keys, VirtualKeyCode::D) {
            camera.position += camera.right() * camera_speed
        }

        if is_pressed(keys, VirtualKeyCode::W) {
            camera.position += camera.forward() * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::S) {
            camera.position -= camera.forward() * camera_speed;
        }
    }

    pub fn main_loop(mut self) {
        let Renderer {
            mut event_loop,
            mut imgui_backend,
            mut imgui,
            mut rc,
            mut recreate_swap_chain,
            mut previous_frame_end,
            mut puffin_ui,
            mut prebuild,
            ..
        } = self;

        let mut mouse_buttons: HashMap<MouseButton, ElementState> = HashMap::new();
        let mut keyboard_buttons: HashMap<VirtualKeyCode, ElementState> = HashMap::new();

        let mut last_frame = Instant::now();
        let mut delta_time = 0.0;

        let mut rotation_x = 0.0;
        let mut rotation_y = 0.0;

        let original_rotation = rc.camera.rotation;

        let mut running = true;
        while running {
            let want_capture_mouse = imgui.io_mut().want_capture_mouse;
            let want_capture_keyboard = imgui.io_mut().want_capture_keyboard;

            event_loop.run_return(|event, _, control_flow| {
                puffin::profile_scope!("event_loop");

                *control_flow = ControlFlow::Poll;

                imgui_backend.handle_event(&rc.context, &mut imgui, &event);

                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => {
                        running = false;
                        *control_flow = ControlFlow::Exit;
                    }

                    Event::WindowEvent {
                        event: WindowEvent::Resized(_),
                        ..
                    } => {
                        recreate_swap_chain = true;
                    }

                    Event::WindowEvent {
                        event: WindowEvent::MouseInput { state, button, .. },
                        ..
                    } if !want_capture_mouse => {
                        mouse_buttons.insert(button, state);
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
                    } if !want_capture_keyboard => {
                        keyboard_buttons.insert(virtual_keycode, state);
                    }

                    Event::DeviceEvent {
                        event: DeviceEvent::MouseMotion { delta, .. },
                        ..
                    } if !want_capture_mouse => {
                        match mouse_buttons.get(&MouseButton::Left) {
                            Some(&ElementState::Pressed) => {
                                let sensitivity = 0.5 * delta_time;
                                let (x, y) = delta;

                                rotation_x += (x * sensitivity) as f32;
                                rotation_y += (y * sensitivity) as f32;

                                let y_quat = Quat::from_axis_angle(Vec3::X, -rotation_y);
                                let x_quat = Quat::from_axis_angle(Vec3::Y, -rotation_x);

                                rc.camera.rotation = original_rotation * x_quat * y_quat;
                            }

                            _ => {}
                        };
                    }

                    Event::NewEvents(_) => {
                        let now = Instant::now();
                        imgui.io_mut().update_delta_time(now - last_frame);
                        last_frame = now;
                    }

                    Event::MainEventsCleared => *control_flow = ControlFlow::Exit,

                    _ => (),
                }
            });

            puffin::GlobalProfiler::lock().new_frame();

            let now = Instant::now();
            delta_time = now.duration_since(self.last_time).as_secs_f64();

            self.last_time = now;

            {
                puffin::profile_scope!("update");
                Self::update(&mut rc.camera, &keyboard_buttons, delta_time);
            }

            {
                puffin::profile_scope!("draw");
                Self::draw_frame(
                    &mut rc,
                    FrameContext {
                        prebuild: &mut prebuild,
                        previous_frame_end: &mut previous_frame_end,
                        recreate_swap_chain: &mut recreate_swap_chain,
                        imgui_backend: &mut imgui_backend,
                        imgui: &mut imgui,
                        puffin_ui: &mut puffin_ui,
                    },
                    &delta_time,
                );
            }
        }
    }
}

struct FrameContext<'a> {
    previous_frame_end: &'a mut Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: &'a mut bool,
    prebuild: &'a mut bool,
    imgui_backend: &'a mut ImguiBackend,
    imgui: &'a mut imgui::Context,
    puffin_ui: &'a mut ProfilerUi,
}

fn is_pressed(keys: &HashMap<VirtualKeyCode, ElementState>, key: VirtualKeyCode) -> bool {
    match keys.get(&key) {
        Some(&ElementState::Pressed) => true,
        _ => false,
    }
}
