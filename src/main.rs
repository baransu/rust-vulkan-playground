pub mod imgui_renderer;
pub mod renderer;

use std::{collections::HashMap, sync::Arc, time::Instant, vec};

use glam::{EulerRot, Quat, Vec3};
use imgui::*;
use imgui_renderer::{shaders::TextureUsage, Renderer};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use renderer::{
    camera::Camera,
    context::Context,
    gbuffer::GBuffer,
    light_system::LightSystem,
    mesh::{GameObject, InstanceData, Material, Transform},
    scene::Scene,
    screen_frame::ScreenFrame,
    skybox_pass::SkyboxPass,
    ssao::Ssao,
    ssao_blur::SsaoBlur,
    vertex::Vertex,
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
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
        DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent,
    },
    event_loop::ControlFlow,
};

const MODEL_PATHS: [&str; 3] = [
    "res/models/damaged_helmet/scene.gltf",
    "res/models/plane/plane.gltf",
    "res/models/cube/cube.gltf",
];

const SKYBOX_PATH: &str = "res/hdr/uffizi_cube.ktx";
// const SKYBOX_PATH: &str = "res/hdr/gcanyon_cube.ktx";
// const SKYBOX_PATH: &str = "res/hdr/pisa_cube.ktx";

const RENDER_SKYBOX: bool = false;

const SHADOW_MAP_DIM: f32 = 2048.0;

pub type FramebufferT = dyn FramebufferAbstract + Send + Sync;

pub struct FramebufferWithAttachment {
    framebuffer: Arc<FramebufferT>,
    attachment: Arc<ImageView<Arc<AttachmentImage>>>,
}

struct Application {
    context: Context,

    gbuffer: GBuffer,
    scene_command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,
    // ambient_command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,
    light_system: LightSystem,

    shadow_graphics_pipeline: Arc<GraphicsPipeline>,
    shadow_framebuffer: FramebufferWithAttachment,
    shadow_command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,

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

    shadow_framebuffer_texture_id: TextureId,
    gbuffer_position_texture_id: TextureId,
    gbuffer_normals_texture_id: TextureId,
    gbuffer_albedo_texture_id: TextureId,
    ssao_texture_id: TextureId,

    ssao: Ssao,
    ssao_blur: SsaoBlur,
}

impl Application {
    pub fn initialize() -> Self {
        // let mut rng = rand::thread_rng();
        let context = Context::initialize();

        let shadow_render_pass = Self::create_shadow_render_pass(&context);
        let shadow_framebuffer = Self::create_shadow_framebuffer(&context, &shadow_render_pass);
        let shadow_graphics_pipeline =
            Self::create_shadow_graphics_pipeline(&context, &shadow_render_pass);

        let gbuffer_target = Self::create_gbuffer_target(&context);
        let gbuffer = GBuffer::initialize(&context, &gbuffer_target);

        let mut scene = Scene::initialize(
            &context,
            MODEL_PATHS.to_vec(),
            &gbuffer.pipeline,
            &shadow_graphics_pipeline,
            &shadow_framebuffer,
            &gbuffer,
        );

        let ssao = Ssao::initialize(&context, &scene.camera_uniform_buffer, &gbuffer);
        let ssao_blur = SsaoBlur::initialize(&context, &ssao.target);

        let light_system = LightSystem::initialize(
            &context,
            &gbuffer_target,
            &shadow_framebuffer,
            &scene.camera_uniform_buffer,
            &scene.light_uniform_buffer,
            &scene.light_space_uniform_buffer,
            &gbuffer,
            &ssao_blur.target,
        );

        let count = 10;
        let start = -(count / 2);
        let end = count / 2;

        // Helmets
        for x in start..end {
            for z in start..end {
                let translation = Vec3::new(x as f32 * 2.0, 2.0, z as f32 * 2.0);
                let material = Material {
                    // diffuse: Vec3::new(
                    //     rng.gen_range(0.0..1.0),
                    //     rng.gen_range(0.0..1.0),
                    //     rng.gen_range(0.0..1.0),
                    // ),
                    ..Default::default()
                };

                let game_object = GameObject::new(
                    "damaged_helmet",
                    Transform {
                        translation,
                        rotation: Quat::from_euler(EulerRot::XYZ, 90.0_f32.to_radians(), 0.0, 0.0),
                        scale: Vec3::ONE,
                    },
                    material,
                );

                scene.add_game_object(game_object);
            }
        }

        // Plane
        scene.add_game_object(GameObject::new(
            "Plane",
            Transform {
                rotation: Quat::from_euler(EulerRot::XYZ, 0.0, 0.0, 0.0),
                scale: Vec3::ONE * 25.0,
                translation: Vec3::new(0.0, 0.0, -5.0),
            },
            Default::default(),
        ));

        // point light cubes for reference
        for light in scene.point_lights.clone() {
            let material = Material {
                diffuse: light.color,
                ..Default::default()
            };

            let transform = Transform {
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE * 0.2,
                translation: light.position,
            };

            scene.add_game_object(GameObject::new("Cube", transform, material));
        }

        // dir light
        scene.add_game_object(GameObject::new(
            "Cube",
            Transform {
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE * 0.2,
                translation: Scene::dir_light_position(),
            },
            Material {
                ..Default::default()
            },
        ));

        let previous_frame_end = Some(Self::create_sync_objects(&context.device));

        let camera = Default::default();

        let skybox = SkyboxPass::initialize(&context, &gbuffer.render_pass, SKYBOX_PATH);

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

        let mut imgui_renderer = Renderer::init(&context, &mut imgui).unwrap();

        let screen_frame = ScreenFrame::initialize(
            &context,
            &gbuffer_target,
            &shadow_framebuffer,
            &imgui_renderer.target,
        );

        let shadow_framebuffer_texture_id = imgui_renderer
            .register_texture(
                &context,
                &shadow_framebuffer.attachment,
                TextureUsage {
                    depth: 1,
                    normal: 0,
                    position: 0,
                },
            )
            .unwrap();

        let gbuffer_position_texture_id = imgui_renderer
            .register_texture(
                &context,
                &gbuffer.position_buffer,
                TextureUsage {
                    depth: 0,
                    // NOTE: they are not normals but also are in [-1, 1] space
                    // so we need to convert them to [0,1]
                    normal: 0,
                    position: 1,
                },
            )
            .unwrap();

        let gbuffer_normals_texture_id = imgui_renderer
            .register_texture(
                &context,
                &gbuffer.normals_buffer,
                TextureUsage {
                    depth: 0,
                    normal: 1,
                    position: 0,
                },
            )
            .unwrap();

        let gbuffer_albedo_texture_id = imgui_renderer
            .register_texture(
                &context,
                &gbuffer.albedo_buffer,
                TextureUsage {
                    depth: 0,
                    normal: 0,
                    position: 0,
                },
            )
            .unwrap();

        let ssao_texture_id = imgui_renderer
            .register_texture(
                &context,
                &ssao_blur.target,
                TextureUsage {
                    depth: 1,
                    normal: 0,
                    position: 0,
                },
            )
            .unwrap();

        let mut app = Self {
            context,

            screen_frame,

            skybox,

            imgui,
            imgui_renderer,
            platform,

            gbuffer,

            scene_command_buffers: vec![],
            light_system,

            shadow_framebuffer,
            shadow_graphics_pipeline,
            shadow_command_buffers: vec![],

            previous_frame_end,
            recreate_swap_chain: false,

            last_time: Instant::now(),

            scene,

            camera,

            shadow_framebuffer_texture_id,
            gbuffer_albedo_texture_id,
            gbuffer_position_texture_id,
            gbuffer_normals_texture_id,
            ssao_texture_id,

            ssao,
            ssao_blur,
        };

        app.create_scene_command_buffers();
        app.create_shadow_command_buffers();

        app
    }

    fn create_shadow_depth_image(context: &Context) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let image = AttachmentImage::with_usage(
            context.device.clone(),
            [SHADOW_MAP_DIM as u32, SHADOW_MAP_DIM as u32],
            context.depth_format,
            ImageUsage {
                sampled: true,
                depth_stencil_attachment: true,
                ..ImageUsage::none()
            },
        )
        .unwrap();

        ImageView::new(image).unwrap()
    }

    fn create_gbuffer_target(context: &Context) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let image = AttachmentImage::with_usage(
            context.device.clone(),
            context.swap_chain.dimensions(),
            context.swap_chain.format(),
            ImageUsage {
                sampled: true,
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
    // fn create_scene_render_pass(context: &Context) -> Arc<RenderPass> {
    //     let color_format = context.swap_chain.format();
    //     let depth_format = context.depth_format;
    //     let sample_count = context.sample_count;

    //     Arc::new(
    //         single_pass_renderpass!(context.device.clone(),
    //                 attachments: {
    //                     multisample_color: {
    //                         load: Clear,
    //                         store: Store,
    //                         format: color_format,
    //                         samples: sample_count,
    //                     },
    //                     multisample_depth: {
    //                         load: Clear,
    //                         store: DontCare,
    //                         format: depth_format,
    //                         samples: sample_count,
    //                         initial_layout: ImageLayout::Undefined,
    //                         final_layout: ImageLayout::DepthStencilAttachmentOptimal,
    //                     },
    //                     resolve_color: {
    //                         load: DontCare,
    //                         store: Store,
    //                         format: color_format,
    //                         samples: 1,
    //                     }
    //                 },
    //                 pass: {
    //                     color: [multisample_color],
    //                     depth_stencil: {multisample_depth},
    //                     resolve: [resolve_color]
    //                 }
    //         )
    //         .unwrap(),
    //     )
    // }

    fn create_shadow_render_pass(context: &Context) -> Arc<RenderPass> {
        let depth_format = context.depth_format;

        Arc::new(
            single_pass_renderpass!(context.device.clone(),
                    attachments: {
                        depth: {
                            load: Clear,
                            store: Store,
                            format: depth_format,
                            samples: 1,
                        }
                    },
                    pass: {
                        color: [],
                        depth_stencil: {depth}
                    }
            )
            .unwrap(),
        )
    }

    fn recreate_swap_chain(&mut self) {
        self.context.recreate_swap_chain();

        // let offscreen_render_pass = Self::create_scene_render_pass(&self.context);
        // self.scene_framebuffers =
        //     Self::create_scene_framebuffers(&self.context, &offscreen_render_pass);
        // self.scene_graphics_pipeline =
        //     Self::create_scene_graphics_pipeline(&self.context, &offscreen_render_pass);

        // TODO: recreate shadow framebuffer and graphics pipeline

        self.screen_frame.recreate_swap_chain(&self.context);

        self.create_scene_command_buffers();
        self.create_shadow_command_buffers();
    }

    // /**
    //  * Creates graphics pipeline from the given render pass, and vertex/fragment shaders.
    //  */
    // fn create_scene_graphics_pipeline(
    //     context: &Context,
    //     render_pass: &Arc<RenderPass>,
    // ) -> Arc<GraphicsPipeline> {
    //     let vert_shader_module =
    //         renderer::shaders::model_vertex_shader::Shader::load(context.device.clone()).unwrap();
    //     let frag_shader_module =
    //         renderer::shaders::model_fragment_shader::Shader::load(context.device.clone()).unwrap();

    //     let dimensions_u32 = context.swap_chain.dimensions();
    //     let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
    //     let viewport = Viewport {
    //         origin: [0.0, 0.0],
    //         dimensions,
    //         depth_range: 0.0..1.0,
    //     };

    //     let pipeline = Arc::new(
    //         GraphicsPipeline::start()
    //             .vertex_input(
    //                 BuffersDefinition::new()
    //                     .vertex::<Vertex>()
    //                     .instance::<InstanceData>(),
    //             )
    //             .vertex_shader(vert_shader_module.main_entry_point(), ())
    //             .triangle_list()
    //             .primitive_restart(false)
    //             .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
    //             .fragment_shader(frag_shader_module.main_entry_point(), ())
    //             .depth_clamp(false)
    //             // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
    //             .polygon_mode_fill() // = default
    //             .line_width(1.0) // = default
    //             // TODO: just to make developing easier we render both faces of models
    //             .cull_mode_back()
    //             .front_face_counter_clockwise()
    //             // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
    //             .blend_pass_through()
    //             .depth_stencil(DepthStencil::simple_depth_test())
    //             .viewports_dynamic_scissors_irrelevant(1)
    //             .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
    //             .build(context.device.clone())
    //             .unwrap(),
    //     );

    //     pipeline
    // }

    fn create_shadow_graphics_pipeline(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module =
            renderer::shaders::shadow_vertex_shader::Shader::load(context.device.clone()).unwrap();
        let frag_shader_module =
            renderer::shaders::shadow_fragment_shader::Shader::load(context.device.clone())
                .unwrap();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [SHADOW_MAP_DIM, SHADOW_MAP_DIM],
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
                // NOTE: when we render shadows we render inners of the models (via front_face_clockwise and cull_mode_front)
                // this is to reduce peter panning as described in learnopengl.com/Advanced-Lighting/Shadows/Shadow-Mapping
                .cull_mode_front()
                .front_face_clockwise()
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
    // fn create_scene_framebuffers(
    //     context: &Context,
    //     render_pass: &Arc<RenderPass>,
    // ) -> Vec<FramebufferWithAttachment> {
    //     let depth_image = Self::create_depth_image(&context);
    //     let color_image = Self::create_color_image(&context);

    //     let mut framebuffers = Vec::new();

    //     for _i in 0..context.swap_chain.num_images() {
    //         let resolve_image = ImageView::new(
    //             AttachmentImage::with_usage(
    //                 context.device.clone(),
    //                 context.swap_chain.dimensions(),
    //                 context.swap_chain.format(),
    //                 ImageUsage {
    //                     sampled: true,
    //                     ..ImageUsage::none()
    //                 },
    //             )
    //             .unwrap(),
    //         )
    //         .unwrap();

    //         let framebuffer: Arc<FramebufferT> = Arc::new(
    //             Framebuffer::start(render_pass.clone())
    //                 .add(color_image.clone())
    //                 .unwrap()
    //                 .add(depth_image.clone())
    //                 .unwrap()
    //                 .add(resolve_image.clone())
    //                 .unwrap()
    //                 .build()
    //                 .unwrap(),
    //         );

    //         framebuffers.push(FramebufferWithAttachment {
    //             framebuffer,
    //             attachment: resolve_image,
    //         });
    //     }

    //     framebuffers
    // }

    fn create_shadow_framebuffer(
        context: &Context,
        render_pass: &Arc<RenderPass>,
    ) -> FramebufferWithAttachment {
        let depth_image = Self::create_shadow_depth_image(&context);

        let framebuffer: Arc<FramebufferT> = Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(depth_image.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        FramebufferWithAttachment {
            framebuffer,
            attachment: depth_image,
        }
    }

    fn create_instance_data_buffers(
        &self,
    ) -> HashMap<String, Arc<CpuAccessibleBuffer<[InstanceData]>>> {
        let mut instance_data: HashMap<String, Vec<InstanceData>> = HashMap::new();

        for game_object in self.scene.game_objects.iter() {
            let model = game_object.transform.get_model_matrix();

            let instances = instance_data
                .entry(game_object.mesh_id.clone())
                .or_insert(Vec::new());

            (*instances).push(InstanceData {
                model: model.to_cols_array_2d(),
                material_diffuse: game_object.material.diffuse.to_array(),
                material_specular: game_object.material.specular.to_array(),
            });
        }

        let mut instance_data_buffers: HashMap<String, Arc<CpuAccessibleBuffer<[InstanceData]>>> =
            HashMap::new();

        instance_data.iter().for_each(|(mesh_id, instances)| {
            let buffer = CpuAccessibleBuffer::from_iter(
                self.context.device.clone(),
                BufferUsage::all(),
                false,
                instances.iter().cloned(),
            )
            .unwrap();

            instance_data_buffers.insert(mesh_id.clone(), buffer);
        });

        instance_data_buffers
    }

    fn create_scene_command_buffers(&mut self) {
        let dimensions_u32 = self.context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        let instance_data_buffers = self.create_instance_data_buffers();

        let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

        for _i in 0..self.context.swap_chain.num_images() {
            let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                self.context.device.clone(),
                self.context.graphics_queue.family(),
                CommandBufferUsage::SimultaneousUse,
                self.gbuffer.pipeline.subpass().clone(),
            )
            .unwrap();

            builder.set_viewport(0, [viewport.clone()]);

            builder.bind_pipeline_graphics(self.gbuffer.pipeline.clone());

            for mesh in self.scene.meshes.values() {
                // if there is no instance_data_buffer it means we have 0 instances for this mesh
                if let Some(instance_data_buffer) = instance_data_buffers.get(&mesh.id) {
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.gbuffer.pipeline.layout().clone(),
                            0,
                            mesh.descriptor_set.clone(),
                        )
                        .bind_vertex_buffers(
                            0,
                            (mesh.vertex_buffer.clone(), instance_data_buffer.clone()),
                        )
                        .bind_index_buffer(mesh.index_buffer.clone())
                        .draw_indexed(mesh.index_count, instance_data_buffer.len() as u32, 0, 0, 0)
                        .unwrap();
                }
            }

            let command_buffer = Arc::new(builder.build().unwrap());

            command_buffers.push(command_buffer);
        }

        self.scene_command_buffers = command_buffers;
    }

    // fn create_ambient_command_buffers(&mut self) {
    //     let dimensions_u32 = self.context.swap_chain.dimensions();
    //     let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

    //     let viewport = Viewport {
    //         origin: [0.0, 0.0],
    //         dimensions,
    //         depth_range: 0.0..1.0,
    //     };

    //     let instance_data_buffers = self.create_instance_data_buffers();

    //     let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

    //     for _i in 0..self.context.swap_chain.num_images() {
    //         let mut builder = AutoCommandBufferBuilder::secondary_graphics(
    //             self.context.device.clone(),
    //             self.context.graphics_queue.family(),
    //             CommandBufferUsage::SimultaneousUse,
    //             self.light_system.pipeline.subpass().clone(),
    //         )
    //         .unwrap();

    //         builder.set_viewport(0, [viewport.clone()]);

    //         builder.bind_pipeline_graphics(self.light_system.pipeline.clone());

    //         builder.bind_descriptor_sets(
    //             PipelineBindPoint::Graphics,
    //             self.light_system.pipeline.layout().clone(),
    //             0,
    //             self.scene.light_descriptor_set.clone(),
    //         );

    //         for mesh in self.scene.meshes.values() {
    //             // if there is no instance_data_buffer it means we have 0 instances for this mesh
    //             if let Some(instance_data_buffer) = instance_data_buffers.get(&mesh.id) {
    //                 builder
    //                     .bind_vertex_buffers(
    //                         0,
    //                         (mesh.vertex_buffer.clone(), instance_data_buffer.clone()),
    //                     )
    //                     .bind_index_buffer(mesh.index_buffer.clone())
    //                     .draw_indexed(mesh.index_count, instance_data_buffer.len() as u32, 0, 0, 0)
    //                     .unwrap();
    //             }
    //         }

    //         let command_buffer = Arc::new(builder.build().unwrap());

    //         command_buffers.push(command_buffer);
    //     }

    //     self.ambient_command_buffers = command_buffers;
    // }

    // fn create_ssao_command_buffers(&mut self) {
    //     let dimensions_u32 = self.context.swap_chain.dimensions();
    //     let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

    //     let viewport = Viewport {
    //         origin: [0.0, 0.0],
    //         dimensions,
    //         depth_range: 0.0..1.0,
    //     };

    //     let instance_data_buffers = self.create_instance_data_buffers();

    //     let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

    //     for _i in 0..self.context.swap_chain.num_images() {
    //         let mut builder = AutoCommandBufferBuilder::secondary_graphics(
    //             self.context.device.clone(),
    //             self.context.graphics_queue.family(),
    //             CommandBufferUsage::SimultaneousUse,
    //             self.ssao.pipeline.subpass().clone(),
    //         )
    //         .unwrap();

    //         builder.set_viewport(0, [viewport.clone()]);

    //         builder.bind_pipeline_graphics(self.ssao.pipeline.clone());

    //         builder.bind_descriptor_sets(
    //             PipelineBindPoint::Graphics,
    //             self.ssao.pipeline.layout().clone(),
    //             0,
    //             self.ssao.descriptor_set.clone(),
    //         );

    //         for mesh in self.scene.meshes.values() {
    //             // if there is no instance_data_buffer it means we have 0 instances for this mesh
    //             if let Some(instance_data_buffer) = instance_data_buffers.get(&mesh.id) {
    //                 builder
    //                     .bind_vertex_buffers(
    //                         0,
    //                         (mesh.vertex_buffer.clone(), instance_data_buffer.clone()),
    //                     )
    //                     .bind_index_buffer(mesh.index_buffer.clone())
    //                     .draw_indexed(mesh.index_count, instance_data_buffer.len() as u32, 0, 0, 0)
    //                     .unwrap();
    //             }
    //         }

    //         let command_buffer = Arc::new(builder.build().unwrap());

    //         command_buffers.push(command_buffer);
    //     }

    //     self.ssao_command_buffers = command_buffers;
    // }

    fn create_shadow_command_buffers(&mut self) {
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [SHADOW_MAP_DIM, SHADOW_MAP_DIM],
            depth_range: 0.0..1.0,
        };

        let instance_data_buffers = self.create_instance_data_buffers();

        let mut command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>> = Vec::new();

        for _i in 0..self.context.swap_chain.num_images() {
            let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                self.context.device.clone(),
                self.context.graphics_queue.family(),
                CommandBufferUsage::SimultaneousUse,
                self.shadow_graphics_pipeline.subpass().clone(),
            )
            .unwrap();

            builder.set_viewport(0, [viewport.clone()]);

            builder.bind_pipeline_graphics(self.shadow_graphics_pipeline.clone());

            // TODO: set depth bias

            builder.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shadow_graphics_pipeline.layout().clone(),
                0,
                self.scene.shadow_descriptor_set.clone(),
            );

            for mesh in self.scene.meshes.values() {
                // if there is no instance_data_buffer it means we have 0 instances for this mesh
                if let Some(instance_data_buffer) = instance_data_buffers.get(&mesh.id) {
                    builder
                        .bind_vertex_buffers(
                            0,
                            (mesh.vertex_buffer.clone(), instance_data_buffer.clone()),
                        )
                        .bind_index_buffer(mesh.index_buffer.clone())
                        .draw_indexed(mesh.index_count, instance_data_buffer.len() as u32, 0, 0, 0)
                        .unwrap();
                }
            }

            let command_buffer = Arc::new(builder.build().unwrap());

            command_buffers.push(command_buffer);
        }

        self.shadow_command_buffers = command_buffers;
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>
    }

    fn draw_ui(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        delta_time: &f64,
    ) {
        let ui = self.imgui.frame();
        let shadow_texture_id = self.shadow_framebuffer_texture_id;
        let gbuffer_position_texture_id = self.gbuffer_position_texture_id;
        let gbuffer_albedo_texture_id = self.gbuffer_albedo_texture_id;
        let gbuffer_normals_texture_id = self.gbuffer_normals_texture_id;
        let ssao_texture_id = self.ssao_texture_id;

        // Here we create a window with a specific size, and force it to always have a vertical scrollbar visible
        Window::new("Debug")
            .position([0.0, 0.0], Condition::Always)
            .size([350.0, 600.0], Condition::Always)
            .build(&ui, || {
                let fps = 1.0 / delta_time;
                ui.text(format!("FPS: {:.2}", fps));
                ui.separator();

                Image::new(ssao_texture_id, [300.0, 300.0]).build(&ui);
                ui.text("SSAO with Blur");

                Image::new(shadow_texture_id, [300.0, 300.0]).build(&ui);
                ui.text("Directional light shadow map");

                Image::new(gbuffer_position_texture_id, [300.0, 300.0]).build(&ui);
                ui.text("GBuffer position");

                Image::new(gbuffer_normals_texture_id, [300.0, 300.0]).build(&ui);
                ui.text("GBuffer normals");

                Image::new(gbuffer_albedo_texture_id, [300.0, 300.0]).build(&ui);
                ui.text("GBuffer albedo");
            });

        let draw_data = ui.render();
        self.imgui_renderer
            .draw_commands(builder, draw_data)
            .unwrap();
    }

    fn draw_frame(&mut self, delta_time: &f64) {
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

        let light_command_buffer = self.light_system.command_buffers[image_index].clone();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.context.device.clone(),
            self.context.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
        )
        .unwrap();

        let offscreen_framebuffer = self.gbuffer.framebuffer.clone();
        let framebuffer = self.screen_frame.framebuffers[image_index].clone();

        let dimensions_u32 = self.context.swap_chain.dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        builder
            .update_buffer(
                self.skybox.uniform_buffer.clone(),
                Arc::new(self.camera.get_skybox_uniform_data(dimensions)),
            )
            .unwrap();

        self.scene
            .update_uniform_buffers(&mut builder, &self.camera, dimensions);

        builder
            .begin_render_pass(
                self.shadow_framebuffer.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Depth(1.0)],
            )
            .unwrap()
            .execute_commands(self.shadow_command_buffers[image_index].clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

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
                    // depth
                    ClearValue::Depth(1.0),
                ],
            )
            .unwrap();

        if RENDER_SKYBOX {
            builder
                .execute_commands(self.skybox.command_buffer.clone())
                .unwrap();
        }

        builder
            .execute_commands(offscreen_command_buffer)
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                self.ssao.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(self.ssao.command_buffers[image_index].clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                self.ssao_blur.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(self.ssao_blur.command_buffers[image_index].clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        builder
            .begin_render_pass(
                self.light_system.framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 0.0])],
            )
            .unwrap()
            .execute_commands(light_command_buffer.clone())
            .unwrap()
            .end_render_pass()
            .unwrap();

        self.draw_ui(&mut builder, delta_time);

        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![ClearValue::Float([0.0, 0.0, 0.0, 1.0])],
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

    fn update(&mut self, keys: &HashMap<VirtualKeyCode, ElementState>, dt: f64) {
        let camera_speed = (10.0 * dt) as f32;

        if is_pressed(keys, VirtualKeyCode::Q) {
            self.camera.position += Vec3::Y * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::E) {
            self.camera.position -= Vec3::Y * camera_speed;
        }

        if is_pressed(keys, VirtualKeyCode::A) {
            self.camera.position -= self.camera.right() * camera_speed
        }

        if is_pressed(keys, VirtualKeyCode::D) {
            self.camera.position += self.camera.right() * camera_speed
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
        let mut delta_time = 0.0;

        let mut rotation_x = 0.0;
        let mut rotation_y = 0.0;

        let original_rotation = self.camera.rotation;

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
                                let sensitivity = 0.5 * delta_time;
                                let (x, y) = delta;

                                rotation_x += (x * sensitivity) as f32;
                                rotation_y += (y * sensitivity) as f32;

                                let y_quat = Quat::from_axis_angle(Vec3::X, -rotation_y);
                                let x_quat = Quat::from_axis_angle(Vec3::Y, -rotation_x);

                                self.camera.rotation = original_rotation * x_quat * y_quat;
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
                        delta_time = now.duration_since(self.last_time).as_secs_f64();

                        self.last_time = now;

                        self.update(&keyboard_buttons, delta_time);

                        self.draw_frame(&delta_time);
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
