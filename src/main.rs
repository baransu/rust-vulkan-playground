use std::{
    collections::{HashMap, HashSet},
    f32::consts::PI,
    fs, io,
    iter::FromIterator,
    path::Path,
    sync::Arc,
    time::Instant,
};

use glam::{Mat4, Quat, Vec3};
use gltf::{image::Source, Semantic};
use image::{DynamicImage, GenericImageView, ImageFormat};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Features, Queue,
    },
    format::{ClearValue, Format},
    image::{
        view::ImageView, AttachmentImage, ImageDimensions, ImageUsage, ImmutableImage,
        MipmapsCount, SampleCount, SwapchainImage,
    },
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, ApplicationInfo, Instance, InstanceExtensions,
    },
    pipeline::{
        depth_stencil::DepthStencil, viewport::Viewport, GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    single_pass_renderpass,
    swapchain::{
        acquire_next_image, AcquireError, ColorSpace, CompositeAlpha, PresentMode,
        SupportedPresentModes, Surface, Swapchain,
    },
    sync::{self, GpuFuture, SharingMode},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
};

// const TEXTURE_PATH: &str = "res/viking_room.png";
const MODEL_PATH: &str = "res/damaged_helmet/scene.gltf";

// const MODEL_PATH: &str = "res/336_lrm/scene.gltf";

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

vulkano::impl_vertex!(Vertex, position, normal, uv, color);

struct Model {
    index_count: u32,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,
    uniform_buffer: Arc<CpuAccessibleBuffer<MVPUniformBufferObject>>,
    descriptor_set: Arc<PersistentDescriptorSet>,

    transform: Transform,
}

#[derive(Clone)]
struct Transform {
    // children: Vec<usize>,
    // final_transform: Mat4,
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}

// impl Transform {
//     fn update_transform(&mut self, root: &mut Root, parent_transform: &Mat4) {
//         self.final_transform = *parent_transform;

//         self.final_transform = self.final_transform
//             * Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation);

//         for node_id in &self.children {
//             let transform = root.unsafe_get_node_mut(*node_id);
//             transform.update_transform(root, &self.final_transform);
//         }
//     }
// }

struct Root {
    transforms: Vec<Transform>,
}

impl Root {
    pub fn unsafe_get_node_mut(&mut self, index: usize) -> &'static mut Transform {
        unsafe { &mut *(&mut self.transforms[index] as *mut Transform) }
    }
}

type MVPUniformBufferObject = vertex_shader::ty::MVPUniformBufferObject;

struct Camera {
    theta: f32,
    phi: f32,
    r: f32,
    target: Vec3,
}

impl Camera {
    fn position(&self) -> Vec3 {
        Vec3::new(
            self.target[0] + self.r * self.phi.sin() * self.theta.sin(),
            self.target[1] + self.r * self.phi.cos(),
            self.target[2] + self.r * self.phi.sin() * self.theta.cos(),
        )
    }

    fn target(&self) -> Vec3 {
        self.target
    }
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            theta: 0.0_f32.to_radians(),
            phi: 90.0_f32.to_radians(),
            r: 5.0,
            target: Vec3::new(0.0, 0.0, 0.0),
        }
    }
}

struct HelloTriangleApplication {
    instance: Arc<Instance>,
    debug_callback: Option<DebugCallback>,
    /**
     * This is why we need to wrap event_loop into Option
     *
     * https://stackoverflow.com/questions/67349506/ownership-issues-when-attempting-to-work-with-member-variables-passed-to-closure
     *
     * I don't really understand how it works and why exactly it's needed.
     */
    event_loop: Option<EventLoop<()>>,
    surface: Arc<Surface<Window>>,

    physical_device_index: usize, // can't store PhysicalDevice directly (lifetime issues)
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,
    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPass>,

    graphics_pipeline: Arc<GraphicsPipeline>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    command_buffers: Vec<Arc<SecondaryAutoCommandBuffer>>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    last_time: Instant,

    models: Vec<Model>,

    camera: Camera,
}

impl HelloTriangleApplication {
    pub fn initialize() -> Self {
        let instance = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);
        let (event_loop, surface) = Self::init_window(&instance);

        let (physical_device_index, unique_queue_families_ids) =
            Self::pick_physical_device(&surface, &instance);

        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &instance,
            physical_device_index,
            unique_queue_families_ids,
        );

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(
            &instance,
            &surface,
            physical_device_index,
            &device,
            &graphics_queue,
        );

        let depth_format = Self::find_depth_format();
        let sample_count = Self::find_sample_count();
        let depth_image =
            Self::create_depth_image(&device, swap_chain.dimensions(), depth_format, sample_count);

        let render_pass =
            Self::create_render_pass(&device, swap_chain.format(), depth_format, sample_count);

        let graphics_pipeline =
            Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_swap_chain_framebuffers(
            &device,
            &swap_chain_images,
            &render_pass,
            &depth_image,
            sample_count,
        );

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let camera = Default::default();

        let image_sampler = Self::create_image_sampler(&device);

        let models = Self::load_models(
            &device,
            &graphics_queue,
            &graphics_pipeline,
            &image_sampler,
            &camera,
            swap_chain_images[0].dimensions(),
        );

        let mut app = Self {
            instance,
            debug_callback,
            event_loop: Some(event_loop),
            physical_device_index,
            device,
            graphics_queue,
            present_queue,
            surface,
            swap_chain,
            swap_chain_images,
            render_pass,
            graphics_pipeline,
            swap_chain_framebuffers,

            command_buffers: vec![],

            previous_frame_end,
            recreate_swap_chain: false,

            last_time: Instant::now(),

            models,

            camera,
        };

        app.create_command_buffers();

        app
    }

    fn init_window(instance: &Arc<Instance>) -> (EventLoop<()>, Arc<Surface<Window>>) {
        let event_loop = EventLoop::new();

        let surface = WindowBuilder::new()
            .with_title("Vulkan")
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        (event_loop, surface)
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core().unwrap();

        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("Hello Triangle".into()),
            application_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
        };

        let required_extensions = Self::get_required_extensions();

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(
                Some(&app_info),
                Version::V1_1,
                &required_extensions,
                VALIDATION_LAYERS.iter().cloned(),
            )
            .unwrap()
        } else {
            Instance::new(Some(&app_info), Version::V1_1, &required_extensions, None).unwrap()
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list()
            .unwrap()
            .map(|l| l.name().to_owned())
            .collect();

        VALIDATION_LAYERS
            .iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_utils = true;
        }

        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: false,
            verbose: true,
        };

        let message_type = MessageType {
            general: true,
            validation: true,
            performance: true,
        };

        DebugCallback::new(&instance, severity, message_type, |msg| {
            println!("validation layer: {:?}", msg.description);
        })
        .ok()
    }

    fn get_unique_queue_families_ids_if_device_suitable(
        device: &PhysicalDevice,
        surface: &Arc<Surface<Window>>,
    ) -> Option<HashSet<u32>> {
        let queue_families = device
            .queue_families()
            .filter(|&q| q.supports_graphics() || surface.is_supported(q).unwrap_or(false));

        let queue_families_uniq_ids =
            HashSet::from_iter(queue_families.map(|q| q.id()).into_iter());

        let extensions_supported = Self::check_device_extension_support(device);

        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface
                .capabilities(*device)
                .expect("failed to get surface capabilities");
            !capabilities.supported_formats.is_empty()
                && capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        if !queue_families_uniq_ids.is_empty() && extensions_supported && swap_chain_adequate {
            Some(queue_families_uniq_ids)
        } else {
            None
        }
    }

    fn pick_physical_device(
        surface: &Arc<Surface<Window>>,
        instance: &Arc<Instance>,
    ) -> (usize, HashSet<u32>) {
        let (physical_device, unique_queue_families_ids) = PhysicalDevice::enumerate(instance)
            .filter(|&p| p.supported_extensions().is_superset_of(&DEVICE_EXTENSIONS))
            .filter_map(|device| {
                Self::get_unique_queue_families_ids_if_device_suitable(&device, surface)
                    .map(|unique_queue_families_ids| (device, unique_queue_families_ids))
            })
            .min_by_key(|(device, _)| {
                // Better score to device types that are likely to be faster/better.
                match device.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                }
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        (physical_device.index(), unique_queue_families_ids.clone())
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        physical_device_index: usize,
        unique_queue_families_ids: HashSet<u32>,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();

        let queue_priority = 1.0;

        let queue_families = physical_device
            .queue_families()
            .filter(|q| unique_queue_families_ids.contains(&q.id()))
            .map(|q| (q, queue_priority));

        // Some devices require certain extensions to be enabled if they are present
        // (e.g. `khr_portability_subset`). We add them to the device extensions that we're going to
        // enable.
        let required_extensions = &physical_device
            .required_extensions()
            .union(&DEVICE_EXTENSIONS);

        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            required_extensions,
            queue_families,
        )
        .unwrap();

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn find_depth_format() -> Format {
        // https://github.com/matthew-russo/vulkan-tutorial-rs/blob/26_depth_buffering/src/bin/26_depth_buffering.rs.diff#L115
        Format::D16_UNORM
    }

    fn find_sample_count() -> SampleCount {
        // https://github.com/matthew-russo/vulkan-tutorial-rs/blob/29_multisampling/src/bin/29_multisampling.rs.diff#L52-L59

        // macOS doesn't support MSAA8, so we'll use MSAA4 instead.
        SampleCount::Sample4
    }

    fn create_depth_image(
        device: &Arc<Device>,
        dimensions: [u32; 2],
        format: Format,
        sample_count: SampleCount,
    ) -> Arc<ImageView<Arc<AttachmentImage>>> {
        let image = AttachmentImage::multisampled_with_usage(
            device.clone(),
            dimensions,
            sample_count,
            format,
            ImageUsage {
                depth_stencil_attachment: true,
                ..ImageUsage::none()
            },
        )
        .unwrap();

        ImageView::new(image).unwrap()
    }

    fn create_render_pass(
        device: &Arc<Device>,
        color_format: Format,
        depth_format: Format,
        sample_count: SampleCount,
    ) -> Arc<RenderPass> {
        Arc::new(
            single_pass_renderpass!(device.clone(),
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

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let supported_extensions = PhysicalDevice::supported_extensions(device);
        supported_extensions.intersection(&DEVICE_EXTENSIONS) == DEVICE_EXTENSIONS
    }

    fn choose_swap_surface_format(
        available_formats: &[(Format, ColorSpace)],
    ) -> (Format, ColorSpace) {
        *available_formats
            .iter()
            .find(|(format, color_space)| {
                *format == Format::B8G8R8A8_SRGB && *color_space == ColorSpace::SrgbNonLinear
            })
            .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn create_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let capabilities = surface
            .capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let (surface_format, surface_color_space) =
            Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some()
            && image_count > capabilities.max_image_count.unwrap()
        {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        // TODO: make present and graphics queue work
        let sharing: SharingMode =
            // if graphics_queue.id_within_family() != present_queue.id_within_family() {
            //     vec![graphics_queue, present_queue].as_slice().into()
            // } else {
                graphics_queue.into();
        // };

        let dimensions: [u32; 2] = surface.window().inner_size().into();

        let (swap_chain, images) = Swapchain::start(device.clone(), surface.clone())
            .num_images(image_count)
            .format(surface_format)
            .color_space(surface_color_space)
            .dimensions(dimensions)
            .composite_alpha(CompositeAlpha::Opaque)
            .usage(image_usage)
            .sharing_mode(sharing)
            .transform(capabilities.current_transform)
            .layers(1)
            .clipped(true)
            .present_mode(present_mode)
            .build()
            .unwrap();

        (swap_chain, images)
    }

    fn recreate_swap_chain(&mut self) {
        let dimensions: [u32; 2] = self.surface.window().inner_size().into();

        let (swap_chain, images) = Swapchain::recreate(&self.swap_chain)
            .dimensions(dimensions)
            .build()
            .unwrap();

        self.swap_chain = swap_chain;
        self.swap_chain_images = images;

        let depth_format = Self::find_depth_format();
        let sample_count = Self::find_sample_count();
        let depth_image =
            Self::create_depth_image(&self.device, dimensions, depth_format, sample_count);

        self.render_pass = Self::create_render_pass(
            &self.device,
            self.swap_chain.format(),
            depth_format,
            sample_count,
        );

        self.graphics_pipeline = Self::create_graphics_pipeline(
            &self.device,
            self.swap_chain.dimensions(),
            &self.render_pass,
        );

        self.swap_chain_framebuffers = Self::create_swap_chain_framebuffers(
            &self.device,
            &self.swap_chain_images,
            &self.render_pass,
            &depth_image,
            sample_count,
        );

        self.create_command_buffers();
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vert_shader_module = vertex_shader::Shader::load(device.clone()).unwrap();
        let frag_shader_module = fragment_shader::Shader::load(device.clone()).unwrap();

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];
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
                .build(device.clone())
                .unwrap(),
        );

        pipeline
    }

    fn create_swap_chain_framebuffers(
        device: &Arc<Device>,
        swap_chain_images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: &Arc<RenderPass>,
        depth_image: &Arc<ImageView<Arc<AttachmentImage>>>,
        sample_count: SampleCount,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images
            .iter()
            .map(|swapchain_image| {
                let image = ImageView::new(swapchain_image.clone()).unwrap();

                let dimensions = swapchain_image.dimensions();
                let multisample_image = ImageView::new(
                    AttachmentImage::transient_multisampled(
                        device.clone(),
                        dimensions,
                        sample_count,
                        swapchain_image.swapchain().format(),
                    )
                    .unwrap()
                    .clone(),
                )
                .unwrap();

                let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(multisample_image.clone())
                        .unwrap()
                        .add(depth_image.clone())
                        .unwrap()
                        .add(image.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                framebuffer
            })
            .collect::<Vec<_>>()
    }

    fn create_command_buffers(&mut self) {
        let dimensions_u32 = self.swap_chain_images[0].dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        self.command_buffers = self
            .swap_chain_framebuffers
            .iter()
            .map(|_framebuffer| {
                let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                    self.device.clone(),
                    self.graphics_queue.family(),
                    CommandBufferUsage::SimultaneousUse,
                    self.graphics_pipeline.subpass().clone(),
                )
                .unwrap();

                builder
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(self.graphics_pipeline.clone());

                for model in self.models.iter() {
                    builder
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.graphics_pipeline.layout().clone(),
                            0,
                            model.descriptor_set.clone(),
                        )
                        .bind_vertex_buffers(0, model.vertex_buffer.clone())
                        .bind_index_buffer(model.index_buffer.clone())
                        .draw_indexed(model.index_count, 1, 0, 0, 0)
                        .unwrap();
                }

                let command_buffer = builder.build().unwrap();

                Arc::new(command_buffer)
            })
            .collect();
    }

    fn create_uniform_buffer(
        device: &Arc<Device>,
        camera: &Camera,
        dimensions_u32: [u32; 2],
        transform: &Transform,
    ) -> Arc<CpuAccessibleBuffer<MVPUniformBufferObject>> {
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let uniform_buffer_data = Self::update_uniform_buffer(&camera, dimensions, transform);

        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer_data,
        )
        .unwrap();

        buffer
    }

    fn update_uniform_buffer(
        camera: &Camera,
        dimensions: [f32; 2],
        transform: &Transform,
    ) -> MVPUniformBufferObject {
        let view = Mat4::look_at_rh(camera.position(), camera.target(), Vec3::Y);

        let mut proj = Mat4::perspective_rh(
            deg_to_rad(45.0),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0,
        );

        proj.y_axis.y *= -1.0;

        // this is needed to fix model rotation
        let model = Mat4::from_rotation_x(deg_to_rad(90.0))
            * Mat4::from_scale_rotation_translation(
                transform.scale,
                transform.rotation,
                transform.translation,
            );

        MVPUniformBufferObject {
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            model: model.to_cols_array_2d(),
        }
    }

    fn create_descriptor_set(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<MVPUniformBufferObject>>,
        texture_image: &Arc<ImageView<Arc<ImmutableImage>>>,
        image_sampler: &Arc<Sampler>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        set_builder
            .add_sampled_image(texture_image.clone(), image_sampler.clone())
            .unwrap();

        Arc::new(set_builder.build().unwrap())
    }

    fn create_vertex_buffer(
        graphics_queue: &Arc<Queue>,
        vertices: &Vec<Vertex>,
    ) -> Arc<ImmutableBuffer<[Vertex]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices.iter().cloned(),
            BufferUsage::vertex_buffer(),
            // TODO: idealy it should be transfer queue?
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_index_buffer(
        graphics_queue: &Arc<Queue>,
        indices: &Vec<u32>,
    ) -> Arc<ImmutableBuffer<[u32]>> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices.iter().cloned(),
            BufferUsage::index_buffer(),
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_texture(
        graphics_queue: &Arc<Queue>,
        image: &DynamicImage,
    ) -> Arc<ImageView<Arc<ImmutableImage>>> {
        let width = image.width();
        let height = image.height();

        let dimensions = ImageDimensions::Dim2d {
            width,
            height,
            // TODO: what are array_layers?
            array_layers: 1,
        };

        let image_rgba = image.to_rgba8();

        let (image, future) = ImmutableImage::from_iter(
            image_rgba.into_raw().iter().cloned(),
            dimensions,
            // vulkano already supports mipmap generation so we don't need to do this by hand
            MipmapsCount::Log2,
            Format::R8G8B8A8_SRGB,
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        ImageView::new(image).unwrap()
    }

    fn create_image_sampler(device: &Arc<Device>) -> Arc<Sampler> {
        Sampler::new(
            device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            // what's the minimul mip map lod we want to use - 0 means we start with highest mipmap which is original texture
            0.0,
            // if something will be super small we set 1_000 so it adjustes automatically
            1_000.0,
        )
        .unwrap()
    }

    fn load_models(
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        graphics_pipeline: &Arc<GraphicsPipeline>,
        image_sampler: &Arc<Sampler>,
        camera: &Camera,
        dimensions_u32: [u32; 2],
    ) -> Vec<Model> {
        let mut models = Vec::new();

        let (document, buffers, _images) = gltf::import(MODEL_PATH).unwrap();

        let transforms = document
            .nodes()
            .map(|node| {
                let (translation, rotation, scale) = node.transform().decomposed();

                Transform {
                    // children: node.children().map(|child| child.index()).collect(),
                    // final_transform: Mat4::IDENTITY,
                    scale: Vec3::new(scale[0], scale[1], scale[2]),
                    rotation: Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
                    translation: Vec3::new(translation[0], translation[1], translation[2]),
                }
            })
            .collect::<Vec<_>>();

        let mut root = Root { transforms };

        // for node_id in document.nodes().map(|child| child.index()) {
        //     let node = root.unsafe_get_node_mut(node_id);
        //     node.update_transform(&mut root, &Mat4::from_rotation_x(deg_to_rad(90.0)));
        // }

        for node in document.nodes() {
            if let Some(mesh) = node.mesh() {
                for primitive in mesh.primitives() {
                    let pbr = primitive.material().pbr_metallic_roughness();

                    let base_color_factor = pbr.base_color_factor();

                    let (texture_image, tex_coord) = pbr
                        .base_color_texture()
                        .map(|color_info| {
                            let img = &color_info.texture();
                            // material.base_color_texture = Some(load_texture(
                            let image = match img.source().source() {
                                Source::View { view, mime_type } => {
                                    let parent_buffer_data = &buffers[view.buffer().index()].0;
                                    let begin = view.offset();
                                    let end = begin + view.length();
                                    let data = &parent_buffer_data[begin..end];
                                    match mime_type {
                                        "image/jpeg" => image::load_from_memory_with_format(
                                            data,
                                            ImageFormat::Jpeg,
                                        ),
                                        "image/png" => image::load_from_memory_with_format(
                                            data,
                                            ImageFormat::Png,
                                        ),
                                        _ => panic!(format!(
                                            "unsupported image type (image: {}, mime_type: {})",
                                            img.index(),
                                            mime_type
                                        )),
                                    }
                                }
                                Source::Uri { uri, mime_type } => {
                                    let base_path = Path::new(MODEL_PATH);

                                    if uri.starts_with("data:") {
                                        let encoded = uri.split(',').nth(1).unwrap();
                                        let data = base64::decode(&encoded).unwrap();
                                        let mime_type = if let Some(ty) = mime_type {
                                            ty
                                        } else {
                                            uri.split(',')
                                                .nth(0)
                                                .unwrap()
                                                .split(':')
                                                .nth(1)
                                                .unwrap()
                                                .split(';')
                                                .nth(0)
                                                .unwrap()
                                        };

                                        match mime_type {
                                            "image/jpeg" => image::load_from_memory_with_format(
                                                &data,
                                                ImageFormat::Jpeg,
                                            ),
                                            "image/png" => image::load_from_memory_with_format(
                                                &data,
                                                ImageFormat::Png,
                                            ),
                                            _ => panic!(format!(
                                                "unsupported image type (image: {}, mime_type: {})",
                                                img.index(),
                                                mime_type
                                            )),
                                        }
                                    } else if let Some(mime_type) = mime_type {
                                        let path = base_path
                                            .parent()
                                            .unwrap_or_else(|| Path::new("./"))
                                            .join(uri);

                                        println!("loading texture from {}", path.display());

                                        let file = fs::File::open(path).unwrap();
                                        let reader = io::BufReader::new(file);
                                        match mime_type {
                                            "image/jpeg" => image::load(reader, ImageFormat::Jpeg),
                                            "image/png" => image::load(reader, ImageFormat::Png),
                                            _ => panic!(format!(
                                                "unsupported image type (image: {}, mime_type: {})",
                                                img.index(),
                                                mime_type
                                            )),
                                        }
                                    } else {
                                        let path = base_path
                                            .parent()
                                            .unwrap_or_else(|| Path::new("./"))
                                            .join(uri);

                                        println!("loading texture from {}", path.display());

                                        image::open(path)
                                    }
                                }
                            }
                            .unwrap();

                            (
                                Self::create_texture(&graphics_queue, &image),
                                color_info.tex_coord(),
                            )
                        })
                        .unwrap_or_else(|| {
                            // just a fillter image to make descriptor set happy
                            let image = DynamicImage::new_rgb8(1, 1);
                            (Self::create_texture(&graphics_queue, &image), 0)
                        });

                    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                    if let Some(_accessor) = primitive.get(&Semantic::Positions) {
                        let positions =
                            &reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();
                        let normals = &reader
                            .read_normals()
                            .map_or(vec![], |normals| normals.collect());

                        let color = &reader
                            .read_colors(0)
                            .map_or(vec![], |colors| colors.into_rgba_f32().collect());

                        // TODO: what gltf has more than one uv channel?
                        let tex_coords_0 = &reader
                            .read_tex_coords(0)
                            .map_or(vec![], |coords| coords.into_f32().collect());

                        let tex_coords_1 = &reader
                            .read_tex_coords(1)
                            .map_or(vec![], |coords| coords.into_f32().collect());

                        let vertices = positions
                            .iter()
                            .enumerate()
                            .map(|(index, position)| {
                                let position = *position;
                                let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                                let tex_coords_0 = *tex_coords_0.get(index).unwrap_or(&[0.0, 0.0]);
                                let tex_coords_1 = *tex_coords_1.get(index).unwrap_or(&[0.0, 0.0]);

                                let color = *color.get(index).unwrap_or(&[1.0, 1.0, 1.0, 1.0]);

                                let uv = [
                                    [tex_coords_0[0], tex_coords_0[1]],
                                    [tex_coords_1[0], tex_coords_1[1]],
                                ][tex_coord as usize];

                                Vertex {
                                    color,
                                    position,
                                    normal,
                                    uv,
                                }
                            })
                            .collect::<Vec<_>>();

                        let indices = reader
                            .read_indices()
                            .map(|indices| indices.into_u32().collect::<Vec<_>>())
                            .unwrap();

                        let vertex_buffer = Self::create_vertex_buffer(&graphics_queue, &vertices);
                        let index_buffer = Self::create_index_buffer(&graphics_queue, &indices);

                        let transform = root.unsafe_get_node_mut(node.index());

                        let uniform_buffer = Self::create_uniform_buffer(
                            &device,
                            &camera,
                            dimensions_u32,
                            &transform,
                        );

                        let descriptor_set = Self::create_descriptor_set(
                            &graphics_pipeline,
                            &uniform_buffer,
                            &texture_image,
                            image_sampler,
                        );

                        let model = Model {
                            transform: (*transform).clone(),
                            vertex_buffer,
                            index_buffer,
                            index_count: indices.len() as u32,
                            uniform_buffer,
                            descriptor_set,
                        };

                        models.push(model);
                    }
                }
            }
        }

        println!("Loaded {} models", models.len());

        models
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
            match acquire_next_image(self.swap_chain.clone(), None) {
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

        let draw_command_buffer = self.command_buffers[image_index].clone();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.graphics_queue.family(),
            CommandBufferUsage::SimultaneousUse,
        )
        .unwrap();

        let framebuffer = self.swap_chain_framebuffers[image_index].clone();
        let dimensions_u32 = self.swap_chain_images[0].dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        for model in &self.models {
            let data = Arc::new(Self::update_uniform_buffer(
                &self.camera,
                dimensions,
                &model.transform,
            ));

            builder
                .update_buffer(model.uniform_buffer.clone(), data)
                .unwrap();
        }

        builder
            .begin_render_pass(
                framebuffer.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![
                    [0.0, 0.0, 0.0, 1.0].into(),
                    ClearValue::Depth(1.0),
                    ClearValue::None,
                ],
            )
            .unwrap()
            .execute_commands(draw_command_buffer)
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                // TODO: swap to present queue???
                self.graphics_queue.clone(),
                self.swap_chain.clone(),
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
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end =
                    Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    fn main_loop(mut self) {
        let mut mouse_buttons: HashMap<MouseButton, ElementState> = HashMap::new();
        // let mut cursor_delta = Vec2::new(0.0, 0.0);

        self.event_loop
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

                        for model in self.models.iter_mut() {
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
                                let screen_width = self.swap_chain.dimensions()[0] as f32;
                                let screen_height = self.swap_chain.dimensions()[1] as f32;

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
    let app = HelloTriangleApplication::initialize();
    app.main_loop();
}

fn deg_to_rad(deg: f32) -> f32 {
    deg * 3.14 / 180.0
}

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(binding = 0) uniform MVPUniformBufferObject {
                mat4 view;
                mat4 proj;
                mat4 model;
            } mvp_ubo;

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 normal;
            layout(location = 2) in vec2 uv;
            layout(location = 3) in vec4 color;

            layout(location = 0) out vec2 f_uv;
            layout(location = 1) out vec3 f_normal;
            layout(location = 2) out vec3 f_position;

            out gl_PerVertex {
                vec4 gl_Position;
            };

            void main() {
                gl_Position = mvp_ubo.proj * mvp_ubo.view * mvp_ubo.model * vec4(position, 1.0);
                f_position = vec3(mvp_ubo.model * vec4(position, 1.0));
                f_uv = uv;
                f_normal = mat3(transpose(inverse(mvp_ubo.model))) * normal;  
            }
        "
    }
}

mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(binding = 1) uniform sampler2D tex_sampler;

            layout(location = 0) in vec2 f_uv;
            layout(location = 1) in vec3 f_normal;
            layout(location = 2) in vec3 f_position;

            layout(location = 0) out vec4 out_color;

            void main() {
                vec3 light_pos = vec3(5.0, 0.0, 0.0);
                vec3 light_color = vec3(1.0, 1.0, 1.0);
                vec3 ambient = 0.1 * light_color;

                vec3 norm = normalize(f_normal);
                vec3 light_dir = normalize(light_pos - f_position);  

                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = diff * light_color;
                vec3 result = (ambient + diffuse) * texture(tex_sampler, f_uv).xyz;
                out_color = vec4(result,  1.0);
            }
        "
    }
}
