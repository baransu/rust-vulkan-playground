use std::{collections::HashSet, iter::FromIterator, sync::Arc, time::Instant};

use dolly::prelude::*;
use glam::{Mat4, Vec3};
use gltf::Semantic;
use image::GenericImageView;
use vulkano::{
    buffer::{
        immutable::ImmutableBufferInitialization, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
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
    sync::{self, GpuFuture, JoinFuture, SharingMode},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
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

const TEXTURE_PATH: &str = "res/viking_room.png";
const MODEL_PATH: &str = "res/336_lrm/scene.gltf";

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex: [f32; 2],
}

impl Vertex {
    fn new(position: [f32; 3], color: [f32; 3], tex: [f32; 2]) -> Vertex {
        Vertex {
            position,
            color,
            tex,
        }
    }
}

vulkano::impl_vertex!(Vertex, position, color, tex);

struct Model {
    index_count: u32,
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    index_buffer: Arc<ImmutableBuffer<[u32]>>,
}

type UniformBufferObject = vertex_shader::ty::UniformBufferObject;

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

    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    descriptor_set: Arc<PersistentDescriptorSet>,
    uniform_buffer: Arc<CpuAccessibleBuffer<UniformBufferObject>>,

    last_time: Instant,

    models: Vec<Model>,

    camera: CameraRig,
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

        let models = Self::load_models(&graphics_queue);

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

        let camera = CameraRig::builder()
            .with(YawPitch::new().yaw_degrees(45.0).pitch_degrees(-30.0))
            .with(Smooth::new_rotation(1.5))
            .with(Arm::new(Vec3::Z * 4.0))
            .build();

        // TODO: should we have uniform buffer per swap chain image?
        // vulkan-tutorial says so but I'm not 100% understanding it how it plays with
        // descriptor sets.
        let uniform_buffer =
            Self::create_uniform_buffer(&device.clone(), &camera, swap_chain.dimensions());

        let image_sampler = Self::create_image_sampler(&device);
        let texture_image = Self::create_texture_image(&graphics_queue);

        let descriptor_set = Self::create_descriptor_sets(
            &graphics_pipeline,
            &uniform_buffer,
            &texture_image,
            &image_sampler,
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
            descriptor_set,
            uniform_buffer,
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

    // fn create_camera_uniform_buffer(&mut self) -> Arc<PrimaryAutoCommandBuffer> {
    //     let queue_family = self.graphics_queue.family();

    //     let dimensions_u32 = self.swap_chain_images[0].dimensions();
    //     let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

    //     let mut builder = AutoCommandBufferBuilder::primary(
    //         self.device.clone(),
    //         queue_family,
    //         CommandBufferUsage::SimultaneousUse,
    //     )
    //     .unwrap();

    //     let uniform_buffer_data = Arc::new(Self::update_uniform_buffer(&self.camera, dimensions));

    //     builder
    //         .update_buffer(self.uniform_buffer.clone(), uniform_buffer_data)
    //         .unwrap();

    //     Arc::new(builder.build().unwrap())
    // }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();

        let dimensions_u32 = self.swap_chain_images[0].dimensions();
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        println!("framebuffers: {}", self.swap_chain_framebuffers.len());

        let model = self.models.iter().nth(0).unwrap();

        self.command_buffers = self
            .swap_chain_framebuffers
            .iter()
            .map(|framebuffer| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.device.clone(),
                    queue_family,
                    CommandBufferUsage::SimultaneousUse,
                )
                .unwrap();

                let uniform_buffer_data =
                    Arc::new(Self::update_uniform_buffer(&self.camera, dimensions));

                builder
                    .update_buffer(self.uniform_buffer.clone(), uniform_buffer_data)
                    .unwrap();

                builder
                    .begin_render_pass(
                        framebuffer.clone(),
                        SubpassContents::Inline,
                        vec![
                            [0.0, 0.0, 0.0, 1.0].into(),
                            ClearValue::Depth(1.0),
                            ClearValue::None,
                        ],
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(self.graphics_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.graphics_pipeline.layout().clone(),
                        0,
                        self.descriptor_set.clone(),
                    )
                    .bind_vertex_buffers(0, model.vertex_buffer.clone())
                    .bind_index_buffer(model.index_buffer.clone())
                    .draw_indexed(model.index_count, 1, 0, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                Arc::new(command_buffer)
            })
            .collect();
    }

    fn create_uniform_buffer(
        device: &Arc<Device>,
        camera: &CameraRig,
        dimensions_u32: [u32; 2],
    ) -> Arc<CpuAccessibleBuffer<UniformBufferObject>> {
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let uniform_buffer = Self::update_uniform_buffer(&camera, dimensions);

        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer,
        )
        .unwrap();

        buffer
    }

    fn update_uniform_buffer(camera: &CameraRig, dimensions: [f32; 2]) -> UniformBufferObject {
        let model = Mat4::from_rotation_x(deg_to_rad(-90.0));

        let view = Mat4::look_at_rh(
            camera.final_transform.position,
            Vec3::ZERO,
            camera.final_transform.up(),
        );

        let mut proj = Mat4::perspective_rh(
            deg_to_rad(45.0),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            10.0,
        );

        proj.y_axis.y *= -1.0;

        UniformBufferObject {
            model: model.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
        }
    }

    fn create_descriptor_sets(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<UniformBufferObject>>,
        texture_image: &Arc<ImageView<Arc<ImmutableImage>>>,
        image_sampler: &Arc<Sampler>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder
            .add_buffer(uniform_buffer.clone())
            .unwrap()
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

    fn create_texture_image(graphics_queue: &Arc<Queue>) -> Arc<ImageView<Arc<ImmutableImage>>> {
        let image = image::open(TEXTURE_PATH).unwrap();

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

    fn load_models(graphics_queue: &Arc<Queue>) -> Vec<Model> {
        let mut models = Vec::new();

        let (document, buffers, _images) = gltf::import(MODEL_PATH).unwrap();

        for mesh in document.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                if let Some(_accessor) = primitive.get(&Semantic::Positions) {
                    let positions = &reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();
                    let normals = &reader
                        .read_normals()
                        .map_or(vec![], |normals| normals.collect());

                    // TODO: what's channel 0/1
                    let tex_coords_0 = &reader
                        .read_tex_coords(0)
                        .map_or(vec![], |coords| coords.into_f32().collect());

                    // let tex_coords_1 = &reader
                    //     .read_tex_coords(1)
                    //     .map_or(vec![], |coords| coords.into_f32().collect());

                    let vertices = positions
                        .iter()
                        .enumerate()
                        .map(|(index, position)| {
                            let position = *position;
                            let normal = *normals.get(index).unwrap_or(&[1.0, 1.0, 1.0]);
                            let tex_coords_0 = *tex_coords_0.get(index).unwrap_or(&[0.0, 0.0]);
                            // let tex_coords_1 = *tex_coords_1.get(index).unwrap_or(&[0.0, 0.0]);

                            Vertex {
                                position,
                                color: normal,
                                tex: tex_coords_0,
                            }
                        })
                        .collect::<Vec<_>>();

                    let indices = reader
                        .read_indices()
                        .map(|indices| indices.into_u32().collect::<Vec<_>>())
                        .unwrap();

                    let vertex_buffer = Self::create_vertex_buffer(&graphics_queue, &vertices);
                    let index_buffer = Self::create_index_buffer(&graphics_queue, &indices);

                    let model = Model {
                        vertex_buffer,
                        index_buffer,
                        index_count: indices.len() as u32,
                    };

                    models.push(model);
                }
            }
        }

        // for buffer in buffers {}

        // for node in document.nodes() {
        //     if let Some(mesh) = node.mesh() {
        //         for primitive in mesh.primitives() {
        //             primitive
        //         }
        //     }
        // }

        // // tobj::load_obj(MODEL_PATH, &tobj::LoadOptions::default()).unwrap();

        // for model in models.iter() {
        //     let mesh = &model.mesh;

        //     for index in &mesh.indices {
        //         let ind_usize = *index as usize;

        //         let pos = [
        //             mesh.positions[ind_usize * 3],
        //             mesh.positions[ind_usize * 3 + 1],
        //             mesh.positions[ind_usize * 3 + 2],
        //         ];

        //         let normal = [0.0, 0.0, 0.0];

        //         let tex_coord = [
        //             mesh.texcoords[ind_usize * 2],
        //             // this is because Vulkan has flipped Y
        //             1.0 - mesh.texcoords[ind_usize * 2 + 1],
        //         ];

        //         let vertex = Vertex::new(pos, normal, tex_coord);
        //         vertices.push(vertex);
        //         let index = indices.len() as u32;
        //         indices.push(index);
        //     }
        // }

        println!("Loaded {} models", models.len());

        models
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

        let command_buffer = self.command_buffers[image_index].clone();

        let future = acquire_future
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

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>
    }

    fn main_loop(mut self) {
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

                    // on key press
                    Event::WindowEvent {
                        event:
                            WindowEvent::KeyboardInput {
                                input:
                                    KeyboardInput {
                                        virtual_keycode,
                                        state: ElementState::Pressed,
                                        ..
                                    },
                                ..
                            },
                        ..
                    } => {
                        let camera_driver = self.camera.driver_mut::<YawPitch>();

                        if let Some(VirtualKeyCode::Z) = virtual_keycode {
                            camera_driver.rotate_yaw_pitch(-90.0, 0.0);
                        }
                        if let Some(VirtualKeyCode::X) = virtual_keycode {
                            camera_driver.rotate_yaw_pitch(90.0, 0.0);
                        }
                    }

                    Event::MainEventsCleared { .. } => {
                        let now = Instant::now();
                        let delta_time = now.duration_since(self.last_time).as_secs_f32();

                        self.last_time = now;
                        self.camera.update(delta_time);

                        // TODO: it's probably not the most efficient to recreate whole command buffer every frame?
                        self.create_command_buffers();
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

            layout(binding = 0) uniform UniformBufferObject {
                mat4 model;
                mat4 view;
                mat4 proj;
            } ubo;

            layout(location = 0) in vec3 position;
            layout(location = 1) in vec3 color;
            layout(location = 2) in vec2 tex;

            layout(location = 0) out vec3 f_color;
            layout(location = 1) out vec2 f_tex_coord;

            out gl_PerVertex {
                vec4 gl_Position;
            };

            void main() {
                gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
                f_tex_coord = tex;
                f_color = color;
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

            layout(location = 0) in vec3 f_color;
            layout(location = 1) in vec2 f_tex_coord;

            layout(location = 0) out vec4 out_color;

            void main() {
                // out_color = vec4(f_color * texture(tex_sampler, f_tex_coord).rgb, 1.0);
                out_color = texture(tex_sampler, f_tex_coord);
            }
        "
    }
}
