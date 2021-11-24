use core::num;
use std::{
    collections::HashSet,
    iter::FromIterator,
    sync::{Arc, Mutex},
    time::Instant,
};

use cgmath::{Deg, Matrix4, Point3, Rad, Vector3};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
    },
    descriptor_set::{
        persistent::PersistentDescriptorSetBuilder, DescriptorSet, PersistentDescriptorSet,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Features, Queue,
    },
    format::Format,
    image::{view::ImageView, ImageUsage, SwapchainImage},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, ApplicationInfo, Instance, InstanceExtensions,
    },
    memory::pool::{PotentialDedicatedAllocation, StdMemoryPoolAlloc},
    pipeline::{viewport::Viewport, GraphicsPipeline, PipelineBindPoint},
    render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass},
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
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_LUNARG_standard_validation"];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const DEVICE_EXTENSIONS: DeviceExtensions = DeviceExtensions {
    khr_swapchain: true,
    ..DeviceExtensions::none()
};

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    fn new(position: [f32; 2], color: [f32; 3]) -> Vertex {
        Vertex { position, color }
    }
}

vulkano::impl_vertex!(Vertex, position, color);

fn vertices() -> [Vertex; 4] {
    [
        Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0]),
        Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0]),
        Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0]),
        Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0]),
    ]
}

fn indices() -> [u16; 6] {
    [0, 1, 2, 2, 3, 0]
}

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
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

    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    uniform_buffers: Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,

    start_time: Instant,
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

        let render_pass = Self::create_render_pass(&device, swap_chain.format());

        let graphics_pipeline =
            Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers =
            Self::create_swap_chain_framebuffers(&swap_chain_images, &render_pass);

        let previous_frame_end = Some(Self::create_sync_objects(&device));

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
            uniform_buffers: vec![],
            previous_frame_end,
            recreate_swap_chain: false,

            start_time: Instant::now(),
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

    fn create_render_pass(device: &Arc<Device>, surface_format: Format) -> Arc<RenderPass> {
        Arc::new(
            single_pass_renderpass!(device.clone(),
                    attachments: {
                        color: {
                            load: Clear,
                            store: Store,
                            format: surface_format,
                            samples: 1,
                        }
                    },
                    pass: {
                        color: [color],
                        depth_stencil: {}
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

        self.render_pass = Self::create_render_pass(&self.device, self.swap_chain.format());
        self.graphics_pipeline = Self::create_graphics_pipeline(
            &self.device,
            self.swap_chain.dimensions(),
            &self.render_pass,
        );

        self.swap_chain_framebuffers =
            Self::create_swap_chain_framebuffers(&self.swap_chain_images, &self.render_pass);

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
                .viewports_dynamic_scissors_irrelevant(1)
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(), // = default
        );

        pipeline
    }

    fn create_swap_chain_framebuffers(
        swap_chain_images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images
            .iter()
            .map(|image| {
                let attachment = ImageView::new(image.clone()).unwrap();

                let framebuffer: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(attachment)
                        .unwrap()
                        .build()
                        .unwrap(),
                );

                framebuffer
            })
            .collect::<Vec<_>>()
    }

    fn create_command_buffers(&mut self) {
        let vertex_buffer = create_vertex_buffer(&self.graphics_queue);
        let index_buffer = create_index_buffer(&self.graphics_queue);

        let uniform_buffer = Self::create_uniform_buffer(
            &self.device,
            self.start_time,
            self.swap_chain.dimensions(),
        );

        let descriptor_sets =
            Self::create_descriptor_sets(&self.graphics_pipeline, &uniform_buffer);

        let index_count = indices().len() as u32;

        let queue_family = self.graphics_queue.family();

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
            .map(|framebuffer| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.device.clone(),
                    queue_family,
                    CommandBufferUsage::SimultaneousUse,
                )
                .unwrap();

                builder
                    .begin_render_pass(
                        framebuffer.clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into()],
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(self.graphics_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        self.graphics_pipeline.layout().clone(),
                        0,
                        descriptor_sets.clone(),
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(index_count, 1, 0, 0, 0)
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
        start_time: Instant,
        dimensions_u32: [u32; 2],
    ) -> Arc<CpuAccessibleBuffer<UniformBufferObject>> {
        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];

        let uniform_buffer = Self::update_uniform_buffer(start_time, dimensions);

        let buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
            false,
            uniform_buffer,
        )
        .unwrap();

        buffer
    }

    fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2]) -> UniformBufferObject {
        let duration = Instant::now().duration_since(start_time);
        let elapsed = (duration.as_secs() * 1000) + u64::from(duration.subsec_millis());

        let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.180)));

        let view = Matrix4::look_at(
            Point3::new(2.0, 2.0, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let mut proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            10.0,
        );

        proj.y.y *= -1.0;

        UniformBufferObject { model, view, proj }
    }

    fn create_descriptor_sets(
        graphics_pipeline: &Arc<GraphicsPipeline>,
        uniform_buffer: &Arc<CpuAccessibleBuffer<UniformBufferObject>>,
    ) -> Arc<PersistentDescriptorSet> {
        let layout = graphics_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();

        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        set_builder.add_buffer(uniform_buffer.clone()).unwrap();

        Arc::new(set_builder.build().unwrap())
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

                    Event::RedrawEventsCleared { .. } => {
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

fn create_vertex_buffer(graphics_queue: &Arc<Queue>) -> Arc<ImmutableBuffer<[Vertex]>> {
    let (buffer, future) = ImmutableBuffer::from_iter(
        vertices().iter().cloned(),
        BufferUsage::vertex_buffer(),
        // TODO: idealy it should be transfer queue?
        graphics_queue.clone(),
    )
    .unwrap();

    future.flush().unwrap();

    buffer
}

fn create_index_buffer(graphics_queue: &Arc<Queue>) -> Arc<ImmutableBuffer<[u16]>> {
    let (buffer, future) = ImmutableBuffer::from_iter(
        indices().iter().cloned(),
        BufferUsage::index_buffer(),
        graphics_queue.clone(),
    )
    .unwrap();

    future.flush().unwrap();

    buffer
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

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec3 color;

            layout(location = 0) out vec3 f_color;

            out gl_PerVertex {
                vec4 gl_Position;
            };

            void main() {
                gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 0.0, 1.0);
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

            layout(location = 0) in vec3 f_color;

            layout(location = 0) out vec4 out_color;

            void main() {
                out_color = vec4(f_color, 1.0);
            }
        "
    }
}
