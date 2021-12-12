use std::{collections::HashSet, iter::FromIterator, sync::Arc};

use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Features, Queue,
    },
    format::Format,
    image::{ImageUsage, SampleCount, SwapchainImage},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        layers_list, ApplicationInfo, Instance, InstanceExtensions,
    },
    sampler::{BorderColor, Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{
        ColorSpace, CompositeAlpha, PresentMode, SupportedPresentModes, Surface, Swapchain,
    },
    sync::SharingMode,
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event_loop::EventLoop,
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

// TODO: should i separate the renderer context from windowing stuff?
pub struct Context {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    /**
     * This is why we need to wrap event_loop into Option
     *
     * https://stackoverflow.com/questions/67349506/ownership-issues-when-attempting-to-work-with-member-variables-passed-to-closure
     *
     * I don't really understand how it works and why exactly it's needed.
     */
    pub event_loop: Option<EventLoop<()>>,
    #[allow(dead_code)]
    debug_callback: Option<DebugCallback>,
    pub surface: Arc<Surface<Window>>,
    #[allow(dead_code)]
    physical_device_index: usize, // can't store PhysicalDevice directly (lifetime issues)
    pub device: Arc<Device>,
    pub graphics_queue: Arc<Queue>,
    #[allow(dead_code)]
    present_queue: Arc<Queue>,
    pub swap_chain: Arc<Swapchain<Window>>,
    pub swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    pub depth_format: Format,
    pub sample_count: SampleCount,
    pub image_sampler: Arc<Sampler>,
    pub depth_sampler: Arc<Sampler>,
}

impl Context {
    pub fn initialize() -> Context {
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

        let image_sampler = Self::create_image_sampler(&device);
        let depth_sampler = Self::create_depth_sampler(&device);

        Context {
            instance,
            debug_callback,
            surface,
            event_loop: Some(event_loop),
            physical_device_index,
            device,
            graphics_queue,
            present_queue,
            swap_chain,
            swap_chain_images,

            depth_format: Self::find_depth_format(),
            sample_count: Self::find_sample_count(),
            image_sampler,
            depth_sampler,
        }
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
            application_name: Some("Application".into()),
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

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_utils = true;
        }

        extensions
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

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let supported_extensions = PhysicalDevice::supported_extensions(device);
        supported_extensions.intersection(&DEVICE_EXTENSIONS) == DEVICE_EXTENSIONS
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

    pub fn recreate_swap_chain(&mut self) {
        let dimensions: [u32; 2] = self.surface.window().inner_size().into();

        let (swap_chain, images) = Swapchain::recreate(&self.swap_chain)
            .dimensions(dimensions)
            .build()
            .unwrap();

        self.swap_chain = swap_chain;
        self.swap_chain_images = images;
    }

    fn choose_swap_surface_format(
        available_formats: &[(Format, ColorSpace)],
    ) -> (Format, ColorSpace) {
        *available_formats
            .iter()
            .find(|(format, color_space)| {
                *format == Format::R16G16B16A16_UNORM && *color_space == ColorSpace::SrgbNonLinear
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

    fn find_depth_format() -> Format {
        // https://github.com/matthew-russo/vulkan-tutorial-rs/blob/26_depth_buffering/src/bin/26_depth_buffering.rs.diff#L115
        Format::D16_UNORM
    }

    fn find_sample_count() -> SampleCount {
        // https://github.com/matthew-russo/vulkan-tutorial-rs/blob/29_multisampling/src/bin/29_multisampling.rs.diff#L52-L59

        // macOS doesn't support MSAA8, so we'll use MSAA4 instead.
        SampleCount::Sample4
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

    fn create_depth_sampler(device: &Arc<Device>) -> Arc<Sampler> {
        Sampler::new(
            device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            0.0,
            1.0,
            0.0,
            1.0,
        )
        .unwrap()
    }
}
