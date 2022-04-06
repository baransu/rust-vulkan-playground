mod buffer;
mod camera;
mod context;
mod swapchain_support_details;
mod texture;
mod utils;

use ash::{vk, Device};
use buffer::Buffer;
use camera::Camera;
use context::VkContext;
use env_logger::Env;
use glam::{Mat4, Quat, Vec3};
use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    mem::{align_of, size_of},
    path::Path,
    time::Instant,
};
use texture::Texture;
use utils::{create_image_view, execute_one_time_commands, find_memory_type};
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
    window::Window,
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct VulkanApp {
    window: Window,
    start_instant: Instant,
    resize_dimensions: Option<[u32; 2]>,
    vk_context: VkContext,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    transient_command_pool: vk::CommandPool,
    msaa_samples: vk::SampleCountFlags,
    color_texture: Texture,
    depth_texture: Texture,
    texture: Texture,
    model_index_count: usize,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    uniform_buffers: Vec<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    command_buffers: Vec<vk::CommandBuffer>,
    query_pool: vk::QueryPool,
    in_flight_frames: InFlightFrames,
    camera: Camera,
}

impl VulkanApp {
    pub fn new(event_loop: &EventLoop<()>) -> VulkanApp {
        let window = winit::window::WindowBuilder::new()
            .with_title("Vulkan")
            .with_resizable(false)
            .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.");

        let vk_context = VkContext::new(&window);

        let msaa_samples = vk_context.get_max_usable_sample_count();

        log::debug!("MSAA samples: {:?}", msaa_samples);

        let render_pass = Self::create_render_pass(&vk_context, msaa_samples);
        let descriptor_set_layout = Self::create_descriptor_set_layout(&vk_context.device);
        let (pipeline, layout) = Self::create_pipeline(
            &vk_context,
            msaa_samples,
            render_pass,
            descriptor_set_layout,
        );

        let command_pool =
            Self::create_command_pool(&vk_context, vk::CommandPoolCreateFlags::empty());

        let color_texture = Self::create_color_texture(&vk_context, command_pool, msaa_samples);

        let depth_texture = Self::create_depth_texture(&vk_context, command_pool, msaa_samples);

        let swapchain_framebuffers =
            Self::create_framebuffers(&vk_context, color_texture, depth_texture, render_pass);

        let transient_command_pool =
            Self::create_command_pool(&vk_context, vk::CommandPoolCreateFlags::TRANSIENT);

        let texture = Self::create_texture_image(&vk_context, command_pool);

        let (vertices, indices) = Self::load_model();
        let vertex_buffer =
            Self::create_vertex_buffer(&vk_context, transient_command_pool, &vertices);
        let index_buffer = Self::create_index_buffer(&vk_context, transient_command_pool, &indices);

        let uniform_buffers = Self::create_uniform_buffers(&vk_context);

        let descriptor_pool = Self::create_descriptor_pool(&vk_context);
        let descriptor_sets = Self::create_descriptor_sets(
            &vk_context.device,
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            vertices.len(),
            &vertex_buffer,
            texture,
        );

        let (command_buffers, query_pool) = Self::create_and_register_command_buffers(
            &vk_context,
            command_pool,
            &swapchain_framebuffers,
            render_pass,
            &index_buffer,
            indices.len(),
            layout,
            &descriptor_sets,
            pipeline,
        );

        let in_flight_frames = Self::create_sync_objects(&vk_context.device);

        Self {
            start_instant: Instant::now(),
            window,
            resize_dimensions: None,
            vk_context,
            render_pass,
            descriptor_set_layout,
            pipeline_layout: layout,
            pipeline,
            swapchain_framebuffers,
            command_pool,
            transient_command_pool,
            msaa_samples,
            color_texture,
            depth_texture,
            texture,
            model_index_count: indices.len(),
            vertex_buffer,
            index_buffer,
            uniform_buffers,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            query_pool,
            in_flight_frames,
            camera: Default::default(),
        }
    }

    fn create_render_pass(
        vk_context: &VkContext,
        msaa_samples: vk::SampleCountFlags,
    ) -> vk::RenderPass {
        let color_attachment_desc = vk::AttachmentDescription::builder()
            .format(vk_context.swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let depth_attachement_desc = vk::AttachmentDescription::builder()
            .format(vk_context.depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_desc = vk::AttachmentDescription::builder()
            .format(vk_context.swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        let attachment_descs = [
            color_attachment_desc,
            depth_attachement_desc,
            resolve_attachment_desc,
        ];

        let color_attachment_ref = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let resolve_attachment_ref = [vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpass_desc = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_ref)
            .resolve_attachments(&resolve_attachment_ref)
            .depth_stencil_attachment(&depth_attachment_ref)
            .build();
        let subpasses = [subpass_desc];

        let subpass_dep = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();

        let dependencies = [subpass_dep];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descs)
            .subpasses(&subpasses)
            .dependencies(&dependencies)
            .build();

        unsafe {
            vk_context
                .device
                .create_render_pass(&render_pass_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout {
        let vertex_binding = Vertex::get_descriptor_set_layout_binding();
        let ubo_binding = UniformBufferObject::get_descriptor_set_layout_binding();
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [vertex_binding, ubo_binding, sampler_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_pool(vk_context: &VkContext) -> vk::DescriptorPool {
        let size = vk_context.swapchain_image_views.len() as _;

        let vertex_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: size,
        };
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: size,
        };

        let pool_sizes = [vertex_pool_size, ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(size)
            .build();

        unsafe {
            vk_context
                .device
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_sets(
        device: &Device,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        uniform_buffers: &[Buffer],
        vertex_count: usize,
        vertex_buffer: &Buffer,
        texture: Texture,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..uniform_buffers.len())
            .map(|_| layout)
            .collect::<Vec<_>>();
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .build();
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap() };

        descriptor_sets
            .iter()
            .zip(uniform_buffers.iter())
            .for_each(|(set, ubo_buffer)| {
                let vertex_info = [vk::DescriptorBufferInfo::builder()
                    .buffer(vertex_buffer.buffer)
                    .offset(0)
                    .range((size_of::<Vertex>() * vertex_count) as vk::DeviceSize)
                    .build()];
                let vertex_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&vertex_info)
                    .build();

                let ubo_info = [vk::DescriptorBufferInfo::builder()
                    .buffer(ubo_buffer.buffer)
                    .offset(0)
                    .range(size_of::<UniformBufferObject>() as vk::DeviceSize)
                    .build()];
                let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&ubo_info)
                    .build();

                let image_info = [vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view)
                    .sampler(texture.sampler.unwrap())
                    .build()];
                let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)
                    .build();

                let descriptor_writes = [
                    vertex_descriptor_write,
                    ubo_descriptor_write,
                    sampler_descriptor_write,
                ];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) }
            });

        descriptor_sets
    }

    fn create_pipeline(
        vk_context: &VkContext,
        msaa_samples: vk::SampleCountFlags,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> (vk::Pipeline, vk::PipelineLayout) {
        let vertex_source = Self::read_shader_from_file("shaders/shader.vert.spv");
        let fragment_source = Self::read_shader_from_file("shaders/shader.frag.spv");

        let vertex_shader_module = Self::create_shader_module(&vk_context.device, &vertex_source);
        let fragment_shader_module =
            Self::create_shader_module(&vk_context.device, &fragment_source);

        let entry_point_name = CString::new("main").unwrap();

        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(&entry_point_name)
            .build();

        let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(&entry_point_name)
            .build();

        let shader_state_infos = [vertex_shader_state_info, fragment_shader_state_info];

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder().build();

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: vk_context.swapchain_properties.extent.width as _,
            height: vk_context.swapchain_properties.extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk_context.swapchain_properties.extent,
        }];
        let viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewport)
            .scissors(&scissor)
            .build();

        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .build();

        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(msaa_samples)
            .sample_shading_enable(false)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default())
            .build();

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ZERO)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let layout = {
            let layouts = [descriptor_set_layout];
            let layout_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&layouts)
                .build();

            unsafe {
                vk_context
                    .device
                    .create_pipeline_layout(&layout_info, None)
                    .unwrap()
            }
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_infos)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0)
            .build();

        let create_infos = [pipeline_info];
        let pipeline = unsafe {
            vk_context
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .unwrap()[0]
        };

        unsafe {
            vk_context
                .device
                .destroy_shader_module(vertex_shader_module, None);
            vk_context
                .device
                .destroy_shader_module(fragment_shader_module, None);
        }

        (pipeline, layout)
    }

    fn read_shader_from_file<P: AsRef<std::path::Path>>(path: P) -> Vec<u32> {
        let mut file = std::fs::File::open(path).unwrap();
        ash::util::read_spv(&mut file).unwrap()
    }

    fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(code).build();
        unsafe { device.create_shader_module(&create_info, None).unwrap() }
    }

    fn create_framebuffers(
        vk_context: &VkContext,
        color_texture: Texture,
        depth_texture: Texture,
        render_pass: vk::RenderPass,
    ) -> Vec<vk::Framebuffer> {
        vk_context
            .swapchain_image_views
            .iter()
            .map(|view| [color_texture.view, depth_texture.view, *view])
            .map(|attachments| {
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(vk_context.swapchain_properties.extent.width)
                    .height(vk_context.swapchain_properties.extent.height)
                    .layers(1)
                    .build();

                unsafe {
                    vk_context
                        .device
                        .create_framebuffer(&framebuffer_info, None)
                        .unwrap()
                }
            })
            .collect::<Vec<_>>()
    }

    fn create_command_pool(
        vk_context: &VkContext,
        create_flags: vk::CommandPoolCreateFlags,
    ) -> vk::CommandPool {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(vk_context.queue_families_indices.graphics_index)
            .flags(create_flags)
            .build();

        unsafe {
            vk_context
                .device
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        }
    }

    fn create_color_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let format = vk_context.swapchain_properties.format.format;
        let (image, memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk_context.swapchain_properties.extent,
            1,
            msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        );

        Self::transition_image_layout(
            &vk_context.device,
            command_pool,
            vk_context.graphics_queue,
            image,
            1,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        let view = create_image_view(
            &vk_context.device,
            image,
            1,
            format,
            vk::ImageAspectFlags::COLOR,
        );

        Texture::new(image, memory, view, None)
    }

    fn create_depth_texture(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        msaa_samples: vk::SampleCountFlags,
    ) -> Texture {
        let (image, mem) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk_context.swapchain_properties.extent,
            1,
            msaa_samples,
            vk_context.depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        Self::transition_image_layout(
            &vk_context.device,
            command_pool,
            vk_context.graphics_queue,
            image,
            1,
            vk_context.depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let view = create_image_view(
            &vk_context.device,
            image,
            1,
            vk_context.depth_format,
            vk::ImageAspectFlags::DEPTH,
        );

        Texture::new(image, mem, view, None)
    }

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_texture_image(vk_context: &VkContext, command_pool: vk::CommandPool) -> Texture {
        let image = image::open("viking_room.png").unwrap().flipv();
        let image_as_rgb = image.to_rgba8();
        let width = image.width();
        let height = image.height();
        let max_mip_levels = (width.min(height) as f32).log2().floor() as u32 + 1;
        let extent = vk::Extent2D { width, height };
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * size_of::<u8>()) as vk::DeviceSize;

        let buffer = Buffer::new(
            vk_context,
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let ptr = vk_context
                .device
                .map_memory(buffer.memory, 0, image_size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(ptr, align_of::<u8>() as _, buffer.size);
            align.copy_from_slice(&pixels);
            vk_context.device.unmap_memory(buffer.memory);
        }

        let (image, image_memory) = Self::create_image(
            vk_context,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            extent,
            max_mip_levels,
            vk::SampleCountFlags::TYPE_1,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        );

        {
            Self::transition_image_layout(
                &vk_context.device,
                command_pool,
                vk_context.graphics_queue,
                image,
                max_mip_levels,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );

            Self::copy_buffer_to_image(
                &vk_context.device,
                command_pool,
                vk_context.graphics_queue,
                buffer.buffer,
                image,
                extent,
            );

            Self::generate_mipmaps(
                vk_context,
                command_pool,
                vk_context.graphics_queue,
                image,
                extent,
                vk::Format::R8G8B8A8_UNORM,
                max_mip_levels,
            );
        }

        buffer.destroy(vk_context);

        let image_view = create_image_view(
            &vk_context.device,
            image,
            max_mip_levels,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
        );

        let sampler = {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(max_mip_levels as _)
                .build();

            unsafe {
                vk_context
                    .device
                    .create_sampler(&sampler_info, None)
                    .unwrap()
            }
        };

        Texture::new(image, image_memory, image_view, Some(sampler))
    }

    fn create_image(
        vk_context: &VkContext,
        mem_properties: vk::MemoryPropertyFlags,
        extent: vk::Extent2D,
        mip_levels: u32,
        sample_count: vk::SampleCountFlags,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(sample_count)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let image = unsafe { vk_context.device.create_image(&image_info, None).unwrap() };
        let mem_requirements = unsafe { vk_context.device.get_image_memory_requirements(image) };
        let mem_type_index = find_memory_type(
            mem_requirements,
            vk_context.get_mem_properties(),
            mem_properties,
        );

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .build();

        let memory = unsafe {
            let mem = vk_context
                .device
                .allocate_memory(&alloc_info, None)
                .unwrap();
            vk_context.device.bind_image_memory(image, mem, 0).unwrap();
            mem
        };

        (image, memory)
    }

    fn transition_image_layout(
        device: &Device,
        command_pool: vk::CommandPool,
        transtion_queue: vk::Queue,
        image: vk::Image,
        mip_levels: u32,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        execute_one_time_commands(device, command_pool, transtion_queue, |buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    ),
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    _ => panic!(
                        "Unsupported layout transtion({:?} => {:?}).",
                        old_layout, new_layout
                    ),
                };

            let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = vk::ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format) {
                    mask |= vk::ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask)
                .build();

            unsafe {
                device.cmd_pipeline_barrier(
                    buffer,
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            }
        });
    }

    fn copy_buffer_to_image(
        device: &Device,
        command_pool: vk::CommandPool,
        transition_queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        extent: vk::Extent2D,
    ) {
        execute_one_time_commands(device, command_pool, transition_queue, |command_buffer| {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .build();

            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                )
            }
        })
    }

    fn generate_mipmaps(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            vk_context
                .instance
                .get_physical_device_format_properties(vk_context.physical_device, format)
        };
        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear blitting is not supported for format {:?}.", format)
        }

        execute_one_time_commands(&vk_context.device, command_pool, transfer_queue, |buffer| {
            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    layer_count: 1,
                    level_count: 1,
                    ..Default::default()
                })
                .build();

            let mut mip_width = extent.width as i32;
            let mut mip_height = extent.height as i32;
            for level in 1..mip_levels {
                let next_mip_width = if mip_width > 1 {
                    mip_width / 2
                } else {
                    mip_width
                };
                let next_mip_height = if mip_height > 1 {
                    mip_height / 2
                } else {
                    mip_height
                };

                barrier.subresource_range.base_mip_level = level - 1;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                unsafe {
                    vk_context.device.cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };

                let blit = vk::ImageBlit::builder()
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_mip_width,
                            y: next_mip_height,
                            z: 1,
                        },
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build();

                unsafe {
                    vk_context.device.cmd_blit_image(
                        buffer,
                        image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[blit],
                        vk::Filter::LINEAR,
                    )
                };

                barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                unsafe {
                    vk_context.device.cmd_pipeline_barrier(
                        buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };

                mip_width = next_mip_width;
                mip_height = next_mip_height;
            }

            barrier.subresource_range.base_mip_level = mip_levels - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                vk_context.device.cmd_pipeline_barrier(
                    buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            };
        });
    }

    fn load_model() -> (Vec<Vertex>, Vec<u32>) {
        log::debug!("Loading model...");

        let (models, _) = tobj::load_obj(
            &Path::new("viking_room.obj"),
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )
        .unwrap();

        let mesh = &models[0].mesh;
        let positions = mesh.positions.as_slice();
        let coords = mesh.texcoords.as_slice();
        let vertex_count = mesh.positions.len() / 3;
        let normals = mesh.normals.as_slice();

        log::debug!(
            "normals len: {}, vertex count: {}",
            normals.len(),
            vertex_count
        );

        let mut vertices = Vec::with_capacity(vertex_count);
        for i in 0..vertex_count {
            let vx = positions[i * 3];
            let vy = positions[i * 3 + 1];
            let vz = positions[i * 3 + 2];

            let nx = normals[i * 3];
            let ny = normals[i * 3 + 1];
            let nz = normals[i * 3 + 2];

            let tu = coords[i * 2];
            let tv = coords[i * 2 + 1];

            let vertex = Vertex {
                pos: [vx, vy, vz],
                normal: [nx, ny, nz, 1.0],
                coords: [tu, tv],
            };
            vertices.push(vertex);
        }

        let indices = mesh.indices.clone();

        let (_vertex_count, remap) = meshopt::generate_vertex_remap(&vertices, Some(&indices));

        let final_vertices = meshopt::remap_vertex_buffer(&vertices, vertices.len(), &remap);
        let final_indices = meshopt::remap_index_buffer(Some(&indices), indices.len(), &remap);

        (final_vertices, final_indices)
    }

    fn create_vertex_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        vertices: &[Vertex],
    ) -> Buffer {
        Buffer::new_device_local_buffer_with_data::<u32, _>(
            vk_context,
            command_pool,
            vk_context.graphics_queue,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            &vertices,
        )
    }

    fn create_index_buffer(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        indices: &[u32],
    ) -> Buffer {
        Buffer::new_device_local_buffer_with_data::<u16, _>(
            vk_context,
            command_pool,
            vk_context.graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        )
    }

    fn create_uniform_buffers(vk_context: &VkContext) -> Vec<Buffer> {
        let count = vk_context.swapchain_image_views.len() as _;
        let size = size_of::<UniformBufferObject>() as vk::DeviceSize;
        let mut buffers = Vec::new();

        for _ in 0..count {
            let buffer = Buffer::new(
                vk_context,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer);
        }

        buffers
    }

    fn create_and_register_command_buffers(
        vk_context: &VkContext,
        pool: vk::CommandPool,
        framebuffers: &[vk::Framebuffer],
        render_pass: vk::RenderPass,
        index_buffer: &Buffer,
        index_count: usize,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
        graphics_pipeline: vk::Pipeline,
    ) -> (Vec<vk::CommandBuffer>, vk::QueryPool) {
        let query_pool = {
            let create_info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2)
                .build();

            unsafe {
                vk_context
                    .device
                    .create_query_pool(&create_info, None)
                    .unwrap()
            }
        };

        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(framebuffers.len() as _)
            .build();

        let buffers = unsafe {
            vk_context
                .device
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        };

        buffers.iter().enumerate().for_each(|(i, buffer)| {
            let buffer = *buffer;
            let framebuffer = framebuffers[i];

            // begin command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build();

                unsafe {
                    vk_context
                        .device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
                        .unwrap()
                }
            }

            unsafe {
                vk_context
                    .device
                    .cmd_reset_query_pool(buffer, query_pool, 0, 2);
            }

            unsafe {
                vk_context.device.cmd_write_timestamp(
                    buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    0,
                );
            }

            // begin render pass
            {
                let clear_values = [
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk_context.swapchain_properties.extent,
                    })
                    .clear_values(&clear_values)
                    .build();

                unsafe {
                    vk_context.device.cmd_begin_render_pass(
                        buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                }
            }

            // bind pipeline
            unsafe {
                vk_context.device.cmd_bind_pipeline(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                )
            };

            // bind index buffer
            unsafe {
                vk_context.device.cmd_bind_index_buffer(
                    buffer,
                    index_buffer.buffer,
                    0,
                    vk::IndexType::UINT32,
                )
            };

            unsafe {
                vk_context.device.cmd_bind_descriptor_sets(
                    buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[i..=i],
                    &[],
                )
            };

            // draw
            unsafe {
                vk_context
                    .device
                    .cmd_draw_indexed(buffer, index_count as _, 1, 0, 0, 0)
            };

            // end render pass
            unsafe { vk_context.device.cmd_end_render_pass(buffer) };

            unsafe {
                vk_context.device.cmd_write_timestamp(
                    buffer,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    query_pool,
                    1,
                );
            }

            // end command buffer
            unsafe { vk_context.device.end_command_buffer(buffer).unwrap() };
        });

        (buffers, query_pool)
    }

    fn create_sync_objects(device: &Device) -> InFlightFrames {
        let mut sync_objects_vec = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let render_finished_semaphore = {
                let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
                unsafe { device.create_semaphore(&semaphore_info, None).unwrap() }
            };

            let in_flight_fence = {
                let fence_info = vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build();
                unsafe { device.create_fence(&fence_info, None).unwrap() }
            };

            let sync_objects = SyncObjects {
                image_available_semaphore,
                render_finished_semaphore,
                fence: in_flight_fence,
            };
            sync_objects_vec.push(sync_objects)
        }

        InFlightFrames::new(sync_objects_vec)
    }

    fn draw_frame(&mut self) {
        log::trace!("Drawing frame...");

        let sync_objects = self.in_flight_frames.next().unwrap();
        let image_available_semaphore = sync_objects.image_available_semaphore;
        let render_finished_semaphore = sync_objects.render_finished_semaphore;
        let in_flight_fence = sync_objects.fence;

        let wait_fences = [in_flight_fence];

        unsafe {
            self.vk_context
                .device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap();
        };

        let result = unsafe {
            self.vk_context.swapchain.acquire_next_image(
                self.vk_context.swapchain_khr,
                std::u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index = match result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };

        unsafe { self.vk_context.device.reset_fences(&wait_fences).unwrap() };

        self.update_uniform_buffers(image_index);

        let wait_semaphores = [image_available_semaphore];
        let signal_semaphores = [render_finished_semaphore];

        // submit command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[image_index as usize]];

            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build();
            let submits = [submit_info];

            unsafe {
                self.vk_context
                    .device
                    .queue_submit(self.vk_context.graphics_queue, &submits, in_flight_fence)
                    .unwrap()
            }
        }

        {
            let swapchains = [self.vk_context.swapchain_khr];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices)
                .build();

            let result = unsafe {
                self.vk_context
                    .swapchain
                    .queue_present(self.vk_context.present_queue, &present_info)
            };

            match result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    log::debug!("Result out of date...");
                    self.recreate_swapchain();
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            if self.resize_dimensions.is_some() {
                self.recreate_swapchain();
            }
        }
    }

    fn recreate_swapchain(&mut self) {
        log::debug!("Recreating swapchain.");

        unsafe { self.vk_context.device.device_wait_idle().unwrap() };

        self.cleanup_swapchain();

        let dimensions = self.resize_dimensions.unwrap_or([
            self.vk_context.swapchain_properties.extent.width,
            self.vk_context.swapchain_properties.extent.height,
        ]);

        self.vk_context.recreate_swapchain(dimensions);

        let render_pass = Self::create_render_pass(&self.vk_context, self.msaa_samples);
        let (pipeline, layout) = Self::create_pipeline(
            &self.vk_context,
            self.msaa_samples,
            render_pass,
            self.descriptor_set_layout,
        );

        let color_texture =
            Self::create_color_texture(&self.vk_context, self.command_pool, self.msaa_samples);

        let depth_texture =
            Self::create_depth_texture(&self.vk_context, self.command_pool, self.msaa_samples);

        let swapchain_framebuffers =
            Self::create_framebuffers(&self.vk_context, color_texture, depth_texture, render_pass);

        let (command_buffers, query_pool) = Self::create_and_register_command_buffers(
            &self.vk_context,
            self.command_pool,
            &swapchain_framebuffers,
            render_pass,
            &self.index_buffer,
            self.model_index_count,
            layout,
            &self.descriptor_sets,
            pipeline,
        );

        self.render_pass = render_pass;
        self.pipeline = pipeline;
        self.pipeline_layout = layout;
        self.color_texture = color_texture;
        self.depth_texture = depth_texture;
        self.swapchain_framebuffers = swapchain_framebuffers;
        self.command_buffers = command_buffers;
        self.query_pool = query_pool;
        self.resize_dimensions = None;

        log::debug!("Swapchain recreated.");
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.vk_context
                .device
                .destroy_query_pool(self.query_pool, None);

            self.depth_texture.destroy(&self.vk_context.device);
            self.color_texture.destroy(&self.vk_context.device);
            self.swapchain_framebuffers
                .iter()
                .for_each(|f| self.vk_context.device.destroy_framebuffer(*f, None));
            self.vk_context
                .device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.vk_context.device.destroy_pipeline(self.pipeline, None);
            self.vk_context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.vk_context
                .device
                .destroy_render_pass(self.render_pass, None);

            self.vk_context.cleanup_swapchain();
        }
    }

    fn update_uniform_buffers(&mut self, current_image: u32) {
        let elapsed = self.start_instant.elapsed();
        let elapsed: f32 =
            elapsed.as_secs() as f32 + (elapsed.subsec_millis() as f32) / 1_000 as f32;

        let aspect_ratio = self.vk_context.width as f32 / self.vk_context.height as f32;

        let rotation = Quat::from_euler(
            glam::EulerRot::XYZ,
            -90.0_f32.to_radians(),
            0.0,
            (45.0 * elapsed).to_radians(),
        );

        let ubo = UniformBufferObject {
            model: Mat4::from_rotation_translation(rotation, Vec3::ZERO).to_cols_array_2d(),
            view: self.camera.look_at_matrix().to_cols_array_2d(),
            proj: self.camera.projection(aspect_ratio).to_cols_array_2d(),
        };
        let ubos = [ubo];

        let buffer_mem = self.uniform_buffers[current_image as usize].memory;
        let size = size_of::<UniformBufferObject>() as vk::DeviceSize;
        unsafe {
            let data_ptr = self
                .vk_context
                .device
                .map_memory(buffer_mem, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align = ash::util::Align::new(data_ptr, align_of::<f32>() as _, size);

            align.copy_from_slice(&ubos);
            self.vk_context.device.unmap_memory(buffer_mem);
        }
    }

    fn update(&mut self, keys: &HashMap<VirtualKeyCode, ElementState>, dt: f32) {
        let camera_speed = 10.0 * dt;

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

    pub fn main_loop(mut self, mut event_loop: EventLoop<()>) {
        let timestamp_period = unsafe {
            self.vk_context
                .instance
                .get_physical_device_properties(self.vk_context.physical_device)
                .limits
                .timestamp_period as f64
        };

        // Delta times are filtered over _this many_ frames.
        const DT_FILTER_WIDTH: usize = 10;
        // Past delta times used for filtering
        let mut dt_queue: VecDeque<f32> = VecDeque::with_capacity(DT_FILTER_WIDTH);
        let mut delta_time: f32 = 0.0;
        let mut last_time = Instant::now();
        // Fake the first frame's delta time. In the first frame, shaders
        // and pipelines are be compiled, so it will most likely have a spike.
        let mut fake_dt_countdown: i32 = 1;

        let mut fps_cpu_avg = 0.0;
        let mut fps_gpu_avg = 0.0;

        let mut mouse_buttons: HashMap<MouseButton, ElementState> = HashMap::new();
        let mut keyboard_buttons: HashMap<VirtualKeyCode, ElementState> = HashMap::new();

        let mut rotation_x = 0.0;
        let mut rotation_y = 0.0;

        let original_rotation = self.camera.rotation;

        event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(PhysicalSize { width, height }) => {
                        if width != self.vk_context.width || height != self.vk_context.height {
                            self.resize_dimensions = Some([width as u32, height as u32]);
                        }
                    }
                    WindowEvent::KeyboardInput { input, .. } => match input {
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(virtual_keycode),
                            state,
                            ..
                        } => match (virtual_keycode, state) {
                            (VirtualKeyCode::Escape, winit::event::ElementState::Pressed) => {
                                dbg!();
                                *control_flow = ControlFlow::Exit;
                            }
                            _ => {
                                keyboard_buttons.insert(virtual_keycode, state);
                            }
                        },
                        _ => {}
                    },
                    WindowEvent::MouseInput { state, button, .. } => {
                        mouse_buttons.insert(button, state);
                    }
                    _ => {}
                },
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta, .. },
                    ..
                } => {
                    match mouse_buttons.get(&MouseButton::Left) {
                        Some(&ElementState::Pressed) => {
                            let sensitivity = 5.0 * delta_time;
                            let (x, y) = delta;

                            rotation_x += x as f32 * sensitivity;
                            rotation_y += y as f32 * sensitivity;

                            let y_quat = Quat::from_axis_angle(Vec3::X, -rotation_y);
                            let x_quat = Quat::from_axis_angle(Vec3::Y, -rotation_x);

                            self.camera.rotation = original_rotation * x_quat * y_quat;
                        }
                        _ => {}
                    };
                }
                Event::MainEventsCleared => self.window.request_redraw(),
                Event::RedrawRequested(_window_id) => {
                    // Filter the frame time before passing it to the application and renderer.
                    // Fluctuations in frame rendering times cause stutter in animations,
                    // and time-dependent effects (such as motion blur).
                    //
                    // Should applications need unfiltered delta time, they can calculate
                    // it themselves, but it's good to pass the filtered time so users
                    // don't need to worry about it.
                    delta_time = {
                        let now = Instant::now();
                        let dt_duration = now - last_time;
                        last_time = now;

                        let dt_raw = dt_duration.as_secs_f32();

                        // >= because rendering (and thus the spike) happens _after_ this.
                        if fake_dt_countdown >= 0 {
                            // First frame. Return the fake value.
                            fake_dt_countdown -= 1;
                            dt_raw.min(1.0 / 60.0)
                        } else {
                            // Not the first frame. Start averaging.

                            if dt_queue.len() >= DT_FILTER_WIDTH {
                                dt_queue.pop_front();
                            }

                            dt_queue.push_back(dt_raw);
                            dt_queue.iter().copied().sum::<f32>() / dt_queue.len() as f32
                        }
                    };

                    let start_time = Instant::now();

                    self.update(&keyboard_buttons, delta_time);

                    self.draw_frame();

                    let end_time = Instant::now();

                    fps_cpu_avg = fps_cpu_avg * 0.9
                        + (end_time.duration_since(start_time).as_secs_f64() * 1000.0) * 0.1;

                    let results = {
                        unsafe {
                            let mut query_results = [0u64; 2];
                            self.vk_context
                                .device
                                .get_query_pool_results(
                                    self.query_pool,
                                    0,
                                    2,
                                    &mut query_results,
                                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                                )
                                .unwrap();
                            query_results
                        }
                    };

                    let gpu_start_time = results[0] as f64 * timestamp_period * 1e-6;
                    let gpu_end_time = results[1] as f64 * timestamp_period * 1e-6;

                    fps_gpu_avg = fps_gpu_avg * 0.9 + (gpu_end_time - gpu_start_time) * 0.1;

                    self.window.set_title(
                        format!("cpu: {:0.2}ms; gpu: {:0.2}ms", fps_cpu_avg, fps_gpu_avg).as_str(),
                    );
                }
                _ => {}
            }
        })
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");

        unsafe {
            self.vk_context.device.device_wait_idle().unwrap();
        }

        self.cleanup_swapchain();

        self.in_flight_frames.destroy(&self.vk_context.device);
        unsafe {
            self.vk_context
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.vk_context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.uniform_buffers
                .iter()
                .for_each(|b| b.destroy(&self.vk_context));
            self.index_buffer.destroy(&self.vk_context);
            self.vertex_buffer.destroy(&self.vk_context);
            self.texture.destroy(&self.vk_context.device);
            self.vk_context
                .device
                .destroy_command_pool(self.transient_command_pool, None);
            self.vk_context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

#[derive(Clone, Copy)]
struct SyncObjects {
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    fence: vk::Fence,
}

impl SyncObjects {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_fence(self.fence, None);
        }
    }
}

struct InFlightFrames {
    sync_objects: Vec<SyncObjects>,
    current_frame: usize,
}

impl InFlightFrames {
    fn new(sync_objects: Vec<SyncObjects>) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    fn destroy(&self, device: &Device) {
        self.sync_objects.iter().for_each(|o| o.destroy(&device));
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next)
    }
}

#[derive(Default, Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 4],
    coords: [f32; 2],
}

impl Vertex {
    fn get_descriptor_set_layout_binding() -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
#[repr(C)]
struct UniformBufferObject {
    model: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
}

impl UniformBufferObject {
    fn get_descriptor_set_layout_binding() -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()
    }
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();

    let event_loop = EventLoop::new();

    let vulkan_app = VulkanApp::new(&event_loop);
    vulkan_app.main_loop(event_loop)
}

fn is_pressed(keys: &HashMap<VirtualKeyCode, ElementState>, key: VirtualKeyCode) -> bool {
    match keys.get(&key) {
        Some(&ElementState::Pressed) => true,
        _ => false,
    }
}
