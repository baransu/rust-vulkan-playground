use ash::{vk, Device};

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    mip_levels: u32,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageView {
    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        })
        .build();

    unsafe { device.create_image_view(&create_info, None).unwrap() }
}

pub fn execute_one_time_commands<F: FnOnce(vk::CommandBuffer)>(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    executor: F,
) {
    let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();

        unsafe { device.allocate_command_buffers(&alloc_info).unwrap()[0] }
    };

    // begin recording
    {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap()
        };
    }

    // execute user function
    executor(command_buffer);

    // end recording
    unsafe { device.end_command_buffer(command_buffer).unwrap() };

    // submit and wait
    {
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer])
            .build();

        unsafe {
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .unwrap();

            device.queue_wait_idle(queue).unwrap();
        }
    }

    // free
    unsafe { device.free_command_buffers(command_pool, &[command_buffer]) };
}

pub fn find_memory_type(
    requirements: vk::MemoryRequirements,
    mem_properties: vk::PhysicalDeviceMemoryProperties,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_properties.memory_type_count {
        if requirements.memory_type_bits & (1 << i) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(required_properties)
        {
            return i;
        }
    }
    panic!("Failed to find suitable memory type.")
}
