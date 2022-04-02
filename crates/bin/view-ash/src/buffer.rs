use std::mem::{align_of, size_of};

use ash::{vk, Device};

use crate::{
    context::VkContext,
    utils::{execute_one_time_commands, find_memory_type},
};

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        vk_context: &VkContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        mem_properties: vk::MemoryPropertyFlags,
    ) -> Self {
        let device = vk_context.device();
        let buffer = {
            let buffer_info = vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .build();

            unsafe { device.create_buffer(&buffer_info, None).unwrap() }
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = {
            let mem_type = find_memory_type(
                mem_requirements,
                vk_context.get_mem_properties(),
                mem_properties,
            );

            let alloc_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(mem_requirements.size)
                .memory_type_index(mem_type)
                .build();
            unsafe { device.allocate_memory(&alloc_info, None).unwrap() }
        };

        unsafe { device.bind_buffer_memory(buffer, memory, 0).unwrap() }

        Buffer {
            buffer,
            memory,
            size: mem_requirements.size,
        }
    }

    pub fn new_device_local_buffer_with_data<A, T: Copy>(
        vk_context: &VkContext,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        usage: vk::BufferUsageFlags,
        data: &[T],
    ) -> Self {
        let device = vk_context.device();
        let size = (data.len() * size_of::<T>()) as vk::DeviceSize;
        let staging_buffer = Self::new(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = device
                .map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())
                .unwrap();
            let mut align =
                ash::util::Align::new(data_ptr, align_of::<A>() as _, staging_buffer.size);

            align.copy_from_slice(data);
            device.unmap_memory(staging_buffer.memory);
        };

        let buffer = Self::new(
            vk_context,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            command_pool,
            transfer_queue,
            staging_buffer.buffer,
            buffer.buffer,
            size,
        );

        staging_buffer.destroy(vk_context);

        buffer
    }

    fn copy_buffer(
        device: &Device,
        command_pool: vk::CommandPool,
        transfer_queue: vk::Queue,
        src: vk::Buffer,
        dst: vk::Buffer,
        size: vk::DeviceSize,
    ) {
        execute_one_time_commands(&device, command_pool, transfer_queue, |buffer| {
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            let regions = [region];

            unsafe { device.cmd_copy_buffer(buffer, src, dst, &regions) };
        });
    }

    pub fn destroy(&self, vk_context: &VkContext) {
        unsafe {
            vk_context.device().destroy_buffer(self.buffer, None);
            vk_context.device().free_memory(self.memory, None);
        }
    }
}
