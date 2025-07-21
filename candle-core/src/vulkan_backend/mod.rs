use std::{ffi::c_char, rc::Rc};

use ash::{
    vk::{
        self, ApplicationInfo, Buffer, BufferCreateInfo, CommandBufferAllocateInfo, CommandBufferBeginInfo, CommandBufferUsageFlags, CommandPoolCreateFlags, CommandPoolCreateInfo, DescriptorBufferInfo, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSetAllocateInfo, DescriptorSetLayoutBinding, DescriptorType, DeviceCreateInfo, DeviceMemory, DeviceQueueCreateInfo, Fence, FenceCreateInfo, InstanceCreateInfo, PipelineBindPoint, SemaphoreGetZirconHandleInfoFUCHSIA, ShaderModuleCreateInfo, ShaderStageFlags, SpecializationInfo, SubmitInfo, WriteDescriptorSet
    },
    Entry, Instance,
};
use rspirv::{binary::Parser, dr::Loader};

use crate::{
    backend::{BackendDevice, BackendStorage},
    CpuStorage, DType, Device, DeviceLocation, Tensor,
};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

#[derive(Clone)]
pub struct VulkanDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,
    pub(crate) handles: Rc<VkHandles>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("id", &self.id)
            .finish()
    }
}

struct VkHandles {
    instance: ash::Instance,
    pdevice: ash::vk::PhysicalDevice,
    device: ash::Device,
    queue_family_index: u32,
    queue: ash::vk::Queue,
    pipeline: ash::vk::Pipeline,
    pipeline_layout: ash::vk::PipelineLayout,
    group_0: ash::vk::DescriptorSetLayout,
    group_1: ash::vk::DescriptorSetLayout,
    command_pool: ash::vk::CommandPool,
    command_buffers: Vec<ash::vk::CommandBuffer>,
    descriptor_pool: ash::vk::DescriptorPool,
}

#[repr(u32)]
enum KernelMethod {
    None = 0,
    Add = 1,
    Sub = 2,
}

#[repr(C)]
struct PushConstants {
    method: KernelMethod
}

fn dispatch_binary(handles: &VkHandles, cmd_buffer_i: usize, method: KernelMethod, l: Buffer, r: Buffer, o: Buffer, group: (u32, u32, u32), fence: Fence) {
    let command_buffer = handles.command_buffers[cmd_buffer_i];

    unsafe {
        let begin_info =
            CommandBufferBeginInfo::default().flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        handles
            .device
            .begin_command_buffer(command_buffer, &begin_info);
        handles.device.cmd_bind_pipeline(
            command_buffer,
            PipelineBindPoint::COMPUTE,
            handles.pipeline,
        );

        // group 0 and group 1
        let set_layouts = [handles.group_0, handles.group_1];
        let allocate_info = DescriptorSetAllocateInfo::default()
            .set_layouts(&set_layouts)
            .descriptor_pool(handles.descriptor_pool);
        let descriptor_sets = handles
            .device
            .allocate_descriptor_sets(&allocate_info)
            .unwrap();

        // Group 0
        let buffer_info = [
            DescriptorBufferInfo::default()
                .buffer(l)
                .range(ash::vk::WHOLE_SIZE),
            DescriptorBufferInfo::default()
                .buffer(r)
                .range(ash::vk::WHOLE_SIZE),
        ];
        let g0 = WriteDescriptorSet::default()
            .buffer_info(&buffer_info)
            .dst_set(descriptor_sets[0])
            .descriptor_count(2)
            .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC);

        // Group 1
        let buffer_info = [DescriptorBufferInfo::default()
            .buffer(o)
            .range(ash::vk::WHOLE_SIZE)];
        let g1 = WriteDescriptorSet::default()
            .buffer_info(&buffer_info)
            .dst_set(descriptor_sets[1])
            .descriptor_count(1)
            .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC);

        let descriptor_writes = vec![g0, g1];
        let descriptor_copies = vec![];
        handles
            .device
            .update_descriptor_sets(&descriptor_writes, &descriptor_copies);

        let dynamic_offsets = [0, 0, 0];
        handles.device.cmd_bind_descriptor_sets(
            command_buffer,
            PipelineBindPoint::COMPUTE,
            handles.pipeline_layout,
            0,
            &descriptor_sets,
            &dynamic_offsets,
        );

        let constants = std::slice::from_raw_parts(
            &PushConstants { method } as *const PushConstants as *const u8,
            std::mem::size_of::<PushConstants>()
        );
        handles.device.cmd_push_constants(
            command_buffer,
            handles.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            constants,
        );
        handles
            .device
            .cmd_dispatch(command_buffer, group.0, group.1, group.2);
        handles
            .device
            .end_command_buffer(command_buffer)
            .unwrap();

        let command_buffers = [command_buffer];
        let submits = [SubmitInfo::default().command_buffers(&command_buffers)];
        handles
            .device
            .queue_submit(handles.queue, &submits, fence)
            .unwrap();

        handles
            .device
            .wait_for_fences(&[fence], true, u64::MAX)
            .unwrap();
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> crate::Result<Self> {
        let app_name = c"candle";

        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut extension_names = vec![];
        extension_names.push(ash::ext::debug_utils::NAME.as_ptr());

        let entry = Entry::linked();
        let appinfo = ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(app_name)
            .engine_version(0)
            .api_version(ash::vk::make_api_version(0, 1, 0, 0));
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            ash::vk::InstanceCreateFlags::default()
        };
        let create_info = InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .flags(create_flags);
        let instance: Instance =
            unsafe { entry.create_instance(&create_info, None) }.expect("Instance creation error");

        let physical_devices =
            unsafe { instance.enumerate_physical_devices() }.expect("Physical device error");
        let pdevice = physical_devices.into_iter().nth(ordinal).unwrap();

        let queue_family_index =
            unsafe { instance.get_physical_device_queue_family_properties(pdevice) }
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    if info.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .expect("Couldn't find suitable device.");
        let queue_family_index = queue_family_index as u32;
        let priorities = [1.0];

        let queue_info = DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_extension_names_raw = [
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::khr::portability_subset::NAME.as_ptr(),
        ];

        let features = ash::vk::PhysicalDeviceFeatures {
            ..Default::default()
        };

        let device_create_info = DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device: ash::Device =
            unsafe { instance.create_device(pdevice, &device_create_info, None) }.unwrap();

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // create shader module
        let bytes = include_bytes!("kernels.spv");
        let bytes: &[u32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) };
        let create_info = ShaderModuleCreateInfo::default().code(bytes);
        let module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();

        // create compute pipeline
        let spec = SpecializationInfo::default();

        let group_0 = unsafe {
            let bindings = [
                DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(ShaderStageFlags::COMPUTE),
                DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                    .descriptor_count(1)
                    .stage_flags(ShaderStageFlags::COMPUTE),
            ];
            let create_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            device.create_descriptor_set_layout(&create_info, None)
        }
        .unwrap();

        let group_1 = unsafe {
            let bindings = &[DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(DescriptorType::STORAGE_BUFFER_DYNAMIC)
                .descriptor_count(1)
                .stage_flags(ShaderStageFlags::COMPUTE)];
            let create_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);
            device.create_descriptor_set_layout(&create_info, None)
        }
        .unwrap();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &ash::vk::PipelineLayoutCreateInfo::default().set_layouts(&[group_0, group_1]),
                    None,
                )
                .unwrap()
        };

        let stage = ash::vk::PipelineShaderStageCreateInfo::default()
            // According to https://raphlinus.github.io/gpu/2020/04/30/prefix-sum.html
            // "Another problem is querying the subgroup size from inside the kernel, which has a
            // surprising gotcha. Unless the VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT
            // flag is set at pipeline creation time, the gl_SubgroupSize variable is defined to have
            // the value from VkPhysicalDeviceSubgroupProperties, which in my experiment is always 32 on
            // Intel no matter the actual subgroup size. But setting that flag makes it give the value expected."
            .flags(
                ash::vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE_EXT
                    | ash::vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS_EXT,
            )
            .module(module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
            .specialization_info(&spec)
            .stage(ash::vk::ShaderStageFlags::COMPUTE);

        let pipeline = unsafe {
            device.create_compute_pipelines(
                ash::vk::PipelineCache::null(),
                &[ash::vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipeline_layout)],
                None,
            )
        }
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

        // CommandPool
        let command_pool = unsafe {
            let create_info = CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_index)
                .flags(CommandPoolCreateFlags::TRANSIENT);
            device
                .create_command_pool(&create_info, None)
                .unwrap()
        };

        // CommandBuffers
        let command_buffers = unsafe {
            let allocate_info = CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool);
            device
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        };

        // Descriptor Pool
        let descriptor_pool =  unsafe {
            let pool_sizes = [DescriptorPoolSize::default().descriptor_count(16).ty(DescriptorType::STORAGE_BUFFER_DYNAMIC)];
            let create_info = DescriptorPoolCreateInfo::default()
                .pool_sizes(&pool_sizes)
                .max_sets(16);
            device
                .create_descriptor_pool(&create_info, None)
                .unwrap()
        };

        Ok(Self {
            id: DeviceId(ordinal),
            handles: Rc::new(VkHandles {
                instance,
                pdevice,
                device,
                queue_family_index,
                queue,
                pipeline,
                pipeline_layout,
                group_0,
                group_1,
                command_pool,
                command_buffers,
                descriptor_pool
            }),
        })
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Vulkan { gpu_id: self.id.0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    unsafe fn alloc_uninit(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, _: &crate::CpuStorage) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage_owned(
        &self,
        cpu_storage: crate::CpuStorage,
    ) -> crate::Result<Self::Storage> {
        let dtype = cpu_storage.dtype();
        let (count, src) = match &cpu_storage {
            CpuStorage::U8(_) => todo!(),
            CpuStorage::U32(_) => todo!(),
            CpuStorage::I64(_) => todo!(),
            CpuStorage::BF16(_) => todo!(),
            CpuStorage::F16(_) => todo!(),
            CpuStorage::F32(items) => (items.len(), unsafe {
                std::slice::from_raw_parts(
                    items.as_ptr() as *const u8,
                    std::mem::size_of::<f32>() * items.len(),
                )
            }),
            CpuStorage::F64(_) => todo!(),
        };

        let create_info = BufferCreateInfo::default();
        let (buffer, mem, size) = create_buffer(
            &self.handles,
            src.len() as u64,
            ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        let dest = unsafe {
            self.handles
                .device
                .map_memory(mem, 0, size, ash::vk::MemoryMapFlags::empty())
        }
        .unwrap();
        let dest: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dest as *mut u8, src.len()) };

        dest.copy_from_slice(src);

        unsafe { self.handles.device.unmap_memory(mem) };

        Ok(VulkanStorage {
            buffer,
            mem,
            device: self.clone(),
            count,
            dtype,
        })
    }

    fn rand_uniform(
        &self,
        _: &crate::Shape,
        _: crate::DType,
        _: f64,
        _: f64,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(
        &self,
        _: &crate::Shape,
        _: crate::DType,
        _: f64,
        _: f64,
    ) -> crate::Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _: u64) -> crate::Result<()> {
        todo!()
    }

    fn synchronize(&self) -> crate::Result<()> {
        todo!()
    }
}

#[derive(Debug)]
pub struct VulkanStorage {
    buffer: ash::vk::Buffer,
    mem: ash::vk::DeviceMemory,
    /// a reference to the device owning this buffer
    device: VulkanDevice,
    /// The count of allocated elements in the buffer
    count: usize,
    /// The dtype is kept since buffers are untyped.
    dtype: DType,
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn dtype(&self) -> crate::DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> crate::Result<crate::CpuStorage> {
        Ok(match self.dtype {
            DType::U8 => todo!(),
            DType::U32 => todo!(),
            DType::I64 => todo!(),
            DType::BF16 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => {
                let mut dst = vec![0.0f32; self.count];
                let size = dst.len() * std::mem::size_of::<f32>();

                let src = unsafe {
                    self.device.handles.device.map_memory(
                        self.mem,
                        0,
                        size as u64,
                        ash::vk::MemoryMapFlags::empty(),
                    )
                }
                .unwrap();
                let src = unsafe { std::slice::from_raw_parts(src as *const f32, self.count) };

                dst.as_mut_slice().copy_from_slice(src);

                unsafe { self.device.handles.device.unmap_memory(self.mem) };

                CpuStorage::F32(dst)
            }
            DType::F64 => todo!(),
        })
    }

    fn affine(&self, _: &crate::Layout, _: f64, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn powf(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn elu(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn reduce_op(
        &self,
        _: crate::op::ReduceOp,
        _: &crate::Layout,
        _: &[usize],
    ) -> crate::Result<Self> {
        todo!()
    }

    fn cmp(
        &self,
        _: crate::op::CmpOp,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &crate::Layout, _: crate::DType) -> crate::Result<Self> {
        todo!()
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        other: &Self,
        l: &crate::Layout,
        r: &crate::Layout,
    ) -> crate::Result<Self> {
        let method = match B::NAME {
            "add" => KernelMethod::Add,
            "sub" => KernelMethod::Sub,
            name => todo!("{name}"),
        };

        let len = match l.shape().dims() {
            [a] => *a,
            [a, b] => a * b,
            dims => todo!("{dims:?}"),
        };

        let size = match self.dtype {
            DType::U8 => todo!(),
            DType::U32 => todo!(),
            DType::I64 => todo!(),
            DType::BF16 => todo!(),
            DType::F16 => todo!(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => todo!(),
        } * len;

        let (buffer, mem, device_size) = create_buffer(
            &self.device.handles,
            size as u64,
            ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        unsafe {
            let create_info = FenceCreateInfo::default();
            let fence = self
                .device
                .handles
                .device
                .create_fence(&create_info, None)
                .unwrap();

            dispatch_binary(&self.device.handles, 0, method, self.buffer, other.buffer, buffer, (len as u32, 1, 1), fence);

            self
                .device
                .handles
                .device
                .destroy_fence(fence, None);
        }

        Ok(Self {
            buffer,
            mem,
            device: self.device.clone(),
            count: l.dims().iter().sum(),
            dtype: self.dtype,
        })
    }

    fn where_cond(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn avg_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn max_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn gather(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn scatter_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<()> {
        todo!()
    }

    fn scatter_add_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<()> {
        todo!()
    }

    fn index_select(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> crate::Result<()> {
        todo!()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> crate::Result<()> {
        todo!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &crate::Layout) -> crate::Result<()> {
        todo!()
    }
}

fn find_memory_type(
    requirements: ash::vk::MemoryRequirements,
    mem_properties: ash::vk::PhysicalDeviceMemoryProperties,
    required_properties: ash::vk::MemoryPropertyFlags,
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

fn create_buffer(
    handles: &VkHandles,
    size: ash::vk::DeviceSize,
    usage: ash::vk::BufferUsageFlags,
    mem_properties: ash::vk::MemoryPropertyFlags,
) -> (ash::vk::Buffer, ash::vk::DeviceMemory, ash::vk::DeviceSize) {
    let buffer = {
        let buffer_info = ash::vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
        unsafe { handles.device.create_buffer(&buffer_info, None).unwrap() }
    };

    let mem_requirements = unsafe { handles.device.get_buffer_memory_requirements(buffer) };
    let memory = {
        let mem_type = find_memory_type(
            mem_requirements,
            unsafe {
                handles
                    .instance
                    .get_physical_device_memory_properties(handles.pdevice)
            },
            mem_properties,
        );

        let alloc_info = ash::vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type);
        unsafe { handles.device.allocate_memory(&alloc_info, None).unwrap() }
    };

    unsafe {
        handles
            .device
            .bind_buffer_memory(buffer, memory, 0)
            .unwrap()
    };

    (buffer, memory, mem_requirements.size)
}

macro_rules! t {
    ($b:expr) => {
        let f = $b;

        let (c_vulkan, d_vulkan) = f(Device::new_vulkan(0).unwrap());
        let (c_cpu, d_cpu) = f(Device::Cpu);

        assert_eq!(c_vulkan, c_cpu);
        assert_eq!(d_vulkan, d_cpu);
    };
}

#[test]
fn vulkan_basics_dims_1() {
    t! {|device: Device| {
        let a = Tensor::new(&[0.0f32, 1.0, 2.0], &device).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();

        let c = a.add(&b).unwrap();
        let d = a.sub(&b).unwrap();

        (c.to_vec1::<f32>().unwrap(), d.to_vec1::<f32>().unwrap())
    }};
}

#[test]
fn vulkan_basics_dims_2() {
     t! {|device: Device| {
        let a = Tensor::new(&[0.0f32, 1.0, 2.0, 3.0], &device).unwrap().reshape((2, 2)).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device).unwrap().reshape((2, 2)).unwrap();

        let c = a.add(&b).unwrap();
        let d = a.sub(&b).unwrap();

        (c.to_vec2::<f32>().unwrap(), d.to_vec2::<f32>().unwrap())
    }};
}
