use std::{borrow::Cow, sync::Arc};

use half::bf16;
use wgpu::{util::DeviceExt, BindGroup, BindGroupLayout, Buffer, ComputePipeline};

use crate::{op::BinaryOpT, CpuStorage, DType, DeviceLocation, Layout, Shape};

#[derive(Debug)]
struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

#[derive(Debug, Clone)]
pub struct WgpuDevice {
    inner: Arc<Device>,
}

fn compile_shaders(device: &wgpu::Device) -> (ComputePipeline, BindGroupLayout) {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(
            r#"
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience

@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    v_indices[global_id.x] = 0u;
}
"#
            .into(),
        )),
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

    (compute_pipeline, bind_group_layout)
}

fn create_and_bind_buffer(
    device: &wgpu::Device,
    bind_group_layout: &BindGroupLayout,
    shape: &Shape,
    dtype: &DType,
) -> (Buffer, Buffer, BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (shape.elem_count() * dtype.size_in_bytes()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (shape.elem_count() * dtype.size_in_bytes()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Tensor"),
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    (buffer, storage_buffer, bind_group)
}

impl WgpuDevice {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let (pipeline, bind_group_layout) = compile_shaders(&device);
        Self {
            inner: Arc::new(Device {
                device,
                queue,
                pipeline,
                bind_group_layout,
            }),
        }
    }

    pub(crate) fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        _lo: f64,
        _up: f64,
    ) -> crate::Result<WgpuStorage> {
        let (buffer, storage_buffer, bind_group) = create_and_bind_buffer(
            &self.inner.device,
            &self.inner.bind_group_layout,
            &shape,
            &dtype,
        );

        Ok(WgpuStorage {
            device: self.clone(),
            shape: shape.clone(),
            dtype,
            buffer,
            storage_buffer,
            bind_group,
        })
    }

    pub(crate) fn location(&self) -> DeviceLocation {
        DeviceLocation::Wgpu
    }
}

#[derive(Debug)]
pub struct WgpuStorage {
    device: WgpuDevice,
    shape: Shape,
    dtype: DType,
    buffer: Buffer,
    storage_buffer: Buffer,
    bind_group: BindGroup,
}

trait Zero {
    fn zero() -> Self;
}

macro_rules! impl_zero {
    ($($t:path,)*; $zero:expr) => {
        $(
            impl Zero for $t {
                fn zero() -> Self {
                    $zero
                }
            }
        )*
    }
}

impl_zero! {u8,u16,u32,u64,; 0}
impl_zero! {i8,i16,i32,i64,; 0}
impl_zero! {f32,f64,; 0.0}
impl_zero! {half::f16,; half::f16::from_bits(0)}
impl_zero! {half::bf16,; half::bf16::from_bits(0)}

impl WgpuStorage {
    pub(crate) fn device(&self) -> &WgpuDevice {
        &self.device
    }

    pub(crate) fn dtype(&self) -> crate::DType {
        self.dtype
    }

    pub(crate) fn to_cpu<T: Zero + Clone>(&self) -> crate::Result<Vec<T>> {
        let w = Arc::new(std::sync::Barrier::new(2));

        std::thread::spawn({
            let device = self.device.clone();
            move || {
                device
                    .inner
                    .device
                    .poll(wgpu::Maintain::wait())
                    .panic_on_timeout();
            }
        });

        let buffer_slice = self.buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, {
            let w = w.clone();
            move |v| {
                w.wait();
            }
        });

        w.wait();

        let data = buffer_slice.get_mapped_range();
        let result = unsafe {
            std::slice::from_raw_parts::<T>(data.as_ptr() as *const T, self.shape.elem_count())
        };
        drop(data);

        self.buffer.unmap();

        Ok(result.to_vec())
    }

    pub(crate) fn to_cpu_storage(&self) -> crate::Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => Ok(CpuStorage::F16(self.to_cpu()?)),
            DType::BF16 => Ok(CpuStorage::BF16(self.to_cpu()?)),
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
        }
    }

    pub(crate) fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> crate::Result<Self> {
        let (buffer, storage_buffer, bind_group) = create_and_bind_buffer(
            &self.device.inner.device,
            &self.device.inner.bind_group_layout,
            &self.shape,
            &self.dtype,
        );

        let mut encoder = self
            .device
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(B::NAME),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.device.inner.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.insert_debug_marker(B::NAME);
                pass.dispatch_workgroups(1, 1, 1);
            }

            self.device.inner.queue.submit(Some(encoder.finish()));
        }

        Ok(WgpuStorage {
            device: self.device.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            buffer,
            storage_buffer,
            bind_group,
        })
    }
}
