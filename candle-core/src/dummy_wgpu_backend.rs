macro_rules! fail {
    () => {
        unimplemented!("wgpu support has not been enabled, add `wgpu` feature to enable it.")
    };
}

#[derive(Debug, Clone)]
pub struct WgpuDevice;

#[derive(Debug)]
pub struct WgpuStorage;

impl WgpuStorage {
    pub(crate) fn dtype(&self) -> crate::DType {
        fail!()
    }
}

impl WgpuDevice {
    pub(crate) fn rand_uniform(
        &self,
        _shape: &crate::Shape,
        _dtype: crate::DType,
        _lo: f64,
        _up: f64,
    ) -> crate::Result<WgpuStorage> {
        Ok(WgpuStorage)
    }
}
