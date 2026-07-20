//! A safe wrapper around `mlx_stream`.
//!
//! MLX schedules every operation on a [`Stream`], which is bound to a device
//! (CPU or GPU). Most ops take a stream argument. MLX's own default is the GPU
//! when a GPU backend is available (as on Apple Silicon), else the CPU.

use mlxr_sys as sys;

/// An execution stream bound to a device.
pub struct Stream {
    handle: sys::mlx_stream,
}

impl Stream {
    /// The default stream on the GPU device.
    pub fn gpu() -> Self {
        crate::error::install();
        // SAFETY: constructor returns an owned handle.
        let handle = unsafe { sys::mlx_default_gpu_stream_new() };
        Self { handle }
    }

    /// The default stream on the CPU device.
    pub fn cpu() -> Self {
        crate::error::install();
        // SAFETY: constructor returns an owned handle.
        let handle = unsafe { sys::mlx_default_cpu_stream_new() };
        Self { handle }
    }

    /// Returns the raw handle. The `Stream` retains ownership.
    pub(crate) fn as_raw(&self) -> sys::mlx_stream {
        self.handle
    }

    /// Makes this the process-wide default stream (and its device the default
    /// device).
    ///
    /// Operations that don't take an explicit stream — the arithmetic operators
    /// (`&a + &b`) and anything built on [`Stream::default`] — resolve the
    /// current default. Call this once at startup to steer them onto a chosen
    /// device.
    pub fn set_as_default(&self) {
        crate::error::install();
        // SAFETY: `handle` is a valid stream owned by `self`; mlx copies out of
        // the pointers we pass and out of the handle.
        unsafe {
            let mut dev = sys::mlx_device_new();
            sys::mlx_stream_get_device(&mut dev, self.handle);
            sys::mlx_set_default_device(dev);
            sys::mlx_device_free(dev);
            sys::mlx_set_default_stream(self.handle);
        }
    }
}

impl Default for Stream {
    /// MLX's *current* default stream — follows [`Stream::set_as_default`].
    ///
    /// The initial default is chosen by MLX, not this crate: it is the GPU when
    /// a GPU backend (Metal) is available, otherwise the CPU. This only reads
    /// that default; use [`Stream::set_as_default`] to change it.
    fn default() -> Self {
        crate::error::install();
        // SAFETY: out-params are valid; mlx writes owned handles into them.
        let handle = unsafe {
            let mut dev = sys::mlx_device_new();
            sys::mlx_get_default_device(&mut dev);
            let mut stream = sys::mlx_stream_new();
            sys::mlx_get_default_stream(&mut stream, dev);
            sys::mlx_device_free(dev);
            stream
        };
        Self { handle }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe {
            sys::mlx_stream_free(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_default_streams() {
        // Just exercises construction + drop on both devices.
        let _gpu = Stream::gpu();
        let _cpu = Stream::cpu();
        let _default = Stream::default();
    }
}
