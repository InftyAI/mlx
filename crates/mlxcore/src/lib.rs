//! Safe, idiomatic Rust bindings for Apple's [MLX](https://github.com/ml-explore/mlx)
//! array framework, built on top of the [`mlxcore-sys`] FFI layer.
//!
//! This crate is Apple Silicon (macOS) only.
//!
//! # Suffix float literals with `f32`
//!
//! Unsuffixed float literals are `f64` in Rust, and Apple GPUs have no float64
//! support. Since the default stream is the GPU, `&[1.0, 2.0]` builds an array
//! that fails on every operation. Write `&[1.0f32, 2.0]`, and `&a * 2.0f32` for
//! scalar operands (MLX promotes to the wider dtype, so an unsuffixed literal
//! widens the result and the operator panics).
//!
//! float64 still works on an explicit [`Stream::cpu`] stream.

mod array;
mod dtype;
mod error;
mod ffi;
mod stream;

pub mod random;

pub use array::Array;
pub use dtype::ArrayElement;
pub use error::{Error, Result};
pub use stream::Stream;

/// Returns the version string of the underlying MLX library.
pub fn version() -> String {
    use std::ffi::CStr;
    // SAFETY: standard mlx-c string-handle dance; all handles are freed.
    unsafe {
        let mut s = mlxcore_sys::mlx_string_new();
        mlxcore_sys::mlx_version(&mut s);
        let v = CStr::from_ptr(mlxcore_sys::mlx_string_data(s))
            .to_string_lossy()
            .into_owned();
        mlxcore_sys::mlx_string_free(s);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_version() {
        assert!(!version().is_empty());
    }
}
