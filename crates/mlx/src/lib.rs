//! Safe, idiomatic Rust bindings for Apple's [MLX](https://github.com/ml-explore/mlx)
//! array framework, built on top of the [`mlx-sys`] FFI layer.
//!
//! This crate is Apple Silicon (macOS) only.

mod array;
mod dtype;
mod error;
mod stream;

pub use array::Array;
pub use dtype::ArrayElement;
pub use error::{Error, Result};
pub use stream::Stream;

/// Returns the version string of the underlying MLX library.
pub fn version() -> String {
    use std::ffi::CStr;
    // SAFETY: standard mlx-c string-handle dance; all handles are freed.
    unsafe {
        let mut s = mlx_sys::mlx_string_new();
        mlx_sys::mlx_version(&mut s);
        let v = CStr::from_ptr(mlx_sys::mlx_string_data(s))
            .to_string_lossy()
            .into_owned();
        mlx_sys::mlx_string_free(s);
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
