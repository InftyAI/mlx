//! Mapping between Rust primitive types and MLX data types.

use mlx_sys as sys;

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that has a corresponding MLX [`mlx_dtype`](sys::mlx_dtype).
///
/// This trait is sealed: it can only be implemented for the primitive types
/// MLX supports, so `T::DTYPE` is always a valid dtype.
pub trait ArrayElement: sealed::Sealed + Copy {
    /// The MLX dtype corresponding to this Rust type.
    const DTYPE: sys::mlx_dtype;
}

macro_rules! impl_array_element {
    ($($rust:ty => $dtype:expr),* $(,)?) => {
        $(
            impl sealed::Sealed for $rust {}
            impl ArrayElement for $rust {
                const DTYPE: sys::mlx_dtype = $dtype;
            }
        )*
    };
}

// Only the MLX dtypes with a native Rust primitive are mapped here. Types
// without a stable Rust equivalent (float16, bfloat16, complex64) are left for
// dedicated newtypes later.
impl_array_element! {
    bool => sys::MLX_BOOL,
    u8   => sys::MLX_UINT8,
    u16  => sys::MLX_UINT16,
    u32  => sys::MLX_UINT32,
    u64  => sys::MLX_UINT64,
    i8   => sys::MLX_INT8,
    i16  => sys::MLX_INT16,
    i32  => sys::MLX_INT32,
    i64  => sys::MLX_INT64,
    f32  => sys::MLX_FLOAT32,
    f64  => sys::MLX_FLOAT64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_rust_types_to_expected_dtypes() {
        assert_eq!(<f32 as ArrayElement>::DTYPE, sys::MLX_FLOAT32);
        assert_eq!(<f64 as ArrayElement>::DTYPE, sys::MLX_FLOAT64);
        assert_eq!(<i32 as ArrayElement>::DTYPE, sys::MLX_INT32);
        assert_eq!(<u8 as ArrayElement>::DTYPE, sys::MLX_UINT8);
        assert_eq!(<bool as ArrayElement>::DTYPE, sys::MLX_BOOL);
    }
}
