//! Mapping between Rust primitive types and MLX data types.

use mlxcore_sys as sys;

mod sealed {
    pub trait Sealed {}
}

/// A Rust type that has a corresponding MLX [`mlx_dtype`](sys::mlx_dtype).
///
/// This trait is sealed: it can only be implemented for the primitive types
/// MLX supports, so `T::DTYPE` is always a valid dtype and the accessors below
/// always match it.
pub trait ArrayElement: sealed::Sealed + Copy + Default {
    /// The MLX dtype corresponding to this Rust type.
    const DTYPE: sys::mlx_dtype;

    /// Reads the value of a scalar (single-element) array as this type.
    ///
    /// # Safety
    /// `arr` must be a valid, already-evaluated scalar `mlx_array`.
    unsafe fn read_item(arr: sys::mlx_array) -> Self;

    /// Returns a pointer to this array's contiguous data of this type.
    ///
    /// # Safety
    /// `arr` must be a valid, already-evaluated `mlx_array` whose dtype is
    /// `Self::DTYPE`. The pointer is valid until `arr` is mutated or freed.
    unsafe fn data_ptr(arr: sys::mlx_array) -> *const Self;
}

macro_rules! impl_array_element {
    ($($rust:ty => $dtype:expr, $item:path, $data:path),* $(,)?) => {
        $(
            impl sealed::Sealed for $rust {}
            impl ArrayElement for $rust {
                const DTYPE: sys::mlx_dtype = $dtype;

                unsafe fn read_item(arr: sys::mlx_array) -> Self {
                    let mut out = <$rust>::default();
                    // SAFETY: caller guarantees `arr` is a valid scalar array.
                    unsafe { $item(&mut out, arr); }
                    out
                }

                unsafe fn data_ptr(arr: sys::mlx_array) -> *const Self {
                    // SAFETY: caller guarantees `arr` is valid with dtype DTYPE.
                    unsafe { $data(arr) }
                }
            }
        )*
    };
}

// Only the MLX dtypes with a native Rust primitive are mapped here. Types
// without a stable Rust equivalent (float16, bfloat16, complex64) are left for
// dedicated newtypes later.
impl_array_element! {
    bool => sys::MLX_BOOL,    sys::mlx_array_item_bool,    sys::mlx_array_data_bool,
    u8   => sys::MLX_UINT8,   sys::mlx_array_item_uint8,   sys::mlx_array_data_uint8,
    u16  => sys::MLX_UINT16,  sys::mlx_array_item_uint16,  sys::mlx_array_data_uint16,
    u32  => sys::MLX_UINT32,  sys::mlx_array_item_uint32,  sys::mlx_array_data_uint32,
    u64  => sys::MLX_UINT64,  sys::mlx_array_item_uint64,  sys::mlx_array_data_uint64,
    i8   => sys::MLX_INT8,    sys::mlx_array_item_int8,    sys::mlx_array_data_int8,
    i16  => sys::MLX_INT16,   sys::mlx_array_item_int16,   sys::mlx_array_data_int16,
    i32  => sys::MLX_INT32,   sys::mlx_array_item_int32,   sys::mlx_array_data_int32,
    i64  => sys::MLX_INT64,   sys::mlx_array_item_int64,   sys::mlx_array_data_int64,
    f32  => sys::MLX_FLOAT32, sys::mlx_array_item_float32, sys::mlx_array_data_float32,
    f64  => sys::MLX_FLOAT64, sys::mlx_array_item_float64, sys::mlx_array_data_float64,
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
