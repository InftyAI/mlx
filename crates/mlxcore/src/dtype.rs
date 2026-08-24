//! Mapping between Rust primitive types and MLX data types.

use std::fmt;

use mlxcore_sys as sys;

mod sealed {
    pub trait Sealed {}
}

/// The element type of an [`Array`](crate::Array).
///
/// Every MLX dtype has a variant here, including the three with no Rust
/// primitive to match them ([`Float16`](Dtype::Float16),
/// [`Bfloat16`](Dtype::Bfloat16), [`Complex64`](Dtype::Complex64)) — an array
/// can carry those even though we cannot yet read or construct them, so
/// [`Array::dtype`](crate::Array::dtype) must be able to name them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Complex64,
}

impl Dtype {
    /// The raw mlx-c enum value, for passing across the FFI boundary.
    pub(crate) fn as_raw(self) -> sys::mlx_dtype {
        match self {
            Dtype::Bool => sys::MLX_BOOL,
            Dtype::Uint8 => sys::MLX_UINT8,
            Dtype::Uint16 => sys::MLX_UINT16,
            Dtype::Uint32 => sys::MLX_UINT32,
            Dtype::Uint64 => sys::MLX_UINT64,
            Dtype::Int8 => sys::MLX_INT8,
            Dtype::Int16 => sys::MLX_INT16,
            Dtype::Int32 => sys::MLX_INT32,
            Dtype::Int64 => sys::MLX_INT64,
            Dtype::Float16 => sys::MLX_FLOAT16,
            Dtype::Float32 => sys::MLX_FLOAT32,
            Dtype::Float64 => sys::MLX_FLOAT64,
            Dtype::Bfloat16 => sys::MLX_BFLOAT16,
            Dtype::Complex64 => sys::MLX_COMPLEX64,
        }
    }

    /// Converts a raw mlx-c enum value.
    ///
    /// # Panics
    /// Panics on a value MLX did not define when these bindings were generated,
    /// which would mean the linked MLX added a dtype we cannot name.
    pub(crate) fn from_raw(raw: sys::mlx_dtype) -> Self {
        match raw {
            sys::MLX_BOOL => Dtype::Bool,
            sys::MLX_UINT8 => Dtype::Uint8,
            sys::MLX_UINT16 => Dtype::Uint16,
            sys::MLX_UINT32 => Dtype::Uint32,
            sys::MLX_UINT64 => Dtype::Uint64,
            sys::MLX_INT8 => Dtype::Int8,
            sys::MLX_INT16 => Dtype::Int16,
            sys::MLX_INT32 => Dtype::Int32,
            sys::MLX_INT64 => Dtype::Int64,
            sys::MLX_FLOAT16 => Dtype::Float16,
            sys::MLX_FLOAT32 => Dtype::Float32,
            sys::MLX_FLOAT64 => Dtype::Float64,
            sys::MLX_BFLOAT16 => Dtype::Bfloat16,
            sys::MLX_COMPLEX64 => Dtype::Complex64,
            other => panic!("unknown mlx dtype: {other}"),
        }
    }

    /// The size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
        }
    }
}

/// Formats as MLX's own name for the dtype, e.g. `float32`, not `Float32`.
impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dtype::Bool => "bool",
            Dtype::Uint8 => "uint8",
            Dtype::Uint16 => "uint16",
            Dtype::Uint32 => "uint32",
            Dtype::Uint64 => "uint64",
            Dtype::Int8 => "int8",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::Bfloat16 => "bfloat16",
            Dtype::Complex64 => "complex64",
        };
        f.write_str(name)
    }
}

/// A Rust type that has a corresponding MLX [`Dtype`].
///
/// This trait is sealed: it can only be implemented for the primitive types
/// MLX supports, so `T::DTYPE` is always a valid dtype and the accessors below
/// always match it.
pub trait ArrayElement: sealed::Sealed + Copy + Default {
    /// The MLX dtype corresponding to this Rust type.
    const DTYPE: Dtype;

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
                const DTYPE: Dtype = $dtype;

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
    bool => Dtype::Bool,    sys::mlx_array_item_bool,    sys::mlx_array_data_bool,
    u8   => Dtype::Uint8,   sys::mlx_array_item_uint8,   sys::mlx_array_data_uint8,
    u16  => Dtype::Uint16,  sys::mlx_array_item_uint16,  sys::mlx_array_data_uint16,
    u32  => Dtype::Uint32,  sys::mlx_array_item_uint32,  sys::mlx_array_data_uint32,
    u64  => Dtype::Uint64,  sys::mlx_array_item_uint64,  sys::mlx_array_data_uint64,
    i8   => Dtype::Int8,    sys::mlx_array_item_int8,    sys::mlx_array_data_int8,
    i16  => Dtype::Int16,   sys::mlx_array_item_int16,   sys::mlx_array_data_int16,
    i32  => Dtype::Int32,   sys::mlx_array_item_int32,   sys::mlx_array_data_int32,
    i64  => Dtype::Int64,   sys::mlx_array_item_int64,   sys::mlx_array_data_int64,
    f32  => Dtype::Float32, sys::mlx_array_item_float32, sys::mlx_array_data_float32,
    f64  => Dtype::Float64, sys::mlx_array_item_float64, sys::mlx_array_data_float64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_rust_types_to_expected_dtypes() {
        assert_eq!(<f32 as ArrayElement>::DTYPE, Dtype::Float32);
        assert_eq!(<f64 as ArrayElement>::DTYPE, Dtype::Float64);
        assert_eq!(<i32 as ArrayElement>::DTYPE, Dtype::Int32);
        assert_eq!(<u8 as ArrayElement>::DTYPE, Dtype::Uint8);
        assert_eq!(<bool as ArrayElement>::DTYPE, Dtype::Bool);
    }

    #[test]
    fn dtypes_round_trip_through_the_raw_enum() {
        const ALL: [Dtype; 14] = [
            Dtype::Bool,
            Dtype::Uint8,
            Dtype::Uint16,
            Dtype::Uint32,
            Dtype::Uint64,
            Dtype::Int8,
            Dtype::Int16,
            Dtype::Int32,
            Dtype::Int64,
            Dtype::Float16,
            Dtype::Float32,
            Dtype::Float64,
            Dtype::Bfloat16,
            Dtype::Complex64,
        ];
        for dtype in ALL {
            assert_eq!(Dtype::from_raw(dtype.as_raw()), dtype);
        }
    }

    #[test]
    fn raw_values_match_the_c_enum() {
        assert_eq!(Dtype::Float32.as_raw(), sys::MLX_FLOAT32);
        assert_eq!(Dtype::Bfloat16.as_raw(), sys::MLX_BFLOAT16);
        assert_eq!(Dtype::from_raw(sys::MLX_COMPLEX64), Dtype::Complex64);
    }

    #[test]
    #[should_panic(expected = "unknown mlx dtype")]
    fn unknown_raw_dtype_panics() {
        let _ = Dtype::from_raw(sys::MLX_COMPLEX64 + 1);
    }

    #[test]
    fn displays_mlx_names_and_element_sizes() {
        assert_eq!(Dtype::Float32.to_string(), "float32");
        assert_eq!(Dtype::Bfloat16.to_string(), "bfloat16");
        assert_eq!(Dtype::Bool.size(), 1);
        assert_eq!(Dtype::Float32.size(), 4);
        assert_eq!(Dtype::Complex64.size(), 8);
    }
}
