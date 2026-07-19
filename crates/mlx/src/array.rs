//! A safe wrapper around `mlx_array`.

use std::ffi::CStr;
use std::fmt;

use mlx_sys as sys;

use crate::dtype::ArrayElement;

/// An N-dimensional MLX array.
///
/// Owns the underlying `mlx_array` handle and frees it on drop.
pub struct Array {
    handle: sys::mlx_array,
}

impl Array {
    /// Creates an `Array` from a raw handle, taking ownership of it.
    ///
    /// # Safety
    /// `handle` must be a valid `mlx_array` that is not freed elsewhere.
    pub(crate) unsafe fn from_raw(handle: sys::mlx_array) -> Self {
        Self { handle }
    }

    /// Returns the raw handle. The `Array` retains ownership.
    // Scaffolding for ops that need the underlying handle; unused for now.
    #[allow(dead_code)]
    pub(crate) fn as_raw(&self) -> sys::mlx_array {
        self.handle
    }

    /// Builds an array from a slice of values with the given shape.
    ///
    /// The MLX dtype is chosen from the element type `T` at compile time (e.g.
    /// `&[f32]` produces a `float32` array, `&[i32]` an `int32` array).
    ///
    /// # Panics
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        let expected: i64 = shape.iter().map(|&d| d as i64).product();
        assert_eq!(
            data.len() as i64,
            expected,
            "data length {} does not match shape product {expected}",
            data.len()
        );
        // SAFETY: pointers/len are valid for the duration of the call; mlx
        // copies the data into its own buffer.
        let handle = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape.as_ptr(),
                shape.len() as i32,
                T::DTYPE,
            )
        };
        unsafe { Self::from_raw(handle) }
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        // SAFETY: handle is valid for the lifetime of `self`.
        unsafe { sys::mlx_array_size(self.handle) }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.handle) }
    }

    /// Shape of the array.
    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        // SAFETY: mlx guarantees the returned pointer is valid for `ndim` ints.
        let ptr = unsafe { sys::mlx_array_shape(self.handle) };
        (0..ndim).map(|i| unsafe { *ptr.add(i) }).collect()
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // SAFETY: a freshly-created string handle is written by tostring.
        let mut s = unsafe { sys::mlx_string_new() };
        unsafe { sys::mlx_array_tostring(&mut s, self.handle) };
        let cstr = unsafe { CStr::from_ptr(sys::mlx_string_data(s)) };
        let out = write!(f, "{}", cstr.to_string_lossy());
        unsafe { sys::mlx_string_free(s) };
        out
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe {
            sys::mlx_array_free(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_slice_reports_shape_size_ndim() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(a.size(), 6);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.shape(), vec![2, 3]);
    }

    #[test]
    fn element_type_selects_dtype() {
        // Both element types build valid arrays; the dtype is carried by `T`.
        let floats = Array::from_slice(&[1.0f32, 2.0], &[2]);
        assert_eq!(floats.shape(), vec![2]);
        let ints = Array::from_slice(&[1i32, 2, 3], &[3]);
        assert_eq!(ints.shape(), vec![3]);
    }

    #[test]
    fn scalar_array_is_zero_dim() {
        let a = Array::from_slice(&[42.0f32], &[]);
        assert_eq!(a.ndim(), 0);
        assert_eq!(a.size(), 1);
        assert!(a.shape().is_empty());
    }

    #[test]
    #[should_panic(expected = "does not match shape product")]
    fn mismatched_len_and_shape_panics() {
        // 5 elements cannot fill a 2x3 (=6) array.
        let _ = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
    }

    #[test]
    fn debug_renders_array_contents() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let s = format!("{a:?}");
        assert!(s.contains("array"), "unexpected debug output: {s}");
    }
}
