//! An owned `mlx_vector_array`, the container mlx-c uses for multi-array
//! arguments and results.

use mlxcore_sys as sys;

use crate::array::Array;
use crate::error::{self, Result};

/// A vector of arrays, owning its `mlx_vector_array` handle.
///
/// A handful of mlx-c entry points speak in vectors rather than single arrays:
/// [`Array::split`] returns one, [`Array::stack`] and [`Array::concatenate`]
/// take one. This is a thin RAII holder so those call sites cannot leak the
/// handle on an early return.
///
/// Internal on purpose — the public API takes `&[&Array]` and returns
/// `Vec<Array>`, so this type never appears in a signature.
pub(crate) struct VectorArray {
    handle: sys::mlx_vector_array,
}

impl VectorArray {
    /// An empty vector.
    pub(crate) fn new() -> Self {
        error::install();
        // SAFETY: allocates a fresh empty vector, whose ownership moves into `self`.
        Self {
            handle: unsafe { sys::mlx_vector_array_new() },
        }
    }

    /// A vector referring to the same arrays as `arrays`.
    ///
    /// mlx-c pushes a *copy* of each `mlx_array` handle, and the copy shares the
    /// underlying buffer rather than taking it over. The borrowed `Array`s stay
    /// the owners, so this vector can be dropped on its own.
    pub(crate) fn from_arrays(arrays: &[&Array]) -> Self {
        let vec = Self::new();
        for array in arrays {
            // SAFETY: both handles are valid; `append_value` copies the handle in.
            unsafe { sys::mlx_vector_array_append_value(vec.handle, array.as_raw()) };
        }
        vec
    }

    /// Returns the raw handle. The `VectorArray` retains ownership.
    pub(crate) fn as_raw(&self) -> sys::mlx_vector_array {
        self.handle
    }

    /// A pointer for the `res` out-parameter of an mlx-c call.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut sys::mlx_vector_array {
        &mut self.handle
    }

    /// Number of arrays held.
    pub(crate) fn len(&self) -> usize {
        // SAFETY: handle is valid for the lifetime of `self`.
        unsafe { sys::mlx_vector_array_size(self.handle) }
    }

    /// Copies every element out as an owned [`Array`].
    pub(crate) fn to_arrays(&self) -> Result<Vec<Array>> {
        (0..self.len())
            .map(|i| {
                let mut out = unsafe { sys::mlx_array_new() };
                // SAFETY: `i < len`; `get` writes into `out` a handle sharing the
                // element's buffer, which `from_op` then owns.
                let status = unsafe { sys::mlx_vector_array_get(&mut out, self.handle, i) };
                Array::from_op(out, status)
            })
            .collect()
    }
}

impl Drop for VectorArray {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe { sys::mlx_vector_array_free(self.handle) };
    }
}
