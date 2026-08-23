//! A safe wrapper around `mlx_array`.

use std::ffi::CStr;
use std::fmt;

use mlxcore_sys as sys;

use crate::dtype::ArrayElement;
use crate::error::{self, Result};
use crate::stream::Stream;

/// An N-dimensional MLX array.
///
/// Owns the underlying `mlx_array` handle and frees it on drop.
pub struct Array {
    handle: sys::mlx_array,
}

/// Returns a pointer to `slice`'s data, or null when it is empty.
///
/// For an empty slice `as_ptr()` is non-null but dangling. The mlx-c functions
/// we call take a `(ptr, len)` pair and build a container from it, so they never
/// dereference the pointer when `len == 0` — but passing an explicit null keeps
/// a dangling pointer from crossing the FFI boundary at all.
fn as_ffi_ptr<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr()
    }
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
        error::install();
        // Both empty cases are reachable here: an empty `shape` for a 0-d
        // scalar, empty `data` for a 0-element array.
        let data_ptr = as_ffi_ptr(data) as *const _;
        let shape_ptr = as_ffi_ptr(shape);
        // SAFETY: pointers/len describe valid slices (or null/0) for the
        // duration of the call; mlx copies the data into its own buffer.
        let handle =
            unsafe { sys::mlx_array_new_data(data_ptr, shape_ptr, shape.len() as i32, T::DTYPE) };
        unsafe { Self::from_raw(handle) }
    }

    /// Builds a 0-dimensional array holding a single value.
    ///
    /// The dtype follows the value's type, so `Array::from_scalar(2.0f32)` is a
    /// `float32` scalar. Handy as the operand of a broadcasting op.
    pub fn from_scalar<T: ArrayElement>(value: T) -> Self {
        Self::from_slice(&[value], &[])
    }

    /// An array of `shape` filled with zeros, with the dtype chosen by `T`.
    ///
    /// Call as `Array::zeros::<f32>(&[2, 3], &stream)`.
    pub fn zeros<T: ArrayElement>(shape: &[i32], stream: &Stream) -> Result<Array> {
        Self::fill_op(shape, T::DTYPE, stream, sys::mlx_zeros)
    }

    /// An array of `shape` filled with ones, with the dtype chosen by `T`.
    ///
    /// Call as `Array::ones::<f32>(&[2, 3], &stream)`.
    pub fn ones<T: ArrayElement>(shape: &[i32], stream: &Stream) -> Result<Array> {
        Self::fill_op(shape, T::DTYPE, stream, sys::mlx_ones)
    }

    /// An array of `shape` filled with `value`, with the dtype chosen by `T`.
    pub fn full<T: ArrayElement>(shape: &[i32], value: T, stream: &Stream) -> Result<Array> {
        error::install();
        // Held in a local so it outlives the call below.
        let vals = Self::from_scalar(value);
        let shape_ptr = as_ffi_ptr(shape);
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: `shape_ptr`/`shape.len()` describe a valid slice (or null/0);
        // all handles are valid; the result is written into `out`.
        let status = unsafe {
            sys::mlx_full(
                &mut out,
                shape_ptr,
                shape.len(),
                vals.as_raw(),
                T::DTYPE,
                stream.as_raw(),
            )
        };
        Self::from_op(out, status)
    }

    /// Values from `start` (inclusive) to `stop` (exclusive), stepping by `step`.
    ///
    /// The bounds are `f64` because MLX's are; they are cast to the dtype chosen
    /// by `T`, as in `Array::arange::<i32>(0.0, 5.0, 1.0, &stream)`.
    pub fn arange<T: ArrayElement>(
        start: f64,
        stop: f64,
        step: f64,
        stream: &Stream,
    ) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: stream is valid; the result is written into `out`.
        let status =
            unsafe { sys::mlx_arange(&mut out, start, stop, step, T::DTYPE, stream.as_raw()) };
        Self::from_op(out, status)
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

    /// Strides of the array, in elements (not bytes), one per dimension.
    pub fn strides(&self) -> Vec<usize> {
        let ndim = self.ndim();
        // SAFETY: mlx guarantees the returned pointer is valid for `ndim`
        // `size_t`s.
        let ptr = unsafe { sys::mlx_array_strides(self.handle) };
        (0..ndim).map(|i| unsafe { *ptr.add(i) }).collect()
    }

    /// Whether the array is laid out row-major (C-order) contiguously.
    ///
    /// Computed from the public shape + strides, so a raw read of the storage
    /// buffer yields elements in logical row-major order iff this is true.
    fn is_row_contiguous(&self) -> bool {
        let shape = self.shape();
        let strides = self.strides();
        // Expected row-major stride for axis i is the product of all later
        // dimensions. Walk from the last axis, tracking that running product.
        // Size-1 axes impose no constraint (any stride works), so skip them.
        let mut expected: usize = 1;
        for i in (0..shape.len()).rev() {
            let dim = shape[i] as usize;
            if dim != 1 && strides[i] != expected {
                return false;
            }
            expected *= dim;
        }
        true
    }

    /// Forces evaluation of this array.
    ///
    /// MLX is lazy: ops build a graph and only compute when the result is
    /// needed. `eval` materializes the values now.
    pub fn eval(&self) {
        error::install();
        // SAFETY: handle is valid for the lifetime of `self`.
        unsafe {
            sys::mlx_array_eval(self.handle);
        }
    }

    /// Reads the value of a scalar (single-element) array.
    ///
    /// The element type `T` selects the accessor at compile time, e.g.
    /// `a.item::<f32>()`. Evaluates the array first. MLX casts the stored dtype
    /// to `T`.
    ///
    /// # Panics
    /// Panics if the array is not a single-element array (`size() != 1`).
    pub fn item<T: ArrayElement>(&self) -> T {
        self.eval();
        let size = self.size();
        assert_eq!(
            size, 1,
            "item() requires a single-element array, but this array has {size} elements"
        );
        // SAFETY: `read_item` requires an evaluated, single-element array — both
        // ensured above — and picks the accessor matching `T`.
        unsafe { T::read_item(self.handle) }
    }

    /// Copies the array's contents into a `Vec<T>`, row-major.
    ///
    /// The element type `T` selects the accessor at compile time, e.g.
    /// `a.to_vec::<f32>()`. Evaluates the array first.
    ///
    /// Row-contiguous arrays are read directly from their storage buffer.
    /// Non-contiguous ones (e.g. from [`transpose`](Self::transpose) or
    /// [`broadcast_to`](Self::broadcast_to)) are first materialized into a
    /// row-contiguous copy, so the result always reflects the logical
    /// (row-major) element order rather than the raw storage buffer.
    ///
    /// # Panics
    /// Panics if `T::DTYPE` does not match the array's dtype, or if
    /// making the array contiguous fails.
    pub fn to_vec<T: ArrayElement>(&self) -> Vec<T> {
        self.eval();
        // Fast path: already row-major, so the storage buffer is already in
        // logical order — read it directly, no copy.
        if self.is_row_contiguous() {
            return self.read_buffer::<T>();
        }
        // Slow path: strided views (transpose) and stride-0 views (broadcast)
        // don't lay their logical elements out contiguously, so reading the raw
        // pointer would return storage order (or read past real data). Only
        // these pay for a materialized copy.
        //
        // Run it on the CPU stream: this is a host-side data-marshalling step
        // (we're about to read the buffer from Rust), and it keeps `to_vec` off
        // the GPU stream so concurrent callers don't contend on Metal.
        let contiguous = self.contiguous(&Stream::cpu()).unwrap_or_else(|e| {
            panic!("to_vec: failed to make array contiguous: {e}");
        });
        contiguous.eval();
        contiguous.read_buffer::<T>()
    }

    /// Bulk-copies a **row-contiguous** array's storage buffer into a `Vec<T>`.
    ///
    /// # Panics
    /// Panics if `T::DTYPE` does not match the array's dtype. Assumes the array
    /// is already evaluated and row-contiguous.
    fn read_buffer<T: ArrayElement>(&self) -> Vec<T> {
        // SAFETY: mlx_array_dtype reads a valid handle.
        let dtype = unsafe { sys::mlx_array_dtype(self.handle) };
        assert_eq!(
            dtype,
            T::DTYPE,
            "array dtype does not match requested element type"
        );
        let len = self.size();
        // `from_raw_parts` requires a non-null, aligned pointer even for a
        // zero-length slice, but mlx may return null for an empty array.
        if len == 0 {
            return Vec::new();
        }
        // SAFETY: dtype matches `T` (checked above) and the array is dense, so
        // mlx guarantees `len` contiguous, aligned `T` at `ptr`, valid until the
        // array is mutated or freed. We copy out of the slice within this call.
        // `T: Copy`, so this is a single bulk copy.
        let ptr = unsafe { T::data_ptr(self.handle) };
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Returns a row-contiguous copy (or the same array if already dense).
    pub fn contiguous(&self, stream: &Stream) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: handle/stream valid; `allow_col_major = false` forces
        // row-major; result written into `out`.
        let status = unsafe { sys::mlx_contiguous(&mut out, self.handle, false, stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Elementwise addition: `self + other`.
    pub fn add(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_add)
    }

    /// Elementwise subtraction: `self - other`.
    pub fn subtract(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_subtract)
    }

    /// Elementwise multiplication: `self * other`.
    pub fn multiply(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_multiply)
    }

    /// Elementwise division: `self / other`.
    pub fn divide(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_divide)
    }

    /// Elementwise square root.
    pub fn sqrt(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_sqrt)
    }

    /// Elementwise exponential.
    pub fn exp(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_exp)
    }

    /// Elementwise absolute value.
    pub fn abs(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_abs)
    }

    /// Elementwise negation.
    pub fn negative(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_negative)
    }

    /// Elementwise natural logarithm.
    pub fn log(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_log)
    }

    /// Elementwise sine.
    pub fn sin(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_sin)
    }

    /// Elementwise cosine.
    pub fn cos(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_cos)
    }

    /// Elementwise square.
    pub fn square(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_square)
    }

    /// Elementwise hyperbolic tangent.
    pub fn tanh(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_tanh)
    }

    /// Elementwise power: `self ** other`.
    pub fn power(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_power)
    }

    /// Elementwise maximum of two arrays.
    pub fn maximum(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_maximum)
    }

    /// Elementwise minimum of two arrays.
    pub fn minimum(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_minimum)
    }

    /// Sum of all elements, returning a scalar array.
    ///
    /// With `keepdims == false` the result is 0-dimensional.
    pub fn sum(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_sum)
    }

    /// Mean of all elements, returning a scalar array.
    ///
    /// With `keepdims == false` the result is 0-dimensional.
    pub fn mean(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_mean)
    }

    /// Sum over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn sum_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_sum_axes)
    }

    /// Mean over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn mean_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_mean_axes)
    }

    /// Maximum over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn max_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_max_axes)
    }

    /// Minimum over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn min_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_min_axes)
    }

    /// Product over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn prod_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_prod_axes)
    }

    /// Returns a new array with the same data reinterpreted as `shape`.
    ///
    /// The product of `shape` must equal [`size`](Self::size).
    pub fn reshape(&self, shape: &[i32], stream: &Stream) -> Result<Array> {
        self.shape_op(shape, stream, sys::mlx_reshape)
    }

    /// Broadcasts the array to `shape`.
    pub fn broadcast_to(&self, shape: &[i32], stream: &Stream) -> Result<Array> {
        self.shape_op(shape, stream, sys::mlx_broadcast_to)
    }

    /// Reverses the order of all axes (a full transpose).
    pub fn transpose(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_transpose)
    }

    /// Removes all axes of length 1.
    pub fn squeeze(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_squeeze)
    }

    /// Inserts a new axis of length 1 at position `axis`.
    pub fn expand_dims(&self, axis: i32, stream: &Stream) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: handle/stream are valid; `mlx_expand_dims` writes the result into `out`.
        let status = unsafe { sys::mlx_expand_dims(&mut out, self.handle, axis, stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(shape, shape_num, dtype, stream)`
    /// constructors.
    fn fill_op(
        shape: &[i32],
        dtype: sys::mlx_dtype,
        stream: &Stream,
        op: unsafe extern "C" fn(
            *mut sys::mlx_array,
            *const i32,
            usize,
            sys::mlx_dtype,
            sys::mlx_stream,
        ) -> i32,
    ) -> Result<Array> {
        error::install();
        let shape_ptr = as_ffi_ptr(shape);
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: `shape_ptr`/`shape.len()` describe a valid slice (or null/0);
        // stream is valid; `op` writes the result into `out`.
        let status = unsafe { op(&mut out, shape_ptr, shape.len(), dtype, stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(a, shape, shape_num, stream)` shape ops.
    fn shape_op(
        &self,
        shape: &[i32],
        stream: &Stream,
        op: unsafe extern "C" fn(
            *mut sys::mlx_array,
            sys::mlx_array,
            *const i32,
            usize,
            sys::mlx_stream,
        ) -> i32,
    ) -> Result<Array> {
        error::install();
        let shape_ptr = as_ffi_ptr(shape);
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: `shape_ptr`/`shape.len()` describe a valid slice (or null/0)
        // for the call; all handles are valid; `op` writes into `out`.
        let status = unsafe {
            op(
                &mut out,
                self.handle,
                shape_ptr,
                shape.len(),
                stream.as_raw(),
            )
        };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(a, b, stream)` binary ops.
    fn binary_op(
        &self,
        other: &Array,
        stream: &Stream,
        op: unsafe extern "C" fn(
            *mut sys::mlx_array,
            sys::mlx_array,
            sys::mlx_array,
            sys::mlx_stream,
        ) -> i32,
    ) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: all handles are valid; `op` writes the result into `out`.
        let status = unsafe { op(&mut out, self.handle, other.as_raw(), stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(a, stream)` unary ops.
    fn unary_op(
        &self,
        stream: &Stream,
        op: unsafe extern "C" fn(*mut sys::mlx_array, sys::mlx_array, sys::mlx_stream) -> i32,
    ) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: handle/stream are valid; `op` writes the result into `out`.
        let status = unsafe { op(&mut out, self.handle, stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(a, keepdims, stream)` full reductions.
    fn reduce_op(
        &self,
        keepdims: bool,
        stream: &Stream,
        op: unsafe extern "C" fn(*mut sys::mlx_array, sys::mlx_array, bool, sys::mlx_stream) -> i32,
    ) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: handle/stream are valid; `op` writes the result into `out`.
        let status = unsafe { op(&mut out, self.handle, keepdims, stream.as_raw()) };
        Self::from_op(out, status)
    }

    /// Shared plumbing for `res = op(a, axes, axes_num, keepdims, stream)`
    /// reductions over specific axes.
    fn reduce_axes_op(
        &self,
        axes: &[i32],
        keepdims: bool,
        stream: &Stream,
        op: unsafe extern "C" fn(
            *mut sys::mlx_array,
            sys::mlx_array,
            *const i32,
            usize,
            bool,
            sys::mlx_stream,
        ) -> i32,
    ) -> Result<Array> {
        error::install();
        let axes_ptr = as_ffi_ptr(axes);
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: `axes_ptr`/`axes.len()` describe a valid slice (or null/0) for
        // the duration of the call; all handles are valid; `op` writes the
        // result into `out`.
        let status = unsafe {
            op(
                &mut out,
                self.handle,
                axes_ptr,
                axes.len(),
                keepdims,
                stream.as_raw(),
            )
        };
        Self::from_op(out, status)
    }

    /// Wraps an op's `out` handle and status code into a `Result`.
    ///
    /// On failure, frees the (unused) `out` handle and returns the captured
    /// MLX error message.
    fn from_op(out: sys::mlx_array, status: i32) -> Result<Array> {
        match error::check(status) {
            Ok(()) => Ok(unsafe { Self::from_raw(out) }),
            Err(e) => {
                // SAFETY: `out` was created by mlx and is owned here; free it so
                // the failed op doesn't leak.
                unsafe { sys::mlx_array_free(out) };
                Err(e)
            }
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        error::install();
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

// Arithmetic operators run on the current default stream (see
// [`Stream::set_as_default`]). For explicit stream control — and to handle
// errors — call the inherent methods (`a.add(&b, &stream)?`) instead.
//
// Operators cannot return `Result`, so they **panic** if the underlying op
// fails (e.g. incompatible shapes). Use the methods when failure is possible.
//
// Implemented on `&Array` so operands are borrowed, not consumed: `&a + &b`
// leaves both arrays usable afterwards.
macro_rules! impl_binop {
    ($($trait:ident :: $method:ident => $op:ident),* $(,)?) => {
        $(
            impl std::ops::$trait for &Array {
                type Output = Array;
                fn $method(self, rhs: &Array) -> Array {
                    self.$op(rhs, &Stream::default()).unwrap_or_else(|e| {
                        panic!(concat!("Array::", stringify!($op), " failed: {}"), e)
                    })
                }
            }
        )*
    };
}

impl_binop! {
    Add::add => add,
    Sub::sub => subtract,
    Mul::mul => multiply,
    Div::div => divide,
}

impl std::ops::Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        self.negative(&Stream::default())
            .unwrap_or_else(|e| panic!("Array::negative failed: {e}"))
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
    fn from_scalar_is_zero_dim() {
        let a = Array::from_scalar(42.0f32);
        assert_eq!(a.ndim(), 0);
        assert_eq!(a.item::<f32>(), 42.0);
    }

    #[test]
    fn empty_array_has_no_elements() {
        // Exercises the null-pointer path for empty `data`.
        let a = Array::from_slice::<f32>(&[], &[0]);
        assert_eq!(a.size(), 0);
        assert_eq!(a.shape(), vec![0]);
        assert!(a.to_vec::<f32>().is_empty());
    }

    #[test]
    fn zeros_and_ones_fill_shape() {
        let s = Stream::cpu();

        let z = Array::zeros::<f32>(&[2, 3], &s).unwrap();
        assert_eq!(z.shape(), vec![2, 3]);
        assert_eq!(z.to_vec::<f32>(), vec![0.0; 6]);

        let o = Array::ones::<f32>(&[2, 2], &s).unwrap();
        assert_eq!(o.to_vec::<f32>(), vec![1.0; 4]);

        // The type parameter picks the dtype, so integer arrays work too.
        let i = Array::ones::<i32>(&[3], &s).unwrap();
        assert_eq!(i.to_vec::<i32>(), vec![1, 1, 1]);
    }

    #[test]
    fn full_broadcasts_a_scalar() {
        let s = Stream::cpu();
        let a = Array::full(&[2, 2], 7.5f32, &s).unwrap();
        assert_eq!(a.shape(), vec![2, 2]);
        assert_eq!(a.to_vec::<f32>(), vec![7.5; 4]);

        // An empty shape yields a 0-d array, not an error.
        let scalar = Array::full(&[], 3i32, &s).unwrap();
        assert_eq!(scalar.ndim(), 0);
        assert_eq!(scalar.item::<i32>(), 3);
    }

    #[test]
    fn arange_is_half_open() {
        let s = Stream::cpu();
        // `stop` is exclusive, so this is 0..5, not 0..=5.
        let a = Array::arange::<i32>(0.0, 5.0, 1.0, &s).unwrap();
        assert_eq!(a.to_vec::<i32>(), vec![0, 1, 2, 3, 4]);

        let f = Array::arange::<f32>(1.0, 2.0, 0.5, &s).unwrap();
        assert_eq!(f.to_vec::<f32>(), vec![1.0, 1.5]);
    }

    #[test]
    fn constructors_compose_with_ops() {
        let s = Stream::cpu();
        // ones + ones == twos, over a shape built by a constructor.
        let o = Array::ones::<f32>(&[4], &s).unwrap();
        assert_eq!(o.add(&o, &s).unwrap().to_vec::<f32>(), vec![2.0; 4]);

        // A 0-d scalar broadcasts against a larger array.
        let a = Array::arange::<f32>(0.0, 4.0, 1.0, &s).unwrap();
        let two = Array::from_scalar(2.0f32);
        assert_eq!(
            a.multiply(&two, &s).unwrap().to_vec::<f32>(),
            vec![0.0, 2.0, 4.0, 6.0]
        );
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

    #[test]
    fn binary_ops_compute_elementwise() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[4.0f32, 5.0, 6.0], &[3]);

        assert_eq!(a.add(&b, &s).unwrap().to_vec::<f32>(), vec![5.0, 7.0, 9.0]);
        assert_eq!(
            b.subtract(&a, &s).unwrap().to_vec::<f32>(),
            vec![3.0, 3.0, 3.0]
        );
        assert_eq!(
            a.multiply(&b, &s).unwrap().to_vec::<f32>(),
            vec![4.0, 10.0, 18.0]
        );
        assert_eq!(
            b.divide(&a, &s).unwrap().to_vec::<f32>(),
            vec![4.0, 2.5, 2.0]
        );
    }

    #[test]
    fn unary_ops_compute_elementwise() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 4.0, 9.0], &[3]);
        assert_eq!(a.sqrt(&s).unwrap().to_vec::<f32>(), vec![1.0, 2.0, 3.0]);

        let b = Array::from_slice(&[-1.0f32, 2.0, -3.0], &[3]);
        assert_eq!(b.abs(&s).unwrap().to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            b.negative(&s).unwrap().to_vec::<f32>(),
            vec![1.0, -2.0, 3.0]
        );
    }

    #[test]
    fn more_unary_ops() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        assert_eq!(a.square(&s).unwrap().to_vec::<f32>(), vec![1.0, 4.0, 9.0]);

        // log(1) == 0, and log(e) ~= 1.
        let b = Array::from_slice(&[1.0f32, std::f32::consts::E], &[2]);
        let logged = b.log(&s).unwrap().to_vec::<f32>();
        assert!((logged[0]).abs() < 1e-6);
        assert!((logged[1] - 1.0).abs() < 1e-6);

        // sin(0) == 0, cos(0) == 1, tanh(0) == 0.
        let z = Array::from_slice(&[0.0f32], &[1]);
        assert!(z.sin(&s).unwrap().item::<f32>().abs() < 1e-6);
        assert!((z.cos(&s).unwrap().item::<f32>() - 1.0).abs() < 1e-6);
        assert!(z.tanh(&s).unwrap().item::<f32>().abs() < 1e-6);
    }

    #[test]
    fn more_binary_ops() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[3.0f32, 2.0, 1.0], &[3]);

        assert_eq!(
            a.maximum(&b, &s).unwrap().to_vec::<f32>(),
            vec![3.0, 2.0, 3.0]
        );
        assert_eq!(
            a.minimum(&b, &s).unwrap().to_vec::<f32>(),
            vec![1.0, 2.0, 1.0]
        );

        let base = Array::from_slice(&[2.0f32, 3.0], &[2]);
        let exp = Array::from_slice(&[3.0f32, 2.0], &[2]);
        assert_eq!(
            base.power(&exp, &s).unwrap().to_vec::<f32>(),
            vec![8.0, 9.0]
        );
    }

    #[test]
    fn reductions_produce_scalars() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);

        let sum = a.sum(false, &s).unwrap();
        assert_eq!(sum.ndim(), 0);
        assert_eq!(sum.item::<f32>(), 10.0);

        assert_eq!(a.mean(false, &s).unwrap().item::<f32>(), 2.5);
    }

    #[test]
    fn keepdims_retains_rank() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let sum = a.sum(true, &s).unwrap();
        assert_eq!(sum.shape(), vec![1, 1]);
        assert_eq!(sum.item::<f32>(), 10.0);
    }

    #[test]
    fn axis_reductions() {
        let s = Stream::cpu();
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        // Sum over axis 0 (rows) -> [5, 7, 9], shape [3].
        let col_sums = a.sum_axes(&[0], false, &s).unwrap();
        assert_eq!(col_sums.shape(), vec![3]);
        assert_eq!(col_sums.to_vec::<f32>(), vec![5.0, 7.0, 9.0]);

        // Sum over axis 1 (cols) -> [6, 15], shape [2].
        let row_sums = a.sum_axes(&[1], false, &s).unwrap();
        assert_eq!(row_sums.to_vec::<f32>(), vec![6.0, 15.0]);

        // keepdims keeps the reduced axis as size 1.
        let kept = a.sum_axes(&[1], true, &s).unwrap();
        assert_eq!(kept.shape(), vec![2, 1]);

        // max / min over axis 0; mean / prod over axis 1.
        assert_eq!(
            a.max_axes(&[0], false, &s).unwrap().to_vec::<f32>(),
            vec![4.0, 5.0, 6.0]
        );
        assert_eq!(
            a.min_axes(&[0], false, &s).unwrap().to_vec::<f32>(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            a.mean_axes(&[1], false, &s).unwrap().to_vec::<f32>(),
            vec![2.0, 5.0]
        );
        assert_eq!(
            a.prod_axes(&[1], false, &s).unwrap().to_vec::<f32>(),
            vec![6.0, 120.0]
        );
    }

    #[test]
    fn empty_axes_reduction_is_noop() {
        // Reducing over no axes must not pass a dangling pointer to C; MLX
        // treats it as an identity that leaves the values unchanged.
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let r = a.sum_axes(&[], false, &s).unwrap();
        assert_eq!(r.shape(), vec![2, 2]);
        assert_eq!(r.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn reshape_changes_shape_not_data() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let r = a.reshape(&[3, 2], &s).unwrap();
        assert_eq!(r.shape(), vec![3, 2]);
        assert_eq!(r.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn row_contiguity_detection() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // Freshly built arrays are row-contiguous (fast path in to_vec).
        assert!(a.is_row_contiguous());
        // A transpose is a strided view. Strides only reflect the real layout
        // after evaluation (MLX is lazy), which is exactly when to_vec checks.
        let t = a.transpose(&s).unwrap();
        t.eval();
        assert!(!t.is_row_contiguous());
    }

    #[test]
    fn transpose_reverses_axes() {
        let s = Stream::cpu();
        // [[1, 2, 3],
        //  [4, 5, 6]]  ->  [[1, 4], [2, 5], [3, 6]]
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = a.transpose(&s).unwrap();
        assert_eq!(t.shape(), vec![3, 2]);
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn broadcast_to_expands() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = a.broadcast_to(&[2, 3], &s).unwrap();
        assert_eq!(b.shape(), vec![2, 3]);
        assert_eq!(b.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn squeeze_and_expand_dims() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3, 1]);
        let sq = a.squeeze(&s).unwrap();
        assert_eq!(sq.shape(), vec![3]);

        let ex = sq.expand_dims(0, &s).unwrap();
        assert_eq!(ex.shape(), vec![1, 3]);
    }

    #[test]
    fn incompatible_shapes_return_err() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[1.0f32, 2.0], &[2]);
        // Broadcasting [3] against [2] is invalid; MLX should report an error
        // rather than aborting the process.
        let err = a.add(&b, &s).unwrap_err();
        assert!(
            !err.message().is_empty(),
            "expected a non-empty error message"
        );
    }

    #[test]
    fn item_reads_scalar() {
        let a = Array::from_slice(&[42.0f32], &[]);
        assert_eq!(a.item::<f32>(), 42.0);
        let b = Array::from_slice(&[7i32], &[]);
        assert_eq!(b.item::<i32>(), 7);
    }

    #[test]
    #[should_panic(expected = "requires a single-element array")]
    fn item_on_non_scalar_panics() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let _ = a.item::<f32>();
    }

    #[test]
    fn to_vec_is_generic_over_dtype() {
        let ints = Array::from_slice(&[1i32, 2, 3], &[3]);
        assert_eq!(ints.to_vec::<i32>(), vec![1, 2, 3]);
        let floats = Array::from_slice(&[1.5f32, 2.5], &[2]);
        assert_eq!(floats.to_vec::<f32>(), vec![1.5, 2.5]);
    }

    #[test]
    #[should_panic(expected = "does not match requested element type")]
    fn to_vec_wrong_dtype_panics() {
        let ints = Array::from_slice(&[1i32, 2, 3], &[3]);
        let _ = ints.to_vec::<f32>();
    }

    #[test]
    fn to_vec_of_empty_array_is_empty() {
        // A zero-element array must not deref a (possibly null) data pointer.
        let empty = Array::from_slice::<f32>(&[], &[0]);
        assert_eq!(empty.size(), 0);
        assert!(empty.to_vec::<f32>().is_empty());
    }

    #[test]
    fn operators_match_methods() {
        // Operators run on the default stream; results should equal the
        // explicit-method equivalents.
        let a = Array::from_slice(&[10.0f32, 20.0, 30.0], &[3]);
        let b = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);

        assert_eq!((&a + &b).to_vec::<f32>(), vec![11.0, 22.0, 33.0]);
        assert_eq!((&a - &b).to_vec::<f32>(), vec![9.0, 18.0, 27.0]);
        assert_eq!((&a * &b).to_vec::<f32>(), vec![10.0, 40.0, 90.0]);
        assert_eq!((&a / &b).to_vec::<f32>(), vec![10.0, 10.0, 10.0]);
        assert_eq!((-&a).to_vec::<f32>(), vec![-10.0, -20.0, -30.0]);

        // Operands are borrowed, so `a` is still usable here.
        assert_eq!(a.to_vec::<f32>(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    #[should_panic(expected = "Array::add failed: MLX error:")]
    fn operator_panic_carries_mlx_message() {
        // A failing operator panics with both the op name and the underlying
        // MLX diagnostic, not a generic message.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let _ = &a + &b;
    }
}
