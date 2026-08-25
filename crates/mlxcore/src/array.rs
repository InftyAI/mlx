//! A safe wrapper around `mlx_array`.

use std::ffi::CStr;
use std::fmt;

use mlxcore_sys as sys;

use crate::dtype::{ArrayElement, Dtype};
use crate::error::{self, Result};
use crate::ffi::as_ffi_ptr;
use crate::stream::Stream;

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
    pub(crate) fn as_raw(&self) -> sys::mlx_array {
        self.handle
    }

    /// Builds an array from a slice of values with the given shape.
    ///
    /// The MLX dtype is chosen from the element type `T` at compile time (e.g.
    /// `&[f32]` produces a `float32` array, `&[i32]` an `int32` array).
    ///
    /// Suffix float literals: `&[1.0, 2.0]` infers `f64`, and float64 is not
    /// supported on the GPU. Write `&[1.0f32, 2.0]`.
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
        let handle = unsafe {
            sys::mlx_array_new_data(data_ptr, shape_ptr, shape.len() as i32, T::DTYPE.as_raw())
        };
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
                T::DTYPE.as_raw(),
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
        let status = unsafe {
            sys::mlx_arange(
                &mut out,
                start,
                stop,
                step,
                T::DTYPE.as_raw(),
                stream.as_raw(),
            )
        };
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

    /// Element type of the array.
    ///
    /// MLX decides this itself: constructors take it from the Rust type, but ops
    /// promote (an `i32` array times a `f32` one is `f32`) and comparisons always
    /// give [`Dtype::Bool`], so this is the way to check what an op produced.
    pub fn dtype(&self) -> Dtype {
        // SAFETY: mlx_array_dtype reads a valid handle.
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.handle) })
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
        let dtype = self.dtype();
        assert_eq!(
            dtype,
            T::DTYPE,
            "array dtype {dtype} does not match requested element type"
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

    /// Converts the elements to the dtype chosen by `T`.
    ///
    /// Call as `a.astype::<f32>(&stream)`. This is the only way to change an
    /// array's dtype — it is otherwise fixed by whichever constructor built the
    /// array. Note that binary ops already promote mixed dtypes on their own
    /// (`int32 + float32` yields `float32`), so this is for *explicit* control.
    ///
    /// Conversions follow C++ cast semantics rather than Rust's, and never
    /// error: float-to-integer truncates toward zero (`2.7` becomes `2`),
    /// `bool` becomes 0 or 1, and numeric-to-`bool` tests `!= 0`. Converting a
    /// float that is out of the target integer's range is *unspecified*.
    ///
    /// Only dtypes with an [`ArrayElement`] impl can be targeted, so MLX's
    /// `float16`, `bfloat16`, and `complex64` are out of reach for now.
    pub fn astype<T: ArrayElement>(&self, stream: &Stream) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: handle/stream are valid; the result is written into `out`.
        let status =
            unsafe { sys::mlx_astype(&mut out, self.handle, T::DTYPE.as_raw(), stream.as_raw()) };
        Self::from_op(out, status)
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

    /// Elementwise `self == other`, as a `bool` array.
    ///
    /// Like the arithmetic ops these broadcast, so comparing against a
    /// [`from_scalar`](Self::from_scalar) array gives a mask over every element.
    /// Read the result back with `to_vec::<bool>()`.
    ///
    /// These are named after the MLX operations rather than spelled `eq`/`lt`,
    /// because they are elementwise and return an array — they are not the
    /// whole-array `bool` answer that `PartialEq`/`PartialOrd` would imply. For
    /// that, see [`array_equal`](Self::array_equal).
    pub fn equal(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_equal)
    }

    /// Elementwise `self != other`, as a `bool` array.
    pub fn not_equal(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_not_equal)
    }

    /// Elementwise `self > other`, as a `bool` array.
    pub fn greater(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_greater)
    }

    /// Elementwise `self >= other`, as a `bool` array.
    pub fn greater_equal(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_greater_equal)
    }

    /// Elementwise `self < other`, as a `bool` array.
    pub fn less(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_less)
    }

    /// Elementwise `self <= other`, as a `bool` array.
    pub fn less_equal(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_less_equal)
    }

    /// Elementwise logical and, as a `bool` array.
    ///
    /// Non-`bool` operands are compared against zero first, so this is a
    /// truthiness test rather than a bitwise one.
    pub fn logical_and(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_logical_and)
    }

    /// Elementwise logical or, as a `bool` array.
    pub fn logical_or(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_logical_or)
    }

    /// Elementwise logical negation, as a `bool` array.
    pub fn logical_not(&self, stream: &Stream) -> Result<Array> {
        self.unary_op(stream, sys::mlx_logical_not)
    }

    /// Whether the two arrays have the same shape and equal elements, as a
    /// 0-dimensional `bool` array.
    ///
    /// This is the whole-array answer, in contrast to the elementwise
    /// [`equal`](Self::equal). Arrays of different shapes are unequal — unlike
    /// `equal`, nothing is broadcast. With `equal_nan == true` two NaNs in the
    /// same position count as equal.
    pub fn array_equal(&self, other: &Array, equal_nan: bool, stream: &Stream) -> Result<Array> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: all handles are valid; the result is written into `out`.
        let status = unsafe {
            sys::mlx_array_equal(
                &mut out,
                self.handle,
                other.as_raw(),
                equal_nan,
                stream.as_raw(),
            )
        };
        Self::from_op(out, status)
    }

    /// Matrix multiplication: `self @ other`.
    ///
    /// Unlike the elementwise ops, this contracts the last axis of `self`
    /// against the second-to-last of `other`: `(n, k) @ (k, m)` gives
    /// `(n, m)`. Mismatched inner dimensions return an error rather than
    /// panicking.
    ///
    /// Following NumPy, a 1-dimensional operand is treated as a matrix for the
    /// duration of the product and then collapsed again: a leading axis is
    /// prepended to a 1-D `self`, a trailing axis appended to a 1-D `other`,
    /// and the corresponding axis is removed from the result. So `(3,) @ (3,)`
    /// is a 0-dimensional dot product, and `(2, 3) @ (3,)` is `(2,)`.
    ///
    /// Leading axes beyond the last two are batch dimensions and broadcast:
    /// `(2, 3, 4) @ (2, 4, 5)` gives `(2, 3, 5)`.
    ///
    /// There is deliberately no operator for this. `*` is already elementwise
    /// [`multiply`](Self::multiply), and Rust has no `@`.
    pub fn matmul(&self, other: &Array, stream: &Stream) -> Result<Array> {
        self.binary_op(other, stream, sys::mlx_matmul)
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

    /// Maximum of all elements, returning a scalar array.
    ///
    /// With `keepdims == false` the result is 0-dimensional.
    pub fn max(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_max)
    }

    /// Minimum of all elements, returning a scalar array.
    ///
    /// With `keepdims == false` the result is 0-dimensional.
    pub fn min(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_min)
    }

    /// Product of all elements, returning a scalar array.
    ///
    /// With `keepdims == false` the result is 0-dimensional.
    pub fn prod(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_prod)
    }

    /// Whether every element is true (nonzero), as a `bool` array.
    ///
    /// With `keepdims == false` the result is 0-dimensional. Empty arrays reduce
    /// to `true`, the identity of logical and.
    pub fn all(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_all)
    }

    /// Whether any element is true (nonzero), as a `bool` array.
    ///
    /// With `keepdims == false` the result is 0-dimensional. Empty arrays reduce
    /// to `false`, the identity of logical or.
    pub fn any(&self, keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_op(keepdims, stream, sys::mlx_any)
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

    /// Whether every element is true (nonzero) over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn all_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_all_axes)
    }

    /// Whether any element is true (nonzero) over the given axes.
    ///
    /// With `keepdims == false` the reduced axes are removed; otherwise they
    /// are kept with size 1.
    pub fn any_axes(&self, axes: &[i32], keepdims: bool, stream: &Stream) -> Result<Array> {
        self.reduce_axes_op(axes, keepdims, stream, sys::mlx_any_axes)
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
        dtype: Dtype,
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
        let status = unsafe {
            op(
                &mut out,
                shape_ptr,
                shape.len(),
                dtype.as_raw(),
                stream.as_raw(),
            )
        };
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
    pub(crate) fn from_op(out: sys::mlx_array, status: i32) -> Result<Array> {
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

// Scalar operands, so `&a * 2.0f32` needs no hand-built 1-element array. The
// scalar becomes a 0-dimensional array that broadcasts across `self`.
//
// Suffix the literal: the scalar's Rust type picks its dtype and MLX promotes to
// the wider one, so `&a * 2.0` widens to float64, which the GPU rejects.
//
// Same panic-on-error contract as the array-to-array operators above.
macro_rules! impl_scalar_rhs_binop {
    ($($trait:ident :: $method:ident => $op:ident),* $(,)?) => {
        $(
            impl<T: ArrayElement> std::ops::$trait<T> for &Array {
                type Output = Array;
                fn $method(self, rhs: T) -> Array {
                    self.$op(&Array::from_scalar(rhs), &Stream::default())
                        .unwrap_or_else(|e| {
                            panic!(concat!("Array::", stringify!($op), " failed: {}"), e)
                        })
                }
            }
        )*
    };
}

impl_scalar_rhs_binop! {
    Add::add => add,
    Sub::sub => subtract,
    Mul::mul => multiply,
    Div::div => divide,
}

/// Applies `op(scalar, rhs)` on the default stream, panicking on failure.
///
/// Shared by the scalar-on-the-left operator impls below, which cannot be
/// written as one blanket impl (see the comment on [`impl_scalar_lhs_binop`]).
fn scalar_lhs_op<T: ArrayElement>(
    lhs: T,
    rhs: &Array,
    op: fn(&Array, &Array, &Stream) -> Result<Array>,
    name: &str,
) -> Array {
    let lhs = Array::from_scalar(lhs);
    op(&lhs, rhs, &Stream::default()).unwrap_or_else(|e| panic!("Array::{name} failed: {e}"))
}

// The mirror of the impls above, for `2.0f32 * &a`. Worth having because `Sub`
// and `Div` are not commutative: `10.0 - &a` cannot be spelled with the
// scalar-on-the-right impls.
//
// These cannot be one blanket `impl<T: ArrayElement> Add<&Array> for T` — that
// implements a foreign trait for an uncovered type parameter, which coherence
// forbids (E0210). So they are generated per concrete element type instead.
macro_rules! impl_scalar_lhs_binop {
    ($($ty:ty),* $(,)?) => {
        $(
            impl std::ops::Add<&Array> for $ty {
                type Output = Array;
                fn add(self, rhs: &Array) -> Array {
                    scalar_lhs_op(self, rhs, Array::add, "add")
                }
            }

            impl std::ops::Sub<&Array> for $ty {
                type Output = Array;
                fn sub(self, rhs: &Array) -> Array {
                    scalar_lhs_op(self, rhs, Array::subtract, "subtract")
                }
            }

            impl std::ops::Mul<&Array> for $ty {
                type Output = Array;
                fn mul(self, rhs: &Array) -> Array {
                    scalar_lhs_op(self, rhs, Array::multiply, "multiply")
                }
            }

            impl std::ops::Div<&Array> for $ty {
                type Output = Array;
                fn div(self, rhs: &Array) -> Array {
                    scalar_lhs_op(self, rhs, Array::divide, "divide")
                }
            }
        )*
    };
}

// Every type with an `ArrayElement` impl, so the two directions stay symmetric.
impl_scalar_lhs_binop!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

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
    fn astype_converts_dtype() {
        let s = Stream::cpu();
        let ints = Array::from_slice(&[1i32, 2, 3], &[3]);

        // Reading as f32 is only possible after converting.
        let floats = ints.astype::<f32>(&s).unwrap();
        assert_eq!(floats.to_vec::<f32>(), vec![1.0, 2.0, 3.0]);
        // The original is untouched: astype returns a new array.
        assert_eq!(ints.to_vec::<i32>(), vec![1, 2, 3]);
        // Shape is preserved; only the dtype changes.
        assert_eq!(floats.shape(), ints.shape());
    }

    #[test]
    fn astype_narrowing_truncates_toward_zero() {
        let s = Stream::cpu();
        // C++ cast semantics: truncation, not rounding, and no error.
        let a = Array::from_slice(&[2.7f32, -2.7, 0.9], &[3]);
        assert_eq!(a.astype::<i32>(&s).unwrap().to_vec::<i32>(), vec![2, -2, 0]);
    }

    #[test]
    fn astype_bool_round_trip() {
        let s = Stream::cpu();
        // numeric -> bool is `!= 0`; bool -> numeric is 0/1.
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0, -3.0], &[4]);
        let flags = a.astype::<bool>(&s).unwrap();
        assert_eq!(flags.to_vec::<bool>(), vec![false, true, true, true]);
        assert_eq!(
            flags.astype::<i32>(&s).unwrap().to_vec::<i32>(),
            vec![0, 1, 1, 1]
        );
    }

    #[test]
    fn astype_to_same_dtype_is_a_noop() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.5f32, 2.5], &[2]);
        assert_eq!(a.astype::<f32>(&s).unwrap().to_vec::<f32>(), vec![1.5, 2.5]);
    }

    #[test]
    fn matmul_contracts_inner_dimension() {
        let s = Stream::cpu();
        // (2,3) @ (3,2) -> (2,2)
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Array::from_slice(&[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        let c = a.matmul(&b, &s).unwrap();
        assert_eq!(c.shape(), vec![2, 2]);
        // [1,2,3]·[7,9,11] = 58, [1,2,3]·[8,10,12] = 64,
        // [4,5,6]·[7,9,11] = 139, [4,5,6]·[8,10,12] = 154.
        assert_eq!(c.to_vec::<f32>(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_is_not_elementwise_multiply() {
        let s = Stream::cpu();
        // Same square operands: matmul and `*` must disagree.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(
            a.matmul(&a, &s).unwrap().to_vec::<f32>(),
            vec![7.0, 10.0, 15.0, 22.0]
        );
        assert_eq!(
            a.multiply(&a, &s).unwrap().to_vec::<f32>(),
            vec![1.0, 4.0, 9.0, 16.0]
        );
    }

    #[test]
    fn matmul_collapses_one_dim_operands() {
        let s = Stream::cpu();
        let v = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);

        // (3,) @ (3,) is a dot product, and the result is 0-dimensional.
        let dot = v.matmul(&v, &s).unwrap();
        assert_eq!(dot.ndim(), 0);
        assert_eq!(dot.item::<f32>(), 14.0); // 1 + 4 + 9

        // (2,3) @ (3,) -> (2,), the appended axis is dropped again.
        let m = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let mv = m.matmul(&v, &s).unwrap();
        assert_eq!(mv.shape(), vec![2]);
        assert_eq!(mv.to_vec::<f32>(), vec![14.0, 32.0]);
    }

    #[test]
    fn matmul_broadcasts_batch_dimensions() {
        let s = Stream::cpu();
        // Leading axes are batch dims: (2,2,3) @ (2,3,2) -> (2,2,2).
        let a = Array::arange::<f32>(0.0, 12.0, 1.0, &s)
            .unwrap()
            .reshape(&[2, 2, 3], &s)
            .unwrap();
        let b = Array::ones::<f32>(&[2, 3, 2], &s).unwrap();
        let c = a.matmul(&b, &s).unwrap();
        assert_eq!(c.shape(), vec![2, 2, 2]);
        // With a ones matrix each output is the row sum, duplicated per column.
        assert_eq!(
            c.to_vec::<f32>(),
            vec![3.0, 3.0, 12.0, 12.0, 21.0, 21.0, 30.0, 30.0]
        );
    }

    #[test]
    fn matmul_with_transpose() {
        let s = Stream::cpu();
        // The shape a linear layer uses: x @ w.T, (2,3) @ (2,3).T -> (2,2).
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let w = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
        let out = x.matmul(&w.transpose(&s).unwrap(), &s).unwrap();
        assert_eq!(out.shape(), vec![2, 2]);
        // Rows of w select element 0 and element 1 of each row of x.
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn matmul_shape_mismatch_returns_err() {
        let s = Stream::cpu();
        // (2,3) @ (2,3) has inner dims 3 and 2 — not an error we should panic on.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let err = a.matmul(&a, &s).unwrap_err();
        assert!(
            !err.message().is_empty(),
            "expected a message from mlx, got {err:?}"
        );
    }

    #[test]
    fn scalar_on_the_right() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        assert_eq!((&a * 2.0f32).to_vec::<f32>(), vec![2.0, 4.0, 6.0]);
        assert_eq!((&a + 1.0f32).to_vec::<f32>(), vec![2.0, 3.0, 4.0]);
        assert_eq!((&a - 1.0f32).to_vec::<f32>(), vec![0.0, 1.0, 2.0]);
        assert_eq!((&a / 2.0f32).to_vec::<f32>(), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn scalar_on_the_left() {
        let a = Array::from_slice(&[1.0f32, 2.0, 4.0], &[3]);
        // Commutative ops match the right-hand form...
        assert_eq!((2.0f32 * &a).to_vec::<f32>(), (&a * 2.0f32).to_vec::<f32>());
        assert_eq!((1.0f32 + &a).to_vec::<f32>(), vec![2.0, 3.0, 5.0]);
        // ...and the non-commutative ones are why these impls exist.
        assert_eq!((10.0f32 - &a).to_vec::<f32>(), vec![9.0, 8.0, 6.0]);
        assert_eq!((8.0f32 / &a).to_vec::<f32>(), vec![8.0, 4.0, 2.0]);
    }

    #[test]
    fn integer_scalars_keep_integer_dtype() {
        let a = Array::from_slice(&[1i32, 2, 3], &[3]);
        let doubled = &a * 2i32;
        assert_eq!(doubled.to_vec::<i32>(), vec![2, 4, 6]);
        assert_eq!((2i32 * &a).to_vec::<i32>(), vec![2, 4, 6]);
    }

    #[test]
    fn suffixed_scalar_keeps_the_array_dtype() {
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let kept = &a * 2.0f32;
        assert!(
            format!("{kept:?}").contains("float32"),
            "expected float32, got {kept:?}"
        );
    }

    #[test]
    fn unsuffixed_float_scalar_widens_to_float64() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        // An unsuffixed literal infers `f64` (Rust's float fallback), and MLX
        // promotes the result to the wider dtype.
        let widened = a.multiply(&Array::from_scalar(2.0), &s).unwrap();
        assert!(
            format!("{widened:?}").contains("float64"),
            "expected float64, got {widened:?}"
        );
    }

    #[test]
    fn float64_is_rejected_on_the_gpu() {
        // Why the widening above matters: Metal has no float64. Checked through
        // the method API so the failure is an `Err` rather than a panic raised
        // in the middle of GPU encoding.
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let err = a
            .multiply(&Array::from_scalar(2.0), &Stream::gpu())
            .unwrap_err();
        assert!(
            err.message().contains("float64"),
            "unexpected message: {}",
            err.message()
        );
    }

    #[test]
    fn scalar_ops_broadcast_over_any_shape() {
        // The scalar is 0-dimensional, so it broadcasts regardless of rank.
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let scaled = &a * 10.0f32;
        assert_eq!(scaled.shape(), vec![2, 3]);
        assert_eq!(
            scaled.to_vec::<f32>(),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );
    }

    #[test]
    fn full_reductions_over_all_elements() {
        let s = Stream::cpu();
        // Reduces across every axis, not just the last one.
        let a = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
        assert_eq!(a.max(false, &s).unwrap().item::<f32>(), 9.0);
        assert_eq!(a.min(false, &s).unwrap().item::<f32>(), 1.0);

        let b = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(b.prod(false, &s).unwrap().item::<f32>(), 24.0);
    }

    #[test]
    fn full_reductions_respect_keepdims() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        // Without keepdims the rank collapses to 0...
        assert_eq!(a.max(false, &s).unwrap().ndim(), 0);
        // ...with it, every reduced axis is kept at size 1.
        assert_eq!(a.max(true, &s).unwrap().shape(), vec![1, 1]);
        assert_eq!(a.min(true, &s).unwrap().shape(), vec![1, 1]);
        assert_eq!(a.prod(true, &s).unwrap().shape(), vec![1, 1]);
    }

    #[test]
    fn full_reductions_agree_with_axis_versions() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
        // Reducing over all axes explicitly must match the full reduction.
        assert_eq!(
            a.max(false, &s).unwrap().item::<f32>(),
            a.max_axes(&[0, 1], false, &s).unwrap().item::<f32>()
        );
        assert_eq!(
            a.min(false, &s).unwrap().item::<f32>(),
            a.min_axes(&[0, 1], false, &s).unwrap().item::<f32>()
        );
        assert_eq!(
            a.prod(false, &s).unwrap().item::<f32>(),
            a.prod_axes(&[0, 1], false, &s).unwrap().item::<f32>()
        );
    }

    #[test]
    fn dtype_reports_the_element_type() {
        let s = Stream::cpu();
        assert_eq!(
            Array::from_slice(&[1.0f32, 2.0], &[2]).dtype(),
            Dtype::Float32
        );
        assert_eq!(Array::from_slice(&[1i32, 2], &[2]).dtype(), Dtype::Int32);
        assert_eq!(Array::from_scalar(true).dtype(), Dtype::Bool);
        assert_eq!(Array::zeros::<u8>(&[2], &s).unwrap().dtype(), Dtype::Uint8);

        // Ops decide the dtype themselves: astype converts, and mixing widens.
        let ints = Array::from_slice(&[1i32, 2], &[2]);
        assert_eq!(ints.astype::<f32>(&s).unwrap().dtype(), Dtype::Float32);
        let floats = Array::from_slice(&[0.5f32, 0.5], &[2]);
        assert_eq!(ints.add(&floats, &s).unwrap().dtype(), Dtype::Float32);
    }

    #[test]
    #[should_panic(expected = "array dtype float32 does not match")]
    fn reading_the_wrong_element_type_panics() {
        let _ = Array::from_slice(&[1.0f32, 2.0], &[2]).to_vec::<i32>();
    }

    #[test]
    fn comparisons_produce_bool_masks() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let b = Array::from_slice(&[3.0f32, 2.0, 1.0], &[3]);

        let eq = a.equal(&b, &s).unwrap();
        // Comparing floats gives bool, not float.
        assert_eq!(eq.dtype(), Dtype::Bool);
        assert_eq!(eq.to_vec::<bool>(), vec![false, true, false]);

        assert_eq!(
            a.not_equal(&b, &s).unwrap().to_vec::<bool>(),
            vec![true, false, true]
        );
        assert_eq!(
            a.greater(&b, &s).unwrap().to_vec::<bool>(),
            vec![false, false, true]
        );
        assert_eq!(
            a.greater_equal(&b, &s).unwrap().to_vec::<bool>(),
            vec![false, true, true]
        );
        assert_eq!(
            a.less(&b, &s).unwrap().to_vec::<bool>(),
            vec![true, false, false]
        );
        assert_eq!(
            a.less_equal(&b, &s).unwrap().to_vec::<bool>(),
            vec![true, true, false]
        );
    }

    #[test]
    fn comparisons_broadcast_against_a_scalar() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let mask = a.greater(&Array::from_scalar(2.0f32), &s).unwrap();
        // The scalar stretches over both axes, so the mask keeps `a`'s shape.
        assert_eq!(mask.shape(), vec![2, 2]);
        assert_eq!(mask.to_vec::<bool>(), vec![false, false, true, true]);
    }

    #[test]
    fn logical_ops_combine_masks() {
        let s = Stream::cpu();
        let x = Array::from_slice(&[true, true, false, false], &[4]);
        let y = Array::from_slice(&[true, false, true, false], &[4]);

        assert_eq!(
            x.logical_and(&y, &s).unwrap().to_vec::<bool>(),
            vec![true, false, false, false]
        );
        assert_eq!(
            x.logical_or(&y, &s).unwrap().to_vec::<bool>(),
            vec![true, true, true, false]
        );
        assert_eq!(
            x.logical_not(&s).unwrap().to_vec::<bool>(),
            vec![false, false, true, true]
        );
    }

    #[test]
    fn logical_ops_test_truthiness_of_numbers() {
        let s = Stream::cpu();
        // 2.0 is neither 1 nor 0: a bitwise `and` would give 0, truthiness gives true.
        let a = Array::from_slice(&[2.0f32, 0.0], &[2]);
        let b = Array::from_slice(&[4.0f32, 7.0], &[2]);
        assert_eq!(
            a.logical_and(&b, &s).unwrap().to_vec::<bool>(),
            vec![true, false]
        );
        assert_eq!(
            a.logical_not(&s).unwrap().to_vec::<bool>(),
            vec![false, true]
        );
    }

    #[test]
    fn all_and_any_reduce_masks_to_one_answer() {
        let s = Stream::cpu();
        let mixed = Array::from_slice(&[true, false], &[2]);
        assert!(!mixed.all(false, &s).unwrap().item::<bool>());
        assert!(mixed.any(false, &s).unwrap().item::<bool>());

        let all_true = Array::from_slice(&[true, true], &[2]);
        assert!(all_true.all(false, &s).unwrap().item::<bool>());

        let none = Array::from_slice(&[false, false], &[2]);
        assert!(!none.any(false, &s).unwrap().item::<bool>());

        // The reduction is over every axis, and `keepdims` behaves as elsewhere.
        let grid = Array::from_slice(&[true, false, true, true], &[2, 2]);
        assert!(!grid.all(false, &s).unwrap().item::<bool>());
        assert_eq!(grid.any(true, &s).unwrap().shape(), vec![1, 1]);
    }

    #[test]
    fn all_and_any_over_axes_reduce_per_row() {
        let s = Stream::cpu();
        let grid = Array::from_slice(&[true, false, true, true], &[2, 2]);
        // Row 0 is mixed, row 1 is all true.
        assert_eq!(
            grid.all_axes(&[1], false, &s).unwrap().to_vec::<bool>(),
            vec![false, true]
        );
        assert_eq!(
            grid.any_axes(&[1], false, &s).unwrap().to_vec::<bool>(),
            vec![true, true]
        );
        // Column 0 is all true, column 1 is mixed.
        assert_eq!(
            grid.all_axes(&[0], false, &s).unwrap().to_vec::<bool>(),
            vec![true, false]
        );
    }

    #[test]
    fn empty_reductions_return_the_identity() {
        let s = Stream::cpu();
        let empty = Array::from_slice::<bool>(&[], &[0]);
        assert!(empty.all(false, &s).unwrap().item::<bool>());
        assert!(!empty.any(false, &s).unwrap().item::<bool>());
    }

    #[test]
    fn array_equal_compares_whole_arrays() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let same = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let different = Array::from_slice(&[1.0f32, 9.0, 3.0], &[3]);

        assert!(a.array_equal(&same, false, &s).unwrap().item::<bool>());
        assert!(!a.array_equal(&different, false, &s).unwrap().item::<bool>());
        // Unlike `equal`, nothing is broadcast: a different shape is unequal
        // even when the elements line up.
        let row = Array::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3]);
        assert!(!a.array_equal(&row, false, &s).unwrap().item::<bool>());
        assert_eq!(a.equal(&row, &s).unwrap().shape(), vec![1, 3]);
    }

    #[test]
    fn array_equal_can_treat_nans_as_equal() {
        let s = Stream::cpu();
        let a = Array::from_slice(&[1.0f32, f32::NAN], &[2]);
        let b = Array::from_slice(&[1.0f32, f32::NAN], &[2]);
        // NaN != NaN, so the default comparison fails.
        assert!(!a.array_equal(&b, false, &s).unwrap().item::<bool>());
        assert!(a.array_equal(&b, true, &s).unwrap().item::<bool>());
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
