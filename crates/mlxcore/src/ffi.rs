//! Small helpers shared by the modules that call into `mlxcore-sys`.

use mlxcore_sys as sys;

/// Returns a pointer to `slice`'s data, or null when it is empty.
///
/// For an empty slice `as_ptr()` is non-null but dangling. The mlx-c functions
/// we call take a `(ptr, len)` pair and build a container from it, so they never
/// dereference the pointer when `len == 0` — but passing an explicit null keeps
/// a dangling pointer from crossing the FFI boundary at all.
pub(crate) fn as_ffi_ptr<T>(slice: &[T]) -> *const T {
    if slice.is_empty() {
        std::ptr::null()
    } else {
        slice.as_ptr()
    }
}

/// An absent `mlx_array`, for the parameters mlx-c documents as `may be null`.
///
/// The C shims test the handle's `ctx` pointer and translate a null one into
/// `std::nullopt`, so this is how an optional array argument is omitted. It owns
/// nothing and must not be freed.
pub(crate) fn absent_array() -> sys::mlx_array {
    sys::mlx_array {
        ctx: std::ptr::null_mut(),
    }
}
