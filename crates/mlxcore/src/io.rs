//! Reading and writing arrays on disk, in the `safetensors` format.
//!
//! This is how a model's weights come in: one file mapping parameter names to
//! arrays. The arrays load lazily like any other MLX result — nothing touches
//! the disk until they are evaluated or used in an operation.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

use mlxcore_sys as sys;

use crate::array::Array;
use crate::error::{self, Error, Result};
use crate::stream::Stream;

/// Loads every tensor in a `safetensors` file, keyed by name.
///
/// Arrays keep whatever dtype the file stores, which for a published checkpoint
/// is usually `float16` or `bfloat16`. Neither has a Rust element type, so
/// [`Array::to_vec`] cannot read them directly — use
/// [`Array::astype_dtype`](Array::astype_dtype) to pick the dtype you want to
/// compute in.
///
/// The file's `__metadata__` header is not returned.
///
/// # Errors
/// Returns an error if the file is missing, is not a valid `safetensors`
/// container, or holds a dtype MLX does not support.
pub fn load_safetensors(path: impl AsRef<Path>, stream: &Stream) -> Result<HashMap<String, Array>> {
    error::install();
    let path = path.as_ref();
    let file = c_path(path)?;

    let mut arrays = ArrayMap::new();
    // mlx-c returns the metadata whether or not we want it, so it needs a
    // destination to be freed from.
    let mut metadata = StringMap::new();
    // SAFETY: both maps are freshly allocated and outlive the call; `file` is a
    // NUL-terminated path; the results are written into the two maps.
    let status = unsafe {
        sys::mlx_load_safetensors(
            arrays.as_mut_ptr(),
            metadata.as_mut_ptr(),
            file.as_ptr(),
            stream.as_raw(),
        )
    };
    error::check(status)?;
    arrays.to_hash_map()
}

/// Writes `arrays` to a `safetensors` file, keyed by name.
///
/// The arrays are evaluated as part of writing, so a lazy graph does not need
/// [`Array::eval`] first. An existing file at `path` is overwritten. No
/// `__metadata__` header is written.
///
/// # Errors
/// Returns an error if the path cannot be opened for writing.
pub fn save_safetensors(path: impl AsRef<Path>, arrays: &HashMap<String, Array>) -> Result<()> {
    error::install();
    let path = path.as_ref();
    let file = c_path(path)?;

    let mut map = ArrayMap::new();
    for (name, array) in arrays {
        map.insert(name, array)?;
    }
    let metadata = StringMap::new();
    // SAFETY: `file` is NUL-terminated; both containers outlive the call.
    let status =
        unsafe { sys::mlx_save_safetensors(file.as_ptr(), map.as_raw(), metadata.as_raw()) };
    error::check(status)
}

/// Converts a path into the NUL-terminated string mlx-c takes.
///
/// Paths are bytes on macOS, so this does not go through `str` and non-UTF-8
/// names work. Only an interior NUL is rejected, which no real path has.
fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        Error::new(format!(
            "path contains an interior NUL byte: {}",
            path.display()
        ))
    })
}

/// An owned `mlx_map_string_to_array`.
struct ArrayMap {
    handle: sys::mlx_map_string_to_array,
}

impl ArrayMap {
    fn new() -> Self {
        // SAFETY: allocates a fresh empty map, whose ownership moves into `self`.
        Self {
            handle: unsafe { sys::mlx_map_string_to_array_new() },
        }
    }

    fn as_raw(&self) -> sys::mlx_map_string_to_array {
        self.handle
    }

    fn as_mut_ptr(&mut self) -> *mut sys::mlx_map_string_to_array {
        &mut self.handle
    }

    /// Adds `array` under `name`, copying the handle in.
    fn insert(&mut self, name: &str, array: &Array) -> Result<()> {
        let key = CString::new(name)
            .map_err(|_| Error::new(format!("tensor name contains a NUL byte: {name:?}")))?;
        // SAFETY: `key` is NUL-terminated and outlives the call; the map stores a
        // copy of the handle, sharing the buffer rather than taking it over, so
        // `array` stays the owner.
        let status = unsafe {
            sys::mlx_map_string_to_array_insert(self.handle, key.as_ptr(), array.as_raw())
        };
        error::check(status)
    }

    /// Copies every entry out into an owned Rust map.
    fn to_hash_map(&self) -> Result<HashMap<String, Array>> {
        let iter = ArrayMapIter::new(self.handle);
        let mut out = HashMap::new();
        loop {
            let mut key: *const c_char = ptr::null();
            let mut value = unsafe { sys::mlx_array_new() };
            // SAFETY: the iterator and map are alive; on success mlx writes a
            // borrowed key pointer and a handle sharing the value's buffer.
            let status = unsafe {
                sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value, iter.handle)
            };

            if status != 0 {
                // SAFETY: `value` was allocated above and mlx did not write to
                // it, so it is ours to free.
                unsafe { sys::mlx_array_free(value) };
                // mlx-c signals "past the last entry" with 2, reserving 1 for a
                // real failure — so this is not something `error::check` can be
                // handed directly.
                if status == 2 {
                    return Ok(out);
                }
                error::check(status)?;
            }

            if key.is_null() {
                unsafe { sys::mlx_array_free(value) };
                return Err(Error::new("safetensors entry has no name"));
            }
            // SAFETY: mlx points `key` at the map's own NUL-terminated string,
            // valid while the map lives; the name is copied out here.
            let name = unsafe { CStr::from_ptr(key) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: `value` holds a handle mlx wrote and nothing else owns.
            out.insert(name, unsafe { Array::from_raw(value) });
        }
    }
}

impl Drop for ArrayMap {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe { sys::mlx_map_string_to_array_free(self.handle) };
    }
}

/// An owned iterator over an [`ArrayMap`].
///
/// Borrows the map it walks; mlx keeps a raw pointer to the underlying container,
/// so the map must outlive this.
struct ArrayMapIter {
    handle: sys::mlx_map_string_to_array_iterator,
}

impl ArrayMapIter {
    fn new(map: sys::mlx_map_string_to_array) -> Self {
        // SAFETY: `map` is a valid handle; the iterator it returns is owned here.
        Self {
            handle: unsafe { sys::mlx_map_string_to_array_iterator_new(map) },
        }
    }
}

impl Drop for ArrayMapIter {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe { sys::mlx_map_string_to_array_iterator_free(self.handle) };
    }
}

/// An owned `mlx_map_string_to_string`, used only to hold the metadata header.
struct StringMap {
    handle: sys::mlx_map_string_to_string,
}

impl StringMap {
    fn new() -> Self {
        // SAFETY: allocates a fresh empty map, whose ownership moves into `self`.
        Self {
            handle: unsafe { sys::mlx_map_string_to_string_new() },
        }
    }

    fn as_raw(&self) -> sys::mlx_map_string_to_string {
        self.handle
    }

    fn as_mut_ptr(&mut self) -> *mut sys::mlx_map_string_to_string {
        &mut self.handle
    }
}

impl Drop for StringMap {
    fn drop(&mut self) {
        // SAFETY: `handle` was created by mlx and is owned solely by `self`.
        unsafe { sys::mlx_map_string_to_string_free(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dtype;

    /// A unique path under the temp directory, removed when the guard drops.
    struct TempFile {
        path: std::path::PathBuf,
    }

    impl TempFile {
        fn new(name: &str) -> Self {
            // Thread id keeps concurrent test binaries from colliding; the tests
            // themselves run serialized (see the Makefile).
            let unique = format!("mlxcore-{}-{name}.safetensors", std::process::id());
            Self {
                path: std::env::temp_dir().join(unique),
            }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn round_trips_named_arrays() {
        let s = Stream::cpu();
        let file = TempFile::new("round-trip");

        let mut weights = HashMap::new();
        weights.insert(
            "encoder.weight".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]),
        );
        weights.insert(
            "encoder.bias".to_string(),
            Array::from_slice(&[5.0f32], &[1]),
        );
        save_safetensors(&file.path, &weights).unwrap();

        let loaded = load_safetensors(&file.path, &s).unwrap();

        assert_eq!(loaded.len(), 2);
        let weight = &loaded["encoder.weight"];
        assert_eq!(weight.shape(), vec![2, 2]);
        assert_eq!(weight.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded["encoder.bias"].to_vec::<f32>(), vec![5.0]);
    }

    #[test]
    fn preserves_half_precision_dtype() {
        let s = Stream::cpu();
        let file = TempFile::new("half");

        // The dtype checkpoints actually ship in. There is no Rust `f16`, so the
        // array is built in f32 and cast — and must come back as float16.
        let half = Array::from_slice(&[1.0f32, 2.0], &[2])
            .astype_dtype(Dtype::Float16, &s)
            .unwrap();
        let mut weights = HashMap::new();
        weights.insert("half".to_string(), half);
        save_safetensors(&file.path, &weights).unwrap();

        let loaded = load_safetensors(&file.path, &s).unwrap();

        let array = &loaded["half"];
        assert_eq!(array.dtype(), Dtype::Float16);
        // Unreadable as f16, but castable back to something Rust can see.
        let recovered = array.astype::<f32>(&s).unwrap();
        assert_eq!(recovered.to_vec::<f32>(), vec![1.0, 2.0]);
    }

    #[test]
    fn empty_map_round_trips() {
        let s = Stream::cpu();
        let file = TempFile::new("empty");

        save_safetensors(&file.path, &HashMap::new()).unwrap();

        // Exercises the iterator's very first `next` returning end-of-iteration.
        assert!(load_safetensors(&file.path, &s).unwrap().is_empty());
    }

    #[test]
    fn missing_file_is_an_error() {
        let s = Stream::cpu();
        let missing = std::env::temp_dir().join("mlxcore-does-not-exist.safetensors");

        let err = load_safetensors(&missing, &s).unwrap_err();

        assert!(!err.message().is_empty());
    }
}
