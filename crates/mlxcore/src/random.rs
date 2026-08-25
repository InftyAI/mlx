//! Random number generation.
//!
//! MLX's RNG is *functional*: a draw is a pure function of an explicit [`Key`],
//! so the same key always produces the same values. Passing `None` instead
//! draws from MLX's implicit global state, which [`seed`] initializes.
//!
//! Prefer an explicit key when reproducibility matters. The global state is
//! process-wide, so two threads (or two `#[test]`s running in parallel) that
//! draw without a key will interleave and neither is reproducible on its own.

use mlxcore_sys as sys;

use crate::array::Array;
use crate::dtype::ArrayElement;
use crate::error::{self, Result};
use crate::ffi::{absent_array, as_ffi_ptr};
use crate::stream::Stream;

/// A key identifying a position in MLX's random stream.
///
/// Draws made with the same key are identical. To get independent draws, either
/// build keys from different seeds or [`split`](Self::split) an existing key.
#[derive(Debug)]
pub struct Key(Array);

impl Key {
    /// Builds a key from a seed.
    ///
    /// Deterministic: the same seed always yields the same key, and therefore
    /// the same draws.
    pub fn new(seed: u64) -> Result<Self> {
        error::install();
        let mut out = unsafe { sys::mlx_array_new() };
        // SAFETY: the result is written into `out`. This takes no stream.
        let status = unsafe { sys::mlx_random_key(&mut out, seed) };
        Array::from_op(out, status).map(Key)
    }

    /// Splits this key into two independent keys.
    ///
    /// Use this to draw more than once from a single seed without reusing a key
    /// (which would give identical values). Splitting is itself deterministic:
    /// the same key always splits into the same pair.
    pub fn split(&self, stream: &Stream) -> Result<(Key, Key)> {
        error::install();
        let mut first = unsafe { sys::mlx_array_new() };
        let mut second = unsafe { sys::mlx_array_new() };
        // SAFETY: both out-params are fresh handles; the key and stream are
        // valid; mlx writes one key into each.
        let status = unsafe {
            sys::mlx_random_split(&mut first, &mut second, self.0.as_raw(), stream.as_raw())
        };
        match error::check(status) {
            // SAFETY: on success both handles are owned by us.
            Ok(()) => Ok(unsafe { (Key(Array::from_raw(first)), Key(Array::from_raw(second))) }),
            Err(e) => {
                // SAFETY: both handles were created by mlx and are owned here;
                // free them so the failed call doesn't leak.
                unsafe {
                    sys::mlx_array_free(first);
                    sys::mlx_array_free(second);
                }
                Err(e)
            }
        }
    }

    /// The raw handle for an optional key argument, or an absent handle.
    fn handle(key: Option<&Key>) -> sys::mlx_array {
        match key {
            Some(k) => k.0.as_raw(),
            None => absent_array(),
        }
    }
}

/// Seeds MLX's implicit global random state.
///
/// Only affects draws made with `key: None`. Because the state is process-wide
/// and mutated by every unkeyed draw, this does not make a draw reproducible in
/// the presence of other threads — use a [`Key`] for that.
pub fn seed(seed: u64) -> Result<()> {
    error::install();
    // SAFETY: takes a plain integer and mutates global mlx state.
    let status = unsafe { sys::mlx_random_seed(seed) };
    error::check(status)
}

/// Samples a normal (Gaussian) distribution with mean `loc` and standard
/// deviation `scale`.
///
/// `loc` and `scale` are `f32` because MLX's parameters are, regardless of the
/// dtype chosen by `T`. Only floating-point dtypes can be generated, so
/// `normal::<i32>` returns an error rather than truncating.
pub fn normal<T: ArrayElement>(
    shape: &[i32],
    loc: f32,
    scale: f32,
    key: Option<&Key>,
    stream: &Stream,
) -> Result<Array> {
    error::install();
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: `shape` describes a valid slice (or null/0); the key handle is
    // either a valid array or an absent one, both of which mlx accepts; the
    // result is written into `out`.
    let status = unsafe {
        sys::mlx_random_normal(
            &mut out,
            as_ffi_ptr(shape),
            shape.len(),
            T::DTYPE.as_raw(),
            loc,
            scale,
            Key::handle(key),
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

/// Samples a uniform distribution over `[low, high)`.
///
/// Unlike [`normal`], MLX takes the bounds as arrays so they can broadcast; this
/// wrapper accepts scalars of the element type and wraps them internally.
pub fn uniform<T: ArrayElement>(
    shape: &[i32],
    low: T,
    high: T,
    key: Option<&Key>,
    stream: &Stream,
) -> Result<Array> {
    error::install();
    // Held in locals so they outlive the call below.
    let low = Array::from_scalar(low);
    let high = Array::from_scalar(high);
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: the bound arrays outlive the call; `shape` describes a valid slice
    // (or null/0); the result is written into `out`.
    let status = unsafe {
        sys::mlx_random_uniform(
            &mut out,
            low.as_raw(),
            high.as_raw(),
            as_ffi_ptr(shape),
            shape.len(),
            T::DTYPE.as_raw(),
            Key::handle(key),
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_gives_identical_draws() {
        let s = Stream::cpu();
        let k1 = Key::new(42).unwrap();
        let k2 = Key::new(42).unwrap();
        let a = normal::<f32>(&[4], 0.0, 1.0, Some(&k1), &s).unwrap();
        let b = normal::<f32>(&[4], 0.0, 1.0, Some(&k2), &s).unwrap();
        assert_eq!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn different_seeds_give_different_draws() {
        let s = Stream::cpu();
        let a = normal::<f32>(&[8], 0.0, 1.0, Some(&Key::new(1).unwrap()), &s).unwrap();
        let b = normal::<f32>(&[8], 0.0, 1.0, Some(&Key::new(2).unwrap()), &s).unwrap();
        assert_ne!(a.to_vec::<f32>(), b.to_vec::<f32>());
    }

    #[test]
    fn normal_respects_shape_and_scale() {
        let s = Stream::cpu();
        let key = Key::new(7).unwrap();
        let a = normal::<f32>(&[2, 3], 0.0, 1.0, Some(&key), &s).unwrap();
        assert_eq!(a.shape(), vec![2, 3]);
        assert_eq!(a.size(), 6);

        // A zero scale collapses the distribution onto `loc` exactly.
        let fixed = normal::<f32>(&[4], 5.0, 0.0, Some(&key), &s).unwrap();
        assert_eq!(fixed.to_vec::<f32>(), vec![5.0; 4]);
    }

    #[test]
    fn uniform_stays_within_bounds() {
        let s = Stream::cpu();
        let key = Key::new(3).unwrap();
        let a = uniform::<f32>(&[64], -2.0, 2.0, Some(&key), &s).unwrap();
        for v in a.to_vec::<f32>() {
            assert!((-2.0..2.0).contains(&v), "{v} out of bounds");
        }
    }

    #[test]
    fn split_yields_independent_but_deterministic_keys() {
        let s = Stream::cpu();
        let (a, b) = Key::new(11).unwrap().split(&s).unwrap();
        let from_a = normal::<f32>(&[8], 0.0, 1.0, Some(&a), &s).unwrap();
        let from_b = normal::<f32>(&[8], 0.0, 1.0, Some(&b), &s).unwrap();
        // The halves are independent of each other...
        assert_ne!(from_a.to_vec::<f32>(), from_b.to_vec::<f32>());

        // ...but splitting the same seed again reproduces the same pair.
        let (a2, _) = Key::new(11).unwrap().split(&s).unwrap();
        let again = normal::<f32>(&[8], 0.0, 1.0, Some(&a2), &s).unwrap();
        assert_eq!(from_a.to_vec::<f32>(), again.to_vec::<f32>());
    }

    #[test]
    fn unkeyed_draws_use_global_state() {
        let s = Stream::cpu();
        seed(99).unwrap();
        // Deliberately only a structural assertion: the global state is shared
        // with every other test running in parallel, so the *values* here are
        // not reproducible. `Key` is the tool for that.
        let a = normal::<f32>(&[3], 0.0, 1.0, None, &s).unwrap();
        assert_eq!(a.shape(), vec![3]);
    }

    #[test]
    fn integer_dtype_is_rejected() {
        let s = Stream::cpu();
        let key = Key::new(1).unwrap();
        // Normal draws are floating-point only; mlx reports this rather than
        // silently truncating.
        let err = normal::<i32>(&[4], 0.0, 1.0, Some(&key), &s).unwrap_err();
        assert!(
            !err.message().is_empty(),
            "expected a message from mlx, got {err:?}"
        );
    }
}
