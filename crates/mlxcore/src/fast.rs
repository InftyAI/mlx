//! Fused kernels from MLX's `fast` namespace.
//!
//! Each of these is expressible with the ops on [`Array`], but MLX ships a
//! single fused kernel that is faster and, for the normalizations, better
//! behaved numerically. They are also exactly the pieces a transformer needs, so
//! a model built on this crate should reach for these rather than composing them
//! by hand.
//!
//! These are stateless kernels: the parameters are arguments, not fields. The
//! layers that own those parameters belong a level up.

use mlxcore_sys as sys;

use crate::array::Array;
use crate::error::{self, Result};
use crate::ffi::absent_array;
use crate::stream::Stream;

/// How [`scaled_dot_product_attention`] should mask its attention scores.
///
/// An enum rather than the string-plus-optional-array pair mlx-c takes, so an
/// array mask cannot be requested without supplying one.
pub enum Mask<'a> {
    /// Every query attends to every key.
    None,
    /// Each query attends only to keys at or before its own position.
    ///
    /// The decoder mask. MLX builds it inside the kernel, so nothing is
    /// materialized.
    Causal,
    /// An explicit mask, broadcast against `(batch, heads, queries, keys)`.
    ///
    /// A `bool` array masks out the `false` positions; any other dtype is *added*
    /// to the scores, which is how an additive bias is applied. Rank must be 4
    /// or less.
    ///
    /// This is what a bidirectional encoder needs: padding masks, and
    /// sliding-window masks that are not simply causal.
    Array(&'a Array),
}

impl Mask<'_> {
    /// The `(mask_mode, mask_arr)` pair mlx-c expects.
    fn as_ffi(&self) -> (&'static [u8], sys::mlx_array) {
        match self {
            // MLX validates the mode string and accepts only "", "causal", and
            // "array". Written as NUL-terminated byte literals so no `CString`
            // allocation is needed to cross the boundary.
            Mask::None => (b"\0", absent_array()),
            Mask::Causal => (b"causal\0", absent_array()),
            Mask::Array(mask) => (b"array\0", mask.as_raw()),
        }
    }
}

/// Layer normalization over the last axis.
///
/// Rescales each row to zero mean and unit variance, then applies the optional
/// affine `weight` and `bias`. Both must be 1-dimensional and match the last
/// axis of `x`; omitting them gives the bare normalization. `eps` is added to
/// the variance before the reciprocal square root, so it must be positive.
pub fn layer_norm(
    x: &Array,
    weight: Option<&Array>,
    bias: Option<&Array>,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    error::install();
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: all handles are valid; the optional operands are passed as null
    // handles, which the C shim turns into `std::nullopt`.
    let status = unsafe {
        sys::mlx_fast_layer_norm(
            &mut out,
            x.as_raw(),
            weight.map_or_else(absent_array, Array::as_raw),
            bias.map_or_else(absent_array, Array::as_raw),
            eps,
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

/// Root-mean-square normalization over the last axis.
///
/// Divides each row by its RMS without centering it first — the normalization
/// most recent decoder models use in place of [`layer_norm`]. `weight` is the
/// optional 1-dimensional scale; there is no bias.
pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32, stream: &Stream) -> Result<Array> {
    error::install();
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: all handles are valid; an omitted `weight` is a null handle.
    let status = unsafe {
        sys::mlx_fast_rms_norm(
            &mut out,
            x.as_raw(),
            weight.map_or_else(absent_array, Array::as_raw),
            eps,
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

/// Rotary position embedding, applied to the last axis of `x`.
///
/// Rotates the first `dims` features of each position by an angle that grows
/// with the position, which is how attention learns relative distances without a
/// position embedding table. `dims` must be even and no larger than the last
/// axis; features past it pass through untouched.
///
/// - `traditional` interleaves the rotated pairs as in the original RoPE paper.
///   Hugging Face checkpoints — and so nearly every published model — use the
///   split-halves layout, which is `false`.
/// - `base` is the geometric base for the frequencies: often `10000.0`, and much
///   larger for long-context models. Pass `None` only together with `freqs`.
/// - `scale` divides the positions, the usual knob for stretching a model past
///   the context it was trained on. `1.0` leaves them alone.
/// - `offset` is the position of `x`'s first element. Nonzero when decoding one
///   token at a time against a cache; `0` for a whole sequence at once.
/// - `freqs` supplies per-dimension frequencies directly, for the scaling
///   schemes `base` alone cannot express.
// Eight parameters, because that is what the kernel takes — MLX's own Python and
// Swift bindings have the same arity. Bundling them into a config struct would
// put a layer between this crate and mlx-c for no gain.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    x: &Array,
    dims: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    freqs: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    error::install();
    let base = sys::mlx_optional_float {
        value: base.unwrap_or_default(),
        has_value: base.is_some(),
    };
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: all handles are valid; an omitted `freqs` is a null handle.
    let status = unsafe {
        sys::mlx_fast_rope(
            &mut out,
            x.as_raw(),
            dims,
            traditional,
            base,
            scale,
            offset,
            freqs.map_or_else(absent_array, Array::as_raw),
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

/// Scaled dot-product attention: `softmax(scale * q @ k.T + mask) @ v`.
///
/// All three operands must be rank 4, shaped `(batch, heads, length, head_dim)`
/// — the layout [`Array::transpose_axes`] produces from a
/// `(batch, length, heads, head_dim)` projection. `queries` and `keys` must agree
/// on `head_dim`, and `keys` and `values` on `length`. Grouped-query attention
/// works: `keys` and `values` may carry fewer heads than `queries`, as long as
/// the count divides evenly.
///
/// `scale` multiplies the scores before the softmax, conventionally
/// `1.0 / (head_dim as f32).sqrt()`.
///
/// The result is `(batch, heads, length, head_dim)`, so it needs transposing
/// back before the output projection.
///
/// MLX's attention-sink parameter is not exposed.
pub fn scaled_dot_product_attention(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: Mask<'_>,
    stream: &Stream,
) -> Result<Array> {
    error::install();
    let (mode, mask_array) = mask.as_ffi();
    let mut out = unsafe { sys::mlx_array_new() };
    // SAFETY: all handles are valid; `mode` is a NUL-terminated literal with
    // 'static lifetime, and `mask_array` is null unless `Mask::Array` was given.
    let status = unsafe {
        sys::mlx_fast_scaled_dot_product_attention(
            &mut out,
            queries.as_raw(),
            keys.as_raw(),
            values.as_raw(),
            scale,
            mode.as_ptr().cast::<std::ffi::c_char>(),
            mask_array,
            absent_array(),
            stream.as_raw(),
        )
    };
    Array::from_op(out, status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_centers_and_scales() {
        let s = Stream::cpu();
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);

        let y = layer_norm(&x, None, None, 1e-5, &s).unwrap();

        // Each row becomes (-1, 1): mean 0, unit variance, up to `eps`.
        let out = y.to_vec::<f32>();
        assert_eq!(out.len(), 4);
        for pair in out.chunks(2) {
            assert!((pair[0] + 1.0).abs() < 1e-3, "{pair:?}");
            assert!((pair[1] - 1.0).abs() < 1e-3, "{pair:?}");
        }
    }

    #[test]
    fn layer_norm_applies_affine() {
        let s = Stream::cpu();
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let weight = Array::from_slice(&[2.0f32, 2.0], &[2]);
        let bias = Array::from_slice(&[1.0f32, 1.0], &[2]);

        let y = layer_norm(&x, Some(&weight), Some(&bias), 1e-5, &s).unwrap();

        // (-1, 1) * 2 + 1.
        let out = y.to_vec::<f32>();
        assert!((out[0] + 1.0).abs() < 1e-3, "{out:?}");
        assert!((out[1] - 3.0).abs() < 1e-3, "{out:?}");
    }

    #[test]
    fn rms_norm_divides_by_root_mean_square() {
        let s = Stream::cpu();
        // The RMS of (3, 4) is sqrt((9 + 16) / 2) = 3.5355, so the row scales to
        // (0.8485, 1.1314). A `layer_norm` would have centered it to (-1, 1).
        let x = Array::from_slice(&[3.0f32, 4.0], &[1, 2]);

        let y = rms_norm(&x, None, 1e-5, &s).unwrap();

        let out = y.to_vec::<f32>();
        assert!((out[0] - 0.8485).abs() < 1e-3, "{out:?}");
        assert!((out[1] - 1.1314).abs() < 1e-3, "{out:?}");
    }

    #[test]
    fn rope_leaves_position_zero_alone() {
        let s = Stream::cpu();
        // (batch, heads, length, head_dim). The rotation angle at position 0 is
        // zero, so the first position passes through while the second does not.
        let x = Array::ones::<f32>(&[1, 1, 2, 4], &s).unwrap();

        let y = rope(&x, 4, false, Some(10000.0), 1.0, 0, None, &s).unwrap();

        assert_eq!(y.shape(), vec![1, 1, 2, 4]);
        let out = y.to_vec::<f32>();
        assert_eq!(&out[..4], &[1.0; 4]);
        assert!(out[4..].iter().any(|v| (v - 1.0).abs() > 1e-3), "{out:?}");
    }

    #[test]
    fn rope_offset_shifts_positions() {
        let s = Stream::cpu();
        let x = Array::ones::<f32>(&[1, 1, 2, 4], &s).unwrap();

        // With `offset = 1` the first element is position 1, not 0, so nothing is
        // left unrotated.
        let y = rope(&x, 4, false, Some(10000.0), 1.0, 1, None, &s).unwrap();

        let out = y.to_vec::<f32>();
        assert!(out[..4].iter().any(|v| (v - 1.0).abs() > 1e-3), "{out:?}");
    }

    #[test]
    fn attention_with_one_key_returns_that_value() {
        let s = Stream::cpu();
        // A single key means a softmax over one score, which is 1 whatever the
        // scale, so the output is exactly `values`.
        let q = Array::ones::<f32>(&[1, 1, 1, 4], &s).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 1, 4], &s).unwrap();
        let v = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);

        let out = scaled_dot_product_attention(&q, &k, &v, 0.5, Mask::None, &s).unwrap();

        assert_eq!(out.shape(), vec![1, 1, 1, 4]);
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn attention_mask_excludes_keys() {
        let s = Stream::cpu();
        let q = Array::ones::<f32>(&[1, 1, 1, 2], &s).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 2], &s).unwrap();
        let v = Array::from_slice(&[1.0f32, 1.0, 5.0, 5.0], &[1, 1, 2, 2]);

        // Unmasked, the two identical keys score the same and the output is the
        // average of the values. Masking the second key leaves only the first.
        let even = scaled_dot_product_attention(&q, &k, &v, 0.5, Mask::None, &s).unwrap();
        assert_eq!(even.to_vec::<f32>(), vec![3.0, 3.0]);

        let mask = Array::from_slice(&[true, false], &[1, 1, 1, 2]);
        let masked = scaled_dot_product_attention(&q, &k, &v, 0.5, Mask::Array(&mask), &s).unwrap();
        assert_eq!(masked.to_vec::<f32>(), vec![1.0, 1.0]);
    }

    #[test]
    fn causal_attention_hides_later_keys() {
        let s = Stream::cpu();
        let q = Array::ones::<f32>(&[1, 1, 2, 2], &s).unwrap();
        let k = Array::ones::<f32>(&[1, 1, 2, 2], &s).unwrap();
        let v = Array::from_slice(&[1.0f32, 1.0, 5.0, 5.0], &[1, 1, 2, 2]);

        let out = scaled_dot_product_attention(&q, &k, &v, 0.5, Mask::Causal, &s).unwrap();

        // Query 0 sees only value 0; query 1 sees both and averages them.
        assert_eq!(out.to_vec::<f32>(), vec![1.0, 1.0, 3.0, 3.0]);
    }

    #[test]
    fn attention_rejects_low_rank_inputs() {
        let s = Stream::cpu();
        let a = Array::ones::<f32>(&[2, 2], &s).unwrap();

        let err = scaled_dot_product_attention(&a, &a, &a, 1.0, Mask::None, &s).unwrap_err();

        assert!(err.message().contains("rank 4"), "{err}");
    }
}
