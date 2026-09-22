//! One transformer attention block, the way an encoder model builds it.
//!
//! A fused-QKV projection, rotary position embeddings, sliding-window masked
//! attention, and an output projection — the shape of a ModernBERT layer. This
//! is the composition [`mlxcore::fast`] exists for; everything here is a real op,
//! not a sketch.
//!
//! Run with:
//! ```sh
//! cargo run --example attention
//! ```

use mlxcore::fast::{self, Mask};
use mlxcore::{Array, Result, Stream, random};

const BATCH: i32 = 1;
const LENGTH: i32 = 6;
const HIDDEN: i32 = 8;
const HEADS: i32 = 2;
const HEAD_DIM: i32 = HIDDEN / HEADS;

/// Local attention window, as a total width. Each query sees keys within half of
/// it on either side, which is how ModernBERT keeps most of its layers cheap.
const WINDOW: i32 = 4;

fn main() -> Result<()> {
    // The CPU stream keeps this reproducible and lets float64 through; a real
    // model would use `Stream::default()` for the GPU.
    let stream = Stream::cpu();

    let key = random::Key::new(0)?;
    let (x_key, w_key) = key.split(&stream)?;
    let x = random::normal::<f32>(&[BATCH, LENGTH, HIDDEN], 0.0, 1.0, Some(&x_key), &stream)?;

    // Checkpoints store a linear layer's weight as (out_features, in_features)
    // and compute `x @ w.T`, so every projection swaps the last two axes.
    let (qkv_key, out_key) = w_key.split(&stream)?;
    let w_qkv = random::normal::<f32>(&[3 * HIDDEN, HIDDEN], 0.0, 0.1, Some(&qkv_key), &stream)?;
    let w_out = random::normal::<f32>(&[HIDDEN, HIDDEN], 0.0, 0.1, Some(&out_key), &stream)?;

    // --- 1. Fused QKV projection ------------------------------------------
    // One matmul for all three tensors, then split by reshaping the output into
    // an explicit axis of 3 rather than slicing it apart.
    let qkv = x
        .matmul(&w_qkv.swapaxes(-2, -1, &stream)?, &stream)?
        .reshape(&[BATCH, LENGTH, 3, HEADS, HEAD_DIM], &stream)?;

    // `take_axis` with a 0-d index drops the axis it selects on, so each of these
    // is (batch, length, heads, head_dim). Transposing brings `heads` forward,
    // which is the layout the attention kernel requires.
    let project = |i: i32| -> Result<Array> {
        qkv.take_axis(&Array::from_scalar(i), 2, &stream)?
            .transpose_axes(&[0, 2, 1, 3], &stream)
    };
    let (q, k, v) = (project(0)?, project(1)?, project(2)?);
    println!("q/k/v: {:?} (batch, heads, length, head_dim)", q.shape());

    // --- 2. Rotary position embeddings ------------------------------------
    // Applied to queries and keys only — values carry no position. `false` is the
    // split-halves layout every Hugging Face checkpoint uses.
    let rope = |a: &Array| fast::rope(a, HEAD_DIM, false, Some(10_000.0), 1.0, 0, None, &stream);
    let (q, k) = (rope(&q)?, rope(&k)?);

    // --- 3. The sliding-window mask ---------------------------------------
    // `|i - j| <= window / 2`, built by broadcasting a position vector against
    // its own transpose. A bool mask drops the false positions.
    let positions = Array::arange::<i32>(0.0, LENGTH as f64, 1.0, &stream)?;
    let distance = positions
        .expand_dims(1, &stream)?
        .subtract(&positions.expand_dims(0, &stream)?, &stream)?
        .abs(&stream)?;
    let mask = distance
        .less_equal(&Array::from_scalar(WINDOW / 2), &stream)?
        // (length, length) -> (batch, heads, length, length) by broadcasting.
        .expand_dims(0, &stream)?
        .expand_dims(0, &stream)?;

    // --- 4. Attention ------------------------------------------------------
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let attended =
        fast::scaled_dot_product_attention(&q, &k, &v, scale, Mask::Array(&mask), &stream)?;

    // --- 5. Merge the heads and project out --------------------------------
    // Undo the transpose from step 1, then collapse (heads, head_dim) back into
    // one hidden axis. `contiguous` because `reshape` needs dense input and the
    // transpose left this strided.
    let merged = attended
        .transpose_axes(&[0, 2, 1, 3], &stream)?
        .contiguous(&stream)?
        .reshape(&[BATCH, LENGTH, HIDDEN], &stream)?;
    let y = merged.matmul(&w_out.swapaxes(-2, -1, &stream)?, &stream)?;

    // A residual connection and a layer norm, the rest of the block. The norm has
    // no affine parameters here, so both optional operands are `None`.
    let normed = fast::layer_norm(&x.add(&y, &stream)?, None, None, 1e-5, &stream)?;
    normed.eval();

    println!("attention out: {:?}", y.shape());
    println!("block out:     {:?}", normed.shape());

    // Every row is normalized, so each one has mean 0 and unit variance.
    let means = normed.mean_axes(&[-1], false, &stream)?;
    println!("row means:     {:?}", means.to_vec::<f32>());

    // Proof the mask did something: position 0 and position 5 are more than
    // `WINDOW / 2` apart, so neither attends to the other. Widening the window to
    // cover the whole sequence changes the answer.
    let full = fast::scaled_dot_product_attention(&q, &k, &v, scale, Mask::None, &stream)?;
    let drift = full
        .subtract(&attended, &stream)?
        .abs(&stream)?
        .max(false, &stream)?
        .item::<f32>();
    println!("windowed vs. full attention, max difference: {drift:.4}");

    Ok(())
}
