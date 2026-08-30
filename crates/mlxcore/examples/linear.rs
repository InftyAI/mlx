//! A linear layer forward pass: `y = relu(x @ W + b)`.
//!
//! Run with:
//! ```sh
//! cargo run --example linear
//! ```

use mlxcore::{Array, Stream, random};

fn main() -> mlxcore::Result<()> {
    let stream = Stream::default();

    // 4 samples, 3 features each. Note the `f32` suffix: an unsuffixed literal
    // is `f64` in Rust, and the default stream here is the GPU, which has no
    // float64 support.
    let x = Array::from_slice(
        &[
            1.0f32, 2.0, 3.0, // sample 0
            -1.0, 0.5, 2.0, // sample 1
            0.0, 0.0, 1.0, // sample 2
            2.0, -3.0, 0.5, // sample 3
        ],
        &[4, 3],
    );

    // A fixed seed, split into two independent keys, so the weights are random
    // but this example prints the same numbers on every run.
    let key = random::Key::new(42)?;
    let (w_key, b_key) = key.split(&stream)?;
    let w = random::normal::<f32>(&[3, 2], 0.0, 1.0, Some(&w_key), &stream)?;
    let b = random::normal::<f32>(&[2], 0.0, 1.0, Some(&b_key), &stream)?;

    // `matmul` contracts the last axis of `x` against the first of `w`, so
    // (4, 3) @ (3, 2) is (4, 2) — one row of 2 outputs per sample. This is not
    // `*`, which is elementwise and would reject these shapes.
    let hidden = x.matmul(&w, &stream)?;

    // `b` is (2,) and `hidden` is (4, 2). Broadcasting stretches `b` across the
    // rows, adding the same bias to every sample without an explicit
    // `broadcast_to`.
    let biased = hidden.add(&b, &stream)?;

    // ReLU. There is no activation op, but clamping at zero is just an
    // elementwise maximum against a 0-d scalar array, which broadcasts.
    let y = biased.maximum(&Array::from_scalar(0.0f32), &stream)?;

    // Nothing above has computed yet: MLX is lazy, and each call only extended a
    // graph. This is the point where the work actually runs.
    y.eval();

    println!("x:      shape {:?}, dtype {}", x.shape(), x.dtype());
    println!("W:      shape {:?}, dtype {}", w.shape(), w.dtype());
    println!("b:      shape {:?}\n{b:?}", b.shape());
    println!("x @ W + b:\n{biased:?}");
    println!("relu(x @ W + b):\n{y:?}");

    // Copy out of MLX into plain Rust values. `to_vec` is row-major, so the
    // outputs for sample `i` are at `2 * i` and `2 * i + 1`.
    let out = y.to_vec::<f32>();
    for (i, row) in out.chunks(2).enumerate() {
        println!("sample {i}: {row:?}");
    }

    // Every ReLU output is non-negative, and the mean is a 0-d array that
    // `item` reads as a single `f32`.
    let negatives = y
        .less(&Array::from_scalar(0.0f32), &stream)?
        .any(false, &stream)?;
    println!("any negative output: {}", negatives.item::<bool>());
    println!("mean activation: {}", y.mean(false, &stream)?.item::<f32>());

    Ok(())
}
