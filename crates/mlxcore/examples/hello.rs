//! A tour of the `mlxcore` API: constructors, arithmetic, reductions, shape
//! manipulation, comparisons, and stream control.
//!
//! For a worked end-to-end computation, see `examples/relu.rs`.
//!
//! Run with:
//! ```sh
//! cargo run --example hello
//! ```

use mlxcore::{Array, Dtype, Stream};

fn main() -> mlxcore::Result<()> {
    println!("MLX version: {}\n", mlxcore::version());

    // MLX's own default stream: the GPU on Apple Silicon. Methods take a stream
    // explicitly; the operators (`&a + &b`) resolve this default themselves.
    let s = Stream::default();

    // --- Constructing arrays -------------------------------------------------

    // Data plus a shape. The `f32` suffix matters: an unsuffixed float literal
    // is `f64` in Rust, and the GPU has no float64 support.
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("from_slice:\n{a:?}");

    // The element type picks the dtype, so `&[i32]` gives an int32 array.
    println!("ints: {:?}", Array::from_slice(&[1i32, 2, 3], &[3]));

    // Filled and generated arrays. The turbofish supplies the dtype.
    println!("zeros: {:?}", Array::zeros::<f32>(&[2, 2], &s)?);
    println!("ones:  {:?}", Array::ones::<f32>(&[3], &s)?);
    println!("full:  {:?}", Array::full::<f32>(&[3], 7.0, &s)?);
    // Half-open like Rust ranges: 0, 2, 4, 6, 8 but not 10.
    println!("arange: {:?}", Array::arange::<i32>(0.0, 10.0, 2.0, &s)?);
    // A 0-dimensional array, which broadcasts against anything.
    println!("scalar: {:?}\n", Array::from_scalar(42.0f32));

    // --- Inspecting ----------------------------------------------------------

    println!("shape: {:?}", a.shape());
    println!("size:  {}", a.size());
    println!("ndim:  {}", a.ndim());
    // MLX decides the dtype: ops promote, and comparisons always give bool.
    println!("dtype: {}\n", a.dtype());

    // --- Arithmetic ----------------------------------------------------------

    let b = Array::from_slice(&[6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]);

    // Operators borrow, so both operands stay usable afterwards. They run on the
    // default stream and panic on failure; the methods return `Result` instead.
    println!("a + b:\n{:?}", &a + &b);
    println!("a * b:\n{:?}", &a * &b);
    println!("-a:\n{:?}", -&a);

    // Scalars work on either side, and become 0-d arrays that broadcast.
    println!("a * 2:\n{:?}", &a * 2.0f32);
    println!("10 - a:\n{:?}", 10.0f32 - &a);

    // Assigning operators rebind rather than mutate: MLX has no in-place ops, so
    // `acc += &b` is `acc = &acc + &b`. Note `acc` must be owned and `mut`.
    let mut acc = Array::zeros::<f32>(&[2, 3], &s)?;
    acc += &a;
    acc += &b;
    acc /= 2.0f32;
    println!("mean of a and b:\n{acc:?}");

    // Elementwise math, one method per op.
    let x = Array::from_slice(&[1.0f32, 4.0, 9.0], &[3]);
    println!("sqrt: {:?}", x.sqrt(&s)?);
    println!("exp:  {:?}", x.exp(&s)?);
    println!("log:  {:?}", x.log(&s)?);
    println!("tanh: {:?}", x.tanh(&s)?);
    // Two-operand math that is not an operator.
    println!("max(a, b):\n{:?}\n", a.maximum(&b, &s)?);

    // --- Reductions ----------------------------------------------------------

    // `keepdims == false` drops the reduced axes, giving a 0-d array here.
    println!("sum:  {}", a.sum(false, &s)?.item::<f32>());
    println!("mean: {}", a.mean(false, &s)?.item::<f32>());
    println!("max:  {}", a.max(false, &s)?.item::<f32>());
    println!("prod: {}", a.prod(false, &s)?.item::<f32>());

    // Or reduce chosen axes. The named axis is the one that disappears: `a` is
    // (2, 3), so reducing axis 0 collapses the rows and leaves 3 column totals.
    println!(
        "sum along axis 0 (per column): {:?}",
        a.sum_axes(&[0], false, &s)?
    );
    println!(
        "sum along axis 1 (per row):    {:?}",
        a.sum_axes(&[1], false, &s)?
    );
    // With `keepdims == true` the reduced axis stays at length 1 instead of
    // vanishing, which is what keeps the result broadcastable against `a` —
    // (2, 1) stretches back over (2, 3), while a bare (2,) would not.
    let row_means = a.mean_axes(&[1], true, &s)?;
    println!("row means, rank kept: shape {:?}", row_means.shape());
    println!("a centred per row:\n{:?}\n", a.subtract(&row_means, &s)?);

    // --- Shape manipulation --------------------------------------------------

    println!("reshape to (3, 2):\n{:?}", a.reshape(&[3, 2], &s)?);
    println!("transpose:\n{:?}", a.transpose(&s)?);
    println!(
        "broadcast (3) to (2, 3):\n{:?}",
        x.broadcast_to(&[2, 3], &s)?
    );
    // `expand_dims` inserts a length-1 axis; `squeeze` removes every one of them.
    let widened = x.expand_dims(0, &s)?;
    println!("expand_dims: {:?}", widened.shape());
    println!("squeeze:     {:?}\n", widened.squeeze(&s)?.shape());

    // --- Comparisons and masks -----------------------------------------------

    // Comparisons are elementwise and produce bool arrays, so they broadcast
    // against a scalar to give a mask over every element.
    let mask = a.greater(&Array::from_scalar(3.0f32), &s)?;
    println!("a > 3: {:?} ({})", mask.to_vec::<bool>(), mask.dtype());

    // Reduce a mask to a single answer.
    println!("any > 3: {}", mask.any(false, &s)?.item::<bool>());
    println!("all > 3: {}", mask.all(false, &s)?.item::<bool>());

    // Combine masks with the bitwise operators. On `bool` arrays `&`, `|`, and
    // `!` are the logical ops, matching what they mean on Rust's own `bool`.
    let lower = a.less(&Array::from_scalar(5.0f32), &s)?;
    println!("3 < a < 5:  {:?}", (&mask & &lower).to_vec::<bool>());
    println!("not(a > 3): {:?}", (!&mask).to_vec::<bool>());
    // `logical_and` is the method form, and unlike `&` it also accepts numbers,
    // testing truthiness rather than bits.
    println!(
        "same via logical_and: {:?}",
        mask.logical_and(&lower, &s)?.to_vec::<bool>()
    );

    // On integers the same operators do bit-twiddling, again as in Rust.
    let bits = Array::from_slice(&[0b1100i32, 0b1010], &[2]);
    println!("bits & 0b0110: {:?}", (&bits & 0b0110i32).to_vec::<i32>());
    println!("bits << 1:     {:?}", (&bits << 1i32).to_vec::<i32>());

    // `array_equal` is the whole-array answer, unlike elementwise `equal`.
    let same = a.array_equal(&a, false, &s)?;
    println!("a == a (whole array): {}\n", same.item::<bool>());

    // --- Converting and reading out ------------------------------------------

    // `astype` converts dtype; float to int truncates toward zero.
    let halves = Array::from_slice(&[1.7f32, -2.9, 3.2], &[3]);
    let truncated = halves.astype::<i32>(&s)?;
    println!("astype to int32: {:?}", truncated.to_vec::<i32>());
    println!("dtype now: {}", truncated.dtype());
    assert_eq!(truncated.dtype(), Dtype::Int32);

    // `to_vec` copies out in row-major order; `item` reads a single-element array.
    println!("to_vec: {:?}", a.to_vec::<f32>());
    println!("item:   {}\n", Array::from_scalar(1.5f32).item::<f32>());

    // --- Streams and errors --------------------------------------------------

    // Any op takes an explicit stream, so work can be placed per device.
    println!("a + b on CPU:\n{:?}", a.add(&b, &Stream::cpu())?);
    println!("a + b on GPU:\n{:?}", a.add(&b, &Stream::gpu())?);

    // Or steer the default — and with it the operators — once, up front.
    Stream::cpu().set_as_default();
    println!(
        "sum on the new default: {}",
        a.sum(false, &Stream::default())?.item::<f32>()
    );

    // Methods return `Result`, so a bad shape is a value to handle, not a panic.
    let wrong_shape = Array::from_slice(&[1.0f32, 2.0], &[2]);
    match a.add(&wrong_shape, &s) {
        Ok(_) => println!("unexpectedly succeeded"),
        Err(e) => println!("\n(2, 3) + (2,) failed as expected:\n  {e}"),
    }

    Ok(())
}
