//! A minimal MLX example.
//!
//! Run with:
//! ```sh
//! cargo run --example hello
//! ```

use mlxcore::{Array, Stream};

fn main() -> mlxcore::Result<()> {
    println!("MLX version: {}", mlxcore::version());

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("shape: {:?}", a.shape());
    println!("size:  {}", a.size());
    println!("ndim:  {}", a.ndim());
    println!("array:\n{a:?}");

    let b = Array::from_slice(&[6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]);

    // Operators use the default stream and panic on error.
    println!("a + b:\n{:?}", &a + &b);

    // Or steer the default onto a specific device once, up front.
    Stream::cpu().set_as_default();
    let total = a.sum(false, &Stream::default())?;
    println!("sum(a) on CPU: {}", total.item::<f32>());

    // Explicit stream control is still available per-op; `?` propagates errors.
    let product = a.multiply(&b, &Stream::gpu())?;
    println!("a * b (GPU):\n{product:?}");

    Ok(())
}
