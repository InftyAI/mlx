//! A minimal MLX example.
//!
//! Run with:
//! ```sh
//! cargo run --example hello
//! ```

use mlx::Array;

fn main() {
    println!("MLX version: {}", mlx::version());

    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("shape: {:?}", a.shape());
    println!("size:  {}", a.size());
    println!("ndim:  {}", a.ndim());
    println!("array:\n{a:?}");
}
