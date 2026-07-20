//! Raw, unsafe FFI bindings to [`mlx-c`](https://github.com/ml-explore/mlx-c),
//! the C API for Apple's MLX framework.
//!
//! These bindings are generated at build time by `bindgen` from the pinned
//! `third_party/mlx-c` submodule. This crate is not meant to be used directly;
//! prefer the safe `mlxr` crate that wraps it.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
