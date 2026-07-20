//! Build script for `mlxcore-sys`.
//!
//! 1. Builds the vendored `mlx-c` C API with CMake. `mlx-c` uses CMake
//!    `FetchContent` to download and build MLX itself, so both `libmlxc` and
//!    `libmlx` come out of a single CMake invocation.
//! 2. Emits the link directives for those static libraries plus the Apple
//!    system frameworks MLX depends on.
//! 3. Runs `bindgen` over `mlx/c/mlx.h` (the umbrella header) to produce the
//!    raw FFI bindings consumed by `src/lib.rs`.

use std::env;
use std::path::PathBuf;

fn main() {
    if !cfg!(target_os = "macos") {
        panic!("mlxcore-sys currently only supports macOS on Apple Silicon");
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Workspace layout: <root>/crates/mlxcore-sys -> <root>/third_party/mlx-c
    let mlx_c_dir = manifest_dir
        .join("../../third_party/mlx-c")
        .canonicalize()
        .expect("third_party/mlx-c submodule not found — run `git submodule update --init`");

    // --- 1. Build mlx-c (+ MLX via FetchContent) with CMake ---------------
    let mut cfg = cmake::Config::new(&mlx_c_dir);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release");

    if cfg!(feature = "metal") {
        cfg.define("MLX_BUILD_METAL", "ON");
    } else {
        cfg.define("MLX_BUILD_METAL", "OFF");
    }
    if cfg!(feature = "accelerate") {
        cfg.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // Build into a fixed dir under `target/<profile>/` rather than the default
    // per-fingerprint `OUT_DIR`. `cargo clippy` fingerprints `mlxcore-sys`
    // differently from `cargo build`/`cargo test`, so an `OUT_DIR`-based build
    // would recompile MLX from scratch for each (~2.5 min of C++). A shared dir
    // lets them all reuse the same CMake build.
    //
    // OUT_DIR = <target>/<profile>/build/mlxcore-sys-<hash>/out; three parents up is
    // <target>/<profile>.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("unexpected OUT_DIR layout");
    cfg.out_dir(profile_dir.join("mlx-c-build"));

    let dst = cfg.build();

    // --- 2. Link directives ----------------------------------------------
    // CMake installs archives under <dst>/lib.
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    // MLX is C++; pull in the C++ standard library.
    println!("cargo:rustc-link-lib=dylib=c++");

    // Apple system frameworks MLX links against.
    for framework in ["Foundation", "Metal", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }

    // --- 3. Generate bindings --------------------------------------------
    let header = mlx_c_dir.join("mlx/c/mlx.h");
    println!("cargo:rerun-if-changed={}", header.display());

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        // mlx-c headers `#include "mlx/c/..."` relative to the repo root.
        .clang_arg(format!("-I{}", mlx_c_dir.display()))
        .allowlist_function("mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .prepend_enum_name(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("failed to generate mlx-c bindings");

    // bindings.rs stays in the real OUT_DIR — it's cheap to regenerate and
    // `src/lib.rs` includes it from there.
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}
