.PHONY: build test fmt clean

# Build the workspace (compiles mlx-c + MLX from source on first run).
build:
	cargo build

# Run all tests across the workspace.
#
# Serialized: MLX's Metal backend aborts if several threads submit GPU work at
# once, and the default stream is the GPU.
test:
	cargo test -- --test-threads=1

# Format all crates.
fmt:
	cargo fmt --all

# Remove build artifacts.
clean:
	cargo clean
