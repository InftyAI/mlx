.PHONY: build test fmt clean

# Build the workspace (compiles mlx-c + MLX from source on first run).
build:
	cargo build

# Run all tests across the workspace.
test:
	cargo test

# Format all crates.
fmt:
	cargo fmt --all

# Remove build artifacts.
clean:
	cargo clean
