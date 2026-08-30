.PHONY: build test fmt clean version publish-dry publish

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

# Set the version of both crates. Usage: make version VERSION=0.0.1
version:
	@test -n "$(VERSION)" || { echo "usage: make version VERSION=0.0.1" >&2; exit 1; }
	./hack/update-version.sh $(VERSION)

# Package and build both crates the way crates.io will, without uploading.
publish-dry:
	cargo publish --workspace --dry-run

# Publish to crates.io. Needs `cargo login` first, and cannot be undone: a
# published version can be yanked, never deleted.
publish: test
	cargo publish --workspace
