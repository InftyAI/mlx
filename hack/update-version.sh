#!/usr/bin/env bash
#
# Set the version of both workspace crates.
#
# Three places have to move together: each crate's own `version`, and the
# `mlxcore-sys` pin in `[workspace.dependencies]`. The pin matters because a
# published `mlxcore` resolves `mlxcore-sys` by version, not by path — leave it
# behind and `cargo publish -p mlxcore` looks for a version nobody uploaded.
#
# Usage: hack/update-version.sh 0.0.1

set -euo pipefail

VERSION="${1:-}"

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]; then
	echo "usage: $(basename "$0") <version>   e.g. $(basename "$0") 0.0.1" >&2
	exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# BSD sed (macOS) requires an argument to -i, GNU sed treats one as the suffix,
# so write a backup either way and delete it.
replace() {
	local pattern="$1" file="$2"
	sed -i.bak -E "$pattern" "$file"
	rm -f "$file.bak"
}

for crate in mlxcore-sys mlxcore; do
	replace "s|^version = \".*\"$|version = \"$VERSION\"|" "$ROOT/crates/$crate/Cargo.toml"
done

replace "s|(mlxcore-sys = \{ path = [^,]+, version = )\"[^\"]*\"|\1\"$VERSION\"|" "$ROOT/Cargo.toml"

# Fail loudly rather than leave a half-bumped workspace: a mismatch here becomes
# a rejected publish at best, and a broken published crate at worst.
for file in "$ROOT/crates/mlxcore-sys/Cargo.toml" "$ROOT/crates/mlxcore/Cargo.toml" "$ROOT/Cargo.toml"; do
	if ! grep -q "\"$VERSION\"" "$file"; then
		echo "error: $file was not updated — check its version line by hand" >&2
		exit 1
	fi
done

# Bring Cargo.lock's record of the two members in line with the manifests.
cargo update --workspace --quiet --manifest-path "$ROOT/Cargo.toml"

echo "version set to $VERSION"
