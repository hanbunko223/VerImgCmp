#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export CARGO_TARGET_DIR="$ROOT/target"
export RUSTFLAGS='-C target-cpu=native'
export CARGO_PROFILE_RELEASE_LTO=thin
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
cargo test --locked --release --manifest-path "$ROOT/Cargo.toml" --bin poseidon_97 -- --test-threads=1
cargo build --locked --release --manifest-path "$ROOT/Cargo.toml" --bins
