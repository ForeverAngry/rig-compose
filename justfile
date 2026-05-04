# rig-compose task runner.
#
# Install just: https://github.com/casey/just
#   brew install just
#
# Run `just` with no args to see the recipe list.

default:
    @just --list

# Build all targets with default features.
build:
    cargo build --all-targets

# Run formatter check + clippy + tests across release-relevant feature sets.
check: fmt clippy test doc examples

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-targets -- -D warnings
    cargo clippy --all-targets --features manifest -- -D warnings

test:
    cargo test --all-targets
    cargo test --all-targets --features manifest

doc:
    RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links" cargo doc --all-features --no-deps

examples:
    cargo build --examples --all-features

# Validate the package as it would be uploaded to crates.io.
publish-dry-run:
    cargo publish --dry-run

# Preview what release-plz would update in a disposable copy.
# Install: `cargo install release-plz`.
release-preview:
    #!/usr/bin/env bash
    set -euo pipefail
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    rsync -a --exclude target --exclude .git ./ "${tmp}/"
    cd "${tmp}"
    git init -q
    git config user.email "release-preview@example.invalid"
    git config user.name "Release Preview"
    git add .
    git commit -q -m "feat: prepare rig-compose release preview"
    release-plz update --repo-url https://github.com/ForeverAngry/rig-compose

# Open a release PR locally (writes to a branch). Same thing CI does on push.
release-pr:
    release-plz release-pr

# Inspect the next semver bump release-plz would compute from current commits.
next-version:
    @just release-preview 2>&1 | grep -E "(next version|already up-to-date|rig-compose)" || true