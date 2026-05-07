---
name: readme-audit
description: Audit and improve README files for all non-"rig" crates in the workspace, and create a unified root README documenting the ecosystem. Ensure all documentation is accurate, specific, and backed by real code examples.
agent: agent
---

## Overview
What the crate does. Concrete, not aspirational. Mention the upstream Rig
trait(s) it implements or the `rig-compose` surface it extends.

## Why it exists
The specific gap in Rig (or in `rig-compose`) that this crate fills. Cite
the trait or pattern it plugs into (e.g. "implements `VectorStoreIndex`").

## Status
Mirror the crate's `Cargo.toml` version + any status notes from `AGENTS.md`.

## Feature flags
Table copied from `Cargo.toml [features]` with one-line descriptions.
Note which combos are exercised by `just check`.

## Key types / architecture
Bullet the public surface re-exported from `lib.rs`. For each, link to
the source file using a relative path. Do NOT list private items.

## Integration with Rig
- Which `rig-core` (or `rig-compose`) version is pinned (read from `Cargo.toml`).
- Which Rig trait(s) it implements or which `rig-compose` extension point
  it plugs into.
- Whether it is runtime-agnostic.

## Usage
A minimal `no_run` example that mirrors an existing test or example file
in the repo. Reference the test by path so reviewers can verify it.

## Validation
The exact command surface from the crate's `justfile`:
```sh
just check
````

List the feature combos that command exercises.

## Gotchas

Real edge cases drawn from code: lock-discipline notes, feature-flag
asymmetries, MSRV implications, WASM caveats, async-runtime constraints.

### Step 3 — Cross-crate map (optional, ask first)

Because these crates live in separate repos, there is no shared root.
Before writing a unified README, ASK the user where it should land:

  1. As a new "## Ecosystem" section appended to each crate's README
     (duplicated, identical text).
  2. As a section in a top-level workspace README.
  3. Skip — link each README to a single canonical source instead.

Do NOT pick unilaterally. If the user picks (1) or (2), the section MUST
contain:

- A Mermaid graph showing dependency edges *with feature flags annotated*
  (e.g. `rig-resources --[graph]--> petgraph`).
- The pinned `rig-core` version each crate consumes (read from `Cargo.toml`).
- One real end-to-end workflow that wires ≥2 of these crates together,
  backed by an actual integration test path.

### Step 4 — Examples must be tests

Every code block in every README MUST be one of:

- A `no_run` doc-test that compiles under `cargo test --doc`.
- A pointer (markdown link) to an existing file under `tests/`,
  `examples/`, or `src/**/*.rs` whose `#[test]` or `#[tokio::test]` is
  known to pass.

If you need a new example to support the README, add it as a real test
under `tests/` (integration) or `examples/` (binary). It must obey the
crate's clippy lint set — `unwrap`/`expect` are allowed in `tests/` and
`examples/` only.

Do NOT inline pseudo-code. Do NOT show APIs that don't exist. Do NOT
fabricate model constants, feature names, or function signatures.

### Step 5 — Validation gate (mandatory, per crate)

For each crate touched, run:

```sh
just check
````

If `just check` is unavailable, run the equivalent:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
# plus each feature combo listed in the justfile
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

A README change is NOT done until every gate passes. If a doctest fails,
fix the doctest — do not weaken the lint set.

### Step 6 — Coherence sweep

After all per-crate writes:

- Re-read each crate's lib.rs `//!` preamble and ensure the README
  Overview matches. If they disagree, prefer the code and update both.
- Re-read each CHANGELOG.md Unreleased section. If the README claims a
  feature not yet in Unreleased, add a Changed/Added bullet. Do NOT
  bump versions — `release-plz` owns version bumps.
- Confirm pinned `rig-core` versions in READMEs match Cargo.toml
  exactly.

## Strict don'ts

- Do NOT fabricate APIs, models, traits, or feature flags.
- Do NOT widen the public surface to make documentation easier.
- Do NOT touch `rig` itself.
- Do NOT add `tokio` to runtime-agnostic libs.
- Do NOT bypass the clippy `deny`/`forbid` set.
- Do NOT rewrite AGENTS.md or CHANGELOG.md as a side effect of
  README work unless the user asked.
- Do NOT bump crate versions; `release-plz` does that.
- Do NOT generate a single shared root README without first asking
  where it should live (separate repos).

## Definition of Done

- Each in-scope crate has a README.md whose every claim is traceable
  to code or Cargo.toml.
- Every code block is a real, passing test or a documented pointer to one.
- `just check` (or its equivalent) is green in every touched crate.
- Crate-level `//!` rustdoc, README.md, and CHANGELOG.md
  Unreleased agree with each other.
- The cross-crate map (if requested) names a real integration test as
  its end-to-end example.

## Mindset

Be skeptical. Read code before writing prose. Prefer linking to a real
test over describing it. When in doubt about scope or location of a
shared artifact, ASK before writing.
