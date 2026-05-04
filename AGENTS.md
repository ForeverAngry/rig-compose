# AGENTS.md

Guidance for AI coding agents working in `rig-compose`.

## Project

Composable agent kernel for [Rig](https://crates.io/crates/rig-core):
stateless skills, transport-agnostic tools, registry-driven agents,
signal-routing coordinator, in-process delegates, and an optional manifest
loader.

## Rules

- Rust 2024, MSRV 1.88. Library is runtime-agnostic; do not add `tokio` to
  `[dependencies]`.
- Errors: `thiserror` enums (`KernelError`, etc.); return `Result<_, _>`.
- Never `.await` while holding a `Mutex`/`RwLock` guard. Scope-drop first.
- No `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`, `dbg!`,
  indexing/slicing, or `unreachable!` in library code — clippy
  `deny`/`forbid`. Use `?`, `ok_or(...)`, `get(..)`, `match`.
  Allowed in `tests/`, `examples/`, `#[cfg(test)]`.
- Use `tracing` for logs; no `println!` in library code.
- Document new `pub` items with `///` rustdoc.
- Re-export new public items from [src/lib.rs](src/lib.rs).

## Features

Default = none. Optional: `manifest` (pulls `serde_yaml`). Gate optional
code with `#[cfg(feature = "manifest")]`. CI matrix: `default`, `manifest`.

## Validation

```sh
just check
# fmt + clippy (× features) + test (× features) + rustdoc strict + examples
```

## Scope

Do not vendor `rig-core`. Update [README.md](README.md) and
[CHANGELOG.md](CHANGELOG.md) for user-visible changes.
