# Changelog

All notable changes to `rig-compose` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are managed automatically by [release-plz](https://release-plz.dev/)
from [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

### Added

- `budget` module: `BudgetGuard` and `TokenBudget` traits with lock-free
  `AtomicBudget` and `AtomicTokenBudget` reference implementations for
  cost-bounded coordination (rows, dispatch slots, LLM tokens). Lifted
  from Azrael so any rig-compose agent can gate work on a finite pool.
- `TokenReservation` handle returned by `TokenBudget::try_reserve_tokens`,
  so `AtomicTokenBudget::record_usage` reconciles actual usage against one
  prompt's reservation instead of a global outstanding-token counter.
- `TokenReservation` now refunds its estimated tokens to the originating
  budget when dropped without `record_usage`, preventing leaks on early
  returns or panics. `TokenRefund` is re-exported from the crate root.
- `KernelError::ToolNotApplicable(String)` variant for soft tool failures
  that callers should treat as a no-op rather than an error.

## [0.1.2](https://github.com/ForeverAngry/rig-compose/compare/v0.1.1...v0.1.2) - 2026-05-05

### Fixed

- Clarify deterministic vs model-driven delegation paths

## [0.1.1](https://github.com/ForeverAngry/rig-compose/compare/v0.1.0...v0.1.1) - 2026-05-04

### Fixed

- Correct span name, author, and doc link

## [0.1.0] - Unreleased

### Added

- Initial release of the composable agent kernel.
- Stateless `Skill` trait, transport-agnostic `Tool` trait, shared registries,
  generic agents, coordinator routing, workflows, instruction data, and
  in-process delegate support.
- Optional `manifest` feature for portable agent manifest loading.
