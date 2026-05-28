# Changelog

<!-- markdownlint-disable MD024 -->

All notable changes to `rig-compose` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Versions are managed automatically by [release-plz](https://release-plz.dev/)
from [Conventional Commits](https://www.conventionalcommits.org/).

## [Unreleased]

## [0.5.0](https://github.com/ForeverAngry/rig-compose/compare/v0.4.3...v0.5.0) - 2026-05-28

### Fixed

- *(tool)* Harden result envelope contract ([#26](https://github.com/ForeverAngry/rig-compose/pull/26))

### Added

- Add `ToolResultEnvelope` redaction policies, path-aware omitted-segment
  continuation tokens, and total serialized-byte bounding so model-visible tool
  results can be redacted and globally bounded before downstream prompt, trace,
  or MCP cache use.

## [0.4.3](https://github.com/ForeverAngry/rig-compose/compare/v0.4.2...v0.4.3) - 2026-05-28

### Documentation

- Refresh README to reflect shipped 0.4.x surface ([#24](https://github.com/ForeverAngry/rig-compose/pull/24))

## [0.4.2](https://github.com/ForeverAngry/rig-compose/compare/v0.4.1...v0.4.2) - 2026-05-28

### Added

- Add `ContextProvenance` and `ContextProjectionState` as typed helpers for
  the existing `ContextItem::provenance` JSON field, giving downstream memory,
  resource, graph, and eval producers stable keys for source URI, principal,
  scope, retention tier, timestamps, confidence, version keys, source frame ids,
  projection state, and machine-readable reasons without adding new
  cross-crate dependencies.

## [0.4.1](https://github.com/ForeverAngry/rig-compose/compare/v0.4.0...v0.4.1) - 2026-05-27

### Added

- Add descriptor snapshots and bounded dispatch helpers

### Added

- Add deterministic descriptor snapshots for tools, skills, and in-process
  delegates. `ToolRegistry::descriptors()` returns the visible tool schemas in
  stable name order, `SkillRegistry::descriptors()` returns `SkillDescriptor`
  records, and `DelegateRegistry::descriptors()` returns `DelegateDescriptor`
  records for host-side discovery and adapter export.
- Add bounded dispatch helpers for model-visible tool results:
  `dispatch_tool_invocations_bounded` and
  `dispatch_tool_invocations_with_hooks_bounded` return
  `BoundedToolInvocationResult` records with `ToolResultEnvelope` metadata
  while preserving the existing raw dispatch helpers.
- Add reliability primitives in a new `reliability` module: `RetryClass`
  (`Transient` / `Permanent`), `RetryClassifier` trait with
  `DefaultRetryClassifier` covering every `KernelError` variant,
  `ToolCallFingerprint` (stable hash of tool name + canonical JSON args)
  exposed via `ToolInvocation::fingerprint`, and a deterministic
  `repair_history` that coalesces retried calls (keep first completion;
  otherwise keep last failure; preserve first-occurrence order).
- Add `DispatchTrace` + `DispatchTraceEvent` (with `TracedAction` /
  `TracedOutcome`) and `dispatch_tool_invocations_with_trace`, giving hosts a
  replayable, backend-agnostic record of every hook decision, hook error,
  reservation cleanup, after-hook invocation, and per-invocation outcome
  during normalized tool dispatch.
- Add `AgentLifecycleHook` and `GenericAgentBuilder::with_lifecycle_hook` so
  downstream observers can instrument step start, skill consideration, skill
  completion, step completion, and step errors without adding an observability
  dependency to `rig-compose`.
- Add `examples/context_pack_mix.rs`, showing how a coordinator can mix
  already-projected memory and resource `ContextItem`s into one bounded
  `ContextPack` without adding dependencies from `rig-compose` back to
  producer crates.

### Changed

- `ToolDispatchAction::Skip` now carries an optional reason, and
  `ToolDispatchHook` gains `after_invocation_with_outcome` so audit hooks can
  distinguish real tool execution from synthetic skip results.

## [0.3.1](https://github.com/ForeverAngry/rig-compose/compare/v0.3.0...v0.3.1) - 2026-05-18

### Documentation

- Normalize quick start section
- *(manifest)* Add runnable doctest for AgentManifest ([#17](https://github.com/ForeverAngry/rig-compose/pull/17))
- Bump README crate version
- Rename coordination references to rig-ecosystem
- Align ecosystem docs with rig-compose 0.3, rig-core 0.37, and rig-model-catalog 0.1 ([#12](https://github.com/ForeverAngry/rig-compose/pull/12))
- Update ecosystem topology with rig-compose 0.3 and rig-model-catalog ([#10](https://github.com/ForeverAngry/rig-compose/pull/10))
- Update ecosystem topology with rig-compose 0.3 and rig-model-catalog ([#9](https://github.com/ForeverAngry/rig-compose/pull/9))

## [0.3.0](https://github.com/ForeverAngry/rig-compose/compare/v0.2.1...v0.3.0) - 2026-05-13

### Added

- *(tool)* Add ToolResultEnvelope for bounding large tool outputs ([#7](https://github.com/ForeverAngry/rig-compose/pull/7))
- Add context packing and dispatch hooks
- Normalize tool calls across provider shapes

### Added

- Add crate-local `ROADMAP.md` documenting maturity status, next work, and
  non-goals for the composition kernel.
- Add provider-neutral tool-call normalization and dispatch helpers:
  `ToolCallNormalizer`, `LfmNormalizer`, `StructuredToolCallNormalizer`,
  `ToolInvocation`, `ToolInvocationResult`, and
  `dispatch_tool_invocations`.
- Support LiquidAI LFM/MLX textual tool-call markers, OpenAI Responses
  `function_call` output items, and OpenAI Chat Completions `tool_calls` as
  standard inputs that normalize into the same `ToolInvocation` shape.
- Add `KernelError::NormalizerFailed(String)` for parse/normalization failures
  that are distinct from tool invocation failures.
- Add deterministic coverage for the complete normalize -> dispatch ->
  second-turn answer pattern.
- Add `examples/tool_loop_harness.rs` as the first deterministic harness
  prototype for recording task input, normalized invocations, tool results,
  final answer, and passed assertions.
- Add `ToolDispatchHook`, `ToolDispatchAction`, and
  `dispatch_tool_invocations_with_hooks` so downstream policy, accounting, and
  tracing layers can continue, skip, or terminate normalized tool dispatch.
- Add `DispatchBudgetHook`, a `BudgetGuard`-backed dispatch hook that gates
  normalized tool invocations and releases reservations after success, skip, or
  dispatch error.
- Add provider-neutral context-window planning primitives: `ContextSourceKind`,
  `ContextItem`, `ContextPackConfig`, `ContextPack`, `ContextOmissionReason`,
  and `OmittedContextItem`.
- Add `ToolResultEnvelope`, `ToolResultEnvelopeConfig`, and
  `bound_tool_result` for deterministic bounding metadata around large tool
  results without changing the `Tool` trait.

### Fixed

- `dispatch_tool_invocations_with_hooks` now notifies earlier hooks via
  `on_invocation_error` when a later hook's `before_invocation` returns an
  error. Previously the error short-circuited the dispatch loop without
  giving resource-reserving hooks (such as `DispatchBudgetHook`) a chance to
  release; reservations would leak on every failed pre-flight.

## [0.2.1](https://github.com/ForeverAngry/rig-compose/compare/v0.2.0...v0.2.1) - 2026-05-07

### Documentation

- Remove retired repo references

## [0.2.0](https://github.com/ForeverAngry/rig-compose/compare/v0.1.2...v0.2.0) - 2026-05-06

### Added

- Add budget guards and audit docs

### Added

- `budget` module: `BudgetGuard` and `TokenBudget` traits with lock-free
  `AtomicBudget` and `AtomicTokenBudget` reference implementations for
  cost-bounded coordination (rows, dispatch slots, LLM tokens). Lifted
  from threat-detection agent work so any rig-compose agent can gate work on a
  finite pool.
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
