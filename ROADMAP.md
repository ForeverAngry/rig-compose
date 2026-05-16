# rig-compose Roadmap

This roadmap is the crate-local operating plan for `rig-compose`. The cross-crate coordination summary lives in [`rig-ecosystem/docs/roadmap.md`](../rig-ecosystem/docs/roadmap.md).

## Role

`rig-compose` is the provider-neutral agent composition kernel. It owns the stable shapes for skills, tools, registries, workflows, delegates, coordinator routing, budgets, tool-call normalization, dispatch hooks, and context-window packing.

It must stay independent from `rig-core`, Memvid, MCP, concrete resource stores, provider SDKs, and product UI/runtime concerns.

## Landed

- Stateless `Skill` and async `Tool` traits with shared registries.
- Generic agents, workflows, in-process delegates, and deterministic coordinator routing.
- Optional YAML `AgentManifest` loading behind `manifest`.
- Budget guards and drop-safe token reservations.
- Provider-neutral tool-call normalization for LFM/MLX markers, OpenAI Responses `function_call`, and OpenAI Chat Completions `tool_calls`.
- Dispatch hooks that can continue, skip, terminate, and clean up reservations on later-hook failures.
- Provider-neutral context packing with `ContextSourceKind`, `ContextItem`, `ContextPackConfig`, `ContextPack`, and explicit omission reasons.
- Deterministic `examples/tool_loop_harness.rs` prototype for tool-loop replay records.

## Prototype Grade

- Harness records exist as examples/tests, but there is no reusable fixture runner yet.
- Context packing is validated against in-crate memory-like and tool-result-like items, but downstream crates still need to project concrete memory, resource, graph, and lineage data into it.
- Dispatch policy has hooks and budget accounting, but richer approval, retry, result-size, stuck-loop, and trace policies still live downstream or remain unbuilt.

## Next Work

1. Stabilize the context vocabulary by wiring second-source downstream projections from `rig-memvid` and `rig-resources` without taking dependencies on those crates.
2. Extend dispatch trace records for hook decisions, skipped calls, termination reasons, reservation cleanup, and downstream policy outcomes.
3. Add reliability primitives around repeated tool-call fingerprints, bounded tool-result envelopes, retry classification, and deterministic history repair.
4. Evolve `examples/tool_loop_harness.rs` into a fixture-compatible run record only after the same shape is proven in `rig-mcp` and memory-aware examples.
5. Keep the `manifest` feature small: portable agent manifests, materialization helpers, and schema evolution only.

## Maturity Bar

- Kernel shapes are validated by at least two independent sources before stabilization.
- A replay can explain which context, tool calls, hook decisions, and omissions were available to a model turn.
- Budget reservations are released on every success and failure path.
- The same dispatch path works for local tools, MCP-adapted tools, and in-process delegates.
- Feature-matrix checks, docs, examples, and rustdoc stay green under `just check`.

## Non-Goals

- Do not call models or depend on provider SDKs.
- Do not depend on MCP, Memvid, resource stores, or product UI/runtime crates.
- Do not absorb concrete security, memory, graph, or approval implementations that belong in downstream crates.
