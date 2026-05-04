# rig-compose

**Composable agent kernel for [Rig](https://github.com/0xPlaygrounds/rig)-shaped systems.**

[![crates.io](https://img.shields.io/crates/v/rig-compose.svg)](https://crates.io/crates/rig-compose)
[![docs.rs](https://img.shields.io/docsrs/rig-compose)](https://docs.rs/rig-compose)
[![license](https://img.shields.io/crates/l/rig-compose.svg)](./LICENSE-MIT)

---

Composable agent kernel for [rig](https://github.com/0xplaygrounds/rig).

- **Stateless `Skill`** trait — assignable to any agent.
- **Transport-agnostic `Tool`** trait — local closures or remote MCP servers
  (via the companion `rig-mcp` crate) look identical at the call site.
- **Registry-driven `Agent`** — `GenericAgent::builder()` with declarative
  skill chains and per-agent tool whitelisting via `ToolRegistry::scoped`.
- **Signal-routing `CoordinatorAgent`** — first-match `RoutingRule`s with
  optional fallback.

Originally extracted from [Azreal](https://github.com/ForeverAngry/rig-compose).

## Install

```toml
[dependencies]
rig-compose = "0.1"
```

Enable manifest loading when you want the portable agent manifest schema:

```toml
[dependencies]
rig-compose = { version = "0.1", features = ["manifest"] }
```

## Quickstart

```rust,no_run
use async_trait::async_trait;
use rig_compose::{
  Agent, GenericAgent, InvestigationContext, KernelError, Skill, SkillOutcome,
  SkillRegistry, ToolRegistry,
};

struct KeywordSkill;

#[async_trait]
impl Skill for KeywordSkill {
  fn id(&self) -> &str {
    "example.keyword"
  }

  fn applies(&self, context: &InvestigationContext) -> bool {
    context.has_signal("keyword.match")
  }

  async fn execute(
    &self,
    _context: &mut InvestigationContext,
    _tools: &ToolRegistry,
  ) -> Result<SkillOutcome, KernelError> {
    Ok(SkillOutcome::default().with_delta(0.25))
  }
}

# async fn run() -> Result<(), KernelError> {
let skills = SkillRegistry::new();
skills.register(std::sync::Arc::new(KeywordSkill));

let tools = ToolRegistry::new();
let agent = GenericAgent::builder("triage")
  .with_skills(["example.keyword"])
  .build(&skills, &tools)?;

let mut context = InvestigationContext::new("entity-1", "default")
  .with_signal("keyword.match");
let result = agent.step(&mut context).await?;

assert_eq!(result.skills_run, vec!["example.keyword".to_string()]);
# Ok(()) }
```

See [examples/basic_agent.rs](examples/basic_agent.rs) for a compiling version.

## Features

| Feature | Default | Description |
| ------- | :-----: | ----------- |
| `manifest` | - | Enables serde-yaml backed portable agent manifests. |

## Compatibility

`rig-compose` targets Rust 2024 and has MSRV 1.88.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
