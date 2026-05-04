//! # rig-compose
//!
//! Composable agent kernel for [`rig`](https://github.com/0xplaygrounds/rig)
//! (and any rig-shaped tool surface): stateless skills, transport-agnostic
//! tools, registry-driven agents, and a signal-routing coordinator.
//!
//! Originally extracted from [Azrael](https://github.com/ForeverAngry/azrael)'s
//! threat-detection kernel, this crate is now domain-neutral and can wrap
//! any tool/agent stack.
//!
//! ## Composability rules
//! 1. [`Skill`] is stateless. Any skill can be assigned to any [`Agent`].
//! 2. [`Tool`] is the only side-effectful interface. Local impls and
//!    remote transports (see `rig-mcp`) share the trait; skills cannot
//!    tell them apart.
//! 3. [`Agent`] holds a slice of the global registries — it never
//!    hard-codes skill or tool names.
//! 4. [`Workflow`] composes agents. The [`CoordinatorAgent`] is one
//!    workflow primitive; specialist routing is another configuration of it.

pub mod agent;
pub mod context;
pub mod coordinator;
pub mod delegate;
pub mod instructions;
#[cfg(feature = "manifest")]
pub mod manifest;
pub mod registry;
pub mod skill;
pub mod tool;
pub mod workflow;

pub use agent::{Agent, AgentId, AgentStepResult, GenericAgent, GenericAgentBuilder};
pub use context::{Evidence, InvestigationContext, NextAction, Signal};
pub use coordinator::{CoordinatorAgent, CoordinatorBuilder, RoutingRule};
pub use delegate::{
    DelegateExecutor, DelegateName, DelegateRegistry, DelegateTool, InProcessAgentDelegate,
};
pub use instructions::Instructions;
#[cfg(feature = "manifest")]
pub use manifest::{
    AgentManifest, DelegateKind, DelegateSpec, InstructionsSpec, KnowledgeSpec, ManifestError,
    McpServerSpec, ModelSpec, ToolSpec, delegate_stub, materialize_local_and_delegate_tools,
    materialize_local_and_delegate_tools_with_delegates,
};
pub use registry::{KernelError, SkillRegistry, ToolRegistry};
pub use skill::{Skill, SkillId, SkillOutcome};
pub use tool::{LocalTool, Tool, ToolName, ToolSchema};
pub use workflow::Workflow;
