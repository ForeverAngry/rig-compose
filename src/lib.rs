//! # rig-compose
//!
//! Composable agent kernel for [`rig`](https://github.com/0xplaygrounds/rig)
//! (and any rig-shaped tool surface): stateless skills, transport-agnostic
//! tools, registry-driven agents, and a signal-routing coordinator.
//!
//! Extracted from a threat-detection kernel, this crate is now domain-neutral
//! and can wrap any tool/agent stack.
//!
//! ## Composability rules
//! 1. [`Skill`] is stateless. Any skill can be assigned to any [`Agent`].
//! 2. [`Tool`] is the only side-effectful interface. Local impls and
//!    remote transports (see `rig-mcp`) share the trait; skills cannot
//!    tell them apart.
//! 3. [`Agent`] holds a slice of the global registries — it never
//!    hard-codes skill or tool names.
//! 4. [`Workflow`] composes agents.
//!
//! ## Two paths for agent-to-agent delegation
//!
//! rig-compose ships **two complementary delegation primitives**. Pick
//! the one that matches who should decide *when* a peer agent is invoked.
//!
//! **Default — [`delegate::DelegateTool`] (model-driven, agentic).**
//! A child agent is exposed as a normal [`Tool`]. The parent agent's
//! model decides when to call it, just like any other tool. This is the
//! recommended path for autonomous agent-to-agent collaboration, because
//! the topology stays open and transport-symmetric: the same tool call
//! works whether the peer runs in-process via
//! [`delegate::InProcessAgentDelegate`] or behind an MCP server via
//! `rig_mcp::McpTool`.
//!
//! **Optional — [`coordinator::CoordinatorAgent`] (deterministic, host-driven).**
//! A signal-tag router that hops to the first matching specialist with
//! no model in the loop. Reach for it only when the routing topology is
//! fixed up-front and you specifically want a free dispatch hop —
//! typical use is an overseer that fans one specialist out per partition
//! before any LLM has loaded. Not a substitute for `DelegateTool` when
//! you want the *agents* to choose how they cooperate.

pub mod agent;
pub mod budget;
pub mod context;
/// Deterministic, no-LLM signal-tag router. See the crate-level
/// "Two paths for agent-to-agent delegation" section: prefer
/// [`delegate::DelegateTool`] for model-driven peer collaboration and
/// reach for [`coordinator::CoordinatorAgent`] only when the topology
/// is fixed and a free dispatch hop matters.
pub mod coordinator;
/// Model-driven, transport-agnostic agent delegation. The default path
/// for autonomous agent-to-agent collaboration. See the crate-level
/// "Two paths for agent-to-agent delegation" section.
pub mod delegate;
pub mod instructions;
#[cfg(feature = "manifest")]
pub mod manifest;
/// Tool-call normalizer: converts raw model text output (in-band markers)
/// into structured [`normalizer::ToolInvocation`]s for kernel dispatch.
pub mod normalizer;
pub mod registry;
pub mod reliability;
pub mod skill;
pub mod tool;
pub mod trace;
pub mod workflow;

pub use agent::{
    Agent, AgentId, AgentLifecycleHook, AgentStepResult, GenericAgent, GenericAgentBuilder,
};
pub use budget::{
    AtomicBudget, AtomicTokenBudget, BudgetError, BudgetGuard, DispatchBudgetHook, TokenBudget,
    TokenRefund, TokenReservation,
};
pub use context::{
    ContextItem, ContextOmissionReason, ContextPack, ContextPackConfig, ContextProjectionState,
    ContextProvenance, ContextSourceKind, Evidence, InvestigationContext, NextAction,
    OmittedContextItem, Signal,
};
pub use coordinator::{CoordinatorAgent, CoordinatorBuilder, RoutingRule};
pub use delegate::{
    DelegateDescriptor, DelegateExecutor, DelegateName, DelegateRegistry, DelegateTool,
    InProcessAgentDelegate,
};
pub use instructions::Instructions;
#[cfg(feature = "manifest")]
pub use manifest::{
    AgentManifest, DelegateKind, DelegateSpec, InstructionsSpec, KnowledgeSpec, ManifestError,
    McpServerSpec, ModelSpec, ToolSpec, delegate_stub, materialize_local_and_delegate_tools,
    materialize_local_and_delegate_tools_with_delegates,
};
pub use normalizer::{
    BoundedToolInvocationResult, LfmNormalizer, StructuredToolCallNormalizer, ToolCallNormalizer,
    ToolDispatchAction, ToolDispatchHook, ToolInvocation, ToolInvocationOutcome,
    ToolInvocationResult, dispatch_tool_invocations, dispatch_tool_invocations_bounded,
    dispatch_tool_invocations_with_hooks, dispatch_tool_invocations_with_hooks_bounded,
    dispatch_tool_invocations_with_trace,
};
pub use registry::{KernelError, SkillDescriptor, SkillRegistry, ToolRegistry};
pub use reliability::{
    DefaultRetryClassifier, HistoryEntry, RetryClass, RetryClassifier, ToolCallFingerprint,
    repair_history,
};
pub use skill::{Skill, SkillId, SkillOutcome};
pub use tool::{
    LocalTool, Tool, ToolName, ToolResultEnvelope, ToolResultEnvelopeConfig, ToolSchema,
    bound_tool_result,
};
pub use trace::{DispatchTrace, DispatchTraceEvent, TracedAction, TracedOutcome};
pub use workflow::Workflow;
