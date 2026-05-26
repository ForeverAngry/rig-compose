//! [`Agent`] — composes a [`SkillRegistry`] slice and a scoped
//! [`ToolRegistry`] to drive an investigation.
//!
//! Agents are *thin*. They do not contain detection logic; that lives in
//! skills. An agent's only responsibility is selecting which skills apply
//! and folding their outcomes into the shared [`InvestigationContext`].

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::context::{InvestigationContext, NextAction};
use crate::registry::{KernelError, SkillRegistry, ToolRegistry};
use crate::skill::{Skill, SkillOutcome};

/// Identifier for an agent instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub Uuid);

impl AgentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Outcome of one [`Agent::step`] call.
#[derive(Debug, Clone)]
pub struct AgentStepResult {
    /// Skills that were considered applicable and executed.
    pub skills_run: Vec<String>,
    /// Skills that were applicable but the agent declined to run (e.g.
    /// because a previous skill emitted [`NextAction::Conclude`]).
    pub skills_skipped: Vec<String>,
    /// Final confidence after this step.
    pub confidence: f32,
    /// Whether the agent considers the investigation terminated.
    pub concluded: bool,
}

/// Lifecycle hook around [`GenericAgent`] step and skill execution.
///
/// Hooks are producer-neutral: they do not depend on any telemetry backend.
/// Downstream crates can implement this trait to emit tracing events, collect
/// deterministic test records, or account for per-skill work without changing
/// the [`Agent`] trait itself.
#[async_trait]
pub trait AgentLifecycleHook: Send + Sync {
    /// Called before a [`GenericAgent`] step starts.
    async fn before_step(
        &self,
        _agent: &GenericAgent,
        _ctx: &InvestigationContext,
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called when a skill is considered by the agent loop.
    async fn before_skill(
        &self,
        _agent: &GenericAgent,
        _skill_id: &str,
        _applies: bool,
        _confidence: f32,
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called after an applicable skill executes.
    async fn after_skill(
        &self,
        _agent: &GenericAgent,
        _skill_id: &str,
        _outcome: &SkillOutcome,
        _confidence: f32,
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called when a step completed successfully.
    async fn after_step(
        &self,
        _agent: &GenericAgent,
        _result: &AgentStepResult,
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called when the agent loop is aborting because a skill or hook failed.
    async fn on_step_error(
        &self,
        _agent: &GenericAgent,
        _error: &KernelError,
    ) -> Result<(), KernelError> {
        Ok(())
    }
}

/// A composable agent: skills + scoped tools + a step driver.
#[async_trait]
pub trait Agent: Send + Sync {
    fn id(&self) -> AgentId;
    fn name(&self) -> &str;

    /// Drive one investigation pass over `ctx`. Implementations decide
    /// the iteration policy (single pass, fixed-point, until-conclude…).
    async fn step(&self, ctx: &mut InvestigationContext) -> Result<AgentStepResult, KernelError>;
}

/// Default-shape agent built from a declared chain of skill ids and a
/// scoped tool registry. Specialist agents in Phase 5 are constructed
/// purely as different `GenericAgent` configurations — no new `Agent`
/// impls required.
pub struct GenericAgent {
    id: AgentId,
    name: String,
    skills: Vec<Arc<dyn Skill>>,
    tools: ToolRegistry,
    lifecycle_hooks: Vec<Arc<dyn AgentLifecycleHook>>,
    /// If true, stop the chain when a skill emits [`NextAction::Conclude`]
    /// or [`NextAction::Discard`]. Default `true`.
    pub short_circuit_on_conclude: bool,
}

impl GenericAgent {
    pub fn builder(name: impl Into<String>) -> GenericAgentBuilder {
        GenericAgentBuilder {
            name: name.into(),
            skill_ids: Vec::new(),
            allowed_tools: None,
            lifecycle_hooks: Vec::new(),
            short_circuit_on_conclude: true,
        }
    }

    pub fn skills(&self) -> &[Arc<dyn Skill>] {
        &self.skills
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }
}

#[async_trait]
impl Agent for GenericAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn step(&self, ctx: &mut InvestigationContext) -> Result<AgentStepResult, KernelError> {
        let mut skills_run = Vec::new();
        let mut skills_skipped = Vec::new();
        let mut concluded = false;

        // Telemetry hook failures must never abort the agent loop, nor mask
        // the real error returned by a skill. Each `notify_*` helper logs
        // and swallows hook errors internally.
        self.notify_before_step(ctx).await;

        for skill in &self.skills {
            if concluded && self.short_circuit_on_conclude {
                let skill_id = skill.id();
                self.notify_before_skill(skill_id, false, ctx.confidence)
                    .await;
                skills_skipped.push(skill.id().to_string());
                continue;
            }
            if !skill.applies(ctx) {
                let skill_id = skill.id();
                self.notify_before_skill(skill_id, false, ctx.confidence)
                    .await;
                skills_skipped.push(skill.id().to_string());
                continue;
            }

            let skill_id = skill.id();
            self.notify_before_skill(skill_id, true, ctx.confidence)
                .await;

            let outcome = match skill.execute(ctx, &self.tools).await {
                Ok(outcome) => outcome,
                Err(error) => {
                    self.notify_step_error(&error).await;
                    return Err(error);
                }
            };
            ctx.confidence = (ctx.confidence + outcome.confidence_delta).clamp(0.0, 1.0);
            ctx.pending_actions = outcome.next_actions.clone();
            self.notify_after_skill(skill_id, &outcome, ctx.confidence)
                .await;
            if outcome
                .next_actions
                .iter()
                .any(|a| matches!(a, NextAction::Conclude | NextAction::Discard))
            {
                concluded = true;
            }
            skills_run.push(skill_id.to_string());
        }

        let result = AgentStepResult {
            skills_run,
            skills_skipped,
            confidence: ctx.confidence,
            concluded,
        };
        self.notify_after_step(&result).await;
        Ok(result)
    }
}

impl GenericAgent {
    async fn notify_before_step(&self, ctx: &InvestigationContext) {
        for hook in &self.lifecycle_hooks {
            if let Err(err) = hook.before_step(self, ctx).await {
                tracing::warn!(error = %err, "lifecycle hook before_step failed; continuing");
            }
        }
    }

    async fn notify_before_skill(&self, skill_id: &str, applies: bool, confidence: f32) {
        for hook in &self.lifecycle_hooks {
            if let Err(err) = hook.before_skill(self, skill_id, applies, confidence).await {
                tracing::warn!(
                    error = %err,
                    skill_id,
                    "lifecycle hook before_skill failed; continuing"
                );
            }
        }
    }

    async fn notify_after_skill(&self, skill_id: &str, outcome: &SkillOutcome, confidence: f32) {
        for hook in &self.lifecycle_hooks {
            if let Err(err) = hook.after_skill(self, skill_id, outcome, confidence).await {
                tracing::warn!(
                    error = %err,
                    skill_id,
                    "lifecycle hook after_skill failed; continuing"
                );
            }
        }
    }

    async fn notify_after_step(&self, result: &AgentStepResult) {
        for hook in &self.lifecycle_hooks {
            if let Err(err) = hook.after_step(self, result).await {
                tracing::warn!(error = %err, "lifecycle hook after_step failed; continuing");
            }
        }
    }

    async fn notify_step_error(&self, error: &KernelError) {
        for hook in &self.lifecycle_hooks {
            if let Err(err) = hook.on_step_error(self, error).await {
                tracing::warn!(error = %err, "lifecycle hook on_step_error failed; continuing");
            }
        }
    }
}

/// Fluent builder for [`GenericAgent`]. The skill chain and tool whitelist
/// are resolved against the supplied registries at [`Self::build`] time.
pub struct GenericAgentBuilder {
    name: String,
    skill_ids: Vec<String>,
    allowed_tools: Option<Vec<String>>,
    lifecycle_hooks: Vec<Arc<dyn AgentLifecycleHook>>,
    short_circuit_on_conclude: bool,
}

impl GenericAgentBuilder {
    pub fn with_skills<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.skill_ids.extend(ids.into_iter().map(Into::into));
        self
    }

    pub fn with_tools<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = Some(names.into_iter().map(Into::into).collect());
        self
    }

    pub fn short_circuit_on_conclude(mut self, v: bool) -> Self {
        self.short_circuit_on_conclude = v;
        self
    }

    /// Add a lifecycle hook invoked around step and skill execution.
    pub fn with_lifecycle_hook(mut self, hook: Arc<dyn AgentLifecycleHook>) -> Self {
        self.lifecycle_hooks.push(hook);
        self
    }

    pub fn build(
        self,
        skills: &SkillRegistry,
        tools: &ToolRegistry,
    ) -> Result<GenericAgent, KernelError> {
        let resolved = skills.resolve_chain(self.skill_ids.iter())?;
        let scoped_tools = match self.allowed_tools {
            Some(list) => tools.scoped(list),
            None => tools.clone(),
        };
        Ok(GenericAgent {
            id: AgentId::new(),
            name: self.name,
            skills: resolved,
            tools: scoped_tools,
            lifecycle_hooks: self.lifecycle_hooks,
            short_circuit_on_conclude: self.short_circuit_on_conclude,
        })
    }
}
