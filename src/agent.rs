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
use crate::skill::Skill;

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

        for skill in &self.skills {
            if concluded && self.short_circuit_on_conclude {
                skills_skipped.push(skill.id().to_string());
                continue;
            }
            if !skill.applies(ctx) {
                skills_skipped.push(skill.id().to_string());
                continue;
            }

            let outcome = skill.execute(ctx, &self.tools).await?;
            ctx.confidence = (ctx.confidence + outcome.confidence_delta).clamp(0.0, 1.0);
            ctx.pending_actions = outcome.next_actions.clone();
            if outcome
                .next_actions
                .iter()
                .any(|a| matches!(a, NextAction::Conclude | NextAction::Discard))
            {
                concluded = true;
            }
            skills_run.push(skill.id().to_string());
        }

        Ok(AgentStepResult {
            skills_run,
            skills_skipped,
            confidence: ctx.confidence,
            concluded,
        })
    }
}

/// Fluent builder for [`GenericAgent`]. The skill chain and tool whitelist
/// are resolved against the supplied registries at [`Self::build`] time.
pub struct GenericAgentBuilder {
    name: String,
    skill_ids: Vec<String>,
    allowed_tools: Option<Vec<String>>,
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
            short_circuit_on_conclude: self.short_circuit_on_conclude,
        })
    }
}
