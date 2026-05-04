//! [`Skill`] — a stateless, composable reasoning unit.
//!
//! Skills are the K4 (Investigative Knowledge) primitive: deterministic Rust
//! code that operates on an [`InvestigationContext`] using the agent's
//! [`super::ToolRegistry`] slice. They are stateless so the same instance
//! can be assigned to any number of agents simultaneously.

use async_trait::async_trait;

use crate::context::{InvestigationContext, NextAction};
use crate::registry::{KernelError, ToolRegistry};

/// Stable identifier for a skill (e.g. `"recon.high_fanout"`,
/// `"general.baseline_compare"`).
pub type SkillId = String;

/// What a skill produced for one execution. The agent loop folds these
/// outcomes into the shared [`InvestigationContext`].
#[derive(Debug, Clone, Default)]
pub struct SkillOutcome {
    /// Confidence delta to apply (positive raises threat probability,
    /// negative lowers). The agent clamps the running confidence after
    /// folding the delta in.
    pub confidence_delta: f32,
    /// Hints for subsequent skills.
    pub next_actions: Vec<NextAction>,
}

impl SkillOutcome {
    pub fn noop() -> Self {
        Self::default()
    }

    pub fn with_delta(mut self, d: f32) -> Self {
        self.confidence_delta = d;
        self
    }

    pub fn with_next(mut self, a: NextAction) -> Self {
        self.next_actions.push(a);
        self
    }
}

/// A composable, stateless reasoning unit.
///
/// Implementations MUST be safe to share across agents: no mutable state in
/// the skill itself. Per-investigation state lives in [`InvestigationContext`].
#[async_trait]
pub trait Skill: Send + Sync {
    fn id(&self) -> &str;

    fn description(&self) -> &str {
        ""
    }

    /// Whether this skill is willing to run given the current context.
    /// Default: always applicable. Specialist skills typically gate on
    /// signal presence or evidence already collected.
    fn applies(&self, _ctx: &InvestigationContext) -> bool {
        true
    }

    /// Execute the skill. Implementations may inspect and mutate `ctx`
    /// directly (appending evidence/signals) and/or return non-evidence
    /// adjustments via [`SkillOutcome`].
    async fn execute(
        &self,
        ctx: &mut InvestigationContext,
        tools: &ToolRegistry,
    ) -> Result<SkillOutcome, KernelError>;
}
