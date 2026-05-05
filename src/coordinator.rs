//! [`CoordinatorAgent`] — **deterministic** signal-tag router. Routes
//! an investigation to the first registered specialist whose signal tag
//! matches the context, with no model in the loop.
//!
//! This is the *optional* delegation path. For autonomous agent-to-agent
//! collaboration where the parent's LLM should choose its peer, use
//! [`crate::delegate::DelegateTool`] instead. Reach for the coordinator
//! only when the routing topology is fixed and a free, no-LLM dispatch
//! hop is the point — for example, an overseer fanning one specialist
//! out per partition before any model has loaded.
//!
//! The coordinator is itself an [`Agent`] so workflows can dispatch to it
//! uniformly. It owns no skills of its own; on `step` it inspects
//! `ctx.signals`, picks the first matching specialist by name, and
//! delegates. Specialists are stored as `Arc<dyn Agent>`, so any kernel
//! agent (including future custom impls) can be registered.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::{Agent, AgentId, AgentStepResult, InvestigationContext, KernelError};

/// One routing rule: any of `signals` matching the context routes to
/// `agent_name`. First-match wins.
#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub agent_name: String,
    pub signals: Vec<String>,
}

impl RoutingRule {
    pub fn new(
        agent_name: impl Into<String>,
        signals: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            agent_name: agent_name.into(),
            signals: signals.into_iter().map(Into::into).collect(),
        }
    }

    fn matches(&self, ctx: &InvestigationContext) -> bool {
        self.signals.iter().any(|s| ctx.has_signal(s))
    }
}

/// Routes investigations across a fixed set of specialist agents.
pub struct CoordinatorAgent {
    id: AgentId,
    name: String,
    rules: Vec<RoutingRule>,
    specialists: HashMap<String, Arc<dyn Agent>>,
    /// Optional fallback agent name. Used when no rule matches.
    fallback: Option<String>,
}

impl CoordinatorAgent {
    pub fn builder(name: impl Into<String>) -> CoordinatorBuilder {
        CoordinatorBuilder {
            name: name.into(),
            rules: Vec::new(),
            specialists: HashMap::new(),
            fallback: None,
        }
    }

    /// Resolve the specialist that should handle `ctx`, if any.
    pub fn route<'a>(&'a self, ctx: &InvestigationContext) -> Option<&'a Arc<dyn Agent>> {
        for rule in &self.rules {
            if !rule.matches(ctx) {
                continue;
            }
            if let Some(agent) = self.specialists.get(&rule.agent_name) {
                return Some(agent);
            }
        }
        self.fallback.as_ref().and_then(|n| self.specialists.get(n))
    }
}

#[async_trait]
impl Agent for CoordinatorAgent {
    fn id(&self) -> AgentId {
        self.id
    }
    fn name(&self) -> &str {
        &self.name
    }
    async fn step(&self, ctx: &mut InvestigationContext) -> Result<AgentStepResult, KernelError> {
        let span = tracing::debug_span!(
            "rig_compose.coordinator.route",
            entity = %ctx.entity_id,
            signals = ctx.signals.len(),
        );
        let _e = span.enter();
        let routed = self.route(ctx).cloned();
        drop(_e);
        match routed {
            Some(agent) => agent.step(ctx).await,
            None => Ok(AgentStepResult {
                skills_run: Vec::new(),
                skills_skipped: Vec::new(),
                confidence: ctx.confidence,
                concluded: false,
            }),
        }
    }
}

pub struct CoordinatorBuilder {
    name: String,
    rules: Vec<RoutingRule>,
    specialists: HashMap<String, Arc<dyn Agent>>,
    fallback: Option<String>,
}

impl CoordinatorBuilder {
    pub fn route(mut self, rule: RoutingRule) -> Self {
        self.rules.push(rule);
        self
    }

    pub fn with_specialist(mut self, agent: Arc<dyn Agent>) -> Self {
        self.specialists.insert(agent.name().to_string(), agent);
        self
    }

    pub fn fallback(mut self, name: impl Into<String>) -> Self {
        self.fallback = Some(name.into());
        self
    }

    pub fn build(self) -> CoordinatorAgent {
        CoordinatorAgent {
            id: AgentId::new(),
            name: self.name,
            rules: self.rules,
            specialists: self.specialists,
            fallback: self.fallback,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;

    use crate::skill::{Skill, SkillOutcome};
    use crate::{GenericAgent, SkillRegistry, ToolRegistry};

    /// Tiny test skill: lifts confidence by 0.3 when its trigger signal is present.
    struct TriggerSkill {
        id: &'static str,
        trigger: &'static str,
    }

    #[async_trait]
    impl Skill for TriggerSkill {
        fn id(&self) -> &str {
            self.id
        }
        fn applies(&self, ctx: &InvestigationContext) -> bool {
            ctx.has_signal(self.trigger)
        }
        async fn execute(
            &self,
            _ctx: &mut InvestigationContext,
            _tools: &ToolRegistry,
        ) -> Result<SkillOutcome, KernelError> {
            Ok(SkillOutcome::default().with_delta(0.3))
        }
    }

    fn build_specialist(name: &str, skill_ids: &[&str], skills: &SkillRegistry) -> Arc<dyn Agent> {
        let tools = ToolRegistry::new();
        let agent = GenericAgent::builder(name)
            .with_skills(skill_ids.iter().copied())
            .build(skills, &tools)
            .unwrap();
        Arc::new(agent)
    }

    fn shared_registry() -> SkillRegistry {
        let r = SkillRegistry::new();
        r.register(Arc::new(TriggerSkill {
            id: "test.fanout",
            trigger: "fanout.high",
        }));
        r.register(Arc::new(TriggerSkill {
            id: "test.spray",
            trigger: "auth.failure.burst",
        }));
        r
    }

    #[tokio::test]
    async fn routes_to_first_matching_specialist() {
        let skills = shared_registry();
        let recon = build_specialist("recon", &["test.fanout"], &skills);
        let credential = build_specialist("credential", &["test.spray"], &skills);
        let coord = CoordinatorAgent::builder("coord")
            .with_specialist(recon)
            .with_specialist(credential)
            .route(RoutingRule::new("recon", ["fanout.high"]))
            .route(RoutingRule::new("credential", ["auth.failure.burst"]))
            .build();
        let mut ctx = InvestigationContext::new("e", "p").with_signal("fanout.high");
        let r = coord.step(&mut ctx).await.unwrap();
        assert!(r.skills_run.iter().any(|s| s == "test.fanout"));
        assert!(ctx.confidence > 0.0);
    }

    #[tokio::test]
    async fn falls_back_when_no_rule_matches() {
        let skills = shared_registry();
        let general = build_specialist("general", &["test.fanout"], &skills);
        let coord = CoordinatorAgent::builder("coord")
            .with_specialist(general)
            .route(RoutingRule::new("nope", ["never.fires"]))
            .fallback("general")
            .build();
        let mut ctx = InvestigationContext::new("e", "p");
        let r = coord.step(&mut ctx).await.unwrap();
        assert!(!r.concluded);
    }

    #[tokio::test]
    async fn unmatched_with_no_fallback_is_noop() {
        let coord = CoordinatorAgent::builder("coord").build();
        let mut ctx = InvestigationContext::new("e", "p");
        let r = coord.step(&mut ctx).await.unwrap();
        assert!(r.skills_run.is_empty());
        assert!(!r.concluded);
    }

    #[tokio::test]
    async fn same_skill_instance_works_for_two_agents() {
        let skills = shared_registry();
        let a = build_specialist("a", &["test.fanout"], &skills);
        let b = build_specialist("b", &["test.fanout"], &skills);
        let mut ctx_a = InvestigationContext::new("x", "p").with_signal("fanout.high");
        let mut ctx_b = InvestigationContext::new("y", "p").with_signal("fanout.high");
        let ra = a.step(&mut ctx_a).await.unwrap();
        let rb = b.step(&mut ctx_b).await.unwrap();
        assert_eq!(ra.skills_run, rb.skills_run);
        assert!(ctx_a.confidence > 0.0);
        assert!(ctx_b.confidence > 0.0);
    }
}
