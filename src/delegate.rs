//! Agent delegation primitives.
//!
//! A delegate is another agent-shaped capability exposed as a normal
//! [`Tool`]. The model still decides when to call it; the runtime merely
//! provides an implementation. This keeps delegation agentic while allowing
//! hosts to run child agents in the same process when that is cheaper than
//! MCP or another remote transport.

use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::context::{InvestigationContext, Signal};
use crate::registry::KernelError;
use crate::tool::{Tool, ToolSchema};

/// Stable key for a delegate executor. Manifests usually reference this
/// through `delegates[].agent`; when omitted, `delegates[].name` is used.
pub type DelegateName = String;

/// Async implementation behind a delegate tool.
#[async_trait]
pub trait DelegateExecutor: Send + Sync {
    async fn invoke(&self, args: Value) -> Result<Value, KernelError>;
}

/// Registry of in-process delegate executors provided by the host.
#[derive(Clone, Default)]
pub struct DelegateRegistry {
    inner: Arc<DashMap<DelegateName, Arc<dyn DelegateExecutor>>>,
}

impl DelegateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, name: impl Into<String>, executor: Arc<dyn DelegateExecutor>) {
        self.inner.insert(name.into(), executor);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn DelegateExecutor>> {
        self.inner.get(name).map(|v| v.clone())
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Tool wrapper around a delegate executor.
pub struct DelegateTool {
    schema: ToolSchema,
    executor: Arc<dyn DelegateExecutor>,
}

impl DelegateTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        executor: Arc<dyn DelegateExecutor>,
    ) -> Self {
        Self {
            schema: ToolSchema {
                name: name.into(),
                description: description.into(),
                args_schema: json!({"type": "object"}),
                result_schema: json!({"type": "object"}),
            },
            executor,
        }
    }

    pub fn with_schema(schema: ToolSchema, executor: Arc<dyn DelegateExecutor>) -> Self {
        Self { schema, executor }
    }
}

#[async_trait]
impl Tool for DelegateTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn name(&self) -> crate::ToolName {
        self.schema.name.clone()
    }

    async fn invoke(&self, args: Value) -> Result<Value, KernelError> {
        self.executor.invoke(args).await
    }
}

/// Delegate executor that drives another [`Agent`] in the same process.
pub struct InProcessAgentDelegate {
    agent: Arc<dyn Agent>,
}

impl InProcessAgentDelegate {
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self { agent }
    }

    pub fn arc(agent: Arc<dyn Agent>) -> Arc<dyn DelegateExecutor> {
        Arc::new(Self::new(agent))
    }
}

#[async_trait]
impl DelegateExecutor for InProcessAgentDelegate {
    async fn invoke(&self, args: Value) -> Result<Value, KernelError> {
        let mut ctx = context_from_args(args)?;
        let result = self.agent.step(&mut ctx).await?;
        Ok(json!({
            "delegate_kind": "in_process",
            "agent": self.agent.name(),
            "result": {
                "skills_run": result.skills_run,
                "skills_skipped": result.skills_skipped,
                "confidence": result.confidence,
                "concluded": result.concluded,
            },
            "context": ctx,
        }))
    }
}

fn context_from_args(args: Value) -> Result<InvestigationContext, KernelError> {
    if let Some(context) = args.get("context") {
        return Ok(serde_json::from_value(context.clone())?);
    }
    if let Ok(ctx) = serde_json::from_value::<InvestigationContext>(args.clone()) {
        return Ok(ctx);
    }

    let entity_id = args
        .get("entity_id")
        .and_then(Value::as_str)
        .unwrap_or("delegate")
        .to_string();
    let partition = args
        .get("partition")
        .and_then(Value::as_str)
        .unwrap_or("default")
        .to_string();
    let mut ctx = InvestigationContext::new(entity_id, partition);
    if let Some(confidence) = args.get("confidence").and_then(Value::as_f64) {
        ctx.confidence = (confidence as f32).clamp(0.0, 1.0);
    }
    if let Some(signals) = args.get("signals").and_then(Value::as_array) {
        ctx.signals.extend(
            signals
                .iter()
                .filter_map(Value::as_str)
                .map(|s| Signal::new(s.to_string())),
        );
    }
    Ok(ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenericAgent, Skill, SkillOutcome, SkillRegistry, ToolRegistry};

    struct ConfidenceSkill;

    #[async_trait]
    impl Skill for ConfidenceSkill {
        fn id(&self) -> &str {
            "test.confidence"
        }

        fn description(&self) -> &str {
            "raises confidence"
        }

        fn applies(&self, ctx: &InvestigationContext) -> bool {
            ctx.has_signal("raise")
        }

        async fn execute(
            &self,
            _ctx: &mut InvestigationContext,
            _tools: &ToolRegistry,
        ) -> Result<SkillOutcome, KernelError> {
            Ok(SkillOutcome::default().with_delta(0.4))
        }
    }

    #[tokio::test]
    async fn in_process_delegate_drives_child_agent() {
        let skills = SkillRegistry::new();
        skills.register(Arc::new(ConfidenceSkill));
        let tools = ToolRegistry::new();
        let agent = GenericAgent::builder("child")
            .with_skills(["test.confidence"])
            .build(&skills, &tools)
            .expect("agent");
        let executor = InProcessAgentDelegate::arc(Arc::new(agent));
        let tool = DelegateTool::new("child_agent", "child", executor);
        let out = tool
            .invoke(json!({
                "entity_id": "host-a",
                "partition": "lab",
                "signals": ["raise"],
            }))
            .await
            .expect("invoke");
        assert_eq!(out["delegate_kind"], "in_process");
        assert_eq!(out["agent"], "child");
        assert_eq!(out["result"]["skills_run"][0], "test.confidence");
        assert!((out["result"]["confidence"].as_f64().unwrap() - 0.4).abs() < 1e-6);
    }
}
