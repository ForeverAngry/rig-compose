//! [`ToolRegistry`] and [`SkillRegistry`] — composition surfaces.
//!
//! Both registries are append-only at runtime and indexed by name. Agents
//! reference entries by name and never own them; the same skill or tool
//! instance can be shared across any number of agents.

use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::skill::{Skill, SkillId};
use crate::tool::{Tool, ToolName};

/// Snapshot record for a visible skill in a [`SkillRegistry`].
///
/// Descriptors are read-only catalog entries for discovery, documentation,
/// and adapter export. They do not participate in execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillDescriptor {
    /// Stable skill identifier.
    pub id: SkillId,
    /// Human-readable skill description. Empty when the skill does not provide
    /// one.
    pub description: String,
}

#[derive(Debug, Error)]
pub enum KernelError {
    #[error("tool `{0}` not found in registry")]
    ToolNotFound(String),

    #[error("tool `{0}` is not authorised for this agent")]
    ToolNotAuthorised(String),

    #[error("skill `{0}` not found in registry")]
    SkillNotFound(String),

    #[error("tool invocation failed: {0}")]
    ToolFailed(String),

    /// Soft failure: the tool ran without infrastructure error but the
    /// requested operation was inapplicable to its current state (e.g.
    /// expanding around an entity the graph has never seen). Callers
    /// can treat this as a no-op rather than propagating an error.
    #[error("tool not applicable: {0}")]
    ToolNotApplicable(String),

    #[error("skill execution failed: {0}")]
    SkillFailed(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Failed to parse tool-call markers in raw model output. Distinct from
    /// [`Self::ToolFailed`] (which signals an invocation-time error) so
    /// callers can distinguish a parse/normalizer failure from a runtime one.
    #[error("tool-call normalizer failed: {0}")]
    NormalizerFailed(String),

    /// A dispatch hook intentionally stopped a normalized tool dispatch loop.
    #[error("tool dispatch terminated: {0}")]
    ToolDispatchTerminated(String),

    /// A budget or accounting hook failed while evaluating dispatch policy.
    #[error("budget failed: {0}")]
    BudgetFailed(String),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

/// Registry of named [`Tool`]s. Cheap to clone (Arc-backed).
#[derive(Clone, Default)]
pub struct ToolRegistry {
    inner: Arc<DashMap<ToolName, Arc<dyn Tool>>>,
    /// Optional whitelist applied at lookup time. `None` = unrestricted;
    /// `Some(set)` = only tools whose name appears in the set are visible
    /// through [`Self::get`]/[`Self::invoke`]. Used to scope a registry
    /// down to an agent's authorised tool surface without copying the
    /// underlying map.
    allowed: Option<Arc<std::collections::HashSet<ToolName>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.schema().name;
        self.inner.insert(name, tool);
    }

    /// Return a new registry view restricted to `names`. The underlying
    /// tools are shared; only the whitelist differs.
    pub fn scoped<I, S>(&self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let allowed: std::collections::HashSet<String> =
            names.into_iter().map(Into::into).collect();
        Self {
            inner: self.inner.clone(),
            allowed: Some(Arc::new(allowed)),
        }
    }

    fn is_authorised(&self, name: &str) -> bool {
        match &self.allowed {
            None => true,
            Some(set) => set.contains(name),
        }
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn Tool>, KernelError> {
        if !self.is_authorised(name) {
            return Err(KernelError::ToolNotAuthorised(name.to_string()));
        }
        self.inner
            .get(name)
            .map(|t| t.clone())
            .ok_or_else(|| KernelError::ToolNotFound(name.to_string()))
    }

    /// Convenience: look up `name` and invoke it.
    pub async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, KernelError> {
        let tool = self.get(name)?;
        tool.invoke(args).await
    }

    pub fn len(&self) -> usize {
        match &self.allowed {
            None => self.inner.len(),
            Some(set) => self.inner.iter().filter(|e| set.contains(e.key())).count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot of every visible tool's schema. Honours the `allowed`
    /// whitelist when present. Used by the MCP loopback transport to
    /// surface a server-side registry to a client.
    pub fn schemas(&self) -> Vec<crate::tool::ToolSchema> {
        let mut schemas: Vec<_> = self
            .inner
            .iter()
            .filter(|e| self.is_authorised(e.key()))
            .map(|e| e.value().schema())
            .collect();
        schemas.sort_by(|left, right| left.name.cmp(&right.name));
        schemas
    }

    /// Deterministic catalog snapshot of every visible tool.
    ///
    /// Today the execution contract and the discovery descriptor are the same
    /// [`ToolSchema`](crate::tool::ToolSchema). This method exists so callers
    /// can ask for a catalog snapshot without depending on implementation
    /// details of registry storage or iteration order.
    pub fn descriptors(&self) -> Vec<crate::tool::ToolSchema> {
        self.schemas()
    }
}

/// Registry of named [`Skill`]s. Identical structure to [`ToolRegistry`].
#[derive(Clone, Default)]
pub struct SkillRegistry {
    inner: Arc<DashMap<SkillId, Arc<dyn Skill>>>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, skill: Arc<dyn Skill>) {
        let id = skill.id().to_string();
        self.inner.insert(id, skill);
    }

    pub fn get(&self, id: &str) -> Result<Arc<dyn Skill>, KernelError> {
        self.inner
            .get(id)
            .map(|s| s.clone())
            .ok_or_else(|| KernelError::SkillNotFound(id.to_string()))
    }

    /// Resolve a list of skill ids in declared order. Errors on the first
    /// missing id. Used by `GenericAgent` to build its skill chain at
    /// construction.
    pub fn resolve_chain<I, S>(&self, ids: I) -> Result<Vec<Arc<dyn Skill>>, KernelError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        ids.into_iter().map(|id| self.get(id.as_ref())).collect()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Deterministic catalog snapshot of every registered skill.
    pub fn descriptors(&self) -> Vec<SkillDescriptor> {
        let mut descriptors: Vec<_> = self
            .inner
            .iter()
            .map(|entry| SkillDescriptor {
                id: entry.key().clone(),
                description: entry.value().description().to_string(),
            })
            .collect();
        descriptors.sort_by(|left, right| left.id.cmp(&right.id));
        descriptors
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tool::{LocalTool, ToolSchema};
    use crate::{
        InvestigationContext, KernelError, Skill, SkillOutcome, SkillRegistry, Tool, ToolRegistry,
    };
    use async_trait::async_trait;
    use serde_json::json;

    fn echo_tool(name: &str) -> Arc<dyn Tool> {
        let schema = ToolSchema {
            name: name.into(),
            description: "echo".into(),
            args_schema: json!({}),
            result_schema: json!({}),
        };
        Arc::new(LocalTool::new(schema, |v| async move { Ok(v) }))
    }

    #[tokio::test]
    async fn tool_registry_authorisation() {
        let reg = ToolRegistry::new();
        reg.register(echo_tool("a"));
        reg.register(echo_tool("b"));

        // Unrestricted view sees both.
        assert!(reg.get("a").is_ok());
        assert!(reg.get("b").is_ok());

        // Scoped view only sees `a`.
        let scoped = reg.scoped(["a"]);
        assert!(scoped.get("a").is_ok());
        match scoped.get("b") {
            Err(KernelError::ToolNotAuthorised(name)) => assert_eq!(name, "b"),
            _ => panic!("expected ToolNotAuthorised"),
        }

        // Invocation works through the scoped view for authorised tools.
        let out = scoped.invoke("a", json!({"x": 1})).await.unwrap();
        assert_eq!(out, json!({"x": 1}));
    }

    #[tokio::test]
    async fn tool_registry_missing() {
        let reg = ToolRegistry::new();
        match reg.get("missing") {
            Err(KernelError::ToolNotFound(name)) => assert_eq!(name, "missing"),
            _ => panic!("expected ToolNotFound"),
        }
    }

    #[test]
    fn tool_registry_descriptors_are_sorted_and_scoped() {
        let reg = ToolRegistry::new();
        reg.register(echo_tool("zeta.tool"));
        reg.register(echo_tool("alpha.tool"));
        reg.register(echo_tool("middle.tool"));

        let names: Vec<_> = reg
            .descriptors()
            .into_iter()
            .map(|schema| schema.name)
            .collect();
        assert_eq!(names, vec!["alpha.tool", "middle.tool", "zeta.tool"]);

        let scoped = reg.scoped(["zeta.tool", "alpha.tool"]);
        let scoped_names: Vec<_> = scoped
            .descriptors()
            .into_iter()
            .map(|schema| schema.name)
            .collect();
        assert_eq!(scoped_names, vec!["alpha.tool", "zeta.tool"]);
    }

    struct DescribedSkill {
        id: &'static str,
        description: &'static str,
    }

    #[async_trait]
    impl Skill for DescribedSkill {
        fn id(&self) -> &str {
            self.id
        }

        fn description(&self) -> &str {
            self.description
        }

        async fn execute(
            &self,
            _ctx: &mut InvestigationContext,
            _tools: &ToolRegistry,
        ) -> Result<SkillOutcome, KernelError> {
            Ok(SkillOutcome::noop())
        }
    }

    #[test]
    fn skill_registry_descriptors_are_sorted() {
        let reg = SkillRegistry::new();
        reg.register(Arc::new(DescribedSkill {
            id: "zeta.skill",
            description: "last",
        }));
        reg.register(Arc::new(DescribedSkill {
            id: "alpha.skill",
            description: "first",
        }));

        let descriptors = reg.descriptors();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].id, "alpha.skill");
        assert_eq!(descriptors[0].description, "first");
        assert_eq!(descriptors[1].id, "zeta.skill");
        assert_eq!(descriptors[1].description, "last");
    }
}
