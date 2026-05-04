//! [`ToolRegistry`] and [`SkillRegistry`] — composition surfaces.
//!
//! Both registries are append-only at runtime and indexed by name. Agents
//! reference entries by name and never own them; the same skill or tool
//! instance can be shared across any number of agents.

use std::sync::Arc;

use dashmap::DashMap;
use thiserror::Error;

use crate::skill::{Skill, SkillId};
use crate::tool::{Tool, ToolName};

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

    #[error("skill execution failed: {0}")]
    SkillFailed(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

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
        self.inner
            .iter()
            .filter(|e| self.is_authorised(e.key()))
            .map(|e| e.value().schema())
            .collect()
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::tool::{LocalTool, ToolSchema};
    use crate::{KernelError, Tool, ToolRegistry};
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
}
