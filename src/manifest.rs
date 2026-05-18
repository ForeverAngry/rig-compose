//! Portable agent manifest schema.
//!
//! [`AgentManifest`] captures the domain-neutral pieces most agent hosts
//! need to assemble at startup: instructions, model selector, local tool
//! references, MCP server declarations, delegates, and knowledge metadata.
//! Product crates can embed it with `#[serde(flatten)]` and add their own
//! runtime knobs without re-defining the common shape.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::{
    DelegateRegistry, DelegateTool, Instructions, KernelError, LocalTool, Tool, ToolRegistry,
    ToolSchema,
};

/// Domain-neutral manifest fragment for a single agent.
///
/// # Example
///
/// Parse a portable manifest from YAML/JSON and materialise local tools plus
/// delegate stubs against an empty host registry. The kernel never spawns
/// MCP servers itself; that wiring is host-owned.
///
/// ```
/// use rig_compose::{AgentManifest, ToolRegistry, manifest::materialize_local_and_delegate_tools};
///
/// let yaml = r#"
/// name: portable-agent
/// instructions:
///   system_prompt: "You are portable."
/// model:
///   provider: openai_compatible
///   base_url: http://localhost:11434/v1
///   name: llama3
/// delegates:
///   - name: child_agent
///     description: A child specialist
/// "#;
///
/// let manifest = AgentManifest::from_yaml(yaml).expect("parse manifest");
/// assert_eq!(manifest.name.as_deref(), Some("portable-agent"));
///
/// let host_tools = ToolRegistry::new();
/// let registry = materialize_local_and_delegate_tools(
///     &manifest.tools,
///     &manifest.delegates,
///     &host_tools,
/// )
/// .expect("materialise tools");
/// assert!(registry.get("child_agent").is_ok());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentManifest {
    /// Optional human-readable name (logged on startup; otherwise host-defined).
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub instructions: Option<InstructionsSpec>,
    #[serde(default)]
    pub model: Option<ModelSpec>,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
    #[serde(default)]
    pub delegates: Vec<DelegateSpec>,
    #[serde(default)]
    pub knowledge: Option<KnowledgeSpec>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstructionsSpec {
    pub system_prompt: String,
    #[serde(default)]
    pub response_schema: Option<serde_json::Value>,
    /// `[user_message, assistant_reply]` pairs.
    #[serde(default)]
    pub examples: Vec<[String; 2]>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl From<InstructionsSpec> for Instructions {
    fn from(s: InstructionsSpec) -> Self {
        let mut instructions = Instructions::new(s.system_prompt);
        if let Some(schema) = s.response_schema {
            instructions = instructions.with_response_schema(schema);
        }
        for [user, assistant] in s.examples {
            instructions = instructions.with_example(user, assistant);
        }
        if !s.metadata.is_null() {
            instructions = instructions.with_metadata(s.metadata);
        }
        instructions
    }
}

/// LLM provider selector. The host decides how to materialise it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider", rename_all = "snake_case")]
pub enum ModelSpec {
    /// `OPENAI_API_KEY` from the environment, OpenAI-hosted models.
    Openai { name: String },
    /// Any OpenAI-compatible Chat Completions endpoint (Ollama, vLLM,
    /// LM Studio, etc.). `api_key` may be empty for endpoints that ignore it.
    OpenaiCompatible {
        base_url: String,
        #[serde(default)]
        api_key: String,
        name: String,
    },
}

/// One tool entry. A `local` reference points at a tool the host has
/// pre-registered in the `host_tools` registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolSpec {
    Local { name: String },
}

/// One MCP server to spawn over stdio. The actual transport lives in a
/// host or companion crate such as `rig-mcp`; this type is only config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerSpec {
    /// Symbolic name for telemetry. Tool names are typically namespaced
    /// by hosts as `<name>.<tool>` after discovery.
    pub name: String,
    /// Argv the host can spawn (e.g. `["mcp-server-fs", "/data"]`).
    /// First element is the program; the rest are args.
    pub command: Vec<String>,
}

/// How a delegate is expected to be wired by the host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegateKind {
    /// No executor is required; materialisation creates a helpful stub.
    #[default]
    Stub,
    /// A child agent running in the same process, registered in a
    /// [`DelegateRegistry`] by `agent` (or by `name` when `agent` is omitted).
    InProcess,
    /// Reserved for hosts that resolve the delegate through a remote runtime.
    Remote,
    /// Reserved for hosts that expose the delegate through MCP.
    Mcp,
}

/// Another agent the model can delegate to, exposed as a tool with a
/// description. Recursive assembly is intentionally host-owned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegateSpec {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub kind: DelegateKind,
    /// Host registry key for this delegate's executor. Defaults to `name`.
    #[serde(default)]
    pub agent: Option<String>,
    /// Path to a child manifest. Optional; a host may wire the delegate
    /// implementation through its own tool catalog instead.
    #[serde(default)]
    pub manifest: Option<String>,
}

impl DelegateSpec {
    pub fn executor_key(&self) -> &str {
        self.agent.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeSpec {
    /// Optional path to domain-specific pattern data. Hosts decide the
    /// concrete pattern type and loader.
    #[serde(default)]
    pub patterns_path: Option<String>,
    /// Free-form metadata bag for host-specific knowledge backends.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("yaml parse: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("missing host tool '{0}' referenced by manifest")]
    MissingHostTool(String),
    #[error("missing delegate executor '{0}' referenced by manifest")]
    MissingDelegateExecutor(String),
}

impl AgentManifest {
    /// Parse a manifest from a YAML or JSON string. YAML is a superset
    /// of JSON so both work.
    pub fn from_yaml(s: &str) -> Result<Self, ManifestError> {
        Ok(serde_yaml::from_str(s)?)
    }

    /// Read a manifest from disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self, ManifestError> {
        let raw = std::fs::read_to_string(path)?;
        Self::from_yaml(&raw)
    }
}

/// Copy local tools selected by `tools` from the host registry and append
/// delegate stubs. MCP spawning is host-owned to avoid coupling this
/// kernel crate to any transport implementation.
pub fn materialize_local_and_delegate_tools(
    tools: &[ToolSpec],
    delegates: &[DelegateSpec],
    host_tools: &ToolRegistry,
) -> Result<ToolRegistry, ManifestError> {
    materialize_local_and_delegate_tools_with_delegates(
        tools,
        delegates,
        host_tools,
        &DelegateRegistry::new(),
    )
}

/// Copy local tools, then materialise delegates. When a delegate has an
/// executor registered in `delegate_executors`, it becomes a real
/// [`DelegateTool`]. Otherwise, `kind: stub`/remote/MCP delegates become
/// stubs and `kind: in_process` fails fast.
pub fn materialize_local_and_delegate_tools_with_delegates(
    tools: &[ToolSpec],
    delegates: &[DelegateSpec],
    host_tools: &ToolRegistry,
    delegate_executors: &DelegateRegistry,
) -> Result<ToolRegistry, ManifestError> {
    let agent_tools = ToolRegistry::new();
    for tool in tools {
        match tool {
            ToolSpec::Local { name } => {
                let registered = host_tools
                    .get(name)
                    .map_err(|_| ManifestError::MissingHostTool(name.clone()))?;
                agent_tools.register(registered);
            }
        }
    }
    for delegate in delegates {
        if let Some(executor) = delegate_executors.get(delegate.executor_key()) {
            agent_tools.register(Arc::new(DelegateTool::new(
                delegate.name.clone(),
                delegate.description.clone(),
                executor,
            )));
        } else if delegate.kind == DelegateKind::InProcess {
            return Err(ManifestError::MissingDelegateExecutor(
                delegate.executor_key().to_string(),
            ));
        } else {
            agent_tools.register(delegate_stub(delegate));
        }
    }
    Ok(agent_tools)
}

/// Description-bearing tool stub for a delegate. Until the host wires
/// real cross-agent dispatch, invoking the tool returns an explanatory
/// JSON object so the model still has signal that the call landed but
/// the runtime has not been provisioned yet.
pub fn delegate_stub(spec: &DelegateSpec) -> Arc<dyn Tool> {
    let name = spec.name.clone();
    let description = spec.description.clone();
    let schema = ToolSchema {
        name: name.clone(),
        description,
        args_schema: json!({"type": "object"}),
        result_schema: json!({"type": "object"}),
    };
    let stub_name = name.clone();
    Arc::new(LocalTool::new(schema, move |args| {
        let stub_name = stub_name.clone();
        async move {
            Ok(json!({
                "status": "delegate_unwired",
                "delegate": stub_name,
                "args": args,
                "hint": "host has not wired delegate execution; see manifest delegates",
            }))
        }
    }))
}

impl From<ManifestError> for KernelError {
    fn from(value: ManifestError) -> Self {
        KernelError::ToolFailed(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const MIN_YAML: &str = r#"{
    "name": "portable-agent",
    "instructions": { "system_prompt": "You are portable." },
    "model": {
        "provider": "openai_compatible",
        "base_url": "http://localhost:11434/v1",
        "name": "llama3"
    },
    "delegates": [
        { "name": "child_agent", "description": "A child specialist" },
        {
            "name": "local_child",
            "description": "An in-process child specialist",
            "kind": "in_process",
            "agent": "child"
        }
    ]
}"#;

    #[test]
    fn parses_portable_manifest() {
        let manifest = AgentManifest::from_yaml(MIN_YAML).expect("parse");
        assert_eq!(manifest.name.as_deref(), Some("portable-agent"));
        assert_eq!(manifest.delegates.len(), 2);
        assert_eq!(manifest.delegates[1].kind, DelegateKind::InProcess);
        assert_eq!(manifest.delegates[1].executor_key(), "child");
        assert!(matches!(
            manifest.model,
            Some(ModelSpec::OpenaiCompatible { .. })
        ));
    }

    #[tokio::test]
    async fn materializes_delegate_stub() {
        let manifest = AgentManifest::from_yaml(MIN_YAML).expect("parse");
        let registry = materialize_local_and_delegate_tools(
            &manifest.tools,
            &manifest.delegates[..1],
            &ToolRegistry::new(),
        )
        .expect("tools");
        let tool = registry.get("child_agent").expect("delegate");
        let out = tool.invoke(json!({"x": 1})).await.expect("invoke");
        assert_eq!(out["status"], "delegate_unwired");
        assert_eq!(out["delegate"], "child_agent");
    }

    #[test]
    fn missing_host_tool_is_an_error() {
        let host = ToolRegistry::new();
        let result = materialize_local_and_delegate_tools(
            &[ToolSpec::Local {
                name: "missing".into(),
            }],
            &[],
            &host,
        );
        let Err(err) = result else {
            panic!("expected missing host tool error");
        };
        assert!(matches!(err, ManifestError::MissingHostTool(ref n) if n == "missing"));
    }

    #[test]
    fn missing_in_process_delegate_executor_is_an_error() {
        let delegate = DelegateSpec {
            name: "child_agent".into(),
            description: "child".into(),
            kind: DelegateKind::InProcess,
            agent: Some("child".into()),
            manifest: None,
        };
        let result = materialize_local_and_delegate_tools_with_delegates(
            &[],
            &[delegate],
            &ToolRegistry::new(),
            &DelegateRegistry::new(),
        );
        let Err(err) = result else {
            panic!("expected missing delegate executor error");
        };
        assert!(matches!(err, ManifestError::MissingDelegateExecutor(ref n) if n == "child"));
    }
}
