//! [`Tool`] — the only side-effectful interface available to skills and agents.
//!
//! A [`Tool`] is a typed, named, async function with a JSON-Schema-compatible
//! signature. Two transports satisfy the trait today: [`LocalTool`] (a closure
//! over a Rust async fn) and — under the `mcp` feature in a later phase — a
//! remote MCP server. Skills never know the difference.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::registry::KernelError;

/// Stable, registry-unique identifier for a tool (e.g. `"grammar.query"`,
/// `"memory.lookup"`, `"sampler.expand"`).
pub type ToolName = String;

/// Lightweight description of a tool's I/O contract. The `args_schema` and
/// `result_schema` are JSON-Schema fragments; the LLM-facing rendering layer
/// uses them to generate `rig` / MCP tool definitions automatically. We do
/// **not** validate against them at the kernel — validation is the tool's
/// responsibility — but downstream MCP exporters need them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: ToolName,
    pub description: String,
    pub args_schema: Value,
    pub result_schema: Value,
}

/// A composable, side-effectful capability.
///
/// Implementations MUST be cheap to clone (typically `Arc`-wrapped state) so
/// the same tool instance can be referenced from multiple agents'
/// [`super::registry::ToolRegistry`] slices.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Return this tool's JSON-Schema-compatible contract.
    fn schema(&self) -> ToolSchema;

    /// Return this tool's registry name.
    fn name(&self) -> ToolName {
        self.schema().name
    }

    /// Invoke the tool with JSON arguments.
    async fn invoke(&self, args: Value) -> Result<Value, KernelError>;
}

/// Adapter that turns any `async Fn(Value) -> Result<Value, KernelError>`
/// into a [`Tool`]. Hosts can use this to surface existing async functions
/// to the kernel without writing a dedicated tool type.
pub struct LocalTool {
    schema: ToolSchema,
    #[allow(clippy::type_complexity)]
    f: Arc<
        dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, KernelError>> + Send>>
            + Send
            + Sync,
    >,
}

impl LocalTool {
    pub fn new<F, Fut>(schema: ToolSchema, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, KernelError>> + Send + 'static,
    {
        Self {
            schema,
            f: Arc::new(move |v| Box::pin(f(v))),
        }
    }
}

#[async_trait]
impl Tool for LocalTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn name(&self) -> ToolName {
        self.schema.name.clone()
    }

    async fn invoke(&self, args: Value) -> Result<Value, KernelError> {
        (self.f)(args).await
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use serde_json::json;

    #[tokio::test]
    async fn local_tool_roundtrip() {
        let schema = ToolSchema {
            name: "test.echo".into(),
            description: "echoes the input".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        };
        let tool = LocalTool::new(schema, |v| async move { Ok(v) });
        let out = tool.invoke(json!({"hello": "world"})).await.unwrap();
        assert_eq!(out, json!({"hello": "world"}));
        assert_eq!(tool.name(), "test.echo");
    }
}
