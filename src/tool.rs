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
use serde_json::{Map, Value};

use crate::registry::KernelError;

/// Stable, registry-unique identifier for a tool (e.g. `"grammar.query"`,
/// `"memory.lookup"`, `"sampler.expand"`).
pub type ToolName = String;

/// Lightweight description of a tool's I/O contract. The `args_schema` and
/// `result_schema` are JSON-Schema fragments; the LLM-facing rendering layer
/// uses them to generate `rig` / MCP tool definitions automatically. We do
/// **not** validate against them at the kernel — validation is the tool's
/// responsibility — but downstream MCP exporters need them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: ToolName,
    pub description: String,
    pub args_schema: Value,
    pub result_schema: Value,
}

/// Configuration for bounding large tool results before they enter a model
/// turn, trace record, or MCP response cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultEnvelopeConfig {
    /// Maximum characters retained for any string value.
    pub max_string_chars: usize,
    /// Maximum items retained for any array value.
    pub max_array_items: usize,
}

impl Default for ToolResultEnvelopeConfig {
    fn default() -> Self {
        Self {
            max_string_chars: 4_000,
            max_array_items: 64,
        }
    }
}

impl ToolResultEnvelopeConfig {
    /// Build a config with a string character limit and otherwise default
    /// limits.
    #[must_use]
    pub fn new(max_string_chars: usize) -> Self {
        Self {
            max_string_chars,
            ..Self::default()
        }
    }

    /// Set the maximum retained array items.
    #[must_use]
    pub fn with_max_array_items(mut self, max_array_items: usize) -> Self {
        self.max_array_items = max_array_items;
        self
    }
}

/// Tool result plus deterministic truncation metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultEnvelope {
    /// Possibly bounded result payload.
    pub payload: Value,
    /// Whether any value was truncated or omitted.
    pub truncated: bool,
    /// Total string characters omitted while bounding the payload.
    pub omitted_chars: usize,
    /// Total array items omitted while bounding the payload.
    pub omitted_items: usize,
    /// Stable follow-up token describing the first omitted segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_token: Option<String>,
}

impl ToolResultEnvelope {
    /// Bound `payload` according to `config` and return truncation metadata.
    #[must_use]
    pub fn bound(payload: Value, config: &ToolResultEnvelopeConfig) -> Self {
        let mut state = ToolResultEnvelopeState::default();
        let payload = bound_value(payload, config, &mut state);
        Self {
            payload,
            truncated: state.omitted_chars > 0 || state.omitted_items > 0,
            omitted_chars: state.omitted_chars,
            omitted_items: state.omitted_items,
            page_token: state.page_token,
        }
    }
}

/// Bound `payload` with the default [`ToolResultEnvelopeConfig`].
#[must_use]
pub fn bound_tool_result(payload: Value) -> ToolResultEnvelope {
    ToolResultEnvelope::bound(payload, &ToolResultEnvelopeConfig::default())
}

#[derive(Default)]
struct ToolResultEnvelopeState {
    omitted_chars: usize,
    omitted_items: usize,
    page_token: Option<String>,
}

fn bound_value(
    value: Value,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
) -> Value {
    match value {
        Value::String(text) => bound_string(text, config, state),
        Value::Array(items) => bound_array(items, config, state),
        Value::Object(fields) => bound_object(fields, config, state),
        scalar => scalar,
    }
}

fn bound_string(
    text: String,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
) -> Value {
    let total_chars = text.chars().count();
    if total_chars <= config.max_string_chars {
        return Value::String(text);
    }
    state.omitted_chars = state
        .omitted_chars
        .saturating_add(total_chars.saturating_sub(config.max_string_chars));
    if state.page_token.is_none() {
        state.page_token = Some(format!("chars:{}", config.max_string_chars));
    }
    Value::String(text.chars().take(config.max_string_chars).collect())
}

fn bound_array(
    items: Vec<Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
) -> Value {
    let total_items = items.len();
    if total_items > config.max_array_items {
        state.omitted_items = state
            .omitted_items
            .saturating_add(total_items.saturating_sub(config.max_array_items));
        if state.page_token.is_none() {
            state.page_token = Some(format!("items:{}", config.max_array_items));
        }
    }
    Value::Array(
        items
            .into_iter()
            .take(config.max_array_items)
            .map(|item| bound_value(item, config, state))
            .collect(),
    )
}

fn bound_object(
    fields: Map<String, Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
) -> Value {
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| (key, bound_value(value, config, state)))
            .collect(),
    )
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

    #[test]
    fn tool_result_envelope_bounds_large_strings() {
        let envelope =
            ToolResultEnvelope::bound(json!({"body": "abcdef"}), &ToolResultEnvelopeConfig::new(3));

        assert_eq!(envelope.payload, json!({"body": "abc"}));
        assert!(envelope.truncated);
        assert_eq!(envelope.omitted_chars, 3);
        assert_eq!(envelope.page_token.as_deref(), Some("chars:3"));
    }

    #[test]
    fn tool_result_envelope_bounds_arrays() {
        let envelope = ToolResultEnvelope::bound(
            json!({"rows": [1, 2, 3, 4]}),
            &ToolResultEnvelopeConfig::new(100).with_max_array_items(2),
        );

        assert_eq!(envelope.payload, json!({"rows": [1, 2]}));
        assert!(envelope.truncated);
        assert_eq!(envelope.omitted_items, 2);
        assert_eq!(envelope.page_token.as_deref(), Some("items:2"));
    }

    #[test]
    fn tool_result_envelope_leaves_small_payloads_unchanged() {
        let payload = json!({"ok": true, "rows": ["a"]});
        let envelope = ToolResultEnvelope::bound(
            payload.clone(),
            &ToolResultEnvelopeConfig::new(100).with_max_array_items(10),
        );

        assert_eq!(envelope.payload, payload);
        assert!(!envelope.truncated);
        assert_eq!(envelope.omitted_chars, 0);
        assert_eq!(envelope.omitted_items, 0);
        assert_eq!(envelope.page_token, None);
    }
}
