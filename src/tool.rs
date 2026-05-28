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
    /// Maximum serialized bytes retained for the entire bounded payload.
    pub max_total_bytes: usize,
    /// Optional redaction policy applied before size bounding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub redaction: Option<RedactionPolicy>,
}

impl Default for ToolResultEnvelopeConfig {
    fn default() -> Self {
        Self {
            max_string_chars: 4_000,
            max_array_items: 64,
            max_total_bytes: 256_000,
            redaction: None,
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

    /// Set the maximum serialized bytes retained for the whole payload.
    #[must_use]
    pub fn with_max_total_bytes(mut self, max_total_bytes: usize) -> Self {
        self.max_total_bytes = max_total_bytes;
        self
    }

    /// Set the redaction policy applied before truncation and total bounding.
    #[must_use]
    pub fn with_redaction_policy(mut self, redaction: RedactionPolicy) -> Self {
        self.redaction = Some(redaction);
        self
    }
}

/// One JSON Pointer redaction rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionRule {
    /// Canonical JSON Pointer to redact, for example `/secret/token`.
    pub pointer: String,
    /// Replacement string for this pointer. Falls back to the policy default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replacement: Option<String>,
}

/// JSON Pointer redaction policy for tool results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionPolicy {
    /// Pointers redacted before size bounding.
    pub deny: Vec<RedactionRule>,
    /// Optional allow-list. Values outside these pointers are redacted while
    /// ancestor containers are retained so allowed descendants can survive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow: Option<Vec<String>>,
    /// Default replacement used when a rule does not specify one.
    pub default_replacement: String,
}

impl RedactionPolicy {
    /// Create a deny-list policy from JSON Pointers.
    #[must_use]
    pub fn deny_pointers<I, S>(pointers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            deny: pointers
                .into_iter()
                .map(|pointer| RedactionRule {
                    pointer: pointer.into(),
                    replacement: None,
                })
                .collect(),
            allow: None,
            default_replacement: "[redacted]".to_string(),
        }
    }

    /// Create an allow-list policy from JSON Pointers.
    #[must_use]
    pub fn allow_pointers<I, S>(pointers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            deny: Vec::new(),
            allow: Some(pointers.into_iter().map(Into::into).collect()),
            default_replacement: "[redacted]".to_string(),
        }
    }

    /// Set the default replacement string.
    #[must_use]
    pub fn with_default_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.default_replacement = replacement.into();
        self
    }

    /// Set a replacement string for one deny-list pointer.
    #[must_use]
    pub fn with_replacement(
        mut self,
        pointer: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        let pointer = pointer.into();
        let replacement = replacement.into();
        for rule in &mut self.deny {
            if rule.pointer == pointer {
                rule.replacement = Some(replacement);
                return self;
            }
        }
        self.deny.push(RedactionRule {
            pointer,
            replacement: Some(replacement),
        });
        self
    }
}

/// Reason an individual payload segment was omitted from a tool result envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultOmissionReason {
    /// A string exceeded `max_string_chars`.
    StringChars,
    /// An array exceeded `max_array_items` or the total byte budget.
    ArrayItems,
    /// An object field or scalar exceeded `max_total_bytes`.
    TotalBytes,
}

/// One omitted payload segment and its opaque continuation token.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OmittedSegment {
    /// JSON Pointer to the omitted segment in the original payload.
    pub pointer: String,
    /// Machine-readable omission reason.
    pub reason: ToolResultOmissionReason,
    /// Opaque continuation token for callers that can page the original result.
    pub page_token: String,
}

/// Decoded continuation token metadata for a bounded tool result segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultPageToken {
    /// JSON Pointer to the omitted segment in the original payload.
    pub pointer: String,
    /// Machine-readable omission reason.
    pub reason: ToolResultOmissionReason,
    /// Limit that caused the omission.
    pub limit: usize,
}

/// Decode a [`ToolResultEnvelope`] page token produced by this crate.
#[must_use]
pub fn decode_tool_result_page_token(token: &str) -> Option<ToolResultPageToken> {
    let payload = token.strip_prefix("v1:")?;
    let bytes = decode_hex(payload)?;
    let text = String::from_utf8(bytes).ok()?;
    serde_json::from_str(&text).ok()
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
    /// Total whole values omitted while enforcing the total byte budget.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub omitted_values: usize,
    /// Total values replaced by the redaction policy.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub redacted_values: usize,
    /// Every omitted segment recorded during bounding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub omitted_segments: Vec<OmittedSegment>,
    /// Stable follow-up token describing the first omitted segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_token: Option<String>,
}

impl ToolResultEnvelope {
    /// Bound `payload` according to `config` and return truncation metadata.
    #[must_use]
    pub fn bound(payload: Value, config: &ToolResultEnvelopeConfig) -> Self {
        let mut state = ToolResultEnvelopeState::default();
        let payload = bound_value(payload, config, &mut state, "");
        let payload = bound_total_bytes(payload, config, &mut state, "");
        Self {
            payload,
            truncated: state.omitted_chars > 0
                || state.omitted_items > 0
                || state.omitted_values > 0,
            omitted_chars: state.omitted_chars,
            omitted_items: state.omitted_items,
            omitted_values: state.omitted_values,
            redacted_values: state.redacted_values,
            omitted_segments: state.omitted_segments,
            page_token: state.page_token,
        }
    }
}

fn is_zero(value: &usize) -> bool {
    *value == 0
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
    omitted_values: usize,
    redacted_values: usize,
    omitted_segments: Vec<OmittedSegment>,
    page_token: Option<String>,
}

impl ToolResultEnvelopeState {
    fn record_omission(&mut self, pointer: &str, reason: ToolResultOmissionReason, limit: usize) {
        let page_token = page_token(pointer, reason, limit);
        if self.page_token.is_none() {
            self.page_token = Some(page_token.clone());
        }
        self.omitted_segments.push(OmittedSegment {
            pointer: pointer.to_string(),
            reason,
            page_token,
        });
    }
}

fn bound_value(
    value: Value,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    if let Some(redaction) = &config.redaction
        && let Some(replacement) = redaction.replacement_for(pointer)
    {
        state.redacted_values = state.redacted_values.saturating_add(1);
        return Value::String(replacement);
    }

    match value {
        Value::String(text) => bound_string(text, config, state, pointer),
        Value::Array(items) => bound_array(items, config, state, pointer),
        Value::Object(fields) => bound_object(fields, config, state, pointer),
        scalar => scalar,
    }
}

fn bound_string(
    text: String,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    let total_chars = text.chars().count();
    if total_chars <= config.max_string_chars {
        return Value::String(text);
    }
    state.omitted_chars = state
        .omitted_chars
        .saturating_add(total_chars.saturating_sub(config.max_string_chars));
    state.record_omission(
        pointer,
        ToolResultOmissionReason::StringChars,
        config.max_string_chars,
    );
    Value::String(text.chars().take(config.max_string_chars).collect())
}

fn bound_array(
    items: Vec<Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    let total_items = items.len();
    if total_items > config.max_array_items {
        state.omitted_items = state
            .omitted_items
            .saturating_add(total_items.saturating_sub(config.max_array_items));
        state.record_omission(
            pointer,
            ToolResultOmissionReason::ArrayItems,
            config.max_array_items,
        );
    }
    Value::Array(
        items
            .into_iter()
            .enumerate()
            .take(config.max_array_items)
            .map(|(index, item)| {
                let child = child_pointer(pointer, &index.to_string());
                bound_value(item, config, state, &child)
            })
            .collect(),
    )
}

fn bound_object(
    fields: Map<String, Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| {
                let child = child_pointer(pointer, &key);
                (key, bound_value(value, config, state, &child))
            })
            .collect(),
    )
}

impl RedactionPolicy {
    fn replacement_for(&self, pointer: &str) -> Option<String> {
        for rule in &self.deny {
            if rule.pointer == pointer {
                return Some(
                    rule.replacement
                        .clone()
                        .unwrap_or_else(|| self.default_replacement.clone()),
                );
            }
        }

        let Some(allow) = &self.allow else {
            return None;
        };
        if allow
            .iter()
            .any(|allowed| pointer_matches(pointer, allowed))
        {
            return None;
        }
        Some(self.default_replacement.clone())
    }
}

fn pointer_matches(pointer: &str, allowed: &str) -> bool {
    pointer == allowed || is_descendant(pointer, allowed) || is_descendant(allowed, pointer)
}

fn is_descendant(pointer: &str, ancestor: &str) -> bool {
    if ancestor.is_empty() {
        return !pointer.is_empty();
    }
    let prefix = format!("{ancestor}/");
    pointer.starts_with(&prefix)
}

fn child_pointer(parent: &str, child: &str) -> String {
    let escaped = escape_pointer_segment(child);
    if parent.is_empty() {
        format!("/{escaped}")
    } else {
        format!("{parent}/{escaped}")
    }
}

fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

fn page_token(pointer: &str, reason: ToolResultOmissionReason, limit: usize) -> String {
    let payload = ToolResultPageToken {
        pointer: pointer.to_string(),
        reason,
        limit,
    };
    match serde_json::to_string(&payload) {
        Ok(serialized) => format!("v1:{}", encode_hex(serialized.as_bytes())),
        Err(_) => "v1:".to_string(),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(input: &str) -> Option<Vec<u8>> {
    let mut chars = input.chars();
    let mut bytes = Vec::new();
    loop {
        let Some(high) = chars.next() else {
            return Some(bytes);
        };
        let low = chars.next()?;
        let high = hex_value(high)?;
        let low = hex_value(low)?;
        bytes.push(high.saturating_mul(16).saturating_add(low));
    }
}

fn hex_value(character: char) -> Option<u8> {
    match character {
        '0'..='9' => Some(character as u8 - b'0'),
        'a'..='f' => Some(character as u8 - b'a' + 10),
        'A'..='F' => Some(character as u8 - b'A' + 10),
        _ => None,
    }
}

fn bound_total_bytes(
    value: Value,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    if serialized_len(&value) <= config.max_total_bytes {
        return value;
    }

    match value {
        Value::Object(fields) => bound_object_total_bytes(fields, config, state, pointer),
        Value::Array(items) => bound_array_total_bytes(items, config, state, pointer),
        Value::String(text) => bound_string_total_bytes(text, config, state, pointer),
        scalar => {
            state.omitted_values = state.omitted_values.saturating_add(1);
            state.record_omission(
                pointer,
                ToolResultOmissionReason::TotalBytes,
                config.max_total_bytes,
            );
            scalar
        }
    }
}

fn bound_object_total_bytes(
    fields: Map<String, Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    let mut retained = Map::new();
    for (key, value) in fields {
        let child = child_pointer(pointer, &key);
        let mut candidate = retained.clone();
        candidate.insert(key.clone(), value.clone());
        if serialized_len(&Value::Object(candidate)) <= config.max_total_bytes {
            retained.insert(key, value);
        } else {
            state.omitted_values = state.omitted_values.saturating_add(1);
            state.record_omission(
                &child,
                ToolResultOmissionReason::TotalBytes,
                config.max_total_bytes,
            );
        }
    }
    Value::Object(retained)
}

fn bound_array_total_bytes(
    items: Vec<Value>,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    let mut retained = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        let mut candidate = retained.clone();
        candidate.push(item.clone());
        if serialized_len(&Value::Array(candidate)) <= config.max_total_bytes {
            retained.push(item);
        } else {
            state.omitted_items = state.omitted_items.saturating_add(1);
            let child = child_pointer(pointer, &index.to_string());
            state.record_omission(
                &child,
                ToolResultOmissionReason::TotalBytes,
                config.max_total_bytes,
            );
        }
    }
    Value::Array(retained)
}

fn bound_string_total_bytes(
    text: String,
    config: &ToolResultEnvelopeConfig,
    state: &mut ToolResultEnvelopeState,
    pointer: &str,
) -> Value {
    let mut retained = String::new();
    for character in text.chars() {
        let mut candidate = retained.clone();
        candidate.push(character);
        if serialized_len(&Value::String(candidate)) <= config.max_total_bytes {
            retained.push(character);
        } else {
            state.omitted_chars = state.omitted_chars.saturating_add(1);
        }
    }
    if retained.chars().count() < text.chars().count() {
        state.record_omission(
            pointer,
            ToolResultOmissionReason::TotalBytes,
            config.max_total_bytes,
        );
    }
    Value::String(retained)
}

fn serialized_len(value: &Value) -> usize {
    match serde_json::to_string(value) {
        Ok(serialized) => serialized.len(),
        Err(_) => usize::MAX,
    }
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
        assert_eq!(
            decode_tool_result_page_token(envelope.page_token.as_deref().unwrap()),
            Some(ToolResultPageToken {
                pointer: "/body".to_string(),
                reason: ToolResultOmissionReason::StringChars,
                limit: 3,
            })
        );
        assert_eq!(envelope.omitted_segments.len(), 1);
        assert!(
            envelope
                .omitted_segments
                .iter()
                .any(|segment| segment.pointer == "/body")
        );
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
        assert_eq!(
            decode_tool_result_page_token(envelope.page_token.as_deref().unwrap()),
            Some(ToolResultPageToken {
                pointer: "/rows".to_string(),
                reason: ToolResultOmissionReason::ArrayItems,
                limit: 2,
            })
        );
        assert_eq!(envelope.omitted_segments.len(), 1);
        assert!(
            envelope
                .omitted_segments
                .iter()
                .any(|segment| segment.pointer == "/rows")
        );
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

    #[test]
    fn tool_result_envelope_redacts_before_truncation() {
        let config = ToolResultEnvelopeConfig::new(4).with_redaction_policy(
            RedactionPolicy::deny_pointers(["/secret"]).with_replacement("/secret", "safe"),
        );
        let envelope = ToolResultEnvelope::bound(
            json!({"public": "abcdef", "secret": "should-not-leak"}),
            &config,
        );

        assert_eq!(
            envelope.payload,
            json!({"public": "abcd", "secret": "safe"})
        );
        assert_eq!(envelope.redacted_values, 1);
        assert_eq!(envelope.omitted_chars, 2);
        assert_eq!(
            decode_tool_result_page_token(envelope.page_token.as_deref().unwrap()),
            Some(ToolResultPageToken {
                pointer: "/public".to_string(),
                reason: ToolResultOmissionReason::StringChars,
                limit: 4,
            })
        );
    }

    #[test]
    fn tool_result_envelope_total_budget_drops_fields_with_path_tokens() {
        let config = ToolResultEnvelopeConfig::new(100).with_max_total_bytes(24);
        let envelope = ToolResultEnvelope::bound(
            json!({"a": "small", "b": "also-small", "c": "extra"}),
            &config,
        );

        assert!(envelope.truncated);
        assert!(envelope.omitted_values > 0);
        assert!(envelope.payload.get("a").is_some());
        assert!(envelope.omitted_segments.iter().any(|segment| {
            segment.reason == ToolResultOmissionReason::TotalBytes
                && decode_tool_result_page_token(&segment.page_token).is_some_and(|token| {
                    token.reason == ToolResultOmissionReason::TotalBytes && token.limit == 24
                })
        }));
    }
}
