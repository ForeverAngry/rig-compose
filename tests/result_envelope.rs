//! Validates `bound_tool_result` against the registry dispatch path that
//! agents and MCP-style transports both feed into.
//!
//! The waiting-room plan calls for bounding to be the supported truncation
//! surface for *all* tool results, regardless of transport. The plan adds:
//! "Use the helper from examples/harness code first; promote into broader
//! dispatch paths only after rig-mcp and local tools both validate it."
//! This fixture covers the local-tool half (rig-mcp pulls a published
//! release that re-exports `bound_tool_result`).
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]

use std::sync::Arc;

use rig_compose::{
    KernelError, LocalTool, RedactionPolicy, ToolInvocation, ToolRegistry, ToolResultEnvelope,
    ToolResultEnvelopeConfig, ToolResultOmissionReason, ToolSchema, bound_tool_result,
    decode_tool_result_page_token, dispatch_tool_invocations, dispatch_tool_invocations_bounded,
};
use serde_json::{Value, json};

fn oversized_tool() -> Arc<LocalTool> {
    Arc::new(LocalTool::new(
        ToolSchema {
            name: "diagnostics.big_payload".into(),
            description: "return a deterministically oversized payload".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({
                "type": "object",
                "properties": {
                    "blob": {"type": "string"},
                    "items": {"type": "array"}
                }
            }),
        },
        |_args| async move {
            let blob: String = std::iter::repeat_n('x', 10_000).collect();
            let items: Vec<Value> = (0..200_i64).map(|i| json!(i)).collect();
            Ok(json!({ "blob": blob, "items": items }))
        },
    ))
}

#[tokio::test]
async fn dispatch_then_bound_tool_result_clamps_oversized_payload() -> Result<(), KernelError> {
    let tools = ToolRegistry::new();
    tools.register(oversized_tool());

    let invocations = vec![ToolInvocation::new("diagnostics.big_payload", json!({}))?];
    let dispatched = dispatch_tool_invocations(&tools, &invocations).await?;
    assert_eq!(dispatched.len(), 1);

    let raw = dispatched[0].output.clone();
    assert_eq!(
        raw["blob"].as_str().expect("blob").chars().count(),
        10_000,
        "registry must not clamp; bounding is an explicit follow-up step"
    );

    let envelope = bound_tool_result(raw);
    assert!(envelope.truncated);
    assert!(envelope.omitted_chars > 0);
    assert!(envelope.omitted_items > 0);
    assert!(envelope.page_token.is_some());
    assert!(envelope.omitted_segments.iter().any(|segment| {
        segment.pointer == "/blob" && segment.reason == ToolResultOmissionReason::StringChars
    }));
    assert!(envelope.omitted_segments.iter().any(|segment| {
        segment.pointer == "/items" && segment.reason == ToolResultOmissionReason::ArrayItems
    }));
    assert_eq!(
        envelope.payload["blob"]
            .as_str()
            .expect("bounded blob")
            .chars()
            .count(),
        4_000
    );
    assert_eq!(
        envelope.payload["items"]
            .as_array()
            .expect("bounded items")
            .len(),
        64
    );
    Ok(())
}

#[tokio::test]
async fn custom_envelope_config_round_trips_through_serde() -> Result<(), KernelError> {
    let tools = ToolRegistry::new();
    tools.register(oversized_tool());

    let invocations = vec![ToolInvocation::new("diagnostics.big_payload", json!({}))?];
    let dispatched = dispatch_tool_invocations(&tools, &invocations).await?;

    let config = ToolResultEnvelopeConfig::new(128).with_max_array_items(8);
    let envelope = ToolResultEnvelope::bound(dispatched[0].output.clone(), &config);

    assert!(envelope.truncated);
    assert_eq!(
        envelope.payload["blob"]
            .as_str()
            .expect("blob")
            .chars()
            .count(),
        128
    );
    assert_eq!(
        envelope.payload["items"].as_array().expect("items").len(),
        8
    );

    let json = serde_json::to_string(&envelope).expect("serialize");
    let parsed: ToolResultEnvelope = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, envelope);
    Ok(())
}

#[test]
fn omitted_segments_disambiguate_two_oversized_fields() {
    let config = ToolResultEnvelopeConfig::new(3);
    let envelope = ToolResultEnvelope::bound(json!({"left": "abcdef", "right": "ghijkl"}), &config);

    let tokens: Vec<&str> = envelope
        .omitted_segments
        .iter()
        .map(|segment| segment.page_token.as_str())
        .collect();

    assert_eq!(tokens.len(), 2);
    assert_ne!(tokens[0], tokens[1]);
    assert!(tokens.iter().any(|token| {
        decode_tool_result_page_token(token).is_some_and(|decoded| decoded.pointer == "/left")
    }));
    assert!(tokens.iter().any(|token| {
        decode_tool_result_page_token(token).is_some_and(|decoded| decoded.pointer == "/right")
    }));
}

#[test]
fn redaction_runs_before_size_bounding() {
    let config = ToolResultEnvelopeConfig::new(6).with_redaction_policy(
        RedactionPolicy::deny_pointers(["/credentials/token"]).with_default_replacement("[secret]"),
    );
    let envelope = ToolResultEnvelope::bound(
        json!({"credentials": {"token": "very-sensitive-token"}, "body": "abcdefghi"}),
        &config,
    );

    assert_eq!(envelope.payload["credentials"]["token"], json!("[secret]"));
    assert_eq!(envelope.redacted_values, 1);
    assert_eq!(envelope.payload["body"], json!("abcdef"));
    assert!(
        !envelope
            .omitted_segments
            .iter()
            .any(|segment| segment.pointer == "/credentials/token")
    );
}

#[test]
fn total_payload_budget_omits_later_fields() {
    let config = ToolResultEnvelopeConfig::new(100).with_max_total_bytes(32);
    let envelope = ToolResultEnvelope::bound(
        json!({"first": "kept", "second": "kept-too", "third": "omitted"}),
        &config,
    );

    assert!(envelope.truncated);
    assert!(envelope.omitted_values > 0);
    assert!(envelope.omitted_segments.iter().any(|segment| {
        segment.reason == ToolResultOmissionReason::TotalBytes
            && decode_tool_result_page_token(&segment.page_token).is_some_and(|decoded| {
                decoded.reason == ToolResultOmissionReason::TotalBytes && decoded.limit == 32
            })
    }));
}

#[tokio::test]
async fn bounded_dispatch_returns_envelopes_without_changing_raw_dispatch()
-> Result<(), KernelError> {
    let tools = ToolRegistry::new();
    tools.register(oversized_tool());

    let invocations = vec![ToolInvocation::new("diagnostics.big_payload", json!({}))?];
    let config = ToolResultEnvelopeConfig::new(64).with_max_array_items(4);
    let bounded = dispatch_tool_invocations_bounded(&tools, &invocations, &config).await?;

    assert_eq!(bounded.len(), 1);
    assert_eq!(bounded[0].invocation.name, "diagnostics.big_payload");
    assert!(bounded[0].envelope.truncated);
    assert_eq!(
        bounded[0].envelope.payload["blob"]
            .as_str()
            .expect("blob")
            .chars()
            .count(),
        64
    );
    assert_eq!(
        bounded[0].envelope.payload["items"]
            .as_array()
            .expect("items")
            .len(),
        4
    );

    let raw = dispatch_tool_invocations(&tools, &invocations).await?;
    assert_eq!(raw[0].output["blob"].as_str().expect("raw").len(), 10_000);
    Ok(())
}

#[tokio::test]
async fn small_payloads_pass_through_envelope_unchanged() -> Result<(), KernelError> {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "math.add".into(),
            description: "add two integers".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move {
            let a = args.get("a").and_then(Value::as_i64).unwrap_or(0);
            let b = args.get("b").and_then(Value::as_i64).unwrap_or(0);
            Ok(json!({ "sum": a + b }))
        },
    )));

    let invocations = vec![ToolInvocation::new("math.add", json!({"a": 20, "b": 22}))?];
    let dispatched = dispatch_tool_invocations(&tools, &invocations).await?;
    let raw = dispatched[0].output.clone();

    let envelope = bound_tool_result(raw.clone());
    assert!(!envelope.truncated);
    assert_eq!(envelope.omitted_chars, 0);
    assert_eq!(envelope.omitted_items, 0);
    assert!(envelope.page_token.is_none());
    assert_eq!(envelope.payload, raw);
    Ok(())
}
