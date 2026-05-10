//! Integration tests for [`rig_compose::normalizer`].
//!
//! These tests exercise the normalizer through the public API surface
//! (re-exported from the crate root) to validate the full end-to-end path
//! from raw model output to structured tool invocations.

use rig_compose::normalizer::{LfmNormalizer, StructuredToolCallNormalizer, ToolCallNormalizer};
use rig_compose::{KernelError, LocalTool, ToolRegistry, ToolSchema, dispatch_tool_invocations};
use serde_json::json;
use std::sync::Arc;

// ── Realistic LFM output patterns ────────────────────────────────────────────

#[test]
fn full_llm_response_with_preamble_and_postamble() {
    // Real model output often wraps the marker in surrounding text.
    let raw = concat!(
        "I'll look up the weather for you.\n",
        "<|tool_call_start|>[get_weather(city='Berlin', units='metric')]<|tool_call_end|>\n",
        "Let me know if you need anything else.",
    );
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].args, json!({"city": "Berlin", "units": "metric"}));
}

#[test]
fn chained_tool_calls_two_separate_blocks() {
    // Model emits two sequential tool calls as separate blocks.
    let raw = concat!(
        "<|tool_call_start|>[fetch_url(url='https://example.com')]<|tool_call_end|>\n",
        "<|tool_call_start|>[parse_html(selector='h1')]<|tool_call_end|>",
    );
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "fetch_url");
    assert_eq!(calls[1].name, "parse_html");
}

#[test]
fn parallel_tool_calls_in_one_block() {
    // Model batches multiple calls in a single marker.
    let raw = "<|tool_call_start|>[search(q='rust'), search(q='tokio')]<|tool_call_end|>";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].args, json!({"q": "rust"}));
    assert_eq!(calls[1].args, json!({"q": "tokio"}));
}

#[test]
fn no_tool_call_in_response() {
    let raw = "The answer to your question is 42. No tools are needed.";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert!(calls.is_empty());
}

// ── Argument type coverage ────────────────────────────────────────────────────

#[test]
fn mixed_arg_types() {
    let raw = "<|tool_call_start|>[configure(name='agent', limit=100, ratio=0.5, debug=True, ctx=None)]<|tool_call_end|>";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 1);
    let args = &calls[0].args;
    assert_eq!(args["name"], json!("agent"));
    assert_eq!(args["limit"], json!(100));
    assert_eq!(args["ratio"].as_f64().unwrap(), 0.5_f64);
    assert_eq!(args["debug"], json!(true));
    assert_eq!(args["ctx"], json!(null));
}

#[test]
fn string_with_comma_in_value() {
    // Comma inside quoted string must not split the kwarg list.
    let raw = "<|tool_call_start|>[translate(text='hello, world', lang='es')]<|tool_call_end|>";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].args["text"], json!("hello, world"));
    assert_eq!(calls[0].args["lang"], json!("es"));
}

#[test]
fn negative_integer_arg() {
    let raw = "<|tool_call_start|>[offset(n=-5)]<|tool_call_end|>";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls[0].args["n"], json!(-5));
}

#[test]
fn nested_list_and_object_args() {
    let raw = "<|tool_call_start|>[plan(items=['a,b', 'c'], meta={'city': 'Berlin', 'coords': [52.52, 13.405], 'active': True})]<|tool_call_end|>";
    let calls = LfmNormalizer.normalize(raw).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].args,
        json!({
            "items": ["a,b", "c"],
            "meta": {
                "city": "Berlin",
                "coords": [52.52, 13.405],
                "active": true
            }
        })
    );
}

// ── Structured standards ─────────────────────────────────────────────────────

#[test]
fn openai_responses_output_normalizes_to_invocation() {
    let value = json!({
        "id": "resp_123",
        "output": [{
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": "{\"city\":\"Berlin\"}",
            "status": "completed"
        }]
    });

    let calls = StructuredToolCallNormalizer::normalize_openai_responses(&value).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].args, json!({"city": "Berlin"}));
}

#[test]
fn openai_chat_completions_tool_calls_normalize_to_invocation() {
    let value = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Berlin"}
                    }
                }]
            }
        }]
    });

    let calls = StructuredToolCallNormalizer::normalize_openai_chat_completions(&value).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].args, json!({"city": "Berlin"}));
}

#[test]
fn unsupported_structured_payload_returns_empty() {
    let value = json!({"output": [{"type": "message", "content": []}]});
    let calls = StructuredToolCallNormalizer::normalize(&value).unwrap();
    assert!(calls.is_empty());
}

// ── Error paths ───────────────────────────────────────────────────────────────

#[test]
fn unclosed_start_marker_is_an_error() {
    let raw = "<|tool_call_start|>[get_weather(city='Berlin')]";
    let err = LfmNormalizer.normalize(raw).unwrap_err();
    assert!(
        matches!(err, KernelError::NormalizerFailed(_)),
        "expected NormalizerFailed, got: {err:?}"
    );
}

#[test]
fn error_message_mentions_unclosed() {
    let raw = "<|tool_call_start|>[incomplete(";
    let err = LfmNormalizer.normalize(raw).unwrap_err();
    assert!(err.to_string().contains("unclosed"), "got: {err}");
}

#[test]
fn kwarg_without_equals_is_an_error() {
    let raw = "<|tool_call_start|>[fn(positional_only)]<|tool_call_end|>";
    let err = LfmNormalizer.normalize(raw).unwrap_err();
    assert!(
        matches!(err, KernelError::NormalizerFailed(_)),
        "expected NormalizerFailed, got: {err:?}"
    );
}

#[test]
fn malformed_identifiers_are_errors() {
    let raw = "<|tool_call_start|>[bad/name(arg=1)]<|tool_call_end|>";
    let err = LfmNormalizer.normalize(raw).unwrap_err();
    assert!(
        matches!(err, KernelError::NormalizerFailed(_)),
        "expected NormalizerFailed, got: {err:?}"
    );
    assert!(err.to_string().contains("invalid tool name"));
}

#[test]
fn duplicate_kwargs_are_errors() {
    let raw = "<|tool_call_start|>[fn(city='Berlin', city='Paris')]<|tool_call_end|>";
    let err = LfmNormalizer.normalize(raw).unwrap_err();
    assert!(
        matches!(err, KernelError::NormalizerFailed(_)),
        "expected NormalizerFailed, got: {err:?}"
    );
    assert!(err.to_string().contains("duplicate kwarg"));
}

#[tokio::test]
async fn normalized_invocations_dispatch_through_tool_registry() {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "gets weather".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move { Ok(json!({"forecast": "sunny", "args": args})) },
    )));

    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();
    let results = dispatch_tool_invocations(&tools, &invocations)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].invocation.name, "get_weather");
    assert_eq!(
        results[0].output,
        json!({"forecast": "sunny", "args": {"city": "Berlin"}})
    );
}

#[tokio::test]
async fn structured_standard_invocations_dispatch_through_tool_registry() {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "gets weather".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move { Ok(json!({"forecast": "clear", "args": args})) },
    )));

    let value = json!({
        "output": [{
            "type": "function_call",
            "name": "get_weather",
            "arguments": "{\"city\":\"Berlin\"}"
        }]
    });
    let invocations = StructuredToolCallNormalizer::normalize(&value).unwrap();
    let results = dispatch_tool_invocations(&tools, &invocations)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].invocation.name, "get_weather");
    assert_eq!(
        results[0].output,
        json!({"forecast": "clear", "args": {"city": "Berlin"}})
    );
}

// ── is_applicable guard ───────────────────────────────────────────────────────

#[test]
fn is_applicable_matches_exactly_when_marker_present() {
    assert!(LfmNormalizer.is_applicable("<|tool_call_start|>[fn()]<|tool_call_end|>"));
    assert!(!LfmNormalizer.is_applicable("plain prose with no markers"));
    assert!(!LfmNormalizer.is_applicable("<|tool_call_end|> only end marker"));
}
