//! Integration tests for [`rig_compose::normalizer`].
//!
//! These tests exercise the normalizer through the public API surface
//! (re-exported from the crate root) to validate the full end-to-end path
//! from raw model output to structured tool invocations.

use rig_compose::normalizer::{LfmNormalizer, StructuredToolCallNormalizer, ToolCallNormalizer};
use rig_compose::{
    AtomicBudget, DispatchBudgetHook, KernelError, LocalTool, ToolDispatchAction, ToolDispatchHook,
    ToolInvocation, ToolInvocationOutcome, ToolInvocationResult, ToolRegistry, ToolSchema,
    dispatch_tool_invocations, dispatch_tool_invocations_with_hooks,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

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

#[tokio::test]
async fn dispatch_hooks_record_after_invocation() {
    let tools = weather_registry("clear");
    let hook = RecordingHook::default();
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let results = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        hook.events(),
        vec!["before:get_weather", "after:get_weather"]
    );
}

#[tokio::test]
async fn dispatch_hooks_can_skip_with_synthetic_output() {
    let tools = weather_registry("clear");
    let hook = SkipWeatherHook;
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let results = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        results[0].output,
        json!({"city": "Berlin", "forecast": "policy-supplied"})
    );
}

#[tokio::test]
async fn dispatch_hooks_report_skip_outcome() {
    let tools = weather_registry("clear");
    let skip = SkipWeatherHook;
    let recorder = OutcomeRecordingHook::default();
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let results = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&skip, &recorder])
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(
        recorder.events(),
        vec!["after:get_weather:skipped:policy-supplied"]
    );
}

#[tokio::test]
async fn dispatch_hooks_can_terminate_before_tool_invocation() {
    let tools = weather_registry("clear");
    let hook = TerminateHook;
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let err = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap_err();

    assert!(matches!(err, KernelError::ToolDispatchTerminated(_)));
    assert!(err.to_string().contains("approval required"));
}

#[tokio::test]
async fn dispatch_budget_hook_releases_after_success() {
    let tools = weather_registry("clear");
    let budget = Arc::new(AtomicBudget::new(1));
    let hook = DispatchBudgetHook::new(budget.clone(), 1);
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let results = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(budget.available(), 1);
}

#[tokio::test]
async fn dispatch_budget_hook_terminates_when_budget_is_denied() {
    let tools = weather_registry("clear");
    let budget = Arc::new(AtomicBudget::new(0));
    let hook = DispatchBudgetHook::new(budget, 1);
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let err = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap_err();

    assert!(matches!(err, KernelError::ToolDispatchTerminated(_)));
    assert!(err.to_string().contains("budget denied"));
}

#[tokio::test]
async fn dispatch_budget_hook_releases_after_tool_error() {
    let tools = failing_registry();
    let budget = Arc::new(AtomicBudget::new(1));
    let hook = DispatchBudgetHook::new(budget.clone(), 1);
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let err = dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&hook])
        .await
        .unwrap_err();

    assert!(matches!(err, KernelError::ToolFailed(_)));
    assert_eq!(budget.available(), 1);
}

#[tokio::test]
async fn dispatch_budget_hook_releases_when_later_hook_errors_in_before_invocation() {
    // Regression: a hook earlier in the chain reserved a slot before a
    // later hook returned `Err` from `before_invocation`. The reservation
    // must be released even though the failing hook never reserved
    // anything and was never given a chance to run `on_invocation_error`.
    let tools = weather_registry("clear");
    let budget = Arc::new(AtomicBudget::new(1));
    let budget_hook = DispatchBudgetHook::new(budget.clone(), 1);
    let failing_hook = ErroringBeforeHook;
    let invocations = LfmNormalizer
        .normalize("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        .unwrap();

    let err =
        dispatch_tool_invocations_with_hooks(&tools, &invocations, &[&budget_hook, &failing_hook])
            .await
            .unwrap_err();

    assert!(matches!(err, KernelError::BudgetFailed(_)));
    assert_eq!(
        budget.available(),
        1,
        "earlier hook's reservation must be released"
    );
}

#[tokio::test]
async fn normalized_tool_results_can_drive_a_second_model_turn() {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "gets weather".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |args| async move {
            Ok(json!({
                "city": args.get("city").and_then(|value| value.as_str()).unwrap_or("unknown"),
                "forecast": "clear and cool"
            }))
        },
    )));

    let run = run_deterministic_tool_loop_harness(
        &tools,
        "What is the weather like in Berlin today?",
        "<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>",
    )
    .await
    .unwrap();

    assert_eq!(run.task, "What is the weather like in Berlin today?");
    assert!(run.first_model_output.contains("<|tool_call_start|>"));
    assert_eq!(run.invocations.len(), 1);
    assert_eq!(run.dispatch_results.len(), 1);
    assert_eq!(run.dispatch_results[0].invocation.name, "get_weather");
    assert!(run.final_answer.contains("Berlin"));
    assert!(run.final_answer.contains("clear and cool"));
    assert_eq!(
        run.passed_assertions,
        vec![
            "model-output-normalized",
            "tool-dispatched",
            "final-answer-grounded"
        ]
    );
}

struct DeterministicToolLoopHarnessRun {
    task: String,
    first_model_output: String,
    invocations: Vec<ToolInvocation>,
    dispatch_results: Vec<ToolInvocationResult>,
    final_answer: String,
    passed_assertions: Vec<&'static str>,
}

async fn run_deterministic_tool_loop_harness(
    tools: &ToolRegistry,
    task: &str,
    first_model_output: &str,
) -> Result<DeterministicToolLoopHarnessRun, KernelError> {
    let invocations = LfmNormalizer.normalize(first_model_output)?;
    let dispatch_results = dispatch_tool_invocations(tools, &invocations).await?;
    let final_answer = fake_second_model_turn(&dispatch_results);
    let passed_assertions = harness_assertions(&invocations, &dispatch_results, &final_answer);

    Ok(DeterministicToolLoopHarnessRun {
        task: task.to_string(),
        first_model_output: first_model_output.to_string(),
        invocations,
        dispatch_results,
        final_answer,
        passed_assertions,
    })
}

fn harness_assertions(
    invocations: &[ToolInvocation],
    dispatch_results: &[ToolInvocationResult],
    final_answer: &str,
) -> Vec<&'static str> {
    let mut passed = Vec::new();

    if !invocations.is_empty() {
        passed.push("model-output-normalized");
    }
    if dispatch_results
        .iter()
        .any(|result| result.invocation.name == "get_weather")
    {
        passed.push("tool-dispatched");
    }
    if final_answer.contains("Berlin") && final_answer.contains("clear and cool") {
        passed.push("final-answer-grounded");
    }

    passed
}

fn fake_second_model_turn(results: &[ToolInvocationResult]) -> String {
    results
        .first()
        .and_then(|result| {
            let city = result.output.get("city")?.as_str()?;
            let forecast = result.output.get("forecast")?.as_str()?;
            Some(format!("The weather in {city} is {forecast}."))
        })
        .unwrap_or_else(|| "No tool result was available.".to_string())
}

fn weather_registry(forecast: &'static str) -> ToolRegistry {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "gets weather".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        move |args| async move {
            Ok(json!({
                "city": args.get("city").and_then(|value| value.as_str()).unwrap_or("unknown"),
                "forecast": forecast
            }))
        },
    )));
    tools
}

fn failing_registry() -> ToolRegistry {
    let tools = ToolRegistry::new();
    tools.register(Arc::new(LocalTool::new(
        ToolSchema {
            name: "get_weather".into(),
            description: "fails weather lookup".into(),
            args_schema: json!({"type": "object"}),
            result_schema: json!({"type": "object"}),
        },
        |_args| async move { Err(KernelError::ToolFailed("weather offline".into())) },
    )));
    tools
}

#[derive(Default)]
struct RecordingHook {
    events: Mutex<Vec<String>>,
}

impl RecordingHook {
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolDispatchHook for RecordingHook {
    async fn before_invocation(
        &self,
        invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("before:{}", invocation.name));
        Ok(ToolDispatchAction::Continue)
    }

    async fn after_invocation(&self, result: &ToolInvocationResult) -> Result<(), KernelError> {
        self.events
            .lock()
            .unwrap()
            .push(format!("after:{}", result.invocation.name));
        Ok(())
    }
}

struct SkipWeatherHook;

#[async_trait]
impl ToolDispatchHook for SkipWeatherHook {
    async fn before_invocation(
        &self,
        invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        let city = invocation
            .args
            .get("city")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown");
        Ok(ToolDispatchAction::Skip {
            output: json!({"city": city, "forecast": "policy-supplied"}),
            reason: Some("policy-supplied".into()),
        })
    }
}

#[derive(Default)]
struct OutcomeRecordingHook {
    events: Mutex<Vec<String>>,
}

impl OutcomeRecordingHook {
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }
}

#[async_trait]
impl ToolDispatchHook for OutcomeRecordingHook {
    async fn after_invocation_with_outcome(
        &self,
        result: &ToolInvocationResult,
        outcome: &ToolInvocationOutcome,
    ) -> Result<(), KernelError> {
        let outcome_text = match outcome {
            ToolInvocationOutcome::Completed => "completed".to_string(),
            ToolInvocationOutcome::Skipped { reason } => {
                format!("skipped:{}", reason.as_deref().unwrap_or("unspecified"))
            }
        };
        self.events
            .lock()
            .unwrap()
            .push(format!("after:{}:{outcome_text}", result.invocation.name));
        Ok(())
    }
}

struct TerminateHook;

#[async_trait]
impl ToolDispatchHook for TerminateHook {
    async fn before_invocation(
        &self,
        _invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        Ok(ToolDispatchAction::Terminate {
            reason: "approval required".into(),
        })
    }
}

struct ErroringBeforeHook;

#[async_trait]
impl ToolDispatchHook for ErroringBeforeHook {
    async fn before_invocation(
        &self,
        _invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        Err(KernelError::BudgetFailed("simulated denial".into()))
    }
}

// ── is_applicable guard ───────────────────────────────────────────────────────

#[test]
fn is_applicable_matches_exactly_when_marker_present() {
    assert!(LfmNormalizer.is_applicable("<|tool_call_start|>[fn()]<|tool_call_end|>"));
    assert!(!LfmNormalizer.is_applicable("plain prose with no markers"));
    assert!(!LfmNormalizer.is_applicable("<|tool_call_end|> only end marker"));
}
