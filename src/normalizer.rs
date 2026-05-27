//! [`ToolCallNormalizer`] — converts raw LLM text output into structured
//! [`ToolInvocation`]s.
//!
//! Models served via OpenAI-compatible APIs (e.g. `mlx_lm.server`) sometimes
//! emit tool-intent as in-band text markers rather than the structured
//! `tool_calls` JSON field. Normalizers detect and decode those markers so the
//! kernel can dispatch them identically to first-class tool calls.
//!
//! # Built-in implementations
//!
//! | Type | Format |
//! |------|--------|
//! | [`LfmNormalizer`] | LiquidAI LFM `<\|tool_call_start\|>[fn(k=v)]<\|tool_call_end\|>` |
//! | [`StructuredToolCallNormalizer`] | OpenAI Responses `function_call` output and Chat Completions `tool_calls` |
//!
//! # Example
//!
//! ```no_run
//! use rig_compose::normalizer::{LfmNormalizer, ToolCallNormalizer};
//!
//! let raw = "<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>";
//! let normalizer = LfmNormalizer;
//! let calls = normalizer.normalize(raw).expect("parse ok");
//! assert_eq!(calls[0].name, "get_weather");
//! ```

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::registry::KernelError;
use crate::registry::ToolRegistry;
use crate::tool::{ToolName, ToolResultEnvelope, ToolResultEnvelopeConfig};
use crate::trace::{DispatchTrace, DispatchTraceEvent, TracedAction, TracedOutcome};

// ── Public types ─────────────────────────────────────────────────────────────

/// A structured tool invocation extracted from raw model output.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocation {
    /// Registry name of the tool to invoke (e.g. `"get_weather"`).
    pub name: ToolName,
    /// JSON object of arguments to pass to the tool.
    pub args: Value,
}

impl ToolInvocation {
    /// Build a validated [`ToolInvocation`] from a tool name and JSON args.
    pub fn new(name: impl Into<ToolName>, args: Value) -> Result<Self, KernelError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(KernelError::NormalizerFailed(
                "empty tool name in structured tool call".into(),
            ));
        }
        validate_identifier("tool name", &name)?;
        Ok(Self { name, args })
    }

    /// Dispatch this invocation through a [`ToolRegistry`].
    pub async fn dispatch(&self, tools: &ToolRegistry) -> Result<Value, KernelError> {
        tools.invoke(&self.name, self.args.clone()).await
    }
}

/// Result of dispatching one normalized [`ToolInvocation`].
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInvocationResult {
    /// The normalized invocation that was dispatched.
    pub invocation: ToolInvocation,
    /// The JSON result returned by the invoked tool.
    pub output: Value,
}

/// Bounded result of dispatching one normalized [`ToolInvocation`].
///
/// This is the model-visible companion to [`ToolInvocationResult`]: dispatch
/// still runs through the same registry path, but the returned payload is
/// wrapped in a [`ToolResultEnvelope`] so callers can see truncation metadata
/// and continuation tokens before placing tool output into a prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedToolInvocationResult {
    /// The normalized invocation that was dispatched.
    pub invocation: ToolInvocation,
    /// Bounded tool output plus deterministic truncation metadata.
    pub envelope: ToolResultEnvelope,
}

/// Decision returned by a [`ToolDispatchHook`] before a tool invocation runs.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolDispatchAction {
    /// Invoke the tool normally.
    Continue,
    /// Do not invoke the tool; record `output` as the invocation result.
    Skip {
        /// Synthetic output to record as the invocation result.
        output: Value,
        /// Optional human-readable reason the tool body was not invoked.
        reason: Option<String>,
    },
    /// Stop dispatching and return [`KernelError::ToolDispatchTerminated`].
    Terminate { reason: String },
}

/// Outcome recorded for one normalized [`ToolInvocation`].
#[derive(Debug, Clone, PartialEq)]
pub enum ToolInvocationOutcome {
    /// The tool body ran and produced the result.
    Completed,
    /// A hook supplied a synthetic skip result instead of invoking the tool.
    Skipped {
        /// Optional human-readable reason the tool body was not invoked.
        reason: Option<String>,
    },
}

/// Hook for policy, accounting, and tracing around normalized tool dispatch.
///
/// Hooks are intentionally provider-neutral: they see only the normalized
/// [`ToolInvocation`] and the resulting [`ToolInvocationResult`]. Concrete
/// policy engines, approval systems, and trace exporters should live in
/// downstream crates and plug into this small kernel surface.
#[async_trait]
pub trait ToolDispatchHook: Send + Sync {
    /// Called before each invocation. Return [`ToolDispatchAction::Continue`]
    /// to invoke the tool, [`ToolDispatchAction::Skip`] to synthesize a result,
    /// or [`ToolDispatchAction::Terminate`] to stop the dispatch loop.
    async fn before_invocation(
        &self,
        _invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        Ok(ToolDispatchAction::Continue)
    }

    /// Called after a tool invocation or hook-provided skip result is recorded.
    async fn after_invocation(&self, _result: &ToolInvocationResult) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called after a dispatch result is recorded, including whether it came
    /// from real tool execution or a hook-provided skip.
    ///
    /// The default implementation preserves compatibility for hooks that only
    /// care about the result payload.
    async fn after_invocation_with_outcome(
        &self,
        result: &ToolInvocationResult,
        _outcome: &ToolInvocationOutcome,
    ) -> Result<(), KernelError> {
        self.after_invocation(result).await
    }

    /// Called when dispatch stops after this hook may have observed the
    /// invocation in [`Self::before_invocation`]. Hooks that reserve resources
    /// before dispatch should release them here.
    async fn on_invocation_error(
        &self,
        _invocation: &ToolInvocation,
        _error: &KernelError,
    ) -> Result<(), KernelError> {
        Ok(())
    }
}

/// Dispatch normalized tool invocations sequentially through a [`ToolRegistry`].
///
/// Sequential dispatch preserves model-emitted call order and avoids adding a
/// runtime-specific concurrency policy to the kernel. Callers that know their
/// tools are independent can still dispatch invocations concurrently by using
/// [`ToolInvocation::dispatch`] directly.
pub async fn dispatch_tool_invocations(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
) -> Result<Vec<ToolInvocationResult>, KernelError> {
    dispatch_tool_invocations_with_hooks(tools, invocations, &[]).await
}

/// Dispatch normalized tool invocations with policy/accounting hooks.
///
/// Hooks run in the order provided. A skip result still triggers every hook's
/// [`ToolDispatchHook::after_invocation`] callback so audit hooks can record
/// synthetic outcomes. A terminate action stops dispatch before the tool is
/// invoked and returns [`KernelError::ToolDispatchTerminated`].
pub async fn dispatch_tool_invocations_with_hooks(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
    hooks: &[&dyn ToolDispatchHook],
) -> Result<Vec<ToolInvocationResult>, KernelError> {
    dispatch_inner(tools, invocations, hooks, None).await
}

/// Dispatch normalized tool invocations and bound every result with `config`.
///
/// Existing dispatch helpers intentionally return raw tool output so hosts can
/// decide where and how to preserve full results. This helper is for the
/// prompt/model boundary: it applies [`ToolResultEnvelope`] after successful
/// dispatch and returns the bounded, replayable result records.
pub async fn dispatch_tool_invocations_bounded(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
    config: &ToolResultEnvelopeConfig,
) -> Result<Vec<BoundedToolInvocationResult>, KernelError> {
    dispatch_tool_invocations_with_hooks_bounded(tools, invocations, &[], config).await
}

/// Dispatch normalized tool invocations with hooks and bound every result.
///
/// Hooks observe the raw [`ToolInvocationResult`] before bounding so accounting
/// and audit layers can decide whether they need full output. The returned
/// value is the bounded, model-visible projection.
pub async fn dispatch_tool_invocations_with_hooks_bounded(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
    hooks: &[&dyn ToolDispatchHook],
    config: &ToolResultEnvelopeConfig,
) -> Result<Vec<BoundedToolInvocationResult>, KernelError> {
    let results = dispatch_tool_invocations_with_hooks(tools, invocations, hooks).await?;
    Ok(bound_invocation_results(results, config))
}

/// Dispatch normalized tool invocations and record a [`DispatchTrace`].
///
/// Behaves identically to [`dispatch_tool_invocations_with_hooks`], but appends
/// a [`DispatchTraceEvent`] for every hook decision, hook error, reservation
/// cleanup, hook-after invocation, and final per-invocation outcome. Use this
/// when a host needs a deterministic, replayable record of policy decisions
/// without depending on a concrete tracing backend.
pub async fn dispatch_tool_invocations_with_trace(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
    hooks: &[&dyn ToolDispatchHook],
    trace: &DispatchTrace,
) -> Result<Vec<ToolInvocationResult>, KernelError> {
    dispatch_inner(tools, invocations, hooks, Some(trace)).await
}

fn bound_invocation_results(
    results: Vec<ToolInvocationResult>,
    config: &ToolResultEnvelopeConfig,
) -> Vec<BoundedToolInvocationResult> {
    results
        .into_iter()
        .map(|result| BoundedToolInvocationResult {
            invocation: result.invocation,
            envelope: ToolResultEnvelope::bound(result.output, config),
        })
        .collect()
}

async fn dispatch_inner(
    tools: &ToolRegistry,
    invocations: &[ToolInvocation],
    hooks: &[&dyn ToolDispatchHook],
    trace: Option<&DispatchTrace>,
) -> Result<Vec<ToolInvocationResult>, KernelError> {
    let mut results = Vec::with_capacity(invocations.len());

    for (invocation_index, invocation) in invocations.iter().enumerate() {
        let mut action = ToolDispatchAction::Continue;
        // Track how many hooks observed `before_invocation` so that, on a
        // hook error, we can notify exactly those hooks via
        // `on_invocation_error`. Without this, a hook that reserved a
        // resource in `before_invocation` (e.g. `DispatchBudgetHook`)
        // would leak that reservation when a later hook errors.
        let mut observed: usize = 0;
        let mut before_err: Option<(usize, KernelError)> = None;
        for (hook_index, hook) in hooks.iter().enumerate() {
            match hook.before_invocation(invocation).await {
                Ok(next) => {
                    observed += 1;
                    if let Some(trace) = trace {
                        trace.push(DispatchTraceEvent::HookBefore {
                            invocation_index,
                            hook_index,
                            decision: TracedAction::from(&next),
                        });
                    }
                    action = next;
                    if !matches!(action, ToolDispatchAction::Continue) {
                        break;
                    }
                }
                Err(error) => {
                    before_err = Some((hook_index, error));
                    break;
                }
            }
        }
        if let Some((hook_index, error)) = before_err {
            if let Some(trace) = trace {
                trace.push(DispatchTraceEvent::HookBeforeError {
                    invocation_index,
                    hook_index,
                    message: error.to_string(),
                });
            }
            notify_invocation_error_subset(
                hooks,
                observed,
                invocation,
                &error,
                trace,
                invocation_index,
            )
            .await?;
            if let Some(trace) = trace {
                trace.push(DispatchTraceEvent::InvocationOutcome {
                    invocation_index,
                    outcome: TracedOutcome::Failed {
                        message: error.to_string(),
                    },
                });
            }
            return Err(error);
        }

        let (output, outcome) = match action {
            ToolDispatchAction::Continue => match invocation.dispatch(tools).await {
                Ok(output) => (output, ToolInvocationOutcome::Completed),
                Err(error) => {
                    notify_invocation_error(hooks, invocation, &error, trace, invocation_index)
                        .await?;
                    if let Some(trace) = trace {
                        trace.push(DispatchTraceEvent::InvocationOutcome {
                            invocation_index,
                            outcome: TracedOutcome::Failed {
                                message: error.to_string(),
                            },
                        });
                    }
                    return Err(error);
                }
            },
            ToolDispatchAction::Skip { output, reason } => {
                (output, ToolInvocationOutcome::Skipped { reason })
            }
            ToolDispatchAction::Terminate { reason } => {
                let error = KernelError::ToolDispatchTerminated(reason.clone());
                notify_invocation_error(hooks, invocation, &error, trace, invocation_index).await?;
                if let Some(trace) = trace {
                    trace.push(DispatchTraceEvent::InvocationOutcome {
                        invocation_index,
                        outcome: TracedOutcome::Terminated { reason },
                    });
                }
                return Err(error);
            }
        };

        let result = ToolInvocationResult {
            invocation: invocation.clone(),
            output,
        };

        for (hook_index, hook) in hooks.iter().enumerate() {
            hook.after_invocation_with_outcome(&result, &outcome)
                .await?;
            if let Some(trace) = trace {
                trace.push(DispatchTraceEvent::HookAfter {
                    invocation_index,
                    hook_index,
                });
            }
        }

        if let Some(trace) = trace {
            let outcome_event = match &outcome {
                ToolInvocationOutcome::Completed => TracedOutcome::Completed,
                ToolInvocationOutcome::Skipped { reason } => TracedOutcome::Skipped {
                    reason: reason.clone(),
                },
            };
            trace.push(DispatchTraceEvent::InvocationOutcome {
                invocation_index,
                outcome: outcome_event,
            });
        }

        results.push(result);
    }

    Ok(results)
}

async fn notify_invocation_error(
    hooks: &[&dyn ToolDispatchHook],
    invocation: &ToolInvocation,
    error: &KernelError,
    trace: Option<&DispatchTrace>,
    invocation_index: usize,
) -> Result<(), KernelError> {
    for (hook_index, hook) in hooks.iter().enumerate() {
        hook.on_invocation_error(invocation, error).await?;
        if let Some(trace) = trace {
            trace.push(DispatchTraceEvent::HookCleanup {
                invocation_index,
                hook_index,
            });
        }
    }
    Ok(())
}

/// Notify the first `upto` hooks that observed `before_invocation` so they
/// can release any resources reserved there. Used when a later hook's
/// `before_invocation` returns an error and we must unwind partial state.
async fn notify_invocation_error_subset(
    hooks: &[&dyn ToolDispatchHook],
    upto: usize,
    invocation: &ToolInvocation,
    error: &KernelError,
    trace: Option<&DispatchTrace>,
    invocation_index: usize,
) -> Result<(), KernelError> {
    for (hook_index, hook) in hooks.iter().take(upto).enumerate() {
        hook.on_invocation_error(invocation, error).await?;
        if let Some(trace) = trace {
            trace.push(DispatchTraceEvent::HookCleanup {
                invocation_index,
                hook_index,
            });
        }
    }
    Ok(())
}

/// Normalizes raw LLM text output into structured [`ToolInvocation`]s.
///
/// Implement this trait to support additional model families that emit tool
/// intent as in-band text markers. The trait is object-safe so normalizers can
/// be stored as `Arc<dyn ToolCallNormalizer>` alongside other kernel objects.
///
/// # Contract
///
/// - [`normalize`](ToolCallNormalizer::normalize) returns an empty `Vec` when
///   `raw` contains no markers this normalizer recognises. An empty result is
///   never an error.
/// - [`is_applicable`](ToolCallNormalizer::is_applicable) must return `true`
///   whenever `normalize` would return a non-empty `Vec`. It is a cheap guard
///   to short-circuit expensive parsing in pipelines.
pub trait ToolCallNormalizer: Send + Sync {
    /// Parse `raw` text into zero or more tool invocations.
    fn normalize(&self, raw: &str) -> Result<Vec<ToolInvocation>, KernelError>;

    /// Quick scan: does `raw` contain markers this normalizer handles?
    fn is_applicable(&self, raw: &str) -> bool;
}

// ── Structured standards normalizer ──────────────────────────────────────────

/// Normalizer for structured tool-call JSON returned by common provider APIs.
///
/// This type keeps the kernel independent from provider-specific Rust types by
/// operating on `serde_json::Value` shapes. It supports:
///
/// - OpenAI Responses API output items: `{"type":"function_call", ...}`
/// - OpenAI Responses API full responses: `{ "output": [function_call, ...] }`
/// - OpenAI Chat Completions tool calls: `{ "tool_calls": [...] }`
/// - OpenAI Chat Completions full responses: `{ "choices": [{ "message": ... }] }`
#[derive(Debug, Clone, Default)]
pub struct StructuredToolCallNormalizer;

impl StructuredToolCallNormalizer {
    /// Parse OpenAI Responses API `function_call` output items from either a
    /// full response object, an `output` array, or a single output item.
    pub fn normalize_openai_responses(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
        match value {
            Value::Object(object) => {
                if let Some(output) = object.get("output") {
                    return normalize_responses_output(output);
                }
                if is_responses_function_call(object) {
                    return parse_responses_function_call(object).map(|call| vec![call]);
                }
                Ok(Vec::new())
            }
            Value::Array(items) => items
                .iter()
                .map(normalize_responses_output_item)
                .collect::<Result<Vec<_>, _>>()
                .map(flatten_invocations),
            _ => Ok(Vec::new()),
        }
    }

    /// Parse OpenAI Chat Completions `tool_calls` from either a full response,
    /// a message object, a `tool_calls` array, or a single tool call object.
    pub fn normalize_openai_chat_completions(
        value: &Value,
    ) -> Result<Vec<ToolInvocation>, KernelError> {
        match value {
            Value::Object(object) => {
                if let Some(choices) = object.get("choices") {
                    return normalize_chat_choices(choices);
                }
                if let Some(tool_calls) = object.get("tool_calls") {
                    return normalize_chat_tool_calls(tool_calls);
                }
                if is_chat_tool_call(object) {
                    return parse_chat_tool_call(object).map(|call| vec![call]);
                }
                Ok(Vec::new())
            }
            Value::Array(items) => normalize_chat_tool_calls_array(items),
            _ => Ok(Vec::new()),
        }
    }

    /// Parse all supported structured standards from `value`.
    ///
    /// This is useful when the caller has a provider JSON blob but does not
    /// want to branch on the provider path first. It preserves the order of
    /// calls within each standard and tries Responses before Chat Completions.
    pub fn normalize(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
        let mut invocations = Self::normalize_openai_responses(value)?;
        invocations.extend(Self::normalize_openai_chat_completions(value)?);
        Ok(invocations)
    }
}

// ── LFM normalizer ────────────────────────────────────────────────────────────

const LFM_START: &str = "<|tool_call_start|>";
const LFM_END: &str = "<|tool_call_end|>";

/// Normalizer for LiquidAI LFM models (e.g. `LFM2.5-1.2B-Thinking`) served
/// through `mlx_lm.server` or similar OpenAI-compatible shims that emit tool
/// intent as in-band text rather than the structured `tool_calls` field.
///
/// Recognised format:
/// ```text
/// <|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>
/// ```
///
/// Multiple calls per block (`[fn1(a=1), fn2(b=2)]`) and multiple blocks per
/// message are both handled.
///
/// # Example
///
/// ```no_run
/// use rig_compose::normalizer::{LfmNormalizer, ToolCallNormalizer};
/// use serde_json::json;
///
/// let raw = "<|tool_call_start|>[add(x=3, y=4)]<|tool_call_end|>";
/// let calls = LfmNormalizer.normalize(raw).unwrap();
/// assert_eq!(calls[0].name, "add");
/// assert_eq!(calls[0].args, json!({"x": 3, "y": 4}));
/// ```
#[derive(Debug, Clone, Default)]
pub struct LfmNormalizer;

impl ToolCallNormalizer for LfmNormalizer {
    fn is_applicable(&self, raw: &str) -> bool {
        raw.contains(LFM_START)
    }

    fn normalize(&self, raw: &str) -> Result<Vec<ToolInvocation>, KernelError> {
        let mut results = Vec::new();
        let mut remaining = raw;

        while let Some(block_start) = remaining.find(LFM_START) {
            // Skip past the start marker.
            let after_start = remaining
                .get(block_start + LFM_START.len()..)
                .ok_or_else(|| KernelError::NormalizerFailed("LFM: start marker overrun".into()))?;

            let block_end = after_start.find(LFM_END).ok_or_else(|| {
                KernelError::NormalizerFailed("LFM: unclosed <|tool_call_start|> marker".into())
            })?;

            let block = after_start.get(..block_end).ok_or_else(|| {
                KernelError::NormalizerFailed("LFM: block slice out of bounds".into())
            })?;

            // Advance past the end marker; if nothing remains, stop.
            remaining = after_start.get(block_end + LFM_END.len()..).unwrap_or("");

            let calls = parse_lfm_block(block)?;
            results.extend(calls);
        }

        Ok(results)
    }
}

// ── Structured standards helpers ─────────────────────────────────────────────

fn normalize_responses_output(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
    match value {
        Value::Array(items) => items
            .iter()
            .map(normalize_responses_output_item)
            .collect::<Result<Vec<_>, _>>()
            .map(flatten_invocations),
        Value::Object(object) if is_responses_function_call(object) => {
            parse_responses_function_call(object).map(|call| vec![call])
        }
        _ => Ok(Vec::new()),
    }
}

fn normalize_responses_output_item(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
    match value {
        Value::Object(object) if is_responses_function_call(object) => {
            parse_responses_function_call(object).map(|call| vec![call])
        }
        _ => Ok(Vec::new()),
    }
}

fn is_responses_function_call(object: &Map<String, Value>) -> bool {
    object
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "function_call")
}

fn parse_responses_function_call(
    object: &Map<String, Value>,
) -> Result<ToolInvocation, KernelError> {
    let name = required_string_field(object, "name", "OpenAI Responses function_call")?;
    let args = object
        .get("arguments")
        .map(parse_standard_arguments)
        .transpose()?
        .unwrap_or_else(|| Value::Object(Map::new()));
    ToolInvocation::new(name, args)
}

fn normalize_chat_choices(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
    let choices = value.as_array().ok_or_else(|| {
        KernelError::NormalizerFailed("OpenAI Chat Completions choices must be an array".into())
    })?;

    let mut invocations = Vec::new();
    for choice in choices {
        let Some(message) = choice.get("message") else {
            continue;
        };
        invocations
            .extend(StructuredToolCallNormalizer::normalize_openai_chat_completions(message)?);
    }

    Ok(invocations)
}

fn normalize_chat_tool_calls(value: &Value) -> Result<Vec<ToolInvocation>, KernelError> {
    match value {
        Value::Array(items) => normalize_chat_tool_calls_array(items),
        Value::Object(object) if is_chat_tool_call(object) => {
            parse_chat_tool_call(object).map(|call| vec![call])
        }
        _ => Ok(Vec::new()),
    }
}

fn normalize_chat_tool_calls_array(items: &[Value]) -> Result<Vec<ToolInvocation>, KernelError> {
    items
        .iter()
        .map(|item| match item {
            Value::Object(object) if is_chat_tool_call(object) => parse_chat_tool_call(object),
            Value::Object(_) => Err(KernelError::NormalizerFailed(
                "OpenAI Chat Completions tool call missing function payload".into(),
            )),
            _ => Err(KernelError::NormalizerFailed(
                "OpenAI Chat Completions tool call must be an object".into(),
            )),
        })
        .collect()
}

fn is_chat_tool_call(object: &Map<String, Value>) -> bool {
    object.get("function").is_some()
}

fn parse_chat_tool_call(object: &Map<String, Value>) -> Result<ToolInvocation, KernelError> {
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            KernelError::NormalizerFailed(
                "OpenAI Chat Completions tool call missing function object".into(),
            )
        })?;
    let name = required_string_field(function, "name", "OpenAI Chat Completions function")?;
    let args = function
        .get("arguments")
        .map(parse_standard_arguments)
        .transpose()?
        .unwrap_or_else(|| Value::Object(Map::new()));

    ToolInvocation::new(name, args)
}

fn parse_standard_arguments(value: &Value) -> Result<Value, KernelError> {
    match value {
        Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(Value::Object(Map::new()));
            }
            serde_json::from_str(trimmed).map_err(|err| {
                KernelError::NormalizerFailed(format!(
                    "failed to parse standard tool-call arguments JSON: {err}"
                ))
            })
        }
        Value::Null => Ok(Value::Object(Map::new())),
        other => Ok(other.clone()),
    }
}

fn required_string_field(
    object: &Map<String, Value>,
    field: &str,
    context: &str,
) -> Result<String, KernelError> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| KernelError::NormalizerFailed(format!("{context} missing `{field}` string")))
}

fn flatten_invocations(nested: Vec<Vec<ToolInvocation>>) -> Vec<ToolInvocation> {
    nested.into_iter().flatten().collect()
}

// ── Parsing helpers ───────────────────────────────────────────────────────────

/// Parse one `[fn1(a=1), fn2(b=2)]` block from an LFM marker.
fn parse_lfm_block(block: &str) -> Result<Vec<ToolInvocation>, KernelError> {
    let block = block.trim();
    // Strip optional surrounding `[ ]`.
    let inner = block
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(block);

    split_top_level(inner, ',')
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .map(|s| parse_lfm_call(s.trim()))
        .collect()
}

/// Parse one `fn_name(k1=v1, k2=v2)` call expression.
fn parse_lfm_call(expr: &str) -> Result<ToolInvocation, KernelError> {
    let (name_raw, rest) = expr.split_once('(').ok_or_else(|| {
        KernelError::NormalizerFailed(format!("LFM: expected '(' in call: {expr:?}"))
    })?;

    let name = name_raw.trim().to_string();
    if name.is_empty() {
        return Err(KernelError::NormalizerFailed(
            "LFM: empty tool name in call expression".into(),
        ));
    }
    validate_identifier("tool name", &name)?;

    // Use rsplit_once to handle nested parentheses in argument values.
    let (kwargs_str, trailing) = rest.rsplit_once(')').ok_or_else(|| {
        KernelError::NormalizerFailed(format!("LFM: missing closing ')' in: {expr:?}"))
    })?;
    if !trailing.trim().is_empty() {
        return Err(KernelError::NormalizerFailed(format!(
            "LFM: trailing content after call expression: {trailing:?}"
        )));
    }

    let args = parse_kwargs(kwargs_str)?;
    Ok(ToolInvocation { name, args })
}

/// Parse a comma-separated `key=value` kwargs string into a JSON object.
fn parse_kwargs(s: &str) -> Result<Value, KernelError> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    let mut map = Map::new();
    for pair in split_top_level(s, ',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key_raw, val_raw) = pair.split_once('=').ok_or_else(|| {
            KernelError::NormalizerFailed(format!("LFM: kwarg without '=': {pair:?}"))
        })?;
        let key = key_raw.trim().to_string();
        if key.is_empty() {
            return Err(KernelError::NormalizerFailed(
                "LFM: empty kwarg name".into(),
            ));
        }
        validate_identifier("kwarg name", &key)?;
        if map.contains_key(&key) {
            return Err(KernelError::NormalizerFailed(format!(
                "LFM: duplicate kwarg: {key}"
            )));
        }
        let val = parse_value(val_raw.trim())?;
        map.insert(key, val);
    }

    Ok(Value::Object(map))
}

/// Best-effort conversion of a Python literal token into a JSON [`Value`].
///
/// Supported: single/double-quoted strings, `True`/`False`, `None`/`null`,
/// integers, floats, lists, and dict/object literals. Anything else is
/// returned as an unquoted string.
fn parse_value(s: &str) -> Result<Value, KernelError> {
    let s = s.trim();

    if s.is_empty() {
        return Ok(Value::String(String::new()));
    }

    // Single-quoted string.
    if let Some(inner) = s.strip_prefix('\'').and_then(|t| t.strip_suffix('\'')) {
        return Ok(Value::String(
            inner.replace("\\'", "'").replace("\\\"", "\""),
        ));
    }
    if s.starts_with('\'') {
        return Err(KernelError::NormalizerFailed(
            "LFM: unterminated single-quoted string".into(),
        ));
    }
    // Double-quoted string.
    if let Some(inner) = s.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        return Ok(Value::String(
            inner.replace("\\'", "'").replace("\\\"", "\""),
        ));
    }
    if s.starts_with('"') {
        return Err(KernelError::NormalizerFailed(
            "LFM: unterminated double-quoted string".into(),
        ));
    }
    // Python booleans.
    if s == "True" {
        return Ok(Value::Bool(true));
    }
    if s == "False" {
        return Ok(Value::Bool(false));
    }
    // Null / None.
    if s == "None" || s == "null" {
        return Ok(Value::Null);
    }
    // List / array literal.
    if let Some(inner) = s.strip_prefix('[').and_then(|t| t.strip_suffix(']')) {
        return parse_array(inner);
    }
    if s.starts_with('[') {
        return Err(KernelError::NormalizerFailed(
            "LFM: unterminated list literal".into(),
        ));
    }
    // Dict / object literal.
    if let Some(inner) = s.strip_prefix('{').and_then(|t| t.strip_suffix('}')) {
        return parse_object(inner);
    }
    if s.starts_with('{') {
        return Err(KernelError::NormalizerFailed(
            "LFM: unterminated object literal".into(),
        ));
    }
    // Integer.
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Value::Number(n.into()));
    }
    // Float.
    if let Ok(f) = s.parse::<f64>() {
        let num = serde_json::Number::from_f64(f).ok_or_else(|| {
            KernelError::NormalizerFailed(format!("LFM: non-finite float in argument: {s:?}"))
        })?;
        return Ok(Value::Number(num));
    }
    // Fall back: treat as an unquoted string literal.
    Ok(Value::String(s.to_string()))
}

fn parse_array(inner: &str) -> Result<Value, KernelError> {
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    let values = split_top_level(inner, ',')
        .into_iter()
        .filter(|part| !part.trim().is_empty())
        .map(|part| parse_value(part.trim()))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Value::Array(values))
}

fn parse_object(inner: &str) -> Result<Value, KernelError> {
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    let mut map = Map::new();
    for entry in split_top_level(inner, ',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }

        let (key_raw, value_raw) = split_once_top_level(entry, ':').ok_or_else(|| {
            KernelError::NormalizerFailed(format!("LFM: object entry without ':': {entry:?}"))
        })?;
        let key = parse_object_key(key_raw.trim())?;
        if map.contains_key(&key) {
            return Err(KernelError::NormalizerFailed(format!(
                "LFM: duplicate object key: {key}"
            )));
        }

        map.insert(key, parse_value(value_raw.trim())?);
    }

    Ok(Value::Object(map))
}

fn parse_object_key(raw: &str) -> Result<String, KernelError> {
    match parse_value(raw)? {
        Value::String(key) => Ok(key),
        _ => Err(KernelError::NormalizerFailed(format!(
            "LFM: object key must be a string: {raw:?}"
        ))),
    }
}

/// Validate model-emitted identifiers before they reach dispatch. Tool names
/// allow the same separator characters commonly used in registries, while
/// keyword argument names stay simple and JSON-object friendly.
fn validate_identifier(kind: &str, value: &str) -> Result<(), KernelError> {
    let valid = value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'));

    if valid {
        return Ok(());
    }

    Err(KernelError::NormalizerFailed(format!(
        "invalid {kind}: {value:?}"
    )))
}

/// Split `s` at top-level occurrences of `delim` (i.e. not inside nested
/// brackets, parentheses, braces, or single/double quotes). Returns the
/// subslices between delimiters — including empty slices at the edges.
fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    let mut depth: usize = 0;
    let mut in_sq = false;
    let mut in_dq = false;
    let mut escape_next = false;
    let mut start = 0usize;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && (in_sq || in_dq) {
            escape_next = true;
            continue;
        }
        if in_sq {
            if ch == '\'' {
                in_sq = false;
            }
            continue;
        }
        if in_dq {
            if ch == '"' {
                in_dq = false;
            }
            continue;
        }
        match ch {
            '\'' => in_sq = true,
            '"' => in_dq = true,
            '(' | '[' | '{' => depth = depth.saturating_add(1),
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            c if c == delim && depth == 0 => {
                // i is always a char boundary from char_indices(); .get() is safe.
                parts.push(s.get(start..i).unwrap_or(""));
                start = i + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(s.get(start..).unwrap_or(""));
    parts
}

fn split_once_top_level(s: &str, delim: char) -> Option<(&str, &str)> {
    split_index_top_level(s, delim).map(|idx| {
        let left = s.get(..idx).unwrap_or("");
        let right = s.get(idx + delim.len_utf8()..).unwrap_or("");
        (left, right)
    })
}

fn split_index_top_level(s: &str, delim: char) -> Option<usize> {
    let mut depth: usize = 0;
    let mut in_sq = false;
    let mut in_dq = false;
    let mut escape_next = false;

    for (i, ch) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && (in_sq || in_dq) {
            escape_next = true;
            continue;
        }
        if in_sq {
            if ch == '\'' {
                in_sq = false;
            }
            continue;
        }
        if in_dq {
            if ch == '"' {
                in_dq = false;
            }
            continue;
        }
        match ch {
            '\'' => in_sq = true,
            '"' => in_dq = true,
            '(' | '[' | '{' => depth = depth.saturating_add(1),
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            c if c == delim && depth == 0 => return Some(i),
            _ => {}
        }
    }

    None
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LocalTool, ToolRegistry, ToolSchema};
    use serde_json::json;
    use std::sync::Arc;

    // ── is_applicable ──────────────────────────────────────────────────────

    #[test]
    fn not_applicable_for_plain_text() {
        assert!(!LfmNormalizer.is_applicable("hello world"));
    }

    #[test]
    fn applicable_when_start_marker_present() {
        assert!(
            LfmNormalizer
                .is_applicable("<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>")
        );
    }

    // ── normalize: clean inputs ────────────────────────────────────────────

    #[test]
    fn plain_text_returns_empty() {
        let calls = LfmNormalizer
            .normalize("The weather in Berlin is sunny.")
            .unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn single_call_string_arg() {
        let raw = "<|tool_call_start|>[get_weather(city='Berlin')]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args, json!({"city": "Berlin"}));
    }

    #[test]
    fn single_call_multiple_args() {
        let raw = "<|tool_call_start|>[search(query='rust async', limit=10)]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].args, json!({"query": "rust async", "limit": 10}));
    }

    #[test]
    fn single_call_no_args() {
        let raw = "<|tool_call_start|>[list_tools()]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "list_tools");
        assert_eq!(calls[0].args, json!({}));
    }

    #[test]
    fn multiple_calls_in_one_block() {
        let raw = "<|tool_call_start|>[get_weather(city='Berlin'), get_time(zone='UTC')]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args, json!({"city": "Berlin"}));
        assert_eq!(calls[1].name, "get_time");
        assert_eq!(calls[1].args, json!({"zone": "UTC"}));
    }

    #[test]
    fn multiple_blocks_in_one_message() {
        let raw = concat!(
            "<|tool_call_start|>[step_one(x=1)]<|tool_call_end|>",
            " some text ",
            "<|tool_call_start|>[step_two(y=2)]<|tool_call_end|>",
        );
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "step_one");
        assert_eq!(calls[1].name, "step_two");
    }

    #[test]
    fn block_without_brackets_is_parsed() {
        // Format without outer [ ] is also handled.
        let raw = "<|tool_call_start|>ping(target='8.8.8.8')<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ping");
        assert_eq!(calls[0].args, json!({"target": "8.8.8.8"}));
    }

    // ── value type coercion ────────────────────────────────────────────────

    #[test]
    fn integer_arg() {
        let raw = "<|tool_call_start|>[set_limit(n=42)]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls[0].args, json!({"n": 42}));
    }

    #[test]
    fn float_arg() {
        let raw = "<|tool_call_start|>[set_temp(t=0.7)]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls[0].args["t"].as_f64().unwrap(), 0.7);
    }

    #[test]
    fn boolean_args() {
        let raw = "<|tool_call_start|>[configure(verbose=True, strict=False)]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls[0].args, json!({"verbose": true, "strict": false}));
    }

    #[test]
    fn null_args() {
        let raw = "<|tool_call_start|>[reset(ctx=None)]<|tool_call_end|>";
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls[0].args, json!({"ctx": null}));
    }

    #[test]
    fn double_quoted_string_arg() {
        let raw = r#"<|tool_call_start|>[greet(name="world")]<|tool_call_end|>"#;
        let calls = LfmNormalizer.normalize(raw).unwrap();
        assert_eq!(calls[0].args, json!({"name": "world"}));
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

    #[test]
    fn openai_responses_function_call_item() {
        let value = json!({
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": "{\"city\":\"Berlin\"}",
            "status": "completed"
        });

        let calls = StructuredToolCallNormalizer::normalize_openai_responses(&value).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args, json!({"city": "Berlin"}));
    }

    #[test]
    fn openai_responses_full_response() {
        let value = json!({
            "id": "resp_123",
            "output": [
                { "type": "message", "content": [] },
                {
                    "type": "function_call",
                    "id": "fc_123",
                    "call_id": "call_123",
                    "name": "search.docs",
                    "arguments": {"query": "tool calls"},
                    "status": "completed"
                }
            ]
        });

        let calls = StructuredToolCallNormalizer::normalize_openai_responses(&value).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search.docs");
        assert_eq!(calls[0].args, json!({"query": "tool calls"}));
    }

    #[test]
    fn openai_chat_completions_tool_calls() {
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
                            "arguments": "{\"city\":\"Berlin\"}"
                        }
                    }]
                }
            }]
        });

        let calls =
            StructuredToolCallNormalizer::normalize_openai_chat_completions(&value).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].args, json!({"city": "Berlin"}));
    }

    #[test]
    fn structured_normalizer_aggregates_supported_shapes() {
        let responses_value = json!({
            "output": [{
                "type": "function_call",
                "name": "first",
                "arguments": "{}"
            }]
        });
        let chat_value = json!({
            "tool_calls": [{
                "function": {
                    "name": "second",
                    "arguments": {"ok": true}
                }
            }]
        });

        let responses_calls = StructuredToolCallNormalizer::normalize(&responses_value).unwrap();
        let chat_calls = StructuredToolCallNormalizer::normalize(&chat_value).unwrap();

        assert_eq!(responses_calls[0].name, "first");
        assert_eq!(chat_calls[0].name, "second");
        assert_eq!(chat_calls[0].args, json!({"ok": true}));
    }

    // ── error paths ────────────────────────────────────────────────────────

    #[test]
    fn unclosed_marker_returns_error() {
        let raw = "<|tool_call_start|>[get_weather(city='Berlin')]";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unclosed"), "expected 'unclosed' in: {msg}");
    }

    #[test]
    fn missing_paren_returns_error() {
        // Block with no '(' — not a valid call expression.
        let raw = "<|tool_call_start|>[not_a_call]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expected '('"), "got: {msg}");
    }

    #[test]
    fn kwarg_without_equals_returns_error() {
        let raw = "<|tool_call_start|>[fn(badarg)]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("kwarg without '='"), "got: {msg}");
    }

    #[test]
    fn invalid_tool_name_returns_error() {
        let raw = "<|tool_call_start|>[bad/name(arg=1)]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid tool name"), "got: {msg}");
    }

    #[test]
    fn empty_kwarg_name_returns_error() {
        let raw = "<|tool_call_start|>[fn(=1)]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("empty kwarg name"), "got: {msg}");
    }

    #[test]
    fn duplicate_kwarg_returns_error() {
        let raw = "<|tool_call_start|>[fn(city='Berlin', city='Paris')]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("duplicate kwarg"), "got: {msg}");
    }

    #[test]
    fn malformed_standard_arguments_return_error() {
        let value = json!({
            "type": "function_call",
            "name": "bad_args",
            "arguments": "{not json}"
        });

        let err = StructuredToolCallNormalizer::normalize_openai_responses(&value).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("arguments JSON"), "got: {msg}");
    }

    #[test]
    fn trailing_call_content_returns_error() {
        let raw = "<|tool_call_start|>[fn(arg=1) extra]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("trailing content"), "got: {msg}");
    }

    #[test]
    fn unterminated_nested_literal_returns_error() {
        let raw = "<|tool_call_start|>[fn(items=['a', 'b')]<|tool_call_end|>";
        let err = LfmNormalizer.normalize(raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unterminated list"), "got: {msg}");
    }

    #[tokio::test]
    async fn dispatch_invocations_runs_tools_in_order() {
        let tools = ToolRegistry::new();
        tools.register(Arc::new(LocalTool::new(
            ToolSchema {
                name: "echo".into(),
                description: "echoes args".into(),
                args_schema: json!({"type": "object"}),
                result_schema: json!({"type": "object"}),
            },
            |args| async move { Ok(json!({"seen": args})) },
        )));

        let invocations = LfmNormalizer
            .normalize("<|tool_call_start|>[echo(value={'nested': [1, 2]})]<|tool_call_end|>")
            .unwrap();
        let results = dispatch_tool_invocations(&tools, &invocations)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].invocation.name, "echo");
        assert_eq!(
            results[0].output,
            json!({"seen": {"value": {"nested": [1, 2]}}})
        );
    }

    // ── split_top_level helper ─────────────────────────────────────────────

    #[test]
    fn split_respects_parens() {
        // Comma inside parens must not split.
        let parts = split_top_level("fn(a, b), fn2(c)", ',');
        assert_eq!(parts, vec!["fn(a, b)", " fn2(c)"]);
    }

    #[test]
    fn split_respects_single_quotes() {
        let parts = split_top_level("a='x,y', b=2", ',');
        assert_eq!(parts, vec!["a='x,y'", " b=2"]);
    }

    #[test]
    fn split_respects_nested_arrays_and_objects() {
        let parts = split_top_level("a=[1, 2], b={'x': 'y,z'}, c=3", ',');
        assert_eq!(parts, vec!["a=[1, 2]", " b={'x': 'y,z'}", " c=3"]);
    }
}
