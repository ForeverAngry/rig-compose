//! Integration tests for [`rig_compose::DispatchTrace`].

use async_trait::async_trait;
use rig_compose::{
    DispatchTrace, DispatchTraceEvent, KernelError, LocalTool, ToolDispatchAction,
    ToolDispatchHook, ToolInvocation, ToolRegistry, ToolSchema, TracedAction, TracedOutcome,
    dispatch_tool_invocations_with_trace,
};
use serde_json::{Value, json};
use std::sync::Arc;

#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    // ── Test fixtures ────────────────────────────────────────────────────────

    fn echo_registry() -> ToolRegistry {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(LocalTool::new(
            ToolSchema {
                name: "echo".into(),
                description: "echo input".into(),
                args_schema: json!({"type": "object"}),
                result_schema: json!({"type": "object"}),
            },
            |args: Value| async move { Ok(args) },
        )));
        registry
    }

    fn echo_invocation() -> ToolInvocation {
        ToolInvocation::new("echo", json!({"v": 1})).unwrap()
    }

    /// Always returns `Skip`.
    struct SkipHook;
    #[async_trait]
    impl ToolDispatchHook for SkipHook {
        async fn before_invocation(
            &self,
            _invocation: &ToolInvocation,
        ) -> Result<ToolDispatchAction, KernelError> {
            Ok(ToolDispatchAction::Skip {
                output: json!({"skipped": true}),
                reason: Some("policy".to_string()),
            })
        }
    }

    /// Always returns `Terminate`.
    struct TerminateHook;
    #[async_trait]
    impl ToolDispatchHook for TerminateHook {
        async fn before_invocation(
            &self,
            _invocation: &ToolInvocation,
        ) -> Result<ToolDispatchAction, KernelError> {
            Ok(ToolDispatchAction::Terminate {
                reason: "limit hit".to_string(),
            })
        }
    }

    /// Errors in `before_invocation` so we can verify cleanup events.
    struct ErroringHook;
    #[async_trait]
    impl ToolDispatchHook for ErroringHook {
        async fn before_invocation(
            &self,
            _invocation: &ToolInvocation,
        ) -> Result<ToolDispatchAction, KernelError> {
            Err(KernelError::BudgetFailed("nope".into()))
        }
    }

    /// Always `Continue`. Used to verify cleanup ordering when a later hook errors.
    struct ReservingHook;
    #[async_trait]
    impl ToolDispatchHook for ReservingHook {
        async fn before_invocation(
            &self,
            _invocation: &ToolInvocation,
        ) -> Result<ToolDispatchAction, KernelError> {
            Ok(ToolDispatchAction::Continue)
        }
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn trace_records_continue_then_completed() {
        let registry = echo_registry();
        let invocations = vec![echo_invocation()];
        let reserving = ReservingHook;
        let hooks: Vec<&dyn ToolDispatchHook> = vec![&reserving];
        let trace = DispatchTrace::new();

        let results = dispatch_tool_invocations_with_trace(&registry, &invocations, &hooks, &trace)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let events = trace.events();
        assert_eq!(
            events,
            vec![
                DispatchTraceEvent::HookBefore {
                    invocation_index: 0,
                    hook_index: 0,
                    decision: TracedAction::Continue,
                },
                DispatchTraceEvent::HookAfter {
                    invocation_index: 0,
                    hook_index: 0,
                },
                DispatchTraceEvent::InvocationOutcome {
                    invocation_index: 0,
                    outcome: TracedOutcome::Completed,
                },
            ],
        );
    }

    #[tokio::test]
    async fn trace_records_skip_with_reason() {
        let registry = echo_registry();
        let invocations = vec![echo_invocation()];
        let skip = SkipHook;
        let hooks: Vec<&dyn ToolDispatchHook> = vec![&skip];
        let trace = DispatchTrace::new();

        let results = dispatch_tool_invocations_with_trace(&registry, &invocations, &hooks, &trace)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        let events = trace.events();
        assert!(events.contains(&DispatchTraceEvent::HookBefore {
            invocation_index: 0,
            hook_index: 0,
            decision: TracedAction::Skip {
                reason: Some("policy".to_string()),
            },
        }));
        assert!(events.contains(&DispatchTraceEvent::InvocationOutcome {
            invocation_index: 0,
            outcome: TracedOutcome::Skipped {
                reason: Some("policy".to_string()),
            },
        }));
    }

    #[tokio::test]
    async fn trace_records_terminate_emits_failure_and_no_completion() {
        let registry = echo_registry();
        let invocations = vec![echo_invocation()];
        let term = TerminateHook;
        let hooks: Vec<&dyn ToolDispatchHook> = vec![&term];
        let trace = DispatchTrace::new();

        let err = dispatch_tool_invocations_with_trace(&registry, &invocations, &hooks, &trace)
            .await
            .unwrap_err();
        assert!(matches!(err, KernelError::ToolDispatchTerminated(_)));

        let events = trace.events();
        // Terminate decision is recorded.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::HookBefore {
                decision: TracedAction::Terminate { .. },
                ..
            }
        )));
        // The dispatcher notified hooks for cleanup.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::HookCleanup {
                invocation_index: 0,
                hook_index: 0,
            }
        )));
        // Final outcome is Terminated with reason carried through.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::InvocationOutcome {
                outcome: TracedOutcome::Terminated { reason },
                ..
            } if reason == "limit hit"
        )));
    }

    #[tokio::test]
    async fn trace_records_hook_before_error_with_cleanup_subset() {
        // Reserving hook observes `before_invocation` and registers; the second
        // hook errors. The trace must record exactly one cleanup event for the
        // first hook, plus the hook error itself and a Failed outcome.
        let registry = echo_registry();
        let invocations = vec![echo_invocation()];
        let reserving = ReservingHook;
        let erroring = ErroringHook;
        let hooks: Vec<&dyn ToolDispatchHook> = vec![&reserving, &erroring];
        let trace = DispatchTrace::new();

        let err = dispatch_tool_invocations_with_trace(&registry, &invocations, &hooks, &trace)
            .await
            .unwrap_err();
        assert!(matches!(err, KernelError::BudgetFailed(_)));

        let events = trace.events();
        // HookBefore from hook 0.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::HookBefore {
                hook_index: 0,
                decision: TracedAction::Continue,
                ..
            }
        )));
        // HookBeforeError reported for hook 1.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::HookBeforeError {
                invocation_index: 0,
                hook_index: 1,
                ..
            }
        )));
        // Cleanup exactly for hook 0 (the one that observed before_invocation).
        let cleanups: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                DispatchTraceEvent::HookCleanup { hook_index, .. } => Some(*hook_index),
                _ => None,
            })
            .collect();
        assert_eq!(cleanups, vec![0]);
        // Failed outcome with the underlying error message.
        assert!(events.iter().any(|e| matches!(
            e,
            DispatchTraceEvent::InvocationOutcome {
                outcome: TracedOutcome::Failed { .. },
                ..
            }
        )));
    }
}
