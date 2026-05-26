//! Dispatch trace records.
//!
//! [`DispatchTrace`] is an additive, deterministic record of what happened
//! during a [`dispatch_tool_invocations_with_trace`](crate::dispatch_tool_invocations_with_trace)
//! call: each hook's decision, the eventual invocation outcome, errors, and
//! reservation cleanups. Hosts use it to explain a tool-loop replay or to
//! export structured policy traces without depending on a concrete tracing
//! backend.
//!
//! # Example
//!
//! ```no_run
//! use rig_compose::{
//!     DispatchTrace, ToolRegistry, dispatch_tool_invocations_with_trace,
//! };
//!
//! # async fn run(registry: ToolRegistry, invocations: Vec<rig_compose::ToolInvocation>) -> Result<(), rig_compose::KernelError> {
//! let trace = DispatchTrace::new();
//! let _results =
//!     dispatch_tool_invocations_with_trace(&registry, &invocations, &[], &trace).await?;
//! for event in trace.events() {
//!     tracing::debug!(?event, "dispatch trace event");
//! }
//! # Ok(()) }
//! ```

use std::sync::{Arc, Mutex};

use crate::normalizer::ToolDispatchAction;

/// One observable event within a dispatch trace. Events are pushed in order;
/// `invocation_index` and `hook_index` refer to positions in the slices passed
/// to
/// [`dispatch_tool_invocations_with_trace`](crate::dispatch_tool_invocations_with_trace).
#[derive(Debug, Clone, PartialEq)]
pub enum DispatchTraceEvent {
    /// A hook's `before_invocation` returned a decision.
    HookBefore {
        invocation_index: usize,
        hook_index: usize,
        decision: TracedAction,
    },
    /// A hook's `before_invocation` returned an error and aborted dispatch.
    HookBeforeError {
        invocation_index: usize,
        hook_index: usize,
        message: String,
    },
    /// A hook's `on_invocation_error` was notified so it could release
    /// resources reserved in `before_invocation`.
    HookCleanup {
        invocation_index: usize,
        hook_index: usize,
    },
    /// A hook's `after_invocation_with_outcome` ran successfully.
    HookAfter {
        invocation_index: usize,
        hook_index: usize,
    },
    /// The invocation finished with a final outcome.
    InvocationOutcome {
        invocation_index: usize,
        outcome: TracedOutcome,
    },
}

/// Lightweight projection of [`ToolDispatchAction`] suitable for trace
/// recording. Decouples the trace shape from the runtime action enum so adding
/// fields to [`ToolDispatchAction`] does not silently widen trace records.
#[derive(Debug, Clone, PartialEq)]
pub enum TracedAction {
    Continue,
    Skip { reason: Option<String> },
    Terminate { reason: String },
}

impl From<&ToolDispatchAction> for TracedAction {
    fn from(value: &ToolDispatchAction) -> Self {
        match value {
            ToolDispatchAction::Continue => TracedAction::Continue,
            ToolDispatchAction::Skip { reason, .. } => TracedAction::Skip {
                reason: reason.clone(),
            },
            ToolDispatchAction::Terminate { reason } => TracedAction::Terminate {
                reason: reason.clone(),
            },
        }
    }
}

/// Final outcome recorded for one invocation.
#[derive(Debug, Clone, PartialEq)]
pub enum TracedOutcome {
    /// The tool body ran and produced a result.
    Completed,
    /// A hook synthesised a skip result.
    Skipped { reason: Option<String> },
    /// A hook stopped the dispatch loop.
    Terminated { reason: String },
    /// The tool body or a hook returned an error.
    Failed { message: String },
}

/// Append-only collector for [`DispatchTraceEvent`] values.
///
/// Clones share the same underlying buffer (Arc + Mutex), so a trace handed to
/// the dispatcher and a trace inspected by the host see the same events.
#[derive(Debug, Default, Clone)]
pub struct DispatchTrace {
    inner: Arc<Mutex<Vec<DispatchTraceEvent>>>,
}

impl DispatchTrace {
    /// Create an empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an event. Used by the dispatcher; hosts normally read via
    /// [`Self::events`].
    pub(crate) fn push(&self, event: DispatchTraceEvent) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.push(event);
        }
    }

    /// Snapshot the events recorded so far.
    pub fn events(&self) -> Vec<DispatchTraceEvent> {
        self.inner
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Whether no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
