//! Reliability primitives for tool dispatch loops.
//!
//! This module groups three small, host-driven utilities that downstream
//! agents combine when wrapping a normalized tool loop:
//!
//! 1. [`RetryClass`] + [`RetryClassifier`] — turn a [`KernelError`] into a
//!    deterministic "should I retry this call?" verdict. The default impl
//!    ([`DefaultRetryClassifier`]) covers every existing
//!    `KernelError` variant; hosts can supply a custom classifier when they
//!    layer in transport-specific errors (timeouts, rate limits, etc.) by
//!    chaining or overriding.
//! 2. [`ToolCallFingerprint`] — a stable, content-addressed hash of a
//!    [`ToolInvocation`] (tool name + canonical JSON args). Used to detect
//!    repeated calls and group retry attempts.
//! 3. [`HistoryEntry`] + [`repair_history`] — deterministic coalescing of
//!    a raw `(invocation, outcome)` sequence into the smallest history the
//!    model should see. Multiple retries of the same fingerprint collapse to
//!    a single canonical entry; the host stays in control of how many
//!    physical retries actually happened.
//! 4. [`StuckLoopHook`] — a [`ToolDispatchHook`] that tracks invocation
//!    fingerprints and terminates dispatch if the model requests identical
//!    calls too many times, preventing runaway costs.
//!
//! These primitives are intentionally synchronous and infallible: they
//! operate on already-materialized invocations and outcomes, never on live
//! transports.
//!
//! # Example
//!
//! ```no_run
//! use rig_compose::{
//!     DefaultRetryClassifier, HistoryEntry, KernelError, RetryClass,
//!     RetryClassifier, ToolInvocation, repair_history,
//! };
//! use serde_json::json;
//!
//! let classifier = DefaultRetryClassifier;
//! let inv = ToolInvocation::new("search", json!({"q": "rig"})).expect("valid");
//! let history = vec![
//!     HistoryEntry::Failed {
//!         invocation: inv.clone(),
//!         class: classifier.classify(&KernelError::ToolFailed("timeout".into())),
//!         message: "timeout".into(),
//!     },
//!     HistoryEntry::Completed {
//!         invocation: inv,
//!         output: json!({"hits": 3}),
//!     },
//! ];
//! let repaired = repair_history(&history);
//! assert_eq!(repaired.len(), 1);
//! assert!(matches!(repaired[0], HistoryEntry::Completed { .. }));
//! ```

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;

use crate::normalizer::{
    ToolDispatchAction, ToolDispatchHook, ToolInvocation, ToolInvocationResult,
};
use crate::registry::KernelError;

// ── Retry classification ─────────────────────────────────────────────────────

/// Deterministic verdict on whether an errored tool invocation may be retried.
///
/// `Transient` means the failure was likely environmental (network blip, flaky
/// dependency) and a retry has a real chance of succeeding. `Permanent` means
/// the inputs or policy are wrong and retrying would just waste the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RetryClass {
    /// Retry is allowed; the failure is likely environmental.
    Transient,
    /// Retry is forbidden; the failure is intrinsic to the inputs or policy.
    Permanent,
}

/// Classify a [`KernelError`] as transient or permanent.
///
/// Hosts can implement this trait to inject transport-specific knowledge
/// (e.g. mapping HTTP 5xx onto `Transient` and 4xx onto `Permanent`). The
/// crate ships [`DefaultRetryClassifier`] as a starting point that covers
/// every current `KernelError` variant.
pub trait RetryClassifier: Send + Sync {
    /// Return the retry verdict for `error`.
    fn classify(&self, error: &KernelError) -> RetryClass;
}

/// Default classifier covering every [`KernelError`] variant.
///
/// The mapping is conservative: anything that *could* be a flake (the tool
/// body errored, a skill body errored) is `Transient`. Everything that
/// signals a permanent disagreement (auth, missing names, invalid args,
/// budget exhaustion, dispatch termination, JSON parse errors) is
/// `Permanent`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultRetryClassifier;

impl RetryClassifier for DefaultRetryClassifier {
    fn classify(&self, error: &KernelError) -> RetryClass {
        match error {
            // Body errors — likely environmental, may succeed on retry.
            KernelError::ToolFailed(_) | KernelError::SkillFailed(_) => RetryClass::Transient,

            // Configuration / policy / argument errors — retrying with the
            // same inputs cannot help.
            KernelError::ToolNotFound(_)
            | KernelError::ToolNotAuthorised(_)
            | KernelError::SkillNotFound(_)
            | KernelError::ToolNotApplicable(_)
            | KernelError::InvalidArgument(_)
            | KernelError::NormalizerFailed(_)
            | KernelError::ToolDispatchTerminated(_)
            | KernelError::BudgetFailed(_)
            | KernelError::Serde(_) => RetryClass::Permanent,
        }
    }
}

// ── Fingerprints ─────────────────────────────────────────────────────────────

/// Stable content hash of a [`ToolInvocation`].
///
/// Two invocations with the same tool name and the same canonical JSON
/// arguments produce the same fingerprint, regardless of which dispatch
/// attempt produced them. Uses `std::collections::hash_map::DefaultHasher`,
/// so values are stable within a single process run but should not be
/// persisted across versions.
///
/// Determinism relies on `serde_json::Map`'s default key ordering
/// (`BTreeMap`-backed). If a downstream crate enables the `preserve_order`
/// feature globally, fingerprints will no longer be argument-order-independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ToolCallFingerprint(pub u64);

impl ToolInvocation {
    /// Return a stable fingerprint over `(name, canonical(args))`.
    ///
    /// Suitable as a hash-map key to group retry attempts of the same
    /// logical call, or to detect a stuck-loop pattern where the model
    /// keeps reissuing identical calls.
    pub fn fingerprint(&self) -> ToolCallFingerprint {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        // Serializing through serde_json gives a canonical form because
        // `serde_json::Map` is BTreeMap-backed without `preserve_order`.
        canonicalize_value(&self.args).to_string().hash(&mut hasher);
        ToolCallFingerprint(hasher.finish())
    }
}

/// Re-serialize `value` to a canonical form for hashing.
///
/// `serde_json::Value::to_string` is already deterministic when the
/// `preserve_order` feature is off, but going through a normalize step
/// makes that contract explicit and gives us a single hook to add float
/// canonicalization later if needed.
fn canonicalize_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonicalize_value).collect()),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, inner) in map {
                out.insert(key.clone(), canonicalize_value(inner));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

// ── Stuck loop detection ───────────────────────────────────────────────────

/// A dispatch hook that cuts off repeated identical tool calls.
///
/// Models sometimes enter a stuck loop where they repeatedly dispatch the
/// exact same tool invocation (same tool name, same arguments) without
/// making progress. This hook tracks invocation fingerprints across
/// dispatches and terminates the loop if the same fingerprint is invoked
/// more than `max_identical_calls` times.
pub struct StuckLoopHook {
    max_identical_calls: usize,
    counts: Mutex<HashMap<ToolCallFingerprint, usize>>,
}

impl StuckLoopHook {
    /// Create a new hook that terminates dispatch if any fingerprint
    /// is requested more than `max_identical_calls` times.
    pub fn new(max_identical_calls: usize) -> Self {
        Self {
            max_identical_calls,
            counts: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl ToolDispatchHook for StuckLoopHook {
    async fn before_invocation(
        &self,
        invocation: &ToolInvocation,
    ) -> Result<ToolDispatchAction, KernelError> {
        let fingerprint = invocation.fingerprint();
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let count = counts.entry(fingerprint).or_insert(0);
        *count = count.saturating_add(1);

        if *count > self.max_identical_calls {
            Ok(ToolDispatchAction::Terminate {
                reason: format!(
                    "stuck loop detected: `{}` has been called {} times",
                    invocation.name, count
                ),
            })
        } else {
            Ok(ToolDispatchAction::Continue)
        }
    }
}

// ── History repair ───────────────────────────────────────────────────────────

/// One entry in a tool-call history slice fed back to the model.
#[derive(Debug, Clone, PartialEq)]
pub enum HistoryEntry {
    /// The invocation completed and produced `output`.
    Completed {
        /// The invocation that ran.
        invocation: ToolInvocation,
        /// The JSON result the tool returned.
        output: Value,
    },
    /// The invocation failed; `class` records the retry verdict and
    /// `message` carries the error rendering.
    Failed {
        /// The invocation that failed.
        invocation: ToolInvocation,
        /// Retry verdict from a [`RetryClassifier`].
        class: RetryClass,
        /// `error.to_string()` of the underlying [`KernelError`].
        message: String,
    },
}

impl HistoryEntry {
    /// Return the fingerprint of this entry's invocation.
    pub fn fingerprint(&self) -> ToolCallFingerprint {
        match self {
            HistoryEntry::Completed { invocation, .. }
            | HistoryEntry::Failed { invocation, .. } => invocation.fingerprint(),
        }
    }

    /// Convenience: build a `Completed` entry from a [`ToolInvocationResult`].
    pub fn completed(result: ToolInvocationResult) -> Self {
        HistoryEntry::Completed {
            invocation: result.invocation,
            output: result.output,
        }
    }

    /// Convenience: classify `error` with `classifier` and build a `Failed`
    /// entry that records the verdict and the error rendering.
    pub fn failed<C: RetryClassifier>(
        invocation: ToolInvocation,
        error: &KernelError,
        classifier: &C,
    ) -> Self {
        HistoryEntry::Failed {
            invocation,
            class: classifier.classify(error),
            message: error.to_string(),
        }
    }
}

/// Deterministically coalesce a tool-call history.
///
/// The repair rule is:
///
/// 1. Walk `entries` in order, grouping by [`ToolCallFingerprint`].
/// 2. For each group, if **any** entry is [`HistoryEntry::Completed`],
///    keep the **first** completion (idempotent: once we have a real
///    answer, later retries don't change the story).
/// 3. Otherwise keep the **last** [`HistoryEntry::Failed`] for that
///    fingerprint (most recent verdict wins for terminal failures).
/// 4. Emit results in **first-occurrence order** of each fingerprint.
///
/// The transform is total, deterministic, and idempotent
/// (`repair_history(repair_history(x)) == repair_history(x)`).
pub fn repair_history(entries: &[HistoryEntry]) -> Vec<HistoryEntry> {
    // Track first-seen position per fingerprint so output preserves order,
    // and the chosen entry index per fingerprint.
    let mut order: Vec<ToolCallFingerprint> = Vec::new();
    let mut chosen: std::collections::HashMap<ToolCallFingerprint, usize> =
        std::collections::HashMap::new();
    let mut has_completed: std::collections::HashSet<ToolCallFingerprint> =
        std::collections::HashSet::new();

    for (idx, entry) in entries.iter().enumerate() {
        let fp = entry.fingerprint();
        if let std::collections::hash_map::Entry::Vacant(slot) = chosen.entry(fp) {
            order.push(fp);
            slot.insert(idx);
            if matches!(entry, HistoryEntry::Completed { .. }) {
                has_completed.insert(fp);
            }
            continue;
        }
        match entry {
            HistoryEntry::Completed { .. } => {
                // Rule 2: keep the *first* completion. If we already have
                // one chosen and it's a completion, skip. If the chosen
                // one is a failure, replace it.
                if !has_completed.contains(&fp) {
                    chosen.insert(fp, idx);
                    has_completed.insert(fp);
                }
            }
            HistoryEntry::Failed { .. } => {
                // Rule 3: last failure wins, but only if we don't already
                // have a completion locked in.
                if !has_completed.contains(&fp) {
                    chosen.insert(fp, idx);
                }
            }
        }
    }

    order
        .into_iter()
        .filter_map(|fp| chosen.get(&fp).and_then(|&i| entries.get(i)).cloned())
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use serde_json::json;

    fn inv(name: &str, args: Value) -> ToolInvocation {
        ToolInvocation::new(name, args).unwrap()
    }

    // ── Classifier ───────────────────────────────────────────────────────

    #[test]
    fn default_classifier_marks_tool_failed_transient() {
        let c = DefaultRetryClassifier;
        assert_eq!(
            c.classify(&KernelError::ToolFailed("boom".into())),
            RetryClass::Transient,
        );
        assert_eq!(
            c.classify(&KernelError::SkillFailed("boom".into())),
            RetryClass::Transient,
        );
    }

    #[test]
    fn default_classifier_marks_policy_errors_permanent() {
        let c = DefaultRetryClassifier;
        for err in [
            KernelError::ToolNotFound("x".into()),
            KernelError::ToolNotAuthorised("x".into()),
            KernelError::SkillNotFound("x".into()),
            KernelError::ToolNotApplicable("x".into()),
            KernelError::InvalidArgument("x".into()),
            KernelError::NormalizerFailed("x".into()),
            KernelError::ToolDispatchTerminated("x".into()),
            KernelError::BudgetFailed("x".into()),
        ] {
            assert_eq!(c.classify(&err), RetryClass::Permanent, "{err:?}");
        }
    }

    // ── Fingerprints ────────────────────────────────────────────────────

    #[test]
    fn fingerprint_is_stable_for_same_invocation() {
        let a = inv("search", json!({"q": "rig", "limit": 5}));
        let b = inv("search", json!({"q": "rig", "limit": 5}));
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_is_order_independent_for_object_args() {
        let a = inv("search", json!({"q": "rig", "limit": 5}));
        let b = inv("search", json!({"limit": 5, "q": "rig"}));
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_differs_when_args_differ() {
        let a = inv("search", json!({"q": "rig"}));
        let b = inv("search", json!({"q": "tokio"}));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_differs_when_tool_name_differs() {
        let a = inv("search", json!({"q": "rig"}));
        let b = inv("lookup", json!({"q": "rig"}));
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    // ── History repair ──────────────────────────────────────────────────

    #[test]
    fn repair_keeps_first_completion_after_retries() {
        let i = inv("search", json!({"q": "rig"}));
        let history = vec![
            HistoryEntry::Failed {
                invocation: i.clone(),
                class: RetryClass::Transient,
                message: "timeout".into(),
            },
            HistoryEntry::Completed {
                invocation: i.clone(),
                output: json!({"hits": 1}),
            },
            HistoryEntry::Completed {
                invocation: i,
                output: json!({"hits": 99}),
            },
        ];
        let repaired = repair_history(&history);
        assert_eq!(repaired.len(), 1);
        match &repaired[0] {
            HistoryEntry::Completed { output, .. } => assert_eq!(output, &json!({"hits": 1})),
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    #[test]
    fn repair_keeps_last_failure_when_no_completion() {
        let i = inv("search", json!({"q": "rig"}));
        let history = vec![
            HistoryEntry::Failed {
                invocation: i.clone(),
                class: RetryClass::Transient,
                message: "first".into(),
            },
            HistoryEntry::Failed {
                invocation: i,
                class: RetryClass::Permanent,
                message: "last".into(),
            },
        ];
        let repaired = repair_history(&history);
        assert_eq!(repaired.len(), 1);
        match &repaired[0] {
            HistoryEntry::Failed { message, class, .. } => {
                assert_eq!(message, "last");
                assert_eq!(*class, RetryClass::Permanent);
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn repair_preserves_first_occurrence_order_across_fingerprints() {
        let a = inv("a", json!({"k": 1}));
        let b = inv("b", json!({"k": 2}));
        let history = vec![
            HistoryEntry::Completed {
                invocation: a.clone(),
                output: json!(null),
            },
            HistoryEntry::Completed {
                invocation: b.clone(),
                output: json!(null),
            },
            HistoryEntry::Completed {
                invocation: a,
                output: json!("ignored"),
            },
        ];
        let repaired = repair_history(&history);
        assert_eq!(repaired.len(), 2);
        // First entry is the first occurrence of `a`.
        assert_eq!(
            repaired[0].fingerprint(),
            inv("a", json!({"k": 1})).fingerprint()
        );
        assert_eq!(
            repaired[1].fingerprint(),
            inv("b", json!({"k": 2})).fingerprint()
        );
    }

    #[test]
    fn repair_is_idempotent() {
        let i = inv("search", json!({"q": "rig"}));
        let history = vec![
            HistoryEntry::Failed {
                invocation: i.clone(),
                class: RetryClass::Transient,
                message: "x".into(),
            },
            HistoryEntry::Completed {
                invocation: i,
                output: json!({"ok": true}),
            },
        ];
        let once = repair_history(&history);
        let twice = repair_history(&once);
        assert_eq!(once, twice);
    }

    #[test]
    fn repair_on_empty_history_returns_empty() {
        assert!(repair_history(&[]).is_empty());
    }

    #[test]
    fn history_entry_failed_helper_records_classifier_verdict() {
        let entry = HistoryEntry::failed(
            inv("search", json!({"q": "rig"})),
            &KernelError::ToolFailed("flake".into()),
            &DefaultRetryClassifier,
        );
        match entry {
            HistoryEntry::Failed { class, message, .. } => {
                assert_eq!(class, RetryClass::Transient);
                assert!(message.contains("flake"));
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    // ── Stuck loop detection tests ───────────────────────────────────────────

    #[tokio::test]
    async fn stuck_loop_terminates_after_threshold() {
        let hook = StuckLoopHook::new(2);
        let i = inv("search", json!({"q": "rig"}));

        // First call: allowed
        let action = hook.before_invocation(&i).await.unwrap();
        assert!(matches!(action, ToolDispatchAction::Continue));

        // Second call: allowed
        let action = hook.before_invocation(&i).await.unwrap();
        assert!(matches!(action, ToolDispatchAction::Continue));

        // Third call: terminated
        let action = hook.before_invocation(&i).await.unwrap();
        match action {
            ToolDispatchAction::Terminate { reason } => {
                assert!(reason.contains("stuck loop detected"));
                assert!(reason.contains("3 times"));
            }
            other => panic!("expected Terminate, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stuck_loop_tracks_multiple_distinct_tools() {
        let hook = StuckLoopHook::new(1);
        let i1 = inv("search", json!({"q": "1"}));
        let i2 = inv("search", json!({"q": "2"}));

        assert!(matches!(
            hook.before_invocation(&i1).await.unwrap(),
            ToolDispatchAction::Continue
        ));
        assert!(matches!(
            hook.before_invocation(&i2).await.unwrap(),
            ToolDispatchAction::Continue
        ));

        // Calling i1 again triggers termination
        assert!(matches!(
            hook.before_invocation(&i1).await.unwrap(),
            ToolDispatchAction::Terminate { .. }
        ));

        // Calling i2 again triggers termination
        assert!(matches!(
            hook.before_invocation(&i2).await.unwrap(),
            ToolDispatchAction::Terminate { .. }
        ));
    }
}
