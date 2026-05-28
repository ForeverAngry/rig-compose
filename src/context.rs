//! [`InvestigationContext`] — the runtime object that flows through every
//! [`super::Skill`] in an agent step.
//!
//! Skills mutate the context by appending [`Evidence`] and adjusting
//! confidence; they do not own it. The owning [`super::Agent`] threads a
//! single context through its skill chain for one investigation.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Provider-neutral category for a piece of context that may enter a model
/// window.
///
/// The enum names where the item came from without coupling the kernel to a
/// concrete backend such as Memvid, MCP, a vector database, or a provider SDK.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContextSourceKind {
    /// Long-term memory, episodic recall, summaries, or structured memory cards.
    Memory,
    /// Result returned by a tool call.
    ToolResult,
    /// Resource lookup such as a graph, baseline, policy, or document store.
    Resource,
    /// File or document content selected for the task.
    File,
    /// Working notes, plans, hypotheses, or other non-durable reasoning state.
    Reasoning,
    /// System, developer, or application instructions carried into context.
    Instruction,
    /// Current user input or task text.
    UserInput,
    /// Caller-defined source kind.
    Other(String),
}

/// Provider-neutral lifecycle state for a projected context item.
///
/// Producer crates can attach this to [`ContextProvenance`] when the host needs
/// to explain why a candidate was expanded, skipped, suppressed, superseded, or
/// escalated before it reached [`ContextPack::pack`]. The packer still records
/// its own final [`ContextOmissionReason`] for items omitted by budget or item
/// count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextProjectionState {
    /// Candidate is eligible for packing.
    Candidate,
    /// Candidate was expanded from a source item into derived context.
    Expanded,
    /// Candidate was skipped before packing.
    Skipped,
    /// Candidate was suppressed by caller policy.
    Suppressed,
    /// Candidate was rejected by caller policy.
    Rejected,
    /// Candidate was superseded by a newer or more authoritative item.
    Superseded,
    /// Candidate is stale relative to a newer version.
    Stale,
    /// Candidate conflicts with another item and needs host resolution.
    Conflict,
    /// Candidate was escalated for higher-level handling.
    Escalated,
    /// Caller-defined state.
    Other(String),
}

/// Shared provenance keys for context projected by memory, resource, graph, or
/// tool-result producers.
///
/// `rig-compose` continues to store provenance on [`ContextItem`] as JSON so
/// downstream crates can attach crate-specific fields without depending on each
/// other. This helper gives those crates a common vocabulary for the fields that
/// matter to replay, evaluation, and omission explanations.
///
/// ```rust
/// use rig_compose::{ContextItem, ContextProvenance, ContextSourceKind};
///
/// let provenance = ContextProvenance::new()
///     .with_source_uri("memory://incident/42")
///     .with_principal("alice")
///     .with_scope("workspace")
///     .with_confidence(0.92);
///
/// let item = ContextItem::new(ContextSourceKind::Memory, "frame-42", "prior incident")
///     .with_context_provenance(provenance);
///
/// assert_eq!(
///     item.context_provenance().unwrap().source_uri.as_deref(),
///     Some("memory://incident/42")
/// );
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextProvenance {
    /// URI or locator for the original source record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_uri: Option<String>,
    /// Principal, actor, tenant, or subject associated with the source record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub principal: Option<String>,
    /// Caller-defined scope such as tenant, workspace, profile, or project.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Retention or archive tier associated with the source record.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_tier: Option<String>,
    /// Milliseconds since the Unix epoch when the source record was recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recorded_at_millis: Option<i64>,
    /// Milliseconds since the Unix epoch when the source record became
    /// effective for supersession or freshness comparisons.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_at_millis: Option<i64>,
    /// Source-provided confidence score, when it is distinct from
    /// [`ContextItem::score`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Stable key used to compare competing versions of the same fact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version_key: Option<String>,
    /// Source frame/document id used by memory stores and eval fixtures.
    ///
    /// Stored as JSON so existing producers can keep numeric frame ids while
    /// others use string document keys.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<Value>,
    /// Lifecycle state assigned before the packer makes final budget decisions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projection_state: Option<ContextProjectionState>,
    /// Machine-readable reason for the projection state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl ContextProvenance {
    /// Create empty provenance ready for builder-style population.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set [`Self::source_uri`].
    #[must_use]
    pub fn with_source_uri(mut self, source_uri: impl Into<String>) -> Self {
        self.source_uri = Some(source_uri.into());
        self
    }

    /// Set [`Self::principal`].
    #[must_use]
    pub fn with_principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = Some(principal.into());
        self
    }

    /// Set [`Self::scope`].
    #[must_use]
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Set [`Self::retention_tier`].
    #[must_use]
    pub fn with_retention_tier(mut self, retention_tier: impl Into<String>) -> Self {
        self.retention_tier = Some(retention_tier.into());
        self
    }

    /// Set [`Self::recorded_at_millis`].
    #[must_use]
    pub fn with_recorded_at_millis(mut self, recorded_at_millis: i64) -> Self {
        self.recorded_at_millis = Some(recorded_at_millis);
        self
    }

    /// Set [`Self::effective_at_millis`].
    #[must_use]
    pub fn with_effective_at_millis(mut self, effective_at_millis: i64) -> Self {
        self.effective_at_millis = Some(effective_at_millis);
        self
    }

    /// Set [`Self::confidence`].
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    /// Set [`Self::version_key`].
    #[must_use]
    pub fn with_version_key(mut self, version_key: impl Into<String>) -> Self {
        self.version_key = Some(version_key.into());
        self
    }

    /// Set [`Self::source_frame_id`].
    #[must_use]
    pub fn with_source_frame_id(mut self, source_frame_id: impl Into<String>) -> Self {
        self.source_frame_id = Some(Value::String(source_frame_id.into()));
        self
    }

    /// Set [`Self::source_frame_id`] from an existing JSON value.
    #[must_use]
    pub fn with_source_frame_id_value(mut self, source_frame_id: Value) -> Self {
        self.source_frame_id = Some(source_frame_id);
        self
    }

    /// Set [`Self::projection_state`].
    #[must_use]
    pub fn with_projection_state(mut self, projection_state: ContextProjectionState) -> Self {
        self.projection_state = Some(projection_state);
        self
    }

    /// Set [`Self::reason`].
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// One ranked piece of context that may be packed into a bounded model window.
///
/// `ContextItem` is intentionally backend-neutral. Memory crates, MCP/resource
/// adapters, and harnesses can all project their native records into this shape
/// so tests can assert what context was selected, omitted, and rendered.
///
/// ```rust
/// use rig_compose::{ContextItem, ContextSourceKind};
///
/// let item = ContextItem::new(
///     ContextSourceKind::Memory,
///     "profile/alice/location",
///     "fact alice lives in Berlin",
/// )
/// .with_rank(0)
/// .with_score(9.5);
///
/// assert_eq!(item.estimated_chars, item.text.chars().count());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextItem {
    /// Backend-neutral source category.
    pub source: ContextSourceKind,
    /// Stable id inside the source system.
    pub source_id: String,
    /// Zero-based rank after source-local selection.
    pub rank: usize,
    /// Relevance score used for ordering within the source or planner.
    pub score: f64,
    /// Prompt-ready text.
    pub text: String,
    /// Character count estimate for early context packing.
    pub estimated_chars: usize,
    /// Source-specific provenance such as frame id, URI, tool call id, or path.
    pub provenance: Value,
    /// Caller-defined metadata not required for packing.
    pub metadata: Value,
}

impl ContextItem {
    /// Build a context item with a source, source id, and prompt-ready text.
    #[must_use]
    pub fn new(
        source: ContextSourceKind,
        source_id: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        let text = text.into();
        Self {
            source,
            source_id: source_id.into(),
            rank: 0,
            score: 0.0,
            estimated_chars: text.chars().count(),
            text,
            provenance: Value::Null,
            metadata: Value::Null,
        }
    }

    /// Set the source-local rank used by [`ContextPack::pack`].
    #[must_use]
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set the relevance score attached by the source or planner.
    #[must_use]
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Override the character estimate when a caller has a better tokenizer or
    /// sizing approximation.
    #[must_use]
    pub fn with_estimated_chars(mut self, estimated_chars: usize) -> Self {
        self.estimated_chars = estimated_chars;
        self
    }

    /// Attach source-specific provenance.
    #[must_use]
    pub fn with_provenance(mut self, provenance: Value) -> Self {
        self.provenance = provenance;
        self
    }

    /// Attach source-specific provenance using the shared typed vocabulary.
    #[must_use]
    pub fn with_context_provenance(mut self, provenance: ContextProvenance) -> Self {
        self.provenance = serde_json::to_value(provenance).unwrap_or(Value::Null);
        self
    }

    /// Decode [`Self::provenance`] as the shared typed vocabulary.
    ///
    /// Returns an empty [`ContextProvenance`] when no provenance was attached.
    pub fn context_provenance(&self) -> serde_json::Result<ContextProvenance> {
        if self.provenance.is_null() {
            Ok(ContextProvenance::default())
        } else {
            serde_json::from_value(self.provenance.clone())
        }
    }

    /// Attach caller-defined metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Reason a context item was not selected for a [`ContextPack`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextOmissionReason {
    /// The pack already reached [`ContextPackConfig::max_items`].
    MaxItems,
    /// Adding the item would exceed the available character budget.
    OverBudget,
}

/// Context item plus the reason it was omitted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OmittedContextItem {
    /// Item considered by the packer.
    pub item: ContextItem,
    /// Why the item was not selected.
    pub reason: ContextOmissionReason,
}

/// Configuration for packing context items into a bounded model window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPackConfig {
    /// Maximum characters available to selected item text, including separators.
    pub max_chars: usize,
    /// Maximum number of items to include.
    pub max_items: usize,
    /// Characters reserved for instructions, user input, or other context.
    pub reserve_chars: usize,
    /// Separator inserted between selected item text when rendering.
    pub separator: String,
}

impl Default for ContextPackConfig {
    fn default() -> Self {
        Self {
            max_chars: 4_000,
            max_items: 16,
            reserve_chars: 0,
            separator: "\n".into(),
        }
    }
}

impl ContextPackConfig {
    /// Build a config with a character budget and otherwise default limits.
    #[must_use]
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            ..Self::default()
        }
    }

    /// Set the maximum number of selected items.
    #[must_use]
    pub fn with_max_items(mut self, max_items: usize) -> Self {
        self.max_items = max_items;
        self
    }

    /// Reserve part of the character budget for non-packed context.
    #[must_use]
    pub fn with_reserve_chars(mut self, reserve_chars: usize) -> Self {
        self.reserve_chars = reserve_chars;
        self
    }

    /// Use a custom separator when rendering selected context.
    #[must_use]
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    fn context_budget(&self) -> usize {
        self.max_chars.saturating_sub(self.reserve_chars)
    }
}

/// Selected and omitted context for one bounded model window.
///
/// ```rust
/// use rig_compose::{ContextItem, ContextPack, ContextPackConfig, ContextSourceKind};
///
/// let item = ContextItem::new(ContextSourceKind::Memory, "m1", "fact alice lives in Berlin");
/// let pack = ContextPack::pack(vec![item], ContextPackConfig::new(1_000));
/// assert_eq!(pack.render_text(), "fact alice lives in Berlin");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextPack {
    /// Configuration used to build this pack.
    pub config: ContextPackConfig,
    /// Items selected for prompt context, in render order.
    pub selected: Vec<ContextItem>,
    /// Items considered but omitted, with explicit reasons.
    pub omitted: Vec<OmittedContextItem>,
    /// Estimated characters consumed by selected text and separators.
    pub total_estimated_chars: usize,
}

impl ContextPack {
    /// Pack ranked context items into the configured character window.
    ///
    /// Items are sorted by `rank` before packing so recorded fixtures can be
    /// replayed even if a source returns equivalent items in a different order.
    #[must_use]
    pub fn pack(mut items: Vec<ContextItem>, config: ContextPackConfig) -> Self {
        items.sort_by_key(|item| item.rank);

        let budget = config.context_budget();
        let separator_chars = config.separator.chars().count();
        let mut selected = Vec::new();
        let mut omitted = Vec::new();
        let mut total_estimated_chars = 0usize;

        for item in items {
            if selected.len() >= config.max_items {
                omitted.push(OmittedContextItem {
                    item,
                    reason: ContextOmissionReason::MaxItems,
                });
                continue;
            }

            let item_chars = item.estimated_chars.max(item.text.chars().count());
            let separator_cost = if selected.is_empty() {
                0
            } else {
                separator_chars
            };
            let Some(next_total) = total_estimated_chars
                .checked_add(separator_cost)
                .and_then(|total| total.checked_add(item_chars))
            else {
                omitted.push(OmittedContextItem {
                    item,
                    reason: ContextOmissionReason::OverBudget,
                });
                continue;
            };

            if next_total > budget {
                omitted.push(OmittedContextItem {
                    item,
                    reason: ContextOmissionReason::OverBudget,
                });
                continue;
            }

            total_estimated_chars = next_total;
            selected.push(item);
        }

        Self {
            config,
            selected,
            omitted,
            total_estimated_chars,
        }
    }

    /// Render selected item text as prompt-ready context.
    #[must_use]
    pub fn render_text(&self) -> String {
        self.selected
            .iter()
            .map(|item| item.text.as_str())
            .collect::<Vec<_>>()
            .join(&self.config.separator)
    }
}

/// A named, lightweight signal lifted from a sketch, baseline check, or
/// upstream skill. Skills key their `applies` predicate on signal names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Signal(pub String);

impl Signal {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A single piece of evidence accumulated during an investigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub source_skill: String,
    pub label: String,
    pub detail: Value,
    pub recorded_at: SystemTime,
}

impl Evidence {
    pub fn new(source_skill: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            source_skill: source_skill.into(),
            label: label.into(),
            detail: Value::Null,
            recorded_at: SystemTime::now(),
        }
    }

    pub fn with_detail(mut self, detail: Value) -> Self {
        self.detail = detail;
        self
    }
}

/// Hint a skill may emit to drive subsequent skill selection. The agent
/// loop is free to honour or ignore these — they are advisory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NextAction {
    /// Suggest a follow-up skill by id.
    RunSkill(String),
    /// Suggest invoking a named tool with prepared args.
    InvokeTool { tool: String, args: Value },
    /// Stop the investigation; sufficient evidence has been gathered.
    Conclude,
    /// Drop the investigation; the entity is benign.
    Discard,
}

/// Runtime state for one investigation. Cheap to construct; passed by
/// `&mut` reference through the skill chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestigationContext {
    /// Stable identifier for the entity under investigation. May be a block
    /// id stringified, an actor id from the grammar layer (Phase 2), or any
    /// caller-defined key.
    pub entity_id: String,

    /// Optional originating block — present when the investigation was
    /// triggered by an upstream pipeline. Stored as an opaque UUID so the
    /// kernel does not depend on any specific block-id newtype.
    pub block_id: Option<Uuid>,

    /// Free-form partition tag (caller-defined).
    pub partition: String,

    /// Signals that triggered this investigation and any signals lifted by
    /// earlier skills. Skills add to this set as evidence accumulates.
    pub signals: Vec<Signal>,

    /// Accumulated evidence in chronological order.
    pub evidence: Vec<Evidence>,

    /// Running confidence in `[0, 1]` that the entity exhibits malicious
    /// behaviour. Skills emit deltas; the agent clamps after each step.
    pub confidence: f32,

    /// Hints from the most recently executed skill.
    pub pending_actions: Vec<NextAction>,
}

impl InvestigationContext {
    pub fn new(entity_id: impl Into<String>, partition: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            block_id: None,
            partition: partition.into(),
            signals: Vec::new(),
            evidence: Vec::new(),
            confidence: 0.0,
            pending_actions: Vec::new(),
        }
    }

    pub fn with_block<I: Into<Uuid>>(mut self, id: I) -> Self {
        self.block_id = Some(id.into());
        self
    }

    pub fn with_signal(mut self, s: impl Into<String>) -> Self {
        self.signals.push(Signal::new(s));
        self
    }

    pub fn has_signal(&self, name: &str) -> bool {
        self.signals.iter().any(|s| s.as_str() == name)
    }
}
