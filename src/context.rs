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
