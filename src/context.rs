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
