//! Agent persona / instruction set.
//!
//! [`Instructions`] is plain data — no LLM dependency. Cognition
//! backends (e.g. `RigCognition` in azreal) consume it to produce a
//! system prompt and constrain output shape; offline backends can
//! ignore it.
//!
//! This is the seam that lets a user reuse an agent's *machinery* (the
//! pipeline, sampler, skills, memory) while supplying their own
//! persona, examples, and response schema.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Static persona + response contract for a cognition backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Instructions {
    /// System prompt. Sent verbatim as the system message.
    pub system_prompt: String,
    /// Optional JSON schema the backend must conform to. Backends that
    /// support structured outputs should enforce this; others may use
    /// it as a hint in the system prompt.
    pub response_schema: Option<Value>,
    /// Few-shot examples. Each entry is `(user_message, assistant_reply)`.
    /// Empty for zero-shot prompting.
    pub examples: Vec<(String, String)>,
    /// Free-form metadata (model name, temperature hint, persona tag).
    /// Backends decide which keys, if any, to honour.
    pub metadata: Value,
}

impl Instructions {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            response_schema: None,
            examples: Vec::new(),
            metadata: Value::Null,
        }
    }

    pub fn with_response_schema(mut self, schema: Value) -> Self {
        self.response_schema = Some(schema);
        self
    }

    pub fn with_example(mut self, user: impl Into<String>, assistant: impl Into<String>) -> Self {
        self.examples.push((user.into(), assistant.into()));
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builder_chains() {
        let i = Instructions::new("you are a detector")
            .with_response_schema(json!({"type": "object"}))
            .with_example("hi", "hello")
            .with_metadata(json!({"model": "x"}));
        assert_eq!(i.system_prompt, "you are a detector");
        assert!(i.response_schema.is_some());
        assert_eq!(i.examples.len(), 1);
        assert_eq!(i.metadata["model"], "x");
    }

    #[test]
    fn round_trips_serde() {
        let i = Instructions::new("x").with_example("a", "b");
        let s = serde_json::to_string(&i).unwrap();
        let back: Instructions = serde_json::from_str(&s).unwrap();
        assert_eq!(back.examples.len(), 1);
    }
}
