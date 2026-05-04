//! [`Workflow`] — composes one or more [`super::Agent`]s into a higher-level
//! orchestration unit (e.g. the Overseer's block-stream pump, or the
//! Phase-5 coordinator/specialist routing).
//!
//! Workflows are intentionally generic: their `Input` and `Output` types
//! are associated, so each concrete workflow defines its own contract.

use async_trait::async_trait;

use crate::registry::KernelError;

#[async_trait]
pub trait Workflow: Send + Sync {
    type Input: Send;
    type Output: Send;

    fn name(&self) -> &str;

    async fn run(&self, input: Self::Input) -> Result<Self::Output, KernelError>;
}
