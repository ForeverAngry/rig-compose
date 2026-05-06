//! Cost-bounded coordination primitives.
//!
//! The kernel often needs to gate dispatch on a finite budget — rows
//! parsed, LLM tokens spent, dollars burned, etc. This module exposes
//! two domain-neutral traits and lock-free reference implementations:
//!
//! - [`BudgetGuard`] — generic unit-cost reservations (rows, requests,
//!   queue slots).
//! - [`TokenBudget`] — LLM-token accounting with optimistic reservation
//!   and after-the-fact reconciliation against provider-reported usage.
//!
//! Both traits are intentionally narrow so a single coordinator can
//! compose multiple budget policies (e.g. a row guard *and* a token
//! guard) without coupling to any particular backend.
//!
//! ## Why two traits?
//!
//! [`BudgetGuard`] is a symmetric reserve/release pair: callers know the
//! cost up front and either commit or roll back. [`TokenBudget`] is
//! optimistic: callers reserve an estimate and receive a
//! [`TokenReservation`], send the prompt, then pass that reservation to
//! [`TokenBudget::record_usage`] with the provider's reported totals to
//! reconcile the over- or under-estimate for that specific call.
//!
//! ## Reference implementations
//!
//! [`AtomicBudget`] and [`AtomicTokenBudget`] are lock-free token-bucket
//! counters built on `AtomicU64::compare_exchange_weak`. They are safe
//! for high-contention dispatch loops and refill on demand via
//! [`AtomicBudget::refill`].
//!
//! ```no_run
//! use rig_compose::budget::{AtomicBudget, BudgetGuard};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let budget = AtomicBudget::new(1_000);
//! if budget.try_reserve(250).await? {
//!     // dispatch happens here ...
//!     budget.release(250).await;
//! }
//! # Ok(()) }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use tracing::{instrument, trace};

/// Errors a budget implementation may surface.
///
/// Soft denial (the budget would be exceeded) is signalled by
/// `Ok(false)` from [`BudgetGuard::try_reserve`] /
/// [`TokenBudget::try_reserve_tokens`], not via this error. This enum is
/// reserved for *infrastructure* failures — a remote budget store
/// timing out, a persistence layer rejecting a write, etc.
#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    /// The backing store could not service the request.
    #[error("budget backend error: {0}")]
    Backend(String),
}

/// Symmetric reserve/release budget guard.
///
/// Implementations gate work on a finite resource pool (rows parsed,
/// dispatch slots, etc.). Returning `Ok(false)` from
/// [`BudgetGuard::try_reserve`] is a soft denial — the caller should
/// back off rather than treat it as an error.
#[async_trait]
pub trait BudgetGuard: Send + Sync {
    /// Try to reserve `cost` units of budget for an upcoming operation.
    /// Returns `Ok(true)` on success, `Ok(false)` on soft denial.
    async fn try_reserve(&self, cost: u64) -> Result<bool, BudgetError>;

    /// Release previously reserved units back to the pool.
    async fn release(&self, cost: u64);
}

/// Refund channel invoked when a [`TokenReservation`] is dropped
/// without being passed to [`TokenBudget::record_usage`].
///
/// Implementations call this once with the reservation's estimate to
/// return the optimistic deduction to the underlying pool.
pub type TokenRefund = Box<dyn FnOnce(u64) + Send + Sync>;

/// Reservation handle returned by [`TokenBudget::try_reserve_tokens`].
///
/// The handle carries the estimate for one model call so reconciliation
/// is per-call even when multiple prompts are outstanding concurrently.
///
/// # Cancellation semantics
///
/// Dropping a `TokenReservation` without calling
/// [`TokenBudget::record_usage`] is treated as cancellation: the full
/// estimate is refunded to the budget via the closure supplied at
/// construction. Implementations of [`TokenBudget::record_usage`] must
/// call [`TokenReservation::disarm`] to suppress the refund-on-drop
/// before performing their own reconciliation.
pub struct TokenReservation {
    estimate: u64,
    refund: Option<TokenRefund>,
}

impl std::fmt::Debug for TokenReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenReservation")
            .field("estimate", &self.estimate)
            .field("armed", &self.refund.is_some())
            .finish()
    }
}

impl TokenReservation {
    /// Construct a reservation. Intended for [`TokenBudget`]
    /// implementations only.
    pub fn new(estimate: u64, refund: TokenRefund) -> Self {
        Self {
            estimate,
            refund: Some(refund),
        }
    }

    /// Estimated tokens reserved for this call.
    pub fn estimate(&self) -> u64 {
        self.estimate
    }

    /// Take ownership of the refund closure, suppressing the
    /// refund-on-drop. Returns `None` if the reservation has already
    /// been disarmed (which would indicate misuse).
    ///
    /// [`TokenBudget::record_usage`] implementations must disarm the
    /// reservation they receive before reconciling against actual
    /// usage; otherwise the refund-on-drop would double-credit the
    /// pool.
    pub fn disarm(&mut self) -> Option<TokenRefund> {
        self.refund.take()
    }
}

impl Drop for TokenReservation {
    fn drop(&mut self) {
        if let Some(refund) = self.refund.take() {
            refund(self.estimate);
        }
    }
}

/// Soft-cap on cumulative LLM token spend.
///
/// `BudgetGuard` constrains *unit-cost work* (rows, dispatches);
/// `TokenBudget` constrains *tokens burned by a model call*. A single
/// prompt is typically a handful of tokens; a multi-round tool-call
/// loop can be 10–100× larger.
///
/// Implementations are expected to be cheap and lock-free in the hot
/// path. [`TokenBudget::try_reserve_tokens`] is called *before* a
/// prompt is sent; [`TokenBudget::record_usage`] is called *after* with
/// the observed totals from the provider so the next reservation
/// reflects reality.
#[async_trait]
pub trait TokenBudget: Send + Sync {
    /// Reserve `est` prompt+completion tokens optimistically.
    ///
    /// Returns `Ok(Some(reservation))` on success and `Ok(None)` on soft
    /// denial.
    async fn try_reserve_tokens(&self, est: u64) -> Result<Option<TokenReservation>, BudgetError>;

    /// Record the actual prompt + completion token usage from a
    /// finished call. The implementation reconciles the supplied
    /// reservation against the observed usage.
    async fn record_usage(&self, reservation: TokenReservation, prompt: u64, completion: u64);

    /// Total tokens consumed since construction (prompt + completion).
    async fn tokens_consumed(&self) -> u64;
}

/// Atomic, lock-free token-bucket budget.
///
/// Reference implementation of [`BudgetGuard`]. Backed by a single
/// `AtomicU64` counter and CAS retries.
#[derive(Debug)]
pub struct AtomicBudget {
    capacity: u64,
    available: AtomicU64,
}

impl AtomicBudget {
    /// Create a budget with the given capacity, initially full.
    pub fn new(capacity: u64) -> Self {
        Self {
            capacity,
            available: AtomicU64::new(capacity),
        }
    }

    /// Total capacity at construction time.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Currently available units.
    pub fn available(&self) -> u64 {
        self.available.load(Ordering::Acquire)
    }

    /// Utilization in `[0, 1]`. Useful for telemetry gauges.
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        let used = self.capacity.saturating_sub(self.available());
        used as f64 / self.capacity as f64
    }

    /// Restore the budget to capacity. Typically called at the start
    /// of each scheduling epoch.
    pub fn refill(&self) {
        self.available.store(self.capacity, Ordering::Release);
    }
}

#[async_trait]
impl BudgetGuard for AtomicBudget {
    #[instrument(name = "rig_compose.budget.try_reserve", skip(self), fields(cost))]
    async fn try_reserve(&self, cost: u64) -> Result<bool, BudgetError> {
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            if current < cost {
                trace!(current, "budget would be exceeded");
                return Ok(false);
            }
            match self.available.compare_exchange_weak(
                current,
                current - cost,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(true),
                Err(observed) => current = observed,
            }
        }
    }

    #[instrument(name = "rig_compose.budget.release", skip(self), fields(cost))]
    async fn release(&self, cost: u64) {
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(cost).min(self.capacity);
            match self.available.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Atomic counterpart of [`AtomicBudget`] that tracks LLM token spend.
///
/// `try_reserve_tokens` deducts an estimate up front so concurrent
/// prompts can't all over-commit. `record_usage` reconciles one
/// [`TokenReservation`] against the provider's reported totals — if the
/// actual usage was smaller than that estimate, the difference is
/// returned to the pool; if larger, the overage is debited from future
/// reservations.
///
/// Reservations also refund their estimate on drop, so an error path
/// that abandons the handle without calling `record_usage` does not
/// permanently leak tokens.
#[derive(Debug)]
pub struct AtomicTokenBudget {
    inner: Arc<AtomicTokenBudgetInner>,
}

#[derive(Debug)]
struct AtomicTokenBudgetInner {
    capacity: u64,
    available: AtomicU64,
    consumed: AtomicU64,
}

impl AtomicTokenBudgetInner {
    fn refund(&self, amount: u64) {
        if amount == 0 {
            return;
        }
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            let next = current.saturating_add(amount).min(self.capacity);
            match self.available.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn debit(&self, amount: u64) {
        if amount == 0 {
            return;
        }
        let mut current = self.available.load(Ordering::Acquire);
        loop {
            let next = current.saturating_sub(amount);
            match self.available.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

impl AtomicTokenBudget {
    /// Create a token budget with the given capacity, initially full.
    pub fn new(capacity: u64) -> Self {
        Self {
            inner: Arc::new(AtomicTokenBudgetInner {
                capacity,
                available: AtomicU64::new(capacity),
                consumed: AtomicU64::new(0),
            }),
        }
    }

    /// Total capacity at construction time.
    pub fn capacity(&self) -> u64 {
        self.inner.capacity
    }

    /// Currently available tokens.
    pub fn available(&self) -> u64 {
        self.inner.available.load(Ordering::Acquire)
    }
}

#[async_trait]
impl TokenBudget for AtomicTokenBudget {
    #[instrument(name = "rig_compose.token_budget.try_reserve", skip(self), fields(est))]
    async fn try_reserve_tokens(&self, est: u64) -> Result<Option<TokenReservation>, BudgetError> {
        let mut current = self.inner.available.load(Ordering::Acquire);
        loop {
            if current < est {
                trace!(current, "token budget would be exceeded");
                return Ok(None);
            }
            match self.inner.available.compare_exchange_weak(
                current,
                current - est,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let weak = Arc::downgrade(&self.inner);
                    let refund: TokenRefund = Box::new(move |amount| {
                        if let Some(inner) = weak.upgrade() {
                            inner.refund(amount);
                        }
                    });
                    return Ok(Some(TokenReservation::new(est, refund)));
                }
                Err(observed) => current = observed,
            }
        }
    }

    #[instrument(name = "rig_compose.token_budget.record_usage", skip(self))]
    async fn record_usage(&self, mut reservation: TokenReservation, prompt: u64, completion: u64) {
        let actual = prompt.saturating_add(completion);
        self.inner.consumed.fetch_add(actual, Ordering::AcqRel);
        // Disarm the refund-on-drop before reconciling, otherwise the
        // estimate would be returned twice.
        let _ = reservation.disarm();
        let estimate = reservation.estimate();
        if estimate >= actual {
            // Refund any over-reservation so concurrent callers see
            // accurate availability before the next prompt fires.
            self.inner.refund(estimate - actual);
        } else {
            // Actuals exceeded the reservation — debit the overage
            // from future reservations rather than letting the bucket
            // silently drift past capacity.
            self.inner.debit(actual - estimate);
        }
    }

    async fn tokens_consumed(&self) -> u64 {
        self.inner.consumed.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reserve_until_empty() {
        let b = AtomicBudget::new(100);
        assert!(b.try_reserve(60).await.unwrap());
        assert!(b.try_reserve(40).await.unwrap());
        assert!(!b.try_reserve(1).await.unwrap());
        b.release(50).await;
        assert!(b.try_reserve(50).await.unwrap());
    }

    #[tokio::test]
    async fn release_caps_at_capacity() {
        let b = AtomicBudget::new(10);
        assert!(b.try_reserve(5).await.unwrap());
        b.release(100).await;
        assert_eq!(b.available(), 10);
    }

    #[tokio::test]
    async fn refill_restores_capacity() {
        let b = AtomicBudget::new(100);
        assert!(b.try_reserve(75).await.unwrap());
        assert_eq!(b.available(), 25);
        b.refill();
        assert_eq!(b.available(), 100);
    }

    #[tokio::test]
    async fn utilization_tracks_consumption() {
        let b = AtomicBudget::new(100);
        assert!((b.utilization() - 0.0).abs() < f64::EPSILON);
        assert!(b.try_reserve(40).await.unwrap());
        assert!((b.utilization() - 0.4).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn token_budget_reserves_records_and_reports() {
        let tb = AtomicTokenBudget::new(1_000);
        let reservation = tb.try_reserve_tokens(400).await.unwrap().unwrap();
        tb.record_usage(reservation, 120, 80).await;
        assert_eq!(tb.tokens_consumed().await, 200);
        // Bind the second reservation so its drop-refund doesn't restore
        // the bucket before the deny-check below.
        let _hold = tb.try_reserve_tokens(800).await.unwrap().unwrap();
        assert!(tb.try_reserve_tokens(1).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn token_budget_debits_overage() {
        let tb = AtomicTokenBudget::new(1_000);
        let reservation = tb.try_reserve_tokens(100).await.unwrap().unwrap();
        tb.record_usage(reservation, 150, 50).await;
        assert_eq!(tb.tokens_consumed().await, 200);
        assert_eq!(tb.available(), 800);
    }

    #[tokio::test]
    async fn token_budget_reconciles_each_reservation_independently() {
        let tb = AtomicTokenBudget::new(1_000);
        let first = tb.try_reserve_tokens(400).await.unwrap().unwrap();
        let second = tb.try_reserve_tokens(400).await.unwrap().unwrap();
        assert_eq!(tb.available(), 200);

        tb.record_usage(first, 100, 100).await;
        assert_eq!(tb.available(), 400);
        assert!(tb.try_reserve_tokens(401).await.unwrap().is_none());

        tb.record_usage(second, 200, 200).await;
        assert_eq!(tb.available(), 400);
        assert_eq!(tb.tokens_consumed().await, 600);
    }

    #[tokio::test]
    async fn token_reservation_reports_estimate() {
        let tb = AtomicTokenBudget::new(10);
        let reservation = tb.try_reserve_tokens(7).await.unwrap().unwrap();
        assert_eq!(reservation.estimate(), 7);
    }

    #[tokio::test]
    async fn token_reservation_refunds_on_drop() {
        let tb = AtomicTokenBudget::new(1_000);
        {
            let _reservation = tb.try_reserve_tokens(400).await.unwrap().unwrap();
            assert_eq!(tb.available(), 600);
        } // dropped without record_usage
        assert_eq!(tb.available(), 1_000);
        assert_eq!(tb.tokens_consumed().await, 0);
    }

    #[tokio::test]
    async fn token_reservation_refund_is_capped_at_capacity() {
        let tb = AtomicTokenBudget::new(100);
        let r = tb.try_reserve_tokens(40).await.unwrap().unwrap();
        // Manually leak to capacity then drop — refund must not exceed cap.
        drop(r);
        assert_eq!(tb.available(), 100);
    }
}
