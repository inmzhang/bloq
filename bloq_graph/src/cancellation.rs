//! Request-scoped cooperative cancellation for graph analysis.

use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

/// A computation was cancelled before producing its result.
/// Cancellation is neither a resource failure nor a semantic diagnosis.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[error("computation cancelled")]
pub struct ComputationCancelled;

/// Shared cancellation state for one request.
///
/// Run synchronous analysis inside [`run`](Self::run). Existing normalization
/// checkpoints observe this token, and [`crate::map_jobs`] carries it into its
/// scoped workers. Other threads and later requests keep independent state.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

thread_local! {
    static ACTIVE: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

impl CancellationToken {
    /// Creates a token for a new request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Repeated calls are harmless.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether this request has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    /// Checks this request at a safe checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`ComputationCancelled`] after [`cancel`](Self::cancel).
    pub fn check(&self) -> Result<(), ComputationCancelled> {
        if self.is_cancelled() {
            Err(ComputationCancelled)
        } else {
            Ok(())
        }
    }

    /// Runs analysis with this token, checking before entry and before returning
    /// a result. Work stops at cooperative checkpoints; threads are not killed.
    ///
    /// # Errors
    ///
    /// Returns cancellation or the computation's own error. No partial result
    /// is returned after cancellation. The preceding scope is restored on exit.
    pub fn run<T, E: From<ComputationCancelled>>(
        &self,
        f: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        self.check()?;
        self.scope(|| {
            let result = f();
            self.check()?;
            result
        })
    }

    /// Checks the active request, if any, at a safe checkpoint.
    pub(crate) fn checkpoint() -> Result<(), crate::StabilizerError> {
        match Self::current() {
            Some(token) => Ok(token.check()?),
            None => Ok(()),
        }
    }

    pub(crate) fn current() -> Option<Self> {
        ACTIVE.with(|active| active.borrow().clone())
    }

    pub(crate) fn scope<T>(&self, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<CancellationToken>);
        impl Drop for Restore {
            fn drop(&mut self) {
                ACTIVE.with(|active| {
                    active.replace(self.0.take());
                });
            }
        }
        let previous = ACTIVE.with(|active| active.replace(Some(self.clone())));
        let _restore = Restore(previous);
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_workers_observe_cancellation_without_leaking_into_the_next_request() {
        for _ in 0..2 {
            let token = CancellationToken::new();
            token.scope(|| {
                let workers =
                    crate::map_jobs(&[1, 2], std::num::NonZeroUsize::new(2).unwrap(), |_| {
                        let active = CancellationToken::current().expect("worker inherits request");
                        active.cancel();
                        active.check()
                    });
                assert!(workers.iter().all(Result::is_err));
            });
            assert!(CancellationToken::current().is_none());
        }
        CancellationToken::new()
            .run(|| Ok::<_, ComputationCancelled>(()))
            .unwrap();
    }
}
