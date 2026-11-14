//! Backoff for failed browser saves without suppressing recovery indefinitely.
#[derive(Default)]
pub(super) struct SaveRetry {
    failed_at: Option<f64>,
    hide_attempted: bool,
}

impl SaveRetry {
    pub(super) fn should_attempt(&mut self, now: f64, hiding: bool) -> bool {
        if hiding {
            // visibilitychange and pagehide commonly arrive together. Try the
            // final snapshot once per hide cycle, not once per event.
            if self.hide_attempted {
                return false;
            }
            self.hide_attempted = true;
            return true;
        }
        self.failed_at.is_none_or(|failed| now - failed >= 30.0)
    }

    pub(super) fn failed(&mut self, now: f64) {
        self.failed_at = Some(now);
    }

    pub(super) fn changed_or_saved(&mut self) {
        self.failed_at = None;
    }

    pub(super) fn visible(&mut self) {
        self.hide_attempted = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_failed_save_retries_after_thirty_seconds() {
        let mut retry = SaveRetry::default();
        assert!(retry.should_attempt(1.0, false));
        retry.failed(1.0);
        assert!(!retry.should_attempt(1.5, false));
        assert!(!retry.should_attempt(30.99, false));
        assert!(retry.should_attempt(31.0, false));
        retry.failed(31.0);
        assert!(!retry.should_attempt(31.5, false));
    }

    #[test]
    fn changed_state_and_success_clear_backoff() {
        let mut retry = SaveRetry::default();
        retry.failed(1.0);
        retry.changed_or_saved();
        assert!(retry.should_attempt(1.5, false));
    }

    #[test]
    fn hiding_forces_one_attempt_and_visible_rearms_it() {
        let mut retry = SaveRetry::default();
        retry.failed(1.0);
        assert!(retry.should_attempt(1.1, true));
        retry.failed(1.1);
        assert!(!retry.should_attempt(1.2, true));
        assert!(!retry.should_attempt(1.3, false));
        retry.visible();
        assert!(retry.should_attempt(1.4, true));
    }
}
