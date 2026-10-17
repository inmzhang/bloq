use bloq_circuit::{CircuitError, MeasurementFrameError};

/// Sentinel for "measurement not yet emitted".
const NO_OCCURRENCE: u32 = u32::MAX;
pub(crate) const MAX_STIM_RECORD_LOOKBACK: u32 = 16_777_215;

pub(crate) fn stim_record_lookback(lookback: u32) -> Option<i32> {
    (lookback != 0 && lookback <= MAX_STIM_RECORD_LOOKBACK).then(|| -(lookback as i32))
}

/// Tracks the absolute output position of each emitted measurement.
///
/// Resolution only ever needs the *latest* occurrence of a measurement, so
/// the frame stores one `u32` per measurement id. Stability probes for
/// repeat preservation record an undo log between [`Self::checkpoint`] and
/// [`Self::rollback_to`] instead of keeping full occurrence history.
#[derive(Debug, Clone, Default)]
pub(crate) struct MeasurementFrame {
    emitted_count: u32,
    /// Latest absolute occurrence per measurement id (`NO_OCCURRENCE` = none).
    last_occurrence: Vec<u32>,
    /// `(measurement, previous occurrence)` pairs recorded while a
    /// checkpoint is active, replayed in reverse on rollback.
    undo_log: Vec<(u32, u32)>,
    checkpoint_depth: u32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MeasurementFrameCheckpoint {
    emitted_count: u32,
    undo_len: usize,
}

impl MeasurementFrame {
    /// Pre-size the per-measurement occurrence table for ids `0..count`.
    ///
    /// Recording grows the table on demand; pre-sizing it once avoids
    /// repeated reallocation when the maximum measurement id is known up
    /// front.
    pub(crate) fn reserve_measurement_ids(&mut self, count: usize) {
        if self.last_occurrence.len() < count {
            self.last_occurrence.resize(count, NO_OCCURRENCE);
        }
    }

    /// How many records have been emitted so far. Segmented emission reads
    /// this at chunk boundaries to report each chunk's measurement span;
    /// everything else tracks positions via the resolve/record methods.
    pub(crate) fn emitted_count(&self) -> u32 {
        self.emitted_count
    }

    pub(crate) fn checkpoint(&mut self) -> MeasurementFrameCheckpoint {
        self.checkpoint_depth += 1;
        MeasurementFrameCheckpoint {
            emitted_count: self.emitted_count,
            undo_len: self.undo_log.len(),
        }
    }

    pub(crate) fn rollback_to(&mut self, checkpoint: MeasurementFrameCheckpoint) {
        while self.undo_log.len() > checkpoint.undo_len {
            let (measurement, previous) = self
                .undo_log
                .pop()
                .expect("the loop guard ensures the undo log is non-empty");
            self.last_occurrence[measurement as usize] = previous;
        }
        self.checkpoint_depth -= 1;
        self.emitted_count = checkpoint.emitted_count;
    }

    pub(crate) fn record_emitted(&mut self, measurement: u32) -> Result<(), CircuitError> {
        let abs = self.emitted_count;
        self.emitted_count = self.emitted_count.checked_add(1).ok_or(
            MeasurementFrameError::EmittedMeasurementCountOverflow {
                emitted_count: self.emitted_count,
                additional: 1,
            },
        )?;
        self.set_last_occurrence(measurement, abs);
        Ok(())
    }

    pub(crate) fn record_repeated_measurements(
        &mut self,
        measurements_per_iteration: u32,
        repetitions: u32,
        measurement_offsets: impl IntoIterator<Item = (u32, u32)>,
    ) -> Result<(), CircuitError> {
        if repetitions == 0 || measurements_per_iteration == 0 {
            return Ok(());
        }
        let Some(additional) = measurements_per_iteration.checked_mul(repetitions) else {
            return Err(MeasurementFrameError::RepeatedMeasurementCountOverflow {
                measurements_per_iteration,
                repetitions,
            }
            .into());
        };
        let base = self.emitted_count;
        let Some(end) = base.checked_add(additional) else {
            return Err(MeasurementFrameError::EmittedMeasurementCountOverflow {
                emitted_count: self.emitted_count,
                additional,
            }
            .into());
        };

        // The last occurrence of each repeated measurement comes from the
        // final iteration.
        let last_iteration_base = base + measurements_per_iteration * (repetitions - 1);
        for (measurement, offset) in measurement_offsets {
            if offset >= measurements_per_iteration {
                return Err(MeasurementFrameError::RepeatMeasurementOffsetOutOfRange {
                    offset,
                    measurements_per_iteration,
                }
                .into());
            }
            self.set_last_occurrence(measurement, last_iteration_base + offset);
        }
        self.emitted_count = end;
        Ok(())
    }

    pub(crate) fn resolve_measurement(&self, measurement: u32) -> Result<i32, CircuitError> {
        let abs = self.resolve_absolute_measurement(measurement)?;
        self.lookback_to_absolute(abs)
    }

    pub(crate) fn resolve_absolute_measurement(
        &self,
        measurement: u32,
    ) -> Result<u32, CircuitError> {
        match self.last_occurrence.get(measurement as usize) {
            Some(&abs) if abs != NO_OCCURRENCE => Ok(abs),
            _ => Err(CircuitError::InvalidMeasurementId(measurement)),
        }
    }

    pub(crate) fn lookback_to_absolute(&self, abs: u32) -> Result<i32, CircuitError> {
        let lookback = self
            .emitted_count
            .checked_sub(abs)
            .filter(|&lookback| lookback != 0)
            .expect("absolute measurement occurrence precedes emitted count");
        stim_record_lookback(lookback).ok_or_else(|| {
            MeasurementFrameError::RecordLookbackOutOfRange {
                lookback,
                max: MAX_STIM_RECORD_LOOKBACK,
            }
            .into()
        })
    }

    fn set_last_occurrence(&mut self, measurement: u32, abs: u32) {
        let index = measurement as usize;
        if self.last_occurrence.len() <= index {
            self.last_occurrence.resize(index + 1, NO_OCCURRENCE);
        }
        if self.checkpoint_depth > 0 {
            self.undo_log
                .push((measurement, self.last_occurrence[index]));
        }
        self.last_occurrence[index] = abs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_measurement_spans_resolve_without_expanding_occurrences() {
        let repeated = 3;
        let mut frame = MeasurementFrame::default();

        frame
            .record_repeated_measurements(1, 1_000_000, [(repeated, 0)])
            .expect("record repeated measurements");

        assert_eq!(frame.resolve_measurement(repeated).unwrap(), -1);
        assert_eq!(
            frame.resolve_absolute_measurement(repeated).unwrap(),
            999_999
        );
    }

    #[test]
    fn rollback_removes_records_after_checkpoint() {
        let mut frame = MeasurementFrame::default();
        frame.record_emitted(0).expect("record measurement");
        let checkpoint = frame.checkpoint();

        frame.record_emitted(0).expect("record measurement");
        frame.record_emitted(3).expect("record measurement");
        frame
            .record_repeated_measurements(2, 3, [(0, 0), (1, 1)])
            .expect("record repeated measurements");

        frame.rollback_to(checkpoint);

        assert_eq!(frame.emitted_count(), 1);
        assert_eq!(frame.resolve_absolute_measurement(0).unwrap(), 0);
        frame.resolve_absolute_measurement(1).unwrap_err();
        frame.resolve_absolute_measurement(3).unwrap_err();
    }

    #[test]
    fn record_emitted_rejects_count_overflow() {
        let mut frame = MeasurementFrame {
            emitted_count: u32::MAX,
            ..Default::default()
        };

        assert_eq!(
            frame.record_emitted(0),
            Err(CircuitError::MeasurementFrame(
                MeasurementFrameError::EmittedMeasurementCountOverflow {
                    emitted_count: u32::MAX,
                    additional: 1,
                }
            ))
        );
        assert_eq!(
            frame.resolve_absolute_measurement(0),
            Err(CircuitError::InvalidMeasurementId(0))
        );
    }

    #[test]
    fn stim_record_lookback_limit_is_inclusive() {
        let mut frame = MeasurementFrame {
            emitted_count: MAX_STIM_RECORD_LOOKBACK + 1,
            ..Default::default()
        };
        frame.set_last_occurrence(0, 1);
        frame.set_last_occurrence(1, 0);

        assert_eq!(
            frame.resolve_measurement(0),
            Ok(-(MAX_STIM_RECORD_LOOKBACK as i32))
        );
        assert_eq!(
            frame.resolve_measurement(1),
            Err(CircuitError::MeasurementFrame(
                MeasurementFrameError::RecordLookbackOutOfRange {
                    lookback: MAX_STIM_RECORD_LOOKBACK + 1,
                    max: MAX_STIM_RECORD_LOOKBACK,
                }
            ))
        );
        assert_eq!(stim_record_lookback(i32::MAX as u32 + 1), None);
        assert_eq!(stim_record_lookback(u32::MAX), None);
    }
}
