use std::collections::BTreeMap;

use bloq_utils::Pauli;
use glam::IVec2;
use itertools::Itertools;
use rustc_hash::FxHashMap;

use bloq_circuit::{
    BodyId, CircuitError, ConditionalCorrection, CoordCircuit, DetectorParity, DetectorTerm,
    GateType, LoopCarriedDetectorState, LoopStateId, MeasurementFrameError, Op, PauliBasis,
    PauliMap, measurement_gate_name,
};
use thiserror::Error;

use crate::layout::QubitLayout;
use crate::measurement_frame::MeasurementFrame;
use crate::text_utils::{push_int, push_number_prefer_int};

/// Reason the static Stim backend could not emit a program.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StimEmissionError {
    /// The input program failed Bloq IR well-formedness validation.
    #[error("invalid Bloq IR: {0}")]
    InvalidProgram(#[from] bloq_ir::BloqValidationError),
    /// A shared detector use has an invalid bundle or owner binding.
    #[error("{0}")]
    DetectorBundle(#[from] bloq_ir::DetectorBundleError),
    /// A node annotation names a measurement outside the prepared owner registry.
    #[error("node annotation names unknown measurement {0:?}")]
    UnknownInstanceMeasurement(bloq_ir::lowering::InstanceMeasurement),
    /// Node annotations have no instance-local loop-state frame.
    #[error("node annotation references unsupported loop state {0:?}")]
    NodeLoopStateUnsupported(LoopStateId),
    /// A gate with no Stim spelling on the static (Clifford) path, e.g. a
    /// non-Clifford `T`.
    #[error("unsupported Stim backend gate: {0}")]
    UnsupportedGate(GateType),
    /// A node the static backend cannot emit (a runtime-control region or
    /// conditional activation), rejected by name before any emission work.
    #[error("unsupported Stim backend node: {0}")]
    UnsupportedNode(&'static str),
    /// Alignment is noiseless.
    #[error("moment-aligned Stim emission does not support noise")]
    AlignMomentsWithNoise,
    /// Alignment cannot preserve per-node segments.
    #[error("moment-aligned Stim emission does not support segmented output")]
    AlignMomentsWithSegments,
    /// An empty circuit has no aligned slot for metadata.
    #[error("moment-aligned Stim emission cannot place metadata for an empty quantum circuit")]
    AlignMomentsWithEmptyMetadata,
    /// Moment-alignment planning failed.
    #[error("{0}")]
    MomentAlignment(#[from] bloq_ir::MomentAlignmentError),
    /// The graph violates an invariant the emit walk relies on (a template,
    /// instance, or producer a node names is missing, or a measurement is
    /// read before it is emitted). Well-formed programs from the compiler
    /// never trigger this; it guards hand-built or unvalidated graphs.
    #[error("malformed graph: {0}")]
    MalformedGraph(&'static str),
    /// A node's template instances failed to merge into one emission circuit.
    /// Validation runs the same merge, so a validated Bloq never triggers
    /// this; it guards hand-built or unvalidated graphs.
    #[error("{0}")]
    InstanceMerge(#[from] bloq_ir::lowering::NodeTemplateInstanceMergeError),
    /// Flattening failed on a straight-line emission path.
    #[error("failed to flatten Stim emission: {0}")]
    Flatten(#[from] bloq_ir::FlattenError),
    /// The uniform physical error probability is not finite or outside `[0, 1]`.
    #[error("physical error probability must be finite and in [0, 1]")]
    InvalidNoiseProbability,
    /// A backend index cannot be represented in its unsigned 32-bit id space.
    #[error("Stim {space} index {index} exceeds the u32 index range")]
    IndexOutOfRange {
        /// Index space being populated.
        space: &'static str,
        /// Index that cannot be represented.
        index: usize,
    },
    /// A qubit target exceeds Stim's 24-bit target payload.
    #[error("Stim qubit index {index} exceeds the target limit {max}")]
    QubitIndexOutOfRange {
        /// Qubit index that cannot be represented.
        index: usize,
        /// Largest qubit index supported by the target payload.
        max: u32,
    },
    /// A template instance offset moves a circuit qubit outside the global
    /// `i32` coordinate lattice.
    #[error("{0}")]
    CoordinateOverflow(#[from] bloq_ir::CoordinateOverflowError),
    /// The circuit layer rejected an instruction while emitting a node's
    /// circuit (e.g. a measurement id outside the emitted range).
    #[error("{0}")]
    Circuit(#[from] CircuitError),
    /// A classical value has no affine record recipe on the static path.
    #[error("{0}")]
    ClassicalResolution(#[from] bloq_ir::ResolveError),
    /// The graph contains a cycle, so it has no topological emit order.
    /// Validated programs are acyclic; this guards hand-built or unvalidated
    /// graphs.
    #[error("{0}")]
    CyclicGraph(#[from] bloq_ir::CycleDetected),
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StimCircuitAnnotations {
    pub detectors: Vec<StimDetector>,
    pub repeat_states: Vec<StimRepeatState>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StimDetector {
    pub scope: StimAnnotationScope,
    pub parity: DetectorParity,
    pub coords: Option<bloq_circuit::DetectorCoords>,
    pub postselection: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StimRepeatState {
    pub body: BodyId,
    pub state: LoopStateId,
    pub initial: DetectorParity,
    pub next: DetectorParity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StimAnnotationScope {
    TopLevel,
    RepeatBody { body: BodyId },
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StimEmitOptions {
    pub enable_clifford_proxy: bool,
    pub allow_non_clifford: bool,
    pub emit_postselection_tags: bool,
    pub qubit_offset: IVec2,
}

impl Default for StimEmitOptions {
    fn default() -> Self {
        Self {
            enable_clifford_proxy: false,
            allow_non_clifford: false,
            emit_postselection_tags: true,
            qubit_offset: IVec2::ZERO,
        }
    }
}

pub(crate) struct StimEmitContext<'a> {
    pub layout: &'a QubitLayout,
    pub frame: &'a mut MeasurementFrame,
    pub resolved_loop_detector_states: &'a mut ResolvedLoopStates,
    pub output: &'a mut String,
    pub indent: usize,
}

/// Finished loop states mapped to their resolved absolute measurement parity.
pub(crate) type ResolvedLoopStates = FxHashMap<LoopStateId, Vec<u32>>;

/// One active `REPEAT` block's resolved detector states, keyed by loop state
/// id. Ordered by id so finish-time scoping can walk the frame and the loop's
/// state list together in ascending id order.
type LoopStateFrame = BTreeMap<LoopStateId, Vec<u32>>;

/// Append ` rec[<lookback>]`, a measurement-record target.
fn push_lookback(output: &mut String, lookback: i32) {
    output.push_str(" rec[");
    push_int(output, lookback);
    output.push(']');
}

pub(crate) trait StimMeasurementMapper {
    /// Return the same mapped id for repeated calls with the same measurement.
    ///
    /// Stim repeat preservation probes future iterations before committing to
    /// the output, so mapping may allocate lazily but must be idempotent.
    fn map_measurement(&mut self, measurement: u32) -> Result<u32, CircuitError>;

    fn map_annotation_measurement(&mut self, measurement: u32) -> Result<u32, CircuitError> {
        self.map_measurement(measurement)
    }
}

/// Passes measurement ids through unchanged. Used by the standalone
/// coordinate-circuit paths (the `verify` feature and tests); the program
/// emitter supplies its own instance-aware mappers instead.
#[cfg(any(test, feature = "verify"))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IdentityMeasurementMapper;

#[cfg(any(test, feature = "verify"))]
impl StimMeasurementMapper for IdentityMeasurementMapper {
    fn map_measurement(&mut self, measurement: u32) -> Result<u32, CircuitError> {
        Ok(measurement)
    }
}

/// Emits `circuit` on its own — fresh layout, fresh measurement frame, ids
/// passed through unchanged. Shared by the `verify` feature and the emitter's
/// own tests; the program emitter drives [`emit_stim_circuit`] directly with an
/// instance-aware mapper instead.
#[cfg(any(test, feature = "verify"))]
pub(crate) fn emit_standalone_stim(
    circuit: &CoordCircuit,
    annotations: Option<&StimCircuitAnnotations>,
    layout: &QubitLayout,
    enable_clifford_proxy: bool,
) -> Result<(String, MeasurementFrame), CircuitError> {
    let mut output = String::new();
    emit_qubit_coords_lines(
        &mut output,
        layout
            .coord_index()
            .iter()
            .map(|(coord, qubit)| (*coord, *qubit))
            .sorted_by_key(|&(_, qubit)| qubit),
    );
    let mut frame = MeasurementFrame::default();
    let mut resolved_loop_detector_states = ResolvedLoopStates::default();
    let mut ctx = StimEmitContext {
        layout,
        frame: &mut frame,
        resolved_loop_detector_states: &mut resolved_loop_detector_states,
        output: &mut output,
        indent: 0,
    };
    emit_stim_circuit(
        circuit,
        annotations,
        &mut ctx,
        &mut IdentityMeasurementMapper,
        StimEmitOptions {
            enable_clifford_proxy,
            ..StimEmitOptions::default()
        },
    )?;
    Ok((output, frame))
}

pub(crate) fn emit_stim_circuit<M: StimMeasurementMapper>(
    circuit: &CoordCircuit,
    annotations: Option<&StimCircuitAnnotations>,
    ctx: &mut StimEmitContext<'_>,
    mapper: &mut M,
    options: StimEmitOptions,
) -> Result<bool, CircuitError> {
    let base_indent = ctx.indent;
    let mut emitter = StimTextEmitter {
        source: circuit,
        annotations,
        ctx,
        mapper,
        options,
        detector_state_stack: Vec::new(),
        base_indent,
        needs_node_boundary_tick: false,
    };
    emitter.emit_body(circuit.entry_body())?;
    emitter.emit_side_table_annotations(circuit.entry_body())?;
    Ok(emitter.needs_node_boundary_tick)
}

struct StimTextEmitter<'circuit, 'ctx, 'data, 'mapper, M>
where
    M: StimMeasurementMapper,
{
    source: &'circuit CoordCircuit,
    annotations: Option<&'circuit StimCircuitAnnotations>,
    ctx: &'ctx mut StimEmitContext<'data>,
    mapper: &'mapper mut M,
    options: StimEmitOptions,
    detector_state_stack: Vec<LoopStateFrame>,
    /// The indent level [`emit_stim_circuit`] started at.
    base_indent: usize,
    needs_node_boundary_tick: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StimTarget {
    Rec(i32),
    Pauli { pauli: Pauli, qubit: u32 },
}

impl<M> StimTextEmitter<'_, '_, '_, '_, M>
where
    M: StimMeasurementMapper,
{
    fn note_instruction(&mut self, needs_node_boundary_tick: bool) {
        if self.ctx.indent == self.base_indent {
            self.needs_node_boundary_tick = needs_node_boundary_tick;
        }
    }

    fn emit_body(&mut self, body: BodyId) -> Result<(), CircuitError> {
        let Some(source) = self.source.body(body) else {
            return Err(CircuitError::InvalidCircuitBody(body));
        };
        for op in source.ops() {
            self.emit_op(op)?;
        }
        Ok(())
    }

    fn emit_side_table_annotations(&mut self, body: BodyId) -> Result<(), CircuitError> {
        let Some(annotations) = self.annotations else {
            return Ok(());
        };
        for detector in &annotations.detectors {
            if detector.scope == scope_for_body(self.source, body) {
                self.emit_detector_annotation(detector)?;
            }
        }
        Ok(())
    }

    fn emit_op(&mut self, op: &Op) -> Result<(), CircuitError> {
        match op {
            Op::Gate { gate, qubits } => self.emit_gate(*gate, qubits),
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability,
            } => {
                let records = self.map_measurements(measurements)?;
                self.emit_measure(*basis, qubits, &records, *flip_probability)
            }
            Op::Tick => {
                self.note_instruction(false);
                self.write_line("TICK");
                Ok(())
            }
            Op::MPP {
                products,
                measurements,
            } => {
                let records = self.map_measurements(measurements)?;
                self.emit_mpp(products, &records)
            }
            Op::Repeat { body, repetitions } => self.emit_repeat(*body, *repetitions),
            Op::ConditionalPauli(corrections) => self.emit_conditional_paulis(corrections),
            Op::Depolarize1 {
                probability,
                qubits,
            } => self.emit_noise("DEPOLARIZE1", *probability, qubits),
            Op::Depolarize2 {
                probability,
                qubits,
            } => self.emit_noise("DEPOLARIZE2", *probability, qubits),
            Op::PauliError {
                probability,
                pauli,
                qubits,
            } => self.emit_noise(pauli_error_gate_name(*pauli), *probability, qubits),
        }
    }

    fn emit_conditional_paulis(
        &mut self,
        corrections: &[ConditionalCorrection],
    ) -> Result<(), CircuitError> {
        for correction in corrections {
            let measurement = self.mapper.map_measurement(correction.control)?;
            let lookback = self.ctx.frame.resolve_measurement(measurement)?;
            let coord = correction.target + self.options.qubit_offset;
            let Some(qubit) = self.ctx.layout.get(coord) else {
                return Err(CircuitError::QubitNotFoundInLayout(coord));
            };

            self.write_instruction_prefix(conditional_pauli_gate_name(correction.pauli), None);
            push_lookback(self.ctx.output, lookback);
            self.ctx.output.push(' ');
            self.ctx.output.push_str(self.ctx.layout.label(qubit));
            self.ctx.output.push('\n');
        }
        Ok(())
    }

    /// Map a body op's raw measurement ids to their backend ids, in order.
    fn map_measurements(&mut self, measurements: &[u32]) -> Result<Vec<u32>, CircuitError> {
        measurements
            .iter()
            .map(|&measurement| self.mapper.map_measurement(measurement))
            .collect()
    }

    /// Emit one `MPP X0*X1 Z2*Z3` line: each product's non-identity Pauli targets
    /// are joined by `*` combiners, products separated by a space. Each product
    /// produced one measurement record, recorded into the frame like a measurement.
    fn emit_mpp(&mut self, products: &[PauliMap], records: &[u32]) -> Result<(), CircuitError> {
        self.write_instruction_prefix("MPP", None);
        for product in products {
            let mut first = true;
            for (coord, pauli) in product
                .iter()
                .filter(|&(_, &pauli)| pauli != Pauli::I)
                .sorted_by_key(|(coord, _)| {
                    let coord = **coord + self.options.qubit_offset;
                    self.ctx.layout.get(coord).unwrap_or(0)
                })
            {
                let coord = *coord + self.options.qubit_offset;
                let Some(qubit) = self.ctx.layout.get(coord) else {
                    return Err(CircuitError::QubitNotFoundInLayout(coord));
                };
                self.ctx.output.push(if first { ' ' } else { '*' });
                first = false;
                self.ctx.output.push(match pauli {
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                    Pauli::I => unreachable!("identity Paulis are filtered out above"),
                });
                self.ctx.output.push_str(self.ctx.layout.label(qubit));
            }
        }
        self.ctx.output.push('\n');
        for &record in records {
            self.ctx.frame.record_emitted(record)?;
        }
        Ok(())
    }

    fn emit_repeat(&mut self, body: BodyId, repetitions: u32) -> Result<(), CircuitError> {
        if repetitions == 0 {
            return Ok(());
        }

        let side_table_detector_states = self.repeat_detector_states(body);
        let detector_states = side_table_detector_states.as_slice();
        let stack_depth = self.detector_state_stack.len();
        let initial_states = self.resolve_initial_detector_states(detector_states)?;
        self.detector_state_stack.push(initial_states);

        // A non-jumpable loop (see `can_jump_loop_detector_states`) never
        // probes for a stable body, so it unrolls all `repetitions` iterations
        // — required for correctness, not a missed optimization.
        let can_jump = can_jump_loop_detector_states(detector_states);
        let mut stable_body = None;
        let mut remaining = repetitions;
        while remaining > 1 {
            if can_jump
                && let Some(rendered) = self.stable_repeat_body_render(body, detector_states)?
            {
                stable_body = Some(rendered);
                break;
            }
            self.emit_repeat_iteration(body, detector_states)?;
            remaining -= 1;
        }

        if remaining == 1 {
            self.emit_repeat_iteration(body, detector_states)?;
            self.finish_loop_detector_states(stack_depth, detector_states)?;
            return Ok(());
        }

        // The while loop above only leaves `remaining > 1` via the stable-body
        // break, so a summarized body is always available here.
        let rendered_body =
            stable_body.expect("remaining > 1 only survives the loop via the stable-body break");
        self.note_instruction(false);
        self.ctx.output.push_str(indent_prefix(self.ctx.indent));
        self.ctx.output.push_str("REPEAT ");
        push_int(self.ctx.output, remaining);
        self.ctx.output.push_str(" {\n");
        self.ctx.output.push_str(&rendered_body);
        self.apply_rendered_repeat_body_effects(body, remaining - 1, detector_states)?;
        self.write_line("}");
        self.finish_loop_detector_states(stack_depth, detector_states)?;
        Ok(())
    }

    fn stable_repeat_body_render(
        &mut self,
        body: BodyId,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<Option<String>, CircuitError> {
        let checkpoint = self.ctx.frame.checkpoint();
        // Render the body twice from the same starting state; the loop is
        // stable once two consecutive iterations produce identical text. The
        // frame and loop-state stack are moved out so the sub-emitter can
        // borrow them, then rolled back and restored afterwards.
        let mut state_stack = self.detector_state_stack.clone();
        let mut frame = std::mem::take(self.ctx.frame);
        let mut first = String::new();
        let mut second = String::new();
        let result = self
            .render_repeat_iteration_for_stability(
                body,
                detector_states,
                &mut frame,
                &mut state_stack,
                &mut first,
            )
            .and_then(|()| {
                self.render_repeat_iteration_for_stability(
                    body,
                    detector_states,
                    &mut frame,
                    &mut state_stack,
                    &mut second,
                )
            })
            .map(|()| (first == second).then_some(first));
        frame.rollback_to(checkpoint);
        *self.ctx.frame = frame;
        result
    }

    fn apply_rendered_repeat_body_effects(
        &mut self,
        body: BodyId,
        tail_repetitions: u32,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<(), CircuitError> {
        let (offsets, body_measurements) =
            repeat_body_measurement_offsets(self.source, body, self.mapper)?;
        self.record_repeat_body_measurements(body_measurements, 1, offsets.iter().copied())?;
        self.update_loop_detector_states(detector_states)?;
        self.record_repeat_body_measurements(body_measurements, tail_repetitions, offsets)?;
        self.update_loop_detector_states(detector_states)
    }

    fn record_repeat_body_measurements(
        &mut self,
        body_measurements: u32,
        repetitions: u32,
        offsets: impl IntoIterator<Item = (u32, u32)>,
    ) -> Result<(), CircuitError> {
        if body_measurements == 0 || repetitions == 0 {
            return Ok(());
        }
        self.ctx
            .frame
            .record_repeated_measurements(body_measurements, repetitions, offsets)
    }

    fn render_repeat_iteration_for_stability(
        &mut self,
        body: BodyId,
        detector_states: &[LoopCarriedDetectorState],
        frame: &mut MeasurementFrame,
        detector_state_stack: &mut Vec<LoopStateFrame>,
        output: &mut String,
    ) -> Result<(), CircuitError> {
        debug_assert!(!detector_state_stack.is_empty());
        output.clear();
        let mut ctx = StimEmitContext {
            layout: self.ctx.layout,
            frame,
            resolved_loop_detector_states: self.ctx.resolved_loop_detector_states,
            output,
            indent: self.ctx.indent + 1,
        };
        // The probe shares the loop-state stack (moved in and back out) but
        // renders with its own throwaway working buffers (including its own
        // last-instruction tracking, discarded with the probe).
        let mut emitter = StimTextEmitter {
            source: self.source,
            annotations: self.annotations,
            ctx: &mut ctx,
            mapper: self.mapper,
            options: self.options,
            detector_state_stack: std::mem::take(detector_state_stack),
            base_indent: self.base_indent,
            needs_node_boundary_tick: false,
        };
        let result = emitter.emit_repeat_iteration(body, detector_states);
        *detector_state_stack = std::mem::take(&mut emitter.detector_state_stack);
        result
    }

    fn emit_repeat_iteration(
        &mut self,
        body: BodyId,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<(), CircuitError> {
        self.emit_body(body)?;
        self.emit_side_table_annotations(body)?;
        self.update_loop_detector_states(detector_states)
    }

    fn emit_detector_annotation(&mut self, detector: &StimDetector) -> Result<(), CircuitError> {
        let lookbacks = self.resolve_detector_lookbacks(&detector.parity)?;
        let targets: Vec<StimTarget> = lookbacks.into_iter().map(StimTarget::Rec).collect();
        self.write_targets_line(
            if detector.postselection && self.options.emit_postselection_tags {
                "DETECTOR[POST-SELECTION]"
            } else {
                "DETECTOR"
            },
            detector.coords.as_deref(),
            &targets,
        );
        Ok(())
    }

    fn repeat_detector_states(&self, body: BodyId) -> Vec<LoopCarriedDetectorState> {
        let Some(annotations) = self.annotations else {
            return Vec::new();
        };
        annotations
            .repeat_states
            .iter()
            .filter(|state| state.body == body)
            .map(|state| LoopCarriedDetectorState {
                state: state.state,
                initial: state.initial.clone(),
                next: state.next.clone(),
            })
            .collect()
    }

    fn emit_gate(&mut self, gate: GateType, qubits: &[IVec2]) -> Result<(), CircuitError> {
        let gate = if self.options.enable_clifford_proxy {
            gate.clifford_proxy()
        } else {
            gate
        };
        debug_assert!(
            self.options.allow_non_clifford || !gate.is_non_clifford(),
            "unsupported Stim gate should be preflighted"
        );
        self.write_qubit_instruction_line(gate.into(), None, qubits)?;
        Ok(())
    }

    fn emit_measure(
        &mut self,
        basis: PauliBasis,
        qubits: &[IVec2],
        records: &[u32],
        flip_probability: f64,
    ) -> Result<(), CircuitError> {
        let args = (flip_probability != 0.0).then_some(std::slice::from_ref(&flip_probability));
        self.write_qubit_instruction_line(measurement_gate_name(basis), args, qubits)?;
        for &record in records {
            self.ctx.frame.record_emitted(record)?;
        }
        Ok(())
    }

    fn emit_noise(
        &mut self,
        name: &str,
        probability: f64,
        qubits: &[IVec2],
    ) -> Result<(), CircuitError> {
        self.write_qubit_instruction_line(name, Some(std::slice::from_ref(&probability)), qubits)?;
        Ok(())
    }

    fn resolve_initial_detector_states(
        &mut self,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<LoopStateFrame, CircuitError> {
        let mut resolved = LoopStateFrame::default();
        for state in detector_states {
            let parity = self.resolve_parity_absolute(&state.initial)?;
            resolved.insert(state.state, parity);
        }
        Ok(resolved)
    }

    fn update_loop_detector_states(
        &mut self,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<(), CircuitError> {
        if detector_states.is_empty() {
            return Ok(());
        }
        // Identity-only updates leave every carried state unchanged, so the
        // current frame already holds the right parities.
        if detector_states.iter().all(is_identity_update) {
            debug_assert!(detector_states.iter().all(|state| {
                self.detector_state_stack
                    .last()
                    .is_some_and(|states| states.contains_key(&state.state))
            }));
            return Ok(());
        }
        let mut next = LoopStateFrame::new();
        for state in detector_states {
            let parity = if is_identity_update(state) {
                let Some(current) = self
                    .detector_state_stack
                    .last()
                    .and_then(|states| states.get(&state.state))
                else {
                    return Err(CircuitError::InvalidLoopDetectorState(state.state));
                };
                current.clone()
            } else {
                self.resolve_parity_absolute(&state.next)?
            };
            next.insert(state.state, parity);
        }
        let Some(current) = self.detector_state_stack.last_mut() else {
            return Err(CircuitError::InvalidLoopDetectorState(
                detector_states[0].state,
            ));
        };
        *current = next;
        Ok(())
    }

    fn finish_loop_detector_states(
        &mut self,
        stack_depth: usize,
        detector_states: &[LoopCarriedDetectorState],
    ) -> Result<(), CircuitError> {
        if detector_states.is_empty() {
            debug_assert_eq!(self.detector_state_stack.len(), stack_depth + 1);
            self.detector_state_stack.pop();
            return Ok(());
        }
        let Some(final_states) = self.detector_state_stack.pop() else {
            return Err(CircuitError::InvalidLoopDetectorState(
                detector_states[0].state,
            ));
        };
        debug_assert_eq!(self.detector_state_stack.len(), stack_depth);
        // Only this loop's listed states survive it; anything else in the
        // frame (states a nested repeat handed back) is dropped here, like a
        // lexical scope ending. Both the frame and `detector_states` are in
        // ascending id order, so a two-pointer pass moves the listed entries.
        debug_assert!(
            detector_states
                .windows(2)
                .all(|pair| pair[0].state < pair[1].state),
            "loop detector states are allocated in ascending id order"
        );
        let mut listed = detector_states.iter().map(|state| state.state).peekable();
        let mut keep = |state: LoopStateId| {
            while listed.next_if(|&id| id < state).is_some() {}
            listed.next_if_eq(&state).is_some()
        };
        if let Some(parent) = self.detector_state_stack.last_mut() {
            for (state, value) in final_states {
                if keep(state) {
                    parent.insert(state, value);
                }
            }
        } else {
            for (state, value) in final_states {
                if keep(state) {
                    self.ctx.resolved_loop_detector_states.insert(state, value);
                }
            }
        }
        Ok(())
    }

    fn resolve_parity_absolute(
        &mut self,
        parity: &DetectorParity,
    ) -> Result<Vec<u32>, CircuitError> {
        let mut absolutes = Vec::with_capacity(parity.terms().len());
        for &term in parity.terms() {
            match term {
                DetectorTerm::Measurement(measurement) => {
                    let measurement = self.mapper.map_annotation_measurement(measurement)?;
                    absolutes.push(self.ctx.frame.resolve_absolute_measurement(measurement)?);
                }
                DetectorTerm::LoopState(state) => {
                    let Some(resolved) = self
                        .detector_state_stack
                        .iter()
                        .rev()
                        .find_map(|states| states.get(&state))
                        .or_else(|| self.ctx.resolved_loop_detector_states.get(&state))
                    else {
                        return Err(CircuitError::InvalidLoopDetectorState(state));
                    };
                    absolutes.extend(resolved.iter().copied());
                }
            }
        }
        canonicalize_parity(&mut absolutes);
        Ok(absolutes)
    }

    fn resolve_detector_lookbacks(
        &mut self,
        parity: &DetectorParity,
    ) -> Result<Vec<i32>, CircuitError> {
        // Fast path: a pure measurement parity resolves each term to a lookback
        // directly, skipping the absolute-id round trip. Duplicate lookbacks
        // still XOR-cancel like the general path. Today no production input
        // produces one here — `DetectorParity` canonicalizes at construction
        // and re-canonicalizes after id remaps, and distinct backend ids
        // resolve to distinct records — so this guards a future mapper whose
        // `map_annotation_measurement` merges ids at emit time.
        if parity
            .terms()
            .iter()
            .all(|term| matches!(term, DetectorTerm::Measurement(_)))
        {
            let mut lookbacks = Vec::with_capacity(parity.terms().len());
            for &term in parity.terms() {
                let DetectorTerm::Measurement(measurement) = term else {
                    unreachable!("the all-Measurement guard admits no LoopState terms")
                };
                let measurement = self.mapper.map_annotation_measurement(measurement)?;
                lookbacks.push(self.ctx.frame.resolve_measurement(measurement)?);
            }
            lookbacks.sort_unstable();
            canonicalize_parity(&mut lookbacks);
            return Ok(lookbacks);
        }

        // General path: resolve terms to absolute ids, cancel duplicates, then
        // convert back to lookbacks.
        let absolutes = self.resolve_parity_absolute(parity)?;
        let mut lookbacks = absolutes
            .iter()
            .map(|&abs| self.ctx.frame.lookback_to_absolute(abs))
            .collect::<Result<Vec<_>, CircuitError>>()?;
        lookbacks.sort_unstable();
        Ok(lookbacks)
    }

    fn write_line(&mut self, line: &str) {
        self.ctx.output.push_str(indent_prefix(self.ctx.indent));
        self.ctx.output.push_str(line);
        self.ctx.output.push('\n');
    }

    fn write_qubit_instruction_line(
        &mut self,
        gate: &str,
        args: Option<&[f64]>,
        qubits: &[IVec2],
    ) -> Result<(), CircuitError> {
        self.write_instruction_prefix(gate, args);
        for &coord in qubits {
            let coord = coord + self.options.qubit_offset;
            let Some(qubit) = self.ctx.layout.get(coord) else {
                return Err(CircuitError::QubitNotFoundInLayout(coord));
            };
            self.ctx.output.push(' ');
            self.ctx.output.push_str(self.ctx.layout.label(qubit));
        }
        self.ctx.output.push('\n');
        Ok(())
    }

    fn write_targets_line(&mut self, gate: &str, args: Option<&[f64]>, targets: &[StimTarget]) {
        self.write_instruction_prefix(gate, args);
        for target in targets {
            target.write_to(self.ctx.output);
        }
        self.ctx.output.push('\n');
    }

    fn write_instruction_prefix(&mut self, gate: &str, args: Option<&[f64]>) {
        self.note_instruction(true);
        self.ctx.output.push_str(indent_prefix(self.ctx.indent));
        self.ctx.output.push_str(gate);
        if let Some(args) = args
            && !args.is_empty()
        {
            self.ctx.output.push('(');
            for (index, arg) in args.iter().enumerate() {
                if index > 0 {
                    self.ctx.output.push_str(", ");
                }
                push_number_prefer_int(self.ctx.output, *arg);
            }
            self.ctx.output.push(')');
        }
    }
}

impl StimTarget {
    /// Append this target with a leading space.
    fn write_to(self, output: &mut String) {
        match self {
            StimTarget::Rec(lookback) => push_lookback(output, lookback),
            StimTarget::Pauli { pauli, qubit } => {
                output.push(' ');
                output.push(match pauli {
                    Pauli::X => 'X',
                    Pauli::Y => 'Y',
                    Pauli::Z => 'Z',
                    // Identity is filtered before a `StimTarget::Pauli` is built
                    // (`append_pauli_map_stim_targets`), matching `emit_mpp`.
                    Pauli::I => {
                        unreachable!("identity Paulis are filtered before StimTarget::Pauli")
                    }
                });
                push_int(output, qubit);
            }
        }
    }
}

pub(crate) fn conditional_pauli_gate_name(pauli: PauliBasis) -> &'static str {
    match pauli {
        PauliBasis::X => "CX",
        PauliBasis::Y => "CY",
        PauliBasis::Z => "CZ",
    }
}

pub(crate) fn pauli_error_gate_name(pauli: PauliBasis) -> &'static str {
    match pauli {
        PauliBasis::X => "X_ERROR",
        PauliBasis::Y => "Y_ERROR",
        PauliBasis::Z => "Z_ERROR",
    }
}

/// Emit one `DETECTOR` line from pre-resolved measurement-record lookbacks,
/// appended directly to `output`. The verify path's counterpart to the
/// program emitter's annotation lines, sharing the `rec[...]` target
/// formatting so the two cannot drift.
#[cfg(feature = "verify")]
pub(crate) fn emit_detector_records(output: &mut String, lookbacks: &[i32]) {
    output.push_str("DETECTOR");
    for &lookback in lookbacks {
        push_lookback(output, lookback);
    }
    output.push('\n');
}

/// Append `QUBIT_COORDS(x, y) index` preamble lines for `coords`, already
/// ordered by qubit index. Both the program emitter and the standalone circuit
/// emitter route through this so the coordinate spelling stays byte-identical.
/// Coordinates are integer grid positions, so they format as `i32` directly
/// instead of through the float path. Ordering is the caller's job: a layout
/// map has no inherent order and must be sorted before it reaches here.
pub(crate) fn emit_qubit_coords_lines(
    output: &mut String,
    coords: impl IntoIterator<Item = (IVec2, u32)>,
) {
    for (coord, qubit_index) in coords {
        output.push_str("QUBIT_COORDS(");
        push_int(output, coord.x);
        output.push_str(", ");
        push_int(output, coord.y);
        output.push_str(") ");
        push_int(output, qubit_index);
        output.push('\n');
    }
}

/// Emit one `OBSERVABLE_INCLUDE(index)` line from measurement-record lookbacks
/// the resolved measurement contribution of an [`Observable`](bloq_ir::ClassicalNode::Observable).
/// Lookbacks are pre-resolved against the global frame (they resolve
/// by absolute position, so the line is correct wherever the `Observable` node
/// lands in emit order, as long as it follows its measurements). An empty set
/// carries nothing, so no line is written.
pub(crate) fn emit_observable_include_records(output: &mut String, index: u32, lookbacks: &[i32]) {
    if lookbacks.is_empty() {
        return;
    }
    output.push_str("OBSERVABLE_INCLUDE(");
    push_int(output, index);
    output.push(')');
    for &lookback in lookbacks {
        StimTarget::Rec(lookback).write_to(output);
    }
    output.push('\n');
}

/// Emit one `OBSERVABLE_INCLUDE(index)` line carrying Pauli targets only — no
/// measurement records — appended directly to `output`.
///
/// Unlike record-only standalone observable nodes, a boundary logical operator's
/// Pauli targets read the qubit state *at the point of emission*, so the backend
/// positions these lines at temporal seams and open boundaries. An operator with
/// no non-identity support (e.g. a seam XOR that cancelled to empty) writes
/// nothing.
pub(crate) fn emit_observable_include_pauli_targets(
    output: &mut String,
    index: u32,
    operator: &PauliMap,
    layout: &QubitLayout,
) -> Result<(), CircuitError> {
    let mut targets = Vec::new();
    append_pauli_map_stim_targets(operator, layout, &mut targets)?;
    if targets.is_empty() {
        return Ok(());
    }
    output.push_str("OBSERVABLE_INCLUDE(");
    push_int(output, index);
    output.push(')');
    for target in &targets {
        target.write_to(output);
    }
    output.push('\n');
    Ok(())
}

fn append_pauli_map_stim_targets(
    pauli_map: &PauliMap,
    layout: &QubitLayout,
    targets: &mut Vec<StimTarget>,
) -> Result<(), CircuitError> {
    for (coord, pauli) in pauli_map
        .iter()
        .filter(|&(_, &pauli)| pauli != Pauli::I)
        .sorted_by_key(|(coord, _)| layout.get(**coord).unwrap_or(0))
    {
        let Some(qubit) = layout.get(*coord) else {
            return Err(CircuitError::QubitNotFoundInLayout(*coord));
        };
        targets.push(StimTarget::Pauli {
            pauli: *pauli,
            qubit,
        });
    }
    Ok(())
}

/// Whether a loop's carried states allow summarizing the tail repetitions into
/// a `REPEAT` block instead of unrolling them.
///
/// Summarization records only the *final* iteration's measurement occurrences,
/// so it is sound exactly when every carried state's post-loop value is
/// reconstructible from that final frame: a cleared state, an identity update,
/// or a pure-measurement `next` parity (see [`loop_state_update_kind`]). A
/// `next` that references loop states (its own, or a nested loop's) can fold a
/// *different* absolute measurement set into the state on every iteration —
/// those intermediate occurrences are unrecoverable from a summarized frame —
/// so such loops must fully unroll, whatever their repetition count. Compiled
/// templates never mint one (loop recurrences lower to pure-measurement `next`
/// parities via the flow engine), so only hand-built graphs pay the unroll.
fn can_jump_loop_detector_states(detector_states: &[LoopCarriedDetectorState]) -> bool {
    detector_states
        .iter()
        .all(|state| loop_state_update_kind(state).is_some())
}

/// Sort a parity's resolved terms and XOR-cancel duplicates: a value appearing
/// an even number of times contributes nothing and is dropped. Shared by the
/// absolute-id and lookback representations (equal terms resolve to equal
/// values in both).
fn canonicalize_parity<T: Ord + Copy>(values: &mut Vec<T>) {
    if values.len() < 2 {
        return;
    }
    if values.windows(2).all(|pair| pair[0] < pair[1]) {
        return;
    }

    values.sort_unstable();
    let mut write = 0;
    let mut read = 0;
    while read < values.len() {
        let value = values[read];
        let mut count = 1;
        read += 1;
        while read < values.len() && values[read] == value {
            count += 1;
            read += 1;
        }
        if count % 2 == 1 {
            values[write] = value;
            write += 1;
        }
    }
    values.truncate(write);
}

fn is_identity_update(state: &LoopCarriedDetectorState) -> bool {
    matches!(
        loop_state_update_kind(state),
        Some(LoopStateUpdate::Identity)
    )
}

fn loop_state_update_kind(state: &LoopCarriedDetectorState) -> Option<LoopStateUpdate> {
    match state.next.terms() {
        [] => Some(LoopStateUpdate::Clear),
        [DetectorTerm::LoopState(loop_state)] if *loop_state == state.state => {
            Some(LoopStateUpdate::Identity)
        }
        terms
            if terms
                .iter()
                .all(|term| !matches!(term, DetectorTerm::LoopState(_))) =>
        {
            Some(LoopStateUpdate::Measurements)
        }
        _ => None,
    }
}

fn scope_for_body(circuit: &CoordCircuit, body: BodyId) -> StimAnnotationScope {
    if body == circuit.entry_body() {
        StimAnnotationScope::TopLevel
    } else {
        StimAnnotationScope::RepeatBody { body }
    }
}

pub(crate) fn first_unsupported_gate(circuit: &CoordCircuit) -> Option<GateType> {
    let mut pending = vec![circuit.entry_body()];
    let mut seen = rustc_hash::FxHashSet::default();
    while let Some(body) = pending.pop() {
        if !seen.insert(body) {
            continue;
        }
        let Some(body) = circuit.body(body) else {
            continue;
        };
        for op in body.ops() {
            match op {
                Op::Gate { gate, .. } if gate.is_non_clifford() => return Some(*gate),
                Op::Repeat { body, repetitions } if *repetitions != 0 => pending.push(*body),
                _ => {}
            }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoopStateUpdate {
    Clear,
    Identity,
    Measurements,
}

/// The measurement offsets of one loop-body iteration that matter for frame
/// accounting, plus the total number of measurement records the iteration
/// emits.
///
/// The frame only ever resolves a measurement's *latest* occurrence, so the
/// offset list carries one entry per occurrence that can be a site's last —
/// for a nested `REPEAT` that is its final iteration alone, keeping the
/// accounting O(body ops) instead of O(inner repetitions). The total count
/// still covers every nested repetition, so callers advance the frame by the
/// true number of records.
fn repeat_body_measurement_offsets(
    circuit: &CoordCircuit,
    body: BodyId,
    mapper: &mut impl StimMeasurementMapper,
) -> Result<(Vec<(u32, u32)>, u32), CircuitError> {
    let mut offsets = Vec::new();
    let mut body_measurements = 0;
    collect_repeat_body_measurement_offsets(
        circuit,
        body,
        mapper,
        &mut offsets,
        &mut body_measurements,
    )?;
    Ok((offsets, body_measurements))
}

fn collect_repeat_body_measurement_offsets(
    circuit: &CoordCircuit,
    body: BodyId,
    mapper: &mut impl StimMeasurementMapper,
    offsets: &mut Vec<(u32, u32)>,
    next_offset: &mut u32,
) -> Result<(), CircuitError> {
    let Some(body) = circuit.body(body) else {
        return Err(CircuitError::InvalidCircuitBody(body));
    };
    for op in body.ops() {
        match op {
            Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                for &measurement in measurements {
                    offsets.push((mapper.map_measurement(measurement)?, *next_offset));
                    let Some(next) = next_offset.checked_add(1) else {
                        return Err(MeasurementFrameError::RepeatMeasurementOffsetOverflow.into());
                    };
                    *next_offset = next;
                }
            }
            Op::Repeat {
                body, repetitions, ..
            } => {
                if *repetitions == 0 {
                    continue;
                }
                let mut nested_offsets = Vec::new();
                let mut nested_body_measurements = 0;
                collect_repeat_body_measurement_offsets(
                    circuit,
                    *body,
                    mapper,
                    &mut nested_offsets,
                    &mut nested_body_measurements,
                )?;
                if nested_body_measurements == 0 {
                    continue;
                }
                // Only the nested loop's final iteration can hold a
                // measurement's last occurrence, so record just that
                // iteration's offsets (closed-form) rather than expanding all
                // `repetitions` of them.
                let Some(last_iteration_offset) =
                    nested_body_measurements.checked_mul(*repetitions - 1)
                else {
                    return Err(MeasurementFrameError::RepeatedMeasurementCountOverflow {
                        measurements_per_iteration: nested_body_measurements,
                        repetitions: *repetitions,
                    }
                    .into());
                };
                for &(measurement, nested_offset) in &nested_offsets {
                    let Some(offset) = next_offset
                        .checked_add(last_iteration_offset)
                        .and_then(|offset| offset.checked_add(nested_offset))
                    else {
                        return Err(MeasurementFrameError::RepeatMeasurementOffsetOverflow.into());
                    };
                    offsets.push((measurement, offset));
                }
                let Some(additional) = nested_body_measurements.checked_mul(*repetitions) else {
                    return Err(MeasurementFrameError::RepeatedMeasurementCountOverflow {
                        measurements_per_iteration: nested_body_measurements,
                        repetitions: *repetitions,
                    }
                    .into());
                };
                let Some(next) = next_offset.checked_add(additional) else {
                    return Err(MeasurementFrameError::RepeatMeasurementOffsetOverflow.into());
                };
                *next_offset = next;
            }
            Op::Gate { .. }
            | Op::Tick
            | Op::ConditionalPauli(_)
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => {}
        }
    }
    Ok(())
}

#[inline]
fn indent_prefix(level: usize) -> &'static str {
    const INDENT_TABLE: [&str; 9] = [
        "",
        "    ",
        "        ",
        "            ",
        "                ",
        "                    ",
        "                        ",
        "                            ",
        "                                ",
    ];

    INDENT_TABLE[level.min(INDENT_TABLE.len() - 1)]
}

#[cfg(test)]
mod tests {
    use glam::ivec2;

    use super::*;
    use bloq_circuit::CircuitBody;

    /// Emit a standalone `CoordCircuit` to Stim text. A test convenience;
    /// production emission goes through [`emit_stim_circuit`] on the
    /// program-assembly path.
    fn emit_stim_text(
        circuit: &CoordCircuit,
        annotations: Option<&StimCircuitAnnotations>,
        layout: FxHashMap<IVec2, u32>,
        enable_clifford_proxy: bool,
    ) -> Result<String, StimEmissionError> {
        if !enable_clifford_proxy && let Some(gate) = first_unsupported_gate(circuit) {
            return Err(StimEmissionError::UnsupportedGate(gate));
        }
        let layout = QubitLayout::new(layout)?;
        let (output, _frame) =
            emit_standalone_stim(circuit, annotations, &layout, enable_clifford_proxy)?;
        Ok(output)
    }

    #[test]
    fn to_stim_text_preserves_repeat_blocks() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![measurement],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, 3);

        let text = emit_stim_text(&circuit, None, circuit.build_coord_to_index(), true).unwrap();

        // The repeat body stays a single `REPEAT 3 { ... }` block rather than
        // unrolling into three `M 0` lines.
        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        REPEAT 3 {
            M 0
        }
        ");
    }

    #[test]
    fn emits_first_class_noise_instructions_and_measurement_flip() {
        let q0 = ivec2(0, 0);
        let q1 = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend([
                Op::Depolarize1 {
                    probability: 0.1,
                    qubits: vec![q0],
                },
                Op::Depolarize2 {
                    probability: 0.2,
                    qubits: vec![q0, q1],
                },
                Op::PauliError {
                    probability: 0.3,
                    pauli: PauliBasis::Z,
                    qubits: vec![q1],
                },
                Op::Measure {
                    basis: PauliBasis::X,
                    qubits: vec![q0],
                    measurements: vec![0],
                    flip_probability: 0.4,
                },
            ]);
        circuit.register_measurement_id(0, q0);

        let text = emit_stim_text(&circuit, None, circuit.build_coord_to_index(), true).unwrap();

        assert!(text.contains("DEPOLARIZE1(0.1) 0"));
        assert!(text.contains("DEPOLARIZE2(0.2) 0 1"));
        assert!(text.contains("Z_ERROR(0.3) 1"));
        assert!(text.contains("MX(0.4) 0"));
    }

    #[test]
    fn mpp_emits_pauli_product_targets_with_combiners() {
        use bloq_circuit::{Pauli, PauliMap};
        // Two stabilizer products: X(0,0)*X(2,0) and Z(0,0)*Z(0,2).
        let x_product =
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::X), (ivec2(2, 0), Pauli::X)]);
        let z_product =
            PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::Z), (ivec2(0, 2), Pauli::Z)]);
        let mut circuit = CoordCircuit::new();
        let ids = circuit
            .measure_pauli_products([x_product, z_product])
            .unwrap();
        assert_eq!(ids.len(), 2, "one record per product");
        assert_eq!(circuit.num_measurements(), 2);

        let text = emit_stim_text(&circuit, None, circuit.build_coord_to_index(), true).unwrap();
        // Layout sorts coords (x,y): (0,0)=0, (0,2)=1, (2,0)=2; each product's
        // Paulis are joined with `*` combiners on one `MPP` line.
        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 2) 1
        QUBIT_COORDS(2, 0) 2
        MPP X0*X2 Z0*Z1
        ");
    }

    #[test]
    fn conditional_pauli_emits_record_controlled_xyz() {
        use bloq_circuit::{ConditionalCorrection, PauliBasis};
        let control_qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let control = circuit.measure(PauliBasis::Z, [control_qubit])[0];
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(
                [PauliBasis::X, PauliBasis::Y, PauliBasis::Z]
                    .into_iter()
                    .enumerate()
                    .map(|(index, pauli)| ConditionalCorrection {
                        pauli,
                        control,
                        target: ivec2(index as i32 + 1, 0),
                    })
                    .collect(),
            ));

        let text = emit_stim_text(&circuit, None, circuit.build_coord_to_index(), true).unwrap();

        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        M 0
        CX rec[-1] 1
        CY rec[-1] 2
        CZ rec[-1] 3
        ");
        #[cfg(feature = "verify")]
        text.parse::<stim::Circuit>()
            .expect("record-controlled Pauli gates should parse as Stim");
    }

    #[test]
    fn conditional_pauli_keeps_repeat_body_stable() {
        use bloq_circuit::ConditionalCorrection;
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![qubit],
                measurements: vec![measurement],
                flip_probability: 0.0,
            },
            Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: measurement,
                target: qubit,
            }]),
        ]));
        circuit.push_repeat(body, 3);

        let text = emit_stim_text(&circuit, None, circuit.build_coord_to_index(), true).unwrap();

        insta::assert_snapshot!(text, @"
        QUBIT_COORDS(0, 0) 0
        REPEAT 3 {
            M 0
            CX rec[-1] 0
        }
        ");
        #[cfg(feature = "verify")]
        text.parse::<stim::Circuit>()
            .expect("feed-forward inside REPEAT should parse as Stim");
    }

    #[test]
    fn side_table_repeat_keeps_following_lookbacks_relative_to_final_iteration() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [qubit])[0];
        let repeated = circuit.reserve_measurement_id(qubit);
        let state = LoopStateId(0);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, 3);
        let annotations = StimCircuitAnnotations {
            detectors: vec![
                StimDetector {
                    scope: StimAnnotationScope::RepeatBody { body },
                    parity: DetectorParity::from_terms([
                        DetectorTerm::Measurement(repeated),
                        DetectorTerm::LoopState(state),
                    ]),
                    coords: None,
                    postselection: false,
                },
                StimDetector {
                    scope: StimAnnotationScope::TopLevel,
                    parity: DetectorParity::from_measurements([repeated]),
                    coords: None,
                    postselection: false,
                },
            ],
            repeat_states: vec![StimRepeatState {
                body,
                state,
                initial: DetectorParity::from_measurements([seed]),
                next: DetectorParity::from_measurements([repeated]),
            }],
        };

        let text = emit_stim_text(
            &circuit,
            Some(&annotations),
            circuit.build_coord_to_index(),
            true,
        )
        .unwrap();

        assert!(text.contains("REPEAT 3 {"), "{text}");
        assert!(text.ends_with("}\nDETECTOR rec[-1]\n"), "{text}");
    }

    #[test]
    fn side_table_repeat_partially_unrolls_until_loop_state_lookbacks_are_stable() {
        let seed_qubit = ivec2(0, 0);
        let repeated_qubit = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [seed_qubit])[0];
        let repeated = circuit.reserve_measurement_id(repeated_qubit);
        let state = LoopStateId(0);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![repeated_qubit],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit.measure(PauliBasis::Z, [seed_qubit]);
        circuit.push_repeat(body, 4);
        let annotations = StimCircuitAnnotations {
            detectors: vec![StimDetector {
                scope: StimAnnotationScope::RepeatBody { body },
                parity: DetectorParity::from_terms([
                    DetectorTerm::Measurement(repeated),
                    DetectorTerm::LoopState(state),
                ]),
                coords: Some(bloq_circuit::DetectorCoords::from_slice(&[1.0, 0.0, 1.0])),
                postselection: false,
            }],
            repeat_states: vec![StimRepeatState {
                body,
                state,
                initial: DetectorParity::from_measurements([seed]),
                next: DetectorParity::from_measurements([repeated]),
            }],
        };

        let text = emit_stim_text(
            &circuit,
            Some(&annotations),
            circuit.build_coord_to_index(),
            true,
        )
        .unwrap();

        assert!(
            text.contains("M 0\nM 0\nM 1\nDETECTOR(1, 0, 1) rec[-3] rec[-1]\nREPEAT 3 {"),
            "{text}"
        );
        assert!(
            text.contains("    DETECTOR(1, 0, 1) rec[-2] rec[-1]"),
            "{text}"
        );
    }

    #[test]
    fn side_table_detector_cancels_loop_state_against_measurement() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [qubit])[0];
        let repeated = circuit.reserve_measurement_id(qubit);
        let state = LoopStateId(0);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, 1);
        let annotations = StimCircuitAnnotations {
            detectors: vec![StimDetector {
                scope: StimAnnotationScope::RepeatBody { body },
                parity: DetectorParity::from_terms([
                    DetectorTerm::Measurement(seed),
                    DetectorTerm::LoopState(state),
                ]),
                coords: None,
                postselection: false,
            }],
            repeat_states: vec![StimRepeatState {
                body,
                state,
                initial: DetectorParity::from_measurements([seed]),
                next: DetectorParity::from_measurements([repeated]),
            }],
        };

        let text = emit_stim_text(
            &circuit,
            Some(&annotations),
            circuit.build_coord_to_index(),
            true,
        )
        .unwrap();

        assert!(text.contains("M 0\nM 0\nDETECTOR\n"), "{text}");
    }
}
