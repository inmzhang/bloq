use std::ops::Index;
use std::sync::Arc;

use bloq_circuit::{
    BodyId, ChunkOrLoop, CircuitBody, CoordCircuit, DetectorParity, DetectorTerm, Flow, FlowEngine,
    FlowMarker, FlowMeasurements, OffsetFlows, Op, PauliMap,
};
use bloq_ir::{
    TemplateDetector,
    lowering::{BloqTemplate, TemplateDetectorScope, TemplateRepeatState, TemplateRestart},
};

use crate::CompileError;
use crate::block::ObservableGateway;

/// The compiled template for a single block.
pub(crate) type CompiledTemplate = Arc<LoweringTemplate>;

/// Index of a [`LoweringTemplate`] within a [`LoweringTemplatePool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct LoweringTemplateId(pub(crate) usize);

/// A block's whole compiled circuit plus the canonical detector, observable,
/// and boundary-flow data the backend instantiates per placement.
#[derive(Debug, Clone)]
pub(crate) struct LoweringTemplate {
    /// The whole circuit plus its canonical template-internal detectors, repeat
    /// states, and residual open flows. The backend instantiates these per
    /// placement by a cheap offset + measurement relabel, so internal detectors
    /// never need replay. The residual (promoted onto the persisted template)
    /// lives in [`BloqTemplate::boundary_flows`]. Shared across
    /// identical placements via `Arc` so a repeated template lowers only once.
    pub(crate) program_template: Arc<BloqTemplate>,
    /// Observable routing, in template-measurement-id space (see
    /// [`ObservableGateway::resolve_to_template_ids`]).
    pub(crate) observable_gateway: ObservableGateway,
}

/// Drain the finished composition engine's boundary residual: the open flows
/// the template does not close internally, fused into independent chains.
///
/// `build_program_template` appends every chunk in residual mode, so this is a
/// read-off of that single pass rather than a second composition: internal
/// detectors already fell out through `drain_completed`, and what the engine
/// still holds (chains open, or closed against a boundary input) is the
/// residual, already fused — matching *is* fusion, so a chain threaded across
/// chunk seams becomes one `start -> end` flow. The result is sorted into a
/// canonical order for reproducible emission; order does not affect correctness.
///
/// A loop's recurrence stays in `program_template.repeat_states` (backend-owned):
/// `start_loop_detector_states` rewrites each carried key's seed into the loop
/// state's `initial`, so the seed never reaches the residual, and a top-open loop
/// creator falls out as a bare-measurement flow resolved by the backend's
/// post-`REPEAT` lookback. A *moving* operator inside a loop body instead leaves
/// a residual chain still carrying a `LoopState` term, refused with
/// `LoopBoundaryMovingOperatorUnsupported`.
fn drain_boundary_flows(
    engine: FlowEngine<'_, DetectorTerm<u32>>,
) -> Result<Vec<Flow>, CompileError> {
    // Convert the fused residual to template-id `Flow`s. A surviving `LoopState`
    // term is the moving-operator-in-loop case.
    let residuals = engine.into_residual_flows();
    let mut flows = Vec::with_capacity(residuals.len());
    for residual in residuals {
        let mut measurements = FlowMeasurements::new();
        for term in residual.measurements {
            match term {
                DetectorTerm::Measurement(measurement) => measurements.push(measurement),
                DetectorTerm::LoopState(_) => {
                    return Err(CompileError::LoopBoundaryMovingOperatorUnsupported);
                }
            }
        }
        // Propagate the fused marker onto the persisted boundary flow so the
        // instance-seam detector emission can suppress detectors (or route to a
        // restart) for a selective block's opposite-basis stabilizers.
        flows.push(
            Flow {
                measurements,
                ..Flow::new(residual.start, residual.end)
            }
            .with_sign(residual.sign)
            .with_center(residual.center)
            .with_marker(residual.marker),
        );
    }

    // `into_residual_flows` yields open chains in hash order; sort into a
    // canonical order for reproducible detector emission. `center` and `marker`
    // are the final tiebreakers so flows that agree on measurements/sign/faces
    // (a `Detector` vs. `Restart` twin, or two centers) no longer fall to hash
    // order.
    sort_boundary_flows(&mut flows);
    Ok(flows)
}

pub(super) fn sort_boundary_flows(flows: &mut [Flow]) {
    flows.sort_unstable_by(|a, b| {
        a.measurements
            .cmp(&b.measurements)
            .then_with(|| a.sign.cmp(&b.sign))
            .then_with(|| pauli_map_cmp(&a.start, &b.start))
            .then_with(|| pauli_map_cmp(&a.end, &b.end))
            .then_with(|| {
                a.center
                    .map(|c| c.to_array())
                    .cmp(&b.center.map(|c| c.to_array()))
            })
            .then_with(|| (a.marker as u8).cmp(&(b.marker as u8)))
    });
}

/// Total order on `PauliMap`s by their coordinate-sorted entries, for
/// deterministic residual ordering.
fn pauli_map_cmp(a: &PauliMap, b: &PauliMap) -> std::cmp::Ordering {
    a.iter()
        .map(|(coord, pauli)| (coord.x, coord.y, *pauli as u8))
        .cmp(
            b.iter()
                .map(|(coord, pauli)| (coord.x, coord.y, *pauli as u8)),
        )
}

impl LoweringTemplate {
    /// Build a template from its ordered chunks and observable gateway.
    ///
    /// Composes the per-chunk flows into the program template's internal
    /// detectors and boundary residual, then canonicalizes the gateway from
    /// chunk-local measurement ids to template ids so observable lowering never
    /// has to translate.
    pub(crate) fn from_chunks(
        chunks: Vec<ChunkOrLoop>,
        observable_gateway: ObservableGateway,
    ) -> Result<Self, CompileError> {
        if chunks.iter().any(chunk_has_conditional_pauli) {
            return Err(CompileError::ConditionalPauliFeedbackUnsupported);
        }
        Self::from_chunks_with_commuting_feedback(chunks, observable_gateway)
    }

    /// Build a template whose physical feedback is known to commute with every
    /// persisted flow and boundary operator.
    pub(crate) fn from_chunks_with_commuting_feedback(
        chunks: Vec<ChunkOrLoop>,
        mut observable_gateway: ObservableGateway,
    ) -> Result<Self, CompileError> {
        let mut chunks = normalize_chunks(chunks);
        let ProgramTemplateBuild {
            template: program_template,
            measurement_remaps,
        } = build_program_template(&mut chunks)?;

        // Canonicalize the gateway from chunk-local ids to template ids so
        // observable lowering never has to translate. The flow remap already
        // maps `(chunk, local) -> template id`, so reuse it rather than build a
        // second lookup table.
        let resolve = |chunk_index: usize, local: u32| {
            measurement_remaps
                .get(chunk_index)
                .and_then(|remap| remap.get(local as usize).copied())
                .filter(|&id| id != MISSING_TEMPLATE_MEASUREMENT)
                .expect("gateway measurement was moved into the template")
        };
        observable_gateway.resolve_to_template_ids(resolve);

        Ok(Self {
            program_template: Arc::new(program_template),
            observable_gateway,
        })
    }
}

fn chunk_has_conditional_pauli(chunk: &ChunkOrLoop) -> bool {
    match chunk {
        ChunkOrLoop::Single(chunk) => circuit_has_conditional_pauli(&chunk.circuit),
        ChunkOrLoop::Loop { body, .. } => body
            .iter()
            .any(|chunk| circuit_has_conditional_pauli(&chunk.circuit)),
    }
}

fn circuit_has_conditional_pauli(circuit: &CoordCircuit) -> bool {
    (0..circuit.body_count()).any(|body| {
        circuit.body(BodyId(body as u32)).is_some_and(|body| {
            body.ops()
                .iter()
                .any(|op| matches!(op, Op::ConditionalPauli(_)))
        })
    })
}

const MISSING_TEMPLATE_MEASUREMENT: u32 = u32::MAX;

struct ProgramTemplateBuild {
    template: BloqTemplate,
    measurement_remaps: Vec<Vec<u32>>,
}

fn build_program_template(
    chunks: &mut [ChunkOrLoop],
) -> Result<ProgramTemplateBuild, CompileError> {
    let circuit = CoordCircuit::new();
    let entry = circuit.entry_body();
    let mut entry_ops = Vec::new();
    let mut state = ProgramTemplateState {
        circuit,
        detectors: Vec::new(),
        repeat_states: Vec::new(),
        restarts: Vec::new(),
        measurement_remaps: Vec::new(),
        engine: FlowEngine::new(),
        next_loop_state: 0,
    };

    for chunk_or_loop in chunks {
        match chunk_or_loop {
            ChunkOrLoop::Single(chunk) => {
                append_template_chunk_ops(
                    &mut state,
                    &mut chunk.circuit,
                    &chunk.flows,
                    TemplateDetectorScope::TopLevel,
                    &mut entry_ops,
                );
            }
            ChunkOrLoop::Loop { body, repetitions } => {
                if *repetitions == 0 || body.is_empty() {
                    continue;
                }

                append_separator_tick(&mut entry_ops);
                let body_id = state.circuit.add_body(CircuitBody::new());
                let mut body_ops = Vec::new();
                // Split circuit ownership from the flow borrows retained by the engine.
                let mut parts = body
                    .iter_mut()
                    .map(|chunk| (&mut chunk.circuit, chunk.flows.as_slice()));
                let first = parts.next().expect("non-empty loop body");
                let pending_loop_states = state
                    .engine
                    .start_loop_detector_states(
                        &[OffsetFlows::new(first.1, glam::IVec2::ZERO)],
                        &mut state.next_loop_state,
                    )
                    .expect("compiler-generated loops fit in the loop detector state id space");
                for (circuit, flows) in std::iter::once(first).chain(parts) {
                    append_template_chunk_ops(
                        &mut state,
                        circuit,
                        flows,
                        TemplateDetectorScope::RepeatBody { body: body_id },
                        &mut body_ops,
                    );
                }
                state.repeat_states.extend(
                    state
                        .engine
                        .finish_loop_detector_states(pending_loop_states)
                        .into_iter()
                        .map(|state| TemplateRepeatState {
                            body: body_id,
                            state: state.state,
                            initial: state.initial,
                            next: state.next,
                        }),
                );
                body_ops.push(Op::Tick);
                *state
                    .circuit
                    .body_mut(body_id)
                    .expect("repeat body was allocated above")
                    .ops_mut() = body_ops;
                entry_ops.push(Op::Repeat {
                    body: body_id,
                    repetitions: *repetitions,
                });
            }
        }
    }

    *state
        .circuit
        .body_mut(entry)
        .expect("entry body is created with the circuit")
        .ops_mut() = entry_ops;
    // The same engine that dropped the internal detectors still holds the
    // boundary residual, so reading it off here needs no second composition.
    let boundary_flows = drain_boundary_flows(state.engine)?;
    // Cached templates never append detectors again. Do not retain the flow
    // builder's geometric growth capacity for every signature.
    state.detectors.shrink_to_fit();
    Ok(ProgramTemplateBuild {
        template: BloqTemplate::with_parts(
            state.circuit,
            state.detectors,
            state.repeat_states,
            boundary_flows,
            state.restarts,
        ),
        measurement_remaps: state.measurement_remaps,
    })
}

struct ProgramTemplateState<'c> {
    circuit: CoordCircuit,
    detectors: Vec<TemplateDetector>,
    repeat_states: Vec<TemplateRepeatState>,
    restarts: Vec<TemplateRestart>,
    measurement_remaps: Vec<Vec<u32>>,
    engine: FlowEngine<'c, DetectorTerm<u32>>,
    next_loop_state: u32,
}

fn append_template_chunk_ops<'c>(
    state: &mut ProgramTemplateState<'c>,
    circuit: &mut CoordCircuit,
    flows: &'c [Flow],
    detector_scope: TemplateDetectorScope,
    destination: &mut Vec<Op>,
) {
    let entry = circuit.entry_body();
    let entry_destination = match detector_scope {
        TemplateDetectorScope::TopLevel => state.circuit.entry_body(),
        TemplateDetectorScope::RepeatBody { body } => body,
    };
    let (ops, measurement_map) = {
        let mut builder = TemplateCircuitBuilder {
            source: circuit,
            entry_destination,
            destination: &mut state.circuit,
            measurement_map: Vec::new(),
            body_map: crate::FxMap::default(),
        };
        let ops = builder.take_body_ops(entry);
        (ops, builder.measurement_map)
    };
    append_separator_tick(destination);
    destination.extend(ops);
    append_template_chunk_flows(state, flows, detector_scope, &measurement_map);
    state.measurement_remaps.push(measurement_map);
}

struct TemplateCircuitBuilder<'a, 'b> {
    source: &'a mut CoordCircuit,
    entry_destination: BodyId,
    destination: &'b mut CoordCircuit,
    measurement_map: Vec<u32>,
    body_map: crate::FxMap<BodyId, BodyId>,
}

impl TemplateCircuitBuilder<'_, '_> {
    fn take_body_ops(&mut self, body: BodyId) -> Vec<Op> {
        let source_body = self
            .source
            .body_mut(body)
            .expect("chunk operation references a body in its source circuit");
        std::mem::take(source_body.ops_mut())
            .into_iter()
            .map(|op| self.remap_op(op))
            .collect()
    }

    fn remap_op(&mut self, mut op: Op) -> Op {
        match &mut op {
            Op::Measure {
                qubits,
                measurements,
                ..
            } => {
                for (measurement, &qubit) in measurements.iter_mut().zip(qubits.iter()) {
                    *measurement = self.remap_measurement(*measurement, qubit);
                }
            }
            Op::MPP {
                products,
                measurements,
            } => {
                for (measurement, product) in measurements.iter_mut().zip(products.iter()) {
                    let qubit = product
                        .representative_coord()
                        .expect("MPP product is non-empty");
                    *measurement = self.remap_measurement(*measurement, qubit);
                }
            }
            Op::Repeat { body, .. } => *body = self.remap_body(*body),
            Op::ConditionalPauli(corrections) => {
                for correction in corrections {
                    correction.control = self
                        .template_measurement(correction.control)
                        .expect("ConditionalPauli control precedes its measurement");
                }
            }
            Op::Gate { .. }
            | Op::Tick
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => {}
        }
        op
    }

    fn remap_measurement(&mut self, source_measurement: u32, qubit: glam::IVec2) -> u32 {
        if let Some(measurement) = self.template_measurement(source_measurement) {
            return measurement;
        }
        let measurement = self.destination.reserve_measurement_id(qubit);
        let local = source_measurement as usize;
        if self.measurement_map.len() <= local {
            self.measurement_map
                .resize(local + 1, MISSING_TEMPLATE_MEASUREMENT);
        }
        self.measurement_map[local] = measurement;
        measurement
    }

    fn template_measurement(&self, source_measurement: u32) -> Option<u32> {
        self.measurement_map
            .get(source_measurement as usize)
            .copied()
            .filter(|&measurement| measurement != MISSING_TEMPLATE_MEASUREMENT)
    }

    fn remap_body(&mut self, body: BodyId) -> BodyId {
        // Preserve a malformed back-edge instead of turning a drained entry into
        // an empty body; ordinary circuit validation must still reject the cycle.
        if body == self.source.entry_body() {
            return self.entry_destination;
        }
        if let Some(&body) = self.body_map.get(&body) {
            return body;
        }
        let mapped = self.destination.add_body(CircuitBody::new());
        self.body_map.insert(body, mapped);
        let ops = self.take_body_ops(body);
        *self
            .destination
            .body_mut(mapped)
            .expect("template body was allocated above")
            .ops_mut() = ops;
        mapped
    }
}

fn append_template_chunk_flows<'c>(
    state: &mut ProgramTemplateState<'c>,
    flows: &'c [Flow],
    scope: TemplateDetectorScope,
    measurement_map: &[u32],
) {
    // Residual mode, not Skip: a flow entering through the template boundary is
    // kept as an open chain instead of dropped, so the same pass that closes the
    // internal detectors below also accumulates the boundary residual
    // `drain_boundary_flows` reads off at the end (§6.1.5).
    state
        .engine
        .append_residual_group(
            &[OffsetFlows::new(flows, glam::IVec2::ZERO)],
            |_, &measurement| {
                measurement_map
                    .get(measurement as usize)
                    .copied()
                    .filter(|&measurement| measurement != MISSING_TEMPLATE_MEASUREMENT)
                    .map(DetectorTerm::Measurement)
                    .expect("compiler-authored template flow measurement was moved")
            },
        )
        .expect("compiler-authored template chunks move every measurement their flows reference");
    let completed = state.engine.drain_completed();
    state.detectors.reserve(completed.len());
    for detector in completed {
        // A discarding chain participates in matching but emits no detector; see
        // `FlowMarker::Discard` (a selective block's opposite-basis stabilizers).
        if detector.marker == FlowMarker::Discard
            || (detector.measurements.is_empty() && !detector.sign)
        {
            continue;
        }
        // A restarting chain is a RUS post-selection syndrome (U17b): it lands in
        // the template's `restarts` side table (no coords — restart parities are
        // never decoder food) instead of `detectors`. `TemplateRestart` carries no
        // repeat-body scope, so a restart chain completing inside a loop body has
        // no per-iteration representation yet; no current builder produces one.
        // Hard assert (chunks are compiler-authored): a release fall-through would
        // silently record a per-iteration syndrome as a whole-template parity.
        if detector.marker == FlowMarker::Restart {
            assert_eq!(
                scope,
                TemplateDetectorScope::TopLevel,
                "restart chains inside repeat bodies are unsupported"
            );
            state.restarts.push(TemplateRestart {
                parity: DetectorParity::from_term_buf(detector.measurements)
                    .with_sign(detector.sign),
            });
            continue;
        }
        state.detectors.push(TemplateDetector {
            scope,
            parity: DetectorParity::from_term_buf(detector.measurements).with_sign(detector.sign),
            coords: detector
                .center
                .map(|coord| smallvec::smallvec![coord.x as f64, coord.y as f64]),
        });
    }
}

fn append_separator_tick(ops: &mut Vec<Op>) {
    if matches!(ops.last(), Some(Op::Tick | Op::Repeat { .. }) | None) {
        return;
    }
    ops.push(Op::Tick);
}

fn normalize_chunks(chunks: Vec<ChunkOrLoop>) -> Vec<ChunkOrLoop> {
    let mut normalized = Vec::with_capacity(chunks.len());
    for chunk_or_loop in chunks {
        match chunk_or_loop {
            ChunkOrLoop::Single(chunk) => normalized.push(ChunkOrLoop::Single(chunk)),
            ChunkOrLoop::Loop { repetitions: 0, .. } => {}
            ChunkOrLoop::Loop {
                body,
                repetitions: 1,
                ..
            } => {
                normalized.extend(
                    body.into_iter()
                        .map(|chunk| ChunkOrLoop::Single(Box::new(chunk))),
                );
            }
            ChunkOrLoop::Loop { body, .. } if body.is_empty() => {}
            ChunkOrLoop::Loop { body, repetitions } => {
                normalized.push(ChunkOrLoop::Loop { body, repetitions });
            }
        }
    }
    normalized
}

/// The compilation's store of [`LoweringTemplate`]s, addressed by
/// [`LoweringTemplateId`]. Identical placements share one entry via `Arc`.
#[derive(Debug, Clone, Default)]
pub(crate) struct LoweringTemplatePool {
    templates: Vec<Arc<LoweringTemplate>>,
}

impl LoweringTemplatePool {
    /// Add `template` and return its freshly assigned [`LoweringTemplateId`].
    pub(crate) fn insert(&mut self, template: Arc<LoweringTemplate>) -> LoweringTemplateId {
        debug_assert!(template.program_template.circuit.body_count() > 0);
        let id = LoweringTemplateId(self.templates.len());
        self.templates.push(template);
        id
    }

    /// The template for `id`, or `None` if it is out of range.
    pub(crate) fn get(&self, id: LoweringTemplateId) -> Option<&LoweringTemplate> {
        self.templates.get(id.0).map(Arc::as_ref)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.templates.len()
    }
}

impl Index<LoweringTemplateId> for LoweringTemplatePool {
    type Output = LoweringTemplate;

    fn index(&self, index: LoweringTemplateId) -> &Self::Output {
        self.templates[index.0].as_ref()
    }
}

#[cfg(test)]
mod tests {
    use bloq_circuit::{Chunk, ConditionalCorrection};
    use bloq_circuit::{GateType, Pauli, PauliBasis};
    use glam::ivec2;

    use super::*;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn lowering_pool_indices_do_not_narrow_to_u32() {
        let index = u64::from(u32::MAX) + 1;
        let id = LoweringTemplateId(index.try_into().expect("pool indices cover usize"));
        assert_eq!(u64::try_from(id.0).unwrap(), index);
    }

    fn measured_chunk(targets: &[glam::IVec2]) -> Chunk {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, targets.iter().copied());
        Chunk {
            circuit,
            flows: vec![],
        }
    }

    #[test]
    fn gateway_resolves_chunk_local_ids_to_stage_order_template_ids() {
        use crate::block::gateway::{ChunkMeasurements, GatewayEntry, LocalStabilizer};

        let first = measured_chunk(&[ivec2(0, 0)]);
        let loop_a = measured_chunk(&[ivec2(1, 0)]);
        let loop_b = measured_chunk(&[ivec2(2, 0), ivec2(3, 0)]);
        let last = measured_chunk(&[ivec2(4, 0)]);

        // Reference each chunk-local measurement; after `from_chunks` resolves
        // the gateway, they must land on template ids assigned in stage order:
        // chunk 0 -> 0, chunk 1 -> 1, chunk 2 -> {2, 3}, chunk 3 -> 4.
        let key = LocalStabilizer::isolated(bloq_circuit::Pauli::Z);
        let mut gateway = ObservableGateway::new();
        gateway.insert(
            key,
            GatewayEntry {
                measurements: vec![
                    ChunkMeasurements {
                        chunk_index: 0,
                        measurements: vec![0],
                    },
                    ChunkMeasurements {
                        chunk_index: 1,
                        measurements: vec![0],
                    },
                    ChunkMeasurements {
                        chunk_index: 2,
                        measurements: vec![0, 1],
                    },
                    ChunkMeasurements {
                        chunk_index: 3,
                        measurements: vec![0],
                    },
                ],
                ..Default::default()
            },
        );

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(first)),
                ChunkOrLoop::Loop {
                    body: vec![loop_a, loop_b],
                    repetitions: 3,
                },
                ChunkOrLoop::Single(Box::new(last)),
            ],
            gateway,
        )
        .expect("gateway-resolution template compiles");

        let resolved: Vec<u32> = template.observable_gateway[&key]
            .measurements
            .iter()
            .flat_map(|chunk| chunk.measurements.iter().copied())
            .collect();
        assert_eq!(resolved, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn boundary_flows_keep_only_open_chains() {
        // A chain that opens and closes inside the template (the `internal`
        // key) is template-local and must be absent from the residual; a flow
        // left open on the `external` boundary key must be retained.
        let internal = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);
        let external = PauliMap::from_iter([(ivec2(1, 0), Pauli::Z)]);
        let mut first = measured_chunk(&[ivec2(0, 0)]);
        first.flows.push(
            Flow::new(PauliMap::empty(), internal.clone())
                .with_measurements([0])
                .with_center(ivec2(0, 0)),
        );
        let mut second = measured_chunk(&[ivec2(1, 0)]);
        second.flows.push(
            Flow::new(internal, PauliMap::empty())
                .with_measurements([0])
                .with_center(ivec2(1, 0)),
        );
        second.flows.push(
            Flow::new(PauliMap::empty(), external.clone())
                .with_measurements([0])
                .with_center(ivec2(1, 0)),
        );

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(first)),
                ChunkOrLoop::Single(Box::new(second)),
            ],
            ObservableGateway::new(),
        )
        .expect("boundary-residual template compiles");

        let boundary: Vec<&Flow> = template.program_template.boundary_flows.iter().collect();
        assert_eq!(boundary.len(), 1, "only the open external chain survives");
        assert_eq!(boundary[0].end, external);
        assert!(boundary[0].start.is_empty());
        // The retained boundary flow's measurement is in template-id space, not
        // chunk-local: `second`'s chunk-local id 0 is the second template
        // measurement, so it remaps to 1.
        assert_eq!(
            boundary[0].measurements.as_slice(),
            [1],
            "boundary flow measurement is remapped to its template id"
        );
    }

    #[test]
    fn boundary_flows_keep_top_open_loop_creator() {
        // A loop whose carried key K is left open at the top — nothing after the
        // loop consumes it — must surface its reopening creator as boundary
        // residual so instance lowering can close it against the node above.
        // Before the fix the creator's marker was absorbed into the loop state's
        // `next` and wrongly dropped, leaving `boundary_flows` empty.
        let k = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);

        // init (chunk 0): open K.
        let mut init = measured_chunk(&[ivec2(0, 0)]);
        init.flows.push(
            Flow::new(PauliMap::empty(), k.clone())
                .with_measurements([0])
                .with_center(ivec2(0, 0)),
        );
        // loop body (chunk 1): the split surface code shape — a consumer that
        // closes the per-round detector and a separate creator that re-opens K.
        let mut body = measured_chunk(&[ivec2(1, 0)]);
        body.flows.push(
            Flow::new(k.clone(), PauliMap::empty())
                .with_measurements([0])
                .with_center(ivec2(1, 0)),
        );
        body.flows.push(
            Flow::new(PauliMap::empty(), k.clone())
                .with_measurements([0])
                .with_center(ivec2(1, 0)),
        );

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(init)),
                ChunkOrLoop::Loop {
                    body: vec![body],
                    repetitions: 3,
                },
            ],
            ObservableGateway::new(),
        )
        .expect("top-open loop template compiles");

        // The init seed (in the loop state's `initial`) and the per-round
        // consumer (a completed `RepeatBody` detector) stay internal /
        // backend-owned. Only the reopening creator survives as residual.
        let boundary: Vec<&Flow> = template.program_template.boundary_flows.iter().collect();
        assert_eq!(boundary.len(), 1, "only the top-open loop creator survives");
        assert!(boundary[0].start.is_empty());
        assert_eq!(boundary[0].end, k);
        // It carries the loop body's bare measurement (template id 1), which the
        // backend resolves to the final iteration at the seam.
        assert_eq!(boundary[0].measurements.as_slice(), [1]);
    }

    #[test]
    fn boundary_flows_fuse_cross_chunk_chain() {
        // A chain threaded across a chunk seam by a moving operator: chunk 0
        // opens K, chunk 1 transforms K -> L and leaves L open at the top.
        // Neither closes inside the template, so both flows survive — and must
        // fuse into one boundary flow `∅ -> L` carrying both measurements. That
        // is exactly the case the old `Vec<Vec<Flow>>` kept as two ordered
        // groups; storing the fused chain lets instance lowering append it as one
        // independent flow with no per-chunk ordering to preserve.
        let k = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);
        let l = PauliMap::from_iter([(ivec2(1, 0), Pauli::Z)]);

        let mut first = measured_chunk(&[ivec2(0, 0)]);
        first.flows.push(
            Flow::new(PauliMap::empty(), k.clone())
                .with_measurements([0])
                .with_center(ivec2(0, 0)),
        );
        let mut second = measured_chunk(&[ivec2(1, 0)]);
        second.flows.push(
            Flow::new(k.clone(), l.clone())
                .with_measurements([0])
                .with_center(ivec2(1, 0)),
        );

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(first)),
                ChunkOrLoop::Single(Box::new(second)),
            ],
            ObservableGateway::new(),
        )
        .expect("cross-chunk fusion template compiles");

        assert_eq!(
            template.program_template.boundary_flows.len(),
            1,
            "the threaded chain fuses to a single boundary flow"
        );
        let fused = &template.program_template.boundary_flows[0];
        assert!(
            fused.start.is_empty(),
            "fused chain starts at the template top"
        );
        assert_eq!(fused.end, l, "fused chain ends on the surviving open key");
        // Both rounds' measurements are spliced in (template ids 0 and 1).
        // Order is irrelevant — `DetectorParity` canonicalizes on emission — so
        // the test sorts before comparing.
        let mut measurements = fused.measurements.to_vec();
        measurements.sort_unstable();
        assert_eq!(measurements, [0, 1]);
    }

    #[test]
    fn restart_chain_lands_in_template_restarts_not_detectors() {
        // A restart-flagged chain that closes inside the template (U17b): the
        // fused parity must land in `restarts` — carrying both rounds'
        // measurements — with no `TemplateDetector` emitted for it.
        let seam = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);
        let mut first = measured_chunk(&[ivec2(0, 0)]);
        first.flows.push(
            Flow::new(PauliMap::empty(), seam.clone())
                .with_measurements([0])
                .with_marker(FlowMarker::Restart),
        );
        let mut second = measured_chunk(&[ivec2(1, 0)]);
        second
            .flows
            .push(Flow::new(seam, PauliMap::empty()).with_measurements([0]));

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(first)),
                ChunkOrLoop::Single(Box::new(second)),
            ],
            ObservableGateway::new(),
        )
        .expect("restart-chain template compiles");

        let program = &template.program_template;
        assert!(program.detectors.is_empty(), "no detector for the chain");
        assert_eq!(program.restarts.len(), 1);
        assert_eq!(
            program.restarts[0].parity,
            DetectorParity::from_measurements([0, 1])
        );
    }

    #[test]
    fn boundary_flow_preserves_restart_marker() {
        // A restart-flagged chain left open at the template boundary must keep
        // its marker on the persisted `boundary_flows`, so the instance-seam
        // pass can route the fused cross-template parity to a `NodeRestart`.
        let open = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);
        let mut chunk = measured_chunk(&[ivec2(0, 0)]);
        chunk.flows.push(
            Flow::new(PauliMap::empty(), open.clone())
                .with_measurements([0])
                .with_marker(FlowMarker::Restart),
        );

        let template = LoweringTemplate::from_chunks(
            vec![ChunkOrLoop::Single(Box::new(chunk))],
            ObservableGateway::new(),
        )
        .expect("open restart-chain template compiles");

        let boundary = &template.program_template.boundary_flows;
        assert_eq!(boundary.len(), 1);
        assert_eq!(boundary[0].end, open);
        assert_eq!(boundary[0].marker, FlowMarker::Restart);
    }

    #[test]
    fn moving_operator_in_loop_is_rejected() {
        // A pass-through (moving) flow `K -> K` inside a loop body makes the
        // carried residual reference the loop state itself (`next(s)` contains a
        // `LoopState`), which a bare-measurement seam cannot express. It must be
        // refused, not silently miscompiled.
        let k = PauliMap::from_iter([(ivec2(0, 0), Pauli::Z)]);

        let mut init = measured_chunk(&[ivec2(0, 0)]);
        init.flows
            .push(Flow::new(PauliMap::empty(), k.clone()).with_measurements([0]));
        let mut body = measured_chunk(&[ivec2(1, 0)]);
        body.flows
            .push(Flow::new(k.clone(), k.clone()).with_measurements([0]));

        let result = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(init)),
                ChunkOrLoop::Loop {
                    body: vec![body],
                    repetitions: 3,
                },
            ],
            ObservableGateway::new(),
        );
        assert!(matches!(
            result,
            Err(CompileError::LoopBoundaryMovingOperatorUnsupported)
        ));
    }

    #[test]
    fn compiler_chunk_feedback_requires_the_commuting_path() {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let control = circuit.measure(PauliBasis::Z, [qubit])[0];
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: qubit,
            }]));

        assert!(matches!(
            LoweringTemplate::from_chunks(
                vec![ChunkOrLoop::Single(Box::new(Chunk {
                    circuit,
                    flows: Vec::new(),
                }))],
                ObservableGateway::new(),
            ),
            Err(CompileError::ConditionalPauliFeedbackUnsupported)
        ));
    }

    #[test]
    fn moved_chunk_ops_remap_shared_bodies_and_feedback_once() {
        let first = measured_chunk(&[ivec2(0, 0)]);
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(1, 0)]);
        let control = circuit.reserve_measurement_id(ivec2(2, 0));
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![ivec2(2, 0)],
                measurements: vec![control],
                flip_probability: 0.0,
            },
            Op::Tick,
        ]));
        circuit.push_repeat(body, 2);
        circuit.push_repeat(body, 3);
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: ivec2(0, 0),
            }]));
        let template = LoweringTemplate::from_chunks_with_commuting_feedback(
            vec![
                ChunkOrLoop::Single(Box::new(first)),
                ChunkOrLoop::Single(Box::new(Chunk {
                    circuit,
                    flows: Vec::new(),
                })),
            ],
            ObservableGateway::new(),
        )
        .unwrap();
        let circuit = &template.program_template.circuit;
        assert_eq!(circuit.body_count(), 2);
        assert_eq!(circuit.meas_registry().records().len(), 3);
        let entry = circuit.body(circuit.entry_body()).unwrap().ops();
        let repeated: Vec<_> = entry
            .iter()
            .filter_map(|op| match op {
                Op::Repeat { body, .. } => Some(*body),
                _ => None,
            })
            .collect();
        assert_eq!(repeated, [BodyId(1), BodyId(1)]);
        assert!(
            matches!(&circuit.body(BodyId(1)).unwrap().ops()[0], Op::Measure { measurements, .. } if measurements == &[2])
        );
        assert!(
            matches!(entry.last(), Some(Op::ConditionalPauli(corrections)) if corrections[0].control == 2)
        );
    }

    #[test]
    fn moving_entry_ops_preserves_cycle_rejection() {
        let mut circuit = CoordCircuit::new();
        circuit.push_repeat(circuit.entry_body(), 1);
        let template = LoweringTemplate::from_chunks(
            vec![ChunkOrLoop::Single(Box::new(Chunk {
                circuit,
                flows: Vec::new(),
            }))],
            ObservableGateway::new(),
        )
        .unwrap();
        template
            .program_template
            .circuit
            .expanded_measurement_columns()
            .unwrap_err();
    }

    #[test]
    fn lowering_template_builds_annotation_free_program_template() {
        let data = ivec2(9, 0);
        let mut first = CoordCircuit::new();
        first.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let first_measurement = first.measure(PauliBasis::Z, [ivec2(0, 0)])[0];
        let mut second = CoordCircuit::new();
        let second_measurement = second.measure(PauliBasis::Z, [ivec2(1, 0)])[0];
        let boundary: PauliMap = [(data, Pauli::Z)].into_iter().collect();

        let template = LoweringTemplate::from_chunks(
            vec![
                ChunkOrLoop::Single(Box::new(Chunk {
                    circuit: first,
                    flows: vec![
                        Flow::new(PauliMap::empty(), PauliMap::empty())
                            .with_measurements([first_measurement])
                            .with_center(ivec2(0, 0)),
                        Flow::new(PauliMap::empty(), boundary.clone())
                            .with_measurements([first_measurement])
                            .with_center(ivec2(0, 0)),
                    ],
                })),
                ChunkOrLoop::Loop {
                    body: vec![Chunk {
                        circuit: second,
                        flows: vec![
                            Flow::new(boundary.clone(), PauliMap::empty())
                                .with_measurements([second_measurement])
                                .with_center(ivec2(1, 0)),
                            Flow::new(PauliMap::empty(), boundary)
                                .with_measurements([second_measurement])
                                .with_center(ivec2(1, 0)),
                        ],
                    }],
                    repetitions: 2,
                },
            ],
            ObservableGateway::new(),
        )
        .expect("annotation-free template compiles");
        let program_circuit = &template.program_template.circuit;

        assert_eq!(program_circuit.meas_registry().records().len(), 2);
        assert!(
            program_circuit
                .body(program_circuit.entry_body())
                .unwrap()
                .ops()
                .iter()
                .all(|op| matches!(
                    op,
                    Op::Gate { .. } | Op::Measure { .. } | Op::Tick | Op::Repeat { .. }
                ))
        );
        assert!(matches!(
            program_circuit
                .body(program_circuit.entry_body())
                .unwrap()
                .ops()
                .last(),
            Some(Op::Repeat { repetitions: 2, .. })
        ));
        assert_eq!(template.program_template.detectors.len(), 2);
        assert!(matches!(
            template.program_template.detectors[0].scope,
            TemplateDetectorScope::TopLevel
        ));
        assert_eq!(
            template.program_template.detectors[0].parity,
            DetectorParity::from_measurements([0])
        );
        assert!(matches!(
            template.program_template.detectors[1].scope,
            TemplateDetectorScope::RepeatBody { .. }
        ));
        assert_eq!(
            template.program_template.detectors[1].parity,
            DetectorParity::from_terms([
                bloq_circuit::DetectorTerm::Measurement(1),
                bloq_circuit::DetectorTerm::LoopState(bloq_circuit::LoopStateId(0)),
            ])
        );
        assert_eq!(template.program_template.repeat_states.len(), 1);
        let TemplateDetectorScope::RepeatBody { body } =
            template.program_template.detectors[1].scope
        else {
            panic!("second detector should live in the repeat body");
        };
        assert_eq!(template.program_template.repeat_states[0].body, body);
        assert_eq!(template.program_template.repeat_states[0].state.0, 0);
        assert_eq!(
            template.program_template.repeat_states[0].initial,
            DetectorParity::from_measurements([0])
        );
    }
}
