//! Repeat-block flattening.
//!
//! [`BloqTemplate::flatten`] unrolls a template's `REPEAT` bodies into a
//! straight-line circuit and rewrites its side tables to match: repeat-scoped
//! detectors expand into one concrete top-level detector **per unrolled
//! iteration**, loop-carried recurrences ([`TemplateRepeatState`]) are
//! resolved away iteration by iteration (the `initial` parity seeds the first
//! iteration, `next` advances the state after each), and the cleared
//! `repeat_states` table is what "no loop state left" means. The expansion
//! replays [`CoordCircuit::flatten`]'s event trace with the same semantics the
//! Stim emitter applies at emission time, so a flattened template emits the
//! same detector set the looped one did.
//!
//! [`Bloq::flatten`] applies this to a whole program: every template an
//! instance references is flattened once (memoized) and instances are
//! re-pointed at the flattened copies. Instance-space references stay valid
//! without any remap because the final unrolled occurrence of each
//! measurement keeps its original id — exactly the record the emission
//! frame's latest-occurrence rule already resolved out-of-loop references to.

use bloq_circuit::{
    BodyId, CircuitError, DetectorCoords, DetectorParity, DetectorTerm, FlattenEvent,
    FlattenLimits, LoopStateId,
};
use thiserror::Error;

use crate::{
    Bloq, BloqNodeKind, BloqTemplate, DetectorBundle, DetectorBundleId, FxMap, SubGraph,
    TemplateDetector, TemplateDetectorParity, TemplateDetectorScope, TemplateId, TemplateRestart,
};

/// Why a template or program could not be flattened. The target is left
/// unmodified on any of these.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FlattenError {
    /// Appending flattened templates exhausted their `u32` pool ids.
    #[error("template id space exhausted while flattening")]
    TemplateIdOverflow,
    /// A detector bundle use or definition is malformed.
    #[error("{0}")]
    DetectorBundle(#[from] crate::DetectorBundleError),
    /// Unrolling the circuit itself failed (dangling/self-recursive body,
    /// measurement-id space overflow, or a resource allowance).
    #[error("{context}: {0}", context = if matches!(.0, CircuitError::FlattenResourceLimit { .. }) {
        "resource limit flattening the circuit"
    } else {
        "flattening the circuit failed"
    })]
    Circuit(#[from] CircuitError),
    /// A parity references a loop state no reachable repeat body defines
    /// before the reference resolves.
    #[error("parity references undefined loop state {0:?}")]
    UnresolvedLoopState(LoopStateId),
    /// An instance references a template missing from the pool.
    #[error("instance references unknown template {0:?}")]
    UnknownTemplate(TemplateId),
}

fn new_template_id(base: usize, index: usize) -> Result<TemplateId, FlattenError> {
    let index = base
        .checked_add(index)
        .ok_or(FlattenError::TemplateIdOverflow)?;
    u32::try_from(index)
        .map(TemplateId)
        .map_err(|_| FlattenError::TemplateIdOverflow)
}

impl FlattenError {
    /// Whether materialization exceeded an allowance rather than failing a
    /// circuit or loop-state check.
    #[must_use]
    pub fn is_resource_limited(&self) -> bool {
        matches!(
            self,
            Self::Circuit(CircuitError::FlattenResourceLimit { .. })
        )
    }
}

struct FlattenWork {
    limits: FlattenLimits,
    used: usize,
}

impl FlattenWork {
    fn new(limits: FlattenLimits) -> Self {
        Self { limits, used: 0 }
    }

    fn charge(&mut self, amount: usize) -> Result<(), FlattenError> {
        let observed = self.used.saturating_add(amount);
        if observed == usize::MAX || observed > self.limits.max_work {
            return Err(CircuitError::FlattenResourceLimit {
                observed,
                limit: self.limits.max_work,
            }
            .into());
        }
        self.used = observed;
        Ok(())
    }

    fn check_circuit(&mut self, circuit: &bloq_circuit::CoordCircuit) -> Result<(), FlattenError> {
        self.charge(crate::instantiation::circuit_work(circuit))?;
        let remaining = FlattenLimits {
            max_work: self.limits.max_work - self.used,
        };
        match circuit.check_flatten_limits(remaining) {
            Ok(amount)
            | Err(CircuitError::FlattenResourceLimit {
                observed: amount, ..
            }) => self.charge(amount),
            Err(error) => Err(error.into()),
        }
    }
}

impl BloqTemplate {
    /// This template with every repeat block flattened away (see the module
    /// docs): the circuit unrolled, repeat-scoped detectors expanded into one
    /// top-level detector per iteration, loop-carried states resolved and
    /// `repeat_states` cleared. Detector coordinates are copied verbatim per
    /// iteration, matching what the emitter writes for each pass over a
    /// repeat body. `boundary_flows` pass through unchanged: they name final
    /// occurrences, which keep their ids.
    ///
    /// A template without repeats (and without loop-state references) round
    /// trips unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FlattenError`] for invalid loop-state references, malformed
    /// repeat bodies, or exhausted expansion limits.
    pub fn flatten(&self) -> Result<BloqTemplate, FlattenError> {
        self.flatten_with_limits(FlattenLimits::default())
    }

    /// [`Self::flatten`] with one allowance for circuit expansion and side-table
    /// replay, including substituted loop-state parity terms.
    ///
    /// # Errors
    ///
    /// Returns [`FlattenError`] under the same conditions as [`Self::flatten`],
    /// including when `limits` is exhausted.
    pub fn flatten_with_limits(&self, limits: FlattenLimits) -> Result<BloqTemplate, FlattenError> {
        let mut work = FlattenWork::new(limits);
        work.check_circuit(&self.circuit)?;
        self.flatten_with_trace_budget(&mut work)
            .map(|(template, _)| template)
    }

    pub(crate) fn flatten_with_trace(
        &self,
    ) -> Result<(BloqTemplate, Vec<FlattenEvent>), FlattenError> {
        let mut work = FlattenWork::new(FlattenLimits::default());
        work.check_circuit(&self.circuit)?;
        self.flatten_with_trace_budget(&mut work)
    }

    fn flatten_with_trace_budget(
        &self,
        work: &mut FlattenWork,
    ) -> Result<(BloqTemplate, Vec<FlattenEvent>), FlattenError> {
        work.charge(self.repeat_states.len())?;
        work.charge(self.detectors.len())?;
        work.charge(self.restarts.len())?;
        work.charge(self.boundary_flows.len())?;
        for flow in &self.boundary_flows {
            work.charge(
                flow.start
                    .len()
                    .saturating_add(flow.end.len())
                    .saturating_add(flow.measurements.len()),
            )?;
        }
        let referenced_bodies = self.repeat_states.iter().map(|state| state.body).chain(
            self.detectors
                .iter()
                .filter_map(|detector| match detector.scope {
                    TemplateDetectorScope::TopLevel => None,
                    TemplateDetectorScope::RepeatBody { body } => Some(body),
                }),
        );
        if let Some(body) = referenced_bodies
            .into_iter()
            .find(|&body| self.circuit.body(body).is_none())
        {
            return Err(CircuitError::InvalidCircuitBody(body).into());
        }

        let mut circuit = self.circuit.clone();
        // Expansion was charged before any template was materialized. Its
        // own preflight still protects callers of this circuit-level API.
        let trace = circuit.flatten_with_limits(work.limits)?;

        // Index the loop-relative side tables by the repeat body they live in.
        let mut states_by_body: FxMap<BodyId, Vec<&crate::TemplateRepeatState>> = FxMap::default();
        for state in &self.repeat_states {
            states_by_body.entry(state.body).or_default().push(state);
        }
        let mut detectors_by_body: FxMap<BodyId, Vec<&TemplateDetector>> = FxMap::default();
        let mut top_level_detectors = Vec::new();
        for detector in &self.detectors {
            match detector.scope {
                TemplateDetectorScope::TopLevel => top_level_detectors.push(detector),
                TemplateDetectorScope::RepeatBody { body } => {
                    detectors_by_body.entry(body).or_default().push(detector);
                }
            }
        }

        // Replay the unroll: `latest` mirrors the emission frame (original id
        // -> the id of its most recent unrolled occurrence). The state stack
        // mirrors Stim's lexical loop frames; the base frame is the resolved
        // top-level state table.
        let mut latest: FxMap<u32, u32> = FxMap::default();
        let mut state_stack: Vec<FxMap<LoopStateId, TemplateDetectorParity>> =
            vec![FxMap::default()];
        let mut flat_detectors = Vec::new();
        // Reused across every `IterationEnd`: each `next` must resolve against
        // the pre-update `states` before any commits, so they stage here first.
        let mut next_states: Vec<(LoopStateId, TemplateDetectorParity)> = Vec::new();
        for event in &trace {
            work.charge(1)?;
            match *event {
                FlattenEvent::Measurement { original, emitted } => {
                    latest.insert(original, emitted);
                }
                FlattenEvent::LoopEnter { body } => {
                    let mut initial_states = FxMap::default();
                    for state in states_in(&states_by_body, body) {
                        let initial = resolve_parity(&state.initial, &latest, &state_stack, work)?;
                        initial_states.insert(state.state, initial);
                    }
                    state_stack.push(initial_states);
                }
                FlattenEvent::IterationEnd { body } => {
                    for detector in detectors_by_body.get(&body).into_iter().flatten() {
                        work.charge(detector.coords.as_ref().map_or(0, DetectorCoords::len))?;
                        flat_detectors.push(TemplateDetector {
                            scope: TemplateDetectorScope::TopLevel,
                            parity: resolve_parity(&detector.parity, &latest, &state_stack, work)?,
                            coords: detector.coords.clone(),
                        });
                    }
                    // The emitter builds the whole next frame before swapping it
                    // in, so stage every `next` (against pre-update states), then
                    // commit.
                    next_states.clear();
                    for state in states_in(&states_by_body, body) {
                        next_states.push((
                            state.state,
                            resolve_parity(&state.next, &latest, &state_stack, work)?,
                        ));
                    }
                    // Stim leaves a state-less loop frame untouched during an
                    // iteration, but a loop with declared states replaces the
                    // frame wholesale with those next values.
                    if !next_states.is_empty() {
                        for (_, parity) in &next_states {
                            work.charge(parity.terms().len())?;
                        }
                        let current = state_stack
                            .last_mut()
                            .expect("LoopEnter precedes IterationEnd");
                        current.clear();
                        current.extend(next_states.iter().cloned());
                    }
                }
                FlattenEvent::LoopExit { body } => {
                    let mut final_states = state_stack.pop().expect("LoopEnter precedes LoopExit");
                    let parent = state_stack
                        .last_mut()
                        .expect("the top-level state frame is never popped");
                    for state in states_in(&states_by_body, body) {
                        let value = final_states
                            .remove(&state.state)
                            .ok_or(FlattenError::UnresolvedLoopState(state.state))?;
                        parent.insert(state.state, value);
                    }
                }
            }
        }

        // Top-level parities resolve after everything emitted: measurement
        // terms already name final occurrences, and any loop-state term takes
        // the state's post-loop value.
        for detector in top_level_detectors {
            work.charge(detector.coords.as_ref().map_or(0, DetectorCoords::len))?;
            flat_detectors.push(TemplateDetector {
                scope: TemplateDetectorScope::TopLevel,
                parity: resolve_parity(&detector.parity, &latest, &state_stack, work)?,
                coords: detector.coords.clone(),
            });
        }
        let restarts = self
            .restarts
            .iter()
            .map(|restart| {
                Ok(TemplateRestart {
                    parity: resolve_parity(&restart.parity, &latest, &state_stack, work)?,
                })
            })
            .collect::<Result<Vec<_>, FlattenError>>()?;

        let mut template = BloqTemplate::with_parts(
            circuit,
            flat_detectors,
            Vec::new(),
            self.boundary_flows.clone(),
            restarts,
        );
        if !self.readout_frames.is_empty() {
            let mut by_flag = FxMap::<_, Vec<_>>::default();
            for frame in &self.readout_frames {
                by_flag.entry(frame.flag).or_default().push(*frame);
            }
            for &event in &trace {
                if let FlattenEvent::Measurement { original, emitted } = event
                    && let Some(frames) = by_flag.get(&original)
                {
                    work.charge(frames.len())?;
                    template.readout_frames.extend(frames.iter().map(|frame| {
                        crate::ReadoutFrame {
                            flag: emitted,
                            ..*frame
                        }
                    }));
                }
            }
        }
        Ok((template, trace))
    }
}

/// The repeat states declared for `body`, or nothing when it carries none.
fn states_in<'a>(
    by_body: &'a FxMap<BodyId, Vec<&'a crate::TemplateRepeatState>>,
    body: BodyId,
) -> impl Iterator<Item = &'a crate::TemplateRepeatState> + 'a {
    by_body.get(&body).into_iter().flatten().copied()
}

/// Resolve a parity to concrete measurement ids on the flattened circuit:
/// measurement terms follow their most recent unrolled occurrence (falling
/// back to the id itself for measurements outside any loop), loop-state terms
/// substitute the state's currently resolved parity. Terms and signs compose
/// by XOR; term cancellation happens in
/// [`DetectorParity::from_measurements`]'s canonicalization.
fn resolve_parity(
    parity: &TemplateDetectorParity,
    latest: &FxMap<u32, u32>,
    state_stack: &[FxMap<LoopStateId, TemplateDetectorParity>],
    work: &mut FlattenWork,
) -> Result<TemplateDetectorParity, FlattenError> {
    work.charge(parity.terms().len().saturating_add(1))?;
    let mut values = Vec::with_capacity(parity.terms().len());
    let mut sign = parity.sign();
    for term in parity.terms() {
        match *term {
            DetectorTerm::Measurement(measurement) => {
                values.push(latest.get(&measurement).copied().unwrap_or(measurement));
            }
            DetectorTerm::LoopState(state) => {
                work.charge(state_stack.len())?;
                let resolved = state_stack
                    .iter()
                    .rev()
                    .find_map(|states| states.get(&state))
                    .ok_or(FlattenError::UnresolvedLoopState(state))?;
                sign ^= resolved.sign();
                work.charge(resolved.terms().len())?;
                values.extend(resolved.measurements());
            }
        }
    }
    Ok(DetectorParity::from_measurements(values).with_sign(sign))
}

impl BloqTemplate {
    /// Whether [`Self::flatten`] would change anything — i.e. the template
    /// carries loop structure (a repeat body, a loop-carried recurrence, a
    /// repeat-scoped detector, or a loop-state parity term). Colocated with
    /// `flatten` because it must cover exactly what `flatten` rewrites: an
    /// under-report would leak a `REPEAT` past a whole-program flatten (an
    /// instance never re-pointed), so a new side table added to `flatten` must
    /// be reflected here too. Checks run cheapest-first (O(1) before the body
    /// and side-table walks).
    fn needs_flatten(&self) -> bool {
        !self.repeat_states.is_empty()
            || self
                .detectors
                .iter()
                .any(|detector| detector.scope != TemplateDetectorScope::TopLevel)
            || self.circuit.has_repeats()
            || self
                .detectors
                .iter()
                .map(|detector| &detector.parity)
                .chain(self.restarts.iter().map(|restart| &restart.parity))
                .any(|parity| {
                    parity
                        .terms()
                        .iter()
                        .any(|term| matches!(term, DetectorTerm::LoopState(_)))
                })
    }
}

impl Bloq {
    /// Flatten every repeat block in the program (see the module docs):
    /// each template an instance references is flattened once via
    /// [`BloqTemplate::flatten`] (flattened copies are appended to the pool —
    /// it is append-only — and instances re-pointed), expanding repeat-scoped
    /// detectors into per-iteration detectors and clearing all loop-carried
    /// detector state. Instance-space side tables (node detectors,
    /// observable record reads, boundary operators) stay valid untouched because
    /// final occurrences keep their measurement ids.
    ///
    /// Verify-before-apply: all templates are flattened and all checks run
    /// before any instance is re-pointed, so the program is unmodified on
    /// error. Decoder-wait insertion after flattening splices looped padding
    /// templates again; flatten again afterwards if needed.
    ///
    /// # Errors
    ///
    /// See [`FlattenError`].
    pub fn flatten(&mut self) -> Result<(), FlattenError> {
        self.flatten_with_limits(FlattenLimits::default())
    }

    /// [`Self::flatten`] with a cumulative allowance across all distinct
    /// referenced templates and their side tables. Repeated instances share
    /// one flattened template; an exhausted allowance leaves the Bloq intact.
    ///
    /// # Errors
    ///
    /// Returns [`FlattenError`] for invalid references or exhausted cumulative
    /// work; `self` remains unchanged.
    pub fn flatten_with_limits(&mut self, limits: FlattenLimits) -> Result<(), FlattenError> {
        let mut work = FlattenWork::new(limits);
        // Bundle definitions are shared. Check their terms once, and preserve
        // their handles while instance-local template ids are repointed.
        for (_, bundle) in self.detector_bundles().iter() {
            work.charge(
                bundle
                    .detectors()
                    .len()
                    .saturating_add(bundle.owner_templates().len()),
            )?;
            for detector in bundle.detectors() {
                work.charge(detector.parity.terms().len())?;
                if let Some(state) = detector.parity.terms().iter().find_map(|term| match term {
                    DetectorTerm::LoopState(state) => Some(*state),
                    DetectorTerm::Measurement(_) => None,
                }) {
                    return Err(FlattenError::UnresolvedLoopState(state));
                }
            }
        }
        // Phase 1 (read-only): flatten every referenced template that needs it.
        let mut memo: FxMap<TemplateId, TemplateId> = FxMap::default();
        let mut referenced = Vec::new();
        check_and_collect(self.top(), self, &mut memo, &mut referenced, &mut work)?;
        if let Some(last) = referenced.len().checked_sub(1) {
            new_template_id(self.templates().len(), last)?;
        }
        // Check the aggregate expansion before materializing even the first
        // template, so a pool cannot evade the limit by splitting its repeats.
        for &id in &referenced {
            work.check_circuit(&self.templates()[id].circuit)?;
        }
        let pending = referenced
            .into_iter()
            .map(|id| {
                self.templates()[id]
                    .flatten_with_trace_budget(&mut work)
                    .map(|(template, _)| (id, template))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if pending.is_empty() {
            return Ok(());
        }
        for (index, (old, _)) in pending.iter().enumerate() {
            memo.insert(*old, new_template_id(self.templates().len(), index)?);
        }
        let mut replacement_bundles = Vec::new();
        for (old, bundle) in self.detector_bundles().iter() {
            work.charge(bundle.owner_templates().len())?;
            let owners = bundle
                .owner_templates()
                .iter()
                .map(|template| memo.get(template).copied().unwrap_or(*template))
                .collect::<Vec<_>>();
            if owners != bundle.owner_templates() {
                work.charge(bundle.detectors().len())?;
                for row in bundle.detectors() {
                    work.charge(
                        row.parity
                            .terms()
                            .len()
                            .saturating_add(row.coords.as_ref().map_or(0, DetectorCoords::len)),
                    )?;
                }
                replacement_bundles.push((
                    old,
                    DetectorBundle::new(owners, bundle.detectors().to_vec()),
                ));
            }
        }
        // Phase 2 (infallible): install only already-budgeted replacements.
        if let Some(last) = replacement_bundles.len().checked_sub(1) {
            let index = self
                .detector_bundles()
                .len()
                .checked_add(last)
                .ok_or(crate::DetectorBundleError::CountOverflow)?;
            u32::try_from(index).map_err(|_| crate::DetectorBundleError::CountOverflow)?;
        }
        for (_, template) in pending {
            self.add_template(template);
        }
        let mut bundle_memo = FxMap::default();
        for (old, bundle) in replacement_bundles {
            bundle_memo.insert(old, self.add_detector_bundle(bundle));
        }
        rewrite_level(self.graph_mut_internal(), &memo, &bundle_memo);
        Ok(())
    }
}

fn check_and_collect(
    level: &SubGraph,
    program: &Bloq,
    memo: &mut FxMap<TemplateId, TemplateId>,
    pending: &mut Vec<TemplateId>,
    work: &mut FlattenWork,
) -> Result<(), FlattenError> {
    let mut levels = vec![level.nodes()];
    while let Some(nodes) = levels.last_mut() {
        let Some((_, node)) = nodes.next() else {
            levels.pop();
            continue;
        };
        work.charge(1)?;
        match &node.kind {
            BloqNodeKind::Quantum(quantum) => {
                work.charge(quantum.detector_bundles.len())?;
                for use_ in &quantum.detector_bundles {
                    work.charge(use_.instances.len())?;
                }
                program.node_detector_count(quantum)?;
                // Node-level loop-state terms have no recurrence table and
                // therefore cannot resolve after template flattening.
                for parity in quantum.stored_parities() {
                    work.charge(parity.terms().len())?;
                    if let Some(state) = parity.terms().iter().find_map(|term| match term {
                        DetectorTerm::LoopState(state) => Some(*state),
                        DetectorTerm::Measurement(_) => None,
                    }) {
                        return Err(FlattenError::UnresolvedLoopState(state));
                    }
                }
                for instance in &quantum.instances {
                    work.charge(1)?;
                    if memo.contains_key(&instance.template_id) {
                        continue;
                    }
                    let template = program
                        .templates()
                        .get(instance.template_id)
                        .ok_or(FlattenError::UnknownTemplate(instance.template_id))?;
                    work.charge(crate::instantiation::circuit_work(&template.circuit))?;
                    if template.needs_flatten() {
                        // The pool id is assigned in phase 2; mark the memo so
                        // the template is flattened only once.
                        pending.push(instance.template_id);
                    }
                    memo.insert(instance.template_id, instance.template_id);
                }
            }
            BloqNodeKind::Region(region) => {
                for (_, body) in region.bodies().collect::<Vec<_>>().into_iter().rev() {
                    levels.push(body.nodes());
                }
            }
            BloqNodeKind::Classical(_) => {}
        }
    }
    Ok(())
}

fn rewrite_level(
    graph: &mut petgraph::stable_graph::StableDiGraph<crate::BloqNode, crate::BloqEdge>,
    memo: &FxMap<TemplateId, TemplateId>,
    bundle_memo: &FxMap<DetectorBundleId, DetectorBundleId>,
) {
    let mut pending = vec![graph];
    while let Some(graph) = pending.pop() {
        for node in graph.node_weights_mut() {
            if node.try_quantum().is_some() {
                let quantum = node.expect_quantum_mut();
                for instance in &mut quantum.instances {
                    instance.template_id = *memo
                        .get(&instance.template_id)
                        .expect("phase 1 visited every instance");
                }
                for use_ in &mut quantum.detector_bundles {
                    if let Some(&new) = bundle_memo.get(&use_.bundle) {
                        use_.bundle = new;
                    }
                }
                continue;
            }
            // Walks the same body structure `check_and_collect` checked via
            // `bodies()`, so the two phases cannot drift.
            if let BloqNodeKind::Region(region) = &mut node.kind {
                for (_, body) in region.bodies_mut() {
                    pending.push(body.graph_mut());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn flattened_template_ids_do_not_wrap() {
        let maximum = u32::MAX as usize;
        assert_eq!(
            super::new_template_id(maximum, 0),
            Ok(crate::TemplateId(u32::MAX))
        );
        if maximum < usize::MAX {
            assert_eq!(
                super::new_template_id(maximum, 1),
                Err(super::FlattenError::TemplateIdOverflow)
            );
        }
    }
    use bloq_circuit::{CircuitBody, CoordCircuit, Op, PauliBasis};
    use glam::ivec2;

    use super::*;
    use crate::{
        BloqNode, BundleDetector, BundleMeasurement, DetectorBundleUse, TemplateInstance,
        TemplateInstanceId, TemplateRepeatState,
    };

    /// A seed measurement, then `REPEAT 3 { M }`, with the loop recurrence
    /// `initial = seed`, `next = repeated` and a body detector comparing the
    /// state against the current round — the memory-template shape the Stim
    /// emitter's repeat tests use.
    fn looped_template() -> (BloqTemplate, BodyId, u32, u32) {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [qubit])[0];
        let repeated = circuit.reserve_measurement_id(qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 3,
            });
        let state = LoopStateId(0);
        let template = BloqTemplate::with_parts(
            circuit,
            vec![TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body },
                parity: DetectorParity::from_terms([
                    DetectorTerm::Measurement(repeated),
                    DetectorTerm::LoopState(state),
                ]),
                coords: Some(bloq_circuit::DetectorCoords::from_slice(&[1.0, 2.0])),
            }],
            vec![TemplateRepeatState {
                body,
                state,
                initial: DetectorParity::from_measurements([seed]),
                next: DetectorParity::from_measurements([repeated]),
            }],
            Vec::new(),
            Vec::new(),
        );
        (template, body, seed, repeated)
    }

    #[test]
    fn template_flatten_expands_one_detector_per_iteration() {
        let (template, _, seed, repeated) = looped_template();

        let flat = template.flatten().unwrap();

        assert!(!flat.circuit.has_repeats());
        assert!(flat.repeat_states.is_empty());
        // Fresh ids 2 and 3 are iterations one and two; the final iteration
        // keeps `repeated`. Consecutive rounds compare, exactly like the
        // emitter's unrolled `DETECTOR rec[-2] rec[-1]` chain.
        let expected: Vec<TemplateDetectorParity> = vec![
            DetectorParity::from_measurements([seed, 2]),
            DetectorParity::from_measurements([2, 3]),
            DetectorParity::from_measurements([3, repeated]),
        ];
        assert_eq!(
            flat.detectors
                .iter()
                .map(|detector| detector.parity.clone())
                .collect::<Vec<_>>(),
            expected
        );
        for detector in &flat.detectors {
            assert_eq!(detector.scope, TemplateDetectorScope::TopLevel);
            assert_eq!(
                detector.coords.as_deref(),
                Some(&[1.0, 2.0][..]),
                "each iteration copy keeps the detector coords, like the emitter"
            );
        }
    }

    #[test]
    fn template_flatten_resolves_top_level_references() {
        let (mut template, _, _, repeated) = looped_template();
        // A top-level detector naming the repeated id means "its latest
        // occurrence" — the final iteration, which keeps the original id.
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([repeated]),
            coords: None,
        });
        // And one reading the loop state after the loop: the state's final
        // value is the last round's parity.
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(0))]),
            coords: None,
        });

        let flat = template.flatten().unwrap();

        let top = &flat.detectors[3..];
        assert_eq!(top[0].parity, DetectorParity::from_measurements([repeated]));
        assert_eq!(top[1].parity, DetectorParity::from_measurements([repeated]));
    }

    #[test]
    fn flatten_rejects_undefined_loop_state() {
        let (mut template, _, _, _) = looped_template();
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(99))]),
            coords: None,
        });

        assert!(matches!(
            template.flatten(),
            Err(FlattenError::UnresolvedLoopState(LoopStateId(99)))
        ));

        for restart in [false, true] {
            let mut bloq = Bloq::new();
            let mut node = BloqNode::from_members(Vec::new());
            let mut guard = crate::QuantumGuard::default();
            let parities = if restart {
                &mut guard.restart_parities
            } else {
                &mut guard.detector_parities
            };
            parities.push((
                0,
                DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(99))]),
            ));
            node.expect_quantum_mut().guards.push(guard);
            bloq.add_node(node);
            let original = bloq.to_binary();
            assert!(matches!(
                bloq.flatten(),
                Err(FlattenError::UnresolvedLoopState(LoopStateId(99)))
            ));
            assert_eq!(bloq.to_binary(), original);
        }
    }

    #[test]
    fn template_flatten_rejects_dangling_side_table_body() {
        let template = BloqTemplate::with_parts(
            CoordCircuit::new(),
            vec![TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body: BodyId(1) },
                parity: DetectorParity::default(),
                coords: None,
            }],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );

        assert!(matches!(
            template.flatten(),
            Err(FlattenError::Circuit(CircuitError::InvalidCircuitBody(body)))
                if body == BodyId(1)
        ));
    }

    #[test]
    fn template_flatten_rejects_same_body_peer_in_initial_state() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::new());
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 1,
            });
        let template = BloqTemplate::with_parts(
            circuit,
            vec![],
            vec![
                TemplateRepeatState {
                    body,
                    state: LoopStateId(0),
                    initial: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(1))]),
                    next: DetectorParity::default(),
                },
                TemplateRepeatState {
                    body,
                    state: LoopStateId(1),
                    initial: DetectorParity::default(),
                    next: DetectorParity::default(),
                },
            ],
            vec![],
            vec![],
        );

        assert!(matches!(
            template.flatten(),
            Err(FlattenError::UnresolvedLoopState(LoopStateId(1)))
        ));
    }

    #[test]
    fn template_flatten_drops_nested_state_when_outer_loop_exits() {
        let mut circuit = CoordCircuit::new();
        let inner = circuit.add_body(CircuitBody::new());
        let outer = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
            body: inner,
            repetitions: 1,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body: outer,
                repetitions: 1,
            });
        let nested = LoopStateId(1);
        let template = BloqTemplate::with_parts(
            circuit,
            vec![TemplateDetector {
                scope: TemplateDetectorScope::TopLevel,
                parity: DetectorParity::from_terms([DetectorTerm::LoopState(nested)]),
                coords: None,
            }],
            vec![TemplateRepeatState {
                body: inner,
                state: nested,
                initial: DetectorParity::default(),
                next: DetectorParity::default(),
            }],
            vec![],
            vec![],
        );

        assert!(matches!(
            template.flatten(),
            Err(FlattenError::UnresolvedLoopState(state)) if state == nested
        ));
    }

    #[test]
    fn bloq_flatten_shares_one_limit_across_distinct_templates() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: bloq_circuit::GateType::H,
            qubits: vec![ivec2(0, 0)],
        }]));
        circuit.push_repeat(body, 20);
        for shared in [false, true] {
            let mut bloq = Bloq::new();
            let first = bloq.add_template(BloqTemplate::new(circuit.clone()));
            let second = if shared {
                first
            } else {
                bloq.add_template(BloqTemplate::new(circuit.clone()))
            };
            let mut node = BloqNode::from_members(Vec::new());
            node.expect_quantum_mut().instances = vec![
                TemplateInstance::new(TemplateInstanceId(0), first, ivec2(0, 0)),
                TemplateInstance::new(TemplateInstanceId(1), second, ivec2(5, 0)),
            ];
            let node = bloq.add_node(node);
            let original = bloq.to_binary();
            let result = bloq.flatten_with_limits(FlattenLimits { max_work: 160 });
            if shared {
                result.expect("instances reuse one flattened template");
            } else {
                assert!(matches!(
                    result,
                    Err(FlattenError::Circuit(CircuitError::FlattenResourceLimit {
                        limit: 160,
                        ..
                    }))
                ));
                assert_eq!(bloq.to_binary(), original);
                bloq.flatten_with_limits(FlattenLimits { max_work: 320 })
                    .unwrap();
            }
            for instance in &bloq[node].expect_quantum().instances {
                assert!(!bloq.templates()[instance.template_id].circuit.has_repeats());
            }
        }
    }

    #[test]
    fn template_flatten_charges_growing_loop_state_parities() {
        let (mut template, body, seed, measurement) = looped_template();
        for op in template
            .circuit
            .body_mut(template.circuit.entry_body())
            .unwrap()
            .ops_mut()
        {
            if let Op::Repeat { repetitions, .. } = op {
                *repetitions = 64;
            }
        }
        let state = template.repeat_states[0].state;
        template.repeat_states[0].next = DetectorParity::from_terms([
            DetectorTerm::LoopState(state),
            DetectorTerm::Measurement(measurement),
        ]);
        assert_eq!(template.repeat_states[0].body, body);
        assert!(matches!(
            template.flatten_with_limits(FlattenLimits { max_work: 1000 }),
            Err(FlattenError::Circuit(
                CircuitError::FlattenResourceLimit { .. }
            ))
        ));
        let flat = template
            .flatten_with_limits(FlattenLimits { max_work: 100_000 })
            .unwrap();
        assert_eq!(flat.detectors.len(), 64);
        let last = &flat.detectors.last().unwrap().parity;
        assert_eq!(last.measurements().count(), 65);
        assert!(last.measurements().any(|measurement| measurement == seed));
    }

    #[test]
    fn bloq_flatten_repoints_instances_and_skips_flat_templates() {
        let (template, _, _, repeated) = looped_template();
        let mut flat_circuit = CoordCircuit::new();
        flat_circuit.measure(PauliBasis::Z, [ivec2(5, 0)]);

        let mut bloq = Bloq::new();
        let looped_id = bloq.add_template(template);
        let flat_id = bloq.add_template(BloqTemplate::new(flat_circuit));
        let bundle = bloq.add_detector_bundle(DetectorBundle::new(
            vec![looped_id],
            vec![BundleDetector {
                parity: DetectorParity::from_measurements([BundleMeasurement {
                    owner: 0,
                    measurement: repeated,
                }]),
                coords: None,
            }],
        ));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), looped_id, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), flat_id, ivec2(5, 0)),
        ];
        node.expect_quantum_mut()
            .detector_bundles
            .push(DetectorBundleUse {
                bundle,
                instances: vec![TemplateInstanceId(0)],
                offset: ivec2(0, 0),
            });
        let node_id = bloq.add_node(node);

        let original = bloq.to_text();
        assert!(
            bloq.flatten_with_limits(FlattenLimits { max_work: 1 })
                .is_err()
        );
        assert_eq!(
            bloq.to_text(),
            original,
            "resource failure leaves bundles and instances untouched"
        );

        bloq.flatten().unwrap();

        let instances = &bloq[node_id].expect_quantum().instances;
        let use_ = &bloq[node_id].expect_quantum().detector_bundles[0];
        assert_eq!(
            bloq.detector_bundles()
                .get(use_.bundle)
                .unwrap()
                .owner_templates(),
            &[instances[0].template_id]
        );
        assert_eq!(
            bloq.node_detectors(bloq[node_id].expect_quantum())
                .unwrap()
                .last()
                .unwrap()
                .measurements()
                .next()
                .unwrap()
                .measurement,
            repeated
        );
        for restored in [
            Bloq::from_text(&bloq.to_text()).unwrap(),
            Bloq::from_binary(&bloq.to_binary()).unwrap(),
        ] {
            assert_eq!(
                restored
                    .detector_bundles()
                    .get(restored[node_id].expect_quantum().detector_bundles[0].bundle)
                    .unwrap()
                    .owner_templates(),
                &[instances[0].template_id]
            );
        }
        let flattened_id = instances[0].template_id;
        assert_ne!(flattened_id, looped_id, "looped template re-pointed");
        assert_eq!(instances[1].template_id, flat_id, "flat template untouched");
        let flattened = bloq.templates().get(flattened_id).unwrap();
        assert!(!flattened.circuit.has_repeats());
        assert!(flattened.repeat_states.is_empty());
        assert_eq!(flattened.detectors.len(), 3);

        let snapshot = bloq.clone();
        bloq.flatten().unwrap();
        assert!(std::ptr::eq(bloq.graph(), snapshot.graph()));
    }
}
