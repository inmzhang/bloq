use bloq_circuit::{
    BodyId, CircuitBody, ConditionalCorrection, CoordCircuit, FlattenEvent, GateType, MeasRecord,
    NoiseModel, Op, PauliMap, checked_translate_coordinate,
};
use glam::IVec2;
use thiserror::Error;

use crate::{
    BloqNode, BloqTemplatePool, CoordinateOverflowError, FxMap, InstanceMeasurement, TemplateId,
    TemplateInstance, TemplateInstanceId,
};

/// Why a template-backed node's instances could not be merged into one circuit.
///
/// Multi-instance emission and [`Bloq::validate`](crate::Bloq::validate) run
/// the same merge. Singleton emission stops after the same structural preflight
/// and translation because merging would change its temporal structure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NodeTemplateInstanceMergeError {
    /// A detector bundle use is malformed.
    #[error("{0}")]
    DetectorBundle(#[from] crate::DetectorBundleError),
    /// Boolean analysis exceeded its configured budget.
    #[error("resource limit: {0}")]
    BooleanResource(#[from] bloq_utils::boolean::BooleanResourceError),
    /// Conditional registrations require selector inputs.
    #[error("conditional component membership requires selector values")]
    MembershipSelectionRequired,
    /// A required selector input is unavailable.
    #[error("component membership input {0} is not available")]
    MissingMembershipInput(u32),
    /// Registration tables violate their structural contract.
    #[error("invalid component membership: {0}")]
    InvalidMembership(String),
    /// An instance names an absent template.
    #[error("template {0:?} does not exist")]
    UnknownTemplate(TemplateId),
    /// A circuit names an absent body.
    #[error("template circuit references invalid body {0:?}")]
    InvalidBody(BodyId),
    /// Circuit bodies recurse.
    #[error("template circuit contains recursive body {0:?}")]
    CyclicBody(BodyId),
    /// MPP products and measurement ids have different lengths.
    #[error("template circuit MPP has {products} products but {measurements} measurement ids")]
    MppMeasurementCountMismatch {
        /// Product count.
        products: usize,
        /// Measurement-id count.
        measurements: usize,
    },
    /// Measure targets and outputs have different lengths.
    #[error("template circuit Measure has {qubits} qubits but {measurements} measurement outputs")]
    MeasureOutputCountMismatch {
        /// Qubit count.
        qubits: usize,
        /// Measurement-id count.
        measurements: usize,
    },
    /// An MPP operation has no products.
    #[error("template circuit MPP requires at least one product")]
    EmptyMpp,
    /// An MPP product has no Pauli terms.
    #[error("template circuit contains an empty MPP product")]
    EmptyPauliProduct,
    /// An operation produces an unregistered measurement.
    #[error("template circuit operation produces unregistered measurement id {0}")]
    UnregisteredMeasurementOutput(u32),
    /// Several operations produce the same measurement id.
    #[error("template circuit measurement id {0} is produced by more than one operation")]
    DuplicateMeasurementOutput(u32),
    /// A measurement registry coordinate disagrees with its operation.
    #[error(
        "template circuit measurement {measurement} is registered at {registered:?} but its operation produces it at {produced:?}"
    )]
    MeasurementOutputCoordinateMismatch {
        /// Measurement id.
        measurement: u32,
        /// Registry coordinate.
        registered: IVec2,
        /// Operation coordinate.
        produced: IVec2,
    },
    /// A two-qubit gate has an odd target count.
    #[error("template circuit two-qubit gate {gate} has odd target count {targets}")]
    OddTwoQubitTargetCount {
        /// Gate kind.
        gate: GateType,
        /// Target count.
        targets: usize,
    },
    /// A two-qubit gate pair repeats one endpoint.
    #[error("template circuit two-qubit gate {gate} repeats target {qubit:?} within one pair")]
    RepeatedTwoQubitGateTarget {
        /// Gate kind.
        gate: GateType,
        /// Repeated endpoint.
        qubit: IVec2,
    },
    /// A two-qubit noise operation has an odd target count.
    #[error("template circuit DEPOLARIZE2 has odd target count {targets}")]
    OddDepolarize2TargetCount {
        /// Target count.
        targets: usize,
    },
    /// A two-qubit noise pair repeats one endpoint.
    #[error("template circuit DEPOLARIZE2 repeats target {0:?} within one pair")]
    RepeatedDepolarize2Target(IVec2),
    /// A probability is non-finite or outside `[0, 1]`.
    #[error("template circuit {instruction} probability must be finite and in [0, 1]")]
    InvalidProbability {
        /// Instruction with the invalid probability.
        instruction: &'static str,
    },
    /// Coordinate translation overflowed.
    #[error("{0}")]
    CoordinateOverflow(#[from] CoordinateOverflowError),
    /// Merged measurement ids exhausted `u32`.
    #[error("template instance measurement id space overflowed")]
    MeasurementIdOverflow,
    /// Circuit-body storage or the merged body-id space exhausted capacity.
    #[error("resource limit: instance merge body-id or storage size overflowed")]
    BodyCountOverflow,
    /// Repeat expansion for idle noise failed.
    #[error("{context}: {0}", context = if .0.is_resource_limited() {
        "resource limit expanding repeat moments before idle noise"
    } else {
        "expanding repeat moments before idle noise"
    })]
    NoiseRepeatExpansion(#[from] crate::FlattenError),
    /// Instance circuits have different tick counts.
    #[error("template instance circuits have mismatched tick segment counts")]
    TickSegmentCountMismatch,
    /// A merged segment mixes repeat and non-repeat shapes.
    #[error("template instance circuits mix repeat and non-repeat ops in one tick segment")]
    RepeatShapeMismatch,
    /// Aligned repeats have different counts.
    #[error("template instance repeat ops have different repetition counts")]
    RepeatCountMismatch,
    /// One body participates in several merge tuples.
    #[error(
        "template instance circuit {circuit_index} body {body:?} participates in multiple aligned body tuples"
    )]
    BodyMergeConflict {
        /// Source circuit index.
        circuit_index: usize,
        /// Conflicting body.
        body: BodyId,
    },
    /// Instance circuits use one qubit incompatibly in a segment.
    #[error("template instance circuits overlap qubit {0:?} in one tick segment")]
    QubitConflict(IVec2),
    /// Conditional feedforward reads a measurement too early.
    #[error("ConditionalPauli control references measurement {0} not produced before it")]
    ControlReferencesUnknownMeasurement(u32),
}

impl NodeTemplateInstanceMergeError {
    /// Whether checking or materializing the instances exhausted a resource
    /// allowance, rather than proving their circuit structure invalid.
    #[must_use]
    pub fn is_resource_limited(&self) -> bool {
        match self {
            Self::BooleanResource(_) | Self::BodyCountOverflow => true,
            Self::NoiseRepeatExpansion(error) => error.is_resource_limited(),
            _ => false,
        }
    }
}

/// A node's merged circuit and instance-space maps.
///
/// `measurements` translates each [`InstanceMeasurement`]
/// to its merged measurement id, and `bodies` translates each
/// `(instance, template body)` to its merged body id. Side tables (detectors,
/// observable recipes) named in instance space resolve against these.
#[derive(Debug, Clone)]
pub struct NodeEmissionPlan {
    /// Merged circuit.
    pub circuit: CoordCircuit,
    /// Annotated source templates after repeat expansion, when idle noise
    /// crosses a repeat boundary. Use [`Self::templates`] when resolving this
    /// plan's detector tables and instance-local measurement ids.
    pub normalized_templates: Option<BloqTemplatePool>,
    /// Instance-space measurement to merged-local measurement id.
    pub measurements: FxMap<InstanceMeasurement, u32>,
    /// Instance/template body to merged body id.
    pub bodies: FxMap<(TemplateInstanceId, BodyId), BodyId>,
}

impl NodeEmissionPlan {
    /// Source templates matching this plan's circuit and id maps. Most plans
    /// use `original`; partial repeat moments carry an expanded annotated pool.
    pub fn templates<'a>(&'a self, original: &'a BloqTemplatePool) -> &'a BloqTemplatePool {
        self.normalized_templates.as_ref().unwrap_or(original)
    }

    /// The plan's [`measurements`](Self::measurements) inverted and grouped by
    /// merged-local id: each `(local, sources)` pair collects every
    /// [`InstanceMeasurement`] that resolves to the same merged record `local`.
    ///
    /// The order is canonical and load-bearing — both backends mint their global
    /// records in it: groups are yielded ascending by `local`, and within a group
    /// the sources are sorted by `(instance, measurement)`. `bloq_vm` mints
    /// one global record per group in this order; `bloq_stim` folds each group to
    /// its minimum backend id. Both folds are order-independent within a group, so
    /// the intra-group order is fixed only to make the grouping canonical.
    #[must_use]
    pub fn grouped_measurements(&self) -> Vec<(u32, Vec<InstanceMeasurement>)> {
        let mut flat: Vec<(u32, InstanceMeasurement)> = self
            .measurements
            .iter()
            .map(|(&measurement, &local)| (local, measurement))
            .collect();
        flat.sort_unstable_by_key(|&(local, measurement)| {
            (local, measurement.instance.0, measurement.measurement)
        });
        let mut groups: Vec<(u32, Vec<InstanceMeasurement>)> = Vec::new();
        for (local, measurement) in flat {
            match groups.last_mut() {
                Some((last, sources)) if *last == local => sources.push(measurement),
                _ => groups.push((local, vec![measurement])),
            }
        }
        groups
    }
}

/// Options applied after a node's instances have been translated and, when
/// needed, merged.
///
/// Validation defaults to 4,000,000 predicate nodes and 128,000,000 work steps.
/// Whole-program audits have a larger work allowance than standalone
/// [`BooleanLimits`](bloq_utils::boolean::BooleanLimits) queries.
#[derive(Debug, Clone, Copy)]
pub struct InstantiationOptions<'a> {
    noise: Option<&'a NoiseModel>,
    boolean_limits: bloq_utils::boolean::BooleanLimits,
}

const DEFAULT_VALIDATION_BOOLEAN_LIMITS: bloq_utils::boolean::BooleanLimits =
    bloq_utils::boolean::BooleanLimits {
        max_steps: 128_000_000,
        ..bloq_utils::boolean::BooleanLimits::DEFAULT
    };

impl Default for InstantiationOptions<'_> {
    fn default() -> Self {
        Self {
            noise: None,
            boolean_limits: DEFAULT_VALIDATION_BOOLEAN_LIMITS,
        }
    }
}

impl<'a> InstantiationOptions<'a> {
    /// Materialize `noise` on the merged circuit.
    #[must_use]
    pub const fn noisy(noise: &'a NoiseModel) -> Self {
        Self {
            noise: Some(noise),
            boolean_limits: DEFAULT_VALIDATION_BOOLEAN_LIMITS,
        }
    }

    /// Bound cumulative validation work and live predicate decision nodes.
    /// Both supplied allowances replace the defaults without adjustment.
    #[must_use]
    pub const fn with_boolean_limits(mut self, limits: bloq_utils::boolean::BooleanLimits) -> Self {
        self.boolean_limits = limits;
        self
    }

    /// Configured Boolean-analysis limits.
    #[must_use]
    pub const fn boolean_limits(self) -> bloq_utils::boolean::BooleanLimits {
        self.boolean_limits
    }

    pub(crate) const fn is_noiseless(self) -> bool {
        self.noise.is_none()
    }
}

impl BloqNode {
    /// Resolve the component's conditional registrations without executing it.
    /// Unselected instances produce no gates, records, detectors, or restarts.
    /// The returned node retains the selected instances' identities.
    ///
    /// # Errors
    ///
    /// Returns [`NodeTemplateInstanceMergeError`] for missing inputs or an
    /// invalid, incomplete, or duplicate membership registration.
    pub fn select_quantum_members(
        &self,
        mut input: impl FnMut(u32) -> Option<bool>,
    ) -> Result<Self, NodeTemplateInstanceMergeError> {
        if let Some(quantum) = self.try_quantum() {
            quantum.check_side_table_counts()?;
        }
        let mut selected = self.clone();
        if selected
            .try_quantum()
            .is_none_or(|quantum| quantum.guards.is_empty())
        {
            return Ok(selected);
        }
        let quantum = selected.expect_quantum_mut();
        let available_instances: crate::FxSet<_> = quantum
            .instances
            .iter()
            .map(|instance| instance.id)
            .collect();
        let mut instances = crate::FxMap::default();
        let mut detectors = crate::FxMap::default();
        let mut detector_bundles = crate::FxMap::default();
        let mut restarts = crate::FxMap::default();
        let mut inputs = crate::FxMap::default();
        for guard in &quantum.guards {
            if guard.instances.is_empty()
                && guard.detectors.is_empty()
                && guard.detector_bundles.is_empty()
                && guard.restarts.is_empty()
                && guard.detector_parities.is_empty()
                && guard.restart_parities.is_empty()
            {
                return Err(NodeTemplateInstanceMergeError::InvalidMembership(
                    "empty registration".into(),
                ));
            }
            let enabled = *inputs
                .entry(guard.input)
                .or_insert_with(|| input(guard.input));
            let enabled = enabled.ok_or(NodeTemplateInstanceMergeError::MissingMembershipInput(
                guard.input,
            ))?;
            for &id in &guard.instances {
                if !available_instances.contains(&id) || instances.insert(id, enabled).is_some() {
                    return Err(NodeTemplateInstanceMergeError::InvalidMembership(format!(
                        "instance i{} is missing or registered more than once",
                        id.0,
                    )));
                }
            }
            for (indices, count, selected) in [
                (&guard.detectors, quantum.detectors.len(), &mut detectors),
                (
                    &guard.detector_bundles,
                    quantum.detector_bundles.len(),
                    &mut detector_bundles,
                ),
                (&guard.restarts, quantum.restarts.len(), &mut restarts),
            ] {
                for &index in indices {
                    if index as usize >= count || selected.insert(index, enabled).is_some() {
                        return Err(NodeTemplateInstanceMergeError::InvalidMembership(format!(
                            "side-table index {index} is missing or registered more than once",
                        )));
                    }
                }
            }
            for (index, parity) in &guard.detector_parities {
                let detector = quantum.detectors.get_mut(*index as usize).ok_or_else(|| {
                    NodeTemplateInstanceMergeError::InvalidMembership(format!(
                        "unknown detector d{index}"
                    ))
                })?;
                if enabled {
                    detector.parity.xor_assign(parity);
                }
            }
            for (index, parity) in &guard.restart_parities {
                let restart = quantum.restarts.get_mut(*index as usize).ok_or_else(|| {
                    NodeTemplateInstanceMergeError::InvalidMembership(format!(
                        "unknown restart r{index}"
                    ))
                })?;
                if enabled {
                    restart.parity.xor_assign(parity);
                }
            }
        }
        quantum
            .instances
            .retain(|instance| instances.get(&instance.id).copied().unwrap_or(true));
        let mut index = 0u32;
        quantum.detectors.retain(|_| {
            let enabled = detectors.get(&index).copied().unwrap_or(true);
            index += 1;
            enabled
        });
        let mut index = 0u32;
        quantum.detector_bundles.retain(|_| {
            let enabled = detector_bundles.get(&index).copied().unwrap_or(true);
            index += 1;
            enabled
        });
        let mut index = 0u32;
        quantum.restarts.retain(|_| {
            let enabled = restarts.get(&index).copied().unwrap_or(true);
            index += 1;
            enabled
        });
        quantum.guards.clear();
        Ok(selected)
    }

    /// The node's merged circuit — its instances translated by offset and
    /// merged moment-by-moment.
    ///
    /// This is [`Self::emission_plan`] keeping only the circuit; take the plan
    /// when the instance-space id maps matter, or
    /// [`Self::emission_plan_with_options`] to apply noise.
    ///
    /// # Errors
    ///
    /// Returns [`NodeTemplateInstanceMergeError`] when the instances do not
    /// merge (mismatched tick/repeat structure, incompatible qubit overlap, a
    /// missing template).
    pub fn instantiate_circuit(
        &self,
        templates: &BloqTemplatePool,
    ) -> Result<CoordCircuit, NodeTemplateInstanceMergeError> {
        Ok(self.emission_plan(templates)?.circuit)
    }

    /// Instantiate and merge the node's template instances into one circuit,
    /// returning the circuit together with the instance-space id maps
    /// ([`NodeEmissionPlan`]). A classical or region node has no instances and
    /// yields an empty plan.
    ///
    /// # Errors
    ///
    /// Returns [`NodeTemplateInstanceMergeError`] when the instances do not
    /// merge (mismatched tick/repeat structure, incompatible qubit overlap, a
    /// missing template).
    pub fn emission_plan(
        &self,
        templates: &BloqTemplatePool,
    ) -> Result<NodeEmissionPlan, NodeTemplateInstanceMergeError> {
        self.emission_plan_with_options(templates, InstantiationOptions::default())
    }

    /// [`Self::emission_plan`] with options applied once after instantiation.
    ///
    /// # Errors
    ///
    /// Returns [`NodeTemplateInstanceMergeError`] for unresolved membership,
    /// invalid templates or noise, coordinate overflow, or an invalid merge.
    ///
    /// # Panics
    ///
    /// Panics only if an internally generated merged circuit body is missing.
    pub fn emission_plan_with_options(
        &self,
        templates: &BloqTemplatePool,
        options: InstantiationOptions<'_>,
    ) -> Result<NodeEmissionPlan, NodeTemplateInstanceMergeError> {
        // Classical / region nodes carry no template instances, so they emit an
        // empty circuit (like `node_qubits`/`node_measurement_count` skip them).
        let Some(quantum) = self.try_quantum() else {
            return Ok(NodeEmissionPlan {
                circuit: CoordCircuit::new(),
                normalized_templates: None,
                measurements: FxMap::default(),
                bodies: FxMap::default(),
            });
        };
        let instances = &quantum.instances;
        if !quantum.guards.is_empty() {
            return Err(NodeTemplateInstanceMergeError::MembershipSelectionRequired);
        }
        if let Some(noise) = options.noise {
            validate_noise_model(noise)?;
        }

        let mut next_measurement = 0u32;
        let capacity = estimate_node_instance_capacity(self, templates)?;
        let mut measurements =
            FxMap::with_capacity_and_hasher(capacity.measurements, Default::default());
        let mut bodies = FxMap::with_capacity_and_hasher(capacity.bodies, Default::default());
        let mut instance_indices =
            FxMap::with_capacity_and_hasher(instances.len(), Default::default());
        let mut circuits = Vec::with_capacity(instances.len());
        // Trigger each template's shared circuit-analysis cache once per node;
        // nodes may carry many instances of the same template (IR-03).
        let mut preflighted = crate::FxSet::default();
        for (index, instance) in instances.iter().enumerate() {
            instance_indices.insert(instance.id, index);
            let circuit = instantiate_instance(
                instance,
                templates,
                &mut next_measurement,
                &mut measurements,
                &mut bodies,
                &mut preflighted,
            )?;
            circuits.push(circuit);
        }
        let ideal_qubits = ideal_instance_qubits(instances, &circuits);
        let circuit = if circuits.len() == 1 {
            // A singleton retains source timing; multi-instance SEM-MERGE
            // introduces its own explicit tick boundaries around repeats.
            circuits.pop().expect("length checked above")
        } else {
            let merged = merge_instantiated_circuits_with_body_aliases(&circuits)?;
            for ((instance, _), body) in &mut bodies {
                if let Some(index) = instance_indices.get(instance).copied()
                    && let Some(&merged) = merged.body_aliases.get(&(index, *body))
                {
                    *body = merged;
                }
            }
            for measurement in measurements.values_mut() {
                *measurement = canonical_measurement(*measurement, &merged.measurement_aliases);
            }
            merged.circuit
        };
        let mut plan = NodeEmissionPlan {
            circuit,
            normalized_templates: None,
            measurements,
            bodies,
        };
        // Check the original SEM-MERGE before noise can normalize its shape.
        // The actual merged stream also owns any synthesized tick boundaries.
        if options
            .noise
            .is_some_and(|noise| noise.requires_repeat_flattening(&plan.circuit))
        {
            flatten_emission_plan(&mut plan, instances, templates)?;
        }
        plan.circuit = apply_instantiation_options(plan.circuit, options, &ideal_qubits)?;
        Ok(plan)
    }
}

/// Expand the merged stream and its source annotations without merging again.
/// Repeat-isolation ticks and idle members already belong to the merged plan.
fn flatten_emission_plan(
    plan: &mut NodeEmissionPlan,
    instances: &[TemplateInstance],
    templates: &BloqTemplatePool,
) -> Result<(), NodeTemplateInstanceMergeError> {
    let trace = plan.circuit.flatten().map_err(crate::FlattenError::from)?;
    let merged_occurrences = measurement_occurrences(&trace);
    let original_measurements = plan.measurements.clone();
    let mut normalized = templates.clone();
    let mut source_occurrences = FxMap::default();
    plan.bodies.clear();
    for instance in instances {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            source_occurrences.entry(instance.template_id)
        {
            let (flat, trace) = templates[instance.template_id].flatten_with_trace()?;
            *normalized
                .make_mut(instance.template_id)
                .expect("source template exists") = flat;
            entry.insert(measurement_occurrences(&trace));
        }
        for (&original, emitted) in &source_occurrences[&instance.template_id] {
            let merged = original_measurements[&InstanceMeasurement {
                instance: instance.id,
                measurement: original,
            }];
            let merged_emitted = &merged_occurrences[&merged];
            assert_eq!(
                emitted.len(),
                merged_emitted.len(),
                "aligned instances preserve measurement occurrence counts"
            );
            // Each aliased source gets the whole sequence; an occurrence must
            // not be consumed by a different instance sharing this measurement.
            for (&measurement, &local) in emitted.iter().zip(merged_emitted) {
                plan.measurements.insert(
                    InstanceMeasurement {
                        instance: instance.id,
                        measurement,
                    },
                    local,
                );
            }
        }
        plan.bodies.insert(
            (
                instance.id,
                normalized[instance.template_id].circuit.entry_body(),
            ),
            plan.circuit.entry_body(),
        );
    }
    plan.normalized_templates = Some(normalized);
    Ok(())
}

fn measurement_occurrences(trace: &[FlattenEvent]) -> FxMap<u32, Vec<u32>> {
    let mut occurrences = FxMap::<_, Vec<_>>::default();
    for &event in trace {
        if let FlattenEvent::Measurement { original, emitted } = event {
            occurrences.entry(original).or_default().push(emitted);
        }
    }
    occurrences
}

/// Materialize the requested noise on a node's instantiated circuit.
fn apply_instantiation_options(
    circuit: CoordCircuit,
    options: InstantiationOptions<'_>,
    ideal_qubits: &[IVec2],
) -> Result<CoordCircuit, NodeTemplateInstanceMergeError> {
    let Some(noise) = options.noise else {
        return Ok(circuit);
    };
    validate_noise_model(noise)?;
    noise
        .noisy_circuit_excluding(&circuit, ideal_qubits)
        .map_err(crate::FlattenError::from)
        .map_err(NodeTemplateInstanceMergeError::from)
}

fn ideal_instance_qubits(instances: &[TemplateInstance], circuits: &[CoordCircuit]) -> Vec<IVec2> {
    // Spec rule SEM-SPATIAL-PORT: source ownership wins on shared qubits.
    let mut derived = crate::FxSet::default();
    let mut source = crate::FxSet::default();
    for (instance, circuit) in instances.iter().zip(circuits) {
        let target = if instance.provenance.is_ideal_boundary() {
            &mut derived
        } else {
            &mut source
        };
        target.extend(circuit.qubits());
    }
    derived.difference(&source).copied().collect()
}

/// Stored circuit work, excluding repeated execution. Saturation keeps this
/// usable as a conservative preallocation charge at every instantiation gate.
pub(crate) fn circuit_work(circuit: &CoordCircuit) -> usize {
    let mut work = circuit
        .body_count()
        .saturating_add(circuit.meas_registry().records().len());
    for index in 0..circuit.body_count() {
        let ops = circuit
            .body(BodyId(index as u32))
            .expect("allocated body")
            .ops();
        work = work.saturating_add(ops.len());
        for op in ops {
            let targets = match op {
                Op::Measure {
                    qubits,
                    measurements,
                    ..
                } => qubits.len().saturating_add(measurements.len()),
                Op::Gate { qubits, .. }
                | Op::Depolarize1 { qubits, .. }
                | Op::Depolarize2 { qubits, .. }
                | Op::PauliError { qubits, .. } => qubits.len(),
                Op::MPP {
                    products,
                    measurements,
                } => products
                    .iter()
                    .fold(products.len(), |total, product| {
                        total.saturating_add(product.len())
                    })
                    .saturating_add(measurements.len()),
                Op::ConditionalPauli(corrections) => corrections.len(),
                Op::Tick | Op::Repeat { .. } => 0,
            };
            work = work.saturating_add(targets);
        }
    }
    work
}

/// Summed `with_capacity` hints for merging a set of circuits.
#[derive(Default)]
struct CircuitCounts {
    measurements: usize,
    bodies: usize,
}

impl CircuitCounts {
    fn add_counts(
        &mut self,
        measurements: usize,
        bodies: usize,
    ) -> Result<(), NodeTemplateInstanceMergeError> {
        let measurements = self
            .measurements
            .checked_add(measurements)
            .ok_or(NodeTemplateInstanceMergeError::MeasurementIdOverflow)?;
        u32::try_from(measurements)
            .map_err(|_| NodeTemplateInstanceMergeError::MeasurementIdOverflow)?;
        let bodies = self
            .bodies
            .checked_add(bodies)
            .ok_or(NodeTemplateInstanceMergeError::BodyCountOverflow)?;
        self.measurements = measurements;
        self.bodies = bodies;
        Ok(())
    }
}

fn count_circuits<'a>(
    circuits: impl IntoIterator<Item = &'a CoordCircuit>,
) -> Result<CircuitCounts, NodeTemplateInstanceMergeError> {
    let mut counts = CircuitCounts::default();
    for circuit in circuits {
        counts.add_counts(
            circuit.meas_registry().records().len(),
            circuit.body_count(),
        )?;
    }
    Ok(counts)
}

fn estimate_node_instance_capacity(
    node: &BloqNode,
    templates: &BloqTemplatePool,
) -> Result<CircuitCounts, NodeTemplateInstanceMergeError> {
    let mut counts = CircuitCounts::default();
    for instance in &node.expect_quantum().instances {
        let circuit = &templates
            .get(instance.template_id)
            .ok_or(NodeTemplateInstanceMergeError::UnknownTemplate(
                instance.template_id,
            ))?
            .circuit;
        counts.add_counts(
            circuit.meas_registry().records().len(),
            circuit.body_count(),
        )?;
    }
    Ok(counts)
}

fn instantiate_instance(
    instance: &TemplateInstance,
    templates: &BloqTemplatePool,
    next_measurement: &mut u32,
    measurements: &mut FxMap<InstanceMeasurement, u32>,
    bodies: &mut FxMap<(TemplateInstanceId, BodyId), BodyId>,
    preflighted: &mut crate::FxSet<TemplateId>,
) -> Result<CoordCircuit, NodeTemplateInstanceMergeError> {
    let template = templates.get(instance.template_id).ok_or(
        NodeTemplateInstanceMergeError::UnknownTemplate(instance.template_id),
    )?;
    // Validate operation shape and measurement ownership once per template;
    // `translate_circuit` relies on this having run and does not re-check ops.
    if preflighted.insert(instance.template_id) {
        template.circuit_analysis()?;
    }
    let mut local_bodies =
        FxMap::with_capacity_and_hasher(template.circuit.body_count(), Default::default());
    let circuit = translate_circuit(
        &template.circuit,
        instance.offset,
        instance.id,
        next_measurement,
        measurements,
        &mut local_bodies,
    )?;
    bodies.extend(
        local_bodies
            .into_iter()
            .map(|(template_body, instance_body)| ((instance.id, template_body), instance_body)),
    );
    Ok(circuit)
}

fn translate_circuit(
    source: &CoordCircuit,
    offset: IVec2,
    instance: TemplateInstanceId,
    next_measurement: &mut u32,
    measurements: &mut FxMap<InstanceMeasurement, u32>,
    body_map: &mut FxMap<BodyId, BodyId>,
) -> Result<CoordCircuit, NodeTemplateInstanceMergeError> {
    // Op shape and measurement ownership were validated once per template by
    // `instantiate_instance` (this fn's sole caller); only per-instance coord
    // translation remains.
    for coordinate in source.qubits() {
        checked_translate_instance_coord(coordinate, offset)?;
    }
    let mut translator = InstanceTranslator {
        source,
        offset,
        instance,
        next_measurement,
        measurements,
        destination: CoordCircuit::new(),
        body_map,
    };
    translator.translate_body(source.entry_body())?;
    Ok(translator.destination)
}

#[derive(Debug)]
pub(crate) struct RepeatMeasurementEvent {
    pub(crate) body: BodyId,
    pub(crate) before: usize,
    pub(crate) after: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum MeasurementSet {
    Dense(Vec<bool>),
    Sparse(crate::FxSet<u32>),
}

impl MeasurementSet {
    fn for_circuit(circuit: &CoordCircuit) -> Self {
        let records = circuit.meas_registry().records();
        // Sorted unique non-negative ids are dense iff their maximum is len - 1.
        if records
            .last()
            .is_none_or(|record| record.id as usize == records.len() - 1)
        {
            Self::Dense(vec![false; records.len()])
        } else {
            Self::Sparse(crate::FxSet::with_capacity_and_hasher(
                records.len(),
                Default::default(),
            ))
        }
    }

    fn empty_like(&self) -> Self {
        match self {
            Self::Dense(ids) => Self::Dense(vec![false; ids.len()]),
            Self::Sparse(ids) => Self::Sparse(crate::FxSet::with_capacity_and_hasher(
                ids.len(),
                Default::default(),
            )),
        }
    }

    fn insert(&mut self, id: u32) -> bool {
        match self {
            Self::Dense(ids) => {
                let seen = &mut ids[id as usize];
                let inserted = !*seen;
                *seen = true;
                inserted
            }
            Self::Sparse(ids) => ids.insert(id),
        }
    }

    pub(crate) fn contains(&self, id: u32) -> bool {
        match self {
            Self::Dense(ids) => ids.get(id as usize).copied().unwrap_or(false),
            Self::Sparse(ids) => ids.contains(&id),
        }
    }
}

#[derive(Debug)]
pub(crate) struct TemplateCircuitAnalysis {
    pub(crate) reachable_bodies: crate::FxSet<BodyId>,
    pub(crate) body_order: Vec<BodyId>,
    pub(crate) produced_measurements: MeasurementSet,
    pub(crate) measurement_order: FxMap<u32, usize>,
    pub(crate) repeat_events: Vec<RepeatMeasurementEvent>,
}

fn preflight_circuit(
    source: &CoordCircuit,
) -> Result<TemplateCircuitAnalysis, NodeTemplateInstanceMergeError> {
    let mut structural_measurements = MeasurementSet::for_circuit(source);
    let mut measurement_controls = Vec::new();
    let mut reachable_bodies = crate::FxSet::default();
    let mut body_order = Vec::new();
    let mut active = crate::FxSet::default();
    let mut pending = vec![(source.entry_body(), 0usize)];
    active.insert(source.entry_body());
    while let Some((body, next)) = pending.last_mut() {
        let ops = source
            .body(*body)
            .ok_or(NodeTemplateInstanceMergeError::InvalidBody(*body))?
            .ops();
        let Some(op) = ops.get(*next) else {
            let body = *body;
            pending.pop();
            active.remove(&body);
            reachable_bodies.insert(body);
            body_order.push(body);
            continue;
        };
        *next += 1;
        validate_source_op(source, op)?;
        register_source_measurement_outputs(op, &mut structural_measurements)?;
        if let Op::ConditionalPauli(corrections) = op {
            measurement_controls.extend(corrections.iter().map(|correction| correction.control));
        }
        if let Op::Repeat { body, .. } = op
            && !reachable_bodies.contains(body)
        {
            if !active.insert(*body) {
                return Err(NodeTemplateInstanceMergeError::CyclicBody(*body));
            }
            pending.push((*body, 0));
        }
    }
    for measurement in measurement_controls {
        if !structural_measurements.contains(measurement) {
            return Err(
                NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(measurement),
            );
        }
    }
    // Record every body structurally reached: the entry plus
    // every transitive `REPEAT` target. The entry is never a valid repeat
    // target, so drop it to leave exactly the bodies a `REPEAT`/detector scope
    // may name (WF template-body reference checks).
    reachable_bodies.remove(&source.entry_body());

    let mut produced_measurements = structural_measurements.empty_like();
    let mut measurement_order = FxMap::default();
    let mut repeat_events = Vec::new();
    let mut visited = crate::FxSet::default();
    visited.insert(source.entry_body());
    let mut pending = vec![(source.entry_body(), 0usize, None)];
    while let Some((body, next, before)) = pending.last_mut() {
        let ops = source
            .body(*body)
            .expect("structural preflight checked bodies")
            .ops();
        let Some(op) = ops.get(*next) else {
            if let Some(before) = *before {
                repeat_events.push(RepeatMeasurementEvent {
                    body: *body,
                    before,
                    after: measurement_order.len(),
                });
            }
            pending.pop();
            continue;
        };
        *next += 1;
        match op {
            Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                for &measurement in measurements {
                    produced_measurements.insert(measurement);
                    measurement_order.insert(measurement, measurement_order.len());
                }
            }
            Op::ConditionalPauli(corrections) => {
                for correction in corrections {
                    if !produced_measurements.contains(correction.control) {
                        return Err(
                            NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(
                                correction.control,
                            ),
                        );
                    }
                }
            }
            // Measurement availability only grows, and every body emits the
            // same ids. Its first execution is therefore the strongest check;
            // later executions add no records and cannot invalidate a reference.
            Op::Repeat { body, repetitions } if *repetitions > 0 && visited.insert(*body) => {
                pending.push((*body, 0, Some(measurement_order.len())));
            }
            _ => {}
        }
    }
    Ok(TemplateCircuitAnalysis {
        reachable_bodies,
        body_order,
        produced_measurements,
        measurement_order,
        repeat_events,
    })
}

fn register_source_measurement_outputs(
    op: &Op,
    produced: &mut MeasurementSet,
) -> Result<(), NodeTemplateInstanceMergeError> {
    let measurements = match op {
        Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => measurements,
        Op::Gate { .. }
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. }
        | Op::Tick
        | Op::Repeat { .. }
        | Op::ConditionalPauli(_) => return Ok(()),
    };
    for &measurement in measurements {
        if !produced.insert(measurement) {
            return Err(NodeTemplateInstanceMergeError::DuplicateMeasurementOutput(
                measurement,
            ));
        }
    }
    Ok(())
}

fn validate_source_op(
    source: &CoordCircuit,
    op: &Op,
) -> Result<(), NodeTemplateInstanceMergeError> {
    validate_circuit_op(op)?;
    match op {
        Op::Measure {
            qubits,
            measurements,
            ..
        } => {
            for (&qubit, &measurement) in qubits.iter().zip(measurements) {
                validate_measurement_output(source, measurement, qubit)?;
            }
            Ok(())
        }
        Op::MPP {
            products,
            measurements,
        } => {
            for (product, &measurement) in products.iter().zip(measurements) {
                validate_measurement_output(source, measurement, mpp_record_coord(product)?)?;
            }
            Ok(())
        }
        Op::Gate { .. }
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. }
        | Op::Tick
        | Op::Repeat { .. }
        | Op::ConditionalPauli(_) => Ok(()),
    }
}

/// Validate one circuit operation's self-contained shape and probabilities.
/// Measurement-registry bindings require a full [`validate_template_circuit`]
/// call instead.
///
/// # Errors
///
/// [`NodeTemplateInstanceMergeError`] if an operation has invalid arity,
/// repeated two-qubit endpoints, an empty MPP or product, or an invalid
/// probability.
fn validate_circuit_op(op: &Op) -> Result<(), NodeTemplateInstanceMergeError> {
    match op {
        Op::Gate { gate, qubits } if gate.is_two_qubit_gate() => {
            if !qubits.len().is_multiple_of(2) {
                return Err(NodeTemplateInstanceMergeError::OddTwoQubitTargetCount {
                    gate: *gate,
                    targets: qubits.len(),
                });
            }
            if let Some(pair) = qubits
                .as_chunks::<2>()
                .0
                .iter()
                .find(|pair| pair[0] == pair[1])
            {
                return Err(NodeTemplateInstanceMergeError::RepeatedTwoQubitGateTarget {
                    gate: *gate,
                    qubit: pair[0],
                });
            }
            Ok(())
        }
        Op::Measure {
            qubits,
            measurements,
            flip_probability,
            ..
        } => {
            validate_probability("Measure", *flip_probability)?;
            if qubits.len() != measurements.len() {
                return Err(NodeTemplateInstanceMergeError::MeasureOutputCountMismatch {
                    qubits: qubits.len(),
                    measurements: measurements.len(),
                });
            }
            Ok(())
        }
        Op::MPP {
            products,
            measurements,
        } => {
            validate_mpp_arity(products, measurements)?;
            for product in products {
                mpp_record_coord(product)?;
            }
            Ok(())
        }
        Op::Depolarize1 { probability, .. } => validate_probability("DEPOLARIZE1", *probability),
        Op::Depolarize2 {
            probability,
            qubits,
        } => {
            validate_probability("DEPOLARIZE2", *probability)?;
            if !qubits.len().is_multiple_of(2) {
                return Err(NodeTemplateInstanceMergeError::OddDepolarize2TargetCount {
                    targets: qubits.len(),
                });
            }
            if let Some(pair) = qubits
                .as_chunks::<2>()
                .0
                .iter()
                .find(|pair| pair[0] == pair[1])
            {
                return Err(NodeTemplateInstanceMergeError::RepeatedDepolarize2Target(
                    pair[0],
                ));
            }
            Ok(())
        }
        Op::PauliError { probability, .. } => validate_probability("Pauli error", *probability),
        Op::Gate { .. } | Op::Tick | Op::Repeat { .. } | Op::ConditionalPauli(_) => Ok(()),
    }
}

fn validate_noise_model(noise: &NoiseModel) -> Result<(), NodeTemplateInstanceMergeError> {
    for (instruction, probability) in [
        ("DEPOLARIZE1", noise.p1),
        ("DEPOLARIZE2", noise.p2),
        ("Measure", noise.p_meas),
        ("Pauli error", noise.p_reset),
        ("DEPOLARIZE1", noise.p_idle),
    ] {
        validate_probability(instruction, probability)?;
    }
    Ok(())
}

fn validate_probability(
    instruction: &'static str,
    probability: f64,
) -> Result<(), NodeTemplateInstanceMergeError> {
    if !(0.0..=1.0).contains(&probability) {
        return Err(NodeTemplateInstanceMergeError::InvalidProbability { instruction });
    }
    Ok(())
}

fn validate_measurement_output(
    source: &CoordCircuit,
    measurement: u32,
    produced: IVec2,
) -> Result<(), NodeTemplateInstanceMergeError> {
    let record = source
        .meas_registry()
        .record(measurement)
        .ok_or(NodeTemplateInstanceMergeError::UnregisteredMeasurementOutput(measurement))?;
    if record.qubit != produced {
        return Err(
            NodeTemplateInstanceMergeError::MeasurementOutputCoordinateMismatch {
                measurement,
                registered: record.qubit,
                produced,
            },
        );
    }
    Ok(())
}

fn checked_translate_pauli_map(
    product: &PauliMap,
    offset: IVec2,
) -> Result<PauliMap, NodeTemplateInstanceMergeError> {
    let entries = product
        .iter()
        .map(|(coordinate, pauli)| Ok((checked_translate_coordinate(*coordinate, offset)?, *pauli)))
        .collect::<Result<Vec<_>, NodeTemplateInstanceMergeError>>()?;
    Ok(PauliMap::from_unique_entries(entries))
}

fn checked_translate_instance_coord(
    coordinate: IVec2,
    offset: IVec2,
) -> Result<IVec2, NodeTemplateInstanceMergeError> {
    Ok(checked_translate_coordinate(coordinate, offset)?)
}

fn translate_qubits(
    qubits: &[IVec2],
    offset: IVec2,
) -> Result<Vec<IVec2>, NodeTemplateInstanceMergeError> {
    qubits
        .iter()
        .map(|&qubit| checked_translate_instance_coord(qubit, offset))
        .collect()
}

struct InstanceTranslator<'a, 'b> {
    source: &'a CoordCircuit,
    offset: IVec2,
    instance: TemplateInstanceId,
    next_measurement: &'b mut u32,
    measurements: &'b mut FxMap<InstanceMeasurement, u32>,
    destination: CoordCircuit,
    body_map: &'b mut FxMap<BodyId, BodyId>,
}

impl InstanceTranslator<'_, '_> {
    /// Mint this instance's record id for `source_measurement`, binding it to
    /// `qubit`. The single place the id-allocation and duplicate-output policy
    /// lives — `Measure` and `MPP` differ only in how they derive `qubit`.
    fn allocate_record(
        &mut self,
        source_measurement: u32,
        qubit: IVec2,
    ) -> Result<u32, NodeTemplateInstanceMergeError> {
        let instance_measurement = InstanceMeasurement {
            instance: self.instance,
            measurement: source_measurement,
        };
        if self.measurements.contains_key(&instance_measurement) {
            return Err(NodeTemplateInstanceMergeError::DuplicateMeasurementOutput(
                source_measurement,
            ));
        }
        let measurement = *self.next_measurement;
        *self.next_measurement = self
            .next_measurement
            .checked_add(1)
            .ok_or(NodeTemplateInstanceMergeError::MeasurementIdOverflow)?;
        self.measurements.insert(instance_measurement, measurement);
        self.destination.register_measurement_id(measurement, qubit);
        Ok(measurement)
    }

    fn translate_body(&mut self, body: BodyId) -> Result<BodyId, NodeTemplateInstanceMergeError> {
        if let Some(mapped) = self.body_map.get(&body).copied() {
            return Ok(mapped);
        }
        let destination_body = self.allocate_body(body);
        let mut pending = vec![(body, destination_body, 0usize, Vec::new())];
        while let Some((source, destination, next, translated)) = pending.last_mut() {
            let ops = self
                .source
                .body(*source)
                .ok_or(NodeTemplateInstanceMergeError::InvalidBody(*source))?
                .ops();
            let Some(op) = ops.get(*next) else {
                *self
                    .destination
                    .body_mut(*destination)
                    .expect("allocated destination")
                    .ops_mut() = std::mem::take(translated);
                pending.pop();
                continue;
            };
            if let Op::Repeat { body, .. } = op
                && !self.body_map.contains_key(body)
            {
                let destination = self.allocate_body(*body);
                pending.push((*body, destination, 0, Vec::new()));
                continue;
            }
            // Finish children before later ops so measurement controls retain
            // the same first-production order as the recursive translation.
            translated.push(self.translate_op(op)?);
            *next += 1;
        }
        Ok(destination_body)
    }

    fn allocate_body(&mut self, body: BodyId) -> BodyId {
        let destination = if body == self.source.entry_body() {
            self.destination.entry_body()
        } else {
            self.destination.add_body(CircuitBody::new())
        };
        self.body_map.insert(body, destination);
        destination
    }

    fn translate_op(&mut self, op: &Op) -> Result<Op, NodeTemplateInstanceMergeError> {
        // `preflight_circuit` (run once per template by `instantiate_instance`)
        // already validated this op's shape via `validate_source_op`; translation
        // only offsets coordinates and allocates instance measurement ids.
        match op {
            Op::Gate { gate, qubits } => Ok(Op::Gate {
                gate: *gate,
                qubits: translate_qubits(qubits, self.offset)?,
            }),
            Op::Measure {
                basis,
                qubits,
                measurements: source_measurements,
                flip_probability,
            } => {
                let qubits = translate_qubits(qubits, self.offset)?;
                let mut op_measurements = Vec::with_capacity(qubits.len());
                for (&source_measurement, &qubit) in source_measurements.iter().zip(&qubits) {
                    op_measurements.push(self.allocate_record(source_measurement, qubit)?);
                }
                Ok(Op::Measure {
                    basis: *basis,
                    qubits,
                    measurements: op_measurements,
                    flip_probability: *flip_probability,
                })
            }
            // Like `Measure`, but the support lives in `products`: offset each
            // product and allocate one fresh instance measurement id per product.
            Op::MPP {
                products,
                measurements: source_measurements,
            } => {
                let products = products
                    .iter()
                    .map(|product| checked_translate_pauli_map(product, self.offset))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut op_measurements = Vec::with_capacity(source_measurements.len());
                for (&source_measurement, product) in source_measurements.iter().zip(&products) {
                    // Derived from the product so the merge stage, which sees only
                    // the op, recovers the same coordinate via `mpp_record_coord`.
                    let qubit = mpp_record_coord(product)?;
                    op_measurements.push(self.allocate_record(source_measurement, qubit)?);
                }
                Ok(Op::MPP {
                    products,
                    measurements: op_measurements,
                })
            }
            Op::Tick => Ok(Op::Tick),
            Op::Depolarize1 {
                probability,
                qubits,
            } => Ok(Op::Depolarize1 {
                probability: *probability,
                qubits: translate_qubits(qubits, self.offset)?,
            }),
            Op::Depolarize2 {
                probability,
                qubits,
            } => Ok(Op::Depolarize2 {
                probability: *probability,
                qubits: translate_qubits(qubits, self.offset)?,
            }),
            Op::PauliError {
                probability,
                pauli,
                qubits,
            } => Ok(Op::PauliError {
                probability: *probability,
                pauli: *pauli,
                qubits: translate_qubits(qubits, self.offset)?,
            }),
            Op::Repeat { body, repetitions } => Ok(Op::Repeat {
                body: self.body_map[body],
                repetitions: *repetitions,
            }),
            // Offset each target qubit and remap each measurement control id into
            // instance space via the same table `Measure` populates. The
            // measurement precedes its controlled correction, so the lookup is
            // already present. A
            // `Value`-slot control is node-scoped (an incoming `Value` edge), not
            // measurement-space, so it passes through untouched (U14).
            Op::ConditionalPauli(corrections) => Ok(Op::ConditionalPauli(
                corrections
                    .iter()
                    .map(|correction| {
                        let control = self
                            .measurements
                            .get(&InstanceMeasurement {
                                instance: self.instance,
                                measurement: correction.control,
                            })
                            .copied()
                            .ok_or(
                                NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(
                                    correction.control,
                                ),
                            )?;
                        Ok(ConditionalCorrection {
                            control,
                            target: checked_translate_instance_coord(
                                correction.target,
                                self.offset,
                            )?,
                            ..*correction
                        })
                    })
                    .collect::<Result<Vec<_>, NodeTemplateInstanceMergeError>>()?,
            )),
        }
    }
}

struct MergedInstantiatedCircuits {
    circuit: CoordCircuit,
    body_aliases: FxMap<(usize, BodyId), BodyId>,
    measurement_aliases: FxMap<u32, u32>,
}

fn merge_instantiated_circuits_with_body_aliases(
    circuits: &[CoordCircuit],
) -> Result<MergedInstantiatedCircuits, NodeTemplateInstanceMergeError> {
    if circuits.is_empty() {
        return Ok(MergedInstantiatedCircuits {
            circuit: CoordCircuit::new(),
            body_aliases: FxMap::default(),
            measurement_aliases: FxMap::default(),
        });
    };

    let mut destination = CoordCircuit::new();
    let capacity = count_circuits(circuits)?;
    let body_sources = circuits
        .iter()
        .enumerate()
        .map(|(index, circuit)| BodySource {
            index,
            circuit,
            body: circuit.entry_body(),
        })
        .collect::<Vec<_>>();
    let mut body_aliases = FxMap::with_capacity_and_hasher(
        capacity.bodies.saturating_sub(circuits.len()),
        Default::default(),
    );
    let mut measurement_records = Vec::with_capacity(capacity.measurements);
    let entry = destination.entry_body();
    let aliases = merge_bodies_into(
        body_sources,
        &mut destination,
        entry,
        &mut body_aliases,
        &mut measurement_records,
    )?;
    destination.register_measurement_records(&mut measurement_records);
    destination.remap_measurement_ids(&aliases);
    Ok(MergedInstantiatedCircuits {
        circuit: destination,
        body_aliases,
        measurement_aliases: aliases,
    })
}

#[derive(Clone, Copy)]
struct BodySource<'a> {
    index: usize,
    circuit: &'a CoordCircuit,
    body: BodyId,
}

/// Split a body's ops into tick-delimited segments, each a contiguous slice of
/// the source ops. Ticks separate segments and are dropped; a `Repeat` op is
/// isolated into its own one-op segment so the merge can pair repeats across
/// instances. Returning borrowed slices avoids cloning every op just to split
/// it — the merge clones only the ops it actually keeps.
fn split_ops_at_ticks(ops: &[Op]) -> Vec<&[Op]> {
    let mut segments = Vec::new();
    let mut start = 0;
    for (index, op) in ops.iter().enumerate() {
        match op {
            Op::Tick => {
                segments.push(&ops[start..index]);
                start = index + 1;
            }
            Op::Repeat { .. } => {
                if start < index {
                    segments.push(&ops[start..index]);
                }
                segments.push(&ops[index..index + 1]);
                start = index + 1;
            }
            Op::Gate { .. }
            | Op::Measure { .. }
            | Op::MPP { .. }
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. }
            | Op::ConditionalPauli(_) => {}
        }
    }
    segments.push(&ops[start..]);
    segments
}

struct BodyMergeFrame<'a> {
    sources: Vec<BodySource<'a>>,
    segments: Vec<Vec<&'a [Op]>>,
    destination: BodyId,
    next_segment: usize,
    merged: Vec<Op>,
}

impl<'a> BodyMergeFrame<'a> {
    fn new(
        sources: Vec<BodySource<'a>>,
        destination: BodyId,
        active_bodies: &mut crate::FxSet<(usize, BodyId)>,
    ) -> Result<Self, NodeTemplateInstanceMergeError> {
        let mut segments = Vec::with_capacity(sources.len());
        for source in &sources {
            if !active_bodies.insert((source.index, source.body)) {
                return Err(NodeTemplateInstanceMergeError::CyclicBody(source.body));
            }
            let body = source
                .circuit
                .body(source.body)
                .ok_or(NodeTemplateInstanceMergeError::InvalidBody(source.body))?;
            segments.push(split_ops_at_ticks(body.ops()));
        }
        // All members must have the same tick/repeat structure (SEM-MERGE).
        let segment_count = segments.first().map_or(0, Vec::len);
        if segments
            .iter()
            .any(|segments| segments.len() != segment_count)
        {
            return Err(NodeTemplateInstanceMergeError::TickSegmentCountMismatch);
        }
        let merged = Vec::with_capacity(estimate_merged_ops(&segments));
        Ok(Self {
            sources,
            segments,
            destination,
            next_segment: 0,
            merged,
        })
    }
}

fn estimate_merged_ops(segments: &[Vec<&[Op]>]) -> usize {
    segments
        .iter()
        .flat_map(|body_segments| body_segments.iter())
        .map(|segment| segment.len())
        .sum()
}

/// An MPP product's measurement-record coordinate.
fn mpp_record_coord(product: &PauliMap) -> Result<IVec2, NodeTemplateInstanceMergeError> {
    product
        .representative_coord()
        .ok_or(NodeTemplateInstanceMergeError::EmptyPauliProduct)
}

fn validate_mpp_arity(
    products: &[PauliMap],
    measurements: &[u32],
) -> Result<(), NodeTemplateInstanceMergeError> {
    if products.len() != measurements.len() {
        return Err(
            NodeTemplateInstanceMergeError::MppMeasurementCountMismatch {
                products: products.len(),
                measurements: measurements.len(),
            },
        );
    }
    if products.is_empty() {
        return Err(NodeTemplateInstanceMergeError::EmptyMpp);
    }
    Ok(())
}

fn estimate_segment_qubit_refs(segments: &[Vec<&[Op]>], index: usize) -> usize {
    segments
        .iter()
        .filter_map(|body_segments| body_segments.get(index))
        .flat_map(|segment| segment.iter())
        .map(|op| match op {
            Op::Gate { qubits, .. } | Op::Measure { qubits, .. } => qubits.len(),
            Op::ConditionalPauli(corrections) => corrections.len(),
            Op::MPP { products, .. } => products.iter().map(PauliMap::len).sum(),
            Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. }
            | Op::Tick
            | Op::Repeat { .. } => 0,
        })
        .sum()
}

fn collect_compatible_repeat_segments<'a>(
    sources: &[BodySource<'a>],
    member_segments: &[Vec<&[Op]>],
    segment_index: usize,
) -> Result<Option<(Vec<BodySource<'a>>, u32)>, NodeTemplateInstanceMergeError> {
    let mut repeat_segments = Vec::new();
    let mut has_plain_segment = false;
    for (source, segments) in sources.iter().copied().zip(member_segments) {
        let Some(&segment) = segments.get(segment_index) else {
            continue;
        };
        if segment.is_empty() {
            continue;
        }
        if let [Op::Repeat { body, repetitions }] = segment {
            repeat_segments.push((
                BodySource {
                    body: *body,
                    ..source
                },
                *repetitions,
            ));
        } else {
            has_plain_segment = true;
        }
    }
    if repeat_segments.is_empty() {
        return Ok(None);
    }
    if has_plain_segment {
        return Err(NodeTemplateInstanceMergeError::RepeatShapeMismatch);
    }
    let repetitions = repeat_segments[0].1;
    if repeat_segments
        .iter()
        .any(|(_, repeat_count)| *repeat_count != repetitions)
    {
        return Err(NodeTemplateInstanceMergeError::RepeatCountMismatch);
    }
    Ok(Some((
        repeat_segments
            .into_iter()
            .map(|(source, _)| source)
            .collect(),
        repetitions,
    )))
}

/// Append one instance's op to the merged segment, deduplicating overlaps that
/// are legal (a shared boundary qubit measured or reset identically by an
/// earlier instance) and rejecting overlaps that are not (a qubit claimed two
/// incompatible ways by *different* instances in one tick segment).
///
/// This is the single place the merge invariant is enforced; emission planning
/// validates by running this merge, so the rules cannot drift between the two.
///
/// `committed` holds the footprint of earlier instances this segment. `current`
/// holds only the current instance, so sequential reuse within that instance is
/// retained while any overlap with an earlier instance is rejected.
fn append_segment_op(
    op: &Op,
    committed: &SegmentDuplicateState,
    current: &mut SegmentDuplicateState,
    aliases: &mut FxMap<u32, u32>,
    measurement_records: &mut Vec<MeasRecord>,
    merged: &mut Vec<Op>,
) -> Result<(), NodeTemplateInstanceMergeError> {
    match op {
        Op::Measure { .. } => append_segment_measurement_op(
            op,
            committed,
            current,
            aliases,
            measurement_records,
            merged,
        ),
        Op::Gate { gate, qubits } if gate.is_reset() => {
            append_segment_reset_op(*gate, qubits, committed, current, merged)
        }
        Op::Gate { qubits, .. } => {
            claim_segment_qubits(qubits.iter().copied(), committed, current)?;
            merged.push(op.clone());
            Ok(())
        }
        Op::ConditionalPauli(corrections) => {
            claim_segment_qubits(
                corrections.iter().map(|correction| correction.target),
                committed,
                current,
            )?;
            merged.push(op.clone());
            Ok(())
        }
        // MPPs do not deduplicate across instances, but every product qubit is
        // still part of this instance's segment footprint. As with plain
        // gates, repeated use within one instance is legal; only an earlier
        // instance's committed footprint conflicts.
        Op::MPP {
            products,
            measurements,
        } => {
            validate_mpp_arity(products, measurements)?;
            for (product, &measurement) in products.iter().zip(measurements) {
                claim_segment_qubits(product.iter().map(|(qubit, _)| *qubit), committed, current)?;
                measurement_records.push(MeasRecord {
                    id: measurement,
                    qubit: mpp_record_coord(product)?,
                });
            }
            merged.push(op.clone());
            Ok(())
        }
        Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. }
        | Op::Repeat { .. }
        | Op::Tick => {
            merged.push(op.clone());
            Ok(())
        }
    }
}

#[derive(Default)]
struct SegmentDuplicateState {
    measurements: FxMap<IVec2, (u32, bloq_circuit::PauliBasis, f64)>,
    reset_gates: FxMap<IVec2, bloq_circuit::GateType>,
    /// Qubits with non-deduplicable activity, including multiple operations
    /// from one instance in this segment.
    occupied: crate::FxSet<IVec2>,
}

impl SegmentDuplicateState {
    fn reserve(&mut self, capacity: usize) {
        self.measurements.reserve(capacity);
        self.reset_gates.reserve(capacity);
        self.occupied.reserve(capacity);
    }

    fn has_any(&self, qubit: IVec2) -> bool {
        self.measurements.contains_key(&qubit)
            || self.reset_gates.contains_key(&qubit)
            || self.occupied.contains(&qubit)
    }

    fn mark_occupied(&mut self, qubit: IVec2) {
        self.measurements.remove(&qubit);
        self.reset_gates.remove(&qubit);
        self.occupied.insert(qubit);
    }

    fn clear(&mut self) {
        self.measurements.clear();
        self.reset_gates.clear();
        self.occupied.clear();
    }

    /// Fold one instance's `current` footprint into the committed state,
    /// draining `current` so the caller reuses its allocations for the next
    /// instance rather than reallocating a fresh state per segment.
    fn commit(&mut self, current: &mut Self) {
        self.measurements.extend(current.measurements.drain());
        self.reset_gates.extend(current.reset_gates.drain());
        self.occupied.extend(current.occupied.drain());
    }
}

fn claim_segment_qubits(
    qubits: impl IntoIterator<Item = IVec2>,
    committed: &SegmentDuplicateState,
    current: &mut SegmentDuplicateState,
) -> Result<(), NodeTemplateInstanceMergeError> {
    for qubit in qubits {
        if committed.has_any(qubit) {
            return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
        }
        current.mark_occupied(qubit);
    }
    Ok(())
}

fn append_segment_measurement_op(
    op: &Op,
    committed: &SegmentDuplicateState,
    current: &mut SegmentDuplicateState,
    aliases: &mut FxMap<u32, u32>,
    measurement_records: &mut Vec<MeasRecord>,
    merged: &mut Vec<Op>,
) -> Result<(), NodeTemplateInstanceMergeError> {
    let Op::Measure {
        basis,
        qubits,
        measurements,
        flip_probability,
    } = op
    else {
        unreachable!("append_segment_measurement_op only accepts Measure")
    };
    let mut kept_qubits = Vec::with_capacity(qubits.len());
    let mut kept_measurements = Vec::with_capacity(measurements.len());
    for (&qubit, &measurement) in qubits.iter().zip(measurements) {
        if current.has_any(qubit) {
            if committed.has_any(qubit) {
                return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
            }
            current.mark_occupied(qubit);
            measurement_records.push(MeasRecord {
                id: measurement,
                qubit,
            });
            kept_qubits.push(qubit);
            kept_measurements.push(measurement);
            continue;
        }

        match committed.measurements.get(&qubit).copied() {
            // A shared boundary qubit already measured by an earlier instance:
            // dedup, but only if both instances agree on the basis.
            Some((canonical, canonical_basis, canonical_flip_probability)) => {
                if canonical_basis != *basis || canonical_flip_probability != *flip_probability {
                    return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
                }
                aliases.insert(measurement, canonical);
                current
                    .measurements
                    .insert(qubit, (canonical, *basis, *flip_probability));
            }
            // No prior measurement to dedup against, so a plain gate or reset
            // left here by an earlier instance is a genuine overlap.
            None => {
                if committed.has_any(qubit) {
                    return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
                }
                current
                    .measurements
                    .insert(qubit, (measurement, *basis, *flip_probability));
                measurement_records.push(MeasRecord {
                    id: measurement,
                    qubit,
                });
                kept_qubits.push(qubit);
                kept_measurements.push(measurement);
            }
        }
    }

    if !kept_qubits.is_empty() {
        merged.push(Op::Measure {
            basis: *basis,
            qubits: kept_qubits,
            measurements: kept_measurements,
            flip_probability: *flip_probability,
        });
    }
    Ok(())
}

fn append_segment_reset_op(
    gate: bloq_circuit::GateType,
    qubits: &[IVec2],
    committed: &SegmentDuplicateState,
    current: &mut SegmentDuplicateState,
    merged: &mut Vec<Op>,
) -> Result<(), NodeTemplateInstanceMergeError> {
    let mut kept_qubits = Vec::with_capacity(qubits.len());
    for &qubit in qubits {
        if current.has_any(qubit) {
            if committed.has_any(qubit) {
                return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
            }
            current.mark_occupied(qubit);
            kept_qubits.push(qubit);
            continue;
        }

        match committed.reset_gates.get(&qubit).copied() {
            // Duplicate identical reset from an earlier instance: keep one.
            Some(previous) if previous == gate => {
                current.reset_gates.insert(qubit, gate);
            }
            Some(_) => return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit)),
            None => {
                if committed.has_any(qubit) {
                    return Err(NodeTemplateInstanceMergeError::QubitConflict(qubit));
                }
                current.reset_gates.insert(qubit, gate);
                kept_qubits.push(qubit);
            }
        }
    }
    if !kept_qubits.is_empty() {
        merged.push(Op::Gate {
            gate,
            qubits: kept_qubits,
        });
    }
    Ok(())
}

fn merge_bodies_into(
    sources: Vec<BodySource<'_>>,
    destination: &mut CoordCircuit,
    destination_body: BodyId,
    body_aliases: &mut FxMap<(usize, BodyId), BodyId>,
    measurement_records: &mut Vec<MeasRecord>,
) -> Result<FxMap<u32, u32>, NodeTemplateInstanceMergeError> {
    let mut active_bodies = crate::FxSet::default();
    let mut merged_body_tuples = FxMap::default();
    let mut pending = vec![BodyMergeFrame::new(
        sources,
        destination_body,
        &mut active_bodies,
    )?];
    // Aliases only gain entries during traversal. Keep one map rather than
    // copying every descendant's aliases once per ancestor body.
    let mut aliases = FxMap::default();
    let mut committed = SegmentDuplicateState::default();
    let mut current = SegmentDuplicateState::default();
    while let Some(frame) = pending.last_mut() {
        let segment_count = frame.segments.first().map_or(0, Vec::len);
        let index = frame.next_segment;
        if index == segment_count {
            let frame = pending.pop().expect("current body");
            *destination
                .body_mut(frame.destination)
                .expect("destination body was allocated above")
                .ops_mut() = frame.merged;
            for source in &frame.sources {
                active_bodies.remove(&(source.index, source.body));
            }
            if !pending.is_empty() {
                let body_tuple = frame
                    .sources
                    .iter()
                    .map(|source| (source.index, source.body))
                    .collect::<Vec<_>>();
                for &source in &body_tuple {
                    body_aliases.insert(source, frame.destination);
                }
                merged_body_tuples.insert(body_tuple, frame.destination);
            }
            continue;
        }
        frame.next_segment += 1;
        if index > 0 {
            frame.merged.push(Op::Tick);
        }
        if let Some((repeats, repetitions)) =
            collect_compatible_repeat_segments(&frame.sources, &frame.segments, index)?
        {
            let body_tuple = repeats
                .iter()
                .map(|source| (source.index, source.body))
                .collect::<Vec<_>>();
            if let Some(&body) = merged_body_tuples.get(&body_tuple) {
                frame.merged.push(Op::Repeat { body, repetitions });
                continue;
            }
            if let Some(source) = repeats
                .iter()
                .find(|source| body_aliases.contains_key(&(source.index, source.body)))
            {
                return Err(NodeTemplateInstanceMergeError::BodyMergeConflict {
                    circuit_index: source.index,
                    body: source.body,
                });
            }
            u32::try_from(destination.body_count())
                .map_err(|_| NodeTemplateInstanceMergeError::BodyCountOverflow)?;
            let body = destination.add_body(CircuitBody::new());
            frame.merged.push(Op::Repeat { body, repetitions });
            pending.push(BodyMergeFrame::new(repeats, body, &mut active_bodies)?);
        } else {
            // Reusing a qubit inside one instance is legal. Only committed
            // footprints from earlier instances participate in overlap checks.
            committed.clear();
            let capacity = estimate_segment_qubit_refs(&frame.segments, index);
            committed.reserve(capacity);
            for segments in &frame.segments {
                current.reserve(capacity);
                for op in segments[index] {
                    append_segment_op(
                        op,
                        &committed,
                        &mut current,
                        &mut aliases,
                        measurement_records,
                        &mut frame.merged,
                    )?;
                }
                committed.commit(&mut current);
            }
        }
    }
    Ok(aliases)
}

/// Preflight one template circuit through the shared analysis used by
/// instance emission, returning every body reached as a `REPEAT` target.
pub(crate) fn preflight_template_circuit(
    circuit: &CoordCircuit,
) -> Result<TemplateCircuitAnalysis, NodeTemplateInstanceMergeError> {
    // Validate operation arities and measurement-registry ownership through the
    // same helpers used by instance translation. This walk deliberately descends
    // through zero-count repeats: they skip runtime events, not structural
    // validation. `preflight_circuit` records every `REPEAT`-target body
    // directly, so the former single-source merge (whose sole role was recovering
    // that set, and whose cross-instance error paths cannot fire for one circuit)
    // is gone — this no longer merges, hence the name.
    preflight_circuit(circuit)
}

/// Validate one template circuit through the shared preflight used by both
/// [`Bloq::validate`](crate::Bloq::validate) and instance emission.
///
/// # Errors
///
/// [`NodeTemplateInstanceMergeError`] if bodies, operation arities,
/// measurement bindings, or probabilities are invalid.
pub fn validate_template_circuit(
    circuit: &CoordCircuit,
) -> Result<(), NodeTemplateInstanceMergeError> {
    preflight_template_circuit(circuit).map(|_| ())
}

fn canonical_measurement(mut measurement: u32, aliases: &FxMap<u32, u32>) -> u32 {
    while let Some(canonical) = aliases.get(&measurement).copied() {
        measurement = canonical;
    }
    measurement
}

#[cfg(test)]
mod tests {
    #[test]
    fn instance_capacity_checks_before_allocation() {
        let mut counts = super::CircuitCounts {
            measurements: u32::MAX as usize,
            bodies: 0,
        };
        assert!(matches!(
            counts.add_counts(1, 0),
            Err(super::NodeTemplateInstanceMergeError::MeasurementIdOverflow)
        ));
        let mut counts = super::CircuitCounts {
            measurements: 0,
            bodies: usize::MAX,
        };
        assert!(matches!(
            counts.add_counts(0, 1),
            Err(super::NodeTemplateInstanceMergeError::BodyCountOverflow)
        ));
        assert_eq!(counts.bodies, usize::MAX);
    }
    use super::{
        InstantiationOptions, NodeEmissionPlan, merge_instantiated_circuits_with_body_aliases,
    };
    use bloq_circuit::{
        BodyId, CircuitBody, ConditionalCorrection, CoordCircuit, GateType, NoiseModel, Op, Pauli,
        PauliBasis, PauliMap,
    };
    use glam::{IVec2, ivec2};

    use crate::test_fixture::instance_measurement;
    use crate::{
        BloqNode, BloqTemplate, BloqTemplatePool, CoordinateOverflowError,
        NodeTemplateInstanceMergeError, TemplateInstance, TemplateInstanceId,
    };

    #[test]
    fn instantiation_validation_defaults_preserve_explicit_limits() {
        use bloq_utils::boolean::BooleanLimits;

        let noise = NoiseModel::uniform_depolarizing(0.0);
        let mut bloq = crate::Bloq::new();
        bloq.add_node(BloqNode::classical(crate::ClassicalNode::Compute {
            expr: crate::ClassicalExpr::Const(true),
        }));
        let expected = BooleanLimits {
            max_nodes: BooleanLimits::DEFAULT.max_nodes,
            max_steps: 128_000_000,
        };
        for options in [
            InstantiationOptions::default(),
            InstantiationOptions::noisy(&noise),
        ] {
            assert_eq!(options.boolean_limits(), expected);
            bloq.validate_with_options(&options).unwrap();
            let explicit = BooleanLimits {
                max_nodes: 0,
                max_steps: 0,
            };
            let limited = options.with_boolean_limits(explicit);
            assert_eq!(limited.boolean_limits(), explicit);
            assert!(matches!(
                bloq.validate_with_options(&limited),
                Err(crate::BloqValidationError::BooleanResource(error))
                    if error.resource == "Boolean work steps" && error.limit == 0 && error.observed > 0
            ));
        }
    }

    fn malformed_mpp_error(
        products: Vec<PauliMap>,
        measurements: Vec<u32>,
    ) -> NodeTemplateInstanceMergeError {
        let qubit = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        for &measurement in &measurements {
            circuit.register_measurement_id(measurement, qubit);
        }
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::MPP {
                products,
                measurements,
            });
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                ivec2(0, 0),
            ));
        node.emission_plan(&templates).unwrap_err()
    }

    fn x_product() -> PauliMap {
        [(ivec2(0, 0), Pauli::X)].into_iter().collect()
    }

    #[derive(Debug, Clone, Copy)]
    enum SegmentOccupant {
        Gate,
        Reset,
        Measure,
    }

    fn append_mpp(circuit: &mut CoordCircuit) {
        circuit
            .measure_pauli_products([x_product()])
            .expect("non-empty product");
    }

    fn append_occupant(circuit: &mut CoordCircuit, occupant: SegmentOccupant) {
        match occupant {
            SegmentOccupant::Gate => circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap(),
            SegmentOccupant::Reset => circuit.do_gate(GateType::RZ, [ivec2(0, 0)]).unwrap(),
            SegmentOccupant::Measure => {
                circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
            }
        }
    }

    /// A correction on (0,0) gated by a measurement of an off-target qubit, so
    /// the control is live without the control's own qubit joining the segment
    /// the correction is being tested against.
    fn append_conditional_pauli(circuit: &mut CoordCircuit) {
        let control = circuit.measure(PauliBasis::Z, [ivec2(9, 9)])[0];
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: ivec2(0, 0),
            }]));
    }

    fn single_instance_plan(circuit: CoordCircuit) -> NodeEmissionPlan {
        single_instance_plan_with_options(circuit, InstantiationOptions::default()).unwrap()
    }

    fn single_instance_plan_with_options(
        circuit: CoordCircuit,
        options: InstantiationOptions<'_>,
    ) -> Result<NodeEmissionPlan, NodeTemplateInstanceMergeError> {
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                ivec2(0, 0),
            ));
        node.emission_plan_with_options(&templates, options)
    }

    fn append_repeats(circuit: &mut CoordCircuit, repeats: &[(BodyId, u32)]) {
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend(
                repeats
                    .iter()
                    .map(|&(body, repetitions)| Op::Repeat { body, repetitions }),
            );
    }

    fn two_instance_overlap_error(
        first: CoordCircuit,
        second: CoordCircuit,
    ) -> NodeTemplateInstanceMergeError {
        let mut templates = BloqTemplatePool::new();
        let first = templates.insert(BloqTemplate::new(first));
        let second = templates.insert(BloqTemplate::new(second));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), first, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), second, ivec2(0, 0)),
        ];
        node.emission_plan(&templates).unwrap_err()
    }

    fn three_instance_overlap_error(
        first: CoordCircuit,
        second: CoordCircuit,
        third: CoordCircuit,
    ) -> NodeTemplateInstanceMergeError {
        let mut templates = BloqTemplatePool::new();
        let template_ids = [first, second, third]
            .into_iter()
            .map(|circuit| templates.insert(BloqTemplate::new(circuit)))
            .collect::<Vec<_>>();
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = template_ids
            .into_iter()
            .enumerate()
            .map(|(index, template_id)| {
                TemplateInstance::new(TemplateInstanceId(index as u32), template_id, ivec2(0, 0))
            })
            .collect();
        node.emission_plan(&templates).unwrap_err()
    }

    fn single_instance_error(
        circuit: CoordCircuit,
        offset: glam::IVec2,
    ) -> NodeTemplateInstanceMergeError {
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                offset,
            ));
        node.emission_plan(&templates).unwrap_err()
    }

    #[test]
    fn instantiate_rejects_mpp_product_without_measurement() {
        assert_eq!(
            malformed_mpp_error(vec![x_product(), PauliMap::empty()], vec![0]),
            NodeTemplateInstanceMergeError::MppMeasurementCountMismatch {
                products: 2,
                measurements: 1,
            }
        );
    }

    #[test]
    fn instantiate_rejects_mpp_measurement_without_product() {
        assert_eq!(
            malformed_mpp_error(vec![x_product()], vec![0, 1]),
            NodeTemplateInstanceMergeError::MppMeasurementCountMismatch {
                products: 1,
                measurements: 2,
            }
        );
    }

    #[test]
    fn mpp_conflicts_with_every_cross_instance_segment_occupant() {
        for occupant in [
            SegmentOccupant::Gate,
            SegmentOccupant::Reset,
            SegmentOccupant::Measure,
        ] {
            for mpp_first in [false, true] {
                let mut mpp = CoordCircuit::new();
                append_mpp(&mut mpp);
                let mut other = CoordCircuit::new();
                append_occupant(&mut other, occupant);
                let error = if mpp_first {
                    two_instance_overlap_error(mpp, other)
                } else {
                    two_instance_overlap_error(other, mpp)
                };
                assert_eq!(
                    error,
                    NodeTemplateInstanceMergeError::QubitConflict(ivec2(0, 0)),
                    "{occupant:?}, mpp_first={mpp_first}"
                );
            }
        }
    }

    #[test]
    fn mpp_reuse_within_one_instance_remains_legal() {
        for occupant in [
            SegmentOccupant::Gate,
            SegmentOccupant::Reset,
            SegmentOccupant::Measure,
        ] {
            for mpp_first in [false, true] {
                let mut circuit = CoordCircuit::new();
                if mpp_first {
                    append_mpp(&mut circuit);
                    append_occupant(&mut circuit, occupant);
                } else {
                    append_occupant(&mut circuit, occupant);
                    append_mpp(&mut circuit);
                }
                let mut templates = BloqTemplatePool::new();
                let template = templates.insert(BloqTemplate::new(circuit));
                let mut node = BloqNode::from_members(vec![]);
                node.expect_quantum_mut()
                    .instances
                    .push(TemplateInstance::new(
                        TemplateInstanceId(0),
                        template,
                        ivec2(0, 0),
                    ));

                node.emission_plan(&templates)
                    .unwrap_or_else(|error| panic!("{occupant:?}, mpp_first={mpp_first}: {error}"));
            }
        }
    }

    #[test]
    fn gate_and_measurement_or_reset_reuse_within_one_instance_in_both_orders() {
        for occupant in [SegmentOccupant::Reset, SegmentOccupant::Measure] {
            for gate_first in [false, true] {
                let mut circuit = CoordCircuit::new();
                if gate_first {
                    append_occupant(&mut circuit, SegmentOccupant::Gate);
                    append_occupant(&mut circuit, occupant);
                } else {
                    append_occupant(&mut circuit, occupant);
                    append_occupant(&mut circuit, SegmentOccupant::Gate);
                }
                let plan = single_instance_plan(circuit);
                assert_eq!(
                    plan.circuit
                        .body(plan.circuit.entry_body())
                        .unwrap()
                        .ops()
                        .len(),
                    2,
                    "{occupant:?}, gate_first={gate_first}"
                );
            }
        }
    }

    #[test]
    fn measurement_and_reset_reuse_within_one_instance_in_both_orders() {
        for measurement_first in [false, true] {
            let mut circuit = CoordCircuit::new();
            if measurement_first {
                append_occupant(&mut circuit, SegmentOccupant::Measure);
                append_occupant(&mut circuit, SegmentOccupant::Reset);
            } else {
                append_occupant(&mut circuit, SegmentOccupant::Reset);
                append_occupant(&mut circuit, SegmentOccupant::Measure);
            }

            let plan = single_instance_plan(circuit);
            let ops = plan.circuit.body(plan.circuit.entry_body()).unwrap().ops();
            assert_eq!(ops.len(), 2, "measurement_first={measurement_first}");
            assert_eq!(plan.circuit.meas_registry().records().len(), 1);
        }
    }

    #[test]
    fn repeated_measurements_within_one_instance_keep_distinct_outputs() {
        let mut circuit = CoordCircuit::new();
        append_occupant(&mut circuit, SegmentOccupant::Measure);
        append_occupant(&mut circuit, SegmentOccupant::Measure);

        let plan = single_instance_plan(circuit);
        let ops = plan.circuit.body(plan.circuit.entry_body()).unwrap().ops();
        let measurements = ops
            .iter()
            .filter_map(|op| match op {
                Op::Measure { measurements, .. } => Some(measurements[0]),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(measurements.len(), 2);
        assert_ne!(measurements[0], measurements[1]);
        assert_eq!(plan.circuit.meas_registry().records().len(), 2);
    }

    #[test]
    fn repeated_resets_within_one_instance_are_both_retained() {
        let mut circuit = CoordCircuit::new();
        append_occupant(&mut circuit, SegmentOccupant::Reset);
        append_occupant(&mut circuit, SegmentOccupant::Reset);

        let plan = single_instance_plan(circuit);
        let reset_count = plan
            .circuit
            .body(plan.circuit.entry_body())
            .unwrap()
            .ops()
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    Op::Gate {
                        gate: GateType::RZ,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(reset_count, 2);
    }

    #[test]
    fn multi_op_instance_cannot_enable_later_measurement_or_reset_dedup() {
        for occupant in [SegmentOccupant::Reset, SegmentOccupant::Measure] {
            for composite_first in [false, true] {
                let mut composite = CoordCircuit::new();
                append_occupant(&mut composite, occupant);
                append_occupant(&mut composite, occupant);
                let mut single = CoordCircuit::new();
                append_occupant(&mut single, occupant);

                let error = if composite_first {
                    two_instance_overlap_error(composite, single)
                } else {
                    two_instance_overlap_error(single, composite)
                };
                assert_eq!(
                    error,
                    NodeTemplateInstanceMergeError::QubitConflict(ivec2(0, 0)),
                    "{occupant:?}, composite_first={composite_first}"
                );
            }
        }
    }

    #[test]
    fn conditional_pauli_conflicts_with_every_cross_instance_segment_occupant() {
        for occupant in [
            SegmentOccupant::Gate,
            SegmentOccupant::Reset,
            SegmentOccupant::Measure,
        ] {
            for correction_first in [false, true] {
                let mut correction = CoordCircuit::new();
                append_conditional_pauli(&mut correction);
                let mut other = CoordCircuit::new();
                append_occupant(&mut other, occupant);
                let error = if correction_first {
                    two_instance_overlap_error(correction, other)
                } else {
                    two_instance_overlap_error(other, correction)
                };
                assert_eq!(
                    error,
                    NodeTemplateInstanceMergeError::QubitConflict(ivec2(0, 0)),
                    "{occupant:?}, correction_first={correction_first}"
                );
            }
        }

        for correction_first in [false, true] {
            let mut correction = CoordCircuit::new();
            append_conditional_pauli(&mut correction);
            let mut mpp = CoordCircuit::new();
            append_mpp(&mut mpp);
            let error = if correction_first {
                two_instance_overlap_error(correction, mpp)
            } else {
                two_instance_overlap_error(mpp, correction)
            };
            assert_eq!(
                error,
                NodeTemplateInstanceMergeError::QubitConflict(ivec2(0, 0)),
                "MPP, correction_first={correction_first}"
            );
        }
    }

    #[test]
    fn conditional_pauli_reuse_within_one_instance_remains_legal() {
        for occupant in [
            SegmentOccupant::Gate,
            SegmentOccupant::Reset,
            SegmentOccupant::Measure,
        ] {
            for correction_first in [false, true] {
                let mut circuit = CoordCircuit::new();
                if correction_first {
                    append_conditional_pauli(&mut circuit);
                    append_occupant(&mut circuit, occupant);
                } else {
                    append_occupant(&mut circuit, occupant);
                    append_conditional_pauli(&mut circuit);
                }
                let plan = single_instance_plan(circuit);
                assert_eq!(
                    plan.circuit
                        .body(plan.circuit.entry_body())
                        .unwrap()
                        .ops()
                        .len(),
                    // occupant + the correction and the measurement gating it
                    3,
                    "{occupant:?}, correction_first={correction_first}"
                );
            }
        }

        for correction_first in [false, true] {
            let mut circuit = CoordCircuit::new();
            if correction_first {
                append_conditional_pauli(&mut circuit);
                append_mpp(&mut circuit);
            } else {
                append_mpp(&mut circuit);
                append_conditional_pauli(&mut circuit);
            }
            let plan = single_instance_plan(circuit);
            assert_eq!(
                plan.circuit
                    .body(plan.circuit.entry_body())
                    .unwrap()
                    .ops()
                    .len(),
                // MPP + the correction and the measurement gating it
                3,
                "MPP, correction_first={correction_first}"
            );
        }
    }

    #[test]
    fn composite_first_instance_cannot_enable_later_measurement_or_reset_dedup() {
        for occupant in [SegmentOccupant::Reset, SegmentOccupant::Measure] {
            let mut composite = CoordCircuit::new();
            append_mpp(&mut composite);
            append_occupant(&mut composite, occupant);
            let mut second = CoordCircuit::new();
            append_occupant(&mut second, occupant);
            let mut third = CoordCircuit::new();
            append_occupant(&mut third, occupant);

            assert_eq!(
                three_instance_overlap_error(composite, second, third),
                NodeTemplateInstanceMergeError::QubitConflict(ivec2(0, 0)),
                "{occupant:?}"
            );
        }
    }

    #[test]
    fn instance_translation_checks_coordinate_overflow_for_every_operation_shape() {
        let coordinate = ivec2(i32::MAX, 0);
        let offset = ivec2(1, 0);

        let mut gate = CoordCircuit::new();
        gate.do_gate(GateType::H, [coordinate]).unwrap();

        let mut measure = CoordCircuit::new();
        measure.measure(PauliBasis::Z, [coordinate]);

        let mut mpp = CoordCircuit::new();
        mpp.measure_pauli_products([[((coordinate), Pauli::X)].into_iter().collect()])
            .unwrap();

        let mut correction = CoordCircuit::new();
        let control = correction.measure(PauliBasis::Z, [ivec2(0, 0)])[0];
        correction
            .body_mut(correction.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: coordinate,
            }]));

        for (kind, circuit) in [
            ("gate", gate),
            ("measure", measure),
            ("MPP", mpp),
            ("correction", correction),
        ] {
            assert_eq!(
                single_instance_error(circuit, offset),
                NodeTemplateInstanceMergeError::CoordinateOverflow(CoordinateOverflowError {
                    coordinate,
                    offset,
                }),
                "{kind}"
            );
        }
    }

    #[test]
    fn instance_translation_checks_unused_registry_coordinates() {
        let coordinate = ivec2(i32::MAX, 0);
        let offset = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(0, coordinate);

        assert_eq!(
            single_instance_error(circuit, offset),
            NodeTemplateInstanceMergeError::CoordinateOverflow(CoordinateOverflowError {
                coordinate,
                offset,
            })
        );
    }

    #[test]
    fn instance_translation_rejects_control_after_zero_repeat() {
        let measured = ivec2(0, 0);
        let target = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(0, measured);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![measured],
            measurements: vec![0],
            flip_probability: 0.0,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend([
                Op::Repeat {
                    body,
                    repetitions: 0,
                },
                Op::ConditionalPauli(vec![ConditionalCorrection {
                    pauli: PauliBasis::X,
                    control: 0,
                    target,
                }]),
            ]);

        assert_eq!(
            single_instance_error(circuit, ivec2(0, 0)),
            NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(0)
        );
    }

    #[test]
    fn instantiate_circuit_merges_translated_template_instances() {
        let mut template_circuit = CoordCircuit::new();
        template_circuit
            .do_gate(GateType::H, [ivec2(0, 0)])
            .unwrap();
        template_circuit.tick();
        template_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), template_id, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), template_id, ivec2(2, 0)),
        ];

        let circuit = node.instantiate_circuit(&templates).unwrap();
        let measurement_map = node.emission_plan(&templates).unwrap().measurements;
        let ops = circuit.body(circuit.entry_body()).unwrap().ops();

        assert!(matches!(
            &ops[0],
            Op::Gate {
                gate: GateType::H,
                qubits,
            } if qubits == &vec![ivec2(0, 0)]
        ));
        assert!(matches!(
            &ops[1],
            Op::Gate {
                gate: GateType::H,
                qubits,
            } if qubits == &vec![ivec2(2, 0)]
        ));
        assert!(matches!(ops[2], Op::Tick));
        assert_eq!(circuit.meas_registry().records().len(), 2);
        assert_eq!(
            circuit
                .meas_registry()
                .records()
                .iter()
                .map(|record| record.qubit)
                .collect::<Vec<_>>(),
            vec![ivec2(0, 0), ivec2(2, 0)]
        );
        assert_eq!(measurement_map[&instance_measurement(0, 0)], 0);
        assert_eq!(measurement_map[&instance_measurement(1, 0)], 1);
    }

    #[test]
    fn noisy_instantiation_annotates_shared_physical_op_once_after_merge() {
        let qubit = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.do_gate(GateType::RZ, [qubit]).unwrap();
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), template_id, IVec2::ZERO),
            TemplateInstance::new(TemplateInstanceId(1), template_id, IVec2::ZERO),
        ];
        let noise = NoiseModel {
            p_reset: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };

        let circuit = node
            .emission_plan_with_options(&templates, InstantiationOptions::noisy(&noise))
            .unwrap()
            .circuit;
        let ops = circuit.body(circuit.entry_body()).unwrap().ops();

        assert_eq!(ops.len(), 2);
        assert!(matches!(
            &ops[0],
            Op::Gate { gate: GateType::RZ, qubits } if qubits == &[qubit]
        ));
        assert!(matches!(
            &ops[1],
            Op::PauliError { probability, pauli: PauliBasis::X, qubits }
                if *probability == 0.125 && qubits == &[qubit]
        ));
    }

    #[test]
    fn noisy_singleton_preserves_repeat_boundaries() {
        let mut template_circuit = CoordCircuit::new();
        template_circuit
            .do_gate(GateType::H, [IVec2::ZERO])
            .unwrap();
        template_circuit.tick();
        let body = template_circuit.add_body(CircuitBody::from_ops(vec![
            Op::Gate {
                gate: GateType::H,
                qubits: vec![ivec2(1, 0)],
            },
            Op::Tick,
        ]));
        append_repeats(&mut template_circuit, &[(body, 2)]);
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };

        let plan = single_instance_plan_with_options(
            template_circuit,
            InstantiationOptions::noisy(&noise),
        )
        .unwrap();
        assert!(plan.normalized_templates.is_none());
        assert_eq!(plan.circuit.entry_top_level_repeats(), [2]);
    }

    #[test]
    fn noise_does_not_make_incompatible_repeat_shapes_valid() {
        let program = crate::Bloq::from_text(
            "BLOQIR 1
template t0 {
  circuit {
    H (0,0)
    REPEAT 2 b1
  }
  body b1 {
    X (0,0)
    TICK
  }
}
template t1 {
  circuit {
    H (1,0)
    X (1,0)
    TICK
    X (1,0)
    TICK
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
    instance i1 t1 @ (0,0)
  }
}",
        )
        .unwrap();
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };
        let clean = program.validate().unwrap_err();
        assert!(matches!(
            clean,
            crate::BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::RepeatShapeMismatch,
                ..
            }
        ));
        assert_eq!(
            program
                .validate_with_plans(&InstantiationOptions::noisy(&noise))
                .unwrap_err(),
            clean
        );
    }

    #[test]
    fn noisy_multi_instance_plan_keeps_merge_isolation_ticks() {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Gate {
                gate: GateType::X,
                qubits: vec![IVec2::ZERO],
            },
            Op::Tick,
        ]));
        circuit.push_repeat(body, 2);
        let mut templates = BloqTemplatePool::new();
        let template = templates.insert(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), template, IVec2::ZERO),
            TemplateInstance::new(TemplateInstanceId(1), template, IVec2::X),
        ];
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };
        let clean = node.emission_plan(&templates).unwrap();
        let noisy = node
            .emission_plan_with_options(&templates, InstantiationOptions::noisy(&noise))
            .unwrap();
        assert!(noisy.normalized_templates.is_none());
        let mut actual = noisy.circuit;
        let mut expected = noise.noisy_circuit(&clean.circuit).unwrap();
        actual.flatten().unwrap();
        expected.flatten().unwrap();
        assert_eq!(
            actual.body(actual.entry_body()).unwrap().ops(),
            expected.body(expected.entry_body()).unwrap().ops()
        );
    }

    #[test]
    fn noisy_repeat_expansion_preserves_idle_members_and_measurement_aliases() {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [IVec2::ZERO]).unwrap();
        let measurement = 0;
        circuit.register_measurement_id(measurement, IVec2::ZERO);
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![IVec2::ZERO],
                measurements: vec![measurement],
                flip_probability: 0.0,
            },
            Op::Tick,
            Op::Gate {
                gate: GateType::RZ,
                qubits: vec![IVec2::ZERO],
            },
        ]));
        circuit.push_repeat(body, 2);
        let mut template = BloqTemplate::new(circuit);
        template.detectors.push(crate::TemplateDetector {
            scope: crate::TemplateDetectorScope::RepeatBody { body },
            parity: crate::TemplateDetectorParity::from_measurements([measurement]),
            coords: None,
        });
        let noise = NoiseModel {
            p_idle: 0.125,
            ..NoiseModel::uniform_depolarizing(0.0)
        };
        // Shared measurements, disjoint measurements, then an idle member
        // whose empty segment aligns with the other instance's repeat.
        for measured_offset in [Some(0), Some(2), None] {
            let mut templates = BloqTemplatePool::new();
            let first = templates.insert(template.clone());
            let second = if measured_offset.is_some() {
                first
            } else {
                let mut idle = CoordCircuit::new();
                idle.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
                idle.tick();
                idle.tick();
                templates.insert(BloqTemplate::new(idle))
            };
            let mut node = BloqNode::from_members(Vec::new());
            node.expect_quantum_mut().instances = vec![
                TemplateInstance::new(TemplateInstanceId(0), first, IVec2::ZERO),
                TemplateInstance::new(
                    TemplateInstanceId(1),
                    second,
                    ivec2(measured_offset.unwrap_or(2), 0),
                ),
            ];
            let mut clean = node.emission_plan(&templates).unwrap().circuit;
            clean.flatten().unwrap();
            let plan = node
                .emission_plan_with_options(&templates, InstantiationOptions::noisy(&noise))
                .unwrap();
            assert!(plan.normalized_templates.is_some());
            let expected = noise.noisy_circuit(&clean).unwrap();
            assert_eq!(
                plan.circuit.body(plan.circuit.entry_body()).unwrap().ops(),
                expected.body(expected.entry_body()).unwrap().ops()
            );
            let normalized = plan.templates(&templates);
            assert_eq!(normalized[first].detectors.len(), 2);
            for instance in &node.expect_quantum().instances {
                for (iteration, detector) in normalized[instance.template_id]
                    .detectors
                    .iter()
                    .enumerate()
                {
                    let source = detector.parity.measurements().next().unwrap();
                    let expected = clean
                        .body(clean.entry_body())
                        .unwrap()
                        .ops()
                        .iter()
                        .filter_map(|op| match op {
                            Op::Measure {
                                qubits,
                                measurements,
                                ..
                            } if qubits == &[instance.offset] => Some(measurements[0]),
                            _ => None,
                        })
                        .nth(iteration)
                        .unwrap();
                    assert_eq!(
                        plan.measurements[&instance_measurement(instance.id.0, source)],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn noisy_instantiation_rejects_non_finite_model_probability() {
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(CoordCircuit::new()));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                IVec2::ZERO,
            ));
        let mut noise = NoiseModel::uniform_depolarizing(0.0);
        noise.p1 = f64::NAN;

        let error = node
            .emission_plan_with_options(&templates, InstantiationOptions::noisy(&noise))
            .unwrap_err();

        assert_eq!(
            error,
            NodeTemplateInstanceMergeError::InvalidProbability {
                instruction: "DEPOLARIZE1"
            }
        );
    }

    #[test]
    fn instantiate_circuit_reports_repeat_body_map() {
        let mut template_circuit = CoordCircuit::new();
        let body = template_circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![ivec2(0, 0)],
            measurements: vec![0],
            flip_probability: 0.0,
        }]));
        template_circuit.register_measurement_id(0, ivec2(0, 0));
        template_circuit
            .body_mut(template_circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 2,
            });
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                ivec2(0, 0),
            ));

        let instantiated = node.emission_plan(&templates).unwrap();
        let Op::Repeat {
            body: instantiated_body,
            ..
        } = instantiated
            .circuit
            .body(instantiated.circuit.entry_body())
            .unwrap()
            .ops()[0]
        else {
            panic!("expected repeat op");
        };

        assert_eq!(
            instantiated.bodies[&(TemplateInstanceId(0), body)],
            instantiated_body
        );
    }

    #[test]
    fn multi_instance_merge_reuses_aligned_repeat_body_tuple() {
        let mut template_circuit = CoordCircuit::new();
        let body = template_circuit.add_body(CircuitBody::new());
        append_repeats(&mut template_circuit, &[(body, 2), (body, 3)]);
        let circuits = [template_circuit.clone(), template_circuit];

        let merged = merge_instantiated_circuits_with_body_aliases(&circuits).unwrap();
        let repeat_bodies = merged
            .circuit
            .body(merged.circuit.entry_body())
            .unwrap()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::Repeat { body, .. } => Some(*body),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(repeat_bodies.len(), 2);
        assert_eq!(repeat_bodies[0], repeat_bodies[1]);
        assert_eq!(merged.circuit.body_count(), 2);
        assert_eq!(merged.body_aliases[&(0, body)], repeat_bodies[0]);
        assert_eq!(merged.body_aliases[&(1, body)], repeat_bodies[0]);
    }

    #[test]
    fn deep_multi_instance_merge_preserves_shared_bodies_and_measurement_aliases() {
        fn circuit(target: IVec2) -> CoordCircuit {
            let mut circuit = CoordCircuit::new();
            let control = circuit.measure(PauliBasis::Z, [IVec2::ZERO])[0];
            circuit
                .body_mut(circuit.entry_body())
                .unwrap()
                .ops_mut()
                .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                    pauli: PauliBasis::X,
                    control,
                    target,
                }]));
            for _ in 0..4096 {
                let child = circuit.entry_body();
                let parent = circuit.add_body(CircuitBody::from_ops(vec![
                    Op::Repeat {
                        body: child,
                        repetitions: 1,
                    },
                    Op::Repeat {
                        body: child,
                        repetitions: 2,
                    },
                ]));
                circuit.set_entry_body(parent).unwrap();
            }
            circuit
        }

        let mut templates = BloqTemplatePool::new();
        let mut node = BloqNode::from_members(Vec::new());
        for index in 0..2 {
            let template = templates.insert(BloqTemplate::new(circuit(ivec2(index + 1, 0))));
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(index as u32),
                    template,
                    IVec2::ZERO,
                ));
        }
        let plan = node.emission_plan(&templates).unwrap();
        assert_eq!(plan.circuit.body_count(), 4097);
        assert_eq!(plan.bodies.len(), 8194);
        assert_eq!(plan.circuit.meas_registry().records().len(), 1);
        assert_eq!(
            plan.measurements[&instance_measurement(0, 0)],
            plan.measurements[&instance_measurement(1, 0)]
        );
        let mut body = plan.circuit.entry_body();
        for _ in 0..4096 {
            let [
                Op::Repeat {
                    body: first,
                    repetitions: 1,
                },
                Op::Tick,
                Op::Repeat {
                    body: second,
                    repetitions: 2,
                },
                Op::Tick,
            ] = plan.circuit.body(body).unwrap().ops()
            else {
                panic!("shared repeats and isolation ticks must retain their order");
            };
            assert_eq!(first, second);
            body = *first;
        }
        let ops = plan.circuit.body(body).unwrap().ops();
        assert_eq!(ops.len(), 3);
        for op in &ops[1..] {
            let Op::ConditionalPauli(corrections) = op else {
                panic!("expected merged corrections");
            };
            assert_eq!(
                corrections[0].control,
                plan.measurements[&instance_measurement(0, 0)]
            );
        }
    }

    #[test]
    fn multi_instance_merge_rejects_one_body_in_different_tuples() {
        let mut first = CoordCircuit::new();
        let shared = first.add_body(CircuitBody::new());
        append_repeats(&mut first, &[(shared, 2), (shared, 2)]);
        let mut second = CoordCircuit::new();
        let left = second.add_body(CircuitBody::new());
        let right = second.add_body(CircuitBody::new());
        append_repeats(&mut second, &[(left, 2), (right, 2)]);
        let circuits = [first, second];

        assert_eq!(
            merge_instantiated_circuits_with_body_aliases(&circuits)
                .err()
                .unwrap(),
            NodeTemplateInstanceMergeError::BodyMergeConflict {
                circuit_index: 0,
                body: shared,
            }
        );
    }

    #[test]
    fn instantiate_circuit_aliases_duplicate_measurements_in_one_segment() {
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut templates = BloqTemplatePool::new();
        let template_id = templates.insert(BloqTemplate::new(template_circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), template_id, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), template_id, ivec2(0, 0)),
        ];

        let instantiated = node.emission_plan(&templates).unwrap();

        assert_eq!(instantiated.circuit.meas_registry().records().len(), 1);
        assert_eq!(
            instantiated.measurements[&instance_measurement(0, 0)],
            instantiated.measurements[&instance_measurement(1, 0)]
        );
    }

    #[test]
    fn conditional_membership_preserves_common_records_and_selected_aliases() {
        use crate::{Bloq, QuantumGuard};

        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), template, IVec2::ZERO),
            TemplateInstance::new(TemplateInstanceId(1), template, IVec2::ZERO),
        ];
        node.expect_quantum_mut().guards.push(QuantumGuard {
            input: 7,
            instances: vec![TemplateInstanceId(0)],
            ..Default::default()
        });
        assert!(matches!(
            node.emission_plan(bloq.templates()),
            Err(NodeTemplateInstanceMergeError::MembershipSelectionRequired)
        ));
        assert!(matches!(
            node.select_quantum_members(|_| None),
            Err(NodeTemplateInstanceMergeError::MissingMembershipInput(7))
        ));
        let id = bloq.add_node(node);
        for restored in [
            Bloq::from_text(&bloq.to_text()).unwrap(),
            Bloq::from_binary(&bloq.to_binary()).unwrap(),
        ] {
            for enabled in [false, true, false] {
                let selected = restored[id]
                    .select_quantum_members(|slot| {
                        assert_eq!(slot, 7);
                        Some(enabled)
                    })
                    .unwrap();
                let plan = selected.emission_plan(restored.templates()).unwrap();
                assert_eq!(plan.circuit.meas_registry().records().len(), 1);
                assert!(plan.measurements.contains_key(&instance_measurement(1, 0)));
                assert_eq!(
                    plan.measurements.contains_key(&instance_measurement(0, 0)),
                    enabled
                );
                if enabled {
                    assert_eq!(
                        plan.measurements[&instance_measurement(0, 0)],
                        plan.measurements[&instance_measurement(1, 0)]
                    );
                }
            }
            assert_eq!(
                restored[id].expect_quantum().instances.len(),
                2,
                "selection never mutates the shared registry"
            );
        }
    }

    #[test]
    fn conditional_membership_rejects_missing_or_duplicate_registrations() {
        use crate::QuantumGuard;

        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().guards.push(QuantumGuard {
            input: 0,
            instances: vec![TemplateInstanceId(8)],
            ..Default::default()
        });
        assert!(matches!(
            node.select_quantum_members(|_| Some(false)),
            Err(NodeTemplateInstanceMergeError::InvalidMembership(_))
        ));
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(8),
                crate::TemplateId(0),
                IVec2::ZERO,
            ));
        node.expect_quantum_mut().guards[0]
            .instances
            .push(TemplateInstanceId(8));
        assert!(matches!(
            node.select_quantum_members(|_| Some(true)),
            Err(NodeTemplateInstanceMergeError::InvalidMembership(_))
        ));
    }

    #[test]
    fn bundle_only_registration_selects_whole_use_and_checks_false_indices() {
        use crate::{DetectorBundleId, DetectorBundleUse, QuantumGuard};

        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .detector_bundles
            .push(DetectorBundleUse {
                bundle: DetectorBundleId(0),
                instances: Vec::new(),
                offset: IVec2::ZERO,
            });
        node.expect_quantum_mut().guards.push(QuantumGuard {
            input: 3,
            detector_bundles: vec![0],
            ..Default::default()
        });
        assert!(
            node.select_quantum_members(|_| Some(false))
                .unwrap()
                .expect_quantum()
                .detector_bundles
                .is_empty()
        );
        assert_eq!(
            node.select_quantum_members(|_| Some(true))
                .unwrap()
                .expect_quantum()
                .detector_bundles
                .len(),
            1
        );
        node.expect_quantum_mut().guards[0].detector_bundles.push(9);
        assert!(matches!(
            node.select_quantum_members(|_| Some(false)),
            Err(NodeTemplateInstanceMergeError::InvalidMembership(_))
        ));
    }

    #[test]
    fn measurement_aliases_rewrite_merged_correction_controls() {
        fn measured_correction(target: glam::IVec2) -> CoordCircuit {
            let measured = ivec2(0, 0);
            let mut circuit = CoordCircuit::new();
            let measurement = circuit.measure(PauliBasis::Z, [measured])[0];
            circuit
                .body_mut(circuit.entry_body())
                .unwrap()
                .ops_mut()
                .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                    pauli: PauliBasis::X,
                    control: measurement,
                    target,
                }]));
            circuit
        }

        let mut templates = BloqTemplatePool::new();
        let first = templates.insert(BloqTemplate::new(measured_correction(ivec2(1, 0))));
        let second = templates.insert(BloqTemplate::new(measured_correction(ivec2(2, 0))));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = vec![
            TemplateInstance::new(TemplateInstanceId(0), first, ivec2(0, 0)),
            TemplateInstance::new(TemplateInstanceId(1), second, ivec2(0, 0)),
        ];

        let plan = node.emission_plan(&templates).unwrap();
        let controls = plan
            .circuit
            .body(plan.circuit.entry_body())
            .unwrap()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::ConditionalPauli(corrections) => Some(corrections[0].control),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(plan.circuit.meas_registry().records().len(), 1);
        assert_eq!(controls, vec![0, 0]);
    }
}
