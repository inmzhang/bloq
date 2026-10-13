use std::collections::BTreeSet;

use bloq_circuit::{BodyId, DetectorTerm, LoopStateId, Op};
use bloq_utils::boolean::BooleanDecisionDiagram;
use glam::{IVec2, IVec3};
use thiserror::Error;

use crate::instantiation::{InstantiationOptions, MeasurementSet, NodeEmissionPlan};
use crate::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, BloqTemplatePool, ClassicalExpr,
    ClassicalNode, FxMap, FxSet, InstanceMeasurement, LevelPath, NodeTemplateInstanceMergeError,
    RegionNode, SubGraph, TemplateDetectorScope, TemplateId, TemplateInstanceId, ValueRole,
};

/// A well-formedness violation or resource exhaustion from [`Bloq::validate`].
///
/// Well-formedness messages begin with their normative rule id. Resource
/// failures are explicitly identified and can be queried without parsing text.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BloqValidationError {
    /// A detector bundle definition or use is malformed.
    #[error("WF-6: {0}")]
    DetectorBundle(#[from] crate::DetectorBundleError),
    /// A bundle use reads a bound owner after the consumer or without a path.
    #[error("WF-7: node {node:?} cannot read bundle owner instance {instance:?}")]
    UnavailableDetectorBundleOwner {
        /// Consumer node.
        node: BloqNodeId,
        /// Bound record owner.
        instance: TemplateInstanceId,
    },
    /// A bundle parity contains an unresolvable template loop state.
    #[error("WF-6: detector bundle {bundle:?} references loop state {state:?}")]
    DetectorBundleLoopStateUnsupported {
        /// Bundle definition.
        bundle: crate::DetectorBundleId,
        /// Unresolvable state.
        state: LoopStateId,
    },
    /// A bundle signature names a template missing from the pool.
    #[error("WF-4: detector bundle {bundle:?} references unknown template {template:?}")]
    UnknownDetectorBundleTemplate {
        /// Bundle definition.
        bundle: crate::DetectorBundleId,
        /// Missing template.
        template: TemplateId,
    },
    /// Boolean validation exceeded its resource budget.
    #[error("resource limit: {0}")]
    BooleanResource(#[from] bloq_utils::boolean::BooleanResourceError),
    /// A node reads a measurement from a later node.
    #[error("WF-7: node {node:?} references measurement {measurement} from later node {owner:?}")]
    FutureMeasurement {
        /// Consumer node.
        node: BloqNodeId,
        /// Measurement owner.
        owner: BloqNodeId,
        /// Owner-local measurement id.
        measurement: u32,
    },
    /// A node reads a measurement from an unrelated node.
    #[error(
        "WF-7: node {node:?} references measurement {measurement} from unrelated node {owner:?}"
    )]
    UnorderedNodeMeasurement {
        /// Consumer node.
        node: BloqNodeId,
        /// Measurement owner.
        owner: BloqNodeId,
        /// Owner-local measurement id.
        measurement: u32,
    },
    /// A node references an absent template.
    #[error("WF-4: node {node:?} references unknown template {template:?}")]
    UnknownTemplate {
        /// Referencing node.
        node: BloqNodeId,
        /// Missing template.
        template: TemplateId,
    },
    /// Pipe padding references an absent template.
    #[error("WF-4: pipe padding references unknown template {template:?}")]
    UnknownPipePaddingTemplate {
        /// Missing template.
        template: TemplateId,
    },
    /// A one-round padding template contains a repeat.
    #[error("WF-4: one-round padding template {template:?} contains a top-level REPEAT")]
    PipePaddingOneRoundLooped {
        /// Invalid template.
        template: TemplateId,
    },
    /// A looped padding template lacks exactly one repeat.
    #[error("WF-4: looped padding template {template:?} must contain exactly one top-level REPEAT")]
    PipePaddingLoopedMalformed {
        /// Invalid template.
        template: TemplateId,
    },
    /// A template-instance id is duplicated.
    #[error("WF-5: template instance id {instance:?} is used more than once")]
    DuplicateTemplateInstance {
        /// Duplicated instance.
        instance: TemplateInstanceId,
    },
    /// A node references an absent instance.
    #[error("WF-6: node {node:?} references unknown template instance {instance:?}")]
    UnknownTemplateInstance {
        /// Referencing node.
        node: BloqNodeId,
        /// Missing instance.
        instance: TemplateInstanceId,
    },
    /// A logical input references an absent instance.
    #[error(
        "WF-6: logical input ({},{},{}) references unknown template instance {instance:?}",
        port.x, port.y, port.z
    )]
    LogicalInputUnknownInstance {
        /// Source port.
        port: IVec3,
        /// Missing instance.
        instance: TemplateInstanceId,
    },
    /// A logical output references a missing owner instance.
    #[error("WF-6: logical output at {port:?} references unknown instance {instance:?}")]
    LogicalOutputUnknownInstance {
        /// Source port.
        port: IVec3,
        /// Missing instance.
        instance: TemplateInstanceId,
    },
    /// A node parity references an unavailable instance measurement.
    #[error(
        "WF-6: node {node:?} references unavailable measurement {measurement} in instance {instance:?}"
    )]
    UnknownInstanceMeasurement {
        /// Referencing node.
        node: BloqNodeId,
        /// Referenced instance.
        instance: TemplateInstanceId,
        /// Missing measurement id.
        measurement: u32,
    },
    /// A template side table references an unreachable body.
    #[error("WF-6: template {template:?} references unreachable repeat body {body:?}")]
    UnknownTemplateBody {
        /// Referencing template.
        template: TemplateId,
        /// Missing body.
        body: BodyId,
    },
    /// A template side table references an unavailable measurement.
    #[error("WF-6: template {template:?} references unavailable measurement {measurement}")]
    UnknownTemplateMeasurement {
        /// Referencing template.
        template: TemplateId,
        /// Missing measurement.
        measurement: u32,
    },
    /// A template side table references an unavailable loop state.
    #[error("WF-6: template {template:?} references unavailable loop state {state:?}")]
    UnknownTemplateLoopState {
        /// Referencing template.
        template: TemplateId,
        /// Missing loop state.
        state: LoopStateId,
    },
    /// A node parity contains a template-local loop state.
    #[error("WF-6: node {node:?} parity references loop state {state:?}")]
    NodeLoopStateUnsupported {
        /// Referencing node.
        node: BloqNodeId,
        /// Unsupported loop state.
        state: LoopStateId,
    },
    /// A template declares one loop-state id twice.
    #[error("WF-6: template {template:?} declares loop state {state:?} more than once")]
    DuplicateTemplateLoopState {
        /// Owning template.
        template: TemplateId,
        /// Duplicated state.
        state: LoopStateId,
    },
    /// A node's template instances cannot be merged.
    #[error("WF-9: node {node:?} {context}: {source}", context = if .source.is_resource_limited() {
        "hit a resource limit while merging template instances"
    } else {
        "template instances do not merge"
    })]
    InvalidInstanceMergeStructure {
        /// Invalid quantum node.
        node: BloqNodeId,
        /// Merge failure.
        source: NodeTemplateInstanceMergeError,
    },
    /// A template circuit is structurally invalid.
    #[error("WF-9: template {template:?} {context}: {source}", context = if .source.is_resource_limited() {
        "hit a resource limit while checking circuit structure"
    } else {
        "has invalid circuit structure"
    })]
    InvalidTemplateCircuit {
        /// Invalid template.
        template: TemplateId,
        /// Circuit failure.
        source: NodeTemplateInstanceMergeError,
    },
    /// A consumer has two producers on one value slot.
    #[error("WF-2: node {node:?} has multiple value producers on input slot {slot}")]
    DuplicateValueSlot {
        /// Consumer node.
        node: BloqNodeId,
        /// Duplicated slot.
        slot: u32,
    },
    /// A quantum face fans out.
    #[error("WF-3: node {node:?} has multiple quantum outputs for one face")]
    QuantumEdgeFanout {
        /// Fan-out node.
        node: BloqNodeId,
    },
    /// A region predicate slot has no producer.
    #[error("WF-10: region {node:?} has no value input for predicate slot {slot}")]
    MissingRegionSelectorInput {
        /// Region node.
        node: BloqNodeId,
        /// Missing slot.
        slot: u32,
    },
    /// The graph is cyclic.
    #[error("WF-1: graph contains a cycle")]
    CyclicGraph,
    /// A classical expression slot has no producer.
    #[error("WF-11: classical node {node:?} has no value input for slot {slot}")]
    MissingClassicalInput {
        /// Consumer node.
        node: BloqNodeId,
        /// Missing slot.
        slot: u32,
    },
    /// A value edge feeds an unread classical slot.
    #[error("WF-11: classical node {node:?} does not read value input slot {slot}")]
    UnusedClassicalInput {
        /// Consumer node.
        node: BloqNodeId,
        /// Unused slot.
        slot: u32,
    },
    /// A discard occurs inside a retry body.
    #[error("WF-8: Discard node {node:?} is inside a RepeatUntilSuccess body")]
    DiscardInsideRepeatUntilSuccess {
        /// Invalid discard.
        node: BloqNodeId,
    },
    /// A retry region is nested.
    #[error("WF-8: RepeatUntilSuccess node {node:?} is nested")]
    NestedRepeatUntilSuccess {
        /// Nested region.
        node: BloqNodeId,
    },
    /// An observable index is duplicated in the program.
    #[error("WF-13: Observable node {node:?} reuses index {index}")]
    DuplicateObservableIndex {
        /// Duplicate observable node.
        node: BloqNodeId,
        /// Duplicate index.
        index: u32,
    },
    /// Restart syndromes occur outside a retry body.
    #[error("WF-8: node {node:?} has restart syndromes outside RepeatUntilSuccess")]
    RestartOutsideRepeatUntilSuccess {
        /// Invalid quantum node.
        node: BloqNodeId,
    },
    /// An output-frame stamp is not on a top-level compute.
    #[error(
        "WF-15: output frame ({},{},{}) references non-top-level-Compute node {node:?}",
        port.x, port.y, port.z
    )]
    OutputFrameNotCompute {
        /// Output port.
        port: IVec3,
        /// Invalid frame node.
        node: BloqNodeId,
    },
    /// An output lacks one frame basis stamp.
    #[error(
        "WF-15: output frame ({},{},{}) needs one X and one Z stamp",
        port.x, port.y, port.z
    )]
    OutputFrameIncomplete {
        /// Incomplete output port.
        port: IVec3,
    },
    /// A quantum node receives a value edge.
    #[error("WF-12: quantum node {node:?} has value input slot {slot}")]
    UnusedQuantumValueInput {
        /// Quantum node.
        node: BloqNodeId,
        /// Invalid input slot.
        slot: u32,
    },
    /// A node's classical activation is invalid.
    ///
    /// Unlike its siblings this variant serves several rules — a separate
    /// activation slot is WF-11, whole-node activation of a quantum node is
    /// WF-12, and the activation's edge role is WF-16 — so it carries the one
    /// it cites rather than hard-coding a prefix.
    #[error("{rule}: invalid classical activation on node {node:?}: {reason}")]
    InvalidActivation {
        /// Activated node.
        node: BloqNodeId,
        /// Normative rule violated.
        rule: &'static str,
        /// Contract violation.
        reason: &'static str,
    },
    /// A quantum seam guard is invalid.
    #[error("WF-3: invalid quantum seam guard {guard:?} before node {node:?}")]
    InvalidSeamGuard {
        /// Guarded target.
        node: BloqNodeId,
        /// Invalid guard producer.
        guard: crate::ValueRef,
    },
    /// A selected node may reference a deselected instance.
    #[error("WF-7: active node {node:?} can reference deselected instance {instance:?}")]
    ConditionalReferenceUnavailable {
        /// Referencing node.
        node: BloqNodeId,
        /// Conditionally unavailable instance.
        instance: TemplateInstanceId,
    },
    /// A classical expression reads one slot more than once.
    #[error("WF-11: node {node:?} reads input slot {slot} more than once")]
    DuplicateExprInput {
        /// Consumer node.
        node: BloqNodeId,
        /// Repeated slot.
        slot: u32,
    },
    /// A value edge comes from a node that produces no value.
    #[error("WF-16: node {node:?} has a value edge from non-value producer {producer:?}")]
    InvalidValueProducer {
        /// Consumer node.
        node: BloqNodeId,
        /// Invalid producer.
        producer: BloqNodeId,
    },
    /// An observable-fold edge has invalid endpoint kinds.
    #[error(
        "WF-16: observable-fold edge {producer:?} -> {node:?} must carry a bit leaf into an Observable"
    )]
    InvalidObservableFold {
        /// Consumer node.
        node: BloqNodeId,
        /// Invalid producer.
        producer: BloqNodeId,
    },
    /// An output has two frame stamps for one basis.
    #[error(
        "WF-15: output frame ({},{},{}) has duplicate basis stamps",
        port.x, port.y, port.z
    )]
    DuplicateOutputFrameStamp {
        /// Output port.
        port: IVec3,
    },
    /// A retry region's explicit restart source is invalid.
    #[error(
        "WF-10: RepeatUntilSuccess region {node:?} has invalid restart source {restart_source:?}"
    )]
    InvalidRestartSource {
        /// Retry region.
        node: BloqNodeId,
        /// Invalid body-local producer.
        restart_source: crate::ValueRef,
    },
    /// A declared bit result is invalid.
    #[error("WF-17: declared result {node:?} must name a live bit producer")]
    InvalidValueOutput {
        /// Invalid result node.
        node: BloqNodeId,
    },
    /// A declared boundary output is invalid.
    #[error("WF-17: declared boundary output {node:?} must name a live Observable or region")]
    InvalidBoundaryOutput {
        /// Invalid boundary node.
        node: BloqNodeId,
    },
    /// A boundary output is declared twice.
    #[error("WF-17: boundary output {node:?} is declared more than once")]
    DuplicateBoundaryOutput {
        /// Duplicated boundary node.
        node: BloqNodeId,
    },
    /// A quantum timeline violates its projection contract.
    #[error("WF-20: quantum node {node:?} has an invalid source-layer timeline")]
    InvalidQuantumTimeline {
        /// Invalid quantum node.
        node: BloqNodeId,
    },
}

impl BloqValidationError {
    /// Whether validation stopped at a resource allowance. Such a failure is
    /// not evidence that the program violates the IR's well-formedness rules.
    #[must_use]
    pub fn is_resource_limited(&self) -> bool {
        match self {
            Self::BooleanResource(_) => true,
            Self::InvalidInstanceMergeStructure { source, .. }
            | Self::InvalidTemplateCircuit { source, .. } => source.is_resource_limited(),
            _ => false,
        }
    }
}

/// Quantum emission stages keyed by scoped node address.
///
/// Static stages retain an exact emission plan. Conditional stages are marked
/// for selected materialization. Only [`Bloq::validate_with_plans`] certifies
/// their full Boolean domain; [`Bloq::emission_plans`] does not.
#[derive(Debug, Clone, Default)]
pub struct ValidatedPlans {
    plans: FxMap<LevelPath, FxMap<BloqNodeId, NodeEmissionPlan>>,
    memberships: FxMap<LevelPath, FxSet<BloqNodeId>>,
}

impl ValidatedPlans {
    fn insert(&mut self, path: LevelPath, node: BloqNodeId, plan: NodeEmissionPlan) {
        self.plans.entry(path).or_default().insert(node, plan);
    }

    /// A static stage's emission plan. Conditional stages use [`Self::has_membership`].
    #[must_use]
    pub fn get(&self, path: &LevelPath, node: BloqNodeId) -> Option<&NodeEmissionPlan> {
        self.plans.get(path)?.get(&node)
    }

    /// Whether this conditional stage requires selected materialization.
    pub fn has_membership(&self, path: &LevelPath, node: BloqNodeId) -> bool {
        self.memberships
            .get(path)
            .is_some_and(|nodes| nodes.contains(&node))
    }

    /// Every static `((path, node), plan)` entry, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = ((&LevelPath, BloqNodeId), &NodeEmissionPlan)> {
        self.plans
            .iter()
            .flat_map(|(path, plans)| plans.iter().map(move |(&node, plan)| ((path, node), plan)))
    }

    /// The number of static and conditional quantum stages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plans.values().map(FxMap::len).sum::<usize>()
            + self.memberships.values().map(FxSet::len).sum::<usize>()
    }

    /// Whether the program has no quantum-node plans.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty() && self.memberships.is_empty()
    }
}

fn valid_value_source(graph: &SubGraph, value: crate::ValueRef) -> bool {
    graph
        .node(value.node)
        .is_some_and(|node| match value.output {
            crate::ObservableOutput::Flip => matches!(
                node.try_classical(),
                Some(ClassicalNode::Observable { index: Some(_), .. })
            ),
            crate::ObservableOutput::Corrected => {
                node.try_region().is_some()
                    || matches!(
                        node.try_classical(),
                        Some(ClassicalNode::Compute { .. } | ClassicalNode::Observable { .. })
                    )
            }
        })
}

fn consumes_observable_folds(kind: &BloqNodeKind) -> bool {
    matches!(kind.try_classical(), Some(ClassicalNode::Observable { .. }))
}

/// Checks unique value-input slots, linear quantum faces, and valid value-edge sources.
// Spec rules WF-2/WF-3 (edge cardinality) and WF-16 (producer kind).
fn validate_edge_cardinality(
    graph: &SubGraph,
    predicates: &mut crate::membership::PredicateAnalysis<'_>,
) -> Result<(), BloqValidationError> {
    for id in graph.node_ids() {
        let mut seen_slots = BTreeSet::new();
        for input in graph.incoming(id).filter_map(|edge| match edge.edge {
            BloqEdge::Value { slot, role, output } => Some(crate::ValueInput {
                slot: *slot,
                producer: edge.source,
                role,
                output: Some(*output),
            }),
            BloqEdge::Compose { slot, role } => Some(crate::ValueInput {
                slot: *slot,
                producer: edge.source,
                role,
                output: None,
            }),
            _ => None,
        }) {
            if !seen_slots.insert(input.slot) {
                return Err(BloqValidationError::DuplicateValueSlot {
                    node: id,
                    slot: input.slot,
                });
            }
            let consumer = &graph[id];
            let producer = &graph[input.producer].kind;
            let valid_source = if let Some(output) = input.output {
                valid_value_source(
                    graph,
                    crate::ValueRef {
                        node: input.producer,
                        output,
                    },
                )
            } else {
                consumer.activation != Some(input.slot)
                    && consumes_observable_folds(&consumer.kind)
                    && (matches!(
                        producer.try_classical(),
                        Some(ClassicalNode::Observable { .. })
                    ) || matches!(producer, BloqNodeKind::Region(_)))
            };
            if !valid_source {
                return Err(BloqValidationError::InvalidValueProducer {
                    node: id,
                    producer: input.producer,
                });
            }
            if consumer.activation == Some(input.slot) && input.role != &ValueRole::Data {
                return Err(BloqValidationError::InvalidActivation {
                    node: id,
                    rule: "WF-16",
                    reason: "activation is not an observable fold",
                });
            }
            // WF-16: fold roles carry runtime bits, never structural recipes.
            if matches!(
                input.role,
                ValueRole::FeedbackFold { .. } | ValueRole::ReadoutFold
            ) && (!consumes_observable_folds(&consumer.kind) || input.output.is_none())
            {
                return Err(BloqValidationError::InvalidObservableFold {
                    node: id,
                    producer: input.producer,
                });
            }
        }
        let mut seen_faces: FxMap<[i32; 3], Vec<Option<crate::ValueRef>>> = FxMap::default();
        for edge in graph.outgoing(id) {
            let BloqEdge::Quantum(quantum) = edge.edge else {
                continue;
            };
            if let Some(guard) = quantum.guard
                && (!valid_value_source(graph, guard) || !graph.has_path(guard.node, edge.target))
            {
                return Err(BloqValidationError::InvalidSeamGuard {
                    node: edge.target,
                    guard,
                });
            }
            for seam in edge.edge.pipes() {
                let guards = seen_faces.entry(seam.pipe.src.to_array()).or_default();
                for &other in guards.iter() {
                    if predicates
                        .overlap_values(other, quantum.guard)
                        .map_err(|error| BloqValidationError::InvalidInstanceMergeStructure {
                            node: id,
                            source: error,
                        })?
                    {
                        return Err(BloqValidationError::QuantumEdgeFanout { node: id });
                    }
                }
                guards.push(quantum.guard);
            }
        }
    }
    Ok(())
}

impl Bloq {
    /// Validates IR well-formedness, including nested region bodies.
    ///
    /// Same-layer qubit overlap depends on source `BlockGraph` soft corridors
    /// and is checked by `bloq_compile` lowering.
    ///
    /// WF-9 checks every reachable selected merge. [`Self::validate_with_plans`]
    /// additionally returns static plans and certified conditional stages.
    /// This entry point releases each merged circuit after checking its node,
    /// bounding temporary circuit storage by the largest quantum stage. Within
    /// one noiseless validation, successful unguarded merges are reused at the
    /// same relative placements; absolute coordinates are still checked.
    ///
    /// # Errors
    ///
    /// Returns [`BloqValidationError`] for any violated IR well-formedness or
    /// bounded-resource rule.
    pub fn validate(&self) -> Result<(), BloqValidationError> {
        self.validate_with_options(&InstantiationOptions::default())
    }

    /// Validate with explicit noise and resource limits, releasing emission
    /// plans after each node instead of retaining them.
    ///
    /// # Errors
    ///
    /// Returns [`BloqValidationError`] for invalid IR, merge, noise, or
    /// exhausted resources.
    pub fn validate_with_options(
        &self,
        options: &InstantiationOptions<'_>,
    ) -> Result<(), BloqValidationError> {
        self.validate_inner(options, &mut None)
    }

    /// [`Self::validate`], additionally returning static [`NodeEmissionPlan`]s
    /// and certified conditional stages, keyed by scoped address (see
    /// [`ValidatedPlans`]). Plans are instantiated with `options`.
    ///
    /// Applies the same well-formedness rules as [`Self::validate`], charging
    /// work for every materialized plan. SEM-MERGE applies noise once to the
    /// *post*-merge circuit, and the returned plans carry the caller's `options`.
    /// Conditional membership uses pairwise Boolean overlap checks.
    /// A malformed [`NoiseModel`](crate::circuit::NoiseModel) also changes the outcome: its
    /// probabilities are only checked while materializing noise, so a bad model
    /// surfaces here (and only under noisy `options`) as
    /// [`InvalidInstanceMergeStructure`](BloqValidationError::InvalidInstanceMergeStructure)
    /// at the first quantum node.
    ///
    /// # Errors
    ///
    /// Returns [`BloqValidationError`] under the same conditions as
    /// [`Self::validate_with_options`].
    ///
    /// # Panics
    ///
    /// Panics only if the internal plan collector is unexpectedly absent.
    pub fn validate_with_plans(
        &self,
        options: &InstantiationOptions<'_>,
    ) -> Result<ValidatedPlans, BloqValidationError> {
        let mut plans = Some(ValidatedPlans::default());
        self.validate_inner(options, &mut plans)?;
        Ok(plans.expect("emission plan collection was requested"))
    }

    /// Materialize static quantum stages without a full well-formedness audit.
    /// Guarded stages are marked for the consumer to select and materialize;
    /// no Boolean proof of their entire choice domain is performed here.
    ///
    /// Required static merges and their noise application still fail here.
    /// A consumer must report failures while materializing selected guarded
    /// stages. Call [`Self::validate_with_plans`] explicitly for the complete
    /// all-choice IR audit as well as plans.
    ///
    /// # Errors
    ///
    /// [`InvalidInstanceMergeStructure`](BloqValidationError::InvalidInstanceMergeStructure)
    /// if a static quantum node's instances cannot merge (SEM-MERGE).
    ///
    /// # Caution
    ///
    /// This is not a substitute for [`Self::validate`]. Malformed IR outside
    /// the materialized stages may not be reported by this method.
    pub fn emission_plans(
        &self,
        options: &InstantiationOptions<'_>,
    ) -> Result<ValidatedPlans, BloqValidationError> {
        let mut plans = ValidatedPlans::default();
        for (path, level) in self.levels() {
            for (id, node) in level.nodes() {
                let Some(quantum) = node.try_quantum() else {
                    continue;
                };
                if !quantum.guards.is_empty() {
                    plans
                        .memberships
                        .entry(path.clone())
                        .or_default()
                        .insert(id);
                    continue;
                }
                let plan = node
                    .emission_plan_with_options(self.templates(), *options)
                    .map_err(
                        |source| BloqValidationError::InvalidInstanceMergeStructure {
                            node: id,
                            source,
                        },
                    )?;
                plans.insert(path.clone(), id, plan);
            }
        }
        Ok(plans)
    }

    /// Shared checks; only consumers requesting emission retain WF-9 plans.
    fn validate_inner(
        &self,
        options: &InstantiationOptions<'_>,
        plans: &mut Option<ValidatedPlans>,
    ) -> Result<(), BloqValidationError> {
        let mut work = BooleanDecisionDiagram::with_limits(options.boolean_limits());
        work.charge(
            self.logical_inputs()
                .len()
                .saturating_add(self.logical_outputs().len()),
        )?;
        // Charge source containers before whole-program maps reserve their entries.
        for (_, level) in self.levels() {
            work.charge(level.node_count().saturating_add(level.edge_count()))?;
            for (_, quantum) in level.quantum_nodes() {
                work.charge(
                    quantum
                        .instances
                        .len()
                        .saturating_add(quantum.guards.len())
                        .saturating_add(quantum.detectors.len())
                        .saturating_add(quantum.detector_bundles.len())
                        .saturating_add(quantum.restarts.len()),
                )?;
            }
        }
        // Templates are shared, so validate the pool once.
        let template_measurements = self
            .templates()
            .iter()
            .map(|(template_id, template)| validate_template(template_id, template, &mut work))
            .collect::<Result<Vec<_>, _>>()?;
        // A definition is shared by every use. Check its record coordinates
        // once; uses below validate only their bindings and activation.
        for (bundle_id, bundle) in self.detector_bundles().iter() {
            work.charge(
                bundle
                    .detectors()
                    .len()
                    .saturating_add(bundle.owner_templates().len()),
            )?;
            for &template in bundle.owner_templates() {
                if self.templates().get(template).is_none() {
                    return Err(BloqValidationError::UnknownDetectorBundleTemplate {
                        bundle: bundle_id,
                        template,
                    });
                }
            }
            for row in bundle.detectors() {
                work.charge(row.parity.terms().len().saturating_add(1))?;
            }
            bundle.used_owners()?;
            for row in bundle.detectors() {
                for term in row.parity.terms() {
                    let term = match term {
                        DetectorTerm::Measurement(term) => term,
                        DetectorTerm::LoopState(state) => {
                            return Err(BloqValidationError::DetectorBundleLoopStateUnsupported {
                                bundle: bundle_id,
                                state: *state,
                            });
                        }
                    };
                    let template = bundle.owner_templates()[term.owner as usize];
                    if !template_measurements
                        .get(template.0 as usize)
                        .is_some_and(|records| records.contains(term.measurement))
                    {
                        return Err(BloqValidationError::UnknownTemplateMeasurement {
                            template,
                            measurement: term.measurement,
                        });
                    }
                }
            }
        }
        let all_instances = collect_all_instances(self)?;
        for (_, level) in self.levels() {
            for (_, quantum) in level.quantum_nodes() {
                work.charge(quantum.detector_bundles.len())?;
                for use_ in &quantum.detector_bundles {
                    work.charge(use_.instances.len())?;
                }
                self.check_detector_bundle_bindings(quantum, |instance| {
                    all_instances.get(&instance).copied()
                })?;
            }
        }

        // Memory-round insertion requires an unrolled template and one with a
        // single top-level REPEAT.
        for entry in self.pipe_padding() {
            work.charge(1)?;
            for template in [entry.one_round, entry.looped] {
                if self.templates().get(template).is_none() {
                    return Err(BloqValidationError::UnknownPipePaddingTemplate { template });
                }
            }
            if !self.templates()[entry.one_round]
                .circuit
                .entry_top_level_repeats()
                .is_empty()
            {
                return Err(BloqValidationError::PipePaddingOneRoundLooped {
                    template: entry.one_round,
                });
            }
            if self.templates()[entry.looped]
                .circuit
                .entry_top_level_repeats()
                .len()
                != 1
            {
                return Err(BloqValidationError::PipePaddingLoopedMalformed {
                    template: entry.looped,
                });
            }
        }

        // Cross-region references use program-global instance ids (WF-5), while
        // read-after-measure order remains graph-local.
        if let Some(input) = self
            .logical_inputs()
            .iter()
            .find(|input| !all_instances.contains_key(&input.instance))
        {
            return Err(BloqValidationError::LogicalInputUnknownInstance {
                port: input.port,
                instance: input.instance,
            });
        }

        if let Some(output) = self
            .logical_outputs()
            .iter()
            .find(|output| !all_instances.contains_key(&output.instance))
        {
            return Err(BloqValidationError::LogicalOutputUnknownInstance {
                port: output.port,
                instance: output.instance,
            });
        }

        // U10 requires one X/Z stamp pair on top-level Compute nodes.
        let mut frame_stamps: FxMap<IVec3, (bool, bool)> = FxMap::default();
        for (path, level) in self.levels() {
            for (id, weight) in level.nodes() {
                if let Some(timeline) = weight
                    .try_quantum()
                    .and_then(|quantum| quantum.timeline.as_ref())
                    && !valid_quantum_timeline(timeline, &weight.provenance)
                {
                    return Err(BloqValidationError::InvalidQuantumTimeline { node: id });
                }
                let crate::NodeProvenance::OutputFrame { port, basis } = &weight.provenance else {
                    continue;
                };
                if !path.segments().is_empty()
                    || !matches!(weight.try_classical(), Some(ClassicalNode::Compute { .. }))
                {
                    return Err(BloqValidationError::OutputFrameNotCompute {
                        port: *port,
                        node: id,
                    });
                }
                let entry = frame_stamps.entry(*port).or_default();
                let seen = match basis {
                    crate::Basis::X => &mut entry.0,
                    crate::Basis::Z => &mut entry.1,
                };
                if std::mem::replace(seen, true) {
                    return Err(BloqValidationError::DuplicateOutputFrameStamp { port: *port });
                }
            }
        }
        let incomplete_stamp_pair = frame_stamps
            .iter()
            .filter(|(_, (x, z))| x != z)
            .map(|(&port, _)| port);
        let missing_logical_output_pair = self.logical_outputs().iter().filter_map(|logical| {
            (frame_stamps.get(&logical.port).copied() != Some((true, true))).then_some(logical.port)
        });
        let incomplete = incomplete_stamp_pair
            .chain(missing_logical_output_pair)
            .min_by_key(|port| (port.x, port.y, port.z));
        if let Some(port) = incomplete {
            return Err(BloqValidationError::OutputFrameIncomplete { port });
        }

        // Observable indices share one global emission space (§4.2/U17a).
        let mut observable_indices = FxSet::default();
        validate_graph(
            self.top(),
            &LevelPath::default(),
            self.templates(),
            self.detector_bundles(),
            &all_instances,
            &template_measurements,
            &mut observable_indices,
            *options,
            plans,
            &mut MergeProofCache::default(),
            &mut work,
            ValidationCtx {
                in_rus: false,
                // Skip per-node scans when no template can restart.
                pool_has_restarts: self
                    .templates()
                    .iter()
                    .any(|(_, template)| !template.restarts.is_empty()),
            },
        )?;
        self.validate_membership_references(*options, &mut work)?;
        Ok(())
    }

    fn validate_membership_references(
        &self,
        options: InstantiationOptions<'_>,
        work: &mut BooleanDecisionDiagram,
    ) -> Result<(), BloqValidationError> {
        let external_instances: FxSet<_> = self
            .logical_inputs()
            .iter()
            .map(|input| input.instance)
            .chain(self.logical_outputs().iter().map(|output| output.instance))
            .collect();
        let mut nested_cut_owners = FxMap::default();
        let mut availability = FxMap::default();
        for (path, level) in self.levels() {
            for (id, quantum) in level.quantum_nodes() {
                if !path.is_top_level() {
                    for instance in &quantum.instances {
                        if external_instances.contains(&instance.id) {
                            nested_cut_owners.insert(instance.id, path.clone());
                        }
                    }
                }
                for guard in &quantum.guards {
                    let producer = level
                        .value_inputs(id)
                        .find(|input| input.slot == guard.input)
                        .expect("validated registration input")
                        .value_ref()
                        .expect("runtime input");
                    for &instance in &guard.instances {
                        availability.insert(instance, (path.clone(), producer));
                    }
                }
            }
        }
        let mut predicates = FxMap::default();
        // WF-7: external metadata is unconditional, so its cut owner must
        // exist in every reachable selection too.
        for instance in self
            .logical_inputs()
            .iter()
            .map(|input| input.instance)
            .chain(self.logical_outputs().iter().map(|output| output.instance))
        {
            if let Some(owner_path) = nested_cut_owners.get(&instance) {
                let mut path = LevelPath::default();
                for segment in owner_path.segments() {
                    let level = self.level_at(&path).expect("cut owner ancestor");
                    let node = &level[segment.region];
                    let relation = predicates.entry(path.clone()).or_insert_with(|| {
                        crate::membership::PredicateAnalysis::with_limits(
                            level,
                            options.boolean_limits(),
                        )
                    });
                    let invalid = |source| BloqValidationError::InvalidInstanceMergeStructure {
                        node: segment.region,
                        source,
                    };
                    let active = if let Some(slot) = node.activation {
                        let guard = level
                            .value_inputs(segment.region)
                            .find(|input| input.slot == slot)
                            .expect("validated ancestor activation")
                            .value_ref()
                            .expect("runtime input");
                        relation
                            .using_work(work, |relation| relation.implies_values(None, guard))
                            .map_err(invalid)?
                    } else {
                        true
                    };
                    // RUS executes its body at least once on every completed shot.
                    if !active {
                        return Err(invalid(NodeTemplateInstanceMergeError::InvalidMembership(
                            format!(
                                "external logical cut i{} is not always selected",
                                instance.0
                            ),
                        )));
                    }
                    path = path.child(segment.region, segment.body);
                }
            }
            if let Some((path, guard)) = availability.get(&instance) {
                let relation = predicates.entry(path.clone()).or_insert_with(|| {
                    crate::membership::PredicateAnalysis::with_limits(
                        self.level_at(path).expect("owner level"),
                        options.boolean_limits(),
                    )
                });
                let invalid = |source| BloqValidationError::InvalidInstanceMergeStructure {
                    node: guard.node,
                    source,
                };
                if !relation
                    .using_work(work, |relation| relation.implies_values(None, *guard))
                    .map_err(invalid)?
                {
                    return Err(invalid(NodeTemplateInstanceMergeError::InvalidMembership(
                        format!(
                            "external logical cut i{} is not always selected",
                            instance.0
                        ),
                    )));
                }
            }
        }
        if availability.is_empty() {
            return Ok(());
        }
        for (path, level) in self.levels() {
            let mut check =
                |node: BloqNodeId, guards: &[crate::ValueRef], instance: TemplateInstanceId| {
                    let Some((owner_path, required)) = availability.get(&instance) else {
                        return Ok(());
                    };
                    let mut gates = Vec::new();
                    let compatible = path.segments().starts_with(owner_path.segments());
                    if compatible {
                        if *owner_path == path {
                            gates.extend(guards.iter().copied());
                        } else {
                            let level = self.level_at(owner_path).expect("registered owner level");
                            let region = path.segments()[owner_path.segments().len()].region;
                            if let Some(slot) = level[region].activation {
                                gates.push(
                                    level
                                        .value_inputs(region)
                                        .find(|input| input.slot == slot)
                                        .expect("validated region activation")
                                        .value_ref()
                                        .expect("runtime input"),
                                );
                            }
                        }
                    }
                    let relation = predicates.entry(owner_path.clone()).or_insert_with(|| {
                        crate::membership::PredicateAnalysis::with_limits(
                            self.level_at(owner_path).expect("registered owner level"),
                            options.boolean_limits(),
                        )
                    });
                    let available = compatible
                        && relation
                            .using_work(work, |relation| {
                                relation.implies_all_values(&gates, *required)
                            })
                            .map_err(|source| {
                                BloqValidationError::InvalidInstanceMergeStructure { node, source }
                            })?;
                    if !available {
                        return Err(BloqValidationError::ConditionalReferenceUnavailable {
                            node,
                            instance,
                        });
                    }
                    Ok(())
                };
            for (id, node) in level.nodes() {
                let input = |slot| {
                    level
                        .value_inputs(id)
                        .find(|input| input.slot == slot)
                        .expect("validated guard input")
                        .value_ref()
                        .expect("runtime input")
                };
                let guard = node.activation.map(input);
                if let Some(classical) = node.try_classical() {
                    for term in classical.measurements() {
                        check(id, guard.as_slice(), term.instance)?;
                    }
                    for operator in classical.operators() {
                        check(id, guard.as_slice(), operator.instance)?;
                    }
                }
                if let Some(quantum) = node.try_quantum() {
                    for (index, detector) in quantum.detectors.iter().enumerate() {
                        let guard = quantum
                            .guards
                            .iter()
                            .find(|guard| guard.detectors.contains(&(index as u32)))
                            .map(|guard| input(guard.input));
                        for term in detector.parity.measurements() {
                            check(id, guard.as_slice(), term.instance)?;
                        }
                    }
                    for (index, use_) in quantum.detector_bundles.iter().enumerate() {
                        let bundle = self
                            .detector_bundles()
                            .get(use_.bundle)
                            .ok_or(crate::DetectorBundleError::UnknownBundle(use_.bundle))?;
                        let guard = quantum
                            .guards
                            .iter()
                            .find(|guard| guard.detector_bundles.contains(&(index as u32)))
                            .map(|guard| input(guard.input));
                        for &owner in bundle.used_owners()? {
                            check(id, guard.as_slice(), use_.instances[owner as usize])?;
                        }
                    }
                    for (index, restart) in quantum.restarts.iter().enumerate() {
                        let guard = quantum
                            .guards
                            .iter()
                            .find(|guard| guard.restarts.contains(&(index as u32)))
                            .map(|guard| input(guard.input));
                        for term in restart.parity.measurements() {
                            check(id, guard.as_slice(), term.instance)?;
                        }
                    }
                    for contribution in &quantum.guards {
                        for (rows, restart) in [
                            (&contribution.detector_parities, false),
                            (&contribution.restart_parities, true),
                        ] {
                            for (index, parity) in rows {
                                let mut guards = vec![input(contribution.input)];
                                if let Some(guard) = quantum.guards.iter().find(|guard| {
                                    if restart {
                                        guard.restarts.contains(index)
                                    } else {
                                        guard.detectors.contains(index)
                                    }
                                }) {
                                    guards.push(input(guard.input));
                                }
                                for term in parity.measurements() {
                                    check(id, &guards, term.instance)?;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn valid_quantum_timeline(
    timeline: &crate::QuantumTimeline,
    provenance: &crate::NodeProvenance,
) -> bool {
    let crate::NodeProvenance::BlockComponent { members } = provenance else {
        return false;
    };
    let Some(min_member_z) = members.iter().map(|member| member.pos.z).min() else {
        return false;
    };
    let ends = &timeline.layer_round_ends;
    let valid_rounds = ends.len() >= 2
        && ends.last().is_some_and(|&last| last > 0)
        && ends.windows(2).all(|pair| pair[0] <= pair[1]);
    let valid_span = ends
        .len()
        .checked_sub(1)
        .and_then(|span| i64::try_from(span).ok())
        .and_then(|span| i64::from(min_member_z).checked_add(span))
        .is_some_and(|z_end| z_end <= i64::from(i32::MAX));
    valid_rounds && valid_span
}

/// Collects program-global instance ids and rejects duplicates (WF-5).
fn collect_all_instances(
    bloq: &Bloq,
) -> Result<FxMap<TemplateInstanceId, TemplateId>, BloqValidationError> {
    let mut instances = FxMap::default();
    for (_, level) in bloq.levels() {
        for (_, quantum) in level.quantum_nodes() {
            for instance in &quantum.instances {
                if bloq.templates().get(instance.template_id).is_some()
                    && instances
                        .insert(instance.id, instance.template_id)
                        .is_some()
                {
                    return Err(BloqValidationError::DuplicateTemplateInstance {
                        instance: instance.id,
                    });
                }
            }
        }
    }
    Ok(instances)
}

#[derive(Clone, Copy)]
struct ValidationCtx {
    /// Nested in a `RepeatUntilSuccess` body.
    in_rus: bool,
    /// Whether any shared template carries restart syndromes.
    pool_has_restarts: bool,
}

/// A merge proof depends on ordered templates and relative placement, while
/// ownership, side tables, guards, and logical cuts remain per-node checks.
/// This cache belongs to one validation of one immutable template pool.
#[derive(Default)]
struct MergeProofCache {
    shapes: FxSet<Vec<(TemplateId, [i64; 2])>>,
    placements: FxSet<(TemplateId, IVec2)>,
}

impl MergeProofCache {
    fn validate(
        &mut self,
        id: BloqNodeId,
        node: &BloqNode,
        templates: &BloqTemplatePool,
        options: InstantiationOptions<'_>,
        work: &mut BooleanDecisionDiagram,
    ) -> Result<(), BloqValidationError> {
        let instances = &node.expect_quantum().instances;
        work.charge(instances.len().saturating_add(1))?;
        let origin = instances
            .first()
            .map_or(IVec2::ZERO, |instance| instance.offset);
        // Subtraction must represent opposite i32 extrema without overflow.
        let shape = instances
            .iter()
            .map(|instance| {
                (
                    instance.template_id,
                    [
                        i64::from(instance.offset.x) - i64::from(origin.x),
                        i64::from(instance.offset.y) - i64::from(origin.y),
                    ],
                )
            })
            .collect::<Vec<_>>();
        let merge_error =
            |source| BloqValidationError::InvalidInstanceMergeStructure { node: id, source };
        if self.shapes.contains(&shape) {
            for instance in instances {
                let placement = (instance.template_id, instance.offset);
                work.charge(1)?;
                if self.placements.contains(&placement) {
                    continue;
                }
                let qubits = templates[instance.template_id].qubits();
                work.charge(qubits.len())?;
                // Includes unused registry entries and stored, unreachable bodies,
                // matching the full instance translator's coordinate preflight.
                for &coordinate in qubits {
                    bloq_circuit::checked_translate_coordinate(coordinate, instance.offset)
                        .map_err(NodeTemplateInstanceMergeError::from)
                        .map_err(merge_error)?;
                }
                self.placements.insert(placement);
            }
            return Ok(());
        }

        for instance in instances {
            work.charge(crate::instantiation::circuit_work(
                &templates[instance.template_id].circuit,
            ))?;
        }
        // Template preflight alone does not prove measurement remapping inside
        // zero-count REPEAT bodies. Cache only successful full SEM-MERGE proofs.
        node.emission_plan_with_options(templates, options)
            .map_err(merge_error)?;
        self.shapes.insert(shape);
        self.placements.extend(
            instances
                .iter()
                .map(|instance| (instance.template_id, instance.offset)),
        );
        Ok(())
    }
}

/// Declared outputs remain valid even when the parent does not consume them.
fn validate_outputs(graph: &SubGraph) -> Result<(), BloqValidationError> {
    if let Some(value) = graph.value_output()
        && !valid_value_source(graph, value)
    {
        return Err(BloqValidationError::InvalidValueOutput { node: value.node });
    }
    let mut seen = FxSet::default();
    for &node in graph.boundary_outputs() {
        if !graph.node(node).is_some_and(|node| {
            node.try_region().is_some()
                || matches!(node.try_classical(), Some(ClassicalNode::Observable { .. }))
        }) {
            return Err(BloqValidationError::InvalidBoundaryOutput { node });
        }
        if !seen.insert(node) {
            return Err(BloqValidationError::DuplicateBoundaryOutput { node });
        }
    }
    Ok(())
}

/// Validates one graph level, then recurses into region bodies. `path` is the
/// level's [`LevelPath`], threaded down so each quantum node's plan is keyed by
/// its program-unique scoped address (matching `Bloq::levels`/`Bloq::walk`).
#[expect(
    clippy::too_many_arguments,
    reason = "whole-level validation shares scoped registries and bounded-work state"
)]
fn validate_graph(
    graph: &SubGraph,
    path: &LevelPath,
    templates: &BloqTemplatePool,
    bundles: &crate::DetectorBundlePool,
    all_instances: &FxMap<TemplateInstanceId, TemplateId>,
    template_measurements: &[&MeasurementSet],
    observable_indices: &mut FxSet<u32>,
    options: InstantiationOptions<'_>,
    plans: &mut Option<ValidatedPlans>,
    merge_proofs: &mut MergeProofCache,
    work: &mut BooleanDecisionDiagram,
    ctx: ValidationCtx,
) -> Result<(), BloqValidationError> {
    work.charge(graph.node_count().saturating_add(graph.edge_count()))?;
    // Hand-built Value edges may create cycles (IR spec §3).
    let order = graph
        .deterministic_emit_order()
        .map_err(|_| BloqValidationError::CyclicGraph)?;
    let mut predicates =
        crate::membership::PredicateAnalysis::with_limits(graph, options.boolean_limits());
    predicates.charge(work.steps()).map_err(|source| {
        BloqValidationError::InvalidInstanceMergeStructure {
            node: order.first().copied().unwrap_or(BloqNodeId(0)),
            source,
        }
    })?;
    validate_edge_cardinality(graph, &mut predicates)?;
    work.charge(predicates.steps().saturating_sub(work.steps()))?;
    validate_outputs(graph)?;
    let order_index = order
        .iter()
        .enumerate()
        .map(|(index, &id)| (id, index))
        .collect::<FxMap<_, _>>();

    let mut instances = FxMap::default();
    for &id in &order {
        let node = &graph[id];
        let Some(quantum) = node.try_quantum() else {
            continue; // classical / region nodes carry no template instances
        };
        for instance in &quantum.instances {
            if templates.get(instance.template_id).is_none() {
                return Err(BloqValidationError::UnknownTemplate {
                    node: id,
                    template: instance.template_id,
                });
            }
            assert!(
                instances
                    .insert(
                        instance.id,
                        InstanceValidation {
                            template: instance.template_id,
                            owner: id,
                            order: order_index[&id],
                        },
                    )
                    .is_none(),
                "the whole-program instance preflight rejected duplicate ids"
            );
        }
    }

    let mut path_scratch = graph.path_scratch();
    let mut verified_paths = FxSet::default();
    for (id, node) in graph.nodes() {
        if let Some(slot) = node.activation {
            if node.try_quantum().is_some() {
                return Err(BloqValidationError::InvalidActivation {
                    node: id,
                    rule: "WF-12",
                    reason: "quantum nodes use member registrations",
                });
            }
            if !graph.value_inputs(id).any(|input| input.slot == slot) {
                return Err(BloqValidationError::MissingClassicalInput { node: id, slot });
            }
        }
        let consumer_order = order_index[&id];
        if let Some(classical) = node.try_classical() {
            // Complete observables retain the same record and boundary-owner
            // checks as independently shared readout fragments.
            work.charge(classical.measurements().len())?;
            validate_instance_measurement_refs(
                id,
                classical.measurements().iter().copied(),
                &instances,
                all_instances,
                template_measurements,
                consumer_order,
            )?;
            work.charge(classical.operators().len())?;
            for operator in classical.operators() {
                if !all_instances.contains_key(&operator.instance) {
                    return Err(BloqValidationError::UnknownTemplateInstance {
                        node: id,
                        instance: operator.instance,
                    });
                }
            }
        }
        if let Some(ClassicalNode::Observable {
            index: Some(observable),
            ..
        }) = node.try_classical()
            && !observable_indices.insert(*observable)
        {
            return Err(BloqValidationError::DuplicateObservableIndex {
                node: id,
                index: *observable,
            });
        }
        let Some(quantum) = node.try_quantum() else {
            continue; // classical / region nodes carry no detectors/repeat states
        };
        for parity in quantum.stored_parities() {
            work.charge(parity.terms().len())?;
            validate_instance_parity_refs(
                graph,
                id,
                parity,
                &instances,
                all_instances,
                template_measurements,
                consumer_order,
                &mut path_scratch,
                &mut verified_paths,
            )?;
        }
        for use_ in &quantum.detector_bundles {
            work.charge(1)?;
            let bundle = bundles
                .get(use_.bundle)
                .ok_or(crate::DetectorBundleError::UnknownBundle(use_.bundle))?;
            for &owner in bundle.used_owners()? {
                work.charge(1)?;
                let instance = use_.instances[owner as usize];
                if let Some(source) = instances.get(&instance) {
                    if source.order > consumer_order
                        || (source.owner != id
                            && !verified_paths.contains(&(source.owner, id))
                            && !graph.has_path_backwards_with_scratch(
                                source.owner,
                                id,
                                &mut path_scratch,
                            ))
                    {
                        return Err(BloqValidationError::UnavailableDetectorBundleOwner {
                            node: id,
                            instance,
                        });
                    }
                    verified_paths.insert((source.owner, id));
                }
            }
        }
        if !ctx.in_rus
            && (!quantum.restarts.is_empty()
                || (ctx.pool_has_restarts
                    && quantum.instances.iter().any(|instance| {
                        templates
                            .get(instance.template_id)
                            .is_some_and(|template| !template.restarts.is_empty())
                    })))
        {
            return Err(BloqValidationError::RestartOutsideRepeatUntilSuccess { node: id });
        }
        // WF-9: the instance merge must succeed. Reuse successful relative-shape
        // proofs only when no noisy or retained emission plan is requested.
        // Retained plans let consumers emit without re-merging (IR-02).
        // Noise is applied post-merge, so a malformed NoiseModel surfaces here as
        // an InvalidInstanceMergeStructure (see `validate_with_plans`).
        let merge_error =
            |source| BloqValidationError::InvalidInstanceMergeStructure { node: id, source };
        if quantum.guards.is_empty() && plans.is_none() && options.is_noiseless() {
            merge_proofs.validate(id, node, templates, options, work)?;
        } else if quantum.guards.is_empty() {
            work.charge(quantum.instances.len())?;
            for instance in &quantum.instances {
                work.charge(crate::instantiation::circuit_work(
                    &templates[instance.template_id].circuit,
                ))?;
            }
            let plan = node
                .emission_plan_with_options(templates, options)
                .map_err(merge_error)?;
            if let Some(plans) = plans {
                plans.insert(path.clone(), id, plan);
            }
        } else {
            predicates
                .using_work(work, |predicates| {
                    crate::membership::validate_merge(graph, id, templates, options, predicates)
                })
                .map_err(merge_error)?;
            if let Some(plans) = plans {
                plans
                    .memberships
                    .entry(path.clone())
                    .or_default()
                    .insert(id);
            }
        }

        if let Some(input) = graph
            .value_inputs(id)
            .find(|input| !quantum.guards.iter().any(|guard| guard.input == input.slot))
        {
            return Err(BloqValidationError::UnusedQuantumValueInput {
                node: id,
                slot: input.slot,
            });
        }
    }

    // RUS failures restart rather than discard, including in nested bodies (§6).
    if ctx.in_rus
        && let Some((node, _)) = graph
            .nodes()
            .find(|(_, node)| matches!(node.try_classical(), Some(ClassicalNode::Discard { .. })))
    {
        return Err(BloqValidationError::DiscardInsideRepeatUntilSuccess { node });
    }

    for (region_id, node) in graph.nodes() {
        let BloqNodeKind::Region(region) = &node.kind else {
            continue;
        };
        if ctx.in_rus {
            return Err(BloqValidationError::NestedRepeatUntilSuccess { node: region_id });
        }
        let RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: predicate,
            restart_source,
        } = region;
        // Region predicates share Compute's linear-input contract (WF-11).
        let mut read = expression_inputs(region_id, predicate, work)?;
        if let Some(slot) = node.activation
            && !read.insert(slot)
        {
            return Err(BloqValidationError::InvalidActivation {
                node: region_id,
                rule: "WF-11",
                reason: "activation slot is also a predicate operand",
            });
        }
        let fed: BTreeSet<_> = graph
            .value_inputs(region_id)
            .map(|input| input.slot)
            .collect();

        // A supplied RUS restart_source must produce a body-local value (WF-10).
        if let Some(source) = restart_source
            && !valid_value_source(body, *source)
        {
            return Err(BloqValidationError::InvalidRestartSource {
                node: region_id,
                restart_source: *source,
            });
        }

        // Restart predicates may use incoming Value edges or a body-local
        // restart_source because edges cannot cross regions (§6/U17).
        if restart_source.is_none()
            && let Some(&slot) = read.difference(&fed).next()
        {
            return Err(BloqValidationError::MissingRegionSelectorInput {
                node: region_id,
                slot,
            });
        }
        if let Some(&slot) = fed.difference(&read).next() {
            return Err(BloqValidationError::UnusedClassicalInput {
                node: region_id,
                slot,
            });
        }

        let body_ctx = ValidationCtx {
            in_rus: true,
            ..ctx
        };
        for (selector, body) in region.bodies() {
            validate_graph(
                body,
                &path.child(region_id, selector),
                templates,
                bundles,
                all_instances,
                template_measurements,
                observable_indices,
                options,
                plans,
                merge_proofs,
                work,
                body_ctx,
            )?;
        }
    }

    // Compute/Discard expressions must read each fed Value slot exactly once.
    for (id, node) in graph.nodes() {
        let expr = match node.try_classical() {
            Some(ClassicalNode::Compute { expr }) => expr,
            Some(ClassicalNode::Discard { condition }) => condition,
            _ => continue,
        };
        let mut read = expression_inputs(id, expr, work)?;
        if let Some(slot) = node.activation
            && !read.insert(slot)
        {
            return Err(BloqValidationError::InvalidActivation {
                node: id,
                rule: "WF-11",
                reason: "activation slot is also a data operand",
            });
        }
        let fed: BTreeSet<_> = graph.value_inputs(id).map(|input| input.slot).collect();
        if let Some(&slot) = read.difference(&fed).next() {
            return Err(BloqValidationError::MissingClassicalInput { node: id, slot });
        }
        if let Some(&slot) = fed.difference(&read).next() {
            return Err(BloqValidationError::UnusedClassicalInput { node: id, slot });
        }
    }

    Ok(())
}

/// WF-11: collect inputs without recursing through expression nesting. Only
/// affine expressions prohibit duplicates; every constant/operand costs work.
fn expression_inputs(
    node: BloqNodeId,
    expr: &ClassicalExpr,
    work: &mut BooleanDecisionDiagram,
) -> Result<BTreeSet<u32>, BloqValidationError> {
    work.charge(1)?;
    let mut pending = vec![expr];
    let mut read = BTreeSet::new();
    let mut linear = true;
    let mut duplicate = None;
    while let Some(expr) = pending.pop() {
        work.charge(1)?;
        let mut insert = |slot| {
            if !read.insert(slot) && duplicate.is_none() {
                duplicate = Some(slot);
            }
        };
        match expr {
            ClassicalExpr::In(slot) => insert(*slot),
            ClassicalExpr::Parity { inputs, .. } => {
                work.charge(inputs.len())?;
                for &slot in inputs {
                    insert(slot);
                }
            }
            ClassicalExpr::And(..) | ClassicalExpr::Or(..) | ClassicalExpr::Select(..) => {
                linear = false;
            }
            _ => {}
        }
        work.charge(expr.operands().len())?;
        pending.extend(expr.operands().iter().rev());
    }
    if linear && let Some(slot) = duplicate {
        return Err(BloqValidationError::DuplicateExprInput { node, slot });
    }
    Ok(read)
}

#[derive(Debug, Clone, Copy)]
struct InstanceValidation {
    template: TemplateId,
    owner: BloqNodeId,
    order: usize,
}

fn validate_template<'a>(
    template_id: TemplateId,
    template: &'a crate::BloqTemplate,
    work: &mut bloq_utils::boolean::BooleanDecisionDiagram,
) -> Result<&'a MeasurementSet, BloqValidationError> {
    // Charge stored input before the circuit-analysis cache may allocate or
    // walk it. Reused caches still obey the caller's declared input allowance.
    charge_template_work(template_id, work, template.circuit.body_count())?;
    for index in 0..template.circuit.body_count() {
        let ops = template
            .circuit
            .body(BodyId(index as u32))
            .expect("allocated body")
            .ops();
        charge_template_work(template_id, work, ops.len())?;
        for op in ops {
            let terms = match op {
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
                } => {
                    charge_template_work(template_id, work, products.len())?;
                    for product in products {
                        charge_template_work(template_id, work, product.len())?;
                    }
                    measurements.len()
                }
                Op::ConditionalPauli(corrections) => corrections.len(),
                Op::Tick | Op::Repeat { .. } => 0,
            };
            charge_template_work(template_id, work, terms)?;
        }
    }
    charge_template_work(
        template_id,
        work,
        template.circuit.meas_registry().records().len(),
    )?;
    let analysis = template.circuit_analysis().map_err(|source| {
        BloqValidationError::InvalidTemplateCircuit {
            template: template_id,
            source,
        }
    })?;
    let produced_measurements = &analysis.produced_measurements;
    if template.validation_cached() {
        return Ok(produced_measurements);
    }

    let produced = |measurement| produced_measurements.contains(measurement);

    charge_template_work(template_id, work, template.repeat_states.len())?;
    charge_template_work(template_id, work, template.detectors.len())?;
    charge_template_work(template_id, work, template.restarts.len())?;
    charge_template_work(template_id, work, template.boundary_flows.len())?;
    for parity in template
        .repeat_states
        .iter()
        .flat_map(|state| [&state.initial, &state.next])
        .chain(template.detectors.iter().map(|detector| &detector.parity))
        .chain(template.restarts.iter().map(|restart| &restart.parity))
    {
        charge_template_work(template_id, work, parity.terms().len())?;
    }
    for flow in &template.boundary_flows {
        charge_template_work(template_id, work, flow.measurements.len())?;
    }

    let mut states_by_body: FxMap<BodyId, Vec<&crate::TemplateRepeatState>> = FxMap::default();
    let mut state_ids =
        FxSet::with_capacity_and_hasher(template.repeat_states.len(), Default::default());
    for state in &template.repeat_states {
        if !state_ids.insert(state.state) {
            return Err(BloqValidationError::DuplicateTemplateLoopState {
                template: template_id,
                state: state.state,
            });
        }
        validate_template_body_ref(template_id, state.body, &analysis.reachable_bodies)?;
        for parity in [&state.initial, &state.next] {
            validate_template_measurement_refs(template_id, parity, produced)?;
        }
        states_by_body.entry(state.body).or_default().push(state);
    }

    let mut detectors_by_body: FxMap<BodyId, Vec<&crate::TemplateDetector>> = FxMap::default();
    for detector in &template.detectors {
        validate_template_measurement_refs(template_id, &detector.parity, produced)?;
        if let TemplateDetectorScope::RepeatBody { body } = detector.scope {
            validate_template_body_ref(template_id, body, &analysis.reachable_bodies)?;
            detectors_by_body.entry(body).or_default().push(detector);
        }
    }

    // Side tables may name only executed measurement sites.
    for flow in &template.boundary_flows {
        for &measurement in &flow.measurements {
            if !produced_measurements.contains(measurement) {
                return Err(BloqValidationError::UnknownTemplateMeasurement {
                    template: template_id,
                    measurement,
                });
            }
        }
    }
    for restart in &template.restarts {
        validate_template_measurement_refs(template_id, &restart.parity, produced)?;
    }

    for event in &analysis.repeat_events {
        for state in states_by_body.get(&event.body).into_iter().flatten() {
            validate_template_measurement_refs_before(
                template_id,
                &analysis.measurement_order,
                event.before,
                &state.initial,
            )?;
            validate_template_measurement_refs_before(
                template_id,
                &analysis.measurement_order,
                event.after,
                &state.next,
            )?;
        }
        for detector in detectors_by_body.get(&event.body).into_iter().flatten() {
            validate_template_measurement_refs_before(
                template_id,
                &analysis.measurement_order,
                event.after,
                &detector.parity,
            )?;
        }
    }

    let available_states = validate_template_body_events(
        template_id,
        template,
        &analysis.body_order,
        &states_by_body,
        &detectors_by_body,
        work,
    )?;
    for detector in &template.detectors {
        if detector.scope == TemplateDetectorScope::TopLevel {
            validate_template_loop_state_refs(template_id, &detector.parity, &available_states)?;
        }
    }
    for restart in &template.restarts {
        validate_template_loop_state_refs(template_id, &restart.parity, &available_states)?;
    }
    template.cache_validation();
    Ok(produced_measurements)
}

fn validate_template_body_events<'a>(
    template_id: TemplateId,
    template: &'a crate::BloqTemplate,
    body_order: &[BodyId],
    states_by_body: &FxMap<BodyId, Vec<&'a crate::TemplateRepeatState>>,
    detectors_by_body: &FxMap<BodyId, Vec<&'a crate::TemplateDetector>>,
    work: &mut bloq_utils::boolean::BooleanDecisionDiagram,
) -> Result<FxSet<LoopStateId>, BloqValidationError> {
    #[derive(Default)]
    struct StateEffects {
        // Preserve first-reference order for the same diagnostic as the walk.
        required: Vec<LoopStateId>,
        provided: FxSet<LoopStateId>,
    }
    let mut effects = FxMap::<BodyId, StateEffects>::default();
    for &body in body_order {
        charge_template_work(template_id, work, 1)?;
        let mut effect = StateEffects::default();
        let mut required = FxSet::default();
        let mut require = |state| {
            if required.insert(state) {
                effect.required.push(state);
            }
        };
        for op in template
            .circuit
            .body(body)
            .expect("preflight checked bodies")
            .ops()
        {
            charge_template_work(template_id, work, 1)?;
            let Op::Repeat {
                body: child,
                repetitions,
            } = op
            else {
                continue;
            };
            if *repetitions == 0 {
                continue;
            }
            let states = states_by_body.get(child);
            let child_effect = &effects[child];
            charge_template_work(template_id, work, child_effect.required.len())?;
            charge_template_work(template_id, work, states.map_or(0, Vec::len))?;
            let declared: FxSet<_> = states
                .into_iter()
                .flatten()
                .map(|state| state.state)
                .collect();
            // Initial parities see the surrounding frame before this loop's
            // own declarations become available.
            for state in states.into_iter().flatten() {
                charge_template_work(template_id, work, state.initial.terms().len())?;
                for term in state.initial.terms() {
                    if let DetectorTerm::LoopState(state) = *term
                        && !effect.provided.contains(&state)
                    {
                        require(state);
                    }
                }
            }
            let in_iteration =
                |state| effect.provided.contains(&state) || declared.contains(&state);
            for &state in &child_effect.required {
                if !in_iteration(state) {
                    require(state);
                }
            }
            // Body detectors and next parities also see completed nested
            // declarations. Those nested states die when this frame exits.
            for parity in detectors_by_body
                .get(child)
                .into_iter()
                .flatten()
                .map(|detector| &detector.parity)
                .chain(states.into_iter().flatten().map(|state| &state.next))
            {
                charge_template_work(template_id, work, parity.terms().len())?;
                for term in parity.terms() {
                    if let DetectorTerm::LoopState(state) = *term
                        && !in_iteration(state)
                        && !child_effect.provided.contains(&state)
                    {
                        require(state);
                    }
                }
            }
            effect.provided.extend(declared);
        }
        effects.insert(body, effect);
    }
    let entry = effects
        .remove(&template.circuit.entry_body())
        .expect("entry was visited");
    if let Some(&state) = entry.required.first() {
        return Err(BloqValidationError::UnknownTemplateLoopState {
            template: template_id,
            state,
        });
    }
    Ok(entry.provided)
}

fn charge_template_work(
    template: TemplateId,
    work: &mut bloq_utils::boolean::BooleanDecisionDiagram,
    amount: usize,
) -> Result<(), BloqValidationError> {
    work.charge(amount)
        .map_err(|source| BloqValidationError::InvalidTemplateCircuit {
            template,
            source: crate::NodeTemplateInstanceMergeError::BooleanResource(source),
        })
}

/// Every measurement term of `parity` must name a site `available` accepts.
fn validate_template_measurement_refs(
    template: TemplateId,
    parity: &crate::TemplateDetectorParity,
    available: impl Fn(u32) -> bool,
) -> Result<(), BloqValidationError> {
    for term in parity.terms() {
        if let DetectorTerm::Measurement(measurement) = *term
            && !available(measurement)
        {
            return Err(BloqValidationError::UnknownTemplateMeasurement {
                template,
                measurement,
            });
        }
    }
    Ok(())
}

/// The repeat-event form: a parity evaluated at an event may only read
/// measurement sites ordered strictly before it.
fn validate_template_measurement_refs_before(
    template: TemplateId,
    order: &FxMap<u32, usize>,
    before: usize,
    parity: &crate::TemplateDetectorParity,
) -> Result<(), BloqValidationError> {
    validate_template_measurement_refs(template, parity, |measurement| {
        order.get(&measurement).is_some_and(|&index| index < before)
    })
}

fn validate_template_loop_state_refs(
    template: TemplateId,
    parity: &crate::TemplateDetectorParity,
    available_states: &FxSet<LoopStateId>,
) -> Result<(), BloqValidationError> {
    for term in parity.terms() {
        if let DetectorTerm::LoopState(state) = *term
            && !available_states.contains(&state)
        {
            return Err(BloqValidationError::UnknownTemplateLoopState { template, state });
        }
    }
    Ok(())
}

fn validate_template_body_ref(
    template: TemplateId,
    body: BodyId,
    reachable_bodies: &FxSet<BodyId>,
) -> Result<(), BloqValidationError> {
    if !reachable_bodies.contains(&body) {
        return Err(BloqValidationError::UnknownTemplateBody { template, body });
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "parity validation needs both local and whole-program ownership tables"
)]
fn validate_instance_parity_refs(
    graph: &SubGraph,
    node: BloqNodeId,
    parity: &crate::NodeDetectorParity,
    instances: &FxMap<TemplateInstanceId, InstanceValidation>,
    all_instances: &FxMap<TemplateInstanceId, TemplateId>,
    template_measurements: &[&MeasurementSet],
    consumer_order: usize,
    path_scratch: &mut crate::PathScratch,
    verified_paths: &mut FxSet<(BloqNodeId, BloqNodeId)>,
) -> Result<(), BloqValidationError> {
    if let Some(state) = parity.terms().iter().find_map(|term| match *term {
        DetectorTerm::LoopState(state) => Some(state),
        DetectorTerm::Measurement(_) => None,
    }) {
        return Err(BloqValidationError::NodeLoopStateUnsupported { node, state });
    }
    validate_instance_measurement_refs(
        node,
        parity.measurements(),
        instances,
        all_instances,
        template_measurements,
        consumer_order,
    )?;
    for measurement in parity.measurements() {
        let Some(instance) = instances.get(&measurement.instance) else {
            continue;
        };
        let pair = (instance.owner, node);
        if instance.owner != node && !verified_paths.contains(&pair) {
            if !graph.has_path_backwards_with_scratch(instance.owner, node, path_scratch) {
                return Err(BloqValidationError::UnorderedNodeMeasurement {
                    node,
                    owner: instance.owner,
                    measurement: measurement.measurement,
                });
            }
            verified_paths.insert(pair);
        }
    }
    Ok(())
}

fn validate_instance_measurement_refs(
    node: BloqNodeId,
    measurements: impl IntoIterator<Item = InstanceMeasurement>,
    instances: &FxMap<TemplateInstanceId, InstanceValidation>,
    all_instances: &FxMap<TemplateInstanceId, TemplateId>,
    template_measurements: &[&MeasurementSet],
    consumer_order: usize,
) -> Result<(), BloqValidationError> {
    let mut cached = None;
    for measurement in measurements {
        // Canonical parities and compiler observable recipes group an instance's refs.
        if cached
            .as_ref()
            .is_none_or(|(instance, _, _)| *instance != measurement.instance)
        {
            // Cross-region refs use global instances; only local refs enforce
            // read order.
            let local = instances.get(&measurement.instance);
            let template = match local {
                Some(instance) => instance.template,
                None => all_instances.get(&measurement.instance).copied().ok_or(
                    BloqValidationError::UnknownTemplateInstance {
                        node,
                        instance: measurement.instance,
                    },
                )?,
            };
            let produced = template_measurements.get(template.0 as usize).copied();
            let future_owner = local
                .filter(|instance| instance.order > consumer_order)
                .map(|instance| instance.owner);
            cached = Some((measurement.instance, produced, future_owner));
        }
        let (_, produced, future_owner) = cached.expect("current instance was resolved");
        if !produced.is_some_and(|produced| produced.contains(measurement.measurement)) {
            return Err(BloqValidationError::UnknownInstanceMeasurement {
                node,
                instance: measurement.instance,
                measurement: measurement.measurement,
            });
        }
        if let Some(owner) = future_owner {
            return Err(BloqValidationError::FutureMeasurement {
                node,
                owner,
                measurement: measurement.measurement,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bloq_circuit::{
        BodyId, CircuitBody, ConditionalCorrection, CoordCircuit, DetectorParity, DetectorTerm,
        GateType, LoopStateId, Op, Pauli, PauliBasis, PauliMap,
    };
    use glam::{ivec2, ivec3};

    use crate::PipeSeam;
    use crate::instantiation::InstantiationOptions;
    use crate::test_fixture::instance_measurement;
    use crate::{
        Bloq, BloqEdge, BloqNode, BloqTemplate, BloqValidationError, BoundaryFace, ClassicalExpr,
        ClassicalNode, InstanceBoundaryOperator, InstanceMeasurement, LevelPath, LogicalInput,
        LogicalOutput, NodeDetector, NodeRestart, NodeTemplateInstanceMergeError, ObservableOutput,
        PipePadding, QuantumEdge, QuantumTimeline, RegionNode, SourceBlockRef, SubGraph,
        TemplateDetector, TemplateDetectorScope, TemplateId, TemplateInstance, TemplateInstanceId,
        TemplateRepeatState, TemplateRestart, TemporalPipeRef, ValueRef,
    };

    fn node(pos: glam::IVec3) -> BloqNode {
        BloqNode::from_members(vec![SourceBlockRef { pos }])
    }

    /// [`BloqValidationError`]'s own contract: a well-formedness message leads
    /// with its normative rule id, so a consumer can attribute a failure
    /// without consulting the spec's error tables. These two variants are the
    /// ones that once reported no id at all.
    #[test]
    fn well_formedness_messages_lead_with_their_rule_id() {
        for error in [
            BloqValidationError::InvalidActivation {
                node: crate::BloqNodeId(0),
                rule: "WF-16",
                reason: "activation is not an observable fold",
            },
            BloqValidationError::InvalidSeamGuard {
                node: crate::BloqNodeId(1),
                guard: crate::BloqNodeId(0).into(),
            },
        ] {
            let message = error.to_string();
            assert!(
                message.starts_with("WF-"),
                "message does not lead with a rule id: {message}"
            );
        }
    }

    #[test]
    fn unused_bundle_signature_still_requires_known_templates() {
        let mut bloq = Bloq::new();
        let id =
            bloq.add_detector_bundle(crate::DetectorBundle::new(vec![TemplateId(99)], Vec::new()));
        assert!(
            matches!(bloq.validate(), Err(BloqValidationError::UnknownDetectorBundleTemplate { bundle, template: TemplateId(99) }) if bundle == id)
        );
    }

    #[test]
    fn membership_certification_does_not_enumerate_independent_selectors() {
        const COUNT: u32 = 40;
        let mut bloq = Bloq::new();
        let mut controls = CoordCircuit::new();
        controls.measure(
            PauliBasis::Z,
            (0..COUNT).map(|index| ivec2(index as i32, 8)),
        );
        let control_template = bloq.add_template(BloqTemplate::new(controls));
        let mut control = node(ivec3(0, 0, 0));
        control
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                control_template,
                ivec2(0, 0),
            ));
        let control = bloq.add_node(control);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut selected = node(ivec3(0, 0, 1));
        for index in 0..COUNT {
            let instance = TemplateInstanceId(index + 1);
            selected
                .expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    instance,
                    template,
                    ivec2(index as i32, 0),
                ));
            selected
                .expect_quantum_mut()
                .guards
                .push(crate::QuantumGuard {
                    input: index,
                    instances: vec![instance],
                    ..Default::default()
                });
        }
        let selected = bloq.add_node(selected);
        for index in 0..COUNT {
            let read = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
                index: None,
                operators: Vec::new(),
                measurements: vec![instance_measurement(0, index)],
            }));
            bloq.add_edge(control, read, BloqEdge::Order);
            bloq.add_edge(read, selected, BloqEdge::value(index));
        }
        let plans = bloq
            .validate_with_plans(&InstantiationOptions::default())
            .unwrap();
        assert!(plans.has_membership(&LevelPath::default(), selected));
        let materialized = bloq
            .emission_plans(&InstantiationOptions::default().with_boolean_limits(
                bloq_utils::boolean::BooleanLimits {
                    max_nodes: 0,
                    max_steps: 0,
                },
            ))
            .unwrap();
        assert!(materialized.has_membership(&LevelPath::default(), selected));
        assert!(materialized.get(&LevelPath::default(), selected).is_none());
        // A residual seam predicate is live even without a Value consumer.
        let raw = bloq.top().value_inputs(selected).next().unwrap().producer;
        let guard = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(raw, guard, BloqEdge::value(0));
        let end = bloq.add_node(node(ivec3(0, 0, 2)));
        bloq.add_edge(guard, end, BloqEdge::Order);
        let mut seam = BloqEdge::quantum(vec![TemporalPipeRef {
            src: ivec3(0, 0, 1),
            dst: ivec3(0, 0, 2),
            hadamard: false,
        }]);
        if let BloqEdge::Quantum(quantum) = &mut seam {
            quantum.guard = Some(guard.into());
        }
        bloq.add_edge(selected, end, seam);
        let pinned = bloq.pin_membership(&Default::default()).unwrap();
        assert!(pinned.node(guard).is_some());

        let mut conflict = CoordCircuit::new();
        conflict.do_gate(GateType::X, [ivec2(0, 0)]).unwrap();
        let conflict = bloq.add_template(BloqTemplate::new(conflict));
        let second = &mut bloq
            .node_mut(selected)
            .unwrap()
            .expect_quantum_mut()
            .instances[1];
        second.offset = ivec2(0, 0);
        second.template_id = conflict;
        assert!(matches!(
            bloq.validate(),
            Err(BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::QubitConflict(_),
                ..
            })
        ));
    }

    #[test]
    fn complementary_membership_checks_domains_references_and_pins() {
        use crate::{BloqNodeKind, MembershipPinError, NodeProvenance, QuantumGuard};

        let mut bloq = Bloq::new();
        let mut control = CoordCircuit::new();
        control.measure(PauliBasis::X, [ivec2(8, 0)]);
        let template = bloq.add_template(BloqTemplate::new(control));
        let mut stage = node(ivec3(8, 0, 0));
        stage
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let measured = bloq.add_node(stage);
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(0, 0)],
        }));
        bloq.add_edge(measured, raw, BloqEdge::Order);
        let selectors = [false, true].map(|invert| {
            let mut selector = BloqNode::classical(ClassicalNode::Compute {
                expr: if invert {
                    ClassicalExpr::Not(Box::new(ClassicalExpr::In(0)))
                } else {
                    ClassicalExpr::In(0)
                },
            });
            selector.provenance = NodeProvenance::BranchSelector {
                name: if invert { "off" } else { "on #測 %" }.into(),
            };
            let id = bloq.add_node(selector);
            bloq.add_edge(raw, id, BloqEdge::value(0));
            id
        });
        let mut alternatives = node(ivec3(0, 0, 1));
        for (index, gate) in [GateType::H, GateType::X].into_iter().enumerate() {
            let mut circuit = CoordCircuit::new();
            circuit.do_gate(gate, [ivec2(0, 0)]).unwrap();
            circuit.tick();
            circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
            let template = bloq.add_template(BloqTemplate::new(circuit));
            let instance = TemplateInstanceId(index as u32 + 1);
            alternatives
                .expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(instance, template, ivec2(0, 0)));
            alternatives.expect_quantum_mut().guards.push(QuantumGuard {
                input: index as u32,
                instances: vec![instance],
                ..Default::default()
            });
        }
        alternatives
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: Default::default(),
                coords: None,
            });
        alternatives.expect_quantum_mut().guards[0]
            .detectors
            .push(0);
        alternatives.expect_quantum_mut().guards[0]
            .detector_parities
            .push((
                0,
                crate::NodeDetectorParity::from_measurements([instance_measurement(1, 0)]),
            ));
        let quantum = bloq.add_node(alternatives);
        for (slot, selector) in selectors.into_iter().enumerate() {
            bloq.add_edge(selector, quantum, BloqEdge::value(slot as u32));
        }
        // Whole-use activation must imply its record owner's selection.
        let mut bundled = bloq.clone();
        let owner_template = bundled[quantum].expect_quantum().instances[0].template_id;
        let bundle = bundled.add_detector_bundle(crate::DetectorBundle::new(
            vec![owner_template],
            vec![crate::BundleDetector {
                parity: DetectorParity::from_measurements([crate::BundleMeasurement {
                    owner: 0,
                    measurement: 0,
                }]),
                coords: None,
            }],
        ));
        let payload = bundled.node_mut(quantum).unwrap().expect_quantum_mut();
        payload.detector_bundles.push(crate::DetectorBundleUse {
            bundle,
            instances: vec![TemplateInstanceId(1)],
            offset: ivec2(0, 0),
        });
        payload.guards[0].detector_bundles.push(0);
        bundled.validate().unwrap();
        let payload = bundled.node_mut(quantum).unwrap().expect_quantum_mut();
        payload.guards[0].detector_bundles.clear();
        payload.guards[1].detector_bundles.push(0);
        assert!(matches!(
            bundled.validate(),
            Err(BloqValidationError::ConditionalReferenceUnavailable { .. })
        ));
        let mut read = BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(1, 0)],
        });
        read.activation = Some(0);
        let read = bloq.add_node(read);
        bloq.add_edge(quantum, read, BloqEdge::Order);
        bloq.add_edge(selectors[0], read, BloqEdge::value(0));
        let plans = bloq
            .validate_with_plans(&InstantiationOptions::default())
            .unwrap();
        assert!(plans.has_membership(&LevelPath::default(), quantum));
        // Nonexhaustive membership still needs a pinned path for timing edits.
        let mut partial = bloq.clone();
        partial.node_mut(selectors[1]).unwrap().kind = BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::And(Box::new([
                    ClassicalExpr::In(0),
                    ClassicalExpr::Const(false),
                ])),
            }
            .into(),
        );
        partial.validate().unwrap();
        let original = partial.to_binary();
        assert!(matches!(
            partial.insert_memory_rounds_batch(&[], 1),
            Err(crate::EditError::MembershipSelectionRequired)
        ));
        assert_eq!(partial.to_binary(), original);
        for restored in [
            Bloq::from_text(&bloq.to_text()).unwrap(),
            Bloq::from_binary(&bloq.to_binary()).unwrap(),
        ] {
            restored.validate().unwrap();
            assert!(matches!(
                restored.pin_membership(&[("on #測 %".into(), true), ("off".into(), true)].into()),
                Err(MembershipPinError::UnreachableAssignment)
            ));
            let pinned = restored
                .pin_membership(&[("on #測 %".into(), false), ("off".into(), true)].into())
                .unwrap();
            assert!(!pinned.has_conditional_membership());
            assert_eq!(
                pinned.node(quantum).unwrap().expect_quantum().instances[0].id,
                TemplateInstanceId(2)
            );
        }
        let mut invalid = bloq.clone();
        invalid.set_logical_inputs(vec![LogicalInput {
            port: ivec3(0, 0, 0),
            instance: TemplateInstanceId(1),
            x: PauliMap::empty(),
            z: PauliMap::empty(),
        }]);
        assert!(matches!(
            invalid.validate(),
            Err(BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::InvalidMembership(_),
                ..
            })
        ));
        // An unconditional readout would read a missing record on the false arm.
        let mut invalid = bloq.clone();
        invalid.node_mut(read).unwrap().activation = None;
        let graph = invalid.top_mut().graph_mut();
        let guard_edge = graph
            .find_edge(
                petgraph::stable_graph::NodeIndex::new(selectors[0].0 as usize),
                petgraph::stable_graph::NodeIndex::new(read.0 as usize),
            )
            .unwrap();
        graph.remove_edge(guard_edge);
        assert!(matches!(
            invalid.validate(),
            Err(BloqValidationError::ConditionalReferenceUnavailable { .. })
        ));
        // Local records and declarative bindings on a complete observable
        // each retain the selected-owner requirement.
        for record_terms in [true, false] {
            let mut complete = bloq.clone();
            complete.node_mut(read).unwrap().kind = BloqNodeKind::Classical(
                ClassicalNode::Observable {
                    index: Some(0),
                    measurements: if record_terms {
                        vec![instance_measurement(1, 0)]
                    } else {
                        Vec::new()
                    },
                    operators: if record_terms {
                        Vec::new()
                    } else {
                        vec![InstanceBoundaryOperator {
                            instance: TemplateInstanceId(1),
                            face: BoundaryFace::Output,
                            operator: PauliMap::empty(),
                        }]
                    },
                }
                .into(),
            );
            complete.validate().unwrap();
            complete.node_mut(read).unwrap().activation = None;
            assert!(matches!(
                complete.validate(),
                Err(BloqValidationError::ConditionalReferenceUnavailable { .. })
            ));
        }
    }

    fn assert_invalid_timeline(mut weight: BloqNode, layer_round_ends: Vec<u32>) {
        weight.expect_quantum_mut().timeline = Some(QuantumTimeline { layer_round_ends });
        let mut bloq = Bloq::new();
        let node = bloq.add_node(weight);
        assert_eq!(
            bloq.validate(),
            Err(BloqValidationError::InvalidQuantumTimeline { node })
        );
    }

    fn invalid_template_circuit_source(circuit: CoordCircuit) -> NodeTemplateInstanceMergeError {
        let mut bloq = Bloq::new();
        bloq.add_template(BloqTemplate::new(circuit));
        match bloq.validate().unwrap_err() {
            BloqValidationError::InvalidTemplateCircuit { source, .. } => source,
            error => panic!("expected invalid template circuit, got {error}"),
        }
    }

    fn sparse_measurement_circuit() -> CoordCircuit {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![q],
                measurements: vec![9],
                flip_probability: 0.0,
            });
        circuit.register_measurement_id(9, q);
        circuit
    }

    fn empty_repeat_circuit() -> (CoordCircuit, BodyId) {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::new());
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 2,
            });
        (circuit, body)
    }

    fn two_repeat_circuit() -> (CoordCircuit, BodyId, BodyId) {
        let mut circuit = CoordCircuit::new();
        let first = circuit.add_body(CircuitBody::new());
        let second = circuit.add_body(CircuitBody::new());
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend([
                Op::Repeat {
                    body: first,
                    repetitions: 2,
                },
                Op::Repeat {
                    body: second,
                    repetitions: 2,
                },
            ]);
        (circuit, first, second)
    }

    /// Builds the unrolled/looped template pair required by pipe padding.
    fn padding_template_pair(bloq: &mut Bloq) -> (TemplateId, TemplateId) {
        let plain = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut looped_circuit = CoordCircuit::new();
        let body = looped_circuit.add_body(CircuitBody::new());
        looped_circuit
            .body_mut(looped_circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 2,
            });
        let looped = bloq.add_template(BloqTemplate::new(looped_circuit));
        (plain, looped)
    }

    fn padding_ref(one_round: TemplateId, looped: TemplateId) -> PipePadding {
        PipePadding {
            offset: ivec2(0, 0),
            one_round,
            looped,
        }
    }

    fn set_padding(bloq: &mut Bloq, padding: PipePadding) {
        let lower = bloq.add_node(BloqNode::from_members(vec![]));
        let upper = bloq.add_node(BloqNode::from_members(vec![]));
        bloq.add_edge(
            lower,
            upper,
            BloqEdge::Quantum(Box::new(QuantumEdge {
                guard: None,
                pipes: vec![PipeSeam {
                    pipe: TemporalPipeRef {
                        src: ivec3(0, 0, 0),
                        dst: ivec3(0, 0, 1),
                        hadamard: false,
                    },
                    padding: Some(padding),
                }],
            })),
        );
    }

    #[test]
    fn well_formed_pipe_padding_validates() {
        let mut bloq = Bloq::new();
        let (plain, looped) = padding_template_pair(&mut bloq);
        set_padding(&mut bloq, padding_ref(plain, looped));
        bloq.validate()
            .expect("a plain one-round + looped pair validates");
    }

    #[test]
    fn pipe_padding_one_round_with_repeat_is_rejected() {
        let mut bloq = Bloq::new();
        let (plain, looped) = padding_template_pair(&mut bloq);
        set_padding(&mut bloq, padding_ref(looped, looped));
        let _ = plain;
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::PipePaddingOneRoundLooped { .. }
        ));
    }

    #[test]
    fn pipe_padding_looped_without_repeat_is_rejected() {
        let mut bloq = Bloq::new();
        let (plain, _looped) = padding_template_pair(&mut bloq);
        set_padding(&mut bloq, padding_ref(plain, plain));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::PipePaddingLoopedMalformed { .. }
        ));
    }

    #[test]
    fn cardinality_rejects_two_producers_on_one_value_slot() {
        let mut bloq = Bloq::new();
        let p0 = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let p1 = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::In(0),
        }));
        bloq.add_edge(p0, consumer, BloqEdge::value(0));
        bloq.add_edge(p1, consumer, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateValueSlot { slot: 0, .. }
        ));
    }

    #[test]
    fn cardinality_allows_value_fanout_and_distinct_slots() {
        let mut bloq = Bloq::new();
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let c0 = bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::In(0),
        }));
        let c1 = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        bloq.add_edge(producer, c0, BloqEdge::value(0));
        bloq.add_edge(producer, c1, BloqEdge::value(0));
        let other = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        bloq.add_edge(other, c1, BloqEdge::value(1));
        bloq.validate().unwrap();
    }

    #[test]
    fn validation_recurses_into_region_bodies() {
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let mut bad = node(ivec3(0, 0, 0));
        bad.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                crate::TemplateId(7), // no such template in the pool
                ivec2(0, 0),
            ));
        body.add_node(bad);
        // Const lets validation reach the nested body.
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplate { .. }
        ));
    }

    #[test]
    fn declared_outputs_validate_even_in_unconsumed_regions() {
        for invalid in [
            "result n99",
            "result n0:flip",
            "bindings n1",
            "bindings n99",
            "bindings n0 n0",
        ] {
            let program = Bloq::from_text(&format!(
                "BLOQIR 1
graph {{
 n0 rus 0 {{
 body {{
 n0 observable fragment
 n1 compute 1
 {invalid}
 }}
 }}
}}"
            ))
            .unwrap();
            let error = program.validate().unwrap_err();
            match invalid {
                "result n99" | "result n0:flip" => assert!(matches!(
                    error,
                    BloqValidationError::InvalidValueOutput { .. }
                )),
                "bindings n0 n0" => assert!(matches!(
                    error,
                    BloqValidationError::DuplicateBoundaryOutput { .. }
                )),
                _ => assert!(matches!(
                    error,
                    BloqValidationError::InvalidBoundaryOutput { .. }
                )),
            }
        }
    }

    #[test]
    fn independent_outputs_allow_multiple_unused_bits_and_local_consumers() {
        let program = Bloq::from_text(
            "BLOQIR 1
 graph {
   n0 rus 0 {
     body {
       n0 observable fragment
       n1 compute 1
       n2 compute !in0
       n3 compute 0
       n1 -> n2 value 0
       result n1
       bindings n0
     }
   }
   n1 observable 0
   n0 -> n1 compose 0
 }",
        )
        .unwrap();
        program.validate().unwrap();
    }

    #[test]
    fn region_selector_without_value_input_is_rejected() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::In(0),
            body: SubGraph::new(),
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::MissingRegionSelectorInput { slot: 0, .. }
        ));
    }

    #[test]
    fn region_selector_rejects_an_unused_value_input() {
        let mut bloq = Bloq::new();
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body: SubGraph::new(),
        }));
        bloq.add_edge(producer, region, BloqEdge::value(7));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnusedClassicalInput { node, slot: 7 } if node == region
        ));
    }

    #[test]
    fn quantum_timeline_must_be_multi_layer_ordered_positive_and_in_range() {
        for layer_round_ends in [vec![], vec![1], vec![0, 0], vec![2, 1]] {
            assert_invalid_timeline(node(ivec3(0, 0, 0)), layer_round_ends);
        }

        assert_invalid_timeline(node(ivec3(0, 0, i32::MAX)), vec![1, 2]);

        let mut bloq = Bloq::new();
        let mut weight = node(ivec3(0, 0, i32::MAX - 2));
        weight.expect_quantum_mut().timeline = Some(QuantumTimeline {
            layer_round_ends: vec![0, 0, 3],
        });
        bloq.add_node(weight);
        bloq.validate()
            .expect("equal cuts and empty leading layers are valid");
    }

    #[test]
    fn quantum_timeline_requires_nonempty_block_component_provenance() {
        for weight in [
            BloqNode::from_members(vec![]),
            BloqNode::from_temporal_pipe(TemporalPipeRef {
                src: ivec3(0, 0, 0),
                dst: ivec3(0, 0, 1),
                hadamard: false,
            }),
        ] {
            assert_invalid_timeline(weight, vec![1, 2]);
        }
    }

    #[test]
    fn cyclic_graph_is_rejected_not_panicked() {
        let mut bloq = Bloq::new();
        let a = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        let b = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(a, b, BloqEdge::value(0));
        bloq.add_edge(b, a, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::CyclicGraph
        ));
    }

    #[test]
    fn compute_slot_read_without_producer_is_rejected() {
        let mut bloq = Bloq::new();
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([
                ClassicalExpr::In(0),
                ClassicalExpr::In(1), // slot 1 has no incoming edge
            ])),
        }));
        bloq.add_edge(producer, compute, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::MissingClassicalInput { slot: 1, .. }
        ));
    }

    #[test]
    fn compute_value_edge_never_read_is_rejected() {
        let mut bloq = Bloq::new();
        let used = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let unused = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(used, compute, BloqEdge::value(0));
        bloq.add_edge(unused, compute, BloqEdge::value(1));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnusedClassicalInput { slot: 1, .. }
        ));
    }

    #[test]
    fn output_frame_stamp_on_non_compute_node_is_rejected() {
        let mut bloq = Bloq::new();
        bloq.add_node(
            BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })
            .with_provenance(crate::NodeProvenance::OutputFrame {
                port: ivec3(0, 0, 1),
                basis: crate::Basis::X,
            }),
        );
        let observable = bloq.add_node(
            BloqNode::classical(ClassicalNode::observable(0)).with_provenance(
                crate::NodeProvenance::OutputFrame {
                    port: ivec3(0, 0, 1),
                    basis: crate::Basis::Z,
                },
            ),
        );
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::OutputFrameNotCompute { node, .. } if node == observable
        ));
    }

    #[test]
    fn output_frame_stamp_inside_region_is_rejected() {
        let mut body = SubGraph::new();
        let nested = body.add_node(
            BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })
            .with_provenance(crate::NodeProvenance::OutputFrame {
                port: ivec3(0, 0, 1),
                basis: crate::Basis::X,
            }),
        );
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::OutputFrameNotCompute { node, .. } if node == nested
        ));
    }

    #[test]
    fn output_frame_missing_one_basis_is_rejected() {
        let mut bloq = Bloq::new();
        bloq.add_node(
            BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            })
            .with_provenance(crate::NodeProvenance::OutputFrame {
                port: ivec3(0, 0, 1),
                basis: crate::Basis::X,
            }),
        );
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::OutputFrameIncomplete { port } if port == ivec3(0, 0, 1)
        ));
    }

    #[test]
    fn logical_cut_owners_require_active_region_ancestors() {
        let port = ivec3(0, 0, 0);
        let mut base = Bloq::new();
        let template = base.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut owner = BloqNode::from_members(Vec::new());
        owner
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let mut body = SubGraph::new();
        body.add_node(owner);
        let restart_source = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        for basis in [crate::Basis::X, crate::Basis::Z] {
            base.add_node(
                BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                })
                .with_provenance(crate::NodeProvenance::OutputFrame { port, basis }),
            );
        }
        let cases = [
            (
                "source-free-rus",
                RegionNode::RepeatUntilSuccess {
                    restart_source: None,
                    restart_condition: ClassicalExpr::Const(false),
                    body: body.clone(),
                },
            ),
            (
                "rus",
                RegionNode::RepeatUntilSuccess {
                    restart_condition: ClassicalExpr::Const(false),
                    restart_source: Some(restart_source.into()),
                    body: body.clone(),
                },
            ),
        ];
        for (name, region) in cases {
            for activation in [None, Some(false), Some(true)] {
                let mut program = base.clone();
                let mut node = BloqNode::region(region.clone());
                node.activation = activation.map(|_| 1);
                let id = program.add_node(node);
                if let Some(value) = activation {
                    let guard = program.add_node(BloqNode::classical(ClassicalNode::Compute {
                        expr: ClassicalExpr::Const(value),
                    }));
                    program.add_edge(guard, id, BloqEdge::value(1));
                }
                // Exercise both metadata channels independently.
                for output in [false, true] {
                    program.set_logical_inputs(if output {
                        Vec::new()
                    } else {
                        vec![LogicalInput {
                            port,
                            instance: TemplateInstanceId(0),
                            x: PauliMap::empty(),
                            z: PauliMap::empty(),
                        }]
                    });
                    program.set_logical_outputs(if output {
                        vec![LogicalOutput {
                            port,
                            instance: TemplateInstanceId(0),
                            x: PauliMap::empty(),
                            z: PauliMap::empty(),
                        }]
                    } else {
                        Vec::new()
                    });
                    let result = program.validate();
                    if activation != Some(false) {
                        result.unwrap_or_else(|error| panic!("{name} {activation:?}: {error}"));
                    } else {
                        assert!(
                            matches!(
                                result,
                                Err(BloqValidationError::InvalidInstanceMergeStructure {
                                    source: NodeTemplateInstanceMergeError::InvalidMembership(_),
                                    ..
                                })
                            ),
                            "{name} {activation:?} output={output}: {result:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn output_frame_requires_pair_for_every_logical_output() {
        let port = ivec3(3, 4, 5);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut owner = BloqNode::from_members(Vec::new());
        owner
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        bloq.add_node(owner);
        bloq.set_logical_outputs(vec![LogicalOutput {
            port,
            instance: TemplateInstanceId(0),
            x: PauliMap::empty(),
            z: PauliMap::empty(),
        }]);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::OutputFrameIncomplete { port: missing } if missing == port
        ));
        let mut outputs = bloq.logical_outputs().to_vec();
        outputs[0].instance = TemplateInstanceId(9);
        bloq.set_logical_outputs(outputs);
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::LogicalOutputUnknownInstance {
                instance: TemplateInstanceId(9),
                ..
            }
        ));
    }

    #[test]
    fn logical_input_requires_existing_instance() {
        let port = ivec3(3, 4, 5);
        let instance = TemplateInstanceId(9);
        let mut bloq = Bloq::new();
        bloq.set_logical_inputs(vec![LogicalInput {
            port,
            instance,
            x: PauliMap::empty(),
            z: PauliMap::empty(),
        }]);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::LogicalInputUnknownInstance {
                port: missing_port,
                instance: missing_instance,
            } if missing_port == port && missing_instance == instance
        ));
    }

    #[test]
    fn complete_hand_built_output_frame_without_logical_metadata_is_allowed() {
        let mut bloq = Bloq::new();
        for basis in [crate::Basis::X, crate::Basis::Z] {
            bloq.add_node(
                BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                })
                .with_provenance(crate::NodeProvenance::OutputFrame {
                    port: ivec3(0, 0, 1),
                    basis,
                }),
            );
        }

        bloq.validate()
            .expect("complete hand-built frame pairs remain legal");
    }

    fn sparse_instance_bloq(measurement: u32) -> Bloq {
        let q = ivec2(0, 0);
        let mut bloq = Bloq::new();
        let template_id = bloq.add_template(BloqTemplate::new(sparse_measurement_circuit()));
        let mut producer = node(ivec3(0, 0, 0));
        producer
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template_id, q));
        let producer = bloq.add_node(producer);
        let accumulate = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }],
        }));
        bloq.add_edge(producer, accumulate, BloqEdge::Order);
        bloq
    }

    #[test]
    fn sparse_instance_measurement_ids_use_registry_membership() {
        sparse_instance_bloq(9).validate().unwrap();
        assert!(matches!(
            sparse_instance_bloq(0).validate().unwrap_err(),
            BloqValidationError::UnknownInstanceMeasurement { measurement: 0, .. }
        ));
    }

    #[test]
    fn complete_observable_checks_local_record_sites() {
        for measurement in [9, 0] {
            let mut bloq = sparse_instance_bloq(measurement);
            let read = bloq
                .nodes()
                .find(|(_, node)| {
                    matches!(
                        node.try_classical(),
                        Some(ClassicalNode::Observable { index: None, .. })
                    )
                })
                .unwrap()
                .0;
            let measurements = bloq[read].try_classical().unwrap().measurements().to_vec();
            bloq.node_mut(read).unwrap().kind = crate::BloqNodeKind::Classical(
                ClassicalNode::Observable {
                    index: Some(0),
                    measurements,
                    operators: Vec::new(),
                }
                .into(),
            );
            if measurement == 9 {
                bloq.validate().unwrap();
            } else {
                assert!(matches!(
                    bloq.validate(),
                    Err(BloqValidationError::UnknownInstanceMeasurement { measurement: 0, .. })
                ));
            }
        }
    }

    #[test]
    fn sparse_template_side_tables_use_registry_membership() {
        let mut template = BloqTemplate::new(sparse_measurement_circuit());
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([9]),
            coords: None,
        });
        template.boundary_flows.push(
            bloq_circuit::Flow::new(PauliMap::empty(), PauliMap::empty()).with_measurements([9]),
        );
        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_measurements([9]),
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        bloq.validate().unwrap();
    }

    #[test]
    fn unused_registry_measurement_is_legal_only_when_unreferenced() {
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(9, ivec2(0, 0));
        let mut bloq = Bloq::new();
        bloq.add_template(BloqTemplate::new(circuit.clone()));
        bloq.validate()
            .expect("an unreferenced registry-only record is legal");

        let mut template = BloqTemplate::new(circuit);
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([9]),
            coords: None,
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateMeasurement { measurement: 9, .. }
        ));
    }

    #[test]
    fn node_side_table_rejects_registry_only_measurement() {
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(9, ivec2(0, 0));
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut quantum = node(ivec3(0, 0, 0));
        quantum
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let producer = bloq.add_node(quantum);
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(0, 9)],
        }));
        bloq.add_edge(producer, consumer, BloqEdge::Order);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownInstanceMeasurement { measurement: 9, .. }
        ));
    }

    #[test]
    fn validation_checks_unused_registry_coordinate_translation() {
        let coordinate = ivec2(i32::MAX, 0);
        let offset = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(0, coordinate);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut quantum = node(ivec3(0, 0, 0));
        quantum
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                offset,
            ));
        bloq.add_node(quantum);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::CoordinateOverflow(_),
                ..
            }
        ));
    }

    #[test]
    fn unused_template_rejects_unknown_measurement_control() {
        let mut circuit = CoordCircuit::new();
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: 7,
                target: ivec2(0, 0),
            }]));
        let mut bloq = Bloq::new();
        bloq.add_template(BloqTemplate::new(circuit));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidTemplateCircuit {
                source: NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(7),
                ..
            }
        ));
    }

    #[test]
    fn template_parities_reject_unknown_measurements_and_loop_states() {
        let mut template = BloqTemplate::new(sparse_measurement_circuit());
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_measurements([0]),
            coords: None,
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateMeasurement { measurement: 0, .. }
        ));

        let mut template = BloqTemplate::new(sparse_measurement_circuit());
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::TopLevel,
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(7))]),
            coords: None,
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(7),
                ..
            }
        ));
    }

    #[test]
    fn node_parities_reject_loop_states() {
        let mut bloq = Bloq::new();
        let mut quantum = node(ivec3(0, 0, 0));
        quantum.expect_quantum_mut().detectors.push(NodeDetector {
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(0))]),
            coords: None,
        });
        let node = bloq.add_node(quantum);

        assert_eq!(
            bloq.validate().unwrap_err(),
            BloqValidationError::NodeLoopStateUnsupported {
                node,
                state: LoopStateId(0),
            }
        );
    }

    #[test]
    fn repeat_state_parities_reject_unknown_measurements() {
        let mut circuit = sparse_measurement_circuit();
        let body = circuit.add_body(CircuitBody::new());
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 2,
            });
        let mut template = BloqTemplate::new(circuit);
        template.repeat_states.push(TemplateRepeatState {
            body,
            state: LoopStateId(0),
            initial: DetectorParity::from_measurements([9]),
            next: DetectorParity::from_measurements([0]),
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateMeasurement { measurement: 0, .. }
        ));
    }

    #[test]
    fn duplicate_template_loop_state_ids_are_rejected() {
        let (circuit, body) = empty_repeat_circuit();
        let state = |state| TemplateRepeatState {
            body,
            state,
            initial: DetectorParity::default(),
            next: DetectorParity::default(),
        };
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            vec![state(LoopStateId(0)), state(LoopStateId(0))],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateTemplateLoopState {
                state: LoopStateId(0),
                ..
            }
        ));
    }

    #[test]
    fn validation_resource_diagnostics_retain_counts_and_semantic_messages() {
        use crate::BloqNodeId;
        use bloq_circuit::CircuitError;
        use bloq_utils::boolean::BooleanResourceError;

        let boolean = BooleanResourceError {
            resource: "Boolean work",
            observed: 11,
            limit: 10,
        };
        let direct = BloqValidationError::BooleanResource(boolean);
        assert!(direct.is_resource_limited());
        assert_eq!(
            direct.to_string(),
            "resource limit: Boolean work requires 11, limit is 10"
        );
        for (source, counts) in [
            (
                NodeTemplateInstanceMergeError::BooleanResource(boolean),
                "requires 11, limit is 10",
            ),
            (
                NodeTemplateInstanceMergeError::NoiseRepeatExpansion(crate::FlattenError::Circuit(
                    CircuitError::FlattenResourceLimit {
                        observed: 31,
                        limit: 30,
                    },
                )),
                "31 work units (limit 30)",
            ),
        ] {
            assert!(source.is_resource_limited());
            assert!(source.to_string().contains(counts));
            for error in [
                BloqValidationError::InvalidInstanceMergeStructure {
                    node: BloqNodeId(3),
                    source: source.clone(),
                },
                BloqValidationError::InvalidTemplateCircuit {
                    template: TemplateId(7),
                    source,
                },
            ] {
                assert!(error.is_resource_limited());
                let message = error.to_string();
                assert!(message.contains("resource limit"), "{message}");
                assert!(message.contains(counts), "{message}");
                assert!(
                    !message.contains("invalid") && !message.contains("do not merge"),
                    "{message}"
                );
            }
        }

        let source = NodeTemplateInstanceMergeError::TickSegmentCountMismatch;
        assert!(!source.is_resource_limited());
        let merge = BloqValidationError::InvalidInstanceMergeStructure {
            node: BloqNodeId(3),
            source: source.clone(),
        };
        assert!(!merge.is_resource_limited());
        assert_eq!(
            merge.to_string(),
            format!(
                "WF-9: node {:?} template instances do not merge: {source}",
                BloqNodeId(3)
            )
        );
        let template = BloqValidationError::InvalidTemplateCircuit {
            template: TemplateId(7),
            source: source.clone(),
        };
        assert!(!template.is_resource_limited());
        assert_eq!(
            template.to_string(),
            format!(
                "WF-9: template {:?} has invalid circuit structure: {source}",
                TemplateId(7)
            )
        );
        let circuit = CircuitError::InvalidCircuitBody(BodyId(9));
        let flatten = crate::FlattenError::Circuit(circuit.clone());
        assert!(!flatten.is_resource_limited());
        assert_eq!(
            flatten.to_string(),
            format!("flattening the circuit failed: {circuit}")
        );
        let noise = NodeTemplateInstanceMergeError::NoiseRepeatExpansion(
            crate::FlattenError::UnresolvedLoopState(LoopStateId(0)),
        );
        assert!(!noise.is_resource_limited());
        assert_eq!(
            noise.to_string(),
            "expanding repeat moments before idle noise: parity references undefined loop state LoopStateId(0)"
        );
    }

    #[test]
    fn shared_circuit_dag_validation_preserves_measurement_and_state_cuts() {
        let mut circuit = CoordCircuit::new();
        let qubit = ivec2(0, 0);
        let measurement = circuit.reserve_measurement_id(qubit);
        let leaf = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![measurement],
            flip_probability: 0.0,
        }]));
        let mut body = leaf;
        for _ in 0..48 {
            body = circuit.add_body(CircuitBody::from_ops(vec![
                Op::Repeat {
                    body,
                    repetitions: 1,
                },
                Op::Repeat {
                    body,
                    repetitions: 1,
                },
            ]));
        }
        circuit.push_repeat(body, 1);
        let state = LoopStateId(0);
        let mut template = BloqTemplate::new(circuit);
        template.repeat_states.push(TemplateRepeatState {
            body: leaf,
            state,
            initial: DetectorParity::default(),
            next: DetectorParity::from_measurements([measurement]),
        });
        template.detectors.push(TemplateDetector {
            scope: TemplateDetectorScope::RepeatBody { body: leaf },
            parity: DetectorParity::from_terms([
                DetectorTerm::LoopState(state),
                DetectorTerm::Measurement(measurement),
            ]),
            coords: None,
        });
        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_measurements([measurement]),
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template.clone());
        bloq.validate()
            .expect("shared bodies do not multiply validation work");

        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_terms([DetectorTerm::LoopState(state)]),
        });
        let mut invalid = Bloq::new();
        invalid.add_template(template);
        assert!(matches!(
            invalid.validate(),
            Err(BloqValidationError::UnknownTemplateLoopState { .. })
        ));
    }

    #[test]
    fn shared_body_state_requirements_are_checked_in_each_enclosing_scope() {
        for outside in [false, true] {
            let mut circuit = CoordCircuit::new();
            let owner = circuit.add_body(CircuitBody::new());
            let shared = circuit.add_body(CircuitBody::new());
            let outer = circuit.add_body(CircuitBody::from_ops(vec![
                Op::Repeat {
                    body: owner,
                    repetitions: 1,
                },
                Op::Repeat {
                    body: shared,
                    repetitions: 1,
                },
                Op::Repeat {
                    body: shared,
                    repetitions: 1,
                },
            ]));
            circuit.push_repeat(outer, 1);
            if outside {
                circuit.push_repeat(shared, 1);
            }
            let state = LoopStateId(0);
            let mut template = BloqTemplate::new(circuit);
            template.repeat_states.push(TemplateRepeatState {
                body: owner,
                state,
                initial: DetectorParity::default(),
                next: DetectorParity::default(),
            });
            template.detectors.push(TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body: shared },
                parity: DetectorParity::from_terms([DetectorTerm::LoopState(state)]),
                coords: None,
            });
            let mut bloq = Bloq::new();
            bloq.add_template(template);
            if outside {
                assert!(matches!(
                    bloq.validate(),
                    Err(BloqValidationError::UnknownTemplateLoopState { .. })
                ));
            } else {
                bloq.validate().unwrap();
            }
        }
    }

    #[test]
    fn deep_circuit_body_validation_does_not_use_recursive_calls() {
        let mut circuit = CoordCircuit::new();
        let mut body = circuit.add_body(CircuitBody::new());
        for _ in 0..4096 {
            body = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
                body,
                repetitions: 1,
            }]));
        }
        circuit.push_repeat(body, 1);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        let node = bloq.add_node(node);
        bloq.validate().unwrap();
        bloq.node_mut(node)
            .unwrap()
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(1),
                template,
                ivec2(1, 0),
            ));
        bloq.validate().unwrap();
        assert!(
            bloq.validate_with_options(&InstantiationOptions::default().with_boolean_limits(
                bloq_utils::boolean::BooleanLimits {
                    max_steps: 100,
                    ..bloq_utils::boolean::BooleanLimits::DEFAULT
                },
            ),)
                .unwrap_err()
                .is_resource_limited()
        );
    }

    #[test]
    fn repeat_state_initial_cannot_reference_itself() {
        let (circuit, body) = empty_repeat_circuit();
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            vec![TemplateRepeatState {
                body,
                state: LoopStateId(0),
                initial: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(0))]),
                next: DetectorParity::default(),
            }],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(0),
                ..
            }
        ));
    }

    #[test]
    fn repeat_state_initial_cannot_reference_measurement_from_its_body() {
        let measured = ivec2(0, 0);
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
            .push(Op::Repeat {
                body,
                repetitions: 2,
            });
        let mut template = BloqTemplate::new(circuit);
        template.repeat_states.push(TemplateRepeatState {
            body,
            state: LoopStateId(0),
            initial: DetectorParity::from_measurements([0]),
            next: DetectorParity::from_measurements([0]),
        });
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateMeasurement { measurement: 0, .. }
        ));
    }

    #[test]
    fn zero_count_repeat_skips_state_event_liveness() {
        let (mut circuit, body) = empty_repeat_circuit();
        let Op::Repeat { repetitions, .. } =
            &mut circuit.body_mut(circuit.entry_body()).unwrap().ops_mut()[0]
        else {
            panic!("fixture entry op is a repeat");
        };
        *repetitions = 0;
        let state = LoopStateId(0);
        let self_reference = DetectorParity::from_terms([DetectorTerm::LoopState(state)]);
        let template = BloqTemplate::with_parts(
            circuit,
            vec![TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body },
                parity: self_reference.clone(),
                coords: None,
            }],
            vec![TemplateRepeatState {
                body,
                state,
                initial: self_reference.clone(),
                next: self_reference,
            }],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        bloq.validate()
            .expect("a zero-count repeat has no state-liveness events");
    }

    #[test]
    fn zero_count_repeat_still_validates_measurement_references() {
        let (mut circuit, body) = empty_repeat_circuit();
        let Op::Repeat { repetitions, .. } =
            &mut circuit.body_mut(circuit.entry_body()).unwrap().ops_mut()[0]
        else {
            panic!("fixture entry op is a repeat");
        };
        *repetitions = 0;
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            vec![TemplateRepeatState {
                body,
                state: LoopStateId(0),
                initial: DetectorParity::from_measurements([99]),
                next: DetectorParity::default(),
            }],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateMeasurement {
                measurement: 99,
                ..
            }
        ));
    }

    #[test]
    fn repeat_state_initial_cannot_reference_a_peer_state() {
        let (circuit, body) = empty_repeat_circuit();
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
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
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(1),
                ..
            }
        ));
    }

    #[test]
    fn repeat_state_recurrence_and_post_loop_refs_validate() {
        let (circuit, body) = empty_repeat_circuit();
        let state = LoopStateId(0);
        let parity = DetectorParity::from_terms([DetectorTerm::LoopState(state)]);
        let template = BloqTemplate::with_parts(
            circuit,
            vec![
                TemplateDetector {
                    scope: TemplateDetectorScope::RepeatBody { body },
                    parity: parity.clone(),
                    coords: None,
                },
                TemplateDetector {
                    scope: TemplateDetectorScope::TopLevel,
                    parity: parity.clone(),
                    coords: None,
                },
            ],
            vec![TemplateRepeatState {
                body,
                state,
                initial: DetectorParity::default(),
                next: parity.clone(),
            }],
            Vec::new(),
            vec![TemplateRestart { parity }],
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        bloq.validate()
            .expect("current-state recurrence and post-loop refs are live");
    }

    #[test]
    fn repeat_state_next_cannot_reference_a_later_loop() {
        let (circuit, first, second) = two_repeat_circuit();
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            vec![
                TemplateRepeatState {
                    body: first,
                    state: LoopStateId(0),
                    initial: DetectorParity::default(),
                    next: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(1))]),
                },
                TemplateRepeatState {
                    body: second,
                    state: LoopStateId(1),
                    initial: DetectorParity::default(),
                    next: DetectorParity::default(),
                },
            ],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(1),
                ..
            }
        ));
    }

    #[test]
    fn repeat_body_detector_cannot_reference_a_later_loop() {
        let (circuit, first, second) = two_repeat_circuit();
        let template = BloqTemplate::with_parts(
            circuit,
            vec![TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body: first },
                parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(1))]),
                coords: None,
            }],
            vec![TemplateRepeatState {
                body: second,
                state: LoopStateId(1),
                initial: DetectorParity::default(),
                next: DetectorParity::default(),
            }],
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(1),
                ..
            }
        ));
    }

    #[test]
    fn restart_cannot_reference_state_dropped_with_enclosing_loop() {
        let mut circuit = CoordCircuit::new();
        let inner = circuit.add_body(CircuitBody::new());
        let outer = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
            body: inner,
            repetitions: 2,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body: outer,
                repetitions: 2,
            });
        let template = BloqTemplate::with_parts(
            circuit,
            Vec::new(),
            vec![TemplateRepeatState {
                body: inner,
                state: LoopStateId(0),
                initial: DetectorParity::default(),
                next: DetectorParity::default(),
            }],
            Vec::new(),
            vec![TemplateRestart {
                parity: DetectorParity::from_terms([DetectorTerm::LoopState(LoopStateId(0))]),
            }],
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateLoopState {
                state: LoopStateId(0),
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_dangling_boundary_flow_measurement() {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]); // only measurement id 0 exists
        let mut template = BloqTemplate::new(template_circuit);
        template.boundary_flows.push(
            bloq_circuit::Flow::new(
                bloq_circuit::PauliMap::empty(),
                bloq_circuit::PauliMap::empty(),
            )
            .with_measurements([1u32]),
        );

        let mut graph = Bloq::new();
        let template_id = graph.add_template(template);

        let error = graph.validate().unwrap_err();
        assert!(matches!(
            error,
            BloqValidationError::UnknownTemplateMeasurement {
                template,
                measurement: 1,
            } if template == template_id
        ));
    }

    #[test]
    fn validate_rejects_future_accumulate_measurement() {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]);
        let mut graph = Bloq::new();
        let template = graph.add_template(BloqTemplate::new(template_circuit));

        // Observable fragment sorts before its producer without an Order edge.
        let accumulate = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(1, 0)],
        }));

        let mut producer_node = node(ivec3(0, 0, 1));
        producer_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(1), template, q));
        let producer = graph.add_node(producer_node);
        let _ = accumulate;

        let error = graph.validate().unwrap_err();
        assert!(matches!(
            error,
            BloqValidationError::FutureMeasurement {
                owner,
                measurement: 0,
                ..
            } if owner == producer
        ));
        graph.node_mut(accumulate).unwrap().kind = crate::BloqNodeKind::Classical(
            ClassicalNode::Observable {
                index: Some(0),
                measurements: vec![instance_measurement(1, 0)],
                operators: Vec::new(),
            }
            .into(),
        );
        assert!(matches!(
            graph.validate(),
            Err(BloqValidationError::FutureMeasurement { owner, .. }) if owner == producer
        ));
        graph.add_edge(producer, accumulate, BloqEdge::Order);
        graph.validate().unwrap();
    }

    #[test]
    fn node_detector_requires_path_from_cross_node_measurement_owner() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [q]);
        let mut graph = Bloq::new();
        let template = graph.add_template(BloqTemplate::new(circuit));

        let mut producer_node = node(ivec3(0, 0, 1));
        producer_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(1), template, q));
        let producer = graph.add_node(producer_node);
        let mut consumer_node = node(ivec3(1, 0, 0));
        consumer_node
            .expect_quantum_mut()
            .detectors
            .push(NodeDetector {
                parity: DetectorParity::from_measurements([instance_measurement(1, 0)]),
                coords: None,
            });
        let consumer = graph.add_node(consumer_node);

        assert_eq!(
            graph.validate().unwrap_err(),
            BloqValidationError::UnorderedNodeMeasurement {
                node: consumer,
                owner: producer,
                measurement: 0,
            }
        );

        graph.add_edge(producer, consumer, BloqEdge::Order);
        graph.validate().unwrap();
    }

    #[test]
    fn bundle_detector_requires_path_from_cross_node_measurement_owner() {
        use crate::{BundleDetector, BundleMeasurement, DetectorBundle, DetectorBundleUse};
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut program = Bloq::new();
        let template = program.add_template(BloqTemplate::new(circuit));
        let bundle = program.add_detector_bundle(DetectorBundle::new(
            vec![template],
            vec![BundleDetector {
                parity: DetectorParity::from_measurements([BundleMeasurement {
                    owner: 0,
                    measurement: 0,
                }]),
                coords: None,
            }],
        ));
        let mut producer = node(ivec3(0, 0, 0));
        producer
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(1),
                template,
                ivec2(0, 0),
            ));
        let producer = program.add_node(producer);
        let mut consumer = node(ivec3(1, 0, 0));
        consumer
            .expect_quantum_mut()
            .detector_bundles
            .push(DetectorBundleUse {
                bundle,
                instances: vec![TemplateInstanceId(1)],
                offset: ivec2(0, 0),
            });
        let consumer = program.add_node(consumer);
        assert!(
            matches!(program.validate(), Err(BloqValidationError::UnavailableDetectorBundleOwner { node, instance: TemplateInstanceId(1) }) if node == consumer)
        );
        program.add_edge(producer, consumer, BloqEdge::Order);
        program.validate().unwrap();
    }

    #[test]
    fn detector_ancestor_search_matches_forward_reachability() {
        let mut graph = SubGraph::new();
        let nodes = (0..10)
            .map(|_| {
                graph.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                }))
            })
            .collect::<Vec<_>>();
        for (source, target) in [
            (0, 1),
            (1, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (2, 5),
            (6, 7),
            (7, 8),
        ] {
            graph.add_edge(nodes[source], nodes[target], BloqEdge::Order);
        }
        graph.remove_node(nodes[9]);
        for _ in 0..64 {
            let leaf = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
                index: None,
                operators: Vec::new(),
                measurements: Vec::new(),
            }));
            graph.add_edge(nodes[0], leaf, BloqEdge::Order);
        }
        let mut scratch = graph.path_scratch();
        for source in graph.node_ids() {
            for target in graph.node_ids() {
                assert_eq!(
                    graph.has_path_backwards_with_scratch(source, target, &mut scratch),
                    graph.has_path(source, target),
                    "{source:?} -> {target:?}"
                );
            }
        }
    }

    #[test]
    fn validate_checks_unused_template_repeat_state_body_refs() {
        let mut graph = Bloq::new();
        let mut circuit = CoordCircuit::new();
        let orphan = circuit.add_body(CircuitBody::new());
        let mut template = BloqTemplate::new(circuit);
        template.repeat_states.push(TemplateRepeatState {
            body: orphan,
            state: LoopStateId(0),
            initial: DetectorParity::default(),
            next: DetectorParity::default(),
        });
        graph.add_template(template);
        graph.add_node(node(ivec3(0, 0, 0)));

        let error = graph.validate().unwrap_err();
        assert!(matches!(
            error,
            BloqValidationError::UnknownTemplateBody {
                body,
                ..
            } if body == orphan
        ));
    }

    #[test]
    fn repeat_body_detector_on_non_repeat_body_is_rejected() {
        // IR-01: reachable bodies now come from `preflight_circuit`'s visited
        // set minus the entry, replacing the former single-source merge. A
        // detector scoped to a body that is never a REPEAT target — here the
        // entry body, which preflight visits but deliberately excludes — must
        // still fail template-body validation.
        let circuit = CoordCircuit::new();
        let entry = circuit.entry_body();
        let template = BloqTemplate::with_parts(
            circuit,
            vec![TemplateDetector {
                scope: TemplateDetectorScope::RepeatBody { body: entry },
                parity: DetectorParity::default(),
                coords: None,
            }],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut bloq = Bloq::new();
        bloq.add_template(template);

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnknownTemplateBody { body, .. } if body == entry
        ));
    }

    #[test]
    fn validate_rejects_empty_mpp_in_unused_template_without_panicking() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::MPP {
                products: vec![PauliMap::empty()],
                measurements: vec![0],
            });
        circuit.register_measurement_id(0, q);
        let mut bloq = Bloq::new();
        bloq.add_template(BloqTemplate::new(circuit));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidTemplateCircuit {
                source: NodeTemplateInstanceMergeError::EmptyPauliProduct,
                ..
            }
        ));
    }

    #[test]
    fn validate_rejects_measure_output_count_mismatch_in_unused_template() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(0, q);
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![q],
                measurements: vec![],
                flip_probability: 0.0,
            });

        assert_eq!(
            invalid_template_circuit_source(circuit),
            NodeTemplateInstanceMergeError::MeasureOutputCountMismatch {
                qubits: 1,
                measurements: 0,
            }
        );
    }

    #[test]
    fn validate_rejects_odd_two_qubit_target_count_in_unused_template() {
        let mut circuit = CoordCircuit::new();
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Gate {
                gate: GateType::CX,
                qubits: vec![ivec2(0, 0)],
            });

        assert_eq!(
            invalid_template_circuit_source(circuit),
            NodeTemplateInstanceMergeError::OddTwoQubitTargetCount {
                gate: GateType::CX,
                targets: 1,
            }
        );
    }

    #[test]
    fn validate_rejects_unregistered_measure_and_mpp_outputs() {
        let q = ivec2(0, 0);
        let mut measure = CoordCircuit::new();
        measure
            .body_mut(measure.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![q],
                measurements: vec![7],
                flip_probability: 0.0,
            });
        assert_eq!(
            invalid_template_circuit_source(measure),
            NodeTemplateInstanceMergeError::UnregisteredMeasurementOutput(7)
        );

        let mut mpp = CoordCircuit::new();
        mpp.body_mut(mpp.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::MPP {
                products: vec![[(q, Pauli::X)].into_iter().collect()],
                measurements: vec![9],
            });
        assert_eq!(
            invalid_template_circuit_source(mpp),
            NodeTemplateInstanceMergeError::UnregisteredMeasurementOutput(9)
        );
    }

    #[test]
    fn validate_rejects_measurement_id_produced_by_multiple_ops() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.register_measurement_id(0, q);
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend([
                Op::Measure {
                    basis: PauliBasis::Z,
                    qubits: vec![q],
                    measurements: vec![0],
                    flip_probability: 0.0,
                },
                Op::MPP {
                    products: vec![[(q, Pauli::X)].into_iter().collect()],
                    measurements: vec![0],
                },
            ]);

        assert_eq!(
            invalid_template_circuit_source(circuit),
            NodeTemplateInstanceMergeError::DuplicateMeasurementOutput(0)
        );
    }

    #[test]
    fn validate_rejects_measure_and_mpp_output_coordinate_mismatches() {
        let produced = ivec2(0, 0);
        let registered = ivec2(1, 0);
        let mut measure = CoordCircuit::new();
        measure.register_measurement_id(0, registered);
        measure
            .body_mut(measure.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![produced],
                measurements: vec![0],
                flip_probability: 0.0,
            });
        assert_eq!(
            invalid_template_circuit_source(measure),
            NodeTemplateInstanceMergeError::MeasurementOutputCoordinateMismatch {
                measurement: 0,
                registered,
                produced,
            }
        );

        let mut mpp = CoordCircuit::new();
        mpp.register_measurement_id(0, registered);
        mpp.body_mut(mpp.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::MPP {
                products: vec![[(produced, Pauli::X)].into_iter().collect()],
                measurements: vec![0],
            });
        assert_eq!(
            invalid_template_circuit_source(mpp),
            NodeTemplateInstanceMergeError::MeasurementOutputCoordinateMismatch {
                measurement: 0,
                registered,
                produced,
            }
        );
    }

    #[test]
    fn validate_rejects_cyclic_repeat_body_in_unused_template() {
        let mut circuit = CoordCircuit::new();
        let entry = circuit.entry_body();
        circuit.body_mut(entry).unwrap().ops_mut().push(Op::Repeat {
            body: entry,
            repetitions: 2,
        });
        let mut bloq = Bloq::new();
        bloq.add_template(BloqTemplate::new(circuit));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidTemplateCircuit {
                source: NodeTemplateInstanceMergeError::CyclicBody(BodyId(0)),
                ..
            }
        ));
    }

    #[test]
    fn discard_inside_rus_body_is_rejected() {
        let mut body = SubGraph::new();
        let producer = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let discard = body.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::In(0),
        }));
        body.add_edge(producer, discard, BloqEdge::value(0));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(producer.into()),
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DiscardInsideRepeatUntilSuccess { .. }
        ));
    }

    #[test]
    fn nested_rus_is_rejected() {
        let mut inner_body = SubGraph::new();
        let inner_restart = inner_body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let inner = BloqNode::region(RegionNode::RepeatUntilSuccess {
            body: inner_body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(inner_restart.into()),
        });

        let mut outer_body = SubGraph::new();
        let outer_restart = outer_body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        outer_body.add_node(inner);
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body: outer_body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(outer_restart.into()),
        }));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::NestedRepeatUntilSuccess { .. }
        ));
    }

    #[test]
    fn retry_wrapped_include_validates() {
        // Cross-level observable bindings resolve program-global instances.
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut producer = node(ivec3(0, 0, 0));
        producer
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        bloq.add_node(producer);

        let mut body = SubGraph::new();
        body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            measurements: Vec::new(),
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: bloq_circuit::PauliMap::empty(),
            }],
        }));
        body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![],
        }));
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        bloq.validate().unwrap();
    }

    #[test]
    fn boundary_payload_with_unknown_instance_is_rejected_at_every_level() {
        for nested in [false, true] {
            let operators = vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(99),
                face: BoundaryFace::Input,
                operator: PauliMap::empty(),
            }];
            for classical in [
                ClassicalNode::Observable {
                    index: None,
                    measurements: Vec::new(),
                    operators: operators.clone(),
                },
                ClassicalNode::Observable {
                    index: Some(0),
                    measurements: Vec::new(),
                    operators,
                },
            ] {
                let include = BloqNode::classical(classical);
                let mut bloq = Bloq::new();
                if nested {
                    let mut body = SubGraph::new();
                    body.add_node(include);
                    bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
                        restart_source: None,
                        restart_condition: ClassicalExpr::Const(false),
                        body,
                    }));
                } else {
                    bloq.add_node(include);
                }

                assert!(matches!(
                    bloq.validate().unwrap_err(),
                    BloqValidationError::UnknownTemplateInstance {
                        instance: TemplateInstanceId(99),
                        ..
                    }
                ));
            }
        }
    }

    #[test]
    fn template_restart_with_unknown_measurement_is_rejected() {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]); // only measurement id 0 exists
        let mut template = BloqTemplate::new(template_circuit);
        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_measurements([1u32]),
        });

        let mut graph = Bloq::new();
        let template_id = graph.add_template(template);

        let error = graph.validate().unwrap_err();
        assert!(matches!(
            error,
            BloqValidationError::UnknownTemplateMeasurement {
                template,
                measurement: 1,
            } if template == template_id
        ));
    }

    #[test]
    fn rus_region_with_restart_condition_and_template_restarts_validates() {
        // Hand-built RUS with template restart and matching observable fragment.
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]); // measurement 0
        let mut template = BloqTemplate::new(template_circuit);
        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_measurements([0u32]),
        });

        let mut graph = Bloq::new();
        let template_id = graph.add_template(template);

        let mut body = SubGraph::new();
        let mut inst_node = node(ivec3(0, 0, 0));
        inst_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template_id, q));
        let producer = body.add_node(inst_node);
        let accumulate = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(0, 0)],
        }));
        body.add_edge(producer, accumulate, BloqEdge::Order);

        let selector = graph.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![],
        }));
        let region = graph.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(accumulate.into()),
        }));
        graph.add_edge(selector, region, BloqEdge::value(0));

        graph.validate().unwrap();
    }

    /// Builds a RUS with a NodeRestart over one measurement.
    fn rus_bloq_with_node_restart(measurement: u32) -> Bloq {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]); // measurement 0
        let mut graph = Bloq::new();
        let template_id = graph.add_template(BloqTemplate::new(template_circuit));

        let mut body = SubGraph::new();
        let mut inst_node = node(ivec3(0, 0, 0));
        inst_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template_id, q));
        inst_node.expect_quantum_mut().restarts.push(NodeRestart {
            parity: DetectorParity::from_measurements([InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement,
            }]),
        });
        body.add_node(inst_node);
        let restart_source = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![],
        }));
        graph.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(restart_source.into()),
        }));
        graph
    }

    #[test]
    fn node_restart_inside_rus_validates() {
        rus_bloq_with_node_restart(0).validate().unwrap();
    }

    #[test]
    fn node_restart_with_unknown_measurement_is_rejected() {
        assert!(matches!(
            rus_bloq_with_node_restart(1) // only measurement id 0 exists
                .validate()
                .unwrap_err(),
            BloqValidationError::UnknownInstanceMeasurement { measurement: 1, .. }
        ));
    }

    #[test]
    fn node_restart_outside_rus_is_rejected() {
        let mut graph = Bloq::new();
        let mut restart_node = node(ivec3(0, 0, 0));
        restart_node
            .expect_quantum_mut()
            .restarts
            .push(NodeRestart {
                parity: DetectorParity::default(),
            });
        graph.add_node(restart_node);
        assert!(matches!(
            graph.validate().unwrap_err(),
            BloqValidationError::RestartOutsideRepeatUntilSuccess { .. }
        ));
    }

    #[test]
    fn restart_template_instantiated_outside_rus_is_rejected() {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]);
        let mut template = BloqTemplate::new(template_circuit);
        template.restarts.push(TemplateRestart {
            parity: DetectorParity::from_measurements([0u32]),
        });

        let mut graph = Bloq::new();
        let template_id = graph.add_template(template);
        let mut inst_node = node(ivec3(0, 0, 0));
        inst_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template_id, q));
        graph.add_node(inst_node);
        assert!(matches!(
            graph.validate().unwrap_err(),
            BloqValidationError::RestartOutsideRepeatUntilSuccess { .. }
        ));
    }

    #[test]
    fn indexed_observable_exposes_corrected_and_flip_values() {
        let mut bloq = Bloq::new();
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(3)));
        for output in [ObservableOutput::Corrected, ObservableOutput::Flip] {
            let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }));
            bloq.add_edge(
                observable,
                consumer,
                BloqEdge::Value {
                    slot: 0,
                    role: crate::ValueRole::Data,
                    output,
                },
            );
        }
        bloq.validate().unwrap();
    }

    #[test]
    fn flip_requires_an_indexed_observable() {
        for producer in [
            ClassicalNode::observable_fragment(Vec::new(), Vec::new()),
            ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            },
        ] {
            let mut bloq = Bloq::new();
            let source = bloq.add_node(BloqNode::classical(producer));
            let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }));
            bloq.add_edge(
                source,
                consumer,
                BloqEdge::Value {
                    slot: 0,
                    role: crate::ValueRole::Data,
                    output: ObservableOutput::Flip,
                },
            );
            assert!(
                matches!(bloq.validate(), Err(BloqValidationError::InvalidValueProducer { producer, .. }) if producer == source)
            );
            bloq.top_mut().set_value_output(Some(ValueRef {
                node: source,
                output: ObservableOutput::Flip,
            }));
            bloq.remove_node(consumer);
            assert!(matches!(
                bloq.validate(),
                Err(BloqValidationError::InvalidValueOutput { .. })
            ));
        }
    }

    #[test]
    fn value_edge_into_quantum_node_is_rejected() {
        let mut bloq = Bloq::new();
        let quantum = bloq.add_node(node(ivec3(0, 0, 0)));
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        bloq.add_edge(producer, quantum, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::UnusedQuantumValueInput { node, slot: 0 }
                if node == quantum
        ));
    }

    #[test]
    fn duplicate_observable_index_is_rejected() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::observable(5)));
        bloq.add_node(BloqNode::classical(ClassicalNode::observable(5)));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateObservableIndex { index: 5, .. }
        ));
    }

    #[test]
    fn duplicate_observable_index_across_region_scopes_is_rejected() {
        let mut body = SubGraph::new();
        body.add_node(BloqNode::classical(ClassicalNode::observable(5)));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::classical(ClassicalNode::observable(5)));
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateObservableIndex { index: 5, .. }
        ));
    }

    /// Builds a RUS-body instance referenced by a top-level detector.
    fn bloq_with_top_level_ref_into_rus_body(
        instance: TemplateInstanceId,
        measurement: u32,
    ) -> Bloq {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]); // measurement 0
        let mut bloq = Bloq::new();
        let template_id = bloq.add_template(BloqTemplate::new(template_circuit));

        let mut body = SubGraph::new();
        let mut inst_node = node(ivec3(0, 0, 0));
        inst_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template_id, q));
        body.add_node(inst_node);
        let restart_source = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![],
        }));
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: Some(restart_source.into()),
        }));

        let mut neighbor = node(ivec3(0, 0, 1));
        neighbor
            .expect_quantum_mut()
            .detectors
            .push(crate::NodeDetector {
                parity: DetectorParity::from_measurements([InstanceMeasurement {
                    instance,
                    measurement,
                }]),
                coords: None,
            });
        let neighbor = bloq.add_node(neighbor);
        bloq.add_edge(region, neighbor, BloqEdge::Order);
        bloq
    }

    /// The checkless plan path a trusted consumer emits from must agree with
    /// the validating one, including for nodes nested inside a region body.
    #[test]
    fn emission_plans_match_the_validating_path() {
        let bloq = bloq_with_top_level_ref_into_rus_body(TemplateInstanceId(0), 0);
        let options = InstantiationOptions::default();
        let validated = bloq
            .validate_with_plans(&options)
            .expect("program is valid");
        let plans_only = bloq.emission_plans(&options).expect("plans build");

        assert_eq!(validated.len(), plans_only.len());
        assert!(!plans_only.is_empty(), "fixture has quantum nodes to plan");
        for ((path, node), plan) in validated.iter() {
            // `NodeEmissionPlan` is not `PartialEq` (its circuit is not), so
            // compare the id maps the backends resolve through plus the merged
            // circuit's shape.
            let other = plans_only
                .get(path, node)
                .unwrap_or_else(|| panic!("checkless path is missing {path:?}/{node:?}"));
            assert_eq!(plan.measurements, other.measurements);
            assert_eq!(plan.bodies, other.bodies);
            assert_eq!(
                plan.circuit.num_measurements(),
                other.circuit.num_measurements()
            );
            assert_eq!(plan.circuit.body_count(), other.circuit.body_count());
        }
    }

    #[test]
    fn top_level_detector_referencing_unknown_instance_is_rejected() {
        assert!(matches!(
            bloq_with_top_level_ref_into_rus_body(TemplateInstanceId(7), 0)
                .validate()
                .unwrap_err(),
            BloqValidationError::UnknownTemplateInstance { .. }
        ));
    }

    #[test]
    fn top_level_detector_with_out_of_range_body_measurement_is_rejected() {
        assert!(matches!(
            bloq_with_top_level_ref_into_rus_body(TemplateInstanceId(0), 1)
                .validate()
                .unwrap_err(),
            BloqValidationError::UnknownInstanceMeasurement { measurement: 1, .. }
        ));
    }

    #[test]
    fn duplicate_instance_id_across_region_boundary_is_rejected() {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]);
        let mut bloq = Bloq::new();
        let template_id = bloq.add_template(BloqTemplate::new(template_circuit));
        let instance = |offset| TemplateInstance::new(TemplateInstanceId(5), template_id, offset);

        let mut top = node(ivec3(0, 0, 0));
        top.expect_quantum_mut().instances.push(instance(q));
        bloq.add_node(top);

        let mut body = SubGraph::new();
        let mut inner = node(ivec3(1, 0, 0));
        inner
            .expect_quantum_mut()
            .instances
            .push(instance(ivec2(2, 0)));
        body.add_node(inner);
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));

        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateTemplateInstance {
                instance: TemplateInstanceId(5)
            }
        ));
    }

    #[test]
    fn duplicated_in_leaf_is_rejected() {
        // Backend folds each producer once, so duplicate linear leaves miscompile.
        let mut bloq = Bloq::new();
        let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Xor(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(0)])),
        }));
        bloq.add_edge(producer, compute, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateExprInput { node, slot: 0 } if node == compute
        ));
    }

    #[test]
    fn value_edge_from_quantum_node_is_rejected() {
        let mut bloq = Bloq::new();
        let quantum = bloq.add_node(node(ivec3(0, 0, 0)));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(quantum, compute, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidValueProducer { producer, .. } if producer == quantum
        ));
    }

    #[test]
    fn value_edge_from_discard_is_rejected() {
        let mut bloq = Bloq::new();
        let discard = bloq.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::Const(false),
        }));
        let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(discard, compute, BloqEdge::value(0));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidValueProducer { producer, .. } if producer == discard
        ));
    }

    #[test]
    fn composition_requires_an_observable_consumer() {
        for compose in [false, true] {
            let mut bloq = Bloq::new();
            let fragment = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                Vec::new(),
            )));
            let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(
                fragment,
                observable,
                if compose {
                    BloqEdge::compose(0)
                } else {
                    BloqEdge::value(0)
                },
            );
            bloq.validate().unwrap();
        }
        let mut bloq = Bloq::new();
        let fragment = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(fragment, consumer, BloqEdge::compose(0));
        assert!(
            matches!(bloq.validate(), Err(BloqValidationError::InvalidValueProducer { node, .. }) if node == consumer)
        );
    }

    #[test]
    fn shared_readout_recipe_roundtrips_and_rejects_invalid_uses() {
        let program = Bloq::from_text(
            "BLOQIR 1
template t0 {
  circuit {
    H (0,0)
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 compute 0
  n2 compute 1
  n3 observable fragment operators i0 output Z(0,0) when v0
  n4 observable fragment operators i0 output X(0,0)
  n5 observable 0
  n6 compute in0
  n7 observable fragment
  n8 observable fragment
  n9 observable 1
  n10 compute in0
  n11 observable 2
  n12 compute in0
  n13 observable fragment when v9
  n0 -> n3 order
  n0 -> n4 order
  n1 -> n3 value 0
  n1 -> n5 value 0
  n5 -> n6 value 0
  n2 -> n7 value 0
  n6 -> n7 value 1 readout
  n2 -> n7 value 2 feedback 7
  n3 -> n7 compose 3
  n4 -> n7 compose 4
  n7 -> n8 compose 0
  n1 -> n8 value 1
  n8 -> n9 compose 0
  n9 -> n10 value 0
  n1 -> n13 value 9
  n8 -> n13 compose 0
  n13 -> n11 compose 0
  n11 -> n12 value 0
}",
        )
        .unwrap();
        program.validate().unwrap();
        let text = Bloq::from_text(&program.to_text()).unwrap();
        let binary = Bloq::from_binary(&program.to_binary()).unwrap();
        assert_eq!(program.to_binary(), text.to_binary());
        assert_eq!(program.to_binary(), binary.to_binary());
        let resolved = program
            .resolve_classical(
                crate::BloqNodeId(10),
                crate::ClassicalAssignment::Uniform(false),
            )
            .unwrap();
        assert_eq!(resolved.decoder_observables, [0, 1].into_iter().collect());
        let pinned = program
            .pin_membership(&std::collections::BTreeMap::new())
            .unwrap();
        pinned.validate().unwrap();
        assert_eq!(pinned.value_inputs(crate::BloqNodeId(13)).count(), 0);
        assert_eq!(pinned[crate::BloqNodeId(13)].activation, None);
        let mut optimized = program.clone();
        optimized.optimize().expect("acyclic test program");
        optimized.validate().unwrap();
        for node in [crate::BloqNodeId(9), crate::BloqNodeId(11)] {
            let original = program
                .resolve_classical(node, crate::ClassicalAssignment::Uniform(false))
                .unwrap();
            for edited in [&pinned, &optimized] {
                assert_eq!(
                    edited
                        .resolve_classical(node, crate::ClassicalAssignment::Uniform(false))
                        .unwrap(),
                    original
                );
            }
        }

        let mut bad = program.clone();
        let compute = bad.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bad.add_edge(crate::BloqNodeId(8), compute, BloqEdge::compose(0));
        assert!(
            matches!(bad.validate(), Err(BloqValidationError::InvalidValueProducer { node, .. }) if node == compute)
        );
        let mut bad = program.clone();
        bad.add_edge(
            crate::BloqNodeId(8),
            crate::BloqNodeId(9),
            BloqEdge::Compose {
                slot: 1,
                role: crate::ValueRole::ReadoutFold,
            },
        );
        assert!(matches!(
            bad.validate(),
            Err(BloqValidationError::InvalidObservableFold { .. })
        ));
        let mut bad = program;
        bad.add_edge(
            crate::BloqNodeId(8),
            crate::BloqNodeId(7),
            BloqEdge::compose(9),
        );
        assert!(matches!(
            bad.validate(),
            Err(BloqValidationError::CyclicGraph)
        ));
    }

    #[test]
    fn observable_folds_require_a_bit_source_and_observable_target() {
        for role in [
            crate::ValueRole::FeedbackFold { action: 0 },
            crate::ValueRole::ReadoutFold,
        ] {
            let mut bloq = Bloq::new();
            let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            }));
            let compute = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::In(0),
            }));
            bloq.add_edge(
                producer,
                compute,
                BloqEdge::Value {
                    slot: 0,
                    output: ObservableOutput::Corrected,
                    role: role.clone(),
                },
            );

            assert!(matches!(
                bloq.validate().unwrap_err(),
                BloqValidationError::InvalidObservableFold { node, .. } if node == compute
            ));

            let mut bloq = Bloq::new();
            let include = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
                index: None,
                measurements: Vec::new(),
                operators: Vec::new(),
            }));
            let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(include, observable, BloqEdge::Compose { slot: 0, role });

            assert!(matches!(
                bloq.validate().unwrap_err(),
                BloqValidationError::InvalidObservableFold { producer, .. } if producer == include
            ));
        }
    }

    #[test]
    fn duplicate_output_frame_stamp_is_rejected() {
        let mut bloq = Bloq::new();
        for _ in 0..2 {
            bloq.add_node(
                BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(false),
                })
                .with_provenance(crate::NodeProvenance::OutputFrame {
                    port: ivec3(0, 0, 1),
                    basis: crate::Basis::X,
                }),
            );
        }
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::DuplicateOutputFrameStamp { port } if port == ivec3(0, 0, 1)
        ));
    }

    #[test]
    fn rus_explicit_restart_source_binds_despite_stray_open_producer() {
        let mut body = SubGraph::new();
        let predicate = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let _stray = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(predicate.into()),
        }));
        bloq.validate().unwrap();
    }

    #[test]
    fn rus_restart_source_naming_missing_body_node_is_rejected() {
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body: SubGraph::new(),
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(crate::BloqNodeId(9).into()),
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidRestartSource {
                restart_source: ValueRef {
                    node: crate::BloqNodeId(9),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn rus_restart_source_naming_non_producer_is_rejected() {
        let mut body = SubGraph::new();
        let binding = body.add_node(BloqNode::classical(ClassicalNode::Discard {
            condition: ClassicalExpr::Const(false),
        }));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(binding.into()),
        }));
        assert!(matches!(
            bloq.validate().unwrap_err(),
            BloqValidationError::InvalidRestartSource { .. }
        ));
    }

    /// A small program with one measuring quantum node. Returns the program and
    /// that node's id.
    fn single_measure_program() -> (Bloq, crate::BloqNodeId) {
        let q = ivec2(0, 0);
        let mut template_circuit = CoordCircuit::new();
        template_circuit.measure(PauliBasis::Z, [q]);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(template_circuit));
        let mut quantum = node(ivec3(0, 0, 0));
        quantum
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(TemplateInstanceId(0), template, q));
        let id = bloq.add_node(quantum);
        (bloq, id)
    }

    #[test]
    fn validate_with_plans_returns_per_node_emission_plan() {
        let (bloq, quantum) = single_measure_program();
        let plans = bloq
            .validate_with_plans(&InstantiationOptions::default())
            .expect("the program is well-formed");

        let stored = plans
            .get(&LevelPath::default(), quantum)
            .expect("a quantum node has a plan");
        assert_eq!(
            stored.measurements,
            [(instance_measurement(0, 0), 0)].into_iter().collect()
        );
        assert_eq!(
            stored.bodies,
            [((TemplateInstanceId(0), BodyId(0)), BodyId(0))]
                .into_iter()
                .collect()
        );
        assert_eq!(
            stored.grouped_measurements(),
            vec![(0, vec![instance_measurement(0, 0)])]
        );
        assert_eq!(stored.circuit.entry_body(), BodyId(0));
        assert_eq!(
            stored.circuit.body(BodyId(0)).unwrap().ops(),
            &[Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![ivec2(0, 0)],
                measurements: vec![0],
                flip_probability: 0.0,
            }]
        );
    }

    #[test]
    fn validate_with_plans_agrees_with_validate_on_failure() {
        // An observable fragment naming a measurement id the instance never produces fails
        // validation; both entry points must reject it identically.
        let (mut bloq, quantum) = single_measure_program();
        let accumulate = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![InstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 1, // only measurement id 0 exists
            }],
        }));
        bloq.add_edge(quantum, accumulate, BloqEdge::Order);

        let via_validate = bloq.validate();
        let via_plans = bloq
            .validate_with_plans(&InstantiationOptions::default())
            .map(|_| ());
        assert_eq!(via_validate, via_plans);
        assert!(matches!(
            via_validate.unwrap_err(),
            BloqValidationError::UnknownInstanceMeasurement {
                instance: TemplateInstanceId(0),
                measurement: 1,
                ..
            }
        ));
    }

    #[test]
    fn relative_merge_cache_reuses_proofs_without_changing_measurement_aliases() {
        use bloq_utils::boolean::BooleanLimits;

        let mut circuit = CoordCircuit::new();
        for _ in 0..128 {
            circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
            circuit.tick();
        }
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        for index in 0..32 {
            let mut quantum = node(ivec3(0, 0, index));
            quantum
                .expect_quantum_mut()
                .instances
                .extend((0..2).map(|member| {
                    TemplateInstance::new(
                        TemplateInstanceId(2 * index as u32 + member),
                        template,
                        ivec2(index, 0),
                    )
                }));
            bloq.add_node(quantum);
        }
        let options = InstantiationOptions::default().with_boolean_limits(BooleanLimits {
            max_steps: 8_000,
            ..BooleanLimits::default()
        });
        bloq.validate_with_options(&options)
            .expect("one merge proof plus each placement fits the work budget");
        assert!(
            bloq.validate_with_plans(&options)
                .unwrap_err()
                .is_resource_limited(),
            "materializing every plan still charges its circuit work"
        );

        let plans = bloq
            .validate_with_plans(&InstantiationOptions::default())
            .expect("retained plans perform the same successful merges");
        assert_eq!(plans.iter().count(), 32);
        for (_, plan) in plans.iter() {
            let groups = plan.grouped_measurements();
            assert_eq!(groups.len(), 128);
            assert!(groups.iter().all(|(_, sources)| sources.len() == 2));
        }
    }

    #[test]
    fn relative_merge_cache_distinguishes_placement_and_preserves_extrema() {
        let mut bloq = Bloq::new();
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [ivec2(0, 0)]).unwrap();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        for (index, offsets) in [[i32::MIN, i32::MAX], [i32::MAX, i32::MIN], [0, 1], [10, 11]]
            .into_iter()
            .enumerate()
        {
            let mut quantum = node(ivec3(0, 0, index as i32));
            quantum
                .expect_quantum_mut()
                .instances
                .extend(offsets.into_iter().enumerate().map(|(member, offset)| {
                    TemplateInstance::new(
                        TemplateInstanceId((2 * index + member) as u32),
                        template,
                        ivec2(offset, 0),
                    )
                }));
            bloq.add_node(quantum);
        }
        assert_eq!(bloq.validate(), Ok(()));
        assert_eq!(
            bloq.validate_with_plans(&InstantiationOptions::default())
                .map(|_| ()),
            Ok(())
        );

        let mut quantum = node(ivec3(0, 0, 4));
        quantum.expect_quantum_mut().instances.extend(
            (8..10).map(|id| TemplateInstance::new(TemplateInstanceId(id), template, ivec2(10, 0))),
        );
        bloq.add_node(quantum);
        let cached = bloq.validate();
        assert_eq!(
            cached,
            bloq.validate_with_plans(&InstantiationOptions::default())
                .map(|_| ())
        );
        assert!(matches!(
            cached,
            Err(BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::QubitConflict(_),
                ..
            })
        ));
    }

    #[test]
    fn relative_merge_cache_checks_unused_absolute_coordinates() {
        for (coordinate, last_valid, overflow) in [
            (ivec2(1, 0), ivec2(i32::MAX - 1, 0), ivec2(i32::MAX, 0)),
            (ivec2(0, -1), ivec2(0, i32::MIN + 1), ivec2(0, i32::MIN)),
        ] {
            let mut circuit = CoordCircuit::new();
            circuit.register_measurement_id(0, coordinate);
            let mut bloq = Bloq::new();
            let template = bloq.add_template(BloqTemplate::new(circuit));
            for (index, offset) in [ivec2(0, 0), last_valid, overflow].into_iter().enumerate() {
                let mut quantum = node(ivec3(0, 0, index as i32));
                quantum
                    .expect_quantum_mut()
                    .instances
                    .push(TemplateInstance::new(
                        TemplateInstanceId(index as u32),
                        template,
                        offset,
                    ));
                bloq.add_node(quantum);
                if index < 2 {
                    assert_eq!(bloq.validate(), Ok(()));
                }
            }
            let cached = bloq.validate();
            assert_eq!(
                cached,
                bloq.validate_with_plans(&InstantiationOptions::default())
                    .map(|_| ())
            );
            assert!(matches!(
                cached,
                Err(BloqValidationError::InvalidInstanceMergeStructure {
                    source: NodeTemplateInstanceMergeError::CoordinateOverflow(_),
                    ..
                })
            ));
        }
    }

    #[test]
    fn relative_merge_cache_requires_zero_repeat_measurement_remapping() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::ConditionalPauli(vec![
            ConditionalCorrection {
                pauli: PauliBasis::X,
                control: 0,
                target: ivec2(0, 0),
            },
        ])]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Repeat {
                body,
                repetitions: 0,
            });
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        assert_eq!(
            bloq.validate(),
            Ok(()),
            "source preflight skips the zero repeat"
        );
        let mut quantum = node(ivec3(0, 0, 0));
        quantum
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                ivec2(0, 0),
            ));
        bloq.add_node(quantum);
        let cached = bloq.validate();
        assert_eq!(
            cached,
            bloq.validate_with_plans(&InstantiationOptions::default())
                .map(|_| ())
        );
        assert!(matches!(
            cached,
            Err(BloqValidationError::InvalidInstanceMergeStructure {
                source: NodeTemplateInstanceMergeError::ControlReferencesUnknownMeasurement(0),
                ..
            })
        ));
    }
}
