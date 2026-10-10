//! Program-level detector slices.
//!
//! This is Layer 2 of the detector-slice viewer: flatten the [`Bloq`], build straight-line
//! tapes along the static executed path, resolve detector and logical-observable
//! contributions into tape space, run the reverse sparse frame tracker, then
//! scatter slices back to node timelines.
//!
//! This module is program-tape slicing over the whole [`Bloq`];
//! `bloq_circuit::detslice` is the reverse sparse frame tracker it drives per
//! straight-line tape.

use std::sync::Arc;

use bloq_circuit::{
    DetectorCoords, DetectorParity, DetsliceError, Op, Pauli, PauliBasis, PauliMap,
    RegionBreakKind, RegionTerm, SliceDetector, SliceRegionSeed, detector_slices_with_seeds,
};
use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use crate::instantiation::InstantiationOptions;
use crate::{
    Bloq, BloqNodeId, BloqNodeKind, BloqValidationError, BodySelector, BoundaryFace, ClassicalExpr,
    ClassicalNode, CycleDetected, FlattenError, InstanceMeasurement, LevelPath, NodeDetectorParity,
    SubGraph, TemplateInstanceId, ValidatedPlans,
};

// ==============================================================================
// Moment bucketing (shared with the editor timeline)
// ==============================================================================

/// The kind of a viewer moment: a bucket ops fall into within one tick segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MomentKind {
    /// Reset operation.
    Reset,
    /// Single-qubit rotation.
    Rotation,
    /// Two-qubit interaction.
    Interaction,
    /// Measurement operation.
    Measurement,
}

/// Which moment bucket an op belongs to, or `None` if it is timeline-invisible.
///
/// Mirrors `bloq_editor::program_view::op_kind` with two deliberate deltas so
/// the tracker sees exactly the ops that carry stabilizer flow:
///
/// - [`Op::MPP`] joins the [`Measurement`](MomentKind::Measurement) bucket. A
///   product measurement births a region, and the editor consumes this same
///   bucketing function so its visible MPP moments cannot drift from the tape.
/// - [`Op::ConditionalPauli`] has no visible moment kind. Measurement-controlled
///   corrections are folded into an adjacent visible moment by
///   [`moment_segments`] so the reverse tracker sees them; value-controlled
///   corrections remain transparent.
fn op_moment_kind(op: &Op) -> Option<MomentKind> {
    match op {
        Op::Gate { gate, .. } if gate.is_reset() => Some(MomentKind::Reset),
        Op::Gate { gate, .. } if gate.is_two_qubit_gate() => Some(MomentKind::Interaction),
        Op::Gate { .. } => Some(MomentKind::Rotation),
        Op::Measure { .. } | Op::MPP { .. } => Some(MomentKind::Measurement),
        // Ticks segment the stream; ConditionalPauli is injected into an
        // adjacent visible bucket by `bucket_tick_segment`; Repeat is not
        // expected on a flattened tape (see `moment_segments`).
        Op::Tick
        | Op::ConditionalPauli(_)
        | Op::Repeat { .. }
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. } => None,
    }
}

/// One viewer moment: a bucket of visible same-kind ops plus any invisible
/// measurement-controlled feedforward assigned to that boundary, in op order.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentSegment {
    /// Common operation kind.
    pub kind: MomentKind,
    /// Operations in source order.
    pub ops: Vec<Op>,
}

/// Incremental viewer-moment segmentation for structural circuit walkers.
///
/// Measurement-controlled corrections from an all-invisible chunk are carried
/// into the next chunk, so callers may split at repeat boundaries without
/// changing the moment order. Call [`Self::finish`] after the final chunk when
/// the returned segments will be sent to the reverse tracker.
#[derive(Debug, Default)]
pub struct MomentSegmenter {
    carried_corrections: Vec<Op>,
}

impl MomentSegmenter {
    /// Segment one straight-line chunk while retaining trailing invisible
    /// feedforward for the next chunk.
    pub fn extend(&mut self, ops: &[Op]) -> Vec<MomentSegment> {
        self.extend_owned(ops.to_vec())
    }

    /// Owned half of [`Self::extend`], used when the caller already has an op
    /// buffer and can move it into the moment buckets.
    fn extend_owned(&mut self, ops: Vec<Op>) -> Vec<MomentSegment> {
        let mut out = Vec::new();
        let mut segment = Vec::new();
        for op in ops {
            if matches!(op, Op::Tick) {
                bucket_tick_segment(
                    std::mem::take(&mut segment),
                    &mut self.carried_corrections,
                    &mut out,
                );
            } else {
                segment.push(op);
            }
        }
        bucket_tick_segment(segment, &mut self.carried_corrections, &mut out);
        out
    }

    /// Attach trailing invisible feedforward to the last visible segment.
    /// With no visible segment there is nothing to display or track.
    pub fn finish(mut self, out: &mut [MomentSegment]) {
        if let Some(last) = out.last_mut() {
            last.ops.append(&mut self.carried_corrections);
        }
    }
}

/// Split a straight-line op stream into viewer moments, first at [`Op::Tick`].
///
/// Every tick segment uses consecutive same-kind runs in source order: even
/// without feedforward, sequential operations may reuse a qubit (SEM-MERGE).
/// Invisible corrections stay at their operation boundary. Empty runs are
/// dropped. The editor timeline consumes this same implementation.
///
/// Expects a flattened (straight-line) stream: an [`Op::Repeat`] would be
/// silently dropped by `op_moment_kind`, corrupting the segmentation, so
/// callers flatten first (Layer 2 always does).
pub fn moment_segments(ops: &[Op]) -> Vec<MomentSegment> {
    moment_segments_owned(ops.to_vec())
}

fn moment_segments_owned(ops: Vec<Op>) -> Vec<MomentSegment> {
    let mut segmenter = MomentSegmenter::default();
    let mut out = segmenter.extend_owned(ops);
    segmenter.finish(&mut out);
    out
}

/// Preserve source order without creating a visible correction-only moment.
/// Corrections before the first visible op prefix its run; later corrections
/// stay at the end of the current run, and an all-invisible tick carries them
/// to the next visible run.
fn bucket_tick_segment(
    segment: Vec<Op>,
    carried_corrections: &mut Vec<Op>,
    out: &mut Vec<MomentSegment>,
) {
    let mut runs = Vec::<MomentSegment>::new();
    let mut pending = std::mem::take(carried_corrections);

    for op in segment {
        let op = match op {
            Op::ConditionalPauli(corrections) => {
                if !corrections.is_empty() {
                    let correction = Op::ConditionalPauli(corrections);
                    if let Some(run) = runs.last_mut() {
                        run.ops.push(correction);
                    } else {
                        pending.push(correction);
                    }
                }
                continue;
            }
            op => op,
        };

        let Some(kind) = op_moment_kind(&op) else {
            continue;
        };
        if runs.last().is_none_or(|run| run.kind != kind) {
            runs.push(MomentSegment {
                kind,
                ops: std::mem::take(&mut pending),
            });
        }
        runs.last_mut()
            .expect("a visible op creates a source-order run")
            .ops
            .push(op);
    }

    if runs.is_empty() {
        carried_corrections.append(&mut pending);
    } else {
        debug_assert!(
            pending.is_empty(),
            "the final visible run must absorb pending corrections"
        );
        out.extend(runs);
    }
}

// ==============================================================================
// Public output types
// ==============================================================================

/// A tape-unique node reference.
///
/// [`BloqNodeId`]s are level-local (they
/// restart at 0 in every region body [`SubGraph`]), so a bare id cannot key
/// across levels; the `path` disambiguates by naming the region bodies descended
/// into (empty for the top level).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeRef {
    /// Region bodies descended into to reach this node, top level first. Empty
    /// for a top-level node.
    pub path: LevelPath,
    /// The node's id within its own level.
    pub node: BloqNodeId,
}

impl NodeRef {
    /// A top-level node's reference (empty path).
    pub fn top_level(node: BloqNodeId) -> Self {
        Self {
            path: LevelPath::default(),
            node,
        }
    }

    /// Whether this refers to a top-level node.
    pub fn is_top_level(&self) -> bool {
        self.path.is_top_level()
    }
}

/// Detector and logical-observable region choices for a pinned program.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramSliceOptions {
    detectors_only: bool,
    physical_observables: bool,
}

impl ProgramSliceOptions {
    /// Show each observable's physical Pauli support without its runtime sign
    /// folds (`FeedbackFold` and `ReadoutFold`). Plain data expressions still
    /// have to be affine; this is not a Boolean-to-parity approximation.
    pub fn set_physical_observables(&mut self, enabled: bool) {
        self.physical_observables = enabled;
    }

    /// Omit logical-observable regions while retaining the same quantum tapes.
    pub fn set_detectors_only(&mut self, enabled: bool) {
        self.detectors_only = enabled;
    }
}

/// Stable identity of a detector or logical-observable region.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProgramRegionId {
    /// A detector from one quantum node's side table.
    Detector {
        /// Node that owns the detector table.
        owner: NodeRef,
        /// Detector index within that table.
        detector: u32,
    },
    /// A logical observable, matching Stim's `L{index}` region identity.
    Observable {
        /// Logical-observable index.
        index: u32,
    },
}

/// Identity and coordinates stored once for every program region.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramRegion {
    /// Stable region identity.
    pub id: ProgramRegionId,
    /// Optional detector coordinates.
    pub coords: Option<Arc<DetectorCoords>>,
}

/// One detector or logical-observable region at one moment.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionView {
    /// Index into [`ProgramSlices::regions`].
    pub region: u32,
    /// The region's Pauli terms, sorted by qubit.
    pub terms: Vec<RegionTerm>,
}

/// Per-moment regions for one node.
///
/// `0[m]` is the regions alive at the end of the
/// node's `m`-th moment. Length equals the node's [`moment_segments`] count, so
/// an empty inner vec means "moment exists, no live regions".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeSlices(pub Vec<Vec<RegionView>>);

/// A gauge anticommutation located back to the node timeline it falls in.
///
/// A break means the supplied region is not preserved by the tracked circuit.
/// Consumers visualizing a non-Clifford program through a Clifford gate proxy
/// must recompute the program's observables for that same proxy first.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramRegionBreak {
    /// The node whose moment the break lands in (the broken segment's node).
    pub node: NodeRef,
    /// Node-local moment index within `node`.
    pub moment: usize,
    /// The detector or logical observable whose region broke.
    pub region: ProgramRegionId,
    /// Qubit where anticommutation occurred.
    pub qubit: IVec2,
    /// Kind of region break.
    pub kind: RegionBreakKind,
}

/// The whole program's detector-slice data, keyed by node.
#[derive(Debug, Clone, Default)]
pub struct ProgramSlices {
    /// Region identity and detector coordinates, stored once rather than copied
    /// into every moment where the region is alive.
    pub regions: Vec<ProgramRegion>,
    /// Every quantum node (top-level and region-body) with its per-moment
    /// regions.
    pub per_node: FxHashMap<NodeRef, NodeSlices>,
    /// Gauge breaks, located back to their node timeline.
    pub breaks: Vec<ProgramRegionBreak>,
    /// How many detector or observable regions were dropped because a
    /// contribution reached an instance outside its own tape.
    pub skipped_cross_tape: usize,
}

/// Why [`program_detector_slices`] could not build the tape.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProgramSliceError {
    /// Detector-slice region ids exhausted their `u32` index space.
    #[error("detector-slice region id space overflowed")]
    RegionIdOverflow,
    /// A detector bundle use is malformed.
    #[error("{0}")]
    DetectorBundle(#[from] crate::DetectorBundleError),
    /// Conditional membership must be pinned before slicing.
    #[error("detector slices require pinned component membership")]
    MembershipSelectionRequired,
    /// A quantum emission plan could not be materialized.
    #[error("quantum plan materialization failed: {0}")]
    Materialization(#[from] BloqValidationError),
    /// A requested region refers to a measurement no emitted stage provides.
    #[error("region {region:?} references unavailable measurement {measurement:?}")]
    UnknownMeasurement {
        /// Region containing the reference.
        region: ProgramRegionId,
        /// Unavailable measurement.
        measurement: InstanceMeasurement,
    },
    /// A requested region refers to an instance absent from the program.
    #[error("region {region:?} references unavailable instance {instance:?}")]
    UnknownInstance {
        /// Region containing the reference.
        region: ProgramRegionId,
        /// Unavailable instance.
        instance: TemplateInstanceId,
    },
    /// Repeat flattening failed.
    #[error("flattening the program failed: {0}")]
    Flatten(#[from] FlattenError),
    /// A graph level is cyclic.
    #[error("a graph level has no topological emit order (cycle)")]
    Cycle(#[from] CycleDetected),
    /// An observable expression is nonlinear.
    #[error("node {node:?} folds a nonlinear observable expression (SEM-FOLD)")]
    NonLinearExpr {
        /// Node containing the nonlinear expression.
        node: NodeRef,
    },
    /// Stabilizer tracking failed.
    #[error("detector-slice tracking failed: {0}")]
    Detslice(#[from] DetsliceError),
    /// A gauge break could not be assigned to a moment.
    #[error("region {region:?} produced a gauge break outside every circuit moment")]
    BreakOutsideTimeline {
        /// Region that broke outside the timeline.
        region: ProgramRegionId,
    },
}

fn checked_region_base(base: usize, additional: usize) -> Result<u32, ProgramSliceError> {
    let count = base
        .checked_add(additional)
        .ok_or(ProgramSliceError::RegionIdOverflow)?;
    u32::try_from(count).map_err(|_| ProgramSliceError::RegionIdOverflow)?;
    u32::try_from(base).map_err(|_| ProgramSliceError::RegionIdOverflow)
}

// ==============================================================================
// Driver
// ==============================================================================

/// Compute detector-slice regions for every quantum node of `program`.
///
/// Clones and flattens `program` (repeats unrolled — regions genuinely differ
/// per iteration), then builds the executed-path tape and runs Layer 1 over it,
/// spawning each non-executed region arm as its own tape. See the module docs
/// for the tape model.
///
/// # Errors
///
/// See [`ProgramSliceError`]: plan materialization, flattening, a cyclic level, an
/// unmergeable node's instances, the tracker rejecting the (post-flatten) op
/// stream, or a boundary-seeded break that cannot be placed because the tape
/// has no moment, or a nonlinear observable record fold (SEM-FOLD).
pub fn program_detector_slices(program: &Bloq) -> Result<ProgramSlices, ProgramSliceError> {
    program_detector_slices_with_options(program, &ProgramSliceOptions::default())
}

/// Compute detector-slice regions using explicit region choices.
///
/// [`ProgramSliceOptions::set_detectors_only`] omits logical-observable regions
/// while retaining the same quantum tapes and detector regions.
/// [`ProgramSliceOptions::set_physical_observables`] keeps logical regions but
/// omits their known runtime sign folds. The default includes those folds and
/// requires their complete record expressions to be affine.
///
/// # Errors
///
/// See [`ProgramSliceError`].
pub fn program_detector_slices_with_options(
    program: &Bloq,
    options: &ProgramSliceOptions,
) -> Result<ProgramSlices, ProgramSliceError> {
    if program.has_conditional_membership() {
        return Err(ProgramSliceError::MembershipSelectionRequired);
    }
    let mut program = program.clone();
    program.flatten()?;
    // SEM-MERGE/IR-02: build exactly the quantum plans tracked below. Full IR
    // validation is an explicit audit, separate from this viewer operation.
    let plans = program.emission_plans(&InstantiationOptions::default())?;
    let known_measurements: FxHashSet<_> = plans
        .iter()
        .flat_map(|(_, plan)| plan.measurements.keys().copied())
        .collect();
    let known_instances: FxHashMap<_, _> = program
        .levels()
        .flat_map(|(_, level)| {
            level.quantum_nodes().flat_map(|(_, quantum)| {
                quantum
                    .instances
                    .iter()
                    .map(|instance| (instance.id, instance.template_id))
            })
        })
        .collect();
    for (_, level) in program.levels() {
        for (_, quantum) in level.quantum_nodes() {
            program.check_detector_bundle_bindings(quantum, |instance| {
                known_instances.get(&instance).copied()
            })?;
        }
    }

    let mut out = ProgramSlices::default();
    let mut tape = TapeBuilder::default();
    add_level_to_tape(
        &program,
        program.top(),
        &LevelPath::default(),
        &plans,
        options,
        &mut tape,
    )?;
    tape.finish(&known_measurements, &known_instances, &mut out)?;
    Ok(out)
}

fn add_level_to_tape(
    program: &Bloq,
    level: &SubGraph,
    path: &LevelPath,
    plans: &ValidatedPlans,
    options: &ProgramSliceOptions,
    tape: &mut TapeBuilder,
) -> Result<(), ProgramSliceError> {
    for id in level.deterministic_emit_order()? {
        match &level[id].kind {
            BloqNodeKind::Quantum(_) => tape.add_node(program, path, id, &level[id], plans)?,
            BloqNodeKind::Region(region) => {
                for (selector, subgraph) in region.bodies() {
                    add_level_to_tape(
                        program,
                        subgraph,
                        &path.child(id, selector),
                        plans,
                        options,
                        tape,
                    )?;
                }
            }
            BloqNodeKind::Classical(classical) => {
                if !options.detectors_only
                    && let ClassicalNode::Observable {
                        index: Some(index), ..
                    } = classical.as_ref()
                {
                    tape.stage_observable(level, path, id, *index, options)?;
                }
            }
        }
    }
    Ok(())
}

fn evaluate_expression(
    level: &SubGraph,
    path: &LevelPath,
    node: BloqNodeId,
    expr: &ClassicalExpr,
) -> Option<bool> {
    evaluate_expression_cached(level, path, node, expr, &mut FxHashMap::default())
}

fn evaluate_expression_cached(
    level: &SubGraph,
    path: &LevelPath,
    node: BloqNodeId,
    expr: &ClassicalExpr,
    cache: &mut FxHashMap<NodeRef, Option<bool>>,
) -> Option<bool> {
    expr.eval(&mut |slot| {
        let producer = level
            .value_inputs(node)
            .find(|input| input.slot == slot)?
            .producer;
        evaluate_producer(level, path, producer, cache)
    })
}

fn evaluate_producer(
    level: &SubGraph,
    path: &LevelPath,
    producer: BloqNodeId,
    cache: &mut FxHashMap<NodeRef, Option<bool>>,
) -> Option<bool> {
    let node_ref = NodeRef {
        path: path.clone(),
        node: producer,
    };
    if let Some(value) = cache.get(&node_ref) {
        return *value;
    }
    cache.insert(node_ref.clone(), None);
    if let Some(slot) = level[producer].activation
        && evaluate_expression_cached(level, path, producer, &ClassicalExpr::In(slot), cache)
            == Some(false)
    {
        cache.insert(node_ref, Some(false));
        return Some(false);
    }
    let value = match level[producer].try_classical()? {
        ClassicalNode::Compute { expr } => {
            evaluate_expression_cached(level, path, producer, expr, cache)
        }
        ClassicalNode::Observable { .. } | ClassicalNode::Discard { .. } => None,
    };
    cache.insert(node_ref, value);
    value
}

// ==============================================================================
// Tape construction
// ==============================================================================

/// A positioned Pauli contribution to a logical observable.
#[derive(Clone)]
struct PendingPauliSeed {
    instance: TemplateInstanceId,
    face: BoundaryFace,
    operator: PauliMap,
}

/// A detector or logical observable staged for resolution once the tape's
/// whole measurement and instance-boundary maps are built.
struct PendingRegion {
    id: ProgramRegionId,
    coords: Option<DetectorCoords>,
    /// The parity's measurement terms, in instance space.
    terms: Vec<InstanceMeasurement>,
    /// Positioned Pauli-target contributions. Empty for detectors.
    seeds: Vec<PendingPauliSeed>,
}

/// Accumulates one level's tape: the concatenated moment segments, the
/// per-node segment ranges, shared instance maps, and regions staged for
/// resolution.
#[derive(Default)]
struct TapeBuilder {
    /// One entry per tape segment: the moment ops (measurement ids already
    /// rewritten to tape-global space).
    segments: Vec<Vec<Op>>,
    /// For each tape segment, the node it belongs to and its node-local moment
    /// index — so a break's segment maps back to a node timeline.
    segment_owner: Vec<(NodeRef, usize)>,
    /// Each quantum node's `(NodeRef, tape start, moment count)`. Present even
    /// for a node with zero moments, so every node appears in `per_node`.
    node_ranges: Vec<(NodeRef, usize, usize)>,
    /// Instance-space measurement → tape-global id, spanning every node in the
    /// tape (a cross-node parity resolves against all of them).
    instance_to_global: FxHashMap<InstanceMeasurement, u32>,
    /// Template instance → `(input boundary, output boundary)` in segment space.
    instance_boundaries: FxHashMap<TemplateInstanceId, (usize, usize)>,
    /// Detector and observable regions staged for resolution after the maps are
    /// complete.
    pending: Vec<PendingRegion>,
    /// Next tape-global measurement id to hand out.
    next_global: u32,
}

impl TapeBuilder {
    /// Add one quantum node's ops and detector side table to the tape.
    fn add_node(
        &mut self,
        program: &Bloq,
        path: &LevelPath,
        id: BloqNodeId,
        node: &crate::BloqNode,
        plans: &ValidatedPlans,
    ) -> Result<(), ProgramSliceError> {
        let node_ref = NodeRef {
            path: path.clone(),
            node: id,
        };
        let plan = plans
            .get(path, id)
            .expect("materialization returns one emission plan per quantum node (IR-02)");
        let ops = plan
            .circuit
            .body(plan.circuit.entry_body())
            .map(bloq_circuit::CircuitBody::ops)
            .unwrap_or_default();

        // Assign a tape-global id to each distinct node-local measurement id, in
        // op order, and rewrite the ops to that space.
        let mut local_to_global: FxHashMap<u32, u32> = FxHashMap::default();
        for op in ops {
            for &local in op_measurement_ids(op) {
                local_to_global.entry(local).or_insert_with(|| {
                    let global = self.next_global;
                    self.next_global += 1;
                    global
                });
            }
        }
        let rewritten: Vec<Op> = ops
            .iter()
            .map(|op| rewrite_measurement_ids(op, &local_to_global))
            .collect();

        // Extend the tape's shared instance→global map with this node's
        // instances. Each instance belongs to exactly one node, so no key
        // collides across the tape.
        for (&instance_measurement, &local) in &plan.measurements {
            if let Some(&global) = local_to_global.get(&local) {
                self.instance_to_global.insert(instance_measurement, global);
            }
        }

        // Bucket into moments and record the node's tape range.
        let moments = moment_segments_owned(rewritten);
        let start = self.segments.len();
        for (moment, segment) in moments.into_iter().enumerate() {
            self.segment_owner.push((node_ref.clone(), moment));
            self.segments.push(segment.ops);
        }
        let len = self.segments.len() - start;
        self.node_ranges.push((node_ref.clone(), start, len));

        let quantum = node.expect_quantum();
        for instance in &quantum.instances {
            self.instance_boundaries
                .insert(instance.id, (start, start + len));
        }
        self.stage_detectors(program, &node_ref, quantum)?;
        Ok(())
    }

    /// Stage this node's template and node detectors for later resolution,
    /// assigning each a stable index into the node's detector table (template
    /// detectors first — instance then table order — then node detectors, as the
    /// Stim emitter's `node_annotations` orders them).
    fn stage_detectors(
        &mut self,
        program: &Bloq,
        node_ref: &NodeRef,
        quantum: &crate::QuantumNode,
    ) -> Result<(), ProgramSliceError> {
        let template_count = quantum
            .instances
            .iter()
            .try_fold(0usize, |count, instance| {
                let rows = program
                    .templates()
                    .get(instance.template_id)
                    .expect("emission plan resolved template")
                    .detectors
                    .len();
                count
                    .checked_add(rows)
                    .ok_or(crate::DetectorBundleError::CountOverflow)
            })?;
        let count = template_count
            .checked_add(program.node_detector_count(quantum)?)
            .ok_or(crate::DetectorBundleError::CountOverflow)?;
        u32::try_from(count).map_err(|_| crate::DetectorBundleError::CountOverflow)?;
        let mut index = 0u32;
        for instance in &quantum.instances {
            let template = program
                .templates()
                .get(instance.template_id)
                .expect("emission plan already resolved every instance template");
            for detector in &template.detectors {
                // Template-local measurement ids name this instance.
                let terms = detector
                    .parity
                    .measurements()
                    .map(|measurement| InstanceMeasurement {
                        instance: instance.id,
                        measurement,
                    })
                    .collect();
                self.pending.push(PendingRegion {
                    id: ProgramRegionId::Detector {
                        owner: node_ref.clone(),
                        detector: index,
                    },
                    coords: detector.coords.as_ref().map(|coords| {
                        bloq_circuit::translate_detector_coords(coords, instance.offset)
                    }),
                    terms,
                    seeds: Vec::new(),
                });
                index += 1;
            }
        }
        for detector in program.node_detectors(quantum)? {
            self.pending.push(PendingRegion {
                id: ProgramRegionId::Detector {
                    owner: node_ref.clone(),
                    detector: index,
                },
                // Node detector coords are already in layout space.
                coords: detector
                    .coords()
                    .map(|coords| DetectorCoords::from_slice(&coords.collect::<Vec<_>>())),
                terms: detector.measurements().collect(),
                seeds: Vec::new(),
            });
            index += 1;
        }
        Ok(())
    }

    /// Stage one logical observable: its record-backed XOR parity plus positioned
    /// boundary operators, including content exported by selected region bodies.
    fn stage_observable(
        &mut self,
        level: &SubGraph,
        path: &LevelPath,
        node: BloqNodeId,
        index: u32,
        options: &ProgramSliceOptions,
    ) -> Result<(), ProgramSliceError> {
        let content = ObservableCollector {
            cache: FxHashMap::default(),
            physical_observables: options.physical_observables,
        }
        .inputs(level, path, node)?;

        self.pending.push(PendingRegion {
            id: ProgramRegionId::Observable { index },
            coords: None,
            terms: content.parity.measurements().collect(),
            seeds: content.seeds,
        });
        Ok(())
    }

    /// Resolve staged regions against the completed tape maps, run Layer 1, and
    /// scatter slices and breaks into `out`.
    fn finish(
        self,
        known_measurements: &FxHashSet<InstanceMeasurement>,
        known_instances: &FxHashMap<TemplateInstanceId, crate::TemplateId>,
        out: &mut ProgramSlices,
    ) -> Result<(), ProgramSliceError> {
        // Resolve record parities and positioned Pauli seeds into tape space;
        // drop any region reaching an instance outside this tape.
        let (detectors, seeds, meta) =
            {
                let mut detectors = Vec::with_capacity(self.pending.len());
                let mut seeds = Vec::new();
                let mut meta: Vec<ProgramRegion> = Vec::with_capacity(self.pending.len());
                for pending in self.pending {
                    let PendingRegion {
                        id: region_id,
                        coords,
                        terms,
                        seeds: pending_seeds,
                    } = pending;
                    let parity = DetectorParity::from_measurements(terms);
                    let mut measurements = Vec::with_capacity(parity.terms().len());
                    let mut cross_tape = false;
                    for term in parity.measurements() {
                        match self.instance_to_global.get(&term) {
                            Some(&global) => measurements.push(global),
                            None => {
                                if !known_measurements.contains(&term) {
                                    return Err(ProgramSliceError::UnknownMeasurement {
                                        region: region_id,
                                        measurement: term,
                                    });
                                }
                                cross_tape = true;
                                break;
                            }
                        }
                    }
                    let pending_seeds = canonicalize_pauli_seeds(pending_seeds);
                    let mut resolved_seeds = Vec::with_capacity(pending_seeds.len());
                    for seed in pending_seeds {
                        let Some(&(input, output)) = self.instance_boundaries.get(&seed.instance)
                        else {
                            if !known_instances.contains_key(&seed.instance) {
                                return Err(ProgramSliceError::UnknownInstance {
                                    region: region_id,
                                    instance: seed.instance,
                                });
                            }
                            cross_tape = true;
                            break;
                        };
                        resolved_seeds.push((
                            match seed.face {
                                BoundaryFace::Input => input,
                                BoundaryFace::Output => output,
                            },
                            pauli_region_terms(&seed.operator),
                        ));
                    }
                    if cross_tape {
                        out.skipped_cross_tape += 1;
                        continue;
                    }
                    let id = u32::try_from(meta.len())
                        .map_err(|_| ProgramSliceError::RegionIdOverflow)?;
                    detectors.push(SliceDetector { id, measurements });
                    seeds.extend(resolved_seeds.into_iter().map(|(boundary, terms)| {
                        SliceRegionSeed {
                            id,
                            boundary,
                            terms,
                        }
                    }));
                    meta.push(ProgramRegion {
                        id: region_id,
                        coords: coords.map(Arc::new),
                    });
                }
                (detectors, seeds, meta)
            };

        let region_base = checked_region_base(out.regions.len(), meta.len())?;
        let segment_refs: Vec<&[Op]> = self.segments.iter().map(Vec::as_slice).collect();
        let mut slices = detector_slices_with_seeds(&segment_refs, &detectors, &seeds)?;

        // Every node appears, with an inner vec per moment (empty if no regions).
        for (node_ref, start, len) in &self.node_ranges {
            let mut node_slices = vec![Vec::new(); *len];
            for (moment, slots) in node_slices.iter_mut().enumerate() {
                for (detector_id, terms) in std::mem::take(&mut slices.slices[start + moment]) {
                    slots.push(RegionView {
                        region: region_base + detector_id,
                        terms,
                    });
                }
            }
            out.per_node
                .insert(node_ref.clone(), NodeSlices(node_slices));
        }

        for brk in slices.breaks {
            let meta = &meta[brk.detector as usize];
            let Some((node, moment)) = self.segment_owner.get(brk.segment).cloned() else {
                return Err(ProgramSliceError::BreakOutsideTimeline {
                    region: meta.id.clone(),
                });
            };
            out.breaks.push(ProgramRegionBreak {
                node,
                moment,
                region: meta.id.clone(),
                qubit: brk.qubit,
                kind: brk.kind,
            });
        }
        out.regions.extend(meta);
        Ok(())
    }
}

/// XOR boundary operators before instance lookup. This lets a selected branch's
/// delta cancel the base arm's symbolic faces even when that base instance lives
/// on the isolated, non-selected tape.
fn canonicalize_pauli_seeds(seeds: Vec<PendingPauliSeed>) -> Vec<PendingPauliSeed> {
    let mut groups: FxHashMap<(TemplateInstanceId, BoundaryFace), PauliMap> = FxHashMap::default();
    for seed in seeds {
        groups
            .entry((seed.instance, seed.face))
            .and_modify(|operator| *operator = &*operator ^ &seed.operator)
            .or_insert(seed.operator);
    }
    let mut seeds: Vec<PendingPauliSeed> = groups
        .into_iter()
        .filter_map(|((instance, face), operator)| {
            (!operator.is_empty()).then_some(PendingPauliSeed {
                instance,
                face,
                operator,
            })
        })
        .collect();
    seeds.sort_unstable_by_key(|seed| {
        (
            seed.instance.0,
            match seed.face {
                BoundaryFace::Input => 0,
                BoundaryFace::Output => 1,
            },
        )
    });
    seeds
}

/// Canonical content is cached before parents merge it, so shared producers
/// retain XOR multiplicity without expanding a DAG into a tree.
#[derive(Clone, Default)]
struct ObservableContent {
    parity: NodeDetectorParity,
    seeds: Vec<PendingPauliSeed>,
}

impl ObservableContent {
    fn xor_assign(&mut self, other: Self) {
        self.parity.xor_assign(&other.parity);
        self.seeds.extend(other.seeds);
        self.seeds = canonicalize_pauli_seeds(std::mem::take(&mut self.seeds));
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum ContentKind {
    Value,
    Boundary,
}

struct ObservableCollector {
    physical_observables: bool,
    cache: FxHashMap<(NodeRef, ContentKind), ObservableContent>,
}

impl ObservableCollector {
    /// Skip a recipe when its pinned dataflow proves its activation false.
    fn inactive(&self, level: &SubGraph, path: &LevelPath, node: BloqNodeId) -> bool {
        level[node].activation.is_some_and(|slot| {
            evaluate_expression(level, path, node, &ClassicalExpr::In(slot)) == Some(false)
        })
    }

    fn inputs(
        &mut self,
        level: &SubGraph,
        path: &LevelPath,
        node: BloqNodeId,
    ) -> Result<ObservableContent, ProgramSliceError> {
        let mut content = ObservableContent::default();
        if self.inactive(level, path, node) {
            return Ok(content);
        }
        if let Some(ClassicalNode::Observable {
            measurements,
            operators,
            ..
        }) = level[node].try_classical()
        {
            content.parity = NodeDetectorParity::from_measurements(measurements.iter().copied());
            content.seeds = canonicalize_pauli_seeds(
                operators
                    .iter()
                    .map(|operator| PendingPauliSeed {
                        instance: operator.instance,
                        face: operator.face,
                        operator: operator.operator.clone(),
                    })
                    .collect(),
            );
        }
        for input in level.data_inputs(node) {
            // Runtime folds change the known sign of this physical equation;
            // they are not extra measurement support for its decoder query.
            if self.physical_observables && *input.role != crate::ValueRole::Data {
                continue;
            }
            if input.output != Some(crate::ObservableOutput::Flip) {
                content.xor_assign(self.producer(
                    level,
                    path,
                    input.producer,
                    ContentKind::Value,
                )?);
            }
            if input.output.is_none() {
                content.xor_assign(self.producer(
                    level,
                    path,
                    input.producer,
                    ContentKind::Boundary,
                )?);
            }
        }
        Ok(content)
    }

    fn producer(
        &mut self,
        level: &SubGraph,
        path: &LevelPath,
        producer: BloqNodeId,
        kind: ContentKind,
    ) -> Result<ObservableContent, ProgramSliceError> {
        let node_ref = NodeRef {
            path: path.clone(),
            node: producer,
        };
        let key = (node_ref.clone(), kind);
        if let Some(content) = self.cache.get(&key) {
            return Ok(content.clone());
        }
        let mut content = ObservableContent::default();
        if self.inactive(level, path, producer) {
            self.cache.insert(key, content.clone());
            return Ok(content);
        }
        match (level[producer].try_classical(), &level[producer].kind) {
            (Some(ClassicalNode::Observable { .. }), _) => {
                content = self.inputs(level, path, producer)?;
                match kind {
                    ContentKind::Value => content.seeds.clear(),
                    ContentKind::Boundary => content.parity = NodeDetectorParity::default(),
                }
            }
            (Some(ClassicalNode::Compute { expr }), _) if kind == ContentKind::Value => {
                // WF-11 makes each linear input occur once. Nonlinear inputs
                // cannot be interpreted as a record parity (SEM-FOLD).
                if !expr.is_linear() {
                    return Err(ProgramSliceError::NonLinearExpr { node: node_ref });
                }
                content = self.inputs(level, path, producer)?;
            }
            (_, BloqNodeKind::Region(region)) => {
                let selected = BodySelector::Body;
                let body = region
                    .bodies()
                    .find_map(|(selector, body)| (selector == selected).then_some(body))
                    .expect("regions have one body");
                let child_path = path.child(producer, selected);
                match kind {
                    ContentKind::Boundary => {
                        for &output in body.boundary_outputs() {
                            content.xor_assign(self.producer(body, &child_path, output, kind)?);
                        }
                    }
                    ContentKind::Value => {
                        if let Some(output) = body.value_output()
                            && output.output != crate::ObservableOutput::Flip
                        {
                            content = self.producer(body, &child_path, output.node, kind)?;
                        }
                    }
                }
            }
            // Other nodes do not produce this channel.
            (_, BloqNodeKind::Classical(_) | BloqNodeKind::Quantum(_)) => {}
        }
        self.cache.insert(key, content.clone());
        Ok(content)
    }
}

fn pauli_region_terms(operator: &bloq_circuit::PauliMap) -> Vec<RegionTerm> {
    operator
        .iter()
        .filter_map(|(&qubit, &pauli)| {
            let pauli = match pauli {
                Pauli::X => PauliBasis::X,
                Pauli::Y => PauliBasis::Y,
                Pauli::Z => PauliBasis::Z,
                Pauli::I => return None,
            };
            Some(RegionTerm { qubit, pauli })
        })
        .collect()
}

/// The measurement ids an op writes (for id remapping). Only `Measure`/`MPP`
/// produce records; other ops write none.
fn op_measurement_ids(op: &Op) -> &[u32] {
    match op {
        Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => measurements,
        Op::Gate { .. }
        | Op::Tick
        | Op::Repeat { .. }
        | Op::ConditionalPauli(_)
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. } => &[],
    }
}

/// Clone `op`, remapping its measurement record ids through `map`.
/// Measurement-controlled feedforward uses the same tape-global id space;
/// node-scoped `Value` controls pass through unchanged.
fn rewrite_measurement_ids(op: &Op, map: &FxHashMap<u32, u32>) -> Op {
    let remap = |ids: &[u32]| -> Vec<u32> {
        ids.iter()
            .map(|id| {
                *map.get(id)
                    .expect("every op measurement id was assigned a global id")
            })
            .collect()
    };
    match op {
        Op::Measure {
            basis,
            qubits,
            measurements,
            flip_probability,
        } => Op::Measure {
            basis: *basis,
            qubits: qubits.clone(),
            measurements: remap(measurements),
            flip_probability: *flip_probability,
        },
        Op::MPP {
            products,
            measurements,
        } => Op::MPP {
            products: products.clone(),
            measurements: remap(measurements),
        },
        Op::ConditionalPauli(corrections) => Op::ConditionalPauli(
            corrections
                .iter()
                .copied()
                .map(|mut correction| {
                    correction.control = *map
                        .get(&correction.control)
                        .expect("every measurement control names a produced record");
                    correction
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn region_indices_reject_overflow_before_scattering() {
        let maximum = u32::MAX as usize;
        assert_eq!(super::checked_region_base(maximum, 0).unwrap(), u32::MAX);
        assert!(matches!(
            super::checked_region_base(maximum, 1),
            Err(super::ProgramSliceError::RegionIdOverflow)
        ));
        assert!(matches!(
            super::checked_region_base(usize::MAX, 1),
            Err(super::ProgramSliceError::RegionIdOverflow)
        ));
    }
    use bloq_circuit::{
        CircuitBody, ConditionalCorrection, CoordCircuit, DetectorTerm, GateType, LoopStateId,
    };
    use glam::ivec2;

    use super::*;
    use crate::test_fixture::instance_measurement;
    use crate::{
        BloqEdge, BloqNode, BloqTemplate, InstanceBoundaryOperator, RegionNode, TemplateDetector,
        TemplateDetectorScope, TemplateInstance, TemplateRepeatState,
    };

    /// A one-measurement `M Z` template at `q`.
    fn z_measure_template(q: IVec2) -> BloqTemplate {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [q]);
        BloqTemplate::new(circuit)
    }

    fn single_instance(id: u32, template: crate::TemplateId) -> Vec<TemplateInstance> {
        vec![TemplateInstance::new(
            TemplateInstanceId(id),
            template,
            ivec2(0, 0),
        )]
    }

    #[test]
    fn pinned_membership_slices_only_the_selected_observable_records() {
        let program = Bloq::from_text(
            "BLOQIR 1
 template t0 {
  circuit {
   Z (0,0)
   M (0,0):m0
  }
 }
 graph {
  n0 quantum {
   instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0
  n2 compute in0 from selector choice
  n3 compute !in0
  n4 quantum {
   instance i1 t0 @ (1,0)
   instance i2 t0 @ (2,0)
   guard 0 i1
   guard 1 i2
  }
  n5 observable fragment measurements i1:m0 when v0
  n6 observable fragment measurements i2:m0 when v0
  n7 observable 0
  n0 -> n1 order
  n1 -> n2 value 0
  n2 -> n3 value 0
  n2 -> n4 value 0
  n3 -> n4 value 1
  n2 -> n5 value 0
  n3 -> n6 value 0
  n4 -> n5 order
  n4 -> n6 order
  n5 -> n7 compose 0
  n6 -> n7 compose 1
 }",
        )
        .unwrap();
        assert!(matches!(
            program_detector_slices(&program),
            Err(ProgramSliceError::MembershipSelectionRequired)
        ));
        for choice in [false, true] {
            let pinned = program
                .pin_membership(&[("choice".into(), choice)].into())
                .unwrap();
            let slices = program_detector_slices(&pinned).unwrap();
            let timeline = &slices.per_node[&NodeRef::top_level(BloqNodeId(4))];
            assert_eq!(timeline.0.len(), 2);
            assert_eq!(timeline.0[0].len(), 1);
            assert_eq!(
                timeline.0[0][0].terms,
                vec![RegionTerm {
                    qubit: ivec2(if choice { 1 } else { 2 }, 0),
                    pauli: PauliBasis::Z
                }]
            );
            assert_eq!(
                slices.regions[timeline.0[0][0].region as usize].id,
                ProgramRegionId::Observable { index: 0 }
            );
        }
    }

    #[test]
    fn physical_observable_support_excludes_runtime_sign_folds() {
        for role in ["feedback 0", "readout"] {
            for expression in ["in0 & in1", "in0 ^ in1"] {
                let program = Bloq::from_text(&format!(
                    "BLOQIR 1
 template t0 {{
   circuit {{
     RX (0,0) (1,0) (2,0)
     TICK
     M (0,0):m0 (1,0):m1 (2,0):m2
   }}
 }}
 graph {{
   n0 quantum {{
     instance i0 t0 @ (0,0)
   }}
   n1 observable fragment measurements i0:m0
   n2 observable fragment measurements i0:m1
   n3 observable fragment measurements i0:m2
   n4 compute {expression}
   n5 observable 0
   n0 -> n1 order
   n0 -> n2 order
   n0 -> n3 order
   n2 -> n4 value 0
   n3 -> n4 value 1
   n1 -> n5 compose 0
   n4 -> n5 value 1 {role}
 }}"
                ))
                .unwrap();
                let mut options = ProgramSliceOptions::default();
                options.set_physical_observables(true);
                let slices = program_detector_slices_with_options(&program, &options).unwrap();
                let timeline = &slices.per_node[&NodeRef::top_level(BloqNodeId(0))];
                assert_eq!(timeline.0[0].len(), 1);
                assert_eq!(
                    timeline.0[0][0].terms,
                    vec![RegionTerm {
                        qubit: ivec2(0, 0),
                        pauli: PauliBasis::Z,
                    }]
                );
                if expression.contains('&') {
                    assert!(matches!(
                        program_detector_slices(&program),
                        Err(ProgramSliceError::NonLinearExpr { .. })
                    ));
                } else {
                    let full = program_detector_slices(&program).unwrap();
                    assert_eq!(
                        full.per_node[&NodeRef::top_level(BloqNodeId(0))].0[0][0]
                            .terms
                            .len(),
                        3
                    );
                }
            }
        }
    }

    #[test]
    fn nested_recipe_preserves_physical_roles_and_boundary_bindings() {
        let source = "BLOQIR 1
template t0 {
  circuit {
    RX (0,0) (1,0)
    TICK
    M (0,0):m0 (1,0):m1
  }
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0
  n2 observable fragment measurements i0:m1
  n3 observable 0
  n4 compute in0
  n5 observable fragment operators i0 output Z(0,0)
  n6 observable fragment
  n7 observable fragment
  n8 observable 1
  n9 compute in0
  n10 observable 2
  n11 compute in0
  n12 compute 0
  n13 observable fragment operators i0 output X(1,0) when v0
  n0 -> n1 order
  n0 -> n2 order
  n0 -> n5 order
  n0 -> n13 order
  n12 -> n13 value 0
  n2 -> n3 compose 0
  n3 -> n4 value 0
  n1 -> n6 compose 0
  n4 -> n6 value 1 readout
  n1 -> n6 value 2 feedback 7
  n5 -> n6 compose 3
  n13 -> n6 compose 4
  n6 -> n7 compose 0
  __OBS_EDGES__
  n8 -> n9 value 0
  n10 -> n11 value 0
}";
        let recipe = Bloq::from_text(
            &source.replace("__OBS_EDGES__", "n7 -> n8 compose 0\n  n7 -> n10 compose 0"),
        )
        .unwrap();
        let flat = Bloq::from_text(&source.replace(
            "__OBS_EDGES__",
            "n1 -> n8 compose 0\n  n4 -> n8 value 1 readout\n  n1 -> n8 value 2 feedback 7\n  n5 -> n8 compose 3\n  n13 -> n8 compose 4\n  n1 -> n10 compose 0\n  n4 -> n10 value 1 readout\n  n1 -> n10 value 2 feedback 7\n  n5 -> n10 compose 3\n  n13 -> n10 compose 4",
        ))
        .unwrap();
        recipe.validate().unwrap();
        flat.validate().unwrap();
        for node in [BloqNodeId(8), BloqNodeId(10)] {
            for physical_observables in [false, true] {
                let collect = |program: &Bloq| {
                    ObservableCollector {
                        physical_observables,
                        cache: FxHashMap::default(),
                    }
                    .inputs(program.top(), &LevelPath::default(), node)
                    .unwrap()
                };
                let shared = collect(&recipe);
                let direct = collect(&flat);
                assert_eq!(shared.parity, direct.parity);
                assert_eq!(shared.seeds.len(), direct.seeds.len());
                assert_eq!(shared.seeds.len(), 1);
                assert_eq!(shared.seeds[0].instance, direct.seeds[0].instance);
                assert_eq!(shared.seeds[0].face, direct.seeds[0].face);
                assert_eq!(shared.seeds[0].operator, direct.seeds[0].operator);
                if physical_observables {
                    assert_eq!(
                        shared.parity,
                        NodeDetectorParity::from_measurements([instance_measurement(0, 0)])
                    );
                }
            }
        }
        for physical_observables in [false, true] {
            let mut options = ProgramSliceOptions::default();
            options.set_physical_observables(physical_observables);
            let shared = program_detector_slices_with_options(&recipe, &options).unwrap();
            let direct = program_detector_slices_with_options(&flat, &options).unwrap();
            assert_eq!(shared.regions, direct.regions);
            assert_eq!(shared.per_node, direct.per_node);
            assert_eq!(shared.breaks, direct.breaks);
            assert_eq!(shared.skipped_cross_tape, direct.skipped_cross_tape);
        }
    }

    #[test]
    fn nonlinear_observable_record_folds_are_rejected() {
        for expression in ["in0 & 0", "in0 | 1", "in0 & in1", "select(in0, 0, 1)"] {
            let second = if expression.contains("in1") {
                "n1 -> n2 value 1"
            } else {
                ""
            };
            let mut program = Bloq::from_text(&format!(
                "BLOQIR 1
 template t0 {{
   circuit {{
     RX (0,0)
     TICK
     M (0,0):m0
   }}
 }}
 graph {{
   n0 quantum {{
     instance i0 t0 @ (0,0)
   }}
   n1 observable fragment measurements i0:m0
   n2 compute {expression}
   n3 observable 0
   n0 -> n1 order
   n1 -> n2 value 0
   {second}
   n2 -> n3 value 0
 }}
"
            ))
            .unwrap();
            program
                .node_mut(BloqNodeId(0))
                .unwrap()
                .expect_quantum_mut()
                .detectors
                .push(crate::NodeDetector {
                    parity: NodeDetectorParity::from_measurements([instance_measurement(0, 0)]),
                    coords: None,
                });
            program.validate().unwrap();
            assert!(
                matches!(program_detector_slices(&program), Err(ProgramSliceError::NonLinearExpr { node }) if node == NodeRef::top_level(BloqNodeId(2)))
            );
            let mut options = ProgramSliceOptions::default();
            options.set_physical_observables(true);
            assert!(matches!(
                program_detector_slices_with_options(&program, &options),
                Err(ProgramSliceError::NonLinearExpr { .. })
            ));
            options.set_detectors_only(true);
            let slices = program_detector_slices_with_options(&program, &options).unwrap();
            assert_eq!(
                slices
                    .regions
                    .iter()
                    .map(|region| &region.id)
                    .collect::<Vec<_>>(),
                [&ProgramRegionId::Detector {
                    owner: NodeRef::top_level(BloqNodeId(0)),
                    detector: 0
                }]
            );
        }
    }

    #[test]
    fn upstream_observable_bits_carry_records_without_boundary_operators() {
        let program = Bloq::from_text(
            "BLOQIR 1
 template t0 {
   circuit {
     RX (0,0)
     R (1,0)
     TICK
     M (1,0):m0
   }
 }
 graph {
   n0 quantum {
     instance i0 t0 @ (0,0)
   }
   n1 observable fragment operators i0 output X(0,0)
   n2 observable 0
   n3 compute in0
   n4 observable 1
   n5 observable fragment measurements i0:m0
   n0 -> n5 order
   n1 -> n2 compose 0
   n5 -> n2 compose 1
   n2 -> n3 value 0
   n3 -> n4 value 0
 }
",
        )
        .unwrap();
        let slices = program_detector_slices(&program).unwrap();
        let mut raw_record = false;
        let mut direct_boundary = false;
        for view in slices
            .per_node
            .values()
            .flat_map(|node| node.0.iter().flatten())
        {
            match slices.regions[view.region as usize].id {
                ProgramRegionId::Observable { index: 0 } => {
                    direct_boundary |= view.terms.iter().any(|term| term.qubit == ivec2(0, 0))
                }
                ProgramRegionId::Observable { index: 1 } => {
                    assert!(view.terms.iter().all(|term| term.qubit != ivec2(0, 0)));
                    raw_record |= view.terms.iter().any(|term| term.qubit == ivec2(1, 0));
                }
                _ => {}
            }
        }
        assert!(raw_record && direct_boundary);
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn complete_observable_owns_direct_records_and_boundary_but_consumers_read_only_records() {
        let mut program = Bloq::from_text(
            "BLOQIR 1
 template t0 {
   circuit {
     RX (0,0)
     R (1,0)
     TICK
     M (1,0):m0
   }
 }
 graph {
   n0 quantum {
     instance i0 t0 @ (0,0)
   }
   n1 observable 0
   n2 observable fragment
   n3 observable 1
   n4 observable 2
   n0 -> n1 order
   n1 -> n2 compose 0
   n1 -> n3 value 0
   n2 -> n4 value 0
 }
",
        )
        .unwrap();
        program.node_mut(BloqNodeId(1)).unwrap().kind = BloqNodeKind::Classical(
            ClassicalNode::Observable {
                index: Some(0),
                measurements: vec![instance_measurement(0, 0)],
                operators: vec![InstanceBoundaryOperator {
                    instance: TemplateInstanceId(0),
                    face: BoundaryFace::Output,
                    operator: PauliMap::from_unique_entries([(ivec2(0, 0), Pauli::X)]),
                }],
            }
            .into(),
        );
        program.validate().unwrap();
        let mut collector = ObservableCollector {
            physical_observables: false,
            cache: FxHashMap::default(),
        };
        let direct = collector
            .inputs(program.top(), &LevelPath::default(), BloqNodeId(1))
            .unwrap();
        assert_eq!(
            direct.parity,
            NodeDetectorParity::from_measurements([instance_measurement(0, 0)])
        );
        assert_eq!(direct.seeds.len(), 1);
        for node in [BloqNodeId(3), BloqNodeId(4)] {
            let consumer = collector
                .inputs(program.top(), &LevelPath::default(), node)
                .unwrap();
            assert_eq!(consumer.parity, direct.parity);
            assert!(consumer.seeds.is_empty());
        }
        let slices = program_detector_slices(&program).unwrap();
        let timeline = &slices.per_node[&NodeRef::top_level(BloqNodeId(0))];
        let mut seen = FxHashSet::default();
        for view in timeline.0.iter().flatten() {
            let ProgramRegionId::Observable { index } = slices.regions[view.region as usize].id
            else {
                continue;
            };
            seen.insert(index);
            if index != 0 {
                assert!(view.terms.iter().all(|term| term.qubit != ivec2(0, 0)));
            }
        }
        assert_eq!(seen, FxHashSet::from_iter([0, 1, 2]));
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn shared_observable_dags_cancel_records_and_positioned_operators() {
        let mut program = Bloq::from_text(
            "BLOQIR 1
 template t0 {
   circuit {
     RX (0,0)
     R (1,0)
     TICK
     M (1,0):m0
   }
 }
 graph {
   n0 quantum {
     instance i0 t0 @ (0,0)
   }
   n1 rus 0 {
     body {
       n0 observable fragment operators i0 output X(0,0)
       n1 observable fragment measurements i0:m0
     }
   }
   n2 observable 0
   n0 -> n1 order
   n1 -> n2 compose 0
 }
",
        )
        .unwrap();
        let expected = program_detector_slices(&program).unwrap();
        let mut previous = BloqNodeId(1);
        let mut predicate = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        for _ in 0..30 {
            for previous in [&mut previous, &mut predicate] {
                let next = program.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Xor(Box::new([
                        ClassicalExpr::In(0),
                        ClassicalExpr::In(1),
                    ])),
                }));
                program.add_edge(*previous, next, BloqEdge::value(0));
                program.add_edge(*previous, next, BloqEdge::value(1));
                *previous = next;
            }
        }
        let BloqNodeKind::Region(RegionNode::RepeatUntilSuccess {
            restart_condition: condition,
            ..
        }) = &mut program.top_mut().node_mut(BloqNodeId(1)).unwrap().kind
        else {
            unreachable!()
        };
        *condition = ClassicalExpr::Or(Box::new([
            ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
            ClassicalExpr::Const(false),
        ]));
        program.add_edge(predicate, BloqNodeId(1), BloqEdge::value(0));
        program.add_edge(previous, BloqNodeId(2), BloqEdge::value(1));
        let actual = program_detector_slices(&program).unwrap();
        assert_eq!(actual.regions, expected.regions);
        assert_eq!(actual.per_node, expected.per_node);
        assert_eq!(actual.breaks, expected.breaks);
    }

    #[test]
    fn post_flatten_validation_is_wf_equivalent_for_looped_detector() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [q])[0];
        let repeated = circuit.reserve_measurement_id(q);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![q],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .expect("entry body")
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
                coords: None,
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
        let mut program = Bloq::new();
        let template = program.add_template(template);
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        program.add_node(node);

        let before = program.validate();
        let mut flattened = program.clone();
        flattened.flatten().expect("representative loop flattens");

        assert_eq!(before, Ok(()));
        assert_eq!(flattened.validate(), before);
    }

    #[test]
    fn malformed_instance_reference_fails_resolution_instead_of_skipping() {
        let q = ivec2(0, 0);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(z_measure_template(q));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        node.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([instance_measurement(7, 0)]),
            coords: None,
        }];
        bloq.add_node(node);

        assert!(matches!(
            program_detector_slices(&bloq),
            Err(ProgramSliceError::UnknownMeasurement {
                measurement: InstanceMeasurement {
                    instance: TemplateInstanceId(7),
                    ..
                },
                ..
            })
        ));
    }

    #[test]
    fn unknown_observable_instance_fails_resolution() {
        let program = Bloq::from_text(
            "BLOQIR 1
template t0 {
 circuit {
  R (0,0)
 }
}
graph {
 n0 quantum {
  instance i0 t0 @ (0,0)
 }
 n1 observable fragment operators i7 output X(0,0)
 n2 observable 0
 n1 -> n2 compose 0
}",
        )
        .unwrap();
        assert!(matches!(
            program_detector_slices(&program),
            Err(ProgramSliceError::UnknownInstance {
                instance: TemplateInstanceId(7),
                ..
            })
        ));
    }

    #[test]
    fn slicing_does_not_audit_unrelated_logical_cut_metadata() {
        let mut program = Bloq::new();
        program.set_logical_inputs(vec![crate::LogicalInput {
            port: glam::IVec3::ZERO,
            instance: TemplateInstanceId(7),
            x: PauliMap::empty(),
            z: PauliMap::empty(),
        }]);
        assert!(matches!(
            program.validate(),
            Err(BloqValidationError::LogicalInputUnknownInstance { .. })
        ));
        let slices = program_detector_slices(&program).unwrap();
        assert!(slices.regions.is_empty());
        assert!(slices.per_node.is_empty());
    }

    #[test]
    fn cross_node_detector_region_spans_both_node_timelines() {
        let q = ivec2(0, 0);
        // Child keeps the qubit alive through a Pauli `Z` (a tracker no-op) so
        // the region survives into the child's first moment before its measure.
        let mut child_circuit = CoordCircuit::new();
        child_circuit.do_gate(GateType::Z, [q]).unwrap();
        child_circuit.measure(PauliBasis::Z, [q]);

        let mut bloq = Bloq::new();
        let parent_tpl = bloq.add_template(z_measure_template(q));
        let child_tpl = bloq.add_template(BloqTemplate::new(child_circuit));

        let mut parent = BloqNode::from_members(vec![]);
        parent.expect_quantum_mut().instances = single_instance(0, parent_tpl);
        let mut child = BloqNode::from_members(vec![]);
        child.expect_quantum_mut().instances = single_instance(1, child_tpl);
        child.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([
                instance_measurement(0, 0),
                instance_measurement(1, 0),
            ]),
            coords: Some(DetectorCoords::from_slice(&[5.0, 6.0])),
        }];

        let parent_id = bloq.add_node(parent);
        let child_id = bloq.add_node(child);
        bloq.add_edge(parent_id, child_id, BloqEdge::quantum(vec![]));

        let slices = program_detector_slices(&bloq).unwrap();
        assert_eq!(slices.skipped_cross_tape, 0);
        assert!(slices.breaks.is_empty());

        let parent_ref = NodeRef::top_level(parent_id);
        let child_ref = NodeRef::top_level(child_id);
        let parent_slices = &slices.per_node[&parent_ref];
        let child_slices = &slices.per_node[&child_ref];
        assert_eq!(parent_slices.0.len(), 1, "parent: one measurement moment");
        assert_eq!(child_slices.0.len(), 2, "child: rotation then measurement");

        let expected = vec![RegionTerm {
            qubit: q,
            pauli: PauliBasis::Z,
        }];
        // Region alive at the end of the parent's measurement and the child's
        // first (pre-measure) moment; gone after the child measures.
        assert_eq!(parent_slices.0[0].len(), 1);
        assert_eq!(child_slices.0[0].len(), 1);
        assert!(child_slices.0[1].is_empty());

        let region = &parent_slices.0[0][0];
        assert_eq!(
            slices.regions[region.region as usize].id,
            ProgramRegionId::Detector {
                owner: child_ref,
                detector: 0,
            },
            "child's side table owns it"
        );
        assert_eq!(
            slices.regions[region.region as usize]
                .coords
                .as_ref()
                .map(|coords| coords.as_slice()),
            Some(&[5.0, 6.0][..])
        );
        assert_eq!(region.terms, expected);
        assert_eq!(child_slices.0[0][0].terms, expected);
    }

    #[test]
    fn measurement_controlled_pauli_cancels_program_detector_dependency() {
        let parent_q = ivec2(9, 0);
        let q0 = ivec2(0, 0);
        let q1 = ivec2(1, 0);

        let mut child_circuit = CoordCircuit::new();
        child_circuit.do_gate(GateType::RX, [q0]).unwrap();
        child_circuit.do_gate(GateType::RZ, [q1]).unwrap();
        child_circuit.tick();
        let m0 = child_circuit.measure(PauliBasis::Z, [q0])[0];
        child_circuit
            .body_mut(child_circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: m0,
                target: q1,
            }]));
        let m1 = child_circuit.measure(PauliBasis::Z, [q1])[0];

        let mut bloq = Bloq::new();
        let parent_template = bloq.add_template(z_measure_template(parent_q));
        let child_template = bloq.add_template(BloqTemplate::new(child_circuit));
        let mut parent = BloqNode::from_members(vec![]);
        parent.expect_quantum_mut().instances = single_instance(0, parent_template);
        let mut child = BloqNode::from_members(vec![]);
        child.expect_quantum_mut().instances = single_instance(1, child_template);
        child.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([
                instance_measurement(1, m0),
                instance_measurement(1, m1),
            ]),
            coords: None,
        }];
        let parent = bloq.add_node(parent);
        let child = bloq.add_node(child);
        bloq.add_edge(parent, child, BloqEdge::quantum(vec![]));

        let slices = program_detector_slices(&bloq).unwrap();
        assert!(slices.breaks.is_empty());
        assert!(
            slices.per_node[&NodeRef::top_level(parent)]
                .0
                .iter()
                .all(Vec::is_empty),
            "the child-local control must not retain the parent's tape-global id"
        );
        let child_slices = &slices.per_node[&NodeRef::top_level(child)];
        assert_eq!(
            child_slices.0.len(),
            2,
            "feedforward adds no visible moment"
        );
        assert_eq!(
            child_slices.0[0],
            vec![RegionView {
                region: 0,
                terms: vec![RegionTerm {
                    qubit: q1,
                    pauli: PauliBasis::Z,
                }],
            }]
        );
        assert!(child_slices.0[1].is_empty());
    }

    #[test]
    fn same_tick_measurement_keeps_its_place_before_a_rotation() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [q]).unwrap();
        let measurement = circuit.measure(PauliBasis::Z, [q])[0];
        circuit.do_gate(GateType::H, [q]).unwrap();
        let ops = circuit.body(circuit.entry_body()).unwrap().ops();
        assert_eq!(
            moment_segments(ops)
                .into_iter()
                .flat_map(|segment| segment.ops)
                .collect::<Vec<_>>(),
            ops
        );
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(Vec::new());
        node.expect_quantum_mut().instances = single_instance(0, template);
        node.expect_quantum_mut()
            .detectors
            .push(crate::NodeDetector {
                parity: DetectorParity::from_measurements([instance_measurement(0, measurement)]),
                coords: None,
            });
        bloq.add_node(node);

        let slices = program_detector_slices(&bloq).unwrap();

        assert!(slices.breaks.is_empty(), "RZ; M is deterministic before H");
    }

    #[test]
    fn measurement_controlled_pauli_stays_before_a_following_reset() {
        let control = ivec2(0, 0);
        let target = ivec2(1, 0);
        let spectator = ivec2(2, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RX, [control]).unwrap();
        let m0 = circuit.measure(PauliBasis::Z, [control])[0];
        circuit.tick();
        circuit.measure(PauliBasis::Z, [spectator]);
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: m0,
                target,
            }]));
        circuit.do_gate(GateType::RZ, [target]).unwrap();
        circuit.tick();
        let final_measurement = circuit.measure(PauliBasis::Z, [target])[0];

        let ops = circuit
            .body(circuit.entry_body())
            .expect("entry body")
            .ops();
        let moments = moment_segments(ops);
        assert!(matches!(
            moments[2].ops.as_slice(),
            [Op::Measure { .. }, Op::ConditionalPauli(_)]
        ));
        assert!(matches!(
            moments[3].ops.as_slice(),
            [Op::Gate {
                gate: GateType::RZ,
                ..
            }]
        ));

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        node.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([instance_measurement(0, final_measurement)]),
            coords: None,
        }];
        let node = bloq.add_node(node);

        let slices = program_detector_slices(&bloq).unwrap();
        assert!(slices.breaks.is_empty());
        let timeline = &slices.per_node[&NodeRef::top_level(node)].0;
        assert_eq!(timeline.len(), 5, "feedforward adds no visible moment");
        assert!(timeline[..3].iter().all(Vec::is_empty));
        assert_eq!(
            timeline[3][0].terms,
            vec![RegionTerm {
                qubit: target,
                pauli: PauliBasis::Z,
            }],
            "the reset erases the earlier correction before it can depend on m0"
        );
    }

    #[test]
    fn same_tick_measurement_control_stays_before_a_following_reset() {
        let control = ivec2(0, 0);
        let target = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RX, [control]).unwrap();
        circuit.tick();
        let m0 = circuit.measure(PauliBasis::Z, [control])[0];
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: m0,
                target,
            }]));
        circuit.do_gate(GateType::RZ, [target]).unwrap();
        circuit.tick();
        let final_measurement = circuit.measure(PauliBasis::Z, [target])[0];

        let ops = circuit
            .body(circuit.entry_body())
            .expect("entry body")
            .ops();
        let moments = moment_segments(ops);
        assert!(matches!(
            moments[1].ops.as_slice(),
            [Op::Measure { .. }, Op::ConditionalPauli(_)]
        ));
        assert!(matches!(
            moments[2].ops.as_slice(),
            [Op::Gate {
                gate: GateType::RZ,
                ..
            }]
        ));

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        node.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([instance_measurement(0, final_measurement)]),
            coords: None,
        }];
        let node = bloq.add_node(node);

        let slices = program_detector_slices(&bloq).unwrap();
        assert!(slices.breaks.is_empty());
        let timeline = &slices.per_node[&NodeRef::top_level(node)].0;
        assert_eq!(timeline.len(), 4, "feedforward adds no visible moment");
        assert!(timeline[..2].iter().all(Vec::is_empty));
        assert_eq!(
            timeline[2][0].terms,
            vec![RegionTerm {
                qubit: target,
                pauli: PauliBasis::Z,
            }],
            "the reset erases the correction before it can depend on m0"
        );
    }

    #[test]
    fn record_backed_observable_tracks_as_logical_region() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::Z, [q]).unwrap();
        circuit.measure(PauliBasis::Z, [q]);

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        let quantum = bloq.add_node(node);
        let accumulate = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(0, 0)],
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(5)));
        bloq.add_edge(quantum, accumulate, BloqEdge::Order);
        bloq.add_edge(accumulate, observable, BloqEdge::compose(0));

        let slices = program_detector_slices(&bloq).unwrap();
        let timeline = &slices.per_node[&NodeRef::top_level(quantum)];

        assert_eq!(timeline.0.len(), 2);
        assert_eq!(timeline.0[0].len(), 1);
        assert_eq!(
            timeline.0[0][0],
            RegionView {
                region: 0,
                terms: vec![RegionTerm {
                    qubit: q,
                    pauli: PauliBasis::Z,
                }],
            }
        );
        assert!(timeline.0[1].is_empty());
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn boundary_observable_seeds_input_and_output_at_node_faces() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [q]).unwrap();

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        let quantum = bloq.add_node(node);
        let include = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            measurements: Vec::new(),
            operators: vec![
                InstanceBoundaryOperator {
                    instance: TemplateInstanceId(0),
                    face: BoundaryFace::Input,
                    operator: PauliMap::from_unique_entries([(q, Pauli::Z)]),
                },
                InstanceBoundaryOperator {
                    instance: TemplateInstanceId(0),
                    face: BoundaryFace::Output,
                    operator: PauliMap::from_unique_entries([(q, Pauli::X)]),
                },
            ],
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(8)));
        bloq.add_edge(include, observable, BloqEdge::compose(0));

        let slices = program_detector_slices(&bloq).unwrap();
        let timeline = &slices.per_node[&NodeRef::top_level(quantum)];

        assert_eq!(timeline.0.len(), 1);
        assert_eq!(timeline.0[0].len(), 1);
        assert_eq!(
            slices.regions[timeline.0[0][0].region as usize].id,
            ProgramRegionId::Observable { index: 8 }
        );
        assert_eq!(
            timeline.0[0][0].terms,
            vec![RegionTerm {
                qubit: q,
                pauli: PauliBasis::X,
            }]
        );
        assert!(
            slices.breaks.is_empty(),
            "input Z cancels output X propagated backward through H"
        );
    }

    #[test]
    fn boundary_seed_on_zero_moment_instance_returns_typed_error() {
        let q = ivec2(0, 0);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        bloq.add_node(node);
        let include = bloq.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            measurements: Vec::new(),
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: PauliMap::from_unique_entries([(q, Pauli::X)]),
            }],
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(4)));
        bloq.add_edge(include, observable, BloqEdge::compose(0));

        assert!(matches!(
            program_detector_slices(&bloq),
            Err(ProgramSliceError::BreakOutsideTimeline {
                region: ProgramRegionId::Observable { index: 4 }
            })
        ));
    }

    #[test]
    fn region_exported_observable_content_follows_selected_body() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::RZ, [q]).unwrap();
        circuit.tick();
        circuit.measure(PauliBasis::X, [q]);
        circuit.tick();
        circuit.do_gate(GateType::Z, [q]).unwrap();

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let mut body = SubGraph::new();
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, template);
        let quantum = body.add_node(node);
        let accumulate = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            operators: Vec::new(),
            measurements: vec![instance_measurement(0, 0)],
        }));
        body.add_edge(quantum, accumulate, BloqEdge::Order);
        let binding = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            measurements: Vec::new(),
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: PauliMap::from_unique_entries([(q, Pauli::X)]),
            }],
        }));
        let unrelated = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::And(Box::new([
                ClassicalExpr::Const(true),
                ClassicalExpr::Const(false),
            ])),
        }));
        // Reading this bit internally must not consume its exported boundary.
        // The boundary-only slice must not inspect the unused nonlinear bit.
        let consume = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        body.add_edge(unrelated, consume, BloqEdge::value(0));
        body.set_value_output(Some(accumulate.into()));
        body.set_boundary_outputs(vec![binding]);

        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(9)));
        bloq.add_edge(region, observable, BloqEdge::compose(0));

        let slices = program_detector_slices(&bloq).unwrap();
        assert_eq!(slices.skipped_cross_tape, 0);
        assert!(slices.breaks.is_empty());

        let body_ref = NodeRef {
            path: LevelPath::default().child(region, BodySelector::Body),
            node: quantum,
        };
        let timeline = &slices.per_node[&body_ref];
        assert_eq!(timeline.0.len(), 3);
        assert_eq!(
            timeline.0[2],
            vec![RegionView {
                region: 0,
                terms: vec![RegionTerm {
                    qubit: q,
                    pauli: PauliBasis::X,
                }],
            }]
        );
    }

    /// A looped template: flatten unrolls the repeat, and each iteration's
    /// round-to-round detector is a distinct region on the unrolled timeline.
    #[test]
    fn repeat_program_yields_distinct_per_iteration_regions() {
        let q = ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [q])[0];
        let repeated = circuit.reserve_measurement_id(q);
        // A tick before each repeated measurement separates the rounds into
        // their own moments once unrolled.
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Tick,
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![q],
                measurements: vec![repeated],
                flip_probability: 0.0,
            },
        ]));
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
                coords: Some(DetectorCoords::from_slice(&[1.0, 2.0])),
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

        let mut bloq = Bloq::new();
        let tpl = bloq.add_template(template);
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut().instances = single_instance(0, tpl);
        let node_id = bloq.add_node(node);

        let slices = program_detector_slices(&bloq).unwrap();
        assert_eq!(slices.skipped_cross_tape, 0);
        assert!(slices.breaks.is_empty());

        let node_slices = &slices.per_node[&NodeRef::top_level(node_id)];
        // seed + three rounds = four measurement moments.
        assert_eq!(node_slices.0.len(), 4);
        let expected = vec![RegionTerm {
            qubit: q,
            pauli: PauliBasis::Z,
        }];
        // Moments 0..3 each carry exactly one region, and the three iteration
        // detectors are distinct (indices 0, 1, 2 in the flattened table).
        for moment in 0..3 {
            let regions = &node_slices.0[moment];
            assert_eq!(regions.len(), 1, "moment {moment}");
            assert_eq!(
                slices.regions[regions[0].region as usize].id,
                ProgramRegionId::Detector {
                    owner: NodeRef::top_level(node_id),
                    detector: moment as u32,
                }
            );
            assert_eq!(regions[0].terms, expected);
            assert_eq!(
                slices.regions[regions[0].region as usize]
                    .coords
                    .as_ref()
                    .map(|coords| coords.as_slice()),
                Some(&[1.0, 2.0][..])
            );
        }
        assert!(
            node_slices.0[3].is_empty(),
            "regions closed after last round"
        );
    }

    /// The executed body of a region inlines into the main tape, so a seam
    /// detector spanning a top-level node and a region-body node resolves and
    /// its region spans both timelines (the point of the executed-path model).
    #[test]
    fn region_body_seam_detector_resolves_across_boundary() {
        let q = ivec2(0, 0);
        // Body node keeps the qubit alive through a Pauli `Z` (a tracker no-op)
        // so the seam region survives into the body's first moment.
        let mut body_circuit = CoordCircuit::new();
        body_circuit.do_gate(GateType::Z, [q]).unwrap();
        body_circuit.measure(PauliBasis::Z, [q]);

        let mut bloq = Bloq::new();
        let top_tpl = bloq.add_template(z_measure_template(q));
        let body_tpl = bloq.add_template(BloqTemplate::new(body_circuit));

        let mut top_node = BloqNode::from_members(vec![]);
        top_node.expect_quantum_mut().instances = single_instance(0, top_tpl);

        let mut body = SubGraph::new();
        let mut body_node = BloqNode::from_members(vec![]);
        body_node.expect_quantum_mut().instances = single_instance(1, body_tpl);
        // Seam detector: top-level instance 0 meets body instance 1.
        body_node.expect_quantum_mut().detectors = vec![crate::NodeDetector {
            parity: DetectorParity::from_measurements([
                instance_measurement(0, 0),
                instance_measurement(1, 0),
            ]),
            coords: Some(DetectorCoords::from_slice(&[5.0, 6.0])),
        }];
        let body_node_id = body.add_node(body_node);

        let region = BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        });
        let top_id = bloq.add_node(top_node);
        let region_id = bloq.add_node(region);
        // Order the top-level measurement before the region body.
        bloq.add_edge(top_id, region_id, BloqEdge::quantum(vec![]));

        let slices = program_detector_slices(&bloq).unwrap();
        assert_eq!(slices.skipped_cross_tape, 0, "seam detector now resolves");
        assert!(slices.breaks.is_empty());

        let top_ref = NodeRef::top_level(top_id);
        let body_ref = NodeRef {
            path: LevelPath::default().child(region_id, BodySelector::Body),
            node: body_node_id,
        };
        let top_slices = &slices.per_node[&top_ref];
        let body_slices = &slices.per_node[&body_ref];
        assert_eq!(top_slices.0.len(), 1, "top: one measurement moment");
        assert_eq!(body_slices.0.len(), 2, "body: rotation then measurement");

        let expected = vec![RegionTerm {
            qubit: q,
            pauli: PauliBasis::Z,
        }];
        // Region alive at the top's measurement and the body's pre-measure
        // moment, spanning the seam; gone after the body measures.
        assert_eq!(top_slices.0[0].len(), 1);
        assert_eq!(body_slices.0[0].len(), 1);
        assert!(body_slices.0[1].is_empty());

        let region_view = &top_slices.0[0][0];
        assert_eq!(
            slices.regions[region_view.region as usize].id,
            ProgramRegionId::Detector {
                owner: body_ref,
                detector: 0,
            },
            "body's side table owns it"
        );
        assert_eq!(
            slices.regions[region_view.region as usize]
                .coords
                .as_ref()
                .map(|coords| coords.as_slice()),
            Some(&[5.0, 6.0][..])
        );
        assert_eq!(region_view.terms, expected);
        assert_eq!(body_slices.0[0][0].terms, expected);
    }
}
