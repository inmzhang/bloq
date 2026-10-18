//! Lower Bloq IR into the backend-independent VM instruction set.

use std::collections::BTreeSet;
use std::sync::Arc;

use bloq_ir::circuit::{
    DetectorParity, DetectorTerm, GateType, NoiseModel, Op, PauliBasis, PauliMap,
};
use bloq_ir::lowering::{
    InstanceMeasurement, InstantiationOptions, NodeEmissionPlan, NodeTemplateInstanceMergeError,
    TemplateDetectorScope, TemplateInstanceId,
};
use bloq_ir::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, BodySelector, BoundaryFace, ClassicalExpr,
    ClassicalNode, CoordinateOverflowError, CycleDetected, FlattenError, LevelPath,
    MemoryRoundTarget, NodeDetectorParity, NodeProvenance, ObservableOutput, PipePadding,
    RegionNode, SubGraph, TemporalPipeRef, ValueRef,
};
use glam::{IVec2, IVec3};
use rustc_hash::FxHashMap;

use crate::instruction::*;

/// Default maximum number of reachable alternatives for one guarded quantum stage.
pub const DEFAULT_MAX_QUANTUM_VARIANTS: usize = 4_096;

type ScopedNode = (LevelPath, BloqNodeId);
#[derive(Clone)]
struct DynamicWait {
    siblings: Vec<ScopedNode>,
    hoist_after: Option<ScopedNode>,
    origin: TaskOrigin,
}
type DynamicWaits = FxHashMap<ScopedNode, DynamicWait>;

fn task_origin(function: TaskFunction, members: impl IntoIterator<Item = IVec3>) -> TaskOrigin {
    let members = members
        .into_iter()
        .map(|member| member.to_array())
        .collect::<BTreeSet<_>>();
    let sites = members
        .iter()
        .map(|member| [member[0], member[1]])
        .collect::<BTreeSet<_>>();
    TaskOrigin {
        function,
        sites: sites.into_iter().collect(),
        members: members.into_iter().collect(),
    }
}

fn pipe_origin(pipes: impl IntoIterator<Item = TemporalPipeRef>) -> TaskOrigin {
    task_origin(
        TaskFunction::Memory,
        pipes.into_iter().flat_map(|pipe| [pipe.src, pipe.dst]),
    )
}

fn merge_origins<'a>(
    function: TaskFunction,
    origins: impl IntoIterator<Item = &'a TaskOrigin>,
) -> TaskOrigin {
    task_origin(
        function,
        origins
            .into_iter()
            .flat_map(|origin| origin.members.iter().copied().map(IVec3::from_array)),
    )
}

#[derive(Clone)]
struct TerminalHold {
    owner: ScopedNode,
    wait: ScopedNode,
    bindings: Vec<ScopedNode>,
}

/// Bounded lowering and timing options.
#[derive(Debug, Clone, Copy)]
pub struct LoweringConfig<'a> {
    /// Uniform duration assigned to every nonempty quantum moment. This must
    /// exceed the runtime clock tolerance.
    pub gate_duration: f64,
    /// Streaming-decoder latency, in physical memory rounds.
    pub decoder_latency_rounds: u32,
    /// Optional post-merge circuit noise. The IR instantiator applies it once.
    pub noise: Option<&'a NoiseModel>,
    /// Absolute source-release epochs. Prepared Y sources are shifted so they
    /// finish with ordinary Clifford source work.
    pub source_timing: SourceTiming,
    /// Maximum reachable alternatives materialized for one guarded quantum stage.
    pub max_quantum_variants: usize,
}

/// Absolute release epochs for independently schedulable source classes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SourceTiming {
    /// Factory source release epoch.
    pub factory: f64,
    /// External logical-input release epoch.
    pub input: f64,
    /// Clifford source release epoch and Prepared-Y completion reference.
    pub clifford: f64,
}

impl Default for LoweringConfig<'_> {
    fn default() -> Self {
        Self {
            gate_duration: 1.0,
            decoder_latency_rounds: DEFAULT_DECODER_LATENCY_ROUNDS,
            noise: None,
            source_timing: SourceTiming::default(),
            max_quantum_variants: DEFAULT_MAX_QUANTUM_VARIANTS,
        }
    }
}

/// Failure to produce a finite, standalone VM program.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LowerError {
    /// An IR emission plan could not be materialized.
    #[error("Bloq IR materialization failed: {0}")]
    InvalidProgram(#[from] bloq_ir::BloqValidationError),
    /// Repeat flattening failed or exceeded its configured IR resource bound.
    #[error("Bloq IR flattening failed: {0}")]
    Flatten(#[from] FlattenError),
    /// Materializing an exact dynamic memory frontier failed.
    #[error("dynamic join memory insertion failed: {0}")]
    Edit(#[from] bloq_ir::EditError),
    /// A region boundary did not have the unique escape required for padding.
    #[error("dynamic join region structure is unsupported: {0}")]
    Structure(#[from] bloq_ir::StructureError),
    /// Output-cut or decode-closure analysis failed.
    #[error("execution closure analysis failed: {0}")]
    Analysis(#[from] crate::ExecError),
    /// Existing IR moment lanes could not be aligned exactly.
    #[error("moment alignment failed: {0}")]
    Alignment(#[from] bloq_ir::MomentAlignmentError),
    /// A node could not be instantiated or noised exactly.
    #[error("quantum stage instantiation failed: {0}")]
    Instantiation(#[from] NodeTemplateInstanceMergeError),
    /// A shared detector use has an invalid bundle or owner binding.
    #[error("{0}")]
    DetectorBundle(#[from] bloq_ir::DetectorBundleError),
    /// A graph level has no topological execution order.
    #[error("Bloq IR graph is cyclic: {0}")]
    Cycle(#[from] CycleDetected),
    /// A placed coordinate overflowed the IR coordinate lattice.
    #[error("{0}")]
    CoordinateOverflow(#[from] CoordinateOverflowError),
    /// Lowering configuration contains an invalid duration or a derived sum overflowed.
    #[error(
        "gate duration must be finite and exceed the runtime clock tolerance, and derived durations must stay finite"
    )]
    InvalidDuration,
    /// A configured or derived source release is negative or non-finite.
    #[error("source release times must be finite and non-negative")]
    InvalidSourceTiming,
    /// Dense VM identifiers exhausted `u32`.
    #[error("{0} identifier space exhausted")]
    IdOverflow(&'static str),
    /// A conditional stage would require too many materialized choices.
    #[error(
        "quantum node {node} has {guards} guard inputs: {variants} reachable choices exceed limit {limit}"
    )]
    QuantumVariantLimit {
        /// Level-local node id.
        node: u32,
        /// Number of guard inputs.
        guards: usize,
        /// Observed reachable choice count, stopping at the first excess.
        variants: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// A permitted choice count could not fit or allocate its choice table.
    #[error("cannot allocate {variants} quantum alternatives: {source}")]
    QuantumVariantAllocation {
        /// Required reachable choice count.
        variants: usize,
        /// Capacity or allocation failure from the choice table.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// A record was referenced outside its producing emission plan.
    #[error("measurement record {instance}:{measurement} has no VM identity")]
    MissingRecord {
        /// Template instance id.
        instance: u32,
        /// Template-local measurement id.
        measurement: u32,
    },
    /// A value edge expected a bit from a binding-only or sink task.
    #[error("node {node} value slot {slot} is not backed by a VM bit")]
    MissingBit {
        /// Consuming level-local node id.
        node: u32,
        /// Missing input slot.
        slot: u32,
    },
    /// A declared logical output has no lowered frame register.
    #[error("logical output {0:?} has no complete lowered frame")]
    MissingOutputFrame([i32; 3]),
    /// A declared logical input has no lowered owning task.
    #[error("logical input instance {0} has no lowered VM task")]
    MissingInputInstance(u32),
    /// A declared logical output has no physical owning node.
    #[error("logical output instance {0} has no physical owner")]
    MissingOutputInstance(u32),
    /// Dynamic execution cannot defer this reused output boundary to program end.
    #[error("dynamic VM output boundary is consumed before terminal reporting")]
    ReusedOutputBoundary,
    /// Dynamic retries are supported only for source-isolated RUS regions.
    #[error("RUS node {0} has an incoming quantum seam and is not source-isolated")]
    NonSourceRus(u32),
    /// A dynamic readiness edge lacks complete persisted one-round padding.
    #[error("RUS readiness seam {from}->{to} has no complete one-round memory template")]
    MissingMemoryTemplate {
        /// Source region node.
        from: u32,
        /// Dependent node.
        to: u32,
    },
    /// A source RUS output seam lost its physical decoder-hold cycle.
    #[error("RUS node {0} has an outgoing padded seam but no decoder hold cycle")]
    MissingDecoderHold(u32),
    /// A flattened emission plan unexpectedly retained a loop or loop-state term.
    #[error("flattened quantum stream retains unsupported {0}")]
    Unflattened(&'static str),
    /// A materialized memory frontier could not be initialized exactly.
    #[error("dynamic memory frontier detector has {0} padding records; expected one")]
    InvalidFrontier(usize),
}

/// Insert one authored memory round as a movable detector frontier on every
/// quantum input of a real join and every physical RUS output. The VM may
/// execute that node zero or many times, but the ordinary IR edit performs the
/// hard part once: source and consumer detector rows are recomposed against
/// the padding records.
fn materialize_dynamic_waits(bloq: &mut Bloq) -> Result<DynamicWaits, LowerError> {
    let mut targets = Vec::new();
    let mut seen = BTreeSet::new();
    for (path, level) in bloq.levels() {
        for to in level.node_ids() {
            let incoming = level.incoming(to).collect::<Vec<_>>();
            let sources = incoming
                .iter()
                .map(|edge| edge.source)
                .collect::<BTreeSet<_>>();
            for edge in incoming {
                let BloqEdge::Quantum(quantum) = edge.edge else {
                    continue;
                };
                let source_rus = matches!(
                    level.node(edge.source).and_then(BloqNode::try_region),
                    Some(RegionNode::RepeatUntilSuccess { .. })
                );
                if sources.len() < 2 && !source_rus {
                    continue;
                }
                if quantum.pipes.is_empty()
                    || quantum.pipes.iter().any(|pipe| pipe.padding.is_none())
                {
                    return Err(LowerError::MissingMemoryTemplate {
                        from: edge.source.0,
                        to: edge.target.0,
                    });
                }
                if seen.insert((path.clone(), edge.source, edge.target)) {
                    targets.push((
                        path.clone(),
                        edge.source,
                        edge.target,
                        sources.clone(),
                        pipe_origin(quantum.pipes.iter().map(|pipe| pipe.pipe)),
                    ));
                }
            }
        }
    }

    let mut waits = DynamicWaits::default();
    for (path, from, to, sources, origin) in targets {
        let siblings = sources
            .iter()
            .copied()
            .filter(|source| *source != from)
            .map(|node| (path.clone(), node))
            .collect();
        let source_region = bloq
            .level_at(&path)
            .and_then(|level| level.node(from))
            .and_then(BloqNode::try_region)
            .is_some();
        let (wait_path, wait) = if source_region {
            let body_path = path.child(from, BodySelector::Body);
            let escape = bloq
                .level_at(&body_path)
                .expect("region body path exists")
                .quantum_tail()?;
            let wait = bloq.insert_memory_rounds(
                MemoryRoundTarget::After {
                    path: body_path.clone(),
                    node: escape,
                },
                1,
            )?;
            (body_path, wait)
        } else {
            let wait = bloq.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: path.clone(),
                    from,
                    to,
                },
                1,
            )?;
            (path.clone(), wait)
        };
        waits.insert(
            (wait_path, wait),
            DynamicWait {
                siblings,
                hoist_after: source_region.then_some((path, from)),
                origin,
            },
        );
    }
    Ok(waits)
}

fn materialize_terminal_holds(
    bloq: &mut Bloq,
    waits: &mut DynamicWaits,
) -> Result<Vec<TerminalHold>, LowerError> {
    let top = LevelPath::default();
    let frames = bloq
        .output_frames()
        .into_iter()
        .map(|frame| (frame.port, [frame.x, frame.z]))
        .collect::<FxHashMap<_, _>>();
    let mut targets = Vec::new();
    for output in bloq.logical_outputs() {
        // Private ideal outputs never occupy a surface-code memory patch.
        if bloq
            .quantum_nodes()
            .flat_map(|(_, node)| &node.instances)
            .any(|instance| {
                instance.id == output.instance
                    && matches!(
                        instance.provenance,
                        bloq_ir::InstanceProvenance::IdealPortAdapter { .. }
                            | bloq_ir::InstanceProvenance::SpatialPortSubstitution {
                                role: bloq_ir::PortRole::Multiplex,
                                ..
                            }
                    )
            })
        {
            continue;
        }
        let owner = bloq.levels().find_map(|(path, level)| {
            level.quantum_nodes().find_map(|(node, quantum)| {
                quantum
                    .instances
                    .iter()
                    .any(|instance| instance.id == output.instance)
                    .then(|| (path.clone(), node))
            })
        });
        let Some((path, owner)) = owner else {
            return Err(LowerError::MissingOutputInstance(output.instance.0));
        };
        if targets.iter().any(|(existing_path, existing, _, _, _, _)| {
            *existing_path == path && *existing == owner
        }) {
            continue;
        }
        let level = bloq
            .level_at(&path)
            .expect("logical output owner path remains valid");
        let (from, edge) = level
            .incoming(owner)
            .find_map(|edge| match edge.edge {
                BloqEdge::Quantum(quantum) => Some((edge.source, quantum)),
                BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
            })
            .ok_or(LowerError::MissingMemoryTemplate {
                from: owner.0,
                to: owner.0,
            })?;
        if edge.pipes.is_empty() || edge.pipes.iter().any(|pipe| pipe.padding.is_none()) {
            return Err(LowerError::MissingMemoryTemplate {
                from: from.0,
                to: owner.0,
            });
        }
        let frame_nodes = frames
            .get(&output.port)
            .copied()
            .ok_or(LowerError::MissingOutputFrame(output.port.to_array()))?;
        let bindings = bloq
            .levels()
            .flat_map(|(binding_path, level)| {
                level
                    .nodes()
                    .filter_map(move |(node, data)| match data.try_classical() {
                        Some(classical)
                            if classical.operators().iter().any(|operator| {
                                operator.face == BoundaryFace::Output
                                    && operator.instance == output.instance
                            }) =>
                        {
                            Some((binding_path.clone(), node))
                        }
                        _ => None,
                    })
            })
            .collect::<Vec<_>>();
        targets.push((
            path,
            owner,
            from,
            frame_nodes,
            bindings,
            pipe_origin(edge.pipes.iter().map(|pipe| pipe.pipe)),
        ));
    }

    let mut holds = Vec::new();
    for (path, owner, from, frame_nodes, bindings, origin) in targets {
        let wait = bloq.insert_memory_rounds(
            MemoryRoundTarget::Edge {
                path: path.clone(),
                from,
                to: owner,
            },
            1,
        )?;
        let wait = (path.clone(), wait);
        waits.insert(
            wait.clone(),
            DynamicWait {
                siblings: frame_nodes
                    .into_iter()
                    .map(|frame| (top.clone(), frame))
                    .collect(),
                hoist_after: None,
                origin,
            },
        );
        holds.push(TerminalHold {
            owner: (path, owner),
            wait,
            bindings,
        });
    }
    Ok(holds)
}

/// Lower `bloq` to a self-contained VM program.
///
/// The input is cloned because repeat expansion is a lowering concern. Noise is
/// passed through [`InstantiationOptions::noisy`], after each selected stage is
/// merged, rather than reimplemented by the VM.
///
/// # Errors
///
/// Returns [`LowerError`] when required materialization exceeds a bound or an
/// IR construct cannot be represented without loss. Full Bloq well-formedness
/// validation is available separately through [`Bloq::validate`].
pub fn lower(bloq: &Bloq, config: &LoweringConfig<'_>) -> Result<Program, LowerError> {
    if !(config.gate_duration.is_finite() && config.gate_duration > crate::runtime::TIME_EPSILON) {
        return Err(LowerError::InvalidDuration);
    }
    if [
        config.source_timing.factory,
        config.source_timing.input,
        config.source_timing.clifford,
    ]
    .into_iter()
    .any(|release| !release.is_finite() || release < 0.0)
    {
        return Err(LowerError::InvalidSourceTiming);
    }
    let mut bloq = bloq.clone();
    let mut dynamic_waits = materialize_dynamic_waits(&mut bloq)?;
    let terminal_holds = materialize_terminal_holds(&mut bloq, &mut dynamic_waits)?;
    bloq.flatten()?;
    if !crate::closure::plan_early_output_reads(&bloq)?.is_empty()
        || crate::closure::logical_output_cuts(&bloq)?
            .iter()
            .any(|cuts| !cuts.is_empty())
    {
        return Err(LowerError::ReusedOutputBoundary);
    }
    crate::closure::guard_decode_observable_closure(&bloq)?;
    let options = config
        .noise
        .map_or_else(InstantiationOptions::default, InstantiationOptions::noisy);
    let plans = bloq.emission_plans(&options)?;
    let instance_templates = bloq
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
    let layout = bloq.sorted_layout_coords()?;
    let qubit_count = u32::try_from(layout.len()).map_err(|_| LowerError::IdOverflow("qubit"))?;
    let coords = layout
        .iter()
        .enumerate()
        .map(|(index, &coord)| (coord, index as QubitId))
        .collect();
    let mut cx = LowerCx {
        bloq: &bloq,
        config,
        options,
        plans: &plans,
        coords,
        tasks: Vec::new(),
        task_origins: Vec::new(),
        next_bit: 0,
        next_record: 0,
        next_resource: 0,
        records: FxHashMap::default(),
        readout_parities: FxHashMap::default(),
        boolean_functions: BooleanFunctions::default(),
        instance_templates,
        dynamic_waits,
        node_tasks: FxHashMap::default(),
        instance_tasks: FxHashMap::default(),
        observable_parts: FxHashMap::default(),
        terminal_holds,
        pending_waits: Vec::new(),
        hoisted_waits: FxHashMap::default(),
    };
    let top_round_duration = cx
        .bloq
        .pipe_padding()
        .map(|padding| cx.one_round_duration(*padding))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .reduce(f64::max)
        .unwrap_or(config.gate_duration);
    let entry = cx
        .lower_level(
            bloq.top(),
            &LevelPath::default(),
            None,
            top_round_duration,
            None,
        )?
        .stream;
    cx.rewire_terminal_holds()?;
    cx.finish_waits()?;
    cx.assign_source_releases()?;
    let inputs = cx.lower_inputs()?;
    let outputs = cx.lower_outputs()?;
    Ok(Program {
        qubit_count,
        bit_count: cx.next_bit,
        record_count: cx.next_record,
        tasks: cx.tasks,
        task_origins: cx.task_origins,
        entry,
        inputs,
        outputs,
        decoder_latency_rounds: config.decoder_latency_rounds,
    })
}

#[derive(Clone, Copy)]
struct LoweredNode {
    entry: TaskId,
    ready: TaskId,
    exit: TaskId,
    bit: Option<BitId>,
    flip: Option<BitId>,
    raw: Option<BitId>,
}

impl LoweredNode {
    fn output(&self, output: ObservableOutput) -> Option<BitId> {
        match output {
            ObservableOutput::Corrected => self.bit,
            ObservableOutput::Flip => self.flip,
        }
    }
}

struct LoweredLevel {
    stream: Stream,
    nodes: FxHashMap<BloqNodeId, LoweredNode>,
}

struct LowerCx<'a, 'c> {
    bloq: &'a Bloq,
    config: &'c LoweringConfig<'c>,
    options: InstantiationOptions<'c>,
    plans: &'a bloq_ir::ValidatedPlans,
    coords: FxHashMap<IVec2, QubitId>,
    tasks: Vec<Task>,
    task_origins: Vec<TaskOrigin>,
    next_bit: BitId,
    next_record: RecordId,
    next_resource: ResourceId,
    records: FxHashMap<InstanceMeasurement, RecordId>,
    readout_parities: FxHashMap<Vec<InstanceMeasurement>, RecordParity>,
    boolean_functions: BooleanFunctions,
    instance_templates: FxHashMap<TemplateInstanceId, bloq_ir::TemplateId>,
    dynamic_waits: DynamicWaits,
    node_tasks: FxHashMap<ScopedNode, LoweredNode>,
    instance_tasks: FxHashMap<TemplateInstanceId, TaskId>,
    observable_parts: FxHashMap<TaskId, Vec<TaskId>>,
    terminal_holds: Vec<TerminalHold>,
    pending_waits: Vec<(TaskId, Vec<ScopedNode>)>,
    hoisted_waits: FxHashMap<ScopedNode, (TaskId, Vec<ScopedNode>)>,
}

impl LowerCx<'_, '_> {
    fn quantum_origin(&self, node: &BloqNode, source: Option<SourceRole>) -> TaskOrigin {
        let quantum = node.expect_quantum();
        let boundary_instances = self
            .bloq
            .logical_inputs()
            .iter()
            .map(|input| input.instance)
            .chain(
                self.bloq
                    .logical_outputs()
                    .iter()
                    .map(|output| output.instance),
            )
            .collect::<BTreeSet<_>>();
        let ideal_boundary = !quantum.instances.is_empty()
            && quantum
                .instances
                .iter()
                .all(|instance| boundary_instances.contains(&instance.id));
        match &node.provenance {
            NodeProvenance::BlockComponent { members } => task_origin(
                if ideal_boundary {
                    TaskFunction::IdealBoundary
                } else if source == Some(SourceRole::PreparedY) {
                    TaskFunction::PreparedY
                } else if !quantum.guards.is_empty() {
                    TaskFunction::AdaptiveMeasurement
                } else {
                    TaskFunction::Cube
                },
                members.iter().map(|member| member.pos),
            ),
            NodeProvenance::TemporalPipe { pipe } => task_origin(
                if pipe.hadamard {
                    TaskFunction::TemporalHadamard
                } else {
                    TaskFunction::Synthetic
                },
                [pipe.src, pipe.dst],
            ),
            NodeProvenance::SpatialPortSubstitution { source, .. } => {
                task_origin(TaskFunction::IdealBoundary, [*source])
            }
            NodeProvenance::MemoryPadding { pipe, .. } => pipe_origin(std::iter::once(*pipe)),
            NodeProvenance::Generator { .. }
            | NodeProvenance::Action { .. }
            | NodeProvenance::BranchSelector { .. }
            | NodeProvenance::OutputFrame { .. }
            | NodeProvenance::None => TaskOrigin::default(),
        }
    }

    fn align_level(
        &mut self,
        level: &SubGraph,
        nodes: &FxHashMap<BloqNodeId, LoweredNode>,
    ) -> Result<(), LowerError> {
        let mut lanes = Vec::new();
        for (node, quantum) in level.quantum_nodes() {
            let Some(lowered) = nodes.get(&node) else {
                continue;
            };
            let Some(stream) = instruction_stream(&self.tasks[lowered.entry as usize].instruction)
            else {
                continue;
            };
            let moments = stream
                .moments
                .iter()
                .map(|moment| moment.kind.map(ir_moment_kind))
                .collect();
            let mut qubits = self
                .bloq
                .node_qubits(&level[node])?
                .into_iter()
                .collect::<Vec<_>>();
            qubits.sort_by_key(|qubit| (qubit.x, qubit.y));
            // Use the persisted timeline and graph dependencies from the real
            // node; the lowered stream only supplies its already split moments.
            let _ = quantum;
            lanes.push(bloq_ir::MomentLane {
                node,
                moments,
                qubits,
            });
        }
        if lanes.len() < 2 {
            return Ok(());
        }
        let aligned = bloq_ir::align_moment_lanes(level, lanes)?;
        let mut positions = FxHashMap::<BloqNodeId, FxHashMap<usize, (i64, usize)>>::default();
        for layer in aligned {
            for (slot, aligned) in layer.slots.into_iter().enumerate() {
                for entry in aligned.entries {
                    positions
                        .entry(entry.node)
                        .or_default()
                        .insert(entry.moment, (layer.layer, slot));
                }
            }
        }
        let lane_tasks = nodes
            .iter()
            .filter(|(node, _)| level[**node].try_quantum().is_some())
            .map(|(&node, lowered)| (node, lowered.entry))
            .collect::<FxHashMap<_, _>>();
        for (node, mut positions) in positions {
            let task = nodes[&node].entry as usize;
            let incoming_quantum = level
                .incoming(node)
                .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)));
            let causally_started = lane_tasks.iter().any(|(&predecessor, &predecessor_task)| {
                predecessor != node
                    && (level.has_path(predecessor, node)
                        || self.tasks[task].dependencies.contains(&predecessor_task))
            });
            if causally_started || !incoming_quantum {
                let mut first = FxHashMap::<i64, usize>::default();
                for &(layer, slot) in positions.values() {
                    first
                        .entry(layer)
                        .and_modify(|current| *current = (*current).min(slot))
                        .or_insert(slot);
                }
                for (layer, slot) in positions.values_mut() {
                    *slot -= first[layer];
                }
            }
            let ideal = level[node]
                .expect_quantum()
                .instances
                .iter()
                .filter(|instance| instance.provenance.is_ideal_boundary())
                .flat_map(|instance| {
                    self.bloq.templates()[instance.template_id]
                        .qubits()
                        .iter()
                        .filter_map(|&qubit| self.qubit(qubit, instance.offset).ok())
                })
                .collect::<BTreeSet<_>>();
            let idle_qubits = self.tasks[task]
                .qubits
                .iter()
                .copied()
                .filter(|qubit| !ideal.contains(qubit))
                .collect::<Box<[_]>>();
            rewrite_instruction_streams(
                &mut self.tasks[task].instruction,
                &positions,
                self.config.gate_duration,
                &idle_qubits,
                self.config.noise.map_or(0.0, |noise| noise.p_idle),
            )?;
            self.tasks[task].duration = instruction_duration(&self.tasks[task].instruction)?;
        }
        Ok(())
    }

    fn lower_outputs(&self) -> Result<Vec<LogicalOutput>, LowerError> {
        let top = LevelPath::default();
        let frames = self
            .bloq
            .output_frames()
            .into_iter()
            .map(|frame| (frame.port.to_array(), frame))
            .collect::<FxHashMap<_, _>>();
        self.bloq
            .logical_outputs()
            .iter()
            .map(|output| {
                let port = output.port.to_array();
                let frame = frames
                    .get(&port)
                    .ok_or(LowerError::MissingOutputFrame(port))?;
                let frame_x = self
                    .node_tasks
                    .get(&(top.clone(), frame.x))
                    .and_then(|node| node.bit)
                    .ok_or(LowerError::MissingOutputFrame(port))?;
                let frame_z = self
                    .node_tasks
                    .get(&(top.clone(), frame.z))
                    .and_then(|node| node.bit)
                    .ok_or(LowerError::MissingOutputFrame(port))?;
                Ok(LogicalOutput {
                    port,
                    x: self.pauli_product(&output.x, IVec2::ZERO)?,
                    z: self.pauli_product(&output.z, IVec2::ZERO)?,
                    frame_x,
                    frame_z,
                })
            })
            .collect()
    }

    fn finish_waits(&mut self) -> Result<(), LowerError> {
        for (task, siblings) in &self.pending_waits {
            let until = siblings
                .iter()
                .map(|sibling| {
                    self.node_tasks
                        .get(sibling)
                        .map(|node| node.ready)
                        .ok_or(LowerError::Unflattened("dynamic join sibling"))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice();
            let Instruction::WaitFor {
                until: task_until, ..
            } = &mut self.tasks[*task as usize].instruction
            else {
                unreachable!("dynamic wait task retains its instruction")
            };
            *task_until = until;
        }
        Ok(())
    }

    fn rewire_terminal_holds(&mut self) -> Result<(), LowerError> {
        for hold in &self.terminal_holds {
            let owner = self
                .node_tasks
                .get(&hold.owner)
                .ok_or(LowerError::Unflattened("terminal output owner"))?
                .exit;
            let wait = self
                .node_tasks
                .get(&hold.wait)
                .ok_or(LowerError::Unflattened("terminal output wait"))?
                .entry;
            let predecessors = self.tasks[wait as usize].dependencies.clone();
            for binding in &hold.bindings {
                let binding = self
                    .node_tasks
                    .get(binding)
                    .ok_or(LowerError::Unflattened("terminal output binding"))?
                    .entry;
                for task in std::iter::once(binding).chain(
                    self.observable_parts
                        .get(&binding)
                        .into_iter()
                        .flatten()
                        .copied(),
                ) {
                    let dependencies = self.tasks[task as usize]
                        .dependencies
                        .iter()
                        .copied()
                        .flat_map(|dependency| {
                            if dependency == owner {
                                predecessors.to_vec()
                            } else {
                                vec![dependency]
                            }
                        })
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect();
                    self.tasks[task as usize].dependencies = dependencies;
                }
            }
        }
        Ok(())
    }

    fn assign_source_releases(&mut self) -> Result<(), LowerError> {
        let reference_duration = self
            .tasks
            .iter()
            .filter(|task| task.source == Some(SourceRole::Clifford))
            .map(|task| task.duration)
            .fold(0.0, f64::max);
        for task in &mut self.tasks {
            task.release = match task.source {
                Some(SourceRole::Factory) => self.config.source_timing.factory,
                Some(SourceRole::LogicalInput) => self.config.source_timing.input,
                Some(SourceRole::Clifford) => self.config.source_timing.clifford,
                Some(SourceRole::PreparedY) => {
                    self.config.source_timing.clifford
                        + (reference_duration - task.duration).max(0.0)
                }
                None => 0.0,
            };
            if !task.release.is_finite() {
                return Err(LowerError::InvalidSourceTiming);
            }
        }
        Ok(())
    }

    fn lower_inputs(&self) -> Result<Vec<LogicalInput>, LowerError> {
        self.bloq
            .logical_inputs()
            .iter()
            .map(|input| {
                let (template_id, offset) = self
                    .bloq
                    .levels()
                    .flat_map(|(_, level)| level.quantum_nodes())
                    .flat_map(|(_, quantum)| quantum.instances.iter())
                    .find(|instance| instance.id == input.instance)
                    .map(|instance| (instance.template_id, instance.offset))
                    .ok_or(LowerError::MissingInputInstance(input.instance.0))?;
                let task = *self
                    .instance_tasks
                    .get(&input.instance)
                    .ok_or(LowerError::MissingInputInstance(input.instance.0))?;
                let stabilizers = self.bloq.templates()[template_id]
                    .boundary_flows
                    .iter()
                    .filter(|flow| flow.start.is_empty() && !flow.end.is_empty())
                    .map(|flow| {
                        let mut product = self.pauli_product(&flow.end, offset)?;
                        product.negative = flow.sign;
                        Ok(product)
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                let x = self.pauli_product(&input.x, IVec2::ZERO)?;
                let z = self.pauli_product(&input.z, IVec2::ZERO)?;
                let data_qubits = stabilizers
                    .iter()
                    .flat_map(|stabilizer| stabilizer.terms.iter().map(|(qubit, _)| *qubit))
                    .chain(
                        x.terms
                            .iter()
                            .chain(z.terms.iter())
                            .map(|(qubit, _)| *qubit),
                    )
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                Ok(LogicalInput {
                    port: input.port.to_array(),
                    task,
                    data_qubits,
                    stabilizers: stabilizers.into_boxed_slice(),
                    x,
                    z,
                })
            })
            .collect()
    }

    fn lower_level(
        &mut self,
        level: &SubGraph,
        path: &LevelPath,
        enclosing_rus: Option<ResourceId>,
        decoder_round_duration: f64,
        restart_source: Option<ValueRef>,
    ) -> Result<LoweredLevel, LowerError> {
        let implicit_demands = crate::closure::implicit_correction_demands(level, restart_source);
        let mut nodes = FxHashMap::<BloqNodeId, LoweredNode>::default();
        let mut stream = Vec::with_capacity(level.node_count());
        let mut last_qubit = FxHashMap::<QubitId, TaskId>::default();
        let mut discard_barrier = None;
        let mut hoisted_here = Vec::new();

        for id in level.deterministic_emit_order()? {
            let node = &level[id];
            let input_bits = self.input_bits(level, id, &nodes)?;
            let mut activation = node
                .activation
                .map(|slot| self.required_slot(id, slot, &input_bits))
                .transpose()?;
            let mut dependencies = level
                .incoming(id)
                .filter_map(|edge| {
                    nodes.get(&edge.source).map(|producer| match edge.edge {
                        BloqEdge::Compose { .. } => producer.entry,
                        _ => producer.exit,
                    })
                })
                .collect::<BTreeSet<_>>();
            if let Some(barrier) = discard_barrier {
                dependencies.insert(barrier);
            }
            if self.dynamic_waits.contains_key(&(path.clone(), id))
                && let Some(guard) = level.incoming(id).find_map(|edge| match edge.edge {
                    BloqEdge::Quantum(quantum) => quantum.guard,
                    BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
                })
            {
                let guard_output = guard.output;
                let guard = nodes[&guard.node];
                activation = Some(guard.output(guard_output).ok_or(LowerError::MissingBit {
                    node: id.0,
                    slot: guard.entry,
                })?);
                dependencies.insert(guard.exit);
            }

            let lowered = match &node.kind {
                BloqNodeKind::Quantum(_) => {
                    let wait = self.dynamic_waits.get(&(path.clone(), id)).cloned();
                    let lowered = self.lower_quantum_node(
                        level,
                        id,
                        path,
                        node,
                        dependencies,
                        activation,
                        &input_bits,
                        wait.is_some(),
                        &mut stream,
                        &mut last_qubit,
                    )?;
                    if let Some(wait) = wait {
                        self.set_task_origin(lowered.entry, wait.origin.clone());
                        if let Some(owner) = wait.hoist_after {
                            hoisted_here.push(lowered.entry);
                            self.hoisted_waits
                                .insert(owner, (lowered.entry, wait.siblings));
                        } else {
                            self.pending_waits.push((lowered.entry, wait.siblings));
                        }
                    }
                    lowered
                }
                BloqNodeKind::Classical(classical) => {
                    let classical = classical.as_ref();
                    if matches!(classical, ClassicalNode::Discard { .. }) {
                        dependencies.extend(stream.iter().copied());
                    }
                    let observable = matches!(classical, ClassicalNode::Observable { .. });
                    let decoded =
                        matches!(classical, ClassicalNode::Observable { index: Some(_), .. })
                            && crate::closure::needs_correction(level, id, &implicit_demands);
                    let output = produces_bit(classical)
                        .then(|| self.alloc_bit())
                        .transpose()?;
                    let flip = decoded.then(|| self.alloc_bit()).transpose()?;
                    let raw_output = if decoded {
                        Some(self.alloc_bit()?)
                    } else {
                        output
                    };
                    let mut local_tasks = Vec::new();
                    let instruction = match classical {
                        ClassicalNode::Compute { expr } => Instruction::Eval(
                            self.boolean_functions.lower(expr, &input_bits, id)?,
                        ),
                        ClassicalNode::Observable {
                            index,
                            measurements,
                            operators,
                        } => {
                            let mut bits = Vec::new();
                            let mut bindings = Vec::new();
                            for input in level.data_inputs(id) {
                                let producer = nodes[&input.producer];
                                if let Some(bit) = match input.output {
                                    None => producer.raw,
                                    Some(output) => producer.output(output),
                                } {
                                    bits.push(bit);
                                }
                                if input.output.is_none()
                                    && task_produces_bindings(
                                        &self.tasks[producer.entry as usize].instruction,
                                    )
                                {
                                    bindings.push(producer.entry);
                                }
                            }
                            if !measurements.is_empty() {
                                let bit = self.alloc_bit()?;
                                let parity = self.readout_parity(measurements)?;
                                let task = self.push_task(Task {
                                    label: task_label(path, id, "observable_measurements"),
                                    release: 0.0,
                                    source: None,
                                    dependencies: dependencies.iter().copied().collect(),
                                    activation,
                                    output: Some(bit),
                                    qubits: Box::new([]),
                                    duration: 0.0,
                                    instruction: Instruction::Accumulate(parity),
                                })?;
                                self.set_task_origin(
                                    task,
                                    task_origin(TaskFunction::Classical, std::iter::empty()),
                                );
                                stream.push(task);
                                local_tasks.push(task);
                                dependencies.insert(task);
                                bits.push(bit);
                            }
                            if !operators.is_empty() {
                                let boundaries = operators
                                    .iter()
                                    .map(|operator| {
                                        Ok(BoundaryBinding {
                                            input: operator.face == BoundaryFace::Input,
                                            operator: self
                                                .pauli_product(&operator.operator, IVec2::ZERO)?,
                                        })
                                    })
                                    .collect::<Result<Vec<_>, LowerError>>()?;
                                let task = self.push_task(Task {
                                    label: task_label(path, id, "observable_boundaries"),
                                    release: 0.0,
                                    source: None,
                                    dependencies: dependencies.iter().copied().collect(),
                                    activation,
                                    output: None,
                                    qubits: Box::new([]),
                                    duration: 0.0,
                                    instruction: Instruction::Bind(boundaries.into_boxed_slice()),
                                })?;
                                self.set_task_origin(
                                    task,
                                    task_origin(TaskFunction::Classical, std::iter::empty()),
                                );
                                stream.push(task);
                                local_tasks.push(task);
                                dependencies.insert(task);
                                bindings.push(task);
                            }
                            match index {
                                Some(index) => Instruction::Observable {
                                    index: *index,
                                    bits: bits.into_boxed_slice(),
                                    bindings: bindings.into_boxed_slice(),
                                },
                                None => Instruction::ReadoutRecipe {
                                    bits: bits.into_boxed_slice(),
                                    bindings: bindings.into_boxed_slice(),
                                },
                            }
                        }
                        ClassicalNode::Discard { condition } => Instruction::Discard(
                            self.boolean_functions.lower(condition, &input_bits, id)?,
                        ),
                    };
                    let task = self.push_task(Task {
                        label: task_label(path, id, "classical"),
                        release: 0.0,
                        source: None,
                        dependencies: dependencies.into_iter().collect(),
                        activation,
                        output: raw_output,
                        qubits: Box::new([]),
                        duration: 0.0,
                        instruction,
                    })?;
                    self.set_task_origin(
                        task,
                        task_origin(TaskFunction::Classical, std::iter::empty()),
                    );
                    stream.push(task);
                    let completed = if let ClassicalNode::Observable { index, .. } = classical {
                        let raw = raw_output.expect("observable allocates a raw bit");
                        self.observable_parts.insert(task, local_tasks);
                        if let (Some(index), Some(output), Some(flip)) = (index, output, flip) {
                            let request = DecodeRequest {
                                output: ObservableOutput::Corrected,
                                observable: task,
                                index: *index,
                                raw,
                                round_duration: decoder_round_duration,
                                timing: if enclosing_rus.is_some() {
                                    DecodeTiming::FactoryCompletion
                                } else {
                                    DecodeTiming::Measurement
                                },
                            };
                            let mut completed = task;
                            for (output, bit, role) in [
                                (ObservableOutput::Corrected, output, "decode"),
                                (ObservableOutput::Flip, flip, "flip"),
                            ] {
                                completed = self.push_task(Task {
                                    label: task_label(path, id, role),
                                    release: 0.0,
                                    source: None,
                                    dependencies: Box::new([task]),
                                    activation,
                                    output: Some(bit),
                                    qubits: Box::new([]),
                                    duration: 0.0,
                                    instruction: Instruction::Decode(DecodeRequest {
                                        output,
                                        ..request
                                    }),
                                })?;
                                self.set_task_origin(
                                    completed,
                                    task_origin(TaskFunction::Classical, std::iter::empty()),
                                );
                                stream.push(completed);
                            }
                            completed
                        } else {
                            task
                        }
                    } else {
                        task
                    };
                    if matches!(classical, ClassicalNode::Discard { .. }) {
                        discard_barrier = Some(task);
                    }
                    LoweredNode {
                        entry: task,
                        ready: completed,
                        exit: completed,
                        bit: output,
                        flip,
                        raw: if observable { raw_output } else { output },
                    }
                }
                BloqNodeKind::Region(region) => {
                    if level
                        .incoming(id)
                        .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
                    {
                        return Err(LowerError::NonSourceRus(id.0));
                    }
                    let before_bit = self.next_bit;
                    let before_record = self.next_record;
                    let RegionNode::RepeatUntilSuccess {
                        body: body_graph,
                        restart_condition: condition,
                        restart_source,
                    } = region;
                    let restart_source = *restart_source;
                    let resource = self.alloc_resource()?;
                    let child_path = path.child(id, BodySelector::Body);
                    let child_round_duration = self
                        .rus_round_duration(level, id)?
                        .unwrap_or(decoder_round_duration);
                    let lowered_body = self.lower_level(
                        body_graph,
                        &child_path,
                        Some(resource),
                        child_round_duration,
                        restart_source,
                    )?;
                    let cultivation_exits = body_graph
                        .quantum_nodes()
                        .filter_map(|(node, _)| {
                            let task = lowered_body.nodes.get(&node)?.exit;
                            matches!(
                                self.tasks[task as usize].instruction,
                                Instruction::Quantum(_)
                                    | Instruction::MemoryRounds { .. }
                                    | Instruction::Idle { .. }
                            )
                            .then_some(task)
                        })
                        .filter(|task| {
                            !body_graph.quantum_nodes().any(|(node, _)| {
                                let Some(candidate) = lowered_body.nodes.get(&node) else {
                                    return false;
                                };
                                matches!(
                                    self.tasks[candidate.entry as usize].instruction,
                                    Instruction::Quantum(_)
                                        | Instruction::MemoryRounds { .. }
                                        | Instruction::Idle { .. }
                                ) && candidate.entry != *task
                                    && self.tasks[candidate.entry as usize]
                                        .dependencies
                                        .contains(task)
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let physical_rus = !cultivation_exits.is_empty();
                    if physical_rus {
                        for &task in &lowered_body.stream.tasks {
                            if matches!(
                                self.tasks[task as usize].instruction,
                                Instruction::Quantum(_)
                                    | Instruction::MemoryRounds { .. }
                                    | Instruction::Idle { .. }
                            ) {
                                self.task_origins[task as usize].function =
                                    TaskFunction::FactoryBody;
                            }
                        }
                    }
                    let region_origin = merge_origins(
                        if physical_rus {
                            TaskFunction::Factory
                        } else {
                            TaskFunction::Control
                        },
                        lowered_body
                            .stream
                            .tasks
                            .iter()
                            .map(|&task| &self.task_origins[task as usize]),
                    );
                    let body = lowered_body.stream;
                    let output = Some(self.alloc_bit()?);
                    let decoder_hold = self
                        .hoisted_waits
                        .get(&(path.clone(), id))
                        .and_then(|(task, _)| match &self.tasks[*task as usize].instruction {
                            Instruction::WaitFor { memory, .. } => Some(memory.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    if physical_rus && decoder_hold.is_empty() {
                        return Err(LowerError::MissingDecoderHold(id.0));
                    }
                    let source_bit = restart_source
                        .map(|source| {
                            lowered_body
                                .nodes
                                .get(&source.node)
                                .and_then(|node| node.output(source.output))
                                .ok_or(LowerError::MissingBit {
                                    node: id.0,
                                    slot: source.node.0,
                                })
                        })
                        .transpose()?;
                    let mut restart_inputs = input_bits.clone();
                    let mut slots = Vec::new();
                    condition.for_each_input(&mut |slot| slots.push(slot));
                    if let Some(source_bit) = source_bit {
                        for slot in slots {
                            restart_inputs.entry(slot).or_insert(source_bit);
                        }
                    }
                    let owned_qubits: Box<[QubitId]> = body
                        .tasks
                        .iter()
                        .flat_map(|&task| self.tasks[task as usize].qubits.iter().copied())
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let qubits = owned_qubits.clone();
                    let instruction = Instruction::Rus {
                        body,
                        restart: self
                            .boolean_functions
                            .lower(condition, &restart_inputs, id)?,
                        owned_qubits,
                        attempt_bits: (before_bit..self.next_bit).collect(),
                        attempt_records: (before_record..self.next_record).collect(),
                        resource,
                        retry_prepare: QuantumStream::default(),
                        decoder_hold,
                        cultivation_exits,
                    };
                    // A supported RUS is a true source: it has no incoming
                    // seam (checked above) and no earlier task owns its support.
                    // Its first attempt therefore starts on fresh backend |0>
                    // qubits; later attempts replay the authored preparation.
                    if qubits.iter().any(|qubit| last_qubit.contains_key(qubit)) {
                        return Err(LowerError::NonSourceRus(id.0));
                    }
                    let task = self.push_task(Task {
                        label: task_label(path, id, region.kind_name()),
                        release: 0.0,
                        source: physical_rus.then_some(SourceRole::Factory),
                        dependencies: dependencies.into_iter().collect(),
                        activation,
                        output,
                        qubits: qubits.clone(),
                        duration: 0.0,
                        instruction,
                    })?;
                    self.set_task_origin(task, region_origin);
                    stream.push(task);
                    let mut exit = {
                        let signal = self.push_task(Task {
                            label: task_label(path, id, "ready"),
                            release: 0.0,
                            source: None,
                            dependencies: Box::new([task]),
                            activation: None,
                            output: None,
                            qubits: Box::new([]),
                            duration: 0.0,
                            instruction: Instruction::SignalReady { resource },
                        })?;
                        self.set_task_origin(
                            signal,
                            task_origin(TaskFunction::Control, std::iter::empty()),
                        );
                        stream.push(signal);
                        signal
                    };
                    let ready = exit;
                    if let Some((template, siblings)) =
                        self.hoisted_waits.remove(&(path.clone(), id))
                    {
                        let mut hoisted = self.tasks[template as usize].clone();
                        let origin = self.task_origins[template as usize].clone();
                        hoisted.label = task_label(path, id, "accepted-join-wait");
                        hoisted.dependencies = Box::new([exit]);
                        let hoisted = self.push_task(hoisted)?;
                        self.set_task_origin(hoisted, origin);
                        stream.push(hoisted);
                        self.pending_waits.push((hoisted, siblings));
                        exit = hoisted;
                    }
                    for &qubit in qubits.iter() {
                        last_qubit.insert(qubit, exit);
                    }
                    LoweredNode {
                        entry: task,
                        ready,
                        exit,
                        bit: output,
                        flip: None,
                        raw: output,
                    }
                }
            };
            nodes.insert(id, lowered);
            self.node_tasks.insert((path.clone(), id), lowered);
        }

        for removed in &hoisted_here {
            let replacements = self.tasks[*removed as usize].dependencies.clone();
            for &task in &stream {
                if task == *removed {
                    continue;
                }
                let dependencies = &mut self.tasks[task as usize].dependencies;
                if dependencies.contains(removed) {
                    let mut rewritten = dependencies
                        .iter()
                        .copied()
                        .filter(|dependency| dependency != removed)
                        .chain(replacements.iter().copied())
                        .collect::<BTreeSet<_>>();
                    *dependencies = std::mem::take(&mut rewritten).into_iter().collect();
                }
            }
        }
        stream.retain(|task| !hoisted_here.contains(task));
        self.align_level(level, &nodes)?;

        Ok(LoweredLevel {
            stream: Stream {
                tasks: stream.into_boxed_slice(),
                value: level
                    .value_output()
                    .and_then(|value| nodes[&value.node].output(value.output)),
                bindings: level
                    .boundary_outputs()
                    .iter()
                    .map(|node| nodes[node].entry)
                    .collect(),
            },
            nodes,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "one node lowering carries its graph context"
    )]
    fn lower_quantum_node(
        &mut self,
        level: &SubGraph,
        id: BloqNodeId,
        path: &LevelPath,
        node: &BloqNode,
        mut dependencies: BTreeSet<TaskId>,
        activation: Option<BitId>,
        input_bits: &FxHashMap<u32, BitId>,
        dynamic_wait: bool,
        stream: &mut Vec<TaskId>,
        last_qubit: &mut FxHashMap<QubitId, TaskId>,
    ) -> Result<LoweredNode, LowerError> {
        self.bloq
            .check_detector_bundle_bindings(node.expect_quantum(), |instance| {
                self.instance_templates.get(&instance).copied()
            })?;
        let mut alternatives = self.quantum_alternatives(level, path, id, node, input_bits)?;
        let top_level_source = *path == LevelPath::default()
            && !level
                .incoming(id)
                .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)));
        let input_instances = self
            .bloq
            .logical_inputs()
            .iter()
            .map(|input| input.instance)
            .collect::<BTreeSet<_>>();
        let output_instances = self
            .bloq
            .logical_outputs()
            .iter()
            .map(|output| output.instance)
            .collect::<BTreeSet<_>>();
        let quantum = node.expect_quantum();
        let owns_input = quantum
            .instances
            .iter()
            .any(|instance| input_instances.contains(&instance.id));
        let prepares_y = alternatives.iter().any(|alternative| {
            alternative.stream.moments.iter().any(|moment| {
                moment.operations.iter().any(|operation| {
                    matches!(
                        operation,
                        QuantumOp::Reset {
                            basis: Pauli::Y,
                            ..
                        }
                    )
                })
            })
        });
        let source = if owns_input {
            Some(SourceRole::LogicalInput)
        } else {
            top_level_source.then_some(if prepares_y {
                SourceRole::PreparedY
            } else {
                SourceRole::Clifford
            })
        };
        let mut origin = self.quantum_origin(node, source);
        if matches!(node.provenance, NodeProvenance::MemoryPadding { .. }) {
            let pipes = level
                .incoming(id)
                .chain(level.outgoing(id))
                .flat_map(|edge| edge.edge.pipes())
                .map(|pipe| pipe.pipe)
                .collect::<Vec<_>>();
            if !pipes.is_empty() {
                origin = pipe_origin(pipes);
            }
        }
        if !quantum.instances.is_empty()
            && quantum.instances.iter().all(|instance| {
                input_instances.contains(&instance.id) || output_instances.contains(&instance.id)
            })
        {
            for alternative in &mut alternatives {
                for moment in &mut alternative.stream.moments {
                    moment.duration = 0.0;
                }
            }
        }
        let qubits = alternatives
            .iter()
            .flat_map(|alternative| alternative.stream.moments.iter())
            .flat_map(|moment| moment.operations.iter())
            .flat_map(QuantumOp::qubits)
            .collect::<BTreeSet<_>>();
        for qubit in &qubits {
            if let Some(previous) = last_qubit.get(qubit) {
                dependencies.insert(*previous);
            }
        }
        let duration = alternatives
            .iter()
            .try_fold(0.0_f64, |longest, alternative| {
                stream_duration(&alternative.stream).map(|duration| longest.max(duration))
            })?;
        let rounds = match node.provenance {
            NodeProvenance::MemoryPadding { rounds, .. } => Some(rounds),
            _ => None,
        };
        let main_instruction = match (dynamic_wait, &node.provenance, alternatives.as_slice()) {
            (true, _, [alternative]) => Instruction::WaitFor {
                until: Box::new([]),
                memory: Box::new([self.lower_inserted_memory_cycle(node, alternative)?]),
            },
            (false, NodeProvenance::MemoryPadding { .. }, [alternative]) => {
                Instruction::MemoryRounds {
                    rounds: rounds.expect("matched padding provenance"),
                    stream: alternative.stream.clone(),
                    detectors: alternative.detectors.clone(),
                }
            }
            (false, _, _) => Instruction::Quantum(QuantumTask {
                alternatives: alternatives.into_boxed_slice(),
            }),
            (true, _, _) => return Err(LowerError::Unflattened("guarded dynamic memory")),
        };
        let main = self.push_task(Task {
            label: task_label(path, id, "quantum"),
            release: 0.0,
            source,
            dependencies: dependencies.into_iter().collect(),
            activation,
            output: None,
            qubits: qubits.iter().copied().collect(),
            duration: if dynamic_wait { 0.0 } else { duration },
            instruction: main_instruction,
        })?;
        self.set_task_origin(main, origin);
        for instance in &quantum.instances {
            self.instance_tasks.insert(instance.id, main);
        }
        stream.push(main);
        for qubit in &qubits {
            last_qubit.insert(*qubit, main);
        }
        Ok(LoweredNode {
            entry: main,
            ready: main,
            exit: main,
            bit: None,
            flip: None,
            raw: None,
        })
    }

    fn quantum_alternatives(
        &mut self,
        level: &SubGraph,
        path: &LevelPath,
        id: BloqNodeId,
        node: &BloqNode,
        inputs: &FxHashMap<u32, BitId>,
    ) -> Result<Vec<QuantumAlternative>, LowerError> {
        let quantum = node.expect_quantum();
        if quantum.guards.is_empty() {
            let plan = match self.plans.get(path, id) {
                Some(plan) => plan.clone(),
                None => node.emission_plan_with_options(self.bloq.templates(), self.options)?,
            };
            return Ok(vec![self.lower_alternative(node, &plan, Box::new([]))?]);
        }
        let slots = quantum
            .guards
            .iter()
            .map(|guard| guard.input)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let producers = level
            .value_inputs(id)
            .map(|input| (input.slot, input.producer))
            .collect::<FxHashMap<_, _>>();
        let predicates = slots
            .iter()
            .map(|slot| {
                producers.get(slot).copied().ok_or(
                    NodeTemplateInstanceMergeError::MissingMembershipInput(*slot),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut assignments = bloq_ir::lowering::PredicateAnalysis::new(level).values_up_to(
            &predicates,
            self.config.max_quantum_variants.saturating_add(1),
        )?;
        let variants = assignments.len();
        if variants > self.config.max_quantum_variants {
            return Err(LowerError::QuantumVariantLimit {
                node: id.0,
                guards: slots.len(),
                variants,
                limit: self.config.max_quantum_variants,
            });
        }
        // Retain the old truth-table order: the first slot varies fastest.
        assignments.sort_unstable_by(|left, right| left.iter().rev().cmp(right.iter().rev()));
        let mut out = Vec::new();
        out.try_reserve_exact(variants)
            .map_err(|source| LowerError::QuantumVariantAllocation { variants, source })?;
        for assignment in assignments {
            let values = slots
                .iter()
                .copied()
                .zip(assignment)
                .collect::<FxHashMap<_, _>>();
            let selected = node.select_quantum_members(|slot| values.get(&slot).copied())?;
            let plan = selected.emission_plan_with_options(self.bloq.templates(), self.options)?;
            let when = slots
                .iter()
                .map(|slot| Ok((self.required_slot(id, *slot, inputs)?, values[slot])))
                .collect::<Result<Vec<_>, LowerError>>()?
                .into_boxed_slice();
            out.push(self.lower_alternative(&selected, &plan, when)?);
        }
        let all_records = out
            .iter()
            .flat_map(|alternative| alternative.selected_records.iter().copied())
            .collect::<BTreeSet<_>>();
        for alternative in &mut out {
            let selected = alternative
                .selected_records
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            alternative.excluded_records = all_records.difference(&selected).copied().collect();
        }
        Ok(out)
    }

    fn lower_alternative(
        &mut self,
        node: &BloqNode,
        plan: &NodeEmissionPlan,
        when: Box<[(BitId, bool)]>,
    ) -> Result<QuantumAlternative, LowerError> {
        let grouped = plan.grouped_measurements();
        let mut local_records = FxHashMap::<u32, Box<[RecordId]>>::default();
        for (local, sources) in grouped {
            let records = sources
                .into_iter()
                .map(|source| self.record(source))
                .collect::<Result<Vec<_>, _>>()?;
            local_records.insert(local, records.into_boxed_slice());
        }
        let stream = self.lower_ops(
            plan.circuit
                .body(plan.circuit.entry_body())
                .expect("emission plan entry body")
                .ops(),
            IVec2::ZERO,
            &local_records,
        )?;
        let quantum = node.expect_quantum();
        let templates = plan.templates(self.bloq.templates());
        let mut detectors = Vec::new();
        let mut restarts = Vec::new();
        for instance in &quantum.instances {
            let template = &templates[instance.template_id];
            for detector in &template.detectors {
                if detector.scope != TemplateDetectorScope::TopLevel {
                    return Err(LowerError::Unflattened("repeat detector"));
                }
                detectors.push(self.template_parity(instance.id, &detector.parity)?);
            }
            restarts.extend(
                template
                    .restarts
                    .iter()
                    .map(|restart| self.template_parity(instance.id, &restart.parity))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        detectors.extend(
            self.bloq
                .node_detectors(quantum)?
                .map(|detector| self.node_parity(&detector.parity()))
                .collect::<Result<Vec<_>, _>>()?,
        );
        restarts.extend(
            quantum
                .restarts
                .iter()
                .map(|restart| self.node_parity(&restart.parity))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(QuantumAlternative {
            when,
            stream,
            selected_records: local_records
                .values()
                .flat_map(|records| records.iter().copied())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            excluded_records: Box::new([]),
            detectors: detectors.into_boxed_slice(),
            restarts: restarts.into_boxed_slice(),
        })
    }

    fn lower_ops(
        &self,
        ops: &[Op],
        offset: IVec2,
        records: &FxHashMap<u32, Box<[RecordId]>>,
    ) -> Result<QuantumStream, LowerError> {
        let mut moments = Vec::new();
        let mut current = Vec::new();
        let mut current_kind = None;
        let mut timed = false;
        let flush = |current: &mut Vec<QuantumOp>,
                     current_kind: &mut Option<MomentKind>,
                     timed: &mut bool,
                     moments: &mut Vec<Moment>| {
            if !current.is_empty() {
                moments.push(Moment {
                    kind: current_kind.take(),
                    duration: if *timed {
                        self.config.gate_duration
                    } else {
                        0.0
                    },
                    operations: std::mem::take(current).into_boxed_slice(),
                });
            } else if *timed {
                moments.push(Moment {
                    kind: current_kind.take(),
                    duration: self.config.gate_duration,
                    operations: Box::new([]),
                });
            }
            *current_kind = None;
            *timed = false;
        };
        for op in ops {
            if let Some(kind) = vm_moment_kind(op) {
                if current_kind.is_some_and(|current| current != kind) {
                    flush(&mut current, &mut current_kind, &mut timed, &mut moments);
                }
                current_kind = Some(kind);
            }
            match op {
                Op::Tick => flush(&mut current, &mut current_kind, &mut timed, &mut moments),
                Op::Repeat { .. } => return Err(LowerError::Unflattened("circuit repeat")),
                Op::Gate { gate, qubits } if gate.is_two_qubit_gate() => {
                    timed = true;
                    let (control_basis, target_basis) = gate
                        .two_qubit_bases()
                        .expect("two-qubit gate has Pauli bases");
                    for pair in qubits.as_chunks::<2>().0 {
                        current.push(QuantumOp::Gate2 {
                            control_basis: pauli(control_basis),
                            target_basis: pauli(target_basis),
                            control: self.qubit(pair[0], offset)?,
                            target: self.qubit(pair[1], offset)?,
                        });
                    }
                }
                Op::Gate { gate, qubits } => {
                    timed = true;
                    for &qubit in qubits {
                        let qubit = self.qubit(qubit, offset)?;
                        if let Some(op) = gate1(*gate, qubit) {
                            current.push(op);
                        }
                    }
                }
                Op::Measure {
                    basis,
                    qubits,
                    measurements,
                    flip_probability,
                } => {
                    timed = true;
                    for (&qubit, measurement) in qubits.iter().zip(measurements) {
                        current.push(QuantumOp::Measure {
                            observable: PauliProduct {
                                negative: false,
                                terms: Box::new([(self.qubit(qubit, offset)?, pauli(*basis))]),
                            },
                            records: records[measurement].clone(),
                            flip_probability: *flip_probability,
                        });
                    }
                }
                Op::MPP {
                    products,
                    measurements,
                } => {
                    timed = true;
                    for (product, measurement) in products.iter().zip(measurements) {
                        current.push(QuantumOp::Measure {
                            observable: self.pauli_product(product, offset)?,
                            records: records[measurement].clone(),
                            flip_probability: 0.0,
                        });
                    }
                }
                Op::ConditionalPauli(corrections) => {
                    timed |= !corrections.is_empty();
                    for correction in corrections {
                        current.push(QuantumOp::ConditionalPauli {
                            basis: pauli(correction.pauli),
                            qubit: self.qubit(correction.target, offset)?,
                            control: records[&correction.control][0],
                        });
                    }
                }
                Op::Depolarize1 {
                    probability,
                    qubits,
                } => current.push(QuantumOp::Depolarize1 {
                    probability: *probability,
                    qubits: qubits
                        .iter()
                        .map(|&qubit| self.qubit(qubit, offset))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                }),
                Op::Depolarize2 {
                    probability,
                    qubits,
                } => current.push(QuantumOp::Depolarize2 {
                    probability: *probability,
                    pairs: qubits
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| {
                            Ok((self.qubit(pair[0], offset)?, self.qubit(pair[1], offset)?))
                        })
                        .collect::<Result<Vec<_>, LowerError>>()?
                        .into_boxed_slice(),
                }),
                Op::PauliError {
                    probability,
                    pauli: basis,
                    qubits,
                } => current.push(QuantumOp::PauliError {
                    probability: *probability,
                    basis: pauli(*basis),
                    qubits: qubits
                        .iter()
                        .map(|&qubit| self.qubit(qubit, offset))
                        .collect::<Result<Vec<_>, _>>()?
                        .into_boxed_slice(),
                }),
            }
        }
        flush(&mut current, &mut current_kind, &mut timed, &mut moments);
        let mut compact: Vec<Moment> = Vec::with_capacity(moments.len());
        let mut leading_noise = Vec::new();
        for mut moment in moments {
            if moment.kind.is_none() && moment.duration == 0.0 {
                if let Some(previous) = compact.last_mut() {
                    previous.operations = std::mem::take(&mut previous.operations)
                        .into_vec()
                        .into_iter()
                        .chain(moment.operations.into_vec())
                        .collect();
                } else {
                    leading_noise.extend(moment.operations.into_vec());
                }
                continue;
            }
            if !leading_noise.is_empty() {
                moment.operations = std::mem::take(&mut leading_noise)
                    .into_iter()
                    .chain(moment.operations.into_vec())
                    .collect();
            }
            compact.push(moment);
        }
        if !leading_noise.is_empty() {
            compact.push(Moment {
                kind: None,
                duration: 0.0,
                operations: leading_noise.into_boxed_slice(),
            });
        }
        moments = compact;
        if self.config.noise.is_none()
            && !ops.iter().any(|op| {
                matches!(
                    op,
                    Op::Depolarize1 { .. } | Op::Depolarize2 { .. } | Op::PauliError { .. }
                )
            })
        {
            let authored = bloq_ir::aligned_moment_segments(ops)?
                .into_iter()
                .filter_map(|segment| segment.map(|segment| segment.kind))
                .collect::<Vec<_>>();
            let lowered = moments
                .iter()
                .filter_map(|moment| moment.kind.map(ir_moment_kind))
                .collect::<Vec<_>>();
            if authored != lowered {
                return Err(LowerError::Unflattened("moment segmentation drift"));
            }
        }
        Ok(QuantumStream {
            moments: moments.into_boxed_slice(),
        })
    }

    fn lower_inserted_memory_cycle(
        &self,
        node: &BloqNode,
        alternative: &QuantumAlternative,
    ) -> Result<MemoryCycle, LowerError> {
        let selected = alternative
            .selected_records
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut initializers = Vec::new();
        for detector in alternative.detectors.iter() {
            let owned = detector
                .records
                .iter()
                .copied()
                .filter(|record| selected.contains(record))
                .collect::<Vec<_>>();
            match owned.as_slice() {
                [] => {}
                [record] => initializers.push(FrontierInitializer {
                    record: *record,
                    parity: RecordParity {
                        records: detector
                            .records
                            .iter()
                            .copied()
                            .filter(|other| other != record)
                            .collect(),
                        constant: detector.constant,
                    },
                }),
                _ => return Err(LowerError::InvalidFrontier(owned.len())),
            }
        }

        let mut boundary_flows = Vec::new();
        for instance in &node.expect_quantum().instances {
            let template = &self.bloq.templates()[instance.template_id];
            for flow in &template.boundary_flows {
                let parity = self.instance_parity(
                    flow.measurements
                        .iter()
                        .map(|&measurement| InstanceMeasurement {
                            instance: instance.id,
                            measurement,
                        }),
                    flow.sign,
                )?;
                boundary_flows.push(BoundaryFlow {
                    start: self.pauli_product(&flow.start, instance.offset)?,
                    end: self.pauli_product(&flow.end, instance.offset)?,
                    frontier_record: (parity.records.len() == 1).then_some(parity.records[0]),
                    parity,
                });
            }
        }
        Ok(MemoryCycle {
            round_duration: stream_duration(&alternative.stream)?,
            stream: alternative.stream.clone(),
            detectors: alternative.detectors.clone(),
            boundary_flows: boundary_flows.into_boxed_slice(),
            initializers: initializers.into_boxed_slice(),
        })
    }

    fn rus_round_duration(
        &self,
        level: &SubGraph,
        region: BloqNodeId,
    ) -> Result<Option<f64>, LowerError> {
        let mut duration = None::<f64>;
        for padding in level
            .outgoing(region)
            .flat_map(|edge| edge.edge.pipes())
            .filter_map(|seam| seam.padding)
        {
            let total = self.one_round_duration(padding)?;
            duration = Some(duration.map_or(total, |old| old.max(total)));
        }
        Ok(duration)
    }

    fn one_round_duration(&self, padding: PipePadding) -> Result<f64, LowerError> {
        let template = &self.bloq.templates()[padding.one_round];
        let ops = template
            .circuit
            .body(template.circuit.entry_body())
            .expect("one-round memory template has an entry body")
            .ops();
        let mut moment_timed = false;
        let mut total = 0.0;
        for op in ops {
            match op {
                Op::Tick => {
                    if moment_timed {
                        total = add_duration(total, self.config.gate_duration)?;
                    }
                    moment_timed = false;
                }
                Op::Repeat { .. } => {
                    return Err(LowerError::Unflattened("one-round memory template"));
                }
                Op::Gate { .. } | Op::Measure { .. } | Op::MPP { .. } => moment_timed = true,
                Op::ConditionalPauli(corrections) => {
                    moment_timed |= !corrections.is_empty();
                }
                Op::Depolarize1 { .. } | Op::Depolarize2 { .. } | Op::PauliError { .. } => {}
            }
        }
        if moment_timed {
            total = add_duration(total, self.config.gate_duration)?;
        }
        Ok(total)
    }

    fn input_bits(
        &self,
        level: &SubGraph,
        id: BloqNodeId,
        nodes: &FxHashMap<BloqNodeId, LoweredNode>,
    ) -> Result<FxHashMap<u32, BitId>, LowerError> {
        level
            .value_inputs(id)
            .map(|input| {
                nodes
                    .get(&input.producer)
                    .and_then(|node| node.output(input.output.expect("value port")))
                    .map(|bit| (input.slot, bit))
                    .ok_or(LowerError::MissingBit {
                        node: id.0,
                        slot: input.slot,
                    })
            })
            .collect()
    }

    fn required_slot(
        &self,
        node: BloqNodeId,
        slot: u32,
        inputs: &FxHashMap<u32, BitId>,
    ) -> Result<BitId, LowerError> {
        inputs
            .get(&slot)
            .copied()
            .ok_or(LowerError::MissingBit { node: node.0, slot })
    }

    fn push_task(&mut self, task: Task) -> Result<TaskId, LowerError> {
        let id = u32::try_from(self.tasks.len()).map_err(|_| LowerError::IdOverflow("task"))?;
        self.tasks.push(task);
        self.task_origins.push(TaskOrigin::default());
        Ok(id)
    }

    fn set_task_origin(&mut self, task: TaskId, origin: TaskOrigin) {
        self.task_origins[task as usize] = origin;
    }

    fn alloc_bit(&mut self) -> Result<BitId, LowerError> {
        let id = self.next_bit;
        self.next_bit = id.checked_add(1).ok_or(LowerError::IdOverflow("bit"))?;
        Ok(id)
    }

    fn alloc_resource(&mut self) -> Result<ResourceId, LowerError> {
        let id = self.next_resource;
        self.next_resource = id
            .checked_add(1)
            .ok_or(LowerError::IdOverflow("resource"))?;
        Ok(id)
    }

    fn record(&mut self, source: InstanceMeasurement) -> Result<RecordId, LowerError> {
        if let Some(&record) = self.records.get(&source) {
            return Ok(record);
        }
        let record = self.next_record;
        self.next_record = record
            .checked_add(1)
            .ok_or(LowerError::IdOverflow("record"))?;
        self.records.insert(source, record);
        Ok(record)
    }

    fn lookup_record(&self, source: InstanceMeasurement) -> Result<RecordId, LowerError> {
        self.records
            .get(&source)
            .copied()
            .ok_or(LowerError::MissingRecord {
                instance: source.instance.0,
                measurement: source.measurement,
            })
    }

    fn instance_parity(
        &self,
        measurements: impl IntoIterator<Item = InstanceMeasurement>,
        constant: bool,
    ) -> Result<RecordParity, LowerError> {
        Ok(RecordParity {
            records: measurements
                .into_iter()
                .map(|measurement| self.lookup_record(measurement))
                .collect::<Result<Vec<_>, _>>()?
                .into(),
            constant,
        })
    }

    fn readout_parity(
        &mut self,
        records: &[InstanceMeasurement],
    ) -> Result<RecordParity, LowerError> {
        if let Some(parity) = self.readout_parities.get(records) {
            return Ok(parity.clone());
        }
        let parity = self.instance_parity(records.iter().copied(), false)?;
        self.readout_parities
            .insert(records.to_vec(), parity.clone());
        Ok(parity)
    }

    fn template_parity(
        &self,
        instance: TemplateInstanceId,
        parity: &DetectorParity,
    ) -> Result<RecordParity, LowerError> {
        let mut measurements = Vec::new();
        for term in parity.terms() {
            match *term {
                DetectorTerm::Measurement(measurement) => measurements.push(InstanceMeasurement {
                    instance,
                    measurement,
                }),
                DetectorTerm::LoopState(_) => {
                    return Err(LowerError::Unflattened("loop-carried detector state"));
                }
            }
        }
        self.instance_parity(measurements, parity.sign())
    }

    fn node_parity(&self, parity: &NodeDetectorParity) -> Result<RecordParity, LowerError> {
        let mut measurements = Vec::new();
        for term in parity.terms() {
            match *term {
                DetectorTerm::Measurement(measurement) => measurements.push(measurement),
                DetectorTerm::LoopState(_) => {
                    return Err(LowerError::Unflattened("loop-carried detector state"));
                }
            }
        }
        self.instance_parity(measurements, parity.sign())
    }

    fn qubit(&self, coordinate: IVec2, offset: IVec2) -> Result<QubitId, LowerError> {
        let coordinate = bloq_ir::circuit::checked_translate_coordinate(coordinate, offset)?;
        self.coords
            .get(&coordinate)
            .copied()
            .ok_or(CoordinateOverflowError { coordinate, offset }.into())
    }

    fn pauli_product(&self, product: &PauliMap, offset: IVec2) -> Result<PauliProduct, LowerError> {
        Ok(PauliProduct {
            negative: false,
            terms: product
                .iter()
                .filter(|(_, basis)| **basis != bloq_ir::circuit::Pauli::I)
                .map(|(coordinate, basis)| {
                    Ok((self.qubit(*coordinate, offset)?, pauli_map_basis(*basis)))
                })
                .collect::<Result<Vec<_>, LowerError>>()?
                .into_boxed_slice(),
        })
    }
}

fn produces_bit(node: &ClassicalNode) -> bool {
    !matches!(node, ClassicalNode::Discard { .. })
}

fn task_label(path: &LevelPath, node: BloqNodeId, role: &str) -> String {
    format!("{path:?}/n{}:{role}", node.0)
}

fn add_duration(total: f64, duration: f64) -> Result<f64, LowerError> {
    let total = total + duration;
    total
        .is_finite()
        .then_some(total)
        .ok_or(LowerError::InvalidDuration)
}

fn stream_duration(stream: &QuantumStream) -> Result<f64, LowerError> {
    stream
        .moments
        .iter()
        .try_fold(0.0, |total, moment| add_duration(total, moment.duration))
}

fn instruction_duration(instruction: &Instruction) -> Result<f64, LowerError> {
    match instruction {
        Instruction::Quantum(task) => task
            .alternatives
            .iter()
            .map(|alternative| stream_duration(&alternative.stream))
            .try_fold(0.0_f64, |longest, duration| {
                duration.map(|duration| longest.max(duration))
            }),
        Instruction::MemoryRounds { stream, .. } => stream_duration(stream),
        _ => Ok(0.0),
    }
}

fn instruction_stream(instruction: &Instruction) -> Option<&QuantumStream> {
    match instruction {
        Instruction::Quantum(task) => {
            let first = &task.alternatives.first()?.stream;
            if first
                .moments
                .iter()
                .any(|moment| moment.kind.is_none() && moment.duration == 0.0)
            {
                return None;
            }
            task.alternatives
                .iter()
                .all(|alternative| {
                    alternative
                        .stream
                        .moments
                        .iter()
                        .map(|moment| moment.kind)
                        .eq(first.moments.iter().map(|moment| moment.kind))
                })
                .then_some(first)
        }
        Instruction::MemoryRounds { stream, .. }
            if !stream
                .moments
                .iter()
                .any(|moment| moment.kind.is_none() && moment.duration == 0.0) =>
        {
            Some(stream)
        }
        // Dynamic waits align at runtime against the sibling completion event.
        Instruction::MemoryRounds { .. }
        | Instruction::WaitFor { .. }
        | Instruction::Eval(_)
        | Instruction::Accumulate(_)
        | Instruction::Bind(_)
        | Instruction::ReadoutRecipe { .. }
        | Instruction::Observable { .. }
        | Instruction::Decode(_)
        | Instruction::Discard(_)
        | Instruction::Rus { .. }
        | Instruction::SignalReady { .. }
        | Instruction::Idle { .. } => None,
    }
}

fn rewrite_instruction_streams(
    instruction: &mut Instruction,
    positions: &FxHashMap<usize, (i64, usize)>,
    duration: f64,
    idle_qubits: &[QubitId],
    idle_probability: f64,
) -> Result<(), LowerError> {
    let idle = || Moment {
        kind: None,
        duration,
        operations: (idle_probability != 0.0 && !idle_qubits.is_empty())
            .then(|| QuantumOp::Depolarize1 {
                probability: idle_probability,
                qubits: idle_qubits.into(),
            })
            .into_iter()
            .collect(),
    };
    let rewrite = |stream: &mut QuantumStream| {
        if stream.moments.len() != positions.len() {
            return Err(LowerError::Unflattened("guarded moment-lane shape"));
        }
        let old = std::mem::take(&mut stream.moments).into_vec();
        let mut moments = Vec::new();
        let mut previous = None;
        for (index, moment) in old.into_iter().enumerate() {
            let &(layer, slot) = positions
                .get(&index)
                .ok_or(LowerError::Unflattened("aligned moment index"))?;
            let start = match previous {
                Some((previous_layer, previous_slot)) if layer == previous_layer => {
                    previous_slot + 1
                }
                _ => 0,
            };
            for _ in start..slot {
                moments.push(idle());
            }
            moments.push(moment);
            previous = Some((layer, slot));
        }
        stream.moments = moments.into_boxed_slice();
        Ok(())
    };
    match instruction {
        Instruction::Quantum(task) => {
            for alternative in &mut task.alternatives {
                rewrite(&mut alternative.stream)?;
            }
        }
        Instruction::MemoryRounds { stream, .. } => rewrite(stream)?,
        _ => {}
    }
    Ok(())
}

fn ir_moment_kind(kind: MomentKind) -> bloq_ir::MomentKind {
    match kind {
        MomentKind::Reset => bloq_ir::MomentKind::Reset,
        MomentKind::Rotation => bloq_ir::MomentKind::Rotation,
        MomentKind::Interaction => bloq_ir::MomentKind::Interaction,
        MomentKind::Measurement => bloq_ir::MomentKind::Measurement,
    }
}

fn task_produces_bindings(instruction: &Instruction) -> bool {
    matches!(
        instruction,
        Instruction::Bind(_)
            | Instruction::ReadoutRecipe { .. }
            | Instruction::Observable { .. }
            | Instruction::Rus { .. }
    )
}

#[derive(Default)]
struct BooleanFunctions {
    definitions: FxHashMap<ClassicalExpr, (Arc<BoolOp>, Box<[u32]>)>,
}

impl BooleanFunctions {
    fn lower(
        &mut self,
        expr: &ClassicalExpr,
        inputs: &FxHashMap<u32, BitId>,
        node: BloqNodeId,
    ) -> Result<BoolOp, LowerError> {
        // Scalars have no body allocation to share and remain direct operations.
        if matches!(expr, ClassicalExpr::In(_) | ClassicalExpr::Const(_)) {
            return lower_expr(expr, inputs, node);
        }
        if !self.definitions.contains_key(expr) {
            let mut slots = BTreeSet::new();
            expr.for_each_input(&mut |slot| {
                slots.insert(slot);
            });
            let slots = slots.into_iter().collect::<Box<[_]>>();
            let local = slots
                .iter()
                .enumerate()
                .map(|(index, &slot)| {
                    Ok((
                        slot,
                        u32::try_from(index)
                            .map_err(|_| LowerError::IdOverflow("function input"))?,
                    ))
                })
                .collect::<Result<FxHashMap<_, _>, LowerError>>()?;
            let body = Arc::new(lower_expr(expr, &local, node)?);
            self.definitions.insert(expr.clone(), (body, slots));
        }
        let (body, slots) = &self.definitions[expr];
        let inputs = slots
            .iter()
            .map(|&slot| {
                inputs
                    .get(&slot)
                    .copied()
                    .ok_or(LowerError::MissingBit { node: node.0, slot })
            })
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(BoolOp::Call {
            body: Arc::clone(body),
            inputs,
        })
    }
}

fn lower_expr(
    expr: &ClassicalExpr,
    inputs: &FxHashMap<u32, BitId>,
    node: BloqNodeId,
) -> Result<BoolOp, LowerError> {
    let input = |slot| {
        inputs
            .get(&slot)
            .copied()
            .ok_or(LowerError::MissingBit { node: node.0, slot })
    };
    Ok(match expr {
        ClassicalExpr::In(slot) => BoolOp::Copy(input(*slot)?),
        ClassicalExpr::Const(value) => BoolOp::Const(*value),
        ClassicalExpr::Not(inner) => BoolOp::Not(Box::new(lower_expr(inner, inputs, node)?)),
        ClassicalExpr::Parity {
            inputs: slots,
            constant,
        } => BoolOp::Parity {
            inputs: slots
                .iter()
                .map(|&slot| input(slot))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
            constant: *constant,
        },
        ClassicalExpr::Xor(operands) => BoolOp::Xor(
            operands
                .iter()
                .map(|expr| lower_expr(expr, inputs, node))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        ),
        ClassicalExpr::And(operands) => BoolOp::And(
            operands
                .iter()
                .map(|expr| lower_expr(expr, inputs, node))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        ),
        ClassicalExpr::Or(operands) => BoolOp::Or(
            operands
                .iter()
                .map(|expr| lower_expr(expr, inputs, node))
                .collect::<Result<Vec<_>, _>>()?
                .into_boxed_slice(),
        ),
        ClassicalExpr::Select(operands) => {
            let [condition, when_false, when_true] = operands.as_ref();
            BoolOp::Select {
                condition: Box::new(lower_expr(condition, inputs, node)?),
                when_false: Box::new(lower_expr(when_false, inputs, node)?),
                when_true: Box::new(lower_expr(when_true, inputs, node)?),
            }
        }
    })
}

fn pauli(basis: PauliBasis) -> Pauli {
    match basis {
        PauliBasis::X => Pauli::X,
        PauliBasis::Y => Pauli::Y,
        PauliBasis::Z => Pauli::Z,
    }
}

fn pauli_map_basis(basis: bloq_ir::circuit::Pauli) -> Pauli {
    match basis {
        bloq_ir::circuit::Pauli::X => Pauli::X,
        bloq_ir::circuit::Pauli::Y => Pauli::Y,
        bloq_ir::circuit::Pauli::Z => Pauli::Z,
        bloq_ir::circuit::Pauli::I => unreachable!("identity was filtered"),
    }
}

fn vm_moment_kind(op: &Op) -> Option<MomentKind> {
    match op {
        Op::Gate { gate, .. } if gate.is_reset() => Some(MomentKind::Reset),
        Op::Gate { gate, .. } if gate.is_two_qubit_gate() => Some(MomentKind::Interaction),
        Op::Gate { .. } => Some(MomentKind::Rotation),
        Op::Measure { .. } | Op::MPP { .. } => Some(MomentKind::Measurement),
        Op::Tick
        | Op::Repeat { .. }
        | Op::ConditionalPauli(_)
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. } => None,
    }
}

fn gate1(gate: GateType, qubit: QubitId) -> Option<QuantumOp> {
    use Clifford1 as C;
    use GateType as G;
    Some(match gate {
        G::RX => QuantumOp::Reset {
            basis: Pauli::X,
            qubit,
        },
        G::RY => QuantumOp::Reset {
            basis: Pauli::Y,
            qubit,
        },
        G::RZ => QuantumOp::Reset {
            basis: Pauli::Z,
            qubit,
        },
        G::I => return None,
        G::X => QuantumOp::Pauli {
            basis: Pauli::X,
            qubit,
        },
        G::Y => QuantumOp::Pauli {
            basis: Pauli::Y,
            qubit,
        },
        G::Z => QuantumOp::Pauli {
            basis: Pauli::Z,
            qubit,
        },
        G::T_YZ | G::T_YZ_DAG => QuantumOp::T {
            basis: Pauli::X,
            qubit,
            adjoint: gate == G::T_YZ_DAG,
        },
        G::T_XZ | G::T_XZ_DAG => QuantumOp::T {
            basis: Pauli::Y,
            qubit,
            adjoint: gate == G::T_XZ_DAG,
        },
        G::T | G::T_DAG => QuantumOp::T {
            basis: Pauli::Z,
            qubit,
            adjoint: gate == G::T_DAG,
        },
        G::H => QuantumOp::Gate1 { gate: C::H, qubit },
        G::H_XY => QuantumOp::Gate1 {
            gate: C::H_XY,
            qubit,
        },
        G::H_YZ => QuantumOp::Gate1 {
            gate: C::H_YZ,
            qubit,
        },
        G::H_NXY => QuantumOp::Gate1 {
            gate: C::H_NXY,
            qubit,
        },
        G::H_NXZ => QuantumOp::Gate1 {
            gate: C::H_NXZ,
            qubit,
        },
        G::H_NYZ => QuantumOp::Gate1 {
            gate: C::H_NYZ,
            qubit,
        },
        G::SQRT_X => QuantumOp::Gate1 {
            gate: C::SQRT_X,
            qubit,
        },
        G::SQRT_X_DAG => QuantumOp::Gate1 {
            gate: C::SQRT_X_DAG,
            qubit,
        },
        G::SQRT_Y => QuantumOp::Gate1 {
            gate: C::SQRT_Y,
            qubit,
        },
        G::SQRT_Y_DAG => QuantumOp::Gate1 {
            gate: C::SQRT_Y_DAG,
            qubit,
        },
        G::S => QuantumOp::Gate1 { gate: C::S, qubit },
        G::S_DAG => QuantumOp::Gate1 {
            gate: C::S_DAG,
            qubit,
        },
        G::C_XYZ => QuantumOp::Gate1 {
            gate: C::C_XYZ,
            qubit,
        },
        G::C_ZYX => QuantumOp::Gate1 {
            gate: C::C_ZYX,
            qubit,
        },
        G::C_NXYZ => QuantumOp::Gate1 {
            gate: C::C_NXYZ,
            qubit,
        },
        G::C_XNYZ => QuantumOp::Gate1 {
            gate: C::C_XNYZ,
            qubit,
        },
        G::C_XYNZ => QuantumOp::Gate1 {
            gate: C::C_XYNZ,
            qubit,
        },
        G::C_NZYX => QuantumOp::Gate1 {
            gate: C::C_NZYX,
            qubit,
        },
        G::C_ZNYX => QuantumOp::Gate1 {
            gate: C::C_ZNYX,
            qubit,
        },
        G::C_ZYNX => QuantumOp::Gate1 {
            gate: C::C_ZYNX,
            qubit,
        },
        G::XCX | G::XCY | G::XCZ | G::YCX | G::YCY | G::YCZ | G::CX | G::CY | G::CZ => {
            unreachable!("two-qubit gates are handled first")
        }
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn shared_boolean_functions_keep_invocation_bindings_and_sparse_slots() {
        let expr = ClassicalExpr::Select(Box::new([
            ClassicalExpr::In(9),
            ClassicalExpr::In(2),
            ClassicalExpr::parity([2, 2, 7], true),
        ]));
        let mut functions = BooleanFunctions::default();
        let first = functions
            .lower(
                &expr,
                &[(2, 20), (7, 70), (9, 90)].into_iter().collect(),
                BloqNodeId(0),
            )
            .unwrap();
        let second = functions
            .lower(
                &expr,
                &[(2, 21), (7, 71), (9, 91)].into_iter().collect(),
                BloqNodeId(1),
            )
            .unwrap();
        let (
            BoolOp::Call {
                body: a,
                inputs: first,
            },
            BoolOp::Call {
                body: b,
                inputs: second,
            },
        ) = (first, second)
        else {
            panic!("compound expressions lower to shared functions");
        };
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(first.as_ref(), &[20, 70, 90]);
        assert_eq!(second.as_ref(), &[21, 71, 91]);
        assert!(matches!(
            functions.lower(
                &expr,
                &[(2, 20), (7, 70)].into_iter().collect(),
                BloqNodeId(3)
            ),
            Err(LowerError::MissingBit { node: 3, slot: 9 })
        ));
    }
    use bloq_compile::{CompileConfig, CompileContext};
    use bloq_graph::GalleryItem;
    use bloq_ir::circuit::CoordCircuit;
    use bloq_ir::lowering::BloqTemplate;

    #[test]
    fn unrelated_decoder_query_does_not_retry_without_an_authored_predicate() {
        let mut body = SubGraph::new();
        let observable = body.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        let result = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        body.add_edge(observable, result, BloqEdge::flip(0));
        body.set_value_output(Some(result.into()));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::Const(false),
            restart_source: None,
        }));
        bloq.validate().unwrap();
        let program = lower(&bloq, &LoweringConfig::default()).unwrap();
        for rejected_accuracy in [1.0, 0.0] {
            let result = crate::runtime::run(
                &program,
                crate::runtime::RuntimeConfig {
                    decoder: crate::decoder::MockDecoderConfig {
                        acceptance_script: vec![false],
                        accepted_accuracy: 1.0,
                        rejected_accuracy,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.artifact.retries.len(), 1);
            assert!(result.artifact.retries[0].accepted);
            assert_eq!(result.artifact.decoder_decisions.len(), 1);
            assert!(!result.artifact.decoder_decisions[0].decision.accepted);
            assert_eq!(
                result.artifact.decoder_decisions[0].decision.flip,
                rejected_accuracy == 0.0
            );
        }
    }

    #[test]
    fn rus_retries_only_for_the_authored_flip_predicate() {
        let mut body = SubGraph::new();
        let observables: Vec<_> = (0..3)
            .map(|index| body.add_node(BloqNode::classical(ClassicalNode::observable(index))))
            .collect();
        let predicate = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Or(Box::new([ClassicalExpr::In(0), ClassicalExpr::In(1)])),
        }));
        for (slot, &observable) in observables[..2].iter().enumerate() {
            body.add_edge(observable, predicate, BloqEdge::flip(slot as u32));
        }
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(predicate.into()),
        }));
        bloq.validate().unwrap();
        let program = lower(&bloq, &LoweringConfig::default()).unwrap();
        assert!(!program.instructions().any(|instruction| matches!(
            instruction,
            Instruction::Decode(DecodeRequest { index: 2, .. })
        )));
        for flips in [
            [false, false, false],
            [true, false, false],
            [false, true, false],
            [true, true, false],
            [false, false, true],
        ] {
            let result = crate::runtime::run(
                &program,
                crate::runtime::RuntimeConfig {
                    decoder: crate::decoder::MockDecoderConfig {
                        acceptance_script: flips
                            .into_iter()
                            .map(|flip| !flip)
                            .chain([true; 3])
                            .collect(),
                        accepted_accuracy: 1.0,
                        rejected_accuracy: 0.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            let retry = flips[0] || flips[1];
            assert_eq!(
                result
                    .artifact
                    .retries
                    .iter()
                    .map(|attempt| attempt.accepted)
                    .collect::<Vec<_>>(),
                if retry { vec![false, true] } else { vec![true] }
            );
            assert_eq!(
                result.artifact.decoder_decisions[..2]
                    .iter()
                    .map(|record| record.decision.flip)
                    .collect::<Vec<_>>(),
                flips[..2]
            );
            assert_eq!(
                result.artifact.decoder_decisions.len(),
                if retry { 4 } else { 2 }
            );
        }
    }

    #[test]
    fn observable_restart_source_requests_correction_without_replacing_body_result() {
        let mut body = SubGraph::new();
        let restart_source = body.add_node(BloqNode::classical(ClassicalNode::observable(0)));
        let result = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        body.set_value_output(Some(result.into()));
        let mut bloq = Bloq::new();
        bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            body,
            restart_condition: ClassicalExpr::In(0),
            restart_source: Some(restart_source.into()),
        }));
        bloq.validate().unwrap();
        let program = lower(&bloq, &LoweringConfig::default()).unwrap();
        let decode = program
            .tasks
            .iter()
            .find(|task| matches!(task.instruction, Instruction::Decode(_)))
            .expect("restart source requests its corrected bit");
        let (body, restart) = program
            .instructions()
            .find_map(|instruction| match instruction {
                Instruction::Rus { body, restart, .. } => Some((body, restart)),
                _ => None,
            })
            .unwrap();
        let corrected = decode.output.unwrap();
        assert_eq!(restart, &BoolOp::Copy(corrected));
        assert_ne!(body.value, Some(corrected));
        assert!(body.value.is_some());
    }

    #[test]
    fn composition_requests_no_solve_until_corrected_value_is_used() {
        for corrected_consumer in [false, true] {
            let mut bloq = Bloq::new();
            let value = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(true),
            }));
            let observable = bloq.add_node(BloqNode::classical(ClassicalNode::observable(0)));
            bloq.add_edge(value, observable, BloqEdge::value(0));
            let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
                Vec::new(),
                Vec::new(),
            )));
            bloq.add_edge(observable, raw, BloqEdge::compose(0));
            if corrected_consumer {
                let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::In(0),
                }));
                bloq.add_edge(observable, consumer, BloqEdge::value(0));
            }
            let program = lower(&bloq, &LoweringConfig::default()).unwrap();
            assert_eq!(
                program
                    .instructions()
                    .filter(|instruction| matches!(instruction, Instruction::Decode(_)))
                    .count(),
                2 * usize::from(corrected_consumer),
            );
            let (raw_task_id, raw_task) = program
                .tasks
                .iter()
                .enumerate()
                .find(|(_, task)| {
                    matches!(task.instruction, Instruction::Observable { index: 0, .. })
                })
                .unwrap();
            let projection = program
                .tasks
                .iter()
                .find(|task| task.label == task_label(&LevelPath::default(), raw, "classical"))
                .unwrap();
            assert_eq!(
                projection.instruction,
                Instruction::ReadoutRecipe {
                    bits: Box::new([raw_task.output.unwrap()]),
                    bindings: Box::new([raw_task_id as TaskId])
                }
            );
        }
    }

    fn guarded_stage(bloq: &mut Bloq, guards: u32) -> BloqNodeId {
        let template = crate::test_support::one_qubit_template(bloq, GateType::H);
        let mut stage = BloqNode::from_members(vec![]);
        let quantum = stage.expect_quantum_mut();
        for input in 0..guards {
            let instance = TemplateInstanceId(input);
            quantum
                .instances
                .push(bloq_ir::lowering::TemplateInstance::new(
                    instance,
                    template,
                    IVec2::new(input as i32, 0),
                ));
            quantum.guards.push(bloq_ir::QuantumGuard {
                input,
                instances: vec![instance],
                ..Default::default()
            });
        }
        bloq.add_node(stage)
    }

    #[test]
    fn constant_guards_do_not_require_an_exponential_choice_table() {
        for limit in [1, usize::MAX] {
            let guards = 2 * usize::BITS;
            let mut bloq = Bloq::new();
            let quantum = guarded_stage(&mut bloq, guards);
            for input in 0..guards {
                let producer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
                    expr: ClassicalExpr::Const(true),
                }));
                bloq.add_edge(producer, quantum, BloqEdge::value(input));
            }
            let program = lower(
                &bloq,
                &LoweringConfig {
                    max_quantum_variants: limit,
                    ..Default::default()
                },
            )
            .unwrap();
            let alternatives = program
                .tasks
                .iter()
                .find_map(|task| match &task.instruction {
                    Instruction::Quantum(quantum) => Some(&quantum.alternatives),
                    _ => None,
                })
                .unwrap();
            assert_eq!(alternatives.len(), 1);
            assert_eq!(alternatives[0].when.len(), guards as usize);
            assert!(alternatives[0].when.iter().all(|&(_, value)| value));
        }
    }

    #[test]
    fn correlated_guard_variants_keep_runtime_inputs_and_the_reachable_cap() {
        let mut bloq = Bloq::new();
        let raw = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let inverse = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Not(Box::new(ClassicalExpr::In(0))),
        }));
        bloq.add_edge(raw, inverse, BloqEdge::value(0));
        let stage = guarded_stage(&mut bloq, 128);
        for input in 0..128 {
            bloq.add_edge(
                if input % 2 == 0 { raw } else { inverse },
                stage,
                BloqEdge::value(input),
            );
        }
        bloq.validate().unwrap();
        let program = lower(
            &bloq,
            &LoweringConfig {
                max_quantum_variants: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let alternatives = program
            .tasks
            .iter()
            .find_map(|task| match &task.instruction {
                Instruction::Quantum(quantum) => Some(&quantum.alternatives),
                _ => None,
            })
            .unwrap();
        assert_eq!(alternatives.len(), 2);
        for (index, alternative) in alternatives.iter().enumerate() {
            assert_eq!(alternative.when.len(), 128);
            assert!(
                alternative
                    .when
                    .iter()
                    .enumerate()
                    .all(|(slot, &(_, value))| value == ((slot + index) % 2 == 0))
            );
            assert_eq!(
                alternative
                    .stream
                    .moments
                    .iter()
                    .map(|moment| moment.operations.len())
                    .sum::<usize>(),
                64
            );
        }
        let artifact = crate::run(&program, crate::RuntimeConfig::default())
            .unwrap()
            .artifact;
        assert_eq!(artifact.stop_reason, None);
        assert!(artifact.events.iter().any(|event| matches!(
            event,
            crate::runtime::ExecutionEvent::QuantumAlternative { alternative: 1, .. }
        )));
        for limit in [0, 1] {
            let error = lower(
                &bloq,
                &LoweringConfig {
                    max_quantum_variants: limit,
                    ..Default::default()
                },
            )
            .unwrap_err();
            assert!(matches!(error, LowerError::QuantumVariantLimit {
                variants, limit: observed_limit, ..
            } if variants == limit + 1 && observed_limit == limit));
        }
    }

    #[test]
    fn independent_guard_variants_remain_bounded() {
        for guards in [12, 13] {
            let mut bloq = Bloq::new();
            let quantum = guarded_stage(&mut bloq, guards);
            for input in 0..guards {
                let producer = bloq.add_node(BloqNode::classical(
                    ClassicalNode::observable_fragment(Vec::new(), Vec::new()),
                ));
                bloq.add_edge(producer, quantum, BloqEdge::value(input));
            }
            let result = lower(&bloq, &LoweringConfig::default());
            if guards == 12 {
                let program = result.unwrap();
                assert!(program.tasks.iter().any(|task| matches!(
                    &task.instruction,
                    Instruction::Quantum(quantum)
                        if quantum.alternatives.len() == DEFAULT_MAX_QUANTUM_VARIANTS
                )));
            } else {
                assert!(matches!(
                    result,
                    Err(LowerError::QuantumVariantLimit {
                        guards: 13,
                        variants: 4097,
                        limit: DEFAULT_MAX_QUANTUM_VARIANTS,
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn vm_materialization_does_not_run_the_full_ir_audit() {
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [IVec2::ZERO]).unwrap();
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let quantum = bloq.add_node(crate::test_support::quantum_node(template, 0, IVec2::ZERO));
        let constant = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        bloq.add_edge(constant, quantum, BloqEdge::value(0));

        assert!(matches!(
            bloq.validate(),
            Err(bloq_ir::BloqValidationError::UnusedQuantumValueInput {
                node,
                slot: 0
            }) if node == quantum
        ));
        crate::prepared::PreparedProgramCore::build(&bloq).unwrap();
        lower(&bloq, &LoweringConfig::default()).unwrap();
    }

    #[test]
    fn multi_pipe_origin_keeps_every_site_and_deduplicates_time() {
        let origin = pipe_origin([
            TemporalPipeRef {
                src: IVec3::new(2, 3, 0),
                dst: IVec3::new(2, 3, 1),
                hadamard: false,
            },
            TemporalPipeRef {
                src: IVec3::new(4, 5, 2),
                dst: IVec3::new(4, 5, 3),
                hadamard: true,
            },
        ]);
        assert_eq!(origin.function, TaskFunction::Memory);
        assert_eq!(origin.sites.as_ref(), [[2, 3], [4, 5]]);
        assert_eq!(
            origin.members.as_ref(),
            [[2, 3, 0], [2, 3, 1], [4, 5, 2], [4, 5, 3]]
        );
    }

    #[test]
    fn thth_sources_and_live_output_hold_have_causal_metadata() {
        let bloq = CompileContext::new(CompileConfig::new(3))
            .compile(&GalleryItem::THTH.build())
            .unwrap()
            .bloq;
        let program = lower(&bloq, &LoweringConfig::default()).unwrap();
        assert_eq!(program.task_origins.len(), program.tasks.len());
        for origin in &program.task_origins {
            assert!(origin.sites.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(origin.members.windows(2).all(|pair| pair[0] < pair[1]));
        }

        let sources = |role| {
            program
                .tasks
                .iter()
                .filter(|task| task.source == Some(role))
                .collect::<Vec<_>>()
        };
        let factories = sources(SourceRole::Factory);
        assert_eq!(factories.len(), 2);
        assert!(factories.iter().all(|task| task.release == 0.0));
        let input = sources(SourceRole::LogicalInput);
        assert_eq!(input.len(), 1);
        assert_eq!((input[0].release, input[0].duration), (0.0, 0.0));
        let clifford = sources(SourceRole::Clifford);
        assert_eq!(clifford.len(), 1);
        assert_eq!(clifford[0].release, 0.0);
        let prepared_y = sources(SourceRole::PreparedY);
        assert_eq!(prepared_y.len(), 2);
        assert!(prepared_y.iter().all(|task| {
            task.release + task.duration == clifford[0].release + clifford[0].duration
        }));
        let source_geometry = program
            .tasks
            .iter()
            .zip(&program.task_origins)
            .filter_map(|(task, origin)| task.source.map(|source| (source, origin)))
            .collect::<Vec<_>>();
        assert_eq!(
            source_geometry
                .iter()
                .filter(|(source, _)| *source == SourceRole::PreparedY)
                .flat_map(|(_, origin)| origin.sites.iter().copied())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([[3, 0], [3, 2]]),
        );
        assert!(source_geometry.iter().all(|(source, origin)| match source {
            SourceRole::Factory => origin.function == TaskFunction::Factory,
            SourceRole::LogicalInput => origin.function == TaskFunction::IdealBoundary,
            SourceRole::PreparedY => origin.function == TaskFunction::PreparedY,
            SourceRole::Clifford => {
                origin.function == TaskFunction::Cube
                    && origin.sites.as_ref() == [[-1, 0], [-1, 1], [-1, 2], [0, 2]]
            }
        }));
        let functions = program
            .task_origins
            .iter()
            .map(|origin| origin.function)
            .collect::<Vec<_>>();
        for function in [
            TaskFunction::FactoryBody,
            TaskFunction::TemporalHadamard,
            TaskFunction::Memory,
            TaskFunction::AdaptiveMeasurement,
            TaskFunction::IdealBoundary,
        ] {
            assert!(functions.contains(&function), "missing {function:?}");
        }
        assert!(
            program
                .task_origins
                .iter()
                .filter(|origin| origin.function == TaskFunction::Memory)
                .all(|origin| !origin.sites.is_empty())
        );

        assert_eq!(program.inputs.len(), 1);
        let input = &program.inputs[0];
        assert_eq!(
            program.tasks[input.task as usize].source,
            Some(SourceRole::LogicalInput)
        );
        assert!(!input.data_qubits.is_empty());
        assert!(!input.stabilizers.is_empty());
        assert!(!input.x.terms.is_empty() && !input.z.terms.is_empty());

        for factory in factories {
            let Instruction::Rus {
                body,
                owned_qubits,
                retry_prepare,
                decoder_hold,
                cultivation_exits,
                ..
            } = &factory.instruction
            else {
                panic!("factory source is not RUS")
            };
            assert!(retry_prepare.moments.is_empty());
            let mut first_use = BTreeMap::<QubitId, bool>::new();
            for &task in &body.tasks {
                let Instruction::Quantum(quantum) = &program.tasks[task as usize].instruction
                else {
                    continue;
                };
                let [alternative] = quantum.alternatives.as_ref() else {
                    panic!("cultivation body quantum task is unguarded")
                };
                for moment in &alternative.stream.moments {
                    for operation in &moment.operations {
                        let (qubits, reset) = match operation {
                            QuantumOp::Reset { qubit, .. } => (vec![*qubit], true),
                            QuantumOp::Gate1 { qubit, .. }
                            | QuantumOp::Pauli { qubit, .. }
                            | QuantumOp::T { qubit, .. }
                            | QuantumOp::ConditionalPauli { qubit, .. } => (vec![*qubit], false),
                            QuantumOp::Gate2 {
                                control, target, ..
                            } => (vec![*control, *target], false),
                            QuantumOp::Measure { observable, .. } => (
                                observable.terms.iter().map(|(qubit, _)| *qubit).collect(),
                                false,
                            ),
                            QuantumOp::Depolarize1 { .. }
                            | QuantumOp::Depolarize2 { .. }
                            | QuantumOp::PauliError { .. } => continue,
                        };
                        for qubit in qubits {
                            first_use.entry(qubit).or_insert(reset);
                        }
                    }
                }
            }
            assert_eq!(
                first_use.keys().copied().collect::<BTreeSet<_>>(),
                owned_qubits.iter().copied().collect(),
                "every owned factory qubit has an authored first use"
            );
            assert!(
                first_use.values().all(|reset| *reset),
                "every owned factory qubit is reset before first use"
            );
            assert!(!cultivation_exits.is_empty());
            assert!(cultivation_exits.iter().all(|&task| {
                body.tasks.contains(&task)
                    && matches!(
                        program.tasks[task as usize].instruction,
                        Instruction::Quantum(_)
                            | Instruction::MemoryRounds { .. }
                            | Instruction::Idle { .. }
                    )
            }));
            assert!(!decoder_hold.is_empty());
            assert!(
                decoder_hold.iter().all(|cycle| {
                    cycle.round_duration == 6.0 && !cycle.stream.moments.is_empty()
                })
            );
            assert!(body.tasks.iter().any(|&task| {
                matches!(
                    program.tasks[task as usize].instruction,
                    Instruction::Decode(DecodeRequest {
                        timing: DecodeTiming::FactoryCompletion,
                        ..
                    })
                )
            }));
        }

        let output = &program.outputs[0];
        let frame_tasks = [output.frame_x, output.frame_z].map(|bit| {
            program
                .tasks
                .iter()
                .position(|task| task.output == Some(bit))
                .unwrap() as TaskId
        });
        let terminal_hold =
            program
                .tasks
                .iter()
                .enumerate()
                .find_map(|(task, data)| match &data.instruction {
                    Instruction::WaitFor { until, memory }
                        if frame_tasks.iter().all(|frame| until.contains(frame)) =>
                    {
                        Some((task as TaskId, memory))
                    }
                    _ => None,
                });
        let (terminal_hold_task, terminal_hold) =
            terminal_hold.expect("output remains in QEC until both frames resolve");
        assert!(terminal_hold.iter().all(|cycle| {
            cycle.round_duration == 6.0
                && !cycle.stream.moments.is_empty()
                && !cycle.initializers.is_empty()
        }));
        let output_boundary = program
            .tasks
            .iter()
            .find(|task| task.dependencies.contains(&terminal_hold_task))
            .expect("terminal hold releases the output boundary");
        assert_eq!(output_boundary.duration, 0.0);
        let Instruction::Quantum(output_boundary) = &output_boundary.instruction else {
            panic!("output boundary is quantum")
        };
        assert!(output_boundary.alternatives.iter().all(|alternative| {
            alternative
                .stream
                .moments
                .iter()
                .all(|moment| moment.duration == 0.0)
        }));
    }

    #[test]
    fn zero_latency_factory_keeps_the_reusable_gap_frontier() {
        let bloq = CompileContext::new(CompileConfig::new(3))
            .compile(&GalleryItem::T.build())
            .unwrap()
            .bloq;
        let program = lower(
            &bloq,
            &LoweringConfig {
                decoder_latency_rounds: 0,
                ..LoweringConfig::default()
            },
        )
        .unwrap();
        assert_eq!(program.decoder_latency_rounds, 0);
        let gaps = program
            .tasks
            .iter()
            .filter_map(|task| match &task.instruction {
                Instruction::Rus { decoder_hold, .. } => Some(decoder_hold),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(gaps.len(), 1);
        for gap in gaps {
            assert!(gap.iter().all(|cycle| {
                !cycle.stream.moments.is_empty()
                    && !cycle.initializers.is_empty()
                    && cycle
                        .boundary_flows
                        .iter()
                        .any(|flow| flow.frontier_record.is_some())
            }));
        }
    }

    #[test]
    fn postmerge_noise_does_not_add_vm_time() {
        let bloq = CompileContext::new(CompileConfig::new(3))
            .compile(&GalleryItem::T.build())
            .unwrap()
            .bloq;
        let ideal = lower(&bloq, &LoweringConfig::default()).unwrap();
        let noise = NoiseModel::uniform_depolarizing(1e-4);
        let noisy = lower(
            &bloq,
            &LoweringConfig {
                noise: Some(&noise),
                ..LoweringConfig::default()
            },
        )
        .unwrap();

        assert_eq!(ideal.tasks.len(), noisy.tasks.len());
        for (ideal, noisy) in ideal.tasks.iter().zip(&noisy.tasks) {
            assert_eq!(ideal.label, noisy.label);
            assert_eq!(ideal.duration, noisy.duration, "{}", ideal.label);
        }
    }

    #[test]
    fn rejects_unrepresentable_durations() {
        let bloq = CompileContext::new(CompileConfig::new(3))
            .compile(&GalleryItem::CNOT.build())
            .unwrap()
            .bloq;
        for gate_duration in [crate::runtime::TIME_EPSILON, f64::MAX] {
            let error = lower(
                &bloq,
                &LoweringConfig {
                    gate_duration,
                    ..LoweringConfig::default()
                },
            )
            .unwrap_err();
            assert!(matches!(error, LowerError::InvalidDuration));
        }
    }
    #[test]
    fn fragment_flip_requires_a_complete_observable_without_a_full_audit() {
        let mut bloq = Bloq::new();
        let fragment = bloq.add_node(BloqNode::classical(ClassicalNode::observable_fragment(
            Vec::new(),
            Vec::new(),
        )));
        let consumer = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        }));
        bloq.add_edge(fragment, consumer, BloqEdge::flip(0));
        assert!(
            matches!(lower(&bloq, &LoweringConfig::default()), Err(LowerError::MissingBit { node, slot: 0 }) if node == consumer.0)
        );
    }
}
