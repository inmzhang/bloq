//! Post-compile seam edits.
//!
//! Edge subdivision is verified against persisted template boundary flows
//! before the graph is mutated, so failed verification leaves it unchanged.

use std::cmp::Ordering;
use std::collections::BTreeSet;

use bloq_circuit::{
    ComposedChain, DetectorCoords, DetectorTerm, FlowEngine, FlowKey, OffsetFlows, Op, PauliMap,
};
use glam::IVec2;
use petgraph::Direction;
use petgraph::stable_graph::NodeIndex;
use petgraph::visit::EdgeRef;
use thiserror::Error;

use crate::{
    Bloq, BloqEdge, BloqNode, BloqNodeId, BloqNodeKind, BloqTemplate, BloqTemplatePool,
    BoundaryFace, ClassicalNode, FxMap, FxSet, InstanceMeasurement, LevelPath, NodeDetector,
    NodeRestart, PipePadding, PipeSeam, QuantumEdge, SubGraph, TemplateId, TemplateInstance,
    TemplateInstanceId, TemporalPipeRef,
};

/// Why a post-compile graph edit was rejected.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EditError {
    /// A detector bundle use is malformed.
    #[error("{0}")]
    DetectorBundle(#[from] crate::DetectorBundleError),
    /// Conditional membership must be pinned first.
    #[error("memory edits require pinned component membership; call Bloq::pin_membership first")]
    MembershipSelectionRequired,
    /// Boolean membership analysis failed.
    #[error("{0}")]
    MembershipAnalysis(#[from] crate::NodeTemplateInstanceMergeError),
    /// A level path no longer resolves.
    #[error("memory-round target has invalid level path {path:?}")]
    InvalidLevelPath {
        /// Invalid level path.
        path: LevelPath,
    },
    /// The target node is missing.
    #[error("memory-round target node {node:?} does not exist at {path:?}")]
    MemoryRoundTargetMissing {
        /// Target level.
        path: LevelPath,
        /// Missing node.
        node: BloqNodeId,
    },
    /// The target node is not quantum.
    #[error("memory-round target node {node:?} at {path:?} is not quantum")]
    MemoryRoundTargetNotQuantum {
        /// Target level.
        path: LevelPath,
        /// Non-quantum node.
        node: BloqNodeId,
    },
    /// The endpoints do not identify one editable quantum seam.
    #[error(
        "quantum edge {from:?} -> {to:?} is not a single non-empty seam between quantum nodes or into a region"
    )]
    InvalidQuantumEdge {
        /// Source node.
        from: BloqNodeId,
        /// Target node.
        to: BloqNodeId,
    },
    /// Requested padding is empty.
    #[error("padding must be non-empty")]
    EmptyPadding,
    /// No template-instance id remains.
    #[error("not enough template-instance ids remain for padding")]
    TemplateInstanceIdExhausted,
    /// A referenced template is absent.
    #[error("padding references unknown template {template:?}")]
    UnknownTemplate {
        /// Missing template id.
        template: TemplateId,
    },
    /// Boundary-flow composition failed.
    #[error("seam flow composition failed: {source}")]
    SeamCompositionFailed {
        /// Underlying flow error.
        #[source]
        source: bloq_circuit::FlowError,
    },
    /// A quantum seam lacks padding provenance.
    #[error("quantum edge {from:?} -> {to:?} has no padding provenance")]
    EdgePaddingMissing {
        /// Source node.
        from: BloqNodeId,
        /// Target node.
        to: BloqNodeId,
    },
    /// A looped padding template has no single repeat.
    #[error("looped padding template {template:?} must contain one top-level REPEAT")]
    PaddingTemplateNotLooped {
        /// Invalid template.
        template: TemplateId,
    },
    /// A template does not contain exactly one top-level repeat.
    #[error("template {template:?} has {repeats} top-level REPEATs, expected exactly one")]
    TemplateNotSingleLooped {
        /// Invalid template.
        template: TemplateId,
        /// Number of top-level repeats found.
        repeats: usize,
    },
    /// Stored seam data disagrees with its templates.
    #[error("stored seam is inconsistent with template boundary flows")]
    InvalidStoredSeam,
    /// Replacement padding does not reproduce the seam.
    #[error("padding does not reproduce the stored seam")]
    SeamRecompositionMismatch,
    /// Appending after a top-level node is unsupported.
    #[error("memory rounds can only be appended after a node inside a region")]
    AfterTargetTopLevel,
    /// The append target is not terminal.
    #[error("node {node:?} is not terminal: it already has an outgoing quantum edge")]
    AfterTargetNotTerminal {
        /// Nonterminal node.
        node: BloqNodeId,
    },
    /// The enclosing boundary has no matching padding.
    #[error("node {node:?} has no matching padding provenance on its enclosing region boundary")]
    AfterPaddingMissing {
        /// Terminal node.
        node: BloqNodeId,
    },
    /// Several padding seams claim one terminal instance.
    #[error("node {node:?} has ambiguous terminal instance placement at offset {offset:?}")]
    AfterPaddingAmbiguous {
        /// Ambiguous node.
        node: BloqNodeId,
        /// Ambiguous placement.
        offset: IVec2,
    },
    /// A boundary target is not quantum.
    #[error("region boundary target {node:?} is not quantum")]
    AfterBoundaryTargetNotQuantum {
        /// Non-quantum boundary node.
        node: BloqNodeId,
    },
    /// Terminal padding does not reproduce its boundary.
    #[error("terminal padding does not reproduce the enclosing region boundary")]
    TerminalRecompositionMismatch,
}

/// Inline only bundle uses whose bound owners meet an edited seam. The whole
/// use has one activation, which transfers to every new inline row.
fn demote_touched_bundles(
    program: &Bloq,
    quantum: &mut crate::QuantumNode,
    touched: &FxSet<TemplateInstanceId>,
) -> Result<(), EditError> {
    if quantum.detector_bundles.is_empty() {
        return Ok(());
    }
    program.node_detector_count(quantum)?;
    let mut registered = FxSet::default();
    for guard in &quantum.guards {
        for &index in &guard.detector_bundles {
            if index as usize >= quantum.detector_bundles.len() || !registered.insert(index) {
                return Err(
                    crate::NodeTemplateInstanceMergeError::InvalidMembership(format!(
                        "bundle use {index} is missing or registered more than once"
                    ))
                    .into(),
                );
            }
        }
    }
    let mut remap = Vec::with_capacity(quantum.detector_bundles.len());
    let mut inline = Vec::with_capacity(quantum.detector_bundles.len());
    let mut kept = Vec::with_capacity(quantum.detector_bundles.len());
    for use_ in &quantum.detector_bundles {
        let bundle = program
            .detector_bundles()
            .get(use_.bundle)
            .ok_or(crate::DetectorBundleError::UnknownBundle(use_.bundle))?;
        if bundle
            .used_owners()?
            .iter()
            .any(|&owner| touched.contains(&use_.instances[owner as usize]))
        {
            let one = crate::QuantumNode {
                detector_bundles: vec![use_.clone()],
                ..Default::default()
            };
            let start = u32::try_from(quantum.detectors.len())
                .map_err(|_| crate::DetectorBundleError::CountOverflow)?;
            let end = start
                .checked_add(
                    u32::try_from(bundle.detectors().len())
                        .map_err(|_| crate::DetectorBundleError::CountOverflow)?,
                )
                .ok_or(crate::DetectorBundleError::CountOverflow)?;
            let indices = start..end;
            inline.push(Some(indices));
            quantum
                .detectors
                .extend(program.node_detectors(&one)?.map(|row| row.to_owned()));
            remap.push(None);
        } else {
            remap.push(Some(kept.len() as u32));
            inline.push(None);
            kept.push(use_.clone());
        }
    }
    quantum.detector_bundles = kept;
    for guard in &mut quantum.guards {
        let mut uses = Vec::new();
        for &index in &guard.detector_bundles {
            if let Some(indices) = &inline[index as usize] {
                guard.detectors.extend(indices.clone());
            }
            if let Some(index) = remap[index as usize] {
                uses.push(index);
            }
        }
        guard.detector_bundles = uses;
    }
    Ok(())
}

/// Program-unique location for physical memory-round insertion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemoryRoundTarget {
    /// Subdivide one quantum seam in `path`.
    Edge {
        /// Graph level containing the seam.
        path: LevelPath,
        /// Seam source.
        from: BloqNodeId,
        /// Seam target.
        to: BloqNodeId,
    },
    /// Append after a terminal quantum node inside a region body.
    After {
        /// Region-body level.
        path: LevelPath,
        /// Terminal quantum node.
        node: BloqNodeId,
    },
}

/// A template and offset to splice into a seam.
///
/// The edit allocates its fresh, program-unique instance id.
#[derive(Debug, Clone, Copy)]
pub struct PaddingInstance {
    /// Template to instantiate.
    pub template: TemplateId,
    /// Placement within the seam.
    pub offset: IVec2,
}

/// Memo key for `Bloq`'s `specialized_padding` cache: the recorded template the
/// specialization derives from and the number of rounds it implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PaddingVariant {
    pub(crate) base: TemplateId,
    pub(crate) rounds: u32,
}

struct TerminalMember {
    sources: Vec<TemplateInstance>,
    pipe: TemporalPipeRef,
    padding: PipePadding,
}

struct BoundaryRewrite {
    target: BloqNodeId,
    quantum: crate::QuantumNode,
}

struct QuantumBoundaryEntry {
    path: LevelPath,
    target: BloqNodeId,
}

struct QuantumBoundaryRewrite {
    path: LevelPath,
    target: BloqNodeId,
    quantum: crate::QuantumNode,
}

/// Completed seam entries keyed by node position.
#[derive(Debug, Default)]
struct SeamComposition {
    detectors: Vec<(usize, NodeDetector)>,
    restarts: Vec<(usize, NodeRestart)>,
}

impl SeamComposition {
    /// Separate the first node's entries from later nodes, retaining their order.
    fn split_first(self) -> [(Vec<NodeDetector>, Vec<NodeRestart>); 2] {
        let mut entries = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        for (position, detector) in self.detectors {
            entries[usize::from(position != 0)].0.push(detector);
        }
        for (position, restart) in self.restarts {
            entries[usize::from(position != 0)].1.push(restart);
        }
        entries
    }

    /// The entries owned by nodes strictly after `position`, dropping the
    /// position key. A composed seam's upper node owns everything past the
    /// lower node it was composed against, and no consumer needs the key for
    /// anything but this bucketing.
    fn after(self, position: usize) -> (Vec<NodeDetector>, Vec<NodeRestart>) {
        let keep = |candidate: usize| candidate > position;
        (
            self.detectors
                .into_iter()
                .filter_map(|(at, detector)| keep(at).then_some(detector))
                .collect(),
            self.restarts
                .into_iter()
                .filter_map(|(at, restart)| keep(at).then_some(restart))
                .collect(),
        )
    }
}

/// Recompose a seam from persisted [`crate::BloqTemplate::boundary_flows`].
///
/// Each node is one parallel layer, as in compile-time lowering: its instances
/// consume the previous layer before any of them publishes an output, so
/// adjacent walks never compose with each other.
///
/// This is partial: unmatched inputs and creators left open are omitted because
/// their peers may be outside the supplied nodes. Callers must verify the
/// resulting composition.
fn compose_seam_detectors(
    templates: &BloqTemplatePool,
    nodes: &[&[TemplateInstance]],
) -> Result<SeamComposition, EditError> {
    let mut engine: FlowEngine<'_, DetectorTerm<InstanceMeasurement>> = FlowEngine::new();
    let mut composition = SeamComposition::default();
    for (position, instances) in nodes.iter().enumerate() {
        let mut owners = Vec::with_capacity(instances.len());
        let mut parts = Vec::with_capacity(instances.len());
        for instance in *instances {
            let template =
                templates
                    .get(instance.template_id)
                    .ok_or(EditError::UnknownTemplate {
                        template: instance.template_id,
                    })?;
            if !template.boundary_flows.is_empty() {
                owners.push(instance.id);
                parts.push(OffsetFlows::new(&template.boundary_flows, instance.offset));
            }
        }
        engine
            .append_group_skipping_unmatched_inputs(&parts, |part, &measurement| {
                DetectorTerm::Measurement(InstanceMeasurement {
                    instance: owners[part],
                    measurement,
                })
            })
            .map_err(|source| EditError::SeamCompositionFailed { source })?;
        let (detectors, restarts) = drain_composed_chains(&mut engine);
        composition
            .detectors
            .extend(detectors.into_iter().map(|detector| (position, detector)));
        composition
            .restarts
            .extend(restarts.into_iter().map(|restart| (position, restart)));
    }
    // Open creators may close against nodes outside this partial seam.
    Ok(composition)
}

/// Find the stored multiset entries replaced by a recomposed seam.
fn removal_mask<T: PartialEq>(stored: &[T], seam: &[T]) -> Option<Vec<bool>> {
    let mut remove = vec![false; stored.len()];
    for expected in seam {
        let index = stored
            .iter()
            .enumerate()
            .find_map(|(index, item)| (!remove[index] && item == expected).then_some(index))?;
        remove[index] = true;
    }
    Some(remove)
}

fn placed_boundary(map: &PauliMap, offset: IVec2) -> Result<FlowKey<'_>, EditError> {
    FlowKey::try_new(map, offset).map_err(|error| EditError::SeamCompositionFailed {
        source: error.into(),
    })
}

/// The `instances` each padding item continues, one list per item.
///
/// Boundaries match exactly as flow composition matches them, so a walk is
/// found at its destination, and a patch whose output stabilizers are split
/// between a cube and its spatial Hadamard pipe is found on both. An item with
/// no matching boundary falls back to the sole instance at its placement.
fn padding_sources(
    templates: &BloqTemplatePool,
    instances: &[TemplateInstance],
    padding: &[PaddingInstance],
) -> Result<Vec<Vec<TemplateInstance>>, EditError> {
    let flows = |template| {
        templates
            .get(template)
            .map(|template: &BloqTemplate| template.boundary_flows.as_slice())
            .ok_or(EditError::UnknownTemplate { template })
    };
    let mut producers = FxMap::default();
    for (index, instance) in instances.iter().enumerate() {
        for flow in flows(instance.template_id)? {
            if !flow.end.is_empty() {
                producers.insert(placed_boundary(&flow.end, instance.offset)?, index);
            }
        }
    }
    padding
        .iter()
        .map(|item| {
            let mut sources = BTreeSet::new();
            for flow in flows(item.template)? {
                if !flow.start.is_empty()
                    && let Some(&index) = producers.get(&placed_boundary(&flow.start, item.offset)?)
                {
                    sources.insert(index);
                }
            }
            if sources.is_empty() {
                let mut placed = instances
                    .iter()
                    .filter(|instance| instance.offset == item.offset);
                return Ok(match (placed.next(), placed.next()) {
                    (Some(&only), None) => vec![only],
                    _ => Vec::new(),
                });
            }
            Ok(sources.into_iter().map(|index| instances[index]).collect())
        })
        .collect()
}

/// Map every source to the padding that replaces it. Each padding item needs a
/// source, and no source may feed two items.
fn padding_replacements(
    sources: &[Vec<TemplateInstance>],
    padding: &[TemplateInstance],
) -> Option<FxMap<TemplateInstanceId, TemplateInstance>> {
    let mut replacements = FxMap::default();
    for (sources, &instance) in sources.iter().zip(padding) {
        if sources.is_empty() {
            return None;
        }
        for source in sources {
            if replacements.insert(source.id, instance).is_some() {
                return None;
            }
        }
    }
    Some(replacements)
}

/// Total order on detector centers, so sorting makes a `Vec` comparison a
/// multiset comparison. `None` — a chain carrying no center — sorts before
/// every `Some` and stays distinct from `Some([])`.
fn compare_centers(left: &Option<DetectorCoords>, right: &Option<DetectorCoords>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => left.len().cmp(&right.len()).then_with(|| {
            std::iter::zip(left, right)
                .map(|(left, right)| left.total_cmp(right))
                .find(|order| order.is_ne())
                .unwrap_or(Ordering::Equal)
        }),
    }
}

/// Whether a recomposed seam reproduces the one it replaces,
/// plaquette-for-plaquette.
///
/// Padding continues the source patch's memory rounds, so the chains that close
/// against the node above compare the same plaquettes as before — one layer
/// later, against different measurement indices. Centers are therefore the
/// invariant: comparing parities would compare indices that legitimately moved.
///
/// The comparison is over the center **multiset**, not the set: sorting both
/// sides and comparing the `Vec`s tests length first, so a seam that dropped or
/// duplicated a chain fails even though the distinct centers still agree.
fn seam_reproduced(original: &[NodeDetector], replacement: &[NodeDetector]) -> bool {
    fn centers(detectors: &[NodeDetector]) -> Vec<Option<DetectorCoords>> {
        let mut centers: Vec<_> = detectors
            .iter()
            .map(|detector| detector.coords.clone())
            .collect();
        centers.sort_by(compare_centers);
        centers
    }
    centers(original) == centers(replacement)
}

/// Rewrite the one top-level repeat whose cardinality the caller checked.
fn set_single_repeat_count(template: &mut BloqTemplate, repetitions: u32) {
    let entry = template.circuit.entry_body();
    for op in template
        .circuit
        .body_mut(entry)
        .expect("every circuit has an entry body")
        .ops_mut()
    {
        if let Op::Repeat { repetitions: r, .. } = op {
            *r = repetitions;
        }
    }
}

fn retain_unmasked<T>(stored: &mut Vec<T>, remove: &[bool]) {
    debug_assert_eq!(stored.len(), remove.len());
    let mut remove = remove.iter();
    stored.retain(|_| !*remove.next().expect("removal mask matches side table"));
}

/// Move every [`BoundaryFace::Output`] operator that named a replaced instance
/// onto its replacement.
///
/// The surface a program's observable ends on now ends after the padding, so an
/// operator stating where the program exits has to follow it.
fn retarget_output_operators(
    level: &mut SubGraph,
    replacements: &FxMap<TemplateInstanceId, TemplateInstanceId>,
) {
    for node in level.graph_mut().node_weights_mut() {
        match &mut node.kind {
            BloqNodeKind::Classical(data)
                if data.operators().iter().any(|operator| {
                    operator.face == BoundaryFace::Output
                        && replacements.contains_key(&operator.instance)
                }) =>
            {
                let ClassicalNode::Observable { operators, .. } = std::sync::Arc::make_mut(data)
                else {
                    unreachable!("classical output checked above")
                };
                for operator in operators {
                    if operator.face == BoundaryFace::Output
                        && let Some(&replacement) = replacements.get(&operator.instance)
                    {
                        operator.instance = replacement;
                    }
                }
            }
            BloqNodeKind::Region(region) => {
                for (_, body) in region.bodies_mut() {
                    retarget_output_operators(body, replacements);
                }
            }
            BloqNodeKind::Quantum(_) | BloqNodeKind::Classical(_) => {}
        }
    }
}

/// Keep observable bindings and logical-output metadata on the same new cut.
fn retarget_outputs(bloq: &mut Bloq, replacements: &FxMap<TemplateInstanceId, TemplateInstanceId>) {
    retarget_output_operators(bloq.top_mut(), replacements);
    let mut outputs = bloq.logical_outputs().to_vec();
    for output in &mut outputs {
        if let Some(&replacement) = replacements.get(&output.instance) {
            output.instance = replacement;
        }
    }
    bloq.set_logical_outputs(outputs);
}

/// Drain completed chains into detector and restart side tables.
///
/// Classification is shared with compile-time detector lowering.
pub fn drain_composed_chains(
    engine: &mut FlowEngine<'_, DetectorTerm<InstanceMeasurement>>,
) -> (Vec<NodeDetector>, Vec<NodeRestart>) {
    let chains = engine.drain_composed_chains();
    let mut detectors = Vec::with_capacity(chains.size_hint().1.unwrap_or(0));
    let mut restarts = Vec::new();
    for chain in chains {
        match chain {
            ComposedChain::Restart { parity } => restarts.push(NodeRestart { parity }),
            ComposedChain::Detector { parity, center } => {
                detectors.push(NodeDetector {
                    parity,
                    coords: center
                        .map(|coords| [coords.x as f64, coords.y as f64].into_iter().collect()),
                });
            }
        }
    }
    (detectors, restarts)
}

impl Bloq {
    /// Insert physical memory rounds at a level-qualified quantum edge or
    /// after a region body's terminal quantum node.
    ///
    /// Templates come from compiler-recorded padding provenance, so the edit
    /// works on a deserialized program without its source graph. The edit is
    /// transactional: any error leaves `self` unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`EditError`] when the target or stored seam provenance is
    /// invalid, membership is unresolved, or `rounds` is zero.
    pub fn insert_memory_rounds(
        &mut self,
        target: MemoryRoundTarget,
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        Ok(self.insert_memory_rounds_batch(&[target], rounds)?[0])
    }

    fn insert_memory_rounds_in_place(
        &mut self,
        target: &MemoryRoundTarget,
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        if self.timing_requires_pins()? {
            return Err(EditError::MembershipSelectionRequired);
        }
        if rounds == 0 {
            return Err(EditError::EmptyPadding);
        }
        match target {
            MemoryRoundTarget::Edge { path, from, to } => {
                self.splice_seam_node_at(path, *from, *to, rounds)
            }
            MemoryRoundTarget::After { path, node } => {
                self.append_memory_rounds_at(path, *node, rounds)
            }
        }
    }

    /// Resolve the seam's recorded padding into instantiable templates and
    /// splice `node` in, shared by every edge-target insertion.
    fn splice_seam_node_at(
        &mut self,
        path: &LevelPath,
        from: BloqNodeId,
        to: BloqNodeId,
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        let level = self
            .level_at(path)
            .ok_or_else(|| EditError::InvalidLevelPath { path: path.clone() })?;
        let seams = level
            .edges_between(from, to)
            .find_map(|edge| match edge.edge {
                BloqEdge::Quantum(edge) => Some(edge.pipes.clone()),
                BloqEdge::Value { .. } | BloqEdge::Compose { .. } | BloqEdge::Order => None,
            })
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;
        // Subdivision replaces the seam whole, so every pipe on it must
        // carry provenance — a partially recorded edge is not editable.
        let padding_refs: Option<Vec<_>> = seams.iter().map(|seam| seam.padding).collect();
        let padding_refs = padding_refs
            .filter(|refs| !refs.is_empty())
            .ok_or(EditError::EdgePaddingMissing { from, to })?;
        let padding = self.padding_for_refs(&padding_refs, rounds)?;
        self.subdivide_quantum_edge_at(path, from, to, &padding, rounds)
    }

    /// Retune a template's bulk duration: rewrite the repetition count of its
    /// single top-level `REPEAT`.
    ///
    /// This is how a compiled program's memory-round budget is changed without
    /// recompiling, and it is a *structural* rewrite: neither the template's
    /// side tables nor its boundary flows depend on the repetition count (a
    /// loop's recurrence is stated per iteration), so nothing else has to move.
    /// Measurement ids are per body, not per iteration, so the program's
    /// measurement registry is unaffected too.
    ///
    /// Templates are pooled and shared, so **every** instance of `template`
    /// changes with it. That is usually the point — one cube kind, one bulk
    /// duration — but a caller that means to retune only some placements must
    /// check the sharing itself; this call cannot tell the two intents apart.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::UnknownTemplate`] for an id outside the pool, and
    /// [`EditError::TemplateNotSingleLooped`] when the template's entry body
    /// has anything other than exactly one top-level `REPEAT` — with zero
    /// there is no loop to retune, and with several the call would silently
    /// mean "all of them".
    ///
    /// # Panics
    ///
    /// Panics only if a template disappears after it is read from the pool.
    pub fn set_template_repetitions(
        &mut self,
        template: TemplateId,
        repetitions: u32,
    ) -> Result<(), EditError> {
        let repeats = self
            .templates()
            .get(template)
            .ok_or(EditError::UnknownTemplate { template })?
            .circuit
            .entry_top_level_repeats()
            .len();
        if repeats != 1 {
            return Err(EditError::TemplateNotSingleLooped { template, repeats });
        }
        let template = self
            .templates_mut()
            .make_mut(template)
            .expect("the template was just read from the pool");
        set_single_repeat_count(template, repetitions);
        self.specialized_padding.clear();
        Ok(())
    }

    /// Insert memory rounds at every target or at none, returning node ids in
    /// target order. Targets may mix edges and terminal nodes at any level.
    /// An empty target list is a no-op, but zero rounds is always rejected.
    ///
    /// # Errors
    ///
    /// Returns [`EditError`] if membership is unresolved, any target or seam
    /// is invalid, or `rounds` is zero. Any error leaves `self` unchanged.
    ///
    /// Editing a clone and swapping on success costs one clone for the whole
    /// batch, where looping over the public [`Bloq::insert_memory_rounds`]
    /// would clone once per target *and* leave the earlier targets padded when
    /// a later one fails. Targets are applied in order. A duplicate or overlapping
    /// target that becomes invalid also rolls back the whole batch.
    pub fn insert_memory_rounds_batch(
        &mut self,
        targets: &[MemoryRoundTarget],
        rounds: u32,
    ) -> Result<Vec<BloqNodeId>, EditError> {
        if rounds == 0 {
            return Err(EditError::EmptyPadding);
        }
        if self.timing_requires_pins()? {
            return Err(EditError::MembershipSelectionRequired);
        }
        let mut edited = self.clone();
        let padding = targets
            .iter()
            .map(|target| edited.insert_memory_rounds_in_place(target, rounds))
            .collect::<Result<Vec<_>, _>>()?;
        *self = edited;
        Ok(padding)
    }

    fn padding_for_refs(
        &mut self,
        padding_refs: &[PipePadding],
        rounds: u32,
    ) -> Result<Vec<PaddingInstance>, EditError> {
        padding_refs
            .iter()
            .map(|entry| {
                let template = self.padding_template_for_rounds(entry, rounds)?;
                Ok(PaddingInstance {
                    template,
                    offset: entry.offset,
                })
            })
            .collect()
    }

    /// Resolve or memoize the template implementing `rounds` memory rounds.
    /// Only the repetition count varies; side tables and boundary flows are
    /// count-independent.
    fn padding_template_for_rounds(
        &mut self,
        entry: &PipePadding,
        rounds: u32,
    ) -> Result<TemplateId, EditError> {
        if rounds == 0 {
            return Err(EditError::EmptyPadding);
        }
        if rounds == 1 {
            return Ok(entry.one_round);
        }
        let repetitions = rounds - 1;
        let template = self
            .templates()
            .get(entry.looped)
            .ok_or(EditError::UnknownTemplate {
                template: entry.looped,
            })?;
        let [current] = template.circuit.entry_top_level_repeats()[..] else {
            return Err(EditError::PaddingTemplateNotLooped {
                template: entry.looped,
            });
        };
        if current == repetitions {
            return Ok(entry.looped);
        }
        let key = PaddingVariant {
            base: entry.looped,
            rounds,
        };
        if let Some(&specialized) = self.specialized_padding.get(&key) {
            return Ok(specialized);
        }

        let mut specialized = template.clone();
        set_single_repeat_count(&mut specialized, repetitions);
        let specialized = self.add_template(specialized);
        self.specialized_padding.insert(key, specialized);
        Ok(specialized)
    }

    /// Splice padding into `from -> to`, producing `from -> padding -> to`.
    ///
    /// Before mutation, this allocates fresh instance ids and verifies the
    /// original stored seam entries before recomposing both replacement
    /// halves. Partial seams leaving a merge may move closures between nodes
    /// and retarget output-face operators; full-node seams preserve side-table
    /// entries one-for-one. `rounds` is recorded as memory-padding provenance
    /// on the new node.
    ///
    /// Returns the new padding node id. Re-resolve node ids held across this
    /// call because vacant slots may be recycled.
    ///
    /// # Errors
    ///
    /// Returns [`EditError`] when the seam, padding instances, or stored
    /// detector composition is invalid.
    pub fn subdivide_quantum_edge(
        &mut self,
        from: BloqNodeId,
        to: BloqNodeId,
        padding: &[PaddingInstance],
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        self.subdivide_quantum_edge_at(&LevelPath::default(), from, to, padding, rounds)
    }

    fn subdivide_quantum_edge_at(
        &mut self,
        path: &LevelPath,
        from: BloqNodeId,
        to: BloqNodeId,
        padding: &[PaddingInstance],
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        if self.timing_requires_pins()? {
            return Err(EditError::MembershipSelectionRequired);
        }
        if rounds == 0 || padding.is_empty() {
            return Err(EditError::EmptyPadding);
        }

        // Verify before mutation.
        let (from_index, to_index) = (
            NodeIndex::new(from.0 as usize),
            NodeIndex::new(to.0 as usize),
        );
        let level = self
            .level_at(path)
            .ok_or_else(|| EditError::InvalidLevelPath { path: path.clone() })?;
        let mut quantum_edges = level
            .graph()
            .edges_connecting(from_index, to_index)
            .filter(|edge| matches!(edge.weight(), BloqEdge::Quantum(_)));
        let edge = quantum_edges
            .next()
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;
        // Removal masks require this to be the sole quantum interface.
        if quantum_edges.next().is_some() {
            return Err(EditError::InvalidQuantumEdge { from, to });
        }
        let edge_id = edge.id();
        let BloqEdge::Quantum(quantum_edge) = edge.weight() else {
            unreachable!("filtered to quantum edges")
        };
        let quantum_edge = quantum_edge.clone();
        if quantum_edge.pipes.is_empty() {
            return Err(EditError::InvalidQuantumEdge { from, to });
        }

        let lower = level
            .node(from)
            .expect("a graph edge's source node is live");
        let upper = level.node(to).expect("a graph edge's target node is live");
        let lower_quantum = lower
            .try_quantum()
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;
        if let Some(region) = upper.try_region() {
            let mut entries = Vec::new();
            for (body_selector, body) in region.bodies() {
                let mut covered = vec![0usize; padding.len()];
                for (target, node) in body.nodes() {
                    let Some(quantum) = node.try_quantum() else {
                        continue;
                    };
                    if body
                        .incoming(target)
                        .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
                    {
                        continue;
                    }
                    let mut matched = false;
                    for (index, padding) in padding.iter().enumerate() {
                        if quantum
                            .instances
                            .iter()
                            .any(|instance| instance.offset == padding.offset)
                        {
                            covered[index] += 1;
                            matched = true;
                        }
                    }
                    if matched {
                        entries.push(QuantumBoundaryEntry {
                            path: path.child(to, body_selector),
                            target,
                        });
                    }
                }
                if covered.iter().any(|&count| count != 1) {
                    return Err(EditError::InvalidQuantumEdge { from, to });
                }
            }
            let lower_quantum = lower_quantum.clone();
            return self.subdivide_quantum_boundary_input_at(
                path,
                from,
                to,
                padding,
                rounds,
                quantum_edge,
                lower_quantum,
                entries,
            );
        }
        let upper_quantum = upper
            .try_quantum()
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;
        if !upper_quantum.guards.is_empty() {
            let entries = vec![QuantumBoundaryEntry {
                path: path.clone(),
                target: to,
            }];
            let lower_quantum = lower_quantum.clone();
            return self.subdivide_quantum_boundary_input_at(
                path,
                from,
                to,
                padding,
                rounds,
                quantum_edge,
                lower_quantum,
                entries,
            );
        }

        // Only instances carried by this edge are adjacent to `upper`; other
        // members of the same node may reach it through later seams.
        let sources = padding_sources(self.templates(), &lower_quantum.instances, padding)?;
        let lower_before: Vec<_> = sources.iter().flatten().copied().collect();

        // Recomposition catches Hadamard-aligned chains that parity alone cannot.
        let original =
            compose_seam_detectors(self.templates(), &[&lower_before, &upper_quantum.instances])?;
        let [
            (lower_detectors, lower_restarts),
            (original_detectors, original_restarts),
        ] = original.split_first();
        let touched = lower_before
            .iter()
            .map(|instance| instance.id)
            .chain(
                upper_quantum
                    .instances
                    .iter()
                    .filter(|instance| padding.iter().any(|item| item.offset == instance.offset))
                    .map(|instance| instance.id),
            )
            .collect::<FxSet<_>>();
        let mut lower_quantum = lower_quantum.clone();
        let mut upper_quantum = upper_quantum.clone();
        demote_touched_bundles(self, &mut lower_quantum, &touched)?;
        demote_touched_bundles(self, &mut upper_quantum, &touched)?;
        if removal_mask(&lower_quantum.detectors, &lower_detectors).is_none()
            || removal_mask(&lower_quantum.restarts, &lower_restarts).is_none()
        {
            return Err(EditError::InvalidStoredSeam);
        }
        let detector_remove = removal_mask(&upper_quantum.detectors, &original_detectors)
            .ok_or(EditError::InvalidStoredSeam)?;
        let restart_remove = removal_mask(&upper_quantum.restarts, &original_restarts)
            .ok_or(EditError::InvalidStoredSeam)?;

        let padding_instances = self.allocate_padding_instances(padding)?;
        let replacements = padding_replacements(&sources, &padding_instances)
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;

        let internal =
            compose_seam_detectors(self.templates(), &[&lower_before, &padding_instances])?;
        let [
            (composed_lower_detectors, composed_lower_restarts),
            (mid_detectors, mid_restarts),
        ] = internal.split_first();
        if removal_mask(&lower_quantum.detectors, &composed_lower_detectors).is_none()
            || removal_mask(&lower_quantum.restarts, &composed_lower_restarts).is_none()
        {
            return Err(EditError::InvalidStoredSeam);
        }

        let replacement = compose_seam_detectors(
            self.templates(),
            &[&padding_instances, &upper_quantum.instances],
        )?;
        let (upper_detectors, upper_restarts) = replacement.after(0);

        // Only the seam against `upper` has a counterpart to reproduce. The
        // `lower -> padding` seam is new, and its size is set by the padding's
        // own boundary rather than by what it displaced: a resolved Y-basis
        // target pairs its plaquettes into composite seam chains, so an
        // ordinary memory seam underneath it legitimately carries twice the
        // detectors the original seam did.
        if !seam_reproduced(&original_detectors, &upper_detectors)
            || upper_restarts.len() != original_restarts.len()
        {
            return Err(EditError::SeamRecompositionMismatch);
        }
        let partial_seam = replacements.len() < lower_quantum.instances.len();

        // Apply only after all checks pass. Any pipe gives the shared seam layer.
        let mut node = BloqNode::memory_padding(quantum_edge.pipes[0].pipe, rounds);
        {
            let quantum = node.expect_quantum_mut();
            quantum.instances = padding_instances;
            quantum.detectors = mid_detectors;
            quantum.restarts = mid_restarts;
        }
        let upper_work = upper_quantum;
        let level = self
            .level_at_mut(path)
            .expect("validated level path remains live during insertion");
        let mid = level.add_node(node);
        let graph = level.graph_mut();
        graph.remove_edge(edge_id);
        *graph[from_index].expect_quantum_mut() = lower_quantum;
        let upper_quantum = graph[to_index].expect_quantum_mut();
        *upper_quantum = upper_work;
        retain_unmasked(&mut upper_quantum.detectors, &detector_remove);
        retain_unmasked(&mut upper_quantum.restarts, &restart_remove);
        upper_quantum.detectors.extend(upper_detectors);
        upper_quantum.restarts.extend(upper_restarts);

        level.add_edge(from, mid, BloqEdge::Quantum(quantum_edge.clone()));
        level.add_edge(mid, to, BloqEdge::Quantum(quantum_edge));
        if partial_seam {
            let output_replacements = replacements
                .into_iter()
                .map(|(source, padding)| (source, padding.id))
                .collect();
            retarget_outputs(self, &output_replacements);
        }
        Ok(mid)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the seam edit needs both endpoints, payloads, and replacement state"
    )]
    fn subdivide_quantum_boundary_input_at(
        &mut self,
        path: &LevelPath,
        from: BloqNodeId,
        to: BloqNodeId,
        padding: &[PaddingInstance],
        rounds: u32,
        quantum_edge: Box<QuantumEdge>,
        mut source: crate::QuantumNode,
        entries: Vec<QuantumBoundaryEntry>,
    ) -> Result<BloqNodeId, EditError> {
        let sources = padding_sources(self.templates(), &source.instances, padding)?;
        let source_before: Vec<_> = sources.iter().flatten().copied().collect();
        let padding_instances = self.allocate_padding_instances(padding)?;
        padding_replacements(&sources, &padding_instances)
            .ok_or(EditError::InvalidQuantumEdge { from, to })?;

        let internal =
            compose_seam_detectors(self.templates(), &[&source_before, &padding_instances])?;
        let [
            (source_detectors, source_restarts),
            (padding_detectors, padding_restarts),
        ] = internal.split_first();
        let touched = source_before.iter().map(|instance| instance.id).collect();
        demote_touched_bundles(self, &mut source, &touched)?;
        if removal_mask(&source.detectors, &source_detectors).is_none()
            || removal_mask(&source.restarts, &source_restarts).is_none()
        {
            return Err(EditError::InvalidStoredSeam);
        }

        let mut rewrites = Vec::with_capacity(entries.len());
        for entry in entries {
            let quantum = self.recompose_boundary_at(
                &entry.path,
                entry.target,
                &source_before,
                &padding_instances,
            )?;
            rewrites.push(QuantumBoundaryRewrite {
                path: entry.path,
                target: entry.target,
                quantum,
            });
        }

        for rewrite in rewrites {
            *self
                .level_at_mut(&rewrite.path)
                .expect("validated level remains live")
                .node_mut(rewrite.target)
                .expect("validated boundary target remains live")
                .expect_quantum_mut() = rewrite.quantum;
        }

        let mut node = BloqNode::memory_padding(quantum_edge.pipes[0].pipe, rounds);
        {
            let quantum = node.expect_quantum_mut();
            quantum.instances = padding_instances;
            quantum.detectors = padding_detectors;
            quantum.restarts = padding_restarts;
        }
        let level = self
            .level_at_mut(path)
            .expect("validated level path remains live during insertion");
        *level
            .node_mut(from)
            .expect("validated source remains live")
            .expect_quantum_mut() = source;
        let mid = level.add_node(node);
        let edge = level
            .graph()
            .edges_connecting(
                NodeIndex::new(from.0 as usize),
                NodeIndex::new(to.0 as usize),
            )
            .find(|edge| matches!(edge.weight(), BloqEdge::Quantum(_)))
            .expect("validated quantum edge remains live")
            .id();
        level.graph_mut().remove_edge(edge);
        level.add_edge(from, mid, BloqEdge::Quantum(quantum_edge.clone()));
        level.add_edge(mid, to, BloqEdge::Quantum(quantum_edge));
        Ok(mid)
    }

    /// Recompose each mutually exclusive payload independently. Guarded rows
    /// retain their indices and input dependencies; only their parity deltas
    /// change under the choice that owns the replaced seam.
    fn recompose_boundary_at(
        &self,
        path: &LevelPath,
        target: BloqNodeId,
        before: &[TemplateInstance],
        after: &[TemplateInstance],
    ) -> Result<crate::QuantumNode, EditError> {
        let level = self.level_at(path).expect("validated boundary level");
        let mut node = level[target].clone();
        let touched = before
            .iter()
            .chain(after)
            .map(|instance| instance.id)
            .chain(
                node.expect_quantum()
                    .instances
                    .iter()
                    .filter(|instance| before.iter().any(|source| source.offset == instance.offset))
                    .map(|instance| instance.id),
            )
            .collect::<FxSet<_>>();
        demote_touched_bundles(self, node.expect_quantum_mut(), &touched)?;
        let quantum = node.expect_quantum();
        let mut updated = quantum.clone();
        let choices = if quantum.guards.is_empty() {
            vec![crate::membership::QuantumSelection {
                input: 0,
                values: FxMap::default(),
            }]
        } else {
            crate::membership::PredicateAnalysis::new(level)
                .quantum_selections(target)?
                .ok_or(EditError::MembershipSelectionRequired)?
        };
        for choice in choices {
            let selected = node.select_quantum_members(|slot| choice.values.get(&slot).copied())?;
            let selected = selected.expect_quantum();
            let original =
                compose_seam_detectors(self.templates(), &[before, &selected.instances])?;
            let (original_detectors, original_restarts) = original.after(0);
            let detector_remove = removal_mask(&selected.detectors, &original_detectors)
                .ok_or(EditError::InvalidStoredSeam)?;
            let restart_remove = removal_mask(&selected.restarts, &original_restarts)
                .ok_or(EditError::InvalidStoredSeam)?;
            let replacement =
                compose_seam_detectors(self.templates(), &[after, &selected.instances])?;
            let (mut detectors, restarts) = replacement.after(0);
            if !seam_reproduced(&original_detectors, &detectors)
                || restarts.len() != original_restarts.len()
            {
                return Err(EditError::SeamRecompositionMismatch);
            }
            if quantum.guards.is_empty() {
                retain_unmasked(&mut updated.detectors, &detector_remove);
                retain_unmasked(&mut updated.restarts, &restart_remove);
                updated.detectors.extend(detectors);
                updated.restarts.extend(restarts);
                continue;
            }
            let detector_indices = (0..quantum.detectors.len() as u32).filter(|index| {
                !quantum
                    .guards
                    .iter()
                    .any(|guard| !choice.values[&guard.input] && guard.detectors.contains(index))
            });
            let mut changed = detector_indices
                .zip(&selected.detectors)
                .zip(detector_remove)
                .filter_map(|((index, row), remove)| remove.then_some((index, row)))
                .collect::<Vec<_>>();
            changed.sort_by(|(_, left), (_, right)| compare_centers(&left.coords, &right.coords));
            detectors.sort_by(|left, right| compare_centers(&left.coords, &right.coords));
            let guard = updated
                .guards
                .iter_mut()
                .find(|guard| guard.input == choice.input)
                .expect("a choice is a member registration");
            for ((index, old), new) in changed.into_iter().zip(detectors) {
                let mut delta = old.parity.clone();
                delta.xor_assign(&new.parity);
                if !delta.is_empty() {
                    guard.detector_parities.push((index, delta));
                }
            }
            let restart_indices = (0..quantum.restarts.len() as u32).filter(|index| {
                !quantum
                    .guards
                    .iter()
                    .any(|guard| !choice.values[&guard.input] && guard.restarts.contains(index))
            });
            let changed = restart_indices
                .zip(&selected.restarts)
                .zip(restart_remove)
                .filter_map(|((index, row), remove)| remove.then_some((index, row)));
            for ((index, old), new) in changed.zip(restarts) {
                let mut delta = old.parity.clone();
                delta.xor_assign(&new.parity);
                if !delta.is_empty() {
                    guard.restart_parities.push((index, delta));
                }
            }
        }
        Ok(updated)
    }

    fn append_memory_rounds_at(
        &mut self,
        path: &LevelPath,
        source: BloqNodeId,
        rounds: u32,
    ) -> Result<BloqNodeId, EditError> {
        let level = self
            .level_at(path)
            .ok_or_else(|| EditError::InvalidLevelPath { path: path.clone() })?;
        let source_node =
            level
                .node(source)
                .ok_or_else(|| EditError::MemoryRoundTargetMissing {
                    path: path.clone(),
                    node: source,
                })?;
        let source_quantum =
            source_node
                .try_quantum()
                .ok_or_else(|| EditError::MemoryRoundTargetNotQuantum {
                    path: path.clone(),
                    node: source,
                })?;
        if !source_quantum.guards.is_empty() {
            return Err(EditError::MembershipSelectionRequired);
        }
        if level
            .outgoing(source)
            .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
        {
            return Err(EditError::AfterTargetNotTerminal { node: source });
        }
        let source_instances = source_quantum.instances.clone();

        let Some((segment, parent_path)) = path.segments().split_last() else {
            return Err(EditError::AfterTargetTopLevel);
        };
        let region = segment.region;
        let (mut members, mut boundary_targets) = (Vec::<TerminalMember>::new(), Vec::new());
        {
            let parent = self
                .level_at_segments(parent_path)
                .ok_or_else(|| EditError::InvalidLevelPath { path: path.clone() })?;
            for edge_ref in parent.outgoing(region) {
                let BloqEdge::Quantum(edge) = edge_ref.edge else {
                    continue;
                };
                let mut matched = false;
                for seam in &edge.pipes {
                    let Some(padding) = seam.padding else {
                        continue;
                    };
                    let sources = padding_sources(
                        self.templates(),
                        &source_instances,
                        &[PaddingInstance {
                            template: padding.one_round,
                            offset: padding.offset,
                        }],
                    )?
                    .remove(0);
                    if sources.is_empty() {
                        continue;
                    }
                    if sources.iter().any(|instance| {
                        members
                            .iter()
                            .flat_map(|member| &member.sources)
                            .any(|claimed| claimed.id == instance.id)
                    }) {
                        return Err(EditError::AfterPaddingAmbiguous {
                            node: source,
                            offset: padding.offset,
                        });
                    }
                    members.push(TerminalMember {
                        sources,
                        pipe: seam.pipe,
                        padding,
                    });
                    matched = true;
                }
                if matched {
                    if parent
                        .node(edge_ref.target)
                        .and_then(BloqNode::try_quantum)
                        .is_none()
                    {
                        return Err(EditError::AfterBoundaryTargetNotQuantum {
                            node: edge_ref.target,
                        });
                    }
                    if boundary_targets.contains(&edge_ref.target) {
                        return Err(EditError::TerminalRecompositionMismatch);
                    }
                    boundary_targets.push(edge_ref.target);
                }
            }
        }
        if members.is_empty() {
            return Err(EditError::AfterPaddingMissing { node: source });
        }
        let mut source_quantum = source_quantum.clone();
        let touched = members
            .iter()
            .flat_map(|member| member.sources.iter().map(|instance| instance.id))
            .collect();
        demote_touched_bundles(self, &mut source_quantum, &touched)?;
        let source_detectors = source_quantum.detectors.clone();
        let source_restarts = source_quantum.restarts.clone();
        let layer = members[0].pipe.src.z.min(members[0].pipe.dst.z);
        if members
            .iter()
            .any(|member| member.pipe.src.z.min(member.pipe.dst.z) != layer)
        {
            return Err(EditError::TerminalRecompositionMismatch);
        }

        let refs: Vec<_> = members.iter().map(|member| member.padding).collect();
        let padding = self.padding_for_refs(&refs, rounds)?;
        let padding_instances = self.allocate_padding_instances(&padding)?;
        let replacements: FxMap<_, _> = members
            .iter()
            .zip(&padding_instances)
            .flat_map(|(member, &padding)| {
                member
                    .sources
                    .iter()
                    .map(move |source| (source.id, padding))
            })
            .collect();
        let lower_after: Vec<_> = source_instances
            .iter()
            .filter(|instance| !replacements.contains_key(&instance.id))
            .chain(&padding_instances)
            .copied()
            .collect();

        let internal =
            compose_seam_detectors(self.templates(), &[&source_instances, &padding_instances])?;
        let [
            (source_composed_detectors, source_composed_restarts),
            (padding_detectors, padding_restarts),
        ] = internal.split_first();
        if removal_mask(&source_detectors, &source_composed_detectors).is_none()
            || removal_mask(&source_restarts, &source_composed_restarts).is_none()
        {
            return Err(EditError::InvalidStoredSeam);
        }

        let mut boundary_rewrites = Vec::with_capacity(boundary_targets.len());
        let boundary_path = parent_path
            .iter()
            .fold(LevelPath::default(), |path, segment| {
                path.child(segment.region, segment.body)
            });
        for target in boundary_targets {
            boundary_rewrites.push(BoundaryRewrite {
                target,
                quantum: self.recompose_boundary_at(
                    &boundary_path,
                    target,
                    &source_instances,
                    &lower_after,
                )?,
            });
        }

        // Every fallible check is complete. Apply parent seam repairs first.
        {
            let parent = self
                .level_at_segments_mut(parent_path)
                .expect("validated parent level remains live during insertion");
            for rewrite in boundary_rewrites {
                let upper = parent
                    .node_mut(rewrite.target)
                    .expect("validated boundary target remains live")
                    .expect_quantum_mut();
                *upper = rewrite.quantum;
            }
        }

        let mut padding_node = BloqNode::memory_padding(members[0].pipe, rounds);
        {
            let quantum = padding_node.expect_quantum_mut();
            quantum.instances = padding_instances;
            quantum.detectors = padding_detectors;
            quantum.restarts = padding_restarts;
        }
        let padding_node = {
            let level = self
                .level_at_mut(path)
                .expect("validated target level remains live during insertion");
            *level
                .node_mut(source)
                .expect("validated source remains live")
                .expect_quantum_mut() = source_quantum;
            let order_edges: Vec<_> = level
                .graph()
                .edges_directed(NodeIndex::new(source.0 as usize), Direction::Outgoing)
                .filter(|edge| matches!(edge.weight(), BloqEdge::Order))
                .map(|edge| (edge.id(), BloqNodeId(edge.target().index() as u32)))
                .collect();
            let padding_node = level.add_node(padding_node);
            for (edge, _) in &order_edges {
                level.graph_mut().remove_edge(*edge);
            }
            level.add_edge(
                source,
                padding_node,
                BloqEdge::Quantum(Box::new(QuantumEdge {
                    guard: None,
                    pipes: members
                        .iter()
                        .map(|member| PipeSeam {
                            pipe: member.pipe,
                            padding: Some(member.padding),
                        })
                        .collect(),
                })),
            );
            for (_, target) in order_edges {
                level.add_edge(padding_node, target, BloqEdge::Order);
            }
            padding_node
        };

        let output_replacements: FxMap<_, _> = replacements
            .into_iter()
            .map(|(source, padding)| (source, padding.id))
            .collect();
        retarget_outputs(self, &output_replacements);
        Ok(padding_node)
    }

    fn allocate_padding_instances(
        &self,
        padding: &[PaddingInstance],
    ) -> Result<Vec<TemplateInstance>, EditError> {
        let base = match self.max_template_instance_id() {
            Some(id) => {
                id.0.checked_add(1)
                    .ok_or(EditError::TemplateInstanceIdExhausted)?
            }
            None => 0,
        };
        let last_index =
            u32::try_from(padding.len() - 1).map_err(|_| EditError::TemplateInstanceIdExhausted)?;
        let last = base
            .checked_add(last_index)
            .ok_or(EditError::TemplateInstanceIdExhausted)?;
        Ok((base..=last)
            .zip(padding)
            .map(|(id, instance)| {
                TemplateInstance::new(TemplateInstanceId(id), instance.template, instance.offset)
            })
            .collect())
    }

    /// Largest template-instance id across all nested regions.
    fn max_template_instance_id(&self) -> Option<TemplateInstanceId> {
        self.levels()
            .flat_map(|(_, level)| level.quantum_nodes())
            .flat_map(|(_, quantum)| quantum.instances.iter().map(|i| i.id))
            .max()
    }
}

#[cfg(test)]
mod tests {
    use bloq_circuit::CoordCircuit;

    use super::*;
    use crate::{
        BodySelector, ClassicalExpr, InstanceBoundaryOperator, NodeDetectorParity, RegionNode,
    };
    use glam::IVec3;

    /// A detector carrying only a center; `seam_reproduced` ignores parities,
    /// which legitimately move to the padding's measurement indices.
    fn centered(center: Option<[f64; 2]>) -> NodeDetector {
        NodeDetector {
            parity: NodeDetectorParity::default(),
            coords: center.map(|center| center.into_iter().collect()),
        }
    }

    #[test]
    fn touched_bundle_demotion_transfers_guard_and_preserves_shared_snapshot() {
        use crate::{
            BundleDetector, BundleMeasurement, DetectorBundle, DetectorBundleUse, QuantumGuard,
        };

        let mut program = Bloq::new();
        let bundle = program.add_detector_bundle(DetectorBundle::new(
            vec![TemplateId(0)],
            vec![BundleDetector {
                parity: bloq_circuit::DetectorParity::from_measurements([BundleMeasurement {
                    owner: 0,
                    measurement: 0,
                }]),
                coords: Some([1.0, 2.0].into_iter().collect()),
            }],
        ));
        let mut quantum = crate::QuantumNode {
            detector_bundles: vec![
                DetectorBundleUse {
                    bundle,
                    instances: vec![TemplateInstanceId(5)],
                    offset: IVec2::ZERO,
                },
                DetectorBundleUse {
                    bundle,
                    instances: vec![TemplateInstanceId(6)],
                    offset: IVec2::ZERO,
                },
            ],
            guards: vec![QuantumGuard {
                input: 7,
                detector_bundles: vec![0, 1],
                ..Default::default()
            }],
            ..Default::default()
        };
        let snapshot = quantum.clone();
        demote_touched_bundles(
            &program,
            &mut quantum,
            &FxSet::from_iter([TemplateInstanceId(5)]),
        )
        .unwrap();
        assert_eq!(
            quantum.detectors[0]
                .parity
                .measurements()
                .next()
                .unwrap()
                .instance,
            TemplateInstanceId(5)
        );
        assert_eq!(quantum.guards[0].detectors, vec![0]);
        assert_eq!(quantum.guards[0].detector_bundles, vec![0]);
        assert_eq!(
            quantum.detector_bundles[0].instances,
            vec![TemplateInstanceId(6)]
        );
        assert_eq!(snapshot.detector_bundles.len(), 2);
        assert!(snapshot.detectors.is_empty());
    }

    /// The seam check compares center *multisets*: a replacement that keeps
    /// every distinct plaquette but loses or gains a chain must be rejected,
    /// which is what makes a short padding fail rather than slip through.
    #[test]
    fn seam_reproduction_counts_repeated_centers() {
        let original = [
            centered(Some([2.0, 0.0])),
            centered(Some([2.0, 0.0])),
            centered(Some([4.0, 2.0])),
        ];

        assert!(seam_reproduced(&original, &original));
        // Order must not matter.
        assert!(seam_reproduced(
            &original,
            &[
                centered(Some([4.0, 2.0])),
                centered(Some([2.0, 0.0])),
                centered(Some([2.0, 0.0])),
            ]
        ));
        // Same distinct centers, one chain short.
        assert!(!seam_reproduced(
            &original,
            &[centered(Some([2.0, 0.0])), centered(Some([4.0, 2.0]))]
        ));
        // Same length, one plaquette swapped.
        assert!(!seam_reproduced(
            &original,
            &[
                centered(Some([2.0, 0.0])),
                centered(Some([2.0, 0.0])),
                centered(Some([6.0, 2.0])),
            ]
        ));
        // A center-less chain is distinct from one centered at the origin.
        assert!(!seam_reproduced(
            &[centered(None)],
            &[centered(Some([0.0, 0.0]))]
        ));
    }

    fn plain_pipe() -> TemporalPipeRef {
        TemporalPipeRef {
            src: IVec3::new(0, 0, 0),
            dst: IVec3::new(0, 0, 1),
            hadamard: false,
        }
    }

    /// A flowless two-node seam for testing edit guards.
    fn seam_bloq(pipe: TemporalPipeRef) -> (Bloq, BloqNodeId, BloqNodeId, TemplateId) {
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut lower = BloqNode::from_members(vec![]);
        lower
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template,
                IVec2::ZERO,
            ));
        let lower = bloq.add_node(lower);
        let upper = bloq.add_node(BloqNode::from_members(vec![]));
        bloq.add_edge(lower, upper, BloqEdge::quantum(vec![pipe]));
        (bloq, lower, upper, template)
    }

    #[test]
    fn empty_padding_is_rejected() {
        let (mut bloq, lower, upper, _) = seam_bloq(plain_pipe());
        assert!(matches!(
            bloq.subdivide_quantum_edge(lower, upper, &[], 1),
            Err(EditError::EmptyPadding)
        ));
    }

    #[test]
    fn exhausted_template_instance_ids_are_rejected_without_mutation() {
        for (existing_id, padding_count) in [(u32::MAX, 1), (u32::MAX - 1, 2)] {
            let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
            bloq.node_mut(lower).unwrap().expect_quantum_mut().instances[0].id =
                TemplateInstanceId(existing_id);
            let padding = vec![
                PaddingInstance {
                    template,
                    offset: IVec2::ZERO,
                };
                padding_count
            ];

            assert!(matches!(
                bloq.subdivide_quantum_edge(lower, upper, &padding, 1),
                Err(EditError::TemplateInstanceIdExhausted)
            ));
            assert_eq!(bloq.node_count(), 2, "failed edit leaves graph unchanged");
            assert_eq!(
                bloq.graph()
                    .edges_connecting(
                        NodeIndex::new(lower.0 as usize),
                        NodeIndex::new(upper.0 as usize),
                    )
                    .count(),
                1,
                "failed edit preserves the original seam"
            );
        }
    }

    #[test]
    fn quantum_edge_without_pipes_is_rejected() {
        let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
        let edge = bloq
            .graph()
            .edges_connecting(
                NodeIndex::new(lower.0 as usize),
                NodeIndex::new(upper.0 as usize),
            )
            .next()
            .unwrap()
            .id();
        *bloq.graph_mut_internal().edge_weight_mut(edge).unwrap() = BloqEdge::quantum(vec![]);
        let padding = [PaddingInstance {
            template,
            offset: IVec2::ZERO,
        }];

        assert!(matches!(
            bloq.subdivide_quantum_edge(lower, upper, &padding, 1),
            Err(EditError::InvalidQuantumEdge { .. })
        ));
        assert_eq!(bloq.node_count(), 2, "failed edit leaves graph unchanged");
    }

    #[test]
    fn missing_quantum_edge_is_rejected() {
        let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
        let padding = [PaddingInstance {
            template,
            offset: IVec2::ZERO,
        }];
        // Reversed direction: no Quantum edge upper -> lower.
        assert!(matches!(
            bloq.subdivide_quantum_edge(upper, lower, &padding, 1),
            Err(EditError::InvalidQuantumEdge { .. })
        ));
    }

    #[test]
    fn hadamard_pipe_can_be_subdivided() {
        let (mut bloq, lower, upper, template) = seam_bloq(TemporalPipeRef {
            hadamard: true,
            ..plain_pipe()
        });
        let padding = [PaddingInstance {
            template,
            offset: IVec2::ZERO,
        }];
        bloq.subdivide_quantum_edge(lower, upper, &padding, 1)
            .expect("Hadamard provenance on an edge does not block subdivision");
    }

    #[test]
    fn parallel_quantum_edges_are_rejected() {
        let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
        bloq.add_edge(lower, upper, BloqEdge::quantum(vec![plain_pipe()]));
        let padding = [PaddingInstance {
            template,
            offset: IVec2::ZERO,
        }];
        assert!(matches!(
            bloq.subdivide_quantum_edge(lower, upper, &padding, 1),
            Err(EditError::InvalidQuantumEdge { .. })
        ));
    }

    /// A node is one parallel layer: a neighbor's output is not its input.
    #[test]
    fn instances_of_one_node_do_not_compose_with_each_other() {
        use bloq_circuit::{Flow, Pauli, PauliMap};

        let key: PauliMap = [(IVec2::ZERO, Pauli::Z)].into_iter().collect();
        let mut creator = BloqTemplate::new(CoordCircuit::new());
        creator.boundary_flows =
            vec![Flow::new(PauliMap::empty(), key.clone()).with_measurements([0])];
        let mut consumer = BloqTemplate::new(CoordCircuit::new());
        consumer.boundary_flows = vec![Flow::new(key, PauliMap::empty()).with_measurements([0])];

        let mut bloq = Bloq::new();
        let creator = bloq.add_template(creator);
        let consumer = bloq.add_template(consumer);
        let node = [
            TemplateInstance::new(TemplateInstanceId(0), creator, IVec2::ZERO),
            TemplateInstance::new(TemplateInstanceId(1), consumer, IVec2::ZERO),
        ];

        let composition = compose_seam_detectors(bloq.templates(), &[&node]).unwrap();
        assert!(composition.detectors.is_empty());
        assert!(composition.restarts.is_empty());
    }

    /// A flowless padding template with one top-level repeat.
    fn looped_template(repetitions: u32) -> BloqTemplate {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(bloq_circuit::CircuitBody::new());
        let entry = circuit.entry_body();
        circuit
            .body_mut(entry)
            .expect("entry body exists")
            .ops_mut()
            .push(Op::Repeat { body, repetitions });
        BloqTemplate::new(circuit)
    }

    /// A seam with one-round and three-round padding templates.
    fn seam_bloq_with_padding() -> (Bloq, BloqNodeId, BloqNodeId) {
        let (mut bloq, lower, upper, one_round) = seam_bloq(plain_pipe());
        let looped = bloq.add_template(looped_template(2));
        assert!(bloq.set_quantum_edge_padding(
            lower,
            upper,
            vec![crate::PipePadding {
                offset: IVec2::ZERO,
                one_round,
                looped,
            }],
        ));
        (bloq, lower, upper)
    }

    #[test]
    fn selective_padding_preserves_shared_rows_and_both_record_choices() {
        let mut bloq = Bloq::from_text(
            "BLOQIR 1
template t0 {
  circuit {
    M (0,0):m0
  }
  flow _ -> Z(0,0) meas m0
}
template t1 {
  circuit {
    M (0,0):m0
  }
  flow Z(0,0) -> _ meas m0
}
template t2 {
  circuit {
    M (0,0):m0
    TICK
    M (0,0):m1
  }
  flow Z(0,0) -> _ meas m1
}
template t3 {
  circuit {
    M (0,0):m0
  }
  flow Z(0,0) -> _ meas m0
  flow _ -> Z(0,0) meas m0
}
graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 quantum {
    instance i1 t1 @ (0,0)
    instance i2 t2 @ (0,0)
    detector i0:m0
    guard 0 i1 xd0=i1:m0
    guard 1 i2 xd0=i2:m1
  }
  n2 observable fragment measurements i0:m0
  n3 compute !in0
  n0 -> n1 quantum (0,0,0)>(0,0,1)
  n0 -> n2 order
  n2 -> n3 value 0
  n2 -> n1 value 0
  n3 -> n1 value 1
}
",
        )
        .unwrap();
        bloq.validate().unwrap();
        let selection = BloqNodeId(1);
        let padding = bloq
            .subdivide_quantum_edge(
                BloqNodeId(0),
                selection,
                &[PaddingInstance {
                    template: TemplateId(3),
                    offset: IVec2::ZERO,
                }],
                1,
            )
            .unwrap();
        bloq.validate().unwrap();
        assert_eq!(bloq[selection].expect_quantum().detectors.len(), 1);
        assert_eq!(bloq.value_inputs(selection).count(), 2);
        for value in [false, true] {
            let selected = bloq[selection]
                .select_quantum_members(|slot| Some(if slot == 0 { value } else { !value }))
                .unwrap();
            let selected = selected.expect_quantum();
            let (expected, _) = compose_seam_detectors(
                bloq.templates(),
                &[
                    &bloq[padding].expect_quantum().instances,
                    &selected.instances,
                ],
            )
            .unwrap()
            .after(0);
            assert_eq!(selected.detectors, expected);
            assert!(
                !selected.detectors[0]
                    .parity
                    .measurements()
                    .any(|term| term.instance == TemplateInstanceId(0))
            );
        }
    }

    fn body_path(bloq: &Bloq, region: BloqNodeId) -> LevelPath {
        bloq.levels()
            .find_map(|(path, _)| {
                (path == LevelPath::default().child(region, BodySelector::Body)).then_some(path)
            })
            .expect("region body path exists")
    }

    /// The `REPEAT` count of the single padding instance's template on `node`.
    fn spliced_repetitions(bloq: &Bloq, node: BloqNodeId) -> Option<u32> {
        let instance = &bloq
            .node(node)
            .expect("node exists")
            .expect_quantum()
            .instances[0];
        bloq.templates()[instance.template_id]
            .circuit
            .entry_top_level_repeats()
            .first()
            .copied()
    }

    #[test]
    fn memory_round_insertion_of_zero_rounds_is_rejected() {
        let (mut bloq, lower, upper) = seam_bloq_with_padding();
        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: lower,
                    to: upper
                },
                0
            ),
            Err(EditError::EmptyPadding)
        ));
    }

    #[test]
    fn memory_round_edge_target_supports_nested_levels() {
        let mut bloq = Bloq::new();
        let one_round = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let looped = bloq.add_template(looped_template(2));
        let pipe = plain_pipe();
        let mut body = SubGraph::new();
        let mut lower = BloqNode::from_members(vec![]);
        lower
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                one_round,
                IVec2::ZERO,
            ));
        let lower = body.add_node(lower);
        let upper = body.add_node(BloqNode::from_members(vec![]));
        body.add_edge(
            lower,
            upper,
            BloqEdge::Quantum(Box::new(QuantumEdge {
                guard: None,
                pipes: vec![PipeSeam {
                    pipe,
                    padding: Some(PipePadding {
                        offset: IVec2::ZERO,
                        one_round,
                        looped,
                    }),
                }],
            })),
        );
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        let path = body_path(&bloq, region);

        let padding = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: path.clone(),
                    from: lower,
                    to: upper,
                },
                2,
            )
            .expect("nested seam padding");

        let body = bloq.level_at(&path).expect("body remains addressable");
        assert_eq!(body[padding].memory_rounds(), Some(2));
        assert!(
            body.edges_between(lower, padding)
                .any(|edge| { matches!(edge.edge, BloqEdge::Quantum(_)) })
        );
        assert!(
            body.edges_between(padding, upper)
                .any(|edge| { matches!(edge.edge, BloqEdge::Quantum(_)) })
        );

        // Activation suppresses physical work, including the retry body.
        let guard = bloq.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        bloq.node_mut(region).unwrap().activation = Some(0);
        bloq.add_edge(guard, region, BloqEdge::value(0));
        let unchanged = bloq.to_binary();
        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path,
                    from: lower,
                    to: padding,
                },
                1,
            ),
            Err(EditError::MembershipSelectionRequired)
        ));
        assert_eq!(bloq.to_binary(), unchanged);
    }

    #[test]
    fn memory_round_target_rejects_a_stale_level_path() {
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let source = body.add_node(BloqNode::from_members(vec![]));
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        let path = body_path(&bloq, region);
        bloq.remove_node(region);

        assert!(matches!(
            bloq.insert_memory_rounds(MemoryRoundTarget::After { path, node: source }, 1),
            Err(EditError::InvalidLevelPath { .. })
        ));
    }

    #[test]
    fn memory_round_after_target_rejects_malformed_nodes_without_mutation() {
        let mut bloq = Bloq::new();
        let mut body = SubGraph::new();
        let classical = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let missing = body.add_node(BloqNode::from_members(vec![]));
        body.remove_node(missing);
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        let path = body_path(&bloq, region);
        let unchanged = bloq.to_binary();

        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::After {
                    path: path.clone(),
                    node: missing,
                },
                1,
            ),
            Err(EditError::MemoryRoundTargetMissing { .. })
        ));
        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::After {
                    path,
                    node: classical,
                },
                1,
            ),
            Err(EditError::MemoryRoundTargetNotQuantum { .. })
        ));
        assert_eq!(bloq.to_binary(), unchanged);
    }

    #[test]
    fn memory_round_after_target_appends_a_real_terminal_node() {
        let mut bloq = Bloq::new();
        let one_round = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let looped = bloq.add_template(looped_template(2));
        let pipe = plain_pipe();
        let mut body = SubGraph::new();
        let mut source_node = BloqNode::from_members(vec![]);
        source_node
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                one_round,
                IVec2::ZERO,
            ));
        let source = body.add_node(source_node);
        let include = body.add_node(BloqNode::classical(ClassicalNode::Observable {
            index: None,
            measurements: Vec::new(),
            operators: vec![InstanceBoundaryOperator {
                instance: TemplateInstanceId(0),
                face: BoundaryFace::Output,
                operator: bloq_circuit::PauliMap::empty(),
            }],
        }));
        let guard = body.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        body.node_mut(include).unwrap().activation = Some(0);
        body.add_edge(guard, include, BloqEdge::value(0));
        body.add_edge(source, include, BloqEdge::Order);
        let region = bloq.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::Const(false),
            body,
        }));
        let upper = bloq.add_node(BloqNode::from_members(vec![]));
        bloq.add_edge(
            region,
            upper,
            BloqEdge::Quantum(Box::new(QuantumEdge {
                guard: None,
                pipes: vec![PipeSeam {
                    pipe,
                    padding: Some(PipePadding {
                        offset: IVec2::ZERO,
                        one_round,
                        looped,
                    }),
                }],
            })),
        );
        let path = body_path(&bloq, region);
        bloq.set_logical_outputs(vec![crate::LogicalOutput {
            port: IVec3::ZERO,
            instance: TemplateInstanceId(0),
            x: bloq_circuit::PauliMap::empty(),
            z: bloq_circuit::PauliMap::empty(),
        }]);
        let original = bloq.clone();

        let padding = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::After {
                    path: path.clone(),
                    node: source,
                },
                2,
            )
            .expect("terminal padding");

        let body = bloq.level_at(&path).expect("body remains addressable");
        let padding_instance = body[padding].expect_quantum().instances[0].id;
        assert_eq!(body[padding].memory_rounds(), Some(2));
        assert!(
            body.edges_between(source, padding)
                .any(|edge| { matches!(edge.edge, BloqEdge::Quantum(_)) })
        );
        assert!(
            body.edges_between(padding, include)
                .any(|edge| matches!(edge.edge, BloqEdge::Order))
        );
        assert!(
            !body
                .edges_between(source, include)
                .any(|edge| matches!(edge.edge, BloqEdge::Order))
        );
        let Some(ClassicalNode::Observable { operators, .. }) = body[include].try_classical()
        else {
            panic!("include remains classical")
        };
        assert_eq!(operators[0].instance, padding_instance);
        assert_eq!(body[include].activation, Some(0));
        assert!(
            body.value_inputs(include)
                .any(|input| input.producer == guard)
        );
        assert_eq!(bloq.logical_outputs()[0].instance, padding_instance);
        let original_body = original.level_at(&path).unwrap();
        let Some(ClassicalNode::Observable { operators, .. }) =
            original_body[include].try_classical()
        else {
            panic!("shared observable remains classical")
        };
        assert_eq!(operators[0].instance, TemplateInstanceId(0));
        assert_eq!(original_body[include].activation, Some(0));
        assert_eq!(
            original.logical_outputs()[0].instance,
            TemplateInstanceId(0)
        );
    }

    #[test]
    fn partial_seam_padding_moves_only_its_logical_output_owner() {
        let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
        bloq.node_mut(lower)
            .unwrap()
            .expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(1),
                template,
                IVec2::X,
            ));
        bloq.set_logical_outputs(
            (0..2)
                .map(|instance| crate::LogicalOutput {
                    port: IVec3::new(instance as i32, 0, 0),
                    instance: TemplateInstanceId(instance),
                    x: bloq_circuit::PauliMap::empty(),
                    z: bloq_circuit::PauliMap::empty(),
                })
                .collect(),
        );
        let padding = bloq
            .subdivide_quantum_edge(
                lower,
                upper,
                &[PaddingInstance {
                    template,
                    offset: IVec2::ZERO,
                }],
                1,
            )
            .unwrap();
        let owner = bloq[padding].expect_quantum().instances[0].id;
        assert_eq!(bloq.logical_outputs()[0].instance, owner);
        assert_eq!(bloq.logical_outputs()[1].instance, TemplateInstanceId(1));
    }

    #[test]
    fn memory_round_insertion_without_padding_provenance_is_rejected() {
        let (mut bloq, lower, upper, _) = seam_bloq(plain_pipe());
        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: lower,
                    to: upper
                },
                2
            ),
            Err(EditError::EdgePaddingMissing { .. })
        ));
    }

    #[test]
    fn memory_round_insertion_uses_recorded_templates_without_cloning() {
        // Both recorded template forms avoid growing the pool.
        for (rounds, expect_repeat) in [(1, None), (3, Some(2))] {
            let (mut bloq, lower, upper) = seam_bloq_with_padding();
            let pool_before = bloq.templates().len();
            let mid = bloq
                .insert_memory_rounds(
                    MemoryRoundTarget::Edge {
                        path: LevelPath::default(),
                        from: lower,
                        to: upper,
                    },
                    rounds,
                )
                .expect("flowless decoder wait splices");
            assert_eq!(
                bloq.node(mid).expect("mid exists").memory_rounds(),
                Some(rounds)
            );
            assert_eq!(spliced_repetitions(&bloq, mid), expect_repeat);
            assert_eq!(bloq.templates().len(), pool_before);
        }
    }

    #[test]
    fn template_edits_invalidate_cached_padding_specializations() {
        let (mut bloq, lower, upper) = seam_bloq_with_padding();
        let pool_before = bloq.templates().len();
        let mid = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: lower,
                    to: upper,
                },
                5,
            )
            .expect("specialized decoder wait splices");
        assert_eq!(spliced_repetitions(&bloq, mid), Some(4));
        assert_eq!(bloq.templates().len(), pool_before + 1);

        let specialized = bloq[mid].expect_quantum().instances[0].template_id;
        bloq.set_template_repetitions(specialized, 99)
            .expect("retuning succeeds");

        // The stale cached template is not reused.
        let second = bloq
            .insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: mid,
                    to: upper,
                },
                5,
            )
            .expect("second decoder wait splices");
        assert_eq!(spliced_repetitions(&bloq, second), Some(4));
        assert_eq!(bloq.templates().len(), pool_before + 2);
    }

    #[test]
    fn memory_round_insertion_on_unlooped_looped_template_is_rejected() {
        // Corrupt provenance: the looped template has no repeat.
        let (mut bloq, lower, upper, flowless) = seam_bloq(plain_pipe());
        assert!(bloq.set_quantum_edge_padding(
            lower,
            upper,
            vec![crate::PipePadding {
                offset: IVec2::ZERO,
                one_round: flowless,
                looped: flowless,
            }],
        ));
        assert!(matches!(
            bloq.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path: LevelPath::default(),
                    from: lower,
                    to: upper
                },
                2
            ),
            Err(EditError::PaddingTemplateNotLooped { .. })
        ));
    }

    /// Every landmark the seam-locating padding helpers address, in one
    /// program: a `RepeatUntilSuccess` whose body escapes into a quantum node,
    /// two selective selections, and a plain block-to-block seam.
    ///
    /// `padded_second_selection` drops the padding provenance from the second
    /// selection's seam, which is what makes a mid-batch insertion failure — and
    /// so the all-or-nothing contract — reachable.
    fn seam_landmarks(padded_second_selection: bool) -> Bloq {
        let second = match padded_second_selection {
            true => " padding offset (4,0) one t0 loop t1",
            false => "",
        };
        Bloq::from_text(&format!(
            "\
BLOQIR 1

template t0 {{
  circuit {{
    MPP X(0,0):m0
  }}
}}

template t1 {{
  circuit {{
    REPEAT 2 b1
  }}
  body b1 {{
    MPP X(0,0):m0
  }}
}}

graph {{
  n0 rus in0 source n2 {{
    body {{
      n0 quantum {{
        instance i0 t0 @ (0,0)
      }}
      n1 quantum {{
        instance i1 t0 @ (0,0)
      }}
      n2 observable fragment measurements i1:m0 from generator 0
      n0 -> n1 quantum (0,0,0)>(0,0,1)
      n1 -> n2 order
    }}
  }}
  n1 quantum {{
    instance i2 t0 @ (0,0)
    from blocks (0,0,2)
  }}
  n2 quantum {{
    instance i3 t0 @ (0,0)
    instance i9 t0 @ (4,0)
    instance i4 t0 @ (0,0)
    instance i11 t0 @ (0,0)
    instance i10 t0 @ (4,0)
    guard 0 i3 i9
    guard 1 i4 i11 i10
  }}
  n6 compute 0
  n7 compute !in0
  n6 -> n7 value 0
  n6 -> n2 value 0
  n7 -> n2 value 1
  n3 quantum {{
    instance i5 t0 @ (4,0)
    from blocks (1,0,0)
  }}
  n4 quantum {{
    instance i6 t0 @ (4,0)
    from blocks (1,0,1)
  }}
  n5 quantum {{
    instance i7 t0 @ (4,0)
    instance i8 t0 @ (4,0)
    guard 0 i7
    guard 1 i8
  }}
  n6 -> n5 value 0
  n7 -> n5 value 1
  n0 -> n1 quantum (0,0,1)>(0,0,2) padding offset (0,0) one t0 loop t1
  n1 -> n2 quantum (0,0,2)>(0,0,3) padding offset (0,0) one t0 loop t1
  n4 -> n2 quantum (1,0,1)>(1,0,2) padding offset (4,0) one t0 loop t1
  n3 -> n4 quantum (1,0,0)>(1,0,1) padding offset (4,0) one t0 loop t1
  n4 -> n5 quantum (1,0,1)>(1,0,2){second}
}}
"
        ))
        .expect("valid .bloqir text")
    }

    fn selection_targets(bloq: &Bloq) -> Vec<MemoryRoundTarget> {
        bloq.selection_seams()
            .unwrap()
            .into_iter()
            .map(|(from, to)| MemoryRoundTarget::Edge {
                path: LevelPath::default(),
                from,
                to,
            })
            .collect()
    }

    #[test]
    fn memory_round_batch_subdivides_every_selected_seam() {
        let mut bloq = seam_landmarks(true);
        let targets = selection_targets(&bloq);
        let padding = bloq.insert_memory_rounds_batch(&targets, 3).unwrap();

        assert_eq!(padding.len(), 3, "one padding node per selection seam");
        for (&padding, selection) in
            padding
                .iter()
                .zip([BloqNodeId(2), BloqNodeId(2), BloqNodeId(5)])
        {
            assert_eq!(bloq[padding].memory_rounds(), Some(3));
            assert!(
                bloq.edges_between(padding, selection)
                    .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
            );
        }
    }

    #[test]
    fn memory_round_batch_is_all_or_nothing() {
        let mut bloq = seam_landmarks(false);
        let targets = selection_targets(&bloq);
        let before = bloq.to_binary();
        // Earlier seams are valid, but the last has no padding provenance.
        assert!(matches!(
            bloq.insert_memory_rounds_batch(&targets, 1),
            Err(EditError::EdgePaddingMissing { .. })
        ));
        assert_eq!(bloq.to_binary(), before);
    }

    #[test]
    fn memory_round_batch_mixes_edges_and_region_terminals() {
        let mut bloq = seam_landmarks(true);
        let path = LevelPath::default().child(BloqNodeId(0), BodySelector::Body);
        let escape = bloq.level_at(&path).unwrap().quantum_tail().unwrap();
        let mut targets = selection_targets(&bloq);
        targets.push(MemoryRoundTarget::After {
            path: path.clone(),
            node: escape,
        });
        let padding = bloq.insert_memory_rounds_batch(&targets, 2).unwrap();

        assert_eq!(padding.len(), 4);
        for &node in &padding[..3] {
            assert_eq!(bloq[node].memory_rounds(), Some(2));
        }
        let body = bloq.level_at(&path).unwrap();
        let inserted = padding[3];
        assert_eq!(body[inserted].memory_rounds(), Some(2));
        assert!(
            body.edges_between(escape, inserted)
                .any(|edge| matches!(edge.edge, BloqEdge::Quantum(_)))
        );
        assert_eq!(body.quantum_tail(), Ok(inserted));
    }

    #[test]
    fn memory_round_batch_rolls_back_a_region_terminal_edit() {
        let mut bloq = seam_landmarks(false);
        let path = LevelPath::default().child(BloqNodeId(0), BodySelector::Body);
        let escape = bloq.level_at(&path).unwrap().quantum_tail().unwrap();
        let bad_edge = selection_targets(&bloq).pop().unwrap();
        let targets = [MemoryRoundTarget::After { path, node: escape }, bad_edge];
        let before = bloq.to_binary();
        assert!(matches!(
            bloq.insert_memory_rounds_batch(&targets, 2),
            Err(EditError::EdgePaddingMissing { .. })
        ));
        assert_eq!(bloq.to_binary(), before);
    }

    #[test]
    fn memory_round_batch_handles_empty_zero_and_duplicate_targets() {
        let mut bloq = seam_landmarks(true);
        let before = bloq.to_binary();
        assert_eq!(bloq.insert_memory_rounds_batch(&[], 1), Ok(Vec::new()));
        assert_eq!(bloq.to_binary(), before);
        assert_eq!(
            bloq.insert_memory_rounds_batch(&[], 0),
            Err(EditError::EmptyPadding)
        );
        let target = selection_targets(&bloq).remove(0);
        assert!(matches!(
            bloq.insert_memory_rounds_batch(&[target.clone(), target], 1),
            Err(EditError::InvalidQuantumEdge { .. })
        ));
        assert_eq!(bloq.to_binary(), before);
    }

    #[test]
    fn template_repetitions_retune_in_place_without_touching_a_shared_pool() {
        let mut bloq = Bloq::new();
        let looped = bloq.add_shared_template(std::sync::Arc::new(looped_template(8)));
        // A second program sharing the same `Arc` stands in for a template
        // handed to a sibling program; the retune must not reach it.
        let mut shared = Bloq::new();
        let shared_id = shared.add_shared_template(std::sync::Arc::clone(
            bloq.templates().iter_shared().next().unwrap().1,
        ));
        assert!(std::ptr::eq(
            &bloq.templates()[looped],
            &shared.templates()[shared_id]
        ));

        bloq.set_template_repetitions(looped, 3)
            .expect("a single-loop template retunes");
        assert!(!std::ptr::eq(
            &bloq.templates()[looped],
            &shared.templates()[shared_id]
        ));

        assert_eq!(
            bloq.templates()[looped].circuit.entry_top_level_repeats(),
            [3]
        );
        assert_eq!(
            shared.templates()[shared_id]
                .circuit
                .entry_top_level_repeats(),
            [8],
            "retuning is copy-on-write"
        );
    }

    #[test]
    fn retuning_rejects_a_template_with_no_single_loop() {
        let mut bloq = Bloq::new();
        let flat = bloq.add_template(BloqTemplate::new(CoordCircuit::new()));
        let mut twice = looped_template(2);
        let body = twice.circuit.entry_body();
        let repeat = twice.circuit.body(body).expect("entry body").ops()[0].clone();
        twice
            .circuit
            .body_mut(body)
            .expect("entry body")
            .ops_mut()
            .push(repeat);
        let twice = bloq.add_template(twice);

        assert_eq!(
            bloq.set_template_repetitions(flat, 1),
            Err(EditError::TemplateNotSingleLooped {
                template: flat,
                repeats: 0,
            })
        );
        assert_eq!(
            bloq.set_template_repetitions(twice, 1),
            Err(EditError::TemplateNotSingleLooped {
                template: twice,
                repeats: 2,
            })
        );
        assert_eq!(
            bloq.set_template_repetitions(TemplateId(9), 1),
            Err(EditError::UnknownTemplate {
                template: TemplateId(9),
            })
        );
    }

    #[test]
    fn splice_rewires_edge_and_allocates_fresh_instance_id() {
        let (mut bloq, lower, upper, template) = seam_bloq(plain_pipe());
        let padding = [PaddingInstance {
            template,
            offset: IVec2::ZERO,
        }];
        let mid = bloq
            .subdivide_quantum_edge(lower, upper, &padding, 1)
            .expect("flowless seam splice succeeds");

        // The original edge is gone.
        assert!(matches!(
            bloq.subdivide_quantum_edge(lower, upper, &padding, 1),
            Err(EditError::InvalidQuantumEdge { .. })
        ));
        let order = bloq
            .deterministic_emit_order()
            .expect("spliced program is acyclic");
        assert_eq!(order, vec![lower, mid, upper]);

        let mid_node = bloq.node(mid).expect("padding node exists");
        assert_eq!(mid_node.layer(), 1, "padding sits on the pipe layer");
        assert_eq!(
            mid_node.memory_rounds(),
            Some(1),
            "padding is identifiable and carries its round count"
        );
        // The lower node already uses id 0.
        assert_eq!(
            mid_node.expect_quantum().instances[0].id,
            TemplateInstanceId(1)
        );
        bloq.validate().expect("spliced program validates");
    }
}
