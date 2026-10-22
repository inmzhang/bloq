use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::ops::Index;
use std::sync::{Arc, OnceLock};

use bloq_circuit::{CoordinateOverflowError, checked_translate_coordinate};
use glam::IVec2;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeIdentifiers, NodeIndexable};

use super::{BloqEdgeRef, PathScratch, ValueInput};
use crate::{
    BloqEdge, BloqNode, BloqNodeId, BloqTemplate, BloqTemplatePool, DetectorBundle,
    DetectorBundleId, DetectorBundlePool, FxSet, LogicalInput, LogicalOutput, PipePadding,
    QuantumNode, SubGraph, TemplateId,
};

/// A portable scalar stored in [`Bloq`] producer metadata.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MetadataValue {
    /// Unsigned integer metadata.
    U64(u64),
    /// UTF-8 string metadata.
    String(String),
}

impl From<u32> for MetadataValue {
    fn from(value: u32) -> Self {
        Self::U64(u64::from(value))
    }
}

impl From<String> for MetadataValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

/// The compiled bloq circuit IR: a graph of quantum, classical, and region
/// nodes produced by `bloq_compile`, consumed by emitters and the verifier.
///
/// The top level is a [`SubGraph`] — the same graph-level type region bodies
/// use — so every read accessor ([`SubGraph::node`], [`SubGraph::nodes`],
/// [`SubGraph::incoming`], …) works uniformly at any nesting level. `Bloq`
/// delegates the common calls and adds whole-program state (templates,
/// frames, provenance) plus recursive traversal ([`Bloq::walk`],
/// [`Bloq::levels`]).
///
/// Cloning shares graph levels, quantum payloads, and templates; edits detach
/// the affected storage. Metadata and populated layout caches are still copied.
// `Clone` is derived: cloning a populated `sorted_layout_coords` cache is
// sound because the cache is a pure function of the cloned graph.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Bloq {
    templates: BloqTemplatePool,
    detector_bundles: DetectorBundlePool,
    top: SubGraph,
    /// Namespaced producer provenance. It never affects IR semantics.
    metadata: BTreeMap<String, MetadataValue>,
    /// Physical input-patch metadata needed by preparation hooks.
    /// Empty for hand-built programs and inputs without a complete pair.
    #[serde(default)]
    pub(super) logical_inputs: Vec<LogicalInput>,
    /// Terminal logical operators needed by output-state execution hooks.
    /// Empty for hand-built programs and outputs without a complete pair.
    pub(super) logical_outputs: Vec<LogicalOutput>,
    /// Memo for [`Bloq::insert_memory_rounds`]'s padding-template
    /// specialization: [`PaddingVariant`] -> specialized template, so padding
    /// many seams at one wait duration shares one pool entry. A
    /// rebuild-on-demand cache, never serialized. Pool ids are append-only;
    /// the template-retuning edit clears this memo when circuit contents
    /// change.
    ///
    /// [`PaddingVariant`]: crate::edit::PaddingVariant
    #[serde(skip)]
    pub(crate) specialized_padding: crate::FxMap<crate::edit::PaddingVariant, TemplateId>,
    /// Derived cache, rebuilt on demand — never serialized.
    #[serde(skip)]
    sorted_layout_coords: OnceLock<Result<Vec<IVec2>, CoordinateOverflowError>>,
}

impl Bloq {
    /// Create an empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Assemble a program from parsed parts (the text codec's constructor).
    pub(crate) fn from_parts(
        templates: BloqTemplatePool,
        detector_bundles: DetectorBundlePool,
        top: SubGraph,
        metadata: BTreeMap<String, MetadataValue>,
        logical_inputs: Vec<LogicalInput>,
        logical_outputs: Vec<LogicalOutput>,
    ) -> Self {
        Self {
            templates,
            detector_bundles,
            top,
            metadata,
            logical_inputs,
            logical_outputs,
            specialized_padding: crate::FxMap::default(),
            sorted_layout_coords: OnceLock::new(),
        }
    }

    /// The top graph level. Region bodies are further [`SubGraph`] levels,
    /// reached through [`crate::RegionNode::bodies`] or [`Bloq::levels`].
    pub fn top(&self) -> &SubGraph {
        &self.top
    }

    /// Mutable top graph level.
    ///
    /// This invalidates derived layout data because arbitrary graph edits may
    /// change the program's qubit footprint.
    pub fn top_mut(&mut self) -> &mut SubGraph {
        self.sorted_layout_coords = OnceLock::new();
        &mut self.top
    }

    /// Producer-specific provenance, keyed by a namespaced stable name.
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    /// Insert producer metadata, returning the previous value for `key`.
    pub fn insert_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Option<MetadataValue> {
        self.metadata.insert(key.into(), value.into())
    }

    /// Every edge-owned memory-padding record in the program.
    pub fn pipe_padding(&self) -> impl Iterator<Item = &PipePadding> {
        self.levels().flat_map(|(_, level)| {
            level
                .edges()
                .flat_map(|edge| edge.edge.pipes())
                .filter_map(|seam| seam.padding.as_ref())
        })
    }

    /// Replace the padding provenance on the unique top-level quantum edge
    /// `from -> to`, one record per pipe in pipe order. Returns `false` when
    /// that edge is missing, ambiguous, or carries a different pipe count.
    ///
    /// This is public only for the compiler crate; ordinary edits consume the
    /// provenance through [`Bloq::insert_memory_rounds`].
    #[doc(hidden)]
    pub fn set_quantum_edge_padding(
        &mut self,
        from: BloqNodeId,
        to: BloqNodeId,
        padding: Vec<PipePadding>,
    ) -> bool {
        let edge_id = {
            let mut edges = self
                .graph()
                .edges_connecting(
                    NodeIndex::new(from.0 as usize),
                    NodeIndex::new(to.0 as usize),
                )
                .filter(|edge| matches!(edge.weight(), BloqEdge::Quantum(_)));
            let Some(edge) = edges.next() else {
                return false;
            };
            if edges.next().is_some() {
                return false;
            }
            edge.id()
        };
        let Some(BloqEdge::Quantum(edge)) = self.graph_mut_internal().edge_weight_mut(edge_id)
        else {
            unreachable!("selected a quantum edge")
        };
        if edge.pipes.len() != padding.len() {
            return false;
        }
        for (seam, padding) in edge.pipes.iter_mut().zip(padding) {
            seam.padding = Some(padding);
        }
        self.specialized_padding.clear();
        true
    }

    /// Add an owned template and return its id.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next template id exceeds `u32`.
    pub fn add_template(&mut self, template: BloqTemplate) -> TemplateId {
        self.sorted_layout_coords = OnceLock::new();
        self.templates.insert(template)
    }

    /// Add a shared template and return its id.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next template id exceeds `u32`.
    pub fn add_shared_template(&mut self, template: Arc<BloqTemplate>) -> TemplateId {
        self.sorted_layout_coords = OnceLock::new();
        self.templates.insert_shared(template)
    }

    /// Program template pool.
    pub fn templates(&self) -> &BloqTemplatePool {
        &self.templates
    }

    /// Reusable detector bundle pool.
    pub fn detector_bundles(&self) -> &DetectorBundlePool {
        &self.detector_bundles
    }

    /// Add owned detector rows to the shared bundle pool.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next bundle id exceeds `u32`.
    pub fn add_detector_bundle(&mut self, bundle: DetectorBundle) -> DetectorBundleId {
        self.detector_bundles.insert(bundle)
    }

    /// Import an immutable shared detector bundle.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next bundle id exceeds `u32`.
    pub fn add_shared_detector_bundle(&mut self, bundle: Arc<DetectorBundle>) -> DetectorBundleId {
        self.detector_bundles.insert_shared(bundle)
    }

    /// Mutable template pool for this crate's in-place template edits;
    /// invalidates the derived layout cache like the other mutators, since a
    /// retuned template may change the program's qubit footprint.
    pub(crate) fn templates_mut(&mut self) -> &mut BloqTemplatePool {
        self.sorted_layout_coords = OnceLock::new();
        &mut self.templates
    }

    /// Add a top-level node and return its id.
    pub fn add_node(&mut self, node: BloqNode) -> BloqNodeId {
        self.sorted_layout_coords = OnceLock::new();
        self.top.add_node(node)
    }

    /// Adds a top-level edge; panics on a stale endpoint id, like
    /// [`SubGraph::add_edge`].
    pub fn add_edge(&mut self, from: BloqNodeId, to: BloqNodeId, edge: BloqEdge) {
        self.top.add_edge(from, to, edge);
    }

    // ---- top-level mutating delegations ------------------------------------
    //
    // The read half of the mirror is generated by `delegate_to_top!` below.
    // These stay hand-written because each also has to drop the derived layout
    // cache: an edit may change the program's qubit footprint.

    /// Mutable top-level node lookup.
    pub fn node_mut(&mut self, id: BloqNodeId) -> Option<&mut BloqNode> {
        self.sorted_layout_coords = OnceLock::new();
        self.top.node_mut(id)
    }

    /// Remove a top-level node and its incident edges.
    pub fn remove_node(&mut self, id: BloqNodeId) -> Option<BloqNode> {
        self.sorted_layout_coords = OnceLock::new();
        self.top.remove_node(id)
    }

    /// The number of top-level quantum nodes. Unlike [`Self::node_count`] this
    /// excludes the classical `Observable` nodes, so it
    /// stays a stable measure of lowered blocks for perf tracking.
    pub fn quantum_node_count(&self) -> usize {
        self.quantum_nodes().count()
    }

    // -------------------------------------------------------------------------

    /// The number of distinct physical qubit coordinates in the layout.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateOverflowError`] if translating an instance qubit
    /// exceeds the coordinate lattice.
    pub fn qubit_count(&self) -> Result<usize, CoordinateOverflowError> {
        Ok(self.sorted_layout_coords()?.len())
    }

    /// The program's distinct layout qubit coordinates, sorted by `(x, y)`.
    /// Computed once and cached; the cache is cleared on any mutation.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateOverflowError`] if an instance offset would move a
    /// template qubit outside the `i32` coordinate lattice.
    pub fn sorted_layout_coords(&self) -> Result<&[IVec2], CoordinateOverflowError> {
        match self
            .sorted_layout_coords
            .get_or_init(|| self.collect_sorted_layout_coords())
        {
            Ok(coords) => Ok(coords),
            Err(error) => Err(*error),
        }
    }

    fn collect_sorted_layout_coords(&self) -> Result<Vec<IVec2>, CoordinateOverflowError> {
        let mut qubits = FxSet::default();
        for (_, node) in self.top.nodes() {
            self.extend_node_qubits(node, &mut qubits)?;
        }
        let mut coords = qubits.into_iter().collect::<Vec<_>>();
        coords.sort_by_key(|coord| (coord.x, coord.y));
        Ok(coords)
    }

    fn extend_node_qubits(
        &self,
        node: &BloqNode,
        qubits: &mut FxSet<IVec2>,
    ) -> Result<(), CoordinateOverflowError> {
        // A region's footprint is its bodies' footprint:
        // arm/body instances resolve against the same shared template pool, so
        // recurse rather than reporting a region as physically empty.
        if let Some(region) = node.try_region() {
            for (_, body) in region.bodies() {
                for (_, body_node) in body.nodes() {
                    self.extend_node_qubits(body_node, qubits)?;
                }
            }
            return Ok(());
        }
        let Some(quantum) = node.try_quantum() else {
            return Ok(()); // classical nodes carry no qubits
        };
        let mut seen: FxSet<(crate::TemplateId, IVec2)> = FxSet::default();
        for instance in &quantum.instances {
            // Instances repeat the same (template, offset) — one round per time
            // layer, sometimes interleaved across layers — and contribute
            // nothing new. Track all seen pairs so a repeat never re-walks the
            // template's qubit list, not just consecutive duplicates.
            if !seen.insert((instance.template_id, instance.offset)) {
                continue;
            }
            if let Some(template) = self.templates.get(instance.template_id) {
                let template_qubits = template.qubits();
                qubits.reserve(template_qubits.len());
                for &qubit in template_qubits {
                    qubits.insert(checked_translate_coordinate(qubit, instance.offset)?);
                }
            }
        }
        Ok(())
    }

    /// The layout qubits a node occupies: each instance's template qubits
    /// translated by the instance offset. Instances pointing at a missing
    /// template are skipped.
    ///
    /// # Errors
    ///
    /// Returns [`CoordinateOverflowError`] if an instance offset would move a
    /// template qubit outside the `i32` coordinate lattice.
    pub fn node_qubits(&self, node: &BloqNode) -> Result<FxSet<IVec2>, CoordinateOverflowError> {
        let mut qubits = FxSet::default();
        self.extend_node_qubits(node, &mut qubits)?;
        Ok(qubits)
    }

    /// The number of static measurement sites in a node and its nested bodies.
    /// RUS bodies contribute; this is not a runtime
    /// execution count.
    pub fn node_measurement_count(&self, node: &BloqNode) -> usize {
        if let Some(region) = node.try_region() {
            return region
                .bodies()
                .flat_map(|(_, body)| body.nodes())
                .map(|(_, body_node)| self.node_measurement_count(body_node))
                .sum();
        }
        let Some(quantum) = node.try_quantum() else {
            return 0; // classical nodes emit no measurements
        };
        quantum
            .instances
            .iter()
            .filter_map(|instance| self.templates.get(instance.template_id))
            .map(|template| template.circuit.num_measurements() as usize)
            .sum()
    }

    /// The number of static measurement sites in the program.
    pub fn measurement_count(&self) -> usize {
        self.top
            .nodes()
            .map(|(_, node)| self.node_measurement_count(node))
            .sum()
    }

    /// Read-only raw graph access for this crate's own passes; consumers use
    /// the id-native accessors instead (petgraph is not public API).
    pub(crate) fn graph(&self) -> &StableDiGraph<BloqNode, BloqEdge> {
        self.top.graph()
    }

    /// Mutable graph access for the edit module (`crate::edit`); invalidates
    /// the cached layout since edits may change node qubit footprints.
    pub(crate) fn graph_mut_internal(&mut self) -> &mut StableDiGraph<BloqNode, BloqEdge> {
        self.sorted_layout_coords = OnceLock::new();
        self.top.graph_mut()
    }
}

/// Mirror [`SubGraph`]'s read surface onto [`Bloq`], targeting the top level.
///
/// These are pure forwarding calls, so hand-writing them only invited drift —
/// `open_value_producers`/`open_bit_producers` had no `Bloq` twin at all until
/// this list made the mirror total. `Deref<Target = SubGraph>` would collapse
/// them further, but a `Bloq` is not a smart pointer to its top level
/// (C-DEREF), and `DerefMut` would hand out `&mut SubGraph` implicitly at every
/// mutating call site, hiding the layout-cache contract the mutators above
/// spell out.
macro_rules! delegate_to_top {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty;)*) => {
        impl Bloq {
            $(
                #[doc = concat!(
                    "The top level's [`SubGraph::", stringify!($name), "`]. Region body \
                     levels are reached through [`Bloq::levels`] / [`Bloq::level_at`]."
                )]
                pub fn $name(&self $(, $arg: $ty)*) -> $ret {
                    self.top.$name($($arg),*)
                }
            )*
        }
    };
}

delegate_to_top! {
    fn node(id: BloqNodeId) -> Option<&BloqNode>;
    fn node_ids() -> impl DoubleEndedIterator<Item = BloqNodeId> + '_;
    fn nodes() -> impl DoubleEndedIterator<Item = (BloqNodeId, &BloqNode)> + '_;
    fn quantum_nodes() -> impl Iterator<Item = (BloqNodeId, &QuantumNode)> + '_;
    fn node_count() -> usize;
    fn edge_count() -> usize;
    fn edges() -> impl Iterator<Item = BloqEdgeRef<'_>> + '_;
    fn incoming(id: BloqNodeId) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_;
    fn outgoing(id: BloqNodeId) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_;
    fn edges_between(from: BloqNodeId, to: BloqNodeId) -> impl Iterator<Item = BloqEdgeRef<'_>> + '_;
    fn has_path(from: BloqNodeId, to: BloqNodeId) -> bool;
    fn path_scratch() -> PathScratch;
    fn has_path_with_scratch(from: BloqNodeId, to: BloqNodeId, scratch: &mut PathScratch) -> bool;
    fn has_path_backwards_with_scratch(from: BloqNodeId, to: BloqNodeId, scratch: &mut PathScratch) -> bool;
    fn value_inputs(id: BloqNodeId) -> impl Iterator<Item = ValueInput<'_>> + '_;
    fn data_inputs(id: BloqNodeId) -> impl Iterator<Item = ValueInput<'_>> + '_;
    fn data_consumers(id: BloqNodeId) -> impl Iterator<Item = (BloqNodeId, u32)> + '_;
    fn compose_consumers(id: BloqNodeId) -> impl Iterator<Item = (BloqNodeId, u32)> + '_ ;
    fn value_consumers(id: BloqNodeId) -> impl Iterator<Item = (BloqNodeId, u32)> + '_;
    fn open_value_producers() -> impl Iterator<Item = BloqNodeId> + '_;
    fn open_bit_producers() -> impl Iterator<Item = BloqNodeId> + '_;
    fn value_output() -> Option<crate::ValueRef>;
    fn boundary_outputs() -> &[BloqNodeId];
    fn is_acyclic() -> bool;
    fn is_acyclic_with_edges(extra: &[(BloqNodeId, BloqNodeId)]) -> bool;
}

impl Bloq {
    /// The top level's deterministic topological emission order.
    ///
    /// # Errors
    ///
    /// Returns [`CycleDetected`] if the top-level graph contains a cycle.
    pub fn deterministic_emit_order(&self) -> Result<Vec<BloqNodeId>, CycleDetected> {
        self.top.deterministic_emit_order()
    }
}

/// The panicking twin of [`Bloq::node`] (see [`SubGraph`]'s `Index` impl).
impl Index<BloqNodeId> for Bloq {
    type Output = BloqNode;

    fn index(&self, id: BloqNodeId) -> &Self::Output {
        &self.top[id]
    }
}

/// The graph contains a cycle, so it has no topological emit order.
///
/// Named `CycleDetected` (not `CyclicGraph`) to avoid colliding with the
/// `CyclicGraph` *variants* of [`BloqValidationError`](crate::BloqValidationError) and `ExecuteError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("bloq graph contains a cycle")]
pub struct CycleDetected;

/// Node-major CSR adjacency for one graph level.
///
/// petgraph threads each node's edges through a linked list inside the edge
/// storage, so a Kahn walk pointer-chases ~E scattered words — at millions of
/// edges that dominates the algorithm. Building CSR costs two sequential edge
/// passes plus one E-sized `u32` array (and the offsets array), and the walk
/// then reads each node's neighbors sequentially.
struct CsrAdjacency {
    /// `offsets[node]..offsets[node + 1]` is `node`'s slice in `targets`.
    offsets: Vec<u32>,
    targets: Vec<u32>,
}

impl CsrAdjacency {
    fn of(graph: &StableDiGraph<BloqNode, BloqEdge>) -> Self {
        let bound = graph.node_bound();
        let mut offsets = vec![0u32; bound + 1];
        for edge in graph.edge_references() {
            offsets[edge.source().index() + 1] += 1;
        }
        for slot in 1..=bound {
            offsets[slot] += offsets[slot - 1];
        }
        // Fill each node's slice from its end, decrementing the end offsets —
        // no separate cursor array. Afterwards `offsets[i + 1]` is node `i`'s
        // start, so rotating one slot left restores the invariant. The fill
        // reverses the within-slice edge order, which no consumer of this
        // structure observes.
        let total = graph.edge_count() as u32;
        let mut targets = vec![0u32; graph.edge_count()];
        for edge in graph.edge_references() {
            let from = edge.source().index();
            offsets[from + 1] -= 1;
            targets[offsets[from + 1] as usize] = edge.target().index() as u32;
        }
        offsets.remove(0);
        offsets.push(total);
        Self { offsets, targets }
    }

    fn slice(&self, node: usize) -> &[u32] {
        &self.targets[self.offsets[node] as usize..self.offsets[node + 1] as usize]
    }
}

/// The deterministic Kahn topological order of any [`BloqNode`]/[`BloqEdge`]
/// graph (a [`Bloq`] or a region [`crate::SubGraph`] body). All edge kinds count
/// equally toward indegree; determinism comes entirely from the min-heap keyed on
/// node index — indices are unique, so ties never arise. Returns [`CycleDetected`]
/// when a cycle leaves some node permanently un-emitted.
///
/// The walk reads a transient CSR copy of the adjacency (`CsrAdjacency`),
/// roughly one `u32` per edge plus one per node slot, so neighbor scans are
/// sequential instead of chasing petgraph's edge lists.
pub(crate) fn deterministic_emit_order_of(
    graph: &StableDiGraph<BloqNode, BloqEdge>,
) -> Result<Vec<BloqNodeId>, CycleDetected> {
    let adjacency = CsrAdjacency::of(graph);
    // A node's indegree is bounded by the live edge count, which cannot
    // approach u32::MAX within addressable memory.
    let mut indegree = vec![0u32; graph.node_bound()];
    for &target in &adjacency.targets {
        indegree[target as usize] += 1;
    }

    let mut ready = BinaryHeap::new();
    ready.extend(
        graph
            .node_identifiers()
            .filter(|node| indegree[node.index()] == 0)
            .map(|node| Reverse(node.index() as u32)),
    );

    let mut order = Vec::with_capacity(graph.node_count());
    while let Some(Reverse(node)) = ready.pop() {
        order.push(BloqNodeId(node));
        // Determinism is entirely the min-heap's: node indices are unique (no
        // ties) and each node is pushed exactly once, so push order cannot affect
        // pop order.
        for &target in adjacency.slice(node as usize) {
            let target = target as usize;
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(Reverse(target as u32));
            }
        }
    }

    if order.len() != graph.node_count() {
        return Err(CycleDetected);
    }
    Ok(order)
}

/// Whether the graph — plus `extra` directed edges (duplicates and parallel
/// edges are allowed; extras naming vacant node slots are ignored) — is
/// acyclic: the deterministic emit order's success condition, without
/// materializing the order. Plain-stack Kahn over CSR adjacency, O(V+E) with
/// no order or graph copy.
pub(crate) fn is_acyclic_of(
    graph: &StableDiGraph<BloqNode, BloqEdge>,
    extra: &[(BloqNodeId, BloqNodeId)],
) -> bool {
    let adjacency = CsrAdjacency::of(graph);
    let mut indegree = vec![0u32; graph.node_bound()];
    for &target in &adjacency.targets {
        indegree[target as usize] += 1;
    }

    // Extra edges grouped by source in a sorted CSR (extras are far fewer
    // than graph edges). `source_marks` is a bitset over node slots so the
    // per-pop probe stays O(1) for the overwhelmingly common non-source node.
    let mut extras: Vec<(u32, u32)> = extra
        .iter()
        .filter(|&(from, to)| {
            graph.contains_node(NodeIndex::new(from.0 as usize))
                && graph.contains_node(NodeIndex::new(to.0 as usize))
        })
        .map(|&(from, to)| (from.0, to.0))
        .collect();
    extras.sort_unstable();
    for &(_, to) in &extras {
        indegree[to as usize] += 1;
    }
    let mut sources = Vec::new();
    let mut starts = Vec::new();
    let mut source_marks = vec![0u64; graph.node_bound().div_ceil(u64::BITS as usize)];
    for (index, &(from, _)) in extras.iter().enumerate() {
        if sources.last() != Some(&from) {
            sources.push(from);
            starts.push(index);
            source_marks[from as usize >> 6] |= 1 << (from & 63);
        }
    }

    let mut stack: Vec<u32> = graph
        .node_identifiers()
        .filter(|node| indegree[node.index()] == 0)
        .map(|node| node.index() as u32)
        .collect();
    let mut visited = 0usize;
    while let Some(node) = stack.pop() {
        visited += 1;
        for &target in adjacency.slice(node as usize) {
            let target = target as usize;
            indegree[target] -= 1;
            if indegree[target] == 0 {
                stack.push(target as u32);
            }
        }
        if source_marks[node as usize >> 6] >> (node & 63) & 1 == 0 {
            continue;
        }
        let group = sources
            .binary_search(&node)
            .expect("marked nodes are exactly the extra-edge sources");
        let start = starts[group];
        let end = starts.get(group + 1).copied().unwrap_or(extras.len());
        for &(_, to) in &extras[start..end] {
            let to = to as usize;
            indegree[to] -= 1;
            if indegree[to] == 0 {
                stack.push(to as u32);
            }
        }
    }

    visited == graph.node_count()
}

#[cfg(test)]
mod tests {
    use bloq_circuit::{CoordCircuit, PauliBasis};
    use glam::ivec2;

    use super::*;
    use crate::{TemplateInstance, TemplateInstanceId};

    #[test]
    fn data_inputs_exclude_activation_at_the_top_level() {
        use crate::{ClassicalExpr, ClassicalNode};

        let mut program = Bloq::new();
        let value = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        }));
        let activation = program.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let mut consumer = BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::In(0),
        });
        consumer.activation = Some(1);
        let consumer = program.add_node(consumer);
        program.add_edge(value, consumer, BloqEdge::value(0));
        program.add_edge(activation, consumer, BloqEdge::value(1));

        assert_eq!(program.value_inputs(consumer).count(), 2);
        assert_eq!(
            program
                .data_inputs(consumer)
                .map(|input| (input.slot, input.producer))
                .collect::<Vec<_>>(),
            [(0, value)]
        );
    }

    /// The mirror's late arrivals: both open-producer queries used to exist
    /// only on `SubGraph`, so a caller holding a `Bloq` had to route through
    /// `top()` for them and through `Bloq` for everything else.
    #[test]
    fn open_producer_queries_mirror_the_top_level() {
        let program = Bloq::from_text(
            "\
BLOQIR 1

template t0 {
  circuit {
    MPP X(0,0):m0
  }
}

graph {
  n0 quantum {
    instance i0 t0 @ (0,0)
  }
  n1 observable fragment measurements i0:m0 from generator 0
  n2 observable fragment operators i0 output X(0,0)
  n0 -> n1 order
}
",
        )
        .expect("valid .bloqir text");

        assert_eq!(
            program.open_value_producers().collect::<Vec<_>>(),
            [BloqNodeId(1), BloqNodeId(2)]
        );
        // Every fragment exposes its assembled parity, including empty parity.
        assert_eq!(
            program.open_bit_producers().collect::<Vec<_>>(),
            [BloqNodeId(1), BloqNodeId(2)]
        );
    }

    #[test]
    fn graph_mutators_invalidate_layout_cache() {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);

        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let make_node = |instance, offset| {
            let mut node = BloqNode::from_members(vec![]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(instance),
                    template,
                    offset,
                ));
            node
        };
        let first = bloq.add_node(make_node(0, ivec2(0, 0)));
        let second = bloq.add_node(make_node(1, ivec2(10, 0)));

        assert_eq!(
            bloq.sorted_layout_coords().unwrap(),
            [ivec2(0, 0), ivec2(10, 0)]
        );
        bloq.node_mut(first).unwrap().expect_quantum_mut().instances[0].offset = ivec2(5, 0);
        assert_eq!(
            bloq.sorted_layout_coords().unwrap(),
            [ivec2(5, 0), ivec2(10, 0)]
        );

        bloq.remove_node(second).unwrap();
        assert_eq!(bloq.sorted_layout_coords().unwrap(), [ivec2(5, 0)]);

        bloq.top_mut().add_node(make_node(2, ivec2(20, 0)));
        assert_eq!(
            bloq.sorted_layout_coords().unwrap(),
            [ivec2(5, 0), ivec2(20, 0)]
        );
    }

    #[test]
    fn declared_outputs_survive_codecs_rewrites_and_snapshot_edits() {
        let original = Bloq::from_text(
            "BLOQIR 1
graph {
 n0 compute 1
 n1 compute !in0
 n2 observable fragment operators
 n0 -> n1 value 0
 result n0
 bindings n2
}",
        )
        .unwrap();
        original.validate().unwrap();
        for mut program in [
            original.clone(),
            Bloq::from_text(&original.to_text()).unwrap(),
            Bloq::from_binary(&original.to_binary()).unwrap(),
            original.pin_membership(&Default::default()).unwrap(),
        ] {
            program.optimize().expect("acyclic test program");
            program.validate().unwrap();
            assert_eq!(program.value_output(), Some(BloqNodeId(0).into()));
            assert_eq!(program.boundary_outputs(), [BloqNodeId(2)]);
            assert!(
                program
                    .resolve_classical(
                        program.value_output().unwrap().node,
                        crate::ClassicalAssignment::Uniform(false),
                    )
                    .unwrap()
                    .sign
            );
        }

        let mut edited = original.clone();
        edited
            .top_mut()
            .set_value_output(Some(BloqNodeId(1).into()));
        edited.top_mut().set_boundary_outputs(Vec::new());
        assert_eq!(original.value_output(), Some(BloqNodeId(0).into()));
        assert_eq!(original.boundary_outputs(), [BloqNodeId(2)]);
        let removed = edited.remove_node(BloqNodeId(1)).unwrap();
        assert_eq!(edited.value_output(), None);
        assert_eq!(edited.add_node(removed), BloqNodeId(1));
        assert_eq!(
            edited.value_output(),
            None,
            "reused ids do not retarget results"
        );
        edited.top_mut().set_boundary_outputs(vec![BloqNodeId(2)]);
        let removed = edited.remove_node(BloqNodeId(2)).unwrap();
        assert!(edited.boundary_outputs().is_empty());
        assert_eq!(edited.add_node(removed), BloqNodeId(2));
        assert!(edited.boundary_outputs().is_empty());
    }

    #[test]
    fn snapshots_share_graph_levels_until_edited() {
        use crate::{BloqNodeKind, ClassicalExpr, ClassicalNode, RegionNode, ValueRole};

        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [ivec2(0, 0)]);
        let mut original = Bloq::new();
        let template = original.add_template(BloqTemplate::new(circuit));
        let make_body = |instance| {
            let mut body = SubGraph::new();
            let mut node = BloqNode::from_members(vec![]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(instance),
                    template,
                    ivec2(0, 0),
                ));
            body.add_node(node);
            body.add_node(BloqNode::classical(ClassicalNode::Compute {
                expr: ClassicalExpr::Const(false),
            }));
            body
        };
        let selector = original.add_node(BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(false),
        }));
        let vacant = original.add_node(original[selector].clone());
        let region = original.add_node(BloqNode::region(RegionNode::RepeatUntilSuccess {
            restart_source: None,
            restart_condition: ClassicalExpr::In(0),
            body: make_body(0),
        }));
        original.remove_node(vacant).unwrap();
        original.add_edge(
            selector,
            region,
            BloqEdge::Value {
                slot: 0,
                role: ValueRole::Data,
                output: crate::ObservableOutput::Corrected,
            },
        );
        original.validate().unwrap();
        let bytes = original.to_binary();
        assert_eq!(original.sorted_layout_coords().unwrap(), [ivec2(0, 0)]);

        let mut edited = original.clone();
        assert!(std::ptr::eq(original.graph(), edited.graph()));
        let RegionNode::RepeatUntilSuccess {
            body: original_body,
            ..
        } = original[region].try_region().unwrap();
        let BloqNodeKind::Region(RegionNode::RepeatUntilSuccess { body, .. }) =
            &mut edited.node_mut(region).unwrap().kind
        else {
            panic!("region")
        };
        assert!(std::ptr::eq(original_body.graph(), body.graph()));
        body.node_mut(BloqNodeId(1)).unwrap().kind = BloqNodeKind::Classical(
            ClassicalNode::Compute {
                expr: ClassicalExpr::Const(true),
            }
            .into(),
        );
        assert!(!std::ptr::eq(original_body.graph(), body.graph()));
        assert!(std::ptr::eq(
            original_body[BloqNodeId(0)].expect_quantum(),
            body[BloqNodeId(0)].expect_quantum(),
        ));
        let selected = body[BloqNodeId(0)]
            .select_quantum_members(|_| panic!("static nodes have no membership inputs"))
            .unwrap();
        assert!(std::ptr::eq(
            selected.expect_quantum(),
            body[BloqNodeId(0)].expect_quantum(),
        ));
        body.node_mut(BloqNodeId(0))
            .unwrap()
            .expect_quantum_mut()
            .instances[0]
            .offset = ivec2(5, 0);
        assert!(!std::ptr::eq(
            original_body[BloqNodeId(0)].expect_quantum(),
            body[BloqNodeId(0)].expect_quantum(),
        ));
        assert_eq!(selected.expect_quantum().instances[0].offset, ivec2(0, 0));
        assert!(!std::ptr::eq(original_body.graph(), body.graph()));
        assert!(!std::ptr::eq(original.graph(), edited.graph()));
        assert_eq!(original.node_ids().collect::<Vec<_>>(), [selector, region]);
        assert_eq!(edited.node_ids().collect::<Vec<_>>(), [selector, region]);
        assert!(edited.has_path(selector, region));
        assert_eq!(original.sorted_layout_coords().unwrap(), [ivec2(0, 0)]);
        assert_eq!(edited.sorted_layout_coords().unwrap(), [ivec2(5, 0)]);
        assert_eq!(original.to_binary(), bytes);
        assert_ne!(edited.to_binary(), bytes);
        edited.validate().unwrap();
        assert_eq!(Bloq::from_binary(&bytes).unwrap().to_binary(), bytes);
    }

    #[test]
    fn layout_accessors_return_coordinate_overflow_for_unvalidated_programs() {
        let coordinate = ivec2(i32::MAX, 0);
        let offset = ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [coordinate]);
        let mut bloq = Bloq::new();
        let template = bloq.add_template(BloqTemplate::new(circuit));
        let node = bloq.add_node({
            let mut node = BloqNode::from_members(vec![]);
            node.expect_quantum_mut()
                .instances
                .push(TemplateInstance::new(
                    TemplateInstanceId(0),
                    template,
                    offset,
                ));
            node
        });
        let expected = CoordinateOverflowError { coordinate, offset };
        let restored = Bloq::from_binary(&bloq.to_binary()).expect("binary shape decodes");

        for program in [&bloq, &restored] {
            assert_eq!(
                program.node_qubits(program.node(node).unwrap()),
                Err(expected)
            );
            assert_eq!(program.sorted_layout_coords(), Err(expected));
            assert_eq!(program.qubit_count(), Err(expected));
        }
    }

    // ---------------------------------------------------------------------
    // Topological order and acyclicity
    // ---------------------------------------------------------------------

    /// The pre-optimization min-heap Kahn, kept as the correctness oracle.
    fn heap_order(
        graph: &StableDiGraph<BloqNode, BloqEdge>,
    ) -> Result<Vec<BloqNodeId>, CycleDetected> {
        let mut indegree = vec![0usize; graph.node_bound()];
        for edge in graph.edge_references() {
            indegree[edge.target().index()] += 1;
        }
        let mut ready = BinaryHeap::new();
        ready.extend(
            graph
                .node_identifiers()
                .filter(|node| indegree[node.index()] == 0)
                .map(|node| Reverse(node.index())),
        );
        let mut order = Vec::with_capacity(graph.node_count());
        while let Some(Reverse(node)) = ready.pop() {
            order.push(BloqNodeId(node as u32));
            for target in
                graph.neighbors_directed(NodeIndex::new(node), petgraph::Direction::Outgoing)
            {
                let target = target.index();
                indegree[target] -= 1;
                if indegree[target] == 0 {
                    ready.push(Reverse(target));
                }
            }
        }
        if order.len() != graph.node_count() {
            return Err(CycleDetected);
        }
        Ok(order)
    }

    /// A tiny deterministic LCG — `rand` is not a bloq_ir dev-dependency.
    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        }
    }

    fn placeholder_node() -> BloqNode {
        use crate::{ClassicalExpr, ClassicalNode};
        BloqNode::classical(ClassicalNode::Compute {
            expr: ClassicalExpr::Const(true),
        })
    }

    /// Random graph with parallel edges, holes (every third node removed), and
    /// — when `cyclic` — a chain over the live nodes closed by a backward edge,
    /// so the cycle survives the removals.
    fn random_graph(
        rng: &mut impl FnMut() -> u64,
        nodes: usize,
        edges: usize,
        cyclic: bool,
    ) -> StableDiGraph<BloqNode, BloqEdge> {
        let mut graph = StableDiGraph::new();
        for _ in 0..nodes {
            graph.add_node(placeholder_node());
        }
        for _ in 0..edges {
            let from = NodeIndex::new((rng() % nodes as u64) as usize);
            let to = NodeIndex::new((rng() % nodes as u64) as usize);
            if from != to {
                graph.add_edge(from, to, BloqEdge::Order);
                if rng().is_multiple_of(4) {
                    graph.add_edge(from, to, BloqEdge::Order);
                }
            }
        }
        for slot in (0..nodes).step_by(3) {
            graph.remove_node(NodeIndex::new(slot));
        }
        let live: Vec<NodeIndex> = graph.node_identifiers().collect();
        if cyclic && live.len() > 1 {
            for pair in live.windows(2) {
                graph.add_edge(pair[0], pair[1], BloqEdge::Order);
            }
            let (first, last) = (live[0], live[live.len() - 1]);
            graph.add_edge(last, first, BloqEdge::Order);
        }
        graph
    }

    #[test]
    fn emit_order_and_acyclicity_match_the_heap_reference() {
        let mut rng = lcg(0x5EED_1234);
        for case in 0..60 {
            let nodes = 1 + (rng() % 40) as usize;
            let edges = (rng() % 80) as usize;
            let cyclic = rng().is_multiple_of(3);
            let graph = random_graph(&mut rng, nodes, edges, cyclic);
            assert_eq!(
                deterministic_emit_order_of(&graph),
                heap_order(&graph),
                "case {case}"
            );
            assert_eq!(
                is_acyclic_of(&graph, &[]),
                deterministic_emit_order_of(&graph).is_ok(),
                "case {case}"
            );
        }
    }

    #[test]
    fn is_acyclic_with_edges_matches_a_copied_graph_oracle() {
        let mut rng = lcg(0x0BAD_C0DE);
        for case in 0..60 {
            let nodes = 1 + (rng() % 30) as usize;
            let edges = (rng() % 60) as usize;
            let cyclic = rng().is_multiple_of(2);
            let graph = random_graph(&mut rng, nodes, edges, cyclic);
            let live: Vec<u32> = graph
                .node_identifiers()
                .map(|node| node.index() as u32)
                .collect();
            let extra: Vec<(BloqNodeId, BloqNodeId)> = if live.is_empty() {
                Vec::new()
            } else {
                (0..rng() % 8)
                    .map(|_| {
                        let from = live[(rng() % live.len() as u64) as usize];
                        let to = live[(rng() % live.len() as u64) as usize];
                        (BloqNodeId(from), BloqNodeId(to))
                    })
                    .collect()
            };
            let mut copy = petgraph::graph::DiGraph::<(), ()>::new();
            let slots: Vec<_> = (0..graph.node_bound()).map(|_| copy.add_node(())).collect();
            for edge in graph.edge_references() {
                copy.add_edge(
                    slots[edge.source().index()],
                    slots[edge.target().index()],
                    (),
                );
            }
            for &(from, to) in &extra {
                copy.add_edge(slots[from.0 as usize], slots[to.0 as usize], ());
            }
            assert_eq!(
                is_acyclic_of(&graph, &extra),
                !petgraph::algo::is_cyclic_directed(&copy),
                "case {case}, extra {extra:?}"
            );
        }
    }

    #[test]
    fn is_acyclic_with_edges_detects_single_and_combined_extra_cycles() {
        // 0 -> 1 -> 2, node 3 isolated.
        let mut graph = StableDiGraph::new();
        for _ in 0..4 {
            graph.add_node(placeholder_node());
        }
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), BloqEdge::Order);
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(2), BloqEdge::Order);
        let id = |slot: u32| BloqNodeId(slot);

        assert!(is_acyclic_of(&graph, &[]));
        assert!(is_acyclic_of(&graph, &[(id(2), id(3))]));
        assert!(is_acyclic_of(&graph, &[(id(0), id(2)), (id(0), id(2))]));
        assert!(!is_acyclic_of(&graph, &[(id(2), id(0))]));
        assert!(!is_acyclic_of(&graph, &[(id(1), id(1))]));
        // Neither extra closes a cycle alone; together they do.
        assert!(is_acyclic_of(&graph, &[(id(2), id(3))]));
        assert!(is_acyclic_of(&graph, &[(id(3), id(0))]));
        assert!(!is_acyclic_of(&graph, &[(id(2), id(3)), (id(3), id(0))]));
        // Extra edges with stale endpoints name no node and are ignored.
        assert_eq!(
            is_acyclic_of(&graph, &[(id(u32::MAX), id(0))]),
            is_acyclic_of(&graph, &[])
        );
    }
}
