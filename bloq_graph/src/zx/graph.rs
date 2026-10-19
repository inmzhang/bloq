//! ZX graph storage, validation, and stabilizer materialization.

use super::ZXLayerView;
use super::modular::StabilizerPhaseBasis;
use super::stabilizer::{SearchBudget, Stabilizer, StabilizerError, contracted_sign};
use crate::{
    Action, ActionDag, BlockGraphError, CubeKind, FeedbackTarget, MeasureTarget,
    MeasurementObservable, ModuleCertificationLimits, SelectiveKind,
};
use bloq_utils::{Pauli, PauliBasis, PauliString};
use glam::IVec3;
#[cfg(feature = "gltf")]
use glam::Vec3;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use strum::Display;
use thiserror::Error;

/// The kind of a [`ZXNode`]: an X/Y/Z spider or a boundary node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum NodeKind {
    /// An X spider.
    X,
    /// A Y spider.
    Y,
    /// A Z spider.
    Z,
    /// An open boundary port.
    Port,
    /// A T (magic-state) boundary node.
    T,
    /// A selective boundary node resolved at runtime.
    #[strum(to_string = "Selective({0})")]
    Selective(SelectiveKind),
}

impl NodeKind {
    /// Returns whether this is a [`Port`](NodeKind::Port).
    pub fn is_port(&self) -> bool {
        matches!(self, NodeKind::Port)
    }

    /// Returns whether this is a [`T`](NodeKind::T) node.
    pub fn is_t(&self) -> bool {
        matches!(self, NodeKind::T)
    }

    /// Returns true for all boundary node kinds: Port, T, and Selective.
    pub fn is_boundary(&self) -> bool {
        matches!(self, NodeKind::Port | NodeKind::T | NodeKind::Selective(_))
    }

    /// The Pauli that crosses an X or Z spider without being absorbed.
    pub(crate) fn cross_pauli(self) -> Pauli {
        match self {
            NodeKind::X => Pauli::X,
            NodeKind::Z => Pauli::Z,
            _ => unreachable!("only X/Z spiders have a cross Pauli"),
        }
    }

    /// Returns true for boundary nodes that can be resolved into concrete ZX spiders.
    pub(crate) fn is_fillable_boundary(&self) -> bool {
        matches!(self, NodeKind::Port | NodeKind::Selective(_))
    }
}

/// An error from constructing, validating, or filling a [`ZXGraph`].
#[derive(Debug, Clone, Error)]
#[non_exhaustive]
pub enum ZXError {
    /// An edge spans non-neighboring lattice positions.
    #[error("ZX edge endpoints are not neighboring positions: {src} -> {dst}")]
    NonNeighboringZXEdge {
        /// Edge source.
        src: IVec3,
        /// Edge target.
        dst: IVec3,
    },
    /// A node forms a non-planar corner.
    #[error("ZX graph has a 3D corner at {pos}")]
    ThreeDimensionalCorner {
        /// Corner position.
        pos: IVec3,
    },
    /// A special node does not have degree one.
    #[error("ZX node at {pos} of kind {kind} must be dangling, got degree {degree}")]
    SpecialNodeNotDangling {
        /// Node position.
        pos: IVec3,
        /// Node kind.
        kind: NodeKind,
        /// Observed degree.
        degree: usize,
    },
    /// A special node is not connected along time.
    #[error("ZX node at {pos} of kind {kind} must connect along the time axis")]
    SpecialNodeNotTimeLike {
        /// Node position.
        pos: IVec3,
        /// Node kind.
        kind: NodeKind,
    },
    /// A T node does not point toward the future.
    #[error("T node at {pos} must connect to the future time direction")]
    TNodeNotFutureDirected {
        /// T-node position.
        pos: IVec3,
    },
    /// Cube-kind inference produced conflicting values.
    #[error("conflicting inferred cube kinds at {pos}: existing {existing}, inferred {inferred}")]
    ConflictingCubeKinds {
        /// Node position.
        pos: IVec3,
        /// Existing cube kind.
        existing: CubeKind,
        /// Newly inferred cube kind.
        inferred: CubeKind,
    },
    /// Cube-kind inference failed.
    #[error("cannot infer a cube kind for ZX node at {pos}")]
    InvalidInferredCubeKind {
        /// Node position.
        pos: IVec3,
        /// Underlying block error.
        #[source]
        source: crate::BlockError,
    },
    /// A fill was requested on a non-port node.
    #[error("cannot fill non-port node at {pos} of kind {kind}")]
    FillNonPort {
        /// Node position.
        pos: IVec3,
        /// Requested fill kind.
        kind: NodeKind,
    },
    /// A port was filled with another port.
    #[error("cannot fill port at {pos} with another port kind {kind}")]
    FillPortWithPort {
        /// Port position.
        pos: IVec3,
        /// Requested fill kind.
        kind: NodeKind,
    },
    /// No port exists at the requested position.
    #[error("no port at {0} to fill")]
    NoPort(IVec3),
    /// A measurement names an invalid ZX site.
    #[error("measurement {name:?} target {target:?} is not a valid ZX measurement site")]
    InvalidMeasurementTarget {
        /// Measurement name.
        name: String,
        /// Invalid target.
        target: MeasureTarget,
    },
    /// A measurement target has no observable.
    #[error("measurement {name:?} target {target:?} is missing an observable")]
    MissingMeasurementObservable {
        /// Measurement name.
        name: String,
        /// Target lacking an observable.
        target: MeasureTarget,
    },
    /// A measurement observable conflicts with its target.
    #[error("measurement {name:?} target {target:?} has incompatible observable {observable:?}")]
    IncompatibleMeasurementObservable {
        /// Measurement name.
        name: String,
        /// Measurement target.
        target: MeasureTarget,
        /// Incompatible observable.
        observable: MeasurementObservable,
    },
    /// A resolve target is not selective.
    #[error("resolve target at {pos} is not a selective ZX node")]
    InvalidResolveTarget {
        /// Invalid target position.
        pos: IVec3,
    },
    /// Source block graph is invalid.
    #[error("{0}")]
    Graph(#[source] Box<BlockGraphError>),
}

/// A node in a [`ZXGraph`]: a spider or boundary at a lattice position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZXNode {
    /// Stable node id, used to index neighbor and edge tables.
    pub id: usize,
    /// Lattice position of the node.
    pub pos: IVec3,
    /// Node kind (spider color or boundary).
    pub kind: NodeKind,
    /// Explicit source Port direction, or `Auto` for non-Ports and temporal
    /// Ports whose direction follows geometry.
    pub role: crate::PortRole,
}

impl ZXNode {
    /// Create a node with automatic port role.
    pub const fn new(id: usize, pos: IVec3, kind: NodeKind) -> Self {
        Self {
            id,
            pos,
            kind,
            role: crate::PortRole::Auto,
        }
    }

    /// Set the source port role.
    pub const fn with_role(mut self, role: crate::PortRole) -> Self {
        self.role = role;
        self
    }

    /// Returns whether this Port participates in the output interface.
    pub fn is_output_port(&self, graph: &ZXGraph) -> bool {
        if self.kind != NodeKind::Port {
            return false;
        }
        match self.role {
            crate::PortRole::Auto => graph.neighbors(self.id).is_some_and(|neighbors| {
                !neighbors.is_empty()
                    && neighbors.iter().all(|&n| graph.nodes[n].pos.z < self.pos.z)
            }),
            role => role.has_output_boundary(),
        }
    }

    /// Returns whether this Port participates in the input interface.
    pub fn is_input_port(&self, graph: &ZXGraph) -> bool {
        if self.kind != NodeKind::Port {
            return false;
        }
        match self.role {
            crate::PortRole::Auto => graph.neighbors(self.id).is_some_and(|neighbors| {
                !neighbors.is_empty()
                    && neighbors.iter().all(|&n| graph.nodes[n].pos.z > self.pos.z)
            }),
            role => role.has_input_boundary(),
        }
    }
}

/// An edge in a [`ZXGraph`] connecting two nodes, optionally Hadamard-typed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZXEdge {
    /// First endpoint node id.
    pub n1: usize,
    /// Second endpoint node id.
    pub n2: usize,
    /// Stable edge id.
    pub id: usize,
    /// Whether the edge carries a Hadamard.
    pub hadamard: bool,
}

impl ZXEdge {
    fn sorted_nodes(&self) -> (usize, usize) {
        if self.n1 < self.n2 {
            (self.n1, self.n2)
        } else {
            (self.n2, self.n1)
        }
    }
}

/// Dense node adjacency paired with each directed edge-column id.
#[derive(Debug, Clone)]
pub(super) struct CsrAdjacency {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    edge_ids: Vec<usize>,
    has_parallel_edges: bool,
}

impl CsrAdjacency {
    pub(super) fn from_edges(node_count: usize, edges: &[(usize, usize, usize)]) -> Self {
        let mut offsets = vec![0; node_count + 1];
        let mut last_edge_ids = FxHashMap::default();
        for &(source, target, edge_id) in edges {
            offsets[source + 1] += 1;
            last_edge_ids.insert((source, target), edge_id);
        }
        for node in 0..node_count {
            offsets[node + 1] += offsets[node];
        }

        let mut neighbors = vec![0; edges.len()];
        let mut edge_ids = vec![0; edges.len()];
        let mut next = offsets[..node_count].to_vec();
        for &(source, target, _) in edges {
            let index = next[source];
            neighbors[index] = target;
            // Preserve the retired `nn2e` map's parallel-edge contract: every
            // duplicate neighbor entry resolves to the last edge for that pair.
            edge_ids[index] = last_edge_ids[&(source, target)];
            next[source] += 1;
        }
        Self {
            offsets,
            neighbors,
            edge_ids,
            has_parallel_edges: edges.len() != last_edge_ids.len(),
        }
    }

    pub(super) fn has_parallel_edges(&self) -> bool {
        self.has_parallel_edges
    }

    fn range(&self, node: usize) -> Option<Range<usize>> {
        let end = node.checked_add(1)?;
        Some(*self.offsets.get(node)?..*self.offsets.get(end)?)
    }

    fn neighbors(&self, node: usize) -> Option<&[usize]> {
        Some(&self.neighbors[self.range(node)?])
    }

    fn edge_id(&self, source: usize, target: usize) -> Option<usize> {
        let range = self.range(source)?;
        self.neighbors[range.clone()]
            .iter()
            .rposition(|&neighbor| neighbor == target)
            .map(|index| self.edge_ids[range.start + index])
    }

    fn neighbor_edges(&self, node: usize) -> Option<impl Iterator<Item = (usize, usize)> + '_> {
        let range = self.range(node)?;
        Some(
            self.neighbors[range.clone()]
                .iter()
                .copied()
                .zip(self.edge_ids[range].iter().copied()),
        )
    }
}

/// A ZX-calculus graph derived from a [`BlockGraph`](crate::BlockGraph).
///
/// Nodes and edges use dense integer ids; the graph carries the source action
/// DAG so measurement and resolve sites can be located by position. It is the
/// input to stabilizer computation and output-correction analysis.
#[derive(Debug, Clone)]
pub struct ZXGraph {
    pub(crate) nodes: Vec<ZXNode>,
    pub(crate) edges: Vec<ZXEdge>,
    pub(super) adjacency: CsrAdjacency,
    /// Block-center position to node id, for O(1) [`ZXGraph::node_at`].
    /// Endpoint offsets of multi-cell blocks are deliberately absent.
    pub(crate) pos_to_node: FxHashMap<IVec3, usize>,
    pub(crate) action_graph: ActionDag,
    pub(crate) total_ids: usize,
    pub(crate) cross_incident: OnceLock<Arc<[CrossIncident]>>,
    /// Composed sign reference or parallel-edge fallback, shared through X/Z fills.
    pub(crate) stabilizer_phase_basis: OnceLock<Arc<StabilizerPhaseBasis>>,
}

pub(crate) type CrossIncident = SmallVec<[(usize, bool); 6]>;

impl ZXGraph {
    /// Returns all nodes.
    pub fn nodes(&self) -> &[ZXNode] {
        &self.nodes
    }

    /// Returns all edges.
    pub fn edges(&self) -> &[ZXEdge] {
        &self.edges
    }

    /// Returns the neighbor node ids of `node_id`, if the node exists.
    pub fn neighbors(&self, node_id: usize) -> Option<&[usize]> {
        self.adjacency.neighbors(node_id)
    }

    pub(super) fn neighbor_ids(&self, node_id: usize) -> &[usize] {
        self.neighbors(node_id).expect("ZX node ids are dense")
    }

    /// Returns the id of the edge between nodes `n1` and `n2`, if any.
    pub fn edge_id(&self, n1: usize, n2: usize) -> Option<usize> {
        self.adjacency.edge_id(n1, n2)
    }

    pub(super) fn neighbor_edges(
        &self,
        node_id: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.adjacency
            .neighbor_edges(node_id)
            .expect("ZX node ids are dense")
    }

    pub(super) fn edge_by_id(&self, edge_id: usize) -> &ZXEdge {
        &self.edges[edge_id - self.nodes.len()]
    }

    /// Returns the total number of node and edge ids, which share one id space.
    pub fn total_ids(&self) -> usize {
        self.total_ids
    }

    /// Returns the node at `pos`, if any.
    pub fn node_at(&self, pos: IVec3) -> Option<&ZXNode> {
        self.pos_to_node.get(&pos).map(|&id| &self.nodes[id])
    }

    /// Returns the sorted, deduplicated set of occupied time layers.
    pub fn z_layers(&self) -> Vec<i32> {
        let mut zs = self.nodes.iter().map(|node| node.pos.z).collect::<Vec<_>>();
        zs.sort_unstable();
        zs.dedup();
        zs
    }

    /// Returns a read-only view of the time layer at `z`.
    pub fn layer(&self, z: i32) -> ZXLayerView {
        ZXLayerView::new(self, z)
    }

    /// Returns the action DAG carried over from the source graph.
    pub fn action_graph(&self) -> &ActionDag {
        &self.action_graph
    }

    /// Returns the actions in ordinal order.
    pub fn actions(&self) -> Vec<Action> {
        self.action_graph
            .ordered_nodes()
            .map(|node| node.action.clone())
            .collect()
    }

    /// Returns the stabilizer-table column (node or edge id) that a measurement
    /// `target` observes, if it maps to a graph element.
    pub fn measurement_column(&self, target: &MeasureTarget) -> Option<usize> {
        match target {
            MeasureTarget::Node(pos) => self.node_at(*pos).map(|node| node.id),
            MeasureTarget::Edge { src, dir } => {
                let dst = crate::checked_add_position(*src, dir.to_ivec3()).ok()?;
                let src_id = self.node_at(*src)?.id;
                let dst_id = self.node_at(dst)?.id;
                self.edge_id(src_id, dst_id)
            }
        }
    }

    /// The wire carrying a source Pauli action, oriented away from its target.
    /// Node feedback uses the first outgoing temporal wire, then the last
    /// incoming temporal wire, then a sole spatial wire. Ordering is by source
    /// position, so local and full graphs make the same choice (SEM-READ).
    pub(crate) fn feedback_edge(&self, target: &FeedbackTarget) -> Option<&ZXEdge> {
        let node = self.node_at(target.target)?;
        let neighbor = if let Some(direction) = target.direction {
            let position = crate::checked_add_position(target.target, direction.to_ivec3()).ok()?;
            self.node_at(position)?.id
        } else {
            let neighbors = self.neighbor_ids(node.id);
            neighbors
                .iter()
                .copied()
                .filter(|&neighbor| self.nodes[neighbor].pos.z > node.pos.z)
                .min_by_key(|&neighbor| self.nodes[neighbor].pos.to_array())
                .or_else(|| {
                    neighbors
                        .iter()
                        .copied()
                        .filter(|&neighbor| self.nodes[neighbor].pos.z < node.pos.z)
                        .max_by_key(|&neighbor| self.nodes[neighbor].pos.to_array())
                })
                .or_else(|| (neighbors.len() == 1).then(|| neighbors[0]))?
        };
        self.edge_between(node.id, neighbor)
    }

    /// A feedback wire and Pauli in the external table's smaller-endpoint frame.
    pub(crate) fn feedback_column(&self, target: &FeedbackTarget) -> Option<(usize, Pauli)> {
        let edge = self.feedback_edge(target)?;
        Some((
            edge.id,
            pauli_at_small_node(edge, edge.n1, Pauli::from(target.pauli)),
        ))
    }

    /// Returns a map from each measurement variable name to its column id.
    pub fn measurement_columns(&self) -> HashMap<String, usize> {
        let mut cols = HashMap::new();

        for node in self.action_graph.ordered_nodes() {
            let Action::Measure { name, target } = &node.action else {
                continue;
            };
            let Some(col) = self.measurement_column(target) else {
                continue;
            };
            cols.insert(name.clone(), col);
        }

        cols
    }

    /// Returns whether the graph is Clifford (contains no T nodes).
    pub fn is_clifford_computation(&self) -> bool {
        self.nodes.iter().all(|node| !node.kind.is_t())
    }

    /// Returns the edge between nodes `n1` and `n2`, if any.
    pub fn edge_between(&self, n1: usize, n2: usize) -> Option<&ZXEdge> {
        self.edge_id(n1, n2)
            .and_then(|edge_id| self.edges.get(edge_id - self.nodes.len()))
    }

    /// Returns the sorted output-port positions.
    pub fn output_ports(&self) -> Vec<glam::IVec3> {
        let mut result: Vec<glam::IVec3> = self
            .nodes
            .iter()
            .filter(|node| node.is_output_port(self))
            .map(|node| node.pos)
            .collect();
        result.sort_by_key(IVec3::to_array);
        result.dedup();
        result
    }

    /// Validates that every measurement and resolve action targets a legal ZX
    /// site with a resolved observable, as required before program lowering.
    ///
    /// # Errors
    ///
    /// Returns a [`ZXError`] describing the first invalid measurement or resolve
    /// action.
    pub fn validate_for_program(&self) -> Result<(), ZXError> {
        for action in self.action_graph.ordered_nodes() {
            match &action.action {
                Action::Measure { target, name } => {
                    let Some(observable) = action.measurement else {
                        return Err(ZXError::MissingMeasurementObservable {
                            name: name.clone(),
                            target: *target,
                        });
                    };
                    let Some(node_id) = self.measurement_column(target) else {
                        return Err(ZXError::InvalidMeasurementTarget {
                            name: name.clone(),
                            target: *target,
                        });
                    };
                    match target {
                        MeasureTarget::Node(_) => {
                            let node = &self.nodes[node_id];
                            let compatible = match node.kind {
                                NodeKind::Selective(kind) => {
                                    observable == MeasurementObservable::Selective(kind)
                                }
                                NodeKind::X | NodeKind::Z
                                    if self.neighbor_ids(node.id).len() >= 2 =>
                                {
                                    matches!(
                                        observable,
                                        MeasurementObservable::Concrete(
                                            PauliBasis::X | PauliBasis::Z
                                        )
                                    )
                                }
                                _ => measurement_observable_for_node_kind(node.kind)
                                    .is_some_and(|expected| expected == observable),
                            };
                            if !compatible {
                                return Err(ZXError::IncompatibleMeasurementObservable {
                                    name: name.clone(),
                                    target: *target,
                                    observable,
                                });
                            }
                        }
                        MeasureTarget::Edge { .. } => {
                            if !matches!(
                                observable,
                                MeasurementObservable::Concrete(PauliBasis::X)
                                    | MeasurementObservable::Concrete(PauliBasis::Z)
                            ) {
                                return Err(ZXError::IncompatibleMeasurementObservable {
                                    name: name.clone(),
                                    target: *target,
                                    observable,
                                });
                            }
                        }
                    }
                }
                Action::Resolve { target, .. } => {
                    let Some(node) = self.node_at(*target) else {
                        return Err(ZXError::InvalidResolveTarget { pos: *target });
                    };
                    let NodeKind::Selective(_) = node.kind else {
                        return Err(ZXError::InvalidResolveTarget { pos: *target });
                    };
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Lookup twin of [`Self::edge_between`] for node pairs the caller already
    /// knows are connected (e.g. endpoints read off an existing edge).
    ///
    /// # Panics
    ///
    /// Panics when no edge connects `n1` and `n2`.
    pub(super) fn get_edge(&self, n1: usize, n2: usize) -> &ZXEdge {
        self.edge_between(n1, n2)
            .expect("callers pass node pairs read off an existing edge")
    }

    pub(super) fn edge_column_pair(&self, edge: &ZXEdge) -> (usize, usize) {
        (
            self.edge_id(edge.n1, edge.n2)
                .expect("edge endpoints came from the graph"),
            self.edge_id(edge.n2, edge.n1)
                .expect("edge endpoints came from the graph"),
        )
    }

    #[cfg(feature = "gltf")]
    pub(crate) fn sink_position(&self, sink: usize) -> Option<Vec3> {
        if sink < self.nodes.len() {
            return Some(self.nodes[sink].pos.as_vec3());
        }
        let edge_idx = sink.checked_sub(self.nodes.len())?;
        let edge = self.edges.get(edge_idx)?;
        debug_assert_eq!(edge.id, sink);
        let p1 = self.nodes[edge.n1].pos.as_vec3();
        let p2 = self.nodes[edge.n2].pos.as_vec3();
        Some((p1 + p2) / 2.0)
    }

    pub(crate) fn fill_ports(
        &self,
        port_map: &HashMap<IVec3, NodeKind>,
    ) -> Result<ZXGraph, ZXError> {
        let mut new_zx = self.clone();
        // X/Z fills restrict the existing signed relation to I/the chosen
        // axis. Every surviving raw row retains its phase in that relation.
        if port_map
            .values()
            .any(|kind| !matches!(kind, NodeKind::X | NodeKind::Z))
        {
            new_zx.stabilizer_phase_basis.take();
        }
        for (&pos, kind) in port_map {
            let Some(node_id) = self.pos_to_node.get(&pos).copied() else {
                return Err(ZXError::NoPort(pos));
            };
            let node = self.nodes[node_id];
            if !node.kind.is_fillable_boundary() {
                return Err(ZXError::FillNonPort {
                    pos,
                    kind: node.kind,
                });
            }
            if kind.is_port() {
                return Err(ZXError::FillPortWithPort { pos, kind: *kind });
            }
            let node = &mut new_zx.nodes[node_id];
            node.kind = *kind;
        }
        new_zx.refresh_measurement_metadata(port_map);
        Ok(new_zx)
    }

    pub(crate) fn local_stabilizer_flow_supports_for_node_kind(
        &self,
        node_id: usize,
        kind: NodeKind,
    ) -> Vec<Vec<(usize, Pauli)>> {
        let mut rows = Vec::new();

        match kind {
            NodeKind::Y => {
                if let Some((_, edge_id)) = self.neighbor_edges(node_id).next() {
                    rows.push(vec![(node_id, Pauli::Y), (edge_id, Pauli::Y)]);
                }
            }
            NodeKind::X | NodeKind::Z => {
                let cross_pauli = kind.cross_pauli();
                let broadcast_pauli = cross_pauli.flip();
                let mut neighbor_edges = self.neighbor_edges(node_id).collect::<Vec<_>>();
                neighbor_edges.sort_unstable_by_key(|&(neighbor, _)| {
                    (self.nodes[neighbor].pos - self.nodes[node_id].pos).to_array()
                });

                for neighbor_pair in neighbor_edges.windows(2) {
                    rows.push(
                        neighbor_pair
                            .iter()
                            .map(|&(_, edge_id)| {
                                let edge = self.edge_by_id(edge_id);
                                (edge.id, pauli_at_small_node(edge, node_id, cross_pauli))
                            })
                            .collect(),
                    );
                }

                let mut broadcast = vec![(node_id, broadcast_pauli)];
                broadcast.extend(neighbor_edges.iter().map(|&(_, edge_id)| {
                    let edge = self.edge_by_id(edge_id);
                    (edge.id, pauli_at_small_node(edge, node_id, broadcast_pauli))
                }));
                rows.push(broadcast);
            }
            NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => {
                if let Some((_, edge_id)) = self.neighbor_edges(node_id).next() {
                    let edge = self.edge_by_id(edge_id);
                    for pauli in [Pauli::X, Pauli::Z] {
                        rows.push(vec![
                            (node_id, pauli),
                            (edge.id, pauli_at_small_node(edge, node_id, pauli)),
                        ]);
                    }
                }
            }
        }

        rows
    }

    pub(crate) fn row_has_external_edge_pair_support(&self, row: &PauliString) -> bool {
        let pair_matches = |edge: &ZXEdge| {
            let (left, right) = self.edge_column_pair(edge);
            row.get(left) == row.get(right)
        };
        // Every mismatch has a nonzero side. Sparse rows need only those
        // pairs; dense rows keep the single check per undirected edge.
        if row.weight() < self.edges.len() / 4 {
            return row
                .iter_support()
                .filter(|(col, _)| *col >= self.nodes.len())
                .all(|(col, _)| pair_matches(self.edge_by_id(col)));
        }
        self.edges
            .iter()
            .filter(|edge| edge.n1 < edge.n2)
            .all(pair_matches)
    }

    /// Recover the linear tensor-center factors used by signed multiplication.
    /// Crossing-center support belongs to physical materialization and is not
    /// included here: its OR over incident arms is not a linear Pauli factor.
    pub(crate) fn reconstruct_raw_centers(&self, row: &mut PauliString) {
        let mut candidates = row
            .iter_support()
            .map(|(column, _)| {
                if column < self.nodes.len() {
                    column
                } else {
                    self.edge_by_id(column).n1
                }
            })
            .collect::<SmallVec<[usize; 32]>>();
        candidates.sort_unstable();
        candidates.dedup();
        for node_id in candidates {
            let Some((_, edge_id)) = self.neighbor_edges(node_id).next() else {
                continue;
            };
            let edge = self.edge_by_id(edge_id);
            let mut pauli = row.get(edge_id);
            if edge.hadamard && node_id == edge.sorted_nodes().1 {
                pauli = pauli.flip();
            }
            let node = &self.nodes[node_id];
            if matches!(node.kind, NodeKind::X | NodeKind::Z) {
                let broadcast = node.kind.cross_pauli().flip();
                pauli = if pauli & broadcast {
                    broadcast
                } else {
                    Pauli::I
                };
            }
            row.set(node_id, pauli);
        }
    }

    /// Removes derived cross-axis center support before raw-table phase recovery.
    pub(crate) fn clear_cross_centers(&self, table: &mut [PauliString]) {
        let mut dense = SmallVec::<[&mut PauliString; 8]>::new();
        let mut supported = SmallVec::<[(usize, Pauli); 32]>::new();
        for row in table {
            if row.weight() >= self.nodes.len() / 4 {
                dense.push(row);
                continue;
            }
            supported.clear();
            supported.extend(
                row.iter_support()
                    .take_while(|(col, _)| *col < self.nodes.len()),
            );
            for &(col, current) in &supported {
                let cross = match self.nodes[col].kind {
                    NodeKind::X | NodeKind::Z => self.nodes[col].kind.cross_pauli(),
                    NodeKind::Y | NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => continue,
                };
                if current & cross {
                    row.set(col, current ^ cross);
                }
            }
        }
        if dense.is_empty() {
            return;
        }
        // Preserve the node-first batch walk for dense rows. Gathering their
        // support into temporary buffers regresses measured wire builds.
        for node in &self.nodes {
            let cross = match node.kind {
                NodeKind::X | NodeKind::Z => node.kind.cross_pauli(),
                NodeKind::Y | NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => continue,
            };
            for row in &mut dense {
                let current = row.get(node.id);
                if current & cross {
                    row.set(node.id, current ^ cross);
                }
            }
        }
    }

    /// Linear components whose OR is a row's support at `column` on `axis`:
    /// the arms of a crossing centre, otherwise the column itself.
    pub(crate) fn readout_components(&self, column: usize, axis: Pauli) -> Vec<(usize, Pauli)> {
        let Some(node) = self.nodes.get(column) else {
            return vec![(column, axis)];
        };
        if !matches!(node.kind, NodeKind::X | NodeKind::Z) || node.kind.cross_pauli() != axis {
            return vec![(column, axis)];
        }
        self.neighbor_edges(node.id)
            .map(|(_, edge)| {
                let edge = self.edge_by_id(edge);
                let flip = edge.hadamard && node.id == edge.sorted_nodes().1;
                (edge.id, if flip { axis.flip() } else { axis })
            })
            .collect()
    }

    pub(crate) fn reconstruct_cross_center(&self, table: &mut [PauliString]) {
        let incident = self.cross_incident.get_or_init(|| {
            self.nodes
                .iter()
                .map(|node| {
                    self.neighbor_edges(node.id)
                        .map(|(_, edge_id)| {
                            let edge = self.edge_by_id(edge_id);
                            (edge.id, edge.hadamard && node.id == edge.sorted_nodes().1)
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
                .into()
        });
        let mut candidates = SmallVec::<[usize; 32]>::new();
        for row in table {
            // ponytail: retain the dense walk above this conservative density;
            // tune only if paired profiles show sparse gathering wins there too.
            if row.weight() < self.nodes.len() / 4 {
                candidates.clear();
                candidates.extend(row.iter_support().map(|(col, _)| {
                    if col < self.nodes.len() {
                        col // Include stale centers even when all their arms cancelled.
                    } else {
                        self.edges[col - self.nodes.len()].n1
                    }
                }));
                candidates.sort_unstable();
                candidates.dedup();
                self.reconstruct_cross_centers_at(row, incident, candidates.iter().copied());
            } else {
                self.reconstruct_cross_centers_at(row, incident, 0..self.nodes.len());
            }
        }
    }

    fn reconstruct_cross_centers_at(
        &self,
        row: &mut PauliString,
        incident: &[CrossIncident],
        nodes: impl Iterator<Item = usize>,
    ) {
        for node_id in nodes {
            let node = &self.nodes[node_id];
            let cross_pauli = match node.kind {
                NodeKind::X | NodeKind::Z => node.kind.cross_pauli(),
                NodeKind::Y | NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => continue,
            };
            // Edge Paulis use the smaller endpoint's frame. Retain the cached
            // incidence lookup, including its last-edge rule for parallel edges.
            let crosses = incident[node.id].iter().any(|&(edge_col, flip)| {
                row.get(edge_col)
                    & if flip {
                        cross_pauli.flip()
                    } else {
                        cross_pauli
                    }
            });
            // Recompute exactly, preserving the arm axis and removing phantom
            // centers when previously reconstructed crossing edges have cancelled.
            let current = row.get(node.id);
            let arm = if current & cross_pauli {
                current ^ cross_pauli
            } else {
                current
            };
            row.set(node.id, if crosses { arm | cross_pauli } else { arm });
        }
    }

    /// Materializes a Pauli row into graph-position support for rendering and
    /// lowering. Row role and measurement identity live on
    /// [`StabilizerGenerator`](super::StabilizerGenerator), not this algebraic
    /// support value.
    pub(crate) fn pauli_string_to_stabilizer(&self, ps: PauliString) -> Stabilizer {
        let row_phase = self.stabilizer_row_phases(std::slice::from_ref(&ps))[0];
        self.pauli_string_to_stabilizer_with_row_phase(ps, row_phase)
    }

    /// Rebuild cross centers before materializing a dense row.
    pub fn materialize_stabilizer(&self, mut ps: PauliString) -> Stabilizer {
        self.reconstruct_cross_center(std::slice::from_mut(&mut ps));
        self.pauli_string_to_stabilizer(ps)
    }

    /// Rebuild cross centers while preserving an already-derived sign.
    pub fn materialize_stabilizer_with_sign(&self, mut ps: PauliString, sign: bool) -> Stabilizer {
        self.reconstruct_cross_center(std::slice::from_mut(&mut ps));
        let mut stabilizer = self.pauli_string_to_stabilizer_with_row_phase(ps, 0);
        stabilizer.sign = sign;
        stabilizer
    }

    pub(crate) fn pauli_string_to_stabilizer_with_row_phase(
        &self,
        ps: PauliString,
        row_phase: u8,
    ) -> Stabilizer {
        let num_nodes = self.nodes.len();
        let mut port_stabilizer = FxHashMap::default();
        let mut interior_nodes = FxHashMap::default();
        let mut interior_edges = FxHashMap::default();
        for (col, pauli) in ps.iter_support() {
            if col >= num_nodes {
                let edge = self.edges[col - num_nodes];
                if edge.n1 > edge.n2 {
                    continue;
                }
                debug_assert_eq!(
                    pauli,
                    ps.get(
                        self.edge_id(edge.n2, edge.n1)
                            .expect("edge endpoints came from the graph")
                    )
                );
                let pos1 = self.nodes[edge.n1].pos;
                let pos2 = self.nodes[edge.n2].pos;
                interior_edges.insert((pos1, pos2), pauli);
                continue;
            }
            let node = self.nodes[col];
            interior_nodes.insert(node.pos, pauli);
            if node.kind.is_port() {
                port_stabilizer.insert(node.pos, pauli);
            }
        }
        Stabilizer {
            paulis: ps,
            sign: contracted_sign(row_phase, &interior_edges),
            port_stabilizer,
            interior_nodes,
            interior_edges,
        }
    }

    #[cfg(feature = "gltf")]
    pub(crate) fn measurement_outcome_position(&self, name: &str) -> Option<Vec3> {
        self.action_graph.ordered_nodes().find_map(|node| {
            let Action::Measure {
                name: candidate,
                target,
            } = &node.action
            else {
                return None;
            };
            (candidate == name)
                .then(|| {
                    self.measurement_column(target)
                        .and_then(|col| self.sink_position(col))
                })
                .flatten()
        })
    }

    fn refresh_measurement_metadata(&mut self, port_map: &HashMap<IVec3, NodeKind>) {
        let ordinals: Vec<usize> = self
            .action_graph
            .ordered_nodes()
            .map(|node| node.ordinal)
            .collect();
        for ordinal in ordinals {
            let Some(node) = self.action_graph.node_by_ordinal(ordinal) else {
                continue;
            };
            let Action::Measure {
                target: MeasureTarget::Node(pos),
                ..
            } = &node.action
            else {
                continue;
            };
            if !port_map.contains_key(pos) {
                continue;
            }
            let Some(&node_id) = self.pos_to_node.get(pos) else {
                continue;
            };
            if let Some(observable) = measurement_observable_for_node_kind(self.nodes[node_id].kind)
            {
                self.action_graph
                    .set_measurement_observable(ordinal, observable)
                    .expect("ordinal comes from ordered_nodes so it is always valid");
            }
        }
    }

    /// Returns whether the graph has any open port node.
    pub fn is_open(&self) -> bool {
        self.nodes.iter().any(|node| node.kind.is_port())
    }

    pub(crate) fn check_external_table_limits(
        &self,
        limits: ModuleCertificationLimits,
    ) -> Result<(), StabilizerError> {
        SearchBudget::with_limits(limits).check_matrix(
            // A phase query retains its input and scratch rows alongside the
            // external phase basis; each rank is bounded by the source rows.
            self.external_source_row_count().saturating_mul(3),
            self.total_ids,
            0,
            0,
        )
    }

    fn external_source_row_count(&self) -> usize {
        self.nodes
            .iter()
            .map(|node| {
                let degree = self.neighbor_ids(node.id).len();
                match node.kind {
                    NodeKind::X | NodeKind::Z => degree.max(1),
                    NodeKind::Y => usize::from(degree != 0),
                    NodeKind::Port | NodeKind::T | NodeKind::Selective(_) => {
                        2 * usize::from(degree != 0)
                    }
                }
            })
            .fold(0, usize::saturating_add)
    }
}

pub(super) fn pauli_at_small_node(edge: &ZXEdge, current_id: usize, current_pauli: Pauli) -> Pauli {
    let (_, n2) = edge.sorted_nodes();
    if current_id == n2 && edge.hadamard {
        current_pauli.flip()
    } else {
        current_pauli
    }
}

fn measurement_observable_for_node_kind(kind: NodeKind) -> Option<MeasurementObservable> {
    match kind {
        NodeKind::X => Some(MeasurementObservable::Concrete(PauliBasis::Z)),
        NodeKind::Y => Some(MeasurementObservable::Concrete(PauliBasis::Y)),
        NodeKind::Z => Some(MeasurementObservable::Concrete(PauliBasis::X)),
        _ => None,
    }
}

/// Validates one selective fill and records the boundary spider it fills with.
pub(super) fn insert_selective_fill(
    port_map: &mut HashMap<IVec3, NodeKind>,
    pos: IVec3,
    kind: crate::SelectiveKind,
    chosen: PauliBasis,
) -> Result<(), super::RuntimeBasisError> {
    if ![kind.pauli_if_true(), kind.pauli_if_false()].contains(&chosen) {
        return Err(super::RuntimeBasisError::InvalidSelectiveBasis { pos, kind, chosen });
    }
    if port_map.insert(pos, filled_node_kind(chosen)).is_some() {
        return Err(super::RuntimeBasisError::DuplicateSelective(pos));
    }
    Ok(())
}

/// The boundary spider a chosen measurement basis fills a selective node with.
pub(super) fn filled_node_kind(chosen: PauliBasis) -> NodeKind {
    match chosen {
        PauliBasis::X => NodeKind::Z,
        PauliBasis::Y => NodeKind::Y,
        PauliBasis::Z => NodeKind::X,
    }
}

impl ZXGraph {
    /// Demotes prepared T resources to ports. Both have the same local flow
    /// rows and cross-center behavior, so a signed phase basis survives.
    pub(super) fn demote_t_nodes_to_ports(&mut self) {
        for node in &mut self.nodes {
            if node.kind == NodeKind::T {
                node.kind = NodeKind::Port;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use bloq_utils::{Pauli, PauliString};
    use glam::ivec3;
    use rustc_hash::FxHashMap;

    use crate::{Action, ActionDag, Direction, GalleryItem, MeasureTarget, MeasurementObservable};

    use super::{CsrAdjacency, NodeKind, ZXEdge, ZXError, ZXGraph, ZXNode};

    #[test]
    fn csr_preserves_last_edge_lookup_for_parallel_neighbors() {
        let adjacency = CsrAdjacency::from_edges(2, &[(0, 1, 2), (1, 0, 3), (0, 1, 4), (1, 0, 5)]);

        assert!(adjacency.has_parallel_edges());
        assert_eq!(adjacency.edge_id(0, 1), Some(4));
        assert_eq!(
            adjacency
                .neighbor_edges(0)
                .expect("node exists")
                .collect::<Vec<_>>(),
            vec![(1, 4), (1, 4)]
        );
    }

    #[test]
    fn local_phase_queries_preserve_parallel_edge_reduction() {
        for hadamard in [false, true] {
            let mut zx =
                manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Z));
            for node in &mut zx.nodes {
                node.kind = NodeKind::Port;
            }
            for mut edge in [zx.edges[0], zx.edges[1]] {
                edge.id = zx.total_ids;
                zx.total_ids += 1;
                zx.edges.push(edge);
            }
            for edge in &mut zx.edges {
                edge.hadamard = hadamard;
            }
            zx.adjacency = CsrAdjacency::from_edges(
                zx.nodes.len(),
                &zx.edges
                    .iter()
                    .map(|edge| (edge.n1, edge.n2, edge.id))
                    .collect::<Vec<_>>(),
            );
            let signed = zx.to_signed_external_generator_table();
            let reference = zx.clone();
            reference
                .stabilizer_phase_basis
                .set(std::sync::Arc::new(super::StabilizerPhaseBasis::new(
                    signed.clone(),
                )))
                .unwrap();
            for bits in 0..4096 {
                let row = PauliString::from_terms(
                    zx.total_ids,
                    (0..zx.total_ids).map(|column| {
                        (
                            column,
                            [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y][(bits >> (2 * column)) & 3],
                        )
                    }),
                );
                let rows = std::slice::from_ref(&row);
                let expected = zx.row_phases_against(rows, &signed);
                assert_eq!(zx.external_stabilizer_row_phases(rows), expected);
                assert_eq!(zx.stabilizer_row_phases(rows), expected);
                assert_eq!(
                    zx.contains_stabilizer_support(&row),
                    reference.contains_stabilizer_support(&row)
                );
            }
        }
    }

    #[test]
    fn local_phase_queries_reject_support_outside_the_graph() {
        let zx = manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Z));
        let mut row = PauliString::new(zx.total_ids + 1);
        assert!(zx.contains_stabilizer_support(&row));
        row.set(zx.total_ids, Pauli::X);
        assert!(!zx.contains_stabilizer_support(&row));
        assert_eq!(zx.stabilizer_row_phases(&[row]), [0]);
    }

    #[test]
    fn measurement_column_rejects_an_edge_step_past_the_coordinate_boundary() {
        let mut graph = manual_single_node_graph(
            NodeKind::Z,
            MeasurementObservable::Concrete(crate::PauliBasis::Z),
        );
        let boundary = ivec3(i32::MAX, 0, 0);
        graph.nodes[0].pos = boundary;
        graph.pos_to_node = FxHashMap::from_iter([(boundary, 0)]);

        assert_eq!(
            graph.measurement_column(&MeasureTarget::Edge {
                src: boundary,
                dir: Direction::XPLUS,
            }),
            None
        );
    }

    #[cfg(feature = "gltf")]
    #[test]
    fn sink_position_widens_boundary_edge_endpoints_before_averaging() {
        let mut graph =
            manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Z));
        graph.nodes[0].pos = ivec3(i32::MAX - 1, 0, 0);
        graph.nodes[1].pos = ivec3(i32::MAX, 0, 0);

        let position = graph.sink_position(graph.edges[0].id).unwrap();
        assert!(position.x > 1.0e9, "boundary midpoint must not wrap");
    }

    #[test]
    fn fill_ports_rejects_t_nodes() {
        let graph = GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = ZXGraph::try_from(&graph).unwrap();
        let port_map = HashMap::from([(ivec3(1, 0, 0), NodeKind::X)]);

        let err = zx
            .fill_ports(&port_map)
            .expect_err("T nodes must not be fillable");

        assert!(matches!(
            err,
            ZXError::FillNonPort {
                pos,
            kind: NodeKind::T
        } if pos == ivec3(1, 0, 0)
        ));
    }

    #[test]
    fn fill_ports_reuses_cached_topology_for_new_cross_centers() {
        let mut zx =
            manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Z));
        zx.nodes[0].kind = NodeKind::Port;
        zx.reconstruct_cross_center(&mut [PauliString::new(zx.total_ids)]);
        zx.stabilizer_phase_basis
            .set(std::sync::Arc::new(super::StabilizerPhaseBasis::new(
                zx.to_signed_external_generator_table(),
            )))
            .unwrap();
        assert!(zx.stabilizer_phase_basis.get().is_some());

        let edge = zx.get_edge(0, 1);
        let mut row = PauliString::new(zx.total_ids);
        row.set(edge.id, Pauli::X);
        let filled = zx
            .fill_ports(&HashMap::from([(zx.nodes[0].pos, NodeKind::X)]))
            .expect("port fills");
        assert!(filled.cross_incident.get().is_some());
        assert!(std::sync::Arc::ptr_eq(
            zx.stabilizer_phase_basis.get().unwrap(),
            filled.stabilizer_phase_basis.get().unwrap()
        ));
        let rows = filled.to_external_generator_table();
        let cached_phases = filled.stabilizer_row_phases(&rows);
        let mut fresh = filled.clone();
        fresh.stabilizer_phase_basis.take();
        assert_eq!(cached_phases, fresh.stabilizer_row_phases(&rows));
        filled.reconstruct_cross_center(std::slice::from_mut(&mut row));

        assert!(row.get(0) & Pauli::X);
    }

    #[test]
    fn sparse_edge_pair_checks_match_dense_reference_with_parallel_edges() {
        let mut zx = ZXGraph::try_from(
            &GalleryItem::CCZFactoryWithTels
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection"),
        )
        .unwrap();
        // Parallel endpoint lookups intentionally select the last edge pair.
        for mut edge in [zx.edges[0], zx.edges[1]] {
            edge.id = zx.total_ids;
            zx.total_ids += 1;
            zx.edges.push(edge);
        }
        zx.adjacency = CsrAdjacency::from_edges(
            zx.nodes.len(),
            &zx.edges
                .iter()
                .map(|e| (e.n1, e.n2, e.id))
                .collect::<Vec<_>>(),
        );
        let dense_reference = |row: &PauliString| {
            zx.edges.iter().filter(|e| e.n1 < e.n2).all(|edge| {
                let (left, right) = zx.edge_column_pair(edge);
                row.get(left) == row.get(right)
            })
        };
        let columns = [
            zx.edges[0].id,
            zx.edges[1].id,
            zx.edges[2].id,
            zx.edges[3].id,
            zx.total_ids - 2,
            zx.total_ids - 1,
        ];
        assert!(zx.edges.len() / 4 > columns.len() + 1);
        assert!(zx.total_ids > 64);
        for bits in 0..4096 {
            let mut row = PauliString::new(zx.total_ids);
            row.set(0, Pauli::Y); // Node support does not constrain edge pairs.
            for (shift, col) in columns.into_iter().enumerate() {
                row.set(
                    col,
                    [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][(bits >> (2 * shift)) & 3],
                );
            }
            assert_eq!(
                zx.row_has_external_edge_pair_support(&row),
                dense_reference(&row),
                "{bits}"
            );
        }
        for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
            let mut row =
                PauliString::from_terms(zx.total_ids, (0..zx.total_ids).map(|col| (col, pauli)));
            assert!(zx.row_has_external_edge_pair_support(&row));
            row.set(zx.total_ids - 1, Pauli::I);
            assert!(!zx.row_has_external_edge_pair_support(&row));
        }
    }

    #[test]
    fn supported_cross_centers_match_dense_reference_with_phantom_and_directed_arms() {
        // Padding puts the exhaustive four-column cases on the sparse path,
        // across packed-word boundaries. The all-Y row exercises the dense path.
        const N: usize = 129;
        for hadamard in [false, true] {
            let mut zx =
                manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Z));
            for id in 2..N {
                zx.nodes
                    .push(ZXNode::new(id, ivec3(id as i32, 0, 0), NodeKind::X));
            }
            for (index, edge) in zx.edges.iter_mut().enumerate() {
                edge.id = N + index;
                edge.hadamard = hadamard;
            }
            zx.adjacency = CsrAdjacency::from_edges(N, &[(0, 1, N), (1, 0, N + 1)]);
            zx.total_ids = N + 2;
            for kinds in [
                [NodeKind::X, NodeKind::Z],
                [NodeKind::Z, NodeKind::X],
                [NodeKind::Port, NodeKind::Y],
                [NodeKind::T, NodeKind::X],
            ] {
                zx.nodes[0].kind = kinds[0];
                zx.nodes[1].kind = kinds[1];
                let mut rows = (0..256)
                    .map(|bits| {
                        let mut row = PauliString::new(zx.total_ids);
                        for (shift, col) in [0, 1, N, N + 1].into_iter().enumerate() {
                            row.set(
                                col,
                                [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][(bits >> (2 * shift)) & 3],
                            );
                        }
                        row.set(N - 1, Pauli::Y); // Isolated phantom cross plus a real arm.
                        row
                    })
                    .collect::<Vec<_>>();
                rows.push(PauliString::new(zx.total_ids));
                rows.push(PauliString::from_terms(
                    zx.total_ids,
                    (0..zx.total_ids).map(|col| (col, Pauli::Y)),
                ));
                let mut expected = rows.clone();
                for row in &mut expected {
                    for node in &zx.nodes {
                        if !matches!(node.kind, NodeKind::X | NodeKind::Z) {
                            continue;
                        }
                        let cross = node.kind.cross_pauli();
                        let crosses = zx.neighbor_edges(node.id).any(|(_, edge_id)| {
                            let edge = zx.edge_by_id(edge_id);
                            super::pauli_at_small_node(edge, node.id, row.get(edge_id)) & cross
                        });
                        let arm = if row.get(node.id) & cross.flip() {
                            cross.flip()
                        } else {
                            Pauli::I
                        };
                        row.set(node.id, arm | if crosses { cross } else { Pauli::I });
                    }
                }
                zx.reconstruct_cross_center(&mut rows);
                assert_eq!(rows, expected);
                for row in &mut expected {
                    for node in &zx.nodes {
                        if matches!(node.kind, NodeKind::X | NodeKind::Z) {
                            let arm = node.kind.cross_pauli().flip();
                            row.set(
                                node.id,
                                if row.get(node.id) & arm {
                                    arm
                                } else {
                                    Pauli::I
                                },
                            );
                        }
                    }
                }
                zx.clear_cross_centers(&mut rows);
                assert_eq!(rows, expected);
            }
        }
    }

    #[test]
    fn validate_for_program_rejects_selective_edge_observables() {
        let zx =
            manual_two_node_edge_graph(MeasurementObservable::Selective(crate::SelectiveKind::XY));

        let err = zx
            .validate_for_program()
            .expect_err("selective edge observables should be rejected");
        assert!(matches!(
            err,
            ZXError::IncompatibleMeasurementObservable {
                target: MeasureTarget::Edge { .. },
                observable: MeasurementObservable::Selective(_),
                ..
            }
        ));
    }

    #[test]
    fn validate_for_program_rejects_concrete_y_edge_observables() {
        let zx = manual_two_node_edge_graph(MeasurementObservable::Concrete(crate::PauliBasis::Y));

        let err = zx
            .validate_for_program()
            .expect_err("concrete Y edge observables should be rejected");
        assert!(matches!(
            err,
            ZXError::IncompatibleMeasurementObservable {
                target: MeasureTarget::Edge { .. },
                observable: MeasurementObservable::Concrete(crate::PauliBasis::Y),
                ..
            }
        ));
    }

    #[test]
    fn validate_for_program_rejects_node_observable_kind_mismatch() {
        let zx = manual_single_node_graph(
            NodeKind::X,
            MeasurementObservable::Concrete(crate::PauliBasis::X),
        );

        let err = zx
            .validate_for_program()
            .expect_err("node observable mismatches should be rejected");
        assert!(matches!(
            err,
            ZXError::IncompatibleMeasurementObservable {
                target: MeasureTarget::Node(_),
                observable: MeasurementObservable::Concrete(crate::PauliBasis::X),
                ..
            }
        ));
    }

    fn manual_single_node_graph(kind: NodeKind, observable: MeasurementObservable) -> ZXGraph {
        let action_graph = {
            let mut dag = ActionDag::from_actions(&[Action::Measure {
                target: MeasureTarget::Node(ivec3(0, 0, 0)),
                name: "m".into(),
            }]);
            dag.set_measurement_observable(0, observable).unwrap();
            dag
        };
        ZXGraph {
            nodes: vec![ZXNode::new(0, ivec3(0, 0, 0), kind)],
            edges: vec![],
            adjacency: CsrAdjacency::from_edges(1, &[]),
            pos_to_node: FxHashMap::from_iter([(ivec3(0, 0, 0), 0)]),
            action_graph,
            total_ids: 1,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        }
    }

    fn manual_two_node_edge_graph(observable: MeasurementObservable) -> ZXGraph {
        let action_graph = {
            let mut dag = ActionDag::from_actions(&[Action::Measure {
                target: MeasureTarget::Edge {
                    src: ivec3(0, 0, 0),
                    dir: Direction::XPLUS,
                },
                name: "m".into(),
            }]);
            dag.set_measurement_observable(0, observable).unwrap();
            dag
        };
        ZXGraph {
            nodes: vec![
                ZXNode::new(0, ivec3(0, 0, 0), NodeKind::X),
                ZXNode::new(1, ivec3(1, 0, 0), NodeKind::Z),
            ],
            edges: vec![
                ZXEdge {
                    n1: 0,
                    n2: 1,
                    id: 2,
                    hadamard: false,
                },
                ZXEdge {
                    n1: 1,
                    n2: 0,
                    id: 3,
                    hadamard: false,
                },
            ],
            adjacency: CsrAdjacency::from_edges(2, &[(0, 1, 2), (1, 0, 3)]),
            pos_to_node: FxHashMap::from_iter([(ivec3(0, 0, 0), 0), (ivec3(1, 0, 0), 1)]),
            action_graph,
            total_ids: 4,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        }
    }
}
