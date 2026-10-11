//! Conversion between block graphs and their local ZX representation.

use super::graph::{CsrAdjacency, ZXEdge, ZXNode};
use super::{NodeKind, ZXError, ZXGraph};
use crate::{Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe};
use bloq_utils::{Basis, UDirection};
use glam::IVec3;
use std::collections::{HashMap, HashSet};

impl TryFrom<&BlockGraph> for ZXGraph {
    type Error = ZXError;

    fn try_from(graph: &BlockGraph) -> Result<Self, Self::Error> {
        graph
            .require_flat_hierarchy("ZX conversion")
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        let graph = graph.copy_local_geometry().fix_shadowed_faces();
        if let Some(target) = graph.action_graph().ordered_nodes().find_map(|node| {
            if let crate::Action::Branch { target, .. } = node.action {
                Some(target)
            } else {
                None
            }
        }) {
            return Err(ZXError::Graph(Box::new(
                crate::InvalidActionError::MissingBranchValue(target).into(),
            )));
        }
        let actions = crate::validate::validate_source(&graph)
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        Self::from_block_graph_parts(&graph, actions)
    }
}

impl ZXGraph {
    /// Converts a structurally/action-syntactically valid source graph while
    /// physical action analysis itself is still deriving stabilizers.
    #[doc(hidden)]
    pub fn from_block_graph_for_analysis(graph: &BlockGraph) -> Result<Self, ZXError> {
        graph
            .require_flat_hierarchy("ZX conversion")
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        let graph = graph.copy_local_geometry().fix_shadowed_faces();
        crate::validate::validate_structure(&graph)
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        let action_graph = graph
            .build_action_graph(&graph.actions())
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        Self::from_block_graph_parts(&graph, action_graph)
    }

    /// Build the center's tensor/incidence view from a validated projection
    /// whose shadowed faces have already been fixed. Neighbor kinds are copied
    /// from that complete projection: normalizing a truncated graph could change
    /// their spider colors. Only the center has its complete neighborhood here.
    ///
    /// Sorted anchor positions preserve the full converter's relative node
    /// order and thus its directed Hadamard frames. Extended block endpoints
    /// still resolve to their owning anchors. The view carries no source actions
    /// and must not be used for whole-program stabilizer or interface analysis.
    pub(crate) fn from_local_block_neighborhood(
        graph: &BlockGraph,
        center: IVec3,
    ) -> Result<Self, ZXError> {
        let center = graph.get_block(center).ok_or_else(|| {
            ZXError::Graph(Box::new(crate::BlockGraphError::BlockNotFound(center)))
        })?;
        let mut blocks = graph.neighbors(center.pos);
        blocks.push(center);
        blocks.sort_unstable_by_key(|block| block.pos.to_array());
        blocks.dedup_by_key(|block| block.pos);
        let nodes = blocks
            .iter()
            .enumerate()
            .map(|(id, block)| {
                ZXNode::new(id, block.pos, block_to_node_kind(block))
                    .with_role(block.port_role().unwrap_or_default())
            })
            .collect::<Vec<_>>();
        let pos_to_node = nodes
            .iter()
            .map(|node| (node.pos, node.id))
            .collect::<crate::FxHashMap<_, _>>();
        let mut pipes = graph
            .pipes_at(center.pos)
            .map(|pipe| {
                let source = graph
                    .get_endpoint_block(pipe.src)
                    .expect("validated pipe source has an owner");
                let target = graph
                    .get_endpoint_block(pipe.dst())
                    .expect("validated pipe target has an owner");
                let (a, b) = (pos_to_node[&source.pos], pos_to_node[&target.pos]);
                (a.min(b), a.max(b), pipe.hadamard)
            })
            .collect::<Vec<_>>();
        pipes.sort_unstable_by_key(|&(a, b, _)| (a, b));
        let mut total_ids = nodes.len();
        let mut edges = Vec::with_capacity(2 * pipes.len());
        let mut directed_edges = Vec::with_capacity(2 * pipes.len());
        for (a, b, hadamard) in pipes {
            push_edge_pair(
                a,
                b,
                hadamard,
                &mut total_ids,
                &mut edges,
                &mut directed_edges,
            );
        }
        let adjacency = CsrAdjacency::from_edges(nodes.len(), &directed_edges);
        Ok(Self {
            nodes,
            edges,
            adjacency,
            pos_to_node,
            action_graph: crate::ActionDag::default(),
            total_ids,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        })
    }

    pub(super) fn from_block_graph_parts(
        graph: &BlockGraph,
        mut action_graph: crate::ActionDag,
    ) -> Result<Self, ZXError> {
        graph
            .require_flat_hierarchy("ZX conversion")
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        let mut nodes = Vec::with_capacity(graph.block_count());
        let mut edges = Vec::with_capacity(2 * graph.pipe_count());
        let mut directed_edges = Vec::with_capacity(2 * graph.pipe_count());
        let mut endpoint_to_node: HashMap<IVec3, usize> =
            HashMap::with_capacity(graph.block_count());
        let mut id = 0;

        // Stable columns keep measurement witnesses independent of source storage order.
        let mut blocks = graph.blocks().collect::<Vec<_>>();
        blocks.sort_unstable_by_key(|block| block.pos.to_array());
        for block in blocks {
            let node = ZXNode::new(id, block.pos, block_to_node_kind(block))
                .with_role(block.port_role().unwrap_or_default());
            for endpoint in block.connectable_offsets() {
                endpoint_to_node.insert(block.pos + endpoint, id);
            }
            nodes.push(node);
            id += 1;
        }

        // Source records use the authored pipe source; both ZX edge columns use
        // the smaller node ID's frame, irrespective of the action's direction.
        for ordinal in 0..action_graph.ordered_nodes().count() {
            let action = action_graph
                .node_by_ordinal(ordinal)
                .expect("ordered action exists");
            let crate::Action::Measure {
                target: crate::MeasureTarget::Edge { src, dir },
                ..
            } = action.action
            else {
                continue;
            };
            let pipe = graph
                .get_pipe(src, src + dir.to_ivec3())
                .expect("validated measurement pipe");
            if pipe.hadamard && endpoint_to_node[&pipe.src] > endpoint_to_node[&pipe.dst()] {
                let Some(crate::MeasurementObservable::Concrete(basis)) = action.measurement else {
                    continue;
                };
                action_graph
                    .set_measurement_observable(
                        ordinal,
                        crate::MeasurementObservable::Concrete(match basis {
                            crate::PauliBasis::X => crate::PauliBasis::Z,
                            crate::PauliBasis::Z => crate::PauliBasis::X,
                            crate::PauliBasis::Y => crate::PauliBasis::Y,
                        }),
                    )
                    .expect("measurement ordinal exists");
            }
        }

        let mut pipes = graph.pipes().collect::<Vec<_>>();
        pipes.sort_unstable_by_key(|pipe| {
            let first = endpoint_to_node[&pipe.src];
            let second = endpoint_to_node[&pipe.dst()];
            (first.min(second), first.max(second))
        });
        for pipe in pipes {
            push_edge_pair(
                endpoint_to_node[&pipe.src],
                endpoint_to_node[&pipe.dst()],
                pipe.hadamard,
                &mut id,
                &mut edges,
                &mut directed_edges,
            );
        }
        let adjacency = CsrAdjacency::from_edges(nodes.len(), &directed_edges);

        // node_at needs only block centers, so key the lookup by `node.pos`
        // rather than reusing the endpoint-offset map above.
        let pos_to_node = nodes.iter().map(|node| (node.pos, node.id)).collect();

        Ok(ZXGraph {
            nodes,
            edges,
            adjacency,
            pos_to_node,
            action_graph,
            total_ids: id,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        })
    }
}

fn push_edge_pair(
    mut n1: usize,
    mut n2: usize,
    hadamard: bool,
    id: &mut usize,
    edges: &mut Vec<ZXEdge>,
    directed_edges: &mut Vec<(usize, usize, usize)>,
) {
    if n1 > n2 {
        std::mem::swap(&mut n1, &mut n2);
    }
    edges.push(ZXEdge {
        n1,
        n2,
        id: *id,
        hadamard,
    });
    directed_edges.push((n1, n2, *id));
    *id += 1;
    edges.push(ZXEdge {
        n1: n2,
        n2: n1,
        id: *id,
        hadamard,
    });
    directed_edges.push((n2, n1, *id));
    *id += 1;
}

impl TryFrom<&ZXGraph> for BlockGraph {
    type Error = ZXError;

    fn try_from(value: &ZXGraph) -> Result<Self, Self::Error> {
        value.to_block_graph()
    }
}

impl ZXGraph {
    /// Converts the ZX graph back into a [`BlockGraph`], inferring cube kinds
    /// from spider structure. `TryFrom<&ZXGraph> for BlockGraph` is the public
    /// entry point.
    fn to_block_graph(&self) -> Result<BlockGraph, ZXError> {
        self.validate_positioned_graph()?;

        let mut nodes_to_handle: HashSet<usize> = self.nodes.iter().map(|node| node.id).collect();
        let mut edges_to_handle: HashSet<(usize, usize)> =
            self.sorted_undirected_edges().into_iter().collect();
        let mut graph = BlockGraph::new();

        self.handle_corner_nodes(&mut graph, &mut nodes_to_handle);
        self.handle_special_nodes_and_pipes(
            &mut graph,
            &mut nodes_to_handle,
            &mut edges_to_handle,
        )?;
        self.greedily_construct_blocks(&mut graph, &mut nodes_to_handle, &mut edges_to_handle)?;
        self.handle_leftover_nodes(&mut graph, &mut nodes_to_handle, &mut edges_to_handle)?;

        graph
            .set_actions(self.actions())
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        graph
            .validate()
            .map_err(|error| ZXError::Graph(Box::new(error)))?;
        Ok(graph)
    }

    fn validate_positioned_graph(&self) -> Result<(), ZXError> {
        for (n1, n2) in self.sorted_undirected_edges() {
            self.direction_between(n1, n2)?;
        }

        for node in &self.nodes {
            let degree = self.neighbor_ids(node.id).len();
            if self.axis_directions(node.id).len() == 3 {
                return Err(ZXError::ThreeDimensionalCorner { pos: node.pos });
            }
            match node.kind {
                NodeKind::Y | NodeKind::T | NodeKind::Selective(_) => {
                    if degree > 1 {
                        return Err(ZXError::SpecialNodeNotDangling {
                            pos: node.pos,
                            kind: node.kind,
                            degree,
                        });
                    }
                    if let Some(&neighbor) = self.neighbor_ids(node.id).first() {
                        let direction = self.direction_between(node.id, neighbor)?;
                        if direction.as_udirection() != UDirection::Z {
                            return Err(ZXError::SpecialNodeNotTimeLike {
                                pos: node.pos,
                                kind: node.kind,
                            });
                        }
                        if node.kind == NodeKind::T && direction != Direction::ZPLUS {
                            return Err(ZXError::TNodeNotFutureDirected { pos: node.pos });
                        }
                    }
                }
                NodeKind::Port => {
                    if degree > 1 {
                        return Err(ZXError::SpecialNodeNotDangling {
                            pos: node.pos,
                            kind: node.kind,
                            degree,
                        });
                    }
                }
                NodeKind::X | NodeKind::Z => {}
            }
        }
        Ok(())
    }

    fn handle_corner_nodes(&self, graph: &mut BlockGraph, nodes_to_handle: &mut HashSet<usize>) {
        for node in &self.nodes {
            if !is_zx_spider(node.kind) || !nodes_to_handle.contains(&node.id) {
                continue;
            }
            let directions = self.axis_directions(node.id);
            if directions.len() != 2 {
                continue;
            }
            let normal_direction = UDirection::iter()
                .find(|direction| !directions.contains(direction))
                .expect("one axis is always missing when exactly two are present");
            let normal_basis = spider_normal_basis(node.kind);
            let mut bases = [normal_basis.flip(); 3];
            bases[normal_direction.index()] = normal_basis;
            let cube_kind =
                CubeKind::try_from(bases).expect("corner inference always produces a valid cube");
            graph.add_block(self.cube_block(node.id, cube_kind));
            nodes_to_handle.remove(&node.id);
        }
    }

    fn handle_special_nodes_and_pipes(
        &self,
        graph: &mut BlockGraph,
        nodes_to_handle: &mut HashSet<usize>,
        edges_to_handle: &mut HashSet<(usize, usize)>,
    ) -> Result<(), ZXError> {
        for node in &self.nodes {
            if is_zx_spider(node.kind) || !nodes_to_handle.contains(&node.id) {
                continue;
            }
            if self.neighbor_ids(node.id).is_empty() {
                graph.add_block(self.special_block(node.id));
                nodes_to_handle.remove(&node.id);
            }
        }

        for (n1, n2) in self.sorted_undirected_edges() {
            if !edges_to_handle.contains(&(n1, n2)) {
                continue;
            }
            let node1 = self.nodes[n1];
            let node2 = self.nodes[n2];
            if is_zx_spider(node1.kind) || is_zx_spider(node2.kind) {
                continue;
            }
            if nodes_to_handle.remove(&n1) {
                graph.add_block(self.special_block(n1));
            }
            if nodes_to_handle.remove(&n2) {
                graph.add_block(self.special_block(n2));
            }
            self.add_pipe_between_nodes(graph, n1, n2)?;
            edges_to_handle.remove(&(n1, n2));
        }
        Ok(())
    }

    fn greedily_construct_blocks(
        &self,
        graph: &mut BlockGraph,
        nodes_to_handle: &mut HashSet<usize>,
        edges_to_handle: &mut HashSet<(usize, usize)>,
    ) -> Result<(), ZXError> {
        let mut previous_nodes_left = nodes_to_handle.len() + 1;
        while nodes_to_handle.len() < previous_nodes_left {
            previous_nodes_left = nodes_to_handle.len();
            self.try_handle_edges(graph, nodes_to_handle, edges_to_handle)?;
        }
        Ok(())
    }

    fn try_handle_edges(
        &self,
        graph: &mut BlockGraph,
        nodes_to_handle: &mut HashSet<usize>,
        edges_to_handle: &mut HashSet<(usize, usize)>,
    ) -> Result<(), ZXError> {
        for (u, v) in self.sorted_undirected_edges() {
            if !edges_to_handle.contains(&(u, v)) {
                continue;
            }
            let can_infer_from_u =
                !nodes_to_handle.contains(&u) && is_zx_spider(self.nodes[u].kind);
            let can_infer_from_v =
                !nodes_to_handle.contains(&v) && is_zx_spider(self.nodes[v].kind);
            if !can_infer_from_u && !can_infer_from_v {
                continue;
            }

            let (infer_from, other_node) = if can_infer_from_u { (u, v) } else { (v, u) };
            let infer_cube_kind = match graph
                .get_block(self.nodes[infer_from].pos)
                .map(|block| block.kind)
            {
                Some(BlockKind::Cube(kind)) => kind,
                _ => continue,
            };

            if is_zx_spider(self.nodes[other_node].kind) {
                let inferred_kind =
                    self.infer_cube_kind_from_neighbor(infer_from, other_node, infer_cube_kind)?;
                if nodes_to_handle.remove(&other_node) {
                    graph.add_block(self.cube_block(other_node, inferred_kind));
                } else if let Some(Block {
                    kind: BlockKind::Cube(existing_kind),
                    ..
                }) = graph.get_block(self.nodes[other_node].pos)
                    && *existing_kind != inferred_kind
                {
                    return Err(ZXError::ConflictingCubeKinds {
                        pos: self.nodes[other_node].pos,
                        existing: *existing_kind,
                        inferred: inferred_kind,
                    });
                }
            } else if nodes_to_handle.remove(&other_node) {
                graph.add_block(self.special_block(other_node));
            }

            self.add_pipe_between_nodes(graph, infer_from, other_node)?;
            edges_to_handle.remove(&(u, v));
        }
        Ok(())
    }

    fn handle_leftover_nodes(
        &self,
        graph: &mut BlockGraph,
        nodes_to_handle: &mut HashSet<usize>,
        edges_to_handle: &mut HashSet<(usize, usize)>,
    ) -> Result<(), ZXError> {
        while nodes_to_handle
            .iter()
            .any(|node_id| is_zx_spider(self.nodes[*node_id].kind))
        {
            self.fix_kind_for_one_node(graph, nodes_to_handle)?;
            self.greedily_construct_blocks(graph, nodes_to_handle, edges_to_handle)?;
        }

        for node_id in nodes_to_handle.drain() {
            graph.add_block(self.special_block(node_id));
        }
        Ok(())
    }

    fn fix_kind_for_one_node(
        &self,
        graph: &mut BlockGraph,
        nodes_to_handle: &mut HashSet<usize>,
    ) -> Result<(), ZXError> {
        let fix_node = nodes_to_handle
            .iter()
            .filter(|node_id| is_zx_spider(self.nodes[**node_id].kind))
            .min_by_key(|node_id| self.nodes[**node_id].pos.to_array())
            .copied()
            .expect("caller guarantees there is an unresolved X/Z node");
        let fix_kind = if self.neighbor_ids(fix_node).is_empty() {
            match self.nodes[fix_node].kind {
                NodeKind::X => CubeKind::ZXZ,
                NodeKind::Z => CubeKind::ZXX,
                _ => unreachable!("only X/Z nodes reach the arbitrary fix path"),
            }
        } else {
            let neighbor = self.neighbor_ids(fix_node)[0];
            let direction = self
                .direction_between(fix_node, neighbor)?
                .as_udirection()
                .index();
            let mut spare = [Basis::X, Basis::Z].into_iter();
            let bases: [Basis; 3] = std::array::from_fn(|axis| {
                if axis == direction {
                    spider_incident_basis(self.nodes[fix_node].kind)
                } else {
                    spare
                        .next()
                        .expect("exactly two axes differ from the edge direction")
                }
            });
            CubeKind::try_from(bases).expect("arbitrary fixing always yields a valid cube")
        };
        graph.add_block(self.cube_block(fix_node, fix_kind));
        nodes_to_handle.remove(&fix_node);
        Ok(())
    }

    fn infer_cube_kind_from_neighbor(
        &self,
        infer_from: usize,
        other_node: usize,
        infer_cube_kind: CubeKind,
    ) -> Result<CubeKind, ZXError> {
        let edge_direction = self
            .direction_between(infer_from, other_node)?
            .as_udirection()
            .index();
        let edge = self.get_edge(infer_from, other_node);
        let mut bases = infer_cube_kind.bases();
        for (index, basis) in bases.iter_mut().enumerate() {
            if index != edge_direction && edge.hadamard {
                *basis = basis.flip();
            }
        }
        bases[edge_direction] = spider_incident_basis(self.nodes[other_node].kind);
        CubeKind::try_from(bases).map_err(|e| ZXError::InvalidInferredCubeKind {
            pos: self.nodes[other_node].pos,
            source: e,
        })
    }

    fn cube_block(&self, node_id: usize, kind: CubeKind) -> Block {
        Block::new(self.nodes[node_id].pos, BlockKind::Cube(kind))
    }

    fn special_block(&self, node_id: usize) -> Block {
        let node = self.nodes[node_id];
        let kind = match node.kind {
            NodeKind::Y => BlockKind::Y,
            NodeKind::T => BlockKind::T,
            NodeKind::Port => BlockKind::Port,
            NodeKind::Selective(k) => BlockKind::Selective(k),
            NodeKind::X | NodeKind::Z => unreachable!("X/Z nodes are synthesized as cubes"),
        };
        let block = Block::new(node.pos, kind);
        if kind.is_port() {
            block
                .with_port_role(node.role)
                .expect("Port nodes accept Port roles")
        } else {
            block
        }
    }

    fn add_pipe_between_nodes(
        &self,
        graph: &mut BlockGraph,
        src_node: usize,
        dst_node: usize,
    ) -> Result<(), ZXError> {
        let direction = self.direction_between(src_node, dst_node)?;
        let edge = self.get_edge(src_node, dst_node);
        let pipe = if edge.hadamard {
            Pipe::new(self.nodes[src_node].pos, direction).with_hadamard()
        } else {
            Pipe::new(self.nodes[src_node].pos, direction)
        };
        graph.add_pipe(pipe);
        Ok(())
    }

    fn sorted_undirected_edges(&self) -> Vec<(usize, usize)> {
        let mut edges: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|edge| edge.n1 < edge.n2)
            .map(|edge| (edge.n1, edge.n2))
            .collect();
        edges.sort_by_key(|&(n1, n2)| {
            (self.nodes[n1].pos.to_array(), self.nodes[n2].pos.to_array())
        });
        edges
    }

    fn axis_directions(&self, node_id: usize) -> HashSet<UDirection> {
        self.neighbor_ids(node_id)
            .iter()
            .copied()
            .filter_map(|neighbor| self.direction_between(node_id, neighbor).ok())
            .map(Direction::as_udirection)
            .collect()
    }

    fn direction_between(&self, src_node: usize, dst_node: usize) -> Result<Direction, ZXError> {
        let src = self.nodes[src_node].pos;
        let dst = self.nodes[dst_node].pos;
        let delta = dst - src;
        Direction::try_from(delta).map_err(|_| ZXError::NonNeighboringZXEdge { src, dst })
    }
}

fn is_zx_spider(kind: NodeKind) -> bool {
    matches!(kind, NodeKind::X | NodeKind::Z)
}

fn spider_normal_basis(kind: NodeKind) -> Basis {
    match kind {
        NodeKind::X => Basis::X,
        NodeKind::Z => Basis::Z,
        _ => unreachable!("only X/Z spiders have a spider normal basis"),
    }
}

fn spider_incident_basis(kind: NodeKind) -> Basis {
    spider_normal_basis(kind).flip()
}

fn block_to_node_kind(block: &Block) -> NodeKind {
    match block.kind {
        BlockKind::Cube(kind) => {
            if kind.is_x_type() {
                NodeKind::X
            } else {
                NodeKind::Z
            }
        }
        BlockKind::Walking(kind) => {
            let cube_kind = CubeKind::from(kind.boundary());
            if cube_kind.is_x_type() {
                NodeKind::X
            } else {
                NodeKind::Z
            }
        }
        BlockKind::PatchRotation(_) | BlockKind::Measurement(Basis::X) => NodeKind::Z,
        BlockKind::Measurement(Basis::Z) => NodeKind::X,
        BlockKind::Selective(kind) => NodeKind::Selective(kind),
        BlockKind::Y => NodeKind::Y,
        BlockKind::T => NodeKind::T,
        BlockKind::Port => NodeKind::Port,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use glam::ivec3;
    use rustc_hash::FxHashMap;

    use crate::{
        Action, ActionDag, Block, BlockGraph, BlockKind, CubeKind, Expr, GalleryItem,
        MeasureTarget, PatchRotationKind, SelectiveKind, WalkingBoundaryKind, WalkingKind,
    };
    use crate::{Basis, Direction, Pipe};

    use super::{CsrAdjacency, NodeKind, ZXEdge, ZXError, ZXGraph, ZXNode, block_to_node_kind};

    #[test]
    fn local_neighborhoods_match_full_tensors_and_crossing_reconstruction() {
        use crate::{Pauli, PauliString};

        for gallery in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::THTH,
            GalleryItem::GHZSlideThenGlide,
            GalleryItem::GHZPatchRotations,
            GalleryItem::CCZInjectedAnd,
            GalleryItem::Stability,
        ] {
            // The native caller receives this normalized, validated projection.
            let source = gallery
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection")
                .project_branches_deferred([])
                .unwrap();
            let full = ZXGraph::from_block_graph_for_analysis(&source).unwrap();
            for center in full.nodes() {
                let local = ZXGraph::from_local_block_neighborhood(&source, center.pos).unwrap();
                let local_center = local.node_at(center.pos).unwrap();
                assert_eq!(local_center.kind, center.kind);
                assert_eq!(local_center.role, center.role);
                assert_eq!(
                    local_center.is_input_port(&local),
                    center.is_input_port(&full)
                );
                assert_eq!(
                    local_center.is_output_port(&local),
                    center.is_output_port(&full)
                );
                assert_eq!(
                    local.edges().len(),
                    full.neighbor_edges(center.id).count() * 2
                );
                assert!(local.nodes().len() <= local.edges().len() / 2 + 1);
                assert_eq!(local.action_graph().ordered_nodes().count(), 0);
                let full_column = |column: usize| {
                    if let Some(node) = local.nodes().get(column) {
                        let global = full.node_at(node.pos).unwrap();
                        assert_eq!(node.kind, global.kind);
                        assert_eq!(node.role, global.role);
                        global.id
                    } else {
                        let edge = local.edge_by_id(column);
                        let a = full.node_at(local.nodes()[edge.n1].pos).unwrap().id;
                        let b = full.node_at(local.nodes()[edge.n2].pos).unwrap().id;
                        assert_eq!(edge.n1 > edge.n2, a > b);
                        let global = full.edge_between(a, b).unwrap();
                        assert_eq!(edge.hadamard, global.hadamard);
                        global.id
                    }
                };
                for direction in std::iter::once(None).chain(Direction::iter().map(Some)) {
                    for pauli in [
                        crate::PauliBasis::X,
                        crate::PauliBasis::Y,
                        crate::PauliBasis::Z,
                    ] {
                        let target = crate::FeedbackTarget {
                            target: center.pos,
                            pauli,
                            direction,
                        };
                        assert_eq!(
                            local
                                .feedback_column(&target)
                                .map(|(column, pauli)| (full_column(column), pauli)),
                            full.feedback_column(&target),
                        );
                    }
                }
                for direction in [
                    Direction::XMINUS,
                    Direction::XPLUS,
                    Direction::YMINUS,
                    Direction::YPLUS,
                    Direction::ZMINUS,
                    Direction::ZPLUS,
                ] {
                    let target = MeasureTarget::Edge {
                        src: center.pos,
                        dir: direction,
                    };
                    assert_eq!(
                        local.measurement_column(&target).map(full_column),
                        full.measurement_column(&target)
                    );
                }
                let terms = local
                    .local_stabilizer_flow_supports_for_node_kind(
                        local_center.id,
                        local_center.kind,
                    )
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|(column, pauli)| (full_column(column), pauli))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    terms,
                    full.local_stabilizer_flow_supports_for_node_kind(center.id, center.kind)
                );
                for seed in 0..16 {
                    let axes = [Pauli::I, Pauli::X, Pauli::Z, Pauli::Y];
                    let mut local_row = PauliString::new(local.total_ids());
                    local_row.set(local_center.id, axes[seed % 4]);
                    for (index, edge) in local
                        .edges()
                        .iter()
                        .filter(|edge| edge.n1 < edge.n2)
                        .enumerate()
                    {
                        let pauli = axes[(seed / 4 + index * seed) % 4];
                        local_row.set(edge.id, pauli);
                        local_row.set(local.edge_id(edge.n2, edge.n1).unwrap(), pauli);
                    }
                    let mut full_row = PauliString::new(full.total_ids());
                    for (column, pauli) in local_row.iter_support() {
                        full_row.set(full_column(column), pauli);
                    }
                    let actual = local.materialize_stabilizer_with_sign(local_row, false);
                    let expected = full.materialize_stabilizer_with_sign(full_row, false);
                    assert_eq!(actual.interior_nodes, expected.interior_nodes);
                    assert_eq!(actual.port_stabilizer, expected.port_stabilizer);
                    assert_eq!(actual.interior_edges, expected.interior_edges);
                }
            }
        }
    }

    #[test]
    fn local_neighborhood_ignores_unrelated_blocks_and_keeps_extended_anchors() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, glam::IVec2::X).unwrap();
        let start = ivec3(0, 0, 0);
        graph.add_block(Block::new(start, BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(ivec3(0, 0, -1), BlockKind::Port));
        graph.add_block(Block::new(ivec3(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(start, Direction::ZMINUS));
        graph.add_pipe(Pipe::new(kind.end_position(start), Direction::ZPLUS));
        for x in 10..1010 {
            graph.add_block(Block::new(ivec3(x, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        }
        let graph = graph.project_branches_deferred([]).unwrap();
        let local = ZXGraph::from_local_block_neighborhood(&graph, start).unwrap();
        assert_eq!(local.nodes().len(), 3);
        assert_eq!(local.total_ids(), 7);
        assert!(local.node_at(kind.end_position(start)).is_none());
        let center = local.node_at(start).unwrap().id;
        let before = local.node_at(ivec3(0, 0, -1)).unwrap().id;
        let after = local.node_at(ivec3(1, 0, 2)).unwrap().id;
        assert!(!local.edge_between(center, before).unwrap().hadamard);
        assert!(local.edge_between(center, after).is_some());
        assert!(before < center && center < after);
        let target = crate::FeedbackTarget {
            target: start,
            pauli: crate::PauliBasis::X,
            direction: None,
        };
        assert_eq!(local.feedback_edge(&target).unwrap().n2, after);
    }

    #[test]
    fn fixed_measurements_lower_to_concrete_zx_caps() {
        assert_eq!(
            block_to_node_kind(&Block::new(
                ivec3(0, 0, 0),
                BlockKind::Measurement(Basis::X),
            )),
            NodeKind::Z,
        );
        assert_eq!(
            block_to_node_kind(&Block::new(
                ivec3(0, 0, 0),
                BlockKind::Measurement(Basis::Z),
            )),
            NodeKind::X,
        );
    }

    fn restore_selectives(graph: &mut BlockGraph, originals: &HashMap<Block, Block>) {
        for (original, resolved) in originals {
            graph
                .get_block_mut(resolved.pos)
                .expect("resolved block must exist in graph")
                .kind = original.kind;
        }
    }

    #[test]
    fn patch_rotation_lowers_to_single_z_node() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, glam::IVec2::X).unwrap();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(ivec3(0, 0, -1), BlockKind::Port));
        graph.add_block(Block::new(ivec3(1, 0, 2), BlockKind::Port));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::ZMINUS));
        graph.add_pipe(Pipe::new(
            kind.end_position(ivec3(0, 0, 0)),
            Direction::ZPLUS,
        ));

        let zx = ZXGraph::try_from(&graph).expect("patch rotation graph converts to ZX");
        let node = zx
            .nodes
            .iter()
            .find(|node| node.pos == ivec3(0, 0, 0))
            .expect("patch rotation node exists");

        assert_eq!(node.kind, NodeKind::Z);
    }

    fn assert_roundtrip_matches(graph: &BlockGraph) {
        let canonical = graph.fix_shadowed_faces();
        let (concretized, selective_blocks) = canonical
            .randomly_resolve_selectives(rand::random())
            .unwrap();
        let zx = ZXGraph::try_from(&concretized).expect("concretized graph converts to ZX");
        let mut rebuilt = BlockGraph::try_from(&zx).expect("ZX graph converts back to BlockGraph");
        restore_selectives(&mut rebuilt, &selective_blocks);

        assert_eq!(rebuilt.block_count(), canonical.block_count());
        assert_eq!(rebuilt.pipe_count(), canonical.pipe_count());
        for block in canonical.blocks() {
            let rebuilt_block = rebuilt
                .get_block(block.pos)
                .expect("rebuilt graph must contain all original blocks");
            assert_eq!(
                rebuilt_block.kind, block.kind,
                "block kind mismatch at {}",
                block.pos
            );
        }
        for pipe in canonical.pipes() {
            let rebuilt_pipe = rebuilt
                .get_pipe(pipe.src, pipe.dst())
                .expect("rebuilt graph must contain all original pipes");
            assert_eq!(
                rebuilt_pipe.hadamard,
                pipe.hadamard,
                "pipe hadamard mismatch at {} -> {}",
                pipe.src,
                pipe.dst()
            );
        }
    }

    #[test]
    fn to_zx_graph_preserves_action_dag_metadata() {
        let graph = crate::GalleryItem::T
            .build()
            .materialize_root_graph()
            .expect("gallery flat projection");
        let zx = graph.to_zx_graph().unwrap();

        let block_measurements = graph
            .action_graph()
            .ordered_nodes()
            .map(|node| node.measurement)
            .collect::<Vec<_>>();
        let zx_measurements = zx
            .action_graph()
            .ordered_nodes()
            .map(|node| node.measurement)
            .collect::<Vec<_>>();

        assert_eq!(zx_measurements, block_measurements);
    }

    #[test]
    fn block_graph_roundtrip_preserves_non_measure_actions() {
        use glam::ivec3;

        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph
            .set_actions(vec![
                Action::Measure {
                    target: MeasureTarget::Node(ivec3(0, 0, 0)),
                    name: "m0".into(),
                },
                Action::Let {
                    name: "alias".into(),
                    expr: Expr::Var("m0".into()),
                },
                Action::DiscardIf(Expr::Var("alias".into())),
            ])
            .unwrap();

        let zx = graph.to_zx_graph().unwrap();
        assert_eq!(zx.actions(), graph.actions());

        let rebuilt = BlockGraph::try_from(&zx).unwrap();
        assert_eq!(rebuilt.actions(), graph.actions());
    }

    #[test]
    fn walking_block_converts_to_single_zx_node_from_boundary_kind() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(
            ivec3(0, 0, 0),
            BlockKind::Walking(
                WalkingKind::new(WalkingBoundaryKind::XZZ, glam::IVec2::new(1, 1)).unwrap(),
            ),
        ));

        let zx = graph.to_zx_graph().unwrap();

        assert_eq!(zx.nodes().len(), 1);
        assert_eq!(zx.nodes()[0].kind, NodeKind::X);
        assert_eq!(zx.nodes()[0].pos, ivec3(0, 0, 0));
    }

    #[test]
    fn test_zx_to_block_graph_roundtrip_for_all_galleries() {
        for gallery in [
            GalleryItem::CNOT,
            GalleryItem::CZSpatialH,
            GalleryItem::CZTemporalH,
            GalleryItem::S,
            GalleryItem::TWithPreparedY,
            GalleryItem::CCZFactoryWithTels,
            GalleryItem::THTH,
        ] {
            let graph = gallery
                .build()
                .materialize_root_graph()
                .expect("gallery flat projection");
            assert_roundtrip_matches(&graph);
        }
    }

    #[test]
    fn test_selective_boundary_nodes_must_be_timelike() {
        let zx = ZXGraph {
            nodes: vec![
                ZXNode::new(0, ivec3(0, 0, 0), NodeKind::Selective(SelectiveKind::XZ)),
                ZXNode::new(1, ivec3(1, 0, 0), NodeKind::X),
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
            action_graph: crate::ActionDag::default(),
            total_ids: 4,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        };

        let err = zx
            .to_block_graph()
            .expect_err("spacelike selective should be rejected");
        assert!(matches!(
            err,
            ZXError::SpecialNodeNotTimeLike {
                pos,
                kind: NodeKind::Selective(_)
            } if pos == ivec3(0, 0, 0)
        ));
    }

    #[test]
    fn validate_for_program_rejects_non_selective_resolve_targets() {
        let zx = ZXGraph {
            nodes: vec![ZXNode::new(0, ivec3(0, 0, 0), NodeKind::X)],
            edges: vec![],
            adjacency: CsrAdjacency::from_edges(1, &[]),
            pos_to_node: FxHashMap::from_iter([(ivec3(0, 0, 0), 0)]),
            action_graph: ActionDag::from_actions(&[Action::Resolve {
                target: ivec3(0, 0, 0),
                condition: Expr::Var("m0".into()),
            }]),
            total_ids: 1,
            cross_incident: Default::default(),
            stabilizer_phase_basis: Default::default(),
        };

        let err = zx
            .validate_for_program()
            .expect_err("non-selective resolve target should be rejected");
        assert!(matches!(
            err,
            ZXError::InvalidResolveTarget { pos } if pos == ivec3(0, 0, 0)
        ));
    }
}
