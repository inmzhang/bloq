use std::collections::BTreeMap;

use bloq_graph::{Basis, BlockGraph, BlockKind, Pipe, UDirection, WalkingKind};
use glam::IVec3;
use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;

use bloq_ir::{NodeProvenance, SourceBlockRef, TemporalPipeRef};

use crate::spatial_port::{SpatialPortExpansion, SpatialPortExpansionMap};

#[derive(Debug, Clone, Default)]
pub(crate) struct LowerPlan {
    graph: DiGraph<NodePlan, EdgePlan>,
    node_by_block: crate::FxMap<IVec3, NodeIndex>,
    node_by_temporal_pipe: crate::FxMap<TemporalPipeRef, NodeIndex>,
    node_by_spatial_port: crate::FxMap<IVec3, NodeIndex>,
}

/// Immutable records emitted by module-object instantiation. Building these
/// records may inspect linked source topology; assembling them into plan nodes
/// is pure link algebra and never calls a compiler phase.
#[derive(Debug, Clone)]
pub(crate) struct LinkedPlanInput {
    blocks: Vec<IVec3>,
    walking: crate::FxMap<IVec3, WalkingKind>,
    pipes: Vec<LinkedPipePlan>,
    spatial_ports: Vec<SpatialPortExpansion>,
}

#[derive(Debug, Clone)]
struct LinkedPipePlan {
    pipe: Pipe,
    owners: (IVec3, IVec3),
    template: Option<TemplatePlan>,
}

#[derive(Debug, Clone)]
pub(crate) struct NodePlan {
    pub(crate) layer: i64,
    /// Whether the complete block component has an incoming quantum cut.
    /// Local physical contexts retain this global timeline fact explicitly.
    pub(crate) incoming_cut: bool,
    pub(crate) provenance: NodeProvenance,
    pub(crate) template: Option<TemplatePlan>,
    /// Spatial Hadamard walls standing inside this block component. Each is an
    /// extra template instance on the node, not a node of its own: the wall
    /// shares data qubits with both endpoint cubes, so it has to merge into the
    /// same round they do.
    pub(crate) walls: Vec<SpatialPipeRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TemplatePlan {
    pub(crate) top_basis: Basis,
    pub(crate) origin: IVec3,
}

/// A spatial Hadamard pipe, oriented from its smaller endpoint position so that
/// `src` is the wall's negative-axis cube — the one whose local coordinates the
/// wall template is built in, and at whose offset its instance is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpatialPipeRef {
    pub(crate) src: IVec3,
    pub(crate) dst: IVec3,
}

impl SpatialPipeRef {
    pub(super) fn new(src: IVec3, dst: IVec3) -> Self {
        if src.to_array() <= dst.to_array() {
            Self { src, dst }
        } else {
            Self { src: dst, dst: src }
        }
    }

    fn sort_key(self) -> ([i32; 3], [i32; 3]) {
        (self.src.to_array(), self.dst.to_array())
    }

    pub(crate) fn axis(self) -> UDirection {
        match (self.dst.x - self.src.x, self.dst.y - self.src.y) {
            (delta, 0) if delta > 0 => UDirection::X,
            (0, delta) if delta > 0 => UDirection::Y,
            _ => unreachable!("spatial Hadamard walls run along +X or +Y"),
        }
    }

    pub(crate) fn arm_directions(self) -> (bloq_graph::Direction, bloq_graph::Direction) {
        match self.axis() {
            UDirection::X => (bloq_graph::Direction::XMINUS, bloq_graph::Direction::XPLUS),
            UDirection::Y => (bloq_graph::Direction::YMINUS, bloq_graph::Direction::YPLUS),
            UDirection::Z => unreachable!("spatial Hadamard walls are not temporal"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct EdgePlan {
    pub(crate) pipes: Vec<TemporalPipeRef>,
    pub(crate) spatial_ports: Vec<SpatialPortExpansion>,
}

impl EdgePlan {
    pub(crate) fn temporal_pipes(&self) -> impl Iterator<Item = TemporalPipeRef> + '_ {
        self.pipes
            .iter()
            .copied()
            .chain(self.spatial_ports.iter().map(|port| TemporalPipeRef {
                // Both virtual halves share the authored spatial Port.
                src: port.source,
                dst: port.source,
                hadamard: false,
            }))
    }
}

impl LinkedPlanInput {
    pub(crate) fn from_module_graph(
        source: &BlockGraph,
        spatial_ports: &SpatialPortExpansionMap,
        linked_positions: impl IntoIterator<Item = IVec3>,
    ) -> Self {
        let linked_positions = linked_positions.into_iter().collect::<crate::FxSet<_>>();
        debug_assert!(
            linked_positions
                .iter()
                .all(|&position| source.get_block(position).is_some())
        );

        let mut seen = crate::FxSet::default();
        let mut pipes = linked_positions
            .iter()
            .flat_map(|&position| source.pipes_at(position))
            .filter(|pipe| seen.insert(*pipe))
            .filter_map(|pipe| {
                let (src, dst) = pipe.endpoints();
                let owners = (
                    source
                        .get_endpoint_block(src)
                        .expect("pipe source endpoint is owned by a block")
                        .pos(),
                    source
                        .get_endpoint_block(dst)
                        .expect("pipe target endpoint is owned by a block")
                        .pos(),
                );
                if !linked_positions.contains(&owners.0) || !linked_positions.contains(&owners.1) {
                    return None;
                }
                let pipe_ref = temporal_pipe_ref(pipe);
                let template =
                    (!pipe.dir().is_spatial() && pipe.is_hadamard()).then(|| TemplatePlan {
                        top_basis: temporal_hadamard_top_basis(source, &pipe_ref),
                        origin: temporal_hadamard_origin(&pipe_ref),
                    });
                Some(LinkedPipePlan {
                    pipe: pipe.clone(),
                    owners,
                    template,
                })
            })
            .collect::<Vec<_>>();
        pipes.sort_unstable_by_key(|record| {
            let (left, right) = record.pipe.endpoints();
            let (left, right) = if left.to_array() <= right.to_array() {
                (left, right)
            } else {
                (right, left)
            };
            (left.to_array(), right.to_array(), record.pipe.is_hadamard())
        });
        let mut spatial_ports = spatial_ports
            .values()
            .filter(|port| linked_positions.contains(&port.source))
            .copied()
            .collect::<Vec<_>>();
        spatial_ports.sort_unstable_by_key(|port| port.source.to_array());
        let mut blocks = linked_positions.into_iter().collect::<Vec<_>>();
        blocks.sort_unstable_by_key(glam::IVec3::to_array);
        let walking = blocks
            .iter()
            .filter_map(|&position| match source.get_block(position)?.kind() {
                BlockKind::Walking(kind) => Some((position, kind)),
                _ => None,
            })
            .collect();
        Self {
            blocks,
            walking,
            pipes,
            spatial_ports,
        }
    }

    pub(crate) fn temporal_templates(&self) -> impl Iterator<Item = TemplatePlan> + '_ {
        self.pipes.iter().filter_map(|record| record.template)
    }

    pub(crate) fn spatial_walls(&self) -> impl Iterator<Item = SpatialPipeRef> + '_ {
        self.pipes
            .iter()
            .filter(|record| record.pipe.dir().is_spatial() && record.pipe.is_hadamard())
            .map(|record| {
                let (src, dst) = record.pipe.endpoints();
                SpatialPipeRef::new(src, dst)
            })
    }
}

impl LowerPlan {
    #[cfg(test)]
    pub(crate) fn from_block_graph(source: &BlockGraph) -> Self {
        let input = LinkedPlanInput::from_module_graph(
            source,
            &SpatialPortExpansionMap::default(),
            source.blocks().map(bloq_graph::Block::pos),
        );
        Self::from_linked_input(&input)
    }

    pub(crate) fn from_linked_input(source: &LinkedPlanInput) -> Self {
        let mut layers: BTreeMap<i32, Vec<IVec3>> = BTreeMap::new();
        for &pos in &source.blocks {
            layers.entry(pos.z).or_default().push(pos);
        }
        let mut spatial_pipes_by_layer: BTreeMap<i32, Vec<&Pipe>> = BTreeMap::new();
        for pipe in source
            .pipes
            .iter()
            .filter(|record| record.pipe.dir().is_spatial())
            .map(|record| &record.pipe)
        {
            spatial_pipes_by_layer
                .entry(pipe.src().z)
                .or_default()
                .push(pipe);
        }

        let mut graph = DiGraph::new();
        let mut node_by_block = crate::FxMap::default();
        let mut node_by_temporal_pipe = crate::FxMap::default();
        let mut node_by_spatial_port = crate::FxMap::default();
        for (layer, mut block_positions) in layers {
            block_positions.sort_by_key(glam::IVec3::to_array);
            let spatial_pipes = spatial_pipes_by_layer
                .get(&layer)
                .map(Vec::as_slice)
                .unwrap_or_default();
            for component in block_components(&block_positions, spatial_pipes, &source.walking) {
                let members = component
                    .iter()
                    .copied()
                    .map(|pos| SourceBlockRef { pos })
                    .collect::<Vec<_>>();
                debug_assert!(members.windows(2).all(|pair| pair[0] <= pair[1]));
                let node = graph.add_node(NodePlan {
                    layer: 2 * i64::from(layer),
                    incoming_cut: false,
                    provenance: NodeProvenance::BlockComponent { members },
                    template: None,
                    walls: Vec::new(),
                });
                for pos in component {
                    node_by_block.insert(pos, node);
                }
            }
        }

        // Use raw pipe endpoints: a virtual endpoint's owner may be in this
        // component even though the wall itself is not part of it.
        for wall in source.spatial_walls() {
            if let (Some(&src), Some(&dst)) =
                (node_by_block.get(&wall.src), node_by_block.get(&wall.dst))
                && src == dst
            {
                graph[src].walls.push(wall);
            }
        }
        for node in graph.node_weights_mut() {
            node.walls.sort_unstable_by_key(|wall| wall.sort_key());
        }

        let mut edge_by_endpoints = crate::FxMap::default();
        for record in source
            .pipes
            .iter()
            .filter(|record| !record.pipe.dir().is_spatial())
        {
            let pipe = &record.pipe;
            let (src_owner, dst_owner) = record.owners;
            let src_node = node_by_block[&src_owner];
            let dst_node = node_by_block[&dst_owner];
            let (plan_source, plan_target) = temporal_edge_endpoints(&graph, src_node, dst_node);
            let pipe_ref = temporal_pipe_ref(pipe);

            if pipe.is_hadamard() {
                let template = record
                    .template
                    .expect("temporal Hadamard link record carries its template ABI");
                let pipe_node = graph.add_node(NodePlan {
                    layer: temporal_hadamard_layer(&pipe_ref),
                    incoming_cut: false,
                    provenance: NodeProvenance::TemporalPipe { pipe: pipe_ref },
                    template: Some(template),
                    walls: Vec::new(),
                });
                node_by_temporal_pipe.insert(pipe_ref, pipe_node);
                graph.add_edge(
                    plan_source,
                    pipe_node,
                    EdgePlan {
                        pipes: vec![pipe_ref],
                        spatial_ports: Vec::new(),
                    },
                );
                graph.add_edge(
                    pipe_node,
                    plan_target,
                    EdgePlan {
                        pipes: vec![pipe_ref],
                        spatial_ports: Vec::new(),
                    },
                );
                continue;
            }

            if let Some(&edge) = edge_by_endpoints.get(&(plan_source, plan_target)) {
                graph
                    .edge_weight_mut(edge)
                    .expect("edge endpoint map references inserted edge")
                    .pipes
                    .push(pipe_ref);
            } else {
                let edge = graph.add_edge(
                    plan_source,
                    plan_target,
                    EdgePlan {
                        pipes: vec![pipe_ref],
                        spatial_ports: Vec::new(),
                    },
                );
                edge_by_endpoints.insert((plan_source, plan_target), edge);
            }
        }

        for &port in &source.spatial_ports {
            let cube = node_by_block[&port.source];
            let virtual_port = graph.add_node(NodePlan {
                layer: 2 * i64::from(port.source.z) + if port.is_input() { -1 } else { 1 },
                incoming_cut: false,
                provenance: NodeProvenance::SpatialPortSubstitution {
                    source: port.source,
                    role: port.role,
                },
                template: None,
                walls: Vec::new(),
            });
            node_by_spatial_port.insert(port.source, virtual_port);
            let (source, target) = if port.is_input() {
                (virtual_port, cube)
            } else {
                (cube, virtual_port)
            };
            graph.add_edge(
                source,
                target,
                EdgePlan {
                    pipes: Vec::new(),
                    spatial_ports: vec![port],
                },
            );
        }

        // Canonicalize each edge's coalesced pipe list: authoring order of the
        // source pipes is not observable, and padding derives from this list in
        // order (WF-4 pairing), so sorting here makes both the seam and its
        // padding reproducible.
        for edge in graph.edge_weights_mut() {
            edge.pipes
                .sort_unstable_by_key(|pipe| (pipe.src.to_array(), pipe.dst.to_array()));
        }
        for node in graph.node_indices().collect::<Vec<_>>() {
            graph[node].incoming_cut = graph
                .neighbors_directed(node, petgraph::Direction::Incoming)
                .next()
                .is_some();
        }

        // The instance detector pass appends nodes layer-major and relies on
        // every plan edge ascending `layer` for that sort to be a topological
        // order (lower/detector.rs); pin the invariant where edges are built.
        debug_assert!(
            graph
                .edge_references()
                .all(|edge| graph[edge.source()].layer <= graph[edge.target()].layer),
            "plan edges must ascend layer: detector composition sorts layer-major"
        );

        Self {
            graph,
            node_by_block,
            node_by_temporal_pipe,
            node_by_spatial_port,
        }
    }

    pub(crate) fn graph(&self) -> &DiGraph<NodePlan, EdgePlan> {
        &self.graph
    }

    pub(crate) fn set_incoming_cuts(&mut self, incoming: &crate::FxMap<IVec3, bool>) {
        for (&position, &incoming) in incoming {
            if let Some(&node) = self.node_by_block.get(&position) {
                self.graph[node].incoming_cut = incoming;
            }
        }
    }

    pub(crate) fn node_by_block(&self) -> &crate::FxMap<IVec3, NodeIndex> {
        &self.node_by_block
    }

    pub(crate) fn node_by_temporal_pipe(&self) -> &crate::FxMap<TemporalPipeRef, NodeIndex> {
        &self.node_by_temporal_pipe
    }

    pub(crate) fn node_by_spatial_port(&self) -> &crate::FxMap<IVec3, NodeIndex> {
        &self.node_by_spatial_port
    }

    pub(crate) fn emission_order(&self) -> Vec<NodeIndex> {
        toposort(&self.graph, None).expect(
            "temporal edges point from lower to upper (layer, index), so the plan is acyclic",
        )
    }

    pub(crate) fn temporal_node_components(&self) -> Vec<Vec<NodeIndex>> {
        let mut visited = vec![false; self.graph.node_count()];
        let mut components = Vec::new();
        for root in self.graph.node_indices() {
            if visited[root.index()] {
                continue;
            }

            let mut stack = vec![root];
            visited[root.index()] = true;
            let mut component = Vec::new();
            while let Some(node) = stack.pop() {
                component.push(node);
                for neighbor in self
                    .graph
                    .neighbors_directed(node, petgraph::Outgoing)
                    .chain(self.graph.neighbors_directed(node, petgraph::Incoming))
                {
                    if !visited[neighbor.index()] {
                        visited[neighbor.index()] = true;
                        stack.push(neighbor);
                    }
                }
            }
            components.push(component);
        }

        components
    }
}

impl NodePlan {
    pub(crate) fn block_members(&self) -> &[SourceBlockRef] {
        match &self.provenance {
            NodeProvenance::BlockComponent { members } => members,
            // Only block/pipe provenance appears in a lowering plan
            // (MemoryPadding is spliced post-compile; classical provenance
            // never reaches a plan node), but the wildcard keeps this total.
            _ => &[],
        }
    }
}

fn temporal_edge_endpoints(
    graph: &DiGraph<NodePlan, EdgePlan>,
    src_node: NodeIndex,
    dst_node: NodeIndex,
) -> (NodeIndex, NodeIndex) {
    let src = &graph[src_node];
    let dst = &graph[dst_node];
    if (src.layer, src_node.index()) <= (dst.layer, dst_node.index()) {
        (src_node, dst_node)
    } else {
        (dst_node, src_node)
    }
}

fn block_components(
    block_positions: &[IVec3],
    spatial_pipes: &[&Pipe],
    walking: &crate::FxMap<IVec3, WalkingKind>,
) -> Vec<Vec<IVec3>> {
    let position_index: crate::FxMap<IVec3, usize> = block_positions
        .iter()
        .copied()
        .enumerate()
        .map(|(index, pos)| (pos, index))
        .collect();
    let mut components = UnionFind::new(block_positions.len());

    for pipe in spatial_pipes {
        let (src, dst) = pipe.endpoints();
        if let (Some(&src_index), Some(&dst_index)) =
            (position_index.get(&src), position_index.get(&dst))
        {
            components.union(src_index, dst_index);
        }
    }

    // Parallel walks share transit qubits and therefore one set of common rounds.
    for (index, &position) in block_positions.iter().enumerate() {
        let Some(kind) = walking.get(&position) else {
            continue;
        };
        for offset in kind
            .reserved_offsets()
            .into_iter()
            .filter(|offset| offset.z == 0 && *offset != IVec3::ZERO)
        {
            if let Ok(neighbor) = bloq_graph::checked_add_position(position, offset)
                && walking
                    .get(&neighbor)
                    .is_some_and(|other| other.movement() == kind.movement())
                && let Some(&neighbor_index) = position_index.get(&neighbor)
            {
                components.union(index, neighbor_index);
            }
        }
    }

    let mut grouped: BTreeMap<usize, Vec<IVec3>> = BTreeMap::new();
    for (index, &pos) in block_positions.iter().enumerate() {
        grouped.entry(components.find(index)).or_default().push(pos);
    }

    let mut components: Vec<_> = grouped.into_values().collect();
    components.sort_by_key(|component| component[0].to_array());
    components
}

/// A canonical [`TemporalPipeRef`] for `pipe`, oriented `src.z <= dst.z` so
/// authoring order (which endpoint the source `Pipe` names first) never leaks
/// into the compiled artifact. Every consumer that cares about time order reads
/// through this orientation or [`TemporalPipeRef::endpoints_by_z`]; keep this the
/// single construction point so map keys (`node_by_temporal_pipe`, pipe
/// templates) stay consistent — see [`super::observable::hadamard_temporal_pipe`].
fn temporal_pipe_ref(pipe: &Pipe) -> TemporalPipeRef {
    let (src, dst) = pipe.endpoints();
    let (src, dst) = if src.z <= dst.z {
        (src, dst)
    } else {
        (dst, src)
    };
    TemporalPipeRef {
        src,
        dst,
        hadamard: pipe.is_hadamard(),
    }
}

fn temporal_hadamard_layer(pipe: &TemporalPipeRef) -> i64 {
    2 * i64::from(pipe.src.z.min(pipe.dst.z)) + 1
}

fn temporal_hadamard_origin(pipe: &TemporalPipeRef) -> IVec3 {
    pipe.endpoints_by_z().0
}

fn temporal_hadamard_top_basis(graph: &BlockGraph, pipe: &TemporalPipeRef) -> Basis {
    let (below, _) = pipe.endpoints_by_z();
    let source = graph
        .get_pipe(pipe.src, pipe.dst)
        .expect("temporal hadamard comes from a source pipe");
    graph.infer_pipe_basis_from_endpoint(source, below)[UDirection::Y.index()]
        .expect("validated temporal pipe has a transverse basis")
}

#[cfg(test)]
mod tests {
    use bloq_graph::{Block, CubeKind, Direction, PatchRotationKind, WalkingBoundaryKind};
    use glam::ivec3;

    use super::*;

    #[test]
    fn linked_positions_select_plan_records() {
        let kept = ivec3(0, 0, 0);
        let omitted = ivec3(1, 0, 0);
        let upper = ivec3(0, 0, 1);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(kept, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(omitted, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(upper, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(kept, Direction::XPLUS));
        graph.add_pipe(Pipe::new(kept, Direction::ZPLUS).with_hadamard());

        let input = LinkedPlanInput::from_module_graph(
            &graph,
            &SpatialPortExpansionMap::default(),
            [kept, upper, kept],
        );
        assert_eq!(input.pipes.len(), 1);
        assert!(input.pipes[0].pipe.is_hadamard());
        let plan = LowerPlan::from_linked_input(&input);

        assert!(plan.node_by_block().contains_key(&kept));
        assert!(plan.node_by_block().contains_key(&upper));
        assert!(!plan.node_by_block().contains_key(&omitted));
        assert_eq!(plan.node_by_temporal_pipe().len(), 1);
    }

    #[test]
    fn groups_spatial_components_and_temporal_edges() {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(1, 0, 0), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(0, 0, 1), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(ivec3(1, 0, 1), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::XPLUS));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 1), Direction::XPLUS));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::ZPLUS));
        graph.add_pipe(Pipe::new(ivec3(1, 0, 1), Direction::ZMINUS));

        let plan = LowerPlan::from_block_graph(&graph);

        assert_eq!(plan.graph().node_count(), 2);
        assert_eq!(
            plan.graph()[NodeIndex::new(0)].block_members(),
            vec![
                SourceBlockRef {
                    pos: ivec3(0, 0, 0)
                },
                SourceBlockRef {
                    pos: ivec3(1, 0, 0)
                },
            ]
            .as_slice()
        );
        assert_eq!(plan.graph().edge_count(), 1);
        let edge = plan.graph().edge_references().next().unwrap();
        assert_eq!(
            (edge.source(), edge.target()),
            (NodeIndex::new(0), NodeIndex::new(1))
        );
        assert_eq!(edge.weight().pipes.len(), 2);
    }

    #[test]
    fn spatial_walls_use_raw_component_endpoints_and_canonical_order() {
        let h = |x, owners| LinkedPipePlan {
            pipe: Pipe::new(ivec3(x, 0, 0), Direction::XPLUS).with_hadamard(),
            owners,
            template: None,
        };
        let input = LinkedPlanInput {
            blocks: [0, 1, 2, 3, 10, 11].map(|x| ivec3(x, 0, 0)).to_vec(),
            walking: crate::FxMap::default(),
            pipes: vec![
                h(2, (ivec3(2, 0, 0), ivec3(3, 0, 0))),
                h(10, (ivec3(10, 0, 0), ivec3(11, 0, 0))),
                // Both owners are in the first component, but neither raw
                // endpoint is a member. This is not a component wall.
                h(8, (ivec3(0, 0, 0), ivec3(1, 0, 0))),
                h(1, (ivec3(1, 0, 0), ivec3(2, 0, 0))),
                h(0, (ivec3(0, 0, 0), ivec3(1, 0, 0))),
            ],
            spatial_ports: Vec::new(),
        };
        let plan = LowerPlan::from_linked_input(&input);
        let first = plan.node_by_block()[&ivec3(0, 0, 0)];
        let second = plan.node_by_block()[&ivec3(10, 0, 0)];
        assert_ne!(first, second);
        assert_eq!(plan.graph().node_count(), 2);
        assert_eq!(
            plan.graph()[first].walls,
            [0, 1, 2].map(|x| SpatialPipeRef::new(ivec3(x, 0, 0), ivec3(x + 1, 0, 0)))
        );
        assert_eq!(
            plan.graph()[second].walls,
            [SpatialPipeRef::new(ivec3(10, 0, 0), ivec3(11, 0, 0))]
        );
    }

    #[test]
    fn temporal_edges_resolve_walking_virtual_endpoints_to_owner_node() {
        let mut graph = BlockGraph::new();
        let walking = WalkingKind::new(WalkingBoundaryKind::ZXZ, glam::ivec2(1, 1)).unwrap();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::Walking(walking)));
        graph.add_block(Block::new(ivec3(1, 1, 2), BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(
            walking.end_position(ivec3(0, 0, 0)),
            Direction::ZPLUS,
        ));

        let plan = LowerPlan::from_block_graph(&graph);

        assert_eq!(plan.graph().node_count(), 2);
        assert_eq!(plan.graph().edge_count(), 1);
        let edge = plan.graph().edge_references().next().unwrap();
        assert_eq!(edge.weight().pipes.len(), 1);
        assert_eq!(edge.weight().pipes[0].src, ivec3(1, 1, 1));
        assert_eq!(edge.weight().pipes[0].dst, ivec3(1, 1, 2));
    }

    #[test]
    fn temporal_edges_resolve_patch_rotation_virtual_endpoints_to_owner_node() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, glam::ivec2(1, 0)).unwrap();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(ivec3(1, 0, 2), BlockKind::Port));
        graph.add_block(Block::new(ivec3(0, 0, -1), BlockKind::Port));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::ZMINUS));
        graph.add_pipe(Pipe::new(
            kind.end_position(ivec3(0, 0, 0)),
            Direction::ZPLUS,
        ));

        let plan = LowerPlan::from_block_graph(&graph);

        assert_eq!(plan.graph().node_count(), 3);
        assert_eq!(plan.graph().edge_count(), 2);
        let patch_node = plan.node_by_block()[&ivec3(0, 0, 0)];
        let future_edge = plan
            .graph()
            .edges(patch_node)
            .find(|edge| edge.weight().pipes[0].dst == ivec3(1, 0, 2))
            .expect("future patch rotation temporal pipe is planned");
        assert_eq!(future_edge.weight().pipes[0].src, ivec3(1, 0, 1));
    }

    #[test]
    fn temporal_hadamard_after_patch_rotation_uses_pipe_endpoint_basis() {
        let mut graph = BlockGraph::new();
        let kind = PatchRotationKind::new(Basis::X, glam::ivec2(1, 0)).unwrap();
        graph.add_block(Block::new(ivec3(0, 0, 0), BlockKind::PatchRotation(kind)));
        graph.add_block(Block::new(ivec3(0, 0, -1), BlockKind::Port));
        graph.add_block(Block::new(ivec3(1, 0, 2), BlockKind::Cube(CubeKind::XZZ)));
        graph.add_pipe(Pipe::new(ivec3(0, 0, 0), Direction::ZMINUS));
        graph.add_pipe(
            Pipe::new(kind.end_position(ivec3(0, 0, 0)), Direction::ZPLUS).with_hadamard(),
        );

        let pipe = TemporalPipeRef {
            src: ivec3(1, 0, 1),
            dst: ivec3(1, 0, 2),
            hadamard: true,
        };

        assert_eq!(temporal_hadamard_top_basis(&graph, &pipe), Basis::X);
        crate::compile(&graph, 3)
            .expect("patch rotation -> H -> XZZ compiles")
            .validate()
            .expect("compiled Bloq validates");
    }

    #[test]
    fn high_z_temporal_hadamard_stays_between_its_endpoints() {
        let lower_pos = ivec3(0, 0, i32::MAX - 1);
        let upper_pos = ivec3(0, 0, i32::MAX);
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(lower_pos, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(upper_pos, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(lower_pos, Direction::ZPLUS).with_hadamard());

        let plan = LowerPlan::from_block_graph(&graph);
        let pipe = plan
            .node_by_temporal_pipe()
            .values()
            .next()
            .copied()
            .unwrap();
        let lower = plan.node_by_block()[&lower_pos];
        let upper = plan.node_by_block()[&upper_pos];

        assert!(plan.graph()[lower].layer < plan.graph()[pipe].layer);
        assert!(plan.graph()[pipe].layer < plan.graph()[upper].layer);
        assert_eq!(plan.graph()[upper].layer, 2 * i64::from(i32::MAX));
    }
}
