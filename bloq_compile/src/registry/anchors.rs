//! Global dynamic connectivity and T lineage for bounded physical gateways.

use std::collections::{BTreeMap, VecDeque};

use bloq_graph::{
    Block, BlockGraph, BlockKind, Direction, GuardedTopology, ModuleCertificationLimits,
};
use bloq_utils::boolean::{
    BooleanDecisionDiagram, BooleanOp, DECISION_FALSE as ZERO, DECISION_TRUE as ONE, DecisionId,
};
use glam::IVec3;

use crate::signature::BlockSignature;
use crate::{CompileError, FxMap, add_resource, check_resource};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DynamicAnchor {
    pub touches_dynamic: bool,
    pub source_t: Option<(IVec3, bool)>,
}

struct GuardedAnchor {
    dynamic: DecisionId,
    sources: Vec<((IVec3, bool), DecisionId)>,
}

pub(super) struct AnchorIncidence {
    positions: Vec<IVec3>,
    indices: FxMap<IVec3, usize>,
    edges: FxMap<(usize, usize), DecisionId>,
    descents: FxMap<(usize, usize, bool), DecisionId>,
}

impl AnchorIncidence {
    pub(super) fn new(
        positions: Vec<IVec3>,
        diagram: &mut BooleanDecisionDiagram,
    ) -> Result<Self, CompileError> {
        diagram.charge(positions.len().saturating_mul(8))?;
        let indices = positions
            .iter()
            .enumerate()
            .map(|(index, &position)| (position, index))
            .collect();
        Ok(Self {
            positions,
            indices,
            edges: FxMap::default(),
            descents: FxMap::default(),
        })
    }

    pub(super) fn add_variant(
        &mut self,
        diagram: &mut BooleanDecisionDiagram,
        position: IVec3,
        graph: &BlockGraph,
        guard: DecisionId,
    ) -> Result<(), CompileError> {
        diagram.charge(7)?;
        let index = self.indices[&position];
        let block = graph.get_block(position).expect("local center exists");
        let connectivity = BlockSignature::graph_connectivity(block, graph);
        for direction in Direction::iter() {
            let Some((neighbor, hadamard)) =
                crate::lower::physical_neighbor(graph, block, connectivity, direction)
            else {
                continue;
            };
            let neighbor_index = self.indices[&neighbor.pos()];
            let value = self.edges.entry((index, neighbor_index)).or_insert(ZERO);
            *value = diagram.apply(BooleanOp::Or, *value, guard)?;
            if direction == Direction::ZMINUS {
                debug_assert!(neighbor.pos().z < position.z);
                let value = self
                    .descents
                    .entry((index, neighbor_index, hadamard))
                    .or_insert(ZERO);
                *value = diagram.apply(BooleanOp::Or, *value, guard)?;
            }
        }
        Ok(())
    }
}

/// One source-wide symbolic index; no selected full graph is retained here.
pub(crate) struct GuardedDynamicAnchors {
    sites: FxMap<IVec3, GuardedAnchor>,
    case_limit: usize,
}

impl GuardedDynamicAnchors {
    pub(crate) fn decisions_mut(&mut self) -> impl Iterator<Item = &mut DecisionId> {
        self.sites.values_mut().flat_map(|site| {
            std::iter::once(&mut site.dynamic)
                .chain(site.sources.iter_mut().map(|(_, guard)| guard))
        })
    }

    pub(crate) fn new(
        topology: &mut GuardedTopology,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, CompileError> {
        let positions = topology
            .sites
            .keys()
            .copied()
            .map(IVec3::from_array)
            .collect::<Vec<_>>();
        let mut incidence = AnchorIncidence::new(positions, &mut topology.diagram)?;
        for (&position, variants) in &topology.sites {
            let position = IVec3::from_array(position);
            for variant in variants {
                incidence.add_variant(
                    &mut topology.diagram,
                    position,
                    &variant.graph,
                    variant.guard,
                )?;
            }
        }

        Self::from_incidence(topology, incidence, limits)
    }

    pub(super) fn from_incidence(
        topology: &mut GuardedTopology,
        incidence: AnchorIncidence,
        limits: ModuleCertificationLimits,
    ) -> Result<Self, CompileError> {
        let AnchorIncidence {
            positions,
            indices,
            edges,
            descents,
        } = incidence;

        let mut reverse = vec![Vec::new(); positions.len()];
        topology.diagram.charge(edges.len())?;
        for ((source, target), guard) in edges {
            reverse[target].push((source, guard));
        }
        let mut dynamic = vec![ZERO; positions.len()];
        let mut pending = VecDeque::new();
        // Source validation forbids dynamic blocks inside structural arms.
        topology.diagram.charge(topology.source.block_count())?;
        for block in topology
            .source
            .blocks()
            .filter(|block| block.kind().is_dynamic())
        {
            let index = indices[&block.pos()];
            if dynamic[index] != ONE {
                dynamic[index] = ONE;
                pending.push_back(index);
            }
        }
        // Saturate paths of always-present reverse edges before composing
        // guarded paths. TRUE then absorbs every conditional contribution.
        while let Some(source) = pending.pop_front() {
            topology
                .diagram
                .charge(reverse[source].len().saturating_add(1))?;
            for &(target, edge) in &reverse[source] {
                if edge == ONE && dynamic[target] != ONE {
                    dynamic[target] = ONE;
                    pending.push_back(target);
                }
            }
        }
        topology.diagram.charge(dynamic.len())?;
        let mut queued = vec![false; positions.len()];
        for (index, &root) in dynamic.iter().enumerate() {
            if root == ONE {
                queued[index] = true;
                pending.push_back(index);
            }
        }
        while let Some(source) = pending.pop_front() {
            queued[source] = false;
            topology
                .diagram
                .charge(reverse[source].len().saturating_add(1))?;
            for &(target, edge) in &reverse[source] {
                if dynamic[target] == ONE {
                    continue;
                }
                let contribution = topology
                    .diagram
                    .apply(BooleanOp::And, edge, dynamic[source])?;
                let next = topology
                    .diagram
                    .apply(BooleanOp::Or, dynamic[target], contribution)?;
                if next != dynamic[target] {
                    dynamic[target] = next;
                    if !std::mem::replace(&mut queued[target], true) {
                        pending.push_back(target);
                    }
                }
            }
        }

        let mut incoming = vec![Vec::new(); positions.len()];
        topology.diagram.charge(descents.len())?;
        for ((source, target, flipped), guard) in descents {
            incoming[source].push((target, flipped, guard));
        }
        topology.diagram.charge(topology.source.block_count())?;
        let t_sources = topology
            .source
            .blocks()
            .filter(|block| block.kind() == BlockKind::T)
            .map(Block::pos)
            .collect::<crate::FxSet<_>>();
        let mut order = (0..positions.len()).collect::<Vec<_>>();
        topology.diagram.charge(
            positions
                .len()
                .saturating_mul(positions.len().max(1).ilog2() as usize + 1),
        )?;
        order.sort_unstable_by_key(|&index| {
            let position = positions[index];
            (position.z, position.x, position.y)
        });
        let mut lineages = vec![Vec::<((IVec3, bool), DecisionId)>::new(); positions.len()];
        let mut coefficients = 0usize;
        for index in order {
            topology
                .diagram
                .charge(incoming[index].len().saturating_add(1))?;
            let position = positions[index];
            let mut sources = BTreeMap::<([i32; 3], bool), DecisionId>::new();
            let mut add_source = |key,
                                  contribution,
                                  diagram: &mut bloq_utils::boolean::BooleanDecisionDiagram|
             -> Result<(), CompileError> {
                if !sources.contains_key(&key) {
                    coefficients = add_resource(
                        "guarded T lineage coefficients",
                        coefficients,
                        1,
                        limits.max_witness_nodes,
                    )?;
                }
                let guard = sources.entry(key).or_insert(ZERO);
                *guard = diagram.apply(BooleanOp::Or, *guard, contribution)?;
                Ok(())
            };
            if t_sources.contains(&position) {
                add_source((position.to_array(), false), ONE, &mut topology.diagram)?;
            } else {
                for &(below, flipped, edge) in &incoming[index] {
                    topology.diagram.charge(lineages[below].len())?;
                    for &((source, previous_flip), guard) in &lineages[below] {
                        let contribution = topology.diagram.apply(BooleanOp::And, edge, guard)?;
                        if contribution == ZERO {
                            continue;
                        }
                        add_source(
                            (source.to_array(), flipped ^ previous_flip),
                            contribution,
                            &mut topology.diagram,
                        )?;
                    }
                }
            }
            topology.diagram.charge(sources.len())?;
            lineages[index] = sources
                .into_iter()
                .map(|((source, flipped), guard)| ((IVec3::from_array(source), flipped), guard))
                .collect();
        }
        topology.diagram.charge(positions.len())?;
        Ok(Self {
            sites: positions
                .into_iter()
                .zip(dynamic)
                .zip(lineages)
                .map(|((position, dynamic), sources)| {
                    (position, GuardedAnchor { dynamic, sources })
                })
                .collect(),
            case_limit: limits.max_guarded_domain_size,
        })
    }

    /// T lineage is a partial function, so intersect its mutually exclusive
    /// alternatives directly instead of enumerating a one-hot Cartesian tuple.
    pub(crate) fn cases(
        &self,
        topology: &mut GuardedTopology,
        position: IVec3,
        enabled: DecisionId,
    ) -> Result<Vec<(DecisionId, DynamicAnchor)>, CompileError> {
        let site = &self.sites[&position];
        topology
            .diagram
            .charge(site.sources.len().saturating_add(3))?;
        let mut remaining = topology
            .diagram
            .apply(BooleanOp::And, enabled, topology.domain)?;
        let mut output = Vec::new();
        let mut push = |guard, anchor| -> Result<(), CompileError> {
            check_resource(
                "local dynamic anchor patterns",
                output.len().saturating_add(1),
                self.case_limit,
            )?;
            output.push((guard, anchor));
            Ok(())
        };
        for &(source, guard) in &site.sources {
            let active = topology.diagram.apply(BooleanOp::And, remaining, guard)?;
            if active == ZERO {
                continue;
            }
            push(
                active,
                DynamicAnchor {
                    touches_dynamic: true,
                    source_t: Some(source),
                },
            )?;
            let absent = topology.diagram.negate(guard)?;
            remaining = topology.diagram.apply(BooleanOp::And, remaining, absent)?;
        }
        let dynamic = topology
            .diagram
            .apply(BooleanOp::And, remaining, site.dynamic)?;
        if dynamic != ZERO {
            push(
                dynamic,
                DynamicAnchor {
                    touches_dynamic: true,
                    source_t: None,
                },
            )?;
        }
        let static_guard = topology.diagram.negate(site.dynamic)?;
        let static_guard = topology
            .diagram
            .apply(BooleanOp::And, remaining, static_guard)?;
        if static_guard != ZERO {
            push(static_guard, DynamicAnchor::default())?;
        }
        topology.diagram.charge(output.len())?;
        for (guard, _) in &mut output {
            *guard = topology.diagram.constrain(*guard, topology.domain)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use bloq_graph::{CubeKind, GalleryItem, Pipe, PortRole};
    use glam::ivec3;

    use super::*;
    use crate::compile::{CompiledTemplateInfo, TemplateRef};

    fn assert_matches_full_queries(source: &BlockGraph) {
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut topology = GuardedTopology::new(source, limits).unwrap();
        let anchors = GuardedDynamicAnchors::new(&mut topology, limits).unwrap();
        let positions = topology.sites.keys().copied().collect::<Vec<_>>();
        for position in positions {
            let position = IVec3::from_array(position);
            let enabled = topology.sites[&position.to_array()]
                .iter()
                .try_fold(ZERO, |guard, variant| {
                    topology.diagram.apply(BooleanOp::Or, guard, variant.guard)
                })
                .unwrap();
            for (guard, anchor) in anchors.cases(&mut topology, position, enabled).unwrap() {
                let graph = topology.project(guard).unwrap();
                let (graph, ports) = crate::port::expand_port_topology(&graph, 3).unwrap();
                let compiled = graph
                    .blocks()
                    .map(|block| {
                        let mut connectivity = BlockSignature::graph_connectivity(block, &graph);
                        if let Some(dir) = ports
                            .get(&block.pos())
                            .and_then(|port| port.cube_pipe_dir())
                        {
                            connectivity = connectivity.with_pipe(dir);
                        }
                        (
                            block.pos(),
                            CompiledTemplateInfo {
                                graph_connectivity: connectivity,
                                // These queries read connectivity and source kinds only.
                                template: TemplateRef::Fixed(crate::block::LoweringTemplateId(0)),
                            },
                        )
                    })
                    .collect();
                assert_eq!(
                    anchor.touches_dynamic,
                    crate::lower::patch_touches_dynamic_block(&graph, &compiled, position),
                    "{position}"
                );
                assert_eq!(
                    anchor.source_t,
                    crate::lower::source_t_block(&graph, &compiled, position),
                    "{position}"
                );
            }
        }
    }

    fn switched_worldlines(second_t: bool) -> BlockGraph {
        let mut source = format!(
            "BLOG 1.0\nmodule main {{\nin q: data = 998\n998: Port [5,0,-2]\n999: ZXZ [5,0,-1]\n[5,0,-2] -> +Z\ns = measure 999\n0: T [0,0,0]\n1: {} [1,1,0]\n",
            if second_t { "T" } else { "XZX" }
        );
        for z in 3..=6 {
            source.push_str(&format!(
                "{}: ZXZ [0,1,{z}]\n{}: XZX [1,0,{z}]\n",
                2 * z,
                2 * z + 1
            ));
            if z < 6 {
                source.push_str(&format!("[0,1,{z}] -> +Z\n[1,0,{z}] -> +Z\n"));
            }
        }
        source.push_str(
            "branch b {
false {
100: walk ZXZ [0,0,1] -> [0,1,2]
101: walk XZX [1,1,1] -> [1,0,2]
[0,0,0] -> +Z
[1,1,0] -> +Z
[0,1,2] -> +Z
[1,0,2] -> +Z
}
true {
102: walk ZXZ [0,0,1] -> [1,0,2]
103: walk ZXZ [1,1,1] -> [0,1,2]
[0,0,0] -> +Z
[1,1,0] -H> +Z
[0,1,2] -> +Z
[1,0,2] -H> +Z
}
}
resolve b if s
}
",
        );
        let ast = bloq_graph::parse_blog_program_to_ast(&source).unwrap();
        bloq_graph::lower_blog_graph_ast_deferred(&ast)
            .unwrap()
            .materialize_flat_graph()
            .unwrap()
    }

    #[test]
    fn distant_t_sources_and_hadamard_parity_follow_selected_worldlines() {
        for second_t in [false, true] {
            let source = switched_worldlines(second_t);
            assert_matches_full_queries(&source);
            if second_t {
                crate::compile::compile(&source, 3)
                    .expect("distant conditional T lineage compiles through local gateways");
            }
            let limits = ModuleCertificationLimits::DEFAULT;
            let mut topology = GuardedTopology::new(&source, limits).unwrap();
            let anchors = GuardedDynamicAnchors::new(&mut topology, limits).unwrap();
            let position = ivec3(0, 1, 6);
            let cases = anchors.cases(&mut topology, position, ONE).unwrap();
            assert_eq!(cases.len(), 2);
            for (guard, anchor) in cases {
                let selected = topology.assignment(guard).unwrap().unwrap()[0].1;
                assert_eq!(
                    anchor,
                    if !selected {
                        DynamicAnchor {
                            touches_dynamic: true,
                            source_t: Some((IVec3::ZERO, false)),
                        }
                    } else if second_t {
                        DynamicAnchor {
                            touches_dynamic: true,
                            source_t: Some((ivec3(1, 1, 0), true)),
                        }
                    } else {
                        DynamicAnchor::default()
                    }
                );
            }
        }
    }

    #[test]
    fn local_anchor_metadata_matches_gallery_queries_and_ignores_virtual_neighbors() {
        for item in [
            GalleryItem::TWithPreparedY,
            GalleryItem::CCZFactoryWithTels,
            GalleryItem::CCZGateTeleport,
        ] {
            assert_matches_full_queries(&item.build().flatten().expect("gallery graph expands"));
        }
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Port)
                .with_port_role(PortRole::Input)
                .unwrap(),
        );
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        graph.add_block(Block::new(-3 * IVec3::Z, BlockKind::T));
        for z in [-2, -1] {
            graph.add_block(Block::new(z * IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            graph.add_pipe(Pipe::new((z - 1) * IVec3::Z, Direction::ZPLUS));
        }
        assert_matches_full_queries(&graph);
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut topology = GuardedTopology::new(&graph, limits).unwrap();
        let anchors = GuardedDynamicAnchors::new(&mut topology, limits).unwrap();
        assert_eq!(
            anchors.cases(&mut topology, IVec3::ZERO, ONE).unwrap(),
            [(ONE, DynamicAnchor::default())]
        );
    }

    #[test]
    fn certain_and_conditional_anchor_paths_keep_reverse_direction() {
        let source = switched_worldlines(false);
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut topology = GuardedTopology::new(&source, limits).unwrap();
        let branch = topology.branches[0].2;
        let seed = IVec3::ZERO;
        let certain = ivec3(10, 0, 0);
        let conditional = ivec3(11, 0, 0);
        let descendant = ivec3(12, 0, 0);
        let backward_only = ivec3(13, 0, 0);
        let positions = vec![seed, certain, conditional, descendant, backward_only];
        let indices = positions
            .iter()
            .enumerate()
            .map(|(index, &position)| (position, index))
            .collect();
        let edges = [
            ((1, 0), ONE),
            ((2, 1), branch),
            ((3, 2), ONE),
            ((0, 4), ONE),
        ]
        .into_iter()
        .collect();
        let incidence = AnchorIncidence {
            positions,
            indices,
            edges,
            descents: FxMap::default(),
        };
        let anchors =
            GuardedDynamicAnchors::from_incidence(&mut topology, incidence, limits).unwrap();
        assert_eq!(anchors.sites[&seed].dynamic, ONE);
        assert_eq!(anchors.sites[&certain].dynamic, ONE);
        assert_eq!(anchors.sites[&backward_only].dynamic, ZERO);
        for selected in [false, true] {
            let outcome = |_: &str| selected;
            for position in [conditional, descendant] {
                assert_eq!(
                    topology.evaluate_source(anchors.sites[&position].dynamic, outcome),
                    selected,
                    "{position} follows the guarded reverse path"
                );
            }
        }
    }

    #[test]
    fn global_anchor_work_storage_and_local_cases_keep_their_limits() {
        let source = switched_worldlines(true);
        let limits = ModuleCertificationLimits::DEFAULT;
        let mut topology = GuardedTopology::new(&source, limits).unwrap();
        let anchors = GuardedDynamicAnchors::new(
            &mut topology,
            ModuleCertificationLimits {
                max_guarded_domain_size: 1,
                ..limits
            },
        )
        .unwrap();
        assert!(
            matches!(anchors.cases(&mut topology, ivec3(0, 1, 6), ONE), Err(CompileError::BooleanResource(error))
            if error.resource == "local dynamic anchor patterns" && error.observed == 2 && error.limit == 1)
        );
        assert!(
            matches!(GuardedDynamicAnchors::new(&mut topology, ModuleCertificationLimits {
            max_witness_nodes: 1,
            ..limits
        }), Err(CompileError::BooleanResource(error))
            if error.resource == "guarded T lineage coefficients" && error.observed == 2 && error.limit == 1)
        );
        let remaining = topology.diagram.limits().max_steps - topology.diagram.steps();
        topology.diagram.charge(remaining - 1).unwrap();
        assert!(
            matches!(GuardedDynamicAnchors::new(&mut topology, limits), Err(CompileError::BooleanResource(error))
            if error.resource == "Boolean work steps")
        );
    }
}
