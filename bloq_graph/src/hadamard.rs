//! Local ZX color changes that move spatial Hadamards onto temporal pipes.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use glam::IVec3;
use petgraph::algo::maximum_flow::dinics;
use petgraph::graph::DiGraph;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;
use thiserror::Error;

use crate::{
    Action, Basis, BlockGraph, BlockGraphError, BlockKind, MeasureTarget, Pipe, UDirection,
};

/// A spatial Hadamard cannot be removed within the supported local identities.
#[derive(Debug, Clone, Error)]
pub enum HadamardPushError {
    /// Protected endpoints require walls on both axes of one spatial component.
    #[error(
        "Hadamard pushing cannot make component {component} use one spatial axis: X-only conflicts along {x_conflict:?}; Y-only conflicts along {y_conflict:?}"
    )]
    MixedAxesRequired {
        /// Lowest cube position in the component.
        component: IVec3,
        /// Oppositely anchored path for X-only walls.
        x_conflict: Vec<IVec3>,
        /// Oppositely anchored path for Y-only walls.
        y_conflict: Vec<IVec3>,
    },
}

impl From<HadamardPushError> for BlockGraphError {
    fn from(error: HadamardPushError) -> Self {
        Self::HadamardPush(Box::new(error))
    }
}

#[derive(Clone, Copy)]
struct SpatialEdge {
    left: usize,
    right: usize,
    axis: UDirection,
}

impl BlockGraph {
    /// Returns an equivalent graph with as few spatial cube-to-cube Hadamards as possible
    /// under protected resource/control frames and one wall axis per component.
    ///
    /// Ordinary cube color changes toggle every incident pipe's Hadamard.
    /// Positions, topology, tags, interfaces, and named outcomes are retained.
    /// Zero spatial walls are preferred; otherwise each spatial component uses
    /// an exact minimum cut restricted to X walls or to Y walls, preferring X
    /// on a tie. Unanchored zero-wall frames use temporal-wall and cube-flip
    /// counts to break ties; their global minimum is not claimed.
    /// Terminal measurement bases absorb temporal Hadamards where their signs
    /// can be preserved, conjugating their feedback and selector conditions.
    /// Resource-edge frames retain their existing physical handling.
    /// Other graphs without spatial Hadamards, and sources with structural
    /// branch regions, are returned as [`Cow::Borrowed`]. Extended and
    /// moving blocks keep their local frames.
    /// Spatial components without cube-to-cube Hadamards keep their cube colors.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid frames, unflattened instances,
    /// incompatible protected anchors, or invalid
    /// transformed readouts. Physical compilation remains a separate check.
    ///
    /// # Panics
    ///
    /// Panics if internal graph or cube-component indexes are inconsistent.
    pub fn with_temporal_hadamards(&self) -> Result<Cow<'_, Self>, BlockGraphError> {
        if !self.branch_definitions().is_empty() {
            return Ok(Cow::Borrowed(self));
        }
        let spatial = self
            .pipes()
            .any(|pipe| pipe.dir().is_spatial() && pipe.is_hadamard());
        let absorbable = crate::hadamard_measurement::absorbable_measurements(self);
        let terminal = absorbable.iter().any(|&position| {
            self.pipes_at(IVec3::from_array(position))
                .any(Pipe::is_hadamard)
        });
        if !spatial && !terminal {
            return Ok(Cow::Borrowed(self));
        }
        self.require_flat_hierarchy("Hadamard pushing")?;
        self.validate_structure()?;
        let fixed = protected_cubes(self, &absorbable)?;
        let mut pipes = self.pipes().cloned().collect::<Vec<_>>();
        pipes.sort_unstable_by_key(|pipe| {
            let (left, right) = (pipe.src().to_array(), pipe.dst().to_array());
            (left.min(right), left.max(right))
        });
        let mut incident = HashMap::<_, Vec<_>>::new();
        for (index, pipe) in pipes.iter().enumerate() {
            for endpoint in [pipe.src(), pipe.dst()] {
                incident.entry(endpoint).or_default().push(index);
            }
        }
        let mut flips = HashSet::new();
        for component in self.cube_layer_components().into_values().flatten() {
            crate::CancellationToken::checkpoint()?;
            let index = component
                .iter()
                .enumerate()
                .map(|(index, &position)| (position, index))
                .collect::<HashMap<_, _>>();
            let mut local_pipes = component
                .iter()
                .flat_map(|position| incident.get(position).into_iter().flatten().copied())
                .collect::<Vec<_>>();
            local_pipes.sort_unstable();
            local_pipes.dedup();
            let spatial = local_pipes
                .iter()
                .map(|&index| &pipes[index])
                .filter(|pipe| {
                    pipe.dir().is_spatial()
                        && index.contains_key(&pipe.src())
                        && index.contains_key(&pipe.dst())
                })
                .collect::<Vec<_>>();
            if !spatial.iter().any(|pipe| pipe.is_hadamard()) {
                continue;
            }
            let old = component
                .iter()
                .map(
                    |&position| match self.get_block(position).expect("component cube").kind() {
                        BlockKind::Cube(kind) => kind.z() == Basis::Z,
                        _ => unreachable!("native component contains cubes"),
                    },
                )
                .collect::<Vec<_>>();
            let anchors = component
                .iter()
                .zip(&old)
                .map(|(position, &color)| fixed.contains(position).then_some(color))
                .collect::<Vec<_>>();
            let score = |colors: &[bool]| {
                let changed = component
                    .iter()
                    .zip(colors)
                    .zip(&old)
                    .filter_map(|((&position, &color), &old)| (color != old).then_some(position))
                    .collect::<HashSet<_>>();
                let temporal = local_pipes
                    .iter()
                    .map(|&index| &pipes[index])
                    .filter(|pipe| !pipe.dir().is_spatial())
                    .filter(|pipe| {
                        pipe.is_hadamard()
                            ^ changed.contains(&pipe.src())
                            ^ changed.contains(&pipe.dst())
                            ^ flips.contains(&pipe.src())
                            ^ flips.contains(&pipe.dst())
                    })
                    .count();
                (temporal, changed.len())
            };
            let mut constant = [false, true]
                .into_iter()
                .filter(|color| anchors.iter().flatten().all(|anchor| anchor == color))
                .map(|color| vec![color; component.len()])
                .collect::<Vec<_>>();
            constant.sort_by_key(|colors| score(colors));
            let colors = if let Some(colors) = constant.into_iter().next() {
                colors
            } else {
                let edges = spatial
                    .iter()
                    .filter_map(|pipe| {
                        Some(SpatialEdge {
                            left: *index.get(&pipe.src())?,
                            right: *index.get(&pipe.dst())?,
                            axis: pipe.dir().as_udirection(),
                        })
                    })
                    .collect::<Vec<_>>();
                let x = axis_cut(&component, &edges, &anchors, UDirection::X);
                let y = axis_cut(&component, &edges, &anchors, UDirection::Y);
                match (x, y) {
                    (Ok(x), Ok(y)) => {
                        let cost = |colors: &[bool]| {
                            edges
                                .iter()
                                .filter(|edge| colors[edge.left] != colors[edge.right])
                                .count()
                        };
                        if cost(&x) <= cost(&y) { x } else { y }
                    }
                    (Ok(colors), Err(_)) | (Err(_), Ok(colors)) => colors,
                    (Err(x_conflict), Err(y_conflict)) => {
                        return Err(HadamardPushError::MixedAxesRequired {
                            component: component[0],
                            x_conflict,
                            y_conflict,
                        }
                        .into());
                    }
                }
            };
            for ((position, color), old) in component.into_iter().zip(colors).zip(old) {
                if color != old {
                    flips.insert(position);
                }
            }
        }
        let mut candidate = self.clone();
        apply_color_changes(&mut candidate, &flips, &pipes);
        if !crate::hadamard_measurement::absorb_terminal_hadamards(&mut candidate, &absorbable)?
            && flips.is_empty()
        {
            return Ok(Cow::Borrowed(self));
        }
        let actions = candidate.actions();
        candidate.set_actions(actions)?;
        candidate.validate()?;
        Ok(Cow::Owned(candidate))
    }
}

fn apply_color_changes(graph: &mut BlockGraph, flips: &HashSet<IVec3>, pipes: &[Pipe]) {
    for &position in flips {
        let block = graph.get_block_mut(position).expect("source cube exists");
        block.kind = block.kind().flip_xz_basis();
    }
    for pipe in pipes {
        let (left, right) = pipe.endpoints();
        crate::graph::flip_pipe_xz_basis(
            graph.get_pipe_mut(left, right).expect("source pipe exists"),
            |position| flips.contains(&position),
        );
    }
}

fn protected_cubes(
    graph: &BlockGraph,
    absorbable: &BTreeSet<[i32; 3]>,
) -> Result<HashSet<IVec3>, BlockGraphError> {
    let mut fixed = HashSet::new();
    for block in graph
        .blocks()
        .filter(|block| !block.kind().is_cube() || block.height_cells() != 1)
    {
        fixed.insert(block.pos());
        if !absorbable.contains(&block.pos().to_array()) {
            fixed.extend(
                graph
                    .neighbors(block.pos())
                    .into_iter()
                    .map(crate::Block::pos),
            );
        }
    }
    let mut protect = |position, direction| -> Result<(), BlockGraphError> {
        fixed.insert(position);
        if let Some(direction) = direction {
            fixed.insert(crate::checked_add_position(position, direction)?);
        }
        Ok(())
    };
    for action in graph.action_iter() {
        match action {
            Action::Measure { target, .. } => match *target {
                MeasureTarget::Node(position) => protect(position, None)?,
                MeasureTarget::Edge { src, dir } => protect(src, Some(dir.to_ivec3()))?,
            },
            Action::Feedback { targets, .. } => {
                for target in targets {
                    protect(
                        target.target,
                        target.direction.map(crate::Direction::to_ivec3),
                    )?;
                }
            }
            Action::Resolve { target, .. } | Action::Branch { target, .. } => {
                protect(*target, None)?
            }
            Action::Let { .. } | Action::DiscardIf(_) => {}
        }
    }
    Ok(fixed)
}

fn axis_cut(
    positions: &[IVec3],
    edges: &[SpatialEdge],
    anchors: &[Option<bool>],
    allowed: UDirection,
) -> Result<Vec<bool>, Vec<IVec3>> {
    let mut union = UnionFind::new(positions.len());
    for edge in edges.iter().filter(|edge| edge.axis != allowed) {
        union.union(edge.left, edge.right);
    }
    let mut groups = BTreeMap::new();
    let owner = (0..positions.len())
        .map(|index| {
            let next = groups.len();
            *groups.entry(union.find(index)).or_insert(next)
        })
        .collect::<Vec<_>>();
    let mut fixed = vec![None; groups.len()];
    for (index, anchor) in anchors.iter().enumerate() {
        if let Some(color) = anchor {
            if let Some((prior_color, prior)) = fixed[owner[index]] {
                if prior_color != *color {
                    return Err(conflict_path(positions, edges, allowed, prior, index));
                }
            } else {
                fixed[owner[index]] = Some((*color, index));
            }
        }
    }
    let mut network = DiGraph::<(), usize>::new();
    let nodes = (0..groups.len())
        .map(|_| network.add_node(()))
        .collect::<Vec<_>>();
    let source = network.add_node(());
    let sink = network.add_node(());
    // Cutting even every finite edge is cheaper than moving a protected anchor.
    let infinite = edges.len() + 1;
    for (group, anchor) in fixed.into_iter().enumerate() {
        match anchor {
            Some((false, _)) => {
                network.add_edge(source, nodes[group], infinite);
            }
            Some((true, _)) => {
                network.add_edge(nodes[group], sink, infinite);
            }
            None => {}
        }
    }
    for edge in edges {
        let (left, right) = (owner[edge.left], owner[edge.right]);
        if left != right {
            network.add_edge(nodes[left], nodes[right], 1);
            network.add_edge(nodes[right], nodes[left], 1);
        }
    }
    let (_, flow) = dinics(&network, source, sink);
    let mut reachable = HashSet::from([source]);
    let mut pending = vec![source];
    while let Some(node) = pending.pop() {
        for edge in network.edges_directed(node, petgraph::Direction::Outgoing) {
            if flow[edge.id().index()] < *edge.weight() && reachable.insert(edge.target()) {
                pending.push(edge.target());
            }
        }
        for edge in network.edges_directed(node, petgraph::Direction::Incoming) {
            if flow[edge.id().index()] > 0 && reachable.insert(edge.source()) {
                pending.push(edge.source());
            }
        }
    }
    Ok(owner
        .into_iter()
        .map(|group| !reachable.contains(&nodes[group]))
        .collect())
}

fn conflict_path(
    positions: &[IVec3],
    edges: &[SpatialEdge],
    allowed: UDirection,
    start: usize,
    end: usize,
) -> Vec<IVec3> {
    let mut adjacent = vec![Vec::new(); positions.len()];
    for edge in edges.iter().filter(|edge| edge.axis != allowed) {
        adjacent[edge.left].push(edge.right);
        adjacent[edge.right].push(edge.left);
    }
    let mut parent = vec![None; positions.len()];
    let mut pending = VecDeque::from([start]);
    parent[start] = Some(start);
    while let Some(node) = pending.pop_front() {
        if node == end {
            break;
        }
        for &next in &adjacent[node] {
            if parent[next].is_none() {
                parent[next] = Some(node);
                pending.push_back(next);
            }
        }
    }
    let mut path = vec![positions[end]];
    let mut node = end;
    while node != start {
        node = parent[node].expect("contracted conflicting anchors are connected");
        path.push(positions[node]);
    }
    path.reverse();
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, Direction, GalleryItem, PauliBasis};
    #[cfg(feature = "verify")]
    use crate::{CubeKind, PortRole};
    use glam::ivec3;

    fn corner_path() -> BlockGraph {
        BlockGraph::from_blog_text(
            "BLOG 1.0\nmodule main {\nin input: data = 0\nout output: data = 5\n\
             0: Port [0, 0, -1] role=input\n\
             1: XZX [0, 0, 0] <input_cube>\n\
             2: XXZ [1, 0, 0]\n\
             3: ZZX [1, 1, 0]\n\
             4: ZXZ [2, 1, 0] <output_cube>\n\
             5: Port [2, 1, 1] role=output\n\
             0 -> +Z\n1 -H> +X <first>\n2 -H> +Y\n3 -H> +X\n4 -> +Z\n}\n",
        )
        .unwrap()
    }

    fn wall_axes(graph: &BlockGraph) -> (usize, usize) {
        let count = |axis| {
            graph
                .pipes()
                .filter(|pipe| pipe.is_hadamard() && pipe.dir().as_udirection() == axis)
                .count()
        };
        (count(UDirection::X), count(UDirection::Y))
    }

    #[test]
    fn eliminates_spatial_walls_and_preserves_metadata_deterministically() {
        let source = BlockGraph::from_blog_text(
            "BLOG 1.0\nmodule main {\nin input: data = 0\nout output: data = 5\n\
             0: Port [0, 0, -1] role=input\n\
             1: ZXZ [0, 0, 0]\n2: ZXZ [0, 0, 1] <turn>\n\
             3: XZX [1, 0, 1]\n4: XZX [1, 0, 2]\n\
             5: Port [1, 0, 3] role=output\n\
             0 -> +Z\n1 -> +Z\n2 -H> +X <wall>\n3 -> +Z\n4 -> +Z\n}\n",
        )
        .unwrap();
        let before = source.to_blog_text();
        let rewritten = source.with_temporal_hadamards().unwrap();
        assert_eq!(wall_axes(&rewritten), (0, 0));
        assert_eq!(rewritten.block_count(), source.block_count());
        assert_eq!(rewritten.pipe_count(), source.pipe_count());
        for block in source.blocks() {
            let updated = rewritten.get_block(block.pos()).unwrap();
            assert_eq!(updated.tag(), block.tag());
            assert_eq!(updated.height(), block.height());
            assert_eq!(updated.port_role(), block.port_role());
        }
        for pipe in source.pipes() {
            assert_eq!(
                rewritten.get_pipe(pipe.src(), pipe.dst()).unwrap().tag(),
                pipe.tag()
            );
        }
        assert_eq!(source.to_blog_text(), before);
        assert_eq!(
            rewritten.to_blog_text(),
            source.with_temporal_hadamards().unwrap().to_blog_text()
        );
        assert_eq!(
            rewritten.to_blog_text(),
            rewritten.with_temporal_hadamards().unwrap().to_blog_text()
        );
        assert_eq!(
            rewritten.to_blog_text(),
            BlockGraph::from_blog_text(&rewritten.to_blog_text())
                .unwrap()
                .to_blog_text()
        );
        #[cfg(feature = "verify")]
        assert_hadamard_channel(&source, &rewritten);
    }

    #[test]
    fn exact_single_axis_fallback_and_protected_action_conflicts() {
        let source = corner_path();
        assert_eq!(wall_axes(&source), (2, 1));
        let rewritten = source.with_temporal_hadamards().unwrap();
        assert_eq!(wall_axes(&rewritten), (1, 0));
        #[cfg(feature = "verify")]
        assert_hadamard_channel(&source, &rewritten);

        let mut protected = source;
        protected
            .set_actions(vec![Action::Feedback {
                targets: vec![crate::FeedbackTarget {
                    target: ivec3(1, 0, 0),
                    direction: Some(Direction::YPLUS),
                    pauli: PauliBasis::X,
                }],
                condition: None,
            }])
            .unwrap();
        let before = protected.to_blog_text();
        let error = protected.with_temporal_hadamards().unwrap_err();
        assert!(matches!(error, BlockGraphError::HadamardPush(_)));
        assert_eq!(protected.to_blog_text(), before);
    }

    #[test]
    fn remaining_walls_do_not_recolor_unrelated_wall_free_components_on_repeated_calls() {
        let mut source = corner_path();
        for (x, kind) in [
            (-10, crate::CubeKind::XZX),
            (-9, crate::CubeKind::XZX),
            (-7, crate::CubeKind::ZXZ),
            (-6, crate::CubeKind::ZXZ),
        ] {
            source.add_block(Block::new(ivec3(x, 0, 0), BlockKind::Cube(kind)));
        }
        for x in -10..=-6 {
            let kind = if x < -8 {
                crate::CubeKind::XZX
            } else {
                crate::CubeKind::ZXZ
            };
            source.add_block(Block::new(ivec3(x, 0, 1), BlockKind::Cube(kind)));
        }
        for x in [-10, -7] {
            source
                .try_add_pipe(Pipe::new(ivec3(x, 0, 0), Direction::XPLUS))
                .unwrap();
        }
        for x in -10..-6 {
            let pipe = Pipe::new(ivec3(x, 0, 1), Direction::XPLUS);
            source
                .try_add_pipe(if x == -9 { pipe.with_hadamard() } else { pipe })
                .unwrap();
        }
        for x in [-10, -9, -7, -6] {
            source
                .try_add_pipe(Pipe::new(ivec3(x, 0, 0), Direction::ZPLUS))
                .unwrap();
        }
        source.validate().unwrap();
        let normalized = source.with_temporal_hadamards().unwrap();
        assert_eq!(wall_axes(&normalized), (1, 0));
        for x in [-10, -9, -7, -6] {
            assert_eq!(
                normalized.get_block(ivec3(x, 0, 0)),
                source.get_block(ivec3(x, 0, 0))
            );
        }
        assert_eq!(
            normalized.to_blog_text(),
            normalized.with_temporal_hadamards().unwrap().to_blog_text()
        );
    }

    #[test]
    fn equal_spatial_cut_counts_use_a_stable_axis_despite_temporal_neighbors() {
        let text = "BLOG 1.0\nmodule main {\n\
            in input: data = 0\nout output: data = 7\nout auxiliary: data = 9\n\
            0: Port [0, 0, -1] role=input\n1: XZX [0, 0, 0]\n\
            2: XZX [1, 0, 0]\n3: ZZX [2, 0, 0]\n4: ZZX [2, 1, 0]\n\
            5: XZX [3, 1, 0]\n6: ZXZ [4, 1, 0]\n\
            7: Port [4, 1, 1] role=output\n\
            8: XZX [1, 0, 1]\n9: Port [1, 0, 2] role=output\n\
            0 -> +Z\n1 -> +X\n2 -> +X\n3 -> +Y\n4 -> +X\n5 -H> +X\n\
            6 -> +Z\n2 -> +Z\n8 -> +Z\n}\n";
        for text in [
            text.to_owned(),
            text.replace("8: XZX", "8: ZXZ")
                .replace("2 -> +Z", "2 -H> +Z"),
        ] {
            let source = BlockGraph::from_blog_text(&text).unwrap();
            let normalized = source.with_temporal_hadamards().unwrap();
            assert_eq!(wall_axes(&normalized), (1, 0));
            assert_eq!(
                normalized.to_blog_text(),
                normalized.with_temporal_hadamards().unwrap().to_blog_text()
            );
        }
    }

    #[test]
    fn incompatible_axis_policies_report_their_anchored_paths() {
        let mut source = corner_path();
        source
            .set_actions_deferred(vec![Action::Measure {
                target: MeasureTarget::Edge {
                    src: ivec3(1, 0, 0),
                    dir: Direction::YPLUS,
                },
                name: "edge_m".into(),
            }])
            .unwrap();
        source.validate_source().unwrap();
        let error = source.with_temporal_hadamards().unwrap_err();
        let BlockGraphError::HadamardPush(error) = error else {
            panic!("{error}")
        };
        let HadamardPushError::MixedAxesRequired {
            component,
            x_conflict,
            y_conflict,
        } = *error;
        assert_eq!(component, ivec3(0, 0, 0));
        assert_eq!(x_conflict, [ivec3(1, 0, 0), ivec3(1, 1, 0)]);
        assert!(!y_conflict.is_empty());
    }

    #[test]
    fn no_op_graphs_and_invalid_frames_are_not_reinterpreted() {
        for source in [
            GalleryItem::T.build(),
            BlockGraph::from_blog_text(include_str!(
                "../../docs/fixtures/conditional_cz_strip.blog"
            ))
            .unwrap(),
        ] {
            assert_eq!(
                source.to_blog_text(),
                source.with_temporal_hadamards().unwrap().to_blog_text()
            );
        }
        let mut invalid = corner_path();
        invalid
            .set_pipe_hadamard(ivec3(0, 0, 0), ivec3(1, 0, 0), false)
            .unwrap();
        assert!(matches!(
            invalid.with_temporal_hadamards(),
            Err(BlockGraphError::Invalid(_))
        ));
    }

    #[test]
    fn reanalysis_preserves_external_inputs_and_unrelated_extended_geometry() {
        let mut source = corner_path().copy_local_geometry();
        let position = ivec3(10, 0, 0);
        let extended = Block::new(position, BlockKind::Cube(crate::CubeKind::ZXZ))
            .with_height("2d".parse().unwrap())
            .unwrap();
        source.add_block(extended.clone());
        let actions = vec![Action::Let {
            name: "alias".into(),
            expr: crate::Expr::Var("external".into()),
        }];
        source
            .set_actions_with_inputs(actions.clone(), ["external".into()])
            .unwrap();
        let mut rewritten = source.with_temporal_hadamards().unwrap().into_owned();
        assert_eq!(wall_axes(&rewritten), (1, 0));
        assert_eq!(rewritten.get_block(position), Some(&extended));
        assert_eq!(rewritten.actions(), actions);
        // Installing the same expression again requires its declared input.
        rewritten.set_actions(actions).unwrap();
        rewritten.validate().unwrap();
    }

    #[cfg(feature = "verify")]
    fn assert_hadamard_channel(source: &BlockGraph, rewritten: &BlockGraph) {
        let expected = quizx::circuit::Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];",
        )
        .unwrap()
        .to_graph();
        for graph in [source, rewritten] {
            crate::verify::LogicalVerifier::new(graph)
                .unwrap()
                .verify(&expected, 1, 17)
                .unwrap();
        }
    }

    #[cfg(feature = "verify")]
    #[test]
    fn local_spider_color_change_toggles_every_incident_leg() {
        let mut source = BlockGraph::new();
        source.add_block(Block::new(IVec3::ZERO, BlockKind::Cube(CubeKind::ZXZ)));
        for (position, role, direction) in [
            (ivec3(0, 0, -1), PortRole::Input, Direction::ZMINUS),
            (ivec3(1, 0, 0), PortRole::Output, Direction::XPLUS),
            (ivec3(-1, 0, 0), PortRole::Output, Direction::XMINUS),
        ] {
            source.add_block(
                Block::new(position, BlockKind::Port)
                    .with_port_role(role)
                    .unwrap(),
            );
            source
                .try_add_pipe(Pipe::new(IVec3::ZERO, direction))
                .unwrap();
        }
        source.validate().unwrap();
        let (_, expected) = crate::verify::LogicalVerifier::new(&source)
            .unwrap()
            .sample_with_seed(0)
            .unwrap();
        let mut rewritten = source.clone();
        apply_color_changes(
            &mut rewritten,
            &HashSet::from([IVec3::ZERO]),
            &source.pipes().cloned().collect::<Vec<_>>(),
        );
        rewritten.validate().unwrap();
        assert!(rewritten.pipes().all(Pipe::is_hadamard));
        crate::verify::LogicalVerifier::new(&rewritten)
            .unwrap()
            .verify(&expected.unwrap(), 1, 0)
            .unwrap();
    }
}
