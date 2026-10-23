//! Move Hadamards into time using exact, phase-zero ZX color changes.

use std::collections::HashMap;

use bloq_graph::{Action, Block, BlockGraph, BlockKind, MeasureTarget, Pipe};
use glam::IVec3;

struct Edge {
    a: usize,
    b: usize,
    pipe: Pipe,
}

impl Edge {
    fn other(&self, node: usize) -> usize {
        if self.a == node { self.b } else { self.a }
    }

    fn hadamard(&self, flips: &[bool]) -> bool {
        self.pipe.is_hadamard() ^ flips[self.a] ^ flips[self.b]
    }
}

pub(crate) fn optimize(graph: BlockGraph) -> Result<BlockGraph, String> {
    if graph.has_module_structure() || !graph.branch_definitions().is_empty() {
        return Err("Hadamard optimization requires a flat native layout".into());
    }
    let mut blocks = graph.blocks().collect::<Vec<_>>();
    blocks.sort_unstable_by_key(|block| block.pos().to_array());
    let indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.pos(), index))
        .collect::<HashMap<_, _>>();
    let edges = graph
        .pipe_endpoints_with_blocks()
        .map(|(_, _, a, b, pipe)| Edge {
            a: indices[&a.pos()],
            b: indices[&b.pos()],
            pipe: pipe.clone(),
        })
        .collect::<Vec<_>>();
    let mut incident = vec![Vec::new(); blocks.len()];
    for (index, edge) in edges.iter().enumerate() {
        incident[edge.a].push(index);
        incident[edge.b].push(index);
    }
    let mut fixed = blocks
        .iter()
        .map(|block| !block.kind().is_cube())
        .collect::<Vec<_>>();
    let mut anchor = |position: IVec3| -> Result<(), String> {
        let block = graph
            .get_endpoint_block(position)
            .ok_or_else(|| format!("Hadamard action anchor has no block at {position}"))?;
        fixed[indices[&block.pos()]] = true;
        Ok(())
    };
    let actions = graph.actions();
    for action in &actions {
        match action {
            Action::Measure { target, .. } => match *target {
                MeasureTarget::Node(position) => anchor(position)?,
                MeasureTarget::Edge { src, dir } => {
                    anchor(src)?;
                    anchor(src + dir.to_ivec3())?;
                }
            },
            Action::Resolve { target, .. } | Action::Branch { target, .. } => anchor(*target)?,
            Action::Feedback { targets, .. } => {
                for target in targets {
                    anchor(target.target)?;
                }
            }
            Action::Let { .. } | Action::DiscardIf(_) => {}
        }
    }

    let mut frames = vec![None; blocks.len()];
    let mut components = Vec::new();
    let mut component_of = vec![0; blocks.len()];
    for start in 0..blocks.len() {
        if frames[start].is_some() {
            continue;
        }
        frames[start] = Some(false);
        let mut nodes = vec![start];
        let mut cursor = 0;
        let mut required = None;
        while cursor < nodes.len() {
            let node = nodes[cursor];
            cursor += 1;
            let frame = frames[node].expect("queued frame exists");
            component_of[node] = components.len();
            if fixed[node] {
                if required.is_some_and(|value| value != frame) {
                    return Err(format!(
                        "spatial Hadamards cannot be moved past fixed frames at {}",
                        blocks[node].pos()
                    ));
                }
                required = Some(frame);
            }
            for &index in &incident[node] {
                let edge = &edges[index];
                if !edge.pipe.dir().is_spatial() {
                    continue;
                }
                let next = edge.other(node);
                let value = frame ^ edge.pipe.is_hadamard();
                match frames[next] {
                    Some(previous) if previous != value => {
                        return Err(format!(
                            "spatial Hadamards have inconsistent parity at {}",
                            blocks[next].pos()
                        ));
                    }
                    Some(_) => {}
                    None => {
                        frames[next] = Some(value);
                        nodes.push(next);
                    }
                }
            }
        }
        let complement = required.unwrap_or_else(|| {
            2 * nodes
                .iter()
                .filter(|&&node| frames[node] == Some(true))
                .count()
                > nodes.len()
        });
        for &node in &nodes {
            frames[node] = Some(frames[node].expect("component frame exists") ^ complement);
        }
        components.push((nodes, required.is_none()));
    }
    let mut flips = frames.into_iter().map(Option::unwrap).collect::<Vec<_>>();
    // One deterministic sweep improves temporal H count without a global search.
    for (component, (nodes, free)) in components.iter().enumerate() {
        if !free {
            continue;
        }
        let mut remove = 0;
        let mut add = 0;
        for &node in nodes {
            for &index in &incident[node] {
                let edge = &edges[index];
                if component_of[edge.other(node)] != component {
                    if edge.hadamard(&flips) {
                        remove += 1;
                    } else {
                        add += 1;
                    }
                }
            }
        }
        let flipped = nodes.iter().filter(|&&node| flips[node]).count();
        if remove > add || (remove == add && 2 * flipped > nodes.len()) {
            for &node in nodes {
                flips[node] ^= true;
            }
        }
    }

    let identity = (0..blocks.len())
        .map(|node| {
            !fixed[node]
                && incident[node].len() == 2
                && incident[node]
                    .iter()
                    .all(|&index| !edges[index].pipe.dir().is_spatial())
                && incident[node]
                    .iter()
                    .any(|&index| blocks[edges[index].other(node)].pos().z < blocks[node].pos().z)
                && incident[node]
                    .iter()
                    .any(|&index| blocks[edges[index].other(node)].pos().z > blocks[node].pos().z)
        })
        .collect::<Vec<_>>();
    for start in 0..blocks.len() {
        if !identity[start] {
            continue;
        }
        let incoming = *incident[start]
            .iter()
            .find(|&&index| blocks[edges[index].other(start)].pos().z < blocks[start].pos().z)
            .expect("identity has an incoming pipe");
        let mut previous = edges[incoming].other(start);
        if identity[previous] {
            continue;
        }
        let mut node = start;
        let mut edge = incoming;
        while identity[node] {
            // Collect the chain's H parity at its last edge; storage time stays intact.
            flips[node] = edges[edge].pipe.is_hadamard() ^ flips[previous];
            let outgoing = *incident[node]
                .iter()
                .find(|&&index| index != edge)
                .expect("identity has an outgoing pipe");
            previous = node;
            node = edges[outgoing].other(node);
            edge = outgoing;
        }
    }

    let mut optimized = BlockGraph::new();
    for (index, block) in blocks.iter().enumerate() {
        let block = if flips[index] {
            let BlockKind::Cube(_) = block.kind() else {
                unreachable!("fixed non-cube frame cannot flip")
            };
            let mut changed = Block::new(block.pos(), block.kind().flip_xz_basis())
                .with_height(block.height())
                .map_err(|error| error.to_string())?;
            if let Some(tag) = block.tag() {
                changed = changed.with_tag(tag).map_err(|error| error.to_string())?;
            }
            changed
        } else {
            (*block).clone()
        };
        optimized
            .try_add_block(block)
            .map_err(|error| error.to_string())?;
    }
    for edge in &edges {
        let hadamard = edge.hadamard(&flips);
        if hadamard && edge.pipe.dir().is_spatial() {
            return Err("Hadamard optimization left an unsupported spatial wall".into());
        }
        let mut pipe = Pipe::new(edge.pipe.src(), edge.pipe.dir());
        if hadamard {
            pipe = pipe.with_hadamard();
        }
        if let Some(tag) = edge.pipe.tag() {
            pipe = pipe.with_tag(tag).map_err(|error| error.to_string())?;
        }
        optimized
            .try_add_pipe(pipe)
            .map_err(|error| error.to_string())?;
    }
    optimized
        .set_actions_deferred(actions)
        .map_err(|error| error.to_string())?;
    Ok(optimized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::verify::{BoundaryOrder, LogicalVerifier, QuizxGraph};
    use bloq_graph::{
        CubeHeight, CubeKind, Direction, FeedbackTarget, GalleryItem, PauliBasis, PortRole,
    };
    use quizx::circuit::Circuit;

    #[test]
    fn spatial_cz_becomes_temporal_without_changing_its_map() {
        let source = GalleryItem::CZSpatialH
            .build()
            .materialize_flat_graph()
            .unwrap();
        let graph = optimize(source).unwrap();
        graph.validate_source().unwrap();
        assert!(
            graph
                .pipes()
                .all(|pipe| !pipe.dir().is_spatial() || !pipe.is_hadamard())
        );
        assert_eq!(graph.pipes().filter(|pipe| pipe.is_hadamard()).count(), 2);
        let mut circuit = Circuit::new(2);
        circuit.add_gate("cz", vec![0, 1]);
        let expected: QuizxGraph = circuit.to_graph();
        LogicalVerifier::with_boundaries(
            &graph,
            BoundaryOrder::new(
                vec![IVec3::ZERO, IVec3::new(1, -1, 1)],
                vec![IVec3::new(0, 0, 2), IVec3::new(1, 1, 1)],
            ),
        )
        .unwrap()
        .verify(&expected, 8, 0x4848)
        .unwrap();
        assert!(bloq_compile::spatial_hadamard_distance_warning(&graph).is_none());
    }

    #[test]
    fn temporal_parity_cancels_across_tall_memory_without_shortening_it() {
        for odd in [false, true] {
            let mut source = BlockGraph::new();
            let input = IVec3::ZERO;
            let middle = IVec3::Z;
            let output = IVec3::Z * 5;
            source.add_block(
                Block::new(input, BlockKind::Port)
                    .with_port_role(PortRole::Input)
                    .unwrap(),
            );
            source.add_block(
                Block::new(middle, BlockKind::Cube(CubeKind::XZX))
                    .with_height(CubeHeight::new(4, 1, 0).unwrap())
                    .unwrap()
                    .with_tag("memory")
                    .unwrap(),
            );
            source.add_block(
                Block::new(output, BlockKind::Port)
                    .with_port_role(PortRole::Output)
                    .unwrap(),
            );
            source.add_pipe(
                Pipe::new(input, Direction::ZPLUS)
                    .with_hadamard()
                    .with_tag("before")
                    .unwrap(),
            );
            let final_pipe = Pipe::new(IVec3::Z * 4, Direction::ZPLUS);
            source.add_pipe(if odd {
                final_pipe
            } else {
                final_pipe.with_hadamard()
            });
            source.validate_source().unwrap();
            let graph = optimize(source).unwrap();
            graph.validate_source().unwrap();
            assert_eq!(graph.block_count(), 3);
            assert_eq!(graph.get_block(middle).unwrap().height_cells(), 4);
            assert_eq!(graph.get_block(middle).unwrap().tag(), Some("memory"));
            assert_eq!(graph.get_pipe(input, middle).unwrap().tag(), Some("before"));
            assert_eq!(
                graph.pipes().filter(|pipe| pipe.is_hadamard()).count(),
                usize::from(odd)
            );
            let mut circuit = Circuit::new(1);
            if odd {
                circuit.add_gate("h", vec![0]);
            }
            let expected: QuizxGraph = circuit.to_graph();
            LogicalVerifier::with_boundaries(&graph, BoundaryOrder::new(vec![input], vec![output]))
                .unwrap()
                .verify(&expected, 8, 0x4849)
                .unwrap();
        }
    }

    #[test]
    fn incompatible_action_anchors_reject_spatial_relocation() {
        let mut source = GalleryItem::CZSpatialH
            .build()
            .materialize_flat_graph()
            .unwrap();
        source
            .set_actions_deferred(vec![Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli: PauliBasis::X,
                    target: IVec3::Z,
                    direction: Some(Direction::ZPLUS),
                }],
                condition: None,
            }])
            .unwrap();
        source.validate_source().unwrap();
        let error = optimize(source).unwrap_err();
        assert!(error.contains("fixed frames"), "{error}");
    }
}
