//! Absorb temporal Hadamards into terminal Pauli measurements.

use std::collections::{BTreeSet, HashSet};

use glam::IVec3;

use crate::{
    Action, BlockGraph, BlockGraphError, BlockKind, FeedbackTarget, MeasureTarget, PauliBasis,
    SelectiveKind,
};

/// Degree-one measurements after an ordinary cube that can absorb a frame.
/// Selective caps need a resolve; edge readouts pin their seam's frame.
pub(crate) fn absorbable_measurements(graph: &BlockGraph) -> BTreeSet<[i32; 3]> {
    let (mut resolved, mut seams) = (HashSet::new(), HashSet::new());
    for action in graph.action_iter() {
        match action {
            Action::Resolve { target, .. } => {
                resolved.insert(*target);
            }
            Action::Measure {
                target: MeasureTarget::Edge { src, dir },
                ..
            } => {
                seams.extend([*src, src.wrapping_add(dir.to_ivec3())]);
            }
            _ => {}
        }
    }
    graph
        .blocks()
        .filter(|block| {
            let position = block.pos();
            let mut pipes = graph.pipes_at(position);
            let (Some(pipe), None) = (pipes.next(), pipes.next()) else {
                return false;
            };
            let neighbor = if pipe.src() == position {
                pipe.dst()
            } else {
                pipe.src()
            };
            matches!(
                block.kind(),
                BlockKind::Measurement(_) | BlockKind::Y | BlockKind::Selective(_)
            ) && !pipe.dir().is_spatial()
                && neighbor.z < position.z
                && graph
                    .get_block(neighbor)
                    .is_some_and(|block| block.kind().is_cube())
                && !seams.contains(&position)
                && (!block.kind().is_selective() || resolved.contains(&position))
        })
        .map(|block| block.pos().to_array())
        .collect()
}

/// Absorbs the Hadamards on `absorbable` caps' pipes; returns whether any was.
/// The caller owns the transactional clone and validates the complete rewrite.
pub(crate) fn absorb_terminal_hadamards(
    graph: &mut BlockGraph,
    absorbable: &BTreeSet<[i32; 3]>,
) -> Result<bool, BlockGraphError> {
    let mut absorbed = HashSet::new();
    let mut signs = Vec::new();
    for position in absorbable.iter().copied().map(IVec3::from_array) {
        let pipe = graph
            .pipes_at(position)
            .next()
            .ok_or(BlockGraphError::BlockNotFound(position))?;
        if !pipe.is_hadamard() {
            continue;
        }
        let (source, target) = (pipe.src(), pipe.dst());
        graph
            .get_pipe_mut(source, target)
            .ok_or(BlockGraphError::PipeNotFound(source, target))?
            .hadamard = false;
        let block = graph
            .get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?;
        let kind = block.kind().flip_xz_basis();
        block.set_kind(kind);
        absorbed.insert(position);
        // HYH=-Y: the other arm's Pauli flips Y while leaving that arm fixed.
        let pauli = match kind {
            BlockKind::Y | BlockKind::Selective(SelectiveKind::XY) => PauliBasis::X,
            BlockKind::Selective(SelectiveKind::YZ) => PauliBasis::Z,
            _ => continue,
        };
        signs.push(FeedbackTarget {
            pauli,
            target: position,
            direction: None,
        });
    }
    if absorbed.is_empty() {
        return Ok(false);
    }
    let mut actions = graph.actions();
    crate::graph::conjugate_xz_actions(&mut actions, |position| absorbed.contains(&position));
    if !signs.is_empty() {
        actions.push(Action::Feedback {
            targets: signs,
            condition: None,
        });
    }
    graph.rebuild_action_graph_lenient(actions);
    Ok(true)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "verify")]
    use std::collections::BTreeMap;

    use crate::{Block, CubeKind, Direction, Pipe};
    #[cfg(feature = "verify")]
    use crate::{Expr, verify::LogicalVerifier};

    use super::*;

    #[cfg(feature = "verify")]
    fn cap(kind: BlockKind, reversed: bool, feedback: Option<PauliBasis>) -> BlockGraph {
        let mut graph = BlockGraph::new();
        graph.add_block(Block::new(IVec3::ZERO, BlockKind::Port));
        graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_block(Block::new(2 * IVec3::Z, kind));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS));
        graph.add_pipe(if reversed {
            Pipe::new(2 * IVec3::Z, Direction::ZMINUS).with_hadamard()
        } else {
            Pipe::new(IVec3::Z, Direction::ZPLUS).with_hadamard()
        });
        let mut actions = vec![Action::Measure {
            target: MeasureTarget::Node(2 * IVec3::Z),
            name: "m".into(),
        }];
        if kind.is_selective() {
            actions.push(Action::Resolve {
                target: 2 * IVec3::Z,
                condition: Expr::Var("c".into()),
            });
        }
        if let Some(pauli) = feedback {
            actions.push(Action::Feedback {
                targets: vec![FeedbackTarget {
                    pauli,
                    target: 2 * IVec3::Z,
                    direction: Some(Direction::ZMINUS),
                }],
                condition: Some(Expr::Var("c".into())),
            });
        }
        graph
            .set_actions_with_inputs(actions, ["c".into()])
            .unwrap();
        graph
    }

    #[cfg(feature = "verify")]
    #[test]
    fn terminal_absorption_preserves_each_named_outcome_and_control_choice() {
        for kind in [
            BlockKind::Measurement(crate::Basis::X),
            BlockKind::Measurement(crate::Basis::Z),
            BlockKind::Y,
            BlockKind::Selective(SelectiveKind::XY),
            BlockKind::Selective(SelectiveKind::XZ),
            BlockKind::Selective(SelectiveKind::YZ),
        ] {
            for reversed in [false, true] {
                for feedback in [
                    None,
                    Some(PauliBasis::X),
                    Some(PauliBasis::Y),
                    Some(PauliBasis::Z),
                ] {
                    let source = cap(kind, reversed, feedback);
                    let mut rewritten = source.clone();
                    let absorbable = absorbable_measurements(&rewritten);
                    absorb_terminal_hadamards(&mut rewritten, &absorbable).unwrap();
                    rewritten
                        .set_actions_with_inputs(rewritten.actions(), ["c".into()])
                        .unwrap();
                    rewritten.validate().unwrap();
                    assert!(
                        !rewritten
                            .get_pipe(IVec3::Z, 2 * IVec3::Z)
                            .unwrap()
                            .is_hadamard()
                    );
                    let before = LogicalVerifier::new(&source).unwrap();
                    let after = LogicalVerifier::new(&rewritten).unwrap();
                    for control in [false, true] {
                        for outcome in [false, true] {
                            let values =
                                BTreeMap::from([("c".into(), control), ("m".into(), outcome)]);
                            let (_, expected) = before.instantiate(&values).unwrap();
                            after.verify_branch(&expected.unwrap(), &values).unwrap_or_else(|error| {
                                panic!("{kind:?} reversed={reversed} feedback={feedback:?} c={control} m={outcome}: {error}")
                            });
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn edge_readouts_keep_their_hadamard_seam() {
        for (src, dir) in [
            (IVec3::Z, Direction::ZPLUS),
            (2 * IVec3::Z, Direction::ZMINUS),
        ] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            graph.add_block(Block::new(
                2 * IVec3::Z,
                BlockKind::Measurement(crate::Basis::X),
            ));
            graph.add_pipe(Pipe::new(IVec3::Z, Direction::ZPLUS).with_hadamard());
            graph.rebuild_action_graph_lenient(vec![Action::Measure {
                target: MeasureTarget::Edge { src, dir },
                name: "m".into(),
            }]);
            assert!(absorbable_measurements(&graph).is_empty());
        }
    }

    #[test]
    fn preparation_endpoints_are_protected() {
        for kind in [
            BlockKind::Y,
            BlockKind::T,
            BlockKind::Selective(SelectiveKind::XY),
        ] {
            let mut graph = BlockGraph::new();
            graph.add_block(Block::new(IVec3::ZERO, kind));
            graph.add_block(Block::new(IVec3::Z, BlockKind::Cube(CubeKind::ZXZ)));
            graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::ZPLUS).with_hadamard());
            let absorbable = absorbable_measurements(&graph);
            assert!(absorbable.is_empty());
            let original = graph.to_blog_text();
            absorb_terminal_hadamards(&mut graph, &absorbable).unwrap();
            assert_eq!(graph.to_blog_text(), original);
        }
    }
}
