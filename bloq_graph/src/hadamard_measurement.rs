//! Absorb temporal Hadamards into terminal Pauli measurements.

use glam::IVec3;

use crate::{
    Action, BlockGraph, BlockGraphError, BlockKind, FeedbackTarget, PauliBasis, SelectiveKind,
};

/// Only degree-one measurements after an ordinary cube can absorb a frame.
pub(crate) fn is_absorbable_measurement(graph: &BlockGraph, position: IVec3) -> bool {
    let Some(block) = graph.get_block(position) else {
        return false;
    };
    if !matches!(
        block.kind(),
        BlockKind::Measurement(_) | BlockKind::Y | BlockKind::Selective(_)
    ) {
        return false;
    }
    let mut pipes = graph.pipes_at(position);
    let Some(pipe) = pipes.next() else {
        return false;
    };
    if pipes.next().is_some() || pipe.dir().is_spatial() {
        return false;
    }
    let neighbor = if pipe.src() == position {
        pipe.dst()
    } else {
        pipe.src()
    };
    if neighbor.z >= position.z
        || !graph
            .get_block(neighbor)
            .is_some_and(|block| block.kind().is_cube())
    {
        return false;
    }
    let actions = graph.actions();
    if block.kind().is_selective()
        && !actions
            .iter()
            .any(|action| matches!(action, Action::Resolve { target, .. } if *target == position))
    {
        return false;
    }
    true
}

/// The caller owns the transactional clone and validates the complete rewrite.
pub(crate) fn absorb_terminal_hadamards(graph: &mut BlockGraph) -> Result<(), BlockGraphError> {
    let mut endpoints = graph
        .positions()
        .filter(|&position| is_absorbable_measurement(graph, position))
        .filter_map(|position| {
            let pipe = graph.pipes_at(position).next()?;
            pipe.is_hadamard()
                .then_some((position, pipe.src(), pipe.dst()))
        })
        .collect::<Vec<_>>();
    endpoints.sort_unstable_by_key(|(position, _, _)| position.to_array());
    if endpoints.is_empty() {
        return Ok(());
    }
    let mut actions = graph.actions();
    let mut signs = Vec::new();
    for (position, source, target) in endpoints {
        let block = graph
            .get_block_mut(position)
            .ok_or(BlockGraphError::BlockNotFound(position))?;
        let kind = block.kind().flip_xz_basis();
        block.set_kind(kind);
        graph
            .get_pipe_mut(source, target)
            .ok_or(BlockGraphError::PipeNotFound(source, target))?
            .hadamard = false;
        for action in &mut actions {
            match action {
                Action::Resolve { target, condition } if *target == position => {
                    *condition = condition.clone().negated();
                }
                Action::Feedback { targets, .. } => {
                    for target in targets
                        .iter_mut()
                        .filter(|target| target.target == position)
                    {
                        target.pauli = match target.pauli {
                            PauliBasis::X => PauliBasis::Z,
                            PauliBasis::Z => PauliBasis::X,
                            PauliBasis::Y => PauliBasis::Y,
                        };
                    }
                }
                _ => {}
            }
        }
        // HYH=-Y: the other arm's Pauli flips Y while leaving that arm fixed.
        let sign = match kind {
            BlockKind::Y | BlockKind::Selective(SelectiveKind::XY) => Some(PauliBasis::X),
            BlockKind::Selective(SelectiveKind::YZ) => Some(PauliBasis::Z),
            _ => None,
        };
        if let Some(pauli) = sign {
            signs.push(FeedbackTarget {
                pauli,
                target: position,
                direction: None,
            });
        }
    }
    if !signs.is_empty() {
        actions.push(Action::Feedback {
            targets: signs,
            condition: None,
        });
    }
    graph.rebuild_action_graph_lenient(actions);
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "verify")]
    use std::collections::BTreeMap;

    use crate::{Block, CubeKind, Direction, Pipe};
    #[cfg(feature = "verify")]
    use crate::{Expr, MeasureTarget, verify::LogicalVerifier};

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
                    absorb_terminal_hadamards(&mut rewritten).unwrap();
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
            assert!(!is_absorbable_measurement(&graph, IVec3::ZERO));
            let original = graph.to_blog_text();
            absorb_terminal_hadamards(&mut graph).unwrap();
            assert_eq!(graph.to_blog_text(), original);
        }
    }
}
