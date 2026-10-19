use std::borrow::Cow;

use bloq_graph::{Basis, BlockGraph, BlockKind, CubeHeight, CubeKind, Direction, PortRole};
use glam::{IVec2, IVec3};

use crate::{CompileError, compile::block_xy_offset};

/// One ideal boundary, with an optional spatial-to-temporal cube substitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PortExpansion {
    pub(crate) source: IVec3,
    pub(crate) role: PortRole,
    pub(crate) cube: Option<(CubeKind, CubeHeight)>,
    pub(crate) pipe_dir: Direction,
    pub(crate) boundary_basis: Basis,
    /// Private input/output in boundary-template-local circuit coordinates.
    pub(crate) private_qubit: Option<IVec2>,
}

impl PortExpansion {
    pub(crate) const fn is_spatial(self) -> bool {
        self.cube.is_some()
    }

    pub(crate) fn has_future_pipe(self) -> bool {
        self.pipe_dir == Direction::ZPLUS
    }

    /// Pipe a spatial Port's derived cube opens toward its virtual temporal half.
    pub(crate) fn cube_pipe_dir(self) -> Option<Direction> {
        self.cube.map(|_| self.pipe_dir.negate())
    }

    pub(crate) fn needs_private_qubit(self) -> bool {
        !self.is_spatial() || self.role == PortRole::Multiplex
    }
}

pub(crate) type PortExpansionMap = crate::FxMap<IVec3, PortExpansion>;

/// Plan ideal boundaries and replace spatial Ports with their cube half.
/// `LowerPlan` adds the positionless temporal half of each spatial boundary.
pub(crate) fn expand_port_topology(
    source: &BlockGraph,
    distance: u32,
) -> Result<(Cow<'_, BlockGraph>, PortExpansionMap), CompileError> {
    let mut expansions = port_expansions(source, source.blocks().map(bloq_graph::Block::pos))?;
    allocate_private_port_qubits(source, distance, &mut expansions)?;
    apply_port_expansions(source, expansions)
}

/// Expand only complete local sites, using the private coordinates reserved
/// against the full source's occupancy. Halo blocks remain lookup context.
pub(crate) fn expand_port_context<'a>(
    source: &'a BlockGraph,
    positions: &[IVec3],
    global: &PortExpansionMap,
) -> Result<(Cow<'a, BlockGraph>, PortExpansionMap), CompileError> {
    let mut expansions = port_expansions(source, positions.iter().copied())?;
    for expansion in expansions.values_mut() {
        if expansion.needs_private_qubit() {
            expansion.private_qubit = global[&expansion.source].private_qubit;
        }
    }
    apply_port_expansions(source, expansions)
}

fn port_expansions(
    source: &BlockGraph,
    positions: impl IntoIterator<Item = IVec3>,
) -> Result<PortExpansionMap, CompileError> {
    let mut expansions = PortExpansionMap::default();
    for position in positions {
        let block = source.get_block(position).expect("selected site exists");
        if !block.kind().is_port() {
            continue;
        }
        let pos = block.pos();
        let Some(pipe) = source.pipes_at(pos).next() else {
            continue;
        };
        let neighbor = source
            .get_endpoint_block(if pipe.src() == pos {
                pipe.dst()
            } else {
                pipe.src()
            })
            .expect("Port pipe has a neighbor");
        let role = block.port_role().expect("Port blocks expose a role");
        if !pipe.dir().is_spatial() {
            let pipe_dir = if neighbor.pos().z > pos.z {
                Direction::ZPLUS
            } else {
                Direction::ZMINUS
            };
            let inferred = if pipe_dir == Direction::ZPLUS {
                PortRole::Input
            } else {
                PortRole::Output
            };
            if role == PortRole::Auto || role == inferred {
                continue;
            }
            let boundary_basis =
                crate::signature::BlockSignature::temporal_neighbor_boundary_basis(block, source)
                    .ok_or(CompileError::MissingBoundaryBasis {
                    kind: BlockKind::Port,
                })?;
            expansions.insert(
                pos,
                PortExpansion {
                    source: pos,
                    role,
                    cube: None,
                    pipe_dir,
                    boundary_basis,
                    private_qubit: None,
                },
            );
            continue;
        }
        debug_assert_ne!(
            role,
            PortRole::Auto,
            "source validation requires a spatial role"
        );
        let BlockKind::Cube(_) = neighbor.kind() else {
            return Err(CompileError::SpatialPortNeighborNotCube {
                port: pos,
                neighbor: neighbor.pos(),
            });
        };
        if neighbor.height_cells() != 1 {
            return Err(CompileError::TallSpatialPortUnsupported {
                port: pos,
                height: neighbor.height(),
            });
        }
        let cube_kind = source
            .infer_spatial_port_cube_kind(pos)
            .ok_or(CompileError::SpatialPortCubeInferenceFailed { port: pos })?;
        expansions.insert(
            pos,
            PortExpansion {
                source: pos,
                role,
                cube: Some((cube_kind, neighbor.height())),
                pipe_dir: if role.has_input_boundary() {
                    Direction::ZPLUS
                } else {
                    Direction::ZMINUS
                },
                boundary_basis: cube_kind.y(),
                private_qubit: None,
            },
        );
    }

    Ok(expansions)
}

fn apply_port_expansions(
    source: &BlockGraph,
    expansions: PortExpansionMap,
) -> Result<(Cow<'_, BlockGraph>, PortExpansionMap), CompileError> {
    if !expansions.values().any(|port| port.is_spatial()) {
        return Ok((Cow::Borrowed(source), expansions));
    }

    let mut graph = source.clone();
    // Geometry edits each refresh the whole action DAG. Its external inputs
    // remain declared while the actions are detached, then one final lenient
    // rebuild preserves the mutators' error and measurement metadata contract.
    let actions = graph.actions();
    graph.set_actions_lenient(Vec::new())?;
    for expansion in expansions.values() {
        if let Some((kind, height)) = expansion.cube {
            graph.set_block_kind(expansion.source, BlockKind::Cube(kind))?;
            graph.set_cube_height(expansion.source, height)?;
        }
    }
    graph.set_actions_lenient(actions)?;
    Ok((Cow::Owned(graph), expansions))
}

pub(crate) fn allocate_private_port_qubits(
    source: &BlockGraph,
    distance: u32,
    expansions: &mut PortExpansionMap,
) -> Result<(), CompileError> {
    let mut ports = expansions
        .values()
        .filter(|port| port.needs_private_qubit())
        .map(|port| port.source)
        .collect::<Vec<_>>();
    if ports.is_empty() {
        return Ok(());
    }
    ports.sort_unstable_by_key(glam::IVec3::to_array);

    let mut occupied: crate::FxSet<_> = crate::signature::occupied_branch_positions(source)?
        .into_iter()
        .map(IVec3::truncate)
        .collect();
    // Keep clear of compiler templates that spill one cell past source occupancy.
    for cell in occupied.clone() {
        for delta in [IVec2::X, IVec2::Y, IVec2::NEG_X, IVec2::NEG_Y] {
            if let (Some(x), Some(y)) = (cell.x.checked_add(delta.x), cell.y.checked_add(delta.y)) {
                occupied.insert(IVec2::new(x, y));
            }
        }
    }
    for port in ports {
        let cell = port.truncate();
        let limit = occupied.len().saturating_add(1);
        let output = (1..=limit)
            .filter_map(|radius| i32::try_from(radius).ok())
            .flat_map(|radius| {
                [
                    IVec2::new(radius, 0),
                    IVec2::new(0, radius),
                    IVec2::new(-radius, 0),
                    IVec2::new(0, -radius),
                ]
            })
            .find_map(|delta| {
                let candidate =
                    IVec2::new(cell.x.checked_add(delta.x)?, cell.y.checked_add(delta.y)?);
                if occupied.contains(&candidate) {
                    return None;
                }
                block_xy_offset(candidate, distance).ok()?;
                Some((candidate, block_xy_offset(delta, distance).ok()?))
            });
        let Some((cell, output)) = output else {
            return Err(if expansions[&port].role == PortRole::Multiplex {
                CompileError::MultiplexOutputCoordinateUnavailable { port }
            } else {
                CompileError::PrivatePortCoordinateUnavailable { port }
            });
        };
        occupied.insert(cell);
        expansions
            .get_mut(&port)
            .expect("private boundary came from expansion map")
            .private_qubit = Some(output);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloq_graph::{Action, Block, Expr, MeasureTarget, Pipe};

    #[test]
    fn batch_expansion_preserves_actions_inputs_and_lenient_error() {
        let mut source = BlockGraph::new();
        for x in [0, 10] {
            let port = IVec3::new(x, 0, 0);
            source.add_block(
                Block::new(port, BlockKind::Port)
                    .with_port_role(PortRole::Input)
                    .unwrap(),
            );
            source.add_block(Block::new(port + IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
            source.add_pipe(Pipe::new(port, Direction::XPLUS));
        }
        source
            .set_actions_with_inputs(Vec::new(), ["external".to_owned()])
            .unwrap();
        let measure = Action::Measure {
            target: MeasureTarget::Node(IVec3::new(11, 0, 0)),
            name: "measured".into(),
        };
        let copy = Action::Let {
            name: "copy".into(),
            expr: Expr::Var("external".into()),
        };
        source
            .set_actions_lenient(vec![measure.clone(), copy.clone()])
            .unwrap();
        assert!(source.action_graph_error().is_none());
        let measurement = source
            .action_graph()
            .node_by_ordinal(0)
            .unwrap()
            .measurement;
        assert!(measurement.is_some());

        let (expanded, ports) = expand_port_topology(&source, 3).unwrap();
        assert_eq!(ports.len(), 2);
        assert_eq!(expanded.actions(), source.actions());
        assert_eq!(
            expanded.action_graph().inputs().collect::<Vec<_>>(),
            ["external"]
        );
        assert!(expanded.action_graph_error().is_none());
        assert_eq!(
            expanded
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement,
            measurement
        );

        let mut degraded = source.clone();
        degraded
            .set_actions_lenient(vec![
                measure,
                copy,
                Action::Let {
                    name: "invalid".into(),
                    expr: Expr::Var("missing".into()),
                },
            ])
            .unwrap();
        assert!(degraded.action_graph_error().is_some());
        let (expanded, _) = expand_port_topology(&degraded, 3).unwrap();
        assert_eq!(expanded.actions(), degraded.actions());
        assert_eq!(
            expanded.action_graph().inputs().collect::<Vec<_>>(),
            ["external"]
        );
        assert_eq!(
            expanded.action_graph_error().map(ToString::to_string),
            degraded.action_graph_error().map(ToString::to_string)
        );
        assert_eq!(
            expanded
                .action_graph()
                .node_by_ordinal(0)
                .unwrap()
                .measurement,
            measurement
        );
        assert_eq!(
            degraded.get_block(IVec3::ZERO).unwrap().kind(),
            BlockKind::Port
        );
    }
}
