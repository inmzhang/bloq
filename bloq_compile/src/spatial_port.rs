use std::borrow::Cow;

use bloq_graph::{Basis, BlockGraph, BlockKind, CubeHeight, CubeKind, Direction, PortRole};
use glam::{IVec2, IVec3};

use crate::{CompileError, compile::block_xy_offset};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpatialPortExpansion {
    pub(crate) source: IVec3,
    pub(crate) role: PortRole,
    pub(crate) cube_kind: CubeKind,
    pub(crate) height: CubeHeight,
    /// Multiplex output in virtual-Port-local circuit coordinates.
    pub(crate) output_qubit: Option<IVec2>,
}

impl SpatialPortExpansion {
    pub(crate) const fn is_input(self) -> bool {
        self.role.has_input_boundary()
    }

    pub(crate) const fn cube_pipe_dir(self) -> Direction {
        match self.role {
            PortRole::Input | PortRole::Multiplex => Direction::ZMINUS,
            PortRole::Output => Direction::ZPLUS,
            PortRole::Auto => unreachable!(),
        }
    }

    pub(crate) const fn port_pipe_dir(self) -> Direction {
        self.cube_pipe_dir().negate()
    }

    pub(crate) const fn boundary_basis(self) -> Basis {
        self.cube_kind.y()
    }
}

pub(crate) type SpatialPortExpansionMap = crate::FxMap<IVec3, SpatialPortExpansion>;

/// Replace authored spatial Ports with their compiler-only cube half. The
/// positionless temporal Port half is added by `LowerPlan`.
pub(crate) fn expand_spatial_port_topology(
    source: &BlockGraph,
    distance: u32,
) -> Result<(Cow<'_, BlockGraph>, SpatialPortExpansionMap), CompileError> {
    let mut expansions =
        spatial_port_expansions(source, source.blocks().map(bloq_graph::Block::pos))?;
    allocate_multiplex_outputs(source, distance, &mut expansions)?;
    apply_spatial_port_expansions(source, expansions)
}

/// Expand only complete local sites, using the output coordinates reserved
/// against the full source's occupancy. Halo blocks remain lookup context.
pub(crate) fn expand_spatial_port_context<'a>(
    source: &'a BlockGraph,
    positions: &[IVec3],
    global: &SpatialPortExpansionMap,
) -> Result<(Cow<'a, BlockGraph>, SpatialPortExpansionMap), CompileError> {
    let mut expansions = spatial_port_expansions(source, positions.iter().copied())?;
    for expansion in expansions.values_mut() {
        if expansion.role == PortRole::Multiplex {
            expansion.output_qubit = global[&expansion.source].output_qubit;
        }
    }
    apply_spatial_port_expansions(source, expansions)
}

fn spatial_port_expansions(
    source: &BlockGraph,
    positions: impl IntoIterator<Item = IVec3>,
) -> Result<SpatialPortExpansionMap, CompileError> {
    let mut expansions = SpatialPortExpansionMap::default();
    for position in positions {
        let block = source.get_block(position).expect("selected site exists");
        if !block.kind().is_port() {
            continue;
        }
        let pos = block.pos();
        let Some(pipe) = source.pipes_at(pos).next() else {
            continue;
        };
        if !pipe.dir().is_spatial() {
            continue;
        }
        let neighbor = source
            .get_endpoint_block(if pipe.src() == pos {
                pipe.dst()
            } else {
                pipe.src()
            })
            .expect("Port pipe has a neighbor");
        let role = block.port_role().expect("Port blocks expose a role");
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
            SpatialPortExpansion {
                source: pos,
                role,
                cube_kind,
                height: neighbor.height(),
                output_qubit: None,
            },
        );
    }

    Ok(expansions)
}

fn apply_spatial_port_expansions(
    source: &BlockGraph,
    expansions: SpatialPortExpansionMap,
) -> Result<(Cow<'_, BlockGraph>, SpatialPortExpansionMap), CompileError> {
    if expansions.is_empty() {
        return Ok((Cow::Borrowed(source), expansions));
    }

    let mut graph = source.clone();
    // Geometry edits each refresh the whole action DAG. Its external inputs
    // remain declared while the actions are detached, then one final lenient
    // rebuild preserves the mutators' error and measurement metadata contract.
    let actions = graph.actions();
    graph.set_actions_lenient(Vec::new())?;
    for expansion in expansions.values() {
        graph.set_block_kind(expansion.source, BlockKind::Cube(expansion.cube_kind))?;
        graph.set_cube_height(expansion.source, expansion.height)?;
    }
    graph.set_actions_lenient(actions)?;
    Ok((Cow::Owned(graph), expansions))
}

pub(crate) fn allocate_multiplex_outputs(
    source: &BlockGraph,
    distance: u32,
    expansions: &mut SpatialPortExpansionMap,
) -> Result<(), CompileError> {
    let mut ports = expansions
        .values()
        .filter(|port| port.role == PortRole::Multiplex)
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
            return Err(CompileError::MultiplexOutputCoordinateUnavailable { port });
        };
        occupied.insert(cell);
        expansions
            .get_mut(&port)
            .expect("multiplex source came from expansion map")
            .output_qubit = Some(output);
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

        let (expanded, ports) = expand_spatial_port_topology(&source, 3).unwrap();
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
        let (expanded, _) = expand_spatial_port_topology(&degraded, 3).unwrap();
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
