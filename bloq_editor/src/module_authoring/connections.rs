//! Mouse and panel composition share one validate-before-commit path.

use super::*;
use crate::components::GraphElement;
use bloq_graph::Block;

pub(crate) fn module_instance_at(
    program: &BlockGraph,
    element: GraphElement,
) -> eyre::Result<Option<String>> {
    let linked = flatten_module_definition(program, program.root(), "")?;
    let position = match element {
        GraphElement::Block(position) => position,
        GraphElement::Pipe(position, _) => linked
            .graph
            .get_endpoint_block(position)
            .map_or(position, Block::pos),
    };
    Ok(linked
        .sites
        .get(&position)
        .and_then(|site| site.instance_path.split("__").next())
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            program
                .root()
                .quantum_connections
                .iter()
                .find_map(|connection| match connection {
                    QuantumConnection::Input {
                        block,
                        input: endpoint,
                        ..
                    }
                    | QuantumConnection::Output {
                        block,
                        output: endpoint,
                        ..
                    } if *block == position => Some(endpoint.instance.clone()),
                    _ => None,
                })
        }))
}

pub(crate) fn module_elements(
    program: &BlockGraph,
    name: &str,
) -> eyre::Result<HashSet<GraphElement>> {
    let group = connected_instances(program.root(), name);
    let linked = flatten_module_definition(program, program.root(), "")?;
    let mut positions = linked
        .sites
        .iter()
        .filter(|(_, site)| {
            site.instance_path
                .split("__")
                .next()
                .is_some_and(|name| group.contains(name))
        })
        .map(|(position, _)| *position)
        .collect::<HashSet<_>>();
    for connection in &program.root().quantum_connections {
        match connection {
            QuantumConnection::Input {
                block,
                input: endpoint,
                ..
            }
            | QuantumConnection::Output {
                block,
                output: endpoint,
                ..
            } if group.contains(&endpoint.instance) => {
                positions.insert(*block);
            }
            _ => {}
        }
    }
    let mut elements = positions
        .iter()
        .copied()
        .map(GraphElement::Block)
        .collect::<HashSet<_>>();
    elements.extend(
        linked
            .graph
            .pipe_endpoints_with_blocks()
            .filter(|(_, _, a, b, _)| positions.contains(&a.pos()) && positions.contains(&b.pos()))
            .map(|(a, b, ..)| GraphElement::Pipe(a, b).canonical()),
    );
    Ok(elements)
}

pub(crate) struct PortSnap {
    pub(crate) moving: IVec3,
    pub(crate) fixed: IVec3,
    pub(crate) offset: IVec3,
}

/// Capture the compatible docking targets once when a drag starts.
pub(crate) fn module_snap_points(program: &BlockGraph, name: &str) -> eyre::Result<Vec<PortSnap>> {
    let ports = exposed_ports(program).collect::<Vec<_>>();
    let group = connected_instances(program.root(), name);
    let mut candidates = Vec::new();
    // ponytail: pairwise port scan for small editor compositions; index by position if it becomes hot.
    for (_, moving) in ports
        .iter()
        .filter(|(endpoint, _)| group.contains(&endpoint.instance))
    {
        for (_, fixed) in ports
            .iter()
            .filter(|(endpoint, _)| !group.contains(&endpoint.instance))
        {
            if moving.direction == fixed.direction || moving.resource_type != fixed.resource_type {
                continue;
            }
            let delta = checked_difference(fixed.position, moving.position)?;
            candidates.push(PortSnap {
                moving: moving.position,
                fixed: fixed.position,
                offset: delta,
            });
        }
    }
    Ok(candidates)
}
