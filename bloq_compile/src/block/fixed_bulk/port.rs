//! Temporal open-port block compilation.
//!
//! An ordinarily oriented `Port` is an ideal open temporal boundary that terminates a
//! logical operator at a temporal face without closing it into an init/meas cube.
//! It does two things:
//!
//! 1. **Anchors the boundary** — a single [`Op::MPP`] jointly measures every patch
//!    stabilizer (one product per tile); composed with the neighbouring cube's
//!    syndrome round, each record pairs into a boundary detector like a cube seam.
//! 2. **Passes the logical through** — its observable gateway carries the X and Z
//!    lines on *both* faces (`operator_in == operator_out`); the seam toward the
//!    cube cancels and the open face survives as the boundary `OBSERVABLE_INCLUDE`.
//!
//! An ordinary `+Z` pipe is an **input** port (past boundary), and `−Z` is an
//! **output** (future). Multiplex uses the input orientation plus a corrected
//! single-qubit output. The patch is rebuilt from the neighbouring cube's
//! boundary basis.

use std::sync::Arc;

use bloq_circuit::{
    Chunk, ChunkOrLoop, ConditionalCorrection, CoordCircuit, GateType, Op, Pauli, PauliBasis,
    PauliMap,
};
use bloq_graph::{Basis, Direction};
use glam::IVec2;

use crate::CompileError;
use crate::block::fixed_bulk::observable::logical_line_operator;
use crate::block::fixed_bulk::utils::{
    TileFlow, make_normal_surface_code_patch, sorted_unique_coords, tile_flow,
};
use crate::block::gateway::{ChunkMeasurements, GatewayEntry, LocalStabilizer, ObservableGateway};
use crate::block::patch::Patch;
use crate::block::{CompiledTemplate, LoweringTemplate};
use crate::signature::Connectivity;

/// Compile a temporal port into a single stabilizer-MPP chunk plus a passthrough
/// observable gateway.
pub(crate) fn compile_port(
    boundary_basis: Basis,
    connectivity: Connectivity,
    distance: u32,
) -> Result<CompiledTemplate, CompileError> {
    // `+Z` pipe ⇒ input port (past boundary); `−Z` pipe ⇒ output port (future).
    let is_input = connectivity.has_pipe(Direction::ZPLUS);
    let pipe_dir = if is_input {
        Direction::ZPLUS
    } else {
        Direction::ZMINUS
    };

    let patch = make_normal_surface_code_patch(distance, boundary_basis);
    let chunk = make_port_mpp_chunk(&patch, is_input)?;
    let gateway = build_port_observable_gateway(boundary_basis, distance, pipe_dir);

    let chunks = vec![ChunkOrLoop::Single(Box::new(chunk))];
    Ok(Arc::new(LoweringTemplate::from_chunks(chunks, gateway)?))
}

/// Ideal simulation Bell boundary. Both preparation and measurement outcomes
/// contribute to correlation-surface parities through the observable gateway.
pub(crate) fn compile_bell_port(
    boundary_basis: Basis,
    connectivity: Connectivity,
    distance: u32,
    private_qubit: IVec2,
) -> Result<CompiledTemplate, CompileError> {
    let prepares = connectivity.has_pipe(Direction::ZPLUS);
    let patch = make_normal_surface_code_patch(distance, boundary_basis);
    let mut circuit = CoordCircuit::new();
    if prepares {
        circuit.do_gate(GateType::RZ, sorted_unique_coords(patch.data_set()))?;
        circuit.do_gate(GateType::RX, [private_qubit])?;
        circuit.tick();
    }
    let mut gateway = ObservableGateway::new();
    for (basis, pauli) in [(Basis::X, Pauli::X), (Basis::Z, Pauli::Z)] {
        let mut product = logical_line_operator(distance, basis, boundary_basis, pauli);
        product.insert(private_qubit, pauli);
        let measurement = circuit.measure_pauli_products([product.clone()])?[0];
        let (operator_in, operator_out) = if prepares {
            (PauliMap::empty(), product)
        } else {
            (product, PauliMap::empty())
        };
        gateway.insert(
            LocalStabilizer::new(pauli, connectivity),
            GatewayEntry {
                measurements: vec![ChunkMeasurements {
                    chunk_index: 0,
                    measurements: vec![measurement],
                }],
                operator_in,
                operator_out,
            },
        );
    }
    let measurements = circuit.measure_pauli_products(
        patch
            .tiles()
            .iter()
            .map(super::super::patch::Tile::pauli_map),
    )?;
    let direction = if prepares {
        TileFlow::Create
    } else {
        TileFlow::Consume
    };
    let flows = patch
        .tiles()
        .iter()
        .zip(measurements)
        .map(|(tile, measurement)| tile_flow(tile, direction, [measurement]))
        .collect();
    Ok(Arc::new(LoweringTemplate::from_chunks(
        vec![ChunkOrLoop::Single(Box::new(Chunk { circuit, flows }))],
        gateway,
    )?))
}

/// Compile the temporal-input half of a Multiplex spatial Port.
pub(crate) fn compile_multiplex_port(
    boundary_basis: Basis,
    distance: u32,
    output_qubit: IVec2,
) -> Result<CompiledTemplate, CompileError> {
    let patch = make_normal_surface_code_patch(distance, boundary_basis);
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(GateType::RX, [output_qubit])?;
    circuit.tick();

    let mut split = logical_line_operator(distance, Basis::Z, boundary_basis, Pauli::Z);
    split.insert(output_qubit, Pauli::Z);
    let control = circuit.measure_pauli_products([split])?[0];
    circuit.tick();
    circuit
        .body_mut(circuit.entry_body())
        .expect("entry body exists")
        .ops_mut()
        .push(Op::ConditionalPauli(vec![ConditionalCorrection {
            pauli: PauliBasis::X,
            control,
            target: output_qubit,
        }]));
    circuit.tick();

    let products = patch
        .tiles()
        .iter()
        .map(super::super::patch::Tile::pauli_map);
    let measurements = circuit.measure_pauli_products(products)?;
    let flows = patch
        .tiles()
        .iter()
        .zip(measurements)
        .map(|(tile, measurement)| tile_flow(tile, TileFlow::Create, [measurement]))
        .collect::<Vec<_>>();
    let connectivity = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
    let mut gateway = build_port_observable_gateway(boundary_basis, distance, Direction::ZPLUS);
    let key = LocalStabilizer::new(Pauli::X, connectivity);
    let mut x_entry = gateway.lookup(key).expect("input gateway spans X");
    x_entry.operator_out.insert(output_qubit, Pauli::X);
    gateway.insert(key, x_entry);

    let anticommutes = |map: &PauliMap| matches!(map.get(&output_qubit), Some(Pauli::Z | Pauli::Y));
    if flows
        .iter()
        .any(|flow| anticommutes(&flow.start) || anticommutes(&flow.end))
        || [Pauli::X, Pauli::Y, Pauli::Z].into_iter().any(|pauli| {
            let entry = gateway
                .lookup(LocalStabilizer::new(pauli, connectivity))
                .expect("Multiplex gateway spans every Pauli");
            anticommutes(&entry.operator_in) || anticommutes(&entry.operator_out)
        })
    {
        return Err(CompileError::ConditionalPauliFeedbackUnsupported);
    }
    Ok(Arc::new(
        LoweringTemplate::from_chunks_with_commuting_feedback(
            vec![ChunkOrLoop::Single(Box::new(Chunk { circuit, flows }))],
            gateway,
        )?,
    ))
}

/// Build the single MPP chunk measuring every patch stabilizer. Each tile's
/// record anchors a flow matching the cube's stabilizer flow across the seam: an
/// input port flows *out* toward the future cube (`start` empty), an output port
/// flows *in* from the past cube (`end` empty).
fn make_port_mpp_chunk(patch: &Patch, is_input: bool) -> Result<Chunk, CompileError> {
    let mut circuit = CoordCircuit::new();

    let measurements = circuit.measure_pauli_products(
        patch
            .tiles()
            .iter()
            .map(super::super::patch::Tile::pauli_map),
    )?;

    let direction = if is_input {
        TileFlow::Create
    } else {
        TileFlow::Consume
    };
    let flows = patch
        .tiles()
        .iter()
        .zip(&measurements)
        .map(|(tile, &measurement)| tile_flow(tile, direction, [measurement]))
        .collect();

    Ok(Chunk { circuit, flows })
}

/// Build the passthrough observable gateway. The logical X and Z lines run
/// perpendicular through the patch centre ([`logical_line_operator`]). Each key
/// carries the same line on `operator_in` and `operator_out`, so the cube-facing
/// face cancels at the seam and the open face survives as the boundary
/// observable.
fn build_port_observable_gateway(
    boundary_basis: Basis,
    distance: u32,
    pipe_dir: Direction,
) -> ObservableGateway {
    let connectivity = Connectivity::ISOLATED.with_pipe(pipe_dir);

    let mut gateway = ObservableGateway::new();
    for (basis, pauli) in [(Basis::X, Pauli::X), (Basis::Z, Pauli::Z)] {
        let line = logical_line_operator(distance, basis, boundary_basis, pauli);
        gateway.insert(
            LocalStabilizer::new(pauli, connectivity),
            GatewayEntry {
                measurements: Vec::new(),
                operator_in: line.clone(),
                operator_out: line,
            },
        );
    }
    gateway
}

#[cfg(test)]
mod tests {
    use glam::ivec2;
    use rstest::rstest;

    use super::*;

    fn pipe_dir(is_input: bool) -> Direction {
        if is_input {
            Direction::ZPLUS
        } else {
            Direction::ZMINUS
        }
    }

    #[rstest]
    fn test_compile_port_emits_single_mpp_chunk(
        #[values(Basis::X, Basis::Z)] boundary_basis: Basis,
        #[values(3, 5, 7)] distance: u32,
        #[values(true, false)] is_input: bool,
    ) {
        let connectivity = Connectivity::ISOLATED.with_pipe(pipe_dir(is_input));
        let template = compile_port(boundary_basis, connectivity, distance).expect("port compiles");

        // Exactly one stabilizer measurement per patch tile, all from a single MPP.
        let patch = make_normal_surface_code_patch(distance, boundary_basis);
        let circuit = &template.program_template.circuit;
        assert_eq!(
            circuit.num_measurements() as usize,
            patch.tiles().len(),
            "one MPP record per patch stabilizer"
        );
        let mpp_count = circuit
            .body(circuit.entry_body())
            .expect("entry body exists")
            .ops()
            .iter()
            .filter(|op| matches!(op, Op::MPP { .. }))
            .count();
        assert_eq!(mpp_count, 1, "all stabilizers ride one MPP instruction");
    }

    #[rstest]
    fn test_port_gateway_is_passthrough(
        #[values(Basis::X, Basis::Z)] boundary_basis: Basis,
        #[values(true, false)] is_input: bool,
    ) {
        let connectivity = Connectivity::ISOLATED.with_pipe(pipe_dir(is_input));
        let template = compile_port(boundary_basis, connectivity, 3).expect("port compiles");
        let gateway = &template.observable_gateway;
        assert_eq!(gateway.len(), 2, "one passthrough key per logical basis");

        for pauli in [Pauli::X, Pauli::Z] {
            let key = LocalStabilizer::new(pauli, connectivity);
            assert!(
                gateway.contains_key(&key),
                "missing {pauli} passthrough key"
            );
            let entry = &gateway[&key];
            assert!(
                entry.measurements.is_empty(),
                "gateway carries operators only"
            );
            assert_eq!(
                entry.operator_in, entry.operator_out,
                "{pauli} face operators must be a passthrough"
            );
            assert!(
                !entry.operator_in.is_empty(),
                "{pauli} operator is non-empty"
            );
        }
    }

    #[rstest]
    fn test_standalone_port_chunk_has_open_boundary_flow(
        #[values(Basis::X, Basis::Z)] boundary_basis: Basis,
        #[values(true, false)] is_input: bool,
    ) {
        // The single MPP chunk is open on one temporal face (it composes with
        // the neighbouring cube), so every flow survives as boundary residual:
        // an input port flows out toward the future cube (`start` empty), an
        // output port flows in from the past cube (`end` empty).
        let patch = make_normal_surface_code_patch(3, boundary_basis);
        let chunk = make_port_mpp_chunk(&patch, is_input).unwrap();
        let chunks = vec![ChunkOrLoop::Single(Box::new(chunk))];
        let template = LoweringTemplate::from_chunks(chunks, ObservableGateway::new()).unwrap();
        let boundary = &template.program_template.boundary_flows;
        assert_eq!(boundary.len(), patch.tiles().len());
        for flow in boundary {
            if is_input {
                assert!(flow.start.is_empty() && !flow.end.is_empty(), "{flow}");
            } else {
                assert!(!flow.start.is_empty() && flow.end.is_empty(), "{flow}");
            }
        }
    }

    #[test]
    fn multiplex_port_splits_x_support_onto_its_corrected_output() {
        let output = ivec2(-8, 0);
        let template = compile_multiplex_port(Basis::X, 3, output).unwrap();
        let ops = template
            .program_template
            .circuit
            .body(template.program_template.circuit.entry_body())
            .unwrap()
            .ops();
        let [
            Op::Gate {
                gate: GateType::RX, ..
            },
            Op::Tick,
            Op::MPP {
                products,
                measurements,
            },
            Op::Tick,
            Op::ConditionalPauli(corrections),
            Op::Tick,
            Op::MPP { .. },
        ] = ops
        else {
            panic!("unexpected Multiplex circuit")
        };
        assert_eq!(products[0].get(&output), Some(&Pauli::Z));
        assert!(matches!(
            corrections.as_slice(),
            [correction]
                if correction.pauli == PauliBasis::X
                    && correction.control == measurements[0]
                    && correction.target == output
        ));

        let connectivity = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        for pauli in [Pauli::X, Pauli::Y] {
            let entry = template
                .observable_gateway
                .lookup(LocalStabilizer::new(pauli, connectivity))
                .unwrap();
            assert_eq!(entry.operator_in.get(&output), None);
            assert_eq!(entry.operator_out.get(&output), Some(&Pauli::X));
        }
        assert_eq!(
            template
                .observable_gateway
                .lookup(LocalStabilizer::new(Pauli::Z, connectivity))
                .unwrap()
                .operator_out
                .get(&output),
            None
        );
    }
}
