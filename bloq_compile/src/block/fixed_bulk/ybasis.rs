//! Y-basis initialization and measurement block compilation.
//!
//! Reference: arXiv:2302.07395 "Inplace Access to the Surface Code Y Basis"
//!
//! Y blocks perform in-place Y-basis initialization or measurement. They connect
//! to cubes via temporal (Z-axis) pipes only, with at most 1 neighbor.
//!
//! The compiled circuit has 3 chunks:
//! - **Transition**: diagonal twist round, with the final ancilla basis changes
//!   folded into the measurement layer
//! - **Degenerate bulk** (loop, `d//2` repetitions): standard syndrome extraction on rotated patch
//! - **Final**: syndrome extraction + mixed-basis data measurement
//!
//! Initialization = time_reversed(Measurement).
//!
//! # Coordinate system
//!
//! All coordinates are native bloq coordinates (`IVec2`, Y-up). On the
//! `d × d` rotated surface code patch:
//!
//! - **Data qubits** sit at odd `(x, y)` positions: `(2i+1, 2j+1)`.
//! - **Measure qubits** sit at even `(x, y)` positions: `(2i, 2j)`.
//!   Boundary ancillas have one coordinate equal to `0` or `2d`.
//!
//! The **twist diagonal** `(q.x - q.y) / 2 + d - 1` partitions the patch:
//! - `< d-1`: upper-left (UL) region
//! - `= d-1`: twist line (`q.x == q.y` for measure qubits)
//! - `> d-1`: down-right (DR) region
//!
//! Measure-qubit neighbors in the tile grid correspond to fixed offsets:
//! - One step up: `IVec2(0, 2)`
//! - One step left: `IVec2(-2, 0)`
//! - Data-qubit corners: `Corner::TR`, `Corner::BL`, etc. (`±1, ±1`)
//!
//! Patch center for observables: `IVec2(d, d)` (a data qubit when `d` is odd).

use std::sync::Arc;

use bloq_circuit::{Chunk, ChunkOrLoop, CoordCircuit, Flow, GateType, Pauli, PauliBasis, PauliMap};
use bloq_graph::{Basis, Direction};
use glam::IVec2;

use crate::CompileError;
use crate::block::CompiledTemplate;
use crate::block::LoweringTemplate;
use crate::block::fixed_bulk::observable::logical_line_operator;
use crate::block::fixed_bulk::utils::{
    Corner, EdgeBases, TileFlow, checkerboard_basis, make_normal_surface_code_patch,
    make_rectangular_surface_code_patch, make_surface_code_chunk, sorted_unique_coords,
    tile_cx_schedule, tile_flow,
};
use crate::block::gateway::{ChunkMeasurements, GatewayEntry, LocalStabilizer, ObservableGateway};
use crate::block::measurements::MeasurementIndex;
use crate::block::patch::Patch;
use crate::signature::Connectivity;

/// Compile a Y block into init/meas chunks.
pub(super) fn compile_y(
    boundary_basis: Basis,
    connectivity: Connectivity,
    distance: u32,
) -> Result<CompiledTemplate, CompileError> {
    let is_init = connectivity.has_pipe(Direction::ZPLUS);

    // padding_rounds >= 1 because distance >= 3 (odd) so distance / 2 >= 1.
    let padding_rounds = distance / 2;
    let (chunks, gateway) = build_y_chunks(boundary_basis, distance, padding_rounds, is_init)?;

    Ok(Arc::new(LoweringTemplate::from_chunks(chunks, gateway)?))
}

/// Build the three Y-block chunks (transition, bulk loop, final) and observable gateway.
///
/// For measurement: chunks are in forward order.
/// For initialization: chunks are time-reversed and reordered.
///
fn build_y_chunks(
    boundary_basis: Basis,
    d: u32,
    padding_rounds: u32,
    is_init: bool,
) -> Result<(Vec<ChunkOrLoop>, ObservableGateway), CompileError> {
    let degenerate_patch = make_degenerate_patch(d, boundary_basis, false);
    let padding_patch = if is_init {
        make_degenerate_patch(d, boundary_basis, true)
    } else {
        degenerate_patch.clone()
    };
    let padding = make_surface_code_chunk(&padding_patch, None, None)?;
    let data_basis =
        get_final_measure_data_basis(&degenerate_patch.data_set(), boundary_basis, d as i32);

    let (transition, transition_ids) = if is_init {
        make_y_transition_round_reversed(d, boundary_basis)?
    } else {
        make_y_transition_round(d, boundary_basis)?
    };

    let chunks = if is_init {
        vec![
            ChunkOrLoop::Single(Box::new(make_surface_code_chunk(
                &degenerate_patch,
                Some(&data_basis),
                None,
            )?)),
            ChunkOrLoop::Loop {
                body: vec![padding],
                repetitions: padding_rounds,
            },
            ChunkOrLoop::Single(Box::new(transition)),
        ]
    } else {
        vec![
            ChunkOrLoop::Single(Box::new(transition)),
            ChunkOrLoop::Loop {
                body: vec![padding],
                repetitions: padding_rounds,
            },
            ChunkOrLoop::Single(Box::new(make_surface_code_chunk(
                &degenerate_patch,
                None,
                Some(&data_basis),
            )?)),
        ]
    };
    let gateway = build_y_observable_gateway(boundary_basis, d, is_init, &transition_ids);

    Ok((chunks, gateway))
}

fn build_y_observable_gateway(
    boundary_basis: Basis,
    d: u32,
    is_init: bool,
    transition_ids: &MeasurementIndex,
) -> ObservableGateway {
    let pipe_dir = if is_init {
        Direction::ZPLUS
    } else {
        Direction::ZMINUS
    };
    let key = LocalStabilizer::new(Pauli::Y, Connectivity::ISOLATED.with_pipe(pipe_dir));

    let transition_chunk_index = if is_init { 2 } else { 0 };

    let sets = TransitionSets::new(d, boundary_basis);
    let obs_meas_qubits = if !is_init {
        collect_observable_meas_qubits(&sets, d as i32, boundary_basis)
    } else {
        collect_observable_meas_qubits_reversed(&sets, d as i32, boundary_basis)
    };

    let measurements = vec![ChunkMeasurements {
        chunk_index: transition_chunk_index,
        measurements: obs_meas_qubits
            .into_iter()
            .map(|q| transition_ids.expect_measurement(q))
            .collect(),
    }];

    // The Y-basis boundary operator is the product of the patch's X and Z
    // middle lines, which meet at the center data qubit: X·Z = Y
    // there, with X or Z along each arm — exactly the `operator` the neighbouring
    // cube emits for its Y observable. The Y block touches a cube on one temporal
    // face only, so the cross rides `operator_out` when it initialises (a `+Z`
    // pipe carries the state upward) and `operator_in` when it measures (`−Z`).
    let x_line = logical_line_operator(d, Basis::X, boundary_basis, Pauli::X);
    let z_line = logical_line_operator(d, Basis::Z, boundary_basis, Pauli::Z);
    let cross = &x_line ^ &z_line;
    let (operator_in, operator_out) = if is_init {
        (PauliMap::empty(), cross)
    } else {
        (cross, PauliMap::empty())
    };

    let mut gateway = ObservableGateway::new();
    gateway.insert(
        key,
        GatewayEntry {
            measurements,
            operator_in,
            operator_out,
        },
    );
    gateway
}

/// Compute the twist diagonal value for a qubit as an integer.
///
/// Unified formula for both data qubits (odd coords) and measure qubits (even
/// coords): `(q.x - q.y) / 2 + d - 1`. Works because `q.x - q.y` is always
/// even on the surface code lattice.
fn twist_diagonal(q: IVec2, d: i32) -> i32 {
    (q.x - q.y) / 2 + d - 1
}

/// Build the degenerate patch after Y-basis transition.
///
/// `top_basis` is the top/bottom boundary basis of the *standard* patch
/// before the transition. It determines both the degenerate boundary layout
/// and the CX hook orientation (passed through as `horizontal_hook_basis`).
fn make_degenerate_patch(d: u32, top_basis: Basis, reverse_schedule: bool) -> Patch {
    // The transition rotates the boundary assignment a quarter turn: top and
    // right take `top_basis.flip()` and `top_basis`, bottom and left the
    // reverse.
    let flipped = top_basis.flip();
    let boundaries = EdgeBases {
        top: flipped,
        bottom: top_basis,
        left: top_basis,
        right: flipped,
    };
    make_rectangular_surface_code_patch(d, d, &boundaries, top_basis, reverse_schedule)
}

/// Pre-computed measure qubit classification for the transition round.
///
/// Shared between `make_y_transition_round` and `build_y_observable_gateway`
/// to avoid duplicating the classification logic.
struct TransitionSets {
    used: crate::FxSet<IVec2>,
    xs: crate::FxSet<IVec2>,
    zs: crate::FxSet<IVec2>,
    top_row: crate::FxSet<IVec2>,
    left_col: crate::FxSet<IVec2>,
}

impl TransitionSets {
    fn new(d: u32, top_basis: Basis) -> Self {
        let start_patch = make_normal_surface_code_patch(d, top_basis);
        let end_patch = make_degenerate_patch(d, top_basis, false);
        Self::from_patches(d, &start_patch, &end_patch)
    }

    fn from_patches(d: u32, start_patch: &Patch, end_patch: &Patch) -> Self {
        let used: crate::FxSet<IVec2> = start_patch
            .used_set()
            .union(&end_patch.used_set())
            .copied()
            .collect();

        let xs: crate::FxSet<IVec2> = used
            .iter()
            .copied()
            .filter(|&q| is_measure_qubit(q) && checkerboard_basis(q) == Basis::X)
            .collect();
        let zs: crate::FxSet<IVec2> = used
            .iter()
            .copied()
            .filter(|&q| is_measure_qubit(q) && checkerboard_basis(q) == Basis::Z)
            .collect();

        let top_row: crate::FxSet<IVec2> = used
            .iter()
            .copied()
            .filter(|&q| q.y == 2 * d as i32 && is_measure_qubit(q))
            .collect();
        let left_col: crate::FxSet<IVec2> = used
            .iter()
            .copied()
            .filter(|&q| q.x == 0 && is_measure_qubit(q))
            .collect();

        Self {
            used,
            xs,
            zs,
            top_row,
            left_col,
        }
    }
}

/// Qubits split by the coarse twist band `q.x - q.y`: upper-left below `-2`,
/// down-right at or above `+2`, middle in between.
#[derive(Default)]
struct DiagSplit {
    ul: crate::FxSet<IVec2>,
    md: crate::FxSet<IVec2>,
    dr: crate::FxSet<IVec2>,
}

impl DiagSplit {
    fn new(qs: &crate::FxSet<IVec2>) -> Self {
        let mut split = Self::default();
        for &q in qs {
            let diff = q.x - q.y;
            let band = if diff < -2 {
                &mut split.ul
            } else if diff >= 2 {
                &mut split.dr
            } else {
                &mut split.md
            };
            band.insert(q);
        }
        split
    }

    fn ul_md(&self) -> crate::FxSet<IVec2> {
        self.ul.union(&self.md).copied().collect()
    }
}

struct TransitionBoundaries {
    old_x: crate::FxSet<IVec2>,
    new_x: crate::FxSet<IVec2>,
    old_z: crate::FxSet<IVec2>,
    new_z: crate::FxSet<IVec2>,
}

impl TransitionBoundaries {
    fn new(sets: &TransitionSets, top_basis: Basis) -> Self {
        if top_basis == Basis::X {
            Self {
                old_x: sets.top_row.clone(),
                new_x: sets.left_col.clone(),
                old_z: sets.left_col.clone(),
                new_z: sets.top_row.clone(),
            }
        } else {
            Self {
                old_x: sets.left_col.clone(),
                new_x: sets.top_row.clone(),
                old_z: sets.top_row.clone(),
                new_z: sets.left_col.clone(),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum BoundarySide {
    Old,
    New,
}

struct TransitionRoundContext {
    d: i32,
    standard_patch: Patch,
    degenerate_patch: Patch,
    sets: TransitionSets,
    xs: DiagSplit,
    zs: DiagSplit,
    boundaries: TransitionBoundaries,
    x_order: [Corner; 4],
    z_order: [Corner; 4],
}

impl TransitionRoundContext {
    fn new(d: u32, top_basis: Basis) -> Self {
        let standard_patch = make_normal_surface_code_patch(d, top_basis);
        let degenerate_patch = make_degenerate_patch(d, top_basis, false);
        let sets = TransitionSets::from_patches(d, &standard_patch, &degenerate_patch);
        let xs = DiagSplit::new(&sets.xs);
        let zs = DiagSplit::new(&sets.zs);
        let boundaries = TransitionBoundaries::new(&sets, top_basis);

        Self {
            d: d as i32,
            standard_patch,
            degenerate_patch,
            sets,
            xs,
            zs,
            boundaries,
            x_order: tile_cx_schedule(Basis::X, top_basis).compact(),
            z_order: tile_cx_schedule(Basis::Z, top_basis).compact(),
        }
    }

    fn boundary_targets(&self, side: BoundarySide) -> (Vec<IVec2>, Vec<IVec2>) {
        match side {
            BoundarySide::Old => (
                swapped_boundary_targets(
                    &self.sets.xs,
                    &self.boundaries.new_x,
                    &self.boundaries.old_x,
                ),
                swapped_boundary_targets(
                    &self.sets.zs,
                    &self.boundaries.new_z,
                    &self.boundaries.old_z,
                ),
            ),
            BoundarySide::New => (
                swapped_boundary_targets(
                    &self.sets.xs,
                    &self.boundaries.old_x,
                    &self.boundaries.new_x,
                ),
                swapped_boundary_targets(
                    &self.sets.zs,
                    &self.boundaries.old_z,
                    &self.boundaries.new_z,
                ),
            ),
        }
    }
}

fn is_measure_qubit(q: IVec2) -> bool {
    q.x % 2 == 0 && q.y % 2 == 0
}

fn swapped_boundary_targets(
    all: &crate::FxSet<IVec2>,
    excluded: &crate::FxSet<IVec2>,
    included: &crate::FxSet<IVec2>,
) -> Vec<IVec2> {
    sorted_unique_coords(
        all.difference(excluded)
            .copied()
            .chain(included.iter().copied()),
    )
}

fn active_targets(
    all: &crate::FxSet<IVec2>,
    excluded: &crate::FxSet<IVec2>,
) -> crate::FxSet<IVec2> {
    all.difference(excluded).copied().collect()
}

fn transition_rotation_targets(used: &crate::FxSet<IVec2>, d: i32) -> (Vec<IVec2>, Vec<IVec2>) {
    let h_targets = sorted_unique_coords(
        used.iter()
            .copied()
            .filter(|&q| twist_diagonal(q, d) < d - 1),
    );
    let twist_targets = sorted_unique_coords(
        used.iter()
            .copied()
            .filter(|&q| is_measure_qubit(q) && twist_diagonal(q, d) == d - 1),
    );
    (h_targets, twist_targets)
}

fn emit_transition_measurements(
    circuit: &mut CoordCircuit,
    mx_targets: Vec<IVec2>,
    my_target: Option<IVec2>,
    mz_targets: Vec<IVec2>,
) {
    let mut measurement_bases = crate::FxMap::default();
    for target in mx_targets {
        measurement_bases.insert(target, PauliBasis::X);
    }
    for target in mz_targets {
        measurement_bases.insert(target, PauliBasis::Z);
    }
    if let Some(target) = my_target {
        measurement_bases.insert(target, PauliBasis::Y);
    }

    for basis in [PauliBasis::X, PauliBasis::Y, PauliBasis::Z] {
        let targets = sorted_unique_coords(
            measurement_bases
                .iter()
                .filter_map(|(&target, &target_basis)| (target_basis == basis).then_some(target)),
        );
        if !targets.is_empty() {
            circuit.measure(basis, targets);
        }
    }
}

fn patch_flows(
    patch: &Patch,
    measurement_ids: &MeasurementIndex,
    direction: TileFlow,
    mids_for_tile: impl Fn(IVec2, Basis) -> Vec<IVec2>,
) -> Vec<Flow> {
    patch
        .tiles()
        .iter()
        .map(|tile| {
            let measurements = mids_for_tile(tile.measure_qubit(), tile.basis())
                .into_iter()
                .map(|q| measurement_ids.expect_measurement(q));
            tile_flow(tile, direction, measurements)
        })
        .collect()
}

/// Generate CX pairs: for each qubit in `qs` (visited in `(y, x)` order), if
/// `q + delta` is in `used`, produce the pair ordered by `sign`
/// (+1 = measure-qubit-first, -1 = target-first).
fn toward(
    qs: &crate::FxSet<IVec2>,
    delta: Corner,
    sign: i32,
    used: &crate::FxSet<IVec2>,
) -> Vec<IVec2> {
    let offset = delta.to_ivec2();
    let mut pairs: Vec<[IVec2; 2]> = Vec::new();
    for q in sorted_unique_coords(qs.iter().copied()) {
        let target = q + offset;
        if used.contains(&target) {
            let pair = if sign > 0 { [q, target] } else { [target, q] };
            pairs.push(pair);
        }
    }
    pairs.into_iter().flatten().collect()
}

#[derive(Clone, Copy)]
enum TransitionDirection {
    Forward,
    Reversed,
}

/// Build the Y-basis transition round chunk, with the measurement index its
/// flows were resolved against — the observable gateway needs the same one.
fn make_y_transition_round(
    d: u32,
    top_basis: Basis,
) -> Result<(Chunk, MeasurementIndex), CompileError> {
    make_transition_round(d, top_basis, TransitionDirection::Forward)
}

fn make_y_transition_round_reversed(
    d: u32,
    top_basis: Basis,
) -> Result<(Chunk, MeasurementIndex), CompileError> {
    make_transition_round(d, top_basis, TransitionDirection::Reversed)
}

fn make_transition_round(
    d: u32,
    top_basis: Basis,
    direction: TransitionDirection,
) -> Result<(Chunk, MeasurementIndex), CompileError> {
    let ctx = TransitionRoundContext::new(d, top_basis);
    let mut circuit = CoordCircuit::new();

    match direction {
        TransitionDirection::Forward => emit_forward_transition(&mut circuit, &ctx, top_basis)?,
        TransitionDirection::Reversed => emit_reversed_transition(&mut circuit, &ctx, top_basis)?,
    }

    // The transition emits its measurements through `CoordCircuit::measure`
    // batches rather than a recording helper, so the index is recovered once
    // here and shared by every consumer.
    let measurement_ids = MeasurementIndex::from_circuit(&circuit);
    Ok((
        Chunk {
            flows: transition_flows(&ctx, &measurement_ids, top_basis, direction),
            circuit,
        },
        measurement_ids,
    ))
}

fn emit_forward_transition(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    top_basis: Basis,
) -> Result<(), CompileError> {
    emit_boundary_resets(circuit, ctx, BoundarySide::Old, None)?;
    emit_outer_interactions(circuit, ctx, [3, 2])?;
    emit_diagonal_interactions(circuit, ctx)?;
    emit_corner_interactions(circuit, ctx)?;
    emit_middle_z_cy(circuit, ctx)?;
    emit_rotation_layer(circuit, ctx, GateType::S)?;
    emit_measurement_layer(
        circuit,
        ctx,
        BoundarySide::New,
        Some(y_corner_qubit(ctx.d, top_basis)),
    );
    Ok(())
}

fn emit_reversed_transition(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    top_basis: Basis,
) -> Result<(), CompileError> {
    emit_boundary_resets(
        circuit,
        ctx,
        BoundarySide::New,
        Some(y_corner_qubit(ctx.d, top_basis)),
    )?;
    emit_rotation_layer(circuit, ctx, GateType::S_DAG)?;
    emit_middle_z_cy(circuit, ctx)?;
    emit_corner_interactions(circuit, ctx)?;
    emit_diagonal_interactions(circuit, ctx)?;
    emit_outer_interactions(circuit, ctx, [2, 3])?;
    emit_measurement_layer(circuit, ctx, BoundarySide::Old, None);
    Ok(())
}

fn emit_boundary_resets(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    side: BoundarySide,
    ry_target: Option<IVec2>,
) -> Result<(), CompileError> {
    let (rx_targets, rz_targets) = ctx.boundary_targets(side);
    circuit.do_gate(GateType::RX, rx_targets)?;
    if let Some(target) = ry_target {
        circuit.do_gate(GateType::RY, std::iter::once(target))?;
    }
    circuit.do_gate(GateType::RZ, rz_targets)?;
    circuit.tick();
    Ok(())
}

fn emit_outer_interactions(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    order: [usize; 2],
) -> Result<(), CompileError> {
    let xs_active = active_targets(&ctx.sets.xs, &ctx.boundaries.new_x);
    let zs_active = active_targets(&ctx.sets.zs, &ctx.boundaries.new_z);

    for idx in order {
        emit_cx_pair(circuit, &xs_active, ctx.x_order[idx], 1, &ctx.sets.used)?;
        emit_cx_pair(circuit, &zs_active, ctx.z_order[idx], -1, &ctx.sets.used)?;
        circuit.tick();
    }
    Ok(())
}

fn emit_diagonal_interactions(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
) -> Result<(), CompileError> {
    emit_cx_pair(circuit, &ctx.xs.ul, ctx.x_order[1], -1, &ctx.sets.used)?;
    emit_cx_pair(circuit, &ctx.zs.ul_md(), ctx.z_order[1], 1, &ctx.sets.used)?;
    emit_gate_pair(
        circuit,
        GateType::CY,
        &ctx.xs.md,
        ctx.x_order[1],
        1,
        &ctx.sets.used,
    )?;
    emit_cx_pair(circuit, &ctx.xs.dr, ctx.x_order[1], 1, &ctx.sets.used)?;
    emit_cx_pair(circuit, &ctx.zs.dr, ctx.z_order[1], -1, &ctx.sets.used)?;
    circuit.tick();
    Ok(())
}

fn emit_corner_interactions(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
) -> Result<(), CompileError> {
    emit_cx_pair(circuit, &ctx.xs.ul, ctx.x_order[3], -1, &ctx.sets.used)?;
    emit_cx_pair(circuit, &ctx.zs.ul, ctx.x_order[3], 1, &ctx.sets.used)?;
    emit_cx_pair(circuit, &ctx.xs.dr, ctx.x_order[0], 1, &ctx.sets.used)?;
    emit_cx_pair(circuit, &ctx.zs.dr, ctx.z_order[0], -1, &ctx.sets.used)?;
    circuit.tick();
    Ok(())
}

fn emit_middle_z_cy(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
) -> Result<(), CompileError> {
    let zs_md_active = active_targets(&ctx.zs.md, &ctx.boundaries.old_z);
    emit_gate_pair(
        circuit,
        GateType::CY,
        &zs_md_active,
        ctx.z_order[3],
        -1,
        &ctx.sets.used,
    )?;
    circuit.tick();
    Ok(())
}

fn emit_rotation_layer(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    twist_gate: GateType,
) -> Result<(), CompileError> {
    let (h_targets, twist_targets) = transition_rotation_targets(&ctx.sets.used, ctx.d);
    circuit.do_gate(GateType::H, h_targets)?;
    circuit.do_gate(twist_gate, twist_targets)?;
    circuit.tick();
    Ok(())
}

fn emit_measurement_layer(
    circuit: &mut CoordCircuit,
    ctx: &TransitionRoundContext,
    side: BoundarySide,
    my_target: Option<IVec2>,
) {
    let (mx_targets, mz_targets) = ctx.boundary_targets(side);
    emit_transition_measurements(circuit, mx_targets, my_target, mz_targets)
}

fn transition_flows(
    ctx: &TransitionRoundContext,
    measurement_ids: &MeasurementIndex,
    top_basis: Basis,
    direction: TransitionDirection,
) -> Vec<Flow> {
    let reversed = matches!(direction, TransitionDirection::Reversed);
    // The transition carries the standard patch to the degenerate one; running
    // it backwards swaps which patch is the input.
    let (input_patch, output_patch) = if reversed {
        (&ctx.degenerate_patch, &ctx.standard_patch)
    } else {
        (&ctx.standard_patch, &ctx.degenerate_patch)
    };

    let mut input_flows = patch_flows(
        input_patch,
        measurement_ids,
        TileFlow::Consume,
        |m, tile_basis| transition_in_flow_mids(m, tile_basis, ctx.d, top_basis, reversed),
    );
    let mut output_flows = patch_flows(
        output_patch,
        measurement_ids,
        TileFlow::Create,
        |m, tile_basis| {
            if reversed {
                standard_out_flow_mids(m, tile_basis, ctx.d, top_basis)
            } else {
                degenerate_out_flow_mids(m, ctx.d, top_basis)
            }
        },
    );

    // The twist turns the degenerate-patch stabilizer adjacent to the Y corner
    // negative. Stim's signed flow check pins this convention in both time
    // directions.
    let y_corner = y_corner_qubit(ctx.d, top_basis);
    let offset = if top_basis == Basis::X {
        IVec2::ONE
    } else {
        IVec2::NEG_ONE
    };
    let signed_flows = if reversed {
        &mut input_flows
    } else {
        &mut output_flows
    };
    signed_flows
        .iter_mut()
        .find(|flow| flow.center == Some(y_corner + offset))
        .expect("the degenerate patch contains the Y-corner stabilizer")
        .sign = true;

    input_flows.extend(output_flows);
    input_flows
}

/// Emit CX gate pairs from `toward()`, skipping if empty.
fn emit_cx_pair(
    circuit: &mut CoordCircuit,
    qs: &crate::FxSet<IVec2>,
    delta: Corner,
    sign: i32,
    used: &crate::FxSet<IVec2>,
) -> Result<(), CompileError> {
    emit_gate_pair(circuit, GateType::CX, qs, delta, sign, used)
}

/// Emit an arbitrary 2-qubit gate with `toward()` pairs, skipping if empty.
fn emit_gate_pair(
    circuit: &mut CoordCircuit,
    gate: GateType,
    qs: &crate::FxSet<IVec2>,
    delta: Corner,
    sign: i32,
    used: &crate::FxSet<IVec2>,
) -> Result<(), CompileError> {
    let pairs = toward(qs, delta, sign, used);
    if !pairs.is_empty() {
        circuit.do_gate(gate, pairs)?;
    }
    Ok(())
}

/// The single MY corner qubit for the transition round.
fn y_corner_qubit(d: i32, top_basis: Basis) -> IVec2 {
    if top_basis == Basis::X {
        // data_coord(0, d-1, d) = (1, 1)
        IVec2::new(1, 1)
    } else {
        // data_coord(d-1, 0, d) = (2d-1, 2d-1)
        IVec2::new(2 * d - 1, 2 * d - 1)
    }
}

/// Collect the measurement qubits for the observable flow in Y measurement block.
///
/// Includes the MY corner qubit plus measure qubits in the upper-right and
/// down-left quadrants (relative to the center at `(d, d)`).
fn collect_observable_meas_qubits(sets: &TransitionSets, d: i32, top_basis: Basis) -> Vec<IVec2> {
    let center = IVec2::new(d, d);

    let my_target = y_corner_qubit(d, top_basis);

    let (mset_ur, mset_dl) = if top_basis == Basis::Z {
        (&sets.xs, &sets.zs)
    } else {
        (&sets.zs, &sets.xs)
    };

    let mut ms = vec![my_target];

    // Upper-right quadrant (bloq Y-up: large x, large y)
    for &q in mset_ur.iter().chain(sets.top_row.iter()) {
        if q.x > center.x && q.y > center.y {
            ms.push(q);
        }
    }
    // Down-left quadrant (bloq Y-up: small x, small y)
    for &q in mset_dl.iter().chain(sets.left_col.iter()) {
        if q.x < center.x && q.y < center.y {
            ms.push(q);
        }
    }
    sorted_unique_coords(ms)
}

/// Collect the measurement qubits for the observable flow in Y init block.
fn collect_observable_meas_qubits_reversed(
    sets: &TransitionSets,
    d: i32,
    top_basis: Basis,
) -> Vec<IVec2> {
    let center = IVec2::new(d, d);

    let mut ms = Vec::new();
    let (mset0, mset1) = if top_basis == Basis::Z {
        (&sets.zs, &sets.xs)
    } else {
        (&sets.xs, &sets.zs)
    };
    for &q in mset0.iter().chain(sets.top_row.iter()) {
        if (q.x > center.x && q.y > q.x) || (q.x < center.x && q.y <= q.x) {
            ms.push(q);
        }
    }
    for &q in mset1.iter().chain(sets.left_col.iter()) {
        if (q.x > center.x && q.y <= q.x && q.y > center.y)
            || (q.x < center.x && q.y > q.x && q.y < center.y)
        {
            ms.push(q);
        }
    }
    sorted_unique_coords(ms)
}

/// Measure-qubit offset: one tile step upward in bloq coords.
const UP: IVec2 = IVec2::new(0, 2);
/// Measure-qubit offset: one tile step leftward in bloq coords.
const LEFT: IVec2 = IVec2::new(-2, 0);

/// One tile step along the twist, as a measure-qubit offset.
///
/// The forward round borrows the partner above for an `X`-top patch and to the
/// left for a `Z`-top one; running the round backwards borrows the other way.
fn twist_step(top_basis: Basis, reversed: bool) -> IVec2 {
    if (top_basis == Basis::X) ^ reversed {
        UP
    } else {
        LEFT
    }
}

/// Measurement references for a transition round's **input** flow, in both
/// time directions.
///
/// `m` is the tile's measure qubit. The patch splits into three: the twist
/// line `m.x == m.y`, the upper-left bulk `m.x < m.y`, and everything else —
/// the down-right region plus the two boundary strips, which all read their
/// own ancilla.
///
/// The forward and reverse rounds differ only in which way the twist leans:
/// upper-left tiles borrow their partner one step along it, with `Z` tiles
/// leaning the opposite way from `X` tiles. (The reverse round used to test
/// all four patch edges where the forward round tested two; the extra two lie
/// in the down-right region, which answers `[m]` either way, so the wider test
/// was vacuous.)
fn transition_in_flow_mids(
    m: IVec2,
    tile_basis: Basis,
    d: i32,
    top_basis: Basis,
    reversed: bool,
) -> Vec<IVec2> {
    if m.x == m.y {
        let partner = m + twist_step(top_basis, reversed);
        return if reversed {
            vec![partner, m]
        } else {
            vec![m, partner]
        };
    }
    if m.x < m.y && m.x != 0 && m.y != 2 * d {
        let lean = reversed ^ (tile_basis == Basis::Z);
        return vec![m + twist_step(top_basis, lean)];
    }
    vec![m]
}

/// Measurement references for the **degenerate** patch's creator flows — the
/// forward round's output interface.
fn degenerate_out_flow_mids(m: IVec2, d: i32, top_basis: Basis) -> Vec<IVec2> {
    let is_xtop = top_basis == Basis::X;

    // Special corner cases at the twist-line endpoints.
    // (2d-2, 2d-2): top-right twist endpoint (Z-top only).
    if m.x == 2 * d - 2 && m.y == 2 * d - 2 && !is_xtop {
        return vec![m, m + LEFT, m + UP, m + Corner::TR.to_ivec2()];
    }
    // (2, 2): bottom-left twist endpoint (X-top only).
    if m.x == 2 && m.y == 2 && is_xtop {
        return vec![m, m + LEFT, m + UP, m + Corner::BL.to_ivec2()];
    }
    // (2, 2d): top boundary singleton (Z-top only).
    if m.x == 2 && m.y == 2 * d && !is_xtop {
        return vec![m];
    }
    // (0, 2(d-1)): left boundary singleton (X-top only).
    if m.x == 0 && m.y == 2 * (d - 1) && is_xtop {
        return vec![m];
    }

    // Top row: pair with left neighbor.
    if m.y == 2 * d {
        return vec![m, m + LEFT];
    }
    // Left column: pair with upper neighbor.
    if m.x == 0 {
        return vec![m, m + UP];
    }

    // Twist line: pair with both neighbors.
    if m.x == m.y {
        return vec![m, m + LEFT, m + UP];
    }

    vec![m]
}

/// Measurement references for the **standard** patch's creator flows — the
/// reverse round's output interface.
///
/// Kept separate from [`degenerate_out_flow_mids`]: the two describe different
/// patches, not two views of one rule.
fn standard_out_flow_mids(m: IVec2, tile_basis: Basis, d: i32, top_basis: Basis) -> Vec<IVec2> {
    if tile_basis == top_basis.flip() && m.x == 2 {
        return vec![m, m + LEFT];
    }
    if tile_basis == top_basis && m.y == 2 * d - 2 {
        return vec![m + UP, m];
    }
    vec![m]
}

fn get_final_measure_data_basis(
    data_qubits: &crate::FxSet<IVec2>,
    top_basis: Basis,
    d: i32,
) -> crate::FxMap<IVec2, Basis> {
    data_qubits
        .iter()
        .map(|&q| {
            // In tile coords: tx < ty (Z-top) or tx <= ty (X-top) determines basis.
            // In bloq coords: q.x + q.y < 2*d or q.x + q.y <= 2*d.
            let basis = if top_basis == Basis::Z {
                if q.x + q.y < 2 * d {
                    Basis::Z
                } else {
                    Basis::X
                }
            } else if q.x + q.y <= 2 * d {
                Basis::X
            } else {
                Basis::Z
            };
            (q, basis)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use bloq_circuit::Op;
    use bloq_graph::{Basis, BlockKind, Direction};
    use bloq_stim::StimFlowVerifier;
    use rstest::rstest;

    use super::*;
    use crate::block::fixed_bulk::{
        compile_fixed_bulk, loop_repetitions, split_ops_into_moments, validate_fixed_bulk,
    };
    use crate::signature::BlockSignature;

    fn compile_and_verify(
        basis: Basis,
        distance: u32,
        pipe_dir: Direction,
    ) -> Arc<LoweringTemplate> {
        let connectivity = Connectivity::ISOLATED.with_pipe(pipe_dir);
        let is_init = connectivity.has_pipe(Direction::ZPLUS);
        let (chunks, gateway) = build_y_chunks(basis, distance, distance / 2, is_init).unwrap();
        for chunk in &chunks {
            chunk.verify_flows(None, None).unwrap();
        }
        Arc::new(LoweringTemplate::from_chunks(chunks, gateway).unwrap())
    }

    /// The two time directions' input-flow tables used to differ in their
    /// boundary test: the reverse round checked all four patch edges, the
    /// forward round only the left column and top row. That difference was
    /// vacuous — the extra two edges lie in the down-right region, which
    /// answers `[m]` under either table — which is what lets one rule serve
    /// both directions.
    #[rstest]
    fn transition_in_flow_extra_edges_lie_in_the_down_right_region(
        #[values(3, 5, 7)] d: u32,
        #[values(Basis::X, Basis::Z)] top_basis: Basis,
    ) {
        let d = d as i32;
        let extra_edge = (0..=d).flat_map(|i| [IVec2::new(2 * d, 2 * i), IVec2::new(2 * i, 0)]);
        for m in extra_edge.filter(|m| m.x != m.y) {
            assert!(m.x > m.y, "{m} is outside the down-right region");
            for tile_basis in [Basis::X, Basis::Z] {
                for reversed in [false, true] {
                    assert_eq!(
                        transition_in_flow_mids(m, tile_basis, d, top_basis, reversed),
                        vec![m],
                        "{m} {tile_basis:?} reversed={reversed}",
                    );
                }
            }
        }
    }

    #[rstest]
    fn test_y_compile(
        #[values(Basis::X, Basis::Z)] basis: Basis,
        #[values(3, 5, 7, 9)] d: u32,
        #[values(Direction::ZMINUS, Direction::ZPLUS)] pipe_dir: Direction,
    ) {
        let template = compile_and_verify(basis, d, pipe_dir);
        let expected_reps = d / 2;
        if expected_reps == 1 {
            assert_eq!(loop_repetitions(&template), None);
        } else {
            assert_eq!(loop_repetitions(&template), Some(expected_reps));
        }
    }

    #[rstest]
    fn test_y_transition_round_extracts_basis_changes_into_rotation_moment(
        #[values(Basis::X, Basis::Z)] basis: Basis,
    ) {
        let (transition, _) = make_y_transition_round(5, basis).expect("transition chunk");

        let moments = split_ops_into_moments(
            transition
                .circuit
                .body(transition.circuit.entry_body())
                .unwrap()
                .ops(),
        );
        let rotation_moment = moments
            .get(6)
            .expect("transition keeps a pre-measurement rotation moment");
        assert!(
            rotation_moment.iter().any(|op| matches!(
                op,
                Op::Gate {
                    gate: GateType::H,
                    ..
                }
            )),
            "data-qubit Hadamards should move into the explicit rotation moment"
        );
        assert!(
            rotation_moment.iter().any(|op| matches!(
                op,
                Op::Gate {
                    gate: GateType::S,
                    ..
                }
            )),
            "twist-line basis changes should be represented explicitly as S rotations"
        );

        let final_moment = moments
            .get(7)
            .expect("transition keeps the final measurement moment");
        assert!(
            final_moment
                .iter()
                .any(|op| matches!(op, Op::Measure { .. })),
            "final moment should still contain the measurements"
        );
        assert!(
            final_moment
                .iter()
                .all(|op| matches!(op, Op::Measure { .. })),
            "final moment should be measurement-only after extracting rotations"
        );
    }

    #[test]
    fn test_fixed_bulk_compiler_y_dispatch() {
        let sig = BlockSignature {
            kind: BlockKind::Y,
            rounds: None,
            connectivity: Connectivity::ISOLATED.with_pipe(Direction::ZMINUS),
            boundary_basis: Some(Basis::X),
            layer_schedule: None,
            surgery_side: None,
        };
        let template = compile_fixed_bulk(sig, 3).unwrap();
        assert!(template.program_template.circuit.num_measurements() > 0);
    }

    #[rstest]
    fn test_fixed_bulk_compiler_y_missing_basis(
        #[values(Connectivity::ISOLATED, Connectivity::ISOLATED.with_pipe(Direction::ZMINUS))]
        connectivity: Connectivity,
    ) {
        let sig = BlockSignature {
            kind: BlockKind::Y,
            rounds: None,
            connectivity,
            boundary_basis: None,
            layer_schedule: None,
            surgery_side: None,
        };
        assert!(matches!(
            validate_fixed_bulk(sig),
            Err(CompileError::MissingBoundaryBasis { kind: BlockKind::Y })
        ));
    }

    #[rstest]
    fn test_fixed_bulk_compiler_y_normalizes_temporal_hadamard(
        #[values(Direction::ZMINUS, Direction::ZPLUS)] dir: Direction,
    ) {
        let sig = BlockSignature {
            kind: BlockKind::Y,
            rounds: None,
            connectivity: Connectivity::ISOLATED.with_hadamard(dir),
            boundary_basis: Some(Basis::X),
            layer_schedule: None,
            surgery_side: None,
        };

        validate_fixed_bulk(sig).expect("temporal H is owned by the realignment pipe node");
        let template = compile_fixed_bulk(sig, 3).unwrap();
        let key = template.observable_gateway.keys().next().unwrap();
        assert!(key.has_arm(dir));
        assert!(!key.connectivity().has_hadamard(dir));
    }

    #[rstest]
    fn test_fixed_bulk_compiler_y_rejects_invalid_connectivity(
        #[values(
            Connectivity::ISOLATED,
            Connectivity::ISOLATED
                .with_pipe(Direction::ZMINUS)
                .with_pipe(Direction::ZPLUS),
            Connectivity::ISOLATED.with_pipe(Direction::XPLUS)
        )]
        connectivity: Connectivity,
    ) {
        let sig = BlockSignature {
            kind: BlockKind::Y,
            rounds: None,
            connectivity,
            boundary_basis: Some(Basis::X),
            layer_schedule: None,
            surgery_side: None,
        };
        assert!(matches!(
            validate_fixed_bulk(sig),
            Err(CompileError::InvalidConnectivity {
                kind: BlockKind::Y,
                ..
            })
        ));
    }

    #[rstest]
    fn test_y_observable_meas_qubits_consistency(
        #[values(Basis::X, Basis::Z)] top_basis: Basis,
        #[values(3, 5, 7, 9)] d: u32,
    ) {
        let set = TransitionSets::new(d, top_basis);
        let init_observable_qubits: crate::FxSet<IVec2> =
            collect_observable_meas_qubits_reversed(&set, d as i32, top_basis)
                .iter()
                .copied()
                .collect();
        let (_chunks, gateway) = build_y_chunks(top_basis, d, d / 2, true).unwrap();
        let key = gateway
            .keys()
            .next()
            .expect("Y block has an observable gateway");
        let entry = &gateway[key];
        let (transition_round_init, _) = make_y_transition_round_reversed(d, top_basis).unwrap();
        let measurement_from_gateway: crate::FxSet<_> = entry.measurements[0]
            .measurements
            .iter()
            .map(|&measurement| {
                transition_round_init
                    .circuit
                    .meas_registry()
                    .record(measurement)
                    .expect("flow references a circuit measurement")
                    .qubit
            })
            .collect();
        assert_eq!(init_observable_qubits, measurement_from_gateway);
    }

    #[rstest]
    fn test_standalone_y_chunks_have_open_boundary_flows(
        #[values(Basis::X, Basis::Z)] basis: Basis,
        #[values(3, 5, 7, 9)] d: u32,
    ) {
        // A measurement-side Y block consumes stabilizers from the cube below,
        // so its template keeps flows with an open `start` as boundary residual.
        let (chunks, _) = build_y_chunks(basis, d, d / 2, false).unwrap();
        let template = LoweringTemplate::from_chunks(chunks, ObservableGateway::new()).unwrap();
        assert!(
            template
                .program_template
                .boundary_flows
                .iter()
                .any(|flow| !flow.start.is_empty()),
            "expected open-start boundary flows on the standalone Y template"
        );
    }

    #[rstest]
    fn test_y_observable_gateway(
        #[values(Direction::ZMINUS, Direction::ZPLUS)] pipe_dir: Direction,
    ) {
        let template = compile_and_verify(Basis::X, 3, pipe_dir);
        assert_eq!(template.observable_gateway.len(), 1);
        let key = template.observable_gateway.keys().next().unwrap();
        assert_eq!(key.center_basis(), Pauli::Y);
        assert!(key.has_arm(pipe_dir));
    }
}
