//! Fixed-bulk spatial Hadamard wall: pipe-owned extended stabilizers between
//! two cubes whose transverse face bases are flipped.

use std::sync::Arc;

use bloq_circuit::{Chunk, ChunkOrLoop, CoordCircuit, Flow, GateType, PauliBasis, PauliMap};
use bloq_graph::{Basis, Direction, Pauli, UDirection};
use glam::IVec2;
use smallvec::SmallVec;

use crate::CompileError;
use crate::block::gateway::{ChunkMeasurements, GatewayEntry, LocalStabilizer, ObservableGateway};
use crate::block::measurements::MeasurementIndex;
use crate::block::{CompiledTemplate, LoweringTemplate};
use crate::signature::Connectivity;

use super::cube_bulk_repetitions;
use super::utils::{
    Corner, DataBatches, checkerboard_basis, collapse_matching_data_boundary, reset_gate,
    sorted_batches,
};

// ==============================================================================
// Signature
// ==============================================================================

/// What one endpoint cube contributes to the wall.
///
/// The wall owns the circuit of the region it stands in, so it has to reproduce
/// the collapse decisions its neighbours make for the data column they share
/// with it — the cubes emit the same resets and readouts, and the two copies
/// dedup when the node's instances merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct WallSide {
    /// The cube's temporal basis `z()`: the basis it collapses data in.
    pub(crate) temporal_basis: Basis,
    /// The endpoint's complete pipe connectivity. The wall derives data
    /// preparation, readout, perpendicular arms, and shared corners from this
    /// single source of truth.
    pub(crate) connectivity: Connectivity,
}

/// Everything a positive-axis spatial Hadamard wall is compiled from, and its
/// template cache key. The code distance is supplied separately, exactly as it
/// is for the realignment: it is a property of the whole compilation, not of
/// one pipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpatialHadamardKey {
    pub(crate) axis: UDirection,
    /// Syndrome rounds of the endpoint cubes, which the wall matches round for
    /// round.
    pub(crate) rounds: u32,
    /// The negative-axis cube's perpendicular face basis. Both halves of an
    /// end row agree on whether it survives, because the positive-axis cube's
    /// corresponding basis is its flip.
    pub(crate) boundary_basis: Basis,
    pub(crate) minus: WallSide,
    pub(crate) plus: WallSide,
}

impl SpatialHadamardKey {
    pub(super) fn introduces_joint_checks(self, init_data: &crate::FxMap<IVec2, Basis>) -> bool {
        self.minus.connectivity.has_pipe(Direction::ZMINUS)
            && self.plus.connectivity.has_pipe(Direction::ZMINUS)
            && init_data.is_empty()
    }

    fn resets(self, side: WallSide) -> bool {
        !side.connectivity.has_pipe(Direction::ZMINUS)
    }

    fn measures(self, side: WallSide) -> bool {
        !side.connectivity.has_pipe(Direction::ZPLUS)
    }

    fn arm(self, side: WallSide, positive: bool) -> bool {
        let direction = match (self.axis, positive) {
            (UDirection::X, true) => Direction::YPLUS,
            (UDirection::X, false) => Direction::YMINUS,
            (UDirection::Y, true) => Direction::XPLUS,
            (UDirection::Y, false) => Direction::XMINUS,
            (UDirection::Z, _) => unreachable!("a spatial wall has an X or Y axis"),
        };
        side.connectivity.has_open_edge(direction)
    }
}

// ==============================================================================
// Geometry
// ==============================================================================

/// CX slots per wall round.
const WALL_CX_SLOTS: usize = 6;

/// The negative-axis half's X-wall data corners and the slots they occupy.
/// Slot `k` is moment `k + 1`; the GHZ grows at slots 0/1 and is undone at
/// slots 4/5, so its data interactions sit strictly between.
const X_MINUS_SLOTS: [(Corner, usize); 2] = [(Corner::TL, 1), (Corner::BL, 3)];
/// The positive-axis X-wall half's, one slot later on both counts.
const X_PLUS_SLOTS: [(Corner, usize); 2] = [(Corner::TR, 2), (Corner::BR, 4)];
/// The Y wall is the diagonal transpose of the X wall.
const Y_MINUS_SLOTS: [(Corner, usize); 2] = [(Corner::BR, 1), (Corner::BL, 3)];
const Y_PLUS_SLOTS: [(Corner, usize); 2] = [(Corner::TR, 2), (Corner::TL, 4)];

const GROW_MINUS_SLOT: usize = 0;
const GROW_PLUS_SLOT: usize = 1;
const UNDO_MINUS_SLOT: usize = 4;
const UNDO_PLUS_SLOT: usize = 5;

/// One extended stabilizer: two ancillas straddling the consumed column, the
/// link qubit chaining them into a GHZ state, and up to four data qubits split
/// between the two cubes.
pub(super) struct WallTile {
    /// The negative-axis ancilla. Its checkerboard colour is [`basis`].
    ///
    /// [`basis`]: Self::basis
    pub(super) minus: IVec2,
    /// The positive-axis ancilla, coloured [`basis`]`.flip()`. It is the
    /// chain's X-basis syndrome readout in forward rounds. In reversed
    /// rounds the negative-axis ancilla carries the syndrome instead.
    ///
    /// [`basis`]: Self::basis
    pub(super) plus: IVec2,
    pub(super) basis: Basis,
    /// Active negative-half data, paired with the slot that drives it.
    pub(super) minus_data: SmallVec<[(IVec2, usize); 2]>,
    /// Active positive-half data, in `basis.flip()`.
    pub(super) plus_data: SmallVec<[(IVec2, usize); 2]>,
    pub(super) axis: UDirection,
}

impl WallTile {
    /// Detector coordinate: the middle of the consumed column, on the tile's row.
    fn center(&self) -> IVec2 {
        (self.minus + self.plus) / 2
    }

    fn link_for(&self, reversed: bool) -> IVec2 {
        let side = if reversed { -1 } else { 1 };
        self.minus + wall_coord(self.axis, 1, side)
    }
}

fn wall_coord(axis: UDirection, across: i32, along: i32) -> IVec2 {
    match axis {
        UDirection::X => IVec2::new(across, along),
        UDirection::Y => IVec2::new(along, across),
        UDirection::Z => unreachable!("a spatial wall has an X or Y axis"),
    }
}

fn along_coord(axis: UDirection, qubit: IVec2) -> i32 {
    match axis {
        UDirection::X => qubit.y,
        UDirection::Y => qubit.x,
        UDirection::Z => unreachable!("a spatial wall has an X or Y axis"),
    }
}

/// Coordinates along `side`'s seam-adjacent data column.
///
/// The `d` interior rows are always there; a perpendicular arm opens the cube's
/// perpendicular edge, which activates that boundary cell's outward corners
/// and extends the column one row past the wall's end.
fn has_seam_data(distance: u32, key: SpatialHadamardKey, side: WallSide, along: i32) -> bool {
    let d = distance as i32;
    match along {
        -1 => key.arm(side, false),
        along if along == 2 * d + 1 => key.arm(side, true),
        along => (1..2 * d).contains(&along),
    }
}

/// Whether `along` is one of the two rows a perpendicular arm adds. Those
/// qubits sit on an open spatial edge, so their cube prepares and reads them
/// out unconditionally — the same override `grid.rs` applies.
fn is_arm_row(distance: u32, along: i32) -> bool {
    along == -1 || along == 2 * distance as i32 + 1
}

/// The wall's extended stabilizers, bottom row first.
///
/// One per measure row of the cubes' cell grid, minus the end row whose colour
/// disagrees with the cubes' wall-perpendicular boundary basis: there the cubes
/// keep their own weight-2 boundary plaquettes, which share exactly one data
/// qubit with the would-be end tile and would anticommute with it. The two end
/// rows always disagree with each other (`d` is odd), so exactly one survives.
pub(super) fn wall_tiles(distance: u32, key: SpatialHadamardKey) -> Vec<WallTile> {
    let d = distance as i32;
    let mut tiles = Vec::with_capacity(distance as usize + 1);
    let (minus_slots, plus_slots) = match key.axis {
        UDirection::X => (X_MINUS_SLOTS, X_PLUS_SLOTS),
        UDirection::Y => (Y_MINUS_SLOTS, Y_PLUS_SLOTS),
        UDirection::Z => unreachable!("a spatial wall has an X or Y axis"),
    };
    for along in (0..=2 * d).step_by(2) {
        let minus = wall_coord(key.axis, 2 * d, along);
        let plus = wall_coord(key.axis, 2 * d + 2, along);
        let basis = checkerboard_basis(minus);
        if (along == 0 || along == 2 * d) && basis != key.boundary_basis {
            continue;
        }
        let half = |anchor: IVec2, side: WallSide, slots: [(Corner, usize); 2]| {
            slots
                .into_iter()
                .filter_map(|(corner, slot)| {
                    let qubit = anchor + corner.to_ivec2();
                    has_seam_data(distance, key, side, along_coord(key.axis, qubit))
                        .then_some((qubit, slot))
                })
                .collect::<SmallVec<[(IVec2, usize); 2]>>()
        };
        tiles.push(WallTile {
            minus,
            plus,
            basis,
            minus_data: half(minus, key.minus, minus_slots),
            plus_data: half(plus, key.plus, plus_slots),
            axis: key.axis,
        });
    }
    tiles
}

/// Which temporal face a collapse map describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Face {
    Init,
    Meas,
}

/// The data qubits the wall collapses at `face`, in their owning cube's temporal
/// basis. Mirrors the cube rule: collapse unless a pipe carries the state across
/// that face, and always on an open spatial edge.
pub(super) fn collapse_map(
    distance: u32,
    key: SpatialHadamardKey,
    tiles: &[WallTile],
    face: Face,
) -> crate::FxMap<IVec2, Basis> {
    let mut map = crate::FxMap::default();
    let mut collect = |data: &SmallVec<[(IVec2, usize); 2]>, side: WallSide| {
        let collapses = match face {
            Face::Init => key.resets(side),
            Face::Meas => key.measures(side),
        };
        for &(qubit, _) in data {
            if collapses || is_arm_row(distance, along_coord(key.axis, qubit)) {
                map.insert(qubit, side.temporal_basis);
            }
        }
    };
    for tile in tiles {
        collect(&tile.minus_data, key.minus);
        collect(&tile.plus_data, key.plus);
    }
    map
}

// ==============================================================================
// Circuit
// ==============================================================================

fn push_interaction(
    cx: &mut Vec<IVec2>,
    cz: &mut Vec<IVec2>,
    ancilla: IVec2,
    data: IVec2,
    basis: Basis,
) {
    match basis {
        Basis::X => cx.extend([ancilla, data]),
        Basis::Z => cz.extend([ancilla, data]),
    }
}

fn data_slot(slot: usize, reversed: bool) -> usize {
    if reversed {
        WALL_CX_SLOTS - 1 - slot
    } else {
        slot
    }
}

/// Emit one wall round.
fn emit_wall_round(
    tiles: &[WallTile],
    init_data: &DataBatches,
    meas_data: &DataBatches,
    reversed: bool,
) -> Result<(CoordCircuit, MeasurementIndex), CompileError> {
    let mut circuit = CoordCircuit::new();
    circuit.do_gate(
        GateType::RX,
        tiles
            .iter()
            .map(|tile| if reversed { tile.plus } else { tile.minus }),
    )?;
    circuit.do_gate(
        GateType::RZ,
        tiles.iter().map(|tile| tile.link_for(reversed)),
    )?;
    circuit.do_gate(
        GateType::RZ,
        tiles
            .iter()
            .map(|tile| if reversed { tile.minus } else { tile.plus }),
    )?;
    for (basis, qubits) in init_data {
        circuit.do_gate(reset_gate(*basis), qubits.iter().copied())?;
    }
    circuit.tick();

    let mut measurements = MeasurementIndex::default();
    for slot in 0..WALL_CX_SLOTS {
        let mut cx = Vec::new();
        let mut cz = Vec::new();
        for tile in tiles {
            let link = tile.link_for(reversed);
            let (start, end) = if reversed {
                (tile.plus, tile.minus)
            } else {
                (tile.minus, tile.plus)
            };
            let bracket = match slot {
                GROW_MINUS_SLOT => Some([start, link]),
                GROW_PLUS_SLOT => Some([link, end]),
                UNDO_MINUS_SLOT => Some([link, start]),
                UNDO_PLUS_SLOT => Some([end, link]),
                _ => None,
            };
            cx.extend(bracket.into_iter().flatten());
            for (anchor, basis, data) in [
                (tile.minus, tile.basis, &tile.minus_data),
                (tile.plus, tile.basis.flip(), &tile.plus_data),
            ] {
                for &(qubit, at) in data {
                    if data_slot(at, reversed) == slot {
                        push_interaction(&mut cx, &mut cz, anchor, qubit, basis);
                    }
                }
            }
        }
        circuit.do_gate(GateType::CX, cx)?;
        circuit.do_gate(GateType::CZ, cz)?;
        circuit.tick();
    }

    let measured = tiles
        .iter()
        .map(|tile| if reversed { tile.minus } else { tile.plus })
        .collect::<Vec<_>>();
    for (&qubit, id) in measured
        .iter()
        .zip(circuit.measure(PauliBasis::X, measured.iter().copied()))
    {
        measurements.record(qubit, id);
    }
    let flags: Vec<_> = tiles
        .iter()
        .flat_map(|tile| {
            [
                if reversed { tile.plus } else { tile.minus },
                tile.link_for(reversed),
            ]
        })
        .collect();
    let records = circuit.measure(PauliBasis::Z, flags.iter().copied());
    for (q, record) in flags.into_iter().zip(records) {
        measurements.record(q, record);
    }
    for (basis, qubits) in meas_data {
        for (&qubit, id) in qubits
            .iter()
            .zip(circuit.measure(PauliBasis::from(*basis), qubits.iter().copied()))
        {
            measurements.record(qubit, id);
        }
    }
    Ok((circuit, measurements))
}

// ==============================================================================
// Flows
// ==============================================================================

/// The collapse rule applied to a mixed-basis stabilizer: once per half, against
/// that half's own Pauli, then recombined. A half that meets an opposite-basis
/// collapse kills the whole flow, exactly as a uniform tile's would.
fn collapse_extended_stabilizer(
    tile: &WallTile,
    collapsed_data: Option<&crate::FxMap<IVec2, Basis>>,
) -> Option<(PauliMap, Vec<IVec2>)> {
    let pauli = Pauli::from(tile.basis);
    let halves = [
        (&tile.minus_data, tile.basis, pauli),
        (&tile.plus_data, tile.basis.flip(), pauli.flip()),
    ];
    let mut support = PauliMap::empty();
    let mut collapsed = Vec::new();
    for (data, basis, pauli) in halves {
        let qubits: SmallVec<[IVec2; 2]> = data.iter().map(|&(qubit, _)| qubit).collect();
        let (half_support, half_collapsed) =
            collapse_matching_data_boundary(&qubits, basis, pauli, collapsed_data)?;
        for (qubit, pauli) in half_support.iter() {
            support.insert(*qubit, *pauli);
        }
        collapsed.extend(half_collapsed);
    }
    Some((support, collapsed))
}

/// Creator/consumer pair per extended stabilizer, collapsed against the round's
/// init/meas maps — the same shape [`standard_round_flows`] gives an ordinary
/// tile, with the chain's single `MX` standing in for the ancilla record.
///
/// [`standard_round_flows`]: super::utils
fn wall_round_flows(
    tiles: &[WallTile],
    measurements: &MeasurementIndex,
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: Option<&crate::FxMap<IVec2, Basis>>,
    reversed: bool,
) -> Vec<Flow> {
    let mut flows = Vec::with_capacity(tiles.len() * 4);
    for tile in tiles {
        let ancilla =
            measurements.expect_measurement(if reversed { tile.minus } else { tile.plus });

        if let Some((start, _)) = collapse_extended_stabilizer(tile, init_data) {
            flows.push(
                Flow::new(start, PauliMap::empty())
                    .with_measurements([ancilla])
                    .with_center(tile.center()),
            );
        }

        if let Some((end, mut collapsed)) = collapse_extended_stabilizer(tile, meas_data) {
            collapsed.sort_unstable_by_key(|qubit| (qubit.y, qubit.x));
            flows.push(
                Flow::new(PauliMap::empty(), end)
                    .with_measurements(
                        collapsed
                            .into_iter()
                            .map(|qubit| measurements.expect_measurement(qubit))
                            .chain(std::iter::once(ancilla)),
                    )
                    .with_center(tile.center()),
            );
        }
    }
    flows
}

// ==============================================================================
// Observables
// ==============================================================================

/// Pipe-frame crossing key: the plus-side Pauli is the minus-side Pauli flipped.
pub(super) fn crossing_key(axis: UDirection, minus_pauli: Pauli) -> LocalStabilizer {
    let (minus, plus) = match axis {
        UDirection::X => (Direction::XMINUS, Direction::XPLUS),
        UDirection::Y => (Direction::YMINUS, Direction::YPLUS),
        UDirection::Z => unreachable!("a spatial wall has an X or Y axis"),
    };
    LocalStabilizer::isolated(Pauli::I)
        .with_arm(minus, minus_pauli, false)
        .with_arm(plus, minus_pauli.flip(), false)
}

/// Crossing gateway: free init-check records restore support killed by resets;
/// final readout and temporal-face operators remain cube-owned.
fn build_gateway(
    key: SpatialHadamardKey,
    tiles: &[WallTile],
    init_data: &crate::FxMap<IVec2, Basis>,
    init_chunk: &Chunk,
) -> ObservableGateway {
    let introduces_joint_checks = key.introduces_joint_checks(init_data);
    let measurements = MeasurementIndex::from_circuit(&init_chunk.circuit);
    let mut gateway = ObservableGateway::new();
    for basis in [Basis::X, Basis::Z] {
        let records: Vec<u32> = tiles
            .iter()
            .filter(|tile| {
                tile.basis == basis
                    && ((introduces_joint_checks && basis == key.boundary_basis)
                        || collapse_extended_stabilizer(tile, Some(init_data)).is_none())
            })
            .map(|tile| measurements.expect_measurement(tile.plus))
            .collect();
        let mut entry = GatewayEntry::default();
        if !records.is_empty() {
            entry.measurements.push(ChunkMeasurements {
                chunk_index: 0,
                measurements: records,
            });
        }
        gateway.insert(crossing_key(key.axis, Pauli::from(basis)), entry);
    }
    gateway
}

fn wall_chunk(
    tiles: &[WallTile],
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: Option<&crate::FxMap<IVec2, Basis>>,
    reversed: bool,
) -> Result<Chunk, CompileError> {
    let (circuit, measurements) = emit_wall_round(
        tiles,
        &sorted_batches(init_data, [Basis::X, Basis::Z]),
        &sorted_batches(meas_data, [Basis::X, Basis::Z]),
        reversed,
    )?;
    let flows = wall_round_flows(tiles, &measurements, init_data, meas_data, reversed);
    Ok(Chunk { circuit, flows })
}

/// Compile a positive-axis spatial Hadamard wall in the negative endpoint's
/// local coordinates.
///
/// The stage structure is the cube's, round for round, so template composition,
/// loop flattening and the instance merge need nothing new — the wall's chunks
/// line up moment for moment with every cube in its z-layer component.
pub(crate) fn compile_spatial_hadamard(
    distance: u32,
    key: SpatialHadamardKey,
) -> Result<CompiledTemplate, CompileError> {
    let tiles = wall_tiles(distance, key);
    let init_data = collapse_map(distance, key, &tiles, Face::Init);
    let meas_data = collapse_map(distance, key, &tiles, Face::Meas);

    let make_chunk =
        |init_data, meas_data, reversed| wall_chunk(&tiles, init_data, meas_data, reversed);
    let init_chunk = make_chunk(Some(&init_data), None, false)?;
    let gateway = build_gateway(key, &tiles, &init_data, &init_chunk);
    let repetitions = cube_bulk_repetitions(key.rounds);
    let forward_bulk = make_chunk(None, None, false)?;
    let reverse_bulk = make_chunk(None, None, true)?;
    let meas = make_chunk(None, Some(&meas_data), repetitions.is_multiple_of(2))?;

    let mut chunks = vec![ChunkOrLoop::Single(Box::new(init_chunk))];
    chunks.push(ChunkOrLoop::Loop {
        body: vec![reverse_bulk.clone(), forward_bulk],
        repetitions: repetitions / 2,
    });
    if repetitions % 2 == 1 {
        chunks.push(ChunkOrLoop::Single(Box::new(reverse_bulk)));
    }
    chunks.push(ChunkOrLoop::Single(Box::new(meas)));

    let mut compiled = LoweringTemplate::from_chunks(chunks, gateway)?;
    let program = Arc::make_mut(&mut compiled.program_template);
    for i in 0..program.circuit.body_count() {
        let body = program
            .circuit
            .body(bloq_circuit::BodyId(i as u32))
            .expect("allocated body");
        for moment in body.ops().split(|op| matches!(op, bloq_circuit::Op::Tick)) {
            let mut records = crate::FxMap::default();
            for op in moment {
                if let bloq_circuit::Op::Measure {
                    basis,
                    qubits,
                    measurements,
                    ..
                } = op
                {
                    records.extend(
                        qubits
                            .iter()
                            .zip(measurements)
                            .map(|(&q, &id)| (q, (*basis, id))),
                    );
                }
            }
            if records.is_empty() {
                continue;
            }
            for tile in &tiles {
                let reversed = records
                    .get(&tile.minus)
                    .is_some_and(|(basis, _)| *basis == PauliBasis::X);
                for (ancilla, basis, data, flag) in [
                    (
                        tile.minus,
                        tile.basis,
                        &tile.minus_data,
                        if reversed {
                            tile.link_for(true)
                        } else {
                            tile.minus
                        },
                    ),
                    (
                        tile.plus,
                        tile.basis.flip(),
                        &tile.plus_data,
                        if reversed {
                            tile.plus
                        } else {
                            tile.link_for(false)
                        },
                    ),
                ] {
                    if data.len() == 2 {
                        let &(target, _) = data
                            .iter()
                            .max_by_key(|(_, slot)| data_slot(*slot, reversed))
                            .expect("two data gates");
                        program
                            .readout_frames
                            .push(bloq_ir::lowering::ReadoutFrame {
                                flag: records[&flag].1,
                                ancilla,
                                target,
                                pauli: PauliBasis::from(basis),
                            });
                    }
                }
            }
        }
    }
    Ok(Arc::new(compiled))
}

#[cfg(test)]
mod tests {
    use bloq_circuit::Op;
    use bloq_graph::CubeKind;
    use bloq_stim::StimFlowVerifier;
    use rstest::rstest;

    use super::*;

    use crate::block::fixed_bulk::regular::build_regular_patch;
    use crate::block::fixed_bulk::spatial::build_spatial_patch;
    use crate::block::fixed_bulk::split_ops_into_moments;
    use crate::block::fixed_bulk::utils::make_surface_code_chunk;
    use crate::block::patch::{Patch, Tile};
    use crate::signature::LayerSchedule;

    /// A side with both temporal pipes and no arms: nothing collapses, which is
    /// the bulk-round baseline.
    fn threaded(temporal_basis: Basis) -> WallSide {
        WallSide {
            temporal_basis,
            connectivity: Connectivity::ISOLATED
                .with_pipe(Direction::ZMINUS)
                .with_pipe(Direction::ZPLUS),
        }
    }

    /// A side with neither temporal pipe: its data are prepared and read out by
    /// the wall's first and last rounds.
    fn terminated(temporal_basis: Basis) -> WallSide {
        WallSide {
            temporal_basis,
            connectivity: Connectivity::ISOLATED,
        }
    }

    fn key(boundary_basis: Basis, minus: WallSide, plus: WallSide) -> SpatialHadamardKey {
        key_axis(UDirection::X, boundary_basis, minus, plus)
    }

    fn key_axis(
        axis: UDirection,
        boundary_basis: Basis,
        minus: WallSide,
        plus: WallSide,
    ) -> SpatialHadamardKey {
        SpatialHadamardKey {
            axis,
            // The round count only sets the bulk loop's repetition count. Every
            // test below inspects tiles, flows or the observable gateway, none
            // of which vary with it, so any legal value does.
            rounds: 3,
            boundary_basis,
            minus,
            plus,
        }
    }

    /// Every side combination the wall is allowed to see, given the validator's
    /// guarantee that the two cubes' transverse bases are flipped.
    fn side_pairs(minus_temporal: Basis) -> Vec<(WallSide, WallSide)> {
        let plus_temporal = minus_temporal.flip();
        vec![
            (threaded(minus_temporal), threaded(plus_temporal)),
            (terminated(minus_temporal), terminated(plus_temporal)),
            (threaded(minus_temporal), terminated(plus_temporal)),
            (terminated(minus_temporal), threaded(plus_temporal)),
        ]
    }

    fn chunks_of(distance: u32, key: SpatialHadamardKey, reversed: bool) -> Vec<Chunk> {
        let tiles = wall_tiles(distance, key);
        let init_data = collapse_map(distance, key, &tiles, Face::Init);
        let meas_data = collapse_map(distance, key, &tiles, Face::Meas);
        vec![
            wall_chunk(&tiles, Some(&init_data), None, reversed).expect("init round"),
            wall_chunk(&tiles, None, None, reversed).expect("bulk round"),
            wall_chunk(&tiles, None, Some(&meas_data), reversed).expect("meas round"),
        ]
    }

    /// Stim is the arbiter for the construction: every wall round must carry the
    /// creator/consumer pair it declares for each extended stabilizer, at every
    /// distance, boundary basis, temporal basis and collapse combination.
    #[rstest]
    fn wall_rounds_have_their_flows(
        #[values(3, 5, 7)] distance: u32,
        #[values(UDirection::X, UDirection::Y)] axis: UDirection,
        #[values(false, true)] reversed: bool,
        #[values(Basis::X, Basis::Z)] boundary_basis: Basis,
        #[values(Basis::X, Basis::Z)] minus_temporal: Basis,
    ) {
        for (minus, plus) in side_pairs(minus_temporal) {
            let wall = key_axis(axis, boundary_basis, minus, plus);
            for chunk in chunks_of(distance, wall, reversed) {
                chunk.verify_flows(None, None).unwrap_or_else(|error| {
                    panic!("{axis} d={distance} reversed={reversed}: {error}")
                });
            }
        }
    }

    #[rstest]
    fn bulk_round_reads_only_stabilizer_ancillas(
        #[values(3, 5)] distance: u32,
        #[values(false, true)] reversed: bool,
    ) {
        let wall = key(Basis::Z, threaded(Basis::X), threaded(Basis::Z));
        let tiles = wall_tiles(distance, wall);
        let bulk = wall_chunk(&tiles, None, None, reversed).expect("bulk round");
        let moments = split_ops_into_moments(
            bulk.circuit
                .body(bulk.circuit.entry_body())
                .expect("entry body")
                .ops(),
        );

        assert_eq!(bulk.circuit.num_measurements(), 3 * tiles.len() as u32);
        assert_eq!(
            moments[0]
                .iter()
                .map(|op| match op {
                    Op::Gate { gate, qubits } if gate.is_reset() => qubits.len(),
                    _ => 0,
                })
                .sum::<usize>(),
            3 * tiles.len()
        );
        assert!(
            moments[1]
                .iter()
                .all(|op| !matches!(op, Op::Gate { gate, .. } if gate.is_reset()))
        );
    }

    /// The same, with the perpendicular arms that decide each wall end's shape.
    #[rstest]
    fn wall_end_shapes_have_their_flows(
        #[values(3, 5)] distance: u32,
        #[values(Basis::X, Basis::Z)] boundary_basis: Basis,
        #[values(false, true)] minus_plus: bool,
        #[values(false, true)] minus_minus: bool,
        #[values(false, true)] plus_plus: bool,
        #[values(false, true)] plus_minus: bool,
    ) {
        let minus = WallSide {
            connectivity: [
                (minus_plus, Direction::YPLUS),
                (minus_minus, Direction::YMINUS),
            ]
            .into_iter()
            .filter(|(present, _)| *present)
            .fold(Connectivity::ISOLATED, |connectivity, (_, direction)| {
                connectivity.with_pipe(direction)
            }),
            ..terminated(Basis::X)
        };
        let plus = WallSide {
            connectivity: [
                (plus_plus, Direction::YPLUS),
                (plus_minus, Direction::YMINUS),
            ]
            .into_iter()
            .filter(|(present, _)| *present)
            .fold(Connectivity::ISOLATED, |connectivity, (_, direction)| {
                connectivity.with_pipe(direction)
            }),
            ..terminated(Basis::Z)
        };
        for chunk in chunks_of(distance, key(boundary_basis, minus, plus), false) {
            chunk.verify_flows(None, None).expect("arm-config flows");
        }
    }

    /// Exactly one end row survives, because the two ends always carry opposite
    /// checkerboard colours at odd distance. The interior rows are all there.
    #[rstest]
    fn one_wall_end_survives(#[values(3, 5, 7)] distance: u32) {
        for boundary_basis in [Basis::X, Basis::Z] {
            let tiles = wall_tiles(
                distance,
                key(boundary_basis, terminated(Basis::X), terminated(Basis::Z)),
            );
            assert_eq!(tiles.len() as u32, distance);
            let rows: Vec<i32> = tiles.iter().map(|tile| tile.minus.y).collect();
            let ends = rows
                .iter()
                .filter(|&&y| y == 0 || y == 2 * distance as i32)
                .count();
            assert_eq!(ends, 1, "d={distance} {boundary_basis:?}: rows {rows:?}");
        }
    }

    // ==========================================================================
    // Merged-round verification
    // ==========================================================================

    fn translated_patch(patch: &Patch, offset: IVec2) -> Patch {
        Patch::new(
            patch
                .tiles()
                .iter()
                .map(|tile| {
                    Tile::new(
                        tile.basis(),
                        tile.measure_qubit() + offset,
                        tile.data_slots()
                            .iter()
                            .map(|slot| slot.map(|qubit| qubit + offset)),
                    )
                })
                .collect(),
        )
    }

    fn translated_map(
        map: &crate::FxMap<IVec2, Basis>,
        offset: IVec2,
    ) -> crate::FxMap<IVec2, Basis> {
        map.iter()
            .map(|(&qubit, &basis)| (qubit + offset, basis))
            .collect()
    }

    /// Splice chunks as instance lowering does, deduplicating shared resets and
    /// measurements so merged wall/cube flows test the real moment schedule.
    fn splice(chunks: &[Chunk]) -> Chunk {
        let moments: Vec<Vec<Vec<Op>>> = chunks
            .iter()
            .map(|chunk| {
                split_ops_into_moments(
                    chunk
                        .circuit
                        .body(chunk.circuit.entry_body())
                        .expect("entry body")
                        .ops(),
                )
            })
            .collect();
        let depth = moments[0].len();
        assert!(
            moments.iter().all(|chunk| chunk.len() == depth),
            "spliced chunks must share a moment structure"
        );

        let mut circuit = CoordCircuit::new();
        let mut remaps: Vec<crate::FxMap<u32, u32>> = vec![Default::default(); chunks.len()];
        for moment in 0..depth {
            let mut measured: crate::FxMap<IVec2, u32> = Default::default();
            let mut reset: crate::FxSet<IVec2> = Default::default();
            for (index, chunk_moments) in moments.iter().enumerate() {
                for op in &chunk_moments[moment] {
                    match op {
                        Op::Gate { gate, qubits } if gate.is_reset() => {
                            let fresh: Vec<IVec2> = qubits
                                .iter()
                                .copied()
                                .filter(|qubit| reset.insert(*qubit))
                                .collect();
                            circuit.do_gate(*gate, fresh).expect("reset");
                        }
                        Op::Gate { gate, qubits } => {
                            circuit
                                .do_gate(*gate, qubits.iter().copied())
                                .expect("gate");
                        }
                        Op::Measure {
                            basis,
                            qubits,
                            measurements,
                            ..
                        } => {
                            let fresh: Vec<IVec2> = qubits
                                .iter()
                                .copied()
                                .filter(|qubit| !measured.contains_key(qubit))
                                .collect();
                            let ids = circuit.measure(*basis, fresh.iter().copied());
                            for (&qubit, id) in fresh.iter().zip(ids) {
                                measured.insert(qubit, id);
                            }
                            for (qubit, &local) in qubits.iter().zip(measurements) {
                                remaps[index].insert(local, measured[qubit]);
                            }
                        }
                        _ => {}
                    }
                }
            }
            if moment + 1 < depth {
                circuit.tick();
            }
        }

        let flows = chunks
            .iter()
            .zip(&remaps)
            .flat_map(|(chunk, remap)| {
                chunk.flows.iter().map(move |flow| {
                    let mut spliced = flow.clone();
                    spliced.measurements =
                        flow.measurements.iter().map(|local| remap[local]).collect();
                    spliced
                })
            })
            .collect();

        Chunk { circuit, flows }
    }

    /// The CZ gallery's endpoint pair: `XZX` threaded through both temporal
    /// faces, `XXZ` terminated with both perpendicular arms.
    fn cz_merged_rounds(distance: u32) -> Vec<Vec<Chunk>> {
        let d = distance as i32;
        let stride = IVec2::new(2 * d + 2, 0);

        let left_connectivity = Connectivity::ISOLATED
            .with_hadamard(Direction::XPLUS)
            .with_pipe(Direction::ZMINUS)
            .with_pipe(Direction::ZPLUS);
        let right_connectivity = Connectivity::ISOLATED
            .with_hadamard(Direction::XMINUS)
            .with_pipe(Direction::YPLUS)
            .with_pipe(Direction::YMINUS);

        let (left_patch, left_init, left_meas) = build_regular_patch(
            CubeKind::XZX,
            distance,
            left_connectivity,
            LayerSchedule::Extended,
        );
        let (right_patch, right_init, right_meas) = build_spatial_patch(
            CubeKind::XXZ,
            distance,
            right_connectivity,
            LayerSchedule::Extended,
        );
        let right_patch = translated_patch(&right_patch, stride);
        let right_init = translated_map(&right_init, stride);
        let right_meas = translated_map(&right_meas, stride);

        let wall = key(
            CubeKind::XZX.y(),
            threaded(CubeKind::XZX.z()),
            WallSide {
                connectivity: Connectivity::ISOLATED
                    .with_pipe(Direction::YPLUS)
                    .with_pipe(Direction::YMINUS),
                ..terminated(CubeKind::XXZ.z())
            },
        );
        let tiles = wall_tiles(distance, wall);
        let wall_init = collapse_map(distance, wall, &tiles, Face::Init);
        let wall_meas = collapse_map(distance, wall, &tiles, Face::Meas);

        let round = |left_data: Option<&crate::FxMap<IVec2, Basis>>,
                     right_data: Option<&crate::FxMap<IVec2, Basis>>,
                     wall_data: Option<&crate::FxMap<IVec2, Basis>>,
                     face: Face| {
            let (left_init, left_meas) = match face {
                Face::Init => (left_data, None),
                Face::Meas => (None, left_data),
            };
            let (right_init, right_meas) = match face {
                Face::Init => (right_data, None),
                Face::Meas => (None, right_data),
            };
            let (wall_init, wall_meas) = match face {
                Face::Init => (wall_data, None),
                Face::Meas => (None, wall_data),
            };
            vec![
                make_surface_code_chunk(&left_patch, left_init, left_meas).expect("left round"),
                make_surface_code_chunk(&right_patch, right_init, right_meas).expect("right round"),
                wall_chunk(&tiles, wall_init, wall_meas, false).expect("wall round"),
            ]
        };

        vec![
            round(
                Some(&left_init),
                Some(&right_init),
                Some(&wall_init),
                Face::Init,
            ),
            round(None, None, None, Face::Init),
            round(
                Some(&left_meas),
                Some(&right_meas),
                Some(&wall_meas),
                Face::Meas,
            ),
        ]
    }

    /// The whole merged round — both cubes plus the wall, moment for moment —
    /// carries every flow all three declare. A phase collision or anticommuting
    /// end tile fails here.
    #[rstest]
    fn merged_cz_rounds_have_their_flows(#[values(3, 5)] distance: u32) {
        for (stage, chunks) in cz_merged_rounds(distance).into_iter().enumerate() {
            splice(&chunks)
                .verify_flows(None, None)
                .unwrap_or_else(|error| panic!("d={distance} stage {stage}: {error}"));
        }
    }

    // ==========================================================================
    // Completeness
    // ==========================================================================

    /// Number of logical degrees of freedom a stabilizer set leaves: data qubits
    /// minus the GF(2) rank of the generators, over the `(X | Z)` bit layout.
    fn logical_freedom(stabilizers: &[PauliMap]) -> usize {
        let qubits: Vec<IVec2> = {
            let mut qubits: Vec<IVec2> = stabilizers
                .iter()
                .flat_map(|stabilizer| stabilizer.iter().map(|(&qubit, _)| qubit))
                .collect::<crate::FxSet<_>>()
                .into_iter()
                .collect();
            qubits.sort_unstable_by_key(|qubit| (qubit.y, qubit.x));
            qubits
        };
        let column = |qubit: IVec2| {
            qubits
                .iter()
                .position(|candidate| *candidate == qubit)
                .expect("stabilizer support is in the qubit list")
        };
        let width = 2 * qubits.len();
        let mut rows: Vec<Vec<bool>> = stabilizers
            .iter()
            .map(|stabilizer| {
                let mut row = vec![false; width];
                for (&qubit, &pauli) in stabilizer.iter() {
                    let offset = if pauli == bloq_circuit::Pauli::X {
                        0
                    } else {
                        1
                    };
                    row[2 * column(qubit) + offset] = true;
                }
                row
            })
            .collect();

        let mut rank = 0;
        for bit in 0..width {
            let Some(pivot) = (rank..rows.len()).find(|&row| rows[row][bit]) else {
                continue;
            };
            rows.swap(rank, pivot);
            let pivot_row = rows[rank].clone();
            for (index, row) in rows.iter_mut().enumerate() {
                if index == rank || !row[bit] {
                    continue;
                }
                for (target, source) in row.iter_mut().zip(&pivot_row).skip(bit) {
                    *target ^= *source;
                }
            }
            rank += 1;
        }
        qubits.len() - rank
    }

    fn patch_stabilizers(patch: &Patch) -> Vec<PauliMap> {
        patch.tiles().iter().map(Tile::pauli_map).collect()
    }

    fn cube_stabilizers(
        kind: CubeKind,
        distance: u32,
        connectivity: Connectivity,
        offset: IVec2,
    ) -> Vec<PauliMap> {
        let (patch, _, _) = if kind.is_spatial() {
            build_spatial_patch(kind, distance, connectivity, LayerSchedule::Extended)
        } else {
            build_regular_patch(kind, distance, connectivity, LayerSchedule::Extended)
        };
        patch_stabilizers(&translated_patch(&patch, offset))
    }

    /// Rank completeness: the wall and a plain merge of the same geometry keep
    /// the same logical count.
    #[rstest]
    fn wall_preserves_the_merged_logical_count(
        #[values(3, 5)] distance: u32,
        #[values(0, 1, 2, 3)] arm_mask: u8,
    ) {
        let d = distance as i32;
        let stride = IVec2::new(2 * d + 2, 0);
        let arm_pipes = |connectivity: Connectivity| {
            [Direction::YPLUS, Direction::YMINUS]
                .into_iter()
                .enumerate()
                .filter(|(index, _)| arm_mask & (1 << index) != 0)
                .fold(connectivity, |connectivity, (_, direction)| {
                    connectivity.with_pipe(direction)
                })
        };

        let hadamard = {
            let left = cube_stabilizers(
                CubeKind::XZX,
                distance,
                Connectivity::ISOLATED.with_hadamard(Direction::XPLUS),
                IVec2::ZERO,
            );
            let right = cube_stabilizers(
                CubeKind::XXZ,
                distance,
                arm_pipes(Connectivity::ISOLATED.with_hadamard(Direction::XMINUS)),
                stride,
            );
            let wall = key(
                CubeKind::XZX.y(),
                terminated(CubeKind::XZX.z()),
                WallSide {
                    connectivity: arm_pipes(Connectivity::ISOLATED),
                    ..terminated(CubeKind::XXZ.z())
                },
            );
            let tiles = wall_tiles(distance, wall);
            assert_eq!(tiles.len() as u32, distance);
            left.into_iter()
                .chain(right)
                .chain(tiles.iter().map(|tile| {
                    collapse_extended_stabilizer(tile, None)
                        .expect("an uncollapsed stabilizer always survives")
                        .0
                }))
                .collect::<Vec<_>>()
        };

        let plain = {
            let left = cube_stabilizers(
                CubeKind::XZX,
                distance,
                Connectivity::ISOLATED.with_pipe(Direction::XPLUS),
                IVec2::ZERO,
            );
            let right = cube_stabilizers(
                CubeKind::ZZX,
                distance,
                arm_pipes(Connectivity::ISOLATED.with_pipe(Direction::XMINUS)),
                stride,
            );
            left.into_iter().chain(right).collect::<Vec<_>>()
        };

        assert_eq!(
            logical_freedom(&hadamard),
            logical_freedom(&plain),
            "d={distance} arm_mask={arm_mask}"
        );
    }

    // ==========================================================================
    // Observables
    // ==========================================================================

    #[test]
    fn threaded_wall_gateway_records_new_joint_checks() {
        let wall = key(Basis::Z, threaded(Basis::X), threaded(Basis::Z));
        let template = compile_spatial_hadamard(3, wall).expect("wall compiles");
        let entry = template
            .observable_gateway
            .lookup(crossing_key(UDirection::X, Pauli::Z))
            .expect("the joint-check crossing has an entry");

        assert_eq!(entry.measurements.len(), 1);
        assert!(!entry.measurements[0].measurements.is_empty());
    }

    /// The crossing gateway carries init records exactly for the tiles the
    /// crossing runs *along* — colour `P` — and only when the seam columns'
    /// resets would otherwise destroy that support, which is `P != z_L`.
    ///
    /// The CZ gallery's configuration: the first cube is threaded through both
    /// temporal faces (`z_L = X`) and the second one terminates with both
    /// perpendicular arms (`z_R = Z`). The `Z` crossing is perpendicular to the second cube's
    /// resets, so it takes the `Z`-coloured tiles' records; the `X` crossing
    /// runs parallel to both collapses and takes none.
    #[rstest]
    fn crossing_records_are_the_tiles_the_string_runs_along(#[values(3, 5)] distance: u32) {
        let wall = key(
            CubeKind::XZX.y(),
            threaded(CubeKind::XZX.z()),
            WallSide {
                connectivity: Connectivity::ISOLATED
                    .with_pipe(Direction::YPLUS)
                    .with_pipe(Direction::YMINUS),
                ..terminated(CubeKind::XXZ.z())
            },
        );
        let template = compile_spatial_hadamard(distance, wall).expect("wall compiles");
        let gateway = &template.observable_gateway;

        let tiles = wall_tiles(distance, wall);
        let init_data = collapse_map(distance, wall, &tiles, Face::Init);
        let init_chunk = wall_chunk(&tiles, Some(&init_data), None, false).expect("init round");
        let measurements = MeasurementIndex::from_circuit(&init_chunk.circuit);

        // `z_L = X`, so the `X` crossing is parallel to both halves' collapse
        // bases and needs nothing.
        let parallel = gateway
            .lookup(crossing_key(UDirection::X, Pauli::from(CubeKind::XZX.z())))
            .expect("the parallel crossing has an entry");
        assert!(
            parallel.measurements.is_empty(),
            "d={distance}: a crossing in the negative-axis cube's temporal basis is read \
             off the seam data, so the wall owes it no records"
        );

        // The perpendicular one takes every tile whose colour it shares — which
        // is the majority set, tiling the seam columns end to end.
        let perpendicular = gateway
            .lookup(crossing_key(
                UDirection::X,
                Pauli::from(CubeKind::XZX.z().flip()),
            ))
            .expect("the perpendicular crossing has an entry");
        // The gateway is canonicalized into template-id space, and the init
        // chunk is first, so its ids are still its chunk-local ones.
        let expected: Vec<u32> = tiles
            .iter()
            .filter(|tile| tile.basis == CubeKind::XZX.z().flip())
            .map(|tile| measurements.expect_measurement(tile.plus))
            .collect();
        assert_eq!(expected.len(), (distance as usize).div_ceil(2));
        assert_eq!(perpendicular.measurements.len(), 1);
        assert_eq!(perpendicular.measurements[0].chunk_index, 0);
        assert_eq!(perpendicular.measurements[0].measurements, expected);

        // The wall spans no temporal face, so it never carries an operator.
        for entry in [&parallel, &perpendicular] {
            assert!(entry.operator_in.is_empty() && entry.operator_out.is_empty());
        }
    }
}
