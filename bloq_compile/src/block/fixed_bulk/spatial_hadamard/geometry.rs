//! Seam geometry and logical crossings of a spatial Hadamard pipe.

use super::super::utils::{Corner, checkerboard_basis};
use crate::block::gateway::LocalStabilizer;
use crate::signature::Connectivity;
use bloq_graph::{Basis, Direction, Pauli, UDirection};
use glam::IVec2;
use smallvec::SmallVec;

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

/// Endpoint data and orientation of one spatial seam.
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

/// Two branches of a seam check. Perpendicular checks can share one branch.
pub(super) struct WallTile {
    pub(super) minus: IVec2,
    pub(super) plus: IVec2,
    pub(super) basis: Basis,
    pub(super) minus_data: SmallVec<[IVec2; 2]>,
    pub(super) plus_data: SmallVec<[IVec2; 2]>,
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
    let (minus_corners, plus_corners) = match key.axis {
        UDirection::X => ([Corner::TL, Corner::BL], [Corner::TR, Corner::BR]),
        UDirection::Y => ([Corner::BR, Corner::BL], [Corner::TR, Corner::TL]),
        UDirection::Z => unreachable!("a spatial wall has an X or Y axis"),
    };
    for along in (0..=2 * d).step_by(2) {
        let minus = wall_coord(key.axis, 2 * d, along);
        let plus = wall_coord(key.axis, 2 * d + 2, along);
        let basis = checkerboard_basis(minus);
        if (along == 0 || along == 2 * d) && basis != key.boundary_basis {
            continue;
        }
        let half = |anchor: IVec2, side: WallSide, corners: [Corner; 2]| {
            corners
                .into_iter()
                .filter_map(|corner| {
                    let qubit = anchor + corner.to_ivec2();
                    has_seam_data(distance, key, side, along_coord(key.axis, qubit))
                        .then_some(qubit)
                })
                .collect::<SmallVec<[IVec2; 2]>>()
        };
        tiles.push(WallTile {
            minus,
            plus,
            basis,
            minus_data: half(minus, key.minus, minus_corners),
            plus_data: half(plus, key.plus, plus_corners),
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
    let mut collect = |data: &SmallVec<[IVec2; 2]>, side: WallSide| {
        let collapses = match face {
            Face::Init => key.resets(side),
            Face::Meas => key.measures(side),
        };
        for &qubit in data {
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
