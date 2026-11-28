use bloq_graph::{Basis, Direction};
use glam::IVec2;
use smallvec::SmallVec;

use crate::block::fixed_bulk::utils::{Corner, checkerboard_basis, grid_to_coord, make_cube_tile};
use crate::block::patch::Patch;
use crate::signature::{Connectivity, LayerSchedule};

pub(super) type PatchData = (
    Patch,
    crate::FxMap<IVec2, Basis>,
    crate::FxMap<IVec2, Basis>,
);

#[derive(Debug, Clone, Copy)]
pub(super) struct Cell {
    pub row: usize,
    pub col: usize,
    pub last: usize,
}

impl Cell {
    pub(crate) const fn new(row: usize, col: usize, last: usize) -> Self {
        Self { row, col, last }
    }

    pub(crate) const fn on_top(self) -> bool {
        self.row == 0
    }

    pub(crate) const fn on_bottom(self) -> bool {
        self.row == self.last
    }

    pub(crate) const fn on_left(self) -> bool {
        self.col == 0
    }

    pub(crate) const fn on_right(self) -> bool {
        self.col == self.last
    }

    pub(crate) const fn is_boundary(self) -> bool {
        self.on_top() || self.on_bottom() || self.on_left() || self.on_right()
    }

    pub(crate) const fn is_outer_corner(self) -> bool {
        (self.on_top() || self.on_bottom()) && (self.on_left() || self.on_right())
    }

    /// Grid boundaries this cell sits on, as outward directions.
    pub(crate) fn edges(self) -> impl Iterator<Item = Direction> {
        [
            self.on_top().then_some(Direction::YPLUS),
            self.on_bottom().then_some(Direction::YMINUS),
            self.on_left().then_some(Direction::XMINUS),
            self.on_right().then_some(Direction::XPLUS),
        ]
        .into_iter()
        .flatten()
    }

    /// The boundaries `corner` faces: [`Self::edges`] restricted to the
    /// corner's own side of the cell.
    pub(crate) fn corner_edges(self, corner: Corner) -> impl Iterator<Item = Direction> {
        self.edges().filter(move |dir| match dir {
            Direction::YPLUS => corner.is_top(),
            Direction::YMINUS => !corner.is_top(),
            Direction::XMINUS => corner.is_left(),
            _ => !corner.is_left(),
        })
    }

    /// Whether this cell sits in the seam-facing column a spatial Hadamard
    /// pipe has taken over. The wall template places those tiles, so the cube
    /// skips the cell outright rather than classifying it.
    pub(crate) fn is_ceded(self, connectivity: Connectivity) -> bool {
        self.edges().any(|dir| connectivity.has_hadamard(dir))
    }

    /// The measure-qubit basis follows the physical coordinate's
    /// checkerboard, not the grid index — the two only coincide while the
    /// grid width `n = distance + 1` is even.
    pub(crate) fn basis(self) -> Basis {
        checkerboard_basis(self.pos())
    }

    pub(crate) fn pos(self) -> IVec2 {
        grid_to_coord(self.row, self.col, self.last + 1)
    }
}

#[derive(Debug, Clone)]
pub(super) struct TileSpec {
    pub basis: Basis,
    pub active: ActiveCorners,
    pub is_vertical: bool,
}

impl TileSpec {
    pub(crate) fn new(basis: Basis, active: impl Into<ActiveCorners>, is_vertical: bool) -> Self {
        Self {
            basis,
            active: active.into(),
            is_vertical,
        }
    }
}

pub(super) const fn hook_orientation(basis: Basis, z_is_vertical: bool) -> bool {
    match basis {
        Basis::Z => z_is_vertical,
        Basis::X => !z_is_vertical,
    }
}

pub(super) fn build_patch_from_cells(
    distance: u32,
    connectivity: Connectivity,
    temporal_basis: Basis,
    layer_schedule: LayerSchedule,
    classify: impl Fn(Cell) -> Option<TileSpec>,
) -> PatchData {
    let n = (distance + 1) as usize;
    let last = n - 1;
    let init_below = !connectivity.has_pipe(Direction::ZMINUS);
    let meas_above = !connectivity.has_pipe(Direction::ZPLUS);

    let mut build = PatchBuild::with_capacity(n * n);
    if init_below {
        build.init_data.reserve(n * n);
    }
    if meas_above {
        build.meas_data.reserve(n * n);
    }
    // `grid_to_coord` flips rows, so descending rows emit the patch's `(y, x)` order.
    for row in (0..n).rev() {
        for col in 0..n {
            let cell = Cell::new(row, col, last);
            if cell.is_ceded(connectivity) {
                continue;
            }
            let Some(spec) = classify(cell) else {
                continue;
            };
            let pos = cell.pos();
            build.tiles.push(make_cube_tile(
                spec.basis,
                pos,
                &spec.active,
                spec.is_vertical,
                layer_schedule,
            ));

            for &corner in &spec.active {
                // A data qubit shared with a spatial neighbour is always
                // prepared and read out, so no stabilizer endpoint dangles. A
                // ceded edge is not such a seam: its shared column belongs to
                // the wall pipe, which owns those qubits' collapse.
                let on_open_edge = cell
                    .corner_edges(corner)
                    .any(|dir| connectivity.has_open_edge(dir));
                let dq = pos + corner.to_ivec2();
                if init_below || on_open_edge {
                    build.init_data.insert(dq, temporal_basis);
                }
                if meas_above || on_open_edge {
                    build.meas_data.insert(dq, temporal_basis);
                }
            }
        }
    }

    (
        Patch::from_sorted_tiles(build.tiles),
        build.init_data,
        build.meas_data,
    )
}

/// Render a built patch as a stable text table: one line per tile giving its
/// basis, measure qubit and per-slot corner, then the collapse maps. Snapshot
/// fodder for the patch-shape tests, which care about exactly these three.
#[cfg(test)]
pub(super) fn describe_patch((patch, init, meas): &PatchData) -> String {
    use itertools::Itertools;
    use std::fmt::Write as _;

    let mut out = String::new();
    for tile in patch.tiles() {
        let m = tile.measure_qubit();
        let slots = tile
            .data_slots()
            .iter()
            .map(|slot| match slot {
                Some(dq) => corner_name(*dq - m),
                None => "..",
            })
            .join(" ");
        let _ = writeln!(out, "{:?} ({:>2},{:>2}) [{slots}]", tile.basis(), m.x, m.y);
    }
    for (label, data) in [("init", init), ("meas", meas)] {
        let qubits = data
            .keys()
            .sorted_by_key(|q| (q.y, q.x))
            .map(|q| format!("({},{})", q.x, q.y))
            .join(" ");
        let _ = writeln!(out, "{label}: {qubits}");
    }
    out
}

#[cfg(test)]
fn corner_name(offset: IVec2) -> &'static str {
    match (offset.x > 0, offset.y > 0) {
        (false, true) => "TL",
        (true, true) => "TR",
        (false, false) => "BL",
        (true, false) => "BR",
    }
}

pub(super) type ActiveCorners = SmallVec<[Corner; 4]>;

#[derive(Default)]
struct PatchBuild {
    tiles: Vec<crate::block::patch::Tile>,
    init_data: crate::FxMap<IVec2, Basis>,
    meas_data: crate::FxMap<IVec2, Basis>,
}

impl PatchBuild {
    fn with_capacity(tile_capacity: usize) -> Self {
        Self {
            tiles: Vec::with_capacity(tile_capacity),
            init_data: crate::FxMap::default(),
            meas_data: crate::FxMap::default(),
        }
    }
}
