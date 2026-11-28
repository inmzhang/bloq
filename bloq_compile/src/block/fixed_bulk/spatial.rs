//! Spatial cube (ZZX, XXZ) patch construction.
//!
//! Builds a surface code `Patch` for cubes where `x() == y()` (same basis on
//! all spatial boundaries). The interior is divided into quadrants
//! (TOP/RIGHT/BOTTOM/LEFT) with per-quadrant hook orientation.

use bloq_graph::{Basis, CubeKind, Direction};

use crate::block::fixed_bulk::grid::{
    ActiveCorners, Cell, PatchData, TileSpec, build_patch_from_cells, hook_orientation,
};
use crate::block::fixed_bulk::utils::{Corner, Corner::*};
use crate::signature::{Connectivity, LayerSchedule};

// =============================================================================
// Quadrant classification
// =============================================================================

/// Interior cells are assigned to quadrants based on diagonal distance from the
/// corners. The quadrant determines hook orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quadrant {
    Top,
    Right,
    Bottom,
    Left,
}

fn classify_quadrant(row: usize, col: usize, last: usize) -> Quadrant {
    // The padded CX order runs TL to BR. The lower diagonals belong to their
    // side quadrants: Bottom hooks there shorten diagonal error chains.
    if row <= col && row <= (last - col) {
        Quadrant::Top
    } else if row <= col && row > (last - col) {
        Quadrant::Right
    } else if row > col && row > (last - col) {
        Quadrant::Bottom
    } else {
        Quadrant::Left
    }
}

// =============================================================================
// Main construction
// =============================================================================

/// Build a Patch + init/meas data maps for a spatial (ZZX/XXZ) cube.
///
/// Returns `(patch, init_data, meas_data)`.
pub(super) fn build_spatial_patch(
    cube_kind: CubeKind,
    distance: u32,
    connectivity: Connectivity,
    layer_schedule: LayerSchedule,
) -> PatchData {
    let sbb = cube_kind.x(); // spatial boundary basis = x() = y()
    let classifier = SpatialClassifier::new(sbb, connectivity);

    build_patch_from_cells(
        distance,
        connectivity,
        cube_kind.z(),
        layer_schedule,
        |cell| classifier.classify(cell),
    )
}

// =============================================================================
// Cell classification helpers
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct SpatialClassifier {
    spatial_boundary_basis: Basis,
    boundary_is_z: bool,
    arms: SpatialArms,
    hooks: SpatialHooks,
    connectivity: Connectivity,
}

impl SpatialClassifier {
    fn new(spatial_boundary_basis: Basis, connectivity: Connectivity) -> Self {
        let boundary_is_z = spatial_boundary_basis == Basis::Z;
        let arms = spatial_arms(connectivity);
        Self {
            spatial_boundary_basis,
            boundary_is_z,
            arms,
            hooks: spatial_hooks(
                PerEdge {
                    up: connectivity.has_pipe(Direction::YPLUS),
                    down: connectivity.has_pipe(Direction::YMINUS),
                    left: connectivity.has_pipe(Direction::XMINUS),
                    right: connectivity.has_pipe(Direction::XPLUS),
                },
                boundary_is_z,
            ),
            connectivity,
        }
    }

    fn classify(self, cell: Cell) -> Option<TileSpec> {
        if cell.is_outer_corner() {
            classify_outer_corner(
                cell,
                self.boundary_is_z,
                self.spatial_boundary_basis,
                self.arms,
            )
        } else if cell.is_boundary() {
            classify_edge(cell, self.spatial_boundary_basis, self.arms, self.hooks)
        } else {
            Some(classify_interior(
                cell,
                self.boundary_is_z,
                self.spatial_boundary_basis,
                self.arms,
                self.hooks,
                self.connectivity,
            ))
        }
    }
}

/// One value per grid edge, addressable by direction, by the boundary a cell
/// sits on, or by quadrant.
#[derive(Debug, Clone, Copy)]
struct PerEdge<T> {
    up: T,
    down: T,
    left: T,
    right: T,
}

impl<T: Copy> PerEdge<T> {
    const fn at_dir(self, dir: Direction) -> T {
        match dir {
            Direction::YPLUS => self.up,
            Direction::YMINUS => self.down,
            Direction::XMINUS => self.left,
            _ => self.right,
        }
    }

    const fn at_edge(self, cell: Cell) -> T {
        if cell.on_top() {
            self.up
        } else if cell.on_bottom() {
            self.down
        } else if cell.on_left() {
            self.left
        } else {
            self.right
        }
    }

    const fn in_quadrant(self, quadrant: Quadrant) -> T {
        match quadrant {
            Quadrant::Top => self.up,
            Quadrant::Bottom => self.down,
            Quadrant::Left => self.left,
            Quadrant::Right => self.right,
        }
    }
}

type SpatialArms = PerEdge<bool>;
type SpatialHooks = PerEdge<bool>;

/// A ceded (spatial Hadamard) edge has no arm: the wall replaces the cube's
/// boundary there rather than continuing its lattice, so the corner, edge and
/// quadrant-hook rules all treat it as closed.
fn spatial_arms(connectivity: Connectivity) -> SpatialArms {
    PerEdge {
        up: connectivity.has_open_edge(Direction::YPLUS),
        down: connectivity.has_open_edge(Direction::YMINUS),
        left: connectivity.has_open_edge(Direction::XMINUS),
        right: connectivity.has_open_edge(Direction::XPLUS),
    }
}

/// A missing arm flips the hook, so hook errors cannot shortcut the boundary
/// that replaced it.
fn spatial_hooks(arms: SpatialArms, boundary_is_z: bool) -> SpatialHooks {
    PerEdge {
        up: arms.up == boundary_is_z,
        down: arms.down == boundary_is_z,
        left: arms.left != boundary_is_z,
        right: arms.right != boundary_is_z,
    }
}

/// Classify an outer corner cell (row 0/last, col 0/last).
///
/// Returns `Some((basis, active_corners, is_vertical))` or `None` (removed).
fn classify_outer_corner(
    cell: Cell,
    boundary_is_z: bool,
    sbb: Basis,
    arms: SpatialArms,
) -> Option<TileSpec> {
    let h_dir = if cell.on_left() {
        Direction::XMINUS
    } else {
        Direction::XPLUS
    };
    let v_dir = if cell.on_top() {
        Direction::YPLUS
    } else {
        Direction::YMINUS
    };
    let (outward_corner, corner_is_z) = match (cell.on_top(), cell.on_left()) {
        (true, true) => (TL, true),
        (true, false) => (TR, false),
        (false, true) => (BL, false),
        (false, false) => (BR, true),
    };

    // tqec's spatial-cube template only ever uses the "same-parity" outer
    // corners: TL/BR for Z-boundary cubes and TR/BL for X-boundary cubes.
    // The opposite-parity outer corners are always empty, even when both
    // adjacent arms are present.
    if corner_is_z != boundary_is_z {
        return None;
    }

    match (arms.at_dir(h_dir), arms.at_dir(v_dir)) {
        // Missing both adjacent arms deletes the external corner. tqec then
        // converts the inner bulk corner into a 3-body plaquette instead.
        (false, false) => None,
        // When both adjacent arms are present, tqec's arm replacement turns the
        // external corner into a 3-body plaquette that omits the outward corner.
        (true, true) => {
            let active: ActiveCorners = Corner::ALL
                .into_iter()
                .filter(|&corner| corner != outward_corner)
                .collect();
            Some(TileSpec::new(sbb, active, boundary_is_z))
        }
        // With exactly one adjacent arm present, the missing arm contributes a
        // 2-body boundary plaquette pointing inward from that boundary.
        (false, true) => Some(TileSpec::new(sbb, corners_away_from(h_dir), false)),
        (true, false) => Some(TileSpec::new(sbb, corners_away_from(v_dir), false)),
    }
}

/// The two corners on the far side of `dir` — the support of the 2-body
/// plaquette that points inward from that boundary.
fn corners_away_from(dir: Direction) -> ActiveCorners {
    match dir {
        Direction::YPLUS => [BL, BR],
        Direction::YMINUS => [TL, TR],
        Direction::XMINUS => [TR, BR],
        _ => [TL, BL],
    }
    .into_iter()
    .collect()
}

/// Classify a boundary edge cell (on one boundary, not a corner).
fn classify_edge(
    cell: Cell,
    sbb: Basis,
    arms: SpatialArms,
    hooks: SpatialHooks,
) -> Option<TileSpec> {
    let basis = cell.basis();

    if arms.at_edge(cell) {
        Some(TileSpec::new(
            basis,
            Corner::ALL,
            hook_orientation(basis, hooks.at_edge(cell)),
        ))
    } else {
        // Arm absent → only keep if checkerboard basis matches sbb.
        let edge = cell.edges().next().expect("edge cells sit on a boundary");
        (basis == sbb).then(|| TileSpec::new(sbb, corners_away_from(edge), false))
    }
}

/// Classify an interior cell (not on any boundary).
fn classify_interior(
    cell: Cell,
    boundary_is_z: bool,
    sbb: Basis,
    arms: SpatialArms,
    hooks: SpatialHooks,
    connectivity: Connectivity,
) -> TileSpec {
    let basis = cell.basis();
    inner_corner_replacement(cell, arms, connectivity)
        .filter(|replacement| replacement.needs_z == boundary_is_z && basis == sbb)
        .map(|replacement| {
            let active: ActiveCorners = Corner::ALL
                .iter()
                .copied()
                .filter(|&corner| corner != replacement.removed_corner)
                .collect();
            TileSpec::new(sbb, active, boundary_is_z)
        })
        .unwrap_or_else(|| {
            let quadrant = classify_quadrant(cell.row, cell.col, cell.last);
            TileSpec::new(
                basis,
                Corner::ALL,
                hook_orientation(basis, hooks.in_quadrant(quadrant)),
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct InnerCornerReplacement {
    removed_corner: Corner,
    needs_z: bool,
}

fn inner_corner_replacement(
    cell: Cell,
    arms: SpatialArms,
    connectivity: Connectivity,
) -> Option<InnerCornerReplacement> {
    let candidates = [
        (
            cell.row == 1 && cell.col == 1 && !arms.left && !arms.up,
            TL,
            true,
            Direction::XMINUS,
            Direction::YPLUS,
        ),
        (
            cell.row == 1 && cell.col == cell.last - 1 && !arms.up && !arms.right,
            TR,
            false,
            Direction::YPLUS,
            Direction::XPLUS,
        ),
        (
            cell.row == cell.last - 1 && cell.col == 1 && !arms.down && !arms.left,
            BL,
            false,
            Direction::YMINUS,
            Direction::XMINUS,
        ),
        (
            cell.row == cell.last - 1 && cell.col == cell.last - 1 && !arms.right && !arms.down,
            BR,
            true,
            Direction::XPLUS,
            Direction::YMINUS,
        ),
    ];

    candidates
        .into_iter()
        .find_map(|(matches, removed_corner, needs_z, edge_a, edge_b)| {
            (matches && !connectivity.has_hadamard(edge_a) && !connectivity.has_hadamard(edge_b))
                .then_some(InnerCornerReplacement {
                    removed_corner,
                    needs_z,
                })
        })
}

#[cfg(test)]
mod tests {
    use bloq_stim::StimFlowVerifier;
    use rstest::rstest;

    use super::*;

    use crate::block::fixed_bulk::grid::describe_patch;
    use crate::block::fixed_bulk::utils::make_surface_code_chunk;

    #[rstest]
    fn test_spatial_bulk_verify_flows(
        #[values(CubeKind::ZZX, CubeKind::XXZ)] kind: CubeKind,
        #[values(3, 5)] d: u32,
    ) {
        let (patch, _, _) =
            build_spatial_patch(kind, d, Connectivity::ISOLATED, LayerSchedule::Padded);
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }

    #[rstest]
    fn test_spatial_init_verify_flows(
        #[values(CubeKind::ZZX, CubeKind::XXZ)] kind: CubeKind,
        #[values(3, 5)] d: u32,
    ) {
        let (patch, init, _) =
            build_spatial_patch(kind, d, Connectivity::ISOLATED, LayerSchedule::Padded);
        let chunk = make_surface_code_chunk(&patch, Some(&init), None).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }

    #[rstest]
    fn test_spatial_meas_verify_flows(
        #[values(CubeKind::ZZX, CubeKind::XXZ)] kind: CubeKind,
        #[values(3, 5)] d: u32,
    ) {
        let (patch, _, meas) =
            build_spatial_patch(kind, d, Connectivity::ISOLATED, LayerSchedule::Padded);
        let chunk = make_surface_code_chunk(&patch, None, Some(&meas)).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }

    #[rstest]
    fn test_spatial_with_pipe(#[values(CubeKind::ZZX, CubeKind::XXZ)] kind: CubeKind) {
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XPLUS);
        let (patch, init, meas) = build_spatial_patch(kind, 3, conn, LayerSchedule::Padded);
        let init_chunk = make_surface_code_chunk(&patch, Some(&init), None).unwrap();
        let bulk_chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        let meas_chunk = make_surface_code_chunk(&patch, None, Some(&meas)).unwrap();
        init_chunk.verify_flows(None, None).unwrap();
        bulk_chunk.verify_flows(None, None).unwrap();
        meas_chunk.verify_flows(None, None).unwrap();
    }

    /// The CZ gallery's far endpoint: a spatial cube whose `-X` edge carries the
    /// wall and whose `±Y` arms merge on. The ceded column is gone and the
    /// `-X` direction counts as armless for the corner, edge and quadrant-hook
    /// rules, so the shape differs from the plain `±Y`-only cube below.
    #[test]
    fn spatial_xxz_d3_hadamard_xminus_with_y_arms_patch() {
        let connectivity = Connectivity::ISOLATED
            .with_hadamard(Direction::XMINUS)
            .with_pipe(Direction::YPLUS)
            .with_pipe(Direction::YMINUS);
        let patch = build_spatial_patch(CubeKind::XXZ, 3, connectivity, LayerSchedule::Wall);

        insta::assert_snapshot!(
            "spatial_xxz_d3_hadamard_xminus_y_arms",
            describe_patch(&patch)
        );
    }

    #[test]
    fn ceded_edge_leaves_the_spatial_cube_armless_there() {
        let arms = Connectivity::ISOLATED
            .with_pipe(Direction::YPLUS)
            .with_pipe(Direction::YMINUS);
        let ceded = arms.with_hadamard(Direction::XMINUS);

        // The classifier sees no `-X` arm...
        assert_eq!(spatial_arms(ceded).left, spatial_arms(arms).left);
        // ...and the wall owns the `-X` cell column outright.
        let (patch, _, _) = build_spatial_patch(CubeKind::XXZ, 3, ceded, LayerSchedule::Wall);
        assert!(patch.tiles().iter().all(|tile| tile.measure_qubit().x > 0));
    }
}
