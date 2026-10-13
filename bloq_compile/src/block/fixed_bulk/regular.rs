//! Regular (non-spatial) cube patch construction.
//!
//! Builds a surface code `Patch` for cubes where the X and Y boundary bases
//! differ (ZXZ, XZX, ZXX, XZZ). The patch is an `n × n` checkerboard grid
//! (`n = distance + 1`) of stabilizer tiles, filtered by boundary basis
//! compatibility and pipe connectivity.

use bloq_graph::{Basis, CubeKind, Direction};

use crate::block::fixed_bulk::grid::{
    ActiveCorners, Cell, PatchData, TileSpec, build_patch_from_cells_with_reverse, hook_orientation,
};
use crate::block::fixed_bulk::utils::Corner;
use crate::signature::{Connectivity, LayerSchedule};

/// Build a Patch + init/meas data maps for a regular (non-spatial) cube.
///
/// Returns `(patch, init_data, meas_data)` where:
/// - `patch` contains all stabilizer tiles
/// - `init_data` maps data qubits → temporal_basis for those that should be reset
/// - `meas_data` maps data qubits → temporal_basis for those that should be measured
#[cfg(test)]
pub(super) fn build_regular_patch(
    cube_kind: CubeKind,
    distance: u32,
    connectivity: Connectivity,
    layer_schedule: LayerSchedule,
) -> PatchData {
    build_regular_patch_with_reverse(cube_kind, distance, connectivity, layer_schedule, false)
}

pub(super) fn build_regular_patch_with_reverse(
    cube_kind: CubeKind,
    distance: u32,
    connectivity: Connectivity,
    layer_schedule: LayerSchedule,
    reverse_schedule: bool,
) -> PatchData {
    let classifier = RegularClassifier {
        x_basis: cube_kind.x(),
        y_basis: cube_kind.y(),
        connectivity,
    };

    build_patch_from_cells_with_reverse(
        distance,
        connectivity,
        cube_kind.z(),
        layer_schedule,
        reverse_schedule,
        |cell| classifier.classify(cell),
    )
}

#[derive(Debug, Clone, Copy)]
struct RegularClassifier {
    x_basis: Basis,
    y_basis: Basis,
    connectivity: Connectivity,
}

impl RegularClassifier {
    /// Two quantifiers over the boundaries a cell touches:
    ///
    /// - the cell survives when every edge either hosts its basis or is open;
    /// - a corner stays active when every edge it faces is open.
    ///
    /// A ceded (spatial Hadamard) edge counts as closed: the wall is not a
    /// continuation of this cube's lattice.
    fn classify(self, cell: Cell) -> Option<TileSpec> {
        let basis = cell.basis();
        let open = |dir: Direction| self.connectivity.has_open_edge(dir);

        let compatible = cell
            .edges()
            .all(|dir| basis == self.face_basis(dir) || open(dir));
        if !compatible {
            return None;
        }

        let mut active = ActiveCorners::new();
        for corner in Corner::ALL {
            if cell.corner_edges(corner).all(open) {
                active.push(corner);
            }
        }

        (!active.is_empty()).then(|| {
            TileSpec::new(
                basis,
                active,
                hook_orientation(basis, self.x_basis == Basis::Z),
            )
        })
    }

    fn face_basis(self, dir: Direction) -> Basis {
        if dir.as_udirection() == bloq_graph::UDirection::X {
            self.x_basis
        } else {
            self.y_basis
        }
    }
}

#[cfg(test)]
mod tests {
    use bloq_stim::StimFlowVerifier;
    use rstest::rstest;

    use super::*;
    use crate::block::fixed_bulk::grid::describe_patch;
    use crate::block::fixed_bulk::utils::make_surface_code_chunk;

    #[rstest]
    #[case(3, 8)]
    #[case(5, 24)]
    fn test_regular_patch_isolated_zxz(#[case] distance: u32, #[case] tiles: usize) {
        let (patch, init, meas) = build_regular_patch(
            CubeKind::ZXZ,
            distance,
            Connectivity::ISOLATED,
            LayerSchedule::Compact,
        );
        assert_eq!(patch.tiles().len(), tiles);
        assert_eq!(patch.z_tiles().count(), tiles / 2);
        assert_eq!(patch.x_tiles().count(), tiles / 2);
        // All data qubits should be in init and meas (isolated = no temporal pipes)
        assert_eq!(init.len(), meas.len());
        assert!(!init.is_empty());
    }

    #[rstest]
    #[case(CubeKind::ZXZ, Connectivity::ISOLATED)]
    #[case(CubeKind::XZX, Connectivity::ISOLATED)]
    #[case(CubeKind::XZZ, Connectivity::ISOLATED)]
    #[case(CubeKind::ZXX, Connectivity::ISOLATED)]
    #[case(CubeKind::ZXZ, Connectivity::ISOLATED.with_pipe(Direction::ZMINUS).with_pipe(Direction::ZPLUS))]
    #[case(CubeKind::XZX, Connectivity::ISOLATED.with_pipe(Direction::ZMINUS).with_pipe(Direction::ZPLUS))]
    #[case(CubeKind::ZXZ, Connectivity::ISOLATED.with_pipe(Direction::XPLUS))]
    #[case(CubeKind::XZX, Connectivity::ISOLATED.with_pipe(Direction::XPLUS))]
    fn test_regular_flows_d3(#[case] kind: CubeKind, #[case] connectivity: Connectivity) {
        let (patch, init, meas) =
            build_regular_patch(kind, 3, connectivity, LayerSchedule::Compact);
        let init_chunk = make_surface_code_chunk(&patch, Some(&init), None).unwrap();
        let bulk_chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        let meas_chunk = make_surface_code_chunk(&patch, None, Some(&meas)).unwrap();
        init_chunk.verify_flows(None, None).unwrap();
        bulk_chunk.verify_flows(None, None).unwrap();
        meas_chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_regular_temporal_pipe_reduces_init() {
        let conn = Connectivity::ISOLATED.with_pipe(Direction::ZMINUS);
        let (_, init_piped, _) =
            build_regular_patch(CubeKind::ZXZ, 3, conn, LayerSchedule::Compact);
        let (_, init_iso, _) = build_regular_patch(
            CubeKind::ZXZ,
            3,
            Connectivity::ISOLATED,
            LayerSchedule::Compact,
        );
        // With temporal pipe below, fewer data qubits are in init_data
        assert!(init_piped.len() < init_iso.len());
    }

    #[test]
    fn test_regular_spatial_pipe_adds_tiles() {
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XPLUS);
        let (patch_piped, _, _) =
            build_regular_patch(CubeKind::ZXZ, 3, conn, LayerSchedule::Compact);
        let (patch_iso, _, _) = build_regular_patch(
            CubeKind::ZXZ,
            3,
            Connectivity::ISOLATED,
            LayerSchedule::Compact,
        );
        // Spatial pipe extends the patch with boundary tiles
        assert!(patch_piped.tiles().len() > patch_iso.tiles().len());
    }

    /// Flow counts for d=3 ZZX isolated must match the expected values.
    #[test]
    fn test_d3_zxz_flow_counts() {
        let (patch, init, meas) = build_regular_patch(
            CubeKind::ZXZ,
            3,
            Connectivity::ISOLATED,
            LayerSchedule::Compact,
        );
        let init_chunk = make_surface_code_chunk(&patch, Some(&init), None).unwrap();
        let bulk_chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        let meas_chunk = make_surface_code_chunk(&patch, None, Some(&meas)).unwrap();
        assert_eq!(init_chunk.flows.len(), 12);
        assert_eq!(bulk_chunk.flows.len(), 16);
        assert_eq!(meas_chunk.flows.len(), 12);
    }

    /// A `+X` spatial Hadamard cedes the cube's rightmost cell column to the
    /// wall pipe. Everything the cube keeps is pinned here: the seam column is
    /// gone, the bulk column behind it is a full-width interior column, and the
    /// slots are the six-deep extended schedule.
    #[test]
    fn regular_xzx_d3_hadamard_xplus_patch() {
        let connectivity = Connectivity::ISOLATED.with_hadamard(Direction::XPLUS);
        let patch = build_regular_patch(CubeKind::XZX, 3, connectivity, LayerSchedule::Extended);

        insta::assert_snapshot!("regular_xzx_d3_hadamard_xplus", describe_patch(&patch));
    }

    /// The ceded edge is *not* an open edge: the cube places no tiles there, and
    /// the shared data column collapses on the cube's own temporal rules only —
    /// unlike a plain merge, whose open-edge override forces init and readout.
    #[test]
    fn ceded_edge_neither_hosts_tiles_nor_forces_collapse() {
        let with = |connectivity| {
            build_regular_patch(CubeKind::XZX, 3, connectivity, LayerSchedule::Compact)
        };
        let seam_column = 6; // `2 * d` — the `+X` edge cell column at d = 3
        let below = Connectivity::ISOLATED.with_pipe(Direction::ZMINUS);

        let (ceded, ceded_init, _) = with(below.with_hadamard(Direction::XPLUS));
        let (open, open_init, _) = with(below.with_pipe(Direction::XPLUS));

        assert!(
            ceded
                .tiles()
                .iter()
                .all(|tile| tile.measure_qubit().x < seam_column)
        );
        assert!(
            open.tiles()
                .iter()
                .any(|tile| tile.measure_qubit().x == seam_column)
        );
        // The `-Z` pipe suppresses init everywhere the open-edge override does
        // not fire, so the two maps separate cleanly.
        assert!(ceded_init.is_empty());
        assert!(!open_init.is_empty());
    }
}
