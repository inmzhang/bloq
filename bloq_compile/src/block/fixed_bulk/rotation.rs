//! Fixed-bulk-convention patch rotation patch construction.
//!
//! This module constructs the intermediate patch-rotation geometries directly
//! on the fixed-bulk surface code lattice. Measure qubits stay on even
//! coordinates, data qubits stay on odd coordinates, and stabilizer bases are
//! always assigned by the fixed-bulk checkerboard, whose top-left bulk tile is
//! Z.

use std::sync::Arc;

use bloq_circuit::{Chunk, ChunkOrLoop, Flow, FlowMeasurements, PauliMap};
use bloq_graph::{Basis, Direction, PatchRotationKind, Pauli};
use glam::IVec2;

use crate::block::fixed_bulk::observable::{
    SelectedChunkMeasurements, SelectedGateway, line_qubits, resolve_gateway_measurements,
};
use crate::block::fixed_bulk::utils::{
    CXSchedule, Corner, EdgeBases, Rect, carve_patch, checkerboard_basis,
    collapse_matching_data_boundary, emit_syndrome_round, sorted_batches, sorted_unique_coords,
    tile_cx_schedule,
};
use crate::block::gateway::{LocalStabilizer, gateway_key_connectivity};
use crate::block::measurements::MeasurementIndex;
use crate::block::patch::{Patch, Tile};
use crate::block::{CompiledTemplate, LoweringTemplate};
use crate::signature::Connectivity;
use crate::{CompileError, FxMap, FxSet};

// Chunk positions in the fixed six-stage schedule built by
// [`patch_rotation_chunks`]: `[grow-init, grown loop, transfer-meas,
// rotated-init, rotated loop, shrink-meas]`.
const GROW_INIT_CHUNK: usize = 0;
const ROTATED_INIT_CHUNK: usize = 3;
const SHRINK_MEAS_CHUNK: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RotationStage {
    Grown,
    Rotated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RotationGeometry {
    width: i32,
    height: i32,
    split_y: i32,
}

impl RotationGeometry {
    fn new(distance: u32) -> Self {
        let distance = distance as i32;
        Self {
            width: distance,
            height: 2 * distance + 1,
            split_y: distance,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RotationLayout {
    input_x_basis: Basis,
    movement: IVec2,
    geometry: RotationGeometry,
}

impl RotationLayout {
    fn new(kind: PatchRotationKind, distance: u32) -> Self {
        Self {
            input_x_basis: kind.x_axis_boundary_basis(),
            movement: kind.movement(),
            geometry: RotationGeometry::new(distance),
        }
    }

    fn local_to_global(self, p: IVec2, offset: IVec2) -> IVec2 {
        (match self.movement.to_array() {
            [0, 1] => p,
            [0, -1] => IVec2::new(-p.x, -p.y),
            [1, 0] => IVec2::new(p.y, p.x),
            [-1, 0] => IVec2::new(-p.y, p.x),
            _ => unreachable!("PatchRotationKind rejects non-cardinal movement"),
        }) + offset
    }
}

pub(super) fn compile_patch_rotation(
    kind: PatchRotationKind,
    connectivity: Connectivity,
    distance: u32,
) -> Result<CompiledTemplate, CompileError> {
    let layout = RotationLayout::new(kind, distance);
    // The spatial offset depends only on the layout, so resolve it once rather than
    // per gateway key (the closure below runs once per observable key).
    let offset = patch_rotation_offset(layout);
    let patches = RotationPatches::new(layout, offset);
    let chunks = patch_rotation_chunks(layout, distance, &patches)?;
    let observable_gateway = resolve_gateway_measurements(
        build_patch_rotation_gateway(layout, offset, connectivity, &patches),
        &chunks,
        |key| patch_rotation_operator_faces(layout, offset, key),
    );
    Ok(Arc::new(LoweringTemplate::from_chunks(
        chunks,
        observable_gateway,
    )?))
}

fn patch_rotation_chunks(
    layout: RotationLayout,
    distance: u32,
    patches: &RotationPatches,
) -> Result<Vec<ChunkOrLoop>, CompileError> {
    let init_basis = layout.input_x_basis.flip();
    let meas_basis = init_basis.flip();
    let grown_init_data = data_difference(&patches.grown, &patches.input, init_basis);
    let grown_meas_data = data_difference_by_adjacent_tile_basis(&patches.grown, &patches.rotated);
    let rotated_init_data =
        data_difference_by_adjacent_tile_basis(&patches.rotated, &patches.grown);
    let rotated_meas_data = data_difference(&patches.rotated, &patches.output, meas_basis);
    let loop_repetitions = distance.saturating_sub(2);

    Ok(vec![
        ChunkOrLoop::Single(Box::new(make_patch_rotation_round_with_flow_boundaries(
            &patches.grown,
            &patches.input,
            &patches.grown,
            non_empty_data_map(&grown_init_data),
            None,
            FlowBoundaryData::default(),
        )?)),
        ChunkOrLoop::Loop {
            body: vec![make_patch_rotation_round_with_flow_boundaries(
                &patches.grown,
                &patches.grown,
                &patches.grown,
                None,
                None,
                FlowBoundaryData::default(),
            )?],
            repetitions: loop_repetitions,
        },
        ChunkOrLoop::Single(Box::new(make_patch_rotation_round_with_flow_boundaries(
            &patches.grown,
            &patches.grown,
            &patches.rotated,
            None,
            non_empty_data_map(&grown_meas_data),
            FlowBoundaryData {
                next_init_data: Some(&rotated_init_data),
                ..FlowBoundaryData::default()
            },
        )?)),
        ChunkOrLoop::Single(Box::new(make_patch_rotation_round_with_flow_boundaries(
            &patches.rotated,
            &patches.grown,
            &patches.rotated,
            non_empty_data_map(&rotated_init_data),
            None,
            FlowBoundaryData {
                previous_meas_data: Some(&grown_meas_data),
                ..FlowBoundaryData::default()
            },
        )?)),
        ChunkOrLoop::Loop {
            body: vec![make_patch_rotation_round_with_flow_boundaries(
                &patches.rotated,
                &patches.rotated,
                &patches.rotated,
                None,
                None,
                FlowBoundaryData::default(),
            )?],
            repetitions: loop_repetitions,
        },
        ChunkOrLoop::Single(Box::new(make_patch_rotation_round_with_flow_boundaries(
            &patches.rotated,
            &patches.rotated,
            &patches.output,
            None,
            non_empty_data_map(&rotated_meas_data),
            FlowBoundaryData::default(),
        )?)),
    ])
}

/// Boundary operators on a patch rotation's temporal faces.
///
/// Unlike a realignment (which flips X↔Z in place), a patch rotation keeps the
/// logical basis and **moves** the operator's support: it enters on the input
/// patch's middle line and leaves on the output patch's middle line, which is
/// the same line turned 90° (the rotation swaps the patch's x- and y-axis
/// boundaries). So `operator_in` equals the lower cube's `operator_out` and
/// `operator_out` the upper cube's `operator_in`, and both seams cancel during
/// lowering.
///
/// The lines are built in the template frame (through `local_to_global`, the
/// same placement the patch tiles use), so the global-frame seam match lines
/// them up with the neighbouring cubes despite the spatial move. Presence is
/// gated on the key's temporal arms, mirroring `cube_operator_faces`.
fn patch_rotation_operator_faces(
    layout: RotationLayout,
    offset: IVec2,
    key: LocalStabilizer,
) -> (PauliMap, PauliMap) {
    let basis =
        Basis::try_from(key.center_basis()).expect("a rotation flow key carries an X/Z center");
    let pauli = Pauli::from(basis);

    // The middle line runs along the axis whose like-basis boundaries it
    // connects. The input patch carries `input_x_basis` on its x boundaries, so a
    // basis-`b` line runs horizontally exactly when `b == input_x_basis`; the
    // output patch flips that axis, turning the support 90°.
    let operator_face = |present: bool, y0: i32, horizontal: bool| -> PauliMap {
        if !present {
            return PauliMap::empty();
        }
        standard_patch_line(layout, offset, y0, horizontal)
            .into_iter()
            .map(|q| (q, pauli))
            .collect()
    };
    let input_horizontal = basis == layout.input_x_basis;
    let operator_in = operator_face(key.has_arm(Direction::ZMINUS), 0, input_horizontal);
    let operator_out = operator_face(
        key.has_arm(Direction::ZPLUS),
        layout.geometry.split_y + 1,
        !input_horizontal,
    );
    (operator_in, operator_out)
}

/// Middle logical line of a standard `d×d` patch rooted at local row `y0`,
/// expressed in the template frame: the shared middle line lifted to `y0` and
/// pushed through the rotation's spatial move, matching the placement of the
/// patch's own data qubits.
fn standard_patch_line(
    layout: RotationLayout,
    offset: IVec2,
    y0: i32,
    horizontal: bool,
) -> Vec<IVec2> {
    let shift = IVec2::new(0, 2 * y0);
    line_qubits(layout.geometry.width as u32, horizontal)
        .into_iter()
        .map(|q| layout.local_to_global(q + shift, offset))
        .collect()
}

fn build_patch_rotation_gateway(
    layout: RotationLayout,
    offset: IVec2,
    connectivity: Connectivity,
    patches: &RotationPatches,
) -> SelectedGateway {
    let mut gateway = SelectedGateway::default();
    let horizontal = Pauli::from(layout.input_x_basis);
    let vertical = Pauli::from(layout.input_x_basis.flip());
    let horizontal_measurements = horizontal_observable_measurements(layout, offset, patches);
    let vertical_measurements = vertical_observable_measurements(layout, offset, patches);

    for gateway_connectivity in patch_rotation_gateway_connectivity_subsets(connectivity) {
        gateway.insert(
            LocalStabilizer::new(horizontal, gateway_connectivity),
            horizontal_measurements.clone(),
        );
        gateway.insert(
            LocalStabilizer::new(vertical, gateway_connectivity),
            vertical_measurements.clone(),
        );
    }

    gateway
}

/// Every subset of the temporal pipe dirs (including the empty one), each as a
/// Hadamard-free gateway-key connectivity.
fn patch_rotation_gateway_connectivity_subsets(connectivity: Connectivity) -> Vec<Connectivity> {
    let temporal_dirs = [Direction::ZMINUS, Direction::ZPLUS]
        .into_iter()
        .filter(|&dir| connectivity.has_pipe(dir))
        .collect::<Vec<_>>();
    (0..1usize << temporal_dirs.len())
        .map(|mask| {
            gateway_key_connectivity(
                temporal_dirs
                    .iter()
                    .enumerate()
                    .filter(|&(index, _)| mask & (1 << index) != 0)
                    .map(|(_, &dir)| dir),
            )
        })
        .collect()
}

fn horizontal_observable_measurements(
    layout: RotationLayout,
    offset: IVec2,
    patches: &RotationPatches,
) -> Vec<SelectedChunkMeasurements> {
    let g = layout.geometry;
    let mixed_right = MixedSide::for_layout(layout).is_right();
    let middle_line_end = if g.width % 4 == 1 {
        g.width / 2 + 1
    } else {
        g.width / 2
    };
    let middle_line_xs = if mixed_right {
        g.width - middle_line_end..g.width
    } else {
        0..middle_line_end
    };

    vec![
        SelectedChunkMeasurements {
            chunk_index: GROW_INIT_CHUNK,
            qubits: selected_tile_measure_qubits_by_local_measure_coord(
                &patches.grown,
                layout,
                offset,
                layout.input_x_basis,
                |m| in_horizontal_grown_observable_region(m, g, mixed_right),
            ),
        },
        SelectedChunkMeasurements {
            chunk_index: SHRINK_MEAS_CHUNK,
            qubits: sorted_unique_coords(local_data_region(
                layout,
                offset,
                middle_line_xs,
                g.split_y..g.split_y + 1,
            )),
        },
    ]
}

fn vertical_observable_measurements(
    layout: RotationLayout,
    offset: IVec2,
    patches: &RotationPatches,
) -> Vec<SelectedChunkMeasurements> {
    let g = layout.geometry;
    let mixed_right = MixedSide::for_layout(layout).is_right();

    vec![SelectedChunkMeasurements {
        chunk_index: ROTATED_INIT_CHUNK,
        qubits: selected_tile_measure_qubits_by_local_measure_coord(
            &patches.rotated,
            layout,
            offset,
            layout.input_x_basis.flip(),
            |m| in_vertical_rotated_observable_region(m, g, mixed_right),
        ),
    }]
}

fn in_horizontal_grown_observable_region(
    tile_coord: IVec2,
    geometry: RotationGeometry,
    mixed_right: bool,
) -> bool {
    let half_start = geometry.width / 2 + 1;
    let in_bottom_patch_top_half = (0..=geometry.width).contains(&tile_coord.x)
        && (half_start..=geometry.split_y).contains(&tile_coord.y);
    let in_top_patch_side_half = if mixed_right {
        (0..=geometry.width / 2).contains(&tile_coord.x)
    } else {
        (half_start..=geometry.width).contains(&tile_coord.x)
    } && (geometry.split_y + 1..=geometry.height)
        .contains(&tile_coord.y);
    in_bottom_patch_top_half || in_top_patch_side_half
}

fn in_vertical_rotated_observable_region(
    tile_coord: IVec2,
    geometry: RotationGeometry,
    mixed_right: bool,
) -> bool {
    let in_lower_patch_side_half = if mixed_right {
        (0..=geometry.width / 2).contains(&tile_coord.x)
    } else {
        (geometry.width / 2 + 1..=geometry.width).contains(&tile_coord.x)
    } && (0..=geometry.split_y).contains(&tile_coord.y);
    let in_upper_patch_bottom_half = (0..=geometry.width).contains(&tile_coord.x)
        && (geometry.split_y + 1..=geometry.split_y + 1 + geometry.width / 2)
            .contains(&tile_coord.y);
    in_lower_patch_side_half || in_upper_patch_bottom_half
}

fn non_empty_data_map(data: &FxMap<IVec2, Basis>) -> Option<&FxMap<IVec2, Basis>> {
    (!data.is_empty()).then_some(data)
}

/// Data qubits in `from` but not `without`, each mapped through `basis_of`.
fn data_difference_with(
    from: &Patch,
    without: &Patch,
    basis_of: impl Fn(IVec2) -> Basis,
) -> FxMap<IVec2, Basis> {
    let without = without.data_set();
    from.data_set()
        .into_iter()
        .filter(|q| !without.contains(q))
        .map(|q| (q, basis_of(q)))
        .collect()
}

fn data_difference(from: &Patch, without: &Patch, basis: Basis) -> FxMap<IVec2, Basis> {
    data_difference_with(from, without, |_| basis)
}

fn data_difference_by_adjacent_tile_basis(from: &Patch, without: &Patch) -> FxMap<IVec2, Basis> {
    data_difference_with(from, without, |q| data_adjacent_tile_basis(from, q))
}

fn data_adjacent_tile_basis(patch: &Patch, data: IVec2) -> Basis {
    patch
        .tiles()
        .iter()
        .filter_map(|tile| {
            let active = tile.active_data_qubits();
            active.contains(&data).then_some((active.len(), tile))
        })
        .max_by_key(|(weight, tile)| (*weight, -tile.measure_qubit().y, -tile.measure_qubit().x))
        .map(|(_, tile)| tile.basis())
        .expect("data qubit from patch data set belongs to at least one tile")
}

fn patch_rotation_offset(layout: RotationLayout) -> IVec2 {
    let geometry = layout.geometry;
    let cell_shift = 2 * geometry.width;
    let origin_shift = match layout.movement.to_array() {
        [-1, 0] => IVec2::new(cell_shift, 0),
        [0, -1] => IVec2::new(cell_shift, cell_shift),
        _ => IVec2::ZERO,
    };

    [
        IVec2::ZERO,
        IVec2::new(2, 0),
        IVec2::new(0, 2),
        IVec2::new(2, 2),
    ]
    .into_iter()
    .map(|phase| phase + origin_shift)
    .find(|&offset| {
        top_left_lattice_basis(layout, offset, geometry.width, geometry.height) == Basis::Z
    })
    .expect("the four phases enumerate every 2-periodic checkerboard parity, one of which is Z")
}

fn top_left_lattice_basis(layout: RotationLayout, offset: IVec2, width: i32, height: i32) -> Basis {
    [
        IVec2::new(0, 0),
        IVec2::new(2 * width, 0),
        IVec2::new(0, 2 * height),
        IVec2::new(2 * width, 2 * height),
    ]
    .into_iter()
    .map(|p| layout.local_to_global(p, offset))
    .max_by_key(|p| (p.y, -p.x))
    .map(checkerboard_basis)
    .expect("rectangular lattice has corners")
}

#[derive(Debug, Clone)]
struct RotationPatches {
    input: Patch,
    grown: Patch,
    rotated: Patch,
    output: Patch,
}

impl RotationPatches {
    fn new(layout: RotationLayout, offset: IVec2) -> Self {
        let geometry = layout.geometry;
        Self {
            input: make_standard_patch(layout, offset, 0, layout.input_x_basis),
            grown: make_patch(RotationStage::Grown, layout, offset),
            rotated: make_patch(RotationStage::Rotated, layout, offset),
            output: make_standard_patch(
                layout,
                offset,
                geometry.split_y + 1,
                layout.input_x_basis.flip(),
            ),
        }
    }
}

#[cfg(test)]
pub(super) fn paper_rotation_patches(kind: PatchRotationKind, distance: u32) -> [Patch; 4] {
    let layout = RotationLayout::new(kind, distance);
    let patches = RotationPatches::new(layout, patch_rotation_offset(layout));
    [
        patches.input,
        patches.grown,
        patches.rotated,
        patches.output,
    ]
}

#[derive(Debug, Clone, Copy)]
struct BoundaryPolicy {
    bottom: Basis,
    top: Basis,
    left_bottom: Basis,
    left_top: Basis,
    right_bottom: Basis,
    right_top: Basis,
    split_y: i32,
}

impl BoundaryPolicy {
    /// The split-edge counterpart of [`EdgeBases::demanded`]: `None` when `m`
    /// is interior or two edges demand different bases.
    fn demanded(self, m: IVec2, rect: Rect) -> Option<Basis> {
        let side = |bottom, top| {
            if is_below_split_midline(m.y, self.split_y) {
                bottom
            } else {
                top
            }
        };
        let mut basis = None;
        for edge_basis in [
            (m.y == 0).then_some(self.bottom),
            (m.y == 2 * rect.height).then_some(self.top),
            (m.x == 0).then_some(side(self.left_bottom, self.left_top)),
            (m.x == 2 * rect.width).then_some(side(self.right_bottom, self.right_top)),
        ]
        .into_iter()
        .flatten()
        {
            if basis.is_some_and(|basis| basis != edge_basis) {
                return None;
            }
            basis = Some(edge_basis);
        }
        basis
    }

    fn keeps(self, m: IVec2, rect: Rect, basis: Basis) -> bool {
        match self.demanded(m, rect) {
            Some(demanded) => basis == demanded,
            None => !rect.on_edge(m),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MixedSide {
    Left,
    Right,
}

impl MixedSide {
    fn for_layout(layout: RotationLayout) -> Self {
        match (layout.input_x_basis, layout.movement.to_array()) {
            (Basis::X, [-1, 0]) => Self::Right,
            (Basis::X, _) | (Basis::Z, [-1, 0]) => Self::Left,
            (Basis::Z, _) => Self::Right,
        }
    }

    fn is_right(self) -> bool {
        self == Self::Right
    }

    fn boundaries(
        self,
        input_x_basis: Basis,
        stage: RotationStage,
        geometry: RotationGeometry,
    ) -> BoundaryPolicy {
        let opposite = input_x_basis.flip();
        let extended_basis = match stage {
            RotationStage::Grown => input_x_basis,
            RotationStage::Rotated => opposite,
        };
        match self {
            MixedSide::Left => BoundaryPolicy {
                bottom: opposite,
                top: input_x_basis,
                left_bottom: input_x_basis,
                left_top: opposite,
                right_bottom: extended_basis,
                right_top: extended_basis,
                split_y: geometry.split_y,
            },
            MixedSide::Right => BoundaryPolicy {
                bottom: opposite,
                top: input_x_basis,
                left_bottom: extended_basis,
                left_top: extended_basis,
                right_bottom: input_x_basis,
                right_top: opposite,
                split_y: geometry.split_y,
            },
        }
    }

    fn schedule_rule(self, stage: RotationStage, geometry: RotationGeometry) -> ScheduleRule {
        match (self, stage) {
            (MixedSide::Left, RotationStage::Grown) => ScheduleRule::new(1, geometry.split_y),
            (MixedSide::Left, RotationStage::Rotated) => ScheduleRule::new(-1, geometry.split_y),
            (MixedSide::Right, RotationStage::Grown) => {
                ScheduleRule::new(-1, geometry.split_y + geometry.width)
            }
            (MixedSide::Right, RotationStage::Rotated) => {
                ScheduleRule::new(1, geometry.split_y - geometry.width)
            }
        }
    }

    fn cx_order(self, stage: RotationStage, schedule: CXSchedule) -> [Corner; 4] {
        match (self, stage, schedule) {
            (MixedSide::Left, RotationStage::Grown, CXSchedule::Horizontal)
            | (MixedSide::Right, RotationStage::Rotated, CXSchedule::Horizontal) => {
                [Corner::BR, Corner::BL, Corner::TR, Corner::TL]
            }
            (MixedSide::Left, RotationStage::Grown, CXSchedule::Vertical)
            | (MixedSide::Right, RotationStage::Rotated, CXSchedule::Vertical) => {
                [Corner::BR, Corner::TR, Corner::BL, Corner::TL]
            }
            (MixedSide::Left, RotationStage::Rotated, CXSchedule::Horizontal)
            | (MixedSide::Right, RotationStage::Grown, CXSchedule::Horizontal) => {
                [Corner::BL, Corner::BR, Corner::TL, Corner::TR]
            }
            (MixedSide::Left, RotationStage::Rotated, CXSchedule::Vertical)
            | (MixedSide::Right, RotationStage::Grown, CXSchedule::Vertical) => {
                [Corner::BL, Corner::TL, Corner::BR, Corner::TR]
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ScheduleRule {
    diagonal_sign: i32,
    line_intercept: i32,
}

impl ScheduleRule {
    const fn new(diagonal_sign: i32, line_intercept: i32) -> Self {
        Self {
            diagonal_sign,
            line_intercept,
        }
    }

    fn schedule(self, basis: Basis, m: IVec2, horizontal_hook_basis: Basis) -> CXSchedule {
        let standard = tile_cx_schedule(basis, horizontal_hook_basis);
        if self.uses_original_schedule(m) {
            standard
        } else {
            opposite_schedule(standard)
        }
    }

    fn uses_original_schedule(self, m: IVec2) -> bool {
        m.y / 2 <= self.diagonal_sign * (m.x / 2) + self.line_intercept
    }
}

fn opposite_schedule(schedule: CXSchedule) -> CXSchedule {
    match schedule {
        CXSchedule::Horizontal => CXSchedule::Vertical,
        CXSchedule::Vertical => CXSchedule::Horizontal,
    }
}

fn is_below_split_midline(measure_y: i32, split_y: i32) -> bool {
    measure_y < 2 * split_y + 1
}

fn make_patch(stage: RotationStage, layout: RotationLayout, offset: IVec2) -> Patch {
    let geometry = layout.geometry;
    let rect = Rect::new(geometry.width, geometry.height);
    let mixed_side = MixedSide::for_layout(layout);
    let boundaries = mixed_side.boundaries(layout.input_x_basis, stage, geometry);
    let schedule_rule = mixed_side.schedule_rule(stage, geometry);

    carve_patch(
        rect.data_qubits(),
        |m| {
            boundaries.keeps(
                m,
                rect,
                checkerboard_basis(layout.local_to_global(m, offset)),
            )
        },
        |m| layout.local_to_global(m, offset),
        |m, data| {
            let global_m = layout.local_to_global(m, offset);
            let basis = checkerboard_basis(global_m);
            let schedule = schedule_rule.schedule(basis, m, layout.input_x_basis.flip());
            let slots = mixed_side
                .cx_order(stage, schedule)
                .into_iter()
                .map(|corner| {
                    let dq = m + corner.to_ivec2();
                    data.contains(&dq)
                        .then_some(layout.local_to_global(dq, offset))
                });
            Tile::new(basis, global_m, slots)
        },
    )
}

fn make_standard_patch(layout: RotationLayout, offset: IVec2, y0: i32, x_basis: Basis) -> Patch {
    let width = layout.geometry.width;
    let rect = Rect::rooted(width, y0, width);
    let boundaries = EdgeBases::standard(x_basis);

    carve_patch(
        rect.data_qubits(),
        |m| {
            boundaries.keeps(
                m,
                rect,
                checkerboard_basis(layout.local_to_global(m, offset)),
            )
        },
        |m| layout.local_to_global(m, offset),
        |m, data| {
            let global_m = layout.local_to_global(m, offset);
            let basis = checkerboard_basis(global_m);
            let slots = tile_cx_schedule(basis, x_basis.flip())
                .compact()
                .into_iter()
                .map(|corner| {
                    let dq = m + corner.to_ivec2();
                    data.contains(&dq)
                        .then_some(layout.local_to_global(dq, offset))
                });
            Tile::new(basis, global_m, slots)
        },
    )
}

fn data_qubit(x: i32, y: i32) -> IVec2 {
    IVec2::new(2 * x + 1, 2 * y + 1)
}

fn local_data_region(
    layout: RotationLayout,
    offset: IVec2,
    xs: impl Iterator<Item = i32> + Clone,
    ys: impl Iterator<Item = i32>,
) -> FxSet<IVec2> {
    ys.flat_map(|y| {
        xs.clone()
            .map(move |x| layout.local_to_global(data_qubit(x, y), offset))
    })
    .collect()
}

fn selected_tile_measure_qubits_by_local_measure_coord(
    patch: &Patch,
    layout: RotationLayout,
    offset: IVec2,
    basis: Basis,
    keep_local_measure: impl Fn(IVec2) -> bool,
) -> Vec<IVec2> {
    patch
        .tiles()
        .iter()
        .filter(|tile| tile.basis() == basis)
        .filter(|tile| {
            let m = global_to_local(layout, offset, tile.measure_qubit());
            debug_assert_eq!(m.x % 2, 0);
            debug_assert_eq!(m.y % 2, 0);
            keep_local_measure(m / 2)
        })
        .map(Tile::measure_qubit)
        .collect()
}

fn global_to_local(layout: RotationLayout, offset: IVec2, p: IVec2) -> IVec2 {
    let q = p - offset;
    match layout.movement.to_array() {
        [0, 1] => q,
        [0, -1] => -q,
        [1, 0] => IVec2::new(q.y, q.x),
        [-1, 0] => IVec2::new(q.y, -q.x),
        _ => unreachable!("PatchRotationKind rejects non-cardinal movement"),
    }
}

fn make_patch_rotation_round_with_flow_boundaries(
    patch: &Patch,
    previous_patch: &Patch,
    next_patch: &Patch,
    init_data: Option<&FxMap<IVec2, Basis>>,
    meas_data: Option<&FxMap<IVec2, Basis>>,
    flow_boundaries: FlowBoundaryData<'_>,
) -> Result<Chunk, CompileError> {
    let round = emit_syndrome_round(
        patch,
        &sorted_batches(init_data, [Basis::Z, Basis::X]),
        &sorted_batches(meas_data, [Basis::Z, Basis::X]),
    )?;
    // Precompute the neighbouring patches' boundary keys once so each tile
    // below does a hash lookup instead of rescanning every neighbour tile.
    let previous_outputs =
        PatchBoundarySets::collapsed_boundaries(previous_patch, flow_boundaries.previous_meas_data);
    let next_inputs =
        PatchBoundarySets::collapsed_boundaries(next_patch, flow_boundaries.next_init_data);
    let flow_context = ExactRoundFlowContext {
        previous_outputs,
        next_inputs,
        init_data,
        meas_data,
        measurement_ids: &round.measurements,
    };
    let mut flows = Vec::new();
    for tile in patch.tiles() {
        flows.extend(tile_flows_with_exact_boundaries(tile, &flow_context)?);
    }

    Ok(Chunk {
        circuit: round.circuit,
        flows,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct FlowBoundaryData<'a> {
    previous_meas_data: Option<&'a FxMap<IVec2, Basis>>,
    next_init_data: Option<&'a FxMap<IVec2, Basis>>,
}

struct ExactRoundFlowContext<'a> {
    previous_outputs: PatchBoundarySets,
    next_inputs: PatchBoundarySets,
    init_data: Option<&'a FxMap<IVec2, Basis>>,
    meas_data: Option<&'a FxMap<IVec2, Basis>>,
    measurement_ids: &'a MeasurementIndex,
}

/// Per-basis boundary keys exposed by a neighbouring patch.
#[derive(Debug, Default)]
struct PatchBoundarySets {
    z: FxSet<PauliMap>,
    x: FxSet<PauliMap>,
}

impl PatchBoundarySets {
    /// Boundary keys the patch's tiles carry once `collapsed_data` is
    /// measured/initialized away: the outputs a previous round still presents,
    /// or equivalently the inputs a next round expects — the continuing
    /// supports of the shared collapse rule either way.
    fn collapsed_boundaries(patch: &Patch, collapsed_data: Option<&FxMap<IVec2, Basis>>) -> Self {
        let mut sets = Self::default();
        for tile in patch.tiles() {
            let basis = tile.basis();
            if let Some((key, _)) = collapse_matching_data_boundary(
                &tile.active_data_qubits(),
                basis,
                Pauli::from(basis),
                collapsed_data,
            ) {
                sets.for_basis_mut(basis).insert(key);
            }
        }
        sets
    }

    fn contains(&self, basis: Basis, key: &PauliMap) -> bool {
        match basis {
            Basis::Z => self.z.contains(key),
            Basis::X => self.x.contains(key),
        }
    }

    fn for_basis_mut(&mut self, basis: Basis) -> &mut FxSet<PauliMap> {
        match basis {
            Basis::Z => &mut self.z,
            Basis::X => &mut self.x,
        }
    }
}

fn tile_flows_with_exact_boundaries(
    tile: &Tile,
    context: &ExactRoundFlowContext<'_>,
) -> Result<Vec<Flow>, CompileError> {
    let mut flows = Vec::with_capacity(2);
    let active_data = tile.active_data_qubits();
    let basis = tile.basis();
    let pauli = Pauli::from(basis);
    let own_measurement = measurement_id(context.measurement_ids, tile.measure_qubit())?;

    if let Some((start, _)) =
        collapse_matching_data_boundary(&active_data, basis, pauli, context.init_data)
    {
        let compared_to_previous = context.previous_outputs.contains(basis, &start);
        let initialized = start.is_empty();
        if initialized || compared_to_previous {
            flows.push(
                Flow::new(start, PauliMap::empty())
                    .with_measurements([own_measurement])
                    .with_center(tile.measure_qubit()),
            );
        }
    }

    if let Some((end, mut measured_data)) =
        collapse_matching_data_boundary(&active_data, basis, pauli, context.meas_data)
    {
        let continues_to_next = context.next_inputs.contains(basis, &end);
        let finalized = end.is_empty();
        if finalized || continues_to_next {
            measured_data.sort_unstable_by_key(|q| (q.y, q.x));
            let mut measurements = FlowMeasurements::with_capacity(measured_data.len() + 1);
            for q in measured_data {
                measurements.push(measurement_id(context.measurement_ids, q)?);
            }
            measurements.push(own_measurement);
            flows.push(
                Flow {
                    measurements,
                    ..Flow::new(PauliMap::empty(), end)
                }
                .with_center(tile.measure_qubit()),
            );
        }
    }

    Ok(flows)
}

fn measurement_id(measurement_ids: &MeasurementIndex, q: IVec2) -> Result<u32, CompileError> {
    measurement_ids
        .get(q)
        .ok_or_else(|| CompileError::PatchRotationConstructionFailed {
            reason: format!("flow references unmeasured qubit {q:?}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_rotation_endpoint_schedules_are_compact() {
        let kind = PatchRotationKind::new(Basis::Z, IVec2::X).expect("cardinal movement");
        let [input, _, _, output] = paper_rotation_patches(kind, 3);
        assert!(
            [input, output]
                .iter()
                .flat_map(Patch::tiles)
                .all(|tile| tile.data_slots().len() == 4)
        );
    }

    #[test]
    fn patch_rotation_d3_circuit_snapshot() {
        // Pins the six-stage rotation circuit byte-for-byte so round-emitter
        // refactors stay behavior-identical.
        let kind = PatchRotationKind::new(Basis::X, IVec2::new(1, 0)).expect("cardinal movement");
        let template =
            compile_patch_rotation(kind, Connectivity::ISOLATED, 3).expect("rotation compiles");
        insta::assert_snapshot!(
            "patch_rotation_d3_circuit",
            template.program_template.circuit.to_string()
        );
    }
}
