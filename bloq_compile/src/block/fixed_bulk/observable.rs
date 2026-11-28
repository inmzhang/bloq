use bloq_circuit::{ChunkOrLoop, PauliMap};
use bloq_graph::{Basis, CubeKind, Direction, Pauli, UDirection};
use glam::IVec2;
use itertools::Itertools;

use crate::block::gateway::{
    ChunkMeasurements, GatewayEntry, LocalStabilizer, ObservableGateway, gateway_key_connectivity,
};
use crate::block::measurements::{MeasurementIndex, stage_chunk};
use crate::block::patch::Patch;
use crate::signature::Connectivity;

pub(super) const INIT_CHUNK: usize = 0;
pub(super) const MEAS_CHUNK: usize = 2;

fn is_plus_direction(dir: Direction) -> bool {
    matches!(dir, Direction::XPLUS | Direction::YPLUS | Direction::ZPLUS)
}

/// Extract measure qubit positions and bases from the patch tiles.
fn extract_measure_qubits(patch: &Patch) -> Vec<(IVec2, Basis)> {
    patch
        .tiles()
        .iter()
        .map(|t| (t.measure_qubit(), t.basis()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SelectedChunkMeasurements {
    pub(crate) chunk_index: usize,
    pub(crate) qubits: Vec<IVec2>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct SelectedGateway(crate::FxMap<LocalStabilizer, Vec<SelectedChunkMeasurements>>);

impl SelectedGateway {
    pub(crate) fn insert(
        &mut self,
        key: LocalStabilizer,
        value: Vec<SelectedChunkMeasurements>,
    ) -> Option<Vec<SelectedChunkMeasurements>> {
        self.0.insert(key, value)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(test)]
    fn contains_key(&self, key: &LocalStabilizer) -> bool {
        self.0.contains_key(key)
    }
}

#[cfg(test)]
impl std::ops::Index<&LocalStabilizer> for SelectedGateway {
    type Output = Vec<SelectedChunkMeasurements>;

    fn index(&self, key: &LocalStabilizer) -> &Self::Output {
        &self.0[key]
    }
}

/// Wrap qubit positions at the given chunk index.
fn chunk_refs(chunk_index: usize, qubits: &[IVec2]) -> Vec<SelectedChunkMeasurements> {
    if qubits.is_empty() {
        return vec![];
    }
    vec![SelectedChunkMeasurements {
        chunk_index,
        qubits: qubits.to_vec(),
    }]
}

/// All measure qubits for the stabilizer basis (for "full bottom stabilizers").
fn full_bottom_stabilizer_qubits(measure_qubits: &[(IVec2, Basis)], basis: Basis) -> Vec<IVec2> {
    measure_qubits
        .iter()
        .filter(|(_, b)| *b == basis)
        .map(|&(q, _)| q)
        .collect()
}

/// Init-chunk selection of the full bottom stabilizers in `basis`.
fn full_bottom_init(
    measure_qubits: &[(IVec2, Basis)],
    basis: Basis,
) -> Vec<SelectedChunkMeasurements> {
    chunk_refs(
        INIT_CHUNK,
        &full_bottom_stabilizer_qubits(measure_qubits, basis),
    )
}

/// Measure qubits on one side of the patch center
fn half_bottom_stabilizer_qubits(
    measure_qubits: &[(IVec2, Basis)],
    basis: Basis,
    towards: Direction,
    distance: u32,
) -> Vec<IVec2> {
    let center = distance as i32;
    measure_qubits
        .iter()
        .filter(|(pos, b)| {
            if *b != basis {
                return false;
            }
            match towards {
                Direction::XPLUS => pos.x >= center,
                Direction::XMINUS => pos.x <= center,
                Direction::YPLUS => pos.y >= center,
                Direction::YMINUS => pos.y <= center,
                _ => false,
            }
        })
        .map(|&(q, _)| q)
        .collect()
}

/// Generate data qubit positions along a line through the patch center.
///
/// `horizontal = true` → line at y = center (along X axis).
/// `horizontal = false` → line at x = center (along Y axis).
pub(crate) fn line_qubits(distance: u32, horizontal: bool) -> Vec<IVec2> {
    let d = distance as i32;
    let center = d;
    (0..d)
        .map(|i| {
            let v = 1 + 2 * i;
            if horizontal {
                IVec2::new(v, center)
            } else {
                IVec2::new(center, v)
            }
        })
        .collect()
}

/// Logical middle-line operator: the `line_basis` line through the patch
/// centre carrying `pauli` on every qubit.
///
/// Names the shared orientation convention: a logical line runs
/// **horizontally exactly when its basis differs from the patch's
/// top/boundary basis**. Y, port, realignment, selective, and walking
/// operators all derive their supports through this rule.
pub(crate) fn logical_line_operator(
    distance: u32,
    line_basis: Basis,
    top_basis: Basis,
    pauli: Pauli,
) -> PauliMap {
    line_qubits(distance, line_basis != top_basis)
        .into_iter()
        .map(|q| (q, pauli))
        .collect()
}

/// Data qubits along the middle line of the surface code patch.
///
/// Orientation is determined by the cube kind's normal direction:
/// - Normal along Y → horizontal line at y = d
/// - Normal along X → vertical line at x = d
fn middle_line_qubits(distance: u32, kind: CubeKind) -> Vec<IVec2> {
    line_qubits(distance, kind.normal_direction() == UDirection::Y)
}

/// Top middle line: orientation from cube kind (non-spatial) or spatial arms.
fn cube_top_middle_line_qubits(
    distance: u32,
    kind: CubeKind,
    spatial_arms: &[Direction],
) -> Vec<IVec2> {
    if !kind.is_spatial() {
        middle_line_qubits(distance, kind)
    } else {
        let x_axis = spatial_arms
            .iter()
            .any(|d| d.as_udirection() == UDirection::X);
        let y_axis = spatial_arms
            .iter()
            .any(|d| d.as_udirection() == UDirection::Y);
        line_qubits(distance, x_axis && !y_axis)
    }
}

/// Single boundary data qubit at a spatial pipe interface.
fn pipe_boundary_readout_qubit(d: i32, dir: Direction) -> IVec2 {
    match dir {
        Direction::XPLUS => IVec2::new(2 * d + 1, d),
        Direction::XMINUS => IVec2::new(-1, d),
        Direction::YPLUS => IVec2::new(d, 2 * d + 1),
        Direction::YMINUS => IVec2::new(d, -1),
        _ => unreachable!("only spatial pipes have top readout qubits"),
    }
}

/// Boundary data qubits at each owned spatial pipe interface.
///
/// Shared pipe-boundary readout qubits belong to the cube at the smaller block
/// position along the pipe, i.e. the cube that sees the connection through a
/// positive spatial direction. This prevents two cubes on the same pipe from
/// generating distinct semantic measurements that later collapse to the same
/// physical register.
fn pipe_readout_qubits(distance: u32, arms: &[Direction]) -> Vec<IVec2> {
    let d = distance as i32;
    arms.iter()
        .filter(|dir| dir.is_spatial() && is_plus_direction(**dir))
        .map(|&dir| pipe_boundary_readout_qubit(d, dir))
        .collect()
}

/// Half of the middle line towards a specific spatial direction.
///
/// For XPLUS/XMINUS: data qubits on the horizontal line (y = center),
///   on the half beyond/before center.
/// For YPLUS/YMINUS: data qubits on the vertical line (x = center),
///   on the half beyond/before center.
fn half_middle_line_qubits(distance: u32, towards: Direction) -> Vec<IVec2> {
    let d = distance as i32;
    let center = d;

    // (start, end, horizontal): half-line range and orientation
    let (start, end, horizontal) = match towards {
        Direction::XPLUS => (center + 2, 2 * d, true),
        Direction::XMINUS => (1, center, true),
        Direction::YPLUS => (center + 2, 2 * d, false),
        Direction::YMINUS => (1, center, false),
        _ => return vec![],
    };

    (0..)
        .map(|i| start + 2 * i)
        .take_while(|&v| v < end)
        .map(|v| {
            if horizontal {
                IVec2::new(v, center)
            } else {
                IVec2::new(center, v)
            }
        })
        .collect()
}

/// Whether the center qubit should be included in an L-shape readout.
///
/// Determined by the arm pair's sign product and the observable basis:
/// - Arms with different signs (e.g. XMINUS+YPLUS): include if basis == Z
/// - Arms with same signs (e.g. XPLUS+YPLUS): include if basis == X
pub(super) fn include_l_shape_corner(arm1: Direction, arm2: Direction, basis: Basis) -> bool {
    let sign = |dir: Direction| -> i32 {
        match dir {
            Direction::XPLUS | Direction::YPLUS => 1,
            Direction::XMINUS | Direction::YMINUS => -1,
            _ => 0,
        }
    };
    let product = sign(arm1) * sign(arm2);
    match product {
        -1 => basis == Basis::Z,
        1 => basis == Basis::X,
        _ => false,
    }
}

/// L-shape readout qubits: two half-lines meeting at the center, with
/// conditional corner inclusion based on arm geometry and basis.
fn l_shape_readout_qubits(
    distance: u32,
    arm1: Direction,
    arm2: Direction,
    basis: Basis,
) -> Vec<IVec2> {
    let mut qubits = half_middle_line_qubits(distance, arm1);
    qubits.extend(half_middle_line_qubits(distance, arm2));

    if include_l_shape_corner(arm1, arm2, basis) {
        let center = distance as i32;
        qubits.push(IVec2::new(center, center));
    }

    qubits
}

/// Measurements for a single gateway key.
///
/// The cube collapses data only in its **temporal** basis `T = kind.z()` — it
/// initializes in `T` at the `−Z` face and reads out in `T` at the `+Z` face —
/// so which face anchors the observable's correlation surface is fixed by the
/// key basis, and the arms say whether the surface actually ends there:
///
/// - **`b == T`** — the surface is read destructively on the top data layer,
///   unless a `+Z` arm carries it onward. Either way the spatial seams
///   contribute their boundary readouts.
/// - **`b != T`** — the surface cannot be read at the top; it is anchored at
///   the `−Z` face, where the init round's stabilizer records reconstruct it.
///   That record is needed only where the surface actually ends or turns here:
///   a purely temporal `b != T` crossing enters through a `−Z` arm and leaves
///   through `+Z` untouched, so it needs nothing.
fn build_measurements_from_key(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    key: LocalStabilizer,
    measure_qubits: &[(IVec2, Basis)],
) -> Vec<SelectedChunkMeasurements> {
    let key_basis: Basis = key
        .center_basis()
        .try_into()
        .expect("key basis must be X or Z");
    let spatial_arms: Vec<Direction> = key.arm_dirs().filter(|dir| dir.is_spatial()).collect();
    let has_temporal_arm = key.arm_dirs().any(|dir| !dir.is_spatial());

    if key_basis != kind.z() {
        return match spatial_arms.as_slice() {
            // Straight temporal crossing: the surface arrives from below rather
            // than being created here, and turns nowhere, so nothing ties it.
            [] if key.has_arm(Direction::ZMINUS) => Vec::new(),
            // One spatial arm plus a temporal one: the surface turns a corner
            // in the space-time plane and covers only the half of the cube
            // facing that arm.
            [arm] if has_temporal_arm => chunk_refs(
                INIT_CHUNK,
                &half_bottom_stabilizer_qubits(measure_qubits, key_basis, *arm, distance),
            ),
            _ => full_bottom_init(measure_qubits, key_basis),
        };
    }

    let mut qubits = if key.has_arm(Direction::ZPLUS) {
        Vec::new()
    } else {
        top_readout_line(kind, distance, key_basis, &spatial_arms)
    };
    // A ceded (spatial Hadamard) arm has no boundary readout: the wall consumed
    // the data column a plain merge would share, so there is nothing on it to
    // read — and nothing measures the qubit `pipe_boundary_readout_qubit` names.
    //
    // It still *bends* the line above, though, exactly as an ordinary spatial
    // arm does: the bend says where the surface leaves the cube, and a ceded
    // arm is a genuine exit — the crossing continues into the wall's extended
    // stabilizers, whose halves sit on this very column. Dropping the bend
    // would leave a readout line that never reaches the seam.
    let readable_arms: Vec<Direction> = spatial_arms
        .iter()
        .copied()
        .filter(|&dir| connectivity.has_open_edge(dir))
        .collect();
    qubits.extend(pipe_readout_qubits(distance, &readable_arms));
    chunk_refs(MEAS_CHUNK, &qubits)
}

/// The `+Z`-face readout support: the middle line, bent into an L where two
/// perpendicular spatial arms turn it, or the XOR of both L-shapes when all
/// four spatial arms are present (their shared centre cancels).
fn top_readout_line(
    kind: CubeKind,
    distance: u32,
    basis: Basis,
    spatial_arms: &[Direction],
) -> Vec<IVec2> {
    match spatial_arms {
        [a1, a2] if *a1 != a2.negate() => l_shape_readout_qubits(distance, *a1, *a2, basis),
        [_, _, _, _] => {
            let l1: crate::FxSet<IVec2> =
                l_shape_readout_qubits(distance, Direction::XPLUS, Direction::YPLUS, basis)
                    .into_iter()
                    .collect();
            let l2: crate::FxSet<IVec2> =
                l_shape_readout_qubits(distance, Direction::XMINUS, Direction::YMINUS, basis)
                    .into_iter()
                    .collect();
            l1.symmetric_difference(&l2).copied().collect()
        }
        _ => cube_top_middle_line_qubits(distance, kind, spatial_arms),
    }
}

/// Build the observable gateway for a cube block from a patch.
///
/// Enumerates all valid (basis, pipe_subset) keys and computes measurements
/// for each using the case analysis in `build_measurements_from_key`.
pub(super) fn build_gateway_from_patch(
    kind: CubeKind,
    connectivity: Connectivity,
    distance: u32,
    patch: &Patch,
) -> SelectedGateway {
    let measure_qubits = extract_measure_qubits(patch);
    let mut gateway = SelectedGateway::default();

    let normal_basis = kind.normal_basis();
    let complement_basis = normal_basis.flip();

    // Weight-0 entry: block's own observable (always complement basis)
    if connectivity.is_isolated() {
        let key0 = LocalStabilizer::new(Pauli::from(complement_basis), Connectivity::ISOLATED);
        let entry0 =
            build_measurements_from_key(kind, connectivity, distance, key0, &measure_qubits);
        gateway.insert(key0, entry0);
        return gateway;
    }

    // Enumerate all non-empty subsets of connected pipes
    let pipes: Vec<Direction> = connectivity.pipe_dirs().collect();
    let n = pipes.len();

    // Normal basis: even number of arms can hold the logical operators
    for m in [2, 4] {
        if m > n {
            break;
        }
        for subset_dirs in pipes.iter().combinations(m) {
            let subset_conn = gateway_key_connectivity(subset_dirs.iter().map(|&&dir| dir));
            let key = LocalStabilizer::new(Pauli::from(normal_basis), subset_conn);
            let entry =
                build_measurements_from_key(kind, connectivity, distance, key, &measure_qubits);
            gateway.insert(key, entry);
        }
    }
    // Complement basis: all arms should hold the logical operators
    let key = LocalStabilizer::new(
        Pauli::from(complement_basis),
        gateway_key_connectivity(connectivity.pipe_dirs()),
    );
    let entry = build_measurements_from_key(kind, connectivity, distance, key, &measure_qubits);
    gateway.insert(key, entry);

    gateway
}

/// Logical boundary operators on a cube's temporal faces.
///
/// The operator rides a temporal face exactly when the key's logical has an **arm**
/// there: `operator_in` iff the `ZMINUS` arm, `operator_out` iff the `ZPLUS` arm,
/// with basis from `arm_basis`. The arm gates presence, which is load-bearing: a
/// bare cube↔cube seam is a measure+reset (no arm), and a Pauli line laid across it
/// back-propagates into the reset and drops decoder distance. An arm is present only
/// where a coherent pipe (a port or transform) actually threads the seam, so the
/// operator only appears where it is sound. (A temporal Hadamard pipe is fine:
/// its arm already carries the flipped basis.)
pub(crate) fn cube_operator_faces(
    kind: CubeKind,
    distance: u32,
    key: LocalStabilizer,
) -> (PauliMap, PauliMap) {
    let face_operator = |dir: Direction| -> PauliMap {
        if !key.has_arm(dir) {
            return PauliMap::empty();
        }
        let pauli = key.arm_basis(dir);
        cube_logical_line(kind, distance, pauli)
            .into_iter()
            .map(|qubit| (qubit, pauli))
            .collect()
    };
    (
        face_operator(Direction::ZMINUS),
        face_operator(Direction::ZPLUS),
    )
}

fn cube_logical_line(kind: CubeKind, distance: u32, basis: Pauli) -> Vec<IVec2> {
    let complement = Pauli::from(kind.normal_basis().flip());
    let complement_horizontal = kind.normal_direction() == UDirection::Y;
    let horizontal = if basis == complement {
        complement_horizontal
    } else {
        !complement_horizontal
    };
    line_qubits(distance, horizontal)
}

pub(crate) fn resolve_gateway_measurements(
    selected: SelectedGateway,
    chunks: &[ChunkOrLoop],
    operator_for: impl Fn(LocalStabilizer) -> (PauliMap, PauliMap),
) -> ObservableGateway {
    resolve_gateway_measurements_at(
        selected,
        chunks,
        MEAS_CHUNK,
        crate::FxMap::default(),
        operator_for,
    )
}

/// Resolve a cube gateway when the measurement round is not the conventional
/// third stage (extended layers insert alternating bulk stages before it).
///
/// `known_ids` seeds the per-stage index cache with whatever the chunk builder
/// already recorded at emission time; stages absent from it are recovered by
/// walking their circuit.
pub(crate) fn resolve_gateway_measurements_at(
    selected: SelectedGateway,
    chunks: &[ChunkOrLoop],
    meas_chunk: usize,
    known_ids: crate::FxMap<usize, MeasurementIndex>,
    operator_for: impl Fn(LocalStabilizer) -> (PauliMap, PauliMap),
) -> ObservableGateway {
    let mut gateway = ObservableGateway::new();
    let mut measurement_ids_by_chunk = known_ids;
    for (key, selected_measurements) in selected.0 {
        let mut measurements = Vec::with_capacity(selected_measurements.len());
        for chunk_measurements in selected_measurements {
            let chunk_index = if chunk_measurements.chunk_index == MEAS_CHUNK {
                meas_chunk
            } else {
                chunk_measurements.chunk_index
            };
            let measurement_ids =
                measurement_ids_by_chunk
                    .entry(chunk_index)
                    .or_insert_with(|| {
                        let chunk = stage_chunk(chunks, chunk_index);
                        MeasurementIndex::from_circuit(&chunk.circuit)
                    });
            let ids = chunk_measurements
                .qubits
                .into_iter()
                .map(|q| measurement_ids.expect_measurement(q))
                .collect();
            measurements.push(ChunkMeasurements {
                chunk_index,
                measurements: ids,
            });
        }
        let (operator_in, operator_out) = operator_for(key);
        gateway.insert(
            key,
            GatewayEntry {
                measurements,
                operator_in,
                operator_out,
            },
        );
    }
    gateway
}

#[cfg(test)]
mod tests {
    use bloq_graph::CubeKind;
    use rstest::rstest;

    use super::*;
    use crate::block::fixed_bulk::compile_cube;
    use crate::signature::LayerSchedule;

    fn expected_gateway_size(n_pipes: usize) -> usize {
        match n_pipes {
            0 | 1 => 1, // weight-0 or complement
            2 => 2,     // C(2,2)=1 normal + 1 complement
            3 => 4,     // C(3,2)=3 normal + 1 complement
            4 => 8,     // C(4,2)=6 + C(4,4)=1 normal + 1 complement
            _ => unreachable!(),
        }
    }

    fn key(basis: Basis, conn: Connectivity) -> LocalStabilizer {
        LocalStabilizer::new(Pauli::from(basis), conn)
    }

    /// A cube may carry up to six pipes; the gateway rule has no weight ceiling.
    #[rstest]
    fn derived_rule_handles_more_than_four_pipes(
        #[values(CubeKind::ZXZ, CubeKind::ZZX)] kind: CubeKind,
    ) {
        use crate::block::fixed_bulk::regular::build_regular_patch;
        use crate::block::fixed_bulk::spatial::build_spatial_patch;

        let connectivity = Direction::iter().fold(Connectivity::ISOLATED, Connectivity::with_pipe);
        let (patch, _, _) = if kind.is_spatial() {
            build_spatial_patch(kind, 3, connectivity, LayerSchedule::Padded)
        } else {
            build_regular_patch(kind, 3, connectivity, LayerSchedule::Compact)
        };
        let gateway = build_gateway_from_patch(kind, connectivity, 3, &patch);
        assert!(
            gateway.0.keys().any(|key| key.weight() > 4),
            "sweep reaches weight > 4"
        );
    }

    /// A ceded (spatial Hadamard) arm bends the `+Z` readout line exactly like
    /// an ordinary spatial arm — it is a genuine exit, and the crossing
    /// continues into the wall's extended stabilizers, whose halves sit on the
    /// cube's own seam-adjacent data column. What it does *not* contribute is a
    /// `pipe_boundary_readout_qubit`: the wall consumed that column.
    ///
    /// This is the CZ gallery's right endpoint. Dropping the bend leaves a
    /// straight middle line that never reaches the seam, which CZ's observables
    /// catch at `d = 5` (at `d = 3` the two selections happen to be
    /// equivalent), so the rule is pinned here where it is cheap to see.
    #[rstest]
    fn ceded_arm_bends_the_readout_line_without_a_seam_readout(#[values(3, 5)] distance: u32) {
        let d = distance as i32;
        let cases = [
            (
                Direction::XMINUS,
                Direction::YPLUS,
                // The `-X` half-line runs inward from the seam column, the
                // corner joins (opposite signs, `Z` basis), and the `+Y` half
                // ends on the arm's own shared readout.
                (1..d)
                    .step_by(2)
                    .map(|x| IVec2::new(x, d))
                    .chain((d..=2 * d + 1).step_by(2).map(|y| IVec2::new(d, y)))
                    .collect::<Vec<_>>(),
            ),
            (
                Direction::XPLUS,
                Direction::YPLUS,
                // Same signs, so the corner drops; `+X` is ceded, so the line
                // stops on the cube's own last column instead of reaching
                // `(2d + 1, d)`.
                (d + 2..2 * d)
                    .step_by(2)
                    .map(|x| IVec2::new(x, d))
                    .chain((d + 2..=2 * d + 1).step_by(2).map(|y| IVec2::new(d, y)))
                    .collect::<Vec<_>>(),
            ),
        ];

        for (ceded, arm, expected) in cases {
            let connectivity = Connectivity::ISOLATED
                .with_hadamard(ceded)
                .with_pipe(arm)
                .with_pipe(arm.negate());
            let (patch, _, _) = crate::block::fixed_bulk::spatial::build_spatial_patch(
                CubeKind::XXZ,
                distance,
                connectivity,
                LayerSchedule::Extended,
            );
            let key = LocalStabilizer::new(
                Pauli::from(CubeKind::XXZ.z()),
                gateway_key_connectivity([ceded, arm]),
            );
            let selected = build_measurements_from_key(
                CubeKind::XXZ,
                connectivity,
                distance,
                key,
                &extract_measure_qubits(&patch),
            );

            assert_eq!(selected.len(), 1, "d={distance} {ceded:?}: one chunk");
            assert_eq!(selected[0].chunk_index, MEAS_CHUNK);
            let mut qubits = selected[0].qubits.clone();
            qubits.sort_unstable_by_key(|qubit| (qubit.y, qubit.x));
            let mut expected = expected;
            expected.sort_unstable_by_key(|qubit| (qubit.y, qubit.x));
            assert_eq!(qubits, expected, "d={distance} ceded={ceded:?} arm={arm:?}");
        }
    }

    fn cube_observable_gateway(
        kind: CubeKind,
        connectivity: Connectivity,
        distance: u32,
    ) -> SelectedGateway {
        use crate::block::fixed_bulk::regular::build_regular_patch;
        use crate::block::fixed_bulk::spatial::build_spatial_patch;
        let (patch, _, _) = if kind.is_spatial() {
            build_spatial_patch(kind, distance, connectivity, LayerSchedule::Padded)
        } else {
            build_regular_patch(kind, distance, connectivity, LayerSchedule::Compact)
        };
        build_gateway_from_patch(kind, connectivity, distance, &patch)
    }

    /// Total number of measurement refs across all chunks in a gateway entry.
    fn total_measurements(entry: &[SelectedChunkMeasurements]) -> usize {
        entry.iter().map(|cm| cm.qubits.len()).sum()
    }

    /// Check that a specific qubit position appears in the entry's measurements.
    fn has_qubit(entry: &[SelectedChunkMeasurements], qubit: IVec2) -> bool {
        entry.iter().any(|cm| cm.qubits.contains(&qubit))
    }

    /// Expected number of complement-basis measure qubits for spatial cubes (XXZ, ZZX)
    /// in isolation. Formula: (d² + 2d - 3) / 2.
    fn expected_spatial_complement_count(d: u32) -> usize {
        ((d * d + 2 * d - 3) / 2) as usize
    }

    /// Expected number of normal-basis half-bottom measure qubits for non-spatial
    /// cubes towards one spatial direction. Formula: (d + 1)² / 4.
    fn expected_nonspatial_normal_half_bottom(d: u32) -> usize {
        (((d + 1) * (d + 1)) / 4) as usize
    }

    #[rstest]
    fn test_nonspatial_isolated(
        #[values(CubeKind::XZX, CubeKind::XZZ, CubeKind::ZXX, CubeKind::ZXZ)] kind: CubeKind,
        #[values(3, 5, 7)] distance: u32,
    ) {
        let gw = cube_observable_gateway(kind, Connectivity::ISOLATED, distance);
        assert_eq!(gw.len(), expected_gateway_size(0));

        // Complement basis (weight-0): top middle line at MEAS_CHUNK
        let kn = key(kind.normal_basis().flip(), Connectivity::ISOLATED);
        assert!(gw.contains_key(&kn));
        let entry = &gw[&kn];
        assert_eq!(entry.len(), 1);
        assert_eq!(entry[0].chunk_index, MEAS_CHUNK);
        assert_eq!(entry[0].qubits.len(), distance as usize);
    }

    #[rstest]
    fn test_spatial_isolated(
        #[values(CubeKind::XXZ, CubeKind::ZZX)] kind: CubeKind,
        #[values(3, 5, 7)] distance: u32,
    ) {
        let gw = cube_observable_gateway(kind, Connectivity::ISOLATED, distance);
        assert_eq!(gw.len(), expected_gateway_size(0));

        // Complement basis (weight-0): full bottom stabilizers at INIT_CHUNK
        let kn = key(kind.normal_basis().flip(), Connectivity::ISOLATED);
        assert!(gw.contains_key(&kn));
        let entry = &gw[&kn];
        assert_eq!(entry.len(), 1);
        assert_eq!(entry[0].chunk_index, INIT_CHUNK);
        // Full bottom stabilizers: (d² + 2d - 3) / 2
        assert_eq!(
            entry[0].qubits.len(),
            expected_spatial_complement_count(distance),
        );

        // All measurement qubits should be at even coordinates (measure qubit positions)
        for meas in &entry[0].qubits {
            assert_eq!(meas.x % 2, 0);
            assert_eq!(meas.y % 2, 0);
        }
    }

    #[rstest]
    fn test_single_spatial_pipe_nonspatial(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: normal=X, complement=Z, temporal=Z
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XPLUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(1));

        // Complement basis (Z), all arms ({XPLUS}): weight-1 spatial
        // Non-spatial cube → top middle line + pipe readout
        let kc = key(Basis::Z, conn);
        assert!(gw.contains_key(&kc));
        assert_eq!(gw[&kc].len(), 1); // merged into MEAS_CHUNK
        assert_eq!(gw[&kc][0].chunk_index, MEAS_CHUNK);
        // d middle line + 1 pipe readout = d + 1
        assert_eq!(gw[&kc][0].qubits.len(), distance as usize + 1,);

        let d = distance as i32;
        assert!(has_qubit(&gw[&kc], IVec2::new(2 * d + 1, d)));
        assert!(has_qubit(&gw[&kc], IVec2::new(1, d)));
    }

    #[rstest]
    fn test_single_negative_spatial_pipe_nonspatial_omits_boundary_readout(
        #[values(3, 5, 7)] distance: u32,
    ) {
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XMINUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(1));

        let kc = key(Basis::Z, conn);
        assert!(gw.contains_key(&kc));
        assert_eq!(gw[&kc].len(), 1);
        assert_eq!(gw[&kc][0].chunk_index, MEAS_CHUNK);
        assert_eq!(gw[&kc][0].qubits.len(), distance as usize);

        let d = distance as i32;
        assert!(!has_qubit(&gw[&kc], IVec2::new(-1, d)));
        assert!(has_qubit(&gw[&kc], IVec2::new(1, d)));
    }

    #[rstest]
    fn test_single_spatial_pipe_spatial(#[values(3, 5, 7)] distance: u32) {
        // ZZX: spatial, normal=X, complement=Z
        let conn = Connectivity::ISOLATED.with_pipe(Direction::XPLUS);
        let gw = cube_observable_gateway(CubeKind::ZZX, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(1));

        // Complement basis (Z), spatial cube → full bottom stabilizers (init chunk)
        let kc = key(Basis::Z, conn);
        assert!(gw.contains_key(&kc));
        assert_eq!(gw[&kc][0].chunk_index, INIT_CHUNK);
        // Full bottom stabilizers with one spatial pipe: (d² + 2d - 1) / 2
        assert_eq!(
            gw[&kc][0].qubits.len(),
            ((distance * distance + 2 * distance - 1) / 2) as usize,
        );
    }

    #[rstest]
    fn test_single_temporal_zminus(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: non-spatial, normal=X, complement=Z
        let conn = Connectivity::ISOLATED.with_pipe(Direction::ZMINUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(1));

        // Complement basis (Z), ZMINUS: weight-1 temporal ZMINUS → top middle line
        let kc = key(Basis::Z, conn);
        assert!(gw.contains_key(&kc));
        assert_eq!(gw[&kc][0].chunk_index, MEAS_CHUNK);
        assert_eq!(gw[&kc][0].qubits.len(), distance as usize,);
        let d = distance as i32;
        assert!(has_qubit(&gw[&kc], IVec2::new(1, d)));
        assert!(has_qubit(&gw[&kc], IVec2::new(2 * d - 1, d)));
    }

    #[rstest]
    fn test_single_temporal_zplus(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: non-spatial, normal=X, complement=Z
        let conn = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(1));

        // Complement basis (Z), ZPLUS: none (empty passthrough)
        let kc = key(Basis::Z, conn);
        assert!(gw.contains_key(&kc));
        assert!(gw[&kc].is_empty());
    }

    #[rstest]
    fn test_temporal_hadamard_keys_are_plain_on_both_faces(#[values(3, 5, 7)] distance: u32) {
        // Gateway keys never carry Hadamard flags (`gateway_key_connectivity`):
        // the realignment pipe node owns the X↔Z flip, on either face.
        for dir in [Direction::ZMINUS, Direction::ZPLUS] {
            let gw = cube_observable_gateway(
                CubeKind::ZXZ,
                Connectivity::ISOLATED.with_hadamard(dir),
                distance,
            );
            let plain = key(Basis::Z, Connectivity::ISOLATED.with_pipe(dir));
            let flagged = key(Basis::Z, Connectivity::ISOLATED.with_hadamard(dir));
            assert!(gw.contains_key(&plain));
            assert!(!gw.contains_key(&flagged));
        }
    }

    #[rstest]
    fn test_opposite_x_pair(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: normal=X, complement=Z, temporal=Z
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::XMINUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(2));

        // Weight-2, temporal_basis(Z): middle line + pipe readouts
        let k2t = key(Basis::Z, conn);
        assert!(gw.contains_key(&k2t));
        assert_eq!(gw[&k2t].len(), 1); // merged MEAS_CHUNK
        // d middle line + owned positive-direction pipe readout = d + 1
        assert_eq!(gw[&k2t][0].qubits.len(), distance as usize + 1);
        let d = distance as i32;
        assert!(has_qubit(&gw[&k2t], IVec2::new(2 * d + 1, d))); // XPLUS
        assert!(!has_qubit(&gw[&k2t], IVec2::new(-1, d))); // XMINUS owned by neighbor
        assert!(has_qubit(&gw[&k2t], IVec2::new(1, d)));
        assert!(has_qubit(&gw[&k2t], IVec2::new(2 * d - 1, d)));

        // Weight-2, normal_basis(X): full bottom stabilizers (normal basis count)
        let k2n = key(Basis::X, conn);
        assert!(gw.contains_key(&k2n));
        assert_eq!(gw[&k2n][0].chunk_index, INIT_CHUNK);
        assert_eq!(
            gw[&k2n][0].qubits.len(),
            (((distance + 1) * (distance + 1)) / 2) as usize
        );
    }

    #[rstest]
    fn test_opposite_temporal_pair(#[values(3, 5, 7)] distance: u32) {
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::ZMINUS)
            .with_pipe(Direction::ZPLUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);

        // Weight-2, both temporal: none (empty passthrough)
        let k2 = key(Basis::X, conn);
        assert!(gw.contains_key(&k2));
        assert!(gw[&k2].is_empty());
    }

    #[rstest]
    fn test_spatial_temporal_pair_zplus(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: normal=X, complement=Z
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::ZPLUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);

        // Weight-2, normal(X), ZPLUS: half bottom stabilizers towards XPLUS
        let k2n = key(Basis::X, conn);
        assert!(gw.contains_key(&k2n));
        assert_eq!(gw[&k2n][0].chunk_index, INIT_CHUNK);
        assert_eq!(
            gw[&k2n][0].qubits.len(),
            expected_nonspatial_normal_half_bottom(distance),
        );

        // Weight-2, complement(Z), ZPLUS: space arm pipe top readout (1 qubit)
        let k2c = key(Basis::Z, conn);
        assert!(gw.contains_key(&k2c));
        assert_eq!(gw[&k2c][0].chunk_index, MEAS_CHUNK);
        assert_eq!(gw[&k2c][0].qubits.len(), 1);
        let d = distance as i32;
        assert!(has_qubit(&gw[&k2c], IVec2::new(2 * d + 1, d)));
    }

    #[rstest]
    fn test_spatial_temporal_pair_zminus(#[values(3, 5, 7)] distance: u32) {
        // ZXZ: normal=X, complement=Z
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::ZMINUS);
        let gw = cube_observable_gateway(CubeKind::ZXZ, conn, distance);

        // Weight-2, normal(X), ZMINUS: half bottom stabilizers towards XPLUS
        let k2n = key(Basis::X, conn);
        assert!(gw.contains_key(&k2n));
        assert_eq!(gw[&k2n][0].chunk_index, INIT_CHUNK);
        assert_eq!(
            gw[&k2n][0].qubits.len(),
            expected_nonspatial_normal_half_bottom(distance),
        );

        // Weight-2, complement(Z), ZMINUS: middle line + space arm pipe readout
        let k2c = key(Basis::Z, conn);
        assert!(gw.contains_key(&k2c));
        assert_eq!(gw[&k2c].len(), 1); // merged MEAS_CHUNK
        // d middle line + 1 pipe readout = d + 1
        assert_eq!(gw[&k2c][0].qubits.len(), distance as usize + 1,);
        let d = distance as i32;
        assert!(has_qubit(&gw[&k2c], IVec2::new(2 * d + 1, d)));
    }

    #[rstest]
    fn test_l_shape_spatial_pair(#[values(3, 5, 7)] distance: u32) {
        // ZZX: spatial, normal=X, complement=Z
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::YPLUS);
        let gw = cube_observable_gateway(CubeKind::ZZX, conn, distance);

        // Weight-2, normal(X): L-shape readout + pipe readouts
        let k2n = key(Basis::X, conn);
        assert!(gw.contains_key(&k2n));
        assert_eq!(gw[&k2n][0].chunk_index, MEAS_CHUNK);
        // L-shape: 2 × (d-1)/2 half-lines + corner + 2 pipe readouts = d + 2
        assert_eq!(total_measurements(&gw[&k2n]), distance as usize + 2);

        let d = distance as i32;
        assert!(has_qubit(&gw[&k2n], IVec2::new(d, d))); // corner
        assert!(has_qubit(&gw[&k2n], IVec2::new(2 * d + 1, d))); // XPLUS pipe
        assert!(has_qubit(&gw[&k2n], IVec2::new(d, 2 * d + 1))); // YPLUS pipe

        // Weight-2, complement(Z): full bottom stabilizers
        let k2c = key(Basis::Z, conn);
        assert!(gw.contains_key(&k2c));
        assert_eq!(gw[&k2c][0].chunk_index, INIT_CHUNK);
        // Two adjacent pipes restore both Z outer corners: (d+1)²/2
        assert_eq!(
            total_measurements(&gw[&k2c]),
            ((distance + 1) * (distance + 1) / 2) as usize,
        );
    }

    #[rstest]
    fn test_four_spatial_pipes_spatial_cube(#[values(3, 5, 7)] distance: u32) {
        // ZZX: spatial, normal=X, 4 spatial pipes
        let conn = Connectivity::ISOLATED
            .with_pipe(Direction::XPLUS)
            .with_pipe(Direction::XMINUS)
            .with_pipe(Direction::YPLUS)
            .with_pipe(Direction::YMINUS);
        let gw = cube_observable_gateway(CubeKind::ZZX, conn, distance);
        assert_eq!(gw.len(), expected_gateway_size(4));

        // Weight-4, normal(X): L-shape XOR decomposition + pipe readouts
        let k4n = key(Basis::X, conn);
        assert!(gw.contains_key(&k4n));
        assert_eq!(gw[&k4n][0].chunk_index, MEAS_CHUNK);
        // L1(XPLUS,YPLUS) ⊕ L2(XMINUS,YMINUS): center cancels → 2*(d-1) data.
        // Only positive-direction pipe readouts are owned locally, so add 2.
        assert_eq!(total_measurements(&gw[&k4n]), 2 * distance as usize);

        let d = distance as i32;
        assert!(has_qubit(&gw[&k4n], IVec2::new(2 * d + 1, d))); // XPLUS
        assert!(!has_qubit(&gw[&k4n], IVec2::new(-1, d))); // XMINUS owned by neighbor
        assert!(has_qubit(&gw[&k4n], IVec2::new(d, 2 * d + 1))); // YPLUS
        assert!(!has_qubit(&gw[&k4n], IVec2::new(d, -1))); // YMINUS owned by neighbor
        // Center qubit (d,d) should NOT be present (canceled by XOR)
        assert!(!has_qubit(&gw[&k4n], IVec2::new(d, d)));

        // Weight-4, complement(Z): full bottom stabilizers
        let k4c = key(Basis::Z, conn);
        assert!(gw.contains_key(&k4c));
        assert_eq!(gw[&k4c][0].chunk_index, INIT_CHUNK);
        // All 4 pipes restore both Z outer corners: (d+1)²/2
        assert_eq!(
            total_measurements(&gw[&k4c]),
            ((distance + 1) * (distance + 1) / 2) as usize,
        );
    }

    #[rstest]
    fn test_gateway_through_compile_includes_field(#[values(3, 5, 7)] distance: u32) {
        let template = compile_cube(
            CubeKind::ZXZ,
            Connectivity::ISOLATED,
            distance,
            distance,
            LayerSchedule::Compact,
        )
        .expect("compile cube");
        assert_eq!(template.observable_gateway.len(), expected_gateway_size(0));
        let k = key(Basis::Z, Connectivity::ISOLATED);
        assert!(template.observable_gateway.contains_key(&k));
        // Middle line has d qubits
        let entry = &template.observable_gateway[&k];
        assert_eq!(entry.measurements[0].measurements.len(), distance as usize);
    }

    #[rstest]
    fn test_middle_line_qubits(#[values(3, 5, 7)] distance: u32) {
        let d = distance as i32;

        // ZXZ: normal_dir=Y -> horizontal line at y = d
        let qubits = middle_line_qubits(distance, CubeKind::ZXZ);
        assert_eq!(qubits.len(), distance as usize);
        for (i, q) in qubits.iter().enumerate() {
            assert_eq!(*q, IVec2::new(1 + 2 * i as i32, d));
        }

        // ZXX: normal_dir=X -> vertical line at x = d
        let qubits = middle_line_qubits(distance, CubeKind::ZXX);
        assert_eq!(qubits.len(), distance as usize);
        for (i, q) in qubits.iter().enumerate() {
            assert_eq!(*q, IVec2::new(d, 1 + 2 * i as i32));
        }
    }

    #[rstest]
    fn test_half_middle_line(#[values(3, 5, 7)] distance: u32) {
        let d = distance as i32;
        let expected_count = (distance as usize - 1) / 2;

        // XPLUS: (center+2, center+4, ...) at y = center
        let qs = half_middle_line_qubits(distance, Direction::XPLUS);
        assert_eq!(qs.len(), expected_count);
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(*q, IVec2::new(d + 2 + 2 * i as i32, d));
        }

        // XMINUS: (1, 3, ..., center-2) at y = center
        let qs = half_middle_line_qubits(distance, Direction::XMINUS);
        assert_eq!(qs.len(), expected_count);
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(*q, IVec2::new(1 + 2 * i as i32, d));
        }

        // YPLUS: (center+2, center+4, ...) at x = center
        let qs = half_middle_line_qubits(distance, Direction::YPLUS);
        assert_eq!(qs.len(), expected_count);
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(*q, IVec2::new(d, d + 2 + 2 * i as i32));
        }

        // YMINUS: (1, 3, ..., center-2) at x = center
        let qs = half_middle_line_qubits(distance, Direction::YMINUS);
        assert_eq!(qs.len(), expected_count);
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(*q, IVec2::new(d, 1 + 2 * i as i32));
        }
    }

    #[rstest]
    fn test_l_shape_readout(#[values(3, 5, 7)] distance: u32) {
        let half_count = (distance as usize - 1) / 2;
        let center = IVec2::new(distance as i32, distance as i32);

        // Same-sign pair (XPLUS + YPLUS): corner included with X basis
        let qs = l_shape_readout_qubits(distance, Direction::XPLUS, Direction::YPLUS, Basis::X);
        assert_eq!(qs.len(), 2 * half_count + 1);
        assert!(qs.contains(&center));

        // Same pair with Z basis: no corner
        let qs = l_shape_readout_qubits(distance, Direction::XPLUS, Direction::YPLUS, Basis::Z);
        assert_eq!(qs.len(), 2 * half_count);
        assert!(!qs.contains(&center));

        // Diff-sign pair (XMINUS + YPLUS): corner included with Z basis
        let qs = l_shape_readout_qubits(distance, Direction::XMINUS, Direction::YPLUS, Basis::Z);
        assert_eq!(qs.len(), 2 * half_count + 1);
        assert!(qs.contains(&center));

        // Same pair with X basis: no corner
        let qs = l_shape_readout_qubits(distance, Direction::XMINUS, Direction::YPLUS, Basis::X);
        assert_eq!(qs.len(), 2 * half_count);
        assert!(!qs.contains(&center));

        for (arm1, arm2, basis) in [
            (Direction::XMINUS, Direction::YMINUS, Basis::X),
            (Direction::XPLUS, Direction::YMINUS, Basis::Z),
        ] {
            let qs = l_shape_readout_qubits(distance, arm1, arm2, basis);
            assert_eq!(qs.len(), 2 * half_count + 1);
            assert!(qs.contains(&center));
        }
    }

    #[rstest]
    fn test_pipe_boundary_readout_qubit(#[values(3, 5, 7)] distance: u32) {
        let d = distance as i32;
        assert_eq!(
            pipe_boundary_readout_qubit(d, Direction::XPLUS),
            IVec2::new(2 * d + 1, d),
        );
        assert_eq!(
            pipe_boundary_readout_qubit(d, Direction::XMINUS),
            IVec2::new(-1, d),
        );
        assert_eq!(
            pipe_boundary_readout_qubit(d, Direction::YPLUS),
            IVec2::new(d, 2 * d + 1),
        );
        assert_eq!(
            pipe_boundary_readout_qubit(d, Direction::YMINUS),
            IVec2::new(d, -1),
        );
    }
}
