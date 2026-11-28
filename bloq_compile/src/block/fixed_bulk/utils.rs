//! Shared utilities and surface code chunk builder for fixed-bulk compilation.
//!
//! Contains gate mapping helpers, the shared syndrome-round circuit emitter
//! ([`emit_syndrome_round`]) and standard round chunk builder
//! ([`make_surface_code_chunk`]), the temporal-face collapse rule
//! ([`collapse_matching_data_boundary`]), and grid coordinate / tile helpers
//! for cube schedules.

use crate::CompileError;
use crate::block::measurements::MeasurementIndex;
use crate::block::patch::{Patch, Tile};
use crate::signature::LayerSchedule;
use bloq_circuit::{Chunk, CoordCircuit, Flow, GateType, Op, PauliBasis, PauliMap};
use bloq_graph::{Basis, Pauli};
use glam::IVec2;
use itertools::Itertools;

/// Map a stabilizer basis to its reset gate.
pub(crate) fn reset_gate(basis: Basis) -> GateType {
    match basis {
        Basis::X => GateType::RX,
        Basis::Z => GateType::RZ,
    }
}

/// A syndrome round's circuit plus the qubit → measurement-id index recorded
/// while emitting it.
pub(crate) struct EmittedRound {
    pub(crate) circuit: CoordCircuit,
    pub(crate) measurements: MeasurementIndex,
}

/// Ordered per-basis batches of data qubits collapsed at a temporal face.
///
/// Batch order fixes measurement-id assignment and is caller-controlled (the
/// cube path emits X before Z, the rotation path Z before X); within each
/// batch qubits are `(y, x)`-sorted. Both are pinned by circuit snapshots.
pub(crate) type DataBatches = Vec<(Basis, Vec<IVec2>)>;

/// Per-basis batches in the given basis order, each sorted by `(y, x)`.
pub(crate) fn sorted_batches(
    data: Option<&crate::FxMap<IVec2, Basis>>,
    bases: [Basis; 2],
) -> DataBatches {
    let Some(data) = data else { return Vec::new() };
    bases
        .into_iter()
        .map(|basis| {
            let mut qubits: Vec<IVec2> = data
                .iter()
                .filter_map(|(&q, &b)| (b == basis).then_some(q))
                .collect();
            qubits.sort_unstable_by_key(|q| (q.y, q.x));
            (basis, qubits)
        })
        .collect()
}

/// Emit one standard syndrome-extraction round over `patch`: reset measure
/// qubits (Z then X) plus the optional data resets, the CX slot layers, then
/// measure the ancillas (Z then X) plus the optional data measurements.
///
/// The one shared syndrome-round circuit emitter (cube, memory padding,
/// Y-block bulk, patch rotation). Flow construction stays with the callers —
/// the cube path pairs it with [`make_surface_code_chunk`]'s standard flows,
/// the rotation path with its exact-boundary flow filter.
pub(crate) fn emit_syndrome_round(
    patch: &Patch,
    init_data: &[(Basis, Vec<IVec2>)],
    meas_data: &[(Basis, Vec<IVec2>)],
) -> Result<EmittedRound, CompileError> {
    let mut circuit = CoordCircuit::new();

    // Reset measure qubits (patch tile order is `(y, x, basis)`-sorted, so the
    // per-basis lists come out `(y, x)`-sorted), then any collapsed data.
    let basis_capacity = patch.len().div_ceil(2);
    let mut x_measure_qubits = Vec::with_capacity(basis_capacity);
    let mut z_measure_qubits = Vec::with_capacity(basis_capacity);
    for tile in patch.tiles() {
        match tile.basis() {
            Basis::X => x_measure_qubits.push(tile.measure_qubit()),
            Basis::Z => z_measure_qubits.push(tile.measure_qubit()),
        }
    }

    circuit.do_gate(GateType::RZ, z_measure_qubits.iter().copied())?;
    circuit.do_gate(GateType::RX, x_measure_qubits.iter().copied())?;
    for (basis, qubits) in init_data {
        circuit.do_gate(reset_gate(*basis), qubits.iter().copied())?;
    }
    circuit.tick();

    // CNOT layers
    let max_data_slots = patch
        .tiles()
        .iter()
        .map(|t| t.data_slots().len())
        .max()
        .unwrap_or(0);

    let mut cx_targets = Vec::with_capacity(patch.len().saturating_mul(2));
    for k in 0..max_data_slots {
        for tile in patch.tiles() {
            if let Some(&Some(dq)) = tile.data_slots().get(k) {
                let pair = match tile.basis() {
                    Basis::X => [tile.measure_qubit(), dq],
                    Basis::Z => [dq, tile.measure_qubit()],
                };
                cx_targets.extend(pair);
            }
        }
        circuit.do_gate(GateType::CX, cx_targets.drain(..))?;
        circuit.tick();
    }

    // Measurement
    let measurement_count = patch.len()
        + meas_data
            .iter()
            .map(|(_, qubits)| qubits.len())
            .sum::<usize>();
    let mut measurements = MeasurementIndex::with_capacity(measurement_count);
    record_measurements(
        &mut circuit,
        PauliBasis::Z,
        &z_measure_qubits,
        &mut measurements,
    );
    record_measurements(
        &mut circuit,
        PauliBasis::X,
        &x_measure_qubits,
        &mut measurements,
    );
    for (basis, qubits) in meas_data {
        record_measurements(
            &mut circuit,
            PauliBasis::from(*basis),
            qubits,
            &mut measurements,
        );
    }

    Ok(EmittedRound {
        circuit,
        measurements,
    })
}

/// Which side of a chunk a per-tile flow attaches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TileFlow {
    /// `support → ∅`: the chunk consumes the stabilizer coming in.
    Consume,
    /// `∅ → support`: the chunk creates the stabilizer going out.
    Create,
}

/// One flow per tile in a fixed direction, carrying `measurements` and centred
/// on the tile's measure qubit. Shared by the port MPP chunk, the selective
/// transversal arms, and the Y transition round.
pub(crate) fn tile_flow(
    tile: &Tile,
    direction: TileFlow,
    measurements: impl IntoIterator<Item = u32>,
) -> Flow {
    let support = tile.pauli_map();
    let (start, end) = match direction {
        TileFlow::Consume => (support, PauliMap::empty()),
        TileFlow::Create => (PauliMap::empty(), support),
    };
    Flow::new(start, end)
        .with_measurements(measurements)
        .with_center(tile.measure_qubit())
}

/// Build a standard surface code syndrome extraction round as a `Chunk`.
pub(crate) fn make_surface_code_chunk(
    patch: &Patch,
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: Option<&crate::FxMap<IVec2, Basis>>,
) -> Result<Chunk, CompileError> {
    let round = emit_syndrome_round(
        patch,
        &sorted_batches(init_data, [Basis::X, Basis::Z]),
        &sorted_batches(meas_data, [Basis::X, Basis::Z]),
    )?;
    let flows = standard_round_flows(patch, &round.measurements, init_data, meas_data);
    Ok(Chunk {
        circuit: round.circuit,
        flows,
    })
}

/// A cube's three chunks plus the measurement indices recorded while emitting
/// them, so gateway resolution does not have to re-scan the circuits.
pub(crate) struct CubeChunks {
    /// The init chunk and its ancilla-only index, present exactly when
    /// `init_data` was supplied.
    pub(crate) init: Option<(Chunk, MeasurementIndex)>,
    pub(crate) bulk: Chunk,
    pub(crate) meas: Chunk,
    /// `meas`'s index: the ancillas plus the collapsed data qubits.
    pub(crate) meas_ids: MeasurementIndex,
}

/// Build a cube's common round once, adding only its boundary resets/readout.
pub(crate) fn make_surface_code_cube_chunks(
    patch: &Patch,
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: &crate::FxMap<IVec2, Basis>,
) -> Result<CubeChunks, CompileError> {
    let mut round = emit_syndrome_round(patch, &[], &[])?;
    let init_batches = sorted_batches(init_data, [Basis::X, Basis::Z]);
    let init_circuit = init_data.map(|_| with_data_resets(round.circuit.clone(), &init_batches));
    let bulk_circuit = round.circuit.clone();
    // Data resets record nothing, so the round's index as it stands is exactly
    // the init chunk's; the readout below only extends the `meas` one.
    let init_ids = init_data.map(|_| round.measurements.clone());

    round.measurements.reserve(meas_data.len());
    for (basis, qubits) in sorted_batches(Some(meas_data), [Basis::X, Basis::Z]) {
        record_measurements(
            &mut round.circuit,
            PauliBasis::from(basis),
            &qubits,
            &mut round.measurements,
        );
    }
    let (init_flows, bulk_flows, meas_flows) =
        cube_round_flows(patch, &round.measurements, init_data, meas_data);
    let init = init_circuit
        .zip(init_flows)
        .zip(init_ids)
        .map(|((circuit, flows), ids)| (Chunk { circuit, flows }, ids));
    let bulk = Chunk {
        circuit: bulk_circuit,
        flows: bulk_flows,
    };
    let meas = Chunk {
        circuit: round.circuit,
        flows: meas_flows,
    };
    Ok(CubeChunks {
        init,
        bulk,
        meas,
        meas_ids: round.measurements,
    })
}

fn cube_round_flows(
    patch: &Patch,
    measurement_ids: &MeasurementIndex,
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: &crate::FxMap<IVec2, Basis>,
) -> (Option<Vec<Flow>>, Vec<Flow>, Vec<Flow>) {
    let capacity = patch.len().saturating_mul(2);
    let mut init_flows = init_data.map(|_| Vec::with_capacity(capacity));
    let mut bulk_flows = Vec::with_capacity(capacity);
    let mut meas_flows = Vec::with_capacity(capacity);
    let mut active_dqs = Vec::new();

    for tile in patch.tiles() {
        let basis = tile.basis();
        let pauli = Pauli::from(basis);
        let measure = tile.measure_qubit();
        let measurement = measurement_ids.expect_measurement(measure);
        collect_active_data_qubits(tile, &mut active_dqs);
        let support = PauliMap::from_unique_entries(active_dqs.iter().map(|&dq| (dq, pauli)));
        let boundary_flow = |start, end| {
            Flow::new(start, end)
                .with_measurements([measurement])
                .with_center(measure)
        };
        let output_flow = |end, mut measured_dqs: Vec<IVec2>| {
            measured_dqs.sort_unstable_by_key(|coord| (coord.y, coord.x));
            let measurements = measured_dqs
                .into_iter()
                .map(|q| measurement_ids.expect_measurement(q))
                .chain(std::iter::once(measurement));
            Flow::new(PauliMap::empty(), end)
                .with_measurements(measurements)
                .with_center(measure)
        };

        bulk_flows.push(boundary_flow(support.clone(), PauliMap::empty()));
        bulk_flows.push(boundary_flow(PauliMap::empty(), support.clone()));

        if let (Some(init_data), Some(init_flows)) = (init_data, &mut init_flows) {
            if let Some((input, _)) =
                collapse_matching_data_boundary(&active_dqs, basis, pauli, Some(init_data))
            {
                init_flows.push(boundary_flow(input, PauliMap::empty()));
            }
            init_flows.push(boundary_flow(PauliMap::empty(), support.clone()));
        }

        meas_flows.push(boundary_flow(support, PauliMap::empty()));
        if let Some((output, measured_dqs)) =
            collapse_matching_data_boundary(&active_dqs, basis, pauli, Some(meas_data))
        {
            meas_flows.push(output_flow(output, measured_dqs));
        }
    }

    (init_flows, bulk_flows, meas_flows)
}

fn with_data_resets(mut circuit: CoordCircuit, init_data: &[(Basis, Vec<IVec2>)]) -> CoordCircuit {
    let entry = circuit.entry_body();
    let ops = circuit
        .body_mut(entry)
        .expect("entry body exists while adding data resets")
        .ops_mut();
    let first_tick = ops
        .iter()
        .position(|op| matches!(op, Op::Tick))
        .expect("a syndrome round contains a reset barrier");
    let resets = init_data
        .iter()
        .filter(|(_, qubits)| !qubits.is_empty())
        .map(|(basis, qubits)| Op::Gate {
            gate: reset_gate(*basis),
            qubits: qubits.clone(),
        });
    ops.splice(first_tick..first_tick, resets);
    circuit
}

/// Creator/consumer flows of a standard round: one pair per tile, collapsed
/// against the init/meas data maps by the shared boundary rule.
fn standard_round_flows(
    patch: &Patch,
    measurement_ids: &MeasurementIndex,
    init_data: Option<&crate::FxMap<IVec2, Basis>>,
    meas_data: Option<&crate::FxMap<IVec2, Basis>>,
) -> Vec<Flow> {
    let mut flows = Vec::with_capacity(patch.len().saturating_mul(2));
    let mut active_dqs = Vec::new();
    if init_data.is_none() && meas_data.is_none() {
        // Syndrome ids are allocated in this basis order and each basis list
        // inherits patch order. Creator-before-consumer is the boundary-flow
        // comparator's order for two flows carrying the same measurement.
        for wanted_basis in [Basis::Z, Basis::X] {
            for tile in patch
                .tiles()
                .iter()
                .filter(|tile| tile.basis() == wanted_basis)
            {
                let pauli = Pauli::from(wanted_basis);
                let measure = tile.measure_qubit();
                let measurement = measurement_ids.expect_measurement(measure);
                collect_active_data_qubits(tile, &mut active_dqs);
                let support =
                    PauliMap::from_unique_entries(active_dqs.iter().map(|&dq| (dq, pauli)));
                flows.push(
                    Flow::new(PauliMap::empty(), support.clone())
                        .with_measurements([measurement])
                        .with_center(measure),
                );
                flows.push(
                    Flow::new(support, PauliMap::empty())
                        .with_measurements([measurement])
                        .with_center(measure),
                );
            }
        }
        return flows;
    }

    for tile in patch.tiles() {
        let basis = tile.basis();
        let pauli = Pauli::from(basis);
        let measure = tile.measure_qubit();
        collect_active_data_qubits(tile, &mut active_dqs);

        let input_boundary = collapse_matching_data_boundary(&active_dqs, basis, pauli, init_data);
        let output_boundary = collapse_matching_data_boundary(&active_dqs, basis, pauli, meas_data);
        let measurement = measurement_ids.expect_measurement(measure);

        if let Some((in_start, _)) = input_boundary {
            flows.push(
                Flow::new(in_start, PauliMap::empty())
                    .with_measurements([measurement])
                    .with_center(measure),
            );
        }

        if let Some((out_end, measured_dqs)) = output_boundary {
            let mut measured_dqs = measured_dqs;
            measured_dqs.sort_unstable_by_key(|coord| (coord.y, coord.x));
            let measurements = measured_dqs
                .into_iter()
                .map(|q| measurement_ids.expect_measurement(q))
                .chain(std::iter::once(measurement));

            flows.push(
                Flow::new(PauliMap::empty(), out_end)
                    .with_measurements(measurements)
                    .with_center(measure),
            );
        }
    }
    flows
}

fn record_measurements(
    circuit: &mut CoordCircuit,
    basis: PauliBasis,
    targets: &[IVec2],
    measurement_ids: &mut MeasurementIndex,
) {
    let ids = circuit.measure(basis, targets.iter().copied());
    for (&target, id) in targets.iter().zip(ids) {
        measurement_ids.record(target, id);
    }
}

/// Return the continuing stabilizer support and same-basis collapsed data qubits.
///
/// The single source of the temporal-face collapse rule: if any active data
/// qubit was collapsed in the opposite basis, the stabilizer flow is
/// invalidated (`None`). Otherwise, qubits absent from `collapsed_data`
/// continue across this boundary, while matching qubits terminate/start at
/// the boundary. (`walk::collapse_matching_output` applies the same rule over
/// its dense sets with a hard error on Pauli mismatch.)
pub(crate) fn collapse_matching_data_boundary(
    active_dqs: &[IVec2],
    basis: Basis,
    pauli: Pauli,
    collapsed_data: Option<&crate::FxMap<IVec2, Basis>>,
) -> Option<(PauliMap, Vec<IVec2>)> {
    let mut collapsed = Vec::new();
    let Some(collapsed_data) = collapsed_data else {
        return Some((
            PauliMap::from_unique_entries(active_dqs.iter().map(|&dq| (dq, pauli))),
            collapsed,
        ));
    };

    let mut continuing = PauliMap::empty();
    for &dq in active_dqs {
        if let Some(&collapsed_basis) = collapsed_data.get(&dq) {
            if collapsed_basis != basis {
                return None;
            }
            collapsed.push(dq);
        } else {
            continuing.insert(dq, pauli);
        }
    }

    Some((continuing, collapsed))
}

fn collect_active_data_qubits(tile: &Tile, active_dqs: &mut Vec<IVec2>) {
    active_dqs.clear();
    for &slot in tile.data_slots() {
        if let Some(coord) = slot
            && !active_dqs.contains(&coord)
        {
            active_dqs.push(coord);
        }
    }
}

/// Convert grid indices `(row, col)` to measure qubit coordinates.
///
/// Row 0 is the visual top of the grid; Y axis points upward, so row 0
/// maps to the highest Y value: `Y = 2 * (nrows - 1 - row)`, `X = 2 * col`.
pub(crate) fn grid_to_coord(row: usize, col: usize, nrows: usize) -> IVec2 {
    IVec2::new(2 * col as i32, 2 * (nrows - 1 - row) as i32)
}

/// Diagonal offset from a measure qubit to one of its four neighboring data qubits.
///
/// The surface code stabilizer patch places measure qubits at even grid coordinates
/// and data qubits at odd coordinates. Each measure qubit has up to four diagonal neighbors
/// at offsets `(±1, ±1)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Corner {
    /// Top-left: `(-1, +1)`.
    TL,
    /// Top-right: `(+1, +1)`.
    TR,
    /// Bottom-left: `(-1, -1)`.
    BL,
    /// Bottom-right: `(+1, -1)`.
    BR,
}

impl Corner {
    /// All four corners in canonical order: `[TL, TR, BL, BR]`.
    pub(crate) const ALL: [Corner; 4] = [Corner::TL, Corner::TR, Corner::BL, Corner::BR];

    /// Convert to the `IVec2` offset vector.
    pub(crate) const fn to_ivec2(self) -> IVec2 {
        match self {
            Corner::TL => IVec2::new(-1, 1),
            Corner::TR => IVec2::new(1, 1),
            Corner::BL => IVec2::new(-1, -1),
            Corner::BR => IVec2::new(1, -1),
        }
    }

    /// Whether this corner is on the top row (`TL` or `TR`).
    pub(crate) const fn is_top(self) -> bool {
        matches!(self, Corner::TL | Corner::TR)
    }

    /// Whether this corner is on the left column (`TL` or `BL`).
    pub(crate) const fn is_left(self) -> bool {
        matches!(self, Corner::TL | Corner::BL)
    }
}

/// Build a `Tile` for a cube CX schedule at the layer's slot depth.
///
/// For each slot in the schedule, the data slot is `Some(pos + offset)` if the
/// corner is in `active`, otherwise `None`.
#[cfg(test)]
pub(crate) fn make_cube_tile(
    basis: Basis,
    pos: IVec2,
    active: &[Corner],
    is_vertical: bool,
    layer_schedule: LayerSchedule,
) -> Tile {
    make_cube_tile_with_reverse(basis, pos, active, is_vertical, layer_schedule, false)
}

/// Build a cube tile using the backward hook schedule for an extended layer.
///
/// Alternation is a property of a round, not of the patch geometry, so callers
/// keep the ordinary constructor above for the overwhelmingly common forward
/// case and opt into this one only for the backward round.
pub(crate) fn make_cube_tile_with_reverse(
    basis: Basis,
    pos: IVec2,
    active: &[Corner],
    is_vertical: bool,
    layer_schedule: LayerSchedule,
    reverse_schedule: bool,
) -> Tile {
    let cx = if is_vertical {
        CXSchedule::Vertical
    } else {
        CXSchedule::Horizontal
    };
    let from_slots = |slots: &[Option<Corner>]| {
        Tile::new(
            basis,
            pos,
            slots.iter().copied().map(|slot| {
                slot.and_then(|corner| active.contains(&corner).then_some(pos + corner.to_ivec2()))
            }),
        )
    };
    match layer_schedule {
        LayerSchedule::Compact => from_slots(&cx.compact().map(Some)),
        LayerSchedule::Padded | LayerSchedule::Mixed => from_slots(&cx.schedule()),
        LayerSchedule::Extended => from_slots(&cx.extended(reverse_schedule)),
        LayerSchedule::ExtendedY => from_slots(&cx.extended_y(reverse_schedule)),
    }
}

/// A `width × height` block of data cells rooted at row `y0`, in measure-qubit
/// coordinates: `x ∈ [0, 2·width]`, `y ∈ [2·y0, 2·(y0 + height)]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rect {
    pub(crate) width: i32,
    pub(crate) y0: i32,
    pub(crate) height: i32,
}

impl Rect {
    pub(crate) const fn new(width: i32, height: i32) -> Self {
        Self {
            width,
            y0: 0,
            height,
        }
    }

    pub(crate) const fn rooted(width: i32, y0: i32, height: i32) -> Self {
        Self { width, y0, height }
    }

    /// Odd-coordinate data qubits filling the rectangle.
    pub(crate) fn data_qubits(self) -> crate::FxSet<IVec2> {
        (0..self.width)
            .flat_map(|i| {
                (self.y0..self.y0 + self.height).map(move |j| IVec2::new(2 * i + 1, 2 * j + 1))
            })
            .collect()
    }

    pub(crate) const fn on_edge(self, m: IVec2) -> bool {
        m.x == 0
            || m.x == 2 * self.width
            || m.y == 2 * self.y0
            || m.y == 2 * (self.y0 + self.height)
    }
}

/// Stabilizer basis demanded by each edge of a [`Rect`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct EdgeBases {
    pub(crate) bottom: Basis,
    pub(crate) top: Basis,
    pub(crate) left: Basis,
    pub(crate) right: Basis,
}

impl EdgeBases {
    /// The standard patch layout: `x_basis` on the left/right edges, its flip
    /// on top and bottom.
    pub(crate) fn standard(x_basis: Basis) -> Self {
        Self {
            bottom: x_basis.flip(),
            top: x_basis.flip(),
            left: x_basis,
            right: x_basis,
        }
    }

    /// The basis every edge through `m` agrees on. `None` when `m` is interior
    /// or when two edges demand different bases (a deleted corner).
    fn demanded(self, m: IVec2, rect: Rect) -> Option<Basis> {
        let mut demanded = None;
        for edge in [
            (m.y == 2 * rect.y0).then_some(self.bottom),
            (m.y == 2 * (rect.y0 + rect.height)).then_some(self.top),
            (m.x == 0).then_some(self.left),
            (m.x == 2 * rect.width).then_some(self.right),
        ]
        .into_iter()
        .flatten()
        {
            if demanded.is_some_and(|basis| basis != edge) {
                return None;
            }
            demanded = Some(edge);
        }
        demanded
    }

    /// Whether a measure qubit of `basis` at `m` survives the boundary rule.
    pub(crate) fn keeps(self, m: IVec2, rect: Rect, basis: Basis) -> bool {
        match self.demanded(m, rect) {
            Some(demanded) => basis == demanded,
            None => !rect.on_edge(m),
        }
    }
}

/// The shared lattice carve: keep the measure qubits touching ≥2 candidate
/// data qubits that also pass `keep`, then the data qubits touching ≥2
/// survivors, then build one tile per surviving measure qubit. `patch_measure`
/// maps local measure coordinates to their final patch coordinates.
pub(crate) fn carve_patch(
    possible_data: crate::FxSet<IVec2>,
    keep: impl Fn(IVec2) -> bool,
    patch_measure: impl Fn(IVec2) -> IVec2,
    tile: impl Fn(IVec2, &crate::FxSet<IVec2>) -> Tile,
) -> Patch {
    let measure_qubits: crate::FxSet<IVec2> = candidate_measure_qubits(&possible_data)
        .into_iter()
        .filter(|&m| touches_multiple(m, &possible_data) && keep(m))
        .collect();
    let data_qubits = data_qubits_with_multiple_measure_neighbors(possible_data, &measure_qubits);
    let mut ordered_measure_qubits = measure_qubits.iter().copied().collect::<Vec<_>>();
    ordered_measure_qubits.sort_unstable_by_key(|&m| {
        let m = patch_measure(m);
        (m.y, m.x)
    });
    Patch::from_sorted_tiles(
        ordered_measure_qubits
            .into_iter()
            .map(|m| tile(m, &data_qubits))
            .collect(),
    )
}

/// A tile at `m` whose slots follow `order`, occupied only where the corner's
/// data qubit survived the carve.
pub(crate) fn corner_tile(
    m: IVec2,
    basis: Basis,
    order: impl IntoIterator<Item = Option<Corner>>,
    data: &crate::FxSet<IVec2>,
) -> Tile {
    Tile::new(
        basis,
        m,
        order.into_iter().map(|corner| {
            corner.and_then(|corner| {
                let dq = m + corner.to_ivec2();
                data.contains(&dq).then_some(dq)
            })
        }),
    )
}

/// Deduplicate coordinates and sort them by `(y, x)`.
pub(crate) fn sorted_unique_coords(coords: impl IntoIterator<Item = IVec2>) -> Vec<IVec2> {
    coords
        .into_iter()
        .collect::<crate::FxSet<_>>()
        .into_iter()
        .sorted_by_key(|c| (c.y, c.x))
        .collect()
}

/// Every corner-neighbor of the data qubits: the candidate measure qubits.
pub(crate) fn candidate_measure_qubits(possible_data: &crate::FxSet<IVec2>) -> crate::FxSet<IVec2> {
    possible_data
        .iter()
        .flat_map(|&dq| Corner::ALL.iter().map(move |c| dq + c.to_ivec2()))
        .collect()
}

/// Whether at least two of `center`'s four diagonal neighbors are in `targets`
/// — the shared "measure qubit touches ≥2 data qubits" predicate (and its dual,
/// a data qubit touching ≥2 measure qubits).
pub(crate) fn touches_multiple(center: IVec2, targets: &crate::FxSet<IVec2>) -> bool {
    Corner::ALL
        .iter()
        .filter(|c| targets.contains(&(center + c.to_ivec2())))
        .nth(1)
        .is_some()
}

/// Data qubits touching at least two of the surviving measure qubits.
pub(crate) fn data_qubits_with_multiple_measure_neighbors(
    possible_data: crate::FxSet<IVec2>,
    measure_qubits: &crate::FxSet<IVec2>,
) -> crate::FxSet<IVec2> {
    possible_data
        .into_iter()
        .filter(|&dq| touches_multiple(dq, measure_qubits))
        .collect()
}

/// Checkerboard basis for a measure qubit at the given bloq coordinates.
///
/// In the surface code layout, measure qubits at even coordinates alternate
/// between X and Z basis in a checkerboard pattern: Z when `(mx + my) / 2` is
/// odd, X when even.
pub(crate) fn checkerboard_basis(m: IVec2) -> Basis {
    debug_assert!(
        m.x % 2 == 0 && m.y % 2 == 0,
        "measure qubits must be at even coordinates"
    );
    if ((m.x + m.y) >> 1) & 1 == 1 {
        Basis::Z
    } else {
        Basis::X
    }
}

pub(crate) fn tile_cx_schedule(tile_basis: Basis, horizontal_hook_basis: Basis) -> CXSchedule {
    if (tile_basis == Basis::X) ^ (horizontal_hook_basis == Basis::Z) {
        CXSchedule::Horizontal
    } else {
        CXSchedule::Vertical
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CXSchedule {
    Horizontal,
    Vertical,
}

impl CXSchedule {
    const HORIZONTAL_SCHEDULE: [Option<Corner>; 5] = [
        Some(Corner::TL),
        Some(Corner::TR),
        Some(Corner::BL),
        None,
        Some(Corner::BR),
    ];
    const VERTICAL_SCHEDULE: [Option<Corner>; 5] = [
        Some(Corner::TL),
        None,
        Some(Corner::BL),
        Some(Corner::TR),
        Some(Corner::BR),
    ];
    const H_COMPACT_SCHEDULE: [Corner; 4] = [Corner::TL, Corner::TR, Corner::BL, Corner::BR];
    const V_COMPACT_SCHEDULE: [Corner; 4] = [Corner::TL, Corner::BL, Corner::TR, Corner::BR];

    // Extended schedules are axis-specific. They are not interchangeable with
    // compact/padded schedules:
    // wall data occupies slots 1 and 3, while slots 0 and 5 bracket the GHZ
    // create/disentangle pair.
    const H_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        Some(Corner::TL),
        None,
        Some(Corner::TR),
        Some(Corner::BL),
        Some(Corner::BR),
        None,
    ];
    const V_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        Some(Corner::TL),
        Some(Corner::BL),
        Some(Corner::TR),
        None,
        Some(Corner::BR),
        None,
    ];
    // Literal time reversals of the forward schedules.
    const H_REVERSED_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        None,
        Some(Corner::BR),
        Some(Corner::BL),
        Some(Corner::TR),
        None,
        Some(Corner::TL),
    ];
    const V_REVERSED_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        None,
        Some(Corner::BR),
        None,
        Some(Corner::TR),
        Some(Corner::BL),
        Some(Corner::TL),
    ];
    // Diagonal transpose of the X-wall schedules. A coordinate swap maps
    // `Horizontal ↔ Vertical` and `TL,TR,BL,BR → BR,TR,BL,TL`.
    const H_Y_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        Some(Corner::BR),
        Some(Corner::BL),
        Some(Corner::TR),
        None,
        Some(Corner::TL),
        None,
    ];
    const V_Y_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        Some(Corner::BR),
        None,
        Some(Corner::TR),
        Some(Corner::BL),
        Some(Corner::TL),
        None,
    ];
    const H_Y_REVERSED_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        None,
        Some(Corner::TL),
        None,
        Some(Corner::TR),
        Some(Corner::BL),
        Some(Corner::BR),
    ];
    const V_Y_REVERSED_EXTENDED_SCHEDULE: [Option<Corner>; 6] = [
        None,
        Some(Corner::TL),
        Some(Corner::BL),
        Some(Corner::TR),
        None,
        Some(Corner::BR),
    ];
    pub(crate) const fn schedule(self) -> [Option<Corner>; 5] {
        match self {
            CXSchedule::Vertical => Self::VERTICAL_SCHEDULE,
            CXSchedule::Horizontal => Self::HORIZONTAL_SCHEDULE,
        }
    }

    pub(crate) const fn compact(self) -> [Corner; 4] {
        match self {
            CXSchedule::Vertical => Self::V_COMPACT_SCHEDULE,
            CXSchedule::Horizontal => Self::H_COMPACT_SCHEDULE,
        }
    }

    pub(crate) const fn extended(self, reversed: bool) -> [Option<Corner>; 6] {
        match (self, reversed) {
            (CXSchedule::Vertical, false) => Self::V_EXTENDED_SCHEDULE,
            (CXSchedule::Horizontal, false) => Self::H_EXTENDED_SCHEDULE,
            (CXSchedule::Vertical, true) => Self::V_REVERSED_EXTENDED_SCHEDULE,
            (CXSchedule::Horizontal, true) => Self::H_REVERSED_EXTENDED_SCHEDULE,
        }
    }

    pub(crate) const fn extended_y(self, reversed: bool) -> [Option<Corner>; 6] {
        match (self, reversed) {
            (CXSchedule::Horizontal, false) => Self::H_Y_EXTENDED_SCHEDULE,
            (CXSchedule::Vertical, false) => Self::V_Y_EXTENDED_SCHEDULE,
            (CXSchedule::Horizontal, true) => Self::H_Y_REVERSED_EXTENDED_SCHEDULE,
            (CXSchedule::Vertical, true) => Self::V_Y_REVERSED_EXTENDED_SCHEDULE,
        }
    }
}

/// Build a rectangular surface code patch with specified boundary bases.
///
/// Generates a rotated surface code patch of size `width` by `height` directly in bloq
/// coordinates (Y-up integer grid):
/// - **Data qubits** at odd coordinates `(2i+1, 2j+1)` for `i` in `0..width`
///   and `j` in `0..height`.
/// - **Measure qubits** at even coordinates, diagonal neighbors of data qubits.
///
/// `boundaries` controls which stabilizer basis is kept on each edge.
/// `horizontal_hook_basis` controls the padded 5-step CX interaction order.
/// It stays separate from the boundary layout because some derived patches,
/// such as the Y-basis degenerate patch, intentionally reuse a hook
/// orientation that does not match the patch's top-edge boundary basis.
pub(crate) fn make_rectangular_surface_code_patch(
    width: u32,
    height: u32,
    boundaries: &EdgeBases,
    horizontal_hook_basis: Basis,
    reverse_schedule: bool,
) -> Patch {
    let rect = Rect::new(width as i32, height as i32);
    carve_patch(
        rect.data_qubits(),
        |m| boundaries.keeps(m, rect, checkerboard_basis(m)),
        |m| m,
        |m, data| {
            let basis = checkerboard_basis(m);
            let mut corners = tile_cx_schedule(basis, horizontal_hook_basis).compact();
            if reverse_schedule {
                corners.reverse();
            }
            corner_tile(m, basis, corners.map(Some), data)
        },
    )
}

/// Build a compact standard surface code patch (rotated code with distance `d`).
///
/// `top_basis` is the stabilizer basis on the top and bottom edges (left and
/// right get `top_basis.flip()`). It also sets the padded 5-step CX hook
/// orientation: tiles matching `top_basis` use horizontal hooks.
pub(crate) fn make_normal_surface_code_patch(d: u32, top_basis: Basis) -> Patch {
    make_rectangular_surface_code_patch(
        d,
        d,
        &EdgeBases::standard(top_basis.flip()),
        top_basis,
        false,
    )
}

#[cfg(test)]
mod tests {
    use bloq_stim::StimFlowVerifier;

    use super::*;

    /// Build a minimal 2-tile patch: one X-tile and one Z-tile sharing two data
    /// qubits, with 4-slot (ybasis-style) schedules.
    fn two_tile_patch() -> Patch {
        let z_tile = Tile::new(
            Basis::Z,
            IVec2::new(0, 2),
            vec![
                None,                   // no TL
                Some(IVec2::new(1, 3)), // TR
                None,                   // no BL
                Some(IVec2::new(1, 1)), // BR
            ],
        );
        let x_tile = Tile::new(
            Basis::X,
            IVec2::new(2, 2),
            vec![
                Some(IVec2::new(1, 3)), // TL
                None,                   // no TR
                Some(IVec2::new(1, 1)), // BL
                None,                   // no BR
            ],
        );
        Patch::new(vec![z_tile, x_tile])
    }

    #[test]
    fn test_bulk_round_flows() {
        let patch = two_tile_patch();
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        assert_eq!(chunk.flows.len(), 4);
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_init_same_basis() {
        let patch = two_tile_patch();
        let init: crate::FxMap<IVec2, Basis> =
            [(IVec2::new(1, 1), Basis::Z), (IVec2::new(1, 3), Basis::Z)]
                .into_iter()
                .collect();
        let chunk = make_surface_code_chunk(&patch, Some(&init), None).unwrap();
        assert_eq!(chunk.flows.len(), 3);
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_meas_same_basis() {
        let patch = two_tile_patch();
        let meas: crate::FxMap<IVec2, Basis> =
            [(IVec2::new(1, 1), Basis::X), (IVec2::new(1, 3), Basis::X)]
                .into_iter()
                .collect();
        let chunk = make_surface_code_chunk(&patch, None, Some(&meas)).unwrap();
        assert_eq!(chunk.flows.len(), 3);
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_init_and_meas() {
        let patch = two_tile_patch();
        let init: crate::FxMap<IVec2, Basis> =
            [(IVec2::new(1, 1), Basis::Z), (IVec2::new(1, 3), Basis::Z)]
                .into_iter()
                .collect();
        let meas: crate::FxMap<IVec2, Basis> =
            [(IVec2::new(1, 1), Basis::Z), (IVec2::new(1, 3), Basis::Z)]
                .into_iter()
                .collect();
        let chunk = make_surface_code_chunk(&patch, Some(&init), Some(&meas)).unwrap();
        assert_eq!(chunk.flows.len(), 2);
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_grid_to_coord() {
        assert_eq!(grid_to_coord(0, 0, 4), IVec2::new(0, 6));
        assert_eq!(grid_to_coord(0, 3, 4), IVec2::new(6, 6));
        assert_eq!(grid_to_coord(3, 0, 4), IVec2::new(0, 0));
        assert_eq!(grid_to_coord(3, 3, 4), IVec2::new(6, 0));
    }

    #[test]
    fn test_make_cube_tile_vertical() {
        let tile = make_cube_tile(
            Basis::Z,
            IVec2::new(2, 4),
            &Corner::ALL,
            true,
            LayerSchedule::Padded,
        );
        assert_eq!(tile.basis(), Basis::Z);
        assert_eq!(tile.measure_qubit(), IVec2::new(2, 4));
        assert_eq!(tile.data_slots().len(), 5);
        // Vertical: TL@0, NOP@1, BL@2, TR@3, BR@4
        assert_eq!(tile.data_slots()[0], Some(IVec2::new(1, 5))); // TL
        assert_eq!(tile.data_slots()[1], None); // NOP
        assert_eq!(tile.data_slots()[2], Some(IVec2::new(1, 3))); // BL
        assert_eq!(tile.data_slots()[3], Some(IVec2::new(3, 5))); // TR
        assert_eq!(tile.data_slots()[4], Some(IVec2::new(3, 3))); // BR
    }

    #[test]
    fn test_make_cube_tile_boundary() {
        let tile = make_cube_tile(
            Basis::Z,
            IVec2::new(2, 4),
            &[Corner::TL, Corner::BL],
            true,
            LayerSchedule::Padded,
        );
        assert_eq!(tile.data_slots()[0], Some(IVec2::new(1, 5))); // TL
        assert_eq!(tile.data_slots()[1], None); // NOP
        assert_eq!(tile.data_slots()[2], Some(IVec2::new(1, 3))); // BL
        assert_eq!(tile.data_slots()[3], None); // TR inactive
        assert_eq!(tile.data_slots()[4], None); // BR inactive
    }

    #[test]
    fn test_make_cube_tile_extended() {
        let tile = make_cube_tile(
            Basis::Z,
            IVec2::new(2, 4),
            &Corner::ALL,
            true,
            LayerSchedule::Extended,
        );
        assert_eq!(tile.data_slots().len(), 6);
        // Vertical: TL@0, BL@1, TR@2, NOP@3, BR@4, NOP@5
        assert_eq!(tile.data_slots()[0], Some(IVec2::new(1, 5))); // TL
        assert_eq!(tile.data_slots()[1], Some(IVec2::new(1, 3))); // BL
        assert_eq!(tile.data_slots()[2], Some(IVec2::new(3, 5))); // TR
        assert_eq!(tile.data_slots()[3], None); // NOP
        assert_eq!(tile.data_slots()[4], Some(IVec2::new(3, 3))); // BR
        assert_eq!(tile.data_slots()[5], None); // NOP
    }

    /// The invariant the wall pipe leans on. `TL`, `TR` and `BR` sit at fixed
    /// slots whichever hook the tile takes, and `BL` — the one corner the hook
    /// moves — occupies slot 1 or 3. A seam-adjacent data qubit is only ever a
    /// cube tile's `TR` or `BR`, so slots 1 and 3 stay free for the wall's
    /// extended stabilizers to drive that column.
    #[test]
    fn extended_schedule_reserves_the_seam_slots() {
        for cx in [CXSchedule::Horizontal, CXSchedule::Vertical] {
            let mut reversed = cx.extended(false);
            reversed.reverse();
            assert_eq!(cx.extended(true), reversed);

            let slot = |corner| {
                cx.extended(false)
                    .iter()
                    .position(|scheduled| *scheduled == Some(corner))
                    .expect("every corner is scheduled")
            };
            assert_eq!(slot(Corner::TL), 0);
            assert_eq!(slot(Corner::TR), 2);
            assert_eq!(slot(Corner::BR), 4);
            assert!(matches!(slot(Corner::BL), 1 | 3));
        }

        let transpose = |corner| match corner {
            Corner::TL => Corner::BR,
            Corner::TR => Corner::TR,
            Corner::BL => Corner::BL,
            Corner::BR => Corner::TL,
        };
        for (x_wall, y_wall) in [
            (CXSchedule::Horizontal, CXSchedule::Vertical),
            (CXSchedule::Vertical, CXSchedule::Horizontal),
        ] {
            let forward = x_wall.extended(false).map(|slot| slot.map(transpose));
            assert_eq!(y_wall.extended_y(false), forward);
            let mut backward = forward;
            backward.reverse();
            assert_eq!(y_wall.extended_y(true), backward);
        }
    }

    /// The hook pair — the corners still to be touched when only two are left —
    /// runs along the axis the schedule is named for: `BL`/`BR` share a row,
    /// `TR`/`BR` share a column.
    #[test]
    fn extended_schedule_hook_runs_along_its_named_axis() {
        let tail = |cx: CXSchedule, after: usize| -> Vec<Corner> {
            cx.extended(false)[after..]
                .iter()
                .flatten()
                .copied()
                .collect()
        };
        assert_eq!(
            tail(CXSchedule::Horizontal, 3),
            vec![Corner::BL, Corner::BR]
        );
        assert_eq!(tail(CXSchedule::Vertical, 2), vec![Corner::TR, Corner::BR]);
    }

    #[test]
    fn test_5slot_chunk_builds() {
        let tile = make_cube_tile(
            Basis::Z,
            IVec2::new(2, 2),
            &Corner::ALL,
            true,
            LayerSchedule::Padded,
        );
        let patch = Patch::new(vec![tile]);
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        assert_eq!(chunk.flows.len(), 2);
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_surface_code_measure_basis() {
        // d=3 layout: verified against tile-grid checkerboard formula
        assert_eq!(checkerboard_basis(IVec2::new(2, 4)), Basis::Z);
        assert_eq!(checkerboard_basis(IVec2::new(4, 4)), Basis::X);
        assert_eq!(checkerboard_basis(IVec2::new(2, 2)), Basis::X);
        assert_eq!(checkerboard_basis(IVec2::new(4, 2)), Basis::Z);
        // d=5 boundary qubits
        assert_eq!(checkerboard_basis(IVec2::new(2, 8)), Basis::Z);
        assert_eq!(checkerboard_basis(IVec2::new(0, 2)), Basis::Z);
    }

    #[test]
    fn test_make_normal_surface_code_patch_d3() {
        let patch = make_normal_surface_code_patch(3, Basis::X);
        assert_eq!(patch.data_set().len(), 9);
        assert_eq!(patch.tiles().len(), 8);
        assert!(
            patch
                .tiles()
                .iter()
                .all(|tile| tile.data_slots().len() == 4)
        );

        // Bulk round should have valid flows.
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_make_rectangular_surface_code_patch_degenerate() {
        // Degenerate boundaries (Y-basis style): top=Z, bot=X, left=X, right=Z
        let boundaries = EdgeBases {
            top: Basis::Z,
            bottom: Basis::X,
            left: Basis::X,
            right: Basis::Z,
        };
        let patch = make_rectangular_surface_code_patch(3, 3, &boundaries, Basis::X, false);
        // Should produce a valid patch with fewer tiles than the standard layout.
        assert!(!patch.tiles().is_empty());
        assert!(
            patch
                .tiles()
                .iter()
                .all(|tile| tile.data_slots().len() == 4)
        );
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }

    #[test]
    fn test_make_rectangular_surface_code_patch_non_square() {
        let boundaries = EdgeBases::standard(Basis::Z);
        let patch = make_rectangular_surface_code_patch(5, 3, &boundaries, Basis::X, false);

        assert!(!patch.tiles().is_empty());
        assert!(patch.data_set().contains(&IVec2::new(9, 5)));
        assert!(!patch.data_set().contains(&IVec2::new(5, 9)));
        assert!(
            patch
                .tiles()
                .iter()
                .all(|tile| tile.data_slots().len() == 4)
        );
        let chunk = make_surface_code_chunk(&patch, None, None).unwrap();
        chunk.verify_flows(None, None).unwrap();
    }
}
