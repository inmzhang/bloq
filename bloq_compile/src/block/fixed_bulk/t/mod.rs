//! T-block compilation: magic state cultivation with lattice surgery (MSC-LS,
//! arXiv:2510.24615).
//!
//! This module ships the Steane stage library ([`steane`]) and the geometry
//! machinery below. The escape template and compiler/lowering wiring build on
//! both.
//!
//! The msc-ls reference hardcodes one geometry: Steane patch north of the
//! surface patch, ZZ merge across the surface top row. Bloq needs the Steane
//! patch on any of the block's four free sides, under either patch
//! orientation — 8 combinations, but only two circuit families, keyed by the
//! basis of the *seam stabilizers* (the flip of the merge-side face's hosted
//! two-weight basis):
//!
//! - seam stabilizers Z-type (merge-side face hosts X two-weights, as msc-ls's
//!   patch top edge does — `lattice_surgery_error_detection.py:68`) → the
//!   msc-ls ZZ construction verbatim;
//! - seam stabilizers X-type → its X↔Z dual (XX merge, destructive MZ
//!   readout).
//!
//! Everything else is a rigid coordinate transform: [`SurgeryLayout::map`]
//! carries the base construction's block-local coordinates onto the chosen
//! side with the one D₄ element whose net checkerboard parity is even,
//! and the dual helpers swap stabilizer/readout bases for the
//! X-type seam family.

pub(crate) mod escape;
pub(crate) mod steane;

use bloq_circuit::{Chunk, ChunkOrLoop, CoordCircuit, GateType, PauliMap};
use bloq_graph::{Basis, Direction, Pauli};
use glam::IVec2;

use crate::CompileError;
use crate::block::LoweringTemplate;
use crate::block::fixed_bulk::observable::logical_line_operator;
use crate::block::fixed_bulk::utils::reset_gate;
use crate::block::fixed_bulk::utils::{TileFlow, make_normal_surface_code_patch, tile_flow};
use crate::block::gateway::{GatewayEntry, LocalStabilizer, ObservableGateway};
use crate::signature::Connectivity;

/// A qubit in msc-ls coordinates (as the reference source writes them).
pub(crate) type MscQubit = (i32, i32);

// ==========================================================================
// Steane geometry ground truth (STE:12–20), shared by both stages: the
// cultivation/escape seam (`compose_instance_seam`) closes only if the two
// templates agree on these tables, so they are defined exactly once.
// ==========================================================================

/// Steane data qubits in msc-ls coordinates, indexed 0..=6 (STE:12) — the
/// positions produced by injection and consumed by the escape merge.
pub(crate) const STEANE_FINAL: [MscQubit; 7] = [
    (3, 13),
    (1, 15),
    (6, 12),
    (5, 15),
    (2, 12),
    (3, 15),
    (2, 10),
];

/// The three Steane stabilizer faces (data-qubit index sets); each face
/// supports one X and one Z stabilizer. Stabilizer tables are indexed
/// `2 * face + {X: 0, Z: 1}`.
pub(crate) const FACES: [[usize; 4]; 3] = [[0, 1, 4, 5], [0, 2, 3, 5], [0, 2, 4, 6]];

/// msc-ls row of the surgery seam. Subtracting it puts the escaped patch at
/// bloq's standard `[1, 2d-1]²` and the Steane spill at negative `y`.
const MSC_SEAM_ROW: i32 = 16;

/// Placement and basis parameters of one T-block surgery variant.
///
/// The base construction (side `-y`, Z-type merge) is authored once in
/// block-local coordinates: escaped patch on `[1, 2d-1]²`, surgery seam on the
/// `y = 0` row, Steane spill at negative `y`. `map` and the dual helpers
/// transport it to the other seven variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SurgeryLayout {
    /// The spatial side holding the Steane spill (must be a free neighbor).
    pub(crate) side: Direction,
    /// Basis of the surgery seam stabilizers — the flip of the escaped
    /// patch's hosted two-weight basis on `side`. `Z` is the msc-ls ZZ merge
    /// (base family); `X` selects the dual circuit family.
    pub(crate) merge_basis: Basis,
    /// Final patch distance `d`; fixes the rotation center `(d, d)`.
    pub(crate) distance: u32,
}

impl SurgeryLayout {
    /// Deterministic side pick order for placement.
    pub(crate) const SIDE_PRIORITY: [Direction; 4] = [
        Direction::YMINUS,
        Direction::XPLUS,
        Direction::YPLUS,
        Direction::XMINUS,
    ];

    /// Build a layout from the chosen spill side and the escaped patch's
    /// boundary orientation (`x_basis`/`y_basis`: stabilizer basis of the
    /// `±x`/`±y` boundaries, as inherited from the +Z peer).
    pub(crate) fn new(side: Direction, x_basis: Basis, y_basis: Basis, distance: u32) -> Self {
        let side_face_basis = match side {
            Direction::XPLUS | Direction::XMINUS => x_basis,
            Direction::YPLUS | Direction::YMINUS => y_basis,
            Direction::ZPLUS | Direction::ZMINUS => {
                unreachable!("surgery side must be spatial")
            }
        };
        // The Z logical lies along a face hosting X two-weights, so merging
        // across it measures Z_L⊗Z_L via Z seam stabilizers: the seam basis is
        // the flip of the face's hosted basis (msc-ls's patch top edge hosts X
        // two-weights and merges with Z, `lattice_surgery_error_detection.py:68`).
        let merge_basis = side_face_basis.flip();
        Self {
            side,
            merge_basis,
            distance,
        }
    }

    /// Hosted stabilizer basis of the escaped patch's top/bottom edges — the
    /// `top_basis` argument for `make_normal_surface_code_patch`. The merge
    /// side hosts `merge_basis.flip()`; the y faces carry that basis when the
    /// spill side is ±y and its flip otherwise.
    pub(crate) fn top_basis(&self) -> Basis {
        match self.side {
            Direction::YPLUS | Direction::YMINUS => self.merge_basis.flip(),
            Direction::XPLUS | Direction::XMINUS => self.merge_basis,
            Direction::ZPLUS | Direction::ZMINUS => {
                unreachable!("surgery side must be spatial")
            }
        }
    }

    /// Surgery-time patch distance: fixed by rule, 3 at d = 3
    /// and 5 above; downward expansion to `distance` first occurs at d = 7.
    pub(crate) fn intermediate_distance(&self) -> u32 {
        if self.distance == 3 { 3 } else { 5 }
    }

    /// Transport a base-variant block-local coordinate to this variant: the
    /// unique D₄ element about the patch center `(d, d)` that takes the base
    /// `-y` seam row to `self.side` *and* has even net checkerboard parity.
    /// The bulk coloring is global — on even coordinates about
    /// an odd-`d` center, rotations by 0°/180° and the diagonal reflections
    /// preserve it while 90°/270° and the axis mirrors flip it, and the X↔Z
    /// dual flips it by definition — so each `(side, family)` pair pins
    /// exactly one transform: base variants get the parity-preserving
    /// elements, dual variants the parity-flipping ones.
    pub(crate) fn map(&self, base: IVec2) -> IVec2 {
        let center = IVec2::splat(self.distance as i32);
        let v = base - center;
        let image = match (self.side, self.is_dual()) {
            (Direction::YMINUS, false) => v,
            (Direction::YPLUS, false) => -v,
            (Direction::XMINUS, false) => IVec2::new(v.y, v.x),
            (Direction::XPLUS, false) => IVec2::new(-v.y, -v.x),
            (Direction::YMINUS, true) => IVec2::new(-v.x, v.y),
            (Direction::YPLUS, true) => IVec2::new(v.x, -v.y),
            (Direction::XPLUS, true) => v.perp(),
            (Direction::XMINUS, true) => -v.perp(),
            (Direction::ZPLUS | Direction::ZMINUS, _) => {
                unreachable!("surgery side must be spatial")
            }
        };
        center + image
    }

    /// [`Self::map`] for a coordinate authored in msc-ls space: mirror the
    /// columns and shift the surgery seam onto the block-local `y = 0` row,
    /// then apply the variant transform. msc-ls draws its lattice y-down
    /// while bloq's grid is y-up, so a plain shift would land every surface
    /// ancilla on the wrong checkerboard color; the x-mirror composed with
    /// the shift fixes the coloring while keeping the Steane spill on the
    /// base `-y` side. Both the cultivation and escape
    /// templates author their tables in msc-ls space, so the shared embedding
    /// lives here.
    pub(crate) fn map_msc(&self, (x, y): MscQubit) -> IVec2 {
        self.map(IVec2::new(2 * self.distance as i32 - x, y - MSC_SEAM_ROW))
    }

    /// Whether this variant is the X↔Z dual of the base construction.
    pub(crate) fn is_dual(&self) -> bool {
        self.merge_basis == Basis::X
    }

    /// Dual-aware stabilizer basis: the base construction's `base` under this
    /// variant's circuit family.
    pub(crate) fn basis(&self, base: Basis) -> Basis {
        if self.is_dual() { base.flip() } else { base }
    }

    /// Dual-aware Pauli (`X ↔ Z`, `Y` fixed).
    pub(crate) fn pauli(&self, base: Pauli) -> Pauli {
        if self.is_dual() { base.flip() } else { base }
    }

    /// Dual-aware reset gate (`RX ↔ R`).
    pub(crate) fn reset_gate(&self, base: Basis) -> GateType {
        reset_gate(self.basis(base))
    }

    /// Dual-aware CX operand order: conjugating CX by transversal H swaps
    /// control and target, so the dual family reverses each pair.
    pub(crate) fn cx_operands(&self, control: IVec2, target: IVec2) -> [IVec2; 2] {
        if self.is_dual() {
            [target, control]
        } else {
            [control, target]
        }
    }
}

/// Compile the compact T-source preparation used by T-proxy experiments.
///
/// The product-state resets and the single physical `T` prepare the logical
/// state; the final stabilizer `MPP` creates the patch boundary flows. The MPP
/// results are intentionally not treated as deterministic inside this chunk:
/// composing them with the neighbouring cube closes the detectors.
pub(crate) fn prepare_t_with_mpps(
    layout: &SurgeryLayout,
) -> Result<LoweringTemplate, CompileError> {
    let patch = make_normal_surface_code_patch(layout.distance, layout.top_basis());
    let data = patch.data_set();
    let logical_x = logical_line_operator(layout.distance, Basis::X, layout.top_basis(), Pauli::X);
    let corner = IVec2::splat(layout.distance as i32);

    let mut circuit = CoordCircuit::new();
    let plus: crate::FxSet<_> = logical_x.iter().map(|(q, _)| *q).collect();
    circuit.do_gate(
        reset_gate(Basis::Z),
        data.iter().copied().filter(|q| !plus.contains(q)),
    )?;
    circuit.do_gate(reset_gate(Basis::X), plus.iter().copied())?;
    circuit.tick();
    circuit.do_gate(GateType::T, [corner])?;
    circuit.tick();

    let measurements = circuit.measure_pauli_products(
        patch
            .tiles()
            .iter()
            .map(super::super::patch::Tile::pauli_map),
    )?;
    let flows = patch
        .tiles()
        .iter()
        .zip(measurements)
        .map(|(tile, measurement)| tile_flow(tile, TileFlow::Create, [measurement]))
        .collect();
    let gateway = prepare_t_observable_gateway(layout.distance, layout.top_basis());
    LoweringTemplate::from_chunks(
        vec![ChunkOrLoop::Single(Box::new(Chunk { circuit, flows }))],
        gateway,
    )
}

fn prepare_t_observable_gateway(distance: u32, top_basis: Basis) -> ObservableGateway {
    let connectivity = Connectivity::ISOLATED.with_pipe(Direction::ZPLUS);
    let mut gateway = ObservableGateway::new();
    for (basis, pauli) in [(Basis::X, Pauli::X), (Basis::Z, Pauli::Z)] {
        gateway.insert(
            LocalStabilizer::new(pauli, connectivity),
            GatewayEntry {
                measurements: Vec::new(),
                operator_in: PauliMap::empty(),
                operator_out: logical_line_operator(distance, basis, top_basis, pauli),
            },
        );
    }
    gateway
}

/// Assert every tick of the circuit holds a single op kind (reset /
/// single-qubit / two-qubit / measurement) — the alignment discipline the
/// hand-ported msc-ls circuits must uphold so the editor's circuit view shows
/// them without sub-moment splitting.
#[cfg(test)]
pub(crate) fn assert_ticks_are_kind_homogeneous(stage: &str, circuit: &bloq_circuit::CoordCircuit) {
    use bloq_circuit::Op;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kind {
        Reset,
        Single,
        Double,
        Measure,
    }

    let body = circuit
        .body(circuit.entry_body())
        .expect("hand-ported circuits have a flat entry body");
    let mut current: Option<Kind> = None;
    for op in body.ops() {
        let kind = match op {
            Op::Tick => {
                current = None;
                continue;
            }
            Op::Gate { gate, .. } if gate.is_reset() => Kind::Reset,
            Op::Gate { gate, .. } if gate.is_two_qubit_gate() => Kind::Double,
            Op::Gate { .. } => Kind::Single,
            Op::Measure { .. } | Op::MPP { .. } => Kind::Measure,
            Op::Repeat { .. }
            | Op::ConditionalPauli(_)
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. } => continue,
        };
        match current {
            None => current = Some(kind),
            Some(existing) => {
                assert_eq!(existing, kind, "{stage}: mixed op kinds within a tick");
            }
        }
    }
}

/// A layout of the requested family on the requested side: the base (ZZ
/// merge) family needs the side face to host X two-weights.
#[cfg(test)]
pub(crate) fn family_layout(side: Direction, dual: bool, distance: u32) -> SurgeryLayout {
    let face = if dual { Basis::Z } else { Basis::X };
    let (x_basis, y_basis) = match side {
        Direction::XPLUS | Direction::XMINUS => (face, face.flip()),
        _ => (face.flip(), face),
    };
    SurgeryLayout::new(side, x_basis, y_basis, distance)
}

#[cfg(test)]
mod tests {
    use bloq_circuit::Op;
    use rstest::rstest;

    use super::*;
    use crate::block::fixed_bulk::utils::checkerboard_basis;

    const SIDES: [Direction; 4] = [
        Direction::YMINUS,
        Direction::XPLUS,
        Direction::YPLUS,
        Direction::XMINUS,
    ];

    fn base_layout(side: Direction, distance: u32) -> SurgeryLayout {
        family_layout(side, false, distance)
    }

    #[rstest]
    fn merge_basis_is_flip_of_side_face_basis(
        #[values(Basis::X, Basis::Z)] x_basis: Basis,
        #[values(
            Direction::XPLUS,
            Direction::XMINUS,
            Direction::YPLUS,
            Direction::YMINUS
        )]
        side: Direction,
    ) {
        let y_basis = x_basis.flip();
        let layout = SurgeryLayout::new(side, x_basis, y_basis, 3);
        let side_face = match side {
            Direction::XPLUS | Direction::XMINUS => x_basis,
            _ => y_basis,
        };
        assert_eq!(layout.merge_basis, side_face.flip());
        assert_eq!(layout.is_dual(), side_face == Basis::Z);
        // The final patch's top/bottom hosted basis matches the y face.
        assert_eq!(layout.top_basis(), y_basis);
    }

    #[test]
    fn intermediate_distance_is_fixed_by_rule() {
        for (d, d_int) in [(3, 3), (5, 5), (7, 5), (9, 5)] {
            assert_eq!(
                base_layout(Direction::YMINUS, d).intermediate_distance(),
                d_int
            );
        }
    }

    #[test]
    fn compact_preparation_has_one_t_and_one_stabilizer_mpp() {
        let template = prepare_t_with_mpps(&base_layout(Direction::YMINUS, 3))
            .expect("compact T preparation compiles");
        let body = template
            .program_template
            .circuit
            .body(template.program_template.circuit.entry_body())
            .expect("compact T body exists");
        assert!(matches!(
            body.ops(),
            [
                Op::Gate {
                    gate: GateType::RZ,
                    ..
                },
                Op::Gate {
                    gate: GateType::RX,
                    ..
                },
                Op::Tick,
                Op::Gate {
                    gate: GateType::T,
                    ..
                },
                Op::Tick,
                Op::MPP { .. },
            ]
        ));
    }

    #[test]
    fn base_side_map_is_identity() {
        let layout = base_layout(Direction::YMINUS, 5);
        for coord in [IVec2::new(1, 1), IVec2::new(3, -5), IVec2::new(9, 9)] {
            assert_eq!(layout.map(coord), coord);
        }
    }

    #[test]
    fn map_matches_pinned_d4_table() {
        // Sample point (3, 1) about center (3, 3) at d = 3: v = (0, -2).
        let p = IVec2::new(3, 1);
        let expected = [
            (Direction::YMINUS, false, IVec2::new(3, 1)),
            (Direction::YPLUS, false, IVec2::new(3, 5)),
            (Direction::XMINUS, false, IVec2::new(1, 3)),
            (Direction::XPLUS, false, IVec2::new(5, 3)),
            (Direction::YMINUS, true, IVec2::new(3, 1)),
            (Direction::YPLUS, true, IVec2::new(3, 5)),
            (Direction::XPLUS, true, IVec2::new(5, 3)),
            (Direction::XMINUS, true, IVec2::new(1, 3)),
        ];
        for (side, dual, image) in expected {
            assert_eq!(family_layout(side, dual, 3).map(p), image, "{side} {dual}");
        }
        // An off-axis point distinguishes the reflections from the rotations:
        // (2, 1) about (3, 3): v = (-1, -2).
        let p = IVec2::new(2, 1);
        let expected = [
            (Direction::YMINUS, false, IVec2::new(2, 1)),
            (Direction::YPLUS, false, IVec2::new(4, 5)),
            (Direction::XMINUS, false, IVec2::new(1, 2)), // main diagonal
            (Direction::XPLUS, false, IVec2::new(5, 4)),  // anti-diagonal
            (Direction::YMINUS, true, IVec2::new(4, 1)),  // x-mirror
            (Direction::YPLUS, true, IVec2::new(2, 5)),   // y-mirror
            (Direction::XPLUS, true, IVec2::new(5, 2)),   // 90°
            (Direction::XMINUS, true, IVec2::new(1, 4)),  // 270°
        ];
        for (side, dual, image) in expected {
            assert_eq!(family_layout(side, dual, 3).map(p), image, "{side} {dual}");
        }
    }

    #[rstest]
    fn map_rotates_seam_row_to_chosen_side(
        #[values(3, 5, 7)] d: u32,
        #[values(false, true)] dual: bool,
    ) {
        // The base seam row y = 0 must land on the edge row facing `side`.
        let d = d as i32;
        let seam = IVec2::new(1, 0);
        let layout = |side| family_layout(side, dual, d as u32);
        assert_eq!(layout(Direction::YMINUS).map(seam).y, 0);
        assert_eq!(layout(Direction::XPLUS).map(seam).x, 2 * d);
        assert_eq!(layout(Direction::YPLUS).map(seam).y, 2 * d);
        assert_eq!(layout(Direction::XMINUS).map(seam).x, 0);
    }

    #[rstest]
    fn map_preserves_patch_center_adjacency_and_sublattice(
        #[values(
            Direction::XPLUS,
            Direction::XMINUS,
            Direction::YPLUS,
            Direction::YMINUS
        )]
        side: Direction,
        #[values(false, true)] dual: bool,
        #[values(3, 6)] d: u32,
    ) {
        let layout = family_layout(side, dual, d);
        let center = IVec2::splat(d as i32);
        assert_eq!(layout.map(center), center);

        for x in -6..=(2 * d as i32 + 2) {
            for y in -6..=(2 * d as i32 + 2) {
                let p = IVec2::new(x, y);
                let q = layout.map(p);
                // Rigid map: distances to the center and coordinate-sum parity
                // (the data/measure sublattice) are preserved.
                assert_eq!((q - center).length_squared(), (p - center).length_squared());
                assert_eq!((q.x + q.y).rem_euclid(2), (p.x + p.y).rem_euclid(2));
                // Unit lattice steps stay unit steps.
                let step = layout.map(p + IVec2::X) - q;
                assert_eq!(step.length_squared(), 1);
            }
        }
    }

    #[rstest]
    fn map_is_a_bijection_on_the_patch(
        #[values(
            Direction::XPLUS,
            Direction::XMINUS,
            Direction::YPLUS,
            Direction::YMINUS
        )]
        side: Direction,
        #[values(false, true)] dual: bool,
    ) {
        let d = 5i32;
        let layout = family_layout(side, dual, d as u32);
        let images: crate::FxSet<IVec2> = (1..=2 * d - 1)
            .flat_map(|x| (1..=2 * d - 1).map(move |y| layout.map(IVec2::new(x, y))))
            .collect();
        assert_eq!(images.len(), ((2 * d - 1) * (2 * d - 1)) as usize);
        for q in images {
            assert!((1..=2 * d - 1).contains(&q.x) && (1..=2 * d - 1).contains(&q.y));
        }
    }

    /// The legality rule: composed with the dual basis flip, every
    /// variant transform preserves the fixed checkerboard —
    /// `checkerboard_basis(map(p)) == layout.basis(checkerboard_basis(p))`.
    #[rstest]
    fn map_dual_composition_preserves_checkerboard(
        #[values(3, 5, 7)] d: u32,
        #[values(false, true)] dual: bool,
    ) {
        for side in SIDES {
            let layout = family_layout(side, dual, d);
            for x in 0..=(d as i32) {
                for y in 0..=(d as i32) {
                    let m = IVec2::new(2 * x, 2 * y);
                    assert_eq!(
                        checkerboard_basis(layout.map(m)),
                        layout.basis(checkerboard_basis(m)),
                        "{side} dual={dual} at {m}"
                    );
                }
            }
        }
    }

    /// The mirrored msc embedding lands the base variant's surface
    /// data qubits on the standard patch with every ancilla on its native
    /// checkerboard color.
    #[test]
    fn map_msc_mirrors_columns_onto_the_patch() {
        for d in [3, 5, 7] {
            let layout = base_layout(Direction::YMINUS, d);
            let d = d as i32;
            // msc surface data (1 + 2j, 17 + 2i) → (2d − 1 − 2j, 1 + 2i).
            assert_eq!(layout.map_msc((1, 17)), IVec2::new(2 * d - 1, 1));
            // msc seam row y = 16 → block-local y = 0.
            assert_eq!(layout.map_msc((4, 16)), IVec2::new(2 * d - 4, 0));
        }
    }

    #[test]
    fn dual_helpers_swap_x_and_z() {
        let base = base_layout(Direction::YMINUS, 3);
        let dual = family_layout(Direction::YMINUS, true, 3);
        assert!(!base.is_dual());
        assert!(dual.is_dual());

        assert_eq!(base.basis(Basis::Z), Basis::Z);
        assert_eq!(dual.basis(Basis::Z), Basis::X);
        assert_eq!(dual.pauli(Pauli::X), Pauli::Z);
        assert_eq!(dual.pauli(Pauli::Y), Pauli::Y);
        assert_eq!(base.reset_gate(Basis::Z), GateType::RZ);
        assert_eq!(dual.reset_gate(Basis::Z), GateType::RX);

        let (a, b) = (IVec2::new(1, 1), IVec2::new(1, 3));
        assert_eq!(base.cx_operands(a, b), [a, b]);
        assert_eq!(dual.cx_operands(a, b), [b, a]);
    }
}
