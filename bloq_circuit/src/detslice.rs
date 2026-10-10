//! Reverse region tracking: what stabilizer each detector or logical observable
//! represents at every moment of a straight-line circuit.
//!
//! A native reimplementation of Stim's `SparseUnsignedRevFrameTracker`
//! (`stim/src/stim/simulators/sparse_rev_frame_tracker.*`) driven the way
//! `DetectorSliceSet` (`stim/src/stim/diagram/detector_slice/`) drives it. Each
//! `undo_*` below mirrors its Stim counterpart; only the packaging differs —
//! sparse per-coordinate maps instead of dense qubit-indexed vectors, side-table
//! detectors instead of `DETECTOR` ops, and pre-segmented input instead of
//! `TICK`-delimited moments. `bloq_ir::detslice` layers program-tape slicing
//! over a whole [`Bloq`] on top of this.
//!
//! A tracked region is the Pauli operator that, propagated to a given moment,
//! has the same value as its detector or observable. The walk runs **backward**,
//! tracking per qubit which region ids hold `X` and/or `Z` sensitivity there
//! (`Y` = both). Each id lives in a sparse [`DetSet`] (a sorted set with
//! symmetric-difference merge — Stim's `SparseXorVec`), so a gate costs
//! `O(region weight)` set XORs rather than a dense conjugation.
//!
//! Before the walk, every record-backed region xors its id into `rec_bits[m]`
//! for each measurement `m` it sums; that entry discharges only once the walk
//! reaches the measurement. Positioned Pauli seeds instead enter at their exact
//! segment boundary.
//!
//! A gauge anticommutation records a [`RegionBreak`] and strips the region from
//! every set so it stops propagating (Stim's `fail_on_anticommute = false`
//! path). Snapshots are taken *before* undoing each segment, so `slices[k]` is
//! the state at the **end of segment k**.

use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use thiserror::Error;

use crate::{GateType, Op, Pauli, PauliBasis};

/// One qubit of a tracked region and the Pauli it carries there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionTerm {
    /// Qubit coordinate.
    pub qubit: IVec2,
    /// Pauli carried on the qubit.
    pub pauli: PauliBasis,
}

/// A detector to track: an opaque id and the measurement ids whose parity it
/// sums. Ids live in the input op stream's id space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceDetector {
    /// Opaque detector or observable id.
    pub id: u32,
    /// Measurement ids summed by the region.
    pub measurements: Vec<u32>,
}

/// A Pauli contribution to a tracked region at an exact segment boundary.
///
/// Boundary `0` is before the first segment and boundary `segments.len()` is
/// after the last. This models positioned Pauli-target
/// `OBSERVABLE_INCLUDE`s; measurement-record contributions remain on
/// [`SliceDetector::measurements`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceRegionSeed {
    /// Opaque detector or observable id.
    pub id: u32,
    /// Boundary where the contribution enters.
    pub boundary: usize,
    /// Pauli contribution.
    pub terms: Vec<RegionTerm>,
}

/// Why a region stopped propagating backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionBreakKind {
    /// Support anticommuted with the basis a reset prepares.
    AnticommutesWithReset,
    /// Support anticommuted with the basis a measurement measured.
    AnticommutesWithMeasurement,
    /// `X`/`Y` support survived to the implicit `|0>` at the start of time.
    AnticommutesWithStart,
}

/// A recorded anticommutation: the region could not continue past this point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionBreak {
    /// Region that stopped propagating.
    pub detector: u32,
    /// Segment whose undo revealed the break.
    pub segment: usize,
    /// Qubit where anticommutation occurred.
    pub qubit: IVec2,
    /// Cause of the break.
    pub kind: RegionBreakKind,
}

/// The region terms one detector's region covers at one moment.
pub type DetectorRegion = (u32, Vec<RegionTerm>);

/// The full slice table for a segmented straight-line circuit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetectorSlices {
    /// `slices[k]` is the regions alive at the end of segment `k`, sorted by
    /// detector id; each region's terms sorted by `(qubit.x, qubit.y)`.
    /// Detectors with empty regions are omitted.
    pub slices: Vec<Vec<DetectorRegion>>,
    /// Every anticommutation, sorted by `(segment, detector, qubit)`.
    pub breaks: Vec<RegionBreak>,
}

/// A failure preparing the op stream for reverse tracking.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DetsliceError {
    /// The circuit still contains a repeat operation.
    #[error(
        "detector-slice tracking requires a straight-line circuit; flatten REPEAT blocks first"
    )]
    RepeatNotFlattened,
    #[error(
        "region seed boundary {boundary} is outside the {segments}-segment circuit (maximum {segments})"
    )]
    /// A positioned seed names a boundary outside the circuit.
    SeedBoundaryOutOfRange {
        /// Requested boundary.
        boundary: usize,
        /// Number of circuit segments.
        segments: usize,
    },
}
/// Track detecting and logical regions with positioned Pauli contributions.
///
/// `seeds` are XORed into the reverse frame at exact boundaries between
/// segments. A seed at boundary `k + 1` is therefore visible in the snapshot at
/// the end of segment `k`, then propagates backward through that segment.
///
/// # Errors
///
/// Returns [`DetsliceError::RepeatNotFlattened`] for nested repeats or
/// [`DetsliceError::SeedBoundaryOutOfRange`] for a boundary past the circuit.
pub fn detector_slices_with_seeds(
    segments: &[&[Op]],
    detectors: &[SliceDetector],
    seeds: &[SliceRegionSeed],
) -> Result<DetectorSlices, DetsliceError> {
    if segments
        .iter()
        .flat_map(|segment| segment.iter())
        .any(|op| matches!(op, Op::Repeat { .. }))
    {
        return Err(DetsliceError::RepeatNotFlattened);
    }

    let mut seeds_by_boundary = vec![Vec::new(); segments.len() + 1];
    for seed in seeds {
        let Some(slot) = seeds_by_boundary.get_mut(seed.boundary) else {
            return Err(DetsliceError::SeedBoundaryOutOfRange {
                boundary: seed.boundary,
                segments: segments.len(),
            });
        };
        slot.push(seed);
    }

    let mut tracker = Tracker::default();
    for detector in detectors {
        for &measurement in &detector.measurements {
            tracker
                .rec_bits
                .entry(measurement)
                .or_default()
                .xor_id(detector.id);
        }
    }

    let mut slices = vec![Vec::new(); segments.len()];
    for k in (0..segments.len()).rev() {
        for seed in &seeds_by_boundary[k + 1] {
            tracker.xor_region(seed.id, &seed.terms);
        }
        slices[k] = tracker.snapshot();
        for op in segments[k].iter().rev() {
            tracker.undo_op(op);
        }
        tracker.process_anticommutations(k);
    }
    for seed in &seeds_by_boundary[0] {
        tracker.xor_region(seed.id, &seed.terms);
    }
    tracker.undo_start_of_circuit();

    let mut breaks = tracker.broken;
    breaks.sort_by_key(|brk| (brk.segment, brk.detector, brk.qubit.x, brk.qubit.y));
    Ok(DetectorSlices { slices, breaks })
}

/// A sorted set of detector ids with symmetric-difference merge — the sparse
/// `X`/`Z` frame column, mirroring Stim's `SparseXorVec<DemTarget>`.
///
/// Inline capacity 4: a detecting region rarely touches more than a handful of
/// detectors on any one qubit, so the common case stays off the heap.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct DetSet(SmallVec<[u32; 4]>);

impl DetSet {
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, id: u32) -> bool {
        self.0.binary_search(&id).is_ok()
    }

    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().copied()
    }

    /// Toggle a single id (add if absent, remove if present).
    fn xor_id(&mut self, id: u32) {
        match self.0.binary_search(&id) {
            Ok(index) => {
                self.0.remove(index);
            }
            Err(index) => self.0.insert(index, id),
        }
    }

    /// Remove `id` if present.
    fn remove(&mut self, id: u32) {
        if let Ok(index) = self.0.binary_search(&id) {
            self.0.remove(index);
        }
    }

    /// XOR `other` in: keep ids present in exactly one of the two sets.
    fn xor_merge(&mut self, other: &DetSet) {
        if other.is_empty() {
            return;
        }
        let mut merged = SmallVec::with_capacity(self.0.len() + other.0.len());
        let (mut i, mut j) = (0, 0);
        let (a, b) = (&self.0, &other.0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    merged.push(a[i]);
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    merged.push(b[j]);
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    i += 1;
                    j += 1;
                }
            }
        }
        merged.extend_from_slice(&a[i..]);
        merged.extend_from_slice(&b[j..]);
        self.0 = merged;
    }
}

/// Ids in exactly one of two sorted sets — the gauge basis for `Y` measurements
/// and resets (`handle_xor_gauge`).
fn symmetric_difference(a: &DetSet, b: &DetSet) -> SmallVec<[u32; 4]> {
    let mut out = a.clone();
    out.xor_merge(b);
    out.0
}

/// A pending gauge anticommutation, collected during a segment's undo and
/// applied at the segment boundary (matching Stim, which strips only at ticks).
struct Pending {
    detector: u32,
    qubit: IVec2,
    kind: RegionBreakKind,
}

#[derive(Default)]
struct Tracker {
    /// Per qubit, the detectors with `X` support there.
    xs: FxHashMap<IVec2, DetSet>,
    /// Per qubit, the detectors with `Z` support there.
    zs: FxHashMap<IVec2, DetSet>,
    /// Per measurement id, the detectors whose parity still awaits it.
    rec_bits: FxHashMap<u32, DetSet>,
    /// Anticommutations pending strip at the current segment boundary.
    pending: Vec<Pending>,
    broken: Vec<RegionBreak>,
}

impl Tracker {
    /// Take a qubit's `(X, Z)` columns out of the maps, so per-qubit gate rules
    /// can rewrite them without cross-map borrow gymnastics.
    fn take_frame(&mut self, qubit: IVec2) -> (DetSet, DetSet) {
        (
            self.xs.remove(&qubit).unwrap_or_default(),
            self.zs.remove(&qubit).unwrap_or_default(),
        )
    }

    /// Put a qubit's columns back, dropping empty ones to keep the maps sparse
    /// (`contains` and snapshotting assume a present entry is non-empty).
    fn put_frame(&mut self, qubit: IVec2, x: DetSet, z: DetSet) {
        if x.is_empty() {
            self.xs.remove(&qubit);
        } else {
            self.xs.insert(qubit, x);
        }
        if z.is_empty() {
            self.zs.remove(&qubit);
        } else {
            self.zs.insert(qubit, z);
        }
    }

    fn record_gauge(
        &mut self,
        detectors: impl IntoIterator<Item = u32>,
        qubit: IVec2,
        kind: RegionBreakKind,
    ) {
        for detector in detectors {
            self.pending.push(Pending {
                detector,
                qubit,
                kind,
            });
        }
    }

    /// XOR a positioned Pauli contribution into one tracked region.
    fn xor_region(&mut self, id: u32, terms: &[RegionTerm]) {
        for term in terms {
            let (mut x, mut z) = self.take_frame(term.qubit);
            match term.pauli {
                PauliBasis::X => x.xor_id(id),
                PauliBasis::Y => {
                    x.xor_id(id);
                    z.xor_id(id);
                }
                PauliBasis::Z => z.xor_id(id),
            }
            self.put_frame(term.qubit, x, z);
        }
    }

    fn undo_op(&mut self, op: &Op) {
        match op {
            Op::Gate { gate, qubits } => self.undo_gate(*gate, qubits),
            Op::Measure {
                basis,
                qubits,
                measurements,
                ..
            } => self.undo_measure(*basis, qubits, measurements),
            Op::MPP {
                products,
                measurements,
            } => self.undo_mpp(products, measurements),
            Op::ConditionalPauli(corrections) => {
                for correction in corrections {
                    self.undo_conditional_pauli(
                        correction.target,
                        correction.pauli,
                        correction.control,
                    );
                }
            }
            // Ticks and noise carry no region; repeats were rejected up front.
            Op::Tick
            | Op::Depolarize1 { .. }
            | Op::Depolarize2 { .. }
            | Op::PauliError { .. }
            | Op::Repeat { .. } => {}
        }
    }

    /// A region anticommuting with the correction picks up its conditional
    /// sign, so xor that region into the control's record dependency — an
    /// explicit detector over the same record then cancels it.
    fn undo_conditional_pauli(&mut self, target: IVec2, pauli: PauliBasis, control: u32) {
        let x = self.xs.get(&target).cloned().unwrap_or_default();
        let z = self.zs.get(&target).cloned().unwrap_or_default();
        let anticommuting = match pauli {
            PauliBasis::X => z,
            PauliBasis::Z => x,
            PauliBasis::Y => DetSet(symmetric_difference(&x, &z)),
        };
        if anticommuting.is_empty() {
            return;
        }
        self.rec_bits
            .entry(control)
            .or_default()
            .xor_merge(&anticommuting);
    }

    fn undo_gate(&mut self, gate: GateType, qubits: &[IVec2]) {
        if gate.is_reset() {
            for &q in qubits {
                self.undo_reset(gate, q);
            }
        } else if gate.is_two_qubit_gate() {
            for pair in qubits.as_chunks::<2>().0.iter().rev() {
                self.undo_two_qubit(gate, pair[0], pair[1]);
            }
        } else {
            // Cliffords proxy to themselves; the T-family tracks through
            // [`GateType::clifford_proxy`] (`T`->`S`, `T_YZ`->`SQRT_X`,
            // `T_XZ`->`SQRT_Y`), matching the workspace's
            // `compile_clifford_proxy` convention. Deliberately diverges from
            // Stim, which cannot track a region through a real `T`.
            for &q in qubits {
                self.undo_single_clifford(gate.clifford_proxy(), q);
            }
        }
    }

    /// `undo_RX`/`undo_RY`/`undo_RZ`: gauge-check the anticommuting basis, then
    /// clear the qubit (the region's birth boundary).
    fn undo_reset(&mut self, gate: GateType, q: IVec2) {
        let (x, z) = self.take_frame(q);
        match gate {
            // RZ prepares Z; surviving X support anticommutes.
            GateType::RZ => self.record_gauge(x.iter(), q, RegionBreakKind::AnticommutesWithReset),
            // RX prepares X; surviving Z support anticommutes.
            GateType::RX => self.record_gauge(z.iter(), q, RegionBreakKind::AnticommutesWithReset),
            // RY prepares Y; support that is X-only or Z-only anticommutes.
            GateType::RY => self.record_gauge(
                symmetric_difference(&x, &z),
                q,
                RegionBreakKind::AnticommutesWithReset,
            ),
            _ => unreachable!("undo_reset only sees reset gates"),
        }
        // Frame cleared: both columns dropped (not put back).
    }

    /// `undo_MX`/`undo_MY`/`undo_MZ`: gauge-check the anticommuting basis, then
    /// drain the measurement's pending detectors into the measured basis.
    /// Measurement does not clear the qubit.
    fn undo_measure(&mut self, basis: crate::PauliBasis, qubits: &[IVec2], measurements: &[u32]) {
        use crate::PauliBasis::{X, Y, Z};
        for (&q, &m) in qubits.iter().zip(measurements) {
            let (mut x, mut z) = self.take_frame(q);
            match basis {
                Z => self.record_gauge(x.iter(), q, RegionBreakKind::AnticommutesWithMeasurement),
                X => self.record_gauge(z.iter(), q, RegionBreakKind::AnticommutesWithMeasurement),
                Y => self.record_gauge(
                    symmetric_difference(&x, &z),
                    q,
                    RegionBreakKind::AnticommutesWithMeasurement,
                ),
            }
            if let Some(dets) = self.rec_bits.remove(&m) {
                match basis {
                    Z => z.xor_merge(&dets),
                    X => x.xor_merge(&dets),
                    Y => {
                        x.xor_merge(&dets);
                        z.xor_merge(&dets);
                    }
                }
            }
            self.put_frame(q, x, z);
        }
    }

    /// `undo_MPP`: gauge-check the product's anticommutation parity, then drain
    /// the record into every `(qubit, pauli)` of the product.
    fn undo_mpp(&mut self, products: &[crate::PauliMap], measurements: &[u32]) {
        for (product, &m) in products.iter().zip(measurements).rev() {
            self.gauge_check_product(product);
            if let Some(dets) = self.rec_bits.remove(&m) {
                for (&q, &pauli) in product {
                    let (mut x, mut z) = self.take_frame(q);
                    if pauli & Pauli::X {
                        x.xor_merge(&dets);
                    }
                    if pauli & Pauli::Z {
                        z.xor_merge(&dets);
                    }
                    self.put_frame(q, x, z);
                }
            }
        }
    }

    /// A detector anticommutes with a joint product measurement iff it
    /// anticommutes on an odd number of the product's qubits. Collect every
    /// detector touching the product's support and break the odd ones.
    fn gauge_check_product(&mut self, product: &crate::PauliMap) {
        let mut candidates: FxHashSet<u32> = FxHashSet::default();
        for (&q, _) in product {
            if let Some(set) = self.xs.get(&q) {
                candidates.extend(set.iter());
            }
            if let Some(set) = self.zs.get(&q) {
                candidates.extend(set.iter());
            }
        }
        let qubit = product.representative_coord().unwrap_or(IVec2::ZERO);
        let mut breaks = Vec::new();
        for detector in candidates {
            let mut parity = false;
            for (&q, &pauli) in product {
                let has_x = self.xs.get(&q).is_some_and(|s| s.contains(detector));
                let has_z = self.zs.get(&q).is_some_and(|s| s.contains(detector));
                // anticommute((has_x,has_z), pauli) in the XZ symplectic basis.
                parity ^= (has_x && pauli & Pauli::Z) ^ (has_z && pauli & Pauli::X);
            }
            if parity {
                breaks.push(detector);
            }
        }
        self.record_gauge(breaks, qubit, RegionBreakKind::AnticommutesWithMeasurement);
    }

    /// Undo one single-qubit Clifford by its inverse symplectic action. Signs
    /// (`DAG`/`N` variants) do not matter for unsigned frames, so families
    /// collapse: `S`/`S_DAG`/`H_XY`/`H_NXY` all fold `zs ^= xs`, etc. Mirrors
    /// Stim's `undo_H_XZ`/`undo_H_XY`/`undo_H_YZ`/`undo_C_XYZ`/`undo_C_ZYX`.
    fn undo_single_clifford(&mut self, gate: GateType, q: IVec2) {
        use GateType::*;
        let (mut x, mut z) = self.take_frame(q);
        match gate {
            // H_XZ: swap X and Z.
            H | H_NXZ | SQRT_Y | SQRT_Y_DAG => std::mem::swap(&mut x, &mut z),
            // H_YZ: xs ^= zs.
            H_YZ | H_NYZ | SQRT_X | SQRT_X_DAG => x.xor_merge(&z),
            // H_XY: zs ^= xs.
            H_XY | H_NXY | S | S_DAG => z.xor_merge(&x),
            // C_XYZ: zs ^= xs; xs ^= zs.
            C_XYZ | C_NXYZ | C_XNYZ | C_XYNZ => {
                z.xor_merge(&x);
                x.xor_merge(&z);
            }
            // C_ZYX: xs ^= zs; zs ^= xs.
            C_ZYX | C_NZYX | C_ZNYX | C_ZYNX => {
                x.xor_merge(&z);
                z.xor_merge(&x);
            }
            // Pauli gates commute with everything unsigned; I is a no-op.
            I | X | Y | Z => {}
            _ => unreachable!("undo_single_clifford only sees single-qubit Cliffords"),
        }
        self.put_frame(q, x, z);
    }

    /// Undo one two-qubit controlled Clifford on `(qa, qb)` (control basis then
    /// target basis in the `ACB` name). Each rule is transcribed statement for
    /// statement from Stim's matching `undo_*` / `undo_*_single`, applied to the
    /// two qubits' four sparse columns.
    fn undo_two_qubit(&mut self, gate: GateType, qa: IVec2, qb: IVec2) {
        use GateType::*;
        let (mut xa, mut za) = self.take_frame(qa);
        let (mut xb, mut zb) = self.take_frame(qb);
        match gate {
            XCX => {
                xa.xor_merge(&zb);
                xb.xor_merge(&za);
            }
            XCY => {
                xa.xor_merge(&xb);
                xa.xor_merge(&zb);
                xb.xor_merge(&za);
                zb.xor_merge(&za);
            }
            XCZ => {
                zb.xor_merge(&za);
                xa.xor_merge(&xb);
            }
            YCX => {
                xb.xor_merge(&xa);
                xb.xor_merge(&za);
                xa.xor_merge(&zb);
                za.xor_merge(&zb);
            }
            YCY => {
                za.xor_merge(&xb);
                za.xor_merge(&zb);
                xa.xor_merge(&xb);
                xa.xor_merge(&zb);
                zb.xor_merge(&xa);
                zb.xor_merge(&za);
                xb.xor_merge(&xa);
                xb.xor_merge(&za);
            }
            YCZ => {
                zb.xor_merge(&za);
                zb.xor_merge(&xa);
                xa.xor_merge(&xb);
                za.xor_merge(&xb);
            }
            CX => {
                za.xor_merge(&zb);
                xb.xor_merge(&xa);
            }
            CY => {
                za.xor_merge(&zb);
                za.xor_merge(&xb);
                xb.xor_merge(&xa);
                zb.xor_merge(&xa);
            }
            CZ => {
                za.xor_merge(&xb);
                zb.xor_merge(&xa);
            }
            _ => unreachable!("undo_two_qubit only sees controlled gates"),
        }
        self.put_frame(qa, xa, za);
        self.put_frame(qb, xb, zb);
    }

    /// `undo_implicit_RZs_at_start_of_circuit`: any surviving `X`/`Y` support
    /// anticommutes with the implicit `|0>` preparation. Attributed to segment 0.
    fn undo_start_of_circuit(&mut self) {
        let breaks: Vec<(u32, IVec2)> = self
            .xs
            .iter()
            .flat_map(|(&q, set)| set.iter().map(move |d| (d, q)))
            .collect();
        for (detector, qubit) in breaks {
            self.pending.push(Pending {
                detector,
                qubit,
                kind: RegionBreakKind::AnticommutesWithStart,
            });
        }
        self.process_anticommutations(0);
    }

    /// Record the segment's pending anticommutations and strip each broken
    /// detector from every column so it stops propagating backward. Mirrors
    /// Stim's `process_anticommutations`; `rec_bits` is intentionally left
    /// alone, matching Stim (a detector referencing an earlier measurement can
    /// still discharge there — malformed detectors only).
    fn process_anticommutations(&mut self, segment: usize) {
        if self.pending.is_empty() {
            return;
        }
        let mut pending = std::mem::take(&mut self.pending);
        pending.sort_by_key(|p| (p.detector, p.qubit.x, p.qubit.y));

        let mut stripped: FxHashSet<u32> = FxHashSet::default();
        let mut last: Option<(u32, IVec2)> = None;
        for p in pending {
            // De-duplicate repeated (detector, qubit) events (one op can gauge
            // the same detector on the same qubit more than once).
            if last == Some((p.detector, p.qubit)) {
                continue;
            }
            last = Some((p.detector, p.qubit));
            self.broken.push(RegionBreak {
                detector: p.detector,
                segment,
                qubit: p.qubit,
                kind: p.kind,
            });
            if stripped.insert(p.detector) {
                self.strip_detector(p.detector);
            }
        }
    }

    fn strip_detector(&mut self, detector: u32) {
        self.xs.retain(|_, set| {
            set.remove(detector);
            !set.is_empty()
        });
        self.zs.retain(|_, set| {
            set.remove(detector);
            !set.is_empty()
        });
    }

    /// The current frame as a sorted slice table entry: per detector, its region
    /// terms. `Y` = present in both `X` and `Z` columns.
    fn snapshot(&self) -> Vec<DetectorRegion> {
        let mut per_detector: FxHashMap<u32, Vec<RegionTerm>> = FxHashMap::default();
        for (&qubit, xset) in &self.xs {
            let zset = self.zs.get(&qubit);
            for detector in xset.iter() {
                per_detector.entry(detector).or_default().push(RegionTerm {
                    qubit,
                    pauli: if zset.is_some_and(|set| set.contains(detector)) {
                        PauliBasis::Y
                    } else {
                        PauliBasis::X
                    },
                });
            }
        }
        for (&qubit, zset) in &self.zs {
            let xset = self.xs.get(&qubit);
            for detector in zset
                .iter()
                .filter(|&detector| xset.is_none_or(|set| !set.contains(detector)))
            {
                per_detector.entry(detector).or_default().push(RegionTerm {
                    qubit,
                    pauli: PauliBasis::Z,
                });
            }
        }

        let mut out: Vec<DetectorRegion> = per_detector.into_iter().collect();
        for (_, terms) in &mut out {
            terms.sort_by_key(|term| (term.qubit.x, term.qubit.y));
        }
        out.sort_by_key(|(detector, _)| *detector);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::ivec2;

    fn gate(gate: GateType, qubits: &[IVec2]) -> Op {
        Op::Gate {
            gate,
            qubits: qubits.to_vec(),
        }
    }

    fn measure(basis: PauliBasis, qubit: IVec2, id: u32) -> Op {
        Op::Measure {
            basis,
            qubits: vec![qubit],
            measurements: vec![id],
            flip_probability: 0.0,
        }
    }

    fn det(id: u32, measurements: &[u32]) -> SliceDetector {
        SliceDetector {
            id,
            measurements: measurements.to_vec(),
        }
    }

    /// Render one segment's regions compactly: `d{id}:P(x,y)...`.
    fn render(slice: &[DetectorRegion]) -> String {
        slice
            .iter()
            .map(|(id, terms)| {
                let body: String = terms
                    .iter()
                    .map(|t| format!("{:?}({},{})", t.pauli, t.qubit.x, t.qubit.y))
                    .collect();
                format!("d{id}:{body}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn run(segments: &[Vec<Op>], detectors: &[SliceDetector]) -> DetectorSlices {
        let refs: Vec<&[Op]> = segments.iter().map(Vec::as_slice).collect();
        detector_slices_with_seeds(&refs, detectors, &[]).expect("straight-line input tracks")
    }

    #[test]
    fn reversing_batched_operations_matches_separate_instructions() {
        let [a, b, c] = [ivec2(0, 0), ivec2(1, 0), ivec2(2, 0)];
        let batched = [gate(GateType::CX, &[a, b, b, c])];
        let split = [gate(GateType::CX, &[a, b]), gate(GateType::CX, &[b, c])];
        let seeds = [SliceRegionSeed {
            id: 0,
            boundary: 2,
            terms: vec![RegionTerm {
                qubit: c,
                pauli: PauliBasis::Z,
            }],
        }];
        let batched = detector_slices_with_seeds(&[&[], &batched], &[], &seeds).unwrap();
        let split = detector_slices_with_seeds(&[&[], &split], &[], &seeds).unwrap();
        assert_eq!(batched, split);
        assert_eq!(render(&batched.slices[0]), "d0:Z(0,0)Z(1,0)Z(2,0)");

        let products = [
            crate::PauliMap::from_unique_entries([(a, Pauli::X)]),
            crate::PauliMap::from_unique_entries([(a, Pauli::Z)]),
        ];
        let batched = vec![Op::MPP {
            products: products.to_vec(),
            measurements: vec![0, 1],
        }];
        let split = products
            .into_iter()
            .enumerate()
            .map(|(m, product)| Op::MPP {
                products: vec![product],
                measurements: vec![m as u32],
            })
            .collect();
        let batched = run(&[batched], &[det(0, &[1])]);
        let split = run(&[split], &[det(0, &[1])]);
        assert_eq!(batched, split);
        assert!(
            !batched.breaks.is_empty(),
            "MPP X0 Z0 leaves the final Z outcome random"
        );
    }

    #[test]
    fn single_round_z_memory_holds_region_until_measurement() {
        let q = ivec2(0, 0);
        let segments = vec![
            vec![gate(GateType::RZ, &[q])],
            vec![Op::Tick],
            vec![measure(PauliBasis::Z, q, 0)],
        ];
        let slices = run(&segments, &[det(9, &[0])]);
        assert_eq!(render(&slices.slices[0]), "d9:Z(0,0)");
        assert_eq!(render(&slices.slices[1]), "d9:Z(0,0)");
        assert_eq!(render(&slices.slices[2]), "");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn two_round_memory_region_lives_only_between_measurements() {
        let q = ivec2(0, 0);
        let segments = vec![
            vec![gate(GateType::RZ, &[q])],
            vec![measure(PauliBasis::Z, q, 0)],
            vec![measure(PauliBasis::Z, q, 1)],
        ];
        // Parity of the two rounds: region cancels outside the interval.
        let slices = run(&segments, &[det(1, &[0, 1])]);
        assert_eq!(render(&slices.slices[0]), "");
        assert_eq!(render(&slices.slices[1]), "d1:Z(0,0)");
        assert_eq!(render(&slices.slices[2]), "");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn hadamard_swaps_region_basis_across_the_gate() {
        let q = ivec2(0, 0);
        let segments = vec![
            vec![gate(GateType::RZ, &[q])],
            vec![gate(GateType::H, &[q])],
            vec![measure(PauliBasis::X, q, 0)],
        ];
        let slices = run(&segments, &[det(0, &[0])]);
        // Z before the H (propagated back), X after it (toward the MX).
        assert_eq!(render(&slices.slices[0]), "d0:Z(0,0)");
        assert_eq!(render(&slices.slices[1]), "d0:X(0,0)");
        assert_eq!(render(&slices.slices[2]), "");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn positioned_pauli_seed_enters_at_its_exact_boundary() {
        let q = ivec2(0, 0);
        let segments = [vec![gate(GateType::H, &[q])], vec![gate(GateType::I, &[q])]];
        let refs: Vec<&[Op]> = segments.iter().map(Vec::as_slice).collect();
        let seed = SliceRegionSeed {
            id: 4,
            boundary: 1,
            terms: vec![RegionTerm {
                qubit: q,
                pauli: PauliBasis::Z,
            }],
        };

        let slices = detector_slices_with_seeds(&refs, &[], &[seed]).unwrap();

        assert_eq!(render(&slices.slices[0]), "d4:Z(0,0)");
        assert_eq!(render(&slices.slices[1]), "");
        assert_eq!(
            slices.breaks,
            vec![RegionBreak {
                detector: 4,
                segment: 0,
                qubit: q,
                kind: RegionBreakKind::AnticommutesWithStart,
            }]
        );
    }

    #[test]
    fn cx_plaquette_region_grows_across_coupled_data_qubits() {
        let a = ivec2(1, 0);
        let d0 = ivec2(0, 0);
        let d1 = ivec2(2, 0);
        let segments = vec![
            vec![gate(GateType::RZ, &[a])],
            vec![gate(GateType::CX, &[d0, a])],
            vec![gate(GateType::CX, &[d1, a])],
            vec![measure(PauliBasis::Z, a, 0)],
        ];
        let slices = run(&segments, &[det(0, &[0])]);
        assert_eq!(render(&slices.slices[3]), "");
        assert_eq!(render(&slices.slices[2]), "d0:Z(1,0)");
        assert_eq!(render(&slices.slices[1]), "d0:Z(1,0)Z(2,0)");
        assert_eq!(render(&slices.slices[0]), "d0:Z(0,0)Z(1,0)Z(2,0)");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn phase_gate_produces_a_y_region() {
        let q = ivec2(0, 0);
        // RY commutes with the Y the S gate builds, so no break: a clean Y.
        let segments = vec![
            vec![gate(GateType::RY, &[q])],
            vec![gate(GateType::S, &[q])],
            vec![measure(PauliBasis::X, q, 0)],
        ];
        let slices = run(&segments, &[det(0, &[0])]);
        assert_eq!(render(&slices.slices[0]), "d0:Y(0,0)");
        assert_eq!(render(&slices.slices[1]), "d0:X(0,0)");
        assert_eq!(render(&slices.slices[2]), "");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn segment_ops_are_undone_in_reverse_order() {
        let q = ivec2(0, 0);
        let segments = vec![
            vec![gate(GateType::H, &[q]), gate(GateType::S, &[q])],
            vec![measure(PauliBasis::X, q, 0)],
        ];
        let slices = run(&segments, &[det(0, &[0])]);

        assert_eq!(
            slices.breaks,
            vec![RegionBreak {
                detector: 0,
                segment: 0,
                qubit: q,
                kind: RegionBreakKind::AnticommutesWithStart,
            }]
        );
    }

    #[test]
    fn mpp_product_region_is_the_product_support() {
        let q0 = ivec2(0, 0);
        let q1 = ivec2(1, 0);
        let product = crate::PauliMap::from_unique_entries([(q0, Pauli::X), (q1, Pauli::X)]);
        let segments = vec![
            vec![gate(GateType::RX, &[q0]), gate(GateType::RX, &[q1])],
            vec![Op::MPP {
                products: vec![product],
                measurements: vec![0],
            }],
        ];
        let slices = run(&segments, &[det(0, &[0])]);
        assert_eq!(render(&slices.slices[0]), "d0:X(0,0)X(1,0)");
        assert_eq!(render(&slices.slices[1]), "");
        assert!(slices.breaks.is_empty());
    }

    #[test]
    fn x_region_anticommuting_with_z_reset_breaks() {
        let q = ivec2(0, 0);
        let segments = vec![
            vec![gate(GateType::I, &[q])],
            vec![gate(GateType::RZ, &[q])],
            vec![measure(PauliBasis::X, q, 0)],
        ];
        let slices = run(&segments, &[det(7, &[0])]);
        // Present at end of the reset segment, stripped before the earlier one.
        assert_eq!(render(&slices.slices[0]), "");
        assert_eq!(render(&slices.slices[1]), "d7:X(0,0)");
        assert_eq!(render(&slices.slices[2]), "");
        assert_eq!(
            slices.breaks,
            vec![RegionBreak {
                detector: 7,
                segment: 1,
                qubit: q,
                kind: RegionBreakKind::AnticommutesWithReset,
            }]
        );
    }

    #[test]
    fn t_gate_tracks_through_its_clifford_proxy_like_s() {
        let q = ivec2(0, 0);
        // T shares S's Z axis, so it must fold an X region into a Y exactly as S
        // does and keep the region alive — unlike Stim, which breaks on a real
        // T. Run the same circuit with T and with its S proxy; they must agree.
        let build = |g: GateType| {
            vec![
                vec![gate(GateType::RY, &[q])],
                vec![gate(g, &[q])],
                vec![measure(PauliBasis::X, q, 0)],
            ]
        };
        let via_t = run(&build(GateType::T), &[det(0, &[0])]);
        let via_s = run(&build(GateType::S), &[det(0, &[0])]);
        assert_eq!(via_t.slices, via_s.slices);
        assert!(via_t.breaks.is_empty(), "T keeps the region alive");
        // X toward the MX, folded to Y on the far side of the T.
        assert_eq!(render(&via_t.slices[1]), "d0:X(0,0)");
        assert_eq!(render(&via_t.slices[0]), "d0:Y(0,0)");
    }

    #[test]
    fn surviving_x_support_breaks_against_implicit_start() {
        let q = ivec2(0, 0);
        // No reset in X: the region reaches the implicit |0> and breaks.
        let segments = vec![
            vec![gate(GateType::I, &[q])],
            vec![measure(PauliBasis::X, q, 0)],
        ];
        let slices = run(&segments, &[det(5, &[0])]);
        assert_eq!(render(&slices.slices[0]), "d5:X(0,0)");
        assert_eq!(render(&slices.slices[1]), "");
        assert_eq!(
            slices.breaks,
            vec![RegionBreak {
                detector: 5,
                segment: 0,
                qubit: q,
                kind: RegionBreakKind::AnticommutesWithStart,
            }]
        );
    }

    #[test]
    fn repeat_op_is_rejected() {
        let body = crate::BodyId(0);
        let segments = [vec![Op::Repeat {
            body,
            repetitions: 2,
        }]];
        let refs: Vec<&[Op]> = segments.iter().map(Vec::as_slice).collect();
        assert_eq!(
            detector_slices_with_seeds(&refs, &[], &[]),
            Err(DetsliceError::RepeatNotFlattened)
        );
    }
}
