use std::cmp::Ordering;

use glam::IVec2;
use smallvec::SmallVec;

use crate::{CoordinateOverflowError, GateType, PauliBasis, PauliMap};

/// Translate one coordinate without wrapping outside the signed 32-bit
/// lattice.
///
/// # Errors
///
/// Returns [`CoordinateOverflowError`] if either translated component overflows.
pub fn checked_translate_coordinate(
    coordinate: IVec2,
    offset: IVec2,
) -> Result<IVec2, CoordinateOverflowError> {
    coordinate
        .x
        .checked_add(offset.x)
        .zip(coordinate.y.checked_add(offset.y))
        .map(|(x, y)| IVec2::new(x, y))
        .ok_or(CoordinateOverflowError { coordinate, offset })
}

/// Index of a [`CircuitBody`] within a [`CoordCircuit`](crate::CoordCircuit).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct BodyId(pub u32);

/// One feedforward correction inside [`Op::ConditionalPauli`].
///
/// Apply `pauli` to `target` iff measurement record `control` is 1. Bundling the fields
/// makes the per-correction invariant true by construction (no parallel-array
/// desync). `control` is a template-local measurement id, remapped through
/// instantiation like any other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConditionalCorrection {
    /// Pauli applied to the target.
    pub pauli: PauliBasis,
    /// Measurement id controlling the correction.
    pub control: u32,
    /// Corrected qubit coordinate.
    pub target: IVec2,
}

/// A measurement id paired with the qubit coordinate it measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MeasRecord {
    /// The measurement's stable id.
    pub id: u32,
    /// The qubit that was measured.
    pub qubit: IVec2,
}

/// Allocator and lookup table for a circuit's measurement ids.
///
/// Records are kept strictly sorted by id. Dense ids use direct indexing;
/// sparse lookups fall back to binary search. On the wire only the records are
/// stored; the next-id counter is recomputed on decode (see the serde impls)
/// rather than trusted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeasRegistry {
    records: Vec<MeasRecord>,
    next_id: u32,
}

// Manual impls: the wire carries only the records. `next_id` is derived
// state — every mutation path leaves it at `last record id + 1` (or 0 when
// empty) — so it is recomputed on decode rather than trusted from untrusted
// bytes, where a low value would make `allocate` hand out duplicate ids.
// Sortedness is likewise an invariant of the constructors (binary searches
// rely on it), so unsorted wire data is rejected.
impl serde::Serialize for MeasRegistry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.records.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for MeasRegistry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let records = Vec::<MeasRecord>::deserialize(deserializer)?;
        if !records.windows(2).all(|pair| pair[0].id < pair[1].id) {
            return Err(serde::de::Error::custom(
                "measurement records must be strictly sorted by id",
            ));
        }
        let next_id = match records.last() {
            None => 0,
            Some(record) => record
                .id
                .checked_add(1)
                .ok_or_else(|| serde::de::Error::custom("measurement id space exhausted"))?,
        };
        Ok(Self { records, next_id })
    }
}

impl MeasRegistry {
    /// Creates an empty measurement registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocates one sequential id per qubit, appending a record for each, and
    /// returns the ids in input order.
    pub fn allocate(&mut self, qubits: impl IntoIterator<Item = IVec2>) -> Vec<u32> {
        qubits
            .into_iter()
            .map(|qubit| self.allocate_one(qubit))
            .collect()
    }

    /// Allocates a single sequential id for `qubit`, appending its record —
    /// the [`Self::allocate`] core without the return `Vec`, for hot callers
    /// that allocate one id at a time.
    ///
    /// # Panics
    ///
    /// Panics if the measurement-id space is exhausted.
    pub fn allocate_one(&mut self, qubit: IVec2) -> u32 {
        let id = self.next_id;
        self.next_id = id_after(id);
        self.records.push(MeasRecord { id, qubit });
        id
    }

    /// Reserve `id` for `qubit`, keeping the record list sorted. Reserving an
    /// already-reserved id for the same qubit is a no-op.
    ///
    /// # Panics
    ///
    /// Panics if `id` is already reserved for a different qubit or is `u32::MAX`.
    /// The registry is unchanged if this check fails.
    pub fn reserve(&mut self, id: u32, qubit: IVec2) {
        // Callers reserve in ascending order, so check the tail before paying
        // for a search.
        let slot = match self.records.last() {
            Some(record) if record.id < id => Err(self.records.len()),
            _ => self.records.binary_search_by_key(&id, |record| record.id),
        };
        match slot {
            Ok(index) => assert_eq!(
                self.records[index].qubit, qubit,
                "measurement id reserved for a different qubit"
            ),
            Err(index) => {
                let next_id = id_after(id);
                self.records.insert(index, MeasRecord { id, qubit });
                self.next_id = self.next_id.max(next_id);
            }
        }
    }

    /// Reserve every record, sorting them by id first (see [`Self::reserve`]).
    ///
    /// # Panics
    ///
    /// Panics if any record's id is already reserved for a different qubit.
    pub(crate) fn reserve_many(&mut self, records: &mut [MeasRecord]) {
        records.sort_unstable_by_key(|record| record.id);
        for record in records {
            self.reserve(record.id, record.qubit);
        }
    }

    /// Rewrites record ids through `remap`, then re-sorts and deduplicates so
    /// the sorted-by-id invariant holds. Ids absent from `remap` are unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a remapped record uses `u32::MAX`, leaving the registry unchanged.
    pub(crate) fn remap_ids(&mut self, remap: &rustc_hash::FxHashMap<u32, u32>) {
        if remap.is_empty() {
            return;
        }

        let next_id = self
            .records
            .iter()
            .map(|record| remap.get(&record.id).copied().unwrap_or(record.id))
            .max()
            .map_or(0, id_after);

        for record in &mut self.records {
            let id = remap.get(&record.id).copied().unwrap_or(record.id);
            record.id = id;
        }
        self.records.sort_by_key(|record| record.id);
        self.records.dedup_by_key(|record| record.id);
        self.next_id = next_id;
    }

    /// Borrows the record for `id`, or `None` if it is not allocated.
    pub fn record(&self, id: u32) -> Option<&MeasRecord> {
        if let Some(record) = self.records.get(id as usize)
            && record.id == id
        {
            return Some(record);
        }
        self.records
            .binary_search_by_key(&id, |record| record.id)
            .ok()
            .map(|index| &self.records[index])
    }

    /// All records, sorted by id.
    pub fn records(&self) -> &[MeasRecord] {
        &self.records
    }

    /// The id the next [`Self::allocate`] call will hand out.
    pub(crate) fn next_id(&self) -> u32 {
        self.next_id
    }
}

/// The successor of an allocated measurement id.
fn id_after(id: u32) -> u32 {
    id.checked_add(1)
        .expect("id space is u32; programs stay far below 2^32 measurements")
}

/// Identifies a loop-carried detector state threaded across iterations of a
/// repeated body.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct LoopStateId(pub u32);

/// One term of a [`DetectorParity`]: either a measurement outcome or a
/// loop-carried state.
///
/// Generic over the measurement type `M` so parities can be built over raw
/// record ids or over already-resolved terms.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum DetectorTerm<M = u32> {
    /// A measurement outcome.
    Measurement(M),
    /// A loop-carried detector state.
    LoopState(LoopStateId),
}

type DetectorTerms<M> = SmallVec<[DetectorTerm<M>; 2]>;

/// A mod-2 sum of [`DetectorTerm`]s and a constant sign bit — the parity a
/// detector or observable checks.
///
/// Terms are held in a canonical form: sorted, with pairs that cancel removed,
/// so equal parities compare and hash equally regardless of construction order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct DetectorParity<M = u32> {
    terms: DetectorTerms<M>,
    sign: bool,
}

impl<M> Default for DetectorParity<M> {
    fn default() -> Self {
        Self {
            terms: SmallVec::new(),
            sign: false,
        }
    }
}

impl<M: Copy + Ord> DetectorParity<M> {
    /// Builds a parity from measurement ids, canonicalizing the result.
    pub fn from_measurements(measurements: impl IntoIterator<Item = M>) -> Self {
        Self::from_terms(measurements.into_iter().map(DetectorTerm::Measurement))
    }

    /// Builds a parity from arbitrary terms, canonicalizing the result.
    pub fn from_terms(terms: impl IntoIterator<Item = DetectorTerm<M>>) -> Self {
        Self::from_term_buf(terms.into_iter().collect())
    }

    /// Like [`Self::from_terms`], but takes the term buffer by value so
    /// callers that already hold one (e.g. a completed flow's measurements)
    /// avoid rebuilding it element by element.
    pub fn from_term_buf(terms: SmallVec<[DetectorTerm<M>; 2]>) -> Self {
        let mut parity = Self { terms, sign: false };
        parity.canonicalize();
        parity
    }

    /// Sets the constant XOR term. `true` represents a negative stabilizer
    /// sign, so a noiseless detector with odd raw measurement parity evaluates
    /// to zero.
    #[must_use]
    pub fn with_sign(mut self, sign: bool) -> Self {
        self.sign = sign;
        self
    }

    /// The constant XOR term (`true` for a negative stabilizer sign).
    #[must_use]
    pub fn sign(&self) -> bool {
        self.sign
    }

    /// The canonicalized terms.
    pub fn terms(&self) -> &[DetectorTerm<M>] {
        &self.terms
    }

    /// Measurement terms only, dropping loop-carried detector state.
    pub fn measurements(&self) -> impl Iterator<Item = M> + '_ {
        self.terms.iter().filter_map(|term| match term {
            DetectorTerm::Measurement(measurement) => Some(*measurement),
            DetectorTerm::LoopState(_) => None,
        })
    }

    /// Returns `true` for the zero parity: no terms and no constant sign.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty() && !self.sign
    }

    /// XORs `other`'s terms into `self` (mod-2 addition) and recanonicalizes.
    pub fn xor_assign(&mut self, other: &Self) {
        self.sign ^= other.sign;
        self.terms.extend_from_slice(&other.terms);
        self.canonicalize();
    }

    /// Maps measurement terms in the existing buffer and recanonicalizes it.
    #[must_use]
    pub fn map_measurements(mut self, mut f: impl FnMut(M) -> M) -> Self {
        for term in &mut self.terms {
            if let DetectorTerm::Measurement(measurement) = term {
                *measurement = f(*measurement);
            }
        }
        self.canonicalize();
        self
    }

    /// Maps each measurement term through `f`, leaving loop-state terms intact,
    /// and canonicalizes the result. Loop-state terms pass through unchanged.
    ///
    /// # Errors
    ///
    /// Returns the first error `f` produces.
    pub fn try_map_measurements<N: Copy + Ord, E>(
        &self,
        mut f: impl FnMut(M) -> Result<N, E>,
    ) -> Result<DetectorParity<N>, E> {
        let mut terms = DetectorTerms::with_capacity(self.terms.len());
        for &term in &self.terms {
            terms.push(match term {
                DetectorTerm::Measurement(measurement) => {
                    DetectorTerm::Measurement(f(measurement)?)
                }
                DetectorTerm::LoopState(state) => DetectorTerm::LoopState(state),
            });
        }
        Ok(DetectorParity::from_term_buf(terms).with_sign(self.sign))
    }

    fn canonicalize(&mut self) {
        // The one- and two-term cases dominate (a detector compares two
        // consecutive stabilizer rounds); resolve them without invoking the
        // sort machinery.
        match self.terms.as_mut_slice() {
            [] | [_] => return,
            [a, b] => {
                match (*a).cmp(&*b) {
                    Ordering::Equal => self.terms.clear(),
                    Ordering::Greater => self.terms.swap(0, 1),
                    Ordering::Less => {}
                }
                return;
            }
            _ => {}
        }

        self.terms.sort_unstable();
        let mut write = 0;
        let mut read = 0;
        while read < self.terms.len() {
            let term = self.terms[read];
            let mut count = 1;
            read += 1;
            while read < self.terms.len() && self.terms[read] == term {
                count += 1;
                read += 1;
            }
            if count % 2 == 1 {
                self.terms[write] = term;
                write += 1;
            }
        }
        self.terms.truncate(write);
    }
}

impl<M: std::fmt::Display> std::fmt::Display for DetectorParity<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.sign {
            write!(f, "-")?;
        }
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (index, term) in self.terms.iter().enumerate() {
            if index > 0 {
                write!(f, "*")?;
            }
            match term {
                DetectorTerm::Measurement(measurement) => write!(f, "{measurement}")?,
                DetectorTerm::LoopState(state) => write!(f, "loop_state({})", state.0)?,
            }
        }
        Ok(())
    }
}

/// A detector state carried between iterations of a repeated body.
///
/// `initial` is the parity substituted for the state on the first iteration;
/// `next` is the parity the state takes for the following iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopCarriedDetectorState<M = u32> {
    /// The state this record defines.
    pub state: LoopStateId,
    /// Parity used for the state on the first iteration.
    pub initial: DetectorParity<M>,
    /// Parity the state carries into the next iteration.
    pub next: DetectorParity<M>,
}

/// An ordered list of operations — the entry block of a circuit or the body of
/// a [`Op::Repeat`].
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CircuitBody {
    ops: Vec<Op>,
}

impl CircuitBody {
    /// Creates an empty circuit body.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a circuit body from an operation sequence.
    pub fn from_ops(ops: Vec<Op>) -> Self {
        Self { ops }
    }

    /// Borrows this body's operations.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// Mutably borrows this body's operations.
    pub fn ops_mut(&mut self) -> &mut Vec<Op> {
        &mut self.ops
    }
}

/// Coordinates attached to a detector annotation, almost always an `(x, y)`
/// pair; kept inline so the thousands of detector ops a compile produces do
/// not heap-allocate their coordinates.
pub type DetectorCoords = SmallVec<[f64; 2]>;

/// Offsets a detector's first two coordinates by a layout offset, leaving any
/// further coordinates untouched.
#[must_use]
pub fn translate_detector_coords(coords: &DetectorCoords, offset: IVec2) -> DetectorCoords {
    let mut coords = coords.clone();
    if let Some(x) = coords.get_mut(0) {
        *x += offset.x as f64;
    }
    if let Some(y) = coords.get_mut(1) {
        *y += offset.y as f64;
    }
    coords
}

/// A single coordinate-addressed circuit operation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Op {
    /// A gate applied to its target qubits (pairwise for two-qubit gates).
    Gate {
        /// Gate kind.
        gate: GateType,
        /// Target qubits.
        qubits: Vec<IVec2>,
    },
    /// A single-qubit measurement of each qubit in `basis`; `measurements[i]`
    /// is the record id produced for `qubits[i]`.
    Measure {
        /// Measured basis.
        basis: PauliBasis,
        /// Measured qubits.
        qubits: Vec<IVec2>,
        /// Measurement ids produced in qubit order.
        measurements: Vec<u32>,
        /// Probability of flipping each reported measurement result.
        flip_probability: f64,
    },
    /// Multi-Pauli-product measurement (Stim `MPP`). Each entry of `products` is
    /// one Pauli-product group (e.g. `X0*X1`) measured jointly; `products[i]`
    /// produces record `measurements[i]`, and the data-qubit support is carried
    /// here for emission. Temporal `Port` blocks use this to measure patch
    /// stabilizers as an ideal noiseless open boundary.
    MPP {
        /// Joint Pauli products.
        products: Vec<PauliMap>,
        /// Measurement ids produced in product order.
        measurements: Vec<u32>,
    },
    /// Independent single-qubit depolarizing channels on `qubits`.
    Depolarize1 {
        /// Error probability.
        probability: f64,
        /// Affected qubits.
        qubits: Vec<IVec2>,
    },
    /// Independent two-qubit depolarizing channels on consecutive qubit pairs.
    Depolarize2 {
        /// Error probability.
        probability: f64,
        /// Affected consecutive qubit pairs.
        qubits: Vec<IVec2>,
    },
    /// Independent Pauli errors on `qubits` (Stim `X_ERROR`/`Y_ERROR`/`Z_ERROR`).
    PauliError {
        /// Error probability.
        probability: f64,
        /// Applied Pauli.
        pauli: PauliBasis,
        /// Affected qubits.
        qubits: Vec<IVec2>,
    },
    /// A moment barrier separating parallel operations.
    Tick,
    /// Executes `body` `repetitions` times (Stim `REPEAT`).
    Repeat {
        /// Repeated body.
        body: BodyId,
        /// Iteration count.
        repetitions: u32,
    },
    /// Physically-applied Pauli feedforward: each correction applies its Pauli
    /// to its target iff its control measurement is 1 (Stim `CZ rec[-k] q`).
    ///
    /// Support-transparent: the [`FlowEngine`](crate::FlowEngine) ignores it, so
    /// flows/detectors derive from the Clifford+[`Measure`](Op::Measure)
    /// structure alone. The *parity* effect of a
    /// measurement-conditioned sign flip (a control entering a downstream
    /// detector/observable parity) is the **backend/DEM generator's**
    /// responsibility, not the IR's — the IR carries the op, not the analysis.
    ConditionalPauli(Vec<ConditionalCorrection>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::ivec2;

    #[test]
    fn registry_overflow_panics_leave_records_unchanged() {
        for remap in [false, true] {
            let mut registry = MeasRegistry::new();
            registry.reserve(3, ivec2(0, 0));
            let before = registry.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if remap {
                    registry.remap_ids(&[(3, u32::MAX)].into_iter().collect());
                } else {
                    registry.reserve(u32::MAX, ivec2(1, 0));
                }
            }));
            assert!(result.is_err(), "exhausted measurement ids must panic");
            assert_eq!(
                registry,
                before,
                "{} mutated the registry after overflow",
                if remap { "remap_ids" } else { "reserve" },
            );
        }
    }

    #[test]
    fn detector_parity_xors_signs_and_terms() {
        let mut parity = DetectorParity::from_measurements([7]).with_sign(true);
        parity.xor_assign(&DetectorParity::from_measurements([7]));

        assert!(parity.terms().is_empty());
        assert!(parity.sign());
        assert!(
            !parity.is_empty(),
            "negative identity is not the zero parity"
        );

        parity.xor_assign(&DetectorParity::default().with_sign(true));
        assert!(parity.is_empty());
    }

    #[test]
    fn detector_parity_map_recanonicalizes_the_reused_buffer() {
        let parity = DetectorParity::from_measurements([1, 2, 3])
            .with_sign(true)
            .map_measurements(|measurement| if measurement == 1 { 3 } else { measurement });

        assert_eq!(parity.measurements().collect::<Vec<_>>(), vec![2]);
        assert!(parity.sign());
    }

    /// The wire carries only records: `next_id` is recomputed on decode (a
    /// trusted-from-the-wire value could make `allocate` reuse an id), and
    /// unsorted records — which would break the binary searches — are
    /// rejected.
    #[test]
    fn deserialize_recomputes_next_id_and_rejects_unsorted_records() {
        let mut registry = MeasRegistry::new();
        registry.reserve(3, ivec2(0, 0));
        registry.reserve(7, ivec2(1, 0));
        let bytes = postcard::to_allocvec(&registry).expect("registry encodes into a Vec");
        let mut restored: MeasRegistry = postcard::from_bytes(&bytes).expect("registry decodes");
        assert_eq!(restored, registry);
        assert_eq!(restored.record(7).unwrap().qubit, ivec2(1, 0));
        assert!(restored.record(1).is_none());
        assert_eq!(restored.allocate([ivec2(2, 0)]), vec![8]);

        let unsorted = vec![
            MeasRecord {
                id: 7,
                qubit: ivec2(1, 0),
            },
            MeasRecord {
                id: 3,
                qubit: ivec2(0, 0),
            },
        ];
        let bytes = postcard::to_allocvec(&unsorted).expect("test wire encodes into a Vec");
        postcard::from_bytes::<MeasRegistry>(&bytes).unwrap_err();
    }

    #[test]
    fn meas_registry_reserves_many_in_sorted_order() {
        let mut registry = MeasRegistry::new();
        let mut records = vec![
            MeasRecord {
                id: 2,
                qubit: ivec2(2, 0),
            },
            MeasRecord {
                id: 0,
                qubit: ivec2(0, 0),
            },
            MeasRecord {
                id: 1,
                qubit: ivec2(1, 0),
            },
        ];

        registry.reserve_many(&mut records);

        assert_eq!(
            registry.records(),
            &[
                MeasRecord {
                    id: 0,
                    qubit: ivec2(0, 0)
                },
                MeasRecord {
                    id: 1,
                    qubit: ivec2(1, 0)
                },
                MeasRecord {
                    id: 2,
                    qubit: ivec2(2, 0)
                },
            ]
        );
    }
}
