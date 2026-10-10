use bloq_utils::PauliBasis;
use glam::IVec2;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    BodyId, CircuitBody, CircuitError, MeasRecord, MeasRegistry, MeasurementFrameError, Op,
    gate::GateType,
};

/// A replay step produced by [`CoordCircuit::flatten`].
///
/// Events appear in emission order. Side-table owners replay them to re-scope
/// loop-relative annotations onto the unrolled circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlattenEvent {
    /// A measurement occurrence was emitted: the pre-flatten id `original`
    /// occurs here as `emitted`. Every occurrence except the last gets a fresh
    /// id; the **final** occurrence keeps `emitted == original`, so a
    /// pre-flatten id read from outside a loop (a top-level detector, a
    /// boundary flow, an instance-space parity) still names the same record
    /// the emission frame's latest-occurrence rule resolved it to.
    Measurement {
        /// Measurement id before flattening.
        original: u32,
        /// Measurement id assigned to this occurrence.
        emitted: u32,
    },
    /// A `REPEAT` over `body` with at least one repetition began unrolling.
    /// Loop-carried states resolve their `initial` parity here, before the
    /// first iteration's measurements shadow anything they reference.
    LoopEnter {
        /// Body whose expansion begins.
        body: BodyId,
    },
    /// One unrolled iteration of `body` finished. Body-scoped detectors are
    /// due here, then loop-carried states advance by their `next` parity.
    IterationEnd {
        /// Body whose iteration ended.
        body: BodyId,
    },
    /// A `REPEAT` over `body` finished. Side-table replay uses this to close
    /// the loop-state frame and expose only this loop's declared final states
    /// to its parent scope.
    LoopExit {
        /// Body whose expansion ended.
        body: BodyId,
    },
}

/// Work allowed when materializing a circuit's repeated operation stream.
///
/// Counts body visits, operations, operation targets, and replay events.
/// The estimate is checked on the shared body DAG before unrolling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlattenLimits {
    /// Maximum charged expansion work.
    pub max_work: usize,
}

impl Default for FlattenLimits {
    fn default() -> Self {
        Self {
            max_work: 16 * 1024 * 1024,
        }
    }
}

/// A quantum circuit whose qubits are addressed by 2-D lattice coordinates
/// rather than dense integer indices.
///
/// Operations live in [`CircuitBody`] blocks (the entry body plus any bodies
/// referenced by [`Op::Repeat`]), stored flat and referenced by [`BodyId`].
/// Measurement records are allocated through an internal [`MeasRegistry`], so
/// every measurement has a stable id independent of qubit index. A
/// coordinate-to-index layout for the Stim FFI boundary is derived on demand
/// via [`Self::build_coord_to_index`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CoordCircuit {
    meas_registry: MeasRegistry,
    #[serde(deserialize_with = "deserialize_bodies")]
    bodies: Vec<CircuitBody>,
    entry: BodyId,
}

impl Default for CoordCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordCircuit {
    /// Creates an empty circuit with one entry body.
    pub fn new() -> Self {
        Self {
            meas_registry: MeasRegistry::new(),
            bodies: vec![CircuitBody::new()],
            entry: BodyId(0),
        }
    }

    /// The set of every qubit coordinate touched by any body or measurement
    /// record.
    pub fn qubits(&self) -> FxHashSet<IVec2> {
        let mut occurrences = 0usize;
        let mut min = IVec2::MAX;
        let mut max = IVec2::MIN;
        self.for_each_qubit(|qubit| {
            occurrences += 1;
            min = min.min(qubit);
            max = max.max(qubit);
        });
        let mut qubits = FxHashSet::with_capacity_and_hasher(
            self.meas_registry.records().len(),
            Default::default(),
        );
        if occurrences == 0 {
            return qubits;
        }
        let width = i64::from(max.x) - i64::from(min.x) + 1;
        let height = i64::from(max.y) - i64::from(min.y) + 1;
        let dense_area = width
            .checked_mul(height)
            .and_then(|area| usize::try_from(area).ok())
            // Bound dense scratch by the number of coordinate visits.
            .filter(|&area| area <= occurrences);

        if let Some(area) = dense_area {
            let mut seen = vec![0u8; area];
            let width = width as usize;
            self.for_each_qubit(|qubit| {
                let x = (i64::from(qubit.x) - i64::from(min.x)) as usize;
                let y = (i64::from(qubit.y) - i64::from(min.y)) as usize;
                let index = y * width + x;
                if seen[index] == 0 {
                    seen[index] = 1;
                    qubits.insert(qubit);
                }
            });
        } else {
            self.for_each_qubit(|qubit| {
                qubits.insert(qubit);
            });
        }
        qubits
    }

    /// The id of the top-level body operations are appended to.
    pub const fn entry_body(&self) -> BodyId {
        self.entry
    }

    /// Select an allocated body as the entry without renumbering body ids.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::InvalidCircuitBody`] if `body` is not allocated.
    pub fn set_entry_body(&mut self, body: BodyId) -> Result<(), CircuitError> {
        self.body(body)
            .ok_or(CircuitError::InvalidCircuitBody(body))?;
        self.entry = body;
        Ok(())
    }

    /// The repetition counts of the top-level [`Op::Repeat`]s in the entry
    /// body, in order. Used to classify memory-padding templates: a one-round
    /// template has none, a looped template has exactly one.
    pub fn entry_top_level_repeats(&self) -> Vec<u32> {
        self.body(self.entry)
            .map(|body| {
                body.ops()
                    .iter()
                    .filter_map(|op| match op {
                        Op::Repeat { repetitions, .. } => Some(*repetitions),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The number of circuit bodies, including repeat bodies.
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Borrows the body with the given id, or `None` if it is not allocated.
    pub fn body(&self, id: BodyId) -> Option<&CircuitBody> {
        self.bodies.get(id.0 as usize)
    }

    /// Mutably borrows the body with the given id, or `None` if it is not
    /// allocated.
    pub fn body_mut(&mut self, id: BodyId) -> Option<&mut CircuitBody> {
        self.bodies.get_mut(id.0 as usize)
    }

    /// Borrows the measurement registry backing this circuit's ids.
    pub fn meas_registry(&self) -> &MeasRegistry {
        &self.meas_registry
    }

    /// Allocates and returns a fresh measurement id for `qubit`.
    pub fn reserve_measurement_id(&mut self, qubit: IVec2) -> u32 {
        self.meas_registry.allocate_one(qubit)
    }

    #[doc(hidden)]
    pub fn register_measurement_id(&mut self, id: u32, qubit: IVec2) {
        self.meas_registry.reserve(id, qubit);
    }

    #[doc(hidden)]
    pub fn register_measurement_records(&mut self, records: &mut [MeasRecord]) {
        self.meas_registry.reserve_many(records);
    }

    /// Rewrites every measurement id through `remap`, in both the registry and
    /// every op that references a measurement. Ids absent from `remap` are left
    /// unchanged.
    pub fn remap_measurement_ids(&mut self, remap: &rustc_hash::FxHashMap<u32, u32>) {
        if remap.is_empty() {
            return;
        }

        self.meas_registry.remap_ids(remap);
        for body in &mut self.bodies {
            for op in body.ops_mut() {
                remap_op_measurement_ids(op, remap);
            }
        }
    }

    /// Appends `body` and returns its newly assigned [`BodyId`].
    ///
    /// # Panics
    ///
    /// Panics if the body-id space is exhausted.
    pub fn add_body(&mut self, body: CircuitBody) -> BodyId {
        let id = checked_body_id(self.bodies.len()).expect("body-id space is u32");
        self.bodies.push(body);
        id
    }

    fn entry_ops_mut(&mut self) -> &mut Vec<Op> {
        self.body_mut(self.entry)
            .expect("entry body is created with the circuit")
            .ops_mut()
    }

    /// Build a deterministic coord-to-index layout for the Stim FFI boundary.
    /// Indices are assigned by sorting coordinates by (x, y).
    ///
    /// # Panics
    ///
    /// Panics if the qubit-index space is exhausted.
    pub fn build_coord_to_index(&self) -> FxHashMap<IVec2, u32> {
        let mut coords = self.qubits().into_iter().collect::<Vec<_>>();
        coords.sort_unstable_by_key(|coord| (coord.x, coord.y));
        coords
            .into_iter()
            .enumerate()
            .map(|(idx, coord)| (coord, u32::try_from(idx).expect("qubit-index space is u32")))
            .collect()
    }

    /// The number of distinct qubit coordinates in the circuit.
    ///
    /// # Panics
    ///
    /// Panics if the qubit count exceeds `u32::MAX`.
    pub fn num_qubits(&self) -> u32 {
        u32::try_from(self.qubits().len()).expect("qubit count fits u32")
    }

    /// The number of allocated measurement records.
    pub fn num_measurements(&self) -> u32 {
        self.meas_registry.records().len() as u32
    }

    /// Appends a [`Op::Tick`] moment barrier to the entry body.
    pub fn tick(&mut self) {
        self.entry_ops_mut().push(Op::Tick);
    }

    /// Appends a [`Op::Repeat`] of `body` to the entry body.
    pub fn push_repeat(&mut self, body: BodyId, repetitions: u32) {
        self.entry_ops_mut().push(Op::Repeat { body, repetitions });
    }

    /// Appends a gate acting on `targets` to the entry body.
    ///
    /// Empty `targets` is a no-op. Two-qubit gates consume targets pairwise.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::InvalidGateTargetCount`] if a two-qubit gate is
    /// given an odd number of targets.
    pub fn do_gate(
        &mut self,
        gate: GateType,
        targets: impl IntoIterator<Item = IVec2>,
    ) -> Result<(), CircuitError> {
        let qubits: Vec<IVec2> = targets.into_iter().collect();
        if qubits.is_empty() {
            return Ok(());
        }
        if gate.is_two_qubit_gate() && !qubits.len().is_multiple_of(2) {
            return Err(CircuitError::InvalidGateTargetCount {
                gate,
                targets: qubits.len(),
            });
        }

        self.entry_ops_mut().push(Op::Gate { gate, qubits });
        Ok(())
    }

    /// Appends a noiseless single-qubit measurement of each target in `basis`,
    /// returning one freshly allocated measurement id per target (in target
    /// order). [`crate::NoiseModel::noisy_circuit`] can add measurement noise.
    ///
    /// Empty `targets` is a no-op returning an empty vector; duplicate targets
    /// each receive a distinct id.
    pub fn measure(
        &mut self,
        basis: PauliBasis,
        targets: impl IntoIterator<Item = IVec2>,
    ) -> Vec<u32> {
        let qubits = targets.into_iter().collect::<Vec<_>>();
        if qubits.is_empty() {
            return Vec::new();
        }

        let measurements = self.meas_registry.allocate(qubits.iter().copied());
        self.entry_ops_mut().push(Op::Measure {
            basis,
            qubits,
            measurements: measurements.clone(),
            flip_probability: 0.0,
        });
        measurements
    }

    /// Append a multi-Pauli-product measurement ([`Op::MPP`]): each product is
    /// measured jointly and produces one record, keyed by the product's
    /// representative coordinate (see [`PauliMap::representative_coord`]) so
    /// records stay consistent across cloning. Empty products are rejected.
    ///
    /// [`PauliMap::representative_coord`]: crate::PauliMap::representative_coord
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::EmptyPauliProduct`] without modifying the circuit
    /// if any product is empty.
    pub fn measure_pauli_products(
        &mut self,
        products: impl IntoIterator<Item = crate::PauliMap>,
    ) -> Result<Vec<u32>, CircuitError> {
        let products = products.into_iter().collect::<Vec<_>>();
        if products.is_empty() {
            return Ok(Vec::new());
        }
        let representatives = products
            .iter()
            .map(|product| {
                product
                    .representative_coord()
                    .ok_or(CircuitError::EmptyPauliProduct)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let measurements = self.meas_registry.allocate(representatives);
        self.entry_ops_mut().push(Op::MPP {
            products,
            measurements: measurements.clone(),
        });
        Ok(measurements)
    }

    /// The output column each measurement id lands in once every
    /// [`Op::Repeat`] is unrolled, together with the total number of records
    /// the unrolled circuit emits.
    ///
    /// Backends address measurement records by position, but this circuit
    /// names them by id and hides repetition behind `Repeat`, so the two
    /// spaces only line up after unrolling. [`Self::flatten`] also unrolls,
    /// but it rewrites the circuit and mints fresh ids to do so; a caller that
    /// only wants "which record does this id land in" pays neither cost here.
    ///
    /// A measurement inside a loop occurs once per iteration; the column
    /// reported is its **last** occurrence, matching the latest-occurrence
    /// rule both [`Self::flatten`] and the Stim emission frame use, so an id
    /// read from outside a loop resolves to the same record everywhere.
    ///
    /// # Errors
    ///
    /// Returns [`CircuitError::InvalidCircuitBody`] on a dangling or
    /// self-recursive body reference, and
    /// [`MeasurementFrameError::EmittedMeasurementCountOverflow`] when the
    /// unrolled circuit would emit more than `u32::MAX` records, or
    /// [`MeasurementFrameError::MeasurementCountOutOfRange`] if one operation
    /// already contains more than `u32::MAX` records.
    ///
    /// [`MeasurementFrameError::EmittedMeasurementCountOverflow`]: crate::MeasurementFrameError
    ///
    /// # Examples
    ///
    /// ```
    /// # use bloq_circuit::{CircuitBody, CoordCircuit, Op, PauliBasis};
    /// # use glam::ivec2;
    /// let qubit = ivec2(0, 0);
    /// let mut circuit = CoordCircuit::new();
    /// let looped = circuit.reserve_measurement_id(qubit);
    /// let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
    ///     basis: PauliBasis::Z,
    ///     qubits: vec![qubit],
    ///     measurements: vec![looped],
    ///     flip_probability: 0.0,
    /// }]));
    /// circuit.push_repeat(body, 3);
    ///
    /// let columns = circuit.expanded_measurement_columns()?;
    /// assert_eq!(columns.count(), 3);
    /// assert_eq!(columns.column(looped), Some(2), "the last of three occurrences");
    /// # Ok::<(), bloq_circuit::CircuitError>(())
    /// ```
    pub fn expanded_measurement_columns(&self) -> Result<ExpandedMeasurementColumns, CircuitError> {
        let mut walk = ExpandedColumnWalk {
            circuit: self,
            columns: FxHashMap::default(),
            count: 0,
            body_counts: FxHashMap::default(),
            stack: Vec::new(),
        };
        walk.visit(self.entry)?;
        Ok(ExpandedMeasurementColumns {
            columns: walk.columns,
            count: walk.count,
        })
    }

    /// Whether any body contains an [`Op::Repeat`].
    pub fn has_repeats(&self) -> bool {
        self.bodies
            .iter()
            .any(|body| body.ops().iter().any(|op| matches!(op, Op::Repeat { .. })))
    }

    /// Unroll every [`Op::Repeat`] into a single straight-line entry body,
    /// returning the replay trace side-table owners need to translate
    /// loop-relative annotations (see [`FlattenEvent`]).
    ///
    /// Measurement-id semantics mirror the emission frame's
    /// latest-occurrence rule: every unrolled occurrence of a measurement gets
    /// a fresh id **except the last, which keeps the original id**, so
    /// references from outside the loop (which always meant "the latest
    /// occurrence") stay valid without any remap. `ConditionalPauli`
    /// measurement controls are rewritten to the occurrence emitted most
    /// recently before them. Zero-repetition repeats are dropped.
    ///
    /// A circuit without repeats is returned untouched with an empty trace.
    ///
    /// # Errors
    ///
    /// [`CircuitError::InvalidCircuitBody`] on a dangling or self-recursive
    /// body reference, [`CircuitError::FlattenMeasurementIdOverflow`] when the
    /// unrolled circuit would need more than `u32::MAX` measurement ids, or
    /// [`CircuitError::FlattenResourceLimit`] when expansion exceeds the default
    /// [`FlattenLimits`].
    pub fn flatten(&mut self) -> Result<Vec<FlattenEvent>, CircuitError> {
        self.flatten_with_limits(FlattenLimits::default())
    }

    /// [`Self::flatten`] with an explicit allowance for expanded work.
    ///
    /// The shared body DAG is counted before the circuit is changed. Rejected
    /// expansions leave both operations and measurement identities untouched.
    ///
    /// # Errors
    ///
    /// Returns the same body, id-space, and resource errors as [`Self::flatten`].
    pub fn flatten_with_limits(
        &mut self,
        limits: FlattenLimits,
    ) -> Result<Vec<FlattenEvent>, CircuitError> {
        if !self.has_repeats() {
            return Ok(Vec::new());
        }

        let (mut counts, _) = self.flatten_counts(limits)?;

        // Pass 2: unroll. The bodies are moved out so fresh-id allocation can
        // borrow the registry mutably while walking them.
        let bodies = std::mem::take(&mut self.bodies);
        let mut out = Vec::new();
        let mut trace = Vec::new();
        let mut latest = FxHashMap::default();
        let result = self.unroll_body(
            &bodies,
            self.entry,
            &mut counts,
            &mut latest,
            &mut out,
            &mut trace,
        );
        match result {
            Ok(()) => {
                self.bodies = vec![CircuitBody::from_ops(out)];
                self.entry = BodyId(0);
                Ok(trace)
            }
            Err(error) => {
                // Leave the circuit structurally intact on failure (freshly
                // allocated ids may remain in the registry, but no op uses them).
                self.bodies = bodies;
                Err(error)
            }
        }
    }

    /// Check expansion without changing the circuit and return its work cost.
    /// A caller flattening several circuits can charge the returned cost to
    /// one shared allowance before materializing any of them.
    ///
    /// # Errors
    ///
    /// Returns the same body, id-space, and resource errors as [`Self::flatten`].
    pub fn check_flatten_limits(&self, limits: FlattenLimits) -> Result<usize, CircuitError> {
        if !self.has_repeats() {
            return Ok(0);
        }
        self.flatten_counts(limits).map(|(_, work)| work)
    }

    fn flatten_counts(
        &self,
        limits: FlattenLimits,
    ) -> Result<(FxHashMap<u32, u64>, usize), CircuitError> {
        // Visit shared bodies once, without using the native call stack. Zero
        // repetitions contribute no executed references, matching unrolling.
        let mut order = Vec::new();
        let mut active = FxHashSet::default();
        let mut visited = FxHashSet::default();
        let mut pending = vec![(self.entry, 0usize)];
        active.insert(self.entry);
        while let Some((body, next)) = pending.last_mut() {
            let ops = self
                .body(*body)
                .ok_or(CircuitError::InvalidCircuitBody(*body))?
                .ops();
            let Some(op) = ops.get(*next) else {
                let body = *body;
                pending.pop();
                active.remove(&body);
                visited.insert(body);
                order.push(body);
                continue;
            };
            *next += 1;
            if let Op::Repeat { body, repetitions } = op
                && *repetitions != 0
                && !visited.contains(body)
            {
                if !active.insert(*body) {
                    return Err(CircuitError::InvalidCircuitBody(*body));
                }
                pending.push((*body, 0));
            }
        }

        // Propagate invocation multiplicities along DAG edges, rather than
        // revisiting a body for every path through the DAG.
        let mut measured_bodies = FxHashSet::default();
        for &body in &order {
            if self
                .body(body)
                .expect("body order was checked")
                .ops()
                .iter()
                .any(|op| match op {
                    Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                        !measurements.is_empty()
                    }
                    Op::Repeat { body, repetitions } => {
                        *repetitions != 0 && measured_bodies.contains(body)
                    }
                    _ => false,
                })
            {
                measured_bodies.insert(body);
            }
        }
        let mut invocations = FxHashMap::default();
        invocations.insert(self.entry, 1u64);
        let mut counts = FxHashMap::default();
        for &body in order.iter().rev() {
            let Some(&multiplier) = invocations.get(&body) else {
                continue;
            };
            for op in self.body(body).expect("body order was checked").ops() {
                match op {
                    Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                        for &measurement in measurements {
                            let count = counts.entry(measurement).or_insert(0u64);
                            *count = count
                                .checked_add(multiplier)
                                .ok_or(CircuitError::FlattenMeasurementIdOverflow)?;
                        }
                    }
                    Op::Repeat { body, repetitions }
                        if *repetitions != 0 && measured_bodies.contains(body) =>
                    {
                        let extra = multiplier
                            .checked_mul(u64::from(*repetitions))
                            .ok_or(CircuitError::FlattenMeasurementIdOverflow)?;
                        let count = invocations.entry(*body).or_insert(0u64);
                        *count = count
                            .checked_add(extra)
                            .ok_or(CircuitError::FlattenMeasurementIdOverflow)?;
                    }
                    _ => {}
                }
            }
        }
        let fresh_needed = counts
            .values()
            .try_fold(0u64, |total, &count| total.checked_add(count - 1))
            .ok_or(CircuitError::FlattenMeasurementIdOverflow)?;
        if fresh_needed > u64::from(u32::MAX - self.meas_registry.next_id()) {
            return Err(CircuitError::FlattenMeasurementIdOverflow);
        }

        let mut work_by_body = FxHashMap::<BodyId, usize>::default();
        for body in order {
            let mut work = 1usize; // Visiting even an empty body costs work.
            for op in self.body(body).expect("body order was checked").ops() {
                let extra = match op {
                    Op::Repeat { body, repetitions } if *repetitions != 0 => {
                        // LoopEnter/LoopExit plus a body visit and IterationEnd
                        // for each repetition. Empty repeats are not free.
                        work_by_body[body]
                            .saturating_add(1)
                            .saturating_mul(*repetitions as usize)
                            .saturating_add(2)
                    }
                    Op::Measure {
                        qubits,
                        measurements,
                        ..
                    } => qubits
                        .len()
                        .saturating_add(measurements.len().saturating_mul(2)),
                    Op::MPP {
                        products,
                        measurements,
                    } => products
                        .iter()
                        .fold(products.len(), |total, product| {
                            total.saturating_add(product.len())
                        })
                        .saturating_add(measurements.len().saturating_mul(2)),
                    Op::Gate { qubits, .. }
                    | Op::Depolarize1 { qubits, .. }
                    | Op::Depolarize2 { qubits, .. }
                    | Op::PauliError { qubits, .. } => qubits.len(),
                    Op::ConditionalPauli(corrections) => corrections.len(),
                    Op::Tick | Op::Repeat { .. } => 0,
                };
                work = work.saturating_add(1).saturating_add(extra);
            }
            work_by_body.insert(body, work);
        }
        let observed = work_by_body[&self.entry];
        if observed == usize::MAX || observed > limits.max_work {
            return Err(CircuitError::FlattenResourceLimit {
                observed,
                limit: limits.max_work,
            });
        }
        Ok((counts, observed))
    }

    fn unroll_body(
        &mut self,
        bodies: &[CircuitBody],
        body: BodyId,
        counts: &mut FxHashMap<u32, u64>,
        latest: &mut FxHashMap<u32, u32>,
        out: &mut Vec<Op>,
        trace: &mut Vec<FlattenEvent>,
    ) -> Result<(), CircuitError> {
        let mut pending = vec![(body, 0usize, 1u32, false)];
        while let Some((body, next, remaining, repeated)) = pending.last_mut() {
            let ops = bodies
                .get(body.0 as usize)
                .ok_or(CircuitError::InvalidCircuitBody(*body))?
                .ops();
            let Some(op) = ops.get(*next) else {
                if *repeated {
                    trace.push(FlattenEvent::IterationEnd { body: *body });
                }
                if *remaining > 1 {
                    *remaining -= 1;
                    *next = 0;
                } else {
                    if *repeated {
                        trace.push(FlattenEvent::LoopExit { body: *body });
                    }
                    pending.pop();
                }
                continue;
            };
            *next += 1;
            match op {
                Op::Repeat { body, repetitions } => {
                    if *repetitions == 0 {
                        continue;
                    }
                    trace.push(FlattenEvent::LoopEnter { body: *body });
                    pending.push((*body, 0, *repetitions, true));
                }
                Op::Measure {
                    basis,
                    qubits,
                    measurements,
                    flip_probability,
                } => {
                    let measurements = self.remap_unrolled_measurements(
                        measurements,
                        qubits.iter().copied(),
                        counts,
                        latest,
                        trace,
                    );
                    out.push(Op::Measure {
                        basis: *basis,
                        qubits: qubits.clone(),
                        measurements,
                        flip_probability: *flip_probability,
                    });
                }
                Op::MPP {
                    products,
                    measurements,
                } => {
                    let record_qubits = products
                        .iter()
                        .map(|product| {
                            product
                                .representative_coord()
                                .ok_or(CircuitError::EmptyPauliProduct)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let measurements = self.remap_unrolled_measurements(
                        measurements,
                        record_qubits,
                        counts,
                        latest,
                        trace,
                    );
                    out.push(Op::MPP {
                        products: products.clone(),
                        measurements,
                    });
                }
                // A measurement control refers to the occurrence emitted most
                // recently before it (the current iteration's id) — exactly the
                // `latest` remap, which `remap_op_measurement_ids` applies while
                // passing `Value` controls through. Absent ids stay unchanged.
                Op::ConditionalPauli(_) => {
                    let mut op = op.clone();
                    remap_op_measurement_ids(&mut op, latest);
                    out.push(op);
                }
                Op::Gate { .. }
                | Op::Depolarize1 { .. }
                | Op::Depolarize2 { .. }
                | Op::PauliError { .. }
                | Op::Tick => out.push(op.clone()),
            }
        }
        Ok(())
    }

    /// Emit one unrolled occurrence of each measurement: fresh ids until the
    /// countdown says this is the final occurrence, which keeps the original.
    fn remap_unrolled_measurements(
        &mut self,
        measurements: &[u32],
        record_qubits: impl IntoIterator<Item = IVec2>,
        counts: &mut FxHashMap<u32, u64>,
        latest: &mut FxHashMap<u32, u32>,
        trace: &mut Vec<FlattenEvent>,
    ) -> Vec<u32> {
        measurements
            .iter()
            .zip(record_qubits)
            .map(|(&original, qubit)| {
                let remaining = counts
                    .get_mut(&original)
                    .expect("pass 1 counted every measurement pass 2 visits");
                *remaining -= 1;
                let emitted = if *remaining == 0 {
                    original
                } else {
                    self.meas_registry.allocate_one(qubit)
                };
                latest.insert(original, emitted);
                trace.push(FlattenEvent::Measurement { original, emitted });
                emitted
            })
            .collect()
    }

    /// Checks if there are any operations that act on the same qubit within
    /// the same moment. A repeat does not imply a tick, including between its
    /// iterations.
    pub fn has_moments_conflict(&self) -> bool {
        self.body_moment_summary(self.entry, &mut FxHashMap::default(), &mut Vec::new())
            .is_none_or(|summary| summary.conflict)
    }

    fn body_moment_summary(
        &self,
        body: BodyId,
        cache: &mut FxHashMap<BodyId, MomentSummary>,
        stack: &mut Vec<BodyId>,
    ) -> Option<MomentSummary> {
        if let Some(summary) = cache.get(&body) {
            return Some(summary.clone());
        }
        if stack.contains(&body) {
            return None;
        }
        let body_ops = self.body(body)?.ops();
        stack.push(body);
        let mut summary = MomentSummary::default();

        for op in body_ops {
            let next = match op {
                Op::Tick => MomentSummary::tick(),
                Op::Repeat { body, repetitions } => {
                    if *repetitions == 0 {
                        continue;
                    }
                    let Some(body) = self.body_moment_summary(*body, cache, stack) else {
                        stack.pop();
                        return None;
                    };
                    body.repeated(*repetitions)
                }
                Op::Gate { qubits, .. } | Op::Measure { qubits, .. } => {
                    MomentSummary::from_qubits(qubits)
                }
                // Noise decorates physical operations in the same moment; it
                // does not claim the qubit a second time.
                Op::Depolarize1 { .. } | Op::Depolarize2 { .. } | Op::PauliError { .. } => {
                    MomentSummary::default()
                }
                // MPP and ConditionalPauli may legally share qubits within a
                // moment, so they don't contribute to conflict detection.
                Op::ConditionalPauli(_) | Op::MPP { .. } => MomentSummary::default(),
            };
            summary.append(next);
        }

        stack.pop();
        cache.insert(body, summary.clone());
        Some(summary)
    }

    fn for_each_qubit(&self, mut visit: impl FnMut(IVec2)) {
        for body in &self.bodies {
            for op in body.ops() {
                match op {
                    Op::Gate { qubits, .. }
                    | Op::Measure { qubits, .. }
                    | Op::Depolarize1 { qubits, .. }
                    | Op::Depolarize2 { qubits, .. }
                    | Op::PauliError { qubits, .. } => {
                        qubits.iter().copied().for_each(&mut visit);
                    }
                    Op::ConditionalPauli(corrections) => {
                        corrections
                            .iter()
                            .map(|correction| correction.target)
                            .for_each(&mut visit);
                    }
                    Op::MPP { products, .. } => {
                        for product in products {
                            product.iter().map(|(coord, _)| *coord).for_each(&mut visit);
                        }
                    }
                    Op::Repeat { .. } | Op::Tick => {}
                }
            }
        }
        self.meas_registry
            .records()
            .iter()
            .map(|record| record.qubit)
            .for_each(visit);
    }

    fn fmt_body(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        body: BodyId,
        indent: usize,
    ) -> std::fmt::Result {
        let body = self.body(body).ok_or(std::fmt::Error)?;
        self.fmt_ops(f, body.ops(), indent)
    }

    fn fmt_ops(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        ops: &[Op],
        indent: usize,
    ) -> std::fmt::Result {
        for (index, op) in ops.iter().enumerate() {
            if index > 0 {
                writeln!(f)?;
            }
            self.fmt_op(f, op, indent)?;
        }
        Ok(())
    }

    fn fmt_op(&self, f: &mut std::fmt::Formatter<'_>, op: &Op, indent: usize) -> std::fmt::Result {
        write_indent(f, indent)?;
        match op {
            Op::Gate { gate, qubits } => {
                write!(f, "{gate}")?;
                write_coord_targets(f, qubits)
            }
            Op::Measure {
                basis,
                qubits,
                flip_probability,
                ..
            } => {
                write!(f, "{}", measurement_gate_name(*basis))?;
                if *flip_probability != 0.0 {
                    write!(f, "({flip_probability})")?;
                }
                write_coord_targets(f, qubits)
            }
            Op::MPP { products, .. } => {
                write!(f, "MPP")?;
                for product in products {
                    write!(f, " ")?;
                    for (index, (coord, pauli)) in product.iter().enumerate() {
                        if index > 0 {
                            write!(f, "*")?;
                        }
                        write!(f, "{pauli}({},{})", coord.x, coord.y)?;
                    }
                }
                Ok(())
            }
            Op::Depolarize1 {
                probability,
                qubits,
            } => {
                write!(f, "DEPOLARIZE1({probability})")?;
                write_coord_targets(f, qubits)
            }
            Op::Depolarize2 {
                probability,
                qubits,
            } => {
                write!(f, "DEPOLARIZE2({probability})")?;
                write_coord_targets(f, qubits)
            }
            Op::PauliError {
                probability,
                pauli,
                qubits,
            } => {
                write!(f, "{pauli}_ERROR({probability})")?;
                write_coord_targets(f, qubits)
            }
            Op::Repeat { body, repetitions } => {
                write!(f, "REPEAT {repetitions} {{")?;
                let repeat_body = self.body(*body).ok_or(std::fmt::Error)?;
                if !repeat_body.ops().is_empty() {
                    writeln!(f)?;
                    self.fmt_ops(f, repeat_body.ops(), indent + 1)?;
                    writeln!(f)?;
                    write_indent(f, indent)?;
                }
                write!(f, "}}")
            }
            Op::Tick => write!(f, "TICK"),
            Op::ConditionalPauli(corrections) => {
                write!(f, "CPAULI")?;
                for correction in corrections {
                    write!(
                        f,
                        " {}[{}]({},{})",
                        correction.pauli,
                        correction.control,
                        correction.target.x,
                        correction.target.y
                    )?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Default)]
struct MomentSummary {
    first: FxHashSet<IVec2>,
    last: FxHashSet<IVec2>,
    has_tick: bool,
    conflict: bool,
}

impl MomentSummary {
    fn from_qubits(qubits: &[IVec2]) -> Self {
        let first: FxHashSet<_> = qubits.iter().copied().collect();
        Self {
            last: first.clone(),
            conflict: first.len() != qubits.len(),
            first,
            has_tick: false,
        }
    }

    fn tick() -> Self {
        Self {
            has_tick: true,
            ..Self::default()
        }
    }

    fn repeated(mut self, repetitions: u32) -> Self {
        if repetitions > 1 && !self.first.is_disjoint(&self.last) {
            self.conflict = true;
        }
        self
    }

    fn append(&mut self, next: Self) {
        self.conflict |= next.conflict || !self.last.is_disjoint(&next.first);
        if !self.has_tick {
            self.first.extend(next.first.iter().copied());
        }
        if next.has_tick {
            self.last = next.last;
        } else {
            self.last.extend(next.last);
        }
        self.has_tick |= next.has_tick;
    }
}

/// Where each measurement id lands once every [`Op::Repeat`] is unrolled — the
/// result of [`CoordCircuit::expanded_measurement_columns`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpandedMeasurementColumns {
    columns: FxHashMap<u32, u32>,
    count: u32,
}

impl ExpandedMeasurementColumns {
    /// The output column of `measurement`'s last occurrence, or `None` if the
    /// circuit never emits it.
    pub fn column(&self, measurement: u32) -> Option<u32> {
        self.columns.get(&measurement).copied()
    }

    /// How many measurement records the unrolled circuit emits.
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Every `(measurement, column)` pair, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.columns
            .iter()
            .map(|(&measurement, &column)| (measurement, column))
    }
}

/// Accumulator for [`CoordCircuit::expanded_measurement_columns`].
struct ExpandedColumnWalk<'a> {
    circuit: &'a CoordCircuit,
    columns: FxHashMap<u32, u32>,
    count: u32,
    /// Unrolled record count per body, memoized: loop bodies are visited once
    /// to count and once to record, and a body may be shared by several loops.
    body_counts: FxHashMap<BodyId, u32>,
    stack: Vec<BodyId>,
}

impl ExpandedColumnWalk<'_> {
    fn visit(&mut self, body: BodyId) -> Result<(), CircuitError> {
        // A body reachable from itself would unroll forever; validated
        // circuits are acyclic, so this guards hand-built ones.
        if self.stack.contains(&body) {
            return Err(CircuitError::InvalidCircuitBody(body));
        }
        self.stack.push(body);
        let ops = self
            .circuit
            .body(body)
            .ok_or(CircuitError::InvalidCircuitBody(body))?
            .ops();
        for op in ops {
            match op {
                Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                    for &measurement in measurements {
                        self.columns.insert(measurement, self.count);
                        self.count = advance(self.count, 1)?;
                    }
                }
                Op::Repeat { body, repetitions } => {
                    if *repetitions == 0 {
                        continue;
                    }
                    // Only the final iteration's columns survive the
                    // latest-occurrence rule, and the earlier ones do nothing
                    // but advance the column counter — which is a
                    // multiplication. So skip to the last iteration instead of
                    // walking a loop that may repeat billions of times.
                    let per_iteration = self.body_count(*body)?;
                    let skipped = per_iteration.checked_mul(*repetitions - 1).ok_or(
                        MeasurementFrameError::RepeatedMeasurementCountOverflow {
                            measurements_per_iteration: per_iteration,
                            repetitions: *repetitions,
                        },
                    )?;
                    self.count = advance(self.count, skipped)?;
                    self.visit(*body)?;
                }
                Op::Gate { .. }
                | Op::Depolarize1 { .. }
                | Op::Depolarize2 { .. }
                | Op::PauliError { .. }
                | Op::Tick
                | Op::ConditionalPauli(_) => {}
            }
        }
        self.stack.pop();
        Ok(())
    }

    /// How many records one unrolled iteration of `body` emits.
    fn body_count(&mut self, body: BodyId) -> Result<u32, CircuitError> {
        if let Some(&count) = self.body_counts.get(&body) {
            return Ok(count);
        }
        if self.stack.contains(&body) {
            return Err(CircuitError::InvalidCircuitBody(body));
        }
        self.stack.push(body);
        let ops = self
            .circuit
            .body(body)
            .ok_or(CircuitError::InvalidCircuitBody(body))?
            .ops();
        let mut count = 0u32;
        for op in ops {
            match op {
                Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                    count = advance(count, measurement_count(measurements.len())?)?;
                }
                Op::Repeat { body, repetitions } => {
                    let per_iteration = self.body_count(*body)?;
                    let total = per_iteration.checked_mul(*repetitions).ok_or(
                        MeasurementFrameError::RepeatedMeasurementCountOverflow {
                            measurements_per_iteration: per_iteration,
                            repetitions: *repetitions,
                        },
                    )?;
                    count = advance(count, total)?;
                }
                Op::Gate { .. }
                | Op::Depolarize1 { .. }
                | Op::Depolarize2 { .. }
                | Op::PauliError { .. }
                | Op::Tick
                | Op::ConditionalPauli(_) => {}
            }
        }
        self.stack.pop();
        self.body_counts.insert(body, count);
        Ok(count)
    }
}

fn advance(count: u32, additional: u32) -> Result<u32, MeasurementFrameError> {
    count
        .checked_add(additional)
        .ok_or(MeasurementFrameError::EmittedMeasurementCountOverflow {
            emitted_count: count,
            additional,
        })
}

fn measurement_count(measurements: usize) -> Result<u32, MeasurementFrameError> {
    u32::try_from(measurements)
        .map_err(|_| MeasurementFrameError::MeasurementCountOutOfRange { measurements })
}

pub(crate) fn checked_body_id(index: usize) -> Result<BodyId, CircuitError> {
    u32::try_from(index)
        .map(BodyId)
        .map_err(|_| CircuitError::BodyIdOutOfRange { index })
}

fn deserialize_bodies<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<CircuitBody>, D::Error> {
    let bodies = <Vec<CircuitBody> as serde::Deserialize>::deserialize(deserializer)?;
    if let Some(last) = bodies.len().checked_sub(1) {
        checked_body_id(last).map_err(serde::de::Error::custom)?;
    }
    Ok(bodies)
}

fn remap_op_measurement_ids(op: &mut Op, remap: &rustc_hash::FxHashMap<u32, u32>) {
    match op {
        // An MPP produces measurement records, so its ids follow the same
        // remap as ordinary measurements.
        Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
            for measurement in measurements {
                if let Some(mapped) = remap.get(measurement).copied() {
                    *measurement = mapped;
                }
            }
        }
        // A conditional Pauli's control follows the same remap as the
        // measurement that produced it.
        Op::ConditionalPauli(corrections) => {
            for correction in corrections {
                if let Some(mapped) = remap.get(&correction.control).copied() {
                    correction.control = mapped;
                }
            }
        }
        Op::Gate { .. }
        | Op::Repeat { .. }
        | Op::Depolarize1 { .. }
        | Op::Depolarize2 { .. }
        | Op::PauliError { .. }
        | Op::Tick => {}
    }
}

impl std::fmt::Display for CoordCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_body(f, self.entry, 0)
    }
}

/// The Stim gate name that measures in `basis`.
pub fn measurement_gate_name(basis: PauliBasis) -> &'static str {
    match basis {
        PauliBasis::X => "MX",
        PauliBasis::Y => "MY",
        PauliBasis::Z => "M",
    }
}

fn write_indent(f: &mut std::fmt::Formatter<'_>, indent: usize) -> std::fmt::Result {
    for _ in 0..indent {
        write!(f, "    ")?;
    }
    Ok(())
}

fn write_coord_targets(f: &mut std::fmt::Formatter<'_>, qubits: &[IVec2]) -> std::fmt::Result {
    for &coord in qubits {
        write!(f, " ({},{})", coord.x, coord.y)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejected_mpp_keeps_the_circuit_and_next_record_unchanged() {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [IVec2::ZERO]);
        let before = circuit.clone();
        let product = crate::PauliMap::from_iter([(IVec2::X, crate::Pauli::X)]);

        assert_eq!(
            circuit.measure_pauli_products([product.clone(), crate::PauliMap::empty()]),
            Err(CircuitError::EmptyPauliProduct)
        );
        assert_eq!(circuit, before);
        assert_eq!(circuit.measure_pauli_products([product]).unwrap(), [1]);
    }

    #[test]
    fn circuit_roundtrip_retains_body_ids_and_entry() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Tick]));
        circuit.set_entry_body(body).unwrap();
        let encoded = postcard::to_allocvec(&circuit).unwrap();
        let decoded: CoordCircuit = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded, circuit);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn wide_counts_do_not_alias_u32_identifiers() {
        let maximum = u32::MAX as usize;
        assert_eq!(measurement_count(maximum), Ok(u32::MAX));
        assert_eq!(checked_body_id(maximum), Ok(BodyId(u32::MAX)));
        assert_eq!(
            measurement_count(maximum + 1),
            Err(MeasurementFrameError::MeasurementCountOutOfRange {
                measurements: maximum + 1,
            })
        );
        assert_eq!(
            checked_body_id(maximum + 1),
            Err(CircuitError::BodyIdOutOfRange { index: maximum + 1 })
        );
    }

    #[test]
    fn display_uses_coordinate_targets_and_nested_repeat_bodies() {
        let qubit = glam::ivec2(2, 3);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [qubit]).unwrap();
        circuit.tick();
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![qubit],
                measurements: vec![0],
                flip_probability: 0.0,
            },
            Op::Repeat {
                body: BodyId(2),
                repetitions: 2,
            },
        ]));
        let nested = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::X,
            qubits: vec![qubit],
        }]));

        assert_eq!(nested, BodyId(2));
        circuit.push_repeat(body, 3);

        assert_eq!(
            circuit.to_string(),
            "H (2,3)\nTICK\nREPEAT 3 {\n    M (2,3)\n    REPEAT 2 {\n        X (2,3)\n    }\n}"
        );
    }

    #[test]
    fn has_moments_conflict_crosses_repeat_boundaries() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, [qubit]).unwrap();
        let repeated = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::X,
            qubits: vec![qubit],
        }]));
        circuit.push_repeat(repeated, 1);

        assert!(circuit.has_moments_conflict());

        let mut circuit = CoordCircuit::new();
        let repeated = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::X,
            qubits: vec![qubit],
        }]));
        circuit.push_repeat(repeated, 2);
        assert!(circuit.has_moments_conflict());

        circuit
            .body_mut(repeated)
            .unwrap()
            .ops_mut()
            .insert(0, Op::Tick);
        assert!(!circuit.has_moments_conflict());
    }

    #[test]
    fn measure_allocates_ids_for_each_target_without_sorting() {
        let first = glam::ivec2(0, 0);
        let second = glam::ivec2(1, 0);
        let mut circuit = CoordCircuit::new();

        circuit.measure(PauliBasis::Z, [second, first]);

        let body = circuit.body(circuit.entry_body()).unwrap();
        assert!(matches!(
            &body.ops()[0],
            Op::Measure { basis: PauliBasis::Z, qubits, measurements, .. }
                if qubits == &[second, first]
                    && measurements == &[0, 1]
        ));
        assert_eq!(circuit.meas_registry().records().len(), 2);
    }

    #[test]
    fn duplicate_measurement_targets_keep_distinct_measurement_ids() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();

        circuit.measure(PauliBasis::Z, [qubit, qubit]);

        let body = circuit.body(circuit.entry_body()).unwrap();
        assert!(matches!(
            &body.ops()[0],
            Op::Measure { measurements, .. }
                if measurements == &[0, 1]
        ));
    }

    #[test]
    fn qubits_deduplicate_dense_and_extreme_coordinates() {
        let dense = [glam::ivec2(0, 0), glam::ivec2(1, 0)];
        let mut circuit = CoordCircuit::new();
        circuit
            .do_gate(GateType::H, [dense[0], dense[1], dense[0]])
            .unwrap();
        assert_eq!(circuit.qubits(), dense.into_iter().collect());

        let extremes = [glam::IVec2::MIN, glam::IVec2::MAX];
        let mut circuit = CoordCircuit::new();
        circuit.do_gate(GateType::H, extremes).unwrap();
        assert_eq!(circuit.qubits(), extremes.into_iter().collect());
    }

    #[test]
    fn qubits_include_products_registered_records_and_unreferenced_bodies() {
        let coords = [IVec2::ZERO, IVec2::X, IVec2::Y, IVec2::ONE];
        let mut circuit = CoordCircuit::new();
        circuit.reserve_measurement_id(coords[0]);
        let control = circuit
            .measure_pauli_products([crate::PauliMap::from_iter([
                (coords[1], crate::Pauli::X),
                (coords[2], crate::Pauli::Z),
            ])])
            .unwrap()[0];
        circuit.add_body(CircuitBody::from_ops(vec![Op::ConditionalPauli(vec![
            crate::ConditionalCorrection {
                pauli: PauliBasis::X,
                control,
                target: coords[3],
            },
        ])]));
        assert_eq!(circuit.qubits(), coords.into_iter().collect());
    }

    #[test]
    fn flatten_rejects_huge_empty_and_measurement_free_repeats_before_mutation() {
        for ops in [
            Vec::new(),
            vec![Op::Tick],
            vec![Op::Gate {
                gate: GateType::H,
                qubits: vec![IVec2::ZERO],
            }],
        ] {
            let mut circuit = CoordCircuit::new();
            let body = circuit.add_body(CircuitBody::from_ops(ops));
            circuit.push_repeat(body, u32::MAX);
            let original = circuit.clone();
            assert!(matches!(
                circuit.flatten(),
                Err(CircuitError::FlattenResourceLimit { observed, limit }) if observed > limit
            ));
            assert_eq!(circuit, original);
        }
    }

    #[test]
    fn flatten_limits_count_targets_and_can_be_overridden() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::H,
            qubits: vec![IVec2::ZERO; 8],
        }]));
        circuit.push_repeat(body, 3);
        let original = circuit.clone();
        let Err(CircuitError::FlattenResourceLimit { observed, .. }) =
            circuit.flatten_with_limits(FlattenLimits { max_work: 8 })
        else {
            panic!("target copies must consume the expansion allowance");
        };
        assert_eq!(circuit, original);
        circuit
            .flatten_with_limits(FlattenLimits { max_work: observed })
            .unwrap();
        assert_eq!(circuit.body(circuit.entry_body()).unwrap().ops().len(), 3);
        assert!(!circuit.has_repeats());
    }

    #[test]
    fn flatten_counts_shared_body_paths_and_handles_deep_bodies_iteratively() {
        let mut shared = CoordCircuit::new();
        let mut body = shared.add_body(CircuitBody::new());
        for _ in 0..60 {
            body = shared.add_body(CircuitBody::from_ops(vec![
                Op::Repeat {
                    body,
                    repetitions: 1,
                },
                Op::Repeat {
                    body,
                    repetitions: 1,
                },
            ]));
        }
        shared.push_repeat(body, 1);
        assert!(matches!(
            shared.flatten(),
            Err(CircuitError::FlattenResourceLimit { .. })
        ));

        let mut deep = CoordCircuit::new();
        let mut body = deep.add_body(CircuitBody::from_ops(vec![Op::Tick]));
        for _ in 0..4096 {
            body = deep.add_body(CircuitBody::from_ops(vec![Op::Repeat {
                body,
                repetitions: 1,
            }]));
        }
        deep.push_repeat(body, 1);
        deep.flatten().unwrap();
        assert_eq!(deep.body(deep.entry_body()).unwrap().ops(), &[Op::Tick]);
    }

    #[test]
    fn flatten_unrolls_repeat_and_keeps_final_occurrence_ids() {
        let seed_qubit = glam::ivec2(0, 0);
        let loop_qubit = glam::ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        let seed = circuit.measure(PauliBasis::Z, [seed_qubit])[0];
        let repeated = circuit.reserve_measurement_id(loop_qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![loop_qubit],
            measurements: vec![repeated],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, 3);

        let trace = circuit.flatten().unwrap();

        assert!(!circuit.has_repeats());
        assert_eq!(circuit.body_count(), 1);
        let ops = circuit.body(circuit.entry_body()).unwrap().ops();
        let emitted: Vec<u32> = ops
            .iter()
            .filter_map(|op| match op {
                Op::Measure { measurements, .. } => Some(measurements[0]),
                _ => None,
            })
            .collect();
        // Iterations 1 and 2 get fresh ids (2, 3); the final iteration keeps
        // the original id so outside references still mean "latest occurrence".
        assert_eq!(emitted, vec![seed, 2, 3, repeated]);
        assert_eq!(
            trace,
            vec![
                FlattenEvent::Measurement {
                    original: seed,
                    emitted: seed
                },
                FlattenEvent::LoopEnter { body },
                FlattenEvent::Measurement {
                    original: repeated,
                    emitted: 2
                },
                FlattenEvent::IterationEnd { body },
                FlattenEvent::Measurement {
                    original: repeated,
                    emitted: 3
                },
                FlattenEvent::IterationEnd { body },
                FlattenEvent::Measurement {
                    original: repeated,
                    emitted: repeated
                },
                FlattenEvent::IterationEnd { body },
                FlattenEvent::LoopExit { body },
            ]
        );
        // Fresh ids record the measured qubit like the originals.
        for id in [2, 3] {
            assert_eq!(
                circuit.meas_registry().record(id).unwrap().qubit,
                loop_qubit
            );
        }
    }

    #[test]
    fn flatten_unrolls_nested_repeats_multiplicatively() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let inner = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![measurement],
            flip_probability: 0.0,
        }]));
        let outer = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
            body: inner,
            repetitions: 2,
        }]));
        circuit.push_repeat(outer, 3);

        circuit.flatten().unwrap();

        let ops = circuit.body(circuit.entry_body()).unwrap().ops();
        assert_eq!(ops.len(), 6, "3 * 2 unrolled measurements");
        let last = match &ops[5] {
            Op::Measure { measurements, .. } => measurements[0],
            other => panic!("expected measure, got {other:?}"),
        };
        assert_eq!(last, measurement, "final occurrence keeps the original id");
        assert_eq!(circuit.num_measurements(), 6);
    }

    #[test]
    fn flatten_reports_total_fresh_measurement_overflow() {
        let qubits = [glam::ivec2(0, 0), glam::ivec2(1, 0)];
        let mut circuit = CoordCircuit::new();
        let measurements = qubits.map(|qubit| circuit.reserve_measurement_id(qubit));
        let inner = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: qubits.to_vec(),
            measurements: measurements.to_vec(),
            flip_probability: 0.0,
        }]));
        let outer = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
            body: inner,
            repetitions: u32::MAX,
        }]));
        circuit.push_repeat(outer, u32::MAX);

        assert!(matches!(
            circuit.flatten(),
            Err(CircuitError::FlattenMeasurementIdOverflow)
        ));
    }

    #[test]
    fn flatten_reports_per_measurement_occurrence_overflow() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let measured = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![measurement],
            flip_probability: 0.0,
        }]));
        let wrapper = circuit.add_body(CircuitBody::from_ops(vec![Op::Repeat {
            body: measured,
            repetitions: u32::MAX,
        }]));
        circuit
            .body_mut(circuit.entry_body())
            .unwrap()
            .ops_mut()
            .extend([
                Op::Repeat {
                    body: wrapper,
                    repetitions: u32::MAX,
                },
                Op::Repeat {
                    body: wrapper,
                    repetitions: u32::MAX,
                },
            ]);

        assert!(matches!(
            circuit.flatten(),
            Err(CircuitError::FlattenMeasurementIdOverflow)
        ));
    }

    #[test]
    fn flatten_drops_zero_repetition_repeats_and_noops_without_repeats() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [qubit]);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Gate {
            gate: GateType::X,
            qubits: vec![qubit],
        }]));
        circuit.push_repeat(body, 0);

        let trace = circuit.flatten().unwrap();
        assert!(
            trace
                .iter()
                .all(|event| matches!(event, FlattenEvent::Measurement { .. }))
        );
        assert_eq!(circuit.body(circuit.entry_body()).unwrap().ops().len(), 1);

        // Already flat: untouched, empty trace.
        let ops_before = circuit.body(circuit.entry_body()).unwrap().ops().to_vec();
        let trace = circuit.flatten().unwrap();
        assert!(trace.is_empty());
        assert_eq!(
            circuit.body(circuit.entry_body()).unwrap().ops(),
            &ops_before
        );
    }

    #[test]
    fn flatten_remaps_conditional_pauli_controls_to_current_iteration() {
        use crate::ConditionalCorrection;
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let measurement = circuit.reserve_measurement_id(qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![
            Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![qubit],
                measurements: vec![measurement],
                flip_probability: 0.0,
            },
            Op::ConditionalPauli(vec![ConditionalCorrection {
                pauli: PauliBasis::X,
                control: measurement,
                target: qubit,
            }]),
        ]));
        circuit.push_repeat(body, 2);

        circuit.flatten().unwrap();

        let ops = circuit.body(circuit.entry_body()).unwrap().ops();
        let controls: Vec<u32> = ops
            .iter()
            .filter_map(|op| match op {
                Op::ConditionalPauli(corrections) => Some(corrections[0].control),
                _ => None,
            })
            .collect();
        let measurements: Vec<u32> = ops
            .iter()
            .filter_map(|op| match op {
                Op::Measure { measurements, .. } => Some(measurements[0]),
                _ => None,
            })
            .collect();
        assert_eq!(
            controls, measurements,
            "each control follows its own iteration"
        );
    }

    #[test]
    fn flatten_rejects_self_recursive_body() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::new());
        circuit.body_mut(body).unwrap().ops_mut().push(Op::Repeat {
            body,
            repetitions: 2,
        });
        circuit.push_repeat(body, 2);

        assert!(matches!(
            circuit.flatten(),
            Err(CircuitError::InvalidCircuitBody(recursive)) if recursive == body
        ));
    }

    /// A measurement in a `REPEAT` body, plus one after the loop.
    fn looped_and_trailing_circuit(repetitions: u32) -> (CoordCircuit, u32, u32) {
        let looped_qubit = glam::ivec2(0, 0);
        let trailing_qubit = glam::ivec2(1, 0);
        let mut circuit = CoordCircuit::new();
        let looped = circuit.reserve_measurement_id(looped_qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![looped_qubit],
            measurements: vec![looped],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, repetitions);
        let trailing = circuit.measure(PauliBasis::Z, [trailing_qubit])[0];
        (circuit, looped, trailing)
    }

    #[test]
    fn expanded_columns_report_the_last_occurrence_of_a_looped_measurement() {
        let (circuit, looped, trailing) = looped_and_trailing_circuit(4);

        let columns = circuit.expanded_measurement_columns().unwrap();

        assert_eq!(columns.count(), 5);
        assert_eq!(columns.column(looped), Some(3));
        assert_eq!(columns.column(trailing), Some(4));
        assert_eq!(columns.column(99), None);
    }

    /// The unrolled column of the final occurrence is a multiplication, not a
    /// walk, so a loop far too large to unroll still resolves instantly.
    #[test]
    fn expanded_columns_resolve_a_loop_too_large_to_unroll() {
        let (circuit, looped, trailing) = looped_and_trailing_circuit(1_000_000_000);

        let columns = circuit.expanded_measurement_columns().unwrap();

        assert_eq!(columns.column(looped), Some(999_999_999));
        assert_eq!(columns.column(trailing), Some(1_000_000_000));
    }

    #[test]
    fn expanded_columns_agree_with_flatten() {
        let (mut circuit, looped, trailing) = looped_and_trailing_circuit(3);
        let columns = circuit.expanded_measurement_columns().unwrap();

        circuit.flatten().unwrap();

        // Flatten keeps the final occurrence's original id, so the emitted
        // order of the flat circuit indexes the same columns.
        let emitted: Vec<u32> = circuit
            .body(circuit.entry_body())
            .unwrap()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::Measure { measurements, .. } => Some(measurements[0]),
                _ => None,
            })
            .collect();
        assert_eq!(emitted.len() as u32, columns.count());
        for measurement in [looped, trailing] {
            let position = emitted.iter().rposition(|&id| id == measurement).unwrap();
            assert_eq!(columns.column(measurement), Some(position as u32));
        }
    }

    /// The contract spelled out literally: re-walk a `Repeat` body once per
    /// repetition and let later iterations overwrite earlier ones.
    /// [`CoordCircuit::expanded_measurement_columns`] skips straight to the
    /// final iteration instead, so this is the oracle that keeps the
    /// optimization honest.
    fn naive_expanded_columns(circuit: &CoordCircuit) -> (FxHashMap<u32, u32>, u32) {
        fn walk(
            circuit: &CoordCircuit,
            body: BodyId,
            columns: &mut FxHashMap<u32, u32>,
            count: &mut u32,
        ) {
            for op in circuit.body(body).expect("body exists").ops() {
                match op {
                    Op::Measure { measurements, .. } | Op::MPP { measurements, .. } => {
                        for &measurement in measurements {
                            columns.insert(measurement, *count);
                            *count += 1;
                        }
                    }
                    Op::Repeat { body, repetitions } => {
                        for _ in 0..*repetitions {
                            walk(circuit, *body, columns, count);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut columns = FxHashMap::default();
        let mut count = 0;
        walk(circuit, circuit.entry_body(), &mut columns, &mut count);
        (columns, count)
    }

    /// Build a circuit from a nesting spec: `repeats[i]` is the repetition
    /// count of loop level `i`, innermost last, each level measuring one qubit.
    fn nested_repeat_circuit(repeats: &[u32], trailing: bool) -> CoordCircuit {
        let mut circuit = CoordCircuit::new();
        let mut inner: Option<BodyId> = None;
        for level in (0..repeats.len()).rev() {
            let qubit = glam::ivec2(level as i32, 0);
            let measurement = circuit.reserve_measurement_id(qubit);
            let mut ops = vec![Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![qubit],
                measurements: vec![measurement],
                flip_probability: 0.0,
            }];
            if let Some(inner) = inner {
                ops.push(Op::Repeat {
                    body: inner,
                    repetitions: repeats[level + 1],
                });
            }
            inner = Some(circuit.add_body(CircuitBody::from_ops(ops)));
        }
        if let Some(outermost) = inner {
            circuit.push_repeat(outermost, repeats[0]);
        }
        if trailing {
            circuit.measure(PauliBasis::Z, [glam::ivec2(-1, 0)]);
        }
        circuit
    }

    /// The loop-skipping optimization must agree with the literal re-walk on
    /// every shape, not just the flat one: nested loop counts multiply.
    #[test]
    fn expanded_columns_agree_with_a_literal_repeat_rewalk() {
        let shapes: &[&[u32]] = &[
            &[],
            &[1],
            &[7],
            &[3, 4],
            &[2, 3, 5],
            &[1, 1, 1],
            &[6, 1, 2, 3],
        ];
        for shape in shapes {
            for trailing in [false, true] {
                let circuit = nested_repeat_circuit(shape, trailing);
                let expected = naive_expanded_columns(&circuit);
                assert_eq!(
                    expected.1,
                    shape
                        .iter()
                        .rev()
                        .fold(0, |inner, count| count * (1 + inner))
                        + u32::from(trailing)
                );

                let actual = circuit.expanded_measurement_columns().unwrap();

                assert_eq!(actual.count(), expected.1, "shape {shape:?} {trailing}");
                assert_eq!(
                    actual.iter().collect::<std::collections::BTreeMap<_, _>>(),
                    expected.0.into_iter().collect(),
                    "shape {shape:?} trailing={trailing}"
                );
            }
        }
    }

    /// A zero-repetition loop contributes nothing in either walk, and must not
    /// leave the column counter shifted for what follows it.
    #[test]
    fn expanded_columns_skip_zero_repetition_loops_like_the_rewalk() {
        let qubit = glam::ivec2(0, 0);
        let mut circuit = CoordCircuit::new();
        let skipped = circuit.reserve_measurement_id(qubit);
        let body = circuit.add_body(CircuitBody::from_ops(vec![Op::Measure {
            basis: PauliBasis::Z,
            qubits: vec![qubit],
            measurements: vec![skipped],
            flip_probability: 0.0,
        }]));
        circuit.push_repeat(body, 0);
        let after = circuit.measure(PauliBasis::Z, [qubit])[0];

        let columns = circuit.expanded_measurement_columns().unwrap();

        assert_eq!(
            (
                columns.iter().collect::<std::collections::BTreeMap<_, _>>(),
                columns.count()
            ),
            {
                let (expected, count) = naive_expanded_columns(&circuit);
                (expected.into_iter().collect(), count)
            }
        );
        assert_eq!(columns.column(skipped), None, "never emitted");
        assert_eq!(columns.column(after), Some(0), "counter not shifted");
    }

    #[test]
    fn expanded_columns_reject_a_self_recursive_body() {
        let mut circuit = CoordCircuit::new();
        let body = circuit.add_body(CircuitBody::new());
        circuit.body_mut(body).unwrap().ops_mut().push(Op::Repeat {
            body,
            repetitions: 2,
        });
        circuit.push_repeat(body, 2);

        assert!(matches!(
            circuit.expanded_measurement_columns(),
            Err(CircuitError::InvalidCircuitBody(recursive)) if recursive == body
        ));
    }

    #[test]
    fn do_gate_rejects_odd_two_qubit_targets() {
        let qubits = [IVec2::new(0, 0), IVec2::new(1, 0), IVec2::new(2, 0)];
        let mut circuit = CoordCircuit::new();

        let err = circuit
            .do_gate(GateType::CX, qubits)
            .expect_err("odd two-qubit targets should fail");

        assert_eq!(
            err,
            CircuitError::InvalidGateTargetCount {
                gate: GateType::CX,
                targets: 3,
            }
        );
        assert!(circuit.body(circuit.entry_body()).unwrap().ops().is_empty());
    }
}
