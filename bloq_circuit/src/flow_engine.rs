//! Flow matching across chunk boundaries.

use std::collections::hash_map::Entry;
use std::fmt;
use std::vec::Drain;

use glam::IVec2;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::chunk::{Flow, FlowMarker};
use crate::error::FlowError;
use crate::pauli_map::PauliMap;
use crate::{
    DetectorParity, DetectorTerm, LoopCarriedDetectorState, LoopStateId,
    checked_translate_coordinate,
};

/// Measurements carried by an open or completed flow.
///
/// Detector flows almost always carry one or two measurements (the current
/// round's outcome plus the consumed previous round's), so they stay inline
/// and flow tracking does not touch the heap per flow. The inline capacity
/// deliberately matches [`DetectorParity`]'s term buffer so completed flows
/// convert into parities by moving the buffer.
pub type FlowEngineMeasurements<M> = SmallVec<[M; 2]>;

/// A matched flow ready to emit as a detector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletedFlow<M> {
    /// Every measurement fused along the chain.
    pub measurements: FlowEngineMeasurements<M>,
    /// XOR of the signs on every fused flow.
    pub sign: bool,
    /// The detector's coordinates, if the chain carries one.
    pub center: Option<IVec2>,
    /// Fused marker over every flow in this chain. See [`FlowMarker`]: a
    /// `Discard` chain emits no detector; a `Restart` chain emits a restart
    /// parity instead.
    pub marker: FlowMarker,
}

/// A completed flow classified for detector synthesis.
///
/// Classification uses the fused [`FlowMarker`]. Produced by
/// [`FlowEngine::drain_composed_chains`]; a
/// `Discard`-marked or empty-parity chain is dropped and never appears here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposedChain<M> {
    /// An ordinary detector at the chain's center coordinates.
    Detector {
        /// The detector's parity.
        parity: DetectorParity<M>,
        /// The detector's coordinates, if known.
        center: Option<IVec2>,
    },
    /// A `RepeatUntilSuccess` restart syndrome (fused marker `Restart`); no
    /// coordinates, as restart parities are never decoder food.
    Restart {
        /// The restart post-selection syndrome.
        parity: DetectorParity<M>,
    },
}

/// A boundary flow left open after residual composition.
///
/// [`FlowEngine::into_residual_flows`] returns one fused chain spliced from
/// `start` (its origin boundary, empty for an internally-created chain) to `end`
/// (the boundary it leaves open, empty if it closed against a boundary input),
/// carrying every measurement composed along the way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualFlow<M> {
    /// The chain's origin boundary; empty for an internally-created chain.
    pub start: PauliMap,
    /// The boundary the chain leaves open; empty if it closed against an input.
    pub end: PauliMap,
    /// Every measurement composed along the chain.
    pub measurements: FlowEngineMeasurements<M>,
    /// XOR of the signs on every fused flow.
    pub sign: bool,
    /// The detector's coordinates, if any.
    pub center: Option<IVec2>,
    /// Carries the fused [`FlowMarker`] onto the template's `boundary_flows` so
    /// the instance-seam pass can suppress detectors (`Discard`) or route the
    /// cross-instance chain to a restart parity (`Restart`).
    pub marker: FlowMarker,
}

#[derive(Clone)]
struct OpenFlow<M> {
    measurements: FlowEngineMeasurements<M>,
    sign: bool,
    center: Option<IVec2>,
    /// Fused [`FlowMarker`] over every flow spliced into this chain so far.
    marker: FlowMarker,
    /// The chain's origin boundary key, threaded through fusion so the residual
    /// can report where the chain began. Absent for an internally-created chain
    /// (one that opened with an empty `start`); the entering key for a chain that
    /// began at an unmatched boundary `start` under
    /// [`UnmatchedInputMode::KeepAsResidual`]. Always absent under the `Error`
    /// and `Skip` modes, which never admit a boundary-entering chain.
    origin_start: Option<Box<PauliMap>>,
}

/// A chunk's flows plus their absolute XY placement offset.
///
/// The lower phase appends block-template flows directly with
/// the instantiating node's offset instead of materializing translated
/// copies per node.
#[derive(Debug, Clone, Copy)]
pub struct OffsetFlows<'c> {
    /// The chunk's template-local flows.
    pub flows: &'c [Flow],
    /// XY offset placing the flows in absolute coordinates.
    pub offset: IVec2,
}

impl<'c> OffsetFlows<'c> {
    /// Pairs a flow slice with its placement offset.
    pub fn new(flows: &'c [Flow], offset: IVec2) -> Self {
        Self { flows, offset }
    }
}

/// A borrowed, translated open-flow boundary key.
///
/// The key holds a [`PauliMap`] plus its placement offset. Hashing and equality
/// act on the translated entries, so equal
/// absolute boundaries match regardless of which (map, offset) pair
/// produced them, and matching never materializes a translated map. The
/// translated hash is computed once at construction: table growth re-hashes
/// every stored key, so a recomputed hash would re-translate every entry on
/// each rehash.
#[derive(Debug, Clone, Copy)]
pub struct FlowKey<'c> {
    map: &'c PauliMap,
    offset: IVec2,
    hash: u64,
}

impl<'c> FlowKey<'c> {
    /// Borrow a boundary at a checked placement without allocating a translated map.
    ///
    /// # Errors
    ///
    /// Returns [`crate::CoordinateOverflowError`] if a translated coordinate
    /// exceeds the signed 32-bit lattice.
    pub fn try_new(
        map: &'c PauliMap,
        offset: IVec2,
    ) -> Result<Self, crate::CoordinateOverflowError> {
        for (coord, _) in map {
            checked_translate_coordinate(*coord, offset)?;
        }
        Ok(Self::new(map, offset))
    }

    fn new(map: &'c PauliMap, offset: IVec2) -> Self {
        use std::hash::Hasher as _;
        let mut hasher = rustc_hash::FxHasher::default();
        map.hash_translated(offset, &mut hasher);
        Self {
            map,
            offset,
            hash: hasher.finish(),
        }
    }

    fn translated(&self) -> PauliMap {
        self.map.translated(self.offset)
    }
}

impl std::hash::Hash for FlowKey<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for FlowKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        if self.offset.to_array() == other.offset.to_array() {
            return self.map == other.map;
        }
        self.map.len() == other.map.len()
            && self.map.iter().zip(other.map.iter()).all(
                |((coord, pauli), (other_coord, other_pauli))| {
                    *coord + self.offset == *other_coord + other.offset && pauli == other_pauli
                },
            )
    }
}

impl Eq for FlowKey<'_> {}

impl fmt::Display for FlowKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.translated().fmt(f)
    }
}

/// Tracks open flows and completed detectors while chunks are appended.
///
/// Open-flow keys borrow the boundary [`PauliMap`]s of the appended chunks
/// (lifetime `'c`), so matching flows across chunks never clones a key. This
/// works because every appended flow outlives the engine; the engine is a
/// per-pass scratch structure, not a long-lived store.
#[must_use]
#[derive(Clone)]
pub struct FlowEngine<'c, M> {
    open_flows: FxHashMap<FlowKey<'c>, OpenFlow<M>>,
    // Each group drains these payloads before publishing outputs. Keep the
    // empty buffer for the next group instead of allocating it every round.
    matched_inputs: Vec<Option<OpenFlow<M>>>,
    /// Known stabilizers that commute through a temporary gauge basis but are
    /// not themselves measured by it. They rejoin the active span when the
    /// original basis is restored.
    latent_flows: Vec<(PauliMap, OpenFlow<M>)>,
    completed: Vec<CompletedFlow<M>>,
    /// Boundary chains that closed against a boundary *input* (a flow that
    /// entered at an unmatched `start` and later reached an empty `end`).
    /// Populated only by [`UnmatchedInputMode::KeepAsResidual`]; empty under the
    /// other modes, so it costs nothing when residual extraction is unused.
    residual_completed: Vec<ResidualFlow<M>>,
}

impl<'c, M> FlowEngine<'c, M> {
    /// Creates an empty flow-composition engine.
    pub fn new() -> Self {
        Self {
            open_flows: FxHashMap::default(),
            matched_inputs: Vec::new(),
            latent_flows: Vec::new(),
            completed: Vec::new(),
            residual_completed: Vec::new(),
        }
    }

    /// Return completed flows after requiring all active flows to terminate.
    /// Latent stabilizers that survive the final measured-check span are output
    /// logical correlations; observable lowering owns them, not the detector
    /// side table.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if any flow remains unterminated.
    pub fn finish(self) -> Result<Vec<CompletedFlow<M>>, FlowError> {
        let active: Vec<String> = self.open_flows.keys().map(ToString::to_string).collect();
        if !active.is_empty() {
            return Err(FlowError::Composition(format!(
                "unterminated flows: {}",
                active.join("; ")
            )));
        }
        Ok(self.completed)
    }

    /// Append one logical chunk assembled from several offset flow slices
    /// (one per node member), while allowing previously-open boundary flows
    /// to remain open for later chunks.
    ///
    /// The mapper receives the slice index alongside each measurement
    /// reference so every member can resolve against its own template
    /// measurement table.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if boundaries cannot be translated or composed.
    pub fn append_group_allowing_dangling(
        &mut self,
        parts: &[OffsetFlows<'c>],
        meas_mapper: impl FnMut(usize, &u32) -> M,
    ) -> Result<(), FlowError> {
        self.append_group_with_mode(parts, meas_mapper, UnmatchedInputMode::Error, None)
    }

    /// Append a boundary group while allowing its stabilizer span to differ
    /// from the currently-open span. Exact matching remains the fast path. At
    /// a gauge transition, products in the two spans' intersection emit
    /// detectors; old-only generators retire and the group's ordinary creator
    /// flows start its new-only generators.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if boundaries cannot be translated or composed.
    pub fn append_group_allowing_stabilizer_transition(
        &mut self,
        parts: &[OffsetFlows<'c>],
        mut meas_mapper: impl FnMut(usize, &u32) -> M,
    ) -> Result<(), FlowError>
    where
        M: Copy,
    {
        let mut consumer_count = 0;
        for part in parts {
            for flow in part.flows.iter().filter(|flow| !flow.start.is_empty()) {
                for (coord, _) in &flow.start {
                    checked_translate_coordinate(*coord, part.offset)?;
                }
                consumer_count += 1;
            }
        }
        if !self.latent_flows.is_empty() {
            return self.append_stabilizer_transition(parts, &mut meas_mapper);
        }

        // Exact matching is overwhelmingly the common case. Remove each input
        // once here and pass its payload into fusion, instead of probing it with
        // `contains_key` and then hashing the same boundary again in
        // `process_flow`. On a miss, restore the table before taking the
        // stabilizer-transition path.
        let consumer_keys = || {
            parts.iter().flat_map(|part| {
                part.flows
                    .iter()
                    .filter(|flow| !flow.start.is_empty())
                    .map(|flow| FlowKey::new(&flow.start, part.offset))
            })
        };
        let mut matched_inputs = std::mem::take(&mut self.matched_inputs);
        matched_inputs.reserve(consumer_count);
        for target in consumer_keys() {
            let Some(open) = self.open_flows.remove(&target) else {
                let duplicate_consumer = consumer_keys()
                    .take(matched_inputs.len())
                    .any(|seen| seen == target);
                for (key, open) in consumer_keys().zip(matched_inputs.into_iter().flatten()) {
                    self.open_flows.insert(key, open);
                }
                if !duplicate_consumer {
                    return self.append_stabilizer_transition(parts, &mut meas_mapper);
                }
                // Let the ordinary path report its usual unmatched-input error.
                return self.append_group_with_mode(
                    parts,
                    meas_mapper,
                    UnmatchedInputMode::Error,
                    None,
                );
            };
            matched_inputs.push(Some(open));
        }
        self.append_group_with_mode(
            parts,
            meas_mapper,
            UnmatchedInputMode::Error,
            Some(matched_inputs),
        )
    }

    /// Append chunks while ignoring flows whose start boundary is not already
    /// open. This is useful when extracting template-local detectors: flows
    /// that enter through a template boundary are handled later by instance
    /// lowering, not by the reusable template side table.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if boundaries cannot be translated or composed.
    pub fn append_group_skipping_unmatched_inputs(
        &mut self,
        parts: &[OffsetFlows<'c>],
        meas_mapper: impl FnMut(usize, &u32) -> M,
    ) -> Result<(), FlowError> {
        self.append_group_with_mode(parts, meas_mapper, UnmatchedInputMode::Skip, None)
    }

    /// Append chunks for **residual extraction**: like
    /// [`Self::append_group_skipping_unmatched_inputs`], but a flow whose `start`
    /// is not already open is *kept* as a boundary-entering chain (recording
    /// where it entered) rather than dropped. Combined with
    /// [`Self::into_residual_flows`], this composes a template's open boundary
    /// flows into fused chains in one pass: internal detectors fall out via
    /// [`Self::drain_completed`] as usual, while every flow touching the boundary
    /// survives in the residual.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if boundaries cannot be translated or composed.
    pub fn append_residual_group(
        &mut self,
        parts: &[OffsetFlows<'c>],
        meas_mapper: impl FnMut(usize, &u32) -> M,
    ) -> Result<(), FlowError> {
        self.append_group_with_mode(parts, meas_mapper, UnmatchedInputMode::KeepAsResidual, None)
    }

    fn append_group_with_mode(
        &mut self,
        parts: &[OffsetFlows<'c>],
        mut meas_mapper: impl FnMut(usize, &u32) -> M,
        unmatched_input_mode: UnmatchedInputMode,
        matched_inputs: Option<Vec<Option<OpenFlow<M>>>>,
    ) -> Result<(), FlowError> {
        let mut inputs = match matched_inputs {
            Some(inputs) => inputs,
            None => {
                let mut inputs = std::mem::take(&mut self.matched_inputs);
                for part in parts {
                    for flow in part.flows.iter().filter(|flow| !flow.start.is_empty()) {
                        for (coord, _) in &flow.start {
                            checked_translate_coordinate(*coord, part.offset)?;
                        }
                        inputs.push(
                            self.open_flows
                                .remove(&FlowKey::new(&flow.start, part.offset)),
                        );
                    }
                }
                inputs
            }
        };
        let mut matched_inputs = inputs.drain(..);
        // Only non-empty outputs occupy the table. Consumer-only detector
        // flows can outnumber creators, and need no hash-table capacity.
        self.open_flows.reserve(
            parts
                .iter()
                .flat_map(|part| part.flows)
                .filter(|flow| !flow.end.is_empty())
                .count(),
        );

        // Consumers must see previous open flows before creators can replace
        // keys, across all of the chunk's member slices.
        for (part_index, part) in parts.iter().enumerate() {
            for flow in part.flows {
                if !flow.start.is_empty() {
                    let previous = matched_inputs
                        .next()
                        .expect("pre-matched inputs cover every consumer");
                    self.process_flow(
                        flow,
                        part.offset,
                        part_index,
                        &mut meas_mapper,
                        unmatched_input_mode,
                        previous,
                    )?;
                }
            }
        }
        debug_assert!(
            matched_inputs.as_slice().is_empty(),
            "every pre-matched input is consumed"
        );
        drop(matched_inputs);
        self.matched_inputs = inputs;
        for (part_index, part) in parts.iter().enumerate() {
            for flow in part.flows {
                if flow.start.is_empty() {
                    self.process_flow(
                        flow,
                        part.offset,
                        part_index,
                        &mut meas_mapper,
                        unmatched_input_mode,
                        None,
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Drains the completed (internal, boundary-detached) detector chains,
    /// leaving open flows untouched for later chunks.
    pub fn drain_completed(&mut self) -> Drain<'_, CompletedFlow<M>> {
        self.completed.drain(..)
    }

    /// Consume the engine after a residual composition (chunks appended with
    /// [`Self::append_residual_group`]) and return every boundary flow as a
    /// fused chain: the chains still open (`origin_start → open key`) plus those
    /// that closed against a boundary input (`origin_start → ∅`). Chains that
    /// closed with an empty origin are internal detectors; they collect in
    /// `completed` (drainable via [`Self::drain_completed`]) and are excluded here.
    ///
    /// The caller owns ordering: `open_flows` iterates in hash order, so a caller
    /// that needs determinism (e.g. reproducible detector emission) must sort the
    /// result by a canonical key.
    pub fn into_residual_flows(self) -> Vec<ResidualFlow<M>> {
        let mut flows = self.residual_completed;
        flows.extend(
            self.open_flows
                .into_iter()
                .map(|(key, open)| (key.translated(), open))
                .chain(self.latent_flows)
                .map(|(end, open)| ResidualFlow {
                    start: open
                        .origin_start
                        .map_or_else(PauliMap::empty, |start| *start),
                    end,
                    measurements: open.measurements,
                    sign: open.sign,
                    center: open.center,
                    marker: open.marker,
                }),
        );
        flows
    }

    fn process_flow(
        &mut self,
        flow: &'c Flow,
        offset: IVec2,
        part_index: usize,
        meas_mapper: &mut impl FnMut(usize, &u32) -> M,
        unmatched_input_mode: UnmatchedInputMode,
        previous: Option<OpenFlow<M>>,
    ) -> Result<(), FlowError> {
        // Flow keys translate coordinates inside Hash/Eq. Check before a key
        // reaches either trait so malformed API/binary input returns an error
        // instead of debug-panicking or release-wrapping.
        if offset != IVec2::ZERO {
            for (coord, _) in flow.start.iter().chain(flow.end.iter()) {
                checked_translate_coordinate(*coord, offset)?;
            }
            if let Some(center) = flow.center {
                checked_translate_coordinate(center, offset)?;
            }
        }

        let mut measurements: FlowEngineMeasurements<M> =
            SmallVec::with_capacity(flow.measurements.len());
        for measurement in &flow.measurements {
            measurements.push(meas_mapper(part_index, measurement));
        }

        let mut center = flow.center.map(|center| center + offset);
        let mut sign = flow.sign;
        let mut marker = flow.marker;
        // The chain's origin boundary, carried so a residual composition can
        // report where each open chain began. Empty unless this flow enters at
        // an unmatched boundary `start` (see `KeepAsResidual` below) or inherits
        // a non-empty origin from the chain it consumes.
        let mut origin_start = None;

        if !flow.start.is_empty() {
            // Inputs were removed before any output of this group was published.
            match previous {
                Some(prev) => {
                    // The chunk's own measurements come first, then the
                    // carried history.
                    measurements.extend(prev.measurements);
                    sign ^= prev.sign;
                    center = center.or(prev.center);
                    origin_start = prev.origin_start;
                    // Two flows fusing into one chain can mix discard and
                    // restart; downstream routing checks `Discard` first, so a
                    // fall-through would silently drop the restart parity. This
                    // is a topology-dependent conflict, so reject rather than panic.
                    marker = marker.fuse(prev.marker).ok_or_else(|| {
                        FlowError::Composition(
                            "fused chain mixes discard and restart components".into(),
                        )
                    })?;
                }
                None => match unmatched_input_mode {
                    UnmatchedInputMode::Skip => return Ok(()),
                    // A boundary input: keep the chain open, recording where it
                    // entered so the residual can close it against a neighbor.
                    UnmatchedInputMode::KeepAsResidual => {
                        origin_start = Some(Box::new(flow.start.translated(offset)));
                    }
                    UnmatchedInputMode::Error => {
                        return Err(FlowError::Composition(format!(
                            "unmatched flow input {}",
                            flow.start.translated(offset)
                        )));
                    }
                },
            }
        }

        if !flow.end.is_empty() {
            // Flow continues: store for future matching
            let end_key = FlowKey::new(&flow.end, offset);
            let Entry::Vacant(entry) = self.open_flows.entry(end_key) else {
                return Err(FlowError::Composition(format!(
                    "multiple flows create boundary {end_key}"
                )));
            };
            entry.insert(OpenFlow {
                measurements,
                sign,
                center,
                origin_start,
                marker,
            });
        } else if !measurements.is_empty() || sign || marker == FlowMarker::Discard {
            // A chain reached an empty end. If it began at a boundary input
            // (non-empty origin) it is a residual flow that closes against a
            // neighbor; otherwise it closed entirely within this composition (an
            // internal detector). Non-residual modes never set a non-empty
            // origin, so they always take the `completed` branch.
            //
            // A measurement-free chain is normally dropped (it can never carry a
            // detector parity). A *discarding* one is kept: it exists precisely to
            // consume an incoming seam stabilizer with no reconstruction of its
            // own (a selective block's opposite-basis stabilizer), so it must
            // survive onto the residual to match that seam and suppress its
            // detector. `discard` is always false in the non-selective paths.
            if let Some(origin_start) = origin_start {
                self.residual_completed.push(ResidualFlow {
                    start: *origin_start,
                    end: PauliMap::empty(),
                    measurements,
                    sign,
                    center,
                    marker,
                });
            } else {
                self.completed.push(CompletedFlow {
                    measurements,
                    sign,
                    center,
                    marker,
                });
            }
        }

        Ok(())
    }

    fn append_stabilizer_transition(
        &mut self,
        parts: &[OffsetFlows<'c>],
        meas_mapper: &mut impl FnMut(usize, &u32) -> M,
    ) -> Result<(), FlowError>
    where
        M: Copy,
    {
        let mut incoming = Vec::new();
        for (part_index, part) in parts.iter().enumerate() {
            for flow in part.flows.iter().filter(|flow| !flow.start.is_empty()) {
                if !flow.end.is_empty() {
                    return Err(FlowError::Composition(
                        "stabilizer-span transitions require boundary consumer flows".into(),
                    ));
                }
                let mut payload = empty_open_flow();
                payload.measurements.extend(
                    flow.measurements
                        .iter()
                        .map(|measurement| meas_mapper(part_index, measurement)),
                );
                payload.sign = flow.sign;
                payload.center = flow
                    .center
                    .map(|center| checked_translate_coordinate(center, part.offset))
                    .transpose()?;
                payload.marker = flow.marker;
                incoming.push((flow.start.translated(part.offset), payload));
            }
        }

        let mut old: Vec<_> = std::mem::take(&mut self.open_flows)
            .into_iter()
            .map(|(key, flow)| (key.translated(), flow))
            .chain(std::mem::take(&mut self.latent_flows))
            .collect();
        old.sort_by(|(a, _), (b, _)| pauli_map_cmp(a, b));
        let old_len = old.len();
        // Borrow the boundaries: reduction clones each one as it consumes it.
        let generators: Vec<&PauliMap> = old
            .iter()
            .map(|(map, _)| map)
            .chain(incoming.iter().map(|(start, _)| start))
            .collect();
        let dependencies = pauli_dependencies(&generators);
        for (index, mask) in dependencies {
            if index < old_len {
                return Err(FlowError::Composition(
                    "open flow boundaries are not independent generators".into(),
                ));
            }
            if first_set_bit(&mask).is_none_or(|first| first >= old_len) {
                return Err(FlowError::Composition(
                    "incoming flow boundaries are not independent generators".into(),
                ));
            }
            let mut completed = empty_open_flow();
            let mut product = PauliMap::empty();
            for (index, (map, flow)) in incoming.iter().enumerate() {
                if mask[(index + old_len) / 64] & (1 << ((index + old_len) % 64)) != 0 {
                    multiply_pauli_flow(&mut product, &mut completed, map, flow)?;
                }
            }
            for (index, (map, flow)) in old.iter().enumerate() {
                if mask[index / 64] & (1 << (index % 64)) != 0 {
                    multiply_pauli_flow(&mut product, &mut completed, map, flow)?;
                }
            }
            if !completed.measurements.is_empty()
                || completed.sign
                || completed.marker == FlowMarker::Discard
            {
                self.completed.push(CompletedFlow {
                    measurements: completed.measurements,
                    sign: completed.sign,
                    center: completed.center,
                    marker: completed.marker,
                });
            }
        }

        // Measuring a new stabilizer destroys one anticommuting old generator;
        // products of the remaining anticommuting generators with that pivot
        // survive. This is the usual stabilizer-tableau measurement update.
        let mut commutations = pauli_commutation_columns(
            old.iter().map(|(map, _)| map),
            incoming.iter().map(|(map, _)| map),
        );
        for measured in 0..incoming.len() {
            let mut targets = std::mem::take(&mut commutations[measured]);
            let Some(pivot) = first_set_bit(&targets) else {
                continue;
            };
            let last = old.len() - 1;
            let (pivot_map, pivot_flow) = old.swap_remove(pivot);
            swap_remove_bit(&mut targets, pivot, last);
            for (word, &bits) in targets.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    let target = word * 64 + bits.trailing_zeros() as usize;
                    let (known, flow) = &mut old[target];
                    multiply_pauli_flow(known, flow, &pivot_map, &pivot_flow)?;
                    bits &= bits - 1;
                }
            }
            // Commutation is bilinear: multiplying by the pivot toggles exactly
            // its anticommuting columns. Keep the original swap-remove order.
            for column in &mut commutations[measured + 1..] {
                let anticommutes = column[pivot / 64] & (1 << (pivot % 64)) != 0;
                swap_remove_bit(column, pivot, last);
                if anticommutes {
                    xor_words(column, &targets);
                }
            }
        }

        // Keep only surviving old stabilizers outside the newly measured span.
        // Exact active checks are represented by the creator flows below.
        let mut basis = FxHashMap::default();
        for (measured, flow) in &incoming {
            let mut measured = measured.clone();
            let mut flow = flow.clone();
            reduce_pauli_flow(&mut measured, &mut flow, &basis)?;
            let pivot = pauli_pivot(&measured)
                .expect("incoming flow boundaries were checked as independent");
            basis.insert(pivot, (measured, flow));
        }
        for (mut known, mut flow) in old {
            reduce_pauli_flow(&mut known, &mut flow, &basis)?;
            let Some(pivot) = pauli_pivot(&known) else {
                continue;
            };
            basis.insert(pivot, (known.clone(), flow.clone()));
            self.latent_flows.push((known, flow));
        }

        // The incoming consumers have now either contributed to an intersection
        // detector or initialized a new-only gauge check. In both cases they do
        // not remain open. The group's creator flows establish the new boundary
        // basis for the next node.
        for (part_index, part) in parts.iter().enumerate() {
            for flow in part.flows.iter().filter(|flow| flow.start.is_empty()) {
                self.process_flow(
                    flow,
                    part.offset,
                    part_index,
                    meas_mapper,
                    UnmatchedInputMode::Error,
                    None,
                )?;
            }
        }
        Ok(())
    }
}

type PauliPivot = (i32, i32, u8);

fn pauli_pivot(map: &PauliMap) -> Option<PauliPivot> {
    map.iter().next_back().map(|(coord, pauli)| {
        let component = if *pauli & crate::Pauli::Z { 2 } else { 1 };
        (coord.x, coord.y, component)
    })
}

// Row XOR can only introduce bits below its pivot. Retained higher bits
// never need revisiting, so walk present bits instead of scanning every basis row.
fn pauli_pivot_before(map: &PauliMap, before: PauliPivot) -> Option<PauliPivot> {
    let maximum = pauli_pivot(map)?;
    if maximum < before {
        return Some(maximum);
    }
    let (x, y, component) = before;
    let coord = IVec2::new(x, y);
    if component == 2
        && map
            .get(&coord)
            .is_some_and(|pauli| *pauli & crate::Pauli::X)
    {
        return Some((x, y, 1));
    }
    map.entry_before(coord).map(|(coord, pauli)| {
        (
            coord.x,
            coord.y,
            if pauli & crate::Pauli::Z { 2 } else { 1 },
        )
    })
}

#[cfg(test)]
fn has_pauli_pivot(map: &PauliMap, pivot: PauliPivot) -> bool {
    let (x, y, component) = pivot;
    map.get(&IVec2::new(x, y)).is_some_and(|pauli| {
        *pauli
            & if component == 2 {
                crate::Pauli::Z
            } else {
                crate::Pauli::X
            }
    })
}

/// Columns name incoming measurements; each bit names an old generator.
/// Construct only shared-coordinate pairs instead of testing every row pair.
fn pauli_commutation_columns<'a>(
    known: impl ExactSizeIterator<Item = &'a PauliMap>,
    measured: impl Iterator<Item = &'a PauliMap>,
) -> Vec<Vec<u64>> {
    let words = known.len().div_ceil(64);
    let mut sites: FxHashMap<IVec2, SmallVec<[(usize, crate::Pauli); 4]>> = FxHashMap::default();
    for (index, map) in known.enumerate() {
        for (&coord, &pauli) in map {
            sites.entry(coord).or_default().push((index, pauli));
        }
    }
    measured
        .map(|map| {
            let mut column = vec![0; words];
            for (coord, &pauli) in map {
                for &(index, other) in sites.get(coord).into_iter().flatten() {
                    if pauli.anticommutes(other) {
                        column[index / 64] ^= 1 << (index % 64);
                    }
                }
            }
            column
        })
        .collect()
}

fn swap_remove_bit(words: &mut [u64], index: usize, last: usize) {
    let value = (words[last / 64] >> (last % 64)) & 1;
    words[index / 64] = (words[index / 64] & !(1 << (index % 64))) | (value << (index % 64));
    words[last / 64] &= !(1 << (last % 64));
}

fn reduce_pauli_flow<M: Copy>(
    value: &mut PauliMap,
    flow: &mut OpenFlow<M>,
    basis: &FxHashMap<PauliPivot, (PauliMap, OpenFlow<M>)>,
) -> Result<(), FlowError> {
    let mut pivot = pauli_pivot(value);
    while let Some(current) = pivot {
        if let Some((row, row_flow)) = basis.get(&current) {
            multiply_pauli_flow(value, flow, row, row_flow)?;
        }
        pivot = pauli_pivot_before(value, current);
    }
    Ok(())
}

fn multiply_pauli_flow<M: Copy>(
    target: &mut PauliMap,
    flow: &mut OpenFlow<M>,
    source: &PauliMap,
    source_flow: &OpenFlow<M>,
) -> Result<(), FlowError> {
    let phase = target.product_phase(source);
    if !phase.is_multiple_of(2) {
        return Err(FlowError::Composition(
            "stabilizer-span products require commuting boundaries".into(),
        ));
    }
    flow.sign ^= phase == 2;
    *target = &*target ^ source;
    xor_open_flow(flow, source_flow)
}

fn first_set_bit(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .enumerate()
        .find_map(|(word, &bits)| (bits != 0).then(|| word * 64 + bits.trailing_zeros() as usize))
}

fn xor_words(target: &mut [u64], source: &[u64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target ^= source;
    }
}

#[cfg(test)]
fn xor_bits(target: &mut [bool], source: &[bool]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target ^= source;
    }
}

fn pauli_dependencies(generators: &[&PauliMap]) -> Vec<(usize, Vec<u64>)> {
    let mut basis: FxHashMap<PauliPivot, (PauliMap, Vec<u64>)> = FxHashMap::default();
    let mut dependencies = Vec::new();
    for (index, generator) in generators.iter().enumerate() {
        let mut value = (*generator).clone();
        let mut mask = vec![0; generators.len().div_ceil(64)];
        mask[index / 64] = 1 << (index % 64);
        let mut pivot = pauli_pivot(&value);
        while let Some(current) = pivot {
            if let Some((row, row_mask)) = basis.get(&current) {
                value = &value ^ row;
                xor_words(&mut mask, row_mask);
            }
            pivot = pauli_pivot_before(&value, current);
        }
        if let Some(pivot) = pauli_pivot(&value) {
            basis.insert(pivot, (value, mask));
        } else {
            dependencies.push((index, mask));
        }
    }
    dependencies
}

#[cfg(test)]
fn eager_pauli_dependencies(generators: &[&PauliMap]) -> Vec<(usize, Vec<bool>)> {
    let mut basis: Vec<(PauliPivot, PauliMap, Vec<bool>)> = Vec::new();
    let mut dependencies = Vec::new();
    for (index, generator) in generators.iter().enumerate() {
        let mut value = (*generator).clone();
        let mut mask = vec![false; generators.len()];
        mask[index] = true;
        let start = pauli_pivot(&value).map_or(basis.len(), |maximum| {
            basis.partition_point(|(pivot, _, _)| *pivot > maximum)
        });
        for (pivot, row, row_mask) in &basis[start..] {
            if has_pauli_pivot(&value, *pivot) {
                value = &value ^ row;
                xor_bits(&mut mask, row_mask);
                if value.is_empty() {
                    break;
                }
            }
        }
        if let Some(pivot) = pauli_pivot(&value) {
            let index = basis.partition_point(|(existing, _, _)| *existing > pivot);
            basis.insert(index, (pivot, value, mask));
        } else {
            dependencies.push((index, mask));
        }
    }
    dependencies
}

fn empty_open_flow<M>() -> OpenFlow<M> {
    OpenFlow {
        measurements: SmallVec::new(),
        sign: false,
        center: None,
        marker: FlowMarker::Detector,
        origin_start: None,
    }
}

fn xor_open_flow<M: Copy>(target: &mut OpenFlow<M>, source: &OpenFlow<M>) -> Result<(), FlowError> {
    target.measurements.extend_from_slice(&source.measurements);
    target.sign ^= source.sign;
    target.center = target.center.or(source.center);
    target.marker = target.marker.fuse(source.marker).ok_or_else(|| {
        FlowError::Composition(
            "stabilizer-span transition mixes discard and restart components".into(),
        )
    })?;
    if let Some(source_origin) = source.origin_start.as_deref() {
        target.origin_start = match target.origin_start.take() {
            Some(mut target_origin) => {
                let phase = target_origin.product_phase(source_origin);
                if !phase.is_multiple_of(2) {
                    return Err(FlowError::Composition(
                        "stabilizer-span products require commuting origin boundaries".into(),
                    ));
                }
                target.sign ^= phase == 2;
                *target_origin = &*target_origin ^ source_origin;
                (!target_origin.is_empty()).then_some(target_origin)
            }
            None => Some(Box::new(source_origin.clone())),
        };
    }
    Ok(())
}

fn pauli_map_cmp(a: &PauliMap, b: &PauliMap) -> std::cmp::Ordering {
    a.iter()
        .map(|(coord, pauli)| (coord.x, coord.y, *pauli as u8))
        .cmp(
            b.iter()
                .map(|(coord, pauli)| (coord.x, coord.y, *pauli as u8)),
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnmatchedInputMode {
    Error,
    Skip,
    /// Keep an unmatched-`start` flow as a boundary-entering residual chain
    /// (used by [`FlowEngine::append_residual_group`]).
    KeepAsResidual,
}

impl<M> Default for FlowEngine<'_, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M: fmt::Debug> fmt::Debug for FlowEngine<'_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlowEngine")
            .field("open_flows", &self.open_flows.len())
            .field("latent_flows", &self.latent_flows.len())
            .field("completed", &self.completed.len())
            .finish()
    }
}

/// Loop-carried detector states opened for a loop body, awaiting their `next`
/// parity once the body's closing flows are known.
///
/// Produced by [`FlowEngine::start_loop_detector_states`] and consumed by
/// [`FlowEngine::finish_loop_detector_states`].
#[derive(Debug, Clone)]
pub struct PendingLoopDetectorStates<'c, M = u32> {
    loop_states: Vec<(FlowKey<'c>, LoopCarriedDetectorState<M>)>,
}

impl<'c, M: Copy + Ord> FlowEngine<'c, DetectorTerm<M>> {
    /// Assign a loop-carried detector state to every open flow the loop body
    /// is about to consume: the open measurements become the state's initial
    /// parity and are replaced by a single loop-state term.
    ///
    /// Open flows are matched against the first body chunk's start keys
    /// directly; a valid body never consumes the same key twice (the append
    /// would fail), so each open flow gains at most one state.
    ///
    /// # Errors
    ///
    /// Returns [`FlowError`] if coordinates or the loop-state id space overflow.
    ///
    /// # Panics
    ///
    /// Panics if the preflighted loop-state range is internally inconsistent.
    pub fn start_loop_detector_states(
        &mut self,
        body_first_chunk: &[OffsetFlows<'c>],
        next_loop_state: &mut u32,
    ) -> Result<PendingLoopDetectorStates<'c, M>, FlowError> {
        for part in body_first_chunk
            .iter()
            .filter(|part| part.offset != IVec2::ZERO)
        {
            for flow in part.flows.iter().filter(|flow| !flow.start.is_empty()) {
                for (coordinate, _) in &flow.start {
                    checked_translate_coordinate(*coordinate, part.offset)?;
                }
            }
        }
        let possible_states =
            loop_state_upper_bound(body_first_chunk.iter().map(|part| part.flows.len()));
        if possible_states == usize::MAX || possible_states > (u32::MAX - *next_loop_state) as usize
        {
            self.check_loop_state_capacity(body_first_chunk, *next_loop_state)?;
        }

        let mut loop_states = Vec::with_capacity(self.open_flows.len().min(possible_states));
        for part in body_first_chunk {
            for flow in part.flows {
                if flow.start.is_empty() {
                    continue;
                }
                let key = FlowKey::new(&flow.start, part.offset);
                let Some(open) = self.open_flows.get_mut(&key) else {
                    continue;
                };
                let state = LoopStateId(*next_loop_state);
                *next_loop_state = next_loop_state
                    .checked_add(1)
                    .expect("loop state range was preflighted");
                let initial = DetectorParity::from_term_buf(std::mem::replace(
                    &mut open.measurements,
                    smallvec::smallvec![DetectorTerm::LoopState(state)],
                ))
                .with_sign(std::mem::take(&mut open.sign));
                loop_states.push((
                    key,
                    LoopCarriedDetectorState {
                        state,
                        initial,
                        next: DetectorParity::default(),
                    },
                ));
            }
        }
        Ok(PendingLoopDetectorStates { loop_states })
    }

    #[cold]
    fn check_loop_state_capacity(
        &self,
        body_first_chunk: &[OffsetFlows<'c>],
        next_loop_state: u32,
    ) -> Result<(), FlowError> {
        body_first_chunk
            .iter()
            .flat_map(|part| {
                part.flows.iter().filter(|flow| {
                    !flow.start.is_empty()
                        && self
                            .open_flows
                            .contains_key(&FlowKey::new(&flow.start, part.offset))
                })
            })
            .try_fold(next_loop_state, |next, _| next.checked_add(1))
            .ok_or_else(|| {
                FlowError::Composition("loop detector state id space exhausted".into())
            })?;
        Ok(())
    }

    /// Drain completed chains, classifying each by its fused [`FlowMarker`] into
    /// a [`ComposedChain`]: a `Discard`-marked or empty-parity chain produces
    /// nothing, a `Restart` chain becomes a restart syndrome, everything else a
    /// detector at the chain's center. This is the shared marker discipline for
    /// detector synthesis — the compile-time boundary pass
    /// (`bloq_compile::lower::detector`) and the seam recompose
    /// (`bloq_ir::edit`) both drain through here so the two cannot drift.
    pub fn drain_composed_chains(&mut self) -> impl Iterator<Item = ComposedChain<M>> + '_ {
        self.completed.drain(..).filter_map(|completed| {
            if completed.marker == FlowMarker::Discard {
                return None;
            }
            let parity =
                DetectorParity::from_term_buf(completed.measurements).with_sign(completed.sign);
            if parity.is_empty() {
                return None;
            }
            Some(if completed.marker == FlowMarker::Restart {
                ComposedChain::Restart { parity }
            } else {
                ComposedChain::Detector {
                    parity,
                    center: completed.center,
                }
            })
        })
    }

    /// Completes the pending loop states: each state's `next` parity is read
    /// from the flow still open at its key after the body was appended.
    pub fn finish_loop_detector_states(
        &self,
        pending: PendingLoopDetectorStates<'c, M>,
    ) -> Vec<LoopCarriedDetectorState<M>> {
        pending
            .loop_states
            .into_iter()
            .map(|(key, mut detector_state)| {
                if let Some(open) = self.open_flows.get(&key) {
                    detector_state.next =
                        DetectorParity::from_terms(open.measurements.iter().copied())
                            .with_sign(open.sign);
                }
                detector_state
            })
            .collect()
    }
}

fn loop_state_upper_bound(lengths: impl IntoIterator<Item = usize>) -> usize {
    lengths.into_iter().fold(0, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pauli;

    #[test]
    fn repeated_flow_slices_cannot_wrap_the_loop_state_bound() {
        assert_eq!(
            loop_state_upper_bound([usize::MAX / 2, usize::MAX / 2, 2]),
            usize::MAX,
        );
    }
    fn pauli_map(entries: impl IntoIterator<Item = (IVec2, Pauli)>) -> PauliMap {
        entries.into_iter().collect()
    }

    fn x(coord: IVec2) -> PauliMap {
        pauli_map([(coord, Pauli::X)])
    }

    fn z(coord: IVec2) -> PauliMap {
        pauli_map([(coord, Pauli::Z)])
    }

    fn flow(start: PauliMap, end: PauliMap, measurement: Option<u32>) -> Flow {
        Flow::new(start, end).with_measurements(measurement)
    }

    fn append_flows<'c>(
        engine: &mut FlowEngine<'c, u32>,
        flows: &'c [Flow],
        mut mapper: impl FnMut(&u32) -> u32,
    ) -> Result<(), FlowError> {
        engine.append_group_allowing_dangling(&[OffsetFlows::new(flows, IVec2::ZERO)], |_, m| {
            mapper(m)
        })
    }

    fn measurement_to_sequential_mapper(offset: &mut u32) -> impl FnMut(&u32) -> u32 + '_ {
        move |_reference| {
            let id = *offset;
            *offset += 1;
            id
        }
    }

    #[test]
    fn offset_flows_match_equal_absolute_boundaries() {
        let mut engine = FlowEngine::<u32>::new();
        let mut offset = 0u32;

        // Creator placed at (2, 0): its local (0, 0) boundary sits at the
        // absolute coordinate the consumer (placed at the origin) names
        // explicitly.
        let creator = [Flow::new(PauliMap::empty(), x(IVec2::new(0, 0)))
            .with_measurements([0])
            .with_center(IVec2::new(1, 1))];
        let consumer = [flow(x(IVec2::new(2, 0)), PauliMap::empty(), Some(0))];

        let mut mapper = measurement_to_sequential_mapper(&mut offset);
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creator, IVec2::new(2, 0))],
                |_, m| mapper(m),
            )
            .expect("creator");
        engine
            .append_group_allowing_dangling(&[OffsetFlows::new(&consumer, IVec2::ZERO)], |_, m| {
                mapper(m)
            })
            .expect("consumer matches translated boundary");

        let completed = engine.finish().expect("clean finish");
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].measurements.as_slice(), [1, 0]);
        assert_eq!(completed[0].center, Some(IVec2::new(3, 1)));
    }

    #[test]
    fn offset_flow_coordinate_overflow_is_typed() {
        let coordinate = IVec2::new(i32::MAX, 0);
        let offset = IVec2::new(1, 0);
        let flows = [Flow::new(PauliMap::empty(), x(coordinate))];
        let error = FlowEngine::<u32>::new()
            .append_group_allowing_dangling(&[OffsetFlows::new(&flows, offset)], |_, &m| m)
            .unwrap_err();

        assert_eq!(
            error,
            FlowError::CoordinateOverflow(crate::CoordinateOverflowError { coordinate, offset })
        );
    }

    #[test]
    fn stabilizer_transition_center_overflow_is_typed() {
        let coordinate = IVec2::new(i32::MAX, 0);
        let offset = IVec2::X;
        let incoming = [flow(z(IVec2::ZERO), PauliMap::empty(), Some(1)).with_center(coordinate)];
        let mut engine = FlowEngine::<u32>::new();

        let error = engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&incoming, offset)],
                |_, &measurement| measurement,
            )
            .unwrap_err();

        assert_eq!(
            error,
            FlowError::CoordinateOverflow(crate::CoordinateOverflowError { coordinate, offset })
        );
    }

    #[test]
    fn zero_offset_accepts_coordinate_extrema() {
        let boundary = x(IVec2::new(i32::MAX, i32::MIN));
        let flows = [Flow::new(PauliMap::empty(), boundary)];

        FlowEngine::<u32>::new()
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&flows, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("zero offset cannot overflow a coordinate");
    }

    #[test]
    fn completes_and_fuses_flows() {
        let mut engine = FlowEngine::<u32>::new();
        let mut offset = 0u32;
        let boundary = x(IVec2::new(0, 0));
        let chunk1 = [Flow::new(PauliMap::empty(), boundary.clone())
            .with_measurements([0])
            .with_sign(true)
            .with_center(IVec2::new(1, 1))];
        let chunk2 = [Flow::new(boundary, PauliMap::empty()).with_measurements([0])];

        append_flows(
            &mut engine,
            &chunk1,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("chunk1");
        append_flows(
            &mut engine,
            &chunk2,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("chunk2");

        let completed = engine.finish().expect("clean finish");
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].measurements.as_slice(), [1, 0]);
        assert!(completed[0].sign);
        assert_eq!(completed[0].center, Some(IVec2::new(1, 1)));
    }

    #[test]
    fn boundary_composition_accepts_an_equivalent_generator_basis() {
        let a = x(IVec2::new(0, 0));
        let b = z(IVec2::new(1, 0));
        let ab = &a ^ &b;
        let creators = [
            flow(PauliMap::empty(), a.clone(), Some(0)),
            flow(PauliMap::empty(), b, Some(1)),
        ];
        let consumers = [
            flow(ab, PauliMap::empty(), Some(2)),
            flow(a, PauliMap::empty(), Some(3)),
        ];
        let mut engine = FlowEngine::<u32>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creators, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("open the original basis");
        engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&consumers, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("consume the equivalent basis");

        let mut completed = engine.finish().expect("the rebased seam closes");
        completed.sort_by_key(|flow| flow.measurements[0]);
        assert_eq!(completed[0].measurements.as_slice(), [2, 0, 1]);
        assert_eq!(completed[1].measurements.as_slice(), [3, 0]);
    }

    #[test]
    fn indexed_pivots_match_eager_dependency_masks() {
        for seed in 0..64u32 {
            let generators = (0..75u32)
                .map(|index| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9)
                        .wrapping_add(index.wrapping_mul(0x85eb_ca6b));
                    pauli_map(
                        [
                            IVec2::new(i32::MIN, 0),
                            IVec2::ZERO,
                            IVec2::Y,
                            IVec2::X,
                            IVec2::new(i32::MAX, 0),
                        ]
                        .into_iter()
                        .enumerate()
                        .map(|(site, coord)| {
                            (
                                coord,
                                [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z]
                                    [((bits >> (2 * site)) & 3) as usize],
                            )
                        }),
                    )
                })
                .collect::<Vec<_>>();
            let generators: Vec<&PauliMap> = generators.iter().collect();
            let actual = pauli_dependencies(&generators)
                .into_iter()
                .map(|(index, mask)| {
                    (
                        index,
                        (0..generators.len())
                            .map(|bit| mask[bit / 64] & (1 << (bit % 64)) != 0)
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, eager_pauli_dependencies(&generators), "seed {seed}");
        }
    }

    #[test]
    fn sparse_commutation_matches_product_phase() {
        let maps: Vec<_> = (0..128)
            .map(|bits| {
                pauli_map(
                    [IVec2::new(-2, 1), IVec2::ZERO, IVec2::new(5, -1)]
                        .into_iter()
                        .enumerate()
                        .map(|(index, coord)| {
                            (
                                coord,
                                [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][(bits >> (2 * index)) & 3],
                            )
                        }),
                )
            })
            .collect();
        for offset in [IVec2::ZERO, IVec2::X, IVec2::new(9, 0)] {
            // Three Pauli sites give 64 unique columns; keep 128 rows to cover
            // swap-removal across both packed words.
            let measured = maps
                .iter()
                .take(64)
                .map(|map| map.translated(offset))
                .collect::<Vec<_>>();
            let columns = pauli_commutation_columns(maps.iter(), measured.iter());
            for (row, a) in maps.iter().enumerate() {
                for (column, b) in measured.iter().enumerate() {
                    let expected = a.product_phase(b) % 2 == 1;
                    assert_eq!(columns[column][row / 64] & (1 << (row % 64)) != 0, expected);
                }
            }
            for removed in [0, 63, 64, 127] {
                let mut remaining = maps.clone();
                remaining.swap_remove(removed);
                let mut shortened = columns.clone();
                for column in &mut shortened {
                    swap_remove_bit(column, removed, maps.len() - 1);
                }
                let expected = pauli_commutation_columns(remaining.iter(), measured.iter());
                assert_eq!(shortened, expected);
            }
        }
    }

    #[test]
    fn borrowed_flow_keys_match_translated_maps_and_reject_overflow() {
        let local = pauli_map([(IVec2::ZERO, Pauli::X), (IVec2::X, Pauli::Z)]);
        let offset = IVec2::new(-7, 13);
        let translated = local.translated(offset);
        let key = FlowKey::try_new(&local, offset).unwrap();
        let placed = FlowKey::try_new(&translated, IVec2::ZERO).unwrap();
        let mut keys = FxHashMap::default();
        keys.insert(key, 42);
        assert_eq!(keys.get(&placed), Some(&42));
        assert_ne!(key, FlowKey::try_new(&local, IVec2::ZERO).unwrap());
        FlowKey::try_new(&local, IVec2::new(i32::MAX, 0)).unwrap_err();
    }

    #[test]
    fn stabilizer_products_keep_their_sign_through_basis_and_gauge_changes() {
        let a = IVec2::ZERO;
        let b = IVec2::X;
        let pair = |pauli| pauli_map([(a, pauli), (b, pauli)]);
        let creators = [
            flow(PauliMap::empty(), pair(Pauli::X), Some(0)),
            flow(PauliMap::empty(), pair(Pauli::Z), Some(1)),
        ];
        let measured = pauli_map([(a, Pauli::X), (b, Pauli::Z)]);
        let gauge = [
            flow(measured.clone(), PauliMap::empty(), Some(2)),
            flow(PauliMap::empty(), measured, Some(2)),
        ];
        let consumers = [flow(pair(Pauli::Y), PauliMap::empty(), Some(3))];
        for change_gauge in [false, true] {
            let mut engine = FlowEngine::<u32>::new();
            engine
                .append_group_allowing_dangling(
                    &[OffsetFlows::new(&creators, IVec2::ZERO)],
                    |_, &m| m,
                )
                .unwrap();
            if change_gauge {
                engine
                    .append_group_allowing_stabilizer_transition(
                        &[OffsetFlows::new(&gauge, IVec2::ZERO)],
                        |_, &m| m,
                    )
                    .unwrap();
            }
            engine
                .append_group_allowing_stabilizer_transition(
                    &[OffsetFlows::new(&consumers, IVec2::ZERO)],
                    |_, &m| m,
                )
                .unwrap();
            let completed = engine.finish().unwrap();
            assert_eq!(completed.len(), 1);
            let parity =
                DetectorParity::from_measurements(completed[0].measurements.iter().copied())
                    .with_sign(completed[0].sign);
            assert_eq!(
                parity,
                DetectorParity::from_measurements([0, 1, 3]).with_sign(true),
                "XX * ZZ = -YY, including when their product survives as a latent check"
            );
        }
    }

    #[test]
    fn a_group_reads_only_boundaries_from_earlier_groups() {
        let (a, b) = (x(IVec2::ZERO), x(IVec2::X));
        let swap = [
            flow(a.clone(), b.clone(), None),
            flow(b.clone(), a.clone(), None),
        ];
        let mut residual = FlowEngine::<u32>::new();
        residual
            .append_residual_group(&[OffsetFlows::new(&swap, IVec2::ZERO)], |_, &m| m)
            .unwrap();
        let residuals = residual.into_residual_flows();
        assert_eq!(residuals.len(), 2);
        assert!(residuals.iter().any(|f| f.start == a && f.end == b));
        assert!(residuals.iter().any(|f| f.start == b && f.end == a));

        let creators = [
            flow(PauliMap::empty(), a.clone(), Some(0)),
            flow(PauliMap::empty(), b.clone(), Some(1)),
        ];
        let consumers = [
            flow(a, PauliMap::empty(), Some(2)),
            flow(b, PauliMap::empty(), Some(3)),
        ];
        let mut engine = FlowEngine::<u32>::new();
        for group in [&creators, &swap, &consumers] {
            engine
                .append_group_allowing_dangling(&[OffsetFlows::new(group, IVec2::ZERO)], |_, &m| m)
                .unwrap();
        }
        let completed = engine.finish().unwrap();
        assert_eq!(completed[0].measurements.as_slice(), [2, 1]);
        assert_eq!(completed[1].measurements.as_slice(), [3, 0]);
    }

    #[test]
    fn boundary_composition_matches_an_exact_generator_basis() {
        let a = x(IVec2::new(0, 0));
        let b = z(IVec2::new(1, 0));
        let creators = [
            flow(PauliMap::empty(), a.clone(), Some(0)),
            flow(PauliMap::empty(), b.clone(), Some(1)),
        ];
        let consumers = [
            flow(a, PauliMap::empty(), Some(2)),
            flow(b, PauliMap::empty(), Some(3)),
        ];
        let mut engine = FlowEngine::<u32>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creators, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("open the basis");
        engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&consumers, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("consume the same basis");

        let completed = engine.finish().expect("the seam closes");
        assert_eq!(completed[0].measurements.as_slice(), [2, 0]);
        assert_eq!(completed[1].measurements.as_slice(), [3, 1]);
    }

    #[test]
    fn gauge_transition_carries_only_commuting_old_products() {
        let z0 = z(IVec2::new(0, 0));
        let z1 = z(IVec2::new(1, 0));
        let joint_x = &x(IVec2::new(0, 0)) ^ &x(IVec2::new(1, 0));
        let old_creators = [
            flow(PauliMap::empty(), z0.clone(), Some(0)),
            flow(PauliMap::empty(), z1.clone(), Some(1)),
        ];
        let merge = [
            flow(joint_x.clone(), PauliMap::empty(), Some(2)),
            flow(PauliMap::empty(), joint_x, Some(3)),
        ];
        let split_left = [flow(z0, PauliMap::empty(), Some(4))];
        let split_right = [flow(z1, PauliMap::empty(), Some(5))];

        let mut engine = FlowEngine::<u32>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&old_creators, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("open the old basis");
        engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&merge, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("measure the temporary gauge");
        engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&split_left, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("restore the left half of the old basis");
        engine
            .append_group_allowing_stabilizer_transition(
                &[OffsetFlows::new(&split_right, IVec2::ZERO)],
                |_, &measurement| measurement,
            )
            .expect("restore the right half of the old basis");

        let completed = engine.finish().expect("the restored basis closes");
        assert_eq!(completed.len(), 1);
        let mut measurements = completed[0].measurements.to_vec();
        measurements.sort_unstable();
        assert_eq!(measurements, [0, 1, 4, 5]);
    }

    #[test]
    fn flow_marker_survives_fusion() {
        for marker in [FlowMarker::Discard, FlowMarker::Restart] {
            let mut engine = FlowEngine::<u32>::new();
            let mut offset = 0u32;
            let boundary = x(IVec2::new(0, 0));
            let creator = [Flow::new(PauliMap::empty(), boundary.clone())
                .with_measurements([0])
                .with_marker(marker)];
            let consumer = [Flow::new(boundary, PauliMap::empty()).with_measurements([0])];

            append_flows(
                &mut engine,
                &creator,
                measurement_to_sequential_mapper(&mut offset),
            )
            .expect("creator");
            append_flows(
                &mut engine,
                &consumer,
                measurement_to_sequential_mapper(&mut offset),
            )
            .expect("consumer");

            let completed = engine.finish().expect("clean finish");
            assert_eq!(completed.len(), 1);
            assert_eq!(completed[0].marker, marker);
        }
    }

    /// A restarting chain left open at the boundary keeps its marker on the
    /// residual, so it lands on the template's `boundary_flows` and the
    /// instance-seam pass can route the cross-template parity to a restart.
    #[test]
    fn restart_marker_survives_residual_extraction() {
        let mut engine = FlowEngine::<u32>::new();
        let creator = [Flow::new(PauliMap::empty(), z(IVec2::new(0, 0)))
            .with_measurements([0])
            .with_marker(FlowMarker::Restart)];
        engine
            .append_residual_group(&[OffsetFlows::new(&creator, IVec2::ZERO)], |_, &m| m)
            .expect("creator");

        let residual = engine.into_residual_flows();
        assert_eq!(residual.len(), 1);
        assert_eq!(residual[0].marker, FlowMarker::Restart);
    }

    #[test]
    fn consumer_before_creator_ordering() {
        let mut engine = FlowEngine::<u32>::new();
        let mut offset = 0u32;

        let shared_key = x(IVec2::new(0, 0));
        let predecessor = [flow(PauliMap::empty(), shared_key.clone(), Some(0))];
        append_flows(
            &mut engine,
            &predecessor,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("predecessor");

        let time_reversed_chunk = [
            flow(PauliMap::empty(), shared_key.clone(), Some(0)),
            flow(shared_key.clone(), PauliMap::empty(), Some(0)),
        ];

        append_flows(
            &mut engine,
            &time_reversed_chunk,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("time-reversed chunk should succeed when consumers are processed first");

        let keys: Vec<_> = engine.open_flows.keys().map(FlowKey::translated).collect();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], shared_key);
        assert_eq!(engine.drain_completed().len(), 1);
    }

    #[test]
    fn flow_errors_cover_unmatched_and_unterminated() {
        let mut offset = 0u32;

        let mut unmatched = FlowEngine::<u32>::new();
        let orphan = [flow(x(IVec2::new(0, 0)), PauliMap::empty(), Some(0))];
        let err = append_flows(
            &mut unmatched,
            &orphan,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect_err("unmatched input");
        assert_eq!(
            err,
            FlowError::Composition(format!("unmatched flow input {}", orphan[0].start))
        );

        let mut unterminated = FlowEngine::<u32>::new();
        let open_ended = [flow(PauliMap::empty(), z(IVec2::new(2, 0)), Some(0))];
        append_flows(
            &mut unterminated,
            &open_ended,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("seed");
        let err = unterminated.finish().expect_err("unterminated flow");
        assert_eq!(
            err,
            FlowError::Composition(format!("unterminated flows: {}", open_ended[0].end))
        );
    }

    #[test]
    fn duplicate_creators_are_rejected_instead_of_overwriting_parity() {
        let boundary = z(IVec2::new(2, 0));
        let creators = [
            flow(PauliMap::empty(), boundary.clone(), Some(0)),
            flow(PauliMap::empty(), boundary.clone(), Some(1)),
        ];
        let mut engine = FlowEngine::<u32>::new();

        let error = append_flows(&mut engine, &creators, |measurement| *measurement)
            .expect_err("two creators of one boundary must collide");

        assert_eq!(
            error,
            FlowError::Composition(format!("multiple flows create boundary {boundary}"))
        );
    }

    #[test]
    fn loop_state_id_exhaustion_is_an_error() {
        let boundaries = [x(IVec2::ZERO), x(IVec2::X)];
        let creator = [
            flow(PauliMap::empty(), boundaries[0].clone(), Some(7)),
            flow(PauliMap::empty(), boundaries[1].clone(), Some(8)),
        ];
        let consumer = [
            flow(boundaries[0].clone(), PauliMap::empty(), None),
            flow(boundaries[1].clone(), PauliMap::empty(), None),
        ];
        let mut engine = FlowEngine::<DetectorTerm<u32>>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creator, IVec2::ZERO)],
                |_, &measurement| DetectorTerm::Measurement(measurement),
            )
            .expect("creator");
        let mut next_loop_state = u32::MAX - 1;

        let error = engine
            .start_loop_detector_states(
                &[OffsetFlows::new(&consumer, IVec2::ZERO)],
                &mut next_loop_state,
            )
            .expect_err("loop state id space is exhausted");

        assert_eq!(
            error,
            FlowError::Composition("loop detector state id space exhausted".into())
        );
        assert_eq!(next_loop_state, u32::MAX - 1);
        assert!(engine.open_flows.values().all(|flow| {
            matches!(
                flow.measurements.as_slice(),
                [DetectorTerm::Measurement(7 | 8)]
            )
        }));
    }

    #[test]
    fn loop_state_start_rejects_coordinate_overflow_before_mutating() {
        let coordinate = IVec2::new(i32::MAX, 0);
        let offset = IVec2::X;
        let creator = [flow(PauliMap::empty(), x(IVec2::new(i32::MIN, 0)), Some(7))];
        let consumer = [flow(x(coordinate), PauliMap::empty(), None)];
        let mut engine = FlowEngine::<DetectorTerm<u32>>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creator, IVec2::ZERO)],
                |_, &measurement| DetectorTerm::Measurement(measurement),
            )
            .expect("creator");
        let mut next_loop_state = 0;

        let error = engine
            .start_loop_detector_states(
                &[OffsetFlows::new(&consumer, offset)],
                &mut next_loop_state,
            )
            .expect_err("overflowing loop boundary");

        assert_eq!(
            error,
            FlowError::CoordinateOverflow(crate::CoordinateOverflowError { coordinate, offset })
        );
        assert_eq!(next_loop_state, 0);
        assert!(matches!(
            engine
                .open_flows
                .values()
                .next()
                .unwrap()
                .measurements
                .as_slice(),
            [DetectorTerm::Measurement(7)]
        ));
    }

    #[test]
    fn loop_state_preflight_counts_only_matching_flows() {
        let boundary = x(IVec2::ZERO);
        let creator = [flow(PauliMap::empty(), boundary.clone(), Some(7))];
        let consumers = [
            flow(x(IVec2::X), PauliMap::empty(), None),
            flow(boundary, PauliMap::empty(), None),
        ];
        let mut engine = FlowEngine::<DetectorTerm<u32>>::new();
        engine
            .append_group_allowing_dangling(
                &[OffsetFlows::new(&creator, IVec2::ZERO)],
                |_, &measurement| DetectorTerm::Measurement(measurement),
            )
            .expect("creator");
        let mut next_loop_state = u32::MAX - 1;

        let pending = engine
            .start_loop_detector_states(
                &[OffsetFlows::new(&consumers, IVec2::ZERO)],
                &mut next_loop_state,
            )
            .expect("only the matching flow needs a state");

        assert_eq!(pending.loop_states.len(), 1);
        assert_eq!(next_loop_state, u32::MAX);
    }

    #[test]
    fn drain_completed_allows_open_boundaries() {
        let mut engine = FlowEngine::<u32>::new();
        let mut offset = 0u32;

        let chunk = [
            flow(PauliMap::empty(), PauliMap::empty(), Some(0)),
            flow(PauliMap::empty(), x(IVec2::new(0, 0)), Some(0)),
        ];

        append_flows(
            &mut engine,
            &chunk,
            measurement_to_sequential_mapper(&mut offset),
        )
        .expect("chunk");

        assert_eq!(engine.drain_completed().len(), 1);
    }

    #[test]
    fn residual_extraction_fuses_a_chain_across_chunks() {
        // ∅→K (chunk 0) then K→L (chunk 1): residual mode threads the chain as
        // it composes, so the residual is one fused ∅→L carrying both
        // measurements.
        let mut engine = FlowEngine::<u32>::new();
        let creator = [flow(PauliMap::empty(), z(IVec2::new(0, 0)), Some(0))];
        let mover = [flow(z(IVec2::new(0, 0)), z(IVec2::new(1, 0)), Some(1))];
        engine
            .append_residual_group(&[OffsetFlows::new(&creator, IVec2::ZERO)], |_, &m| m)
            .expect("creator");
        engine
            .append_residual_group(&[OffsetFlows::new(&mover, IVec2::ZERO)], |_, &m| m)
            .expect("mover");

        let residual = engine.into_residual_flows();
        assert_eq!(residual.len(), 1, "the chain fuses to a single flow");
        assert!(residual[0].start.is_empty());
        assert_eq!(residual[0].end, z(IVec2::new(1, 0)));
        let mut measurements = residual[0].measurements.to_vec();
        measurements.sort_unstable();
        assert_eq!(measurements, [0, 1]);
    }

    #[test]
    fn residual_extraction_keeps_opposite_boundary_faces_separate() {
        // One chunk with a bottom consumer K→∅ and a top creator ∅→K. They share
        // the key K but face opposite neighbors; consumers are matched before
        // creators, so the consumer never matches this chunk's own creator and
        // the two stay distinct residual flows.
        let mut engine = FlowEngine::<u32>::new();
        let faces = [
            flow(z(IVec2::new(0, 0)), PauliMap::empty(), Some(0)),
            flow(PauliMap::empty(), z(IVec2::new(0, 0)), Some(1)),
        ];
        engine
            .append_residual_group(&[OffsetFlows::new(&faces, IVec2::ZERO)], |_, &m| m)
            .expect("faces");

        let mut residual = engine.into_residual_flows();
        residual.sort_by_key(|flow| flow.measurements.first().copied());
        assert_eq!(residual.len(), 2, "opposite faces are not fused");
        // Bottom consumer K→∅ closed against a boundary input below.
        assert_eq!(residual[0].start, z(IVec2::new(0, 0)));
        assert!(residual[0].end.is_empty());
        // Top creator ∅→K stays open for the neighbor above.
        assert!(residual[1].start.is_empty());
        assert_eq!(residual[1].end, z(IVec2::new(0, 0)));
    }
}
