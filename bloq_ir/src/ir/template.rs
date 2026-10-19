use std::ops::Index;
use std::sync::Arc;

use bloq_circuit::{BodyId, CoordCircuit, DetectorCoords, Flow, LoopStateId};
use glam::{IVec2, IVec3};

use super::{NodeDetectorParity, TemplateDetectorParity, TemplateId, TemplateInstanceId};
use crate::instantiation::{
    NodeTemplateInstanceMergeError, TemplateCircuitAnalysis, preflight_template_circuit,
};

/// A reusable circuit template and its local side tables.
///
/// Together they form the connection interface a persisted `.bloq` is edited against.
/// Pooled and shared by [`TemplateId`]; a [`TemplateInstance`] places one at an
/// `offset`.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BloqTemplate {
    /// The template's circuit, in template-local qubit coordinates.
    pub circuit: CoordCircuit,
    /// Detectors closing entirely within this template.
    pub detectors: Vec<TemplateDetector>,
    /// Loop-carried detector recurrences for the circuit's repeat bodies.
    pub repeat_states: Vec<TemplateRepeatState>,
    /// The template's residual open flows — the only detectors instance lowering
    /// must compose, since they close against a temporally adjacent node. The
    /// retained connection interface that makes a persisted `.bloq` editable
    /// (load → edit → recompose).
    ///
    /// A flat list of **fused** boundary chains: each is one maximal open chain
    /// spliced into a single `start -> end` flow carrying all its measurements,
    /// in template-measurement-id space. Cross-chunk chains are threaded once at
    /// build time, so entries are mutually independent (no entry's `end` is
    /// another's `start`). A loop's recurrence lives in `repeat_states`, not
    /// here. Computed by the lowering flow pass and deduped per signature.
    pub boundary_flows: Vec<Flow>,
    /// Restart syndromes for `RepeatUntilSuccess` regions instantiating this
    /// template. The executor evaluates them after each sandboxed attempt.
    pub restarts: Vec<TemplateRestart>,
    /// Lazily computed cache of the circuit's qubits (hot in footprint
    /// queries). Safe because templates are immutable once pooled; a template
    /// whose `circuit` is still being mutated must not call [`Self::qubits`].
    #[serde(skip)]
    qubit_cache: std::sync::OnceLock<Vec<IVec2>>,
    /// Lazily cached structural circuit analysis. Safe under the same immutable
    /// pooled-template contract as [`Self::qubits`].
    #[serde(skip)]
    circuit_analysis_cache:
        std::sync::OnceLock<Result<TemplateCircuitAnalysis, NodeTemplateInstanceMergeError>>,
    /// Successful validation of this immutable template's side tables.
    #[serde(skip)]
    validation_cache: std::sync::OnceLock<()>,
}

impl Clone for BloqTemplate {
    fn clone(&self) -> Self {
        Self {
            circuit: self.circuit.clone(),
            detectors: self.detectors.clone(),
            repeat_states: self.repeat_states.clone(),
            boundary_flows: self.boundary_flows.clone(),
            restarts: self.restarts.clone(),
            qubit_cache: std::sync::OnceLock::new(),
            circuit_analysis_cache: std::sync::OnceLock::new(),
            validation_cache: std::sync::OnceLock::new(),
        }
    }
}

impl PartialEq for BloqTemplate {
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
            && self.detectors == other.detectors
            && self.repeat_states == other.repeat_states
            && self.boundary_flows == other.boundary_flows
            && self.restarts == other.restarts
    }
}

impl BloqTemplate {
    /// Create a template containing only `circuit`.
    pub fn new(circuit: CoordCircuit) -> Self {
        Self::with_parts(circuit, Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    /// Create a template with all side tables supplied explicitly.
    pub fn with_parts(
        circuit: CoordCircuit,
        detectors: Vec<TemplateDetector>,
        repeat_states: Vec<TemplateRepeatState>,
        boundary_flows: Vec<Flow>,
        restarts: Vec<TemplateRestart>,
    ) -> Self {
        Self {
            circuit,
            detectors,
            repeat_states,
            boundary_flows,
            restarts,
            qubit_cache: std::sync::OnceLock::new(),
            circuit_analysis_cache: std::sync::OnceLock::new(),
            validation_cache: std::sync::OnceLock::new(),
        }
    }

    /// The circuit's qubits, computed once per template (the circuit walk is
    /// expensive and footprint queries re-visit templates heavily).
    pub fn qubits(&self) -> &[IVec2] {
        self.qubit_cache
            .get_or_init(|| self.circuit.qubits().into_iter().collect())
    }

    pub(crate) fn circuit_analysis(
        &self,
    ) -> Result<&TemplateCircuitAnalysis, NodeTemplateInstanceMergeError> {
        self.circuit_analysis_cache
            .get_or_init(|| preflight_template_circuit(&self.circuit))
            .as_ref()
            .map_err(Clone::clone)
    }

    pub(crate) fn validation_cached(&self) -> bool {
        self.validation_cache.get().is_some()
    }

    pub(crate) fn cache_validation(&self) {
        self.validation_cache.get_or_init(|| ());
    }
}

/// A shared, append-only template pool.
///
/// Templates are keyed by [`TemplateId`] (the insertion index). Ids are never
/// reused or renumbered, so anything that
/// stores a `TemplateId` outside the pool — [`crate::PipePadding`], the
/// decoder-wait specialization memo — can rely on it staying valid for the
/// program's life.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BloqTemplatePool {
    #[serde(deserialize_with = "super::id::deserialize_pool")]
    templates: Vec<Arc<BloqTemplate>>,
}

impl BloqTemplatePool {
    /// Create an empty template pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an owned template and return its append-only id.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next id exceeds `u32`.
    pub fn insert(&mut self, mut template: BloqTemplate) -> TemplateId {
        template.qubit_cache = std::sync::OnceLock::new();
        template.circuit_analysis_cache = std::sync::OnceLock::new();
        template.validation_cache = std::sync::OnceLock::new();
        self.insert_shared(Arc::new(template))
    }

    /// Unlike [`Self::insert`], warmed qubit, circuit-analysis, and validation caches are
    /// kept: an `Arc`-shared template is already immutable (the pool hands out
    /// no `&mut`), so they cannot be stale. Clearing them here would deep-clone
    /// every template the caller shares with a live pool.
    ///
    /// # Panics
    ///
    /// Panics before insertion if the next id exceeds `u32`.
    pub fn insert_shared(&mut self, template: Arc<BloqTemplate>) -> TemplateId {
        let id = TemplateId(super::id::pool_index(self.templates.len()));
        self.templates.push(template);
        id
    }

    /// Look up a template by id.
    pub fn get(&self, id: TemplateId) -> Option<&BloqTemplate> {
        self.templates.get(id.0 as usize).map(Arc::as_ref)
    }

    /// Mutable access to a pooled template, for the few edits that retune a
    /// template in place rather than pooling a new one.
    ///
    /// Copy-on-write: a template shared with another pool (via
    /// [`Self::insert_shared`]) is cloned first, so the edit cannot reach
    /// across programs. Its lazy caches describe the circuit that is about to
    /// change, so they are dropped — the pool's usual immutability contract,
    /// which is what lets those caches exist, is suspended for exactly this
    /// call.
    pub(crate) fn make_mut(&mut self, id: TemplateId) -> Option<&mut BloqTemplate> {
        let template = Arc::make_mut(self.templates.get_mut(id.0 as usize)?);
        template.qubit_cache = std::sync::OnceLock::new();
        template.circuit_analysis_cache = std::sync::OnceLock::new();
        template.validation_cache = std::sync::OnceLock::new();
        Some(template)
    }

    /// Iterate over templates in id order.
    pub fn iter(&self) -> impl Iterator<Item = (TemplateId, &BloqTemplate)> {
        self.iter_shared()
            .map(|(id, template)| (id, template.as_ref()))
    }

    /// Borrow immutable template handles for importing into another pool
    /// without copying their circuits or discarding cached analyses.
    pub fn iter_shared(&self) -> impl Iterator<Item = (TemplateId, &Arc<BloqTemplate>)> {
        self.templates
            .iter()
            .enumerate()
            .map(|(index, template)| (TemplateId(super::id::pool_index(index)), template))
    }

    /// Number of templates.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Whether the pool contains no templates.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

impl Index<TemplateId> for BloqTemplatePool {
    type Output = BloqTemplate;

    fn index(&self, index: TemplateId) -> &Self::Output {
        self.templates[index.0 as usize].as_ref()
    }
}

/// One detector inside a template: a measurement `parity` at optional decoder
/// `coords`, tagged by the body `scope` it lives in.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TemplateDetector {
    /// Circuit body containing the detector.
    pub scope: TemplateDetectorScope,
    /// Template-local detector parity.
    pub parity: TemplateDetectorParity,
    /// Optional decoder coordinates.
    pub coords: Option<DetectorCoords>,
}

/// Where a [`TemplateDetector`] lives: the template's top level, or inside one
/// of its repeat bodies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TemplateDetectorScope {
    /// Detector lives in the entry body.
    TopLevel,
    /// Detector lives in a repeat body.
    RepeatBody {
        /// Repeat body id.
        body: BodyId,
    },
}

/// A loop-carried detector recurrence for one repeat body: the `initial`
/// parity on the first iteration and the `next` parity relating consecutive
/// iterations of loop `state`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TemplateRepeatState {
    /// Repeat body carrying the state.
    pub body: BodyId,
    /// Loop-state id.
    pub state: LoopStateId,
    /// First-iteration parity.
    pub initial: TemplateDetectorParity,
    /// Recurrence parity for later iterations.
    pub next: TemplateDetectorParity,
}

/// A template-local restart syndrome for a `RepeatUntilSuccess` region.
///
/// Reuses [`TemplateDetectorParity`] — a restart syndrome
/// is just another template-local parity — so instance lowering translates it
/// like any other side-table entry. A named side table (like `detectors` /
/// `repeat_states`) rather than a bare `Vec<TemplateDetectorParity>` so a
/// bounded-retry scope/bound field can land here without a signature churn.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TemplateRestart {
    /// Template-local measurement parity that, when odd (failure syndrome),
    /// restarts the enclosing RUS attempt.
    pub parity: TemplateDetectorParity,
}

/// An instance-space detector on a [`crate::QuantumNode`].
///
/// This is the [`NodeDetectorParity`] analogue of [`TemplateDetector`], for chains that close across the node's
/// instances rather than within a single template.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct NodeDetector {
    /// Instance-space detector parity.
    pub parity: NodeDetectorParity,
    /// Optional decoder coordinates.
    pub coords: Option<DetectorCoords>,
}

/// A `RepeatUntilSuccess` restart syndrome composed across the template seam:
///
/// the instance-measurement-space analogue of [`TemplateRestart`],
/// exactly as [`NodeDetector`] is of [`TemplateDetector`]. Produced when a
/// post-selected parity's chain crosses two template instances (e.g. an escape
/// template's first merge round comparing against the cultivation template's
/// last syndrome round), so neither template's local `restarts` table can carry
/// it. Only meaningful on a node inside a RUS body; validation rejects it
/// elsewhere. No coords — restart parities are never decoder food.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NodeRestart {
    /// Instance-measurement parity that, when odd (failure syndrome), restarts
    /// the enclosing RUS attempt.
    pub parity: NodeDetectorParity,
}

/// Memory-padding provenance for one temporal pipe.
///
/// This identifies the precompiled templates [`crate::Bloq::insert_memory_rounds`] instantiates on
/// the pipe's seam, and the layout offset placing them over its cross-section.
/// Edge scope matters for a temporal Hadamard, whose parent and child seams use
/// different adjacent cube faces. A region's outgoing edge may instead carry
/// the terminal-face templates used to append padding inside that region.
///
/// Which pipe this describes is the [`crate::PipeSeam`] that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PipePadding {
    /// Layout offset positioning the templates over the pipe's cross-section.
    pub offset: IVec2,
    /// Template implementing exactly one memory round (no repeat body).
    pub one_round: TemplateId,
    /// Template implementing `1 + r` rounds as a first round plus a
    /// `REPEAT r` body. Side tables and boundary flows are repetition-count
    /// independent, so an edit specializes it to any `rounds >= 2` by cloning
    /// and rewriting `r` to `rounds - 1`.
    pub looped: TemplateId,
}

/// Why a template instance exists.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum InstanceProvenance {
    /// Direct source instance.
    #[default]
    Source,
    /// Instance emitted for a source block.
    Block {
        /// Source block position.
        source: IVec3,
    },
    /// Instance emitted for a source pipe.
    Pipe {
        /// Pipe source.
        src: IVec3,
        /// Pipe destination.
        dst: IVec3,
    },
    /// Instance emitted while replacing a spatial port.
    SpatialPortSubstitution {
        /// Source port position.
        source: IVec3,
        /// Source port role.
        role: bloq_utils::PortRole,
        /// Realized substitution half.
        part: SpatialPortPart,
    },
    /// Ideal simulation Bell preparation or Bell measurement at a
    /// temporal Port whose declared role reverses circuit time.
    IdealPortAdapter {
        /// Authored Port position.
        source: IVec3,
        /// Declared boundary role.
        role: bloq_utils::PortRole,
    },
}

impl InstanceProvenance {
    /// Whether the compiler implements this instance as an ideal boundary.
    #[must_use]
    pub const fn is_ideal_boundary(self) -> bool {
        matches!(
            self,
            Self::SpatialPortSubstitution { .. } | Self::IdealPortAdapter { .. }
        )
    }
    /// Whether this instance belongs to a spatial-port substitution.
    #[must_use]
    pub const fn is_spatial_port_substitution(self) -> bool {
        matches!(self, Self::SpatialPortSubstitution { .. })
    }
}

/// Which half of a spatial-Port substitution an instance realizes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum SpatialPortPart {
    /// Replacement cube.
    Cube,
    /// Replacement temporal port.
    TemporalPort,
}

/// One placement of a template: template `template_id` from the shared pool,
/// laid down at layout `offset`, under program-global id `id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TemplateInstance {
    /// Program-global instance id.
    pub id: TemplateInstanceId,
    /// Placed template.
    pub template_id: TemplateId,
    /// Layout offset.
    pub offset: IVec2,
    /// Source provenance.
    #[serde(default)]
    pub provenance: InstanceProvenance,
}

impl TemplateInstance {
    /// Create a source-provenance placement.
    #[must_use]
    pub const fn new(id: TemplateInstanceId, template_id: TemplateId, offset: IVec2) -> Self {
        Self {
            id,
            template_id,
            offset,
            provenance: InstanceProvenance::Source,
        }
    }
}

#[cfg(test)]
mod tests {

    use bloq_circuit::CoordCircuit;
    use glam::ivec2;

    use super::{BloqTemplate, BloqTemplatePool};

    #[test]
    fn owned_insertion_clears_warmed_template_caches() {
        use bloq_circuit::{Op, PauliBasis};

        let mut template = BloqTemplate::new(CoordCircuit::new());
        template.circuit.register_measurement_id(0, ivec2(0, 0));
        assert!(template.qubits().contains(&ivec2(0, 0)));
        template.circuit_analysis().unwrap();
        template.cache_validation();
        template.circuit.register_measurement_id(1, ivec2(1, 0));
        template
            .circuit
            .body_mut(template.circuit.entry_body())
            .unwrap()
            .ops_mut()
            .push(Op::Measure {
                basis: PauliBasis::Z,
                qubits: vec![ivec2(2, 0)],
                measurements: vec![2],
                flip_probability: 0.0,
            });

        let mut pool = BloqTemplatePool::new();
        let id = pool.insert(template);

        assert!(pool[id].qubits().contains(&ivec2(1, 0)));
        assert!(matches!(
            pool[id].circuit_analysis(),
            Err(crate::NodeTemplateInstanceMergeError::UnregisteredMeasurementOutput(2))
        ));
        assert!(!pool[id].validation_cached());
    }
}
