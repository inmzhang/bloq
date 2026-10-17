//! Bloq IR bindings: the full structural surface of `bloq_ir::Bloq` plus the
//! memory-round edge/terminal mutation.
//!
//! Structure mirrors the Rust IR faithfully (per the approved binding map and
//! the user's full-structural override):
//!
//! - Data-carrying Rust enums become pyo3 *complex enums* — each variant is a
//!   Python class (`BloqEdge.Quantum`, `NodeProvenance.MemoryPadding`, ...)
//!   with per-field getters, so Python can `isinstance`-dispatch on them.
//! - Structs become read-only pyclasses. All reads clone owned data out of the
//!   wrapped [`bloq_ir::Bloq`]; nothing borrows across the boundary.
//! - `BloqNodeId` / `TemplateId` / `TemplateInstanceId` / `BodyId` /
//!   `LoopStateId` are plain Python `int`s at the boundary.
//!
//! Mutation surface (user-required): [`PyBloq::insert_memory_rounds`] — the
//! user-facing splice instantiating the program's recorded edge-owned padding
//! provenance (`bloq_ir::Bloq::insert_memory_rounds`) — the lower-level
//! [`PyBloq::subdivide_quantum_edge`] for callers that already hold padding
//! template ids, and [`PyBloq::flatten`] (`bloq_ir::Bloq::flatten`), which
//! unrolls every `REPEAT` block and expands loop-carried detector state into
//! per-iteration detectors. Hand-building programs from Python
//! (`add_node`/`add_edge`/`add_template`) stays excluded: only the compiler
//! should construct Bloqs.
//!
//! Whole-program traversal ([`bloq_ir::Bloq::walk`] / [`bloq_ir::Bloq::levels`])
//! is exposed as the eager, Pythonic [`PyBloq::walk`] (a flat list of
//! [`PyWalkNode`]) and [`PyBloq::levels`]; the Rust callback flow-control
//! (`WalkControl`, `NodeCx`, borrowed `LevelPath`) stays Rust-only — Python
//! callers filter or break over the returned lists themselves.
//!
//! Deliberately excluded (compiler-internal or lifetimed): `WalkControl` /
//! `NodeCx`, `PathScratch`, `optimize`, template construction and
//! `NodeEmissionPlan`, the raw `BloqTemplate` circuit payloads (`CoordCircuit`
//! is below the `Bloq` abstraction).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use bloq_circuit::DetectorTerm;
use bloq_ir::{
    BLOQ_BINARY_EXTENSION, BLOQ_BINARY_VERSION, BLOQ_TEXT_EXTENSION, BLOQ_TEXT_VERSION, BloqEdge,
    BloqNode, BloqNodeId, BloqNodeKind, BloqStats, BodySelector, BoundaryFace, ClassicalAssignment,
    ClassicalExpr, ClassicalNode, ClassicalResolution, InstanceProvenance, LevelPath,
    LogicalOutput, MemoryRoundTarget, MetadataValue, NodeCx, NodeDetector, NodeDetectorParity,
    NodeKey, NodeProvenance, ObservableOutput, RegionKind, RegionNode, RegionRef, SpatialPortPart,
    SubGraph, TemplateId, TemporalPipeRef, ValueInput, ValueRef, ValueRole, WalkControl,
    lowering::{
        InstanceBoundaryOperator, InstanceMeasurement, InstantiationOptions, NodeRestart,
        TemplateInstance,
    },
};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3_stub_gen::derive::{
    gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pyclass_enum, gen_stub_pymethods,
};

use crate::errors;
use crate::primitives::{PyBasis, PyPauli, py_bool};

// ==============================================================================
// Leaf pieces: pipes, frames, measurements
// ==============================================================================

/// A temporal pipe between two source blocks, named by its endpoint lattice
/// positions. Appears on `BloqEdge.Quantum` edges as the seam's connecting
/// pipes, and in `NodeProvenance.TemporalPipe` / `MemoryPadding` stamps.
/// `src` and `dst` are the endpoint block positions `(x, y, z)`; `hadamard`
/// marks a basis-swapping (H) pipe.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemporalPipe",
    module = "bloq._core",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemporalPipe {
    /// Source block position `(x, y, z)`.
    src: (i32, i32, i32),
    /// Destination block position `(x, y, z)`.
    dst: (i32, i32, i32),
    /// `True` if the seam applies a Hadamard.
    hadamard: bool,
}

impl From<&TemporalPipeRef> for PyTemporalPipe {
    fn from(pipe: &TemporalPipeRef) -> Self {
        Self {
            src: pipe.src.into(),
            dst: pipe.dst.into(),
            hadamard: pipe.hadamard,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemporalPipe {
    fn __repr__(&self) -> String {
        format!(
            "TemporalPipe(src={:?}, dst={:?}, hadamard={})",
            self.src,
            self.dst,
            py_bool(self.hadamard)
        )
    }
}

/// One temporal pipe on a `BloqEdge.Quantum` seam, with the memory-padding
/// templates the compiler recorded for it. `padding` is `None` on a
/// hand-built or synthetic edge, and on a pipe whose patch the compiler could
/// not resolve; `Bloq.insert_memory_rounds` needs every pipe on a seam padded.
#[gen_stub_pyclass]
#[pyclass(
    name = "PipeSeam",
    module = "bloq._core",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyPipeSeam {
    /// The seam's temporal pipe.
    pipe: PyTemporalPipe,
    /// The pipe's memory-padding provenance, when the compiler recorded it.
    padding: Option<crate::circuit::PyPipePadding>,
}

impl From<&bloq_ir::PipeSeam> for PyPipeSeam {
    fn from(seam: &bloq_ir::PipeSeam) -> Self {
        Self {
            pipe: (&seam.pipe).into(),
            padding: seam.padding.as_ref().map(Into::into),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipeSeam {
    fn __repr__(&self) -> String {
        format!(
            "PipeSeam(pipe={}, padded={})",
            self.pipe.__repr__(),
            py_bool(self.padding.is_some())
        )
    }
}

/// One measurement site inside a template instance (`i<instance>:m<index>`).
///
/// Ordered by `(instance, measurement)`, matching the IR's own order, so a
/// collection gathered from several detectors has a stable total order.
/// Instance ids are allocation order, not emission order: an edit that splices
/// padding into the middle of a program gives it the highest instance id, so
/// sorting does not recover the emission sequence.
#[gen_stub_pyclass]
#[pyclass(
    name = "InstanceMeasurement",
    module = "bloq._core",
    frozen,
    eq,
    ord,
    hash,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PyInstanceMeasurement {
    /// The template instance id.
    instance: u32,
    /// The measurement index within that instance's template circuit.
    measurement: u32,
}

impl From<InstanceMeasurement> for PyInstanceMeasurement {
    fn from(m: InstanceMeasurement) -> Self {
        Self {
            instance: m.instance.0,
            measurement: m.measurement,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInstanceMeasurement {
    fn __str__(&self) -> String {
        format!("i{}:m{}", self.instance, self.measurement)
    }

    fn __repr__(&self) -> String {
        format!(
            "InstanceMeasurement(instance={}, measurement={})",
            self.instance, self.measurement
        )
    }
}

/// One terminal output's logical Pauli-frame sign bits: the `Compute` nodes
/// carrying the X- and Z-frame corrections for the source output port.
#[gen_stub_pyclass]
#[pyclass(
    name = "FramePair",
    module = "bloq._core",
    eq,
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyFramePair {
    /// The source output port this frame corrects.
    port: (i32, i32, i32),
    /// The node id whose bit is the X-frame sign.
    x: u32,
    /// The node id whose bit is the Z-frame sign.
    z: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFramePair {
    fn __repr__(&self) -> String {
        format!(
            "FramePair(port={:?}, x={}, z={})",
            self.port, self.x, self.z
        )
    }
}

/// Complete terminal logical operators for one output, in layout-global
/// coordinates.
#[gen_stub_pyclass]
#[pyclass(
    name = "LogicalOutput",
    module = "bloq._core",
    eq,
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyLogicalOutput {
    port: (i32, i32, i32),
    /// Template instance that owns this output's logical worldline.
    instance: u32,
    x: Vec<((i32, i32), PyPauli)>,
    z: Vec<((i32, i32), PyPauli)>,
}

impl From<&LogicalOutput> for PyLogicalOutput {
    fn from(output: &LogicalOutput) -> Self {
        let terms = |map: &bloq_circuit::PauliMap| {
            map.iter()
                .map(|(coord, pauli)| ((*coord).into(), PyPauli::from(*pauli)))
                .collect()
        };
        Self {
            port: output.port.into(),
            instance: output.instance.0,
            x: terms(&output.x),
            z: terms(&output.z),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLogicalOutput {
    fn __repr__(&self) -> String {
        format!(
            "<LogicalOutput port={:?} x={} z={}>",
            self.port,
            self.x.len(),
            self.z.len()
        )
    }
}

// ==============================================================================
// Detector side tables
// ==============================================================================

/// An XOR of measurement (and loop-state) terms — a detector or restart
/// parity in instance-measurement space.
#[gen_stub_pyclass]
#[pyclass(
    name = "NodeParity",
    module = "bloq._core",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyNodeParity(NodeDetectorParity);

/// One XOR term of a `NodeParity`: a template-instance measurement or a
/// loop-carried detector state.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "DetectorTerm", module = "bloq._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyDetectorTerm {
    Measurement { measurement: PyInstanceMeasurement },
    LoopState { state: u32 },
}

impl From<&DetectorTerm<InstanceMeasurement>> for PyDetectorTerm {
    fn from(term: &DetectorTerm<InstanceMeasurement>) -> Self {
        match *term {
            DetectorTerm::Measurement(m) => PyDetectorTerm::Measurement {
                measurement: m.into(),
            },
            DetectorTerm::LoopState(state) => PyDetectorTerm::LoopState { state: state.0 },
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDetectorTerm {
    fn __repr__(&self) -> String {
        match self {
            PyDetectorTerm::Measurement { measurement } => format!(
                "DetectorTerm.Measurement(measurement={})",
                measurement.__str__()
            ),
            PyDetectorTerm::LoopState { state } => {
                format!("DetectorTerm.LoopState(state={state})")
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNodeParity {
    /// Whether this parity includes the constant XOR term.
    #[getter]
    fn sign(&self) -> bool {
        self.0.sign()
    }

    /// All XOR terms in order: measurements and loop states.
    fn terms(&self) -> Vec<PyDetectorTerm> {
        self.0.terms().iter().map(Into::into).collect()
    }

    /// The measurement terms, dropping any loop-state terms.
    fn measurements(&self) -> Vec<PyInstanceMeasurement> {
        self.0.measurements().map(Into::into).collect()
    }

    /// The loop-state ids among the terms (a loop-carried detector
    /// recurrence), in term order.
    fn loop_states(&self) -> Vec<u32> {
        self.0
            .terms()
            .iter()
            .filter_map(|term| match term {
                DetectorTerm::LoopState(state) => Some(state.0),
                DetectorTerm::Measurement(_) => None,
            })
            .collect()
    }

    /// `True` if the parity has no terms.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!("NodeParity(\"{}\")", self.0)
    }
}

/// One top-level detector on a quantum node: a parity and optional decoder
/// coordinates.
#[gen_stub_pyclass]
#[pyclass(
    name = "NodeDetector",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyNodeDetector {
    /// The detector's measurement parity.
    parity: PyNodeParity,
    /// Decoder coordinates, or `None` when the detector carries none.
    coords: Option<Vec<f64>>,
}

impl From<&NodeDetector> for PyNodeDetector {
    fn from(detector: &NodeDetector) -> Self {
        Self {
            parity: PyNodeParity(detector.parity.clone()),
            coords: detector.coords.as_ref().map(|coords| coords.to_vec()),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNodeDetector {
    fn __repr__(&self) -> String {
        format!(
            "<NodeDetector terms={} coords={}>",
            self.parity.0.terms().len(),
            self.coords.as_ref().map_or(0, Vec::len)
        )
    }
}

/// One shared detector row in bundle-local owner coordinates.
#[gen_stub_pyclass_complex_enum]
#[pyclass(
    name = "BundleDetectorTerm",
    module = "bloq._core",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) enum PyBundleDetectorTerm {
    Measurement { owner: u32, measurement: u32 },
    LoopState { state: u32 },
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBundleDetectorTerm {
    fn __repr__(&self) -> String {
        match self {
            Self::Measurement { owner, measurement } => {
                format!("BundleDetectorTerm.Measurement(owner={owner}, measurement={measurement})")
            }
            Self::LoopState { state } => format!("BundleDetectorTerm.LoopState(state={state})"),
        }
    }
}

/// One shared detector row in bundle-local owner coordinates.
#[gen_stub_pyclass]
#[pyclass(
    name = "BundleDetector",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyBundleDetector {
    /// Ordered measurement and loop-state terms.
    terms: Vec<PyBundleDetectorTerm>,
    /// Expected parity sign.
    sign: bool,
    /// Decoder coordinates before placement offset.
    coords: Option<Vec<f64>>,
}

impl From<&bloq_ir::BundleDetector> for PyBundleDetector {
    fn from(detector: &bloq_ir::BundleDetector) -> Self {
        Self {
            terms: detector
                .parity
                .terms()
                .iter()
                .map(|term| match term {
                    DetectorTerm::Measurement(term) => PyBundleDetectorTerm::Measurement {
                        owner: term.owner,
                        measurement: term.measurement,
                    },
                    DetectorTerm::LoopState(state) => {
                        PyBundleDetectorTerm::LoopState { state: state.0 }
                    }
                })
                .collect(),
            sign: detector.parity.sign(),
            coords: detector.coords.as_ref().map(|coords| coords.to_vec()),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBundleDetector {
    fn __repr__(&self) -> String {
        format!(
            "<BundleDetector terms={} sign={} coords={}>",
            self.terms.len(),
            py_bool(self.sign),
            self.coords.as_ref().map_or(0, Vec::len)
        )
    }
}

/// Detector rows shared by placed uses of a bundle.
#[gen_stub_pyclass]
#[pyclass(
    name = "DetectorBundle",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDetectorBundle {
    /// Expected template id for each owner slot.
    owner_templates: Vec<u32>,
    /// Rows in bundle order.
    detectors: Vec<PyBundleDetector>,
}

impl From<&bloq_ir::DetectorBundle> for PyDetectorBundle {
    fn from(bundle: &bloq_ir::DetectorBundle) -> Self {
        Self {
            owner_templates: bundle.owner_templates().iter().map(|id| id.0).collect(),
            detectors: bundle.detectors().iter().map(Into::into).collect(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDetectorBundle {
    fn __repr__(&self) -> String {
        format!(
            "<DetectorBundle owners={} detectors={}>",
            self.owner_templates.len(),
            self.detectors.len()
        )
    }
}

/// Bind a shared detector bundle to instances at a layout offset.
#[gen_stub_pyclass]
#[pyclass(
    name = "DetectorBundleUse",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyDetectorBundleUse {
    /// Shared bundle pool id.
    bundle: u32,
    /// Instance id bound to each owner slot.
    instances: Vec<u32>,
    /// Added to each detector center when placed.
    offset: (i32, i32),
}

impl From<&bloq_ir::DetectorBundleUse> for PyDetectorBundleUse {
    fn from(bundle_use: &bloq_ir::DetectorBundleUse) -> Self {
        Self {
            bundle: bundle_use.bundle.0,
            instances: bundle_use.instances.iter().map(|id| id.0).collect(),
            offset: (bundle_use.offset.x, bundle_use.offset.y),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDetectorBundleUse {
    fn __repr__(&self) -> String {
        format!(
            "DetectorBundleUse(bundle={}, instances={:?}, offset={:?})",
            self.bundle, self.instances, self.offset
        )
    }
}

/// A `RepeatUntilSuccess` restart syndrome composed across a template seam,
/// listed in `QuantumNode.restarts`. Produced when a post-selected parity's
/// chain crosses two template instances, so neither template's local table
/// can carry it. When `parity` is odd (the failure syndrome), the enclosing
/// repeat-until-success attempt restarts. Only meaningful on a node inside
/// a `RepeatUntilSuccess` body.
#[gen_stub_pyclass]
#[pyclass(
    name = "NodeRestart",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyNodeRestart {
    /// The parity that, when odd (failure syndrome), restarts the RUS attempt.
    parity: PyNodeParity,
}

impl From<&NodeRestart> for PyNodeRestart {
    fn from(restart: &NodeRestart) -> Self {
        Self {
            parity: PyNodeParity(restart.parity.clone()),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNodeRestart {
    fn __repr__(&self) -> String {
        format!("<NodeRestart terms={}>", self.parity.0.terms().len())
    }
}

#[gen_stub_pyclass_complex_enum]
#[pyclass(
    name = "InstanceProvenance",
    module = "bloq._core",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) enum PyInstanceProvenance {
    Source(),
    Block {
        source: (i32, i32, i32),
    },
    Pipe {
        src: (i32, i32, i32),
        dst: (i32, i32, i32),
    },
    SpatialPortSubstitution {
        source: (i32, i32, i32),
        role: String,
        part: String,
    },
}

impl From<InstanceProvenance> for PyInstanceProvenance {
    fn from(provenance: InstanceProvenance) -> Self {
        match provenance {
            InstanceProvenance::Source => Self::Source(),
            InstanceProvenance::Block { source } => Self::Block {
                source: source.into(),
            },
            InstanceProvenance::Pipe { src, dst } => Self::Pipe {
                src: src.into(),
                dst: dst.into(),
            },
            InstanceProvenance::SpatialPortSubstitution { source, role, part } => {
                Self::SpatialPortSubstitution {
                    source: source.into(),
                    role: role.as_str().to_owned(),
                    part: match part {
                        SpatialPortPart::Cube => "cube",
                        SpatialPortPart::TemporalPort => "temporal_port",
                    }
                    .to_owned(),
                }
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInstanceProvenance {
    fn __repr__(&self) -> String {
        match self {
            Self::Source() => "InstanceProvenance.Source()".to_owned(),
            Self::Block { source } => format!("InstanceProvenance.Block(source={source:?})"),
            Self::Pipe { src, dst } => format!("InstanceProvenance.Pipe(src={src:?}, dst={dst:?})"),
            Self::SpatialPortSubstitution { source, role, part } => format!(
                "InstanceProvenance.SpatialPortSubstitution(source={source:?}, role={role:?}, part={part:?})"
            ),
        }
    }
}

/// One placement of a pooled template, listed in `QuantumNode.instances`:
/// template `template_id` (look it up with `Bloq.template()`) laid down at
/// layout offset `offset`, under instance id `id`. Instance ids are
/// program-unique — unlike node ids, they are not level-local — and
/// instance-space references such as `InstanceMeasurement` key on them.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemplateInstance",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplateInstance {
    /// The program-unique instance id.
    id: u32,
    /// The id of the backing template in the shared pool.
    template_id: u32,
    /// Layout offset `(x, y)`.
    offset: (i32, i32),
    /// Compiler provenance for source and spatial-Port-derived instances.
    provenance: PyInstanceProvenance,
}

impl From<&TemplateInstance> for PyTemplateInstance {
    fn from(instance: &TemplateInstance) -> Self {
        Self {
            id: instance.id.0,
            template_id: instance.template_id.0,
            offset: instance.offset.into(),
            provenance: instance.provenance.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateInstance {
    fn __repr__(&self) -> String {
        format!(
            "<TemplateInstance id={} template={} offset={:?}>",
            self.id, self.template_id, self.offset
        )
    }
}

/// A template-backed quantum node's payload, reached via
/// `BloqNodeKind.Quantum` or `Bloq.quantum_nodes()`. Its operations come
/// from instantiating the templates its `instances` place from the
/// program's shared pool (`Bloq.template()`), not from any per-node
/// circuit. `detectors` and `restarts` are the node's instance-space side
/// tables for chains that close across its instances.
#[gen_stub_pyclass]
#[pyclass(
    name = "QuantumNode",
    module = "bloq._core",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyQuantumNode {
    /// The template instances this node places.
    instances: Vec<PyTemplateInstance>,
    /// Inline node detectors; shared rows are in `detector_bundles`.
    detectors: Vec<PyNodeDetector>,
    /// Shared detector bundle uses, in row order after inline detectors.
    detector_bundles: Vec<PyDetectorBundleUse>,
    /// The node's repeat-until-success restart syndromes.
    restarts: Vec<PyNodeRestart>,
    /// Cumulative circuit-round ends from the minimum source z, if present.
    timeline: Option<Vec<u32>>,
    /// Conditional registrations. Entries absent from these guards are common.
    guards: Vec<PyQuantumGuard>,
}

impl From<&bloq_ir::QuantumNode> for PyQuantumNode {
    fn from(quantum: &bloq_ir::QuantumNode) -> Self {
        Self {
            instances: quantum.instances.iter().map(Into::into).collect(),
            detectors: quantum.detectors.iter().map(Into::into).collect(),
            detector_bundles: quantum.detector_bundles.iter().map(Into::into).collect(),
            restarts: quantum.restarts.iter().map(Into::into).collect(),
            timeline: quantum
                .timeline
                .as_ref()
                .map(|t| t.layer_round_ends.clone()),
            guards: quantum
                .guards
                .iter()
                .map(|guard| PyQuantumGuard {
                    input: guard.input,
                    instances: guard.instances.iter().map(|instance| instance.0).collect(),
                    detectors: guard.detectors.clone(),
                    detector_bundles: guard.detector_bundles.clone(),
                    restarts: guard.restarts.clone(),
                    detector_parities: guard
                        .detector_parities
                        .iter()
                        .map(|(index, parity)| (*index, PyNodeParity(parity.clone())))
                        .collect(),
                    restart_parities: guard
                        .restart_parities
                        .iter()
                        .map(|(index, parity)| (*index, PyNodeParity(parity.clone())))
                        .collect(),
                })
                .collect(),
        }
    }
}

/// Members and side-table entries registered when one Boolean input is true.
#[gen_stub_pyclass]
#[pyclass(
    name = "QuantumGuard",
    module = "bloq._core",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyQuantumGuard {
    /// The owning quantum node's selector Value input slot.
    input: u32,
    /// Program-global instance ids selected by this guard.
    instances: Vec<u32>,
    /// Indexes in the owning node's detector table.
    detectors: Vec<u32>,
    /// Indices in the owning node's shared detector bundle use table.
    detector_bundles: Vec<u32>,
    /// Indexes in the owning node's restart table.
    restarts: Vec<u32>,
    /// Conditional XOR contributions to detector rows.
    detector_parities: Vec<(u32, PyNodeParity)>,
    /// Conditional XOR contributions to restart rows.
    restart_parities: Vec<(u32, PyNodeParity)>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyQuantumGuard {
    fn __repr__(&self) -> String {
        format!(
            "<QuantumGuard input={} instances={} detectors={} restarts={}>",
            self.input,
            self.instances.len(),
            self.detectors.len(),
            self.restarts.len()
        )
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyQuantumNode {
    fn __repr__(&self) -> String {
        format!(
            "<QuantumNode instances={} detectors={} detector_bundles={} restarts={}>",
            self.instances.len(),
            self.detectors.len(),
            self.detector_bundles.len(),
            self.restarts.len()
        )
    }
}

// ==============================================================================
// Classical dataflow
// ==============================================================================

/// Combinational logic over a classical node's incoming `Value` edges: an
/// expression tree walked via `kind` + `operands()`. `In(slot)` reads the
/// producer wired into that slot. Python truth testing raises `TypeError`;
/// use `Bloq.classical_value` to evaluate a node with explicit inputs.
#[gen_stub_pyclass]
#[pyclass(
    name = "ClassicalExpr",
    module = "bloq._core",
    frozen,
    eq,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyClassicalExpr(ClassicalExpr);

/// Compact single-line render for `str()`/`repr()`: `xor(in(0), not(in(1)))`.
fn render_expr(expr: &ClassicalExpr, out: &mut String) {
    use std::fmt::Write;
    match expr {
        ClassicalExpr::In(slot) => write!(out, "in({slot})"),
        ClassicalExpr::Const(value) => write!(out, "const({value})"),
        ClassicalExpr::Parity { inputs, constant } => {
            write!(out, "parity({constant}").expect("write to String");
            for slot in inputs {
                write!(out, ", in({slot})").expect("write to String");
            }
            write!(out, ")")
        }
        ClassicalExpr::Not(inner) => {
            out.push_str("not(");
            render_expr(inner, out);
            write!(out, ")")
        }
        ClassicalExpr::Xor(_)
        | ClassicalExpr::And(_)
        | ClassicalExpr::Or(_)
        | ClassicalExpr::Select(_) => {
            out.push_str(match expr {
                ClassicalExpr::Xor(..) => "xor(",
                ClassicalExpr::And(..) => "and(",
                ClassicalExpr::Select(..) => "select(",
                _ => "or(",
            });
            for (index, operand) in expr.operands().iter().enumerate() {
                if index != 0 {
                    out.push_str(", ");
                }
                render_expr(operand, out);
            }
            write!(out, ")")
        }
    }
    .expect("writing to a String cannot fail")
}

#[gen_stub_pymethods]
#[pymethods]
impl PyClassicalExpr {
    /// The variant name: `"in"`, `"const"`, `"not"`, `"xor"`, `"and"`, `"or"`, `"select"`, `"parity"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.0 {
            ClassicalExpr::In(_) => "in",
            ClassicalExpr::Const(_) => "const",
            ClassicalExpr::Not(_) => "not",
            ClassicalExpr::Xor(..) => "xor",
            ClassicalExpr::And(..) => "and",
            ClassicalExpr::Or(..) => "or",
            ClassicalExpr::Select(..) => "select",
            ClassicalExpr::Parity { .. } => "parity",
        }
    }

    /// The input slot for an `in` expression, else `None`.
    #[getter]
    fn slot(&self) -> Option<u32> {
        match &self.0 {
            ClassicalExpr::In(slot) => Some(*slot),
            _ => None,
        }
    }

    /// The constant bit for a `const` or `parity` expression, else `None`.
    #[getter]
    fn value(&self) -> Option<bool> {
        match &self.0 {
            ClassicalExpr::Const(value) => Some(*value),
            ClassicalExpr::Parity { constant, .. } => Some(*constant),
            _ => None,
        }
    }

    /// Ordered input slots for a compact `parity`, else `None`.
    #[getter]
    fn input_slots(&self) -> Option<Vec<u32>> {
        match &self.0 {
            ClassicalExpr::Parity { inputs, .. } => Some(inputs.to_vec()),
            _ => None,
        }
    }

    /// Ordered children: one for `not`, any number for `xor`/`and`/`or`,
    /// and `(condition, when_false, when_true)` for `select`. Leaves and compact
    /// `parity` have none; read the latter's `input_slots` and `value` instead.
    fn operands(&self) -> Vec<PyClassicalExpr> {
        self.0
            .operands()
            .iter()
            .cloned()
            .map(PyClassicalExpr)
            .collect()
    }

    /// Whether the expression is XOR-expressible (`in`/`const`/`not`/`xor`/`parity`
    /// only) — a static backend can fold a linear condition.
    fn is_linear(&self) -> bool {
        self.0.is_linear()
    }

    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "symbolic ClassicalExpr has no truth value; use Bloq.classical_value to evaluate a node",
        ))
    }

    fn __str__(&self) -> String {
        let mut out = String::new();
        render_expr(&self.0, &mut out);
        out
    }

    fn __repr__(&self) -> String {
        format!("ClassicalExpr({})", self.__str__())
    }
}

/// Which temporal face of a template instance a boundary operator binds —
/// see `InstanceBoundaryOperator.face`. The face fixes emission position: an
/// `Input` operator reads the qubit state entering the instance, so it emits
/// before the instance circuit; an `Output` operator reads on exit, so it emits
/// after.
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "BoundaryFace",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyBoundaryFace {
    Input,
    Output,
}

/// A symbolic logical boundary operator at one template instance's face —
/// one binding payload element of `ClassicalNode.Observable.operators`. Names
/// the operator by instance reference, never per-shot bits: `instance` is
/// the `TemplateInstance.id`, `face` the temporal face bound, and `operator`
/// maps instance-global `(x, y)` qubit coordinates to `Pauli` terms.
#[gen_stub_pyclass]
#[pyclass(
    name = "InstanceBoundaryOperator",
    module = "bloq._core",
    frozen,
    from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyInstanceBoundaryOperator {
    instance: u32,
    face: PyBoundaryFace,
    operator: Vec<((i32, i32), PyPauli)>,
}

impl From<&InstanceBoundaryOperator> for PyInstanceBoundaryOperator {
    fn from(op: &InstanceBoundaryOperator) -> Self {
        Self {
            instance: op.instance.0,
            face: match op.face {
                BoundaryFace::Input => PyBoundaryFace::Input,
                BoundaryFace::Output => PyBoundaryFace::Output,
            },
            operator: op
                .operator
                .iter()
                .map(|(coord, pauli)| ((*coord).into(), PyPauli::from(*pauli)))
                .collect(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInstanceBoundaryOperator {
    /// The template instance id this operator binds to.
    #[getter]
    fn instance(&self) -> u32 {
        self.instance
    }

    /// Which temporal face (`Input` or `Output`) the operator binds.
    #[getter]
    fn face(&self) -> PyBoundaryFace {
        self.face
    }

    /// The operator as a dict mapping instance-global `(x, y)` qubit
    /// coordinates to `Pauli` terms.
    #[getter]
    fn operator(&self) -> std::collections::HashMap<(i32, i32), PyPauli> {
        self.operator.iter().copied().collect()
    }

    fn __repr__(&self) -> String {
        let face = match self.face {
            PyBoundaryFace::Input => "Input",
            PyBoundaryFace::Output => "Output",
        };
        format!(
            "<InstanceBoundaryOperator instance={} face={} terms={}>",
            self.instance,
            face,
            self.operator.len()
        )
    }
}

/// A classical-dataflow payload. Observable fragments compose through `Compose`
/// edges; indexed observables expose decoder-backed `Corrected` and `Flip` ports.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "ClassicalNode", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyClassicalNode {
    /// Combinational logic over incoming value slots.
    Compute { expr: PyClassicalExpr },
    /// A parity and boundary recipe. `None` identifies an unindexed fragment;
    /// an index identifies a complete logical readout and its decoder query.
    Observable {
        index: Option<u32>,
        measurements: Vec<PyInstanceMeasurement>,
        operators: Vec<PyInstanceBoundaryOperator>,
    },
    /// Rejects the shot when the condition holds.
    Discard { condition: PyClassicalExpr },
}

impl From<&ClassicalNode> for PyClassicalNode {
    fn from(node: &ClassicalNode) -> Self {
        match node {
            ClassicalNode::Compute { expr } => Self::Compute {
                expr: PyClassicalExpr(expr.clone()),
            },
            ClassicalNode::Observable {
                index,
                measurements,
                operators,
            } => Self::Observable {
                index: *index,
                measurements: measurements.iter().copied().map(Into::into).collect(),
                operators: operators.iter().map(Into::into).collect(),
            },
            ClassicalNode::Discard { condition } => Self::Discard {
                condition: PyClassicalExpr(condition.clone()),
            },
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyClassicalNode {
    fn __repr__(&self) -> String {
        match self {
            Self::Compute { expr } => format!("ClassicalNode.Compute(expr={})", expr.__str__()),
            Self::Observable {
                index,
                measurements,
                operators,
            } => format!(
                "ClassicalNode.Observable(index={}, measurements={}, operators={})",
                index.map_or_else(|| "None".to_owned(), |index| index.to_string()),
                measurements.len(),
                operators.len(),
            ),
            Self::Discard { condition } => {
                format!("ClassicalNode.Discard(condition={})", condition.__str__())
            }
        }
    }
}

/// Selects an observable's corrected parity or its decoder flip estimate.
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "ObservableOutput",
    module = "bloq._core",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PyObservableOutput {
    Corrected,
    Flip,
}

impl From<ObservableOutput> for PyObservableOutput {
    fn from(output: ObservableOutput) -> Self {
        match output {
            ObservableOutput::Corrected => Self::Corrected,
            ObservableOutput::Flip => Self::Flip,
        }
    }
}

impl From<PyObservableOutput> for ObservableOutput {
    fn from(output: PyObservableOutput) -> Self {
        match output {
            PyObservableOutput::Corrected => Self::Corrected,
            PyObservableOutput::Flip => Self::Flip,
        }
    }
}

/// A level-local Boolean source, including its selected output port.
#[gen_stub_pyclass]
#[pyclass(
    name = "ValueRef",
    module = "bloq._core",
    frozen,
    get_all,
    eq,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PyValueRef {
    node: u32,
    output: PyObservableOutput,
}

impl From<ValueRef> for PyValueRef {
    fn from(value: ValueRef) -> Self {
        Self {
            node: value.node.0,
            output: value.output.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyValueRef {
    #[new]
    #[pyo3(signature = (node, output=PyObservableOutput::Corrected))]
    fn new(node: u32, output: PyObservableOutput) -> Self {
        Self { node, output }
    }

    fn __repr__(&self) -> String {
        format!("ValueRef(node={}, output={:?})", self.node, self.output)
    }
}

// ==============================================================================
// Edges
// ==============================================================================

/// Edge provenance on a `Value` edge (`BloqEdge.Value.role`,
/// `ValueInput.role`): what source construct routed this bit here. Value
/// semantics ignore it — every role reads the producer's bit the same way.
/// `Data` is plain dataflow; `FeedbackFold` marks a `Feedback` action's
/// condition bit folding into an observable. `ReadoutFold` folds an earlier
/// corrected readout; dynamic decoder symptoms exclude both runtime folds.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "ValueRole", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PyValueRole {
    /// Plain dataflow.
    Data(),
    /// Folds a `Feedback` action's condition bit into an observable. `action`
    /// is the source `Feedback` ordinal.
    FeedbackFold { action: u32 },
    /// Folds an already corrected readout into a composed named parity.
    ReadoutFold(),
}

impl From<&ValueRole> for PyValueRole {
    fn from(role: &ValueRole) -> Self {
        match role {
            ValueRole::Data => PyValueRole::Data(),
            ValueRole::FeedbackFold { action } => PyValueRole::FeedbackFold { action: *action },
            ValueRole::ReadoutFold => PyValueRole::ReadoutFold(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyValueRole {
    fn __repr__(&self) -> String {
        match self {
            PyValueRole::Data() => "ValueRole.Data()".to_owned(),
            PyValueRole::ReadoutFold() => "ValueRole.ReadoutFold()".to_owned(),
            PyValueRole::FeedbackFold { action } => {
                format!("ValueRole.FeedbackFold(action={action})")
            }
        }
    }
}

/// A graph edge payload: a quantum face seam, a classical-bit dataflow
/// dependency, or a payload-free ordering constraint.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "BloqEdge", module = "bloq._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyBloqEdge {
    /// A quantum face seam: its temporal pipes, each with its edge-owned
    /// decoder-wait padding provenance.
    Quantum {
        pipes: Vec<PyPipeSeam>,
        guard: Option<PyValueRef>,
    },
    /// A classical-bit dependency feeding the consumer's `slot`-th input.
    Value {
        slot: u32,
        role: PyValueRole,
        output: PyObservableOutput,
    },
    /// Structural recipe composition, including raw parity and boundary bindings.
    Compose { slot: u32, role: PyValueRole },
    /// Pure sequencing, no data.
    Order(),
}

impl From<&BloqEdge> for PyBloqEdge {
    fn from(edge: &BloqEdge) -> Self {
        match edge {
            BloqEdge::Quantum(edge) => PyBloqEdge::Quantum {
                pipes: edge.pipes.iter().map(Into::into).collect(),
                guard: edge.guard.map(Into::into),
            },
            BloqEdge::Value { slot, role, output } => PyBloqEdge::Value {
                slot: *slot,
                role: role.into(),
                output: (*output).into(),
            },
            BloqEdge::Compose { slot, role } => PyBloqEdge::Compose {
                slot: *slot,
                role: role.into(),
            },
            BloqEdge::Order => PyBloqEdge::Order(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBloqEdge {
    fn __repr__(&self) -> String {
        match self {
            PyBloqEdge::Quantum { pipes, guard } => {
                format!("BloqEdge.Quantum(pipes={}, guard={guard:?})", pipes.len())
            }
            PyBloqEdge::Value { slot, role, output } => {
                format!(
                    "BloqEdge.Value(slot={slot}, output={output:?}, role={})",
                    role.__repr__()
                )
            }
            PyBloqEdge::Compose { slot, role } => {
                format!("BloqEdge.Compose(slot={slot}, role={})", role.__repr__())
            }
            PyBloqEdge::Order() => "BloqEdge.Order()".to_owned(),
        }
    }
}

/// One graph edge with its endpoint node ids — the element type of
/// `Bloq.edges()`, `incoming()`, `outgoing()`, and `edges_between()` (and
/// their `SubGraph` twins). `source` and `target` are level-local node ids;
/// `edge` is the `BloqEdge` payload (dispatch on its variant with
/// `isinstance`).
#[gen_stub_pyclass]
#[pyclass(
    name = "BloqEdgeRef",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyBloqEdgeRef {
    /// The source node id.
    source: u32,
    /// The target node id.
    target: u32,
    /// The edge payload.
    edge: PyBloqEdge,
}

impl From<bloq_ir::BloqEdgeRef<'_>> for PyBloqEdgeRef {
    fn from(edge: bloq_ir::BloqEdgeRef<'_>) -> Self {
        Self {
            source: edge.source.0,
            target: edge.target.0,
            edge: edge.edge.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBloqEdgeRef {
    /// `True` if this is a quantum face seam.
    ///
    /// Filtering an edge list by payload variant is the common read; the
    /// predicates spell it without an `isinstance` on `edge`.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    ///     >>> [e.source for e in program.incoming(3) if e.is_quantum()]
    ///     [2]
    fn is_quantum(&self) -> bool {
        matches!(self.edge, PyBloqEdge::Quantum { .. })
    }

    /// `True` if this is a classical-bit dataflow dependency.
    fn is_value(&self) -> bool {
        matches!(self.edge, PyBloqEdge::Value { .. })
    }

    /// `True` if this structurally composes an observable recipe.
    fn is_compose(&self) -> bool {
        matches!(self.edge, PyBloqEdge::Compose { .. })
    }

    /// `True` if this is a payload-free ordering constraint.
    fn is_order(&self) -> bool {
        matches!(self.edge, PyBloqEdge::Order())
    }

    fn __repr__(&self) -> String {
        format!(
            "BloqEdgeRef(source={}, target={}, edge={})",
            self.source,
            self.target,
            self.edge.__repr__()
        )
    }
}

/// A Boolean or composition input, returned by `value_inputs()` or `data_inputs()`.
/// The producer feeds the consumer's slot with the selected output and role;
/// `output=None` identifies structural recipe composition.
#[gen_stub_pyclass]
#[pyclass(
    name = "ValueInput",
    module = "bloq._core",
    eq,
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyValueInput {
    /// The consumer input slot this producer feeds.
    slot: u32,
    /// The producer node id.
    producer: u32,
    /// The input edge's role.
    role: PyValueRole,
    /// Selected Boolean output; `None` denotes structural composition.
    output: Option<PyObservableOutput>,
}

impl From<ValueInput<'_>> for PyValueInput {
    fn from(input: ValueInput<'_>) -> Self {
        Self {
            slot: input.slot,
            producer: input.producer.0,
            role: input.role.into(),
            output: input.output.map(Into::into),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyValueInput {
    fn __repr__(&self) -> String {
        format!(
            "ValueInput(slot={}, producer={}, output={:?}, role={})",
            self.slot,
            self.producer,
            self.output,
            self.role.__repr__()
        )
    }
}

// ==============================================================================
// Provenance
// ==============================================================================

/// Node provenance: what source-level construct an IR node realizes, read
/// from `BloqNode.provenance`. Quantum nodes tie back to source blocks and
/// pipes, classical nodes to stabilizer generators, source actions, or
/// output frames. Provenance is never consulted for emission semantics —
/// its consumers are `Bloq.output_frames()`, program views, and decoder
/// runtimes. Dispatch on the variant with `isinstance`.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "NodeProvenance", module = "bloq._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyNodeProvenance {
    /// A fused component of source blocks.
    BlockComponent { members: Vec<(i32, i32, i32)> },
    /// A temporal pipe between two blocks.
    TemporalPipe { pipe: PyTemporalPipe },
    /// Positionless temporal Port derived from an authored spatial Port.
    SpatialPortSubstitution {
        source: (i32, i32, i32),
        role: String,
    },
    /// Memory-round padding spliced into a seam post-compile, waiting
    /// `rounds` syndrome-extraction rounds.
    MemoryPadding { pipe: PyTemporalPipe, rounds: u32 },
    /// A classical node realizing stabilizer generator `ordinal`'s readout.
    Generator { ordinal: u32 },
    /// A classical node lowered from the source action at `ordinal`.
    Action { ordinal: u32 },
    /// A structural branch or selective measurement's scoped selector name.
    BranchSelector { name: String },
    /// The `basis`-frame correction bit for source output `port`.
    OutputFrame {
        port: (i32, i32, i32),
        basis: PyBasis,
    },
    /// No provenance: hand-built or synthetic.
    Missing(),
}

impl From<&NodeProvenance> for PyNodeProvenance {
    fn from(provenance: &NodeProvenance) -> Self {
        match provenance {
            NodeProvenance::BlockComponent { members } => PyNodeProvenance::BlockComponent {
                members: members.iter().map(|m| m.pos.into()).collect(),
            },
            NodeProvenance::TemporalPipe { pipe } => {
                PyNodeProvenance::TemporalPipe { pipe: pipe.into() }
            }
            NodeProvenance::SpatialPortSubstitution { source, role } => {
                PyNodeProvenance::SpatialPortSubstitution {
                    source: (*source).into(),
                    role: role.as_str().to_owned(),
                }
            }
            NodeProvenance::MemoryPadding { pipe, rounds } => PyNodeProvenance::MemoryPadding {
                pipe: pipe.into(),
                rounds: *rounds,
            },
            NodeProvenance::Generator { ordinal } => {
                PyNodeProvenance::Generator { ordinal: *ordinal }
            }
            NodeProvenance::Action { ordinal } => PyNodeProvenance::Action { ordinal: *ordinal },
            NodeProvenance::BranchSelector { name } => {
                PyNodeProvenance::BranchSelector { name: name.clone() }
            }
            NodeProvenance::OutputFrame { port, basis } => PyNodeProvenance::OutputFrame {
                port: (*port).into(),
                basis: (*basis).into(),
            },
            NodeProvenance::None => PyNodeProvenance::Missing(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNodeProvenance {
    fn __repr__(&self) -> String {
        match self {
            PyNodeProvenance::BlockComponent { members } => {
                format!("NodeProvenance.BlockComponent(members={})", members.len())
            }
            PyNodeProvenance::TemporalPipe { pipe } => {
                format!("NodeProvenance.TemporalPipe(pipe={})", pipe.__repr__())
            }
            PyNodeProvenance::SpatialPortSubstitution { source, role } => {
                format!("NodeProvenance.SpatialPortSubstitution(source={source:?}, role={role:?})")
            }
            PyNodeProvenance::MemoryPadding { rounds, .. } => {
                format!("NodeProvenance.MemoryPadding(rounds={rounds})")
            }
            PyNodeProvenance::Generator { ordinal } => {
                format!("NodeProvenance.Generator(ordinal={ordinal})")
            }
            PyNodeProvenance::Action { ordinal } => {
                format!("NodeProvenance.Action(ordinal={ordinal})")
            }
            PyNodeProvenance::BranchSelector { name } => {
                format!("NodeProvenance.BranchSelector(name={name:?})")
            }
            // `PyBasis` renders through the upstream `Basis` Display (`"X"` /
            // `"Z"`); its own `__str__` is private to `crate::primitives`.
            PyNodeProvenance::OutputFrame { port, basis } => format!(
                "NodeProvenance.OutputFrame(port={:?}, basis={})",
                port,
                bloq_utils::Basis::from(*basis)
            ),
            PyNodeProvenance::Missing() => "NodeProvenance.Missing()".to_owned(),
        }
    }
}

// ==============================================================================
// Regions and nodes
// ==============================================================================

/// A structured control-flow region's payload, reached via
/// `BloqNodeKind.Region`: a node whose body is a nested `SubGraph` over the
/// same shared template pool. Restart predicates use incoming `Value` edges
/// or a body-local `restart_source`. Edges never cross region boundaries.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "RegionNode", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyRegionNode {
    /// Region carrying a postselection/restart predicate. Native execution
    /// retries failed physical or decoded attempts up to its configured cap.
    RepeatUntilSuccess {
        body: PySubGraph,
        restart_condition: PyClassicalExpr,
        /// The body-local Boolean source feeding the predicate's unfed slots.
        restart_source: Option<PyValueRef>,
    },
}

impl From<&RegionNode> for PyRegionNode {
    fn from(region: &RegionNode) -> Self {
        let RegionNode::RepeatUntilSuccess {
            body,
            restart_condition,
            restart_source,
        } = region;
        Self::RepeatUntilSuccess {
            body: PySubGraph(body.clone()),
            restart_condition: PyClassicalExpr(restart_condition.clone()),
            restart_source: restart_source.map(Into::into),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegionNode {
    /// Which region this is, without destructuring its payload.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> kinds = {n.region.kind for _, n in program.nodes() if n.is_region()}
    ///     >>> bloq.ir.RegionKind.RepeatUntilSuccess in kinds
    ///     True
    #[getter]
    fn kind(&self) -> PyRegionKind {
        PyRegionKind::RepeatUntilSuccess
    }

    fn __repr__(&self) -> String {
        let Self::RepeatUntilSuccess {
            body,
            restart_source,
            ..
        } = self;
        format!(
            "RegionNode.RepeatUntilSuccess(nodes={}, restart_source={:?})",
            body.0.node_count(),
            restart_source
        )
    }
}

/// What a `BloqNode` computes: a quantum payload, classical dataflow, or a
/// control-flow region. Read from `BloqNode.kind` and dispatch with
/// `isinstance` (`BloqNodeKind.Quantum` / `Classical` / `Region`); each
/// variant carries its payload. The `BloqNode` predicates `is_quantum()` /
/// `is_classical()` / `is_region()` test the variant without an
/// `isinstance` check.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "BloqNodeKind", module = "bloq._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyBloqNodeKind {
    Quantum { node: PyQuantumNode },
    Classical { node: PyClassicalNode },
    Region { region: PyRegionNode },
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBloqNodeKind {
    fn __repr__(&self) -> String {
        match self {
            PyBloqNodeKind::Quantum { node } => {
                format!("BloqNodeKind.Quantum(instances={})", node.instances.len())
            }
            PyBloqNodeKind::Classical { node } => {
                format!("BloqNodeKind.Classical(node={})", node.__repr__())
            }
            PyBloqNodeKind::Region { region } => {
                format!("BloqNodeKind.Region(region={})", region.__repr__())
            }
        }
    }
}

/// A graph node: `kind` is what it computes, `provenance` is why it exists.
///
/// `kind` is a `BloqNodeKind` (quantum, classical, or region); the
/// `is_quantum` / `is_classical` / `is_region` predicates dispatch on it
/// without an `isinstance` check. `provenance` records the source-level
/// construct the node realizes.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
///     >>> node = program.node(0)
///     >>> node.is_quantum(), node.is_classical()
///     (True, False)
#[gen_stub_pyclass]
#[pyclass(
    name = "BloqNode",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyBloqNode {
    /// What the node computes (quantum, classical, or region).
    kind: PyBloqNodeKind,
    /// Why the node exists — its source-level origin.
    provenance: PyNodeProvenance,
    /// The node's compile-stable identity, or `None` for a provenance that
    /// carries none. Computed when the snapshot is taken, because the snapshot
    /// does not keep the node it was taken from.
    stable_key: Option<PyNodeKey>,
    /// Optional activation Value slot. A false activation skips this value or region.
    activation: Option<u32>,
}

impl From<&BloqNode> for PyBloqNode {
    fn from(node: &BloqNode) -> Self {
        let kind = match &node.kind {
            BloqNodeKind::Quantum(quantum) => PyBloqNodeKind::Quantum {
                node: quantum.as_ref().into(),
            },
            BloqNodeKind::Classical(classical) => PyBloqNodeKind::Classical {
                node: classical.as_ref().into(),
            },
            BloqNodeKind::Region(region) => PyBloqNodeKind::Region {
                region: region.into(),
            },
        };
        PyBloqNode {
            kind,
            provenance: (&node.provenance).into(),
            stable_key: node.stable_key().map(Into::into),
            activation: node.activation,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBloqNode {
    /// `True` if the node carries a quantum payload.
    fn is_quantum(&self) -> bool {
        matches!(self.kind, PyBloqNodeKind::Quantum { .. })
    }

    /// `True` if the node carries classical dataflow.
    fn is_classical(&self) -> bool {
        matches!(self.kind, PyBloqNodeKind::Classical { .. })
    }

    /// `True` if the node is a control-flow region.
    fn is_region(&self) -> bool {
        matches!(self.kind, PyBloqNodeKind::Region { .. })
    }

    /// The node's quantum payload, or `None` for a classical or region node.
    ///
    /// The unwrapping twin of `is_quantum()`: reaching a payload through
    /// `kind` otherwise costs an `isinstance` narrowing plus an attribute whose
    /// name differs by variant.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    ///     >>> len(program.node(0).quantum.instances)
    ///     1
    ///     >>> program.node(0).region is None
    ///     True
    #[getter]
    fn quantum(&self) -> Option<PyQuantumNode> {
        match &self.kind {
            PyBloqNodeKind::Quantum { node } => Some(node.clone()),
            PyBloqNodeKind::Classical { .. } | PyBloqNodeKind::Region { .. } => None,
        }
    }

    /// The node's classical payload, or `None` for a quantum or region node.
    #[getter]
    fn classical(&self) -> Option<PyClassicalNode> {
        match &self.kind {
            PyBloqNodeKind::Classical { node } => Some(node.clone()),
            PyBloqNodeKind::Quantum { .. } | PyBloqNodeKind::Region { .. } => None,
        }
    }

    /// The node's region payload, or `None` for a quantum or classical node.
    ///
    /// Pair with `RegionNode.kind` to select one region variant in a single
    /// test:
    ///
    /// >>> import bloq
    /// >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    /// >>> bodies = [
    /// ...     node.region.body
    /// ...     for _, node in program.nodes()
    /// ...     if node.region is not None
    /// ...     and node.region.kind is bloq.ir.RegionKind.RepeatUntilSuccess
    /// ... ]
    /// >>> len(bodies)
    /// 1
    #[getter]
    fn region(&self) -> Option<PyRegionNode> {
        match &self.kind {
            PyBloqNodeKind::Region { region } => Some(region.clone()),
            PyBloqNodeKind::Quantum { .. } | PyBloqNodeKind::Classical { .. } => None,
        }
    }

    /// The spliced node's wait duration in whole syndrome-extraction rounds,
    /// or `None` for anything other than memory padding.
    fn memory_rounds(&self) -> Option<u32> {
        match &self.provenance {
            PyNodeProvenance::MemoryPadding { rounds, .. } => Some(*rounds),
            _ => None,
        }
    }

    /// The source block positions this node was fused from (empty unless the
    /// provenance is `BlockComponent`).
    fn block_members(&self) -> Vec<(i32, i32, i32)> {
        match &self.provenance {
            PyNodeProvenance::BlockComponent { members } => members.clone(),
            _ => Vec::new(),
        }
    }

    fn __repr__(&self) -> String {
        // Only the kind's *name*, not its full repr: a node prints inside
        // node lists, so the payload counts would swamp the line.
        let kind = match &self.kind {
            PyBloqNodeKind::Quantum { .. } => "Quantum",
            PyBloqNodeKind::Classical { .. } => "Classical",
            PyBloqNodeKind::Region { .. } => "Region",
        };
        format!(
            "<BloqNode kind={} provenance={}>",
            kind,
            self.provenance.__repr__()
        )
    }
}

// ==============================================================================
// Whole-program traversal
// ==============================================================================

/// One node visited by `Bloq.walk()`: its owner path, its level-local id, and
/// an owned snapshot of the node itself.
///
/// A node id is only unique within its graph level, so `id`
/// alone does not address a node inside a region body. The `path` — the chain
/// of `(owner_node_id, body_selector)` hops from the top level down to this
/// node's level, outermost first — is the missing half of a program-unique
/// address. Each hop is a 2-tuple `(owner_id, selector)` where `selector` is
/// `"body"`; `path` is empty for a
/// top-level node.
///
/// Getters:
///     - `path`: the `(owner_id, selector)` hops to this node's level.
///     - `id`: the node's level-local id.
///     - `node`: an owned `BloqNode` snapshot (dispatch on `node.kind`).
///
/// Method `top_level_ancestor()` returns the outermost enclosing region's id
/// for a nested node, else the node's own `id`. Because ids are level-local, it
/// is the only piece of a nested node's `id`/`path` that keys a top-level
/// `Bloq.node()` lookup.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
///     >>> walk = program.walk()
///
/// The first visited node is the first top-level node — empty path, and
/// `top_level_ancestor()` is just its own id:
///
/// >>> top = walk[0]
/// >>> top.path
/// []
/// >>> top.id
/// 0
/// >>> top.top_level_ancestor()
/// 0
///
/// A node inside a region body carries a non-empty path; its
/// `top_level_ancestor()` resolves to the enclosing region at the top level,
/// which `Bloq.node()` can look up (a body-local `id` cannot):
///
/// >>> nested = next(w for w in walk if w.path)
/// >>> owner_id, selector = nested.path[0]
/// >>> selector == "body"
/// True
/// >>> nested.top_level_ancestor() == owner_id
/// True
/// >>> program.node(nested.top_level_ancestor()).is_region()
/// True
#[gen_stub_pyclass]
#[pyclass(
    name = "WalkNode",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyWalkNode {
    /// The `(owner_node_id, body_selector)` hops from the top level down to
    /// this node's level, outermost first. Empty for a top-level node; the
    /// selector is `"body"`.
    path: Vec<(u32, String)>,
    /// The node's level-local id (unique only within its level).
    id: u32,
    /// An owned snapshot of the visited node.
    node: PyBloqNode,
}

impl From<NodeCx<'_>> for PyWalkNode {
    fn from(cx: NodeCx<'_>) -> Self {
        Self {
            path: cx
                .path
                .segments()
                .iter()
                .map(|segment| (segment.region.0, segment.body.name().to_owned()))
                .collect(),
            id: cx.id.0,
            node: cx.node.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWalkNode {
    /// The top-level ancestor node id: the outermost enclosing region's id for
    /// a node inside a region body, else this node's own `id`.
    ///
    /// Because ids are level-local, this is the only piece of `id`/`path` that
    /// is a valid top-level `Bloq.node()` lookup key.
    fn top_level_ancestor(&self) -> u32 {
        self.path.first().map_or(self.id, |(owner, _)| *owner)
    }

    fn __repr__(&self) -> String {
        format!("<WalkNode path={:?} id={}>", self.path, self.id)
    }
}

// ==============================================================================
// Graph levels: SubGraph and Bloq
// ==============================================================================

// Shared bodies for the graph-level read surface. Upstream `Bloq`'s reads
// delegate 1:1 to its top `SubGraph`, and the Python mirror keeps that shape:
// both pyclasses declare each method separately (stub generation cannot see
// through macros), but every body is a one-line call into these helpers.
mod level_reads {
    use super::*;

    pub(super) fn nodes(level: &SubGraph) -> Vec<(u32, PyBloqNode)> {
        level
            .nodes()
            .map(|(id, node)| (id.0, node.into()))
            .collect()
    }

    pub(super) fn node(level: &SubGraph, id: u32) -> Option<PyBloqNode> {
        level.node(BloqNodeId(id)).map(Into::into)
    }

    pub(super) fn node_ids(level: &SubGraph) -> Vec<u32> {
        level.node_ids().map(|id| id.0).collect()
    }

    pub(super) fn quantum_nodes(level: &SubGraph) -> Vec<(u32, PyQuantumNode)> {
        level
            .quantum_nodes()
            .map(|(id, quantum)| (id.0, quantum.into()))
            .collect()
    }

    pub(super) fn edges(level: &SubGraph) -> Vec<PyBloqEdgeRef> {
        level.edges().map(Into::into).collect()
    }

    pub(super) fn incoming(level: &SubGraph, id: u32) -> Vec<PyBloqEdgeRef> {
        level.incoming(BloqNodeId(id)).map(Into::into).collect()
    }

    pub(super) fn outgoing(level: &SubGraph, id: u32) -> Vec<PyBloqEdgeRef> {
        level.outgoing(BloqNodeId(id)).map(Into::into).collect()
    }

    pub(super) fn edges_between(level: &SubGraph, from: u32, to: u32) -> Vec<PyBloqEdgeRef> {
        level
            .edges_between(BloqNodeId(from), BloqNodeId(to))
            .map(Into::into)
            .collect()
    }

    pub(super) fn has_path(level: &SubGraph, from: u32, to: u32) -> bool {
        level.has_path(BloqNodeId(from), BloqNodeId(to))
    }

    pub(super) fn value_inputs(level: &SubGraph, id: u32) -> Vec<PyValueInput> {
        level.value_inputs(BloqNodeId(id)).map(Into::into).collect()
    }

    pub(super) fn data_inputs(level: &SubGraph, id: u32) -> Vec<PyValueInput> {
        level.data_inputs(BloqNodeId(id)).map(Into::into).collect()
    }

    pub(super) fn value_consumers(level: &SubGraph, id: u32) -> Vec<(u32, u32)> {
        level
            .value_consumers(BloqNodeId(id))
            .map(|(consumer, slot)| (consumer.0, slot))
            .collect()
    }

    pub(super) fn deterministic_emit_order(level: &SubGraph) -> PyResult<Vec<u32>> {
        level
            .deterministic_emit_order()
            .map(|order| order.into_iter().map(|id| id.0).collect())
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }
}

/// Resolves a `WalkNode.path` — `(owner_node_id, body_selector)` hops from
/// the top level, outermost first — to the graph level it names, accumulating
/// the equivalent [`LevelPath`] on the way down.
fn resolve_level_with_path<'a>(
    bloq: &'a bloq_ir::Bloq,
    path: &[(u32, String)],
) -> PyResult<(LevelPath, &'a SubGraph)> {
    let mut level = bloq.top();
    let mut resolved = LevelPath::default();
    for (owner, selector) in path {
        let node = level.node(BloqNodeId(*owner)).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("no node {owner} at the given level"))
        })?;
        let Some(region) = node.try_region() else {
            return Err(errors::InvalidArgumentError::new_err(format!(
                "node {owner} is not a region node"
            )));
        };
        let body_selector = BodySelector::from_name(selector).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("unknown body selector `{selector}`"))
        })?;
        level = region
            .bodies()
            .find(|(candidate, _)| *candidate == body_selector)
            .map(|(_, body)| body)
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err(format!(
                    "region node {owner} has no body selector `{selector}`"
                ))
            })?;
        resolved = resolved.child(BloqNodeId(*owner), body_selector);
    }
    Ok((resolved, level))
}

fn resolve_level<'a>(bloq: &'a bloq_ir::Bloq, path: &[(u32, String)]) -> PyResult<&'a SubGraph> {
    resolve_level_with_path(bloq, path).map(|(_, level)| level)
}

fn resolve_level_path(bloq: &bloq_ir::Bloq, path: &[(u32, String)]) -> PyResult<LevelPath> {
    resolve_level_with_path(bloq, path).map(|(resolved, _)| resolved)
}

/// The Python spelling of a [`LevelPath`]: `(owner_node_id, body_selector)`
/// hops, matching what `WalkNode.path` hands out and what `path=` takes back.
fn level_path_into(path: &LevelPath) -> Vec<(u32, String)> {
    path.segments()
        .iter()
        .map(|segment| (segment.region.0, segment.body.name().to_owned()))
        .collect()
}

/// A level-qualified target for atomic memory-round insertion.
///
/// `Edge(from_node, to_node, path=None)` splits a quantum edge.
/// `After(node, path=...)` appends after a region body's terminal quantum node.
/// Paths use `(region_node_id, "body")` hops, as returned by `walk()`.
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "MemoryRoundTarget", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyMemoryRoundTarget {
    /// A quantum edge at the specified graph level.
    #[pyo3(constructor = (from_node, to_node, *, path=None))]
    Edge {
        from_node: u32,
        to_node: u32,
        path: Option<Vec<(u32, String)>>,
    },
    /// A terminal quantum node inside a region body.
    #[pyo3(constructor = (node, *, path))]
    After { node: u32, path: Vec<(u32, String)> },
}

impl PyMemoryRoundTarget {
    fn resolve(&self, program: &bloq_ir::Bloq) -> PyResult<MemoryRoundTarget> {
        match self {
            Self::Edge {
                from_node,
                to_node,
                path,
            } => Ok(MemoryRoundTarget::Edge {
                path: resolve_level_path(program, path.as_deref().unwrap_or_default())?,
                from: BloqNodeId(*from_node),
                to: BloqNodeId(*to_node),
            }),
            Self::After { node, path } => Ok(MemoryRoundTarget::After {
                path: resolve_level_path(program, path)?,
                node: BloqNodeId(*node),
            }),
        }
    }
}

// ==============================================================================
// Structural queries, stable identity, and classical resolution
// ==============================================================================

/// What kind of control flow a region node implements.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
///     >>> len(program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)) >= 1
///     True
#[gen_stub_pyclass_enum]
#[pyclass(
    name = "RegionKind",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PyRegionKind {
    RepeatUntilSuccess,
}

impl From<PyRegionKind> for RegionKind {
    fn from(kind: PyRegionKind) -> Self {
        match kind {
            PyRegionKind::RepeatUntilSuccess => RegionKind::RepeatUntilSuccess,
        }
    }
}

impl From<RegionKind> for PyRegionKind {
    fn from(kind: RegionKind) -> Self {
        match kind {
            RegionKind::RepeatUntilSuccess => PyRegionKind::RepeatUntilSuccess,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegionKind {
    fn __str__(&self) -> &'static str {
        match self {
            PyRegionKind::RepeatUntilSuccess => "repeat_until_success",
        }
    }
}

/// A program-unique address of one region node: the level it lives at plus its
/// level-local id.
///
/// `path` is in the same `(owner_node_id, body_selector)` shape as
/// `WalkNode.path`, so it can be passed straight back to any `path=` argument.
#[gen_stub_pyclass]
#[pyclass(
    name = "RegionRef",
    module = "bloq._core",
    frozen,
    eq,
    hash,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyRegionRef {
    /// The `(owner_id, selector)` hops to the region's level; empty at the top.
    path: Vec<(u32, String)>,
    /// The region node's level-local id.
    node: u32,
}

impl From<&RegionRef> for PyRegionRef {
    fn from(region: &RegionRef) -> Self {
        Self {
            path: level_path_into(&region.path),
            node: region.node.0,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRegionRef {
    /// The region node this reference names, read out of `program`.
    ///
    /// Args:
    ///     program: The program the reference came from.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `program` has no such region — the usual
    ///         cause is an edit that recycled node ids since the query ran.
    fn resolve(&self, program: PyRef<'_, PyBloq>) -> PyResult<PyRegionNode> {
        self.region(&program.0).map(Into::into)
    }

    /// The body subgraph this region executes, read out of `program`.
    ///
    /// Args:
    ///     program: The program the reference came from.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `program` has no such region.
    fn body(&self, program: PyRef<'_, PyBloq>) -> PyResult<PySubGraph> {
        let RegionNode::RepeatUntilSuccess { body, .. } = self.region(&program.0)?;
        Ok(PySubGraph(body.clone()))
    }

    fn __repr__(&self) -> String {
        format!("<RegionRef node={} depth={}>", self.node, self.path.len())
    }
}

impl PyRegionRef {
    /// The region node this reference names, shared by `resolve` and `body`.
    fn region<'a>(&self, program: &'a bloq_ir::Bloq) -> PyResult<&'a RegionNode> {
        resolve_level(program, &self.path)?
            .node(BloqNodeId(self.node))
            .and_then(BloqNode::try_region)
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err(format!(
                    "no region node {} at the given level",
                    self.node
                ))
            })
    }
}

/// A compile-stable node identity.
///
/// Two programs compiled from the same block graph — including a
/// `compile_clifford_proxy` of it — give corresponding nodes equal keys, even
/// though their `BloqNodeId`s need not match. That makes a key the right thing
/// to carry across compiles, and the right thing to use as a `dict` key;
/// `Bloq.stable_key_map()` turns keys back into ids for one program.
///
/// Opaque by design: compare, hash, and print them, but do not parse `str()`.
///
/// Examples:
///     >>> import bloq
///     >>> graph = bloq.GalleryItem.X_MEMORY.load()
///     >>> keys = bloq.compile(graph, distance=3).stable_key_map()
///     >>> proxy = bloq.compile_clifford_proxy(graph, [], distance=3).stable_key_map()
///     >>> set(keys) & set(proxy) != set()
///     True
#[gen_stub_pyclass]
#[pyclass(
    name = "NodeKey",
    module = "bloq._core",
    frozen,
    eq,
    ord,
    hash,
    str,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PyNodeKey(NodeKey);

impl std::fmt::Display for PyNodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<NodeKey> for PyNodeKey {
    fn from(key: NodeKey) -> Self {
        PyNodeKey(key)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNodeKey {
    /// Tie-break index among nodes that would otherwise share a key, assigned
    /// in deterministic emit order. `0` unless the program has such a clash.
    #[getter]
    fn ordinal(&self) -> u32 {
        self.0.ordinal()
    }

    fn __repr__(&self) -> String {
        format!("<NodeKey {}>", self.0)
    }
}

/// What a classical node computes under a fixed predicate assignment.
///
/// `measurements` is the set of measurement sites XORed together (already
/// cancelled: a site reached an even number of times is absent), `sign` the
/// constant the parity is XORed with, and `decoder_observables` the exact
/// decoder-flip basis XORed into the value.
///
/// The sites are `(instance_id, template_measurement_id)` pairs rather than
/// `InstanceMeasurement` records: a resolution exists to be looked up in
/// `EmissionPlan.measurements` and `PlanStim.measurement_columns`, which are
/// keyed that way, so the recipe indexes the column maps with no conversion.
#[gen_stub_pyclass]
#[pyclass(
    name = "ClassicalResolution",
    module = "bloq._core",
    frozen,
    eq,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyClassicalResolution {
    /// The measurement sites XORed into the value, in ascending order, as
    /// `(instance_id, template_measurement_id)` pairs.
    measurements: Vec<(u32, u32)>,
    /// The constant XORed into the parity.
    sign: bool,
    /// Decoder observable ids XORed into the value, in ascending order.
    decoder_observables: Vec<u32>,
}

impl From<ClassicalResolution> for PyClassicalResolution {
    fn from(resolution: ClassicalResolution) -> Self {
        Self {
            measurements: resolution
                .measurements
                .into_iter()
                .map(|m| (m.instance.0, m.measurement))
                .collect(),
            sign: resolution.sign,
            decoder_observables: resolution.decoder_observables.into_iter().collect(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyClassicalResolution {
    fn __repr__(&self) -> String {
        format!(
            "<ClassicalResolution measurements={} sign={} observables={:?}>",
            self.measurements.len(),
            py_bool(self.sign),
            self.decoder_observables
        )
    }
}

/// The owned counterpart of [`ClassicalAssignment`], which borrows its maps:
/// the Python dicts are materialized here and outlive the borrowed assignment
/// handed to the resolver.
enum PredicateAssignment {
    Uniform(bool),
    Pinned(BTreeMap<u32, bool>),
}

impl PredicateAssignment {
    fn from_arguments(
        pins: Option<bool>,
        forced_observables: Option<HashMap<u32, bool>>,
    ) -> PyResult<Self> {
        match pins {
            Some(_) if forced_observables.is_some() => Err(errors::InvalidArgumentError::new_err(
                "pins=<bool> already fixes every predicate; drop forced_observables=",
            )),
            Some(value) => Ok(Self::Uniform(value)),
            None => Ok(Self::Pinned(
                forced_observables.unwrap_or_default().into_iter().collect(),
            )),
        }
    }

    fn borrow(&self) -> ClassicalAssignment<'_> {
        match self {
            Self::Uniform(value) => ClassicalAssignment::Uniform(*value),
            Self::Pinned(forced_observables) => ClassicalAssignment::Pinned { forced_observables },
        }
    }
}

/// A nested region body: a graph level with the same read accessors as the
/// top level. Snapshot semantics — this is an owned copy, detached from the
/// program it was read from.
#[gen_stub_pyclass]
#[pyclass(name = "SubGraph", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PySubGraph(pub(crate) SubGraph);

#[gen_stub_pymethods]
#[pymethods]
impl PySubGraph {
    /// Every `(id, node)` at this level, in ascending id order.
    fn nodes(&self) -> Vec<(u32, PyBloqNode)> {
        level_reads::nodes(&self.0)
    }

    /// The single quantum node at this level feeding `node`.
    ///
    /// Args:
    ///     node: The consumer node id.
    ///
    /// Raises:
    ///     BloqError: If the node has zero or several quantum inputs.
    fn quantum_input(&self, node: u32) -> PyResult<u32> {
        self.0
            .quantum_input(BloqNodeId(node))
            .map(|id| id.0)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// The unique quantum tail at this level: a quantum node with no outgoing
    /// quantum edge. Classical consumers do not disqualify it.
    ///
    /// Raises:
    ///     BloqError: If the level has zero or several such nodes.
    fn quantum_tail(&self) -> PyResult<u32> {
        self.0
            .quantum_tail()
            .map(|id| id.0)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Every `(key, id)` at this level, keyed by compile-stable identity.
    ///
    /// Raises:
    ///     BloqError: If the level is cyclic.
    fn stable_keys(&self) -> PyResult<Vec<(PyNodeKey, u32)>> {
        self.0
            .stable_keys()
            .map(|keys| {
                keys.into_iter()
                    .map(|(key, node)| (key.into(), node.0))
                    .collect()
            })
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// The node at `id`, or `None` when the id is stale.
    ///
    /// Args:
    ///     id: The node id to look up at this level.
    fn node(&self, id: u32) -> Option<PyBloqNode> {
        level_reads::node(&self.0, id)
    }

    /// Every node id at this level, in ascending id order.
    fn node_ids(&self) -> Vec<u32> {
        level_reads::node_ids(&self.0)
    }

    /// This level's `(id, quantum_node)` pairs in ascending id order (like
    /// `nodes()`), skipping classical and region nodes.
    ///
    /// Ascending id, *not* emit order: a caller appending circuit text wants
    /// `deterministic_emit_order()` filtered by `is_quantum()` instead.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> rus = next(n for _, n in program.nodes() if n.is_region())
    ///     >>> len(rus.region.body.quantum_nodes())
    ///     2
    fn quantum_nodes(&self) -> Vec<(u32, PyQuantumNode)> {
        level_reads::quantum_nodes(&self.0)
    }

    /// The number of nodes at this level.
    #[getter]
    fn node_count(&self) -> usize {
        self.0.node_count()
    }

    /// The number of edges at this level.
    #[getter]
    fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// Every edge at this level.
    fn edges(&self) -> Vec<PyBloqEdgeRef> {
        level_reads::edges(&self.0)
    }

    /// The edges into `id`. Empty for a stale id.
    ///
    /// Args:
    ///     id: The node whose incoming edges to return.
    fn incoming(&self, id: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::incoming(&self.0, id)
    }

    /// The edges out of `id`. Empty for a stale id.
    ///
    /// Args:
    ///     id: The node whose outgoing edges to return.
    fn outgoing(&self, id: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::outgoing(&self.0, id)
    }

    /// Every edge `from_node -> to_node` (parallel edges are legal).
    ///
    /// Args:
    ///     from_node: The source node id.
    ///     to_node: The target node id.
    fn edges_between(&self, from_node: u32, to_node: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::edges_between(&self.0, from_node, to_node)
    }

    /// Whether a directed path `from_node -> to_node` exists (any edge kind).
    ///
    /// Args:
    ///     from_node: The start node id.
    ///     to_node: The destination node id.
    fn has_path(&self, from_node: u32, to_node: u32) -> bool {
        level_reads::has_path(&self.0, from_node, to_node)
    }

    /// The producers wired into `id`'s `Value` slots.
    ///
    /// Args:
    ///     id: The consumer node id whose `Value` inputs to return.
    fn value_inputs(&self, id: u32) -> Vec<PyValueInput> {
        level_reads::value_inputs(&self.0, id)
    }

    /// Boolean and structural composition inputs, excluding activation.
    fn data_inputs(&self, id: u32) -> Vec<PyValueInput> {
        level_reads::data_inputs(&self.0, id)
    }

    /// The consumers of `id`'s Boolean outputs, as `(consumer, slot)`.
    ///
    /// Args:
    ///     id: The producer node id whose `Value` consumers to return.
    fn value_consumers(&self, id: u32) -> Vec<(u32, u32)> {
        level_reads::value_consumers(&self.0, id)
    }

    /// Value producers with no local consumer; independent of declared outputs.
    fn open_value_producers(&self) -> Vec<u32> {
        self.0.open_value_producers().map(|id| id.0).collect()
    }

    /// The declared Boolean source, or None for constant false.
    #[getter]
    fn value_output(&self) -> Option<PyValueRef> {
        self.0.value_output().map(Into::into)
    }

    /// Declared Observable or region ids exporting boundary bindings.
    #[getter]
    fn boundary_outputs(&self) -> Vec<u32> {
        self.0.boundary_outputs().iter().map(|id| id.0).collect()
    }

    /// The level's deterministic Kahn topological order. Raises `BloqError`
    /// on a cyclic graph.
    fn deterministic_emit_order(&self) -> PyResult<Vec<u32>> {
        level_reads::deterministic_emit_order(&self.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "<SubGraph nodes={} edges={}>",
            self.0.node_count(),
            self.0.edge_count()
        )
    }
}

/// Structural counts over every level of stored Bloq IR, returned by ``Bloq.stats``.
/// Physical circuit payloads and runtime executions are not counted.
#[gen_stub_pyclass]
#[pyclass(name = "BloqStats", module = "bloq._core", frozen, from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyBloqStats(BloqStats);

#[gen_stub_pymethods]
#[pymethods]
impl PyBloqStats {
    /// Nodes at every graph level, including region containers.
    #[getter]
    fn node_count(&self) -> usize {
        self.0.node_count
    }
    /// Edges at every graph level.
    #[getter]
    fn edge_count(&self) -> usize {
        self.0.edge_count
    }
    /// Templates in the compiled pool, including unreferenced templates.
    #[getter]
    fn template_count(&self) -> usize {
        self.0.template_count
    }
    /// Node counts by concrete variant, including zero-count types.
    #[getter]
    fn node_counts(&self) -> BTreeMap<String, usize> {
        self.0
            .node_counts
            .iter()
            .map(|(&kind, &count)| (kind.to_owned(), count))
            .collect()
    }
    /// Edge counts by Quantum, Value, and Order, including zero-count types.
    #[getter]
    fn edge_counts(&self) -> BTreeMap<String, usize> {
        self.0
            .edge_counts
            .iter()
            .map(|(&kind, &count)| (kind.to_owned(), count))
            .collect()
    }
    /// Fixed IR execution structure: no retry regions, node activation,
    /// membership guards, guarded seams, or Discard. Ordinary classical
    /// readout/decoder computation and fixed circuit repeats remain static.
    /// This does not certify validity or backend emittability.
    #[getter]
    fn is_static(&self) -> bool {
        self.0.is_static
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A compiled Bloq IR program: a graph of quantum, classical, and region
/// nodes plus a shared template pool. Produced by `compile()` or loaded from
/// the `.bloqir` / `.bloq` exchange formats.
///
/// Read the top graph level with `nodes` / `node` / `node_ids` / `edges`;
/// nested region bodies hang off their `RegionNode` payloads. Serialize with
/// `to_text` / `to_binary` (or `save` to a path) and reload with `from_text`
/// / `from_binary` (or `load`), picking the codec by `.bloqir` (text) or
/// `.bloq` (binary) extension.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> program
///     <Bloq nodes=2 edges=1 qubits=17>
///     >>> len(program.node_ids())
///     2
///     >>> program.node(0).is_quantum()
///     True
///     >>> program.node(99) is None
///     True
///     >>> len(program.edges())
///     1
///     >>> bloq.Bloq.from_text(program.to_text()).to_binary() == program.to_binary()
///     True
///     >>> bloq.Bloq.from_binary(program.to_binary()).to_binary() == program.to_binary()
///     True
#[gen_stub_pyclass]
#[pyclass(name = "Bloq", module = "bloq._core", from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyBloq(pub(crate) bloq_ir::Bloq);

impl From<bloq_ir::Bloq> for PyBloq {
    fn from(bloq: bloq_ir::Bloq) -> Self {
        PyBloq(bloq)
    }
}

impl PyBloq {
    /// Shared `.bloq` decode path for `from_binary` and `load`.
    fn decode_binary(data: &[u8]) -> PyResult<Self> {
        bloq_ir::Bloq::from_binary(data)
            .map(PyBloq)
            .map_err(|e| errors::BinaryDecodeError::new_err(e.to_string()))
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBloq {
    // ---- codecs --------------------------------------------------------------

    /// Return a selected copy for named structural branches and selective
    /// measurements. Keys are `NodeProvenance.BranchSelector.name` values;
    /// every selector must be pinned consistently. Guards implied by the pins
    /// are removed. The original program stays unchanged, and RUS regions
    /// retain their runtime semantics.
    ///
    /// Raises:
    ///     BloqError: If pins are missing, unknown, inconsistent, or the program is invalid.
    fn pin_membership(&self, py: Python<'_>, choices: BTreeMap<String, bool>) -> PyResult<Self> {
        let program = &self.0;
        py.detach(move || program.pin_membership(&choices))
            .map(Self)
            .map_err(|error| errors::BloqError::new_err(error.to_string()))
    }

    /// Whether guarded quantum stages, seams, or activated quantum regions need
    /// membership selection. Classical activation alone does not require pinning.
    fn has_conditional_membership(&self) -> bool {
        self.0.has_conditional_membership()
    }

    /// Parses a program from the `.bloqir` text exchange format.
    ///
    /// Args:
    ///     text: The `.bloqir` text-format source to parse.
    ///
    /// Raises:
    ///     TextParseError: If the text is not a valid `.bloqir` program.
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        bloq_ir::Bloq::from_text(text)
            .map(PyBloq)
            .map_err(|e| errors::TextParseError::new_err(e.to_string()))
    }

    /// Renders the program in the `.bloqir` text exchange format.
    fn to_text(&self) -> String {
        self.0.to_text()
    }

    /// Renders the nested IR dependency graph as a self-contained SVG.
    ///
    /// Quantum stages and region bodies are always shown. Set
    /// `include_classical=False` to hide classical nodes and their edges.
    /// This is a structural view, not a physical circuit or timing diagram.
    #[pyo3(signature = (*, include_classical=true))]
    fn to_svg(&self, py: Python<'_>, include_classical: bool) -> String {
        let program = &self.0;
        py.detach(move || program.to_svg(include_classical))
    }

    /// Decodes a program from the binary `.bloq` exchange format.
    ///
    /// Args:
    ///     data: The binary `.bloq` artifact bytes to decode.
    ///
    /// Raises:
    ///     BinaryDecodeError: If the bytes are not a valid `.bloq` artifact.
    #[staticmethod]
    fn from_binary(py: Python<'_>, data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        // The borrowed buffer cannot cross `detach`, so the bytes are copied
        // first — cheap next to the decode itself.
        let data = data.as_bytes().to_vec();
        py.detach(move || Self::decode_binary(&data))
    }

    /// Encodes the program as a binary `.bloq` artifact.
    fn to_binary<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let source = &self.0;
        let bytes = py.detach(move || source.to_binary());
        PyBytes::new(py, &bytes)
    }

    /// Loads a program from `path`, picking the codec by extension
    /// (`.bloqir` text, `.bloq` binary).
    ///
    /// Args:
    ///     path: The file path to read; its extension selects the codec.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the extension is neither `.bloqir` nor `.bloq`.
    ///     OSError: If the file cannot be read.
    ///     TextParseError: If a `.bloqir` file fails to parse.
    ///     BinaryDecodeError: If a `.bloq` file fails to decode.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        match codec_for(&path)? {
            Codec::Text => {
                let text =
                    std::fs::read_to_string(&path).map_err(|e| errors::io_error(path, &e))?;
                Self::from_text(&text)
            }
            Codec::Binary => {
                let bytes = std::fs::read(&path).map_err(|e| errors::io_error(path, &e))?;
                Self::decode_binary(&bytes)
            }
        }
    }

    /// Saves the program to `path`, picking the codec by extension
    /// (`.bloqir` text, `.bloq` binary).
    ///
    /// Args:
    ///     path: The destination file path; its extension selects the codec.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the extension is neither `.bloqir` nor `.bloq`.
    ///     OSError: If the file cannot be written.
    ///
    /// Examples:
    ///     >>> import bloq, tempfile, os
    ///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    ///     >>> directory = tempfile.mkdtemp()
    ///     >>> for name in ("prog.bloqir", "prog.bloq"):
    ///     ...     path = os.path.join(directory, name)
    ///     ...     program.save(path)
    ///     ...     bloq.Bloq.load(path).to_binary() == program.to_binary()
    ///     True
    ///     True
    fn save(&self, path: PathBuf) -> PyResult<()> {
        let bytes = match codec_for(&path)? {
            Codec::Text => self.0.to_text().into_bytes(),
            Codec::Binary => self.0.to_binary(),
        };
        std::fs::write(&path, bytes).map_err(|e| errors::io_error(path, &e))
    }

    // ---- program state -------------------------------------------------------

    /// Checks the program against the IR's well-formedness rules. Raises
    /// `BloqValidationError` with the violated rule in the message.
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        let source = &self.0;
        py.detach(move || source.validate())
            .map_err(|e| errors::BloqValidationError::new_err(e.to_string()))
    }

    /// Namespaced producer metadata. Values are Python `int` or `str` scalars.
    #[gen_stub(override_return_type(type_repr = "dict[str, int | str]"))]
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        self.0
            .metadata()
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    MetadataValue::U64(value) => (*value).into_py_any(py)?,
                    MetadataValue::String(value) => value.as_str().into_py_any(py)?,
                };
                Ok((key.clone(), value))
            })
            .collect()
    }

    /// Per-output terminal Pauli-frame signs, ordered by source output port.
    fn output_frames(&self) -> Vec<PyFramePair> {
        self.0
            .output_frames()
            .into_iter()
            .map(|pair| PyFramePair {
                port: pair.port.into(),
                x: pair.x.0,
                z: pair.z.0,
            })
            .collect()
    }

    /// Complete terminal logical X/Z operators used by execution hooks.
    fn logical_outputs(&self) -> Vec<PyLogicalOutput> {
        self.0.logical_outputs().iter().map(Into::into).collect()
    }

    // ---- counts and layout -----------------------------------------------------

    /// The number of top-level nodes. Use ``stats()`` for all levels.
    #[getter]
    fn node_count(&self) -> usize {
        self.0.node_count()
    }

    /// The number of top-level edges. Use ``stats()`` for all levels.
    #[getter]
    fn edge_count(&self) -> usize {
        self.0.edge_count()
    }

    /// The number of top-level quantum nodes (excludes classical/region).
    #[getter]
    fn quantum_node_count(&self) -> usize {
        self.0.quantum_node_count()
    }

    /// The number of distinct layout qubits the program touches.
    ///
    /// Raises:
    ///     InvalidArgumentError: If translating a template qubit by its
    ///         instance offset exceeds the 32-bit coordinate range.
    #[getter]
    fn qubit_count(&self) -> PyResult<usize> {
        self.0.qubit_count().map_err(errors::invalid_argument)
    }

    /// Structural IR inventory across every graph level, without flattening.
    ///
    /// Counts include every concrete node/edge type and the compiled template
    /// pool. Each stored region body contributes once. Static means no retry
    /// regions, node activation, quantum membership guards, guarded seams, or
    /// shot discard. Ordinary classical readout/decoder computation and fixed
    /// circuit repeats remain static. This does not certify validity or backend
    /// emittability. Physical qubits, gates, and measurements are not counted.
    ///
    /// Returns:
    ///     BloqStats: Read-only structural counts; ``print(program.stats())``
    ///         prints a tree summary.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a structural count overflows.
    fn stats(&self) -> PyResult<PyBloqStats> {
        self.0
            .stats()
            .map(PyBloqStats)
            .map_err(errors::invalid_argument)
    }

    /// The number of static measurement sites. Both branch arms count.
    #[getter]
    fn measurement_count(&self) -> usize {
        self.0.measurement_count()
    }

    /// The number of templates in the program's shared pool.
    #[getter]
    fn template_count(&self) -> usize {
        self.0.templates().len()
    }

    /// The template with the given pool id. Ids run `0..template_count` and
    /// are stable for the program's life.
    ///
    /// Args:
    ///     id: The template pool id (e.g. `TemplateInstance.template_id`).
    ///
    /// Raises:
    ///     InvalidArgumentError: If `id` is not in the pool.
    fn template(&self, id: u32) -> PyResult<crate::circuit::PyTemplate> {
        self.0
            .templates()
            .get(TemplateId(id))
            .map(Into::into)
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err(format!("unknown template id {id}"))
            })
    }

    /// Number of shared detector bundles in the program pool.
    #[getter]
    fn detector_bundle_count(&self) -> usize {
        self.0.detector_bundles().len()
    }

    /// A shared detector bundle by pool id.
    fn detector_bundle(&self, id: u32) -> PyResult<PyDetectorBundle> {
        self.0
            .detector_bundles()
            .get(bloq_ir::DetectorBundleId(id))
            .map(Into::into)
            .ok_or_else(|| {
                errors::InvalidArgumentError::new_err(format!("unknown detector bundle id {id}"))
            })
    }

    /// The node's merged circuit plus the instance-space id maps — the
    /// per-node unit a custom backend emits. See `EmissionPlan`.
    ///
    /// A minimal gate-counting backend over a program with no region nodes:
    ///
    /// >>> import bloq
    /// >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
    /// >>> gates = 0
    /// >>> for node_id in program.deterministic_emit_order():
    /// ...     plan = program.emission_plan(node_id)
    /// ...     for body in range(plan.circuit.body_count):
    /// ...         gates += sum(isinstance(op, bloq.ir.CircuitOp.Gate)
    /// ...                      for op in plan.circuit.ops(body))
    /// >>> gates > 0
    /// True
    ///
    /// `deterministic_emit_order()` lists only the top level, so this loop
    /// covers a program with no region nodes. A backend that must handle
    /// control flow walks every level instead — iterate `walk()` (or
    /// `levels()`) and pass each node's `path` — because region bodies hold
    /// quantum nodes this top-level loop never reaches.
    ///
    /// Args:
    ///     node: The node's level-local id.
    ///     path: The node's graph level as `(owner_node_id, body_selector)`
    ///         hops from the top level (a `WalkNode.path`). `None` or empty
    ///         for a top-level node.
    ///     noise: Uniform circuit-level noise probability in `[0, 1]`.
    ///         `None` (the default) leaves the circuit noiseless. Noise is
    ///         materialized after instance merging. When a moment crosses a
    ///         repeat boundary, use `plan.normalized_template(id) or
    ///         program.template(id)` for the returned plan's side tables.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `noise` is invalid, `path` names a missing
    ///         region or body, or `node` is not at that level.
    ///     BloqError: If the node's instances do not merge.
    #[pyo3(signature = (node, path=None, *, noise=None))]
    fn emission_plan(
        &self,
        py: Python<'_>,
        node: u32,
        path: Option<Vec<(u32, String)>>,
        noise: Option<f64>,
    ) -> PyResult<crate::circuit::PyEmissionPlan> {
        let noise = noise.map(crate::compile::uniform_noise).transpose()?;
        let level = resolve_level(&self.0, path.as_deref().unwrap_or(&[]))?;
        let node = level.node(BloqNodeId(node)).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("no node {node} at the given level"))
        })?;
        let templates = self.0.templates();
        py.detach(move || {
            let options = noise
                .as_ref()
                .map_or_else(InstantiationOptions::default, InstantiationOptions::noisy);
            node.emission_plan_with_options(templates, options)
        })
        .map(Into::into)
        .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// A flattened view of the edge-owned memory-padding provenance.
    /// Inspect `BloqEdge.Quantum.padding` when the owning seam matters.
    fn pipe_padding(&self) -> Vec<crate::circuit::PyPipePadding> {
        self.0.pipe_padding().map(Into::into).collect()
    }

    /// The layout qubit coordinates, sorted by `(x, y)`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If translating a template qubit by its
    ///         instance offset exceeds the 32-bit coordinate range.
    fn sorted_layout_coords(&self) -> PyResult<Vec<(i32, i32)>> {
        Ok(self
            .0
            .sorted_layout_coords()
            .map_err(errors::invalid_argument)?
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// The layout qubits the node at `id` occupies, sorted by `(x, y)`.
    ///
    /// Args:
    ///     id: The node id whose layout qubits to return.
    ///
    /// Raises:
    ///     InvalidArgumentError: If no node has the given id (stale id), or
    ///         translating a template qubit by its instance offset exceeds
    ///         the 32-bit coordinate range.
    fn node_qubits(&self, id: u32) -> PyResult<Vec<(i32, i32)>> {
        let node = self.0.node(BloqNodeId(id)).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("no node with id {id}"))
        })?;
        let mut coords: Vec<(i32, i32)> = self
            .0
            .node_qubits(node)
            .map_err(errors::invalid_argument)?
            .into_iter()
            .map(Into::into)
            .collect();
        coords.sort_unstable();
        Ok(coords)
    }

    /// The static measurement-site count under `id`; all region bodies count.
    ///
    /// Args:
    ///     id: The node id whose measurement count to return.
    ///
    /// Raises:
    ///     InvalidArgumentError: If no node has the given id (stale id).
    fn node_measurement_count(&self, id: u32) -> PyResult<usize> {
        let node = self.0.node(BloqNodeId(id)).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("no node with id {id}"))
        })?;
        Ok(self.0.node_measurement_count(node))
    }

    /// The deterministic Kahn topological emit order. Raises `BloqError` on a
    /// cyclic (unvalidated or hand-built) graph.
    fn deterministic_emit_order(&self) -> PyResult<Vec<u32>> {
        level_reads::deterministic_emit_order(self.0.top())
    }

    // ---- top-level graph reads (mirror `SubGraph`) -----------------------------
    //
    // There is deliberately no `top()`: every `SubGraph` read below is mirrored
    // straight onto `Bloq`, so `program.top().nodes()` only ever meant
    // `program.nodes()` plus a deep copy of the whole level. Code that really
    // wants the top level as a `SubGraph` value — to hand to a helper that also
    // takes region bodies — spells the snapshot out as `level_at([])`.

    /// The one graph level `path` names, as an owned snapshot.
    ///
    /// `path` is the `(owner_node_id, body_selector)` chain `levels()` and
    /// `WalkNode.path` hand out; an empty path is the top level. This is the
    /// targeted read that `levels()` is the exhaustive one of — reach for it
    /// instead of scanning `levels()` for a known path, which snapshots every
    /// level to keep one.
    ///
    /// Args:
    ///     path: The `(owner_id, selector)` hops to the level, outermost
    ///         first.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a hop names a missing node, a non-region
    ///         node, or a body selector the region does not have.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> rus = program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess)[0]
    ///     >>> program.level_at(rus.path + [(rus.node, "body")]).node_count
    ///     5
    ///     >>> program.level_at([]).node_ids() == program.node_ids()
    ///     True
    fn level_at(&self, path: Vec<(u32, String)>) -> PyResult<PySubGraph> {
        resolve_level(&self.0, &path).map(|level| PySubGraph(level.clone()))
    }

    /// Every node at every nesting level, as `WalkNode` snapshots in pre-order
    /// depth-first order: a level's nodes in ascending id order, descending
    /// into each region node's bodies before moving on.
    ///
    /// This is the whole-program counterpart to `nodes()` (top level only). It
    /// eagerly visits *every* node, including those in region bodies; there is
    /// no pruning or early exit — filter or slice the returned list yourself.
    /// The ordering is the deterministic scan order (ascending id per level),
    /// not the execution schedule; use `deterministic_emit_order()` per level
    /// for that.
    ///
    /// Each `WalkNode` carries the node's `path` (the `(owner_id, selector)`
    /// hops to its level, empty at the top level), its level-local `id`, and an
    /// owned `node` snapshot. Dispatch on what a node computes with `isinstance`
    /// over `w.node.kind` (`BloqNodeKind.Quantum` / `Classical` / `Region`).
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> walk = program.walk()
    ///
    /// The top-level nodes come out first, in node-id order; the walk then
    /// descends into region bodies, so it visits more than the top level:
    ///
    /// >>> [w.id for w in walk if not w.path] == program.node_ids()
    /// True
    /// >>> len(walk) > program.node_count
    /// True
    ///
    /// Dispatch on what each node computes. The magic-state (`T`) program
    /// lowers to nested control-flow regions, so the walk sees region nodes:
    ///
    /// >>> tally = {"quantum": 0, "classical": 0, "region": 0}
    /// >>> for w in walk:
    /// ...     kind = w.node.kind
    /// ...     if isinstance(kind, bloq.ir.BloqNodeKind.Quantum):
    /// ...         tally["quantum"] += 1
    /// ...     elif isinstance(kind, bloq.ir.BloqNodeKind.Classical):
    /// ...         tally["classical"] += 1
    /// ...     elif isinstance(kind, bloq.ir.BloqNodeKind.Region):
    /// ...         tally["region"] += 1
    /// >>> sum(tally.values()) == len(walk)
    /// True
    /// >>> tally["region"] > 0
    /// True
    ///
    /// A node inside a region body has a non-empty `path`; its
    /// `top_level_ancestor()` is the outermost enclosing region's id — and
    /// because ids are level-local, that is the only piece of a nested node's
    /// `id`/`path` that is a valid top-level `node()` lookup key:
    ///
    /// >>> nested = next(w for w in walk if w.path)
    /// >>> nested.path[0][1] == "body"
    /// True
    /// >>> program.node(nested.top_level_ancestor()).is_region()
    /// True
    fn walk(&self) -> Vec<PyWalkNode> {
        let mut visited = Vec::new();
        self.0.walk(|cx| {
            visited.push(cx.into());
            WalkControl::Continue
        });
        visited
    }

    /// Every graph level as `(path, level)`, pre-order top-down: the top level
    /// first (empty path), then each region body.
    ///
    /// `path` is the `(owner_node_id, body_selector)` chain to the level and
    /// `level` is an owned `SubGraph` snapshot. Use this for whole-program table
    /// passes that read a level as a unit — instance collection, id allocation,
    /// per-level edge queries — rather than reacting to individual nodes (that
    /// is `walk()`).
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> levels = program.levels()
    ///     >>> path, top = levels[0]
    ///     >>> path
    ///     []
    ///     >>> top.node_ids() == program.node_ids()
    ///     True
    ///     >>> top.node_count == program.node_count
    ///     True
    ///
    /// A per-level table pass. Summing each level's node count reaches every
    /// nested node, so the total matches the full `walk()`:
    ///
    /// >>> sum(level.node_count for _path, level in levels) == len(program.walk())
    /// True
    ///
    /// Nested levels carry a non-empty path naming their owner region and body
    /// selector:
    ///
    /// >>> len(levels) > 1
    /// True
    /// >>> nested_paths = [path for path, _level in levels if path]
    /// >>> all(
    /// ...     selector == "body"
    /// ...     for path in nested_paths
    /// ...     for _owner, selector in path
    /// ... )
    /// True
    fn levels(&self) -> Vec<(Vec<(u32, String)>, PySubGraph)> {
        self.0
            .levels()
            .map(|(path, level)| {
                let path = path
                    .segments()
                    .iter()
                    .map(|segment| (segment.region.0, segment.body.name().to_owned()))
                    .collect();
                (path, PySubGraph(level.clone()))
            })
            .collect()
    }

    /// Every top-level `(id, node)`, in ascending id order.
    fn nodes(&self) -> Vec<(u32, PyBloqNode)> {
        level_reads::nodes(self.0.top())
    }

    /// The top-level node at `id`, or `None` when the id is stale.
    ///
    /// Args:
    ///     id: The top-level node id to look up.
    fn node(&self, id: u32) -> Option<PyBloqNode> {
        level_reads::node(self.0.top(), id)
    }

    /// Every top-level node id, in ascending id order.
    fn node_ids(&self) -> Vec<u32> {
        level_reads::node_ids(self.0.top())
    }

    /// The top level's `(id, quantum_node)` pairs in ascending id order,
    /// skipping classical and region nodes. See the `SubGraph` twin for why
    /// this is not the emit order.
    fn quantum_nodes(&self) -> Vec<(u32, PyQuantumNode)> {
        level_reads::quantum_nodes(self.0.top())
    }

    /// Every top-level edge.
    fn edges(&self) -> Vec<PyBloqEdgeRef> {
        level_reads::edges(self.0.top())
    }

    /// The top-level edges into `id`. Empty for a stale id.
    ///
    /// Args:
    ///     id: The top-level node whose incoming edges to return.
    fn incoming(&self, id: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::incoming(self.0.top(), id)
    }

    /// The top-level edges out of `id`. Empty for a stale id.
    ///
    /// Args:
    ///     id: The top-level node whose outgoing edges to return.
    fn outgoing(&self, id: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::outgoing(self.0.top(), id)
    }

    /// Every top-level edge `from_node -> to_node`.
    ///
    /// Args:
    ///     from_node: The source node id.
    ///     to_node: The target node id.
    fn edges_between(&self, from_node: u32, to_node: u32) -> Vec<PyBloqEdgeRef> {
        level_reads::edges_between(self.0.top(), from_node, to_node)
    }

    /// Whether a directed top-level path `from_node -> to_node` exists.
    ///
    /// Args:
    ///     from_node: The start node id.
    ///     to_node: The destination node id.
    fn has_path(&self, from_node: u32, to_node: u32) -> bool {
        level_reads::has_path(self.0.top(), from_node, to_node)
    }

    /// The producers wired into `id`'s `Value` slots.
    ///
    /// Args:
    ///     id: The consumer node id whose `Value` inputs to return.
    fn value_inputs(&self, id: u32) -> Vec<PyValueInput> {
        level_reads::value_inputs(self.0.top(), id)
    }

    /// Boolean and structural composition inputs, excluding activation.
    fn data_inputs(&self, id: u32) -> Vec<PyValueInput> {
        level_reads::data_inputs(self.0.top(), id)
    }

    /// The consumers of `id`'s Boolean outputs, as `(consumer, slot)`.
    ///
    /// Args:
    ///     id: The producer node id whose `Value` consumers to return.
    fn value_consumers(&self, id: u32) -> Vec<(u32, u32)> {
        level_reads::value_consumers(self.0.top(), id)
    }

    /// Top-level value producers with no local consumer; independent of declared outputs.
    fn open_value_producers(&self) -> Vec<u32> {
        self.0.top().open_value_producers().map(|id| id.0).collect()
    }

    /// The declared Boolean source, or None for constant false.
    #[getter]
    fn value_output(&self) -> Option<PyValueRef> {
        self.0.top().value_output().map(Into::into)
    }

    /// Declared Observable or region ids exporting boundary bindings.
    #[getter]
    fn boundary_outputs(&self) -> Vec<u32> {
        self.0
            .top()
            .boundary_outputs()
            .iter()
            .map(|id| id.0)
            .collect()
    }

    // ---- structural queries -----------------------------------------------------

    /// Every region of `kind`, at every nesting level, in level order.
    ///
    /// Args:
    ///     kind: The `RegionKind` to filter by.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.T_GATE.load(), distance=3)
    ///     >>> len(program.regions_of(bloq.ir.RegionKind.RepeatUntilSuccess))
    ///     1
    fn regions_of(&self, kind: PyRegionKind) -> Vec<PyRegionRef> {
        self.0
            .regions_of(kind.into())
            .iter()
            .map(Into::into)
            .collect()
    }

    /// The `(source, selection)` quantum edges feeding top-level guarded
    /// quantum selections, in ascending target- then source-node order.
    ///
    /// Use these seams as targets for `insert_memory_rounds_batch`.
    ///
    /// Raises:
    ///     BloqError: If a selection has no quantum input edge.
    fn selection_seams(&self) -> PyResult<Vec<(u32, u32)>> {
        self.0
            .selection_seams()
            .map(|seams| seams.into_iter().map(|(from, to)| (from.0, to.0)).collect())
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// The top-level node fused from the source block at `pos`, or `None`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` source-graph block position.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    ///     >>> program.node_by_block((0, 0, 0)) is not None
    ///     True
    fn node_by_block(&self, pos: (i32, i32, i32)) -> Option<u32> {
        self.0.node_by_block(pos.into()).map(|id| id.0)
    }

    /// The single top-level quantum node feeding `node`.
    ///
    /// Args:
    ///     node: The consumer node id.
    ///
    /// Raises:
    ///     BloqError: If the node has zero or several quantum inputs.
    fn quantum_input(&self, node: u32) -> PyResult<u32> {
        self.0
            .top()
            .quantum_input(BloqNodeId(node))
            .map(|id| id.0)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// The unique top-level quantum node with no outgoing quantum edge.
    /// Classical consumers do not disqualify it.
    ///
    /// Raises:
    ///     BloqError: If the level has zero or several such nodes.
    fn quantum_tail(&self) -> PyResult<u32> {
        self.0
            .top()
            .quantum_tail()
            .map(|id| id.0)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Top-level nodes with a compile-stable identity, keyed by that identity.
    ///
    /// Use it to carry node references between two compiles of one graph — a
    /// program and its `compile_clifford_proxy`, say — where ids need not
    /// agree but keys do.
    ///
    /// Raises:
    ///     BloqError: If two nodes share a key, or the graph is cyclic.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    ///     >>> keys = program.stable_key_map()
    ///     >>> len(keys)
    ///     2
    ///     >>> all(program.node(node_id) is not None for node_id in keys.values())
    ///     True
    fn stable_key_map(&self) -> PyResult<HashMap<PyNodeKey, u32>> {
        self.0
            .stable_key_map()
            .map(|map| {
                map.into_iter()
                    .map(|(key, node)| (key.into(), node.0))
                    .collect()
            })
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Resolves classical dataflow to its measurement parity, constant sign,
    /// and decoder observable under a fixed predicate assignment.
    ///
    /// Args:
    ///     node: The top-level classical node id to resolve.
    ///     output: The selected Corrected or Flip port.
    ///     pins: One Boolean value for every predicate leaf.
    ///     forced_observables: Observable index to assumed decoded value.
    ///         Cannot be combined with `pins`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If both assignment arguments are supplied.
    ///     BloqError: If the node is missing, is not classical, or its
    ///         dataflow cannot be resolved.
    #[pyo3(signature = (node, *, output=PyObservableOutput::Corrected, pins=None, forced_observables=None))]
    fn resolve_classical(
        &self,
        node: u32,
        output: PyObservableOutput,
        pins: Option<bool>,
        forced_observables: Option<HashMap<u32, bool>>,
    ) -> PyResult<PyClassicalResolution> {
        let assignment = PredicateAssignment::from_arguments(pins, forced_observables)?;
        self.0
            .resolve_value(
                ValueRef {
                    node: BloqNodeId(node),
                    output: output.into(),
                },
                assignment.borrow(),
            )
            .map(Into::into)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// The Boolean value fixed by the predicate assignment.
    ///
    /// Complete observables read their assumed corrected `forced_observables`
    /// values; compute nodes fold their expressions over those values. A decoder
    /// flip cannot be inferred from a corrected-value assignment.
    ///
    /// Args:
    ///     node: The top-level node id to evaluate.
    ///     output: The selected Corrected or Flip port.
    ///     pins: One Boolean value for every predicate leaf.
    ///     forced_observables: Observable index to assumed decoded value.
    ///         Cannot be combined with `pins`.
    ///
    /// Raises:
    ///     InvalidArgumentError: If both assignment arguments are supplied.
    ///     BloqError: If the node is missing, is quantum, is a
    ///         repeat-until-success region, or reaches an unfixed value.
    #[pyo3(signature = (node, *, output=PyObservableOutput::Corrected, pins=None, forced_observables=None))]
    fn classical_value(
        &self,
        node: u32,
        output: PyObservableOutput,
        pins: Option<bool>,
        forced_observables: Option<HashMap<u32, bool>>,
    ) -> PyResult<bool> {
        let assignment = PredicateAssignment::from_arguments(pins, forced_observables)?;
        self.0
            .classical_ref_value(
                ValueRef {
                    node: BloqNodeId(node),
                    output: output.into(),
                },
                assignment.borrow(),
            )
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    // ---- mutation: templates ---------------------------------------------------

    /// Retunes how many times a template's `REPEAT` body runs.
    ///
    /// Templates are pooled, so this changes every instance of the template.
    /// `repetitions` is the loop's repeat count, not a round count — the
    /// template's straight-line prologue still runs once on top of it.
    ///
    /// Args:
    ///     template: The template pool id.
    ///     repetitions: The new `REPEAT` count.
    ///
    /// Raises:
    ///     BloqError: If the id is not in the pool, or the template does not
    ///         have exactly one repeat loop.
    fn set_template_repetitions(&mut self, template: u32, repetitions: u32) -> PyResult<()> {
        self.0
            .set_template_repetitions(TemplateId(template), repetitions)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    // ---- mutation: memory-round insertion --------------------------------------

    /// Splices memory-round padding into the `Quantum` edge
    /// `from_node -> to_node` at `path`, waiting `rounds` rounds.
    ///
    /// Self-contained: the padding templates come from the program's recorded
    /// edge-owned padding provenance (stamped at compile time), so a program
    /// loaded from a `.bloq` needs neither the source graph nor a recompile.
    /// Node ids held across this call must be re-resolved (id recycling).
    ///
    /// Args:
    ///     from_node: The source node id of the `Quantum` edge to splice into.
    ///     to_node: The target node id of that `Quantum` edge.
    ///     rounds: The number of syndrome-extraction rounds the padding waits.
    ///     path: Optional `(region_id, body_selector)` hops. Omit for a
    ///         top-level edge.
    ///
    /// Returns:
    ///     The id of the newly spliced padding node.
    ///
    /// Raises:
    ///     BloqError: If the seam carries no padding provenance (hand-built
    ///         program, or one compiled before provenance recording) or the
    ///         splice fails.
    #[pyo3(signature = (from_node, to_node, rounds, *, path=None))]
    fn insert_memory_rounds(
        &mut self,
        py: Python<'_>,
        from_node: u32,
        to_node: u32,
        rounds: u32,
        path: Option<Vec<(u32, String)>>,
    ) -> PyResult<u32> {
        let path = resolve_level_path(&self.0, path.as_deref().unwrap_or_default())?;
        let program = &mut self.0;
        py.detach(move || {
            program.insert_memory_rounds(
                MemoryRoundTarget::Edge {
                    path,
                    from: BloqNodeId(from_node),
                    to: BloqNodeId(to_node),
                },
                rounds,
            )
        })
        .map(|id| id.0)
        .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Appends physical memory rounds after a terminal quantum node inside a
    /// region body. Existing ordering consumers and cross-region detectors are
    /// moved to the inserted padding node.
    ///
    /// Args:
    ///     node: Terminal quantum node id within `path`.
    ///     rounds: Number of syndrome-extraction rounds to append.
    ///     path: `(region_id, body_selector)` hops to the node's graph level.
    ///
    /// Returns:
    ///     The inserted padding node's level-local id.
    #[pyo3(signature = (node, rounds, *, path))]
    fn insert_memory_rounds_after(
        &mut self,
        py: Python<'_>,
        node: u32,
        rounds: u32,
        path: Vec<(u32, String)>,
    ) -> PyResult<u32> {
        let path = resolve_level_path(&self.0, &path)?;
        let program = &mut self.0;
        py.detach(move || {
            program.insert_memory_rounds(
                MemoryRoundTarget::After {
                    path,
                    node: BloqNodeId(node),
                },
                rounds,
            )
        })
        .map(|id| id.0)
        .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Inserts memory rounds at all targets atomically.
    ///
    /// Targets may mix `MemoryRoundTarget.Edge` and `MemoryRoundTarget.After`
    /// at different graph levels. Returns inserted node ids in target order;
    /// each id belongs to its target's graph level. An empty list is a no-op.
    /// Any error leaves the entire program unchanged.
    ///
    /// Args:
    ///     targets: Edge or terminal-region targets, located before the edit.
    ///     rounds: Number of memory rounds at every target; must be at least 1.
    ///
    /// Raises:
    ///     InvalidArgumentError: If a target path cannot be resolved.
    ///     BloqError: If membership is unresolved, a target or seam is invalid,
    ///         or `rounds` is zero.
    fn insert_memory_rounds_batch(
        &mut self,
        py: Python<'_>,
        targets: Vec<PyMemoryRoundTarget>,
        rounds: u32,
    ) -> PyResult<Vec<u32>> {
        let targets = targets
            .iter()
            .map(|target| target.resolve(&self.0))
            .collect::<PyResult<Vec<_>>>()?;
        let program = &mut self.0;
        py.detach(move || program.insert_memory_rounds_batch(&targets, rounds))
            .map(|ids| ids.into_iter().map(|id| id.0).collect())
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    /// Lower-level splice for callers that already hold padding template ids.
    /// Verify-before-apply — a failed splice leaves the program untouched.
    ///
    /// Args:
    ///     from_node: The source node id of the `Quantum` edge to splice into.
    ///     to_node: The target node id of that `Quantum` edge.
    ///     padding: The padding instances to insert, each a
    ///         `(template_id, (offset_x, offset_y))` pair.
    ///     rounds: The number of syndrome-extraction rounds the padding waits.
    ///
    /// Returns:
    ///     The id of the newly spliced padding node.
    ///
    /// Raises:
    ///     BloqError: If the splice fails (any `EditError`).
    fn subdivide_quantum_edge(
        &mut self,
        from_node: u32,
        to_node: u32,
        padding: Vec<(u32, (i32, i32))>,
        rounds: u32,
    ) -> PyResult<u32> {
        let padding: Vec<bloq_ir::lowering::PaddingInstance> = padding
            .into_iter()
            .map(|(template, offset)| bloq_ir::lowering::PaddingInstance {
                template: bloq_ir::TemplateId(template),
                offset: offset.into(),
            })
            .collect();
        self.0
            .subdivide_quantum_edge(BloqNodeId(from_node), BloqNodeId(to_node), &padding, rounds)
            .map(|id| id.0)
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    // ---- mutation: repeat-block flattening --------------------------------------

    /// Flattens every `REPEAT` block in the program: each looped template is
    /// unrolled into a straight-line circuit, repeat-scoped detectors expand
    /// into one concrete detector per unrolled iteration (seeded and advanced
    /// by the loop-carried recurrence, matching Stim emission semantics), and
    /// all loop-carried detector state is cleared. Flattened templates are
    /// appended to the pool and instances re-pointed; instance-space
    /// references stay valid because each measurement's final unrolled
    /// occurrence keeps its original id.
    ///
    /// Verify-before-apply — a failed flatten leaves the program untouched.
    /// Inserting memory rounds afterwards splices looped padding templates
    /// again; flatten again if needed.
    ///
    /// Raises:
    ///     BloqError: If a parity references an undefined loop state, an
    ///         instance references a missing template.
    fn flatten(&mut self, py: Python<'_>) -> PyResult<()> {
        let program = &mut self.0;
        py.detach(move || program.flatten())
            .map_err(|e| errors::BloqError::new_err(e.to_string()))
    }

    // ---- dunders ---------------------------------------------------------------

    /// The `.bloqir` text form (same as `to_text()`).
    fn __str__(&self) -> String {
        self.0.to_text()
    }

    /// Program equality is byte equality of the `.bloq` encoding — the same
    /// state the codecs round-trip, so two programs compare equal exactly when
    /// they serialize the same.
    fn __eq__(
        &self,
        #[gen_stub(override_type(type_repr = "builtins.object"))] other: &Self,
    ) -> bool {
        self.0.to_binary() == other.0.to_binary()
    }

    /// Pickles through the binary `.bloq` codec, which makes a program
    /// shippable to a `multiprocessing` worker (sinter's model, chiefly).
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (Py<PyBytes>,))> {
        let constructor = py.get_type::<Self>().getattr("from_binary")?.unbind();
        Ok((
            constructor,
            (PyBytes::new(py, &self.0.to_binary()).unbind(),),
        ))
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// A program owns no shared substructure Python can see — every read hands
    /// out a snapshot — so a deep copy is a plain clone; `memo` is accepted to
    /// satisfy the protocol and ignored.
    #[pyo3(signature = (memo=None))]
    fn __deepcopy__(&self, memo: Option<Bound<'_, PyAny>>) -> Self {
        let _ = memo;
        self.clone()
    }

    fn __repr__(&self) -> String {
        // `qubit_count` fails on a program whose instance offsets overflow the
        // coordinate range. A repr that raises breaks debuggers and pytest
        // failure rendering, so an unrepresentable layout prints as `?`
        // instead — `qubit_count` itself still reports the error.
        let qubits = self
            .0
            .qubit_count()
            .map_or_else(|_| "?".to_owned(), |count| count.to_string());
        format!(
            "<Bloq nodes={} edges={} qubits={qubits}>",
            self.0.node_count(),
            self.0.edge_count(),
        )
    }
}

/// Which exchange codec a path's extension selects.
enum Codec {
    Text,
    Binary,
}

fn codec_for(path: &std::path::Path) -> PyResult<Codec> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == BLOQ_TEXT_EXTENSION => Ok(Codec::Text),
        Some(ext) if ext == BLOQ_BINARY_EXTENSION => Ok(Codec::Binary),
        other => Err(errors::InvalidArgumentError::new_err(format!(
            "cannot pick a Bloq codec for extension {:?}: expected .{BLOQ_TEXT_EXTENSION} \
             (text) or .{BLOQ_BINARY_EXTENSION} (binary)",
            other.unwrap_or("<none>")
        ))),
    }
}

// ==============================================================================
// Registration
// ==============================================================================

pyo3_stub_gen::module_variable!("bloq._core", "BLOQ_TEXT_EXTENSION", String);
pyo3_stub_gen::module_variable!("bloq._core", "BLOQ_BINARY_EXTENSION", String);
pyo3_stub_gen::module_variable!("bloq._core", "BLOQ_TEXT_VERSION", u32);
pyo3_stub_gen::module_variable!("bloq._core", "BLOQ_BINARY_VERSION", u8);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTemporalPipe>()?;
    m.add_class::<PyPipeSeam>()?;
    m.add_class::<PyInstanceMeasurement>()?;
    m.add_class::<PyFramePair>()?;
    m.add_class::<PyLogicalOutput>()?;
    m.add_class::<PyNodeParity>()?;
    m.add_class::<PyDetectorTerm>()?;
    m.add_class::<PyNodeDetector>()?;
    m.add_class::<PyBundleDetectorTerm>()?;
    m.add_class::<PyBundleDetector>()?;
    m.add_class::<PyDetectorBundle>()?;
    m.add_class::<PyDetectorBundleUse>()?;
    m.add_class::<PyNodeRestart>()?;
    m.add_class::<PyInstanceProvenance>()?;
    m.add_class::<PyTemplateInstance>()?;
    m.add_class::<PyQuantumNode>()?;
    m.add_class::<PyQuantumGuard>()?;
    m.add_class::<PyClassicalExpr>()?;
    m.add_class::<PyBoundaryFace>()?;
    m.add_class::<PyInstanceBoundaryOperator>()?;
    m.add_class::<PyClassicalNode>()?;
    m.add_class::<PyObservableOutput>()?;
    m.add_class::<PyValueRef>()?;
    m.add_class::<PyValueRole>()?;
    m.add_class::<PyBloqEdge>()?;
    m.add_class::<PyBloqEdgeRef>()?;
    m.add_class::<PyValueInput>()?;
    m.add_class::<PyNodeProvenance>()?;
    m.add_class::<PyRegionNode>()?;
    m.add_class::<PyBloqNodeKind>()?;
    m.add_class::<PyBloqNode>()?;
    m.add_class::<PyWalkNode>()?;
    m.add_class::<PyRegionKind>()?;
    m.add_class::<PyRegionRef>()?;
    m.add_class::<PyMemoryRoundTarget>()?;
    m.add_class::<PyNodeKey>()?;
    m.add_class::<PyClassicalResolution>()?;
    m.add_class::<PySubGraph>()?;
    m.add_class::<PyBloqStats>()?;
    m.add_class::<PyBloq>()?;

    m.add("BLOQ_TEXT_EXTENSION", BLOQ_TEXT_EXTENSION)?;
    m.add("BLOQ_BINARY_EXTENSION", BLOQ_BINARY_EXTENSION)?;
    m.add("BLOQ_TEXT_VERSION", BLOQ_TEXT_VERSION)?;
    m.add("BLOQ_BINARY_VERSION", BLOQ_BINARY_VERSION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use bloq_circuit::{CoordCircuit, PauliBasis};
    use bloq_ir::lowering::{BloqTemplate, TemplateInstanceId};

    use super::*;

    fn overflowing_layout() -> PyBloq {
        let mut circuit = CoordCircuit::new();
        circuit.measure(PauliBasis::Z, [(i32::MAX, 0).into()]);

        let mut bloq = bloq_ir::Bloq::new();
        let template_id = bloq.add_template(BloqTemplate::new(circuit));
        let mut node = BloqNode::from_members(vec![]);
        node.expect_quantum_mut()
            .instances
            .push(TemplateInstance::new(
                TemplateInstanceId(0),
                template_id,
                (1, 0).into(),
            ));
        bloq.add_node(node);
        PyBloq(bloq)
    }

    fn assert_coordinate_error(error: PyErr) {
        Python::initialize();
        Python::attach(|py| {
            assert!(error.is_instance_of::<errors::InvalidArgumentError>(py));
            assert!(error.value(py).to_string().contains("overflows i32"));
        });
    }

    #[test]
    fn layout_accessors_map_coordinate_overflow_to_invalid_argument() {
        let bloq = overflowing_layout();

        assert_coordinate_error(bloq.qubit_count().unwrap_err());
        assert_coordinate_error(bloq.sorted_layout_coords().unwrap_err());
        assert_coordinate_error(bloq.node_qubits(0).unwrap_err());
    }

    /// A `__repr__` that raises breaks debuggers, `print(list_of_programs)`,
    /// and pytest's failure rendering — so the one layout read it makes must
    /// degrade instead of propagating.
    #[test]
    fn repr_survives_an_unrepresentable_layout() {
        assert_eq!(
            overflowing_layout().__repr__(),
            "<Bloq nodes=1 edges=0 qubits=?>"
        );
    }
}
