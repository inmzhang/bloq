//! Backend-authoring read surface: template circuits, per-node emission
//! plans, and pipe-padding provenance.
//!
//! Everything here is a read-only snapshot, converted eagerly like the rest
//! of the IR bindings. The surface mirrors exactly what the built-in Stim
//! backend (`bloq_stim`) consumes from a compiled program, so a custom
//! Python backend target can be written against the same data: for each node
//! take its `Bloq.emission_plan()`, walk the merged circuit's ops, and
//! resolve the node's instance-space side tables (detectors, repeat states,
//! restarts) through the plan's measurement map.
//!
//! To visit every node, iterate `Bloq.walk()` (or `Bloq.levels()`) and pass
//! each node's `path` to `emission_plan` — `Bloq.deterministic_emit_order()`
//! lists only the top level, so it suffices only for programs with no region
//! nodes (whose bodies hold quantum nodes a top-level scan never reaches).
//!
//! Mutable circuit access is deliberately omitted: only the compiler
//! constructs circuits.

use std::collections::HashMap;

use bloq_circuit::{CoordCircuit, DetectorTerm, ExpandedMeasurementColumns, Flow, FlowMarker, Op};
use bloq_ir::lowering::{
    BloqTemplate, NodeEmissionPlan, TemplateDetectorParity, TemplateDetectorScope,
};
use bloq_ir::{PipePadding, TemplateDetector};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_complex_enum, gen_stub_pymethods};

use crate::errors;
use crate::primitives::{PyPauli, PyPauliBasis, py_bool, py_option};

// ==============================================================================
// Circuit ops
// ==============================================================================

/// One feedforward correction inside a `CircuitOp.ConditionalPauli`: apply
/// `pauli` to `target` iff the `control` bit is 1.
#[gen_stub_pyclass]
#[pyclass(
    name = "ConditionalCorrection",
    module = "bloq._core",
    eq,
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PyConditionalCorrection {
    /// The Pauli axis to apply.
    pauli: PyPauliBasis,
    /// The measurement id whose outcome gates the correction.
    control: u32,
    /// The qubit coordinate the correction targets.
    target: (i32, i32),
}

impl From<&bloq_circuit::ConditionalCorrection> for PyConditionalCorrection {
    fn from(correction: &bloq_circuit::ConditionalCorrection) -> Self {
        Self {
            pauli: correction.pauli.into(),
            control: correction.control,
            target: correction.target.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConditionalCorrection {
    fn __repr__(&self) -> String {
        format!(
            "ConditionalCorrection(pauli={}, control={}, target={:?})",
            pauli_basis_name(self.pauli),
            self.control,
            self.target
        )
    }
}

/// The `"X"` / `"Y"` / `"Z"` spelling of a bound Pauli basis, taken from the
/// upstream `PauliBasis` Display: `PyPauliBasis`'s own `__str__` is private to
/// `crate::primitives`, and its `Debug` would leak the binding type's name.
fn pauli_basis_name(basis: PyPauliBasis) -> String {
    bloq_utils::PauliBasis::from(basis).to_string()
}

/// One operation of a `Circuit` body. Dispatch with `isinstance` over the
/// variant classes (`CircuitOp.Gate`, `CircuitOp.Measure`, `CircuitOp.Mpp`,
/// `CircuitOp.Tick`, `CircuitOp.Repeat`, `CircuitOp.ConditionalPauli`).
/// Noise operations use `CircuitOp.Depolarize1`, `CircuitOp.Depolarize2`, and
/// `CircuitOp.PauliError`.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
///     >>> ops = program.template(0).circuit.ops()
///     >>> isinstance(ops[0], (bloq.ir.CircuitOp.Gate, bloq.ir.CircuitOp.Measure,
///     ...                     bloq.ir.CircuitOp.Mpp, bloq.ir.CircuitOp.Tick,
///     ...                     bloq.ir.CircuitOp.Repeat, bloq.ir.CircuitOp.ConditionalPauli,
///     ...                     bloq.ir.CircuitOp.Depolarize1, bloq.ir.CircuitOp.Depolarize2,
///     ...                     bloq.ir.CircuitOp.PauliError))
///     True
#[gen_stub_pyclass_complex_enum]
#[pyclass(name = "CircuitOp", module = "bloq._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) enum PyCircuitOp {
    /// A gate applied to its target qubits (pairwise for two-qubit gates).
    /// `gate` is the Stim-style gate name (e.g. `"H"`, `"CX"`, `"R"`).
    Gate {
        gate: String,
        qubits: Vec<(i32, i32)>,
    },
    /// A single-qubit measurement of each qubit in `basis`;
    /// `measurements[i]` is the record id produced for `qubits[i]`.
    Measure {
        basis: PyPauliBasis,
        qubits: Vec<(i32, i32)>,
        measurements: Vec<u32>,
        flip_probability: f64,
    },
    /// A multi-Pauli-product measurement (Stim `MPP`): `products[i]` is one
    /// jointly measured Pauli-product group as `(qubit, pauli)` pairs, and
    /// produces record `measurements[i]`.
    Mpp {
        products: Vec<Vec<((i32, i32), PyPauli)>>,
        measurements: Vec<u32>,
    },
    /// A moment barrier separating parallel operations.
    Tick(),
    /// Executes body `body` `repetitions` times (Stim `REPEAT`).
    Repeat { body: u32, repetitions: u32 },
    /// Physically applied Pauli feedforward: each correction applies its
    /// Pauli to its target iff its control bit is 1.
    ConditionalPauli {
        corrections: Vec<PyConditionalCorrection>,
    },
    /// Single-qubit depolarizing noise on each target.
    Depolarize1 {
        probability: f64,
        qubits: Vec<(i32, i32)>,
    },
    /// Two-qubit depolarizing noise on consecutive target pairs.
    Depolarize2 {
        probability: f64,
        qubits: Vec<(i32, i32)>,
    },
    /// A fixed-axis Pauli error on each target.
    PauliError {
        probability: f64,
        pauli: PyPauliBasis,
        qubits: Vec<(i32, i32)>,
    },
}

impl From<&Op> for PyCircuitOp {
    fn from(op: &Op) -> Self {
        match op {
            Op::Gate { gate, qubits } => PyCircuitOp::Gate {
                gate: gate.to_string(),
                qubits: qubits.iter().copied().map(Into::into).collect(),
            },
            Op::Measure {
                basis,
                qubits,
                measurements,
                flip_probability,
            } => PyCircuitOp::Measure {
                basis: (*basis).into(),
                qubits: qubits.iter().copied().map(Into::into).collect(),
                measurements: measurements.clone(),
                flip_probability: *flip_probability,
            },
            Op::MPP {
                products,
                measurements,
            } => PyCircuitOp::Mpp {
                products: products
                    .iter()
                    .map(|product| {
                        product
                            .iter()
                            .map(|(coord, pauli)| ((*coord).into(), (*pauli).into()))
                            .collect()
                    })
                    .collect(),
                measurements: measurements.clone(),
            },
            Op::Tick => PyCircuitOp::Tick(),
            Op::Repeat { body, repetitions } => PyCircuitOp::Repeat {
                body: body.0,
                repetitions: *repetitions,
            },
            Op::ConditionalPauli(corrections) => PyCircuitOp::ConditionalPauli {
                corrections: corrections.iter().map(Into::into).collect(),
            },
            Op::Depolarize1 {
                probability,
                qubits,
            } => PyCircuitOp::Depolarize1 {
                probability: *probability,
                qubits: qubits.iter().copied().map(Into::into).collect(),
            },
            Op::Depolarize2 {
                probability,
                qubits,
            } => PyCircuitOp::Depolarize2 {
                probability: *probability,
                qubits: qubits.iter().copied().map(Into::into).collect(),
            },
            Op::PauliError {
                probability,
                pauli,
                qubits,
            } => PyCircuitOp::PauliError {
                probability: *probability,
                pauli: (*pauli).into(),
                qubits: qubits.iter().copied().map(Into::into).collect(),
            },
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCircuitOp {
    fn __repr__(&self) -> String {
        match self {
            PyCircuitOp::Gate { gate, qubits } => {
                format!("CircuitOp.Gate(gate={gate:?}, qubits={})", qubits.len())
            }
            PyCircuitOp::Measure { basis, qubits, .. } => format!(
                "CircuitOp.Measure(basis={}, qubits={})",
                pauli_basis_name(*basis),
                qubits.len()
            ),
            PyCircuitOp::Mpp { products, .. } => {
                format!("CircuitOp.Mpp(products={})", products.len())
            }
            PyCircuitOp::Tick() => "CircuitOp.Tick()".to_owned(),
            PyCircuitOp::Repeat { body, repetitions } => {
                format!("CircuitOp.Repeat(body={body}, repetitions={repetitions})")
            }
            PyCircuitOp::ConditionalPauli { corrections } => format!(
                "CircuitOp.ConditionalPauli(corrections={})",
                corrections.len()
            ),
            PyCircuitOp::Depolarize1 {
                probability,
                qubits,
            } => format!(
                "CircuitOp.Depolarize1(probability={probability}, qubits={})",
                qubits.len()
            ),
            PyCircuitOp::Depolarize2 {
                probability,
                qubits,
            } => format!(
                "CircuitOp.Depolarize2(probability={probability}, qubits={})",
                qubits.len()
            ),
            PyCircuitOp::PauliError {
                probability,
                pauli,
                qubits,
            } => format!(
                "CircuitOp.PauliError(probability={probability}, pauli={}, qubits={})",
                pauli_basis_name(*pauli),
                qubits.len()
            ),
        }
    }
}

// ==============================================================================
// Circuit
// ==============================================================================

/// A physical circuit in qubit-coordinate space: the entry body plus any
/// repeat bodies referenced by `CircuitOp.Repeat`, and a measurement
/// registry giving every measurement a stable id independent of qubit order.
///
/// Read-only snapshot; only the compiler constructs circuits.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
///     >>> circuit = program.template(0).circuit
///     >>> circuit.num_qubits == len(circuit.qubits())
///     True
///     >>> all(record_id < circuit.num_measurements
///     ...     for record_id, _qubit in circuit.measurement_records())
///     True
#[gen_stub_pyclass]
#[pyclass(name = "Circuit", module = "bloq._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyCircuit(pub(crate) CoordCircuit);

impl From<&CoordCircuit> for PyCircuit {
    fn from(circuit: &CoordCircuit) -> Self {
        Self(circuit.clone())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCircuit {
    /// The id of the top-level (entry) body.
    #[getter]
    fn entry_body(&self) -> u32 {
        self.0.entry_body().0
    }

    /// The number of circuit bodies, including repeat bodies.
    #[getter]
    fn body_count(&self) -> usize {
        self.0.body_count()
    }

    /// The ops of one body, in program order. Defaults to the entry body;
    /// pass a `CircuitOp.Repeat`'s `body` id to read a repeat body.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `body` is not an allocated body id.
    #[pyo3(signature = (body=None))]
    fn ops(&self, body: Option<u32>) -> PyResult<Vec<PyCircuitOp>> {
        let id = bloq_circuit::BodyId(body.unwrap_or(self.0.entry_body().0));
        let body = self.0.body(id).ok_or_else(|| {
            errors::InvalidArgumentError::new_err(format!("unknown circuit body id {}", id.0))
        })?;
        Ok(body.ops().iter().map(Into::into).collect())
    }

    /// Every qubit coordinate touched by any body or measurement record,
    /// sorted.
    fn qubits(&self) -> Vec<(i32, i32)> {
        let mut qubits: Vec<(i32, i32)> = self.0.qubits().into_iter().map(Into::into).collect();
        qubits.sort_unstable();
        qubits
    }

    /// The number of distinct qubit coordinates in the circuit.
    #[getter]
    fn num_qubits(&self) -> u32 {
        self.0.num_qubits()
    }

    /// The number of measurement records in the circuit.
    #[getter]
    fn num_measurements(&self) -> u32 {
        self.0.num_measurements()
    }

    /// Every `(measurement_id, qubit)` record, sorted by id.
    fn measurement_records(&self) -> Vec<(u32, (i32, i32))> {
        self.0
            .meas_registry()
            .records()
            .iter()
            .map(|record| (record.id, record.qubit.into()))
            .collect()
    }

    /// Where each measurement id lands once every `REPEAT` block is unrolled.
    ///
    /// A measurement inside a repeat body occupies one column per iteration;
    /// the reported column is its *last* occurrence, which is the one a
    /// `rec[-k]` lookback taken after the loop refers to.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the circuit references an unknown
    ///         measurement id or body, or the unrolled column count overflows.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
    ///     >>> node = program.deterministic_emit_order()[0]
    ///     >>> columns = program.emission_plan(node).circuit.expanded_measurement_columns()
    ///     >>> columns.count >= len(columns)
    ///     True
    fn expanded_measurement_columns(&self) -> PyResult<PyExpandedMeasurementColumns> {
        self.0
            .expanded_measurement_columns()
            .map(PyExpandedMeasurementColumns)
            .map_err(errors::invalid_argument)
    }

    /// The circuit's text form.
    fn __str__(&self) -> String {
        self.0.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Circuit bodies={} qubits={} measurements={}>",
            self.0.body_count(),
            self.0.num_qubits(),
            self.0.num_measurements()
        )
    }
}

/// Post-unroll measurement columns of one circuit, from
/// `Circuit.expanded_measurement_columns()`.
///
/// Behaves like a read-only mapping from measurement id to column: `len()`,
/// `in`, `[]`, and `dict(...)` all work. `count` is the total number of columns
/// the unrolled circuit produces, which is larger than `len()` whenever a
/// measurement repeats.
#[gen_stub_pyclass]
#[pyclass(
    name = "ExpandedMeasurementColumns",
    module = "bloq._core",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyExpandedMeasurementColumns(ExpandedMeasurementColumns);

#[gen_stub_pymethods]
#[pymethods]
impl PyExpandedMeasurementColumns {
    /// The total number of measurement columns after unrolling.
    #[getter]
    fn count(&self) -> u32 {
        self.0.count()
    }

    /// The column of `measurement`, or `None` if the circuit has no such
    /// measurement.
    ///
    /// Args:
    ///     measurement: The pre-unroll measurement id.
    fn column(&self, measurement: u32) -> Option<u32> {
        self.0.column(measurement)
    }

    /// Every `measurement id -> column` pair.
    fn as_dict(&self) -> HashMap<u32, u32> {
        self.0.iter().collect()
    }

    fn __len__(&self) -> usize {
        self.0.iter().count()
    }

    fn __contains__(&self, measurement: u32) -> bool {
        self.0.column(measurement).is_some()
    }

    fn __getitem__(&self, measurement: u32) -> PyResult<u32> {
        self.0
            .column(measurement)
            .ok_or_else(|| PyKeyError::new_err(format!("no measurement {measurement}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "<ExpandedMeasurementColumns measurements={} columns={}>",
            self.0.iter().count(),
            self.0.count()
        )
    }
}

// ==============================================================================
// Templates
// ==============================================================================

/// One compiler-authored open stabilizer flow in template-local coordinates.
///
/// `measurements` contains template-local measurement ids. Resolve them through
/// the owning instance and an [`PyEmissionPlan`]'s `measurements` map before
/// emitting records.
#[gen_stub_pyclass]
#[pyclass(
    name = "BoundaryFlow",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyBoundaryFlow {
    /// Pauli boundary entering the template.
    start: Vec<((i32, i32), PyPauli)>,
    /// Pauli boundary leaving the template.
    end: Vec<((i32, i32), PyPauli)>,
    /// Template-local measurement ids accumulated by the flow.
    measurements: Vec<u32>,
    /// Expected XOR when both boundaries have positive eigenvalue.
    sign: bool,
    /// Flow-center coordinate, when present.
    center: Option<(i32, i32)>,
    /// One of `"detector"`, `"discard"`, or `"restart"`.
    marker: String,
}

impl From<&Flow> for PyBoundaryFlow {
    fn from(flow: &Flow) -> Self {
        let terms = |map: &bloq_circuit::PauliMap| {
            map.iter()
                .map(|(coord, pauli)| ((*coord).into(), PyPauli::from(*pauli)))
                .collect()
        };
        Self {
            start: terms(&flow.start),
            end: terms(&flow.end),
            measurements: flow.measurements.to_vec(),
            sign: flow.sign,
            center: flow.center.map(Into::into),
            marker: match flow.marker {
                FlowMarker::Detector => "detector",
                FlowMarker::Discard => "discard",
                FlowMarker::Restart => "restart",
            }
            .into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBoundaryFlow {
    fn __repr__(&self) -> String {
        format!(
            "<BoundaryFlow marker=\"{}\" start={} end={} measurements={}>",
            self.marker,
            self.start.len(),
            self.end.len(),
            self.measurements.len()
        )
    }
}

/// An XOR of template-local terms — a detector, restart, or recurrence
/// parity in template space. Measurement terms are circuit-local measurement
/// ids of the owning template's circuit; translate them to instance space
/// through an `EmissionPlan`'s `measurements` map.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemplateParity",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplateParity {
    /// Whether this parity includes the constant XOR term.
    sign: bool,
    /// The template-local measurement-id terms, in term order.
    measurements: Vec<u32>,
    /// The loop-state-id terms, in term order.
    loop_states: Vec<u32>,
}

impl From<&TemplateDetectorParity> for PyTemplateParity {
    fn from(parity: &TemplateDetectorParity) -> Self {
        let mut measurements = Vec::new();
        let mut loop_states = Vec::new();
        for term in parity.terms() {
            match term {
                DetectorTerm::Measurement(id) => measurements.push(*id),
                DetectorTerm::LoopState(state) => loop_states.push(state.0),
            }
        }
        Self {
            sign: parity.sign(),
            measurements,
            loop_states,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateParity {
    fn __repr__(&self) -> String {
        format!(
            "<TemplateParity sign={} measurements={} loop_states={}>",
            py_bool(self.sign),
            self.measurements.len(),
            self.loop_states.len()
        )
    }
}

/// A detector closing entirely within one template, in template-local
/// measurement space.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemplateDetector",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplateDetector {
    /// The repeat body the detector fires in, or `None` at the template's
    /// top level.
    body: Option<u32>,
    /// The detector's measurement parity.
    parity: PyTemplateParity,
    /// Decoder coordinates, or `None` when the detector carries none.
    coords: Option<Vec<f64>>,
}

impl From<&TemplateDetector> for PyTemplateDetector {
    fn from(detector: &TemplateDetector) -> Self {
        Self {
            body: match detector.scope {
                TemplateDetectorScope::TopLevel => None,
                TemplateDetectorScope::RepeatBody { body } => Some(body.0),
            },
            parity: (&detector.parity).into(),
            coords: detector.coords.as_ref().map(|coords| coords.to_vec()),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateDetector {
    fn __repr__(&self) -> String {
        format!(
            "<TemplateDetector body={} measurements={} coords={}>",
            py_option(self.body),
            self.parity.measurements.len(),
            self.coords.as_ref().map_or(0, Vec::len)
        )
    }
}

/// A loop-carried detector recurrence on one repeat body, in template-local
/// measurement space: `initial` seeds the state on the first iteration,
/// `next` advances it on every later one.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemplateRepeatState",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplateRepeatState {
    /// The repeat body id.
    body: u32,
    /// The loop-state id this recurrence carries.
    state: u32,
    /// The parity on the first iteration.
    initial: PyTemplateParity,
    /// The parity recurrence for subsequent iterations.
    next: PyTemplateParity,
}

impl From<&bloq_ir::lowering::TemplateRepeatState> for PyTemplateRepeatState {
    fn from(state: &bloq_ir::lowering::TemplateRepeatState) -> Self {
        Self {
            body: state.body.0,
            state: state.state.0,
            initial: (&state.initial).into(),
            next: (&state.next).into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateRepeatState {
    fn __repr__(&self) -> String {
        format!(
            "<TemplateRepeatState body={} state={}>",
            self.body, self.state
        )
    }
}

/// A `RepeatUntilSuccess` restart syndrome in template-local measurement
/// space: when the parity is odd (failure), the enclosing RUS attempt
/// restarts.
#[gen_stub_pyclass]
#[pyclass(
    name = "TemplateRestart",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplateRestart {
    /// The restart syndrome parity.
    parity: PyTemplateParity,
}

impl From<&bloq_ir::lowering::TemplateRestart> for PyTemplateRestart {
    fn from(restart: &bloq_ir::lowering::TemplateRestart) -> Self {
        Self {
            parity: (&restart.parity).into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplateRestart {
    fn __repr__(&self) -> String {
        format!(
            "<TemplateRestart measurements={}>",
            self.parity.measurements.len()
        )
    }
}

/// A reusable circuit template plus its template-local side tables. Pooled
/// and shared by template id; a `TemplateInstance` places one at a layout
/// offset.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.CNOT.load(), distance=3)
///     >>> template = program.template(0)
///     >>> len(template.qubits()) > 0
///     True
///     >>> all(m < template.circuit.num_measurements
///     ...     for d in template.detectors for m in d.parity.measurements)
///     True
#[gen_stub_pyclass]
#[pyclass(
    name = "Template",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyTemplate {
    /// The template's circuit, in template-local qubit coordinates.
    circuit: PyCircuit,
    /// Detectors closing entirely within this template.
    detectors: Vec<PyTemplateDetector>,
    /// Loop-carried detector recurrences for the circuit's repeat bodies.
    repeat_states: Vec<PyTemplateRepeatState>,
    /// Restart syndromes for RUS regions instantiating this template.
    restarts: Vec<PyTemplateRestart>,
    /// Residual open stabilizer flows used to close circuit frontiers.
    boundary_flows: Vec<PyBoundaryFlow>,
}

impl From<&BloqTemplate> for PyTemplate {
    fn from(template: &BloqTemplate) -> Self {
        Self {
            circuit: (&template.circuit).into(),
            detectors: template.detectors.iter().map(Into::into).collect(),
            repeat_states: template.repeat_states.iter().map(Into::into).collect(),
            restarts: template.restarts.iter().map(Into::into).collect(),
            boundary_flows: template.boundary_flows.iter().map(Into::into).collect(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTemplate {
    /// The template circuit's qubit coordinates, sorted.
    fn qubits(&self) -> Vec<(i32, i32)> {
        self.circuit.qubits()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Template qubits={} detectors={} repeat_states={} restarts={} boundary_flows={}>",
            self.circuit.0.num_qubits(),
            self.detectors.len(),
            self.repeat_states.len(),
            self.restarts.len(),
            self.boundary_flows.len()
        )
    }
}

// ==============================================================================
// Emission plans and padding provenance
// ==============================================================================

/// A quantum node's merged circuit plus the maps relating instance-space ids
/// to it — the per-node unit a backend emits. `measurements` translates each
/// instance-space measurement (as an `(instance_id, template_measurement_id)`
/// key, matching `InstanceMeasurement.instance`/`.measurement`) to its merged
/// circuit measurement id; `bodies` translates each `(instance_id, template_body_id)`
/// to its merged body id. Side tables named in instance space (node detectors,
/// accumulates, repeat states) resolve against these maps.
///
/// Classical and region nodes have no instances and yield an empty plan.
// The wrapped `NodeEmissionPlan` is kept whole rather than decomposed into the
// three Python-shaped fields: `emit_plan_stim` lowers the real plan, and the
// tuple-keyed maps below cannot be turned back into one. The getters convert on
// access, which costs the same as the eager conversion did — the circuit was
// already cloned per read.
#[gen_stub_pyclass]
#[pyclass(
    name = "EmissionPlan",
    module = "bloq._core",
    frozen,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyEmissionPlan(NodeEmissionPlan);

impl From<NodeEmissionPlan> for PyEmissionPlan {
    fn from(plan: NodeEmissionPlan) -> Self {
        Self(plan)
    }
}

impl PyEmissionPlan {
    /// The wrapped plan, for the lowering entry points in `crate::plan`.
    pub(crate) fn plan(&self) -> &NodeEmissionPlan {
        &self.0
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmissionPlan {
    /// The source template after repeat expansion for this noisy plan, or
    /// `None` when the original program template still applies.
    ///
    /// Read side tables using
    /// `plan.normalized_template(id) or program.template(id)`. Expansion keeps
    /// template ids but updates detector/restart recurrences and body scopes.
    fn normalized_template(&self, id: u32) -> Option<PyTemplate> {
        self.0
            .normalized_templates
            .as_ref()?
            .get(bloq_ir::TemplateId(id))
            .map(Into::into)
    }

    /// The node's instances translated by offset and merged into one circuit.
    #[getter]
    fn circuit(&self) -> PyCircuit {
        PyCircuit(self.0.circuit.clone())
    }

    /// `(instance_id, template_measurement_id)` → merged measurement id.
    #[getter]
    fn measurements(&self) -> HashMap<(u32, u32), u32> {
        self.0
            .measurements
            .iter()
            .map(|(key, merged)| ((key.instance.0, key.measurement), *merged))
            .collect()
    }

    /// `(instance_id, template_body_id)` → merged body id.
    #[getter]
    fn bodies(&self) -> HashMap<(u32, u32), u32> {
        self.0
            .bodies
            .iter()
            .map(|((instance, body), merged)| ((instance.0, body.0), merged.0))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<EmissionPlan measurements={} bodies={}>",
            self.0.measurements.len(),
            self.0.bodies.len()
        )
    }
}

/// Memory-padding provenance for one temporal pipe on one quantum edge:
/// the precompiled memory-round templates `Bloq.insert_memory_rounds`
/// instantiates and their layout offset. The owning `BloqEdge.Quantum`
/// distinguishes temporal-Hadamard sides and may describe a region terminal.
/// Template ids are valid `Bloq.template()` / `Bloq.subdivide_quantum_edge`
/// inputs.
#[gen_stub_pyclass]
#[pyclass(
    name = "PipePadding",
    module = "bloq._core",
    eq,
    hash,
    frozen,
    get_all,
    from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PyPipePadding {
    /// Layout offset positioning the templates over the pipe's cross-section.
    offset: (i32, i32),
    /// Template implementing exactly one memory round.
    one_round: u32,
    /// Template implementing `1 + r` rounds via a `REPEAT r` body.
    looped: u32,
}

impl From<&PipePadding> for PyPipePadding {
    fn from(padding: &PipePadding) -> Self {
        Self {
            offset: padding.offset.into(),
            one_round: padding.one_round.0,
            looped: padding.looped.0,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPipePadding {
    fn __repr__(&self) -> String {
        format!(
            "PipePadding(offset={:?}, one_round={}, looped={})",
            self.offset, self.one_round, self.looped
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConditionalCorrection>()?;
    m.add_class::<PyCircuitOp>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyExpandedMeasurementColumns>()?;
    m.add_class::<PyBoundaryFlow>()?;
    m.add_class::<PyTemplateParity>()?;
    m.add_class::<PyTemplateDetector>()?;
    m.add_class::<PyTemplateRepeatState>()?;
    m.add_class::<PyTemplateRestart>()?;
    m.add_class::<PyTemplate>()?;
    m.add_class::<PyEmissionPlan>()?;
    m.add_class::<PyPipePadding>()?;
    Ok(())
}
