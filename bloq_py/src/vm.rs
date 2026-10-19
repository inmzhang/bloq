//! Python surface for noiseless physical checks and dynamic VM shots.

use std::collections::BTreeMap;

use bloq_vm::{
    DEFAULT_DECODER_LATENCY_ROUNDS, DEFAULT_MAX_QUANTUM_VARIANTS, ExecutionArtifact,
    LogicalInputState, LoweringConfig, Program, RuntimeConfig, RuntimeLimits, SourceRole,
    SourceTiming, lower, run,
};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::compile::{probability, uniform_noise};
use crate::errors::{self, invalid_argument};
use crate::ir::PyBloq;

fn noiseless_payload(
    program: &bloq_ir::Bloq,
    shots: usize,
    seed: u64,
    input_state: LogicalInputState,
    include_signature: bool,
) -> Result<serde_json::Value, bloq_vm::ExecError> {
    let mut samples = Vec::new();
    let report = bloq_vm::run_bloq_with_io_and_captured_outputs(
        program,
        shots,
        seed,
        |sim, context| {
            if input_state == LogicalInputState::Zero {
                for input in context.inputs {
                    sim.h(input.qubit);
                }
            }
            Ok(())
        },
        |sim, context| {
            let outputs = context.output_logicals()?;
            let corrected = outputs
                .iter()
                .map(|output| {
                    let frame = context
                        .frames
                        .iter()
                        .find(|frame| frame.port == output.port)
                        .and_then(|frame| frame.x.zip(frame.z))
                        .ok_or(bloq_vm::ExecError::MalformedGraph(
                            "noiseless output frame is unavailable",
                        ))?;
                    Ok((output, frame))
                })
                .collect::<Result<Vec<_>, bloq_vm::ExecError>>()?;
            let bloch = corrected
                .iter()
                .map(|(output, frame)| bloq_vm::verify::logical_bloch(sim, output, *frame))
                .collect::<Result<Vec<_>, _>>()?;
            let signature = include_signature
                .then(|| bloq_vm::verify::logical_signature(sim, &corrected))
                .transpose()?;
            samples.push(serde_json::json!({
                "shot": context.shot,
                "logical_bloch": bloch,
                "logical_signature": signature,
                "captured_output_ports": context.saved_outputs.iter()
                    .map(|output| output.port.to_array()).collect::<Vec<_>>(),
                "named_selectors": context.named_branch_selectors,
            }));
            Ok(())
        },
    )?;
    Ok(serde_json::json!({
        "execution": "noiseless",
        "ideal_boundary_capture": true,
        "input_state": match input_state {
            LogicalInputState::Plus => "plus", LogicalInputState::Zero => "zero",
        },
        "shots": report.shots,
        "discarded": report.discarded,
        "peak_rank": report.max_rank,
        "all_detectors_constant": report.all_detectors_constant(),
        "detectors": report.detectors.iter().map(|detector| serde_json::json!({
            "constant": detector.constant, "value": detector.value, "per_shot": detector.per_shot,
        })).collect::<Vec<_>>(),
        "output_ports": program.logical_outputs().iter()
            .map(|output| output.port.to_array()).collect::<Vec<_>>(),
        "samples": samples,
    }))
}

/// Execute compiled physical circuits without added noise or hardware timing.
///
/// Native retry regions and adaptive controls execute normally. External input
/// states and output capture are ideal boundaries. Capturing an output before
/// its physical patch is reused preserves joint output coherence on a private
/// simulator qubit. This function does not replace T circuits with proxies.
///
/// Args:
///     program: Compiled Bloq IR, including any native T retry circuits.
///     shots: Positive number of physical shots.
///     seed: Root simulator seed.
///     input_state: State of every external input, ``"plus"`` or ``"zero"``.
///     include_signature: Include all joint Pauli expectations for at most six
///         logical outputs. This makes ``4**n - 1`` exact expectation reads.
///
/// Returns:
///     A JSON-ready dictionary with ``shots``, ``discarded``, ``peak_rank``,
///     ``output_ports``, ``detectors``, ``all_detectors_constant``, and accepted
///     ``samples``. Each sample contains corrected ``logical_bloch`` vectors,
///     ``captured_output_ports``, ``named_selectors``, and an optional
///     ``logical_signature``. Output order matches ``program.logical_outputs()``.
///     Signature index ``k - 1`` is Pauli word ``k`` in base four, with output
///     zero as the least-significant digit and digits ``I, X, Y, Z``.
///     Discarded shots have no sample. Detector constancy concerns applicable
///     accepted shots; an empty detector history is vacuously constant.
///     Checks spanning ideal input replacement are omitted, because their
///     earlier syndrome records describe the state before replacement.
///
/// Raises:
///     InvalidArgumentError: For zero shots, an invalid input state, or a
///         requested signature with more than six outputs.
///     RuntimeError: If physical execution or logical readout fails.
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "dict[str, object]"))]
#[pyo3(signature = (program, *, shots=1, seed=0, input_state="plus", include_signature=false))]
fn run_noiseless(
    py: Python<'_>,
    program: PyRef<'_, PyBloq>,
    shots: usize,
    seed: u64,
    input_state: &str,
    include_signature: bool,
) -> PyResult<Py<PyAny>> {
    if shots == 0 {
        return Err(invalid_argument("shots must be positive"));
    }
    let input_state = parse_input_state(input_state)?;
    if include_signature && program.0.logical_outputs().len() > 6 {
        return Err(invalid_argument(
            "joint Pauli signatures are limited to six logical outputs",
        ));
    }
    let source = &program.0;
    let payload = py
        .detach(move || noiseless_payload(source, shots, seed, input_state, include_signature))
        .map_err(|error| errors::RuntimeError::new_err(error.to_string()))?;
    let json = serde_json::to_string(&payload)
        .map_err(|error| errors::RuntimeError::new_err(error.to_string()))?;
    Ok(py.import("json")?.call_method1("loads", (json,))?.unbind())
}

const MIN_GATE_DURATION: f64 = 1e-9;

fn nonnegative(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(invalid_argument(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

fn parse_input_state(name: &str) -> PyResult<LogicalInputState> {
    match name {
        "plus" => Ok(LogicalInputState::Plus),
        "zero" => Ok(LogicalInputState::Zero),
        _ => Err(invalid_argument("input_state must be 'plus' or 'zero'")),
    }
}

fn source_role(role: SourceRole) -> &'static str {
    match role {
        SourceRole::Factory => "factory",
        SourceRole::LogicalInput => "logical_input",
        SourceRole::PreparedY => "prepared_y",
        SourceRole::Clifford => "clifford",
    }
}

/// A Bloq IR program lowered once to the dynamic VM instruction stream.
#[gen_stub_pyclass]
#[pyclass(name = "VmProgram", module = "bloq._core", frozen)]
pub(crate) struct PyVmProgram {
    program: Program,
    runtime_defaults: RuntimeConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVmProgram {
    /// Number of simulator qubits.
    #[getter]
    fn qubit_count(&self) -> u32 {
        self.program.qubit_count
    }

    /// Number of VM tasks across all dynamic regions.
    #[getter]
    fn task_count(&self) -> usize {
        self.program.tasks.len()
    }

    /// Number of terminal logical outputs.
    #[getter]
    fn output_count(&self) -> usize {
        self.program.outputs.len()
    }

    /// Configured mock-decoder latency carried by this program, in memory rounds.
    #[getter]
    fn decoder_latency_rounds(&self) -> u32 {
        self.program.decoder_latency_rounds
    }

    /// Independently released source tasks in dense task order.
    #[gen_stub(override_return_type(type_repr = "list[dict[str, int | float | str]]"))]
    #[getter]
    fn sources(&self, py: Python<'_>) -> PyResult<Vec<BTreeMap<String, Py<PyAny>>>> {
        self.program
            .sources()
            .into_iter()
            .map(|source| {
                Ok(BTreeMap::from([
                    ("task".to_owned(), source.task.into_py_any(py)?),
                    ("role".to_owned(), source_role(source.role).into_py_any(py)?),
                    ("release".to_owned(), source.release.into_py_any(py)?),
                    ("duration".to_owned(), source.duration.into_py_any(py)?),
                ]))
            })
            .collect()
    }

    /// Serializes the complete executable instruction stream as JSON.
    ///
    /// This includes tasks, gates, dependencies, entry stream, logical inputs,
    /// logical outputs, and source release times. It is an inspection format,
    /// not a stable persistence protocol.
    ///
    /// Args:
    ///     indent: ``None`` for compact JSON or ``2`` for pretty JSON.
    ///
    /// Raises:
    ///     InvalidArgumentError: If ``indent`` is neither ``None`` nor ``2``.
    ///     RuntimeError: If JSON serialization fails.
    #[pyo3(signature = (*, indent=None))]
    fn to_json(&self, indent: Option<i32>) -> PyResult<String> {
        let result = match indent {
            None => self.program.to_json(),
            Some(2) => self.program.to_json_pretty(),
            Some(_) => return Err(invalid_argument("indent must be None or 2")),
        };
        result.map_err(|error| errors::RuntimeError::new_err(error.to_string()))
    }

    /// Runs one dynamic shot with the Ticit tableau backend.
    ///
    /// Args:
    ///     seed: Root seed for physical noise and mock-decoder draws.
    ///     input_state: Encoded external input, ``"plus"`` or ``"zero"``.
    ///     decoder_acceptance: Mock decoder acceptance probability.
    ///     accepted_accuracy: Residual accuracy for accepted solves.
    ///     rejected_accuracy: Residual accuracy for rejected solves.
    ///     gap_threshold: Confidence threshold, strictly between 0 and 1.
    ///     idle_error_rate: Optional dynamic-idle depolarization rate per time
    ///         unit. Omit to use the rate derived from ``lower_vm(noise=...)``.
    ///     max_steps: Maximum task, moment, and memory-round executions.
    ///     max_attempts: Maximum attempts of one repeat-until-success task.
    ///
    /// Raises:
    ///     InvalidArgumentError: If an option is out of range.
    ///     RuntimeError: If execution fails.
    #[pyo3(signature = (*, seed=0, input_state="plus", decoder_acceptance=0.8, accepted_accuracy=0.999, rejected_accuracy=0.9, gap_threshold=0.5, idle_error_rate=None, max_steps=RuntimeLimits::default().max_steps, max_attempts=RuntimeLimits::default().max_attempts))]
    #[expect(clippy::too_many_arguments, reason = "flat Python keyword API")]
    fn run(
        &self,
        py: Python<'_>,
        seed: u64,
        input_state: &str,
        decoder_acceptance: f64,
        accepted_accuracy: f64,
        rejected_accuracy: f64,
        gap_threshold: f64,
        idle_error_rate: Option<f64>,
        max_steps: u64,
        max_attempts: u32,
    ) -> PyResult<PyVmRunResult> {
        let input_state = parse_input_state(input_state)?;
        let decoder_acceptance = probability("decoder_acceptance", decoder_acceptance)?;
        let accepted_accuracy = probability("accepted_accuracy", accepted_accuracy)?;
        let rejected_accuracy = probability("rejected_accuracy", rejected_accuracy)?;
        if !gap_threshold.is_finite() || !(0.0 < gap_threshold && gap_threshold < 1.0) {
            return Err(invalid_argument(
                "gap_threshold must be finite and strictly between 0 and 1",
            ));
        }
        let idle_error_rate = idle_error_rate
            .map(|rate| nonnegative("idle_error_rate", rate))
            .transpose()?;
        if max_steps == 0 || max_attempts == 0 {
            return Err(invalid_argument(
                "max_steps and max_attempts must be positive",
            ));
        }

        let mut config = self.runtime_defaults.clone();
        config.seed = seed;
        config.input_state = input_state;
        config.decoder.acceptance_probability = decoder_acceptance;
        config.decoder.accepted_accuracy = accepted_accuracy;
        config.decoder.rejected_accuracy = rejected_accuracy;
        config.decoder.gap_threshold = gap_threshold;
        if let Some(rate) = idle_error_rate {
            config.idle_error_rate = rate;
        }
        config.limits = RuntimeLimits {
            max_steps,
            max_attempts,
        };

        let program = &self.program;
        let (artifact, bloch) = py
            .detach(move || {
                let result = run(program, config)?;
                let bloch = program
                    .outputs
                    .iter()
                    .map(|output| {
                        result
                            .logical_bloch(output)
                            .map_err(|error| error.to_string())
                    })
                    .collect();
                Ok::<_, bloq_vm::RuntimeError>((result.artifact, bloch))
            })
            .map_err(|error| errors::RuntimeError::new_err(error.to_string()))?;
        Ok(PyVmRunResult { artifact, bloch })
    }

    fn __repr__(&self) -> String {
        format!(
            "<VmProgram qubits={} tasks={} outputs={}>",
            self.program.qubit_count,
            self.program.tasks.len(),
            self.program.outputs.len()
        )
    }
}

/// One dynamic shot: a JSON-ready trace and terminal logical Bloch vectors.
#[gen_stub_pyclass]
#[pyclass(name = "VmRunResult", module = "bloq._core", frozen)]
pub(crate) struct PyVmRunResult {
    artifact: ExecutionArtifact,
    bloch: Vec<Result<(f64, f64, f64), String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVmRunResult {
    /// Full execution trace as nested Python dictionaries and lists.
    #[gen_stub(override_return_type(type_repr = "dict[str, object]"))]
    #[getter]
    fn trace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let json = self
            .artifact
            .to_json()
            .map_err(|error| errors::RuntimeError::new_err(error.to_string()))?;
        py.import("json")?
            .call_method1("loads", (json,))
            .map(Bound::unbind)
    }

    /// Whether the shot ended without a committed result.
    #[getter]
    fn discarded(&self) -> bool {
        self.artifact.discarded
    }

    /// Physical stop time.
    #[getter]
    fn finished_at(&self) -> f64 {
        self.artifact.finished_at
    }

    /// Highest stabilizer-frame rank reached by the backend.
    #[getter]
    fn peak_rank(&self) -> usize {
        self.artifact.peak_rank
    }

    /// Serializes the trace as compact JSON, or pretty JSON for ``indent=2``.
    ///
    /// Args:
    ///     indent: ``None`` for compact JSON or ``2`` for pretty JSON.
    ///
    /// Raises:
    ///     InvalidArgumentError: If ``indent`` is neither ``None`` nor ``2``.
    ///     RuntimeError: If JSON serialization fails.
    #[pyo3(signature = (*, indent=None))]
    fn to_json(&self, indent: Option<i32>) -> PyResult<String> {
        let result = match indent {
            None => self.artifact.to_json(),
            Some(2) => self.artifact.to_json_pretty(),
            Some(_) => return Err(invalid_argument("indent must be None or 2")),
        };
        result.map_err(|error| errors::RuntimeError::new_err(error.to_string()))
    }

    /// Terminal ``(X, Y, Z)`` Bloch vector with Pauli-frame corrections.
    ///
    /// Args:
    ///     output: Zero-based logical-output index.
    ///
    /// Raises:
    ///     InvalidArgumentError: If ``output`` is out of range.
    ///     RuntimeError: If the shot has no committed logical result.
    #[pyo3(signature = (output=0))]
    fn logical_bloch(&self, output: usize) -> PyResult<(f64, f64, f64)> {
        if self.artifact.discarded {
            return Err(errors::RuntimeError::new_err(
                "discarded shot has no committed logical output",
            ));
        }
        let result = self.bloch.get(output).ok_or_else(|| {
            invalid_argument(format!(
                "logical output index {output} is out of range for {} output(s)",
                self.bloch.len()
            ))
        })?;
        result
            .as_ref()
            .copied()
            .map_err(|error| errors::RuntimeError::new_err(error.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "<VmRunResult discarded={} finished_at={} outputs={}>",
            self.artifact.discarded,
            self.artifact.finished_at,
            self.bloch.len()
        )
    }
}

/// Lowers Bloq IR once to a reusable dynamic VM program.
///
/// Args:
///     program: Compiled Bloq IR.
///     noise: Uniform circuit and idle-noise probability.
///     gate_duration: Duration of each nonempty physical moment; must exceed
///         ``1e-9``.
///     decoder_latency_rounds: Mock-decoder latency in local memory rounds.
///     source_release_time: Release epoch for Clifford sources.
///     input_release_time: External-input release epoch; defaults to
///         ``source_release_time``.
///     factory_release_time: Release epoch for resource factories.
///     max_quantum_variants: Maximum jointly reachable guard tuples for one
///         quantum stage. Defaults to 4096.
///
/// Returns:
///     A reusable ``VmProgram``. Prepared-Y release offsets follow the emitted
///     preparation durations.
///
/// Raises:
///     InvalidArgumentError: If an option is out of range.
///     LowerError: If the IR cannot be lowered exactly within the bound.
///
/// Examples:
///     >>> import bloq
///     >>> ir = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> vm = bloq.lower_vm(ir)
///     >>> vm.task_count > 0
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (program, *, noise=0.0, gate_duration=1.0, decoder_latency_rounds=DEFAULT_DECODER_LATENCY_ROUNDS, source_release_time=0.0, input_release_time=None, factory_release_time=0.0, max_quantum_variants=None))]
#[expect(clippy::too_many_arguments, reason = "flat Python keyword API")]
fn lower_vm(
    py: Python<'_>,
    program: PyRef<'_, PyBloq>,
    noise: f64,
    gate_duration: f64,
    decoder_latency_rounds: u32,
    source_release_time: f64,
    input_release_time: Option<f64>,
    factory_release_time: f64,
    max_quantum_variants: Option<usize>,
) -> PyResult<PyVmProgram> {
    let noise = uniform_noise(noise)?;
    if !gate_duration.is_finite() || gate_duration <= MIN_GATE_DURATION {
        return Err(invalid_argument(
            "gate_duration must be finite and greater than 1e-9",
        ));
    }
    let source_release_time = nonnegative("source_release_time", source_release_time)?;
    let input_release_time = nonnegative(
        "input_release_time",
        input_release_time.unwrap_or(source_release_time),
    )?;
    let factory_release_time = nonnegative("factory_release_time", factory_release_time)?;
    let max_quantum_variants = max_quantum_variants.unwrap_or(DEFAULT_MAX_QUANTUM_VARIANTS);
    if max_quantum_variants == 0 {
        return Err(invalid_argument("max_quantum_variants must be positive"));
    }

    let config = LoweringConfig {
        gate_duration,
        decoder_latency_rounds,
        noise: Some(&noise),
        source_timing: SourceTiming {
            factory: factory_release_time,
            input: input_release_time,
            clifford: source_release_time,
        },
        max_quantum_variants,
    };
    let runtime_defaults = config.runtime_config(0);
    let source = &program.0;
    let program = py
        .detach(move || lower(source, &config))
        .map_err(|error| errors::LowerError::new_err(error.to_string()))?;
    Ok(PyVmProgram {
        program,
        runtime_defaults,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVmProgram>()?;
    m.add_class::<PyVmRunResult>()?;
    m.add_function(wrap_pyfunction!(lower_vm, m)?)?;
    m.add_function(wrap_pyfunction!(run_noiseless, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_python_options_use_rust_vm_defaults() {
        Python::initialize();
        Python::attach(|py| {
            let source = Py::new(py, PyBloq(bloq_ir::Bloq::new())).unwrap();
            let function = wrap_pyfunction!(lower_vm, py).unwrap();
            let program = function.call1((source,)).unwrap();
            let lowered = program.extract::<PyRef<'_, PyVmProgram>>().unwrap();
            assert_eq!(
                lowered.program.decoder_latency_rounds,
                LoweringConfig::default().decoder_latency_rounds
            );
            let result = program.call_method0("run").unwrap();
            let result = result.extract::<PyRef<'_, PyVmRunResult>>().unwrap();
            assert_eq!(result.artifact.metadata.limits, RuntimeLimits::default());
        });
    }
}
