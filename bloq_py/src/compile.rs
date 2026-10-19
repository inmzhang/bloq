//! Compilation and Stim-emission bindings: `BlockGraph` -> `Bloq` -> Stim text.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fmt::Write as _;
use std::str::FromStr;

use bloq_circuit::NoiseModel;
use bloq_compile::{
    CLIFFORD_PROXY_SEED_METADATA_KEY, CODE_DISTANCE_METADATA_KEY, CONVENTION_METADATA_KEY,
    CompileConfig, CompileContext, SharedCompileCache,
};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use stim::{Circuit, CircuitItem, GateTarget};

use crate::errors;
use crate::graph::PyBlockGraph;
use crate::ir::PyBloq;
use crate::primitives::PyBasis;

fn compile_error(e: bloq_compile::CompileError) -> PyErr {
    errors::CompileError::new_err(compile_error_message(&e))
}

pub(crate) const LIMIT_OVERRIDE_HELP: &str = "Retry with a larger count or None in the matching limits entry \
     (limits={\"FIELD\": count_or_None}); see https://bloqec.com/docs/dev/api/rust/bloq_graph/struct.ModuleCertificationLimits.html for fields.";

pub(crate) fn compile_error_message(error: &bloq_compile::CompileError) -> String {
    if error.resource_limit_help().is_some() {
        format!("{error}\n{LIMIT_OVERRIDE_HELP}")
    } else {
        error.to_string()
    }
}

/// Republishes a compile's in-band advisories as Python warnings.
///
/// The messages come from the compile result rather than being re-derived from
/// the source graph, so Python sees exactly what the compiler recorded — and
/// only for a compile that actually succeeded.
///
/// `stacklevel` 1 attributes the warning to the caller's frame: an extension
/// function pushes no Python frame of its own, so the innermost Python frame at
/// this point is already the `bloq.compile(...)` call site.
fn warn_compile(py: Python<'_>, warnings: &[&'static str]) -> PyResult<()> {
    if warnings.is_empty() {
        return Ok(());
    }
    let category = py.get_type::<errors::CompileWarning>();
    for message in warnings {
        let message = CString::new(*message).map_err(errors::invalid_argument)?;
        PyErr::warn(py, &category, &message, 1)?;
    }
    Ok(())
}

/// Builds a validated [`CompileConfig`]. An out-of-range distance is a bad
/// argument (a code distance must be odd and in `3..=255`), so it raises
/// `InvalidArgumentError` rather than surfacing deep in the compile pipeline.
fn resolve_config(
    distance: u32,
    prepare_t_with_mpps: bool,
    limits: Option<HashMap<String, Option<usize>>>,
) -> PyResult<CompileConfig> {
    let limits = resolve_limits(limits)?;
    CompileConfig::try_new(distance)
        .map(|config| {
            config
                .with_prepare_t_with_mpps(prepare_t_with_mpps)
                .with_certification_limits(limits)
        })
        .map_err(errors::invalid_argument)
}

pub(crate) fn resolve_limits(
    overrides: Option<HashMap<String, Option<usize>>>,
) -> PyResult<bloq_graph::ModuleCertificationLimits> {
    let mut limits = bloq_graph::ModuleCertificationLimits::DEFAULT;
    for (field, count) in overrides.into_iter().flatten() {
        limits
            .set(&field, count.unwrap_or(usize::MAX))
            .map_err(errors::invalid_argument)?;
    }
    Ok(limits)
}

/// A reusable compilation context with a fixed configuration and a private
/// template cache. The free `compile` function instead shares a process-global
/// cache; use a context when its cache lifetime should follow one Python object.
///
/// `compile` releases the GIL and takes a shared receiver, so one
/// context can serve several threads at once — the template cache is internally
/// synchronized.
///
/// Examples:
///     >>> import bloq
///     >>> context = bloq.CompileContext(distance=3)
///     >>> program = context.compile(bloq.GalleryItem.X_MEMORY.load())
///     >>> program.metadata[bloq.ir.CODE_DISTANCE_METADATA_KEY]
///     3
// `frozen` and lock-free: `bloq_compile::CompileContext::compile` takes `&self`
// and the type is `Sync`, so neither a Mutex nor PyO3's `&mut self` borrow
// flags (which would reject a concurrent caller with `RuntimeError` rather
// than making it wait) are needed.
#[gen_stub_pyclass]
#[pyclass(name = "CompileContext", module = "bloq._core", frozen)]
pub(crate) struct PyCompileContext {
    context: CompileContext,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCompileContext {
    /// Creates a context for the given code `distance`.
    ///
    /// Args:
    ///     distance: The odd code distance to compile at, in `3..=255`.
    ///     prepare_t_with_mpps: Use compact T-source preparation with one
    ///         stabilizer `MPP` instead of MSC-LS cultivation.
    ///     limits: Compiler budget overrides keyed by `ModuleCertificationLimits`
    ///         field name. Non-negative counts impose a budget, `None` disables
    ///         it, and omitted fields keep their defaults.
    ///
    /// Raises:
    ///     InvalidArgumentError: If the distance or a limit field is invalid.
    ///     OverflowError: If a limit count is negative or exceeds `usize`.
    #[new]
    #[pyo3(signature = (distance=3, *, prepare_t_with_mpps=false, limits=None))]
    fn new(
        distance: u32,
        prepare_t_with_mpps: bool,
        limits: Option<HashMap<String, Option<usize>>>,
    ) -> PyResult<Self> {
        let config = resolve_config(distance, prepare_t_with_mpps, limits)?;
        Ok(Self {
            context: CompileContext::new(config),
        })
    }

    /// The fixed code distance used by this context.
    #[getter]
    fn distance(&self) -> u32 {
        self.context.config().code_distance()
    }

    /// Whether this context uses compact T-source MPP preparation.
    #[getter]
    fn prepare_t_with_mpps(&self) -> bool {
        self.context.config().prepare_t_with_mpps()
    }

    /// Compiles `graph` into a Bloq IR program.
    ///
    /// Args:
    ///     graph: The block graph to compile.
    ///     validate: Explicitly audit the resulting IR, using this context's
    ///         Boolean limits. Defaults to `False`; compilation always performs
    ///         its required local checks.
    ///
    /// Raises:
    ///     CompileError: If compilation fails.
    ///
    /// Emits:
    ///     CompileWarning: Once per advisory the compile recorded — currently
    ///         a spatial Hadamard pipe under the fixed-bulk convention, whose
    ///         effective distance can be lower than the requested code
    ///         distance.
    ///
    /// Examples:
    ///     >>> import bloq
    ///     >>> context = bloq.CompileContext(distance=3)
    ///     >>> context.compile(bloq.GalleryItem.X_MEMORY.load())
    ///     <Bloq nodes=2 edges=1 qubits=17>
    ///     >>> context.compile(bloq.GalleryItem.CNOT.load())  # reuses cached templates
    ///     <Bloq nodes=... edges=... qubits=65>
    #[pyo3(signature = (graph, *, validate=false))]
    fn compile(
        &self,
        py: Python<'_>,
        graph: PyRef<'_, PyBlockGraph>,
        validate: bool,
    ) -> PyResult<PyBloq> {
        // The compile touches no Python object: `graph` stays borrowed for the
        // whole call, so Python cannot mutate it meanwhile.
        let source = &graph.0;
        let artifacts = py
            .detach(move || {
                if validate {
                    self.context.compile_and_validate(source)
                } else {
                    self.context.compile(source)
                }
            })
            .map_err(compile_error)?;
        warn_compile(py, &artifacts.warnings)?;
        Ok(PyBloq::from(artifacts.bloq))
    }

    fn __repr__(&self) -> String {
        let config = self.context.config();
        format!("<CompileContext distance={}>", config.code_distance())
    }
}

/// Circuit templates shared by every `bloq.compile` call in the process.
///
/// The free function creates a context per call and attaches it to this cache.
/// Templates are immutable and keyed by compile configuration and block signature,
/// so sharing does not change results. Misses compile outside the lock: distinct
/// signatures run concurrently, while simultaneous cold misses for one signature
/// may duplicate work before keeping one result.
///
/// Clifford proxies use a separate compiler-owned cache; `clear_compile_cache`
/// clears both caches.
///
/// The cache is never evicted, so it grows with every distinct distance,
/// convention, and block signature the process compiles — a distance sweep
/// keeps every distance's templates resident. `clear_compile_cache` gives that
/// memory back.
static COMPILE_CACHE: std::sync::OnceLock<SharedCompileCache> = std::sync::OnceLock::new();

/// Drops templates, module artifacts, and summaries cached by `compile` and
/// Clifford proxy compilation, releasing their memory.
///
/// `compile` caches templates process-wide and never evicts them, so a process
/// that compiles at many code distances — a distance sweep, say — keeps every
/// distance's templates resident. Call this between phases of such a run to
/// give the memory back; the next `compile` simply refills the cache.
///
/// Compilations already in flight on other threads are unaffected.
///
/// Examples:
///     >>> import bloq
///     >>> bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     <Bloq nodes=2 edges=1 qubits=17>
///     >>> bloq.clear_compile_cache()
///     >>> bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)  # recompiles
///     <Bloq nodes=2 edges=1 qubits=17>
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
fn clear_compile_cache() {
    if let Some(cache) = COMPILE_CACHE.get() {
        cache.clear();
    }
    bloq_compile::clear_clifford_proxy_cache();
}

/// Compiles `graph` into a Bloq IR program with a one-shot context. Circuit
/// templates are still cached process-wide, so repeated calls at one distance
/// stay warm.
///
/// Use `CompileContext` when one object should own the configuration and cache
/// lifetime instead of sharing the process-global cache.
///
/// Args:
///     graph: The block graph to compile.
///     distance: The odd code distance to compile at, in `3..=255`.
///     prepare_t_with_mpps: Use compact T-source preparation with one
///         stabilizer `MPP` instead of MSC-LS cultivation.
///     validate: Explicitly audit the compiled IR using the configured Boolean
///         limits. Defaults to `False`.
///     limits: Compiler budget overrides keyed by field name. Non-negative
///         counts impose a budget, `None` disables it, and omitted fields
///         keep their defaults. See the field list at
///         https://bloqec.com/docs/dev/api/rust/bloq_graph/struct.ModuleCertificationLimits.html.
///
/// Raises:
///     CompileError: If compilation fails.
///     InvalidArgumentError: If the distance or a limit field is invalid.
///     OverflowError: If a limit count is negative or exceeds `usize`.
///
/// Emits:
///     CompileWarning: Once per advisory the compile recorded — currently a
///         spatial Hadamard pipe under the fixed-bulk convention, whose
///         effective distance can be lower than the requested code distance.
///
/// Examples:
///     >>> import bloq
///     >>> graph = bloq.GalleryItem.X_MEMORY.load()
///     >>> bloq.compile(graph, distance=3)
///     <Bloq nodes=2 edges=1 qubits=17>
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (graph, distance=3, *, prepare_t_with_mpps=false, validate=false, limits=None))]
fn compile(
    py: Python<'_>,
    graph: PyRef<'_, PyBlockGraph>,
    distance: u32,
    prepare_t_with_mpps: bool,
    validate: bool,
    limits: Option<HashMap<String, Option<usize>>>,
) -> PyResult<PyBloq> {
    let config = resolve_config(distance, prepare_t_with_mpps, limits)?;
    let cache = COMPILE_CACHE.get_or_init(SharedCompileCache::new);
    PyCompileContext {
        context: CompileContext::with_shared_cache(config, cache),
    }
    .compile(py, graph, validate)
}

/// Compiles `graph` with every dynamic element replaced by a Clifford
/// stand-in — a distance oracle, not a semantics oracle. Each selective block
/// is statically pinned to one measurement arm by the corresponding `pins`
/// bit (deterministic block order, one bit per selective site); T blocks
/// become perfect input ports. The result suits Stim distance searches such
/// as `shortest_graphlike_error`.
///
/// `pins` carries one bit per selective site in deterministic block order;
/// pass an empty list for a graph with no selective blocks.
///
/// Both this function and `compile_random_clifford_proxy` accept simple and
/// hierarchical block graphs. The detector-slice proxy remains an editor view
/// and retains T regions, so it is not a Stim distance-search circuit.
///
/// Args:
///     graph: The block graph to compile.
///     pins: One bit per selective site in deterministic block order, pinning
///         each to a measurement arm; empty when the graph has no selective
///         blocks.
///     distance: The odd code distance to compile at, in `3..=255`.
///     prepare_t_with_mpps: Use compact T-source preparation with one
///         stabilizer `MPP` instead of MSC-LS cultivation.
///     limits: Compiler budget overrides; non-negative counts impose a budget,
///         `None` disables it, and omitted fields keep their defaults.
///
/// Raises:
///     CompileError: If compilation fails.
///     InvalidArgumentError: If the distance or a limit field is invalid.
///     OverflowError: If a limit count is negative or exceeds `usize`.
///
/// Emits:
///     CompileWarning: Once per advisory the compile recorded — currently a
///         spatial Hadamard pipe under the fixed-bulk convention, whose
///         effective distance can be lower than the requested code distance.
///
/// Examples:
///     >>> import bloq
///     >>> graph = bloq.GalleryItem.X_MEMORY.load()
///     >>> bloq.compile_clifford_proxy(graph, [], distance=3).quantum_node_count
///     1
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (graph, pins, distance=3, *, prepare_t_with_mpps=false, limits=None))]
fn compile_clifford_proxy(
    py: Python<'_>,
    graph: PyRef<'_, PyBlockGraph>,
    pins: Vec<bool>,
    distance: u32,
    prepare_t_with_mpps: bool,
    limits: Option<HashMap<String, Option<usize>>>,
) -> PyResult<PyBloq> {
    let config = resolve_config(distance, prepare_t_with_mpps, limits)?;
    // Release the GIL during compilation; distance sweeps call this in a loop.
    let source = &graph.0;
    let artifacts = py
        .detach(move || bloq_compile::compile_clifford_proxy(config, source, &pins))
        .map_err(compile_error)?;
    warn_compile(py, &artifacts.warnings)?;
    Ok(PyBloq::from(artifacts.bloq))
}

/// Compiles one random static Clifford proxy path. Shared selective controls
/// are sampled jointly; the seed is stored under
/// `CLIFFORD_PROXY_SEED_METADATA_KEY`.
///
/// Args:
///     graph: The block graph to compile.
///     distance: The odd code distance to compile at, in `3..=255`.
///     prepare_t_with_mpps: T preparation option; proxy T blocks remain ports.
///     seed: Reproducible path seed; `None` generates one.
///     limits: Compiler budget overrides; non-negative counts impose a budget,
///         `None` disables it, and omitted fields keep their defaults.
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (graph, distance=3, *, prepare_t_with_mpps=false, seed=None, limits=None))]
fn compile_random_clifford_proxy(
    py: Python<'_>,
    graph: PyRef<'_, PyBlockGraph>,
    distance: u32,
    prepare_t_with_mpps: bool,
    seed: Option<u64>,
    limits: Option<HashMap<String, Option<usize>>>,
) -> PyResult<PyBloq> {
    let config = resolve_config(distance, prepare_t_with_mpps, limits)?;
    let seed = seed.unwrap_or_else(rand::random);
    let source = &graph.0;
    let artifacts = py
        .detach(move || bloq_compile::compile_random_clifford_proxy(config, source, seed))
        .map_err(compile_error)?;
    warn_compile(py, &artifacts.warnings)?;
    Ok(PyBloq::from(artifacts.bloq))
}

/// One node's chunk of a segmented Stim program.
///
/// `text` is the node's Stim text; `measurement_start` is the absolute column
/// of its first measurement in the whole program, and `measurement_count` how
/// many it contributes — so this segment owns program columns
/// `measurement_start .. measurement_start + measurement_count`.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> segment = bloq.emit_stim_segments(program).segments[0]
///     >>> segment.measurement_start
///     0
///     >>> segment.measurement_count > 0
///     True
#[gen_stub_pyclass]
#[pyclass(
    name = "StimSegment",
    module = "bloq._core",
    frozen,
    eq,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PyStimSegment {
    /// The id of the node this chunk was emitted from.
    node_id: u32,
    /// The node's Stim text.
    text: String,
    /// Absolute column of this chunk's first measurement.
    measurement_start: usize,
    /// How many measurement columns this chunk contributes.
    measurement_count: usize,
}

impl From<&bloq_stim::StimSegment> for PyStimSegment {
    fn from(segment: &bloq_stim::StimSegment) -> Self {
        Self {
            node_id: segment.node_id.0,
            text: segment.text.clone(),
            measurement_start: segment.measurement_start,
            measurement_count: segment.measurement_count,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStimSegment {
    fn __repr__(&self) -> String {
        format!(
            "<StimSegment node={} measurements={}..{}>",
            self.node_id,
            self.measurement_start,
            self.measurement_start + self.measurement_count
        )
    }
}

/// A Stim program linearized into per-node chunks, for callers transforming
/// nodes in isolation (chiefly noise injection that must leave port `MPP`
/// boundaries noiseless). Plain emission wants `emit_stim`.
///
/// The `header` holds the shared `QUBIT_COORDS` preamble; `segments` holds one
/// `StimSegment` per node in emission order. `to_text()` reassembles the raw
/// circuit text; `stim.Circuit(segments.to_text())` matches `emit_stim(program)`.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> segments = bloq.emit_stim_segments(program)
///     >>> "QUBIT_COORDS" in segments.header
///     True
///     >>> len(segments.segments) >= 1
///     True
///     >>> import stim
///     >>> stim.Circuit(segments.to_text()) == bloq.emit_stim(program)
///     True
#[gen_stub_pyclass]
#[pyclass(name = "StimSegments", module = "bloq._core", frozen)]
pub(crate) struct PyStimSegments(bloq_stim::BloqStimSegments);

#[gen_stub_pymethods]
#[pymethods]
impl PyStimSegments {
    /// `QUBIT_COORDS` preamble shared by every node.
    #[getter]
    fn header(&self) -> String {
        self.0.header.clone()
    }

    /// Per-node chunks in emission order.
    #[getter]
    fn segments(&self) -> Vec<PyStimSegment> {
        self.0.segments.iter().map(Into::into).collect()
    }

    /// The total number of measurement columns across every chunk.
    #[getter]
    fn num_measurements(&self) -> usize {
        self.0.num_measurements()
    }

    /// The header and every chunk concatenated — the whole-program Stim text.
    fn to_text(&self) -> String {
        self.0.to_text()
    }

    fn __len__(&self) -> usize {
        self.0.segments.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<StimSegments segments={} measurements={}>",
            self.0.segments.len(),
            self.0.num_measurements()
        )
    }
}

/// Detector, frame, GAP, frontier, and probe columns for one isolated-T
/// attempt artifact.
///
/// The `frontier_*` lists are parallel. Frontier entry `i` maps to synthetic
/// detector `base_companion_detector_count + i` in the causal DEM transform.
/// The `exp_val_signs` list is ordered X, Y, Z. Each corrected frame XORs its
/// measurement parity, `frame_*_sign`, and the associated GAP prediction.
#[gen_stub_pyclass]
#[pyclass(
    name = "IsolatedTAttemptManifest",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyIsolatedTAttemptManifest {
    /// `(region_id, body_selector)` hops to the emitted RUS body.
    rus_path: Vec<(u32, String)>,
    /// Expected noiseless parity of every detector, in emitted order.
    detector_signs: Vec<bool>,
    /// Detector columns used for physical restart postselection.
    postselection_detectors: Vec<u32>,
    /// Logical-X GAP observable id.
    gap_x_observable: u32,
    /// Logical-Z GAP observable id.
    gap_z_observable: u32,
    /// Measurement columns XORed into the output X-frame bit.
    frame_x_measurements: Vec<u32>,
    /// Constant XORed with the X-frame measurement parity before decoder correction.
    frame_x_sign: bool,
    /// GAP observable whose correction contributes to the X-frame bit.
    frame_x_gap_observable: u32,
    /// Measurement columns XORed into the output Z-frame bit.
    frame_z_measurements: Vec<u32>,
    /// Constant XORed with the Z-frame measurement parity before decoder correction.
    frame_z_sign: bool,
    /// GAP observable whose correction contributes to the Z-frame bit.
    frame_z_gap_observable: u32,
    /// Frontier observable ids, parallel with `frontier_bases` and
    /// `frontier_signs`.
    frontier_observables: Vec<u32>,
    /// Frontier stabilizer bases.
    frontier_bases: Vec<PyBasis>,
    /// Expected frontier parities.
    frontier_signs: Vec<bool>,
    /// Static signs multiplying direct X/Y/Z `EXP_VAL` probes.
    exp_val_signs: [i8; 3],
}

impl From<bloq_stim::IsolatedTAttemptManifest> for PyIsolatedTAttemptManifest {
    fn from(manifest: bloq_stim::IsolatedTAttemptManifest) -> Self {
        let mut frontier_observables = Vec::with_capacity(manifest.frontier_sheets.len());
        let mut frontier_bases = Vec::with_capacity(manifest.frontier_sheets.len());
        let mut frontier_signs = Vec::with_capacity(manifest.frontier_sheets.len());
        for sheet in manifest.frontier_sheets {
            frontier_observables.push(sheet.observable);
            frontier_bases.push(sheet.basis.into());
            frontier_signs.push(sheet.sign);
        }
        Self {
            rus_path: manifest
                .rus_path
                .segments()
                .iter()
                .map(|segment| (segment.region.0, segment.body.name().to_owned()))
                .collect(),
            detector_signs: manifest.detector_signs,
            postselection_detectors: manifest.postselection_detectors,
            gap_x_observable: manifest.gap_x_observable,
            gap_z_observable: manifest.gap_z_observable,
            frame_x_measurements: manifest.frame_x.measurements,
            frame_x_sign: manifest.frame_x.sign,
            frame_x_gap_observable: manifest.frame_x.gap_observable,
            frame_z_measurements: manifest.frame_z.measurements,
            frame_z_sign: manifest.frame_z.sign,
            frame_z_gap_observable: manifest.frame_z.gap_observable,
            frontier_observables,
            frontier_bases,
            frontier_signs,
            exp_val_signs: manifest.exp_val_signs,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyIsolatedTAttemptManifest {
    fn __repr__(&self) -> String {
        format!(
            "<IsolatedTAttemptManifest detectors={} postselection={} frontiers={}>",
            self.detector_signs.len(),
            self.postselection_detectors.len(),
            self.frontier_observables.len()
        )
    }
}

/// Clifft physical circuits and Stim causal companions for one isolated-T
/// source attempt.
#[gen_stub_pyclass]
#[pyclass(
    name = "IsolatedTAttemptArtifacts",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyIsolatedTAttemptArtifacts {
    /// Full-noise S-control Clifft circuit with direct X/Y/Z probes.
    physical_s: String,
    /// Full-noise honest-T Clifft circuit with direct X/Y/Z probes.
    physical_t: String,
    /// Partially-noiseless S-proxy Stim companion carrying the GAP pair.
    companion: String,
    /// Terminal stabilizer frontier sheets, as a suffix to `companion`.
    sheets: String,
    /// Dense-column alignment data shared by all four circuits.
    manifest: PyIsolatedTAttemptManifest,
}

impl From<bloq_stim::IsolatedTAttemptArtifacts> for PyIsolatedTAttemptArtifacts {
    fn from(artifacts: bloq_stim::IsolatedTAttemptArtifacts) -> Self {
        Self {
            physical_s: artifacts.physical_s,
            physical_t: artifacts.physical_t,
            companion: artifacts.companion,
            sheets: artifacts.sheets,
            manifest: artifacts.manifest.into(),
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyIsolatedTAttemptArtifacts {
    fn __repr__(&self) -> String {
        // The four circuits run to tens of thousands of characters each, so the
        // repr reports their combined size rather than any of the texts.
        let chars = self.physical_s.len()
            + self.physical_t.len()
            + self.companion.len()
            + self.sheets.len();
        format!(
            "<IsolatedTAttemptArtifacts chars={} detectors={}>",
            chars,
            self.manifest.detector_signs.len()
        )
    }
}

/// Emits `program` as a `stim.Circuit`, ready for sampling or saving with
/// `circuit.to_file(path)`. Use `str(circuit)` when circuit text is needed.
///
/// Args:
///     program: The compiled Bloq IR program to emit.
///     noise: Uniform probability for gate, reset, measurement, and idle
///         noise, in the inclusive range `[0, 1]`. ``None`` (the default)
///         emits the noiseless circuit.
///     validate: Check the program's well-formedness before emitting. Defaults
///         to ``False``, which suits a program that came straight from
///         :func:`compile` in this process. Pass ``True`` for one loaded with
///         :meth:`Bloq.from_binary` / :meth:`Bloq.load`, edited by hand, or
///         produced elsewhere: the backend assumes a well-formed program and
///         may otherwise emit a silently wrong circuit.
///     align_moments: Flatten repeats and merge compatible Clifford moments by
///         z layer. Requires noiseless emission; defaults to ``False``.
///
/// Raises:
///     InvalidArgumentError: If `noise` is not finite or outside `[0, 1]`.
///     StimEmissionError: If the static backend cannot emit the program
///         (dynamic control regions, non-Clifford gates).
///     BloqValidationError: If ``validate`` is set and the program is malformed.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> bloq.emit_stim(program).num_qubits > 0
///     True
///     >>> "DEPOLARIZE1" in str(bloq.emit_stim(program, noise=1e-3))
///     True
///     >>> bloq.emit_stim(bloq.Bloq.from_binary(program.to_binary()), validate=True) \
///     ...     == bloq.emit_stim(program)
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "stim.Circuit", imports = ("stim")))]
#[pyo3(signature = (program, *, noise = None, validate = false, align_moments = false))]
fn emit_stim(
    py: Python<'_>,
    program: &PyBloq,
    noise: Option<f64>,
    validate: bool,
    align_moments: bool,
) -> PyResult<Py<PyAny>> {
    let noise = noise.map(uniform_noise).transpose()?;
    let options = stim_options(noise.as_ref(), validate).with_align_moments(align_moments);
    let source = &program.0;
    let text = py
        .detach(move || bloq_stim::emit_bloq_stim_with(source, &options))
        .map_err(stim_error)?;
    Ok(py
        .import("stim")?
        .getattr("Circuit")?
        .call1((text,))?
        .unbind())
}

fn stim_options(noise: Option<&NoiseModel>, validate: bool) -> bloq_stim::BloqStimOptions<'_> {
    let trust = if validate {
        bloq_stim::InputTrust::Checked
    } else {
        bloq_stim::InputTrust::Trusted
    };
    let options = bloq_stim::BloqStimOptions::new().with_trust(trust);
    match noise {
        Some(noise) => options.with_noise(noise),
        None => options,
    }
}

pub(crate) fn stim_error(error: bloq_stim::StimEmissionError) -> PyErr {
    match error {
        bloq_stim::StimEmissionError::InvalidProgram(error) => {
            errors::BloqValidationError::new_err(error.to_string())
        }
        error => errors::StimEmissionError::new_err(error.to_string()),
    }
}

/// Emits `program` as per-node Stim segments (see `StimSegments`).
///
/// Args:
///     program: The compiled Bloq IR program to emit.
///     noise: Uniform probability for gate, reset, measurement, and idle
///         noise, in the inclusive range `[0, 1]`. ``None`` (the default)
///         emits the noiseless circuit.
///     validate: Check the program's well-formedness before emitting; see
///         :func:`emit_stim`. Defaults to ``False``.
///
/// Raises:
///     InvalidArgumentError: If `noise` is not finite or outside `[0, 1]`.
///     StimEmissionError: If the static backend cannot emit the program.
///     BloqValidationError: If ``validate`` is set and the program is malformed.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> segments = bloq.emit_stim_segments(program)
///     >>> segments.header.startswith("QUBIT_COORDS")
///     True
///     >>> len(segments.segments) == program.node_count
///     True
///     >>> import stim
///     >>> stim.Circuit(segments.to_text()) == bloq.emit_stim(program)
///     True
///     >>> stim.Circuit(bloq.emit_stim_segments(program, noise=1e-3).to_text()) \
///     ...     == bloq.emit_stim(program, noise=1e-3)
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (program, *, noise = None, validate = false))]
fn emit_stim_segments(
    py: Python<'_>,
    program: PyRef<'_, PyBloq>,
    noise: Option<f64>,
    validate: bool,
) -> PyResult<PyStimSegments> {
    let noise = noise.map(uniform_noise).transpose()?;
    let options = stim_options(noise.as_ref(), validate);
    let source = &program.0;
    py.detach(move || bloq_stim::emit_bloq_stim_segments_with(source, &options))
        .map(PyStimSegments)
        .map_err(stim_error)
}

/// Emits `program` twice from one pass: noiseless segments and segments with
/// uniform circuit-level noise, chunked identically.
///
/// Both results carry the same node ids and the same measurement spans, so a
/// caller comparing or splicing the two — the usual noise-injection workflow —
/// no longer has to emit twice and assert the chunkings agree.
///
/// Args:
///     program: The compiled Bloq IR program to emit.
///     noise: Uniform probability for gate, reset, measurement, and idle
///         noise, in the inclusive range `[0, 1]`.
///
/// Returns:
///     tuple[StimSegments, StimSegments]: The clean and noisy segmentations.
///
/// Raises:
///     InvalidArgumentError: If `noise` is not finite or outside `[0, 1]`.
///     StimEmissionError: If the static backend cannot emit the program.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> clean, noisy = bloq.emit_stim_segments_pair(program, noise=1e-3)
///     >>> [s.node_id for s in clean.segments] == [s.node_id for s in noisy.segments]
///     True
///     >>> import stim
///     >>> stim.Circuit(clean.to_text()) == bloq.emit_stim(program)
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (program, *, noise))]
fn emit_stim_segments_pair(
    py: Python<'_>,
    program: PyRef<'_, PyBloq>,
    noise: f64,
) -> PyResult<(PyStimSegments, PyStimSegments)> {
    let noise = uniform_noise(noise)?;
    let source = &program.0;
    py.detach(move || bloq_stim::emit_bloq_stim_segments_pair(source, &noise))
        .map(|(clean, noisy)| (PyStimSegments(clean), PyStimSegments(noisy)))
        .map_err(stim_error)
}

/// Rejects an out-of-range probability as a bad argument rather than letting
/// it surface from the backend. Shared so every probability-valued keyword in
/// the module — noise here, the decoder knobs in [`crate::vm`] — fails the
/// same way, named after the keyword the caller passed.
pub(crate) fn probability(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(errors::invalid_argument(format!(
            "{name} must be finite and in [0, 1]"
        )))
    }
}

/// Builds a uniform depolarizing model from a validated `noise` probability.
pub(crate) fn uniform_noise(noise: f64) -> PyResult<NoiseModel> {
    probability("noise", noise).map(NoiseModel::uniform_depolarizing)
}

fn remap_target(
    target: GateTarget,
    qubit_map: &HashMap<u32, u32>,
    measurement_map: Option<&HashMap<i64, i64>>,
    source_measurements: i64,
    destination_measurements: i64,
) -> Result<GateTarget, String> {
    if target.is_measurement_record_target() {
        let Some(measurement_map) = measurement_map else {
            return Ok(target);
        };
        let source_record = source_measurements
            .checked_add(i64::from(target.value()))
            .ok_or_else(|| "source measurement index overflowed".to_owned())?;
        let destination_record = measurement_map
            .get(&source_record)
            .ok_or_else(|| format!("unmapped measurement record {source_record}"))?;
        let lookback = destination_record
            .checked_sub(destination_measurements)
            .ok_or_else(|| "destination measurement lookback overflowed".to_owned())?;
        let lookback = i32::try_from(lookback)
            .map_err(|_| "destination measurement lookback does not fit i32".to_owned())?;
        return GateTarget::rec(lookback).map_err(|error| error.to_string());
    }
    let Some(qubit) = target.qubit_value() else {
        return Ok(target);
    };
    let mapped = *qubit_map
        .get(&qubit)
        .ok_or_else(|| format!("unmapped qubit {qubit}"))?;
    GateTarget::pauli(
        mapped,
        target.pauli_type(),
        target.is_inverted_result_target(),
    )
    .map_err(|error| error.to_string())
}

fn write_stim_instruction(
    output: &mut String,
    instruction: &stim::CircuitInstruction,
    targets: &[GateTarget],
    observable: Option<u64>,
    honest_t: bool,
) {
    if !output.is_empty() {
        output.push('\n');
    }
    output.push_str(instruction.name());
    let tag = if honest_t && matches!(instruction.name(), "S" | "S_DAG") {
        "HONEST_T"
    } else {
        instruction.tag()
    };
    if !tag.is_empty() {
        bloq_stim::write_stim_tag(output, tag);
    }
    let gate_args = instruction.gate_args();
    if observable.is_some() || !gate_args.is_empty() {
        output.push('(');
        if let Some(observable) = observable {
            write!(output, "{observable}").expect("writing to a String is infallible");
        } else {
            for (index, argument) in gate_args.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write!(output, "{argument}").expect("writing to a String is infallible");
            }
        }
        output.push(')');
    }
    let mut after_combiner = false;
    for target in targets {
        if !target.is_combiner() && !after_combiner {
            output.push(' ');
        }
        write!(output, "{target}").expect("writing to a String is infallible");
        after_combiner = target.is_combiner();
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the remapping inputs are independent parts of the public operation"
)]
fn remap_stim_circuit_impl(
    text: &str,
    qubit_map: &HashMap<u32, u32>,
    skip: &HashSet<String>,
    honest_t: bool,
    observable_map: Option<&HashMap<u64, Vec<u64>>>,
    mut measurement_map: Option<HashMap<i64, i64>>,
    source_measurement_start: i64,
    destination_measurement_start: i64,
) -> Result<(String, HashMap<i64, i64>), String> {
    let circuit = Circuit::from_str(text)
        .map_err(|error| error.to_string())?
        .flattened();
    let mut output = String::with_capacity(text.len());
    let mut updates = HashMap::new();
    let mut source_measurements = source_measurement_start;
    let mut destination_measurements = destination_measurement_start;
    for item in circuit {
        let CircuitItem::Instruction(instruction) = item else {
            return Err("flattened Stim circuit retained a REPEAT block".to_owned());
        };
        let produced = i64::try_from(instruction.num_measurements())
            .map_err(|_| "instruction measurement count does not fit i64".to_owned())?;
        let source_end = source_measurements
            .checked_add(produced)
            .ok_or_else(|| "source measurement index overflowed".to_owned())?;
        if skip.contains(instruction.name()) {
            source_measurements = source_end;
            continue;
        }
        let targets = instruction
            .targets()
            .iter()
            .copied()
            .map(|target| {
                remap_target(
                    target,
                    qubit_map,
                    measurement_map.as_ref(),
                    source_measurements,
                    destination_measurements,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if instruction.name() == "OBSERVABLE_INCLUDE"
            && let Some(observable_map) = observable_map
        {
            let source = instruction
                .gate_args()
                .first()
                .copied()
                .filter(|value| {
                    value.is_finite()
                        && *value >= 0.0
                        // u64::MAX rounds up to 2^64 in f64, so equality is out of range.
                        && *value < u64::MAX as f64
                        && value.fract() == 0.0
                })
                .ok_or_else(|| {
                    "OBSERVABLE_INCLUDE needs a non-negative integer index fitting u64".to_owned()
                })? as u64;
            for &destination in observable_map.get(&source).into_iter().flatten() {
                write_stim_instruction(
                    &mut output,
                    &instruction,
                    &targets,
                    Some(destination),
                    honest_t,
                );
            }
            source_measurements = source_end;
            continue;
        }
        let destination_end = destination_measurements
            .checked_add(produced)
            .ok_or_else(|| "destination measurement index overflowed".to_owned())?;
        write_stim_instruction(&mut output, &instruction, &targets, None, honest_t);
        if let Some(measurement_map) = measurement_map.as_mut() {
            for (source, destination) in
                (source_measurements..source_end).zip(destination_measurements..destination_end)
            {
                measurement_map.insert(source, destination);
                updates.insert(source, destination);
            }
        }
        source_measurements = source_end;
        destination_measurements = destination_end;
    }
    Ok((output, updates))
}

/// Flattens and remaps Stim circuit text without constructing Python target
/// objects.
///
/// Args:
///     text: A Stim circuit.
///     qubit_map: Source qubit index to destination qubit index.
///     skip: Instruction names to omit; defaults to omitting nothing. A
///         skipped instruction still advances the source measurement offset,
///         so dropping `"QUBIT_COORDS"` does not shift the record mapping.
///     honest_t: Replace tags on `S` and `S_DAG` with `HONEST_T`.
///     observable_map: Optional source observable to destination observables.
///     measurement_map: Optional absolute source-to-destination measurement map.
///     source_measurement_start: Absolute source measurement offset.
///     destination_measurement_start: Absolute destination measurement offset.
///
/// Returns:
///     tuple[str, dict[int, int]]: Remapped Stim text and newly produced
///     measurement-map entries.
///
/// Raises:
///     InvalidArgumentError: If Stim text or a requested mapping is invalid.
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    reason = "the Python signature exposes independent remapping controls"
)]
#[pyo3(signature = (
    text,
    qubit_map,
    *,
    skip=None,
    honest_t=false,
    observable_map=None,
    measurement_map=None,
    source_measurement_start=0,
    destination_measurement_start=0,
))]
fn remap_stim_circuit(
    py: Python<'_>,
    text: &str,
    qubit_map: HashMap<u32, u32>,
    skip: Option<HashSet<String>>,
    honest_t: bool,
    observable_map: Option<HashMap<u64, Vec<u64>>>,
    measurement_map: Option<HashMap<i64, i64>>,
    source_measurement_start: i64,
    destination_measurement_start: i64,
) -> PyResult<(String, HashMap<i64, i64>)> {
    let text = text.to_owned();
    let skip = skip.unwrap_or_default();
    py.detach(move || {
        remap_stim_circuit_impl(
            &text,
            &qubit_map,
            &skip,
            honest_t,
            observable_map.as_ref(),
            measurement_map,
            source_measurement_start,
            destination_measurement_start,
        )
    })
    .map_err(errors::invalid_argument)
}

/// Emits the physical S/honest-T circuits and shared causal GAP companions for
/// one isolated T-source attempt.
///
/// `program` must be the compiled isolated `T -> +Z -> Port` fixture. Insert
/// any required memory rounds with `Bloq.insert_memory_rounds_after` before
/// calling this function.
///
/// Args:
///     program: The prepared isolated-T Bloq IR program.
///     noise: Uniform circuit-level error probability in the inclusive range
///         `[0, 1]`, spelled as in :func:`emit_stim`.
///
/// Raises:
///     InvalidArgumentError: If `noise` is not finite or outside `[0, 1]`.
///     StimEmissionError: If the program is not the supported isolated-T shape,
///         or emission fails.
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (program, *, noise))]
fn emit_isolated_t_attempts(
    py: Python<'_>,
    program: PyRef<'_, PyBloq>,
    noise: f64,
) -> PyResult<PyIsolatedTAttemptArtifacts> {
    let noise = probability("noise", noise)?;
    let source = &program.0;
    py.detach(move || bloq_stim::emit_isolated_t_attempts(source, noise))
        .map(Into::into)
        .map_err(|e| errors::StimEmissionError::new_err(e.to_string()))
}

/// Compiles `graph` and emits the resulting program as a `stim.Circuit`.
///
/// The whole pipeline in one call, for the common case that never needs the
/// intermediate program. Compiles with the given distance and limits, then
/// calls `emit_stim(program, noise=noise)` with identical results.
///
/// Args:
///     graph: The block graph to compile.
///     distance: The odd code distance to compile at, in `3..=255`.
///     noise: Uniform circuit-level noise probability, as in `emit_stim`.
///     limits: Compiler budget overrides; non-negative counts impose a budget,
///         `None` disables it, and omitted fields keep their defaults.
///
/// Raises:
///     CompileError: If compilation fails.
///     InvalidArgumentError: If the distance, noise, or a limit field is invalid.
///     OverflowError: If a limit count is negative or exceeds `usize`.
///     StimEmissionError: If the static backend cannot emit the program.
///
/// Emits:
///     CompileWarning: Once per advisory the compile recorded.
///
/// Examples:
///     >>> import bloq
///     >>> circuit = bloq.compile_to_stim(bloq.GalleryItem.X_MEMORY.load())
///     >>> circuit.num_qubits > 0
///     True
///     >>> "DEPOLARIZE" in str(bloq.compile_to_stim(bloq.GalleryItem.X_MEMORY.load(), noise=0.001))
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[gen_stub(override_return_type(type_repr = "stim.Circuit", imports = ("stim")))]
#[pyo3(signature = (graph, distance=3, *, noise=None, limits=None))]
fn compile_to_stim(
    py: Python<'_>,
    graph: PyRef<'_, PyBlockGraph>,
    distance: u32,
    noise: Option<f64>,
    limits: Option<HashMap<String, Option<usize>>>,
) -> PyResult<Py<PyAny>> {
    let program = compile(py, graph, distance, false, false, limits)?;
    emit_stim(py, &program, noise, false, false)
}

/// Whether `distance` is a code distance this compiler accepts: odd and in
/// `3..=255`.
///
/// Use it to filter a sweep before compiling. Negative and arbitrarily large
/// integers return `False`; non-integers raise `TypeError`.
///
/// Args:
///     distance: The candidate code distance.
///
/// Examples:
///     >>> import bloq
///     >>> [d for d in (1, 2, 3, 4, 1000) if bloq.is_valid_distance(d)]
///     [3]
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
fn is_valid_distance(
    #[gen_stub(override_type(type_repr = "int"))] distance: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match distance.extract::<u32>() {
        Ok(distance) => Ok(CompileConfig::is_valid_distance(distance)),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyOverflowError>(distance.py()) => {
            Ok(false)
        }
        Err(error) => Err(error),
    }
}

pyo3_stub_gen::module_variable!("bloq._core", "CODE_DISTANCE_METADATA_KEY", String);
pyo3_stub_gen::module_variable!("bloq._core", "CONVENTION_METADATA_KEY", String);
pyo3_stub_gen::module_variable!("bloq._core", "CLIFFORD_PROXY_SEED_METADATA_KEY", String);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CODE_DISTANCE_METADATA_KEY", CODE_DISTANCE_METADATA_KEY)?;
    m.add("CONVENTION_METADATA_KEY", CONVENTION_METADATA_KEY)?;
    m.add(
        "CLIFFORD_PROXY_SEED_METADATA_KEY",
        CLIFFORD_PROXY_SEED_METADATA_KEY,
    )?;
    m.add_class::<PyCompileContext>()?;
    m.add_class::<PyStimSegment>()?;
    m.add_class::<PyStimSegments>()?;
    m.add_class::<PyIsolatedTAttemptManifest>()?;
    m.add_class::<PyIsolatedTAttemptArtifacts>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(compile_clifford_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(compile_random_clifford_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(compile_to_stim, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_distance, m)?)?;
    m.add_function(wrap_pyfunction!(clear_compile_cache, m)?)?;
    m.add_function(wrap_pyfunction!(emit_stim, m)?)?;
    m.add_function(wrap_pyfunction!(emit_stim_segments, m)?)?;
    m.add_function(wrap_pyfunction!(emit_stim_segments_pair, m)?)?;
    m.add_function(wrap_pyfunction!(remap_stim_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(emit_isolated_t_attempts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remapping_preserves_escaped_instruction_tags() {
        for text in [r"H[a\Cb] 0", r"H[a\Bb] 0", r"H[a\nb\rc] 0", "H[λ] 0"] {
            let (remapped, _) = remap_stim_circuit_impl(
                text,
                &HashMap::from([(0, 0)]),
                &HashSet::new(),
                false,
                None,
                None,
                0,
                0,
            )
            .unwrap();
            assert_eq!(remapped, text);
            assert_eq!(
                Circuit::from_str(&remapped).unwrap(),
                Circuit::from_str(text).unwrap()
            );
        }
    }

    #[test]
    fn remap_stim_circuit_maps_targets_records_observables_and_tags() {
        let text = "\
QUBIT_COORDS(0, 0) 0
REPEAT 2 {
    M 0
    DETECTOR rec[-1]
}
S[old] 0
MPP !X0*Y1
OBSERVABLE_INCLUDE(3) rec[-1]";
        let qubit_map = HashMap::from([(0, 4), (1, 5)]);
        let skip = HashSet::from(["QUBIT_COORDS".to_owned()]);
        let observable_map = HashMap::from([(3, vec![7, 8])]);
        let (remapped, updates) = remap_stim_circuit_impl(
            text,
            &qubit_map,
            &skip,
            true,
            Some(&observable_map),
            Some(HashMap::new()),
            10,
            20,
        )
        .expect("valid remap");

        assert_eq!(
            remapped,
            "\
M 4
DETECTOR rec[-1]
M 4
DETECTOR rec[-1]
S[HONEST_T] 4
MPP !X4*Y5
OBSERVABLE_INCLUDE(7) rec[-1]
OBSERVABLE_INCLUDE(8) rec[-1]"
        );
        assert_eq!(updates, HashMap::from([(10, 20), (11, 21), (12, 22)]));
    }
}
