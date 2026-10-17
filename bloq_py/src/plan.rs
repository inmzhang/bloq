//! Emission-plan lowering: one node's `EmissionPlan` to Stim text in either
//! dialect, plus the text codecs that translate between them.
//!
//! This is the piece a Python backend needs that `emit_stim` cannot give it:
//! `emit_stim` lowers a whole program with the compiler's own qubit layout,
//! while `emit_plan_stim` lowers *one node* against a caller-chosen qubit map
//! and hands back the post-unroll measurement column of every instance-space
//! measurement — the coordinates a caller needs to stitch nodes into a larger
//! circuit of its own.

use std::collections::HashMap;

use bloq_stim::{HONEST_T_TAG, PlanStimOptions, StimDialect};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rustc_hash::FxHashMap;

use crate::circuit::PyEmissionPlan;
use crate::errors;

fn parse_dialect(name: &str) -> PyResult<StimDialect> {
    if name.eq_ignore_ascii_case("stim") {
        Ok(StimDialect::Stim)
    } else if name.eq_ignore_ascii_case("clifft") {
        Ok(StimDialect::Clifft)
    } else {
        Err(errors::InvalidArgumentError::new_err(format!(
            "unknown Stim dialect {name:?}: expected \"stim\" or \"clifft\""
        )))
    }
}

/// One lowered emission plan: its Stim text plus where its measurements landed.
///
/// `measurement_columns` maps each instance-space measurement — the
/// `(instance_id, template_measurement_id)` keys of `EmissionPlan.measurements`
/// — to its column in `text` *after* every `REPEAT` block is unrolled, counting
/// from 0 within this plan. Add the plan's own offset to place it in a larger
/// circuit. `measurement_count` is how many columns `text` produces in total,
/// which is that same offset for the next plan.
#[gen_stub_pyclass]
#[pyclass(
    name = "PlanStim",
    module = "bloq._core",
    frozen,
    get_all,
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub(crate) struct PyPlanStim {
    /// The lowered Stim text.
    text: String,
    /// `(instance_id, template_measurement_id)` → post-unroll column.
    measurement_columns: HashMap<(u32, u32), u32>,
    /// Total measurement columns the text produces.
    measurement_count: u32,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPlanStim {
    fn __repr__(&self) -> String {
        format!(
            "<PlanStim measurements={} chars={}>",
            self.measurement_count,
            self.text.len()
        )
    }
}

/// Lowers one node's emission plan to Stim text over a caller-chosen layout.
///
/// Args:
///     plan: The node's `EmissionPlan` (from `Bloq.emission_plan`).
///     qubit_map: Every `(x, y)` layout coordinate the plan touches, mapped to
///         its Stim qubit index. A coordinate the plan uses but the map omits
///         is an error, so the usual source is the whole program's layout:
///         `{c: i for i, c in enumerate(program.sorted_layout_coords())}`.
///     dialect: `"stim"` (default) or `"clifft"`, case-insensitive. Stim uses
///         `S`/`S_DAG` tagged `HONEST_T` for T gates; Clifft keeps literal
///         `T`/`T_DAG` gates.
///     tag: Optional Stim instruction tag applied to the emitted gates.
///
/// Raises:
///     InvalidArgumentError: If the dialect name is unknown.
///     StimEmissionError: If the plan cannot be lowered — most often a
///         coordinate missing from `qubit_map`.
///
/// Examples:
///     >>> import bloq
///     >>> program = bloq.compile(bloq.GalleryItem.X_MEMORY.load(), distance=3)
///     >>> layout = {c: i for i, c in enumerate(program.sorted_layout_coords())}
///     >>> node = program.deterministic_emit_order()[0]
///     >>> lowered = bloq.emit_plan_stim(program.emission_plan(node), layout)
///     >>> lowered.measurement_count > 0
///     True
///     >>> max(lowered.measurement_columns.values()) < lowered.measurement_count
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
#[pyo3(signature = (plan, qubit_map, *, dialect="stim", tag=None))]
pub(crate) fn emit_plan_stim(
    py: Python<'_>,
    plan: PyRef<'_, PyEmissionPlan>,
    qubit_map: HashMap<(i32, i32), u32>,
    dialect: &str,
    tag: Option<String>,
) -> PyResult<PyPlanStim> {
    let dialect = parse_dialect(dialect)?;
    let qubit_map: FxHashMap<_, _> = qubit_map
        .into_iter()
        .map(|(coord, index)| (coord.into(), index))
        .collect();
    let source = plan.plan();
    let lowered = py
        .detach(move || {
            // `PlanStimOptions` borrows both the map and the tag, so they are
            // built here and the options value lives only for this call.
            let mut options = PlanStimOptions::new(&qubit_map).with_dialect(dialect);
            if let Some(tag) = tag.as_deref() {
                options = options.with_tag(tag);
            }
            bloq_stim::emit_plan_stim(source, &options)
        })
        .map_err(crate::compile::stim_error)?;
    Ok(PyPlanStim {
        text: lowered.text,
        measurement_columns: lowered
            .measurement_columns
            .into_iter()
            .map(|(key, column)| ((key.instance.0, key.measurement), column))
            .collect(),
        measurement_count: lowered.measurement_count,
    })
}

/// Rewrites Clifft text into Stim text: every `T` / `T_DAG` becomes `S` /
/// `S_DAG` tagged `HONEST_T_TAG`.
///
/// Lossy in one direction: `EXP_VAL` lines have no Stim spelling and are
/// dropped. `stim_to_clifft_text` inverts the gate mapping, not the drop.
///
/// Args:
///     text: Clifft circuit text.
///
/// Examples:
///     >>> import bloq
///     >>> bloq.clifft_to_stim_text("T 0")
///     'S[HONEST_T] 0'
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
fn clifft_to_stim_text(text: &str) -> String {
    bloq_stim::clifft_to_stim_text(text)
}

/// Rewrites Stim text into Clifft text: every `S` / `S_DAG` tagged
/// `HONEST_T_TAG` becomes `T` / `T_DAG`, and untagged gates are left alone.
///
/// Args:
///     text: Stim circuit text.
///
/// Examples:
///     >>> import bloq
///     >>> bloq.stim_to_clifft_text("S[HONEST_T] 0")
///     'T 0'
///     >>> bloq.stim_to_clifft_text("S 0")
///     'S 0'
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
fn stim_to_clifft_text(text: &str) -> String {
    bloq_stim::stim_to_clifft_text(text)
}

pyo3_stub_gen::module_variable!("bloq._core", "HONEST_T_TAG", String);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HONEST_T_TAG", HONEST_T_TAG)?;
    m.add_class::<PyPlanStim>()?;
    m.add_function(wrap_pyfunction!(emit_plan_stim, m)?)?;
    m.add_function(wrap_pyfunction!(clifft_to_stim_text, m)?)?;
    m.add_function(wrap_pyfunction!(stim_to_clifft_text, m)?)?;
    Ok(())
}
