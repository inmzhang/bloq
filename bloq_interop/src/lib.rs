//! Native ZX compilation into Bloq block graphs.
//!
//! Graph construction preserves source boundary order and authors adaptive
//! resource actions before Bloq's correlation analysis. A layout is not a
//! verified physical execution: [`Layout::compile`] additionally lowers and
//! audits the IR at the caller's requested distance.
//!
//! ```
//! use bloq_interop::{LayoutOptions, layout_qasm};
//! use bloq_compile::CompileConfig;
//!
//! let layout = layout_qasm(
//!     "OPENQASM 2.0; qreg q[1]; t q[0];",
//!     &LayoutOptions::default(),
//! )?;
//! let program = layout.compile(CompileConfig::new(3).with_prepare_t_with_mpps(true))?;
//! assert!(!program.bloq.logical_outputs().is_empty());
//! # Ok::<(), bloq_interop::Error>(())
//! ```

mod embed;
mod flow;
mod source;
mod spider;
mod topols;

use bloq_compile::{CompileArtifacts, CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use glam::IVec3;
use quizx::{circuit::Circuit, vec_graph::Graph};

/// Native graph planning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Layer the circuit topology with bounded fusion and a graph-state sweep.
    TopoLs,
    /// Reduce with QuiZX and execute the remaining spider frontier.
    SpiderLs,
}

/// Bounds and strategy for logical layout generation.
#[derive(Debug, Clone, Copy)]
pub struct LayoutOptions {
    /// Graph planning strategy.
    pub strategy: Strategy,
    /// Maximum number of emitted blocks, including resources and idle memory.
    pub max_blocks: usize,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            strategy: Strategy::SpiderLs,
            max_blocks: 1_000_000,
        }
    }
}

/// Native source, planning, geometry, or physical compilation failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Source is not a supported exact static Clifford+T circuit.
    #[error("source: {0}")]
    Source(String),
    /// The transformed graph cannot be scheduled.
    #[error("ZX planning: {0}")]
    Planning(String),
    /// Geometry or correlation validation failed, or the layout budget expired.
    #[error("layout: {0}")]
    Layout(String),
    /// Native physical lowering or full IR validation failed.
    #[error(transparent)]
    Compile(#[from] bloq_compile::CompileError),
}

/// Generated graph with its source-ordered quantum interface.
#[derive(Debug, Clone)]
pub struct Layout {
    /// Directly constructed block graph, including adaptive actions.
    pub graph: BlockGraph,
    /// Input positions in source qubit order.
    pub input_ports: Vec<IVec3>,
    /// Output positions in source qubit order.
    pub output_ports: Vec<IVec3>,
    /// ZX reduction actually used, including the chosen bounded-fusion candidate.
    pub reduction: &'static str,
    /// Why full reduction was replaced by causal-flow-preserving reduction.
    pub reduction_fallback: Option<String>,
}

impl Layout {
    /// Lower at the requested distance and run the complete IR audit.
    ///
    /// This establishes compilation success, not execution fidelity.
    ///
    /// # Errors
    /// Returns Bloq's correlation, lifetime, lowering, or IR validation failure.
    pub fn compile(&self, config: CompileConfig) -> Result<CompileArtifacts, Error> {
        Ok(CompileContext::new(config).compile_and_validate(&self.graph)?)
    }
}

/// Construct a layout from discrete Clifford+T OpenQASM 2.
///
/// Parameterized QASM rotations must be synthesized first. The [`layout`]
/// entry point also accepts exact rational quarter-turn rotations.
///
/// # Errors
/// Rejects dynamic operations, unsupported gates, invalid geometry, and
/// exhausted block budgets. No approximate rotation synthesis is performed.
pub fn layout_qasm(text: &str, options: &LayoutOptions) -> Result<Layout, Error> {
    layout_normalized(&source::qasm(text).map_err(Error::Source)?, options)
}

/// Construct a layout from an exact static QuiZX circuit.
///
/// # Errors
/// Rejects non-Clifford+T phases, malformed operands, dynamic operations,
/// invalid geometry, and exhausted block budgets.
pub fn layout(circuit: &Circuit, options: &LayoutOptions) -> Result<Layout, Error> {
    layout_normalized(&source::normalize(circuit).map_err(Error::Source)?, options)
}

fn layout_normalized(circuit: &Circuit, options: &LayoutOptions) -> Result<Layout, Error> {
    if options.max_blocks == 0 {
        return Err(Error::Layout("max_blocks must be positive".into()));
    }
    let mut graph: Graph = circuit.to_graph();
    let build = |plan, reduction, reduction_fallback| {
        let (graph, input_ports, output_ports) =
            embed::embed(&plan, options.max_blocks).map_err(Error::Layout)?;
        Ok(Layout {
            graph,
            input_ports,
            output_ports,
            reduction,
            reduction_fallback,
        })
    };
    let layout = match options.strategy {
        Strategy::TopoLs => {
            let mut best: Option<Layout> = None;
            let mut failure = None;
            for (reduction, plan) in topols::candidates(graph) {
                match plan
                    .map_err(Error::Planning)
                    .and_then(|plan| build(plan, reduction, None))
                {
                    Ok(candidate) => {
                        if best
                            .as_ref()
                            .is_none_or(|current| layout_cost(&candidate) < layout_cost(current))
                        {
                            best = Some(candidate);
                        }
                    }
                    Err(error) => failure = Some(error),
                }
            }
            best.ok_or_else(|| failure.expect("both bounded candidates were attempted"))?
        }
        Strategy::SpiderLs => {
            quizx::simplify::full_simp(&mut graph);
            match spider::plan(&graph) {
                Ok(plan) => build(plan, "full", None)?,
                Err(reason) => {
                    // Full reduction may introduce phase gadgets without XY
                    // gflow. Replan the same exact source with causal reduction.
                    let mut conservative: Graph = circuit.to_graph();
                    quizx::simplify::flow_simp(&mut conservative);
                    let plan = spider::plan(&conservative).map_err(Error::Planning)?;
                    build(plan, "flow", Some(reason))?
                }
            }
        }
    };
    if layout.input_ports.len() != circuit.num_qubits()
        || layout.output_ports.len() != circuit.num_qubits()
    {
        return Err(Error::Layout(
            "generated interface does not match the source width".into(),
        ));
    }
    Ok(layout)
}

fn layout_cost(layout: &Layout) -> (usize, i64, i64) {
    let (x, y, z) = layout
        .graph
        .spans()
        .expect("a layout has source boundaries");
    let [x, y, z] = [x, y, z].map(|s| i64::from(*s.end()) - i64::from(*s.start()) + 1);
    (layout.graph.block_count(), x * y, z)
}
