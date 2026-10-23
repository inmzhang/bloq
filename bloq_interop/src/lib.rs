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
mod grid;
mod source;
mod spider;
mod topols;

use bloq_compile::{CompileArtifacts, CompileConfig, CompileContext};
use bloq_graph::BlockGraph;
use glam::IVec3;
use quizx::{circuit::Circuit, vec_graph::Graph};

pub use grid::GridOptions;
pub use topols::TopologyOptions;

/// Native graph planning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Layer the circuit topology with bounded fusion and a graph-state sweep.
    TopoLs,
    /// Reduce with QuiZX and execute the remaining spider frontier.
    SpiderLs,
}

/// Spatial reuse versus temporal extent during layout selection.
///
/// ```
/// use bloq_interop::{GridOptions, LayoutOptions, LayoutTradeoff, layout_qasm};
///
/// let options = LayoutOptions {
///     grid: Some(GridOptions { columns: 2, ..Default::default() }),
///     tradeoff: LayoutTradeoff::Depth { extra_patches: 4 },
///     ..Default::default()
/// };
/// let layout = layout_qasm("OPENQASM 2.0; qreg q[1]; t q[0];", &options)?;
/// # Ok::<(), bloq_interop::Error>(())
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum LayoutTradeoff {
    /// Reuse patch locations; minimize blocks, then footprint, then time.
    #[default]
    Compact,
    /// Minimize footprint, then time, then blocks among generated candidates.
    Area,
    /// Minimize footprint times temporal extent, then footprint, then blocks.
    SpaceTimeVolume,
    /// Minimize time, then footprint, then blocks, among bounded candidates.
    ///
    /// Also try placement with at most this many extra patch locations beyond
    /// the plan's peak live frontier. This bounds lanes, not total footprint;
    /// routing and resource columns still contribute to area. Compact placement
    /// remains a candidate. Neither placement nor candidate search is optimal.
    Depth { extra_patches: usize },
}

/// QuiZX reduction policy for the SpiderLS-style planner.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SpiderReduction {
    /// Try full simplification; restart with conservative simplification if XY gflow fails.
    #[default]
    Auto,
    /// Use conservative simplification and recompute XY gflow.
    Conservative,
    /// Require full simplification to have XY gflow; do not fall back.
    Full,
}

/// Bounds and strategy for logical layout generation.
#[derive(Debug, Clone, Copy)]
pub struct LayoutOptions {
    /// Graph planning strategy.
    pub strategy: Strategy,
    /// Patch-location budget and layout selection objective.
    pub tradeoff: LayoutTradeoff,
    /// Two-dimensional placement; `None` preserves the original single-row router.
    pub grid: Option<GridOptions>,
    /// SpiderLS-style reduction policy. TopoLS uses `topology` instead.
    pub spider_reduction: SpiderReduction,
    /// TopoLS-style bounded fusion options; unused by SpiderLS.
    pub topology: TopologyOptions,
    /// Maximum measurements per causal scheduling batch; `None` uses a whole layer.
    /// Smaller batches can reduce the live frontier, but are not an ancilla-count cap.
    pub measurement_batch_size: Option<usize>,
    /// Maximum graph temporal extent in cells, including input/output boundaries.
    /// This is a failure limit, not a target duration or an execution-time bound.
    pub max_temporal_extent: Option<u32>,
    /// Maximum number of emitted blocks, including resources and idle memory.
    /// A tall idle block occupies multiple cells but counts as one graph node.
    pub max_blocks: usize,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            strategy: Strategy::SpiderLs,
            tradeoff: LayoutTradeoff::Compact,
            grid: None,
            spider_reduction: SpiderReduction::Auto,
            topology: TopologyOptions::default(),
            measurement_batch_size: None,
            max_temporal_extent: None,
            max_blocks: 1_000_000,
        }
    }
}

impl LayoutOptions {
    /// Validate configuration before allocating graph or output artifacts.
    ///
    /// # Errors
    /// Rejects zero budgets and invalid grid pitch.
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_blocks == 0 {
            return Err(Error::Layout("max_blocks must be positive".into()));
        }
        if self.measurement_batch_size == Some(0)
            || self.topology.block_rows == 0
            || self.max_temporal_extent == Some(0)
        {
            return Err(Error::Layout(
                "batch size, topology block rows and temporal limit must be positive".into(),
            ));
        }
        if let Some(grid) = self.grid {
            grid.validate().map_err(Error::Layout)?;
        }
        Ok(())
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
    /// Why full reduction was replaced by conservative reduction.
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
    options.validate()?;
    let mut graph: Graph = circuit.to_graph();
    let build = |plan, reduction, reduction_fallback: Option<String>| {
        let place = |extra_patches| {
            let emit = |placement| {
                let (graph, input_ports, output_ports) =
                    embed::embed(&plan, options, extra_patches, placement)
                        .map_err(Error::Layout)?;
                Ok(Layout {
                    graph,
                    input_ports,
                    output_ports,
                    reduction,
                    reduction_fallback: reduction_fallback.clone(),
                })
            };
            let mut best = emit(embed::TPlacement::Fixed);
            if options.grid.is_some()
                && plan
                    .instructions
                    .iter()
                    .any(|instruction| match instruction {
                        spider::Instruction::Phase { phase, .. } => {
                            *phase.to_rational().denom() == 4
                        }
                        _ => false,
                    })
                && let Ok(candidate) = emit(embed::TPlacement::Available)
                && best.as_ref().ok().is_none_or(|current| {
                    layout_cost(&candidate, options.tradeoff)
                        < layout_cost(current, options.tradeoff)
                })
            {
                best = Ok(candidate);
            }
            best
        };
        let mut best = place(0);
        if let LayoutTradeoff::Depth { extra_patches } = options.tradeoff
            && extra_patches != 0
            && let Ok(candidate) = place(extra_patches)
            && best.as_ref().ok().is_none_or(|current| {
                layout_cost(&candidate, options.tradeoff) < layout_cost(current, options.tradeoff)
            })
        {
            best = Ok(candidate);
        }
        best
    };
    let layout = match options.strategy {
        Strategy::TopoLs => {
            let mut best: Option<Layout> = None;
            let mut failure = None;
            for (reduction, plan) in
                topols::candidates(graph, options.topology, options.measurement_batch_size)
            {
                match plan
                    .map_err(Error::Planning)
                    .and_then(|plan| build(plan, reduction, None))
                {
                    Ok(candidate) => {
                        if best.as_ref().is_none_or(|current| {
                            layout_cost(&candidate, options.tradeoff)
                                < layout_cost(current, options.tradeoff)
                        }) {
                            best = Some(candidate);
                        }
                    }
                    Err(error) => failure = Some(error),
                }
            }
            best.ok_or_else(|| failure.expect("a topology candidate was attempted"))?
        }
        Strategy::SpiderLs => {
            if options.spider_reduction == SpiderReduction::Conservative {
                quizx::simplify::flow_simp(&mut graph);
            } else {
                quizx::simplify::full_simp(&mut graph);
            }
            let reduction = if options.spider_reduction == SpiderReduction::Conservative {
                "flow"
            } else {
                "full"
            };
            match spider::plan(&graph, options.measurement_batch_size) {
                Ok(plan) => build(plan, reduction, None)?,
                Err(reason) if options.spider_reduction == SpiderReduction::Auto => {
                    // Full reduction may introduce phase gadgets without XY
                    // gflow. Replan the same exact source with causal reduction.
                    let mut conservative: Graph = circuit.to_graph();
                    quizx::simplify::flow_simp(&mut conservative);
                    let plan = spider::plan(&conservative, options.measurement_batch_size)
                        .map_err(Error::Planning)?;
                    build(plan, "flow", Some(reason))?
                }
                Err(reason) => return Err(Error::Planning(reason)),
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

fn layout_cost(layout: &Layout, tradeoff: LayoutTradeoff) -> (u128, u128, u128) {
    let (x, y, z) = layout
        .graph
        .spans()
        .expect("a layout has source boundaries");
    let [x, y, z] = [x, y, z].map(|s| (i64::from(*s.end()) - i64::from(*s.start()) + 1) as u128);
    let blocks = layout.graph.block_count() as u128;
    match tradeoff {
        LayoutTradeoff::Compact => (blocks, x * y, z),
        LayoutTradeoff::Area => (x * y, z, blocks),
        LayoutTradeoff::SpaceTimeVolume => (x * y * z, x * y, blocks),
        LayoutTradeoff::Depth { .. } => (z, x * y, blocks),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_layouts_preserve_nonlocal_channels_and_passthroughs() {
        use bloq_graph::verify::{BoundaryOrder, LogicalVerifier};

        let source = "OPENQASM 2.0; qreg q[5]; h q[0]; t q[0]; cz q[0],q[3]; h q[3]; tdg q[3]; cx q[2],q[0]; cz q[1],q[2]; h q[1]; t q[1]; cz q[3],q[1];";
        let expected = Circuit::from_qasm(source).unwrap().to_graph();
        for strategy in [Strategy::TopoLs, Strategy::SpiderLs] {
            for (columns, spacing) in [(1, 4), (2, 4), (4, 6)] {
                let layout = layout_qasm(
                    source,
                    &LayoutOptions {
                        strategy,
                        grid: Some(GridOptions {
                            columns,
                            spacing,
                            ..Default::default()
                        }),
                        measurement_batch_size: Some(2),
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(layout.input_ports.iter().any(|p| p.y >= spacing as i32));
                LogicalVerifier::with_boundaries_and_internal_measurements(
                    &layout.graph,
                    BoundaryOrder::new(layout.input_ports.clone(), layout.output_ports.clone()),
                )
                .unwrap()
                .verify(&expected, 8, 1042)
                .unwrap();
                layout
                    .compile(CompileConfig::new(25).with_prepare_t_with_mpps(true))
                    .unwrap();
            }
        }
    }

    #[test]
    fn grid_and_scheduling_limits_reject_invalid_or_exhausted_budgets() {
        let source = "OPENQASM 2.0; qreg q[2]; cz q[0],q[1];";
        let defaults = LayoutOptions::default();
        for options in [
            LayoutOptions {
                grid: Some(GridOptions {
                    columns: 0,
                    ..Default::default()
                }),
                ..defaults
            },
            LayoutOptions {
                grid: Some(GridOptions {
                    spacing: 3,
                    ..Default::default()
                }),
                ..defaults
            },
            LayoutOptions {
                grid: Some(GridOptions {
                    routing_margin: 0,
                    ..Default::default()
                }),
                ..defaults
            },
            LayoutOptions {
                grid: Some(GridOptions {
                    max_route_expansions: 1,
                    ..Default::default()
                }),
                ..defaults
            },
            LayoutOptions {
                measurement_batch_size: Some(0),
                ..defaults
            },
            LayoutOptions {
                topology: TopologyOptions {
                    block_rows: 0,
                    ..Default::default()
                },
                ..defaults
            },
            LayoutOptions {
                max_temporal_extent: Some(0),
                ..defaults
            },
        ] {
            assert!(layout_qasm(source, &options).is_err(), "{options:?}");
        }
        let options = LayoutOptions {
            grid: Some(GridOptions::default()),
            ..defaults
        };
        let layout = layout_qasm(source, &options).unwrap();
        let (_, _, z) = layout.graph.spans().unwrap();
        let extent = (*z.end() - *z.start() + 1) as u32;
        layout_qasm(
            source,
            &LayoutOptions {
                max_temporal_extent: Some(extent),
                ..options
            },
        )
        .unwrap();
        layout_qasm(
            source,
            &LayoutOptions {
                max_temporal_extent: Some(extent - 1),
                ..options
            },
        )
        .unwrap_err();
    }

    #[test]
    fn depth_search_bounds_unused_slots_and_keeps_a_compact_budget_fallback() {
        let source = "OPENQASM 2.0; qreg q[3]; cz q[1],q[2]; tdg q[2]; cx q[1],q[0]; h q[2]; tdg q[1]; cz q[1],q[2]; h q[2]; t q[1];";
        for strategy in [Strategy::TopoLs, Strategy::SpiderLs] {
            let compact = layout_qasm(
                source,
                &LayoutOptions {
                    strategy,
                    ..Default::default()
                },
            )
            .unwrap();
            let bounded = layout_qasm(
                source,
                &LayoutOptions {
                    strategy,
                    tradeoff: LayoutTradeoff::Depth {
                        extra_patches: usize::MAX,
                    },
                    max_blocks: compact.graph.block_count(),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(bounded.graph.block_count() <= compact.graph.block_count());
            let time = |l: &Layout| {
                let (_, _, z) = l.graph.spans().unwrap();
                *z.end() - *z.start()
            };
            assert!(time(&bounded) <= time(&compact));
            assert_eq!(bounded.input_ports.len(), 3);
            assert_eq!(bounded.output_ports.len(), 3);
        }
    }
}
