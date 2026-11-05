//! Clifford+T compilation into Bloq layouts with shared selective measurements.
//!
//! The front end defers CNOT and CZ gates, so phases become commuting Z parities
//! over physical columns. The back end measures those parities with in-plane
//! hubs, and optionally shares them through in-place T blocks.

mod machine;
pub mod program;

pub use machine::{Built, MachineOptions, Placement, TMode};
pub use program::{Bits, Deferral, Op, PhaseTerm, Program, lower};

use bloq_graph::{BlockGraph, BlockKind};
use quizx::circuit::Circuit;

/// Circuit rewriting before lowering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reduction {
    /// Compile the source gates.
    #[default]
    Raw,
    /// Full ZX simplification, then circuit extraction, else the source.
    Zx,
}

/// Compiler settings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Circuit rewriting.
    pub reduction: Reduction,
    /// Clifford deferral.
    pub deferral: Deferral,
    /// Back-end tuning.
    pub machine: MachineOptions,
}

/// Compile a normalized circuit.
///
/// # Errors
///
/// Returns an error when the circuit or generated graph is invalid.
pub fn compile(circuit: &Circuit, options: &Options) -> Result<Built, String> {
    let program = lower(&reduce(circuit, options.reduction), options.deferral)?;
    let order: Vec<usize> = (0..program.cols).collect();
    let mut machine = machine::Machine::new(&order, options.machine);
    machine.run(&program);
    machine.finish(&program)
}

/// Rewrite a circuit; the result has the same map up to a scalar.
pub fn reduce(circuit: &Circuit, reduction: Reduction) -> Circuit {
    use quizx::extract::ToCircuit;
    match reduction {
        Reduction::Raw => circuit.clone(),
        Reduction::Zx => {
            for simplify in [quizx::simplify::full_simp, quizx::simplify::flow_simp] {
                let mut graph: quizx::vec_graph::Graph = circuit.to_graph();
                simplify(&mut graph);
                if let Ok(extracted) = graph.to_circuit() {
                    return extracted.to_basic_gates();
                }
            }
            circuit.clone()
        }
    }
}

/// Layout cost figures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Bounding box extents.
    pub extents: [i32; 3],
    /// Bounding box volume.
    pub volume: i64,
    /// Cells held by blocks.
    pub occupied: usize,
    /// T blocks.
    pub t_blocks: usize,
    /// Selective caps.
    pub selective: usize,
    /// Blocks.
    pub blocks: usize,
}

/// Measure a graph.
pub fn metrics(graph: &BlockGraph) -> Metrics {
    let mut lo = [i32::MAX; 3];
    let mut hi = [i32::MIN; 3];
    let mut occupied = 0;
    let (mut t_blocks, mut selective, mut blocks) = (0, 0, 0);
    for block in graph.blocks() {
        blocks += 1;
        match block.kind() {
            BlockKind::T => t_blocks += 1,
            BlockKind::Selective(_) => selective += 1,
            _ => {}
        }
        for offset in block.reserved_offsets() {
            let p = block.pos() + offset;
            for (axis, v) in p.to_array().into_iter().enumerate() {
                lo[axis] = lo[axis].min(v);
                hi[axis] = hi[axis].max(v);
            }
            occupied += 1;
        }
    }
    let extents = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    Metrics {
        extents,
        volume: extents.iter().map(|&e| i64::from(e)).product(),
        occupied,
        t_blocks,
        selective,
        blocks,
    }
}
