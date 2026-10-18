//! Applying a `stim`-crate noise model to an emitted program, chunk by chunk.
//!
use bloq_ir::{Bloq, BloqNodeId};
use rustc_hash::{FxHashMap, FxHashSet};
use stim::{Circuit, CircuitInstruction, CircuitItem, GateTarget, noise::NoiseModel};
use thiserror::Error;

use crate::bloq::emit_bloq_stim_segments;
use crate::emit::StimEmissionError;

/// Reason a program could not be emitted with `stim`-side noise.
#[derive(Debug, Error)]
pub enum StimNoiseError {
    /// The program could not be emitted in the first place.
    #[error("{0}")]
    Emission(#[from] StimEmissionError),
    /// Stim rejected one node's chunk while parsing or noising it. The node id
    /// is carried because a whole-program message would not say which chunk.
    #[error("applying noise to node {node:?}: {source}")]
    Node {
        /// Node whose emitted chunk failed.
        node: BloqNodeId,
        /// Stim parse or noise-model failure.
        #[source]
        source: stim::StimError,
    },
}

/// Emit `program` as Stim text with `noise` applied to each node's chunk
/// separately.
///
/// Distinct from [`BloqStimOptions::with_noise`], which materializes a
/// [`bloq_circuit::NoiseModel`] into each node's *instantiated circuit* before
/// emission. This one runs stim's own noise models over the *emitted text*,
/// and it does so per chunk for a reason: a node's leading and trailing `MPP`
/// boundaries are ideal, noiseless operations (proxy ports, boundary
/// observables), and stim's `noisy_circuit_skipping_mpp_boundaries` only
/// recognizes them at the ends of the circuit it is handed. Noising the whole
/// program in one call would bury each node's boundaries in the middle and
/// noise them, and it would spread idle depolarization across qubits belonging
/// to patches that are merely coexisting in time.
///
/// Chunks are reassembled afterwards, so cross-chunk `rec[-k]` targets resolve
/// exactly as they do in [`emit_bloq_stim`](crate::emit_bloq_stim).
///
/// [`BloqStimOptions::with_noise`]: crate::BloqStimOptions::with_noise
///
/// # Errors
///
/// [`StimNoiseError::Emission`] if the program does not emit, and
/// [`StimNoiseError::Node`] if stim rejects a chunk.
///
/// # Examples
///
/// ```no_run
/// # use bloq_ir::Bloq;
/// # use bloq_stim::emit_bloq_stim_with_stim_noise;
/// # use stim::noise::UniformDepolarizing;
/// # fn example(program: &Bloq) -> Result<(), Box<dyn std::error::Error>> {
/// let noise = UniformDepolarizing::new(1e-3)?;
/// let text = emit_bloq_stim_with_stim_noise(program, &noise)?;
/// # Ok(())
/// # }
/// ```
pub fn emit_bloq_stim_with_stim_noise(
    program: &Bloq,
    noise: &impl NoiseModel,
) -> Result<String, StimNoiseError> {
    let segments = emit_bloq_stim_segments(program)?;
    let coords = program
        .sorted_layout_coords()
        .map_err(StimEmissionError::from)?;
    let layout = crate::layout::coordinate_index(coords.iter().copied())?;
    let mut out = segments.header;
    for segment in segments.segments {
        if segment.text.trim().is_empty() {
            continue;
        }
        let circuit = segment
            .text
            .parse::<Circuit>()
            .map_err(|source| StimNoiseError::Node {
                node: segment.node_id,
                source,
            })?;
        // A boundary-only node — a port `MPP` plus its boundary observables and
        // a `TICK` — is the ideal noiseless boundary itself, and has no bulk to
        // noise. stim reports that as an error (its noisy window closes empty),
        // so recognize the shape up front and pass the chunk through instead of
        // reading the failure back out of a message.
        if (&circuit)
            .into_iter()
            .all(|item| is_mpp_boundary_item(&item))
        {
            out.push_str(&segment.text);
            continue;
        }
        let noisy = noise
            .noisy_circuit_skipping_mpp_boundaries(&circuit)
            .map_err(|source| StimNoiseError::Node {
                node: segment.node_id,
                source,
            })?;
        let ideal = ideal_qubits(program, segment.node_id, &layout);
        let noisy = exclude_ideal_noise(&noisy, &ideal).map_err(|source| StimNoiseError::Node {
            node: segment.node_id,
            source,
        })?;
        out.push_str(&noisy.to_string());
        out.push('\n');
    }
    Ok(out)
}

fn ideal_qubits(
    program: &Bloq,
    node: BloqNodeId,
    layout: &FxHashMap<glam::IVec2, u32>,
) -> FxHashSet<u32> {
    // Spec rule SEM-SPATIAL-PORT, applied to this emitted node segment.
    let Some(quantum) = program[node].try_quantum() else {
        return FxHashSet::default();
    };
    let mut derived = FxHashSet::default();
    let mut source = FxHashSet::default();
    for instance in &quantum.instances {
        let template = program
            .templates()
            .get(instance.template_id)
            .expect("successful emission resolved every template instance");
        let owned = if instance.provenance.is_ideal_boundary() {
            &mut derived
        } else {
            &mut source
        };
        owned.extend(
            template
                .qubits()
                .iter()
                .filter_map(|qubit| layout.get(&(*qubit + instance.offset)).copied()),
        );
    }
    derived.difference(&source).copied().collect()
}

/// Remove error groups touching ideal qubits. Measurement instructions stay in
/// place, but their flip argument becomes zero for an ideal target group.
fn exclude_ideal_noise(
    circuit: &Circuit,
    ideal: &FxHashSet<u32>,
) -> Result<Circuit, stim::StimError> {
    if ideal.is_empty() {
        return Ok(circuit.clone());
    }
    let mut clean = Circuit::new();
    for item in circuit {
        match item {
            CircuitItem::Instruction(instruction) => {
                append_clean_instruction(&mut clean, &instruction, ideal)?;
            }
            CircuitItem::RepeatBlock(block) => {
                let body = exclude_ideal_noise(block.body(), ideal)?;
                clean.append_repeat_block(block.repeat_count(), &body, block.tag())?;
            }
        }
    }
    Ok(clean)
}

fn append_clean_instruction(
    clean: &mut Circuit,
    instruction: &CircuitInstruction,
    ideal: &FxHashSet<u32>,
) -> Result<(), stim::StimError> {
    if !instruction.gate().is_noisy_gate() || !touches_ideal(instruction.targets(), ideal) {
        return clean.append_operation(instruction);
    }

    let measures = instruction.gate().produces_measurements();
    let zero_args = vec![0.0; instruction.gate_args().len()];
    for mut group in instruction.target_groups() {
        let touches_ideal = touches_ideal(&group, ideal);
        if touches_ideal && !measures {
            continue;
        }
        if instruction.name() == "MPP" {
            group = intersperse_combiners(&group);
        }
        let args = if touches_ideal {
            &zero_args
        } else {
            instruction.gate_args()
        };
        clean.append_operation(CircuitInstruction::new(
            instruction.gate(),
            group,
            args.iter().copied(),
            instruction.tag(),
        )?)?;
    }
    Ok(())
}

fn touches_ideal(targets: &[GateTarget], ideal: &FxHashSet<u32>) -> bool {
    targets
        .iter()
        .filter_map(|target| target.qubit_value())
        .any(|qubit| ideal.contains(&qubit))
}

fn intersperse_combiners(group: &[GateTarget]) -> Vec<GateTarget> {
    let mut targets = Vec::with_capacity(group.len().saturating_mul(2).saturating_sub(1));
    for (index, target) in group.iter().copied().enumerate() {
        if index != 0 {
            targets.push(GateTarget::combiner());
        }
        targets.push(target);
    }
    targets
}

/// Whether an item is one stim treats as an `MPP` boundary rather than bulk.
/// Mirrors the private predicate behind
/// `NoiseModel::noisy_circuit_skipping_mpp_boundaries`.
fn is_mpp_boundary_item(item: &CircuitItem) -> bool {
    let name = match item {
        CircuitItem::Instruction(instruction) => instruction.name(),
        CircuitItem::RepeatBlock(block) => block.name(),
    };
    matches!(
        name,
        "TICK" | "OBSERVABLE_INCLUDE" | "DETECTOR" | "MPP" | "QUBIT_COORDS" | "SHIFT_COORDS"
    )
}

#[cfg(test)]
mod tests {
    use bloq_graph::{Block, BlockGraph, BlockKind, CubeKind, Direction, Pipe, PortRole};
    use glam::IVec3;

    use super::*;

    #[test]
    fn ideal_qubits_keep_measurements_but_lose_all_error_groups() {
        let circuit: Circuit = "X_ERROR(0.1) 0 1\nDEPOLARIZE2(0.1) 0 1 1 2\nM(0.1) 0 1\nREPEAT 2 {\nX_ERROR(0.1) 0 2\n}"
            .parse()
            .unwrap();

        let clean = exclude_ideal_noise(&circuit, &FxHashSet::from_iter([0])).unwrap();
        let text = clean.to_string();

        assert_eq!(
            text,
            "X_ERROR(0.1) 1\nDEPOLARIZE2(0.1) 1 2\nM(0) 0\nM(0.1) 1\nREPEAT 2 {\n    X_ERROR(0.1) 2\n}"
        );
    }

    #[test]
    fn multiplex_port_qubits_stay_ideal_in_both_noise_paths() {
        let mut graph = BlockGraph::new();
        graph.add_block(
            Block::new(IVec3::ZERO, BlockKind::Port)
                .with_port_role(PortRole::Multiplex)
                .unwrap(),
        );
        graph.add_block(Block::new(IVec3::X, BlockKind::Cube(CubeKind::ZXZ)));
        graph.add_pipe(Pipe::new(IVec3::ZERO, Direction::XPLUS));
        let program = bloq_compile::compile(&graph, 3).expect("compile Multiplex Port");
        let coords = program.sorted_layout_coords().unwrap();
        let layout = coords
            .iter()
            .copied()
            .enumerate()
            .map(|(index, coord)| (coord, index as u32))
            .collect::<FxHashMap<_, _>>();
        let ideal = program
            .quantum_nodes()
            .flat_map(|(node, _)| ideal_qubits(&program, node, &layout))
            .collect::<FxHashSet<_>>();
        let output = program.logical_outputs()[0].x.iter().next().unwrap().0;
        assert!(ideal.contains(&layout[output]));

        let internal = bloq_circuit::NoiseModel::uniform_depolarizing(1e-3);
        let internal_text = crate::emit_bloq_stim_with(
            &program,
            &crate::BloqStimOptions::new().with_noise(&internal),
        )
        .unwrap();
        let external = stim::noise::UniformDepolarizing::new(1e-3).unwrap();
        let external_text = emit_bloq_stim_with_stim_noise(&program, &external).unwrap();

        for text in [internal_text, external_text] {
            let circuit = text.parse::<Circuit>().unwrap();
            assert_noise_free(&circuit, &ideal);
        }
    }

    fn assert_noise_free(circuit: &Circuit, ideal: &FxHashSet<u32>) {
        for item in circuit {
            match item {
                CircuitItem::Instruction(instruction) => {
                    let touches_ideal = instruction
                        .targets()
                        .iter()
                        .filter_map(|target| target.qubit_value())
                        .any(|qubit| ideal.contains(&qubit));
                    if touches_ideal && instruction.gate().is_noisy_gate() {
                        assert!(
                            instruction.gate().produces_measurements()
                                && instruction.gate_args().iter().all(|&arg| arg == 0.0),
                            "{instruction:?}"
                        );
                    }
                }
                CircuitItem::RepeatBlock(block) => assert_noise_free(block.body(), ideal),
            }
        }
    }
}
