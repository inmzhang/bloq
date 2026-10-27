//! Execute a reduced graph directly, retaining patch lifetimes and exact phases.

use std::collections::{BTreeMap, BTreeSet};

use quizx::graph::{EType, GraphLike, V, VType};
use quizx::phase::Phase;
use quizx::vec_graph::Graph;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Instruction {
    Prepare {
        patch: usize,
        input: Option<usize>,
    },
    Cz {
        patches: [usize; 2],
    },
    Phase {
        patch: usize,
        phase: Phase,
    },
    Hadamard {
        patch: usize,
    },
    /// Measure X, recording the result under the source vertex identity.
    Release {
        patch: usize,
        vertex: V,
    },
    Feedback {
        measurement: V,
        x_targets: Vec<usize>,
        z_targets: Vec<usize>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectWire {
    pub input: usize,
    pub output: usize,
    pub hadamard: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    pub instructions: Vec<Instruction>,
    /// Source output index and its final physical patch; indices preserve permutations.
    pub outputs: Vec<(usize, usize)>,
    pub direct_wires: Vec<DirectWire>,
    /// Excludes the independent passthrough wires in `direct_wires`.
    pub peak_patches: usize,
}

impl Plan {
    fn cancel_hadamards(&mut self) {
        fn flush(patch: usize, pending: &mut BTreeSet<usize>, instructions: &mut Vec<Instruction>) {
            if pending.remove(&patch) {
                instructions.push(Instruction::Hadamard { patch });
            }
        }
        let mut pending = BTreeSet::new();
        for instruction in std::mem::take(&mut self.instructions) {
            match &instruction {
                Instruction::Hadamard { patch } => {
                    if !pending.insert(*patch) {
                        pending.remove(patch);
                    }
                    continue;
                }
                Instruction::Prepare { patch, .. }
                | Instruction::Phase { patch, .. }
                | Instruction::Release { patch, .. } => {
                    flush(*patch, &mut pending, &mut self.instructions)
                }
                Instruction::Cz { patches } => {
                    for &patch in patches {
                        flush(patch, &mut pending, &mut self.instructions);
                    }
                }
                Instruction::Feedback {
                    x_targets,
                    z_targets,
                    ..
                } => {
                    for &patch in x_targets.iter().chain(z_targets) {
                        flush(patch, &mut pending, &mut self.instructions);
                    }
                }
            }
            self.instructions.push(instruction);
        }
        self.instructions.extend(
            pending
                .into_iter()
                .map(|patch| Instruction::Hadamard { patch }),
        );
    }
}

#[derive(Default)]
pub(crate) struct Interface {
    pub inputs: Vec<(usize, V, bool)>,
    pub outputs: Vec<(usize, V, bool)>,
    pub direct_wires: Vec<DirectWire>,
}

/// Validate graph-like form and preserve source boundary identity and order.
pub(crate) fn interface(graph: &Graph) -> Result<Interface, String> {
    let boundaries: BTreeSet<_> = graph
        .inputs()
        .iter()
        .chain(graph.outputs())
        .copied()
        .collect();
    if boundaries.len() != graph.inputs().len() + graph.outputs().len() {
        return Err("duplicate or overlapping input/output boundaries".into());
    }
    for v in graph.vertices() {
        match graph.vertex_type(v) {
            VType::Z => {
                if graph.vars(v) != Default::default()
                    || crate::pi_quarters(graph.phase(v)).is_none()
                {
                    return Err(format!("spider {v} has a non-static Clifford+T phase"));
                }
            }
            VType::B if boundaries.contains(&v) && graph.degree(v) == 1 => {}
            _ => {
                return Err(format!(
                    "vertex {v} is not an internal Z spider or a degree-one boundary"
                ));
            }
        }
    }
    if boundaries
        .iter()
        .any(|&v| graph.vertex_data_opt(v).is_none_or(|d| d.ty != VType::B))
    {
        return Err("interface references a missing or non-boundary vertex".into());
    }
    for (u, v, kind) in graph.edges() {
        if !matches!(kind, EType::N | EType::H) {
            return Err(format!("unsupported edge type on {u}--{v}"));
        }
        if graph.vertex_type(u) == VType::Z
            && graph.vertex_type(v) == VType::Z
            && (kind != EType::H || u == v)
        {
            return Err("internal edges must be Hadamard edges between distinct spiders".into());
        }
    }
    let mut interface = Interface::default();
    let output_indices: BTreeMap<_, _> = graph
        .outputs()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let mut input_vertices = BTreeSet::new();
    let mut direct_outputs = BTreeSet::new();
    for (input, &boundary) in graph.inputs().iter().enumerate() {
        let v = graph
            .neighbors(boundary)
            .next()
            .expect("boundary degree was validated");
        let hadamard = graph.edge_type(boundary, v) == EType::H;
        if let Some(&output) = output_indices.get(&v) {
            interface.direct_wires.push(DirectWire {
                input,
                output,
                hadamard,
            });
            direct_outputs.insert(v);
        } else {
            if graph.vertex_type(v) != VType::Z {
                return Err("an input connects to a non-output boundary".into());
            }
            if !input_vertices.insert(v) {
                return Err("two inputs share one spider".into());
            }
            interface.inputs.push((input, v, hadamard));
        }
    }
    let mut output_vertices = BTreeSet::new();
    for (output, &boundary) in graph.outputs().iter().enumerate() {
        if direct_outputs.contains(&boundary) {
            continue;
        }
        let v = graph
            .neighbors(boundary)
            .next()
            .expect("boundary degree was validated");
        if graph.vertex_type(v) != VType::Z {
            return Err("an output connects to a non-input boundary".into());
        }
        if !output_vertices.insert(v) {
            return Err("two outputs share one spider".into());
        }
        interface
            .outputs
            .push((output, v, graph.edge_type(v, boundary) == EType::H));
    }
    Ok(interface)
}

struct Frontier<'a> {
    graph: &'a Graph,
    rank: BTreeMap<V, usize>,
    remaining: BTreeMap<V, BTreeSet<V>>,
    live: BTreeMap<V, usize>,
    measured: BTreeSet<V>,
    free: BTreeSet<usize>,
    plan: Plan,
}

impl Frontier<'_> {
    fn prepare(&mut self, vertex: V, input: Option<usize>) -> Result<(), String> {
        if self.live.contains_key(&vertex) {
            return Ok(());
        }
        if self.measured.contains(&vertex) {
            return Err("flow targets a released spider".into());
        }
        let patch = self.free.pop_first().unwrap_or_else(|| {
            let patch = self.plan.peak_patches;
            self.plan.peak_patches += 1;
            patch
        });
        self.live.insert(vertex, patch);
        self.plan
            .instructions
            .push(Instruction::Prepare { patch, input });
        Ok(())
    }

    fn phase(&mut self, vertex: V) {
        let phase = self.graph.phase(vertex);
        if phase != Phase::from(0) {
            self.plan.instructions.push(Instruction::Phase {
                patch: self.live[&vertex],
                phase,
            });
        }
    }

    fn transfer(&mut self, vertex: V, x: &[V], z: &[V]) -> Result<bool, String> {
        let [target] = x else {
            return Ok(false);
        };
        if self.live.contains_key(target)
            || self.measured.contains(target)
            || !self.remaining[&vertex].contains(target)
        {
            return Ok(false);
        }
        let neighbors: BTreeSet<_> = self.remaining[target]
            .iter()
            .copied()
            .filter(|&v| v != vertex)
            .collect();
        if neighbors != z.iter().copied().collect() {
            return Ok(false);
        }
        // Finish the source's interactions before reusing its patch for the fresh target.
        let edges = self.remaining[&vertex]
            .iter()
            .copied()
            .filter(|v| v != target)
            .map(|v| (v.min(vertex), v.max(vertex)))
            .collect();
        self.entangle_edges(edges, BTreeSet::new())?;
        self.prepare(vertex, None)?;
        self.phase(vertex);
        let patch = self
            .live
            .remove(&vertex)
            .expect("transfer source was prepared");
        self.live.insert(*target, patch);
        self.remaining
            .get_mut(&vertex)
            .expect("transfer source exists")
            .remove(target);
        self.remaining
            .get_mut(target)
            .expect("transfer target exists")
            .remove(&vertex);
        self.measured.insert(vertex);
        // Corrected one-edge teleportation is H; delaying the target's other CZs
        // absorbs their Z byproducts without introducing a measurement outcome.
        self.plan.instructions.push(Instruction::Hadamard { patch });
        Ok(true)
    }

    fn entangle(&mut self, targets: &BTreeSet<V>, corrections: &BTreeSet<V>) -> Result<(), String> {
        let mut needed = corrections.clone();
        needed.extend(targets);
        let mut edges = BTreeSet::new();
        for &v in targets {
            for &u in &self.remaining[&v] {
                edges.insert((u.min(v), u.max(v)));
            }
        }
        self.entangle_edges(edges, needed)
    }

    fn entangle_edges(
        &mut self,
        edges: BTreeSet<(V, V)>,
        mut needed: BTreeSet<V>,
    ) -> Result<(), String> {
        for &(u, v) in &edges {
            needed.extend([u, v]);
        }
        let mut needed: Vec<_> = needed.into_iter().collect();
        needed.sort_by_key(|v| self.rank[v]);
        for v in needed {
            self.prepare(v, None)?;
        }
        for (u, v) in edges {
            self.remaining
                .get_mut(&u)
                .expect("edge endpoint is an internal spider")
                .remove(&v);
            self.remaining
                .get_mut(&v)
                .expect("edge endpoint is an internal spider")
                .remove(&u);
            self.plan.instructions.push(Instruction::Cz {
                patches: [self.live[&u], self.live[&v]],
            });
        }
        Ok(())
    }
}

/// Schedule a graph with causal XY measurements and explicit Pauli corrections.
/// `order` controls deterministic allocation without changing the flow constraints.
/// The batch cap splits each causal layer, allowing released patches to be reused.
pub(crate) fn causal_plan(
    graph: &Graph,
    interface: Interface,
    order: &[V],
    measurement_batch_size: Option<usize>,
) -> Result<Plan, String> {
    let flow = crate::flow::xy_gflow(graph, &interface)?;
    let mut f = Frontier {
        graph,
        rank: order.iter().enumerate().map(|(i, &v)| (v, i)).collect(),
        remaining: order
            .iter()
            .map(|&v| {
                (
                    v,
                    graph
                        .neighbors(v)
                        .filter(|&u| graph.vertex_type(u) == VType::Z)
                        .collect(),
                )
            })
            .collect(),
        live: BTreeMap::new(),
        measured: BTreeSet::new(),
        free: BTreeSet::new(),
        plan: Plan {
            direct_wires: interface.direct_wires,
            ..Plan::default()
        },
    };
    for (input, v, hadamard) in interface.inputs {
        f.prepare(v, Some(input))?;
        if hadamard {
            f.plan
                .instructions
                .push(Instruction::Hadamard { patch: f.live[&v] });
        }
    }
    for layer in flow {
        for batch in layer.chunks(measurement_batch_size.unwrap_or(layer.len())) {
            let mut measurements = Vec::with_capacity(batch.len());
            for correction in batch {
                if !f.transfer(correction.vertex, &correction.x, &correction.z)? {
                    measurements.push(correction);
                }
            }
            let mut entangle = BTreeSet::new();
            let mut corrections = BTreeSet::new();
            for correction in &measurements {
                entangle.insert(correction.vertex);
                entangle.extend(&correction.x);
                corrections.extend(&correction.z);
            }
            // Later CZs commute with every correction: X targets have no edges left,
            // and Z corrections commute with CZ without extra byproducts.
            f.entangle(&entangle, &corrections)?;
            for correction in measurements {
                let v = correction.vertex;
                f.phase(v);
                let patch = f
                    .live
                    .remove(&v)
                    .ok_or("flow measures an inactive spider")?;
                f.plan
                    .instructions
                    .push(Instruction::Release { patch, vertex: v });
                f.measured.insert(v);
                f.free.insert(patch);
                let targets = |vertices: &[V]| {
                    vertices
                        .iter()
                        .map(|v| {
                            f.live.get(v).copied().ok_or_else(|| {
                                format!("flow correction targets inactive spider {v}")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                };
                f.plan.instructions.push(Instruction::Feedback {
                    measurement: v,
                    x_targets: targets(&correction.x)?,
                    z_targets: targets(&correction.z)?,
                });
            }
        }
    }
    let outputs: BTreeSet<_> = interface.outputs.iter().map(|&(_, v, _)| v).collect();
    f.entangle(&outputs, &BTreeSet::new())?;
    if f.remaining.values().any(|neighbors| !neighbors.is_empty())
        || f.live.keys().copied().collect::<BTreeSet<_>>() != outputs
    {
        return Err("causal execution left edges or an incorrect output frontier".into());
    }
    for (output, v, hadamard) in interface.outputs {
        f.phase(v);
        let patch = f.live[&v];
        if hadamard {
            f.plan.instructions.push(Instruction::Hadamard { patch });
        }
        f.plan.outputs.push((output, patch));
    }
    f.plan.cancel_hadamards();
    Ok(f.plan)
}

/// Reduce externally with QuiZX, then execute its graph using a causal frontier.
pub(crate) fn plan(graph: &Graph, measurement_batch_size: Option<usize>) -> Result<Plan, String> {
    let interface = interface(graph)?;
    let mut vertices: Vec<_> = graph
        .vertices()
        .filter(|&v| graph.vertex_type(v) == VType::Z)
        .collect();
    vertices.sort_by(|&a, &b| vertex_order(graph, a, b));
    causal_plan(graph, interface, &vertices, measurement_batch_size)
}

/// Deterministic row, qubit, then id order shared by the causal planners.
pub(crate) fn vertex_order(graph: &Graph, a: V, b: V) -> std::cmp::Ordering {
    graph
        .row(a)
        .total_cmp(&graph.row(b))
        .then_with(|| graph.qubit(a).total_cmp(&graph.qubit(b)))
        .then(a.cmp(&b))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use quizx::circuit::Circuit;
    use quizx::tensor::{CompareTensors, TensorF};

    pub(crate) fn execution_graph(p: &Plan, qubits: usize) -> Graph {
        branch_graph(p, qubits, &BTreeSet::new())
    }

    fn branch_graph(p: &Plan, qubits: usize, outcomes: &BTreeSet<V>) -> Graph {
        let mut g = Graph::new();
        let inputs: Vec<_> = (0..qubits).map(|_| g.add_vertex(VType::B)).collect();
        let outputs: Vec<_> = (0..qubits).map(|_| g.add_vertex(VType::B)).collect();
        let mut live = BTreeMap::new();
        for instruction in &p.instructions {
            match *instruction {
                Instruction::Prepare { patch, input } => {
                    let v = g.add_vertex(VType::Z);
                    assert!(live.insert(patch, v).is_none());
                    if let Some(i) = input {
                        g.add_edge(inputs[i], v);
                    }
                }
                Instruction::Cz { patches } => {
                    let ends = patches.map(|patch| {
                        let v = g.add_vertex(VType::Z);
                        g.add_edge(live[&patch], v);
                        live.insert(patch, v);
                        v
                    });
                    g.add_edge_with_type(ends[0], ends[1], EType::H);
                }
                Instruction::Phase { patch, phase } => {
                    let v = g.add_vertex_with_phase(VType::Z, phase);
                    g.add_edge(live[&patch], v);
                    live.insert(patch, v);
                }
                Instruction::Hadamard { patch } => {
                    let v = g.add_vertex(VType::Z);
                    g.add_edge_with_type(live[&patch], v, EType::H);
                    live.insert(patch, v);
                }
                Instruction::Release { patch, vertex } => {
                    let end = live.remove(&patch).unwrap();
                    let cap =
                        g.add_vertex_with_phase(VType::Z, i64::from(outcomes.contains(&vertex)));
                    g.add_edge(end, cap);
                }
                Instruction::Feedback {
                    measurement,
                    ref x_targets,
                    ref z_targets,
                } => {
                    if outcomes.contains(&measurement) {
                        for (targets, basis) in [(x_targets, VType::X), (z_targets, VType::Z)] {
                            for &patch in targets {
                                let v = g.add_vertex_with_phase(basis, 1);
                                g.add_edge(live[&patch], v);
                                live.insert(patch, v);
                            }
                        }
                    }
                }
            }
        }
        for &(output, patch) in &p.outputs {
            g.add_edge(live[&patch], outputs[output]);
        }
        for wire in &p.direct_wires {
            g.add_edge_with_type(
                inputs[wire.input],
                outputs[wire.output],
                if wire.hadamard { EType::H } else { EType::N },
            );
        }
        g.set_inputs(inputs);
        g.set_outputs(outputs);
        g
    }

    #[test]
    fn frontier_execution_preserves_reduced_circuit_map() {
        for gates in [
            "h q[0]; h q[1]; cx q[0],q[1]; t q[0]; tdg q[1]; h q[0]; cx q[0],q[1]; s q[1];",
            "h q[0]; cx q[0],q[1]; h q[1]; t q[1]; cx q[1],q[0]; sdg q[0]; h q[0];",
            "cx q[0],q[1]; cx q[1],q[0]; cx q[0],q[1]; h q[0]; z q[1];",
            "h q[0]; cx q[0],q[1]; h q[1]; cx q[1],q[0];",
            "h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[0]; tdg q[0];",
        ] {
            let circuit = Circuit::from_qasm(&format!(
                "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; {gates}"
            ))
            .unwrap();
            let mut g: Graph = circuit.to_graph();
            quizx::simplify::flow_simp(&mut g);
            let p = plan(&g, None).unwrap();
            assert_eq!(p, plan(&g, Some(usize::MAX)).unwrap());
            for measurement_batch_size in [None, Some(1), Some(2)] {
                verify_branches(&plan(&g, measurement_batch_size).unwrap(), &circuit);
            }
        }
    }

    pub(crate) fn verify_branches(p: &Plan, circuit: &Circuit) {
        verify_graph_branches(p, &circuit.to_graph());
    }

    fn verify_graph_branches(p: &Plan, graph: &Graph) {
        assert_eq!(graph.inputs().len(), graph.outputs().len());
        let measured: Vec<_> = p
            .instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::Release { vertex, .. } => Some(*vertex),
                _ => None,
            })
            .collect();
        assert!(
            measured.len() <= 10,
            "fixture has too many measurement branches"
        );
        for mask in 0..1usize << measured.len() {
            let outcomes = measured
                .iter()
                .enumerate()
                .filter(|&(i, _)| mask & (1 << i) != 0)
                .map(|(_, &v)| v)
                .collect();
            assert!(
                TensorF::scalar_compare(graph, &branch_graph(p, graph.inputs().len(), &outcomes)),
                "outcome mask {mask}: {p:?}"
            );
        }
    }

    #[test]
    fn smaller_measurement_batches_reuse_patches_and_preserve_every_branch() {
        let mut graph = Graph::new();
        let sources: Vec<_> = (0..4)
            .map(|i| {
                graph.add_vertex_with_phase(VType::Z, if i % 2 == 0 { (1, 4) } else { (-1, 4) })
            })
            .collect();
        let targets: Vec<_> = (0..4).map(|_| graph.add_vertex(VType::Z)).collect();
        let inputs = sources
            .iter()
            .map(|&v| {
                let boundary = graph.add_vertex(VType::B);
                graph.add_edge(boundary, v);
                boundary
            })
            .collect();
        let outputs = targets
            .iter()
            .map(|&v| {
                let boundary = graph.add_vertex(VType::B);
                graph.add_edge(v, boundary);
                boundary
            })
            .collect();
        graph.set_inputs(inputs);
        graph.set_outputs(outputs);
        // Invertible adjacency with multi-target corrections in one causal layer.
        for (source, row) in sources.iter().zip([
            [true, false, false, false],
            [true, true, true, true],
            [true, true, true, false],
            [true, true, false, true],
        ]) {
            for (&target, edge) in targets.iter().zip(row) {
                if edge {
                    graph.add_edge_with_type(*source, target, EType::H);
                }
            }
        }
        let parallel = plan(&graph, None).unwrap();
        let limited = plan(&graph, Some(1)).unwrap();
        assert!(limited.peak_patches < parallel.peak_patches);
        verify_graph_branches(&parallel, &graph);
        verify_graph_branches(&limited, &graph);
    }

    #[test]
    fn boundaries_preserve_direct_hadamard_and_permutation() {
        let mut g = Graph::new();
        let vertices: Vec<_> = (0..4).map(|_| g.add_vertex(VType::B)).collect();
        g.add_edge_with_type(vertices[0], vertices[3], EType::H);
        g.add_edge(vertices[1], vertices[2]);
        g.set_inputs(vertices[..2].to_vec());
        g.set_outputs(vertices[2..].to_vec());
        let p = plan(&g, None).unwrap();
        assert_eq!(
            p.direct_wires,
            [
                DirectWire {
                    input: 0,
                    output: 1,
                    hadamard: true
                },
                DirectWire {
                    input: 1,
                    output: 0,
                    hadamard: false
                },
            ]
        );
        assert!(p.instructions.is_empty());
    }

    #[test]
    fn causal_h_transfer_matches_both_explicit_teleportation_outcomes() {
        let circuit = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; t q[0]; h q[0]; tdg q[0];",
        )
        .unwrap();
        let mut graph: Graph = circuit.to_graph();
        quizx::simplify::flow_simp(&mut graph);
        let mut expanded = Plan {
            instructions: vec![
                Instruction::Prepare {
                    patch: 0,
                    input: Some(0),
                },
                Instruction::Prepare {
                    patch: 1,
                    input: None,
                },
                Instruction::Cz { patches: [0, 1] },
                Instruction::Phase {
                    patch: 0,
                    phase: (1, 4).into(),
                },
                Instruction::Release {
                    patch: 0,
                    vertex: 0,
                },
                Instruction::Hadamard { patch: 1 },
                Instruction::Feedback {
                    measurement: 0,
                    x_targets: vec![],
                    z_targets: vec![1],
                },
                Instruction::Hadamard { patch: 1 },
                Instruction::Phase {
                    patch: 1,
                    phase: (-1, 4).into(),
                },
            ],
            outputs: vec![(0, 1)],
            peak_patches: 2,
            ..Plan::default()
        };
        let contracted = plan(&graph, None).unwrap();
        assert_eq!(contracted.peak_patches, 1);
        assert!(
            !contracted
                .instructions
                .iter()
                .any(|i| matches!(i, Instruction::Release { .. }))
        );
        verify_branches(&expanded, &circuit);
        expanded.cancel_hadamards();
        verify_branches(&expanded, &circuit);
        verify_branches(&contracted, &circuit);
    }

    #[test]
    fn entangled_transfers_match_all_branches_in_either_source_order() {
        let circuit = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; cz q[0],q[1]; t q[0]; tdg q[1]; h q[0]; h q[1]; cz q[0],q[1];",
        )
        .unwrap();
        for reverse in [false, true] {
            let mut graph = Graph::new();
            let mut sources = [graph.add_vertex(VType::Z), graph.add_vertex(VType::Z)];
            if reverse {
                sources.reverse();
            }
            let targets = [graph.add_vertex(VType::Z), graph.add_vertex(VType::Z)];
            let [v, w] = sources;
            let [u, t] = targets;
            for (i, phase) in [(0, Phase::from((1, 4))), (1, Phase::from((-1, 4)))] {
                graph.set_phase(sources[i], phase);
            }
            let inputs: Vec<_> = sources
                .iter()
                .map(|&v| {
                    let boundary = graph.add_vertex(VType::B);
                    graph.add_edge(boundary, v);
                    boundary
                })
                .collect();
            let outputs: Vec<_> = targets
                .iter()
                .map(|&v| {
                    let boundary = graph.add_vertex(VType::B);
                    graph.add_edge(v, boundary);
                    boundary
                })
                .collect();
            graph.set_inputs(inputs);
            graph.set_outputs(outputs);
            for (a, b) in [(v, u), (w, t), (v, w), (u, t)] {
                graph.add_edge_with_type(a, b, EType::H);
            }
            let mut expanded = Plan {
                outputs: vec![(0, 2), (1, 3)],
                peak_patches: 4,
                ..Plan::default()
            };
            for patch in 0..4 {
                expanded.instructions.push(Instruction::Prepare {
                    patch,
                    input: (patch < 2).then_some(patch),
                });
            }
            for patches in [[0, 2], [1, 3], [0, 1], [2, 3]] {
                expanded.instructions.push(Instruction::Cz { patches });
            }
            for patch in if reverse { [1, 0] } else { [0, 1] } {
                let vertex = sources[patch];
                expanded.instructions.extend([
                    Instruction::Phase {
                        patch,
                        phase: graph.phase(vertex),
                    },
                    Instruction::Release { patch, vertex },
                    Instruction::Feedback {
                        measurement: vertex,
                        x_targets: vec![patch + 2],
                        z_targets: vec![3 - patch],
                    },
                ]);
            }
            verify_branches(&expanded, &circuit);
            let contracted = plan(&graph, None).unwrap();
            assert_eq!(contracted.peak_patches, 2);
            assert_eq!(
                contracted
                    .instructions
                    .iter()
                    .filter(|i| matches!(i, Instruction::Cz { .. }))
                    .count(),
                2
            );
            for measurement_batch_size in [None, Some(1), Some(2)] {
                verify_branches(&plan(&graph, measurement_batch_size).unwrap(), &circuit);
            }
        }
    }

    #[test]
    fn independent_patch_operations_do_not_obstruct_h_cancellation() {
        let circuit =
            Circuit::from_qasm("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; z q[1];").unwrap();
        let mut p = Plan {
            instructions: vec![
                Instruction::Prepare {
                    patch: 0,
                    input: Some(0),
                },
                Instruction::Prepare {
                    patch: 1,
                    input: Some(1),
                },
                Instruction::Hadamard { patch: 0 },
                Instruction::Phase {
                    patch: 1,
                    phase: 1.into(),
                },
                Instruction::Hadamard { patch: 0 },
            ],
            outputs: vec![(0, 0), (1, 1)],
            peak_patches: 2,
            ..Plan::default()
        };
        p.cancel_hadamards();
        assert_eq!(p.instructions.len(), 3);
        verify_branches(&p, &circuit);
    }

    #[test]
    fn shared_inputs_and_non_graphlike_edges_are_rejected() {
        let mut g = Graph::new();
        let a = g.add_vertex(VType::Z);
        let b = g.add_vertex(VType::Z);
        g.add_edge(a, b);
        assert!(plan(&g, None).unwrap_err().contains("Hadamard"));
        g.remove_edge(a, b);
        let i = g.add_vertex(VType::B);
        let j = g.add_vertex(VType::B);
        g.add_edge(i, a);
        g.add_edge(j, a);
        g.set_inputs(vec![i, j]);
        assert!(plan(&g, None).unwrap_err().contains("two inputs"));
    }
}
