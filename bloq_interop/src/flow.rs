//! Causal XY-plane generalized flow for graph-like ZX diagrams.

use std::collections::BTreeSet;

use quizx::graph::{GraphLike, V, VType};
use quizx::linalg::Mat2;
use quizx::vec_graph::Graph;

/// Corrections applied after the X measurement following this vertex's phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Correction {
    pub vertex: V,
    pub x: Vec<V>,
    pub z: Vec<V>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Flow {
    /// Execution order; vertices in one layer have no mutual dependencies.
    pub layers: Vec<Vec<Correction>>,
}

/// Solve Odd(g(v)) intersect unmeasured = {v}, with g(v) in future minus inputs.
pub(crate) fn xy_gflow(graph: &Graph) -> Result<Flow, String> {
    let interface = crate::spider::interface(graph)?;
    let inputs: BTreeSet<_> = interface.inputs.iter().map(|&(_, v, _)| v).collect();
    let outputs: BTreeSet<_> = interface.outputs.iter().map(|&(_, v, _)| v).collect();
    let mut remaining: BTreeSet<_> = graph
        .vertices()
        .filter(|&v| graph.vertex_type(v) == VType::Z && !outputs.contains(&v))
        .collect();
    let mut future: BTreeSet<_> = outputs.difference(&inputs).copied().collect();
    let mut flow = Flow::default();
    while !remaining.is_empty() {
        // Only the live adjacency cut contributes to this layer's linear system.
        future.retain(|&v| graph.neighbors(v).any(|u| remaining.contains(&u)));
        let columns: Vec<_> = future.iter().copied().collect();
        let rows: Vec<_> = columns
            .iter()
            .flat_map(|&v| graph.neighbors(v))
            .filter(|v| remaining.contains(v))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut matrix = Mat2::build(rows.len(), columns.len(), |r, c| {
            graph.connected(rows[r], columns[c])
        });
        let mut transform = Mat2::id(rows.len());
        let rank = matrix.gauss_x(true, 3, &mut transform);
        let pivots: Vec<_> = (0..rank)
            .map(|r| {
                matrix[r]
                    .iter()
                    .position(|&bit| bit != 0)
                    .expect("nonzero echelon row has a pivot")
            })
            .collect();
        let mut layer = Vec::new();
        for (column, &vertex) in rows.iter().enumerate() {
            if (rank..rows.len()).any(|r| transform[r][column] != 0) {
                continue;
            }
            let x: Vec<_> = pivots
                .iter()
                .enumerate()
                .filter_map(|(r, &pivot)| (transform[r][column] != 0).then_some(columns[pivot]))
                .collect();
            let mut odd = BTreeSet::new();
            for &v in &x {
                for u in graph
                    .neighbors(v)
                    .filter(|&u| graph.vertex_type(u) == VType::Z)
                {
                    if !odd.insert(u) {
                        odd.remove(&u);
                    }
                }
            }
            odd.remove(&vertex);
            layer.push(Correction {
                vertex,
                x,
                z: odd.into_iter().collect(),
            });
        }
        if layer.is_empty() {
            return Err(format!(
                "no causal XY gflow for {} remaining spiders",
                remaining.len()
            ));
        }
        for correction in &layer {
            remaining.remove(&correction.vertex);
            if !inputs.contains(&correction.vertex) {
                future.insert(correction.vertex);
            }
        }
        flow.layers.push(layer);
    }
    flow.layers.reverse();
    Ok(flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::circuit::Circuit;
    use quizx::graph::EType;
    use quizx::tensor::{CompareTensors, TensorF};
    use std::collections::BTreeMap;

    fn branch(source: &Graph, flow: &Flow, outcomes: usize) -> Graph {
        let mut graph = Graph::new();
        let mapping: BTreeMap<_, _> = source
            .vertices()
            .map(|v| (v, graph.add_vertex(source.vertex_type(v))))
            .collect();
        let outputs: BTreeSet<_> = source.outputs().iter().copied().collect();
        for (u, v, edge) in source.edges() {
            if !outputs.contains(&u) && !outputs.contains(&v) {
                graph.add_edge_with_type(mapping[&u], mapping[&v], edge);
            }
        }
        let mut live = mapping.clone();
        for (index, correction) in flow.layers.iter().flatten().enumerate() {
            let v = correction.vertex;
            let measured = graph.add_vertex_with_phase(VType::Z, source.phase(v));
            graph.add_edge(live[&v], measured);
            if outcomes & (1 << index) == 0 {
                continue;
            }
            graph.add_to_phase(measured, 1);
            for (targets, kind) in [(&correction.x, VType::X), (&correction.z, VType::Z)] {
                for &target in targets {
                    let next = graph.add_vertex_with_phase(kind, 1);
                    graph.add_edge(live[&target], next);
                    live.insert(target, next);
                }
            }
        }
        for &output in source.outputs() {
            let v = source.neighbors(output).next().unwrap();
            let phased = graph.add_vertex_with_phase(VType::Z, source.phase(v));
            graph.add_edge(live[&v], phased);
            graph.add_edge_with_type(phased, mapping[&output], source.edge_type(v, output));
        }
        graph.set_inputs(source.inputs().iter().map(|v| mapping[v]).collect());
        graph.set_outputs(source.outputs().iter().map(|v| mapping[v]).collect());
        graph
    }

    #[test]
    fn mixed_clifford_t_flow_corrects_every_measurement_outcome() {
        for (full, gates) in [
            (
                true,
                "h q[0]; cx q[0],q[1]; t q[1]; h q[1]; cx q[1],q[0]; tdg q[0];",
            ),
            (
                false,
                "h q[0]; h q[1]; cx q[0],q[1]; t q[0]; tdg q[1]; h q[0]; cx q[0],q[1]; s q[1];",
            ),
        ] {
            let circuit = Circuit::from_qasm(&format!(
                "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; {gates}"
            ))
            .unwrap();
            let mut graph: Graph = circuit.to_graph();
            quizx::simplify::full_simp(&mut graph);
            if !full {
                assert!(xy_gflow(&graph).unwrap_err().contains("no causal XY gflow"));
                graph = circuit.to_graph();
                quizx::simplify::flow_simp(&mut graph);
            }
            let flow = xy_gflow(&graph).unwrap_or_else(|e| panic!("{gates}: {e}"));
            assert_eq!(flow, xy_gflow(&graph).unwrap());
            let count: usize = flow.layers.iter().map(Vec::len).sum();
            assert!(count < 10);
            for outcomes in 0..1 << count {
                assert!(
                    TensorF::scalar_compare(&circuit, &branch(&graph, &flow, outcomes)),
                    "{gates}, outcomes={outcomes:b}"
                );
            }
        }
    }

    #[test]
    fn parallel_layers_obey_future_and_input_constraints() {
        let mut graph = Graph::new();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for _ in 0..2 {
            let input = graph.add_vertex(VType::B);
            let a = graph.add_vertex(VType::Z);
            let b = graph.add_vertex(VType::Z);
            let output = graph.add_vertex(VType::B);
            graph.add_edge(input, a);
            graph.add_edge_with_type(a, b, EType::H);
            graph.add_edge(b, output);
            inputs.push(input);
            outputs.push(output);
        }
        graph.set_inputs(inputs);
        graph.set_outputs(outputs);
        let flow = xy_gflow(&graph).unwrap();
        assert_eq!(flow.layers.len(), 1);
        assert_eq!(flow.layers[0].len(), 2);
        for correction in &flow.layers[0] {
            assert_eq!(correction.x.len(), 1);
            assert!(correction.z.is_empty());
            assert!(graph.connected(correction.vertex, correction.x[0]));
        }
        graph.add_vertex(VType::Z);
        assert!(xy_gflow(&graph).unwrap_err().contains("no causal XY gflow"));
    }
}
