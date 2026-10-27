//! Causal XY-plane generalized flow for graph-like ZX diagrams.

use std::collections::BTreeSet;

use quizx::graph::{GraphLike, V, VType};
use quizx::linalg::{Mat2, RowOps};
use quizx::vec_graph::Graph;

use crate::spider::Interface;

/// Corrections applied after the X measurement following this vertex's phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Correction {
    pub vertex: V,
    pub x: Vec<V>,
    pub z: Vec<V>,
}

/// Correction layers in execution order; vertices in one layer have no mutual dependencies.
pub(crate) type Flow = Vec<Vec<Correction>>;

// Keep the same QuiZX row operations while packing their identity transform.
struct PackedTransform {
    rows: Vec<Vec<u64>>,
}

impl PackedTransform {
    fn identity(width: usize) -> Self {
        let mut rows = vec![vec![0; width.div_ceil(64)]; width];
        for (index, row) in rows.iter_mut().enumerate() {
            row[index / 64] = 1 << (index % 64);
        }
        Self { rows }
    }

    fn bit(&self, row: usize, column: usize) -> bool {
        assert!(column < self.rows.len());
        self.rows[row][column / 64] & (1 << (column % 64)) != 0
    }
}

impl RowOps for PackedTransform {
    fn row_add(&mut self, source: usize, target: usize) {
        if source == target {
            self.rows[target].fill(0);
            return;
        }
        let [source, target] = self
            .rows
            .get_disjoint_mut([source, target])
            .expect("distinct transform rows");
        for (target, source) in target.iter_mut().zip(source) {
            *target ^= *source;
        }
    }

    fn row_swap(&mut self, source: usize, target: usize) {
        self.rows.swap(source, target);
    }
}

/// Solve Odd(g(v)) intersect unmeasured = {v}, with g(v) in future minus inputs.
pub(crate) fn xy_gflow(graph: &Graph, interface: &Interface) -> Result<Flow, String> {
    let inputs: BTreeSet<_> = interface.inputs.iter().map(|&(_, v, _)| v).collect();
    let outputs: BTreeSet<_> = interface.outputs.iter().map(|&(_, v, _)| v).collect();
    let mut remaining: BTreeSet<_> = graph
        .vertices()
        .filter(|&v| graph.vertex_type(v) == VType::Z && !outputs.contains(&v))
        .collect();
    let mut future: BTreeSet<_> = outputs.difference(&inputs).copied().collect();
    let mut flow = Flow::new();
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
        let mut transform = PackedTransform::identity(rows.len());
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
            if (rank..rows.len()).any(|r| transform.bit(r, column)) {
                continue;
            }
            let x: Vec<_> = pivots
                .iter()
                .enumerate()
                .filter_map(|(r, &pivot)| (transform.bit(r, column)).then_some(columns[pivot]))
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
        // A nonempty subset preserves gflow and can expose single-target corrections next.
        if layer.iter().any(|correction| correction.x.len() == 1) {
            layer.retain(|correction| correction.x.len() == 1);
        }
        for correction in &layer {
            remaining.remove(&correction.vertex);
            if !inputs.contains(&correction.vertex) {
                future.insert(correction.vertex);
            }
        }
        flow.push(layer);
    }
    flow.reverse();
    Ok(flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::circuit::Circuit;
    use quizx::graph::EType;
    use quizx::tensor::{CompareTensors, TensorF};
    use std::collections::BTreeMap;

    fn gflow(graph: &Graph) -> Result<Flow, String> {
        xy_gflow(graph, &crate::spider::interface(graph)?)
    }

    #[test]
    fn packed_transform_matches_byte_row_operations_and_elimination() {
        for size in [0, 1, 2, 3, 63, 64, 65, 127, 128, 129] {
            let mut packed = PackedTransform::identity(size);
            let mut bytes = Mat2::id(size);
            for row in 0..size {
                let target = (row + 1) % size;
                packed.row_add(row, target);
                bytes.row_add(row, target);
                packed.row_swap(row, size - row - 1);
                bytes.row_swap(row, size - row - 1);
            }
            if size != 0 {
                packed.row_add(0, 0);
                bytes.row_add(0, 0);
            }
            for row in 0..size {
                for column in 0..size {
                    assert_eq!(packed.bit(row, column), bytes[row][column] != 0);
                }
            }
        }
        for rows in 0..=3 {
            for columns in 0..=3 {
                for bits in 0_u64..1 << (rows * columns) {
                    let original =
                        Mat2::build(rows, columns, |r, c| bits & (1 << (r * columns + c)) != 0);
                    let mut expected = original.clone();
                    let mut actual = original.clone();
                    let mut bytes = Mat2::id(rows);
                    let mut packed = PackedTransform::identity(rows);
                    assert_eq!(
                        actual.gauss_x(true, 3, &mut packed),
                        expected.gauss_x(true, 3, &mut bytes),
                    );
                    assert_eq!(actual, expected);
                    for r in 0..rows {
                        for c in 0..rows {
                            assert_eq!(packed.bit(r, c), bytes[r][c] != 0);
                        }
                        for c in 0..columns {
                            let product = (0..rows).fold(false, |value, k| {
                                value ^ (packed.bit(r, k) && original[k][c] != 0)
                            });
                            assert_eq!(product, actual[r][c] != 0);
                        }
                    }
                }
            }
        }
    }

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
        for (index, correction) in flow.iter().flatten().enumerate() {
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
                assert!(gflow(&graph).unwrap_err().contains("no causal XY gflow"));
                graph = circuit.to_graph();
                quizx::simplify::flow_simp(&mut graph);
            }
            let flow = gflow(&graph).unwrap_or_else(|e| panic!("{gates}: {e}"));
            assert_eq!(flow, gflow(&graph).unwrap());
            let count: usize = flow.iter().map(Vec::len).sum();
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
    fn deferring_multi_target_corrections_preserves_every_branch() {
        let mut graph = Graph::new();
        let inputs: Vec<_> = (0..2).map(|_| graph.add_vertex(VType::B)).collect();
        let v = graph.add_vertex_with_phase(VType::Z, (1, 4));
        let w = graph.add_vertex_with_phase(VType::Z, (-1, 4));
        let a = graph.add_vertex(VType::Z);
        let b = graph.add_vertex(VType::Z);
        let outputs: Vec<_> = (0..2).map(|_| graph.add_vertex(VType::B)).collect();
        for (input, vertex) in inputs.iter().zip([v, w]) {
            graph.add_edge(*input, vertex);
        }
        for (vertex, output) in [a, b].into_iter().zip(&outputs) {
            graph.add_edge(vertex, *output);
        }
        for (source, target) in [(v, a), (v, b), (w, b)] {
            graph.add_edge_with_type(source, target, EType::H);
        }
        graph.set_inputs(inputs);
        graph.set_outputs(outputs);
        // The maximal first backward layer has g(v)={a}, g(w)={a,b}.
        let flow = gflow(&graph).unwrap();
        assert_eq!(flow.len(), 2);
        for layer in &flow {
            assert_eq!(layer.len(), 1);
            assert_eq!(layer[0].x.len(), 1);
        }
        for outcomes in 0..4 {
            assert!(
                TensorF::scalar_compare(&graph, &branch(&graph, &flow, outcomes)),
                "outcomes={outcomes:b}"
            );
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
        let flow = gflow(&graph).unwrap();
        assert_eq!(flow.len(), 1);
        assert_eq!(flow[0].len(), 2);
        for correction in &flow[0] {
            assert_eq!(correction.x.len(), 1);
            assert!(correction.z.is_empty());
            assert!(graph.connected(correction.vertex, correction.x[0]));
        }
        graph.add_vertex(VType::Z);
        assert!(gflow(&graph).unwrap_err().contains("no causal XY gflow"));
    }
}
