//! TopoLS-inspired bounded fusion and layered graph execution.
//!
//! This preserves graph topology through placement; it does not reproduce the
//! upstream stochastic 3D placement search.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use quizx::basic_rules::spider_fusion;
use quizx::graph::{EType, GraphLike, V, VType};
use quizx::phase::Phase;
use quizx::vec_graph::Graph;

use crate::spider::{Plan, causal_plan, interface};

fn vertex_order(graph: &Graph, a: V, b: V) -> std::cmp::Ordering {
    graph
        .row(a)
        .total_cmp(&graph.row(b))
        .then_with(|| graph.qubit(a).total_cmp(&graph.qubit(b)))
        .then(a.cmp(&b))
}

fn prepare(mut graph: Graph) -> Result<Graph, String> {
    for v in graph.vertices() {
        if !matches!(graph.vertex_type(v), VType::B | VType::X | VType::Z)
            || graph.vars(v) != Default::default()
            || !matches!(*graph.phase(v).to_rational().denom(), 1 | 2 | 4)
            || !graph.row(v).is_finite()
            || !graph.qubit(v).is_finite()
        {
            return Err(format!("unsupported topology vertex {v}"));
        }
    }
    if graph
        .edges()
        .any(|(u, v, kind)| u == v || !matches!(kind, EType::N | EType::H))
    {
        return Err(
            "topology requires ordinary or Hadamard edges between distinct vertices".into(),
        );
    }
    // Identity removal also cancels adjacent H edges without losing their parity.
    quizx::simplify::id_simp(&mut graph);
    let mut rows: Vec<_> = graph.vertices().map(|v| graph.row(v)).collect();
    rows.sort_by(f64::total_cmp);
    rows.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let blocks: BTreeMap<_, _> = graph
        .vertices()
        .map(|v| {
            let row = rows
                .binary_search_by(|row| row.total_cmp(&graph.row(v)))
                .expect("every vertex row is in the sorted row table");
            (v, row / 20)
        })
        .collect();
    let inputs: BTreeSet<_> = graph.inputs().iter().copied().collect();
    let outputs: BTreeSet<_> = graph.outputs().iter().copied().collect();
    let mut vertices: Vec<_> = graph.vertices().collect();
    vertices.sort_by(|&a, &b| vertex_order(&graph, a, b));
    for u in vertices {
        if graph.vertex_data_opt(u).is_none() || graph.degree(u) >= 4 {
            continue;
        }
        let mut neighbors: Vec<_> = graph.neighbors(u).collect();
        neighbors.sort_by(|&a, &b| vertex_order(&graph, a, b));
        for v in neighbors {
            if graph.degree(v) >= 4 || blocks[&u] != blocks[&v] {
                continue;
            }
            let zero = Phase::from(0);
            let phases_fit = (graph.phase(u) == zero && graph.phase(v) == zero)
                || (graph.phase(u) == zero && graph.degree(v) == 2)
                || (graph.phase(v) == zero && graph.degree(u) == 2);
            if !phases_fit {
                continue;
            }
            // Fusing source interfaces would turn distinct qubits into one patch.
            let interfaces_fit = [&inputs, &outputs].into_iter().all(|boundaries| {
                graph
                    .neighbors(u)
                    .chain(graph.neighbors(v))
                    .filter(|w| boundaries.contains(w))
                    .count()
                    <= 1
            });
            if interfaces_fit && spider_fusion(&mut graph, u, v) {
                break;
            }
        }
    }
    graph.x_to_z();
    let ordinary: Vec<_> = graph
        .edges()
        .filter(|&(u, v, kind)| {
            kind == EType::N && graph.vertex_type(u) == VType::Z && graph.vertex_type(v) == VType::Z
        })
        .collect();
    for (u, v, _) in ordinary {
        graph.remove_edge(u, v);
        let middle = graph.add_vertex(VType::Z);
        graph.set_row(middle, graph.row(u).midpoint(graph.row(v)));
        graph.set_qubit(middle, graph.qubit(u).midpoint(graph.qubit(v)));
        graph.add_edge_with_type(u, middle, EType::H);
        graph.add_edge_with_type(middle, v, EType::H);
    }
    interface(&graph)?;
    Ok(graph)
}

pub(crate) fn plan(graph: Graph) -> Result<Plan, String> {
    let graph = prepare(graph)?;
    let interface = interface(&graph)?;
    let inputs: BTreeMap<_, _> = interface
        .inputs
        .iter()
        .map(|&(i, v, h)| (v, (i, h)))
        .collect();
    let mut vertices: Vec<_> = graph
        .vertices()
        .filter(|&v| graph.vertex_type(v) == VType::Z)
        .collect();
    vertices.sort_by(|&a, &b| vertex_order(&graph, a, b));
    let remaining: BTreeMap<_, BTreeSet<_>> = vertices
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
        .collect();
    let mut layers: BTreeMap<V, usize> = inputs.keys().map(|&v| (v, 0)).collect();
    let mut queue: VecDeque<_> = inputs.keys().copied().collect();
    loop {
        while let Some(v) = queue.pop_front() {
            let next_layer = layers[&v] + 1;
            for &u in &remaining[&v] {
                if let std::collections::btree_map::Entry::Vacant(entry) = layers.entry(u) {
                    entry.insert(next_layer);
                    queue.push_back(u);
                }
            }
        }
        let Some(&root) = vertices.iter().find(|v| !layers.contains_key(v)) else {
            break;
        };
        layers.insert(root, 0);
        queue.push_back(root);
    }
    vertices.sort_by(|&a, &b| {
        layers[&a]
            .cmp(&layers[&b])
            .then_with(|| vertex_order(&graph, a, b))
    });
    causal_plan(&graph, &vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spider::tests::execution_graph;
    use quizx::circuit::Circuit;
    use quizx::tensor::{CompareTensors, TensorF};

    #[test]
    fn transformed_graph_and_layered_execution_preserve_maps() {
        for gates in [
            "h q[0]; h q[0]; h q[1];",
            "cx q[0],q[1]; cx q[0],q[2]; t q[0]; tdg q[2]; h q[1]; sdg q[0];",
            "cx q[0],q[1]; cx q[1],q[0]; cx q[0],q[1]; h q[0]; z q[1];",
        ] {
            let circuit = Circuit::from_qasm(&format!(
                "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; {gates}"
            ))
            .unwrap();
            let graph: Graph = circuit.to_graph();
            let prepared = prepare(graph.clone()).unwrap();
            assert!(
                TensorF::scalar_compare(&circuit, &prepared),
                "graph: {gates}"
            );
            let p = plan(graph.clone()).unwrap();
            assert_eq!(p, plan(graph).unwrap());
            assert!(
                TensorF::scalar_compare(&circuit, &execution_graph(&p, 3)),
                "plan: {gates}"
            );
        }
    }

    #[test]
    fn rejects_aliasing_source_inputs() {
        let mut g = Graph::new();
        let a = g.add_vertex(VType::B);
        let b = g.add_vertex(VType::B);
        let v = g.add_vertex_with_phase(VType::Z, (1, 4));
        g.add_edge(a, v);
        g.add_edge(b, v);
        g.set_inputs(vec![a, b]);
        assert!(plan(g).unwrap_err().contains("two inputs"));
    }

    #[test]
    fn conditional_corrections_preserve_every_measurement_branch() {
        let circuit = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0],q[1]; t q[1]; h q[1];"
        ).unwrap();
        let p = plan(circuit.to_graph()).unwrap();
        crate::spider::tests::verify_branches(&p, &circuit);
    }
}
