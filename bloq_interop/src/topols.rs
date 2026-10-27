//! TopoLS-inspired bounded fusion and layered graph execution.
//!
//! This preserves graph topology through placement; it does not reproduce the
//! upstream stochastic 3D placement search.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use quizx::basic_rules::{remove_id, spider_fusion};
use quizx::graph::{EType, GraphLike, V, VType};
use quizx::phase::Phase;
use quizx::vec_graph::Graph;

use crate::spider::{Plan, causal_plan, interface, vertex_order};

/// Bounded ZX-fusion settings for the TopoLS-style planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopologyOptions {
    /// Try bounded fusion candidates before graph execution.
    ///
    /// Disabling fusion retains exact graph normalization and causal-flow checks.
    pub fusion: bool,
    /// Maximum number of distinct source ZX rows in each fusion partition.
    /// Must be positive; this does not bound physical layout rows.
    pub block_rows: usize,
}

impl Default for TopologyOptions {
    fn default() -> Self {
        Self {
            fusion: true,
            block_rows: 20,
        }
    }
}

#[derive(Clone, Copy)]
enum Fusion {
    Disabled,
    SinglePass,
    FixedPoint,
}

pub(crate) fn candidates(
    graph: Graph,
    options: TopologyOptions,
    measurement_batch_size: Option<usize>,
) -> Vec<(&'static str, Result<Plan, String>)> {
    if !options.fusion {
        return vec![(
            "unfused",
            plan(
                graph,
                Fusion::Disabled,
                options.block_rows,
                measurement_batch_size,
            ),
        )];
    }
    vec![
        (
            "bounded_single_pass",
            plan(
                graph.clone(),
                Fusion::SinglePass,
                options.block_rows,
                measurement_batch_size,
            ),
        ),
        (
            "bounded_fixed_point",
            plan(
                graph,
                Fusion::FixedPoint,
                options.block_rows,
                measurement_batch_size,
            ),
        ),
    ]
}

fn prepare(mut graph: Graph, fusion: Fusion, block_rows: usize) -> Result<Graph, String> {
    for v in graph.vertices() {
        if !matches!(graph.vertex_type(v), VType::B | VType::X | VType::Z)
            || graph.vars(v) != Default::default()
            || crate::pi_quarters(graph.phase(v)).is_none()
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
    if !matches!(fusion, Fusion::Disabled) {
        fuse(&mut graph, fusion, block_rows);
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
    Ok(graph)
}

fn fuse(graph: &mut Graph, fusion: Fusion, block_rows: usize) {
    let fixed_point = matches!(fusion, Fusion::FixedPoint);
    let mut rows: Vec<_> = graph.vertices().map(|v| graph.row(v)).collect();
    rows.sort_by(f64::total_cmp);
    rows.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let blocks: BTreeMap<_, _> = graph
        .vertices()
        .map(|v| {
            let row = rows
                .binary_search_by(|row| row.total_cmp(&graph.row(v)))
                .expect("every vertex row is in the sorted row table");
            (v, row / block_rows)
        })
        .collect();
    let inputs: BTreeSet<_> = graph.inputs().iter().copied().collect();
    let outputs: BTreeSet<_> = graph.outputs().iter().copied().collect();
    let mut vertices: Vec<_> = graph.vertices().collect();
    vertices.sort_by(|&a, &b| vertex_order(graph, a, b));
    loop {
        let before = graph.num_vertices();
        for &u in &vertices {
            if graph.vertex_data_opt(u).is_none() {
                continue;
            }
            loop {
                let mut neighbors: Vec<_> = graph.neighbors(u).collect();
                neighbors.sort_by(|&a, &b| vertex_order(graph, a, b));
                let fused = neighbors.into_iter().any(|v| {
                    if graph.degree(u) + graph.degree(v) > 6 || blocks[&u] != blocks[&v] {
                        return false;
                    }
                    if !fixed_point {
                        let zero = Phase::from(0);
                        let phases_fit = (graph.phase(u) == zero && graph.phase(v) == zero)
                            || (graph.phase(u) == zero && graph.degree(v) == 2)
                            || (graph.phase(v) == zero && graph.degree(u) == 2);
                        if graph.degree(u) >= 4 || graph.degree(v) >= 4 || !phases_fit {
                            return false;
                        }
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
                    interfaces_fit && spider_fusion(graph, u, v)
                });
                if !fused || !fixed_point {
                    break;
                }
            }
        }
        if !fixed_point {
            break;
        }
        // Exact phase sums can expose identity wires and further local fusions.
        for v in graph.vertex_vec() {
            // Keep vertex IDs stable so partition and boundary identities survive.
            remove_id(graph, v);
        }
        if graph.num_vertices() == before {
            break;
        }
    }
}

fn plan(
    graph: Graph,
    fusion: Fusion,
    block_rows: usize,
    measurement_batch_size: Option<usize>,
) -> Result<Plan, String> {
    let graph = prepare(graph, fusion, block_rows)?;
    let interface = interface(&graph)?;
    let mut vertices: Vec<_> = graph
        .vertices()
        .filter(|&v| graph.vertex_type(v) == VType::Z)
        .collect();
    vertices.sort_by(|&a, &b| vertex_order(&graph, a, b));
    let mut layers: BTreeMap<V, usize> = interface.inputs.iter().map(|&(_, v, _)| (v, 0)).collect();
    let mut queue: VecDeque<_> = layers.keys().copied().collect();
    // Layered vertices stay layered, so each component root search resumes here.
    let mut unlayered = vertices.iter();
    loop {
        while let Some(v) = queue.pop_front() {
            let next_layer = layers[&v] + 1;
            for u in graph
                .neighbors(v)
                .filter(|&u| graph.vertex_type(u) == VType::Z)
            {
                if let std::collections::btree_map::Entry::Vacant(entry) = layers.entry(u) {
                    entry.insert(next_layer);
                    queue.push_back(u);
                }
            }
        }
        let Some(&root) = unlayered.find(|v| !layers.contains_key(v)) else {
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
    causal_plan(&graph, interface, &vertices, measurement_batch_size)
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
            for block_rows in [1, 20] {
                for fusion in [Fusion::Disabled, Fusion::SinglePass, Fusion::FixedPoint] {
                    let prepared = prepare(graph.clone(), fusion, block_rows).unwrap();
                    assert!(
                        TensorF::scalar_compare(&circuit, &prepared),
                        "graph: {gates}"
                    );
                    let p = plan(graph.clone(), fusion, block_rows, None).unwrap();
                    assert_eq!(p, plan(graph.clone(), fusion, block_rows, None).unwrap());
                    assert!(
                        TensorF::scalar_compare(&circuit, &execution_graph(&p, 3)),
                        "plan: {gates}"
                    );
                }
            }
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
        for (_, p) in candidates(g, TopologyOptions::default(), None) {
            assert!(p.unwrap_err().contains("two inputs"));
        }
    }

    #[test]
    fn conditional_corrections_preserve_every_measurement_branch() {
        let circuit = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0],q[1]; t q[1]; h q[1];"
        ).unwrap();
        for fusion in [false, true] {
            for measurement_batch_size in [None, Some(1), Some(2)] {
                let candidates = candidates(
                    circuit.to_graph(),
                    TopologyOptions {
                        fusion,
                        block_rows: 1,
                    },
                    measurement_batch_size,
                );
                assert_eq!(candidates.len(), if fusion { 2 } else { 1 });
                for (_, p) in candidates {
                    crate::spider::tests::verify_branches(&p.unwrap(), &circuit);
                }
            }
        }
    }

    #[test]
    fn exact_phase_cancellation_needs_no_resources() {
        let circuit = Circuit::from_qasm(
            "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; t q[0]; t q[0]; sdg q[0];",
        )
        .unwrap();
        let p = plan(circuit.to_graph(), Fusion::FixedPoint, 20, None).unwrap();
        assert!(p.instructions.is_empty());
        assert_eq!(p.direct_wires.len(), 1);
        crate::spider::tests::verify_branches(&p, &circuit);
        let layout = crate::layout(
            &circuit,
            &crate::LayoutOptions {
                strategy: crate::Strategy::TopoLs,
                max_blocks: 3,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(layout.graph.block_count(), 3);
        assert_eq!(layout.reduction, "bounded_fixed_point");
    }
}
