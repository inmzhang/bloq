//! ZX-graph introspection bindings: read-only `ZXGraph`, `ZXNode`, and
//! `ZXEdge` views plus the module-level `to_zx_graph` conversion.
//!
//! Deliberately read-only: the ZX layer is an analysis product of a
//! `BlockGraph`, not an authoring surface. Action-level accessors
//! (`action_graph`, `measurement_column`) stay unbound; the string-keyed
//! `measurement_columns` covers Python-side needs without structured
//! `MeasureTarget` types.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use crate::errors;
use crate::graph::PyBlockGraph;
use crate::primitives::{ivec3_from, ivec3_into, py_bool};

/// ZX analysis failures are graph-level diagnoses, so they map onto the same
/// exception as other `BlockGraph` analysis errors.
fn zx_err(e: impl std::fmt::Display) -> PyErr {
    errors::BlockGraphError::new_err(e.to_string())
}

// ==============================================================================
// ZXNode / ZXEdge (read-only views)
// ==============================================================================

/// A node (spider or boundary) of a ZX diagram (read-only view).
///
/// Nodes come from `ZXGraph.nodes()` (indexable by `id`) or `ZXGraph.node_at()`.
/// Interior nodes are `"X"`, `"Y"`, or `"Z"` spiders; the boundary kinds are
/// `"Port"` (an open boundary left by a port block), `"T"` (a magic-state
/// boundary), and `"Selective(...)"` (a boundary whose basis is resolved at
/// runtime). `is_boundary` is true for all three boundary kinds.
#[gen_stub_pyclass]
#[pyclass(name = "ZXNode", module = "bloq._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PyZXNode(pub(crate) bloq_graph::ZXNode);

impl From<bloq_graph::ZXNode> for PyZXNode {
    fn from(n: bloq_graph::ZXNode) -> Self {
        PyZXNode(n)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyZXNode {
    /// The node id; also its index into `ZXGraph.nodes()`.
    #[getter]
    fn id(&self) -> usize {
        self.0.id
    }

    /// The node's `(x, y, z)` position.
    #[getter]
    fn pos(&self) -> (i32, i32, i32) {
        ivec3_into(self.0.pos)
    }

    /// The node kind: `"X"`, `"Y"`, `"Z"`, `"Port"`, `"T"`, or
    /// `"Selective(<bases>)"`.
    #[getter]
    fn kind(&self) -> String {
        self.0.kind.to_string()
    }

    /// `True` if this node is a port.
    #[getter]
    fn is_port(&self) -> bool {
        self.0.kind.is_port()
    }

    /// `True` if this node is a T node.
    #[getter]
    fn is_t(&self) -> bool {
        self.0.kind.is_t()
    }

    /// True for all boundary node kinds: Port, T, and Selective.
    #[getter]
    fn is_boundary(&self) -> bool {
        self.0.kind.is_boundary()
    }

    fn __repr__(&self) -> String {
        format!(
            "<ZXNode id={} pos={:?} kind=\"{}\">",
            self.0.id,
            ivec3_into(self.0.pos),
            self.0.kind
        )
    }
}

/// An edge of a ZX diagram (read-only view).
///
/// Edges come from `ZXGraph.edges()` or `ZXGraph.edge_between()`; each one
/// connects the two nodes `n1` / `n2` (by node id) and may carry a Hadamard.
/// Edge ids live in the same id space as node ids, following them (the space
/// that `ZXGraph.total_ids` sizes and stabilizer Pauli strings index into).
#[gen_stub_pyclass]
#[pyclass(name = "ZXEdge", module = "bloq._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct PyZXEdge(pub(crate) bloq_graph::ZXEdge);

impl From<bloq_graph::ZXEdge> for PyZXEdge {
    fn from(e: bloq_graph::ZXEdge) -> Self {
        PyZXEdge(e)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyZXEdge {
    /// The id of one endpoint node.
    #[getter]
    fn n1(&self) -> usize {
        self.0.n1
    }

    /// The id of the other endpoint node.
    #[getter]
    fn n2(&self) -> usize {
        self.0.n2
    }

    /// The edge id (edge ids follow node ids in the graph's id space).
    #[getter]
    fn id(&self) -> usize {
        self.0.id
    }

    /// True if the edge carries a Hadamard.
    #[getter]
    fn hadamard(&self) -> bool {
        self.0.hadamard
    }

    fn __repr__(&self) -> String {
        format!(
            "<ZXEdge id={} n1={} n2={} hadamard={}>",
            self.0.id,
            self.0.n1,
            self.0.n2,
            py_bool(self.0.hadamard)
        )
    }
}

// ==============================================================================
// ZXGraph
// ==============================================================================

/// The ZX diagram derived from a `BlockGraph` (read-only view).
///
/// Build one with `to_zx_graph`. The ZX layer is an analysis product of a
/// block graph, not an authoring surface: its nodes, edges, and relations
/// are queryable but immutable.
///
/// Examples:
///     >>> import bloq
///     >>> zx = bloq.to_zx_graph(bloq.GalleryItem.CNOT.load())
///     >>> zx.node_count == len(zx.nodes())
///     True
///     >>> zx.is_open
///     True
///     >>> zx.is_clifford_computation
///     True
#[gen_stub_pyclass]
#[pyclass(name = "ZXGraph", module = "bloq._core", frozen, skip_from_py_object)]
#[derive(Debug, Clone)]
pub(crate) struct PyZXGraph(pub(crate) bloq_graph::ZXGraph);

impl From<bloq_graph::ZXGraph> for PyZXGraph {
    fn from(g: bloq_graph::ZXGraph) -> Self {
        PyZXGraph(g)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyZXGraph {
    /// All nodes, indexable by node id.
    fn nodes(&self) -> Vec<PyZXNode> {
        self.0.nodes().iter().copied().map(Into::into).collect()
    }

    /// All edges.
    fn edges(&self) -> Vec<PyZXEdge> {
        self.0.edges().iter().copied().map(Into::into).collect()
    }

    /// The number of nodes.
    #[getter]
    fn node_count(&self) -> usize {
        self.0.nodes().len()
    }

    /// The number of edges.
    #[getter]
    fn edge_count(&self) -> usize {
        self.0.edges().len()
    }

    /// The size of the combined node+edge id space.
    #[getter]
    fn total_ids(&self) -> usize {
        self.0.total_ids()
    }

    /// True if the graph still has unfilled ports.
    #[getter]
    fn is_open(&self) -> bool {
        self.0.is_open()
    }

    /// True if the computation is Clifford (no T or selective nodes).
    #[getter]
    fn is_clifford_computation(&self) -> bool {
        self.0.is_clifford_computation()
    }

    /// The node at `pos`, or `None`.
    ///
    /// Args:
    ///     pos: The `(x, y, z)` position to look up.
    fn node_at(&self, pos: (i32, i32, i32)) -> Option<PyZXNode> {
        self.0.node_at(ivec3_from(pos)).copied().map(Into::into)
    }

    /// Neighbor node ids of `node_id`, or `None` for an unknown id.
    ///
    /// Args:
    ///     node_id: The node whose neighbors to return.
    fn neighbors(&self, node_id: usize) -> Option<Vec<usize>> {
        self.0.neighbors(node_id).map(<[usize]>::to_vec)
    }

    /// The id of an edge between two node ids, or `None`. The ZX layer is a
    /// multigraph: with parallel edges this returns one of them.
    ///
    /// Args:
    ///     n1: One endpoint node id.
    ///     n2: The other endpoint node id.
    fn edge_id(&self, n1: usize, n2: usize) -> Option<usize> {
        self.0.edge_id(n1, n2)
    }

    /// An edge between two node ids, or `None`. The ZX layer is a multigraph:
    /// with parallel edges this returns one of them.
    ///
    /// Args:
    ///     n1: One endpoint node id.
    ///     n2: The other endpoint node id.
    fn edge_between(&self, n1: usize, n2: usize) -> Option<PyZXEdge> {
        self.0.edge_between(n1, n2).copied().map(Into::into)
    }

    /// True if `node_id` is an output port — a Port whose neighbors all sit
    /// at a lower z-coordinate.
    ///
    /// Args:
    ///     node_id: The node to classify.
    ///
    /// Raises:
    ///     InvalidArgumentError: If `node_id` is not a valid node id.
    fn is_output_port(&self, node_id: usize) -> PyResult<bool> {
        let node = self
            .0
            .nodes()
            .get(node_id)
            .ok_or_else(|| errors::InvalidArgumentError::new_err("unknown node id"))?;
        Ok(node.is_output_port(&self.0))
    }

    /// The distinct z-coordinates that carry nodes, ascending.
    fn z_layers(&self) -> Vec<i32> {
        self.0.z_layers()
    }

    /// Output-port positions as `(x, y, z)` triples, sorted.
    ///
    /// A layout column is not a qubit: two output ports can share `(x, y)` when
    /// the second patch reuses tiles the first vacated, so ports are reported
    /// with their z-coordinate rather than collapsed to columns.
    fn output_ports(&self) -> Vec<(i32, i32, i32)> {
        self.0
            .output_ports()
            .iter()
            .map(|v| (v.x, v.y, v.z))
            .collect()
    }

    /// Measurement variable name -> column id in the combined node/edge id space.
    fn measurement_columns(&self) -> HashMap<String, usize> {
        self.0.measurement_columns()
    }

    /// Validates that the graph's actions form a runnable program.
    ///
    /// Raises:
    ///     BlockGraphError: If the actions do not form a runnable program.
    fn validate_for_program(&self) -> PyResult<()> {
        self.0.validate_for_program().map_err(zx_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "<ZXGraph nodes={} edges={} open={}>",
            self.0.nodes().len(),
            self.0.edges().len(),
            self.0.is_open()
        )
    }
}

// ==============================================================================
// Module functions and registration
// ==============================================================================

/// Converts a `BlockGraph` into its ZX diagram, a read-only `ZXGraph`.
///
/// Args:
///     graph: The block graph to lower.
///
/// Returns:
///     ZXGraph: The derived ZX diagram.
///
/// Raises:
///     BlockGraphError: If the graph does not lower to ZX (e.g. a malformed
///         pipe structure).
///
/// Examples:
///     >>> import bloq
///     >>> zx = bloq.to_zx_graph(bloq.GalleryItem.CNOT.load())
///     >>> zx.node_count > 0
///     True
///     >>> zx.is_clifford_computation
///     True
#[gen_stub_pyfunction(module = "bloq._core")]
#[pyfunction]
pub(crate) fn to_zx_graph(graph: PyRef<'_, PyBlockGraph>) -> PyResult<PyZXGraph> {
    graph.0.to_zx_graph().map(Into::into).map_err(zx_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZXNode>()?;
    m.add_class::<PyZXEdge>()?;
    m.add_class::<PyZXGraph>()?;
    m.add_function(wrap_pyfunction!(to_zx_graph, m)?)?;
    Ok(())
}
