# Portions adapted from PyZX gflow.py and pauliweb.py.
# Copyright (C) 2018, 2024 - Aleks Kissinger and John van de Wetering
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for Bloq: incremental elimination and checked flow certificates.

"""Certified focused Pauli flow for the pinned Topologiq/PyZX integration.

PyZX rebuilds and solves the same growing binary matrix for every vertex and
layer. Keeping its column span makes large Clifford+T inputs practical without
dropping the Pauli webs Topologiq uses for causal placement.

The graph expansion and web construction follow PyZX 0.10.7's pauliweb.py;
the flow algorithm follows its gflow.py (Apache-2.0, copyright 2018/2024
Aleks Kissinger and John van de Wetering).
"""

from importlib.metadata import version
from typing import Any

from pyzx.pauliweb import PauliWeb
from pyzx.utils import (
    EdgeType,
    VertexType,
    phase_is_clifford,
    phase_is_pauli,
    vertex_is_zx,
)
from topologiq.input.zx_manager import AugmentedZXGraph


class _UnitSpan:
    """Track which unit vectors lie in an incrementally growing column span."""

    def __init__(self, rows: int):
        self.basis: dict[int, tuple[int, int]] = {}
        self.residuals = [1 << row for row in range(rows)]
        self.witnesses = [0] * rows
        self.pending = set(range(rows))

    def add(self, column: int, witness: int) -> None:
        for pivot, (value, bits) in sorted(self.basis.items()):
            if column & (1 << pivot):
                column ^= value
                witness ^= bits
        if not column:
            return
        pivot = (column & -column).bit_length() - 1
        self.basis[pivot] = column, witness
        solved = []
        for row in self.pending:
            if self.residuals[row] & (1 << pivot):
                self.residuals[row] ^= column
                self.witnesses[row] ^= witness
                if not self.residuals[row]:
                    solved.append(row)
        self.pending.difference_update(solved)


def _focused_gflow(graph: Any, *, backwards: bool) -> Any:
    """Use PyZX's focused XY/X/Y flow rules and check every correction witness."""
    types = graph.types()
    vertices = {v for v in graph.vertices() if vertex_is_zx(types[v])}
    inputs = {
        v for boundary in graph.inputs() for v in graph.neighbors(boundary)
        if vertex_is_zx(types[v])
    }
    outputs = {
        v for boundary in graph.outputs() for v in graph.neighbors(boundary)
        if vertex_is_zx(types[v])
    }
    if backwards:
        inputs, outputs = outputs, inputs
    pauli_x, pauli_y = set(), set()
    for vertex in vertices:
        phase = graph.phase(vertex) % 2
        if phase_is_pauli(phase):
            pauli_x.add(vertex)
        elif phase_is_clifford(phase):
            pauli_y.add(vertex)

    processed = outputs | graph.grounds()
    clean = list(vertices - outputs)
    row_of = {vertex: row for row, vertex in enumerate(clean)}
    columns = list(vertices)
    column_of = {vertex: column for column, vertex in enumerate(columns)}
    neighbors = {vertex: set(graph.neighbors(vertex)) for vertex in vertices}
    span = _UnitSpan(len(clean))
    known: set[int] = set()
    layers = {vertex: 0 for vertex in processed}
    flow: dict[int, set[int]] = {}
    layer = 1

    while True:
        candidates = (processed | pauli_x | pauli_y) - inputs
        for vertex in candidates - known:
            column = 0
            for neighbor in neighbors[vertex]:
                if neighbor in row_of:
                    column |= 1 << row_of[neighbor]
            if vertex in pauli_y and vertex in row_of:
                column |= 1 << row_of[vertex]
            span.add(column, 1 << column_of[vertex])
            known.add(vertex)

        correct = set()
        for row, vertex in enumerate(clean):
            if vertex in processed or row in span.pending:
                continue
            if not (
                (vertex in pauli_y and vertex in candidates)
                or neighbors[vertex] & candidates
            ):
                continue
            bits = span.witnesses[row]
            correction = set()
            while bits:
                low = bits & -bits
                correction.add(columns[low.bit_length() - 1])
                bits ^= low

            # Check the actual graph independently of the elimination state.
            if not correction <= candidates:
                raise ValueError("Focused-flow certificate uses an unavailable correction")
            for other in clean:
                parity = len(neighbors[other] & correction)
                parity += other in pauli_y and other in correction
                if parity % 2 != (other == vertex):
                    raise ValueError("Focused-flow certificate violates a parity equation")
            flow[vertex] = correction
            layers[vertex] = layer
            correct.add(vertex)

        if not correct:
            if len(vertices) == len(processed):
                order = {
                    vertex: depth if backwards else layer - depth - 1
                    for vertex, depth in layers.items()
                }
                return order, flow
            raise ValueError("Graph must have gFlow")
        processed.update(correct)
        layer += 1


def _compute_pauli_webs(graph: Any) -> Any:
    """Build native PyZX webs using a checked focused-flow certificate."""
    expanded = graph.clone()
    color_edges: dict[int, tuple[int, int]] = {}
    boundaries: dict[int, int] = {}
    for edge in graph.edges():
        start, end = graph.edge_st(edge)
        start_type, end_type = graph.type(start), graph.type(end)
        edge_type = graph.edge_type(edge)
        if not (vertex_is_zx(start_type) and vertex_is_zx(end_type)):
            continue
        if (start_type == end_type and edge_type == EdgeType.SIMPLE) or (
            start_type != end_type and edge_type == EdgeType.HADAMARD
        ):
            opposite = VertexType.Z if start_type == VertexType.X else VertexType.X
            middle = expanded.add_vertex(opposite)
            expanded.remove_edge(edge)
            expanded.add_edge((start, middle), EdgeType.SIMPLE)
            expanded.add_edge((middle, end), edge_type)
            color_edges[middle] = start, end

    for boundary in expanded.inputs() + expanded.outputs():
        edge = next(iter(expanded.incident_edges(boundary)))
        neighbor = next(iter(expanded.neighbors(boundary)))
        neighbor_type, edge_type = expanded.type(neighbor), expanded.edge_type(edge)
        expanded.remove_edge(edge)
        first_type = VertexType.X if (
            (neighbor_type == VertexType.Z and edge_type == EdgeType.SIMPLE)
            or (neighbor_type == VertexType.X and edge_type == EdgeType.HADAMARD)
        ) else VertexType.Z
        second_type = VertexType.Z if first_type == VertexType.X else VertexType.X
        qubit, row = expanded.qubit(boundary), expanded.row(boundary)
        first = expanded.add_vertex(first_type, qubit=qubit, row=row)
        second = expanded.add_vertex(second_type, qubit=qubit, row=row)
        expanded.add_edge((neighbor, first), edge_type)
        expanded.add_edge((first, second), EdgeType.SIMPLE)
        expanded.add_edge((second, boundary), EdgeType.SIMPLE)
        color_edges[first] = boundary, neighbor
        boundaries[first] = boundaries[second] = boundary

    order, corrections = _focused_gflow(expanded, backwards=True)
    vertices = graph.vertex_set()
    order = {
        vertex: depth for vertex, depth in order.items()
        if vertex in vertices and not phase_is_clifford(graph.phase(vertex))
    }
    zwebs, xwebs = {}, {}
    for vertex, correction in corrections.items():
        web = PauliWeb(graph)
        for target in correction:
            if target in vertices:
                pauli = "X" if graph.type(target) == VertexType.Z else "Z"
                for neighbor in graph.neighbors(target):
                    if graph.type(neighbor) == VertexType.BOUNDARY or expanded.connected(target, neighbor):
                        web.add_edge((target, neighbor), pauli)
                    else:
                        web.add_half_edge((target, neighbor), pauli)
            elif target in color_edges:
                pauli = "Z" if expanded.type(target) == VertexType.X else "X"
                web.add_edge(color_edges[target], pauli)
        reference = boundaries.get(vertex, vertex)
        if expanded.type(vertex) == VertexType.Z:
            zwebs[reference] = web
        elif expanded.type(vertex) == VertexType.X:
            xwebs[reference] = web
    return order, zwebs, xwebs


class _AugmentedZXGraph(AugmentedZXGraph):
    def add_pauli_packs(self) -> None:
        # Routing preserves the source graph; reduced webs are not used.
        self.pauli_pack = _compute_pauli_webs(self.zx_graph)


def prepare_graph(source: Any) -> Any:
    """Return an upstream augmented graph without changing global PyZX state."""
    if version("pyzx") != "0.10.7":
        raise ValueError("Topologiq's certified Pauli-flow adapter requires pyzx==0.10.7")
    return _AugmentedZXGraph(source)
