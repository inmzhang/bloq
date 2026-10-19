"""Import SpiderLS's routed measurements with exact phases and source ports.

The optional upstream checkout must include ``examples/spiderls.patch``.
Zero-time events become ordered slices; this is a Bloq embedding rather than
SpiderLS's reported optimized volume. No upstream pickle or bgraph is loaded.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
import subprocess
from typing import Any

import bloq

from bloq.interop._geometry import (
    GRID_SPACING, GeometryError, Position, _direction, _space_diagram,
    add_t_resource, build_graph,
)
from bloq.interop._qasm import QASMImportError, parse_qasm

AUDITED_REVISION = '2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea'


class SpiderLSImportError(ValueError):
    """A source or upstream result that cannot be imported faithfully."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f'{code}: {message}')


@dataclass(frozen=True)
class SpiderLSImport:
    graph: bloq.BlockGraph
    input_ports: tuple[Position, ...]
    output_ports: tuple[Position, ...]
    source_qasm: str
    upstream_revision: str | None
    layout_options: dict[str, Any]
    reduced_graph: Any
    plan: Any
    target: Any
    schedule: Any
    routing: Any


def _source(qasm):
    clean = re.sub(r'//[^\n]*|/\*.*?\*/', '', qasm, flags=re.S)
    if not re.match(r'\s*OPENQASM\s+2(?:\.\d+)?\s*;', clean):
        raise SpiderLSImportError('qasm_version', 'Expected an OpenQASM 2 source circuit')
    if re.search(r'\b(measure|reset|if)\b', clean):
        raise SpiderLSImportError('classical_control', 'Only static unitary sources are supported')
    try:
        circuit = parse_qasm(clean)
    except QASMImportError as error:
        raise SpiderLSImportError(error.code, error.message) from error
    if circuit.qubits < 1:
        raise SpiderLSImportError('boundaries', 'The source has no quantum interface')
    for gate in circuit.gates:
        if hasattr(gate, 'phase') and (Fraction(gate.phase) * 4).denominator != 1:
            raise SpiderLSImportError('unsupported_phase', f'No exact Clifford+T lowering for {gate}')
    return circuit


def _convert(source, target, schedule, routing):
    from bloq.interop._spiderls_timeline import diagram

    nodes, edges, zs = diagram(source, target, schedule, routing)
    nodes, edges = _space_diagram(nodes,
        {pair: (nodes[pair[0]]['position'], nodes[pair[1]]['position']) for pair in edges})
    edges = list(edges)
    measurements = {m.measurement_id: m for slot in schedule.slots for m in slot.measurements}
    actions = [bloq.Action.feedback([bloq.FeedbackTarget('Z', nodes[n]['position'], direction='+Z')]) for n in zs]
    resources = []
    s_gadgets = set()
    for n, d in nodes.items():
        if d['other'] == 'T':
            match = re.fullmatch(r'magic_m(\d+)_\d+_t\d+', n)
            if match is None or int(match[1]) not in measurements:
                raise SpiderLSImportError('source_provenance', f'Magic endpoint {n!r} has no measurement')
            m = measurements[int(match[1])]
            if m.exact_phase is None or target.instructions[m.target_instruction].exact_phase != m.exact_phase:
                raise SpiderLSImportError('source_provenance', f'Magic endpoint {n!r} lost its exact phase')
            resources.append((n, Fraction(m.exact_phase) % 2, m.measurement_id))
        elif d['other'] == 'S':
            match = re.fullmatch(r'ancilla_g(\d+)_t\d+', n)
            if match is None:
                raise SpiderLSImportError('source_provenance', f'S endpoint {n!r} has no gadget')
            ms = [m for m in measurements.values() if m.gadget_id == int(match[1])]
            if sorted(m.stage for m in ms) != [1, 2] or any(
                    m.exact_phase != target.instructions[m.target_instruction].exact_phase for m in ms):
                raise SpiderLSImportError('source_provenance', f'S endpoint {n!r} lost its gadget history')
            if int(match[1]) in s_gadgets:
                raise SpiderLSImportError('source_provenance', f'Duplicate S endpoint {n!r}')
            s_gadgets.add(int(match[1]))
            phases = {Fraction(m.exact_phase) % 2 for m in ms if m.exact_phase is not None}
            if len(phases) != 1 or not phases <= {Fraction(1, 2), Fraction(3, 2)}:
                raise SpiderLSImportError('source_provenance', f'S endpoint {n!r} has no signed phase')
            d['tqec'] = 'Y'
            if phases == {Fraction(1, 2)}:
                # A future Y effect contributes -pi/2. Z on the data leg of
                # the gadget's Z spider supplies pi and hence positive S.
                first = next(m for m in ms if m.stage == 1)
                connection, layer = next((c, routed_layer) for routed_layer in routing.layers for c in routed_layer.connections
                                          if c.measurement_id == first.measurement_id)
                terminal = next(t for t in connection.terminals if t.kind == 'data')
                root = f'q{int(terminal.operand[1:])}_t{layer.time}'
                p = nodes[root]['position']
                leg = next(b if a == root else a for a, b in edges if root in (a, b)
                           and nodes[b if a == root else a]['position'][2] > p[2])
                actions.append(bloq.Action.feedback([bloq.FeedbackTarget('Z', p, direction=_direction(p, nodes[leg]['position']))]))
    expected_s = {m.gadget_id for m in measurements.values()
                  if m.exact_phase is not None and Fraction(m.exact_phase) % 2 in (Fraction(1, 2), Fraction(3, 2))}
    if s_gadgets != expected_s:
        raise SpiderLSImportError('source_provenance', 'Missing S endpoint')
    expected = {m.measurement_id for m in measurements.values() if m.exact_phase is not None and (Fraction(m.exact_phase) * 4) % 2}
    if {mid for _, _, mid in resources} != expected or len(resources) != len(expected):
        raise SpiderLSImportError('source_provenance', 'Missing or duplicate T endpoint')
    for node, phase, mid in resources:
        actions.extend(add_t_resource(nodes, edges, node, phase=phase, name=f'spiderls_t_{mid}'))
    return build_graph(nodes, edges, len(source.inputs()), actions)


def from_qasm(qasm: str) -> SpiderLSImport:
    """Reduce, schedule, route, and import an exact static Clifford+T source."""
    import pyzx as zx
    import pyzx_execution_order as execution
    from pyzx_target_code import generate_target_code
    from pyzx_parallelization import parallelize_target_program
    from pyzx_routing import route_parallel_schedule, validate_routing_result

    circuit = _source(qasm)
    if execution._phase_gate_label(Fraction(1)) != 'Z':
        raise SpiderLSImportError('upstream_patch', 'Apply bloq_py/examples/spiderls.patch to preserve pi phases')
    source = circuit.to_graph()
    zx.full_reduce(source)
    plan = execution.extract_execution_order(source)
    for vertex, phase in plan.spider_phases:
        if (Fraction(phase) * 4).denominator != 1:
            raise SpiderLSImportError('unsupported_phase', f'Non-Clifford+T spider {vertex}: {phase}')
    target = generate_target_code(plan)
    phases = {vertex: Fraction(phase) % 2 for vertex, phase in plan.spider_phases if Fraction(phase) % 2}
    emitted = {}
    for instruction in target.instructions:
        if instruction.exact_phase is not None:
            if len(instruction.vertices) != 1 or instruction.vertices[0] in emitted:
                raise SpiderLSImportError('source_provenance', 'Duplicate or unbound target phase')
            emitted[instruction.vertices[0]] = Fraction(instruction.exact_phase) % 2
    if emitted != phases:
        raise SpiderLSImportError('source_provenance', 'Target code lost or changed a source spider phase')
    schedule = parallelize_target_program(target, include_boundary_h=True)
    if target.initial_patch_map or schedule.slots:
        routing = route_parallel_schedule(schedule)
        validate_routing_result(routing, schedule)
    else:
        from pyzx_routing import BoundingBox, RoutingResult

        routing = RoutingResult((), BoundingBox(0, 0, 0, 0), 0, {}, {}, {}, 0, 0, 0, 0)
    try:
        graph, inputs, outputs = _convert(source, target, schedule, routing)
    except GeometryError as error:
        raise SpiderLSImportError(error.code, error.message) from error
    root = Path(execution.__file__).resolve().parents[1]
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    options = dict(routing.parameters, include_boundary_h=True, grid_spacing=GRID_SPACING, ordered_zero_time_events=True)
    return SpiderLSImport(graph, inputs, outputs, qasm, revision.stdout.strip() if revision.returncode == 0 else None,
                          options, source, plan, target, schedule, routing)
