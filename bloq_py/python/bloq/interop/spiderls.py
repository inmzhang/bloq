"""Import SpiderLS's routed measurements with exact phases and source ports.

The optional upstream checkout must include ``examples/spiderls.patch``.
Clifford events fold into pipes and feedback; endpoint collisions insert local
gaps. This is a Bloq embedding rather than SpiderLS's reported optimized volume.
No upstream pickle or bgraph is loaded.
"""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re
import subprocess
from typing import Any

import bloq

from bloq.interop._geometry import (
    GeometryError as SpiderLSImportError, Position, _latest_leg, _parse_static_qasm,
    add_t_resources, build_graph, push_spatial_hadamards,
)

AUDITED_REVISION = '2c8c205ed3b9c707aca8b2f73c1b3bfd9b025bea'
_PHASES = {
    'Z': (Fraction(1),),
    'S': (Fraction(1, 2), Fraction(3, 2)),
    'T': tuple(Fraction(n, 4) for n in (1, 3, 5, 7)),
}


@dataclass(frozen=True)
class SpiderLSImport:
    """Imported graph, source-ordered interface, and auditable upstream stages."""

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
    circuit = _parse_static_qasm(qasm)
    if circuit.qubits < 1:
        raise SpiderLSImportError('boundaries', 'The source has no quantum interface')
    for gate in circuit.gates:
        if hasattr(gate, 'phase') and (Fraction(gate.phase) * 4).denominator != 1:
            raise SpiderLSImportError('unsupported_phase', f'No exact Clifford+T lowering for {gate}')
    return circuit


def _phase_history(source, target, schedule):
    """Bind every reduced source phase to its complete scheduled instrument."""
    expected = {v: Fraction(source.phase(v)) % 2 for v in source.vertices() if source.phase(v)}
    emitted, instructions = {}, {}
    for index, instruction in enumerate(target.instructions):
        phase = instruction.exact_phase
        if phase is None and instruction.kind not in ('S', 'T', 'Z'):
            continue
        if phase is None or len(instruction.vertices) != 1 or len(instruction.patches) != 1:
            raise SpiderLSImportError('source_provenance', 'Unbound target phase')
        phase = Fraction(phase) % 2
        vertex, = instruction.vertices
        if phase not in _PHASES.get(instruction.kind, ()) or vertex in emitted:
            raise SpiderLSImportError('source_provenance', 'Duplicate or misclassified target phase')
        emitted[vertex] = phase
        instructions[index] = instruction
    if emitted != expected:
        raise SpiderLSImportError('source_provenance', 'Target code lost or changed a source spider phase')

    measurements = [m for slot in schedule.slots for m in slot.measurements]
    by_instruction = defaultdict(list)
    for measurement in measurements:
        index = measurement.target_instruction
        if not 0 <= index < len(target.instructions):
            raise SpiderLSImportError('source_provenance', 'Measurement has no target instruction')
        instruction = target.instructions[index]
        if instruction.kind in ('S', 'T') or measurement.exact_phase is not None:
            if instruction.kind not in ('S', 'T') or measurement.exact_phase != instruction.exact_phase:
                raise SpiderLSImportError('source_provenance', 'Measurement lost or changed its target phase')
            by_instruction[index].append(measurement)
    z_events = defaultdict(list)
    for event in schedule.zero_time_events:
        if event.kind == 'Z':
            if event.target_instruction not in instructions or instructions[event.target_instruction].kind != 'Z':
                raise SpiderLSImportError('source_provenance', 'Z event has no target phase')
            z_events[event.target_instruction].append(event)

    gadgets = {}
    for index, instruction in instructions.items():
        if instruction.kind == 'Z':
            events = z_events[index]
            if len(events) != 1 or events[0].patch != instruction.patches[0] or events[0].vertices != instruction.vertices:
                raise SpiderLSImportError('source_provenance', 'Missing or misbound Z event')
            continue
        ms = sorted(by_instruction[index], key=lambda m: m.stage)
        stages = 2 if instruction.kind == 'S' else 1
        if [m.stage for m in ms] != list(range(1, stages + 1)) or any(m.stages != stages for m in ms):
            raise SpiderLSImportError('source_provenance', f'Incomplete {instruction.kind} instrument')
        gadget = ms[0].gadget_id
        if gadget in gadgets or any(m.gadget_id != gadget for m in ms):
            raise SpiderLSImportError('source_provenance', 'Phase instruments share or change gadget IDs')
        patch, = instruction.patches
        resource = ms[0].operands[-1] if ms[0].operands else ''
        expected_resource = re.fullmatch(r'q[A-Z]+', resource) if stages == 2 else re.fullmatch(r'qM\d+', resource)
        if not expected_resource or ms[0].operands != (f'q{patch}', resource) or ms[0].paulis != ('Z', 'Z'):
            raise SpiderLSImportError('source_provenance', 'Phase instrument changed its source patch or parity')
        if stages == 2 and (ms[1].operands != (resource,) or ms[1].paulis != ('Y',)):
            raise SpiderLSImportError('source_provenance', 'S instrument lost its ancilla Y effect')
        gadgets[gadget] = (instruction.kind, Fraction(instruction.exact_phase) % 2, ms)
    return {m.measurement_id: m for m in measurements}, gadgets


def _convert(source, target, schedule, routing, *, _build=None):
    from bloq.interop._spiderls_timeline import diagram

    measurements, gadgets = _phase_history(source, target, schedule)
    nodes, edges, zs = diagram(source, target, schedule, routing)
    routes = {c.measurement_id: (c, layer) for layer in routing.layers for c in layer.connections}
    actions = [bloq.Action.feedback([bloq.FeedbackTarget(pauli, nodes[n]['position'], direction='+Z')]) for n, pauli in zs]
    resources = []
    s_gadgets = set()
    for n, d in nodes.items():
        if d['other'] == 'T':
            match = re.fullmatch(r'magic_m(\d+)_\d+_t\d+', n)
            if match is None or int(match[1]) not in measurements:
                raise SpiderLSImportError('source_provenance', f'Magic endpoint {n!r} has no measurement')
            m = measurements[int(match[1])]
            if m.gadget_id not in gadgets or gadgets[m.gadget_id][0] != 'T':
                raise SpiderLSImportError('source_provenance', f'Magic endpoint {n!r} has no T instrument')
            resources.append((n, gadgets[m.gadget_id][1], m.measurement_id))
        elif d['other'] == 'S':
            match = re.fullmatch(r'ancilla_g(\d+)_t\d+', n)
            if match is None:
                raise SpiderLSImportError('source_provenance', f'S endpoint {n!r} has no gadget')
            gadget = int(match[1])
            if gadget not in gadgets or gadgets[gadget][0] != 'S':
                raise SpiderLSImportError('source_provenance', f'S endpoint {n!r} has no S instrument')
            if gadget in s_gadgets:
                raise SpiderLSImportError('source_provenance', f'Duplicate S endpoint {n!r}')
            s_gadgets.add(gadget)
            _, phase, ms = gadgets[gadget]
            d['tqec'] = 'Y'
            if phase == Fraction(1, 2):
                # A future Y effect contributes -pi/2. Z on the data leg of
                # the gadget's Z spider supplies pi and hence positive S.
                first = ms[0]
                connection, layer = routes[first.measurement_id]
                terminal = next(t for t in connection.terminals if t.kind == 'data')
                root = f'q{int(terminal.operand[1:])}_t{layer.time}'
                p = nodes[root]['position']
                leg = _latest_leg(p, (nodes[b if a == root else a]['position'] for a, b in edges if root in (a, b)))
                actions.append(bloq.Action.feedback([bloq.FeedbackTarget('Z', p, direction=leg)]))
    expected_s = {gadget for gadget, (kind, _, _) in gadgets.items() if kind == 'S'}
    if s_gadgets != expected_s:
        raise SpiderLSImportError('source_provenance', 'Missing S endpoint')
    expected = {ms[0].measurement_id for kind, _, ms in gadgets.values() if kind == 'T'}
    if {mid for _, _, mid in resources} != expected or len(resources) != len(expected):
        raise SpiderLSImportError('source_provenance', 'Missing or duplicate T endpoint')
    actions = push_spatial_hadamards(nodes, edges, actions=actions)
    resources = [(node, phase, f'spiderls_t_{mid}') for node, phase, mid in resources]
    if _build is not None:
        return _build(nodes, edges, len(source.inputs()), actions, resources)
    actions.extend(add_t_resources(nodes, edges, resources))
    return build_graph(nodes, edges, len(source.inputs()), actions)


def from_qasm(qasm: str, *, spacing: int = 3, routing_margin: int = 3,
              placement_margin: int | None = None,
              max_physical_layers: int | None = None, _build=None) -> SpiderLSImport:
    """Reduce, schedule, route, and import an exact static Clifford+T source.

    Routing spacing must be at least two; margins must be positive. Smaller
    values trade routing room for footprint and can make a layout infeasible.
    Bloq preserves the routed unit grid, adding local gaps for endpoint cubes.
    The upstream layer budget scales with source depth unless an explicit
    positive ``max_physical_layers`` hard cap is supplied.
    """
    import pyzx as zx
    import pyzx_execution_order as execution
    from pyzx_target_code import generate_target_code
    from pyzx_parallelization import parallelize_target_program
    from pyzx_routing import route_parallel_schedule, validate_routing_result

    for name, value, minimum in [('spacing', spacing, 2), ('routing_margin', routing_margin, 1),
                                  ('placement_margin', placement_margin, 1),
                                  ('max_physical_layers', max_physical_layers, 1)]:
        if name in ('placement_margin', 'max_physical_layers') and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    circuit = _source(qasm)
    if execution._phase_gate_label(Fraction(1)) != 'Z':
        raise SpiderLSImportError('upstream_patch', 'Apply bloq_py/examples/spiderls.patch to preserve pi phases')
    source = circuit.to_graph()
    zx.full_reduce(source)
    plan = execution.extract_execution_order(source)
    target = generate_target_code(plan)
    schedule = parallelize_target_program(target, include_boundary_h=True)
    if target.initial_patch_map or schedule.slots:
        routing = route_parallel_schedule(schedule, spacing=spacing, routing_margin=routing_margin,
                                          placement_margin=placement_margin,
                                          max_physical_layers=max_physical_layers)
        validate_routing_result(routing, schedule)
    else:
        from pyzx_routing import BoundingBox, RoutingResult

        routing = RoutingResult((), BoundingBox(0, 0, 0, 0), 0, {}, {}, {}, 0, 0, 0, 0)
    graph, inputs, outputs = _convert(source, target, schedule, routing, _build=_build)
    root = Path(execution.__file__).resolve().parents[1]
    try:
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    except OSError:
        revision = None
    options = dict(routing.parameters, include_boundary_h=True, grid_spacing=(1, 1, 1), ordered_zero_time_events=True)
    return SpiderLSImport(graph, inputs, outputs, qasm, revision.stdout.strip() if revision and revision.returncode == 0 else None,
                          options, source, plan, target, schedule, routing)
