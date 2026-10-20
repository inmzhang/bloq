"""Unit-grid geometry shared by the layout adapters."""

from bisect import bisect_left
from collections import defaultdict
from fractions import Fraction
from numbers import Integral
from typing import Any

import bloq

Position = tuple[int, int, int]
GRID_SPACING = 3
_CUBES = {"XZZ", "ZXZ", "ZZX", "ZXX", "XZX", "XXZ"}


class GeometryError(ValueError):
    """A source or layout that cannot be represented faithfully."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _position(value: Any) -> Position:
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in value)):
        raise GeometryError("coordinates", f"Expected an integer grid position, got {value!r}")
    result = (int(value[0]), int(value[1]), int(value[2]))
    if any(not -(2**31) <= x < 2**31 for x in result):
        raise GeometryError("coordinates", "Coordinates exceed signed 32-bit range")
    return result


def _direction(start: Position, end: Position) -> str:
    delta = tuple(b - a for a, b in zip(start, end))
    if sum(abs(x) for x in delta) != 1:
        raise GeometryError("pipe_geometry", f"Pipe is not unit axial: {start} -> {end}")
    axis = next(i for i, x in enumerate(delta) if x)
    return ("+" if delta[axis] > 0 else "-") + "XYZ"[axis]


def _space_diagram(nodes, edges):
    """Leave a local resource collar while preserving topology and H parity.

    Every original unit pipe becomes three unit pipes. Its inserted cubes
    have degree two and phase zero (identity tensors); any H stays on the
    middle segment. Endpoint gadgets then fit between original grid cells.
    """
    positions = {node: _position(data["position"]) for node, data in nodes.items()}
    if len(set(positions.values())) != len(positions):
        raise GeometryError("coordinates", "Exported nodes overlap")
    spaced = {node: dict(data, position=_position(tuple(GRID_SPACING * x for x in positions[node])))
              for node, data in nodes.items()}
    pipes = {}
    for index, ((a, b), endpoints) in enumerate(edges.items()):
        if a not in positions or b not in positions:
            raise GeometryError("pipe_geometry", f"Unknown exported edge endpoint: {a!r} -> {b!r}")
        if tuple(_position(p) for p in endpoints) != (positions[a], positions[b]):
            raise GeometryError("pipe_geometry", f"Stored endpoints disagree with node positions: {a!r} -> {b!r}")
        _direction(positions[a], positions[b])
        p, q = spaced[a]["position"], spaced[b]["position"]
        step = tuple((y - x) // GRID_SPACING for x, y in zip(p, q))
        ka, kb = nodes[a]["tqec"], nodes[b]["tqec"]
        if ka not in _CUBES and kb not in _CUBES:
            raise GeometryError("pipe_basis", "A resource wire has no ordinary frame")
        previous = a
        for distance in range(1, GRID_SPACING):
            node = f"bloq_spacing_{index}_{distance}"
            if node in spaced:
                raise GeometryError("source_provenance", "Reserved subdivision node ID already exists")
            frame = ka if distance == 1 else kb
            if frame is None:
                frame = kb if ka is None else ka
            position = tuple(x + distance * dx for x, dx in zip(p, step))
            spaced[node] = dict(position=position, tqec=frame, other=None)
            pipes[previous, node] = (spaced[previous]["position"], position)
            previous = node
        pipes[previous, b] = (spaced[previous]["position"], q)
    return spaced, pipes


def add_t_resource(nodes, edges, node, *, phase, name):
    """Expand one terminal into the channel-checked ZZ / adaptive XY gadget."""
    phase = Fraction(phase) % 2
    if phase not in {Fraction(n, 4) for n in (1, 3, 5, 7)}:
        raise GeometryError('unsupported_phase', f'Unsupported magic phase {phase}')
    dagger = phase in (Fraction(3, 4), Fraction(7, 4))
    p = _position(nodes[node]['position'])
    adjacent = [b if a == node else a for a, b in edges if node in (a, b)]
    if len(adjacent) != 1:
        raise GeometryError('resource_frame', f'T resource {node!r} is not terminal')
    neighbor = adjacent[0]
    q = _position(nodes[neighbor]['position'])
    direction = _direction(q, p)
    axis = 'XYZ'.index(direction[1])
    if axis == 2 or nodes[neighbor]['tqec'] not in _CUBES:
        raise GeometryError('resource_frame', f'T resource {node!r} has no proved spatial ZZ seam')
    occupied = {tuple(d['position']) for d in nodes.values()}

    def add(suffix, position, kind):
        key = f'{name}_{suffix}'
        if key in nodes or position in occupied:
            raise GeometryError('resource_collision', f'T gadget {node!r} collides at {position}')
        _position(position)
        occupied.add(position)
        nodes[key] = dict(position=position, tqec=kind, other=None)
        return key

    if nodes[neighbor]['tqec'][2] == 'Z':
        # A phase-zero elbow reaches the canonical seam without changing the
        # routed path. Its three legs use each pipe axis once.
        nodes[node]['tqec'] = 'ZXZ' if axis == 0 else 'XZZ'
        turn = (p[0], p[1], p[2] + 1)
        center = (p[0] + (axis == 1), p[1] + (axis == 0), p[2] + 1)
        canonical = 'ZXX' if axis == 0 else 'XZX'
        a, b = add('turn', turn, canonical), add('center', center, canonical)
        edges.extend([(node, a), (a, b)])
        node, p, q = b, center, turn
        direction = '+Y' if axis == 0 else '+X'
    else:
        nodes[node]['tqec'] = 'XZX' if axis == 0 else 'ZXX'
    before = add('source', (p[0], p[1], p[2] - 1), 'T')
    after = add('cap', (p[0], p[1], p[2] + 1), 'XY')
    edges.extend([(before, node), (node, after)])
    actions = [
        bloq.Action.measure(bloq.MeasureTarget.edge(q, direction), name),
        bloq.Action.resolve(nodes[after]['position'], bloq.Expr.var(name) if dagger else ~bloq.Expr.var(name)),
    ]
    if phase in (Fraction(3, 4), Fraction(5, 4)):
        # 3pi/4 = pi - pi/4 and 5pi/4 = pi + pi/4; the residual
        # Pauli acts on the same canonical resource seam.
        actions.append(bloq.Action.feedback([bloq.FeedbackTarget('Z', q, direction=direction)]))
    return actions


def _compact_temporal_layers(graph):
    """Contract complete identity slices in a flat, unit-height import.

    This is an import optimization, not a rewrite of authored memory duration.
    Named actions, tagged geometry and nonordinary blocks are anchors.
    Temporal H seams compose by parity when their identity cubes are removed.
    The native compiler must still validate physical patch lifetimes afterward.
    """
    if graph.has_module_structure:
        raise GeometryError('time_compaction', 'Flatten the imported graph before time compaction')
    graph.validate()
    blocks = {b.pos: b for b in graph.blocks()}
    positions = {p: p for p in blocks}
    actions = graph.actions()
    if (any(b.height != 'd' for b in blocks.values())
            or any(a.kind not in ('let', 'measure', 'feedback', 'resolve', 'discard_if') for a in actions)):
        return graph, positions
    rows = defaultdict(list)
    neighbors: dict[Position, dict[Position, tuple[bool, str | None]]] = {p: {} for p in blocks}
    for b in blocks.values():
        rows[b.pos[2]].append(b)
    for pipe in graph.pipes():
        neighbors[pipe.src][pipe.dst] = neighbors[pipe.dst][pipe.src] = (pipe.hadamard, pipe.tag)
    protected = set()

    def protect(pos, direction=None):
        protected.add(pos[2])
        if direction is not None and str(direction)[1] == 'Z':
            protected.add(pos[2] + (1 if str(direction)[0] == '+' else -1))

    for action in actions:
        if action.target is not None:
            protect(action.target.pos, action.target.direction)
        if action.resolve_target is not None:
            protect(action.resolve_target)
        for target in action.feedback_targets or []:
            protect(target.pos, target.direction)

    def identity(block):
        kind, p = str(block.kind), block.pos
        return (kind in _CUBES and kind[0] != kind[1] and block.tag is None
                and len(neighbors[p]) == 2
                and all(q[:2] == p[:2] and abs(q[2] - p[2]) == 1
                        and tag is None for q, (_, tag) in neighbors[p].items()))

    removed = sorted(z for z, row in rows.items() if z not in protected and all(map(identity, row)))
    if not removed:
        return graph, positions
    for z in removed:
        for block in rows[z]:
            p = block.pos
            adjacent = neighbors.pop(p)
            lower, upper = sorted(adjacent, key=lambda q: q[2])
            del neighbors[lower][p]
            del neighbors[upper][p]
            neighbors[lower][upper] = neighbors[upper][lower] = (adjacent[lower][0] ^ adjacent[upper][0], None)
            del positions[p]
    positions = {p: (p[0], p[1], p[2] - bisect_left(removed, p[2])) for p in positions}
    compact = bloq.BlockGraph()
    for p, q in positions.items():
        b = blocks[p]
        compact.add_block(bloq.Block(q, b.kind, height=b.height, tag=b.tag, role=b.role or 'auto'))
    for p, adjacent in neighbors.items():
        for q, (hadamard, tag) in adjacent.items():
            if p < q:
                compact.add_pipe(bloq.Pipe(positions[p], _direction(positions[p], positions[q]),
                                          hadamard=hadamard, tag=tag))
    remapped = []
    for action in actions:
        if action.target is not None:
            target = action.target
            target = (bloq.MeasureTarget.edge(positions[target.pos], target.direction) if target.is_edge
                      else bloq.MeasureTarget.node(positions[target.pos]))
            action = bloq.Action.measure(target, action.name)
        elif action.resolve_target is not None:
            action = bloq.Action.resolve(positions[action.resolve_target], action.condition)
        elif action.feedback_targets is not None:
            action = bloq.Action.feedback([
                bloq.FeedbackTarget(t.pauli, positions[t.pos], direction=t.direction)
                for t in action.feedback_targets
            ], action.condition)
        remapped.append(action)
    compact.set_actions(remapped)
    compact.validate()
    return compact, positions


def build_graph(nodes, edges, qubits, actions):
    """Validate frames and bind the complete, source-ordered interface."""
    positions = {n: _position(d['position']) for n, d in nodes.items()}
    if len(set(positions.values())) != len(positions):
        raise GeometryError('coordinates', 'Exported nodes overlap')
    ports: dict[str, dict[int, Position]] = {'input': {}, 'output': {}}
    graph = bloq.BlockGraph()
    for n, data in nodes.items():
        role = data['other']
        if isinstance(role, dict):
            kind, wire = role.get('type'), role.get('qubit')
            if kind not in ports or type(wire) is not int or not 0 <= wire < qubits or wire in ports[kind]:
                raise GeometryError('boundaries', f'Invalid or duplicate source port {role}')
            if sum(n in edge for edge in edges) != 1:
                raise GeometryError('boundaries', f'Port {n!r} is not terminal')
            ports[kind][wire] = positions[n]
        graph.add_block(bloq.Block(positions[n], 'Port' if isinstance(role, dict) else data['tqec'],
                                   role=role['type'] if isinstance(role, dict) else 'auto'))
    for a, b in edges:
        p, q = positions[a], positions[b]
        direction = _direction(p, q)
        axis = 'XYZ'.index(direction[1])
        ka, kb = nodes[a]['tqec'], nodes[b]['tqec']
        flips = [ka[i] != kb[i] for i in range(3) if i != axis] if ka in _CUBES and kb in _CUBES else [False, False]
        if flips[0] != flips[1]:
            raise GeometryError('pipe_basis', f'Mixed transverse frames on {a!r} -> {b!r}')
        graph.add_pipe(bloq.Pipe(p, direction, hadamard=all(flips)))
    if any(set(mapping) != set(range(qubits)) for mapping in ports.values()):
        raise GeometryError('boundaries', 'Layout does not cover the source interface')
    graph.set_actions(actions)
    # Resource collars can precede the upstream origin. Keep interface maps
    # aligned with the nonnegative coordinates used by native compilation.
    min_z = min(p[2] for p in positions.values())
    graph = graph.with_zero_min_z()
    inputs, outputs = (tuple((x, y, z - min_z) for _, (x, y, z) in sorted(ports[role].items()))
                       for role in ('input', 'output'))
    graph, positions = _compact_temporal_layers(graph)
    inputs, outputs = (tuple(positions[p] for p in interface) for interface in (inputs, outputs))
    graph = bloq.BlockGraph.definition('main', graph,
        inputs={f'in_{i}': p for i, p in enumerate(inputs)}, outputs={f'out_{i}': p for i, p in enumerate(outputs)})
    graph.validate()
    return graph, inputs, outputs
