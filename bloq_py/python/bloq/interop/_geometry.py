"""Unit-grid geometry shared by the layout adapters."""

from bisect import bisect_left
from collections import Counter, defaultdict, deque
from fractions import Fraction
from importlib import metadata
from itertools import count, islice
import json
from math import isqrt
from numbers import Integral
import re
from typing import Any

import bloq

Position = tuple[int, int, int]
GRID_SPACING = (1, 1, 3)
_CUBES = {"XZZ", "ZXZ", "ZZX", "ZXX", "XZX", "XXZ"}


class GeometryError(ValueError):
    """A source or layout that cannot be represented faithfully."""

    def __init__(self, code: str, message: str, *, nodes: tuple[int, ...] = ()):
        self.code = code
        self.message = message
        self.nodes = nodes
        super().__init__(f"{code}: {message}" + (f" (nodes {nodes})" if nodes else ""))


def _static_qasm(qasm: str) -> str:
    """Strip comments, rejecting other versions and classical control before parsing."""
    clean = re.sub(r"//[^\n]*|/\*.*?\*/", "", qasm, flags=re.S)
    if not re.match(r"\s*OPENQASM\s+2(?:\.\d+)?\s*;", clean):
        raise GeometryError("qasm_version", "Expected an OpenQASM 2 source circuit")
    if re.search(r"\b(measure|reset|if)\b", clean):
        raise GeometryError("classical_control", "Only static unitary source circuits are supported")
    return clean


def _parse_static_qasm(qasm: str) -> Any:
    from bloq.interop._qasm import QASMImportError, parse_qasm

    try:
        return parse_qasm(_static_qasm(qasm))
    except QASMImportError as error:
        raise GeometryError(error.code, error.message) from error


def _installed_revision(distribution: str) -> str | None:
    try:
        direct = metadata.distribution(distribution).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None


def _position(value: Any) -> Position:
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, Integral)) for x in value)):
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


_COLOR_CHANGE = str.maketrans("XZ", "ZX")


def _transverse_flips(nodes, a, b):
    axis = "XYZ".index(_direction(_position(nodes[a]["position"]), _position(nodes[b]["position"]))[1])
    ka, kb = nodes[a]["tqec"], nodes[b]["tqec"]
    return [ka[i] != kb[i] for i in range(3) if i != axis] if ka in _CUBES and kb in _CUBES else [False, False]


def push_spatial_hadamards(nodes, edges, fixed=()):
    """Move spatial Hadamards into time, or retain walls on only one axis.

    A ZX color change swaps a cube's X and Z faces, toggling every incident
    Hadamard. Cubes touching other blocks, or listed in ``fixed``, keep their
    frames. If those anchors prevent removing every wall, solve each choice
    of remaining spatial axis. Unsatisfiable components keep their frames.
    """
    def cube(n):
        return nodes[n]["other"] is None and nodes[n]["tqec"] in _CUBES

    fixed = set(fixed)
    spatial = defaultdict(list)
    for a, b in edges:
        if not (cube(a) and cube(b)):
            fixed.update((a, b))
        elif (axis := _direction(_position(nodes[a]["position"]), _position(nodes[b]["position"]))[1]) != "Z":
            wall = all(_transverse_flips(nodes, a, b))
            spatial[a].append((b, wall, axis))
            spatial[b].append((a, wall, axis))

    def solve(component, allowed_axis=None):
        result: dict[Any, bool] = {}
        for start in component:
            if start in result:
                continue
            flips, queue = {start: False}, [start]
            for n in queue:
                for m, wall, axis in spatial[n]:
                    if axis == allowed_axis:
                        continue
                    flip = flips[n] ^ wall
                    if m not in flips:
                        flips[m] = flip
                        queue.append(m)
                    elif flips[m] != flip:
                        return None
            anchors = {flips[n] for n in flips if n in fixed}
            if len(anchors) > 1:
                return None
            keep = anchors.pop() if anchors else 2 * sum(flips.values()) > len(flips)
            result.update((n, flip ^ keep) for n, flip in flips.items())
        return result

    seen: set[Any] = set()
    for start in list(spatial):
        if start in seen:
            continue
        component, queue, wall_axes = {start}, [start], set()
        for n in queue:
            for m, wall, axis in spatial[n]:
                if wall:
                    wall_axes.add(axis)
                if m not in component:
                    component.add(m)
                    queue.append(m)
        seen.update(component)
        if not wall_axes:
            continue
        flips = solve(queue)
        if flips is None and len(wall_axes) > 1:
            candidates = [solution for axis in ("X", "Y") if (solution := solve(queue, axis)) is not None]
            flips = min(candidates, key=lambda solution: (
                sum(wall ^ solution[n] ^ solution[m] for n in queue for m, wall, _ in spatial[n]),
                sum(solution.values()),
            ), default=None)
        if flips is None:
            continue
        for n, flip in flips.items():
            if flip:
                nodes[n]["tqec"] = nodes[n]["tqec"].translate(_COLOR_CHANGE)


def _latest_leg(start: Position, legs) -> str:
    # Z on any leg of a Z spider adds the same phase; only determinism matters.
    return _direction(start, max(map(_position, legs), key=lambda leg: (leg[2], leg)))


def _space_diagram(nodes, edges):
    """Preserve spatial routing and leave temporal room for resource endpoints.

    Temporal pipes become three unit pipes with phase-zero degree-two cubes.
    Spatial pipes and their H parity retain the upstream unit grid.
    """
    positions = {node: _position(data["position"]) for node, data in nodes.items()}
    if len(set(positions.values())) != len(positions):
        raise GeometryError("coordinates", "Exported nodes overlap")
    spaced = {node: dict(data, position=_position(tuple(s * x for s, x in zip(GRID_SPACING, positions[node]))))
              for node, data in nodes.items()}
    pipes = {}
    for index, ((a, b), endpoints) in enumerate(edges.items()):
        if a not in positions or b not in positions:
            raise GeometryError("pipe_geometry", f"Unknown exported edge endpoint: {a!r} -> {b!r}")
        if tuple(_position(p) for p in endpoints) != (positions[a], positions[b]):
            raise GeometryError("pipe_geometry", f"Stored endpoints disagree with node positions: {a!r} -> {b!r}")
        direction = _direction(positions[a], positions[b])
        length = GRID_SPACING['XYZ'.index(direction[1])]
        p, q = spaced[a]["position"], spaced[b]["position"]
        step = tuple((y - x) // length for x, y in zip(p, q))
        ka, kb = nodes[a]["tqec"], nodes[b]["tqec"]
        if ka not in _CUBES and kb not in _CUBES:
            raise GeometryError("pipe_basis", "A resource wire has no ordinary frame")
        previous = a
        for distance in range(1, length):
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


def _t_placements(nodes, edges, node, *, bends=False, margin=2, positions=None):
    """Enumerate verified arm/elbow footprints without changing the routed graph."""
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
    if positions is None:
        positions = {n: _position(d['position']) for n, d in nodes.items()}
    occupied = {p for n, p in positions.items() if n != node}
    space = {r[:2] for r in occupied}
    factories = {positions[n][:2] for n, d in nodes.items()
                 if n != node and d['tqec'] == 'T'}
    bounds = [(min(r[i] for r in space), max(r[i] for r in space)) for i in (0, 1)]
    frame = nodes[neighbor]['tqec']
    step = tuple(b - a for a, b in zip(q, p))
    limit = (bounds[axis][1 if step[axis] > 0 else 0] - p[axis]) * step[axis] + margin
    def paths():
        path = [p]
        for distance in range(max(0, limit) + 1):
            end = _position(tuple(a + distance * dx for a, dx in zip(p, step)))
            if distance and end in occupied:
                break
            if distance:
                path.append(end)
            yield path, frame
        if not bends:
            return
        # Degree-two, phase-zero spiders are identities in either color. Spatial
        # corner frames avoid degenerate faces; finish on the original axis so
        # the certified injection frame and recipe are unchanged.
        side = 1 - axis
        # Breadth first exposes short arms from different corridors before
        # filling a domain with longer variants of the same outlet.
        queue = deque([([p], 0, 0, 0, 0)])
        while queue:
            path, forward, sign, offset, distance = queue.popleft()
            moves = []
            if not sign:
                if forward < max(0, limit):
                    moves.append((step, forward + 1, 0, 0, 0))
                for s in (1, -1):
                    reach = (bounds[side][1 if s > 0 else 0] - p[side]) * s + margin
                    if reach > 0:
                        moves.append((tuple(s if i == side else 0 for i in range(3)), forward, s, 1, 0))
            elif not distance:
                reach = (bounds[side][1 if sign > 0 else 0] - p[side]) * sign + margin
                if offset < reach:
                    moves.append((tuple(sign if i == side else 0 for i in range(3)), forward, sign, offset + 1, 0))
                moves.append((step, forward, sign, offset, 1))
            else:
                yield path, 'ZZX' if frame[2] == 'X' else 'XXZ'
                if distance < max(1, limit - forward):
                    moves.append((step, forward, sign, offset, distance + 1))
            for delta, f, s, o, d in moves:
                end = _position(tuple(a + dx for a, dx in zip(path[-1], delta)))
                if end not in occupied and end[:2] not in factories:
                    queue.append((path + [end], f, s, o, d))

    for path, arm_frame in paths():
        end, distance = path[-1], len(path) - 1
        choices = []
        for sign in ((1, -1) if frame[2] == 'Z' else (1,)):
            if frame[2] == 'Z':
                terminal = 'ZXZ' if axis == 0 else 'XZZ'
                turn = (end[0], end[1], end[2] + 1)
                center = (end[0] + sign * (axis == 1), end[1] + sign * (axis == 0), end[2] + 1)
                canonical = 'ZXX' if axis == 0 else 'XZX'
            else:
                terminal = 'XZX' if axis == 0 else 'ZXX'
                center = end
            # The adaptive cap may wait beyond its source layer. A different
            # Z coordinate does not certify that its stored state was consumed.
            if center[:2] in space:
                continue
            additions = {f'arm_{i}': (position, arm_frame) for i, position in enumerate(path[:-1])}
            if frame[2] == 'Z':
                additions.update(turn=(turn, canonical), center=(center, canonical))
            additions.update(source=((center[0], center[1], center[2] - 1), 'T'),
                             cap=((center[0], center[1], center[2] + 1), 'XY'))
            footprint = [end, *(r for r, _ in additions.values())]
            if (len(set(footprint)) != len(footprint) or any(r in occupied for r in footprint)
                    or any(r[:2] in factories for r in footprint)):
                continue
            # Arm positions were validated while enumerating paths.
            for r in footprint[len(path):]:
                _position(r)
            extra = {r[:2] for r in footprint} - space
            width, height = (max(hi, max((r[i] for r in extra), default=hi))
                             - min(lo, min((r[i] for r in extra), default=lo)) + 1
                             for i, (lo, hi) in enumerate(bounds))
            choices.append((width * height, len(space) + len(extra), end, terminal, additions, frozenset(footprint)))
        for _, _, end, terminal, additions, cells in sorted(choices, key=lambda c: c[:2]):
            yield distance, end, terminal, additions, cells


def _t_phase(phase):
    phase = Fraction(phase) % 2
    if phase not in {Fraction(n, 4) for n in (1, 3, 5, 7)}:
        raise GeometryError('unsupported_phase', f'Unsupported magic phase {phase}')
    return phase


def _attach_t_resource(nodes, edges, node, phase, name, placement):
    dagger = phase in (Fraction(3, 4), Fraction(7, 4))
    neighbor = next(b if a == node else a for a, b in edges if node in (a, b))
    frame = nodes[neighbor]['tqec']
    distance, end, terminal, additions, _ = placement
    if any(f'{name}_{suffix}' in nodes for suffix in additions):
        raise GeometryError('source_provenance', f'Reserved T gadget node already exists: {name}')
    # Commit only after the entire arm, elbow and endpoint footprint fits.
    nodes[node] = dict(nodes[node], position=end, tqec=terminal)
    for suffix, (position, kind) in additions.items():
        nodes[f'{name}_{suffix}'] = dict(position=position, tqec=kind, other=None)
    arms = [f'{name}_arm_{i}' for i in range(distance)]
    if arms:
        edges.remove((neighbor, node) if (neighbor, node) in edges else (node, neighbor))
        path = [neighbor, *arms, node]
        edges.extend(zip(path, path[1:]))
    if frame[2] == 'Z':
        turn_id, center_id = f'{name}_turn', f'{name}_center'
        edges.extend([(node, turn_id), (turn_id, center_id)])
        node, q, p = center_id, nodes[turn_id]['position'], nodes[center_id]['position']
    else:
        q, p = nodes[arms[-1] if arms else neighbor]['position'], end
    direction = _direction(q, p)
    before, after = f'{name}_source', f'{name}_cap'
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


def add_t_resource(nodes, edges, node, *, phase, name, positions=None):
    """Place one T gadget on its shortest free verified outward arm."""
    phase = _t_phase(phase)
    placement = (next(_t_placements(nodes, edges, node, positions=positions), None)
                 or next(_t_placements(nodes, edges, node, bends=True, positions=positions), None))
    if placement is None:
        raise GeometryError('resource_collision', f'T resource {node!r} has no free outward arm')
    actions = _attach_t_resource(nodes, edges, node, phase, name, placement)
    if positions is not None:
        positions[node] = placement[1]
        positions.update((f'{name}_{suffix}', p) for suffix, (p, _) in placement[3].items())
    return actions


def _match_t_outlets(domains):
    matched: dict[tuple[int, int], int] = {}
    assigned: dict[int, tuple[int, int]] = {}
    for i in domains:
        queue, via = deque([i]), {}
        free = None
        while queue and free is None:
            current = queue.popleft()
            for column in {p[2] for p in domains[current]}:
                if column in via:
                    continue
                via[column] = current
                if column not in matched:
                    free = column
                    break
                # Every owner has exactly one matched column; `via` visits it once.
                queue.append(matched[column])
        if free is None:
            return None
        while free is not None:
            owner = via[free]
            matched[free], assigned[owner], free = owner, free, assigned.get(owner)
    return assigned


def _joint_t_placements(domains, *, max_attempts=100_000):
    ids = count()
    domains = [[(p, frozenset(r[:2] for r in p[-1]), p[3]['source'][0][:2], next(ids)) for p in choices]
               for choices in domains]
    # Distinct factory columns are necessary. Prove this before searching
    # paths, so an impossible straight-arm set falls back to doglegs promptly.
    if _match_t_outlets(dict(enumerate(domains))) is None:
        raise GeometryError('resource_collision', 'T resources have no distinct factory columns')
    # Index conflicts once instead of intersecting every path on each step.
    cells, columns, factories = defaultdict(set), defaultdict(set), defaultdict(set)
    for choices in domains:
        for p, space, outlet, index in choices:
            for r in p[-1]:
                cells[r].add(index)
            for r in space:
                columns[r].add(index)
            factories[outlet].add(index)
    # Leave scarce outlets and their access paths available to other gadgets.
    outlets = Counter(p for choices in domains for p in {choice[2] for choice in choices})
    passages = Counter(p for choices in domains for p in set().union(*(choice[1] for choice in choices)))
    for choices in domains:
        choices.sort(key=lambda p: (sum(outlets[q] for q in p[1]) + passages[p[2]], p[0][0]))
    # Disjoint footprint components must not multiply an unrelated failed search.
    parents = list(range(len(domains)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners: dict[tuple[int, int], int] = {}
    for i, choices in enumerate(domains):
        for pos in set().union(*(choice[1] for choice in choices)):
            parents[root(i)] = root(owners.setdefault(pos, i))
    components = defaultdict(list)
    for i in range(len(domains)):
        components[root(i)].append(i)
    selected: dict[int, Any] = {}
    attempts = 0
    for component in components.values():
        options = {i: domains[i] for i in sorted(component)}
        stack = []
        while options:
            # Each domain already excludes conflicts with earlier choices.
            assigned = _match_t_outlets(options)
            if assigned is not None:
                i = min(options, key=lambda i: len(options[i]))
                choices = sorted(options[i], key=lambda p: p[2] != assigned[i])
                stack.append((i, iter(choices), options))
            while stack:
                i, choices, previous = stack[-1]
                placement = next(choices, None)
                if placement is not None:
                    attempts += 1
                    if attempts > max_attempts:
                        raise GeometryError('resource_placement_limit', f'Joint T placement exceeded {max_attempts} choices')
                    selected[i] = placement[0]
                    blocked = set().union(*(cells[r] for r in placement[0][-1]),
                                          *(factories[r] for r in placement[1]), columns[placement[2]])
                    options = {j: [p for p in choices if p[3] not in blocked]
                               for j, choices in previous.items() if j != i}
                    break
                stack.pop()
                selected.pop(i, None)
            else:
                raise GeometryError('resource_collision', 'T resources have no disjoint verified outward arms')
    return selected


def add_t_resources(nodes, edges, resources):
    """Preserve working placements; jointly replan collisions before committing."""
    resources = [(node, _t_phase(phase), name) for node, phase, name in resources]
    positions = {n: _position(d['position']) for n, d in nodes.items()} if resources else {}
    staged_positions = positions.copy()
    staged, pipes = nodes.copy(), list(edges)
    actions = []
    try:
        for node, phase, name in resources:
            actions.extend(add_t_resource(staged, pipes, node, phase=phase, name=name, positions=staged_positions))
    except GeometryError as error:
        if error.code != 'resource_collision':
            raise
        domains, remaining = [], 100_000
        for node, _, _ in resources:
            domain = list(islice(_t_placements(nodes, edges, node, positions=positions), remaining + 1))
            remaining -= len(domain)
            if remaining < 0:
                raise GeometryError('resource_placement_limit', 'T placement exceeded 100000 candidates')
            domains.append(domain)
        try:
            placements = _joint_t_placements(domains)
        except GeometryError as error:
            if error.code != 'resource_collision':
                raise
            streams = [_t_placements(nodes, edges, node, bends=True,
                                     margin=max(2, isqrt(len(resources)) + 1), positions=positions)
                       for node, _, _ in resources]
            domains = [[] for _ in resources]
            # Try small candidate prefixes; grow only after a bounded solve.
            batch = 4
            while True:
                added = 0
                for domain, stream in zip(domains, streams):
                    choices = list(islice(stream, batch))
                    domain.extend(choices)
                    added += len(choices)
                if not added:
                    placements = _joint_t_placements(domains)
                    break
                if sum(map(len, domains)) > 100_000:
                    raise GeometryError('resource_placement_limit', 'T placement exceeded 100000 candidates')
                try:
                    placements = _joint_t_placements(domains, max_attempts=max(64, 2 * len(resources)))
                    break
                except GeometryError as error:
                    if error.code not in ('resource_collision', 'resource_placement_limit'):
                        raise
                batch *= 2
        staged, pipes, actions = nodes.copy(), list(edges), []
        for i, (node, phase, name) in enumerate(resources):
            actions.extend(_attach_t_resource(staged, pipes, node, phase, name, placements[i]))
    nodes.clear()
    nodes.update(staged)
    edges[:] = pipes
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
        flips = _transverse_flips(nodes, a, b)
        if flips[0] != flips[1]:
            raise GeometryError('pipe_basis', f'Mixed transverse frames on {a!r} -> {b!r}')
        graph.add_pipe(bloq.Pipe(positions[a], _direction(positions[a], positions[b]), hadamard=all(flips)))
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
