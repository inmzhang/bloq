"""Embed SpiderLS's event DAG on a compact clock with distinct patch lifetimes."""

from collections import defaultdict
from graphlib import CycleError, TopologicalSorter

from bloq.interop._geometry import GeometryError


def _require(condition, message):
    if not condition:
        raise GeometryError('event_order', message)


def _place(nodes, edges, route_times, endpoints, depth):
    """Keep routed layers adjacent unless endpoint cubes need a larger gap."""
    adjacent = defaultdict(list)
    for a, b in edges:
        adjacent[a].append(b)
        adjacent[b].append(a)
    anchors = {n: (time, 0) for n, time in route_times.items()}
    for n, sign in endpoints.items():
        _require(len(adjacent[n]) == 1, f'Boundary {n!r} is not terminal')
        neighbor, = adjacent[n]
        role = nodes[n]['other']
        if isinstance(role, dict) and role['type'] == 'output':
            anchors[n] = (depth + 1, 0)
        elif neighbor in route_times:
            anchors[n] = (route_times[neighbor], sign)
        else:
            anchors[n] = (0 if sign < 0 else depth + 1, 0)
    _require(anchors.keys() == nodes.keys(), 'Some embedded nodes have no clock anchor')
    columns = defaultdict(list)
    for n, data in nodes.items():
        columns[data['position'][:2]].append(n)
    constraints = defaultdict(list)
    for column in columns.values():
        column.sort(key=lambda n: nodes[n]['position'][2])
        for a, b in zip(column, column[1:]):
            ta, oa = anchors[a]
            tb, ob = anchors[b]
            # Endpoints are one tick from their junction: release then prepare
            # between adjacent routed layers requires a gap of three ticks.
            gap = 1 + oa - ob
            _require(ta <= tb and (ta != tb or gap <= 0), f'Conflicting endpoint order: {a!r} -> {b!r}')
            if ta != tb:
                constraints[tb].append((ta, gap))
    clock = [0]
    for time in range(1, depth + 2):
        clock.append(max([clock[-1] + (2 if depth == 0 else 1),
                          *(clock[a] + gap for a, gap in constraints[time])]))
    placed = {n: dict(d, position=(*d['position'][:2], clock[anchors[n][0]] + anchors[n][1]))
              for n, d in nodes.items()}
    occupied = {d['position'] for d in placed.values()}
    _require(len(occupied) == len(placed), 'Embedded endpoints overlap')
    pipes = []
    for a, b in edges:
        p, q = placed[a]['position'], placed[b]['position']
        if p[:2] != q[:2]:
            pipes.append((a, b))
            continue
        _require(p[2] < q[2], f'Patch reverses time: {a!r} -> {b!r}')
        previous = a
        for z in range(p[2] + 1, q[2]):
            n, position = f'world_{a}_{z}', (*p[:2], z)
            _require(n not in placed and position not in occupied, f'Overlapping patch lifetimes at {position}')
            placed[n] = dict(position=position, tqec=nodes[a]['tqec'], other=None)
            occupied.add(position)
            pipes.append((previous, n))
            previous = n
        pipes.append((previous, b))
    return placed, pipes


def diagram(source, target, schedule, routed):
    from pyzx_conversion import routing_to_pipe_inputs, _load_local_pipe_module

    events = {e.dag_node_id: e for e in schedule.zero_time_events}
    _require(None not in events and len(events) == len(schedule.zero_time_events), 'Missing or duplicate event DAG IDs')
    scheduled = [m for slot in schedule.slots for m in slot.measurements]
    measurements = {m.measurement_id: m for m in scheduled}
    _require(len(measurements) == len(scheduled), 'Duplicate scheduled measurement IDs')
    _require(all(m.dag_node_id is not None and m.dag_node_id == m.measurement_id and m.dag_node_id not in events for m in scheduled),
             'Missing, duplicate, or overlapping measurement DAG IDs')
    routed_measurements = [c.measurement_id for layer in routed.layers for c in layer.connections]
    _require(len(routed_measurements) == len(set(routed_measurements))
             and set(routed_measurements) == measurements.keys(), 'Missing, duplicate, or unknown routed measurements')
    placements = [(i, layer.time) for layer in routed.layers for i in layer.event_node_ids_after]
    after = dict(placements)
    _require(len(after) == len(placements), 'An event was routed more than once')
    _require(set(after) <= events.keys(), 'Routing refers to an unknown event')
    _require([layer.time for layer in routed.layers] == list(range(1, len(routed.layers) + 1)),
             'Routed layer times must be contiguous from one')
    grouped = defaultdict(set)
    for i, event in events.items():
        _require(i in after or event.after_time == 0, f'Event {i} has no routed placement')
        _require(after.get(i, 0) >= event.after_time, f'Event {i} precedes its source boundary')
        grouped[after.get(i, 0)].add(i)
    try:
        order = {i: rank for rank, i in enumerate(TopologicalSorter(
            {i: set(e.dependency_node_ids) & events.keys() for i, e in events.items()}
        ).static_order())}
    except CycleError as error:
        raise GeometryError('event_order', 'Zero-time events contain a dependency cycle') from error
    raw = (routing_to_pipe_inputs(routed, include_worldlines=False, include_io_ports=False)
           if routed.layers else {'nodes': {}, 'edges': []})
    nodes_by_layer: dict[int, dict] = defaultdict(dict)
    edges_by_layer = defaultdict(list)
    node_layers = {}
    for n, d in raw['nodes'].items():
        time = d['position'][2]
        nodes_by_layer[time][n] = d
        node_layers[n] = time
    edge_nodes: set[str] = set()
    for a, b in raw['edges']:
        edge_nodes.update((a, b))
        time = node_layers.get(a)
        if time is not None and time == node_layers.get(b):
            edges_by_layer[time].append((a, b))
    nodes, edges = {}, []
    active: dict[int, int] = {}
    previous: dict[int, str] = {}
    ancillas: dict[int, str] = {}
    completed: set[int] = set()
    z_actions = []
    route_times, endpoints, orientations = {}, {}, {}
    index = -1

    def body(orientation):
        # Time-only input/preparation/release cells are Z spiders, including
        # rotated patches. The temporal face X gives the required |+>/X effect.
        return 'ZXX' if orientation == 0 else 'XZX'

    def add(name, position, frame, other=None):
        _require(name not in nodes, f'Duplicate embedded node {name!r}')
        nodes[name] = dict(position=position, tqec=frame, other=other)
        return name

    def connect(a, b):
        p, q = nodes[a]['position'], nodes[b]['position']
        _require(p[:2] == q[:2] and p[2] < q[2], f'Patch moves or overlaps: {a!r} -> {b!r}')
        edges.append((a, b))

    def data(patch, name, frame, other=None):
        _require(patch in routed.data_positions, f'Patch {patch} has no routed position')
        n = add(name, (*routed.data_positions[patch], index), frame, other)
        if patch in previous:
            connect(previous[patch], n)
        previous[patch] = n
        orientations[n] = active[patch]
        return n

    def zero(event):
        nonlocal index
        _require(set(event.dependency_node_ids) <= completed, f'Event {event.dag_node_id} precedes a dependency')
        # This index records order only; _place assigns physical times later.
        index += 1
        patch, kind = event.patch, event.kind
        if kind in ('PREPARE_INPUT', 'PREPARE_PLUS'):
            _require(patch not in active, f'Preparation overwrites active patch {patch}')
            active[patch] = 0
            other = None
            if kind == 'PREPARE_INPUT':
                _require(len(event.vertices) == 1, f'Input patch {patch} has no source spider')
                boundaries = [b for b in source.inputs() if event.vertices[0] in source.neighbors(b)]
                _require(len(boundaries) == 1, f'Input patch {patch} has no unique source boundary')
                other = dict(type='input', qubit=list(source.inputs()).index(boundaries[0]))
            n = data(patch, f'event_{event.dag_node_id}', body(0), other)
            endpoints[n] = -1
        else:
            _require(patch in active, f'{kind} targets inactive patch {patch}')
            _require(kind in ('H', 'H_INPUT', 'H_OUTPUT', 'Z', 'RELEASE'), f'Unsupported event {kind!r}')
            if kind.startswith('H'):
                active[patch] = 90 - active[patch]
            elif kind == 'Z':
                n = previous[patch]
                # Move Z back through pending H gates onto the preceding wire.
                z_actions.append((n, 'Z' if active[patch] == orientations[n] else 'X'))
            elif kind == 'RELEASE':
                n = data(patch, f'event_{event.dag_node_id}', body(active[patch]))
                endpoints[n] = 1
                del active[patch]
                del previous[patch]
        completed.add(event.dag_node_id)

    def zero_group(ids):
        for i in sorted(ids, key=order.__getitem__):
            zero(events[i])

    zero_group(grouped[0])
    for layer in routed.layers:
        index += 1
        prefix = nodes_by_layer[layer.time]
        inactive = {f'q{p.patch_id}_t{layer.time}' for p in layer.patches
                    if p.kind == 'data' and p.patch_id not in active}
        for n, d in prefix.items():
            if n not in inactive:
                add(n, (*d['position'][:2], index), d.get('tqec'), d.get('other'))
                route_times[n] = layer.time
        for a, b in edges_by_layer[layer.time]:
            _require(a not in inactive and b not in inactive, 'A route touches a released data patch')
            edges.append((a, b))
        for patch in layer.patches:
            n = (f'q{patch.patch_id}' if patch.kind == 'data' else f'ancilla_g{patch.gadget_id}') + f'_t{layer.time}'
            if patch.kind == 'data':
                p = patch.patch_id
                if p not in active:
                    continue
                _require(active[p] == patch.orientation, f'Patch {p} orientation disagrees with ordered events')
                if n not in edge_nodes:
                    nodes[n]['tqec'] = body(patch.orientation)
                connect(previous[p], n)
                previous[p] = n
                orientations[n] = active[p]
            else:
                if patch.gadget_id in ancillas:
                    connect(ancillas[patch.gadget_id], n)
                ancillas[patch.gadget_id] = n
        for connection in layer.connections:
            m = measurements[connection.measurement_id]
            _require(m.time <= layer.time and set(m.dependency_node_ids) <= completed,
                     f'Measurement {m.measurement_id} precedes a dependency')
        completed.update(measurements[c.measurement_id].dag_node_id for c in layer.connections)
        zero_group(grouped[layer.time])
    _require(events.keys() | measurements.keys() == completed, 'Some scheduled operations were not embedded')
    index = max(index + 1, 2)
    mapping = dict(target.final_patch_map)
    _require(len(mapping) == len(target.final_patch_map) == len(set(mapping.values())),
             'Final source patches are not unique')
    _require(set(active) == set(mapping.values()), "Active lifetimes disagree with the final source interface")
    for wire, boundary in enumerate(source.outputs()):
        vertex, = source.neighbors(boundary)
        if vertex in mapping:
            patch = mapping[vertex]
            _require(patch in active, f'Output patch {patch} was released')
            n = data(patch, f'output_{wire}', body(active[patch]), dict(type='output', qubit=wire))
            endpoints[n] = 1
    # Full reduction can leave disconnected I/H wires outside target code.
    # Their source boundary order is independent of physical patch IDs.
    for ib, ob, h in target.ignored_direct_io:
        wire, out = list(source.inputs()).index(ib), list(source.outputs()).index(ob)
        x = routed.final_bbox.x_max + 3 + 3 * wire
        a = add(f'direct_in_{wire}', (x, 0, 0), 'ZXX', dict(type='input', qubit=wire))
        b = add(f'direct_out_{out}', (x, 0, index), 'XZX' if int(h) == 2 else 'ZXX', dict(type='output', qubit=out))
        connect(a, b)
        endpoints[a], endpoints[b] = -1, 1
    nodes, edges = _place(nodes, edges, route_times, endpoints, len(routed.layers))
    result = _load_local_pipe_module().create_manual_pipe_diagram(nodes=nodes, edges=edges, require_tqec=False, plot=False)
    return result['bgraph_metadata'], edges, z_actions
