"""Embed SpiderLS's event DAG without joining distinct patch lifetimes."""

from collections import defaultdict
from graphlib import CycleError, TopologicalSorter

from bloq.interop._geometry import GeometryError


def _require(condition, message):
    if not condition:
        raise GeometryError('event_order', message)


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
        order = tuple(TopologicalSorter({i: set(e.dependency_node_ids) & events.keys() for i, e in events.items()}).static_order())
    except CycleError as error:
        raise GeometryError('event_order', 'Zero-time events contain a dependency cycle') from error
    raw = (routing_to_pipe_inputs(routed, include_worldlines=False, include_io_ports=False)
           if routed.layers else {'nodes': {}, 'edges': []})
    nodes, edges = {}, []
    active: dict[int, int] = {}
    previous: dict[int, str] = {}
    ancillas: dict[int, str] = {}
    completed: set[int] = set()
    z_actions = []
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
        last = a
        for z in range(p[2] + 1, q[2]):
            n = add(f'world_{a}_{z}', (*p[:2], z), nodes[a]['tqec'])
            edges.append((last, n))
            last = n
        edges.append((last, b))

    def data(patch, name, frame, other=None):
        _require(patch in routed.data_positions, f'Patch {patch} has no routed position')
        n = add(name, (*routed.data_positions[patch], index), frame, other)
        if patch in previous:
            connect(previous[patch], n)
        previous[patch] = n
        return n

    def zero(event):
        nonlocal index
        _require(set(event.dependency_node_ids) <= completed, f'Event {event.dag_node_id} precedes a dependency')
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
            data(patch, f'event_{event.dag_node_id}', body(0), other)
        else:
            _require(patch in active, f'{kind} targets inactive patch {patch}')
            _require(kind in ('H', 'H_INPUT', 'H_OUTPUT', 'Z', 'RELEASE'), f'Unsupported event {kind!r}')
            if kind.startswith('H'):
                active[patch] = 90 - active[patch]
            n = data(patch, f'event_{event.dag_node_id}', body(active[patch]))
            if kind == 'Z':
                z_actions.append(n)
            elif kind == 'RELEASE':
                del active[patch]
                del previous[patch]
        completed.add(event.dag_node_id)

    for i in order:
        if i in grouped[0]:
            zero(events[i])
    for layer in routed.layers:
        index += 1
        prefix = {n: d for n, d in raw['nodes'].items() if d['position'][2] == layer.time}
        inactive = {f'q{p.patch_id}_t{layer.time}' for p in layer.patches
                    if p.kind == 'data' and p.patch_id not in active}
        for n, d in prefix.items():
            if n not in inactive:
                add(n, (*d['position'][:2], index), d.get('tqec'), d.get('other'))
        for a, b in raw['edges']:
            if a in prefix and b in prefix:
                _require(a not in inactive and b not in inactive, 'A route touches a released data patch')
                edges.append((a, b))
        for patch in layer.patches:
            n = (f'q{patch.patch_id}' if patch.kind == 'data' else f'ancilla_g{patch.gadget_id}') + f'_t{layer.time}'
            if patch.kind == 'data':
                p = patch.patch_id
                if p not in active:
                    continue
                _require(active[p] == patch.orientation, f'Patch {p} orientation disagrees with ordered events')
                if not any(n in pair for pair in raw['edges']):
                    nodes[n]['tqec'] = body(patch.orientation)
                connect(previous[p], n)
                previous[p] = n
            else:
                if patch.gadget_id in ancillas:
                    connect(ancillas[patch.gadget_id], n)
                ancillas[patch.gadget_id] = n
        for connection in layer.connections:
            m = measurements[connection.measurement_id]
            _require(m.time <= layer.time and set(m.dependency_node_ids) <= completed,
                     f'Measurement {m.measurement_id} precedes a dependency')
        completed.update(measurements[c.measurement_id].dag_node_id for c in layer.connections)
        for i in order:
            if i in grouped[layer.time]:
                zero(events[i])
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
            data(patch, f'output_{wire}', body(active[patch]), dict(type='output', qubit=wire))
    # Full reduction can leave disconnected I/H wires outside target code.
    # Their source boundary order is independent of physical patch IDs.
    for ib, ob, h in target.ignored_direct_io:
        wire, out = list(source.inputs()).index(ib), list(source.outputs()).index(ob)
        x = routed.final_bbox.x_max + 3 + 3 * wire
        a = add(f'direct_in_{wire}', (x, 0, 0), 'ZXX', dict(type='input', qubit=wire))
        b = add(f'direct_out_{out}', (x, 0, index), 'XZX' if int(h) == 2 else 'ZXX', dict(type='output', qubit=out))
        connect(a, b)
    result = _load_local_pipe_module().create_manual_pipe_diagram(nodes=nodes, edges=edges, require_tqec=False, plot=False)
    return result['bgraph_metadata'], edges, z_actions
