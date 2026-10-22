"""Compact resource placement checked by the native physical compiler."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any

import bloq

from ._geometry import (
    GeometryError, _attach_t_resource, _greedy_t_resources, _maze_t_placements, _position,
    _spatial_t_endpoints, _t_phase, build_graph,
)


@dataclass(frozen=True)
class InteropCompilation:
    """The accepted upstream import, validated IR, and placement search record."""

    imported: Any
    program: bloq.Bloq
    placement_attempts: tuple[dict[str, Any], ...]


def _resource_owners(nodes, resources, placements, position_map):
    owners: dict[Any, Any] = {}
    for i, (node, _, name) in enumerate(resources):
        footprint = set(placements[i][-1])
        if (elbow := nodes.get(f'{name}_temporal_elbow')) is not None:
            footprint.add(_position(elbow['position']))
        owners.update((position_map[p], node) for p in footprint if p in position_map)
    return owners


def _conflicting_resources(conflict, owners, centers, columns):
    def owned(role):
        return {owners[tuple(p)] for p in conflict[role]['blocks'] if tuple(p) in owners}

    if conflict['kind'] == 'live_qubit_reuse':
        left, right = owned('producer') | owned('consumer'), owned('reuser')
    elif conflict['kind'] == 'overlapping_node_qubit':
        left, right = owned('first'), owned('second')
    else:
        return set()
    pairs = {frozenset((a, b)) for a in left for b in right
             if a != b and (centers[a] in columns[b] or centers[b] in columns[a])}
    if not pairs:
        # Factory sharing can delay a data consumer: the other resource may
        # be absent from the local native witness. Exclude its current aliases
        # conservatively, then let full native validation decide the new layout.
        pairs = {frozenset((a, b)) for a in left | right for b in centers
                 if a != b and (centers[a] in columns[b] or centers[b] in columns[a])}
    return pairs


def _spatial_cost(nodes):
    points = {tuple(node['position'][:2]) for node in nodes.values()}
    area = 1
    for axis in range(2):
        area *= max(p[axis] for p in points) - min(p[axis] for p in points) + 1
    return area, len(points)


class _Compilation:
    def __init__(self, distance, prepare_t_with_mpps, limits, max_attempts):
        self.context = bloq.CompileContext(distance, prepare_t_with_mpps=prepare_t_with_mpps, limits=limits)
        self.max_attempts = max_attempts
        self.attempts = []
        self.program: bloq.Bloq | None = None

    def build(self, nodes, edges, qubits, actions, resources):
        nodes, edges = nodes.copy(), list(edges)
        resources = [(node, _t_phase(phase), name) for node, phase, name in resources]
        _spatial_t_endpoints(nodes, edges, resources)
        positions = {n: _position(d['position']) for n, d in nodes.items()}
        endpoints = {node for node, _, _ in resources}
        indices = {n: i for i, edge in enumerate(edges) for n in edge if n in endpoints}
        conflicts: dict[Any, set[Any]] = defaultdict(set)
        seen_waiters: set[Any] = set()
        exclusive: set[Any] = set()
        planning_started = perf_counter()
        try:
            greedy = _greedy_t_resources(nodes, edges, resources, positions)
        except GeometryError as error:
            if error.code != 'resource_collision':
                raise
            greedy = None
        last_native_error = None
        owned_failure = private_tried = False
        private_requested = False
        alternate = pending = None

        def route(strategy):
            nonlocal private_tried
            private_tried |= strategy == 'private'
            try:
                placements = (_maze_t_placements(nodes, edges, resources, positions,
                                                 conflicts=conflicts if strategy == 'reuse' else None,
                                                 private=exclusive)
                              if resources else {})
            except GeometryError as error:
                if last_native_error is not None:
                    raise error from last_native_error
                raise
            staged, pipes, staged_actions = nodes.copy(), list(edges), []
            for i, (node, phase, name) in enumerate(resources):
                staged_actions.extend(_attach_t_resource(staged, pipes, node, phase, name, placements[i],
                                                         edge_index=indices[node]))
            return staged, pipes, staged_actions, placements, strategy

        for attempt in range(self.max_attempts):
            started = perf_counter() if attempt else planning_started
            if (pending is None and owned_failure and attempt + 1 == self.max_attempts
                    and alternate is not None):
                pending, alternate, greedy = alternate, None, None
            if pending is not None:
                staged, pipes, staged_actions, placements, strategy = pending
                pending = None
            else:
                # Keep one bounded escape from refinement that discovers aliases
                # serially; private columns still require full native validation.
                strategy = ('private' if private_requested or (owned_failure and not private_tried
                            and attempt + 1 == self.max_attempts) else 'reuse')
                private_requested = False
                try:
                    staged, pipes, staged_actions, placements, strategy = route(strategy)
                except GeometryError as error:
                    if error.code not in ('resource_collision', 'resource_placement_limit'):
                        raise
                    if greedy is None:
                        if not owned_failure or private_tried:
                            raise
                        staged, pipes, staged_actions, placements, strategy = route('private')
                    else:
                        staged, pipes, staged_actions, placements = greedy
                        greedy = alternate = None
                        strategy = 'short_arm'
                else:
                    # A fresh reuse plan supersedes its previously cached plan.
                    if alternate is not None and alternate[-1] == 'reuse':
                        alternate = None
                    if greedy is not None and strategy == 'reuse':
                        alternate = (*greedy, 'short_arm')
                        if _spatial_cost(greedy[0]) <= _spatial_cost(staged):
                            alternate = (staged, pipes, staged_actions, placements, 'reuse')
                            staged, pipes, staged_actions, placements = greedy
                            greedy = None
                            strategy = 'short_arm'
            staged_actions = [*actions, *staged_actions]
            centers = {node: placements[i][3]['source'][0][:2] for i, (node, _, _) in enumerate(resources)}
            columns = {node: {p[:2] for p in placements[i][-1]} for i, (node, _, _) in enumerate(resources)}
            position_map: dict[tuple[int, int, int], tuple[int, int, int]] = {}
            graph, inputs, outputs = build_graph(staged, pipes, qubits, staged_actions,
                                                 defer_analysis=True, position_map=position_map)
            points = graph.flatten().positions()
            extents = [max(p[i] for p in points) - min(p[i] for p in points) + 1 for i in range(3)]
            record = dict(attempt=attempt + 1, strategy=strategy, factory_columns=len(set(centers.values())), extents=extents,
                          occupied_columns=len({p[:2] for p in points}), blocks=len(points),
                          layout_seconds=perf_counter() - started)
            self.attempts.append(record)
            compile_started = perf_counter()
            try:
                program = self.context.compile(graph, validate=True)
            except bloq.CompileError as error:
                last_native_error = error
                record.update(status='rejected', seconds=perf_counter() - started,
                              native_seconds=perf_counter() - compile_started, error=str(error))
                conflict = getattr(error, 'layout_conflict', None)
                pairs = set()
                if conflict is not None:
                    owners = _resource_owners(staged, resources, placements, position_map)
                    for witness in [conflict, *conflict.get('related', ())]:
                        pairs.update(_conflicting_resources(witness, owners, centers, columns))
                    waiting = {owners[tuple(p)]
                               for witness in [conflict, *conflict.get('related', ())]
                               for role in ('producer', 'consumer')
                               for p in witness.get(role, {}).get('blocks', ()) if tuple(p) in owners}
                    # A repeated pending state survived pair refinement. Once
                    # sharing stalls, reserve implicated outlets instead of
                    # discovering their replacement users one at a time.
                    isolated = waiting if exclusive else waiting & seen_waiters
                    new_exclusive = isolated - exclusive
                    exclusive.update(isolated)
                    seen_waiters.update(waiting)
                else:
                    new_exclusive = set()
                involved = (conflict is not None
                            and conflict['kind'] in ('live_qubit_reuse', 'overlapping_node_qubit') and any(
                    tuple(position) in owners
                    for witness in [conflict, *conflict.get('related', ())]
                    for role in ('producer', 'consumer', 'reuser', 'first', 'second')
                    for position in witness.get(role, {}).get('blocks', ())))
                owned_failure |= involved
                if involved and attempt + 1 == self.max_attempts:
                    raise GeometryError('resource_placement_limit',
                                        f'Native-certified T placement exhausted {self.max_attempts} attempts') from error
                new_pairs = [pair for pair in pairs if any(b not in conflicts[a] for a in pair for b in pair if a != b)]
                if not new_pairs and not new_exclusive:
                    if not involved:
                        raise
                    if alternate is not None:
                        pending, alternate, greedy = alternate, None, None
                    elif not private_tried:
                        private_requested = True
                    else:
                        raise GeometryError('resource_placement_limit',
                                            f'Native-certified T placement stalled after {attempt + 1} attempts') from error
                for pair in new_pairs:
                    a, b = pair
                    conflicts[a].add(b)
                    conflicts[b].add(a)
                continue
            if bloq.Bloq.from_binary(program.to_binary()).to_binary() != program.to_binary():
                raise RuntimeError('Validated interop IR failed its binary round trip')
            record.update(status='accepted', seconds=perf_counter() - started,
                          native_seconds=perf_counter() - compile_started)
            self.program = program
            return graph, inputs, outputs
        raise AssertionError('A positive placement attempt budget is required')


def compile_qasm(qasm: str, *, compiler: str, distance: int = 3,
                 prepare_t_with_mpps: bool = False, limits=None, max_placement_attempts: int = 32,
                 **layout_options) -> InteropCompilation:
    """Route TopoLS/SpiderLS once and compile compact, reusable T-resource patches.

    Every returned candidate passes the native compiler's full validation at
    ``distance``. Lifetime conflicts refine resource assignments; neither source
    Z separation nor a successful placement is treated as a reuse certificate.
    Pure graph import remains available through each adapter's ``from_qasm``.
    """
    if compiler not in ('topols', 'spiderls'):
        raise ValueError('compiler must be topols or spiderls')
    if type(max_placement_attempts) is not int or max_placement_attempts < 1:
        raise ValueError('max_placement_attempts must be a positive integer')
    compilation = _Compilation(distance, prepare_t_with_mpps, limits, max_placement_attempts)
    imported = import_module(f'bloq.interop.{compiler}').from_qasm(qasm, _build=compilation.build, **layout_options)
    assert compilation.program is not None
    return InteropCompilation(imported, compilation.program, tuple(compilation.attempts))
