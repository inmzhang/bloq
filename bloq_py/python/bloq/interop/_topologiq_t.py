"""Signed T-injection recipes for intact, causally embedded Topologiq gadgets."""

from fractions import Fraction
from typing import Any, NoReturn

import bloq


def lower_t_gadgets(
    manager: Any, source_graph: Any, phases: dict[int, Fraction],
) -> tuple[dict[int, str], list[bloq.Action]]:
    """Lower proved Z-quarter-phase gadgets without changing upstream routes.

    ``source_graph`` precedes Topologiq's rewrites and retains dagger phases.
    Degree-two ordinary cubes on a simple gadget arm are identity tensors;
    the last spatial turn chooses the measurement frame, even when temporal
    padding separates it from T. Simple Z-twin trees carry the cap correction.
    """
    from .topologiq import (
        TopologiqImportError, _CUBES, _direction, _outgoing_edge,
        _position, _routed_path,
    )

    roots = {node for node, phase in phases.items() if phase in (Fraction(1, 4), Fraction(7, 4))}
    if set(manager.t_gates) != roots or set(manager.t_zx_tracker) != roots:
        raise TopologiqImportError(
            "t_provenance", "T gadgets do not match the original quarter-phase vertices",
            nodes=tuple(sorted(roots | set(manager.t_gates))),
        )
    if manager.msc_factory:
        raise TopologiqImportError("t_provenance", "Factory-exchanged resource histories are unsupported")

    nodes = manager.bgraph.nodes
    special: dict[int, str] = {}
    actions: list[bloq.Action] = []

    def position(node: int) -> tuple[int, int, int]:
        return _position(nodes[node].get("coords"), node)

    def kind(node: int) -> str:
        return nodes[node]["zx_block"].kind

    def direction(first: int, second: int) -> str:
        return _direction(position(first), position(second))

    def simple(first: int, second: int) -> bool:
        return manager.bgraph.edges[first, second].get("edge_type") == "SIMPLE"

    def fail(root: int, message: str, code: str = "t_control_frame") -> NoReturn:
        raise TopologiqImportError(code, message, nodes=(root,))

    def feedback(first: int, second: int, pauli: str, name: str) -> bloq.Action:
        return bloq.Action.feedback(
            [bloq.FeedbackTarget(pauli, position(first), direction=direction(first, second))],
            condition=name,
        )

    def check_fused_twins(root: int, family: list[int]) -> set[tuple[int, ...]]:
        # Simple identity wires fuse same-color twins back to one Z spider.
        members, originals = set(family), set(source_graph.vertices())
        links: set[tuple[int, ...]] = set()
        reached = {root}
        for first in family:
            for neighbor in manager.bgraph.neighbors(first):
                path = [first, neighbor]
                while path[-1] not in members:
                    node = path[-1]
                    if node in originals or kind(node) not in _CUBES or manager.bgraph.degree(node) != 2:
                        break
                    following = next(v for v in manager.bgraph.neighbors(node) if v != path[-2])
                    if following in path:
                        break
                    path.append(following)
                else:
                    if first == path[-1] or any(not simple(a, b) for a, b in zip(path, path[1:])):
                        fail(root, "Spider twins require simple identity connections", "t_provenance")
                    links.add(min(tuple(path), tuple(reversed(path))))
        for _ in family:
            for link in links:
                if link[0] in reached or link[-1] in reached:
                    reached.update((link[0], link[-1]))
        if reached != members or len(links) != len(family) - 1:
            fail(root, "Spider twins must form one tree of identity wires", "t_provenance")
        return links

    def direct_input_readout(root: int, family: list[int], resource_path: list[int],
                             cap_path: list[int], incoming: list[int], outgoing: tuple[int, int]):
        links = check_fused_twins(root, family)
        tree: dict[int, set[int]] = {node: set() for node in family}
        for link in links:
            for a, b in zip(link, link[1:]):
                tree.setdefault(a, set()).add(b)
                tree.setdefault(b, set()).add(a)
        attachments = [incoming[:2], outgoing, resource_path[:2], cap_path[:2]]
        external = {frozenset(edge) for edge in attachments}
        for node in family:
            for neighbor in manager.bgraph.neighbors(node):
                if neighbor not in tree[node] and frozenset((node, neighbor)) not in external:
                    fail(root, "Z-twin tree has an untracked external leg", "t_provenance")
        for a, b in attachments:
            if a not in family or b in tree:
                fail(root, "Source and gadget legs must leave the Z-twin tree", "t_provenance")
            tree[a].add(b)
            tree[b] = {a}
        # The selector needs input-to-resource X transport. An output-only
        # record can hide an incoming X byproduct and silently invert T.
        route: list[int] = []
        paths: list[tuple[int, list[int]]] = [(incoming[1], [])]
        while paths:
            node, path = paths.pop()
            if node == resource_path[0]:
                route = path + [node]
                break
            paths.extend((neighbor, path + [node]) for neighbor in tree[node]
                         if neighbor not in path)
        for a, b in reversed(list(zip(route, route[1:]))):
            if direction(a, b).endswith("Z") or not simple(a, b) or kind(b)[2] != "X":
                continue
            reached: set[int] = set()
            pending = [incoming[1]]
            while pending:
                node = pending.pop()
                if node in reached:
                    continue
                reached.add(node)
                pending.extend(neighbor for neighbor in tree[node]
                               if frozenset((node, neighbor)) != frozenset((a, b)))
            if any(endpoint in reached for endpoint in (outgoing[1], resource_path[1], cap_path[1])):
                continue
            return bloq.MeasureTarget.edge(position(a), direction(a, b))
        # The input Z transport can continue through an original identity and
        # cross H as X. A time-X Z spider at that turn supplies its X record.
        previous = incoming[-1]
        originals = set(source_graph.vertices())
        if (
            previous in originals and int(source_graph.type(previous)) == 1
            and source_graph.phase(previous) == 0
            and len(tuple(source_graph.neighbors(previous))) == 2
            and manager.twin_trace.get(previous, [previous]) == [previous]
            and all(simple(a, b) for a, b in zip(incoming, incoming[1:]))
            and all(node not in originals and kind(node) in _CUBES
                    and manager.bgraph.degree(node) == 2 for node in incoming[1:-1])
            and manager.bgraph.degree(previous) == 2
        ):
            boundary = next(node for node in source_graph.neighbors(previous) if node != root)
            if boundary in source_graph.inputs():
                _, zwebs, xwebs = manager.aug_zx.pauli_pack
                web = zwebs.get(root)
                path = _routed_path(manager, previous, boundary)
                if (
                    root not in xwebs and web is not None and web.half_edges() == {
                        (boundary, previous): "X", (previous, boundary): "Z",
                        (previous, root): "Z", (root, previous): "Z",
                    }
                    and len(path) >= 3 and path[0] == previous and path[-1] == boundary
                    and int(source_graph.type(boundary)) == 0
                    and all(node not in originals and kind(node) in _CUBES
                            and manager.bgraph.degree(node) == 2 for node in path[1:-1])
                    and not simple(*path[:2])
                    and direction(*path[:2]).endswith("Z")
                    and all(simple(a, b) for a, b in zip(path[1:], path[2:]))
                    and kind(path[1]) in ("ZXX", "XZX")
                    and not direction(path[1], path[2]).endswith("Z")
                ):
                    return position(path[1])
        fail(root, "Direct resource has no proved input-transport readout")

    for root in sorted(roots):
        if (
            int(source_graph.type(root)) != 1
            or Fraction(source_graph.phase(root)) % 2 != phases[root]
            or len(tuple(source_graph.neighbors(root))) != 2
        ):
            fail(root, "Expected an unchanged degree-two Z phase on the original circuit wire", "t_provenance")
        pattern = manager.t_gates[root]
        if len(pattern) != 2 or any(not edge or len(edge) != 2 or edge[0] != root for edge in pattern):
            fail(root, "Malformed resource/conditional association", "t_provenance")
        resource, cap = pattern[0][1], pattern[1][1]
        if manager.t_zx_tracker[root] != cap or resource in special or cap in special:
            fail(root, "Ambiguous resource/conditional ownership", "t_provenance")
        if (
            resource not in nodes or cap not in nodes
            or kind(resource) != "TTO" or kind(cap) not in ("XZ*", "ZX*")
            or manager.bgraph.degree(resource) != 1 or manager.bgraph.degree(cap) != 1
            or manager.msc_cubes.get(resource) != "Mm"
        ):
            fail(root, "Expected an unresolved conditional and its scheduled T resource", "t_provenance")

        resource_path = _routed_path(manager, root, resource)
        cap_path = _routed_path(manager, root, cap)
        family = list(manager.twin_trace.get(root, [root]))
        if (
            not family or family[0] != root or len(set(family)) != len(family)
            or resource_path[0] not in family or cap_path[0] not in family
            or any(node not in nodes or kind(node) not in _CUBES or kind(node).count("Z") != 1 for node in family)
        ):
            fail(root, "Spider twins no longer preserve the source Z junction", "t_provenance")
        for path in (resource_path, cap_path):
            if any(not simple(a, b) for a, b in zip(path, path[1:])):
                fail(root, "Hadamard resource/conditional arms need a separate signed recipe")
            if any(
                node in family or kind(node) not in _CUBES or manager.bgraph.degree(node) != 2
                for node in path[1:-1]
            ):
                fail(root, "Gadget arms must contain only degree-two routing cubes", "t_provenance")
        if direction(resource, resource_path[-2]) != "+Z" or direction(cap_path[-2], cap) != "+Z":
            fail(root, "T resources must face the future and conditional caps the past", "t_causality")
        # Source z is not an execution deadline. Native dependency and lifetime
        # validation decide whether the corrected readout can precede the cap.

        dagger = phases[root] == Fraction(7, 4)
        name = f"topologiq_t_{root}"
        correction: bloq.Action | None = None
        measure: bloq.MeasureTarget | tuple[int, int, int]
        if len(resource_path) == 2:
            # A direct T source occupies the incoming temporal leg of the junction.
            if kind(resource_path[0]) not in ("ZXX", "XZX"):
                fail(root, "Unproved direct-resource junction frame")
            previous = [v for v in source_graph.neighbors(root) if source_graph.row(v) < source_graph.row(root)]
            if len(previous) != 1:
                fail(root, "Expected one explicit source predecessor", "t_provenance")
            incoming = _routed_path(manager, root, previous[0])
            outgoing = _outgoing_edge(manager, source_graph, root)
            measure = direct_input_readout(root, family, resource_path, cap_path, incoming, outgoing)
            correction = feedback(*outgoing, "X", name)
        else:
            check_fused_twins(root, family)
            spatial = [
                (first, second)
                for first, second in zip(resource_path[:-2], resource_path[1:-1])
                if not direction(first, second).endswith("Z")
            ]
            if not spatial:
                fail(root, "Resource arm has no proved spatial injection seam")
            first, corner = spatial[-1]
            tail = resource_path[resource_path.index(corner):]
            # Temporal identity padding can separate the actual turn from T.
            # It does not change the injection frame or move its measurement.
            if any(direction(a, b) != "-Z" for a, b in zip(tail, tail[1:])):
                fail(root, "Resource tail must be a straight future-facing identity wire")
            if kind(corner) in ("XZZ", "ZXZ"):
                measure = position(corner)
            elif kind(corner) in ("ZXX", "XZX"):
                measure = bloq.MeasureTarget.edge(position(first), direction(first, corner))
            else:
                fail(root, "No proved signed recipe for this resource corner frame")
            if not dagger:
                correction = feedback(cap_path[0], cap_path[1], "Z", name)

        special.update({resource: "T", cap: "XY"})
        actions.extend([
            bloq.Action.measure(measure, name),
            bloq.Action.resolve(position(cap), name if dagger else ~bloq.Expr.var(name)),
        ])
        if correction is not None:
            actions.append(correction)
    return special, actions
