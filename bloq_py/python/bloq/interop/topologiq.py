"""Import audited Topologiq layouts without changing their geometry.

Topologiq is optional. Source gates are checked before its frontend can erase
phases or classical operations; native Bloq validation remains authoritative.
"""

from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata
import json
from numbers import Integral
import re
from typing import Any

import bloq

AUDITED_REVISION = "d010bd3e63686c93c12c922a84a944a3386ac785"
Position = tuple[int, int, int]
_CUBES = {"XZZ", "ZXZ", "ZZX", "ZXX", "XZX", "XXZ"}


class TopologiqImportError(ValueError):
    """A source feature or layout that cannot be imported faithfully."""

    def __init__(self, code: str, message: str, *, nodes: tuple[int, ...] = ()):
        self.code = code
        self.nodes = nodes
        super().__init__(f"{code}: {message}" + (f" (nodes {nodes})" if nodes else ""))


@dataclass(frozen=True)
class TopologiqImport:
    """Native graph and the original circuit's ordered quantum interface."""

    graph: bloq.BlockGraph
    input_ports: tuple[Position, ...]
    output_ports: tuple[Position, ...]
    node_positions: dict[int, Position]
    source_qasm: str
    upstream_revision: str | None
    source_phases: dict[int, Fraction]
    layout_options: dict[str, Any]


def _installed_revision() -> str | None:
    try:
        direct = metadata.distribution("topologiq").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if direct is None:
        return None
    return json.loads(direct).get("vcs_info", {}).get("commit_id")


def _position(value: Any, node: int) -> Position:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(x, bool) or not isinstance(x, Integral) for x in value)
    ):
        raise TopologiqImportError("coordinates", f"Expected an integer grid position, got {value!r}", nodes=(node,))
    return int(value[0]), int(value[1]), int(value[2])


def _direction(start: Position, end: Position) -> str:
    delta = tuple(b - a for a, b in zip(start, end))
    if sum(abs(x) for x in delta) != 1:
        raise TopologiqImportError("pipe_geometry", f"Pipe is not unit axial: {start} -> {end}")
    axis = next(i for i, x in enumerate(delta) if x)
    return ("+" if delta[axis] > 0 else "-") + "XYZ"[axis]


def _routed_path(manager: Any, u: int, v: int) -> list[int]:
    """Follow an original edge through upstream subdivision and spider twins."""
    path = manager.completed_base_edges.get((u, v))
    if path is None:
        path = manager.completed_base_edges.get((v, u))
    if not path or len(path) < 2:
        raise TopologiqImportError("source_provenance", "Missing routed source edge", nodes=(u, v))
    first = set(manager.twin_trace.get(u, [u]))
    last = set(manager.twin_trace.get(v, [v]))
    if path[0] in last and path[-1] in first:
        path = list(reversed(path))
    if path[0] not in first or path[-1] not in last:
        raise TopologiqImportError("source_provenance", "Routed edge no longer matches its source endpoints", nodes=(u, v))
    # Upstream records intermediate IDs from a set. Recover their unique path
    # using actual pipes, rather than interpreting set iteration as wire order.
    members = set(path)
    ordered = [path[0]]
    while ordered[-1] != path[-1]:
        following = [n for n in manager.bgraph.neighbors(ordered[-1]) if n in members and n not in ordered]
        if len(following) != 1:
            raise TopologiqImportError("source_provenance", "Routed source edge is missing or ambiguous", nodes=(u, v))
        ordered.append(following[0])
    if set(ordered) != members:
        raise TopologiqImportError("source_provenance", "Routed source edge contains disconnected members", nodes=(u, v))
    return ordered


def _outgoing_edge(manager: Any, source: Any, root: int) -> tuple[int, int]:
    following = [v for v in source.neighbors(root) if source.row(v) > source.row(root)]
    if len(following) != 1:
        raise TopologiqImportError("source_provenance", "Expected one explicit source continuation", nodes=(root,))
    path = _routed_path(manager, root, following[0])
    return path[0], path[1]


def _lower_clifford_phases(
    manager: Any, source: Any, phases: dict[int, Fraction],
) -> tuple[dict[int, str], list[bloq.Action]]:
    special: dict[int, str] = {}
    actions: list[bloq.Action] = []
    nodes = manager.bgraph.nodes
    for root, phase in phases.items():
        if phase not in (Fraction(1, 2), Fraction(1), Fraction(3, 2)):
            continue
        color = "X" if int(source.type(root)) == 2 else "Z"
        feedback = bool(phase // 1)
        if phase % 1:
            arms = manager.s_gates.get(root)
            if not arms or len(arms) != 1:
                raise TopologiqImportError("unproved_y", "Expected one identified S/Y arm", nodes=(root,))
            y = arms[0][1]
            path = _routed_path(manager, root, y)
            if any(manager.bgraph.degree(v) != 2 for v in path[1:-1]):
                raise TopologiqImportError("unproved_y", "Y arm has a branched intermediate cube", nodes=tuple(path))
            if nodes[y]["zx_block"].kind != "YYO" or manager.bgraph.degree(y) != 1:
                raise TopologiqImportError("unproved_y", "Missing degree-one Y endpoint", nodes=(y,))
            end = _position(nodes[y]["coords"], y)
            neck = _position(nodes[path[-2]]["coords"], path[-2])
            if _direction(neck, end)[1] != "Z":
                raise TopologiqImportError("unproved_y", "Y endpoint is not temporal", nodes=(y,))
            hadamards = sum(manager.bgraph.edges[a, b]["edge_type"] == "HADAMARD" for a, b in zip(path, path[1:]))
            # Native Y preparation and measurement have opposite signed phases.
            feedback ^= (end[2] > neck[2]) ^ bool(hadamards % 2) ^ (color == "X")
            special[y] = "Y"
        if feedback:
            first, second = _outgoing_edge(manager, source, root)
            start = _position(nodes[first]["coords"], first)
            end = _position(nodes[second]["coords"], second)
            actions.append(bloq.Action.feedback([
                bloq.FeedbackTarget(color, start, direction=_direction(start, end)),
            ]))
    return special, actions


def _convert_geometry(
    manager: Any,
    *,
    special_kinds: dict[int, str],
    actions: list[bloq.Action],
) -> tuple[bloq.BlockGraph, dict[int, Position], tuple[Position, ...], tuple[Position, ...]]:
    """Validate the complete upstream snapshot before native construction."""
    nodes = manager.bgraph.nodes
    if not nodes:
        raise TopologiqImportError("empty_layout", "The completed layout is empty")
    inputs, outputs = tuple(manager.inputs), tuple(manager.outputs)
    boundary_ids = inputs + outputs
    if len(set(boundary_ids)) != len(boundary_ids) or not set(boundary_ids) <= set(nodes):
        raise TopologiqImportError("boundaries", "Missing, duplicate, or overlapping boundary IDs")
    positions = {node: _position(data.get("coords"), node) for node, data in nodes(data=True)}
    if len(set(positions.values())) != len(positions):
        raise TopologiqImportError("coordinates", "Multiple nodes occupy the same position")
    z_origin = min(p[2] for p in positions.values())
    shifted = {node: (x, y, z - z_origin) for node, (x, y, z) in positions.items()}
    if any(not -(2**31) <= x < 2**31 for p in shifted.values() for x in p):
        raise TopologiqImportError("coordinates", "Normalized coordinates exceed signed 32-bit range")

    graph = bloq.BlockGraph()
    for node, data in nodes(data=True):
        completion = data.get("completions", {})
        degree = manager.bgraph.degree(node)
        if degree == 0 or completion.get("pending") != 0 or completion.get("degree") != degree:
            raise TopologiqImportError("incomplete_layout", "Incomplete or inconsistent node degree", nodes=(node,))
        block = data["zx_block"]
        kind = block.kind
        if kind in _CUBES:
            minority = "X" if kind.count("X") == 1 else "Z"
            if block.zx_type != minority:
                raise TopologiqImportError("cube_basis", "Cube kind disagrees with its spider type", nodes=(node,))
            native = kind
        elif kind == "OOO" and block.zx_type == "O" and node in boundary_ids:
            if degree != 1:
                raise TopologiqImportError("boundaries", "Open port must have exactly one pipe", nodes=(node,))
            native = "Port"
        elif node in special_kinds:
            native = special_kinds[node]
        else:
            raise TopologiqImportError("unsupported_block", f"No proved lowering for {kind!r}", nodes=(node,))
        if node in boundary_ids and native != "Port":
            raise TopologiqImportError("boundaries", "Declared boundary is not an open port", nodes=(node,))
        role = "input" if node in inputs else "output" if node in outputs else "auto"
        graph.add_block(bloq.Block(shifted[node], native, role=role))

    for u, v, data in manager.bgraph.edges(data=True):
        first, second = sorted((u, v), key=positions.__getitem__)
        start, end = positions[first], positions[second]
        direction = _direction(start, end)
        axis = "XYZ".index(direction[1])
        kind = data.get("kind")
        if not isinstance(kind, str) or len(kind) not in (3, 4) or (len(kind) == 4 and not kind.endswith("H")):
            raise TopologiqImportError("pipe_basis", f"Missing or unknown pipe kind {kind!r}", nodes=(u, v))
        hadamard = kind.endswith("H")
        if kind[:3][axis] != "O" or any(kind[i] not in "XZ" for i in range(3) if i != axis):
            raise TopologiqImportError("pipe_basis", f"Invalid transverse bases {kind!r}", nodes=(u, v))
        if data.get("edge_type") != ("HADAMARD" if hadamard else "SIMPLE"):
            raise TopologiqImportError("pipe_basis", "Hadamard suffix and edge type disagree", nodes=(u, v))
        if tuple(data.get("start_coords") or ()) != start or tuple(data.get("end_coords") or ()) != end:
            raise TopologiqImportError("pipe_geometry", "Stored endpoints disagree with node positions", nodes=(u, v))
        for node, flip in [(first, False), (second, hadamard)]:
            cube = nodes[node]["zx_block"].kind
            if cube not in _CUBES:
                continue
            for i in range(3):
                if i != axis and (cube[i] != kind[i]) != flip:
                    raise TopologiqImportError("pipe_basis", "Pipe frame disagrees with its cube endpoint", nodes=(u, v))
        graph.add_pipe(bloq.Pipe(shifted[first], direction, hadamard=hadamard))

    # Actions arrive in source coordinates, just like upstream geometry.
    graph.set_actions([action.with_shift((0, 0, -z_origin)) for action in actions])
    in_ports = tuple(shifted[node] for node in inputs)
    out_ports = tuple(shifted[node] for node in outputs)
    graph = bloq.BlockGraph.definition(
        "main", graph,
        inputs={f"in_{i}": pos for i, pos in enumerate(in_ports)},
        outputs={f"out_{i}": pos for i, pos in enumerate(out_ports)},
    )
    graph.validate()
    return graph, shifted, in_ports, out_ports


def from_qasm(qasm: str, *, seed: int = 1330) -> TopologiqImport:
    """Compile a static OpenQASM 2 circuit through the audited upstream API.

    Hadamard pipes use only the temporal axis. Unsupported operations and unproved gadget frames raise
    :class:`TopologiqImportError`; no phases, ports, or failures are repaired
    implicitly. Topologiq and PyZX must be installed separately.
    """
    return _from_qasm(qasm, seed=seed, temporal_hadamards_only=True)


def _from_qasm(qasm: str, *, seed: int, temporal_hadamards_only: bool) -> TopologiqImport:
    """The private policy argument supports replaying historical proof fixtures."""
    clean = re.sub(r"//[^\n]*|/\*.*?\*/", "", qasm, flags=re.S)
    if not re.match(r"\s*OPENQASM\s+2(?:\.\d+)?\s*;", clean):
        raise TopologiqImportError("qasm_version", "Expected an OpenQASM 2 source circuit")
    if re.search(r"\b(measure|reset|if)\b", clean):
        raise TopologiqImportError("classical_control", "Only static unitary source circuits are supported")
    try:
        from topologiq.core.graph_manager.graph_manager import BlockGraphManager
        from bloq.interop._topologiq_flow import prepare_graph
        from bloq.interop._qasm import QASMImportError, parse_qasm
    except ImportError as error:
        raise ImportError("Install the pinned Topologiq integration dependencies; base Bloq needs none of them") from error
    try:
        circuit = parse_qasm(clean)
    except QASMImportError as error:
        raise TopologiqImportError(error.code, error.message) from error
    source = circuit.to_graph()
    phases = {}
    normalized = source.copy()
    for node in source.vertices():
        value = source.phase(node)
        if not isinstance(value, (Fraction, int)):
            raise TopologiqImportError("unsupported_phase", "Symbolic source phase", nodes=(node,))
        phase = Fraction(value) % 2
        if phase == 0:
            continue
        if int(source.type(node)) not in (1, 2):
            raise TopologiqImportError("unsupported_phase", "Non-spider source phase", nodes=(node,))
        if phase not in (Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(7, 4)):
            raise TopologiqImportError("unsupported_phase", f"No exact lowering for phase {phase} pi", nodes=(node,))
        if phase.denominator == 4 and int(source.type(node)) != 1:
            raise TopologiqImportError("unsupported_phase", "Quarter phases require a Z spider", nodes=(node,))
        phases[node] = phase
        # This is layout normalization only. Original signs are lowered into
        # native actions below, never inferred from the resulting geometry.
        normalized.set_phase(node, Fraction(1, 4) if phase == Fraction(7, 4) else phase % 1)
    try:
        augmented = prepare_graph(normalized)
    except ValueError as error:
        raise TopologiqImportError("upstream_flow", f"Pauli-flow preparation failed: {error}") from error
    options = dict(graph_traverse_mode="tfs", seed=seed, gravity=7,
                   temporal_hadamards_only=temporal_hadamards_only,
                   z_stretch=2, debug=0, animate=False,
                   # The upstream default can exhaust before reaching long
                   # resource arms imposed by its causal constraints.
                   a_star_max_visits_standard=20_000)
    manager = BlockGraphManager(augmented, **options)
    try:
        manager.build()
    except ValueError as error:
        failed_nodes = tuple(
            int(node) for node in (manager.curr_src_id, manager.curr_tgt_id)
            if isinstance(node, Integral) and not isinstance(node, bool)
        )
        raise TopologiqImportError("upstream_layout", str(error), nodes=failed_nodes) from error
    if tuple(manager.inputs) != tuple(source.inputs()) or tuple(manager.outputs) != tuple(source.outputs()):
        raise TopologiqImportError("boundaries", "Upstream changed the ordered source interface")
    if temporal_hadamards_only:
        for u, v, edge in manager.bgraph.edges(data=True):
            if edge.get("edge_type") == "HADAMARD" and (
                _position(manager.bgraph.nodes[u]["coords"], u)[:2]
                != _position(manager.bgraph.nodes[v]["coords"], v)[:2]
            ):
                raise TopologiqImportError("hadamard_axis", "Upstream violated temporal-only Hadamard routing", nodes=(u, v))
    special, actions = _lower_clifford_phases(manager, source, phases)
    if any(phase.denominator == 4 for phase in phases.values()):
        from bloq.interop._topologiq_t import lower_t_gadgets

        t_kinds, t_actions = lower_t_gadgets(manager, source, phases)
        special.update(t_kinds)
        actions.extend(t_actions)
    graph, positions, inputs, outputs = _convert_geometry(manager, special_kinds=special, actions=actions)
    return TopologiqImport(graph, inputs, outputs, positions, qasm, _installed_revision(), phases, options)
