"""Import audited TopoLS layouts with explicit S and adaptive T semantics.

TopoLS is optional. Only static Clifford+T circuits are accepted; signed
phases must survive its patched classifier, fusion, and fallback paths.
"""

from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata
import json
from math import ceil, sqrt
from numbers import Integral
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import bloq

AUDITED_REVISION = "e564d350a596cfea3227e09698733210e2e14dcf"
GRID_SPACING = 3
Position = tuple[int, int, int]
_CUBES = {"XZZ", "ZXZ", "ZZX", "ZXX", "XZX", "XXZ"}


class TopoLSImportError(ValueError):
    """A source feature or layout that cannot be imported faithfully."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class TopoLSImport:
    """Native graph, ordered source interface, and reproducible layout inputs."""

    graph: bloq.BlockGraph
    input_ports: tuple[Position, ...]
    output_ports: tuple[Position, ...]
    source_qasm: str
    normalized_qasm: str
    upstream_revision: str | None
    layout_options: dict[str, Any]
    embedding: dict[str, Any]


def _normalize_qasm(qasm: str) -> tuple[str, int]:
    from pyzx.circuit.gates import CNOT, CZ, HAD, XPhase, ZPhase
    from bloq.interop._qasm import QASMImportError, parse_qasm

    clean = re.sub(r"//[^\n]*|/\*.*?\*/", "", qasm, flags=re.S)
    if not re.match(r"\s*OPENQASM\s+2(?:\.\d+)?\s*;", clean):
        raise TopoLSImportError("qasm_version", "Expected an OpenQASM 2 source circuit")
    if re.search(r"\b(measure|reset|if)\b", clean):
        raise TopoLSImportError("classical_control", "Only static unitary source circuits are supported")
    try:
        circuit = parse_qasm(clean)
    except QASMImportError as error:
        raise TopoLSImportError(error.code, error.message) from error
    lines = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{circuit.qubits}];']
    for gate in circuit.gates:
        if isinstance(gate, (ZPhase, XPhase)):
            phase = gate.phase
            if not isinstance(phase, (Fraction, int)) or (Fraction(phase) * 4).denominator != 1:
                raise TopoLSImportError("unsupported_phase", f"No exact Clifford+T lowering for {gate}")
            quarters = int(phase * 4) % 8
            if quarters == 0:
                continue
            x_axis = isinstance(gate, XPhase)
            if x_axis:
                lines.append(f'h q[{gate.target}];')
            signed = quarters if quarters <= 4 else quarters - 8
            s_gate, t_gate = ("s", "t") if signed > 0 else ("sdg", "tdg")
            lines.extend([f'{s_gate} q[{gate.target}];'] * (abs(signed) // 2))
            if signed % 2:
                lines.append(f'{t_gate} q[{gate.target}];')
            if x_axis:
                lines.append(f'h q[{gate.target}];')
        elif isinstance(gate, HAD):
            lines.append(f'h q[{gate.target}];')
        elif isinstance(gate, (CNOT, CZ)):
            if isinstance(gate, CZ):
                lines.append(f'h q[{gate.target}];')
            lines.append(f'cx q[{gate.control}],q[{gate.target}];')
            if isinstance(gate, CZ):
                lines.append(f'h q[{gate.target}];')
        else:
            raise TopoLSImportError("unsupported_gate", f"No exact lowering for {gate}")
    if circuit.qubits < 1:
        raise TopoLSImportError("boundaries", "The source has no quantum interface")
    if len(lines) == 3:
        raise TopoLSImportError("identity_layout", "TopoLS does not preserve distinct ports for identity-only sources")
    return '\n'.join(lines) + '\n', circuit.qubits


def _position(value: Any) -> Position:
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in value)):
        raise TopoLSImportError("coordinates", f"Expected an integer grid position, got {value!r}")
    result = (int(value[0]), int(value[1]), int(value[2]))
    if any(not -(2**31) <= x < 2**31 for x in result):
        raise TopoLSImportError("coordinates", "Coordinates exceed signed 32-bit range")
    return result


def _direction(start: Position, end: Position) -> str:
    delta = tuple(b - a for a, b in zip(start, end))
    if sum(abs(x) for x in delta) != 1:
        raise TopoLSImportError("pipe_geometry", f"Pipe is not unit axial: {start} -> {end}")
    axis = next(i for i, x in enumerate(delta) if x)
    return ("+" if delta[axis] > 0 else "-") + "XYZ"[axis]


def _phase_history(prep: Any, types: dict[Any, int]) -> dict[Any, str]:
    """Bind each actual resource to its exact prepared or fallback phase.

    TopoLS moves real junctions to ``<id>_old`` and uses ``<id>_<block>``
    for reparsed fallback blocks. Signs are read from those source graphs,
    never inferred from geometry or reconstructed from a gate count.
    """
    from topols.pipeline import fallback_block

    graphs = {}
    phases = {}
    for node, resource in types.items():
        if resource not in (4, 5):
            continue
        match = re.fullmatch(r"(\d+)(?:_(\d+))?(?:_old)?", str(node))
        if match is None:
            raise TopoLSImportError("source_provenance", f"Unknown resource history ID {node!r}")
        vertex = int(match[1])
        graph = prep.graph
        if match[2] is not None:
            block = int(match[2])
            if block not in prep.block_info:
                raise TopoLSImportError("source_provenance", f"Unknown fallback block for {node!r}")
            if block not in graphs:
                graphs[block] = fallback_block(prep.circuit, block, prep.block_info, prep.idx_to_row)[0]
            graph = graphs[block]
        if vertex not in graph.vertices() or int(graph.type(vertex)) != 1:
            raise TopoLSImportError("source_provenance", f"Missing source Z phase for {node!r}")
        phase = Fraction(graph.phase(vertex)) % 2
        allowed = (Fraction(1, 2), Fraction(3, 2)) if resource == 4 else (Fraction(1, 4), Fraction(7, 4))
        if phase not in allowed:
            raise TopoLSImportError("source_provenance", f"Resource {node!r} disagrees with source phase {phase}")
        phases[node] = str(phase)
    return phases


def _pipe_diagram(embedding: dict[str, Any]):
    from topols.export import bgraph as export

    pos, ori, typ = (dict(embedding[key]) for key in ("pos_hist", "ori_hist", "type_hist"))
    real = {node for node, t in typ.items() if t in (0, 1, 4, 5)}
    if not pos or set(pos) != set(typ) or not real <= set(ori):
        raise TopoLSImportError("incomplete_layout", "Position, orientation, and type histories differ")
    pos = {node: _position(p) for node, p in pos.items()}
    owners: dict[Position, Any] = {}
    for node, p in pos.items():
        if p in owners and (typ[node] not in (2, 3) or typ[owners[p]] not in (2, 3)):
            raise TopoLSImportError("coordinates", "Multiple real history nodes occupy the same position")
        owners[p] = node  # Idle-chain aliases denote one wire, not extra cubes.
    pos = {node: p for p, node in owners.items()}
    if any(type(t) is not int or t not in range(6) for t in typ.values()) or any(o not in ("i", "j", "k") for o in ori.values()):
        raise TopoLSImportError("cube_basis", "Unknown history type or orientation")
    paths = [tuple(_position(p) for p in path) for path in embedding["path_hist"]]
    for path in paths:
        if len(path) < 2 or len(set(path)) != len(path):
            raise TopoLSImportError("pipe_geometry", "A path is empty, singular, or self-intersecting")
        for a, b in zip(path, path[1:]):
            _direction(a, b)
    # Validate before the upstream exporter can truncate coordinates or silently
    # ignore invalid endpoints. Identical historical paths are idempotent.
    paths = export.remove_duplicate_paths(paths)
    paths = export.merge_idle_paths(paths, pos, typ)
    paths = export.remove_duplicate_paths(paths)
    nodes = export.combine_metadata(pos, export.build_tqec_type(ori, typ), embedding["io_info"])
    paths, invalid, t_nodes = export.check_paths_endpoints(paths, pos, typ, schedule_t=0)
    if invalid:
        raise TopoLSImportError("incomplete_layout", f"Invalid routed endpoints: {invalid}")
    export.add_missing_endpoint_nodes(paths, pos, typ, nodes)
    edges, reverse = export.get_edge(pos, paths)
    nodes, edges = export.edge_process(edges, nodes, reverse, ori, typ, t_nodes)
    edges = export.remove_duplicate_geometric_edges(edges)
    return nodes, edges


def _convert(embedding: dict[str, Any], qubits: int) -> tuple[bloq.BlockGraph, tuple[Position, ...], tuple[Position, ...]]:
    nodes, edges = _pipe_diagram(embedding)
    nodes, edges = _space_diagram(nodes, edges)
    positions = {node: _position(data["position"]) for node, data in nodes.items()}
    if len(set(positions.values())) != len(positions):
        raise TopoLSImportError("coordinates", "Exported nodes overlap")
    kinds = {node: data["tqec"] for node, data in nodes.items()}
    neighbors: dict[Any, list[Any]] = {node: [] for node in nodes}
    for (a, b), (p, q) in edges.items():
        if a not in nodes or b not in nodes or (positions[a], positions[b]) != (tuple(p), tuple(q)):
            raise TopoLSImportError("pipe_geometry", "Exported edge endpoints disagree with node positions")
        _direction(positions[a], positions[b])
        neighbors[a].append(b)
        neighbors[b].append(a)
    ports: dict[str, dict[int, Position]] = {"input": {}, "output": {}}
    actions = []
    extra: list[tuple[Position, str]] = []
    extra_pipes: list[tuple[Position, str]] = []
    occupied = set(positions.values())
    resources = {"S": 0, "T": 0}
    phases = embedding["phase_hist"]
    for node, data in nodes.items():
        p, other = positions[node], data["other"]
        if isinstance(other, dict):
            role, wire = other.get("type"), other.get("qubit")
            if (role not in ports or type(wire) is not int or not 0 <= wire < qubits
                    or wire in ports[role] or len(neighbors[node]) != 1):
                raise TopoLSImportError("boundaries", f"Invalid or duplicate port {node!r}: {other}")
            ports[role][wire] = p
            kinds[node] = "Port"
        elif other in ("S", "T"):
            resources[other] += 1
            if len(neighbors[node]) != 1:
                raise TopoLSImportError("resource_frame", f"Resource endpoint {node!r} is not terminal")
            neighbor = neighbors[node][0]
            q = positions[neighbor]
            direction = _direction(q, p)
            roots = [n for n in nodes if str(n) == str(node)[:-2]]
            if len(roots) != 1 or not str(node).endswith("_" + other.lower()):
                raise TopoLSImportError("source_provenance", f"Resource endpoint {node!r} has no source junction")
            root = roots[0]
            allowed = ("1/2", "3/2") if other == "S" else ("1/4", "7/4")
            if phases.get(root) not in allowed:
                raise TopoLSImportError("source_provenance", f"Resource endpoint {node!r} has no signed source phase")
            dagger = phases[root] == allowed[1]
            if other == "S":
                if direction.endswith("Z"):
                    kinds[node] = "Y"
                else:
                    axis = "XYZ".index(direction[1])
                    kind = kinds[neighbor]
                    if kind not in _CUBES or kind[1 - axis] == kind[2]:
                        raise TopoLSImportError("resource_frame", f"S endpoint {node!r} has no temporal elbow frame")
                    # Turn a spatial terminal wire into a temporal Y cap. The
                    # extra degree-two cube is an identity; no source phase or
                    # existing pipe is changed.
                    frame = list(kind)
                    frame[axis] = frame[2]
                    kinds[node] = "".join(frame)
                    caps = [(p[0], p[1], p[2] + dz) for dz in (1, -1)]
                    available = [cap for cap in caps if cap not in occupied]
                    if not available:
                        raise TopoLSImportError("resource_collision", f"S endpoint {node!r} has no free temporal cap")
                    cap = available[0]
                    occupied.add(cap)
                    extra.append((cap, "Y"))
                    direction = _direction(p, cap)
                    extra_pipes.append((p, direction))
                # The future Y effect contributes -pi/2. Z on any one leg of
                # its Z spider supplies pi, giving the required +pi/2 phase.
                if (direction == "+Z") != dagger:
                    leg = max(neighbors[root], key=lambda n: positions[n][2])
                    actions.append(bloq.Action.feedback([bloq.FeedbackTarget(
                        "Z", positions[root], direction=_direction(positions[root], positions[leg]),
                    )]))
            else:
                axis = "XYZ".index(direction[1])
                if axis == 2 or kinds[neighbor] not in _CUBES:
                    raise TopoLSImportError("resource_frame", f"T endpoint {node!r} has no proved spatial ZZ seam")
                if kinds[neighbor][2] == "Z":
                    # A degree-two, phase-zero elbow transports the resource
                    # to the canonical ZZ injection frame without changing any
                    # existing cell or pipe. Use the other spatial axis after
                    # one temporal step, so all three pipe axes occur once.
                    kinds[node] = "ZXZ" if axis == 0 else "XZZ"
                    turn = (p[0], p[1], p[2] + 1)
                    center = (p[0] + (axis == 1), p[1] + (axis == 0), p[2] + 1)
                    canonical = "ZXX" if axis == 0 else "XZX"
                    for cell in (turn, center):
                        if cell in occupied:
                            raise TopoLSImportError("resource_collision", f"T elbow for {node!r} collides at {cell}")
                        occupied.add(cell)
                    extra.extend([(turn, canonical), (center, canonical)])
                    extra_pipes.extend([(p, "+Z"), (turn, "+Y" if axis == 0 else "+X")])
                    p, q = center, turn
                    direction = "+Y" if axis == 0 else "+X"
                else:
                    kinds[node] = "XZX" if axis == 0 else "ZXX"
                before, after = (p[0], p[1], p[2] - 1), (p[0], p[1], p[2] + 1)
                for cell in (before, after):
                    _position(cell)
                    if cell in occupied:
                        raise TopoLSImportError("resource_collision", f"T column for {node!r} collides at {cell}")
                    occupied.add(cell)
                extra.extend([(before, "T"), (after, "XY")])
                extra_pipes.extend([(before, "+Z"), (p, "+Z")])
                name = f"topols_t_{node}"
                actions.extend([
                    bloq.Action.measure(bloq.MeasureTarget.edge(q, direction), name),
                    bloq.Action.resolve(after, bloq.Expr.var(name) if dagger else ~bloq.Expr.var(name)),
                ])
        elif other is not None or kinds[node] not in _CUBES:
            raise TopoLSImportError("unsupported_block", f"Unknown exported node {node!r}: {data}")
    for resource, typ in (("S", 4), ("T", 5)):
        expected = sum(t == typ for t in embedding["type_hist"].values())
        if resources[resource] != expected:
            raise TopoLSImportError("source_provenance", f"Missing or duplicate {resource} attachment")
    if any(set(mapping) != set(range(qubits)) for mapping in ports.values()):
        raise TopoLSImportError("boundaries", "Exported interface does not cover every source qubit")
    graph = bloq.BlockGraph()
    for node, p in positions.items():
        other = nodes[node]["other"]
        graph.add_block(bloq.Block(p, kinds[node], role=other["type"] if isinstance(other, dict) else "auto"))
    for p, kind in extra:
        graph.add_block(bloq.Block(p, kind))
    for (a, b) in edges:
        p, q = positions[a], positions[b]
        direction = _direction(p, q)
        axis = "XYZ".index(direction[1])
        # Port stub colours carry boundary Hadamards even though the native
        # port is open. Never discard those colours before comparing frames.
        ka, kb = nodes[a]["tqec"] or kinds[a], nodes[b]["tqec"] or kinds[b]
        flips = [ka[i] != kb[i] for i in range(3) if i != axis] if ka in _CUBES and kb in _CUBES else [False, False]
        if flips[0] != flips[1]:
            raise TopoLSImportError("pipe_basis", f"Mixed transverse frames on {a!r} -> {b!r}")
        graph.add_pipe(bloq.Pipe(p, direction, hadamard=all(flips)))
    for p, direction in extra_pipes:
        graph.add_pipe(bloq.Pipe(p, direction))
    graph.set_actions(actions)
    inputs, outputs = (tuple(ports[role][i] for i in range(qubits)) for role in ("input", "output"))
    graph = bloq.BlockGraph.definition("main", graph,
        inputs={f"in_{i}": p for i, p in enumerate(inputs)},
        outputs={f"out_{i}": p for i, p in enumerate(outputs)})
    graph.validate()
    return graph, inputs, outputs


def _space_diagram(nodes, edges):
    """Leave a local resource collar while preserving topology and H parity.

    Every original unit pipe becomes three unit pipes. Its inserted cubes
    have degree two and phase zero (identity tensors); any H stays on the
    middle segment. Endpoint gadgets then fit between original grid cells.
    """
    spaced = {node: dict(data, position=tuple(GRID_SPACING * x for x in data["position"]))
              for node, data in nodes.items()}
    pipes = {}
    for index, ((a, b), _) in enumerate(edges.items()):
        p, q = spaced[a]["position"], spaced[b]["position"]
        step = tuple((y - x) // GRID_SPACING for x, y in zip(p, q))
        ka, kb = nodes[a]["tqec"], nodes[b]["tqec"]
        if ka not in _CUBES and kb not in _CUBES:
            raise TopoLSImportError("pipe_basis", "A resource wire has no ordinary frame")
        previous = a
        for distance in range(1, GRID_SPACING):
            node = f"bloq_spacing_{index}_{distance}"
            if node in spaced:
                raise TopoLSImportError("source_provenance", "Reserved subdivision node ID already exists")
            frame = ka if distance == 1 else kb
            if frame is None:
                frame = kb if ka is None else ka
            position = tuple(x + distance * dx for x, dx in zip(p, step))
            spaced[node] = dict(position=position, tqec=frame, other=None)
            pipes[previous, node] = (spaced[previous]["position"], position)
            previous = node
        pipes[previous, b] = (spaced[previous]["position"], q)
    return spaced, pipes


def from_qasm(qasm: str, *, seed: int = 0, engine: str = "python",
              time_bound: float = 0.1, iterations: int = 100) -> TopoLSImport:
    """Route a static Clifford+T QASM circuit with the pinned TopoLS API.

    Only S/T endpoint frames covered by native channel tests are accepted.
    Resource-column collisions and unsupported layouts fail without rerouting.
    ``engine='rust'`` requires the matching optional ``topols_core`` extension.
    """
    if engine not in ("python", "rust"):
        raise ValueError("engine must be 'python' or 'rust'")
    if type(seed) is not int or type(iterations) is not int or iterations < 1 or not 0 < time_bound < float("inf"):
        raise ValueError("Expected an integer seed, positive iterations, and a finite positive time_bound")
    normalized, qubits = _normalize_qasm(qasm)
    from topols.pipeline import prepare_graph

    options: dict[str, Any] = dict(seed_init=seed, seed_step=1, time_bound=time_bound,
                   iter_num=iterations, move_num=6, length=ceil(sqrt(qubits)),
                   dir_opt=True, backtrack=0, z_floor=1)
    with TemporaryDirectory(prefix="bloq-topols-") as temporary:
        path = Path(temporary) / "source.qasm"
        path.write_text(normalized, encoding="utf-8")
        prep = prepare_graph(path, block_size_max=20, zx_opt=1, dir_opt=1)
    from topols.zx_transform.layering import node_type_convert

    for node in prep.graph.vertices():
        if int(prep.graph.type(node)) == 1 and prep.graph.phase(node) and node_type_convert(prep.graph, node) not in (4, 5):
            raise TopoLSImportError("upstream_patch", "Install bloq_py/examples/topols.patch to preserve signed S/T phases")
    if engine == "rust":
        from topols.engine import run_rust

        pos, ori, typ, paths, io, _, _ = run_rust(prep, options)
    else:
        from topols.driver import operation

        _, pos, ori, paths, typ = operation(
            prep.circuit, prep.graph, prep.layer_labels, prep.layer_to_block, prep.block_info,
            prep.idx_to_row, prep.rows, prep.q_num, z_floor=1,
            seed_init_tuple=(seed, 1), time_bound=time_bound, iter_num=iterations,
            move_num=6, length=options["length"], dir_opt=1,
            hadamard_edges=prep.h_table, io_info=prep.io_info)
        io = prep.io_info
    phases = _phase_history(prep, typ)
    for gate, phase in (("s", "1/2"), ("sdg", "3/2"), ("t", "1/4"), ("tdg", "7/4")):
        if normalized.count(f"\n{gate} q[") != sum(p == phase for p in phases.values()):
            raise TopoLSImportError("source_provenance", f"Layout lost or duplicated signed {gate.upper()} resources")
    embedding = dict(pos_hist=pos, ori_hist=ori, type_hist=typ, path_hist=paths, io_info=io, phase_hist=phases)
    graph, inputs, outputs = _convert(embedding, qubits)
    try:
        direct = metadata.distribution("topols").read_text("direct_url.json")
        revision = json.loads(direct).get("vcs_info", {}).get("commit_id") if direct else None
    except metadata.PackageNotFoundError:
        revision = None
    options.update(engine=engine, block_size_max=20, zx_opt=1, grid_spacing=GRID_SPACING)
    return TopoLSImport(graph, inputs, outputs, qasm, normalized, revision, options, embedding)
