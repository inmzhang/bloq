"""Import audited TopoLS layouts with explicit S and adaptive T semantics.

TopoLS is optional. Only static Clifford+T circuits are accepted; signed
phases must survive its patched classifier, fusion, and fallback paths.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, sqrt
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

import bloq

from bloq.interop._geometry import (
    GRID_SPACING, GeometryError as TopoLSImportError, Position, _CUBES, _installed_revision,
    _latest_leg, _parse_static_qasm, _position, _direction, _space_diagram, add_t_resources, build_graph, push_spatial_hadamards,
)

AUDITED_REVISION = "e564d350a596cfea3227e09698733210e2e14dcf"


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

    circuit = _parse_static_qasm(qasm)
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


def _convert(embedding: dict[str, Any], qubits: int, *, _build=None) -> tuple[bloq.BlockGraph, tuple[Position, ...], tuple[Position, ...]]:
    nodes, edges = _pipe_diagram(embedding)
    nodes, pipes = _space_diagram(nodes, edges)
    edges = list(pipes)
    neighbors: dict[Any, list[Any]] = {n: [] for n in nodes}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    actions = []
    occupied = {_position(data["position"]) for data in nodes.values()}
    resources = {"S": 0, "T": 0}
    t_resources = []
    # Reserve signed Y caps before placing the movable T endpoints.
    for node, data in list(nodes.items()):
        other = data["other"]
        if isinstance(other, dict):
            continue
        if other not in ("S", "T"):
            if other is not None or data["tqec"] not in _CUBES:
                raise TopoLSImportError("unsupported_block", f"Unknown exported node {node!r}: {data}")
            continue
        resources[other] += 1
        if len(neighbors[node]) != 1:
            raise TopoLSImportError("resource_frame", f"Resource endpoint {node!r} is not terminal")
        roots = [n for n in nodes if str(n) == str(node)[:-2]]
        if len(roots) != 1 or not str(node).endswith("_" + other.lower()):
            raise TopoLSImportError("source_provenance", f"Resource endpoint {node!r} has no source junction")
        root = roots[0]
        allowed = ("1/2", "3/2") if other == "S" else ("1/4", "7/4")
        if embedding["phase_hist"].get(root) not in allowed:
            raise TopoLSImportError("source_provenance", f"Resource endpoint {node!r} has no signed source phase")
        dagger = embedding["phase_hist"][root] == allowed[1]
        if other == "T":
            t_resources.append((node, embedding["phase_hist"][root], f"topols_t_{node}"))
            continue
        p = _position(data["position"])
        neighbor = neighbors[node][0]
        q = _position(nodes[neighbor]["position"])
        direction = _direction(q, p)
        if direction.endswith("Z"):
            data["tqec"] = "Y"
        else:
            axis = "XYZ".index(direction[1])
            kind = nodes[neighbor]["tqec"]
            if kind not in _CUBES or kind[1 - axis] == kind[2]:
                raise TopoLSImportError("resource_frame", f"S endpoint {node!r} has no temporal elbow frame")
            frame = list(kind)
            frame[axis] = frame[2]
            data["tqec"] = "".join(frame)
            available = [(p[0], p[1], p[2] + dz) for dz in (1, -1) if (p[0], p[1], p[2] + dz) not in occupied]
            if not available:
                raise TopoLSImportError("resource_collision", f"S endpoint {node!r} has no free temporal cap")
            cap = available[0]
            _position(cap)
            occupied.add(cap)
            key = f"topols_s_{node}_cap"
            if key in nodes:
                raise TopoLSImportError("source_provenance", f"Reserved endpoint name {key!r} already exists")
            nodes[key] = dict(position=cap, tqec="Y", other=None)
            edges.append((node, key))
            direction = _direction(p, cap)
        # A future Y effect contributes -pi/2. Z on a Z-spider leg
        # supplies pi, giving the required +pi/2 phase.
        if (direction == "+Z") != dagger:
            p = _position(nodes[root]["position"])
            leg = _latest_leg(p, (nodes[n]["position"] for n in neighbors[root]))
            actions.append(bloq.Action.feedback([bloq.FeedbackTarget("Z", p, direction=leg)]))
    for resource, typ in (("S", 4), ("T", 5)):
        if resources[resource] != sum(t == typ for t in embedding["type_hist"].values()):
            raise TopoLSImportError("source_provenance", f"Missing or duplicate {resource} attachment")
    actions = push_spatial_hadamards(nodes, edges, actions=actions)
    if _build is not None:
        return _build(nodes, edges, qubits, actions, t_resources)
    actions.extend(add_t_resources(nodes, edges, t_resources))
    return build_graph(nodes, edges, qubits, actions)


def from_qasm(qasm: str, *, seed: int = 0, engine: str = "python",
              time_bound: float = 0.1, iterations: int = 100, _build=None) -> TopoLSImport:
    """Route a static Clifford+T QASM circuit with the pinned TopoLS API.

    Only S/T endpoint frames covered by native channel tests are accepted.
    Blocked resource arms and unsupported layouts fail without upstream rerouting.
    ``engine='rust'`` requires the matching optional ``topols_core`` extension.
    """
    if engine not in ("python", "rust"):
        raise ValueError("engine must be 'python' or 'rust'")
    # The Rust engine adds one to its u64 seed and parses iterations as u64.
    if (type(seed) is not int or not 0 <= seed < 2**64 - 1
            or type(iterations) is not int or not 0 < iterations < 2**64):
        raise ValueError("Expected a seed in [0, 2**64-2] and iterations in [1, 2**64-1]")
    if type(time_bound) not in (int, float):
        raise ValueError("Expected a finite positive time_bound")
    try:
        time_bound = float(time_bound)
    except OverflowError as error:
        raise ValueError("Expected a finite positive time_bound") from error
    if not 0 < time_bound < float("inf"):
        raise ValueError("Expected a finite positive time_bound")
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
    graph, inputs, outputs = _convert(embedding, qubits, _build=_build)
    revision = _installed_revision("topols")
    options.update(engine=engine, block_size_max=20, zx_opt=1, grid_spacing=GRID_SPACING)
    return TopoLSImport(graph, inputs, outputs, qasm, normalized, revision, options, embedding)
