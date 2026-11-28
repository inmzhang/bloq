"""Certify a noisy Bloq Stim circuit and save or verify its distance certificate.

Add a checkout of Fanerst/tropidist to PYTHONPATH. Run
    python certify.py circuit.stim certificate.json
    python certify.py circuit.stim certificate.json --verify

The discovery step uses single-Pauli faults to propose groups of detectors.
The proof and its independent verification always use every column of the
original undecomposed DEM. For each observable, add detector rows to its
logical parity and XOR groups of detector checks. Every undetectable fault set
flipping the original bit still flips the new bit with zero grouped syndrome.
Tropidist's dense rank DP therefore gives a lower bound. The physical replay
gives an upper bound, and equality certifies the distance.
"""

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import stim
from cdist import auto, problem, replay


def masks(op):
    detectors = []
    logical = 0
    for target in op.targets_copy():
        if target.is_relative_detector_id():
            detectors.append(target.val)
        elif target.is_logical_observable_id():
            logical ^= 1 << target.val
    return detectors, logical


def propose_sectors(circuit):
    singles = stim.Circuit()
    for op in circuit.flattened():
        if op.name in {"DEPOLARIZE1", "DEPOLARIZE2"}:
            for gate in ("X_ERROR", "Z_ERROR"):
                singles.append(gate, op.targets_copy(), 0.001)
        else:
            singles.append(op)
    edges = [
        masks(op)
        for op in singles.detector_error_model().flattened()
        if op.type == "error"
    ]
    parents = list(range(circuit.num_detectors))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for detectors, _ in edges:
        if len(detectors) == 2:
            a, b = detectors
            parents[root(a)] = root(b)
    labels = [root(i) for i in range(len(parents))]
    logicals = defaultdict(int)
    for detectors, logical in edges:
        for d in detectors:
            logicals[labels[d]] |= logical
    return labels, logicals


def detector_ids(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def proposed_reduction(full, bit, labels, logicals):
    active = [i for i, label in enumerate(labels) if logicals[label] & (1 << bit)]
    sector = sum(1 << i for i in active)
    interior = defaultdict(list)
    boundary = set()
    for h, logical in {
        (h & sector, (logical >> bit) & 1) for h, logical in zip(full.h, full.l)
    } - {(0, 0)}:
        ends = list(detector_ids(h))
        if len(ends) == 1:
            boundary.add((ends[0], logical))
        elif len(ends) == 2:
            a, b = ends
            interior[a].append((b, logical))
            interior[b].append((a, logical))
        # Higher-degree columns are omitted only from the proposed graph.
        # Every projected distance calculation retains all original columns.
    signs = {}
    for start in interior:
        if start in signs:
            continue
        signs[start] = 0
        todo = [start]
        while todo:
            node = todo.pop()
            for neighbor, parity in interior[node]:
                sign = signs[node] ^ parity
                if neighbor not in signs:
                    signs[neighbor] = sign
                    todo.append(neighbor)
                elif signs[neighbor] != sign:
                    raise ValueError("Interior contains a logical cycle")
    distances = {n: 1 for n, logical in boundary if signs.get(n, 0) == logical}
    todo = deque(distances)
    while todo:
        node = todo.popleft()
        for neighbor, _ in interior[node]:
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                todo.append(neighbor)
    length = min(
        distances[n] + 1
        for n, logical in boundary
        if n in distances and signs.get(n, 0) != logical
    )
    groups = [[] for _ in range(length - 1)]
    for node in active:
        groups[min(distances.get(node, length - 1), length - 1) - 1].append(node)
    return {
        "groups": groups,
        "add_to_observable": sorted(n for n, sign in signs.items() if sign),
    }


def lower_bound(full, bit, record):
    groups = record["groups"]
    ids = [i for group in groups for i in group]
    add = record["add_to_observable"]
    if len(set(ids)) != len(ids) or len(set(add)) != len(add):
        raise ValueError("Repeated detector in reduction")
    if any(i < 0 or i >= full.nrows for i in ids + add):
        raise ValueError("Unknown detector")
    mask = sum(1 << i for i in add)
    logical = [
        ((value >> bit) & 1) ^ ((h & mask).bit_count() % 2)
        for value, h in zip(full.l, full.h, strict=True)
    ]
    query = problem.Problem("logical", full.nrows, 1, full.h, logical, full.row_meta)
    reduced = problem.project_rows(
        query, groups, "original fault columns", list(range(len(groups)))
    )
    order = list(range(reduced.ncols))
    if record["order"] == "spatial":
        order.sort(key=lambda j: reduced.h[j].bit_length())
    elif record["order"] != "natural":
        raise ValueError("Unknown column order")
    return auto.dp_distance(reduced, order, max_table_bytes=2**24)[0]


def discover_bound(circuit, full, bit, distance, labels, logicals):
    initial = proposed_reduction(full, bit, labels, logicals)
    coords = circuit.get_detector_coordinates()
    # Distinguish transverse positions within a graph-distance layer when a
    # hyperedge could otherwise cancel there. The exact DP decides whether
    # the proposed projection is sufficient.
    for mode in (
        "base",
        "x",
        "y",
        "min",
        "max",
        "sum",
        "diff",
        "bend",
        "xy",
        "rotated",
    ):
        for cross_first in [False] if mode == "base" else [False, True]:
            groups = defaultdict(list)
            for layer, rows in enumerate(initial["groups"]):
                for i in rows:
                    x, y = (coords[i] + [0, 0])[:2]
                    cross = {
                        "base": 0,
                        "x": x,
                        "y": y,
                        "min": min(x, y),
                        "max": max(x, y),
                        "sum": x + y,
                        "diff": x - y,
                        "bend": x if x > 2 * distance else y,
                        "xy": (x, y),
                        "rotated": (x - y, x + y),
                    }[mode]
                    key = (cross, layer) if cross_first else (layer, cross)
                    groups[key].append(i)
            candidate = {
                "groups": [rows for _, rows in sorted(groups.items())],
                "add_to_observable": initial["add_to_observable"],
                "order": "natural" if mode == "base" else "spatial",
            }
            try:
                value = lower_bound(full, bit, candidate)
            except MemoryError:
                continue
            if value == distance:
                return candidate
    raise ValueError(f"No sufficient projection found for observable {bit}")


def replay_frames(circuit, faults):
    """Replay physical locations in Stim, including Pauli-target observable boundaries."""
    ops = replay._expand(circuit)
    by_instruction = defaultdict(list)
    flipped_records = set()
    for fault in faults:
        index = replay._flat_index(circuit, fault["stack"])
        if ops[index].name != fault["instruction"]:
            raise ValueError("Fault location differs from original instruction")
        if fault["paulis"]:
            by_instruction[index].extend(fault["paulis"])
        else:
            flipped_records.symmetric_difference_update([fault["flipped_measurement"]])
    injected = stim.Circuit()
    record = 0
    for i, op in enumerate(ops):
        for pauli, q in by_instruction[i]:
            injected.append(pauli + "_ERROR", [q], 1)
        if op.name in replay.NOISE:
            continue
        if op.name in replay.MEASURE:
            for t in op.targets_copy():
                injected.append(op.name, [t], int(record in flipped_records))
                record += 1
        else:
            if any(record <= r < record + op.num_measurements for r in flipped_records):
                raise ValueError("Unsupported record flip on a multi-Pauli measurement")
            injected.append(op)
            record += op.num_measurements
    sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
    sim.do(injected)
    return (
        sim.get_detector_flips()[:, 0].nonzero()[0].tolist(),
        sim.get_observable_flips()[:, 0].nonzero()[0].tolist(),
    )


def certificate(path, output, verify):
    circuit = stim.Circuit.from_file(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    full = problem.load_dem_problem(str(path))
    if verify:
        record = json.loads(output.read_text())
        if record["sha256"] != digest:
            raise ValueError("Circuit hash differs")
    else:
        labels, logicals = propose_sectors(circuit)
        dem = circuit.detector_error_model(flatten_loops=True)
        indices = {}
        for i, op in enumerate(op for op in dem if op.type == "error"):
            indices[tuple(op.targets_copy())] = i
        witness = dem.shortest_graphlike_error(ignore_ungraphlike_errors=True)
        raw = [
            indices[tuple(op.targets_copy())] for op in witness if op.type == "error"
        ]
        reductions = [
            discover_bound(circuit, full, bit, len(raw), labels, logicals)
            for bit in range(full.nobs)
        ]
        record = {
            "sha256": digest,
            "circuit": path.name,
            "raw_dem_indices": raw,
            "reductions": reductions,
        }
    if len(record["reductions"]) != full.nobs:
        raise ValueError("Missing observable proof")
    lower = [
        lower_bound(full, i, reduction)
        for i, reduction in enumerate(record["reductions"])
    ]
    raw = record["raw_dem_indices"]
    if len(set(raw)) != len(raw) or any(
        i < 0 or i >= full.notes["raw_mechanisms"] for i in raw
    ):
        raise ValueError("Invalid witness indices")
    detectors, logicals, _ = replay.check_in_dem(circuit, raw)
    faults = replay.physical_faults(circuit, raw)
    # A postprocessed measurement record cannot capture a Pauli at an
    # observable boundary or a readout flip that controls a later gate.
    needs_frame_replay = any(
        (
            op.name == "OBSERVABLE_INCLUDE"
            and any(
                t.is_x_target or t.is_y_target or t.is_z_target
                for t in op.targets_copy()
            )
        )
        or (
            op.name in {"CX", "CY", "CZ"}
            and any(t.is_measurement_record_target for t in op.targets_copy())
        )
        for op in circuit.flattened()
    )
    fired, flipped = (replay_frames if needs_frame_replay else replay.replay)(
        circuit, faults
    )
    if (
        len(faults) != len(raw)
        or detectors
        or fired
        or not flipped
        or flipped != logicals
        or any(b is None for b in lower)
        or min(lower) != len(raw)
    ):
        raise ValueError(
            f"Bounds or physical replay failed: {lower}, {len(raw)}, {fired}, {flipped}"
        )
    results = {
        "lower_by_observable": lower,
        "exact_distance": len(raw),
        "replayed_observables": flipped,
    }
    if verify:
        if any(record[k] != v for k, v in results.items()):
            raise ValueError("Stored results differ from verification")
    else:
        record.update(results)
        output.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    return len(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(certificate(args.circuit, args.certificate, args.verify))
