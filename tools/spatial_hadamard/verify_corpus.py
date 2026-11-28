"""Recompile the simulated port circuits and verify their exact-distance certificates."""

import argparse
import csv
import hashlib
import json
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import bloq
import stim
from certify import certificate, masks
from simulate import DISTANCES, LAYOUTS, RESULTS, SHOTS, SOURCES


def check_rounds(circuit, distance):
    coords = circuit.get_final_qubit_coordinates()
    used, resets, gates, readouts = set(), set(), set(), set()
    tick = 0
    for op in circuit.flattened():
        if op.name == "TICK":
            used.clear()
            tick += 1
            continue
        targets = op.targets_copy()
        if op.name in {"CX", "CZ"}:
            qubits = []
            for a, b in zip(targets[::2], targets[1::2], strict=True):
                if a.is_measurement_record_target or b.is_measurement_record_target:
                    continue
                qubits.extend((a.value, b.value))
                delta = [
                    abs(x - y)
                    for x, y in zip(
                        coords[a.value][:2], coords[b.value][:2], strict=True
                    )
                ]
                if delta != [1, 1]:
                    raise ValueError("Nonlocal physical gate")
            if qubits:
                gates.add(tick)
        elif op.name in {"R", "RX", "RY", "M", "MX", "MY"}:
            (resets if op.name.startswith("R") else readouts).add(tick)
            qubits = [t.value for t in targets]
        else:
            continue
        for q in qubits:
            if q in used:
                raise ValueError(f"Qubit {q} used twice at physical tick {tick}")
            used.add(q)
    if (
        len(resets) != distance
        or readouts != {t + 6 for t in resets}
        or gates != {t + k for t in resets for k in range(1, 6)}
    ):
        raise ValueError(
            "Expected preparation, five interaction moments and readout in every round"
        )


def check_matchability(circuit, label):
    raw = circuit.detector_error_model(flatten_loops=True)
    if any(len(masks(op)[0]) > 4 for op in raw if op.type == "error"):
        raise ValueError(f"Physical fault flips more than four detectors in {label}")
    circuit.detector_error_model(
        decompose_errors=True, block_decomposition_from_introducing_remnant_edges=True
    )


def check(task):
    layout, distance, record, sweep = task
    source = SOURCES / f"spatial-hadamard-{layout}.blog"
    graph = bloq.BlockGraph.from_text(source.read_text())
    circuit = bloq.compile_to_stim(graph, distance=distance, noise=0.001)
    check_matchability(circuit, f"{layout}, d={distance}")
    check_rounds(circuit, distance)
    for p, expected in sweep:
        candidate = bloq.compile_to_stim(graph, distance=distance, noise=p)
        if hashlib.sha256(str(candidate).encode()).hexdigest() != expected:
            raise ValueError(f"Different sampled circuit for {layout}, d={distance}, p={p}")
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / f"{layout}-d{distance}.stim"
        path.write_text(str(circuit))
        proof = path.with_suffix(".json")
        if record is None:
            certificate(path, proof, verify=False)
        else:
            proof.write_bytes(record)
        actual = certificate(path, proof, verify=True)
        if actual != distance:
            raise ValueError(f"{path.name} has circuit distance {actual}, expected {distance}")
        return proof.name, proof.read_bytes()


def check_junction(task):
    directory, layout, distance, digest = task
    circuit = directory / layout / f"d{distance}.stim"
    if hashlib.sha256(circuit.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Different Stim circuit for {layout}, d={distance}")
    check_matchability(stim.Circuit.from_file(circuit), layout)
    return certificate(circuit, circuit.with_suffix(".cert.json"), verify=True) == distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="discover and verify new certificates")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--archive", type=Path, default=RESULTS / "spatial-hadamard-certificates.zip")
    parser.add_argument("--sweep", type=Path, help="check the sampled circuit hashes in this CSV")
    parser.add_argument("--junctions", type=Path, help="verify the extracted signed-wall and junction corpus")
    args = parser.parse_args()
    if args.junctions:
        if args.generate or args.sweep:
            parser.error("--generate and --sweep apply to the port-to-port noise sweep")
        manifest = json.loads((args.junctions / "manifest.json").read_text())
        layouts = {layout for layout, _, _, _ in manifest}
        if len(layouts) != 117 or sorted((layout, d) for layout, d, _, _ in manifest) != sorted(
            product(layouts, [3, 5, 7, 9])
        ):
            raise ValueError("Incomplete junction corpus")
        with ProcessPoolExecutor(args.jobs) as pool:
            if not all(pool.map(check_junction, [(args.junctions, layout, d, sha)
                                                 for layout, d, _, sha in manifest])):
                raise ValueError("Junction circuit distance differs from its requested size")
        print(f"Verified all {len(manifest)} signed-wall and junction circuits")
        return
    inputs = list(product(LAYOUTS, DISTANCES))
    if args.generate:
        tasks = [(layout, d, None) for layout, d in inputs]
    else:
        with zipfile.ZipFile(args.archive) as archive:
            tasks = [(layout, d, archive.read(f"{layout}-d{d}.json")) for layout, d in inputs]
    sweeps = {key: [] for key in inputs}
    if args.sweep:
        with args.sweep.open() as source:
            for row in csv.DictReader(source):
                if row["decoder"] == "correlated":
                    sweeps[row["layout"], int(row["distance"])].append((float(row["p"]), row["circuit_sha256"]))
        if any(sorted(p for p, _ in rows) != sorted(SHOTS) for rows in sweeps.values()):
            raise ValueError("Incomplete Monte Carlo sweep")
    tasks = [(layout, d, record, sweeps[layout, d]) for layout, d, record in tasks]
    with ProcessPoolExecutor(args.jobs) as pool:
        results = list(pool.map(check, tasks))
    if args.generate:
        with zipfile.ZipFile(args.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, record in results:
                archive.writestr(name, record)
    for name, record in results:
        print(name, json.loads(record)["exact_distance"])


if __name__ == "__main__":
    main()
