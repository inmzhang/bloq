"""Recompile the simulated port circuits and verify their exact-distance certificates."""

import argparse
import json
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import bloq
from certify import certificate
from simulate import DISTANCES, LAYOUTS, RESULTS, SOURCES


def check(task):
    layout, distance, record = task
    source = SOURCES / f"spatial-hadamard-{layout}.blog"
    graph = bloq.BlockGraph.from_text(source.read_text())
    circuit = bloq.compile_to_stim(graph, distance=distance, noise=0.001)
    circuit.detector_error_model(
        decompose_errors=True, block_decomposition_from_introducing_remnant_edges=True
    )
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="discover and verify new certificates")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--archive", type=Path, default=RESULTS / "spatial-hadamard-certificates.zip")
    args = parser.parse_args()
    inputs = list(product(LAYOUTS, DISTANCES))
    if args.generate:
        tasks = [(layout, d, None) for layout, d in inputs]
    else:
        with zipfile.ZipFile(args.archive) as archive:
            tasks = [(layout, d, archive.read(f"{layout}-d{d}.json")) for layout, d in inputs]
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
