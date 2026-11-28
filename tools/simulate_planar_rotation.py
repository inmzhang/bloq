"""Measure decoded logical failure rates of the planar rotate-Z block.

Run with an environment containing the worktree's bloq bindings, Stim 1.16.0,
PyMatching 2.4.0 and NumPy 2.5.4. The sampler uses the emitted circuit with
Bloq's aligned uniform circuit noise and ideal spatial Port substitutions.
Standard minimum-weight perfect matching decodes the detector syndromes.
For identical shot streams use the same Stim version, SIMD width, and batch size.

`x_errors` and `z_errors` count wrong predictions of the logical X and Z
observable bits. A logical Pauli Z flips the X observable. `both_errors` is
the intersection and `any_errors` is the union. Each rate is per block use
with a two-sided 95% Wilson binomial interval. Every sampled shot is counted.
"""

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist
from tempfile import NamedTemporaryFile

import bloq
import numpy as np
import pymatching


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/experiments/planar-rotation-rates.csv"
INPUT_PORTS = {"W": [-1, 0, 0], "S": [0, -1, 0]}
OUTPUT_PORTS = {"N": [1, 2, 0], "E": [2, 1, 0]}
DISTANCES = (3, 5, 7, 9)
# Fixed after an independent pilot. Rare failures at low p need more shots.
SHOTS = {
    0.0015: 2_097_152,
    0.0025: 262_144,
    0.0035: 131_072,
    0.005: 65_536,
    0.007: 32_768,
    0.008: 32_768,
    0.01: 32_768,
}
BATCH = 2048
FIRST_SEED = 28_110_000
Z95 = NormalDist().inv_cdf(0.975)


def wilson(errors, shots):
    rate = errors / shots
    denominator = 1 + Z95**2 / shots
    center = (rate + Z95**2 / (2 * shots)) / denominator
    radius = (
        Z95
        * math.sqrt(rate * (1 - rate) / shots + Z95**2 / (4 * shots**2))
        / denominator
    )
    return rate, max(0.0, center - radius), min(1.0, center + radius)


def simulate(case):
    seed, input_face, output_face, distance, p = case
    graph = bloq.BlockGraph.from_text(
        f"BLOG 1.0\n0: Port {INPUT_PORTS[input_face]} role=input\n"
        "1: rotate Z [0,0,0] -> [1,1,0]\n"
        f"2: Port {OUTPUT_PORTS[output_face]} role=output\n"
        "0 -> [0,0,0]\n[1,1,0] -> 2\n"
    )
    program = bloq.compile(graph, distance=distance, validate=True)
    clean = bloq.emit_stim(program, align_moments=True)
    clean_dets, clean_obs = clean.compile_detector_sampler(seed=0).sample(
        32, separate_observables=True
    )
    assert not clean_dets.any() and not clean_obs.any()
    # Identify the logical bits from the physical Pauli terms at both Ports.
    labels = []
    for op in clean.flattened():
        if op.name == "OBSERVABLE_INCLUDE":
            targets = op.targets_copy()
            if targets and all(t.is_x_target for t in targets):
                labels.append((int(op.gate_args_copy()[0]), "X"))
            if targets and all(t.is_z_target for t in targets):
                labels.append((int(op.gate_args_copy()[0]), "Z"))
    assert sorted(labels) == [(0, "X"), (0, "X"), (1, "Z"), (1, "Z")]

    circuit = bloq.emit_stim(program, noise=p, align_moments=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    decoder = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True), enable_correlations=False
    )
    assert decoder.num_fault_ids == circuit.num_observables == 2
    shots = SHOTS[p]
    counts = np.zeros(4, dtype=np.int64)
    for start in range(0, shots, BATCH):
        detectors, actual = sampler.sample(
            min(BATCH, shots - start), separate_observables=True, bit_packed=True
        )
        predicted = decoder.decode_batch(
            detectors,
            bit_packed_shots=True,
            bit_packed_predictions=True,
            enable_correlations=False,
        )
        # Two logicals fit into one byte. Bin 3 flips both.
        counts += np.bincount((predicted ^ actual)[:, 0], minlength=4)
    correct, x_only, z_only, both = map(int, counts)
    assert correct + x_only + z_only + both == shots
    row = dict(
        input_face=input_face,
        output_face=output_face,
        distance=distance,
        p=p,
        shots=shots,
        seed=seed,
        x_errors=x_only + both,
        z_errors=z_only + both,
        both_errors=both,
        any_errors=shots - correct,
    )
    for label in ("x", "z", "any", "both"):
        rate, low, high = wilson(row[f"{label}_errors"], shots)
        row.update({f"{label}_rate": rate, f"{label}_low": low, f"{label}_high": high})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cases = [
        (FIRST_SEED + i, *case)
        for i, case in enumerate(product(INPUT_PORTS, OUTPUT_PORTS, DISTANCES, SHOTS))
    ]
    # Start the largest jobs early. Case seeds and CSV ordering are independent of the pool.
    cases.sort(key=lambda case: SHOTS[case[4]] * case[3] ** 3, reverse=True)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn")
    ) as pool:
        rows = sorted(pool.map(simulate, cases), key=lambda row: row["seed"])
    with NamedTemporaryFile(
        "w", newline="", dir=args.output.parent, delete=False
    ) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    Path(tmp.name).replace(args.output)
    print(
        f"Wrote {len(rows)} cases ({sum(row['shots'] for row in rows):,} shots) to {args.output}"
    )


if __name__ == "__main__":
    main()
