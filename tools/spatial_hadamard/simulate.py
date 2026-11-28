"""Sample the four port-to-port spatial Hadamard constructions and plot logical failure."""

import argparse
import csv
import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "docs/_static/constructions"
RESULTS = ROOT / "docs/assets/experiments"
LAYOUTS = {
    "x-zxz": "Single X wall",
    "y-zxz": "Single Y wall",
    "elbow-zzx": "X/Y elbow · ZZX hub",
    "elbow-xxz": "X/Y elbow · XXZ hub",
}
DISTANCES = (3, 5, 7, 9, 11, 13)
# A fixed budget per point gives rare failures more shots at low noise.
SHOTS = {
    0.0005: 262144, 0.001: 131072, 0.002: 65536,
    0.003: 32768, 0.004: 32768, 0.005: 16384,
    0.006: 8192, 0.008: 8192, 0.01: 8192,
}
DECODERS = ("correlated", "ordinary")


def interval(errors, shots):
    """Pointwise two-sided 95% Wilson score limits, including zero failures."""
    z = NormalDist().inv_cdf(0.975)
    p = errors / shots
    center = (errors + z * z / 2) / (shots + z * z)
    radius = z * math.sqrt(shots * p * (1 - p) + z * z / 4) / (shots + z * z)
    return max(0, center - radius), min(1, center + radius)


def sample(task):
    import bloq
    import numpy as np
    import pymatching

    layout, d, p = task
    source = SOURCES / f"spatial-hadamard-{layout}.blog"
    graph = bloq.BlockGraph.from_text(source.read_text())
    circuit = bloq.compile_to_stim(graph, distance=d, noise=p)
    if circuit.num_observables != 2:
        raise ValueError(f"Expected two port stabilizers in {layout} at d={d}")
    digest = hashlib.sha256(str(circuit).encode()).hexdigest()
    seed = int.from_bytes(hashlib.sha256(f"wall-v1/{layout}/{d}/{p}".encode()).digest()[:8], "little")
    dem = circuit.detector_error_model(decompose_errors=True,
                                       block_decomposition_from_introducing_remnant_edges=True)
    matchers = {name: pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=name == "correlated") for name in DECODERS}
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = SHOTS[p]
    if d >= 9 and p >= 0.005:
        shots //= 4
    hist = {name: np.zeros(4, dtype=np.int64) for name in DECODERS}
    for start in range(0, shots, 1024):
        det, obs = sampler.sample(shots=min(1024, shots - start), separate_observables=True,
                                  bit_packed=True)
        for name, matcher in matchers.items():
            predicted = matcher.decode_batch(det, bit_packed_shots=True, bit_packed_predictions=True,
                                             enable_correlations=name == "correlated")
            hist[name] += np.bincount((obs ^ predicted)[:, 0], minlength=4)
    rows = []
    for name, (ok, only0, only1, both) in hist.items():
        errors = int(only0 + only1 + both)
        assert int(ok) + errors == shots
        low, high = interval(errors, shots)
        rows.append(dict(layout=layout, distance=d, p=p, decoder=name, shots=shots, errors=errors,
                         only_L0=int(only0), only_L1=int(only1), both=int(both), rate=errors/shots,
                         low95=low, high95=high, seed=seed, circuit_sha256=digest))
    return rows


def plot(rows, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True, layout="constrained")
    colors = ["#0b7285", "#cb4b16", "#7042a0", "#466920", "#2352a4", "#90314e"]
    for axis, (layout, title) in zip(axes.flat, LAYOUTS.items(), strict=True):
        for d, color in zip(DISTANCES, colors, strict=True):
            series = sorted((r for r in rows if r["layout"] == layout and int(r["distance"]) == d
                             and r["decoder"] == "correlated"), key=lambda r: float(r["p"]))
            positive = [r for r in series if int(r["errors"])]
            x = [float(r["p"]) for r in positive]
            y = [float(r["rate"]) for r in positive]
            axis.errorbar(x, y, yerr=[[v-float(r["low95"]) for v,r in zip(y,positive)],
                                     [float(r["high95"])-v for v,r in zip(y,positive)]],
                          color=color, marker="o", ms=3, lw=1.2, capsize=2)
            zero = [r for r in series if not int(r["errors"])]
            axis.scatter([float(r["p"]) for r in zero], [float(r["high95"]) for r in zero],
                         marker="v", facecolor="none", edgecolor=color, s=30)
        axis.set(title=title, xscale="log", yscale="log")
        axis.set_xticks([.0005,.001,.002,.004,.01], ["0.05%", "0.1%", "0.2%", "0.4%", "1%"])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:g}%"))
        axis.grid(alpha=.2, which="major")
    fig.supylabel("Logical failure per complete circuit")
    fig.supxlabel("Physical error probability")
    handles = [Line2D([], [], color=c, marker="o", ms=4, label=f"d = {d}")
               for c, d in zip(colors, DISTANCES, strict=True)]
    handles.append(Line2D([], [], color="#555", marker="v", markerfacecolor="none", ls="",
                          label="95% upper limit (zero failures)"))
    fig.legend(handles=handles, loc="outside upper center", ncols=4, frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output / "spatial-hadamard-rates.csv"
    if args.plot_only:
        with table.open() as f:
            rows = list(csv.DictReader(f))
    else:
        tasks = list(product(LAYOUTS, DISTANCES, SHOTS))
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = [r for pair in pool.map(sample, tasks) for r in pair]
        with table.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    plot(rows, args.output / "spatial-hadamard-rates.png")


if __name__ == "__main__":
    main()
