"""Plot the retained Monte Carlo counts for the four planar rotation pipe choices."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from plot_tutorial_results import COLORS, MARKERS


ASSETS = Path(__file__).resolve().parents[1] / "docs/assets/experiments"


def main():
    with (ASSETS / "planar-rotation-rates.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    distances = sorted({int(row["distance"]) for row in rows})
    rates = sorted({float(row["p"]) for row in rows})
    faces = {"W": "West", "S": "South", "E": "East", "N": "North"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.98, top=.80, bottom=.13, wspace=.12, hspace=.28)
    fig.text(.105, .95, "Decoded error rate of a planar patch rotation", fontsize=22,
             fontweight="bold", color="#172e3c")
    fig.text(.105, .906, "Failure of either logical observable per rotation  ·  95% Wilson intervals",
             fontsize=12, color="#475e6c")
    for ax, inp, out in zip(axes.flat, ("W", "W", "S", "S"), ("N", "E", "N", "E")):
        for d, color, marker in zip(distances, COLORS, MARKERS):
            sample = sorted((r for r in rows if r["input_face"] == inp and
                             r["output_face"] == out and int(r["distance"]) == d),
                            key=lambda r: float(r["p"]))
            assert [float(r["p"]) for r in sample] == rates
            x = 100 * np.array(rates)
            y, low, high = (np.array([float(r[k]) for r in sample])
                            for k in ("any_rate", "any_low", "any_high"))
            positive = y > 0
            ax.plot(x, np.where(positive, y, np.nan), color=color, linewidth=1.5)
            ax.errorbar(x[positive], y[positive], yerr=(y[positive] - low[positive],
                        high[positive] - y[positive]), fmt=marker, color=color,
                        markersize=5, elinewidth=1, capsize=2)
            if not positive.all():
                ax.plot(x[~positive], high[~positive], "v", markerfacecolor="none",
                        color=color, markersize=7)
        ax.set_title(f"{faces[inp]} input → {faces[out].lower()} output", loc="left",
                     fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(100 * np.array(rates), [f"{100 * p:g}" for p in rates])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", color="#d7e0e5", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    legend = [Line2D([], [], color=color, marker=marker, label=f"d = {d}")
              for d, color, marker in zip(distances, COLORS, MARKERS)]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.096, .885),
               ncol=len(legend), frameon=False, columnspacing=2.2)
    fig.supxlabel("Physical error probability p (%)", y=.053, fontsize=12)
    fig.supylabel("Logical failure probability per rotation", x=.015, fontsize=12)
    if any(int(r["any_errors"]) == 0 for r in rows):
        fig.text(.105, .012, "Open downward triangles show the upper interval endpoint for zero observed failures.",
                 color="#475e6c", fontsize=10)
    fig.savefig(ASSETS / "planar-rotation-rates.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
