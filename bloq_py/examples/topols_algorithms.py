"""Run an exact algorithm through TopoLS, validated Bloq IR, and the noiseless VM.

Start with ising4, qpe4, or adder4. All candidates use independent
matrix oracles and ideal T-source proxies. Unsupported layouts fail explicitly.
"""

import argparse
import json
import math
from pathlib import Path
import time

from _algorithm_candidates import CANDIDATES, generate_candidate
from _interop_demo import run_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, default="ising4")
    parser.add_argument("--output-dir", type=Path, default=Path("topols-algorithm"))
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--engine", choices=("rust", "python"), default="rust")
    parser.add_argument("--layout-seed", type=int, default=0)
    parser.add_argument("--time-bound", type=float, default=0.1, help="search work budget per layer and seed")
    parser.add_argument("--iterations", type=int, default=100, help="maximum MCTS iterations per search")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--synthesis-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.shots < 1 or args.iterations < 1 or not math.isfinite(args.time_bound) or args.time_bound <= 0:
        parser.error("shots, iterations, and time-bound must be positive and finite")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    started = time.perf_counter()
    candidate = generate_candidate(args.candidate)
    candidate.summary["synthesis_seconds"] = time.perf_counter() - started
    result = run_demo(candidate.source_qasm, candidate.clifford_t_qasm, candidate.ideal_unitary,
                      args.output_dir, candidate.summary, compiler="topols", shots=args.shots,
                      synthesis_only=args.synthesis_only, compile_only=args.compile_only,
                      layout_options=dict(engine=args.engine, seed=args.layout_seed,
                                          time_bound=args.time_bound, iterations=args.iterations))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
