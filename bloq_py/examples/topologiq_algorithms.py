"""Compare small algorithm candidates through Topologiq, Bloq IR, and the VM.

Each candidate has an independent complete-matrix oracle and prepares its input
from zero. Run in the optional, pinned interop environment described in the
Topologiq tutorial. Each subprocess retains artifacts, errors, and timings.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from _algorithm_candidates import CANDIDATES, generate_candidate


def _worker(name, args):
    from _interop_demo import run_demo

    started = time.perf_counter()
    candidate = generate_candidate(name)
    candidate.summary["synthesis_seconds"] = time.perf_counter() - started
    return run_demo(candidate.source_qasm, candidate.clifford_t_qasm, candidate.ideal_unitary,
                    args.output_dir, candidate.summary, shots=args.shots,
                    synthesis_only=args.synthesis_only, compile_only=args.compile_only,
                    layout_options=dict(seed=args.layout_seed, gravity=args.gravity,
                                        z_stretch=args.z_stretch))


def _comparison_markdown(comparison):
    lines = ["# Topologiq algorithm candidates", "",
             "Noiseless VM execution with ideal T sources and ideal boundaries; requested distance three.",
             "Every accepted demo shot checks all joint Pauli expectations against an independent state.", "",
             "| Candidate | Qubits | T/T† | Blocks | VM seconds | Peak rank | Result |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for result in comparison["candidates"]:
        status = result["status"]
        if status in ("failed", "timeout"):
            status += f" ({result['failed_stage']})"
        values = [result.get("algorithm", result["candidate"]), result.get("qubits", "—"),
                  result.get("t_count", "—"), result.get("layout_blocks", "—"),
                  f"{result['execution_seconds']:.3f}" if "execution_seconds" in result else "—",
                  result.get("peak_rank", "—"), status]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    lines += ["", "## Diagnostics and artifacts", ""]
    for result in comparison["candidates"]:
        name = result["candidate"]
        detail = result.get("error", result.get("vm_verification", "source/compilation artifacts retained"))
        lines.append(f"- [{name}]({name}/result.json): {detail}. [Log]({result['log']})")
    lines += ["", "VM times cover the requested shots on this host; they are not hardware timing estimates.",
              "The routing profile permits Hadamard pipes only on the temporal axis.", ""]
    return "\n".join(lines)


def main():
    from _interop_demo import save_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--output-dir", type=Path, default=Path("topologiq-algorithms"))
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--layout-seed", type=int, default=1330)
    parser.add_argument("--gravity", type=int, default=7)
    parser.add_argument("--z-stretch", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180, help="seconds allowed per candidate")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--synthesis-only", action="store_true")
    modes.add_argument("--compile-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shots < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("shots and timeout must be positive and finite")
    if args.gravity < 0 or args.z_stretch < 0:
        parser.error("gravity and z-stretch must be nonnegative")
    if len(set(args.candidates)) != len(args.candidates):
        parser.error("candidates must be unique")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    if args.worker:
        if len(args.candidates) != 1:
            parser.error("worker requires exactly one candidate")
        try:
            print(json.dumps(_worker(args.candidates[0], args), indent=2))
        except Exception as error:
            result_file = args.output_dir / "result.json"
            if not result_file.exists():
                args.output_dir.mkdir(parents=True, exist_ok=True)
                save_json(result_file, dict(candidate=args.candidates[0], status="failed",
                    failed_stage="synthesis", error_type=type(error).__name__, error=str(error)))
            raise
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = dict(shots_requested=args.shots, timeout_seconds=args.timeout, candidates=[])
    for name in args.candidates:
        output = args.output_dir / name
        output.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--candidates", name,
                   "--shots", str(args.shots), "--layout-seed", str(args.layout_seed),
                   "--output-dir", str(output.resolve()), "--gravity", str(args.gravity),
                   "--z-stretch", str(args.z_stretch)]
        if args.synthesis_only:
            command.append("--synthesis-only")
        if args.compile_only:
            command.append("--compile-only")
        print(f"Trying {name}: {CANDIDATES[name]}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        with (args.output_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                status = "failed" if completed.returncode else None
            except subprocess.TimeoutExpired:
                status = "timeout"
        result_file = output / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else dict(candidate=name)
        if status:
            result.update(status=status, failed_stage=result.get("failed_stage", result.get("stage", "synthesis")))
            save_json(result_file, result)
        result["total_seconds"] = time.perf_counter() - started
        result["log"] = f"{name}.log"
        comparison["candidates"].append(result)
        save_json(args.output_dir / "comparison.json", comparison)
        (args.output_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    if any(c["status"] in ("failed", "timeout") for c in comparison["candidates"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
