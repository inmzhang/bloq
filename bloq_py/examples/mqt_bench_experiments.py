"""Run pinned MQT Bench circuits through Topologiq, TopoLS, and SpiderLS into Bloq IR.

Every compiler receives the identical lowered Clifford+T QASM for a case.
Each run retains its artifacts and records which checks passed:

- lowering: dense-operator or statevector distance between the MQT source and
  the lowered circuit, within the stated synthesis bound;
- layout: sampled QuiZX branches of the imported block graph against the
  lowered QASM (requires the ``verify_interop_layout`` example binary);
- vm_d3: noiseless physical execution at d=3 with ideal T-source proxies,
  checking every joint output Pauli expectation (``execute`` tier only);
- ir_d25: validated d=25 compilation with full T cultivation and an exact
  serialization round trip.

Requires the interop dependency group, mqt.bench==2.3.0, and the pinned
upstream checkouts described in the interop tutorials. Build the QuiZX checker
with ``cargo build --release -p bloq_vm --example verify_interop_layout``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from _interop_demo import run_demo, save_json
from _mqt_bench import SUITE, UnsupportedSource, lower

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_OPTIONS = {"topologiq": {}, "topols": dict(engine="rust"), "spiderls": {}}
QUIZX_QUBITS = 10


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True).stdout.strip()


def _upstream(compiler):
    module = __import__(compiler if compiler != "spiderls" else "pyzx_routing")
    checkout = Path(module.__file__).resolve().parent
    while not (checkout / ".git").exists() and checkout != checkout.parent:
        checkout = checkout.parent
    patch = ROOT / f"bloq_py/examples/{compiler}.patch"
    applied = subprocess.run(["git", "-C", str(checkout), "apply", "--check", "-R", str(patch)], capture_output=True)
    return dict(revision=_git(checkout, "rev-parse", "HEAD"), patch_sha256=sha256(patch.read_bytes()).hexdigest(),
                patch_applied=applied.returncode == 0)


def worker(case_dir: Path, compiler: str, shots: int):
    """Route, compile, and check one case in a disposable process."""
    import bloq

    lowering = json.loads((case_dir / "lowering.json").read_text())
    expected = json.loads((case_dir / "expected_state.json").read_text()) if lowering["tier"] == "execute" else None
    run_dir = case_dir / compiler
    summary = dict(lowering, layout_compiler=compiler)
    state = None
    if expected is not None:
        import numpy as np
        state = (np.array(expected["real"]) + 1j * np.array(expected["imag"])).reshape(-1, 1)
    try:
        run_demo((case_dir / "source.qasm").read_text().rstrip("\n"),
                 (case_dir / "clifford_t.qasm").read_text().rstrip("\n"), state, run_dir, summary,
                 compiler=compiler, shots=shots, compile_only=state is None,
                 layout_options=LAYOUT_OPTIONS[compiler])
    except Exception:
        return  # run_demo retains the failed stage in result.json
    graph = bloq.BlockGraph.load(run_dir / "layout.blog")
    summary.update(stage="ir_d25", blog_round_trip=graph.to_text() == bloq.BlockGraph.from_text(graph.to_text()).to_text())
    save_json(run_dir / "result.json", summary)
    started = time.perf_counter()
    try:
        program = bloq.compile(graph, distance=25, validate=True)
        program.save(run_dir / "program_d25.bloq")
        stats = program.stats()
        summary["ir_d25"] = dict(
            status="compiled", seconds=time.perf_counter() - started, t_preparation="cultivation",
            physical_qubits=program.qubit_count, measurements=program.measurement_count,
            nodes=stats.node_count, validated=True,
            round_trip=bloq.Bloq.load(run_dir / "program_d25.bloq").to_text() == program.to_text())
    except Exception as error:
        summary["ir_d25"] = dict(status="failed", error=str(error), error_type=type(error).__name__)
    summary.pop("stage", None)
    save_json(run_dir / "result.json", summary)


def _render(run_dir: Path, title: str):
    sys.path.insert(0, str(ROOT / "tools"))
    import bloq
    from render_block_thumbnails import thumbnail

    graph = bloq.BlockGraph.load(run_dir / "layout.blog")
    if graph.flatten().block_count <= 4000:
        thumbnail(graph, run_dir / "layout.svg", title, opaque=True)


def _quizx(run_dir: Path, checker: Path, timeout: float):
    command = [str(checker), str(run_dir), "32", "7"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(status="inconclusive", reason=f"timed out after {timeout:.0f} s")
    if done.returncode:
        return dict(status="failed", error=(done.stderr or done.stdout).strip()[-2000:])
    report = json.loads(done.stdout)
    return dict(report, status="sampled" if report["verified"] == report["samples"] else "partial")


def classify(result):
    """Summarize which claims a run supports, strongest first."""
    if result.get("status") in (None, "failed", "running", "synthesized", "timeout"):
        return "failed"
    layout = result.get("quizx", {}).get("status")
    vm = result.get("status") == "executed"
    if layout == "failed":
        return "failed"
    if layout == "sampled" and vm:
        return "verified (sampled branches and VM)"
    if layout == "sampled":
        return "verified (sampled branches)"
    if vm:
        return "verified (VM only)"
    if result.get("ir_d25", {}).get("status") == "compiled":
        return "compilation only"
    return "inconclusive"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("target/mqt-bench"))
    parser.add_argument("--cases", nargs="*", help="case slugs such as qft_3; default all")
    parser.add_argument("--compilers", nargs="*", choices=tuple(LAYOUT_OPTIONS))
    parser.add_argument("--epsilon", type=float, default=1e-2, help="total synthesis error per circuit")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per compiler run")
    parser.add_argument("--quizx", type=Path, default=ROOT / "target/release/examples/verify_interop_layout")
    parser.add_argument("--worker", nargs=2, metavar=("CASE_DIR", "COMPILER"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(Path(args.worker[0]), args.worker[1], args.shots)
    suite = [b for b in SUITE if not args.cases or b.slug in args.cases]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; each run retains its own artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=platform.platform(), python=platform.python_version(),
        cpus=os.cpu_count(), bloq_revision=_git(ROOT, "rev-parse", "HEAD"),
        bloq_dirty=bool(_git(ROOT, "status", "--porcelain", "--untracked-files=no")),
        packages={p: version(p) for p in ("bloq-py", "mqt.bench", "qiskit", "pyzx", "pygridsynth", "numpy")},
        upstream={c: _upstream(c) for c in LAYOUT_OPTIONS}, epsilon=args.epsilon, shots=args.shots,
        timeout_seconds=args.timeout, vm_distance=3, ir_distance=25, layout_options=LAYOUT_OPTIONS,
    )
    save_json(args.output_dir / "manifest.json", manifest)
    jobs, rows = [], []
    for benchmark in suite:
        case_dir = args.output_dir / "cases" / benchmark.slug
        case_dir.mkdir(parents=True)
        try:
            lowered = lower(benchmark, epsilon=args.epsilon)
        except UnsupportedSource as error:
            rows.append(dict(case=benchmark.slug, benchmark=benchmark.name, tier=benchmark.tier,
                             note=benchmark.note, status="rejected", error_code=error.code, error=str(error)))
            save_json(case_dir / "lowering.json", rows[-1])
            continue
        (case_dir / "source.qasm").write_text(lowered.source_qasm, encoding="utf-8")
        (case_dir / "clifford_t.qasm").write_text(lowered.clifford_t_qasm + "\n", encoding="utf-8")
        save_json(case_dir / "lowering.json", lowered.summary)
        if benchmark.tier == "execute":
            save_json(case_dir / "expected_state.json", dict(real=lowered.expected_state.real.tolist(),
                                                              imag=lowered.expected_state.imag.tolist()))
        jobs += [(benchmark, case_dir, c) for c in benchmark.compilers
                 if not args.compilers or c in args.compilers]

    def run(job):
        benchmark, case_dir, compiler = job
        run_dir = case_dir / compiler
        started = time.perf_counter()
        print(f"[start] {benchmark.slug} {compiler}", flush=True)
        try:
            subprocess.run([sys.executable, __file__, "--worker", str(case_dir), compiler, "--shots", str(args.shots)],
                           timeout=args.timeout, capture_output=True, check=False)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if timed_out:
            result.update(status="timeout", failed_stage=result.get("stage"), timeout_seconds=args.timeout)
        result["wall_seconds"] = time.perf_counter() - started
        if (run_dir / "layout.blog").exists():
            _render(run_dir, f"{benchmark.slug} via {compiler}")
            if result.get("logical_qubits", 99) <= QUIZX_QUBITS and args.quizx.exists():
                result["quizx"] = _quizx(run_dir, args.quizx, min(args.timeout, 1800))
        result["verification"] = classify(result)
        if run_dir.exists():
            save_json(result_path, result)
        print(f"[done] {benchmark.slug} {compiler}: {result['verification']} ({result.get('status')})", flush=True)
        return result

    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(run, jobs):
            rows.append(result)
    save_json(args.output_dir / "summary.json", dict(manifest=manifest, runs=rows))


if __name__ == "__main__":
    main()
