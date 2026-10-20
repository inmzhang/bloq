"""Run pinned MQT Bench circuits through upstream layouts into validated d=25 IR.

Lowering, layout/IR compilation, sampled QuiZX checks, and optional d=3 VM
execution run in separate bounded processes. A later timeout or execution
failure never erases a retained, validated compilation. VM circuits use ideal
T-source proxies. Reports and screenshots are generated separately.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import resource
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_OPTIONS = {"topologiq": {"seed": 1330}, "topols": {"engine": "rust", "seed": 1330}, "spiderls": {}}
QUIZX_QUBITS = 10


def save_json(path, value):
    # Keep the last complete checkpoint if a worker is killed while writing.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def _provenance(output, compilers):
    diff = _git(ROOT, "diff", "--binary", "HEAD")
    (output / "bloq.diff").write_bytes(diff)
    source_paths = _git(ROOT, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    hashes = {name: _digest(ROOT / name) for name in sorted(set(source_paths))
              if name and (ROOT / name).is_file() and Path(name).suffix in {".rs", ".py", ".toml", ".lock", ".patch"}}
    save_json(output / "source-files.json", hashes)
    upstream = {}
    for compiler in compilers:
        module = import_module(compiler if compiler != "spiderls" else "pyzx_routing")
        checkout = Path(module.__file__).resolve().parent
        checkout = Path(_git(checkout, "rev-parse", "--show-toplevel").decode().strip())
        patch = ROOT / f"bloq_py/examples/{compiler}.patch"
        applied = subprocess.run(["git", "-C", str(checkout), "apply", "--check", "-R", str(patch)],
                                 capture_output=True).returncode == 0
        upstream_diff = output / f"{compiler}.diff"
        upstream_diff.write_bytes(_git(checkout, "diff", "--binary", "HEAD"))
        upstream[compiler] = dict(revision=_git(checkout, "rev-parse", "HEAD").decode().strip(),
                                  patch_sha256=_digest(patch), patch_applied=applied,
                                  diff_file=upstream_diff.name, diff_sha256=_digest(upstream_diff))
    packages = {}
    for package in ("bloq-py", "mqt.bench", "qiskit", "pyzx", "pygridsynth", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return dict(bloq_revision=_git(ROOT, "rev-parse", "HEAD").decode().strip(),
                bloq_dirty=bool(_git(ROOT, "status", "--porcelain")),
                bloq_diff_file="bloq.diff", bloq_diff_sha256=_digest(output / "bloq.diff"),
                source_files_file="source-files.json", source_files_sha256=_digest(output / "source-files.json"),
                packages=packages, upstream=upstream)


def _bounded(command, directory, label, timeout, memory_gib=8):
    """Limit the child before exec, and terminate its entire group on timeout."""
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    stdout, stderr = logs / f"{label}.stdout.log", logs / f"{label}.stderr.log"
    started = time.perf_counter()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--limit-exec", str(int(memory_gib * 1024**3)), *command],
            stdout=out, stderr=err, start_new_session=True, env=env)
        try:
            process.wait(timeout=timeout)
            result = dict(status="completed" if process.returncode == 0 else "failed", returncode=process.returncode)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            result = dict(status="inconclusive", reason=f"timed out after {timeout:g} s", returncode=process.returncode)
    diagnostic = stderr.read_text(errors="replace")
    if process.returncode < 0 or any(text in diagnostic.lower() for text in
                                     ("memoryerror", "memory allocation", "cannot allocate memory", "out of memory")):
        result.update(status="inconclusive", reason=result.get("reason", "process crashed or exceeded memory bound"))
    return dict(result, seconds=time.perf_counter() - started,
                stdout=str(stdout.relative_to(directory)), stderr=str(stderr.relative_to(directory)))


def _worker_command(stage, case_dir, *, compiler=None, epsilon=1e-2, shots=3):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage,
               "--case-dir", str(case_dir), "--epsilon", str(epsilon), "--shots", str(shots)]
    return command + (["--compiler", compiler] if compiler else [])


def _failure(error):
    limited = (isinstance(error, MemoryError) or getattr(error, "code", None) == "resource_placement_limit"
               or re.search(
        r"stabilizer rank \d+ exceeds cap \d+|runtime step cap \d+ exceeded|"
        r"(?:Boolean work steps|Boolean nodes|row witness operations) requires \d+, limit is \d+|"
        r"physical readout binding expansion exceeds \d+ work units|"
        r"Routing exceeded max_physical_layers", str(error)))
    result = dict(status="inconclusive" if limited else "failed",
                  error=str(error), error_type=type(error).__name__)
    if limited:
        result["reason"] = "resource limit exhausted"
    if hasattr(error, "code"):
        result.update(error_code=error.code, error_nodes=getattr(error, "nodes", ()))
    return result


def _lower_worker(case_dir, epsilon):
    from _mqt_bench import SUITE, UnsupportedSource, lower

    benchmark = next(case for case in SUITE if case.slug == case_dir.name)
    result = dict(case=benchmark.slug, benchmark=benchmark.name, circuit_size=benchmark.size,
                  tier=benchmark.tier, note=benchmark.note, status="running", stage="lowering")
    save_json(case_dir / "lowering.json", result)
    try:
        lowered = lower(benchmark, epsilon=epsilon)
        (case_dir / "source.qasm").write_text(lowered.source_qasm, encoding="utf-8")
        (case_dir / "clifford_t.qasm").write_text(lowered.clifford_t_qasm + "\n", encoding="utf-8")
        for name, state in (("expected_state", lowered.expected_state), ("source_state", lowered.source_state)):
            if state is not None:
                save_json(case_dir / f"{name}.json", dict(real=state.real.tolist(), imag=state.imag.tolist()))
        result.update(lowered.summary, status="lowered")
        if lowered.benchmark_qasm is not None:
            path = case_dir / "benchmark.qasm"
            path.write_text(lowered.benchmark_qasm, encoding="utf-8")
            result["benchmark_sha256"] = _digest(path)
    except Exception as error:
        if isinstance(error, UnsupportedSource):
            result.update(error.summary or {}, **_failure(error))
            result["status"] = "rejected"
            if error.source_qasm is not None:
                (case_dir / "source.qasm").write_text(error.source_qasm, encoding="utf-8")
        else:
            result.update(_failure(error))
        traceback.print_exc()
    result.pop("stage", None)
    save_json(case_dir / "lowering.json", result)


def _output_indices(program, ports):
    outputs = program.logical_outputs()
    indices = {tuple(output.port): index for index, output in enumerate(outputs)}
    if len(outputs) != len(ports) or len(indices) != len(outputs) or set(indices) != {tuple(port) for port in ports}:
        raise RuntimeError("IR output interface differs from the source circuit")
    return [indices[tuple(port)] for port in ports]


def _compile_worker(case_dir, compiler):
    import bloq

    run_dir = case_dir / compiler
    result = _read(run_dir / "result.json")
    stage = "layout"
    started = time.perf_counter()
    try:
        adapter = import_module(f"bloq.interop.{compiler}")
        imported = adapter.from_qasm((case_dir / "clifford_t.qasm").read_text(), **LAYOUT_OPTIONS[compiler])
        graph = imported.graph
        graph.save(run_dir / "layout.blog")
        saved_graph = bloq.BlockGraph.load(run_dir / "layout.blog")
        if saved_graph.to_text() != graph.to_text():
            raise RuntimeError("BLOG round trip changed the imported layout")
        graph = saved_graph
        graph.export_html_viewer(run_dir / "layout.html")
        if compiler == "topols":
            (run_dir / "normalized.qasm").write_text(imported.normalized_qasm, encoding="utf-8")
        flat = graph.flatten()
        positions = flat.positions()
        result.update(input_ports=imported.input_ports, output_ports=imported.output_ports,
                      layout_options=imported.layout_options, audited_revision=adapter.AUDITED_REVISION,
                      upstream_revision=imported.upstream_revision, blocks=flat.block_count,
                      pipes=flat.pipe_count, routed_t=flat.t_count, layout_blocks=flat.block_count,
                      extents=[max(p[i] for p in positions) - min(p[i] for p in positions) + 1 for i in range(3)],
                      hadamard_pipes_by_axis={axis: sum(pipe.hadamard and pipe.src[i] != pipe.dst[i]
                                                       for pipe in flat.pipes()) for i, axis in enumerate("XYZ")},
                      layout_seconds=time.perf_counter() - started,
                      layout=dict(status="passed", round_trip=True, blog_sha256=_digest(run_dir / "layout.blog")))
        stage = "ir_d25"
        result["stage"] = stage
        save_json(run_dir / "result.json", result)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            program = bloq.compile(graph, distance=25, prepare_t_with_mpps=True, validate=True)
        path = run_dir / "program_d25.bloq"
        program.save(path)
        if bloq.Bloq.load(path).to_binary() != program.to_binary():
            raise RuntimeError("binary round trip changed the validated IR")
        _output_indices(program, imported.output_ports)
        result.update(status="compiled", ir_d25=dict(
            status="compiled", distance=25, t_preparation="ideal_t_proxy", validated=True, round_trip=True,
            seconds=time.perf_counter() - started, physical_qubits=program.qubit_count,
            measurements=program.measurement_count, nodes=program.stats().node_count, sha256=_digest(path),
            warnings=[str(warning.message) for warning in caught]))
    except Exception as error:
        result.update(status="failed", failed_stage=stage)
        result[stage] = _failure(error)
        traceback.print_exc()
    result.pop("stage", None)
    save_json(run_dir / "result.json", result)


def _signature(state, indices):
    import numpy as np
    from qiskit.quantum_info import Pauli, Statevector

    state = Statevector(state)
    return np.array([float(state.expectation_value(Pauli("".join(reversed([
        "IXYZ"[(word >> (2 * index)) & 3] for index in indices])))).real)
        for word in range(1, 4 ** len(indices))])


def _verify_samples(execution, expected, source, indices, epsilon, shots):
    import numpy as np

    if not execution["all_detectors_constant"]:
        raise RuntimeError("noiseless execution has nonconstant detectors")
    if execution["discarded"] or len(execution["samples"]) != shots:
        raise RuntimeError("unitary execution unexpectedly discarded shots")
    exact_signature, source_signature = _signature(expected, indices), _signature(source, indices)
    verified = []
    for sample in execution["samples"]:
        actual = np.asarray(sample["logical_signature"])
        if actual.shape != exact_signature.shape or not np.allclose(actual, exact_signature, atol=1e-7, rtol=0):
            raise RuntimeError("VM output differs from the lowered circuit")
        fidelity = float((1 + actual @ exact_signature) / len(expected))
        source_fidelity = float((1 + actual @ source_signature) / len(source))
        source_pauli_error = float(np.max(np.abs(actual - source_signature)))
        if abs(fidelity - 1) > 1e-7:
            raise RuntimeError("VM output fails the lowered full-state fidelity check")
        if source_fidelity < 1 - epsilon**2 - 1e-7 or source_pauli_error > 2 * epsilon + 1e-7:
            raise RuntimeError("VM output exceeds the original MQT source synthesis budget")
        verified.append(dict(shot=sample["shot"], lowered_state_fidelity=fidelity,
                             source_state_fidelity=source_fidelity, source_max_pauli_error=source_pauli_error))
    return verified


def _vm_worker(case_dir, compiler, shots):
    import bloq
    import numpy as np

    run_dir = case_dir / compiler
    result = _read(run_dir / "result.json")
    started = time.perf_counter()
    stage = "compile_d3"
    result["vm_d3"] = dict(status="running", stage=stage)
    save_json(run_dir / "result.json", result)
    try:
        graph = bloq.BlockGraph.load(run_dir / "layout.blog")
        program = bloq.compile(graph, distance=3, prepare_t_with_mpps=True, validate=True)
        path = run_dir / "program_d3.bloq"
        program.save(path)
        if bloq.Bloq.load(path).to_binary() != program.to_binary():
            raise RuntimeError("d=3 binary round trip changed IR")
        indices = _output_indices(program, result["output_ports"])
        stage = "execution"
        result["vm_d3"].update(stage=stage)
        save_json(run_dir / "result.json", result)
        execution = bloq.run_noiseless(program, shots=shots, seed=17, input_state="zero", include_signature=True)
        save_json(run_dir / "execution.json", execution)
        stage = "state_verification"
        states = []
        for name in ("expected_state", "source_state"):
            state = _read(case_dir / f"{name}.json")
            states.append(np.asarray(state["real"]) + 1j * np.asarray(state["imag"]))
        verified = _verify_samples(execution, *states, indices, result["approximation_bound"], shots)
        result["vm_d3"] = dict(status="sampled", distance=3, t_preparation="ideal_t_proxy", shots=verified,
                               seconds=time.perf_counter() - started, peak_rank=execution["peak_rank"],
                               sha256=_digest(path), checked_paulis=4 ** len(indices) - 1,
                               seed=17, ideal_boundary_capture=True)
    except Exception as error:
        result["vm_d3"] = dict(_failure(error), failed_stage=stage, seconds=time.perf_counter() - started)
        traceback.print_exc()
    save_json(run_dir / "result.json", result)


def classify(result):
    """Keep successful compilation and independent semantic checks distinct."""
    if result.get("ir_d25", {}).get("status") != "compiled":
        return "not compiled"
    checks = [result.get(name, {}).get("status") for name in ("quizx", "vm_d3")]
    if "failed" in checks:
        return "compiled; check failed"
    labels = [label for label, status in zip(("QuiZX branches", "VM shots"), checks) if status == "sampled"]
    return "compiled; sampled " + " and ".join(labels) if labels else "compilation only"


def _reconcile(result, stage, outcome):
    current = result.get(stage, {})
    if outcome["status"] != "completed" and current.get("status") not in ("compiled", "passed", "sampled", "failed"):
        result[stage] = dict(outcome, failed_stage=current.get("stage", stage))
        if stage in ("layout", "ir_d25"):
            result.update(status="failed", failed_stage=stage)
    result.setdefault("processes", {})[stage] = outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("target/mqt-bench"))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--compilers", nargs="+", choices=tuple(LAYOUT_OPTIONS), default=list(LAYOUT_OPTIONS))
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", "--compile-timeout", dest="compile_timeout", type=float, default=180)
    parser.add_argument("--vm-timeout", type=float, default=120)
    parser.add_argument("--quizx-timeout", type=float, default=120)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--quizx", type=Path, default=ROOT / "target/release/examples/verify_interop_layout")
    parser.add_argument("--worker", choices=("lower", "compile", "vm"), help=argparse.SUPPRESS)
    parser.add_argument("--case-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--compiler", choices=tuple(LAYOUT_OPTIONS), help=argparse.SUPPRESS)
    parser.add_argument("--limit-exec", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.limit_exec:
        memory, *command = args.limit_exec
        resource.setrlimit(resource.RLIMIT_AS, (int(memory), int(memory)))
        os.execvp(command[0], command)
    if args.worker:
        if args.worker == "lower":
            return _lower_worker(args.case_dir, args.epsilon)
        if args.worker == "compile":
            return _compile_worker(args.case_dir, args.compiler)
        return _vm_worker(args.case_dir, args.compiler, args.shots)
    from _mqt_bench import SUITE

    unknown = set(args.cases or ()) - {benchmark.slug for benchmark in SUITE}
    if unknown:
        parser.error("unknown cases: " + ", ".join(sorted(unknown)))
    if (args.jobs < 1 or args.shots < 1 or not 0 < args.epsilon < 1
            or any(not math.isfinite(value) or value <= 0 for value in
                   (args.compile_timeout, args.vm_timeout, args.quizx_timeout, args.memory_gib))):
        parser.error("jobs/shots/budgets must be positive and epsilon between zero and one")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty; retained runs are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(_provenance(output, args.compilers),
                    started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=platform.platform(),
                    python=platform.python_version(), cpus=os.cpu_count(), epsilon=args.epsilon, shots=args.shots,
                    compile_timeout=args.compile_timeout, vm_timeout=args.vm_timeout, quizx_timeout=args.quizx_timeout,
                    memory_gib=args.memory_gib, jobs=args.jobs, vm_distance=3, ir_distance=25, layout_options=LAYOUT_OPTIONS)
    save_json(output / "manifest.json", manifest)
    rows, jobs = [], []
    def checkpoint():
        save_json(output / "summary.json", dict(manifest=manifest, runs=rows))

    for benchmark in SUITE:
        if args.cases and benchmark.slug not in args.cases:
            continue
        case_dir = output / "cases" / benchmark.slug
        case_dir.mkdir(parents=True)
        print(f"[lower] {benchmark.slug}", flush=True)
        outcome = _bounded(_worker_command("lower", case_dir, epsilon=args.epsilon), case_dir,
                           "lowering", args.compile_timeout, args.memory_gib)
        result = _read(case_dir / "lowering.json")
        result["process"] = outcome
        save_json(case_dir / "lowering.json", result)
        if result.get("status") != "lowered":
            if result.get("status") not in ("failed", "rejected", "inconclusive"):
                result.update(case=benchmark.slug, benchmark=benchmark.name, tier=benchmark.tier,
                              status="inconclusive" if outcome["status"] == "inconclusive" else "failed",
                              error=outcome.get("reason", "lowering process returned no valid circuit"))
            save_json(case_dir / "lowering.json", result)
            rows.append(result)
            checkpoint()
            continue
        jobs.extend((benchmark, case_dir, compiler) for compiler in benchmark.compilers if compiler in args.compilers)

    def run(job):
        benchmark, case_dir, compiler = job
        run_dir = case_dir / compiler
        run_dir.mkdir()
        for name in ("source.qasm", "clifford_t.qasm", "benchmark.qasm"):
            if (case_dir / name).exists():
                shutil.copyfile(case_dir / name, run_dir / name)
        result = dict(_read(case_dir / "lowering.json"), case=benchmark.slug, compiler=compiler,
                      run_dir=str(run_dir.relative_to(output)), status="running", stage="layout")
        save_json(run_dir / "result.json", result)
        print(f"[start] {benchmark.slug} {compiler}", flush=True)
        outcome = _bounded(_worker_command("compile", case_dir, compiler=compiler), run_dir,
                           "compile", args.compile_timeout, args.memory_gib)
        result = _read(run_dir / "result.json")
        _reconcile(result, result.get("stage", result.get("failed_stage", "ir_d25")), outcome)
        result.pop("stage", None)
        save_json(run_dir / "result.json", result)
        if result.get("ir_d25", {}).get("status") == "compiled" and benchmark.tier == "execute":
            outcome = _bounded(_worker_command("vm", case_dir, compiler=compiler, shots=args.shots), run_dir,
                               "vm_d3", args.vm_timeout, args.memory_gib)
            result = _read(run_dir / "result.json")
            _reconcile(result, "vm_d3", outcome)
        else:
            result["vm_d3"] = dict(status="skipped", reason="compile tier or d=25 compilation failed")
        if not (run_dir / "layout.blog").exists():
            result["quizx"] = dict(status="skipped", reason="no retained layout")
        elif result["logical_qubits"] > QUIZX_QUBITS or not args.quizx.is_file():
            result["quizx"] = dict(status="inconclusive", reason="size limit or checker unavailable")
        else:
            save_json(run_dir / "result.json", result)
            outcome = _bounded([str(args.quizx.resolve()), str(run_dir), "32", "7"], run_dir,
                               "quizx", args.quizx_timeout, args.memory_gib)
            result["quizx"] = outcome
            if outcome["status"] == "completed":
                try:
                    report = json.loads((run_dir / outcome["stdout"]).read_text())
                    result["quizx"].update(report, status="sampled" if report["verified"] == report["samples"] else "inconclusive")
                except (ValueError, KeyError) as error:
                    result["quizx"].update(status="failed", error=f"invalid checker report: {error}")
        result["verification"] = classify(result)
        save_json(run_dir / "result.json", result)
        print(f"[done] {benchmark.slug} {compiler}: {result['verification']}", flush=True)
        return result

    checkpoint()
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            checkpoint()


if __name__ == "__main__":
    main()
