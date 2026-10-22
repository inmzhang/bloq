"""Failure isolation and independent state checks for retained MQT runs."""

from importlib import import_module
import json
from pathlib import Path
import signal
import subprocess
import sys
import types

import pytest


@pytest.fixture
def driver(monkeypatch):
    pytest.importorskip("resource", reason="experiment runner uses POSIX resource limits")
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    return import_module("mqt_bench_experiments")


def test_bounded_process_applies_limit_and_retains_logs(driver, tmp_path):
    result = driver._bounded([sys.executable, "-c",
                              "import resource; print(resource.getrlimit(resource.RLIMIT_AS)[0])"],
                             tmp_path, "limit", 5, memory_gib=1)
    assert result["status"] == "completed"
    assert int((tmp_path / result["stdout"]).read_text()) == 1024**3
    assert (tmp_path / result["stderr"]).exists()


@pytest.mark.parametrize("termination", [signal.SIGSEGV, signal.SIGABRT, signal.SIGKILL, None]
                         + ([signal.SIGRTMIN + 1] if hasattr(signal, "SIGRTMIN") else []))
def test_process_crashes_are_failures_without_resource_limit_evidence(driver, tmp_path, termination):
    code = ("raise MemoryError('test allocation')" if termination is None else
            "import os,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
            f"os.kill(os.getpid(),{int(termination)})")
    result = driver._bounded([sys.executable, "-c", code], tmp_path, "crash", 5)
    if termination is None:
        assert result["status"] == "inconclusive"
        assert result["reason"] == "process exceeded memory bound"
    else:
        assert result["status"] == "failed"
        assert result["returncode"] == -termination
        assert result["reason"] == f"process terminated by signal {termination}: {signal.strsignal(termination)}"


@pytest.mark.parametrize("state", ["audited", "extra_change", "wrong_revision", "staged"])
def test_provenance_requires_exact_audited_upstream_patch(driver, monkeypatch, tmp_path, state):
    def git(path, *arguments):
        return subprocess.check_output(["git", "-C", str(path), *arguments])

    repository, upstream, output = [tmp_path / name for name in ("bloq", "upstream", "output")]
    for checkout in (repository, upstream):
        checkout.mkdir()
        git(checkout, "init", "-q")
        (checkout / "source.py").write_text("original = True\n")
        git(checkout, "add", ".")
        git(checkout, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    revision = git(upstream, "rev-parse", "HEAD").decode().strip()
    source = upstream / "source.py"
    source.write_text("patched = True\n")
    patch = repository / "bloq_py/examples/topols.patch"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(git(upstream, "diff", "HEAD"))
    if state == "extra_change":
        source.write_text(source.read_text() + "unaudited = True\n")
    elif state == "staged":
        git(upstream, "add", ".")
    monkeypatch.setattr(driver, "ROOT", repository)
    monkeypatch.setattr(driver, "import_module", lambda name:
                        types.SimpleNamespace(__file__=source) if name == "topols" else
                        types.SimpleNamespace(AUDITED_REVISION="wrong" if state == "wrong_revision" else revision))
    output.mkdir()
    if state != "audited":
        with pytest.raises(RuntimeError, match="staged changes" if state == "staged" else "audited revision"):
            driver._provenance(output, ["topols"])
    else:
        provenance = driver._provenance(output, ["topols"])["upstream"]["topols"]
        assert provenance["revision"] == revision
        assert provenance["patch_applied"]
        assert (output / provenance["diff_file"]).read_bytes() == patch.read_bytes()


def test_timeout_kills_descendants_and_preserves_compilation(driver, tmp_path):
    child_pid = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"open({str(child_pid)!r},'w').write(str(p.pid)); time.sleep(60)")
    outcome = driver._bounded([sys.executable, "-c", code], tmp_path, "vm", .5)
    assert outcome["status"] == "inconclusive"
    pid = int(child_pid.read_text())
    status = Path(f"/proc/{pid}/stat")
    try:
        assert status.read_text().split()[2] == "Z"
    except (FileNotFoundError, ProcessLookupError):
        pass
    result = dict(status="compiled", ir_d25=dict(status="compiled", round_trip=True),
                  vm_d3=dict(status="running", stage="execution"))
    driver._reconcile(result, "vm_d3", outcome)
    assert result["status"] == "compiled"
    assert result["ir_d25"]["round_trip"]
    assert result["vm_d3"]["status"] == "inconclusive"
    assert driver.classify(result) == "compilation only"


@pytest.mark.parametrize("requested", [None, {"spacing": 4}])
@pytest.mark.parametrize("compiler", ["topologiq", "topols", "spiderls"])
def test_compile_worker_compiles_d25_first_with_proxy_and_binary_roundtrip(driver, monkeypatch, tmp_path,
                                                                        requested, compiler):
    run_dir = tmp_path / compiler
    run_dir.mkdir()
    (tmp_path / "clifford_t.qasm").write_text("input")
    initial = dict(status="running", stage="layout")
    if requested is not None:
        initial["requested_layout_options"] = requested
    driver.save_json(run_dir / "result.json", initial)
    graph = types.SimpleNamespace(
        save=lambda path: path.write_text("BLOG"),
        export_html_viewer=lambda path: pytest.fail("IR compilation must not render a viewer"),
        to_text=lambda: "BLOG", flatten=lambda: graph, positions=lambda: [(0, 0, 0), (0, 0, 1)],
        pipes=lambda: [], block_count=2, pipe_count=1, t_count=0)
    program = types.SimpleNamespace(
        save=lambda path: path.write_bytes(b"IR"), to_binary=lambda: b"IR",
        logical_outputs=lambda: [types.SimpleNamespace(port=(0, 0, 1))],
        qubit_count=12, measurement_count=8, stats=lambda: types.SimpleNamespace(node_count=3))
    calls = []
    clock = [0.0]
    monkeypatch.setattr(driver.time, "perf_counter", lambda: clock[0])

    def compile_graph(actual, **options):
        assert actual is graph
        calls.append(options)
        clock[0] += 5
        return program

    fake_bloq = types.SimpleNamespace(compile=compile_graph, Bloq=types.SimpleNamespace(load=lambda path: program),
                                      BlockGraph=types.SimpleNamespace(load=lambda path: graph))
    monkeypatch.setitem(sys.modules, "bloq", fake_bloq)
    actual_options = {"spacing": 4, "effective": "upstream settings"}

    def from_qasm(qasm, **options):
        assert options == (requested if requested is not None else driver.LAYOUT_OPTIONS[compiler])
        clock[0] += 2
        return types.SimpleNamespace(graph=graph, input_ports=[(0, 0, 0)], output_ports=[(0, 0, 1)],
                                     layout_options=actual_options, upstream_revision="test", normalized_qasm="normalized")

    attempts = (dict(status="rejected", native_seconds=3), dict(status="accepted", native_seconds=2))

    def compile_qasm(qasm, **options):
        calls.append(options)
        assert qasm == "input"
        imported = from_qasm(qasm, **{key: value for key, value in options.items()
                                     if key not in ("compiler", "distance", "prepare_t_with_mpps")})
        clock[0] += 5
        return types.SimpleNamespace(imported=imported, program=program, placement_attempts=attempts)

    adapter = types.SimpleNamespace(AUDITED_REVISION="test", from_qasm=from_qasm)
    monkeypatch.setattr(driver, "import_module", lambda name:
                        types.SimpleNamespace(compile_qasm=compile_qasm) if name == "bloq.interop" else adapter)
    driver._compile_worker(tmp_path, compiler)
    result = json.loads((run_dir / "result.json").read_text())
    if compiler == "topologiq":
        assert calls == [dict(distance=25, prepare_t_with_mpps=True, validate=True)]
        assert "placement_attempts" not in result
    else:
        assert calls == [dict(compiler=compiler, distance=25, prepare_t_with_mpps=True,
                              **(requested if requested is not None else driver.LAYOUT_OPTIONS[compiler]))]
        assert result["placement_attempts"] == list(attempts)
    assert result["status"] == "compiled"
    assert result["ir_d25"]["round_trip"]
    assert result["layout_options"] == actual_options
    assert result["layout_seconds"] == 2
    assert result["ir_d25"]["seconds"] == 5
    assert result["pipeline_seconds"] == 7
    assert not (run_dir / "program_d3.bloq").exists()


@pytest.mark.parametrize("native", [True, False])
def test_native_certified_worker_keeps_native_and_placement_failure_stages(driver, monkeypatch, tmp_path, native):
    import bloq
    from bloq.interop._geometry import GeometryError

    run_dir = tmp_path / "spiderls"
    run_dir.mkdir()
    (tmp_path / "clifford_t.qasm").write_text("input")
    error = bloq.CompileError("native lifetime conflict") if native else GeometryError(
        "resource_placement_limit", "bounded search exhausted")

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(driver, "import_module", lambda name: types.SimpleNamespace(compile_qasm=reject))
    driver._compile_worker(tmp_path, "spiderls")
    result = json.loads((run_dir / "result.json").read_text())
    stage = "ir_d25" if native else "layout"
    assert result["failed_stage"] == stage
    assert result[stage]["error"] == str(error)
    assert result[stage]["status"] == ("failed" if native else "inconclusive")


def test_vm_keeps_exact_lowered_check_and_source_error_bound(driver):
    np = pytest.importorskip("numpy")
    pytest.importorskip("qiskit")
    expected = np.array([1, np.exp(0.01j)]) / np.sqrt(2)
    source = np.array([1, 1]) / np.sqrt(2)
    # Reversed port order must still compare against source-wire state order.
    expected = np.kron(expected, [1, 0])
    source = np.kron(source, [1, 0])
    signature = driver._signature(expected, [1, 0]).tolist()
    execution = dict(all_detectors_constant=True, discarded=0,
                     samples=[dict(shot=0, logical_signature=signature)])
    verified = driver._verify_samples(execution, expected, source, [1, 0], .02, 1)
    assert verified[0]["lowered_state_fidelity"] == pytest.approx(1)
    with pytest.raises(RuntimeError, match="source synthesis budget"):
        driver._verify_samples(execution, expected, source, [1, 0], .0001, 1)
    execution["samples"][0]["logical_signature"][0] += 5e-7
    with pytest.raises(RuntimeError, match="lowered circuit"):
        driver._verify_samples(execution, expected, source, [1, 0], .02, 1)


def test_failed_vm_does_not_become_failed_compilation(driver):
    result = dict(status="compiled", ir_d25=dict(status="compiled"),
                  quizx=dict(status="sampled"), vm_d3=dict(status="failed"))
    assert driver.classify(result) == "compiled; check failed"


@pytest.mark.parametrize("message", [
    "engine error: stabilizer rank 2097152 exceeds cap 1048576",
    "Boolean work steps requires 128000001, limit is 128000000; retry with larger limits",
    "graph: Boolean nodes requires 1001, limit is 1000",
    "row witness operations requires 1001, limit is 1000",
    "runtime step cap 100 exceeded",
    "physical readout binding expansion exceeds 100 work units",
    "Routing exceeded max_physical_layers with unresolved measurements: m12",
])
def test_resource_exhaustion_is_inconclusive_with_original_error(driver, message):
    result = driver._failure(RuntimeError(message))
    assert result["status"] == "inconclusive"
    assert result["error"] == message
    assert result["error_type"] == "RuntimeError"
    assert result["reason"] == "resource limit exhausted"


@pytest.mark.parametrize("message", [
    "VM output differs from the lowered circuit",
    "required control parity is unavailable",
    "stabilizer rank is invalid",
    "unknown runtime error",
])
def test_arbitrary_errors_remain_failures(driver, message):
    assert driver._failure(RuntimeError(message))["status"] == "failed"


def test_bounded_placement_exhaustion_is_not_a_geometry_disproof(driver):
    class PlacementLimit(ValueError):
        code = "resource_placement_limit"

    assert driver._failure(PlacementLimit("candidate budget exhausted"))["status"] == "inconclusive"


@pytest.mark.parametrize("worker_status,expected", [("inconclusive", "inconclusive"), ("running", "failed")])
def test_successful_process_cannot_erase_missing_lowering_evidence(driver, monkeypatch, tmp_path,
                                                                 worker_status, expected):
    benchmark = types.SimpleNamespace(slug="limited_3", name="limited", tier="compile", layout_options={})
    monkeypatch.setitem(sys.modules, "_mqt_bench", types.SimpleNamespace(SUITE=[benchmark]))
    monkeypatch.setattr(driver, "_provenance", lambda *args: {})
    monkeypatch.setattr(sys, "argv", ["mqt_bench_experiments", "--output-dir", str(tmp_path)])

    def completed_without_circuit(command, directory, *args):
        driver.save_json(directory / "lowering.json", dict(status=worker_status))
        return dict(status="completed", returncode=0)

    monkeypatch.setattr(driver, "_bounded", completed_without_circuit)
    driver.main()
    row, = json.loads((tmp_path / "summary.json").read_text())["runs"]
    assert row["status"] == expected


@pytest.mark.parametrize("raw_benchmark", [None, "unmodified MQT source\n"])
def test_lower_worker_retains_optional_raw_benchmark(driver, monkeypatch, tmp_path, raw_benchmark):
    benchmark = types.SimpleNamespace(slug=tmp_path.name, name="full_adder", size=4,
                                      tier="execute", note="explicit input preparation")
    lowered = types.SimpleNamespace(source_qasm="prepared source", clifford_t_qasm="lowered source",
                                    summary={}, expected_state=None, source_state=None,
                                    benchmark_qasm=raw_benchmark)
    monkeypatch.setitem(sys.modules, "_mqt_bench", types.SimpleNamespace(
        SUITE=[benchmark], UnsupportedSource=ValueError, lower=lambda *args, **kwargs: lowered))
    driver._lower_worker(tmp_path, .01)
    result = json.loads((tmp_path / "lowering.json").read_text())
    assert result["status"] == "lowered"
    assert (tmp_path / "source.qasm").read_text() == "prepared source"
    if raw_benchmark is None:
        assert not (tmp_path / "benchmark.qasm").exists()
        assert "benchmark_sha256" not in result
    else:
        assert (tmp_path / "benchmark.qasm").read_text() == raw_benchmark
        assert result["benchmark_sha256"] == driver._digest(tmp_path / "benchmark.qasm")
