"""Shared artifact retention and complete-state checks for exact algorithm demos."""

from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from qiskit.quantum_info import Pauli, Statevector

from _algorithm_candidates import check_algorithm_probabilities


def save_json(path: Path, value):
    # A timed-out worker must leave either the previous checkpoint or the next one.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_demo(source_qasm, clifford_t_qasm, ideal_unitary, output_dir: Path, summary,
             *, shots=3, synthesis_only=False, compile_only=False, seed=17, layout_seed=1330,
             compiler="topologiq", layout_options=None):
    """Retain each stage; verify every output correlation against an ideal matrix.

    The caller checks the exported source and Clifford+T channels against the
    independent matrix. Every circuit includes its preparation from zero inputs.
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    if compiler not in ("topologiq", "topols", "spiderls"):
        raise ValueError("compiler must be topologiq, topols, or spiderls")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("--output-dir must be empty; each run retains its own artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("source.qasm", source_qasm), ("clifford_t.qasm", clifford_t_qasm)):
        (output_dir / name).write_text(text + "\n", encoding="utf-8")
    summary.update(compiler=compiler, input_state="zero", qiskit_version=version("qiskit"), status="synthesized",
                   clifford_t_sha256=sha256((clifford_t_qasm + "\n").encode()).hexdigest())

    def save():
        save_json(output_dir / "result.json", summary)

    save()
    if synthesis_only:
        return summary

    import bloq

    stage = "layout"
    try:
        options = dict(layout_options or {})
        if compiler != "spiderls":
            options.setdefault("seed", layout_seed)
        summary.update(status="running", stage=stage, layout_options=options)
        save()
        print(f"Routing with {compiler}...", file=sys.stderr, flush=True)
        pipeline_started = time.perf_counter()
        adapter = import_module(f"bloq.interop.{compiler}")
        program, native_seconds, compile_warnings = None, 0.0, []
        if compiler in ("topols", "spiderls"):
            try:
                with warnings.catch_warnings(record=True) as compile_warnings:
                    warnings.simplefilter("always")
                    compiled = import_module("bloq.interop").compile_qasm(
                        clifford_t_qasm, compiler=compiler, distance=3, prepare_t_with_mpps=True, **options)
            except bloq.CompileError:
                stage = "compile"
                raise
            imported, program = compiled.imported, compiled.program
            summary["placement_attempts"] = compiled.placement_attempts
            native_seconds = sum(attempt["native_seconds"] for attempt in compiled.placement_attempts)
        else:
            imported = adapter.from_qasm(clifford_t_qasm, **options)
        if compiler == "topols":
            (output_dir / "normalized.qasm").write_text(imported.normalized_qasm, encoding="utf-8")
        imported.graph.save(output_dir / "layout.blog")
        imported.graph.export_html_viewer(output_dir / "layout.html")
        flat = imported.graph.flatten()
        hadamards = {axis: sum(pipe.hadamard and pipe.src[i] != pipe.dst[i]
                               for pipe in flat.pipes()) for i, axis in enumerate("XYZ")}
        summary.update(
            layout_seconds=time.perf_counter() - pipeline_started - native_seconds, layout_options=imported.layout_options,
            layout_blocks=flat.block_count, hadamard_pipes_by_axis=hadamards,
            pyzx_version=version("pyzx"), bloq_version=bloq.__version__,
            input_ports=imported.input_ports, output_ports=imported.output_ports,
        )
        summary.update({f"audited_{compiler}_revision": adapter.AUDITED_REVISION,
                        f"detected_{compiler}_revision": imported.upstream_revision})
        stage = "compile"
        summary["stage"] = stage
        save()
        print("Compiling validated IR with ideal T-source proxies...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        if program is None:
            with warnings.catch_warnings(record=True) as compile_warnings:
                warnings.simplefilter("always")
                program = bloq.compile(imported.graph, distance=3, prepare_t_with_mpps=True, validate=True)
        for warning in compile_warnings:
            warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
        program.save(output_dir / "program.bloq")
        summary.update(
            compile_seconds=native_seconds + time.perf_counter() - started,
            pipeline_seconds=time.perf_counter() - pipeline_started, distance=3, t_preparation="ideal_t_proxy",
            physical_qubits=program.qubit_count, ir_nodes=program.stats().node_count,
            compile_warnings=[str(w.message) for w in compile_warnings], status="compiled",
        )
        save()
        outputs = program.logical_outputs()
        indices = {output.port: i for i, output in enumerate(outputs)}
        if (len(outputs) != len(imported.output_ports) or len(indices) != len(outputs)
                or set(indices) != set(imported.output_ports)):
            raise RuntimeError("VM output interface differs from the source circuit")
        if compile_only:
            return summary
        qubits = len(imported.output_ports)
        state = Statevector(ideal_unitary[:, 0].copy())
        dimension = 1 << qubits
        z_words = [sum(3 << (2 * indices[port]) for wire, port in enumerate(imported.output_ports)
                       if mask & (1 << wire)) for mask in range(1, dimension)]
        # Invert the Z-moment Walsh transform in source wire order.
        signs = np.array([[1 - 2 * ((basis & mask).bit_count() % 2)
                           for mask in range(dimension)] for basis in range(dimension)]) / dimension
        expected = []
        for word in range(1, 4**qubits):
            # VM words use coordinate order; Qiskit labels use reversed source order.
            label = ["IXYZ"[(word >> (2 * indices[port])) & 3] for port in imported.output_ports]
            expected.append(float(state.expectation_value(Pauli("".join(reversed(label)))).real))
        stage = "vm_execution"
        summary["stage"] = stage
        save()
        print(f"Executing {shots} noiseless VM shots...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        execution = bloq.run_noiseless(program, shots=shots, seed=seed,
                                      input_state="zero", include_signature=True)
        summary["execution_seconds"] = time.perf_counter() - started
        save_json(output_dir / "execution.json", execution)
        stage = "vm_verification"
        if not execution["all_detectors_constant"]:
            raise RuntimeError("noiseless VM execution has nonconstant detectors")
        if execution["discarded"] or len(execution["samples"]) != shots:
            raise RuntimeError("unitary demo execution unexpectedly discarded shots")
        verified = []
        for sample in execution["samples"]:
            actual = sample["logical_signature"]
            if not np.allclose(actual, expected, atol=1e-7, rtol=0):
                raise RuntimeError("joint VM output differs from the independent algorithm state")
            fidelity = float((1 + np.dot(actual, expected)) / (1 << qubits))
            if abs(fidelity - 1) > 1e-7:
                raise RuntimeError("joint VM output fails the full-state fidelity check")
            probabilities = signs @ np.array([1, *(actual[word - 1] for word in z_words)])
            if not np.allclose(probabilities, state.probabilities(), atol=1e-7, rtol=0):
                raise RuntimeError("VM measurement probabilities differ from the algorithm")
            verified.append(dict(shot=sample["shot"], joint_state_fidelity=fidelity,
                                 logical_probabilities=probabilities.tolist(),
                                 algorithm_checks=check_algorithm_probabilities(summary, probabilities),
                                 logical_bloch=[sample["logical_bloch"][indices[p]] for p in imported.output_ports],
                                 captured_output_ports=sample["captured_output_ports"]))
        summary.update(status="executed", noiseless=True, ideal_boundary_capture=True,
                       execution_model="noiseless physical circuits with ideal T sources and boundary capture",
                       attempted_shots=execution["shots"], discarded_shots=execution["discarded"],
                       root_seed=seed, peak_rank=execution["peak_rank"], shots=verified,
                       vm_verification=f"all {4**qubits - 1} joint Pauli expectations and full-state fidelity")
        summary.pop("stage", None)
        save()
        return summary
    except Exception as error:
        summary.update(status="failed", failed_stage=stage, error=str(error), error_type=type(error).__name__)
        if hasattr(error, "code"):
            summary.update(error_code=error.code, error_nodes=getattr(error, "nodes", ()))
            summary["failure_layer"] = "upstream" if error.code in ("upstream_layout", "upstream_flow") else "adapter"
        else:
            summary["failure_layer"] = stage
        save()
        raise
