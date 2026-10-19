"""Regenerate small TopoLS channel fixtures with both audited layout engines.

Requires the pinned checkout and its matching topols_core extension. Run from
the repository root with --upstream-root /path/to/TopoLS; no upstream pickle
is loaded. The checkout must contain exactly the checked-in integration patch.
"""

import argparse
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import runpy
import subprocess
from urllib.parse import unquote, urlparse

import topols

from bloq.interop.topols import AUDITED_REVISION, from_qasm

SOURCES = {
    "s": (1, "s q[0];"),
    "sdg": (1, "sdg q[0];"),
    "t": (1, "t q[0];"),
    "tdg": (1, "tdg q[0];"),
    "thth": (1, "t q[0]; h q[0]; t q[0]; h q[0];"),
    "paulis": (1, "h q[0]; x q[0]; z q[0];"),
    "entangled_t": (2, "h q[0]; cx q[0],q[1]; t q[1];"),
    "entangled_st": (2, "h q[0]; s q[0]; t q[0]; cx q[0],q[1];"),
    "entangled_tdg": (2, "h q[0]; cx q[0],q[1]; tdg q[1]; cx q[1],q[0];"),
}


def checked_upstream(repository, upstream):
    upstream = upstream.resolve()
    if Path(topols.__file__).resolve() != upstream / "src/topols/__init__.py":
        raise RuntimeError("Imported TopoLS does not come from --upstream-root")

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(upstream), *arguments], text=True)

    revision = git("rev-parse", "HEAD").strip()
    patch = repository / "bloq_py/examples/topols.patch"
    if revision != AUDITED_REVISION or git("diff", "HEAD") != patch.read_text():
        raise RuntimeError("Upstream must match the audited revision plus exactly topols.patch")
    if git("diff", "--cached"):
        raise RuntimeError("Upstream checkout contains staged changes")
    direct = metadata.distribution("topols-core").read_text("direct_url.json")
    source = json.loads(direct or "{}")
    url = urlparse(source.get("url", ""))
    if (url.scheme != "file" or url.netloc not in ("", "localhost")
            or not source.get("dir_info", {}).get("editable")
            or Path(unquote(url.path)).resolve() != upstream / "rust/topols_core"):
        raise RuntimeError("topols-core must be installed editable from this checkout's rust/topols_core")
    return dict(upstream_revision=revision, upstream_patch="bloq_py/examples/topols.patch",
                upstream_patch_sha256=sha256(patch.read_bytes()).hexdigest())


def fixture_sources(repository):
    sources = {
        name: dict(qasm=f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}\n', seed=0)
        for name, (width, gates) in SOURCES.items()
    }
    generate = runpy.run_path(str(repository / "bloq_py/examples/_algorithm_candidates.py"))["generate_candidate"]
    for name, algorithm, seed in (("ising4", "ising4", 0), ("adder4", "adder4", 0),
                                  ("adder4_early_t", "adder4", 1), ("qae3", "qae3", 0)):
        candidate = generate(algorithm)
        sources[name] = dict(
            qasm=candidate.clifford_t_qasm + "\n", seed=seed, algorithm=candidate.summary,
            source_qasm=candidate.source_qasm,
            expected_state=[[float(a.real), float(a.imag)] for a in candidate.ideal_unitary[:, 0]],
        )
    return sources


def main():
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    args = parser.parse_args()
    provenance = checked_upstream(repository, args.upstream_root)
    directory = repository / "bloq_vm/tests/data/topols"
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in fixture_sources(repository).items():
        qasm = source["qasm"]
        algorithm = "algorithm" in source
        options = dict(seed=source["seed"], iterations=100 if algorithm else 10,
                       time_bound=0.1 if algorithm else 0.05)
        python = from_qasm(qasm, engine="python", **options)
        rust = from_qasm(qasm, engine="rust", **options)
        def plain(value):
            return json.loads(json.dumps(value, default=int))
        assert plain(python.embedding) == plain(rust.embedding), f"{name}: layout engines disagree"
        assert python.graph.to_text() == rust.graph.to_text(), f"{name}: native graphs disagree"
        (directory / f"{name}.blog").write_text(python.graph.to_text().rstrip() + "\n")
        (directory / f"{name}.qasm").write_text(qasm)
        info = provenance | dict(
            input_ports=python.input_ports, output_ports=python.output_ports,
            normalized_qasm=python.normalized_qasm,
            t_gadgets=sum(t == 5 for t in python.embedding["type_hist"].values()),
            layout_options=python.layout_options,
        )
        if algorithm:
            info.update({key: source[key] for key in ("algorithm", "source_qasm", "expected_state")})
        (directory / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(name, python.graph.block_count, flush=True)


if __name__ == "__main__":
    main()
