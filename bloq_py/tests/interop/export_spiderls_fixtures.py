"""Regenerate independent channel fixtures from the pinned, patched SpiderLS.

Run with the checkout's ``src`` on PYTHONPATH and pass --upstream-root.
Only the audited revision plus exactly ``examples/spiderls.patch`` is accepted.
"""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess

import pyzx_execution_order

from bloq.interop.spiderls import AUDITED_REVISION, from_qasm

SOURCES = {
    "identity": (1, ""),
    "h": (1, "h q[0];"),
    "z": (1, "z q[0];"),
    "hzh": (1, "h q[0];\nz q[0];\nh q[0];"),
    "s": (1, "s q[0];"),
    "sdg": (1, "sdg q[0];"),
    "t": (1, "t q[0];"),
    "tdg": (1, "tdg q[0];"),
    "phase3": (1, "rz(3*pi/4) q[0];"),
    "phase5": (1, "rz(5*pi/4) q[0];"),
    "thth": (1, "t q[0];\nh q[0];\nt q[0];\nh q[0];"),
    "entangled_t": (2, "h q[0];\ncx q[0],q[1];\nt q[1];"),
    "passthrough": (2, "t q[0];"),
    "crossed_h": (2, "h q[0];\nswap q[0],q[1];"),
    "lifecycle": (2, "t q[0];\nh q[0];\nt q[1];\nh q[0];\ncx q[1],q[0];\nt q[0];\nh q[1];\ncx q[1],q[0];\nt q[0];\nh q[0];\ncx q[0],q[1];\nh q[0];\ncx q[1],q[0];"),
}


def checked_upstream(repository, upstream):
    upstream = upstream.resolve()
    if Path(pyzx_execution_order.__file__).resolve() != upstream / "src/pyzx_execution_order.py":
        raise RuntimeError("Imported SpiderLS does not come from --upstream-root")

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(upstream), *arguments], text=True)

    revision = git("rev-parse", "HEAD").strip()
    patch = repository / "bloq_py/examples/spiderls.patch"
    if revision != AUDITED_REVISION or git("diff", "HEAD") != patch.read_text():
        raise RuntimeError("Upstream must match the audited revision plus exactly spiderls.patch")
    if git("diff", "--cached"):
        raise RuntimeError("Upstream checkout contains staged changes")
    return dict(upstream_revision=revision, upstream_patch="bloq_py/examples/spiderls.patch",
                upstream_patch_sha256=sha256(patch.read_bytes()).hexdigest())


def fixture_sources(repository):
    sources = {
        name: dict(qasm=f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{width}];\n{gates}'.rstrip() + "\n")
        for name, (width, gates) in SOURCES.items()
    }
    generate = runpy.run_path(str(repository / "bloq_py/examples/_algorithm_candidates.py"))["generate_candidate"]
    candidate = generate("walk3")
    sources["walk3"] = dict(
        qasm=candidate.clifford_t_qasm + "\n", algorithm=candidate.summary,
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
    directory = repository / "bloq_vm/tests/data/spiderls"
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in fixture_sources(repository).items():
        imported = from_qasm(source["qasm"])
        (directory / f"{name}.blog").write_text(imported.graph.to_text().rstrip() + "\n")
        (directory / f"{name}.qasm").write_text(source["qasm"])
        info = provenance | dict(
            input_ports=imported.input_ports, output_ports=imported.output_ports,
            t_gadgets=sum(
                m.exact_phase is not None and (Fraction(m.exact_phase) * 4) % 2 == 1
                for slot in imported.schedule.slots for m in slot.measurements
            ),
            layout_options=imported.layout_options,
        )
        info.update({key: source[key] for key in ("algorithm", "source_qasm", "expected_state") if key in source})
        (directory / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
        print(name, imported.graph.block_count, flush=True)


if __name__ == "__main__":
    main()
