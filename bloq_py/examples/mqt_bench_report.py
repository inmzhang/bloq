"""Build a fresh Typst catalog from an MQT experiment directory.

Run after mqt_bench_experiments.py. The directory remains self-contained:
report.typ reads its real summary.json, viewers use relative links, and
artifacts.json fingerprints the report and retained evidence. No old reports
are read. Re-run with --verify to check the recorded artifact hashes.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_suffix(".typ")
ARTIFACTS = ("benchmark.qasm", "source.qasm", "clifford_t.qasm", "lowering.json", "result.json",
             "layout.blog", "layout.html", "layout.svg", "program_d25.bloq",
             "program_d3.bloq", "execution.json")


def fingerprint(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


def check_hash(path, expected):
    if fingerprint(path)["sha256"] != expected:
        raise ValueError(f"artifact differs from experiment evidence: {path}")


def verify(root):
    manifest = json.loads((root / "artifacts.json").read_text())
    for relative, expected in manifest["files"].items():
        if fingerprint(inside(root, relative)) != expected:
            raise ValueError(f"artifact changed: {relative}")
    for relative in manifest["links"]:
        if not inside(root, relative).is_file():
            raise ValueError(f"missing report link: {relative}")
    print(f"Verified {len(manifest['files'])} artifact hashes and {len(manifest['links'])} local links.")


def build(root):
    from mqt_bench_experiments import _bounded

    summary = json.loads((root / "summary.json").read_text())
    files = {"summary.json", "manifest.json"}
    manifest = summary["manifest"]
    provenance = [(manifest["bloq_diff_file"], manifest["bloq_diff_sha256"]),
                  (manifest["source_files_file"], manifest["source_files_sha256"])]
    provenance += [(upstream["diff_file"], upstream["diff_sha256"])
                   for upstream in manifest["upstream"].values()]
    for name, digest in provenance:
        check_hash(inside(root, name), digest)
        files.add(name)
    assets = dict(runs={}, cases={}, thumbnails={}, provenance=sorted(files))
    for row in summary["runs"]:
        case = row["case"]
        case_dir = inside(root, f"cases/{case}")
        sources = {name: (case_dir / name).relative_to(root).as_posix()
                   for name in ARTIFACTS if (case_dir / name).is_file()}
        assets["cases"][case] = sources
        files.update(sources.values())
        for name, field in (("source.qasm", "source_sha256"), ("clifford_t.qasm", "clifford_t_sha256")):
            if field in row:
                check_hash(root / sources[name], row[field])
        if "benchmark_sha256" in row:
            name = "benchmark.qasm" if row.get("source_variant") == "explicit_input_preparation" else "source.qasm"
            check_hash(root / sources[name], row["benchmark_sha256"])
        if not row.get("run_dir"):
            continue
        run_dir = inside(root, row["run_dir"])
        for name, digest in (("layout.blog", row.get("layout", {}).get("blog_sha256")),
                             ("program_d25.bloq", row.get("ir_d25", {}).get("sha256")),
                             ("program_d3.bloq", row.get("vm_d3", {}).get("sha256"))):
            if digest is not None:
                check_hash(run_dir / name, digest)
        compiled = row.get("ir_d25", {}).get("status") == "compiled"
        if compiled:
            ir = row["ir_d25"]
            if not (ir.get("validated") and ir.get("round_trip")
                    and ir.get("distance") == 25 and ir.get("t_preparation") == "ideal_t_proxy"
                    and ir.get("sha256") and row.get("layout", {}).get("blog_sha256")):
                raise ValueError(f"{case}/{row['compiler']}: incomplete distance-25 audit evidence")
            for name in ("source.qasm", "clifford_t.qasm", "layout.blog", "layout.html",
                         "program_d25.bloq", "result.json"):
                if not (run_dir / name).is_file():
                    raise ValueError(f"missing successful-run artifact: {run_dir / name}")
            # Each native mesh and its SVG triangles are released before the
            # next layout; rendering failures never change compilation evidence.
            rendered = _bounded([sys.executable, str(Path(__file__).resolve()),
                                 "--thumbnail", str(run_dir / "layout.blog")], run_dir,
                                "thumbnail", 120, memory_gib=min(8, manifest.get("memory_gib", 8)))
            assets["thumbnails"][row["run_dir"]] = rendered
            if rendered["status"] != "completed":
                (run_dir / "layout.svg").unlink(missing_ok=True)
        artifacts = {name: (run_dir / name).relative_to(root).as_posix()
                     for name in ARTIFACTS if (run_dir / name).is_file()}
        assets["runs"][row["run_dir"]] = artifacts
        files.update(artifacts.values())
    save(root / "report-assets.json", assets)
    shutil.copyfile(TEMPLATE, root / "report.typ")
    subprocess.run(["typst", "compile", "report.typ", "report.pdf"], cwd=root, check=True)
    files.update(("report.typ", "report.pdf", "report-assets.json"))
    # Logs and checker reports are retained even for rejected and timed-out runs.
    files.update(path.relative_to(root).as_posix() for path in (root / "cases").rglob("*")
                 if path.is_file())
    links = sorted(files | {"artifacts.json"})
    save(root / "artifacts.json", dict(
        builder_sha256=fingerprint(Path(__file__))["sha256"],
        files={name: fingerprint(inside(root, name)) for name in sorted(files)}, links=links))
    verify(root)
    print(root / "report.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, nargs="?", default=ROOT / "target/mqt-bench")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--thumbnail", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.thumbnail:
        sys.path.insert(0, str(ROOT / "tools"))
        import bloq
        from render_block_thumbnails import thumbnail

        thumbnail(bloq.BlockGraph.load(args.thumbnail), args.thumbnail.with_suffix(".svg"),
                  args.thumbnail.parent.name, opaque=True)
    elif args.verify:
        verify(args.output_dir.resolve())
    else:
        build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
