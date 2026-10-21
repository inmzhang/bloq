"""Build a fresh Typst catalog from an MQT experiment directory.

Run after mqt_bench_experiments.py. The directory remains self-contained:
report.typ reads a derived report-summary.json, viewers use relative links, and
artifacts.json fingerprints the report and retained evidence. No old reports
are read. --include copies separately recorded experiments into the catalog
without modifying their ledgers. Re-run with --verify to check artifact hashes.
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
             "layout.blog", "layout.html", "layout.svg", "layout.png", "program_d25.bloq",
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


def experiment(root):
    """Check captured evidence before copying or presenting an experiment."""
    summary = json.loads((root / "summary.json").read_text())
    manifest = summary["manifest"]
    if json.loads((root / "manifest.json").read_text()) != manifest:
        raise ValueError(f"manifest differs from experiment ledger: {root}")
    files = {"summary.json", "manifest.json"}
    provenance = [(manifest["bloq_diff_file"], manifest["bloq_diff_sha256"]),
                  (manifest["source_files_file"], manifest["source_files_sha256"])]
    provenance += [(upstream["diff_file"], upstream["diff_sha256"])
                   for upstream in manifest["upstream"].values()]
    for name, digest in provenance:
        check_hash(inside(root, name), digest)
        files.add(name)
    for row in summary["runs"]:
        case_dir = inside(root, f"cases/{row['case']}")
        for name, field in (("source.qasm", "source_sha256"), ("clifford_t.qasm", "clifford_t_sha256")):
            if field in row:
                check_hash(case_dir / name, row[field])
        if "benchmark_sha256" in row:
            name = "benchmark.qasm" if row.get("source_variant") == "explicit_input_preparation" else "source.qasm"
            check_hash(case_dir / name, row["benchmark_sha256"])
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
                raise ValueError(f"{row['case']}/{row['compiler']}: incomplete distance-25 audit evidence")
            for name in ("source.qasm", "clifford_t.qasm", "layout.blog",
                         "program_d25.bloq", "result.json"):
                if not (run_dir / name).is_file():
                    raise ValueError(f"missing successful-run artifact: {run_dir / name}")
    return summary, files


def collect(root, includes):
    """Keep original ledgers and give each report row its own artifact namespace."""
    summary, provenance = experiment(root)
    sessions = [("default", "Default run", root, summary, provenance)]
    for index, source in enumerate(includes, 1):
        source = source.resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("included experiment must be outside the report directory")
        original, _ = experiment(source)
        if (source / "artifacts.json").exists():
            verify(source)
        name = f"{index:02d}-{source.name}"
        destination = root / "additional" / name
        if destination.exists():
            if (destination / "summary.json").read_bytes() != (source / "summary.json").read_bytes():
                raise ValueError(f"included ledger changed; use a fresh report directory: {source}")
        else:
            shutil.copytree(source, destination)
        copied, files = experiment(destination)
        if copied != original:
            raise ValueError(f"included ledger changed during copying: {source}")
        sessions.append((name, source.name, destination, copied, files))
    report, files = dict(experiments=[], runs=[]), set()
    for identifier, label, directory, original, provenance in sessions:
        prefix = directory.relative_to(root)
        links = [(prefix / name).as_posix() for name in sorted(provenance)]
        report["experiments"].append(dict(id=identifier, label=label, manifest=original["manifest"],
                                          summary_file=(prefix / "summary.json").as_posix(), provenance=links))
        files.update(links)
        files.update(path.relative_to(root).as_posix() for path in (directory / "cases").rglob("*")
                     if path.is_file())
        for original_row in original["runs"]:
            row = dict(original_row, source_case=original_row["case"], experiment=identifier,
                       experiment_label=label, case=f"{identifier}/{original_row['case']}",
                       case_dir=(prefix / "cases" / original_row["case"]).as_posix())
            if row.get("run_dir"):
                row["run_dir"] = (prefix / row["run_dir"]).as_posix()
            report["runs"].append(row)
    return report, files


def build(root, includes=()):
    from mqt_bench_experiments import _bounded

    summary, files = collect(root, includes)
    save(root / "report-summary.json", summary)
    assets = dict(runs={}, cases={}, thumbnails={}, viewers={}, provenance=sorted(
        path for session in summary["experiments"] for path in session["provenance"]))
    limits = {session["id"]: session["manifest"].get("memory_gib", 8) for session in summary["experiments"]}
    for row in summary["runs"]:
        case_dir = inside(root, row["case_dir"])
        assets["cases"][row["case"]] = {name: (case_dir / name).relative_to(root).as_posix()
                                         for name in ARTIFACTS if (case_dir / name).is_file()}
        if not row.get("run_dir"):
            continue
        run_dir = inside(root, row["run_dir"])
        image_dir = inside(root, f"report-images/{row['run_dir']}")
        if row.get("ir_d25", {}).get("status") == "compiled":
            image_dir.mkdir(parents=True, exist_ok=True)
            if not (run_dir / "layout.html").is_file():
                viewer = _bounded([sys.executable, str(Path(__file__).resolve()),
                                   "--viewer", str(run_dir / "layout.blog"), str(image_dir / "layout.html")],
                                  image_dir, "viewer", 120, memory_gib=min(8, limits[row["experiment"]]))
                assets["viewers"][row["run_dir"]] = viewer
                if viewer["status"] != "completed":
                    (image_dir / "layout.html").unlink(missing_ok=True)
            # Each native mesh and its SVG triangles are released before the
            # next layout; rendering failures never change compilation evidence.
            rendered = _bounded([sys.executable, str(Path(__file__).resolve()),
                                 "--thumbnail", str(run_dir / "layout.blog"), str(image_dir / "layout.svg")],
                                image_dir, "thumbnail", 120, memory_gib=min(8, limits[row["experiment"]]))
            assets["thumbnails"][row["run_dir"]] = rendered
            if rendered["status"] != "completed":
                for name in ("layout.svg", "layout.png"):
                    (image_dir / name).unlink(missing_ok=True)
        artifacts = {name: (run_dir / name).relative_to(root).as_posix()
                     for name in ARTIFACTS if name not in ("layout.svg", "layout.png") and (run_dir / name).is_file()}
        for name in ("layout.svg", "layout.png", "layout.html"):
            if (image_dir / name).is_file():
                artifacts[name] = (image_dir / name).relative_to(root).as_posix()
        assets["runs"][row["run_dir"]] = artifacts
        files.update(artifacts.values())
    save(root / "report-assets.json", assets)
    shutil.copyfile(TEMPLATE, root / "report.typ")
    subprocess.run(["typst", "compile", "report.typ", "report.pdf"], cwd=root, check=True)
    files.update(("report.typ", "report.pdf", "report-assets.json", "report-summary.json"))
    files.update(path.relative_to(root).as_posix() for path in (root / "report-images").rglob("*")
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
    parser.add_argument("--include", type=Path, nargs="+", default=[], help="additional recorded experiment directories")
    parser.add_argument("--verify", action="store_true")
    render = parser.add_mutually_exclusive_group()
    render.add_argument("--thumbnail", type=Path, nargs=2, metavar=("BLOG", "SVG"), help=argparse.SUPPRESS)
    render.add_argument("--viewer", type=Path, nargs=2, metavar=("BLOG", "HTML"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.thumbnail or args.viewer:
        sys.path.insert(0, str(ROOT / "tools"))
        import bloq

        source, output = args.thumbnail or args.viewer
        graph = bloq.BlockGraph.load(source)
        if args.viewer:
            graph.export_html_viewer(output)
            return
        from render_block_thumbnails import thumbnail

        raster = output.with_suffix(".png")
        raster.unlink(missing_ok=True)
        thumbnail(graph, output, source.parent.name, opaque=True)
        # Large meshes exceed Typst's SVG node limit; retain a print-size preview.
        if output.stat().st_size > 8 * 1024**2:
            subprocess.run(["magick", "-background", "white", "-density", "1800", f"MSVG:{output}",
                            "-resize", "2400x", str(raster)], check=True)
            output.unlink()
    elif args.verify:
        verify(args.output_dir.resolve())
    else:
        build(args.output_dir.resolve(), args.include)


if __name__ == "__main__":
    main()
