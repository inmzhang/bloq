"""Separate experiment ledgers remain immutable and correctly linked in one report."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def reporter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    return import_module("mqt_bench_report")


def make_experiment(reporter, root, source):
    case = root / "cases/adder_32"
    run = case / "spiderls"
    run.mkdir(parents=True)
    (case / "source.qasm").write_text(source)
    (case / "clifford_t.qasm").write_text(source)
    reporter.save(run / "result.json", dict(status="failed"))
    for name in ("bloq.diff", "source-files.json", "spiderls.diff"):
        (root / name).write_text("{}")
    manifest = dict(
        bloq_diff_file="bloq.diff", bloq_diff_sha256=reporter.fingerprint(root / "bloq.diff")["sha256"],
        source_files_file="source-files.json",
        source_files_sha256=reporter.fingerprint(root / "source-files.json")["sha256"],
        upstream=dict(spiderls=dict(diff_file="spiderls.diff",
                                   diff_sha256=reporter.fingerprint(root / "spiderls.diff")["sha256"])))
    row = dict(case="adder_32", compiler="spiderls", run_dir="cases/adder_32/spiderls", status="failed",
               source_sha256=reporter.fingerprint(case / "source.qasm")["sha256"],
               clifford_t_sha256=reporter.fingerprint(case / "clifford_t.qasm")["sha256"])
    reporter.save(root / "manifest.json", manifest)
    reporter.save(root / "summary.json", dict(manifest=manifest, runs=[row]))
    return root


def test_include_preserves_ledgers_and_disambiguates_same_case(reporter, tmp_path):
    root = make_experiment(reporter, tmp_path / "default", "default circuit")
    other = make_experiment(reporter, tmp_path / "wider-routing", "additional circuit")
    originals = [(path, path.read_bytes()) for folder in (root, other)
                 for path in folder.rglob("*") if path.is_file()]
    report, files = reporter.collect(root, [other])
    default, included = report["runs"]
    assert default["case"] != included["case"]
    assert default["source_case"] == included["source_case"] == "adder_32"
    assert (root / default["case_dir"] / "source.qasm").read_text() == "default circuit"
    assert (root / included["case_dir"] / "source.qasm").read_text() == "additional circuit"
    assert (root / included["run_dir"] / "result.json").is_file()
    assert report["experiments"][1]["summary_file"] in files
    assert all(path.read_bytes() == expected for path, expected in originals)
    assert reporter.collect(root, [other])[0] == report


@pytest.mark.parametrize("artifact", ["source.qasm", "spiderls.diff"])
def test_include_rechecks_original_evidence_even_when_copy_exists(reporter, tmp_path, artifact):
    root = make_experiment(reporter, tmp_path / "default", "default circuit")
    other = make_experiment(reporter, tmp_path / "wider-routing", "additional circuit")
    reporter.collect(root, [other])
    path = other / ("cases/adder_32/source.qasm" if artifact == "source.qasm" else artifact)
    path.write_text("changed after experiment")
    with pytest.raises(ValueError, match="differs from experiment evidence"):
        reporter.collect(root, [other])


def test_include_rejects_recursive_copy_and_escaping_artifact_path(reporter, tmp_path):
    root = make_experiment(reporter, tmp_path / "default", "source")
    with pytest.raises(ValueError, match="outside the report directory"):
        reporter.collect(root, [root])
    summary, _ = reporter.experiment(root)
    summary["runs"][0]["run_dir"] = "../outside"
    reporter.save(root / "summary.json", summary)
    with pytest.raises(ValueError):
        reporter.collect(root, [])
