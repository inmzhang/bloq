"""Allow a development-bump PR only after every publication job succeeds."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from workspace_version import STABLE_VERSION, current_version


PUBLICATION_JOBS = {
    "release-plz.yml": {"Publish Rust crates"},
    "pypi.yml": {"Publish to PyPI"},
    "binaries.yml": {
        "x86_64-unknown-linux-gnu",
        "aarch64-apple-darwin",
        "x86_64-pc-windows-msvc",
    },
}


def jobs_succeeded(jobs, required):
    results = {job["name"]: job["conclusion"] for job in jobs}
    return all(results.get(name) == "success" for name in required)


def github_items(path, key):
    pages = json.loads(subprocess.check_output(
        ["gh", "api", "--paginate", "--slurp", path],
        text=True,
    ))
    return [item for page in pages for item in page[key]]


def ready_version(repository):
    version = current_version()
    if not re.fullmatch(STABLE_VERSION, version):
        print("main already has a development version; no bump needed.")
        return None
    tag = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/v{version}^{{commit}}"],
                         text=True, capture_output=True)
    if tag.returncode:
        print(f"Waiting for release tag v{version}.")
        return None
    sha = tag.stdout.strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"], check=True)
    for workflow, required in PUBLICATION_JOBS.items():
        runs = github_items(
            f"repos/{repository}/actions/workflows/{workflow}/runs?head_sha={sha}&status=completed&per_page=100",
            "workflow_runs",
        )
        for run in runs:
            if run["head_sha"] != sha:
                continue
            jobs = github_items(
                f"repos/{repository}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100", "jobs",
            )
            if jobs_succeeded(jobs, required):
                break
        else:
            print(f"Waiting for successful publication jobs in {workflow} for v{version}.")
            return None
    return version


def check():
    required = PUBLICATION_JOBS["binaries.yml"]
    jobs = [{"name": name, "conclusion": "success"} for name in required]
    assert jobs_succeeded(jobs, required)
    assert not jobs_succeeded(jobs[:-1], required)
    for conclusion in ("skipped", "failure", "cancelled", None):
        assert not jobs_succeeded([*jobs[:-1], dict(jobs[-1], conclusion=conclusion)], required)
    assert not jobs_succeeded([{"name": "Build wheels", "conclusion": "success"}], {"Publish to PyPI"})
    print("Publication gate checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check publication gating without GitHub access")
    args = parser.parse_args()
    if args.check:
        check()
        return
    version = ready_version(os.environ["GITHUB_REPOSITORY"])
    if version:
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(f"version={version}\n")
        print(f"v{version} is fully published; prepare the development bump.")


if __name__ == "__main__":
    main()
