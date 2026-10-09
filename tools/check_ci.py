"""Require every selected CI job, including PR approval, to succeed."""

import json
import os


def failures(jobs, event):
    selected = jobs["quick"]["outputs"]
    checks = json.loads(selected.get("checks") or "[]")
    required = {
        "quick": True,
        "approve": event == "pull_request" and selected.get("heavy") == "true",
        "checks": bool(checks),
        "windows": "ci-rust" in checks,
        "site": selected.get("site") == "true",
        "packages": selected.get("packages") == "true",
    }
    errors = []
    for name, needed in required.items():
        result = jobs[name]["result"]
        expected = "success" if needed else "skipped"
        if result != expected:
            errors.append(f"{name}: expected {expected}, got {result}")
    return errors


if __name__ == "__main__":
    errors = failures(json.loads(os.environ["CI_RESULTS"]), os.environ["CI_EVENT"])
    if errors:
        raise SystemExit("\n".join(errors))
    print("All selected CI checks passed.")
