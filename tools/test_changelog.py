"""Release metadata regressions, including the pinned changelog renderer."""

import os
import subprocess
import unittest
from pathlib import Path

from changelog import CORRECTIONS, REVIEWED_HISTORY, corrected_message, git, validate


class ChangelogTests(unittest.TestCase):
    def test_unknown_messages_are_unchanged(self):
        message = "feat(rust)!: change API\n\nBREAKING CHANGE: use the new API\n"
        self.assertEqual(corrected_message("unknown", message), message)
        validate(message)

    def test_rejects_ambiguous_squashes(self):
        for message in (
            "fix: result\n\n* feat(rust)!: old change\n\nBREAKING CHANGE: inherited",
            "fix: result\n\n- fix(editor): copied message",
            "feat!: result\n\nBREAKING CHANGE: first\n\nBREAKING-CHANGE: second",
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                validate(message)
        validate("fix: result\n\n- Python: Preserve normal prose bullets.\n\nAssisted-by: Tool (model)")

    def test_known_history_repairs(self):
        messages = {sha: corrected_message(sha, git("show", "-s", "--format=%B", sha))
                    for sha in CORRECTIONS}
        for message in messages.values():
            validate(message)
        liveness, api, merge = messages.values()
        self.assertIn("LiveQubitReuse", liveness)
        self.assertNotIn("IdealPortAdapter", liveness)
        self.assertIn("analyze_actions", api)
        self.assertIn("compile_clifford_distance", api)
        self.assertIn("Python:", api)
        self.assertNotIn("BREAKING", merge)

    def test_rendered_notes(self):
        # Same pinned renderer as `just changelog` and release-plz's postprocessor.
        notes = subprocess.check_output([
            "uvx", "--from", "git-cliff==2.14.2", "git-cliff", "--config", "cliff.toml",
            "--unreleased", f"v0.1.1..{REVIEWED_HISTORY}",
        ], text=True)
        entries = {line.split("/pull/")[1].split(")")[0]: line
                   for line in notes.splitlines() if "/pull/" in line}
        self.assertIn("**BREAKING:**", entries["28"])
        self.assertIn("**BREAKING:**", entries["31"])
        self.assertNotIn("**BREAKING:**", entries["32"])
        for migration in ("LiveQubitReuse", "compile_clifford_distance", "ZXGraph.stabilizers"):
            self.assertIn(migration, notes)
        self.assertEqual(notes.count("InstanceProvenance includes IdealPortAdapter"), 1)
        self.assertEqual(notes.count("BlockGraph::analyze_actions and analyze_actions_with_limits"), 1)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    unittest.main()
