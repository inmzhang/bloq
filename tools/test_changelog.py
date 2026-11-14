"""Focused release-message format checks; no repository-history exceptions."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from changelog import release_range, validate


class ChangelogTests(unittest.TestCase):
    def test_valid_messages(self):
        for message in (
            "fix(editor): preserve selection\n\n- Keep normal prose bullets.\n\nAssisted-by: Codex (model)",
            "feat(rust)!: change API\n\nBREAKING CHANGE: use the new API",
            "chore: release v0.2.0",
        ):
            validate(message)

    def test_rejects_ambiguous_metadata(self):
        for message in (
            "Update editor",
            "fix: result\n\n* feat(rust)!: copied change",
            "fix: result\n\n- fix(editor): copied message",
            "feat!: result\n\nBREAKING CHANGE: first\n\nBREAKING-CHANGE: second",
            "feat(rust)!: result without migration guidance",
            "feat(rust)!: result\n\nBREAKING CHANGE: ",
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                validate(message)

    def test_release_boundary_excludes_old_history_not_bad_new_messages(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                def git(*args):
                    return subprocess.check_output(["git", *args], text=True).strip()
                git("init", "-q")
                git("config", "user.name", "Fixture")
                git("config", "user.email", "fixture@example.invalid")
                git("commit", "--allow-empty", "-qm", "old metadata")
                Path("tools").mkdir()
                Path("tools/changelog.py").touch()
                git("add", ".")
                git("commit", "-qm", "ci: adopt metadata check")
                adoption = git("rev-parse", "HEAD")
                git("commit", "--allow-empty", "-qm", "bad metadata")
                git("commit", "--allow-empty", "-qm", "fix: later clean tip")
                self.assertEqual(release_range(), adoption + "^..HEAD")
                messages = git("log", "--format=%s", release_range()).splitlines()
                self.assertIn("bad metadata", messages)
                self.assertNotIn("old metadata", messages)
                git("tag", "v0.2.0")
                self.assertEqual(release_range(), git("rev-parse", "HEAD") + "..HEAD")
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
