"""Focused release-message format checks; no repository-history exceptions."""

import unittest

from changelog import validate


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
        ):
            with self.subTest(message=message), self.assertRaises(ValueError):
                validate(message)


if __name__ == "__main__":
    unittest.main()
