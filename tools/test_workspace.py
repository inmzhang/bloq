"""Check root Cargo defaults without compiling or resolving dependencies."""

from pathlib import Path
import re
import tomllib
import unittest

ROOT = Path(__file__).resolve().parent.parent
EDITOR = "bloq_editor"
DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")


def load_toml(path):
    with path.open("rb") as file:
        return tomllib.load(file)


def dependency_names(manifest):
    tables = [manifest.get(table, {}) for table in DEPENDENCY_TABLES]
    for target in manifest.get("target", {}).values():
        tables.extend(target.get(table, {}) for table in DEPENDENCY_TABLES)
    names = set()
    for table in tables:
        for key, spec in table.items():
            package = spec.get("package", key) if isinstance(spec, dict) else key
            names.update({key, package})
    return names


def upload_steps(workflow):
    steps = re.split(r"\n\s+- (?=name:|uses:)", workflow)
    return [step for step in steps if "taiki-e/upload-rust-binary-action" in step]


class WorkspaceDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_toml(ROOT / "Cargo.toml")
        cls.workspace = cls.manifest["workspace"]

    def test_default_members_are_all_members_but_the_editor(self):
        members = set(self.workspace["members"])
        defaults = self.workspace["default-members"]
        self.assertEqual(len(defaults), len(set(defaults)), "duplicate default member")
        self.assertIn(EDITOR, members)
        self.assertEqual(set(defaults), members - {EDITOR})

    def test_default_members_do_not_depend_on_bevy_or_the_editor(self):
        for member in self.workspace["default-members"]:
            with self.subTest(member=member):
                names = dependency_names(load_toml(ROOT / member / "Cargo.toml"))
                self.assertNotIn(EDITOR, names)
                self.assertEqual([name for name in names if name.startswith("bevy")], [])

    def test_dev_and_test_builds_keep_only_line_tables(self):
        profiles = self.manifest["profile"]
        self.assertEqual(profiles["dev"]["debug"], "line-tables-only")
        self.assertNotIn("debug", profiles.get("test", {}))

    def test_binary_uploads_name_a_workspace_package(self):
        # Without a package, the action's `cargo build --bin` sees only default members.
        manifests = {member: load_toml(ROOT / member / "Cargo.toml") for member in self.workspace["members"]}
        packages = {manifest["package"]["name"] for manifest in manifests.values()}
        steps = upload_steps((ROOT / ".github/workflows/binaries.yml").read_text())
        self.assertGreater(len(steps), 0)
        for step in steps:
            bin_name = re.search(r"^\s+bin:\s*(\S+)", step, re.MULTILINE).group(1)
            with self.subTest(bin=bin_name):
                package = re.search(r"^\s+package:\s*(\S+)", step, re.MULTILINE)
                self.assertIsNotNone(package, "upload step has no package")
                self.assertIn(package.group(1), packages)


if __name__ == "__main__":
    unittest.main()
