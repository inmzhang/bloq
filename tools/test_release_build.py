"""Check release path mappings without compiling dependencies."""

from pathlib import Path
import shlex
import unittest

from release_build import build_environment


class ReleaseBuildTests(unittest.TestCase):
    def test_preserves_flags_and_maps_paths_with_spaces(self):
        env = build_environment(
            {"RUSTFLAGS": "-C debuginfo=1", "CXXFLAGS": "-O2", "CARGO_BUILD_TARGET": "x86_64-unknown-linux-gnu"},
            Path("/home/builder/my checkout"), Path("/home/builder"),
        )
        rust = env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        self.assertEqual(rust[:2], ["-C", "debuginfo=1"])
        self.assertIn("--remap-path-prefix=/home/builder/my checkout=/bloq", rust)
        self.assertIn("-ffile-prefix-map=/home/builder/my checkout=/bloq", shlex.split(env["CXXFLAGS"]))
        self.assertEqual(shlex.split(env["CXXFLAGS"])[0], "-O2")
        encoded = build_environment(
            {"CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=2", "RUSTFLAGS": "ignored"},
            Path("/src"), Path("/home/builder"),
        )["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        self.assertEqual(encoded[:2], ["-C", "opt-level=2"])
        self.assertNotIn("ignored", encoded)

    def test_msvc_path_mappings(self):
        env = build_environment(
            {"CARGO_BUILD_TARGET": "x86_64-pc-windows-msvc"},
            Path("/checkout"), Path("/builder"),
        )
        self.assertIn("/pathmap:/checkout=/bloq", shlex.split(env["CXXFLAGS"]))
        self.assertIn("/experimental:deterministic", shlex.split(env["CXXFLAGS"]))


if __name__ == "__main__":
    unittest.main()
