"""Focused checks for snapshot retention, aliases, and editor subpaths."""

from pathlib import Path
import tempfile
import unittest

import importlib.util

# Do not shadow Python's standard startup module named `site`.
spec = importlib.util.spec_from_file_location("bloq_site", Path(__file__).with_name("site.py"))
site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(site)


class SiteAssemblyTest(unittest.TestCase):
    def test_retention_immutable_sources_aliases_and_relative_editor_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            for name, source in (("dev", "development"), ("v0.1.0", "release-commit")):
                root = output / "docs" / name
                pages = ["index.html", "api/python.html", "api/rust/bloq/index.html", "guide.html"]
                for page in pages:
                    path = root / page
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("retained content")
                (root / "searchindex.js").write_text("search index")
                site.write_json(root / "build.json", {"name": name, "source": source, "label": name, "path": f"docs/{name}/", "package_version": "0.1.0", "pages": pages})
            release = output / "docs" / "v0.1.0" / "guide.html"
            before = release.read_bytes()
            self.assertTrue(site.release_exists(output, "v0.1.0", "release-commit"))
            with self.assertRaisesRegex(ValueError, "different source"):
                site.release_exists(output, "v0.1.0", "moved-tag")
            with self.assertRaises(ValueError):
                site.tag_commit("../outside")
            manifest = site.version_manifest(output)
            self.assertIsNone(manifest["stable"])
            self.assertEqual(manifest["default"], "dev")
            with self.assertRaisesRegex(ValueError, "release-tag"):
                site.stable_alias(output, "dev")
            site.stable_alias(output, "v0.1.0")
            manifest = site.version_manifest(output)
            self.assertEqual(manifest["stable"], "v0.1.0")
            self.assertEqual(manifest["default"], "dev")
            self.assertEqual(release.read_bytes(), before)
            self.assertEqual((output / "docs" / "stable" / "guide.html").read_bytes(), before)
            site.site_entrypoints(output, manifest)
            self.assertEqual(site.HtmlReferences((output / "index.html").read_text()).redirect, "docs/dev/")
            self.assertEqual(site.HtmlReferences((output / "docs" / "index.html").read_text()).redirect, "dev/")
            for page in ("index.html", "api.html"):
                self.assertEqual(site.HtmlReferences((output / "pydoc" / page).read_text()).redirect, "../docs/dev/api/python.html")
            self.assertIn("<title>bloq</title>", (output / "index.html").read_text())
            rustdoc = Path(temporary) / "rustdoc"
            caller = rustdoc / "bloq_compile" / "index.html"
            target = rustdoc / "bloq_graph" / "struct.BlockGraph.html"
            caller.parent.mkdir(parents=True)
            target.parent.mkdir()
            caller.write_text('<a href="bloq_graph::BlockGraph">BlockGraph</a>')
            target.write_text('<h1>BlockGraph</h1>')
            site.fix_rustdoc_links(site.ROOT, rustdoc)
            site.check_links_and_privacy(rustdoc)
            (output / "index.html").write_text('<a href="target.html#example">Example</a>')
            (output / "target.html").write_text('<h1 id="example">Example</h1>')
            site.check_links_and_privacy(output)
            (output / "index.html").write_text('<a href="target.html#impl-%3CT%3E">Type</a>')
            (output / "target.html").write_text('<h1 id="impl-%3CT%3E">Type</h1>')
            site.check_links_and_privacy(output)
            (output / "index.html").write_text('<a href="target.html#example">Example</a>')
            (output / "target.html").write_text('<h1 id="other">Example</h1>')
            with self.assertRaisesRegex(ValueError, "missing anchor"):
                site.check_links_and_privacy(output)
            for project in ("bloq-paper", "bloq-experiments"):
                (output / "index.html").write_text(
                    f'<a href="https://github.com/inmzhang/{project}">Source</a>'
                )
                with self.assertRaisesRegex(ValueError, "private source"):
                    site.check_links_and_privacy(output)
            (output / "index.html").write_text('<p>../bloq-paper/src/main.tex</p>')
            with self.assertRaisesRegex(ValueError, "private source"):
                site.check_links_and_privacy(output)
            (output / "index.html").write_text('<a href="target.html#example">Example</a>')
            (output / "target.html").write_text('<h1 id="example">Example</h1><p>/Users/private/build</p>')
            with self.assertRaisesRegex(ValueError, "build-host"):
                site.check_links_and_privacy(output)
            (output / "target.html").write_text('<h1 id="example">Example</h1>')
            for filename, content in (
                ("metadata.json", '{"creator": "bloq-experiments"}'),
                ("figure.svg", '<svg><metadata>bloq-paper</metadata></svg>'),
                ("bundle.wasm", "/home/build-user/workspace/source.rs"),
            ):
                asset = output / filename
                asset.write_text(content)
                with self.assertRaisesRegex(ValueError, "private source or build-host"):
                    site.check_links_and_privacy(output)
                asset.unlink()
            cache = output / "docs" / "dev" / ".doctrees"
            cache.mkdir()
            (cache / "environment.pickle").write_bytes(b"build cache")
            with self.assertRaisesRegex(ValueError, "build cache must not be published"):
                site.check_links_and_privacy(output)
            (cache / "environment.pickle").unlink()
            cache.rmdir()
            bundle = Path(temporary) / "bundle"
            for filename, content in {"index.html": "bundle wrapper", "compile_worker.js": 'import init from "./build/bloq_editor.js";', "build/bloq_editor.js": "export default function init() {}", "build/bloq_editor_bg.wasm": "wasm placeholder"}.items():
                path = bundle / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            site.copy_editor(bundle, output, "editor-commit", "0.1.0")
            site.check(output, links=False)
            entry = manifest["versions"][0]
            entry["pages"].append("../../versions.json")
            site.write_json(output / "versions.json", manifest)
            with self.assertRaisesRegex(ValueError, "invalid page"):
                site.check(output, links=False)


if __name__ == "__main__":
    unittest.main()
