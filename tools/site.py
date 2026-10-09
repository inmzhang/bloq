#!/usr/bin/env python3
"""Assemble the homepage, versioned Sphinx/Rust API docs, and browser editor."""

import argparse
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def run(args, *, root=ROOT, env=None, capture=False):
    return subprocess.run(
        list(map(str, args)), cwd=root, env=env, check=True,
        text=True, stdout=subprocess.PIPE if capture else None,
    ).stdout


def tag_commit(tag, root=ROOT):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag) or tag in {"dev", "stable"}:
        raise ValueError("Release tags must be safe URL names other than dev or stable")
    return run(["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], root=root, capture=True).strip()


def package_version(root):
    metadata = json.loads(run(["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"], root=root, capture=True))
    return next(package["version"] for package in metadata["packages"] if package["name"] == "bloq")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def release_exists(output, tag, commit):
    path = output / "docs" / tag
    if not path.exists():
        return False
    metadata = path / "build.json"
    if not metadata.is_file() or json.loads(metadata.read_text())["source"] != commit:
        raise ValueError(f"Immutable snapshot {tag} already exists with a different source")
    return True


def build_snapshot(root, destination, name, source, *, docs_only=False, isolated=False):
    if not (root / "docs" / "conf.py").is_file():
        raise ValueError(f"{name} has no website documentation sources; do not mix it with another release")
    version = package_version(root)
    env = os.environ.copy()
    env["BLOQ_DOCS_VERSION"] = f"Development ({version})" if name == "dev" else name
    env["BLOQ_DOCS_SOURCE"] = source
    env["BLOQ_DOCS_REF"] = source.split("+", 1)[0]
    # Tags never share compiled dependencies or installed bindings with dev.
    if isolated:
        env["CARGO_TARGET_DIR"] = str(root / "target")
    run(["uv", "sync", "--project", root / "bloq_py", "--group", "docs", "--locked"], root=root, env=env)
    verify_binding = (
        "import bloq, bloq._core; from pathlib import Path; "
        f"assert bloq.__version__ == {version!r}; "
        f"root = Path({str(root / 'bloq_py' / 'python')!r}).resolve(); "
        "assert Path(bloq.__file__).resolve().is_relative_to(root); "
        "assert Path(bloq._core.__file__).resolve().is_relative_to(root)"
    )
    run(["uv", "run", "--project", root / "bloq_py", "--no-sync", "python", "-c", verify_binding], root=root, env=env)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".build-{name}-", dir=destination.parent) as temporary:
        staged = Path(temporary) / name
        run(["uv", "run", "--project", root / "bloq_py", "--no-sync", "sphinx-build", "-E", "-W", "--keep-going", "-b", "html", "-d", Path(temporary) / "doctrees", root / "docs", staged], root=root, env=env)
        if not docs_only:
            rust_env = env.copy()
            target = root / "target" if isolated else ROOT / "target" / "site-rust"
            rust_env["CARGO_TARGET_DIR"] = str(target)
            # Removed API items must not survive in a cached rustdoc tree.
            if (target / "doc").exists():
                shutil.rmtree(target / "doc")
            run(["cargo", "doc", "--workspace", "--exclude", "bloq_editor", "--exclude", "bloq_test", "--exclude", "xtask", "--exclude", "bloq_py", "--all-features", "--no-deps", "--locked"], root=root, env=rust_env)
            shutil.copytree(target / "doc", staged / "api" / "rust")
            fix_rustdoc_links(root, staged / "api" / "rust")
            redirect(staged / "api" / "rust" / "index.html", "bloq/index.html", "Rust API")
        entry = {
            "name": name,
            "label": env["BLOQ_DOCS_VERSION"],
            "path": f"docs/{name}/",
            "source": source,
            "package_version": version,
            "pages": sorted(path.relative_to(staged).as_posix() for path in staged.rglob("*.html")),
        }
        write_json(staged / "build.json", entry)
        if destination.exists():
            if name != "dev":
                raise ValueError(f"Refusing to replace immutable snapshot {name}")
            shutil.rmtree(destination)
        staged.rename(destination)
    return entry


def build_release(tag, output, *, root=ROOT):
    commit = tag_commit(tag, root)
    if release_exists(output, tag, commit):
        return
    with tempfile.TemporaryDirectory(prefix="bloq-release-") as temporary:
        checkout = Path(temporary) / "source"
        checkout.mkdir()
        archive = Path(temporary) / "source.tar"
        run(["git", "archive", "--format=tar", "-o", archive, commit], root=root)
        with tarfile.open(archive) as source_archive:
            source_archive.extractall(checkout, filter="data")
        build_snapshot(checkout, output / "docs" / tag, tag, commit, isolated=True)


def stable_alias(output, tag):
    source = output / "docs" / tag
    entry = json.loads((source / "build.json").read_text())
    if entry["name"] in {"dev", "stable"}:
        raise ValueError("Stable must identify a release-tag snapshot")
    destination = output / "docs" / "stable"
    entry.update(name="stable", label=f"Stable ({tag})", path="docs/stable/", tag=tag)
    with tempfile.TemporaryDirectory(prefix=".stable-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "stable"
        shutil.copytree(source, staged)
        write_json(staged / "build.json", entry)
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)


def version_manifest(output, editor=None):
    entries = [json.loads(path.read_text()) for path in sorted((output / "docs").glob("*/build.json"))]
    entries.sort(key=lambda entry: (entry["name"] != "stable", entry["name"] != "dev", entry["name"]))
    previous = output / "versions.json"
    previous_editor = json.loads(previous.read_text()).get("editor") if previous.exists() else None
    manifest = {
        "default": "dev",
        "stable": next((entry["tag"] for entry in entries if entry["name"] == "stable"), None),
        "versions": entries,
        "editor": editor or previous_editor,
    }
    write_json(previous, manifest)
    return manifest


def copy_editor(bundle, output, source, version):
    required = ("index.html", "compile_worker.js", "build/bloq_editor.js", "build/bloq_editor_bg.wasm")
    for filename in required:
        if not (bundle / filename).is_file():
            raise ValueError(f"Editor bundle is missing {filename}")
    wasm = (bundle / "build" / "bloq_editor_bg.wasm").read_bytes()
    if str(Path.home()).encode() in wasm or str(ROOT).encode() in wasm:
        raise ValueError("Editor bundle contains build host paths; rebuild with --remap-path-prefix")
    destination = output / "editor"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(bundle, destination)
    # Always use the maintained wrapper, including its same-domain help links.
    wrapper = (ROOT / "bloq_editor" / "web" / "index.html").read_text()
    (destination / "index.html").write_text(wrapper)
    entry = {"package_version": version, "source": source, "path": "editor/"}
    write_json(destination / "version.json", entry)
    return entry


def site_entrypoints(output, manifest):
    redirect(output / "index.html", f"docs/{manifest['default']}/", "bloq")
    (output / "CNAME").write_text("bloqec.com\n")
    (output / ".nojekyll").touch()
    redirect(output / "docs" / "index.html", f"{manifest['default']}/", "bloq")
    # Retain old Python API URLs and their anchors without duplicating docs.
    old_api = output / "pydoc"
    old_api.mkdir(exist_ok=True)
    target = f"../docs/{manifest['default']}/api/python.html"
    redirect(old_api / "index.html", target)
    redirect(old_api / "api.html", target)
    api_items = output / "docs" / manifest["default"] / "api" / "_autosummary"
    for page in api_items.glob("*.html"):
        redirect(old_api / "_autosummary" / page.name, f"../../docs/{manifest['default']}/api/_autosummary/{page.name}")


def redirect(path, target, title="Python API"):
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = html.escape(target, quote=True)
    title = html.escape(title)
    path.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={escaped}"><title>{title}</title><script>location.replace({json.dumps(target)} + location.hash)</script><a href="{escaped}">{title}</a>\n')


def fix_rustdoc_links(root, rustdoc):
    """Resolve literal destinations in generated rustdoc output."""
    versions = {package["name"]: package["version"] for package in tomllib.loads((root / "Cargo.lock").read_text())["package"]}
    qualified = {
        "bloq_graph::BlockGraph": "bloq_graph/struct.BlockGraph.html",
        "bloq_ir::Bloq": "bloq_ir/struct.Bloq.html",
        "bloq_ir::ClassicalNode::Decode": "bloq_ir/enum.ClassicalNode.html#variant.Decode",
        "bloq_ir::circuit::CoordCircuit": "bloq_ir/circuit/struct.CoordCircuit.html",
        "bloq_ir::circuit::CoordCircuit::flatten": "bloq_ir/circuit/struct.CoordCircuit.html#method.flatten",
        "bloq_ir::circuit::Op": "bloq_ir/circuit/enum.Op.html",
        "bloq_compile::CompileConfig": "bloq_compile/struct.CompileConfig.html",
        "bloq_compile::CompileContext": "bloq_compile/struct.CompileContext.html",
    }
    for path in rustdoc.rglob("*.html"):
        original = path.read_text()
        fixed = original
        for name, destination in qualified.items():
            fixed = fixed.replace(f'href="{name}"', f'href="{os.path.relpath(rustdoc / destination, path.parent)}"')
        if 'href="crate#per-style"' in fixed:
            fixed = fixed.replace('href="crate#per-style"', f'href="https://docs.rs/yansi/{versions["yansi"]}/yansi/#per-style"')
        if path.relative_to(rustdoc).as_posix() == "bloq_vm/struct.Simulator.html":
            fixed = fixed.replace('href="self"', f'href="https://docs.rs/ticit/{versions["ticit"]}/ticit/tableau_simulator/index.html"')
        if fixed != original:
            path.write_text(fixed)


class HtmlReferences(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.redirect = None
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        for key in (("id", "name") if tag == "a" else ("id",)):
            if attributes.get(key):
                self.ids.add(attributes[key])
                self.ids.add(unquote(attributes[key]))
        for key in ("href", "src", "poster", "xlink:href"):
            if attributes.get(key):
                self.links.append(attributes[key])
        if tag == "object" and attributes.get("data"):
            self.links.append(attributes["data"])
        if attributes.get("srcset") and not attributes["srcset"].startswith("data:"):
            self.links.extend(item.strip().split()[0] for item in attributes["srcset"].split(",") if item.strip())
        if tag == "meta" and attributes.get("http-equiv", "").lower() == "refresh":
            match = re.search(r"url\s*=\s*(.+)", attributes.get("content", ""), re.I)
            if match:
                self.redirect = match[1].strip("\"'")


def check_links_and_privacy(output, *, incomplete=False):
    output = output.resolve()
    documents = {path: HtmlReferences(path.read_text()) for path in output.rglob("*.html")}
    errors = []

    def resolve(origin, reference):
        url = urlsplit(reference)
        if url.scheme in {"data", "mailto", "tel", "javascript"}:
            return None, ""
        if url.netloc and url.netloc not in {"bloqec.com", "www.bloqec.com"}:
            return None, ""
        if url.scheme and url.scheme not in {"http", "https"}:
            return None, ""
        path = unquote(url.path)
        target = ((output / path.lstrip("/")) if path.startswith("/") else origin.parent / path).resolve() if path else origin
        if target.is_dir():
            target /= "index.html"
        return target, unquote(url.fragment)

    def anchor_exists(target, fragment, seen=None):
        document = documents.get(target)
        if document is None or fragment in document.ids:
            return True
        # Rustdoc highlights source line ranges with JavaScript.
        if "/src/" in target.as_posix() and re.fullmatch(r"\d+(?:-\d+)?", fragment):
            return all(line in document.ids for line in fragment.split("-"))
        seen = set() if seen is None else seen
        if document.redirect and target not in seen:
            seen.add(target)
            destination, _ = resolve(target, document.redirect)
            return destination is not None and anchor_exists(destination, fragment, seen)
        return False

    for origin, document in documents.items():
        for reference in document.links:
            target, fragment = resolve(origin, reference)
            if target is None:
                continue
            if incomplete and (target.is_relative_to(output / "editor") or "/api/rust/" in target.as_posix()):
                continue
            if not target.is_relative_to(output) or not target.is_file():
                errors.append(f"{origin.relative_to(output)}: missing local target {reference}")
            elif fragment and not anchor_exists(target, fragment):
                errors.append(f"{origin.relative_to(output)}: missing anchor {reference}")
    for origin in output.rglob("*.css"):
        for reference in re.findall(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)", origin.read_text()):
            target, _ = resolve(origin, reference.strip())
            if target is not None and (not target.is_relative_to(output) or not target.is_file()):
                errors.append(f"{origin.relative_to(output)}: missing CSS asset {reference}")
    forbidden = ("bloq-paper", "bloq-experiments", "/Users/", "/home/", "/var/folders/")
    text_suffixes = {".html", ".js", ".json", ".svg", ".css", ".txt", ".md", ".rst", ".blog", ".bloqir", ".qasm", ".py"}
    for path in output.rglob("*"):
        if path.name == ".doctrees":
            errors.append(f"{path.relative_to(output)}: Sphinx build cache must not be published")
        if path.is_file() and (path.suffix in text_suffixes or path.suffix == ".wasm"):
            content = path.read_bytes()
            if any(token.encode() in content for token in forbidden):
                errors.append(f"{path.relative_to(output)}: private source or build-host reference")
    if errors:
        raise ValueError(f"Site validation found {len(errors)} invalid reference(s):\n" + "\n".join(errors[:40]))
    print(f"Local links, anchors, assets, and privacy checked: {len(documents)} HTML pages")


def check(output, *, complete=True, links=True):
    manifest = json.loads((output / "versions.json").read_text())
    for entry in manifest["versions"]:
        version_root = output / entry["path"]
        if not version_root.resolve().is_relative_to((output / "docs").resolve()):
            raise ValueError(f"{entry['name']}: invalid version path")
        for required in ("index.html", "searchindex.js", "api/python.html"):
            if not (version_root / required).is_file():
                raise ValueError(f"{entry['name']}: missing {required}")
        if complete and not (version_root / "api" / "rust" / "bloq" / "index.html").is_file():
            raise ValueError(f"{entry['name']}: missing Rust API")
        for page in entry["pages"]:
            if not (version_root / page).resolve().is_relative_to(version_root.resolve()) or not (version_root / page).is_file():
                raise ValueError(f"{entry['name']}: invalid page manifest entry {page}")
    if manifest["stable"] and not any(entry["name"] == manifest["stable"] for entry in manifest["versions"]):
        raise ValueError("Stable alias has no retained release snapshot")
    if complete:
        for required in ("index.html", "version.json", "compile_worker.js", "build/bloq_editor.js", "build/bloq_editor_bg.wasm"):
            if not (output / "editor" / required).is_file():
                raise ValueError(f"Editor is missing {required}")
        worker = (output / "editor" / "compile_worker.js").read_text()
        wrapper = (output / "editor" / "index.html").read_text()
        if '"./build/bloq_editor.js"' not in worker or '"./build/bloq_editor.js"' not in wrapper:
            raise ValueError("Editor and worker must import the relative WASM loader")
    if links:
        check_links_and_privacy(output, incomplete=not complete)
    print(f"Site check passed: {len(manifest['versions'])} documentation version(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--output", type=Path, default=ROOT / "target" / "site")
    parser.add_argument("--tag", action="append", default=[], help="Add an immutable snapshot from an actual Git tag")
    parser.add_argument("--stable", help="Point stable to this actual release tag")
    parser.add_argument("--editor-bundle", type=Path, help="Use an already-built editor bundle from this source")
    parser.add_argument("--docs-only", action="store_true", help="Local Sphinx preview without Rust API/editor builds")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or not output.is_relative_to(ROOT / "target"):
        parser.error("Generated output must be below this checkout's target/ directory")
    if args.docs_only and (args.tag or args.stable):
        parser.error("Release snapshots always include generated Python and Rust APIs")
    if args.command == "check":
        check(output, complete=not args.docs_only)
        return
    tags = list(dict.fromkeys([*args.tag, *([args.stable] if args.stable else [])]))
    for tag in tags:
        tag_commit(tag)
    output.mkdir(parents=True, exist_ok=True)
    source = run(["git", "rev-parse", "HEAD"], capture=True).strip()
    if run(["git", "status", "--porcelain"], capture=True).strip():
        source += "+working-tree"
    build_snapshot(ROOT, output / "docs" / "dev", "dev", source, docs_only=args.docs_only)
    for tag in tags:
        build_release(tag, output)
    if args.stable:
        stable_alias(output, args.stable)
    editor = None
    version = package_version(ROOT)
    if not args.docs_only:
        bundle = args.editor_bundle
        if bundle is None:
            env = os.environ.copy()
            # Dependency panic locations otherwise disclose the build host.
            remap = f"--remap-path-prefix={Path.home()}=. --remap-path-prefix={ROOT}=."
            env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} {remap}".strip()
            run([
                "bevy", "build", "--locked", "--release", "--yes", "-p", "bloq_editor", "--bin", "bloq_editor",
                "web", "--bundle", "--wasm-opt=--strip-debug", "--wasm-opt=-Os",
                "--wasm-opt=--enable-bulk-memory", "--wasm-opt=--enable-nontrapping-float-to-int",
            ], env=env)
            bundle = ROOT / "target" / "bevy_web" / "web-release" / "bloq_editor"
        editor = copy_editor(bundle.resolve(), output, source, version)
    manifest = version_manifest(output, editor)
    site_entrypoints(output, manifest)
    check(output, complete=not args.docs_only)


if __name__ == "__main__":
    main()
