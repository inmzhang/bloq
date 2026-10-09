"""Start development after a release, keeping the Cargo workspace in lockstep."""

import argparse
from pathlib import Path
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]
STABLE_VERSION = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
VERSION = re.compile(STABLE_VERSION + r"(?:-dev)?")


def current_version(root=ROOT):
    return tomllib.loads((root / "Cargo.toml").read_text())["workspace"]["package"]["version"]


def version_manifest(source, version):
    if not VERSION.fullmatch(version):
        raise ValueError("Use a stable Cargo version or MAJOR.MINOR.PATCH-dev")
    workspace = tomllib.loads(source)["workspace"]
    current = workspace["package"]["version"]
    # The root manifest owns the only non-inherited package version.
    source, count = re.subn(r'^version = "' + re.escape(current) + r'"$',
                            f'version = "{version}"', source, flags=re.M)
    if count != 1:
        raise ValueError("Expected one shared workspace package version")
    for name, dependency in workspace["dependencies"].items():
        if not isinstance(dependency, dict) or "path" not in dependency or "version" not in dependency:
            continue
        if dependency["version"] != f"={current}":
            raise ValueError(f"{name} must require the exact workspace version")
        pattern = r"^(" + re.escape(name) + r'\s*=\s*\{[^\n]*\bversion\s*=\s*")[^"]*(")'
        source, count = re.subn(pattern, lambda match: f"{match[1]}={version}{match[2]}",
                                source, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected one inline dependency definition for {name}")
    return source


def refresh_lockfiles(root=ROOT):
    for manifest in ("Cargo.toml", "docs/examples/Cargo.toml"):
        subprocess.run(["cargo", "update", "--workspace",
                        "--manifest-path", manifest], cwd=root, check=True)


def set_version(version, root=ROOT):
    paths = [root / name for name in ("Cargo.toml", "Cargo.lock", "docs/examples/Cargo.lock")]
    originals = {path: path.read_bytes() for path in paths}
    updated = version_manifest(originals[paths[0]].decode(), version)
    try:
        paths[0].write_text(updated)
        refresh_lockfiles(root)
    except BaseException:
        for path, content in originals.items():
            path.write_bytes(content)
        raise


def next_dev_version(version):
    match = re.fullmatch(STABLE_VERSION, version)
    if not match:
        raise ValueError("Start development from a stable release version")
    major, minor, patch = map(int, match.groups())
    return f"{major}.{minor}.{patch + 1}-dev"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dev", "release-baseline", "lockfiles", "current", "set"])
    parser.add_argument("version", nargs="?")
    args = parser.parse_args()
    if args.version is not None and args.command != "set":
        parser.error("only set accepts a version argument")
    current = current_version()
    try:
        if args.command == "current":
            print(current)
            return
        if args.command == "lockfiles":
            refresh_lockfiles()
            return
        if args.command == "set":
            if args.version is None:
                parser.error("set requires a version")
            target = args.version
        elif args.command == "dev":
            target = next_dev_version(current)
            subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/v{current}^{{commit}}"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        else:
            if re.fullmatch(STABLE_VERSION, current):
                return
            tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0",
                                           "--match", "v[0-9]*", "--exclude", "*-*",
                                           "--exclude", "*+*"], cwd=ROOT, text=True).strip()
            target = tag.removeprefix("v")
            if not re.fullmatch(STABLE_VERSION, target):
                raise ValueError("Release baseline must be a stable version tag")
        set_version(target)
        print(f"Workspace version: {current} -> {target}")
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
