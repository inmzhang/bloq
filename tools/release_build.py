"""Remove build-host paths from Rust/C/C++ release artifacts."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


def build_environment(environ, root, home):
    """Preserve caller flags and append source-path mappings for all compilers."""
    env = dict(environ)
    cargo_home = Path(env.get("CARGO_HOME", home / ".cargo")).resolve()
    mappings = [(home, "/build"), (cargo_home, "/cargo"), (root, "/bloq")]
    rustflags = (
        env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        if env.get("CARGO_ENCODED_RUSTFLAGS")
        else shlex.split(env.get("RUSTFLAGS", ""))
    )
    rustflags.extend(f"--remap-path-prefix={source}={dest}" for source, dest in mappings)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rustflags)
    # cc-rs uses shell parsing when requested, including paths containing spaces.
    env["CC_SHELL_ESCAPED_FLAGS"] = "1"
    target = env.get("CARGO_BUILD_TARGET")
    msvc = target.endswith("-msvc") if target else os.name == "nt"
    prefix = "/pathmap:" if msvc else "-ffile-prefix-map="
    for name in ("CFLAGS", "CXXFLAGS"):
        flags = shlex.split(env.get(name, ""))
        if msvc:
            flags.append("/experimental:deterministic")
        flags.extend(f"{prefix}{source}={dest}" for source, dest in mappings)
        env[name] = shlex.join(flags)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--github-env", action="store_true")
    output.add_argument("--shell", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = build_environment(os.environ, Path.cwd().resolve(), Path.home().resolve())
    keys = ("CARGO_ENCODED_RUSTFLAGS", "CC_SHELL_ESCAPED_FLAGS", "CFLAGS", "CXXFLAGS")
    if args.github_env:
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as output:
            for key in keys:
                output.write(f"{key}={env[key]}\n")
    elif args.shell:
        for key in keys:
            print(f"export {key}={shlex.quote(env[key])}")
    elif args.command:
        return subprocess.call(args.command, env=env)
    else:
        parser.error("provide a build command, --github-env, or --shell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
