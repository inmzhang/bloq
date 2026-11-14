"""Validate proposed PR metadata or an explicit range of new merge messages."""

import argparse
import json
import re
import subprocess
from pathlib import Path

TITLE = re.compile(r"(?:feat|fix|perf|refactor|docs|test|chore|ci|build|style|revert)(?:\([^\n)]+\))?!?: .+")
STACK = re.compile(r"^\s*[*-] (?:feat|fix|perf|refactor|docs|test|chore|ci|build|style|revert)(?:\([^\n)]+\))?!?: ", re.MULTILINE)
BREAKING = re.compile(r"^BREAKING[ -]CHANGE:([^\n]*)", re.MULTILINE)


def validate(message: str) -> None:
    title = message.partition("\n")[0]
    if not TITLE.fullmatch(title):
        raise ValueError("use a Conventional Commit title, e.g. fix(editor): preserve selection")
    if STACK.search(message) or len(BREAKING.findall(message)) > 1:
        raise ValueError("write one PR summary, not a copied commit stack; consolidate breaking changes into one footer")
    if any(not detail.strip() for detail in BREAKING.findall(message)):
        raise ValueError("provide migration guidance after BREAKING CHANGE:")
    if re.search(r"!:", title) and not BREAKING.search(message):
        raise ValueError("describe the migration in a BREAKING CHANGE: footer")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def release_range() -> str:
    # The first mainline commit introducing this checker starts enforcement.
    # A subsequent stable release becomes the next reviewed boundary.
    adoption = git("log", "--first-parent", "--diff-filter=A", "--format=%H",
                   "--", "tools/changelog.py").splitlines()[-1]
    stable_tags = set(git("tag", "--merged", "HEAD", "--list", "v*").splitlines())
    stable_tags = {tag for tag in stable_tags if re.fullmatch(r"v\d+\.\d+\.\d+", tag)}
    releases = {git("rev-parse", tag + "^{commit}").strip() for tag in stable_tags}
    for sha in git("rev-list", "--first-parent", "HEAD").splitlines():
        if sha in releases:
            return sha + "..HEAD"
        if sha == adoption:
            return sha + "^..HEAD"
    raise ValueError("cannot find release metadata adoption on the mainline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=Path, help="GitHub pull_request event JSON")
    source.add_argument("--range", help="explicit new first-parent commit range")
    source.add_argument("--release", action="store_true", help="check all prospective commits in the pending release")
    args = parser.parse_args()
    if args.event:
        pr = json.loads(args.event.read_text())["pull_request"]
        validate(pr["title"] + "\n\n" + (pr["body"] or ""))
    else:
        revision_range = release_range() if args.release else args.range
        for sha in git("rev-list", "--first-parent", revision_range).splitlines():
            try:
                validate(git("show", "-s", "--format=%B", sha))
            except ValueError as error:
                raise ValueError(f"{sha[:12]}: {error}") from error


if __name__ == "__main__":
    main()
