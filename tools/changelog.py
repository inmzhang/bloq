"""Validate proposed PR metadata or an explicit range of new merge messages."""

import argparse
import json
import re
import subprocess
from pathlib import Path

TITLE = re.compile(r"(?:feat|fix|perf|refactor|docs|test|chore|ci|build|style|revert)(?:\([^\n)]+\))?!?: .+")
STACK = re.compile(r"^\s*[*-] (?:feat|fix|perf|refactor|docs|test|chore|ci|build|style|revert)(?:\([^\n)]+\))?!?: ", re.MULTILINE)
BREAKING = re.compile(r"^BREAKING[ -]CHANGE:", re.MULTILINE)


def validate(message: str) -> None:
    title = message.partition("\n")[0]
    if not TITLE.fullmatch(title):
        raise ValueError("use a Conventional Commit title, e.g. fix(editor): preserve selection")
    if STACK.search(message) or len(BREAKING.findall(message)) > 1:
        raise ValueError("write one PR summary, not a copied commit stack; consolidate breaking changes into one footer")
    if re.search(r"!:", title) and not BREAKING.search(message):
        raise ValueError("describe the migration in a BREAKING CHANGE: footer")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--event", type=Path, help="GitHub pull_request event JSON")
    source.add_argument("--range", help="explicit new first-parent commit range")
    args = parser.parse_args()
    if args.event:
        pr = json.loads(args.event.read_text())["pull_request"]
        validate(pr["title"] + "\n\n" + (pr["body"] or ""))
    else:
        for sha in git("rev-list", "--first-parent", args.range).splitlines():
            try:
                validate(git("show", "-s", "--format=%B", sha))
            except ValueError as error:
                raise ValueError(f"{sha[:12]}: {error}") from error


if __name__ == "__main__":
    main()
