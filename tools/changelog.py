"""Repair known squash metadata and reject ambiguous new release messages."""

import argparse
import os
import re
import subprocess
import sys

# Last reviewed history before this guard; older valid squash lists remain intact.
REVIEWED_HISTORY = "fe8fd09c5635e78a0102ae2da17811a7f8d47b03"
CORRECTIONS = {
    "31ebfd460165d30d9675abdc2410aafc23a0d5b6": "last",  # #28 inherited #27
    "11aad7c5a852be6cbef7ad28eaa9664d8c0983e3": "all",   # #31 has two real migrations
    "bdbbd66b98577e7f5e098cf7461a41edb6fc4441": "none",  # #32 inherited #31
}
BREAKING = re.compile(r"^BREAKING[ -]CHANGE: (.*(?:\n(?!\n|[\w-]+:).*)*)", re.MULTILINE)
STACK = re.compile(
    r"^\s*[*-] (?:feat|fix|perf|refactor|docs|test|chore|ci|build|style|revert)(?:\([^\n)]+\))?!?: ",
    re.MULTILINE,
)


def corrected_message(sha: str, message: str) -> str:
    mode = CORRECTIONS.get(sha)
    if mode is None:
        return message
    footers = BREAKING.findall(message)
    if mode == "last":
        footers = footers[-1:]
    elif mode == "none":
        footers = []
    subject = message.splitlines()[0]
    migration = " ".join(" ".join(footer.splitlines()) for footer in footers)
    return subject + (f"\n\nBREAKING CHANGE: {migration}" if migration else "") + "\n"


def validate(message: str) -> None:
    if STACK.search(message) or len(re.findall(r"^BREAKING[ -]CHANGE:", message, re.MULTILINE)) > 1:
        raise ValueError("replace the stacked commit list with one squash summary and "
                         "one consolidated BREAKING CHANGE footer (if needed)")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check commits since reviewed history")
    args = parser.parse_args()
    if args.check:
        # Fail if history is shallow or unrelated rather than silently skipping checks.
        subprocess.run(["git", "merge-base", "--is-ancestor", REVIEWED_HISTORY, "HEAD"], check=True)
        failures = []
        for sha in git("rev-list", f"{REVIEWED_HISTORY}..HEAD").splitlines():
            message = git("show", "-s", "--format=%B", sha)
            try:
                validate(corrected_message(sha, message))
            except ValueError as error:
                failures.append(f"{sha[:12]}: {error}")
        if failures:
            sys.exit("Ambiguous release metadata:\n" + "\n".join(failures))
    else:
        sys.stdout.write(corrected_message(os.environ["COMMIT_SHA"], sys.stdin.read()))


if __name__ == "__main__":
    main()
