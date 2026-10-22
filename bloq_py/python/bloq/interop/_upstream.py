"""Check source-pinned upstream checkouts without changing their index."""

import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def checked_revision(upstream: Path, audited_revision: str, patch: Path) -> str:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args], env=env)

    revision = git("rev-parse", "HEAD").decode().strip()
    message = f"Upstream must match the audited revision plus exactly {patch.name}"
    if revision != audited_revision:
        raise RuntimeError(message)
    if git("diff", "--cached", "--name-only", "--no-ext-diff", "--no-textconv"):
        raise RuntimeError("Upstream checkout contains staged changes")
    with TemporaryDirectory(prefix="bloq-upstream-") as temporary:
        # Expected new files are indexed here even when upstream ignores them.
        env["GIT_INDEX_FILE"] = str(Path(temporary) / "index")
        git("read-tree", "HEAD")
        git("apply", "--cached", "--whitespace=nowarn", str(patch.resolve()))
        if (git("diff", "--name-only", "--no-ext-diff", "--no-textconv")
                or git("ls-files", "--others", "--exclude-standard")):
            raise RuntimeError(message)
    return revision
