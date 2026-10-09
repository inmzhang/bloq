"""Smoke-test a wheel or source installation from outside the checkout."""

from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys

import bloq
from packaging.version import Version
import stim


def main():
    expected = os.environ["BLOQ_EXPECTED_VERSION"]
    assert bloq.__version__ == expected
    assert Version(version("bloq-py")) == Version(expected)
    cli = Path(sys.executable).with_name("bloq.exe" if os.name == "nt" else "bloq")
    subprocess.run([cli, "--help"], check=True)
    assert subprocess.check_output([cli, "--version"], text=True).strip() == f"bloq {expected}"
    subprocess.run([cli, "compile", "--gallery", "cnot", "-d", "3", "--quiet", "-o", "cli-cnot.stim"], check=True)
    graph = bloq.GalleryItem.CNOT.load()
    program = bloq.compile(graph, distance=3)
    assert bloq.emit_stim(program) == bloq.compile_to_stim(graph, 3)
    assert stim.Circuit.from_file("cli-cnot.stim") == bloq.emit_stim(program)


if __name__ == "__main__":
    main()
