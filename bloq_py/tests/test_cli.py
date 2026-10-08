"""Exercise the installed console script, not a separate Python CLI parser."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest
import stim

import bloq

CLI = Path(sysconfig.get_path("scripts")) / ("bloq.exe" if os.name == "nt" else "bloq")


@pytest.mark.parametrize(
    "args, status, text",
    [
        (["--help"], 0, "Usage: bloq"),
        (["--version"], 0, f"bloq {bloq.__version__}"),
        ([], 2, "Usage: bloq"),
        (["--unknown"], 2, "unexpected argument"),
        (["compile", "--help"], 0, "--distance"),
        (["gallery"], 0, "cnot"),
        (["completion", "bash"], 0, "_bloq"),
        (["--gallery", "cnot", "-d", "4", "--quiet"], 1, "code distance"),
    ],
)
def test_cli_output_and_exit_status(args, status, text):
    result = subprocess.run([CLI, *args], capture_output=True, text=True, encoding="utf-8", check=False)
    assert result.returncode == status, result.stderr
    assert text in result.stdout + result.stderr
    if status == 0:
        assert not result.stderr


def test_cli_compile_validate_emit_and_view(tmp_path):
    # Linux permits non-UTF-8 filenames; APFS requires valid UTF-8.
    name = os.fsdecode(b"source-\xff.blog") if sys.platform == "linux" else "source.blog"
    source = tmp_path / name
    source.write_text(bloq.GalleryItem.CNOT.source())
    program = tmp_path / "program.bloq"
    circuit = tmp_path / "circuit.stim"
    commands = [
        [source, "-d", "3", "--backend", "ir-binary", "-o", program],
        ["validate", program],
        ["emit", program, "--validate", "-o", circuit],
        ["view", source, "--html"],
    ]
    for args in commands:
        result = subprocess.run([CLI, *args, "--quiet"], capture_output=True, check=False)
        assert result.returncode == 0, result.stderr
        assert not result.stdout
        assert not result.stderr
    assert stim.Circuit.from_file(circuit) == bloq.compile_to_stim(bloq.GalleryItem.CNOT.load(), 3)
    assert list(tmp_path.glob("*.html"))
    result = subprocess.run([CLI, "stats", program, "--quiet"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(bloq.Bloq.load(program).stats()) + "\n"
    assert not result.stderr


def test_cli_broken_pipe_exits_successfully():
    with subprocess.Popen([CLI, "completion", "bash"], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        process.stdout.close()
        assert process.wait(timeout=30) == 0
        assert not process.stderr.read()
