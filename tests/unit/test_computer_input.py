"""Real subprocesses around a fake X11 boundary; not native desktop proof.

The guest's generated source is executed intact, with tiny stand-ins for xclip
and pyautogui. This catches quoting, argv limits, pipe hangs and owned-child
cleanup without importing or installing GUI libraries on the host.
"""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator import computer_input
from local_operator.computer_input import paste_text_source, python_source_argv


@pytest.fixture
def guest(tmp_path: Path) -> Iterator[dict[str, str]]:
    xclip = tmp_path / "xclip"
    xclip.write_text(f"#!{sys.executable}\n" + """
import os, pathlib, sys, time
root = pathlib.Path(os.environ["PASTE_TEST_ROOT"])
assert sys.argv[1:3] == ["-selection", "clipboard"]
assert sys.argv[4:6] == ["-target", "UTF8_STRING"]
if "-in" in sys.argv:
    assert sys.argv[6:] == ["-quiet", "-loops", "0"]
    root.joinpath("owner.pid").write_text(str(os.getpid()))
    data = sys.stdin.buffer.read()
    if os.environ.get("PASTE_FAILURE") == "owner":
        sys.exit(7)
    root.joinpath("payload").write_bytes(data)
    while True:
        time.sleep(0.1)
else:
    failure = os.environ.get("PASTE_FAILURE")
    if failure == "hang":
        root.joinpath("reader.pid").write_text(str(os.getpid()))
        time.sleep(30)
    elif failure == "overflow":
        sys.stdout.buffer.write(b"x" * 1_000_000)
    elif failure == "read":
        sys.stderr.write("PRIVATE_PAYLOAD")
        sys.exit(4)
    else:
        try:
            sys.stdout.buffer.write(root.joinpath("payload").read_bytes())
        except FileNotFoundError:
            sys.exit(1)
""")
    xclip.chmod(0o700)
    (tmp_path / "pyautogui.py").write_text("""
import json, os, pathlib
root = pathlib.Path(os.environ["PASTE_TEST_ROOT"])
def hotkey(*keys):
    root.joinpath("keys").write_text(json.dumps(keys))
    if os.environ.get("PASTE_FAILURE") == "key":
        raise ValueError("PRIVATE_PAYLOAD")
def press(key):
    hotkey(key)
""")
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONPATH": str(tmp_path),
        "PASTE_TEST_ROOT": str(tmp_path),
    }
    yield environment
    # Only children whose PID was written by this test's executable are owned
    # here. Production ownership ends on selection replacement, tested natively
    # by the parent proof rather than claimed by this fake X11 service.
    for name in ("owner.pid", "reader.pid"):
        path = tmp_path / name
        if path.exists():
            try:
                os.kill(int(path.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def run_source(source: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    argv = python_source_argv(source)
    assert all(len(arg.encode("utf-8")) <= 64_000 for arg in argv[1:])
    return subprocess.run(
        [sys.executable, *argv[1:]],
        env=environment,
        text=True,
        capture_output=True,
        timeout=12,
    )


@pytest.mark.parametrize(
    "text",
    [
        "café 東京🙂",
        "a\tb\nnext\r\n",
        " \t\n",
        "'; __import__('os').abort(); #",
        "🙂" * 100_000,
    ],
    ids=["unicode", "tabs-newlines", "whitespace", "injection", "100k-unicode"],
)
def test_generated_source_transfers_exact_utf8_and_only_explicit_chord(
    guest: dict[str, str],
    tmp_path: Path,
    text: str,
) -> None:
    import json

    # The actual controller prefixes a semicolon, so do not test a more
    # permissive multiline-only script host than production uses.
    source = "import pyautogui; " + paste_text_source(text, ["ctrl", "shift", "v"])
    result = run_source(source, guest)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "payload").read_bytes() == text.encode("utf-8")
    assert json.loads((tmp_path / "keys").read_text()) == ["ctrl", "shift", "v"]
    assert result.stdout == ""
    assert text not in result.stderr


@pytest.mark.parametrize("failure", ["owner", "overflow", "hang", "read", "key"])
def test_failure_is_bounded_redacted_and_does_not_retry(
    guest: dict[str, str],
    tmp_path: Path,
    failure: str,
) -> None:
    guest["PASTE_FAILURE"] = failure
    result = run_source(paste_text_source("PRIVATE_PAYLOAD", ["ctrl", "v"]), guest)
    assert result.returncode != 0
    assert "PRIVATE_PAYLOAD" not in result.stderr
    assert "clipboard paste failed" in result.stderr
    assert (tmp_path / "keys").exists() == (failure == "key")
    # The generated helper reaps the owned foreground owner before it exits.
    owner = int((tmp_path / "owner.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(owner, 0)


def test_source_arguments_are_literals_not_caller_injection(
    guest: dict[str, str], tmp_path: Path
) -> None:
    import json

    key = "v'); __import__('os').abort(); #"
    result = run_source(paste_text_source("x", [key]), guest)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "keys").read_text()) == [key]


def test_even_a_small_command_is_encoded_rather_than_quoted_literally() -> None:
    """No size fast path: a short statement is exactly the dangerous case.

    This test previously asserted the opposite (``["python", "-c", source]``).
    That literal shape is the argv exposure itself, and the payload that killed
    ep-0ce67ac2d3a1 was a few dozen characters — well under any threshold a
    "small commands stay legible" carve-out would have used.
    """
    source = "pyautogui.typewrite('abc')"
    argv = python_source_argv(source)
    assert source not in " ".join(argv)
    assert argv[:3] == ["python", "-c", computer_input._SOURCE_BOOTSTRAP]
    assert base64.b64decode("".join(argv[3:])).decode("utf-8") == source


def test_large_argv_reconstruction_executes_once(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    source = (
        f"from pathlib import Path; Path({str(marker)!r}).write_text('once')\n#" + "🙂" * 100_000
    )
    result = run_source(source, os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "once"
