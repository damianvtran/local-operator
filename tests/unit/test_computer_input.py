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
from local_operator.computer_input import (
    drag_source,
    paste_text_source,
    python_source_argv,
)


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


# ---------------------------------------------------------------------------
# Drag: the generated helper is executed intact against a fake pyautogui,
# exactly as the paste tests drive a fake X11 boundary. The fake records call
# order, the FAILSAFE value seen at each call and the tracked button state, so
# the release guarantees are asserted without any real pointer.

_FAKE_PYAUTOGUI = """
import json, os, pathlib

root = pathlib.Path(os.environ["DRAG_TEST_ROOT"])
held = None
FAILSAFE = True


def _record(call, **fields):
    with root.joinpath("calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"call": call, "failsafe": FAILSAFE, "held": held, **fields}))
        handle.write("\\n")


def moveTo(x, y, duration=0.0):
    _record("moveTo", x=x, y=y, duration=duration)
    failure = os.environ.get("DRAG_FAILURE")
    if failure == "motion" and duration:
        raise RuntimeError("motion refused")
    if failure == "cancel" and duration:
        raise KeyboardInterrupt()
    if os.environ.get("DRAG_CORNER") == "1" and duration:
        raise RuntimeError("corner refused (FailSafeException)")


def mouseDown(button="left"):
    global held
    held = button
    _record("mouseDown", button=button)
    if os.environ.get("DRAG_FAILURE") == "press":
        raise RuntimeError("press injection failed")


def mouseUp(button="left"):
    global held
    _record("mouseUp", button=button)
    failure = os.environ.get("DRAG_FAILURE")
    if failure in ("up", "up-all"):
        marker = root.joinpath("up-attempts")
        attempts = int(marker.read_text()) if marker.exists() else 0
        marker.write_text(str(attempts + 1))
        if failure == "up-all" or attempts == 0:
            raise RuntimeError("release refused")
    held = None


def _probe():
    _record("probe")
"""

_DRAG_ARGS = {
    "start_x": 10,
    "start_y": 20,
    "end_x": 30,
    "end_y": 40,
    "button": "left",
    "duration_ms": 500,
}


def _drag_source_with_probe() -> str:
    # The probe must still run when the drag raises, so it goes in a finally:
    # the assertions inspect the RESTORED FAILSAFE and the tracked button
    # state after the failure paths, not only after success.
    drag = drag_source(**_DRAG_ARGS)
    return f"import pyautogui\ntry:\n    {drag}\nfinally:\n    pyautogui._probe()\n"


def _drag_calls(tmp_path: Path) -> list[dict[str, object]]:
    import json

    path = tmp_path / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def drag_guest(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "pyautogui.py").write_text(_FAKE_PYAUTOGUI)
    return {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "DRAG_TEST_ROOT": str(tmp_path),
    }


def test_drag_source_bounds_mirror_the_protocol_and_refuse_bad_input() -> None:
    from local_operator.computer_input import (
        DRAG_MAX_DURATION_MS,
        DRAG_MIN_DURATION_MS,
        MAX_DRAG_COORDINATE,
    )
    from local_operator.evaluation.protocol import MAX_DRAG_MS

    # Drift guard: the helper's ceiling must move with the protocol's, or a
    # batch the protocol admits would die in the guest, and one it refuses
    # could still be compiled.
    assert DRAG_MIN_DURATION_MS == 100
    assert DRAG_MAX_DURATION_MS == MAX_DRAG_MS
    for overrides, message in [
        ({"duration_ms": 99}, "duration"),
        ({"duration_ms": 5001}, "duration"),
        ({"button": "grab"}, "button"),
        ({"start_x": -1}, "coordinates"),
        ({"end_y": MAX_DRAG_COORDINATE + 1}, "coordinates"),
        ({"start_x": True}, "coordinates"),
    ]:
        with pytest.raises(ValueError, match=message):
            drag_source(**{**_DRAG_ARGS, **overrides})
    source = drag_source(**_DRAG_ARGS)
    # The compiled statement embeds its guest source as a single-quoted
    # literal; decode it so the assertion reads the actual guest code rather
    # than its outer escaping.
    import ast

    decoded = ast.literal_eval(source[len("exec(") : -1])
    assert "_drag_x11(10, 20, 30, 40, 'left', 500)" in decoded


def test_drag_success_orders_moves_around_press_and_release(
    drag_guest: dict[str, str], tmp_path: Path
) -> None:
    result = run_source(_drag_source_with_probe(), drag_guest)
    assert result.returncode == 0, result.stderr
    calls = _drag_calls(tmp_path)
    assert [call["call"] for call in calls] == [
        "moveTo",
        "mouseDown",
        "moveTo",
        "mouseUp",
        "probe",
    ]
    assert (calls[0]["x"], calls[0]["y"], calls[0]["duration"]) == (10, 20, 0.0)
    assert calls[1]["button"] == "left"
    # The duration rides the PRESSED move, in seconds.
    assert (calls[2]["x"], calls[2]["y"], calls[2]["duration"]) == (30, 40, 0.5)
    assert calls[3]["button"] == "left"
    # Success path never touches FAILSAFE, and leaves no held button.
    assert all(call["failsafe"] is True for call in calls)
    assert calls[-1]["held"] is None


@pytest.mark.parametrize("failure", ["motion", "press", "corner"])
def test_drag_failure_still_releases_under_a_restored_failsafe(
    drag_guest: dict[str, str], tmp_path: Path, failure: str
) -> None:
    if failure == "corner":
        drag_guest["DRAG_CORNER"] = "1"
    else:
        drag_guest["DRAG_FAILURE"] = failure
    result = run_source(_drag_source_with_probe(), drag_guest)
    assert result.returncode != 0
    assert "button was released" in result.stderr
    assert "could not be released" not in result.stderr
    calls = _drag_calls(tmp_path)
    ups = [call for call in calls if call["call"] == "mouseUp"]
    assert len(ups) == 1
    # FAILSAFE is disarmed ONLY for the cleanup release, and the probe sees
    # the original value restored afterwards.
    assert ups[0]["failsafe"] is False
    assert calls[-1]["call"] == "probe"
    assert calls[-1]["failsafe"] is True
    assert calls[-1]["held"] is None


def test_drag_release_failure_is_reported_not_laundered(
    drag_guest: dict[str, str], tmp_path: Path
) -> None:
    drag_guest["DRAG_FAILURE"] = "up"
    result = run_source(_drag_source_with_probe(), drag_guest)
    assert result.returncode != 0
    # The normal release failed, the cleanup release succeeded: truthful
    # partial report, never a success claim, never a "could not" claim.
    assert "button was released" in result.stderr
    assert "could not be released" not in result.stderr
    calls = _drag_calls(tmp_path)
    ups = [call for call in calls if call["call"] == "mouseUp"]
    assert len(ups) == 2
    assert ups[0]["failsafe"] is True
    assert ups[1]["failsafe"] is False
    assert calls[-1]["failsafe"] is True
    assert calls[-1]["held"] is None


def test_drag_unobservable_release_is_a_truthful_failure(
    drag_guest: dict[str, str], tmp_path: Path
) -> None:
    drag_guest["DRAG_FAILURE"] = "up-all"
    result = run_source(_drag_source_with_probe(), drag_guest)
    assert result.returncode != 0
    assert "could not be released" in result.stderr
    assert "button was released" not in result.stderr
    calls = _drag_calls(tmp_path)
    ups = [call for call in calls if call["call"] == "mouseUp"]
    assert len(ups) == 2
    # FAILSAFE is still restored even though the release failed.
    assert calls[-1]["failsafe"] is True


def test_drag_cancellation_unwinds_and_releases(drag_guest: dict[str, str], tmp_path: Path) -> None:
    drag_guest["DRAG_FAILURE"] = "cancel"
    result = run_source(_drag_source_with_probe(), drag_guest)
    assert result.returncode != 0
    assert "KeyboardInterrupt" in result.stderr
    calls = _drag_calls(tmp_path)
    ups = [call for call in calls if call["call"] == "mouseUp"]
    assert len(ups) == 1
    assert ups[0]["failsafe"] is False
    assert calls[-1]["failsafe"] is True
    assert calls[-1]["held"] is None
