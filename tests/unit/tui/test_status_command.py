"""The status command runner: what a script's output becomes on the band.

Real subprocesses through ``/bin/sh``, because the contract under test IS the
process boundary — stdin payload in, first stdout line out, failures and
timeouts collapsing to an empty segment, and no child outliving its run.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from rich.cells import cell_len

from local_operator.tui import status_command as sc

_PAYLOAD = sc.build_payload(
    session_id="abc123",
    cwd="/tmp",
    provider="anthropic",
    model_id="claude-x",
    model_label="Claude X",
    context_tokens=1200,
    context_window=200_000,
    cost="$0.12",
    version="0.0.0",
)


def _run(command: str, **kwargs) -> str:  # noqa: ANN003
    return asyncio.run(sc.run(command, _PAYLOAD, **kwargs))


def test_the_first_non_blank_line_is_the_segment() -> None:
    assert _run("printf '\\n  \\nfirst line\\nsecond line\\n'") == "first line"


def test_the_payload_arrives_as_json_on_stdin() -> None:
    """A script can read the session it is describing; the shape is the contract."""
    line = _run(
        "python3 -c 'import json,sys; d=json.load(sys.stdin); "
        'print(d["model"]["id"], d["context"]["window"], d["session_id"])\''
    )
    assert line == "claude-x 200000 abc123"
    assert set(_PAYLOAD) == {"session_id", "cwd", "model", "context", "cost", "version"}
    json.dumps(_PAYLOAD)  # serialisable as-is


def test_a_script_that_ignores_stdin_still_works() -> None:
    """Most scripts never read stdin; an early exit must not read as failure."""
    assert _run("echo ok") == "ok"


@pytest.mark.parametrize("command", ["exit 3", "echo partial; exit 1", "true", "printf ''"])
def test_failure_or_no_output_clears_the_segment(command: str) -> None:
    assert _run(command) == ""


def test_an_empty_command_runs_nothing() -> None:
    assert _run("   ") == ""


def test_a_hung_script_times_out_and_its_process_group_is_reaped(tmp_path) -> None:
    """The child AND anything it spawned are killed: a helper left running
    would live for the whole TUI session, once per tick."""
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    line = _run(f"sleep 30 & echo $! > {pidfile}; wait", timeout=0.5)
    assert line == ""
    assert time.monotonic() - started < 5
    grandchild = int(pidfile.read_text().strip())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(grandchild, 9)
        raise AssertionError("the script's background child survived the timeout")


def test_cancellation_reaps_the_child(tmp_path) -> None:
    """An exclusive worker cancelled by the next tick must not leak its shell."""
    pidfile = tmp_path / "child.pid"

    async def scenario() -> int:
        task = asyncio.ensure_future(sc.run(f"echo $$ > {pidfile}; sleep 30", _PAYLOAD, timeout=60))
        for _ in range(100):
            await asyncio.sleep(0.02)
            if pidfile.exists() and pidfile.read_text().strip():
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return int(pidfile.read_text().strip())

    pid = asyncio.run(scenario())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    raise AssertionError("the cancelled run's shell survived")


def test_escapes_and_controls_never_reach_the_band() -> None:
    """Colour, OSC titles/hyperlinks and control bytes are stripped.

    The band is measured cell arithmetic, and an escape from a script would be
    a write to the operator's terminal that bypasses the TUI.
    """
    raw = (
        "\x1b[31mred\x1b[0m \x1b]0;title\x07\x1b]8;;https://x\x1b\\link\x1b]8;;\x1b\\\ttab\x07bell"
    )
    assert sc.sanitize(raw) == "red link tab bell"


def test_long_output_is_capped_to_the_bounded_rung() -> None:
    line = sc.sanitize("x" * 500)
    assert cell_len(line) <= sc.MAX_CELLS
    assert line.endswith("…")
    wide = sc.sanitize("用" * 100)
    assert cell_len(wide) <= sc.MAX_CELLS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (30, 30),
        ("45", 45),
        (1, sc.MIN_INTERVAL_S),
        (10**9, sc.MAX_INTERVAL_S),
        (None, sc.DEFAULT_INTERVAL_S),
        ("soon", sc.DEFAULT_INTERVAL_S),
    ],
)
def test_the_interval_is_clamped_not_trusted(value: object, expected: int) -> None:
    assert sc.clamp_interval(value) == expected
