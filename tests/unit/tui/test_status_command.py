"""The status command runner: what a script's output becomes on the band.

Real subprocesses through ``/bin/sh``, because the contract under test IS the
process boundary — stdin payload in, first stdout line out, failures and
timeouts collapsing to an empty segment, and no child outliving its run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
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


def _outcome(command: str, **kwargs) -> sc.Outcome:  # noqa: ANN003
    return asyncio.run(sc.run(command, _PAYLOAD, **kwargs))


def _run(command: str, **kwargs) -> str:  # noqa: ANN003
    return _outcome(command, **kwargs).line


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


@pytest.mark.parametrize(
    ("command", "error"),
    [
        ("exit 3", "exit 3"),
        ("echo partial; exit 1", "exit 1"),
        ("/no/such/status-script", "exit 127"),
    ],
)
def test_a_failure_reports_why_and_paints_nothing(command: str, error: str) -> None:
    assert _outcome(command) == sc.Outcome(line="", error=error)


@pytest.mark.parametrize("command", ["true", "printf ''"])
def test_no_output_is_a_success_that_clears(command: str) -> None:
    assert _outcome(command) == sc.Outcome()


def test_an_empty_command_runs_nothing() -> None:
    assert _run("   ") == ""


def test_a_hung_script_times_out_and_its_process_group_is_reaped(tmp_path) -> None:
    """The child AND anything it spawned are killed: a helper left running
    would live for the whole TUI session, once per tick."""
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    outcome = _outcome(f"sleep 30 & echo $! > {pidfile}; wait", timeout=0.5)
    assert outcome == sc.Outcome(error="timed out after 0.5s")
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


def test_timeout_reaps_pipe_holder_after_shell_exits(tmp_path) -> None:
    """The session leader can exit before its background child closes stdout."""
    pidfile = tmp_path / "orphan.pid"
    outcome = _outcome(f"sleep 30 & echo $! > {pidfile}", timeout=0.5)
    pid = int(pidfile.read_text().strip())
    try:
        assert outcome == sc.Outcome(error="timed out after 0.5s")
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("pipe-holding child survived its shell and timeout")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


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


def test_a_flooding_script_is_bounded_in_memory_and_time() -> None:
    """``yes`` writes forever: the run must time out promptly with bounded memory.

    ``communicate()`` buffered all of stdout before the 4 KiB slice, measured at
    3.6 GB inside a one-second timeout, and the reap then waited out its own
    limit because nothing drained the pipe (review round 1, M1). Peak RSS is
    the process's high-water mark, so it is compared before and after.
    """
    import resource

    def peak_mb() -> float:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024

    before = peak_mb()
    started = time.monotonic()
    assert _run("yes flooding", timeout=1.0) == ""
    elapsed = time.monotonic() - started
    assert elapsed < 2.5, f"timeout+reap took {elapsed:.2f}s"
    assert peak_mb() - before < 64, f"peak RSS grew {peak_mb() - before:.0f} MB"


def test_a_script_that_prints_a_lot_then_exits_keeps_its_first_line() -> None:
    """Draining past the cap is what lets a verbose script finish successfully."""
    assert _run("echo headline; head -c 2000000 /dev/zero | tr '\\0' x; echo") == "headline"


def test_escapes_and_controls_never_reach_the_band() -> None:
    """Colour, OSC titles/hyperlinks and control bytes are stripped.

    The band is measured cell arithmetic, and an escape from a script would be
    a write to the operator's terminal that bypasses the TUI.
    """
    raw = (
        "\x1b[31mred\x1b[0m \x1b]0;title\x07\x1b]8;;https://x\x1b\\link\x1b]8;;\x1b\\\ttab\x07bell"
    )
    assert sc.sanitize(raw) == "red link tabbell"  # BEL is removed, not spaced


def test_a_large_payload_does_not_deadlock_a_script_that_prints_first() -> None:
    """stdin is fed concurrently with the stdout read (review round 2, N3)."""
    # ~2 MB both ways: well past the pipe buffers, which is where the old
    # write-then-read order deadlocked; 200 KB passed with or without the fix.
    big = dict(_PAYLOAD, cwd="x" * 2_000_000)
    started = time.monotonic()
    outcome = asyncio.run(
        sc.run("head -c 2000000 /dev/zero | tr '\\0' y; echo; cat >/dev/null", big, timeout=5)
    )
    assert time.monotonic() - started < 3
    assert outcome.error == "" and outcome.line.startswith("yyyy")


def test_bidi_overrides_and_invisible_characters_are_stripped() -> None:
    """A right-to-left override would visually reorder every segment after it
    (alarms included) in a bidi-aware terminal; zero-width characters make the
    measured and painted widths disagree (review round 1, m3)."""
    raw = "abc\u202edef\u200bghi\u2066x\u2069\ufeffy\u2028z"
    assert sc.sanitize(raw) == "abcdefghixyz"
    # A DCS string is removed with its payload, not left behind as text.
    assert sc.sanitize("\x1bP1;2q payload\x1b\\ ok") == "ok"
    # An 8-bit C1 CSI takes its parameters with it (review round 2, N4).
    assert sc.sanitize("\x9b31mred") == "red"


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
