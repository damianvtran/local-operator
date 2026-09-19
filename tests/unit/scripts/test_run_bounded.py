"""The bounded gate wrapper: a bound that leaves a gigabyte running is not a bound.

WHY THESE EXIST
---------------
`pyright` is a Python wrapper around an npm/node analyzer, and the local gate ran
it as `timeout 900 .venv/bin/python -m pyright …`. `timeout(1)` signals only the
process it started, so when the bound fired the analyzer kept running as an
orphan (measured on this host: `ppid 1`, 2.28 GB and 1.50 GB RSS, one alive 81
minutes after its parent died). Sessions then queue more analyzers behind a host
those orphans are saturating, more of them hit the bound, and the leak compounds.

`scripts/run_bounded.py` exists to make that impossible, and these tests drive it
as a real process — the defect IS a process-tree behaviour, so no amount of
mocking would be evidence. Each test plants a grandchild whose command line
carries a unique token, then asserts on what is alive afterwards, which is the
same instrument the field observation used.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, Sequence

import pytest

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts" / "run_bounded.py"

#: These spawn real processes and wait on real signals; the suite marks that
#: shape `slow` (see pyproject.toml's marker list).
pytestmark = pytest.mark.slow


def _marker(token: str) -> str:
    """A token that appears ONLY in the spawned tree's command lines."""
    return f"run-bounded-test-{token}-{uuid.uuid4().hex}"


def _pids_carrying(token: str) -> list[int]:
    """Every process whose command line carries the token, including orphans.

    `pgrep -f` is the same instrument the leak was observed with, and it sees
    re-parented processes that a `ps --ppid` walk would miss.
    """
    result = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    return [int(line) for line in result.stdout.split() if line.strip()]


def _wait_until_gone(token: str, timeout: float = 15.0) -> list[int]:
    """The pids still carrying the token once the tree should be dead."""
    deadline = time.monotonic() + timeout
    remaining = _pids_carrying(token)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.1)
        remaining = _pids_carrying(token)
    return remaining


@pytest.fixture
def reap_markers() -> Iterator[list[str]]:
    """Kill anything a failing test leaves behind, by exact pid, and say so.

    A test for a leak must not become one: the operator's screen is shared with
    two dozen sessions, and an orphaned sleeper from a red test would outlive it.
    """
    tokens: list[str] = []
    try:
        yield tokens
    finally:
        for token in tokens:
            for pid in _pids_carrying(token):
                with pytest.MonkeyPatch.context() as guard:
                    guard.setattr(os, "getpid", lambda: -1)  # never skip our own
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def _wrapper(args: Sequence[str], *, timeout_s: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WRAPPER), *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _spawning_inner(token: str, *, linger: bool) -> list[str]:
    """A command that plants one grandchild carrying `token`, then lingers.

    The grandchild is a plain sleeper: it exists so that a wrapper which kills
    only its direct child leaves a visible orphan, which is the whole defect.
    """
    plant = (
        "import subprocess, sys;"
        f" subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)', {token!r}])"
    )
    tail = "import time; time.sleep(600)" if linger else "pass"
    return [sys.executable, "-c", f"{plant}; {tail}"]


def test_a_command_runs_with_its_own_status_and_untouched_stdout(reap_markers):
    result = _wrapper(["--", sys.executable, "-c", "print('gate output')"])

    assert result.returncode == 0
    assert result.stdout == "gate output\n", "the gate's stdout must pass through"
    assert "[run_bounded]" in result.stderr, "the wrapper must say what it did"
    assert "[run_bounded]" not in result.stdout, (
        "diagnostics on stdout would pollute a parses-able gate output (the "
        "same rule the pytest worker-cap line follows)"
    )


def test_a_command_that_cannot_start_is_not_a_pass(reap_markers):
    """A gate that never ran must never report success — the #423 shape."""
    result = _wrapper(["--timeout", "5", "--", "/nonexistent/interpreter-xyz"])

    assert result.returncode == 125
    assert "could not start" in result.stderr


def test_a_bound_kills_the_whole_group_not_just_the_leader(reap_markers):
    """The field defect, reproduced: the leader dies, the work does not.

    The wrapper's own child is the node analyzer in production and a sleeper
    here; the token-carrying grandchild stands in for it because the point is
    the process group, not pyright.
    """
    token = _marker("timeout")
    reap_markers.append(token)

    inner = _spawning_inner(token, linger=True)
    result = _wrapper(["--timeout", "1", "--grace", "3", "--", *inner])

    assert result.returncode == 124, "a fired bound reports timeout(1)'s status"
    assert "timeout after 1s" in result.stderr
    assert _wait_until_gone(token) == [], "the group outlived the bound"


def test_a_group_that_outlives_its_leader_is_reaped_with_no_bound_at_all(reap_markers):
    """The half `timeout(1)` cannot cover: an ordinary exit that leaves work."""
    token = _marker("early-exit")
    reap_markers.append(token)

    result = _wrapper(["--", *_spawning_inner(token, linger=False)])

    assert result.returncode == 0
    assert _wait_until_gone(token) == [], "the leader exited and left its children running"


def test_a_signal_to_the_wrapper_reaches_the_group(reap_markers):
    """Ctrl-C and `kill` must not leave the analyzer running either.

    The child leads its own session, so the terminal's SIGINT goes to the
    wrapper alone; an unforwarded signal is the same leak by another road.
    """
    token = _marker("signal")
    reap_markers.append(token)
    wrapper = subprocess.Popen(
        [sys.executable, str(WRAPPER), "--", *_spawning_inner(token, linger=True)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for the grandchild to exist before signalling, so the signal cannot
    # win a race against the spawn and pass vacuously.
    deadline = time.monotonic() + 15.0
    while not _pids_carrying(token) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _pids_carrying(token), "the grandchild never started"

    wrapper.send_signal(signal.SIGTERM)
    try:
        returncode = wrapper.wait(timeout=30)
    finally:
        if wrapper.poll() is None:  # pragma: no cover - only on a wedged wrapper
            wrapper.kill()
    stderr = wrapper.stderr.read() if wrapper.stderr else ""

    assert returncode == 128 + signal.SIGTERM, f"rc={returncode}, stderr={stderr!r}"
    assert "signal 15" in stderr
    assert _wait_until_gone(token) == [], "a signalled wrapper left its group running"
