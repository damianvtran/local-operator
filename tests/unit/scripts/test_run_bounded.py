"""The bounded gate wrapper: a bound that leaves a gigabyte running is not a bound.

WHY THESE EXIST
---------------
`pyright` is a Python wrapper around an npm/node analyzer, and the local gate ran
it as `timeout 900 .venv/bin/python -m pyright …`. The fleet showed analyzers
re-parented to launchd and holding their heap (`ppid 1`, 2.28 GB and 1.50 GB RSS,
one alive 81 minutes after its parent died), which is the leak this wrapper
removes. The tidy explanation — `timeout` signals only the wrapper, node
survives — is NOT what this host's GNU coreutils does (it signals the child's
process group, and no `timeout` spelling here left anything behind), so the
wrapper is justified by the cases that ARE reproducible: a descendant alive when
the leader exits by itself, and a group that ignores SIGTERM where a bare
`timeout` wedges. See the module docstring for the full statement of what is
measured and what is only observed.

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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pytest

REPO = Path(__file__).resolve().parents[3]
WRAPPER = REPO / "scripts" / "run_bounded.py"

#: These spawn real processes and wait on real signals; the suite marks that
#: shape `slow` (see pyproject.toml's marker list).
pytestmark = pytest.mark.slow


def _marker(token: str) -> str:
    """A token that appears ONLY in the spawned tree's command lines."""
    return f"run-bounded-test-{token}-{uuid.uuid4().hex}"


def _pids_carrying(token: str, *, exclude: Iterable[int] = ()) -> list[int]:
    """Every process whose command line carries the token, including orphans.

    `pgrep -f` is the same instrument the leak was observed with, and it sees
    re-parented processes that a `ps --ppid` walk would miss. `exclude` exists
    because the WRAPPER's own command line contains the whole inner command — and
    so the token — which otherwise satisfies a "has it started yet?" wait before
    the wrapper has done anything. Measured on the Linux CI leg: that wait passed,
    the test signalled a wrapper still installing its handlers, and the wrapper
    died by signal (-15) with nothing on stderr.
    """
    result = subprocess.run(["pgrep", "-f", token], capture_output=True, text=True, check=False)
    skip = set(exclude)
    return [
        pid
        for pid in (int(line) for line in result.stdout.split() if line.strip())
        if pid not in skip
    ]


def _wait_for_pid_carrying(
    token: str, *, exclude: Iterable[int], timeout: float = 20.0
) -> list[int]:
    """Token-carrying pids other than `exclude`, once one exists."""
    deadline = time.monotonic() + timeout
    found = _pids_carrying(token, exclude=exclude)
    while not found and time.monotonic() < deadline:
        time.sleep(0.1)
        found = _pids_carrying(token, exclude=exclude)
    return found


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
    """Run the wrapper with its output in FILES, not pipes.

    `capture_output=True` reads until EOF, and EOF needs every holder of the pipe
    write end to be gone — including a descendant the sweep is in the middle of
    reaping. That couples the assertion to pipe lifetimes rather than to the
    behaviour under test, which is whether the process GROUP survived (measured:
    it made this test hang for 60 s on one run whose group reap had raced). A
    file is read after the fact and cannot block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "stdout"
        err_path = Path(tmp) / "stderr"
        with out_path.open("w") as out, err_path.open("w") as err:
            result = subprocess.run(
                [sys.executable, str(WRAPPER), *args],
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        return subprocess.CompletedProcess(
            result.args, result.returncode, out_path.read_text(), err_path.read_text()
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


def test_handlers_are_installed_before_the_child_is_spawned():
    """A signal in that window used to kill the wrapper with nothing reaped.

    Measured on the Linux CI leg: the test signalled a wrapper whose handlers
    were not yet installed, and it died by signal (`rc=-15`) with an EMPTY stderr
    — a wrapper that could not report, let alone reap the group. Structural
    rather than behavioural, because the window is a few milliseconds wide and
    cannot be hit on demand.
    """
    source = (REPO / "scripts" / "run_bounded.py").read_text()
    body = source[source.index("def run(") : source.index("def main(")]

    assert "signal.signal(" in body, "the wrapper installs no signal handler at all"
    assert body.index("signal.signal(") < body.index("subprocess.Popen("), (
        "the signal handlers must be installed BEFORE the child is spawned, or a "
        "signal in that window kills the wrapper and nothing forwards it"
    )


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


@pytest.mark.parametrize("forwarded", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_to_the_wrapper_reaches_the_group(reap_markers, forwarded):
    """Ctrl-C, `kill` and SIGHUP must not leave the analyzer running either.

    The child leads its own session, so the terminal's SIGINT goes to the
    wrapper alone; an unforwarded signal is the same leak by another road.
    SIGHUP is the other common route — a terminal, tmux or ssh session going
    away — and only `SIGINT`/`SIGTERM` used to be forwarded, which was measured
    leaving the group alive at `ppid 1` while the wrapper died (review MINOR 3
    on #1322). The test is parametrised rather than duplicated so a THIRD signal
    is one list entry, and the shape is asserted for each.
    """
    token = _marker("signal")
    reap_markers.append(token)
    # Files, not pipes: see `_wrapper` for why a pipe turns a raced reap into a
    # 60-second hang rather than a readable failure.
    with tempfile.TemporaryDirectory() as tmp:
        err_path = Path(tmp) / "stderr"
        with err_path.open("w") as err:
            wrapper = subprocess.Popen(
                [sys.executable, str(WRAPPER), "--", *_spawning_inner(token, linger=True)],
                stdout=subprocess.DEVNULL,
                stderr=err,
                text=True,
            )
            # Wait for the inner command (or its grandchild) to exist before
            # signalling: the wrapper's OWN command line carries the token too, so
            # it is excluded or the wait passes while the wrapper is still
            # installing its handlers.
            assert _wait_for_pid_carrying(
                token, exclude={wrapper.pid}
            ), "the inner command never started"
            wrapper.send_signal(forwarded)
            try:
                returncode = wrapper.wait(timeout=30)
            finally:
                if wrapper.poll() is None:  # pragma: no cover - wedged wrapper
                    wrapper.kill()
        stderr = err_path.read_text()

    assert returncode == 128 + forwarded, f"rc={returncode}, stderr={stderr!r}"
    assert f"signal {forwarded}" in stderr
    assert _wait_until_gone(token) == [], "a signalled wrapper left its group running"


def test_a_child_killed_by_someone_else_reports_128_plus_the_signal(reap_markers):
    """A signal this wrapper did not send must still exit in its contract.

    A `kill -9` by hand, the OOM killer or a job runner ends the child without
    the wrapper forwarding anything, and the status then arrives as `-9`, which
    `sys.exit` renders as 247 while the diagnostic line reads `rc=-9` — neither
    is in the documented EXIT STATUS set, and 247 in a log explains nothing. A
    shell reports 128 + the signal for a child killed this way, so this does too
    (review MINOR 4 on #1322).
    """
    token = _marker("external")
    reap_markers.append(token)
    with tempfile.TemporaryDirectory() as tmp:
        err_path = Path(tmp) / "stderr"
        with err_path.open("w") as err:
            wrapper = subprocess.Popen(
                [sys.executable, str(WRAPPER), "--", *_spawning_inner(token, linger=True)],
                stdout=subprocess.DEVNULL,
                stderr=err,
                text=True,
            )
            victims = _wait_for_pid_carrying(token, exclude={wrapper.pid})
            assert victims, "the inner command never started"
            # EVERY carrier, so the kill cannot land on the grandchild while the
            # leader this wrapper waits on lives on — that would make the test
            # pass on a timeout instead of on the status it is asserting.
            for pid in victims:
                os.kill(pid, signal.SIGKILL)
            try:
                returncode = wrapper.wait(timeout=30)
            finally:
                if wrapper.poll() is None:  # pragma: no cover - wedged wrapper
                    wrapper.kill()
        stderr = err_path.read_text()

    assert returncode == 128 + signal.SIGKILL, f"rc={returncode}, stderr={stderr!r}"
    assert "killed by signal 9" in stderr
