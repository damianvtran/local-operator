"""The teardown's executor join is bounded, and the bound is a DIAGNOSTIC.

``main`` closes its loop with an explicit :func:`process._teardown_loop`, which
gives the shared default executor a bounded join (half the configured stall
bound, see ``_TEARDOWN_EXECUTOR_JOIN_FRACTION``) instead of CPython's own
300-second constant, and names the workers still holding the pool when that
bound expires. The two children here pin what that does and — just as
importantly — what it does NOT do, because the shape reads like a fix:

* **B1** parks one default-executor worker in a FIFO read with NO writer (a
  thread stopped in the kernel, never a sleep), drives the real teardown, and
  shows that the teardown returns and the process is STILL ALIVE afterwards:
  the interpreter's own exit joins non-daemon workers, so a timeout cannot force
  the join to return. What the operator gains is the early, NAMED reading.
* **B2** does the identical blocking call on a raw DAEMON thread: the child exits
  on its own and its sentinel is present, with no warning at all. That is the
  contrast that makes B1 a measurement rather than a rhetorical claim — and it
  is why the store-maintenance passes moved to a daemon thread in the first
  place.

Neither child ever touches the operator's store or config: the store here is one
FIFO inside the test's own ``tmp_path``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator.session.runtime import process as process_module
from local_operator.session.runtime import stall_watchdog

#: The stall bound the children derive their join bound from. Deliberately tiny
#: and set by PATCHING THE FUNCTION rather than the environment: the env knob's
#: floor is three heartbeat intervals (45 s), so a test that used it would have
#: to wait 22 s to see an expiry. The derivation itself is asserted against the
#: real watchdog in `test_the_join_bound_is_half_the_configured_stall_bound`.
_CHILD_STALL_BOUND_S = 4.0

_CHILD_PREAMBLE = r"""
import asyncio, os, sys
from pathlib import Path
from local_operator.session.runtime import process, stall_watchdog

fifo = sys.argv[1]
sentinel = Path(sys.argv[2])
os.mkfifo(fifo)
stall_watchdog.bound_seconds = lambda: {bound!r}

def park_forever():
    # A FIFO read with no writer: stopped in the kernel, so nothing can reclaim
    # this thread — not the join, not the interpreter's exit.
    with open(fifo, "r") as handle:
        handle.read(1)

async def idle():
    await asyncio.sleep(0.2)

def run_scenario(scenario):
    runner = asyncio.Runner()
    try:
        runner.run(scenario())
    finally:
        # THE PRODUCTION TEARDOWN, not a re-implementation of it.
        process._teardown_loop(runner)
"""


def _run_teardown_child(tmp_path: Path, body: str) -> subprocess.Popen[str]:
    script = _CHILD_PREAMBLE.format(bound=_CHILD_STALL_BOUND_S) + body
    env = os.environ.copy()
    env.pop("XPC_FLAGS", None)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path / "park.fifo"),
            str(tmp_path / "sentinel"),
        ],
        cwd=Path(__file__).parents[4],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_sentinel(proc: subprocess.Popen[str], sentinel: Path, *, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if sentinel.exists():
            return
        if proc.poll() is not None:
            break
        time.sleep(0.02)
    raise AssertionError(
        f"the child wrote no sentinel (returncode {proc.poll()}): "
        + (proc.communicate(timeout=10)[1] or "")[-4000:]
    )


def test_the_bounded_join_reports_the_worker_and_the_process_survives_it(
    tmp_path: Path,
) -> None:
    """B1: the bound expires, the stuck worker is NAMED, and the process lives on.

    The last clause is the honest half: a thread stopped in the kernel is joined
    by the interpreter's own exit whatever this code does, so nothing here is a
    bound on the process. The reading is early and attributable instead of an
    anonymous wedge reported by the stall bound after the turn is over.
    """
    sentinel = tmp_path / "sentinel"
    proc = _run_teardown_child(
        tmp_path,
        r"""
async def scenario():
    # THE POOL: one default-executor worker parked in the kernel.
    asyncio.get_running_loop().run_in_executor(None, park_forever)
    await idle()

run_scenario(scenario)
# Written AFTER the teardown returned, so the file is the proof that the
# ``asyncio.run``-equivalent teardown completed rather than a clock reading.
sentinel.write_text("teardown returned; worker still parked\n", encoding="utf-8")
""",
    )
    try:
        _wait_for_sentinel(proc, sentinel, seconds=60)
        # THE POINT: the teardown returned and this process is still alive,
        # because the interpreter's exit is now joining the parked worker.
        assert proc.poll() is None, "the process exited; the bound would be a real bound"
    finally:
        # Reap by exact pid — never by name; this process is ours alone.
        proc.kill()
        stdout, stderr = proc.communicate(timeout=30)

    output = stdout + stderr
    assert "the shared default executor did not join within 2s" in output, output[-4000:]
    assert "park_forever" in output, "the dump did not name the stuck worker"
    assert "every thread is dumped below" in output, output[-4000:]


def test_the_same_blocking_call_on_a_daemon_thread_lets_the_child_exit(
    tmp_path: Path,
) -> None:
    """B2: the same kernel-stopped call on a daemon thread does not hold anything.

    This is the falsifier for B1's claim. If the warning and the dump were just
    noise around a process that would have left anyway, this child would emit
    them too; instead the executor pool is empty, the bounded join returns
    immediately, and the process exits 0 with the daemon thread still parked.
    """
    sentinel = tmp_path / "sentinel"
    proc = _run_teardown_child(
        tmp_path,
        r"""
import threading

async def scenario():
    # THE SAME CALL, on a raw daemon thread instead of the shared pool.
    threading.Thread(target=park_forever, name="parked-daemon", daemon=True).start()
    await idle()

run_scenario(scenario)
sentinel.write_text("teardown returned; daemon parked\n", encoding="utf-8")
""",
    )
    try:
        stdout, stderr = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=30)

    assert proc.returncode == 0, stderr[-4000:]
    assert sentinel.read_text(encoding="utf-8").startswith("teardown returned")
    output = stdout + stderr
    assert "did not join within" not in output, output[-4000:]
    assert "Traceback" not in output


def test_the_join_bound_is_half_the_configured_stall_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is derived from the one the timer will use, and is strictly below it.

    Every host shape matters, not just the default: an operator who tightens the
    watchdog tightens the number this has to beat, and one who switches it off
    leaves no timer at all — the report still gets a deadline of its own.
    """
    monkeypatch.setattr(stall_watchdog, "bound_seconds", lambda: 300.0)
    assert process_module._teardown_executor_join_bound_s() == 150.0
    monkeypatch.setattr(stall_watchdog, "bound_seconds", lambda: 45.0)
    assert process_module._teardown_executor_join_bound_s() == 22.5
    monkeypatch.setattr(stall_watchdog, "bound_seconds", lambda: None)
    assert process_module._teardown_executor_join_bound_s() == 150.0


def test_the_real_join_bound_is_strictly_below_the_real_stall_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped default, not a patched function: 150 s against a 300 s fire."""
    monkeypatch.delenv(stall_watchdog.ENV_SECONDS, raising=False)
    bound = stall_watchdog.bound_seconds()
    # Narrowed for the comparison below: ``bound_seconds`` returns ``None`` when
    # the watchdog is switched off, and the assertion is about the armed case.
    assert bound is not None and bound == 300.0
    assert process_module._teardown_executor_join_bound_s() < bound
