"""The serve event-loop watchdog: dump on a stall, restart past the hard bound.

Regression tests for the 2026-11-11 serve wedges (pids 11196, 9633): a stalled
daemon must (1) write an all-thread traceback naming the frame that holds it,
(2) re-exec itself past the hard bound unless a reload or unrecoverable work
owns the moment, and (3) never fire on a loop that is merely busy.

``check`` is driven with an explicit clock so each threshold is asserted at its
edge without sleeping; one test runs the real thread against a real loop
blocked by a real GIL-holding pure-Python loop on an executor thread, the shape
``sample 9633`` recorded.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.server import loop_watchdog
from local_operator.server import reload as serve_reload


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _watchdog(tmp_path: Path, **kwargs: Any) -> tuple[loop_watchdog.LoopWatchdog, Clock, list[int]]:
    clock = Clock()
    restarts: list[int] = []
    options: dict[str, Any] = {
        "dump_path": tmp_path / "logs" / "serve-stall-1.log",
        "restart": lambda: restarts.append(1) or "stubbed exec",
        "dump_lag_s": 10.0,
        "restart_lag_s": 120.0,
        "redump_s": 60.0,
        "clock": clock,
    }
    options.update(kwargs)
    return loop_watchdog.LoopWatchdog(**options), clock, restarts


def test_no_fire_under_normal_lag(tmp_path: Path) -> None:
    """A busy-but-turning loop never dumps: lag just under the bound is silent."""
    guard, clock, restarts = _watchdog(tmp_path)
    for _ in range(50):
        clock.now += 9.9
        assert guard.check() is None
        guard.last_beat = clock.now  # the loop beat again
    assert guard.dumps == 0
    assert restarts == []
    assert not guard.dump_path.exists()


def test_lag_past_threshold_writes_an_all_thread_dump(tmp_path: Path) -> None:
    guard, clock, restarts = _watchdog(tmp_path)
    clock.now += 10.0
    assert guard.check() == "dump"
    text = guard.dump_path.read_text()
    assert "serve loop stalled: no heartbeat for 10.0s" in text
    # faulthandler names every thread, and this one is in this very test.
    assert "test_lag_past_threshold_writes_an_all_thread_dump" in text
    assert restarts == []
    # One dump per stall, not one per check ...
    clock.now += 1.0
    assert guard.check() is None
    assert guard.dumps == 1
    # ... plus one more per REDUMP_S while it persists.
    clock.now += 60.0
    assert guard.check() == "dump"
    assert guard.dumps == 2


def test_recovery_resets_the_episode(tmp_path: Path) -> None:
    guard, clock, _ = _watchdog(tmp_path)
    clock.now += 15.0
    assert guard.check() == "dump"
    guard.last_beat = clock.now
    assert guard.check() is None
    clock.now += 11.0
    assert guard.check() == "dump"  # a new stall dumps at once again
    assert guard.dumps == 2


def test_hard_bound_requests_a_restart(tmp_path: Path) -> None:
    guard, clock, restarts = _watchdog(tmp_path, restart=lambda: None)
    clock.now += 120.0
    assert guard.check() == "restart"
    assert guard.restarts_requested == 1
    assert "serve loop restarting" in guard.dump_path.read_text()


def test_restart_withheld_while_a_blocker_owns_the_moment(tmp_path: Path) -> None:
    """A reload mid-drain or unrecoverable work keeps the daemon; dumps continue."""
    guard, clock, restarts = _watchdog(tmp_path, restart_blocker=lambda: "1 scheduler-owned run(s)")
    clock.now += 130.0
    assert guard.check() == "dump"
    assert restarts == []
    clock.now += 1.0
    assert guard.check() == "refused"


def test_failed_restart_falls_back_to_dumping(tmp_path: Path) -> None:
    guard, clock, restarts = _watchdog(tmp_path)  # restart returns "stubbed exec"
    clock.now += 125.0
    assert guard.check() == "dump"
    assert restarts == [1]


@pytest.mark.asyncio
async def test_real_thread_names_the_gil_holding_executor_frame(tmp_path: Path) -> None:
    """End to end in one process: real heartbeat, real thread, real executor hog.

    The loop is blocked the way 9633's was from the client's side (a synchronous
    stretch the heartbeat cannot interrupt), while an executor thread runs a
    pure-Python tuple loop; the dump must name that thread's function.
    """
    restarted = threading.Event()
    guard = loop_watchdog.LoopWatchdog(
        dump_path=tmp_path / "serve-stall.log",
        restart=lambda: restarted.set(),
        dump_lag_s=0.6,
        restart_lag_s=1.8,
    )
    original = loop_watchdog.CHECK_S
    loop_watchdog.CHECK_S = 0.1
    try:
        beat = asyncio.get_running_loop().create_task(guard.heartbeat())
        await asyncio.sleep(0.05)
        guard.start()
        stop = threading.Event()

        def hog_tuple_walk() -> None:
            items = tuple(range(1000))
            while not stop.is_set():
                for _ in items:
                    pass

        hog = asyncio.get_running_loop().run_in_executor(None, hog_tuple_walk)
        time.sleep(2.5)  # the loop thread is now not beating
        stop.set()
        await hog
        guard.stop()
        beat.cancel()
    finally:
        loop_watchdog.CHECK_S = original
    text = guard.dump_path.read_text()
    assert "hog_tuple_walk" in text
    assert restarted.is_set()


def test_restart_blocker_defers_to_a_pending_reload() -> None:
    app = SimpleNamespace(state=SimpleNamespace(**{serve_reload.PENDING_ATTR: True}))
    assert "reload" in (serve_reload.restart_blocker(app) or "")
