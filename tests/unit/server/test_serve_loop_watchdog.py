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


def _tick(
    guard: loop_watchdog.LoopWatchdog, clock: Clock, seconds: int, *, beat_every: int | None = None
) -> list[str | None]:
    """Advance one CHECK_S at a time; the loop beats only every ``beat_every`` ticks."""
    verdicts = []
    for step in range(1, seconds + 1):
        clock.now += 1.0
        if beat_every is not None and step % beat_every == 0:
            guard.last_beat = clock.now
        verdicts.append(guard.check())
    return verdicts


def test_recovery_needs_a_continuous_healthy_window(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Q-1: one beat does not end an episode; RECOVERY_S of healthy beats does."""
    guard, clock, _ = _watchdog(tmp_path)
    clock.now += 15.0
    assert guard.check() == "dump"
    # 29 healthy checks, then a new 11 s gap: still the SAME episode, no new dump.
    for _ in range(29):
        clock.now += 1.0
        guard.last_beat = clock.now
        assert guard.check() is None
    assert guard.in_episode
    clock.now += 11.0
    assert guard.check() is None
    assert guard.dumps == 1
    # 30 s continuously healthy closes it; the next stall dumps at once.
    for _ in range(31):
        clock.now += 1.0
        guard.last_beat = clock.now
        guard.check()
    assert not guard.in_episode
    assert "recovered after" in caplog.text
    clock.now += 11.0
    assert guard.check() == "dump"
    assert guard.dumps == 2


def test_hard_bound_requests_a_restart(tmp_path: Path) -> None:
    guard, clock, restarts = _watchdog(tmp_path, restart=lambda: None)
    clock.now += 120.0
    assert guard.check() == "dump"  # first sight of the gap: unconfirmed (R2)
    clock.now += 1.0
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


def test_blocker_clearing_lets_the_restart_through(tmp_path: Path) -> None:
    blocked = ["hub work is active"]
    guard, clock, restarts = _watchdog(
        tmp_path, restart_blocker=lambda: blocked[0] if blocked else None
    )
    clock.now += 125.0
    guard.check()
    _tick(guard, clock, 10)
    assert restarts == []
    blocked.clear()
    clock.now += 1.0
    guard.check()
    assert restarts == [1]


def test_failed_restart_is_attempted_once_per_episode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R1: a refused/failed exec must not re-dump and retry on every 1 s tick."""
    guard, clock, restarts = _watchdog(tmp_path)  # restart returns "stubbed exec"
    clock.now += 10.0
    assert guard.check() == "dump"
    verdicts = _tick(guard, clock, 390)  # a 400 s wedge, checked every second
    assert restarts == [1]
    # Dumps at 10 s, 70 s, the one "restarting" dump at 120 s, then the 60 s
    # cadence from there: 180, 240, 300, 360. Not one per tick.
    assert guard.dumps == 7
    assert verdicts.count("dump") == 5
    assert caplog.text.count("self-restart failed") == 1
    assert guard.dump_path.read_text().count("serve loop restarting") == 1


def test_reviewer_repro_ticks_125_to_127(tmp_path: Path) -> None:
    guard, clock, restarts = _watchdog(tmp_path)
    clock.now += 125.0
    guard.check()
    _tick(guard, clock, 2)
    assert restarts == [1]
    assert guard.dumps == 2  # the first sighting + the one "restarting" dump


def test_no_restart_callable_is_dump_only(tmp_path: Path) -> None:
    """R1: a process with no listener of its own registers no restart at all."""
    guard, clock, _ = _watchdog(tmp_path, restart=None)
    clock.now += 10.0
    guard.check()
    verdicts = _tick(guard, clock, 290)
    assert "restart" not in verdicts and "refused" not in verdicts
    assert guard.restarts_requested == 0
    assert guard.dumps == 5  # 10, 70, 130, 190, 250


def test_convoy_that_beats_now_and_then_still_restarts(tmp_path: Path) -> None:
    """Q-1 (QA Q1d): the loop beats every ~15 s, lag never reaches 120 s, /health
    is dead throughout. The old "any beat < 10 s resets" rule never restarted."""
    guard, clock, restarts = _watchdog(tmp_path)
    verdicts = _tick(guard, clock, 222, beat_every=15)
    assert guard.max_lag < 120.0
    assert restarts == [1]
    # Once the episode was >= 120 s old: 10 s (first crossing) + ~110 s.
    assert verdicts.index("refused") < 135


def test_busy_but_turning_loop_never_opens_an_episode(tmp_path: Path) -> None:
    """Q2 normal load: beats every check (even with ~0.5 s phase) never fire."""
    guard, clock, restarts = _watchdog(tmp_path)
    for _ in range(600):
        clock.now += 1.0
        guard.last_beat = clock.now - 0.5
        assert guard.check() is None
    assert not guard.in_episode and guard.dumps == 0 and restarts == []


def test_a_pause_then_resume_does_not_restart(tmp_path: Path) -> None:
    """R2: SIGSTOP/debugger past the bound, then a healthy loop: one sample only."""
    guard, clock, restarts = _watchdog(tmp_path, restart=lambda: restarts.append(1))
    clock.now += 300.0  # resumed; the stale stamp reads 300 s
    assert guard.check() == "dump"
    _tick(guard, clock, 60, beat_every=1)  # the loop beats as soon as it runs
    assert restarts == []
    assert not guard.in_episode


def test_restart_budget_switches_to_dump_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R3: 3 restarts in 30 min (carried across exec) means dump-only."""
    wall = [10_000.0]
    environ: dict[str, str] = {}
    history = loop_watchdog.read_restart_history(
        {loop_watchdog.RESTARTS_ENV: "8300,9000,9500,garbage,100"}, wall[0]
    )
    assert history == [8300.0, 9000.0, 9500.0]  # expired and unparsable dropped
    guard, clock, restarts = _watchdog(
        tmp_path, restart_history=history, environ=environ, wall_clock=lambda: wall[0]
    )
    clock.now += 125.0
    _tick(guard, clock, 300)
    assert restarts == []
    assert caplog.text.count("dump-only from now on") == 1
    # The oldest stamp ages out of the window: the next episode may restart again.
    wall[0] = 8300.0 + loop_watchdog.RESTART_WINDOW_S + 1
    guard2, clock2, restarts2 = _watchdog(
        tmp_path,
        restart_history=history,
        environ=environ,
        wall_clock=lambda: wall[0],
        restart=lambda: None,
    )
    clock2.now += 125.0
    _tick(guard2, clock2, 2)
    assert restarts2 == [] and guard2.restarts_requested == 1
    assert environ[loop_watchdog.RESTARTS_ENV].split(",") == ["9000", "9500", f"{wall[0]:.0f}"]


def test_failed_restart_spends_no_budget(tmp_path: Path) -> None:
    environ: dict[str, str] = {}
    guard, clock, restarts = _watchdog(tmp_path, environ=environ)  # "stubbed exec" failure
    clock.now += 125.0
    _tick(guard, clock, 2)
    assert restarts == [1]
    assert loop_watchdog.RESTARTS_ENV not in environ
    assert guard.restart_history == []


def test_failed_restart_leaves_no_budget_in_the_environment(tmp_path: Path) -> None:
    """Round 2 m1: the budget is in the environment only across the exec; a
    failed attempt with an earlier stamp must not leave it for children."""
    environ: dict[str, str] = {}
    seen: list[str | None] = []

    def failing_exec() -> str:
        seen.append(environ.get(loop_watchdog.RESTARTS_ENV))
        return "stubbed exec"

    guard, clock, _ = _watchdog(
        tmp_path,
        environ=environ,
        restart=failing_exec,
        restart_history=[999.0],
        wall_clock=lambda: 1000.0,
    )
    clock.now += 125.0
    _tick(guard, clock, 2)
    assert seen == ["999,1000"]  # set for the exec itself
    assert loop_watchdog.RESTARTS_ENV not in environ
    assert guard.restart_history == [999.0]


def test_sub_threshold_hiccups_do_not_confirm_a_restart(tmp_path: Path) -> None:
    """Round 2 m2: an episode opened by one blip and kept open by 1-9 s lags
    (/health still answers) never restarts; lags >= DUMP_LAG_S still do."""
    guard, clock, restarts = _watchdog(tmp_path)
    clock.now += 15.0
    assert guard.check() == "dump"  # the blip opens the episode
    for _ in range(300):
        clock.now += 1.0
        guard.last_beat = clock.now - 3.0  # turning, but never healthy for 30 s
        guard.check()
    assert guard.in_episode and restarts == []
    # Control: the same open episode, now with two lags past the threshold.
    guard.last_beat = clock.now - 10.0
    _tick(guard, clock, 1)
    assert restarts == []  # one confirming sample is not two
    _tick(guard, clock, 1)
    assert restarts == [1]


def test_dumps_back_off_after_the_per_episode_cap(tmp_path: Path) -> None:
    """R5: a stall held by a blocker for hours stops writing a dump a minute."""
    guard, clock, _ = _watchdog(tmp_path, restart_blocker=lambda: "hub work is active")
    clock.now += 10.0
    guard.check()
    _tick(guard, clock, 3 * 3600)
    # 10 at the 60 s cadence (10 s .. 550 s), then hourly after the 10th:
    # 4150 s and 7750 s inside the 10810 s run.
    assert guard.dumps == 10 + 2


def test_stall_log_size_cap(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    guard, clock, _ = _watchdog(tmp_path, max_log_bytes=1)
    clock.now += 10.0
    guard.check()  # writes the first dump into an empty file
    assert guard.dumps == 1
    _tick(guard, clock, 200)
    assert guard.dumps == 1
    assert caplog.text.count("cap; no further dumps") == 1


def test_prune_keeps_the_newest_stall_logs(tmp_path: Path) -> None:
    import os

    logs = tmp_path / "logs"
    logs.mkdir()
    for index in range(8):
        path = logs / f"serve-stall-{100 + index}.log"
        path.write_text("x")
        os.utime(path, (1000 + index, 1000 + index))
    (logs / "serve.log").write_text("unrelated")
    current = logs / "serve-stall-999.log"
    removed = loop_watchdog.prune_stall_logs(logs, keep=5, current=current)
    left = sorted(p.name for p in logs.iterdir())
    assert left == [
        "serve-stall-104.log",
        "serve-stall-105.log",
        "serve-stall-106.log",
        "serve-stall-107.log",
        "serve.log",
    ]
    assert len(removed) == 4


def test_empty_boot_logs_never_evict_a_real_dump(tmp_path: Path) -> None:
    """Round 2 M1: every boot creates an empty serve-stall-<pid>.log for the
    SIGUSR2 lever. Six ordinary boots after a wedge must not prune its dump."""
    import os

    logs = tmp_path / "logs"
    logs.mkdir()
    dump = logs / "serve-stall-100.log"
    dump.write_text("Thread 0x1 (most recent call first):\n  File wedge.py, line 1\n")
    os.utime(dump, (1000, 1000))
    for index, pid in enumerate(range(200, 206)):
        current = logs / f"serve-stall-{pid}.log"
        loop_watchdog.prune_stall_logs(logs, keep=5, current=current)
        current.touch()  # what register_dump_signal's append-open does
        os.utime(current, (2000 + index, 2000 + index))
    left = sorted(p.name for p in logs.iterdir())
    # The dump survives; earlier empty boot logs are gone; the current one stays.
    assert left == ["serve-stall-100.log", "serve-stall-205.log"]
    assert dump.read_text().startswith("Thread 0x1")


def test_prune_never_deletes_the_current_log_even_when_empty(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    current = logs / "serve-stall-7.log"
    current.touch()
    assert loop_watchdog.prune_stall_logs(logs, keep=1, current=current) == []
    assert current.exists()


def test_dump_signal_writes_all_threads_without_the_watchdog(tmp_path: Path) -> None:
    """R4: ``kill -USR2 <pid>`` dumps from C, the lever for a GIL-starved watchdog."""
    import faulthandler
    import os
    import signal

    if not hasattr(signal, "SIGUSR2") or not hasattr(faulthandler, "register"):
        pytest.skip("no SIGUSR2 / faulthandler.register on this platform")
    guard, _, _ = _watchdog(tmp_path)
    try:
        assert guard.register_dump_signal() == signal.SIGUSR2
        os.kill(os.getpid(), signal.SIGUSR2)
        time.sleep(0.2)
        guard.signal_handle.flush()
        text = guard.dump_path.read_text()
        assert "test_dump_signal_writes_all_threads_without_the_watchdog" in text
        assert os.getpid() > 0  # still alive: the default (terminate) is replaced
    finally:
        # Production never unregisters (stall_watchdog's measured rule); the test
        # process does, so the handler does not outlive tmp_path.
        faulthandler.unregister(signal.SIGUSR2)
        guard.signal_handle.close()


@pytest.mark.asyncio
async def test_install_without_a_listener_is_dump_only(tmp_path: Path) -> None:
    import faulthandler
    import signal

    app = SimpleNamespace(state=SimpleNamespace())
    guard, beat = loop_watchdog.install(app, tmp_path)
    try:
        assert guard._restart is None
        assert getattr(app.state, loop_watchdog.STATE_ATTR) is guard
    finally:
        guard.stop()
        beat.cancel()
        if guard.dump_signal is not None:
            faulthandler.unregister(signal.SIGUSR2)
            guard.signal_handle.close()


@pytest.mark.asyncio
async def test_install_adopts_the_budget_out_of_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 2 m1: a nested ``lop serve`` (any child) must not inherit a spent budget."""
    import faulthandler
    import os
    import signal
    import subprocess
    import sys

    monkeypatch.setenv(loop_watchdog.RESTARTS_ENV, f"{time.time() - 60:.0f}")
    app = SimpleNamespace(state=SimpleNamespace())
    guard, beat = loop_watchdog.install(app, tmp_path)
    try:
        assert len(guard.restart_history) == 1  # read before it was dropped
        assert loop_watchdog.RESTARTS_ENV not in os.environ
        probe = f"import os; print(os.environ.get({loop_watchdog.RESTARTS_ENV!r}))"
        child = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        assert child.stdout.strip() == "None"
    finally:
        guard.stop()
        beat.cancel()
        if guard.dump_signal is not None:
            faulthandler.unregister(signal.SIGUSR2)
            guard.signal_handle.close()


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
