"""The shared self-revoke teardown guard (``self_revoke_guard``) does what its callers rely on.

WHY THESE CELLS. The guard's failure mode is silent: a join that runs out
returns ``None`` exactly like one that succeeded, so without an assertion a
timer could keep living into the next cell while every guarded cell stays
green — the leak the guard exists to stop. These cells pin both halves: a
recorded, unfired timer is cancelled and never fires, and a timer that outlives
its join fails teardown naming itself. No cell here reaches the network: the
scheduled timer's window is far in the future and is cancelled before it ends.
"""

from __future__ import annotations

import threading
import time

import pytest

from local_operator.network.credentials import github as github_mod
from tests.unit.network.self_revoke_guard import (  # noqa: F401 — autouse fixture by import
    _TimerRecordingThreading,
    no_self_revoke_outlives_its_cell,
    stop_recorded_timers,
)


def test_a_scheduled_self_revoke_is_recorded_cancelled_and_never_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fired: list[str] = []
    monkeypatch.setattr(
        github_mod, "revoke_installation_token", lambda token, **_: fired.append(token)
    )
    timers: list[threading.Timer] = []
    monkeypatch.setattr(github_mod, "threading", _TimerRecordingThreading(threading, timers))

    github_mod._schedule_self_revoke("tok_guard", int((time.time() + 600) * 1000))  # noqa: SLF001

    assert len(timers) == 1 and timers[0].is_alive(), timers
    stop_recorded_timers(timers)
    assert not timers[0].is_alive()
    assert fired == [], "a cancelled self-revoke must never DELETE"


def test_a_timer_that_outlives_its_join_fails_teardown_by_name() -> None:
    """R2: a fire already running ignores ``cancel()``; when the bounded join
    expires on it, the guard must say so instead of returning quietly."""
    started = threading.Event()
    release = threading.Event()

    def _stuck_fire() -> None:
        started.set()
        release.wait()

    timer = threading.Timer(0.0, _stuck_fire)
    timer.daemon = True
    timer.start()
    try:
        assert started.wait(10.0), "the timer never started its fire"
        with pytest.raises(AssertionError, match=r"still alive .*_stuck_fire"):
            stop_recorded_timers([timer], join_timeout_s=0.05)
    finally:
        release.set()
        timer.join(10.0)
    assert not timer.is_alive()
