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
    stop_every_borrower_revoke,
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


def test_the_package_backstop_cancels_a_belt_armed_without_the_by_import_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run 44408211857: a module that never imported the guard left a 600 s belt
    that fired in ``test_cli_remote``. The conftest backstop uses only product
    handles, so it must stop a belt it never recorded."""
    fired: list[str] = []
    monkeypatch.setattr(github_mod, "revoke_installation_token", fired.append)
    # The real module, not the recording wrapper: what an unguarded module arms.
    monkeypatch.setattr(github_mod, "threading", threading)
    github_mod._schedule_self_revoke("tok-unguarded", int(time.time() * 1000) + 600_000)
    assert [t.name for t in github_mod.pending_self_revoke_threads()], "belt not armed"

    stop_every_borrower_revoke()

    assert github_mod.pending_self_revoke_threads() == []
    assert "tok-unguarded" not in github_mod._SELF_REVOKE_SCHEDULED
    assert fired == [], "a cancelled belt must never DELETE"


def test_a_belt_already_firing_is_joined_not_just_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cancel()`` is a no-op on a running fire; the join is what keeps its
    DELETE inside the cell that armed it."""
    started = threading.Event()
    done: list[str] = []

    def slow_revoke(token: str) -> bool:
        started.set()
        time.sleep(0.3)
        done.append(token)
        return True

    monkeypatch.setattr(github_mod, "revoke_installation_token", slow_revoke)
    github_mod._schedule_self_revoke("tok-firing", 0)  # the 0.1 s floor
    assert started.wait(10.0), "the belt never fired"

    stop_every_borrower_revoke()

    assert done == ["tok-firing"], "teardown returned while the DELETE was in flight"
    assert github_mod.pending_self_revoke_threads() == []


def test_a_revoke_now_thread_that_outlives_its_join_fails_teardown_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    stuck = threading.Thread(
        target=release.wait, name=github_mod.REVOKE_NOW_THREAD_NAME, daemon=True
    )
    stuck.start()
    try:
        with pytest.raises(AssertionError, match=github_mod.REVOKE_NOW_THREAD_NAME):
            stop_every_borrower_revoke(join_timeout_s=0.05)
    finally:
        release.set()
        stuck.join(10.0)
    assert not stuck.is_alive()
