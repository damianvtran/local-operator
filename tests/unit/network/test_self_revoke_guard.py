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
from types import SimpleNamespace

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
    # Through the real revoke_now, not a hand-named thread: the teardown reads
    # the module's own registry, so only a thread revoke_now started is tracked.
    monkeypatch.setattr(github_mod, "revoke_installation_token", lambda *_a, **_k: release.wait())
    [stuck] = github_mod.revoke_now([_installation_grant("tok-stuck-revoke-now")])
    try:
        with pytest.raises(AssertionError, match=github_mod.REVOKE_NOW_THREAD_NAME):
            stop_every_borrower_revoke(join_timeout_s=0.05)
    finally:
        release.set()
        stuck.join(10.0)
    assert not stuck.is_alive()


def test_a_firing_belt_that_outlives_its_join_fails_teardown_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #4263 R1/Q1: a belt already inside its fire ignores ``cancel()``. If the
    cancel also dropped it from the registry, the join could expire on it while
    :func:`pending_self_revoke_threads` no longer saw it, the guard passed, and
    the DELETE landed in a later cell (QA: a 40 s hang hit ``test_cli_remote``'s
    tripwire). The overrun must fail the arming cell and name the timer."""
    started = threading.Event()
    release = threading.Event()

    def blocking_revoke(token: str) -> bool:
        started.set()
        release.wait(30.0)
        return True

    monkeypatch.setattr(github_mod, "revoke_installation_token", blocking_revoke)
    # The real module, so the belt is invisible to the by-import recorder: only
    # the product registry can make the guard see it.
    monkeypatch.setattr(github_mod, "threading", threading)
    github_mod._schedule_self_revoke("tok-stuck-firing", 0)  # noqa: SLF001 — the 0.1 s floor
    assert started.wait(10.0), "the belt never fired"
    belts = [t for t in github_mod.pending_self_revoke_threads() if isinstance(t, threading.Timer)]
    assert len(belts) == 1, belts
    try:
        with pytest.raises(AssertionError, match=github_mod.SELF_REVOKE_BELT_THREAD_NAME):
            stop_every_borrower_revoke(join_timeout_s=0.05)
    finally:
        release.set()
        belts[0].join(10.0)
    assert not belts[0].is_alive()
    assert github_mod.pending_self_revoke_threads() == []
    assert "tok-stuck-firing" not in github_mod._SELF_REVOKE_TIMERS  # noqa: SLF001


def _installation_grant(token: str) -> SimpleNamespace:
    """The three fields :func:`github_mod.is_installation_grant` reads, nothing more."""
    return SimpleNamespace(
        access_token=token,
        token_expires_at_ms=int(time.time() * 1000) + 3_600_000,
        credential_ref=SimpleNamespace(kind=github_mod.GITHUB_KIND),
    )


def test_teardown_with_nothing_scheduled_never_samples_threading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tests/unit/conftest.py`` runs this teardown after EVERY unit cell, so it
    must never scan the process's threads: the thread-leak reporter's own tests
    monkeypatch ``threading.enumerate``/``active_count`` to fail, and the PR #4263
    head that enumerated broke them on merge gate 44481472281 (shard 4, 3.12 and
    3.13). With no revoke ever scheduled the teardown reads empty registries only."""

    def forbidden(*_args: object) -> object:
        pytest.fail("the per-cell revoke teardown must not sample threading")

    monkeypatch.setattr(threading, "enumerate", forbidden)
    monkeypatch.setattr(threading, "active_count", forbidden)
    stop_every_borrower_revoke()
    assert github_mod.pending_self_revoke_threads() == []


def test_a_revoke_now_thread_is_tracked_until_it_finishes_then_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry that replaced the thread scan still sees an in-flight
    :func:`github_mod.revoke_now` DELETE (so the teardown joins it), and a
    foreign thread that merely shares the name is not mistaken for one."""
    release = threading.Event()
    monkeypatch.setattr(github_mod, "revoke_installation_token", lambda *_a, **_k: release.wait())
    impostor = threading.Thread(
        target=release.wait, name=github_mod.REVOKE_NOW_THREAD_NAME, daemon=True
    )
    impostor.start()
    try:
        [thread] = github_mod.revoke_now([_installation_grant("tok-tracked-revoke-now")])
        assert github_mod.pending_self_revoke_threads() == [thread]
    finally:
        release.set()
    stop_every_borrower_revoke()
    impostor.join(10.0)
    assert not thread.is_alive()
    assert github_mod.pending_self_revoke_threads() == []
