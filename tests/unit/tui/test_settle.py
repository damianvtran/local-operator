"""Contract tests for ``tests.unit.tui.waiting.settle`` on a tiny synthetic App.

Each case pins one join the helper promises (worker, thread hop, after-refresh
queue) or one bound (turn cap, deadline). The ``time.sleep`` calls run inside
the synthetic workers' threads: they simulate the off-loop work under test, and
no test-side wait depends on them.
"""

from __future__ import annotations

import asyncio
import gc
import itertools
import logging
import time

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tests.unit.tui.waiting import settle, track_threads


class StarvedPilot:
    """A pilot whose pause drains nothing: the starved-scheduler case in-test.

    ``Pilot.pause`` waits for the screen and an idle probe, which on a healthy
    box already drains the after-refresh queue and work queued before it. Swapping in
    ``sleep(0)`` leaves only settle's own joins, so a test that must prove a
    join (not a lucky pause) did the work fails without one.
    """

    def __init__(self, pilot) -> None:
        self.app = pilot.app

    async def pause(self, delay=None) -> None:
        await asyncio.sleep(0)


class Probe(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.threaded = False
        self.after_refresh = False
        self.release = None

    def compose(self) -> ComposeResult:
        yield Static("probe")

    async def thread_hop(self) -> None:
        await asyncio.to_thread(time.sleep, 0.3)
        self.threaded = True

    async def wedged(self) -> None:
        # Never completes: the deadline, not the turn cap, must end the wait.
        await asyncio.Event().wait()

    async def broken(self) -> None:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_a_worker_group_is_joined_through_its_thread_hop():
    app = Probe()
    async with app.run_test() as pilot:
        app.run_worker(app.thread_hop(), group="probe")
        # max_turns=1: only the join, not repeated polling, can make this pass.
        await settle(
            pilot, lambda: app.threaded, what="thread hop", workers=("probe",), max_turns=1
        )
        assert app.threaded


@pytest.mark.asyncio
async def test_a_tracked_raw_to_thread_is_joined(monkeypatch):
    app = Probe()
    threads = track_threads(monkeypatch)
    async with app.run_test() as pilot:
        # No worker owns this task: only the thread registry can see the hop.
        task = asyncio.ensure_future(app.thread_hop())
        await pilot.pause()  # let the task reach its to_thread call
        await settle(pilot, lambda: app.threaded, what="raw hop", joins=(threads,), max_turns=1)
        assert app.threaded
        await task


@pytest.mark.asyncio
async def test_call_after_refresh_is_drained():
    app = Probe()
    async with app.run_test() as pilot:
        # StarvedPilot: a real pause drains this queue on its own, so only with
        # the pause starved does the one allowed turn prove the refresh join.
        app.screen.call_after_refresh(setattr, app, "after_refresh", True)
        await settle(
            StarvedPilot(pilot), lambda: app.after_refresh, what="after refresh", max_turns=1
        )
        assert app.after_refresh


@pytest.mark.asyncio
async def test_a_never_true_predicate_raises_at_the_turn_cap_naming_what():
    app = Probe()
    async with app.run_test() as pilot:
        with pytest.raises(AssertionError) as raised:
            await settle(pilot, lambda: False, what="the impossible", max_turns=5)
    message = str(raised.value)
    assert "the impossible" in message
    assert "turn cap 5" in message
    assert "after 5 turn(s)" in message


@pytest.mark.asyncio
async def test_a_wedged_worker_raises_at_the_deadline_and_is_named():
    app = Probe()
    async with app.run_test() as pilot:
        app.run_worker(app.wedged(), group="stuck", name="wedge")
        started = time.monotonic()
        with pytest.raises(AssertionError) as raised:
            await settle(
                pilot, lambda: True, what="wedged join", workers=("stuck",), deadline_s=0.5
            )
        elapsed = time.monotonic() - started
    message = str(raised.value)
    assert "wedged join" in message
    assert "deadline 0.5s" in message
    assert "stuck/wedge=RUNNING" in message
    # The deadline bounds the wedged join itself, not just the loop around it.
    assert elapsed < 5, elapsed


@pytest.mark.asyncio
async def test_a_failed_worker_re_raises_naming_what():
    app = Probe()
    async with app.run_test() as pilot:
        app.run_worker(app.broken(), group="bad", exit_on_error=False)
        with pytest.raises(AssertionError, match="broken load: worker bad/.* failed: RuntimeError"):
            await settle(pilot, lambda: True, what="broken load", workers=("bad",))


@pytest.mark.asyncio
async def test_a_cancelled_worker_is_ignored():
    app = Probe()
    async with app.run_test() as pilot:
        app.run_worker(app.wedged(), group="swap", exclusive=True)
        await pilot.pause()  # let it start, so the replacement really cancels it
        # exclusive=True cancels the first: the helper must not treat that as failure.
        app.run_worker(app.thread_hop(), group="swap", exclusive=True)
        await settle(pilot, lambda: app.threaded, what="replaced worker", workers=("swap",))
        assert app.threaded


@pytest.mark.asyncio
async def test_no_turn_cap_by_default_so_a_starved_cascade_still_settles():
    # Q-2 on PR #3732: under a starved pause 300 turns took 0.20-0.49 s, so a
    # default turn cap failed cascades that were still progressing. Only the
    # wall deadline bounds by default.
    app = Probe()
    async with app.run_test() as pilot:
        checks = itertools.count(1)
        await settle(
            StarvedPilot(pilot), lambda: next(checks) >= 400, what="a long cascade", refresh=False
        )


@pytest.mark.asyncio
async def test_a_deadline_on_a_worker_leaves_no_unretrieved_task(caplog):
    # R1/Q-3 on PR #3732: the shielded Worker.wait() outlives the deadline and
    # ends WorkerCancelled when the app tears the worker down; unretrieved, asyncio
    # logs "Task exception was never retrieved" from the task's finalizer.
    caplog.set_level(logging.ERROR, logger="asyncio")
    app = Probe()
    async with app.run_test() as pilot:
        app.run_worker(app.wedged(), group="stuck", name="wedge")
        with pytest.raises(AssertionError, match="deadline 0.2s"):
            await settle(pilot, lambda: True, what="wedged", workers=("stuck",), deadline_s=0.2)
    for _ in range(3):
        await asyncio.sleep(0)
    gc.collect()  # the unretrieved-exception log fires from the task's finalizer
    assert "never retrieved" not in caplog.text


@pytest.mark.asyncio
async def test_a_coroutine_join_is_cancelled_when_settle_raises():
    # R2 on PR #3732: settle owns the task it wraps a coroutine join in.
    app = Probe()
    cancelled = asyncio.Event()

    async def wedged_join() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with app.run_test() as pilot:
        with pytest.raises(AssertionError, match="deadline 0.2s"):
            await settle(
                pilot, lambda: True, what="owned join", joins=(wedged_join(),), deadline_s=0.2
            )
        await asyncio.wait_for(cancelled.wait(), 5)


@pytest.mark.asyncio
async def test_a_failing_join_re_raises_naming_what_not_a_deadline():
    # R2 + N2 on PR #3732: a join's own error, a TimeoutError included, is that
    # join's failure under ``what``, never reported as settle's deadline.
    app = Probe()

    async def timing_out_join() -> None:
        raise TimeoutError("upstream gave up")

    async with app.run_test() as pilot:
        with pytest.raises(AssertionError) as raised:
            await settle(pilot, lambda: True, what="probe join", joins=(timing_out_join(),))
    message = str(raised.value)
    assert message.startswith("probe join: join ")
    assert "TimeoutError('upstream gave up')" in message
    assert "never settled" not in message
