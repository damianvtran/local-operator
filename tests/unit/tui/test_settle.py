"""Contract tests for ``tests.unit.tui.waiting.settle`` on a tiny synthetic App.

Each case pins one join the helper promises (worker, thread hop, after-refresh
queue) or one bound (turn cap, deadline). The ``time.sleep`` calls run inside
the synthetic workers' threads: they simulate the off-loop work under test, and
no test-side wait depends on them.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tests.unit.tui.waiting import settle, track_threads


class Probe(App):
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
        # Under a starved pause (Pilot.pause reduced to sleep(0)) only the
        # refresh join drains this within the one allowed turn.
        app.screen.call_after_refresh(setattr, app, "after_refresh", True)
        await settle(pilot, lambda: app.after_refresh, what="after refresh", max_turns=1)
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
