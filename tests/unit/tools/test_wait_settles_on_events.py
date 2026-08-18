"""``wait`` returns on the settle EVENT, not on the next poll tick.

The measured problem: across recorded sessions 70% of ``wait`` calls hit their
deadline and returned "still running", and each of those cost a full parent
model round trip to learn nothing. Two things fix that — waking the instant a
job settles (so a generous budget is free), and being able to wait on several
jobs at once (so a fan-out does not need one poll per child).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.tools.builtin import execute_wait


def _runner(delay: float, result: str = "done") -> Any:
    """A JobRunFn that finishes after ``delay`` seconds."""

    async def run(job_id: str, signal: Any, report_progress: Any) -> str:
        await asyncio.sleep(delay)
        return result

    return run


async def _wait(context: ToolContext, **args: Any):
    return await execute_wait("wc", args, None, None, context)


@pytest.mark.asyncio
async def test_wait_returns_as_soon_as_the_job_settles() -> None:
    """A generous budget must cost nothing when the work finishes early."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "reviewer", _runner(0.2, "reviewed"))

    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=300_000)
    elapsed = time.perf_counter() - started

    assert "reviewed" in result.text
    assert elapsed < 2.0, f"waited {elapsed:.2f}s for a job that took 0.2s"
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_job_that_already_settled_returns_immediately() -> None:
    """The race the poll loop hid by re-reading status: a waiter arriving after
    the transition must not block on news that already happened."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "quick", _runner(0.01))
    await _wait(context, job_id=job_id, wait_ms=5_000)

    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=300_000)
    assert time.perf_counter() - started < 1.0
    assert "completed" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_waiting_on_several_jobs_wakes_on_the_first_to_finish() -> None:
    """Awaiting a fan-out without polling each child in turn."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    slow = manager.register("task", "slow", _runner(30.0, "slow"))
    fast = manager.register("task", "fast", _runner(0.2, "fast"))

    started = time.perf_counter()
    result = await _wait(context, job_id=[slow, fast], wait_ms=300_000)
    elapsed = time.perf_counter() - started

    text = result.text
    assert "fast" in text
    assert elapsed < 5.0, f"waited {elapsed:.2f}s for the first of two"
    assert "still running" in text, "the caller must learn the others are unfinished"
    assert result.details is not None and result.details["job_id"] == fast
    await manager.dispose()


@pytest.mark.asyncio
async def test_the_timeout_path_still_reports_honestly() -> None:
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))

    result = await _wait(context, job_id=job_id, wait_ms=150)
    assert "still running after 150ms" in result.text
    assert result.details is not None and result.details["status"] == "running"
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_unknown_job_is_an_error_not_a_hang() -> None:
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    result = await _wait(context, job_id="nope", wait_ms=100)
    assert result.is_error
    assert "unknown job" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_repeated_timed_out_waits_leak_no_tasks() -> None:
    """Each wait registers one waiter per job; leaving them pending would leak
    a task per call for the life of the session."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))

    await _wait(context, job_id=job_id, wait_ms=10)
    await asyncio.sleep(0.05)
    before = len(asyncio.all_tasks())
    for _ in range(15):
        await _wait(context, job_id=job_id, wait_ms=10)
    await asyncio.sleep(0.05)
    assert len(asyncio.all_tasks()) <= before
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_manager_without_the_event_hook_still_works() -> None:
    """A third-party job manager satisfying only the older protocol must keep
    working — which is why ``wait`` probes for the hook instead of requiring
    it on JobManagerProtocol."""

    class LegacyManager:
        """Old surface: get/list/cancel and nothing else."""

        def __init__(self) -> None:
            self._inner = AsyncJobManager()
            self.register = self._inner.register
            self.mark_consumed = self._inner.mark_consumed

        def get(self, job_id: str, *, owner_id: str | None = None) -> Any:
            return self._inner.get(job_id, owner_id=owner_id)

        def list(self, *, owner_id: str | None = None) -> list[Any]:
            return self._inner.list(owner_id=owner_id)

        async def cancel(self, job_id: str, *, owner_id: str | None = None) -> bool:
            return await self._inner.cancel(job_id, owner_id=owner_id)

    manager = LegacyManager()
    assert not hasattr(manager, "settled_event")
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "legacy", _runner(0.2, "legacy done"))

    result = await _wait(context, job_id=job_id, wait_ms=10_000)
    assert "legacy done" in result.text
    await manager._inner.dispose()


@pytest.mark.asyncio
async def test_dispose_wakes_a_waiter_instead_of_stranding_it() -> None:
    """A waiter must not sleep to its deadline against a manager that will
    never run again."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))

    waiter = asyncio.ensure_future(_wait(context, job_id=job_id, wait_ms=300_000))
    await asyncio.sleep(0.05)
    await manager.dispose()

    result = await asyncio.wait_for(waiter, timeout=5.0)
    assert "cancelled" in result.text.lower()


# ---------------------------------------------------------------------------
# Round-1 review regressions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_aborted_wait_returns_promptly() -> None:
    """C3: the old 50ms poll re-read `signal.aborted` every tick. Parking on
    the settle events alone made the abort branch dead for a job that never
    settles — an aborted wait sat for its whole budget (up to five minutes).
    The TUI masks this through its own interruptible-tool poll, but that is a
    different mechanism and does not cover deadline signals or non-TUI hosts.
    """
    from local_operator.harness.types import AbortSignal

    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))
    signal = AbortSignal()

    async def abort_soon() -> None:
        await asyncio.sleep(0.1)
        signal.abort("user")

    asyncio.ensure_future(abort_soon())
    started = time.perf_counter()
    result = await execute_wait("wc", {"job_id": job_id, "wait_ms": 10_000}, signal, None, context)
    elapsed = time.perf_counter() - started

    assert "aborted" in result.text
    assert elapsed < 3.0, f"abort took {elapsed:.2f}s to be noticed"
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_event_is_not_stored_for_a_job_that_has_no_row() -> None:
    """C5: the pre-set branch also STORED the event, and `_sweep_due` only pops
    ids it finds in `_jobs`, so those entries were unreachable by every cleanup
    path."""
    manager = AsyncJobManager()
    for index in range(50):
        event = manager.settled_event(f"ghost-{index}")
        assert event.is_set(), "an id with no row can never settle"
    assert manager._settled_events == {}
    await manager.dispose()


@pytest.mark.asyncio
async def test_the_timeout_names_a_job_that_is_actually_running() -> None:
    """C6: `details` pinned `job_ids[0]`, which on the multi-id path is just
    the first id the caller passed and may itself have settled."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    first = manager.register("task", "A", _runner(30.0))
    second = manager.register("task", "B", _runner(30.0))

    async def evict() -> None:
        await asyncio.sleep(0.05)
        del manager._jobs[first]

    asyncio.ensure_future(evict())
    result = await _wait(context, job_id=[first, second], wait_ms=250)

    assert result.details is not None
    assert result.details["job_id"] == second
    assert first not in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_job_swept_mid_wait_does_not_spin_the_loop() -> None:
    """C12: an evicted row yields a pre-set event, so `asyncio.wait` returned
    at once, `_settled()` found nothing, and the caller's loop re-entered at
    full speed — burning the event loop this path exists to protect, and faster
    than the poll it replaced (measured 635/s)."""
    import local_operator.tools.builtin as builtin_module

    manager = AsyncJobManager(retention_ms=0)
    context = ToolContext(cwd=".", jobs=manager)
    calls = {"n": 0}
    original = builtin_module._await_any_settled

    async def counted(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        await original(*args, **kwargs)

    settles_soon = manager.register("task", "A", _runner(0.15, "done"))
    keeps_running = manager.register("task", "B", _runner(60.0))

    builtin_module._await_any_settled = counted
    try:
        result = await _wait(context, job_id=[settles_soon, keeps_running], wait_ms=800)
    finally:
        builtin_module._await_any_settled = original

    assert "still running" in result.text
    assert calls["n"] < 20, f"spun {calls['n']} times; the wait is busy-looping"
    await manager.dispose()
