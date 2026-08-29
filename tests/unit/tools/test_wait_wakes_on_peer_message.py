"""``wait`` wakes when a peer message lands, and on a steer cancel.

The defect these pin: `lop send` in its default mailbox mode was invisible to a
session parked in a long `wait`. The wait had exactly three wake sources (a job
settling, the abort signal, the deadline), so a message sent to a session
waiting on a 40-minute build was not read until the wait's budget expired.

The second half covers `--now`: the steer DID cancel the wait promptly, but
`interruptible_runner` then handed the model "Tool call skipped: interrupted by
steering", discarding the job id and status it needs to resume. Both paths now
return the same still-running shape.

The abort test is the load-bearing one. Absorbing `CancelledError` is how a
tool can accidentally defeat Esc, so the re-raise on `signal.aborted` is pinned
here deliberately.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools.builtin import execute_wait


class _Peer:
    """Test double for PeerArrivalProtocol, matching the session's semantics.

    Set-only and monotonic, exactly like `Session._PeerArrival`: a double that
    cleared the event would hide the very lost-wakeup bug the count exists to
    prevent.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._count = 0

    def event(self) -> asyncio.Event:
        return self._event

    def count(self) -> int:
        return self._count

    def mark(self) -> None:
        self._count += 1
        self._event.set()


def _runner(delay: float, result: str = "done") -> Any:
    async def run(job_id: str, signal: Any, report_progress: Any) -> str:
        await asyncio.sleep(delay)
        return result

    return run


async def _wait(context: ToolContext, signal: AbortSignal | None = None, **args: Any):
    return await execute_wait("wc", args, signal, None, context)


@pytest.mark.asyncio
async def test_a_peer_message_wakes_a_blocking_wait() -> None:
    """The headline fix: a mailbox delivery returns the wait in ~0s, not in
    300s, and the model keeps the job id so it can re-issue the wait."""
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))

    async def send_soon() -> None:
        await asyncio.sleep(0.2)
        peer.mark()

    asyncio.ensure_future(send_soon())
    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=300_000)
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0, f"waited {elapsed:.2f}s for a peer message sent at 0.2s"
    assert "still running" in result.text
    assert "another session" in result.text
    assert result.details is not None
    assert result.details["job_id"] == job_id
    assert result.details["status"] == "running"
    assert result.details["interrupted_by"] == "peer_message"
    # Nothing was cancelled: the job the model asked about keeps running.
    row = manager.get(job_id)
    assert row is not None and row.status == "running"
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_already_delivered_message_does_not_re_wake_a_new_wait() -> None:
    """A message that landed BEFORE this wait started must NOT wake it.

    It has already reached the model: the loop drains the journal at the top
    of each continuation iteration, i.e. before the model call that issued
    this wait. Firing on it would return instantly, burn a provider round trip
    to re-read a message the model just read, and do it again on every retry.

    This also pins the re-arm. A permanently-set event makes `asyncio.wait`
    return immediately on every iteration, spinning the wait loop at full
    speed until its deadline instead of parking — the same event-loop burn
    `_await_any_settled` documents for evicted job rows.
    """
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))

    peer.mark()  # already delivered and already read

    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=150)
    elapsed = time.perf_counter() - started

    assert "still running after 150ms" in result.text
    assert result.details is not None and "interrupted_by" not in result.details
    # Parked for its budget rather than spinning through it.
    assert elapsed >= 0.1, f"returned in {elapsed:.3f}s — the wait did not park"
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_message_arriving_between_two_parks_is_not_lost() -> None:
    """The lost-wakeup regression test.

    A wait parks repeatedly (each `_await_any_settled` returns on its own
    deadline slice). A message landing while the tool is between two parks
    must still be seen, which is what the count comparison buys over reading
    `is_set()` after a clear.
    """
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    job_id = manager.register("task", "slow", _runner(30.0))

    # Mark from a plain callback so it lands without this coroutine awaiting:
    # the tool is mid-iteration, not parked, when the count moves.
    asyncio.get_running_loop().call_later(0.2, peer.mark)

    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=300_000)
    assert time.perf_counter() - started < 5.0
    assert result.details is not None
    assert result.details["interrupted_by"] == "peer_message"
    await manager.dispose()


@pytest.mark.asyncio
async def test_peer_wake_works_when_no_job_row_survives() -> None:
    """The no-jobs early-out (`not events and signal is None`) sleeps out the
    whole remainder. Without the peer event in that condition the waiter is
    silently dropped and the mailbox wake is dead on this path."""
    manager = AsyncJobManager()
    peer = _Peer()

    class _NoEventManager:
        """Older manager surface: no settled_event hook at all."""

        def __init__(self, inner: AsyncJobManager) -> None:
            self._inner = inner
            self.register = inner.register

        def get(self, job_id: str, **kw: Any) -> Any:
            return self._inner.get(job_id, **kw)

        def list(self, **kw: Any) -> Any:
            return self._inner.list(**kw)

        async def cancel(self, job_id: str, **kw: Any) -> bool:
            return await self._inner.cancel(job_id, **kw)

        def mark_consumed(self, job_id: str) -> None:
            self._inner.mark_consumed(job_id)

    legacy = _NoEventManager(manager)
    context = ToolContext(cwd=".", jobs=legacy, peer_arrival=peer)
    job_id = legacy.register("task", "slow", _runner(30.0))

    async def send_soon() -> None:
        await asyncio.sleep(0.2)
        peer.mark()

    asyncio.ensure_future(send_soon())
    started = time.perf_counter()
    result = await _wait(context, job_id=job_id, wait_ms=300_000)
    assert time.perf_counter() - started < 5.0
    assert result.details is not None
    assert result.details["interrupted_by"] == "peer_message"
    await manager.dispose()


@pytest.mark.asyncio
async def test_no_peer_surface_leaves_wait_unchanged() -> None:
    """`peer_arrival=None` (a host with no peer surface) must keep the old
    three wake sources exactly."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager)
    job_id = manager.register("task", "slow", _runner(30.0))

    result = await _wait(context, job_id=job_id, wait_ms=150)
    assert "still running after 150ms" in result.text
    assert result.details is not None and "interrupted_by" not in result.details
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_steer_cancel_reports_the_job_instead_of_skipped() -> None:
    """`lop send --now`: the tool task is cancelled, and the model must learn
    the job is still running rather than that the wait was skipped."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=_Peer())
    job_id = manager.register("task", "slow", _runner(30.0))

    task = asyncio.ensure_future(_wait(context, job_id=job_id, wait_ms=300_000))
    await asyncio.sleep(0.2)
    task.cancel()  # what interruptible_runner does on a steer
    result = await task

    assert "still running" in result.text
    assert "steering" in result.text
    assert result.details is not None
    assert result.details["job_id"] == job_id
    assert result.details["status"] == "running"
    assert result.details["interrupted_by"] == "steering"
    row = manager.get(job_id)
    assert row is not None and row.status == "running"
    await manager.dispose()


@pytest.mark.asyncio
async def test_abort_still_stops_a_wait_and_is_not_swallowed() -> None:
    """HIGHEST-SEVERITY guard. Abort must stay strictly stronger than
    steering: with `signal.aborted` set, the CancelledError has to propagate,
    or Esc silently stops stopping a wait."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=_Peer())
    job_id = manager.register("task", "slow", _runner(30.0))
    signal = AbortSignal()

    task = asyncio.ensure_future(_wait(context, signal=signal, job_id=job_id, wait_ms=300_000))
    await asyncio.sleep(0.2)
    signal.abort("user")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await manager.dispose()


@pytest.mark.asyncio
async def test_the_cancel_path_does_not_consume_the_job() -> None:
    """`mark_consumed` must NOT run when the wait is cut short: auto-delivery
    (Session._on_job_completed, keyed on AsyncJob.consumed) is what hands the
    result over later. Correct today, one line from a bug."""
    manager = AsyncJobManager()
    peer = _Peer()
    context = ToolContext(cwd=".", jobs=manager, peer_arrival=peer)
    consumed: list[str] = []
    original = manager.mark_consumed

    def _spy(job_id: str) -> None:
        consumed.append(job_id)
        original(job_id)

    manager.mark_consumed = _spy  # type: ignore[method-assign]

    # Peer path.
    job_id = manager.register("task", "slow", _runner(30.0))
    asyncio.ensure_future(_late_mark(peer))
    await _wait(context, job_id=job_id, wait_ms=300_000)
    assert consumed == [], "a peer-interrupted wait must not consume the job"

    # Steer path.
    task = asyncio.ensure_future(_wait(context, job_id=job_id, wait_ms=300_000))
    await asyncio.sleep(0.2)
    task.cancel()
    await task
    assert consumed == [], "a steer-interrupted wait must not consume the job"
    await manager.dispose()


async def _late_mark(peer: _Peer) -> None:
    await asyncio.sleep(0.2)
    peer.mark()
