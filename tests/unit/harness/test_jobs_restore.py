"""Restoring a task roster across a process boundary.

The subagent panel, the ``jobs`` tool, and ``hub op='list'`` all read
``AsyncJobManager._jobs``, which lives only in memory. Before this feature a
resumed session opened with an empty panel even though the children ran and
their transcripts survived on disk. ``AsyncJobManager.restore`` re-seeds the
table from a persisted snapshot; these tests pin the four properties that make
the rehydrated rows honest:

* a row that was ``running`` when the process died comes back as
  ``interrupted`` (its task is gone; a live-looking row would spin forever);
* every restored row is flagged so retention and capacity treat it as the
  historical record it is, and the sweep never evicts it;
* a live job of the current session is never clobbered by a stale snapshot;
* the roster-change hook fires on registration, settle, and restore so the
  owner can persist without polling.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.jobs import AsyncJob, AsyncJobManager


async def wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


def _row(job_id: str, status: str = "completed", **kw) -> AsyncJob:
    return AsyncJob(id=job_id, type="task", status=status, start_time=1.0, label=job_id, **kw)


@pytest.mark.asyncio
async def test_restore_seeds_rows_and_flags_them() -> None:
    manager = AsyncJobManager()
    manager.restore([_row("aaa", status="completed", settled_at=2.0)])
    job = manager.get("aaa")
    assert job is not None
    assert job.status == "completed"
    assert job.restored is True
    # It appears in the list surface the panel and the jobs tool read.
    assert [j.id for j in manager.list()] == ["aaa"]


@pytest.mark.asyncio
async def test_running_row_restores_as_interrupted() -> None:
    """A child cut off mid-run has no task any more; ``running`` would lie."""
    manager = AsyncJobManager()
    manager.restore([_row("bbb", status="running")])
    job = manager.get("bbb")
    assert job is not None
    assert job.status == "interrupted"
    assert job.restored is True


@pytest.mark.asyncio
async def test_restored_rows_are_never_swept() -> None:
    """Retention ages against ``settled_at``; a restored row carries the
    PREVIOUS session's stamp, which is already past the window — so an
    unguarded sweep would evict every rehydrated child on the first pass."""
    manager = AsyncJobManager(retention_ms=1)
    # settled_at far in the past → past the 1 ms window immediately.
    manager.restore([_row("ccc", status="completed", settled_at=1.0)])
    manager._sweep_due()  # the sweep the runner calls after every settle
    assert manager.get("ccc") is not None


@pytest.mark.asyncio
async def test_restore_never_clobbers_a_live_row() -> None:
    manager = AsyncJobManager()

    async def runner(job_id, signal, report_progress):
        await asyncio.sleep(5)
        return "late"

    live_id = manager.register("task", "live", runner)
    await asyncio.sleep(0)  # let the runner enter so dispose can cancel it cleanly
    # A stale snapshot naming the same id must not overwrite the running job.
    manager.restore([_row(live_id, status="completed")])
    live = manager.get(live_id)
    assert live is not None
    assert live.status == "running"
    assert live.restored is False
    await manager.dispose()


@pytest.mark.asyncio
async def test_roster_change_hook_fires_on_register_settle_and_restore() -> None:
    calls: list[str] = []
    manager = AsyncJobManager(on_roster_change=lambda: calls.append("x"))

    async def runner(job_id, signal, report_progress):
        return "ok"

    job_id = manager.register("task", "t", runner)
    assert calls  # registration fired it
    before_settle = len(calls)
    await wait_for(lambda: manager.get(job_id).status == "completed")  # type: ignore[union-attr]
    assert len(calls) > before_settle  # settle fired it again
    before_restore = len(calls)
    manager.restore([_row("zzz")])
    assert len(calls) > before_restore  # restore fired it too
    await manager.dispose()


@pytest.mark.asyncio
async def test_bash_rows_do_not_trigger_the_roster_hook() -> None:
    """Only ``task`` rows carry a resumable transcript; a bash job firing the
    roster hook would append a snapshot for a process a resume can never
    touch."""
    calls: list[str] = []
    manager = AsyncJobManager(on_roster_change=lambda: calls.append("x"))

    async def runner(job_id, signal, report_progress):
        return "ok"

    manager.register("bash", "b", runner)
    await asyncio.sleep(0)  # let the runner enter and settle so nothing is left unawaited
    assert calls == []
    await manager.dispose()


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_break_bookkeeping() -> None:
    def boom() -> None:
        raise RuntimeError("bad listener")

    manager = AsyncJobManager(on_roster_change=boom)

    async def runner(job_id, signal, report_progress):
        return "ok"

    # Registration must still return an id and track the job despite the raise.
    job_id = manager.register("task", "t", runner)
    assert manager.get(job_id) is not None
    await wait_for(lambda: manager.get(job_id).status == "completed")  # type: ignore[union-attr]
    await manager.dispose()
