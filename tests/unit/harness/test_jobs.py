"""AsyncJobManager tests: lifecycle, capacity, retention, and — the part most
systems get wrong — owner-scoped delivery with dead-lettering."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.jobs import (
    DEFAULT_MAX_RUNNING_JOBS,
    AsyncJob,
    AsyncJobManager,
)


async def wait_for(predicate, timeout: float = 2.0) -> None:
    """Poll a predicate until true; fail the test on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


def require_job(manager: AsyncJobManager, job_id: str) -> AsyncJob:
    """``manager.get`` narrowed to non-None for assertions."""
    job = manager.get(job_id)
    assert job is not None
    return job


async def quick_runner(job_id, signal, report_progress):
    report_progress("halfway")
    return f"done:{job_id}"


@pytest.mark.asyncio
async def test_register_runs_and_completes():
    manager = AsyncJobManager()
    job_id = manager.register("task", "quick", quick_runner)
    job = manager.get(job_id)
    assert job is not None and job.status == "running"
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    assert require_job(manager, job_id).result_text == f"done:{job_id}"
    await manager.dispose()


@pytest.mark.asyncio
async def test_failed_job_reports_error_text():
    async def boom(job_id, signal, report_progress):
        raise ValueError("kaboom")

    manager = AsyncJobManager()
    job_id = manager.register("task", "boom", boom)
    await wait_for(lambda: require_job(manager, job_id).status == "failed")
    job = manager.get(job_id)
    assert job is not None
    assert "kaboom" in (job.error_text or "")
    await manager.dispose()


@pytest.mark.asyncio
async def test_max_running_enforced_queued_dont_count():
    """Queued jobs hold no execution slot: 15 running + any queued is fine."""
    manager = AsyncJobManager(max_running=2)
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()
        return "ok"

    a = manager.register("task", "a", blocked)
    manager.register("task", "b", blocked)  # consumes a slot for the capacity cap
    assert manager.at_capacity() is True
    with pytest.raises(RuntimeError):
        manager.register("task", "c", blocked)
    # Queued registration succeeds and does not count against the cap.
    q = manager.register("task", "q", blocked, queued=True)
    assert require_job(manager, q).queued is True
    assert manager.at_capacity() is True  # still only counts a + b
    gate.set()
    await wait_for(lambda: require_job(manager, a).status == "completed")
    await manager.dispose()


@pytest.mark.asyncio
async def test_start_queued_promotes_and_runs():
    manager = AsyncJobManager()
    started: list[str] = []

    async def runner(job_id, signal, report_progress):
        started.append(job_id)
        return "ran"

    job_id = manager.register("task", "parked", runner, queued=True)
    assert started == []
    assert manager.start_queued(job_id) is True
    assert manager.start_queued(job_id) is False  # already promoted
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    assert started == [job_id]
    await manager.dispose()


@pytest.mark.asyncio
async def test_cancel_aborts_signal_and_sets_status():
    manager = AsyncJobManager()
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()
        return "ok"

    job_id = manager.register("task", "long", blocked)
    assert await manager.cancel(job_id) is True
    assert require_job(manager, job_id).status == "cancelled"
    await manager.dispose()


@pytest.mark.asyncio
async def test_cancel_owner_mismatch_is_not_found():
    """A subagent teardown cannot cancel its parent's jobs."""
    manager = AsyncJobManager()
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()

    job_id = manager.register("task", "parent job", blocked, owner_id="Main")
    assert await manager.cancel(job_id, owner_id="Sub") is False
    assert manager.get(job_id, owner_id="Sub") is None  # scoped get too
    assert manager.get(job_id, owner_id="Main") is not None
    assert await manager.cancel(job_id, owner_id="Main") is True
    gate.set()
    await manager.dispose()


@pytest.mark.asyncio
async def test_delivery_sink_scoping():
    """Owned completions route exclusively through the owner's sink; a
    different owner's sink never sees them."""
    manager = AsyncJobManager()
    main_inbox: list[tuple[str, str]] = []
    sub_inbox: list[tuple[str, str]] = []
    manager.register_delivery_sink(
        "Main", lambda job_id, text, job: main_inbox.append((job_id, text))
    )
    manager.register_delivery_sink(
        "Sub", lambda job_id, text, job: sub_inbox.append((job_id, text))
    )

    job_id = manager.register("task", "owned", quick_runner, owner_id="Main")
    await wait_for(lambda: require_job(manager, job_id).status == "completed")

    assert main_inbox == [(job_id, f"done:{job_id}")]
    assert sub_inbox == []  # never leaked across owners
    await manager.dispose()


@pytest.mark.asyncio
async def test_dead_letter_when_no_sink():
    """Owned job, no live sink: dead-lettered — NOT routed to the fallback."""
    fallback: list[str] = []

    async def on_complete(job_id, text, job):
        fallback.append(job_id)

    manager = AsyncJobManager(on_job_complete=on_complete)
    job_id = manager.register("task", "orphan", quick_runner, owner_id="Ghost")
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    # The row keeps its result for retention, but nothing was delivered.
    assert fallback == []
    assert require_job(manager, job_id).result_text == f"done:{job_id}"
    await manager.dispose()


@pytest.mark.asyncio
async def test_unowned_job_uses_fallback():
    fallback: list[tuple[str, str]] = []

    async def on_complete(job_id, text, job):
        fallback.append((job_id, text))

    manager = AsyncJobManager(on_job_complete=on_complete)
    job_id = manager.register("task", "unowned", quick_runner)
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    assert fallback == [(job_id, f"done:{job_id}")]
    await manager.dispose()


@pytest.mark.asyncio
async def test_async_sink_is_awaited():
    manager = AsyncJobManager()
    delivered: list[str] = []

    async def sink(job_id, text, job):
        await asyncio.sleep(0.01)
        delivered.append(job_id)

    manager.register_delivery_sink("Main", sink)
    job_id = manager.register("task", "async", quick_runner, owner_id="Main")
    await wait_for(lambda: delivered == [job_id])
    await manager.dispose()


@pytest.mark.asyncio
async def test_unregister_sink():
    manager = AsyncJobManager()
    inbox: list[str] = []
    unregister = manager.register_delivery_sink("Main", lambda j, t, job: inbox.append(j))
    unregister()
    job_id = manager.register("task", "after-unregister", quick_runner, owner_id="Main")
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    assert inbox == []  # dead-lettered again
    await manager.dispose()


@pytest.mark.asyncio
async def test_retention_sweep_drops_old_settled_jobs():
    manager = AsyncJobManager(retention_ms=10)
    job_id = manager.register("task", "ephemeral", quick_runner)
    await wait_for(lambda: require_job(manager, job_id).status == "completed")
    await asyncio.sleep(0.05)
    # The next lifecycle event sweeps settled jobs past the retention window.
    gate = asyncio.Event()

    async def blocked(jid, signal, report_progress):
        await gate.wait()

    trigger = manager.register("task", "trigger", blocked)
    await manager.cancel(trigger)
    gate.set()
    assert manager.get(job_id) is None
    await manager.dispose()


@pytest.mark.asyncio
async def test_list_scoped_by_owner():
    manager = AsyncJobManager()
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()

    manager.register("task", "main job", blocked, owner_id="Main")
    manager.register("task", "sub job", blocked, owner_id="Sub")
    assert [j.label for j in manager.list(owner_id="Main")] == ["main job"]
    assert len(manager.list()) == 2
    gate.set()
    await manager.dispose()


def test_defaults():
    assert DEFAULT_MAX_RUNNING_JOBS == 15


@pytest.mark.asyncio
async def test_a_completion_promotes_the_longest_queued_job():
    """C14-01: when a running job settles, its freed slot runs the oldest
    parked job — a queued subagent must not sit ``running + queued`` forever.

    Previously nothing called ``start_queued`` after a completion, so a task
    launched at a full manager parked indefinitely (the wait tool timed out on
    "still running" and the band painted it ✓). This asserts the manager is
    self-healing: 2 fill capacity, a 3rd queues, one completes, and the
    queued job runs.
    """
    manager = AsyncJobManager(max_running=2)
    gate = [asyncio.Event(), asyncio.Event()]
    started: list[str] = []
    completed: list[str] = []

    async def gated(i: int):
        async def runner(job_id, signal, report_progress):
            started.append(job_id)
            await gate[i].wait()
            completed.append(job_id)
            return f"done:{job_id}"
        return runner

    a = manager.register("task", "a", await gated(0))
    b = manager.register("task", "b", await gated(1))
    q = manager.register("task", "queued", quick_runner, queued=True)
    assert require_job(manager, q).queued is True
    assert require_job(manager, q).status == "running"  # parked, not settled
    # The two running jobs start on their own schedules; wait for both to have
    # grabbed their slots before asserting the queued one holds none.
    await wait_for(lambda: len(started) >= 2)
    assert set(started) == {a, b}  # queued holds no slot

    # Free ONE slot (a completes); the queued job should promote and run.
    gate[0].set()
    await wait_for(lambda: require_job(manager, a).status == "completed")
    # The promotion fires a.settled: q flips from parked to running and runs
    # to completion on its own (the whole point — no manual start_queued).
    await wait_for(lambda: require_job(manager, q).status == "completed")
    assert require_job(manager, q).queued is False
    assert require_job(manager, q).result_text == f"done:{q}"

    # b is still running; free it so dispose is clean.
    gate[1].set()
    await wait_for(lambda: require_job(manager, b).status == "completed")
    await manager.dispose()
