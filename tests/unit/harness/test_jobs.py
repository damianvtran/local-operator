"""AsyncJobManager tests: lifecycle, capacity, retention, and — the part most
systems get wrong — owner-scoped delivery with dead-lettering."""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.jobs import (
    CANCELLED_BEFORE_START,
    DEFAULT_MAX_RUNNING_JOBS,
    OUTPUT_TAIL_CHARS,
    AsyncJob,
    AsyncJobManager,
    JobStatus,
)
from local_operator.harness.types import Usage


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


def require_output(manager: AsyncJobManager, job_id: str, since: int = 0) -> tuple[str, int, bool]:
    """``manager.read_output`` narrowed to non-None for assertions."""
    window = manager.read_output(job_id, since)
    assert window is not None
    return window


async def quick_runner(job_id, signal, report_progress):
    report_progress("halfway")
    return f"done:{job_id}"


@pytest.mark.asyncio
async def test_task_notification_callers_classify_transient_and_durable_mutations() -> None:
    """Only fields retained by the resume projection schedule persistence.

    This enumerates every public manager mutation that notifies while a task is
    alive: registration, queue promotion, progress, output, and settlement.
    Launch metadata/model/usage use ``_notify_roster_change`` directly in the
    subagent runner and are covered by the same durable callback below.
    """
    durable: list[str] = []
    live: list[str] = []
    release_running = asyncio.Event()
    release_queued = asyncio.Event()
    started = asyncio.Event()
    queued = ""

    async def blocked(job_id, signal, report_progress):  # noqa: ANN001, ANN202
        started.set()
        await (release_queued if job_id == queued else release_running).wait()
        return "settled result"

    manager = AsyncJobManager(
        max_running=1,
        on_roster_change=lambda: durable.append("persist"),
        on_job_change=lambda: live.append("publish"),
    )
    running = manager.register("task", "running", blocked)
    queued = manager.register("task", "queued", blocked, queued=True)
    assert (len(durable), len(live)) == (2, 2)  # both registrations
    await started.wait()

    manager._progress_fn(running)("reading files")
    manager.append_output(running, "one live line\n")
    assert (len(durable), len(live)) == (2, 4)

    release_running.set()
    await manager.settled_event(running).wait()
    # Settlement frees capacity and atomically promotes the queued row: both are
    # durable lifecycle moves, so each publishes and persists once.
    assert (len(durable), len(live)) == (4, 6)
    assert manager.start_queued(queued) is False  # already auto-promoted
    release_queued.set()
    await manager.settled_event(queued).wait()
    assert (len(durable), len(live)) == (5, 7)  # second settlement
    await manager.dispose()


@pytest.mark.asyncio
async def test_progress_burst_publishes_every_edge_without_persisting() -> None:
    """Live activity stays smooth while the durable writer remains untouched."""
    durable = 0
    live = 0
    release = asyncio.Event()

    def persisted() -> None:
        nonlocal durable
        durable += 1

    def published() -> None:
        nonlocal live
        live += 1

    async def blocked(job_id, signal, report_progress):  # noqa: ANN001, ANN202
        await release.wait()
        return "done"

    manager = AsyncJobManager(on_roster_change=persisted, on_job_change=published)
    job_id = manager.register("task", "streaming", blocked)
    durable = live = 0
    report = manager._progress_fn(job_id)
    for index in range(60):
        report(f"edge {index}")
    assert (durable, live) == (0, 60)

    release.set()
    await manager.settled_event(job_id).wait()
    assert (durable, live) == (1, 61)
    assert require_job(manager, job_id).result_text == "done"
    await manager.dispose()


def test_accumulate_usage_preserves_provider_reported_calls() -> None:
    """A tool-using child's receipts stay attached to their original calls."""
    from local_operator.harness.subagent import _accumulate_usage

    class _Job:
        def __init__(self, usage=None) -> None:
            self.usage = usage

    job = _Job()
    _accumulate_usage(job, Usage(input_tokens=10, output_tokens=0, usd_cost=0.001))
    _accumulate_usage(job, Usage(input_tokens=20, output_tokens=0, usd_cost=0.002))
    assert job.usage is not None
    assert job.usage.input_tokens == 30
    assert job.usage.usd_cost is None
    assert [component.usd_cost for component in job.usage.cost_components] == [0.001, 0.002]


def test_accumulate_usage_leaves_reported_dollar_none_when_unreported() -> None:
    """A child whose providers never report a dollar figure keeps ``usd_cost``
    ``None`` — "not reported" is not a sum of zeros and must fall back to the
    token estimate downstream."""
    from local_operator.harness.subagent import _accumulate_usage

    class _Job:
        def __init__(self) -> None:
            self.usage = None

    job = _Job()
    _accumulate_usage(job, Usage(input_tokens=10))
    _accumulate_usage(job, Usage(input_tokens=20))
    assert job.usage is not None
    assert job.usage.usd_cost is None
    assert job.usage.input_tokens == 30
    assert len(job.usage.cost_components) == 2


def test_accumulate_usage_folds_cache_write_ttl_split() -> None:
    """The 5m/1h cache-write split folds wherever ``cache_write_tokens`` does
    (review F4) — otherwise the job aggregate's split reads as the FIRST
    child call's value only, and a per-rate price split silently
    mis-reports from its first reader."""
    from local_operator.harness.subagent import _accumulate_usage

    class _Job:
        def __init__(self, usage=None) -> None:
            self.usage = usage

    job = _Job()
    _accumulate_usage(
        job,
        Usage(
            input_tokens=1,
            cache_write_tokens=1_000,
            cache_write_5m_tokens=1_000,
            cache_write_1h_tokens=0,
        ),
    )
    _accumulate_usage(
        job,
        Usage(
            input_tokens=1,
            cache_write_tokens=3_000,
            cache_write_5m_tokens=0,
            cache_write_1h_tokens=3_000,
        ),
    )
    assert job.usage is not None
    assert job.usage.cache_write_tokens == 4_000
    assert job.usage.cache_write_5m_tokens == 1_000
    assert job.usage.cache_write_1h_tokens == 3_000


def _task_row(
    job_id: str,
    *,
    usage: Usage | None = None,
    model_label: str = "test/model",
    descendant_usage: list[Usage] | None = None,
    status: JobStatus = "completed",
) -> AsyncJob:
    return AsyncJob(
        id=job_id,
        type="task",
        status=status,
        start_time=1.0,
        label=job_id,
        model_label=model_label,
        usage=usage,
        descendant_usage=descendant_usage or [],
    )


def test_accounting_summary_is_bounded_for_hundred_child_fanout() -> None:
    manager = AsyncJobManager()
    manager.restore(
        [
            _task_row(
                f"child-{index}",
                usage=Usage(input_tokens=1, provider="test", model_id="model"),
            )
            for index in range(100)
        ]
    )
    summary = manager.accounting_components()
    assert len(summary) == 1
    assert summary[0].input_tokens == 100


def test_accounting_summary_covers_every_production_nesting_level() -> None:
    leaf_manager = AsyncJobManager()
    leaf_manager.restore([_task_row("grandchild", usage=Usage(input_tokens=3))])
    child_manager = AsyncJobManager()
    child = _task_row("child", usage=Usage(input_tokens=2), status="running")
    child.child_jobs = leaf_manager
    child_manager._jobs[child.id] = child
    root = _task_row("root", usage=Usage(input_tokens=1), status="running")
    root.child_jobs = child_manager
    manager = AsyncJobManager()
    manager._jobs[root.id] = root

    summary = manager.accounting_components()
    assert sum(item.input_tokens for item in summary) == 6


def test_unchanged_accounting_reads_do_not_rewalk_children(monkeypatch) -> None:
    child = AsyncJobManager()
    child.restore([_task_row("leaf", usage=Usage(input_tokens=2))])
    parent = AsyncJobManager()
    parent._jobs["root"] = _task_row("root", status="running")
    parent.attach_child_manager("root", child)
    assert sum(item.input_tokens for item in parent.accounting_components()) == 2

    def fail_if_rebuilt(seen):  # noqa: ANN001, ANN202
        raise AssertionError("unchanged accounting read rebuilt the child tree")

    monkeypatch.setattr(child, "_collect_accounting_components", fail_if_rebuilt)
    for _ in range(100):
        assert sum(item.input_tokens for item in parent.accounting_components()) == 2


def test_grandchild_accounting_invalidation_reaches_root_once() -> None:
    leaf = AsyncJobManager()
    leaf._jobs["leaf"] = _task_row("leaf", usage=Usage(input_tokens=2), status="running")
    child = AsyncJobManager()
    child._jobs["child"] = _task_row("child", status="running")
    child.attach_child_manager("child", leaf)
    root = AsyncJobManager()
    root._jobs["root"] = _task_row("root", status="running")
    root.attach_child_manager("root", child)
    assert sum(item.input_tokens for item in root.accounting_components()) == 2
    revision = root._accounting_revision

    leaf._jobs["leaf"].usage.input_tokens += 3  # type: ignore[union-attr]
    leaf.note_usage_changed()
    assert root._accounting_revision == revision + 1
    assert sum(item.input_tokens for item in root.accounting_components()) == 5


def test_accounting_listener_cycle_is_bounded() -> None:
    left = AsyncJobManager()
    right = AsyncJobManager()
    left._jobs["left"] = _task_row("left", status="running")
    right._jobs["right"] = _task_row("right", status="running")
    left.attach_child_manager("left", right)
    right.attach_child_manager("right", left)

    revision = left._accounting_revision
    right.note_usage_changed()
    assert left._accounting_revision == revision + 1
    assert left.accounting_components() == []


def test_accounting_summary_keeps_duplicate_ids_from_independent_managers() -> None:
    left = AsyncJobManager()
    right = AsyncJobManager()
    left.restore([_task_row("same-id", usage=Usage(input_tokens=2))])
    right.restore([_task_row("same-id", usage=Usage(input_tokens=3))])
    root = AsyncJobManager()
    root.restore(
        [
            _task_row(
                "root",
                descendant_usage=[*left.accounting_components(), *right.accounting_components()],
            )
        ]
    )
    assert sum(item.input_tokens for item in root.accounting_components()) == 5


def test_accounting_summary_preserves_mixed_receipts_and_estimates() -> None:
    manager = AsyncJobManager()
    manager.restore(
        [
            _task_row(
                "mixed",
                descendant_usage=[
                    Usage(
                        input_tokens=9,
                        usd_cost=0.25,
                        provider="openrouter",
                        model_id="routed",
                    ),
                    Usage(input_tokens=2, provider="test", model_id="model"),
                ],
            )
        ]
    )
    summary = manager.accounting_components()
    assert len(summary) == 2
    assert sorted(item.usd_cost for item in summary if item.usd_cost is not None) == [0.25]
    assert sum(item.input_tokens for item in summary) == 11


def test_accumulate_usage_mixes_receipt_and_estimated_calls_without_double_counting() -> None:
    """A receipt covers only its call; another child's call remains estimated."""
    from local_operator.harness.subagent import _accumulate_usage
    from local_operator.model.registry import ModelInfo
    from local_operator.tui.costs import job_cost

    class _Job:
        def __init__(self) -> None:
            self.usage = None
            self.model_label = "test/model"

    job = _Job()
    _accumulate_usage(
        job,
        Usage(
            input_tokens=1_000_000,
            usd_cost=0.001,
            provider="openrouter",
            model_id="routed",
        ),
    )
    _accumulate_usage(job, Usage(input_tokens=1_000_000, provider="test", model_id="model"))
    priced = ModelInfo(id="model", name="model", description="", input_price=20.0)
    from unittest.mock import patch

    # Both resolvers patched: job_cost prices through the paint-safe one
    # (resolve_model_info_paint); patching the full resolver alone no longer
    # reaches the pricing path.
    with (
        patch("local_operator.model.configure.resolve_model_info", return_value=priced),
        patch(
            "local_operator.model.configure.resolve_model_info_paint",
            return_value=(priced, True),
        ),
    ):
        assert job_cost(job) == pytest.approx(20.001)


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
async def test_cancelling_a_parked_job_clears_its_queued_flag_and_runner():
    """``queued`` means "waiting for a slot", and a cancelled job waits for
    nothing.

    Left set, every reader that branches on the flag reported the job as
    pending: the subagent panel painted ``⏳ queued`` on a row whose status was
    ``cancelled``, which reads as work about to start rather than work that
    was stopped. The parked runner is dropped with it — ``start_queued``
    refuses a non-running job, so the entry could never run again and only
    pinned its closure (prompt, parent session, model spec) for the life of
    the manager.
    """
    manager = AsyncJobManager(max_running=1)
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()
        return "ok"

    running = manager.register("task", "running", blocked)
    parked = manager.register("task", "parked", blocked, queued=manager.at_capacity())
    assert require_job(manager, parked).queued is True

    assert await manager.cancel(parked) is True
    job = require_job(manager, parked)
    assert job.status == "cancelled"
    assert job.queued is False, "a cancelled job is not waiting for a slot"
    # The runner is gone, so nothing can promote it and nothing pins its closure.
    assert parked not in manager._queued_runners
    # The surviving running job still holds the manager's only slot: clearing
    # a cancelled row's ``queued`` must not free or double-count one.
    # (``queued_ids() == []`` was asserted here and could not fail — it filters
    # on ``status == "running"``, which a cancelled row never is.)
    assert manager.at_capacity() is True

    gate.set()
    await wait_for(lambda: require_job(manager, running).status == "completed")
    await manager.dispose()


@pytest.mark.asyncio
async def test_cancelling_a_job_that_never_ran_records_that_it_never_ran():
    """Cancelling erases the record that the job never started, and every
    surface then presents its WAITING time as work time.

    Both the panel row and the full-page view measure ``settled_at -
    start_time``, which for a job that never began is how long it sat — printed
    in the column where every other row's number is time a child spent working.
    A child that ran 0 s and spent $0 rendered ``⊘ 1m36s`` with no word beside
    it, which reads as a run somebody killed a minute and a half in.

    Three states reach ``cancel`` without the runner ever being entered, and
    ``queued`` only identifies one of them. The ADMITTED-but-not-yet-entered
    job is the state that motivated this fix at all: the observed ledger had
    ``at_capacity: False`` with nothing parked, so a stamp keyed on ``queued``
    would have labelled none of those rows.
    """
    manager = AsyncJobManager(max_running=1)
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()
        return "ok"

    running = manager.register("task", "running", blocked)
    parked = manager.register("task", "parked", blocked, queued=manager.at_capacity())
    # `register` only SCHEDULES the runner, so the yield is what makes
    # `running` a job that genuinely started — and what stops its coroutine
    # being abandoned un-awaited when `cancel` kills the task.
    await asyncio.sleep(0)
    assert require_job(manager, running).started_at is not None
    assert require_job(manager, parked).started_at is None

    assert await manager.cancel(parked) is True
    assert require_job(manager, parked).result_text == CANCELLED_BEFORE_START

    # A job whose runner DID begin owns this field — it may be mid-flight in it
    # — so a genuinely running job is never stamped.
    assert await manager.cancel(running) is True
    assert require_job(manager, running).result_text is None

    gate.set()
    await manager.dispose()


@pytest.mark.asyncio
async def test_an_admitted_job_cancelled_before_it_ran_is_not_called_a_run():
    """The state this fix exists for, and the one ``queued`` does not identify.

    ``register`` calls ``ensure_future``, which SCHEDULES the runner without
    entering it, so between registration and the parent's next await the job is
    admitted, counted against capacity, ``queued=False``, and has not executed a
    line. Cancelled there it did no work, and a stamp keyed on ``queued`` left
    it reading ``⊘ cancelled · 1m36s`` — the exact failure the parked case was
    fixed for, on the row the original incident actually produced.
    """
    manager = AsyncJobManager(max_running=15)

    async def never_reached(job_id, signal, report_progress):
        return "ok"

    admitted = manager.register("task", "admitted", never_reached)
    job = require_job(manager, admitted)
    # No yield: the runner is scheduled and has not been entered.
    assert job.queued is False, "premise: this job is admitted, not parked"
    assert job.started_at is None, "premise: its runner has not begun"

    assert await manager.cancel(admitted) is True
    assert job.result_text == CANCELLED_BEFORE_START

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
async def test_retention_zero_preserves_accounting_before_any_harvest():
    manager = AsyncJobManager(retention_ms=0)

    async def priced(job_id, signal, report_progress):
        job = require_job(manager, job_id)
        job.model_label = "test/model"
        job.usage = Usage(input_tokens=7, provider="test", model_id="model")
        return "done"

    job_id = manager.register("task", "priced", priced)
    await manager.settled_event(job_id).wait()
    assert manager.list() == []
    summary = manager.accounting_components()
    assert len(summary) == 1
    assert summary[0].input_tokens == 7


@pytest.mark.asyncio
async def test_resumed_run_adds_to_restored_accounting_once():
    manager = AsyncJobManager()
    manager.restore([_task_row("prior", usage=Usage(input_tokens=4))])

    async def resumed(job_id, signal, report_progress):
        job = require_job(manager, job_id)
        job.model_label = "test/model"
        job.usage = Usage(input_tokens=6, provider="test", model_id="model")
        return "continued"

    job_id = manager.register("task", "resumed", resumed)
    await manager.settled_event(job_id).wait()
    assert sum(item.input_tokens for item in manager.accounting_components()) == 10
    # Retention and repeated snapshots read the manager accumulator; neither
    # re-folds the still-retained terminal row.
    assert sum(item.input_tokens for item in manager.accounting_components()) == 10


@pytest.mark.asyncio
async def test_cancel_hands_prior_usage_to_accounting_once():
    manager = AsyncJobManager(retention_ms=0)
    started = asyncio.Event()

    async def spending(job_id, signal, report_progress):
        job = require_job(manager, job_id)
        job.model_label = "test/model"
        job.usage = Usage(input_tokens=4, provider="test", model_id="model")
        started.set()
        await signal.wait()

    job_id = manager.register("task", "spending", spending)
    await started.wait()
    assert await manager.cancel(job_id)
    assert sum(item.input_tokens for item in manager.accounting_components()) == 4
    assert not await manager.cancel(job_id)
    assert sum(item.input_tokens for item in manager.accounting_components()) == 4


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


# ---------------------------------------------------------------------------
# live output tail: the data behind `jobs(op="peek")`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_output_returns_only_what_is_new() -> None:
    """The incremental contract: a cursor read never re-sends what it saw.

    This is what makes polling a long job affordable — re-sending the whole
    tail on every peek would grow the caller's context by the same bytes over
    and over.
    """
    manager = AsyncJobManager()
    job_id = manager.register("bash", "long", quick_runner)
    manager.append_output(job_id, "first\n")
    text, seq, gap = require_output(manager, job_id, 0)
    assert (text, gap) == ("first\n", False)

    # Nothing appended since: the same cursor yields nothing, not a repeat.
    assert require_output(manager, job_id, seq) == ("", seq, False)

    manager.append_output(job_id, "second\n")
    text, seq2, gap = require_output(manager, job_id, seq)
    assert (text, gap) == ("second\n", False)
    assert seq2 > seq
    # A cursor from the very start still replays everything retained.
    assert require_output(manager, job_id, 0)[0] == "first\nsecond\n"
    await manager.dispose()


@pytest.mark.asyncio
async def test_output_tail_is_bounded_and_reports_the_gap() -> None:
    """Past the cap the oldest bytes go, and a stale cursor is TOLD they went.

    Silently returning the surviving tail would hand the caller an excerpt that
    looks contiguous with its last peek but has a hole in it — the caller would
    conclude a step never happened.
    """
    manager = AsyncJobManager()
    job_id = manager.register("bash", "chatty", quick_runner)
    manager.append_output(job_id, "A" * 10)
    cursor = require_job(manager, job_id).output_seq
    manager.append_output(job_id, "B" * (OUTPUT_TAIL_CHARS + 100))

    job = require_job(manager, job_id)
    assert len(job.output_tail) == OUTPUT_TAIL_CHARS  # bounded
    assert job.output_seq == 10 + OUTPUT_TAIL_CHARS + 100  # counts everything

    text, _seq, gap = require_output(manager, job_id, cursor)
    assert gap is True, "a cursor whose bytes were evicted must be told"
    assert "A" not in text  # the evicted prefix is genuinely gone

    # A fresh cursor at the head is contiguous again: no false gap.
    assert require_output(manager, job_id, job.output_seq)[2] is False
    await manager.dispose()


@pytest.mark.asyncio
async def test_output_helpers_tolerate_unknown_and_empty() -> None:
    """An unknown id reads as None (not an exception): a runner draining a pipe
    must not die because retention already swept its row."""
    manager = AsyncJobManager()
    assert manager.read_output("nope") is None
    manager.append_output("nope", "ignored")  # must not raise
    job_id = manager.register("bash", "quiet", quick_runner)
    manager.append_output(job_id, "")  # empty write is a no-op, not a bump
    assert require_job(manager, job_id).output_seq == 0
    await manager.dispose()


@pytest.mark.asyncio
async def test_cancel_before_the_runner_starts_still_runs_cleanup() -> None:
    """A resource spawned before ``register`` is torn down even if the runner
    is never entered.

    ``register`` only ``ensure_future``s the runner, so a cancel landing in the
    SAME event-loop turn settles the row without the coroutine body — and
    therefore without its ``finally`` — ever running. Anything the caller had
    already spawned would outlive the job that was supposed to own it. Note the
    zero intervening awaits: that is the whole window.
    """
    manager = AsyncJobManager()
    torn_down: list[str] = []
    entered: list[str] = []

    async def runner(job_id, signal, report_progress):
        entered.append(job_id)
        await asyncio.sleep(30)
        return "never"

    job_id = manager.register(
        "bash", "spawned-already", runner, on_cancel=lambda: torn_down.append("cleaned")
    )
    await manager.cancel(job_id)  # no awaits in between: runner never stepped

    assert entered == [], "precondition: the runner must not have started"
    assert torn_down == ["cleaned"], "pre-start cleanup did not run"
    assert require_job(manager, job_id).status == "cancelled"
    await manager.dispose()


@pytest.mark.asyncio
async def test_cleanup_does_not_fire_once_the_runner_owns_the_resource() -> None:
    """Exactly one owner. Once the runner starts, its own teardown is
    authoritative and the pre-start hook must be dropped — firing both would
    kill the same process twice (or kill one a retry had just replaced)."""
    manager = AsyncJobManager()
    torn_down: list[str] = []
    started = asyncio.Event()

    async def runner(job_id, signal, report_progress):
        started.set()
        await asyncio.sleep(30)
        return "never"

    job_id = manager.register(
        "bash", "running", runner, on_cancel=lambda: torn_down.append("cleaned")
    )
    await started.wait()  # the runner is now entered and owns its resources
    await manager.cancel(job_id)
    assert torn_down == [], "pre-start cleanup ran for a job whose runner owned it"
    await manager.dispose()


@pytest.mark.asyncio
async def test_dispose_leaves_no_job_uncleaned() -> None:
    """Session teardown is the realistic trigger, and it must clean EVERY job.

    Each job is torn down by exactly one of the two owners, and which one is a
    scheduling detail rather than a guarantee: ``dispose`` awaits ``cancel``
    per job, and that await lets a later job's runner start, after which the
    runner's own ``finally`` is what cleans it. So this asserts the property
    that actually matters — every job cleaned, none cleaned twice — rather than
    pinning which path did it.
    """
    manager = AsyncJobManager()
    cleaned: list[str] = []

    def runner_for(name: str):
        async def runner(job_id, signal, report_progress):
            try:
                await asyncio.sleep(30)
            finally:
                cleaned.append(name)  # the runner-owned path
            return "never"

        return runner

    for name in ("a", "b", "c"):
        manager.register(
            "bash",
            name,
            runner_for(name),
            on_cancel=lambda name=name: cleaned.append(name),  # the pre-start path
        )
    await manager.dispose()
    # Let any runner that had started finish unwinding its `finally`.
    await asyncio.sleep(0.05)
    assert sorted(cleaned) == ["a", "b", "c"], "every job must be cleaned exactly once"


@pytest.mark.asyncio
async def test_a_raising_cleanup_does_not_break_the_cancel() -> None:
    """Teardown is best-effort: a hook that throws must not leave the job row
    un-settled or propagate out of ``cancel``."""
    manager = AsyncJobManager()

    async def runner(job_id, signal, report_progress):
        await asyncio.sleep(30)
        return "never"

    def _boom() -> None:
        raise RuntimeError("cleanup failed")

    job_id = manager.register("bash", "bad-cleanup", runner, on_cancel=_boom)
    assert await manager.cancel(job_id) is True
    assert require_job(manager, job_id).status == "cancelled"
    await manager.dispose()


@pytest.mark.asyncio
async def test_read_output_handles_a_cursor_from_another_job() -> None:
    """A ``since`` past the head (e.g. a cursor copied from a busier job) reads
    as 'nothing new' rather than slicing backwards into old output."""
    manager = AsyncJobManager()
    job_id = manager.register("bash", "quiet", quick_runner)
    manager.append_output(job_id, "hello")
    text, seq, gap = require_output(manager, job_id, 10_000)
    assert (text, gap) == ("", False)
    assert seq == 5
    await manager.dispose()
