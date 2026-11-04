"""Warm subscriptions reuse accounting structure while prices stay fresh."""

import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.jobs import AsyncJob, AsyncJobManager
from local_operator.harness.types import Usage
from local_operator.session import frontend_state as fs
from local_operator.session.session import Session


def _components(usage):
    """The cost components of a usage the fixture always populates.

    ``AsyncJob.usage`` and ``Usage.cost_components`` are Optional in the model, so
    pyright rejects a bare chain. Asserting here narrows them for the type checker
    AND turns a fixture that lost its components into a clear failure rather than an
    ``AttributeError`` or an index error three lines later.
    """
    assert usage is not None and usage.cost_components is not None
    return usage.cost_components


def _fixture():
    manager = AsyncJobManager()
    job = AsyncJob(
        id="projection-job",
        label="projection",
        type="task",
        status="completed",
        start_time=time.time(),
        settled_at=time.time(),
        usage=Usage(cost_components=[Usage(input_tokens=20, usd_cost=0.001) for _ in range(40)]),
    )
    manager.restore([job])
    session = SimpleNamespace(jobs=manager)
    store = fs.FrontendStateStore(fs.FrontendSessionState(session_id="projection", epoch="e"))
    return manager, job, session, store


def test_warm_refresh_does_not_refreeze_accounting(monkeypatch):
    manager, job, session, store = _fixture()
    store.refresh_from_session(session)
    counts = {"price": 0, "freeze": 0}
    price, freeze = fs.job_subtree_cost, fs._freeze_usage

    def counted_price(*args, **kwargs):
        counts["price"] += 1
        return price(*args, **kwargs)

    def counted_freeze(value):
        counts["freeze"] += 1
        return freeze(value)

    monkeypatch.setattr(fs, "job_subtree_cost", counted_price)
    monkeypatch.setattr(fs, "_freeze_usage", counted_freeze)
    store.refresh_from_session(session)
    # The price memo can warm independently of a job mutation; only immutable
    # accounting structure is reused, never the derived dollar/knowledge pair.
    assert counts == {"price": 1, "freeze": 0}


def test_usage_and_progress_notifications_invalidate_before_observers():
    manager, job, session, store = _fixture()
    store.refresh_from_session(session)
    prior = store.state.jobs[0]
    observed = []
    manager.subscribe_changes(lambda: observed.append(store.refresh_from_session(session).jobs[0]))
    _components(job.usage)[0].usd_cost = 0.025
    job.latest_details = {"progress": "updated"}
    manager.note_usage_changed()
    assert _components(observed[-1].usage)[0].usd_cost == 0.025
    assert observed[-1].latest_details["progress"] == "updated"
    assert _components(prior.usage)[0].usd_cost == 0.001


def test_child_accounting_invalidates_parent_row_before_listener():
    manager, job, session, store = _fixture()
    child = AsyncJobManager()
    manager.attach_child_manager(job.id, child)
    store.refresh_from_session(session)
    revision = job.projection_revision
    observed = []
    manager.subscribe_accounting(lambda seen: observed.append(job.projection_revision))
    child.note_usage_changed()
    assert observed and observed[-1] > revision


def test_unversioned_extension_rows_are_rebuilt():
    manager, job, session, store = _fixture()
    # A source outside the manager's notification contract must not be cached.
    job._projection_source = None
    store.refresh_from_session(session)
    _components(job.usage)[0].usd_cost = 0.25
    store.refresh_from_session(session)
    assert _components(store.state.jobs[0].usage)[0].usd_cost == 0.25


def test_independent_paint_pricing_changes_without_row_mutation(monkeypatch):
    manager, job, session, store = _fixture()
    # Mixed element types by design (None/float and bool), so typed as Any.
    quote: list[Any] = [None, True]
    monkeypatch.setattr(fs, "job_subtree_cost", lambda *a, **k: tuple(quote))
    store.refresh_from_session(session)
    before = store.state.jobs[0]
    revision = job.projection_revision
    assert before.direct_cost is None
    quote[:] = [0.17, False]
    store.refresh_from_session(session)
    after = store.state.jobs[0]
    assert job.projection_revision == revision
    assert after.usage == before.usage
    assert after.direct_cost == 0.17
    assert after.direct_cost_knowledge == fs.CostKnowledge.EXACT
    quote[:] = [0.08, True]
    store.refresh_from_session(session)
    assert store.state.jobs[0].direct_cost_knowledge == fs.CostKnowledge.PARTIAL


def test_live_memo_releases_rows_and_rebinds_epochs():
    manager, job, session, store = _fixture()
    store.refresh_from_session(session)
    memo = store._live_memo()
    assert memo._rows
    memo.rebind("new-owner")
    assert not memo._rows
    store.refresh_from_session(session)
    memo.retain(set())
    assert not memo._rows


@pytest.mark.asyncio
async def test_runner_entry_invalidates_registration_snapshot():
    import asyncio

    manager = AsyncJobManager()
    started = asyncio.Event()
    release = asyncio.Event()

    async def run(job_id, signal, progress):
        started.set()
        await release.wait()
        return "done"

    jid = manager.register(type="task", label="suspended", run=run)
    session = SimpleNamespace(jobs=manager)
    store = fs.FrontendStateStore(fs.FrontendSessionState(session_id="runner", epoch="e"))
    store.refresh_from_session(session)
    assert store.state.jobs[0].started_at is None
    try:
        await started.wait()
        store.refresh_from_session(session)
        assert store.state.jobs[0].started_at is not None
        running = manager.get(jid)
        assert running is not None and running.status == "running"
    finally:
        release.set()
        await asyncio.gather(*list(manager._tasks.values()))


def test_real_subscribe_boundary_reuses_warm_usage(monkeypatch):
    manager, job, session, store = _fixture()
    session._frontend_state_store = store
    # A stand-in on purpose: the real ``Session`` needs a provider, a harness and
    # a transcript, none of which the subscribe boundary reads. The cast says so
    # to the type checker instead of hiding it.
    as_session = cast(Session, session)
    first = Session.subscribe_frontend(as_session, lambda event: None)
    original = fs._freeze_usage
    count = 0

    def freeze(value):
        nonlocal count
        count += 1
        return original(value)

    monkeypatch.setattr(fs, "_freeze_usage", freeze)
    second = Session.subscribe_frontend(as_session, lambda event: None)
    try:
        assert count == 0
        assert second.sync.snapshot.jobs[0].usage == first.sync.snapshot.jobs[0].usage
    finally:
        second.unsubscribe()
        first.unsubscribe()


def test_live_lineage_is_reapplied_on_cache_hit():
    manager, job, session, store = _fixture()
    node = SimpleNamespace(parent_job_id="parent-one", session_id="child", live=False)
    session._subagent_comms = SimpleNamespace(node=lambda job_id: node)
    store.refresh_from_session(session)
    assert store.state.jobs[0].parent_job_id == "parent-one"
    revision = job.projection_revision
    node.parent_job_id = "parent-two"
    store.refresh_from_session(session)
    assert job.projection_revision == revision
    assert store.state.jobs[0].parent_job_id == "parent-two"
    assert "projection_revision" not in job.model_dump()
    assert "_projection_source" not in job.model_dump()
