"""Money crosses runtime, retention and restart boundaries, not pricing caches."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from local_operator.harness.jobs import AsyncJob, AsyncJobManager
from local_operator.harness.subagent import _accumulate_usage
from local_operator.harness.types import Usage
from local_operator.session.frontend_state import (
    CostKnowledge,
    FrontendSessionState,
    JobState,
    _folded_components,
    _ledger_cost,
)
from local_operator.tui.costs import cost_summary, turn_cost
from local_operator.tui.widgets.subagent_panel import job_stats


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_estimates_cannot_enter_durable_usage(bad):
    with pytest.raises(ValueError):
        Usage(estimated_usd_cost=bad)


def test_recorded_estimate_is_not_a_receipt_and_does_not_reprice(monkeypatch):
    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint", lambda *_: pytest.fail("repriced")
    )
    usage = Usage(provider="test", model_id="dynamic", estimated_usd_cost=0.125)
    assert usage.usd_cost is None
    assert turn_cost("test/dynamic", usage) == 0.125
    usage.usd_cost = 0.2
    assert turn_cost("test/dynamic", usage) == 0.2


def test_relay_prices_leaf_calls_before_aggregation(monkeypatch):
    from local_operator.model.registry import ModelInfo

    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint",
        lambda *_: ModelInfo(id="dynamic", name="Dynamic", description="test", input_price=2),
    )
    job = AsyncJob(
        start_time=0, id="priced", type="task", label="priced", model_label="test/dynamic"
    )
    _accumulate_usage(job, Usage(input_tokens=1000))
    _accumulate_usage(job, Usage(input_tokens=2000))
    assert job.usage is not None
    assert job.usage.estimated_usd_cost is None
    assert job.usage.usd_cost is None
    assert [part.estimated_usd_cost for part in job.usage.cost_components] == [0.002, 0.004]
    assert all(part.usd_cost is None for part in job.usage.cost_components)


def test_known_unknown_and_free_survive_wire_folding(monkeypatch):
    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint",
        lambda *_: SimpleNamespace(input_price=0, output_price=0),
    )
    components = [
        Usage(provider="test", model_id="m", usd_cost=0.125),
        Usage(provider="test", model_id="m", estimated_usd_cost=0.25),
        Usage(provider="test", model_id="m", estimated_usd_cost=0),
        Usage(provider="test", model_id="m", input_tokens=100),
    ]
    folded = _folded_components(components)
    assert len(folded) == 3
    assert cost_summary(folded) == (0.375, True)
    row = JobState.from_job(
        AsyncJob(
            start_time=0,
            id="mixed",
            type="task",
            label="mixed",
            model_label="test/m",
            usage=Usage(cost_components=folded),
        )
    )
    wire = JobState.model_validate_json(row.model_dump_json())
    assert wire.direct_cost == 0.375
    assert wire.direct_cost_knowledge == CostKnowledge.PARTIAL
    assert job_stats(wire).cost_partial
    assert cost_summary([components[2]]) == (0, False)
    assert cost_summary([components[3]]) == (None, True)

    from local_operator.session.session import _subagent_job_row

    durable = _subagent_job_row(
        AsyncJob(
            start_time=0,
            id="mixed",
            type="task",
            label="mixed",
            model_label="test/m",
            usage=Usage(cost_components=folded),
        )
    )
    # RemoteSession's daemonless sidecar overlay uses this exact validation
    # path, without an owner or any permission to discover model prices.
    cold = JobState.model_validate(durable)
    assert cold.usage is not None
    assert cost_summary(cold.usage.cost_components, recorded_only=True) == (0.375, True)


@pytest.mark.asyncio
async def test_canonical_rows_never_discover_in_viewer_thread(monkeypatch):
    monkeypatch.setattr(
        "local_operator.tui.costs.job_cost", lambda *_args, **_kwargs: pytest.fail("viewer priced")
    )
    row = JobState(
        id="child",
        type="task",
        usage=Usage(context_tokens=113735),
        direct_cost=0.412,
        direct_cost_knowledge=CostKnowledge.EXACT,
    )
    stats = await asyncio.to_thread(job_stats, row)
    assert stats.cost == 0.412
    assert stats.context_tokens == 113735
    unknown = row.model_copy(
        update={"direct_cost": None, "direct_cost_knowledge": CostKnowledge.UNKNOWN}
    )
    assert (await asyncio.to_thread(job_stats, unknown)).cost is None


@pytest.mark.asyncio
async def test_whole_ledger_includes_live_descendant_and_folded_attempts():
    manager = AsyncJobManager()
    child = AsyncJobManager()
    release = asyncio.Event()
    started = asyncio.Queue()

    async def run(*_):
        started.put_nowait(True)
        await release.wait()

    root = manager.register("task", "root", run)
    nested = child.register("task", "nested", run)
    await started.get()
    await started.get()
    root_row, nested_row = manager.get(root), child.get(nested)
    assert root_row is not None and nested_row is not None
    root_row.usage = Usage(usd_cost=1, provider="test", model_id="m")
    nested_row.usage = Usage(usd_cost=2, provider="test", model_id="m")
    manager.attach_child_manager(root, child)
    child.note_usage_changed()
    assert _ledger_cost(SimpleNamespace(jobs=manager))["subagent_cost"] == 3
    release.set()
    await child.dispose()
    await manager.dispose()


def test_authoritative_checkpoint_replaces_retained_rows():
    manager = AsyncJobManager()
    row = AsyncJob(
        start_time=0,
        id="retained",
        type="task",
        label="retained",
        status="completed",
        model_label="test/m",
        usage=Usage(usd_cost=1),
    )
    manager.restore([row])
    manager.restore_accounting([Usage(provider="test", model_id="m", usd_cost=4)])
    assert cost_summary(manager.accounting_components()) == (4, False)
    state = FrontendSessionState(
        epoch="test",
        session_id="test",
        cumulative_parent_cost=2,
        child_costs={"obsolete": 99},
        **_ledger_cost(SimpleNamespace(jobs=manager)),
    )
    assert state.cumulative_cost == 6


def test_mixed_child_unknown_keeps_parent_lower_bound():
    state = FrontendSessionState(
        epoch="test",
        session_id="test",
        cumulative_parent_cost=1,
        cost_knowledge=CostKnowledge.EXACT,
        subagent_cost=0.125,
        subagent_cost_knowledge=CostKnowledge.PARTIAL,
    )
    assert state.cumulative_cost == 1.125
    assert state.cumulative_cost_knowledge == CostKnowledge.PARTIAL


@pytest.mark.asyncio
async def test_rendered_footer_uses_whole_owner_ledger():
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        state = FrontendSessionState(
            epoch="money",
            session_id="money",
            cumulative_parent_cost=0,
            cost_knowledge=CostKnowledge.EXACT,
            child_costs={"estimated": 0.006},
            subagent_cost=0.131,
            subagent_cost_knowledge=CostKnowledge.PARTIAL,
        )
        app._apply_frontend_state(state)
        await pilot.pause()
        assert app._spend_total() == 0.131
        assert "0.131" in app._spend_text()
        assert app._spend_is_floor
        assert app.screen.virtual_size == app.screen.size
        assert not app.screen.show_vertical_scrollbar
