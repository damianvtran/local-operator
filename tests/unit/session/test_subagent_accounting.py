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


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("swept", [True, False])
@pytest.mark.parametrize("ledger", ["new", "malformed", "legacy", "empty"])
async def test_cold_facade_restores_sidecar_ledger_without_parent_checkpoint(
    tmp_path, monkeypatch, checkpoint, swept, ledger
):
    import json

    from local_operator.session.frontend_state import FRONTEND_CHECKPOINT_CUSTOM_TYPE
    from local_operator.session.remote import RemoteSession
    from local_operator.session.session import SUBAGENT_ROSTER_SIDECAR
    from local_operator.session.transcript import Transcript

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint", lambda *_: pytest.fail("cold discovery")
    )
    directory = tmp_path / "sessions" / "coldledger01"
    directory.mkdir(parents=True)
    transcript = Transcript(directory)
    if checkpoint:
        state = FrontendSessionState(
            epoch="old",
            session_id="coldledger01",
            subagent_cost=0.125,
            subagent_cost_knowledge=CostKnowledge.EXACT,
        )
        await transcript.append_custom(
            FRONTEND_CHECKPOINT_CUSTOM_TYPE, {"state": state.model_dump(mode="json")}
        )
    payload = {
        "version": 1,
        "generation": 1,
        "records": [],
        "jobs": (
            []
            if swept
            else [
                {
                    "id": "new",
                    "type": "task",
                    "status": "completed",
                    "usage": Usage(usd_cost=0.25).model_dump(),
                }
            ]
        ),
    }
    if ledger != "legacy":
        payload["accounting"] = {
            "new": [Usage(estimated_usd_cost=0.25).model_dump()],
            "empty": [],
            "malformed": [{"estimated_usd_cost": -1}],
        }[ledger]
    (directory / SUBAGENT_ROSTER_SIDECAR).write_text(json.dumps(payload))

    async def never():
        pytest.fail("cold restore started owner")

    viewer = await RemoteSession.cold(
        "coldledger01", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=never
    )
    try:
        assert viewer.is_cold
        expected = (
            0.25
            if ledger == "new"
            else None if ledger == "empty" else 0.125 if checkpoint else None
        )
        assert viewer.frontend_state.cumulative_cost == expected
        assert len(viewer.frontend_state.jobs) == (0 if swept else 1)
    finally:
        await viewer.dispose()


@pytest.mark.parametrize(
    "provider,model_id,counts",
    [
        (
            "anthropic",
            "claude-fable-5-1",
            dict(
                input_tokens=116,
                output_tokens=52767,
                cache_read_tokens=4282533,
                cache_write_tokens=164387,
                context_tokens=113735,
            ),
        ),
        (
            "openai",
            "gpt-6-astra",
            dict(
                input_tokens=5935061,
                output_tokens=7159,
                cache_read_tokens=4740864,
                context_tokens=179608,
            ),
        ),
    ],
)
def test_both_table_priced_models_preserve_cache_semantics_offline(
    monkeypatch, provider, model_id, counts
):
    from local_operator.model.configure import cost_for_usage
    from local_operator.model.registry import ModelInfo

    # Controlled test prices, NOT vendor-rate claims. The actual model/cache
    # shapes exercise Anthropic's separate cache buckets versus OpenAI's cached
    # subset of input, through the shared pricing helper rather than a workaround.
    info = ModelInfo(
        id=model_id,
        name=model_id,
        description="controlled pricing fixture",
        input_price=2,
        output_price=10,
        cache_reads_price=0.2,
        cache_writes_price=2.5,
    )
    monkeypatch.setattr("local_operator.tui.costs._resolve_for_paint", lambda *_: info)
    usage = Usage(provider=provider, model_id=model_id, **counts)
    expected = cost_for_usage(provider, info, usage)
    plain_input = (
        counts["input_tokens"]
        if provider == "anthropic"
        else counts["input_tokens"] - counts["cache_read_tokens"]
    )
    assert expected == pytest.approx(
        (
            plain_input * 2
            + counts["output_tokens"] * 10
            + counts["cache_read_tokens"] * 0.2
            + counts.get("cache_write_tokens", 0) * 2.5
        )
        / 1_000_000
    )
    job = AsyncJob(
        start_time=0, id="priced", type="task", label="priced", model_label=f"{provider}/{model_id}"
    )
    _accumulate_usage(job, usage)
    assert job.usage is not None
    persisted = Usage.model_validate_json(job.usage.model_dump_json())
    assert persisted.cost_components[0].usd_cost is None
    monkeypatch.setattr(
        "local_operator.tui.costs._resolve_for_paint", lambda *_: pytest.fail("offline discovery")
    )
    assert cost_summary(persisted.cost_components, recorded_only=True) == (expected, False)
    _accumulate_usage(job, Usage(provider=provider, model_id=model_id, usd_cost=0.125))
    assert job_stats(JobState.from_job(job)).cost == pytest.approx(expected + 0.125)
    unknown = Usage(provider="test", model_id="unpriced", input_tokens=10)
    assert cost_summary([*persisted.cost_components, unknown], recorded_only=True) == (
        expected,
        True,
    )
