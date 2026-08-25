"""Canonical full-TUI state round trips, ordering and authoritative semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.jobs import AsyncJob
from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    Message,
    ModelSpec,
    Usage,
)
from local_operator.session.frontend_state import (
    CommandScope,
    CostKnowledge,
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendUpdate,
    FrontendUsage,
    JobState,
    SnapshotJobs,
    TodoItemState,
    TodoPhaseState,
    WakeState,
)


def _spec(window: int = 1_000_000) -> ModelSpec:
    return ModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        display_name="GPT 5.6 Solid",
        context_window=window,
        max_output_tokens=128_000,
        supports_images=True,
        supports_tools=True,
        supports_prompt_cache=True,
        supports_responses_api=True,
        supports_sampling_params=False,
        reasoning=True,
        reasoning_effort="high",
        reasoning_efforts=("low", "medium", "high"),
    )


def _state(**changes) -> FrontendSessionState:  # noqa: ANN003
    state = FrontendSessionState(
        session_id="s1",
        epoch="owner-a",
        cwd="/repo",
        conversation_title="Canonical state",
        conversation_title_user_set=True,
        goal="ship parity",
        active_agent="coder",
        active_team="lopdev",
        selected_model=FrontendModelSpec.model_validate(_spec().model_dump()),
        effective_model=FrontendModelSpec.model_validate(_spec().model_dump()),
        last_usage=FrontendUsage(
            input_tokens=400_000,
            output_tokens=2_000,
            context_tokens=402_000,
            usd_cost=1.25,
            provider="openai",
            model_id="gpt-5.6-sol",
        ),
        context_tokens=402_000,
        context_is_estimate=False,
        context_window=1_000_000,
        cumulative_parent_cost=1.25,
        cost_knowledge=CostKnowledge.EXACT,
        jobs=[JobState(id="j1", type="task", label="reviewer", status="running")],
        todos=[TodoPhaseState(name="Build", items=[TodoItemState(text="wire state")])],
        wakes=[WakeState(id="w1", message="check build", next_due_at=1_800_000_000_000)],
    )
    return state.model_copy(update=changes)


def test_state_json_roundtrip_preserves_full_model_usage_and_future_fields() -> None:
    payload = _state().model_dump(mode="json")
    payload["future_owner_field"] = {"new": True}
    payload["selected_model"]["future_model_field"] = "kept"
    payload["last_usage"]["future_usage_field"] = 7

    restored = FrontendSessionState.model_validate(payload)
    wire = restored.model_dump(mode="json")

    assert restored.context_window == 1_000_000
    assert restored.selected_model is not None
    assert restored.selected_model.context_window == _spec().context_window
    assert restored.selected_model.display_name == _spec().display_name
    assert restored.last_usage is not None
    assert restored.last_usage.usd_cost == 1.25
    assert wire["future_owner_field"] == {"new": True}
    assert wire["selected_model"]["future_model_field"] == "kept"
    assert wire["last_usage"]["future_usage_field"] == 7


def test_missing_model_and_cost_remain_explicit_unknowns() -> None:
    restored = FrontendSessionState.model_validate({"session_id": "s1", "epoch": "e"})
    assert restored.selected_model is None
    assert restored.context_window is None
    assert restored.cumulative_cost is None
    assert restored.cost_knowledge is CostKnowledge.UNKNOWN


def test_atomic_join_at_every_sequence_has_exact_suffix() -> None:
    store = FrontendStateStore(_state())
    all_updates: list[FrontendUpdate] = []
    store.subscribe(all_updates.append)
    for value in range(1, 9):
        store.mutate(context_tokens=402_000 + value)

    for join_after in range(0, 9):
        replay = FrontendStateStore(_state())
        for update in all_updates[:join_after]:
            replay.apply_update(update)
        delivered: list[int] = []
        subscription = replay.subscribe(lambda update: delivered.append(update.sequence))
        for update in all_updates[join_after:]:
            replay.apply_update(update)
        assert subscription.sync.sequence == join_after
        assert delivered == list(range(join_after + 1, 9))


def test_usage_join_and_turn_end_does_not_double_count_mixed_calls() -> None:
    store = FrontendStateStore(_state(cumulative_parent_cost=1.25))
    first = Usage(
        input_tokens=10,
        output_tokens=2,
        context_tokens=410_000,
        usd_cost=0.4,
        provider="openrouter",
        model_id="fallback-a",
    )
    second = Usage(
        input_tokens=20,
        output_tokens=3,
        context_tokens=430_000,
        usd_cost=0.6,
        provider="openai",
        model_id="gpt-5.6-sol",
    )
    session = SimpleNamespace(effective_model=_spec())
    store.observe_event(session, AgentStartEvent(generation=2))
    store.observe_event(
        session,
        AgentEndEvent(
            messages=[Message.assistant("a", usage=first), Message.assistant("b", usage=second)]
        ),
    )

    state = store.state
    assert state.context_tokens == 430_000
    assert state.cumulative_parent_cost == pytest.approx(2.25)
    assert [(u.provider, u.model_id, u.usd_cost) for u in state.usage_components[-2:]] == [
        ("openrouter", "fallback-a", 0.4),
        ("openai", "gpt-5.6-sol", 0.6),
    ]


def test_slash_capabilities_classify_every_advertised_command_and_images() -> None:
    from local_operator.session.frontend_state import _slash_capabilities
    from local_operator.tui.app import SLASH_COMMANDS

    capabilities = {value.command: value for value in _slash_capabilities()}
    assert set(capabilities) == {command.name for command in SLASH_COMMANDS}
    assert all(value.scope is not CommandScope.UNAVAILABLE for value in capabilities.values())
    assert capabilities["context"].scope is CommandScope.AUTHORITATIVE_SESSION
    assert capabilities["mcp"].scope is CommandScope.FRONTEND_LOCAL
    assert capabilities["btw"].scope is CommandScope.FRONTEND_LOCAL
    assert capabilities["agent"].supports_images is True
    assert capabilities["team"].supports_images is True


def test_compaction_semantics_replace_context_and_preserve_lifetime_cost() -> None:
    store = FrontendStateStore(_state(cumulative_parent_cost=8.5, context_tokens=900_000))
    store.mutate(context_tokens=120_000, context_is_estimate=True)
    assert store.state.context_tokens == 120_000
    assert store.state.context_is_estimate is True
    assert store.state.cumulative_parent_cost == 8.5


def test_real_async_job_roundtrips_progress_trajectory_and_accounting() -> None:
    job = AsyncJob(
        id="child-1",
        type="task",
        label="reviewer",
        start_time=10.0,
        started_at=11.0,
        latest_details={"progress": "reviewing diff"},
        trajectory=[{"type": "message_start", "message": {"role": "assistant"}}],
        prompt="Review the change",
        model_label="anthropic/claude-fable-5",
        context_window=1_000_000,
        usage=Usage(input_tokens=12, output_tokens=3, context_tokens=42_000, usd_cost=0.25),
        agent_role="reviewer",
        effort="hi",
    )
    state = JobState.from_job(job)
    restored = JobState.model_validate_json(state.model_dump_json())

    assert restored.latest_details == {"progress": "reviewing diff"}
    assert restored.trajectory == job.trajectory
    assert restored.prompt == "Review the change"
    assert restored.started_at == 11.0
    assert restored.usage is not None and restored.usage.usd_cost == 0.25
    snapshot = SnapshotJobs([restored]).get("child-1")
    assert snapshot is not None and snapshot.trajectory == job.trajectory
