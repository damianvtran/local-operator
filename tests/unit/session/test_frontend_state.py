"""Canonical full-TUI state round trips, ordering and authoritative semantics."""

from __future__ import annotations

import copy
import pickle
import threading
import time
from collections import deque, namedtuple
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.jobs import AsyncJob
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSpec,
    SubagentProgressEvent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    Usage,
)
from local_operator.session.frontend_state import (
    LIVE_EVENT_END_ROWS_MAX,
    CommandScope,
    CostKnowledge,
    FrontendModelSpec,
    FrontendSessionState,
    FrontendStateStore,
    FrontendSync,
    FrontendUpdate,
    FrontendUsage,
    JobState,
    SnapshotJobs,
    TodoItemState,
    TodoPhaseState,
    WakeState,
    sync_wire_payload,
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


def test_snapshot_jobs_preserve_immutable_mapping_and_sequence_interfaces() -> None:
    state = FrontendStateStore(
        _state(
            jobs=[
                JobState(
                    id="child",
                    type="task",
                    latest_details={"progress": "reading files"},
                    trajectory=[{"type": "message_update", "delta": "hello"}],
                )
            ]
        )
    ).state
    jobs = SnapshotJobs(state.jobs)

    for job in [*jobs.list(), jobs.get("child")]:
        assert job is not None
        assert isinstance(job.latest_details, Mapping)
        assert job.latest_details.get("progress") == "reading files"
        with pytest.raises((AttributeError, TypeError)):
            job.latest_details["progress"] = "corrupted"  # type: ignore[index]
        assert isinstance(job.trajectory, Sequence)
        assert isinstance(job.trajectory[0], Mapping)
        assert job.trajectory[0].get("delta") == "hello"


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


def test_usage_join_folds_cache_write_ttl_split() -> None:
    """The frontend's turn-end usage join folds the 5m/1h cache-write split
    wherever it folds ``cache_write_tokens`` (review F4) — the split prices
    differently (1.25x vs 2x base), so a join that dropped it would read as
    zero from its first reader."""
    from local_operator.session.frontend_state import _aggregate_usage

    aggregate = _aggregate_usage(
        [
            Usage(
                input_tokens=1,
                cache_write_tokens=1_000,
                cache_write_5m_tokens=1_000,
                context_tokens=200_000,
            ),
            Usage(
                input_tokens=1,
                cache_write_tokens=3_000,
                cache_write_1h_tokens=3_000,
                context_tokens=210_000,
            ),
        ]
    )
    assert aggregate.cache_write_tokens == 4_000
    assert aggregate.cache_write_5m_tokens == 1_000
    assert aggregate.cache_write_1h_tokens == 3_000
    assert aggregate.context_tokens == 210_000


def test_subagent_progress_defers_job_publication_to_the_coalescer(monkeypatch) -> None:
    """A raw progress edge neither rescans nor publishes canonical state itself.

    The manager's 50 ms ``refresh_jobs`` callback is the one owner publication;
    followers receive that same update, so owner and follower visual cadence are
    identical without a duplicate full-session fallback on every boundary.
    """
    job = AsyncJob(
        id="child",
        type="task",
        status="running",
        start_time=1.0,
        label="child",
        latest_details={"progress": "reading files"},
    )
    session = SimpleNamespace(
        jobs=SimpleNamespace(list=lambda: [job]),
        _subagent_comms=None,
        model=None,
        effective_model=None,
        session_id="s1",
        cwd="/repo",
        queued_steering=lambda: [],
        conversation_name="Canonical state",
        goal="",
        active_agent="",
        active_team_name="",
        wake_scheduler=None,
        mcp_manager=None,
        mcp_startup=None,
    )
    store = FrontendStateStore(_state(jobs=[]))
    updates: list[FrontendUpdate] = []
    store.subscribe(updates.append)
    rescans = 0
    original = store.refresh_from_session

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal rescans
        rescans += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "refresh_from_session", counted)
    assert (
        store.observe_event(
            session,
            SubagentProgressEvent(job_id="child", label="child", progress="reading files"),
        )
        is None
    )
    assert (rescans, updates) == (0, [])

    update = store.refresh_jobs(session)
    assert update is not None
    assert len(updates) == 1
    assert updates[0].changes["jobs"][0]["latest_details"] == {"progress": "reading files"}


def test_slash_capabilities_classify_every_advertised_command_and_images() -> None:
    from local_operator.session.frontend_state import _slash_capabilities
    from local_operator.tui.app import SLASH_COMMANDS

    capabilities = {value.command: value for value in _slash_capabilities()}
    assert set(capabilities) == {command.name for command in SLASH_COMMANDS}
    assert all(value.scope is not CommandScope.UNAVAILABLE for value in capabilities.values())
    assert capabilities["context"].scope is CommandScope.AUTHORITATIVE_SESSION
    # ``/mcp`` is advertised authoritative because its grant subcommands route
    # to the owner; the follower's dispatch keeps the BARE listing local from
    # its canonical snapshot facade (see ``_run_slash_command``).
    assert capabilities["mcp"].scope is CommandScope.AUTHORITATIVE_SESSION
    assert capabilities["btw"].scope is CommandScope.FRONTEND_LOCAL
    assert capabilities["agent"].supports_images is True
    assert capabilities["team"].supports_images is True


def test_compaction_semantics_replace_context_and_preserve_lifetime_cost() -> None:
    store = FrontendStateStore(_state(cumulative_parent_cost=8.5, context_tokens=900_000))
    store.mutate(context_tokens=120_000, context_is_estimate=True)
    assert store.state.context_tokens == 120_000
    assert store.state.context_is_estimate is True
    assert store.state.cumulative_parent_cost == 8.5


def test_post_compaction_agent_end_keeps_settled_occupancy_and_provider_bill() -> None:
    """A held boundary is emitted after automatic compaction, so its message
    usage is older than the context it closes. Occupancy follows the stamped
    post-pass level while billing still consumes the provider's real receipt.
    """
    store = FrontendStateStore(
        _state(
            cumulative_parent_cost=8.5,
            context_tokens=590_400,
            last_usage=Usage(context_tokens=590_400),
        )
    )
    usage = Usage(context_tokens=590_400, input_tokens=590_400, output_tokens=1_000, usd_cost=2.5)
    session = SimpleNamespace(effective_model=_spec(), restored_usage=lambda: usage)

    store.observe_event(
        session,
        AgentEndEvent(
            messages=[Message.assistant("done", usage=usage)],
            context_tokens=131_100,
        ),
    )

    assert store.state.context_tokens == 131_100
    assert store.state.context_is_estimate is True
    assert store.state.last_usage is not None
    assert store.state.last_usage.context_tokens == 590_400
    assert store.state.cumulative_parent_cost == pytest.approx(11.0)


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


def test_two_subscribers_never_hold_different_state_at_one_sequence() -> None:
    """N3: a second join must not silently rewrite state under the same number.

    The old join path replaced state via ``initial=True`` without a sequence
    bump, so subscriber 1 held ``seq N / team-x`` while subscriber 2 received
    ``seq N / team-y`` — the exact divergence the client's exact-`+1` gap
    check exists to rule out.
    """
    store = FrontendStateStore(_state(active_team="team-x"))
    first_updates: list[FrontendUpdate] = []
    first = store.subscribe(first_updates.append)

    # The publishing path (what subscribe_frontend now uses) consumes a
    # sequence and notifies the existing subscriber before the second join.
    update = store.mutate(active_team="team-y")
    assert update is not None and update.sequence == first.sync.sequence + 1
    assert [u.sequence for u in first_updates] == [update.sequence]

    second = store.subscribe(lambda _u: None)
    assert second.sync.sequence == update.sequence
    assert second.sync.snapshot.active_team == "team-y"


def test_large_job_snapshots_share_only_immutable_retained_events() -> None:
    """The fast snapshot boundary cannot expose a mutation path into canonical state."""
    trajectory = [{"type": "message_update", "delta": "x" * 80} for _ in range(500)]
    jobs = [
        JobState(id=f"child-{index}", type="task", trajectory=trajectory) for index in range(100)
    ]
    store = FrontendStateStore(_state(jobs=jobs))

    first = store.state
    second = store.state

    assert first is not second
    assert first.jobs is not second.jobs
    assert first.jobs[0] is not second.jobs[0]
    assert first.jobs[0].trajectory is second.jobs[0].trajectory
    first.sequence = 99
    with pytest.raises((AttributeError, TypeError)):
        first.jobs.append(JobState(id="injected", type="task"))
    with pytest.raises((AttributeError, TypeError)):
        first.jobs[0].trajectory.append({"type": "notice"})
    with pytest.raises((AttributeError, TypeError)):
        first.jobs[0].trajectory[0]["delta"] = "corrupted"
    assert store.state.sequence != 99
    assert len(store.state.jobs) == 100
    assert len(store.state.jobs[0].trajectory) == 500
    assert store.state.jobs[0].trajectory[0]["delta"] != "corrupted"


def test_shared_job_usage_descendants_and_future_extras_are_immutable() -> None:
    """Every nested JobState value is safe to share, including unknown future fields."""
    job = JobState.model_validate(
        {
            "id": "child",
            "type": "task",
            "usage": Usage(input_tokens=4, output_tokens=2).model_dump(mode="json"),
            "descendant_usage": [
                FrontendUsage(input_tokens=8, output_tokens=3).model_dump(mode="json")
            ],
            "future_payload": {"nested": [1]},
            "future_tags": {"alpha", "beta"},
            "future_queue": deque([{"nested": [1]}]),
            "future_bytes": bytearray(b"abc"),
        }
    )
    store = FrontendStateStore(_state(jobs=[job]))
    snapshot = store.state

    assert snapshot.jobs[0].usage is not None
    with pytest.raises(Exception, match="frozen"):
        snapshot.jobs[0].usage.input_tokens = 99
    with pytest.raises(Exception, match="frozen"):
        snapshot.jobs[0].descendant_usage[0].output_tokens = 99
    future_payload = getattr(snapshot.jobs[0], "future_payload")
    with pytest.raises((AttributeError, TypeError)):
        future_payload["nested"].append(2)
    future_tags = getattr(snapshot.jobs[0], "future_tags")
    with pytest.raises(AttributeError):
        future_tags.add("corrupted")
    future_queue = getattr(snapshot.jobs[0], "future_queue")
    with pytest.raises((AttributeError, TypeError)):
        future_queue.append({"nested": [2]})
    assert getattr(snapshot.jobs[0], "future_bytes") == b"abc"

    canonical = store.state.jobs[0]
    assert canonical.usage is not None and canonical.usage.input_tokens == 4
    assert canonical.descendant_usage[0].output_tokens == 3
    assert getattr(canonical, "future_payload") == {"nested": [1]}
    assert getattr(canonical, "future_tags") == {"alpha", "beta"}
    assert getattr(canonical, "future_queue") == [{"nested": [1]}]
    assert getattr(canonical, "future_bytes") == b"abc"
    dumped = canonical.model_dump(mode="json")
    assert set(dumped["future_tags"]) == {"alpha", "beta"}
    assert dumped["future_queue"] == [{"nested": [1]}]
    assert dumped["future_bytes"] == "abc"
    restored = JobState.model_validate(dumped)
    restored_tags = getattr(restored, "future_tags")
    assert restored_tags == ["alpha", "beta"] or restored_tags == ["beta", "alpha"]
    assert store.state.sequence == 0


def test_structured_future_extras_normalize_to_closed_immutable_values() -> None:
    @dataclass
    class DataclassPayload:
        nested: list[int]

    NamedPayload = namedtuple("NamedPayload", "nested")

    class PydanticPayload(FrontendUsage):
        nested: list[int]

    cases = [
        (DataclassPayload([1]), lambda payload: payload["nested"]),
        (NamedPayload([1]), lambda payload: payload[0]),
        (PydanticPayload(nested=[1]), lambda payload: payload.nested),
    ]
    for value, nested in cases:
        store = FrontendStateStore(
            _state(
                jobs=[
                    JobState.model_validate(
                        {"id": "child", "type": "task", "future_payload": value}
                    )
                ]
            )
        )
        payload = getattr(store.state.jobs[0], "future_payload")
        assert nested(payload) == [1]
        with pytest.raises((AttributeError, TypeError)):
            nested(payload).append(2)
        assert store.state.sequence == 0
        assert nested(getattr(store.state.jobs[0], "future_payload")) == [1]


def test_arbitrary_future_objects_are_rejected_before_canonical_storage() -> None:
    class DictObject:
        def __init__(self) -> None:
            self.nested = [1]

    class SlotsObject:
        __slots__ = ("nested",)

        def __init__(self) -> None:
            self.nested = [1]

    for value in (DictObject(), SlotsObject(), object()):
        state = _state(
            jobs=[JobState.model_validate({"id": "child", "type": "task", "future_payload": value})]
        )
        with pytest.raises(TypeError, match="unsupported canonical frontend value"):
            FrontendStateStore(state)


def test_immutable_wrappers_have_no_builtin_base_class_bypass() -> None:
    initial = _state(
        jobs=[
            JobState(
                id="child",
                type="task",
                trajectory=[{"type": "notice", "details": {"nested": [1]}}],
            )
        ]
    )
    owner = FrontendStateStore(initial)
    follower = FrontendStateStore(initial)
    snapshot = owner.state
    jobs = snapshot.jobs
    trajectory = snapshot.jobs[0].trajectory
    event = trajectory[0]

    assert not isinstance(jobs, list)
    assert not isinstance(trajectory, list)
    assert not isinstance(event, dict)
    with pytest.raises(TypeError):
        list.append(jobs, JobState(id="injected", type="task"))
    with pytest.raises(TypeError):
        list.append(trajectory, {"type": "injected"})
    with pytest.raises(TypeError):
        dict.__setitem__(event, "type", "corrupted")
    with pytest.raises(TypeError):
        event["details"]["nested"] = [2]
    for wrapper in (jobs, trajectory, event):
        assert not hasattr(wrapper, "__dict__")
        with pytest.raises(TypeError):
            vars(wrapper)
        with pytest.raises(AttributeError):
            object.__setattr__(wrapper, "_values", [])
        assert copy.copy(wrapper) is wrapper
        assert copy.deepcopy(wrapper) is wrapper
        restored = pickle.loads(pickle.dumps(wrapper))
        assert restored == wrapper
        assert type(restored) is type(wrapper)

    dumped = owner.state.model_dump(mode="json")
    assert dumped["jobs"][0]["trajectory"] == [{"type": "notice", "details": {"nested": [1]}}]
    assert owner.state.sequence == 0
    update = owner.mutate(jobs=list(owner.state.jobs))
    assert update is None
    assert owner.state.jobs == follower.state.jobs
    assert owner.state.model_dump(mode="json") == follower.state.model_dump(mode="json")


def test_public_and_input_pydantic_owners_never_alias_canonical_state() -> None:
    input_job = JobState.model_validate(
        {
            "id": "child",
            "type": "task",
            "status": "running",
            "usage": Usage(input_tokens=4).model_dump(mode="json"),
            "descendant_usage": [FrontendUsage(output_tokens=3).model_dump(mode="json")],
            "trajectory": [{"type": "notice", "index": 1}],
            "future_model": FrontendUsage(input_tokens=7),
        }
    )
    initial = _state(jobs=[input_job])
    owner = FrontendStateStore(initial)
    follower = FrontendStateStore(initial)
    baseline = owner.state.model_dump(mode="json")
    snapshot = owner.state
    public_job = snapshot.jobs[0]

    cast(Any, public_job.__dict__)["status"] = "corrupted"
    cast(Any, public_job.__dict__)["trajectory"] = ()
    assert public_job.usage is not None
    cast(Any, public_job.usage.__dict__)["input_tokens"] = 99
    cast(Any, public_job.descendant_usage[0].__dict__)["output_tokens"] = 99
    assert public_job.__pydantic_extra__ is not None
    cast(Any, public_job.__pydantic_extra__)["future_model"].__dict__["input_tokens"] = 99
    cast(Any, public_job.__pydantic_extra__)["injected"] = "bad"

    cast(Any, input_job.__dict__)["status"] = "input-corrupted"
    cast(Any, input_job.__dict__)["trajectory"].append({"type": "input-corrupted"})
    assert input_job.usage is not None
    cast(Any, input_job.usage.__dict__)["input_tokens"] = 88
    assert input_job.__pydantic_extra__ is not None
    cast(Any, input_job.__pydantic_extra__)["future_model"].__dict__["input_tokens"] = 88

    assert owner.state.sequence == 0
    assert owner.state.model_dump(mode="json") == baseline
    assert follower.state.model_dump(mode="json") == baseline

    repaired = JobState.model_validate(baseline["jobs"][0])
    update = owner.mutate(jobs=[repaired])
    if update is not None:
        follower.apply_update(update)
    assert owner.state.jobs == follower.state.jobs
    assert owner.state.model_dump(mode="json") == follower.state.model_dump(mode="json")


def test_input_job_is_detached_again_on_mutation() -> None:
    store = FrontendStateStore(_state(jobs=[]))
    incoming = JobState(id="child", type="task", trajectory=[{"type": "notice"}])
    update = store.mutate(jobs=[incoming])
    assert update is not None
    cast(Any, incoming.__dict__)["status"] = "corrupted"
    cast(Any, incoming.__dict__)["trajectory"].append({"type": "corrupted"})

    canonical = store.state.jobs[0]
    assert canonical.status == "running"
    assert canonical.trajectory == [{"type": "notice"}]


def test_rejected_snapshot_mutation_cannot_diverge_owner_and_follower() -> None:
    """An alias attempt cannot hide a trajectory event from the next wire delta."""
    initial = _state(jobs=[JobState(id="child", type="task", trajectory=[])])
    owner = FrontendStateStore(initial)
    follower = FrontendStateStore(initial)
    snapshot = owner.state

    with pytest.raises((AttributeError, TypeError)):
        snapshot.jobs[0].trajectory.append({"type": "notice", "index": 1})

    changed = JobState(
        id="child",
        type="task",
        status="completed",
        trajectory=[{"type": "notice", "index": 1}],
    )
    update = owner.mutate(jobs=[changed])
    assert update is not None
    assert update.job_trajectory_appends == {"child": [{"type": "notice", "index": 1}]}
    follower.apply_update(update)
    assert follower.state.jobs[0].trajectory == owner.state.jobs[0].trajectory


@pytest.mark.parametrize("replacement", [False, True])
def test_trajectory_delta_preserves_nested_json_objects(replacement: bool) -> None:
    """Exercise JSON, not an in-process delta that still recognizes frozen maps."""
    earlier = {"type": "notice", "text": "Earlier activity"}
    initial = _state(jobs=[JobState(id="child", type="task", trajectory=[earlier])])
    owner = FrontendStateStore(initial)
    follower = FrontendStateStore(initial)
    events = [
        {
            "type": "tool_execution_start",
            "tool_call_id": "call",
            "tool_name": "read",
            "intent": "Reading documentation",
            "args": {"path": "README.md", "options": {"ranges": [{"start": 1}]}},
        },
        {
            "type": "tool_execution_end",
            "tool_call_id": "call",
            "tool_name": "read",
            "result": {
                "content": [{"type": "text", "text": "Synthetic documentation"}],
                "details": {"count": 1, "rows": [{"ok": True, "extra": None}]},
            },
        },
        {
            "type": "message_end",
            "message": {
                "id": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Finished reading."}],
            },
        },
    ]
    trajectory = events if replacement else [earlier, *events]
    update = owner.mutate(jobs=[JobState(id="child", type="task", trajectory=trajectory)])
    assert update is not None
    wire = FrontendUpdate.model_validate_json(update.model_dump_json())
    assert wire.job_trajectory_appends == {"child": events}
    assert wire.job_trajectory_replacements == (["child"] if replacement else [])
    assert follower.apply_update(wire)
    assert follower.state.jobs[0].trajectory == owner.state.jobs[0].trajectory
    assert follower.state.jobs[0].model_dump(mode="json")["trajectory"] == trajectory
    # Thawing the emitted payload must not expose canonical state to a consumer.
    update.job_trajectory_appends["child"][0]["args"]["options"]["ranges"][0]["start"] = 99
    assert owner.state.jobs[0].model_dump(mode="json")["trajectory"] == trajectory


def test_one_large_roster_progress_update_sends_only_the_new_event() -> None:
    """A one-child append must not serialize 50,000 unchanged events."""
    jobs = [
        JobState(
            id=f"child-{index}",
            type="task",
            trajectory=[{"type": "notice", "index": event} for event in range(500)],
        )
        for index in range(100)
    ]
    store = FrontendStateStore(_state(jobs=jobs))
    changed = jobs[-1].model_copy(
        update={"trajectory": [*jobs[-1].trajectory, {"type": "notice", "index": 500}]}
    )

    update = store.mutate(jobs=[*jobs[:-1], changed])

    assert update is not None
    assert update.job_trajectory_appends == {jobs[-1].id: [{"type": "notice", "index": 500}]}
    assert all("trajectory" not in summary for summary in update.changes["jobs"])
    assert len(store.state.jobs[-1].trajectory) == 501


def test_noop_refresh_consumes_no_sequence_for_model_list_fields() -> None:
    """N4: identical jobs/capabilities must not publish a frame each refresh."""
    jobs = [
        JobState(id="j1", type="task", trajectory=[{"type": "e", "n": index} for index in range(4)])
    ]
    store = FrontendStateStore(_state())
    first = store.mutate(jobs=jobs)
    assert first is not None
    again = store.mutate(jobs=[job.model_copy(deep=True) for job in jobs])
    assert again is None, "unchanged list-of-model fields consumed a sequence"


def test_rotated_trajectory_ships_replacement_and_follower_stays_bounded() -> None:
    """N2: past TRAJECTORY_CAP the delta is a replacement, never endless appends.

    REPLACEMENT *for these rows*, and the reason is now a property of the rows
    rather than of the cap: the classifier proves a rotation row for row from the
    rows' ``_lo_seq`` stamps, so a STAMPED rotation ships the appended tail with no
    marker (see ``test_frontend_row_window``, ``_capped_overlap_tail``). These rows
    carry no stamp, nothing about the overlap can be proven, and the delta keeps
    the replacement it has always sent. The precondition is asserted rather than
    assumed so this cell cannot drift into the proven-tail case and quietly stop
    covering the fallback it exists for.
    """
    from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
    from local_operator.harness.subagent import TRAJECTORY_CAP

    owner = FrontendStateStore(_state(jobs=[]))
    follower = FrontendStateStore(_state(jobs=[]))
    seed = owner.mutate(
        jobs=[
            JobState(
                id="child",
                type="task",
                trajectory=[{"type": "e", "n": index} for index in range(TRAJECTORY_CAP)],
            )
        ]
    )
    assert seed is not None
    follower.apply_update(seed)
    for round_no in range(1, 4):
        rotated = [{"type": "e", "n": index + round_no} for index in range(TRAJECTORY_CAP)]
        assert all(TRAJECTORY_SEQ_KEY not in row for row in rotated), (
            "stamping these rows makes the rotation provable, which ships a tail "
            "instead of a replacement -- this cell is the unprovable fallback"
        )
        update = owner.mutate(jobs=[JobState(id="child", type="task", trajectory=rotated)])
        assert update is not None
        assert update.job_trajectory_replacements == ["child"]
        follower.apply_update(update)
        assert len(follower.state.jobs[0].trajectory) == TRAJECTORY_CAP
        assert follower.state.jobs[0].trajectory == rotated


def test_child_costs_price_descendant_usage_like_the_owner() -> None:
    """N5: nested (#297) spend reaches canonical child_costs at descendant rates."""
    job = AsyncJob(
        id="root",
        type="task",
        label="manager",
        start_time=1.0,
        model_label="anthropic/sonnet",
        usage=Usage(input_tokens=1_000_000),
        descendant_usage=[Usage(input_tokens=1_000_000, provider="anthropic", model_id="sonnet")],
    )
    manager = SimpleNamespace(list=lambda: [job])
    session = SimpleNamespace(
        jobs=manager,
        model=_spec(),
        session_id="s1",
        queued_steering=lambda: [],
    )
    dto = JobState.from_job(job)
    assert [component.model_id for component in dto.descendant_usage] == ["sonnet"]

    from unittest.mock import patch

    from local_operator.model.registry import ModelInfo

    priced = ModelInfo(id="sonnet", name="sonnet", description="", input_price=10.0)
    # The pricing path is paint-safe (#300): ``turn_cost`` reads
    # ``resolve_model_info_paint``'s memo-or-registry answer, never the full
    # discovery resolver, so that is the seam a priced test must feed.
    with patch(
        "local_operator.model.configure.resolve_model_info_paint",
        side_effect=lambda provider, model_id: (priced, True),
    ):
        store = FrontendStateStore(_state(jobs=[], child_costs={}))
        update = store.refresh_jobs(session)
    assert update is not None
    # $10/MTok on 1M direct + 1M descendant tokens: the whole subtree, not half.
    assert store.state.child_costs["root"] == pytest.approx(20.0)

    # Follower re-pricing from the wire DTO reaches the same figure.
    remote_manager = SimpleNamespace(list=lambda: [JobState.model_validate(dto.model_dump())])
    with patch(
        "local_operator.model.configure.resolve_model_info_paint",
        side_effect=lambda provider, model_id: (priced, True),
    ):
        remote_store = FrontendStateStore(_state(jobs=[], child_costs={}))
        remote_store.refresh_jobs(
            SimpleNamespace(
                jobs=remote_manager, model=_spec(), session_id="s1", queued_steering=lambda: []
            )
        )
    assert remote_store.state.child_costs["root"] == pytest.approx(20.0)


def test_checkpoint_strips_trajectories_and_live_events() -> None:
    """n2: durable checkpoints must not carry ~71 KiB of reconstructable events."""
    import asyncio

    state = _state(
        jobs=[
            JobState(
                id="busy",
                type="task",
                trajectory=[{"type": "e", "n": index} for index in range(50)],
            )
        ],
        live_events=[{"type": "message_update"}],
    )
    store = FrontendStateStore(state)

    class _Transcript:
        def __init__(self) -> None:
            self.appended: list[tuple[str, dict[str, Any]]] = []

        async def append_custom(self, custom_type: str, payload: dict[str, Any]) -> None:
            self.appended.append((custom_type, payload))

    transcript = _Transcript()
    asyncio.run(store.checkpoint(transcript))
    ((_, payload),) = transcript.appended
    assert payload["state"]["live_events"] == []
    assert payload["state"]["jobs"][0]["trajectory"] == []
    # The in-memory state a live follower reads keeps its trajectory.
    assert len(store.state.jobs[0].trajectory) == 50


def test_queued_custom_steers_project_their_human_text() -> None:
    """The queued-steering snapshot reads ``text``/``content``, which only a
    plain user Message has. A busy-path peer steer and a busy-path wake queue
    their CustomMessage instead, and those used to project as ``{"text": ""}``
    — a blank row for any follower that renders the queue. A peer row keeps
    its raw text in ``details["body"]`` (``details["text"]`` is the
    model-facing envelope); a wake's human text is its ``details["text"]``."""
    from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
    from local_operator.harness.types import CustomMessage
    from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE

    peer = CustomMessage(
        custom_type=PEER_MESSAGE_MESSAGE_TYPE,
        attribution="user",
        details={
            "text": "<peer-session-message from_pid=3>\nredirect now\n</peer-session-message>",
            "body": "redirect now",
            "sender": {"pid": 3},
        },
    )
    wake = CustomMessage(
        custom_type=WAKE_PROMPT_MESSAGE_TYPE,
        attribution="user",
        details={"wake_id": "w1", "occurrence": 1, "text": "wake: check the build"},
    )
    typed = Message.user("plain steer")
    session = SimpleNamespace(
        jobs=SimpleNamespace(list=lambda: []),
        _subagent_comms=None,
        model=_spec(),
        effective_model=_spec(),
        session_id="s1",
        cwd="/repo",
        queued_steering=lambda: [peer, wake, typed],
        conversation_name="Canonical state",
        goal="",
        active_agent="",
        active_team_name="",
        wake_scheduler=None,
        mcp_manager=None,
        mcp_startup=None,
    )
    store = FrontendStateStore(_state(jobs=[]))
    state = store.refresh_from_session(session)
    assert [entry["text"] for entry in state.queued_steering] == [
        "redirect now",
        "wake: check the build",
        "plain steer",
    ]
    assert [entry["id"] for entry in state.queued_steering] == [peer.id, wake.id, typed.id]
    assert all(entry["image_count"] == 0 for entry in state.queued_steering)


class _CheckpointTranscript:
    """The one method ``FrontendStateStore._restored_state`` reads."""

    def __init__(self, state: FrontendSessionState | None, checkpoint_id: str = "cp-parent"):
        self._state = state
        self._checkpoint_id = checkpoint_id

    def latest_custom(self, custom_type: str) -> dict[str, Any] | None:
        if self._state is None:
            return None
        return {
            "checkpoint_id": self._checkpoint_id,
            "state": self._state.model_dump(mode="json"),
        }


def _owner_over(session_id: str, checkpoint: FrontendSessionState | None) -> SimpleNamespace:
    return SimpleNamespace(session_id=session_id, _transcript=_CheckpointTranscript(checkpoint))


def test_a_fork_restores_its_own_session_id_not_the_parents(tmp_path) -> None:
    """#573: the directory a session runs in is authoritative for who it is.

    A fork copies ``transcript.jsonl`` verbatim, so the newest checkpoint it
    restores was written BY THE PARENT and carries the parent's ``session_id``.
    The runtime then served that id in every ``frontend_sync`` and
    ``AttachedSession._install_frontend`` refused the frame — a fork nobody could
    attach to, and (in switch mode) a fork whose own viewer never got a state
    install, so ``/model`` looked inert and the band never painted its context.
    The same shape as the ``COPIED_SIDECARS`` set-equality test in
    ``tests/unit/test_fork.py``: fork with a checkpoint present, assert the
    restored identity equals the new directory name.
    """
    from local_operator.fork import fork_session

    parent = _state(session_id="parent000001", checkpoint_id="cp-parent")
    parent_dir = tmp_path / "sessions" / "parent000001"
    parent_dir.mkdir(parents=True)
    row = {
        "id": "r1",
        "ts": 1.0,
        "type": "custom",
        "payload": {
            "custom_type": "frontend_state_checkpoint_v1",
            "details": {"checkpoint_id": "cp-parent", "state": parent.model_dump(mode="json")},
        },
    }
    import json

    (parent_dir / "transcript.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    fork_id = fork_session(tmp_path, "parent000001")

    from local_operator.session.transcript import Transcript

    owner = SimpleNamespace(
        session_id=fork_id, _transcript=Transcript(tmp_path / "sessions" / fork_id)
    )
    restored = FrontendStateStore.from_checkpoint(owner).state
    assert restored.session_id == fork_id
    # Everything that is genuinely the conversation's carries over: spend,
    # title, occupancy against the window it was measured on.
    assert restored.conversation_title == parent.conversation_title
    assert restored.cumulative_parent_cost == parent.cumulative_parent_cost
    assert restored.context_tokens == parent.context_tokens
    assert restored.context_window == parent.context_window


def test_a_same_session_resume_keeps_its_checkpoint_id_and_jobs() -> None:
    """The fixup is scoped to INHERITED checkpoints. A resume of the session
    that wrote the row keeps its own bookkeeping: the checkpoint id names a row
    this transcript wrote for itself, and the jobs are its own children."""
    own = _state(session_id="s1", checkpoint_id="cp-own")
    restored = FrontendStateStore.from_checkpoint(_owner_over("s1", own)).state
    assert restored.session_id == "s1"
    assert restored.checkpoint_id == "cp-own"
    assert [job.id for job in restored.jobs] == ["j1"]


def test_an_inherited_checkpoint_drops_the_parents_checkpoint_id_and_jobs() -> None:
    """The two identity-bearing fields #573 flagged beside ``session_id``.

    ``checkpoint_id`` would otherwise name the parent's last row until the
    fork's first turn end; ``jobs`` are the parent's children, which
    ``fork.EXCLUDED_SIDECARS`` already keeps out of the roster sidecar — the
    checkpoint was the one door left open.
    """
    inherited = _state(session_id="parent000001", checkpoint_id="cp-parent")
    restored = FrontendStateStore.from_checkpoint(_owner_over("fork00000001", inherited)).state
    assert restored.session_id == "fork00000001"
    assert restored.checkpoint_id is None
    assert restored.jobs == ()
    # ``from_session`` (the TUI-hosted construction) restores through the same
    # helper, so the two hosts cannot disagree about the fork's identity.
    assert FrontendStateStore._restored_state(
        _owner_over("fork00000001", inherited)
    ).session_id == ("fork00000001")


def test_a_fork_of_a_fork_is_stamped_with_its_own_id_at_every_hop() -> None:
    """#573 observed the GRANDPARENT's id two hops down: each hop inherited
    whatever the previous one was already serving. With the re-stamp on
    restore, a checkpoint written by the first fork names the first fork, and
    the second fork corrects it again to itself."""
    grandparent = _state(session_id="grand0000001")
    first_fork = FrontendStateStore.from_checkpoint(_owner_over("fork10000001", grandparent)).state
    assert first_fork.session_id == "fork10000001"
    # The first fork writes ITS checkpoint (now correctly stamped); the second
    # fork inherits that row.
    second_fork = FrontendStateStore.from_checkpoint(_owner_over("fork20000001", first_fork)).state
    assert second_fork.session_id == "fork20000001"
    assert second_fork.jobs == ()


def test_agent_end_records_last_turn_outcome() -> None:
    """A rebinding viewer needs the logical-turn outcome after live_events clear."""
    store = FrontendStateStore(_state())
    session = SimpleNamespace(effective_model=_spec())
    store.observe_event(session, AgentStartEvent(generation=3))
    store.observe_event(session, AgentEndEvent(messages=[Message.assistant("ok")]))
    assert store.state.last_turn_outcome == "completed"
    store.observe_event(session, AgentStartEvent(generation=4))
    store.observe_event(session, AgentEndEvent(aborted=True))
    assert store.state.last_turn_outcome == "aborted"
    store.observe_event(session, AgentStartEvent(generation=5))
    store.observe_event(session, AgentEndEvent(error="boom"))
    assert store.state.last_turn_outcome == "error"


# ---------------------------------------------------------------------------
# The in-flight seed and the placeholder compose key.
#
# `_fold_live_event` keys the seed by `tool_call_id`. A provider that announces
# a call's name before its id makes the loop announce the row under
# `compose:{index}` while every `tool_execution_start`/`_end` carries the real
# id, so without the supersession hand-off the seed keeps BOTH id spaces. A
# viewer joining mid-turn then receives a composing row that no start or end
# can ever match, and turn-end retirement paints it `⊘ interrupted` on a call
# that SUCCEEDED (the round-1 Q2 bug, reached by a new route).
# ---------------------------------------------------------------------------


def _fold(store: FrontendStateStore, *events) -> list[dict[str, Any]]:  # noqa: ANN002
    session = SimpleNamespace(effective_model=_spec())
    for event in events:
        store.observe_event(session, event)
    return list(store.state.live_events)


def test_seed_drops_the_placeholder_row_when_the_real_id_is_announced() -> None:
    """One call, one seed entry — keyed by the id execution will actually use."""
    store = FrontendStateStore(_state())
    live = _fold(
        store,
        AgentStartEvent(generation=1),
        ToolCallComposeEvent(tool_call_id="compose:0", tool_name="bash", argument_bytes=10),
        ToolCallComposeEvent(
            tool_call_id="real_0",
            tool_name="bash",
            argument_bytes=20,
            supersedes_tool_call_id="compose:0",
        ),
    )
    tool_rows = [row for row in live if row.get("type") == "tool_call_compose"]
    assert [row["tool_call_id"] for row in tool_rows] == ["real_0"]


def test_seed_supersession_is_per_call_with_several_in_flight() -> None:
    """Three concurrent calls: each promotion drops ONLY its own placeholder.

    This is why the promotion is announced rather than inferred from a start.
    A rule like "any tool_execution_start clears the composing entries" cannot
    tell which placeholder a real id belongs to, so it would drop the rows of
    calls still being dictated.
    """
    store = FrontendStateStore(_state())
    events = [AgentStartEvent(generation=1)]
    events += [
        ToolCallComposeEvent(tool_call_id=f"compose:{i}", tool_name="bash", argument_bytes=5)
        for i in range(3)
    ]
    # Only the middle call learns its id.
    events.append(
        ToolCallComposeEvent(
            tool_call_id="real_1",
            tool_name="bash",
            argument_bytes=9,
            supersedes_tool_call_id="compose:1",
        )
    )
    live = _fold(store, *events)

    ids = [row["tool_call_id"] for row in live if row.get("type") == "tool_call_compose"]
    # The other two rows are still being dictated and must survive.
    assert sorted(ids) == ["compose:0", "compose:2", "real_1"]


def test_seed_carries_one_row_per_call_through_start_and_end() -> None:
    """End to end: the joiner's seed never holds a row it cannot settle.

    Without the hand-off this seed held six entries — three unadoptable
    `compose:N` rows beside the three real ones.
    """
    store = FrontendStateStore(_state())
    events = [AgentStartEvent(generation=1)]
    events += [
        ToolCallComposeEvent(tool_call_id=f"compose:{i}", tool_name="bash", argument_bytes=5)
        for i in range(3)
    ]
    events += [
        ToolCallComposeEvent(
            tool_call_id=f"real_{i}",
            tool_name="bash",
            argument_bytes=9,
            supersedes_tool_call_id=f"compose:{i}",
        )
        for i in range(3)
    ]
    events += [
        ToolExecutionStartEvent(tool_call_id=f"real_{i}", tool_name="bash", args={})
        for i in range(3)
    ]
    live = _fold(store, *events)

    assert len(live) == 3
    assert [row["type"] for row in live] == ["tool_execution_start"] * 3
    assert [row["tool_call_id"] for row in live] == ["real_0", "real_1", "real_2"]


def test_seed_fold_is_unchanged_when_an_older_runtime_omits_the_field() -> None:
    """BACKWARD COMPATIBILITY, pinned.

    `supersedes_tool_call_id` is additive: an older runtime relaying through a
    newer viewer simply never sets it. The fold must then behave exactly as it
    did before the field existed — replace by `tool_call_id`, drop nothing
    else — so a viewer cannot break on a payload that omits it.

    Asserted on a RAW wire payload with the key genuinely absent, not merely
    set to None, because that is the shape an older owner actually sends.
    Built through ``model_validate`` on the base ``AgentEvent``, which is
    exactly how ``deserialize_event`` rehydrates a frame whose type an older
    peer relayed: extras are allowed and the absent field stays absent, so the
    dump is the old payload byte for byte rather than a null-valued imitation.
    """
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))

    for call_id in ("compose:0", "compose:1"):
        legacy = AgentEvent.model_validate(
            {
                "type": "tool_call_compose",
                "tool_call_id": call_id,
                "tool_name": "bash",
                "argument_bytes": 7,
            }
        )
        assert "supersedes_tool_call_id" not in legacy.model_dump(mode="json")
        store.observe_event(session, legacy)

    live = list(store.state.live_events)
    rows = [row for row in live if row.get("type") == "tool_call_compose"]
    # Today's behaviour: both rows kept, keyed by their own ids, and no
    # `supersedes_tool_call_id` key anywhere in the payload.
    assert [row["tool_call_id"] for row in rows] == ["compose:0", "compose:1"]
    assert all("supersedes_tool_call_id" not in row for row in rows)


def test_compose_event_tolerates_a_payload_without_the_field() -> None:
    """A viewer deserializing an older owner's frame must not raise.

    The field is optional with a None default precisely so this validates.
    """
    event = ToolCallComposeEvent.model_validate(
        {"type": "tool_call_compose", "tool_call_id": "c1", "tool_name": "bash"}
    )
    assert event.supersedes_tool_call_id is None


def test_a_repeated_supersession_is_idempotent_in_the_seed() -> None:
    """The hand-off is repeated on every later frame; replaying it changes nothing.

    The repeat exists so a lossy path cannot drop the only copy of the identity
    change. That is only safe if applying it twice is the same as applying it
    once — the second time there is no placeholder left to drop, and the fold
    must simply replace the real id's own row as it always does.
    """
    store = FrontendStateStore(_state())
    events = [
        AgentStartEvent(generation=1),
        ToolCallComposeEvent(tool_call_id="compose:0", tool_name="bash", argument_bytes=5),
    ]
    events += [
        ToolCallComposeEvent(
            tool_call_id="real_0",
            tool_name="bash",
            argument_bytes=size,
            supersedes_tool_call_id="compose:0",
        )
        for size in (10, 20, 30)
    ]
    live = _fold(store, *events)

    assert len(live) == 1
    assert live[0]["tool_call_id"] == "real_0"
    # The newest frame wins, so the size the viewer reads is the current one.
    assert live[0]["argument_bytes"] == 30


# ---------------------------------------------------------------------------
# The live-call clock anchors.
#
# A frontend that ATTACHES to work already in flight — a sidebar switch back to
# a conversation whose tool is still running, a re-attach, a `/resume` onto a
# live turn — paints its rows at the moment it arrives, so every widget it owns
# counts from zero unless the session hands it the producer's own start
# instants. Two folds supply them, and both are asserted here rather than on a
# widget, because a widget test can only observe what these already published:
#
# * ``live_tool_started_at`` — one epoch per call executing now, for the row.
#   Membership answers "has this call started" (a replay needs it to tell a
#   queued call from an executing one) and the value is the instant to count
#   from, ``None`` when the start carried no epoch.
# * ``activity_phase``/``activity_phase_started_at`` — the phase edge, for the
#   band's ``thinking``/``responding``/``composing`` arm, which has no call
#   behind it at all.
#
# The rule they share, and the one worth stating before the assertions: a
# MISSING epoch is never filled in with the fold's own ``now``. For an attached
# viewer that value is its arrival instant wearing the call's name, and it
# would print a plausible wrong age where the widget's blank column is the
# truth. The tests below pin that refusal as hard as they pin the fold.
# ---------------------------------------------------------------------------


def _live_session() -> SimpleNamespace:
    """A session in the MIDDLE of a turn, which is the only state these drive.

    ``is_streaming`` is not decoration: ``observe_event`` calls
    ``refresh_from_session`` at tool and message boundaries, and that method's
    non-streaming gate is the documented way a settled session publishes no
    phase and no live anchors at all. A fake without the flag therefore blanks
    both fields on the first ``tool_execution_end`` — which is correct
    behaviour being exercised by the wrong fixture.
    """
    return SimpleNamespace(effective_model=_spec(), is_streaming=True)


def test_live_tool_starts_fold_per_call_and_pop_on_their_own_end() -> None:
    """One anchor per LIVE call, keyed by id, released by that call's end.

    Keyed rather than scalar because a batch has several: the operator's report
    was a row and a band disagreeing about one call's age, and only a per-call
    map can tell both surfaces which call's start they are counting from.
    """
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))
    assert store.live_tool_start_epochs() == {}

    started = time.time() - 27.0
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="call-bash", tool_name="bash", args={}, started_at_epoch=started
        ),
    )
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="call-read",
            tool_name="read",
            args={},
            started_at_epoch=started + 1.0,
        ),
    )
    assert store.live_tool_start_epochs() == {"call-bash": started, "call-read": started + 1.0}

    # One call of the batch finishes; the survivor keeps ITS own zero, so the
    # band's floor cannot move because a sibling settled.
    store.observe_event(
        session,
        ToolExecutionEndEvent(
            tool_call_id="call-read",
            tool_name="read",
            result=ToolResult(tool_call_id="call-read", tool_name="read", content=[]),
        ),
    )
    assert store.live_tool_start_epochs() == {"call-bash": started}


def test_a_start_without_an_epoch_records_the_start_but_no_instant() -> None:
    """A legacy start is PRESENT with ``None``: the fact yes, the stamp never.

    An older runtime's events carry no ``started_at_epoch``, and an attached
    viewer that substituted its own fold instant would be inventing the age the
    whole change exists to stop inventing. What the event DOES carry is the
    fact that the call began, and the map has to keep the two apart: a replay
    that painted a merely ANNOUNCED call — queued behind a sibling's execution
    group — as executing used to answer "has it started?" by the same absence
    this refusal produced for a genuinely running legacy call. Presence is the
    start fact; ``None`` is what makes the widget withhold the clock.
    """
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))
    store.observe_event(
        session, ToolExecutionStartEvent(tool_call_id="call-legacy", tool_name="bash", args={})
    )
    assert store.live_tool_start_epochs() == {"call-legacy": None}
    # A call that never started is absent entirely, and that is the other half
    # of the same contract: no event, no entry.
    assert "call-never-announced" not in store.live_tool_start_epochs()
    # And the phase is still folded: the SEQ is knowable even when the instant
    # is not, and the two answers are independent on purpose.
    assert store.state.activity_phase == "running"


def test_both_ends_of_a_turn_clear_the_live_anchors() -> None:
    """Stale anchors are worse than none, so the turn boundary clears them all.

    Not left to the individual ends: a turn that dies without emitting every
    ``tool_execution_end`` — an abort, a killed provider stream — would leave
    an entry behind, and a later turn's row seeding from it would wear a
    previous turn's age.
    """
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="call-1", tool_name="bash", args={}, started_at_epoch=time.time() - 5.0
        ),
    )
    assert store.live_tool_start_epochs() != {}

    store.observe_event(session, AgentEndEvent(messages=[Message.assistant("done")]))
    assert store.live_tool_start_epochs() == {}

    # A NEW turn starts clean too, so an anchor that somehow survived a turn end
    # still cannot seed the next turn's rows.
    store.observe_event(session, AgentStartEvent(generation=2))
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="call-2", tool_name="bash", args={}, started_at_epoch=time.time()
        ),
    )
    store.observe_event(session, AgentStartEvent(generation=3))
    assert store.live_tool_start_epochs() == {}


def test_live_tool_start_epochs_hands_out_a_copy() -> None:
    """The map is the store's own object, so callers must not be able to reach it."""
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="call-1", tool_name="bash", args={}, started_at_epoch=1_700_000_000.0
        ),
    )
    handed_out = store.live_tool_start_epochs()
    handed_out["call-evil"] = 1.0
    assert store.live_tool_start_epochs() == {"call-1": 1_700_000_000.0}


# ---------------------------------------------------------------------------
# The retained END carries the clock its START announced.
#
# The seed keeps a settled ``tool_execution_end`` so a viewer joining mid-turn
# can settle a card for work that finished while it was away — and that end is
# the only frame of the pair that states NO time, because it replaced the start
# that did. A client cannot date such a row itself (it refuses to place a frame
# at its own arrival instant), so without the stamp the joining pane paints a
# wall of rows that are in the wrong place and name nothing that ran, on a
# runtime whose TUI has had them right since they ran.
#
# The stamp is therefore CARRIED, never invented: absent start, absent key.
# ---------------------------------------------------------------------------


def _fold_live(store: FrontendStateStore, *events) -> list[dict[str, Any]]:  # noqa: ANN002
    """``_fold`` against a session MID-TURN, which the settled rows require.

    ``observe_event`` refreshes from the session at every ``tool_execution_end``,
    and a session with no turn in flight publishes NO seed at all — so a
    non-streaming fake blanks ``live_events`` on the first end and would make
    every assertion below vacuous.
    """
    session = _live_session()
    for event in events:
        store.observe_event(session, event)
    return list(store.state.live_events)


def test_a_settled_seed_end_carries_the_clock_its_start_announced() -> None:
    """The joining viewer is handed the instant the call really began.

    Taken from ``live_tool_started_at`` at the ONE point it is readable: the end
    arm of the seed fold, which ``observe_event`` runs before the anchor fold
    pops the entry on this same event. Read anywhere later and there is nothing
    left to read, which is what makes this the only place the stamp can be made.
    """
    store = FrontendStateStore(_state())
    started = time.time() - 27.0
    live = _fold_live(
        store,
        AgentStartEvent(generation=1),
        ToolExecutionStartEvent(
            tool_call_id="call-bash", tool_name="bash", args={}, started_at_epoch=started
        ),
        ToolExecutionEndEvent(
            tool_call_id="call-bash",
            tool_name="bash",
            result=ToolResult(tool_call_id="call-bash", tool_name="bash", content=[]),
        ),
    )

    assert [row["type"] for row in live] == ["tool_execution_end"]
    assert live[0]["started_at_epoch"] == started
    # The start is still DROPPED and its anchor still released: the clock is
    # carried onto the retained row rather than the row being kept beside it.
    assert store.live_tool_start_epochs() == {}


def test_a_settled_seed_end_states_no_time_when_no_start_was_seen() -> None:
    """Never fabricate: the shapes that must carry no stamp at all.

    A missing instant is a real answer, not a value to default, and the three
    ways to reach one are pinned here:

    * a LEGACY start, recorded in the anchor map as a start with no instant;
    * an end whose start this fold never saw at all;
    * a call that will never run — ``not_run_reason`` deliberately gets no
      ``tool_execution_start``/``_end``, so its row is the compose surface and
      nothing may be stamped onto it.
    """
    store = FrontendStateStore(_state())
    session = _live_session()
    store.observe_event(session, AgentStartEvent(generation=1))
    store.observe_event(
        session, ToolExecutionStartEvent(tool_call_id="legacy", tool_name="bash", args={})
    )
    assert store.live_tool_start_epochs() == {"legacy": None}
    store.observe_event(
        session,
        ToolExecutionEndEvent(
            tool_call_id="legacy",
            tool_name="bash",
            result=ToolResult(tool_call_id="legacy", tool_name="bash", content=[]),
        ),
    )
    store.observe_event(
        session,
        ToolExecutionEndEvent(
            tool_call_id="orphan",
            tool_name="read",
            result=ToolResult(tool_call_id="orphan", tool_name="read", content=[]),
        ),
    )
    store.observe_event(
        session,
        ToolCallComposeEvent(
            tool_call_id="parked",
            tool_name="bash",
            argument_bytes=5,
            dictation_complete=True,
            not_run_reason="Tool call not run: planning failure",
        ),
    )

    rows = {row["tool_call_id"]: row for row in store.state.live_events}
    assert rows["legacy"]["type"] == "tool_execution_end"
    assert rows["orphan"]["type"] == "tool_execution_end"
    assert rows["parked"]["type"] == "tool_call_compose"
    for call_id in ("legacy", "orphan", "parked"):
        assert (
            "started_at_epoch" not in rows[call_id]
        ), f"{call_id} carries a clock no producer ever announced"


def test_the_stamp_is_one_key_on_the_retained_row_and_adds_no_row() -> None:
    """The superseded/compose id space is untouched: one call, one row.

    The stamp is a key on the dict the seed already ships, NOT a second
    retained start. Re-keeping the start beside the end would re-open the two
    id spaces the supersession hand-off exists to close and double the rows the
    line budget pays for, for nothing the stamped end does not already state.
    """
    store = FrontendStateStore(_state())
    started = time.time() - 3.0
    live = _fold_live(
        store,
        AgentStartEvent(generation=1),
        ToolCallComposeEvent(tool_call_id="compose:0", tool_name="bash", argument_bytes=5),
        ToolCallComposeEvent(
            tool_call_id="real_0",
            tool_name="bash",
            argument_bytes=20,
            supersedes_tool_call_id="compose:0",
        ),
        ToolExecutionStartEvent(
            tool_call_id="real_0", tool_name="bash", args={}, started_at_epoch=started
        ),
        ToolExecutionEndEvent(
            tool_call_id="real_0",
            tool_name="bash",
            result=ToolResult(tool_call_id="real_0", tool_name="bash", content=[]),
        ),
    )

    # The placeholder was retired by the promotion and the end replaced the
    # start, so the seed holds one row and the stamp moves no count.
    assert len(live) == 1
    assert live[0]["type"] == "tool_execution_end"
    assert live[0]["tool_call_id"] == "real_0"
    assert live[0]["started_at_epoch"] == started


def test_the_seed_row_cap_still_holds_when_every_row_is_stamped() -> None:
    """The cap is a ROW count, so a key per row cannot move it.

    Asserted through the wire boundary rather than on the fold: the fold is
    unbounded by design and ``LIVE_EVENT_END_ROWS_MAX`` is what a mid-turn
    joiner actually receives.
    """
    store = FrontendStateStore(_state())
    events: list[AgentEvent[Any]] = [AgentStartEvent(generation=1)]
    total = LIVE_EVENT_END_ROWS_MAX + 25
    for index in range(total):
        events.append(
            ToolExecutionStartEvent(
                tool_call_id=f"call-{index}",
                tool_name="bash",
                args={},
                started_at_epoch=1_700_000_000.0 + index,
            )
        )
        events.append(
            ToolExecutionEndEvent(
                tool_call_id=f"call-{index}",
                tool_name="bash",
                result=ToolResult(tool_call_id=f"call-{index}", tool_name="bash", content=[]),
            )
        )
    _fold_live(store, *events)
    assert store.live_tool_start_epochs() == {}

    payload = sync_wire_payload(
        FrontendSync(
            epoch=store.state.epoch,
            sequence=store.state.sequence,
            snapshot=store.state,
            live_cursor=None,
        )
    )
    sent = payload["snapshot"]["live_events"]
    ends = [row for row in sent if row["type"] == "tool_execution_end"]

    assert len(ends) == LIVE_EVENT_END_ROWS_MAX
    assert len({row["tool_call_id"] for row in ends}) == len(ends), "a call was sent twice"
    # Oldest-first eviction, unchanged — and every survivor keeps its own clock.
    assert ends[-1]["tool_call_id"] == f"call-{total - 1}"
    assert all(
        row["started_at_epoch"] == 1_700_000_000.0 + int(row["tool_call_id"][5:]) for row in ends
    )


def test_the_phase_fold_restarts_only_when_the_kind_of_work_changes() -> None:
    """The band's zero, folded from the events that BEGIN a phase.

    Every case here is the phone projection's rule, which is the same row on
    another screen: a phase restarts the zero when it begins a KIND of work,
    never when it merely relabels one. The composed batch is the case that shows
    the difference — three calls announced in one dictation are one zero, and
    restarting per announcement would show "still composing" counting from zero
    three times over.
    """
    store = FrontendStateStore(_state())
    session = _live_session()

    def seen() -> tuple[str, float | None]:
        return store.activity_phase_clock()

    store.observe_event(session, AgentStartEvent(generation=1))
    phase, first = seen()
    assert phase == "thinking" and first is not None

    # A second compose event of the SAME dictation: no restart.
    store.observe_event(
        session, ToolCallComposeEvent(tool_call_id="c1", tool_name="bash", argument_bytes=10)
    )
    composing, composed_at = seen()
    assert composing == "composing"
    store.observe_event(
        session, ToolCallComposeEvent(tool_call_id="c2", tool_name="read", argument_bytes=10)
    )
    assert seen() == (composing, composed_at), "one dictation, one zero"

    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="c1", tool_name="bash", args={}, started_at_epoch=time.time()
        ),
    )
    store.observe_event(
        session,
        ToolExecutionStartEvent(
            tool_call_id="c2", tool_name="read", args={}, started_at_epoch=time.time()
        ),
    )
    running, running_at = seen()
    assert running == "running"

    # A SIBLING is still executing, so the batch has not gone back to waiting on
    # the model: restarting here would reset the number the surviving row's
    # label still claims.
    store.observe_event(
        session,
        ToolExecutionEndEvent(
            tool_call_id="c1",
            tool_name="bash",
            result=ToolResult(tool_call_id="c1", tool_name="bash", content=[]),
        ),
    )
    assert seen() == (running, running_at), "a batch with a live call is still running"

    # The LAST call of the batch: now the turn is waiting on the model again,
    # and the zero belongs to that wait.
    store.observe_event(
        session,
        ToolExecutionEndEvent(
            tool_call_id="c2",
            tool_name="read",
            result=ToolResult(tool_call_id="c2", tool_name="read", content=[]),
        ),
    )
    thinking, waited_at = seen()
    assert thinking == "thinking" and waited_at != running_at

    # A model call OPENING is its own phase edge, and the placeholder it yields
    # before the first token is the edge rather than the first token itself:
    # waiting on the model is what the turn is doing while nothing streams.
    store.observe_event(session, MessageStartEvent(message=Message.assistant("")))
    model_call, calling_at = seen()
    assert model_call == "thinking" and calling_at != waited_at

    # The first non-empty delta is the transition to prose; an empty one (a
    # repeat placeholder mid-stream) is not.
    store.observe_event(session, MessageUpdateEvent(message=Message.assistant(""), delta=""))
    assert seen() == (model_call, calling_at)
    store.observe_event(session, MessageUpdateEvent(message=Message.assistant("h"), delta="h"))
    responding, responded_at = seen()
    assert responding == "responding" and responded_at != calling_at

    store.observe_event(session, MessageEndEvent(message=Message.assistant("hi")))
    assert seen()[0] == "thinking"

    store.observe_event(session, AgentEndEvent(messages=[Message.assistant("hi")]))
    assert seen() == ("", None), "a settled turn has no phase to match"


def test_the_phase_pair_rides_the_wire_and_is_not_durable() -> None:
    """One round trip, and one strip — the two places a transient pair is read.

    The pair is carried in ``refresh_from_session`` under the streaming gate so
    a viewer attaching mid-turn receives the producer's own phase zero; it is
    NOT carried into the checkpoint, because a checkpoint describes a turn that
    has ended and there is no phase left to date.
    """
    import asyncio

    state = _state(
        streaming=True,
        activity_phase="thinking",
        activity_phase_started_at=1_700_000_000.0,
        live_tool_started_at={"call-1": 1_699_999_900.0},
    )
    store = FrontendStateStore(state)
    wire = sync_wire_payload(store.subscribe(lambda _u: None).sync)
    snapshot = wire["snapshot"]
    assert snapshot["activity_phase"] == "thinking"
    assert snapshot["activity_phase_started_at"] == 1_700_000_000.0
    assert snapshot["live_tool_started_at"] == {"call-1": 1_699_999_900.0}

    class _Transcript:
        def __init__(self) -> None:
            self.appended: list[tuple[str, dict[str, Any]]] = []

        async def append_custom(self, custom_type: str, payload: dict[str, Any]) -> None:
            self.appended.append((custom_type, payload))

    transcript = _Transcript()
    asyncio.run(store.checkpoint(transcript))
    ((_, payload),) = transcript.appended
    assert payload["state"]["live_tool_started_at"] == {}, (
        "the live anchors describe calls running NOW in this process; nothing "
        "in a durable checkpoint can restore them"
    )
    # The pair is a scalar the turn-end fold clears, so it is not stripped.
    assert payload["state"]["activity_phase"] == "thinking"


def test_a_restored_runtime_publishes_the_directory_IT_works_in() -> None:
    """The checkpoint says where a session USED to work; the runtime says where it does.

    ``/move`` (and the desktop's move route) rewrites the durable marker, and the
    canonical frontend checkpoint keeps naming the directory the PREVIOUS runtime
    worked in. A successor that restored ``cwd`` from the checkpoint published a
    ``frontend.cwd`` for a directory the session had LEFT: the receipt, the marker
    and a real ``bash pwd`` all named the new one while the stream named the old
    one — and the renderer's own rule is that the stream is authoritative, so it
    kept showing it until something forced a refresh (QA Q1 on the desktop move).
    """
    stored = _state(cwd="/evidence/before")
    owner = SimpleNamespace(
        session_id="s1",
        cwd="/evidence/after",
        _transcript=_CheckpointTranscript(stored),
    )

    restored = FrontendStateStore.from_checkpoint(owner).state

    assert restored.cwd == "/evidence/after"
    # Everything else the row carries is still the conversation's own durable
    # state; this is about one field, not a licence to drop the restore.
    assert restored.conversation_title == stored.conversation_title
    assert restored.cumulative_parent_cost == stored.cumulative_parent_cost


def test_a_host_with_no_directory_of_its_own_restores_the_checkpoints() -> None:
    """The fallback half: a reduced host restores exactly as it did before.

    ``_owner_over`` exposes neither ``cwd`` nor ``_cwd``, which is the shape every
    test double and any host that does not model a directory has — there the
    checkpoint's own value is still the only answer available.
    """
    stored = _state(cwd="/evidence/before")

    restored = FrontendStateStore.from_checkpoint(_owner_over("s1", stored)).state

    assert restored.cwd == "/evidence/before"


def test_the_at_ms_stamp_survives_the_wire_and_the_frozen_wrapper() -> None:
    """``Usage.at_ms`` is additive and optional, and must survive both hops.

    It travels the wire to the phone and back through a restore, and it is kept
    inside the immutable snapshot a shared job hands out — the two ways a
    recorded call reaches a pricing surface after the fact.
    """
    stamp = 1_700_000_000_123
    payload = _state(
        last_usage=FrontendUsage(input_tokens=1_000, output_tokens=0, at_ms=stamp)
    ).model_dump(mode="json")
    restored = FrontendSessionState.model_validate(payload)
    assert restored.last_usage is not None
    assert restored.last_usage.at_ms == stamp
    assert restored.model_dump(mode="json")["last_usage"]["at_ms"] == stamp

    # The frozen wrapper: a job's own usage is retained as an immutable value.
    job = JobState.model_validate(
        {
            "id": "child",
            "type": "task",
            "usage": Usage(input_tokens=4, output_tokens=2, at_ms=stamp).model_dump(mode="json"),
        }
    )
    snapshot = FrontendStateStore(_state(jobs=[job])).state
    assert snapshot.jobs[0].usage is not None
    assert snapshot.jobs[0].usage.at_ms == stamp

    # And an old transcript that lacks the field still validates: this is a
    # purely additive field, with no version bump and no migration.
    legacy = _state().model_dump(mode="json")
    legacy["last_usage"].pop("at_ms", None)
    older = FrontendSessionState.model_validate(legacy)
    assert older.last_usage is not None
    assert older.last_usage.at_ms is None


def test_a_restored_usage_prices_at_the_calls_window_not_at_the_viewers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why ``at_ms`` exists at all.

    A call made at 07:00 UTC and restored at noon would be halved if it were
    priced at view time (its token buckets never change but the window does), so
    the restored usage must price at the window ITS OWN stamp names. This is the
    surface that was a FLOOR before this change, not merely an approximation.
    """
    from datetime import datetime, timezone

    from local_operator.model import tariff
    from local_operator.model.configure import cost_for_usage
    from local_operator.model.registry import deepseek_models

    peak = datetime(2026, 9, 14, 7, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        tariff, "now_utc", lambda: datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
    )
    payload = _state(
        last_usage=FrontendUsage(
            input_tokens=1_000_000,
            output_tokens=0,
            at_ms=int(peak.timestamp() * 1000),
        )
    ).model_dump(mode="json")
    restored = FrontendSessionState.model_validate(payload)
    assert restored.last_usage is not None
    flash = deepseek_models["deepseek-flash"]
    assert cost_for_usage("deepseek", flash, restored.last_usage) == pytest.approx(0.30)


def test_the_published_catalogue_carries_a_rows_schedule() -> None:
    """MINOR 1: the follower's round trip must not drop the tariff.

    `refresh_model_catalogue` serializes the owner's rows key by key for a
    follower, and `tui/app.py` rebuilds `CatalogueEntry`s from those dicts. Without
    `time_of_use` an attached session rendered a tariffed row at its stored PEAK
    price with no window tag while the owner's own picker showed the rate in force
    — the two-surface disagreement the shared renderer exists to prevent. The
    reach is narrow (a row the follower already knows wins with its own entry), and
    it bites exactly when the owner publishes a row the follower's list lacks,
    which is the case this merge exists for.
    """
    store = FrontendStateStore(_state())
    entry = SimpleNamespace(
        provider="deepseek",
        model_id="deepseek-flash",
        label="DeepSeek Flash",
        context_window=1_000_000,
        default_context_window=None,
        max_context_window=None,
        input_price=0.30,
        output_price=1.20,
        connected=True,
        aggregated=False,
        routed=False,
        time_of_use="deepseek-tou",
    )
    store.refresh_model_catalogue([entry])

    (row,) = store.state.model_catalogue
    assert row["time_of_use"] == "deepseek-tou"
    # An older owner (or a duck-typed entry from an embedding host) that does not
    # publish the key reads back as None, which is the honest "no time-of-day
    # structure known" rather than a crash or a wrong default.
    plain = SimpleNamespace(**{k: v for k, v in vars(entry).items() if k != "time_of_use"})
    store.refresh_model_catalogue([plain])
    (row,) = store.state.model_catalogue
    assert row["time_of_use"] is None


# ---------------------------------------------------------------------------
# Off-loop subscribe: the ordering proof and the invariant it rests on
# ---------------------------------------------------------------------------
#
# These two guard the seam a busy owner is served through
# (``FrontendStateStore.subscribe_threadsafe`` ←
# ``ServingSessionHandle.subscribe_frontend_nowait``). Registration used to be
# atomic only because exactly one thread ever touched the store; the off-loop
# path admits a subscriber from the runtime's thread while the owner's loop is
# mid-turn, so the lock in the store is now the mechanism and these are what
# hold it to its contract.


#: How many deltas the ordering test publishes AFTER the last joiner is in.
#: ``subscribe_threadsafe`` is only proven differential if a subscriber admitted
#: at the very worst moment still receives a stream to check, so this is what
#: makes the contiguity assertion non-vacuous.
_TAIL_PUBLISHES = 25


def test_off_loop_subscribers_see_every_delta_once_across_a_live_publisher() -> None:
    """One publisher against 50 joiners racing it: no gap, no duplicate.

    WHY THIS IS DIFFERENTIAL RATHER THAN A UNIT CHECK. A joiner runs against a
    publisher that is already publishing and races its critical section. Exactly
    two outcomes are legal: the joiner is in the list that publish snapshotted,
    so it receives the sequence AFTER the one it captured; or it is not, so the
    one it captured IS that sequence. Both leave the stream contiguous from the
    joiner's own sequence, and anything else shows up here as a gap or a repeat
    — which is what the client's exact-``+1`` check turns into a redial, so a
    silent violation is a reconnect loop in production rather than a wrong
    number.

    The publisher is bounded by a COUNT, never by a clock: the assertion is about
    ordering, and a time bound would make it a bet on this host's load.

    WHAT THIS ONE IS NOT: the guard that reliably goes red. The split it looks
    for is a couple of bytecodes wide at the default 5 ms switch interval —
    measured, the pre-change shape produces one duplicate in six runs of this loop
    — so this is the wide net over a REAL concurrent schedule, and the forced
    schedule in the companion test below is what pins the mechanism. Both are
    kept, because a wide net over real threads is the only thing here that runs
    the schedule production actually has.
    """
    store = FrontendStateStore(_state())
    subscribers = 50
    joiners_ready = threading.Barrier(subscribers + 1)
    all_joined = threading.Event()
    outstanding = [subscribers]
    outstanding_lock = threading.Lock()
    stop = threading.Event()
    deliveries: list[list[int]] = [[] for _ in range(subscribers)]
    # Indexed by the thread's own slot rather than appended: threads finish in an
    # arbitrary order, and a pair of append-order lists would zip a subscriber's
    # sequence against another subscriber's deliveries.
    bases: list[int] = [0] * subscribers

    def subscribe_and_hold(index: int) -> None:
        joiners_ready.wait()
        seen = deliveries[index]

        def on_update(update: FrontendUpdate) -> None:
            # Called from the PUBLISHER's thread; ``seen`` is read only after
            # that thread has been joined.
            seen.append(update.sequence)

        subscription = store.subscribe_threadsafe(on_update)
        bases[index] = subscription.sync.sequence
        with outstanding_lock:
            outstanding[0] -= 1
            if not outstanding[0]:
                all_joined.set()
        while not stop.is_set():
            time.sleep(0.0005)

    threads = [
        threading.Thread(target=subscribe_and_hold, args=(index,)) for index in range(subscribers)
    ]
    for thread in threads:
        thread.start()

    def publish() -> None:
        joiners_ready.wait()
        n = 0
        while not all_joined.is_set() and n < 4000:
            if store.mutate(goal=f"goal-{n}") is not None:
                n += 1
        # EVERY JOINER IS IN (``all_joined`` is set only after the last one's
        # subscribe returned), so publish a known number MORE. That is what makes
        # the contiguity assertion non-vacuous: a subscriber admitted at the last
        # possible instant still receives all of these, and none of them can be
        # skipped by the publisher stopping at the same moment.
        for extra in range(_TAIL_PUBLISHES):
            store.mutate(goal=f"tail-{extra}")
        stop.set()

    publisher = threading.Thread(target=publish)
    publisher.start()
    publisher.join(timeout=60)
    assert not publisher.is_alive(), "the publisher thread did not finish"
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "a subscriber thread did not finish"

    assert outstanding[0] == 0, "a joiner never completed its subscribe"
    assert min(len(seen) for seen in deliveries) >= _TAIL_PUBLISHES, (
        "a subscriber admitted before the publisher's tail received less than the "
        "tail — the test is not differential"
    )
    for seen, base in zip(deliveries, bases, strict=True):
        assert seen == list(
            range(base + 1, base + 1 + len(seen))
        ), f"subscriber at sequence {base} saw {seen[:8]}… — a gap or a duplicate"


def test_canonical_state_is_replaced_never_mutated_in_place() -> None:
    """The invariant the off-loop snapshot clone rests on.

    ``subscribe_threadsafe`` deep-copies ``_state`` OUTSIDE the publish lock —
    that is what keeps a publisher from ever waiting on a joiner's snapshot — and
    it is only safe while no publisher EDITS the object a reader already holds.
    A state mutated in place would let a concurrent clone observe a
    half-applied update under a sequence number that says it did not happen, and
    nothing downstream could detect it.

    So this drives every publishing path and asserts the object captured before
    each one is byte-identical afterwards AND that the store installed a
    different object. Both halves are needed: identity alone would pass for an
    in-place edit that happens to be followed by a copy, and content alone would
    pass for a mutation that leaves the captured object equal.
    """

    def drive(store: FrontendStateStore, captured: list[tuple[str, Any, dict[str, Any]]]) -> None:
        def capture(label: str) -> None:
            ref = store._state
            captured.append((label, ref, ref.model_dump(mode="json")))

        capture("construction")
        store.mutate(goal="through mutate")
        capture("mutate")
        store.replace(_state(goal="through replace"))
        capture("replace")
        store.replace_and_notify(_state(goal="through replace_and_notify"))
        capture("replace_and_notify")
        assert store.seed_job_trajectory(
            "j1", [{"type": "message_update", "delta": "hi"}]
        ), "the seed found no job to install into"
        capture("seed_job_trajectory")
        assert store.seed_job_todos(
            "j1", [{"text": "wire it"}], epoch="owner-a", sequence=1, session_id=None
        ), "the seed found no job to install into"
        capture("seed_job_todos")

    captured: list[tuple[str, Any, dict[str, Any]]] = []
    store = FrontendStateStore(_state())
    drive(store, captured)

    # The applied-delta path too, since it is the one a VIEWER runs: its state
    # is built from a validated candidate rather than from a snapshot payload.
    follower = FrontendStateStore(_state())
    delta = store.mutate(goal="from the producer")
    assert delta is not None, "the producer published nothing to apply"
    follower_ref = follower._state
    follower_before = follower_ref.model_dump(mode="json")
    follower.apply_update(copy.deepcopy(delta))
    captured.append(("apply_update", follower_ref, follower_before))

    for label, ref, snapshot in captured:
        assert ref.model_dump(mode="json") == snapshot, (
            f"the canonical state captured before {label} was edited in place — the "
            "off-loop clone is only sound while publishes REPLACE it"
        )
    identities = [id(ref) for _label, ref, _snapshot in captured]
    assert len(set(identities)) == len(
        identities
    ), "a publish reused the state object instead of installing a new one"

    # And what the store HANDS OUT is a clone, not the object it publishes from:
    # editing it must not reach canonical state.
    public = store.state
    public.goal = "edited by a caller"
    assert store.state.goal != "edited by a caller"

    # The RETAINED rows are frozen at the model level, which is the second half
    # of the same invariant: a reader holding a job cannot edit the store's copy
    # of it either.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        store._state.jobs[0].status = "edited by a caller"  # type: ignore[misc]


class _AppendSwitchesThreads(list[Any]):
    """A subscriber list whose ``append`` yields the GIL at the worst moment.

    WHY A FORCED SCHEDULE RATHER THAN A RACED ONE. The property under test —
    "registering a callback and capturing the sequence are ONE step as far as a
    publisher is concerned" — is only observable if a publisher runs BETWEEN the
    two, and at the default 5 ms switch interval that window is a couple of
    bytecodes wide. Measured on the pre-change shape (append, then read the
    state, no lock): one duplicate in six runs of the 50-subscriber loop above.
    A guard that fires one time in six is not a guard.

    ``append`` therefore blocks for a real switch interval. That is not a trick
    played on a correct store: the appended callback is registered with
    ``_publish_lock`` held, so a publisher cannot run during the sleep at all and
    the delay is invisible in the outcome. It is only observable by a store that
    appended WITHOUT the lock, which is exactly the shape this pins.
    """

    def __init__(self, values, *, switch_s: float = 0.02) -> None:  # noqa: ANN001
        super().__init__(values)
        self.switch_s = switch_s
        self.sleeps = 0

    def append(self, value) -> None:  # noqa: ANN001
        super().append(value)
        self.sleeps += 1
        time.sleep(self.switch_s)


def test_a_join_is_one_step_to_a_publisher_even_at_the_worst_instruction() -> None:
    """The lock's contract, forced rather than raced.

    The companion to the differential loop above, and the one that can actually
    go red: it puts a publisher's whole publish (state install AND subscriber
    list) inside the window a joiner opens between registering and capturing —
    the schedule a loaded fleet produces and the 5 ms switch interval hides.

    A correct store blocks that publisher on the lock, so the joiner captures the
    OLD sequence and then receives exactly that publish; a store that registered
    without the lock lets the publish through first, the joiner captures the NEW
    sequence AND is in that publish's list, and it receives the sequence it
    already holds as its own — a duplicate the client reads as a gap and
    redials over. Both outcomes are asserted, so the test names the failure it
    prevents rather than only the absence of one.
    """
    store = FrontendStateStore(_state())
    assert store.mutate(goal="before the join") is not None
    base_sequence = store.state.sequence
    injected = _AppendSwitchesThreads(store._subscribers)
    store._subscribers = injected

    seen: list[int] = []
    captured: list[int] = []

    def join() -> None:
        subscription = store.subscribe_threadsafe(lambda update: seen.append(update.sequence))
        captured.append(subscription.sync.sequence)

    joiner = threading.Thread(target=join)
    joiner.start()
    # Let the joiner reach the append (and the sleep inside it) before publishing
    # from THIS thread: the publish has to arrive while the registration is in
    # flight, which is the whole point of the injected switch.
    deadline = time.monotonic() + 5.0
    while injected.sleeps == 0 and time.monotonic() < deadline:
        time.sleep(0.0005)
    assert injected.sleeps == 1, "the joiner never reached its registration"
    assert store.mutate(goal="while the join is in flight") is not None
    joiner.join(timeout=10)
    assert not joiner.is_alive(), "the joiner never returned"

    assert captured == [base_sequence], (
        f"the joiner captured sequence {captured} instead of {base_sequence} — its "
        "registration did not take effect before the publisher's list was taken"
    )
    assert seen == [base_sequence + 1], (
        f"the joiner saw {seen} — a publisher that ran between its registration and "
        "its snapshot gives it the sequence it already holds, which the client's "
        "exact-+1 check reads as a gap"
    )


def test_the_turn_end_refresh_keeps_the_decode_window_on_last_usage() -> None:
    """The frame's ``last_usage`` must still carry the window once the turn ends.

    ``refresh_from_session`` runs from ``observe_event`` at the END of the message
    and of the turn — after the branches that record usage — so when its own
    ``last_usage`` came from a plain ``model_dump`` of ``restored_usage()``, it
    overwrote the materialised pair with a payload that cannot carry a private
    attribute. Measured by review round 2: ``decode_us`` absent after BOTH events,
    while the ledger held the call's window. That is the worst moment to lose it —
    "the last completed call decoded at N tok/s" is exactly what a status band
    wants to show when a turn settles.

    Four steps, and the third is the one that distinguishes the winning writer: the
    turn's aggregate sums to a DIFFERENT output token count than the call's, so a
    ``last_usage`` carrying the sum is a different failure with the same symptom.
    """
    usage = Usage(input_tokens=100, output_tokens=240, context_tokens=100)
    usage._decode_window = (12_345, 240)
    sibling = Usage(input_tokens=10, output_tokens=300, context_tokens=100)
    session = SimpleNamespace(effective_model=_spec(), restored_usage=lambda: usage)
    store = FrontendStateStore(FrontendSessionState(session_id="s1", epoch="e1"))
    store.observe_event(session, AgentStartEvent(generation=1))

    # 2 — the message boundary.
    store.observe_event(session, MessageEndEvent(message=Message.assistant("a", usage=usage)))
    after_message = store.state.last_usage
    assert after_message is not None
    assert (
        after_message.model_dump().get("decode_us") == 12_345
    ), "the message-end refresh dropped the decode window"

    # 3 — the turn boundary, where the aggregate would win if the refresh did not.
    store.observe_event(
        session,
        AgentEndEvent(
            messages=[
                Message.assistant("a", usage=usage),
                Message.assistant("b", usage=sibling),
            ]
        ),
    )
    last = store.state.last_usage
    assert last is not None
    dumped = last.model_dump()
    assert dumped.get("decode_us") == 12_345, (
        "the turn-end refresh dropped the decode window: the band would read unknown "
        "for a call the ledger measured"
    )
    assert dumped.get("decode_tokens") == 240
    # The refresh's LIVE reading wins over the aggregate, which is what makes the
    # pair survivable at all: a per-call window is not summable, so a last_usage
    # carrying the summed 540 would have nowhere to have got one.
    assert dumped.get("output_tokens") == 240, (
        "last_usage carries the turn aggregate (540); the window cannot be summed, "
        "so the pair and the aggregate can never both be right"
    )

    # 4 — the no-op rule: an unstamped usage adds NOTHING, because a zeroed pair
    # would spend the frame's slack on every frame to say nothing.
    plain = Usage(input_tokens=1, output_tokens=2, context_tokens=3)
    fresh = SimpleNamespace(effective_model=_spec(), restored_usage=lambda: plain)
    store2 = FrontendStateStore(FrontendSessionState(session_id="s2", epoch="e1"))
    store2.observe_event(fresh, AgentStartEvent(generation=1))
    store2.observe_event(fresh, AgentEndEvent(messages=[Message.assistant("b", usage=plain)]))
    last2 = store2.state.last_usage
    assert last2 is not None
    dumped2 = last2.model_dump()
    assert "decode_us" not in dumped2 and "decode_tokens" not in dumped2
