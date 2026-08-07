"""Session facade tests: turn flow, events, steering, abort, wake wiring,
compaction hook, dispose."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable, Callable, Sequence

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentTool,
    ChatRequest,
    CompactionEndEvent,
    CompactionStartEvent,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


class ScriptedStream:
    """Replays per-call event scripts; records requests."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def echo_tool(executed: list[str], delay: float = 0.0, name: str = "echo") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        if delay:
            await asyncio.sleep(delay)
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(
        name=name,
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )


def make_session(tmp_path, stream, tools=None, **kwargs) -> Session:
    transcript = Transcript(tmp_path / "sess")
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools or [],
        transcript=transcript,
        system_blocks_provider=kwargs.pop("system_blocks_provider", lambda: ["stable", "env"]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_prompt_full_turn_events_and_persistence(tmp_path):
    """prompt() drives a text→tool→text turn; handlers see ordered events;
    every produced message lands in the transcript."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="Hi"),
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Bye"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[echo_tool(executed)])

    events: list[AgentEvent] = []
    session.subscribe(events.append)

    await session.prompt("hello")

    assert executed == ["echo"]
    assert events[0].type == "agent_start"
    assert events[-1].type == "agent_end"
    assert isinstance(events[-1], AgentEndEvent)
    assert events[-1].aborted is False
    assert session.is_streaming is False

    # Transcript: user msg + assistant1 + tool result + assistant2.
    message_entries = [e for e in session._transcript.entries() if e.type == "message"]
    assert len(message_entries) == 4

    # Context carries the same four messages.
    assert len(session._context.messages) == 4

    # System blocks reached the provider.
    assert stream.requests[0].system_blocks == ["stable", "env"]
    await session.dispose()


@pytest.mark.asyncio
async def test_subscribe_sync_async_and_exception_isolation(tmp_path):
    """Sync and async handlers both receive events in order; a raising handler
    is isolated with a warning and never breaks the others."""
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    seen_sync: list[str] = []
    seen_async: list[str] = []
    broken_calls: list[int] = []

    def broken(event):
        broken_calls.append(1)
        raise RuntimeError("handler exploded")

    async def async_handler(event):
        seen_async.append(event.type)

    session.subscribe(lambda event: seen_sync.append(event.type))
    session.subscribe(broken)
    unsubscribe = session.subscribe(async_handler)

    await session.prompt("hi")

    assert len(broken_calls) == len(seen_sync)  # broken ran for every event...
    assert seen_sync == seen_async  # ...without breaking the others' ordering
    assert seen_sync[0] == "agent_start" and seen_sync[-1] == "agent_end"

    seen_sync.clear()
    seen_async.clear()
    broken_calls.clear()
    unsubscribe()
    await session.prompt("again")
    # Unsubscribed async handler stops receiving; the remaining two still fire.
    assert seen_async == []
    assert "agent_end" in seen_sync
    assert len(broken_calls) == len(seen_sync)
    await session.dispose()


@pytest.mark.asyncio
async def test_steer_interrupts_and_persists(tmp_path):
    """steer() during a tool batch: the steering message reaches the next
    model call and is persisted to the transcript."""
    executed: list[str] = []
    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        started.set()
        await asyncio.sleep(0.05)
        executed.append("echo")
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    tool = AgentTool(
        name="echo",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        execute=execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="adjusted"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])

    prompt_task = asyncio.ensure_future(session.prompt("start"))
    await started.wait()
    session.steer("change direction")
    await prompt_task

    # Steering drained mid-turn and reached the second model call.
    assert len(stream.requests) == 2
    assert any(
        isinstance(m, Message) and m.text == "change direction" for m in stream.requests[1].messages
    )
    texts = [
        block.get("text")
        for entry in session._transcript.entries()
        if entry.type == "message" and entry.payload.get("kind") == "message"
        for block in entry.payload.get("content", [])
    ]
    assert "change direction" in texts
    await session.dispose()


@pytest.mark.asyncio
async def test_abort_emits_aborted_agent_end(tmp_path):
    """abort() mid-stream: the turn ends with an aborted agent_end."""
    started = asyncio.Event()

    async def slow_stream(request, signal):
        started.set()
        assert signal is not None
        await signal.wait()
        yield StreamEndEvent(stop_reason="aborted")

    session = make_session(tmp_path, slow_stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    prompt_task = asyncio.ensure_future(session.prompt("long task"))
    await started.wait()
    session.abort("user cancelled")
    await prompt_task

    end = events[-1]
    assert isinstance(end, AgentEndEvent)
    assert end.aborted is True
    await session.dispose()


@pytest.mark.asyncio
async def test_yolo_disables_approval(tmp_path):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    approvals: list[str] = []

    async def approve(name, summary):
        approvals.append(name)
        return True

    session = make_session(tmp_path, stream, yolo=True, request_approval=approve)
    context = session._build_tool_context()
    assert context.request_approval is None
    assert approvals == []

    strict = make_session(tmp_path, stream, request_approval=approve, session_id="s2")
    assert strict._build_tool_context().request_approval is not None
    await session.dispose()
    await strict.dispose()


@pytest.mark.asyncio
async def test_wake_schedule_persistence_and_reload(tmp_path):
    """set_wake_schedules persists a wake_schedules custom entry; a reopened
    session loads them (newest entry wins)."""
    from local_operator.harness.wake import WakeSchedule

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    schedule = WakeSchedule(
        id="w1", message="check in", next_due_at=1_700_000_060_000, created_at=1_700_000_000_000
    )
    await session.set_wake_schedules([schedule])
    assert len(session.wake_scheduler.schedules) == 1

    details = session._transcript.latest_custom("wake_schedules")
    assert details is not None and len(details["schedules"]) == 1

    # Reopen: fresh Session over the same transcript adopts the schedules.
    reopened = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )
    assert len(reopened.wake_scheduler.schedules) == 1
    assert reopened.wake_scheduler.schedules[0].id == "w1"
    await session.dispose()
    await reopened.dispose()


@pytest.mark.asyncio
async def test_wake_delivery_goes_through_prompt_path(tmp_path):
    """A fired wake is delivered as a user-attributed wake_prompt custom
    message through the prompt machinery."""
    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    from local_operator.harness.wake import DueWake, WakeSchedule

    schedule = WakeSchedule(id="w1", message="wake up", next_due_at=0, created_at=0)
    due = DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
    await session._deliver_wake(due)
    await wait_for(lambda: bool(stream.requests))  # the spawned turn ran

    delivered = stream.requests[0].messages
    assert any(m.text.startswith("(alarm) Scheduled wake w1") for m in delivered)
    assert "wake up" in delivered[-1].text

    # The wake_prompt message was persisted with user attribution.
    wake_entries = [
        e
        for e in session._transcript.entries()
        if e.type == "message" and e.payload.get("kind") == "custom"
    ]
    assert wake_entries
    assert wake_entries[0].payload["attribution"] == "user"
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_runs_when_due(tmp_path, monkeypatch):
    """Post-turn compaction: prune -> trigger (compaction_context_tokens) ->
    summarize -> transcript entry -> context rebuilt to marker + kept messages.
    Default threshold min(window*0.8, 600_000) applies at the call site."""
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = -1.0
        threshold_tokens: int = Field(default=-1)
        auto_continue: bool = True

    prune_calls: list[tuple[int, int]] = []

    def prune_tool_outputs(messages, now_ts, last_activity_ts, **kwargs):
        prune_calls.append((now_ts, last_activity_ts))
        return list(messages), False

    fake_api = types.ModuleType("local_operator.compaction.api")
    setattr(fake_api, "CompactionSettings", CompactionSettings)
    setattr(fake_api, "prune_tool_outputs", prune_tool_outputs)
    setattr(fake_api, "estimate_messages_tokens", lambda messages: 90_000)
    # Rigorous upper bound (>= the exact estimate) used by the cheap pre-check
    # that keeps tiktoken off the no-compaction path.
    setattr(fake_api, "messages_tokens_upper_bound", lambda messages: 95_000)
    setattr(
        fake_api, "compaction_context_tokens", lambda provider, local: max(provider or 0, local)
    )
    setattr(fake_api, "find_cut_point", lambda messages, keep: 1)  # cut after first message
    setattr(
        fake_api, "resolve_threshold_tokens", lambda window, settings: settings.threshold_tokens
    )
    setattr(fake_api, "RECOVERY_BAND", 0.8)

    seen_threshold: list[int] = []

    def should_compact(ctx_tokens, window, settings):
        seen_threshold.append(settings.threshold_tokens)
        return ctx_tokens > settings.threshold_tokens

    setattr(fake_api, "should_compact", should_compact)

    async def summarize(
        messages: Sequence[Message | CustomMessage],
        complete_fn: Callable[[str, str], Awaitable[str]],
    ) -> str:
        assert callable(complete_fn)
        summary = await complete_fn("sys", "summarize this")
        return f"SUMMARY({summary})"

    setattr(fake_api, "summarize_messages", summarize)

    fake_pkg = types.ModuleType("local_operator.compaction")
    setattr(fake_pkg, "api", fake_api)
    monkeypatch.setitem(sys.modules, "local_operator.compaction", fake_pkg)
    monkeypatch.setitem(sys.modules, "local_operator.compaction.api", fake_api)
    # Stub out strategy resolution to context-full (no snapcompact module).
    fake_thresholds = types.ModuleType("local_operator.compaction.thresholds")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.thresholds", fake_thresholds)
    fake_snap = types.ModuleType("local_operator.compaction.snapcompact")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.snapcompact", fake_snap)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="turn one"), StreamEndEvent(stop_reason="stop")],
            # The one-shot summary call (tool_choice none).
            [StreamTextDelta(delta="compressed"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())

    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("do work")

    # Default threshold applied at the call site: min(100_000 * 0.8, 600_000).
    # Twice, because the trigger is two-stage: the cheap upper bound is tested
    # first and only a bound that clears the threshold buys the exact estimate.
    assert seen_threshold == [80_000, 80_000]
    # Prune ran BEFORE the trigger with millisecond timestamps.
    assert prune_calls and prune_calls[0][0] > 10**12

    # Compaction summary call was issued with no tools.
    summary_request = stream.requests[1]
    assert summary_request.tools == []
    assert summary_request.tool_choice == "none"

    # Context rebuilt: marker + the kept assistant message (cut=1 keeps it).
    assert len(session._context.messages) == 2
    marker = session._context.messages[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    assert marker.details["summary"] == "SUMMARY(compressed)"

    # Transcript got a compaction entry.
    compactions = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert len(compactions) == 1
    assert compactions[0].payload["summary"] == "SUMMARY(compressed)"

    assert [e.type for e in events if e.type.startswith("compaction")] == [
        "compaction_start",
        "compaction_end",
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_below_bound_never_pays_for_the_exact_estimate(tmp_path, monkeypatch):
    """A conversation the cheap UPPER bound already clears must not reach the
    exact estimator.

    The exact estimator is what loads tiktoken's cl100k_base table (~84 ms and
    ~43.6 MB RSS), and compaction runs after every turn, so a session that
    never approaches its threshold must never touch it. Asserting the call
    simply did not happen is the only way to pin that: the outcome (no
    compaction) is identical either way, so a behavioural assertion cannot
    tell the fast path from the slow one.
    """
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = -1.0
        threshold_tokens: int = Field(default=-1)
        auto_continue: bool = True

    exact_calls: list[int] = []

    fake_api = types.ModuleType("local_operator.compaction.api")
    setattr(fake_api, "CompactionSettings", CompactionSettings)
    setattr(fake_api, "prune_tool_outputs", lambda messages, *a, **k: (list(messages), False))
    setattr(fake_api, "messages_tokens_upper_bound", lambda messages: 1_000)

    def estimate_messages_tokens(messages):
        exact_calls.append(len(messages))
        return 1_000

    setattr(fake_api, "estimate_messages_tokens", estimate_messages_tokens)
    setattr(
        fake_api, "compaction_context_tokens", lambda provider, local: max(provider or 0, local)
    )
    setattr(
        fake_api, "resolve_threshold_tokens", lambda window, settings: settings.threshold_tokens
    )
    setattr(
        fake_api, "should_compact", lambda ctx, window, settings: ctx > settings.threshold_tokens
    )
    setattr(fake_api, "RECOVERY_BAND", 0.8)

    fake_pkg = types.ModuleType("local_operator.compaction")
    setattr(fake_pkg, "api", fake_api)
    monkeypatch.setitem(sys.modules, "local_operator.compaction", fake_pkg)
    monkeypatch.setitem(sys.modules, "local_operator.compaction.api", fake_api)
    fake_thresholds = types.ModuleType("local_operator.compaction.thresholds")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.thresholds", fake_thresholds)
    fake_snap = types.ModuleType("local_operator.compaction.snapcompact")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.snapcompact", fake_snap)

    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    await session.prompt("do work")

    assert exact_calls == []  # bound alone settled it; tiktoken never needed
    assert [e.type for e in session._transcript.entries() if e.type == "compaction"] == []
    await session.dispose()


@pytest.mark.asyncio
async def test_no_compaction_module_degrades_gracefully(tmp_path, monkeypatch):
    """Without the compaction package, turns run without compaction."""
    monkeypatch.setitem(sys.modules, "local_operator.compaction", None)
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session.prompt("hi")
    compactions = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert compactions == []
    await session.dispose()


@pytest.mark.asyncio
async def test_events_carry_monotonic_generation(tmp_path):
    """agent_start/agent_end carry the per-session turn generation (TUI
    supersede guard)."""
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="one"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="two"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("first")
    await session.prompt("second")

    gens = [
        (e.type, e.generation) for e in events if isinstance(e, (AgentStartEvent, AgentEndEvent))
    ]
    assert gens == [("agent_start", 1), ("agent_end", 1), ("agent_start", 2), ("agent_end", 2)]
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_preserve_data_round_trip(tmp_path):
    """append_compaction stores preserve_data; replay surfaces it on the
    marker details."""
    transcript = Transcript(tmp_path / "sess")
    m1 = Message.user("before")
    await transcript.append_message(m1)
    keep = Message.user("kept")
    entry = await transcript.append_message(keep)
    await transcript.append_compaction(
        "S", entry.id, 100, preserve_data={"snapcompact": {"text": "archive"}}
    )
    compactions = [e for e in transcript.entries() if e.type == "compaction"]
    assert compactions[0].payload["preserve_data"] == {"snapcompact": {"text": "archive"}}

    history = transcript.build_llm_history()
    marker = history[0]
    assert isinstance(marker, CustomMessage)
    assert marker.details["preserve_data"] == {"snapcompact": {"text": "archive"}}


@pytest.mark.asyncio
async def test_dispose_cancels_jobs_and_wakes(tmp_path):
    stream = ScriptedStream([])
    session = make_session(tmp_path, stream)

    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()

    job_id = session.jobs.register("task", "bg", blocked)
    await session.dispose()
    job = session.jobs.get(job_id)
    assert job is not None
    assert job.status == "cancelled"
    assert session.wake_scheduler.disposed is True

    with pytest.raises(RuntimeError):
        await session.prompt("after dispose")


@pytest.mark.asyncio
async def test_async_system_blocks_provider(tmp_path):
    """system_blocks_provider may be async (skill selection needs await)."""

    async def provider():
        await asyncio.sleep(0)
        return ["block-a", "block-b"]

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, system_blocks_provider=provider)
    await session.prompt("hi")
    assert stream.requests[0].system_blocks == ["block-a", "block-b"]
    await session.dispose()


@pytest.mark.asyncio
async def test_model_label_and_ids(tmp_path):
    stream = ScriptedStream([])
    session = make_session(tmp_path, stream, session_id="sess-42", agent_id="Sub")
    assert session.session_id == "sess-42"
    assert session.agent_id == "Sub"
    assert session.model_label == "test/m"
    await session.dispose()


@pytest.mark.asyncio
async def test_refresh_tools_takes_effect_next_model_call(tmp_path):
    """refresh_tools (MCP-20 hook): swapping the inventory is live from the
    next model call onward — the refreshed set is what the provider sees."""

    def mk(name: str) -> AgentTool:
        async def execute(tool_call_id, args, signal, on_update, context):
            return ToolResult(
                tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="x")]
            )

        return AgentTool(name=name, execute=execute)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="first"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="second"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[])
    await session.prompt("one")
    assert [t.name for t in stream.requests[0].tools] == []

    session.refresh_tools([mk("mcp__late_one"), mk("mcp__late_two")])
    await session.prompt("two")
    assert [t.name for t in stream.requests[1].tools] == ["mcp__late_one", "mcp__late_two"]
    await session.dispose()


@pytest.mark.asyncio
async def test_fallback_tool_resolver_dispatches_deferred_tool(tmp_path):
    """set_fallback_tool_resolver: a call to a name NOT in the inventory is
    routed through the resolver and executed (deferred MCP tools)."""
    executed: list[str] = []
    deferred = echo_tool(executed, name="deferred_mcp")

    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="deferred_mcp", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[])  # inventory empty
    session.set_fallback_tool_resolver(lambda name: deferred if name == "deferred_mcp" else None)
    await session.prompt("go")
    assert executed == ["deferred_mcp"]
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_continuation_emits_one_run_boundary(tmp_path, monkeypatch):
    """A post-compaction continuation is the SAME logical run.

    Compaction runs after the loop has already yielded agent_end, so forwarding
    that end plus the continuation run's agent_start would tell every UI the
    task finished and immediately restarted. The session holds the boundary
    events: exactly one agent_start and one agent_end per prompt, with the
    compaction events and the continuation's turns in between.
    """
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="first"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="resumed"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    # Force exactly one compaction that schedules exactly one continuation.
    calls = {"n": 0}
    original = session._maybe_compact

    async def fake_compact() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            await session._emit(CompactionStartEvent(reason="context-window"))
            await session._emit(CompactionEndEvent(reason="context-window", success=True))
            session._continuation_queue.append(Message.user("continue"))

    monkeypatch.setattr(session, "_maybe_compact", fake_compact)
    assert original is not None  # sanity: we replaced a real method

    seen: list[str] = []
    session.subscribe(lambda event: seen.append(type(event).__name__))
    await session.prompt("do the thing")

    assert seen.count("AgentStartEvent") == 1
    assert seen.count("AgentEndEvent") == 1
    # Ordering: the single end lands after the compaction pair, and the
    # continuation's turns sit between them.
    end_at = seen.index("AgentEndEvent")
    assert seen.index("CompactionEndEvent") < end_at
    assert seen.index("AgentStartEvent") < seen.index("CompactionStartEvent")
    assert seen.count("TurnStartEvent") == 2  # original turn + continuation
    assert len(stream.requests) == 2

    await session.dispose()


@pytest.mark.asyncio
async def test_aborted_run_end_is_never_held(tmp_path, monkeypatch):
    """An aborted run is a real boundary: it must surface immediately rather
    than waiting on a compaction pass that may never queue a continuation."""
    stream = ScriptedStream([[StreamTextDelta(delta="x"), StreamEndEvent(stop_reason="aborted")]])
    session = make_session(tmp_path, stream)

    async def no_compact() -> None:
        return None

    monkeypatch.setattr(session, "_maybe_compact", no_compact)
    ends: list[AgentEndEvent] = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    await session.prompt("go")
    assert len(ends) == 1
    assert ends[0].aborted is True
    await session.dispose()
