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
    ImageContent,
    Message,
    ModelSpec,
    NoticeEvent,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.providers.failover import ProviderError
from local_operator.session.session import IMAGE_DROPPED_NOTICE, Session
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


class PreflightStream(ScriptedStream):
    def __init__(self) -> None:
        super().__init__([[StreamEndEvent(stop_reason="stop")]])
        self.notice_handler = None
        self.preflight_models: list[ModelSpec] = []

    def set_notice_handler(self, handler) -> None:
        self.notice_handler = handler

    async def preflight_usage(self, model: ModelSpec) -> None:
        self.preflight_models.append(model)
        assert self.notice_handler is not None
        await self.notice_handler("anthropic quota low — falling back to openai", "warning")


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
async def test_usage_preflight_warns_without_starting_model_turn(tmp_path):
    stream = PreflightStream()
    session = make_session(tmp_path, stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    await session.preflight_usage()

    assert stream.preflight_models == [MODEL]
    assert stream.requests == []
    assert [event.type for event in events] == ["notice"]
    assert isinstance(events[0], NoticeEvent)
    assert events[0].text == "anthropic quota low — falling back to openai"


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
        max_threshold_tokens: int = 600_000
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
        fake_api,
        "resolve_threshold_tokens",
        lambda window, settings: (
            settings.threshold_tokens
            if settings.threshold_tokens > 0
            else min(int(window * 0.8), getattr(settings, "max_threshold_tokens", 600_000))
        ),
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
        max_threshold_tokens: int = 600_000
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
        fake_api,
        "resolve_threshold_tokens",
        lambda window, settings: (
            settings.threshold_tokens
            if settings.threshold_tokens > 0
            else min(int(window * 0.8), getattr(settings, "max_threshold_tokens", 600_000))
        ),
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
    # A session always self-merges its OWN capability tools (task/wait/jobs/
    # wake are createIf-gated on the session's launcher + job manager), so the
    # construction-time inventory is not empty even when the caller passed [].
    first_names = [t.name for t in stream.requests[0].tools]
    assert "task" in first_names and "wait" in first_names
    assert "mcp__late_one" not in first_names

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


@pytest.mark.asyncio
async def test_session_merges_capability_tools_into_inventory(tmp_path):
    """task/wait/jobs (and wake) are gated on the session's OWN capabilities
    (subagent_launcher/jobs/wake_scheduler), which the factory context lacks.
    A session must merge them at construction so the model can delegate even
    when the factory inventory was built without the engine (reproduced live:
    144 requests and 4M input tokens with zero task calls because task was
    never advertised)."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    names = [t.name for t in session._tools]
    assert "task" in names, f"task tool missing from inventory: {names}"
    assert "wait" in names
    assert "jobs" in names
    # The merge replaced/added without duplicating existing tools.
    assert len(names) == len(set(names))
    # And the merged tools are the createIf ones, wired to this session.
    task_tool = next((t for t in session._tools if t.name == "task"), None)
    assert task_tool is not None
    assert task_tool.name == "task"
    await session.dispose()


class TestMeasurePreloadedContext:
    """What a session is already carrying before the user has typed.

    The status line's only source used to be a provider's ``prompt_tokens``,
    which does not exist until a turn completes — so a session opened with a
    large tool inventory read as empty at the exact moment it was most loaded.
    """

    @pytest.mark.asyncio
    async def test_counts_system_blocks_and_tool_schemas(self, tmp_path) -> None:
        executed: list[str] = []
        session = make_session(
            tmp_path,
            ScriptedStream([]),
            tools=[echo_tool(executed)],
            system_blocks_provider=lambda: ["a system prompt", "an environment block"],
        )
        try:
            total = await session.measure_preloaded_context()

            blocks_only = make_session(
                tmp_path / "b",
                ScriptedStream([]),
                tools=[],
                system_blocks_provider=lambda: ["a system prompt", "an environment block"],
            )
            try:
                without_tools = await blocks_only.measure_preloaded_context()
            finally:
                await blocks_only.dispose()

            assert without_tools > 0, "system blocks alone must count for something"
            assert total > without_tools, "the tool schema is context too"
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_async_blocks_provider_is_awaited(self, tmp_path) -> None:
        """The real provider is a coroutine (it builds the skills index)."""

        async def blocks() -> list[str]:
            return ["resolved asynchronously"]

        session = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=blocks)
        try:
            assert await session.measure_preloaded_context() > 0
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_it_tracks_a_tool_inventory_that_grows(self, tmp_path) -> None:
        """MCP servers connect AFTER boot, and their schemas are the big term.

        A measurement taken once at boot would understate the context for the
        rest of the session, so the figure must follow ``refresh_tools``. The
        new set is the incumbent PLUS the arrivals, which is what the manager
        hands over — ``refresh_tools`` replaces rather than appends.
        """
        executed: list[str] = []
        session = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=lambda: ["sys"])
        try:
            before = await session.measure_preloaded_context()
            arrivals = [echo_tool(executed, name=f"mcp_{i}") for i in range(12)]
            session.refresh_tools([*session._tools, *arrivals])
            after = await session.measure_preloaded_context()
            assert after > before
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_an_empty_context_reports_zero(self, tmp_path) -> None:
        """Zero must stay reachable: the segment renders nothing for it, and a
        host with no system prompt and no tools genuinely has nothing loaded.

        The inventory is emptied through ``refresh_tools`` because construction
        merges this session's own capability tools in regardless of what the
        caller passed.
        """
        session = make_session(
            tmp_path, ScriptedStream([]), tools=[], system_blocks_provider=lambda: []
        )
        try:
            session.refresh_tools([])
            assert await session.measure_preloaded_context() == 0
        finally:
            await session.dispose()


class TestMeasurementCosts:
    """The two costs a pre-turn status readout must not incur.

    Both were live defects: the measurement loaded tiktoken (~43.6 MB RSS, and
    a NETWORK fetch of the BPE ranks on a cold cache) and ran its counting on
    the caller's event loop — on the very boot path a sibling change cleared of
    a 700 ms freeze.
    """

    @pytest.mark.asyncio
    async def test_it_never_loads_the_tokenizer(self, tmp_path, monkeypatch) -> None:
        from local_operator.compaction import tokens as tokens_mod

        def _boom(*_args, **_kwargs):
            raise AssertionError("the boot measurement must not load tiktoken")

        monkeypatch.setattr(tokens_mod, "_get_encoding", _boom)
        monkeypatch.setattr(tokens_mod, "_get_model_encoding", _boom)

        executed: list[str] = []
        session = make_session(
            tmp_path,
            ScriptedStream([]),
            tools=[echo_tool(executed)],
            system_blocks_provider=lambda: ["a system prompt " * 200],
        )
        try:
            assert await session.measure_preloaded_context() > 0
        finally:
            await session.dispose()

    @pytest.mark.asyncio
    async def test_everything_that_scales_leaves_the_event_loop(
        self, tmp_path, monkeypatch
    ) -> None:
        """Not "it is fast enough": the inventory is unbounded, so the work has
        to leave the loop rather than merely be small today.

        Both halves are checked, because an earlier version passed a
        counting-only version of this test while leaving ~97% of the CPU work
        on the pump: it serialized every tool schema with ``json.dumps`` to
        BUILD the list, then crossed to the thread only to add up
        ``len(text) // 4``. The term that grows with the inventory was the one
        that stayed, and the hop cost more than it carried.
        """
        import json as json_mod
        import threading

        from local_operator.compaction import tokens as tokens_mod
        from local_operator.session import session as session_mod

        loop_thread = threading.get_ident()
        counted: list[int] = []
        dumped: list[int] = []
        real_count = tokens_mod.approx_text_tokens
        real_dumps = json_mod.dumps

        def count_spy(text: str) -> int:
            counted.append(threading.get_ident())
            return real_count(text)

        def dumps_spy(*args, **kwargs):
            dumped.append(threading.get_ident())
            return real_dumps(*args, **kwargs)

        monkeypatch.setattr(session_mod, "approx_text_tokens", count_spy)
        monkeypatch.setattr(session_mod.json, "dumps", dumps_spy)

        executed: list[str] = []
        session = make_session(
            tmp_path,
            ScriptedStream([]),
            tools=[echo_tool(executed, name=f"t{i}") for i in range(6)],
            system_blocks_provider=lambda: ["sys"],
        )
        try:
            await session.measure_preloaded_context()
        finally:
            await session.dispose()

        assert counted, "nothing was counted"
        assert dumped, "no schema was serialized — the test tools lost their parameters"
        assert loop_thread not in counted, "counting ran on the caller's loop"
        assert loop_thread not in dumped, "schema serialization ran on the caller's loop"

    @pytest.mark.asyncio
    async def test_a_tool_swap_mid_measurement_cannot_race(self, tmp_path) -> None:
        """The inventory is snapshotted before the thread hop, so an MCP
        refresh landing mid-measurement cannot mutate the list being walked."""
        executed: list[str] = []
        session = make_session(
            tmp_path,
            ScriptedStream([]),
            tools=[echo_tool(executed, name=f"t{i}") for i in range(8)],
            system_blocks_provider=lambda: ["sys"],
        )
        try:
            task = asyncio.create_task(session.measure_preloaded_context())
            await asyncio.sleep(0)
            session.refresh_tools([])
            assert await task > 0
        finally:
            await session.dispose()


class PoisonedThenFine:
    """Raises the provider's image refusal on the first call, then behaves.

    Models the reported failure exactly: the block is in HISTORY, so the
    refusal is not tied to what the user just typed and recurs on every
    request until something stops sending it.
    """

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.calls = 0

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        self.calls += 1
        first = self.calls == 1

        async def gen():
            if first:
                raise ProviderError(400, "Could not process image")
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="endTurn")

        return gen()


def _image_blocks(request: ChatRequest) -> int:
    return sum(
        isinstance(block, ImageContent) for message in request.messages for block in message.content
    )


@pytest.mark.asyncio
async def test_an_image_the_provider_refuses_does_not_brick_the_session(tmp_path):
    """The reported bug: every turn after the refusal failed identically.

    An image block lives in the conversation history, so once the provider
    starts refusing it, the next request sends it again and gets the same 400 —
    and so does the one after that, and so does ``/compact``, which has to send
    the history in order to summarise it. The session could only be abandoned.

    Not preventable on our side: providers accept the same bytes for hours and
    then start refusing them (anthropics/claude-code#50708), so the client
    cannot validate its way out. It can only notice and stop.
    """
    stream = PoisonedThenFine()
    session = make_session(tmp_path, stream)
    await session.seed_history(
        [
            Message(
                role="user",
                content=[TextContent(text="look at this"), ImageContent(data="Zm9v")],
            )
        ]
    )

    await session.prompt("does it work")
    assert stream.calls == 1
    assert _image_blocks(stream.requests[0]) == 1, "the poisoned block was sent, as it must be"
    assert session._images_rejected, "the refusal was not recognised"

    # The whole point: the NEXT turn goes through.
    await session.prompt("try again")
    assert stream.calls == 2
    assert _image_blocks(stream.requests[1]) == 0, "the session is still sending the bad image"
    sent = [
        block.text
        for message in stream.requests[1].messages
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert "look at this" in sent, "the surrounding turn was dropped along with the image"
    assert IMAGE_DROPPED_NOTICE in sent, "the model was left with a silent hole"
    await session.dispose()


@pytest.mark.asyncio
async def test_an_ordinary_failure_leaves_images_alone(tmp_path):
    """The degrade is permanent and invisible, so it must not fire on weather.

    A 5xx or an unrelated 400 has nothing to do with the images, and stripping
    them would quietly cost the model every screenshot for the rest of the
    session.
    """

    class AlwaysDown:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)

            async def gen():
                raise ProviderError(503, "upstream connect error")
                yield  # pragma: no cover - generator shape only

            return gen()

    stream = AlwaysDown()
    session = make_session(tmp_path, stream)
    await session.seed_history(
        [Message(role="user", content=[TextContent(text="hi"), ImageContent(data="Zm9v")])]
    )
    await session.prompt("go")
    assert not session._images_rejected
    assert _image_blocks(stream.requests[0]) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_mid_turn_compaction_at_continuing_boundary(tmp_path, monkeypatch):
    """A tool-loop run that crosses the threshold mid-run compacts at the safe
    boundary — after the tool batch lands, before the next model call — inside
    the run's event boundary, and the next model call sees the compacted
    context. Post-run persistence must not resurrect what the pass summarized:
    the loop prunes its run accumulator to the replacement's survivors, so the
    transcript after the compaction entry holds only surviving run messages."""
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = -1.0
        threshold_tokens: int = Field(default=-1)
        max_threshold_tokens: int = 600_000
        auto_continue: bool = False
        mid_turn_enabled: bool = True

    compacted = {"done": False}

    def estimate_messages_tokens(messages):
        return 5_000 if compacted["done"] else 90_000

    def messages_tokens_upper_bound(messages):
        return 5_500 if compacted["done"] else 95_000

    def prune_tool_outputs(messages, now_ts, last_activity_ts, **kwargs):
        return list(messages), False

    fake_api = types.ModuleType("local_operator.compaction.api")
    setattr(fake_api, "CompactionSettings", CompactionSettings)
    setattr(fake_api, "prune_tool_outputs", prune_tool_outputs)
    setattr(fake_api, "estimate_messages_tokens", estimate_messages_tokens)
    setattr(fake_api, "messages_tokens_upper_bound", messages_tokens_upper_bound)
    setattr(
        fake_api, "compaction_context_tokens", lambda provider, local: max(provider or 0, local)
    )
    # Cut before the run's user prompt: kept = [user, assistant, tool result].
    # The user prompt is a transcript entry (appended pre-run), so the cut is
    # replayable; a cut into run-produced-only history is refused by design.
    setattr(fake_api, "find_cut_point", lambda messages, keep: 3)
    setattr(
        fake_api,
        "resolve_threshold_tokens",
        lambda window, settings: min(int(window * 0.8), 600_000),
    )
    setattr(fake_api, "RECOVERY_BAND", 0.8)
    setattr(
        fake_api,
        "should_compact",
        lambda ctx_tokens, window, settings: ctx_tokens > settings.threshold_tokens,
    )

    async def summarize(
        messages: Sequence[Message | CustomMessage],
        complete_fn: Callable[[str, str], Awaitable[str]],
    ) -> str:
        compacted["done"] = True
        summary = await complete_fn("sys", "summarize this")
        return f"SUMMARY({summary})"

    setattr(fake_api, "summarize_messages", summarize)

    fake_pkg = types.ModuleType("local_operator.compaction")
    setattr(fake_pkg, "api", fake_api)
    monkeypatch.setitem(sys.modules, "local_operator.compaction", fake_pkg)
    monkeypatch.setitem(sys.modules, "local_operator.compaction.api", fake_api)
    fake_thresholds = types.ModuleType("local_operator.compaction.thresholds")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.thresholds", fake_thresholds)
    fake_snap = types.ModuleType("local_operator.compaction.snapcompact")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.snapcompact", fake_snap)

    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="working"),
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            # The mid-run one-shot summary call.
            [StreamTextDelta(delta="compressed"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], compaction_settings=CompactionSettings()
    )
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    # Prime persisted + live history so the cut lands on a replayable entry:
    # three older user turns exist in both the transcript and the context.
    primed = [Message.user(f"earlier {i}") for i in range(3)]
    for message in primed:
        await session._transcript.append_message(message)
    session._context.messages.extend(primed)

    await session.prompt("go")

    assert executed == ["echo"]

    # Compaction events fired with the mid-turn reason, INSIDE the run
    # boundary (between agent_start and agent_end — the streaming contract's
    # SC-4 pairing rule).
    kinds = [(type(e).__name__, getattr(e, "reason", None)) for e in events]
    assert ("CompactionStartEvent", "mid-turn") in kinds
    start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentStartEvent))
    end_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentEndEvent))
    compact_idx = next(i for i, e in enumerate(events) if isinstance(e, CompactionStartEvent))
    assert start_idx < compact_idx < end_idx

    # Exactly one compaction ran: the post-turn pass found the compacted
    # context below threshold and refused.
    compactions = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert len(compactions) == 1
    assert compactions[0].payload["summary"] == "SUMMARY(compressed)"

    # Context rebuilt: marker + kept window (user, assistant, tool result)
    # plus the final assistant turn the loop went on to produce after the
    # compaction — the whole point of compacting mid-run instead of failing.
    assert len(session._context.messages) == 5
    marker = session._context.messages[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"

    # The next model call saw the compacted context: the rendered marker
    # (a user message carrying the summary) leads the request.
    third = stream.requests[2]
    assert "SUMMARY(compressed)" in third.messages[0].text

    # No resurrection: after the compaction entry the transcript holds only
    # the run's three surviving messages (assistant, tool result, final
    # assistant) — the primed history the pass summarized never re-lands
    # after it, and nothing is persisted twice.
    entries = session._transcript.entries()
    c_index = next(i for i, e in enumerate(entries) if e.type == "compaction")
    after = [e for e in entries[c_index + 1 :] if e.type == "message"]
    assert len(after) == 3

    await session.dispose()


@pytest.mark.asyncio
async def test_mid_turn_compaction_disabled_by_setting(tmp_path, monkeypatch):
    """``mid_turn_enabled=False`` restores the old posture exactly: the
    boundary hook is a no-op and only the post-turn pass compacts."""
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = -1.0
        threshold_tokens: int = Field(default=-1)
        max_threshold_tokens: int = 600_000
        auto_continue: bool = False
        mid_turn_enabled: bool = False

    def prune_tool_outputs(messages, now_ts, last_activity_ts, **kwargs):
        return list(messages), False

    fake_api = types.ModuleType("local_operator.compaction.api")
    setattr(fake_api, "CompactionSettings", CompactionSettings)
    setattr(fake_api, "prune_tool_outputs", prune_tool_outputs)
    setattr(fake_api, "estimate_messages_tokens", lambda messages: 90_000)
    setattr(fake_api, "messages_tokens_upper_bound", lambda messages: 95_000)
    setattr(
        fake_api, "compaction_context_tokens", lambda provider, local: max(provider or 0, local)
    )
    setattr(fake_api, "find_cut_point", lambda messages, keep: 0)
    setattr(
        fake_api,
        "resolve_threshold_tokens",
        lambda window, settings: min(int(window * 0.8), 600_000),
    )
    setattr(fake_api, "RECOVERY_BAND", 0.8)
    setattr(
        fake_api,
        "should_compact",
        lambda ctx_tokens, window, settings: ctx_tokens > settings.threshold_tokens,
    )

    async def summarize(
        messages: Sequence[Message | CustomMessage],
        complete_fn: Callable[[str, str], Awaitable[str]],
    ) -> str:
        return "SUMMARY(post-turn)"

    setattr(fake_api, "summarize_messages", summarize)

    fake_pkg = types.ModuleType("local_operator.compaction")
    setattr(fake_pkg, "api", fake_api)
    monkeypatch.setitem(sys.modules, "local_operator.compaction", fake_pkg)
    monkeypatch.setitem(sys.modules, "local_operator.compaction.api", fake_api)
    fake_thresholds = types.ModuleType("local_operator.compaction.thresholds")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.thresholds", fake_thresholds)
    fake_snap = types.ModuleType("local_operator.compaction.snapcompact")
    monkeypatch.setitem(sys.modules, "local_operator.compaction.snapcompact", fake_snap)

    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], compaction_settings=CompactionSettings()
    )
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("go")

    reasons = [
        getattr(e, "reason", None)
        for e in events
        if isinstance(e, (CompactionStartEvent, CompactionEndEvent))
    ]
    # No mid-turn pass; the single pass (if any) is the post-turn one.
    assert "mid-turn" not in reasons
    compactions = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert len(compactions) <= 1
    await session.dispose()


# ---------------------------------------------------------------------------
# session incidents (why a run died, model-visible)
# ---------------------------------------------------------------------------


class ExplodingStream:
    """First call raises a provider-shaped error; later calls succeed."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        if len(self.requests) == 1:

            async def boom():
                yield StreamTextDelta(delta="partial")
                raise RuntimeError("429 Too Many Requests: rate limit exceeded")

            return boom()

        async def ok():
            yield StreamTextDelta(delta="recovered")
            yield StreamEndEvent(stop_reason="stop")

        return ok()


@pytest.mark.asyncio
async def test_error_run_journals_model_visible_incident(tmp_path):
    """A run that dies on a provider error leaves a classified incident in
    the LIVE context (the next prompt sees it) and in the transcript (a
    resumed session replays it) — the model learns WHY, not just the UI."""
    stream = ExplodingStream()
    session = make_session(tmp_path, stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("go")

    incidents = [
        m
        for m in session._context.messages
        if isinstance(m, CustomMessage) and m.custom_type == "session_incident"
    ]
    assert incidents, "error run must journal a session incident"
    assert "rate-limit" in incidents[-1].details["text"]
    assert "429" in incidents[-1].details["raw"]

    # Persisted: the transcript replay carries it.
    dumped = "\n".join(
        __import__("json").dumps(e.payload, default=str) for e in session._transcript.entries()
    )
    assert "session_incident" in dumped

    # The NEXT prompt renders it into the request the provider sees.
    await session.prompt("continue")
    rendered = "\n".join(getattr(m, "text", "") for m in stream.requests[1].messages)
    assert "[session incident" in rendered and "rate-limit" in rendered
    await session.dispose()


@pytest.mark.asyncio
async def test_job_completion_auto_delivers_when_idle(tmp_path):
    """A settled model-owned job re-wakes the idle session: the result lands
    as a conversation turn without the model polling 'jobs'."""
    turn_count = {"n": 0}

    class TwoTurnStream:
        def __init__(self) -> None:
            self.requests: list[ChatRequest] = []

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.requests.append(request)
            turn_count["n"] += 1

            async def gen():
                yield StreamTextDelta(delta=f"turn {turn_count['n']}")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    stream = TwoTurnStream()
    session = make_session(tmp_path, stream)
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("start something")

    async def quick(job_id, signal, report_progress):
        return "the answer is 42"

    session.jobs.register("task", "researcher", quick)
    await wait_for(lambda: sum(1 for e in events if isinstance(e, AgentStartEvent)) >= 2)
    delivered = [
        m
        for m in session._context.messages
        if isinstance(m, CustomMessage) and m.custom_type == "job_result"
    ]
    assert delivered
    assert "the answer is 42" in delivered[-1].details["text"]
    await session.dispose()


@pytest.mark.asyncio
async def test_consumed_and_foreign_jobs_do_not_auto_deliver(tmp_path):
    """wait already returned the result (consumed) and host-registered job
    types stay quiet; only fresh model-owned work re-wakes the session."""
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    await session.prompt("go")
    before = len(session._context.messages)

    async def runner(job_id, signal, report_progress):
        return "done"

    consumed_id = session.jobs.register("task", "consumed-job", runner)
    job = session.jobs.get(consumed_id)
    assert job is not None
    job.consumed = True
    await session._on_job_completed(consumed_id, "done", job)

    # A streaming session stays quiet too: the in-turn model owns the floor.
    # The runner must still be RUNNING when the flag flips, or the manager's
    # own settle-delivery wins the race and delivers while idle.
    async def slow_runner(job_id, signal, report_progress):
        await asyncio.sleep(0.3)
        return "done"

    session.jobs.register("task", "while-busy", slow_runner)
    session._is_streaming = True
    try:
        await asyncio.sleep(0.45)  # settles while the session is "streaming"
    finally:
        session._is_streaming = False
    await asyncio.sleep(0.05)
    assert len(session._context.messages) == before  # nothing delivered
    await session.dispose()


def test_context_breakdown_counts_wire_schemas_and_messages(tmp_path):
    """The `/context` source measures what the provider actually receives:
    four system blocks, wire tool schemas (not just names), rendered messages,
    window and the last cache-read bucket."""
    tool = echo_tool([])
    session = make_session(tmp_path, ScriptedStream([]), tools=[tool])
    session._context.system_blocks = ["instructions", "inventory", "env", "skills"]
    session._context.messages = [Message.user("hello"), Message.assistant("world")]
    session._last_usage = Usage(input_tokens=10, cache_read_tokens=7, context_tokens=17)
    data = session.context_breakdown()
    assert data["instructions"] > 0
    assert data["tool_schemas"] > 0
    assert data["messages"] > 0
    assert data["context_window"] == MODEL.context_window
    assert data["cache_read"] == 7
    assert data["total"] == sum(
        data[key]
        for key in (
            "instructions",
            "tool_inventory",
            "environment",
            "knowledge_mcp_goal",
            "tool_schemas",
            "messages",
        )
    )
