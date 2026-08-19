"""Session facade tests: turn flow, events, steering, abort, wake wiring,
compaction hook, dispose."""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable, Callable, Sequence

import pytest

from local_operator.compaction.api import CompactionSettings
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
    ToolCall,
    ToolResult,
    Usage,
)
from local_operator.providers.failover import ProviderError
from local_operator.session.session import (
    IMAGE_DROPPED_NOTICE,
    SESSION_INCIDENT_MESSAGE_TYPE,
    Session,
    _paired_prefix,
)
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import TODO_REMINDER_MESSAGE_TYPE

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
        model=kwargs.pop("model", MODEL),
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
    The gate holds NO threshold arithmetic of its own: it hands the settings to
    ``should_compact``, which resolves min(percent * window, absolute)."""
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = 0.80
        threshold_tokens: int = Field(default=600_000)
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
        lambda window, settings: min(
            int(window * settings.threshold_percent), settings.threshold_tokens
        ),
    )
    setattr(fake_api, "RECOVERY_BAND", 0.8)

    seen_threshold: list[int] = []

    def should_compact(ctx_tokens, window, settings):
        threshold = fake_api.resolve_threshold_tokens(window, settings)
        seen_threshold.append(threshold)
        return ctx_tokens > threshold

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

    # The threshold the gate acted on came from the resolver, not from the
    # session: min(100_000 * 0.80, 600_000). Twice, because the trigger is
    # two-stage — the cheap upper bound is tested first and only a bound that
    # clears the threshold buys the exact estimate.
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
        threshold_percent: float = 0.80
        threshold_tokens: int = Field(default=600_000)
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
        lambda window, settings: min(
            int(window * settings.threshold_percent), settings.threshold_tokens
        ),
    )
    setattr(
        fake_api,
        "should_compact",
        lambda ctx, window, settings: ctx > fake_api.resolve_threshold_tokens(window, settings),
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

    #: Overridable so a subclass can pin a DIFFERENT provider wording against
    #: the same end-to-end path. The predicate's own tests cover the strings;
    #: what this class covers is the session actually recovering, which is the
    #: part that was broken.
    refusal = "Could not process image"

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []
        self.calls = 0

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        self.calls += 1
        first = self.calls == 1
        refusal = self.refusal

        async def gen():
            if first:
                raise ProviderError(400, refusal)
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="endTurn")

        return gen()


class RefusedForTooManyImages(PoisonedThenFine):
    """The many-image dimension refusal, which is the one seen in the wild.

    Distinct from a malformed block in how it ARRIVES: nothing about the image
    changed and no request was malformed. The conversation simply grew past
    twenty images, at which point the provider applies a stricter 2000-pixel
    per-image limit and refuses a frame it had accepted for a hundred turns.
    """

    refusal = (
        "messages.0.content.2.image.source.base64.data: At least one of the image "
        "dimensions exceed max allowed size for many-image requests: 2000 pixels"
    )


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
async def test_the_many_image_dimension_refusal_also_unbricks_the_session(tmp_path):
    """The refusal that wedged a real session on 2026-08-18.

    The degrade recognised several provider wordings and not this one, so the
    session answered every prompt — and every ``/compact``, which has to send
    the history in order to summarise it — with the same 400 until it was
    abandoned. The composer no longer attaches an image that can trip the
    2000-pixel many-image limit, but this backstop still has to hold: history
    written by an OLDER build already contains such blocks, and a resumed
    session replays them.
    """
    stream = RefusedForTooManyImages()
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
    assert session._images_rejected, "the many-image refusal was not recognised"

    await session.prompt("try again")
    assert stream.calls == 2
    assert _image_blocks(stream.requests[1]) == 0, "the session is still sending the bad image"
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
        threshold_percent: float = 0.80
        threshold_tokens: int = Field(default=600_000)
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
        lambda ctx_tokens, window, settings: ctx_tokens
        > fake_api.resolve_threshold_tokens(window, settings),
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

    # No resurrection and no duplication. The mid-turn gate flushes the run's
    # messages BEFORE it cuts (the cut target has to be a persisted entry), so
    # surviving run messages legitimately sit on either side of the compaction
    # entry and counting the tail is no longer the question. What must hold is
    # the property that count stood in for: every message is stored exactly
    # once, and replaying the transcript reproduces the live context — the
    # summarized history does not come back, and the kept window is not lost.
    entries = session._transcript.entries()
    message_ids = [e.id for e in entries if e.type == "message"]
    assert len(message_ids) == len(set(message_ids)), "a message was persisted twice"

    replayed = Transcript(session._transcript.directory).build_llm_history()
    assert [
        (getattr(m, "role", None) or getattr(m, "custom_type", None), getattr(m, "text", ""))
        for m in replayed
    ] == [
        (getattr(m, "role", None) or getattr(m, "custom_type", None), getattr(m, "text", ""))
        for m in session._context.messages
    ]

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
        threshold_percent: float = 0.80
        threshold_tokens: int = Field(default=600_000)
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
        lambda ctx_tokens, window, settings: ctx_tokens
        > fake_api.resolve_threshold_tokens(window, settings),
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


def _config_dir_with(tmp_path, monkeypatch, subagents: dict[str, object] | None):
    """Point the process at a throwaway config dir, optionally with a tier map.

    ``_errand_model`` reads ``subagents.models.lo`` from the real user config,
    so a test that did not redirect this would answer differently on a machine
    that happens to have a cheap tier configured.
    """
    from local_operator.config import ConfigManager

    config = tmp_path / f"config-{'tiered' if subagents else 'bare'}"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    manager = ConfigManager(config)
    manager.set_config_value("subagents", subagents or {})
    return config


@pytest.mark.asyncio
async def test_the_naming_errand_leaves_the_session_isolated_and_cheap(tmp_path, monkeypatch):
    """``complete_once`` is auto-naming's only route to a provider, and it now
    runs CONCURRENTLY with the user's turn, so everything that makes it safe to
    do so lives on the REQUEST it builds.

    The transport suites prove those flags are honoured, but they construct the
    flag themselves, and every host suite that reaches naming replaces
    ``complete_once`` on a fake — so nothing watched the one line that SETS it.
    This asserts the request exactly as it leaves the session.
    """
    _config_dir_with(tmp_path, monkeypatch, None)  # no `lo` tier configured
    captured: list[tuple[ChatRequest, AbortSignal | None]] = []

    def stream_fn(request: ChatRequest, signal: AbortSignal | None):
        captured.append((request, signal))

        async def gen():
            yield StreamTextDelta(delta="<title>the login redirect loop</title>")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    session = make_session(tmp_path, stream_fn)
    text = await session.complete_once("name this conversation", "fix the login redirect loop")
    await session.dispose()

    assert text == "<title>the login redirect loop</title>"
    assert len(captured) == 1, "the errand is one call, not a loop"
    request, signal = captured[0]
    assert request.isolated is True, "the naming call reached the wire un-isolated"
    assert request.replayable is False, "a title is worth one attempt, not a replay"
    assert request.max_tokens == Session.ERRAND_MAX_TOKENS
    assert request.model == session._errand_model()
    assert request.model.model_id == MODEL.model_id, "no `lo` tier is configured here"
    assert request.tools == [] and request.tool_choice == "none"
    assert signal is None, "the errand carries no abort signal"
    # It is not a turn: nothing about it may reach the transcript.
    assert [entry.type for entry in session._transcript.entries()] == []


@pytest.mark.asyncio
async def test_the_errand_model_is_effort_clamped_on_both_routes(tmp_path, monkeypatch):
    """``ERRAND_MAX_TOKENS`` is an output cap that COUNTS REASONING TOKENS, so a
    128-token errand left on a reasoning model's default effort can spend the
    whole budget thinking, emit no ``<title>`` at all and make ``parse_title``
    return ``None`` — auto-naming would silently never work for that operator
    while still billing the thinking.

    Both routes out of ``_errand_model`` therefore clamp: the configured ``lo``
    tier (whose ``build_model_spec`` effort is ``None`` for OpenAI reasoning
    models, i.e. no ``reasoning.effort`` on the wire and the provider's own
    default applied) and the fallback to the session's own model.
    """
    from local_operator.model.configure import build_model_spec

    _config_dir_with(tmp_path, monkeypatch, {"models": {"lo": "openai/gpt-5-mini"}})
    reasoning_model = build_model_spec("anthropic", "claude-opus-5")
    session = make_session(tmp_path, ScriptedStream([]), model=reasoning_model)

    tier = build_model_spec("openai", "gpt-5-mini")
    assert tier.reasoning_efforts, "this test needs a tier WITH an effort ladder"
    assert tier.reasoning_effort is None, "unclamped, this spec sends no effort at all"

    errand = session._errand_model()
    assert (errand.provider, errand.model_id) == ("openai", "gpt-5-mini"), "the tier is preferred"
    assert errand.reasoning_effort == tier.reasoning_efforts[0]

    # Same clamp on the other route: no tier, so the session's own model.
    _config_dir_with(tmp_path, monkeypatch, None)
    fallback = session._errand_model()
    assert fallback.model_id == reasoning_model.model_id
    assert reasoning_model.reasoning_effort != reasoning_model.reasoning_efforts[0]
    assert fallback.reasoning_effort == reasoning_model.reasoning_efforts[0]


@pytest.mark.asyncio
async def test_the_mid_turn_flush_never_persists_a_compaction_marker(tmp_path):
    """F1: the mid-turn flush must not write the compaction summary marker.

    The mid-turn gate persists the run's messages before it cuts, because the
    cut target has to be a replayable transcript entry. It flushes from the
    LIVE loop context — and after a pass that context is ``[marker, *kept]``.
    The marker is not a todo reminder and its id is in no transcript, so a
    flush excluding only reminders wrote it as a MESSAGE entry, while the pass
    had already stored it as its own ``compaction`` entry. Replay then
    reconstructs a superseded summary beside the live one.

    Driven through the real tool loop rather than ``compact_now``, because the
    flush lives in ``_on_turn_end`` and a manual pass never reaches it — an
    earlier version of this test used ``compact_now`` and passed with the fix
    reverted, pinning nothing.

    Asserted on transcript ENTRIES, not on a duplicate-id count: every marker
    carries a fresh uuid, so a resurrected marker is a new entry and never a
    duplicate. A duplicate check is structurally blind to this.
    """
    big = "lorem ipsum dolor sit amet consectetur " * 1200

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text=big)]
        )

    tool = AgentTool(name="echo", parameters={"type": "object", "properties": {}}, execute=execute)

    class ToolRunStream:
        """Many tool batches, each reporting the prompt size it was really sent."""

        def __init__(self, batches: int) -> None:
            self.batches = batches
            self.calls = 0

        def __call__(self, request, signal):
            index = self.calls
            self.calls += 1
            from local_operator.compaction.api import estimate_messages_tokens

            size = estimate_messages_tokens(list(request.messages))
            usage = Usage(input_tokens=size, context_tokens=size)

            async def gen():
                if index < self.batches:
                    yield StreamTextDelta(delta=f"step {index} ")
                    yield StreamToolCallDelta(
                        index=0, id=f"c{index}", name="echo", argument_delta="{}"
                    )
                    yield StreamEndEvent(stop_reason="toolUse", usage=usage)
                else:
                    yield StreamTextDelta(delta="done")
                    yield StreamEndEvent(stop_reason="stop", usage=usage)

            return gen()

    stream = ToolRunStream(batches=40)
    session = make_session(
        tmp_path,
        stream,
        tools=[tool],
        compaction_settings=CompactionSettings(keep_recent_tokens=20_000),
    )

    # Long enough to force several mid-run passes, so a marker is in the live
    # context at a later boundary — which is the state that reproduces this.
    await session.prompt("do the long thing " + "detail " * 200)

    entries = session._transcript.entries()
    assert [e for e in entries if e.type == "compaction"], "no pass ran; the test proves nothing"

    markers_as_messages = [
        entry
        for entry in entries
        if entry.type == "message" and entry.payload.get("custom_type") == "compaction_summary"
    ]
    assert markers_as_messages == [], (
        f"{len(markers_as_messages)} compaction marker(s) were persisted as message "
        "entries; on replay they resurrect superseded summaries beside the live one"
    )

    replayed = Transcript(session._transcript.directory).build_llm_history()
    markers = [
        m
        for m in replayed
        if isinstance(m, CustomMessage) and m.custom_type == "compaction_summary"
    ]
    assert len(markers) == 1, "replay must carry exactly one compaction summary"

    await session.dispose()


@pytest.mark.asyncio
async def test_a_cancelled_subagent_never_persists_a_marker_or_reminder(tmp_path):
    """F4: the cancelled-child writer uses the same allow-list as the session.

    ``_persist_inflight`` writes a cancelled subagent's LIVE context straight
    to its transcript so the turn is not lost. That context has the same two
    ephemeral inhabitants as the parent's: after a compaction pass it begins
    with the summary marker, and it may carry a todo reminder. Persisting
    either corrupts the child's history — the marker replays a superseded
    summary beside the live one, and a stored reminder comes back as a user
    message nobody sent.

    This path predates the mid-turn flush (it is byte-identical on the base
    commit), but mid-turn compaction landing for real is what puts a marker in
    a child's live context in the first place, so the exposure is new.
    """
    from local_operator.harness.subagent import _persist_inflight

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    child = make_session(tmp_path, stream)

    # The live context a cancelled child can plausibly hold: a real user turn,
    # a compaction marker from a pass that already ran, and a live reminder.
    marker = CustomMessage(
        custom_type="compaction_summary",
        attribution="system",
        details={"summary": "an earlier stretch of this session"},
    )
    reminder = CustomMessage(
        custom_type=TODO_REMINDER_MESSAGE_TYPE,
        attribution="system",
        details={"text": "<system-reminder>still open</system-reminder>"},
    )
    real = Message.user("the work the child was doing")
    child._context.messages = [marker, real, reminder]

    await _persist_inflight(child)

    entries = child._transcript.entries()
    kinds = [
        entry.payload.get("custom_type")
        for entry in entries
        if entry.type == "message" and entry.payload.get("kind") == "custom"
    ]
    assert "compaction_summary" not in kinds, (
        "the cancelled child persisted a compaction marker; on resume it replays "
        "a superseded summary beside the live one"
    )
    assert TODO_REMINDER_MESSAGE_TYPE not in kinds, (
        "the cancelled child persisted a todo reminder, which replays as a user "
        "message the user never sent"
    )

    # The real work still lands — the allow-list must not cost the turn.
    assert any(
        entry.type == "message" and entry.payload.get("role") == "user" for entry in entries
    ), "the cancelled child's actual turn was dropped"

    await child.dispose()


# --- durability: a run's progress survives a crash or interrupt -------------
#
# Regression cover for the loss reported from a long KOHO debugging session:
# a transcript held ONLY the user's prompt while the run was in flight, so a
# session that died mid-run replayed to nothing and every completed tool call
# was gone. The transcript store always flushed per append; what was missing
# was any append between the turn's first message and its post-run pass.


def _persisted_texts(session: Session) -> list[str]:
    """Message texts on DISK, read back through a fresh store (never the
    in-memory entry list, which would pass even if nothing was flushed)."""
    reopened = Transcript(session._transcript.directory)
    return [m.text for m in reopened.build_llm_history() if isinstance(m, Message) and m.text]


@pytest.mark.asyncio
async def test_tool_progress_is_on_disk_before_the_run_finishes(tmp_path):
    """Work completed at a tool boundary is durable IMMEDIATELY, not at the
    end of the run: a reader opening the file mid-run sees the assistant
    message and its tool result."""
    seen: list[list[str]] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="probe", content=[TextContent(text="TOOL-OUTPUT")]
        )

    probe = AgentTool(
        name="probe", parameters={"type": "object", "properties": {}}, execute=execute
    )

    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="FIRST-STEP"),
                StreamToolCallDelta(index=0, id="c1", name="probe", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="DONE"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[probe])
    # Sample the file at the boundary between the two model calls.
    original = session._on_turn_end

    async def sampling_hook(messages):
        outcome = await original(messages)
        seen.append(_persisted_texts(session))
        return outcome

    session._on_turn_end = sampling_hook  # type: ignore[assignment]

    await session.prompt("go")

    assert seen, "the boundary hook never fired"
    mid_run = seen[0]
    assert "FIRST-STEP" in mid_run  # assistant message durable mid-run
    assert "TOOL-OUTPUT" in mid_run  # and so is the completed tool result
    await session.dispose()


@pytest.mark.asyncio
async def test_completed_work_survives_a_crash_mid_run(tmp_path):
    """The failure the report described: the run dies AFTER real work, before
    any normal persistence. Everything completed must still replay."""

    async def execute(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="probe", content=[TextContent(text="EXPENSIVE")]
        )

    probe = AgentTool(
        name="probe", parameters={"type": "object", "properties": {}}, execute=execute
    )

    class CrashOnSecondCall(ScriptedStream):
        def __call__(self, request, signal):
            self.requests.append(request)
            if len(self.requests) == 1:

                async def first():
                    yield StreamTextDelta(delta="REASONING")
                    yield StreamToolCallDelta(index=0, id="c1", name="probe", argument_delta="{}")
                    yield StreamEndEvent(stop_reason="toolUse")

                return first()

            async def boom():
                raise RuntimeError("provider exploded")
                yield  # pragma: no cover - generator shape only

            return boom()

    session = make_session(tmp_path, CrashOnSecondCall([]), tools=[probe])
    # A HARD failure: everything downstream of the run is dead. That is what a
    # SIGKILL, an OOM, or a bug between boundaries looks like from the
    # transcript's point of view, and it is the case the loop's own error
    # handling cannot cover, because the post-run pass never executes at all.
    # (A mere provider/stream error does NOT reproduce the loss: the loop
    # catches it, the run ends normally and the post-run pass still writes
    # everything. Measured on a build with the boundary flush removed: this
    # sabotage leaves ``['go']`` on disk, a stream error leaves the full run.)
    #
    # Sabotage is scoped to the POST-RUN call site rather than the method,
    # because the durability flushes share that method and disabling it
    # outright would remove the very thing under test.
    real_persist = session._persist_new_messages
    inside_durability_flush = {"now": False}
    real_progress = session._persist_progress

    async def tracking_progress(messages):
        inside_durability_flush["now"] = True
        try:
            return await real_progress(messages)
        finally:
            inside_durability_flush["now"] = False

    async def persist_or_die(messages):
        if inside_durability_flush["now"]:
            return await real_persist(messages)
        raise RuntimeError("post-run persistence never ran")

    session._persist_progress = tracking_progress  # type: ignore[assignment]
    session._persist_new_messages = persist_or_die  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        await session.prompt("go")

    replayed = _persisted_texts(session)
    assert "go" in replayed  # the prompt
    assert "REASONING" in replayed  # the assistant turn that ran
    assert "EXPENSIVE" in replayed  # the tool output that cost real time
    await session.dispose()


@pytest.mark.asyncio
async def test_completed_work_survives_cancellation(tmp_path):
    """Ctrl+C / dispose cancels the turn task. Cancellation must still
    propagate (the turn really is over) AND leave the COMPLETED work on disk.

    Completed is the operative word. An earlier version of this test cancelled
    during the FIRST batch and asserted that batch's assistant message was
    persisted — which is precisely the dangling ``tool_use`` that
    :func:`_paired_prefix` now refuses to write (see
    ``test_durability_flush_never_persists_a_dangling_tool_call``). The
    assertion was wrong, not the trim: an assistant message whose calls never
    reported is not work to preserve, it is a row that 400s every later
    resume. So the cancel here lands in the SECOND batch, and what must
    survive is the first batch, which really did finish.
    """
    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        if tool_call_id == "c2":
            started.set()
            await asyncio.sleep(30)  # cancelled here, one batch in
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="slow", content=[TextContent(text="FINISHED")]
        )

    slow = AgentTool(name="slow", parameters={"type": "object", "properties": {}}, execute=execute)

    class Steps(ScriptedStream):
        def __call__(self, request, signal):
            self.requests.append(request)
            index = len(self.requests)

            async def gen():
                yield StreamTextDelta(delta=f"BEFORE-CANCEL-{index}")
                yield StreamToolCallDelta(index=0, id=f"c{index}", name="slow", argument_delta="{}")
                yield StreamEndEvent(stop_reason="toolUse")

            return gen()

    session = make_session(tmp_path, Steps([]), tools=[slow])

    task = asyncio.create_task(session.prompt("go"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replayed = _persisted_texts(session)
    assert "go" in replayed
    assert "BEFORE-CANCEL-1" in replayed  # the batch that completed
    assert "FINISHED" in replayed  # and its tool result

    # Pairing is asserted HERE, on the cancellation path itself, and not only
    # in the dedicated dangling-call test. An earlier version of this test
    # checked text alone, and text alone is satisfied by a transcript carrying
    # an unanswered `tool_use` — so it ratified the corruption instead of
    # catching it (review round 1, R2).
    history = Transcript(session._transcript.directory).build_llm_history()
    calls = {
        call.id
        for message in history
        if isinstance(message, Message) and message.role == "assistant"
        for call in message.tool_calls
    }
    results = {
        message.tool_call_id
        for message in history
        if isinstance(message, Message) and message.role == "tool"
    }
    assert not calls - results, f"cancellation left a dangling tool_use: {sorted(calls - results)}"
    await session.dispose()


@pytest.mark.asyncio
async def test_durability_flush_never_duplicates_messages(tmp_path):
    """The boundary flush, the finally flush and the post-run pass all offer
    the same messages. Idempotence by id means each is stored exactly once."""
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                StreamTextDelta(delta="ONE"),
                StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta='{"text":"x"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="TWO"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[echo_tool(executed)])
    await session.prompt("go")

    texts = _persisted_texts(session)
    for expected in ("go", "ONE", "TWO"):
        assert texts.count(expected) == 1, f"{expected!r} stored {texts.count(expected)}x: {texts}"
    await session.dispose()


@pytest.mark.asyncio
async def test_progress_survives_when_mid_turn_compaction_is_off(tmp_path):
    """The reported loss, end to end.

    Before the durability flush, the ONLY thing that persisted mid-run was a
    side effect of the mid-turn compaction gate: it writes the run so far to
    get a replayable cut target. Every early return above that line therefore
    left the whole run unpersisted — and with ``mid_turn_enabled`` off the
    hook returns immediately, so a long tool run kept 100% of its work in
    memory. Measured against a build without the flush: this scenario left
    exactly ONE entry (the user's prompt) on disk against ten with it, which
    is the transcript the report described.
    """
    from pydantic import BaseModel, Field

    class CompactionSettings(BaseModel):
        enabled: bool = True
        reserve_tokens: int = 16384
        keep_recent_tokens: int = 20000
        threshold_percent: float = 0.80
        threshold_tokens: int = Field(default=600_000)
        auto_continue: bool = False
        mid_turn_enabled: bool = False  # the gate that used to carry persistence

    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        started.set()
        if tool_call_id == "c2":
            await asyncio.sleep(30)  # interrupted here, after real work landed
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="work", content=[TextContent(text="REAL-WORK")]
        )

    work = AgentTool(name="work", parameters={"type": "object", "properties": {}}, execute=execute)

    class Steps(ScriptedStream):
        def __call__(self, request, signal):
            self.requests.append(request)
            index = len(self.requests)

            async def gen():
                yield StreamTextDelta(delta=f"STEP-{index}")
                yield StreamToolCallDelta(index=0, id=f"c{index}", name="work", argument_delta="{}")
                yield StreamEndEvent(stop_reason="toolUse")

            return gen()

    session = make_session(
        tmp_path, Steps([]), tools=[work], compaction_settings=CompactionSettings()
    )
    task = asyncio.create_task(session.prompt("debug the tenant"))
    await started.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replayed = _persisted_texts(session)
    assert "STEP-1" in replayed, f"the first step was lost: {replayed}"
    assert "REAL-WORK" in replayed, f"completed tool work was lost: {replayed}"
    await session.dispose()


@pytest.mark.asyncio
async def test_durability_flush_never_persists_a_dangling_tool_call(tmp_path):
    """A crash or Ctrl+C MID-BATCH must not write an unanswered tool call.

    The loop appends the assistant message when the model turn ends and the
    tool results only when the batch finishes, so for the whole duration of a
    tool batch the live context ends in an assistant message whose calls have
    no answers. The ``finally`` flush persists that context — and a dangling
    ``tool_use`` on disk is permanent: every later resume replays it into a
    400 ("must be followed by tool messages responding to each
    tool_call_id"). Caught on a real Ctrl+C, which left ``['c1a', 'c1b']``
    unpaired before ``_paired_prefix`` trimmed the tail.

    Completed batches must still survive, or the trim would have thrown away
    the work the flush exists to save.
    """
    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        if tool_call_id.startswith("c2"):
            started.set()
            await asyncio.sleep(30)  # cancelled inside the SECOND batch
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="work",
            content=[TextContent(text=f"DONE-{tool_call_id}")],
        )

    work = AgentTool(name="work", parameters={"type": "object", "properties": {}}, execute=execute)

    class TwoCallBatches(ScriptedStream):
        def __call__(self, request, signal):
            self.requests.append(request)
            index = len(self.requests)

            async def gen():
                yield StreamTextDelta(delta=f"STEP-{index}")
                yield StreamToolCallDelta(
                    index=0, id=f"c{index}a", name="work", argument_delta="{}"
                )
                yield StreamToolCallDelta(
                    index=1, id=f"c{index}b", name="work", argument_delta="{}"
                )
                yield StreamEndEvent(stop_reason="toolUse")

            return gen()

    session = make_session(tmp_path, TwoCallBatches([]), tools=[work])
    task = asyncio.create_task(session.prompt("go"))
    await started.wait()
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replayed = Transcript(session._transcript.directory).build_llm_history()
    calls = {
        call.id
        for message in replayed
        if isinstance(message, Message) and message.role == "assistant"
        for call in message.tool_calls
    }
    results = {
        message.tool_call_id
        for message in replayed
        if isinstance(message, Message) and message.role == "tool"
    }
    assert calls, "the first batch should have been persisted"
    assert not calls - results, f"dangling tool_use persisted: {sorted(calls - results)}"

    texts = [m.text for m in replayed if isinstance(m, Message) and m.text]
    assert "DONE-c1a" in texts and "DONE-c1b" in texts, f"completed work lost: {texts}"
    await session.dispose()


@pytest.mark.asyncio
async def test_dispose_does_not_suppress_the_turns_own_flush(tmp_path):
    """``dispose()`` sets ``_disposed`` BEFORE aborting and awaiting the
    in-flight turn — deliberately, so that turn's persistence "must land on a
    live transcript" (HC-14). A ``_disposed`` guard inside the durability
    flush would therefore suppress exactly the flush dispose is waiting for
    (review round 1, R3). Completed work must be on disk after a dispose that
    lands mid-batch, and the tail must still be legal.
    """
    started = asyncio.Event()

    async def execute(tool_call_id, args, signal, on_update, context):
        if tool_call_id == "c2":
            started.set()
            await asyncio.sleep(30)  # dispose lands here
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="work",
            content=[TextContent(text=f"DONE-{tool_call_id}")],
        )

    work = AgentTool(name="work", parameters={"type": "object", "properties": {}}, execute=execute)

    class Steps(ScriptedStream):
        def __call__(self, request, signal):
            self.requests.append(request)
            index = len(self.requests)

            async def gen():
                yield StreamTextDelta(delta=f"STEP-{index}")
                yield StreamToolCallDelta(index=0, id=f"c{index}", name="work", argument_delta="{}")
                yield StreamEndEvent(stop_reason="toolUse")

            return gen()

    session = make_session(tmp_path, Steps([]), tools=[work])
    asyncio.create_task(session.prompt("go"))
    await started.wait()
    await asyncio.sleep(0.05)
    await session.dispose()

    history = Transcript(session._transcript.directory).build_llm_history()
    texts = [m.text for m in history if isinstance(m, Message) and m.text]
    assert "STEP-1" in texts and "DONE-c1" in texts, f"dispose dropped completed work: {texts}"

    calls = {
        call.id
        for message in history
        if isinstance(message, Message) and message.role == "assistant"
        for call in message.tool_calls
    }
    results = {
        message.tool_call_id
        for message in history
        if isinstance(message, Message) and message.role == "tool"
    }
    assert not calls - results, f"dispose left a dangling tool_use: {sorted(calls - results)}"


def test_paired_prefix_is_not_defeated_by_a_custom_message_in_the_tail():
    """A persistable ``CustomMessage`` must not shield an unanswered assistant.

    ``journal_incident`` appends straight to the live context and
    ``_on_mcp_incident`` fires it from a background task, so an MCP breaker
    tripping mid-batch leaves ``[..., assistant(tool_calls), session_incident]``.
    A tail scan that stopped at the first non-assistant entry declared that
    legal and persisted the dangling ``tool_use`` beneath it — R1's corruption
    through a narrower door (review round 2, R5).

    The custom itself is KEPT: it is real history, and dropping it would lose
    the incident the model needs to see on resume.
    """
    answered = Message(role="assistant", content=[TextContent(text="A1")])
    answered.tool_calls = [ToolCall(id="c1", name="work", arguments={})]
    result = Message(
        role="tool", tool_call_id="c1", tool_name="work", content=[TextContent(text="R1")]
    )
    unanswered = Message(role="assistant", content=[TextContent(text="A2")])
    unanswered.tool_calls = [ToolCall(id="c2", name="work", arguments={})]
    incident = CustomMessage(
        custom_type=SESSION_INCIDENT_MESSAGE_TYPE,
        attribution="system",
        details={"text": "MCP server 'linear': breaker tripped"},
    )

    prompt = Message.user("go")  # bound once: Message.user() mints a fresh id

    kept = _paired_prefix([prompt, answered, result, unanswered, incident])

    assert unanswered not in kept, "a custom in the tail shielded an unanswered assistant"
    assert incident in kept, "the incident must survive — it is real history"
    assert kept == [prompt, answered, result, incident]

    # And the ordinary cases still behave: an ANSWERED tail is untouched, and a
    # run of unanswered assistants under several customs is fully trimmed.
    answered_tail = [answered, result]
    assert _paired_prefix(answered_tail) == answered_tail
    assert _paired_prefix([answered, result, unanswered, incident, incident]) == [
        answered,
        result,
        incident,
        incident,
    ]


class TestASwitchedModelTakesEffectAtTheNextCall:
    """``set_model`` reaches the RUNNING turn's next provider call.

    The session's spec used to be snapshotted into ``LoopConfig`` once per
    turn, so a ``/model`` switch made while the agent was working could not
    reach any of that turn's remaining calls: the status band repainted, and
    every remaining call still went to the old model. A user only reaches for
    ``/model`` mid-turn because the running model is doing badly, so "it
    applies on your next message" was the wrong boundary.
    """

    OTHER = ModelSpec(provider="test", model_id="other", context_window=100_000)

    @staticmethod
    def _labels(stream: ScriptedStream) -> list[str]:
        return [f"{r.model.provider}/{r.model.model_id}" for r in stream.requests]

    @staticmethod
    def _tool_then_text() -> ScriptedStream:
        return ScriptedStream(
            [
                [
                    StreamToolCallDelta(index=0, id="c1", name="echo", argument_delta="{}"),
                    StreamEndEvent(stop_reason="toolUse"),
                ],
                [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            ]
        )

    @pytest.mark.asyncio
    async def test_a_switch_during_a_tool_reaches_the_next_call(self, tmp_path):
        holder: dict[str, Session] = {}
        stream = self._tool_then_text()

        async def execute(tool_call_id, args, signal, on_update, context):
            holder["session"].set_model(self.OTHER)
            return ToolResult(
                tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
            )

        tool = AgentTool(name="echo", parameters={"type": "object"}, execute=execute)
        session = make_session(tmp_path, stream, tools=[tool])
        holder["session"] = session
        try:
            await session.prompt("go")
        finally:
            await session.dispose()

        assert self._labels(stream) == ["test/m", "test/other"]

    @pytest.mark.asyncio
    async def test_a_switch_mid_stream_never_re_targets_the_call_in_flight(self, tmp_path):
        """The boundary is BETWEEN calls: one response is never split in two.

        Without this, "read the model later" would be free to mean "read it
        while a stream is already producing tokens", which would attribute a
        half-finished response to a model that never produced it.
        """
        holder: dict[str, Session] = {}
        seen: list[str] = []

        class SwitchingStream:
            def __call__(self, request, signal):
                seen.append(f"{request.model.provider}/{request.model.model_id}")

                async def gen():
                    yield StreamTextDelta(delta="a")
                    holder["session"].set_model(TestASwitchedModelTakesEffectAtTheNextCall.OTHER)
                    yield StreamTextDelta(delta="b")
                    yield StreamEndEvent(stop_reason="stop")

                return gen()

        session = make_session(tmp_path, SwitchingStream())
        holder["session"] = session
        try:
            await session.prompt("go")
            assert seen == ["test/m"]
            # ...and the switch is not lost, it simply lands on the next call.
            await session.prompt("again")
        finally:
            await session.dispose()

        assert seen == ["test/m", "test/other"]

    @pytest.mark.asyncio
    async def test_a_real_switch_invalidates_state_frozen_for_the_old_model(self, tmp_path):
        """Auto-effort and the quota memo are frozen per message, per MODEL."""
        stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
        changed: list[ModelSpec] = []
        stream.on_model_changed = changed.append  # type: ignore[attr-defined]
        session = make_session(tmp_path, stream)
        try:
            session.set_model(self.OTHER)
        finally:
            await session.dispose()

        assert [f"{m.provider}/{m.model_id}" for m in changed] == ["test/other"]

    @pytest.mark.asyncio
    async def test_changing_only_the_effort_does_not_invalidate_anything(self, tmp_path):
        """``/effort`` and the server's sampling overrides write the spec constantly.

        Treating those as a model change would re-classify effort on every one
        of them — and re-classification is exactly what freezing exists to
        prevent within a single user message.
        """
        stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
        changed: list[ModelSpec] = []
        stream.on_model_changed = changed.append  # type: ignore[attr-defined]
        session = make_session(tmp_path, stream)
        try:
            session.set_model(MODEL.model_copy(update={"reasoning_effort": "high"}))
            session.set_model(MODEL.model_copy(update={"temperature": 0.9}))
        finally:
            await session.dispose()

        assert changed == []

    @pytest.mark.asyncio
    async def test_a_stream_fn_without_the_hook_is_fine(self, tmp_path):
        """Most stream fns are plain callables; the hook is optional."""
        stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
        session = make_session(tmp_path, stream)
        try:
            session.set_model(self.OTHER)
            assert session.model_label == "test/other"
        finally:
            await session.dispose()
