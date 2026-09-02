"""``Session.complete_aside`` / ``adopt_aside`` — the no-trace contract itself.

The TUI tests in ``tests/unit/tui/test_aside.py`` pin the SURFACE against a
fake session. These pin the enforcement: that the real ``Session`` reads the
live conversation, writes nothing back, and hands the provider a message list
it will actually accept — including mid-tool-batch, which is when someone asks
"what are you doing?" and is exactly when the live list is not yet legal.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class RecordingStream:
    """Answers every request with fixed events; keeps the requests.

    ``scripted`` answers request N with ``scripted[N]`` instead (the last
    script repeats), for the tests that need the second request to differ
    from the first — the bare-tool-call retry.
    """

    def __init__(
        self,
        events: list[StreamEvent] | None = None,
        *,
        scripted: list[list[StreamEvent]] | None = None,
    ) -> None:
        self.events = events if events is not None else [StreamTextDelta(delta="answer.")]
        self.scripted = scripted
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        if self.scripted is not None:
            events = self.scripted[min(len(self.requests) - 1, len(self.scripted) - 1)]
        else:
            events = self.events

        async def gen():
            for event in events:
                yield event

        return gen()


async def _noop_execute(*_args, **_kwargs) -> ToolResult:
    """A tool body that is never invoked — asides send tools but call none."""
    raise AssertionError("aside tool must not be executed")


def _tool(name: str) -> AgentTool:
    """A minimal live tool, so an aside request carries a real schema."""
    return AgentTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        execute=_noop_execute,
    )


def make_session(tmp_path, stream, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=kwargs.pop("tools", []),
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=kwargs.pop("system_blocks_provider", lambda: ["stable", "env"]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_complete_aside_leaves_no_trace(tmp_path) -> None:
    """THE contract. It reads everything and writes nothing.

    All four surfaces a turn would touch: the live message list (by identity,
    not just length — a replaced list is still a mutation), the transcript, the
    event stream, and the conversation the next real turn will be built from.
    """
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend([Message.user("port it"), Message.assistant("done.")])
    events: list[object] = []
    session.subscribe(events.append)
    before = list(session._context.messages)

    answer = await session.complete_aside([Message.user("why?")])

    assert answer == "answer."
    assert session._context.messages == before
    assert all(a is b for a, b in zip(session._context.messages, before))
    assert session._transcript.entries() == []
    assert events == []
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_sends_the_live_context_and_no_tools(tmp_path) -> None:
    """It answers from the conversation, and cannot act on it.

    The aside sends the SAME live tool schema a working turn sends, not ``[]``:
    the tools block is the front of every provider's cache prefix, so dropping
    it would push the aside off the turn's cached prefix and force a full
    re-process (the regression this fixes). ``tool_choice="none"`` states the
    "reads the turn, calls nothing" intent; the OpenAI and Gemini wires carry
    it literally, and on Anthropic — where it would break the messages cache —
    the session enforces it by never executing a tool-call delta instead.
    """
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream()
    session = make_session(
        tmp_path, stream, tools=tools, system_blocks_provider=lambda: ["stable", "goal: x"]
    )
    session._context.messages.append(Message.user("port it"))

    await session.complete_aside([Message.user("why?")])

    request = stream.requests[-1]
    assert request.system_blocks == ["stable", "goal: x"]
    assert [m.text for m in request.messages] == ["port it", "why?"]
    # Tools mirror the session's live set (the working turn's prefix) exactly —
    # this is the whole point: the aside must send what the turn sends. The
    # session augments the caller's tools with its own built-ins (task, hub,
    # ...), so assert identity with the live set rather than the literal input.
    assert request.tools == session._context.tools
    assert {"bash", "read"} <= {t.name for t in request.tools}
    # And it still cannot call any of them.
    assert request.tool_choice == "none"
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_awaits_an_async_system_blocks_provider(tmp_path) -> None:
    """The real provider is async; a coroutine must not reach the request."""

    async def blocks() -> list[str]:
        return ["stable", "async env"]

    stream = RecordingStream()
    session = make_session(tmp_path, stream, system_blocks_provider=blocks)

    await session.complete_aside([Message.user("why?")])

    assert stream.requests[-1].system_blocks == ["stable", "async env"]
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_pairs_a_dangling_tool_call(tmp_path) -> None:
    """Mid-batch the LIVE list is not legal, and this is the case that matters.

    ``AgentLoop`` appends the assistant message the moment the model turn ends
    and the tool results only once the batch finishes, so for the whole
    duration of every tool call the list ends in an unanswered ``tool_calls``.
    Sending that is a 400 on both wires — and mid-batch is precisely when a
    user asks the aside what the agent is doing.
    """
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend(
        [
            Message.user("run it"),
            Message(
                role="assistant",
                content=[TextContent(text="running")],
                tool_calls=[ToolCall(id="call_1", name="bash", arguments={})],
            ),
        ]
    )

    await session.complete_aside([Message.user("what are you doing?")])

    messages = stream.requests[-1].messages
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    for message in messages:
        for call in message.tool_calls:
            assert call.id in answered, "every tool call must be answered on the wire"
    # And the repair is request-scoped: the live list is untouched. The
    # `isinstance` is a real precondition, not a type-checker appeasement —
    # `_context.messages` is `Message | CustomMessage`, and an aside that had
    # somehow written a bookkeeping entry into it would fail here first.
    live = session._context.messages
    assert all(isinstance(m, Message) for m in live)
    assert [m.role for m in live if isinstance(m, Message)] == ["user", "assistant"]
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_reports_what_it_spent(tmp_path) -> None:
    """An aside carries the whole conversation, so it is not free."""
    stream = RecordingStream(
        [
            StreamTextDelta(delta="answer."),
            StreamUsageEvent(usage=Usage(input_tokens=1200, output_tokens=40)),
            StreamEndEvent(stop_reason="stop"),
        ]
    )
    session = make_session(tmp_path, stream)
    seen: list[Usage] = []

    await session.complete_aside([Message.user("why?")], on_usage=seen.append)

    assert [(u.input_tokens, u.output_tokens) for u in seen] == [(1200, 40)]
    await session.dispose()


#: A model that answered by reaching for a tool. On Anthropic the aside's wire
#: ``tool_choice`` is the turn's ``auto`` (the messages-level cache is keyed on
#: it), so this CAN come back; the session is what keeps it inert.
_BARE_TOOL_CALL: list[StreamEvent] = [
    StreamToolCallDelta(index=0, id="call_1", name="read"),
    StreamToolCallDelta(index=0, argument_delta='{"path": "x"}'),
    StreamUsageEvent(usage=Usage(input_tokens=1000, output_tokens=20)),
    StreamEndEvent(stop_reason="toolUse"),
]


@pytest.mark.asyncio
async def test_complete_aside_retries_a_bare_tool_call_without_tools(tmp_path) -> None:
    """Tool call and no text -> one retry with ``tools=[]``, and its text wins.

    The retry is off the cache prefix by construction (the tools block is the
    front of it), so it must be BOUNDED to this case: exactly one extra
    request, never a loop. Both requests' usage is reported, because both were
    paid for. Nothing executes and nothing joins the history either way.
    """
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream(
        scripted=[
            _BARE_TOOL_CALL,
            [
                StreamTextDelta(delta="because."),
                StreamUsageEvent(usage=Usage(input_tokens=1100, output_tokens=5)),
                StreamEndEvent(stop_reason="stop"),
            ],
        ]
    )
    session = make_session(tmp_path, stream, tools=tools)
    session._context.messages.append(Message.user("port it"))
    before = list(session._context.messages)
    deltas: list[str] = []
    seen: list[Usage] = []

    answer = await session.complete_aside(
        [Message.user("why?")], on_delta=deltas.append, on_usage=seen.append
    )

    assert answer == "because."
    assert deltas == ["because."]
    assert len(stream.requests) == 2
    first, retry = stream.requests
    assert first.tools == session._context.tools and first.tool_choice == "none"
    assert retry.tools == [] and retry.tool_choice == "none"
    # The retry is the same request minus the tools: same history, same system.
    assert [m.text for m in retry.messages] == [m.text for m in first.messages]
    assert retry.system_blocks == first.system_blocks
    assert [u.input_tokens for u in seen] == [1000, 1100]
    assert session._context.messages == before
    assert session._transcript.entries() == []
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_retry_is_bounded_to_one(tmp_path) -> None:
    """A second bare tool call is NOT retried again: the answer is empty and
    the caller sees it, rather than an unbounded loop off the cache prefix."""
    stream = RecordingStream(scripted=[_BARE_TOOL_CALL])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    answer = await session.complete_aside([Message.user("why?")])

    assert answer == ""
    assert len(stream.requests) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_keeps_text_beside_an_inert_tool_call(tmp_path) -> None:
    """Text plus a tool call is an answer: the text is returned, the call is
    dropped unread, and there is no retry (the text is what was asked for)."""
    stream = RecordingStream(
        [
            StreamTextDelta(delta="let me check "),
            StreamToolCallDelta(index=0, id="call_1", name="read"),
            StreamTextDelta(delta="— it was the sed step."),
            StreamEndEvent(stop_reason="toolUse"),
        ]
    )
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    answer = await session.complete_aside([Message.user("why?")])

    assert answer == "let me check — it was the sed step."
    assert len(stream.requests) == 1
    assert session._transcript.entries() == []
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_aside_empty_text_without_a_tool_call_is_not_retried(tmp_path) -> None:
    """An empty answer that was NOT a tool call (a refusal, a length stop) is
    the provider's answer; retrying it would only pay the full prefix twice."""
    stream = RecordingStream([StreamEndEvent(stop_reason="refusal")])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    answer = await session.complete_aside([Message.user("why?")])

    assert answer == ""
    assert len(stream.requests) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_adopt_aside_writes_to_both_the_context_and_the_transcript(tmp_path) -> None:
    """Forking is the door out, so it has to land where a resume will find it."""
    session = make_session(tmp_path, RecordingStream())
    pair = [Message.user("why sed?"), Message.assistant("generated file.")]

    await session.adopt_aside(pair)

    adopted = session._context.messages
    assert all(isinstance(m, Message) for m in adopted)
    assert [m.text for m in adopted if isinstance(m, Message)] == [
        "why sed?",
        "generated file.",
    ]
    assert len(session._transcript.entries()) == 2
    # A resume replays it, which is what "keep this" has to mean.
    replayed = session._transcript.build_llm_history()
    assert all(isinstance(m, Message) for m in replayed)
    assert [m.text for m in replayed if isinstance(m, Message)] == [
        "why sed?",
        "generated file.",
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_adopt_aside_is_refused_mid_turn(tmp_path) -> None:
    """The loop owns the message list then; a splice makes it unsendable."""
    session = make_session(tmp_path, RecordingStream())
    session._is_streaming = True

    with pytest.raises(RuntimeError, match="while a turn is running"):
        await session.adopt_aside([Message.user("why?")])

    assert session._context.messages == []
    assert session._transcript.entries() == []
    await session.dispose()


@pytest.mark.asyncio
async def test_adopt_aside_is_refused_while_the_turn_lock_is_held(tmp_path) -> None:
    """``_is_streaming`` alone is the half ``prompt()`` documents as insufficient.

    The lock is held across the whole pipeline, including a post-compaction
    auto-continuation, where ``_is_streaming`` is already False. A fork landing
    in that gap would be swept into a continuation the user never saw.
    """
    session = make_session(tmp_path, RecordingStream())
    await session._turn_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="while a turn is running"):
            await session.adopt_aside([Message.user("why?")])
    finally:
        session._turn_lock.release()
    await session.dispose()


@pytest.mark.asyncio
async def test_record_shell_writes_a_user_call_and_result(tmp_path) -> None:
    """A bang-mode command is context, not a secret: the next turn and a
    resume both have to see it. Synthetic assistant+tool so the TUI's
    existing replay path mounts a ToolCard rather than a wall of stdout
    attributed to the user."""
    from local_operator.harness.types import ToolResult

    session = make_session(tmp_path, RecordingStream())
    result = ToolResult(
        tool_call_id="shell-1",
        tool_name="bash",
        content=[TextContent(text="exit code: 0\n--- stdout ---\nhi\n--- stderr ---\n(empty)")],
    )

    await session.record_shell("echo hi", result)

    messages = session._context.messages
    assert all(isinstance(m, Message) for m in messages)
    typed = [m for m in messages if isinstance(m, Message)]
    assert [m.role for m in typed] == ["user", "assistant", "tool"]
    assert typed[0].text == "! echo hi"
    assert typed[1].tool_calls[0].name == "bash"
    assert typed[1].tool_calls[0].arguments == {"command": "echo hi"}
    assert typed[2].tool_call_id == "shell-1"
    assert typed[2].text.startswith("exit code: 0")
    replayed = session._transcript.build_llm_history()
    assert all(isinstance(m, Message) for m in replayed)
    assert [m.role for m in replayed if isinstance(m, Message)] == ["user", "assistant", "tool"]
    await session.dispose()


@pytest.mark.asyncio
async def test_record_shell_queues_mid_turn_and_flushes_before_the_next_prompt(tmp_path) -> None:
    """Splicing into a live tool batch is unsendable, but a visible command
    must not disappear. The receipt waits behind the lock and becomes context
    before the next prompt builds its request."""
    from local_operator.harness.types import ToolResult

    session = make_session(tmp_path, RecordingStream())
    result = ToolResult(tool_call_id="shell-1", tool_name="bash", content=[TextContent(text="x")])
    await session._turn_lock.acquire()
    session._is_streaming = True
    try:
        await session.record_shell("echo hi", result)
        assert session._context.messages == []
        assert session._transcript.entries() == []
        assert session._pending_shell_records == [("echo hi", result)]
    finally:
        session._is_streaming = False
        await session._flush_shell_records()
        session._turn_lock.release()

    assert session._pending_shell_records == []
    typed = [m for m in session._context.messages if isinstance(m, Message)]
    assert typed == list(session._context.messages)  # every entry is a real message
    assert [m.role for m in typed] == ["user", "assistant", "tool"]
    first = session._context.messages[0]
    assert isinstance(first, Message)
    assert first.text == "! echo hi"
    replayed = session._transcript.build_llm_history()
    assert [m.role for m in replayed if isinstance(m, Message)] == [
        "user",
        "assistant",
        "tool",
    ]
    await session.dispose()
