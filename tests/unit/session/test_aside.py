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
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class RecordingStream:
    """Answers every request with fixed events; keeps the requests."""

    def __init__(self, events: list[StreamEvent] | None = None) -> None:
        self.events = events if events is not None else [StreamTextDelta(delta="answer.")]
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            for event in self.events:
                yield event

        return gen()


def make_session(tmp_path, stream, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
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
    """It answers from the conversation, and cannot act on it."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream, system_blocks_provider=lambda: ["stable", "goal: x"])
    session._context.messages.append(Message.user("port it"))

    await session.complete_aside([Message.user("why?")])

    request = stream.requests[-1]
    assert request.system_blocks == ["stable", "goal: x"]
    assert [m.text for m in request.messages] == ["port it", "why?"]
    assert request.tools == []
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
    assert [m.role for m in session._context.messages if isinstance(m, Message)] == [
        "user",
        "assistant",
        "tool",
    ]
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
