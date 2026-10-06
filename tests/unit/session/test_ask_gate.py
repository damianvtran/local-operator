"""``Session._gate_ask`` / ``Session.complete_clearance`` — the gate's two halves.

``_gate_ask`` is the ask gate's TOTALITY TABLE (design docs/design/ask-gate.md
§2.2): every non-divert path returns ``None`` and falls through to the
unchanged enqueue, and the only way an ask is lost would be a bug in the
enqueue path that exists today. ``complete_clearance`` is the no-write fork
(§2.1): it reads exactly what an aside reads and writes nothing — no
transcript entry, no context append, no event — on every path including
failure. Both are pinned against REAL Session objects with a fake stream,
mirroring ``test_aside.py``'s enforcement style.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.harness.types import (
    AskOption,
    AskQuestion,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.session import clearance
from local_operator.session.errors import AsideUnanswered
from local_operator.session.session import Session

from .test_aside import _BARE_TOOL_CALL, RecordingStream, _tool, make_session


def _questions(*, secret: bool = False, question: str = "Deploy now?") -> list[AskQuestion]:
    if secret:
        # A secret question carries no options (the answer is a pasted value).
        return [AskQuestion(id="q0", question="Paste the deploy token.", secret=True)]
    return [
        AskQuestion(
            id="q0",
            question=question,
            options=[
                AskOption(label="Ship it", description="after the freeze"),
                AskOption(label="Wait", description="until Monday"),
            ],
        )
    ]


async def _noop_hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
    return {questions[0].id: ["Ship it"]}


def _queue_live(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give ``session`` the queued engine the way production does: flag + hook."""
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    session.set_ask_handler(_noop_hook)


class FakeClearance:
    """A scripted ``complete_clearance`` that records the turns it was given."""

    def __init__(self, answers: str | BaseException) -> None:
        self.answers = answers
        self.calls: list[list[Any]] = []

    async def __call__(self, turns: list[Any]) -> str:
        self.calls.append(list(turns))
        if isinstance(self.answers, BaseException):
            # CancelledError is a BaseException, NOT an Exception — the helper
            # must raise it as itself or the fail-open path never runs.
            raise self.answers
        return self.answers


def _install(
    session: Session, monkeypatch: pytest.MonkeyPatch, answers: str | BaseException
) -> FakeClearance:
    fake = FakeClearance(answers)
    monkeypatch.setattr(session, "complete_clearance", fake)
    return fake


# --- the totality table ------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_flag_off_never_calls_the_fork(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    monkeypatch.setattr(policy, "ASK_GATE", False)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: x")

    assert await session._gate_ask(_questions()) is None
    assert fake.calls == []
    assert session._ask_gate_diverts == {}
    await session.dispose()


@pytest.mark.asyncio
async def test_blocking_arm_never_calls_the_fork(tmp_path, monkeypatch) -> None:
    """No queue (defense; unreachable through the tool) ⇒ enqueue unchanged."""
    session = make_session(tmp_path, RecordingStream())
    session.set_ask_handler(_noop_hook)  # hook present, flag off
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: x")

    assert session.ask_queue() is None
    assert await session._gate_ask(_questions()) is None
    assert fake.calls == []
    await session.dispose()


@pytest.mark.asyncio
async def test_secret_question_skips_the_fork_entirely(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: x")

    assert await session._gate_ask(_questions(secret=True)) is None
    assert fake.calls == []
    assert session._ask_gate_diverts == {}
    await session.dispose()


@pytest.mark.asyncio
async def test_clear_diverts_with_the_marker_and_records_the_fingerprint(
    tmp_path, monkeypatch
) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: the log settles it")
    questions = _questions()

    verdict = await session._gate_ask(questions)

    assert verdict is not None
    assert verdict["verdict"] == "clear"
    assert "No question was put to the user." in verdict["text"]
    assert "Check's reason: the log settles it" in verdict["text"]
    assert verdict["details"] == {
        "ask_gate": {"hidden": True, "verdict": "clear", "reason": "the log settles it"}
    }
    digest = clearance.fingerprint(questions)
    assert digest in session._ask_gate_diverts
    # The fork saw exactly ONE user-role turn carrying the gate message.
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 1
    assert "VERDICT: clear|resolve|raise" in fake.calls[0][0].text
    await session.dispose()


@pytest.mark.asyncio
async def test_resolve_diverts_with_the_one_subagent_note(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, "VERDICT: resolve\nREASON: ask the architect")

    verdict = await session._gate_ask(_questions())

    assert verdict is not None and verdict["verdict"] == "resolve"
    assert "ONE `task` subagent" in verdict["text"]
    await session.dispose()


@pytest.mark.asyncio
async def test_fingerprint_hit_skips_the_second_check(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: x")
    questions = _questions()

    first = await session._gate_ask(questions)
    second = await session._gate_ask(questions)

    assert first is not None and second is None
    assert len(fake.calls) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_a_hit_moves_the_entry_to_the_lru_end(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, "VERDICT: clear\nREASON: x")
    a, b = _questions(question="A?"), _questions(question="B?")

    await session._gate_ask(a)
    await session._gate_ask(b)
    await session._gate_ask(a)  # hit → recency moves A last

    assert list(session._ask_gate_diverts) == [
        clearance.fingerprint(b),
        clearance.fingerprint(a),
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_lru_eviction_drops_the_oldest_fingerprint(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    fake = _install(session, monkeypatch, "VERDICT: clear\nREASON: x")
    digests = []
    for index in range(clearance.GATE_FINGERPRINT_CAP + 1):
        questions = _questions(question=f"Question {index}?")
        digests.append(clearance.fingerprint(questions))
        await session._gate_ask(questions)

    assert len(session._ask_gate_diverts) == clearance.GATE_FINGERPRINT_CAP
    assert digests[0] not in session._ask_gate_diverts
    assert digests[-1] in session._ask_gate_diverts
    # The evicted content is now a fresh ask again: one new check, never a
    # lost ask — the skip direction is the safe one (design §2.5).
    calls_before = len(fake.calls)
    await session._gate_ask(_questions(question="Question 0?"))
    assert len(fake.calls) == calls_before + 1
    await session.dispose()


@pytest.mark.asyncio
async def test_timeout_fails_open(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    monkeypatch.setattr(policy, "GATE_TIMEOUT_S", 0.01)

    async def slow(turns: list[Any]) -> str:
        await asyncio.sleep(30)
        return "VERDICT: clear"

    monkeypatch.setattr(session, "complete_clearance", slow)
    assert await session._gate_ask(_questions()) is None
    assert session._ask_gate_diverts == {}
    await session.dispose()


@pytest.mark.asyncio
async def test_provider_exception_fails_open(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, RuntimeError("provider exploded"))

    assert await session._gate_ask(_questions()) is None
    assert session._ask_gate_diverts == {}
    await session.dispose()


@pytest.mark.asyncio
async def test_unparseable_answer_fails_open(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, "I have no idea what to say.")

    assert await session._gate_ask(_questions()) is None
    assert session._ask_gate_diverts == {}
    await session.dispose()


@pytest.mark.asyncio
async def test_raise_enqueues_and_records_nothing(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    fake = _install(session, monkeypatch, "VERDICT: raise\nREASON: their call")

    assert await session._gate_ask(_questions()) is None
    assert session._ask_gate_diverts == {}
    # Nothing recorded ⇒ the next raise of the same content runs one more
    # check: the queue's own caps own the re-ask path there.
    assert await session._gate_ask(_questions()) is None
    assert len(fake.calls) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_cancelled_error_propagates(tmp_path, monkeypatch) -> None:
    """An aborted turn must keep aborting — the gate never swallows it (§2.2)."""
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await session._gate_ask(_questions())
    await session.dispose()


@pytest.mark.asyncio
async def test_missing_reason_still_diverts(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    _queue_live(session, monkeypatch)
    _install(session, monkeypatch, "VERDICT: clear")

    verdict = await session._gate_ask(_questions())

    assert verdict is not None and verdict["verdict"] == "clear"
    assert "Check's reason" not in verdict["text"]
    assert verdict["details"]["ask_gate"]["reason"] == ""
    await session.dispose()


# --- the binding -------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_door_binds_under_the_same_conditions_as_enqueue(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())

    assert session._ask_gate_callable() is None
    context = session._build_tool_context()
    assert context.gate_ask is None and context.enqueue_ask is None

    _queue_live(session, monkeypatch)
    assert callable(session._ask_gate_callable())
    context = session._build_tool_context()
    assert callable(context.gate_ask)
    assert callable(context.enqueue_ask)
    await session.dispose()


@pytest.mark.asyncio
async def test_gate_door_is_none_on_the_blocking_arm(tmp_path, monkeypatch) -> None:
    session = make_session(tmp_path, RecordingStream())
    session.set_ask_handler(_noop_hook)
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", False)

    assert session._ask_gate_callable() is None
    assert session._build_tool_context().gate_ask is None
    await session.dispose()


# --- complete_clearance: the no-write fork -----------------------------------


@pytest.mark.asyncio
async def test_complete_clearance_leaves_no_trace(tmp_path) -> None:
    """The §2.1 contract: reads everything, writes nothing, on every surface."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend([Message.user("port it"), Message.assistant("done.")])
    events: list[object] = []
    session.subscribe(events.append)
    before = list(session._context.messages)

    answer = await session.complete_clearance([Message.user("<ask-clearance>x</ask-clearance>")])

    assert answer == "answer."
    assert all(a is b for a, b in zip(session._context.messages, before))
    assert session._transcript.entries() == []
    assert events == []
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_request_shape(tmp_path) -> None:
    """purpose / tools / tool_choice / warm-prefix, exactly as the design tables."""
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream()
    session = make_session(tmp_path, stream, tools=tools)
    session._context.messages.append(Message.user("port it"))

    await session.complete_clearance([Message.user("gate message")])

    request = stream.requests[-1]
    assert request.purpose == "clearance"
    assert request.replayable is True
    # ``isolated`` stays absent/False — load-bearing: isolation would strip the
    # session's cache key and put the gate on a cold namespace (design §2.1).
    assert request.isolated is False
    assert request.prompt_cache_key is None  # the session's stream fn stamps the lineage
    assert request.tool_choice == "none"
    # Tools mirror the live set — the front of the cached prefix (never []).
    assert request.tools == session._context.tools
    assert [m.text for m in request.messages] == ["port it", "gate message"]
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_pairs_a_dangling_tool_call(tmp_path) -> None:
    """Mid-batch the live list is not wire-legal; the fork rides ``_wire_legal_snapshot``."""
    from local_operator.harness.types import ToolCall

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

    await session.complete_clearance([Message.user("gate message")])

    messages = stream.requests[-1].messages
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    for message in messages:
        for call in message.tool_calls:
            assert call.id in answered, "every tool call must be answered on the wire"
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retries_a_bare_tool_call_without_tools(tmp_path) -> None:
    """The aside's one bounded retry, inherited: rejected call handed back, paired."""
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream(
        scripted=[
            _BARE_TOOL_CALL,
            [
                StreamTextDelta(delta="VERDICT: clear\nREASON: fine."),
                StreamEndEvent(stop_reason="stop"),
            ],
        ]
    )
    session = make_session(tmp_path, stream, tools=tools)

    answer = await session.complete_clearance([Message.user("gate message")])

    assert answer == "VERDICT: clear\nREASON: fine."
    assert len(stream.requests) == 2
    first, retry = stream.requests
    assert first.tools == session._context.tools
    assert retry.tools == []
    call, refusal = retry.messages[-2:]
    assert call.role == "assistant" and [(c.id, c.name) for c in call.tool_calls] == [
        ("call_1", "read")
    ]
    assert refusal.role == "tool" and refusal.tool_call_id == "call_1"
    assert "clearance check" in refusal.text
    assert "VERDICT: clear|resolve|raise" in refusal.text
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retry_is_bounded_to_one(tmp_path) -> None:
    stream = RecordingStream(scripted=[_BARE_TOOL_CALL])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    with pytest.raises(AsideUnanswered):
        await session.complete_clearance([Message.user("gate message")])

    assert len(stream.requests) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retry_that_answers_nothing_raises(tmp_path) -> None:
    stream = RecordingStream(scripted=[_BARE_TOOL_CALL, [StreamEndEvent(stop_reason="refusal")]])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    with pytest.raises(AsideUnanswered):
        await session.complete_clearance([Message.user("gate message")])

    assert len(stream.requests) == 2
    await session.dispose()
