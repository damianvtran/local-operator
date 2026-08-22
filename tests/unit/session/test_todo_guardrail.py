"""The todo continuation guardrail: the session half of it.

``tests/unit/harness/test_loop.py`` pins the loop's re-entry mechanism against a
fake callback. These pin the policy the session feeds it — WHEN a nudge fires,
when the no-progress latch swallows it, and that the renderer actually lets it
reach the model. The failure being guarded against is silent in both directions:
a reminder the allow-list drops re-enters the loop with nothing to react to, and
a reminder that fires on an unchanged list burns the loop's continuation budget
and ends the turn with a warning notice instead of an answer.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session, _default_convert_to_llm
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)

#: The store is module state on the tool, so every test here owns its own key.
SESSION_ID = "todo-guardrail"


class ScriptedStream:
    """Replays a per-call event script; records every request it received."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        # Past the script: answer with a bare stop so a guardrail that nudges
        # more than the test expects fails on the assertion, not an IndexError.
        turn = self.turns[len(self.requests) - 1] if len(self.requests) <= len(self.turns) else []

        async def gen():
            for event in turn:
                yield event
            if not turn:
                yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        session_id=SESSION_ID,
        system_blocks_provider=lambda: ["stable"],
    )


def prose_turns(count: int) -> list[list[StreamEvent]]:
    """``count`` turns that answer in prose and call no tool — the exact shape
    that used to end a turn with every todo still open."""
    return [
        [StreamTextDelta(delta=f"prose {index}"), StreamEndEvent(stop_reason="stop")]
        for index in range(count)
    ]


@pytest.fixture(autouse=True)
def clean_store():
    builtin.TODO_STORE.pop(SESSION_ID, None)
    yield
    builtin.TODO_STORE.pop(SESSION_ID, None)


def reminders(messages) -> list[CustomMessage]:
    return [
        message
        for message in messages
        if isinstance(message, CustomMessage)
        and message.custom_type == builtin.TODO_REMINDER_MESSAGE_TYPE
    ]


@pytest.mark.asyncio
async def test_pending_todo_reenters_the_turn_with_the_items(tmp_path) -> None:
    """A prose-only answer with work still open does not end the turn."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "add the columns", "status": "pending"},
        {"text": "backfill", "status": "done"},
    ]
    stream = ScriptedStream(prose_turns(2))
    session = make_session(tmp_path, stream)

    await session.prompt("also add these columns")

    assert len(stream.requests) == 2  # the loop re-entered
    injected = reminders(session._context.messages)
    assert len(injected) == 1
    # It reached the model verbatim, naming the open item and the honest exits.
    sent = stream.requests[1].messages[-1].text
    assert "add the columns" in sent
    assert "backfill" not in sent  # settled work is not re-asserted
    assert "<system-reminder>" in sent
    assert "todo block" in sent and "`ask` tool" in sent
    await session.dispose()


@pytest.mark.asyncio
async def test_unchanged_list_is_nudged_exactly_once(tmp_path) -> None:
    """The no-progress latch. A model that yields twice on a byte-identical
    list cannot proceed; nudging it again would spend the loop's continuation
    budget and delay the user's answer by up to eight model calls."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "decide the domain", "status": "pending"}]
    stream = ScriptedStream(prose_turns(4))
    session = make_session(tmp_path, stream)

    await session.prompt("go")

    assert len(stream.requests) == 2  # one nudge, then the turn ends
    assert len(reminders(session._context.messages)) == 1
    assert session._todo_reminder_fingerprint == (("Todos", "decide the domain", "pending"),)
    await session.dispose()


@pytest.mark.asyncio
async def test_phased_unchanged_list_is_nudged_exactly_once(tmp_path) -> None:
    """THE §5.3 coupling guard. A PHASED no-progress list must be nudged exactly
    once, the same as a flat one. If ``session.py:_stamped_todo_fingerprint`` is
    NOT widened to 3-tuples in lockstep with ``builtin.todo_fingerprint``, the
    stamped side compares empty, every reminder expires on every render, and the
    latch nudges again on the second identical yield. That failure is SILENT in
    the flat suite (whose stamps happen to survive) and in any 'does it nudge?'
    test — only a phased 'does it STOP nudging?' assertion catches it."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"name": "Foundation", "items": [{"text": "decide the domain", "status": "pending"}]},
        {"name": "Verification", "items": [{"text": "run the gate", "status": "pending"}]},
    ]
    stream = ScriptedStream(prose_turns(4))
    session = make_session(tmp_path, stream)

    await session.prompt("go")

    assert len(stream.requests) == 2, "one nudge, then the turn ends — not re-nudged every yield"
    assert len(reminders(session._context.messages)) == 1
    assert session._todo_reminder_fingerprint == (
        ("Foundation", "decide the domain", "pending"),
        ("Verification", "run the gate", "pending"),
    )
    # The reminder must actually REACH the model on the continuation request:
    # that render passes through ``_expire_stale_todo_reminders`` →
    # ``_stamped_todo_fingerprint``. If that normaliser is NOT widened to
    # 3-tuples in lockstep (§5.3), the stamped side compares empty against the
    # live 3-tuple, the reminder is judged stale on an UNCHANGED list, and it is
    # stripped from this request — so the model never sees the nudge. Asserting
    # the text is present on request 2 is what catches the silent coupling break.
    sent = stream.requests[1].messages[-1].text
    assert "decide the domain" in sent and "<system-reminder>" in sent
    await session.dispose()


@pytest.mark.asyncio
async def test_progress_earns_another_nudge(tmp_path) -> None:
    """Movement re-arms the guardrail: the latch is about a STUCK model, not a
    cap on how many times a working one may be reminded."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "one", "status": "pending"},
        {"text": "two", "status": "pending"},
    ]
    stream = ScriptedStream(prose_turns(3))
    session = make_session(tmp_path, stream)
    original = session._todo_continuation

    async def continuation_then_progress():
        result = await original()
        # Stand in for the model closing an item between boundaries.
        for item in builtin.TODO_STORE[SESSION_ID]:
            if item["status"] == "pending":
                item["status"] = "done"
                break
        return result

    session._todo_continuation = continuation_then_progress  # type: ignore[method-assign]

    await session.prompt("go")

    # Nudge, item closed, nudge again, item closed, list empty -> turn ends.
    assert len(stream.requests) == 3
    assert len(reminders(session._context.messages)) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_user_turn_rearms_the_latch(tmp_path) -> None:
    """The user's next message may well be the answer the list was waiting on,
    so the latch must not carry across turns."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "decide the domain", "status": "pending"}]
    stream = ScriptedStream(prose_turns(6))
    session = make_session(tmp_path, stream)

    await session.prompt("go")
    assert session._todo_reminder_fingerprint is not None
    await session.prompt("use example.com")

    # Two requests per turn: the prose answer plus one nudge each.
    assert len(stream.requests) == 4
    assert len(reminders(session._context.messages)) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_blocked_only_list_does_not_fire(tmp_path) -> None:
    """``blocked`` is the honest stop: the guardrail must let the turn end so
    the user can answer, instead of pushing the model to invent progress."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "pick a domain", "status": "blocked", "reason": "needs the user's call"},
        {"text": "old plan", "status": "dropped"},
        {"text": "schema", "status": "done"},
    ]
    stream = ScriptedStream(prose_turns(1))
    session = make_session(tmp_path, stream)

    await session.prompt("go")

    assert len(stream.requests) == 1
    assert reminders(session._context.messages) == []
    await session.dispose()


@pytest.mark.asyncio
async def test_no_todos_at_all_does_not_fire(tmp_path) -> None:
    stream = ScriptedStream(prose_turns(1))
    session = make_session(tmp_path, stream)

    await session.prompt("go")

    assert len(stream.requests) == 1
    assert reminders(session._context.messages) == []
    await session.dispose()


@pytest.mark.asyncio
async def test_reminder_leaves_no_trace_in_the_transcript(tmp_path) -> None:
    """The nudge is model-visible and user-invisible: nothing persists it, so a
    resume never replays a stale claim about a list that has moved on."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "ship it", "status": "pending"}]
    stream = ScriptedStream(prose_turns(2))
    session = make_session(tmp_path, stream)
    events: list[object] = []
    session.subscribe(events.append)

    await session.prompt("go")

    entries = session._transcript.entries()
    assert not any(
        builtin.TODO_REMINDER_MESSAGE_TYPE in repr(entry) for entry in entries
    ), "the reminder must never be persisted"
    # Every event shape this session emits is checked by repr, not by one known
    # attribute: a nudge leaking through any field is the failure, not just a
    # nudge leaking through `text`.
    assert not any(
        "<system-reminder>" in repr(event) for event in events
    ), "the reminder must never reach the event stream"
    await session.dispose()


# --- the renderer -----------------------------------------------------------


def _reminder(text: str, fingerprint=None) -> CustomMessage:
    details: dict[str, object] = {"text": text}
    if fingerprint is not None:
        details["fingerprint"] = fingerprint
    return CustomMessage(
        custom_type=builtin.TODO_REMINDER_MESSAGE_TYPE,
        attribution="system",
        details=details,
    )


def test_renderer_passes_the_reminder_through() -> None:
    """``_default_convert_to_llm`` is an ALLOW-LIST: an unlisted custom type is
    dropped as bookkeeping, which would leave the loop re-entering with nothing
    for the model to react to."""
    reminder = _reminder("still open: ship it")

    rendered = _default_convert_to_llm([Message.user("go"), reminder])

    assert [message.role for message in rendered] == ["user", "user"]
    assert rendered[-1].text == "still open: ship it"
    assert rendered[-1].id == reminder.id  # entry id preserved, like every branch


def test_renderer_keeps_only_the_newest_reminder() -> None:
    """An older reminder asserts a list that has since changed — replaying it
    would feed the model a claim that is now false."""
    old = _reminder("open: a, b")
    new = _reminder("open: b")

    rendered = _default_convert_to_llm([Message.user("go"), old, Message.assistant("working"), new])

    assert [message.text for message in rendered] == ["go", "working", "open: b"]


@pytest.mark.asyncio
async def test_reminder_expires_when_the_list_moves(tmp_path) -> None:
    """A reminder is a point-in-time assertion, and the ONE reminder case is the
    one that bites: it stays in the live message list after the model resolves
    the work, and every later request would keep insisting the finished item is
    still open."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "ship it", "status": "pending"}]
    stream = ScriptedStream(prose_turns(2))
    session = make_session(tmp_path, stream)
    await session.prompt("go")
    assert len(reminders(session._context.messages)) == 1
    # While the list stands still the assertion is still true, so it is still sent.
    still_sent = session._render_history(session._context.messages)
    assert any("<system-reminder>" in message.text for message in still_sent)

    builtin.TODO_STORE[SESSION_ID][0]["status"] = "done"

    rendered = session._render_history(session._context.messages)
    assert not any("<system-reminder>" in message.text for message in rendered)
    # Expiry is a RENDER decision: the live list is never rewritten behind the
    # loop's back, so the guardrail's own latch still reads what it wrote.
    assert len(reminders(session._context.messages)) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_unstamped_reminder_expires(tmp_path) -> None:
    """No stamp means the claim cannot be checked, and an unverifiable nudge is
    worth less than one turn without it."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "ship it", "status": "pending"}]
    stream = ScriptedStream(prose_turns(1))
    session = make_session(tmp_path, stream)

    kept = session._live_todo_reminders(
        [Message.user("go"), _reminder("stamped", (("Todos", "ship it", "pending"),))]
    )
    dropped = session._live_todo_reminders([Message.user("go"), _reminder("no stamp")])

    assert len(kept) == 2
    assert len(dropped) == 1
    await session.dispose()


def test_phased_open_todos_flattens_pending_and_excludes_blocked() -> None:
    """The guardrail's \"open\" is pending across ALL phases; blocked stays out so
    a phased list can still stop honestly."""
    builtin.TODO_STORE[SESSION_ID] = [
        {
            "name": "A",
            "items": [
                {"text": "one", "status": "pending"},
                {"text": "settled", "status": "done"},
            ],
        },
        {
            "name": "B",
            "items": [
                {"text": "two", "status": "pending"},
                {"text": "held", "status": "blocked", "reason": "waiting"},
            ],
        },
    ]

    assert [item["text"] for item in builtin.open_todos(SESSION_ID)] == ["one", "two"]


def test_stamped_fingerprint_round_trips_a_three_tuple() -> None:
    """A reminder's stamped fingerprint survives the JSON list round trip and
    compares equal to the live 3-tuple, so an unchanged phased list keeps its
    reminder instead of expiring it (the §5.3 failure mode)."""
    from local_operator.session.session import _stamped_todo_fingerprint

    # JSON turns the nested tuples into lists; the normaliser must still yield a
    # 3-tuple that equals what todo_fingerprint emits.
    details = {"fingerprint": [["Foundation", "decide", "pending"]]}
    assert _stamped_todo_fingerprint(details) == (("Foundation", "decide", "pending"),)
