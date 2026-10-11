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

from local_operator.harness.message_types import TODO_REMINDER_MESSAGE_TYPE
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
from local_operator.session.session import (
    Session,
    _default_convert_to_llm,
    _todo_reminder_text,
)
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
        if isinstance(message, CustomMessage) and message.custom_type == TODO_REMINDER_MESSAGE_TYPE
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
    assert session._todo_reminder_fingerprint == (("Todos", "decide the domain", "pending", ""),)
    await session.dispose()


@pytest.mark.asyncio
async def test_phased_unchanged_list_is_nudged_exactly_once(tmp_path) -> None:
    """THE §5.3 coupling guard. A PHASED no-progress list must be nudged exactly
    once, the same as a flat one. If ``session.py:_stamped_todo_fingerprint`` is
    NOT widened to 4-tuples in lockstep with ``builtin.todo_fingerprint``, the
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
        ("Foundation", "decide the domain", "pending", ""),
        ("Verification", "run the gate", "pending", ""),
    )
    # The reminder must actually REACH the model on the continuation request:
    # that render passes through ``_expire_stale_todo_reminders`` →
    # ``_stamped_todo_fingerprint``. If that normaliser is NOT widened to
    # 4-tuples in lockstep (§5.3), the stamped side compares empty against the
    # live 4-tuple, the reminder is judged stale on an UNCHANGED list, and it is
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
async def test_blocked_only_list_fires_a_reconciliation_nudge(tmp_path) -> None:
    """Blocked is no longer a list the guardrail ignores. The honest stop still
    ends the turn, but the model must account for what it parked first — this is
    the ~14-hours-on-six-blocked-items case. Exactly ONE pass: the byte-identical
    re-yield after the accounting is released, so the block is reconciled, never
    forbidden."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "pick a domain", "status": "blocked", "reason": "needs the user's call"},
        {"text": "old plan", "status": "dropped"},
        {"text": "schema", "status": "done"},
    ]
    stream = ScriptedStream(prose_turns(4))
    session = make_session(tmp_path, stream)

    await session.prompt("go")

    assert len(stream.requests) == 2  # one accounting pass, then released
    assert len(reminders(session._context.messages)) == 1
    # It reached the model on the continuation request, naming the parked item,
    # the condition it is parked on, and the reconciliation it owes.
    sent = stream.requests[1].messages[-1].text
    assert "pick a domain" in sent
    assert "needs the user's call" in sent  # the reason IS the condition to re-check
    assert "account for each blocked item" in sent
    # Settled work is still not re-asserted, and the pending paragraph stays out
    # of a list with nothing pending: it is an instruction about work that is not
    # there, and the model would have to reconcile it against its own list.
    assert "old plan" not in sent and "schema" not in sent
    assert "Do not end the turn while pending items remain" not in sent
    # The latch still compares the FULL list, settled items included, and a
    # blocked item's reason is part of it (the round-1 remediation).
    assert session._todo_reminder_fingerprint == (
        ("Todos", "pick a domain", "blocked", "needs the user's call"),
        ("Todos", "old plan", "dropped", ""),
        ("Todos", "schema", "done", ""),
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_user_turn_rearms_the_latch_on_a_parked_list(tmp_path) -> None:
    """The re-arm applies to blocked work too: the user's next message may be the
    answer the block was waiting on, so the latch must not carry across turns —
    otherwise a parked list is accounted for exactly once, ever."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "pick a domain", "status": "blocked", "reason": "needs the user's call"}
    ]
    stream = ScriptedStream(prose_turns(6))
    session = make_session(tmp_path, stream)

    await session.prompt("go")
    assert session._todo_reminder_fingerprint is not None
    await session.prompt("use example.com")

    # Two requests per turn: the prose answer plus one reconciliation each, and
    # the second turn is released after its own single accounting pass.
    assert len(stream.requests) == 4
    assert len(reminders(session._context.messages)) == 2
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
        TODO_REMINDER_MESSAGE_TYPE in repr(entry) for entry in entries
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
        custom_type=TODO_REMINDER_MESSAGE_TYPE,
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
async def test_reminder_expires_and_re_arms_when_a_blocked_reason_is_rewritten(
    tmp_path,
) -> None:
    """The reason is rendered AND fingerprinted, which is the round-1 finding: a
    re-``block`` that corrects the condition leaves status and text alone, so a
    reason-blind latch called the list unmoved, kept the live reminder asserting
    the OLD wait, and never let the expiry check retire it."""
    builtin.TODO_STORE[SESSION_ID] = [
        {"text": "deploy", "status": "blocked", "reason": "waiting on legal"}
    ]
    stream = ScriptedStream(prose_turns(2))
    session = make_session(tmp_path, stream)
    await session.prompt("go")
    assert len(reminders(session._context.messages)) == 1
    # The latch holds while nothing moves — the one-accounting-pass bound.
    assert await session._todo_continuation() == []

    # Status and text unchanged: only the condition the reminder renders moved.
    builtin.TODO_STORE[SESSION_ID][0]["reason"] = "legal cleared it; waiting on finance"

    rendered = session._render_history(session._context.messages)
    expired = not any("<system-reminder>" in message.text for message in rendered)
    assert expired, "the stale reminder asserts the old condition and must expire"
    # And the latch sees it as movement, so the replacement text carries the
    # corrected condition rather than repeating the stale one.
    nudges = await session._todo_continuation()
    assert len(nudges) == 1
    assert "waiting on finance" in nudges[0].details["text"]
    assert "waiting on legal" not in nudges[0].details["text"]
    await session.dispose()


@pytest.mark.asyncio
async def test_unstamped_reminder_expires(tmp_path) -> None:
    """No stamp means the claim cannot be checked, and an unverifiable nudge is
    worth less than one turn without it."""
    builtin.TODO_STORE[SESSION_ID] = [{"text": "ship it", "status": "pending"}]
    stream = ScriptedStream(prose_turns(1))
    session = make_session(tmp_path, stream)

    kept = session._live_todo_reminders(
        [Message.user("go"), _reminder("stamped", (("Todos", "ship it", "pending", ""),))]
    )
    dropped = session._live_todo_reminders([Message.user("go"), _reminder("no stamp")])

    assert len(kept) == 2
    assert len(dropped) == 1
    await session.dispose()


def test_phased_open_todos_flattens_pending_and_excludes_blocked() -> None:
    """``open_todos`` is the PENDING projection across all phases — blocked stays
    out of it (the guardrail draws it separately through ``blocked_todos``, which
    is what keeps the two obligations apart). Both flatten the same phased store
    in phase-then-item order."""
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
    assert [item["text"] for item in builtin.blocked_todos(SESSION_ID)] == ["held"]


# --- the reminder text ------------------------------------------------------


def test_reminder_text_states_each_group_only_when_present() -> None:
    """One text, two groups. Pending rows carry their paragraph; blocked rows
    carry their reason and the reconciliation paragraph. A group with no items
    contributes no paragraph, because an instruction about work that is not on
    the list is noise the model has to reconcile away."""
    pending = [{"text": "add the columns", "status": "pending"}]
    blocked = [{"text": "pick a domain", "status": "blocked", "reason": "needs the user's call"}]

    mixed = _todo_reminder_text(pending, blocked)
    assert mixed.startswith("<system-reminder>") and mixed.endswith("</system-reminder>")
    assert "- add the columns" in mixed
    assert "- pick a domain — blocked: needs the user's call" in mixed
    assert "Do not end the turn while pending items remain" in mixed
    assert "Before you finish, account for each blocked item" in mixed

    pending_only = _todo_reminder_text(pending, [])
    assert "Before you finish, account for each blocked item" not in pending_only
    # The exits the tool actually offers, and the ask clause, stay named.
    assert "`todo block`" in pending_only and "`ask` tool" in pending_only

    blocked_only = _todo_reminder_text([], blocked)
    assert "Do not end the turn while pending items remain" not in blocked_only
    assert "- pick a domain — blocked: needs the user's call" in blocked_only

    # A blocked row with no reason (a hand-edited or pre-``reason`` store) keeps
    # the state and drops the empty tail — the panel's own ``— blocked`` form.
    bare = _todo_reminder_text([], [{"text": "x", "status": "blocked"}])
    assert bare.count("— blocked") == 1
    assert "— blocked:" not in bare


def test_stamped_fingerprint_round_trips_a_four_tuple() -> None:
    """A reminder's stamped fingerprint survives the JSON list round trip and
    compares equal to the live 4-tuple, so an unchanged phased list keeps its
    reminder instead of expiring it (the §5.3 failure mode)."""
    from local_operator.session.session import _stamped_todo_fingerprint

    # JSON turns the nested tuples into lists; the normaliser must still yield a
    # 4-tuple that equals what todo_fingerprint emits.
    details = {"fingerprint": [["Foundation", "decide", "blocked", "needs the user"]]}
    assert _stamped_todo_fingerprint(details) == (
        ("Foundation", "decide", "blocked", "needs the user"),
    )
