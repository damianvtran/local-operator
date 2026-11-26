"""End-to-end: a steering message mid-turn must not end the turn early.

These drive the REAL :class:`Session`, the REAL :class:`AgentLoop` and the REAL
``todo`` tool against a scripted provider. ``tests/unit/session/
test_todo_guardrail.py`` pins the guardrail's units; these pin the scenario
that motivated it, because every piece of it was individually correct while the
whole was not:

    user prompts → model lists 6 todos → user types a new requirement while the
    turn is still running → the model answers THAT and stops → turn over with
    ``Todos · 0/6``.

Steering is not what breaks it: the loop injects a steering message and keeps
going. The break is one hop later, when the model treats the answer to the
steering message as its final word. Nothing re-asserted the outstanding work,
so ``has_more_tool_calls`` went false and the loop reached ``break``.

The reminder must also stay OFF the transcript: it is harness bookkeeping, and
a user who scrolls back should see their own words, not a synthesized turn they
never typed.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import pytest

from local_operator.harness.message_types import TODO_REMINDER_MESSAGE_TYPE
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentEvent,
    AgentMessage,
    ChatRequest,
    CustomMessage,
    Message,
    ModelSpec,
    OutputValidationEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
)
from local_operator.harness.wake_types import WakeSchedule
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


@pytest.fixture(autouse=True)
def _clean_todo_store():
    """The todo store is a process-global table (``builtin.TODO_STORE``), so a
    leaked list would leak into the next test's guardrail decision."""
    builtin.TODO_STORE.clear()
    yield
    builtin.TODO_STORE.clear()


class SteeringStream:
    """Scripted provider that can act like a user typing mid-turn.

    ``turns`` is replayed one entry per request. An entry may carry a
    ``steer`` callable, invoked when that request arrives — which is how a
    message typed WHILE the model is streaming is reproduced deterministically,
    without racing a real keyboard against a real turn.
    """

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        index = len(self.requests) - 1
        # Past the end of the script the model has nothing left to say. Ending
        # with plain text (rather than raising) keeps a script that under-runs
        # readable in the failure output: the assertion reports the request
        # count, not an exception from the fake.
        turn: dict[str, Any] = self.turns[index] if index < len(self.turns) else {"events": []}
        steer: Callable[[], None] | None = turn.get("steer")
        if steer is not None:
            steer()

        async def gen():
            if error := turn.get("error"):
                raise RuntimeError(error)
            for event in turn.get("events", []):
                yield event

        return gen()

    def request_texts(self, index: int) -> list[str]:
        """Every text block the provider was handed on request ``index``."""
        return [
            block.text
            for message in self.requests[index].messages
            for block in message.content
            if isinstance(block, TextContent)
        ]


def todo_call(call_id: str, payload: dict[str, Any]) -> list[StreamEvent]:
    return [
        StreamToolCallDelta(index=0, id=call_id, name="todo"),
        StreamToolCallDelta(index=0, argument_delta=json.dumps(payload)),
    ]


def make_session(tmp_path, stream, **kwargs) -> Session:
    """A session holding the REAL todo tool, built the way a host builds it."""
    context = ToolContext(cwd=str(tmp_path), session_id="e2e", has_ui=False)
    tools = [tool for tool in create_tools(context, ["todo"])]
    assert [tool.name for tool in tools] == ["todo"], "the real todo tool must be present"
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(tmp_path / "sess"),
        session_id="e2e",
        cwd=str(tmp_path),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


def reminders_in(stream: SteeringStream, index: int) -> list[str]:
    return [text for text in stream.request_texts(index) if "<system-reminder>" in text]


def reminders(session: Session) -> list[Any]:
    """Every todo reminder appended to the live context so far, expired ones
    included — expiry is a RENDER decision, the live list is never rewritten."""
    return [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == TODO_REMINDER_MESSAGE_TYPE
    ]


@pytest.mark.asyncio
async def test_steering_mid_turn_does_not_end_the_turn_with_open_todos(tmp_path) -> None:
    """THE regression. The model answers a steered requirement and stops; the
    guardrail must put it back to work instead of letting the turn end 0/4."""
    session: Session | None = None

    def steer_new_requirement() -> None:
        # Typed while the first tool batch is still in flight — the loop drains
        # it at the next boundary, exactly like the reported session.
        assert session is not None
        session.steer("also add an assigned_user_email column")

    stream = SteeringStream(
        [
            # 1. Plan the work.
            {
                "events": todo_call("c1", {"op": "init", "items": ["read repo", "write code"]}),
                "steer": steer_new_requirement,
            },
            # 2. Answer the steering message in prose and try to stop. Before
            #    the guardrail this was the end of the turn.
            {"events": [StreamTextDelta(delta="Good idea, I'll add that column.")]},
            # 3. Nudged: record the new requirement instead of losing it.
            {"events": todo_call("c2", {"op": "add", "items": ["add email column"]})},
            # 4. Do the work: one call closing all three (the old tool silently
            #    dropped items[1:] here).
            {
                "events": todo_call(
                    "c3",
                    {"op": "done", "items": ["read repo", "write code", "add email column"]},
                )
            },
            # 5. Nothing open: this yield is allowed to stand.
            {"events": [StreamTextDelta(delta="All done.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("do the work")

    # The turn did NOT stop at request 2.
    assert len(stream.requests) == 5, f"turn ended after {len(stream.requests)} requests"

    # Request 3 is the one that only exists because the guardrail fired.
    nudges = reminders_in(stream, 2)
    assert len(nudges) == 1, "exactly one reminder should be in play"
    assert "read repo" in nudges[0] and "write code" in nudges[0]
    assert "todo" in nudges[0], "the reminder must name the tool that resolves the items"

    # The steered requirement survived as a tracked item rather than a promise
    # in prose, and every item ended up resolved.
    # The store is now phased; a flat init lives in one implicit "Todos" phase,
    # so walk into its items rather than indexing the owner-list directly.
    assert [item["text"] for phase in builtin.TODO_STORE["e2e"] for item in phase["items"]] == [
        "read repo",
        "write code",
        "add email column",
    ]
    assert builtin.open_todos("e2e") == []

    # The final request carries no reminder: with nothing open there is nothing
    # to assert, and a stale one would lie about the list.
    assert reminders_in(stream, 4) == []

    await session.dispose()


@pytest.mark.asyncio
async def test_reminder_is_never_written_to_the_transcript(tmp_path) -> None:
    """Invisible to the user means invisible on disk too.

    A persisted reminder would replay into every later turn as a user message
    that the user never sent, and ``--resume`` would paint it in the
    transcript.
    """
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["ship it"]})},
            {"events": [StreamTextDelta(delta="Here is my plan.")]},
            {"events": todo_call("c2", {"op": "done", "items": ["ship it"]})},
            {"events": [StreamTextDelta(delta="Done.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("ship it")

    assert len(stream.requests) == 4, "the guardrail should have forced one continuation"
    assert len(reminders_in(stream, 2)) == 1, "the model must have been nudged"

    raw = (tmp_path / "sess" / "transcript.jsonl").read_text()
    assert "todo_reminder" not in raw
    assert "<system-reminder>" not in raw

    # Nor does it survive into a replayed history.
    assert not any(
        "<system-reminder>" in getattr(block, "text", "")
        for message in Transcript(tmp_path / "sess").build_llm_history()
        for block in getattr(message, "content", [])
    )

    await session.dispose()


@pytest.mark.asyncio
async def test_a_model_that_cannot_proceed_is_nudged_once_and_then_released(tmp_path) -> None:
    """The anti-trap. Two identical yields mean the model is stuck, usually
    waiting on a decision only the user can make. Nudging a stuck model again
    would spend the loop's continuation budget and delay the question it is
    trying to ask."""
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["pick an option"]})},
            {"events": [StreamTextDelta(delta="Which option do you want, A or B?")]},
            {"events": [StreamTextDelta(delta="I still need your answer to continue.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("start")

    assert len(stream.requests) == 3, "one nudge, then the turn is allowed to end"
    assert len(reminders_in(stream, 2)) == 1, "the single reminder is still in context"
    assert builtin.open_todos("e2e"), "the item is deliberately still open"

    await session.dispose()


@pytest.mark.asyncio
async def test_blocking_an_item_earns_one_reconciliation_nudge(tmp_path) -> None:
    """The honest escape hatch, now ACCOUNTED for instead of merely permitted.

    ``block`` is how a model stops with work outstanding without lying that it
    finished it — and it is MOVEMENT, so it earns exactly one nudge asking the
    model to say what it did with the parked item. That is the contract change:
    a blocked list used to earn no nudge at all, which is how a session parked
    six items and ended turns for ~14 hours. The turn still ends — after ONE
    accounting pass, not a loop — so the honest stop survives.
    """
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["get prod credentials"]})},
            {"events": [StreamTextDelta(delta="I need credentials for this.")]},
            {
                "events": todo_call(
                    "c2",
                    {
                        "op": "block",
                        "items": ["get prod credentials"],
                        "reason": "needs a credential only the user can provide",
                    },
                )
            },
            {"events": [StreamTextDelta(delta="Blocked on credentials — over to you.")]},
            # The accounting turn the reconciliation nudge buys: the model states
            # what the block waits on. Its yield is byte-identical to the last,
            # so the turn ends here rather than nudging again.
            {
                "events": [
                    StreamTextDelta(
                        delta=(
                            "Still blocked on the prod credential; nothing is watching it "
                            "until you provide it."
                        )
                    )
                ]
            },
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("deploy")

    assert len(stream.requests) == 5, "one reconciliation nudge, then released"
    assert reminders_in(stream, 3) == [], "the block itself is not yet accounted for"
    accounted = reminders_in(stream, 4)
    assert len(accounted) == 1, "the reconciliation nudge must reach the model"
    assert "get prod credentials" in accounted[0]
    assert "needs a credential only the user can provide" in accounted[0]
    assert "account for each blocked item" in accounted[0]
    # Pending is empty and the item is parked, not closed: blocked is reconciled
    # as blocked, never quietly treated as open work or as done.
    assert builtin.open_todos("e2e") == []
    assert [item["text"] for item in builtin.blocked_todos("e2e")] == ["get prod credentials"]
    assert builtin.TODO_STORE["e2e"][0]["items"][0]["status"] == "blocked"

    await session.dispose()


@pytest.mark.asyncio
async def test_restored_blocker_reminder_only_renders_string_reasons(tmp_path) -> None:
    session = make_session(tmp_path, SteeringStream([]))
    # Restoration can load JSON that bypassed the todo tool's schema.
    private = "synthetic-private-value"
    builtin.restore_todos(
        "e2e",
        [
            {
                "name": "Deploy",
                "items": [
                    {"text": "migrate", "status": "blocked", "reason": {"token": private}},
                    {"text": "publish", "status": "blocked", "reason": "waiting for approval"},
                ],
            }
        ],
    )
    try:
        (reminder,) = await session._todo_continuation()
        assert isinstance(reminder, CustomMessage)
        text = reminder.details["text"]
        assert private not in text
        assert "- migrate — blocked" in text.splitlines()
        assert "- publish — blocked: waiting for approval" in text.splitlines()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_accounted_blocked_set_lets_a_turn_finish_quietly(tmp_path) -> None:
    """THE over-fire repro, end to end: an accounted parked set must not be
    re-asserted on the next turn's first yield. Before the cross-turn latch,
    every fresh turn reset it and the first prose yield was overridden — the
    measured shape: ~97 of ~116 quiet-end attempts overridden across two
    sessions holding 5-6 blocked items and ZERO pending. Turn 2 here yields in
    plain prose, the same guard decision a quiet ``no_reply`` batch takes at
    the boundary (a scripted ``no_reply`` is expressible in this harness, but
    against a fresh USER prompt its quiet-end door refuses by design — "A
    person asked this turn" — so prose is the honest shape for this
    assertion)."""
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["get prod credentials"]})},
            {"events": [StreamTextDelta(delta="I need credentials for this.")]},
            {
                "events": todo_call(
                    "c2",
                    {
                        "op": "block",
                        "items": ["get prod credentials"],
                        "reason": "needs a credential only the user can provide",
                    },
                )
            },
            {"events": [StreamTextDelta(delta="Blocked on credentials — over to you.")]},
            # The accounting pass the reconciliation nudge buys.
            {
                "events": [
                    StreamTextDelta(
                        delta=(
                            "Still blocked on the prod credential; nothing is watching it "
                            "until you provide it."
                        )
                    )
                ]
            },
            # Second turn, fresh prompt: the set is already accounted, so this
            # prose yield must STAND — no new request, no new reminder. The old
            # per-turn re-arm bought one more of each here.
            {"events": [StreamTextDelta(delta="Nothing further from me.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("deploy")
    await session.prompt("thanks — anything else?")

    # Turn 1: five requests (the reconciliation pass included); turn 2: ONE.
    assert len(stream.requests) == 6, "turn 2 was re-asserted — the parked set re-fired"
    # Exactly two nudges ever fired — the pending one and the blocked one; the
    # fresh turn added none.
    assert len(reminders(session)) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_changed_blocker_reconciles_on_next_worked_turn(tmp_path) -> None:
    def block(call_id: str, reason: str) -> dict[str, Any]:
        return {
            "events": todo_call(call_id, {"op": "block", "items": ["deploy"], "reason": reason})
        }

    answer = {"events": [StreamTextDelta(delta="At the current blocker.")]}
    stream = SteeringStream(
        [
            block("c1", "database unavailable"),
            answer,
            block("c2", "database unavailable; rechecked"),
            answer,
            block("c3", "awaiting permission to schedule the migration"),
            answer,
            answer,
            answer,
        ]
    )
    builtin.TODO_STORE["e2e"] = [{"text": "deploy", "status": "pending"}]
    session = make_session(tmp_path, stream)
    try:
        await session.prompt("Try the deployment")
        assert len(stream.requests) == 4, "a reason rephrased during accounting cannot loop"
        await session.prompt("Recheck the deployment")
        assert len(stream.requests) == 7
        assert "awaiting permission to schedule the migration" in reminders_in(stream, 6)[0]
        await session.prompt("What else?")
        assert len(stream.requests) == 8, "nothing re-arms an unchanged hold"
    finally:
        await session.dispose()


def _quiet(call_id: str) -> dict[str, Any]:
    return {
        "events": [StreamToolCallDelta(index=0, id=call_id, name="no_reply", argument_delta="{}")]
    }


async def _peer_end(session, ends, expected_count, text="Synthetic peer check-in"):
    await session.receive_peer_message(
        text,
        wake=True,
        sender={"pid": 700042, "conversation_name": "synthetic"},
    )
    async with asyncio.timeout(5):
        while len(ends) < expected_count or session._turn_lock.locked():
            await asyncio.sleep(0.001)
    return ends[-1]


@pytest.mark.parametrize("quiet", [False, True], ids=["plain-stop", "no-reply"])
@pytest.mark.asyncio
async def test_unchanged_passive_stops_remind_once_without_public_outcome(tmp_path, quiet) -> None:
    answer = {"events": [StreamTextDelta(delta="The gate is still pending.")]}
    passive = [_quiet(f"q{i}") if quiet else answer for i in range(4)]
    stream = SteeringStream([*passive, answer, answer])
    session = make_session(tmp_path, stream)
    # Pin this normally deferred built-in for a declared-tool control.
    session.set_tool_deferral(pins={"no_reply"})
    builtin.TODO_STORE["e2e"] = [{"text": "run the gate", "status": "pending"}]
    ends: list[AgentEndEvent] = []
    events: list[AgentEvent] = []

    def observe(event: AgentEvent) -> None:
        events.append(event)
        if isinstance(event, AgentEndEvent):
            ends.append(event)

    session.subscribe(observe)
    try:
        for i, text in enumerate(
            (
                "Synthetic peer check-in. No authorized action; leave pending and use no_reply.",
                "Synthetic peer check-in. No change and no update; use no_reply.",
                "Synthetic peer check-in. Still nothing to report.",
            ),
            1,
        ):
            end = await _peer_end(session, ends, i, text)
            assert end.quiet_accepted is quiet and not end.notify
            assert not (end.error or end.cut_off or end.cut_off_cause or end.aborted)
            assert session.frontend_state.last_turn_outcome == "completed"
            assert not session.frontend_state.last_turn_cut_off
            assert len(stream.requests) == i + 1
        assert "no_reply" in [tool.name for tool in stream.requests[0].tools]
        assert not reminders_in(stream, 0)
        assert "run the gate" in reminders_in(stream, 1)[0]
        assert not any(reminders_in(stream, i) for i in (2, 3))
        assert [item["text"] for item in builtin.open_todos("e2e")] == ["run the gate"]

        await session.prompt("Please continue running the gate")
        assert len(stream.requests) == 6 and reminders_in(stream, 5)
        end = ends[-1]
        assert end.notify and not (end.error or end.cut_off or end.cut_off_cause or end.aborted)
        assert [item["text"] for item in builtin.open_todos("e2e")] == ["run the gate"]
        published = (tmp_path / "sess" / "transcript.jsonl").read_text()
        public_events = "\n".join(event.model_dump_json() for event in events)
        for private in ("todo_reminder", "<system-reminder>", "todo-incomplete", "Todo follow-up:"):
            assert private not in published and private not in public_events
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_live_wake_keeps_quiet_but_reused_handle_does_not(tmp_path) -> None:
    reason = "[wake:w1] Recheck whether the deployment queue drained"
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "block", "items": ["deploy"], "reason": reason})},
            {"events": [StreamTextDelta(delta="Waiting on the queue.")]},
            _quiet("q1"),
        ]
    )
    session = make_session(tmp_path, stream)
    ends = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    now = int(time.time() * 1000)
    wake = WakeSchedule(
        id="w1",
        message='For "deploy", recheck whether the deployment queue drained.',
        next_due_at=now + 600_000,
        created_at=now,
    )
    builtin.TODO_STORE["e2e"] = [{"text": "deploy", "status": "pending"}]
    try:
        await session.wake_scheduler.update([wake])
        await session.prompt("Check on deployment")
        assert len(stream.requests) == 2, "verified backing skips the model pass"
        binding = builtin.blocked_todos("e2e")[0]["block_binding"]
        assert binding.startswith(f"wake:w1:{now}:")
        end = await _peer_end(session, ends, 2)
        assert end.quiet_accepted and len(stream.requests) == 3
        # Reuse the exact purpose as well as the id. Only the incarnation
        # distinguishes this replacement from the receipt originally bound.
        await session.wake_scheduler.update([wake.model_copy(update={"created_at": now + 1})])
    finally:
        await session.dispose()

    # A cold restart must remember the old incarnation. The new w1 is real,
    # armed and owner-local, but it is not the schedule this item was tied to.
    builtin.TODO_STORE.clear()
    stream = SteeringStream([_quiet("q2"), _quiet("q3"), _quiet("q4")])
    restored = make_session(tmp_path, stream)
    restored.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    try:
        await asyncio.sleep(0)
        assert not stream.requests, "restoring the receipt must not start a turn or sweep"
        assert restored.wake_scheduler.armed_schedules[0].created_at == now + 1
        assert builtin.blocked_todos("e2e")[0]["block_binding"] == binding
        end = await _peer_end(restored, ends, 3)
        assert end.quiet_accepted and not end.cut_off_cause
        assert len(stream.requests) == 2
        assert "Recheck whether the deployment queue drained" in reminders_in(stream, 1)[0]
        assert "[wake:w1]" not in reminders_in(stream, 1)[0]
        end = await _peer_end(restored, ends, 4)
        assert end.quiet_accepted and len(stream.requests) == 3
        assert not reminders_in(stream, 2)
    finally:
        await restored.dispose()


@pytest.mark.asyncio
async def test_approval_refusal_does_not_retry_via_todo(tmp_path) -> None:
    refused = []

    async def deny(name, _summary):
        refused.append(name)
        return False

    stream = SteeringStream(
        [
            {
                "events": [
                    StreamToolCallDelta(
                        index=0, id="park", name="sessions", argument_delta='{"op":"park"}'
                    )
                ]
            },
            {"events": [StreamTextDelta(delta="The requested action was refused.")]},
        ]
    )
    session = make_session(tmp_path, stream)
    session.set_approval_handler(deny)
    session.refresh_tools(
        [*session._tools, *create_tools(session._build_tool_context(), ["sessions"])]
    )
    builtin.TODO_STORE["e2e"] = [{"text": "park", "status": "pending"}]
    try:
        await session.prompt("Park the session")
        assert refused == ["sessions"]
        assert len(stream.requests) == 2 and not reminders_in(stream, 1)
        assert len(builtin.open_todos("e2e")) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_accepted_self_park_does_not_resume_pending(tmp_path, monkeypatch) -> None:
    from local_operator.session.runtime import park_after_turn

    park_receipts = []
    monkeypatch.setattr(park_after_turn, "spawn", lambda *a: park_receipts.append(a) or 700043)
    stream = SteeringStream(
        [
            {
                "events": [
                    StreamToolCallDelta(
                        index=0, id="park", name="sessions", argument_delta='{"op":"park"}'
                    )
                ]
            },
            {"events": [StreamTextDelta(delta="Parking as requested.")]},
        ]
    )
    session = make_session(tmp_path, stream)
    session.refresh_tools(
        [*session._tools, *create_tools(session._build_tool_context(), ["sessions"])]
    )
    builtin.TODO_STORE["e2e"] = [{"text": "deploy", "status": "pending"}]
    ends = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    try:
        await session.prompt("Park this session and keep the todo pending.")
        assert len(park_receipts) == 1
        assert len(stream.requests) == 2 and not reminders_in(stream, 1)
        assert len(ends) == 1 and not ends[0].cut_off_cause
        assert len(builtin.open_todos("e2e")) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_failed_accounting_does_not_close_the_next_turn_check(tmp_path) -> None:
    answer = {"events": [StreamTextDelta(delta="Still blocked.")]}
    stream = SteeringStream([answer, {"error": "Synthetic provider error"}, answer, answer])
    session = make_session(tmp_path, stream)
    builtin.TODO_STORE["e2e"] = [
        {"text": "deploy", "status": "blocked", "reason": "Waiting for the owner"}
    ]
    try:
        await session.prompt("Check the deploy")
        assert len(stream.requests) == 2 and reminders_in(stream, 1)
        assert not session._todo_accounted
        await session.prompt("Check again")
        assert len(stream.requests) == 4 and reminders_in(stream, 3)
        assert session._todo_accounted
    finally:
        await session.dispose()


@pytest.mark.parametrize("relevant", [1, 2])
@pytest.mark.asyncio
async def test_phase_block_changes_every_open_row_but_only_binds_relevant_wakes(
    tmp_path, relevant: int
) -> None:
    tasks = ["deploy schema", "verify schema"]
    condition = "Await deployment approval"
    reason = f"[wake:gate] {condition}"
    plan = {"op": "init", "phases": [{"phase": "Release", "items": tasks}]}
    stream = SteeringStream(
        [
            {"events": todo_call("phase", plan)},
            {"events": todo_call("block", {"op": "block", "phase": "Release", "reason": reason})},
        ]
    )
    session = make_session(tmp_path, stream)
    now = int(time.time() * 1000)
    purpose = f"{condition}: " + ", ".join(f'"{task}"' for task in tasks[:relevant])
    try:
        await session.wake_scheduler.update(
            [WakeSchedule(id="gate", message=purpose, next_due_at=now + 60_000, created_at=now)]
        )
        await session.prompt("Park the whole Release phase")
        assert [row["text"] for row in builtin.blocked_todos("e2e")] == tasks
        assert all(row["reason"] == reason for row in builtin.blocked_todos("e2e"))
        covered = [key[1] for key, block in session._todo_state()[1].items() if block[-1]]
        assert covered == tasks[:relevant]
    finally:
        await session.dispose()


@pytest.mark.parametrize(
    ("tasks", "condition", "purpose", "expected_coverage"),
    [
        (
            ["deploy schema", "deploy schema to staging"],
            "Await deployment approval",
            "Await deployment approval for deploy schema to staging only; "
            "production is out of scope.",
            [],
        ),
        (
            ["deploy"],
            "Await another team to deploy first",
            "Infrastructure handoff: await another team to deploy first, "
            "then archive the queue report.",
            [],
        ),
        (
            # Repeating even a quoted title inside the blocker cannot make it independent.
            ["deploy", "deploy schema", "deploy schema to staging"],
            'Await another team to "deploy" first',
            "For “deploy schema to staging”, await another team to “deploy” first. "
            "Reminder: await another team to “deploy” first.",
            ["deploy schema to staging"],
        ),
        (
            ['Review "Send code" copy', "Send code"],
            "Await copy approval",
            'For “Review "Send code" copy”, await copy approval.',
            ['Review "Send code" copy'],
        ),
        (
            ["Review “Send code” copy", "Send code"],
            "Await copy approval",
            'For "Review “Send code” copy", await copy approval.',
            ["Review “Send code” copy"],
        ),
        (
            ['Review "Send code" copy', "Send code"],
            "Await copy approval",
            r'For "Review \"Send code\" copy", await copy approval.',
            ['Review "Send code" copy'],
        ),
        (
            ["Check print layout"],
            'Await the 3" sample',
            'Await the 3" sample for "Check print layout".',
            ["Check print layout"],
        ),
        (
            ["Send code", 'Review "Send code" copy'],
            "Await copy approval",
            'Await copy approval for “Review "Send code" copy in French”.',
            [],
        ),
        (
            ["Archive", "Send code"],
            "Await approval",
            'For "Archive "[draft]" / "Send code" copy", await approval.',
            [],
        ),
        (
            ["Send code"],
            'Await the 3" sample labelled "Send code"',
            'Await the 3" sample labelled "Send code". Await the 3" sample labelled "Send code".',
            [],
        ),
        (
            ["Send code"],
            'Await label "Send',
            'Await label "Send code" on tray. Await label "Send.',
            [],
        ),
    ],
    ids=[
        "unquoted-prefix",
        "only-in-condition",
        "exact-title-in-same-phase",
        "literal-straight-quotes",
        "literal-curly-quotes",
        "escaped-same-delimiter",
        "inch-mark-in-blocker",
        "nested-and-longer-title",
        "same-family-nesting-is-not-independent",
        "inch-mark-and-embedded-title-in-blocker",
        "title-cannot-start-in-blocker",
    ],
)
@pytest.mark.asyncio
async def test_phase_block_requires_independent_exact_task_title(
    tmp_path, tasks, condition, purpose, expected_coverage
) -> None:
    reply = {"events": [StreamTextDelta(delta="Waiting on the handoff.")]}
    stream = SteeringStream([])
    session = make_session(tmp_path, stream)
    try:
        context = session._build_tool_context()
        (wake_tool,) = create_tools(context, ["wake"])
        created = await wake_tool.execute(
            "create", {"op": "create", "message": purpose, "in": "10m"}, None, None, context
        )
        assert not created.is_error, created.text
        (wake,) = session.wake_scheduler.armed_schedules
        assert wake.message == purpose
        reason = f"[wake:{wake.id}] {condition}"
        stream.turns[:] = [
            {
                "events": todo_call(
                    "init", {"op": "init", "phases": [{"phase": "Release", "items": tasks}]}
                )
            },
            {"events": todo_call("block", {"op": "block", "phase": "Release", "reason": reason})},
            reply,
            reply,
        ]
        await session.prompt("Park the whole Release phase")
        needs_check = len(expected_coverage) != len(tasks)
        assert len(stream.requests) == (4 if needs_check else 3)
        reminders = reminders_in(stream, len(stream.requests) - 1)
        assert len(reminders) == int(needs_check), "only uncovered rows need a private check"
        assert [row["text"] for row in builtin.blocked_todos("e2e")] == tasks
        for task in tasks:
            was_reminded = any(f"- {task} — blocked: {condition}" in text for text in reminders)
            assert was_reminded == (task not in expected_coverage)
        covered = [key[1] for key, block in session._todo_state()[1].items() if block[-1]]
        assert covered == expected_coverage
    finally:
        await session.dispose()


@pytest.mark.parametrize("scope", ["relevant", "unrelated", "split questions"])
@pytest.mark.asyncio
async def test_actionable_ask_keeps_harness_metadata_off_public_surfaces(
    tmp_path, scope: str
) -> None:
    from local_operator.asks.render import mirror_card
    from local_operator.paths import config_dir
    from local_operator.tui.widgets.ask_queue import ask_rows

    task = "deploy database migration"
    condition = "Waiting for operator approval."
    question = {
        "relevant": f'{condition} May I proceed with "{task}"?',
        "unrelated": "May I replenish the office coffee?",
        "split questions": f'May I proceed with "{task}"?',
    }[scope]
    marker = "Todo follow-up:"
    stream = SteeringStream(
        [
            {"events": todo_call("init", {"op": "init", "items": [task]})},
            {"events": todo_call("park", {"op": "block", "items": [task], "reason": condition})},
            {"events": [StreamTextDelta(delta="Still waiting.")]},
            {},
            {},
        ]
    )
    session = make_session(tmp_path, stream)

    async def ask_handler(*_args, **_kwargs):
        raise AssertionError("queued Ask must not use the blocking transport")

    session.set_ask_handler(ask_handler)

    def ask_after_reminder():
        # Replay the documented producer path on the old prompt; with no copy
        # instruction the model sends only its authored question.
        (reminder,) = reminders_in(stream, 3)
        scope_line = next((line for line in reminder.splitlines() if line.startswith(marker)), "")
        text = f"{question}\n{scope_line}" if scope == "relevant" and scope_line else question
        texts = [text]
        if scope == "split questions":
            texts.append(f"No decision recorded: {condition} Should I keep waiting?")
        payload = {
            "questions": [
                {
                    "id": f"gate-{i}",
                    "question": text,
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
                for i, text in enumerate(texts)
            ],
            "timeout": 600,
        }
        stream.turns[3]["events"] = [
            StreamToolCallDelta(index=0, id="ask-gate", name="ask"),
            StreamToolCallDelta(index=0, argument_delta=json.dumps(payload)),
        ]

    def bind_ask():
        (row,) = session.ask_queue().projection()
        stream.turns[4]["events"] = todo_call(
            "bind", {"op": "block", "items": [task], "reason": f"[ask:{row['ask_id']}] {condition}"}
        )

    stream.turns[3]["steer"] = ask_after_reminder
    stream.turns[4]["steer"] = bind_ask
    try:
        await session.prompt("Check the deployment gate")
        rows = session.ask_queue().projection()
        (display,) = ask_rows(rows)
        card = mirror_card(rows)
        assert card is not None
        journal = (config_dir() / "sessions" / "e2e" / "asks.jsonl").read_text()
        assert (display.head_question, card["title"]) == (question, question)
        (queued,) = [
            event for line in journal.splitlines() if (event := json.loads(line)).get("questions")
        ]
        assert queued["questions"][0]["question"] == question
        for private in (marker, "<system-reminder>", "todo_reminder"):
            assert private not in journal
            assert private not in json.dumps(rows)
            assert private not in stream.turns[3]["events"][1].argument_delta
        assert next(iter(session._todo_state()[1].values()))[-1] == (scope == "relevant")
    finally:
        await session.dispose()


@pytest.mark.parametrize("kind", ["ask", "monitor", "wake"])
@pytest.mark.parametrize(
    "scope", ["relevant", "unrelated", "weak condition", "wrong item", "partial item"]
)
@pytest.mark.asyncio
async def test_only_actionable_followup_backs_a_block(tmp_path, kind: str, scope: str) -> None:
    turn = {"events": [StreamTextDelta(delta="Waiting, not finished.")]}
    stream = SteeringStream([])
    session = make_session(tmp_path, stream)
    now = int(time.time() * 1000)
    task = "deploy schema"
    condition = "Waiting on the deployment gate"
    purpose = {
        "relevant": 'For "DEPLOY   SCHEMA", keep waiting on the deployment gate.',
        "unrelated": "Check office coffee signage.",
        "weak condition": f'Check the gate for "{task}".',
        "wrong item": f'For "run tests": {condition}.',
        "partial item": f'For "re{task}": {condition}.',
    }[scope]
    withdraw: Callable[[], Any] | None = None
    try:
        if kind == "ask":

            async def ask_handler(*_args, **_kwargs):
                raise AssertionError("queued ask must not use the blocking transport")

            session.set_ask_handler(ask_handler)
            queue = session.ask_queue()
            outcome = queue.enqueue([{"id": "go", "question": purpose}], 600)
            assert outcome["ok"], outcome
            handle = outcome["details"]["ask_id"]
            assert (
                session._live_todo_receipt(kind, handle, outcome["details"]["expires_at"]) is None
            )

            def withdraw_ask():
                return queue.withdraw(handle, reason="moot")

            withdraw = withdraw_ask
        elif kind == "monitor":
            handle = "m1"
            session.monitor_scheduler.load(
                [
                    MonitorSpec(
                        id=handle,
                        name="Deployment gate",
                        description=purpose,
                        tool="todo",
                        arguments={"op": "view"},
                        every_ms=600_000,
                        created_at=now,
                    )
                ]
            )
        else:
            handle = "w1"
            await session.wake_scheduler.update(
                [
                    WakeSchedule(
                        id=handle, message=purpose, next_due_at=now + 600_000, created_at=now
                    )
                ]
            )
        assert session._live_todo_receipt(kind, handle, now) is not None
        reason = f"[{kind}:{handle}] {condition}"
        stream.turns[:] = [
            {"events": todo_call("block", {"op": "block", "items": [task], "reason": reason})},
            turn,
            turn,
            turn,
            turn,
        ]
        builtin.TODO_STORE["e2e"] = [{"text": task, "status": "pending"}]
        await session.prompt("Check on the deploy")
        if scope != "relevant":
            assert len(stream.requests) == 3 and condition in reminders_in(stream, 2)[0]
            assert "block_binding" not in builtin.blocked_todos("e2e")[0]
            assert reason not in reminders_in(stream, 2)[0]
            return
        assert len(stream.requests) == 2
        assert builtin.blocked_todos("e2e")[0]["block_binding"].startswith(f"{kind}:")
        # Withdraw without an answer notice or retire the producer. The
        # previously proved condition cannot retain a dead receipt's coverage.
        if withdraw is not None:
            assert withdraw()["ok"]
        elif kind == "monitor":
            assert (await session.monitor_scheduler.cancel(handle))["cancelled"] == handle
        else:
            await session.wake_scheduler.update([])
        await session.prompt("Check again")
        assert len(stream.requests) == 4 and condition in reminders_in(stream, 3)[0]
        await session.prompt("Unrelated prompt")
        assert len(stream.requests) == 5 and not reminders_in(stream, 4)
    finally:
        await session.dispose()


@pytest.mark.parametrize("kind", ["monitor", "wake"])
@pytest.mark.asyncio
async def test_retained_but_dormant_or_expired_receipt_no_longer_covers(tmp_path, kind) -> None:
    answer = {"events": [StreamTextDelta(delta="Still waiting.")]}
    handle = "next"
    condition = "Await the deployment gate"
    purpose = f'For "deploy", {condition.lower()}.'
    reason = f"[{kind}:{handle}] {condition}"
    stream = SteeringStream(
        [
            {"events": todo_call("link", {"op": "block", "items": ["deploy"], "reason": reason})},
            *([answer] * 4),
        ]
    )
    session = make_session(tmp_path, stream)
    now = int(time.time() * 1000)
    builtin.TODO_STORE["e2e"] = [{"text": "deploy", "status": "pending"}]
    wake = WakeSchedule(id=handle, message=purpose, next_due_at=now + 600_000, created_at=now)
    try:
        if kind == "monitor":
            session.monitor_scheduler.load(
                [
                    MonitorSpec(
                        id=handle,
                        name="Gate",
                        description=purpose,
                        tool="todo",
                        arguments={"op": "view"},
                        every_ms=600_000,
                        created_at=now,
                    )
                ]
            )
        else:
            await session.wake_scheduler.update([wake])
        await session.prompt("Check the gate")
        assert len(stream.requests) == 2 and not reminders_in(stream, 1)
        binding = builtin.blocked_todos("e2e")[0]["block_binding"]

        if kind == "monitor":
            entry = session.monitor_scheduler._entries[handle]
            entry.counters["next_due_at"] = None
            assert session.monitor_scheduler.runtime(handle) is not None
        else:
            # A retained row with no check possible before expiry isn't live
            # backing, even if it has yet to leave the scheduler's row list.
            await session.wake_scheduler.update([wake.model_copy(update={"until_at": now + 1})])
            assert session.wake_scheduler.armed_schedules[0].id == handle
        assert session._live_todo_receipt(kind, handle, now + 1) is None
        await session.prompt("Check again")
        assert len(stream.requests) == 4 and condition in reminders_in(stream, 3)[0]
        assert builtin.blocked_todos("e2e")[0]["block_binding"] == binding
        await session.prompt("Nothing changed")
        assert len(stream.requests) == 5 and not reminders_in(stream, 4)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_new_pending_work_still_fires_after_a_parked_set_is_accounted(tmp_path) -> None:
    """The direction the blocked latch must NOT close: with the parked set
    accounted, movement in the PENDING group still re-enters the turn. The next
    turn records new work, yields, and earns a nudge whose text carries the
    pending section only — the accounted parked set contributes no section."""
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["get prod credentials"]})},
            {"events": [StreamTextDelta(delta="I need credentials for this.")]},
            {
                "events": todo_call(
                    "c2",
                    {
                        "op": "block",
                        "items": ["get prod credentials"],
                        "reason": "needs a credential only the user can provide",
                    },
                )
            },
            {"events": [StreamTextDelta(delta="Blocked on credentials — over to you.")]},
            {
                "events": [
                    StreamTextDelta(
                        delta=(
                            "Still blocked on the prod credential; nothing is watching it "
                            "until you provide it."
                        )
                    )
                ]
            },
            # Turn 2: new work arrives and is recorded.
            {"events": todo_call("c3", {"op": "add", "items": ["write the follow-up"]})},
            # The yield after the add must be held: pending work is open work.
            {"events": [StreamTextDelta(delta="That's next on the list.")]},
            # Nudged, the model settles honestly.
            {"events": [StreamTextDelta(delta="Waiting on the credential before that.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("deploy")
    await session.prompt("one more thing: the follow-up")

    assert len(stream.requests) == 8
    nudge = reminders_in(stream, 7)
    assert len(nudge) == 1, "the new pending item must put the model back to work"
    assert "write the follow-up" in nudge[0]
    assert "Do not end the turn while pending items remain" in nudge[0]
    # The accounted parked set contributes NO section: blocked did not move.
    assert "account for each blocked item" not in nudge[0]
    assert "get prod credentials" not in nudge[0]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_late_steer_still_re_enters_a_quiet_turn(tmp_path) -> None:
    """The yield collector is untouched by the blocked latch — this pins it.
    With the parked set accounted (the guard silent), a steer typed while the
    accounting response is in flight must still re-enter the turn and be
    handled: the injections are the loop's, and a quiet guard is not a quiet
    session."""
    session: Session | None = None

    def steer_late() -> None:
        # Typed mid-request; the loop drains it at the yield boundary the guard
        # just declined.
        assert session is not None
        session.steer("also fold in the staging check")

    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["get prod credentials"]})},
            {"events": [StreamTextDelta(delta="I need credentials for this.")]},
            {
                "events": todo_call(
                    "c2",
                    {
                        "op": "block",
                        "items": ["get prod credentials"],
                        "reason": "needs a credential only the user can provide",
                    },
                )
            },
            {"events": [StreamTextDelta(delta="Blocked on credentials — over to you.")]},
            # The accounting pass — with the steer arriving mid-request.
            {
                "events": [
                    StreamTextDelta(
                        delta=(
                            "Still blocked on the prod credential; nothing is watching it "
                            "until you provide it."
                        )
                    )
                ],
                "steer": steer_late,
            },
            # The steer, not the guard, bought this request.
            {"events": [StreamTextDelta(delta="Noted — the staging check is on the list.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("deploy")

    assert len(stream.requests) == 6, "the late steer must re-enter a quiet turn"
    assert any("staging check" in text for text in stream.request_texts(5))
    # The guard itself did not: the two nudges of turn 1, and nothing new.
    assert len(reminders(session)) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_progress_earns_another_nudge(tmp_path) -> None:
    """Robustness in the other direction: the latch keys on the LIST, not on a
    once-per-turn counter, so a model that keeps closing items keeps being held
    to the rest of them."""
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["one", "two"]})},
            {"events": [StreamTextDelta(delta="First one is done I think.")]},
            {"events": todo_call("c2", {"op": "done", "items": ["one"]})},
            # Tries to stop again — but the list moved, so this earns a nudge.
            {"events": [StreamTextDelta(delta="That's the main part finished.")]},
            {"events": todo_call("c3", {"op": "done", "items": ["two"]})},
            {"events": [StreamTextDelta(delta="Both done.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("do both")

    assert len(stream.requests) == 6, "two separate nudges, one per plateau"
    assert len(reminders_in(stream, 4)) == 1, "only the newest reminder is rendered"
    assert builtin.open_todos("e2e") == []

    await session.dispose()


@pytest.mark.asyncio
async def test_a_fresh_user_turn_re_arms_the_guardrail(tmp_path) -> None:
    """The latch is per user turn. A user who says "carry on" after a released
    turn must get the guardrail back, or the second turn ends early for free."""
    stream = SteeringStream(
        [
            {"events": todo_call("c1", {"op": "init", "items": ["the work"]})},
            {"events": [StreamTextDelta(delta="Stopping here.")]},
            {"events": [StreamTextDelta(delta="Still stopping.")]},
            # Second user turn, same unchanged list.
            {"events": [StreamTextDelta(delta="Okay, stopping again.")]},
            {"events": todo_call("c2", {"op": "done", "items": ["the work"]})},
            {"events": [StreamTextDelta(delta="Finished.")]},
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("start")
    assert len(stream.requests) == 3

    await session.prompt("carry on")
    assert len(stream.requests) == 6, "the new turn must be nudged despite the same list"
    assert builtin.open_todos("e2e") == []

    await session.dispose()


@pytest.mark.asyncio
async def test_unseen_blocker_rewrite_and_quiet_are_not_disposition(tmp_path) -> None:
    answer = {"events": [StreamTextDelta(delta="Waiting for the database.")]}
    rewrite = {
        "events": todo_call(
            "rewrite",
            {
                "op": "block",
                "items": ["deploy"],
                "reason": "Waiting on production approval",
            },
        )
    }
    stream = SteeringStream([answer, answer, rewrite, *(_quiet(f"q{i}") for i in range(6))])
    session = make_session(tmp_path, stream)
    builtin.TODO_STORE["e2e"] = [
        {"text": "deploy", "status": "blocked", "reason": "Waiting for the database"}
    ]
    ends: list[AgentEndEvent] = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    try:
        await session.prompt("Check the database")
        end = await _peer_end(session, ends, 2)
        assert end.quiet_accepted and not end.cut_off_cause
        assert "Waiting on production approval" in reminders_in(stream, 4)[0]
        end = await _peer_end(session, ends, 3)
        assert end.quiet_accepted and len(stream.requests) == 6
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_duplicate_blocked_rows_use_the_exact_prior_condition_before_mutation(tmp_path):
    first_reason = "[wake:w1] Await source"
    second_reason = "[wake:w2] Await destination"
    stream = SteeringStream(
        [
            {"events": [StreamTextDelta(delta="Neither deployment is ready.")]},
            {
                "events": todo_call(
                    "bind-second", {"op": "block", "items": ["deploy"], "reason": second_reason}
                )
            },
            {"events": [StreamTextDelta(delta="Waiting on the destination.")]},
        ]
    )
    session = make_session(tmp_path, stream)
    now = int(time.time() * 1000)

    def wake(handle: str, message: str) -> WakeSchedule:
        return WakeSchedule(id=handle, message=message, next_due_at=now + 600_000, created_at=now)

    async def block(reason: str, *items: str):
        return await builtin.execute_todo(
            "setup",
            {"op": "block", "items": list(items or ("deploy",)), "reason": reason},
            None,
            None,
            session._build_tool_context(),
        )

    source = wake("w1", 'For "deploy", await source and await destination.')
    destination = wake("w2", 'For "deploy", await destination.')
    try:
        builtin.TODO_STORE["e2e"] = [
            {"text": "deploy", "status": "blocked", "reason": "Await source"},
            {"text": "deploy", "status": "blocked", "reason": "Await destination"},
        ]
        await session.wake_scheduler.update([source])
        assert not (await block(first_reason)).is_error
        first, second = builtin.blocked_todos("e2e")
        assert first["block_binding"].startswith("wake:w1:") and "block_binding" not in second
        # Let the reminder show the exact blocker before its follow-up receipt is
        # created; the model then repeats the documented block operation.
        original = session._todo_continuation

        async def arm_destination():
            offer = await original()
            if offer:
                await session.wake_scheduler.update([source, destination])
            return offer

        session._todo_continuation = arm_destination
        await session.prompt("Check both deployments")
        (sent,) = reminders_in(stream, 1)
        assert "deploy — blocked: Await destination" in sent and "Await source" not in sent
        assert len(stream.requests) == 3
        current_first, current_second = builtin.blocked_todos("e2e")
        assert current_first == first and current_second["reason"] == second_reason
        assert current_second["block_binding"].startswith("wake:w2:")

        reused = destination.model_copy(update={"created_at": now + 1})
        await session.wake_scheduler.update([source, reused])
        for reason in (second_reason, "[wake:w1] Await both", "[wake:w4] Stale"):
            assert (await block(reason)).is_error
            assert builtin.blocked_todos("e2e") == [current_first, current_second]

        # The phase can replace both reasons; a prior binding on the first row
        # must not transfer to the second condition just because its wake names both.
        phase_reason = "[wake:w1] Await destination"
        result = await builtin.execute_todo(
            "phase",
            {"op": "block", "phase": "Todos", "reason": phase_reason},
            None,
            None,
            session._build_tool_context(),
        )
        assert not result.is_error
        assert [block[-1] for block in session._todo_state()[1].values()] == [False, True]
        assert all(row["reason"] == phase_reason for row in builtin.blocked_todos("e2e"))

        # Both deploy rows now have the same condition. Reject an explicit
        # duplicate selector without even partially mutating another named item.
        await builtin.execute_todo(
            "add", {"op": "add", "items": ["verify"]}, None, None, session._build_tool_context()
        )
        before = builtin.todo_snapshot("e2e")
        assert (await block(phase_reason, "verify", "deploy")).is_error
        assert builtin.todo_snapshot("e2e") == before
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_reordered_pending_membership_does_not_reprompt(tmp_path) -> None:
    items = [{"text": text, "status": "pending"} for text in ("alpha", "beta", "delta")]
    answer = [StreamTextDelta(delta="Not finished.")]
    stream = SteeringStream([{"events": answer}, {"events": answer, "steer": items.reverse}])
    session = make_session(tmp_path, stream)
    builtin.TODO_STORE["e2e"] = items
    try:
        await session.prompt("Try the gates")
        assert len(stream.requests) == 2 and len(reminders_in(stream, 1)) == 1
        assert session._last_turn_outcome == "completed"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_render_trims_only_the_expired_offer(tmp_path, monkeypatch) -> None:
    answer = {"events": [StreamTextDelta(delta="Not finished.")]}
    stream = SteeringStream([answer, answer])
    session = make_session(tmp_path, stream)
    pending = {"text": "deploy", "status": "pending"}
    blocked = {"text": "backup", "status": "blocked", "reason": "Waiting for storage"}
    builtin.TODO_STORE["e2e"] = [pending, blocked]
    original = session._todo_continuation

    async def settle_backup_after_offer() -> list[AgentMessage]:
        reminders = await original()
        if reminders:
            blocked["status"] = "done"
        return reminders

    monkeypatch.setattr(session, "_todo_continuation", settle_backup_after_offer)
    try:
        await session.prompt("Try the deployment and backup")
        assert len(stream.requests) == 2
        (sent,) = reminders_in(stream, 1)
        assert "deploy" in sent and "backup" not in sent
        assert session._todo_pending_seen == {(("Todos", "deploy", 1),)}
        assert not session._todo_accounted, "the trimmed offer did not carry the backup condition"
    finally:
        await session.dispose()


@pytest.mark.parametrize("reason", ["abort", "discard"])
@pytest.mark.asyncio
async def test_unsettled_held_end_does_not_commit_private_assessment(
    tmp_path, monkeypatch, reason: str
) -> None:
    from local_operator.session import session as session_module

    answer = {"events": [StreamTextDelta(delta="The deployment remains blocked.")]}
    stream = SteeringStream([answer] * 4)
    session = make_session(tmp_path, stream)
    builtin.TODO_STORE["e2e"] = [
        {"text": "deploy", "status": "blocked", "reason": "Waiting for owner approval"}
    ]
    original = session._maybe_compact

    async def supersede_held_end() -> None:
        if reason == "abort":
            session.abort()
        else:
            session._continuation_queue.append(Message.user("synthetic required continuation"))

    if reason == "discard":
        monkeypatch.setattr(session_module, "_MAX_CONTINUATIONS", 0)
    monkeypatch.setattr(session, "_maybe_compact", supersede_held_end)
    try:
        await session.prompt("Try the deployment")
        assert len(stream.requests) == 2 and not session._todo_accounted
        monkeypatch.setattr(session, "_maybe_compact", original)
        await session.prompt("Check the hold again")
        assert len(stream.requests) == 4 and reminders_in(stream, 3)
        assert session._todo_accounted
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_quiet_after_approval_refusal_does_not_trigger_private_retry(tmp_path) -> None:
    park = {
        "events": [
            StreamToolCallDelta(
                index=0,
                id="park",
                name="sessions",
                argument_delta='{"op":"park"}',
            )
        ]
    }
    stream = SteeringStream([park, _quiet("q")])
    session = make_session(tmp_path, stream)
    denied = []

    async def deny(name, _summary):
        denied.append(name)
        return False

    session.set_approval_handler(deny)
    session.refresh_tools(
        [*session._tools, *create_tools(session._build_tool_context(), ["sessions"])]
    )
    builtin.TODO_STORE["e2e"] = [{"text": "deploy", "status": "pending"}]
    ends: list[AgentEndEvent] = []
    session.subscribe(
        lambda event: ends.append(event) if isinstance(event, AgentEndEvent) else None
    )
    try:
        end = await _peer_end(session, ends, 1)
        assert denied == ["sessions"] and len(stream.requests) == 2
        assert not reminders_in(stream, 1)
        assert not end.cut_off_cause and not session._todo_accounted_pending
    finally:
        await session.dispose()


@pytest.mark.parametrize("end_kind", ["normal", "abort", "discard"])
@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.asyncio
async def test_private_todo_check_preserves_print_contract(
    tmp_path, capsys, monkeypatch, contract, end_kind
) -> None:
    from local_operator.headless_print import run_print_mode
    from local_operator.output_contract import OutputContract
    from local_operator.session import session as session_module

    reply = 'The gate is pending.\n```json\n{"status":"pending"}\n```'
    answer = {"events": [StreamTextDelta(delta=reply)]}
    stream = SteeringStream([answer, answer])
    session = make_session(tmp_path, stream)
    builtin.TODO_STORE["e2e"] = [{"text": "run the gate", "status": "pending"}]
    if contract:
        session.set_output_contract(OutputContract(format="json"))
    validation: list[OutputValidationEvent] = []
    session.subscribe(
        lambda event: validation.append(event) if isinstance(event, OutputValidationEvent) else None
    )

    async def post_loop() -> None:
        if end_kind == "abort":
            session.abort()
        elif end_kind == "discard":
            session._continuation_queue.append(Message.user("synthetic required continuation"))

    monkeypatch.setattr(session_module, "_MAX_CONTINUATIONS", 0)
    monkeypatch.setattr(session, "_maybe_compact", post_loop)
    # Pristine Session reports 0 even when abort lands after the loop has
    # produced its terminal event. Keep that separate print-host contract.
    assert await run_print_mode(session, ["Run the gate"]) == 0
    out, err = capsys.readouterr()
    assert out == ('{"status":"pending"}\n' if contract else reply + "\n")
    assert [event.ok for event in validation] == ([True] if contract else [])
    assert "todo-incomplete" not in err and "<system-reminder>" not in out + err
    assert len(stream.requests) == 2 and reminders_in(stream, 1)
    assert bool(session._todo_accounted_pending) is (end_kind == "normal")
