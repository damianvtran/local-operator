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

import json
from collections.abc import Callable
from typing import Any

import pytest

from local_operator.harness.message_types import TODO_REMINDER_MESSAGE_TYPE
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    CustomMessage,
    ModelSpec,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
)
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
    assert session._blocked_reminder_fingerprint == (("Todos", "get prod credentials", "blocked"),)
    await session.prompt("thanks — anything else?")

    # Turn 1: five requests (the reconciliation pass included); turn 2: ONE.
    assert len(stream.requests) == 6, "turn 2 was re-asserted — the parked set re-fired"
    # Exactly two nudges ever fired — the pending one and the blocked one; the
    # fresh turn added none.
    assert len(reminders(session)) == 2
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
