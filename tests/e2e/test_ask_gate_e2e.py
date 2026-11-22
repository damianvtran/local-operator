"""End-to-end: queued ask admission over a real session, queue and transcript.

The provider is scripted, but each call enters through ``Session.prompt`` and
uses the production tool, queue, and transcript paths. A legacy clearance
request would still receive a valid ``clear`` verdict, so the first-call test
can distinguish a diverted receipt from a visible queued receipt.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.harness.rows import is_ask_gate_divert_row
from local_operator.harness.types import AbortSignal, ChatRequest, StreamEvent
from local_operator.session.frontend_state import FrontendUpdate
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from tests.e2e.harness import (
    ScriptedStream,
    build_session,
    dispose_quietly,
    text_turn,
    tool_call_turn,
)
from tests.e2e.watchdog import bounded

pytestmark = pytest.mark.e2e

BOUND_S = 90.0
VERDICT_CLEAR = "VERDICT: clear\nREASON: the recommendation is plainly best."
VERDICT_RAISE = "VERDICT: raise\nREASON: the brand name is the operator's to state."


def _ask_args(question: str = "Deploy now?") -> dict[str, Any]:
    return {
        "questions": [
            {
                "id": "q0",
                "question": question,
                "options": [
                    {"label": "Ship it", "description": "after the freeze"},
                    {"label": "Wait", "description": "until Monday"},
                ],
                "recommended": 0,
            }
        ]
    }


def _session(config_dir: Path, name: str, stream: ScriptedStream) -> Session:
    return build_session(config_dir / "sessions" / name, stream, cwd=config_dir)


async def _hook_never_awaited(questions: list[Any]) -> dict[str, list[str]] | None:
    """The queued arm's host hook: present (it is the mode's second condition),
    but the queue owns admission and must not await the blocking hook."""
    return None


def _tool_texts(request: Any) -> list[str]:
    return [m.text for m in request.messages if getattr(m, "role", "") == "tool"]


def _ask_turn(call_id: str) -> list[StreamEvent]:
    return tool_call_turn(
        text="Asking.", tool_name="ask", tool_call_id=call_id, arguments=_ask_args()
    )


class _AdmissionStream(ScriptedStream):
    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        super().__init__(turns)
        self.clearance = ScriptedStream([text_turn(VERDICT_CLEAR)])

    def __call__(
        self, request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        # A regressed fork should hide the ask, not exhaust the ordinary tape.
        if request.purpose == "clearance":
            return self.clearance(request, signal)
        return super().__call__(request, signal)


def _tool_result(session: Session, call_id: str) -> dict[str, Any]:
    results = [
        entry.payload
        for entry in session.transcript.entries()
        if entry.payload.get("role") == "tool" and entry.payload.get("tool_call_id") == call_id
    ]
    assert len(results) == 1, "each ask call must leave exactly one tool result"
    return results[0]


@pytest.mark.asyncio
async def test_first_ask_queues_unchanged_with_a_visible_transcript_receipt(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(policy, "ASK_GATE", True)
    stream = _AdmissionStream([_ask_turn("ask-1"), text_turn("Waiting on the operator.")])
    session = _session(headless_tui_env, "first-ask", stream)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(headless_tui_env), install_ask_surface=True
    )
    desktop_updates: list[FrontendUpdate] = []
    phone_asks: list[str] = []
    desktop_sub = await handle.subscribe_frontend(desktop_updates.append)
    unsub_phone = handle.subscribe(
        lambda: phone_asks.extend(row.ask_id for row in handle.session_projection_seed.asks or [])
    )
    try:
        with bounded(BOUND_S, "first ask queues without diversion"):
            await session.prompt("deploy it")

        result = _tool_result(session, "ask-1")
        # Replay hides both the call and its result from this stored marker.
        # Check it before queue/provider counts: on base the call was hidden.
        assert not is_ask_gate_divert_row({"type": "message", "payload": result}), result
        calls = [
            call
            for entry in session.transcript.entries()
            for call in entry.payload.get("tool_calls", [])
            if call["id"] == "ask-1"
        ]
        assert len(calls) == 1
        assert calls[0]["name"] == "ask"
        assert calls[0]["arguments"] == _ask_args()

        (card,) = session.ask_queue().open_records()
        assert card["tool_call_id"] == "ask-1"
        (question,) = card["questions"]
        for key, value in _ask_args()["questions"][0].items():
            assert question[key] == value
        events = store.read_events(session.transcript.directory)
        assert [(event["kind"], event["ask_id"]) for event in events] == [
            (store.EVENT_QUEUED, card["ask_id"])
        ]

        assert not result.get("is_error")
        details = result["provider_payload"]["details"]
        assert details["ask_id"] == card["ask_id"]
        assert details["status"] == "queued"
        assert "answers" not in details and "ask_gate" not in details
        assert any(
            row["ask_id"] == card["ask_id"] and row["status"] == store.STATUS_OPEN
            for update in desktop_updates
            for row in update.changes.get("asks") or []
        )
        assert card["ask_id"] in phone_asks
        assert stream.clearance.requests == []
        assert len(stream.requests) == 2
        (receipt,) = _tool_texts(stream.requests[-1])
        assert card["ask_id"] in receipt and "queued" in receipt.lower()
        assert all(fragment not in receipt for fragment in ("[Ask clearance]", "Ship it", "Wait"))
        assert stream.exhausted_at is None
    finally:
        desktop_sub.unsubscribe()
        unsub_phone()
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# raise — today's queue path, byte-for-byte
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raise_verdict_queues_untouched_and_the_receipt_reaches_the_model(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Asking.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args(),
            ),
            text_turn(VERDICT_RAISE),
            text_turn("Waiting on the operator."),
        ]
    )
    session = _session(headless_tui_env, "gate-raise", stream)
    session.set_ask_handler(_hook_never_awaited)
    try:
        with bounded(BOUND_S, "raise verdict queues"):
            await session.prompt("deploy it")

        directory = session.transcript.directory
        ask_ids = store.ask_ids(store.read_events(directory))
        assert len(ask_ids) == 1, "the raise must queue exactly once"
        assert len(session.ask_queue().open_records()) == 1

        # The model got TODAY'S receipt, not a note: nothing was diverted.
        texts = _tool_texts(stream.requests[-1])
        assert any("queued" in text and "[Ask clearance]" not in text for text in texts), texts
        assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


# ---------------------------------------------------------------------------
# the kill switch — the fork is NEVER called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_off_never_calls_the_fork_and_the_ask_still_queues(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(policy, "ASK_GATE", False)
    stream = ScriptedStream(
        [
            tool_call_turn(
                text="Asking.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args(),
            ),
            text_turn("Queued."),
        ]
    )
    session = _session(headless_tui_env, "gate-off", stream)
    session.set_ask_handler(_hook_never_awaited)
    try:
        with bounded(BOUND_S, "gate off queues without a fork"):
            await session.prompt("deploy it")

        # TWO provider calls total: the ask turn and the final turn. A fork
        # would be a third — this is the provider-call-counter assertion the
        # scenario calls for.
        assert len(stream.requests) == 2, [r.purpose for r in stream.requests]
        assert all(r.purpose != "clearance" for r in stream.requests)
        ask_ids = store.ask_ids(store.read_events(session.transcript.directory))
        assert len(ask_ids) == 1
        assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)


@pytest.mark.asyncio
async def test_identical_reask_is_refused_while_the_first_card_is_open(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(policy, "ASK_GATE", True)
    stream = _AdmissionStream(
        [
            _ask_turn("ask-1"),
            text_turn("Waiting on the operator."),
            _ask_turn("ask-2"),
            text_turn("Still waiting."),
        ]
    )
    session = _session(headless_tui_env, "reask", stream)
    session.set_ask_handler(_hook_never_awaited)
    try:
        with bounded(BOUND_S, "admit the first ask"):
            await session.prompt("deploy it")

        first = _tool_result(session, "ask-1")
        assert not is_ask_gate_divert_row({"type": "message", "payload": first}), first
        (card,) = session.ask_queue().open_records()
        assert card["tool_call_id"] == "ask-1"
        directory = session.transcript.directory
        before = store.read_events(directory)
        assert [event["kind"] for event in before] == [store.EVENT_QUEUED]

        # The first prompt has returned and the real card is still open, so
        # the second prompt must refuse the duplicate, not admit it in its place.
        assert len(stream.requests) == 2
        with bounded(BOUND_S, "refuse the identical re-ask"):
            await session.prompt("check the deployment decision again")

        assert _tool_result(session, "ask-2")["is_error"] is True
        texts = _tool_texts(stream.requests[-1])
        refusal = "an open ask already carries that question text"
        assert any(refusal in text for text in texts), texts
        assert session.ask_queue().open_records() == [card]
        assert store.read_events(directory) == before
        assert len(stream.requests) == 4
        assert stream.clearance.requests == []
        assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)
