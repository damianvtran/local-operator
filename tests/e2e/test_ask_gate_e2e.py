"""End-to-end: the ask gate over a REAL session, queue, transcript and provider.

Why this file exists
--------------------

The unit matrix pins the gate's parts; these cells pin the ASSEMBLED path the
operator named in the brief — "a forked, hidden check runs before the ask
reaches them" — with the real ``ask`` tool, the real queued engine, a real
transcript and a scripted provider answering the fork. Each cell drives one of
the three verdicts and reads the OBSERVABLE each one must leave (or not leave):

* **clear** — nothing queues, nothing reaches ``asks.jsonl``, and the model's
  next request carries the decision-point note (the transcript keeps it).
* **raise** — the ask reaches the queue untouched, exactly one row, and the
  model's next request carries today's receipt.
* **kill switch** — ``LOP_ASK_GATE`` off: the fork is NEVER called (asserted on
  the provider-call count, not on an absence of text) and the ask queues.
* **honor rule** — a re-raise of the same content reaches the queue WITHOUT a
  second fork (again on the call count).

The stream is scripted (one canned turn per model call, the fork included —
it is a real provider call through the session's own stream fn) for the reason
``test_ask_queue_e2e`` gives: the failure signal is "the fork never ran / ran
twice", and live latency would only make that signal slower to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, store
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
    but the gate and the queue mean it must never be awaited."""
    return None


def _tool_texts(request: Any) -> list[str]:
    return [m.text for m in request.messages if getattr(m, "role", "") == "tool"]


# ---------------------------------------------------------------------------
# clear — nothing reaches the user, the note reaches the model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_clear_verdict_diverts_with_no_queue_row_and_the_note_reaches_the_model(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            # 1: the model asks.
            tool_call_turn(
                text="Asking.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args(),
            ),
            # 2: the fork's OWN request, answered by the scripted provider.
            text_turn(VERDICT_CLEAR),
            # 3: the model, having read the note, carries on.
            text_turn("Proceeding with Ship it."),
        ]
    )
    session = _session(headless_tui_env, "gate-clear", stream)
    session.set_ask_handler(_hook_never_awaited)
    try:
        with bounded(BOUND_S, "clear verdict diverts"):
            await session.prompt("deploy it")

        directory = session.transcript.directory
        # NOTHING queued: no log, no open record, no events.
        assert not store.asks_log_path(directory).exists()
        assert store.ask_ids(store.read_events(directory)) == []
        assert session.ask_queue().open_records() == []

        # The residual call/result pair IS in the transcript, marker included —
        # that is what lets the human surfaces filter it owner-side.
        entries = session.transcript.entries()
        results = [
            entry
            for entry in entries
            if entry.payload.get("role") == "tool" and entry.payload.get("tool_call_id") == "ask-1"
        ]
        assert len(results) == 1, "exactly one result row for the diverted call"
        details = (results[0].payload.get("provider_payload") or {}).get("details") or {}
        assert details.get("ask_gate", {}).get("hidden") is True

        # The MODEL saw the note: the next request carries it as the tool result.
        # (Request #2 was the fork itself — purpose "clearance" — recorded by the
        # session's own stream fn, which is what keeps the cache lineage.)
        assert [r.purpose for r in stream.requests][1] == "clearance"
        texts = _tool_texts(stream.requests[-1])
        assert any("[Ask clearance]" in text for text in texts), texts
        assert stream.exhausted_at is None
    finally:
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


# ---------------------------------------------------------------------------
# the honor rule — a re-raise of the same content skips the fork
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reraise_of_the_same_content_reaches_the_queue_without_a_second_fork(
    headless_tui_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream(
        [
            # 1: the model asks; 2: the fork answers clear; 3: the model asks
            # AGAIN with the same content (allowed — the note licenses it);
            # 4: the final answer. NO turn 5: a second fork would consume the
            # script's tail and the request count assertion below would fail.
            tool_call_turn(
                text="Asking.",
                tool_name="ask",
                tool_call_id="ask-1",
                arguments=_ask_args(),
            ),
            text_turn(VERDICT_CLEAR),
            tool_call_turn(
                text="Actually, this one is theirs.",
                tool_name="ask",
                tool_call_id="ask-2",
                arguments=_ask_args(),
            ),
            text_turn("Waiting."),
        ]
    )
    session = _session(headless_tui_env, "gate-reraise", stream)
    session.set_ask_handler(_hook_never_awaited)
    try:
        with bounded(BOUND_S, "re-raise skips the fork"):
            await session.prompt("deploy it")

        ask_ids = store.ask_ids(store.read_events(session.transcript.directory))
        assert len(ask_ids) == 1, "the re-raise is the one that queued"
        # Four provider calls: ask, fork, re-raise-ask, final. Five would mean
        # a second fork fired for identical content.
        assert len(stream.requests) == 4, [r.purpose for r in stream.requests]
        assert stream.exhausted_at is None
    finally:
        await dispose_quietly(session)
