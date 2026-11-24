"""The display-only row an AGENT withdrawal writes (PR B of the ask-id work).

WHY THIS FILE EXISTS. ``ask_withdraw`` with reason ``moot`` used to leave no
trace a person could read: the fold settled the ask, every list dropped it, and
the transcript said nothing about the question having existed or being gone.
Ben, 2026-11-23: a withdrawn ask gets ``a-xxxx withdrawn by the agent`` — a row
for the operator, NOT sent to the model.

The properties that make that safe, each pinned here rather than left to review:

- a SEPARATE type (``ask_withdrawn``), not ``ask_response`` with a new status:
  the response type is a model-visible carrier whose reconcile buys a turn, and
  a status check is something every one of those paths has to remember;
- the row never reaches the model, checked through the REAL renderer and the
  REAL provider request of a later turn;
- the row id IS the delivery marker, so a replay or a re-plan writes one row;
- ``answered_in_chat`` is not a withdrawal and writes none of this;
- a secret ask is NAMED generically — its question is never quoted in the row;
- the TUI and the phone both paint it, through the shared receipt.

The session shape (``_make_session``/``_install_queue``/``_ask``) comes from
``test_open_asks_section`` rather than being re-spelled here: one session
harness for the whole ask suite, as ``test_dismissed_row`` does with
``test_queue``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import policy, store
from local_operator.harness.message_types import (
    ASK_WITHDRAWN_MESSAGE_TYPE,
    MODEL_VISIBLE_INJECTED_TYPES,
)
from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.rows import ask_name, ask_withdrawn_notice
from local_operator.harness.types import CustomMessage
from local_operator.session.goal import GoalState
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.session.test_open_asks_section import (
    ScriptedStream,
    _ask,
    _install_queue,
    _make_session,
)

ASK_TEXT = "Should the release go out tonight?"


async def _session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Session, Any, ScriptedStream]:
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    stream = ScriptedStream()
    session = _make_session(tmp_path / "sess", stream, GoalState())
    return session, _install_queue(session, monkeypatch), stream


async def _wait_entry(session: Session, entry_id: str, timeout: float = 10.0) -> bool:
    """Wait for the row's durable write (the append runs in the background group)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if session._transcript.has_entry(entry_id):
            return True
        await asyncio.sleep(0.02)
    return False


def _custom_rows(session: Session, custom_type: str) -> list[Any]:
    """Every durable row of one custom type — the honest count, not a set."""
    return [
        entry
        for entry in session._transcript.entries()
        if entry.payload.get("custom_type") == custom_type
    ]


def _texts(messages: list[Any]) -> list[str]:
    out: list[str] = []
    for message in messages:
        for block in getattr(message, "content", ()) or ():
            text = getattr(block, "text", None)
            if text:
                out.append(str(text))
    return out


# ---------------------------------------------------------------------------
# The write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_moot_withdraw_writes_one_display_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row says what was withdrawn, by whom, and in the shared receipt's words."""
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        outcome = session.withdraw_ask(ask_id, reason="moot")
        assert outcome["ok"], outcome
        row_id = store.withdrawn_row_id(ask_id)
        assert await _wait_entry(session, row_id), "the display row was never written"

        rows = _custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE)
        assert len(rows) == 1
        details = rows[0].payload["details"]
        assert rows[0].id == row_id
        assert rows[0].payload.get("attribution") == "agent"
        assert details["ask_id"] == ask_id
        assert details["status"] == "withdrawn"
        assert details["by"] == "agent"
        assert isinstance(details["at"], int) and details["at"] > 0
        # The questions ride the row so a surface can expand it — the response
        # row's own contract — and the text is the shared receipt, id first,
        # with the name fitted the way PR A fits it (a handle, not a summary).
        assert [q["question"] for q in details["questions"]] == [ASK_TEXT]
        assert details["text"] == ask_withdrawn_notice(details)[0]
        assert details["text"] == f'{ask_id} · "{ask_name(details)}" — Withdrawn by the agent'
        assert details["text"].startswith(f'{ask_id} · "')
        assert details["text"].endswith('" — Withdrawn by the agent')
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_row_id_is_the_marker_across_a_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay re-reads the journal but must not re-plan a withdrawal.

    Two writes are attempted after the first: the same recorder called again in
    this runtime, and a fresh ``Transcript`` over the same directory (which is
    what a restart has) with the recorder called once more. One row, not three.
    """
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        assert session.withdraw_ask(ask_id, reason="moot")["ok"]
        row_id = store.withdrawn_row_id(ask_id)
        assert await _wait_entry(session, row_id)

        session._record_ask_withdrawn(ask_id, [{"id": "ship", "question": ASK_TEXT}])
        await asyncio.sleep(0.05)
        assert len(_custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE)) == 1
        assert session._transcript.has_entry(row_id)
    finally:
        await session.dispose()

    replayed = Transcript(tmp_path / "sess")
    assert replayed.has_entry(row_id), "the durable id is the marker"
    assert (
        len(
            [
                entry
                for entry in replayed.entries()
                if entry.payload.get("custom_type") == ASK_WITHDRAWN_MESSAGE_TYPE
            ]
        )
        == 1
    )


@pytest.mark.asyncio
async def test_answered_in_chat_writes_no_withdrawn_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``answered_in_chat`` records an ANSWER — its receipt is the response row.

    The queue routes it through the ``answered`` machinery, so a "withdrawn" row
    beside it would state the opposite of what happened.
    """
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        outcome = session.withdraw_ask(
            ask_id, reason="answered_in_chat", answers={"ship": ["yes tonight"]}
        )
        assert outcome["ok"], outcome
        await asyncio.sleep(0.05)
        assert _custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_refused_withdraw_adds_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``moot`` on a settled ask is refused, and the refusal is silent."""
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        assert session.withdraw_ask(ask_id, reason="moot")["ok"]
        assert await _wait_entry(session, store.withdrawn_row_id(ask_id))

        refused = session.withdraw_ask(ask_id, reason="moot")
        assert refused["ok"] is False
        await asyncio.sleep(0.05)
        assert len(_custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE)) == 1
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_secret_ask_is_named_generically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row names a secret ask by the label; its question never reaches it."""
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        outcome = queue.enqueue(
            [
                {
                    "id": "API_KEY",
                    "question": "Paste the sk-live key",
                    "options": [],
                    "secret": True,
                }
            ],
            3600,
        )
        ask_id = str(outcome["details"]["ask_id"])
        assert session.withdraw_ask(ask_id, reason="moot")["ok"]
        assert await _wait_entry(session, store.withdrawn_row_id(ask_id))

        text = _custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE)[0].payload["details"]["text"]
        assert "sk-live" not in text
        assert "API_KEY" not in text, "the key name is not the receipt's business either"
        assert "secret question" in text
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The model boundary — the whole point of a separate type
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_row_never_reaches_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither the renderer nor a real turn's request carries it."""
    session, queue, stream = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        assert session.withdraw_ask(ask_id, reason="moot")["ok"]
        assert await _wait_entry(session, store.withdrawn_row_id(ask_id))
        receipt = _custom_rows(session, ASK_WITHDRAWN_MESSAGE_TYPE)[0].payload["details"]["text"]

        # The carrier is IN the transcript and is still dropped by the renderer.
        history = session._transcript.build_llm_history()
        assert any(
            isinstance(message, CustomMessage) and message.custom_type == ASK_WITHDRAWN_MESSAGE_TYPE
            for message in history
        ), "the row must be kept for the operator"
        rendered = _texts(_default_convert_to_llm(list(history)))
        assert not any(receipt in text for text in rendered), "the renderer delivered it"

        # And the real path: a later turn's request to the provider.
        await session.prompt("thanks")
        prompt_text = _texts(stream.requests[-1].messages)
        assert not any(receipt in text for text in prompt_text)
        assert not any("Withdrawn by the agent" in text for text in prompt_text)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_no_response_or_relay_row_follows_a_withdrawn_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing owes the model a turn: the reconcile writes no ask row at all."""
    session, queue, _ = await _session(tmp_path, monkeypatch)
    try:
        ask_id = _ask(queue, ASK_TEXT, "ship")
        assert session.withdraw_ask(ask_id, reason="moot")["ok"]
        assert await _wait_entry(session, store.withdrawn_row_id(ask_id))
        await session.reconcile_asks()

        ids = {entry.id for entry in session._transcript.entries()}
        for owed in (
            store.response_row_id(ask_id),
            store.timeout_row_id(ask_id),
            store.relay_row_id(ask_id),
        ):
            assert owed not in ids, f"a withdrawn ask owes nothing, yet {owed} exists"
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# The two surfaces
# ---------------------------------------------------------------------------


def _row(ask_id: str = "a-745e", question: str = "Settings layout") -> CustomMessage:
    details: dict[str, Any] = {
        "ask_id": ask_id,
        "status": "withdrawn",
        "questions": [{"id": "q1", "question": question}],
        "by": "agent",
        "at": 1_763_000_000_000,
    }
    details["text"] = ask_withdrawn_notice(details)[0]
    return CustomMessage(
        custom_type=ASK_WITHDRAWN_MESSAGE_TYPE,
        attribution="agent",
        id=store.withdrawn_row_id(ask_id),
        details=details,
    )


def test_the_tui_paints_it_as_an_ask_row_and_never_cuts_the_state() -> None:
    """An unbranched custom type paints NOWHERE; this is the branch's proof.

    Driven through the real projection entry point, then read off the widget the
    branch builds — the state half must survive a narrow row, because it is the
    audit content and the name is only the handle beside the id.
    """
    from local_operator.tui.session_presentation import (
        PreparedReplay,
        project_settled_rows,
    )
    from local_operator.tui.widgets.transcript import AskResponseBlock

    replay = PreparedReplay()
    assert project_settled_rows(replay, [_row()], fold_width=80) is True
    blocks = [block for block in replay.blocks if isinstance(block, AskResponseBlock)]
    assert blocks, "the withdrawn row painted nothing"
    assert blocks[0]._kind == "withdrawn"

    wide = blocks[0]._build_content(100).plain
    assert 'a-745e · "Settings layout" — Withdrawn by the agent' in wide
    # Narrow enough that the name is not worth printing: the id and the STATE
    # survive together, because the state is the audit content of the row.
    narrow = blocks[0]._build_content(52).plain.strip()
    assert narrow.endswith("a-745e — Withdrawn by the agent"), narrow


def test_the_phone_paints_it_as_a_notice_with_the_same_words() -> None:
    """The phone's fold must name the same fact in the same sentence."""
    from local_operator.mobile.projection import fold_messages_to_entries

    entries = fold_messages_to_entries([_row()])
    notices = [
        entry
        for entry in entries
        if entry.kind == "notice" and entry.details.get("status") == "withdrawn"
    ]
    assert notices, "the phone painted nothing for the withdrawn row"
    entry = notices[0]
    assert entry.text == 'a-745e · "Settings layout" — Withdrawn by the agent'
    assert entry.details["ask_id"] == "a-745e"
    assert entry.details["severity"] == "info"
    assert entry.id == store.withdrawn_row_id("a-745e")


def test_the_type_is_declared_and_deliberately_not_a_carrier() -> None:
    """The one-line statement of the design, so a later editor cannot miss it."""
    assert ASK_WITHDRAWN_MESSAGE_TYPE == "ask_withdrawn"
    assert ASK_WITHDRAWN_MESSAGE_TYPE not in MODEL_VISIBLE_INJECTED_TYPES
    assert store.withdrawn_row_id("a-745e") == "ask-withdrawn-a-745e"
    assert store.withdrawn_row_id("") == ""
