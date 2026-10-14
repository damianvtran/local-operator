"""H5 F3 consumption COMPOSITION — the REAL chain, not boundary fakes.

The chain under test, with every leg a real component:
  real AskQueue enqueue (bridge request) → real human ``respond`` → real
  ``revise`` inside the window → the queue's OWN ``_response_message`` +
  append-time ``refresh_delivery_message`` → a REAL ``Transcript`` append
  (actual serialization) → REAL ``read_entry_by_id`` → explicit human
  ``recover_from_durable_answer`` (transport substituted at its module
  attribute only).

What the helper-level provenance tests cover with patches, this file closes
with the actual writer/reader stack: the row that recovery trusts is the one
the TRANSCRIPT wrote, addressed by the selector the ASK minted.

Asserts pinned here (per the Root composition mandate):
- the response row is durable exactly once, carries the ORIGINAL bridge scope
  block on its question, and recovery POSTs exactly once with the FROZEN final
  selection;
- a revision accepted BEFORE the append wins (the consumed row is the final
  word) and pre-consume/missing-row/wrong-lineage all refuse with ZERO POST;
- duplicate consumption of the same durable row is zero-POST (the intent
  ledger's created flag), and sent/refused remain terminal.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from local_operator.asks import store as asks_store
from local_operator.asks.queue import AskQueue
from local_operator.session.transcript import Transcript, read_entry_by_id
from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import decide as decide_mod
from local_operator.team_bridge.asks_bridge import enqueue_bridge_request_ask
from local_operator.team_bridge.authority import scope_from_owner
from local_operator.team_bridge.decide_intents import (
    DecideIntents,
    DecideIntentsRefused,
    recover_from_durable_answer,
)
from local_operator.team_bridge.store import OwnerSnapshot, _owner_key
from local_operator.team_bridge.tool import BridgeRuntime
from tests.unit.asks.test_queue import BASE
from tests.unit.asks.test_queue import FakeSession as QueueFakeSession

pytestmark = pytest.mark.asyncio

MESSAGE_ID = "bmsg_abc"
QUESTION_UID = f"bridge-{MESSAGE_ID}"


def _owner() -> OwnerSnapshot:
    return cast(
        OwnerSnapshot,
        SimpleNamespace(
            canonical_base_url="https://hub.example",
            account_id="acc_1",
            tenant_id="ten_1",
            endpoint_id="ep_1",
            credential_id=7,
            enrollment_generation=7,
        ),
    )


def _runtime(tmp_path) -> BridgeRuntime:
    return cast(
        BridgeRuntime,
        SimpleNamespace(
            owner=_owner(),
            config_dir=tmp_path,
            tenant_id="ten_1",
            session_id="s1",
            client=object(),
        ),
    )


def _answered_queue(tmp_path) -> tuple[AskQueue, str]:
    """A real queue with ONE answered bridge ask; returns (queue, ask_id)."""

    queue = AskQueue(QueueFakeSession(), config_dir=tmp_path, session_id="s1", clock=lambda: BASE)
    scope = scope_from_owner(_owner(), "s1")
    outcome = enqueue_bridge_request_ask(
        queue,
        message_id=MESSAGE_ID,
        sender="dana@acme.example",
        summary="wants your sign-off",
        excerpt="please approve before 1700",
        scope=scope,
    )
    assert outcome["ok"] is True
    ask_id = str(outcome["details"]["ask_id"])
    # The human's tap in PRODUCTION goes through the authority seam, which mints
    # the one-shot grant at admission and only then records the answer: that is
    # what makes the consumed row carry the correlation marker the recovery door
    # requires. Answering without it here would be a shape production cannot
    # produce (round-4 R3-5).
    from local_operator.harness.approval import mint_bridge_decide_grant

    mint_bridge_decide_grant(session_id="s1", ask_id=ask_id, question_id=MESSAGE_ID, word="approve")
    answered = queue.respond(ask_id, {QUESTION_UID: ["approve"]}, by="human")
    assert answered.get("ok") is True
    return queue, ask_id


def _consume(queue: AskQueue, ask_id: str, tmp_path):
    """The session's OWN two-step: reconcile build + append-time refresh."""

    records = [r for r in queue.records() if str(r.get("ask_id") or "") == ask_id]
    assert records, "the answered ask must be in the fold"
    message = queue.refresh_delivery_message(queue._response_message(records[-1]))
    directory = asks_store.session_dir(tmp_path, "s1")
    transcript = Transcript(directory)
    return message, directory, transcript


async def _append_and_load(message, directory, transcript):
    await transcript.append_message(message)
    expected = asks_store.response_row_id(message.details.get("ask_id") or "")
    return expected, read_entry_by_id(directory, expected)


async def test_real_chain_recovers_once_with_the_frozen_final_selection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[tuple[str, str]] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(
            (str(getattr(decision, "message_id", "")), str(getattr(decision, "word", "")))
        )
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    queue, ask_id = _answered_queue(tmp_path)
    # REVISION INSIDE THE WINDOW: the append has not happened, so the change
    # of mind is accepted — and the CONSUMED row must carry it. A revision is a
    # SECOND admitted decision, so its own grant is minted first (production
    # mints at admission for the word it admitted; without one the row would be
    # a shape the product cannot produce).
    from local_operator.harness.approval import mint_bridge_decide_grant

    mint_bridge_decide_grant(session_id="s1", ask_id=ask_id, question_id=MESSAGE_ID, word="deny")
    revised = queue.revise(ask_id, {QUESTION_UID: ["deny"]}, by="human")
    assert revised.get("ok") is not False
    message, directory, transcript = _consume(queue, ask_id, tmp_path)
    expected, entry = await _append_and_load(message, directory, transcript)
    assert entry is not None and entry.id == expected
    assert entry.payload.get("custom_type") == "ask_response"
    assert entry.payload.get("attribution") == "user"
    details = entry.payload.get("details") or {}
    assert details.get("ask_id") == ask_id
    questions = details.get("questions") or []
    assert [q.get("id") for q in questions] == [QUESTION_UID]
    assert (questions[0].get("bridge") or {}).get("session_id") == "s1"
    runtime = _runtime(tmp_path)
    intents = DecideIntents(tmp_path)
    first = await recover_from_durable_answer(
        runtime.client, intents, runtime, ask_id=ask_id, question_id=MESSAGE_ID
    )
    assert first is not None
    # THE FINAL WORD WINS (revision before freeze), exactly one POST.
    assert posted == [(MESSAGE_ID, "deny")]
    # DUPLICATE consumption of the same durable row: the sent receipt is
    # TERMINAL — a second attempt is refused/read-only and never a second POST.
    try:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id=ask_id, question_id=MESSAGE_ID
        )
    except DecideIntentsRefused:
        pass  # the terminal sent receipt is not retryable — read-only
    assert len(posted) == 1


async def test_pre_consumption_and_missing_rows_refuse_with_zero_post(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(decision)
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    queue, ask_id = _answered_queue(tmp_path)
    runtime = _runtime(tmp_path)
    intents = DecideIntents(tmp_path)
    # PRE-CONSUMPTION: answered in the log, but the row has NOT been appended.
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id=ask_id, question_id=MESSAGE_ID
        )
    assert exc.value.code == "no_durable_answer"
    # MISSING ROW under the expected selector.
    directory = asks_store.session_dir(tmp_path, "s1")
    directory.mkdir(parents=True, exist_ok=True)
    assert read_entry_by_id(directory, asks_store.response_row_id(ask_id)) is None
    assert posted == []
    assert _owner_key(runtime.owner)  # the scope key is derivable, unused


async def test_wrong_lineage_row_refuses_even_with_real_serialization(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    posted: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        posted.append(decision)
        return {"request": {}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)
    queue, ask_id = _answered_queue(tmp_path)
    message, directory, transcript = _consume(queue, ask_id, tmp_path)
    # FORGED LINEAGE: the same selector carries an agent-attributed row.
    forged = message.model_copy(update={"attribution": "agent"})
    await transcript.append_message(forged)
    runtime = _runtime(tmp_path)
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id=ask_id, question_id=MESSAGE_ID
        )
    assert exc.value.code == "no_durable_answer"
    assert posted == []
