"""H5 remediation — ORIGINAL provenance survives the crash window, and
settlement is terminal-safe (S-3, S-5, H5-2; security matrix rows 5/7/8).

The durability rules pinned here:
- a consume records the row from the ask's ORIGINAL scope/generation — NEVER
  from the current runtime — and reports ``created`` exactly once;
- a pre-scope database is DAMAGE (no backfill), and an empty generation is
  non-authorizing through every door;
- recovery re-derives ONLY from the durable ask record: absent record,
  generic human_required question (no scope block), scope/generation drift and
  multi-pick answers all refuse with NO insert and NO POST;
- settlement UPDATEs are conditional: ``sent``/``refused`` are terminal and a
  settled receipt is byte-identical under a later transient/signin/refusal.

Fakes: sqlite is real (tmp_path); the fence and the wire transport are
substituted at their module attributes (both are imported at CALL time).
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from local_operator.asks import store as asks_store
from local_operator.team_bridge import authority as fence_mod
from local_operator.team_bridge import decide as decide_mod
from local_operator.team_bridge import decide_intents as di
from local_operator.team_bridge.decide_intents import (
    DecideIntents,
    DecideIntentsRefused,
    attempt_decide_intent,
    recover_from_durable_answer,
    retry_from_human_action,
    scope_owner_key,
)

pytestmark = pytest.mark.asyncio

OWNER_KEY = json.dumps(["https://hub.example", "acc_1", "ten_1"])


def _runtime(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        owner=SimpleNamespace(
            canonical_base_url="https://hub.example",
            account_id="acc_1",
            tenant_id="ten_1",
            endpoint_id="ep_1",
            credential_id=7,
            enrollment_generation=7,
        ),
        config_dir=tmp_path,
        tenant_id="ten_1",
        session_id="sess-1",
        client=object(),
    )


@pytest.fixture(autouse=True)
def _fence_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fence_mod, "authority_fence", lambda runtime: None)


def _record(
    intents: DecideIntents,
    runtime,
    *,
    generation: str | None = None,
    scope=None,
    ask_id: str = "a-1",
    question_id: str = "bmsg_1",
):
    return intents.record(
        OWNER_KEY,
        ask_id=ask_id,
        question_id=question_id,
        word="approve",
        generation=str(runtime.owner.enrollment_generation) if generation is None else generation,
        scope_json=json.dumps(
            scope if scope is not None else fence_mod.scope_from_runtime(runtime)
        ),
        consumption_row_id=asks_store.response_row_id(ask_id),
    )


async def test_record_reports_created_exactly_once(tmp_path) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    first, created = _record(intents, runtime)
    second, created_again = _record(intents, runtime)
    assert created is True and created_again is False
    assert first.intent_id == second.intent_id
    assert first.generation == "7"


async def test_pre_scope_database_is_refused_as_damage(tmp_path) -> None:
    db = di.decide_intents_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE decide_intents (intent_id TEXT PRIMARY KEY, owner_key TEXT,"
        " ask_id TEXT, question_id TEXT, word TEXT, state TEXT, error_code TEXT,"
        " retry_after TEXT, attempts INTEGER, consumed_at TEXT, created_at TEXT,"
        " settled_at TEXT, receipt_json TEXT)"
    )
    conn.execute(
        "INSERT INTO decide_intents (intent_id, owner_key, ask_id, question_id, word,"
        " state) VALUES ('x','k','a','q','approve','pending')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        _record(intents, _runtime(tmp_path))
    assert exc.value.code == "intents_damaged"  # no backfill, ever


async def test_empty_generation_is_non_authorizing_in_retry_state(tmp_path) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime, generation="")
    assert (
        intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1", current_generation="7")
        == "owner_changed"
    )
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == intent.state


async def test_sent_is_terminal_and_receipt_is_byte_identical(tmp_path) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    intents.mark_sent(intent.intent_id, {"id": "req_original"})
    before = intents.get(intent.intent_id)
    assert before.state == "sent"
    intents.mark_transient(intent.intent_id, "bridge_unreachable")
    intents.mark_signin(intent.intent_id)
    intents.mark_refused(intent.intent_id, "bridge_disabled")
    after = intents.get(intent.intent_id)
    assert after.state == "sent"
    assert after.receipt_json == before.receipt_json  # byte-identical, m8
    assert json.loads(after.receipt_json)["id"] == "req_original"


async def test_refused_is_terminal_for_later_success(tmp_path) -> None:
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    intents.mark_refused(intent.intent_id, "bridge_disabled")
    intents.mark_sent(intent.intent_id, {"id": "should_not_apply"})
    after = intents.get(intent.intent_id)
    assert after.state == "refused"


async def test_attempt_refuses_terminal_and_empty_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        calls.append(decision)
        return {"request": {"id": "req_1"}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    intents = DecideIntents(tmp_path)
    runtime = _runtime(tmp_path)
    intent, _ = _record(intents, runtime)
    intents.mark_sent(intent.intent_id, {"id": "req_0"})
    with pytest.raises(DecideIntentsRefused) as exc:
        await attempt_decide_intent(runtime.client, intents, intent, runtime=runtime)
    assert exc.value.code == "not_retryable"
    assert calls == []  # settled: never re-POST
    empty, _ = _record(intents, runtime, generation="", ask_id="a-2", question_id="bmsg_2")
    with pytest.raises(DecideIntentsRefused) as exc2:
        await attempt_decide_intent(runtime.client, intents, empty, runtime=runtime)
    assert exc2.value.code == "owner_changed"  # the SHARED validator refuses
    assert calls == []
    # The retry door agrees: a settled row is not retry-eligible.
    with pytest.raises(DecideIntentsRefused):
        await retry_from_human_action(runtime.client, intents, empty.intent_id, runtime=runtime)


def _patch_consumed(
    monkeypatch: pytest.MonkeyPatch,
    details: dict | None,
    *,
    entry_id: str = "",
    custom_type: str = "ask_response",
    attribution: str = "user",
) -> None:
    """Serve ONE consumed ASK_RESPONSE row (or ``None``) from the reader.

    ``read_entry_by_id`` is imported at CALL time inside the loader, so
    patching the module attribute keeps the test at the loader's trust
    boundary: the transcript ROW — not the ask-log fold — is what recovery
    must trust. ``entry_id`` overrides the returned row's id to exercise the
    selector check; the other overrides exercise type/lineage refusals.
    """

    from local_operator.session import transcript as transcript_mod

    def _reader(directory: str, wanted: str):
        if details is None:
            return None
        payload_details = dict(details)
        # What the PRODUCTION row builder stamps for a verdict-backed bridge
        # answer (``AskQueue._response_message``), and what the recovery door
        # requires before it will consider the row at all (R3-5). A row shape
        # the product cannot write is not a test of the product.
        payload_details.setdefault("bridge_provenance", "authority_grant")
        return SimpleNamespace(
            id=entry_id or wanted,
            type="message",
            payload={
                "kind": "custom",
                "custom_type": custom_type,
                "attribution": attribution,
                "details": payload_details,
            },
        )

    monkeypatch.setattr(transcript_mod, "read_entry_by_id", _reader)


async def test_recovery_without_durable_record_refuses(tmp_path, monkeypatch) -> None:
    _patch_consumed(monkeypatch, None)
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            _runtime(tmp_path).client,
            intents,
            _runtime(tmp_path),
            ask_id="a-1",
            question_id="bmsg_1",
        )
    assert exc.value.code == "no_durable_answer"
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == "none"


async def test_recovery_generic_question_and_scope_drift_refuse(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    base_question = {"id": "bridge-bmsg_1", "question_id": "bmsg_1", "human_required": True}
    generic = {
        "ask_id": "a-1",
        "questions": [dict(base_question)],  # NO bridge scope block
        "answers": {"bridge-bmsg_1": ["approve"]},
        "at": 42,
    }
    _patch_consumed(monkeypatch, generic)
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
        )
    assert exc.value.code == "scope_missing"
    # Empty-generation scope: also non-authorizing (never backfilled).
    scoped = dict(generic)
    scoped["questions"] = [
        {**base_question, "bridge": {**fence_mod.scope_from_runtime(runtime), "generation": ""}}
    ]
    _patch_consumed(monkeypatch, scoped)
    with pytest.raises(DecideIntentsRefused) as exc2:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
        )
    assert exc2.value.code == "owner_changed"
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == "none"  # nothing minted


async def test_recovery_positive_uses_original_values_and_posts_once(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    calls: list[object] = []

    async def _fake_decide(client, decision, **kwargs):
        calls.append(decision)
        return {"request": {"id": "req_original"}}

    monkeypatch.setattr(decide_mod, "decide_from_human_answer", _fake_decide)
    reply = {
        "ask_id": "a-1",
        "questions": [
            {
                "id": "bridge-bmsg_1",
                "question_id": "bmsg_1",
                "human_required": True,
                "bridge": fence_mod.scope_from_runtime(runtime),  # generation "7"
            }
        ],
        "answers": {"bridge-bmsg_1": ["approve"]},
        "at": 42,
    }
    _patch_consumed(monkeypatch, reply)
    intents = DecideIntents(tmp_path)
    await recover_from_durable_answer(
        runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
    )
    assert len(calls) == 1  # exactly one POST, human-derived
    stored = intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1", current_generation="7")
    assert stored == "sent"
    row = intents.get(_stored_id(intents, "a-1"))
    assert row is not None
    assert json.loads(row.receipt_json)["id"] == "req_original"
    assert row.generation == "7"  # ORIGINAL generation, not a current-gen rebind
    # The row key derives from the DURABLE scope, so a changed org finds NOTHING.
    other_key = scope_owner_key({**fence_mod.scope_from_runtime(runtime), "tenant_id": "ten_2"})
    assert intents.retry_state_for(other_key, "a-1", "bmsg_1") == "none"


def _stored_id(intents: DecideIntents, ask_id: str) -> str:
    conn = sqlite3.connect(intents._path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT intent_id FROM decide_intents WHERE ask_id = ?", (ask_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return str(row["intent_id"])


async def test_recovery_message_key_only_and_ambiguous_refuse(tmp_path, monkeypatch) -> None:
    """S-6: the answer map is keyed by the IMMUTABLE question id only.

    A durable record that keeps the message id as the key (the pre-fix
    assumption — never produced by the real queue) yields zero picks and must
    refuse WITHOUT inserting; a record whose two questions both match the
    bridge message id is ambiguous and refuses too. Neither case may mint.
    """
    runtime = _runtime(tmp_path)
    scoped_q = {
        "id": "bridge-bmsg_1",
        "question_id": "bmsg_1",
        "human_required": True,
        "bridge": fence_mod.scope_from_runtime(runtime),
    }
    message_key_only = {
        "ask_id": "a-1",
        "questions": [dict(scoped_q)],
        "answers": {"bmsg_1": ["approve"]},  # WRONG key: a message id, not the question id
        "at": 42,
    }
    _patch_consumed(monkeypatch, message_key_only)
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
        )
    assert exc.value.code == "not_human_decision"
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == "none"
    ambiguous = dict(message_key_only)
    ambiguous["questions"] = [dict(scoped_q), dict(scoped_q)]
    ambiguous["answers"] = {"bridge-bmsg_1": ["approve"]}
    _patch_consumed(monkeypatch, ambiguous)
    with pytest.raises(DecideIntentsRefused) as exc2:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
        )
    assert exc2.value.code == "frozen_mismatch"
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == "none"


async def test_recovery_forged_row_type_lineage_and_selector_refuse(tmp_path, monkeypatch) -> None:
    """S-7: the consumed-row verification is fail-closed on every arm.

    A wrong custom type, a non-human attribution, a row whose parsed id is not
    the expected selector, and details naming a DIFFERENT ask each answer
    ``no_durable_answer`` with NOTHING inserted — a forged or foreign row can
    never mint a current-generation intent.
    """
    runtime = _runtime(tmp_path)
    good = {
        "ask_id": "a-1",
        "questions": [
            {
                "id": "bridge-bmsg_1",
                "question_id": "bmsg_1",
                "human_required": True,
                "bridge": fence_mod.scope_from_runtime(runtime),
            }
        ],
        "answers": {"bridge-bmsg_1": ["approve"]},
        "at": 42,
    }
    intents = DecideIntents(tmp_path)
    for patch_kwargs in (
        {"custom_type": "peer_message"},
        {"attribution": "agent"},
        {"entry_id": "resp-SOMEONE-ELSE"},
    ):
        _patch_consumed(monkeypatch, good, **patch_kwargs)
        with pytest.raises(DecideIntentsRefused) as exc:
            await recover_from_durable_answer(
                runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
            )
        assert exc.value.code == "no_durable_answer"
    _patch_consumed(monkeypatch, {**good, "ask_id": "a-OTHER"})
    with pytest.raises(DecideIntentsRefused) as exc4:
        await recover_from_durable_answer(
            runtime.client, intents, runtime, ask_id="a-1", question_id="bmsg_1"
        )
    assert exc4.value.code == "no_durable_answer"
    assert intents.retry_state_for(OWNER_KEY, "a-1", "bmsg_1") == "none"


async def test_a_row_without_the_marker_is_not_considered(tmp_path, monkeypatch) -> None:
    """R3-5: the marker is a CORRELATION marker, and the door asks for it first.

    It says "the harness rendered this row under a confirmed admitted verdict";
    it is not unforgeable (the transcript is same-uid writable) and the AUTHORITY
    remains the consumed grant, the original scope block and the live fence. Its
    job is to keep the door from even considering a row some other writer
    appended, so its absence is a typed refusal rather than a fall-through.
    """

    _patch_consumed(monkeypatch, {"ask_id": "a-1", "bridge_provenance": ""})
    intents = DecideIntents(tmp_path)
    with pytest.raises(DecideIntentsRefused) as exc:
        await recover_from_durable_answer(
            _runtime(tmp_path).client,
            intents,
            _runtime(tmp_path),
            ask_id="a-1",
            question_id="bmsg_1",
        )
    assert exc.value.code == "no_durable_answer"
