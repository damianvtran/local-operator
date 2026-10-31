"""Ordering controls at the shared model renderer; no hub or model network."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from local_operator.harness.render import _default_convert_to_llm
from local_operator.harness.types import CustomMessage, Message, TextContent
from local_operator.session.transcript import Transcript
from local_operator.team_bridge.card_projection import PENDING_CARD
from local_operator.team_bridge.connector.conversation import (
    ConversationCardMetadata,
    build_conversation_card,
)
from local_operator.team_bridge.pending_turn import PendingTurn
from local_operator.team_bridge.store import state_db_path
from tests.unit.team_bridge.test_body_turn_e2e import (  # noqa: F401 -- autouse fixture
    BODY,
    CONVERSATION_ID,
    MESSAGE_ID,
    SESSION_ID,
    TENANT,
    CapturingStream,
    FakeBridgeClient,
    _message_texts,
    _owner,
    _reserve,
    _row,
    _runtime,
    _session,
    _store,
    telemetry_off,
)


def _card(session, *, message_id=MESSAGE_ID, coalesced=()):
    return session._peer_custom_message(
        build_conversation_card(
            ConversationCardMetadata(
                conversation_id=CONVERSATION_ID,
                tenant_id=TENANT,
                conversation_label="room-canary",
                sender_label="name-canary",
                sender_kind="human",
                mention=False,
                message_id=message_id,
                conversation_seq=3,
                conversation_kind="dm",
                coalesced_message_ids=coalesced,
            )
        ),
        {"source": "team_bridge"},
        custom_type="team_bridge_message",
    )


def _text(messages):
    return "\n".join(_message_texts(messages))


def _assert_pending(text):
    assert PENDING_CARD in text, text
    for value in (
        "name-canary",
        "room-canary",
        "mention:",
        "expect:",
        CONVERSATION_ID,
        MESSAGE_ID,
        BODY,
    ):
        assert value not in text, text


async def _claim(session, config_dir):
    turn = PendingTurn(_owner(), SESSION_ID, config_dir=config_dir)
    assert await turn.open(budget=session._pending_body_budget(), fetch=session._fetch_pending_body)
    return turn


async def _complete(session, config_dir, card):
    turn = await _claim(session, config_dir)
    candidate = turn.transform([card])
    admitted = _text(session._render_history(candidate))
    assert "mention: no" in admitted and BODY in admitted, admitted
    entry = await session._transcript.append_message(
        Message(
            role="assistant",
            content=[TextContent(text="the admitted body was read")],
        )
    )
    assert (
        turn.settle(entry_id=entry.id, error=None, stop_reason="stop", aborted=False) == "completed"
    )
    return turn, candidate


@pytest.mark.asyncio
async def test_raw_persisted_card_never_leaks_in_any_render_or_cold_replay(tmp_path):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        card = _card(session)
        await session._transcript.append_message(card)
        raw = session._transcript.build_llm_history()
        # The UI keeps its raw entry; model conversion never receives its text.
        assert "mention: no" in raw[0].details["text"]
        for rendered in (
            _default_convert_to_llm(raw),
            session._render_history(raw),
            session._render_history(raw, keep_images=True),
            session._render_for_request(raw),
        ):
            _assert_pending(_text(rendered))
        cold = Transcript(tmp_path / "sess")
        _assert_pending(_text(_default_convert_to_llm(cold.build_llm_history())))
        assert _row(cfg)["state"] == "pending"
        assert _row(cfg)["attempts"] == 0
        assert _row(cfg)["card_posted_at"] is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_completed_admission_preserves_history_but_not_stale_cached_eligibility(tmp_path):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    runtime = _runtime(cfg, FakeBridgeClient())
    session = _session(tmp_path, CapturingStream([]), runtime)
    try:
        card = _card(session)
        await session._transcript.append_message(card)
        original = json.dumps(card.model_dump(), sort_keys=True)
        _, candidate = await _complete(session, cfg, card)
        assert json.dumps(card.model_dump(), sort_keys=True) == original, "no journal mutation"
        raw = session._transcript.build_llm_history()
        assert "mention: no" in _text(session._render_history(raw))
        assert BODY not in _text(session._render_history(raw))
        # A new owner binding cannot reuse either a historical or live copy.
        from local_operator.team_bridge.card_projection import project_history

        cached = project_history(
            raw, runtime=lambda: session.bridge_runtime, transcript=session._transcript
        )
        session.bridge_runtime = SimpleNamespace(**{**vars(runtime), "tenant_id": "foreign"})
        _assert_pending(_text(_default_convert_to_llm(cached)))
        _assert_pending(_text(session._render_history(raw)))
        # The candidate's lease ended at settlement: only historical proof,
        # not its stale in-memory token, may authorize a later envelope.
        _assert_pending(_text(_default_convert_to_llm(candidate[:-1])))
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_entry",
        "user_entry",
        "older_entry",
        "policy_epoch",
        "consent_version",
        "enrollment_generation",
        "credential_id",
        "tenant_id",
        "endpoint_id",
        "message_id",
    ],
)
async def test_completed_flag_cannot_bypass_exact_live_epoch_and_committed_entry(
    tmp_path, corruption
):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        card = _card(session)
        await session._transcript.append_message(card)
        await _complete(session, cfg, card)
        if corruption in ("user_entry", "older_entry"):
            role = "user" if corruption == "user_entry" else "assistant"
            entry = await session._transcript.append_message(
                Message(role=role, content=[TextContent(text="unrelated")])
            )
            if corruption == "older_entry":
                entry.ts = 0
            column, value = "completed_entry_id", entry.id
        else:
            column = "completed_entry_id" if corruption == "missing_entry" else corruption
            value = (
                999
                if corruption in ("policy_epoch", "enrollment_generation", "credential_id")
                else "foreign"
            )
        with sqlite3.connect(state_db_path(cfg)) as conn:
            conn.execute(f"UPDATE body_reservations SET {column} = ?", (value,))
        _assert_pending(_text(session._render_history([card])))
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_coalesced_card_requires_every_independent_completed_admission(tmp_path):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    second = "bcmsg_abcdef02"
    _reserve(store, message_id=second, seq=4, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        grouped = _card(session, message_id=second, coalesced=(MESSAGE_ID, second))
        first = _card(session)
        await _complete(session, cfg, first)
        _assert_pending(_text(session._render_history([grouped])))
        await _complete(session, cfg, _card(session, message_id=second))
        rendered = _text(session._render_history([grouped]))
        assert f"message_ids: {MESSAGE_ID},{second}" in rendered, rendered
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("injected", [False, True])
async def test_crash_without_completion_stays_pending_and_restart_can_retry(tmp_path, injected):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        card = _card(session)
        await session._transcript.append_message(card)
        if injected:
            turn = await _claim(session, cfg)
            admitted = _text(session._render_history(turn.transform([card])))
            assert BODY in admitted and "mention: no" in admitted
            assert _row(cfg)["state"] == "claimed"
        _assert_pending(_text(session._render_history(session._transcript.build_llm_history())))
        before = _row(cfg)
        assert before["completed_entry_id"] is None
        assert before["card_posted_at"] is None
        if injected:
            # Simulate elapsed crash-recovery time; the production restart
            # reclaims this SAME reservation after its native lease expires.
            with sqlite3.connect(state_db_path(cfg)) as conn:
                conn.execute("UPDATE body_reservations SET claim_expires_at = 0")
        session._transcript.close()
        restarted = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
        try:
            raw = restarted._transcript.build_llm_history()
            _assert_pending(_text(restarted._render_history(raw)))
            assert _row(cfg)["attempts"] == before["attempts"], "render must not claim/ack"
            await _complete(restarted, cfg, raw[0])
            assert _row(cfg)["state"] == "completed"
            assert "mention: no" in _text(restarted._render_history(raw))
            assert _row(cfg)["card_posted_at"] is None
        finally:
            await restarted.dispose()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_failed_hydration_and_unrelated_peer_controls(tmp_path):
    class Unavailable(FakeBridgeClient):
        async def conversation_body(self, *args, **kwargs):
            raise RuntimeError("offline")

    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, Unavailable()))
    try:
        card = _card(session)
        turn = PendingTurn(_owner(), SESSION_ID, config_dir=cfg)
        assert not await turn.open(
            budget=session._pending_body_budget(), fetch=session._fetch_pending_body
        )
        _assert_pending(_text(session._render_history([card])))
        assert _row(cfg)["state"] == "pending"
        assert _row(cfg)["completed_entry_id"] is None
        peer = CustomMessage(
            custom_type="peer_message", attribution="user", details={"text": "ordinary-peer-canary"}
        )
        assert "ordinary-peer-canary" in _text(_default_convert_to_llm([peer]))
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_presented_ref_is_pure_nonserialized_and_requires_exact_admitted_frame(tmp_path):
    from local_operator.team_bridge.card_projection import presented_bridge_ref

    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        card = _card(session)
        assert presented_bridge_ref(session._render_history([card])) is None
        raw = Message.user(card.details["text"])
        assert presented_bridge_ref([raw]) is None
        turn = await _claim(session, cfg)
        candidate = turn.transform([card])
        rendered = session._render_history(candidate)
        ref = presented_bridge_ref(rendered)
        assert ref is not None
        assert (ref.conversation_id, ref.message_id, ref.conversation_seq) == (
            CONVERSATION_ID,
            MESSAGE_ID,
            3,
        )
        # Byte-identical bodies reconstructed from serialised provider/history
        # data are NOT host admission proof. No descriptor is in model_dump.
        assert all("_bridge_" not in json.dumps(message.model_dump()) for message in rendered)
        reconstructed = [Message.model_validate(message.model_dump()) for message in rendered]
        assert presented_bridge_ref(reconstructed) is None
        changed = [message.model_copy(deep=True) for message in rendered]
        changed[-1].content = [TextContent(text="different bytes")]
        assert presented_bridge_ref(changed) is None
        # Purity's deliberate boundary: old payload recognises its render-time
        # attestation, but a NEW render observes epoch revocation immediately.
        with sqlite3.connect(state_db_path(cfg)) as conn:
            conn.execute("UPDATE screening_policy SET epoch = epoch + 1")
        assert presented_bridge_ref(rendered) == ref
        rerendered = session._render_history(candidate)
        assert presented_bridge_ref(rerendered) is None
        _assert_pending(_text(rerendered))
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["tenant_id", "conversation_id", "message_id"])
async def test_live_body_never_authorizes_an_envelope_for_a_different_ref(tmp_path, field):
    cfg = tmp_path / "cfg"
    store = _store(cfg)
    _reserve(store, config_dir=cfg)
    session = _session(tmp_path, CapturingStream([]), _runtime(cfg, FakeBridgeClient()))
    try:
        card = _card(session)
        original = {
            "tenant_id": TENANT,
            "conversation_id": CONVERSATION_ID,
            "message_id": MESSAGE_ID,
        }[field]
        details = {
            key: (
                value.replace(f"{field}: {original}", f"{field}: foreign")
                if isinstance(value, str)
                else value
            )
            for key, value in card.details.items()
        }
        wrong = card.model_copy(update={"details": details})
        turn = await _claim(session, cfg)
        rendered = session._render_history(turn.transform([wrong]))
        assert _message_texts(rendered)[0] == PENDING_CARD
        assert BODY in _text(rendered), "the exact admitted body is still present"
    finally:
        await session.dispose()
