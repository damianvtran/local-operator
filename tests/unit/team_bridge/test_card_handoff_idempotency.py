"""Card hand-off idempotency across a failed posted-marker write (PR #3478 QA r1).

THE FAULT (QA comment 6663928534, author reply 6663979598): the designated
session accepted a conversation card, ``Store.mark_body_card_posted`` then
refused, and a forced replay of the handler posted a SECOND card, because every
hand-off minted a fresh ``peer-<hex>`` id that the receiver had never seen.

WHAT IS REAL HERE: the SQLite ``Store`` (a disposable file under ``tmp_path``),
``handle_conversation_available``, the ``WakeCoalescer`` flush,
``conversation._deliver_card``, ``wake.deliver_wake``,
``peer_send.deliver_peer_message_outcome`` with its ``_engaged_outcome`` /
``_classify_ack`` classification, and the RECEIVER: a real ``Session`` whose
``receive_peer_message`` duplicate gate (transcript index + in-flight set) is
the boundary under test. Only the hub (``FakeServer``), screening, and the
process hop (``engage_runtime``: socket + runtime spawn) are synthetic; the hop
forwards the errand's ``command_id`` exactly as ``launch._deliver`` does.

The receiver is driven record-only (``wake=False``, no custom type): the
duplicate gate runs BEFORE any wake/custom-type branch in
``Session.receive_peer_message``, so it is the identical gate, without starting
a model turn. No production traffic, model, provider or network is involved.
"""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.mobile import peer_send
from local_operator.mobile.peer_client import PeerAck
from local_operator.session.runtime import launch
from local_operator.team_bridge.connector import coalesce, conversation, wake
from local_operator.team_bridge.store import Store, StoreRefused, state_db_path
from tests.unit.session.test_session import ScriptedStream, make_session
from tests.unit.team_bridge import test_conversation_delivery as rig


class _Receiver:
    """The real receiving ``Session`` behind a synthetic process hop."""

    def __init__(self, tmp_path) -> None:
        self.session = make_session(tmp_path / "receiver", ScriptedStream([[]]))
        self.dispatched: list[str] = []  # every id that reached the hop
        self.accepted: list[str] = []  # ids the REAL gate appended as new rows
        self.hold_inflight = False

    async def engage(self, session_id, cwd, work, *, config_dir):
        assert isinstance(work, launch.PeerMessageErrand)
        self.dispatched.append(work.command_id)
        if self.hold_inflight:
            # The receiver's own in-flight set holds the id (an earlier delivery
            # is mid-await): put it there, ask the REAL gate, take it back out.
            self.session._peer_inflight_ids.add(work.command_id)
        try:
            detail = await self.session.receive_peer_message(
                work.text,
                mode="mailbox",
                wake=False,
                sender=dict(work.sender),
                message_id=work.command_id or None,
            )
        finally:
            if self.hold_inflight:
                self.session._peer_inflight_ids.discard(work.command_id)
        delivery = getattr(detail, "delivery", None) or {}
        if not delivery.get("duplicate"):
            self.accepted.append(work.command_id)
        return SimpleNamespace(detail=PeerAck(str(detail), delivery), state="delivered")


def _rig(tmp_path, monkeypatch):
    owner = rig._owner("a", 1)
    config_dir = tmp_path / "a"
    config_dir.mkdir()
    rig._enroll(config_dir, owner, "sess_a")
    server = rig.FakeServer(notify={"bep_a": "mute"}, active={"bep_a": True}, mentioned={"bep_a"})
    rig._wire_rig(monkeypatch, server, {"a": (owner, config_dir)})
    monkeypatch.setattr(conversation, "deliver_wake", wake.deliver_wake)
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: True)
    monkeypatch.setattr(peer_send, "probe_transcript_for", lambda *_a, **_k: False)
    receiver = _Receiver(tmp_path)
    monkeypatch.setattr(launch, "engage_runtime", receiver.engage)
    return owner, config_dir, server, receiver


def _fail_marker_once(monkeypatch, ids=None):
    """``mark_body_card_posted`` refuses ONCE per targeted message id."""

    real = Store.mark_body_card_posted
    failed: set[str] = set()

    def flaky(self, owner, conversation_id, message_id, *, now):
        if (ids is None or message_id in ids) and message_id not in failed:
            failed.add(message_id)
            raise StoreRefused("store_refused")
        return real(self, owner, conversation_id, message_id, now=now)

    monkeypatch.setattr(Store, "mark_body_card_posted", flaky)
    return failed


def _handle(owner, config_dir, server):
    return asyncio.run(
        conversation.handle_conversation_available(
            owner,
            rig._ref(server),
            config_dir=config_dir,
            auth_store=None,
            access=SimpleNamespace(credential_id=owner.credential_id),
        )
    )


def _handoff_ids(config_dir) -> dict[str, Any]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        return dict(conn.execute("SELECT message_id, card_handoff_id FROM body_reservations"))


def test_marker_failure_then_forced_replay_posts_exactly_one_card(tmp_path, monkeypatch):
    """The QA reproduction. Base (59888078/3b7ef2fd): two accepted cards. Now: one."""

    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    _fail_marker_once(monkeypatch)
    assert _handle(owner, config_dir, server)["action"] == "delivered"
    row = rig._reservation_rows(config_dir)[0]
    assert row["card_posted_at"] is None, "the marker write was refused"
    bound = _handoff_ids(config_dir)[rig.MESSAGE_ID]
    assert peer_send.valid_peer_message_id(bound)

    # The forced replay: the reservation reads CARD_OWED and re-drives the card.
    assert _handle(owner, config_dir, server)["action"] == "delivered"
    assert receiver.dispatched == [bound, bound], "same id, never a fresh one"
    assert receiver.accepted == [bound], "the receiver's own gate kept ONE card"
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is not None
    # And the next replay does not dispatch at all.
    assert _handle(owner, config_dir, server)["action"] == "duplicate"
    assert len(receiver.dispatched) == 2


def test_replay_while_receiver_holds_the_id_in_flight_stays_owed(tmp_path, monkeypatch):
    """Duplicate-not-committed is unknown: held, owed, same id on the next retry."""

    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    _fail_marker_once(monkeypatch)
    _handle(owner, config_dir, server)
    bound = _handoff_ids(config_dir)[rig.MESSAGE_ID]
    # Simulate the in-flight shape on a FRESH receiver transcript, so the id is
    # not committed there and only the in-flight half of the gate can answer.
    receiver.session = make_session(tmp_path / "receiver2", ScriptedStream([[]]))
    receiver.hold_inflight = True
    outcome = _handle(owner, config_dir, server)
    assert outcome["action"] == "hold" and outcome["code"] == "card_unconfirmed"
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is None
    assert receiver.dispatched == [bound, bound]
    receiver.hold_inflight = False
    # In-flight resolved as NEW on this receiver (nothing committed before under
    # this id there): the same id is accepted once and the card is recorded.
    assert _handle(owner, config_dir, server)["action"] == "delivered"
    assert receiver.dispatched == [bound, bound, bound]
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is not None


def test_invalid_explicit_id_refuses_before_any_dispatch(monkeypatch):
    calls = []

    async def engage(*args, **kwargs):  # pragma: no cover - must not run
        calls.append(args)

    monkeypatch.setattr(launch, "engage_runtime", engage)
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: True)
    for bad in ("peer-XYZ", "", "bcmsg_abcdef01", "peer-" + "0" * 31):
        with pytest.raises(ValueError):
            asyncio.run(
                peer_send.deliver_peer_message_outcome(
                    None,
                    session_id="sess_a",
                    text="t",
                    mode="mailbox",
                    wake=True,
                    sender={},
                    message_id=bad,
                )
            )
    assert calls == []


def test_omitted_id_still_mints_a_fresh_id_per_call(monkeypatch, tmp_path):
    """Every other caller is unchanged: no argument, a new id each call."""

    receiver = _Receiver(tmp_path)
    monkeypatch.setattr(launch, "engage_runtime", receiver.engage)
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: True)
    for _ in range(2):
        asyncio.run(
            peer_send.deliver_peer_message_outcome(
                None, session_id="s", text="t", mode="mailbox", wake=True, sender={}
            )
        )
    assert len(set(receiver.dispatched)) == 2 and receiver.accepted == receiver.dispatched


def test_wrong_owner_generation_or_session_binds_nothing_and_dispatches_nothing(
    tmp_path, monkeypatch
):
    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    _fail_marker_once(monkeypatch)
    _handle(owner, config_dir, server)
    store = Store(config_dir)
    other = rig._owner("b", 9)
    with pytest.raises(StoreRefused):
        store.bind_card_handoff(
            other,
            rig.CONVERSATION_ID,
            [rig.MESSAGE_ID],
            designated_session="sess_a",
            minted_id="peer-" + "1" * 32,
        )
    with pytest.raises(StoreRefused) as wrong_session:
        store.bind_card_handoff(
            owner,
            rig.CONVERSATION_ID,
            [rig.MESSAGE_ID],
            designated_session="sess_other",
            minted_id="peer-" + "1" * 32,
        )
    assert wrong_session.value.code == "card_binding_mismatch"
    with pytest.raises(StoreRefused):
        store.bind_card_handoff(
            owner,
            rig.CONVERSATION_ID,
            [rig.MESSAGE_ID],
            designated_session="sess_a",
            minted_id="not-an-id",
        )
    # Through _deliver_card: a different designated session is refused pre-dispatch.
    metadata = _metadata(rig.MESSAGE_ID, 1)
    before = len(receiver.dispatched)
    with pytest.raises(StoreRefused):
        asyncio.run(
            conversation._deliver_card(
                owner, metadata, session_id="sess_other", config_dir=config_dir
            )
        )
    assert len(receiver.dispatched) == before


def test_legacy_unbound_row_is_held_with_zero_dispatch(tmp_path, monkeypatch):
    """A row from before the column (NULL) has no provenance: no automatic replay."""

    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    _fail_marker_once(monkeypatch)
    _handle(owner, config_dir, server)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE body_reservations SET card_handoff_id = NULL")
    dispatched = list(receiver.dispatched)
    outcome = _handle(owner, config_dir, server)
    assert outcome["action"] == "hold" and outcome["code"] == "card_provenance_unknown"
    assert receiver.dispatched == dispatched
    assert rig._reservation_rows(config_dir)[0]["card_posted_at"] is None
    assert _handoff_ids(config_dir)[rig.MESSAGE_ID] is None, "nothing inferred or rewritten"


def test_concurrent_binds_converge_on_one_identity(tmp_path, monkeypatch):
    """Binders racing the same owed rows (threads, separate Store handles, one
    SQLite file): exactly one mints, every other one replays THAT id."""

    import threading

    owner, config_dir, _server, _receiver = _rig(tmp_path, monkeypatch)
    ids = _reserve(owner, config_dir, ["bcmsg_abcdef0a", "bcmsg_abcdef0b"])
    barrier = threading.Barrier(6)
    results: list[tuple[str, bool]] = []

    def bind(n: int) -> None:
        barrier.wait()
        results.append(
            Store(config_dir).bind_card_handoff(
                owner,
                rig.CONVERSATION_ID,
                ids,
                designated_session="sess_a",
                minted_id="peer-" + f"{n:x}" * 32,
            )
        )

    threads = [threading.Thread(target=bind, args=(n,)) for n in range(1, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 6
    assert len({handoff for handoff, _ in results}) == 1
    assert sorted(replay for _, replay in results) == [False] + [True] * 5
    assert set(_handoff_ids(config_dir).values()) == {results[0][0]}


def test_concurrent_replays_of_one_owed_card_accept_one(tmp_path, monkeypatch):
    """Two hand-offs of the same owed card in flight together: same id, one row."""

    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    ids = _reserve(owner, config_dir, ["bcmsg_abcdef0a"])

    async def both():
        return await asyncio.gather(
            *(
                conversation._deliver_card(
                    owner, _metadata(ids[0], 1), session_id="sess_a", config_dir=config_dir
                )
                for _ in range(2)
            ),
            return_exceptions=True,
        )

    asyncio.run(both())
    assert len(receiver.dispatched) == 2 and len(set(receiver.dispatched)) == 1
    assert len(receiver.accepted) == 1


# -- the coalesced group: A+B accepted, B's marker fails, retry B (+C) ---------


def _metadata(message_id: str, seq: int):
    return conversation.ConversationCardMetadata(
        conversation_id=rig.CONVERSATION_ID,
        tenant_id="tnt_a",
        conversation_label="room",
        sender_label="Ada",
        sender_kind="human",
        mention=False,
        message_id=message_id,
        conversation_seq=seq,
    )


def _reserve(owner, config_dir, ids):
    store = Store(config_dir)
    for seq, mid in enumerate(ids, start=1):
        store.record_body_reservation(
            owner,
            conversation_id=rig.CONVERSATION_ID,
            message_id=mid,
            conversation_seq=seq,
            mention=False,
            body_sha256="a" * 64,
            screening_fingerprint="b" * 64,
            policy_epoch=1,
            consent_version=None,
            designated_session="sess_a",
            now=1.0,
        )
    return list(ids)


def _flush(owner, config_dir, items):
    coalescer = coalesce.WakeCoalescer(window_s=0)
    for mid, seq in items:
        coalescer.enqueue(
            owner,
            _metadata(mid, seq),
            ref_key=f"ref-{mid}",
            session_id="sess_a",
            config_dir=config_dir,
            now=0.0,
        )
    return asyncio.run(coalescer.flush_due(deliver=conversation._deliver_card, now=10.0))


def _posted(config_dir) -> dict[str, bool]:
    return {
        r["message_id"]: r["card_posted_at"] is not None for r in rig._reservation_rows(config_dir)
    }


def test_partial_group_marker_failure_retry_with_new_member(tmp_path, monkeypatch):
    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(coalesce, "_persist_hold", lambda *a, **k: None)
    monkeypatch.setattr(Store, "delete_hold", lambda *a, **k: None)
    a, b, c = "bcmsg_abcdef0a", "bcmsg_abcdef0b", "bcmsg_abcdef0c"
    _reserve(owner, config_dir, [a, b, c])
    _fail_marker_once(monkeypatch, ids={b})

    first = _flush(owner, config_dir, [(a, 1), (b, 2)])
    assert [o["action"] for o in first] == ["delivered"]
    assert _posted(config_dir) == {a: True, b: False, c: False}
    group_id = _handoff_ids(config_dir)[a]
    assert _handoff_ids(config_dir)[b] == group_id and _handoff_ids(config_dir)[c] == ""
    assert receiver.accepted == [group_id]

    # Retry: B is still owed, C arrives in the same window.
    second = _flush(owner, config_dir, [(b, 2), (c, 3)])
    assert sorted(o["message_ids"][0] for o in second) == [b, c], "B and C split"
    c_id = _handoff_ids(config_dir)[c]
    assert c_id != group_id and peer_send.valid_peer_message_id(c_id)
    # B re-sent under its ORIGINAL id and was deduplicated; C is a new card.
    assert receiver.dispatched == [group_id, group_id, c_id]
    assert receiver.accepted == [group_id, c_id], "B not duplicated, C not dropped"
    assert _posted(config_dir) == {a: True, b: True, c: True}


def test_group_replay_never_reached_receiver_narrows_unsent_member(tmp_path, monkeypatch):
    """A+B bound, the dispatch failed (nothing landed); retry carries only B.

    The id is accepted as NEW for B alone, so A (still owed, still bound to that
    id) is released: otherwise A's own replay would be swallowed as a duplicate.
    """

    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(coalesce, "_persist_hold", lambda *a, **k: None)
    monkeypatch.setattr(Store, "delete_hold", lambda *a, **k: None)
    a, b = "bcmsg_abcdef0a", "bcmsg_abcdef0b"
    _reserve(owner, config_dir, [a, b])
    real_engage = receiver.engage

    async def refused(*args, **kwargs):
        raise launch_dial_failed()

    from local_operator.mobile.peer_client import (
        ControlDialFailed as launch_dial_failed,
    )

    monkeypatch.setattr(launch, "engage_runtime", refused)
    assert [o["action"] for o in _flush(owner, config_dir, [(a, 1), (b, 2)])] == ["hold"]
    group_id = _handoff_ids(config_dir)[a]
    assert _handoff_ids(config_dir)[b] == group_id
    monkeypatch.setattr(launch, "engage_runtime", real_engage)

    assert [o["action"] for o in _flush(owner, config_dir, [(b, 2)])] == ["delivered"]
    assert receiver.accepted == [group_id]
    assert _handoff_ids(config_dir)[a] == "", "A released for its own identity"
    _flush(owner, config_dir, [(a, 1)])
    a_id = _handoff_ids(config_dir)[a]
    assert a_id != group_id and receiver.accepted == [group_id, a_id]
    assert _posted(config_dir) == {a: True, b: True}


def test_unreserved_card_keeps_the_minted_id(tmp_path, monkeypatch):
    """No reservation row (the unadmitted recovery card): historical behaviour."""

    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    for _ in range(2):
        asyncio.run(
            conversation._deliver_card(
                owner,
                _metadata("bcmsg_abcdef0f", 1),
                session_id="sess_a",
                config_dir=config_dir,
            )
        )
    assert len(set(receiver.dispatched)) == 2 and receiver.accepted == receiver.dispatched


def test_cold_target_without_history_is_unchanged_at_least_once(tmp_path, monkeypatch):
    """Documented residual: the no-history errand carries no id; nothing is claimed."""

    owner, config_dir, server, _receiver = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda *_a, **_k: False)
    errands = []

    async def engage(session_id, cwd, work, *, config_dir):
        errands.append(work)
        return SimpleNamespace(state="started")

    monkeypatch.setattr(launch, "engage_runtime", engage)
    _fail_marker_once(monkeypatch)
    _handle(owner, config_dir, server)
    _handle(owner, config_dir, server)
    assert len(errands) == 2 and all(not e.command_id for e in errands)
