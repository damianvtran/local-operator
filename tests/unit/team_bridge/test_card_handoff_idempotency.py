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
from local_operator.team_bridge.store import (
    Store,
    StoreRefused,
    delivery_hold_class,
    state_db_path,
)
from tests.unit.session.test_session import ScriptedStream, make_session
from tests.unit.team_bridge import test_conversation_delivery as rig


class _Receiver:
    """The real receiving ``Session`` behind a synthetic process hop."""

    def __init__(self, tmp_path) -> None:
        self.session = make_session(tmp_path / "receiver", ScriptedStream([[]]))
        self.dispatched: list[str] = []  # every id that reached the hop
        self.accepted: list[str] = []  # ids the REAL gate appended as new rows
        self.accepted_text: list[str] = []  # the card text of each accepted id
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
            self.accepted_text.append(work.text)
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
    # Visible upgrade effect (R4/S-3): the owner sees it in the review class,
    # never the auto-retried delivery_pending class.
    assert delivery_hold_class("card_provenance_unknown") == "awaiting_review"
    assert delivery_hold_class("card_unconfirmed") == "delivery_pending"


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


def _dial_refused(monkeypatch):
    from local_operator.mobile.peer_client import ControlDialFailed

    async def refused(*args, **kwargs):
        raise ControlDialFailed()

    monkeypatch.setattr(launch, "engage_runtime", refused)


def _accepted_carrying(receiver, config_dir) -> set[str]:
    """Message ids named by a card the REAL receiver gate accepted as new.

    Read from the accepted card text itself (its ``message_id:`` /
    ``message_ids:`` lines), not from the sender's bookkeeping: a card
    "carries" a member only if the card the receiver took names it.
    """

    blob = "\n".join(receiver.accepted_text)
    return {mid for mid in _posted(config_dir) if mid in blob}


def _group_bound_then_split(tmp_path, monkeypatch):
    """A+B bound to X; the first dispatch is a CONFIRMED non-delivery."""

    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    monkeypatch.setattr(coalesce, "_persist_hold", lambda *a, **k: None)
    monkeypatch.setattr(Store, "delete_hold", lambda *a, **k: None)
    a, b = "bcmsg_abcdef0a", "bcmsg_abcdef0b"
    _reserve(owner, config_dir, [a, b])
    real_engage = receiver.engage
    _dial_refused(monkeypatch)
    assert [o["action"] for o in _flush(owner, config_dir, [(a, 1), (b, 2)])] == ["hold"]
    group_id = _handoff_ids(config_dir)[a]
    assert peer_send.valid_peer_message_id(group_id)
    assert _handoff_ids(config_dir)[b] == group_id
    monkeypatch.setattr(launch, "engage_runtime", real_engage)
    return owner, config_dir, receiver, real_engage, a, b, group_id


def _carry_counts(receiver, config_dir) -> dict[str, int]:
    """How many ACCEPTED cards name each member (the zero-LOSS/zero-DUP counter)."""

    return {mid: sum(mid in text for text in receiver.accepted_text) for mid in _posted(config_dir)}


def _no_holds(monkeypatch):
    monkeypatch.setattr(coalesce, "_persist_hold", lambda *a, **k: None)
    monkeypatch.setattr(Store, "delete_hold", lambda *a, **k: None)


def test_subset_replay_after_refused_dial_widens_to_the_original_card(tmp_path, monkeypatch):
    """R10 shape, r2 R5/S-4/Q2 remedy: B re-drives alone after X's dial was
    refused. No row is rebound: X is re-sent WIDENED to its original {A, B}
    card, so the one accepted card carries both and A's later replay is a
    truthful duplicate."""

    owner, config_dir, receiver, _engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )
    assert [o["action"] for o in _flush(owner, config_dir, [(b, 2)])] == ["delivered"]
    assert _handoff_ids(config_dir) == {a: group_id, b: group_id}, "nothing rekeyed"
    assert receiver.accepted == [group_id]
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}
    # Labels are not in the (contentless) record; the rebuild takes them live.
    assert "· room · from Ada (human" in receiver.accepted_text[0]
    assert f"message_ids: {a},{b}" in receiver.accepted_text[0]
    _flush(owner, config_dir, [(a, 1)])
    assert receiver.dispatched == [group_id, group_id]
    assert receiver.accepted == [group_id]
    assert _posted(config_dir) == {a: True, b: True}


@pytest.mark.parametrize("fault", ["refused", "killed"])
@pytest.mark.parametrize("order", ["b_then_a", "a_then_b"])
def test_committed_group_marker_fault_then_split_redrive_posts_one_card(
    tmp_path, monkeypatch, fault, order
):
    """DUP-1 (both markers refused) and DUP-2 (a kill before any marker): X{A,B}
    was ACCEPTED, then the members re-drive one at a time. Exactly one card
    is ever accepted and it names A and B exactly once."""

    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    _no_holds(monkeypatch)
    a, b = "bcmsg_abcdef0a", "bcmsg_abcdef0b"
    _reserve(owner, config_dir, [a, b])
    real_mark = Store.mark_body_card_posted
    if fault == "refused":
        _fail_marker_once(monkeypatch)
        assert [o["action"] for o in _flush(owner, config_dir, [(a, 1), (b, 2)])] == ["delivered"]
    else:
        killed: list[str] = []

        def kill_once(self, owner_, conversation_id, message_id, *, now):
            if not killed:
                killed.append(message_id)
                raise KeyboardInterrupt()
            return real_mark(self, owner_, conversation_id, message_id, now=now)

        monkeypatch.setattr(Store, "mark_body_card_posted", kill_once)
        with pytest.raises(KeyboardInterrupt):
            _flush(owner, config_dir, [(a, 1), (b, 2)])
    group_id = _handoff_ids(config_dir)[a]
    assert receiver.accepted == [group_id]
    assert _posted(config_dir) == {a: False, b: False}
    for mid, seq in [(b, 2), (a, 1)] if order == "b_then_a" else [(a, 1), (b, 2)]:
        _flush(owner, config_dir, [(mid, seq)])
    assert receiver.accepted == [group_id], "no second card, no replacement id"
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}
    assert _posted(config_dir) == {a: True, b: True}
    assert set(_handoff_ids(config_dir).values()) == {group_id}


def test_unconfirmed_replay_then_split_redrive_mints_nothing(tmp_path, monkeypatch):
    """DUP-3: X dial-refused, the A+B replay lands but the sender sees a timeout
    (held ``card_unconfirmed``); then B alone, then A alone. Still one card."""

    owner, config_dir, receiver, real_engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )

    async def lands_then_timeout(*args, **kwargs):
        await real_engage(*args, **kwargs)
        raise TimeoutError()

    monkeypatch.setattr(launch, "engage_runtime", lands_then_timeout)
    _flush(owner, config_dir, [(a, 1), (b, 2)])
    assert receiver.accepted == [group_id]
    assert _posted(config_dir) == {a: False, b: False}, "unknown stays owed"
    monkeypatch.setattr(launch, "engage_runtime", real_engage)
    _flush(owner, config_dir, [(b, 2)])
    _flush(owner, config_dir, [(a, 1)])
    assert receiver.accepted == [group_id]
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}
    assert _posted(config_dir) == {a: True, b: True}
    assert set(_handoff_ids(config_dir).values()) == {group_id}


@pytest.mark.parametrize("answer", ["queued", "timeout"])
def test_uncertain_subset_replay_then_full_retry_still_one_card(tmp_path, monkeypatch, answer):
    """Reviewer R1 / security repro A (loss side): B alone re-drives after a
    refused dial, lands, and the sender sees queued/timeout; then A+B retry.
    The subset was widened, so the card that landed already carried A."""

    owner, config_dir, receiver, real_engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )

    async def lands_then_uncertain(*args, **kwargs):
        result = await real_engage(*args, **kwargs)
        if answer == "timeout":
            raise TimeoutError()
        return SimpleNamespace(detail=PeerAck("queued", {"queued": True}), state=result.state)

    monkeypatch.setattr(launch, "engage_runtime", lands_then_uncertain)
    _flush(owner, config_dir, [(b, 2)])
    assert receiver.accepted == [group_id]
    monkeypatch.setattr(launch, "engage_runtime", real_engage)
    _flush(owner, config_dir, [(a, 1), (b, 2)])
    assert receiver.accepted == [group_id]
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}
    assert _posted(config_dir) == {a: True, b: True}


def test_new_member_is_not_held_behind_an_owed_group(tmp_path, monkeypatch):
    """A new C arriving with B (A+B's id still owed, A absent) gets its own
    fresh card at once, and B's partition is widened rather than waiting for
    A to co-buffer: two cards, each member named exactly once."""

    owner, config_dir, receiver, _engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )
    c = "bcmsg_abcdef0c"
    _reserve_one(owner, config_dir, c, 3)
    outcomes = _flush(owner, config_dir, [(b, 2), (c, 3)])
    assert [o["action"] for o in outcomes] == ["delivered", "delivered"]
    c_id = _handoff_ids(config_dir)[c]
    assert c_id != group_id and peer_send.valid_peer_message_id(c_id)
    assert sorted(receiver.accepted) == sorted([group_id, c_id])
    assert _posted(config_dir) == {a: False, b: True, c: True}
    _flush(owner, config_dir, [(a, 1)])
    assert _posted(config_dir)[a] is True
    assert len(receiver.accepted) == 2
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1, c: 1}


@pytest.mark.parametrize("fault", ["refused", "killed"])
def test_no_post_send_write_can_strand_a_sibling(tmp_path, monkeypatch, fault):
    """QA Q1 / reviewer R2: a refused write or a kill right after the widened
    accept of a re-driven subset. A is carried by that card; the restart's
    A+B retry is a duplicate, never a second card."""

    owner, config_dir, receiver, _engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )
    assert not hasattr(Store, "narrow_card_handoff")
    real_mark = Store.mark_body_card_posted
    hit: list[str] = []

    def faulty(self, owner_, conversation_id, message_id, *, now):
        if not hit:
            hit.append(message_id)
            if fault == "killed":
                raise KeyboardInterrupt()
            raise StoreRefused("store_refused")
        return real_mark(self, owner_, conversation_id, message_id, now=now)

    monkeypatch.setattr(Store, "mark_body_card_posted", faulty)
    if fault == "killed":
        with pytest.raises(KeyboardInterrupt):
            _flush(owner, config_dir, [(b, 2)])
    else:
        _flush(owner, config_dir, [(b, 2)])
    assert hit == [b] and receiver.accepted == [group_id]
    # "Restart": the held group re-drives together.
    _flush(owner, config_dir, [(a, 1), (b, 2)])
    assert _posted(config_dir) == {a: True, b: True}
    assert receiver.accepted == [group_id]
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}


def _reserve_one(owner, config_dir, mid, seq):
    Store(config_dir).record_body_reservation(
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


@pytest.mark.parametrize("committed", [True, False])
def test_twenty_member_group_redriven_in_retry_batches_never_stalls(
    tmp_path, monkeypatch, committed
):
    """A coalesced group holds up to 20 messages; a retry pass re-drives 8, so a
    held group can only come back in pieces. Each piece must settle at once
    from the stored card (no waiting for the group to re-form) and the group
    must still be exactly one accepted card naming every member once."""

    from local_operator.team_bridge.connector.loop import RETRY_BATCH_LIMIT

    assert coalesce.MAX_COALESCED_MESSAGES == 20 and RETRY_BATCH_LIMIT == 8
    owner, config_dir, _server, receiver = _rig(tmp_path, monkeypatch)
    _no_holds(monkeypatch)
    ids = _reserve(owner, config_dir, [f"bcmsg_abcdef{n:02x}" for n in range(20)])
    items = [(mid, seq) for seq, mid in enumerate(ids, start=1)]
    real_engage = receiver.engage
    if committed:
        _fail_marker_once(monkeypatch)  # accepted, every marker refused
    else:
        _dial_refused(monkeypatch)  # confirmed non-delivery
    _flush(owner, config_dir, items)
    monkeypatch.setattr(launch, "engage_runtime", real_engage)
    group_id = _handoff_ids(config_dir)[ids[0]]
    assert receiver.accepted == ([group_id] if committed else [])
    assert not any(_posted(config_dir).values())
    for start in range(0, 20, RETRY_BATCH_LIMIT):
        batch = items[start : start + RETRY_BATCH_LIMIT]
        assert [o["action"] for o in _flush(owner, config_dir, batch)] == ["delivered"]
    assert receiver.accepted == [group_id]
    assert set(_carry_counts(receiver, config_dir).values()) == {1}
    assert all(_posted(config_dir).values())
    assert set(_handoff_ids(config_dir).values()) == {group_id}


@pytest.mark.parametrize("column", ["credential_id", "enrollment_generation", "designated_session"])
def test_widening_refuses_a_sibling_outside_current_authority(tmp_path, monkeypatch, column):
    """Widening may not carry a still-owed sibling this owner no longer holds
    authority for: the whole card is held, nothing is dispatched or posted."""

    owner, config_dir, receiver, _engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )
    value = "sess_other" if column == "designated_session" else 999
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute(f"UPDATE body_reservations SET {column} = ? WHERE message_id = ?", (value, a))
    assert [o["action"] for o in _flush(owner, config_dir, [(b, 2)])] == ["hold"]
    assert receiver.dispatched == [] and receiver.accepted == []
    assert _posted(config_dir) == {a: False, b: False}
    assert _handoff_ids(config_dir) == {a: group_id, b: group_id}


def test_subset_of_an_id_without_a_card_record_is_held(tmp_path, monkeypatch):
    """A row bound before the record existed: a strict subset is held
    (``card_provenance_unknown``, zero sends); the whole owed set still replays
    the same id with the live card."""

    owner, config_dir, receiver, _engage, a, b, group_id = _group_bound_then_split(
        tmp_path, monkeypatch
    )
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE body_reservations SET card_handoff_card = NULL")
    with pytest.raises(StoreRefused) as held:
        asyncio.run(
            conversation._deliver_card(
                owner, _metadata(b, 2), session_id="sess_a", config_dir=config_dir
            )
        )
    assert held.value.code == "card_provenance_unknown"
    assert receiver.dispatched == []
    _flush(owner, config_dir, [(a, 1), (b, 2)])
    assert receiver.accepted == [group_id]
    assert _carry_counts(receiver, config_dir) == {a: 1, b: 1}


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


def test_a_rebuilt_card_takes_its_routing_from_the_live_metadata():
    """Merge of #4258: routing is the CURRENT delivery decision, not presentation.

    The caller re-reads the stored route on every replay and passes the matching
    session, so a record minted for a routed card that was since handed back to
    the designated session must rebuild as an ordinary card (and vice versa). The
    record never stores the routing fields, which also keeps session ids out of it.
    """

    from dataclasses import replace

    routed = replace(
        _metadata("bcmsg_abcdef0f", 1),
        routed=True,
        routed_ask_ref="bcmsg_ask00001",
        routed_via_thread=True,
        route_candidates=(("sess_other", 3, None),),
    )
    record = conversation._card_record(routed)
    for name in ("routed", "routed_ask_ref", "routed_via_thread", "route_candidates"):
        assert name not in record
    plain = _metadata("bcmsg_abcdef0f", 1)
    rebuilt = conversation._card_from_record(record, plain, ["bcmsg_abcdef0f"])
    assert (rebuilt.routed, rebuilt.routed_ask_ref, rebuilt.route_candidates) == (False, None, ())
    again = conversation._card_from_record(record, routed, ["bcmsg_abcdef0f"])
    assert again == routed


def test_agent_display_refresh_replay_keeps_original_handoff_and_no_stored_label(
    tmp_path, monkeypatch
):
    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    server.sender = {"account_id": "acc_peer", "kind": "agent", "display_label": "Endpoint (Owner)"}
    _fail_marker_once(monkeypatch)
    original = rig.FakeWire.conversation_body
    fetches = []

    async def body(self, cid, mid, **kwargs):
        fetches.append(bool(kwargs.get("expected_account_id")))
        result = await original(self, cid, mid, **kwargs)
        result["agent_session"] = {"display_label": "EPHEMERAL_TASK_CANARY"}
        return result

    monkeypatch.setattr(rig.FakeWire, "conversation_body", body)
    assert _handle(owner, config_dir, server)["action"] == "delivered"
    bound = _handoff_ids(config_dir)[rig.MESSAGE_ID]
    assert _handle(owner, config_dir, server)["action"] == "delivered"
    assert receiver.dispatched == [bound, bound]
    assert receiver.accepted == [bound]
    assert "EPHEMERAL_TASK_CANARY" in receiver.accepted_text[0]
    assert fetches == [False, True, False, True]
    assert all("EPHEMERAL_TASK_CANARY" not in str(row) for row in rig._reservation_rows(config_dir))
    before_duplicate = list(fetches)
    assert _handle(owner, config_dir, server)["action"] == "duplicate"
    assert not any(fetches[len(before_duplicate) :])  # no post-allow read for a duplicate
    assert receiver.dispatched == [bound, bound]


def test_coalesced_agent_session_names_are_presentation_only():
    from dataclasses import replace

    first = conversation.ConversationCardMetadata(
        conversation_id=rig.CONVERSATION_ID,
        tenant_id="tnt_1",
        conversation_label="channel",
        sender_label="Endpoint (Owner)",
        sender_kind="agent",
        mention=True,
        message_id="bcmsg_first001",
        conversation_seq=1,
        session_agent_display="First task",
    )
    second = replace(
        first, message_id="bcmsg_second01", conversation_seq=2, session_agent_display="Second task"
    )
    merged = coalesce.merge_metadata([first, second])
    assert merged.session_agent_display is None
    assert merged.sender_label == "several members"
    assert merged.coalesced_message_ids == (first.message_id, second.message_id)
    assert merged.message_id == second.message_id
    record = conversation._card_record(merged)
    assert "First task" not in record and "Second task" not in record
    assert "session_agent_display" not in record
    assert (
        conversation._card_from_record(record, second, [first.message_id]).session_agent_display
        is None
    )


@pytest.mark.parametrize("phase", ["initial", "owed"])
@pytest.mark.parametrize("failure", ["403", "typed404", "body", "hash", "id", "source", "owner"])
def test_agent_display_denial_never_wakes_initial_or_owed(tmp_path, monkeypatch, phase, failure):
    owner, config_dir, server, receiver = _rig(tmp_path, monkeypatch)
    server.sender = {"account_id": "acc_peer", "kind": "agent", "display_label": "Endpoint (Owner)"}
    original = rig.FakeWire.conversation_body
    reject = phase == "initial"
    refreshes = []

    async def body(self, cid, mid, **kwargs):
        value = await original(self, cid, mid, **kwargs)
        if not kwargs.get("expected_account_id"):
            return value
        refreshes.append(mid)
        if reject:
            if failure in ("403", "typed404"):
                raise conversation.BridgeRefusal(
                    "agent_not_permitted" if failure == "403" else "message_not_found",
                    "fixture",
                    http_status=403 if failure == "403" else 404,
                    server_code_present=True,
                )
            if failure == "body":
                value["body"] += "changed"
            if failure == "hash":
                value["body_hash"] = "0" * 64
            if failure == "id":
                value["message_id"] = "bcmsg_other001"
            if failure == "source":
                self._store = object()
            if failure == "owner":
                Store(config_dir).invalidate(owner)
        value["agent_session"] = {"display_label": "Authorized task"}
        return value

    monkeypatch.setattr(rig.FakeWire, "conversation_body", body)
    if phase == "owed":
        _fail_marker_once(monkeypatch)
        assert _handle(owner, config_dir, server)["action"] == "delivered"
        reject = True
    dispatched = list(receiver.dispatched)
    assert _handle(owner, config_dir, server)["action"] == "hold"
    assert receiver.dispatched == dispatched
    assert len(refreshes) == (1 if phase == "initial" else 2)
