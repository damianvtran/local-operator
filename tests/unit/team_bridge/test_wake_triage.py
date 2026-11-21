"""Channel-wake triage, ambient dwell and the urgent failover (PR "ambient dwell").

Operator, 2026-11-21: "When a message drops in a channel, all agents seem to always
respond ... find ways to make that more token efficient"; "if something seems urgent
it should be responded to immediately but not by all agents somehow if possible".
These tests pin the deterministic triage (``conversation.classify_wake``), the
``deferred`` reservation state that keeps a non-waking row out of every turn path,
the coalescer's dwell window, the failover slot and answered check, the deferred
sweep, and the ``off``/``shadow`` modes that must reproduce the pre-triage wakes.

This module is EXEMPT from the conftest's ``ambient_triage = off`` pin: it reads
the real config file (``config.yml``) the connector reads.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge.connector import coalesce as coalesce_mod
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector.conversation import (
    ConversationCardMetadata,
    WakeDecision,
    classify_wake,
    promote_staggered,
    urgent_slot,
)
from local_operator.team_bridge.settings import (
    DEFAULT_AMBIENT_TRIAGE,
    DEFAULT_AMBIENT_WAKE_DWELL_SECONDS,
    DEFAULT_URGENT_FAILOVER_SECONDS,
    TeamBridgeSettings,
)
from local_operator.team_bridge.store import (
    CARD_SETTLED,
    DEFERRED_RESERVATION_TTL_S,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import ConversationFeedRef
from tests.unit.team_bridge.test_body_reservations import (
    CONVERSATION_ID as RES_CONVERSATION,
)
from tests.unit.team_bridge.test_body_reservations import DIGEST, FINGERPRINT
from tests.unit.team_bridge.test_body_reservations import _owner as _res_owner
from tests.unit.team_bridge.test_body_reservations import _store
from tests.unit.team_bridge.test_conversation_delivery import (
    CONVERSATION_ID,
    MESSAGE_ID,
    SENDER,
    FakeServer,
)
from tests.unit.team_bridge.test_conversation_delivery import _enroll as _enroll_base
from tests.unit.team_bridge.test_conversation_delivery import (
    _owner,
    _reservation_rows,
    _run,
    _wire_rig,
)


@pytest.fixture(autouse=True)
def _fresh_triage_memory() -> None:
    # The roster cache, thread record and stagger senders are process-wide.
    conversation_mod._ROSTERS.clear()
    conversation_mod._THREAD_SEEN.clear()
    conversation_mod._STAGGER_SENDER.clear()
    coalesce_mod.CONNECTOR_COALESCER._groups.clear()
    coalesce_mod.CONNECTOR_COALESCER._last_wake.clear()


def _enroll(config_dir: Path, owner: Any, session: str, **team_bridge: Any) -> Store:
    store = _enroll_base(config_dir, owner, session)
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True, **team_bridge}}}),
        encoding="utf-8",
    )
    return store


def _reserve(store: Store, message_id: str, *, wake_class: str | None, now: float) -> bool:
    return store.record_body_reservation(
        _res_owner(),
        conversation_id=RES_CONVERSATION,
        message_id=message_id,
        conversation_seq=3,
        mention=False,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=3,
        consent_version="bridge_consent_v1",
        designated_session="sess_cos",
        now=now,
        wake_class=wake_class,
    )


# -- store: the deferred state ------------------------------------------------


@pytest.mark.parametrize("wake_class", ["passive", "ambient", "staggered"])
def test_a_deferred_row_is_never_drainable_by_this_or_an_older_build(tmp_path, wake_class) -> None:
    """The passive leak: a non-waking row used to sit ``pending`` and be drained.

    The session's follow-up re-arm served every ``pending`` row as a full turn, so a
    CARD_PASSIVE hand-off still cost a turn. A deferred row must be invisible to the
    current pending lookup AND to the predicate an OLDER binary runs (rollback /
    mixed-version safety), while its card status reads settled (no replay card).
    """

    store = _store(tmp_path)
    now = time.time()
    assert _reserve(store, "bcmsg_deferred1", wake_class=wake_class, now=now)
    owner = _res_owner()
    assert store.pending_body_reservation_for(owner, "sess_cos", now=now) is None
    assert store.body_card_status(owner, RES_CONVERSATION, "bcmsg_deferred1") == CARD_SETTLED
    # The pre-triage predicate, verbatim from the previous release.
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        old = conn.execute(
            "SELECT message_id FROM body_reservations WHERE (state = 'pending' OR "
            "(state = 'claimed' AND claim_expires_at IS NOT NULL AND claim_expires_at < ?))",
            (now,),
        ).fetchall()
    assert old == []
    # Still admitted: an agent that reads it in history may answer it.
    assert (
        store.admitted_inbound_reservation_readonly(
            owner, in_reply_to_message_id="bcmsg_deferred1", designated_session="sess_cos"
        )
        is not None
    )


def test_an_unknown_wake_class_is_refused(tmp_path) -> None:
    # The column is written from a caller argument: a closed vocabulary, never text.
    with pytest.raises(StoreRefused):
        _reserve(_store(tmp_path), "bcmsg_deferred1", wake_class="loud", now=time.time())


def test_a_lapsed_claim_on_a_classed_row_never_becomes_a_turn(tmp_path) -> None:
    """A later digest reader may claim a deferred row; its lapsed lease is not a turn."""

    store = _store(tmp_path)
    now = time.time()
    _reserve(store, "bcmsg_deferred1", wake_class="ambient", now=now)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "UPDATE body_reservations SET state = 'claimed', claim_expires_at = ?",
            (now - 10,),
        )
    assert store.pending_body_reservation_for(_res_owner(), "sess_cos", now=now) is None


def test_the_sweep_retires_only_day_old_deferred_rows(tmp_path) -> None:
    # Deferred rows are never served as turns here, so only the sweep bounds them.
    store = _store(tmp_path)
    now = time.time()
    _reserve(
        store, "bcmsg_old00001", wake_class="ambient", now=now - DEFERRED_RESERVATION_TTL_S - 5
    )
    _reserve(store, "bcmsg_fresh0001", wake_class="ambient", now=now)
    _reserve(store, "bcmsg_woken0001", wake_class=None, now=now - DEFERRED_RESERVATION_TTL_S - 5)
    loop_mod._LAST_DEFERRED_SWEEP.clear()
    assert loop_mod._sweep_deferred(_res_owner(), tmp_path, now=now) == 1
    rows = {r["message_id"]: r for r in _rows(tmp_path)}
    assert (rows["bcmsg_old00001"]["state"], rows["bcmsg_old00001"]["terminal_code"]) == (
        "retired",
        "ambient_not_shown",
    )
    assert rows["bcmsg_fresh0001"]["state"] == "deferred"
    assert rows["bcmsg_woken0001"]["state"] == "pending"  # a woken row is not the sweep's
    # Throttled: a second pass inside the minute does nothing, even with work due.
    assert loop_mod._sweep_deferred(_res_owner(), tmp_path, now=now + 1) == 0


def _rows(config_dir: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM body_reservations")]


# -- the triage table ---------------------------------------------------------


def _ref(**kw: Any) -> ConversationFeedRef:
    base: dict[str, Any] = {
        "conversation_id": CONVERSATION_ID,
        "message_id": MESSAGE_ID,
        "conversation_seq": 5,
        "mention": False,
        "sender_kind": "agent",
    }
    base.update(kw)
    return ConversationFeedRef(**base)


def _meta(**kw: Any) -> ConversationCardMetadata:
    base: dict[str, Any] = {
        "conversation_id": CONVERSATION_ID,
        "tenant_id": "tnt_1",
        "conversation_label": "#x",
        "sender_label": "Bot",
        "sender_kind": "agent",
        "mention": False,
        "message_id": MESSAGE_ID,
        "conversation_seq": 5,
        "conversation_kind": "channel",
    }
    base.update(kw)
    return ConversationCardMetadata(**base)


@pytest.mark.parametrize(
    ("ref_kw", "meta_kw", "body", "rule", "wake_class", "dwell"),
    [
        # Passive comes FIRST outside a DM (review r1 R1): the hand-off posts no card
        # for it, so a "direct" class would leave a drainable pending row.
        ({"sender_relation": "owner", "passive": True}, {}, "fyi", "passive", "passive", None),
        ({"passive": True}, {"routed": True}, "fyi", "passive", "passive", None),
        ({"passive": True}, {"conversation_kind": "dm"}, "fyi", "direct", None, 0.0),
        # Wake now beats everything after it, even a quiet agent post.
        ({}, {"conversation_kind": "dm"}, "fyi", "direct", None, 0.0),
        ({"mention": True}, {"mention": True}, "fyi", "direct", None, 0.0),
        ({"mention": True, "addressed": "direct"}, {"mention": True}, "fyi", "direct", None, 0.0),
        ({"addressed": "direct"}, {}, "fyi", "direct", None, 0.0),
        ({}, {"routed": True}, "fyi", "direct", None, 0.0),
        ({"sender_relation": "owner"}, {}, "fyi", "direct", None, 0.0),
        # An older hub that names neither the kind nor ``addressed``: fail open (Q2).
        ({}, {"conversation_kind": None}, "fyi", "direct", None, 0.0),
        # ...but a stated ``addressed`` proves a conversation kind on the hub.
        ({"addressed": "ambient"}, {"conversation_kind": None}, "fyi", "ambient", "ambient", None),
        # Rule 2: hub-marked passive is deferred even for a human's question.
        (
            {"passive": True, "sender_kind": "human"},
            {"sender_kind": "human"},
            "?",
            "passive",
            "passive",
            None,
        ),
        # Rule 3: a followed thread waits out the dwell window to batch.
        (
            {"thread_root_id": "bcmsg_root0001"},
            {"thread_follower": True},
            "?",
            "thread_follower",
            None,
            15.0,
        ),
        ({"addressed": "thread"}, {}, "fyi", "thread_follower", None, 15.0),
        # Rule 4: urgent — everyone, a human, or a question.
        ({"addressed": "everyone"}, {}, "fyi", "urgent", None, 0.0),
        ({"sender_kind": "human"}, {"sender_kind": "human"}, "fyi", "urgent", None, 0.0),
        ({}, {}, "can someone look?", "urgent", None, 0.0),
        # @channel / @here / @everyone (review r1 m3): the hub sets ``mention`` for a
        # channel-wide mention too; ``addressed`` says it is not a personal one.
        ({"mention": True, "addressed": "everyone"}, {"mention": True}, "fyi", "urgent", None, 0.0),
        # An app / integration alert (Q3): exactly one agent picks it up.
        ({"sender_kind": None}, {"sender_kind": "app"}, "ALERT: p99", "urgent", None, 0.0),
        # A body that omits sender_kind falls back to the FEED's kind (Q2).
        ({"sender_kind": "human"}, {"sender_kind": "unknown"}, "fyi", "urgent", None, 0.0),
        ({"sender_kind": "agent"}, {"sender_kind": "unknown"}, "fyi", "ambient", "ambient", None),
        # Rule 5: other ambient agent chatter is deferred.
        ({}, {}, "deployed build 41", "ambient", "ambient", None),
        ({}, {"thread_follower": False}, "done", "ambient", "ambient", None),
        # No body available (never fetched here): the "?" signal is simply absent.
        ({}, {}, None, "ambient", "ambient", None),
    ],
)
def test_classify_wake_first_match_wins(ref_kw, meta_kw, body, rule, wake_class, dwell) -> None:
    decision = classify_wake(_ref(**ref_kw), _meta(**meta_kw), body_text=body, dwell_s=15)
    assert (decision.rule, decision.wake_class, decision.dwell_s) == (rule, wake_class, dwell)


# -- dwell ----------------------------------------------------------------------


def test_dwell_batches_and_a_mention_pulls_the_group_forward(tmp_path) -> None:
    """A followed-thread reply waits ``dwell_s``; a mention in the same group goes now.

    The operator's batching ask: neighbours inside the window share one card. A DM or
    mention must never queue behind ambient chatter, so dwell 0 pulls the due time
    back to the ordinary rate floor (none here: the conversation is quiet).
    """

    coalescer = coalesce_mod.WakeCoalescer(window_s=3.0)
    owner = _owner("b", 2)
    now = 1000.0
    kw = {"ref_key": "r1", "session_id": "sess_b", "config_dir": tmp_path}
    coalescer.enqueue(owner, _meta(message_id="bcmsg_a0000001"), now=now, dwell_s=15.0, **kw)
    assert coalescer.seconds_until_due(now=now) == pytest.approx(15.0)
    # A second ambient neighbour joins the same group without extending it.
    coalescer.enqueue(
        owner,
        _meta(message_id="bcmsg_a0000002", conversation_seq=6),
        now=now + 5,
        dwell_s=15.0,
        **kw,
    )
    assert coalescer.seconds_until_due(now=now + 5) == pytest.approx(10.0)
    coalescer.enqueue(
        owner,
        _meta(message_id="bcmsg_a0000003", conversation_seq=7, mention=True),
        now=now + 6,
        dwell_s=0.0,
        **kw,
    )
    assert coalescer.seconds_until_due(now=now + 6) == 0.0
    assert len(coalescer._groups) == 1  # one card for all three
    # ``None`` (triage off) keeps the pre-dwell rate: a quiet conversation is due now.
    off = coalesce_mod.WakeCoalescer(window_s=3.0)
    off.enqueue(owner, _meta(message_id="bcmsg_a0000004"), now=now, dwell_s=None, **kw)
    assert off.seconds_until_due(now=now) == 0.0


def test_a_staggered_standby_keeps_its_own_group_and_delay(tmp_path) -> None:
    # An unrelated mention in the thread must not flush a failover standby early.
    coalescer = coalesce_mod.WakeCoalescer(window_s=3.0)
    owner = _owner("b", 2)
    kw = {"session_id": "sess_b", "config_dir": tmp_path}
    coalescer.enqueue(
        owner,
        _meta(message_id="bcmsg_s0000001"),
        ref_key="r1",
        now=0.0,
        staggered_delay_s=330.0,
        **kw,
    )
    coalescer.enqueue(
        owner,
        _meta(message_id="bcmsg_m0000001", mention=True),
        ref_key="r2",
        now=1.0,
        dwell_s=0.0,
        **kw,
    )
    dues = sorted(group.due_at for group in coalescer._groups.values())
    assert dues == [1.0, 330.0]


# -- urgent slots ---------------------------------------------------------------


def test_urgent_slots_are_stable_distinct_and_fair() -> None:
    """Every connector computes the same order alone; first-responder duty rotates.

    Distinct: for one message the agents hold slots 0..n-1 exactly once (no two wake
    first). Fair: over many messages each agent is first about 1/n of the time.
    """

    agents = ["acc_a", "acc_b", "acc_c"]
    firsts: Counter[str] = Counter()
    for index in range(3000):
        message_id = f"bcmsg_{index:08d}"
        slots = {a: urgent_slot(message_id, a, agents) for a in agents}
        assert sorted(slots.values()) == [0, 1, 2]
        assert slots == {a: urgent_slot(message_id, a, list(reversed(agents))) for a in agents}
        firsts[min(slots, key=slots.__getitem__)] += 1
    assert all(900 <= firsts[a] <= 1100 for a in agents), firsts
    # Not on the roster (stale cache, new member): never lose the wake.
    assert urgent_slot("bcmsg_00000001", "acc_new", agents) == 0


def _stagger_store(tmp_path: Path) -> tuple[Store, Any]:
    store = _store(tmp_path)
    _reserve(store, MESSAGE_ID, wake_class="staggered", now=time.time())
    return store, _res_owner()


class _Replies:
    def __init__(self, replies: list[dict[str, Any]] | Exception) -> None:
        self.replies = replies
        self.calls = 0

    async def conversation_replies(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        if isinstance(self.replies, Exception):
            raise self.replies
        return {"replies": self.replies}


def _stagger_meta() -> ConversationCardMetadata:
    return _meta(conversation_id=RES_CONVERSATION, conversation_seq=3)


def test_a_reply_the_connector_already_saw_stands_the_standby_down(tmp_path) -> None:
    # The free local record answers first: no hub read is spent.
    store, owner = _stagger_store(tmp_path)
    conversation_mod._STAGGER_SENDER[(RES_CONVERSATION, MESSAGE_ID)] = ("acc_human", "human")
    reply = _ref(
        conversation_id=RES_CONVERSATION,
        message_id="bcmsg_reply001",
        conversation_seq=4,
        thread_root_id=MESSAGE_ID,
    )
    conversation_mod._note_thread_activity(owner, reply, ("acc_other_agent", "agent"))
    client = _Replies([])
    promoted = asyncio.run(
        promote_staggered(owner, _stagger_meta(), config_dir=tmp_path, client=client)
    )
    assert promoted is False and client.calls == 0
    assert _rows(tmp_path)[0]["state"] == "deferred"


@pytest.mark.parametrize(
    ("replies", "promoted"),
    [
        pytest.param(
            [{"sender": {"account_id": "acc_other_agent", "kind": "agent"}}], False, id="answered"
        ),
        # The asker's OWN agent shares their account (agent-server gives an agent its
        # owner's account id): it is a different sender, so its reply is an answer
        # (review r1 R2 (b)).
        pytest.param(
            [{"sender": {"account_id": "acc_human", "kind": "agent"}}], False, id="owners-agent"
        ),
        # The original sender adding a line is not an answer.
        pytest.param(
            [{"sender": {"account_id": "acc_human", "kind": "human"}}], True, id="sender-only"
        ),
        pytest.param([], True, id="unanswered"),
        # Unreadable: a possible duplicate answer beats a lost urgent wake.
        pytest.param(RuntimeError("hub down"), True, id="unreadable"),
    ],
)
def test_the_answered_check_promotes_only_an_unanswered_message(
    tmp_path, replies, promoted
) -> None:
    store, owner = _stagger_store(tmp_path)
    conversation_mod._STAGGER_SENDER[(RES_CONVERSATION, MESSAGE_ID)] = ("acc_human", "human")
    client = _Replies(replies)
    assert (
        asyncio.run(promote_staggered(owner, _stagger_meta(), config_dir=tmp_path, client=client))
        is promoted
    )
    assert client.calls == 1
    row = _rows(tmp_path)[0]
    assert (row["state"], row["wake_class"]) == (
        ("pending", None) if promoted else ("deferred", "staggered")
    )


def test_a_flushed_standby_wakes_only_through_promote(tmp_path) -> None:
    """End of the failover: the flush offers the standby to ``promote`` before the
    owed check, so an answered one is dropped with no card and an unanswered one
    goes out as an ordinary card."""

    store, owner = _stagger_store(tmp_path)
    delivered: list[Any] = []

    async def deliver(_owner: Any, metadata: Any, **_kw: Any) -> str:
        delivered.append(metadata.message_id)
        return "ok"

    async def run(promote: Any) -> list[dict[str, Any]]:
        coalescer = coalesce_mod.WakeCoalescer(window_s=3.0)
        coalescer.enqueue(
            owner,
            _stagger_meta(),
            ref_key="r1",
            session_id="sess_cos",
            config_dir=tmp_path,
            now=0.0,
            staggered_delay_s=330.0,
        )
        assert await coalescer.flush_due(deliver=deliver, now=10.0, promote=promote) == []
        return await coalescer.flush_due(deliver=deliver, now=331.0, promote=promote)

    async def answered(*_a: Any, **_k: Any) -> bool:
        return False

    out = asyncio.run(run(answered))
    assert delivered == [] and out[0]["action"] == "already_posted"
    assert _rows(tmp_path)[0]["state"] == "deferred"

    async def unanswered(group_owner: Any, metadata: Any, *, config_dir: Path) -> bool:
        return store.promote_staggered_reservation(
            group_owner,
            conversation_id=metadata.conversation_id,
            message_id=metadata.message_id,
            now=time.time(),
        )

    # The answered pass settled the item WITHOUT marking a card (review r1 m1/n1):
    # nothing went out, and a mark would refuse the later promote's CAS.
    assert _rows(tmp_path)[0]["card_posted_at"] is None
    out = asyncio.run(run(unanswered))
    assert delivered == [MESSAGE_ID] and out[0]["action"] == "delivered"
    assert _rows(tmp_path)[0]["state"] == "pending"


# -- end to end through the connector iteration -------------------------------

AGENT_SENDER = {"account_id": "acc_peer_agent", "kind": "agent", "display_label": "Peer"}


def test_ambient_agent_chatter_is_admitted_but_wakes_nobody_and_replays_quietly(
    tmp_path, monkeypatch
) -> None:
    """Rule 5 through the real iteration: screened, admitted, reserved ``deferred``,
    no card; the replay (hub re-delivery / hold re-drive) is a duplicate, no card."""

    b = _owner("b", 2)
    config_dir = tmp_path / "b"
    config_dir.mkdir()
    _enroll(config_dir, b, "sess_b")
    server = FakeServer(
        notify={"bep_b": "all"},
        active={"bep_b": True},
        sender=dict(AGENT_SENDER),
        conversation_kind="channel",
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})

    out = _run(b, config_dir)

    assert [h["detail"] for h in out["handled"]] == [conversation_mod.CARD_NO_WAKE]
    assert [(r["endpoint"], r["verdict"]) for r in server.admits] == [("bep_b", "allow")]
    row = _reservation_rows(config_dir)[0]
    assert (row["state"], row["wake_class"]) == ("deferred", "ambient")
    assert transcripts["sess_b"] == [] and not out.get("coalesced_wakes")
    assert Store(config_dir).pending_body_reservation_for(b, "sess_b") is None

    replay = _run(b, config_dir)
    assert [h["action"] for h in replay["handled"]] == ["duplicate"]
    assert transcripts["sess_b"] == []
    # No card went out, so none is recorded (review r1 n1).
    assert _reservation_rows(config_dir)[0]["card_posted_at"] is None


def test_a_passive_reply_leaves_no_drainable_row_even_with_triage_off(
    tmp_path, monkeypatch
) -> None:
    # The leak fix is not a triage choice: it applies in every mode.
    b = _owner("b", 2)
    config_dir = tmp_path / "b"
    config_dir.mkdir()
    _enroll(config_dir, b, "sess_b", ambient_triage="off")
    server = FakeServer(
        notify={"bep_b": "all"},
        active={"bep_b": True},
        thread_root="bcmsg_root0001",
        passive=True,
        conversation_kind="channel",
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})

    out = _run(b, config_dir)

    assert [h["detail"] for h in out["handled"]] == [conversation_mod.CARD_PASSIVE]
    row = _reservation_rows(config_dir)[0]
    assert (row["state"], row["wake_class"]) == ("deferred", "passive")
    assert Store(config_dir).pending_body_reservation_for(b, "sess_b") is None
    assert transcripts["sess_b"] == []


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_off_and_shadow_reproduce_the_pre_triage_wake(tmp_path, monkeypatch, caplog, mode) -> None:
    """The same ambient chatter rule 5 defers wakes exactly as before in both modes;
    ``shadow`` only LOGS what ``on`` would have done (closed rule label, no content)."""

    b = _owner("b", 2)
    config_dir = tmp_path / "b"
    config_dir.mkdir()
    _enroll(config_dir, b, "sess_b", ambient_triage=mode)
    server = FakeServer(
        notify={"bep_b": "all"},
        active={"bep_b": True},
        sender=dict(AGENT_SENDER),
        conversation_kind="channel",
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (b, config_dir)})

    with caplog.at_level(logging.INFO, logger=conversation_mod.logger.name):
        out = _run(b, config_dir)

    assert out["handled"][0]["action"] == "delivered"
    assert len(transcripts["sess_b"]) == 1
    row = _reservation_rows(config_dir)[0]
    assert (row["state"], row["wake_class"]) == ("pending", None)
    shadowed = "wake triage shadow rule=ambient" in caplog.text
    assert shadowed is (mode == "shadow")


def test_a_human_post_wakes_slot_zero_and_parks_a_standby(tmp_path, monkeypatch) -> None:
    """Rule 4 end to end: the roster decides; slot 0 wakes now, slot 1 is buffered
    ``deferred``/``staggered`` for ``urgent_failover_seconds`` and is not a turn."""

    # Agent-server topology (review r1 R2): every agent member carries its OWNER's
    # account id; only the endpoint tells them apart.
    agents = ["bep_b", "bep_c", "bep_d"]
    ranks = {a: urgent_slot(MESSAGE_ID, a, agents) for a in agents}
    first = min(ranks, key=ranks.__getitem__)
    second = next(a for a in agents if ranks[a] == 1)
    outcomes: dict[str, tuple[Any, ...]] = {}
    for account in (first, second):
        name = account.removeprefix("bep_")
        owner = _owner(name, 2)
        config_dir = tmp_path / name
        config_dir.mkdir()
        _enroll(config_dir, owner, f"sess_{name}", urgent_failover_seconds=120)
        server = FakeServer(
            notify={f"bep_{name}": "all"},
            active={f"bep_{name}": True},
            sender=dict(SENDER),
            conversation_kind="channel",
        )
        monkeypatch.setattr(
            server,
            "view",
            lambda _endpoint: {
                "members": [
                    {
                        "kind": "agent",
                        "account_id": "acc_owner",
                        "endpoint_id": a,
                        "state": "active",
                    }
                    for a in agents
                ]
                + [{"kind": "human", "account_id": "acc_human", "state": "active"}]
            },
        )
        coalesce_mod.CONNECTOR_COALESCER._groups.clear()
        transcripts, _modes = _wire_rig(monkeypatch, server, {name: (owner, config_dir)})
        before = time.time()
        out = _run(owner, config_dir)
        row = _reservation_rows(config_dir)[0]
        wait = coalesce_mod.CONNECTOR_COALESCER.seconds_until_due()
        outcomes[account] = (
            len(transcripts[f"sess_{name}"]),
            row["state"],
            row["wake_class"],
            out["handled"][0]["detail"],
            None if wait is None else round(wait + (time.time() - before)),
        )
    assert outcomes[first][:3] == (1, "pending", None)
    assert outcomes[second][:4] == (0, "deferred", "staggered", coalesce_mod.CARD_COALESCED)
    assert outcomes[second][4] == pytest.approx(120, abs=2)


# -- review round 1 regressions ---------------------------------------------------


class _RosterClient:
    """The wire seams ``decide_wake`` / ``promote_staggered`` touch, counting reads."""

    def __init__(self, members: list[dict[str, Any]] | Exception) -> None:
        self.members = members
        self.view_calls = 0

    async def conversation(self, conversation_id: str, **_kw: Any) -> dict[str, Any]:
        self.view_calls += 1
        if isinstance(self.members, Exception):
            raise self.members
        return {"id": conversation_id, "members": self.members}


def _agent(endpoint: str, account: str = "acc_owner", **kw: Any) -> dict[str, Any]:
    return {
        "kind": "agent",
        "account_id": account,
        "endpoint_id": endpoint,
        "state": "active",
        **kw,
    }


def _mode_dir(tmp_path: Path, name: str = "cfg", **team_bridge: Any) -> Path:
    config_dir = tmp_path / name
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"enabled": True, **team_bridge}}}),
        encoding="utf-8",
    )
    return config_dir


def _decide(
    owner: Any,
    ref: ConversationFeedRef,
    meta: ConversationCardMetadata,
    config_dir: Path,
    client: Any,
    *,
    sender: str = "acc_human",
    now: float = 1000.0,
) -> WakeDecision:
    return asyncio.run(
        conversation_mod.decide_wake(
            owner,
            ref,
            meta,
            config_dir=config_dir,
            client=client,
            access=None,
            body_text="fyi",
            sender_account_id=sender,
            now=now,
        )
    )


_HUMAN_POST = {"sender_kind": "human"}


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
@pytest.mark.parametrize(
    ("ref_kw", "meta_kw"),
    [
        pytest.param({"sender_relation": "owner"}, {}, id="owner-reply"),
        pytest.param({}, {"routed": True}, id="routed"),
        pytest.param({"addressed": "direct"}, {}, id="addressed-direct"),
    ],
)
def test_a_passive_ref_is_deferred_before_any_wake_rule_in_every_mode(
    tmp_path, mode, ref_kw, meta_kw
) -> None:
    """R1: the owner replying in a thread this agent does not follow is a real hub
    shape (``passive`` and ``relation=owner`` are set independently). The hand-off
    posts no card for a passive ref, so a woken class would leave a pending row the
    session's re-arm later drains as a full turn."""

    config_dir = _mode_dir(tmp_path, ambient_triage=mode)
    ref = _ref(passive=True, thread_root_id="bcmsg_root0001", **ref_kw)
    decision = _decide(_owner("b", 2), ref, _meta(**meta_kw), config_dir, _RosterClient([]))
    assert (decision.rule, decision.wake_class) == ("passive", "passive")


def test_one_owners_agents_take_distinct_slots_by_endpoint(tmp_path) -> None:
    """R2 (a): the hub gives every agent its owner's account, so a single-owner team
    ranked by account put every agent in slot 0. By endpoint they split 0/1/2."""

    config_dir = _mode_dir(tmp_path)
    roster = [_agent(e) for e in ("bep_b", "bep_c", "bep_d")]
    for index in range(40):
        message_id = f"bcmsg_{index:08d}"
        slots = []
        for name in ("b", "c", "d"):
            conversation_mod._ROSTERS.clear()
            d = _decide(
                dataclasses.replace(_owner(name, 2), account_id="acc_owner"),
                _ref(message_id=message_id, **_HUMAN_POST),
                _meta(message_id=message_id, **_HUMAN_POST),
                config_dir,
                _RosterClient(roster),
            )
            slots.append(0 if d.stagger_delay_s is None else round(d.stagger_delay_s / 330))
        assert sorted(slots) == [0, 1, 2], (message_id, slots)


def test_an_agent_senders_siblings_never_leave_slot_zero_empty(tmp_path) -> None:
    """R2 + R3: an AGENT sender's endpoint is not in any view, and one owner's
    agents share its account. The sender never receives, so it must not hold slot
    0 alone: someone always wakes now, at most two do, the rest stagger."""

    config_dir = _mode_dir(tmp_path)
    roster = [_agent(e) for e in ("bep_a", "bep_b", "bep_c", "bep_d")]  # bep_a sent it
    for index in range(200):
        message_id = f"bcmsg_{index:08d}"
        now_count = 0
        # An agent's plain "fyi" is ambient chatter; its QUESTION is urgent.
        for name in ("b", "c", "d"):
            conversation_mod._ROSTERS.clear()
            d = asyncio.run(
                conversation_mod.decide_wake(
                    _owner(name, 2),
                    _ref(message_id=message_id),
                    _meta(message_id=message_id),
                    config_dir=config_dir,
                    client=_RosterClient(roster),
                    access=None,
                    body_text="can someone look?",
                    sender_account_id="acc_owner",
                    now=1000.0,
                )
            )
            now_count += d.wake_class is None
        assert 1 <= now_count <= 2, (message_id, now_count)


def test_a_rank_past_the_last_slot_still_fails_over(tmp_path) -> None:
    """R3: rank >= 3 used to be deferred for good — with a roster wider than the
    real recipients that lost the message. It now takes the LAST failover delay."""

    config_dir = _mode_dir(tmp_path, urgent_failover_seconds=100)
    endpoints = [f"bep_{c}" for c in "bcdefg"]
    roster = [_agent(e) for e in endpoints]
    message_id = next(
        f"bcmsg_{i:08d}"
        for i in range(500)
        if urgent_slot(f"bcmsg_{i:08d}", "bep_g", endpoints) >= 4
    )
    d = _decide(
        _owner("g", 2),
        _ref(message_id=message_id, **_HUMAN_POST),
        _meta(message_id=message_id, **_HUMAN_POST),
        config_dir,
        _RosterClient(roster),
    )
    assert (d.wake_class, d.stagger_delay_s) == ("staggered", 300.0)


def test_members_that_cannot_have_received_it_hold_no_slot(tmp_path) -> None:
    """R3: visible ``mute``, ``mention``-only (for a non-mention) and joined-after
    members are not recipients. Here each of them outranks ``bep_g``; excluded,
    ``bep_g`` is slot 0 and wakes now."""

    config_dir = _mode_dir(tmp_path)
    others = ["bep_m", "bep_n", "bep_j"]
    message_id = next(
        f"bcmsg_{i:08d}"
        for i in range(5000)
        if urgent_slot(f"bcmsg_{i:08d}", "bep_g", [*others, "bep_g"]) == 3
    )
    roster = [
        _agent("bep_m", notify="mute"),
        _agent("bep_n", notify="mention"),
        _agent("bep_j", joined_seq=99),
        _agent("bep_g"),
    ]
    ref = _ref(message_id=message_id, **_HUMAN_POST)
    meta = _meta(message_id=message_id, **_HUMAN_POST)
    d = _decide(_owner("g", 2), ref, meta, config_dir, _RosterClient(roster))
    assert (d.rule, d.wake_class) == ("urgent", None)
    # Control: the same members as receivers put bep_g at rank 3 (last failover).
    conversation_mod._ROSTERS.clear()
    receivers = [_agent(e) for e in (*others, "bep_g")]
    d = _decide(_owner("g", 2), ref, meta, config_dir, _RosterClient(receivers))
    assert (d.wake_class, d.stagger_delay_s) == ("staggered", 3.0 * DEFAULT_URGENT_FAILOVER_SECONDS)


def test_a_cached_roster_missing_self_is_reread_and_failures_are_cached(tmp_path) -> None:
    """R3 (refresh on a self miss) and Q4 (a failed read is remembered briefly, so a
    403 is not one failing GET per urgent message; unreadable = slot 0)."""

    config_dir = _mode_dir(tmp_path)
    owner = _owner("g", 2)
    meta = _meta(**_HUMAN_POST)
    # The roster cache is keyed per receiving endpoint; seed bep_g's with a stale one.
    conversation_mod._ROSTERS[("bep_g", CONVERSATION_ID)] = (
        1000.0,
        (conversation_mod._RosterMember("bep_x", "acc_owner"),),
    )
    fresh = _RosterClient([_agent("bep_x"), _agent("bep_g")])
    _decide(owner, _ref(**_HUMAN_POST), meta, config_dir, fresh, now=1001.0)
    assert fresh.view_calls == 1  # the self miss forced one re-read

    conversation_mod._ROSTERS.clear()
    broken = _RosterClient(RuntimeError("403"))
    first = _decide(owner, _ref(**_HUMAN_POST), meta, config_dir, broken, now=2000.0)
    second = _decide(
        owner,
        _ref(message_id="bcmsg_second01", **_HUMAN_POST),
        _meta(message_id="bcmsg_second01", **_HUMAN_POST),
        config_dir,
        broken,
        now=2030.0,
    )
    assert (first.wake_class, second.wake_class) == (None, None)  # slot 0: wakes now
    assert broken.view_calls == 1
    _decide(owner, _ref(**_HUMAN_POST), meta, config_dir, broken, now=2061.0)
    assert broken.view_calls == 2  # the negative cache expires


def test_the_roster_read_happens_only_for_urgent_messages_under_on(tmp_path) -> None:
    # Q4: the one sanctioned conversation-view read, pinned in both directions.
    owner = _owner("g", 2)
    for mode, ref_kw, body_kind, expected in [
        ("on", _HUMAN_POST, _HUMAN_POST, 1),
        ("shadow", _HUMAN_POST, _HUMAN_POST, 0),
        ("off", _HUMAN_POST, _HUMAN_POST, 0),
        ("on", {}, {}, 0),  # ambient agent chatter: no roster needed
    ]:
        conversation_mod._ROSTERS.clear()
        client = _RosterClient([_agent("bep_g")])
        _decide(
            owner,
            _ref(**ref_kw),
            _meta(**body_kind),
            _mode_dir(tmp_path, mode, ambient_triage=mode),
            client,
        )
        assert client.view_calls == expected, mode


def test_a_replay_leaves_a_parked_standby_promotable(tmp_path) -> None:
    """m1 / Q1: a replay of a staggered ref (or ``_still_owed`` on a settled item)
    used to mark ``card_posted_at`` and the promote CAS then refused: the failover
    never fired. Duplicate preferred over loss."""

    store, owner = _stagger_store(tmp_path)
    assert (
        store.mark_body_card_posted(owner, RES_CONVERSATION, MESSAGE_ID, now=time.time()) is False
    )
    assert _rows(tmp_path)[0]["card_posted_at"] is None
    assert store.promote_staggered_reservation(
        owner, conversation_id=RES_CONVERSATION, message_id=MESSAGE_ID, now=time.time()
    )
    # An ordinary (pending) row is still marked exactly as before.
    assert store.mark_body_card_posted(owner, RES_CONVERSATION, MESSAGE_ID, now=time.time())


def test_a_missing_wire_still_runs_the_failover_check(tmp_path, monkeypatch) -> None:
    # m1: a promoter whose client cannot be built used to raise, and the flush left
    # the row deferred — a lost standby. No wire = local record only, then promote.
    store, owner = _stagger_store(tmp_path)

    async def no_wire(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no credential")

    monkeypatch.setattr(conversation_mod, "_wire_for", no_wire)
    promote = loop_mod._staggered_promoter(None, None)
    assert asyncio.run(promote(owner, _stagger_meta(), config_dir=tmp_path)) is True
    assert _rows(tmp_path)[0]["state"] == "pending"


def test_two_standbys_in_one_thread_keep_their_own_due_times(tmp_path) -> None:
    # m2: the staggered group is per MESSAGE, so the second never inherits the first's due.
    coalescer = coalesce_mod.WakeCoalescer(window_s=3.0)
    owner = _owner("b", 2)
    kw = {"session_id": "sess_b", "config_dir": tmp_path}
    for message_id, now, delay in (
        ("bcmsg_s0000001", 0.0, 330.0),
        ("bcmsg_s0000002", 100.0, 660.0),
    ):
        coalescer.enqueue(
            owner,
            _meta(message_id=message_id, thread_root_id="bcmsg_root0001"),
            ref_key=message_id,
            now=now,
            staggered_delay_s=delay,
            **kw,
        )
    assert sorted(g.due_at for g in coalescer._groups.values()) == [330.0, 760.0]


def test_the_stagger_sender_memory_is_bounded() -> None:
    # m4: entries a standby never consumes must not accumulate forever.
    cap = conversation_mod._STAGGER_SENDER_CAP
    for index in range(cap + 50):
        conversation_mod._remember_stagger_sender(CONVERSATION_ID, f"m{index}", ("a", "human"))
    assert len(conversation_mod._STAGGER_SENDER) == cap
    assert (CONVERSATION_ID, "m0") not in conversation_mod._STAGGER_SENDER
    assert (CONVERSATION_ID, f"m{cap + 49}") in conversation_mod._STAGGER_SENDER


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("ambient_triage: off", "off"),
        ("ambient_triage: 'off'", "off"),
        ("ambient_triage: on", "on"),
        ("ambient_triage: 'on'", "on"),
        ("ambient_triage: shadow", "shadow"),
        ("ambient_triage: false", "off"),
    ],
)
def test_the_triage_switch_reads_unquoted_yaml(tmp_path, line, expected) -> None:
    """R4: YAML 1.1 reads a bare ``off`` as ``False``; the rollback switch must not
    fail toward ``on``. Through the same reader the connector uses."""

    from local_operator.config import read_config_values

    (tmp_path / "config.yml").write_text(
        f"values:\n  team_bridge:\n    enabled: true\n    {line}\n", encoding="utf-8"
    )
    settings = TeamBridgeSettings.from_values(read_config_values(tmp_path) or {})
    assert settings.ambient_triage == expected


# -- settings ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            {},
            (
                DEFAULT_AMBIENT_WAKE_DWELL_SECONDS,
                DEFAULT_AMBIENT_TRIAGE,
                DEFAULT_URGENT_FAILOVER_SECONDS,
            ),
        ),
        (
            {
                "ambient_wake_dwell_seconds": 0,
                "ambient_triage": "SHADOW",
                "urgent_failover_seconds": 600,
            },
            (0, "shadow", 600),
        ),
        (
            {
                "ambient_wake_dwell_seconds": 60,
                "ambient_triage": "off",
                "urgent_failover_seconds": 0,
            },
            (60, "off", 0),
        ),
        # Out of range, wrong type, bool-as-int, typo: each reads as its default.
        (
            {
                "ambient_wake_dwell_seconds": 61,
                "ambient_triage": "loud",
                "urgent_failover_seconds": 601,
            },
            (
                DEFAULT_AMBIENT_WAKE_DWELL_SECONDS,
                DEFAULT_AMBIENT_TRIAGE,
                DEFAULT_URGENT_FAILOVER_SECONDS,
            ),
        ),
        (
            {
                "ambient_wake_dwell_seconds": True,
                "ambient_triage": 1,
                "urgent_failover_seconds": "30",
            },
            (
                DEFAULT_AMBIENT_WAKE_DWELL_SECONDS,
                DEFAULT_AMBIENT_TRIAGE,
                DEFAULT_URGENT_FAILOVER_SECONDS,
            ),
        ),
        (
            {"ambient_wake_dwell_seconds": -1, "urgent_failover_seconds": 12.5},
            (
                DEFAULT_AMBIENT_WAKE_DWELL_SECONDS,
                DEFAULT_AMBIENT_TRIAGE,
                DEFAULT_URGENT_FAILOVER_SECONDS,
            ),
        ),
    ],
)
def test_triage_settings_are_validated(values, expected) -> None:
    s = TeamBridgeSettings.from_values({"team_bridge": values})
    assert (s.ambient_wake_dwell_seconds, s.ambient_triage, s.urgent_failover_seconds) == expected


def test_the_wake_decision_default_is_the_pre_triage_one() -> None:
    # ``WAKE_AS_BEFORE`` carries no class and no timing: exactly the old hand-off.
    assert conversation_mod.WAKE_AS_BEFORE == WakeDecision("triage_off")
