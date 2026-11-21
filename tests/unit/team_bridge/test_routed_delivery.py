"""The connector's wire-in of DM routing: which session a card is delivered to.

Driven through the real ``run_iteration`` with the seams the neighbouring
``test_conversation_delivery`` rig fakes (hub, screening, the wake itself). The
decision table lives in ``test_connector_routing``; these tests pin what the
connector DOES with it: the reservation records the target, the card says why,
a vanished target falls back, a replay never re-decides, and the pair lane is
untouched.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.team_bridge import reply_guard
from local_operator.team_bridge.connector import coalesce as coalesce_mod
from local_operator.team_bridge.connector import conversation as conversation_mod
from local_operator.team_bridge.connector import loop as loop_mod
from local_operator.team_bridge.connector import routing
from local_operator.team_bridge.connector.wake import CARD_HANDOFF, RoutedSessionGone
from tests.unit.team_bridge import test_conversation_delivery as rig
from tests.unit.team_bridge.test_conversation_delivery import (
    CONVERSATION_ID,
    MESSAGE_ID,
    FakeServer,
    _enroll,
    _owner,
    _run,
    _wire_rig,
)

PEER = "acc_peer"
ASKER = "sess_task"
OTHER = "sess_other"
DESIGNATED = "sess_b"


class RoutedServer(FakeServer):
    """The body read states the sender's account, as the hub's classification read does."""

    def body_response(self) -> dict[str, Any]:
        response = super().body_response()
        response["sender_account_id"] = PEER
        return response


@pytest.fixture(autouse=True)
def _fresh_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routing, "ROUTED_BUDGET", routing.RoutedWakeBudget())
    monkeypatch.setattr(conversation_mod, "ROUTED_BUDGET", routing.ROUTED_BUDGET)
    loop_mod._LAST_ROUTED_SWEEP.clear()


@pytest.fixture(autouse=True)
def _instant_wakes(monkeypatch: pytest.MonkeyPatch) -> None:
    """This rig's contract is "one iteration writes its card", so no window.

    ``CONNECTOR_COALESCER`` is the PROCESS's coalescer: every test in the worker
    shares it, and its rate window is keyed on the config dir's PATH. A config
    root an earlier test already woke therefore makes this test's group "not
    due"; the flush then hands the wake to nobody, and the iteration still
    reports ``detail: coalesced`` because the card WAS buffered (full-suite
    shards 1 and 3, ``test_a_group_or_a_hub_that_names_no_kind_never_routes
    [channel]``: ``calls == []`` with ``coalesced_wakes == []``). The window is a
    production behaviour with its own coverage (``test_wake_every_message``),
    and the routed gate below pins its own ``WakeCoalescer`` so it stays
    meaningful; here it is only an INPUT that can make a card late.
    ``test_conversation_delivery`` and ``test_dm_runtime_transport`` pin the same
    seam, and they pin a NEW instance per test — which is the part that matters:
    zero also flattens the seeded key, so no window state can cross a test
    boundary.
    """

    monkeypatch.setattr(
        coalesce_mod, "CONNECTOR_COALESCER", coalesce_mod.WakeCoalescer(window_s=0.0)
    )


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str | None = "dm",
    live: frozenset[str] = frozenset({ASKER, OTHER, DESIGNATED}),
    gone: frozenset[str] = frozenset(),
    thread_root: str | None = None,
    old_designated: bool = False,
):
    owner = _owner("b", 2)
    config_dir = tmp_path / "b"
    config_dir.mkdir()
    store = _enroll(config_dir, owner, DESIGNATED)
    server = RoutedServer(
        notify={"bep_b": "all"},
        active={"bep_b": True},
        conversation_kind=kind,
        thread_root=thread_root,
        sender={"account_id": PEER, "kind": "agent", "display_label": "Robin's agent"},
    )
    transcripts, _modes = _wire_rig(monkeypatch, server, {"b": (owner, config_dir)})
    calls: list[dict[str, Any]] = []

    async def fake_deliver(
        owner: Any,
        payload: str,
        *,
        session_id: str,
        config_dir: Path,
        bridge_meta: Any = None,
        sender_display: Any = None,
        live_only: bool = False,
    ) -> str:
        # The persisted card hand-off id (PR #4348) the card would carry on the
        # wire, read from the same context variable the real ``deliver_wake``
        # reads: None means no id was bound for this wake.
        handoff = CARD_HANDOFF.get()
        calls.append(
            {
                "session": session_id,
                "live_only": live_only,
                "payload": payload,
                "handoff": handoff.message_id if handoff is not None else None,
            }
        )
        if live_only and session_id in gone:
            raise RoutedSessionGone(session_id)
        transcripts.setdefault(session_id, []).append({"payload": payload})
        return "delivered and woke the session"

    monkeypatch.setattr(conversation_mod, "deliver_wake", fake_deliver)
    monkeypatch.setattr(routing, "live_capable_sessions", lambda root, fresh=False: live)
    monkeypatch.setattr(
        routing, "designated_blocks_routing", lambda root, designated: old_designated
    )
    return owner, config_dir, store, calls


def _asked(
    config_dir: Path,
    session: str,
    *,
    ago_s: float,
    owner: Any,
    message_id: str,
    ask: bool = True,
) -> None:
    record = reply_guard.record_outbound(
        config_dir,
        endpoint_id=owner.endpoint_id,
        conversation_id=CONVERSATION_ID,
        session_id=session,
        client_ref=f"c-{message_id}",
        kind="dm",
        ask=ask,
        now=time.time() - ago_s,
    )
    reply_guard.confirm_outbound(config_dir, record, message_id=message_id, seq=1)


def _reservation(store: Any, owner: Any) -> dict[str, Any]:
    row = store.body_reservation(owner, CONVERSATION_ID, MESSAGE_ID)
    assert row is not None
    return row


def test_a_reply_to_a_dm_reaches_the_session_that_asked(tmp_path, monkeypatch) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=95, owner=owner, message_id="bcmsg_ask00001")

    out = _run(owner, config_dir)

    assert [item["action"] for item in out["handled"]] == ["delivered"]
    assert out["handled"][0]["session"] == ASKER
    assert [(c["session"], c["live_only"]) for c in calls] == [(ASKER, True)]
    card = calls[0]["payload"]
    assert "routed to you (ref " in card
    assert "you posted in this DM recently and no other session of yours did" in card
    # The row is reserved for the asker; the designated session is blind to it.
    assert _reservation(store, owner)["routed_session"] == ASKER
    assert store.pending_body_reservation_for(owner, DESIGNATED) is None
    assert store.pending_body_reservation_for(owner, ASKER) is not None


def test_two_askers_go_to_the_designated_session_with_candidates_listed(
    tmp_path, monkeypatch
) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=600, owner=owner, message_id="bcmsg_ask00001")
    _asked(config_dir, OTHER, ago_s=120, owner=owner, message_id="bcmsg_ask00002")

    _run(owner, config_dir)

    assert [(c["session"], c["live_only"]) for c in calls] == [(DESIGNATED, False)]
    card = calls[0]["payload"]
    assert "other sessions of yours posted in this DM recently" in card
    assert f"session {OTHER}" in card and "posted 2 min ago" in card
    assert f"session {ASKER}" in card and "posted 10 min ago" in card
    assert "Read channel_history" in card
    assert "ask your owner before answering" in card
    assert "this person" not in card
    assert _reservation(store, owner)["routed_session"] is None


@pytest.mark.parametrize("kind", ["group", "channel", None])
def test_a_group_or_a_hub_that_names_no_kind_never_routes(tmp_path, monkeypatch, kind) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, kind=kind)
    _asked(config_dir, ASKER, ago_s=30, owner=owner, message_id="bcmsg_ask00001")

    out = _run(owner, config_dir)

    # The failure message names what the run did when no wake was made (claimed
    # elsewhere, badge-only, a hold): an empty ``calls`` alone says nothing.
    assert [c["session"] for c in calls] == [DESIGNATED], out
    assert _reservation(store, owner)["routed_session"] is None


def test_a_rate_window_from_an_earlier_test_never_makes_this_card_late(
    tmp_path, monkeypatch
) -> None:
    """A config root an earlier test already woke must not delay this card.

    The CI failure was a wake that said ``coalesced`` and woke nobody: the
    worker's coalescer held this path's last-wake instant, so the group was not
    due when the iteration flushed. Seed exactly that state — a non-routed kind,
    because a routed destination's own 10 s gate is a production behaviour — and
    the wake must still go out. Without ``_instant_wakes`` this fails with the
    CI signature (``calls == []``, ``coalesced_wakes == []``).
    """

    from types import SimpleNamespace

    owner, config_dir, _store, calls = _setup(tmp_path, monkeypatch, kind="channel")
    _asked(config_dir, ASKER, ago_s=30, owner=owner, message_id="bcmsg_ask00001")
    card_facts = SimpleNamespace(
        conversation_id=CONVERSATION_ID,
        thread_root_id=None,
        message_id=MESSAGE_ID,
        conversation_seq=1,
        mention=False,
    )
    key = coalesce_mod._group_key(config_dir, owner, card_facts, DESIGNATED)
    coalesce_mod.CONNECTOR_COALESCER._last_wake[key] = time.time()

    out = _run(owner, config_dir)

    assert [c["session"] for c in calls] == [DESIGNATED], out


def test_the_pair_lane_never_meets_the_router() -> None:
    """``send``, requests and approvals ride ``handle_message_available`` (the pair
    inbox), which reads designated-only state. The router, the routed column and
    the ``live_only`` wake seam are reachable only from the conversation lane."""

    import inspect

    from local_operator.team_bridge.connector import wake

    pair_lane = inspect.getsource(wake.handle_message_available)
    for name in ("plan_route", "routed_session", "live_only", "routing"):
        assert name not in pair_lane, name
    # And the conversation lane is the ONLY caller of the planner.
    assert "plan_route" in inspect.getsource(conversation_mod.handle_conversation_available)


def test_a_routed_card_binds_its_hand_off_id_against_the_designation(tmp_path, monkeypatch) -> None:
    """Merge of #4258 into PR #4348: a routed row keeps ``designated_session``.

    The bind fences rows against the designation; passing the routed TARGET
    would mismatch the row and hold the card forever (``card_binding_mismatch``).
    """

    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=95, owner=owner, message_id="bcmsg_ask00001")

    out = _run(owner, config_dir)

    assert [item["action"] for item in out["handled"]] == ["delivered"]
    assert [c["session"] for c in calls] == [ASKER]
    row = _reservation(store, owner)
    assert row["routed_session"] == ASKER
    assert row["designated_session"] == DESIGNATED
    assert calls[0]["handoff"] is not None
    assert calls[0]["handoff"] == row["card_handoff_id"]


def test_the_designated_fallback_reuses_the_routed_attempts_hand_off_id(
    tmp_path, monkeypatch
) -> None:
    """The routed attempt delivered nothing, so the fallback is the SAME card.

    It must carry the id bound before the routed attempt (not a fresh one, not
    none), so a replay after a lost posted marker is deduplicated by the
    designated session's receiver.
    """

    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, gone=frozenset({ASKER}))
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    assert [c["session"] for c in calls] == [ASKER, DESIGNATED]
    bound = _reservation(store, owner)["card_handoff_id"]
    assert bound and calls[0]["handoff"] == calls[1]["handoff"] == bound


def test_a_target_that_vanished_before_the_wake_falls_back_to_the_designated_session(
    tmp_path, monkeypatch
) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, gone=frozenset({ASKER}))
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    # First the routed attempt (refused: nothing delivered), then the designated one.
    assert [(c["session"], c["live_only"]) for c in calls] == [(ASKER, True), (DESIGNATED, False)]
    assert "routed to you" not in calls[1]["payload"]
    assert _reservation(store, owner)["routed_session"] is None
    assert store.pending_body_reservation_for(owner, DESIGNATED) is not None


def test_a_session_that_is_not_live_is_never_chosen(tmp_path, monkeypatch) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, live=frozenset({DESIGNATED}))
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    assert [c["session"] for c in calls] == [DESIGNATED]


def test_a_replay_takes_the_target_from_the_stored_row_and_never_recomputes(
    tmp_path, monkeypatch
) -> None:
    owner, config_dir, store, _calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")
    _run(owner, config_dir)
    assert _reservation(store, owner)["routed_session"] == ASKER

    # The ledger moves on (a second session now posted) — a recompute would change.
    _asked(config_dir, OTHER, ago_s=1, owner=owner, message_id="bcmsg_ask00002")
    ref = conversation_mod.ConversationFeedRef(CONVERSATION_ID, MESSAGE_ID, 1, False)
    plan = conversation_mod._stored_route(store, owner, ref, config_dir)

    assert plan.target == ASKER and plan.reason == "replay"


def test_over_budget_goes_to_the_designated_session_and_is_kept(tmp_path, monkeypatch) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")
    for _ in range(routing.ROUTED_WAKES_PER_HOUR):
        routing.ROUTED_BUDGET.charge(ASKER)

    _run(owner, config_dir)

    assert [c["session"] for c in calls] == [DESIGNATED]
    assert _reservation(store, owner)["state"] == "pending"  # kept, not held or dropped


# -- the coalescer's per-destination gate ------------------------------------------


def test_a_routed_group_waits_ten_seconds_after_its_last_wake_an_ordinary_one_three(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    coalescer = coalesce_mod.WakeCoalescer(window_s=3.0)
    owner = _owner("b", 2)
    meta = SimpleNamespace(
        conversation_id=CONVERSATION_ID,
        thread_root_id=None,
        message_id="bcmsg_aaaaaaa1",
        conversation_seq=1,
        mention=False,
    )
    key = coalesce_mod._group_key(tmp_path, owner, meta, ASKER)
    coalescer._last_wake[key] = 100.0

    coalescer._groups.clear()
    coalescer.enqueue(owner, meta, ref_key="r1", session_id=ASKER, config_dir=tmp_path, now=101.0)
    assert coalescer._groups[key].due_at == 103.0  # ordinary: last + 3 s

    coalescer._groups.clear()
    coalescer.enqueue(
        owner,
        meta,
        ref_key="r1",
        session_id=ASKER,
        config_dir=tmp_path,
        now=101.0,
        min_interval_s=routing.ROUTED_WAKE_MIN_INTERVAL_S,
    )
    assert coalescer._groups[key].due_at == 110.0  # routed: last + 10 s


def test_the_budget_counts_per_session_per_hour() -> None:
    budget = routing.RoutedWakeBudget(per_hour=2)
    budget.charge("s", now=0.0)
    budget.charge("s", now=10.0)
    assert not budget.has_room("s", now=20.0)
    assert budget.has_room("other", now=20.0)
    assert budget.has_room("s", now=3601.0)  # the first one aged out of the hour


# -- the loop sweep ---------------------------------------------------------------------


def test_the_sweep_hands_a_stale_routed_row_back_and_is_throttled(tmp_path, monkeypatch) -> None:
    owner, config_dir, store, _calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")
    _run(owner, config_dir)
    assert _reservation(store, owner)["routed_session"] == ASKER
    created = _reservation(store, owner)["created_at"]

    # Too young: untouched.
    assert loop_mod._sweep_routed(owner, config_dir, now=created + 60) == 0
    assert _reservation(store, owner)["routed_session"] == ASKER

    # Past the reclaim age (and the throttle): handed back.
    later = created + routing.ROUTED_RECLAIM_AFTER_S + 120
    assert loop_mod._sweep_routed(owner, config_dir, now=later) == 1
    assert _reservation(store, owner)["routed_session"] is None
    assert store.pending_body_reservation_for(owner, DESIGNATED) is not None

    # Throttled: a second call inside the minute does nothing at all.
    assert loop_mod._sweep_routed(owner, config_dir, now=later + 5) == 0


def test_the_sweep_leaves_a_row_a_live_turn_has_claimed(tmp_path, monkeypatch) -> None:
    owner, config_dir, store, _calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")
    _run(owner, config_dir)
    assert store.claim_body_reservation(
        owner,
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="pt:aaaa",
        turn_token="tt:bbbb",
        now=time.time(),
        ttl=10**6,
    )

    assert loop_mod._sweep_routed(owner, config_dir, now=time.time() + 7200) == 0
    assert _reservation(store, owner)["routed_session"] == ASKER


def test_the_rig_module_is_the_one_the_neighbours_use() -> None:
    assert rig.CONVERSATION_ID == CONVERSATION_ID


# -- card wording and data (review remediation) ----------------------------------------


def test_the_designated_session_is_never_listed_as_another_session(tmp_path, monkeypatch) -> None:
    owner, config_dir, _store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, DESIGNATED, ago_s=300, owner=owner, message_id="bcmsg_ask00001")
    _asked(config_dir, OTHER, ago_s=0, owner=owner, message_id="bcmsg_ask00002")

    _run(owner, config_dir)

    card = calls[0]["payload"]
    assert f"session {OTHER}, posted just now" in card or f"session {OTHER} (ref" in card
    assert "posted just now" in card and "<1 min" not in card
    assert f"session {DESIGNATED}" not in card


def test_the_ref_and_the_time_come_from_the_same_row(tmp_path, monkeypatch) -> None:
    """An ask 25 min ago, then a plain reply 3 min ago: the time is the reply's and the
    reply carries no ref, so the card shows none rather than the older ask's."""

    owner, config_dir, _store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=25 * 60, owner=owner, message_id="bcmsg_ask00001")
    older = reply_guard.lookup_authorship(
        config_dir,
        endpoint_id=owner.endpoint_id,
        conversation_id=CONVERSATION_ID,
        message_ids=["bcmsg_ask00001"],
    )["bcmsg_ask00001"][1]
    assert older
    _asked(config_dir, ASKER, ago_s=180, owner=owner, message_id="bcmsg_rep00001", ask=False)

    _run(owner, config_dir)

    card = calls[0]["payload"]
    assert card.count("routed to you") == 1 and older not in card
    assert "routed to you (ref" not in card


def test_a_thread_reply_says_so_and_names_the_ref_of_the_message_it_answers(
    tmp_path, monkeypatch
) -> None:
    """The thread route has no window and wins with two posters, so "no other session
    of yours did" would be false; the sentence is its own."""

    root = rig.THREAD_ROOT
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, thread_root=root)
    _asked(config_dir, ASKER, ago_s=600, owner=owner, message_id=root)
    _asked(config_dir, OTHER, ago_s=30, owner=owner, message_id="bcmsg_ask00002")
    asked_ref = reply_guard.lookup_authorship(
        config_dir,
        endpoint_id=owner.endpoint_id,
        conversation_id=CONVERSATION_ID,
        message_ids=[root],
    )[root][1]

    _run(owner, config_dir)

    assert [c["session"] for c in calls] == [ASKER]
    card = calls[0]["payload"]
    assert f"routed to you (ref {asked_ref})" in card
    assert "replied in a thread under a message you posted" in card
    assert "no other session of yours did" not in card
    assert _reservation(store, owner)["routed_session"] == ASKER


def test_a_routed_card_does_not_contradict_itself_and_says_how_to_bind_the_reply(
    tmp_path, monkeypatch
) -> None:
    owner, config_dir, _store, calls = _setup(tmp_path, monkeypatch)
    _asked(config_dir, ASKER, ago_s=95, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    card = calls[0]["payload"]
    assert "Answer it here" not in card
    assert "expect: reply if it answers what you posted; otherwise no_reply" in card
    assert "expect: weigh relevance" not in card
    assert "in_reply_to_message_id = this message_id" in card
    assert "this person" not in card


def test_an_ordinary_card_keeps_its_reply_line_unchanged(tmp_path, monkeypatch) -> None:
    owner, _config_dir, _store, calls = _setup(tmp_path, monkeypatch)

    _run(owner, _config_dir)

    card = calls[0]["payload"]
    assert "if you reply: bridge channel_post with channel_id = this conversation_id; " in card
    assert "in_reply_to_message_id" not in card


def test_an_old_designated_build_keeps_the_reply_with_the_designated_session(
    tmp_path, monkeypatch
) -> None:
    """The live designated session does not advertise the capability, so it would claim a
    row routed away from it: the decision waits for a build that knows the column."""

    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, old_designated=True)
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    assert [(c["session"], c["live_only"]) for c in calls] == [(DESIGNATED, False)]
    assert _reservation(store, owner)["routed_session"] is None


def test_a_row_the_routed_session_claimed_meanwhile_does_not_wake_the_designated_one(
    tmp_path, monkeypatch
) -> None:
    """The recheck says gone, but the compare-and-set loses: a live turn holds the row.
    The designated session cannot read that body, so it is not woken for it."""

    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, gone=frozenset({ASKER}))
    _asked(config_dir, ASKER, ago_s=40, owner=owner, message_id="bcmsg_ask00001")
    from local_operator.team_bridge.store import Store

    monkeypatch.setattr(Store, "unroute_reservation", lambda self, owner, **kw: False)

    _run(owner, config_dir)

    assert [(c["session"], c["live_only"]) for c in calls] == [(ASKER, True)]
    assert _reservation(store, owner)["routed_session"] == ASKER


def test_a_recovery_card_goes_to_the_routed_session(tmp_path, monkeypatch) -> None:
    """A retry after an image admit takes the ``recovery`` branch, which used to hand the
    card to the designated session even for a row reserved for another one.

    The recovery branch also commits the body reservation and re-proves the local
    receipt before it wakes anything. Both must hold together: the fence is asked
    about the row's OWN session, so a routed row is not refused as "unavailable"
    (which would hold this message for ever), and the stored route decides where
    the card goes.
    """

    import asyncio
    import sqlite3

    from local_operator.team_bridge.store import Store
    from tests.unit.team_bridge import test_attachments as att

    hub = att.Hub([att._png(), att._png((0, 0, 255))])
    owner, ref, _screened, _cards = att._connector_rig(
        monkeypatch, tmp_path, hub, verdict="deliver"
    )
    sessions: list[tuple[str, bool]] = []
    first = {"done": False}

    async def deliver(owner: Any, payload: str, *, session_id: str, **kwargs: Any) -> str:
        if not first["done"]:
            first["done"] = True
            raise RuntimeError("session closed mid-wake")
        sessions.append((session_id, "routed to you" in payload))
        return "delivered"

    monkeypatch.setattr(conversation_mod, "deliver_wake", deliver)
    held = asyncio.run(att._deliver(owner, ref, tmp_path, hub))
    assert held["action"] == "hold"
    with sqlite3.connect(tmp_path / "team_bridge" / "state.sqlite3") as conn:
        conn.execute("UPDATE body_reservations SET routed_session = ?", (ASKER,))
    # No faked receipt: the recovery proof compares the stored envelope digest
    # and creator generation, and only the receipt the rig's own screen wrote
    # carries them.
    out = asyncio.run(att._deliver(owner, ref, tmp_path, hub))

    assert out["action"] == "delivered" and out["recovered"] == "images_admitted"
    assert sessions == [(ASKER, True)]
    # The row the recovery found is the row it delivered: one row, still routed,
    # and re-recording it was a no-op rather than a second promise.
    store = Store(tmp_path)
    row = store.body_reservation(owner, ref.conversation_id, ref.message_id)
    assert row is not None and row["routed_session"] == ASKER and row["state"] == "pending"


def test_a_recovery_with_no_stored_row_goes_to_the_designated_session(
    tmp_path, monkeypatch
) -> None:
    """A lost admit response leaves a receipt but no reservation.

    The recovery branch must then commit the body obligation itself, before the
    wake, and the fresh row carries no stored route — so the card goes to the
    designated session. Routing never runs on a recovery: there was no decision
    to replay, and inventing one here would wake a session the admission never
    chose.
    """

    import asyncio

    from local_operator.team_bridge.store import Store
    from tests.unit.team_bridge import test_attachments as att

    hub = att.Hub([att._png()])
    owner, ref, _screened, _cards = att._connector_rig(
        monkeypatch, tmp_path, hub, verdict="deliver", preferred=True
    )
    store = Store(tmp_path)
    real_admit = att.HubWire.conversation_admit

    async def lost_response(self, *args, **kwargs):
        await real_admit(self, *args, **kwargs)
        raise RuntimeError("controlled lost admission response")

    monkeypatch.setattr(att.HubWire, "conversation_admit", lost_response)
    first = asyncio.run(att._deliver(owner, ref, tmp_path, hub))
    assert first["action"] == "hold"
    assert store.receipt_for_message(owner, ref.message_id) is not None
    assert store.body_reservation(owner, ref.conversation_id, ref.message_id) is None

    sessions: list[str] = []

    async def deliver(owner: Any, payload: str, *, session_id: str, **kwargs: Any) -> str:
        sessions.append(session_id)
        return "delivered"

    monkeypatch.setattr(conversation_mod, "deliver_wake", deliver)
    out = asyncio.run(att._deliver(owner, ref, tmp_path, hub))

    assert out["action"] == "delivered" and out["recovered"] == "images_admitted"
    assert sessions == ["sess_a"], "the designated session, never a routed one"
    row = store.body_reservation(owner, ref.conversation_id, ref.message_id)
    assert row is not None and row["routed_session"] is None and row["state"] == "pending"


# -- the wake seam: a routed delivery only ever dials ------------------------------------------


def test_a_routed_wake_dials_the_scanned_record_and_never_resolves_the_id(
    tmp_path, monkeypatch
) -> None:
    import asyncio
    from types import SimpleNamespace

    from local_operator.mobile import peer_send
    from local_operator.team_bridge.connector import wake as wake_mod

    record = SimpleNamespace(pid=1, session_id=ASKER, started=True)
    monkeypatch.setattr(routing, "live_capable_record", lambda root, session_id: record)
    seen: list[Any] = []

    async def outcome(rec: Any, **kwargs: Any) -> Any:
        seen.append(rec)
        return SimpleNamespace(state=peer_send.DELIVERY_DELIVERED, detail="ok")

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", outcome)
    # A session with no durable history would otherwise take the cold-engage path.
    monkeypatch.setattr(peer_send, "session_has_durable_history", lambda session_id: False)

    asyncio.run(
        wake_mod.deliver_wake(
            _owner("b", 2), "card", session_id=ASKER, config_dir=tmp_path, live_only=True
        )
    )

    assert seen == [record]  # the exact record the recheck scanned, never ``None``


def test_a_routed_wake_whose_dial_fails_falls_back_instead_of_holding(
    tmp_path, monkeypatch
) -> None:
    import asyncio
    from types import SimpleNamespace

    from local_operator.mobile import peer_send
    from local_operator.team_bridge.connector import wake as wake_mod

    record = SimpleNamespace(pid=1, session_id=ASKER, started=True)
    monkeypatch.setattr(routing, "live_capable_record", lambda root, session_id: record)

    async def failed(rec: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(state=peer_send.DELIVERY_FAILED, detail="dial refused")

    monkeypatch.setattr(peer_send, "deliver_peer_message_outcome", failed)

    with pytest.raises(RoutedSessionGone):
        asyncio.run(
            wake_mod.deliver_wake(
                _owner("b", 2), "card", session_id=ASKER, config_dir=tmp_path, live_only=True
            )
        )
    # And a target that is no longer in the registry never reaches the primitive at all.
    monkeypatch.setattr(routing, "live_capable_record", lambda root, session_id: None)
    with pytest.raises(RoutedSessionGone):
        asyncio.run(
            wake_mod.deliver_wake(
                _owner("b", 2), "card", session_id=ASKER, config_dir=tmp_path, live_only=True
            )
        )
