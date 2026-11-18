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
from local_operator.team_bridge.connector.wake import RoutedSessionGone
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


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str | None = "dm",
    live: frozenset[str] = frozenset({ASKER, OTHER, DESIGNATED}),
    gone: frozenset[str] = frozenset(),
    thread_root: str | None = None,
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
        calls.append({"session": session_id, "live_only": live_only, "payload": payload})
        if live_only and session_id in gone:
            raise RoutedSessionGone(session_id)
        transcripts.setdefault(session_id, []).append({"payload": payload})
        return "delivered and woke the session"

    monkeypatch.setattr(conversation_mod, "deliver_wake", fake_deliver)
    monkeypatch.setattr(routing, "live_capable_sessions", lambda root, fresh=False: live)
    return owner, config_dir, store, calls


def _asked(config_dir: Path, session: str, *, ago_s: float, owner: Any, message_id: str) -> None:
    record = reply_guard.record_outbound(
        config_dir,
        endpoint_id=owner.endpoint_id,
        conversation_id=CONVERSATION_ID,
        session_id=session,
        client_ref=f"c-{message_id}",
        kind="dm",
        ask=True,
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
    assert "routed to you (your ask, ref " in card
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
    assert "other sessions of yours messaged this person" in card
    assert f"session {OTHER}" in card and "asked 2 min ago" in card
    assert f"session {ASKER}" in card and "asked 10 min ago" in card
    assert "Read channel_history" in card
    assert _reservation(store, owner)["routed_session"] is None


@pytest.mark.parametrize("kind", ["group", "channel", None])
def test_a_group_or_a_hub_that_names_no_kind_never_routes(tmp_path, monkeypatch, kind) -> None:
    owner, config_dir, store, calls = _setup(tmp_path, monkeypatch, kind=kind)
    _asked(config_dir, ASKER, ago_s=30, owner=owner, message_id="bcmsg_ask00001")

    _run(owner, config_dir)

    assert [c["session"] for c in calls] == [DESIGNATED]
    assert _reservation(store, owner)["routed_session"] is None


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
