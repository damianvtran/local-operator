"""Every reviewable held message has a working Discard (operator report 2026-11-24).

Approvals -> Held messages showed an "Unknown sender ... could not be loaded"
row with no Discard and "Dismiss all" disabled. Measured read-only on the live
store: a ``work_scope_hold`` on a THREAD REPLY (its hold ref carries
``thread_root=``); the renderer joins sender/body from the owner-audit messages
page, which the hub never serves replies on, so the body could not load. And
PR #4546 had made a ``write_once_conflict`` hold Deliver-only, so a conflict
whose body could not load would have had no button at all.

These pin, for EACH listable hold class, the list contract the renderer needs
(``can_discard``, a body-free ``discard_receipt``, ``actionable``,
``thread_root_id``) and that the Discard route never delivers: it either
queues the ordinary signed Discard (which the connector turns into a bounce,
never a delivery) or settles it as an audited LOCAL terminal that touches
nothing on the hub.

The review codes other than ``model_hold`` are reached by re-coding a real
``model_hold`` decision: the list and the route read only the code, the
decision row and the hold row, and each code's own producer is pinned in
``test_review_held.py`` / ``test_hold_release.py``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.store import Store, state_db_path
from local_operator.team_bridge.types import ConversationFeedRef
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — fixtures
    CONVERSATION_ID,
    DELIVER,
    DOUBT,
    EIGHT_DAYS_S,
    HOLDS,
    MESSAGE_ID,
    NEW_WORK,
    PRYING,
    Runner,
    _age_decision,
    _body,
    _listed,
    _receipt_body,
    _screen,
    api,
    rig,
)
from tests.unit.team_bridge.test_holds_triage_release_path import _live
from tests.unit.team_bridge.test_write_once_conflict_review import _conflict

LOCAL_TERMINAL = "discarded_local_unconfirmed"
ROOT = "bcmsg_root0001"
OTHER_REVIEW_CODES = sorted(hr.REVIEW_HOLD_CODES - {"model_hold", hr.WRITE_ONCE_CONFLICT_HOLD})


def _recode(config_dir, code: str) -> None:
    """Turn the real ``model_hold`` hold + decision for MESSAGE_ID into ``code``."""

    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE screening_holds SET hold_code = ?", (code,))
        conn.execute(
            "UPDATE work_scope_decisions SET hold_code = ? WHERE message_id = ?",
            (code, MESSAGE_ID),
        )


def _reply_ref(root: str = ROOT) -> str:
    return (
        ConversationFeedRef(
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            conversation_seq=403,
            mention=True,
            sender_kind="agent",
            thread_root_id=root,
        )
        .validated()
        .encode()
    )


def _local_terminals(config_dir) -> list[dict[str, Any]]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM owner_release_local_terminals")]
        except sqlite3.OperationalError:
            return []


def _hold(config_dir, store, owner, code: str) -> None:
    """A real eligible hold of ``code`` on MESSAGE_ID."""

    if code == hr.WORK_SCOPE_HOLD:
        assert _screen(config_dir, owner, Runner(NEW_WORK)).hold_code == code
    elif code == hr.WRITE_ONCE_CONFLICT_HOLD:
        _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    else:
        assert _screen(config_dir, owner, Runner(DOUBT)).hold_code == "model_hold"
        if code != "model_hold":
            _recode(config_dir, code)


def _only(call) -> dict[str, Any]:
    (item,) = [i for i in _listed(call) if i["message_id"] == MESSAGE_ID]
    return item


@pytest.fixture
def made_due(monkeypatch) -> list[Any]:
    """Connector passes the route makes due: none for a LOCAL terminal."""

    due: list[Any] = []
    monkeypatch.setattr(Store, "make_holds_due", lambda *a, **k: due.append(a) or 0)
    return due


LISTABLE = [hr.WORK_SCOPE_HOLD, "model_hold", *OTHER_REVIEW_CODES, hr.WRITE_ONCE_CONFLICT_HOLD]


@pytest.mark.parametrize("code", LISTABLE)
def test_every_listable_class_offers_a_receipt_discard(api, code) -> None:  # noqa: F811
    config_dir, store, owner, call = api
    _hold(config_dir, store, owner, code)
    item = _only(call)
    assert item["hold_code"] == code and item["releasable"] is True
    assert item["can_discard"] is True and item["actionable"] is True
    assert isinstance(item["discard_receipt"], str) and len(item["discard_receipt"]) == 64
    assert item["can_deliver"] is (code not in hr.DISCARD_ONLY_HOLD_CODES)
    assert "thread_root_id" in item and item["thread_root_id"] is None


@pytest.mark.parametrize("code", LISTABLE)
@pytest.mark.parametrize("expired", [False, True], ids=["in_window", "expired"])
def test_a_receipt_discard_never_delivers(api, made_due, code, expired) -> None:  # noqa: F811
    config_dir, store, owner, call = api
    _hold(config_dir, store, owner, code)
    if expired:
        _age_decision(config_dir, EIGHT_DAYS_S)
    item = _only(call)
    out = call("POST", f"{HOLDS}/{MESSAGE_ID}/discard", json=_receipt_body(item["discard_receipt"]))
    assert out.status_code == 200, out.text
    result = out.json()["result"]
    assert result["state"] == "discarded"
    release = store.owner_release(owner, MESSAGE_ID)
    assert release is not None and release["word"] == "discard"
    assert hr.verify_row(release, hr.anchor_spki(config_dir)[0])
    local = expired or code in hr.LOCAL_DISCARD_HOLD_CODES
    if local:
        # Settled here: no pass made due, nothing posted, one audit row.
        assert result["outcome"] == LOCAL_TERMINAL and made_due == []
        (audit,) = _local_terminals(config_dir)
        assert audit["reason"] == (
            "write_once_conflict_committed_allow"
            if code in hr.LOCAL_DISCARD_HOLD_CODES
            else "hub_unconfirmed_after_deliver_window"
        )
    else:
        # The ordinary signed Discard, queued for the connector.
        assert result["outcome"] == "pending" and len(made_due) == 1
        assert _local_terminals(config_dir) == []
    after = _only(call)
    assert after["actionable"] is False and after["discard_receipt"] is None
    # Whatever the screener would now say, the connector never delivers it.
    policy = _live(store, owner) if code == hr.WRITE_ONCE_CONFLICT_HOLD else None
    again = _screen(config_dir, owner, Runner(DELIVER), policy=policy)
    assert again.disposition != "deliver"


def test_a_body_bound_conflict_discard_settles_locally_and_stays_held(
    api, made_due  # noqa: F811
) -> None:
    """Body on screen, in the window: still local, since the connector could
    never commit a reject over the committed allow (write-once)."""

    config_dir, store, owner, call = api
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    out = call("POST", f"{HOLDS}/{MESSAGE_ID}/discard", json=_body())
    assert out.status_code == 200, out.text
    assert out.json()["result"]["outcome"] == LOCAL_TERMINAL and made_due == []
    # A fully-current allow on a re-drive would otherwise supersede and deliver.
    out = _screen(config_dir, owner, Runner(DELIVER), policy=_live(store, owner))
    assert out.disposition == "hold" and out.hold_code == hr.WRITE_ONCE_CONFLICT_HOLD
    # The local terminal is not overwritten by the connector's refusal.
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == LOCAL_TERMINAL
    # And the other word is closed.
    deliver = call("POST", f"{HOLDS}/{MESSAGE_ID}/deliver", json=_body())
    assert deliver.status_code == 409 and "already_decided" in deliver.text


def test_a_thread_reply_hold_lists_its_root_and_discards_by_receipt(api) -> None:  # noqa: F811
    """The live shape: a work-scope hold whose connector ref is a reply."""

    config_dir, store, owner, call = api
    _hold(config_dir, store, owner, hr.WORK_SCOPE_HOLD)
    screening._persist_hold(store, owner, _reply_ref(), hr.WORK_SCOPE_HOLD)
    item = _only(call)
    assert item["thread_root_id"] == ROOT and item["seq"] == 403
    assert item["can_discard"] is True and item["discard_receipt"]
    out = call("POST", f"{HOLDS}/{MESSAGE_ID}/discard", json=_receipt_body(item["discard_receipt"]))
    assert out.status_code == 200 and out.json()["result"]["state"] == "discarded"


def test_a_damaged_thread_root_is_never_listed(api) -> None:  # noqa: F811
    config_dir, store, owner, call = api
    _hold(config_dir, store, owner, hr.WORK_SCOPE_HOLD)
    damaged = _reply_ref().replace(f"thread_root={ROOT}", "thread_root=../../etc")
    assert damaged != _reply_ref()
    screening._persist_hold(store, owner, damaged, hr.WORK_SCOPE_HOLD)
    item = _only(call)
    assert item["thread_root_id"] is None


def test_a_hold_with_no_decision_lists_as_not_actionable(api) -> None:  # noqa: F811
    """The one listed shape with no Discard: nothing to sign (no decision row).

    It is never counted (``actionable`` false), so it cannot hold the badge or
    disable "Dismiss all"; the connector re-drives it on its own."""

    config_dir, store, owner, call = api
    store.persist_hold(owner, MESSAGE_ID, "retry_later")
    item = _only(call)
    assert item["can_discard"] is False and item["discard_receipt"] is None
    assert item["actionable"] is False and item["kind"] == "waiting"
