"""Owner review of ``write_once_conflict`` holds (security review S-4 on #4249).

Live evidence 2026-11-24: seq 3318 and 3319 each had a committed screener
ALLOW at a stale admission revision and a ``write_once_conflict`` hold under
both key spellings. Agents were told ``awaiting_review`` ("only the owner can
move this"), yet the owner's holds list offered nothing: the code was not a
review code and no decision row existed. #4533 already lets a verified owner
Deliver commit over a committed screener allow; these tests pin the list side
of that exit, and that it is offered ONLY where such a Deliver can commit.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import pytest

from local_operator.server.utils import desktop_team_bridge_holds as holds
from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.store import delivery_hold_class, state_db_path
from local_operator.team_bridge.types import ConversationFeedRef
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    CONVERSATION_ID,
    DELIVER,
    DOUBT,
    MESSAGE_ID,
    PRYING,
    Policy,
    Runner,
    _decide,
    _item,
    _screen,
    rig,
)
from tests.unit.team_bridge.test_holds_triage_release_path import (
    _envelope,
    _live,
    _republish,
    _tamper,
)

CONFLICT = "write_once_conflict"


def _composite() -> str:
    """The connector's spelling of the same hold (the screener writes the bare id)."""

    return (
        ConversationFeedRef(
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            conversation_seq=1,
            mention=False,
        )
        .validated()
        .encode()
    )


def _conflict(config_dir, store, owner, *, first: str, second: str) -> None:
    """A REAL kept conflict: ``first`` commits, the revision moves, ``second``
    disagrees. Then the connector's composite spelling is held too, as live."""

    _screen(config_dir, owner, Runner(first))
    _republish(store, owner, revision="rev-2")
    out = _screen(config_dir, owner, Runner(second), policy=_live(store, owner))
    assert out.disposition == "hold" and out.hold_code == CONFLICT
    screening._persist_hold(store, owner, _composite(), CONFLICT)


def _hold_rows(config_dir) -> list[tuple[str, str, float]]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        return [
            (str(r[0]), str(r[1]), float(r[2] or 0.0))
            for r in conn.execute("SELECT message_id, hold_code, retry_at FROM screening_holds")
        ]


def _items(config_dir, owner) -> list[dict[str, Any]]:
    return [
        i for i in holds.list_holds(config_dir, owner)["items"] if i["message_id"] == MESSAGE_ID
    ]


def _review_items(config_dir, owner) -> list[dict[str, Any]]:
    """What the owner's review list shows: the rows the person can act on."""

    return [i for i in _items(config_dir, owner) if i["actionable"]]


def test_a_stale_allow_conflict_is_one_deliver_only_review_item(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    # Two hold rows, one message: the list, the badge and the rollup count it once.
    assert sorted(code for _m, code, _r in _hold_rows(config_dir)) == [CONFLICT, CONFLICT]
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["hold_code"] == CONFLICT and decision["eligible"] == 1
    # The row names the COMMITTED screener allow the person is agreeing with.
    assert decision["screener_verdict"] == "allow" and decision["screener_provider"] == ""
    (item,) = _review_items(config_dir, owner)
    assert item["hold_code"] == CONFLICT and item["kind"] == "owner_decision"
    assert item["releasable"] and item["can_deliver"] and not item["unscreened"]
    # Deliver-only: a Discard could only ever sit refused (write-once).
    assert item["can_discard"] is False and item["discard_receipt"] is None
    # A review hold enters the long-hold rollup a day late (finding b); three
    # days on, the two spellings are still ONE held message.
    later = time.time() + 3 * 86400
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=later)["count"] == 1
    notice = hr.raise_notice(
        store, owner, config_dir=config_dir, session_id="s1", message_id=MESSAGE_ID
    )
    assert notice == "badge_only"
    # Agents keep the class they were told; it is now true.
    assert delivery_hold_class(CONFLICT) == "awaiting_review"


def test_the_owner_deliver_rearms_the_hold_and_the_next_pass_delivers(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    assert all(retry > time.time() + 3600 for _m, _c, retry in _hold_rows(config_dir))
    receipt = _decide(config_dir, owner, "deliver")
    assert receipt["state"] == "released" and receipt["outcome"] == "pending"
    # #4533: both spellings are due now, not a day out.
    assert all(retry <= time.time() for _m, _c, retry in _hold_rows(config_dir))
    out = _screen(config_dir, owner, Runner(PRYING), policy=_live(store, owner))
    assert out.disposition == "deliver" and out.receipt is not None
    assert out.receipt.template_sha256 == hr.OWNER_RELEASE_SHA256
    # The commit clears the screener's bare row; the composite one is the
    # connector's to clear on delivery (the spine test below sees both go).
    assert [m for m, _c, _r in _hold_rows(config_dir)] == [_composite()]


def test_a_discard_on_a_conflict_is_refused_before_signing(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    with pytest.raises(holds.HoldRefused) as refused:
        _decide(config_dir, owner, "discard")
    assert refused.value.code == "hold_not_releasable"
    assert store.owner_release(owner, MESSAGE_ID) is None


def test_a_reject_backed_conflict_is_not_listed_and_deliver_is_refused(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=PRYING, second=DELIVER)
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    assert _review_items(config_dir, owner) == []
    (item,) = _items(config_dir, owner)
    assert not item["releasable"] and not item["can_deliver"] and not item["can_discard"]
    with pytest.raises(holds.HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"
    # Still a terminal hold, published under the conservative class.
    assert delivery_hold_class(CONFLICT) == "awaiting_review"


def test_a_corrupt_committed_receipt_is_not_listed_or_signed(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    assert store.work_scope_decision(owner, MESSAGE_ID) is not None
    # Damage AFTER the row was written: the list and the route re-read the head.
    _tamper(config_dir, "categories", "not json")
    assert _review_items(config_dir, owner) == []
    with pytest.raises(holds.HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"
    # And damage BEFORE: no decision row is written at all.
    store.drop_review_decision(owner, MESSAGE_ID)
    epoch = int(store.current_screening_policy(owner)["epoch"])
    target = screening._review_target(_envelope(owner, _item()), Policy(), epoch, _item())
    screening._record_conflict_review(store, owner, target)
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


def test_the_predicate_binds_the_body_and_needs_a_row(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    body = _envelope(owner, _item()).body_sha256
    assert store.write_once_conflict_deliverable(owner, MESSAGE_ID, body_sha256=body) is False
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    assert store.write_once_conflict_deliverable(owner, MESSAGE_ID, body_sha256=body) is True
    assert store.write_once_conflict_deliverable(owner, MESSAGE_ID, body_sha256="0" * 64) is False
    # Other review codes need no receipt at all.
    assert hr.owner_reviewable(store, owner, "bcmsg_x", "model_hold", body_sha256="") is True


def test_a_listed_conflict_is_still_re_screened_so_a_live_allow_resolves_it(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """The decision row must NOT replay the hold before screening (unlike a
    cached answer's): #4533's fully-current preferred allow is the other exit."""

    config_dir, store, owner = rig
    _conflict(config_dir, store, owner, first=DELIVER, second=PRYING)
    assert store.work_scope_decision(owner, MESSAGE_ID) is not None
    out = _screen(config_dir, owner, Runner(DELIVER), policy=_live(store, owner))
    assert out.disposition == "deliver"
    assert out.receipt is not None and out.receipt.template_sha256 != hr.OWNER_RELEASE_SHA256


def test_other_review_codes_keep_both_words(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(DOUBT))
    assert out.disposition == "hold" and out.hold_code == "model_hold"
    (item,) = _review_items(config_dir, owner)
    assert item["can_deliver"] and item["can_discard"] and item["discard_receipt"]


def test_an_owner_deliver_on_the_real_spine_wakes_once_with_the_late_line(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    """End to end through ``handle_conversation_available``: a stale screener
    allow, a live reject → listed conflict; the owner's Deliver; ONE wake that
    carries the delivered-late line."""

    from tests.unit.team_bridge import test_hold_release as thr
    from tests.unit.team_bridge.test_holds_triage_release_path import _screener_receipt

    config_dir, store, owner = rig
    runner = thr.Runner(DOUBT)
    _server, wakes, run = thr._spine(monkeypatch, config_dir, owner, runner)
    assert run()["action"] == "hold"  # publishes the live policy; nothing committed
    store.drop_review_decision(owner, MESSAGE_ID)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("DELETE FROM screening_answer_holds")
    _screener_receipt(config_dir, store, owner)  # stale model + revision
    runner.reply = PRYING
    assert run()["action"] == "hold"
    assert {code for _m, code, _r in _hold_rows(config_dir)} == {CONFLICT}
    (item,) = _review_items(config_dir, owner)
    assert item["can_deliver"]
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE screening_holds SET created_at = created_at - ?", (3 * 3600 + 30,))
    thr._decide(config_dir, owner, "deliver")
    assert run()["action"] == "delivered"
    assert len(wakes) == 1 and wakes[0].count("delivered late") == 1
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "admitted"
    assert _hold_rows(config_dir) == []
