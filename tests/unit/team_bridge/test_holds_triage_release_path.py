"""Holds triage 2026-11-18: the four release-path fixes (findings a-d).

The triage of 12 unseen Team Bridge holds found four bugs on the path from a
screening hold to the person who can release it:

* (a) the per-sender budget counted REVIEW decision rows, so three ordinary
  ``model_hold`` rows from one sender wrote that sender's next genuine
  ``work_scope_hold`` ineligible (H12: no Deliver/Discard);
* (b) review holds were left out of the long-hold rollup although no holds list
  ships, so nine of them sat for up to 18 days with no alert of any kind;
* (c) ``write_once_conflict`` was published to the sender as ``awaiting_review``
  and could never clear (H5: re-driven 368 times with a committed allow);
* (d) the screener's REASON was never persisted, so calibration was inference.
"""

from __future__ import annotations

import logging
import time

from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.store import (
    SCREENER_REASON_MAX,
    StoreRefused,
    capped_screener_reason,
    delivery_hold_class,
)
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    DELIVER,
    DOUBT,
    MESSAGE_ID,
    NEW_WORK,
    Runner,
    _item,
    _screen,
    rig,
)

SENDER_DIGEST = hr.sender_digest("acc_sender")


def _decision(store, owner, message_id: str, *, code: str, now: float, reason=None):
    return store.record_work_scope_decision(
        owner,
        message_id,
        conversation_id="bcv_abcdefgh",
        body_sha256="b" * 64,
        envelope_digest="e" * 64,
        screener_provider="anthropic",
        screener_model="claude-sonnet-5-5",
        template_sha256="t" * 64,
        prompt_version="bridge-screen-v2",
        categories=("new_work_request",) if code == hr.WORK_SCOPE_HOLD else (),
        policy_epoch=1,
        sender_digest=SENDER_DIGEST,
        now=now,
        screener_verdict="bounce" if code == hr.WORK_SCOPE_HOLD else "hold",
        screener_hop=0,
        screener_preferred=True,
        hold_code=code,
        screener_reason=reason,
    )


# --- (a) the sender budget counts only work-scope holds ---------------------------


def test_review_holds_do_not_spend_the_sender_budget(rig) -> None:  # noqa: F811 — imported fixture
    _config_dir, store, owner = rig
    now = time.time()
    # More review rows than the whole budget, all from the same sender, today.
    for index in range(hr.SENDER_BUDGET + 2):
        row = _decision(store, owner, f"bcmsg_review{index:02d}", code="model_hold", now=now)
        assert row["eligible"] == 1  # review rows stay always-eligible
    row = _decision(store, owner, "bcmsg_workscope", code=hr.WORK_SCOPE_HOLD, now=now)
    # Before the fix: eligible=0, ineligible_reason=sender_budget (H12).
    assert row["eligible"] == 1 and row["ineligible_reason"] is None


def test_work_scope_holds_still_close_the_budget(rig) -> None:  # noqa: F811 — imported fixture
    _config_dir, store, owner = rig
    now = time.time()
    _decision(store, owner, "bcmsg_review00", code="model_hold", now=now)
    rows = [
        _decision(store, owner, f"bcmsg_ws{index:02d}", code=hr.WORK_SCOPE_HOLD, now=now)
        for index in range(hr.SENDER_BUDGET + 1)
    ]
    assert [r["eligible"] for r in rows] == [1] * hr.SENDER_BUDGET + [0]
    assert rows[-1]["ineligible_reason"] == hr.INELIGIBLE_BUDGET


# --- (b) review holds reach the long-hold rollup, at most daily -------------------


def test_an_unactioned_review_hold_enters_the_rollup_after_a_day(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    held = _screen(config_dir, owner, Runner(DOUBT))
    assert held.hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID)["hold_code"] == "model_hold"
    now = time.time()
    # Inside a day: the flood rule of 2026-11-10 still holds (no hourly re-banner).
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=now + 2 * 3600.0) == {
        "count": 0
    }
    # Before the fix this stayed 0 forever (H1-H4, H6-H10: up to 18 days unseen).
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=now + 25 * 3600.0) == {
        "count": 1
    }
    # An owner threshold longer than a day wins over the review floor.
    assert store.long_holds_readonly(owner, older_than_s=48 * 3600.0, now=now + 25 * 3600.0) == {
        "count": 0
    }


def test_a_retired_legacy_review_notice_does_not_hide_the_hold(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """The operator's four oldest holds: ``notified_at`` set by the 2026-11-09
    notice, which ``retire_review_notices`` dismissed (``cleared_at``)."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DOUBT))
    store.mark_work_scope_notified(owner, MESSAGE_ID, now=time.time())
    store.mark_work_scope_cleared(owner, MESSAGE_ID, now=time.time())
    later = time.time() + 30 * 3600.0
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=later) == {"count": 1}


def test_a_noticed_work_scope_hold_still_leaves_the_rollup(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(NEW_WORK)).hold_code == hr.WORK_SCOPE_HOLD
    later = time.time() + 30 * 3600.0
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=later) == {"count": 1}
    store.mark_work_scope_notified(owner, MESSAGE_ID, now=time.time())
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=later) == {"count": 0}


# --- (c) write_once_conflict: honest class, resolved by the committed verdict -----


def test_write_once_conflict_is_not_published_as_awaiting_review() -> None:
    assert delivery_hold_class("write_once_conflict") == "delivery_pending"


def _envelope(owner, item):
    envelope, bad = screening.validate_envelope(owner, item)
    assert envelope is not None, bad
    return envelope


def test_a_committed_preferred_allow_resolves_the_conflict(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    first = _screen(config_dir, owner, Runner(DELIVER))
    assert first.disposition == "deliver"
    resolved = screening._resolve_write_once_conflict(store, owner, _envelope(owner, _item()))
    assert resolved is not None and resolved.disposition == "deliver"
    assert resolved.receipt.verdict == "allow"
    # Text only: no image or file set is admitted on a resolution.
    assert resolved.attachments_hash is None and resolved.files_hash is None


def test_the_conflict_is_kept_when_the_committed_verdict_does_not_match(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DELIVER))
    # Different bytes under the same id: never answered by the old receipt.
    other = _item(body="a different body entirely")
    assert screening._resolve_write_once_conflict(store, owner, _envelope(owner, other)) is None
    # No receipt at all.
    fresh = _item(message_id="bcmsg_noreceipt")
    assert screening._resolve_write_once_conflict(store, owner, _envelope(owner, fresh)) is None
    # A non-preferred allow is provisional: it must be re-screened, not revived.
    real = store.receipt_for_message

    def not_preferred(owner_, message_id):
        row = real(owner_, message_id)
        return None if row is None else {**row, "screener_preferred": False}

    monkeypatch.setattr(store, "receipt_for_message", not_preferred)
    assert screening._resolve_write_once_conflict(store, owner, _envelope(owner, _item())) is None
    # Another enrollment generation's receipt.
    monkeypatch.setattr(
        store,
        "receipt_for_message",
        lambda o, m: {**real(o, m), "creator_generation": owner.enrollment_generation + 1},
    )
    assert screening._resolve_write_once_conflict(store, owner, _envelope(owner, _item())) is None


def test_the_pipeline_answers_a_write_once_refusal_with_the_committed_verdict(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    """The real commit path: a re-draw whose receipt differs is refused, then resolved."""

    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"

    def refuse(*_a, **_k):
        raise StoreRefused("write_once_conflict", MESSAGE_ID)

    # Every commit door refuses as the store does for an already-committed id.
    monkeypatch.setattr(store.__class__, "finish_screening", refuse, raising=False)
    monkeypatch.setattr(store.__class__, "adopt_cached_receipt", refuse, raising=False)
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "deliver" and out.hold_code is None
    assert out.receipt is not None and out.receipt.verdict == "allow"


# --- (d) the screener's reason is persisted, capped, and never logged -------------


def test_the_reason_is_capped_and_normalised() -> None:
    assert capped_screener_reason(None) is None
    assert capped_screener_reason("   ") is None
    assert capped_screener_reason(42) is None
    assert capped_screener_reason("a\x00b ") == "ab"
    assert len(capped_screener_reason("x" * 1000)) == SCREENER_REASON_MAX == 200


def test_a_review_hold_records_its_reason_and_never_logs_it(
    rig, caplog  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    caplog.set_level(logging.DEBUG)
    _screen(config_dir, owner, Runner(DOUBT))
    reason = store.work_scope_decision(owner, MESSAGE_ID)["screener_reason"]
    assert reason == "Too ambiguous to tell; it may be a new work request."
    assert reason not in caplog.text


def test_a_work_scope_hold_records_its_reason(rig) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    row = store.work_scope_decision(owner, MESSAGE_ID)
    assert row["hold_code"] == hr.WORK_SCOPE_HOLD
    assert row["screener_reason"] == "This is a new_work_request to start team work."


def test_the_store_caps_a_reason_whatever_the_caller_passes(
    rig,  # noqa: F811 — imported fixture
) -> None:
    # The parser already refuses a REASON over 200 characters
    # (``screening._MAX_REASON_CHARS``); the store cap holds on its own so a
    # future caller cannot turn the row into a copy of the message.
    _config_dir, store, owner = rig
    row = _decision(
        store, owner, "bcmsg_longreason", code="model_hold", now=time.time(), reason="z" * 900
    )
    assert row["screener_reason"] == "z" * SCREENER_REASON_MAX


def test_an_identical_resend_copies_the_reason(rig) -> None:  # noqa: F811 — imported fixture
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    copy = Runner(NEW_WORK)
    _screen(config_dir, owner, copy, item=_item(message_id="bcmsg_abcdef02"))
    assert copy.calls == 0
    first = store.work_scope_decision(owner, MESSAGE_ID)["screener_reason"]
    assert first and store.work_scope_decision(owner, "bcmsg_abcdef02")["screener_reason"] == first
