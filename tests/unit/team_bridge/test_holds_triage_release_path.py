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

import dataclasses
import logging
import math
import sqlite3
import time

from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.store import (
    SCREENER_REASON_MAX,
    capped_screener_reason,
    delivery_hold_class,
    state_db_path,
)
from tests.unit.team_bridge.test_hold_release import (  # noqa: F401 — ``rig`` is a fixture
    DELIVER,
    DOUBT,
    MESSAGE_ID,
    NEW_WORK,
    PRYING,
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_VERSION_V2,
    SONNET,
    Policy,
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
    assert store.long_holds_readonly(owner, older_than_s=3600.0, now=now + 49 * 3600.0) == {
        "count": 1
    }
    # An owner threshold longer than a day wins over the review floor.
    assert store.long_holds_readonly(owner, older_than_s=48 * 3600.0, now=now + 25 * 3600.0) == {
        "count": 0
    }


def test_review_holds_enter_the_rollup_together_at_a_day_boundary(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """Agent review m1 on #4249: at most one review-driven count change a day.

    Two review holds written at different times on the same UTC day enter the
    count at the SAME instant (the start of the day after next), so a flood
    revises the alert once, not once per held message."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DOUBT))
    _screen(config_dir, owner, Runner(DOUBT), item=_item(body="another", message_id="bcmsg_ab02"))
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        stamps = [r[0] for r in conn.execute("SELECT created_at FROM screening_holds")]
    day = 24 * 3600.0
    start = math.floor(min(stamps) / day) * day
    assert math.floor(max(stamps) / day) * day == start  # same UTC day
    boundary = start + 2 * day

    def count(now: float) -> int:
        return store.long_holds_readonly(owner, older_than_s=3600.0, now=now)["count"]

    assert count(boundary - 1.0) == 0
    assert count(boundary) == 2


def test_a_retired_legacy_review_notice_does_not_hide_the_hold(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """The operator's four oldest holds: ``notified_at`` set by the 2026-11-09
    notice, which ``retire_review_notices`` dismissed (``cleared_at``)."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DOUBT))
    store.mark_work_scope_notified(owner, MESSAGE_ID, now=time.time())
    store.mark_work_scope_cleared(owner, MESSAGE_ID, now=time.time())
    later = time.time() + 49 * 3600.0
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


def _republish(store, owner, *, revision="rev-1", default_model=SONNET[1]) -> int:
    """Publish a policy the way the scheduler does (a new epoch on any change)."""

    current = store.current_screening_policy(owner)
    return store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        default_model=default_model,
        revision=revision,
        expected_epoch=int(current["epoch"]),
    )


def _bump_epoch_only(store, owner) -> int:
    """H5's shape: the epoch advanced, the template/revision/serving model did not."""

    _republish(store, owner, default_model="claude-opus-5-5")
    return _republish(store, owner)


def _tamper(config_dir, column: str, value) -> None:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute(f"UPDATE screening_receipts SET {column} = ?", (value,))


def test_a_current_preferred_allow_resolves_the_conflict(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    resolved = screening._resolve_write_once_conflict(store, owner, _envelope(owner, _item()))
    assert resolved is not None and resolved.disposition == "deliver"
    assert resolved.receipt.verdict == "allow"
    assert resolved.attachments_hash is None and resolved.files_hash is None


def test_the_commit_door_resolves_h5s_shape_epoch_moved_substance_current(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """Real ``screen_message``, no store patch: allow at epoch 1, the epoch moves
    with the substance unchanged, a re-draw answers reject, ``finish_screening``
    refuses ``write_once_conflict``, and the committed allow stands."""

    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    assert _bump_epoch_only(store, owner) > 1
    redraw = Runner(PRYING)
    out = _screen(config_dir, owner, redraw)
    assert redraw.calls == 1  # a real re-draw reached the commit door
    assert out.disposition == "deliver" and out.receipt.verdict == "allow"


def test_a_committed_reject_resolves_to_its_bounce(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(PRYING)).disposition == "bounce"
    _bump_epoch_only(store, owner)
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "bounce" and out.receipt.verdict == "reject"
    assert out.receipt.categories == ("personal_info_request",)


def test_a_tightened_policy_keeps_the_hold_instead_of_reviving_the_old_allow(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """Security review S-1 on #4249, through the real path: allow under policy A,
    the owner publishes a tightened policy B, the re-screen rejects, and the
    message stays held rather than going out on A's allow."""

    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    _republish(store, owner, revision="rev-2-tightened")
    tightened = Policy()
    tightened.admission_revision = "rev-2-tightened"
    redraw = Runner(PRYING)
    out = _screen(config_dir, owner, redraw, policy=tightened)
    assert redraw.calls == 1
    assert out.disposition == "hold" and out.hold_code == "write_once_conflict"
    assert (
        store.write_once_resolution(
            owner,
            MESSAGE_ID,
            body_sha256=_envelope(owner, _item()).body_sha256,
            envelope_digest=_envelope(owner, _item()).envelope_digest,
        )
        is None
    )


def test_the_conflict_is_kept_when_the_committed_verdict_cannot_stand(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DELIVER))

    def resolve(item=None):
        return screening._resolve_write_once_conflict(
            store, owner, _envelope(owner, item or _item())
        )

    # Different bytes under the same id, and an id with no receipt at all.
    assert resolve(_item(body="a different body entirely")) is None
    assert resolve(_item(message_id="bcmsg_noreceipt")) is None
    # Images or files: no grant to carry, so the hold stays here (S-3).
    envelope = _envelope(owner, _item())
    for field in ("attachments_hash", "files_hash"):
        carrying = dataclasses.replace(envelope, **{field: "f" * 64})
        assert screening._resolve_write_once_conflict(store, owner, carrying) is None
    assert resolve() is not None  # the control: the untouched row resolves
    # A non-preferred allow is provisional: re-screened, never revived.
    _tamper(config_dir, "screener_preferred", 0)
    assert resolve() is None
    _tamper(config_dir, "screener_preferred", 1)
    # Another enrollment generation's receipt, and damaged integers (S-2).
    _tamper(config_dir, "creator_generation", owner.enrollment_generation + 1)
    assert resolve() is None
    _tamper(config_dir, "creator_generation", "not-a-number")
    assert resolve() is None
    _tamper(config_dir, "creator_generation", owner.enrollment_generation)
    # An allow carrying categories, and a verdict outside the closed set (S-2).
    _tamper(config_dir, "categories", '["personal_info_request"]')
    assert resolve() is None
    _tamper(config_dir, "categories", "[]")
    _tamper(config_dir, "verdict", "hold")
    assert resolve() is None


def test_a_privacy_pass_without_a_final_allow_keeps_the_hold(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DELIVER))
    envelope = _envelope(owner, _item())
    args = {"body_sha256": envelope.body_sha256, "envelope_digest": envelope.envelope_digest}
    assert store.write_once_resolution(owner, MESSAGE_ID, **args) is not None
    generation = store.advance_privacy_control(enabled=True)
    current = store.current_screening_policy(owner)
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        default_model=SONNET[1],
        revision="rev-1",
        expected_epoch=int(current["epoch"]),
        privacy_generation=generation,
    )
    # The pass is on and this message has no final privacy allow on record.
    assert store.write_once_resolution(owner, MESSAGE_ID, **args) is None


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
