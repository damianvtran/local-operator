"""Holds triage 2026-11-18: the four release-path fixes (findings a-d).

The triage of 12 unseen Team Bridge holds found four bugs on the path from a
screening hold to the person who can release it:

* (a) the per-sender budget counted REVIEW decision rows, so three ordinary
  ``model_hold`` rows from one sender wrote that sender's next genuine
  ``work_scope_hold`` ineligible (H12: no Deliver/Discard);
* (b) review holds were left out of the long-hold rollup although no holds list
  ships, so nine of them sat for up to 18 days with no alert of any kind;
* (c) ``write_once_conflict`` could never clear (H5: re-driven 368 times with a
  committed allow);
* (d) the screener's REASON was never persisted, so calibration was inference.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import math
import sqlite3
import time
from typing import Any

import pytest

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
    _decide,
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


def test_a_kept_write_once_conflict_is_not_published_as_transient() -> None:
    # Security review S-4 on #4249: a conflict the resolver KEEPS has a
    # committed verdict the live policy no longer supports, so it is terminal.
    # ``delivery_pending`` would tell readers it is a transient delay.
    assert delivery_hold_class("write_once_conflict") == "awaiting_review"


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
    assert resolved.receipt is not None and resolved.receipt.verdict == "allow"
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
    assert out.disposition == "deliver"
    assert out.receipt is not None and out.receipt.verdict == "allow"


def test_a_committed_reject_resolves_to_its_bounce(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(PRYING)).disposition == "bounce"
    _bump_epoch_only(store, owner)
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "bounce"
    assert out.receipt is not None and out.receipt.verdict == "reject"
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


def test_a_kept_conflict_hold_is_scheduled_a_day_out_and_named(
    rig,  # noqa: F811 — imported fixture
    caplog,
) -> None:
    """2026-11-24 (PR-1 of the write-once fix): the kept conflict is persisted
    with a day-long ``retry_at`` — not NULL, which is due on the very next pass
    — and the refusal logs the NAMES of the replay fields that differ. The hold
    stays a hold; nothing about the verdict changes."""

    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    _republish(store, owner, revision="rev-2-tightened")
    tightened = Policy()
    tightened.admission_revision = "rev-2-tightened"
    before = time.time()
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge.store"):
        out = _screen(config_dir, owner, Runner(PRYING), policy=tightened)
    assert out.disposition == "hold" and out.hold_code == "write_once_conflict"
    row = store.screening_hold(owner, MESSAGE_ID)
    assert row is not None and row["hold_code"] == "write_once_conflict"
    assert before + screening.WRITE_ONCE_CONFLICT_RETRY_S <= row["retry_at"]
    assert row["retry_at"] <= time.time() + screening.WRITE_ONCE_CONFLICT_RETRY_S
    # Not due an hour (the old cap) from now, nor 23 hours from now.
    assert store.due_screening_holds(owner, now=time.time() + 23 * 3600, limit=8) == []
    lines = [r.getMessage() for r in caplog.records if "write_once_conflict:" in r.getMessage()]
    assert len(lines) == 1
    assert "differing=verdict,categories" in lines[0]
    assert "revision_moved=yes" in lines[0]
    assert MESSAGE_ID not in lines[0] and "personal_info_request" not in lines[0]


def test_other_screening_holds_keep_their_own_schedule(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """Only ``write_once_conflict`` gets the day: any other code keeps the time
    its caller passed (NULL = due next pass), and an explicit time is kept."""

    _config_dir, store, owner = rig
    screening._persist_hold(store, owner, "bcmsg_other", "retry_later")
    other = store.screening_hold(owner, "bcmsg_other")
    assert other is not None and other["retry_at"] is None
    screening._persist_hold(store, owner, "bcmsg_named", "write_once_conflict", 123.0)
    named = store.screening_hold(owner, "bcmsg_named")
    assert named is not None and named["retry_at"] == 123.0


def test_the_conflict_diagnostic_names_fields_and_never_values(caplog) -> None:
    """Injection-shaped values in every compared field: the line carries the
    closed field names only — no value, digest, category or message id, and no
    value can forge a second ``key=`` or line."""

    from local_operator.team_bridge.store import _log_write_once_conflict

    evil = "x\nrow_ref=forged differing=none revision_moved=no SECRET-BODY"
    existing = {
        "verdict": "allow",
        "model": "anthropic/claude-sonnet-5.5" + evil,
        "prompt_version": "v-old" + evil,
        "template_sha256": "a" * 64,
        "body_sha256": "b" * 64,
        "categories": "[]",
        "envelope_digest": "c" * 64,
        "admission_revision": "rev-old" + evil,
    }
    receipt = screening.ScreeningReceipt(
        verdict="reject",
        model="claude-sonnet-5-5" + evil,
        prompt_version="v-new" + evil,
        template_sha256="d" * 64,
        body_sha256="e" * 64,
        categories=("personal_info_request",),
    )
    message_id = "bcmsg_" + evil
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge.store"):
        _log_write_once_conflict(
            message_id,
            existing,
            receipt,
            envelope_digest="f" * 64,
            admission_revision="rev-new" + evil,
        )
    (line,) = [r.getMessage() for r in caplog.records if "write_once_conflict:" in r.getMessage()]
    assert line == (
        "team bridge write_once_conflict: row_ref=%s path=screening "
        "differing=verdict,model,prompt_version,template,body_sha256,categories,"
        "envelope_digest revision_moved=yes"
        % hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:12]
    )
    for leaked in ("SECRET-BODY", "forged", "\n", "a" * 12, "f" * 12, "personal_info_request"):
        assert leaked not in line
    # Only the model differs (a served-model failover) -> exactly that name.
    caplog.clear()
    same = dict(existing, model="m1", categories='["personal_info_request"]', verdict="reject")
    same.update(
        prompt_version=receipt.prompt_version,
        template_sha256=receipt.template_sha256,
        body_sha256=receipt.body_sha256,
        envelope_digest="f" * 64,
        admission_revision="rev-new" + evil,
    )
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge.store"):
        _log_write_once_conflict(
            message_id,
            same,
            dataclasses.replace(receipt, model="m2"),
            envelope_digest="f" * 64,
            admission_revision="rev-new" + evil,
        )
    (line,) = [r.getMessage() for r in caplog.records if "write_once_conflict:" in r.getMessage()]
    assert line.endswith("differing=model revision_moved=no")


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
    capped = capped_screener_reason("x" * 1000)
    assert capped is not None and len(capped) == SCREENER_REASON_MAX == 200


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


# --- write-once fix PR-2 (2026-11-24): a live allow supersedes a stale allow -----

STALE_MODEL = "anthropic/claude-sonnet-5.5"  # the id the 27 stuck receipts name


def _receipt_row(config_dir) -> dict[str, Any]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM screening_receipts").fetchone()
    return dict(row) if row is not None else {}


def _history(config_dir) -> list[dict[str, Any]]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM screening_receipt_history").fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def _history_columns(config_dir) -> list[str]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        return [row[1] for row in conn.execute("PRAGMA table_info(screening_receipt_history)")]


def _history_row_ref(config_dir) -> tuple[int, str]:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        row = conn.execute("SELECT history_id, model FROM screening_receipt_history").fetchone()
    return int(row[0]), str(row[1])


def _live(store, owner) -> Policy:
    """The rig policy object matching the store's live revision."""

    policy = Policy()
    policy.admission_revision = str(store.current_screening_policy(owner)["revision"])
    return policy


def _stuck_allow(config_dir, store, owner) -> dict[str, Any]:
    """The 2026-11-24 shape: an allow served by the other approved id, then the
    admission revision moves, and the message sits in a day-long conflict hold."""

    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    _tamper(config_dir, "model", STALE_MODEL)
    screening._persist_hold(store, owner, MESSAGE_ID, "write_once_conflict")
    _republish(store, owner, revision="rev-2")
    return _receipt_row(config_dir)


def test_a_live_allow_supersedes_a_stale_allow_and_keeps_the_old_row(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    prior = _stuck_allow(config_dir, store, owner)
    redraw = Runner(DELIVER)
    out = _screen(config_dir, owner, redraw, policy=_live(store, owner))
    assert redraw.calls == 1  # a real re-screen reached the commit door
    assert out.disposition == "deliver" and out.receipt is not None
    head = _receipt_row(config_dir)
    live = store.current_screening_policy(owner)
    assert head["model"] == SONNET[1] and head["admission_revision"] == live["revision"]
    assert head["policy_epoch"] == live["epoch"] and head["screener_preferred"] == 1
    (history,) = _history(config_dir)
    # The prior row, column for column, byte for byte.
    assert {name: history[name] for name in prior if name != "owner_key"} == {
        name: value for name, value in prior.items() if name != "owner_key"
    }
    assert history["reason"] == "live_allow_supersedes_stale_allow"
    assert history["attempt_id"] and history["new_model"] == SONNET[1]
    assert history["new_admission_revision"] == live["revision"]
    assert store.screening_hold(owner, MESSAGE_ID) is None
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        committed = conn.execute(
            "SELECT committed_message_id FROM screening_attempts WHERE attempt_id = ?",
            (history["attempt_id"],),
        ).fetchone()
    assert committed == (MESSAGE_ID,)
    # Now CURRENT: the write-once resolver accepts it, so a later fence can.
    envelope = _envelope(owner, _item())
    assert store.write_once_resolution(
        owner,
        MESSAGE_ID,
        body_sha256=envelope.body_sha256,
        envelope_digest=envelope.envelope_digest,
    )


def _classify(config_dir, store, owner, receipt, **override):
    from local_operator.team_bridge.store import Store, _owner_key

    policy = store.current_screening_policy(owner)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute("SELECT * FROM screening_receipts").fetchone()
        args = {
            "envelope_digest": existing["envelope_digest"],
            "fresh_revision": policy["revision"],
            "expected_epoch": int(policy["epoch"]),
            **override,
        }
        return Store._write_once_transition(
            conn, owner, _owner_key(owner), existing=existing, receipt=receipt, **args
        )


def _fresh(row, **change):
    return screening.ScreeningReceipt(
        verdict=change.get("verdict", "allow"),
        model=change.get("model", SONNET[1]),
        prompt_version=row["prompt_version"],
        template_sha256=row["template_sha256"],
        body_sha256=change.get("body_sha256", row["body_sha256"]),
        categories=change.get("categories", ()),
        screener_preferred=change.get("preferred", True),
    )


def test_only_a_fully_current_preferred_allow_may_supersede(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """The transition table, one change at a time against the stuck row."""

    config_dir, store, owner = rig
    prior = _stuck_allow(config_dir, store, owner)
    assert _classify(config_dir, store, owner, _fresh(prior)) == "supersede"  # control
    kept = {
        "fresh reject": (
            _fresh(prior, verdict="reject", categories=("personal_info_request",)),
            {},
        ),
        "non-preferred allow": (_fresh(prior, preferred=False), {}),
        "unapproved model": (_fresh(prior, model="qa-unapproved"), {}),
        "other body": (_fresh(prior, body_sha256="0" * 64), {}),
        "other envelope": (_fresh(prior), {"envelope_digest": "0" * 64}),
        "fresh at a stale revision": (_fresh(prior), {"fresh_revision": "rev-1"}),
        "fresh at a stale epoch": (_fresh(prior), {"expected_epoch": 1}),
    }
    for name, (receipt, override) in kept.items():
        assert _classify(config_dir, store, owner, receipt, **override) is None, name
    # A committed REJECT is never superseded, stale or not.
    _tamper(config_dir, "verdict", "reject")
    _tamper(config_dir, "categories", '["personal_info_request"]')
    assert _classify(config_dir, store, owner, _fresh(prior)) is None
    # A damaged committed allow stays a conflict, never a quiet supersession.
    _tamper(config_dir, "verdict", "allow")
    for damaged in ("", "not json", '{"a": 1}', '["personal_info_request"]'):
        _tamper(config_dir, "categories", damaged)
        assert _classify(config_dir, store, owner, _fresh(prior)) is None, damaged
    # #4533 m2: every integrity check runs on a STALE row, not only categories.
    _tamper(config_dir, "categories", "[]")
    assert _classify(config_dir, store, owner, _fresh(prior)) == "supersede"  # control
    for column in ("creator_generation", "policy_epoch"):
        original = _receipt_row(config_dir)[column]
        _tamper(config_dir, column, "not-a-number")
        assert _classify(config_dir, store, owner, _fresh(prior)) is None, column
        _tamper(config_dir, column, original)
    assert _history(config_dir) == []


def test_a_current_receipt_from_another_approved_model_replays_without_a_write(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    assert _screen(config_dir, owner, Runner(DELIVER)).disposition == "deliver"
    current = store.current_screening_policy(owner)
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        default_model=SONNET[1],
        revision=current["revision"],
        expected_epoch=int(current["epoch"]),
        approved_serving=[SONNET, ("anthropic", STALE_MODEL)],
    )
    row = _receipt_row(config_dir)
    assert _classify(config_dir, store, owner, _fresh(row, model=STALE_MODEL)) == "replay"
    # Replay agrees on the verdict too: a differing verdict is still a conflict.
    reject = _fresh(row, model=STALE_MODEL, verdict="reject", categories=("personal_info_request",))
    assert _classify(config_dir, store, owner, reject) is None
    assert _receipt_row(config_dir) == row and _history(config_dir) == []


@pytest.mark.parametrize(
    ("first", "second"),
    [(DELIVER, PRYING), (PRYING, DELIVER)],
    ids=["allow-then-reject", "reject-then-allow"],
)
def test_a_kept_conflict_writes_no_history_and_leaves_the_receipt(
    rig, first, second  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(first))
    _republish(store, owner, revision="rev-2")
    before = _receipt_row(config_dir)
    out = _screen(config_dir, owner, Runner(second), policy=_live(store, owner))
    assert out.disposition == "hold" and out.hold_code == "write_once_conflict"
    assert _receipt_row(config_dir) == before and _history(config_dir) == []


@pytest.mark.parametrize("damaged", ["", "not json", '{"a": 1}'])
def test_a_damaged_row_keeps_the_conflict_code_and_logs_unreadable(
    rig, damaged, caplog, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    """#4517 m1/S-1: the diagnostic decoded categories unconditionally, so a
    damaged row turned ``write_once_conflict`` into ``proof_corrupt`` →
    ``retry_later`` (hourly). The refusal code must not depend on the log.

    The receipt-cache read that normally catches a damaged row first is stubbed
    to a miss, so the re-screen reaches the commit door with the row in place.
    """

    from local_operator.team_bridge.store import _log_write_once_conflict

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(DELIVER))
    _tamper(config_dir, "categories", damaged)
    monkeypatch.setattr(screening.Store, "consume_screening_receipt", lambda *a, **k: None)
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge.store"):
        out = _screen(config_dir, owner, Runner(PRYING))
        # The helper alone, for each damaged shape: it returns, it never raises.
        _log_write_once_conflict(
            MESSAGE_ID,
            _receipt_row(config_dir),
            screening.ScreeningReceipt(
                verdict="reject",
                model=SONNET[1],
                prompt_version="v",
                template_sha256="a" * 64,
                body_sha256="b" * 64,
                categories=("personal_info_request",),
            ),
            envelope_digest="c" * 64,
            admission_revision=None,
        )
    assert out.disposition == "hold" and out.hold_code == "write_once_conflict"
    lines = [r.getMessage() for r in caplog.records if "write_once_conflict:" in r.getMessage()]
    assert len(lines) == 2 and all("differing=unreadable" in line for line in lines)
    assert _history(config_dir) == []


def test_a_policy_publish_cannot_interleave_with_a_supersession(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    """Two connections: a publisher racing the commit either lands FIRST (the
    commit refuses ``policy_stale``, nothing moves) or waits for the commit's
    write lock and lands AFTER it — never in between."""

    import threading

    from local_operator.team_bridge.store import Store

    config_dir, store, owner = rig
    _stuck_allow(config_dir, store, owner)
    original = Store._write_once_transition
    published: list[int] = []
    blocked: list[bool] = []

    def racing(conn, *args, **kwargs):
        verdict = original(conn, *args, **kwargs)

        def publish() -> None:
            published.append(_republish(Store(config_dir), owner, revision="rev-3"))

        racer = threading.Thread(target=publish)
        racer.start()
        racer.join(0.5)
        blocked.append(racer.is_alive())  # held off by this transaction's lock
        racing.thread = racer  # type: ignore[attr-defined]
        return verdict

    monkeypatch.setattr(Store, "_write_once_transition", staticmethod(racing))
    live = _live(store, owner)
    out = _screen(config_dir, owner, Runner(DELIVER), policy=live)
    racing.thread.join(10)  # type: ignore[attr-defined]
    assert blocked == [True] and len(published) == 1
    assert out.disposition == "deliver"
    head = _receipt_row(config_dir)
    # Committed under the epoch it was fenced on; the publish came after.
    assert head["admission_revision"] == "rev-2" and len(_history(config_dir)) == 1
    assert store.current_screening_policy(owner)["revision"] == "rev-3"


def test_a_publish_that_lands_first_refuses_the_supersession(
    rig, monkeypatch  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    prior = _stuck_allow(config_dir, store, owner)
    live = _live(store, owner)
    original = screening.Store.finish_screening

    def publish_then_finish(self, *args, **kwargs):
        _republish(store, owner, revision="rev-3")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(screening.Store, "finish_screening", publish_then_finish)
    out = _screen(config_dir, owner, Runner(DELIVER), policy=live)
    assert out.disposition == "hold"
    assert _receipt_row(config_dir) == prior and _history(config_dir) == []


def test_receipt_history_is_append_only_and_additive(
    rig,  # noqa: F811 — imported fixture
) -> None:
    from local_operator.team_bridge.store import _V3_TABLES

    config_dir, store, owner = rig
    _stuck_allow(config_dir, store, owner)
    _screen(config_dir, owner, Runner(DELIVER), policy=_live(store, owner))
    assert len(_history(config_dir)) == 1
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        for statement in (
            "UPDATE screening_receipt_history SET reason = 'x'",
            "DELETE FROM screening_receipt_history",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
    # S-1 / Q-1: REPLACE's implicit delete fires no DELETE trigger, so an
    # explicit existing ``history_id`` must be refused at INSERT.
    history_id, model = _history_row_ref(config_dir)
    columns = _history_columns(config_dir)
    forged = ", ".join("'FORGED'" if name == "model" else name for name in columns)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        for verb in ("INSERT OR REPLACE INTO", "REPLACE INTO"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(
                    f"{verb} screening_receipt_history ({', '.join(columns)}) "
                    f"SELECT {forged} FROM screening_receipt_history WHERE history_id = ?",
                    (history_id,),
                )
    assert _history_row_ref(config_dir) == (history_id, model) and model != "FORGED"
    assert len(_history(config_dir)) == 1
    # A table created before the INSERT trigger existed gains it on the next
    # supersession's ensure, the only writer, before it writes.
    from local_operator.team_bridge.store import _ensure_receipt_history_table

    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("DROP TRIGGER screening_receipt_history_no_replace")
        _ensure_receipt_history_table(conn)
        triggers = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name = 'screening_receipt_history'"
            )
        }
    assert "screening_receipt_history_no_replace" in triggers
    # Additive, never part of the damage floor: a store without it still opens.
    assert "screening_receipt_history" not in _V3_TABLES
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        for verb in ("update", "delete", "replace"):
            conn.execute(f"DROP TRIGGER screening_receipt_history_no_{verb}")
        conn.execute("DROP TABLE screening_receipt_history")
    assert store.current_screening_policy(owner) is not None


# --- B: an owner Deliver agreeing with a screener allow ------------------------


def _screener_receipt(
    config_dir, store, owner, *, verdict="allow", categories="[]"
) -> dict[str, Any]:
    """Commit-shaped screener row for a message already in a work-scope hold."""

    envelope = _envelope(owner, _item())
    policy = store.current_screening_policy(owner)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at, "
            "rendered_template_sha256, screener_preferred) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _owner_key_of(owner),
                MESSAGE_ID,
                envelope.body_sha256,
                SCREENING_TEMPLATE_V2_SHA256,
                SCREENING_VERSION_V2,
                STALE_MODEL,
                verdict,
                categories,
                envelope.envelope_digest,
                "rev-0",
                int(policy["epoch"]),
                owner.enrollment_generation,
                time.time() - 3 * 86400,
                None,
                1,
            ),
        )
    return _receipt_row(config_dir)


def _owner_key_of(owner) -> str:
    from local_operator.team_bridge.store import _owner_key

    return _owner_key(owner)


def test_an_owner_deliver_agreeing_with_a_screener_allow_moves_the_head(
    rig,  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    prior = _screener_receipt(config_dir, store, owner)
    # Before this fix the first consume was refused; that outcome must not
    # stop a later consume from committing and settling.
    store.set_owner_release_outcome(owner, MESSAGE_ID, "refused:write_once_conflict", now=1.0)
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "deliver" and out.receipt is not None
    assert out.receipt.template_sha256 == hr.OWNER_RELEASE_SHA256
    store.set_owner_release_outcome(owner, MESSAGE_ID, "admitted", now=time.time())
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "admitted"
    head = _receipt_row(config_dir)
    assert head["template_sha256"] == hr.OWNER_RELEASE_SHA256
    assert head["admission_revision"] == store.current_screening_policy(owner)["revision"]
    (history,) = _history(config_dir)
    assert history["reason"] == "owner_release_deliver" and history["attempt_id"] is None
    assert {k: history[k] for k in prior if k != "owner_key"} == {
        k: v for k, v in prior.items() if k != "owner_key"
    }
    # A re-drive replays the consumed release (no second commit, no new history).
    again = _screen(config_dir, owner, Runner(DELIVER))
    assert again.disposition == "deliver" and len(_history(config_dir)) == 1


def test_an_owner_deliver_never_overrides_a_screener_reject(
    rig, caplog  # noqa: F811 — imported fixture
) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    prior = _screener_receipt(
        config_dir, store, owner, verdict="reject", categories='["personal_info_request"]'
    )
    with caplog.at_level(logging.INFO, logger="local_operator.team_bridge.store"):
        out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "hold" and out.hold_code == "write_once_conflict"
    assert _receipt_row(config_dir) == prior and _history(config_dir) == []
    lines = [r.getMessage() for r in caplog.records if "write_once_conflict:" in r.getMessage()]
    assert lines and "path=owner_release" in lines[0] and "differing=" in lines[0]


def test_an_owner_deliver_never_supersedes_a_damaged_screener_allow(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """#4533 m2: readable empty categories are not enough. A committed allow
    that ``receipt_authority`` classifies CORRUPT (here a creator generation
    that is not a number) keeps its head and gets no history, exactly as on
    the screener path."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    _screener_receipt(config_dir, store, owner)
    _tamper(config_dir, "creator_generation", "not-a-number")
    prior = _receipt_row(config_dir)
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "hold"
    assert _receipt_row(config_dir) == prior and _history(config_dir) == []


def test_an_unverified_release_cannot_move_the_head(
    rig,  # noqa: F811 — imported fixture
) -> None:
    from local_operator.team_bridge.store import StoreRefused

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    prior = _screener_receipt(config_dir, store, owner)
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        sig = conn.execute("SELECT signature FROM owner_releases").fetchone()[0]
        conn.execute(
            "UPDATE owner_releases SET signature = ?",
            (sig[:-2] + ("00" if sig[-2:] != "00" else "11"),),
        )
    envelope = _envelope(owner, _item())
    receipt = screening.ScreeningReceipt(
        verdict="allow",
        model=hr.OWNER_RELEASE_MODEL,
        prompt_version=hr.OWNER_RELEASE_PROMPT_VERSION,
        template_sha256=hr.OWNER_RELEASE_SHA256,
        body_sha256=envelope.body_sha256,
        categories=(),
    )
    with pytest.raises(StoreRefused) as refused:
        store.commit_owner_release(
            owner,
            MESSAGE_ID,
            receipt=receipt,
            envelope_digest=envelope.envelope_digest,
            expected_epoch=int(store.current_screening_policy(owner)["epoch"]),
        )
    assert refused.value.code == "proof_corrupt"
    assert _receipt_row(config_dir) == prior and _history(config_dir) == []


def test_an_owner_deliver_makes_a_conflict_hold_due_now(
    rig,  # noqa: F811 — imported fixture
) -> None:
    """#4517 n2: the conflict code is scheduled a day out by NAME; a Deliver is
    the event that can now commit, so it re-arms the row itself."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    screening._persist_hold(store, owner, MESSAGE_ID, "write_once_conflict")
    screening._persist_hold(store, owner, "bcmsg_untouched", "write_once_conflict")
    assert store.due_screening_holds(owner, now=time.time() + 60, limit=8) == []
    before = time.time()
    store.record_owner_release(owner, store.owner_release(owner, MESSAGE_ID))
    row = store.screening_hold(owner, MESSAGE_ID)
    assert row is not None and row["retry_at"] <= time.time() and row["retry_at"] >= before
    untouched = store.screening_hold(owner, "bcmsg_untouched")
    assert untouched is not None and untouched["retry_at"] > time.time() + 23 * 3600
