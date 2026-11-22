"""Terminal dispositions for holds that can never be delivered (the immortal-hold fix).

WHY THIS MODULE EXISTS (operator report 2026-11-21). A hold whose payload could
never be loaded again had NO exit: ``screening_holds`` is re-upserted by every
retry, ``list_holds`` served the row ``releasable: true`` for ever, and nothing on
screen could bind a Discard, so the Approvals badge could not reach zero
(``bcmsg_5ac4…`` on the operator's real store). These cells drive the REAL store
in a tmp dir and the REAL functions over it — no stubs of the things under test —
and they pin the two rules that must not drift:

* a ``conversation_failed`` row is REPAIRABLE, so no bulk or age pass may ever
  dispose of one, even though it is SERVED as ``unknown_hold`` (the trap);
* ``cleared_at`` means "this device retired the OS banner", never "the hold was
  settled", so it may not arm a proof-free Discard.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from local_operator.server.utils import desktop_team_bridge_holds as holds
from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge.store import OwnerReceipt, Store, StoreRefused
from local_operator.team_bridge.types import OwnerSnapshot

DAY = 86400.0
CONVERSATION = "bcv_abcdefgh"

# The operator's own stuck row, in shape: a review hold with an ELIGIBLE decision
# whose Deliver window has closed.
STUCK = "bcmsg_5ac4b3e8a40641c4b90eafaf71a0dba7"
# A row with no decision record at all: nothing can ever release or deliver it.
ORPHAN = "bcmsg_c21c1ee0614948aaa8bd2ddd348f5627"
# The retryable delivery class: 22 of these on the real store, all must survive.
RETRYABLE = "bcmsg_8f1b0a1d2c3e4f5061728394a5b6c7d8"
DELIVERABLE = "bcmsg_0a1b2c3d4e5f60718293a4b5c6d7e8f9"


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_recv",
        tenant_id="tnt_1",
        endpoint_id="bep_recv",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )


@pytest.fixture()
def rig(tmp_path):
    """A REAL store with one committed owner; returns (config_dir, store, owner)."""

    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1")
    return tmp_path, store, owner


def _decision(store: Store, owner: OwnerSnapshot, message_id: str, *, now: float, **kwargs):
    """Write a review-hold decision row through the store's own writer.

    ``model_hold`` is a ``REVIEW_HOLD_CODES`` member, so ``trusted_decision``
    accepts it without a preferred screener — the same shape as the operator's
    stuck row.
    """

    fields = {
        "conversation_id": CONVERSATION,
        "body_sha256": "a" * 64,
        "envelope_digest": "b" * 64,
        "screener_provider": "anthropic",
        "screener_model": "claude-sonnet-5-5",
        "template_sha256": "c" * 64,
        "prompt_version": "v2",
        "categories": [],
        "policy_epoch": 1,
        "sender_digest": "d" * 64,
        "now": now,
        "screener_verdict": "hold",
        "hold_code": hr.REVIEW_HOLD_CODES and "model_hold",
    }
    fields.update(kwargs)
    return store.record_work_scope_decision(owner, message_id, **fields)


def _counts(config_dir, owner, *, now: float) -> tuple[int, int]:
    """(items served, items the person owes an answer) at one moment."""

    items = holds.list_holds(config_dir, owner, now=now)["items"]
    return len(items), sum(1 for item in items if item["actionable"])


# --- the operator's stuck row -------------------------------------------------


def test_a_hold_past_its_deliver_window_is_disposable_with_no_proof(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)

    # Inside the window the row is a normal Deliver item and NOT disposable.
    inside = holds.list_holds(config_dir, owner, now=created + 60.0)["items"]
    assert len(inside) == 1
    assert inside[0]["can_deliver"] is True
    assert inside[0]["terminal_discard"] is False
    assert inside[0]["undeliverable_reason"] is None

    later = created + 8 * DAY
    outside = holds.list_holds(config_dir, owner, now=later)["items"]
    assert outside[0]["can_deliver"] is False
    assert outside[0]["terminal_discard"] is True
    assert outside[0]["undeliverable_reason"] == "deliver_window_passed"

    before_items, before_actionable = _counts(config_dir, owner, now=later)
    # The row WAS actionable — that is the defect (the badge could not clear):
    # the person owes an answer and no button could be armed.
    assert before_actionable == 1

    answer = holds.discard_undeliverable_hold(config_dir, owner, message_id=STUCK, now=later)
    assert answer["disposed"] is True and answer["already"] is False
    assert answer["terminal"]["reason"] == "discarded_undeliverable"
    assert answer["terminal"]["detail"] == "deliver_window_passed"
    assert answer["terminal"]["word"] == "discard"
    assert answer["terminal"]["notice_state"] == "none"  # no deliveries row: no sender
    # The marker is keyed by the BARE id, which is what the retry loop will read.
    assert store.hold_terminal(owner, STUCK)["message_id"] == STUCK

    after_items, after_actionable = _counts(config_dir, owner, now=later)
    assert after_items == before_items - 1
    assert after_actionable == before_actionable - 1 == 0

    # Idempotent: the same receipt, never ``already_decided``.
    again = holds.discard_undeliverable_hold(config_dir, owner, message_id=STUCK, now=later)
    assert again["already"] is True and again["disposed"] is False
    assert again["terminal"]["decided_at"] == answer["terminal"]["decided_at"]


def test_evidence_records_the_facts_an_auditor_needs(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)
    holds.discard_undeliverable_hold(config_dir, owner, message_id=STUCK, now=created + 8 * DAY)

    row = store.hold_terminal(owner, STUCK)
    assert row["screener_verdict"] == "hold"
    assert row["detail"] == "deliver_window_passed"
    # Numbers and booleans only: the store refuses any string value, so no body,
    # sender or label can ride along in the evidence column.
    facts = row["evidence"]
    assert '"deliver_until_passed":true' in facts
    assert '"allow_receipt":false' in facts
    assert '"body_reservation":false' in facts
    with pytest.raises(StoreRefused):
        store.record_hold_terminal(
            owner,
            ORPHAN,
            CONVERSATION,
            "model_hold",
            reason="discarded_undeliverable",
            evidence={"leaked": "a sender's own words"},
        )


# --- the two traps ------------------------------------------------------------


def test_conversation_failed_is_never_disposable_though_served_as_unknown(rig) -> None:
    """THE TRAP: ``conversation_failed`` is not in ``HOLD_CODES``, so it is SERVED
    as ``unknown_hold``. Classifying on the served word would eat all 22 of the
    operator's repairable delivery defects."""

    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, RETRYABLE, "conversation_failed")
    later = created + 30 * DAY

    items = holds.list_holds(config_dir, owner, now=later)["items"]
    assert len(items) == 1
    assert items[0]["hold_code"] == "unknown_hold"  # the served value, mapped
    assert items[0]["terminal_discard"] is False
    assert items[0]["undeliverable_reason"] is None

    with pytest.raises(holds.HoldRefused) as refused:
        holds.discard_undeliverable_hold(config_dir, owner, message_id=RETRYABLE, now=later)
    assert refused.value.code == "hold_not_releasable"

    counts = holds.discard_all_undeliverable(config_dir, owner, now=later)
    assert counts == {"discarded": 0, "already": 0, "skipped": 1}
    assert holds.expire_undeliverable_holds(config_dir, owner, now=later) == 0
    assert store.hold_terminal(owner, RETRYABLE) is None


def test_cleared_but_deliverable_is_not_terminal(rig) -> None:
    """``cleared_at`` is the OS banner being retired, NOT the hold being settled.

    Pinned so nobody "optimises" by reintroducing the cleared_at shortcut: this
    row is inside its Deliver window and the person can still deliver it.
    """

    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, DELIVERABLE, "model_hold")
    _decision(store, owner, DELIVERABLE, now=created)
    store.mark_work_scope_cleared(owner, DELIVERABLE, now=created + 5.0)
    assert store.work_scope_decision(owner, DELIVERABLE)["cleared_at"] is not None

    items = holds.list_holds(config_dir, owner, now=created + 60.0)["items"]
    assert len(items) == 1
    assert items[0]["can_deliver"] is True
    assert items[0]["actionable"] is True
    assert items[0]["terminal_discard"] is False
    with pytest.raises(holds.HoldRefused):
        holds.discard_undeliverable_hold(config_dir, owner, message_id=DELIVERABLE)


# --- no decision record, and a hold row that vanished -------------------------


def test_a_hold_with_no_decision_record_is_disposable_once_past_the_window(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, ORPHAN, "write_once_conflict")

    # INSIDE the Deliver window: an in-flight screener may still write the
    # decision, and racing it would lose a deliverable message.
    inside = holds.list_holds(config_dir, owner, now=created + 60.0)["items"]
    assert inside[0]["terminal_discard"] is False

    later = created + 8 * DAY
    outside = holds.list_holds(config_dir, owner, now=later)["items"]
    assert outside[0]["terminal_discard"] is True
    assert outside[0]["undeliverable_reason"] == "no_decision_record"
    assert outside[0]["decision"] is None

    answer = holds.discard_undeliverable_hold(config_dir, owner, message_id=ORPHAN, now=later)
    assert answer["terminal"]["detail"] == "no_decision_record"
    assert store.hold_terminal(owner, ORPHAN) is not None


def test_a_vanished_hold_row_is_nothing_to_dispose_not_an_error(rig) -> None:
    """``screening_holds`` deletes its row when a delivery finally hands off."""

    config_dir, store, owner = rig
    # (a) A terminal write for a message with no hold row at all still lands.
    terminal = store.record_hold_terminal(
        owner,
        ORPHAN,
        CONVERSATION,
        "model_hold",
        reason="discarded_undeliverable",
        detail="no_decision_record",
        now=time.time(),
    )
    assert terminal["already"] is False
    assert store.hold_terminal(owner, ORPHAN) is not None

    # (b) A message with neither a hold row nor a marker answers "nothing left".
    answer = holds.discard_undeliverable_hold(config_dir, owner, message_id=DELIVERABLE)
    assert answer == {
        "message_id": DELIVERABLE,
        "disposed": False,
        "already": True,
        "terminal": None,
    }

    # (c) A marker standing over a hold whose row has since been deleted: the
    # list is empty and nothing raises, in either surface.
    store.persist_hold(owner, STUCK, "model_hold")
    store.delete_hold(owner, STUCK)
    assert holds.list_holds(config_dir, owner)["items"] == []
    assert holds.expire_undeliverable_holds(config_dir, owner) == 0
    assert store.hold_terminal(owner, ORPHAN)["message_id"] == ORPHAN


# --- bulk and age-based passes ------------------------------------------------


def test_expiry_is_bounded_and_idempotent(rig, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir, store, owner = rig
    created = time.time()
    ids = [f"bcmsg_orphan{i:04d}" for i in range(5)]
    for message_id in ids:
        store.persist_hold(owner, message_id, "write_once_conflict")
    later = created + 15 * DAY  # past the 14-day default
    monkeypatch.setattr(holds, "EXPIRY_MAX_PER_PASS", 3)

    assert holds.expire_undeliverable_holds(config_dir, owner, now=later) == 3
    assert holds.expire_undeliverable_holds(config_dir, owner, now=later) == 2
    assert holds.expire_undeliverable_holds(config_dir, owner, now=later) == 0
    for message_id in ids:
        row = store.hold_terminal(owner, message_id)
        assert row["detail"] == "payload_expired"
        assert row["reason"] == "discarded_undeliverable"


def test_expiry_never_touches_a_deliverable_row(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, DELIVERABLE, "model_hold")
    _decision(store, owner, DELIVERABLE, now=created)

    # Fresh: deliverable, so nothing to sweep whatever the threshold says.
    assert holds.expire_undeliverable_holds(config_dir, owner, now=created + 60.0) == 0
    # Even 400 days later the row is past its window, so it is undeliverable and
    # the pass may settle it — but ``can_deliver`` was false by then.
    assert (
        holds.list_holds(config_dir, owner, now=created + 400 * DAY)["items"][0]["can_deliver"]
        is False
    )


def test_a_disabled_expiry_setting_never_disposes_anything(rig, tmp_path) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, ORPHAN, "write_once_conflict")
    (config_dir / "config.yml").write_text(
        json.dumps({"values": {"team_bridge": {"held_expiry_days": 0}}}), encoding="utf-8"
    )
    assert holds.expire_undeliverable_holds(config_dir, owner, now=created + 400 * DAY) == 0
    assert store.hold_terminal(owner, ORPHAN) is None


def test_discard_all_disposes_every_candidate_and_reports_the_rest(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    later = created + 8 * DAY
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)
    store.persist_hold(owner, ORPHAN, "write_once_conflict")
    store.persist_hold(owner, RETRYABLE, "conversation_failed")
    store.persist_hold(owner, DELIVERABLE, "model_hold")
    # A review decision can be written DAYS after the hold (measured on the
    # operator's store: 17.9 d vs 8.1 d), so a hold written eight days ago can
    # still be perfectly deliverable — that row must be left alone.
    _decision(store, owner, DELIVERABLE, now=later - 1 * DAY)

    counts = holds.discard_all_undeliverable(config_dir, owner, now=later)
    assert counts == {"discarded": 2, "already": 0, "skipped": 2}
    # The retryable class and the still-deliverable row are untouched.
    assert store.hold_terminal(owner, RETRYABLE) is None
    assert store.hold_terminal(owner, DELIVERABLE) is None
    assert store.hold_terminal(owner, STUCK) is not None

    again = holds.discard_all_undeliverable(config_dir, owner, now=later)
    assert again["discarded"] == 0
    assert again["already"] == 2
    assert again["skipped"] == 2  # the two that were always left alone


def test_the_long_hold_alert_stops_counting_a_disposed_row(rig) -> None:
    """Otherwise the badge clears and a zombie "long hold" notice stands for ever."""

    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)
    assert store.long_holds_readonly(owner, older_than_s=0.0, now=created + 8 * DAY)["count"] == 1

    holds.discard_undeliverable_hold(config_dir, owner, message_id=STUCK, now=created + 8 * DAY)
    assert store.long_holds_readonly(owner, older_than_s=0.0, now=created + 8 * DAY)["count"] == 0


# --- negative control: the OLD path leaves the row immortal -------------------


def test_negative_control_without_a_marker_the_row_is_still_served(rig) -> None:
    """The filter IS the fix: no marker, no disposal, an immortal actionable row.

    Structured to fail loudly if someone reverts the ``hold_terminals`` filter in
    ``list_holds``: the first half proves the old behaviour (listed and
    actionable with a closed Deliver window and nothing able to settle it), the
    second proves the marker is what removes it.
    """

    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)
    later = created + 8 * DAY

    old = holds.list_holds(config_dir, owner, now=later)["items"]
    assert [item["message_id"] for item in old] == [STUCK]
    assert old[0]["actionable"] is True
    assert old[0]["can_deliver"] is False

    store.record_hold_terminal(
        owner,
        STUCK,
        CONVERSATION,
        "model_hold",
        reason="discarded_undeliverable",
        detail="deliver_window_passed",
        now=later,
    )
    assert holds.list_holds(config_dir, owner, now=later)["items"] == []
    assert holds.list_holds(config_dir, owner, now=later)["items"] == []


# --- the reversal verb and the schema ----------------------------------------


def test_reopen_removes_the_marker_and_the_row_is_listed_again(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, ORPHAN, "write_once_conflict")
    later = created + 8 * DAY
    holds.discard_undeliverable_hold(config_dir, owner, message_id=ORPHAN, now=later)
    assert holds.list_holds(config_dir, owner, now=later)["items"] == []

    assert holds.reopen_hold(config_dir, owner, message_id=ORPHAN) is True
    assert [
        item["message_id"] for item in holds.list_holds(config_dir, owner, now=later)["items"]
    ] == [ORPHAN]
    assert holds.reopen_hold(config_dir, owner, message_id=ORPHAN) is False


def test_a_stamped_v9_store_without_the_marker_table_refuses_as_damaged(rig) -> None:
    config_dir, store, owner = rig
    store.persist_hold(owner, ORPHAN, "write_once_conflict")  # creates the v9 layout
    with sqlite3.connect(store._path) as conn:
        conn.execute("DROP TABLE hold_terminals")
    with pytest.raises(StoreRefused) as refused:
        store.persist_hold(owner, STUCK, "model_hold")
    assert refused.value.code == "schema_damaged"


def test_a_v8_store_upgrades_additively_and_keeps_every_row(rig) -> None:
    config_dir, store, owner = rig
    created = time.time()
    store.persist_hold(owner, ORPHAN, "write_once_conflict")
    store.persist_hold(owner, STUCK, "model_hold")
    _decision(store, owner, STUCK, now=created)
    # Rewind to a genuine v8 shape: no marker table, stamp 8, rows untouched.
    with sqlite3.connect(store._path) as conn:
        conn.execute("DROP TABLE hold_terminals")
        conn.execute("UPDATE schema_meta SET value = '8' WHERE key = 'schema_version'")

    reopened = Store(config_dir)
    # A READ still works on the v8 store and changes nothing (the read-only
    # connection never migrates): every row is intact.
    assert sorted(
        str(item["message_id"]) for item in reopened.screening_hold_rows(owner)
    ) == sorted([ORPHAN, STUCK])
    assert reopened.work_scope_decision(owner, STUCK) is not None
    # The UPGRADE happens on the first WRITE, through the same ``_init`` every
    # mutation runs (the v7->v8 reservation table's own rule), and it is additive.
    reopened.persist_hold(owner, RETRYABLE, "conversation_failed")
    with sqlite3.connect(store._path) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert version == "9"
    assert "hold_terminals" in tables
    # Every v8 row survived byte-for-byte: the hold rows, the decision, and the
    # retryable row the upgrade itself wrote.
    assert sorted(
        str(item["message_id"]) for item in reopened.screening_hold_rows(owner)
    ) == sorted([ORPHAN, STUCK, RETRYABLE])
    assert reopened.work_scope_decision(owner, STUCK) is not None
    # And the marker works on the upgraded store.
    assert reopened.hold_terminal(owner, ORPHAN) is None


def test_the_discard_body_models_keep_deliver_strict(rig) -> None:
    """Deliver still requires a binding; the proof-free discard needs none."""

    from local_operator.server.routes.desktop_team_bridge import (
        TeamBridgeHoldBatchDiscard,
        TeamBridgeHoldDecision,
        TeamBridgeHoldDiscard,
    )

    account = "acc_recv"
    with pytest.raises(ValueError):
        TeamBridgeHoldDecision(expected_account_id=account, conversation_id=CONVERSATION)
    with pytest.raises(ValueError):
        TeamBridgeHoldDecision(
            expected_account_id=account,
            conversation_id=CONVERSATION,
            body_sha256="a" * 64,
            hold_receipt="b" * 64,
        )
    # No binding at all: the proof-free path, and it needs no conversation id.
    proof_free = TeamBridgeHoldDiscard(expected_account_id=account)
    assert proof_free.body_sha256 is None and proof_free.hold_receipt is None
    # A binding without the conversation is the OLD request, and is refused.
    with pytest.raises(ValueError):
        TeamBridgeHoldDiscard(expected_account_id=account, body_sha256="a" * 64)
    with pytest.raises(ValueError):
        TeamBridgeHoldBatchDiscard(expected_account_id=account, scope="everything")
    assert TeamBridgeHoldBatchDiscard(expected_account_id=account, scope="undeliverable").scope == (
        "undeliverable"
    )
