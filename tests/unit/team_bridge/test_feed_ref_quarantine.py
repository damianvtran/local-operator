"""Regression pins for the 2026-11-10 receiver wedge: batch isolation.

A conversation feed ref the parser refuses made ``record_feed_batch`` raise
``StoreRefused("ref_migration_malformed")``; the batch being atomic, no ref and
no cursor ever committed again and every inbound message stalled. #3752 fixed
the parser for the 8-part shape that triggered it (its own tests live in
``test_feed_ref_round_trip.py``); these tests pin the store side, so the NEXT
unreadable ref cannot wedge the receiver:

- one unreadable ref is QUARANTINED as a content-free, visible hold while the
  rest of the batch and the cursor commit;
- the quarantine is idempotent and never re-driven;
- ``delete_feed_refs`` tolerates an unreadable stored ref.

The malformed refs used here are ones the current parser refuses (asserted
below), so the tests exercise quarantine rather than a parse that happens to
succeed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from local_operator.team_bridge.store import (
    FEED_REF_QUARANTINE_CODE,
    FEED_REF_QUARANTINE_PREFIX,
    OwnerReceipt,
    Store,
    state_db_path,
)
from local_operator.team_bridge.types import (
    CONVERSATION_FEED_KIND,
    CONVERSATION_FEED_TYPE,
    OwnerSnapshot,
    feed_seq,
    parse_conversation_ref_value,
)

PREFIX = f"{CONVERSATION_FEED_TYPE}:{CONVERSATION_FEED_KIND}="
CONV = "bcv_" + "a" * 32
MSG = "bcmsg_" + "b" * 32
ROOT = "bcmsg_" + "c" * 32

# The exact live shape (synthetic ids): 4 locator parts + kind + 3 tags.
EIGHT_PART = f"{CONV}/{MSG}/558/0/agent/thread_root={ROOT}/addressed=ambient/relation=member"

# Refs the parser refuses on any grammar (not arity): a non-canonical sequence,
# fewer than the four locator parts, and a known tag with an unknown value.
NON_CANONICAL_SEQ = f"{CONV}/{MSG}/0558/0/agent"
BELOW_FLOOR = f"{CONV}/{MSG}/1"
UNKNOWN_RELATION = f"{CONV}/{MSG}/1/0/agent/relation=stranger"

OWNER = OwnerSnapshot(
    credential_id=7,
    account_id="acc_1",
    tenant_id="ten_1",
    endpoint_id="ep_1",
    canonical_base_url="https://hub.example",
    enrollment_generation=7,
)


def _enrolled_store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    store.commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1")
    return store


# -- quarantine ----------------------------------------------------------------


@pytest.mark.parametrize("value", [NON_CANONICAL_SEQ, BELOW_FLOOR, UNKNOWN_RELATION])
def test_fixture_refs_are_refused_by_the_parser(value: str) -> None:
    # Guard: if the grammar ever accepts one of these, the quarantine tests
    # below would silently stop exercising quarantine.
    assert parse_conversation_ref_value(value) is None


def test_live_eight_part_ref_is_readable() -> None:
    # The "good" ref in the batch test is the shape that caused the wedge.
    assert parse_conversation_ref_value(EIGHT_PART) is not None


def test_one_malformed_ref_does_not_wedge_the_batch(tmp_path: Path) -> None:
    store = _enrolled_store(tmp_path)
    good = [PREFIX + EIGHT_PART, "message_available:message_id=bmsg_1"]
    bad = PREFIX + NON_CANONICAL_SEQ
    store.record_feed_batch(OWNER, feed_seq(42), [good[0], bad, good[1]], resync_required=False)

    assert store.sync_state(OWNER)["feed_seq"] == 42
    assert sorted(store.drain_feed_refs(OWNER, 42)) == sorted(good)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        holds = conn.execute("SELECT message_id, hold_code FROM screening_holds").fetchall()
    digest = hashlib.sha256(bad.encode()).hexdigest()
    assert holds == [(f"{FEED_REF_QUARANTINE_PREFIX}{digest}", FEED_REF_QUARANTINE_CODE)]
    # Content-free: the raw ref is never stored, only its digest.
    assert bad not in repr(holds)

    # Re-fetching the same page neither duplicates nor re-dates the quarantine.
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        first_at = conn.execute("SELECT created_at FROM screening_holds").fetchone()[0]
    store.record_feed_batch(OWNER, feed_seq(43), [bad], resync_required=False)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        rows = conn.execute("SELECT created_at FROM screening_holds").fetchall()
    assert rows == [(first_at,)]
    assert store.sync_state(OWNER)["feed_seq"] == 43


def test_quarantine_is_visible_in_status_and_never_redriven(tmp_path: Path) -> None:
    store = _enrolled_store(tmp_path)
    bad = PREFIX + UNKNOWN_RELATION
    store.record_feed_batch(OWNER, feed_seq(1), [bad], resync_required=False)

    assert store.due_screening_holds(OWNER, now=1e12) == []
    assert store.hold_count_readonly(OWNER, FEED_REF_QUARANTINE_CODE) == 1
    status = store.screening_status_readonly(OWNER)
    assert FEED_REF_QUARANTINE_CODE in repr(status)


def test_delete_feed_refs_tolerates_a_stored_unreadable_ref(tmp_path: Path) -> None:
    store = _enrolled_store(tmp_path)
    bad = PREFIX + BELOW_FLOOR
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        key = conn.execute("SELECT owner_key FROM enrollments").fetchone()[0]
        conn.execute("INSERT INTO feed_refs (owner_key, ref, seq) VALUES (?,?,?)", (key, bad, 1))
    assert store.delete_feed_refs(OWNER, [bad]) == 1
