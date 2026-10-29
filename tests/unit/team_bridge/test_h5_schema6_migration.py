"""H5 S-11 — schema v6: the ADDITIVE migration and its refusals (SOURCE).

fa5d884c's schema v5 never required ``screened_request_deliveries``; the
producer table is a v6 addition. Pinned here against a REAL sqlite store:
- a baseline-v5 database (no table) migrates to v6 in ONE transaction with
  every enrollment/claim/attempt/receipt/text-floor row byte-preserved and
  an EMPTY producer table (no legacy backfill — old messages stay
  non-authorizing until a fresh trusted producer validation writes one);
- a WIP-v5 store that already carries the table AND rows migrates by
  validating + stamping only — no row is lost or recreated;
- a FAILED migration attempt (a DDL error mid-way) leaves no half-state:
  reopening migrates cleanly with rows intact (transaction rollback, never
  a "repair");
- damaged old floors refuse BEFORE any migration; a v6 store missing the
  table, or carrying a partial one, refuses with nothing repaired.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import pytest

from local_operator.team_bridge import store as store_mod
from local_operator.team_bridge.store import SCHEMA_VERSION, Store, StoreRefused

PRESERVED_TABLES = (
    "enrollments",
    "claims",
    "screening_policy",
    "screening_receipts",
    "text_floors",
    "screening_attempts",
)


def _open_store(store: Store) -> None:
    """Run the store's REAL open path (connect + ``_init``) and let it raise.

    ``Store.active_owner`` deliberately folds ``StoreRefused``/``sqlite3.Error``
    into ``None`` ("damaged reads as no authority" — the fail-closed law), so
    it can never show the TYPED refusal or the failing migration. Every door
    that writes goes through exactly this pair, unswallowed; the swallowing
    reader is asserted separately as the fail-closed ``None``.
    """

    with store._connect() as conn:
        store._init(conn)


def _seed_full_store(tmp_path) -> Store:
    store = Store(tmp_path)
    store.active_owner()  # create + stamp via the real _init
    assert store._path.exists()
    conn = sqlite3.connect(store._path)
    try:
        key = '["https://hub.example","acc_1","ten_1","ep_1"]'
        conn.execute(
            "INSERT INTO enrollments (owner_key, credential_id, account_id, tenant_id, "
            "endpoint_id, canonical_base_url, enrollment_generation, consent_version, "
            "consent_ref, designated_session, state) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                7,
                "acc_1",
                "ten_1",
                "ep_1",
                "https://hub.example",
                7,
                "c-v1",
                "c-ref",
                "s1",
                "committed",
            ),
        )
        conn.execute(
            "INSERT INTO claims (owner_key, message_id, session_id, token, generation, "
            "expires_at, state) VALUES (?,?,?,?,?,?,?)",
            (key, "bmsg_1", "s1", "tok-1", 7, time.time() + 300, "held"),
        )
        conn.execute(
            "INSERT INTO screening_policy (owner_key, version, template_sha256, "
            "default_model, revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (key, "v1", "aa11" * 16, "model-a", "rev-1", 3, time.time()),
        )
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                "bmsg_1",
                "9f2a" * 16,
                "aa11" * 16,
                "prompt-v1",
                "model-a",
                "allow",
                '["spam"]',
                "bb22" * 16,
                "rev-1",
                3,
                7,
                time.time(),
            ),
        )
        conn.execute(
            "INSERT INTO text_floors (owner_key, hub, kind, version, sha256, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (key, "https://hub.example", "guidance", "g-v1", "cc33" * 16, time.time()),
        )
        conn.execute(
            "INSERT INTO screening_attempts (attempt_id, kind, account_scope, owner_key, "
            "message_id, fingerprint, at, committed_message_id) VALUES (?,?,?,?,?,?,?,?)",
            ("att-1", "screen", "scope-1", key, "bmsg_1", "fp-1", time.time(), ""),
        )
        conn.commit()
    finally:
        conn.close()
    return store


def _snapshot(path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        out: dict[str, list[dict[str, Any]]] = {}
        for table in PRESERVED_TABLES:
            out[table] = [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
        version = str(
            conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
        )
        return out, version
    finally:
        conn.close()


def _downgrade_to_v5(path, *, drop_table: bool) -> None:
    """Turn the file into a legitimately-stamped V5 (the pre-producer era)."""

    conn = sqlite3.connect(path)
    try:
        if drop_table:
            conn.execute("DROP TABLE screened_request_deliveries")
        conn.execute("UPDATE schema_meta SET value = '5' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()


def _table_exists(path, name: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _count(path, table: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def test_baseline_v5_migrates_additively_preserving_every_row(tmp_path) -> None:
    store = _seed_full_store(tmp_path)
    before, _ = _snapshot(store._path)
    _downgrade_to_v5(store._path, drop_table=True)

    reopened = Store(tmp_path)
    assert reopened.active_owner() is not None  # the migration ran on open

    after, version = _snapshot(store._path)
    assert version == str(SCHEMA_VERSION)
    assert after == before  # byte-for-byte on every preserved table
    assert _table_exists(store._path, "screened_request_deliveries")
    # EMPTY: legacy messages never backfill producer authority.
    assert _count(store._path, "screened_request_deliveries") == 0


def test_wip_v5_with_table_and_rows_migrates_without_losing_them(tmp_path) -> None:
    store = _seed_full_store(tmp_path)
    owner = store.active_owner()
    assert owner is not None, "the seeded store must have an active owner"
    envelope = store.write_screened_request_delivery(
        owner,
        delivery_ref="dref_wip",
        message_id="bmsg_1",
        designated_session="s1",
        display_digest="dd44" * 16,
        body_sha256="9f2a" * 16,
        claim_token="tok-1",
        request={
            "thread_id": "t-1",
            "message_kind": "request",
            "request_state": "pending",
            "request_expires_at": "",
            "request_ref": "",
            "sender_account_id": "acc_1",
            "sender_endpoint_id": "ep_1",
            "recipient_account_id": "acc_2",
            "recipient_endpoint_id": "ep_2",
        },
    )
    assert envelope["delivery_ref"] == "dref_wip"
    _downgrade_to_v5(store._path, drop_table=False)  # keep the WIP table + row

    reopened = Store(tmp_path)
    assert reopened.active_owner() is not None
    _, version = _snapshot(store._path)
    assert version == str(SCHEMA_VERSION)
    assert _count(store._path, "screened_request_deliveries") == 1


def test_a_failed_migration_leaves_no_half_state(tmp_path, monkeypatch) -> None:
    store = _seed_full_store(tmp_path)
    before, _ = _snapshot(store._path)
    _downgrade_to_v5(store._path, drop_table=True)

    # Fail the migration mid-way: a DDL statement that cannot run.
    monkeypatch.setattr(
        store_mod, "_PRODUCER_DDL_STATEMENTS", ("SELECT * FROM definitely_missing_table",)
    )
    with pytest.raises((sqlite3.OperationalError, StoreRefused)):
        _open_store(Store(tmp_path))
    assert Store(tmp_path).active_owner() is None  # fail-closed, not half-migrated
    # The attempt rolled back: still v5, still no table, rows untouched.
    _, version_after_failure = _snapshot(store._path)
    assert version_after_failure == "5"
    assert not _table_exists(store._path, "screened_request_deliveries")

    monkeypatch.undo()
    reopened = Store(tmp_path)
    assert reopened.active_owner() is not None
    after, version = _snapshot(store._path)
    assert version == str(SCHEMA_VERSION)
    assert after == before
    assert _table_exists(store._path, "screened_request_deliveries")


def test_v5_with_a_damaged_old_floor_refuses_before_migration(tmp_path) -> None:
    store = _seed_full_store(tmp_path)
    conn = sqlite3.connect(store._path)
    try:
        conn.execute("DROP TABLE text_floors")
        conn.execute("UPDATE schema_meta SET value = '5' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as refusal:
        _open_store(Store(tmp_path))
    assert refusal.value.code == "schema_damaged"
    assert Store(tmp_path).active_owner() is None


def test_v6_missing_the_producer_table_refuses(tmp_path) -> None:
    store = _seed_full_store(tmp_path)
    _downgrade_to_v5(store._path, drop_table=True)
    conn = sqlite3.connect(store._path)
    try:
        conn.execute("UPDATE schema_meta SET value = '6' WHERE key = 'schema_version'")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as refusal:
        _open_store(Store(tmp_path))
    assert refusal.value.code == "schema_damaged"
    assert Store(tmp_path).active_owner() is None


def test_a_partial_producer_table_refuses_instead_of_being_repaired(tmp_path) -> None:
    store = _seed_full_store(tmp_path)
    _downgrade_to_v5(store._path, drop_table=True)
    conn = sqlite3.connect(store._path)
    try:
        conn.execute("CREATE TABLE screened_request_deliveries (delivery_ref TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(StoreRefused) as refusal:
        _open_store(Store(tmp_path))
    assert refusal.value.code == "schema_damaged"
    assert Store(tmp_path).active_owner() is None
