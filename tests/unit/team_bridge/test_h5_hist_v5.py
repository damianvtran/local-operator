"""H5 S-11 — the HISTORICAL baseline V5 store migrates to 6 (SOURCE).

The baseline here is built from the PINNED fa5d884c ``store.py`` DDL, quoted
verbatim from ``git show fa5d884c:local_operator/team_bridge/store.py`` (the
``_FLOORS_SCHEMA`` + ``_CORE_SCHEMA`` + ``_SCREENING_SCHEMA`` statements) — it
is the REAL old-era layout, not a current store downgraded by hand. The
constructed current-downgrade case remains separate in
``test_h5_schema6_migration.py``; this file never replaces that one.

Covers: preservation of populated old rows across the real migration; a
mid-DDL failure injected AFTER one producer statement succeeded proving the
whole upgrade rolls back (new table absent, stamp 5, rows byte-identical); and
damaged old floors still refusing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from local_operator.team_bridge import store as store_mod
from local_operator.team_bridge.store import (
    SCHEMA_VERSION,
    Store,
    StoreRefused,
    state_db_path,
)

# -- pinned fa5d884c DDL (verbatim) ------------------------------------------

_V5_FLOORS = """
CREATE TABLE IF NOT EXISTS text_floors (
    owner_key TEXT NOT NULL,
    hub TEXT NOT NULL,
    kind TEXT NOT NULL,
    version TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (owner_key, hub, kind)
);
"""

_V5_CORE = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS enrollments (
    owner_key TEXT PRIMARY KEY,
    credential_id INTEGER NOT NULL,
    account_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    canonical_base_url TEXT NOT NULL,
    enrollment_generation INTEGER NOT NULL,
    consent_version TEXT,
    consent_ref TEXT,
    designated_session TEXT,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sync (
    owner_key TEXT PRIMARY KEY,
    feed_seq INTEGER NOT NULL DEFAULT 0,
    pending_resync INTEGER NOT NULL DEFAULT 0,
    inbox_checkpoint TEXT,
    walk_generation INTEGER NOT NULL DEFAULT 0,
    walk_terminal INTEGER NOT NULL DEFAULT 0,
    last_observed_at REAL
);
CREATE TABLE IF NOT EXISTS deliveries (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    seq INTEGER,
    msg_type TEXT,
    origin TEXT,
    sender_account_id TEXT,
    sender_endpoint_id TEXT,
    recipient_account_id TEXT,
    recipient_endpoint_id TEXT,
    body_hash TEXT,
    delivery_state TEXT,
    state_at TEXT,
    request_state TEXT,
    request_expires_at TEXT,
    request_ref TEXT,
    pending_reconciliation INTEGER NOT NULL DEFAULT 0,
    local_hold_code TEXT,
    retry_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    blocked_reviewed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS ack_intents (
    intent_id TEXT PRIMARY KEY,
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    target TEXT NOT NULL,
    client_ref TEXT NOT NULL,
    verdict TEXT,
    model TEXT,
    prompt_version TEXT,
    template_sha256 TEXT,
    body_sha256 TEXT,
    categories TEXT,
    state TEXT NOT NULL,
    error_code TEXT,
    retry_after REAL,
    created_at REAL NOT NULL,
    UNIQUE (owner_key, message_id, target)
);
CREATE TABLE IF NOT EXISTS feed_refs (
    owner_key TEXT NOT NULL,
    ref TEXT NOT NULL,
    seq INTEGER NOT NULL,
    PRIMARY KEY (owner_key, ref)
);
CREATE TABLE IF NOT EXISTS claims (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    token TEXT NOT NULL,
    generation INTEGER NOT NULL,
    expires_at REAL NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
"""

_V5_SCREENING = """
CREATE TABLE IF NOT EXISTS screening_attempts (
    attempt_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    account_scope TEXT NOT NULL,
    owner_key TEXT NOT NULL,
    message_id TEXT,
    fingerprint TEXT,
    at REAL NOT NULL,
    committed_message_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_screening_attempts_scope_kind_at
    ON screening_attempts (account_scope, kind, at);
CREATE TABLE IF NOT EXISTS screening_receipts (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    body_sha256 TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    verdict TEXT NOT NULL,
    categories TEXT NOT NULL,
    envelope_digest TEXT NOT NULL,
    admission_revision TEXT NOT NULL,
    policy_epoch INTEGER NOT NULL,
    creator_generation INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS screening_holds (
    owner_key TEXT NOT NULL,
    message_id TEXT NOT NULL,
    hold_code TEXT NOT NULL,
    retry_at REAL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_key, message_id)
);
CREATE TABLE IF NOT EXISTS screening_schedule (
    owner_key TEXT PRIMARY KEY,
    last_sender TEXT,
    lease_holder TEXT,
    lease_expires REAL
);
CREATE TABLE IF NOT EXISTS screening_policy (
    owner_key TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    template_sha256 TEXT NOT NULL,
    default_model TEXT,
    revision TEXT NOT NULL,
    epoch INTEGER NOT NULL,
    accepted_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS screening_breaker (
    account_scope TEXT PRIMARY KEY,
    failures INTEGER NOT NULL,
    open_until REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

KEY = "k1"
NOW = 1_700_000_000.0

#: The tables whose rows must survive the upgrade untouched.
_COMPARED = (
    "enrollments",
    "claims",
    "ack_intents",
    "screening_receipts",
    "screening_policy",
    "screening_schedule",
    "screening_holds",
    "screening_attempts",
    "text_floors",
    "deliveries",
    "feed_refs",
    "sync",
)


def _baseline(tmp_path: Path) -> Path:
    """A genuine old-era V5 file: pinned DDL + stamp 5 + own-scope rows."""

    path = state_db_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_V5_FLOORS)
        conn.executescript(_V5_CORE)
        conn.executescript(_V5_SCREENING)
        conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '5')")
        conn.execute(
            "INSERT INTO enrollments (owner_key, credential_id, account_id, tenant_id, "
            "endpoint_id, canonical_base_url, enrollment_generation, consent_version, "
            "consent_ref, designated_session, state) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                KEY,
                7,
                "acc_1",
                "ten_1",
                "ep_1",
                "https://hub.example",
                3,
                "consent-v1",
                "ref-1",
                "s1",
                "committed",
            ),
        )
        conn.execute(
            "INSERT INTO claims (owner_key, message_id, session_id, token, generation, "
            "expires_at, state) VALUES (?,?,?,?,?,?,?)",
            (KEY, "bmsg_1", "s1", "tok-1", 3, NOW + 300, "held"),
        )
        conn.execute(
            "INSERT INTO ack_intents (intent_id, owner_key, message_id, target, client_ref, "
            "verdict, model, prompt_version, template_sha256, body_sha256, categories, "
            "state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "intent-1",
                KEY,
                "bmsg_1",
                "read",
                "cref-1",
                "allow",
                "model-a",
                "prompt-v1",
                "aa11" * 16,
                "9f2a" * 16,
                '["spam"]',
                "confirmed",
                NOW,
            ),
        )
        conn.execute(
            "INSERT INTO screening_receipts (owner_key, message_id, body_sha256, "
            "template_sha256, prompt_version, model, verdict, categories, envelope_digest, "
            "admission_revision, policy_epoch, creator_generation, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                KEY,
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
                3,
                NOW,
            ),
        )
        conn.execute(
            "INSERT INTO screening_policy (owner_key, version, template_sha256, default_model, "
            "revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (KEY, "v1", "aa11" * 16, "model-a", "rev-1", 3, NOW),
        )
        conn.execute(
            "INSERT INTO screening_schedule (owner_key, last_sender, lease_holder, lease_expires) "
            "VALUES (?,?,?,?)",
            (KEY, "snd-1", "holder-1", NOW + 60),
        )
        conn.execute(
            "INSERT INTO screening_holds (owner_key, message_id, hold_code, retry_at, created_at) "
            "VALUES (?,?,?,?,?)",
            (KEY, "bmsg_0", "screening_hold", NOW + 30, NOW),
        )
        conn.execute(
            "INSERT INTO screening_attempts (attempt_id, kind, account_scope, owner_key, "
            "message_id, fingerprint, at, committed_message_id) VALUES (?,?,?,?,?,?,?,?)",
            ("att-1", "logical", "scope-1", KEY, "bmsg_1", "fp-1", NOW, "bmsg_1"),
        )
        conn.execute(
            "INSERT INTO text_floors (owner_key, hub, kind, version, sha256, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (KEY, "hub-1", "guidance", "v1", "cc33" * 16, NOW),
        )
        conn.execute(
            "INSERT INTO deliveries (owner_key, message_id, thread_id, tenant_id, body_hash, "
            "delivery_state) VALUES (?,?,?,?,?,?)",
            (KEY, "bmsg_1", "t-1", "ten_1", "9f2a" * 16, "unread"),
        )
        conn.execute("INSERT INTO feed_refs (owner_key, ref, seq) VALUES (?,?,?)", (KEY, "r-1", 5))
        conn.execute("INSERT INTO sync (owner_key, feed_seq) VALUES (?,?)", (KEY, 5))
        conn.commit()
    finally:
        conn.close()
    return path


def _snapshot(path: Path) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(path)
    try:
        return {
            table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
            for table in _COMPARED
        }
    finally:
        conn.close()


def _version(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        return str(row[0]) if row else ""
    finally:
        conn.close()


def _table_exists(path: Path, name: str) -> bool:
    conn = sqlite3.connect(path)
    try:
        return (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            is not None
        )
    finally:
        conn.close()


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


def test_pinned_historical_v5_migrates_to_6_preserving_every_row(tmp_path: Path) -> None:
    path = _baseline(tmp_path)
    before = _snapshot(path)
    store = Store(tmp_path)
    assert store.active_owner() is not None  # the migration ran on first touch
    assert _version(path) == str(SCHEMA_VERSION) == "6"
    assert _snapshot(path) == before  # byte-for-byte on every compared table
    assert _table_exists(path, "screened_request_deliveries")
    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM screened_request_deliveries").fetchone()[0] == 0
    finally:
        conn.close()


def test_mid_ddl_failure_rolls_back_and_preserves_the_v5_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _baseline(tmp_path)
    before = _snapshot(path)
    original = store_mod._ensure_producer_table

    def exploding(conn: sqlite3.Connection) -> None:
        # Inject AFTER one producer statement has ACTUALLY executed: the whole
        # upgrade (validation + DDL + stamp) rides ONE BEGIN IMMEDIATE, so the
        # rollback must remove the partial table and restore the old stamp.
        conn.execute(store_mod._PRODUCER_DDL_STATEMENTS[0])
        raise StoreRefused("injected_failure", "mid-DDL cut")

    monkeypatch.setattr(store_mod, "_ensure_producer_table", exploding)
    store = Store(tmp_path)
    with pytest.raises(StoreRefused) as refused:
        _open_store(store)
    assert refused.value.code == "injected_failure"
    assert store.active_owner() is None  # the reader fails closed, never half-open
    # On disk: old stamp, NO producer table, every row intact.
    assert _version(path) == "5"
    assert not _table_exists(path, "screened_request_deliveries")
    assert _snapshot(path) == before
    # With the injection lifted, the SAME file migrates cleanly.
    monkeypatch.setattr(store_mod, "_ensure_producer_table", original)
    reopened = Store(tmp_path)
    assert reopened.active_owner() is not None
    assert _version(path) == "6"
    assert _snapshot(path) == before


def test_damaged_v5_floors_still_refuse(tmp_path: Path) -> None:
    path = _baseline(tmp_path)
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE text_floors")
        conn.execute(
            "CREATE TABLE text_floors (owner_key TEXT, hub TEXT, kind TEXT, version TEXT, "
            "updated_at REAL)"
        )
        conn.commit()
    finally:
        conn.close()
    store = Store(tmp_path)
    with pytest.raises(StoreRefused) as refused:
        _open_store(store)
    assert refused.value.code == "schema_damaged"
    assert store.active_owner() is None  # no authority from a damaged store
    # Nothing was "fixed": the damaged file is still stamped 5.
    assert _version(path) == "5"
    assert not _table_exists(path, "screened_request_deliveries")
