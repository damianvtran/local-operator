"""BODY reservations — the durable store slice (schema v8).

Pinned here against a REAL sqlite store, not a mock:

- a v7 store gains the additive ``body_reservations`` table, keeps every other
  row, and stamps v8 — a crash mid-upgrade simply retries;
- two consumers cannot both claim one reservation, and an EXPIRED claim is
  reclaimable by the next claimant in the same statement;
- a claim loser's completion is a no-op (it cannot write ``completed`` by
  primary key over a row another turn is serving);
- ``per_call_admission_state`` returns the four local facts over ONE read-only
  connection and creates nothing when the store is absent.

The table is contentless by construction: there is no body or label column, so
these tests never assert on text that cannot exist.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any

import pytest

from local_operator.team_bridge.store import (
    BODY_TERMINAL_CODES,
    SCHEMA_VERSION,
    OwnerReceipt,
    OwnerSnapshot,
    Store,
    StoreRefused,
    state_db_path,
)

CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
DIGEST = "9f2a" * 16
FINGERPRINT = "aa11" * 16


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_1",
        tenant_id="ten_1",
        endpoint_id="ep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


def _store(tmp_path) -> Store:
    store = Store(tmp_path)
    store.commit_enrollment(
        _owner(), OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="sess_cos"
    )
    return store


def _open(store: Store) -> None:
    """Run the REAL write-capable open path and let it raise."""

    with store._connect() as conn:
        store._init(conn)


def _reserve(store: Store, *, now: float | None = None, seq: int = 3) -> bool:
    return store.record_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        conversation_seq=seq,
        mention=True,
        body_sha256=DIGEST,
        screening_fingerprint=FINGERPRINT,
        policy_epoch=3,
        consent_version="bridge_consent_v1",
        designated_session="sess_cos",
        now=time.time() if now is None else now,
    )


def _policy_epoch(store: Store) -> int | None:
    with sqlite3.connect(state_db_path(store.config_dir)) as conn:
        row = conn.execute("SELECT epoch FROM screening_policy").fetchone()
    return int(row[0]) if row is not None else None


def test_a_v7_store_gains_the_table_keeps_rows_and_stamps_v8(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        # A pre-upgrade policy row makes "rows are preserved" a real assertion
        # rather than a tautology over an empty table.
        conn.execute(
            "INSERT INTO screening_policy (owner_key, version, template_sha256, "
            "default_model, revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (
                '["https://hub.example","acc_1","ten_1","ep_1"]',
                "v1",
                "aa11" * 16,
                "model-a",
                "rev-1",
                3,
                time.time(),
            ),
        )
        conn.execute("DROP TABLE body_reservations")
        conn.execute("UPDATE schema_meta SET value='7' WHERE key='schema_version'")
        conn.commit()

    _open(store)

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        enrollments = conn.execute("SELECT COUNT(*) FROM enrollments").fetchone()[0]
        policy = conn.execute("SELECT COUNT(*) FROM screening_policy").fetchone()[0]
    assert version == str(SCHEMA_VERSION) == "8"
    assert "body_reservations" in tables
    # The upgrade is ADDITIVE: authority rows written before it survive.
    assert enrollments == 1
    assert policy == 1


def test_a_real_v6_store_without_the_table_migrates(tmp_path) -> None:
    """B1: a genuine v6 store predates the table and must MIGRATE, not refuse."""

    store = _store(tmp_path)
    assert _reserve(store)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("DROP TABLE body_reservations")
        conn.execute("UPDATE schema_meta SET value='6' WHERE key='schema_version'")
        conn.commit()

    _open(store)

    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert version == str(SCHEMA_VERSION) == "8"
    assert "body_reservations" in tables


def test_a_partial_reservation_table_is_a_typed_refusal(tmp_path) -> None:
    """M1: a partial table must refuse as schema_damaged, not raise raw SQL."""

    store = _store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("DROP TABLE body_reservations")
        conn.execute("CREATE TABLE body_reservations (owner_key TEXT)")
        conn.execute("UPDATE schema_meta SET value='7' WHERE key='schema_version'")
        conn.commit()

    with pytest.raises(StoreRefused) as refused:
        _open(store)
    assert refused.value.code == "schema_damaged"


def test_write_path_refuses_text_in_identity_columns(tmp_path) -> None:
    """B2: the DDL has no body column; the write path must not smuggle text."""

    store = _store(tmp_path)
    with pytest.raises(StoreRefused) as body_in_id:
        store.record_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id="bcmsg_x/body=SECRET",
            conversation_seq=1,
            mention=False,
            body_sha256=DIGEST,
            screening_fingerprint=FINGERPRINT,
            policy_epoch=3,
            consent_version="bridge_consent_v1",
            designated_session="sess_cos",
            now=time.time(),
        )
    assert body_in_id.value.code == "invalid_body_reservation"

    with pytest.raises(StoreRefused):
        store.record_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            conversation_seq=1,
            mention=False,
            body_sha256="not-a-digest-SECRET",
            screening_fingerprint=FINGERPRINT,
            policy_epoch=3,
            consent_version="bridge_consent_v1",
            designated_session="sess_cos",
            now=time.time(),
        )


def test_release_refuses_a_reason_outside_the_closed_vocabulary(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )

    with pytest.raises(StoreRefused) as refused:
        store.release_body_claim(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            claim_token="tok-a",
            reason="the body said SECRET",
            now=now + 1,
        )
    assert refused.value.code == "invalid_body_reason"
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "claimed" and row["last_error"] is None


def _token_probe(tmp_path) -> Store:
    store = _store(tmp_path)
    assert _reserve(store)
    return store


def test_claim_and_completion_refuse_a_body_string_in_their_slots(tmp_path) -> None:
    """B2 (round 2): the capability columns are caller-supplied too."""

    store = _token_probe(tmp_path)
    now = time.time()

    with pytest.raises(StoreRefused) as bad_claim:
        store.claim_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            claim_token="tok SECRET body",
            turn_token="turn-a",
            now=now,
            ttl=300,
        )
    assert bad_claim.value.code == "invalid_body_reservation"

    with pytest.raises(StoreRefused):
        store.claim_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            claim_token="tok-a",
            turn_token="turn\nSECRET",
            now=now,
            ttl=300,
        )

    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )
    with pytest.raises(StoreRefused) as bad_entry:
        store.complete_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            claim_token="tok-a",
            entry_id="entry SECRET body",
            now=now + 1,
        )
    assert bad_entry.value.code == "invalid_body_reservation"


def test_completion_clears_the_claim_capability(tmp_path) -> None:
    store = _token_probe(tmp_path)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )
    assert store.complete_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        entry_id="entry-1",
        now=now + 1,
    )
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "completed"
    assert row["claim_token"] is None and row["turn_token"] is None
    assert row["claim_expires_at"] is None


@pytest.mark.parametrize("shape", ["no_key", "single_unique"])
def test_a_column_complete_table_with_the_wrong_key_shape_refuses(tmp_path, shape) -> None:
    """M1 (round 2): names alone are not the contract."""

    store = _store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(body_reservations)")]
        conn.execute("DROP TABLE body_reservations")
        body = ", ".join(f"{name} TEXT" for name in columns)
        extra = {
            "no_key": "",
            "single_unique": ", UNIQUE (owner_key)",
        }[shape]
        conn.execute(f"CREATE TABLE body_reservations ({body}{extra})")
        conn.execute("UPDATE schema_meta SET value='7' WHERE key='schema_version'")
        conn.commit()

    with pytest.raises(StoreRefused) as refused:
        _open(store)
    assert refused.value.code == "schema_damaged"


def test_two_consumers_cannot_both_claim(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()

    first = store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )
    second = store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-b",
        turn_token="turn-b",
        now=now,
        ttl=300,
    )

    assert (first, second) == (True, False)
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "claimed" and row["claim_token"] == "tok-a"


def test_an_expired_claim_is_reclaimed_by_the_next_claimant(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=1,
    )

    reclaimed = store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-b",
        turn_token="turn-b",
        now=now + 10,
        ttl=300,
    )

    assert reclaimed is True
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["claim_token"] == "tok-b"
    assert row["attempts"] == 2


def test_a_claim_loser_cannot_complete(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )

    won = store.complete_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        entry_id="entry-1",
        now=now + 1,
    )
    lost = store.complete_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-b",
        entry_id="entry-2",
        now=now + 2,
    )

    assert (won, lost) == (True, False)
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "completed" and row["completed_entry_id"] == "entry-1"


def test_renew_extends_only_for_the_holding_turn(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )

    renewed = store.renew_body_claim(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now + 10,
        ttl=300,
    )
    wrong_turn = store.renew_body_claim(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-other",
        now=now + 20,
        ttl=300,
    )

    assert (renewed, wrong_turn) == (True, False)


def test_release_returns_the_row_to_pending_with_its_reason(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)
    now = time.time()
    store.claim_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        turn_token="turn-a",
        now=now,
        ttl=300,
    )

    released = store.release_body_claim(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        claim_token="tok-a",
        reason="body_not_fitted",
        now=now + 1,
    )

    assert released is True
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "pending"
    assert row["last_error"] == "body_not_fitted"
    assert row["claim_token"] is None


def test_retire_requires_a_known_terminal_code(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store)

    with pytest.raises(StoreRefused):
        store.retire_body_reservation(
            _owner(),
            conversation_id=CONVERSATION_ID,
            message_id=MESSAGE_ID,
            terminal_code="made_up",
            now=time.time(),
        )

    assert store.retire_body_reservation(
        _owner(),
        conversation_id=CONVERSATION_ID,
        message_id=MESSAGE_ID,
        terminal_code=BODY_TERMINAL_CODES[0],
        now=time.time(),
    )
    row: dict[str, Any] = store.body_reservation(_owner(), CONVERSATION_ID, MESSAGE_ID)
    assert row["state"] == "retired" and row["terminal_code"] == BODY_TERMINAL_CODES[0]


def test_recording_twice_is_a_no_op(tmp_path) -> None:
    store = _store(tmp_path)
    assert _reserve(store) is True
    assert _reserve(store) is False


def test_per_call_admission_state_reads_the_four_facts(tmp_path) -> None:
    store = _store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "INSERT INTO screening_policy (owner_key, version, template_sha256, "
            "default_model, revision, epoch, accepted_at) VALUES (?,?,?,?,?,?,?)",
            (
                '["https://hub.example","acc_1","ten_1","ep_1"]',
                "v1",
                "aa11" * 16,
                "model-a",
                "rev-1",
                3,
                time.time(),
            ),
        )
        conn.commit()

    state = store.per_call_admission_state(_owner(), "sess_cos")

    assert state is not None
    assert state["owner_live"] is True
    assert state["designation_matches"] is True
    assert state["policy_epoch"] == _policy_epoch(store) == 3
    assert state["consent_version"] == "bridge_consent_v1"

    drifted = store.per_call_admission_state(_owner(), "sess_other")
    assert drifted is not None and drifted["designation_matches"] is False


def test_per_call_admission_state_creates_nothing_when_absent(tmp_path) -> None:
    absent = Store(tmp_path / "never-created")

    assert absent.per_call_admission_state(_owner(), "sess_cos") is None
    assert not absent._path.exists(), "a read-only check must not create the store"
