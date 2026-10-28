"""H2 store: fail-closed authority, generation fences, CAS claims, write-once intents."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from local_operator.team_bridge.store import (
    ACK_PENDING,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import (
    MessageRef,
    OwnerSnapshot,
    ScreeningReceipt,
    feed_seq,
)


def _owner(credential_id: int = 7, generation: int = 1) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=credential_id,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=generation,
    ).validated()


def _receipt(verdict: str = "allow") -> ScreeningReceipt:
    return ScreeningReceipt(
        verdict=verdict,
        model="m",
        prompt_version="bridge-screen-v1",
        template_sha256="a" * 64,
        body_sha256="b" * 64,
        categories=("spam", "link"),
    )


def _ref(message_id: str = "bmsg_1") -> MessageRef:
    return MessageRef(
        message_id=message_id,
        thread_id="bth_1",
        tenant_id="tnt_1",
        message_type="note",
        body_hash="c" * 64,
        seq=1,
        origin="member",
        delivery_state="unread",
        state_at="2026-10-12T00:00:00Z",
        sender_account_id="acc_b",
        sender_endpoint_id="bep_2",
        recipient_account_id="acc_a",
        recipient_endpoint_id="bep_1",
        request_state=None,
        request_expires_at=None,
        request_ref=None,
    ).validated()


def test_missing_or_corrupt_state_is_no_authority(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.active_owner() is None
    db = state_db_path(tmp_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"not a sqlite file")
    assert store.active_owner() is None
    with pytest.raises(StoreRefused):
        store.merge_page(_owner(), [_ref()], None)


def test_enrollment_roundtrip_and_credential_exactness(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    assert store.active_owner(credential_id=7) == owner
    # A same-account re-login with a NEW credential row is a DIFFERENT owner.
    assert store.active_owner(credential_id=9) is None


def OwnerReceiptForTest():
    from local_operator.team_bridge.store import OwnerReceipt

    return OwnerReceipt(consent_version="bridge_consent_v1", consent_ref="ref-1")


def test_stale_generation_refuses_and_fences_claims(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    store.claim(owner, "bmsg_1", "sess-1", "tok-1", expiry=time.time() + 60)
    live = store.invalidate(owner)
    assert live == 2
    with pytest.raises(StoreRefused) as refusal:
        store.merge_page(owner, [_ref()], None)
    assert refusal.value.code == "stale_generation"
    with pytest.raises(StoreRefused):
        store.claim(owner, "bmsg_2", "sess-1", "tok-2", expiry=time.time() + 60)


def test_claim_cas_two_contenders_expiry_and_stale_release(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    store.claim(owner, "bmsg_1", "sess-1", "tok-1", expiry=time.time() + 0.2)
    with pytest.raises(StoreRefused) as refusal:
        store.claim(owner, "bmsg_1", "sess-2", "tok-2", expiry=time.time() + 60)
    assert refusal.value.code == "claimed"
    # Expired: a NEW token may take it.
    time.sleep(0.25)
    store.claim(owner, "bmsg_1", "sess-2", "tok-2", expiry=time.time() + 60)
    with pytest.raises(StoreRefused) as refusal:
        store.release_claim(owner, "bmsg_1", "tok-1")
    assert refusal.value.code == "not_holder"
    store.release_claim(owner, "bmsg_1", "tok-2")


def test_write_once_intent_idempotent_and_conflict(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    first = store.record_ack_intent(owner, "bmsg_1", "read", _receipt("allow"), "r" * 36)
    again = store.record_ack_intent(owner, "bmsg_1", "read", _receipt("allow"), "r" * 36)
    assert first == again
    with pytest.raises(StoreRefused) as refusal:
        store.record_ack_intent(owner, "bmsg_1", "read", _receipt("reject"), "r" * 36)
    assert refusal.value.code == "receipt_required"
    intent = store.ack_intent(owner, first)
    assert intent is not None and intent["state"] == ACK_PENDING
    assert store.ack_intent(owner, "missing") is None


def test_feed_refs_durable_and_drained(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    store.record_feed_batch(owner, feed_seq(5), ["ev-1", "ev-2"], resync_required=False)
    assert store.drain_feed_refs(owner, 5) == ["ev-1", "ev-2"]
    assert store.sync_state(owner)["feed_seq"] == 5


def test_unsupported_schema_is_no_authority(tmp_path: Path) -> None:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, receipt=OwnerReceiptForTest())
    db = state_db_path(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE schema_meta SET value = '99' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    assert store.active_owner() is None
    with pytest.raises(StoreRefused):
        store.merge_page(owner, [_ref()], None)


def test_concurrent_settings_commits_never_create_two_owners(tmp_path: Path) -> None:
    """R2-2: the one-owner rule is enforced INSIDE the store transaction.

    The seam under test is the one the client calls
    (``connector.enrollment.commit_settings_enrollment``), driven from two
    threads — as it now runs, in a worker thread, since R1-1 moved it off the
    event loop. Count-then-commit was atomic only while both steps shared the
    loop: two enables on a fresh store both saw zero owners and both inserted,
    and a two-row store answers ``active_owner() is None`` for ever — a dark
    bridge that never heals (the reviewer measured 3/40).

    Forty rounds, a fresh store each, two different endpoints per round: every
    round must end with AT MOST one committed owner, and at least one, since
    ``BEGIN IMMEDIATE`` serialises the pair rather than dropping both.
    """

    import concurrent.futures

    from local_operator.team_bridge.connector.enrollment import (
        commit_settings_enrollment,
    )
    from local_operator.team_bridge.store import Store

    for round_index in range(40):
        root = tmp_path / f"round-{round_index}"
        # INITIALISE THE SCHEMA FIRST, single-threaded. Two connections
        # bootstrapping a brand-new database race on ``schema_meta`` (a
        # pre-existing, separate concern), and on a real machine the store
        # exists long before two enables race — the status read and the
        # enrolment gate both open it. What this test pins is the OWNER rule.
        Store(root).committed_owner_count()

        def attempt(index: int) -> bool:
            return commit_settings_enrollment(
                config_dir=root,
                credential_id=1,
                account_id=f"acc_{round_index}_{index}",
                tenant_id=f"tnt_{index}",
                endpoint_id=f"bep_{index}",
                consent_version="bridge_consent_v1",
                canonical_base_url="https://hub.example/v1",
                cos_session_id=None,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, (0, 1)))

        committed = Store(root).committed_owner_count()
        assert committed <= 1, f"round {round_index}: {committed} committed owners"
        assert committed == 1, f"round {round_index}: neither enable committed"
        assert results.count(True) == 1, (round_index, results)
        owner = Store(root).active_owner()
        assert owner is not None, f"round {round_index}: an owner was committed but is unusable"
