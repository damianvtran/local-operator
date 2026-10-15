"""H2 remediation round 1 — independent repros for the batched QA/code findings.

Each test reproduces one discriminating case from the preflight memos against
the FIXED head (they fail on `1bd37c25` — the pre-fix baseline — and pass
here): destination binding (Q1), frozen-DTO normalization and refs-only feed
projection (Q2), atomic page adoption and the resync witness (Q3), ack
ambiguity vs terminal verdicts (Q4), authority bootstrapping and generation
monotonicity (Q5), atomic claim CAS (Q6) and the full write-once receipt
tuple (Q7). No network, no services: injected wire or native SQLite only.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge import pull
from local_operator.team_bridge.client import (
    BridgeRefusal,
    BridgeUnreachable,
    TeamBridgeClient,
)
from local_operator.team_bridge.store import (
    ACK_CONFIRMED,
    ACK_CONFLICT,
    ACK_PENDING,
    OwnerReceipt,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import (
    OwnerSnapshot,
    ScreeningReceipt,
    inbox_cursor,
)


def _owner(generation: int = 1) -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=generation,
    ).validated()


def _access() -> OAuthAccess:
    return OAuthAccess(access_token="tok", credential_id=7, account_id="acc_a", kind="oauth")


def _store_with_owner(tmp_path: Path) -> tuple[Store, OwnerSnapshot]:
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    return store, owner


def _receipt(body: str = "c" * 64) -> ScreeningReceipt:
    return ScreeningReceipt(
        verdict="allow",
        model="qa/model",
        prompt_version="bridge-screen-v1",
        template_sha256="b" * 64,
        body_sha256=body,
        categories=("link",),
    )


def _item(
    *,
    message_id: str = "bmsg_1",
    recipient_endpoint: str = "bep_1",
    tenant_on_message: bool = False,
) -> dict[str, Any]:
    """A REAL server-DTO-shaped inbox item (message has no tenant; thread does)."""

    message = {
        "id": message_id,
        "thread_id": "bth_1",
        "type": "note",
        "seq": 1,
        "origin": "user",
        "body_hash": "c" * 64,
        "sender": {"account_id": "acc_b", "endpoint_id": "bep_2"},
        "recipient": {"account_id": "acc_a", "endpoint_id": recipient_endpoint},
        "body": {"text": "CANARY-BODY-TEXT"},
    }
    if tenant_on_message:
        message["tenant_id"] = "tnt_1"
    return {
        "message": message,
        "delivery": {"state": "unread", "state_at": "t"},
        "thread": {"id": "bth_1", "tenant_id": "tnt_1"},
        "sender": {"account_id": "acc_b", "endpoint_id": "bep_2", "name": "N", "email": "e@x"},
    }


class FakeWire:
    #: The explicit injected-fixture seam: a fake must NAME its destination
    #: like a real client (the guard fails closed without one).
    base_url = "https://qa.invalid/v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.items: list[dict[str, Any]] = [_item()]
        self.next_cursor: str | None = None
        self.ack_result: dict[str, Any] = {"delivery": {"state": "read", "state_at": "t"}}
        self.ack_error: Exception | None = None
        self.event_result: dict[str, Any] | None = None

    async def inbox(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("inbox", kwargs))
        return {"items": self.items, "next_cursor": self.next_cursor, "counts": {}}

    async def message(self, message_id, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("message", kwargs))
        return self.items[0]

    async def ack(self, message_id, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("ack", kwargs))
        if self.ack_error is not None:
            raise self.ack_error
        return self.ack_result

    async def events(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("events", kwargs))
        return self.event_result


# ---------------------------------------------------------------------------
# Q1 — exact destination + recipient binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q1_foreign_origin_refused_before_dispatch(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    wire.base_url = "https://foreign.invalid/v1"  # type: ignore[attr-defined]
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert refusal.value.code == "destination_mismatch"
    assert wire.calls == []
    assert store.list_metadata(owner) == []


@pytest.mark.asyncio
async def test_q1_real_sdk_foreign_hub_refused(tmp_path: Path) -> None:
    """QA's discriminating case: the CONCRETE client is pointed at another hub."""

    _store, owner = _store_with_owner(tmp_path)
    client = TeamBridgeClient(tmp_path, base_url="https://foreign.invalid/v1")
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(
            owner, config_dir=tmp_path, access=_access(), client=cast(pull.WireClient, client)
        )
    assert refusal.value.code == "destination_mismatch"


@pytest.mark.asyncio
async def test_q1_foreign_and_missing_recipient_refuse_before_persistence(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    for item, label in ((_item(recipient_endpoint="bep_foreign"), "foreign"), (_item(), "missing")):
        wire = FakeWire()
        wire.items = [_item(recipient_endpoint="bep_foreign") if label == "foreign" else _item()]
        if label == "missing":
            del wire.items[0]["message"]["recipient"]
        with pytest.raises(pull.PullRefused) as refusal:
            await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
        assert refusal.value.code == "wrong_recipient"
        assert store.list_metadata(owner) == []
    # Single-message read is not a loophole.
    wire = FakeWire()
    wire.items = [_item(recipient_endpoint="bep_foreign")]
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.message(owner, "bmsg_1", config_dir=tmp_path, access=_access(), client=wire)
    assert refusal.value.code == "wrong_recipient"
    assert store.list_metadata(owner) == []
    # The store itself refuses the same way, transaction rolled back.
    from local_operator.team_bridge.types import project_message_ref

    bad_ref = project_message_ref(_item(recipient_endpoint="bep_foreign"))
    with pytest.raises(StoreRefused) as refusal:
        Store(tmp_path).merge_refs(owner, [bad_ref])
    assert refusal.value.code == "wrong_recipient"
    assert store.list_metadata(owner) == []


# ---------------------------------------------------------------------------
# Q2 — frozen DTO normalization; canonical decimal cursor; refs-only feed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q2_thread_tenant_normalized_and_no_display_block_persisted(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    page = await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert page.items[0].tenant_id == "tnt_1"  # derived from the THREAD block
    db_bytes = state_db_path(tmp_path).read_bytes()
    assert b"CANARY-BODY-TEXT" not in db_bytes
    assert b"e@x" not in db_bytes and b'"N"' not in db_bytes


@pytest.mark.asyncio
async def test_q2_decimal_string_cursor_becomes_int_feed_seq_and_canaries_absent(
    tmp_path: Path,
) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    wire.event_result = {
        "events": [
            {
                "type": "message_available",
                "ids": {"message_id": "bmsg_1", "thread_id": "bth_1"},
                "at": "2026-10-12T00:00:00Z",
                "body": "FAKE_EVENT_BODY_CANARY",
                "raw_http_detail": "FAKE_EVENT_RAW_CANARY",
            }
        ],
        "cursor": "42",
    }
    result = await pull.events(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert result == {
        "cursor": 42,
        "refs": ["message_available:message_id=bmsg_1"],
        "resync": False,
    }
    assert store.sync_state(owner)["feed_seq"] == 42
    assert isinstance(store.sync_state(owner)["feed_seq"], int)
    db_bytes = state_db_path(tmp_path).read_bytes()
    assert b"FAKE_EVENT_BODY_CANARY" not in db_bytes
    assert b"FAKE_EVENT_RAW_CANARY" not in db_bytes
    refs = store.drain_feed_refs(owner, 42)
    assert refs == ["message_available:message_id=bmsg_1"]


@pytest.mark.asyncio
async def test_q2_bad_event_shapes_refuse_whole_batch(tmp_path: Path) -> None:
    _store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    for bad in (
        {"type": "bogus", "ids": {"message_id": "m"}, "at": "t"},
        {"type": "state_changed", "ids": {"message_id": 17}, "at": "t"},
        {"type": "state_changed", "ids": {"message_id": "m"}, "at": ""},
        "not-a-mapping",
    ):
        # An INT cursor here on purpose: this test isolates the event-SHAPE
        # axis, so the answer must fail ONLY on the event itself.
        wire.event_result = {"events": [bad], "cursor": 5}
        with pytest.raises(pull.PullRefused) as refusal:
            await pull.events(owner, config_dir=tmp_path, access=_access(), client=wire)
        assert refusal.value.code == "malformed_events"


# ---------------------------------------------------------------------------
# Q3 — atomic page adoption; resync witness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q3_malformed_page_item_refuses_whole_page(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    # tenant_on_message=True: the item is projectable on BOTH heads, so the
    # refusal can only come from the malformed page shape under test.
    wire.items = [_item(tenant_on_message=True), 17]  # type: ignore[list-item]
    wire.next_cursor = "opaque-page2"
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert refusal.value.code == "malformed_page"
    assert store.list_metadata(owner) == []
    assert store.sync_state(owner)["inbox_checkpoint"] is None


@pytest.mark.asyncio
async def test_q3_non_string_cursor_refuses(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    wire.items = [_item(tenant_on_message=True)]
    wire.next_cursor = 17  # type: ignore[assignment]
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert refusal.value.code == "malformed_page"
    assert store.list_metadata(owner) == []


@pytest.mark.asyncio
async def test_q3_finish_resync_refuses_unfinished_walk(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    wire = FakeWire()
    wire.event_result = {"events": [], "cursor": "9"}
    await pull.events(owner, config_dir=tmp_path, access=_access(), client=wire)
    assert store.sync_state(owner)["pending_resync"] is True
    # A page with a CONTINUATION merged, then finish early: refused.
    wire.items = [_item()]
    wire.next_cursor = "opaque-page2"
    await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    with pytest.raises(StoreRefused) as refusal:
        pull.finish_resync(owner, inbox_cursor("baseline"), config_dir=tmp_path)
    assert refusal.value.code == "walk_incomplete"
    # After the terminal page, the witness opens.
    wire.next_cursor = None
    await pull.inbox(owner, config_dir=tmp_path, access=_access(), client=wire)
    pull.finish_resync(owner, inbox_cursor("baseline"), config_dir=tmp_path)
    assert store.sync_state(owner)["pending_resync"] is False


# ---------------------------------------------------------------------------
# Q4 — ack ambiguity stays pending; only a 409 is terminal; malformed 200
# ---------------------------------------------------------------------------


async def _intent(tmp_path: Path, owner: OwnerSnapshot) -> str:
    store = Store(tmp_path)
    return store.record_ack_intent(owner, "bmsg_1", "read", _receipt(), "ref-stable")


@pytest.mark.asyncio
async def test_q4_transport_failure_stays_pending_and_replays_same_request(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _intent(tmp_path, owner)
    wire = FakeWire()
    wire.ack_error = BridgeUnreachable()
    first = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert first.state == ACK_PENDING
    # Lost response: the second flush re-dispatches the SAME tuple + ref.
    wire.ack_error = None
    second = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert second.state == ACK_CONFIRMED
    assert [c[0] for c in wire.calls] == ["ack", "ack"]
    assert wire.calls[0][1]["client_ref"] == wire.calls[1][1]["client_ref"] == "ref-stable"
    assert wire.calls[0][1]["screening"] == wire.calls[1][1]["screening"]


@pytest.mark.asyncio
async def test_q4_rate_limited_and_503_stay_pending_with_retry_after(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _intent(tmp_path, owner)
    wire = FakeWire()
    wire.ack_error = BridgeRefusal(
        "rate_limited", "slow down", http_status=429, retry_after_seconds=30
    )
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert outcome.state == ACK_PENDING and outcome.retry_after == 30
    row = store.ack_intent(owner, intent_id)
    assert row is not None and row["state"] == ACK_PENDING and row["retry_after"] == 30
    wire.ack_error = BridgeRefusal("messaging_unavailable", "down", http_status=503)
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert outcome.state == ACK_PENDING


@pytest.mark.asyncio
async def test_q4_409_conflict_is_terminal_and_not_resent(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _intent(tmp_path, owner)
    wire = FakeWire()
    wire.ack_error = BridgeRefusal("invalid_transition", "differing verdict", http_status=409)
    first = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert first.state == ACK_CONFLICT
    second = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert second.state == ACK_CONFLICT
    assert [c[0] for c in wire.calls] == ["ack"]  # NO second dispatch


@pytest.mark.asyncio
async def test_q4_malformed_success_does_not_confirm(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _intent(tmp_path, owner)
    wire = FakeWire()
    wire.ack_result = {}
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert outcome.state == ACK_PENDING and outcome.error_code == "malformed_ack"
    row = store.ack_intent(owner, intent_id)
    assert row is not None and row["state"] == ACK_PENDING
    # And it is still replayable to a real confirmation.
    wire.ack_result = {"delivery": {"state": "read", "state_at": "t"}}
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=wire
    )
    assert outcome.state == ACK_CONFIRMED


# ---------------------------------------------------------------------------
# Q5 — authority bootstrap, monotonic generation, namespace ambiguity
# ---------------------------------------------------------------------------


def test_q5_unversioned_nonempty_db_refuses_authority(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    db = state_db_path(tmp_path)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM schema_meta WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    fresh = Store(tmp_path)
    assert fresh.active_owner() is None
    with pytest.raises(StoreRefused) as refusal:
        fresh.merge_page(owner, [], None)
    assert refusal.value.code == "schema_unversioned"


def test_q5_stale_recommit_cannot_roll_generation_back(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    live = store.invalidate(owner)
    assert live == 2
    with pytest.raises(StoreRefused) as refusal:
        store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-2"))
    assert refusal.value.code == "stale_enrollment"
    current = store.active_owner(credential_id=7)
    assert current is not None and current.enrollment_generation == 2


def test_q5_two_namespaces_are_never_guessed_between(tmp_path: Path) -> None:
    store = Store(tmp_path)
    first = _owner()
    second = OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_other",
        endpoint_id="bep_other",
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=1,
    ).validated()
    store.commit_enrollment(first, OwnerReceipt("v1", "ref-1"))
    store.commit_enrollment(second, OwnerReceipt("v1", "ref-2"))
    # The partial selector must NOT pick one: ambiguity is not authority.
    assert store.active_owner(credential_id=7) is None
    # The FULL namespace selects exactly, and BOTH rows are retained.
    resolved = store.resolve_owner(
        credential_id=7, account_id="acc_a", tenant_id="tnt_other", endpoint_id="bep_other"
    )
    assert resolved == second
    assert (
        store.resolve_owner(
            credential_id=7, account_id="acc_a", tenant_id="tnt_1", endpoint_id="bep_1"
        )
        == first
    )


# ---------------------------------------------------------------------------
# Q6 — atomic claim CAS
# ---------------------------------------------------------------------------


def test_q6_two_connections_exactly_one_winner(tmp_path: Path) -> None:
    _store, owner = _store_with_owner(tmp_path)
    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def contender(token: str) -> None:
        store = Store(tmp_path)
        barrier.wait()
        try:
            store.claim(owner, "bmsg_1", f"sess-{token}", token, expiry=time.time() + 60)
            outcome = "acquired"
        except StoreRefused as refusal:
            outcome = f"refused:{refusal.code}"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=contender, args=(t,)) for t in ("tok-a", "tok-b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(results) == ["acquired", "refused:claimed"]
    row = (
        sqlite3.connect(state_db_path(tmp_path))
        .execute("SELECT state FROM claims WHERE message_id = 'bmsg_1'")
        .fetchone()
    )
    assert row == ("held",)


def test_q6_expired_operations_all_refused(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    store.claim(owner, "bmsg_1", "sess-a", "tok-a", expiry=time.time() + 0.2)
    conn = sqlite3.connect(state_db_path(tmp_path))
    conn.execute("UPDATE claims SET expires_at = ?", (time.time() - 5,))
    conn.commit()
    conn.close()
    with pytest.raises(StoreRefused) as refusal:
        store.renew_claim(owner, "bmsg_1", "tok-a", time.time() + 60)
    assert refusal.value.code == "expired_claim"
    with pytest.raises(StoreRefused) as refusal:
        store.claim(owner, "bmsg_1", "sess-b", "tok-a", expiry=time.time() + 60)
    assert refusal.value.code == "reused_claim_token"
    store.claim(owner, "bmsg_1", "sess-b", "tok-b", expiry=time.time() + 60)
    with pytest.raises(StoreRefused) as refusal:
        store.release_claim(owner, "bmsg_1", "tok-a")
    assert refusal.value.code == "not_holder"
    store.release_claim(owner, "bmsg_1", "tok-b")


# ---------------------------------------------------------------------------
# Q7 — full write-once tuple; malformed receipts never persist
# ---------------------------------------------------------------------------


def test_q7_malformed_receipt_refused_before_persistence(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    for bad in (
        ScreeningReceipt("allow", "", "bridge-screen-v1", "b" * 64, "c" * 64, ()),
        ScreeningReceipt("allow", "m", "", "b" * 64, "c" * 64, ()),
        ScreeningReceipt("allow", "m", "v", "not-a-hash", "c" * 64, ()),
        ScreeningReceipt("allow", "m", "v", "b" * 64, "not-a-hash", ()),
    ):
        with pytest.raises(StoreRefused) as refusal:
            store.record_ack_intent(owner, "bmsg_1", "read", bad, "ref-1")
        assert refusal.value.code == "invalid_receipt"
    conn = sqlite3.connect(state_db_path(tmp_path))
    assert conn.execute("SELECT COUNT(*) FROM ack_intents").fetchone() == (0,)
    conn.close()


def test_q7_every_tuple_field_is_write_once(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    receipt = _receipt()
    first = store.record_ack_intent(owner, "bmsg_1", "read", receipt, "ref-1")
    assert store.record_ack_intent(owner, "bmsg_1", "read", receipt, "ref-1") == first
    for changed, label in (
        (
            ScreeningReceipt("allow", "OTHER", "bridge-screen-v1", "b" * 64, "c" * 64, ("link",)),
            "model",
        ),
        (ScreeningReceipt("allow", "qa/model", "v2", "b" * 64, "c" * 64, ("link",)), "prompt"),
        (
            ScreeningReceipt(
                "allow", "qa/model", "bridge-screen-v1", "d" * 64, "c" * 64, ("link",)
            ),
            "template",
        ),
        (
            ScreeningReceipt(
                "allow", "qa/model", "bridge-screen-v1", "b" * 64, "e" * 64, ("link",)
            ),
            "body",
        ),
        (
            ScreeningReceipt("allow", "qa/model", "bridge-screen-v1", "b" * 64, "c" * 64, ("x",)),
            "categories",
        ),
    ):
        with pytest.raises(StoreRefused) as refusal:
            store.record_ack_intent(owner, "bmsg_1", "read", changed, "ref-1")
        assert refusal.value.code == "intent_conflict", label
    with pytest.raises(StoreRefused) as refusal:
        store.record_ack_intent(owner, "bmsg_1", "read", receipt, "ref-OTHER")
    assert refusal.value.code == "intent_conflict"
    # Reordered categories agree (the server compares as a set).
    reordered = ScreeningReceipt(
        "allow", "qa/model", "bridge-screen-v1", "b" * 64, "c" * 64, ("link",)
    )
    assert store.record_ack_intent(owner, "bmsg_1", "read", reordered, "ref-1") == first
