"""H2 residual remediation round 2 — repros for the QA R2 majors (Q4/Q5/Q6) + R5.

Three of these tests use a REAL two-connection schedule: a test-only Store
subclass pauses the method under test after its actual SELECT (the cursor is
closed by fetchone before the pause completes), lets an independent Store
connection run its own real mutations, then resumes. No SQL result is faked —
every statement executed is the store's own.

Discrimination shape (run against bfcf852808 = pre-fix, then here):
- pre-fix: the second connection's invalidate/claim completes DURING the
  pause and the resumed old operation alters the new successor row;
- fixed: the resumed operation holds the write lock, so the second
  connection is refused busy during the pause and its retry lands after —
  and the successor is never touched.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge import pull
from local_operator.team_bridge.client import BridgeRefusal
from local_operator.team_bridge.store import (
    ACK_CONFIRMED,
    ACK_PENDING,
    OwnerReceipt,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import (
    MessageRef,
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


def _ref(message_id: str = "bmsg_1") -> MessageRef:
    return MessageRef(
        message_id=message_id,
        thread_id="bth_1",
        tenant_id="tnt_1",
        message_type="note",
        body_hash="c" * 64,
        seq=1,
        origin="user",
        delivery_state="unread",
        state_at="t",
        sender_account_id="acc_b",
        sender_endpoint_id="bep_2",
        recipient_account_id="acc_a",
        recipient_endpoint_id="bep_1",
        request_state=None,
        request_expires_at=None,
        request_ref=None,
    ).validated()


def _successor_row(tmp_path: Path, message_id: str = "bmsg_1") -> Any:
    conn = sqlite3.connect(state_db_path(tmp_path))
    try:
        return conn.execute(
            "SELECT session_id, token, generation, state, expires_at FROM claims "
            "WHERE message_id = ?",
            (message_id,),
        ).fetchone()
    finally:
        conn.close()


class _PausingStore(Store):
    """Pause the method under test after its Nth real SELECT (test seam).

    The pause happens INSIDE the method's own transaction, so what the second
    connection can or cannot do while it is held is exactly the property under
    test. Statements are real; only timing is controlled.
    """

    def __init__(self, config_dir: Path, marker: str, pause_on: int = 1) -> None:
        super().__init__(config_dir)
        self._marker = marker
        self._pause_on = pause_on
        self._hits = 0
        self.paused = threading.Event()
        self.resume = threading.Event()

    @contextmanager
    def _connect(self) -> Iterator[Any]:
        pauser = self
        with super()._connect() as conn:

            class _Proxy:
                def execute(self, sql: str, *args: Any) -> Any:
                    cur = conn.execute(sql, *args)
                    if pauser._marker in sql:
                        pauser._hits += 1
                        if pauser._hits == pauser._pause_on:
                            pauser.paused.set()
                            pauser.resume.wait(timeout=5)
                    return cur

                def __getattr__(self, name: str) -> Any:
                    return getattr(conn, name)

            yield _Proxy()


def _b_during_pause(tmp_path: Path, owner: OwnerSnapshot, message_id: str) -> bool:
    """Try the invalidate+reclaim during the pause. True if it COMPLETED.

    On the fixed head the resumed operation holds the write lock, so this
    raises ``busy`` after the busy timeout; on the pre-fix head it completes
    (no lock), which is the defect the successor assertions then catch.
    """

    store2 = Store(tmp_path)
    try:
        store2.invalidate(owner)
    except StoreRefused as refusal:
        assert refusal.code == "busy"
        return False
    store2.claim(_owner(2), message_id, "session-new", "new-token", expiry=time.time() + 60)
    return True


def _retry_reclaim(tmp_path: Path, owner: OwnerSnapshot, message_id: str) -> None:
    """After the old operation commits: invalidate + reclaim, bounded retry."""

    store2 = Store(tmp_path)
    for _ in range(40):
        try:
            current = store2.active_owner(credential_id=7)
            if current is not None and current.enrollment_generation == 1:
                store2.invalidate(owner)
        except StoreRefused as refusal:
            if refusal.code != "busy":
                raise
            time.sleep(0.05)
            continue
        try:
            store2.claim(_owner(2), message_id, "session-new", "new-token", expiry=time.time() + 60)
            return
        except StoreRefused as refusal:
            if refusal.code == "claimed":
                # The successor already exists (pre-fix interleaving put it
                # there during the pause); the intactness assertion is what
                # discriminates.
                return
            if refusal.code == "busy":
                time.sleep(0.05)
                continue
            raise
    raise AssertionError("reclaim never landed")


def _assert_successor_intact(tmp_path: Path, message_id: str = "bmsg_1") -> None:
    row = _successor_row(tmp_path, message_id)
    session, token, generation, state, expiry = row
    assert (session, token, generation, state) == ("session-new", "new-token", 2, "held")
    # The old caller's ~now+300 lease must never reach the successor; its own
    # ~now+60 stands (window generous for slow CI).
    assert time.time() < expiry < time.time() + 200


@pytest.mark.parametrize("operation", ["renew", "release"])
def test_q6r2_stale_lease_operation_cannot_touch_successor(tmp_path: Path, operation: str) -> None:
    store, owner = _store_with_owner(tmp_path)
    store.claim(owner, "bmsg_1", "session-old", "old-token", expiry=time.time() + 60)
    pausing = _PausingStore(tmp_path, marker="FROM claims")
    outcome: dict[str, Any] = {}

    def old_operation() -> None:
        try:
            if operation == "renew":
                pausing.renew_claim(owner, "bmsg_1", "old-token", time.time() + 300)
                outcome["result"] = "renewed"
            else:
                pausing.release_claim(owner, "bmsg_1", "old-token")
                outcome["result"] = "released"
        except StoreRefused as refusal:
            outcome["result"] = refusal.code

    thread = threading.Thread(target=old_operation)
    thread.start()
    assert pausing.paused.wait(timeout=5)
    b_completed_in_pause = _b_during_pause(tmp_path, owner, "bmsg_1")
    pausing.resume.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    # Fixed: the second connection could NOT complete while the old operation
    # held its write lock, so the reclaim had to be retried afterwards.
    assert b_completed_in_pause is False

    # The successor must survive whatever the old operation concluded.
    _retry_reclaim(tmp_path, owner, "bmsg_1")
    _assert_successor_intact(tmp_path)
    if operation == "release":
        # A release that had serialized BEFORE the invalidation may only have
        # touched the old row; the successor is still held regardless.
        assert outcome["result"] in ("released", "not_holder", "busy", "stale_generation")


def test_q5r2_invalidation_cannot_interleave_page_commit(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    store.claim(owner, "bmsg_1", "session-old", "old-token", expiry=time.time() + 60)
    pausing = _PausingStore(tmp_path, marker="FROM enrollments")
    outcome: dict[str, Any] = {}

    def old_merge() -> None:
        try:
            pausing.merge_page(owner, [_ref()], inbox_cursor("opaque-old-page"))
            outcome["result"] = "merged"
        except StoreRefused as refusal:
            outcome["result"] = refusal.code

    thread = threading.Thread(target=old_merge)
    thread.start()
    assert pausing.paused.wait(timeout=5)
    b_completed_in_pause = _b_during_pause(tmp_path, owner, "bmsg_1")
    pausing.resume.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    _retry_reclaim(tmp_path, owner, "bmsg_1")

    # Fixed: the page commit serialized BEFORE the invalidation (the second
    # connection's attempt had to be retried after the pause) — a generation-2
    # adoption of the generation-1 checkpoint can never be observed.
    assert b_completed_in_pause is False, (
        "invalidation completed inside the page's check/write window "
        f"(old merge outcome: {outcome['result']})"
    )
    # And the old snapshot can no longer commit anything at all.
    with pytest.raises(StoreRefused) as refusal:
        store.merge_page(owner, [_ref("bmsg_2")], None)
    assert refusal.value.code == "stale_generation"
    live = store.active_owner(credential_id=7)
    assert live is not None and live.enrollment_generation == 2


# ---------------------------------------------------------------------------
# Q4 — 409 classification and target-achieved validation
# ---------------------------------------------------------------------------


class FakeAckWire:
    base_url = "https://qa.invalid/v1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.ack_error: Exception | None = None
        self.ack_result: dict[str, Any] = {"delivery": {"state": "read", "state_at": "t"}}

    async def ack(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.ack_error is not None:
            raise self.ack_error
        return self.ack_result


async def _pending_intent(tmp_path: Path, owner: OwnerSnapshot) -> str:
    receipt = ScreeningReceipt(
        verdict="allow",
        model="qa/model",
        prompt_version="bridge-screen-v1",
        template_sha256="b" * 64,
        body_sha256="c" * 64,
        categories=(),
    )
    return Store(tmp_path).record_ack_intent(owner, "bmsg_1", "read", receipt, "stable-ref")


@pytest.mark.asyncio
async def test_q4r2_unknown_409_stays_pending_and_replays(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _pending_intent(tmp_path, owner)
    wire = FakeAckWire()
    wire.ack_error = BridgeRefusal("unknown_future_retryable", "additive", http_status=409)
    first = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=cast(pull.WireClient, wire)
    )
    assert first.state == ACK_PENDING
    wire.ack_error = None
    second = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=cast(pull.WireClient, wire)
    )
    assert second.state == ACK_CONFIRMED
    assert len(wire.calls) == 2


@pytest.mark.asyncio
async def test_q4r2_contradictory_success_state_not_confirmed(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _pending_intent(tmp_path, owner)
    wire = FakeAckWire()
    wire.ack_result = {"delivery": {"state": "unread", "state_at": "t"}}
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=cast(pull.WireClient, wire)
    )
    assert outcome.state == ACK_PENDING
    assert outcome.error_code == "result_mismatch"
    intent = store.ack_intent(owner, intent_id)
    assert intent is not None and intent["state"] == ACK_PENDING


@pytest.mark.asyncio
async def test_q4r2_target_achieved_confirms(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _pending_intent(tmp_path, owner)
    wire = FakeAckWire()
    wire.ack_result = {"delivery": {"state": "read", "state_at": "t"}}
    outcome = await pull.flush_ack(
        owner, intent_id, config_dir=tmp_path, access=_access(), client=cast(pull.WireClient, wire)
    )
    assert outcome.state == ACK_CONFIRMED and outcome.delivery_state == "read"


# ---------------------------------------------------------------------------
# R5 / Q7-recovery — pending intent lookup by namespace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r5_pending_intent_recovered_and_refuses_other_ref(tmp_path: Path) -> None:
    store, owner = _store_with_owner(tmp_path)
    intent_id = await _pending_intent(tmp_path, owner)
    wire = FakeAckWire()
    # A different ref is refused: the stored intent's identity is immutable.
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.recover_ack(
            owner,
            "bmsg_1",
            "read",
            config_dir=tmp_path,
            access=_access(),
            client_ref="different-ref",
            client=cast(pull.WireClient, wire),
        )
    assert refusal.value.code == "ref_mismatch"
    assert wire.calls == []
    # The namespace lookup finds the pending row and replays the STORED ref.
    outcome = await pull.recover_ack(
        owner,
        "bmsg_1",
        "read",
        config_dir=tmp_path,
        access=_access(),
        client=cast(pull.WireClient, wire),
    )
    assert outcome.state == ACK_CONFIRMED and outcome.intent_id == intent_id
    assert wire.calls[0]["client_ref"] == "stable-ref"
    # Another owner namespace can never see it.
    other = OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_other",
        endpoint_id="bep_1",
        canonical_base_url="https://qa.invalid/v1",
        enrollment_generation=1,
    ).validated()
    assert store.find_pending_intent(other, "bmsg_1", "read") is None
    assert store.list_pending_intents(owner) == []


# ---------------------------------------------------------------------------
# Nits — a wire without a destination is refused, never silently trusted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nit_base_url_less_wire_refused(tmp_path: Path) -> None:
    _store, owner = _store_with_owner(tmp_path)

    class _NoDestination:
        async def inbox(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
            raise AssertionError("must not dispatch")

    from typing import cast

    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(
            owner,
            config_dir=tmp_path,
            access=_access(),
            client=cast(pull.WireClient, _NoDestination()),
        )
    assert refusal.value.code == "no_destination_evidence"
