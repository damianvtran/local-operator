"""H2 pull/ack: exact-owner guard, intent-before-dispatch, fence-after-await."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge import pull
from local_operator.team_bridge.store import ACK_PENDING, OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot, ScreeningReceipt


def _setup(tmp_path: Path) -> tuple[Store, OwnerSnapshot, OAuthAccess]:
    store = Store(tmp_path)
    owner = OwnerSnapshot(
        credential_id=7,
        account_id="acc_a",
        tenant_id="tnt_1",
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    ).validated()
    store.commit_enrollment(owner, receipt=OwnerReceipt("bridge_consent_v1", "ref-1"))
    access = OAuthAccess(access_token="tok", credential_id=7, account_id="acc_a", org_id="tnt_1")
    return store, owner, access


class FakeWire:
    base_url = "https://hub.example"

    def __init__(self, *, on_ack: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.on_ack = on_ack
        self.ack_result: dict[str, Any] = {"delivery": {"state": "read", "state_at": "t"}}
        self.event_result: dict[str, Any] | None = None

    async def inbox(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("inbox", kwargs))
        return {
            "items": [
                {
                    "message": {
                        "id": "bmsg_1",
                        "thread_id": "bth_1",
                        "tenant_id": "tnt_1",
                        "type": "note",
                        "body_hash": "c" * 64,
                        "seq": 1,
                        "origin": "member",
                        "recipient": {"account_id": "acc_a", "endpoint_id": "bep_1"},
                        "body": {"text": "CANARY-BODY-TEXT"},
                    },
                    "delivery": {"state": "unread", "state_at": "t"},
                    "sender": {"account_id": "acc_b", "endpoint_id": "bep_2"},
                }
            ],
            "next_cursor": None,
            "counts": {"unread": 1},
        }

    async def message(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("message", kwargs))
        return (await self.inbox())["items"][0]

    async def ack(self, message_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("ack", {**kwargs, "message_id": message_id}))
        if self.on_ack is not None:
            self.on_ack()
        return self.ack_result

    async def events(self, **kwargs: Any) -> dict[str, Any] | None:
        self.calls.append(("events", kwargs))
        return self.event_result


@pytest.mark.asyncio
async def test_inbox_guards_owner_and_persists_metadata_without_body(tmp_path: Path) -> None:
    store, owner, access = _setup(tmp_path)
    wire = FakeWire()
    page = await pull.inbox(owner, config_dir=tmp_path, access=access, client=wire)
    assert page.items[0].message_id == "bmsg_1"
    rows = store.list_metadata(owner)
    assert len(rows) == 1 and "CANARY-BODY-TEXT" not in repr(rows)
    # Wrong account behind the right credential row refuses before any write.
    bad = OAuthAccess(access_token="tok", credential_id=7, account_id="acc_other")
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(owner, config_dir=tmp_path, access=bad, client=FakeWire())
    assert refusal.value.code == "owner_mismatch"
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.inbox(owner, config_dir=tmp_path, access=None, client=FakeWire())
    assert refusal.value.code == "no_access"


@pytest.mark.asyncio
async def test_ack_intent_is_on_disk_before_dispatch_and_confirms(tmp_path: Path) -> None:
    store, owner, access = _setup(tmp_path)
    seen: dict[str, Any] = {}

    def on_ack() -> None:
        # At dispatch time the intent row must already exist with the exact
        # tuple (read via the DB itself — the flow constructs its own Store).
        import sqlite3 as _sqlite

        from local_operator.team_bridge.store import state_db_path as _path

        conn = _sqlite.connect(_path(tmp_path))
        try:
            row = conn.execute(
                "SELECT state, body_sha256, client_ref FROM ack_intents "
                "WHERE message_id = 'bmsg_1'"
            ).fetchone()
        finally:
            conn.close()
        seen["intent_at_dispatch"] = row

    receipt = ScreeningReceipt("allow", "m", "bridge-screen-v1", "a" * 64, "c" * 64, ())
    wire = FakeWire(on_ack=on_ack)
    outcome = await pull.ack(
        owner, "bmsg_1", "read", receipt, config_dir=tmp_path, access=access, client=wire
    )
    assert outcome.state == "confirmed" and outcome.delivery_state == "read"
    row = seen["intent_at_dispatch"]
    assert row is not None and row[0] == ACK_PENDING and row[1] == "c" * 64
    call = wire.calls[-1]
    assert call[0] == "ack"
    assert call[1]["screening"]["body_sha256"] == "c" * 64
    assert call[1]["tenant_id"] == "tnt_1"


@pytest.mark.asyncio
async def test_ack_stale_generation_after_await_leaves_intent_pending(tmp_path: Path) -> None:
    store, owner, access = _setup(tmp_path)
    receipt = ScreeningReceipt("allow", "m", "bridge-screen-v1", "a" * 64, "c" * 64, ())
    intent_id = store.record_ack_intent(owner, "bmsg_1", "read", receipt, "r" * 36)

    def invalidate_now() -> None:
        store.invalidate(owner)

    wire = FakeWire(on_ack=invalidate_now)
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.flush_ack(owner, intent_id, config_dir=tmp_path, access=access, client=wire)
    assert refusal.value.code == "stale_generation"
    intent = store.ack_intent(owner, intent_id)
    assert intent is not None and intent["state"] == ACK_PENDING


@pytest.mark.asyncio
async def test_flush_without_intent_holds(tmp_path: Path) -> None:
    store, owner, access = _setup(tmp_path)
    with pytest.raises(pull.PullRefused) as refusal:
        await pull.flush_ack(
            owner, "missing", config_dir=tmp_path, access=access, client=FakeWire()
        )
    assert refusal.value.code == "no_intent"


@pytest.mark.asyncio
async def test_events_204_none_and_empty_resync_flag(tmp_path: Path) -> None:
    store, owner, access = _setup(tmp_path)
    wire = FakeWire()
    wire.event_result = None
    assert await pull.events(owner, config_dir=tmp_path, access=access, client=wire) is None
    wire.event_result = {"events": [], "cursor": 9}
    result = await pull.events(owner, config_dir=tmp_path, access=access, client=wire)
    assert result == {"cursor": 9, "refs": [], "resync": True}
    assert store.sync_state(owner)["pending_resync"] is True
    assert store.sync_state(owner)["feed_seq"] == 9
