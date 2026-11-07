"""Unregistered orchestrator source tests: synthetic provider and MockTransport only."""

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as client_module
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from local_operator.team_bridge.file_action_orchestrator import (
    HumanFileActionOrchestrator,
    TrustedMainView,
)
from local_operator.team_bridge.file_action_registry import (
    ActionUnavailable,
    InertFileActionRegistry,
)

pytestmark = pytest.mark.asyncio


@dataclass
class Row:
    id: int = 1
    provider: str = "radient"
    credential_type: str = "oauth"
    disabled_cause: str | None = None
    updated_at: int = 42
    data: dict[str, Any] = field(
        default_factory=lambda: {
            "access": "synthetic-token",
            "account_id": "acc_1",
            "expires": 4102444800000,
        }
    )


class Store:
    db_path = Path("synthetic-auth.db")

    def __init__(self):
        self.row = Row()
        self.incarnation = "a" * 32
        self.refreshes = 0

    def list_credentials(self, _provider):
        return [self.row]

    def native_lifecycle_snapshot(self, _path, _provider):
        return [(1, 42, self.incarnation, None, "oauth", "acc_1", None, None, None)]

    def get_credential(self, credential_id):
        return self.row if credential_id == 1 else None

    async def ensure_oauth_fresh_or_raise(self, *args, **kwargs):
        self.refreshes += 1
        raise AssertionError("unexpected refresh")


def human_page(*, downloadable=True, conversation="bcv_aaaaaaaa"):
    return {
        "msg": "ok",
        "result": {
            "messages": [
                {
                    "id": "bcmsg_aaaaaaaa",
                    "seq": 1001,
                    "conversation_id": conversation,
                    "files": [
                        {
                            "id": "batt_aaaaaaaa",
                            "mime": "application/octet-stream",
                            "download_only": downloadable,
                            "state": "bound",
                            "class": "other",
                            "size": 0,
                            "expires_at": "2099-01-01T00:00:00Z",
                        }
                    ],
                }
            ]
        },
    }


def streamed_json(status, value):
    # Preserve the unconsumed raw stream that a real HTTP transport returns.
    return httpx.Response(
        status,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(json.dumps(value).encode()),
    )


def setup(monkeypatch, *, history_status=200, download_status=200, downloadable=True):
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda _: True)
    store, calls = Store(), []
    state: dict[str, TrustedMainView | None] = {
        "view": TrustedMainView("window_1", 4, "tnt_alpha", "bcv_aaaaaaaa", "bcmsg_aaaaaaaa", 1001)
    }

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/messages"):
            return streamed_json(history_status, human_page(downloadable=downloadable))
        return httpx.Response(
            download_status,
            stream=httpx.ByteStream(b""),
            headers={"content-type": "application/octet-stream"},
        )

    client = TeamBridgeClient(
        base_url="https://synthetic.invalid/v1", store=store, transport=httpx.MockTransport(handler)
    )
    registry = InertFileActionRegistry()
    broker = HumanFileActionOrchestrator(
        client, registry, lambda window: state["view"] if window == "window_1" else None
    )
    return broker, registry, store, calls, state


async def test_commission_begin_bytes_verify_revoke(monkeypatch):
    broker, registry, store, calls, state = setup(monkeypatch)
    context = await broker.commission("window_1")
    op = broker.begin("window_1", context, "batt_aaaaaaaa")
    assert await broker.bytes("window_1", op) == b""
    broker.verify("window_1", op)
    with pytest.raises(ActionUnavailable):
        await broker.bytes("window_1", op)
    assert len(calls) == 2 and store.refreshes == 0
    assert dict(calls[0].url.params) == {
        "tenant_id": "tnt_alpha",
        "limit": "1",
        "before_seq": "1002",
    }
    assert dict(calls[1].url.params) == {"tenant_id": "tnt_alpha"}
    assert all("endpoint_id" not in r.url.params and "purpose" not in r.url.params for r in calls)
    broker.revoke("window_1", context)
    with pytest.raises(ActionUnavailable):
        broker.verify("window_1", op)


async def test_only_returned_file_and_window_can_begin_or_cancel(monkeypatch):
    broker, registry, store, calls, state = setup(monkeypatch)
    context = await broker.commission("window_1")
    with pytest.raises(ActionUnavailable):
        broker.begin("window_1", context, "batt_bbbbbbbb")
    with pytest.raises(ActionUnavailable):
        broker.begin("window_2", context, "batt_aaaaaaaa")
    op = broker.begin("window_1", context, "batt_aaaaaaaa")
    with pytest.raises(ActionUnavailable):
        broker.cancel("window_2", op)
    broker.cancel("window_1", op)
    with pytest.raises(ActionUnavailable):
        await broker.bytes("window_1", op)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change", ["tenant", "conversation", "message", "generation", "incarnation", "window_closed"]
)
async def test_stale_scope_never_fetches_bytes(monkeypatch, change):
    broker, registry, store, calls, state = setup(monkeypatch)
    context = await broker.commission("window_1")
    op = broker.begin("window_1", context, "batt_aaaaaaaa")
    view = state["view"]
    assert view is not None
    if change == "tenant":
        state["view"] = replace(view, tenant_id="tnt_beta")
    if change == "conversation":
        state["view"] = replace(view, conversation_id="bcv_bbbbbbbb")
    if change == "message":
        state["view"] = replace(view, message_id="bcmsg_bbbbbbbb")
    if change == "generation":
        state["view"] = replace(view, generation=5)
    if change == "window_closed":
        state["view"] = None
    if change == "incarnation":
        store.incarnation = "b" * 32
    with pytest.raises((ActionUnavailable, BridgeRefusal)):
        await broker.bytes("window_1", op)
    assert len(calls) == 1 and store.refreshes == 0


async def test_invalidation_revokes_pending_and_fresh_generation_can_commission(monkeypatch):
    broker, registry, store, calls, state = setup(monkeypatch)
    context = await broker.commission("window_1")
    op = broker.begin("window_1", context, "batt_aaaaaaaa")
    broker.invalidate("window_1", 4)
    with pytest.raises(ActionUnavailable):
        await broker.bytes("window_1", op)
    view = state["view"]
    assert view is not None
    state["view"] = replace(view, generation=5)
    assert await broker.commission("window_1") != context


async def test_invalidation_during_history_blocks_delayed_commission(monkeypatch):
    monkeypatch.setattr(client_module, "org_oauth_destination_allowed", lambda _: True)
    store, registry = Store(), InertFileActionRegistry()
    entered, release = asyncio.Event(), asyncio.Event()
    state = {
        "view": TrustedMainView("window_1", 4, "tnt_alpha", "bcv_aaaaaaaa", "bcmsg_aaaaaaaa", 1001)
    }

    async def handler(_request):
        entered.set()
        await release.wait()
        return streamed_json(200, human_page())

    client = TeamBridgeClient(
        base_url="https://synthetic.invalid/v1", store=store, transport=httpx.MockTransport(handler)
    )
    broker = HumanFileActionOrchestrator(client, registry, lambda _: state["view"])
    task = asyncio.create_task(broker.commission("window_1"))
    await entered.wait()
    broker.invalidate("window_1", 4)
    state["view"] = replace(state["view"], generation=5)
    release.set()
    with pytest.raises((ActionUnavailable, BridgeRefusal)):
        await task
    assert registry._contexts == {} and store.refreshes == 0


@pytest.mark.parametrize("mode", ["history_401", "file_401", "agent_descriptor", "expired_bearer"])
async def test_refusal_never_refreshes_or_retries(monkeypatch, mode):
    broker, registry, store, calls, state = setup(
        monkeypatch,
        history_status=401 if mode == "history_401" else 200,
        download_status=401 if mode == "file_401" else 200,
        downloadable=mode != "agent_descriptor",
    )
    if mode == "expired_bearer":
        store.row.data["expires"] = 1
    with pytest.raises(BridgeRefusal):
        context = await broker.commission("window_1")
        op = broker.begin("window_1", context, "batt_aaaaaaaa")
        await broker.bytes("window_1", op)
    assert len(calls) == (0 if mode == "expired_bearer" else 2 if mode == "file_401" else 1)
    assert store.refreshes == 0
