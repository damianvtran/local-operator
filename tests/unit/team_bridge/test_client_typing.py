"""TeamBridgeClient typing-indicator calls (typing-indicators design §5).

Driven end to end against ``httpx.MockTransport`` with the shared fake
credential store, so the owner guard, the wire shape (PUT body vs DELETE
query) and the hub-version re-coding are all the real client's behaviour.
The hub side is simulated at its contract boundary (agent-server §18).
"""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import BridgeRefusal
from tests.unit.team_bridge.test_client import (
    FakeRow,
    FakeStore,
    _err,
    _make_client,
    _ok,
)

pytestmark = pytest.mark.asyncio

CID = "bcv_" + "a" * 32
ROOT = "bcmsg_" + "b" * 16
TYPING_PATH = f"/v1/messaging/conversations/{CID}/typing"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


async def test_get_passes_the_hub_view_through_verbatim():
    view = {
        "enabled": True,
        "poll_after_ms": 3000,
        "message_seq": 12,
        "truncated": False,
        "typers": [
            {"account_id": "acc_x", "kind": "human", "thread_root_id": None, "expires_in_ms": 6120}
        ],
    }
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(view)

    got = await _make_client(handler).conversation_typing(CID, tenant_id="org_1")
    assert got == view
    assert seen[0].method == "GET"
    assert seen[0].url.path == TYPING_PATH
    assert dict(seen[0].url.params) == {"tenant_id": "org_1"}


async def test_put_sends_body_and_returns_cadence():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"ttl_ms": 8000, "resend_after_ms": 3000})

    got = await _make_client(handler).conversation_typing_set(
        CID, tenant_id="org_1", thread_root_id=ROOT, expected_account_id="acc_1"
    )
    assert got == {"ttl_ms": 8000, "resend_after_ms": 3000}
    assert seen[0].method == "PUT"
    assert seen[0].url.path == TYPING_PATH
    # The owner receipt is compared locally and NEVER forwarded upstream.
    assert json.loads(seen[0].content) == {"tenant_id": "org_1", "thread_root_id": ROOT}
    assert not seen[0].url.params


async def test_put_on_a_flag_off_hub_answering_204_is_an_empty_ack_not_a_502():
    got = await _make_client(lambda r: httpx.Response(204)).conversation_typing_set(
        CID, tenant_id="org_1"
    )
    assert got == {}


async def test_delete_carries_fields_in_the_query_and_204_is_empty():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    got = await _make_client(handler).conversation_typing_clear(
        CID, tenant_id="org_1", thread_root_id=ROOT, expected_account_id="acc_1"
    )
    assert got == {}
    assert seen[0].method == "DELETE"
    assert dict(seen[0].url.params) == {"tenant_id": "org_1", "thread_root_id": ROOT}
    assert not seen[0].content


@pytest.mark.parametrize("status", [404, 405])
@pytest.mark.parametrize("call", ["get", "put", "delete"])
async def test_an_uncoded_404_or_405_is_outdated_hub(status: int, call: str):
    """A hub predating §18 answers echo's own uncoded not-found/method refusal."""

    client = _make_client(lambda r: httpx.Response(status, json={"message": "Not Found"}))
    with pytest.raises(BridgeRefusal) as info:
        if call == "get":
            await client.conversation_typing(CID, tenant_id="org_1")
        elif call == "put":
            await client.conversation_typing_set(CID, tenant_id="org_1")
        else:
            await client.conversation_typing_clear(CID, tenant_id="org_1")
    assert info.value.code == tb_client.OUTDATED_HUB
    assert info.value.http_status == 409
    assert info.value.details == {"hub_status": status, "hub_code": "bridge_error"}


async def test_a_coded_conversation_not_found_passes_through():
    client = _make_client(lambda r: _err(404, "conversation_not_found"))
    with pytest.raises(BridgeRefusal) as info:
        await client.conversation_typing(CID, tenant_id="org_1")
    assert info.value.code == "conversation_not_found"
    assert info.value.http_status == 404


async def test_a_503_passes_through_unchanged():
    client = _make_client(lambda r: _err(503, "bridge_disabled"))
    with pytest.raises(BridgeRefusal) as info:
        await client.conversation_typing_set(CID, tenant_id="org_1")
    assert info.value.code == "bridge_disabled"
    assert info.value.http_status == 503


async def test_a_429_keeps_its_retry_after():
    client = _make_client(
        lambda r: httpx.Response(
            429, json={"code": "rate_limited", "error": "slow down"}, headers={"Retry-After": "7"}
        )
    )
    with pytest.raises(BridgeRefusal) as info:
        await client.conversation_typing(CID, tenant_id="org_1")
    assert info.value.code == "rate_limited"
    assert info.value.retry_after_seconds == 7


async def test_an_owner_mismatch_refuses_before_any_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok({"ttl_ms": 8000, "resend_after_ms": 3000})

    store = FakeStore([FakeRow(1, data={"access": "token-1", "account_id": "acc_b"})])
    client = _make_client(handler, store=store)
    with pytest.raises(BridgeRefusal) as info:
        await client.conversation_typing_set(CID, tenant_id="org_1", expected_account_id="acc_a")
    assert info.value.code == "account_mismatch"
    assert seen == []
