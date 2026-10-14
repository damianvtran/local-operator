"""H5c/H5b client half — sends, decides, the ledger (contract §6.2, §10).

Every cell drives the REAL client against ``httpx.MockTransport`` and asserts
the bytes that would cross the wire: path, query, body. Local refusals must
produce ZERO requests — the whole point of validating before dispatch.
"""

from __future__ import annotations

import json

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import BridgeRefusal, TeamBridgeClient
from tests.unit.team_bridge.test_client import FakeStore, _err, _ok

pytestmark = pytest.mark.asyncio

TENANT = "ten_1"
REF = "9f8a7b6c-1234-4abc-9def-0123456789ab"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _make_client(handler) -> TeamBridgeClient:
    return TeamBridgeClient(
        None,
        base_url="https://hub.example/v1",
        store=FakeStore(),
        transport=httpx.MockTransport(handler),
    )


def _capture(requests: list[httpx.Request], *, status: int = 201, result=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ok(result if result is not None else {"message": {"id": "bmsg_1"}}, status=status)

    return handler


def _refusal_requests(requests: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"error": "should not be reached"})

    return handler


# --- send ------------------------------------------------------------------


async def test_a_send_carries_the_caller_ref_and_the_named_destination() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests))
    result = await client.send(
        tenant_id=TENANT,
        to={"account_id": "acc_2"},
        type="note",
        text="hello there",
        client_ref=REF,
    )
    assert result == {"message": {"id": "bmsg_1"}}
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path.endswith("/v1/messaging/messages")
    assert request.url.params["tenant_id"] == TENANT
    body = json.loads(request.content)
    assert body == {
        "to": {"account_id": "acc_2"},
        "type": "note",
        "body": {"text": "hello there"},
        "client_ref": REF,
    }


async def test_a_request_send_carries_the_request_block_only() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests))
    await client.send(
        tenant_id=TENANT,
        to={"endpoint_id": "bep_9"},
        type="request",
        text="please approve",
        client_ref=REF,
        expires_at="2026-10-20T00:00:00Z",
    )
    body = json.loads(requests[0].content)
    assert body["request"] == {"expires_at": "2026-10-20T00:00:00Z"}
    assert body["to"] == {"endpoint_id": "bep_9"}


async def test_a_retry_reuses_the_same_client_ref_byte_identically() -> None:
    """§6.2 idempotency: the ONLY safe retry carries the SAME ref — asserted
    on the wire for two attempts, not on a mock of the client."""
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests))
    for _ in range(2):
        await client.send(
            tenant_id=TENANT,
            to={"account_id": "acc_2"},
            type="question",
            text="which env?",
            client_ref=REF,
        )
    refs = [json.loads(request.content)["client_ref"] for request in requests]
    assert refs == [REF, REF]


async def test_send_refusals_never_reach_the_wire() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_refusal_requests(requests))

    with pytest.raises(ValueError):
        await client.send(
            tenant_id=TENANT, to={"account_id": "acc_2"}, type="note", text="x", client_ref=""
        )
    # `to` must be exactly one of the two forms — never both, never neither.
    with pytest.raises(ValueError):
        await client.send(
            tenant_id=TENANT,
            to={"account_id": "acc_2", "endpoint_id": "bep_9"},
            type="note",
            text="x",
            client_ref=REF,
        )
    with pytest.raises(ValueError):
        await client.send(tenant_id=TENANT, to={}, type="note", text="x", client_ref=REF)
    with pytest.raises(ValueError):
        await client.send(
            tenant_id=TENANT, to={"account_id": "acc_2"}, type="ping", text="x", client_ref=REF
        )
    # expires_at belongs to request-typed sends only.
    with pytest.raises(ValueError):
        await client.send(
            tenant_id=TENANT,
            to={"account_id": "acc_2"},
            type="note",
            text="x",
            client_ref=REF,
            expires_at="2026-10-20T00:00:00Z",
        )
    # The per-type byte cap is on NFC-normalized UTF-8, not len(str).
    with pytest.raises(ValueError):
        await client.send(
            tenant_id=TENANT,
            to={"account_id": "acc_2"},
            type="note",
            text="é" * (16 * 1024),
            client_ref=REF,
        )
    assert requests == []


async def test_the_request_cap_is_the_larger_type_cap() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests))
    await client.send(
        tenant_id=TENANT,
        to={"account_id": "acc_2"},
        type="request",
        text="a" * (16 * 1024 + 8),
        client_ref=REF,
    )
    assert len(requests) == 1


# --- decide ----------------------------------------------------------------


async def test_a_decide_carries_the_ask_ref_receipt() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests, result={"request": {"state": "approved"}}))
    result = await client.decide(
        "bmsg_1",
        tenant_id=TENANT,
        decision="approved",
        ask_ref={"ask_id": "a-3f9c", "question_id": "bmsg_1"},
    )
    assert result["request"]["state"] == "approved"
    request = requests[0]
    assert request.url.path.endswith("/v1/messaging/messages/bmsg_1/decide")
    assert json.loads(request.content) == {
        "decision": "approved",
        "ask_ref": {"ask_id": "a-3f9c", "question_id": "bmsg_1"},
        "decided_via": "bearer-only",
    }


async def test_a_decide_without_or_with_a_mismatched_receipt_never_ships() -> None:
    """Missing ask_ref, the wrong question binding, the wrong decision word and
    the closed-via set are all local refusals with ZERO requests — a caller
    bug must not become traffic the server has to refuse."""
    requests: list[httpx.Request] = []
    client = _make_client(_refusal_requests(requests))

    with pytest.raises(ValueError):
        # type: ignore[arg-type] — the point of the case: the server word
        await client.decide("bmsg_1", tenant_id=TENANT, decision="approved", ask_ref=None)
    with pytest.raises(ValueError):
        await client.decide(
            "bmsg_1",
            tenant_id=TENANT,
            decision="approved",
            ask_ref={"ask_id": "a-3f9c", "question_id": "bmsg_OTHER"},
        )
    with pytest.raises(ValueError):
        await client.decide(
            "bmsg_1",
            tenant_id=TENANT,
            decision="maybe",
            ask_ref={"ask_id": "a-3f9c", "question_id": "bmsg_1"},
        )
    with pytest.raises(ValueError):
        await client.decide(
            "bmsg_1",
            tenant_id=TENANT,
            decision="denied",
            ask_ref={"ask_id": "a-3f9c", "question_id": "bmsg_1"},
            decided_via="classifier",
        )
    assert requests == []


async def test_a_decide_refusal_maps_through_the_shared_table() -> None:
    client = _make_client(lambda request: _err(409, "request_revoked"))
    with pytest.raises(BridgeRefusal):
        await client.decide(
            "bmsg_1",
            tenant_id=TENANT,
            decision="approved",
            ask_ref={"ask_id": "a-3f9c", "question_id": "bmsg_1"},
        )


# --- batch -----------------------------------------------------------------


async def test_a_batch_decide_is_bounded_and_refused_whole_pre_flight() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests, result={"results": []}))

    def item(index: int) -> dict:
        message_id = f"bmsg_{index}"
        return {
            "message_id": message_id,
            "decision": "approved",
            "ask_ref": {"ask_id": f"a-{index:04d}", "question_id": message_id},
        }

    with pytest.raises(ValueError):
        await client.decide_batch([item(i) for i in range(51)], tenant_id=TENANT)
    assert requests == []

    # One bad item refuses the whole set (the server's whole-refusal rule
    # mirrored locally, nothing commits).
    poisoned = [item(0), {**item(1), "ask_ref": {"ask_id": "bad id!", "question_id": "bmsg_1"}}]
    with pytest.raises(ValueError):
        await client.decide_batch(poisoned, tenant_id=TENANT)
    assert requests == []

    await client.decide_batch([item(0), item(1)], tenant_id=TENANT)
    body = json.loads(requests[0].content)
    assert requests[0].url.path.endswith("/v1/messaging/requests/decide")
    assert [entry["message_id"] for entry in body["decisions"]] == ["bmsg_0", "bmsg_1"]


async def test_the_ledger_read_names_its_role() -> None:
    requests: list[httpx.Request] = []
    client = _make_client(_capture(requests, result={"decisions": []}))
    with pytest.raises(ValueError):
        await client.decisions(tenant_id=TENANT, role="all")
    assert requests == []
    await client.decisions(tenant_id=TENANT, role="requested", thread_id="bth_1")
    request = requests[0]
    assert request.url.path.endswith("/v1/messaging/decisions")
    assert request.url.params["role"] == "requested"
    assert request.url.params["thread_id"] == "bth_1"
