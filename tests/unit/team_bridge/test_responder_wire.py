"""TeamBridgeClient responder + claim-lease calls against agent-server #451's contract.

Driven through ``httpx.MockTransport`` with the shared fake credential store, so the
wire shape (paths, bodies, the 204 Stop/release) and the coded-refusal mapping are
the real client's. Then the lease module is driven over the SAME real client, so
``already_claimed`` / ``lease_fenced`` / ``autonomy_disabled`` are classified from
real ``BridgeRefusal``s rather than hand-built exceptions.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.responders import lease
from tests.unit.team_bridge.test_client import _err, _make_client, _ok

pytestmark = pytest.mark.asyncio

CID = "bcv_" + "a" * 32
ROOT = "bcmsg_" + "b" * 16
EP = "bep_" + "c" * 16
NOW = dt.datetime(2026, 11, 21, 12, 0, tzinfo=dt.UTC)
CLAIM_PATH = f"/v1/messaging/conversations/{CID}/messages/{ROOT}/claim"


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


def _claim_row(
    epoch: int = 1, *, holder: str = EP, expires: str = "2026-11-21T12:03:00.123456789Z"
) -> dict[str, Any]:
    return {
        "claim_id": "icl_x",
        "conversation_id": CID,
        "root_message_id": ROOT,
        "state": "held",
        "holder_kind": "agent",
        "holder_endpoint_id": holder,
        "lease_epoch": epoch,
        "lease_expires_at": expires,
    }


async def test_claim_posts_tenant_and_endpoint_and_returns_a_lease():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(_claim_row(), status=201)

    got = await lease.claim(
        _make_client(handler),
        tenant_id="org_1",
        conversation_id=CID,
        root_message_id=ROOT,
        endpoint_id=EP,
    )
    assert got.outcome == "claimed"
    assert got.lease is not None and got.lease.epoch == 1
    # Go's RFC3339Nano (9 fractional digits) parses; the lease keeps tz-aware UTC.
    assert got.lease.expires_at == dt.datetime(2026, 11, 21, 12, 3, 0, 123456, tzinfo=dt.UTC)
    assert seen[0].method == "POST" and seen[0].url.path == CLAIM_PATH
    assert json.loads(seen[0].content) == {"tenant_id": "org_1", "actor_endpoint_id": EP}


@pytest.mark.parametrize(
    ("status", "code", "outcome"),
    [
        (409, "already_claimed", "stand_down"),
        (409, "alert_resolved", "stand_down"),
        (403, "autonomy_disabled", "refused"),
        (403, "responder_not_enabled", "refused"),
        (403, "not_endpoint_owner", "refused"),
    ],
)
async def test_claim_refusals_are_classified_never_raised(status, code, outcome):
    got = await lease.claim(
        _make_client(lambda r: _err(status, code)),
        tenant_id="org_1",
        conversation_id=CID,
        root_message_id=ROOT,
        endpoint_id=EP,
    )
    assert (got.outcome, got.code, got.lease) == (outcome, code, None)


async def test_claim_on_a_dark_hub_uncoded_404_is_refused():
    # TEAM_INTEGRATIONS_ENABLED=false: the route is not registered (echo's bare 404).
    got = await lease.claim(
        _make_client(lambda r: httpx.Response(404, json={"message": "Not Found"})),
        tenant_id="org_1",
        conversation_id=CID,
        root_message_id=ROOT,
        endpoint_id=EP,
    )
    assert got.outcome == "refused" and got.lease is None


async def test_claim_granted_to_another_holder_is_not_a_lease():
    got = await lease.claim(
        _make_client(lambda r: _ok(_claim_row(holder="bep_" + "d" * 16), status=201)),
        tenant_id="org_1",
        conversation_id=CID,
        root_message_id=ROOT,
        endpoint_id=EP,
    )
    assert got.outcome == "refused" and got.lease is None


def _held(epoch: int = 3) -> lease.Lease:
    return lease.Lease("org_1", CID, ROOT, EP, epoch, NOW + dt.timedelta(minutes=3))


async def test_heartbeat_carries_epoch_status_and_wait():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(_claim_row(3, expires="2026-11-21T12:04:00Z"))

    renewed = await lease.heartbeat(
        _make_client(handler),
        _held(3),
        now=NOW,
        status_line="Status: checking pods",
        waiting_approval_for={"ask_id": "ask_1", "session_id": "s1"},
    )
    assert renewed.epoch == 3 and renewed.expires_at == dt.datetime(
        2026, 11, 21, 12, 4, tzinfo=dt.UTC
    )
    assert seen[0].url.path == CLAIM_PATH + "/heartbeat"
    assert json.loads(seen[0].content) == {
        "tenant_id": "org_1",
        "actor_endpoint_id": EP,
        "lease_epoch": 3,
        "status_line": "Status: checking pods",
        "waiting_approval_for": {"ask_id": "ask_1", "session_id": "s1"},
    }


@pytest.mark.parametrize(
    "code", ["lease_fenced", "autonomy_disabled", "responder_not_enabled", "alert_resolved"]
)
async def test_heartbeat_refusal_loses_the_lease(code):
    status = 409 if code in ("lease_fenced", "alert_resolved") else 403
    with pytest.raises(lease.LeaseLost) as lost:
        await lease.heartbeat(_make_client(lambda r: _err(status, code)), _held(), now=NOW)
    assert lost.value.code == code


async def test_heartbeat_epoch_change_is_a_fence():
    with pytest.raises(lease.LeaseLost) as lost:
        await lease.heartbeat(_make_client(lambda r: _ok(_claim_row(4))), _held(3), now=NOW)
    assert lost.value.code == "lease_fenced"


async def test_heartbeat_5xx_keeps_the_lease_until_local_expiry():
    client = _make_client(lambda r: _err(503, "bridge_unavailable"))
    held = dataclasses.replace(_held(), deadline=500.0)
    kept = await lease.heartbeat(client, held, now=NOW, monotonic=lambda: 499.0)
    assert kept == held
    # Review m5: the MONOTONIC deadline decides. A wall clock that says the lease
    # is far from over (a laptop running behind) cannot keep it alive.
    with pytest.raises(lease.LeaseLost) as lost:
        await lease.heartbeat(
            client, held, now=NOW - dt.timedelta(hours=1), monotonic=lambda: 500.0
        )
    assert lost.value.code == "local_expiry"


@pytest.mark.parametrize(
    ("expires", "now_offset_s", "remaining"),
    [
        # The hub's 3-minute lease from when the request was SENT…
        ("2026-11-21T12:03:00Z", 0, 180.0),
        # …never lengthened by a wall clock running behind (says 13 min left)…
        ("2026-11-21T12:03:00Z", -600, 180.0),
        # …and shortened when the wall clock says less is left.
        ("2026-11-21T12:03:00Z", 120, 60.0),
    ],
)
async def test_renewed_deadline_is_the_sooner_of_lease_and_wall_reading(
    expires, now_offset_s, remaining
):
    renewed = await lease.heartbeat(
        _make_client(lambda r: _ok(_claim_row(3, expires=expires))),
        _held(3),
        now=NOW + dt.timedelta(seconds=now_offset_s),
        monotonic=lambda: 1000.0,
    )
    assert renewed.deadline == 1000.0 + remaining


async def test_unreadable_2xx_claim_is_released_when_it_names_an_epoch():
    """S-7: a 2xx we cannot read as our lease is given back, not left to lapse."""

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(204)
        return _ok({"lease_epoch": 7, "holder_endpoint_id": EP, "lease_expires_at": "garbage"})

    got = await lease.claim(
        _make_client(handler),
        tenant_id="org_1",
        conversation_id=CID,
        root_message_id=ROOT,
        endpoint_id=EP,
    )
    assert (got.outcome, got.code, got.lease) == ("refused", "invalid_claim_response", None)
    assert seen == ["POST", "DELETE"]


async def test_release_sends_epoch_in_a_delete_body_and_accepts_204():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204)

    assert await lease.release(_make_client(handler), _held(5)) == "released"
    assert seen[0].method == "DELETE" and seen[0].url.path == CLAIM_PATH
    assert json.loads(seen[0].content) == {
        "tenant_id": "org_1",
        "actor_endpoint_id": EP,
        "lease_epoch": 5,
    }


async def test_release_of_a_fenced_claim_is_already_gone():
    assert (
        await lease.release(_make_client(lambda r: _err(409, "lease_fenced")), _held())
        == "already_gone"
    )


async def test_path_ids_cannot_inject_segments():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok(_claim_row(), status=201)

    client = _make_client(handler)
    await client.alert_claim(CID, "x/y?z", tenant_id="org_1", actor_endpoint_id=EP)
    assert seen[0].url.raw_path.decode().endswith("/messages/x%2Fy%3Fz/claim")
    # A dot segment never leaves the process (the client's shared dot-segment refusal).
    with pytest.raises(tb_client.BridgeRefusal):
        await client.alert_claim(CID, "..", tenant_id="org_1", actor_endpoint_id=EP)
    assert len(seen) == 1


async def test_publish_and_stop_wire_shapes():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        return _ok({"conversation_id": CID, "endpoint_id": EP, "responder_enabled": True})

    client = _make_client(handler)
    await client.responder_publish(
        CID,
        EP,
        tenant_id="org_1",
        enabled=True,
        autonomy_tier="T0",
        consent_version="responder_consent_v1",
        consent_sha256="e" * 64,
        policy_sha256="f" * 64,
    )
    await client.responder_stop(CID, EP, tenant_id="org_1")
    assert [r.method for r in seen] == ["PUT", "DELETE"]
    assert seen[0].url.path == f"/v1/messaging/conversations/{CID}/responders/{EP}"
    assert json.loads(seen[0].content) == {
        "tenant_id": "org_1",
        "responder_enabled": True,
        "autonomy_tier": "T0",
        "consent_version": "responder_consent_v1",
        "consent_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
    }
    assert json.loads(seen[1].content) == {"tenant_id": "org_1"}


async def test_status_line_matches_the_hubs_validator():
    line = lease.sanitize_status_line(
        "Status: *pods* <crashloop> see https://grafana.example/x and www.example.com\n"
        "`next` #1 | [ok]\x07"
    )
    assert line == "Status: pods crashloop see and next 1 ok"
    assert lease.sanitize_status_line("x" * 300) is not None
    assert len(lease.sanitize_status_line("x" * 300) or "") == 140
    assert lease.sanitize_status_line("https://only.example") is None
