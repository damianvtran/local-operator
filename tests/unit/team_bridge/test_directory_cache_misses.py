"""A cached DM miss is decided by the answer the RE-READ returns.

``TeamBridgeClient.directory_cached`` holds the directory for
``DIRECTORY_CACHE_TTL_S`` per account and tenant, so a burst of named sends and
DMs costs one read instead of one per message (incident 2026-11-17: a read per
named send from an address the edge rate-limits). A resolution that MISSES on
the cached copy re-reads it ONCE, so a teammate who joined, enrolled or turned
their bridge on since is found on the first try. A copy this call read off the
WIRE is already the server's latest, so its miss is believed without a second
read — :meth:`TeamBridgeClient.directory_cached_read` reports which one a
caller got.

What the re-read is FOR is the verdict, not the traffic: a stale copy answers
"no active member has that account_id" for someone who joined since, which is a
WRONG answer rather than a slow one. The subject here is that the refusal the
caller sees comes from the re-read — a teammate who appears between the prime
and the DM and has their bridge off must be told ``no_agent_disabled``, not
``not_a_member``.

These drive the REAL client through an ``httpx`` transport, so the request count
and the refusal are the ones a caller sees. The cold/warm request COUNTS are
pinned elsewhere and deliberately not repeated here: ``test_h5d_tool`` for a
named send, ``test_direct_directory_tool`` for a DM, and
``test_rate_limit_refusals`` for ``directory_cached_read``'s ``fetched`` flag.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge import host as host_mod
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import OwnerSnapshot
from local_operator.team_bridge.tool import BridgeRuntime, execute_bridge
from tests.unit.team_bridge.test_client import FakeStore, _ok

TENANT = "ten_1"
BO = {"account_id": "acc_other", "display_label": "Bo Chen", "bridge_enabled": True}
ALLOWLIST = {
    "allowlist": {
        "entries": [{"direction": "send", "effect": "allow", "target": {"account_id": "acc_other"}}]
    }
}


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture(autouse=True)
def _gates_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """The entry gates are covered elsewhere; these subjects are the reads."""

    monkeypatch.setattr(host_mod, "owner_fence_refusal", lambda runtime: None)
    monkeypatch.setattr(host_mod, "outbound_tool_fence", lambda runtime: None)
    monkeypatch.setattr(host_mod, "admission_refusal", lambda runtime, *, is_read=False: None)
    # A denied call may schedule admission; no background status request here.
    monkeypatch.setattr(host_mod, "admit_soon", lambda runtime: None)


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=1,
        account_id="acc_1",
        tenant_id=TENANT,
        endpoint_id="bep_1",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    )


class Wire:
    """The hub's directory and allowlist, counting the reads a resolution makes.

    ``members`` is mutable so a test can change what the NEXT directory read
    answers — a teammate joining between the prime and the resolution.
    """

    def __init__(self, members: list[dict[str, Any]] | None = None) -> None:
        self.members = [dict(BO)] if members is None else members
        self.directory_reads: list[httpx.Request] = []
        self.allowlist_reads: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/directory"):
            self.directory_reads.append(request)
            return _ok({"members": self.members})
        self.allowlist_reads.append(request)
        return _ok(ALLOWLIST)

    def client(self) -> TeamBridgeClient:
        return TeamBridgeClient(
            None,
            base_url="https://hub.example/v1",
            store=FakeStore(),
            transport=httpx.MockTransport(self.handler),
        )


def _runtime(wire: Wire, tmp_path: Any) -> BridgeRuntime:
    return BridgeRuntime(
        client=wire.client(),
        config_dir=tmp_path,
        tenant_id=TENANT,
        owner=_owner(),
        session_id="synthetic-session",
    )


@pytest.mark.asyncio
async def test_a_warm_dm_miss_is_decided_by_the_FRESH_answer(tmp_path) -> None:
    """The re-read is not ceremony: its members are the ones the verdict uses."""

    wire = Wire()
    runtime = _runtime(wire, tmp_path)
    await runtime.client.directory_cached(TENANT)  # prime: acc_2 is not a member yet
    # Between the prime and the dm, acc_2 appears — with the bridge off.
    wire.members = [
        {
            "account_id": "acc_2",
            "display_label": "Dana",
            "bridge_enabled": False,
            "bridge_state": "disabled",
        }
    ]
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_2", "body": "hi"})
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == "no_agent_disabled", "the verdict must come from the re-read"
    assert len(wire.directory_reads) == 2
