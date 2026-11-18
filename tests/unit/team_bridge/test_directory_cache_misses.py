"""A directory MISS pays for ONE re-read — and only of a CACHED answer.

``TeamBridgeClient.directory_cached`` holds the directory for
``DIRECTORY_CACHE_TTL_S`` per account and tenant, so a burst of named sends costs
one read instead of one per message (incident 2026-11-17: a read per named send
from an address the edge rate-limits). A resolution that MISSES on the cached
answer re-reads it once, so a teammate who joined, enrolled or turned their
bridge on since is found on the first try.

A miss on an answer this call read off the WIRE is already the server's latest,
and re-reading it returns the same members: that redundant GET is what the cache
exists to remove. The origin travels ON the answer
(``client.DirectoryAnswer.from_cache``), not in client state, so two resolutions
sharing one client each judge the answer they actually got.

These drive the REAL client through an ``httpx`` transport and count the
requests, because the property is about traffic, not about a return value.
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
async def test_a_cold_name_miss_reads_the_directory_once(tmp_path) -> None:
    """The first read IS the wire, so the miss re-reads nothing."""

    wire = Wire()
    result = await execute_bridge(
        _runtime(wire, tmp_path), {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "hi"}
    )
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert len(wire.directory_reads) == 1
    assert len(wire.allowlist_reads) == 1


@pytest.mark.asyncio
async def test_a_warm_name_miss_pays_exactly_one_re_read(tmp_path) -> None:
    """A cached answer may be stale, so the miss buys exactly one fresh read."""

    wire = Wire()
    runtime = _runtime(wire, tmp_path)
    await runtime.client.directory_cached(TENANT)  # prime: one read
    result = await execute_bridge(
        runtime, {"op": "send", "to": {"name": "Ada Lovelace"}, "text": "hi"}
    )
    assert result["ok"] is False and result["code"] == "invalid_request"
    assert len(wire.directory_reads) == 2, "the cached miss must re-read exactly once"
    assert len(wire.allowlist_reads) == 1


@pytest.mark.asyncio
async def test_a_cold_dm_not_reachable_reads_the_directory_once(tmp_path) -> None:
    wire = Wire()
    result = await execute_bridge(
        _runtime(wire, tmp_path), {"op": "dm", "account_id": "acc_absent", "body": "hi"}
    )
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == "not_a_member"
    assert len(wire.directory_reads) == 1


@pytest.mark.asyncio
async def test_a_warm_dm_not_reachable_pays_exactly_one_re_read(tmp_path) -> None:
    wire = Wire()
    runtime = _runtime(wire, tmp_path)
    await runtime.client.directory_cached(TENANT)  # prime: one read
    result = await execute_bridge(runtime, {"op": "dm", "account_id": "acc_absent", "body": "hi"})
    assert result["ok"] is False and result["code"] == "not_reachable"
    assert result["reason"] == "not_a_member"
    assert len(wire.directory_reads) == 2, "the cached miss must re-read exactly once"


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
