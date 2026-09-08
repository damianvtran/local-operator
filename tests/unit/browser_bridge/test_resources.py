from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.browser_bridge import resources
from local_operator.browser_bridge.resources import (
    BrowserResource,
    cleanup_exact,
    read_inventory,
)
from local_operator.session_lease import SessionLeaseHeldError, acquire_session_lease


class BridgeFixture:
    """Disposable protocol peer; never discovers the operator's bridge."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_close = False
        self.retained = False

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "owner_recover":
            return {"ownership_version": 1, "state": "owned", "tab": "bridge:100:private"}
        if method == "owner_retain":
            self.retained = True
            return {"state": "retained"}
        if method == "owner_finish":
            if self.fail_close:
                raise ConnectionError("disposable fixture unavailable")
            return {"state": "retained" if self.retained else "closed"}
        raise AssertionError(method)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> BridgeFixture:
    fixture = BridgeFixture()
    monkeypatch.setattr(resources, "BridgeClient", lambda: fixture)
    return fixture


def test_private_sidecar_and_redacted_inventory(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic"
    resource = BrowserResource(directory, directory.name)
    resource.initialize()
    resource.remember("bridge:100:secret-capability")
    assert os.stat(resource.path).st_mode & 0o777 == 0o600
    raw = json.dumps(read_inventory(tmp_path))
    assert resource.record["proof"] not in raw
    assert "secret-capability" not in raw
    assert read_inventory(tmp_path)[0]["cleanup_candidate"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "cancelled", "failed", "disposed"])
async def test_terminal_cleanup_is_durable_and_idempotent(
    tmp_path: Path, bridge: BridgeFixture, outcome: str
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    result = await resource.finish(resource.generation, outcome)
    assert result.state == "closed"
    assert json.loads(resource.path.read_text())["terminal"] == outcome
    assert json.loads(resource.path.read_text())["surface_id"] == ""
    assert (await resource.finish(resource.generation, outcome)).state == "closed"


@pytest.mark.asyncio
async def test_failed_cleanup_survives_reconstruction(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    bridge.fail_close = True
    assert (await resource.finish(resource.generation, "failed")).state == "pending"
    record = json.loads(resource.path.read_text())
    assert record["terminal"] == "failed"
    assert record["surface_id"] == "bridge:100:private"
    # Reconstruction models a restarted host. The stable proof and allocation
    # survive; a new generation is presented with the previous CAS token.
    resumed = BrowserResource(tmp_path, tmp_path.name)
    resumed.initialize()
    assert resumed.params()["owner_proof"] == resource.record["proof"]
    assert resumed.params()["previous_generation"] == resource.generation
    assert resumed.generation != resource.generation
    bridge.fail_close = False
    await resumed.recover()
    assert resumed.record["surface_id"] == "bridge:100:private"
    with pytest.raises(RuntimeError, match="stale"):
        resource.remember("")


@pytest.mark.asyncio
async def test_pending_login_and_paused_retention_protect_terminal_scope(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.record["retention"] = "pending login"
    resource.remember("bridge:100:private")
    assert (await resource.finish(resource.generation, "completed")).state == "retained"
    assert resource.record["surface_id"] == "bridge:100:private"
    assert [call[0] for call in bridge.calls] == ["owner_recover", "owner_retain", "owner_finish"]


def test_execution_lease_fences_even_lazy_old_host(tmp_path: Path) -> None:
    lease = acquire_session_lease(tmp_path)
    old = BrowserResource(tmp_path, tmp_path.name)
    lease.release()
    successor = acquire_session_lease(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="lease changed"):
            old.initialize()
        current = BrowserResource(tmp_path, tmp_path.name)
        current.initialize()
        assert current.generation == successor.generation
    finally:
        successor.release()


@pytest.mark.asyncio
async def test_exact_cleanup_rejects_live_lease_and_stale_selection(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    lease = acquire_session_lease(tmp_path)
    try:
        resource = BrowserResource(tmp_path, tmp_path.name)
        resource.initialize()
        resource.record["terminal"] = "failed"
        resource.remember("bridge:100:private")
        with pytest.raises(SessionLeaseHeldError):
            await cleanup_exact(tmp_path, resource.generation)
        assert (await cleanup_exact(tmp_path, "wrong-generation")).state == "unresolved"
        assert not bridge.calls
    finally:
        lease.release()
    assert (await cleanup_exact(tmp_path, resource.generation)).state == "closed"


@pytest.mark.asyncio
async def test_dead_process_alone_never_authorizes_cleanup(
    tmp_path: Path, bridge: BridgeFixture
) -> None:
    resource = BrowserResource(tmp_path, tmp_path.name)
    resource.initialize()
    resource.remember("bridge:100:private")
    assert (await cleanup_exact(tmp_path, resource.generation)).state == "unresolved"
    assert not bridge.calls
