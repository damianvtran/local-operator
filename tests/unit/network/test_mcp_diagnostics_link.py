"""MCP diagnostics on the normal network tool/CLI/peer wire, not helper-only."""

from __future__ import annotations

import json
import time
from contextlib import closing
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import readiness, tool
from local_operator.network.credentials import offers
from local_operator.providers.auth_store import AuthStore
from tests.unit.mcp.test_grant_diagnostics import URL, _sdk_login
from tests.unit.network.test_readiness_link import (
    _bind_and_start,
    _capability,
    _pair,
    _set_peer_endpoints,
    _settle_the_peer_link,
)
from tests.unit.network.test_relay_e2e import devices as _devices_fixture

# Reuse the package's two real software-key relays and their exact-pid teardown.
devices = _devices_fixture


@pytest.mark.asyncio
async def test_mcp_diagnostics_reach_actual_network_tool_without_claiming_inheritance(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pair_devices: Any = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = pair_devices
    record, _h, _p = _pair(pair_devices, monkeypatch)
    capsys.readouterr()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    live = _bind_and_start(server_b, record)
    _settle_the_peer_link(server_a, server_b, record, live)
    _set_peer_endpoints(server_a, record, server_b.identity.device_id, [live])
    (server_b.root / "mcp.json").write_text(
        json.dumps({"mcpServers": {"fixture": {"type": "http", "url": URL}}})
    )
    with closing(AuthStore(db_path=server_b.root / "auth.db", config_dir=server_b.root)) as store:
        await _sdk_login(store, monkeypatch)
        result = await tool.execute_network(
            "synthetic-mcp-ready",
            {"action": "ready", "peer": server_b.identity.name},
            None,
            None,
            ToolContext(cwd="."),
        )
        assert result.details is not None
        row = _capability(result.details["network"], readiness.CAPABILITY_MCP_CREDENTIAL)
        diagnostic = row["observed"]["auth_diagnostics"]
        assert diagnostic["provenance"] == "native"
        assert diagnostic["identity_verified"] is False
        assert diagnostic["sharing_mode"] == "finite_only"
        assert diagnostic["refresh_presence"] == "present"
        assert result.details["network"]["ok"] is False
        item = next(
            item for item in offers.build_items(server_b.root) if item["kind"] == "mcp-rotating"
        )
        assert item["auth_diagnostics"] == diagnostic
        assert "[redacted]" not in json.dumps(item)
        from local_operator.mcp.auth import TOKENS_OBTAINED_AT_KEY, McpTokenStorage

        storage = McpTokenStorage(URL, store)
        data = dict(store.list_credentials("mcp-oauth")[0].data)
        data["tokens"] = {"access_token": "[redacted]", "token_type": "Bearer", "expires_in": 1}
        data[TOKENS_OBTAINED_AT_KEY] = time.time() - 120
        assert storage._write(data)
        expired = await tool.execute_network(
            "synthetic-mcp-expired",
            {"action": "ready", "peer": server_b.identity.name},
            None,
            None,
            ToolContext(cwd="."),
        )
        assert expired.details is not None
        row = _capability(expired.details["network"], readiness.CAPABILITY_MCP_CREDENTIAL)
        assert row["ok"] is False and row["code"] == "local_grant_unavailable"
        assert row["observed"]["auth_diagnostics"]["access_state"] == "expired"
