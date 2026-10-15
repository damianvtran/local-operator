"""``SessionSpec.mcp_servers``: session-only MCP servers (D8 revised).

Built through the REAL ``sdk.open_session`` and mock hosting, so the assertion
is about the session's own ``McpManager`` — the object every tool load,
reconnect and ``/mcp`` listing reads — not about an argument being forwarded.
The server URL is a closed loopback port: the connect fails fast, which is
irrelevant here; what is pinned is that the CONFIG reached the session.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import sdk
from local_operator.mcp.config import MCPHttpServerConfig
from local_operator.mcp.manager import SESSION_MCP_SOURCE, McpManager
from local_operator.session.spec import SessionRoots, SessionSpec

GATEWAY = {
    "type": "http",
    "url": "http://127.0.0.1:1/mcp",
    "headers": {"Authorization": "Bearer gateway-token"},
}


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionRoots:
    """The same isolated shape ``tests/unit/test_sdk.py`` uses."""
    from local_operator import agent_shell

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.delenv("LOCAL_OPERATOR_HOME", raising=False)
    monkeypatch.delenv(agent_shell.AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(agent_shell.MAY_DELEGATE_ENV, raising=False)
    monkeypatch.delenv(agent_shell.ALLOW_NESTED_SESSION_ENV, raising=False)
    return SessionRoots(config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True)


def _mcp_files(*roots: Path) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        found.extend(p for p in root.rglob("*") if p.name in (".mcp.json", "mcp.json"))
    return found


@pytest.mark.asyncio
async def test_a_passed_http_server_reaches_the_sessions_mcp_config(scratch: SessionRoots) -> None:
    spec = SessionSpec(hosting="test", model="mock", mcp_servers={"gateway": GATEWAY})
    async with sdk.open_session(spec, roots=scratch) as session:
        manager = cast(Any, session).mcp_manager
        assert isinstance(manager, McpManager)
        cfg = manager.get_server_config("gateway")
        assert isinstance(cfg, MCPHttpServerConfig)
        assert cfg.url == GATEWAY["url"]
        assert cfg.headers == GATEWAY["headers"], "the client's headers pass through"
        assert manager.get_source("gateway") == SESSION_MCP_SOURCE
        assert "gateway" in cast(Any, session).mcp_startup.configured
    # Never written anywhere a file loader (or another session) could read it.
    assert _mcp_files(scratch.config_path, scratch.agent_home_path, scratch.cwd_path) == []


@pytest.mark.asyncio
async def test_without_mcp_servers_nothing_is_added(scratch: SessionRoots) -> None:
    async with sdk.open_session(SessionSpec(hosting="test", model="mock"), roots=scratch) as s:
        manager = cast(Any, s).mcp_manager
        names = manager.get_all_server_names() if manager is not None else []
        assert "gateway" not in names


def test_a_file_configured_server_wins_a_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("HOME", str(tmp_path))
    (config_dir / "mcp.json").write_text(
        json.dumps({"mcpServers": {"gateway": {"type": "http", "url": "http://file.example/mcp"}}})
    )
    manager = McpManager(
        tmp_path,
        extra_configs={"gateway": GATEWAY, "extra": {"url": "http://127.0.0.1:1/x"}},
    )
    with caplog.at_level("WARNING"):
        configs, sources = manager._load_configs()
    assert cast(Any, configs["gateway"]).url == "http://file.example/mcp"
    assert sources["gateway"] == str(config_dir / "mcp.json")
    assert sources["extra"] == SESSION_MCP_SOURCE
    assert "shadowed" in caplog.text


def test_a_spawned_session_refuses_mcp_servers() -> None:
    """A spawned runtime cannot carry session-only servers, so it says so."""
    from local_operator.session.spec import SessionSpecError

    with pytest.raises(SessionSpecError, match="mcp_servers"):
        sdk._refuse_spawn_extras(SessionSpec(mcp_servers={"gateway": GATEWAY}))
