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
from local_operator.harness.types import ToolContext
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


def _cache_rows(path: Path, server: str) -> int:
    """Rows for ``server`` in a tool cache file, read raw.

    Raw because ``McpToolCache.get`` needs the config digest for a hit — a
    name-only read is always a miss, so it cannot prove a row is ABSENT.
    """
    import sqlite3

    if not path.exists():
        return 0
    with sqlite3.connect(path) as conn:
        try:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM mcp_tool_cache WHERE server = ?", (server,)
            ).fetchone()
        except sqlite3.OperationalError:  # no table: nothing was ever written
            return 0
    return int(count)


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
async def test_without_mcp_servers_the_extra_configs_kwarg_is_left_out(
    scratch: SessionRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative case stays byte-for-byte today's call: no ``extra_configs``
    kwarg reaches ``McpManager`` at all (review R6), rather than an empty one."""
    from local_operator import mcp as mcp_package

    seen: list[dict[str, Any]] = []
    real = mcp_package.McpManager

    def recording(*args: Any, **kwargs: Any) -> McpManager:
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mcp_package, "McpManager", recording)
    async with sdk.open_session(SessionSpec(hosting="test", model="mock"), roots=scratch):
        pass
    assert seen, "the session built a manager"
    assert all("extra_configs" not in kwargs for kwargs in seen)


def test_an_empty_mcp_servers_mapping_is_the_absent_value() -> None:
    """R7: ``{}`` declares nothing, so spawn/attach must not refuse it."""
    spec = SessionSpec(mcp_servers={})
    assert spec.mcp_servers is None
    sdk._refuse_spawn_extras(spec)


# --- a REAL streamable-HTTP server ------------------------------------------


@pytest.fixture
def gateway_server() -> Any:
    """A loopback streamable-HTTP MCP server with one tool, in a thread.

    The Gateway's coordination server stand-in (QA round 1's Q1 repro): the
    tests below need its tools to really list, so a closed port will not do.
    """
    import socket
    import threading
    import time

    uvicorn = pytest.importorskip("uvicorn")
    from mcp.server.mcpserver import MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    server = MCPServer("gw")

    @server.tool()
    def coord_echo(text: str) -> str:
        """Echo text back so a caller can prove the round trip."""
        return f"GW-ECHO:{text}"

    app = server.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not uv.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the loopback MCP server did not start")
        time.sleep(0.02)
    yield {"type": "http", "url": f"http://127.0.0.1:{port}/mcp", "preloadTools": True}
    uv.should_exit = True
    thread.join(timeout=10)


async def _settled(session: Any, name: str) -> None:
    """Wait (bounded) until the session's tool list holds ``name``."""
    import asyncio

    async def poll() -> None:
        while name not in {tool.name for tool in session._tools}:
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout=20)


@pytest.mark.asyncio
async def test_a_session_only_servers_tools_are_directly_callable(
    scratch: SessionRoots, gateway_server: dict[str, Any]
) -> None:
    """Q1: the model must SEE the client's tools — in the tools array, not
    behind ``mcp://`` — because a model that has never heard of the server
    will never read its URL. ``preloadTools`` (set by the ACP mapping) is the
    lever; this pins that it takes effect for a session-only server."""
    spec = SessionSpec(hosting="test", model="mock", mcp_servers={"gw": gateway_server})
    async with sdk.open_session(spec, roots=scratch) as session:
        live = cast(Any, session)
        await _settled(live, "mcp__gw_coord_echo")
        [tool] = [t for t in live._tools if t.name == "mcp__gw_coord_echo"]
        result = await tool.execute("call-1", {"text": "ping"}, None, None, ToolContext())
        assert "GW-ECHO:ping" in result.text

        # R6: a reload re-reads the files, and must not drop the client's server.
        manager = live.mcp_manager
        await manager.reload()
        assert manager.get_source("gw") == SESSION_MCP_SOURCE
        await _settled(live, "mcp__gw_coord_echo")

    # R4: the operator's shared tool cache holds nothing for it.
    from local_operator.mcp.tool_cache import default_cache_path

    assert _cache_rows(default_cache_path(), "gw") == 0
    assert _mcp_files(scratch.config_path, scratch.agent_home_path, scratch.cwd_path) == []


@pytest.mark.asyncio
async def test_connect_by_name_finds_a_session_only_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gateway_server: dict[str, Any]
) -> None:
    """R6: ``connect_configured_server`` (the by-name path ``/mcp login`` and
    the desktop Test use) resolves a session-only server, which lives in no
    file it could re-read — and writes nothing to the tool cache for it (R4)."""
    from local_operator.mcp.tool_cache import McpToolCache

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_path = tmp_path / "cache.db"
    cache = McpToolCache(cache_path)
    # An operator row for a same-named server configured elsewhere: a put for
    # the session-only server would DELETE it first (review R4).
    cache.put("gw", [{"name": "theirs"}], "operator-digest")
    assert _cache_rows(cache_path, "gw") == 1
    manager = McpManager(tmp_path, cache, extra_configs={"gw": gateway_server})
    try:
        conn = await manager.connect_configured_server("gw", interactive=False)
        assert [tool.name for tool in conn.tools] == ["coord_echo"]
        assert manager.get_source("gw") == SESSION_MCP_SOURCE
    finally:
        # Opened in THIS task, so closed in it (see ``disconnect_all``).
        await manager.disconnect_all(in_task=True)
    assert cache.get("gw", "operator-digest") == [{"name": "theirs"}], "nothing evicted"
    assert _cache_rows(cache_path, "gw") == 1, "and nothing added"


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
