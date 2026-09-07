"""McpToolCache: config_dir isolation, digest keys, 0600, silent degradation."""

from __future__ import annotations

import os
from pathlib import Path

from local_operator.mcp.config import MCPStdioServerConfig
from local_operator.mcp.tool_cache import (
    McpToolCache,
    config_digest,
    default_cache_path,
)
from local_operator.paths import config_dir

_TOOLS = [{"name": "echo", "description": "say", "inputSchema": {"type": "object"}}]


def test_default_path_honours_config_dir_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    path = default_cache_path()
    assert path == config_dir() / "mcp_cache.db"
    assert path.parent == tmp_path / "cfg"
    cache = McpToolCache()
    digest = config_digest(MCPStdioServerConfig(command="echo"))
    cache.put("echo", _TOOLS, digest)
    assert cache.get("echo", digest) == _TOOLS
    leaked = tmp_path / "home" / ".local-operator" / "mcp_cache.db"
    assert not leaked.exists()


def test_digest_mismatch_deletes_stale_row(tmp_path: Path) -> None:
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    old = config_digest(MCPStdioServerConfig(command="old-cmd"))
    new = config_digest(MCPStdioServerConfig(command="new-cmd"))
    assert old != new
    cache.put("echo", _TOOLS, old)
    assert cache.get("echo", old) == _TOOLS
    assert cache.get("echo", new) is None
    assert cache.get("echo", old) is None  # stale row dropped on mismatch


def test_name_only_get_is_a_miss_and_drops_rows(tmp_path: Path) -> None:
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    digest = config_digest(MCPStdioServerConfig(command="echo"))
    cache.put("echo", _TOOLS, digest)
    assert cache.get("echo") is None
    assert cache.get("echo", digest) is None


def test_put_without_digest_is_a_noop(tmp_path: Path) -> None:
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    cache.put("echo", _TOOLS)
    assert cache.get("echo", config_digest(MCPStdioServerConfig(command="echo"))) is None


def test_cache_file_is_0600(tmp_path: Path) -> None:
    path = tmp_path / "mcp_cache.db"
    cache = McpToolCache(path)
    cache.put("echo", _TOOLS, config_digest(MCPStdioServerConfig(command="echo")))
    assert path.exists()
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600


def test_discover_and_load_defaults_a_config_dir_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "cfg").mkdir()
    (tmp_path / "home").mkdir()
    captured: list[object] = []

    class FakeManager:
        def __init__(self, cwd, tool_cache, auth_store=None):
            captured.append(tool_cache)

        async def discover_and_connect(self):
            from local_operator.mcp.manager import McpLoadResult

            return McpLoadResult()

    monkeypatch.setattr("local_operator.mcp.McpManager", FakeManager)
    import asyncio

    from local_operator.mcp import discover_and_load_mcp_tools
    from local_operator.mcp.tool_cache import McpToolCache

    asyncio.run(discover_and_load_mcp_tools(str(tmp_path)))
    assert len(captured) == 1
    assert isinstance(captured[0], McpToolCache)
    assert captured[0]._path.parent == tmp_path / "cfg"


def test_removed_server_delete_drops_every_digest(tmp_path: Path) -> None:
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    a = config_digest(MCPStdioServerConfig(command="a"))
    cache.put("echo", _TOOLS, a)
    cache.delete("echo")
    assert cache.get("echo", a) is None
