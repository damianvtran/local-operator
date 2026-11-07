"""McpToolCache: config_dir isolation, digest keys, 0600, silent degradation."""

from __future__ import annotations

import os
import time
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


def test_get_entry_returns_the_save_time_in_epoch_seconds(tmp_path: Path) -> None:
    """The catalog's ``last_seen_at`` is this value, so its unit is pinned here.

    Seconds, from ``time.time()`` at the ``put``: a client that read it as
    milliseconds rendered "20700 d ago", so the bound is on the real clock.
    The stale-row rules are :meth:`get`'s, since ``get`` is built on this.
    """
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    digest = config_digest(MCPStdioServerConfig(command="echo"))
    before = time.time()
    cache.put("echo", _TOOLS, digest)
    after = time.time()

    entry = cache.get_entry("echo", digest)

    assert entry is not None
    tools, saved_at = entry
    assert tools == _TOOLS
    assert before <= saved_at <= after
    other = config_digest(MCPStdioServerConfig(command="other"))
    assert cache.get_entry("echo", other) is None
    assert cache.get_entry("echo", digest) is None, "a mismatch drops the row for both readers"


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
    forwarded: list[object] = []

    class FakeManager:
        def __init__(
            self,
            cwd,
            tool_cache,
            auth_store=None,
            *,
            secret_base=None,
            register_secret=None,
            boundary=None,
        ):
            captured.append(tool_cache)
            forwarded.append(boundary)
            # The owner config root and the redaction sink are forwarded, not
            # defaulted away: a loader that dropped them would resolve `${NAME}`
            # against the wrong store and register nothing for the sinks to
            # scrub, and this double exists to catch exactly that at the call
            # site (`mcp/secret_refs.py`). ``boundary`` rides the same seam for
            # the world clip (QA round 1, Q1): the loader must forward it, and
            # the default stays ``None`` for every ordinary call.
            assert secret_base is None and register_secret is None

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
    assert forwarded == [None], "the ordinary call must not invent a boundary"
    asyncio.run(discover_and_load_mcp_tools(str(tmp_path), boundary=tmp_path))
    assert forwarded[-1] == tmp_path, "the loader must forward the world boundary"
    assert captured[0]._path.parent == tmp_path / "cfg"


def test_removed_server_delete_drops_every_digest(tmp_path: Path) -> None:
    cache = McpToolCache(tmp_path / "mcp_cache.db")
    a = config_digest(MCPStdioServerConfig(command="a"))
    cache.put("echo", _TOOLS, a)
    cache.delete("echo")
    assert cache.get("echo", a) is None


# --- annotations survive the cache (monitor readOnlyHint) ------------------
#
# MEASURED (v0.101.7, 2026-11-05 23:43): a runtime rolled over, registered its
# Slack tools from this cache before the live connect, and a monitor's first
# tick was refused "does not declare readOnlyHint" and struck — the writer had
# dropped ``annotations``, so every cache-built tool read as hint-less.


async def _noop_call(*_args, **_kwargs):  # pragma: no cover - never executed
    raise AssertionError("not called")


def test_cache_round_trip_keeps_read_only_hint(tmp_path: Path) -> None:
    from mcp.types import Tool, ToolAnnotations

    from local_operator.mcp.manager import _tool_to_cache_entry
    from local_operator.mcp.tool_bridge import build_agent_tool

    tool = Tool(
        name="read_channel",
        description="read",
        input_schema={"type": "object"},
        annotations=ToolAnnotations(read_only_hint=True, title="Read"),
    )
    entry = _tool_to_cache_entry(tool)
    # Wire spelling, never the SDK's snake-case field name.
    assert entry["annotations"] == {"readOnlyHint": True, "title": "Read"}

    cache = McpToolCache(tmp_path / "mcp_cache.db")
    digest = config_digest(MCPStdioServerConfig(command="slack"))
    cache.put("slack", [entry], digest)
    loaded = cache.get("slack", digest)
    assert loaded is not None and loaded == [entry]

    agent_tool = build_agent_tool("slack", loaded[0], _noop_call)
    assert agent_tool.mcp_annotations == {"readOnlyHint": True, "title": "Read"}


def test_a_tool_without_annotations_writes_no_annotations_key() -> None:
    from mcp.types import Tool

    from local_operator.mcp.manager import _tool_to_cache_entry

    entry = _tool_to_cache_entry(Tool(name="echo", input_schema={"type": "object"}))
    assert "annotations" not in entry


def test_an_old_cache_row_without_annotations_still_builds_a_tool(tmp_path: Path) -> None:
    """Rows written before annotations were cached must keep loading: the
    digest hashes the server CONFIG only, so the new key neither invalidates
    them nor forces a re-auth — they simply carry no hint until the live
    ``tools/list`` replaces them.
    """
    from local_operator.mcp.tool_bridge import build_agent_tool

    cache = McpToolCache(tmp_path / "mcp_cache.db")
    config = MCPStdioServerConfig(command="echo")
    digest = config_digest(config)
    cache.put("echo", _TOOLS, digest)  # the pre-fix row shape
    # The digest is a function of the config alone (no tool payload input).
    assert config_digest(config) == digest
    loaded = cache.get("echo", digest)
    assert loaded is not None and loaded == _TOOLS
    agent_tool = build_agent_tool("echo", loaded[0], _noop_call)
    assert agent_tool.mcp_annotations is None
    assert agent_tool.name.endswith("echo")
