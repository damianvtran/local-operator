"""Manager: fast-startup gate, deferred tools, reconnect breaker, epochs.

Every test stubs the ``_connect_server`` seam — no real MCP server, network,
or SDK transport is required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import ToolContext, ToolResult
from local_operator.mcp.manager import (
    RECONNECT_BURST_LIMIT,
    McpManager,
    ServerConnection,
    stdio_start_new_session,
    build_cmd_exe_argv,
    build_stdio_argv,
)
from local_operator.mcp.tool_cache import McpToolCache


def _tool(name: str, schema: dict | None = None) -> SimpleNamespace:
    """SDK-shaped Tool model stand-in."""
    return SimpleNamespace(
        name=name,
        description=f"{name} desc",
        input_schema=schema or {"type": "object", "properties": {"q": {"type": "string"}}},
    )


class FakeSession:
    """ClientSession stand-in: records calls, returns canned results."""

    def __init__(self, call_result: Any = None, raise_on_call: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.call_result = call_result or SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")], is_error=False
        )
        self.raise_on_call = raise_on_call

    async def list_tools(self, params: Any = None) -> SimpleNamespace:
        return SimpleNamespace(tools=[_tool("search")], next_cursor=None)

    async def call_tool(self, name: str, arguments: dict | None, **kwargs: Any) -> Any:
        self.calls.append((name, dict(arguments or {})))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.call_result


def _make_conn(name: str, cfg: Any, session: FakeSession | None = None) -> ServerConnection:
    return ServerConnection(name=name, config=cfg, session=session or FakeSession(), tools=[_tool("search")])


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    """Project dir with two stdio servers configured."""
    (tmp_path / ".local-operator").mkdir()
    (tmp_path / ".local-operator" / "mcp.json").write_text(
        '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd"},'
        ' "slow": {"type": "stdio", "command": "slow-cmd"}}}',
        encoding="utf-8",
    )
    return tmp_path


class TestFastStartupGate:
    @pytest.mark.asyncio
    async def test_instant_live_slow_deferred_from_cache(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One instant connect + one 5s connect raced with the 250 ms gate."""
        cache = McpToolCache(tmp_path / "cache.db")
        cache.put(
            "slow",
            [{"name": "search", "description": "cached search", "inputSchema": {"type": "object"}}],
        )
        manager = McpManager(str(project), tool_cache=cache)

        slow_release = asyncio.Event()
        sessions: dict[str, FakeSession] = {}

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            session = FakeSession()
            sessions[name] = session
            if name == "slow":
                await asyncio.wait_for(slow_release.wait(), timeout=10)
            return _make_conn(name, cfg, session)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        changed: list[list[str]] = []
        manager.set_on_tools_changed(lambda tools: changed.append([t.name for t in tools]))

        result = await manager.discover_and_connect()

        # Gate outcome: fast live, slow deferred from cache, no errors.
        assert result.errors == {}
        assert result.connected_servers == ["fast"]
        names = [t.name for t in result.tools]
        assert names == ["mcp__fast_search", "mcp__slow_search"]  # sorted by name
        assert manager.get_connection_status("fast") == "connected"
        assert manager.get_connection_status("slow") == "connecting"
        assert manager.get_tool_meta("mcp__slow_search")["deferred"] is True
        assert manager.get_tool_meta("mcp__fast_search")["deferred"] is False

        # Deferred execute parks until the connection lands.
        slow_tool = next(t for t in manager.get_tools() if t.name == "mcp__slow_search")
        exec_task = asyncio.create_task(
            slow_tool.execute("call-1", {"q": "hi"}, None, None, ToolContext())
        )
        await asyncio.sleep(0.02)
        assert not exec_task.done()

        # Release the slow connect: continuation swaps the live tool in.
        slow_release.set()
        exec_result = await asyncio.wait_for(exec_task, timeout=10)
        assert isinstance(exec_result, ToolResult)
        assert exec_result.is_error is False
        assert exec_result.text == "ok"
        assert sessions["slow"].calls == [("search", {"q": "hi"})]

        # Live connection registered; tools rebuilt from the real list.
        assert manager.get_connection_status("slow") == "connected"
        assert manager.get_tool_meta("mcp__slow_search")["deferred"] is False
        # on_tools_changed fired when the deferred server landed.
        assert any("mcp__slow_search" in snap for snap in changed)

        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_pending_without_cache_contributes_nothing(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        slow_release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(slow_release.wait(), timeout=10)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        result = await manager.discover_and_connect()
        assert [t.name for t in result.tools] == ["mcp__fast_search"]

        slow_release.set()
        await asyncio.sleep(0.02)  # let the continuation run
        assert [t.name for t in manager.get_tools()] == ["mcp__fast_search", "mcp__slow_search"]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_rejected_connect_becomes_error_entry_others_continue(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "fast":
                raise RuntimeError("boom: spawn failed")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        result = await manager.discover_and_connect()
        assert "fast" in result.errors and "boom" in result.errors["fast"]
        assert [t.name for t in result.tools] == ["mcp__slow_search"]
        await manager.disconnect_all()


class TestDeferredExecuteFailure:
    @pytest.mark.asyncio
    async def test_deferred_execute_error_when_connect_fails(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = McpToolCache(tmp_path / "cache.db")
        cache.put("slow", [{"name": "search", "description": "", "inputSchema": {}}])
        manager = McpManager(str(project), tool_cache=cache)
        release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(release.wait(), timeout=10)
                raise RuntimeError("slow server exploded")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        slow_tool = next(t for t in manager.get_tools() if t.name == "mcp__slow_search")

        release.set()
        result = await asyncio.wait_for(
            slow_tool.execute("c", {}, None, None, ToolContext()), timeout=10
        )
        assert result.is_error is True
        assert "exploded" in result.text
        # The deferred tool slice is dropped and the change fires.
        await asyncio.sleep(0.01)
        assert "mcp__slow_search" not in [t.name for t in manager.get_tools()]
        await manager.disconnect_all()


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_breaker_trips_after_five_failures_in_window(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        real_sleep = asyncio.sleep
        sleeps: list[float] = []

        async def instant_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        async def failing_connect(name: str, cfg: Any) -> ServerConnection:
            raise RuntimeError("still down")

        monkeypatch.setattr(manager, "_connect_server", failing_connect)
        await manager.discover_and_connect()

        # Drive the reconnect chain: each failed attempt schedules the next
        # (backoff sleeps are instant), until the breaker trips.
        manager._schedule_reconnect("fast")
        for _ in range(60):
            await real_sleep(0)
            if manager.reconnect_suspended("fast"):
                break

        assert manager.reconnect_suspended("fast") is True
        # Backoff escalates 0.5, 1, 2, 4 and caps at 4 (five failed attempts).
        assert [d for d in sleeps if d > 0] == [0.5, 1.0, 2.0, 4.0, 4.0]

        # Manual reconnect resets the breaker and reconnects.
        async def good_connect(name: str, cfg: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        conn = await manager.reconnect_server("fast")
        assert conn is not None
        assert manager.reconnect_suspended("fast") is False
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_epoch_prevents_resurrection_after_disconnect_all(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connected = 0

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            nonlocal connected
            connected += 1
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        assert connected == 2

        epoch_at_disconnect = manager._epoch
        await manager.disconnect_all()
        assert manager._epoch == epoch_at_disconnect + 1

        # A stale reconnect task from the old epoch must not resurrect anything.
        cfg = manager.get_server_config("fast")
        assert cfg is not None
        await manager._reconnect("fast", 0.0, epoch_at_disconnect)
        assert manager.get_connection("fast") is None
        assert connected == 2


class TestToolCallHygieneAndRetry:
    @pytest.mark.asyncio
    async def test_call_strips_intent_and_retries_once_on_transport_close(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        sessions: list[FakeSession] = []
        first_session = FakeSession(raise_on_call=RuntimeError("Transport closed"))
        sessions.append(first_session)
        call_count = {"n": 0}

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            call_count["n"] += 1
            if name == "fast" and call_count["n"] == 1:
                return _make_conn(name, cfg, first_session)
            return _make_conn(name, cfg, FakeSession())

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()

        tool = next(t for t in manager.get_tools() if t.name == "mcp__fast_search")
        result = await tool.execute("c1", {"q": "x", "i": "intent"}, None, None, ToolContext())
        assert result.is_error is False
        assert result.text == "ok"
        # First session got the hygienic args (no 'i') then raised.
        assert first_session.calls == [("search", {"q": "x"})]
        # Exactly one reconnect + one retry: two connects total for 'fast',
        # and the successful session saw the same cleaned args.
        assert call_count["n"] == 3  # fast, slow, fast-reconnect
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_non_retriable_error_returns_error_result_without_reconnect(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connects = {"n": 0}

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            connects["n"] += 1
            if name == "fast":
                return _make_conn(name, cfg, FakeSession(raise_on_call=ValueError("tool not found")))
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        baseline = connects["n"]

        tool = next(t for t in manager.get_tools() if t.name == "mcp__fast_search")
        result = await tool.execute("c1", {"q": "x"}, None, None, ToolContext())
        assert result.is_error is True
        assert "tool not found" in result.text
        assert connects["n"] == baseline  # no reconnect for non-retriable errors
        await manager.disconnect_all()


class TestStdioHardening:
    def test_start_new_session_platform_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import local_operator.mcp.manager as manager_mod

        monkeypatch.setattr(manager_mod.sys, "platform", "linux")
        assert stdio_start_new_session() is True
        monkeypatch.setattr(manager_mod.sys, "platform", "darwin")
        assert stdio_start_new_session() is False
        monkeypatch.setattr(manager_mod.sys, "platform", "win32")
        assert stdio_start_new_session() is False

    def test_posix_argv_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import local_operator.mcp.manager as manager_mod

        monkeypatch.setattr(manager_mod.sys, "platform", "linux")
        assert build_stdio_argv("npx", ["-y", "pkg"]) == ["npx", "-y", "pkg"]

    def test_windows_batch_cmd_exe_hardening(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import local_operator.mcp.manager as manager_mod

        monkeypatch.setattr(manager_mod.sys, "platform", "win32")
        argv = build_cmd_exe_argv("cmd.exe", r"C:\work\%TOKEN%\server.cmd", ['a"b', "plain"])
        assert argv[:5] == ["cmd.exe", "/d", "/e:ON", "/v:OFF", "/c"]
        line = argv[5]
        # Outer quote pair, percent neutralized, interior quote doubled.
        assert line.startswith('""') and line.endswith('"')
        assert "%%cd:~,%" in line
        assert 'a""b' in line
        assert "plain" in line
        # NUL/CR/LF rejected outright.
        with pytest.raises(ValueError):
            build_cmd_exe_argv("cmd.exe", "evil\ncmd", [])
