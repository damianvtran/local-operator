"""Manager: fast-startup gate, deferred tools, reconnect breaker, epochs.

Every test stubs the ``_connect_server`` seam — no real MCP server, network,
or SDK transport is required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from local_operator.harness.types import AbortSignal, ToolContext, ToolResult
from local_operator.mcp.manager import (
    McpManager,
    ServerConnection,
    build_cmd_exe_argv,
    build_stdio_argv,
    stdio_start_new_session,
)
from local_operator.mcp.tool_cache import McpToolCache


def _tool(name: str, schema: dict[str, Any] | None = None) -> Tool:
    """One SDK ``Tool`` as a server would advertise it."""
    return Tool(
        name=name,
        description=f"{name} desc",
        input_schema=schema or {"type": "object", "properties": {"q": {"type": "string"}}},
    )


class FakeSession:
    """ClientSession stand-in: records calls, returns canned results."""

    def __init__(
        self,
        call_result: CallToolResult | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.call_result = call_result or CallToolResult(
            content=[TextContent(type="text", text="ok")], is_error=False
        )
        self.raise_on_call = raise_on_call

    async def list_tools(self, *, params: Any = None) -> ListToolsResult:
        return ListToolsResult(tools=[_tool("search")], next_cursor=None)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> CallToolResult:
        self.calls.append((name, dict(arguments or {})))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.call_result


def _make_conn(name: str, cfg: Any, session: FakeSession | None = None) -> ServerConnection:
    return ServerConnection(
        name=name, config=cfg, session=session or FakeSession(), tools=[_tool("search")]
    )


def _tool_meta(manager: McpManager, tool_name: str) -> dict[str, Any]:
    """``get_tool_meta`` narrowed to non-None for assertions."""
    meta = manager.get_tool_meta(tool_name)
    assert meta is not None
    return cast(dict[str, Any], meta)


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


class TestSingleServerConnection:
    @pytest.mark.asyncio
    async def test_connect_configured_server_targets_only_named_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {'
            '"only": {"type": "http", "url": "https://example.com/mcp"},'
            '"other": {"type": "stdio", "command": "other-cmd"}'
            "}}",
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        attempted: list[tuple[str, float | None]] = []

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            attempted.append((name, cfg.timeout))
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        conn = await manager.connect_configured_server("only", timeout_ms=600_000)
        assert conn.name == "only"
        assert attempted == [("only", 600_000)]
        assert manager.get_connection("only") is conn
        assert manager.get_connection("other") is None
        with pytest.raises(Exception, match="not configured"):
            await manager.connect_configured_server("missing")
        await manager.disconnect_all()


class TestFastStartupGate:
    @pytest.mark.asyncio
    async def test_instant_live_slow_deferred_from_cache(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One instant connect + one 5s connect raced with the 250 ms gate."""
        cache = McpToolCache(tmp_path / "cache.db")
        cache.put(
            "slow",
            [
                {
                    "name": "search",
                    "description": "cached search",
                    "inputSchema": {"type": "object"},
                }
            ],
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
        assert _tool_meta(manager, "mcp__slow_search")["deferred"] is True
        assert _tool_meta(manager, "mcp__fast_search")["deferred"] is False

        # Deferred execute parks until the connection lands.
        slow_tool = next(t for t in manager.get_tools() if t.name == "mcp__slow_search")

        async def _slow_call() -> ToolResult:
            return await slow_tool.execute("call-1", {"q": "hi"}, None, None, ToolContext())

        exec_task = asyncio.create_task(_slow_call())
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
        assert _tool_meta(manager, "mcp__slow_search")["deferred"] is False
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
        assert [t.name for t in manager.get_tools()] == [
            "mcp__fast_search",
            "mcp__slow_search",
        ]
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
                return _make_conn(
                    name, cfg, FakeSession(raise_on_call=ValueError("tool not found"))
                )
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


class TestToolsListChangedNoInlineAwait:
    """MCP-05: refresh runs as a spawned task, never inline on the SDK path."""

    @pytest.mark.asyncio
    async def test_notification_spawns_task_not_inline(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        calls: list[str] = []

        async def fake_refresh(name: str) -> None:
            calls.append(name)

        monkeypatch.setattr(manager, "refresh_server_tools", fake_refresh)
        conn = _make_conn("fast", manager.get_server_config("fast") or SimpleNamespace())
        message = SimpleNamespace(method="notifications/tools/list_changed")

        await manager._on_session_message("fast", conn, cast(Any, message))
        # The handler must NOT have awaited the refresh inline.
        assert calls == []
        await asyncio.sleep(0.01)
        assert calls == ["fast"]
        await manager.disconnect_all()


class TestCallSiteReconnectGuards:
    """MCP-06: _reconnect_for_call respects epoch/disposed/breaker."""

    @pytest.mark.asyncio
    async def test_retry_across_disconnect_all_does_not_resurrect(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connects = {"n": 0}

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            connects["n"] += 1
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        assert connects["n"] == 2

        await manager.disconnect_all()
        # A call-site retry firing after dispose must NOT reconnect.
        assert await manager._reconnect_for_call("fast") is None
        assert connects["n"] == 2  # no new connection attempts
        assert manager.get_connection("fast") is None

    @pytest.mark.asyncio
    async def test_call_site_retry_respects_suspended_breaker(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connects = {"n": 0}

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            connects["n"] += 1
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        baseline = connects["n"]
        manager._reconnect_suspended.add("fast")

        assert await manager._reconnect_for_call("fast") is None
        assert connects["n"] == baseline
        await manager.disconnect_all()


class TestBreakerWindowSeparateFromLadder:
    """MCP-07: success resets the backoff ladder but NOT the breaker window."""

    @pytest.mark.asyncio
    async def test_flapping_server_trips_after_burst(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connects = {"n": 0}
        real_sleep = asyncio.sleep

        async def instant_sleep(delay: float) -> None:
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        async def good_connect(name: str, cfg: Any) -> ServerConnection:
            connects["n"] += 1
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        assert manager.get_connection("fast") is not None

        # Connect/die cycle: each death schedules a reconnect (window event);
        # each success resets the LADDER but the window keeps accumulating.
        for _ in range(6):
            conn = manager.get_connection("fast")
            assert conn is not None
            conn.closed_event.set()  # transport dies
            # Wait until the watcher reconnects (fresh conn) or trips.
            for _ in range(50):
                await real_sleep(0)
                if manager.reconnect_suspended("fast"):
                    break
                new = manager.get_connection("fast")
                if new is not None and new is not conn:
                    break
            if manager.reconnect_suspended("fast"):
                break

        # >5 events inside the 30 s window: auto-reconnect is suspended.
        assert manager.reconnect_suspended("fast") is True
        await manager.disconnect_all()


class TestBreakerTrippedCallsFailPromptly:
    """MCP-08/MCP-19: parked waiters get McpConnectionError, never a hang."""

    @pytest.mark.asyncio
    async def test_breaker_tripped_deferred_call_raises_promptly(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = McpToolCache(tmp_path / "cache.db")
        cache.put(
            "fast",
            [{"name": "search", "description": "", "inputSchema": {"type": "object"}}],
        )
        manager = McpManager(str(project), tool_cache=cache)
        state = {"fail": False}
        real_sleep = asyncio.sleep

        async def instant_sleep(delay: float) -> None:
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        async def flaky_connect(name: str, cfg: Any) -> ServerConnection:
            if state["fail"]:
                raise RuntimeError("still down")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", flaky_connect)
        await manager.discover_and_connect()
        fast_conn = manager.get_connection("fast")
        assert fast_conn is not None

        # Kill the transport, then make every reconnect fail until the
        # breaker trips and abandons the waiter future.
        state["fail"] = True
        fast_conn.closed_event.set()
        for _ in range(200):
            await real_sleep(0)
            if manager.reconnect_suspended("fast"):
                break
        assert manager.reconnect_suspended("fast") is True

        # The deferred execute must fail promptly with McpConnectionError
        # (surfaced as a tool error), not hang on a never-settled future.
        tool = next(t for t in manager.get_tools() if t.name == "mcp__fast_search")
        result = await asyncio.wait_for(
            tool.execute("c1", {"q": "x"}, None, None, ToolContext()), timeout=5
        )
        assert result.is_error is True
        assert "MCP error" in result.text
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_disconnect_server_fails_parked_deferred_execute(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = McpToolCache(tmp_path / "cache.db")
        cache.put("slow", [{"name": "search", "description": "", "inputSchema": {}}])
        manager = McpManager(str(project), tool_cache=cache)
        release = asyncio.Event()

        async def slow_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(release.wait(), timeout=30)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", slow_connect)
        await manager.discover_and_connect()
        slow_tool = next(t for t in manager.get_tools() if t.name == "mcp__slow_search")

        # Park the deferred execute on the connect waiter FIRST, then
        # disconnect: the parked call must fail, not hang (MCP-19).
        async def _parked_call() -> ToolResult:
            return await slow_tool.execute("c", {}, None, None, ToolContext())

        exec_task = asyncio.create_task(_parked_call())
        await asyncio.sleep(0.02)
        assert not exec_task.done()
        await manager.disconnect_server("slow")
        result = await asyncio.wait_for(exec_task, timeout=5)
        assert result.is_error is True
        assert "disconnected" in result.text
        release.set()
        await manager.disconnect_all()


class TestToolNameCollision:
    """MCP-09: cross-server collisions resolved by stable origin key."""

    @staticmethod
    def _echo_list(tool_name: str) -> Any:
        async def list_tools(*, params: Any = None) -> ListToolsResult:
            return ListToolsResult(tools=[_tool(tool_name)], next_cursor=None)

        return list_tools

    @pytest.mark.asyncio
    async def test_reviewer_collision_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"my-server": {"type": "stdio", "command": "a"},'
            ' "my": {"type": "stdio", "command": "b"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            tool_name = "a_b" if name == "my-server" else "server_a_b"
            conn = _make_conn(name, cfg)
            conn.tools = [_tool(tool_name)]
            # Echo the same tool on refresh so the origin set stays stable.
            assert conn.session is not None
            conn.session.list_tools = self._echo_list(tool_name)
            return conn

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()

        names = [t.name for t in manager.get_tools()]
        # Both create_mcp_tool_name calls mint the same base name:
        #   ("my-server", "a_b") and ("my", "server_a_b") -> mcp__my_server_a_b
        # The origin that sorts FIRST keeps the base; the later one gets _2.
        assert sorted(names) == ["mcp__my_server_a_b", "mcp__my_server_a_b_2"]
        # Deterministic by origin key, not registration order:
        # ("my", "server_a_b") < ("my-server", "a_b").
        assert _tool_meta(manager, "mcp__my_server_a_b")["server_name"] == "my"
        assert _tool_meta(manager, "mcp__my_server_a_b_2")["server_name"] == "my-server"
        assert any("collision" in rec.message for rec in caplog.records)

        # A refresh of the LATER server must not flip ownership.
        await manager.refresh_server_tools("my")
        assert _tool_meta(manager, "mcp__my_server_a_b")["server_name"] == "my"
        await manager.disconnect_all()


class TestAbortStaysAbort:
    """MCP-16: abort raises real CancelledError, never an error result."""

    @pytest.mark.asyncio
    async def test_execute_raises_cancelled_on_abort(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))

        class HangingSession:
            async def list_tools(self, *, params: Any = None) -> ListToolsResult:
                return ListToolsResult(tools=[_tool("search")], next_cursor=None)

            async def call_tool(
                self,
                name: str,
                arguments: dict[str, Any] | None = None,
                read_timeout_seconds: float | None = None,
            ) -> CallToolResult:
                await asyncio.sleep(3600)
                raise AssertionError("unreachable: the call is aborted first")

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            return ServerConnection(
                name=name, config=cfg, session=HangingSession(), tools=[_tool("search")]
            )

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        tool = next(t for t in manager.get_tools() if t.name == "mcp__fast_search")

        signal = AbortSignal()
        signal.abort()  # abort already set when the call starts
        with pytest.raises(asyncio.CancelledError):
            await tool.execute("c1", {"q": "x"}, signal, None, ToolContext())
        await manager.disconnect_all()


class TestReload:
    """MCP-17: reload bumps epoch, cancels reconnects, drops removed servers."""

    @pytest.mark.asyncio
    async def test_reload_drops_removed_server(
        self, project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cache = McpToolCache(tmp_path / "cache.db")
        manager = McpManager(str(project), tool_cache=cache)

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        assert sorted(t.name for t in manager.get_tools()) == [
            "mcp__fast_search",
            "mcp__slow_search",
        ]
        epoch_before = manager._epoch

        # Remove 'slow' from the config, then reload in place.
        (project / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd"}}}',
            encoding="utf-8",
        )
        result = await manager.reload()
        assert result.errors == {}
        assert manager._epoch == epoch_before + 1
        assert [t.name for t in manager.get_tools()] == ["mcp__fast_search"]
        assert manager.get_connection("slow") is None
        await manager.disconnect_all()


class TestSecuritySurface:
    """MCP-12: first connect of a project stdio server logs a WARNING."""

    @pytest.mark.asyncio
    async def test_project_stdio_server_warns_once(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        import logging

        manager = McpManager(str(project))

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        with caplog.at_level(logging.WARNING, logger="local_operator.mcp.manager"):
            await manager.discover_and_connect()

        warns = [r for r in caplog.records if "project-configured stdio server" in r.getMessage()]
        assert len(warns) == 2  # fast + slow, once each
        assert any("fast-cmd" in r.getMessage() for r in warns)
        assert any("mcp.json" in r.getMessage() for r in warns)

        # Reconnect must not repeat the warning.
        await manager.reconnect_server("fast")
        warns2 = [r for r in caplog.records if "project-configured stdio server" in r.getMessage()]
        assert len(warns2) == 2
        await manager.disconnect_all()


class TestWindowsProcessTarget:
    """MCP-10: the Win32 spawn target is ONE string (no list2cmdline pass)."""

    def test_single_string_command_line(self) -> None:
        from local_operator.mcp.manager import win32_process_target

        argv = build_cmd_exe_argv("cmd.exe", r"C:\work\%TOKEN%\server.cmd", ['a"b'])
        target = win32_process_target(argv)
        assert isinstance(target, str)
        # The BatBadBut-escaped /c payload survives byte-for-byte.
        assert target.startswith('"cmd.exe" /d /e:ON /v:OFF /c ""C:\\work\\%%cd:~,%')
        assert 'a""b' in target


class TestChildOutputContainment:
    """A stdio child's own output belongs in the log, never on the terminal."""

    @pytest.mark.asyncio
    async def test_quiet_env_reaches_the_child_and_config_env_still_wins(self) -> None:
        """``CHILD_QUIET_ENV`` is delivered, and is a DEFAULT rather than a law.

        The child echoes the variables back on stderr, which is also the only
        proof that the stderr pump is wired: a broken pump means an empty tail
        here rather than a passing assertion on nothing.
        """
        import sys

        from local_operator.mcp.config import MCPStdioServerConfig
        from local_operator.mcp.manager import (
            CHILD_QUIET_ENV,
            McpServerStderr,
            _stdio_transport,
        )

        script = (
            "import os, sys\n"
            "for key in ('NO_COLOR', 'TERM', 'FORCE_COLOR', 'PY_COLORS'):\n"
            "    sys.stderr.write(f'{key}={os.environ.get(key, \"<unset>\")}\\n')\n"
            "sys.stderr.flush()\n"
        )
        cfg = MCPStdioServerConfig(
            command=sys.executable,
            args=["-c", script],
            # The config's own env is merged LAST: a server that needs colour,
            # or a real TERM, must be able to say so.
            env={"TERM": "xterm-256color"},
        )
        stderr_log = McpServerStderr("echo")
        async with _stdio_transport(cfg, lambda: None, stderr_log):
            for _ in range(100):
                if "PY_COLORS" in stderr_log.tail_text():
                    break
                await asyncio.sleep(0.05)

        lines = stderr_log.tail_text().splitlines()
        assert f"NO_COLOR={CHILD_QUIET_ENV['NO_COLOR']}" in lines
        assert f"FORCE_COLOR={CHILD_QUIET_ENV['FORCE_COLOR']}" in lines
        assert f"PY_COLORS={CHILD_QUIET_ENV['PY_COLORS']}" in lines
        assert "TERM=xterm-256color" in lines

    def test_tail_is_bounded_stripped_and_truncated(self) -> None:
        """A chatty or hostile server cannot pin memory or smuggle escapes."""
        from local_operator.mcp.manager import (
            STDERR_LINE_LIMIT,
            STDERR_TAIL_LINES,
            McpServerStderr,
        )

        stderr_log = McpServerStderr("noisy")
        for index in range(STDERR_TAIL_LINES * 3):
            stderr_log.feed(f"line {index}")
        lines = stderr_log.tail_text().splitlines()
        assert len(lines) == STDERR_TAIL_LINES
        assert lines[0] == f"line {STDERR_TAIL_LINES * 2}"  # oldest dropped, newest kept

        stderr_log.feed("\x1b[2J\x1b[1;31mwiped your screen\x1b[0m")
        assert stderr_log.tail_text().endswith("wiped your screen")
        assert "\x1b" not in stderr_log.tail_text()

        stderr_log.feed("x" * (STDERR_LINE_LIMIT * 2))
        assert stderr_log.tail_text().endswith("…[truncated]")
        assert len(stderr_log.tail_text().splitlines()[-1]) < STDERR_LINE_LIMIT + 40

    def test_explain_leaves_a_silent_server_alone(self) -> None:
        """No stderr, no invention: the original error propagates untouched.

        A server that fails without saying anything (``command not found`` never
        reaches a pipe) must not gain a colon and an empty quote.
        """
        from local_operator.mcp.manager import (
            STDERR_QUOTED_CHARS,
            McpConnectionError,
            McpServerStderr,
        )

        original = McpConnectionError("Connection closed")
        stderr_log = McpServerStderr("silent")
        assert stderr_log.explain(original) is original

        stderr_log.feed("fatal: no credentials")
        explained = stderr_log.explain(original)
        assert explained is not original
        assert str(explained) == "Connection closed: fatal: no credentials"

        # A server whose last words are a wall of text becomes a bounded
        # reason: this message is rendered whole in the transcript notice.
        stderr_log.feed("x" * 1000)
        assert len(stderr_log.quoted_tail()) <= STDERR_QUOTED_CHARS + 1
        assert stderr_log.quoted_tail().endswith("…")
