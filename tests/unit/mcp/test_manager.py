"""Manager: fast-startup gate, deferred tools, reconnect breaker, epochs.

Every test stubs the ``_connect_server`` seam — no real MCP server, network,
or SDK transport is required.
"""

from __future__ import annotations

import asyncio
from collections import deque
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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


class TestStartupSettleReporting:
    """The startup outcome is reported after the round SETTLES, not at the gate.

    A server that misses the 250 ms gate (OAuth HTTP servers, which do metadata
    discovery + refresh before connecting) is still connecting when the boot
    snapshot is taken. A single fast failure must not flip the report to
    "N of M up — failed: X" while the slow successes are in flight; the manager
    accumulates the combined tally and fires ``on_startup_settled`` once every
    deferred server has reached a terminal state.
    """

    @pytest.mark.asyncio
    async def test_settling_true_until_deferred_drains_then_callback_fires(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        slow_release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(slow_release.wait(), timeout=10)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        settled: list[bool] = []
        manager.on_startup_settled = lambda: settled.append(True)

        await manager.discover_and_connect()

        # At the gate: fast is live, slow is deferred, so the round is settling.
        assert manager.startup_settling() is True
        assert settled == []
        assert manager.startup_failures() == {}

        # Release the deferred connect; its continuation settles the round.
        slow_release.set()
        await asyncio.sleep(0.05)

        assert manager.startup_settling() is False
        assert settled == [True]  # fired exactly once
        assert manager.startup_failures() == {}
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_fast_failure_does_not_settle_while_slow_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server failing FAST leaves the round settling until the slow one
        lands, and the settled failure map names the fast failure alone."""
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"boom": {"type": "stdio", "command": "boom-cmd"},'
            ' "slow": {"type": "stdio", "command": "slow-cmd"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        slow_release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "boom":
                raise RuntimeError("boom: spawn failed")
            await asyncio.wait_for(slow_release.wait(), timeout=10)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        settled: list[dict[str, str]] = []
        manager.on_startup_settled = lambda: settled.append(manager.startup_failures())

        await manager.discover_and_connect()

        # boom failed at the gate; slow is deferred, so NOT settled yet even
        # though a failure already exists.
        assert manager.startup_settling() is True
        assert settled == []
        assert "boom" in manager.startup_failures()

        slow_release.set()
        await asyncio.sleep(0.05)

        assert manager.startup_settling() is False
        assert len(settled) == 1
        # The settled failure map names the fast failure and not the slow
        # success that eventually landed.
        assert set(settled[0]) == {"boom"}
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_all_fast_means_not_settling_and_no_callback(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every server settles inside the gate the boot snapshot is final:
        nothing was deferred, so the settle callback never fires."""
        manager = McpManager(str(project))

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        settled: list[bool] = []
        manager.on_startup_settled = lambda: settled.append(True)

        await manager.discover_and_connect()

        assert manager.startup_settling() is False
        assert settled == []  # never armed
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_deferred_failure_recorded_in_settled_map(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server that fails AFTER the gate contributes its failure to the
        settled tally — the exact case a gate-only snapshot never saw."""
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd"},'
            ' "slow": {"type": "stdio", "command": "slow-cmd"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        slow_release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(slow_release.wait(), timeout=10)
                raise RuntimeError("slow exploded after the gate")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        settled: list[dict[str, str]] = []
        manager.on_startup_settled = lambda: settled.append(manager.startup_failures())

        await manager.discover_and_connect()
        assert manager.startup_settling() is True

        slow_release.set()
        await asyncio.sleep(0.05)

        assert manager.startup_settling() is False
        assert len(settled) == 1
        assert "slow" in settled[0]
        await manager.disconnect_all()


class TestStartupSettleStaleRound:
    """A continuation that fails AFTER its round was superseded must not write
    the new round's startup accounting or fire its settle callback (F2)."""

    @pytest.mark.asyncio
    async def test_stale_failed_continuation_does_not_touch_the_current_round(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd"},'
            ' "slow": {"type": "stdio", "command": "slow-cmd"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        slow_release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            if name == "slow":
                await asyncio.wait_for(slow_release.wait(), timeout=10)
                raise RuntimeError("slow failed after the gate")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        settled: list[bool] = []
        manager.on_startup_settled = lambda: settled.append(True)

        await manager.discover_and_connect()
        assert manager.startup_settling() is True

        # Supersede the round the way reload()/dispose() would, WITHOUT going
        # through _connect_round (which would legitimately reset the accumulators
        # anyway): bump the epoch so the in-flight continuation is now stale.
        manager._epoch += 1

        # Now let the stale continuation fail. It must not record into the
        # (freshly bumped) round's accumulator, nor fire the settle callback.
        slow_release.set()
        await asyncio.sleep(0.05)

        assert settled == []
        assert manager.startup_failures() == {}
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
        incidents: list[tuple[str, str]] = []
        manager.on_incident = lambda server, reason: incidents.append((server, reason))
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
        assert incidents == [
            (
                "fast",
                "auto-reconnect suspended after >5 attempts in 30s; its tools are "
                "unavailable until a reconnect succeeds",
            )
        ]
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


class TestTeardownLatency:
    """Quit-path teardown: concurrent across servers, bounded per connection.

    The user-visible cost of these properties is the pause between the app
    releasing the terminal and the resume hint printing: ``disconnect_all``
    runs inside ``Session.dispose`` on every quit, so a serial or unbounded
    teardown is experienced as "quit hangs".
    """

    @pytest.mark.asyncio
    async def test_disconnect_all_closes_connections_concurrently(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Total teardown time is the slowest close, not the sum of closes.

        Serial teardown made quit latency grow with the server count
        (measured: 1.6 s across seven real servers where the slowest single
        one needed 0.95 s). Two closes of 0.2 s each must therefore finish
        in ~0.2 s, not ~0.4 s — asserted through a concurrency high-water
        mark rather than wall time, so a slow CI box cannot flake this.
        """
        manager = McpManager(str(project))
        in_flight = 0
        peak = 0

        class SlowStack:
            async def aclose(self) -> None:
                nonlocal in_flight, peak
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.05)
                in_flight -= 1

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            conn = _make_conn(name, cfg)
            conn.stack = cast(Any, SlowStack())
            return conn

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        assert len(manager._connections) == 2
        await manager.disconnect_all()
        assert peak == 2, "closes ran serially; teardown latency sums per server"
        assert manager._connections == {}

    @pytest.mark.asyncio
    async def test_teardown_connection_bounds_a_wedged_close(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close that never returns is cancelled at the teardown bound.

        A remote transport's close is network I/O (streamable-HTTP DELETEs
        its session on a client whose connect timeout alone is 30 s), so a
        dead network could otherwise hold quit hostage. The bound is patched
        down for the test; what is asserted is that the cancel is delivered
        and dispose proceeds.
        """
        import local_operator.mcp.manager as manager_mod

        monkeypatch.setattr(manager_mod, "CONNECTION_TEARDOWN_TIMEOUT_S", 0.05)
        manager = McpManager(str(project))
        cancelled = asyncio.Event()

        class WedgedStack:
            async def aclose(self) -> None:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        async def fake_connect(name: str, cfg: Any) -> ServerConnection:
            conn = _make_conn(name, cfg)
            conn.stack = cast(Any, WedgedStack())
            return conn

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        await manager.discover_and_connect()
        await asyncio.wait_for(manager.disconnect_all(), timeout=2.0)
        assert cancelled.is_set(), "the wedged close was never cancelled"
        assert manager._connections == {}


async def _pid_from_stderr(stderr_log: Any) -> int:
    """The pid a transport child reported on its own stderr.

    The tests below assert about a SPECIFIC process, not about anything that
    merely matches a ``pgrep`` pattern — a pattern match can hit an unrelated
    process on a shared machine in both directions (false leak, and a false
    pass of the ``== ""`` assertion). The child prints ``pid=<n>`` to stderr,
    the transport's stderr pump captures it, and this reads it back.
    """
    import re

    for _ in range(100):
        match = re.search(r"pid=(\d+)", stderr_log.tail_text())
        if match:
            return int(match.group(1))
        await asyncio.sleep(0.05)
    raise AssertionError("child never reported its pid on stderr")


def _process_is_alive(pid: int) -> bool:
    """True when ``pid`` names a RUNNING process; a zombie counts as dead.

    ``os.kill(pid, 0)`` cannot be the probe: a child killed but not yet
    waited on is a zombie, which still answers signal 0 — the kill-on-cancel
    path deliberately does not wait (see ``_stop``), so the reaped-vs-leaked
    question must accept ``Z`` as dead.
    """
    import subprocess as sp

    out = sp.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True
    ).stdout.strip()
    return bool(out) and not out.startswith("Z")


class TestStdioStopIsEventDriven:
    """The stdio ``_stop`` waits on process exit, not on a polling tick.

    The 0.1 s polling loop it replaced charged up to a full tick of latency
    per server on every quit — pure wait after the child had already exited.
    """

    @pytest.mark.asyncio
    async def test_prompt_child_costs_no_polling_tick(self) -> None:
        """A child that exits on stdin EOF is reaped in well under 0.1 s.

        Wall-clock bound on purpose: the property under test IS latency, and
        the old implementation fails this by construction (its first
        returncode check happens at the 0.1 s tick). The margin (0.09 s vs
        the old floor of ~0.1 s) is small, so the child does nothing but
        exit; a busier assertion would flake instead of measure.
        """
        import sys
        import time

        from local_operator.mcp.config import MCPStdioServerConfig
        from local_operator.mcp.manager import McpServerStderr, _stdio_transport

        # Exits the moment stdin reaches EOF — the polite-quit handshake.
        script = "import sys\nsys.stdin.read()\n"
        cfg = MCPStdioServerConfig(command=sys.executable, args=["-c", script])
        stderr_log = McpServerStderr("prompt")
        async with _stdio_transport(cfg, lambda: None, stderr_log):
            # Give the child a beat to reach its read() before teardown.
            await asyncio.sleep(0.3)
            t0 = time.monotonic()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.09, f"teardown took {elapsed:.3f}s; polling tick is back"

    @pytest.mark.asyncio
    async def test_stubborn_child_is_killed_on_cancellation(self) -> None:
        """Cancelling a bounded teardown must not leak the child process.

        ``_teardown_connection`` cancels the stack close at its bound; the
        cancel lands inside ``_stop``'s waits, and absorbing it without a
        kill would leave the server running past the session.

        TWO cancels, deliberately. The first is consumed at the parked
        ``sleep(3600)`` — after it, the transport's ``finally`` runs ``_stop``
        UNcancelled, and the ordinary kill rung would reap the child even if
        the ``except CancelledError`` handler were deleted (review round 1,
        F1: the single-cancel version of this test passed with the handler
        removed). The second cancel is timed to land while ``_stop`` sits in
        its EOF-rung wait, which is the state a bounded dispose actually
        delivers: only the handler reaps the child from there, so this is
        the arrangement that fails when the handler is gone.
        """
        import sys

        import local_operator.mcp.manager as manager_mod
        from local_operator.mcp.config import MCPStdioServerConfig
        from local_operator.mcp.manager import McpServerStderr, _stdio_transport

        # Reports its pid, then ignores stdin EOF and SIGTERM: only SIGKILL
        # removes it, so a surviving pid can only mean the kill never came.
        script = (
            "import os, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "sys.stderr.write(f'pid={os.getpid()}\\n')\n"
            "sys.stderr.flush()\n"
            "while True: time.sleep(0.2)\n"
        )
        cfg = MCPStdioServerConfig(command=sys.executable, args=["-c", script])
        stderr_log = McpServerStderr("stubborn")

        async def run() -> None:
            cm = _stdio_transport(cfg, lambda: None, stderr_log)
            async with cm:
                # Park until cancelled; teardown then runs under cancellation,
                # which is the state a bounded dispose delivers it in.
                await asyncio.sleep(3600)

        original = manager_mod.STDIO_EXIT_GRACE_S
        manager_mod.STDIO_EXIT_GRACE_S = 0.2  # keep the ladder fast in CI
        try:
            task = asyncio.get_running_loop().create_task(run())
            pid = await _pid_from_stderr(stderr_log)
            task.cancel()
            # Let the first cancel unwind the park and start ``_stop``; well
            # under the 0.2 s grace, so the second cancel lands inside the
            # EOF-rung wait rather than after the ladder has finished.
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            manager_mod.STDIO_EXIT_GRACE_S = original
        # The kill-on-cancel path does not wait on the child, so it may be a
        # zombie — which is dead for this purpose. Alive means leaked.
        assert not _process_is_alive(pid), f"stubborn child leaked: pid {pid}"

    @pytest.mark.asyncio
    async def test_sigterm_rung_still_reaps_a_deaf_reader(self) -> None:
        """A child that ignores stdin EOF but honours SIGTERM exits at rung 2.

        Guards the escalation ladder itself: event-driven waits must still
        escalate (EOF grace → terminate → kill) rather than returning early
        on the first rung's timeout. The assertion is on the child's PID, not
        on reaching the end of the context — ``_stop`` runs under
        ``suppress(Exception)`` and every wait in it is bounded, so the
        context exits cleanly even when the ladder is broken (review round 1,
        F2: a ``_stop`` mutated to return after the first rung passed the
        reach-the-end version of this test while leaking the child).
        """
        import sys

        import local_operator.mcp.manager as manager_mod
        from local_operator.mcp.config import MCPStdioServerConfig
        from local_operator.mcp.manager import McpServerStderr, _stdio_transport

        # Reports its pid; never reads stdin; dies on SIGTERM (the default
        # disposition). Ladder rung 1 (EOF grace) therefore expires, and only
        # rung 2's terminate can reap it.
        script = (
            "import os, sys, time\n"
            "sys.stderr.write(f'pid={os.getpid()}\\n')\n"
            "sys.stderr.flush()\n"
            "while True: time.sleep(0.2)\n"
        )
        cfg = MCPStdioServerConfig(command=sys.executable, args=["-c", script])
        stderr_log = McpServerStderr("deaf")
        # Shrink the per-rung grace so the EOF rung times out quickly.
        original = manager_mod.STDIO_EXIT_GRACE_S
        manager_mod.STDIO_EXIT_GRACE_S = 0.2
        try:
            async with _stdio_transport(cfg, lambda: None, stderr_log):
                pid = await _pid_from_stderr(stderr_log)
        finally:
            manager_mod.STDIO_EXIT_GRACE_S = original
        # Rung 2 awaits ``process.wait()`` after terminating, so a reaped
        # child is GONE (not even a zombie); alive means the ladder never
        # escalated past the EOF rung.
        assert not _process_is_alive(pid), f"deaf child leaked: pid {pid}"


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

        An ABSENCE is delivery too. ``FORCE_COLOR`` and ``CLICOLOR_FORCE`` are
        sensed by presence, so the only way to hand a child "off" for them is
        to hand it neither — ``FORCE_COLOR=0`` reads as colour ON. The child
        echoes them so the dict cannot regrow one without this failing.
        """
        import sys

        from local_operator.mcp.config import MCPStdioServerConfig
        from local_operator.mcp.manager import (
            CHILD_QUIET_ENV,
            McpServerStderr,
            _stdio_transport,
        )

        forcing = ("FORCE_COLOR", "CLICOLOR_FORCE")
        echoed = (*CHILD_QUIET_ENV, *forcing)
        script = (
            "import os, sys\n"
            f"for key in {echoed!r}:\n"
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
                if f"{echoed[-1]}=" in stderr_log.tail_text():
                    break
                await asyncio.sleep(0.05)

        lines = stderr_log.tail_text().splitlines()
        for key, value in CHILD_QUIET_ENV.items():
            if key == "TERM":
                continue  # overridden by the config's own env, asserted next
            assert f"{key}={value}" in lines
        assert "TERM=xterm-256color" in lines
        for key in forcing:
            assert f"{key}=<unset>" in lines

    def test_the_quiet_env_actually_silences_rich(self) -> None:
        """The dict is judged on output, not on reading like an opt-out.

        ``FORCE_COLOR=0`` sat in ``CHILD_QUIET_ENV`` looking like one and doing
        the opposite. Rich is the renderer the reported server (``workspace-
        mcp``) draws its logo with and is a first-party dependency here, so the
        claim is measured against it rather than argued. The forced case is the
        control: without it a child that never colours anything would pass the
        first two assertions just as well.
        """
        import subprocess
        import sys

        from mcp.client.stdio import get_default_environment

        from local_operator.mcp.manager import CHILD_QUIET_ENV

        script = (
            "import sys\nfrom rich.console import Console\n"
            "Console(file=sys.stdout).print('[bold red]LOGO[/]')\n"
        )

        def render(**overrides: str) -> bytes:
            env = get_default_environment() | CHILD_QUIET_ENV | overrides
            done = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                env=env,
                timeout=60,
                check=False,
            )
            assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
            return done.stdout

        assert render() == b"LOGO\n"
        # A server whose config restores a real TERM is still not a terminal:
        # stdout is our pipe, and we have not claimed otherwise.
        assert render(TERM="xterm-256color") == b"LOGO\n"
        # And the regression this dict used to carry: presence, any value.
        assert b"\x1b[" in render(TERM="xterm-256color", FORCE_COLOR="0")

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


def test_per_tool_filter_allow_deny_and_deny_wins(project: Path) -> None:
    from local_operator.mcp.config import MCPStdioServerConfig

    manager = McpManager(str(project))
    manager._configs["srv"] = MCPStdioServerConfig(
        command="x",
        enabledTools=["search_*", "get_one"],
        disabledTools=["search_private", "get_one"],
    )
    assert manager._tool_is_enabled("srv", "search_public") is True
    assert manager._tool_is_enabled("srv", "search_private") is False
    assert manager._tool_is_enabled("srv", "get_one") is False  # deny wins
    assert manager._tool_is_enabled("srv", "unlisted") is False
    assert manager._tool_is_enabled("missing", "anything") is True


class TestAuthRequiredHandling:
    """An expired OAuth grant surfaces as an actionable failure, never a popup.

    Startup and auto-reconnect are non-interactive: when the stored grant
    cannot be refreshed, the connect raises ``McpAuthRequiredError``. The
    manager turns that into a ``run /mcp login <name>`` message for the toast,
    and the reconnect loop abandons (an expired grant will not heal by
    retrying) instead of burning the breaker window.
    """

    def test_unwrap_auth_required_recognises_plain_and_grouped(self) -> None:
        """The transport wraps the handler's raise in an ExceptionGroup; the
        manager must recognise the auth error in BOTH delivery shapes."""
        from local_operator.mcp.auth import McpAuthRequiredError
        from local_operator.mcp.manager import _unwrap_auth_required

        plain = McpAuthRequiredError("https://srv.example/mcp")
        assert _unwrap_auth_required(plain) is plain

        grouped = ExceptionGroup("unhandled errors in a TaskGroup (1 sub-exception)", [plain])
        unwrapped = _unwrap_auth_required(grouped)
        assert isinstance(unwrapped, McpAuthRequiredError)
        assert unwrapped.server_url == "https://srv.example/mcp"

        # Non-auth exceptions pass through untouched.
        other = RuntimeError("boom")
        assert _unwrap_auth_required(other) is other
        other_group = ExceptionGroup("g", [other])
        assert _unwrap_auth_required(other_group) is other_group

    def test_unwrap_auth_required_walks_nested_groups(self) -> None:
        """The transport's anyio group can sit INSIDE the session's group, so
        the auth error arrives double-wrapped; ``subgroup`` preserves that
        nesting, and a depth-1 read returns the inner GROUP, not the leaf."""
        from local_operator.mcp.auth import McpAuthRequiredError
        from local_operator.mcp.manager import _unwrap_auth_required

        leaf = McpAuthRequiredError("https://srv.example/mcp")
        nested = ExceptionGroup("outer", [ExceptionGroup("inner", [leaf])])
        unwrapped = _unwrap_auth_required(nested)
        assert isinstance(unwrapped, McpAuthRequiredError)
        assert unwrapped.server_url == "https://srv.example/mcp"

        # Triple depth, with sibling noise, still resolves to the leaf.
        deep = ExceptionGroup(
            "outermost",
            [
                ExceptionGroup("mid", [ExceptionGroup("in", [leaf])]),
            ],
        )
        assert isinstance(_unwrap_auth_required(deep), McpAuthRequiredError)

    def test_fire_auth_required_calls_the_ui_sink(self) -> None:
        """The UI hook receives the server name and the actionable message."""
        from local_operator.mcp.auth import McpAuthRequiredError

        manager = McpManager("/tmp")
        seen: list[tuple[str, str]] = []
        manager.on_auth_required = lambda name, msg: seen.append((name, msg))
        manager._fire_auth_required("notion", McpAuthRequiredError("https://mcp.notion.com/mcp"))
        assert seen == [("notion", "needs authorization — run /mcp login notion")]

    def test_fire_auth_required_survives_a_raising_sink(self) -> None:
        """A broken UI hook must not take down the connect machinery."""
        from local_operator.mcp.auth import McpAuthRequiredError

        manager = McpManager("/tmp")

        def broken(name: str, msg: str) -> None:
            raise RuntimeError("ui exploded")

        manager.on_auth_required = broken
        # Must not raise.
        manager._fire_auth_required("notion", McpAuthRequiredError("https://mcp.notion.com/mcp"))

    def test_fire_auth_required_is_deduped_until_reconnect(self) -> None:
        """A dead grant a tool call keeps retrying must not re-toast every time;
        a successful reconnect clears the latch so a later expiry toasts again."""
        from local_operator.mcp.auth import McpAuthRequiredError

        manager = McpManager("/tmp")
        seen: list[str] = []
        manager.on_auth_required = lambda name, msg: seen.append(name)
        exc = McpAuthRequiredError("https://mcp.notion.com/mcp")

        manager._fire_auth_required("notion", exc)
        manager._fire_auth_required("notion", exc)
        manager._fire_auth_required("notion", exc)
        assert seen == ["notion"]  # only the first fires

        # A successful (re)connect clears the latch.
        manager._auth_toasted.discard("notion")
        manager._fire_auth_required("notion", exc)
        assert seen == ["notion", "notion"]

    @pytest.mark.asyncio
    async def test_auth_required_text_names_the_login_command(self) -> None:
        from local_operator.mcp.auth import McpAuthRequiredError

        manager = McpManager("/tmp")
        exc = McpAuthRequiredError("https://srv.example/mcp")
        text = manager._auth_required_text("datadog", exc)
        assert "datadog" in text
        assert "/mcp login datadog" in text

    @pytest.mark.asyncio
    async def test_connect_round_reports_auth_required_actionably(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A server that needs auth lands in ``errors`` with the login command."""
        from local_operator.mcp.auth import McpAuthRequiredError
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth"),
        )

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            raise McpAuthRequiredError("https://srv.example/mcp")

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        result = await manager._connect_round({"dd": cfg}, {"dd": "global"})
        assert "dd" in result.errors
        assert "/mcp login dd" in result.errors["dd"]
        assert result.connected_servers == []

    @pytest.mark.asyncio
    async def test_reconnect_abandons_on_auth_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An auth failure during reconnect abandons rather than re-scheduling.

        An expired grant will not heal by retrying, so further attempts would
        only burn the breaker window. The manager abandons auto-reconnect and
        leaves ``/mcp login`` as the recovery path.
        """
        from local_operator.mcp.auth import McpAuthRequiredError
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(
            url="https://srv.example/mcp",
            auth=MCPAuthConfig(type="oauth"),
        )
        manager._configs["dd"] = cfg

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            raise McpAuthRequiredError("https://srv.example/mcp")

        monkeypatch.setattr(manager, "_connect_server", fake_connect)

        scheduled = {"called": False}

        def fake_schedule(name: str) -> None:
            scheduled["called"] = True

        monkeypatch.setattr(manager, "_schedule_reconnect", fake_schedule)

        # Run one reconnect attempt with zero delay.
        await manager._reconnect("dd", 0.0, manager._epoch)

        # The reconnect must NOT have re-scheduled itself (abandoned instead).
        assert scheduled["called"] is False
        # And the server must be marked as having abandoned auto-reconnect.
        assert manager.reconnect_suspended("dd") is True

    @pytest.mark.asyncio
    async def test_an_abandoned_grant_survives_the_real_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The named cancel receipt must reach the caller through the SDK.

        Regression for the review finding that settled this design: a named
        exception raised out of ``callback_handler`` does NOT survive the
        streamable-HTTP transport — the SDK's ``post_writer`` swallows it and
        the caller gets an opaque ``CancelledError``. So the flow records the
        abandonment in the ledger and raises a raw cancellation; this test
        drives a full grant through the REAL ``streamable_http_client``
        against a stub OAuth server and asserts the manager converts what
        comes back into ``McpLoginCancelledError``.
        """
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from local_operator.mcp.auth import McpLoginCancelledError
        from local_operator.mcp.config import (
            MCPAuthConfig,
            MCPHttpServerConfig,
            MCPOAuthConfig,
        )

        class StubOAuthServer(BaseHTTPRequestHandler):
            """Just enough of an OAuth AS + MCP endpoint to reach the grant."""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return  # silence the access log

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path.startswith("/.well-known/oauth-authorization-server"):
                    assert isinstance(self.server, ThreadingHTTPServer)
                    base = f"http://127.0.0.1:{self.server.server_port}"
                    self._json(
                        {
                            "issuer": base,
                            "authorization_endpoint": f"{base}/authorize",
                            "token_endpoint": f"{base}/token",
                            "registration_endpoint": f"{base}/register",
                        }
                    )
                else:
                    self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if self.path == "/register":
                    self._json(
                        {
                            "client_id": "stub-client",
                            "client_secret": "stub-secret",
                            "client_id_issued_at": 0,
                        },
                        status=201,
                    )
                elif self.path == "/token":
                    # No stored grant and the refresh path is not what is under
                    # test: refuse, so the SDK escalates to the browser grant.
                    self._json({"error": "invalid_grant"}, status=400)
                elif self.path.startswith("/mcp"):
                    # Unreachable: the grant abandons before the MCP session
                    # starts. Answer 401 anyway so a regression that SKIPS the
                    # grant fails here loudly rather than hanging.
                    self._json({"error": "unauthorized"}, status=401)
                else:
                    self._json({"error": "not found"}, status=404)

        server = ThreadingHTTPServer(("127.0.0.1", 0), StubOAuthServer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/mcp"
            # An ephemeral loopback callback: the flow binds whatever is free,
            # never the shared default port (parallel suites each run grants).
            import socket

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                redirect_port = int(probe.getsockname()[1])
            cfg = MCPHttpServerConfig(
                url=url,
                auth=MCPAuthConfig(type="oauth"),
                oauth=MCPOAuthConfig(redirect_uri=f"http://127.0.0.1:{redirect_port}/callback"),
            )
            # An in-memory credential store: the grant must run against the
            # real auth flow, never this machine's real auth.db.
            from tests.unit.mcp.test_auth import FakeAuthStore

            manager = McpManager(str(tmp_path))
            manager.auth_store = cast(Any, FakeAuthStore())
            monkeypatch.setattr("webbrowser.open", lambda _url: False)
            # The idle guard is what fires when the browser never answers;
            # shrink it so the grant abandons in test time.
            monkeypatch.setattr("local_operator.mcp.auth.INTERACTIVE_GRANT_TIMEOUT_S", 0.3)
            with pytest.raises(McpLoginCancelledError, match="browser never completed"):
                await manager._connect_server("dd", cfg, interactive=True)
        finally:
            server.shutdown()
            server.server_close()

    @pytest.mark.asyncio
    async def test_login_resets_the_breaker_and_scopes_the_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``/mcp login`` must (a) clear the auth suspension so the
        server's NEXT disconnect auto-reconnects again, and (b) keep the widened
        login timeout out of the persisted config — otherwise every later tool
        call on the server inherits a 10-minute request budget."""
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"dd": {"type": "http", "url": "https://srv.example/mcp",'
            ' "auth": {"type": "oauth"}}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        # Simulate the state an auth-abandoned reconnect leaves behind.
        manager._reconnect_suspended.add("dd")
        manager._reconnect_history["dd"] = deque([0.0])
        manager._backoff_index["dd"] = 3

        seen_timeout: list[float | None] = []

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            seen_timeout.append(cfg.timeout)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        conn = await manager.connect_configured_server("dd", timeout_ms=600_000)

        # (a) breaker state cleared — auto-reconnect lives again.
        assert manager.reconnect_suspended("dd") is False
        assert "dd" not in manager._reconnect_history
        assert "dd" not in manager._backoff_index
        # (b) the CONNECT saw the widened timeout…
        assert seen_timeout == [600_000]
        # …but neither the persisted config nor the live connection kept it.
        stored = manager.get_server_config("dd")
        assert stored is not None and stored.timeout is None
        assert conn.config.timeout is None
        await manager.disconnect_all()
