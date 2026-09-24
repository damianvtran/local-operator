"""Manager: fast-startup gate, deferred tools, reconnect breaker, epochs.

No test needs a real MCP server, network, or SDK transport. Most tests stub the
``_connect_server`` seam; the auth-block tests deliberately do not — they
replace only ``_open_transport_and_session`` so the real ``_connect_server``
body and its attempt seam run (agent review round 2, major-1).
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from local_operator.harness.types import AbortSignal, ToolContext, ToolResult
from local_operator.mcp.auth import (
    REFRESH_REFUSAL_ENDPOINT,
    REFRESH_REFUSAL_INFLIGHT,
    REFRESH_REFUSAL_LOCK,
    REFRESH_REFUSAL_UNATTRIBUTED,
    REFRESH_REFUSAL_UNREACHABLE,
    REFRESH_REFUSAL_UNSENT,
)
from local_operator.mcp.config import MCPStdioServerConfig
from local_operator.mcp.manager import (
    DEFAULT_MCP_TIMEOUT_MS,
    McpManager,
    ServerConnection,
    build_cmd_exe_argv,
    build_stdio_argv,
    resolve_mcp_timeout_s,
    stdio_start_new_session,
)
from local_operator.mcp.tool_cache import McpToolCache, config_digest
from local_operator.session.protocol import RuntimeLocality


def _stdio_digest(command: str) -> str:
    return config_digest(MCPStdioServerConfig(command=command))


def _tool(name: str, schema: dict[str, Any] | None = None) -> Tool:
    """One SDK ``Tool`` as a server would advertise it."""
    return Tool(
        name=name,
        description=f"{name} desc",
        input_schema=schema or {"type": "object", "properties": {"q": {"type": "string"}}},
    )


class FakeSession:
    """ClientSession stand-in: records calls, returns canned results."""

    # Runtime role (SessionProtocol). This fake stands in for an OWNER:
    # it carries no attached runtime, which is what the absent legacy
    # `is_remote` meant.
    owns_runtime = True
    outcome_is_synchronous = True
    runtime_locality: RuntimeLocality = "this-process"

    async def variables_op(
        self, action: str, key: str = "", value: str = "", value_type: str = ""
    ) -> dict[str, Any]:
        """The REAL verb table against this fake's (empty) kernel registry.

        ``SessionProtocol`` declares code memory for every session shape and the
        desktop route reaches it BY NAME through the bridge's facade, so a double
        without it does not type as a session at all — the drift the declaration
        exists to catch rather than a test-only nuisance.

        The fake owns no interpreter, so the table answers exactly what a real
        session whose runtime has never run a cell answers: observed/absent for a
        read, ``no_kernel`` for a write. Delegating rather than hand-writing that
        envelope keeps ONE copy of the frozen shape in the tree, so the double
        cannot certify a branch the real session does not have.
        """
        from local_operator.session.variable_ops import run_variable_verb

        return await run_variable_verb(
            f"fake-{id(self):x}",
            action,
            key,
            value,
            value_type,
            redact=getattr(getattr(self, "variables", None), "redact", None),
        )

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
            _stdio_digest("slow-cmd"),
        )
        manager = McpManager(str(project), tool_cache=cache)

        slow_release = asyncio.Event()
        sessions: dict[str, FakeSession] = {}

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
        cache.put(
            "slow",
            [{"name": "search", "description": "", "inputSchema": {}}],
            _stdio_digest("slow-cmd"),
        )
        manager = McpManager(str(project), tool_cache=cache)
        release = asyncio.Event()

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def failing_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
    async def test_call_finalizes_with_context_off_event_loop(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        calls: list[tuple[Any, ...]] = []

        async def capture_to_thread(function: Any, *args: Any) -> ToolResult:
            calls.append((function, *args))
            return function(*args)

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        monkeypatch.setattr(asyncio, "to_thread", capture_to_thread)
        await manager.discover_and_connect()

        tool = next(t for t in manager.get_tools() if t.name == "mcp__fast_search")
        context = ToolContext(session_id="mcp-context")
        result = await tool.execute("c1", {"q": "x"}, None, None, context)
        assert result.text == "ok"
        assert len(calls) == 1
        assert calls[0][-1] is context
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_non_retriable_error_returns_error_result_without_reconnect(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = McpManager(str(project))
        connects = {"n": 0}

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
            _stdio_digest("fast-cmd"),
        )
        manager = McpManager(str(project), tool_cache=cache)
        state = {"fail": False}
        real_sleep = asyncio.sleep

        async def instant_sleep(delay: float) -> None:
            await real_sleep(0)

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        async def flaky_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
        cache.put(
            "slow",
            [{"name": "search", "description": "", "inputSchema": {}}],
            _stdio_digest("slow-cmd"),
        )
        manager = McpManager(str(project), tool_cache=cache)
        release = asyncio.Event()

        async def slow_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
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
    async def test_stubborn_child_is_killed_on_cancellation(self, monkeypatch) -> None:
        """Cancelling a bounded teardown must not leak the child process.

        ``_teardown_connection`` cancels the stack close at its bound; the
        cancel lands inside ``_stop``'s waits, and absorbing it without a
        kill would leave the server running past the session.

        TWO cancels, deliberately. The first is consumed at the parked
        application wait — after it, the transport's ``finally`` runs ``_stop``
        UNcancelled, and the ordinary kill rung would reap the child even if
        the ``except CancelledError`` handler were deleted (review round 1,
        F1: the single-cancel version of this test passed with the handler
        removed). A proxy over the REAL process publishes entry into the
        EOF-rung wait; the second cancel is sent at that event. Assert the
        handler sent kill BEFORE the test waits for actual process exit.
        Sending SIGKILL is synchronous; the OS finishing it is not, so an
        immediate ps probe was a race even when the kill correctly happened.
        """
        import sys

        import anyio

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
        ready = asyncio.Event()
        stop_waiting = asyncio.Event()
        kill_calls = []
        processes: list[Any] = []
        original_feed = stderr_log.feed

        def feed(line: str) -> None:
            original_feed(line)
            if line.startswith("pid="):
                ready.set()

        monkeypatch.setattr(stderr_log, "feed", feed)
        original_open = anyio.open_process

        class ObservedProcess:
            def __init__(self, process: Any) -> None:
                self.process = process

            def __getattr__(self, name: str) -> Any:
                return getattr(self.process, name)

            async def wait(self) -> int:
                stop_waiting.set()
                return await self.process.wait()

            def kill(self) -> None:
                kill_calls.append(self.process.pid)
                self.process.kill()

        async def open_process(*args: Any, **kwargs: Any) -> Any:
            process = await original_open(*args, **kwargs)
            processes.append(process)
            return ObservedProcess(process)

        monkeypatch.setattr(anyio, "open_process", open_process)

        async def run() -> None:
            cm = _stdio_transport(cfg, lambda: None, stderr_log)
            async with cm:
                # Park until cancelled; teardown then runs under cancellation,
                # which is the state a bounded dispose delivers it in.
                await asyncio.Event().wait()

        # No grace timeout should win this test: only the observed cancellation
        # may reach kill. The deadline is a hang backstop, never a timing target.
        monkeypatch.setattr(manager_mod, "STDIO_EXIT_GRACE_S", 60.0)
        task = asyncio.get_running_loop().create_task(run())
        try:
            await asyncio.wait_for(ready.wait(), 10)
            process = processes[0]
            pid = process.pid
            task.cancel()
            await asyncio.wait_for(stop_waiting.wait(), 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 10)
            assert kill_calls == [pid], "cancelling _stop did not kill the stubborn child"
            await asyncio.wait_for(process.wait(), 10)
            assert not _process_is_alive(pid), f"stubborn child leaked: pid {pid}"
        finally:
            # A deliberately broken kill handler must fail without leaking the
            # actual process used to prove it. Bypass the spy for test cleanup.
            for process in processes:
                if process.returncode is None:
                    process.kill()
                await asyncio.wait_for(process.wait(), 10)
                await process.aclose()
            if not task.done():
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)

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
    manager turns that into a ``/mcp login <name>`` message for the toast,
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
        assert seen == [("notion", "/mcp login notion to authorize")]

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
        leaves ``/mcp login`` — or a peer session's, via
        ``revalidate_auth_blocked`` — as the recovery path.

        The abandonment is recorded in ``_auth_blocked``, NOT in the flap
        breaker: the two conditions recover differently (a breaker on time, an
        auth block on the shared grant changing), and this assertion used to
        read ``reconnect_suspended`` only because they shared one set. Keeping
        that predicate breaker-only is what lets the block be lifted by a
        store change without also resurrecting a flapping server.
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
        # And the server must be marked as having abandoned auto-reconnect over
        # AUTHORIZATION, which the flap breaker deliberately does not claim.
        assert manager.auth_blocked("dd") is True
        assert manager.reconnect_suspended("dd") is False
        assert manager.get_connection_status("dd") == "auth-required"

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
    async def test_transient_refresh_failure_does_not_block_on_auth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contention is NOT an auth failure, and the manager must retry it.

        The decisive property of the coordination contract, measured through the
        REAL ``streamable_http_client``: when the in-flight coordinator cannot
        refresh under the lock, it refuses to let the SDK POST the in-memory
        refresh token unlocked. That refusal does not survive the transport as
        an exception — the transport runs the request inside anyio cancel
        scopes, so it arrives as a bare
        ``CancelledError('Cancelled via cancel scope …')`` — so the provider arms
        ``REFRESH_CONTENTION`` and ``_connect_server`` re-voices the
        cancellation from that record.

        Two outcomes are asserted, and they are the difference between "a
        healthy server on a login prompt" and "a healthy server that retries":
        the reconnect is re-scheduled with backoff and the server does NOT take
        an auth block. ``token_posts`` is the same claim measured at the wire —
        ZERO POSTs to the token endpoint, locked or unlocked.
        """
        import json
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import (
            McpAuthRequiredError,
            McpRefreshContendedError,
            McpTokenStorage,
        )
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig
        from tests.unit.mcp.test_auth import FakeAuthStore

        token_posts: list[dict[str, Any]] = []

        class StubServer(BaseHTTPRequestHandler):
            """Just enough OAuth metadata and MCP endpoint to reach the flow."""

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            @property
            def _base(self) -> str:
                assert isinstance(self.server, ThreadingHTTPServer)
                return f"http://127.0.0.1:{self.server.server_port}"

            def do_GET(self) -> None:
                if self.path.startswith("/.well-known/oauth-protected-resource"):
                    self._json(
                        {"resource": self._base + "/mcp", "authorization_servers": [self._base]}
                    )
                elif self.path.startswith("/.well-known/oauth-authorization-server"):
                    self._json(
                        {
                            "issuer": self._base,
                            "authorization_endpoint": self._base + "/authorize",
                            "token_endpoint": self._base + "/token",
                            "registration_endpoint": self._base + "/register",
                        }
                    )
                else:
                    self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                if self.path == "/token":
                    # Every POST here — locked or unlocked — is a failure of the
                    # claim under test, so record it and answer dead.
                    token_posts.append(
                        dict(
                            __import__("urllib.parse", fromlist=["parse_qsl"]).parse_qsl(
                                body.decode()
                            )
                        )
                    )
                    self._json({"error": "invalid_grant"}, status=400)
                else:
                    self._json({"error": "unauthorized"}, status=401)

        server = ThreadingHTTPServer(("127.0.0.1", 0), StubServer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/mcp"
            cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
            store = FakeAuthStore()
            manager = McpManager(str(tmp_path))
            manager.auth_store = cast(Any, store)
            manager._configs["dd"] = cfg

            # An EXPIRED, refreshable stored grant: exactly the state the
            # coordinator exists to act on. Without an expiry the coordinator
            # would return early and nothing would be proven.
            from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

            storage = McpTokenStorage(url, store)
            await storage.set_client_info(OAuthClientInformationFull(client_id="stub-client"))
            await storage.set_tokens(
                OAuthToken(access_token="stale", refresh_token="r-old", expires_in=60)
            )
            store.rows[0].data["tokens_obtained_at"] = time.time() - 600

            # A peer process holds the refresh lock: the bounded acquire gives
            # up, which is the contention this contract is about. Patched on the
            # module the provider looks the name up on, so the REAL provider
            # runs — only the lock's outcome is forced.
            import contextlib as _contextlib

            @_contextlib.asynccontextmanager
            async def _foreign_held_lock(server_url: str):
                yield False

            monkeypatch.setattr(auth_mod, "_oauth_refresh_lock", _foreign_held_lock)

            def _leaves(exc: BaseException) -> list[BaseException]:
                if isinstance(exc, BaseExceptionGroup):
                    out: list[BaseException] = []
                    for child in exc.exceptions:
                        out.extend(_leaves(child))
                    return out
                return [exc]

            with pytest.raises(BaseException) as excinfo:
                await manager._connect_server("dd", cfg)
            leaves = _leaves(excinfo.value)
            assert any(isinstance(leaf, McpRefreshContendedError) for leaf in leaves), (
                "the refused unlocked refresh was not re-voiced from the ledger: "
                f"{excinfo.value!r}"
            )
            assert not any(
                isinstance(leaf, McpAuthRequiredError) for leaf in leaves
            ), "contention was flattened into an auth requirement"
            # And it cost ZERO token POSTs: the SDK's unlocked refresh never ran.
            assert token_posts == [], f"a refresh POST reached the wire: {token_posts}"

            # The manager arm: a generic failure re-schedules with backoff and
            # does NOT block on auth (the whole point of a non-auth raise).
            scheduled = {"called": False}
            monkeypatch.setattr(
                manager, "_schedule_reconnect", lambda name: scheduled.__setitem__("called", True)
            )
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert scheduled["called"] is True
            assert manager.auth_blocked("dd") is False
            assert manager.reconnect_suspended("dd") is False
            assert manager.get_connection_status("dd") != "auth-required"
        finally:
            server.shutdown()
            server.server_close()

    @pytest.mark.asyncio
    async def test_a_genuine_dispose_cancellation_is_not_re_voiced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real teardown must stay a teardown, even with a record armed.

        The guard that keeps the re-voice safe: a cancellation the TASK ITSELF
        was asked for (``cancelling() > 0`` — a dispose, a reload, an epoch
        change, the user leaving) keeps its priority, and the contention record
        is discarded rather than believed. Without this, a refusal armed a moment
        before a disposal would convert the disposal into a reconnect attempt —
        the retry storm the abandoned-grant arm already guards against.

        Also asserts the record is CONSUMED on that path, so it cannot be
        attributed to an unrelated cancellation of the same server later.
        """
        from local_operator.mcp.auth import REFRESH_CONTENTION
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        url = "https://srv.example/mcp"
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
        manager._configs["dd"] = cfg

        parked = asyncio.Event()

        async def park(*_a: Any, **_kw: Any) -> Any:
            await parked.wait()
            raise AssertionError("released without being cancelled")

        monkeypatch.setattr(manager, "_open_transport_and_session", park)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        scheduled = {"called": False}
        monkeypatch.setattr(
            manager, "_schedule_reconnect", lambda name: scheduled.__setitem__("called", True)
        )

        # Fresh ledger entries are cleared around every test by the fixture
        # below; arm one here to prove the guard beats the record.
        REFRESH_CONTENTION.record(url)
        task = asyncio.ensure_future(manager._connect_server("dd", cfg))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert scheduled["called"] is False
        assert manager.auth_blocked("dd") is False
        assert REFRESH_CONTENTION.pop(url) is None, "the record leaked past the disposal"

    @pytest.mark.asyncio
    async def test_the_contention_record_is_single_use(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One refusal re-voices ONE cancellation, never two.

        The transport's rewrite of an auth-flow failure is a bare CancelledError
        with no cancelling count, which is exactly what the abandoned-grant arm
        sees too — so the record is what distinguishes them, and a record that
        outlived its own cancellation would turn the NEXT unrelated one into a
        retry.
        """
        from local_operator.mcp.auth import REFRESH_CONTENTION, McpRefreshContendedError
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        url = "https://srv.example/mcp"
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
        manager._configs["dd"] = cfg

        async def bare_cancel(*_a: Any, **_kw: Any) -> Any:
            # What the transport delivers: a CancelledError with NO cancelling
            # count on the task, i.e. not an external cancellation.
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", bare_cancel)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        REFRESH_CONTENTION.record(url)
        with pytest.raises(McpRefreshContendedError):
            await manager._connect_server("dd", cfg)
        # Second attempt, same server, nothing armed: the record really was
        # single-use — the cancellation is NOT re-voiced as a retry. What it IS
        # re-voiced as changed with the settle fix: a bare, unarmed cancellation
        # with no cancelling count is anyio's own delivery (the transport's
        # reader/writer dying), so ``_connect_server`` now converts it to
        # ``McpTransportError`` so it is REPORTED instead of dropped by
        # ``_finish_pending``. The assertion that matters here is the negative
        # one: it is not a contention retry.
        from local_operator.mcp.manager import McpTransportError

        with pytest.raises(McpTransportError) as second_attempt:
            await manager._connect_server("dd", cfg)
        assert not isinstance(second_attempt.value, McpRefreshContendedError)
        assert second_attempt.value.url == url

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


class TestAuthChallengeMessaging:
    """A 401/403 must name the command that fixes it (issue #367 follow-up).

    The SDK erases the status code: a 401 it cannot resolve arrives from
    ``session.initialize()`` as ``MCPError(-32603, 'Server returned an error
    response')`` — which is exactly what the user saw on the splash. The
    watcher observes the response before that happens.
    """

    def _exc(self, *, oauth_available: bool = True, has_stored_grant: bool = False) -> Any:
        from local_operator.mcp.auth import McpAuthChallengeError

        return McpAuthChallengeError(
            "https://srv.example/mcp",
            status_code=401,
            oauth_available=oauth_available,
            has_stored_grant=has_stored_grant,
        )

    def test_a_challenge_with_oauth_names_the_login_command(self) -> None:
        text = McpManager._auth_failure_text("gitlab", self._exc())
        assert "/mcp login gitlab" in text

    def test_no_discoverable_oauth_does_not_promise_a_login(self) -> None:
        """Datadog's shape: a 401 with no WWW-Authenticate. Claiming OAuth is
        available when discovery found none sends the user at a dead command."""
        text = McpManager._auth_failure_text("datadog", self._exc(oauth_available=False))
        assert "/mcp login" not in text
        assert "401" in text

    def test_a_stale_stored_grant_says_reauth_not_login(self) -> None:
        """The operator's ask: reaching this error means the stored grant could
        not be refreshed, so a server we hold a row for is holding a dead one.
        ``login`` would leave that credential in place."""
        from local_operator.mcp.auth import McpAuthRequiredError

        exc = McpAuthRequiredError("https://srv.example/mcp")
        with mock.patch("local_operator.mcp.auth.server_has_stored_grant", return_value=True):
            text = McpManager._auth_failure_text("gitlab", exc)
        assert "/mcp reauth gitlab" in text
        assert "/mcp login" not in text

    def test_never_authorized_says_login_not_reauth(self) -> None:
        from local_operator.mcp.auth import McpAuthRequiredError

        exc = McpAuthRequiredError("https://srv.example/mcp")
        with mock.patch("local_operator.mcp.auth.server_has_stored_grant", return_value=False):
            text = McpManager._auth_failure_text("gitlab", exc)
        assert "/mcp login gitlab" in text

    def test_the_actionable_command_survives_a_narrow_terminal(self) -> None:
        """Design constraint D1: the toast clamps to the card width and the
        TAIL is what truncates, so the command has to lead."""
        text = McpManager._auth_failure_text("launchdarkly", self._exc())
        assert text.startswith("/mcp login launchdarkly")
        assert text.count("—") <= 1  # D4: not a chain of dashes

    @pytest.mark.asyncio
    async def test_a_401_on_the_mcp_endpoint_is_observed(self) -> None:
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher("https://srv.example/mcp")
        await watcher.observe(
            SimpleNamespace(
                status_code=401,
                request=SimpleNamespace(url="https://srv.example/mcp"),
            )
        )
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_a_401_from_a_metadata_probe_is_not_the_server_refusing_us(
        self,
    ) -> None:
        """The same client carries the provider's discovery requests; a 401
        from one of those is part of discovery, not the connect being denied."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher("https://srv.example/mcp")
        await watcher.observe(
            SimpleNamespace(
                status_code=401,
                request=SimpleNamespace(
                    url="https://srv.example/.well-known/oauth-protected-resource"
                ),
            )
        )
        assert watcher.status_code is None

    @pytest.mark.asyncio
    async def test_a_non_auth_failure_is_never_relabelled_as_an_auth_problem(
        self, tmp_path: Path
    ) -> None:
        """Routing a network outage into "/mcp login" would be a worse
        error than the opaque one this change replaces. No observed challenge
        means no conversion."""
        from local_operator.mcp.config import MCPHttpServerConfig
        from local_operator.mcp.manager import _AuthChallengeWatcher

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")
        watcher = _AuthChallengeWatcher("https://srv.example/mcp")  # nothing observed
        assert await manager._challenge_error(cfg, watcher) is None
        assert await manager._challenge_error(cfg, None) is None

    @pytest.mark.asyncio
    async def test_connect_round_reports_a_challenge_actionably(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through the surface the user screenshotted: the splash
        notice reads from ``startup_failures``."""
        from local_operator.mcp.config import MCPHttpServerConfig

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")

        async def fake_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            raise self._exc()

        monkeypatch.setattr(manager, "_connect_server", fake_connect)
        result = await manager._connect_round({"gitlab": cfg}, {"gitlab": "codex"})
        assert "/mcp login gitlab" in result.errors["gitlab"]
        assert "Server returned an error response" not in result.errors["gitlab"]


class TestAuthChallengeWatcherAttribution:
    """Which response the watcher attributes to this connect (F1, F2)."""

    URL = "https://srv.example/mcp"

    def _response(self, status: int, url: str) -> Any:
        return SimpleNamespace(status_code=status, request=SimpleNamespace(url=url))

    @pytest.mark.asyncio
    async def test_a_challenge_after_a_canonicalising_redirect_is_observed(self) -> None:
        """F1. A server may 307 ``/mcp`` to ``/mcp/`` and only then challenge.
        The client follows the redirect, so the final response's request URL is
        not the configured string; comparing raw text dropped the challenge and
        the user kept getting the SDK's opaque error."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.observe(self._response(307, self.URL))
        await watcher.observe(self._response(401, "https://srv.example/mcp/"))
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_case_and_query_differences_still_match_the_endpoint(self) -> None:
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.observe(self._response(401, "https://SRV.example/mcp?session=abc"))
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_a_later_non_auth_response_supersedes_an_earlier_challenge(self) -> None:
        """F2. The watcher used to latch the first 401 forever, so an attempt
        that was challenged and then failed on a retry with a 500 was reported
        as an authorization problem — exactly the misrouting of a non-auth
        failure this feature promises not to do."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.observe(self._response(401, self.URL))
        assert watcher.status_code == 401
        await watcher.observe(self._response(500, self.URL))
        assert watcher.status_code is None

    @pytest.mark.asyncio
    async def test_a_redirect_hop_never_clears_a_pending_verdict(self) -> None:
        """A 3xx is the client being sent elsewhere, not the server's verdict."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.observe(self._response(401, self.URL))
        await watcher.observe(self._response(307, self.URL))
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_a_metadata_probe_neither_sets_nor_clears(self) -> None:
        """Discovery rides the same client; its responses are not verdicts on
        the connect, in either direction."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.observe(self._response(401, self.URL))
        await watcher.observe(
            self._response(200, "https://srv.example/.well-known/oauth-protected-resource")
        )
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_the_stored_grant_fact_comes_from_the_classifying_store(
        self, tmp_path: Path
    ) -> None:
        """F4. The challenge already resolved this against the manager's own
        (possibly injected) store; re-reading the default machine store here
        answered a different question and rendered ``login`` for a server
        demonstrably holding a grant."""
        from local_operator.mcp.auth import McpAuthChallengeError

        exc = McpAuthChallengeError(
            "https://srv.example/mcp",
            status_code=401,
            oauth_available=True,
            has_stored_grant=True,
        )
        # The default store is empty here, so a second lookup would say
        # "login". The carried fact must win.
        with mock.patch(
            "local_operator.mcp.auth.server_has_stored_grant", return_value=False
        ) as lookup:
            text = McpManager._auth_failure_text("gitlab", exc)
        assert "/mcp reauth gitlab" in text
        assert lookup.call_count == 0

    @pytest.mark.asyncio
    async def test_the_challenge_error_carries_the_managers_store_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manager's injected store is what classifies the failure."""
        from local_operator.mcp.config import MCPHttpServerConfig
        from local_operator.mcp.manager import _AuthChallengeWatcher

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")
        watcher = _AuthChallengeWatcher("https://srv.example/mcp")
        watcher.status_code = 401

        async def fake_discover(url: str) -> object:
            return object()

        monkeypatch.setattr("local_operator.mcp.auth.discover_oauth_endpoints", fake_discover)
        monkeypatch.setattr(
            "local_operator.mcp.auth.server_has_stored_grant", lambda url, store=None: True
        )
        exc = await manager._challenge_error(cfg, watcher)
        assert exc is not None and exc.has_stored_grant is True
        assert "/mcp reauth" in McpManager._auth_failure_text("gitlab", exc)


class TestChallengeIsBoundToTheTerminalRequest:
    """F5: a challenge must not outlive the request that produced it.

    Clearing only when a later RESPONSE arrives left the verdict latched when
    the next request died at DNS/connect/TLS/read time — no response hook runs
    then — so a terminal network failure was reported as an auth problem.
    """

    URL = "https://srv.example/mcp"

    def _response(self, status: int, url: str) -> Any:
        return SimpleNamespace(status_code=status, request=SimpleNamespace(url=url))

    @pytest.mark.asyncio
    async def test_a_request_that_never_answers_leaves_no_verdict(self) -> None:
        """The exact F5 state: 401, then a retry that fails with no response."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401, self.URL))
        assert watcher.status_code == 401
        # The retry starts and then dies at the socket: begin() runs, observe()
        # never does.
        await watcher.begin(SimpleNamespace(url=self.URL))
        assert watcher.status_code is None

    @pytest.mark.asyncio
    async def test_a_network_failure_after_a_challenge_is_not_auth_advice(
        self, tmp_path: Path
    ) -> None:
        """End to end through the classifier: no verdict means no /mcp login."""
        from local_operator.mcp.config import MCPHttpServerConfig
        from local_operator.mcp.manager import _AuthChallengeWatcher

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=self.URL)
        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401, self.URL))
        await watcher.begin(SimpleNamespace(url=self.URL))  # retry, then a socket error
        assert await manager._challenge_error(cfg, watcher) is None

    @pytest.mark.asyncio
    async def test_a_request_to_another_url_never_clears_the_verdict(self) -> None:
        """Discovery and token requests ride the same client; a probe starting
        must not erase the endpoint's own challenge."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401, self.URL))
        await watcher.begin(
            SimpleNamespace(url="https://srv.example/.well-known/oauth-protected-resource")
        )
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_a_challenge_still_survives_to_classification(self) -> None:
        """The ordinary path must keep working: begin() then a 401 classifies."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401, self.URL))
        assert watcher.status_code == 401

    @pytest.mark.asyncio
    async def test_the_eligibility_gate_uses_the_managers_effective_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F6. With discovery offline, only the injected store can supply the
        evidence — a store-less probe refuses a server this manager would
        itself classify as needing reauth."""
        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.config import MCPHttpServerConfig

        auth_mod.OAUTH_CHALLENGES.clear()

        async def discovery_offline(url: str) -> None:
            return None

        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", discovery_offline)

        from tests.unit.mcp.test_auth import FakeAuthStore

        store = FakeAuthStore()
        url = "https://srv.example/mcp"
        auth_mod.McpTokenStorage(url, store)._write({"tokens": {"access_token": "a"}})

        manager = McpManager(str(tmp_path))
        manager.auth_store = cast(Any, store)
        cfg = MCPHttpServerConfig(url=url)

        # The store-less call both login paths used to make.
        assert await auth_mod.probe_oauth_capability(cfg) is False
        # The manager-owned operation consults its own store and allows it.
        assert await manager.server_supports_oauth_login(cfg) is True


class TestTheStartupNetworkSubsetFollowsEveryClear:
    """R1-2: the network subset is cleared wherever the failure itself is.

    ``_startup_network`` is documented as a subset of ``_startup_failures`` "by
    construction", and one site wrote ``_startup_failures`` directly instead of
    going through ``_clear_startup_failure`` — so a server that healed kept its
    name in the network set, and ``network_failures <= set(failures)`` (the
    invariant ``mcp_status.all_failures_are_network`` reads) was False for any
    caller that trusts it. Latent rather than user-visible today: both wiring
    reads filter by ``name in failures``.
    """

    @pytest.mark.asyncio
    async def test_a_healed_server_leaves_neither_set(self, tmp_path: Path) -> None:
        from local_operator.mcp.config import MCPHttpServerConfig

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url="https://srv.example/mcp")
        manager._configs["remote"] = cfg
        manager._note_startup_failure("remote", "network: cannot reach srv.example", network=True)
        assert manager.startup_network_failures() == {"remote"}

        # A real loop: ``_register_connection`` arms the connection watchdog.
        manager._register_connection(_make_conn("remote", cfg))

        assert manager.startup_failures() == {}
        assert manager.startup_network_failures() == set()
        assert manager.startup_network_failures() <= set(manager.startup_failures())
        await manager.disconnect_all()


class TestAnObservedChallengeOutranksTheTransportLabel:
    """Q1-1: a peer that ANSWERED must not be reported as the network failing.

    QA measured a reachable, answering 401 peer rendering as ``network: no
    response from <host> (timed out)`` in the majority of 29 runs while the
    stub's own request log showed the initialize POST had received its 401: the
    SDK's initialize died inside its own task group, the transport gave up as a
    bare cancellation, and the watcher's ``begin`` hook had already cleared the
    challenge the retry never answered. The contract this change carries ("a
    reachable 401 must not be called a network failure") therefore held only in
    the minority of runs.

    The transport-failure arm is the only caller that reads the LATENT
    observation (:attr:`_AuthChallengeWatcher.saw_challenge`); every other
    classifier keeps F5's last-request rule, which the test below pins on the
    same watcher state.
    """

    URL = "https://srv.example/mcp"

    def _response(self, status: int) -> Any:
        return SimpleNamespace(status_code=status, request=SimpleNamespace(url=self.URL))

    async def _challenged_then_dark(self) -> Any:
        """A watcher whose peer answered 401 and whose retry never answered."""
        from local_operator.mcp.manager import _AuthChallengeWatcher

        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401))  # the initialize POST answered
        await watcher.begin(SimpleNamespace(url=self.URL))  # the retry starts ...
        return watcher

    def _oauth_capable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the eligibility gate and discovery answer without the network."""
        from local_operator.mcp import auth as auth_mod

        auth_mod.OAUTH_CHALLENGES.clear()
        auth_mod.record_oauth_challenge(self.URL, oauth_available=True)

        async def discovery(url: str) -> object:
            return object()

        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", discovery)
        monkeypatch.setattr(auth_mod, "server_has_stored_grant", lambda url, store=None: False)

    @pytest.mark.asyncio
    async def test_the_latent_observation_is_only_read_when_asked_for_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default stays F5's; the transport arm opts in explicitly."""
        from local_operator.mcp.config import MCPHttpServerConfig

        self._oauth_capable(monkeypatch)
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=self.URL)
        watcher = await self._challenged_then_dark()
        assert watcher.status_code is None, "F5: the retry that never answered clears it"
        assert watcher.saw_challenge == 401

        assert await manager._challenge_error(cfg, watcher) is None
        exc = await manager._challenge_error(cfg, watcher, prefer_observed=True)
        assert exc is not None
        assert exc.status_code == 401
        assert exc.oauth_available is True

    @pytest.mark.asyncio
    async def test_the_connect_records_the_auth_failure_not_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end: the race QA measured, asserted on the recorded outcome.

        The transport seam is stubbed with the deterministic sequence the real
        SDK produces (answer 401, retry, die with no second response), so the
        assertion is about the CLASSIFICATION and needs no socket.
        """
        from local_operator.mcp.config import MCPHttpServerConfig
        from local_operator.mcp.manager import NETWORK_FAILURE_MARKER

        self._oauth_capable(monkeypatch)
        monkeypatch.setattr("local_operator.mcp.manager.STARTUP_GATE_MS", 1)

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=self.URL)
        settled = asyncio.Event()
        manager.on_startup_settled = settled.set

        async def answering_then_dying(
            stack: Any,
            name: str,
            cfg_: Any,
            timeout_s: float | None,
            stderr_log: Any,
            *,
            interactive: bool = False,
            challenge_watcher: Any = None,
        ) -> ServerConnection:
            await challenge_watcher.begin(SimpleNamespace(url=self.URL))
            await challenge_watcher.observe(self._response(401))
            await challenge_watcher.begin(SimpleNamespace(url=self.URL))
            # Past the gate, so the failure lands on the deferred path.
            await asyncio.sleep(0.05)
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", answering_then_dying)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        await manager._connect_round({"remote": cfg}, {})
        await asyncio.wait_for(settled.wait(), timeout=10)

        failures = manager.startup_failures()
        assert set(failures) == {"remote"}, failures
        assert failures["remote"] == "/mcp login remote to authorize"
        assert NETWORK_FAILURE_MARKER not in failures["remote"]
        assert manager.startup_network_failures() == set()
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_a_challenge_the_attempt_satisfied_stops_winning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R2-1: 401 → 200 → transport death must NOT read as an auth failure.

        The latch is bounded by the peer's own answers: an endpoint response that
        is not a challenge proves this endpoint answers and is satisfied (a 200
        after a grant is the case that matters), so the earlier 401 must not
        outlive it. Reproduced on the first cut as ``/mcp login remote to
        authorize`` with an EMPTY network subset — and, worse, with a durable
        OAuth-challenge record written for a grant the attempt had just proved
        good.
        """
        from local_operator.mcp.config import MCPHttpServerConfig
        from local_operator.mcp.manager import (
            NETWORK_FAILURE_MARKER,
            _AuthChallengeWatcher,
        )

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=self.URL)
        watcher = _AuthChallengeWatcher(self.URL)
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(401))  # the peer challenges us ...
        assert watcher.saw_challenge == 401
        await watcher.begin(SimpleNamespace(url=self.URL))
        await watcher.observe(self._response(200))  # ... and the retry is SATISFIED
        assert watcher.status_code is None
        assert watcher.saw_challenge is None, "a proven-good answer clears the latch"

        # No challenge, so nothing to classify: this is the path that also
        # short-circuits BEFORE the durable ``record_oauth_challenge`` write.
        assert await manager._challenge_error(cfg, watcher, prefer_observed=True) is None

        # And on the real connect path, the same sequence ends in the network
        # label — with the server counted in the network subset, which is what
        # "network ⊆ failures by construction" needs to stay meaningful.
        monkeypatch.setattr("local_operator.mcp.manager.STARTUP_GATE_MS", 1)
        settled = asyncio.Event()
        manager.on_startup_settled = settled.set

        async def satisfied_then_dying(
            stack: Any,
            name: str,
            cfg_: Any,
            timeout_s: float | None,
            stderr_log: Any,
            *,
            interactive: bool = False,
            challenge_watcher: Any = None,
        ) -> ServerConnection:
            await challenge_watcher.begin(SimpleNamespace(url=self.URL))
            await challenge_watcher.observe(self._response(401))
            await challenge_watcher.begin(SimpleNamespace(url=self.URL))
            await challenge_watcher.observe(self._response(200))
            await asyncio.sleep(0.05)  # past the gate: the deferred failure path
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", satisfied_then_dying)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        await manager._connect_round({"remote": cfg}, {})
        await asyncio.wait_for(settled.wait(), timeout=10)

        failures = manager.startup_failures()
        assert set(failures) == {"remote"}, failures
        assert failures["remote"].startswith(NETWORK_FAILURE_MARKER), failures
        assert manager.startup_network_failures() == {"remote"}
        await manager.disconnect_all()


class TestMcpAuthRecoveryHint:
    """The remedy an MCP auth failure earns, and the one it must never get.

    An expired MCP grant and an expired MODEL PROVIDER key are different
    credentials with different fixes, and the provider-side hint was the only
    one wired up — so an MCP failure either got nothing or, worse, would have
    been told to `/login anthropic`, re-authorizing a provider that was never
    broken.
    """

    def test_it_matches_the_error_shape_not_a_prefix(self) -> None:
        """The prefix gate is why NO MCP error got a hint on either path.

        ``append_auth_recovery`` keys off the rendered string STARTING WITH
        "authentication failed", the form ``ProviderError.__str__`` produces.
        An MCP failure never comes through that renderer and reaches the
        transcript already wrapped ("MCP error: …"), so the auth fact is not at
        position zero and a prefix test can never see it.
        """
        from local_operator.mcp.manager import mcp_auth_recovery_hint
        from local_operator.providers.failover import append_auth_recovery

        wrapped = "MCP error: MCP OAuth authorization required for https://mcp.linear.app/mcp"
        assert append_auth_recovery(wrapped, "anthropic") == wrapped, "the prefix gate"
        assert mcp_auth_recovery_hint(wrapped, "linear") is not None

    def test_it_carries_the_remedy_it_is_given_and_invents_none(self) -> None:
        """The VERB is the manager's to decide, never this function's.

        An earlier revision hard-coded ``/mcp reauth`` here, which is right for
        exactly one of the three real auth shapes (review R5). This seam sees
        only a rendered string, so it cannot tell a never-logged-in server from
        an expired grant — the fix is that it no longer tries.
        """
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        hint = mcp_auth_recovery_hint(
            "MCP authorization failed", "/mcp login minerva-qa to authorize"
        )
        assert hint is not None
        assert "/mcp login minerva-qa" in hint
        assert "reauth" not in hint, "the verb must come from the manager, not from here"
        assert "/login " not in hint, "the model provider's credential is not the broken one"

    def test_an_unnamed_server_still_gets_a_reachable_remedy(self) -> None:
        """A failure naming no configured server must not guess a VERB either.

        ``/mcp`` is the one instruction true for every shape: it lists each
        server with the failure recorded against it. Naming ``reauth`` here
        instead sent a never-logged-in user into a command that finds no row to
        replace and returns without logging in (review R5).
        """
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        hint = mcp_auth_recovery_hint(
            "MCP server at https://x.example/mcp refused the connection (401)"
        )
        assert hint is not None
        assert "`/mcp`" in hint
        assert "reauth" not in hint

    def test_non_auth_failures_get_no_hint(self) -> None:
        """A hint that fires on everything is noise, and a timeout has no remedy."""
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        assert mcp_auth_recovery_hint("MCP error: server 'linear' is not connected") is None
        assert mcp_auth_recovery_hint("rate limit or quota exceeded (429)") is None
        assert mcp_auth_recovery_hint("") is None

    def test_an_ambiguous_message_refuses_to_name_one(self) -> None:
        """Naming the WRONG server is the failure this remediation is about."""
        from local_operator.mcp.manager import mcp_server_name_in

        servers = ["linear", "minerva-qa"]
        both = "MCP authorization failed for 'linear' and 'minerva-qa'"
        assert mcp_server_name_in(both, servers) is None
        assert mcp_server_name_in("MCP authorization failed for 'linear'", servers) == "linear"

    def test_a_name_inside_a_url_path_is_not_a_match(self) -> None:
        """R6: the loose pass matched anywhere, including inside a URL segment.

        A server called ``git`` was named out of ``/git-things`` and reported as
        the UNAMBIGUOUS answer, which is the "we named the wrong server" failure
        this helper exists to avoid. Short generic names (``git``, ``api``,
        ``db``) are the plausible ones.
        """
        from local_operator.mcp.manager import mcp_server_name_in

        servers = ["linear", "minerva-qa", "notion", "git"]
        assert (
            mcp_server_name_in(
                "MCP OAuth authorization required for https://example.com/git-things", servers
            )
            is None
        )
        # Still matches when the message is genuinely ABOUT it, and a hyphenated
        # name still matches its own hyphen (a `\b` anchor would break this).
        assert mcp_server_name_in("MCP authorization failed for 'git'", servers) == "git"
        assert (
            mcp_server_name_in("MCP authorization failed for minerva-qa", servers) == "minerva-qa"
        )

    def test_a_name_inside_the_hostname_IS_a_match(self) -> None:
        """R9/U9: the URL-shaped message is the ONLY one the field produces.

        The R6 boundary put ``.`` in the non-name class, which reads as
        harmless — names may contain a dot — but a dot is exactly what a
        hostname puts on both sides of the name. That made the loose pass, whose
        documented job is "the fallback for the URL-shaped messages", match
        NOTHING: ``linear``, ``notion``, ``sentry`` and ``minerva-qa`` all went
        from resolved to ``None``, and every actionable ``/mcp reauth
        linear`` silently became the generic referral.

        Both existing layers missed it because the quoted form asserted above is
        emitted NOWHERE in the product for this path and the app-level stub
        bypasses the matcher, so this drives the REAL exceptions' own
        ``__str__`` — a hand-written string is what let the gap open.
        """
        from local_operator.mcp.auth import McpAuthChallengeError, McpAuthRequiredError
        from local_operator.mcp.manager import mcp_server_name_in

        servers = ["linear", "minerva-qa", "notion", "sentry", "git"]
        hosts = {
            "linear": "https://mcp.linear.app/sse",
            "notion": "https://mcp.notion.com/mcp",
            "sentry": "https://mcp.sentry.dev/mcp",
            "minerva-qa": "https://minerva-qa.gominerva.com/mcp",
        }
        for name, url in hosts.items():
            challenge = McpAuthChallengeError(
                url, status_code=401, oauth_available=True, has_stored_grant=True
            )
            assert mcp_server_name_in(str(challenge), servers) == name
            assert mcp_server_name_in(str(McpAuthRequiredError(url)), servers) == name

        # And the R6 case stays closed, because what blocks it is the `-`, not
        # the `.` — including when the decoy sits in the hostname rather than
        # the path, which is where dropping `.` could plausibly have re-opened it.
        assert (
            mcp_server_name_in(
                str(McpAuthRequiredError("https://git-things.example.com/mcp")), servers
            )
            is None
        )


class TestTheManagerDerivesTheAuthRemedy:
    """R5: which command fixes a server is decided from STATE, not guessed.

    The round-2 review found the hint hard-coding ``/mcp reauth`` for every
    shape, correct for exactly one of the three real ones. The five tests above
    missed it because every one passed a bare STRING; these drive real configs
    and a real credential store, which is where the distinction lives.

    ``_auth_failure_text`` is the dispatcher that already answers this for the
    startup toast, the transcript incident and ``/mcp``, so routing through it
    is also what stops a fourth surface drifting from those three.
    """

    def _manager(self, tmp_path: Path) -> Any:
        from local_operator.mcp.config import MCPHttpServerConfig, MCPStdioServerConfig
        from local_operator.mcp.manager import McpManager

        manager = McpManager.__new__(McpManager)
        manager._configs = {
            "linear": MCPHttpServerConfig(url="https://mcp.linear.app/mcp"),
            "datadog": MCPHttpServerConfig(url="https://mcp.datadoghq.com/api/mcp"),
            "filesys": MCPStdioServerConfig(command="npx"),
        }
        # An EMPTY store, so "has a stored grant" is answered by this test's own
        # fixture rather than by whatever the developer happens to hold.
        manager._effective_auth_store = lambda: None  # type: ignore[method-assign]
        return manager

    def test_a_server_never_logged_into_is_sent_to_login_not_reauth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dead end R5 named: ``reauth`` finds no row and never logs in.

        ``_mcp_logout`` returns False for a server with no stored credential and
        ``_mcp_command`` then RETURNS without starting a grant, so the user runs
        the command, nothing happens, and the failure survives it.
        """
        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        monkeypatch.setattr(auth_mod, "server_has_stored_grant", lambda url, store=None: False)
        auth_mod.OAUTH_CHALLENGES["https://mcp.linear.app/mcp"] = True

        manager = self._manager(tmp_path)
        error = "MCP error: MCP server 'linear' refused the connection (401)"
        hint = mcp_auth_recovery_hint(error, manager.auth_recovery_hint(error))
        assert hint is not None
        assert "/mcp login linear" in hint
        assert "reauth" not in hint

    def test_an_expired_grant_is_sent_to_reauth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The one shape the hard-coded verb got right; it must stay right."""
        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        monkeypatch.setattr(auth_mod, "server_has_stored_grant", lambda url, store=None: True)
        auth_mod.OAUTH_CHALLENGES["https://mcp.linear.app/mcp"] = True

        manager = self._manager(tmp_path)
        error = "MCP error: MCP server 'linear' refused the connection (401)"
        hint = mcp_auth_recovery_hint(error, manager.auth_recovery_hint(error))
        assert hint is not None and "/mcp reauth linear" in hint

    def test_a_server_with_no_oauth_endpoint_is_never_promised_a_login(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Datadog shape: 401 with NO ``WWW-Authenticate`` at all.

        No grant command can work here by construction, which ``auth.py`` already
        documents; the honest remedy is the config's own headers.
        """
        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        monkeypatch.setattr(auth_mod, "server_has_stored_grant", lambda url, store=None: False)
        auth_mod.OAUTH_CHALLENGES["https://mcp.datadoghq.com/api/mcp"] = False

        manager = self._manager(tmp_path)
        error = "MCP error: MCP server 'datadog' refused the connection (401)"
        hint = mcp_auth_recovery_hint(error, manager.auth_recovery_hint(error))
        assert hint is not None
        assert "login" not in hint and "reauth" not in hint
        assert "API key or headers" in hint

    def test_a_stdio_server_is_sent_to_its_config(self, tmp_path: Path) -> None:
        """A stdio server has no transport that can carry a bearer token."""
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        manager = self._manager(tmp_path)
        error = "MCP error: MCP server 'filesys' rejected our credentials (401)"
        hint = mcp_auth_recovery_hint(error, manager.auth_recovery_hint(error))
        assert hint is not None
        assert "login" not in hint and "reauth" not in hint
        assert "MCP config" in hint

    def test_the_remedy_matches_what_every_other_surface_says(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The anti-drift claim, asserted rather than argued.

        A FOURTH wording for the same fact is how the startup toast and the
        transcript hint start disagreeing about what the user should run.
        """
        from local_operator.mcp import auth as auth_mod
        from local_operator.mcp.auth import McpAuthChallengeError
        from local_operator.mcp.manager import McpManager

        monkeypatch.setattr(auth_mod, "server_has_stored_grant", lambda url, store=None: False)
        auth_mod.OAUTH_CHALLENGES["https://mcp.linear.app/mcp"] = True

        manager = self._manager(tmp_path)
        typed = McpAuthChallengeError(
            "https://mcp.linear.app/mcp",
            status_code=401,
            oauth_available=True,
            has_stored_grant=False,
        )
        toast_text = McpManager._auth_failure_text("linear", typed)
        derived = manager.auth_recovery_hint(
            "MCP error: MCP server 'linear' refused the connection (401)"
        )
        assert derived == toast_text

    def test_an_unknown_server_degrades_to_the_unnamed_hint(self, tmp_path: Path) -> None:
        """No config for it means no state to read, so no verb may be claimed."""
        from local_operator.mcp.manager import mcp_auth_recovery_hint

        manager = self._manager(tmp_path)
        error = "MCP error: MCP server at https://who.example/mcp refused the connection (401)"
        assert manager.auth_recovery_hint(error) is None
        hint = mcp_auth_recovery_hint(error, None)
        assert hint is not None and "`/mcp`" in hint


class TestMcpRecoveryNotice:
    """The RECOVERY half of the model-visible MCP pair.

    The failure half has always reached the model (``on_incident`` ->
    ``Session._on_mcp_incident`` -> a ``session_mcp_unavailable`` WARNING row).
    The recovery half did not, so an operator who ran ``/mcp login <server>``
    mid-session left the model holding a death notice — and its "do not call
    its tools" advice — for a server that had been usable for the rest of the
    session. Observed live against ``minerva-qa``.

    Two properties are load-bearing and each has its own tests below:

    * a recovery fires ONLY for a server whose failure the MODEL was told
      about, which is why the gate is armed inside the ``if sink is not None``
      branches at the three ``on_incident`` sites and not derived from
      ``_auth_toasted`` / ``_reconnect_suspended`` / ``_startup_failures``; and
    * ``reload()`` re-registers EVERY connection, so an ungated notice would
      be a storm of "is connected again" about servers that never broke.
    """

    @staticmethod
    def _sinks(manager: McpManager) -> tuple[list[tuple[str, str]], list[tuple[str, int]]]:
        """Install both sinks and return their recording lists."""
        incidents: list[tuple[str, str]] = []
        recoveries: list[tuple[str, int]] = []
        manager.on_incident = lambda server, reason: incidents.append((server, reason))
        manager.on_recovery = lambda server, count: recoveries.append((server, count))
        return incidents, recoveries

    @staticmethod
    async def _trip_breaker(
        manager: McpManager, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive ``name``'s reconnect chain until the breaker trips.

        Mirrors ``TestCircuitBreaker``: backoff sleeps are made instant so the
        five attempts inside the 30 s window happen in a few loop turns.
        """
        real_sleep = asyncio.sleep

        async def instant_sleep(delay: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", instant_sleep)

        async def failing_connect(server: str, cfg: Any, **_: Any) -> ServerConnection:
            raise RuntimeError("still down")

        monkeypatch.setattr(manager, "_connect_server", failing_connect)
        manager._schedule_reconnect(name)
        for _ in range(60):
            await real_sleep(0)
            if manager.reconnect_suspended(name):
                break
        assert manager.reconnect_suspended(name) is True

    @pytest.mark.asyncio
    async def test_breaker_incident_then_reconnect_fires_recovery(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Route 4: the reconnect the incident text itself promises.

        The breaker incident says the tools are "unavailable until a reconnect
        succeeds"; when one does, the model must be told so.
        """
        manager = McpManager(str(project))
        incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        assert recoveries == []  # a healthy boot announces nothing

        await self._trip_breaker(manager, "fast", monkeypatch)
        assert [server for server, _ in incidents] == ["fast"]

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        conn = await manager.reconnect_server("fast")
        assert conn is not None
        assert recoveries == [("fast", len(manager.get_server_tools("fast")))]
        assert recoveries[0][1] == 1
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_after_gate_auth_failure_then_login_fires_recovery(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The operator's exact scenario, end to end.

        An HTTP server's grant expires, its connect misses the 250 ms startup
        gate and fails with ``McpAuthRequiredError`` (the after-gate
        continuation fires the incident), then ``/mcp login`` reconnects it via
        ``connect_configured_server``. Exactly one recovery, naming the server.
        """
        from local_operator.mcp.auth import McpAuthRequiredError

        manager = McpManager(str(project))
        incidents, recoveries = self._sinks(manager)
        gate_passed = asyncio.Event()

        # ``interactive`` is accepted because ``connect_configured_server``
        # (the /mcp login path) passes it; the other routes do not.
        async def slow_auth_failure(name: str, cfg: Any, **_kw: Any) -> ServerConnection:
            if name == "fast":
                return _make_conn(name, cfg)
            # Miss the gate, then fail with the auth error, so the failure runs
            # through _finish_pending's continuation rather than the gate arm.
            await gate_passed.wait()
            raise McpAuthRequiredError("https://srv.example/mcp")

        monkeypatch.setattr(manager, "_connect_server", slow_auth_failure)
        result = await manager.discover_and_connect()
        assert "slow" not in result.errors  # still in flight at the gate
        gate_passed.set()
        for _ in range(100):
            await asyncio.sleep(0)
            if incidents:
                break
        assert [server for server, _ in incidents] == ["slow"]
        # The sink payload is the REMEDY alone, command-first (D3): the row's own
        # head already says the server is unavailable, so an "MCP authorization
        # failed;" prefix only pushed the command off the front of the line.
        assert incidents[0][1] == "/mcp login slow to authorize", incidents
        assert recoveries == []

        async def good_connect(name: str, cfg: Any, **_kw: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        conn = await manager.connect_configured_server("slow")
        assert conn is not None
        assert recoveries == [("slow", 1)]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_recovery_reports_registered_not_raw_tool_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The count must survive ``enabledTools``/``disabledTools`` filtering.

        ``_register_tools`` drops filtered tools, so ``len(conn.tools)`` — what
        the TUI's own login receipt prints — overstates what the model can
        actually call. The notice must not promise tools that are not in the
        inventory.
        """
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd",'
            ' "disabledTools": ["hidden"]}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            conn = _make_conn(name, cfg)
            conn.tools = [_tool("search"), _tool("hidden")]
            return conn

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        conn = manager.get_connection("fast")
        assert conn is not None and len(conn.tools) == 2
        assert len(manager.get_server_tools("fast")) == 1

        await self._trip_breaker(manager, "fast", monkeypatch)
        monkeypatch.setattr(manager, "_connect_server", good_connect)
        assert await manager.reconnect_server("fast") is not None
        # 1 (registered), NOT 2 (raw): the filtered tool is not callable.
        assert recoveries == [("fast", 1)]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_recovery_sink_raising_does_not_break_connect(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising sink is contained AND disarms the server.

        The disarm happens before the call precisely so a broken session sink
        cannot leave a server re-announcing on every later reconnect.
        """
        manager = McpManager(str(project))
        calls: list[str] = []

        def exploding(server: str, count: int) -> None:
            calls.append(server)
            raise RuntimeError("sink exploded")

        manager.on_incident = lambda server, reason: None
        manager.on_recovery = exploding

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        await self._trip_breaker(manager, "fast", monkeypatch)
        monkeypatch.setattr(manager, "_connect_server", good_connect)

        conn = await manager.reconnect_server("fast")
        assert conn is not None  # the connection still registered
        assert manager.get_connection("fast") is not None
        assert calls == ["fast"]

        # Disarmed: a second clean reconnect says nothing more.
        assert await manager.reconnect_server("fast") is not None
        assert calls == ["fast"]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_no_recovery_for_a_server_that_never_failed(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The core negative: healthy servers are never announced."""
        manager = McpManager(str(project))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        assert recoveries == []
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_reload_emits_no_recovery_storm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE storm test. ``reload()`` re-registers every connection.

        Every server passes through ``_register_connection`` on a reload, so an
        ungated notice would tell the model that three servers which never
        broke are "connected again". This is the case the gate exists for.
        """
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"a": {"type": "stdio", "command": "a-cmd"},'
            ' "b": {"type": "stdio", "command": "b-cmd"},'
            ' "c": {"type": "stdio", "command": "c-cmd"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        assert len(manager.get_tools()) == 3

        await manager.reload()
        assert len(manager.get_tools()) == 3  # all three genuinely re-registered
        assert recoveries == []
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_reload_with_one_armed_server_emits_exactly_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other side of the storm test: the armed server is not lost.

        Gating must not be so broad that a genuine recovery is swallowed when
        it arrives through a reload rather than a login.
        """
        (tmp_path / ".local-operator").mkdir()
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"a": {"type": "stdio", "command": "a-cmd"},'
            ' "b": {"type": "stdio", "command": "b-cmd"},'
            ' "c": {"type": "stdio", "command": "c-cmd"}}}',
            encoding="utf-8",
        )
        manager = McpManager(str(tmp_path))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        await self._trip_breaker(manager, "b", monkeypatch)
        monkeypatch.setattr(manager, "_connect_server", good_connect)

        await manager.reload()
        assert recoveries == [("b", 1)]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_startup_gate_failure_then_connect_emits_no_recovery(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A boot failure fires no incident, so it earns no recovery.

        This pins the §2 decision to arm from the SINK rather than from
        ``_startup_failures``: the gate arm records the failure for the boot
        report but never tells the model, so announcing a recovery from it
        would "supersede" an incident the model never received.
        """
        manager = McpManager(str(project))
        incidents, recoveries = self._sinks(manager)
        attempt = {"n": 0}

        # ``**_kw`` absorbs ``interactive``, which the /mcp login route passes.
        async def fail_then_succeed(name: str, cfg: Any, **_kw: Any) -> ServerConnection:
            if name == "fast" and attempt["n"] == 0:
                attempt["n"] += 1
                raise RuntimeError("boot failure")
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", fail_then_succeed)
        result = await manager.discover_and_connect()
        assert "fast" in result.errors
        assert incidents == []  # the gate arm fires no incident — the premise

        conn = await manager.connect_configured_server("fast")
        assert conn is not None
        assert recoveries == []
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_recovery_fires_once_per_failure_not_per_reconnect(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One notice per announced failure, not one per connect.

        A server that keeps reconnecting cleanly (a ``/mcp reload`` habit, a
        transport that re-establishes) must not re-announce: the model was
        told once and corrected once.
        """
        manager = McpManager(str(project))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        await self._trip_breaker(manager, "fast", monkeypatch)
        monkeypatch.setattr(manager, "_connect_server", good_connect)

        assert await manager.reconnect_server("fast") is not None
        assert recoveries == [("fast", 1)]
        for _ in range(3):
            assert await manager.reconnect_server("fast") is not None
        assert recoveries == [("fast", 1)]
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_removed_server_does_not_inherit_stale_arming(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A re-added server starts clean.

        ``_drop_removed_servers`` clears the arming with the rest of the
        per-server state; otherwise a config edit that removes and re-adds a
        server would announce a recovery for an incident about the old entry.
        """
        manager = McpManager(str(project))
        _incidents, recoveries = self._sinks(manager)

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        await self._trip_breaker(manager, "fast", monkeypatch)
        monkeypatch.setattr(manager, "_connect_server", good_connect)

        # Remove 'fast' from the config and reload: the arming goes with it.
        (project / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"slow": {"type": "stdio", "command": "slow-cmd"}}}',
            encoding="utf-8",
        )
        await manager.reload()
        assert recoveries == []
        assert manager.get_connection("fast") is None

        # Re-add it and connect: a first connection, not a recovery.
        (project / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"fast": {"type": "stdio", "command": "fast-cmd"},'
            ' "slow": {"type": "stdio", "command": "slow-cmd"}}}',
            encoding="utf-8",
        )
        await manager.reload()
        assert manager.get_connection("fast") is not None
        assert recoveries == []
        await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_no_recovery_sink_installed_is_a_no_op(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI shape: ``on_recovery is None`` and nothing breaks.

        ``local-operator mcp login`` builds a throwaway manager with no session
        behind it, so there is no sink to fire. The connect must still succeed,
        and the server must still be disarmed so a later session-backed manager
        does not inherit a phantom arming.

        Asserted through the SINK rather than through ``_incident_announced``
        (review round 1, R4), which is what the rest of this class does: a sink
        installed AFTER the sinkless reconnect must hear nothing, because that
        reconnect already consumed the arming. It is the discard's placement
        — unconditional, and ahead of the ``sink is None`` return — that makes
        that true, so this fails if the discard is ever moved below the return.
        """
        manager = McpManager(str(project))
        manager.on_incident = lambda server, reason: None
        assert manager.on_recovery is None

        async def good_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", good_connect)
        await manager.discover_and_connect()
        await self._trip_breaker(manager, "fast", monkeypatch)

        # The sinkless reconnect: it must succeed rather than trip over the
        # missing sink.
        monkeypatch.setattr(manager, "_connect_server", good_connect)
        assert await manager.reconnect_server("fast") is not None

        # Now a session-backed manager's sink arrives. The arming was consumed
        # by the reconnect above, so this must stay silent.
        recoveries: list[tuple[str, int]] = []
        manager.on_recovery = lambda server, count: recoveries.append((server, count))
        assert await manager.reconnect_server("fast") is not None
        assert recoveries == [], "a sinkless connect left the server armed to re-announce"
        await manager.disconnect_all()


class TestAuthBlockRevalidation:
    """Propagation of a SHARED grant change into a session that gave up.

    ``~/.local-operator/auth.db`` is shared by every running process, but a
    session that hit an auth failure never re-read it: the block was cleared
    only by user action IN THAT PROCESS. So completing ``/mcp reauth`` in one
    session left every other running session with a dead server for its whole
    lifetime — observed on the operator's machine as sessions booted at 08:52
    still reporting ``notion [disconnected]`` at 13:30 against a grant re-authed
    at 12:31 with eight hours left on it.

    Two properties here are load-bearing rather than stylistic, and each has a
    test that must not be deleted:

    * the retry is keyed on ``tokens_obtained_at``, NOT on the row's
      ``updated_at`` — a marker that moves on our own writes turns a 60 s poll
      in nine processes into a refresh-token retry storm against a provider
      running reuse detection, which revokes the whole token family
      (``test_a_client_info_write_is_not_a_new_grant``);
    * the poll never connects interactively, because it runs unattended
      (``test_the_poll_never_opens_an_interactive_grant``).
    """

    URL = "https://srv.example/mcp"

    #: How far the steady-state fixture's chain stamp sits BEHIND the row's
    #: ``tokens_obtained_at``. Any positive distance does the work, and the work
    #: it does is the point: it is what makes the fixture a row that has already
    #: rotated (``issued_at != tokens_obtained_at``), so a read rule that reaches
    #: for the raw timestamp instead of the carried stamp is a rule UNDER TEST
    #: rather than one that happens to agree with the fixture.
    CHAIN_STAMP_AGE_S = 500.0

    @staticmethod
    def _oauth_manager(tmp_path: Path, store: Any = None) -> McpManager:
        """A manager with one OAuth HTTP server configured, no connections."""
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        manager = McpManager(str(tmp_path), auth_store=store)
        manager._configs["dd"] = MCPHttpServerConfig(
            url=TestAuthBlockRevalidation.URL, auth=MCPAuthConfig(type="oauth")
        )
        manager._sources["dd"] = "global"
        return manager

    @staticmethod
    def _real_store(tmp_path: Path, obtained_at: float) -> Any:
        """A REAL ``AuthStore`` over a temp file holding one STEADY-STATE grant.

        A real store rather than a fake because the whole mechanism rests on
        what ``McpTokenStorage`` actually writes: a fake that returns invented
        markers would pass while the production read looked at the wrong key.

        The row is in the shape EVERY production row reaches after its first
        rotation — a valid ``grant_chain`` pair whose ``issued_at`` is older than
        the row's ``tokens_obtained_at`` — and that is deliberate rather than
        incidental (agent review round 3, minor-1). The previous revision of this
        fixture seeded a row with NO pair, so ``_chain_stamp_of`` fell back to the
        raw timestamp in every guard test and three mutations of the rule passed
        all 624 tests: carrying the raw ``tokens_obtained_at`` in place of the
        chain stamp, ``seed_client_info`` popping the pair, and
        ``mark_grant_dead`` popping it. Each of those reds now, because the
        fixture no longer lets the fallback path and the rule coincide.
        """
        from local_operator.mcp.auth import (
            GRANT_CHAIN_KEY,
            MCP_OAUTH_PROVIDER,
            TOKENS_OBTAINED_AT_KEY,
        )
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(str(tmp_path / "auth.db"))
        store.upsert_credential(
            MCP_OAUTH_PROVIDER,
            {
                "project_id": TestAuthBlockRevalidation.URL,
                "tokens": {
                    "access_token": "STALE",
                    "refresh_token": "R1",
                    "token_type": "Bearer",
                    "expires_in": 28800,
                },
                TOKENS_OBTAINED_AT_KEY: obtained_at,
                # The row carries a pair, so this is a row that has already
                # rotated: the read must take the carried stamp, not the raw
                # timestamp sitting next to it.
                GRANT_CHAIN_KEY: {
                    "issued_at": obtained_at - TestAuthBlockRevalidation.CHAIN_STAMP_AGE_S,
                    "attested_at": obtained_at,
                },
            },
        )
        return store

    @staticmethod
    async def _block_via_reconnect(manager: McpManager, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drive the REAL ``_reconnect`` auth arm so the block is genuine."""
        from local_operator.mcp.auth import McpAuthRequiredError

        async def failing(name: str, cfg: Any, **_: Any) -> ServerConnection:
            raise McpAuthRequiredError(TestAuthBlockRevalidation.URL)

        monkeypatch.setattr(manager, "_connect_server", failing)
        await manager._reconnect("dd", 0.0, manager._epoch)

    async def _a_sibling_connect_stands_up(
        self,
        tmp_path: Path,
        store: Any,
        monkeypatch: pytest.MonkeyPatch,
        *,
        ours: Any,
    ) -> float | None:
        """A SECOND session connects for real on the same store, then ``ours`` is restored.

        This is the only thing that may write a success witness — a connect that
        stood up end to end — so the witness guards drive it through the real
        path rather than writing the payload, which is what makes them evidence
        about the mechanism instead of about a test's own bookkeeping.

        The transport stub is class-wide, so the sibling needs a dial that
        returns a connection for the length of its own connect and no longer:
        ``ours`` is the dial the calling test's session must keep (a refusal,
        normally) and is re-installed before this returns. The sibling is a
        separate MANAGER over the same file, which is the production shape — the
        shared ``auth.db`` is the whole reason this subsystem exists.

        Returns the row's witness value after the sibling stood up, or ``None``
        if it wrote none — the caller decides whether that is a failure or the
        subject under test.
        """
        from local_operator.mcp.auth import McpTokenStorage

        async def sibling_dial(name: str, cfg: Any) -> ServerConnection:
            return _make_conn(name, cfg)

        sibling = self._oauth_manager(tmp_path, store)
        self._stub_transport(monkeypatch, sibling_dial)
        try:
            await sibling._reconnect("dd", 0.0, sibling._epoch)
            assert sibling.get_connection_status("dd") == "connected", (
                "the sibling session did not stand up, so it witnessed nothing"
            )
        finally:
            await sibling.disconnect_all()
            self._stub_transport(monkeypatch, ours)
        marker = McpTokenStorage(self.URL, store).grant_marker()
        return None if marker is None else marker.witness_at

    @pytest.mark.asyncio
    async def test_an_auth_block_is_not_the_flap_breaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two conditions recover differently, so they are separate state.

        A breaker recovers when a server stops flapping; an auth block recovers
        when the shared grant is replaced. Fusing them is what forced the auth
        arm to borrow "permanent" semantics it could never shed.
        """
        manager = self._oauth_manager(tmp_path)
        await self._block_via_reconnect(manager, monkeypatch)

        assert manager.auth_blocked("dd") is True
        assert manager.reconnect_suspended("dd") is False
        assert "dd" not in manager._reconnect_suspended

    @pytest.mark.asyncio
    async def test_a_blocked_server_refuses_reconnects_exactly_as_before(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Splitting the sets must not weaken the refusal the block exists for.

        Both reconnect entry points still decline, and the parked waiter still
        gets a real error rather than hanging (MCP-08).
        """
        manager = self._oauth_manager(tmp_path)
        await self._block_via_reconnect(manager, monkeypatch)

        attempts: list[str] = []

        async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
            attempts.append(name)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", counting)
        manager._schedule_reconnect("dd")
        await asyncio.sleep(0)
        assert await manager._reconnect_for_call("dd") is None
        assert attempts == [], "a blocked server must not be re-dialled"

        future = manager._connect_futures.get("dd")
        assert future is None or future.done(), "a waiter was left parked forever"

    @pytest.mark.asyncio
    async def test_revalidate_retries_only_when_the_grant_marker_moved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unchanged grant costs zero connects, however many ticks run."""
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._block_via_reconnect(manager, monkeypatch)

            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", counting)
            for _ in range(5):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == []

            # A peer writes a fresh grant: exactly ONE attempt, and it heals.
            await self._write_fresh_grant(store)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert attempts == ["dd"]
            assert manager.get_connection_status("dd") == "connected"

            # …and a healed server is no longer polled at all.
            assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"]
        finally:
            await manager.disconnect_all()
            store.close()

    @staticmethod
    async def _write_fresh_grant(store: Any) -> None:
        """What a peer's completed ``/mcp reauth`` writes: a real ``set_tokens``."""
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import McpTokenStorage

        storage = McpTokenStorage(TestAuthBlockRevalidation.URL, store)
        await storage.set_tokens(
            OAuthToken(
                access_token="FRESH", refresh_token="R2", token_type="Bearer", expires_in=28800
            )
        )

    @pytest.mark.asyncio
    async def test_a_client_info_write_is_not_a_new_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE guard against a fleet-wide retry storm. Do not delete or weaken.

        ``wire_oauth_auth`` calls ``seed_client_info`` on EVERY connect for a
        pinned-client server, and that write moves the row's ``updated_at``. If
        the marker were keyed on ``updated_at`` — which looks like the same
        signal and is not — this session would retry on its own writes, every
        tick, in every process, re-presenting a refresh token that Notion's
        reuse detection answers by revoking the entire token family. Keying on
        ``tokens_obtained_at`` (written only by ``set_tokens``) is what makes
        the retry mean "somebody obtained a NEW grant".
        """
        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._block_via_reconnect(manager, monkeypatch)

            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", counting)

            storage = McpTokenStorage(self.URL, store)
            row_before = storage._read_row()
            assert row_before is not None
            for _ in range(3):
                storage.seed_client_info("client-abc")
            row_after = storage._read_row()
            assert row_after is not None
            assert row_after.updated_at >= row_before.updated_at

            assert await manager.revalidate_auth_blocked() == []
            assert attempts == [], "a client-info write was mistaken for a new grant"
            assert manager.auth_blocked("dd") is True
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_revalidate_reblocks_on_the_new_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A grant that changed but is still bad costs ONE attempt, not a loop.

        Re-blocking against the NEW marker is what bounds the cost to one
        connect per genuine grant change rather than one per tick.
        """
        from local_operator.mcp.auth import McpAuthRequiredError

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._block_via_reconnect(manager, monkeypatch)
            attempts: list[str] = []

            async def still_failing(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", still_failing)
            await self._write_fresh_grant(store)

            assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"]
            assert manager.auth_blocked("dd") is True

            for _ in range(3):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"], "a still-bad grant was retried on every tick"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_the_poll_never_opens_an_interactive_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This runs unattended in every session; a browser must never open."""
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._block_via_reconnect(manager, monkeypatch)
            seen: list[bool] = []

            async def recording(
                name: str, cfg: Any, *, interactive: bool = False, **_: Any
            ) -> ServerConnection:
                seen.append(interactive)
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", recording)
            await self._write_fresh_grant(store)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert seen == [False]
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_startup_gate_auth_failure_is_blocked_and_heals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second dead state: after-gate auth failures were FORGOTTEN.

        ``_finish_pending``'s auth arm fired the incident and the toast but
        neither suspended nor rescheduled, so the server was invisible to
        anything inspecting the breaker and was never retried. HTTP OAuth
        servers land here rather than in ``_reconnect``'s arm — discovery plus a
        refresh makes them miss the 250 ms gate — so this is the arm the
        operator's dead sessions were actually stuck in.
        """
        from local_operator.mcp.auth import McpAuthRequiredError

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            incidents: list[tuple[str, str]] = []
            recoveries: list[tuple[str, int]] = []
            manager.on_incident = lambda server, reason: incidents.append((server, reason))
            manager.on_recovery = lambda server, count: recoveries.append((server, count))

            # A connect that resolves AFTER the gate, exactly as a real OAuth
            # HTTP server does, so the failure runs through _finish_pending.
            async def slow_auth_failure(name: str, cfg: Any, **_: Any) -> ServerConnection:
                await asyncio.sleep(0.05)
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", slow_auth_failure)
            monkeypatch.setattr("local_operator.mcp.manager.STARTUP_GATE_MS", 1)
            await manager._connect_round({"dd": manager._configs["dd"]}, {"dd": "global"})
            for _ in range(200):
                await asyncio.sleep(0.005)
                if manager.auth_blocked("dd"):
                    break

            assert manager.auth_blocked("dd") is True
            assert manager.get_connection_status("dd") == "auth-required"
            assert [name for name, _ in incidents] == ["dd"]

            async def good(name: str, cfg: Any, **_: Any) -> ServerConnection:
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", good)
            await self._write_fresh_grant(store)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert manager.get_connection_status("dd") == "connected"
            # The model was told it broke, so it must be told it came back.
            assert recoveries == [("dd", 1)]
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_healing_a_server_the_model_never_heard_about_stays_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``on_recovery`` stays gated on ``_incident_announced`` through the poll.

        Healing routes through ``_register_connection`` precisely so the
        existing gate applies: a server blocked on a host with no incident sink
        must not produce a recovery notice for a failure nobody announced.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            # No incident sink installed => nothing armed.
            await self._block_via_reconnect(manager, monkeypatch)
            assert "dd" not in manager._incident_announced

            recoveries: list[tuple[str, int]] = []
            manager.on_recovery = lambda server, count: recoveries.append((server, count))

            async def good(name: str, cfg: Any, **_: Any) -> ServerConnection:
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", good)
            await self._write_fresh_grant(store)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert recoveries == []
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_status_reports_auth_required_only_while_actually_blocked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connected server never reports ``auth-required``.

        The status surfaces render this string directly, so a stale block must
        never outrank a live connection.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            assert manager.get_connection_status("dd") == "disconnected"
            await self._block_via_reconnect(manager, monkeypatch)
            assert manager.get_connection_status("dd") == "auth-required"

            async def good(name: str, cfg: Any, **_: Any) -> ServerConnection:
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", good)
            assert await manager.reconnect_server("dd") is not None
            assert manager.get_connection_status("dd") == "connected"
            assert manager.auth_blocked("dd") is False
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_healthy_fleet_reads_the_store_zero_times(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cost gate: this runs on a 60 s timer in every session.

        With nothing blocked the tick must not touch SQLite at all, or nine
        processes pay for a condition none of them are in.
        """
        manager = self._oauth_manager(tmp_path)
        reads: list[str] = []
        monkeypatch.setattr(
            manager, "_grant_marker", lambda name: (reads.append(name), (0.0, False, None))[1]
        )
        for _ in range(100):
            assert await manager.revalidate_auth_blocked() == []
        assert reads == []

    @pytest.mark.asyncio
    async def test_a_disposed_manager_never_registers_a_healed_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The poller and dispose race: a tick must not resurrect a connection.

        ``disconnect_all`` bumps the epoch; a connect already in flight when it
        runs must close its own stack rather than land in a torn-down manager.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._block_via_reconnect(manager, monkeypatch)
            await self._write_fresh_grant(store)
            closed = {"stack": False}

            class _Stack:
                async def aclose(self) -> None:
                    closed["stack"] = True

            async def disposing_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
                await manager.disconnect_all()
                conn = _make_conn(name, cfg)
                conn.stack = cast(Any, _Stack())
                return conn

            monkeypatch.setattr(manager, "_connect_server", disposing_connect)
            assert await manager.revalidate_auth_blocked() == []
            assert manager.get_connection("dd") is None
            assert closed["stack"] is True
        finally:
            store.close()

    # --- the REAL seam: these tests replace only the transport --------------
    #
    # Agent review round 2, major-1. The previous revision's guard tests
    # replaced ``_connect_server`` wholesale and hand-wrote the marker line
    # inside the stub, so they asserted the AUTHOR'S MODEL of the seam rather
    # than the seam: they passed with the production seam deleted and with it
    # moved back above the refresh (the reviewer measured 15 passed in both
    # cases). Everything below replaces ``_open_transport_and_session`` and
    # nothing else, so the real ``_connect_server`` body — and the real attempt
    # seam inside it — executes.
    #
    # The rotations are the three real ones, driven through production code.
    # ``ensure_mcp_oauth_fresh`` says of itself that it "is one of three ways a
    # refresh can start": the pre-dial refresh is driven by the real
    # ``_ensure_oauth_fresh``, and the other two — the in-flight coordinator an
    # ``async_auth_flow`` runs before every request, and the 401-recovery refresh
    # — by pumping the real ``async_auth_flow`` of the provider the transport
    # builds, which is how httpx drives it in production.

    @pytest.fixture(autouse=True)
    def _isolated_lock_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep this class's refresh lock out of the operator's real config dir.

        These tests drive the REAL refresh path, which takes the cross-process
        lock under ``config_dir()``. Without a per-test dir, two xdist workers
        running them at once contend on the SAME lock path, and a sibling
        worker's exchange would be read as this test's — the coupling the
        durability tests document. A per-test dir also means a leaked lock
        cannot outlive the test that took it.
        """
        from local_operator.paths import CONFIG_DIR_ENV

        monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "cfg"))

    @staticmethod
    def _endpoints() -> Any:
        """The discovered endpoint set the provider and the refresh both use."""
        from mcp.shared.auth import OAuthMetadata

        from local_operator.mcp.auth import DiscoveredOAuthEndpoints

        return DiscoveredOAuthEndpoints(
            oauth_metadata=OAuthMetadata.model_validate(
                {
                    "issuer": TestAuthBlockRevalidation.URL,
                    "authorization_endpoint": "https://as.example/authorize",
                    "token_endpoint": "https://as.example/token",
                }
            )
        )

    @staticmethod
    def _stub_discovery(
        monkeypatch: pytest.MonkeyPatch, endpoints: Any, *, during: Any = None
    ) -> None:
        """Stand in for the network at the DISCOVERY boundary.

        Discovery is the first thing the pre-dial refresh does and it is a real
        HTTP call, so it is stubbed at the boundary rather than mocked away.
        ``during`` is awaited while that call is "in flight", which is how a test
        places a peer's ``/mcp reauth`` INSIDE our pre-dial window — a window
        that is seconds wide in production (discovery plus the token POST).
        """
        from local_operator.mcp import auth as auth_mod

        async def discovery(url: str) -> Any:
            if during is not None:
                await during()
            return endpoints

        monkeypatch.setattr(auth_mod, "discover_oauth_endpoints", discovery)

    @staticmethod
    def _stub_token_endpoint(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """A rotating authorization server, counting POSTs at the TRANSPORT.

        Counting at the transport is what makes "no rotation happened" a fact
        about the wire rather than about our logging.
        """
        import httpx
        from mcp.shared.auth import OAuthToken

        calls = {"posts": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["posts"] += 1
            n = calls["posts"]
            token = OAuthToken(
                access_token=f"A{n}",
                refresh_token=f"R{n + 1}",
                token_type="Bearer",
                expires_in=28800,
            )
            return httpx.Response(200, json=token.model_dump(mode="json"), request=request)

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched)
        return calls

    @staticmethod
    async def _seed_client_info(store: Any) -> None:
        """Register a client, without which no refresh site does anything.

        ``ensure_mcp_oauth_fresh`` returns before the lock when the row has no
        client registration, so a rotation test that skipped this would measure
        nothing at all.
        """
        from mcp.shared.auth import OAuthClientInformationFull

        from local_operator.mcp.auth import McpTokenStorage

        await McpTokenStorage(TestAuthBlockRevalidation.URL, store).set_client_info(
            OAuthClientInformationFull(client_id="cid")
        )

    @staticmethod
    def _write_fresh_grant_once(store: Any) -> Any:
        """``_write_fresh_grant``, once — the peer re-auths once, not per attempt.

        The discovery stub runs on EVERY attempt that gets that far, so an
        unguarded peer write there would have the peer re-authing once per poll
        and the test would measure its own storm instead of ours.
        """
        written = False

        async def once() -> None:
            nonlocal written
            if written:
                return
            written = True
            await TestAuthBlockRevalidation._write_fresh_grant(store)

        return once

    @staticmethod
    def _row(store: Any) -> dict[str, Any]:
        """The one ``mcp-oauth`` row's payload, as written to disk."""
        from local_operator.mcp.auth import MCP_OAUTH_PROVIDER

        return store.list_credentials(MCP_OAUTH_PROVIDER)[0].data

    @staticmethod
    def _auth_error() -> Exception:
        from local_operator.mcp.auth import McpAuthRequiredError

        return McpAuthRequiredError(TestAuthBlockRevalidation.URL)

    @staticmethod
    def _stub_transport(monkeypatch: pytest.MonkeyPatch, dial: Any) -> None:
        """Replace ONLY ``_open_transport_and_session``.

        ``dial`` is the whole of what the transport does — rotate the grant, take
        a peer's write, tombstone it, or nothing — and it either returns a
        connection (the connect succeeds) or returns ``None`` for the documented
        "refused us on authorization" failure. Dials that want a different
        failure raise it themselves.

        Module-level ``setattr`` on the class, so the real ``_connect_server``
        runs for every caller (``_reconnect``, the revalidation poll, the
        call-site retry) exactly as it does in production.
        """

        async def transport(
            self: McpManager,
            stack: Any,
            name: str,
            cfg: Any,
            timeout_s: float | None,
            stderr_log: Any,
            **_: Any,
        ) -> ServerConnection:
            result = await dial(name, cfg)
            if result is None:
                raise TestAuthBlockRevalidation._auth_error()
            return result

        monkeypatch.setattr(McpManager, "_open_transport_and_session", transport)

    @staticmethod
    async def _drive_flow(store: Any, cfg: Any, endpoints: Any, *, refuse_original: bool) -> str:
        """Pump the REAL ``async_auth_flow`` the way the transport does.

        Returns the ``Authorization`` header of the first request the flow
        yields, which is the proof that the in-transport site under test really
        carried a token: after a coordinator refresh it is the ROTATED one, and
        without one it is the stored token.

        ``refuse_original`` answers that first request with a 401, which is what
        starts the flow's own 401-recovery refresh — the documented provider
        shape, where our locally-valid token was revoked server-side. The
        recovery re-yields the original request only after it has persisted a
        rotation, so reaching the second ``asend`` is itself the evidence.
        """
        import httpx

        from local_operator.mcp.auth import build_oauth_provider

        provider = build_oauth_provider(
            TestAuthBlockRevalidation.URL, cfg, store=store, endpoints=endpoints
        )
        async with provider.context.lock:
            await provider._initialize()
        gen = provider.async_auth_flow(
            httpx.Request("POST", TestAuthBlockRevalidation.URL, content=b"payload")
        )
        try:
            request = await gen.__anext__()
            header = request.headers.get("Authorization", "")
            if refuse_original:
                retried = await gen.asend(httpx.Response(401, request=request))
                assert retried is not None
            return header
        finally:
            await gen.aclose()

    @pytest.mark.asyncio
    async def test_a_grant_written_during_the_failing_connect_still_heals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE unfalsifiable-block defect. Do not delete or weaken.

        The marker used to be read AFTER the connect failed, so a peer's re-auth
        that landed WHILE that connect was in flight was recorded as "the grant
        we already failed on". The marker then never moves again — the new grant
        is valid, so nobody re-obtains it — and the block is unfalsifiable: the
        server stays dead for the life of the process against a perfectly good
        credential, and every ``/mcp reauth`` in another session is a no-op
        because the block being cleared is per-process in-memory state.

        Observed on the operator's machine 2026-09-18: a session failed a
        ``linear`` connect at 22:13:18 having read the marker for a grant written
        at 22:11:06, blocked on it, and was still dead ten hours later while a
        sibling session used the same row happily.

        The window is not exotic. It is exactly as wide as an OAuth connect —
        PRM/ASM discovery plus a token exchange, seconds — and a human who has
        just been told "this server needs authorizing" is re-authing during it
        BY CONSTRUCTION. This is the common case, not the race nobody hits.

        Driven through the REAL seam (only the transport is replaced), and the
        peer's write lands INSIDE our pre-dial window, at the discovery call —
        so the assertion below distinguishes a marker taken before that window
        from one taken after it.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            # The peer's re-auth lands DURING this connect, before it fails — so
            # a read taken after the failure would see the NEW grant and block
            # on it.
            self._stub_discovery(
                monkeypatch,
                self._endpoints(),
                during=self._write_fresh_grant_once(store),
            )

            async def refuses(name: str, cfg: Any) -> None:
                return None

            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            # The block names the grant the attempt STARTED with, not the peer's.
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)

            # The grant on disk is NEWER than the one this attempt actually
            # used, so the very next tick owes it one attempt. Before the fix
            # this returned [] forever: the block had been taken against a grant
            # the failed connect never tried.
            attempts: list[str] = []

            async def counting(name: str, cfg: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            self._stub_transport(monkeypatch, counting)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert attempts == ["dd"]
            assert manager.get_connection_status("dd") == "connected"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_the_healed_retry_is_still_exactly_one_per_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing the window must not reopen the retry storm it guards.

        Healing on a marker the attempt did not use is correct exactly once: if
        that attempt ALSO fails, the block must re-take against the grant it just
        tried, so a dead-but-newer grant still costs one connect per change
        rather than one per 60 s tick in nine processes.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            self._stub_discovery(
                monkeypatch,
                self._endpoints(),
                during=self._write_fresh_grant_once(store),
            )

            async def refuses(name: str, cfg: Any) -> None:
                return None

            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)

            # The retry this earns also fails, and writes nothing new.
            attempts: list[str] = []

            async def still_failing(name: str, cfg: Any) -> None:
                attempts.append(name)
                return None

            self._stub_transport(monkeypatch, still_failing)
            assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"], "the newer grant earned exactly one attempt"

            # …and now it is quiet again: same grant, no further connects.
            for _ in range(5):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"], "a re-block must not retry on an unchanged grant"
        finally:
            await manager.disconnect_all()
            store.close()

    async def _run_rotation_scenario(
        self, site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Any:
        """Drive one of OUR three refresh sites inside a failing connect.

        Returns a namespace with the manager, the store, the stamp the attempt
        started with, the 30-poll connect count and the refresh-POST count. Does
        not assert the behaviour under test — the two tests below do that, one
        for the storm and one for the mechanism that closes it, so each mutation
        reds the assertion that names it rather than whichever comes first.
        """
        import time as _time

        from local_operator.mcp.auth import McpTokenStorage

        # A locally-VALID token is what the 401-recovery shape needs: the
        # coordinator skips a fresh token, and the server's refusal is what
        # starts the recovery refresh. The other two sites need an expired one,
        # so the pre-dial refresh and the coordinator have something to do.
        obtained_at = _time.time() if site == "401_recovery" else 1000.0
        store = self._real_store(tmp_path, obtained_at=obtained_at)
        manager = self._oauth_manager(tmp_path, store)
        endpoints = self._endpoints()
        await self._seed_client_info(store)
        storage = McpTokenStorage(self.URL, store)
        posts = self._stub_token_endpoint(monkeypatch)
        before = storage.grant_marker()
        assert before == (obtained_at - self.CHAIN_STAMP_AGE_S, False, None)

        # The pre-dial site rotates BEFORE the transport opens, so it needs
        # discovery to answer. The other two rotate INSIDE the transport, and
        # discovery returning nothing is what keeps the pre-dial refresh out of
        # their way: ``ensure_mcp_oauth_fresh`` returns before the lock when
        # there is no endpoint to POST to.
        self._stub_discovery(monkeypatch, endpoints if site == "pre_dial" else None)

        async def refuses(name: str, cfg: Any) -> None:
            if site == "coordinator":
                # The real in-flight coordinator, which the transport drives
                # before its first request. It rotates an expired token.
                header = await self._drive_flow(store, cfg, endpoints, refuse_original=False)
                assert header == "Bearer A1", header
            elif site == "401_recovery":
                # The real recovery refresh: our token is locally valid, the
                # server refuses it with a 401, and the flow rotates under the
                # lock before re-yielding the request.
                header = await self._drive_flow(store, cfg, endpoints, refuse_original=True)
                assert header == "Bearer STALE", header
            return None

        self._stub_transport(monkeypatch, refuses)
        await manager._reconnect("dd", 0.0, manager._epoch)
        assert manager.auth_blocked("dd") is True

        connects: list[str] = []

        async def same_shape(name: str, cfg: Any) -> None:
            connects.append(name)
            if site == "401_recovery":
                # The storm shape: this provider keeps refusing the token it
                # holds, so every attempt that DOES happen rotates again.
                await self._drive_flow(store, cfg, endpoints, refuse_original=True)
            return None

        self._stub_transport(monkeypatch, same_shape)
        for _ in range(30):
            await manager.revalidate_auth_blocked()
        return SimpleNamespace(
            manager=manager,
            store=store,
            storage=storage,
            before=before,
            connects=connects,
            posts=posts,
            row=self._row(store),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("site", ["pre_dial", "coordinator", "401_recovery"])
    async def test_our_own_in_connect_refresh_is_not_a_peer_reauth(
        self, site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE self-write storm guard for the attempt marker. Do not weaken.

        Each of our three refresh sites moves ``tokens_obtained_at``, and the
        authorization server ROTATES on every one of them. A marker keyed on that
        timestamp is therefore stale the instant a connect that rotated fails,
        and every later poll reads the movement as "a peer re-authed" — one
        connect and one refresh-token POST per tick, per process, forever.
        Against a provider running reuse detection that revokes the whole token
        family: the storm ``GRANT_DEAD_AT_KEY`` and ``_grant_marker`` exist to
        prevent, reached through the self-write rather than a naive
        ``updated_at``.

        Round 1 measured 30 connects over 30 polls on the pre-dial site and round
        2 30/30 on the in-transport one, against 0 on ``main``. All three sites
        are exercised here on their own real path — a marker that held for one of
        them is exactly the half-fix round 2 rejected — and the assertion below
        is the observable one the storm was measured on, so the failure it
        reports carries the count.
        """
        from local_operator.mcp.auth import TOKENS_OBTAINED_AT_KEY

        scenario = await self._run_rotation_scenario(site, tmp_path, monkeypatch)
        try:
            assert scenario.posts["posts"] >= 1, "no refresh POST: this test measured nothing"
            assert (
                scenario.row[TOKENS_OBTAINED_AT_KEY] != scenario.before[0]
            ), "the grant did not rotate: this test measured nothing"
            assert scenario.connects == [], (
                "our own refresh was mistaken for a peer's re-auth: "
                f"{len(scenario.connects)} extra connects over 30 polls"
            )
        finally:
            await scenario.manager.disconnect_all()
            scenario.store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("site", ["pre_dial", "coordinator", "401_recovery"])
    async def test_a_rotation_carries_the_chain_stamp_it_started_from(
        self, site: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mechanism, asserted separately from the storm it prevents.

        Two facts, per site: the pair's ``issued_at`` is the stamp the row
        ALREADY carried (the carry-forward, not a mint), and the block names that
        stamp rather than one read after the connect. The second is what a marker
        read at block time gets wrong for a PEER's write; the first is what makes
        it right for ours.
        """
        from local_operator.mcp.auth import GRANT_CHAIN_KEY, TOKENS_OBTAINED_AT_KEY

        scenario = await self._run_rotation_scenario(site, tmp_path, monkeypatch)
        try:
            pair = scenario.row[GRANT_CHAIN_KEY]
            assert (
                pair["issued_at"] == scenario.before[0]
            ), "the rotation minted a new chain stamp instead of carrying the old one"
            assert pair["attested_at"] == scenario.row[TOKENS_OBTAINED_AT_KEY], (
                "the pair does not attest the row's own timestamp, so the read rule "
                "will never believe it"
            )
            assert (
                scenario.manager._auth_grant_marker.get("dd") == scenario.before
            ), "the block did not name the grant the attempt started with"
        finally:
            await scenario.manager.disconnect_all()
            scenario.store.close()

    @pytest.mark.asyncio
    async def test_a_rotation_of_ours_moves_no_marker_another_process_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The state that closes the storm is IN THE ROW, not in this process.

        A peer session — a different process, sharing only ``auth.db`` — must see
        our rotation as "no movement" too. Anything process-local (an own-write
        ledger, a memoized marker) passes an in-process test and fails here,
        which is why the alternative designs were rejected by measurement: two
        real processes scored 30/31 with a ledger against 1/1 with the chain
        stamp. This is the hermetic form of that measurement: a SECOND manager
        and storage, built fresh over the same file, reads the same stamp and
        spends nothing.
        """
        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        endpoints = self._endpoints()
        try:
            await self._seed_client_info(store)
            posts = self._stub_token_endpoint(monkeypatch)
            self._stub_discovery(monkeypatch, endpoints)

            async def refuses(name: str, cfg: Any) -> None:
                return None

            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)
            assert posts["posts"] >= 1

            # A second process: same file, same server, nothing shared in memory.
            peer = TestAuthBlockRevalidation._oauth_manager(tmp_path, store)
            peer._auth_blocked.add("dd")
            # The baseline both processes agreed on is the stamp the blocked
            # attempt held. Reading the row again here instead would make this
            # test tautological: the peer would be handed whatever our rotation
            # wrote, and no implementation could ever fail it.
            peer._auth_grant_marker["dd"] = manager._auth_grant_marker["dd"]
            connects: list[str] = []

            async def counting(name: str, cfg: Any) -> None:
                connects.append(name)
                return None

            self._stub_transport(monkeypatch, counting)
            for _ in range(30):
                await peer.revalidate_auth_blocked()
            assert connects == [], "another process read our own rotation as movement"
            await peer.disconnect_all()
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_tombstone_written_inside_the_transport_is_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tombstone costs AT MOST one connect, then flat — never a storm.

        ``mark_grant_dead`` runs inside the transport when the authorization
        server has rejected the grant, and it deliberately leaves the chain
        stamp where it is. Whether it also moves the marker's dead element
        depends on the REAL store, and on current main it does not:
        ``AuthStore.upsert_credential`` strips ``grant_dead_at`` from every
        payload it writes, so the tombstone never persists and the marker never
        moves (zero connects). If that strip is ever lifted the dead element
        moves once, the block's dead flag goes stale, and the attempt earns
        exactly one retry that re-blocks (one connect — the shape round 1
        measured and accepted before the strip landed).

        The strip is a separate, DEFERRED defect recorded on PR #1329 ("the MCP
        dead-grant tombstone is a no-op in production"), awaiting the operator's
        decision. So this test pins only what the chain rule guarantees in BOTH
        worlds — the bound and the unmoved chain stamp — and deliberately pins
        neither side of the strip: asserting the tombstone persists is not true
        on main, and asserting it does not would make the deferred fix break an
        unrelated test.
        """
        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            storage = McpTokenStorage(self.URL, store)
            self._stub_discovery(monkeypatch, None)

            async def tombstone(name: str, cfg: Any) -> None:
                storage.mark_grant_dead(rejected_refresh_token="R1")
                return None

            self._stub_transport(monkeypatch, tombstone)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)

            connects: list[str] = []

            async def tombstone_again(name: str, cfg: Any) -> None:
                connects.append(name)
                storage.mark_grant_dead(rejected_refresh_token="R1")
                return None

            self._stub_transport(monkeypatch, tombstone_again)
            for _ in range(30):
                await manager.revalidate_auth_blocked()
            assert connects in ([], ["dd"]), (
                "a tombstone must cost at most one connect, then go quiet: "
                f"got {len(connects)} over 30 polls"
            )
            # …and the tombstone did not move the chain stamp underneath it, on
            # the block record OR on disk (the dead element is not asserted: see
            # the docstring for why neither value is pinned).
            # The stamp the fixture's steady-state chain carries: the pair's
            # ``issued_at``, NOT the row's ``tokens_obtained_at`` — see
            # ``_real_store``, and ``_chain_stamp_of``'s F rule for why the
            # distinction is the whole mechanism.
            expected_stamp = 1000.0 - self.CHAIN_STAMP_AGE_S
            blocked = manager._auth_grant_marker.get("dd")
            assert blocked is not None and blocked[0] == expected_stamp
            on_disk = storage.grant_marker()
            assert on_disk is not None and on_disk[0] == expected_stamp
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("order", ["peer_then_rotation", "rotation_then_peer"])
    async def test_a_peer_login_and_our_rotation_in_one_attempt_still_heals(
        self, order: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both writers land in ONE attempt, in both orders: heal, exactly once.

        This is the case the CHAIN rule has to get right and a single timestamp
        cannot: the attempt contains a peer's ``/mcp reauth`` AND a rotation of
        ours. Whichever order they land in, the grant the session ends up blocked
        against must be one a later poll can see replaced — so the next tick
        spends exactly one connect and the server is connected.
        """
        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        endpoints = self._endpoints()
        try:
            await self._seed_client_info(store)
            self._stub_token_endpoint(monkeypatch)
            # The pre-dial refresh stays out of the way: OUR rotation below is the
            # in-transport recovery one, which is a real site and the one the
            # transport can reach.
            self._stub_discovery(monkeypatch, None)

            async def both(name: str, cfg: Any, *, order: str = order) -> None:
                if order == "peer_then_rotation":
                    await self._write_fresh_grant(store)
                    # …and now we rotate the grant we were just handed. The
                    # provider is initialized from the store AFTER the peer's
                    # write, so the 401 goes down the recovery path (no peer
                    # token to adopt: store and memory agree).
                    await self._drive_flow(store, cfg, endpoints, refuse_original=True)
                else:
                    await self._drive_flow(store, cfg, endpoints, refuse_original=True)
                    await self._write_fresh_grant(store)
                return None

            self._stub_transport(monkeypatch, both)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            # The attempt's record is the grant it started with (the stale one),
            # NOT the newer state both writers left behind — which is what makes
            # the next tick's comparison meaningful.
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)
            blocked = manager._auth_grant_marker["dd"]
            assert McpTokenStorage(self.URL, store).grant_marker() != blocked, (
                "the two writers between them must leave the disk newer than the "
                "attempt's record, or there is nothing to heal from"
            )

            attempts: list[str] = []

            async def counting(name: str, cfg: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            self._stub_transport(monkeypatch, counting)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert attempts == ["dd"], "exactly one retry, earned by the replaced grant"
            assert manager.get_connection_status("dd") == "connected"
            for _ in range(5):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"], "a heal must not leave a retry running"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_chain_unaware_writer_is_not_a_new_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REWRITTEN for agent review round 3, major-1: the claim was false.

        This test used to assert the opposite — that a writer which moves
        ``tokens_obtained_at`` without maintaining the pair reads as a replaced
        grant, costing one retry that heals. That is true of ONE write and false
        of an old BUILD, which is a process that keeps writing: the new side
        reads each of its rotations as a new grant, the old side reads each of
        the new side's rotations as movement in turn, and the two feed each
        other (two real processes: ``main+head`` 26/26 rotations per process over
        30 polls, against 1/1 with the F rule).

        So the assertion is inverted rather than the test deleted, because the
        behaviour it pins still matters in the other direction: a chain-unaware
        writer must NOT be read as a new grant. Driven through the REAL seam
        (only the transport is replaced), with a real old build's write through
        the production write path.

        The heals this test used to cover are not lost — they moved to the two
        bridges that remain, each with its own guard: an ABSENT pair (a legacy
        row, or ``set_tokens``' pop) falls back to ``tokens_obtained_at``, and a
        cleared row reads as ``0.0``
        (``test_a_peer_login_and_our_rotation_in_one_attempt_still_heals``,
        ``test_an_interactive_login_heals_a_blocked_session``).
        """
        from local_operator.mcp.auth import (
            GRANT_CHAIN_KEY,
            TOKENS_OBTAINED_AT_KEY,
            McpTokenStorage,
        )

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            storage = McpTokenStorage(self.URL, store)
            self._stub_discovery(monkeypatch, None)

            async def refuses(name: str, cfg: Any) -> Any:
                return None

            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)

            # An OLD-VERSION writer: tokens + the timestamp, the pair left as it
            # was, written through the production write path.
            creds = storage._read() or {}
            creds["tokens"] = {
                "access_token": "OLD",
                "refresh_token": "OLD-R",
                "token_type": "Bearer",
            }
            creds[TOKENS_OBTAINED_AT_KEY] = 2000.0
            assert GRANT_CHAIN_KEY in creds
            storage._write(creds)

            # The pair still stands, so the stamp did not move: the block stays,
            # and no refresh token is spent answering an old build's write.
            attempts: list[str] = []

            async def counting(name: str, cfg: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            self._stub_transport(monkeypatch, counting)
            assert await manager.revalidate_auth_blocked() == []
            assert attempts == [], (
                "a chain-unaware writer was read as a new grant: "
                f"{len(attempts)} extra connects"
            )
            assert manager.auth_blocked("dd") is True
            assert manager.get_connection_status("dd") != "connected"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_non_auth_failure_leaves_no_attempt_state_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round 2, major-2: the record dies with its attempt, by construction.

        The per-server dict this replaced recorded a marker for every connect,
        including the ones whose caller can never block, so a marker could
        outlive its attempt and be adopted by a later, unrelated failure. The
        record is a local owned by the blocking caller, so the durable state here
        is the only attempt state that exists — asserted directly rather than
        argued from the code's shape.
        """
        from local_operator.mcp.manager import McpTransportError

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            self._stub_discovery(monkeypatch, None)

            async def dead_transport(name: str, cfg: Any) -> None:
                raise McpTransportError(self.URL, "connection refused")

            self._stub_transport(monkeypatch, dead_transport)
            for _ in range(5):
                await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is False
            assert manager._auth_grant_marker == {}
            # No per-server attempt state of any kind survives: the state that
            # used to hold it is gone, so a stale entry cannot be adopted later.
            assert not [name for name in vars(manager) if "attempt" in name]
        finally:
            await manager.disconnect_all()
            store.close()


    @pytest.mark.asyncio
    async def test_a_chain_unaware_writer_never_storms(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Agent review round 3, major-1 (and QA round 1, Q2): an old BUILD is a
        PROCESS, not one write. Do not weaken.

        The claim this replaces — "a writer that moves ``tokens_obtained_at``
        without the pair costs ONE unearned retry and never a storm" — is false
        while an old process is still RUNNING, which is exactly what a fleet
        mid-rollout guarantees: the old side rotates every tick and moves the
        timestamp without maintaining the pair, the new side reads each of those
        rotations as a new grant and retries, and the old side reads each of the
        NEW side's rotations as movement in turn, so the two feed each other.
        Measured with two real processes over one ``auth.db``, 30 polls each,
        rotations per process: ``main+main`` 2/1, ``main+head`` 26/26 (reviewer
        27/27, 21/21, 14/14; QA 14/14, 12/11), ``head+head`` 1/1, and with the F
        rule ``main+this tree`` 1/1. At the production 60 s poll that is one
        refresh POST per minute per process for every server blocked in BOTH
        versions, for the whole rollout window, on every release.

        The foreign writer here is that old build's write through the production
        write path — tokens plus the raw timestamp, the pair left exactly as it
        was — repeated on every tick, which is the shape one old process
        produces. Under the F rule the stamp is the pair's ``issued_at``
        throughout, so there is nothing for the poll to react to; a read rule
        that falls back on the moved timestamp reds this with one connect per
        tick.
        """
        from local_operator.mcp.auth import (
            GRANT_CHAIN_KEY,
            TOKENS_OBTAINED_AT_KEY,
            McpAuthRequiredError,
            McpTokenStorage,
        )

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            storage = McpTokenStorage(self.URL, store)
            await self._block_via_reconnect(manager, monkeypatch)
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)

            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", counting)

            for tick in range(30):
                creds = storage._read() or {}
                creds["tokens"] = {
                    "access_token": f"OLD{tick}",
                    "refresh_token": f"OLD-R{tick}",
                    "token_type": "Bearer",
                    "expires_in": 28800,
                }
                # The only thing an old build moves: the raw timestamp. Whatever
                # it does with the pair, it cannot make the pair lie about it.
                creds[TOKENS_OBTAINED_AT_KEY] = 2000.0 + tick
                assert GRANT_CHAIN_KEY in creds, "the old writer must LEAVE the pair alone"
                storage._write(creds)
                await manager.revalidate_auth_blocked()

            assert attempts == [], (
                "a chain-unaware writer's rotation was read as a new grant: "
                f"{len(attempts)} extra connects over 30 polls"
            )
            assert manager.auth_blocked("dd") is True
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_sibling_success_heals_a_blocked_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QA round 1, Q1 (major): the heal no chain rule can see. Do not delete.

        The operator's "stuck while siblings work" symptom, on the path ``main``
        heals and this head did not. A session blocks after a TEMPORARY
        provider-side rejection; the incident ends; a sibling stands up on the
        same grant — and that sibling's rotation CARRIES the chain stamp forward,
        so the only new fact on the row is that the chain WORKS. Measured with
        real processes and a real authorization server: ``auth-required`` for 60
        polls on the head, 2/2 runs, while ``main`` healed it 2/2 only by letting
        EVERY blocked session retry on every tick (the 17/17 storm).

        Two things are pinned here, and the second is why the baseline must be
        taken by the ATTEMPT rather than at block time:

        * the heal itself, and that it costs exactly ONE attempt;
        * that the block names the attempt's own baseline while the row carries a
          witness written DURING that attempt — so the next tick consumes it. A
          baseline read at block time would silently swallow it (the round-2
          major-2 defect class, on the witness axis) and the session would stay
          blocked forever with the evidence already on the row.
        """
        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._seed_client_info(store)
            storage = McpTokenStorage(self.URL, store)
            assert storage.grant_marker().witness_at is None, "nothing has stood up yet"

            async def refuses(name: str, cfg: Any) -> Any:
                return None  # "refused us on authorization"

            # OUR attempt: refused on the stale grant, with the sibling standing
            # up INSIDE it, at the discovery call — the window that is seconds
            # wide in production.
            done = {"ran": False}

            async def sibling_stands_up_during_our_attempt() -> None:
                if done["ran"]:
                    return
                done["ran"] = True
                value = await self._a_sibling_connect_stands_up(
                    tmp_path, store, monkeypatch, ours=refuses
                )
                assert value is not None, "the sibling's connect wrote no witness"

            self._stub_discovery(
                monkeypatch, None, during=sibling_stands_up_during_our_attempt
            )
            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            assert done["ran"], "the sibling never landed inside our attempt: measured nothing"

            marker = storage.grant_marker()
            assert marker.witness_at is not None, "the sibling's success left no witness"
            assert marker.stamp == 500.0, (
                "the sibling's rotation moved the chain stamp, so this test would "
                "be measuring the chain heal instead of the witness"
            )
            blocked = manager._auth_grant_marker.get("dd")
            assert blocked is not None and blocked[0] == 500.0
            assert blocked[2] is None, (
                "the block's baseline is the sibling's witness, which landed during "
                "our attempt: it was read at BLOCK time and the evidence was swallowed"
            )

            attempts: list[str] = []

            async def counting(name: str, cfg: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            self._stub_transport(monkeypatch, counting)
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert attempts == ["dd"], (
                f"a sibling's success bought {len(attempts)} attempts, not 1"
            )
            assert manager.get_connection_status("dd") == "connected"
            for _ in range(5):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == ["dd"], "a consumed witness must buy nothing further"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_success_witness_buys_exactly_one_retry_each(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The BOUND, which is what keeps the witness from being a timer.

        A witness value exists only because a connect STOOD UP, and a successful
        connect clears that session's own block — so the session it wakes either
        heals and stops polling, or fails and writes no witness of its own.
        Retries are therefore counted in EVENTS, and this is the measurement: six
        distinct witness values (each one a sibling standing up for real over the
        shared store) buy six retries across sixty polls, never sixty. That is
        the difference between a heal signal and the per-tick retry that revokes
        token families, and it is the whole reason a timer cannot be substituted
        for the witness.
        """
        import asyncio as _asyncio

        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._seed_client_info(store)
            storage = McpTokenStorage(self.URL, store)

            ours: list[str] = []

            async def refuses(name: str, cfg: Any) -> Any:
                ours.append(name)
                return None  # our attempt is still refused on authorization

            self._stub_discovery(monkeypatch, None)
            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            baseline_attempts = len(ours)

            values: list[float] = []
            for _ in range(6):
                # Distinct wall-clock values: each one is a real connect that
                # stood up, which is the only thing that may produce one.
                await _asyncio.sleep(0.002)
                value = await self._a_sibling_connect_stands_up(
                    tmp_path, store, monkeypatch, ours=refuses
                )
                assert value is not None, "a sibling that stood up wrote no witness"
                values.append(value)
                for _ in range(10):
                    await manager.revalidate_auth_blocked()

            assert len(set(values)) == 6, (
                f"the sibling produced {len(set(values))} distinct witness values, not 6: "
                "the wall clock was too coarse to measure the bound"
            )
            assert len(ours) - baseline_attempts == 6, (
                f"{len(ours) - baseline_attempts} retries for {len(set(values))} witness "
                "values over 60 polls"
            )
            assert storage.grant_marker().stamp == 500.0, "the chain moved: wrong axis measured"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_stale_witness_is_not_evidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A value already consumed buys nothing — the rule is strictly NEWER.

        This is what stops the witness from degenerating into a clock: if a value
        at or behind the one this attempt consumed counted as movement, every
        blocked session would retry on every tick again, with a witness-shaped
        excuse. The payload is written directly here because the VALUE is the
        subject — the writer itself (a connect that stood up, and nothing else)
        is pinned by ``test_a_sibling_success_heals_a_blocked_session`` and
        ``test_the_success_witness_never_clobbers_a_racing_rotation``.
        """
        from local_operator.mcp.auth import GRANT_OK_KEY, McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._seed_client_info(store)
            storage = McpTokenStorage(self.URL, store)

            ours: list[str] = []

            async def refuses(name: str, cfg: Any) -> Any:
                ours.append(name)
                return None

            self._stub_discovery(monkeypatch, None)
            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            baseline_attempts = len(ours)

            value = await self._a_sibling_connect_stands_up(
                tmp_path, store, monkeypatch, ours=refuses
            )
            assert value is not None
            # The first poll CONSUMES it (and the retry fails, re-blocking on it).
            await manager.revalidate_auth_blocked()
            assert len(ours) - baseline_attempts == 1

            def write_witness(at: float) -> None:
                creds = storage._read() or {}
                creds[GRANT_OK_KEY] = {"at": at, "chain": 500.0}
                storage._write(creds)

            for at in (value, value - 0.001, value - 100.0):
                write_witness(at)
                for _ in range(5):
                    assert await manager.revalidate_auth_blocked() == []
            assert len(ours) - baseline_attempts == 1, (
                f"a witness at or behind the consumed value bought "
                f"{len(ours) - baseline_attempts - 1} more retries"
            )
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_new_interactive_grant_is_a_new_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``set_tokens`` pop is LOAD-BEARING, not tidiness.

        The read rule honours a present pair's ``issued_at`` unconditionally, so
        a pair left behind by an interactive grant would be read as THAT grant's
        stamp: the row's next rotation would carry the previous chain forward,
        and the key's contract would invert — a brand-new grant would read as the
        one the session already failed on (the unfalsifiable block) and its
        subsequent rotations would read as no movement at all.
        """
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import (
            GRANT_CHAIN_KEY,
            TOKENS_OBTAINED_AT_KEY,
            McpTokenStorage,
        )

        store = self._real_store(tmp_path, obtained_at=1000.0)
        storage = McpTokenStorage(self.URL, store)
        try:
            before = storage.grant_marker()
            assert before is not None and before.stamp == 500.0

            await storage.set_tokens(
                OAuthToken(
                    access_token="FRESH",
                    refresh_token="R2",
                    token_type="Bearer",
                    expires_in=28800,
                )
            )

            row = self._row(store)
            assert GRANT_CHAIN_KEY not in row, (
                "set_tokens left the previous chain's pair on a brand-new interactive "
                "grant: the next rotation will carry the FAILED chain forward"
            )
            after = storage.grant_marker()
            assert after is not None
            assert after.stamp != before.stamp, "a new interactive grant must be a new chain"
            assert after.stamp == row[TOKENS_OBTAINED_AT_KEY], (
                "the new chain's stamp must be the tokens_obtained_at this write made"
            )
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_our_own_rotation_on_a_steady_state_row_is_no_movement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Agent review round 3, minor-1(a): the carry is the CHAIN STAMP.

        Every production row has carried a pair since its first rotation, and on
        such a row ``issued_at`` and ``tokens_obtained_at`` are DIFFERENT
        numbers. That is what makes this test able to tell the two apart: a
        rotation that carried the raw timestamp forward instead of the row's
        carried stamp would move the stamp to the row's own (older-written,
        newer-valued) timestamp, so the poll would see movement after one of OUR
        rotations — the self-write storm, from the carry instead of the read.
        Measured on the pre-fix fixture, which seeded no pair: that mutation
        passed all 624 tests and added 30 connects over 30 polls.
        """
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import (
            GRANT_CHAIN_KEY,
            TOKENS_OBTAINED_AT_KEY,
            McpTokenStorage,
        )

        store = self._real_store(tmp_path, obtained_at=1000.0)
        storage = McpTokenStorage(self.URL, store)
        try:
            before = storage.grant_marker()
            assert before is not None and before.stamp == 500.0

            assert storage.store_refresh_result(
                OAuthToken(
                    access_token="A1",
                    refresh_token="R2",
                    token_type="Bearer",
                    expires_in=28800,
                ),
                presented_refresh_token="R1",
            )

            row = self._row(store)
            after = storage.grant_marker()
            assert after is not None
            assert row[TOKENS_OBTAINED_AT_KEY] != 500.0, (
                "the rotation did not move the row's timestamp: measured nothing"
            )
            assert after.stamp == before.stamp == 500.0, (
                "our own rotation moved the chain stamp: the carry took the raw "
                "tokens_obtained_at instead of the stamp the row already carried"
            )
            assert row[GRANT_CHAIN_KEY]["issued_at"] == 500.0
            assert row[GRANT_CHAIN_KEY]["attested_at"] == row[TOKENS_OBTAINED_AT_KEY]
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_a_failed_connect_writes_no_success_witness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a connect that STOOD UP is evidence. Do not weaken.

        The witness is a claim about the world ("this chain works"), so it may
        only be made by a path that has proved it: the transport entered, the
        session initialized and the tools listed. A failure path that wrote one
        would turn every transient outage into fleet-wide "the grant works",
        which is precisely the per-tick retry this feature replaced — with a
        witness-shaped excuse instead of a timer.

        The second half exists so the first cannot pass by being vacuous: the
        same row and the same writer DO produce a witness when a connect stands
        up (through a sibling session, since this one is now blocked).
        """
        from local_operator.mcp.auth import McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._seed_client_info(store)
            storage = McpTokenStorage(self.URL, store)

            async def refuses(name: str, cfg: Any) -> Any:
                return None

            self._stub_discovery(monkeypatch, None)
            self._stub_transport(monkeypatch, refuses)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True
            assert storage.grant_marker().witness_at is None, (
                "a FAILED connect wrote a success witness"
            )

            value = await self._a_sibling_connect_stands_up(
                tmp_path, store, monkeypatch, ours=refuses
            )
            assert value is not None, (
                "a connect that stood up wrote no witness, so the assertion above "
                "proves nothing about the failure path"
            )
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_the_success_witness_never_clobbers_a_racing_rotation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The witness write is a compare-and-skip, and that is not optional.

        It is a whole-payload read-modify-write, like every other writer on this
        row, so a rotation landing between the read and the write would be undone
        by our snapshot — and for a witness that means putting the SPENT refresh
        token back, which the next refresh then re-presents: the reuse-detecting
        POST that revokes the whole token family. The race is driven through the
        real funnel (a second storage rotating between our two reads), so what is
        asserted is the state of the ROW, not our return value.
        """
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import GRANT_OK_KEY, McpTokenStorage

        store = self._real_store(tmp_path, obtained_at=1000.0)
        storage = McpTokenStorage(self.URL, store)
        try:
            real_read = storage._read
            calls = {"n": 0}

            def racing_read() -> Any:
                calls["n"] += 1
                if calls["n"] == 2:
                    # A sibling's rotation lands between our two reads. It goes
                    # through the REAL funnel, so the row really does move on.
                    assert McpTokenStorage(self.URL, store).store_refresh_result(
                        OAuthToken(
                            access_token="A2",
                            refresh_token="R2",
                            token_type="Bearer",
                            expires_in=28800,
                        ),
                        presented_refresh_token="R1",
                    )
                return real_read()

            monkeypatch.setattr(storage, "_read", racing_read)
            assert storage.record_grant_ok() is False, (
                "the witness was written from a payload older than the rotation"
            )

            row = self._row(store)
            assert row["tokens"]["access_token"] == "A2", (
                "our stale snapshot overwrote the rotation, putting the SPENT refresh "
                "token back on the row"
            )
            assert row["tokens"]["refresh_token"] == "R2"
            assert GRANT_OK_KEY not in row
            assert calls["n"] == 2, f"the compare re-read did not happen (reads: {calls['n']})"
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_a_recovered_challenge_records_a_success_witness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flow's own seam: a session that recovers IN PLACE also witnesses it.

        The connect seam covers QA's repro, which is a sibling that RECONNECTS.
        The reviewer's "stranded sessions" framing is wider than that: a session
        whose token the resource first refused, and whose 401-recovery the
        resource then ACCEPTED, has proved the chain works as well — and it never
        reconnects, because the request simply succeeds on the retry. Driven
        through the REAL auth flow (a 401 on the stored token, the recovery
        refresh under the lock, a 200 for the re-yielded request), so what is
        asserted is that the row carries a witness afterwards. A normal request
        and a failed recovery both write nothing, which is what keeps the witness
        a fact about success rather than a heartbeat.
        """
        import httpx

        from local_operator.mcp.auth import McpTokenStorage, build_oauth_provider

        import time as _time

        store = self._real_store(tmp_path, obtained_at=_time.time())
        manager = self._oauth_manager(tmp_path, store)
        try:
            await self._seed_client_info(store)
            self._stub_token_endpoint(monkeypatch)
            storage = McpTokenStorage(self.URL, store)
            assert storage.grant_marker().witness_at is None, "nothing has stood up yet"

            provider = build_oauth_provider(
                self.URL, manager._configs["dd"], store=store, endpoints=self._endpoints()
            )
            async with provider.context.lock:
                await provider._initialize()
            gen = provider.async_auth_flow(
                httpx.Request("POST", self.URL, content=b"payload")
            )
            try:
                request = await gen.__anext__()
                retried = await gen.asend(httpx.Response(401, request=request))
                assert retried is not None, "the flow did not recover from the 401"
                # The resource ACCEPTS the recovered token on the retry. A
                # StopAsyncIteration here is the flow ENDING on that accepted
                # response — a successful request's normal conclusion, not a
                # failure — so it is tolerated rather than asserted away.
                try:
                    await gen.asend(httpx.Response(200, request=retried))
                except StopAsyncIteration:
                    pass
            finally:
                await gen.aclose()

            marker = storage.grant_marker()
            assert marker is not None and marker.witness_at is not None, (
                "a recovered challenge in place recorded no success witness, so a "
                "blocked peer learns nothing from a session that never reconnects"
            )
        finally:
            await manager.disconnect_all()
            store.close()


def test_the_retry_rule_still_sees_a_tombstone() -> None:
    """The witness remediation must not have narrowed the rule to two axes.

    Before the witness existed the rule was ``marker == known`` over the whole
    tuple, so a peer tombstoning the grant we blocked on bought one attempt. The
    first cut of the two-axis rule compared only the stamp and the witness and
    silently dropped that. It is inert through the real store today — the store
    strips ``grant_dead_at`` on write (the deferred tombstone finding) — which is
    exactly why the end-to-end guards cannot pin it, and why this one tests the
    rule directly: the strip's fix must inherit an axis that still works.
    """
    from local_operator.mcp.auth import GrantMarker
    from local_operator.mcp.manager import _grant_change_is_evidence

    known = GrantMarker(500.0, False, 900.0)
    assert _grant_change_is_evidence(GrantMarker(500.0, True, 900.0), known) is True
    assert _grant_change_is_evidence(GrantMarker(500.0, False, 900.0), known) is False


class TestAttemptMarkerUnknownSemantics:
    """``None`` (the store was unreadable) is not "the caller did not say".

    Review round 1, minor-1. ``_block_on_auth`` used one value for both, so an
    attempt whose pre-connect read FAILED fell through to a read taken after the
    connect failed — the exact post-failure read this feature removes, restored
    for precisely that case.

    With the attempt record, "the caller did not say" is reachable only for a
    caller that never watched its record fill (the seam is now the FIRST thing an
    attempt does, so a connect that reaches ``_connect_server`` always records
    something — and a read that FAILS records ``None``, which is a value). These
    are therefore unit tests of the resolution rule itself; the guard tests above
    cover the recorded cases end to end through the real seam.
    """

    URL = TestAuthBlockRevalidation.URL

    @staticmethod
    def _fresh_grant(store: Any) -> Any:
        return TestAuthBlockRevalidation._write_fresh_grant(store)

    @pytest.mark.asyncio
    async def test_an_unreadable_attempt_read_does_not_become_a_post_failure_read(
        self, tmp_path: Path
    ) -> None:
        """A recorded ``None`` keeps the good marker; it never re-reads the store.

        The block is re-taken while a peer's NEW grant is already on disk — the
        only state where the two readings differ — so adopting the peer's grant
        here is exactly the unfalsifiable block the feature exists to remove.
        """
        store = TestAuthBlockRevalidation._real_store(tmp_path, obtained_at=1000.0)
        manager = TestAuthBlockRevalidation._oauth_manager(tmp_path, store)
        try:
            # A first ordinary block, as a real arm would have taken it.
            manager._block_on_auth("dd", (1000.0, False, None))
            assert manager._auth_grant_marker.get("dd") == (1000.0, False, None)

            # The attempt's own read failed (a store hiccup) AND a peer re-auths
            # before the block is taken. ``None`` is that recorded fact.
            await self._fresh_grant(store)
            manager._block_on_auth("dd", None)
            assert manager._auth_grant_marker.get("dd") == (1000.0, False, None), (
                "an unreadable attempt read fell through to a read taken at "
                "block time and adopted the peer's grant"
            )
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_a_caller_that_cannot_say_falls_back_to_reading_the_store(
        self, tmp_path: Path
    ) -> None:
        """The sentinel means "cannot say", and only that, reads at block time.

        Preserved deliberately: it is the pre-feature behaviour, and it is still
        correct whenever no grant was written during the attempt. What must not
        happen is a RECORDED unreadable read (``None``) sharing this value, and
        what must not happen either is the sentinel escaping into the durable
        state — so both halves are asserted here.
        """
        from local_operator.mcp.manager import (
            _MARKER_NOT_RECORDED,
            _AttemptRecord,
            _NotRecorded,
        )

        store = TestAuthBlockRevalidation._real_store(tmp_path, obtained_at=1000.0)
        manager = TestAuthBlockRevalidation._oauth_manager(tmp_path, store)
        try:
            # The sentinel cannot be mistaken for a marker, and a fresh record
            # starts as "not recorded" rather than as an unreadable read.
            assert isinstance(_MARKER_NOT_RECORDED, _NotRecorded)
            assert _MARKER_NOT_RECORDED is not None
            assert _MARKER_NOT_RECORDED != (0.0, False, None)
            assert _AttemptRecord().marker is _MARKER_NOT_RECORDED

            # A caller that cannot say: the store is the evidence.
            manager._block_on_auth("dd")
            assert manager._auth_grant_marker.get("dd") == (500.0, False, None)

            # …and once the store has moved, that same call adopts the new
            # grant. It never records the sentinel itself: the durable marker is
            # always a pair or ``None``.
            await self._fresh_grant(store)
            manager._block_on_auth("dd")
            adopted = manager._auth_grant_marker.get("dd")
            assert adopted is not None
            assert adopted != (500.0, False, None)
            assert not isinstance(adopted, _NotRecorded)
        finally:
            store.close()


class TestAuthBlockClearsWhereverAServerHeals:
    """Review round 1, blocker-1: an auth block must not outlive its condition.

    The block is durable state, and durable state that survives the condition
    that justified it is the exact defect this feature exists to fix. Clearing
    it only at the user-initiated sites (``connect_configured_server``,
    ``reconnect_server``, ``_drop_removed_servers``) missed the route users
    actually take — ``/mcp reload`` — leaving a server ``connected`` while still
    blocked. Its next disconnect for an ORDINARY reason (server process died, a
    network blip) was then abandoned by the ``_schedule_reconnect`` guard, and
    ``revalidate_auth_blocked`` could never rescue it: the grant is valid, so
    its marker never moves again. Permanently dead on a healthy grant.

    ``_register_connection`` is the single choke point every route to a live
    connection passes through — the same argument ``_fire_recovery``'s docstring
    makes for living there — so the clear belongs there and nowhere else.
    """

    URL = "https://reload.example/mcp"

    @staticmethod
    def _project(tmp_path: Path) -> Path:
        """A real config file, so ``reload()`` genuinely re-finds the server."""
        (tmp_path / ".local-operator").mkdir(exist_ok=True)
        (tmp_path / ".local-operator" / "mcp.json").write_text(
            '{"mcpServers": {"dd": {"type": "http",'
            f' "url": "{TestAuthBlockClearsWhereverAServerHeals.URL}",'
            ' "auth": {"type": "oauth"}}}}',
            encoding="utf-8",
        )
        return tmp_path

    @pytest.mark.asyncio
    async def test_a_reload_heal_leaves_no_block_and_the_next_drop_reconnects(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The full sequence: auth failure -> /mcp reload -> ordinary drop."""
        from local_operator.mcp.auth import McpAuthRequiredError

        self._project(tmp_path)
        manager = McpManager(str(tmp_path))
        manager.on_incident = lambda server, reason: None
        healthy = {"v": False}

        async def connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
            if not healthy["v"]:
                raise McpAuthRequiredError(self.URL)
            return _make_conn(name, cfg)

        monkeypatch.setattr(manager, "_connect_server", connect)
        try:
            await manager.discover_and_connect()
            assert manager.auth_blocked("dd") is True
            assert manager.get_connection_status("dd") == "auth-required"

            # The user re-auths elsewhere and reloads, which is the ordinary way
            # a running session picks up a new grant.
            healthy["v"] = True
            await manager.reload()
            assert manager.get_connection_status("dd") == "connected"
            assert manager.auth_blocked("dd") is False, "a connected server is still blocked"
            # …and the stale startup error stops being reported with it.
            assert "dd" not in manager.startup_failures()

            # An ORDINARY disconnect, nothing to do with auth: it must reconnect.
            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", counting)
            manager._handle_disconnect("dd", manager.get_connection("dd"))
            for _ in range(200):
                await asyncio.sleep(0.01)
                if manager.get_connection_status("dd") == "connected":
                    break
            assert attempts == ["dd"], "an ordinary drop was abandoned, not retried"
            assert manager.get_connection_status("dd") == "connected"
        finally:
            await manager.disconnect_all()

    @pytest.mark.asyncio
    async def test_an_in_gate_auth_failure_blocks_like_the_after_gate_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review round 1, major-1: the fourth auth arm.

        ``_connect_round``'s own auth arm handles failures that beat the 250 ms
        gate. It recorded a startup failure and settled the waiter but never
        blocked, so the IDENTICAL failure was revalidatable or not purely by
        whether it was fast. Fast is the common case after the first failure,
        not the exotic one: a tombstoned grant short-circuits before any POST
        and endpoint discovery is cached process-wide, so every later reload
        fails warm.
        """
        from local_operator.mcp.auth import McpAuthRequiredError
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=self.URL, auth=MCPAuthConfig(type="oauth"))

        async def fast_auth_failure(name: str, cfg: Any, **_: Any) -> ServerConnection:
            raise McpAuthRequiredError(self.URL)  # immediate: wins the gate race

        monkeypatch.setattr(manager, "_connect_server", fast_auth_failure)
        try:
            result = await manager._connect_round({"dd": cfg}, {"dd": "global"})
            assert "dd" in result.errors
            assert manager.auth_blocked("dd") is True
            assert manager.get_connection_status("dd") == "auth-required"
        finally:
            await manager.disconnect_all()


class TestAnUnreadableGrantMarkerIsNotAChangedGrant:
    """Review round 1, blocker-2 (and QA's Q-1): the degrade path must not storm.

    ``_grant_marker`` used to degrade to the VALUE ``(0.0, False)``, which is
    indistinguishable from a real marker and participates in the equality test.
    So a store failing on alternating ticks made the marker appear to oscillate
    ``real -> (0.0, False) -> real``, and EVERY tick counted as "a peer
    re-authed" and spent a refresh token — across nine processes, against a
    provider running refresh-token reuse detection, which answers by revoking
    the entire token family. That is §7 risk 1 reached through the degrade path.

    QA measured the failure as unreachable under ordinary contention (0 degraded
    reads in 208k reads against 22.6k concurrent writes; WAL plus
    ``busy_timeout`` absorbs it), so this needs a genuinely broken store. That
    is a reason to keep the guard cheap, not a reason to omit it: the shape is
    one widened ``except`` away from being live.
    """

    URL = "https://flaky.example/mcp"

    @staticmethod
    def _blocked_manager(tmp_path: Path, store: Any) -> McpManager:
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        manager = McpManager(str(tmp_path), auth_store=store)
        manager._configs["dd"] = MCPHttpServerConfig(
            url=TestAnUnreadableGrantMarkerIsNotAChangedGrant.URL,
            auth=MCPAuthConfig(type="oauth"),
        )
        manager._sources["dd"] = "global"
        return manager

    @staticmethod
    def _store(tmp_path: Path) -> Any:
        from local_operator.mcp.auth import MCP_OAUTH_PROVIDER, TOKENS_OBTAINED_AT_KEY
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(str(tmp_path / "auth.db"))
        store.upsert_credential(
            MCP_OAUTH_PROVIDER,
            {
                "project_id": TestAnUnreadableGrantMarkerIsNotAChangedGrant.URL,
                "tokens": {
                    "access_token": "A",
                    "refresh_token": "R1",
                    "token_type": "Bearer",
                    "expires_in": 28800,
                },
                TOKENS_OBTAINED_AT_KEY: 1000.0,
            },
        )
        return store

    @pytest.mark.asyncio
    async def test_a_flapping_store_buys_zero_connect_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The oscillation repro: alternating read failures, grant never changed."""
        from local_operator.mcp.auth import McpAuthRequiredError

        store = self._store(tmp_path)
        manager = self._blocked_manager(tmp_path, store)
        try:

            async def failing(name: str, cfg: Any, **_: Any) -> ServerConnection:
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", failing)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager.auth_blocked("dd") is True

            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", counting)

            # Every other read raises, as an intermittently broken store does.
            real = store.list_credentials
            calls = {"n": 0}

            def flaky(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                if calls["n"] % 2 == 0:
                    raise RuntimeError("database is locked")
                return real(*args, **kwargs)

            monkeypatch.setattr(store, "list_credentials", flaky)
            for _ in range(10):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == [], (
                "an unreadable store was mistaken for a new grant: "
                f"{len(attempts)} refresh-token spends over 10 ticks"
            )
            assert manager.auth_blocked("dd") is True
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_totally_dead_store_never_retries_and_still_heals_later(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown at block time is not evidence either, but it is not a trap.

        Blocking while the store is unreadable records ``None``. The next
        successful read is adopted as the BASELINE rather than treated as
        movement — we never knew which grant we failed on, so it is not evidence
        of a new one — and a genuine later change still heals.
        """
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import McpAuthRequiredError, McpTokenStorage

        store = self._store(tmp_path)
        manager = self._blocked_manager(tmp_path, store)
        try:
            real = store.list_credentials
            monkeypatch.setattr(
                store,
                "list_credentials",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no such table")),
            )

            async def failing(name: str, cfg: Any, **_: Any) -> ServerConnection:
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", failing)
            await manager._reconnect("dd", 0.0, manager._epoch)
            assert manager._auth_grant_marker["dd"] is None, "a failed read stored a value"

            attempts: list[str] = []

            async def counting(name: str, cfg: Any, **_: Any) -> ServerConnection:
                attempts.append(name)
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", counting)

            # The store comes back. That is NOT movement — adopt the baseline.
            monkeypatch.setattr(store, "list_credentials", real)
            for _ in range(5):
                assert await manager.revalidate_auth_blocked() == []
            assert attempts == [], "recovering from an unreadable store counted as a new grant"
            assert manager._auth_grant_marker["dd"] is not None, "no baseline was adopted"

            # A REAL new grant on top of that baseline still heals, so the
            # conservative choice costs one poll cycle, never the recovery.
            storage = McpTokenStorage(self.URL, store)
            await storage.set_tokens(
                OAuthToken(
                    access_token="FRESH",
                    refresh_token="R2",
                    token_type="Bearer",
                    expires_in=28800,
                )
            )
            assert await manager.revalidate_auth_blocked() == ["dd"]
            assert attempts == ["dd"]
            assert manager.get_connection_status("dd") == "connected"
        finally:
            await manager.disconnect_all()
            store.close()

    @pytest.mark.asyncio
    async def test_a_missing_row_is_a_known_marker_not_an_unreadable_one(
        self, tmp_path: Path
    ) -> None:
        """Absent and unreadable must not collapse into one answer.

        ``McpTokenStorage._read_row`` deliberately returns ``None`` for both,
        which is right for its other callers (either way, start a fresh flow)
        and wrong here: an absent grant is a stable marker that MOVES when a
        peer writes one — that is how a never-authorized server heals — while an
        unreadable store is no information at all. This is why ``grant_marker``
        reads the store itself rather than reusing ``_read_row``.
        """
        from local_operator.mcp.auth import McpTokenStorage
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore(str(tmp_path / "auth.db"))
        try:
            marker = McpTokenStorage(
                "https://never-authed.example/mcp", store
            ).grant_marker()
            assert marker == (0.0, False, None)
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_the_heal_reports_connecting_and_settles_a_parked_execute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Review round 1, minor-2: publish a waiter before the poller's connect.

        A real OAuth connect takes seconds. Without a waiter the server reports
        ``disconnected`` for that whole window — a visible flicker on the way to
        healing, and worse, a deferred execute parked on this server fails
        instead of riding the heal it is one await away from.
        """
        from mcp.shared.auth import OAuthToken

        from local_operator.mcp.auth import McpAuthRequiredError, McpTokenStorage

        store = self._store(tmp_path)
        manager = self._blocked_manager(tmp_path, store)
        try:

            async def failing(name: str, cfg: Any, **_: Any) -> ServerConnection:
                raise McpAuthRequiredError(self.URL)

            monkeypatch.setattr(manager, "_connect_server", failing)
            await manager._reconnect("dd", 0.0, manager._epoch)

            gate = asyncio.Event()
            seen: list[str] = []

            async def slow_connect(name: str, cfg: Any, **_: Any) -> ServerConnection:
                seen.append(manager.get_connection_status(name))
                await gate.wait()
                return _make_conn(name, cfg)

            monkeypatch.setattr(manager, "_connect_server", slow_connect)
            storage = McpTokenStorage(self.URL, store)
            await storage.set_tokens(
                OAuthToken(
                    access_token="F", refresh_token="R2", token_type="Bearer", expires_in=28800
                )
            )

            tick = asyncio.create_task(manager.revalidate_auth_blocked())
            for _ in range(200):
                await asyncio.sleep(0.005)
                if seen:
                    break
            # A deferred execute arriving mid-heal parks on the waiter…
            parked = asyncio.create_task(manager.wait_for_connection("dd"))
            await asyncio.sleep(0)
            assert (
                manager.get_connection_status("dd") == "connecting"
            ), "the server reported a terminal state while its heal was in flight"
            gate.set()
            assert await tick == ["dd"]
            # …and rides the heal instead of failing.
            assert (await asyncio.wait_for(parked, timeout=5)) is manager.get_connection("dd")
        finally:
            await manager.disconnect_all()
            store.close()


class TestRefreshRefusalCopy:
    """The exact RENDERED strings for the refusal reasons and the auth line.

    Round 1 split one message into three truthful sentences and design review
    round 1 (D1/D2) showed why that was not enough: every one of those sentences
    opens with ``MCP OAuth token refresh for <full server URL>``, which is ~55
    of the toast card's 58 content cells, so the distinguishing clause was
    always the part tail-truncated away and all of them rendered byte-identically
    at 100 columns and below. The split now lives on the exception as a stable
    reason CODE and the manager composes the short text from it, so these
    assertions are made against the RENDERED line at both real card widths —
    not against the code's idea of the sentence, which is exactly the mistake
    that let four different refusals look the same.

    Two rounds later the same seam carries the LOCAL shapes (unsent, unattributed)
    that must never borrow the endpoint's "the server returned no token", and the
    auth requirement's three pinned lines (D9), which are asserted here as the
    painted rows rather than as source strings.

    The 44-column case is the tight one and the reason the wording is this
    short: ``✗ failed: notion — `` spends 19 of the 36 available cells, leaving
    17 for the reason, so each reason must be DISTINGUISHABLE inside its first
    17 cells.

    Every pinned row below carries the row's own glyph (design review round 2,
    D2-2). It is 2 of the card's 58 content cells, and that is the whole reason
    these strings moved: the row a user reads is ``✗ `` + the copy, so the
    boundaries are re-derived at the PAINTED width (58 → 56 for the copy, 36 →
    34) rather than the pre-glyph 58/36 (review round 4, R4-3; round 5, R5-3).
    Two consequences are pinned deliberately rather than absorbed: the auth
    family's 57-cell whole row no longer fits the 58-cell card and sheds its
    reason, and at 44 columns the reauth command's name argument is truncated
    one cell from the end (the D9 constraint at that width, priced below).
    """

    URL = "https://mcp.example.com/v1/mcp"

    #: Rendered at 100 columns (58 content cells) and at 44 (36). The row the
    #: composer returns INCLUDES the card row's own ``✗ `` glyph (D2-2), which is
    #: why the truncation boundaries sit two cells earlier than they did before
    #: the glyph was added — measured on the painted row, not shaved to fit.
    EXPECTED = {
        REFRESH_REFUSAL_LOCK: (
            "✗ failed: notion — another session is refreshing",
            "✗ failed: notion — another session…",
        ),
        REFRESH_REFUSAL_INFLIGHT: (
            "✗ failed: notion — refresh still in progress",
            "✗ failed: notion — refresh still in…",
        ),
        REFRESH_REFUSAL_ENDPOINT: (
            "✗ failed: notion — the server returned no token",
            "✗ failed: notion — the server retur…",
        ),
        REFRESH_REFUSAL_UNREACHABLE: (
            "✗ failed: notion — cannot reach the server",
            "✗ failed: notion — cannot reach the…",
        ),
    }

    @staticmethod
    def _toast_failure_line(reason: str, cells: int, name: str = "notion") -> str:
        """The failure row the REAL toast composer paints, for ``reason``.

        ``name`` defaults to the 6-cell pseudonym every D9 pin was measured on;
        the long-name cases pass the server name they are about, because the
        row's budget depends on its width (D11).
        """
        from local_operator.session.mcp_status import McpStartupOutcome
        from local_operator.tui.widgets.toast import format_mcp_startup

        outcome = McpStartupOutcome(configured=(name,), failures={name: reason})
        payload = format_mcp_startup(outcome, max_cells=cells)
        assert payload is not None
        return payload[0].plain.split("\n")[1]

    def test_the_reason_codes_compose_short_copy_with_no_url_or_internals(self) -> None:
        """Each code maps to its own sentence, and the sentence is safe to render."""
        from local_operator.mcp.auth import McpRefreshContendedError

        rendered = McpManager._auth_failure_text  # readability
        texts: dict[str, str] = {}
        for reason in self.EXPECTED:
            exc = McpRefreshContendedError(self.URL, reason_code=reason)
            texts[reason] = rendered("notion", exc)

        assert len(set(texts.values())) == 4, texts
        for reason, text in texts.items():
            assert "http" not in text, (reason, text)
            # The internals the design review named: a user cannot act on either.
            for jargon in ("lock", "rotation", "token refresh for"):
                assert jargon not in text, (reason, text)
            # No promise of a retry: the startup gate schedules none, so a card
            # that said "retrying" would be untrue at the surface most users see.
            assert "retry" not in text.lower() and "retrying" not in text.lower(), text

    def test_the_toast_card_renders_every_reason_distinguishably_at_both_widths(self) -> None:
        """The whole point of the fix, asserted on the painted line.

        A green unit test on ``str(exc)`` is what let this defect through round
        1: the strings differed, and the CARD did not. So this pins the exact
        rendered row at 58 cells (a 100-column terminal) and 36 (44 columns) —
        the row the widget paints, glyph included.
        """
        from local_operator.mcp.auth import McpRefreshContendedError

        for cells_index, cells in enumerate((58, 36)):
            lines = []
            for reason, expected in self.EXPECTED.items():
                reason_text = McpManager._auth_failure_text(
                    "notion", McpRefreshContendedError(self.URL, reason_code=reason)
                )
                line = self._toast_failure_line(reason_text, cells)
                assert line == expected[cells_index], (reason, cells, line)
                lines.append(line)
            assert len(set(lines)) == 4, (cells, lines)

    def test_an_unknown_reason_code_never_falls_back_to_the_verbose_sentence(self) -> None:
        """A code this build does not know must still render short and URL-free.

        A newer peer process can write a reason we cannot name. Falling back to
        ``str(exc)`` there would put the ~55-cell URL preamble back on the card
        — the defect this mapping exists to remove — so the fallback states only
        what is true of every refusal in the set.
        """
        from local_operator.mcp.auth import McpRefreshContendedError

        exc = McpRefreshContendedError(self.URL, reason_code="a-code-from-the-future")
        text = McpManager._auth_failure_text("notion", exc)
        assert text == "the refresh did not complete"
        assert "http" not in text
        line = self._toast_failure_line(text, 36)
        assert line == "✗ failed: notion — the refresh did…"

    #: The three auth lines design review round 2 (D9) pinned CHARACTER FOR
    #: CHARACTER, with the composed toast row each produces at the two widths
    #: the designer measured (58 content cells for a 100-column terminal, 36 for
    #: 44). The `run ` wrapper is gone because it cost four cells — exactly the
    #: shortfall that pushed the reason past the card's clamp at 100 columns and
    #: cut the server name mid-word at 44 — and the bare slash command is the
    #: app's own habit for a runnable command (the splash, the usage panel).
    #: The rows below are the POST-glyph measurements: `unconfirmed` (57 cells
    #: whole, now 59 with the glyph) sheds its reason at 100 columns, the other
    #: two keep it, and at 44 the name argument loses its last cell.
    AUTH_LINE_EXPECTED = {
        "unconfirmed": (
            "/mcp reauth notion — refresh unconfirmed",
            "✗ failed: notion — /mcp reauth notion…",
            "✗ failed: notion — /mcp reauth noti…",
        ),
        "default": (
            "/mcp reauth notion — sign-in expired",
            "✗ failed: notion — /mcp reauth notion — sign-in expired",
            "✗ failed: notion — /mcp reauth noti…",
        ),
        "no-grant": (
            "/mcp login notion to authorize",
            "✗ failed: notion — /mcp login notion to authorize",
            "✗ failed: notion — /mcp login notio…",
        ),
    }

    #: The two LOCAL refusal reasons (review round 3, M2). Neither may describe a
    #: request that never went out, so neither may borrow the endpoint wording.
    LOCAL_REFUSAL_EXPECTED = {
        REFRESH_REFUSAL_UNSENT: (
            "✗ failed: notion — no stored token to send",
            "✗ failed: notion — no stored token…",
        ),
        REFRESH_REFUSAL_UNATTRIBUTED: (
            "✗ failed: notion — the refresh did not complete",
            "✗ failed: notion — the refresh did…",
        ),
    }

    def test_the_three_auth_lines_render_exactly_as_pinned_at_both_widths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """D9, asserted on the PAINTED row at 58 and 36 content cells.

        The old assertion pinned the CLIPPED row (``… — run /mcp reauth notion —
        refresh unconfi…``), which was the finding: the word that says WHY was
        off the card on the first surface a user reads, and at 44 columns the
        server name the command hands over was itself cut mid-word.
        """
        from local_operator.mcp.auth import (
            McpAuthRequiredError,
            McpRefreshUnconfirmedError,
        )

        cases = {
            # Unconfirmed carries the reason on the error, so it never looks the
            # grant up; the other two go through the store lookup, which is
            # stubbed here rather than read from the developer's machine.
            "unconfirmed": (McpRefreshUnconfirmedError(self.URL), True),
            "default": (McpAuthRequiredError(self.URL), True),
            "no-grant": (McpAuthRequiredError(self.URL), False),
        }
        for key, (exc, has_grant) in cases.items():
            monkeypatch.setattr(
                "local_operator.mcp.auth.server_has_stored_grant",
                lambda url, store=None, _g=has_grant: _g,
            )
            text, wide, narrow = self.AUTH_LINE_EXPECTED[key]
            assert McpManager._auth_required_text("notion", exc) == text, key
            assert text.count("—") <= 1, (key, text)  # D4: not a chain of dashes
            assert text.startswith("/mcp "), (key, text)
            assert "run /mcp" not in text, (key, text)
            assert (
                self._toast_failure_line(text, 58) == wide
            ), f"{key}: D9 measured {wide!r} at 100 columns"
            assert len(wide) <= 58, (key, wide)
            assert self._toast_failure_line(text, 36) == narrow, (key, narrow)
            if key != "no-grant":
                # The D9 constraint at 44 columns, priced for the glyph: the row
                # spends 2 of its 36 content cells on ``✗ ``, so what survives
                # whole is the COMMAND VERB (``/mcp reauth`` — typeable with the
                # name this same row's head carries) while the name's last cell
                # truncates. The alternative the ladder rejects — dropping to
                # the bare command to keep the name whole — would cost the row
                # its server, the D2-3 harm, and is a copy decision for the
                # design round rather than a pin to shave (review rounds 4/5).
                assert "/mcp reauth" in narrow, (key, narrow)
                assert "failed: notion" in narrow, (key, narrow)

    #: The D11 rows, measured on the head that fixes them. Both names are this
    #: project's own server names: ``minerva-qa`` (10 cells) is the one
    #: ``test_turn_abandoned.py`` uses, and ``launchdarkly`` (12) the one this
    #: file already copies. The wide row keeps the whole command and sheds the
    #: reason; the narrow one is asserted VERBATIM as the base renders it, which
    #: is the recorded deferral (see the test's docstring). Both were re-measured
    #: with the row's ``✗ `` glyph in the budget (review rounds 4/5).
    LONG_NAME_EXPECTED = {
        "minerva-qa": (
            "✗ failed: minerva-qa — /mcp reauth minerva-qa…",
            "✗ failed: minerva-qa — /mcp reauth…",
        ),
        "launchdarkly": (
            "✗ failed: launchdarkly — /mcp reauth launchdarkly…",
            "✗ failed: launchdarkly — /mcp reaut…",
        ),
    }

    def test_a_long_name_sheds_the_reason_rather_than_clipping_the_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Design review round 3, D11: the D9 pin held only for 6-cell names.

        The row is ``failed: <name> — <command> — <reason>`` and the command
        repeats the name, so it needs ``45 + 2n`` cells against the card's 58
        and fits only while the name is 6 cells. ``minerva-qa`` and
        ``launchdarkly`` are 10 and 12, and the clamp ate the END of the line:
        the REASON at 100 columns (``— refresh unc…``) and the command's own name
        argument at 44 (``/mcp reauth mi…`` — a command that errors if followed).
        The budget now decides what is SHED: the command survives whole and the
        reason is dropped, marked with the ``…`` the 44-column card has always
        shown (D9 accepted it as marking the shed reason). The reason is the
        right part to lose — ``/mcp`` and the durable transcript notice carry it
        whole.
        """
        from local_operator.mcp.auth import McpRefreshUnconfirmedError

        monkeypatch.setattr(
            "local_operator.mcp.auth.server_has_stored_grant", lambda url, store=None: True
        )
        for name, (wide, narrow) in self.LONG_NAME_EXPECTED.items():
            text = McpManager._auth_required_text(name, McpRefreshUnconfirmedError(self.URL))
            assert text == f"/mcp reauth {name} — refresh unconfirmed", name
            rendered = self._toast_failure_line(text, 58, name)
            assert rendered == wide, (name, rendered)
            # The whole command with its name, and NOTHING of the reason: what
            # this removes is a half-rendered reason, which the user cannot act
            # on and which the other two surfaces state in full.
            assert f"/mcp reauth {name}" in rendered, (name, rendered)
            assert "refresh" not in rendered and "unconfirmed" not in rendered, rendered
            assert rendered.count("—") == 1, rendered  # D4: not a chain of dashes
            assert len(rendered) <= 58, (name, rendered)
            # At 44 columns the row's head plus the command is ``25 + 2n`` cells
            # against 36 (the glyph's 2 included), so no composition of this
            # line fits for names >= 6 without a second row or a different card
            # shape — both layout decisions this change does not make (D11
            # resolution 2). The base row is pinned verbatim instead, so a future
            # regression is visible rather than silent.
            assert self._toast_failure_line(text, 36, name) == narrow, (name, narrow)

    def test_the_shed_boundary_is_the_sixth_cell_of_the_name(self) -> None:
        """The boundary the shed starts at, pinned from BOTH sides.

        The rung-1 row is ``47 + 2n`` cells against the card's 58 — 45 for
        ``failed: <name> — <command-with-name> — <reason>`` plus the row's own
        ``✗ `` glyph — so ``n <= 5`` is the range where the whole row fits.
        ``slack`` (5) keeps its reason and is 57 cells; ``notion`` (6) is 59 and
        sheds it. Both names are real servers in this repo's own config
        vocabulary, so the boundary is asserted on two names a user would
        actually read. The glyph moved this boundary one cell left: before it,
        a 6-cell name still fitted (review rounds 4/5).
        """
        kept = self._toast_failure_line("/mcp reauth slack — refresh unconfirmed", 58, "slack")
        assert kept == "✗ failed: slack — /mcp reauth slack — refresh unconfirmed"
        assert len(kept) == 57, kept
        shed = self._toast_failure_line("/mcp reauth notion — refresh unconfirmed", 58, "notion")
        assert shed == "✗ failed: notion — /mcp reauth notion…"
        assert len(shed) == 38, shed

    def test_the_local_refusals_never_blame_a_server(self) -> None:
        """The endpoint copy is false about a request that was never made.

        Review round 3, M2: the exchange returns ``"failed"`` for shapes that
        produced no request at all (an empty row), and the manager's default
        mapped that to the endpoint code — "the server returned no token" —
        which is untrue about the WIRE (nothing was sent) and about the SERVER
        (it was never asked). Each local shape now carries its own code, and
        both still fit the 44-column card's 17-cell reason budget (36 content
        cells minus the row's 19-cell ``✗ failed: notion — `` head).
        """
        from local_operator.mcp.auth import McpRefreshContendedError

        endpoint = McpManager._auth_failure_text(
            "notion", McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_ENDPOINT)
        )
        rendered: dict[str, str] = {}
        for reason, (wide, narrow) in self.LOCAL_REFUSAL_EXPECTED.items():
            text = McpManager._auth_failure_text(
                "notion", McpRefreshContendedError(self.URL, reason_code=reason)
            )
            rendered[reason] = text
            assert "server" not in text, (reason, text)
            assert "http" not in text, (reason, text)
            assert text != endpoint, (reason, text)
            assert self._toast_failure_line(text, 58) == wide, (reason, wide)
            assert self._toast_failure_line(text, 36) == narrow, (reason, narrow)
        assert len(set(rendered.values())) == 2, rendered

    def test_the_log_sentence_still_carries_the_url_and_the_technical_reason(self) -> None:
        """``str(exc)`` stays verbose for the logs, which is the other half of D1.

        The reason code exists so the COPY can be short; the sentence is not
        deleted, just moved off the user's surfaces — support needs the URL and
        the mechanism when reading a log after an incident.
        """
        from local_operator.mcp.auth import (
            REFRESH_REFUSAL_ENDPOINT,
            REFRESH_REFUSAL_INFLIGHT,
            REFRESH_REFUSAL_LOCK,
            REFRESH_REFUSAL_UNATTRIBUTED,
            REFRESH_REFUSAL_UNREACHABLE,
            REFRESH_REFUSAL_UNSENT,
            McpRefreshContendedError,
        )

        assert str(McpRefreshContendedError(self.URL)) == (
            f"MCP OAuth token refresh for {self.URL} was skipped: another session "
            "holds the refresh lock"
        )
        assert str(McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_INFLIGHT)) == (
            f"MCP OAuth token refresh for {self.URL} did not finish in time; a rotation "
            "is kept if the response lands"
        )
        # A pre-send transport failure is NOT a rejection by the authorization
        # server: it never reached one (review round 2, minor 1).
        unreachable = str(
            McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_UNREACHABLE)
        )
        assert "unreachable" in unreachable
        assert "no token was presented" in unreachable
        assert "rejected" not in unreachable
        endpoint = str(McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_ENDPOINT))
        assert "rejected" not in endpoint
        # The lock sentence still says the lock: it is the accurate log line.
        assert "holds the refresh lock" in str(
            McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_LOCK)
        )
        # The two LOCAL codes say what happened to THEM instead of borrowing
        # that sentence, which was untrue for both — neither ran into another
        # session's lock (QA round 3, Q2; review round 4, N4.1 is its sibling in
        # the proactive site's outcome tuple).
        unsent = str(McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_UNSENT))
        assert "was not sent" in unsent
        assert "nothing to present" in unsent
        assert "holds the refresh lock" not in unsent
        unattributed = str(
            McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_UNATTRIBUTED)
        )
        assert "cannot be attributed" in unattributed
        assert "holds the refresh lock" not in unattributed
        # …and the split is log-only by construction: the rendered copy is
        # composed from the CODE, so neither sentence can reach a user surface.
        assert (
            McpManager._auth_failure_text(
                "notion", McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_UNSENT)
            )
            == "no stored token to send"
        )
        assert (
            McpManager._auth_failure_text(
                "notion",
                McpRefreshContendedError(self.URL, reason_code=REFRESH_REFUSAL_UNATTRIBUTED),
            )
            == "the refresh did not complete"
        )

    @pytest.mark.asyncio
    async def test_an_unacknowledged_refusal_is_re_voiced_as_an_auth_requirement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The re-voice: a spent-possibly token must NOT be retried on backoff.

        Contention is transient and retries. A token that may already be spent is
        the opposite: retrying IS the reuse-detection POST, so this refusal has to
        arrive as an auth requirement — auth block, actionable toast, abandoned
        auto-reconnect — while carrying the truthful reason.
        """
        from local_operator.mcp.auth import (
            REFRESH_CONTENTION,
            REFRESH_REFUSAL_UNCONFIRMED,
            McpRefreshUnconfirmedError,
        )
        from local_operator.mcp.config import MCPAuthConfig, MCPHttpServerConfig

        url = "https://srv.example/mcp"
        manager = McpManager(str(tmp_path))
        cfg = MCPHttpServerConfig(url=url, auth=MCPAuthConfig(type="oauth"))
        manager._configs["dd"] = cfg

        async def bare_cancel(*_a: Any, **_kw: Any) -> Any:
            # Exactly what the transport delivers: a CancelledError with no
            # cancelling count, i.e. not an external cancellation.
            raise asyncio.CancelledError()

        monkeypatch.setattr(manager, "_open_transport_and_session", bare_cancel)
        monkeypatch.setattr(manager, "_ensure_oauth_fresh", lambda *a, **k: asyncio.sleep(0))

        REFRESH_CONTENTION.record(url, REFRESH_REFUSAL_UNCONFIRMED)
        with pytest.raises(McpRefreshUnconfirmedError) as excinfo:
            await manager._connect_server("dd", cfg)
        assert excinfo.value.detail == "refresh unconfirmed"

        # The manager arm for an auth requirement: blocked, not retried.
        scheduled = {"called": False}
        monkeypatch.setattr(
            manager, "_schedule_reconnect", lambda name: scheduled.__setitem__("called", True)
        )
        # Arm a FRESH record: the connect above consumed the first one (single
        # use), and a reconnect is its own refused attempt in reality.
        incidents: list[tuple[str, str]] = []
        manager.on_incident = lambda name, text: incidents.append((name, text))
        REFRESH_CONTENTION.record(url, REFRESH_REFUSAL_UNCONFIRMED)
        # ``_reconnect`` does not re-raise an auth requirement: it blocks on the
        # grant, abandons the ladder and tells the model why.
        await manager._reconnect("dd", 0.0, manager._epoch)
        assert scheduled["called"] is False, "a spent-possibly token must not be retried"
        assert manager.auth_blocked("dd") is True
        assert manager.get_connection_status("dd") == "auth-required"
        # Command-FIRST, with no "MCP authorization failed;" in front of it: the
        # prefix restated the row's own head and pushed the remedy off the front
        # of the line, where it wrapped apart from its server name between 56 and
        # 60 columns, orphaning only the tail at 64 (design review rounds 1-2,
        # D3/Q-F2). The sink payload is what
        # ``journal_mcp_unavailable`` renders verbatim into ``Reason:``.
        assert incidents[-1][1] == "/mcp reauth dd — refresh unconfirmed", incidents

    def test_each_refusal_reason_is_carried_on_the_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason survives the transport with the record, one per refusal.

        Two concurrent connects for the same server can both refuse; the ledger
        holds one record each, so the second is not silently dropped (that reached
        ``_reconnect`` as a bare cancellation and killed the reconnect task
        instead of retrying it).
        """
        from local_operator.mcp.auth import (
            REFRESH_CONTENTION,
            REFRESH_REFUSAL_ENDPOINT,
            REFRESH_REFUSAL_INFLIGHT,
            REFRESH_REFUSAL_LOCK,
        )

        url = "https://srv.example/mcp/two"
        REFRESH_CONTENTION.record(url, REFRESH_REFUSAL_LOCK)
        REFRESH_CONTENTION.record(url, REFRESH_REFUSAL_INFLIGHT)

        assert REFRESH_CONTENTION.pop(url) == REFRESH_REFUSAL_LOCK
        assert REFRESH_CONTENTION.pop(url) == REFRESH_REFUSAL_INFLIGHT
        assert REFRESH_CONTENTION.pop(url) is None

        # A reason is not a boolean: an unarmed server must pop as None, and a
        # reason must never be confused with the un-armed answer.
        REFRESH_CONTENTION.record(url, REFRESH_REFUSAL_ENDPOINT)
        assert REFRESH_CONTENTION.pop(url) == REFRESH_REFUSAL_ENDPOINT


class TestMcpRequestTimeoutDefault:
    """The client-side per-request budget, and why its default is a derived figure.

    These tests exist because the default is not a free choice: it decides
    whether the SAME MCP server is flaky here and reliable under the sibling CLI
    harness, and that asymmetry is invisible from either side alone.

    It has already been wrong once in exactly that direction -- it said 60 s on
    the belief that it matched the sibling, which was true of codex only up to
    about v0.100, while the codex actually installed here (``codex-cli 0.147.0``)
    defaults its tool timeout to 300 s. Hence the equality assertions below: a
    figure this easy to get stale must be pinned where changing it fails a test,
    and derived from the installed sibling rather than from memory.

    The durable fix for the whole class of bug is elsewhere: the workloads that
    care declare a budget in the shared per-run MCP document, so they bind to a
    value we own instead of to either harness's default.
    """

    def test_default_matches_the_installed_sibling_harnesss_per_tool_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default is the installed Codex CLI's per-tool budget, in seconds.

        300 s is ``DEFAULT_TOOL_TIMEOUT`` in ``codex-rs/codex-mcp/src/rmcp_client.rs``
        at the tag matching the installed ``codex-cli 0.147.0``. Both harnesses
        drive the same MCP servers over the same transports, so a tighter default
        here is the one configuration that makes them disagree about a server
        neither of them controls. Pinned as an equality rather than a floor so a
        future edit has to restate the comparison deliberately.
        """
        monkeypatch.delenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", raising=False)
        assert DEFAULT_MCP_TIMEOUT_MS == 300_000.0
        assert resolve_mcp_timeout_s(None) == 300.0

    def test_env_override_and_per_server_config_still_beat_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising the default must not disturb the established precedence.

        env > ``config.timeout`` > default, and ``0`` keeps meaning "off" rather
        than "use the default" -- the distinction the whole precedence chain
        rests on, and the one a careless ``or`` would silently collapse.
        """
        monkeypatch.delenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", raising=False)
        per_server = MCPStdioServerConfig(command="srv", timeout=300_000)
        assert resolve_mcp_timeout_s(per_server) == 300.0

        monkeypatch.setenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", "5000")
        assert resolve_mcp_timeout_s(per_server) == 5.0

        monkeypatch.setenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", "0")
        assert resolve_mcp_timeout_s(per_server) is None

    def test_unparsable_env_value_falls_back_to_the_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", "not-a-number")
        assert resolve_mcp_timeout_s(None) == 300.0

    def test_our_own_zero_still_disables_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``timeout = 0`` keeps meaning "off" -- for OUR key only.

        This is the other half of the asymmetry the config tests pin: a foreign
        ``tool_timeout_sec = 0`` must not disable the bound, but this tool's own
        ``timeout = 0`` still does. Collapsing the two would silently change
        established behaviour for anyone who set it here.
        """
        monkeypatch.delenv("LOCAL_OPERATOR_MCP_TIMEOUT_MS", raising=False)
        assert resolve_mcp_timeout_s(MCPStdioServerConfig(command="srv", timeout=0)) is None
        assert resolve_mcp_timeout_s(MCPStdioServerConfig(command="srv", timeout=-1)) is None
