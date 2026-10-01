"""The session half of §D3: flap vs gone, and what a tick does about each.

The live evidence that shapes these tests: a Datadog monitor armed against an
MCP tool that flapped between turns died the same death twice (~50 minutes,
five consecutive "not in this session's tool set" checks), and then needed a
manual re-arm. Both halves of that are asserted here — the tool surviving an
inventory refresh, and an absent tool being classified as "not reachable right
now" rather than "failed".

No session is opened against a real config: the manager is a fake, and the
session is the shared ``make_session`` fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    FAULT_INVALID_ARGUMENTS,
    FAULT_KEY,
    AgentTool,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.monitors.spec import MonitorSpec
from tests.unit.session.test_session import ScriptedStream, make_session

SERVER = "datadog"


def mcp_name(tool: str = "search_datadog_hosts") -> str:
    from local_operator.mcp.tool_bridge import create_mcp_tool_name

    return create_mcp_tool_name(SERVER, tool)


def mcp_tool(
    name: str | None = None,
    *,
    text: str = "ok",
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> AgentTool:
    """A stand-in MCP tool carrying the read-only hint the verdict gates on."""

    async def _exec(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=name or mcp_name(),
            content=[TextContent(text=error or text)],
            is_error=error is not None,
            details=details,
        )

    return AgentTool(
        name=name or mcp_name(),
        approval_tier="read",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        mcp_annotations={"readOnlyHint": True},
        execute=_exec,
    )


class FakeManager:
    """The subset of ``McpManager`` the monitor path reads.

    ``status_apis`` can be switched off: a fake (or a snapshot manager) without
    the status methods must behave exactly as it did before the availability
    classification existed, which is what the last test pins.
    """

    def __init__(
        self,
        tools: list[AgentTool] | None = None,
        *,
        status: str = "connected",
        settling: bool = False,
        suspended: bool = False,
        servers: list[str] | None = None,
        with_status_apis: bool = True,
    ) -> None:
        self._tools = list(tools or [])
        self._status = status
        self._settling = settling
        self._suspended = suspended
        self._servers = list(servers if servers is not None else [SERVER])
        self._with_status_apis = with_status_apis

    def get_tools(self) -> list[AgentTool]:
        return list(self._tools)

    def get_tool_meta(self, name: str) -> dict[str, Any] | None:
        if any(tool.name == name for tool in self._tools):
            return {"server_name": SERVER, "mcp_tool_name": name, "deferred": False}
        return None

    def get_all_server_names(self) -> list[str]:
        return list(self._servers)

    def get_connection_status(self, name: str) -> str:
        if not self._with_status_apis:
            raise AttributeError("no status APIs")
        return self._status

    def startup_settling(self) -> bool:
        if not self._with_status_apis:
            raise AttributeError("no status APIs")
        return self._settling

    def reconnect_suspended(self, name: str) -> bool:
        if not self._with_status_apis:
            raise AttributeError("no status APIs")
        return self._suspended


def _session(tmp_path: Any, manager: Any = None, tools: list[AgentTool] | None = None) -> Any:
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=tools or [])
    session.mcp_manager = manager
    return session


def _spec(tool: str | None = None, **extra: Any) -> MonitorSpec:
    return MonitorSpec(
        id="m1",
        name="watch",
        tool=tool or mcp_name(),
        arguments={"query": "up"},
        every_ms=60_000,
        created_at=1_756_000_000_000,
        **extra,
    )


@pytest.mark.asyncio
async def test_a_registry_tool_outside_the_inventory_still_resolves(tmp_path: Any) -> None:
    """The core of the flap fix: ``session._tools`` holds only the ACTIVATED
    subset, and one refresh (or one reconnect, or one ``_rebuild_agent_names``)
    drops an armed monitor's tool out of it for good. The manager's registry is
    the stable source, and the monitor resolves through it.
    """
    session = _session(tmp_path, FakeManager([mcp_tool()]))
    try:
        assert not any(tool.name == mcp_name() for tool in session._tools)
        assert session._resolve_monitor_tool(mcp_name()) is not None
        assert session._monitor_availability(mcp_name()) == ("ok", "")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_repeated_refreshes_do_not_lose_the_monitor(tmp_path: Any) -> None:
    """A ``tools_changed`` storm between ticks must not lose the watch: the
    resolution is computed per tick, not latched at arm time.
    """
    manager = FakeManager([mcp_tool()])
    session = _session(tmp_path, manager)
    try:
        for _ in range(5):
            session.refresh_tools([])  # the inventory drops the tool every time
            assert session._resolve_monitor_tool(mcp_name()) is not None
        assert (await session._run_monitor_check(_spec()))["error"] is None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_server_still_connecting_is_unavailable_not_a_failure(tmp_path: Any) -> None:
    manager = FakeManager([], status="connecting", settling=True)
    session = _session(tmp_path, manager)
    try:
        state, detail = session._monitor_availability(mcp_name())
        assert state == "unavailable"
        assert "still connecting" in detail
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "unavailable"
        assert "still connecting" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_auth_required_names_its_own_remedy(tmp_path: Any) -> None:
    manager = FakeManager([], status="auth-required")
    session = _session(tmp_path, manager)
    try:
        state, detail = session._monitor_availability(mcp_name())
        assert state == "unavailable"
        assert "/mcp reauth datadog" in detail
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_suspended_reconnect_is_unavailable(tmp_path: Any) -> None:
    manager = FakeManager([], status="disconnected", suspended=True)
    session = _session(tmp_path, manager)
    try:
        state, detail = session._monitor_availability(mcp_name())
        assert state == "unavailable"
        assert "/mcp reauth datadog" in detail
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_same_monitor_runs_when_the_server_is_restored(tmp_path: Any) -> None:
    """Flap-back needs no re-arm: the next tick simply runs."""
    manager = FakeManager([], status="connecting", settling=True)
    session = _session(tmp_path, manager)
    try:
        first = await session._run_monitor_check(_spec())
        assert first["kind"] == "unavailable"
        manager._tools = [mcp_tool(text="host-a up")]
        manager._status = "connected"
        manager._settling = False
        second = await session._run_monitor_check(_spec())
        assert second.get("error") is None
        assert second["text"] == "host-a up"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unknown_server_prefix_is_gone(tmp_path: Any) -> None:
    """A name whose server is not configured at all is genuinely gone, and
    keeps the strike path (five failures then a disable).
    """
    manager = FakeManager([], servers=["other-server"])
    session = _session(tmp_path, manager)
    try:
        assert session._monitor_availability("mcp__nowhere_thing") == ("gone", "")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_builtin_that_disappeared_is_gone(tmp_path: Any) -> None:
    """A builtin's absence IS a real change (a settings flip), so it must not be
    excused as a transient.
    """
    session = _session(tmp_path, FakeManager([]))
    try:
        assert session._monitor_availability("bash") == ("gone", "")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_connected_server_missing_the_tool_is_unavailable(tmp_path: Any) -> None:
    manager = FakeManager([], status="connected")
    session = _session(tmp_path, manager)
    try:
        state, detail = session._monitor_availability(mcp_name())
        assert state == "unavailable"
        assert "no longer lists" in detail
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_mcp_error_with_the_server_down_is_unavailable(tmp_path: Any) -> None:
    """The call never reached the server, so nothing is known about the watched
    thing: counting it as a failed check is how one reconnect window disables a
    monitor.
    """
    tool = mcp_tool(error="MCP error: connection closed")
    manager = FakeManager([tool], status="disconnected")
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "unavailable"
        assert "MCP error" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_mcp_error_with_the_server_up_is_an_ordinary_failure(tmp_path: Any) -> None:
    """The server answered with a failure: that is the tool's own opinion and
    keeps the strike policy, because the manager's own retry already ran.
    """
    tool = mcp_tool(error="MCP error: bad request")
    manager = FakeManager([tool], status="connected")
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome.get("kind") is None
        assert "MCP error" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_invalid_arguments_fault_is_fatal(tmp_path: Any) -> None:
    """Deterministic and never self-healing: the arguments the monitor captured
    are rejected by the tool's own schema on every future tick, so the ladder
    would only spend four more checks reaching the same answer.
    """
    tool = mcp_tool(
        error="invalid arguments:\n- path: Extra inputs are not permitted",
        details={FAULT_KEY: FAULT_INVALID_ARGUMENTS},
    )
    manager = FakeManager([tool], status="connected")
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "fatal"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_manager_without_the_status_apis_behaves_as_before(tmp_path: Any) -> None:
    manager = FakeManager([], with_status_apis=False)
    session = _session(tmp_path, manager)
    try:
        assert session._monitor_availability(mcp_name()) == ("gone", "")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_declared_inventory_still_bounds_the_fallback(tmp_path: Any) -> None:
    """The fallback widens WHICH names resolve, never what may run: a host that
    declared its inventory must not have a monitor reach past it.
    """
    manager = FakeManager([mcp_tool()])
    session = _session(tmp_path, manager)
    try:
        session._declared_tools = frozenset({"bash"})
        assert session._resolve_monitor_tool(mcp_name()) is None
        session._declared_tools = None
        assert session._resolve_monitor_tool(mcp_name()) is not None
    finally:
        await session.dispose()
