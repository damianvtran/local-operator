"""The first-turn wait for declared/preloaded MCP servers (``session/declared_mcp_ready``).

The real-path proof (a cold ``lop exec --tools`` against a slow stdio server)
is ``tests/e2e/test_exec_declared_mcp_ready_e2e.py``; these cells pin the
decision table with a manager double: which servers are awaited, that an
unrelated slow server is not, that a warm cache waits on nothing, and the two
failure shapes (declared raises, preload-only warns).
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.mcp.manager import McpManager
from local_operator.session.declared_mcp_ready import (
    DeclaredMcpToolsUnavailable,
    await_named_mcp_servers,
    servers_to_await,
)


class FakeManager:
    """Servers that become ``connected`` (with tools) when their event is set."""

    def __init__(self, servers: dict[str, dict[str, Any]]) -> None:
        self.servers = servers
        self.waited: list[tuple[float, frozenset[str] | None]] = []

    def get_all_server_names(self) -> list[str]:
        return sorted(self.servers)

    def get_connection_status(self, name: str) -> str:
        return self.servers[name]["status"]

    def get_server_config(self, name: str):
        return SimpleNamespace(preload_tools=self.servers[name].get("preload", False))

    def get_server_tools(self, name: str) -> list[Any]:
        return list(self.servers[name].get("tools", ()))

    async def wait_settled(self, timeout_s: float, *, servers=None) -> bool:
        self.waited.append((timeout_s, None if servers is None else frozenset(servers)))
        for name in servers or self.servers:
            arrives = self.servers[name].get("arrives")
            if arrives is not None:
                self.servers[name]["status"] = "connected"
                self.servers[name]["tools"] = arrives
        return True


def _run(manager, declared, inventory: list[str], **kwargs):
    granted: list[str] = []

    def materialize() -> None:
        for server in manager.servers.values():
            for name in server.get("tools", ()):
                if name in (declared or ()) and name not in inventory:
                    inventory.append(name)
                    granted.append(name)

    asyncio.run(
        await_named_mcp_servers(
            manager,
            declared=declared,
            reachable=lambda: tuple(inventory),
            materialize=materialize,
            **kwargs,
        )
    )
    return granted


def test_only_the_declared_owner_is_awaited_and_its_tools_are_granted():
    manager = FakeManager(
        {
            "bd": {"status": "connecting", "arrives": ["mcp__bd_search", "mcp__bd_fetch"]},
            # An unrelated slow server: the run never named it, so it must not
            # hold the turn (the wait is narrowed to the owners).
            "oauthy": {"status": "connecting"},
        }
    )
    inventory: list[str] = []
    granted = _run(manager, frozenset({"mcp__bd_search", "mcp__bd_fetch"}), inventory)
    assert manager.waited == [(30.0, frozenset({"bd"}))]
    assert sorted(granted) == ["mcp__bd_fetch", "mcp__bd_search"]


def test_a_warm_cache_waits_on_nothing():
    # Deferred tools from the cache are already in the inventory: the first
    # request can carry them, so a warm run keeps today's latency.
    manager = FakeManager({"bd": {"status": "connecting", "tools": ["mcp__bd_search"]}})
    _run(manager, frozenset({"mcp__bd_search"}), ["mcp__bd_search"])
    assert manager.waited == []


def test_an_undeclared_session_without_preload_waits_on_nothing():
    manager = FakeManager({"lazy": {"status": "connecting"}})
    _run(manager, None, ["bash"])
    assert manager.waited == []


def test_a_cold_preload_server_is_awaited_without_a_declaration():
    manager = FakeManager(
        {"pre": {"status": "connecting", "preload": True, "arrives": ["mcp__pre_x"]}}
    )
    _run(manager, None, [])
    assert manager.waited == [(30.0, frozenset({"pre"}))]


def test_a_declared_server_still_connecting_after_the_bound_raises_before_sending():
    manager = FakeManager({"bd": {"status": "connecting"}})  # never arrives
    with pytest.raises(DeclaredMcpToolsUnavailable) as caught:
        _run(manager, frozenset({"mcp__bd_search", "bash"}), ["bash"], timeout_s=0.5)
    assert caught.value.servers == {"bd": "connecting"}
    assert caught.value.tools == ("mcp__bd_search",)
    assert "bd (connecting)" in str(caught.value)
    assert isinstance(caught.value, RuntimeError)


def test_a_preload_only_server_that_misses_the_bound_warns_and_proceeds(caplog):
    manager = FakeManager({"pre": {"status": "connecting", "preload": True}})
    with caplog.at_level(logging.WARNING, logger="local_operator.session.declared_mcp_ready"):
        _run(manager, None, [], timeout_s=0.5)
    assert "preload_tools MCP server 'pre' not ready" in caplog.text


def test_servers_not_connecting_are_left_to_the_startup_report():
    # Failed / auth-blocked / connected servers are not waited on and never
    # raise here: their failure was already reported at startup.
    manager = FakeManager({"dead": {"status": "disconnected"}, "auth": {"status": "auth-required"}})
    owners, preload = servers_to_await(manager, {"mcp__dead_x", "mcp__auth_y"}, ())
    assert owners == frozenset() and preload == frozenset()


def test_server_ownership_is_by_sanitized_prefix_not_by_splitting_the_name():
    # ``mcp__foo_bar_x`` cannot be split into (server, tool); both a server
    # ``foo`` and a server ``foo_bar`` plausibly own it, so both are awaited.
    manager = FakeManager(
        {
            "foo": {"status": "connecting"},
            "Foo-Bar": {"status": "connecting"},
            "baz": {"status": "connecting"},
        }
    )
    owners, _ = servers_to_await(manager, {"mcp__foo_bar_x"}, ())
    assert owners == frozenset({"foo", "Foo-Bar"})


def test_wait_settled_narrowed_to_servers_ignores_an_unrelated_continuation():
    async def scenario() -> tuple[bool, bool]:
        manager = McpManager.__new__(McpManager)
        never = asyncio.get_running_loop().create_future()
        quick = asyncio.get_running_loop().create_task(asyncio.sleep(0))
        slow = asyncio.ensure_future(never)
        # The narrowing reads only this mapping; a bare instance keeps the cell
        # free of transports.
        setattr(manager, "_pending_continuations", {"bd": (quick, quick), "slow": (slow, slow)})
        narrowed = await manager.wait_settled(5.0, servers={"bd"})
        whole = await manager.wait_settled(0.05)
        slow.cancel()
        return narrowed, whole

    narrowed, whole = asyncio.run(scenario())
    assert narrowed is True
    assert whole is False


# --------------------------------------------------------------------------
# Through a real Session: the FIRST ChatRequest, and the inventory's spill bit
# --------------------------------------------------------------------------


class SlowServerManager(FakeManager):
    """A one-server manager whose tools arrive only once the wait runs."""

    def __init__(self, arriving, *, arrives: bool = True) -> None:
        super().__init__({"bd": {"status": "connecting"}})
        self._arriving = list(arriving)
        self._arrives = arrives
        self._tools: list[Any] = []

    def get_tools(self) -> list[Any]:
        return list(self._tools)

    async def wait_settled(self, timeout_s: float, *, servers=None) -> bool:
        self.waited.append((timeout_s, None if servers is None else frozenset(servers)))
        await asyncio.sleep(0)
        if not self._arrives:
            return False
        self._tools = list(self._arriving)
        self.servers["bd"].update(status="connected", tools=[t.name for t in self._arriving])
        return True


@pytest.mark.asyncio
async def test_the_first_request_carries_declared_tools_from_a_server_still_connecting(tmp_path):
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_declared_tool_inventory import (
        ScriptedStream,
        echo_tool,
        make_session,
        surfaced_tools,
    )

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")])
    manager = SlowServerManager([echo_tool([], name="mcp__bd_search")])
    setattr(session, "mcp_manager", manager)  # a double, as ``attach_manager`` does
    session.set_tool_inventory(["mcp__bd_search"], unattended=True)
    assert session.tool_inventory == (), "precondition: nothing reachable before the wait"

    await session.prompt("go")

    assert surfaced_tools(stream, 0) == ["mcp__bd_search"]
    assert manager.waited == [(30.0, frozenset({"bd"}))]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declared_server_that_never_arrives_fails_the_turn_before_any_request(tmp_path):
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_declared_tool_inventory import (
        ScriptedStream,
        echo_tool,
        make_session,
    )

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[])
    manager = SlowServerManager([echo_tool([], name="mcp__bd_search")], arrives=False)
    setattr(session, "mcp_manager", manager)
    session.set_tool_inventory(["mcp__bd_search"], unattended=True)

    with pytest.raises(DeclaredMcpToolsUnavailable, match=r"bd \(connecting\)"):
        await session.prompt("go")

    assert stream.requests == [], "nothing may be sent without the declared reach"
    await session.dispose()


@pytest.mark.asyncio
async def test_spill_readability_is_derived_from_holding_read(tmp_path):
    from tests.unit.session.test_declared_tool_inventory import (
        ScriptedStream,
        echo_tool,
        make_session,
    )

    with_read = make_session(tmp_path / "a", ScriptedStream([]), tools=[echo_tool([], "read")])
    bounded = make_session(tmp_path / "b", ScriptedStream([]), tools=[echo_tool([], "read")])
    bounded.set_tool_inventory(["mcp__bd_search"], unattended=True)

    assert with_read._build_tool_context().spill_readable is True
    assert bounded._build_tool_context().spill_readable is False
    await with_read.dispose()
    await bounded.dispose()
