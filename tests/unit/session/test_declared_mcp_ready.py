"""The first-turn wait for declared/preloaded MCP servers (``session/declared_mcp_ready``).

The real-path proof (a cold ``lop exec --tools`` against a slow stdio server)
is ``tests/e2e/test_exec_declared_mcp_ready_e2e.py``; these cells pin the
decision table with a manager double: which servers are awaited, that an
unrelated slow server is not, that a warm cache waits on nothing, and the two
failure shapes (declared raises, preload-only warns). ``TimedManager`` cells run
the REAL ``McpManager.wait_settled`` over real continuation tasks, so the
reachability loop's timing (stop the moment the declared tool lands, fail fast
on a server that already died) is measured, not assumed.
"""

from __future__ import annotations

import asyncio
import logging
import time
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

    async def wait_settled(self, timeout_s: float, *, servers=None, first=False) -> bool:
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


def _waited_on(manager) -> list[frozenset[str] | None]:
    return [servers for _timeout, servers in manager.waited]


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
    assert _waited_on(manager) == [frozenset({"bd"})]
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
    assert _waited_on(manager) == [frozenset({"pre"})]


def test_a_connecting_server_with_nothing_left_to_wait_on_raises_without_claiming_a_timeout():
    # The double's wait returns at once and the server still reads
    # ``connecting`` (the reconnect-backoff shape: no connect attempt in
    # flight). The gate must not spin, and must not say "timed out" (QA O2);
    # the real bound is the ``TimedManager`` hang cell below.
    manager = FakeManager({"bd": {"status": "connecting"}})  # never arrives
    with pytest.raises(DeclaredMcpToolsUnavailable) as caught:
        _run(manager, frozenset({"mcp__bd_search", "bash"}), ["bash"], timeout_s=5.0)
    assert caught.value.servers == {"bd": "connecting"}
    assert caught.value.tools == ("mcp__bd_search",)
    assert caught.value.elapsed_s < 1.0
    assert "bd still connecting, with no connect attempt in flight" in str(caught.value)
    assert "timed out" not in str(caught.value)
    assert isinstance(caught.value, RuntimeError)


def test_a_preload_only_server_that_misses_the_bound_warns_and_proceeds(caplog):
    manager = FakeManager({"pre": {"status": "connecting", "preload": True}})
    with caplog.at_level(logging.WARNING, logger="local_operator.session.declared_mcp_ready"):
        _run(manager, None, [], timeout_s=0.5)
    assert "preload_tools MCP server 'pre' not ready" in caplog.text


def test_servers_not_connecting_are_never_waited_on():
    # Nothing a wait could change: they are decided at once (see the
    # already-failed cells below), never awaited.
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


class TimedManager:
    """The REAL ``McpManager.wait_settled`` over real continuation tasks.

    ``plan`` maps each server to how it comes up: ``("up", delay_s, tools)``,
    ``("dies", delay_s)`` (a crash or a bad command noticed late), ``("hangs",)``
    (wedged), or the states a server can already be in when the first turn
    checks: ``("connected", tools)`` and ``("disconnected",)`` (a bad command or
    an instant crash: no continuation left to wait on). Build it inside a loop.
    """

    wait_settled = McpManager.wait_settled

    def __init__(self, plan: dict[str, tuple[Any, ...]], *, preload: tuple[str, ...] = ()) -> None:
        self.status: dict[str, str] = {}
        self.tools: dict[str, list[str]] = {}
        self.preload = set(preload)
        self._pending_continuations: dict[str, tuple[asyncio.Task[None], Any]] = {}
        loop = asyncio.get_running_loop()
        for name, spec in plan.items():
            self.tools[name] = list(spec[1]) if spec[0] == "connected" else []
            if spec[0] in ("connected", "disconnected"):
                self.status[name] = spec[0]
                continue
            self.status[name] = "connecting"
            task = loop.create_task(self._come_up(name, spec))
            self._pending_continuations[name] = (task, task)
            # The manager's own bookkeeping: a settled continuation leaves the map.
            task.add_done_callback(lambda _t, _n=name: self._pending_continuations.pop(_n, None))

    async def _come_up(self, name: str, spec: tuple[Any, ...]) -> None:
        if spec[0] == "hangs":
            await asyncio.Event().wait()
        await asyncio.sleep(spec[1])
        if spec[0] == "up":
            self.tools[name] = list(spec[2])
            self.status[name] = "connected"
        else:
            self.status[name] = "disconnected"

    def get_all_server_names(self) -> list[str]:
        return sorted(self.status)

    def get_connection_status(self, name: str) -> str:
        return self.status[name]

    def get_server_config(self, name: str):
        return SimpleNamespace(preload_tools=name in self.preload)

    def get_server_tools(self, name: str) -> list[str]:
        return list(self.tools[name])

    def get_tools(self) -> list[Any]:
        return [SimpleNamespace(name=tool) for tools in self.tools.values() for tool in tools]

    def close(self) -> None:
        for task, _ in list(self._pending_continuations.values()):
            task.cancel()


async def _gate(manager: TimedManager, declared, inventory: list[str] | None = None, **kwargs):
    """``(settled, elapsed_s, inventory)`` for one real-time gate run."""
    inventory = [] if inventory is None else inventory
    wanted = frozenset(declared or ())

    def materialize() -> None:
        for tool in manager.get_tools():
            if tool.name in wanted and tool.name not in inventory:
                inventory.append(tool.name)

    started = time.monotonic()
    try:
        settled = await await_named_mcp_servers(
            manager,
            declared=wanted if declared is not None else None,
            reachable=lambda: tuple(inventory),
            materialize=materialize,
            **kwargs,
        )
    finally:
        elapsed = time.monotonic() - started
        manager.close()
    return settled, elapsed, inventory


@pytest.mark.asyncio
async def test_a_wedged_prefix_sibling_neither_holds_nor_fails_a_reachable_declared_tool():
    # QA Q4r / review R1: ``bd`` hangs, ``bd_x`` serves ``mcp__bd_x_search``.
    # The status-keyed gate waited the whole bound on ``bd`` (an ambiguous
    # owner of the name) and then raised with an EMPTY unreachable list.
    manager = TimedManager(
        {"bd": ("hangs",), "bd_x": ("up", 0.05, ["mcp__bd_x_search"])},
    )
    settled, elapsed, inventory = await _gate(manager, {"mcp__bd_x_search"}, timeout_s=5.0)
    assert settled is True
    assert inventory == ["mcp__bd_x_search"]
    assert elapsed < 1.0, f"held {elapsed:.2f} s by the wedged sibling"


@pytest.mark.asyncio
async def test_a_slow_healthy_prefix_sibling_does_not_hold_the_turn():
    # QA Q4s: ``bd`` is healthy but slow (6 s in QA's run, 1.5 s here); the
    # declared tool arrived from ``bd_x`` long before, so the turn goes then.
    manager = TimedManager(
        {"bd": ("up", 1.5, ["mcp__bd_search"]), "bd_x": ("up", 0.05, ["mcp__bd_x_search"])},
    )
    settled, elapsed, _inventory = await _gate(manager, {"mcp__bd_x_search"}, timeout_s=5.0)
    assert settled is True
    assert elapsed < 1.0, f"waited {elapsed:.2f} s for undeclared-for-this-name bd"


@pytest.mark.asyncio
async def test_a_declared_server_that_already_failed_raises_at_once_naming_the_tool():
    # QA Q-F2 / Q2a (bad command) and Q2c (instant crash): the server is
    # already ``disconnected`` when the first turn checks. The status-keyed
    # gate neither waited nor raised, so exec sent a turn with NO tools and
    # exited 0.
    manager = TimedManager({"ghost": ("disconnected",), "other": ("connected", ["mcp__other_x"])})
    with pytest.raises(DeclaredMcpToolsUnavailable) as caught:
        await _gate(manager, {"mcp__ghost_search", "mcp__other_x"}, timeout_s=30.0)
    assert caught.value.tools == ("mcp__ghost_search",)
    assert caught.value.servers == {"ghost": "disconnected"}
    assert caught.value.elapsed_s < 1.0, "a dead server is decided at once, not after the bound"
    message = str(caught.value)
    assert "ghost failed: disconnected" in message
    assert "after 30" not in message, "the message must not claim a timeout that never ran"


@pytest.mark.asyncio
async def test_a_declared_server_dying_mid_wait_raises_when_it_dies_not_at_the_bound():
    # QA Q2d, and O2: the reason is "failed", the time is the real one.
    manager = TimedManager({"late": ("dies", 0.1)})
    with pytest.raises(DeclaredMcpToolsUnavailable) as caught:
        await _gate(manager, {"mcp__late_search"}, timeout_s=10.0)
    assert caught.value.servers == {"late": "disconnected"}
    assert 0.05 < caught.value.elapsed_s < 2.0
    assert "late failed: disconnected" in str(caught.value)


@pytest.mark.asyncio
async def test_a_wedged_declared_server_raises_at_the_bound_saying_it_timed_out():
    manager = TimedManager({"stuck": ("hangs",)})
    with pytest.raises(DeclaredMcpToolsUnavailable) as caught:
        await _gate(manager, {"mcp__stuck_search"}, timeout_s=0.4)
    assert caught.value.elapsed_s >= 0.4
    assert "stuck timed out: still connecting at the 0.4 s bound" in str(caught.value)


@pytest.mark.asyncio
async def test_a_name_no_failed_server_owes_is_left_to_the_end_of_run_report():
    # A connected server without the tool (a typo), and a name no configured
    # server could own (a role naming a tool that exists on another machine):
    # nothing failed to come up, so the gate does not refuse the run; the
    # end-of-run report (``report_unresolved_declared_tools``) names them.
    manager = TimedManager({"bd": ("connected", ["mcp__bd_search"])})
    settled, elapsed, _inventory = await _gate(
        manager, {"mcp__bd_serch", "mcp__elsewhere_x"}, timeout_s=5.0
    )
    assert settled is True and elapsed < 0.5


@pytest.mark.asyncio
async def test_an_abort_cuts_the_wait_short_and_leaves_the_gate_unsettled():
    # Review R4: the wait runs before the turn's signal exists, so Esc reaches
    # it only through the sticky flag the predicate reads.
    manager = TimedManager({"stuck": ("hangs",)})
    deadline = time.monotonic() + 0.2
    settled, elapsed, _inventory = await _gate(
        manager,
        {"mcp__stuck_search"},
        timeout_s=10.0,
        cancelled=lambda: time.monotonic() > deadline,
    )
    assert settled is False
    assert elapsed < 1.5, f"Esc took {elapsed:.2f} s to land"


@pytest.mark.asyncio
async def test_a_front_end_host_does_not_wait_for_a_preload_only_server():
    # Review R4 / QA O1: with a front end attached the preload server's tools
    # merge mid-session (``session_factory.create_session``'s documented
    # contract), so the turn keeps the fast gate. Headless still waits.
    interactive = TimedManager({"pl": ("hangs",)}, preload=("pl",))
    settled, elapsed, _ = await _gate(interactive, None, timeout_s=5.0, wait_for_preload=False)
    assert settled is True and elapsed < 0.2
    headless = TimedManager({"pl": ("up", 0.1, ["mcp__pl_x"])}, preload=("pl",))
    settled, elapsed, _ = await _gate(headless, None, timeout_s=5.0, wait_for_preload=True)
    assert settled is True and elapsed >= 0.1
    assert headless.status["pl"] == "connected"


def test_wait_settled_first_returns_when_one_of_its_servers_settles():
    async def scenario() -> tuple[bool, float]:
        manager = McpManager.__new__(McpManager)
        never = asyncio.ensure_future(asyncio.get_running_loop().create_future())
        quick = asyncio.get_running_loop().create_task(asyncio.sleep(0.05))
        setattr(manager, "_pending_continuations", {"a": (quick, quick), "b": (never, never)})
        started = time.monotonic()
        all_settled = await manager.wait_settled(5.0, servers={"a", "b"}, first=True)
        elapsed = time.monotonic() - started
        never.cancel()
        return all_settled, elapsed

    all_settled, elapsed = asyncio.run(scenario())
    assert all_settled is False, "b is still pending"
    assert elapsed < 1.0


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

    async def wait_settled(self, timeout_s: float, *, servers=None, first=False) -> bool:
        self.waited.append((timeout_s, None if servers is None else frozenset(servers)))
        await asyncio.sleep(0)
        if not self._arrives:
            # It died while connecting: decided at once, so the cell needs no
            # real 30 s bound (``Session`` passes the default).
            self.servers["bd"]["status"] = "disconnected"
            return True
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
    assert _waited_on(manager) == [frozenset({"bd"})]
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

    with pytest.raises(DeclaredMcpToolsUnavailable, match=r"bd failed: disconnected"):
        await session.prompt("go")

    assert stream.requests == [], "nothing may be sent without the declared reach"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_retry_after_the_loud_failure_meets_the_gate_again(tmp_path):
    # Review R3: the once-per-session flag was set BEFORE the wait, so an SDK
    # caller's second ``prompt()`` after the raise skipped the gate and sent
    # without the declared tools: the silent failure the gate exists to remove.
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

    with pytest.raises(DeclaredMcpToolsUnavailable):
        await session.prompt("go")
    with pytest.raises(DeclaredMcpToolsUnavailable):
        await session.prompt("go again")

    assert stream.requests == [], "the retry must not send without the declared reach"
    assert session._named_mcp_ready_checked is False
    await session.dispose()


@pytest.mark.asyncio
async def test_a_front_end_session_keeps_the_fast_gate_for_a_preload_only_server(tmp_path):
    # Review R4: ``has_ui`` (or an attached viewer) means the late preload
    # server merges mid-session, so the first turn does not wait on it; a
    # headless session still does.
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_declared_tool_inventory import (
        ScriptedStream,
        make_session,
    )

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[], has_ui=True)
    manager = FakeManager({"pl": {"status": "connecting", "preload": True}})
    setattr(session, "mcp_manager", manager)

    await session.prompt("go")

    assert manager.waited == [], "a front-end session must not hold its turn for preload"
    assert stream.requests, "the turn went out without waiting"
    await session.dispose()

    headless_stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    headless = make_session(tmp_path / "h", headless_stream, tools=[])
    headless_manager = FakeManager(
        {"pl": {"status": "connecting", "preload": True, "arrives": ["mcp__pl_x"]}}
    )
    setattr(headless, "mcp_manager", headless_manager)
    await headless.prompt("go")
    assert _waited_on(headless_manager) == [frozenset({"pl"})]
    await headless.dispose()


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
