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
    ToolContext,
    ToolResult,
)
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.mcp_status import MCP_DISCOVERY_KEY, McpStartupOutcome
from local_operator.session.session import (
    MCP_DISCOVERY_FAILED_DETAIL,
    MCP_NOT_READY_DETAIL,
)
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
    hint: bool = True,
    calls: list[str] | None = None,
) -> AgentTool:
    """A stand-in MCP tool carrying the read-only hint the verdict gates on.

    ``hint=False`` is a tool whose annotations were not (yet) known — the
    shape a cache row written before annotations were cached builds.
    ``calls`` records every execute, so a wait path can prove nothing ran.
    """

    async def _exec(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        if calls is not None:
            calls.append(tool_call_id)
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
        mcp_annotations={"readOnlyHint": True} if hint else None,
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
        deferred: bool = False,
    ) -> None:
        self._tools = list(tools or [])
        self._deferred = deferred
        self._status = status
        self._settling = settling
        self._suspended = suspended
        self._servers = list(servers if servers is not None else [SERVER])
        self._with_status_apis = with_status_apis

    def get_tools(self) -> list[AgentTool]:
        return list(self._tools)

    def get_tool_meta(self, name: str) -> dict[str, Any] | None:
        if any(tool.name == name for tool in self._tools):
            return {"server_name": SERVER, "mcp_tool_name": name, "deferred": self._deferred}
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
    """The flap breaker's detail, pinned verbatim.

    It becomes the stall notice's ``Reason:``, so it must agree with that
    notice's "retrying" body. Any change to this sentence must re-verify the
    route it promises: the manager's half-open probe retries a tripped server
    unattended (``McpManager._schedule_probe``). It names no ``/mcp reauth``: a
    flap has a valid grant, and reauth would re-spend it.
    """
    manager = FakeManager([], status="disconnected", suspended=True)
    session = _session(tmp_path, manager)
    try:
        assert session._monitor_availability(mcp_name()) == (
            "unavailable",
            'MCP server "datadog" keeps dropping; reconnects are paused for a cool-down, '
            "then retried automatically",
        )
        # The stall notice clips its ``Reason:`` at NOTICE_ERROR_CHARS: even the
        # longest legal server name (``SERVER_NAME_RE``, 100 chars) must leave
        # the "retried automatically" promise whole (195 of 200).
        from local_operator.monitors.delivery import NOTICE_ERROR_CHARS

        _, detail = session._monitor_availability(mcp_name())
        longest = detail.replace('"datadog"', '"' + "x" * 100 + '"')
        assert len(longest) <= NOTICE_ERROR_CHARS
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_probe_mid_connect_reads_connecting_not_paused(tmp_path: Any) -> None:
    """The breaker stays open until the probe's connect lands, so for that
    instant the server is both suspended and connecting; connecting is true."""
    manager = FakeManager([], status="connecting", suspended=True)
    session = _session(tmp_path, manager)
    try:
        assert session._monitor_availability(mcp_name()) == (
            "unavailable",
            'MCP server "datadog" is still connecting',
        )
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

    The server drops DURING the call: a server already down before the tick is
    answered by the pre-execute liveness wait and never reaches ``execute``.
    """
    calls: list[str] = []
    tool = mcp_tool(error="MCP error: connection closed", calls=calls)

    class DropsMidCall(FakeManager):
        def get_connection_status(self, name: str) -> str:
            return "disconnected" if calls else "connected"

    manager = DropsMidCall([tool])
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
async def test_regression_pin_a_stored_intent_no_longer_fails_the_tick(tmp_path: Any) -> None:
    """FAIL-BEFORE/AFTER PIN for the live `i` class.

    Two live monitors (`4eabc50d61bd` m1 swap-step, m2 monitor-control) failed
    EVERY tick with ``invalid arguments:\n- i: Extra inputs are not permitted``:
    the model includes ``i`` because every tool schema advertises it, and every
    builtin params model is ``extra="forbid"``. The pre-fix failure is
    REPRODUCED here against the raw tool, and then the same stored arguments are
    shown to tick cleanly — and to observe a change — with no re-arm.
    """
    from local_operator.tools.registry import create_tools

    workdir = tmp_path / "watch"
    workdir.mkdir()
    (workdir / "a.py").write_text("a = 1\n", encoding="utf-8")

    tools = create_tools(ToolContext(cwd=str(workdir)))
    glob_tool = next(tool for tool in tools if tool.name == "glob")
    original = glob_tool.execute
    seen: list[dict[str, Any]] = []

    async def record(tool_call_id: str, args: dict[str, Any], *rest: Any) -> ToolResult:
        seen.append(dict(args))
        return await original(tool_call_id, args, *rest)

    glob_tool.execute = record  # type: ignore[method-assign]

    # The monitor as it was armed: the model's intent rode along in the args.
    stored = {"pattern": "*.py", "i": "Listing the python files in the workspace"}

    session = _session(tmp_path, FakeManager([]), tools=[glob_tool])
    try:
        # BEFORE (the pre-fix behaviour, reproduced): the raw call fails.
        raw = await glob_tool.execute(
            "probe", dict(stored), None, None, ToolContext(cwd=str(workdir))
        )
        assert raw.is_error is True
        assert "Extra inputs are not permitted" in (raw.text or "")
        seen.clear()

        spec = MonitorSpec(
            id="m1",
            name="py-files",
            tool="glob",
            arguments=stored,
            # The check runs in the armed cwd, as a real monitor's does.
            cwd=str(workdir),
            every_ms=60_000,
            created_at=1_756_000_000_000,
        )
        first = await session._run_monitor_check(spec)
        assert first.get("error") is None, first
        assert first.get("kind") is None
        # The tool got its real arguments — the intent was lifted, not forwarded.
        assert seen and "i" not in seen[-1] and seen[-1].get("pattern") == "*.py"
        baseline = first.get("text") or ""
        assert "a.py" in baseline

        # AFTER, second tick: the call still runs and now observes the change
        # (this is the text the scheduler diffs).
        (workdir / "b.py").write_text("b = 2\n", encoding="utf-8")
        second = await session._run_monitor_check(spec)
        assert second.get("error") is None, second
        assert second.get("text") != baseline
        assert "b.py" in (second.get("text") or "")
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


# --- The lazy-MCP window (mcp__ names only) -------------------------------
#
# Every test below pins the same distinction: an MCP tool that is absent
# because the session has not STARTED yet is not the same thing as a tool that
# is gone. The first class cost a real strike in production (m4, 2026-11-05).


@pytest.mark.asyncio
async def test_a_session_without_its_mcp_layer_is_unavailable_not_a_strike(tmp_path: Any) -> None:
    """MEASURED (m4, 2026-11-05): a scheduled wake re-hosted a cold session, the
    monitor's tick landed before the MCP layer connected, and the absent tool
    charged ``consecutive_failures: 1`` with "it is not in this session's tool
    set". Nothing had been removed — calling the tool answered ``Tool not
    found``, and after the server was asked for a tool (which connects it
    lazily) the same session read the channel — so the tick must WAIT, or five
    wake cycles silently disable a watch everyone believes is armed.
    """
    session = _session(tmp_path, None)
    assert session.mcp_manager is None
    try:
        assert session._monitor_availability(mcp_name()) == ("unavailable", MCP_NOT_READY_DETAIL)
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "unavailable"
        assert outcome["error"] == MCP_NOT_READY_DETAIL
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_manager_with_no_enabled_servers_is_gone_not_waiting(tmp_path: Any) -> None:
    """The excuse above is scoped to a session with NO manager, and this is the
    boundary. A manager exists only after ``attach_mcp_dispose``, which runs
    after the connect round, so a manager with an empty registry means zero
    ENABLED servers (disabled entries never reach it) — a real absence, not a
    load in progress. Calling that "the runtime is still starting" would be an
    untrue diagnosis for 24 hours, which is the class this method removes.
    """
    manager = FakeManager([], servers=[])
    session = _session(tmp_path, manager)
    try:
        assert session._monitor_availability(mcp_name()) == ("gone", "")
        outcome = await session._run_monitor_check(_spec())
        assert outcome.get("kind") is None
        assert "not in this session's tool set" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_manager_whose_discovery_did_not_complete_is_unavailable(tmp_path: Any) -> None:
    """Discovery RAISING leaves an ATTACHED manager with no roster at all.

    MEASURED by the reviewer of #3506 against the real discovery wrapper: the
    manager is built before the raise, so the session holds one with
    ``_configs`` never populated — an empty roster that read as "nothing is
    configured" and charged ``consecutive_failures: 1`` with "it is not in this
    session's tool set". Nothing was removed: the config was never read. The
    wiring records that raise on the session, and that record is the
    difference between this shape and a real removal (the test below).
    """
    manager = FakeManager([], servers=[])
    session = _session(tmp_path, manager)
    session.mcp_startup = McpStartupOutcome(failures={MCP_DISCOVERY_KEY: "config layer exploded"})
    try:
        assert session._monitor_availability(mcp_name()) == (
            "unavailable",
            MCP_DISCOVERY_FAILED_DETAIL,
        )
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "unavailable"
        assert outcome["error"] == MCP_DISCOVERY_FAILED_DETAIL
    finally:
        await session.dispose()


def test_the_discovery_failed_copy_is_the_verified_sentence() -> None:
    """The exact sentence, pinned, because every earlier version named a route
    that does not retry discovery for this state: ``/mcp reload`` (no such
    verb, review round 2), the app's Reload (its catalog rows never carry one,
    round 3) and ``/reload`` (viewer-only in shipped builds, round 4).

    A change to this copy must re-verify, against a FRESH runtime, every route
    it names before this pin moves: the sentence is the operator's only way
    out of the wait, and a route that does not work is worse than none.
    """
    from local_operator.monitors.delivery import NOTICE_ERROR_CHARS

    assert MCP_DISCOVERY_FAILED_DETAIL == (
        "MCP discovery failed at startup; it is retried when the session restarts"
    )
    assert len(MCP_DISCOVERY_FAILED_DETAIL) < NOTICE_ERROR_CHARS


@pytest.mark.asyncio
async def test_a_completed_discovery_with_an_empty_roster_is_still_gone(tmp_path: Any) -> None:
    """CONTROL for the test above: the SAME empty roster with a COMPLETED round.

    ``McpStartupOutcome()`` is what the factory writes when discovery ran and
    found nothing configured — zero enabled servers, a real absence — so the
    excuse must not extend to it. A session with NO record at all
    (``mcp_startup is None``) is pinned one test up and answers ``gone`` too.
    """
    manager = FakeManager([], servers=[])
    session = _session(tmp_path, manager)
    session.mcp_startup = McpStartupOutcome()
    try:
        assert session._monitor_availability(mcp_name()) == ("gone", "")
        outcome = await session._run_monitor_check(_spec())
        assert outcome.get("kind") is None
        assert "not in this session's tool set" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_name_outside_a_completed_rounds_roster_is_still_gone(tmp_path: Any) -> None:
    """CONTROL, nearer corner: the discovery key is ALSO the collapse target
    when every configured server failed for one setup reason (the MCP SDK
    missing) — a COMPLETED round that knows its roster. A name that roster does
    not mint is a stale name, not a discovery in flight, so the strike stays.
    """
    manager = FakeManager([], servers=[])
    session = _session(tmp_path, manager)
    session.mcp_startup = McpStartupOutcome(
        configured=("datadog",), failures={MCP_DISCOVERY_KEY: "the MCP SDK is missing"}
    )
    try:
        assert session._monitor_availability("mcp__retired_tool") == ("gone", "")
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_non_mcp_tool_that_is_gone_still_charges_a_strike(tmp_path: Any) -> None:
    """CONTROL for the two tests above: the excuse is scoped to ``mcp__`` names.
    A builtin whose absence is real (a settings flip, a pruned tool) keeps the
    ordinary ladder, and the outcome carries no ``kind`` — which is what makes
    the scheduler charge the strike.
    """
    session = _session(tmp_path, None)
    try:
        assert session._monitor_availability("bash") == ("gone", "")
        outcome = await session._run_monitor_check(_spec(tool="bash"))
        assert outcome.get("kind") is None
        assert "not in this session's tool set" in str(outcome["error"])
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_status_this_build_does_not_know_is_unavailable_and_named(tmp_path: Any) -> None:
    """A state outside the documented four is not proof of removal either, and
    reporting it as "disconnected" would invent a cause the manager never gave.
    """
    manager = FakeManager([], status="warming-up")
    session = _session(tmp_path, manager)
    try:
        state, detail = session._monitor_availability(mcp_name())
        assert state == "unavailable"
        assert 'reported status "warming-up"' in detail
    finally:
        await session.dispose()


# --- A tool from the cache never runs; the tick waits for the live tool ---
#
# MEASURED (v0.101.7, 2026-11-05 23:43): session b96a7ed9353a's runtime rolled
# over, its Slack tools were registered from the MCP tool cache (which then
# dropped annotations) ahead of the live connect, and monitor m4's first tick
# was refused "does not declare readOnlyHint" and STRUCK (1 -> 2).
# QA round 1 on #3546 (Q-1): once the cache kept the hint, a cache-built tool
# passed the gate and its deferred execute called the server after the connect
# WITHOUT re-reading the live list, so a tool whose live definition had dropped
# the hint ran. Both are pinned below: until the live tool is registered the
# tick waits (no strike) and executes nothing, whatever the cached hint says.


async def _tick_through_scheduler(
    session: Any, manager: Any, swap: Any, cwd: Any
) -> dict[str, Any]:
    """Arm against a hinted live tool, apply ``swap`` (the restart), tick once.

    Arming goes through the real arm gate, so the watch is armed the way m4
    was: against a live tool. The counters come from the scheduler, i.e. what
    the ladder actually charged.
    """
    from tests.unit.monitors.support import drain_checks

    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    created = await session._monitors.create(
        {"name": "watch", "tool": mcp_name(), "arguments": {"query": "up"}},
        cwd=str(cwd),
    )
    assert created.get("created"), created
    swap(manager)
    clock[0] += 120_000
    await session._monitors.pump(now_ms=clock[0])
    await drain_checks(session._monitors)
    runtime = session._monitors.runtime(created["spec"].id)
    assert runtime is not None
    return runtime.counters


def _restart_onto_cache(
    manager: Any, calls: list[str], *, hint: bool, status: str, deferred: bool
) -> None:
    manager._tools = [mcp_tool(hint=hint, calls=calls)]
    manager._status = status
    manager._deferred = deferred


@pytest.mark.asyncio
async def test_a_cached_hint_never_runs_a_tool_whose_live_list_dropped_it(tmp_path: Any) -> None:
    """Q-1: cache says read-only, the live server no longer does. The cached
    tool must not execute while it waits, and the first LIVE tick is refused
    and struck by the unchanged gate.
    """
    calls: list[str] = []
    manager = FakeManager([mcp_tool(calls=calls)])
    session = _session(tmp_path, manager)
    try:
        counters = await _tick_through_scheduler(
            session,
            manager,
            lambda m: _restart_onto_cache(m, calls, hint=True, status="connecting", deferred=True),
            tmp_path,
        )
        assert calls == []
        assert counters["consecutive_failures"] == 0, counters
        assert counters["unavailable_ticks"] == 1, counters
        assert counters["last_error"] == f'MCP server "{SERVER}" is still connecting', counters

        # The connect lands and the live list (without the hint) replaces it.
        _restart_onto_cache(manager, calls, hint=False, status="connected", deferred=False)
        outcome = await session._run_monitor_check(_spec())
        assert outcome.get("kind") is None
        assert "does not declare readOnlyHint" in outcome["error"]
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_cache_built_tool_without_its_hint_waits_while_its_server_connects(
    tmp_path: Any,
) -> None:
    """The measured m4 strike: a hint-less cached tool waits instead."""
    calls: list[str] = []
    manager = FakeManager([mcp_tool(calls=calls)])
    session = _session(tmp_path, manager)
    try:
        counters = await _tick_through_scheduler(
            session,
            manager,
            lambda m: _restart_onto_cache(
                m, calls, hint=False, status="disconnected", deferred=True
            ),
            tmp_path,
        )
        assert counters["consecutive_failures"] == 0, counters
        assert counters["unavailable_ticks"] == 1, counters
        assert counters["last_note"] == "tool unavailable — retrying", counters
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_auth_required_server_names_the_reauth_remedy(tmp_path: Any) -> None:
    """Review MINOR-1: the wait reuses the availability ladder's sentence, so a
    server needing re-auth says so instead of "still connecting"."""
    calls: list[str] = []
    manager = FakeManager([mcp_tool(calls=calls)], status="auth-required", deferred=True)
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome == {
            "error": f'MCP server "{SERVER}" needs re-authentication — run /mcp reauth {SERVER}',
            "kind": "unavailable",
        }
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_connected_server_still_serving_the_cached_tool_waits(tmp_path: Any) -> None:
    """DEFENSIVE branch: ``connected`` but the registry still holds the
    cache-built (``deferred``) tool. Round-1 review found no observable window
    for this; the pin keeps a future reordering from reopening Q-1.
    """
    calls: list[str] = []
    manager = FakeManager([mcp_tool(calls=calls)], status="connected", deferred=True)
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome["kind"] == "unavailable"
        assert "still comes from the tool cache" in outcome["error"]
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unreadable_liveness_probe_strikes_and_runs_nothing(tmp_path: Any) -> None:
    """Fail closed: no wait (the ladder would never end), no pass (an unproven
    tool would run) — a strike."""
    calls: list[str] = []
    manager = FakeManager([mcp_tool(calls=calls)], with_status_apis=False)
    session = _session(tmp_path, manager)
    try:
        outcome = await session._run_monitor_check(_spec())
        assert outcome.get("kind") is None
        assert "could not be read" in outcome["error"]
        assert calls == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_live_tool_without_its_hint_is_still_refused_and_struck(tmp_path: Any) -> None:
    """THE CONTROL: the safety rule is unchanged for a live-connected tool."""
    calls: list[str] = []
    manager = FakeManager([mcp_tool()])
    session = _session(tmp_path, manager)
    try:
        counters = await _tick_through_scheduler(
            session,
            manager,
            lambda m: _restart_onto_cache(m, calls, hint=False, status="connected", deferred=False),
            tmp_path,
        )
        assert counters["consecutive_failures"] == 1, counters
        assert "does not declare readOnlyHint" in str(counters["last_error"]), counters
        assert calls == []
        # Arm time is untouched: a hint-less tool still cannot be armed.
        refused = await session._monitors.create(
            {"name": "again", "tool": mcp_name(), "arguments": {"query": "up"}},
            cwd=str(tmp_path),
        )
        assert not refused.get("created"), refused
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_waiting_tick_runs_once_the_live_tool_carries_the_hint(
    tmp_path: Any,
) -> None:
    calls: list[str] = []
    manager = FakeManager([mcp_tool(hint=True, calls=calls)], status="connecting", deferred=True)
    session = _session(tmp_path, manager)
    try:
        assert (await session._run_monitor_check(_spec()))["kind"] == "unavailable"
        assert calls == []
        manager._tools = [mcp_tool(calls=calls)]
        manager._status = "connected"
        manager._deferred = False
        assert (await session._run_monitor_check(_spec()))["error"] is None
        assert len(calls) == 1
    finally:
        await session.dispose()
