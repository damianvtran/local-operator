"""Cache-safe MCP schema lifecycle, pinned against what the provider is SENT.

The request's tools array is position 0 of the prompt-cache prefix: adding or
removing a tool while the cache is warm reprices the whole conversation. These
tests drive a real ``Session`` wired by ``wire_mcp_into_session`` and read the
``tools`` of every ``ChatRequest`` it sends, because that array — not the
inventory — is what the cache contract is about. They pin:

- a per-tool ``mcp://<server>/<tool>`` read leaves the array untouched and the
  tool callable at once, directly and through eval's ``tool(...)`` bridge;
- an over-cap schema (the publication fallback) leaves the array again only at
  a cold boundary — a compaction commit or the idle flush — and lands on the
  SAME request that boundary already makes cold, never a later warm one;
- recently used and ``preload_tools`` origins are never demoted, a demoted
  tool still resolves, and a child inherits the lean set plus the demoted ones
  as deferred-resolvable.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from local_operator import session_factory
from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import AgentTool, TextContent, ToolResult
from local_operator.mcp.resources import MAX_INLINE_TOOL_SCHEMA_CHARS
from local_operator.session_factory import (
    MCP_DEMOTE_AFTER_IDLE_RUNS,
    wire_mcp_into_session,
)
from tests.unit.session.test_deferred_tools import (
    ScriptedStream,
    _call,
    _results,
    _session,
    _text,
    _tool,
)
from tests.unit.test_session_factory import FakeMcpManager

SERVER = "datadog"
SMALL = "get_logs"
BIG = "query_metrics"


def _mcp_tool(manager: FakeMcpManager, raw: str, executed: list[str], *, big: bool) -> AgentTool:
    name = f"mcp__{SERVER}_{raw}"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    if big:
        parameters["description"] = "x" * (MAX_INLINE_TOOL_SCHEMA_CHARS + 1)

    async def execute(tool_call_id, args, signal, on_update, context):
        # What ``McpManager._execute_tool_call`` does first on every route.
        if manager.on_tool_called is not None:
            manager.on_tool_called(SERVER, raw)
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=name,
            content=[TextContent(text=f"{name} ran")],
        )

    tool = AgentTool(
        name=name,
        description=f"{raw} on {SERVER}",
        parameters=parameters,
        approval_tier="read",
        execute=execute,
    )
    manager.tools.append(tool)
    manager.meta[name] = {
        "server_name": SERVER,
        "mcp_tool_name": raw,
        "deferred": False,
    }
    return tool


def _manager(executed: list[str], *, preload: bool = False) -> FakeMcpManager:
    manager = FakeMcpManager(
        configured=[SERVER], connected=[SERVER], preload={SERVER} if preload else None
    )
    manager.on_tool_called = None  # type: ignore[attr-defined]
    _mcp_tool(manager, SMALL, executed, big=False)
    _mcp_tool(manager, BIG, executed, big=True)
    return manager


async def _wire(session, manager, monkeypatch) -> session_factory._KnowledgeHooks:
    hooks = session_factory._KnowledgeHooks()

    async def fake_discover(cwd, auth_store=None):
        return manager, list(manager.tools), []

    monkeypatch.setattr("local_operator.mcp.discover_and_load_mcp_tools", fake_discover)
    await wire_mcp_into_session(session, list(session._tools), ".", hooks)
    session.mcp_manager = manager
    return hooks


def _deferred(session: Any) -> set[tuple[str, str]]:
    """The deferred-origin set ``wire_mcp_into_session`` attaches with setattr.

    It is dynamic (never declared on ``Session``), so a typed accessor is the
    only way to read it without pyright reporting a missing attribute.
    """
    return getattr(session, "_mcp_deferred_origins")


def _sent(stream: ScriptedStream, index: int) -> list[str]:
    return [tool.name for tool in stream.requests[index].tools if tool.name.startswith("mcp__")]


def _last_sent(stream: ScriptedStream) -> list[str]:
    return _sent(stream, len(stream.requests) - 1)


BIG_NAME = f"mcp__{SERVER}_{BIG}"
SMALL_NAME = f"mcp__{SERVER}_{SMALL}"


@pytest.mark.asyncio
async def test_a_per_tool_read_never_changes_the_array_and_is_callable_the_same_turn(
    tmp_path, monkeypatch
) -> None:
    executed: list[str] = []
    manager = _manager(executed)
    hooks_box: dict[str, Any] = {}
    replies: list[str | None] = []
    dispatched: list[dict[str, Any]] = []

    async def read_execute(tool_call_id, args, signal, on_update, context):
        replies.append(hooks_box["hooks"].mcp_resolver(f"mcp://{SERVER}/{SMALL}"))
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="read", content=[TextContent(text="r")]
        )

    async def eval_execute(tool_call_id, args, signal, on_update, context):
        dispatched.append(await context.dispatch_tool(SMALL_NAME, {"q": "status:error"}))
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="e")]
        )

    stream = ScriptedStream(
        [
            _call("read"),
            _call(SMALL_NAME, '{"q": "x"}', call_id="c2"),
            _call("eval", '{"code": "x"}', call_id="c3"),
            _text("done"),
            _text("again"),
        ]
    )
    session = _session(
        tmp_path,
        stream,
        [
            _tool("read", executed, execute=read_execute),
            _tool("eval", executed, execute=eval_execute),
        ],
    )
    hooks_box["hooks"] = await _wire(session, manager, monkeypatch)

    await session.prompt("look at the logs")
    await session.prompt("next")

    assert replies[0] is not None and "Input schema:" in replies[0]
    # No request of either turn ever carried an MCP schema.
    assert [_sent(stream, i) for i in range(len(stream.requests))] == [[]] * len(stream.requests)
    # Callable directly (fallback resolver) and via the eval bridge, same turn.
    assert _results(session)["c2"] == f"{SMALL_NAME} ran"
    assert dispatched and dispatched[0]["is_error"] is False
    assert executed.count(SMALL_NAME) == 2
    assert (SERVER, SMALL) in _deferred(session)
    await session.dispose()


async def _published_big(tmp_path, monkeypatch, *, preload: bool = False, **kwargs: Any):
    """A session whose over-cap tool was published before its first turn."""
    executed: list[str] = []
    manager = _manager(executed, preload=preload)
    stream = ScriptedStream([])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed)],
        compaction_settings=CompactionSettings(keep_recent_tokens=40),
        **kwargs,
    )
    hooks = await _wire(session, manager, monkeypatch)
    if not preload:
        assert hooks.mcp_resolver is not None
        reply = hooks.mcp_resolver(f"mcp://{SERVER}/{BIG}")
        assert reply is not None and "too large to show inline" in reply
    await session.prompt("first " + "detail " * 30)
    assert BIG_NAME in _last_sent(stream)
    return session, stream, executed


async def _idle_runs(session, count: int) -> None:
    for index in range(count):
        await session.prompt(f"turn {index} " + "detail " * 30)


@pytest.mark.asyncio
async def test_warm_turns_never_change_the_array_even_when_the_tool_goes_unused(
    tmp_path, monkeypatch
) -> None:
    session, stream, _ = await _published_big(tmp_path, monkeypatch)
    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS + 2)
    assert all(BIG_NAME in _sent(stream, i) for i in range(len(stream.requests)))
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_commit_demotes_an_idle_published_schema_on_the_next_request(
    tmp_path, monkeypatch
) -> None:
    session, stream, executed = await _published_big(tmp_path, monkeypatch)
    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS)
    assert BIG_NAME in _last_sent(stream)

    assert (await session.compact_now()).ran is True
    before = len(stream.requests)
    await session.prompt("after compaction")
    # The first request after the commit (already a cache rewrite) is lean...
    assert BIG_NAME not in _sent(stream, before)
    # ...and the array then holds still: no second change on a warm turn.
    await session.prompt("warm again")
    assert _last_sent(stream) == _sent(stream, before)

    # Demoted is not unresolvable: the deferred resolver still serves it.
    assert (SERVER, BIG) in _deferred(session)
    resolved = session._resolve_tool_outside_inventory(BIG_NAME)
    assert resolved is not None and resolved.name == BIG_NAME
    await session.dispose()


@pytest.mark.asyncio
async def test_a_mid_turn_cold_boundary_shrinks_the_very_next_request_not_the_next_turn(
    tmp_path, monkeypatch
) -> None:
    """The once-per-turn latch would otherwise hold the old array until the
    next turn — by then warm again, so the change would cost a second miss."""
    session, stream, _ = await _published_big(tmp_path, monkeypatch)
    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS)
    demoted: list[bool] = []

    async def boundary_execute(tool_call_id, args, signal, on_update, context):
        # Exactly what ``_finish_compaction`` calls once its commit has landed.
        demoted.append(session._demote_cold_mcp_tools())
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="read", content=[TextContent(text="r")]
        )

    session._tools = [
        tool.model_copy(update={"execute": boundary_execute}) if tool.name == "read" else tool
        for tool in session._tools
    ]
    session._context.tools = session._tools
    stream.turns = [
        *stream.turns,
        *([_text("pad")] * (len(stream.requests) - len(stream.turns))),
    ]
    stream.turns += [_call("read"), _text("done")]
    first = len(stream.requests)
    await session.prompt("go")

    assert demoted == [True]
    assert BIG_NAME in _sent(stream, first)  # the turn started warm
    assert BIG_NAME not in _sent(stream, first + 1)  # the boundary's own request
    await session.dispose()


@pytest.mark.asyncio
async def test_the_idle_flush_demotes_on_the_first_request_of_the_cold_run(
    tmp_path, monkeypatch
) -> None:
    from local_operator.compaction import api as compaction_api

    session, stream, _ = await _published_big(tmp_path, monkeypatch)
    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS)
    assert BIG_NAME in _last_sent(stream)

    session._last_provider_request_ms = int(time.time() * 1000) - compaction_api.IDLE_FLUSH_MS - 1
    before = len(stream.requests)
    await session.prompt("back after lunch")
    assert BIG_NAME not in _sent(stream, before)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_recently_used_schema_survives_a_cold_boundary(tmp_path, monkeypatch) -> None:
    session, stream, executed = await _published_big(tmp_path, monkeypatch)
    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS)
    stream.turns = [
        *stream.turns,
        *([_text("pad")] * (len(stream.requests) - len(stream.turns))),
    ]
    stream.turns += [_call(BIG_NAME, '{"q": "cpu"}'), _text("done")]
    await session.prompt("use it")
    assert BIG_NAME in executed

    assert (await session.compact_now()).ran is True
    await session.prompt("after compaction")
    assert BIG_NAME in _last_sent(stream)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_preload_server_is_never_demoted(tmp_path, monkeypatch) -> None:
    session, stream, _ = await _published_big(tmp_path, monkeypatch, preload=True)
    assert set(_last_sent(stream)) == {SMALL_NAME, BIG_NAME}
    # Two boundaries, each after more idle runs than the window: an origin with
    # no use stamp has its clock started at the first, so only the second can
    # prove the preload guard (rather than the unknown-origin grace) holds.
    for _ in range(2):
        await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS + 1)
        assert (await session.compact_now()).ran is True
        await session.prompt("after compaction")
        assert set(_last_sent(stream)) == {SMALL_NAME, BIG_NAME}
    await session.dispose()


@pytest.mark.asyncio
async def test_a_child_publishes_only_the_parents_live_set_and_resolves_demoted_ones(
    tmp_path, monkeypatch
) -> None:
    from tests.unit.session.test_launch_subagent import build_child

    session, stream, _ = await _published_big(tmp_path, monkeypatch)
    child_before = await build_child(session, job_id="job-before")
    assert BIG_NAME in {tool.name for tool in child_before._tools}
    await child_before.dispose()

    await _idle_runs(session, MCP_DEMOTE_AFTER_IDLE_RUNS)
    assert (await session.compact_now()).ran is True

    child = await build_child(session, job_id="job-after")
    assert BIG_NAME not in {tool.name for tool in child._tools}
    resolved = child._resolve_tool_outside_inventory(BIG_NAME)
    assert resolved is not None and resolved.name == BIG_NAME
    await child.dispose()

    # A restricted child inherits the same resolvable set and still cannot widen.
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness import subagent as subagent_mod

    restricted = await subagent_mod._build_child_session(
        label="review",
        prompt="review",
        parent_session=session,
        model_spec=None,
        job_id="job-r",
        agent="reviewer",
        profile=AgentProfile(name="reviewer", tools=("read", "grep")),
    )
    assert restricted._resolve_tool_outside_inventory(BIG_NAME) is not None
    reply = restricted._skill_resolver(f"mcp://{SERVER}/{SMALL}")
    assert reply is not None and "# MCP tool not enabled" in reply
    assert (SERVER, SMALL) not in _deferred(restricted)
    assert restricted._resolve_tool_outside_inventory(SMALL_NAME) is None
    await restricted.dispose()
    await session.dispose()


@pytest.mark.asyncio
async def test_an_invalid_call_to_a_deferred_mcp_tool_hints_a_tool_doc_that_serves_its_schema(
    tmp_path, monkeypatch
) -> None:
    """Q1 (#3749 follow-up): the loop's invalid-argument hint must lead somewhere.

    A deferred MCP tool is dispatchable through the fallback resolver but absent
    from ``session._tools``, so ``read tool://<name>`` used to answer ``Unknown
    tool``. It must now render the schema — and, because the array is the
    prompt-cache prefix, neither the invalid call nor the doc read may publish
    the schema: every request of both turns carries no MCP tool.
    """
    executed: list[str] = []
    manager = _manager(executed)
    hooks_box: dict[str, Any] = {}
    docs: list[str | None] = []

    async def read_execute(tool_call_id, args, signal, on_update, context):
        url = args.get("path", "")
        if url.startswith("mcp://"):
            text = hooks_box["hooks"].mcp_resolver(url)
        else:
            text = session._skill_resolver(url)
            docs.append(text)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="read", content=[TextContent(text=text or "")]
        )

    read_tool = _tool("read", executed, execute=read_execute)
    read_tool.parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    stream = ScriptedStream(
        [
            _call("read", f'{{"path": "mcp://{SERVER}/{SMALL}"}}'),
            # ``q`` is a string in the schema: an integer fails validation.
            _call(SMALL_NAME, '{"q": 5}', call_id="c2"),
            _call("read", f'{{"path": "tool://{SMALL_NAME}"}}', call_id="c3"),
            _text("done"),
            _text("again"),
        ]
    )
    session = _session(tmp_path, stream, [read_tool])
    hooks_box["hooks"] = await _wire(session, manager, monkeypatch)

    await session.prompt("look at the logs")
    await session.prompt("next")

    fault = _results(session)["c2"]
    assert f"read tool://{SMALL_NAME}" in fault
    assert SMALL_NAME not in executed  # the invalid call never ran
    assert docs and docs[0] is not None
    assert docs[0].startswith(f"# Tool: `{SMALL_NAME}`")
    assert "Unknown tool" not in docs[0]
    assert "`q`" in docs[0]  # the input schema's parameter is rendered
    assert "does not add it to the list" in docs[0]
    # Nothing published: not the invalid call, not the doc read, either turn.
    assert [_sent(stream, i) for i in range(len(stream.requests))] == [[]] * len(stream.requests)
    assert SMALL_NAME not in session._activated_tools
    assert SMALL_NAME not in session._deferred_now()
    await session.dispose()
