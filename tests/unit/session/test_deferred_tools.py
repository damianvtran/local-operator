"""Deferred tool schemas: withheld from the wire, never from the session.

``tools/deferral.py`` names the built-in tools whose schema a session leaves out
of the provider ``tools`` array until it is activated. The capability contract
is that NOTHING about reachability changes: the tool resolves, validates, passes
the approval gate and answers ``tool://`` exactly as before. These tests are
written against what the provider is SENT and what the loop RESOLVES, because
those are the two halves the contract is made of.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from local_operator.harness.loop import validate_tool_arguments
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.prompts_api import build_system_blocks, render_tool_inventory_block
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.deferral import (
    DEFERRED_TOOL_PURPOSES,
    DEFERRED_TOOLS,
    deferred_tool_names,
    tool_deferral_enabled,
)
from local_operator.tools.registry import DEFAULT_TOOL_NAMES, collapse_optional_nulls

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class ScriptedStream:
    """Replays per-call event scripts; records every request it was sent."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        index = len(self.requests) - 1
        turn = self.turns[index] if index < len(self.turns) else _text("done")

        async def gen():
            for event in turn:
                yield event

        return gen()

    #: The names a test supplied. ``Session.__init__`` also merges its own
    #: capability tools (``task``, ``hub``, ``wait``...) into the inventory;
    #: they are not what these tests are about, so ``names`` reads only these.
    under_test: frozenset[str] = frozenset()

    def names(self, index: int) -> list[str]:
        names = [tool.name for tool in self.requests[index].tools]
        return [name for name in names if not self.under_test or name in self.under_test]


def _text(text: str) -> list[StreamEvent]:
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


def _call(name: str, args: str = "{}", call_id: str = "c1") -> list[StreamEvent]:
    return [
        StreamToolCallDelta(index=0, id=call_id, name=name, argument_delta=args),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def _tool(
    name: str,
    executed: list[str],
    *,
    parameters: dict[str, Any] | None = None,
    tier: str = "read",
    execute: Any = None,
) -> AgentTool:
    async def default_execute(tool_call_id, args, signal, on_update, context):
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text=f"{name} ran")]
        )

    return AgentTool(
        name=name,
        description=f"{name} tool",
        parameters=parameters or {"type": "object", "properties": {}},
        approval_tier=tier,  # type: ignore[arg-type]
        execute=execute or default_execute,
    )


def _session(tmp_path, stream, tools: Sequence[AgentTool], **kwargs: Any) -> Session:
    if isinstance(stream, ScriptedStream):
        stream.under_test = frozenset(tool.name for tool in tools)
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=list(tools),
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
        **kwargs,
    )
    # Hermetic: the constructor reads ``tools.defer`` from the (isolated) config.
    session._tool_deferral = True
    return session


def _results(session: Session) -> dict[str, str]:
    return {
        message.tool_call_id: "".join(getattr(part, "text", "") for part in message.content)
        for message in session._context.messages
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }


# -- the sets ----------------------------------------------------------------


def test_every_deferred_name_is_a_real_tool_with_a_purpose() -> None:
    """A typo in a deferral set defers nothing, silently; a missing purpose
    phrase leaves a tool on the inventory line with no reason to reach for it."""
    for kind, names in DEFERRED_TOOLS.items():
        assert names <= set(DEFAULT_TOOL_NAMES), (kind, names - set(DEFAULT_TOOL_NAMES))
        assert names <= set(DEFERRED_TOOL_PURPOSES), (kind, names - set(DEFERRED_TOOL_PURPOSES))
    assert DEFERRED_TOOLS["top"] < DEFERRED_TOOLS["child"]
    # ``ask`` is named by the <interactivity> bodies as the channel to the
    # operator, so a top-level session must never have to discover it.
    assert "ask" not in DEFERRED_TOOLS["top"]


def test_a_role_pin_keeps_a_named_tool_published() -> None:
    assert "project" in deferred_tool_names("child")
    assert "project" not in deferred_tool_names("child", pinned=("project", "read"))


def test_the_kill_switch_reads_only_an_explicit_false() -> None:
    assert tool_deferral_enabled({}) is True
    assert tool_deferral_enabled({"tools": {"defer": False}}) is False
    assert tool_deferral_enabled({"tools": {"defer": "no"}}) is True  # not a bool
    assert tool_deferral_enabled({"tools": "garbage"}) is True


# -- plan item 1: the array, the side channels, the inventory ---------------


@pytest.mark.asyncio
async def test_deferred_schema_leaves_the_array_but_not_the_inventory(tmp_path) -> None:
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(tmp_path, stream, [_tool("read", executed), _tool("console", executed)])
    await session.prompt("go")

    mine = {"read", "console"}
    assert stream.names(0) == ["read"]
    # The side channels must reproduce the turn's array byte for byte.
    assert session._side_channel_tools() == list(stream.requests[0].tools)
    assert session._wire_tools() == list(stream.requests[0].tools)
    # Resolution reads the full inventory.
    assert [tool.name for tool in session._context.tools if tool.name in mine] == [
        "read",
        "console",
    ]
    assert session.deferred_tool_names() == {"console"}
    # ``context_breakdown`` counts what is SENT — the latched array — so the
    # kill switch moves it only from the next publish (the turn boundary
    # clears the latch, reproduced here).
    published = session.context_breakdown()["tool_schemas"]
    session._tool_deferral = False
    assert session.context_breakdown()["tool_schemas"] == published
    session._published_tools = None
    assert session.context_breakdown()["tool_schemas"] > published
    await session.dispose()


# -- plan item 2: a direct call executes, through the approval gate ----------


@pytest.mark.asyncio
async def test_a_direct_call_to_a_deferred_tool_runs_and_is_gated(tmp_path) -> None:
    executed: list[str] = []
    asked: list[str] = []

    async def approve(tool_name: str, description: str) -> bool:
        asked.append(tool_name)
        return True

    stream = ScriptedStream([_call("console"), _text("done")])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed), _tool("console", executed, tier="exec")],
        request_approval=approve,
    )
    await session.prompt("go")

    assert "console" not in stream.names(0)
    assert executed == ["console"]
    assert asked == ["console"]
    assert _results(session)["c1"] == "console ran"
    # A VALID direct call does not publish: the model already had the shape.
    assert "console" in session._deferred_now()
    await session.dispose()


# -- plan item 3: the eval tool() bridge ------------------------------------


@pytest.mark.asyncio
async def test_the_eval_bridge_dispatches_a_deferred_tool(tmp_path) -> None:
    executed: list[str] = []
    seen: list[dict[str, Any]] = []
    recorded: list[tuple[str, str, str]] = []

    async def eval_execute(tool_call_id, args, signal, on_update, context):
        seen.append(await context.dispatch_tool("lsp", {}))
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="eval", content=[TextContent(text="ok")]
        )

    stream = ScriptedStream([_call("eval", '{"code": "x"}'), _text("done")])
    session = _session(
        tmp_path,
        stream,
        [
            _tool("eval", executed, execute=eval_execute),
            _tool("lsp", executed),
        ],
        session_id="sess-eval",
    )
    original = session._record_tool_call

    def spy(tool_name: str, origin: str, fault: str, duration_ms: float) -> None:
        recorded.append((tool_name, origin, fault))
        original(tool_name, origin, fault, duration_ms)

    session._record_tool_call = spy  # type: ignore[method-assign]
    await session.prompt("go")

    assert "lsp" not in stream.names(0)
    assert executed == ["lsp"]
    assert seen and seen[0]["is_error"] is False
    assert ("lsp", "nested", "") in recorded
    await session.dispose()


# -- plan item 4: read tool://X activates for the NEXT turn -----------------


@pytest.mark.asyncio
async def test_reading_tool_doc_mid_turn_publishes_at_the_next_turn(tmp_path) -> None:
    executed: list[str] = []
    replies: list[str | None] = []

    async def read_execute(tool_call_id, args, signal, on_update, context):
        replies.append(context.resolve_internal_url("tool://console"))
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="read", content=[TextContent(text="read")]
        )

    stream = ScriptedStream([_call("read"), _text("done"), _text("again")])
    session = _session(
        tmp_path,
        stream,
        [
            _tool("read", executed, execute=read_execute),
            _tool("bash", executed),
            _tool("console", executed),
        ],
    )
    await session.prompt("go")
    # Both calls of the turn carry the array the turn started with.
    assert stream.names(0) == ["read", "bash"]
    assert stream.names(1) == ["read", "bash"]
    assert replies[0] is not None
    # The doc renders as before, with the availability note appended.
    assert replies[0].splitlines()[0].startswith("# ") and "console" in replies[0].splitlines()[0]
    assert "NEXT TURN" in replies[0]

    await session.prompt("next")
    # Registry order, not appended: console sits where the inventory has it.
    assert stream.names(2) == ["read", "bash", "console"]
    # Sticky: a second read activates nothing and appends no note.
    assert "schema loaded" not in (session._skill_resolver("tool://console") or "")
    await session.dispose()


@pytest.mark.asyncio
async def test_reading_tool_doc_before_a_turn_publishes_on_its_first_call(tmp_path) -> None:
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(tmp_path, stream, [_tool("read", executed), _tool("console", executed)])
    reply = session._skill_resolver("tool://console") or ""
    assert "next model call" in reply
    await session.prompt("go")
    assert stream.names(0) == ["read", "console"]
    await session.dispose()


# -- plan item 5: an invalid direct call activates and names tool:// --------


@pytest.mark.asyncio
async def test_an_invalid_direct_call_activates_and_points_at_the_doc(tmp_path) -> None:
    executed: list[str] = []
    schema = {
        "type": "object",
        "properties": {"method": {"type": "string"}},
        "required": ["method"],
    }
    stream = ScriptedStream([_call("console", "{}"), _text("done"), _text("again")])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed), _tool("console", executed, parameters=schema)],
    )
    await session.prompt("go")

    text = _results(session)["c1"]
    assert text.startswith("Invalid arguments: missing required argument 'method'")
    assert "read tool://console" in text
    assert executed == []
    await session.prompt("next")
    assert "console" in stream.names(2)
    await session.dispose()


@pytest.mark.asyncio
async def test_an_invalid_call_to_a_published_tool_gets_no_tool_doc_hint(tmp_path) -> None:
    """The hint is for a GUESSED shape: a published schema was already seen."""
    executed: list[str] = []
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    stream = ScriptedStream([_call("read"), _text("done")])
    session = _session(
        tmp_path, stream, [_tool("read", executed, parameters=schema), _tool("console", executed)]
    )
    await session.prompt("go")
    text = _results(session)["c1"]
    assert text == "Invalid arguments: missing required argument 'path'"
    assert "console" in session._deferred_now()
    await session.dispose()


# -- plan item 6: allowlists and profile pins -------------------------------


@pytest.mark.asyncio
async def test_a_declared_inventory_still_excludes_a_deferred_tool(tmp_path) -> None:
    """Deferral never widens reach: an excluded deferred tool stays unreachable."""
    executed: list[str] = []
    stream = ScriptedStream([_call("console"), _text("done")])
    session = _session(tmp_path, stream, [_tool("read", executed), _tool("console", executed)])
    session.set_tool_inventory(["read"])
    await session.prompt("go")
    assert executed == []
    assert _results(session)["c1"] == "Tool not found: console"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_child_defers_its_set_and_a_role_pin_keeps_project(tmp_path) -> None:
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed), _tool("send", executed), _tool("project", executed)],
    )
    session.set_tool_deferral("child", pins=("read", "project"))
    await session.prompt("go")
    assert stream.names(0) == ["read", "project"]
    await session.dispose()


# -- plan item 7: the inventory block does not move on activation ----------


@pytest.mark.asyncio
async def test_the_inventory_block_is_identical_before_and_after_activation(tmp_path) -> None:
    executed: list[str] = []
    tools = [_tool("read", executed), _tool("console", executed)]
    session = _session(tmp_path, ScriptedStream([]), tools)
    before = render_tool_inventory_block(
        tools, host_has_browser=True, host_has_console=True, deferred=session.deferred_tool_names()
    )
    assert session.activate_deferred_tool("console") is True
    after = render_tool_inventory_block(
        tools, host_has_browser=True, host_has_console=True, deferred=session.deferred_tool_names()
    )
    assert before == after
    assert "- read" in before and "- console" not in before
    assert "Schema on demand" in before and "console (drive an interactive terminal)" in before
    assert "call these directly by name" in before
    await session.dispose()


# -- plan item 8: ``ask`` stays published at top level ----------------------


def test_ask_and_its_interactivity_body_are_unchanged() -> None:
    from local_operator.prompts_api import CHANNEL_ASK

    assert "ask" not in deferred_tool_names("top")
    ask = _tool("ask", [])
    with_deferral = build_system_blocks(
        [ask], "", "", "2026-01-01", interactive=True, channel=CHANNEL_ASK, deferred_tools={"lsp"}
    )
    without = build_system_blocks(
        [ask], "", "", "2026-01-01", interactive=True, channel=CHANNEL_ASK
    )
    assert with_deferral == without


# -- plan item 10: the inverse canary ---------------------------------------


@pytest.mark.asyncio
async def test_turning_deferral_off_publishes_every_schema_again(tmp_path) -> None:
    executed: list[str] = []
    stream = ScriptedStream([_text("hi"), _text("again")])
    session = _session(tmp_path, stream, [_tool("read", executed), _tool("console", executed)])
    await session.prompt("go")
    assert stream.names(0) == ["read"]

    class Change:
        changed_keys = frozenset({"tools.defer"})
        values = {"tools": {"defer": False}}
        source = "disk"

    session._apply_config_change(Change())
    await session.prompt("next")
    assert stream.names(1) == ["read", "console"]
    assert session.deferred_tool_names() == frozenset()
    await session.dispose()


# -- the nullable trim and its validator ------------------------------------


def test_optional_null_unions_collapse_and_required_ones_do_not() -> None:
    schema = {
        "type": "object",
        "properties": {
            "opt": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "d",
            },
            "req": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "multi": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]},
            "kept_default": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": 3},
        },
        "required": ["req"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {
                    "x": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}
                },
            }
        },
    }
    out = collapse_optional_nulls(schema)
    assert out["properties"]["opt"] == {"type": "string", "description": "d"}
    assert out["properties"]["req"] == schema["properties"]["req"]
    assert out["properties"]["multi"] == schema["properties"]["multi"]
    assert out["properties"]["kept_default"] == schema["properties"]["kept_default"]
    assert out["$defs"]["Item"]["properties"]["x"] == {"type": "integer"}
    # The input is not mutated (schemas are shared by pydantic's cache).
    assert "anyOf" in schema["properties"]["opt"]


def test_the_validator_accepts_null_for_an_optional_property_only() -> None:
    tool = _tool(
        "t",
        [],
        parameters=collapse_optional_nulls(
            {
                "type": "object",
                "properties": {
                    "opt": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                    "req": {"type": "string"},
                },
                "required": ["req"],
            }
        ),
    )
    assert validate_tool_arguments(tool, {"req": "a", "opt": None}) == []
    assert validate_tool_arguments(tool, {"req": "a", "opt": 3}) == [
        "argument 'opt' does not match type string"
    ]
    assert validate_tool_arguments(tool, {"req": None}) == [
        "argument 'req' does not match type string"
    ]


@pytest.mark.asyncio
async def test_a_builtin_still_accepts_an_explicit_null_for_a_collapsed_field(tmp_path) -> None:
    """End to end over a real builtin: ``read``'s optional ``range`` reaches the
    wire without a null branch, and a model that sends null is not refused."""
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    (read,) = create_tools(ToolContext(cwd=str(tmp_path)), enabled=["read"])
    assert "anyOf" not in read.parameters["properties"]["range"]
    target = tmp_path / "f.txt"
    target.write_text("hello\n")
    assert validate_tool_arguments(read, {"path": str(target), "range": None}) == []
    result = await read.execute(
        "c1", {"path": str(target), "range": None}, None, None, ToolContext(cwd=str(tmp_path))
    )
    assert not result.is_error and "hello" in result.text
