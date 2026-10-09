"""Deferred tool schemas: withheld from the wire, never from the session.

``tools/deferral.py`` names the built-in tools whose schema a session leaves out
of the provider ``tools`` array until it is activated. The capability contract
is that NOTHING about reachability changes: the tool resolves, validates, passes
the approval gate and answers ``tool://`` exactly as before. These tests are
written against what the provider is SENT and what the loop RESOLVES, because
those are the two halves the contract is made of.
"""

from __future__ import annotations

import json
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
    ToolContext,
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
    """A typo in the deferral set defers nothing, silently; a missing purpose
    phrase leaves a tool on the inventory line with no reason to reach for it."""
    assert DEFERRED_TOOLS <= set(DEFAULT_TOOL_NAMES), DEFERRED_TOOLS - set(DEFAULT_TOOL_NAMES)
    assert DEFERRED_TOOLS <= set(DEFERRED_TOOL_PURPOSES), DEFERRED_TOOLS - set(
        DEFERRED_TOOL_PURPOSES
    )
    # ``ask`` is named by the <interactivity> bodies as the channel to the
    # operator, so a session must never have to discover it.
    assert "ask" not in DEFERRED_TOOLS
    # The child-only extras were measured and DROPPED (adoption collapse); this
    # pins the finding so a later "cheap win" does not quietly re-add them.
    assert not ({"project", "sessions", "send", "agent", "secret", "network"} & DEFERRED_TOOLS)


def test_a_pin_keeps_a_named_tool_published() -> None:
    assert "team" in deferred_tool_names()
    assert "team" not in deferred_tool_names(pinned=("team", "read"))


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


# -- plan item 6: allowlists and pins ---------------------------------------


@pytest.mark.asyncio
async def test_a_declared_inventory_pins_what_the_host_named(tmp_path) -> None:
    """``--tools console,bash`` (and ``AGENTS_CONFIG_TOOLS``) is a host saying
    which tools this run IS, so the run must not have to discover them."""
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed), _tool("console", executed), _tool("lsp", executed)],
    )
    session.set_tool_inventory(["read", "console"])
    await session.prompt("go")
    assert stream.names(0) == ["read", "console"]
    assert session.deferred_tool_names() == frozenset()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_tool_that_refuses_itself_gets_the_hint_too(tmp_path) -> None:
    """QA round 1, Q2: a tool whose own pydantic model rejects the arguments
    returns the refusal from its BODY, so the plan-time hint never runs and the
    model got no pointer to a schema it was never sent."""
    from local_operator.harness.types import FAULT_INVALID_ARGUMENTS, FAULT_KEY

    async def refuse(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="console",
            is_error=True,
            content=[TextContent(text="invalid arguments:\n- method: Input should be 'list'")],
            details={FAULT_KEY: FAULT_INVALID_ARGUMENTS},
        )

    stream = ScriptedStream([_call("console"), _text("done")])
    session = _session(tmp_path, stream, [_tool("read", []), _tool("console", [], execute=refuse)])
    await session.prompt("go")

    text = _results(session)["c1"]
    assert text.endswith("read tool://console"), text
    await session.dispose()


@pytest.mark.asyncio
async def test_a_published_tool_that_refuses_itself_gets_no_hint(tmp_path) -> None:
    """The converse: the hint would be a lie for a schema the model was sent."""
    from local_operator.harness.types import FAULT_INVALID_ARGUMENTS, FAULT_KEY

    async def refuse(tool_call_id, args, signal, on_update, context):
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="read",
            is_error=True,
            content=[TextContent(text="invalid arguments:\n- range: bad range")],
            details={FAULT_KEY: FAULT_INVALID_ARGUMENTS},
        )

    stream = ScriptedStream([_call("read"), _text("done")])
    session = _session(tmp_path, stream, [_tool("read", [], execute=refuse), _tool("console", [])])
    await session.prompt("go")

    assert _results(session)["c1"] == "invalid arguments:\n- range: bad range"
    await session.dispose()


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
async def test_a_pin_keeps_the_role_tools_published(tmp_path) -> None:
    """A role's ``tools:`` list is a request for those tools, so a sandboxed
    session must not make its own player discover them."""
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(
        tmp_path,
        stream,
        [_tool("read", executed), _tool("team", executed), _tool("console", executed)],
    )
    session.set_tool_deferral(pins=("read", "team"))
    await session.prompt("go")
    assert stream.names(0) == ["read", "team"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declared_inventory_pins_its_own_names(tmp_path) -> None:
    """A host that declares the run's tools asked for them (review round 1,
    MINOR-2): ``lop exec --tools console,bash`` must not make the model
    discover ``console``'s shape."""
    executed: list[str] = []
    stream = ScriptedStream([_text("hi")])
    session = _session(tmp_path, stream, [_tool("bash", executed), _tool("console", executed)])
    session.set_tool_inventory(["bash", "console"])
    await session.prompt("go")
    assert stream.names(0) == ["bash", "console"]
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

    assert "ask" not in deferred_tool_names()
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
    # NOTHING but the collapse's own rewrite lands in the schema: an earlier
    # revision carried the "I am collapsed" flag as an ``x-`` key here, which
    # rode 121 properties onto every published request (review round 2,
    # MAJOR-2). The names travel beside the schema instead.
    assert out.parameters["properties"]["opt"] == {"type": "string", "description": "d"}
    assert out.parameters["properties"]["req"] == schema["properties"]["req"]
    assert out.parameters["properties"]["multi"] == schema["properties"]["multi"]
    assert out.parameters["properties"]["kept_default"] == schema["properties"]["kept_default"]
    assert out.parameters["$defs"]["Item"]["properties"]["x"] == {"type": "integer"}
    # ROOT-level names only: the loop reads the root ``properties`` and nothing
    # else, so the nested ``Item.x`` is deliberately NOT reported.
    assert out.unchecked == frozenset({"opt"})
    # The input is not mutated (schemas are shared by pydantic's cache).
    assert "anyOf" in schema["properties"]["opt"]


def test_the_loop_does_not_type_check_a_collapsed_property() -> None:
    """A collapsed property keeps the BASE semantics: unchecked at the loop.

    The union shape this loop has always read as "not mine to check" (no
    top-level ``type``) is restored by the marker, so a tool's own coercer
    still decides. A REQUIRED ``string`` is checked exactly as before.
    """
    collapsed = collapse_optional_nulls(
        {
            "type": "object",
            "properties": {
                "opt": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                "req": {"type": "string"},
            },
            "required": ["req"],
        }
    )
    tool = _tool("t", [], parameters=collapsed.parameters).model_copy(
        update={"optional_null_unions": collapsed.unchecked}
    )
    assert validate_tool_arguments(tool, {"req": "a", "opt": None}) == []
    assert validate_tool_arguments(tool, {"req": "a", "opt": 3}) == []
    assert validate_tool_arguments(tool, {"req": None}) == [
        "argument 'req' does not match type string"
    ]


def _uncollapsed_parameters() -> dict[str, Any]:
    """Every tool's schema with the collapse DISABLED.

    The source of truth for "what did the collapse change": rebuilding the real
    surface with the rewrite neutralised and diffing it against the published
    one needs no walk over the schema tree, so it cannot miss a nested ``$defs``
    entry the way a hand-written traversal did (review round 2, MINOR-3).
    """
    from local_operator.tools import registry
    from scripts.real_tool_surface import build_real_tools

    original = registry.collapse_optional_nulls
    registry.collapse_optional_nulls = lambda schema: registry.CollapsedSchema(schema, frozenset())
    try:
        return {tool.name: tool.parameters for tool in build_real_tools(".")}
    finally:
        registry.collapse_optional_nulls = original


def test_the_reported_set_is_exactly_what_the_collapse_changed() -> None:
    """THE structural invariant MAJOR-1 needs, over the FULL tool surface.

    A collapsed property missing from ``optional_null_unions`` is a call the
    loop refuses before the tool can coerce it — which is how ``hub``'s ``to``
    and ``jobs``' ``job_id`` lost their documented input forms — and a name in
    it that was never collapsed is a type check silently switched off. So the
    set is compared for EQUALITY against a diff of the real surface against
    itself with the collapse neutralised: no sampling, no walk, every tool.

    The converse half is what gives it teeth: a ROOT property that is required
    and plain must still be type-checked, so a validator that simply stopped
    checking cannot pass this test.
    """
    from scripts.real_tool_surface import build_real_tools

    uncollapsed = _uncollapsed_parameters()
    collapsed = build_real_tools(".")
    assert len(uncollapsed) == len(collapsed) > 20, "the surface moved"

    for tool in collapsed:
        before = uncollapsed[tool.name].get("properties") or {}
        after = tool.parameters.get("properties") or {}
        changed = {
            name
            for name, prop in before.items()
            if json.dumps(prop, sort_keys=True) != json.dumps(after.get(name), sort_keys=True)
        }
        assert set(tool.optional_null_unions) == changed, (
            f"{tool.name}: reported {sorted(tool.optional_null_unions)} but the "
            f"collapse changed {sorted(changed)}"
        )
        for name in changed:
            # Each reported name really did lose its union AND gained a type the
            # loop would otherwise enforce — that is the whole reason it is here.
            assert "anyOf" not in after[name], (tool.name, name)
            assert isinstance(after[name].get("type"), str), (tool.name, name)
            errors = validate_tool_arguments(tool, {name: {"wrong": "type"}})
            assert not [error for error in errors if f"'{name}'" in error], (
                f"{tool.name}.{name} is collapsed, but the loop still type-checks "
                "it — a coerced form would be refused before the tool ran"
            )

        properties = tool.parameters.get("properties") or {}
        typed = next(
            (
                field
                for field in tool.parameters.get("required") or []
                if isinstance(properties.get(field), dict)
                and isinstance(properties[field].get("type"), str)
                and field not in tool.optional_null_unions
            ),
            None,
        )
        if typed is None:
            continue
        errors = validate_tool_arguments(tool, {typed: {"wrong": "type"}})
        assert [error for error in errors if f"'{typed}'" in error], (
            f"{tool.name}.{typed} is a plain required property and the loop no "
            "longer type-checks it — this test would pass on a validator that "
            "checked nothing"
        )

    reported = sum(len(tool.optional_null_unions) for tool in collapsed)
    assert reported > 100, f"only {reported} collapsed properties reported — the collapse moved?"


def test_an_unmarked_optional_property_is_still_type_checked() -> None:
    """MINOR-1's guard: the null tolerance lives ONLY in the reported set.

    The runtime scope was right and unguarded — an unconditional "a null is
    acceptable for an optional property" escape could be re-added and every
    other test stayed green. This is the shape it would swallow: an optional
    property with a plain ``type`` and no null branch, which is what an MCP
    server's schema looks like.
    """
    tool = _tool(
        "mcp_like",
        [],
        parameters={
            "type": "object",
            "properties": {"opt": {"type": "string"}, "req": {"type": "integer"}},
            "required": ["req"],
        },
    )
    assert validate_tool_arguments(tool, {"req": 1, "opt": None}) == [
        "argument 'opt' does not match type string"
    ]
    assert validate_tool_arguments(tool, {"req": 1, "opt": "a"}) == []


@pytest.mark.asyncio
async def test_a_builtin_still_accepts_an_explicit_null_for_a_collapsed_field(tmp_path) -> None:
    """End to end over a real builtin: ``read``'s optional ``range`` reaches the
    wire without a null branch, and a model that sends null is not refused."""
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


# -- MAJOR-1: the collapse must not make the loop refuse a coerced form ------


def test_the_loop_still_accepts_the_forms_the_tools_coerce() -> None:
    """Over the REAL schemas, through the loop's own validator.

    ``hub``'s ``to`` takes a bare job id or the JSON of the list and ``jobs``'
    ``job_id`` takes a number — both via ``mode="before"`` coercers the tools
    ship on purpose. At base those properties were ``anyOf: [T, null]``, so the
    loop checked nothing and the coercer decided; collapsing the union put a
    ``type`` at the top level and refused the call before the tool ran (review
    round 1, MAJOR-1). The existing coverage could not catch it because it
    called ``execute_hub`` directly and so never touched this validator.
    """
    from scripts.real_tool_surface import build_real_tools

    by_name = {tool.name: tool for tool in build_real_tools(".")}
    hub, jobs = by_name["hub"], by_name["jobs"]
    # The documented string forms, and a third the union itself allowed.
    assert validate_tool_arguments(hub, {"op": "list", "to": "job-1"}) == []
    assert validate_tool_arguments(hub, {"op": "list", "to": '["job-1"]'}) == []
    assert validate_tool_arguments(hub, {"op": "list", "to": ["job-1"]}) == []
    assert validate_tool_arguments(hub, {"op": "list", "to": 5}) == []
    # ``jobs``: the numeric form a model emits for an id it read as a number.
    assert validate_tool_arguments(jobs, {"op": "peek", "job_id": 1}) == []
    assert validate_tool_arguments(jobs, {"op": "peek", "job_id": "1"}) == []


@pytest.mark.asyncio
async def test_a_numeric_job_id_survives_the_loop_and_reaches_the_tool(tmp_path) -> None:
    """The same finding, driven end to end through ``_plan_call`` on a Session.

    A validator-level assertion proves the schema is right; this proves the loop
    actually carries the call through to execution, which is the half the
    reviewer's repro measured going wrong.
    """
    from local_operator.harness.jobs import AsyncJobManager
    from local_operator.tools.registry import create_tools

    stream = ScriptedStream([_call("jobs", '{"op": "peek", "job_id": 1}'), _text("done")])
    (jobs_tool,) = create_tools(
        ToolContext(cwd=str(tmp_path), jobs=AsyncJobManager()), enabled=["jobs"]
    )
    session = _session(tmp_path, stream, [jobs_tool])
    await session.prompt("go")

    text = _results(session)["c1"]
    assert "does not match type string" not in text, text
    # It reached the executor: an unknown id is the tool's OWN refusal.
    assert "unknown job" in text, text
    await session.dispose()


def test_the_side_set_is_what_the_loop_reads_not_an_absence_of_type() -> None:
    """A reported property IS type-checked again once the report is dropped.

    The fail-proof direction for the invariants above: empty the tool's
    ``optional_null_unions`` and the loop refuses the coerced form, so those
    tests go red on a regression instead of passing because they found nothing.
    """
    from scripts.real_tool_surface import build_real_tools

    hub = next(tool for tool in build_real_tools(".") if tool.name == "hub")
    unreported = hub.model_copy(update={"optional_null_unions": frozenset()})
    errors = validate_tool_arguments(unreported, {"op": "list", "to": "job-1"})
    assert [error for error in errors if "'to'" in error], (
        "with the set dropped the loop must type-check 'to' again — if this passes, "
        "the set is not what makes the coerced forms work"
    )
