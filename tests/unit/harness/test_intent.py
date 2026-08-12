"""The `i` intent field: sanitisation, schema injection, streaming scrape, and
the loop plumbing that turns it into `ToolExecutionStartEvent.intent`.

The regression these pin is narrow and specific: the intent must reach the UI
and must NOT reach the tool. Every builtin params model is pydantic with
``extra="forbid"``, so a leaked key fails the call outright; and the TUI's
argument summary scans argument VALUES for a row caption, so a leaked key also
captions the tool row with the narration — reinstating exactly the duplication
that splitting the fact (the command) from the claim (the intent) removes.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from local_operator.harness.intent import (
    INTENT_DESCRIPTION,
    INTENT_FIELD,
    INTENT_MAX_CHARS,
    INTENT_PROPERTY,
    apply_intent_schema,
    intent_is_injected,
    sanitize_intent,
    scan_streaming_intent,
)
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentTool,
    Message,
    StreamEndEvent,
    TextContent,
    ToolCallComposeEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from tests.unit.harness.test_loop import ScriptedStream, make_config, tool_call_delta

# ---------------------------------------------------------------------------
# sanitize_intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, 3, 3.5, True, {"a": 1}, ["Auditing merged MRs"], b"Auditing merged MRs"],
)
def test_non_string_intent_is_dropped(value: Any) -> None:
    """Streamed JSON delivers objects, numbers and booleans before anything has
    schema-validated them; the guard is a type check, not a null check."""
    assert sanitize_intent(value) is None


@pytest.mark.parametrize("value", ["", "   ", "\t\n  \r\n", "\x1b[2K\x1b[A"])
def test_empty_and_content_free_intents_are_none_not_empty(value: str) -> None:
    """`None`, never `""`: "no intent" already has a spelling, and a second one
    is a case every renderer would have to know about."""
    assert sanitize_intent(value) is None


def test_control_sequences_are_stripped() -> None:
    # Erase-line + cursor-up inside a live frame repaints rows the model does
    # not own. Same treatment the tool name gets before it reaches a frame.
    assert sanitize_intent("\x1b[2K\x1b[AAuditing merged MRs") == "Auditing merged MRs"


def test_newlines_collapse_to_one_line() -> None:
    # The working line is ONE row. A newline that survived would let the model
    # push whatever it liked onto the rows around it.
    assert sanitize_intent("Auditing\nmerged\r\n\tMRs") == "Auditing merged MRs"


def test_long_intent_is_capped() -> None:
    result = sanitize_intent("A" * 10_000)
    assert result is not None
    assert len(result) == INTENT_MAX_CHARS


def test_bidi_override_is_escaped_not_dropped() -> None:
    """RLO reverses the rendered order of what follows, so a narration can read
    as the opposite of what it says. Escaped to something visible rather than
    silently removed."""
    result = sanitize_intent("Writing \u202egnp.terces")
    assert result is not None
    assert "\u202e" not in result
    assert "\\u202e" in result


# ---------------------------------------------------------------------------
# apply_intent_schema / intent_is_injected
# ---------------------------------------------------------------------------


def test_injection_puts_intent_first_and_optional() -> None:
    schema = apply_intent_schema(
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    )
    assert list(schema["properties"]) == [INTENT_FIELD, "path"]
    assert schema["properties"][INTENT_FIELD] == INTENT_PROPERTY
    assert schema["required"] == ["path"]
    assert INTENT_DESCRIPTION in schema["properties"][INTENT_FIELD]["description"]


def test_injection_does_not_mutate_the_input_schema() -> None:
    original = {"type": "object", "properties": {"path": {"type": "string"}}}
    apply_intent_schema(original)
    assert original == {"type": "object", "properties": {"path": {"type": "string"}}}


def test_injection_handles_a_schema_with_no_properties() -> None:
    assert apply_intent_schema({"type": "object"})["properties"] == {INTENT_FIELD: INTENT_PROPERTY}
    assert apply_intent_schema(None)["properties"] == {INTENT_FIELD: INTENT_PROPERTY}


def test_a_schema_owning_i_is_left_alone() -> None:
    """An MCP server's schema is its own. Overwriting a real `i` parameter
    would drop a real argument, so injection skips it and `intent_is_injected`
    reports False — which is what stops the loop lifting the value away."""
    own = {"type": "object", "properties": {INTENT_FIELD: {"type": "integer"}}}
    assert apply_intent_schema(own) == own
    assert intent_is_injected(apply_intent_schema(own)) is False


def test_intent_is_injected_recognises_only_our_property() -> None:
    assert intent_is_injected(apply_intent_schema({"type": "object"})) is True
    assert intent_is_injected({"type": "object", "properties": {}}) is False
    assert intent_is_injected({}) is False
    assert intent_is_injected(None) is False


# ---------------------------------------------------------------------------
# scan_streaming_intent
# ---------------------------------------------------------------------------


def test_scrape_reads_a_closed_leading_intent() -> None:
    assert scan_streaming_intent('{"i": "Auditing merged MRs", "path": "a') == "Auditing merged MRs"


def test_scrape_is_none_until_the_string_closes() -> None:
    """A label that grows character by character on a repainting row is worse
    than no label."""
    assert scan_streaming_intent('{"i": "Auditing mer') is None
    assert scan_streaming_intent('{"i": ') is None
    assert scan_streaming_intent("{") is None
    assert scan_streaming_intent("") is None


def test_scrape_ignores_a_non_leading_intent() -> None:
    """Anchored at the head of the arguments: that is what makes a false
    positive impossible without tracking nesting depth."""
    assert scan_streaming_intent('{"path": "a.py", "i": "Reading a file"}') is None


def test_scrape_cannot_be_forged_from_inside_a_string_value() -> None:
    # An `"i"` written inside another JSON string arrives backslash-escaped,
    # so it can never open this match.
    assert scan_streaming_intent('{"content": "\\"i\\": \\"Deleting everything\\""}') is None


def test_scrape_decodes_escapes_and_survives_malformed_ones() -> None:
    assert scan_streaming_intent('{"i":"Reading \\u00e9tudes"}') == "Reading études"
    assert scan_streaming_intent('{"i":"Reading \\q"}') is None


def test_scrape_applies_the_same_sanitisation() -> None:
    assert scan_streaming_intent('{"i":"Auditing\\nmerged MRs"}') == "Auditing merged MRs"
    assert scan_streaming_intent('{"i":"   "}') is None


# ---------------------------------------------------------------------------
# Loop plumbing
# ---------------------------------------------------------------------------


class _StrictParams(BaseModel):
    """Mirrors every builtin params model: extra keys are a hard error, which
    is why a leaked `i` is a failed call and not untidiness."""

    model_config = ConfigDict(extra="forbid")

    text: str


def _strict_tool(seen: list[dict[str, Any]], *, inject: bool = True) -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        seen.append(dict(args))
        params = _StrictParams(**args)  # raises on a leaked intent key
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text=params.text)]
        )

    schema = _StrictParams.model_json_schema()
    return AgentTool(
        name="echo",
        parameters=apply_intent_schema(schema) if inject else schema,
        execute=execute,
    )


async def _run(tool: AgentTool, raw_args: str) -> list[Any]:
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_1", name="echo"),
                tool_call_delta(0, args=raw_args),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(system_blocks=["sys"], tools=[tool])
    return [
        event
        async for event in AgentLoop().run([Message.user("go")], context, make_config(stream), None)
    ]


def _starts(events: list[Any]) -> list[ToolExecutionStartEvent]:
    return [e for e in events if isinstance(e, ToolExecutionStartEvent)]


@pytest.mark.asyncio
async def test_intent_reaches_the_event_and_never_the_tool() -> None:
    """The regression the whole split exists for: `i` in, narration out, and
    the key absent from BOTH the executed arguments and the event's args."""
    seen: list[dict[str, Any]] = []
    events = await _run(_strict_tool(seen), '{"i": "Auditing merged MRs", "text": "hi"}')

    start = _starts(events)[0]
    assert start.intent == "Auditing merged MRs"
    assert INTENT_FIELD not in start.args
    assert start.args == {"text": "hi"}
    assert seen == [{"text": "hi"}]
    assert INTENT_FIELD not in seen[0]
    assert not any(getattr(e, "is_error", False) for e in events)


@pytest.mark.asyncio
async def test_call_without_an_intent_behaves_exactly_as_before() -> None:
    seen: list[dict[str, Any]] = []
    events = await _run(_strict_tool(seen), '{"text": "hi"}')

    start = _starts(events)[0]
    assert start.intent is None
    assert start.args == {"text": "hi"}
    assert seen == [{"text": "hi"}]


@pytest.mark.asyncio
async def test_a_malformed_intent_costs_the_narration_and_nothing_else() -> None:
    """A non-string `i` type-checks as a declared property, so validating
    before lifting would fail the whole call — and a planning failure emits no
    `tool_execution_start` at all, silently swallowing the user's work."""
    seen: list[dict[str, Any]] = []
    events = await _run(_strict_tool(seen), '{"i": 3, "text": "hi"}')

    start = _starts(events)[0]
    assert start.intent is None
    assert start.args == {"text": "hi"}
    assert seen == [{"text": "hi"}]


@pytest.mark.asyncio
async def test_intent_is_sanitised_before_it_reaches_the_event() -> None:
    seen: list[dict[str, Any]] = []
    events = await _run(
        _strict_tool(seen), '{"i": "\\u001b[2K\\u001b[AAuditing\\nmerged MRs", "text": "hi"}'
    )
    assert _starts(events)[0].intent == "Auditing merged MRs"


@pytest.mark.asyncio
async def test_a_tool_owning_i_keeps_its_own_argument() -> None:
    """No injection means no lift: an MCP server that declares `i` gets its
    value forwarded, and the harness claims no narration it was not given."""
    seen: list[dict[str, Any]] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        seen.append(dict(args))
        return ToolResult(tool_call_id=tool_call_id, tool_name="echo", content=[])

    tool = AgentTool(
        name="echo",
        parameters=apply_intent_schema(
            {"type": "object", "properties": {INTENT_FIELD: {"type": "string"}}}
        ),
        execute=execute,
    )
    events = await _run(tool, '{"i": "server-owned"}')

    start = _starts(events)[0]
    assert start.intent is None
    assert start.args == {INTENT_FIELD: "server-owned"}
    assert seen == [{INTENT_FIELD: "server-owned"}]


@pytest.mark.asyncio
async def test_compose_events_carry_the_intent_once_it_closes() -> None:
    """The longest silence in a turn is a large argument streaming in, and
    until now the row read `composing write` for the whole of it."""
    seen: list[dict[str, Any]] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_1", name="echo"),
                tool_call_delta(0, args='{"i": "Auditing mer'),
                tool_call_delta(0, args='ged MRs", "text": "hi"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(system_blocks=["sys"], tools=[_strict_tool(seen)])
    events = [
        event
        async for event in AgentLoop().run([Message.user("go")], context, make_config(stream), None)
    ]

    composes = [e for e in events if isinstance(e, ToolCallComposeEvent)]
    assert composes, "compose events are what this feature rides on"
    # Never a half-word: every frame shows either nothing or the closed string.
    assert {e.intent for e in composes} <= {None, "Auditing merged MRs"}
    assert composes[-1].intent == "Auditing merged MRs"
