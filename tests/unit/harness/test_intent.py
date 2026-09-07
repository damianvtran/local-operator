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


def test_injection_strips_pydantic_titles_but_keeps_descriptions() -> None:
    """``title`` is a restatement of the key; ``description`` is the adherence
    surface.

    Pydantic emits ``"title": "Path"`` beside every property and a model title
    on every ``$defs`` entry. No provider requires it and no model needs it,
    but it rides the tools array on every request — 148 keys costing 2,953
    characters (~1,060 billed) across the 24-tool default surface, measured
    with ``json.dumps`` DEFAULT separators; 2,657 with compact separators.
    The serializer is stated because the same saving has two legitimate
    numbers and a figure without one becomes the next agent's evidence.
    Stripping happens here because this is the one point every schema passes,
    builtin and MCP alike.
    """
    schema = apply_intent_schema(
        {
            "type": "object",
            "title": "ReadParams",
            "properties": {
                "path": {"type": "string", "title": "Path", "description": "File to read."},
                "opts": {
                    "type": "object",
                    "title": "Opts",
                    "properties": {"deep": {"type": "boolean", "title": "Deep"}},
                },
                "tags": {"type": "array", "items": {"type": "string", "title": "Tag"}},
            },
            "$defs": {"Item": {"type": "object", "title": "Item"}},
        }
    )

    def titles(value: object) -> list[str]:
        if isinstance(value, dict):
            found = ["title"] if "title" in value else []
            for sub in value.values():
                found += titles(sub)
            return found
        if isinstance(value, list):
            return [t for item in value for t in titles(item)]
        return []

    assert titles(schema) == []
    # Nested/array/$defs shapes survive the walk intact apart from the titles.
    assert schema["properties"]["path"]["description"] == "File to read."
    assert schema["properties"]["opts"]["properties"]["deep"] == {"type": "boolean"}
    assert schema["properties"]["tags"]["items"] == {"type": "string"}
    assert schema["$defs"]["Item"] == {"type": "object"}


def test_a_property_named_title_survives_the_strip() -> None:
    """`title` is BOTH a JSON-Schema keyword and a very common property NAME.

    Regression guard for a shipped-breaking defect: a blind ``k != "title"``
    filter at every dict level deleted the ARGUMENT on the MCP shape used by
    Linear ``create_issue``, Notion ``create_page``, GitHub ``create_issue``
    and Jira. Three compounding failures, all reproduced end to end through
    ``mcp/tool_bridge.py``:

    1. the property vanished from ``properties``, so the model was never told
       the argument existed;
    2. it remained in ``required``, making the schema internally invalid —
       a ``required`` naming an undeclared property;
    3. under the ``additionalProperties: false`` these servers ship, the
       model's own correct payload then became unrepresentable. Validated
       against the broken schema: ``Additional properties are not allowed
       ('title' was unexpected)`` — a strict provider rejects the right call
       and the tool is uncallable.

    NOT an outbound-argument drop: the manager feeds ``prepare_outbound_args``
    the SERVER's schema (``McpManager._schema_parts``), never this stripped
    copy. The harm is that the model is never told the argument exists, which
    breaks every call rather than mangling one.

    The schema below is deliberately the real Linear shape.
    """
    schema = apply_intent_schema(
        {
            "type": "object",
            "title": "CreateIssueInput",  # the keyword — must go
            "additionalProperties": False,
            "properties": {
                # the PROPERTY named title — must survive, minus its own
                # annotation
                "title": {"type": "string", "description": "Issue title.", "title": "Title"},
                "teamId": {"type": "string", "description": "Team.", "title": "Team Id"},
                "meta": {
                    "type": "object",
                    "title": "Meta",
                    "properties": {"title": {"type": "string", "title": "T"}},
                },
            },
            "required": ["title", "teamId"],
        }
    )

    props = schema["properties"]
    assert "title" in props, "the property named `title` was deleted"
    assert props["title"] == {"type": "string", "description": "Issue title."}
    # An invalid schema is the failure that makes strict providers reject the
    # tool outright, so pin the consistency rule and not merely the presence.
    assert set(schema["required"]) <= set(props)
    # Nested property maps get the same treatment.
    assert "title" in props["meta"]["properties"]
    assert props["meta"]["properties"]["title"] == {"type": "string"}
    # The keyword is still stripped everywhere it IS an annotation.
    assert "title" not in schema
    assert "title" not in props["meta"]


def test_property_name_maps_other_than_properties_are_also_protected() -> None:
    """`dependencies`/`dependentRequired` are keyed by property name too.

    Same class as the `properties` defect: the un-filtered branch deleted a
    key literally named `title`. Narrower blast radius — these constrain what
    must ACCOMPANY a field rather than what may be sent, so no argument is
    dropped from a call — but a property named `title` still silently loses
    its conditional requirement.

    Note the two shapes: `dependencies` values are schemas (so a keyword
    `title` inside one must still be stripped), while `dependentRequired`
    values are plain arrays of names.
    """
    schema = apply_intent_schema(
        {
            "type": "object",
            "properties": {"title": {"type": "string"}, "teamId": {"type": "string"}},
            "dependencies": {
                "title": {"required": ["teamId"], "title": "DepAnnotation"},
                "teamId": {"required": ["title"]},
            },
            "dependentRequired": {"title": ["teamId"]},
        }
    )

    assert "title" in schema["dependencies"], "the `title` dependency was deleted"
    assert schema["dependencies"]["title"] == {"required": ["teamId"]}
    assert schema["dependencies"]["teamId"] == {"required": ["title"]}
    assert schema["dependentRequired"] == {"title": ["teamId"]}
    # Stripping less is not the fix: the ANNOTATION inside a schema value goes.
    assert "title" not in schema["dependencies"]["title"]


def test_instance_data_is_never_rewritten_by_the_strip() -> None:
    """``default``/``const``/``enum``/``examples`` hold VALUES, not schemas.

    A ``default`` of ``{"title": "untitled"}`` is data the model is told to
    send. Walking into it edited what the default actually is — changing
    behaviour, not just token count.
    """
    schema = apply_intent_schema(
        {
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "title": "Cfg",
                    "default": {"title": "untitled", "x": 1},
                    "examples": [{"title": "a"}],
                },
                "mode": {"type": "string", "enum": ["title", "body"], "const": "title"},
            },
        }
    )
    cfg = schema["properties"]["cfg"]
    assert cfg["default"] == {"title": "untitled", "x": 1}
    assert cfg["examples"] == [{"title": "a"}]
    assert "title" not in cfg  # the annotation still goes
    mode = schema["properties"]["mode"]
    assert mode["enum"] == ["title", "body"]
    assert mode["const"] == "title"


def test_injection_does_not_mutate_a_nested_input_schema() -> None:
    """Title stripping must build new containers, never edit the caller's.

    A params model's ``model_json_schema()`` is reused across builds, so an
    in-place strip would corrupt a shared object.
    """
    original = {
        "type": "object",
        "title": "P",
        "properties": {"path": {"type": "string", "title": "Path"}},
    }
    apply_intent_schema(original)
    assert original["title"] == "P"
    assert original["properties"]["path"]["title"] == "Path"


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
