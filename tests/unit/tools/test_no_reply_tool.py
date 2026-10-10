"""The ``no_reply`` tool: an argumentless call that ends a turn quietly.

WHY THE SHAPE IS WHAT IT IS (docs/design/quiet-turns.md §4). The affirmative
quiet end is a TOOL CALL, not a text sentinel and not an empty reply: a sentinel
streams to every surface before it can be recognised, and an empty reply is
indistinguishable from the provider glitches the design measured (68
``error``+empty stops in one scan). These tests pin the tool's own contract:

* the params model declares NO fields, and the only argument a call can carry
  is the harness-injected ``i`` intent — ``create_tools`` injects it
  (``apply_intent_schema``) and the loop lifts it off before validation
  (review R7), so a model-supplied call arrives as ``{}`` and anything else is
  refused rather than silently ignored;
* the createIf gate is the ``quiet_end`` door ALONE: no door, no tool
  (footprint rung 3);
* a refusal (the door's sentence) surfaces as an ``is_error`` result, verbatim,
  and leaves NO quiet marker — the loop must not end the turn on it;
* an allowed end stamps ``QUIET_TURN_KEY`` on the result's ``details``, the
  marker the loop's batch check and the session's quiet predicate read.

The door's own binding conditions — a subagent child, a one-shot host, the
``LOP_NO_REPLY`` kill switch — are pinned here too, because they are the
availability facts the registry-level delta cannot see.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import QUIET_TURN_KEY, StreamEndEvent, ToolContext
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools
from tests.unit.session.test_session import ScriptedStream, make_session


async def _allow() -> str | None:
    """A door that allows every call — the tool's happy path."""
    return None


def _tool(context: ToolContext | None = None) -> Any:
    """The REAL builder's tool, built exactly as a session builds it.

    Through ``create_tools`` rather than ``build_no_reply_tool`` directly, so
    the schema carries the harness-injected ``i`` — the injection is what the
    loop's pop keys on, and a hand-rolled tool would let tests pass on a shape
    no session ever holds.
    """
    if context is None:
        context = ToolContext(cwd=".", quiet_end=_allow)
    tools = create_tools(context, enabled=["no_reply"])
    assert tools, "the builder must produce the tool over an open door"
    return tools[0]


def test_the_params_model_declares_no_fields() -> None:
    """No ``reason``, no fields at all (docs §4): the call's whole content is
    its existence. A reason could only live in the persisted arguments, which
    are re-billed on every later request, while the trigger row beside the call
    already says why."""
    schema = builtin.NoReplyParams.model_json_schema()
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


def test_the_built_schema_declares_only_the_injected_intent() -> None:
    """``create_tools`` injects ``i`` and nothing else, and the loop pops it
    pre-validation (review R7) — otherwise the harness-advertised property
    would hit the argumentless params model as an unknown key."""
    tool = _tool()
    assert list(tool.parameters["properties"]) == ["i"]


def test_the_builder_is_createif_gated_on_the_quiet_end_door() -> None:
    """Presence-only, one fact (rung 3): the door IS "this session may end
    quietly", so its absence removes the tool rather than making it inert."""
    assert builtin.build_no_reply_tool(ToolContext(cwd=".")) is None
    assert create_tools(ToolContext(cwd="."), enabled=["no_reply"]) == []
    assert builtin.build_no_reply_tool(ToolContext(cwd=".", quiet_end=_allow)) is not None


@pytest.mark.asyncio
async def test_a_key_other_than_the_injected_intent_is_refused() -> None:
    """``extra="forbid"`` is what keeps "only the injected ``i`` is accepted"
    true at the TOOL boundary: the loop's validator does not reject unknown
    top-level keys, so the refusal has to come from the params model — as an
    invalid-arguments result, never a silent ignore."""
    tool = _tool()
    context = ToolContext(cwd=".", quiet_end=_allow)
    refused = await tool.execute("c1", {"reason": "because"}, None, None, context)
    assert refused.is_error
    assert "reason" in refused.content[0].text, "the refusal names the bad key"
    assert QUIET_TURN_KEY not in (refused.details or {})


@pytest.mark.asyncio
async def test_an_allowed_end_stamps_the_marker_with_the_quiet_word() -> None:
    tool = _tool()
    context = ToolContext(cwd=".", quiet_end=_allow)
    result = await tool.execute("c1", {}, None, None, context)
    assert not result.is_error
    assert result.content[0].text == "Quiet."
    assert result.details == {QUIET_TURN_KEY: True}
    assert result.useless is True, "the prune pass blanks the content later"


@pytest.mark.asyncio
async def test_a_refusal_returns_its_sentence_and_leaves_no_marker() -> None:
    """The refusal is the door's own sentence, verbatim: the model reads what
    to do instead, the loop reads no marker, and the turn continues."""

    async def refuse() -> str | None:
        return "A person asked this turn; answer them in one line."

    context = ToolContext(cwd=".", quiet_end=refuse)
    tool = _tool(context)
    result = await tool.execute("c1", {}, None, None, context)
    assert result.is_error
    assert result.content[0].text == "A person asked this turn; answer them in one line."
    assert QUIET_TURN_KEY not in (result.details or {})


@pytest.mark.asyncio
async def test_a_doorless_execute_reports_a_wiring_fault_not_a_quiet_end() -> None:
    """Unreachable through the advertised tool (the builder refuses the door),
    so a call that gets here is a host wiring fault: reported as one, never as
    a quiet end that would tell the model its turn settled."""
    tool = _tool()
    result = await tool.execute("c1", {}, None, None, ToolContext(cwd="."))
    assert result.is_error
    assert "no quiet-end door" in result.content[0].text
    assert QUIET_TURN_KEY not in (result.details or {})


# ---------------------------------------------------------------------------
# The door's binding: the three cases the registry delta cannot see.
# ---------------------------------------------------------------------------


def _session(tmp_path, **kwargs):
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    return make_session(tmp_path, stream, **kwargs)


@pytest.mark.asyncio
async def test_the_door_binds_on_a_plain_session(tmp_path) -> None:
    session = _session(tmp_path)
    try:
        assert callable(session._quiet_end_callable())
        assert callable(session._build_tool_context().quiet_end)
        tools = create_tools(session._build_tool_context(), enabled=["no_reply"])
        assert [tool.name for tool in tools] == ["no_reply"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_door_is_none_for_a_subagent_child(tmp_path) -> None:
    """A child's final text is what its parent's ``wait`` reads as the report;
    silence there is a lost result, so the tool must not exist (docs §4). The
    child never even receives it: ``_job_id`` is set before the constructor's
    capability merge runs, so the builder refuses and — the mechanism
    ``harness/subagent``'s derived prune relies on — there is nothing to undo."""
    session = _session(tmp_path, job_id="job-s0a")
    try:
        assert session._quiet_end_callable() is None
        assert session._build_tool_context().quiet_end is None
        assert create_tools(session._build_tool_context(), enabled=["no_reply"]) == []
        assert "no_reply" not in {
            tool.name for tool in session._tools
        }, "the constructor's merge must not have mounted it for a child"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_door_is_none_for_a_one_shot_host(tmp_path) -> None:
    """A one-shot host's product IS the final text (``headless_print`` prints
    ``if final_text``), and nothing there distinguishes "silent on purpose"
    from "produced nothing" — so the tool is absent, not inert. The declaration
    lands AFTER construction (the merge already mounted it), so the setter
    itself must drop it from the live inventory."""
    session = _session(tmp_path)
    assert "no_reply" in {tool.name for tool in session._tools}, "mounted before"
    session.declare_one_shot_exit()
    try:
        assert session._quiet_end_callable() is None
        assert create_tools(session._build_tool_context(), enabled=["no_reply"]) == []
        assert "no_reply" not in {
            tool.name for tool in session._tools
        }, "the declaration must drop the mounted tool, not just null the door"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_door_is_none_under_the_kill_switch(tmp_path, monkeypatch) -> None:
    """``LOP_NO_REPLY=0`` (env only, no config key — docs §10): the tool goes
    away for the whole process, and a typo cannot do it by accident."""
    monkeypatch.setattr(builtin, "NO_REPLY_ENABLED", False)
    session = _session(tmp_path)
    try:
        assert builtin.no_reply_enabled() is False
        assert session._quiet_end_callable() is None
        assert create_tools(session._build_tool_context(), enabled=["no_reply"]) == []
    finally:
        await session.dispose()
