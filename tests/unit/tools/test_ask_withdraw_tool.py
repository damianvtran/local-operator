"""The agent-side settle's tool (design ``docs/design/ask-nonblocking.md`` §12).

The invariants these tests exist for:

* the builder is a ``createIf`` factory gated on the ``withdraw_ask`` callable
  ALONE — absent, not inert, on the blocking arm, on headless hosts and inside
  subagents (footprint ladder rung 3), so a session whose log cannot hold a
  ``withdrawn`` row pays no schema for the op;
* the tool relays the queue's sentences byte-for-byte, refusals included — a
  paraphrase here would be a second wording of one rule (the op-vs-state split
  §10/§12 keep);
* the advertised params carry exactly the two reasons and the cells map, and no
  ``message_id`` slot the model cannot fill truthfully (the queue still accepts
  one for callers that know it; the schema must not invite a fabricated id).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


def _tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    tool = _tools(context)["ask_withdraw"]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


def _context(*, withdraw: Any = None) -> ToolContext:
    return ToolContext(cwd=".", session_id="s", has_ui=True, withdraw_ask=withdraw)


def test_the_tool_is_absent_without_the_withdraw_door():
    """Rung 3: no callable, no tool. This is the exact state of the blocking
    arm, a headless host and every subagent — none of them may pay schema for
    an op their log cannot hold."""
    assert "ask_withdraw" not in _tools(_context())


def test_the_tool_is_present_with_the_door_and_appended_last():
    """With the door bound, the tool exists and sits at the END of the array —
    appending never shifts a provider-visible prefix, which is what the prompt
    cache keys on."""
    tools = create_tools(_context(withdraw=lambda *_a, **_k: {"ok": True}))
    assert tools[-1].name == "ask_withdraw"


def test_the_advertised_schema_is_the_two_reasons_and_the_cells_map():
    tool = _tools(_context(withdraw=lambda *_a, **_k: {"ok": True}))["ask_withdraw"]
    props = tool.parameters["properties"]
    assert {"ask_id", "reason", "answers"} <= set(props)
    # ``Literal`` renders as an enum: the two reasons travel in the schema, so
    # the model reads the choices rather than learning them from a refusal.
    assert props["reason"]["enum"] == ["moot", "answered_in_chat"]
    # ``message_id`` is deliberately NOT advertised (the model cannot see chat
    # message ids); the queue and session still accept it for callers that can.
    assert "message_id" not in props


@pytest.mark.asyncio
async def test_the_receipt_and_details_are_relayed_byte_for_byte():
    calls: list[tuple[str, dict[str, Any]]] = []

    def withdraw(ask_id: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((ask_id, kwargs))
        return {
            "ok": True,
            "text": "Ask a-1 withdrawn — no longer waiting on an answer.",
            "details": {"ask_id": "a-1", "reason": "moot"},
        }

    result = await _call(_context(withdraw=withdraw), {"ask_id": "a-1", "reason": "moot"})
    assert calls == [("a-1", {"reason": "moot", "answers": None})]
    assert result.is_error is False
    assert result.text == "Ask a-1 withdrawn — no longer waiting on an answer."
    assert result.details == {"ask_id": "a-1", "reason": "moot"}


@pytest.mark.asyncio
async def test_a_refusal_is_reported_with_the_queues_own_words():
    """The op-level sentences (§12) reach the model unchanged: "the user
    already declined this ask" must not become a generic failure, and it must
    not become the state table's user-voiced twin either."""
    sentence = "the user already declined this ask — there is nothing to withdraw."

    def withdraw(ask_id: str, **kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "error": sentence}

    result = await _call(_context(withdraw=withdraw), {"ask_id": "a-1", "reason": "moot"})
    assert result.is_error is True
    assert result.text == sentence


@pytest.mark.asyncio
async def test_the_cells_map_rides_through_untouched():
    seen: dict[str, Any] = {}

    def withdraw(ask_id: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"ok": True, "text": "ok", "details": {}}

    cells = {"q0": ["the audit-log one"], "q1": []}
    result = await _call(
        _context(withdraw=withdraw),
        {"ask_id": "a-2", "reason": "answered_in_chat", "answers": cells},
    )
    assert result.is_error is False
    assert seen == {"reason": "answered_in_chat", "answers": cells}


@pytest.mark.asyncio
async def test_a_bad_reason_is_a_validation_error_naming_the_choices():
    result = await _call(
        _context(withdraw=lambda *_a, **_k: {"ok": True}),
        {"ask_id": "a-1", "reason": "settle"},
    )
    assert result.is_error is True
    assert "moot" in result.text and "answered_in_chat" in result.text


@pytest.mark.asyncio
async def test_a_known_ask_id_is_required():
    result = await _call(_context(withdraw=lambda *_a, **_k: {"ok": True}), {"reason": "moot"})
    assert result.is_error is True
    assert "ask_id" in result.text


@pytest.mark.asyncio
async def test_a_context_without_the_door_is_a_wiring_fault_never_a_settled_ask():
    """Unreachable through the advertised tool (the builder refuses to create
    it without the door) — but if a host calls the executor anyway, the result
    must say NOTHING WAS SETTLED, never "ok": a false success here would tell
    the model its withdrawal landed when nothing was written."""
    result = await builtin.execute_ask_withdraw(
        "call-1", {"ask_id": "a-1", "reason": "moot"}, None, None, _context()
    )
    assert result.is_error is True
    assert "no ask was settled" in result.text
