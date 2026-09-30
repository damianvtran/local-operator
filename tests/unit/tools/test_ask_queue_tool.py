"""The queued-ask door on the ``ask`` tool (design ``docs/design/ask-nonblocking.md`` §2.1/§5).

The invariant these tests exist for is the PR's flag-off promise: with
``enqueue_ask`` absent — which is what a session binds while the feature is dark —
the tool takes EXACTLY today's path and awaits the host's hook. The queued path
is reached only when a host bound a callable, so the mode is one fact rather than
a flag read in two places.

``tests/unit/tools/test_ask_tool.py`` is deliberately NOT edited by this change:
it pins the blocking shape, and it must stay green and unmodified until the flip
PR rewrites it (design §8 risk 9).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import AgentTool, AskQuestion, ToolContext, ToolResult
from local_operator.tools.registry import create_tools


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "id": "q0",
            "question": "Which database should I target?",
            "options": [{"label": "staging"}, {"label": "prod", "description": "careful"}],
            "multi": False,
        }
    ]


async def _noop_hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
    return {questions[0].id: ["staging"]}


def _tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(context: ToolContext, args: dict[str, Any]) -> ToolResult:
    tool = _tools(context)["ask"]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


def _context(*, enqueue: Any = None, hook: Any = _noop_hook) -> ToolContext:
    return ToolContext(cwd=".", session_id="s", has_ui=True, ask_user=hook, enqueue_ask=enqueue)


@pytest.mark.asyncio
async def test_flag_off_takes_todays_blocking_path():
    """``enqueue_ask`` absent ⇒ the hook is awaited and its answer reported."""
    called: list[int] = []

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        called.append(len(questions))
        return {questions[0].id: ["staging"]}

    result = await _call(_context(hook=hook), {"questions": _questions()})
    assert called == [1]
    assert result.is_error is False
    assert "staging" in result.text


@pytest.mark.asyncio
async def test_flag_off_ignores_a_timeout_but_still_validates_it():
    """With the queue dark the value cannot act, but an out-of-range one is still
    refused: the model learns the bounds now, and the lesson survives the flip."""
    result = await _call(
        _context(),
        {"questions": _questions(), "timeout": 119},
    )
    assert result.is_error is True
    assert "120" in result.text and "86400" in result.text


@pytest.mark.asyncio
async def test_flag_off_accepts_an_in_range_timeout_and_awaits():
    called: list[bool] = []

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        called.append(True)
        return {questions[0].id: ["staging"]}

    result = await _call(_context(hook=hook), {"questions": _questions(), "timeout": "30m"})
    assert called == [True]
    assert "staging" in result.text


@pytest.mark.asyncio
async def test_the_queued_door_returns_a_receipt_and_never_calls_the_hook():
    calls: list[tuple[Any, ...]] = []

    def enqueue(questions: list[Any], timeout: Any, *, tool_call_id: str = "") -> dict[str, Any]:
        calls.append((questions, timeout, tool_call_id))
        return {
            "ok": True,
            "text": "Ask a-1 queued (1 question(s)); showing on terminal.",
            "details": {"ask_id": "a-1", "status": "queued", "timeout_s": 1800},
        }

    hook_called: list[bool] = []

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        hook_called.append(True)
        return None

    result = await _call(
        _context(enqueue=enqueue, hook=hook),
        {"questions": _questions(), "timeout": "30m"},
    )
    assert hook_called == []
    assert calls and calls[0][1] == "30m"
    # The model's own call id rides the queue call (review round 1, MINOR 5):
    # without it the receipt's ask could not be linked back to the call that
    # asked, and every ask in the tree carried an empty id.
    assert calls[0][2] == "call-1"
    assert "queued" in result.text
    assert result.details is not None and result.details["ask_id"] == "a-1"
    assert result.details["status"] == "queued"


@pytest.mark.asyncio
async def test_a_refused_enqueue_is_reported_as_an_error_with_its_own_words():
    """The caps live in the queue, and their sentence is what the model must read
    — the tool does not paraphrase it into a generic failure."""

    def enqueue(questions: list[Any], timeout: Any, *, tool_call_id: str = "") -> dict[str, Any]:
        return {
            "ok": False,
            "error": "this session already has 8 open asks (the cap is 8); "
            "wait for responses; do not re-ask.",
        }

    result = await _call(_context(enqueue=enqueue), {"questions": _questions()})
    assert result.is_error is True
    assert "do not re-ask" in result.text
    assert "8" in result.text


@pytest.mark.asyncio
async def test_the_queued_path_still_requires_a_hook():
    """Availability is unchanged: no hook means no ``ask`` tool at all, so a
    subagent or headless host cannot queue a question either."""
    context = _context(enqueue=lambda q, t: {"ok": True, "text": "x"}, hook=None)
    assert "ask" not in _tools(context)


@pytest.mark.asyncio
async def test_the_timeout_field_is_in_the_advertised_schema():
    """The field ships in this PR (§2.1); the DESCRIPTION rewrite does not (§9)."""
    tool = _tools(_context())["ask"]
    schema = tool.parameters
    assert "timeout" in schema.get("properties", {})


@pytest.mark.asyncio
async def test_a_bool_timeout_is_refused():
    result = await _call(_context(), {"questions": _questions(), "timeout": True})
    assert result.is_error is True
    assert "120" in result.text
