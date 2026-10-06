"""``execute_ask`` + the gate door — the orchestration the design pins (§2.2).

The gate is composed AROUND the unchanged enqueue: one awaited clearance
check whose only diverting outcome is a Mapping, and every other outcome —
``None`` from any fail-open path, a gate that raises, no gate at all — falls
through to the existing enqueue call byte-for-byte. The ordering is pinned
here: an invalid argument never reaches the gate, a divert never reaches the
enqueue, and a broken gate never costs the ask.
"""

from __future__ import annotations

import asyncio
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


DIVERT = {
    "verdict": "clear",
    "text": "[Ask clearance] No question was put to the user. Proceed.",
    "details": {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}},
}


def _context(*, enqueue: Any = None, gate: Any = None, hook: Any = _noop_hook) -> ToolContext:
    return ToolContext(
        cwd=".",
        session_id="s",
        has_ui=True,
        ask_user=hook,
        enqueue_ask=enqueue,
        gate_ask=gate,
    )


class RecordingGate:
    def __init__(self, verdict: Any) -> None:
        self.verdict = verdict
        self.calls: list[tuple[Any, Any, str]] = []

    async def __call__(self, questions: Any, timeout: Any, *, tool_call_id: str = "") -> Any:
        self.calls.append((questions, timeout, tool_call_id))
        if isinstance(self.verdict, BaseException):
            raise self.verdict
        return self.verdict


def _recording_enqueue():
    calls: list[tuple[Any, ...]] = []

    def enqueue(questions: list[Any], timeout: Any, *, tool_call_id: str = "") -> dict[str, Any]:
        calls.append((questions, timeout, tool_call_id))
        return {
            "ok": True,
            "text": "Ask a-1 queued (1 question(s)); showing on terminal.",
            "details": {"ask_id": "a-1", "status": "queued"},
        }

    return enqueue, calls


# --- the divert path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_divert_returns_the_note_with_the_hidden_marker_and_never_enqueues():
    enqueue, enqueue_calls = _recording_enqueue()
    gate = RecordingGate(DIVERT)

    result = await _call(
        _context(enqueue=enqueue, gate=gate),
        {"questions": _questions(), "timeout": "30m"},
    )

    assert result.is_error is False
    assert result.text.startswith("[Ask clearance]")
    assert result.details == DIVERT["details"]  # the marker rides the result
    assert enqueue_calls == []
    # The gate saw the VALIDATED questions and the model's own timeout + call id.
    assert gate.calls and gate.calls[0][2] == "call-1"
    assert gate.calls[0][1] == "30m"


@pytest.mark.asyncio
async def test_a_none_verdict_still_enqueues():
    enqueue, enqueue_calls = _recording_enqueue()

    result = await _call(
        _context(enqueue=enqueue, gate=RecordingGate(None)),
        {"questions": _questions()},
    )

    assert result.is_error is False
    assert "queued" in result.text
    assert len(enqueue_calls) == 1


@pytest.mark.asyncio
async def test_a_gate_that_raises_still_enqueues():
    """THE LAST LINE: even a contract breach cannot cost the ask."""
    enqueue, enqueue_calls = _recording_enqueue()

    result = await _call(
        _context(enqueue=enqueue, gate=RecordingGate(RuntimeError("broken gate"))),
        {"questions": _questions()},
    )

    assert result.is_error is False
    assert len(enqueue_calls) == 1


@pytest.mark.asyncio
async def test_cancelled_error_from_the_gate_propagates():
    """An aborted turn must abort — the tool-level guard catches Exception only."""
    enqueue, enqueue_calls = _recording_enqueue()
    gate = RecordingGate(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _call(_context(enqueue=enqueue, gate=gate), {"questions": _questions()})

    assert enqueue_calls == []


@pytest.mark.asyncio
async def test_invalid_params_never_call_the_gate_or_enqueue():
    gate = RecordingGate(DIVERT)
    enqueue, enqueue_calls = _recording_enqueue()

    result = await _call(
        _context(enqueue=enqueue, gate=gate),
        {"questions": []},  # a questions list is required to be non-empty
    )

    assert result.is_error is True
    assert gate.calls == []
    assert enqueue_calls == []


@pytest.mark.asyncio
async def test_bounds_error_never_calls_the_gate():
    """The timeout is validated ABOVE the branch, so both arms learn the bounds."""
    gate = RecordingGate(DIVERT)

    result = await _call(
        _context(enqueue=lambda *a, **k: {"ok": True, "text": "x", "details": {}}, gate=gate),
        {"questions": _questions(), "timeout": 119},
    )

    assert result.is_error is True
    assert gate.calls == []


# --- the untouched arms ------------------------------------------------------


@pytest.mark.asyncio
async def test_blocking_arm_is_byte_for_byte_today():
    """No ``enqueue_ask`` — and therefore no gate — means the hook is awaited.

    The gate door is bound under the SAME two conditions as the enqueue door,
    so a context without one is a context without the other; this pins that a
    blocking host never sees a gate call even if a stray one were attached.
    """
    hook_called: list[bool] = []

    async def hook(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
        hook_called.append(True)
        return {questions[0].id: ["staging"]}

    gate = RecordingGate(DIVERT)
    result = await _call(_context(enqueue=None, gate=None, hook=hook), {"questions": _questions()})

    assert hook_called == [True]
    assert "staging" in result.text
    assert gate.calls == []


@pytest.mark.asyncio
async def test_no_gate_door_takes_todays_enqueue_path_unchanged():
    """``gate_ask`` absent while the queue lives: the enqueue call is the whole path."""
    enqueue, enqueue_calls = _recording_enqueue()

    result = await _call(_context(enqueue=enqueue, gate=None), {"questions": _questions()})

    assert len(enqueue_calls) == 1
    assert result.details == {"ask_id": "a-1", "status": "queued"}


# --- schema stays frozen (the footprint statement, §7) ----------------------


def test_the_ask_schema_and_default_tool_names_do_not_move() -> None:
    """Zero new schema: no new parameter, no new tool, so the cached prefix holds.

    The ``i`` intent property is the shared tool-surface field every builtin
    carries; the point is that the gate added NONE.
    """
    context = _context(enqueue=_recording_enqueue()[0], gate=RecordingGate(None))
    schema = _tools(context)["ask"].parameters
    properties = schema.get("properties", {})
    assert set(properties) == {"i", "questions", "timeout"}

    from local_operator.tools.registry import DEFAULT_TOOL_NAMES

    assert "ask" in DEFAULT_TOOL_NAMES
