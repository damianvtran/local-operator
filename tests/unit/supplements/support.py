"""Builders for turn items shared by the C1a pre-filter, decision and golden-set tests."""

from __future__ import annotations

from local_operator.supplements.trigger import ToolCallItem, TurnItem


def call(name: str, call_id: str = "c1", **args: str) -> TurnItem:
    """An assistant message carrying one tool call (``path``/``command``/``code``/``i``)."""
    return TurnItem(
        id=f"a-{call_id}",
        role="assistant",
        tool_calls=(ToolCallItem(id=call_id, name=name, args=dict(args)),),
    )


def result(text: str, call_id: str = "c1", *, tool: str = "bash", error: bool = False) -> TurnItem:
    return TurnItem(
        id=f"t-{call_id}",
        role="tool",
        text=text,
        tool_call_id=call_id,
        tool_name=tool,
        is_error=error,
    )


def answer(text: str, answer_id: str = "final") -> TurnItem:
    return TurnItem(id=answer_id, role="assistant", text=text)
