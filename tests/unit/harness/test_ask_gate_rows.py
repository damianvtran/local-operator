"""The ask gate's shared predicates (``harness.rows``): one decision per shape.

Design docs/design/ask-gate.md §3's predicate table. Every human surface reads
the divert marker through these functions and none re-derives it, so each
predicate is pinned against ALL the shapes it must accept — a live details
mapping, a stored ``{type, payload}`` row, a rendered ``Message`` — and against
the near-misses it must refuse (a normal ask row, the patience pair, the wake
marker), so an over-broad predicate fails here rather than silently blanking a
transcript.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from local_operator.harness.rows import (
    ask_gate_diverted_call_ids,
    is_ask_gate_divert_details,
    is_ask_gate_divert_message,
    is_ask_gate_divert_row,
    is_settle_only_ask,
    queued_ask_engine_live,
    without_ask_gate_divert,
)
from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult

MARKER = {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}}


def _result_message(*, marker: bool, text: str = "receipt") -> Message:
    return Message.tool_result(
        ToolResult(
            tool_call_id="call-ask",
            tool_name="ask",
            content=[TextContent(text=text)],
            details=dict(MARKER) if marker else {},
        )
    )


# --- is_ask_gate_divert_details ---------------------------------------------


def test_details_marker_shapes() -> None:
    assert is_ask_gate_divert_details(MARKER) is True
    assert is_ask_gate_divert_details(dict(MARKER)) is True
    # Negatives: wrong shape, absent, a non-mapping, and hidden not True.
    assert is_ask_gate_divert_details({}) is False
    assert is_ask_gate_divert_details(None) is False
    assert is_ask_gate_divert_details("ask_gate") is False
    assert is_ask_gate_divert_details({"ask_gate": {"hidden": False}}) is False
    assert is_ask_gate_divert_details({"ask_gate": "nope"}) is False
    assert is_ask_gate_divert_details({"other": {"hidden": True}}) is False


# --- is_ask_gate_divert_row --------------------------------------------------


def test_stored_row_predicate() -> None:
    row = {
        "type": "message",
        "payload": {"role": "tool", "tool_name": "ask", "provider_payload": {"details": MARKER}},
    }
    assert is_ask_gate_divert_row(row) is True

    # Negatives: a normal ask result row, a patience row, the wake marker, and
    # the malformed shapes a caller may hand a duck-typed predicate.
    assert (
        is_ask_gate_divert_row({"type": "message", "payload": {"role": "tool", "tool_name": "ask"}})
        is False
    )
    assert (
        is_ask_gate_divert_row(
            {"type": "message", "payload": {"role": "tool", "tool_name": "patience"}}
        )
        is False
    )
    assert (
        is_ask_gate_divert_row(
            {
                "type": "message",
                "payload": {"custom_type": "wake_prompt", "details": {"hidden": True}},
            }
        )
        is False
    )
    assert is_ask_gate_divert_row("row") is False
    assert is_ask_gate_divert_row({"payload": None}) is False


# --- is_ask_gate_divert_message ---------------------------------------------


def test_rendered_message_predicate() -> None:
    assert is_ask_gate_divert_message(_result_message(marker=True)) is True
    assert is_ask_gate_divert_message(_result_message(marker=False)) is False
    assert is_ask_gate_divert_message(Message.user("hello")) is False
    assert is_ask_gate_divert_message(None) is False


# --- ask_gate_diverted_call_ids ----------------------------------------------


def test_call_id_set_reads_the_marked_results() -> None:
    messages: list[Any] = [
        Message.user("go"),
        _result_message(marker=True),
        Message.tool_result(
            ToolResult(tool_call_id="call-read", tool_name="read", content=[TextContent(text="f")])
        ),
    ]
    assert ask_gate_diverted_call_ids(messages) == {"call-ask"}
    assert ask_gate_diverted_call_ids([]) == set()


# --- without_ask_gate_divert -------------------------------------------------


def test_transform_drops_the_result_and_strips_the_call() -> None:
    ids = {"call-ask"}
    assert without_ask_gate_divert(_result_message(marker=True), ids) is None

    # An assistant row with no prose and only the diverted call goes entirely.
    bare_call = Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(id="call-ask", name="ask", arguments={})],
    )
    assert without_ask_gate_divert(bare_call, ids) is None

    # A mixed row keeps its prose and its other calls.
    mixed = Message(
        role="assistant",
        content=[TextContent(text="let me check")],
        tool_calls=[
            ToolCall(id="call-ask", name="ask", arguments={}),
            ToolCall(id="call-read", name="read", arguments={}),
        ],
    )
    stripped = without_ask_gate_divert(mixed, ids)
    assert stripped is not None
    assert stripped.text == "let me check"
    assert [call.id for call in stripped.tool_calls] == ["call-read"]

    # Passthrough when nothing matches.
    untouched = Message.user("hi")
    assert without_ask_gate_divert(untouched, ids) is untouched
    assert without_ask_gate_divert(untouched, set()) is untouched


# --- is_settle_only_ask ------------------------------------------------------


def test_settle_only_predicate_table() -> None:
    assert is_settle_only_ask("ask", queued_engine=True) is True
    assert is_settle_only_ask("ask", queued_engine=False) is False
    assert is_settle_only_ask("bash", queued_engine=True) is False
    assert is_settle_only_ask(None, queued_engine=True) is False
    assert is_settle_only_ask("ASK", queued_engine=True) is False


# --- queued_ask_engine_live --------------------------------------------------


def test_mode_probe_owner_and_viewer() -> None:
    # The owner: ``ask_queue()`` is the one authority (None on the blocking arm).
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=lambda: object())) is True
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=lambda: None)) is False
    assert queued_ask_engine_live(SimpleNamespace(ask_queue=object())) is False  # not callable

    def _boom() -> object:
        raise RuntimeError("no queue")

    assert queued_ask_engine_live(SimpleNamespace(ask_queue=_boom)) is False

    # The viewer: the ``asks`` wire field's PRESENCE on the frontend state.
    assert (
        queued_ask_engine_live(SimpleNamespace(frontend_state=SimpleNamespace(asks=[{"id": "a1"}])))
        is True
    )
    assert (
        queued_ask_engine_live(SimpleNamespace(frontend_state=SimpleNamespace(asks=None))) is False
    )
    # A facade that cannot say, and a bare object: False, never a raise.
    assert queued_ask_engine_live(SimpleNamespace()) is False
    assert queued_ask_engine_live(object()) is False
