"""The ask gate in ``lop exec``'s human renderer: settle-only, machine keeps all.

Design docs/design/ask-gate.md §3 row 11. On the text surface while the queued
engine is live, an ask call's ``●`` line is withheld at start and emitted at
settle for a RAISE ("today's line", just later) or not at all for a divert (the
marker). The JSON stream is a MACHINE surface and is deliberately untouched: it
keeps every frame, and the marker rides ``result.details`` so a supervisor can
filter it (the ``FAULT_KEY`` precedent).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from local_operator.harness.types import (
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.headless_print import PrintRenderer

ASK_ARGS = {"questions": [{"id": "q0", "question": "Which database?", "options": []}]}
MARKER = {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}}


def _renderer(*, json_mode: bool = False, queued: bool = True) -> PrintRenderer:
    renderer = PrintRenderer(json_mode=json_mode)
    # The mode read is direct: ``attach`` holds the session (design §3 row 11),
    # and the probe asks it ``ask_queue()`` — bound, in production, exactly
    # where ``ask`` queues at all. Cast: this stub answers only what the probe
    # reads; the protocol's wider surface is not this test's subject.
    renderer._session = cast(
        Any, SimpleNamespace(ask_queue=(lambda: object()) if queued else (lambda: None))
    )
    return renderer


def _start() -> ToolExecutionStartEvent:
    return ToolExecutionStartEvent(
        tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS, intent="asking about the db"
    )


def _end(*, marker: bool) -> ToolExecutionEndEvent:
    return ToolExecutionEndEvent(
        tool_call_id="call-ask",
        tool_name="ask",
        result=ToolResult(
            tool_call_id="call-ask",
            tool_name="ask",
            content=[TextContent(text="Ask a-1 queued.")],
            details=dict(MARKER) if marker else {},
        ),
    )


def test_a_diverted_ask_prints_no_human_lines(capsys) -> None:
    renderer = _renderer()
    renderer.handle(_start())
    assert capsys.readouterr().err == "", "the start line must be withheld"

    renderer.handle(_end(marker=True))
    captured = capsys.readouterr()
    assert (
        captured.out == "" and captured.err == ""
    ), "a divert leaves no trace on the human surface"


def test_a_raised_ask_prints_todays_start_line_at_settle(capsys) -> None:
    renderer = _renderer()
    renderer.handle(_start())
    assert capsys.readouterr().err == ""

    renderer.handle(_end(marker=False))
    err = capsys.readouterr().err
    assert err.count("● ask") == 1
    assert "asking about the db" in err, "the start line's own bytes, just later"


def test_the_blocking_arm_prints_today(capsys) -> None:
    renderer = _renderer(queued=False)
    renderer.handle(_start())
    err = capsys.readouterr().err
    assert "● ask" in err and "asking about the db" in err

    renderer.handle(_end(marker=False))
    assert capsys.readouterr().err == "", "success prints nothing further, as today"


def test_the_json_stream_keeps_every_frame_with_the_marker(capsys) -> None:
    renderer = _renderer(json_mode=True)
    renderer.handle(_start())
    renderer.handle(_end(marker=True))

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    types = [line.get("type") for line in lines]
    assert "tool_execution_start" in types and "tool_execution_end" in types
    end = next(line for line in lines if line.get("type") == "tool_execution_end")
    assert end["result"]["details"]["ask_gate"]["hidden"] is True
