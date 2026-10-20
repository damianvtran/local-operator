"""Monitors compare the FULL result, not the spilled display projection.

The 2026-10-20 failure this file pins: an MCP result over the 8 KiB tool-output
limit arrives as ``spill_truncate``'s head+tail plus a footer naming a char
count and a ``spill://`` handle. The scheduler compared that projection, so a
13.8 KB single-line Slack thread had no visible change beyond the footer — the
gate read the footer as non-material metadata, suppressed seven new replies and
advanced the baseline past them.

Two halves, at the two levels the bug crossed:

* :meth:`Session._run_monitor_check` — a spilled success is compared from the
  stored body, and an unreadable/incomplete spill is ``kind="blind"`` instead of
  a truncated comparison;
* the session's own scheduler — a blind tick keeps the last good baseline and
  tells the operator once, then reports what it missed as one delta.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.spec import MonitorSpec
from local_operator.tools.spill import get_store
from tests.unit.monitors.support import drain_checks
from tests.unit.session.test_session import ScriptedStream, make_session

TOOL = "read"
MISSING = "spill://0123456789abcdef0123456789abcdef"


def thread_result(replies: int) -> str:
    """A ``slack_read_thread``-shaped result: the whole thread on one line."""
    messages = [
        {
            "type": "message",
            "user": f"U{index:04d}",
            "text": f"reply {index}: " + "status update " * 90,
            "ts": f"17560000{index:02d}.000100",
        }
        for index in range(replies)
    ]
    return json.dumps(
        {"messages": messages, "pagination_info": {"has_more": False, "next_cursor": ""}},
        separators=(",", ":"),
    )


def _display(body: str) -> str:
    """What ``spill_truncate`` would hand the model: a head and a footer."""
    return body[:1000] + f"… [Output was {len(body)} chars — spill://abc]"


def _details(body: str) -> dict[str, Any]:
    meta = get_store().write(body, tool_name=TOOL)
    assert meta is not None, "the spill store refused a test body"
    return {
        "spill": {
            "handle": meta.handle,
            "lines": meta.lines,
            "bytes": meta.bytes,
            "complete": meta.complete,
        }
    }


def spill_tool(details: dict[str, Any] | None, *, display: str = "") -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=TOOL,
            content=[TextContent(text=display)],
            details=details,
        )

    return AgentTool(
        name=TOOL,
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


def spec() -> MonitorSpec:
    return MonitorSpec(
        id="m1",
        name="slack thread",
        tool=TOOL,
        arguments={"path": "thread"},
        every_ms=60_000,
        created_at=1_756_000_000_000,
    )


def _session(tmp_path: Path, tool: AgentTool) -> Any:
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    return make_session(tmp_path, stream, tools=[tool])


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


@pytest.mark.asyncio
async def test_a_spilled_result_is_compared_from_its_full_body(
    tmp_path: Path, config_dir: Path
) -> None:
    body = thread_result(10)
    assert "\n" not in body and len(body) > 8 * 1024
    display = _display(body)
    session = _session(tmp_path, spill_tool(_details(body), display=display))
    try:
        outcome = await session._run_monitor_check(spec())
    finally:
        await session.dispose()

    assert outcome.get("error") is None
    assert outcome["text"] == body, "the monitor compared the display projection"
    assert "spill://" not in outcome["text"]
    assert len(outcome["text"]) > len(display)


@pytest.mark.asyncio
async def test_an_evicted_spill_is_blind_not_a_truncated_comparison(
    tmp_path: Path, config_dir: Path
) -> None:
    session = _session(
        tmp_path,
        spill_tool(
            {"spill": {"handle": MISSING, "lines": 1, "bytes": 10, "complete": True}},
            display="a projection nobody can compare",
        ),
    )
    try:
        outcome = await session._run_monitor_check(spec())
    finally:
        await session.dispose()

    assert outcome["kind"] == "blind"
    assert "no longer in the spill store" in str(outcome["error"])


@pytest.mark.asyncio
async def test_an_incomplete_spill_is_blind(tmp_path: Path, config_dir: Path) -> None:
    body = thread_result(3)
    meta = get_store().write(body, tool_name=TOOL, source_complete=False)
    assert meta is not None and meta.complete is False
    session = _session(
        tmp_path,
        spill_tool(
            {
                "spill": {
                    "handle": meta.handle,
                    "lines": meta.lines,
                    "bytes": meta.bytes,
                    "complete": False,
                }
            },
            display=_display(body),
        ),
    )
    try:
        outcome = await session._run_monitor_check(spec())
    finally:
        await session.dispose()

    assert outcome["kind"] == "blind"
    assert "stored incompletely" in str(outcome["error"])


@pytest.mark.asyncio
async def test_an_unspilled_result_is_compared_verbatim(tmp_path: Path, config_dir: Path) -> None:
    """The non-spill path is untouched: no store read, no format change."""
    session = _session(tmp_path, spill_tool(None, display="one\nline"))
    try:
        outcome = await session._run_monitor_check(spec())
    finally:
        await session.dispose()
    assert outcome == {"text": "one\nline", "error": None, "handle": None}


@pytest.mark.asyncio
async def test_a_blind_tick_through_the_session_keeps_the_baseline(
    tmp_path: Path, config_dir: Path
) -> None:
    """The end-to-end shape: spill compare → blind → seeing again.

    The body lives behind a mutable state so the same tool can be watched
    across ticks; the blind tick swaps in a handle the store does not know,
    exactly as an eviction presents itself to a running monitor.
    """
    state: dict[str, Any] = {"body": thread_result(10), "blind": False}

    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        if state["blind"]:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=TOOL,
                content=[TextContent(text="a projection nobody can compare")],
                details={"spill": {"handle": MISSING, "lines": 1, "bytes": 10}},
            )
        body = state["body"]
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=TOOL,
            content=[TextContent(text=_display(body))],
            details=_details(body),
        )

    tool = AgentTool(
        name=TOOL,
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )
    session = _session(tmp_path, tool)
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    try:
        await session.set_monitor_schedules([spec()])
        await session._monitors.pump(now_ms=clock[0] + 5_000)  # baseline
        await drain_checks(session._monitors, session)
        baseline = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert baseline is not None and baseline["checks"] == 1

        state["blind"] = True
        clock[0] += 60_000
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        blind = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert blind is not None
        assert blind["blind"] is True
        assert "spill store" in blind["blind_reason"]
        assert (
            blind["content_hash"] == baseline["content_hash"]
        ), "the baseline advanced while blind"
        assert blind["blind_notified"] is True
        assert [
            getattr(event, "text", "") for event in events if "BLIND" in getattr(event, "text", "")
        ]
        # The DERIVED INDEX is what cold readers render from; the session's
        # change hook must have rewritten it with the blind fields.
        index_row = monitor_store.read_index(config_dir)[session.session_id]["monitors"][0]
        assert index_row["blind"] is True
        assert "spill store" in index_row["blind_reason"]

        # Seeing again: the full body resumes and the missed reply is delivered
        # against the LAST GOOD baseline (the pre-blind thread).
        state["blind"] = False
        state["body"] = thread_result(11)
        clock[0] += 60_000
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        resumed = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert resumed is not None
        assert resumed["blind"] is False and resumed["blind_reason"] == ""
        assert resumed["deliveries"] == 1
        index_row = monitor_store.read_index(config_dir)[session.session_id]["monitors"][0]
        assert index_row["blind"] is False and index_row["blind_reason"] == ""
        texts = [getattr(event, "text", "") for event in events]
        assert any("reply 10:" in text for text in texts), texts
        assert any("seeing its full result again" in text for text in texts), texts
    finally:
        await session.dispose()


def test_the_spill_footer_is_not_part_of_a_comparable_body() -> None:
    """A guard against reintroducing the projection by accident.

    ``spill_truncate``'s footer is the only part of the old comparison that
    changed between two long single-line results; the body the monitor reads
    must never carry it.
    """
    body = thread_result(10)
    display = _display(body)
    assert display != body
    assert f"Output was {len(body)} chars" in display
    assert "Output was" not in body


def long_body(changed: bool = False) -> str:
    """~100 KB over 2,000 lines — far past the 32 KiB snapshot cap."""
    lines = [f"line {index:04d} " + "payload " * 5 for index in range(2_000)]
    if changed:
        lines[1_500] = "line 1500 CHANGED-REPLY " + "payload " * 5
    return "\n".join(lines)


def _long_body_tool(state: dict[str, Any]) -> AgentTool:
    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        body = state["body"]
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=TOOL,
            content=[TextContent(text=_display(body))],
            details=_details(body),
        )

    return AgentTool(
        name=TOOL,
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


async def _run_body_check(session: Any, clock: list[int], *, advance: int = 60_000) -> None:
    clock[0] += advance
    await session._monitors.pump(now_ms=clock[0] + 5_000)
    await drain_checks(session._monitors, session)


@pytest.mark.asyncio
async def test_a_change_past_the_snapshot_cap_is_diffed_from_the_full_body(
    tmp_path: Path, config_dir: Path
) -> None:
    """Review round 1, F1: the 32 KiB snapshot is not the previous result.

    A 100 KB body with one edit at line 1500 was diffed against the snapshot's
    prefix, so the previews were boundary continuation lines and the gate could
    suppress a change it never saw. The spill handle recorded with the commit
    addresses the whole body and is used instead.
    """
    state: dict[str, Any] = {"body": long_body()}
    assert len(state["body"]) > 32_768
    calls: list[str] = []

    async def classify(gate_text: str) -> str | None:
        calls.append(gate_text)
        return None

    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[_long_body_tool(state)],
        monitor_classify=classify,
    )
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    try:
        await session.set_monitor_schedules([spec()])
        await _run_body_check(session, clock)  # baseline
        baseline = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert baseline is not None and baseline["content_hash"]
        assert baseline["last_handle"].startswith("spill://")

        state["body"] = long_body(changed=True)
        await _run_body_check(session, clock)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert counters is not None
        assert counters["deliveries"] == 1
        assert counters["content_hash"] != baseline["content_hash"]
        texts = [getattr(event, "text", "") for event in events]
        assert any("CHANGED-REPLY" in text for text in texts), texts
        # hidden=False, so the gate saw the change itself (not a boundary line).
        assert calls and "CHANGED-REPLY" in calls[0], calls
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_evicted_previous_body_is_a_beyond_window_delta_that_skips_the_gate(
    tmp_path: Path, config_dir: Path
) -> None:
    """Review round 1, F1: no full baseline and a clipped snapshot -> fail open."""
    state: dict[str, Any] = {"body": long_body()}
    calls: list[str] = []

    async def classify(gate_text: str) -> str | None:
        calls.append(gate_text)
        return "non-material-metadata"  # would suppress, if it were asked

    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[_long_body_tool(state)],
        monitor_classify=classify,
    )
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    try:
        await session.set_monitor_schedules([spec()])
        await _run_body_check(session, clock)  # baseline

        # The previous full result is gone; the stored snapshot is a 32 KiB
        # prefix, so there is no honest old side to diff against.
        session._monitors._entries["m1"].counters["last_handle"] = MISSING
        state["body"] = long_body(changed=True)
        await _run_body_check(session, clock)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert counters is not None
        assert counters["deliveries"] == 1
        assert calls == [], "a boundary delta is not gate evidence"
        texts = [getattr(event, "text", "") for event in events]
        assert any("beyond the stored snapshot window" in text for text in texts), texts
        assert not any("CHANGED-REPLY" in text for text in texts)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_body_past_the_diff_bound_delivers_a_checksum_delta(
    tmp_path: Path, config_dir: Path
) -> None:
    """The full baseline is bounded for the quadratic matcher (F1's cost).

    Above ``MAX_DIFF_CHARS`` the change is delivered as the beyond-window
    sentence with ``hidden`` set, so the gate is never asked about a delta that
    does not show the change, and a check cannot stall in ``difflib``.
    """
    state: dict[str, Any] = {"body": ("payload " * 250 + "\n") * 600}
    assert len(state["body"]) > 1_048_576
    calls: list[str] = []

    async def classify(gate_text: str) -> str | None:
        calls.append(gate_text)
        return "non-material-metadata"

    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[_long_body_tool(state)],
        monitor_classify=classify,
    )
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    try:
        await session.set_monitor_schedules([spec()])
        await _run_body_check(session, clock)  # baseline
        state["body"] = state["body"].replace("payload payload", "PAYLOAD payload", 1)
        await _run_body_check(session, clock)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert counters is not None
        assert counters["deliveries"] == 1
        assert calls == [], "a checksum delta is not gate evidence"
        texts = [getattr(event, "text", "") for event in events]
        assert any("beyond the stored snapshot window" in text for text in texts), texts
    finally:
        await session.dispose()
