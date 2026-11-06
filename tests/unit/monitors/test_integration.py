"""End-to-end engine rigs (§18).

Rig 1 is the durability path the contract calls out: arm → the process DIES
(``os._exit(9)``, no dispose, no flush beyond what already landed) → reopen →
one consolidated material delta naming the ticks the dead window skipped.

Rig 2 is the normalization quiet tick: a source whose output changes only in
lines the monitor was told to ignore (and in timestamps) must stay silent.
"""

from __future__ import annotations

import asyncio  # noqa: F401 — kept for a content-identical merge; main's tests use it
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.monitors.support import drain_checks
from tests.unit.monitors.test_session_monitor_availability import FakeManager

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def make_read_tool(path: Path) -> AgentTool:
    """A real read-tier tool the session owns; its output is the watched file."""

    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        try:
            text = Path(str(args.get("path") or path)).read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(tool_call_id=tool_call_id, content=[TextContent(text=str(exc))])
        return ToolResult(tool_call_id=tool_call_id, content=[TextContent(text=text)])

    return AgentTool(
        name="read",
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )


def _stream(request: Any, signal: Any):  # noqa: ANN202
    async def gen():
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _session(tmp_path: Path, tool: AgentTool, session_id: str = "sess") -> Session:
    return Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[tool],
        transcript=Transcript(tmp_path / session_id),
        system_blocks_provider=lambda: [],
        cwd=str(tmp_path),
    )


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


CHILD_SCRIPT = """
import asyncio, os, sys

from local_operator.harness.types import (
    AgentTool,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

WATCH = os.environ["WATCH_FILE"]
SESS = os.environ["SESS_DIR"]


async def execute(tool_call_id, args, signal=None, on_update=None, context=None):
    text = open(WATCH, encoding="utf-8").read()
    return ToolResult(tool_call_id=tool_call_id, content=[TextContent(text=text)])


async def stream(request, signal):
    yield StreamEndEvent(stop_reason="stop")


async def main():
    tool = AgentTool(
        name="read",
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )
    session = Session(
        model=ModelSpec(provider="t", model_id="m", context_window=1000),
        stream_fn=stream,
        tools=[tool],
        transcript=Transcript(SESS),
        system_blocks_provider=lambda: [],
        cwd=SESS,
    )
    await session.set_monitor_schedules(
        [
            MonitorSpec(
                id="m1",
                name="watch",
                tool="read",
                arguments={"path": WATCH},
                every_ms=60_000,
                created_at=1,
            )
        ]
    )
    # The first check is due 1-3 s after arm; letting the loop run past that
    # window is what makes the baseline land before the crash.
    await asyncio.sleep(3.6)
    os._exit(9)


asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_arm_kill_reopen_delivers_one_consolidated_delta(
    tmp_path: Path, config_dir: Path
) -> None:
    watch = tmp_path / "watched.txt"
    watch.write_text("A\nB\n", encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "WATCH_FILE": str(watch),
            "SESS_DIR": str(tmp_path / "sess"),
            "LOCAL_OPERATOR_CONFIG_DIR": str(config_dir),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT], env=env, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 9, completed.stderr[-2000:]

    # The crash left the durable state behind: transcript rows, the index
    # entry, and counters carrying the baseline's last check.
    entry = monitor_store.read_entry(config_dir, "sess")
    assert entry is not None and entry["monitors"][0]["id"] == "m1"
    counters_path = config_dir / "monitors" / "state" / "sess" / "m1.json"
    counters = json.loads(counters_path.read_text(encoding="utf-8"))
    last_check = int(counters["last_check_at"])
    assert last_check > 0 and counters["content_hash"].startswith("sha256:")

    # Reopen in-process; drive the clock explicitly so the down-time gap is a
    # number the test owns.
    reopened = _session(tmp_path, make_read_tool(watch))
    clock = [last_check + 4 * 60_000]
    reopened._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    reopened._emit = record  # type: ignore[method-assign]
    watch.write_text("A\nB\nC\nD\nE\n", encoding="utf-8")  # several changes, one gap
    try:
        await reopened._monitors.pump(now_ms=clock[0] + 30_000)
        await drain_checks(reopened._monitors, reopened)
        deltas = [event for event in events if getattr(event, "type", "") == "monitor_delta"]
        assert len(deltas) == 1, "one consolidated delta, never one per missed tick"
        text = deltas[0].text
        match = re.search(r"(\d+) skipped while the session was down", text)
        assert match is not None, text
        # A four-interval gap: three checks came due unrun, and this resume
        # check supersedes the fourth (§9.3).
        assert int(match.group(1)) == 3
        assert deltas[0].monitor_id == "m1"
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_a_delivery_that_hops_to_a_thread_is_still_awaited(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin for the CI flake the loop-turn waits caused.

    ``Session._deliver_monitor`` awaits a thread hop (``_mark_code_requests_dirty``)
    BEFORE it emits the delta, so how many event-loop turns the delivery needs is
    the host's scheduling, not the code's. This forces the worst case on any
    machine: the hop takes real time, far more turns than any fixed count would
    cover. ``drain_checks`` must still see the delta, because it waits on the
    check task that awaits the delivery, not on a number of turns.
    """

    async def slow_mark(self: Session) -> None:
        await asyncio.to_thread(time.sleep, 0.3)

    monkeypatch.setattr(Session, "_mark_code_requests_dirty", slow_mark)
    watch = tmp_path / "watched.txt"
    watch.write_text("A\n", encoding="utf-8")
    session = _session(tmp_path, make_read_tool(watch))
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    spec = MonitorSpec(
        id="m1",
        name="watch",
        tool="read",
        arguments={"path": str(watch)},
        every_ms=60_000,
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)  # baseline
        await drain_checks(session._monitors, session)
        clock[0] += 70_000
        watch.write_text("A\nB\n", encoding="utf-8")  # a pure addition: no gate
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)
        deltas = [event for event in events if getattr(event, "type", "") == "monitor_delta"]
        assert len(deltas) == 1, "the delivery behind the thread hop must have landed"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_normalized_quiet_tick_is_silent(tmp_path: Path, config_dir: Path) -> None:
    watch = tmp_path / "noisy.txt"
    watch.write_text("keep\nnoise 1\nupdated 2026-09-28T12:00:03Z\n", encoding="utf-8")
    session = _session(tmp_path, make_read_tool(watch))
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    spec = MonitorSpec(
        id="m1",
        name="noisy",
        tool="read",
        arguments={"path": str(watch)},
        every_ms=30_000,
        ignore=["^noise"],
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)  # baseline
        await drain_checks(session._monitors, session)

        # Only ignored lines and the timestamp changed.
        clock[0] += 60_000
        watch.write_text("keep\nnoise 555\nupdated 2026-09-29T09:00:00Z\n", encoding="utf-8")
        await session._monitors.pump(now_ms=clock[0] + 15_000)
        await drain_checks(session._monitors, session)

        deltas = [event for event in events if getattr(event, "type", "") == "monitor_delta"]
        assert deltas == [], "normalization must absorb timestamp and ignore-line churn"
        counters_path = config_dir / "monitors" / "state" / "sess" / "m1.json"
        counters = json.loads(counters_path.read_text(encoding="utf-8"))
        assert counters["checks"] == 2 and counters["deliveries"] == 0
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_session_gate_suppresses_a_non_material_change(
    tmp_path: Path, config_dir: Path
) -> None:
    """The §8 wiring's session path: ``Session(monitor_classify=...)`` reaches
    the scheduler the session builds, and a suppression writes the counters
    and emits NO delta — the whole fork, one layer above the scheduler's own
    suite.

    The callback is a fake seam on purpose: the classification package is
    ``tests/unit/classification``'s subject, and this rig is about the
    plumbing between the session and its scheduler.
    """
    watch = tmp_path / "watched.txt"
    watch.write_text("A\nB\n", encoding="utf-8")
    calls: list[str] = []

    async def classify(state: str) -> str | None:
        calls.append(state)
        return "non-material-metadata"

    session = Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[make_read_tool(watch)],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        cwd=str(tmp_path),
        monitor_classify=classify,
    )
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    session._emit = record  # type: ignore[method-assign]
    spec = MonitorSpec(
        id="m1",
        name="watch",
        tool="read",
        arguments={"path": str(watch)},
        every_ms=60_000,
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)  # baseline
        await drain_checks(session._monitors, session)

        clock[0] += 70_000  # past the interval plus the scheduler's jitter
        # An EDIT, not an append: a pure addition skips the gate by design
        # (``is_pure_addition``), so only a changed line proves this plumbing.
        watch.write_text("A\nC\n", encoding="utf-8")
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        deltas = [event for event in events if getattr(event, "type", "") == "monitor_delta"]
        assert deltas == [], "a suppressed change must not reach the conversation"
        assert calls, "the gate ran exactly because the heuristic hit"
        assert "C" in calls[0], calls[0]
        counters = monitor_state.read_counters(config_dir, session.session_id, "m1")
        assert counters is not None
        assert counters["suppressed"]["non_material_metadata"] == 1
        assert counters["deliveries"] == 0
        # ... and the derived index carries the count, so a cold listing can
        # say the change was suppressed instead of "nothing has changed".
        index_row = monitor_store.read_index(config_dir)[session.session_id]["monitors"][0]
        assert index_row["suppressed"]["non_material_metadata"] == 1
    finally:
        await session.dispose()


# --- The lazy-MCP window, through a REAL scheduler tick --------------------
#
# m4, 2026-11-05: a monitor over ``mcp__slack_read_channel`` lived in a session
# that went cold. A scheduled wake re-hosted it at 05:27, the tick beat the lazy
# MCP connect, and the state file recorded a genuine strike:
#
#   consecutive_failures: 1
#   last_error: monitor can't watch "mcp__slack_read_channel": it is not in
#               this session's tool set.
#
# These two tests drive the same path a wake does — ``set_monitor_schedules``
# (the restore entry point, which does not validate) then a due pump — and pin
# what the ladder records either side of the ``mcp__`` boundary.


def _dark_mcp_session(tmp_path: Path, config_dir: Path) -> tuple[Session, list[int]]:
    """A hosted session whose MCP layer has not connected, plus its clock.

    ``mcp_manager`` stays None exactly as it is before ``discover_and_connect``:
    the runtime boots and engages with MCP lazy, and the tool is absent only for
    that window.
    """
    session = _session(tmp_path, make_read_tool(tmp_path / "unused.txt"))
    session.mcp_manager = None
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    return session, clock


def _notice_kinds(session: Session) -> list[str]:
    """The lifecycle notices this session actually emitted, in order."""
    return [
        str(entry.payload.get("details", {}).get("kind"))
        for entry in session._transcript.entries()
        if entry.type == "message"
        and entry.payload.get("kind") == "custom"
        and entry.payload.get("custom_type") == "monitor_prompt"
    ]


@pytest.mark.asyncio
async def test_a_tick_before_the_mcp_layer_connects_charges_no_strike(
    tmp_path: Path, config_dir: Path
) -> None:
    """The measured m4 strike, end to end: an ``mcp__`` tick that beats the lazy
    connect records an unavailable EPISODE and stays armed, and the notice it
    eventually earns is the stall one — the ladder to a disable never starts.
    """
    session, clock = _dark_mcp_session(tmp_path, config_dir)
    spec = MonitorSpec(
        id="m4",
        name="slack",
        tool="mcp__slack_read_channel",
        arguments={"channel_id": "C0TEST"},
        every_ms=60_000,
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m4")
        assert counters is not None
        assert counters["consecutive_failures"] == 0, "the lazy-MCP window is not a failure"
        assert counters["unavailable_ticks"] == 1
        assert counters["last_note"] == "tool unavailable — retrying"
        assert "not connected in this session yet" in counters["last_error"]
        assert not counters.get("disabled")

        # 31 minutes later the episode crosses the stall window: one notice, and
        # the monitor is still armed. The 24-hour disable is the only thing past
        # this point, and it is unreachable while the watch keeps ticking.
        clock[0] += 31 * 60_000
        await session._monitors.pump(now_ms=clock[0] + 15_000)
        await drain_checks(session._monitors, session)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m4")
        assert counters is not None
        assert not counters.get("disabled"), counters.get("disabled_reason")
        assert counters["consecutive_failures"] == 0
        assert _notice_kinds(session) == ["stalled"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_builtin_that_is_gone_still_charges_a_strike_on_the_same_rig(
    tmp_path: Path, config_dir: Path
) -> None:
    """CONTROL: the same session and pump, a NON-MCP name that is not in the tool
    set. That absence is a real change, so the ordinary ladder must charge it —
    the lazy-MCP excuse may not leak onto builtins.
    """
    session, clock = _dark_mcp_session(tmp_path, config_dir)
    spec = MonitorSpec(
        id="m4",
        name="builtin",
        tool="bash",
        arguments={"command": "date -u"},
        every_ms=60_000,
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m4")
        assert counters is not None
        assert counters["consecutive_failures"] == 1
        assert counters["unavailable_ticks"] == 0
        assert "not in this session's tool set" in counters["last_error"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_genuine_removal_walks_the_disable_ladder(tmp_path: Path, config_dir: Path) -> None:
    """PROPERTY PIN: the lazy-MCP excuse must never cover a REAL absence.

    A manager exists only after ``attach_mcp_dispose``, which runs after the
    connect round — so a manager with an EMPTY registry means zero enabled
    servers (disabled entries never reach it): a name whose server is gone (a
    removed config entry, every entry disabled), not a load in progress. That
    has to keep charging strikes and end in a disable; classifying it
    ``unavailable`` would replace five truthful failures with 24 hours of
    "the runtime is still starting", which is the untrue-diagnosis class this
    PR removes. The five ticks below are the whole ladder.
    """
    session = _session(tmp_path, make_read_tool(tmp_path / "unused.txt"))
    session.mcp_manager = FakeManager([], servers=[])  # type: ignore[assignment]
    clock = [1_756_000_000_000]
    session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    spec = MonitorSpec(
        id="m9",
        name="retired",
        tool="mcp__retired_server_tool",
        arguments={"query": "x"},
        every_ms=60_000,
        created_at=1,
    )
    try:
        await session.set_monitor_schedules([spec])
        await session._monitors.pump(now_ms=clock[0] + 5_000)
        await drain_checks(session._monitors, session)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m9")
        assert counters is not None
        assert counters["consecutive_failures"] == 1, "a removal is a real failure"
        assert counters["unavailable_ticks"] == 0

        # Past each backoff (60s, 120s, 240s, 480s) to the fifth failure.
        for advance in (60_000, 120_000, 240_000, 480_000):
            clock[0] += advance
            await session._monitors.pump(now_ms=clock[0] + 5_000)
            await drain_checks(session._monitors, session)

        counters = monitor_state.read_counters(config_dir, session.session_id, "m9")
        assert counters is not None
        assert counters["consecutive_failures"] == 5
        assert counters.get("disabled"), "the ladder must still reach a disable"
        assert counters["disabled_kind"] == "checks"
        assert "not in this session's tool set" in counters["disabled_reason"]
        assert "still starting" not in counters["disabled_reason"]
    finally:
        await session.dispose()
