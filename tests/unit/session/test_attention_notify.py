"""§14.6: the origin-aware notify rule — computed once, read by every surface.

WHAT THESE TESTS PIN. One value per run, computed by the session at turn end
from the run's trigger record, stamped on the emitted ``AgentEndEvent`` and
published on the ``completions`` row. The rule's edges are R15a's:

* a user turn notifies, byte for byte today's behaviour (case 1);
* a wake-only / monitor-only run follows its delivery's own ``notify``
  (cases 2-4) — including the delivery's value being OR'd over several;
* user semantics win: a mixed run (a user message consumed OR merely queued
  while a delivery folds in) notifies with EXACTLY one publication — the
  failing assertion is duplication, not presence (cases 4-5);
* errors notify regardless of the parameter (case 6);
* a run whose non-user inputs are not exclusively wake/monitor deliveries —
  a peer message, a job result, a wake+peer mix — behaves exactly as today
  and a quiet delivery never suppresses it (case 8, the §14.2 F7 note);
* a pre-§14 row reads as notify=1 and a write migrates the store (case 7).

THE RIG IS THE REAL ONE: ``ScriptedStream`` + a real ``Session`` + the real
``AttentionStore``; deliveries spawn their turns exactly as in production
(``_deliver_wake``/``_deliver_monitor`` -> ``_prompt_messages``) and the
assertions read the store's state, the ``agent_end`` events the session
emitted, and the session's own trigger record (kept for the assertions only —
the product consumes it at turn end).
"""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.session.attention import AttentionStore
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _complete_stream() -> ScriptedStream:
    return ScriptedStream([[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]])


async def _wait_published(session: Any, stream: ScriptedStream) -> None:
    """Wait for the spawned turn to have published its completion row.

    Both terms are needed: the request proves the turn RAN, the settled flag
    proves it finished (``_publish_attention_outcome`` is the last step), and
    the token proves the row exists in the store handle the session keeps.
    """
    await wait_for(lambda: bool(stream.requests))
    await wait_for(
        lambda: session._attention_run_settled and bool(session._attention.get("completion_token"))
    )


def _ends(events: list[Any]) -> list[AgentEndEvent]:
    return [event for event in events if getattr(event, "type", None) == "agent_end"]


# ---------------------------------------------------------------------------
# Case 1 — a user turn notifies, byte for byte.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_user_turn_notifies_and_the_value_rides_the_end_and_the_row(
    tmp_path: Path,
) -> None:
    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("Work")
        state = await session.refresh_attention()
        assert state["kind"] == "complete"
        assert state["unseen"] is True
        assert state["notify"] is True
        assert session._run_triggers == {"user"}
        ends = _ends(events)
        assert len(ends) == 1, "exactly one end per prompt — the ONE value rides it"
        assert ends[0].notify is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Cases 2-3 — wake-only runs follow the delivery's own notify.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_quiet_wake_turn_is_silent_on_the_event_and_the_row(
    tmp_path: Path,
) -> None:
    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        schedule = WakeSchedule(id="w1", message="check the build", next_due_at=0, created_at=0)
        assert schedule.notify is False, "the schedule's default is the contract's quiet"
        await session._deliver_wake(
            DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        )
        await _wait_published(session, stream)
        state = await session.refresh_attention()
        assert state["kind"] == "complete"
        assert state["notify"] is False
        assert session._run_triggers == {"wake_prompt"}
        assert session._run_notify_requested is False
        ends = _ends(events)
        assert len(ends) == 1
        assert ends[0].notify is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_discretionary_wake_notifies(tmp_path: Path) -> None:
    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        schedule = WakeSchedule(
            id="w1",
            message="check the build",
            next_due_at=0,
            created_at=0,
            notify=True,
        )
        await session._deliver_wake(
            DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        )
        await _wait_published(session, stream)
        state = await session.refresh_attention()
        assert state["notify"] is True
        assert session._run_triggers == {"wake_prompt"}
        assert session._run_notify_requested is True
        ends = _ends(events)
        assert len(ends) == 1
        assert ends[0].notify is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 4 — a monitor-only run follows the delivery's own notify.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_turns_follow_the_specs_notify(tmp_path: Path) -> None:
    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    try:
        quiet = MonitorDelivery(
            monitor_id="m1",
            name="pricing page",
            tool="web_fetch",
            changes=2,
            checks=5,
            skipped=0,
            delta_text="- $10\n+ $12",
            at_ms=0,
        )
        assert quiet.notify is False
        await session._deliver_monitor(quiet)
        await _wait_published(session, stream)
        state = await session.refresh_attention()
        assert state["kind"] == "complete"
        assert state["notify"] is False
        assert session._run_triggers == {"monitor_prompt"}
    finally:
        await session.dispose()

    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    try:
        loud = MonitorDelivery(
            monitor_id="m2",
            name="pricing page",
            tool="web_fetch",
            changes=1,
            checks=3,
            skipped=0,
            delta_text="- $10\n+ $11",
            at_ms=0,
            notify=True,
        )
        await session._deliver_monitor(loud)
        await _wait_published(session, stream)
        state = await session.refresh_attention()
        assert state["notify"] is True
        assert session._run_notify_requested is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 5 — user semantics win, including queued-but-unconsumed (awaiting_user).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_queued_user_message_makes_a_quiet_wake_notify(tmp_path: Path) -> None:
    session = make_session(tmp_path, _complete_stream())
    try:
        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = False
        # No user message anywhere yet: the rule's default for a quiet
        # wake-only run holds.
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False
        # A typed message sits in the steering queue, unconsumed: §14.2's
        # `awaiting_user` clause fires, and it fires at OUTCOME time.
        session.steer("actually, also check staging")
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_queued_delivery_does_not_count_as_an_awaiting_user(tmp_path: Path) -> None:
    """The queue holds a custom delivery, not a person: no awaiting_user."""
    from local_operator.harness.types import CustomMessage

    session = make_session(tmp_path, _complete_stream())
    try:
        session._steering_queue.put_nowait(
            CustomMessage(
                custom_type="peer_message",
                attribution="user",
                details={"text": "beep"},
            )
        )
        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_mixed_user_and_quiet_wake_run_notifies_with_one_publication(
    tmp_path: Path,
) -> None:
    """§14.6.4 driven end to end: a typed prompt in flight, a quiet wake folded
    in mid-turn. The run keeps user semantics AND stays one publication — the
    failing assertion is a second frame/row, not merely presence."""
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="block",
            content=[TextContent(text="finished")],
        )

    tool = AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=blocking_execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="acked"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        prompt_task = asyncio.ensure_future(session.prompt("long task"))
        await wait_for(lambda: tool_started.is_set())
        schedule = WakeSchedule(id="w1", message="wake up", next_due_at=0, created_at=0)
        await session._deliver_wake(
            DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        )
        release_tool.set()
        await prompt_task

        state = await session.refresh_attention()
        assert state["kind"] == "complete"
        assert state["notify"] is True, "user semantics win over a quiet delivery"
        assert session._run_triggers == {"user", "wake_prompt"}, (
            "the wake folded into the SAME run — and its quiet flag did not "
            "release user semantics"
        )
        ends = _ends(events)
        assert len(ends) == 1, "one run, one end — a second would double-announce"
        # One publication: the receipt watermark advanced by exactly one row.
        assert state["revision"][0] == 1
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 6 — errors notify regardless; interrupted stays formula-quiet.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_errored_quiet_wake_still_notifies(tmp_path: Path) -> None:
    session = make_session(tmp_path, _complete_stream())
    try:
        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = False
        await session._emit(AgentEndEvent(messages=[], error="provider exploded"))
        assert session._attention_outcome is not None
        assert session._attention_outcome.notify is True, "errors always notify"
        await session._publish_attention_outcome()
        state = await session.refresh_attention()
        assert state["kind"] == "error"
        assert state["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_interrupted_quiet_wake_is_formula_quiet(tmp_path: Path) -> None:
    """`interrupted` is suppressed by the readers' kind filters either way; the
    VALUE still follows the rule (no error clause), which is what the readers'
    filters are layered on."""
    session = make_session(tmp_path, _complete_stream())
    try:
        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[], aborted=True)) is False
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 8 (§14.2 F7) — non-wake/monitor origins behave exactly as today.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_quiet_delivery_never_suppresses_internal_or_mixed_runs(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, _complete_stream())
    try:
        # A peer message (or any internal input) joined the run: unchanged —
        # notifies even though every delivery was quiet, because the delivery
        # parameter only governs runs whose non-user triggers are EXCLUSIVELY
        # wake/monitor.
        session._run_triggers = {"internal"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        session._run_triggers = {"wake_prompt", "internal"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # The exclusive case, as the control: quiet stays quiet.
        session._run_triggers = {"wake_prompt", "monitor_prompt"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False

        # ...until ONE delivery asks: the OR over deliveries wins.
        session._run_notify_requested = True
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 7 — a pre-§14 row reads as notify=1, and a write migrates the store.
# ---------------------------------------------------------------------------


def test_a_pre_field_row_reads_as_notifying_and_a_write_migrates(tmp_path: Path) -> None:
    path = tmp_path / "attention.db"
    old_token = str(uuid.uuid4())
    store = AttentionStore(path)
    store.publish("session/a", old_token, "completion-old", "complete", notify=True)

    # Simulate a database written before the field existed: drop the column
    # outright, so both read paths face exactly what an old build left behind
    # (not a mocked reader, the real missing-column read).
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE completions DROP COLUMN notify")

    # READ-ONLY paths must not need the column; the old row reads as
    # notifying, which was its behaviour.
    stored = AttentionStore(path)
    assert stored.state("session/a")["notify"] is True
    assert [row["notify"] for row in stored.published_since(0)] == [True]

    # A WRITE migrates (additive ALTER), backfills the old row with DEFAULT 1,
    # and lands the new explicit value.
    new_token = str(uuid.uuid4())
    migrated = stored.publish("session/a", new_token, "message-new", "complete", notify=False)
    assert migrated["notify"] is False
    with sqlite3.connect(path) as conn:
        old_value = conn.execute(
            "SELECT notify FROM completions WHERE token=?", (old_token,)
        ).fetchone()
        new_value = conn.execute(
            "SELECT notify FROM completions WHERE token=?", (new_token,)
        ).fetchone()
    assert old_value == (1,), "a pre-field row must be backfilled to notify=1"
    assert new_value == (0,)
