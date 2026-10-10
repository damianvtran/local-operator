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
  a job result, an incident notice, a wake+job mix — behaves exactly as today
  and a quiet delivery never suppresses it (case 8, the §14.2 F7 note);
* EXCEPT a peer-only run (the peer-notify policy, 2026-10-10): non-user
  inputs exclusively peer-family — possibly beside quiet wake/monitor
  deliveries — stay SILENT unless a user joins, a delivery asked to notify,
  or the run errors; that includes a peer reply that writes ordinary TEXT
  (case 9);
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
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.message_types import SESSION_INCIDENT_MESSAGE_TYPE
from local_operator.harness.types import (
    QUIET_TURN_KEY,
    AgentEndEvent,
    AgentTool,
    CustomMessage,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.monitors.delivery import MonitorDelivery
from local_operator.providers.failover import ProviderError
from local_operator.session.attention import AttentionStore
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin
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
async def test_a_queued_user_message_makes_a_peer_run_notify(tmp_path: Path) -> None:
    """The peer variant of the cell above: a peer-only run is quiet until a
    person is waiting — a typed message on the steering queue makes the run
    notify, exactly as it does for a quiet wake."""
    session = make_session(tmp_path, _complete_stream())
    try:
        session._run_triggers = {"internal"}
        session._run_input_types = {"peer_message"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False
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
# Case 8 (§14.2 F7, amended by the peer-notify policy) — non-peer internal and
# mixed runs stay loud; only an explicitly quieted class may silence a run.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_quiet_delivery_never_suppresses_non_peer_internal_or_mixed_runs(
    tmp_path: Path,
) -> None:
    session = make_session(tmp_path, _complete_stream())
    try:
        # A NON-peer internal input (a job result; any raw type outside the
        # peer family): unchanged — notifies even though every delivery was
        # quiet, because only a class the policy explicitly quieted may
        # silence a run.
        session._run_triggers = {"internal"}
        session._run_input_types = {"job_result"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # An internal row with NO custom type records the empty sentinel: it
        # can never match the peer family, so it keeps the loud default.
        session._run_input_types = {""}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # A non-peer internal folded beside a quiet wake: still loud.
        session._run_triggers = {"wake_prompt", "internal"}
        session._run_input_types = {"job_result"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # The exclusive case, as the control: quiet stays quiet.
        session._run_triggers = {"wake_prompt", "monitor_prompt"}
        session._run_input_types = set()
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False

        # ...until ONE delivery asks: the OR over deliveries wins.
        session._run_notify_requested = True
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# Case 9 (peer-notify policy, 2026-10-10) — a peer-only run stays silent by
# default; every non-peer class, a person, and the error arm keep it loud.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_only_run_is_silent_by_default(tmp_path: Path) -> None:
    """The policy matrix at the formula boundary. ``_run_input_types`` is set
    directly (as the trigger record is elsewhere in this file) so each row is
    the exact input set the pipeline would have recorded for it."""
    session = make_session(tmp_path, _complete_stream())
    try:
        # peer only → quiet.
        session._run_triggers = {"internal"}
        session._run_input_types = {"peer_message"}
        session._run_notify_requested = False
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False

        # hub_message is a member defensively — it cannot reach the record
        # today (every hub delivery routes through queue_aside) — so a future
        # producer lands quiet rather than loud.
        session._run_input_types = {"hub_message"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False

        # peer + quiet wake → quiet. Wake/monitor deliveries are not recorded
        # in _run_input_types; their own notify bit is the only raise.
        session._run_triggers = {"wake_prompt", "internal"}
        session._run_input_types = {"peer_message"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False

        # peer + a wake that asked to be told → loud.
        session._run_notify_requested = True
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
        session._run_notify_requested = False

        # peer + monitor: quiet / notify, same as the wake row.
        session._run_triggers = {"monitor_prompt", "internal"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is False
        session._run_notify_requested = True
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
        session._run_notify_requested = False

        # peer + job result → LOUD: the settled-job banner is deliberately
        # kept, and this row is the tripwire if the policy ever flips it.
        session._run_triggers = {"internal"}
        session._run_input_types = {"peer_message", "job_result"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # peer + an ask response/timeout → LOUD: a person deciding is never
        # quieter than a peer.
        session._run_input_types = {"peer_message", "ask_response"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True
        session._run_input_types = {"peer_message", "ask_timeout"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # peer + a CONSUMED typed user → user semantics win.
        session._run_triggers = {"user", "internal"}
        session._run_input_types = {"peer_message"}
        assert session._finalize_attention_notify(AgentEndEvent(messages=[])) is True

        # peer only + error → the error arm is untouched.
        session._run_triggers = {"internal"}
        session._run_input_types = {"peer_message"}
        assert (
            session._finalize_attention_notify(
                AgentEndEvent(messages=[], error="provider exploded")
            )
            is True
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_peer_message_answered_with_text_stays_quiet(tmp_path: Path) -> None:
    """THE OPERATOR'S REPRO, end to end: a peer message into an idle session
    whose reply is ordinary TEXT (not ``no_reply``) leaves notify=False on the
    end and on the store row. This is the write that used to re-arm the
    banner: before this policy, a text reply to a peer notified a user nobody
    had asked to hear. The row still exists and the session still reads
    unseen — discoverability is deliberately independent of ``notify`` — only
    the announcement is gone."""
    stream = _complete_stream()
    session = make_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.receive_peer_message(
            "child reporting in",
            mode="mailbox",
            wake=True,
            sender={"pid": 42, "conversation_name": "child"},
        )
        await _wait_published(session, stream)
        assert session._run_triggers == {"internal"}, "a peer run is an internal run"
        assert session._run_input_types == {"peer_message"}
        ends = _ends(events)
        assert len(ends) == 1
        assert ends[0].notify is False, "a text reply to a peer does not notify"
        state = await session.refresh_attention()
        assert state["kind"] == "complete"
        assert state["notify"] is False, "the one value every notifier reads"
        assert state["unseen"] is True, "the unread mark stays: read, not announced"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_peer_turn_raising_an_ask_stays_quiet_while_the_ask_surfaces(
    tmp_path: Path,
) -> None:
    """CHANNEL INDEPENDENCE: raising an ask mid-peer-turn surfaces the question
    on the queue's own path (an open record) while the run's completion value
    stays False. The ask's surface is the ask machinery's business — pending
    gate state, the picker, the phone card — and none of it reads §14's value,
    so this suppression cannot touch it.

    The third scripted call is the ask CLEARANCE gate's fork (the default
    ``LOP_ASK_GATE`` arm runs one short check off the main turn); its reply
    carries no ``VERDICT:`` line, so it parses as no-verdict and the question
    enqueues unchanged. With the gate switched off the third turn is simply
    unused — the script must not assume either way."""
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(
                    index=0,
                    id="a1",
                    name="ask",
                    argument_delta=json.dumps(
                        {
                            "questions": [
                                {
                                    "id": "q1",
                                    "question": "Which deploy window?",
                                    "options": [
                                        {"label": "now"},
                                        {"label": "later"},
                                    ],
                                }
                            ]
                        }
                    ),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [
                StreamTextDelta(delta="no structured verdict here"),
                StreamEndEvent(stop_reason="stop"),
            ],
            [StreamTextDelta(delta="the question is queued"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    async def _never_answered(_questions):
        return None  # pragma: no cover — the queued arm enqueues without the hook

    session.set_ask_handler(_never_answered)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.receive_peer_message(
            "child reporting in",
            mode="mailbox",
            wake=True,
            sender={"pid": 42, "conversation_name": "child"},
        )
        await _wait_published(session, stream)
        assert session._run_input_types == {
            "peer_message"
        }, "tool traffic never joins the input record — the ask call itself is not an input"
        ends = _ends(events)
        assert ends and ends[-1].notify is False
        queue = session.ask_queue()
        assert queue is not None
        open_records = [record for record in queue.open_records() if record["status"] == "open"]
        assert len(open_records) == 1, "the ask is on its own path, unanswered"
        assert open_records[0]["questions"][0]["question"] == "Which deploy window?"
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


# ---------------------------------------------------------------------------
# The quiet turn (docs/design/quiet-turns.md §4): a ``no_reply`` end publishes
# nothing and notifies nobody — and the two arms that must NOT be silenced.
# ---------------------------------------------------------------------------


def _quiet_session(tmp_path, stream, **kwargs) -> Any:
    """A session whose inventory holds the REAL ``no_reply`` tool.

    Mounted by the constructor's own capability merge — NO manual
    ``refresh_tools``: the hand-splice this helper used to carry is what let
    round 1's blocker (the tool absent from every real session) ship, because
    a spliced inventory proves the tool works, never that a session has it.
    """
    session = make_session(tmp_path, stream, **kwargs)
    assert any(tool.name == "no_reply" for tool in session._tools)
    return session


def _rows(directory: Path) -> list[dict[str, Any]]:
    """Every transcript row's payload, in order."""
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["payload"] for line in path.read_text().splitlines() if line]


@pytest.mark.asyncio
async def test_a_peer_only_quiet_run_publishes_nothing(tmp_path: Path) -> None:
    """A peer message answered with ``no_reply``: the end carries notify=False
    and the store gets NO completion row at all — hence no unread mark, no
    banner and no notifier call — while the transcript still holds the pair
    (the assistant call and its stamped result), which is what a viewer needs
    and what a replay must find wire-legal."""
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        ]
    )
    session = _quiet_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.receive_peer_message(
            "child reporting in",
            mode="mailbox",
            wake=True,
            sender={"pid": 42, "conversation_name": "child"},
        )
        await wait_for(lambda: bool(stream.requests))
        await wait_for(lambda: session._attention_run_settled)
        await wait_for(lambda: not session.is_streaming and not session._turn_lock.locked())

        assert session._run_triggers == {"internal"}, "a peer run is an internal run"
        assert session._run_input_types == {"peer_message"}, "the raw input type rides beside it"
        ends = _ends(events)
        assert len(ends) == 1
        assert ends[0].notify is False, "the one value every notifier reads"

        state = await session.refresh_attention()
        assert state["completion_token"] is None, "no completion row was published"
        assert state["unseen"] is False

        payloads = _rows(tmp_path / "sess")
        assert any(
            call.get("name") == "no_reply"
            for payload in payloads
            for call in (payload.get("tool_calls") or ())
        ), "the assistant call is persisted"
        assert any(
            ((payload.get("provider_payload") or {}).get("details") or {}).get(QUIET_TURN_KEY)
            is True
            for payload in payloads
            if payload.get("role") == "tool"
        ), "and so is its stamped result"
        marker = Transcript(tmp_path / "sess").latest_custom("completion_attention")
        assert (
            marker is not None and marker["eligible"] is False
        ), "the journal holds the eligible:false marker (what a republish boots from)"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_user_turn_refuses_the_quiet_call_and_answers_in_text(tmp_path: Path) -> None:
    """R15a: user semantics win. The quiet call comes back as an ``is_error``
    result carrying the refusal sentence, the model writes the answer on the
    next call, and the turn notifies as any user turn does."""
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="All good here."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _quiet_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session.prompt("how are you?")

        assert len(stream.requests) == 2, "the refusal bought the answer call"
        results = [
            message
            for message in session._context.messages
            if getattr(message, "role", None) == "tool"
        ]
        assert results and results[-1].is_error
        assert "A person asked this turn" in results[-1].text
        # ``details`` is None on a plain error result, so the read must sink
        # through it rather than iterate it.
        assert QUIET_TURN_KEY not in ((results[-1].provider_payload or {}).get("details") or {})
        state = await session.refresh_attention()
        assert state["kind"] == "complete" and state["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_notify_requested_wake_refuses_the_quiet_call(tmp_path: Path) -> None:
    """A delivery that asked to tell the user keeps its sentence path: the
    refusal names the ask, and the run notifies (§4)."""
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="The build is green."), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _quiet_session(tmp_path, stream)
    try:
        schedule = WakeSchedule(
            id="w-notify", message="report the build", next_due_at=0, created_at=0, notify=True
        )
        await session._deliver_wake(
            DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        )
        await wait_for(lambda: len(stream.requests) >= 2)
        await wait_for(lambda: session._attention_run_settled)

        results = [
            message
            for message in session._context.messages
            if getattr(message, "role", None) == "tool"
        ]
        assert results and results[-1].is_error
        assert "asked to tell the user" in results[-1].text
        state = await session.refresh_attention()
        assert state["kind"] == "complete" and state["notify"] is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_refusal_rule_is_the_denylist(tmp_path: Path) -> None:
    """R4, pinned at the seam that decides it: refuse iff a person asked or a
    delivery asked to notify — everything else, including every ``internal``
    run, may end quietly."""
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    try:
        session._run_triggers = {"internal"}
        assert await session._quiet_end_refusal() is None

        session._run_triggers = {"user"}
        assert await session._quiet_end_refusal() == (
            "A person asked this turn; answer them in one line."
        )

        session._run_triggers = {"wake_prompt"}
        session._run_notify_requested = True
        assert await session._quiet_end_refusal() == (
            "This wake or monitor asked to tell the user; say what they need to know."
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_queued_user_message_refuses_the_quiet_end(tmp_path: Path) -> None:
    """``awaiting_user`` counts too: a typed message sitting on the steering
    queue is a person waiting for an answer, and the runner must not end the
    turn silently underneath it."""
    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    try:
        session._run_triggers = {"wake_prompt"}
        session.steer("actually, also check staging")
        assert await session._quiet_end_refusal() == (
            "A person asked this turn; answer them in one line."
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_incident_notice_run_may_end_quietly(tmp_path: Path) -> None:
    """R4's reach: a run opened by a ``session_incident`` notice is ``internal``
    and may end quietly — the incident's operator-visible row is already
    published, so silence about it loses nothing. This is the denylist working
    as designed, not an accident: the same call in a user run is refused."""
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        ]
    )
    session = _quiet_session(tmp_path, stream)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        await session._prompt_messages(
            [
                CustomMessage(
                    custom_type=SESSION_INCIDENT_MESSAGE_TYPE,
                    attribution="system",
                    details={"text": "[session incident] provider hiccup"},
                )
            ]
        )
        await wait_for(lambda: session._attention_run_settled)

        assert session._run_triggers == {"internal"}
        ends = _ends(events)
        assert ends and ends[-1].notify is False
        state = await session.refresh_attention()
        assert state["completion_token"] is None
    finally:
        await session.dispose()


class _FailingSecondCall(ScriptedStream):
    """The first call answers the quiet batch; every call after it fails."""

    def __call__(self, request, signal):
        if self.requests:
            self.requests.append(request)

            async def gen():
                raise ProviderError(400, "the re-entry failed")
                yield  # pragma: no cover — generator shape only

            return gen()
        return super().__call__(request, signal)


@pytest.mark.asyncio
async def test_a_quiet_batch_whose_re_entry_errors_still_notifies(tmp_path: Path) -> None:
    """REVIEW R3: the notify force is skipped for the error arm, and the case
    is reachable — a quiet batch, a todo reminder at the yield boundary
    re-enters the loop, and the re-entry's provider call fails. The run then
    ends as an error with the quiet marker still the last tool result, and
    "an error always notifies, whatever the origins were" must not be silenced
    by the earlier quiet call."""
    session_id = "quiet-reentry"
    builtin.TODO_STORE[session_id] = [{"text": "ship it", "status": "pending"}]
    stream = _FailingSecondCall(
        [
            [
                StreamToolCallDelta(index=0, id="q1", name="no_reply", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        ]
    )
    session = _quiet_session(tmp_path, stream, session_id=session_id)
    events: list[Any] = []
    session.subscribe(events.append)
    try:
        schedule = WakeSchedule(
            id="w-quiet", message="check the build", next_due_at=0, created_at=0
        )
        await session._deliver_wake(
            DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        )
        await wait_for(lambda: len(stream.requests) >= 2)
        await wait_for(lambda: session._attention_run_settled)

        ends = _ends(events)
        assert ends, "the run still ends with an end event"
        end = ends[-1]
        assert session._run_ended_quiet(end) is True, (
            "the quiet marker is still the run's last word — this test is about "
            "the error arm's exemption, not about the predicate"
        )
        assert end.notify is True, "the error arm always notifies"
        state = await session.refresh_attention()
        assert state["kind"] == "error"
        assert state["notify"] is True
    finally:
        builtin.TODO_STORE.pop(session_id, None)
        await session.dispose()
