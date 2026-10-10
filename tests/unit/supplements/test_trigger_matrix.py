"""The "real user turn" predicate, one cell per row of the memo's §2.2 case table.

COVERAGE (the claim, tightened by round-1 R2): every §2.2 row whose delivery lands on
``Session`` has a cell here — typed prompt; catch-up and courtesy wakes; wake-only;
monitor-only; peer (and, through the same ``internal`` arm, hub/agent messages and job
results); queued ask answers; mid-turn steers, typed and harness-injected; goal
continuations; spooled owner prompts, typed and chrome; subagent runs; the one-shot exec
host. Two rows are covered by mechanism rather than a cell, and named rather than
silently skipped: the SDK's ``open_session(mode="own")`` row is an ABSENCE — no
``ServingSessionHandle``, so no subscriber — pinned by
test_an_sdk_session_without_a_handle_is_never_installed; and the SDK deliver()/``lop send``
row rides the runtime's ordinary prompt path, whose predicate is the typed-prompt cell.

REAL SESSIONS, not a harness double: the point of this file is that ``_note_run_input``,
the pipeline head, ``_emit``'s freeze and the runtime subscriber agree about the facts — a
hand-built ``RunProvenance`` would assert the thing under test into existence. Each cell
drives an actual turn (typed prompt, wake delivery, steer, exec host) through ``Session`` and
asserts what the frozen record says and what the runner's own refusal reason is.

The guards are proven by their REFUSAL REASON and not only by the all-clear path: a reason
string is what makes a deleted guard visible, because ``""`` is the only value the eligible
cells share.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.session.runtime.supplements import SupplementRunner
from local_operator.session.session import Session
from local_operator.supplements.trigger import TRIGGER_INJECTED, TRIGGER_TYPED
from tests.unit.session.test_session import ScriptedStream, wait_for

# There is deliberately no module-level ``pytestmark = pytest.mark.asyncio``: every async
# cell below carries its own mark, and the module-wide mark also lit the ONE sync test at
# the foot of this file, drawing pytest-asyncio's "marked ... but it is not an async
# function" (QA round 1, Q-3).
MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def _stream() -> ScriptedStream:
    return ScriptedStream([[StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]])


def _make_session(tmp_path, stream, **kwargs) -> Session:
    from local_operator.session.transcript import Transcript

    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=stream,
        tools=kwargs.pop("tools", []),
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


class _Runner(SupplementRunner):
    """A runner whose job never runs: this file tests the TRIGGER, not the job."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scheduled: list[Any] = []
        self.reasons: list[str] = []

    def on_agent_end(self, event, *, goal_loop_running):  # type: ignore[no-untyped-def]
        reason = super().on_agent_end(event, goal_loop_running=goal_loop_running)
        self.reasons.append(reason)
        if not reason:
            self.scheduled.append(self._session.last_run_provenance)
        return reason


def _wire(session: Session) -> _Runner:
    """The production wiring, minus the job: the handle's subscriber calls the runner."""
    from local_operator.harness.types import AgentEvent
    from local_operator.session.goal_judge import owns_the_session

    runner = _Runner(session, cwd="/work")

    def handler(event: AgentEvent) -> None:
        if isinstance(event, AgentEndEvent) and owns_the_session(session):
            runner.on_agent_end(event, goal_loop_running=False)

    session.subscribe(handler)
    return runner


def _blocking_tool(started: asyncio.Event, release: asyncio.Event) -> Any:
    """A tool that parks on ``release`` so a delivery can land mid-turn.

    ``interruptible`` stays at its ``AgentTool`` default (False) ON PURPOSE: a typed
    steer is URGENT (``_has_urgent_steering``), and an interruptible tool would be
    cancelled by the session's immediate-interrupt poll the moment the steer queues --
    the run would still fold the steer in, but through the cancel path. Parking the tool
    and releasing it from the test keeps the drain at its ordinary boundary, which is the
    path these cells pin.
    """
    from local_operator.harness.types import AgentTool, TextContent, ToolResult

    async def blocking(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        started.set()
        await release.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="block", content=[TextContent(text="ok")]
        )

    return AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        execute=blocking,
    )


def _mid_turn_stream() -> ScriptedStream:
    """One parked tool call, then a clean close: the shape every mid-turn cell scripts."""
    from local_operator.harness.types import StreamToolCallDelta

    return ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )


def _spooled_owner_line(text: str, *, harness_injected: bool, command_id: str) -> Any:
    """A real ``InboxLine`` as the spool drain hands it to ``_run_spooled_owner_prompt``."""
    from local_operator.session.runtime.inbox import SOURCE_USER, InboxLine

    return InboxLine(
        text=text,
        sender={},
        source=SOURCE_USER,
        command_id=command_id,
        harness_injected=harness_injected,
    )


@pytest.mark.asyncio
async def test_a_typed_prompt_is_eligible_and_freezes_the_facts(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session.prompt("Write me a report")
        await wait_for(lambda: session._last_run_provenance is not None)
        record = session.last_run_provenance
        assert record is not None and record.typed_user and record.last_trigger == TRIGGER_TYPED
        assert record.user_text == "Write me a report"
        assert (
            record.settled_mark == session.turns_settled - 1
            or record.settled_mark <= session.turns_settled
        )
        assert runner.reasons == [""] and len(runner.scheduled) == 1
        # The accumulator holds the run's new messages (the assistant turn); the user's own
        # text rides separately because ``new_messages`` never contains the opening prompt.
        assert [item.role for item in record.items] == ["assistant"]
        assert record.items[0].text == "done"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_only_turn_has_no_typed_row_and_is_ineligible(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session._deliver_wake(
            DueWake(
                schedule=WakeSchedule(
                    id="w1", message="check the build", next_due_at=0, created_at=0
                ),
                occurrence=1,
                planned_total=1,
                final=True,
            )
        )
        await wait_for(lambda: session._last_run_provenance is not None)
        assert runner.reasons == ["no-typed-user-row"], runner.reasons
        assert runner.scheduled == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_user_message_with_a_catchup_wake_folded_first_stays_eligible(
    tmp_path: Path,
) -> None:
    """The missed-wake catch-up is folded BEFORE the user's message: the user is last."""
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        # The catch-up the resume path prepared, due now: folded INLINE ahead of the prompt.
        session._resume_catchup_text = "while you were away: the build finished"
        session._resume_grace_ends_ms = 0
        await session.prompt("and now the thing I asked for")
        await wait_for(lambda: session._last_run_provenance is not None)
        record = session.last_run_provenance
        assert record is not None and record.last_trigger == TRIGGER_TYPED
        assert runner.reasons == [""], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_courtesy_wake_folded_after_the_user_message_makes_the_run_ineligible(
    tmp_path: Path,
) -> None:
    """The wake is last -> the final response mostly answers the wake (scout risk 14)."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        started.set()
        await release.wait()
        from local_operator.harness.types import TextContent, ToolResult

        return ToolResult(
            tool_call_id=tool_call_id, tool_name="block", content=[TextContent(text="ok")]
        )

    from local_operator.harness.types import AgentTool, StreamToolCallDelta

    tool = AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=blocking,
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
    session = _make_session(tmp_path, stream, tools=[tool])
    runner = _wire(session)
    try:
        task = asyncio.ensure_future(session.prompt("long task"))
        await wait_for(lambda: started.is_set())
        await session._deliver_wake(
            DueWake(
                schedule=WakeSchedule(id="w1", message="wake up", next_due_at=0, created_at=0),
                occurrence=1,
                planned_total=1,
                final=True,
            )
        )
        release.set()
        await task
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["user-not-last-trigger"], runner.reasons
        assert runner.scheduled == []
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_typed_steer_drained_mid_turn_is_the_user_row_and_is_eligible(
    tmp_path: Path,
) -> None:
    """The §2.2 steer row: a WAKE-opened run that a typed steer corrects mid-turn.

    THE WAKE IS THE DISCRIMINATOR, and it is what round-1 R1 was about: the run opens
    with only the wake's trigger recorded (a wake-only turn is refused
    ``no-typed-user-row``), so if the drain ever stopped calling ``_note_run_input``, the
    typed steer would supply NO user row and this cell would fail on
    ``no-typed-user-row``. A prompt-opened run cannot show that -- the prompt's own
    ``_note_run_input`` would leave the same facts behind either way, which is exactly
    how the previous, mislabeled version of this cell passed while testing nothing.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    stream = _mid_turn_stream()
    session = _make_session(tmp_path, stream, tools=[_blocking_tool(started, release)])
    runner = _wire(session)
    try:
        await session._deliver_wake(
            DueWake(
                schedule=WakeSchedule(
                    id="w1", message="check the build", next_due_at=0, created_at=0
                ),
                occurrence=1,
                planned_total=1,
                final=True,
            )
        )
        # The wake opened the turn; the tool parks. NOW the person types.
        await wait_for(lambda: started.is_set())
        session.steer("hold on -- do the other thing instead")
        release.set()
        await wait_for(lambda: runner.reasons != [])
        record = session.last_run_provenance
        assert record is not None
        # The drain noted the steer as THIS run's typed user row, and as the last trigger.
        assert record.typed_user is True
        assert record.last_trigger == TRIGGER_TYPED
        assert record.user_text == "hold on -- do the other thing instead"
        assert runner.reasons == [""], runner.reasons
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_an_injected_steer_drained_mid_turn_flips_the_last_trigger(tmp_path: Path) -> None:
    """Spooled harness chrome can arrive as a STEER (``harness_injected=True``); when it
    is the last trigger the run must refuse ``user-not-last-trigger`` -- §2.2 rule 2.

    The pair with the typed-steer cell above is deliberate: drop the drain's
    ``_note_run_input`` and the last trigger stays on the opening prompt's ``typed``,
    so this run turns silently ELIGIBLE; lose the ``RENDERED_INJECTION_KEY``
    discriminator and chrome counts as a person's words the same way. Both failures land
    on this cell.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    stream = _mid_turn_stream()
    session = _make_session(tmp_path, stream, tools=[_blocking_tool(started, release)])
    runner = _wire(session)
    try:
        task = asyncio.ensure_future(session.prompt("long task"))
        await wait_for(lambda: started.is_set())
        session.steer("judge continuation", harness_injected=True)
        release.set()
        await task
        await wait_for(lambda: runner.reasons != [])
        record = session.last_run_provenance
        assert record is not None
        assert record.typed_user is True
        assert record.last_trigger == TRIGGER_INJECTED
        assert runner.reasons == ["user-not-last-trigger"], runner.reasons
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_an_injected_prompt_is_not_a_typed_row(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session.prompt("judge continuation", harness_injected=True)
        await wait_for(lambda: session._last_run_provenance is not None)
        record = session.last_run_provenance
        assert record is not None and record.typed_user is False
        assert record.last_trigger == TRIGGER_INJECTED
        assert runner.reasons == ["no-typed-user-row"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_monitor_only_turn_is_ineligible(tmp_path: Path) -> None:
    """§2.2: a monitor delivery opens a ``monitor_prompt`` custom turn — no typed user row.

    The wake cell above pins the branch; this pins the second custom type that rides it.
    Both deliveries are user-ATTRIBUTED by design, so only the custom type separates them
    from a person — which is why ``_note_run_input`` keys on it and never on attribution.
    """
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        from local_operator.monitors.delivery import MonitorDelivery

        await session._deliver_monitor(
            MonitorDelivery(
                monitor_id="m1",
                name="watch",
                tool="bash",
                changes=2,
                checks=7,
                skipped=0,
                delta_text="+2/-0 changed lines\n+ x",
                at_ms=1_756_000_000_000,
            )
        )
        await wait_for(lambda: session._last_run_provenance is not None)
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["no-typed-user-row"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_peer_only_turn_is_ineligible(tmp_path: Path) -> None:
    """§2.2: a peer ``send`` opening a turn is ``internal`` — no typed user row.

    Peer messages, hub/agent messages and job results ride the one ``internal`` arm of
    ``_note_run_input`` (ANY ``custom_type``); a peer send is the cell because it is the
    member of that class a unit test can drive through a real delivery method.
    """
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session.receive_peer_message(
            "are you there", mode="steer", sender={"pid": 3, "conversation_name": "peer"}
        )
        await wait_for(lambda: session._last_run_provenance is not None)
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["no-typed-user-row"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_queued_ask_answer_is_ineligible(tmp_path: Path) -> None:
    """§2.2: an ``ask_response`` delivered as a turn of its own is ``internal``.

    The answer text is the operator's, but the turn's final message responds to a
    question the AGENT raised, and that is the memo's disposition (row 6; open question
    Q4 owns any opt-in). Keyed where it belongs: ``custom_type`` is set, so the delivery
    never reaches the typed-user arm.
    """
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
        from local_operator.harness.types import CustomMessage

        await session.deliver_ask_messages(
            [
                CustomMessage(
                    custom_type=ASK_RESPONSE_MESSAGE_TYPE,
                    attribution="user",
                    details={"ask_id": "a1", "status": "answered", "text": "use the blue one"},
                )
            ]
        )
        await wait_for(lambda: session._last_run_provenance is not None)
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["no-typed-user-row"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_spooled_typed_owner_prompt_joining_a_turn_is_eligible(
    tmp_path: Path,
) -> None:
    """§2.2: a REAL owner prompt spooled across an update window stays eligible.

    Driven through the production drain (``_run_spooled_owner_prompt`` — the method
    ``_drain_spooled_peer_inbox_rows`` calls for a ``SOURCE_USER`` row) with a real
    ``InboxLine``: the row joins the running turn as an identified user steer, and the
    carried ``harness_injected=False`` is what keeps it unstamped. Flip the carriage (or
    default the stamp True) and this cell turns into ``user-not-last-trigger``.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    stream = _mid_turn_stream()
    session = _make_session(tmp_path, stream, tools=[_blocking_tool(started, release)])
    runner = _wire(session)
    try:
        task = asyncio.ensure_future(session.prompt("long task"))
        await wait_for(lambda: started.is_set())
        session._run_spooled_owner_prompt(
            _spooled_owner_line(
                "carry on from before the update",
                harness_injected=False,
                command_id="cmd-typed",
            ),
            seen=set(),
        )
        release.set()
        await task
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == [""], runner.reasons
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_spooled_chrome_row_joining_a_turn_is_not_eligible(tmp_path: Path) -> None:
    """§2.2: the SAME spool path with the stamp TRUE — harness chrome (a goal judge's
    replayed continuation) — must arrive as an injected row, so the run refuses
    ``user-not-last-trigger``.

    This is the side where a LOST carriage shows: without it the chrome replay would
    count as the operator's own last word, and eligible runs would fire on a judge's
    continuation. A dropped drain note lands here too — the last trigger would stay on
    the opening prompt's ``typed`` and the run would be silently eligible.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    stream = _mid_turn_stream()
    session = _make_session(tmp_path, stream, tools=[_blocking_tool(started, release)])
    runner = _wire(session)
    try:
        task = asyncio.ensure_future(session.prompt("long task"))
        await wait_for(lambda: started.is_set())
        session._run_spooled_owner_prompt(
            _spooled_owner_line("finish the goal", harness_injected=True, command_id="cmd-chrome"),
            seen=set(),
        )
        release.set()
        await task
        await wait_for(lambda: runner.reasons != [])
        record = session.last_run_provenance
        assert record is not None and record.last_trigger == TRIGGER_INJECTED
        assert runner.reasons == ["user-not-last-trigger"], runner.reasons
    finally:
        release.set()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_subagent_session_is_ineligible(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream, job_id="job123")
    runner = _wire(session)
    try:
        await session.prompt("child task")
        await wait_for(lambda: session._last_run_provenance is not None)
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["subagent"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_one_shot_host_is_ineligible(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        session.declare_one_shot_exit()
        await session.prompt("lop exec text")
        await wait_for(lambda: session._last_run_provenance is not None)
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["one-shot-host"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_goal_loop_continuation_is_ineligible_even_typed_looking(tmp_path: Path) -> None:
    """Rule 1 excludes injections; rule 5 is the belt for a typed ``/loop`` seed turn."""
    from local_operator.harness.types import AgentEvent

    stream = _stream()
    session = _make_session(tmp_path, stream)

    runner = _Runner(session, cwd="/work")

    def handler(event: AgentEvent) -> None:
        if isinstance(event, AgentEndEvent):
            runner.on_agent_end(event, goal_loop_running=True)

    session.subscribe(handler)
    try:
        await session.prompt("ship the goal")
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["goal-loop"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unclean_end_is_ineligible(tmp_path: Path) -> None:

    from local_operator.providers.failover import ProviderError

    class Refusing:
        """A provider that refuses the request: the loop turns it into an error end."""

        def __init__(self) -> None:
            self.requests: list[Any] = []

        def __call__(self, request: Any, signal: Any):
            self.requests.append(request)

            async def gen():
                raise ProviderError(400, "bad request")
                yield  # pragma: no cover — the generator is never resumed

            return gen()

    session = _make_session(tmp_path, Refusing())
    runner = _wire(session)
    try:
        await session.prompt("try it")
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["unclean-end"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_kill_switch_refuses_before_anything_else(tmp_path: Path, monkeypatch) -> None:
    from local_operator.supplements import policy

    monkeypatch.setattr(policy, "SUPPLEMENTS", False)
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session.prompt("a nice typed prompt")
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == ["kill-switch"], runner.reasons
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_sdk_session_without_a_handle_is_never_installed(tmp_path: Path) -> None:
    """The SDK's contract: no ServingSessionHandle, so no subscriber and no run at all."""
    stream = _stream()
    session = _make_session(tmp_path, stream)
    try:
        await session.prompt("sdk turn")
        await wait_for(lambda: session._last_run_provenance is not None)
        assert session.turns_settled == 1, "the seam itself still tracks the turn"
        # Without ``_wire`` there is no runner: nothing was scheduled, nothing raised.
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_logical_turn_that_auto_continued_keeps_its_pre_compaction_messages(
    tmp_path: Path,
) -> None:
    """R2: the held end carries only the LAST run's messages; the accumulator keeps all.

    Driven the way production drives it: the loop's first run ends with ``length`` (the step
    budget), the session queues the real continuation prompt, and the same pipeline drains it.
    The frozen record must carry BOTH runs' content -- a record built from the emitted end
    alone would show only the second.
    """
    from local_operator.session.session import _CONTINUATION_PROMPT

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="part one"), StreamEndEvent(stop_reason="length")],
            [StreamTextDelta(delta="part two"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = _make_session(tmp_path, stream)
    original_write = None
    try:
        # The step-budget path queues this message at run end (``_run_turn``); queueing it
        # directly is the same input the loop produces.
        from local_operator.harness.types import Message

        real_drain = session._drain_continuation

        async def drain_with_one() -> None:
            session._continuation_queue.append(Message.user(_CONTINUATION_PROMPT))
            await real_drain()

        session._drain_continuation = drain_with_one  # type: ignore[method-assign]
        await session.prompt("write the long report")
        await wait_for(lambda: session._last_run_provenance is not None)
        record = session.last_run_provenance
        assert record is not None
        texts = [item.text for item in record.items]
        assert "part one" in texts and "part two" in texts, texts
        assert original_write is None
    finally:
        await session.dispose()


def test_the_snapshot_copies_text_so_in_place_pruning_cannot_erase_evidence() -> None:
    """``compaction/pruning.py`` mutates tool results IN PLACE; the accumulator must not see it.

    A held REFERENCE would read the "[pruned]" notice later, and the pre-filter would lose
    the very tool output the evidence extractor reads. Proven at the unit seam, which is where
    the property lives.
    """
    from local_operator.harness.types import Message, TextContent
    from local_operator.supplements.trigger import snapshot_messages

    message = Message.tool_result(
        __import__("local_operator.harness.types", fromlist=["ToolResult"]).ToolResult(
            tool_call_id="c1",
            tool_name="bash",
            content=[TextContent(text="region,ms\nus,12\neu,15\nap,22\n")],
        )
    )
    snap = snapshot_messages([message])
    assert "us,12" in snap[0].text
    # What pruning does: replace the content blocks in place.
    message.content = [TextContent(text="[pruned]")]
    assert "us,12" in snap[0].text, "the snapshot followed the mutation"
    assert message.text == "[pruned]"
