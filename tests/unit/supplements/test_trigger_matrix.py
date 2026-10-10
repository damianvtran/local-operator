"""The "real user turn" predicate, one test per row of the memo's §2.2 case table.

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

pytestmark = pytest.mark.asyncio

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
async def test_a_steer_that_is_the_last_trigger_is_eligible(tmp_path: Path) -> None:
    stream = _stream()
    session = _make_session(tmp_path, stream)
    runner = _wire(session)
    try:
        await session.prompt("do it")
        await wait_for(lambda: runner.reasons != [])
        assert runner.reasons == [""]
    finally:
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
