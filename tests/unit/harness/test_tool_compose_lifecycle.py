"""The compose surface's ENDINGS: what the producer owes a row it announced.

A composing row is the UI's optimistic prediction that a call exists. The
producer used to announce the prediction's beginning and then only one of its
three endings — the call starts — so a call that was QUEUED behind a sibling's
execution group, or that NEVER RAN at all (a planning failure, a duplicate id,
a steering skip), kept a row claiming the model was still dictating it. That is
the operator's report: a `wake` composed into the same step as a
`wait(wait_ms=1800000)` and executed by a LATER group, shown as `composing…`
with a ticking clock for the sibling's whole half-hour.

Two endings, both on the compose wire rather than on the tool-record wire, and
the second half of these tests is as important as the first:

* a terminal `dictation_complete` frame per latched call at stream end;
* a terminal frame carrying a bounded `not_run_reason` for a call the harness
  judged and will not execute.

The tool-record surface is deliberately untouched — the API server pairs records
by id, and a synthetic `tool_execution_start`/`_end` for a call that never ran
would either resurrect a record that was never opened or close the REAL call's
record early. Every test here asserts BOTH halves: the frame arrives, and no
lifecycle event does.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.loop import NOT_RUN_REASON_MAX_CHARS, AgentLoop, LoopContext
from local_operator.harness.types import (
    AgentEvent,
    AgentTool,
    Message,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)

from .test_loop import ScriptedStream, echo_tool, make_config, tool_call_delta

#: Two calls in one step. The first is executed by the batch's shared group, the
#: second is `exclusive` and therefore runs in a LATER group — the shape that
#: produced the reported row (a `wake` behind a long `wait`).
WAIT_ID = "call_wait_bf321"
WAKE_ID = "call_wake_9d1c"


async def _events(loop: AgentLoop, stream: ScriptedStream, tools: list[Any]) -> list[AgentEvent]:
    context = LoopContext(system_blocks=["sys"], tools=tools)
    seen: list[AgentEvent] = []
    async for event in loop.run([Message.user("go")], context, make_config(stream), None):
        seen.append(event)
    return seen


def _frames(events: list[AgentEvent], call_id: str) -> list[ToolCallComposeEvent]:
    return [
        event
        for event in events
        if isinstance(event, ToolCallComposeEvent) and event.tool_call_id == call_id
    ]


def _terminal(events: list[AgentEvent], call_id: str) -> ToolCallComposeEvent:
    """The NEWEST terminal frame for a call — the one a consumer must act on.

    A call that never runs has TWO, and the order is the contract: the stream
    end says the dictation is over (the call is queued), and the batch's plan,
    formed moments later, says the call will not run and why. The compose slot
    keeps the newest frame per call, so the verdict is what a late joiner and a
    live viewer both end up acting on.
    """
    terminal = _terminal_frames(events, call_id)
    assert terminal, f"{call_id} has a terminal dictation frame"
    return terminal[-1]


def _terminal_frames(events: list[AgentEvent], call_id: str) -> list[ToolCallComposeEvent]:
    return [frame for frame in _frames(events, call_id) if frame.dictation_complete]


def _two_call_step() -> ScriptedStream:
    """One step composing two calls, then the model's ordinary closing turn."""
    return ScriptedStream(
        [
            [
                tool_call_delta(0, id=WAIT_ID, name="wait", args='{"text":"1800000"}'),
                tool_call_delta(1, id=WAKE_ID, name="wake", args='{"text":"30m"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )


def _lifecycle(events: list[AgentEvent], call_id: str) -> tuple[list[Any], list[Any]]:
    return (
        [e for e in events if isinstance(e, ToolExecutionStartEvent) and e.tool_call_id == call_id],
        [e for e in events if isinstance(e, ToolExecutionEndEvent) and e.tool_call_id == call_id],
    )


# ---------------------------------------------------------------------------
# F1 — the dictation ending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_announced_call_gets_one_terminal_dictation_frame() -> None:
    """Unconditional, one per call, and it says the model has stopped writing.

    Unconditional is the point. The flush used to be gated on the argument size
    having MOVED since the last frame, which is a throttle's rule rather than a
    lifecycle one: this single-delta script reports its size on the first
    announcement, so under the old gate the stream end emitted nothing at all —
    and a viewer had no way to learn that dictation was over. The frame is also
    the one the UI keys the `queued` state off, so missing it is not a missed
    nicety, it is the reported stuck row.
    """
    executed: list[str] = []
    events = await _events(
        AgentLoop(),
        _two_call_step(),
        [
            echo_tool(executed, name="wait"),
            echo_tool(executed, name="wake", concurrency="exclusive"),
        ],
    )

    for call_id, expected in ((WAIT_ID, 18), (WAKE_ID, 14)):
        frames = _frames(events, call_id)
        assert frames, f"{call_id} was announced"
        assert [f.dictation_complete for f in frames][-1] is True
        assert not any(
            f.dictation_complete for f in frames[:-1]
        ), "only the LAST frame may claim the dictation is over"
        assert _terminal(events, call_id).argument_bytes == expected

    # The terminal frame is emitted AT the stream end, before any execution:
    # that ordering is what lets the row stop claiming to compose while the
    # sibling runs, rather than after it finishes.
    terminal_index = max(
        index for index, event in enumerate(events) if isinstance(event, ToolCallComposeEvent)
    )
    first_start = next(
        index for index, event in enumerate(events) if isinstance(event, ToolExecutionStartEvent)
    )
    assert terminal_index < first_start


@pytest.mark.asyncio
async def test_the_terminal_frame_carries_the_size_the_throttle_swallowed() -> None:
    """A burst inside one throttle window still ends with the whole call.

    The rows that need this are the ones streaming fastest — a `write` dumping
    fourteen kilobytes in under a second arrives in a handful of deltas inside
    one window, so the ordinary frames are throttled away and only the terminal
    frame carries a truthful size. A row that settled at the first delta's
    count would under-report the call for as long as it stayed on screen.
    """
    executed: list[str] = []
    parts = ['{"text":"', "a" * 300, '"}']
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_write_1", name="write", args=parts[0]),
                tool_call_delta(0, args=parts[1]),
                tool_call_delta(0, args=parts[2]),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = await _events(AgentLoop(), stream, [echo_tool(executed, name="write")])

    assert _terminal(events, "call_write_1").argument_bytes == len("".join(parts))


@pytest.mark.asyncio
async def test_the_terminal_frame_repeats_the_placeholder_hand_off() -> None:
    """Invariant: the identity moves once, is announced, and the end says so too.

    A provider that withholds the call id until the end of its stream leaves the
    row keyed by an index-derived placeholder. The promotion frame announces the
    hand-off, and it is REPEATED on later frames precisely because the queue may
    drop the one that carried it (exceptional case: the compose slot is
    replace-in-place, per connection). The terminal frame is the last chance that
    announcement has, so it carries `supersedes_tool_call_id` as well — and
    applying it twice stays idempotent on the consumer side.
    """
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                # The name arrives BEFORE the id, which is the regime the whole
                # hand-off exists for: the row is announced under an
                # index-derived placeholder first.
                tool_call_delta(0, name="echo", args='{"text":'),
                tool_call_delta(0, id="call_real_1", args='"hi"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = await _events(AgentLoop(), stream, [echo_tool(executed)])

    frames = [event for event in events if isinstance(event, ToolCallComposeEvent)]
    # The first announcement is keyed by the placeholder and has nothing to hand
    # off; every frame FROM the promotion onward — the terminal one included —
    # names the key it supersedes, so the announcement survives the loss of any
    # one frame.
    assert frames[0].supersedes_tool_call_id is None
    assert frames[0].tool_call_id.startswith("compose:")
    promoted = frames[1:]
    assert len(promoted) >= 2, f"promotion and terminal frames expected: {frames!r}"
    assert all(frame.tool_call_id == "call_real_1" for frame in promoted)
    assert all(frame.supersedes_tool_call_id == frames[0].tool_call_id for frame in promoted)
    assert promoted[-1].dictation_complete


# ---------------------------------------------------------------------------
# F2 — the never-run ending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_planning_failure_settles_its_announcement_with_the_reason() -> None:
    """Unknown tool: announced, judged, and given a compose ending — and no END.

    Both halves are asserted together on purpose. The end event is what the API
    server pairs tool records by, so it must stay withheld; without a compose
    ending, though, the row announcing the call had nothing to act on and stayed
    `composing…` until the turn died, where it was labelled `interrupted` for a
    call that was never interrupted.
    """
    executed: list[str] = []
    events = await _events(
        AgentLoop(),
        _two_call_step(),
        [echo_tool(executed, name="wait")],  # `wake` is not offered
    )

    terminal = _terminal(events, WAKE_ID)
    assert terminal.not_run_reason == "Tool not found: wake"
    assert terminal.argument_bytes == len('{"text":"30m"}')
    # Two terminal frames, in this order: the stream end (no verdict — the call
    # was merely queued) and then the plan's judgement. The LAST is the one a
    # consumer acts on, and the earlier one is why a viewer never had to wait
    # for the batch to learn that dictation was over.
    assert [frame.not_run_reason for frame in _terminal_frames(events, WAKE_ID)] == [
        None,
        "Tool not found: wake",
    ]
    started, ended = _lifecycle(events, WAKE_ID)
    assert started == [] and ended == [], "a never-run call gets no tool-record events"
    assert "wake" not in executed

    # The sibling is untouched: its own lifecycle is exactly what it was.
    sibling_started, sibling_ended = _lifecycle(events, WAIT_ID)
    assert len(sibling_started) == 1 and len(sibling_ended) == 1


def _strict_tool(executed: list[str]) -> AgentTool:
    """An `echo` whose `text` argument is REQUIRED, so `{}` fails validation.

    ``echo_tool`` declares no required properties, which means an empty object
    validates cleanly and the call runs — the exact opposite of what the
    invalid-arguments arm needs to exercise.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append("strict")
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="strict", content=[TextContent(text="ran")]
        )

    return AgentTool(
        name="strict",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        execute=execute,
    )


@pytest.mark.asyncio
async def test_an_invalid_arguments_failure_settles_with_the_validation_message() -> None:
    """The reason is the harness's own words, not a second vocabulary.

    A row that says `not run` while the transcript says `missing required
    argument 'text'` is two accounts of one verdict, and the second is the
    actionable one.
    """
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_strict_1", name="strict", args="{}"),
                tool_call_delta(1, id=WAKE_ID, name="wake", args='{"text":"30m"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = await _events(
        AgentLoop(),
        stream,
        [_strict_tool(executed), echo_tool(executed, name="wake", concurrency="exclusive")],
    )

    reason = _terminal(events, "call_strict_1").not_run_reason or ""
    assert reason.startswith("Invalid arguments:")
    assert "missing required argument 'text'" in reason
    started, ended = _lifecycle(events, "call_strict_1")
    assert started == [] and ended == []
    # The call queued behind it still executes normally, exactly as before.
    assert executed == ["wake"]


@pytest.mark.asyncio
async def test_the_never_run_reason_is_bounded_to_one_clipped_line() -> None:
    """It rides the wire and the seed, both of which budget text.

    The synthetic result this is drawn from may hold an argument dump or a
    validation complaint of any size; the frame carries the first line, clipped,
    and says so with an ellipsis rather than trailing off mid-sentence.
    """
    executed: list[str] = []
    long_name = "tool_" + "x" * 400
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_long_1", name=long_name, args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = await _events(AgentLoop(), stream, [echo_tool(executed)])

    reason = _terminal(events, "call_long_1").not_run_reason or ""
    assert len(reason) == NOT_RUN_REASON_MAX_CHARS
    assert reason.endswith("…")
    assert "\n" not in reason


@pytest.mark.asyncio
async def test_a_duplicate_id_settles_the_loser_without_closing_the_winners_record() -> None:
    """Two calls, one id: the loser is announced as never-run, the winner runs.

    The loser's compose ending is the only place the duplicate is visible to a
    ledger — the tool-record surface deliberately says nothing about it (an end
    for the shared id would close the winner's record early). The row the two
    share is then adopted by the winner's real start, which is the consumer's
    half of this and is covered in `tests/unit/tui/test_queued_tool_rows.py`.
    """
    executed: list[str] = []
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_dup_1", name="echo", args='{"text":"first"}'),
                tool_call_delta(1, id="call_dup_1", name="echo", args='{"text":"second"}'),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    events = await _events(AgentLoop(), stream, [echo_tool(executed)])

    verdicts = [f for f in _frames(events, "call_dup_1") if f.not_run_reason]
    assert len(verdicts) == 1
    assert verdicts[0].not_run_reason == "Duplicate call id 'call_dup_1' skipped."
    started, ended = _lifecycle(events, "call_dup_1")
    assert len(started) == 1 and len(ended) == 1, "the winner runs exactly once"
    assert executed == ["echo"]


@pytest.mark.asyncio
async def test_a_steering_skip_settles_every_remaining_announcement() -> None:
    """The batch's remaining calls are dropped OUTSIDE `_execute_batch`.

    That site appends synthetic results and breaks out of the loop, so it never
    reaches `park` — and before this, a call dropped there had no ending on any
    surface: its row sat `composing…` and the user had to infer the skip from
    the steering notice. Every dropped call gets the ending, once, and none of
    them may be reported as executing.
    """
    executed: list[str] = []
    steering = {"queued": False, "drained": False}

    async def first_execute(tool_call_id, args, signal, on_update, context):
        executed.append("a")
        steering["queued"] = True
        return ToolResult(tool_call_id=tool_call_id, tool_name="a", content=[TextContent(text="a")])

    tool_a = AgentTool(
        name="a",
        parameters={"type": "object"},
        concurrency="exclusive",
        execute=first_execute,
    )
    stream = ScriptedStream(
        [
            [
                tool_call_delta(0, id="call_a", name="a", args="{}"),
                tool_call_delta(1, id="call_b", name="b", args="{}"),
                tool_call_delta(2, id="call_c", name="c", args="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    async def get_steering():
        if steering["queued"] and not steering["drained"]:
            steering["drained"] = True
            return [Message.user("stop that")]
        return []

    context = LoopContext(
        system_blocks=["sys"],
        tools=[tool_a, echo_tool(executed, name="b"), echo_tool(executed, name="c")],
    )
    config = make_config(
        stream,
        interrupt_mode="immediate",
        get_steering_messages=get_steering,
        has_steering_messages=lambda: steering["queued"] and not steering["drained"],
    )
    events: list[Any] = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        events.append(event)

    assert executed == ["a"]
    for call_id in ("call_b", "call_c"):
        terminal = _terminal(events, call_id)
        assert terminal.not_run_reason == "Tool call skipped: interrupted by steering."
        # ONE verdict per call. The skip describes the whole remaining tail, so
        # an emitter that announced it per member of that tail would hand every
        # viewer the same fact twice — and a seed would carry it twice.
        assert len([f for f in _terminal_frames(events, call_id) if f.not_run_reason]) == 1
        started, ended = _lifecycle(events, call_id)
        assert started == [] and ended == []


@pytest.mark.asyncio
async def test_a_call_that_ran_gets_no_not_run_reason() -> None:
    """The negative control: the ordinary path is unchanged.

    Without this, "every call gets a reason" would pass as easily as the
    intended behaviour, and a stray reason on a running call would settle a live
    row in the UI.
    """
    executed: list[str] = []
    events = await _events(
        AgentLoop(),
        _two_call_step(),
        [
            echo_tool(executed, name="wait"),
            echo_tool(executed, name="wake", concurrency="exclusive"),
        ],
    )

    assert all(f.not_run_reason is None for f in _frames(events, WAIT_ID))
    assert all(f.not_run_reason is None for f in _frames(events, WAKE_ID))
    assert executed == ["wait", "wake"]
