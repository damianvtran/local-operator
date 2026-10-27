"""The in-stream degeneration guard: what it stops, what it must never stop.

Two layers, in two halves of this file.

1. ``StreamDegenerationGuard`` on its own, against a corpus built to look like
   legitimate long output that REPEATS STRUCTURE (a long markdown table, a
   numbered list of near-identical lines, varied prose, code) and against the
   measured degenerate shape. The corpus half is the reason the guard can be
   default-on: a false trip there would cut a real answer.
2. The REAL loop (``AgentLoop.run``) with a provider stream that keeps producing
   until it is cancelled, the way a looping model does. These pin the behaviour
   the guard exists for: the stream is cut long before the budget, the turn
   recovers through a steer rather than a replay, the recovery is capped, the
   budget-exhausted copy is never said about a stream that was stopped early,
   and the genuine empty-length path is untouched. Visible text is NOT guarded
   (review round 1, M1): every periodic answer shape the review found tripping
   the earlier text guard is delivered intact here.
3. What the cut leaves behind: every surface that folds the cut message labels
   it neutrally rather than as an output-limit/budget stop (review m2), and the
   cut call's spend is recorded as an ESTIMATE, never as unknown or billed
   (review m3).

THE MEASURED SHAPE. ``_TRACE_UNITS`` are the three repeating units recovered
from the 2026-10-26 OSWorld trace (``osworld-task_063``, session
``ep-de966d3c95c2``): three consecutive model calls each spent the full
131,072-token output budget, streaming 225-253 K characters of reasoning that
had become one of these units repeated verbatim (periods 36, 333 and 450), in
deltas averaging 1.9 characters. Recovered by replaying the recorded
``reasoning_delta`` events through the guard; the units are taken from the
trace's own tails, not composed by hand.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, cast

import pytest

from local_operator.harness.degeneration import (
    CHECK_STRIDE_CHARS,
    MAX_PERIOD_CHARS,
    MIN_REPEATS,
    WINDOW_CHARS,
    StreamDegenerationGuard,
)
from local_operator.harness.loop import (
    DEGENERATE_STREAM_RECOVERY_PROMPT,
    LENGTH_STOP_RECOVERY_PROMPT,
    MAX_DEGENERATE_STREAM_RECOVERIES,
    NO_PROGRESS_RECOVERY_PROMPT,
    NO_PROGRESS_WARNING,
    AgentLoop,
    LoopContext,
)
from local_operator.harness.rows import (
    assistant_stop_notice,
    harness_chrome_prompts,
    is_harness_chrome,
)
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentTool,
    ChatRequest,
    Message,
    MessageEndEvent,
    ModelSpec,
    NoticeEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamEvent,
    StreamReasoningDelta,
    StreamTextDelta,
    TextContent,
    ToolResult,
    Usage,
)
from local_operator.mobile.projection import ProjectionFold, fold_messages_to_entries
from local_operator.mobile.types import SessionProjection
from tests.unit.harness.test_loop import ScriptedStream, make_config, tool_call_delta

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

#: The three repeating units of the measured trace (see the module docstring).
_TRACE_UNITS = (
    ("mm. Hmm.\n\nHmm! Hmm, hmm: hmm, hmm, h"),
    (
        ".\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm.\n\nHmm! Let me go.\n\nHmm, hmm. Hmm! Hmm, hm"
        "m: hmm, hmm, hmm. Hmm.\n\nHmm! Let me go.\n\nHmm, hmm. Hmm! Hmm, hmm: hmm, hmm, hmm."
        " Hmm.\n\nHmm! Let me go.\n\nHmm! Let me go.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm.\n"
        "\nHmm! Let me go.\n\nHmm, hmm. Hmm! Hmm, hmm: hmm, hmm. Hmm.\n\nHmm! Let me go.\n\nH"
        "mm! Hmm, OK, going now"
    ),
    (
        "mm. Hmm.\n\nHmm! Let me go.\n\nHmm! Hmm, hmm: hmm, hmm. Hmm, hmm, hmm. Hmm.\n\nHmm! "
        "Hmm, hmm, hmm: hmm, hmm. Hmm.\n\nHmm! Let me go.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hm"
        "m, hmm. Hmm.\n\nHmm! Let me go.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm. Hmm.\n\nHmm! L"
        "et me go.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm, hmm. Hmm.\n\nHmm! Let me go.\n\nHmm!"
        " Hmm, hmm, hmm: hmm, hmm. Hmm.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm, hmm. Hmm.\n\nHm"
        "m! Let me go.\n\nHmm! Hmm, hmm: hmm, hmm, hmm. Hmm, hmm, h"
    ),
)

#: The measured stream's prefix shape: a few KB of genuine reasoning before the
#: loop sets in (the trace's loops began 3.3-5.5 K characters in).
_PREAMBLE = (
    "Hmm, the Siri bubble is now SELECTED (shape mode, Drawing Tools) and its box is at "
    "(395,183)-(540,268) — that's roughly the same position (not moved?). The handles sit "
    "at (395,183), (467,183), (540,183), (395,268), (467,268), (540,268). "
) * 16


def _deltas(text: str, size: int = 2) -> list[str]:
    """Split like the measured stream did: ~2-character fragments."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _feed(text: str, *, size: int = 2):
    guard = StreamDegenerationGuard()
    for delta in _deltas(text, size):
        trip = guard.feed(delta)
        if trip is not None:
            return trip
    return None


def _markdown_table(rows: int) -> str:
    header = "| # | host | region | p50 ms | p99 ms | status |\n|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {i} | web-{i:04d}.internal | eu-west-{i % 3} | {17 + i % 13} | "
        f"{140 + (i * 7) % 97} | {'ok' if i % 11 else 'degraded'} |\n"
        for i in range(rows)
    )
    return header + body


def _numbered_near_duplicates(lines: int) -> str:
    # The hardest legitimate case: every line is identical except its number.
    return "".join(
        f"{i}. Check the configuration file and restart the service.\n" for i in range(lines)
    )


def _code(functions: int) -> str:
    return "".join(
        f"def handler_{i}(request):\n"
        f"    value = request.get('field_{i}')\n"
        f"    if value is None:\n"
        f"        return {{'error': 'missing field_{i}'}}\n"
        f"    return {{'ok': True, 'value': value}}\n\n"
        for i in range(functions)
    )


def _varied_reasoning(chars: int) -> str:
    # Varied prose with recurring vocabulary, deterministic, no verbatim period.
    words = (
        "the screenshot shows the dialog is still open so I should check whether "
        "the selection moved before clicking again because the previous attempt "
        "landed on the header instead of the body and the crop handles are visible"
    ).split()
    #
    # A RUNNING length, never ``sum(...)`` over ``out`` per word. This runs at
    # IMPORT -- it builds a ``parametrize`` case -- so it runs during collection
    # in every xdist worker. The re-summed form was quadratic: 15 s for 200 K
    # chars on an idle laptop, and CI shards holding this file sat 292-405 s
    # before their first test against a fleet median of 168 s (PR #2993 CI
    # investigation, round 3). That was enough to push a shard past its job
    # timeout. The text is byte-identical to the old loop's: the same words,
    # the same stop condition.
    out: list[str] = []
    length = 0
    i = 0
    while length < chars:
        word = words[(i * 7 + i // len(words)) % len(words)]
        out.append(word)
        length += len(word) + 1
        if i % 13 == 12:
            step = f"(step {i})."
            out.append(step)
            length += len(step) + 1
        i += 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# The guard alone
# ---------------------------------------------------------------------------


def test_the_thresholds_are_derived_from_each_other() -> None:
    """A trip means MIN_REPEATS verbatim copies at the longest period, by construction."""
    assert WINDOW_CHARS == MAX_PERIOD_CHARS * MIN_REPEATS
    assert MAX_PERIOD_CHARS >= max(len(unit) for unit in _TRACE_UNITS)
    assert CHECK_STRIDE_CHARS < WINDOW_CHARS


@pytest.mark.parametrize("unit", _TRACE_UNITS, ids=lambda u: f"period-{len(u)}")
def test_the_measured_loop_trips_early(unit: str) -> None:
    """Each measured unit trips within the window plus one stride of loop.

    The trace streamed 225-253 K characters per turn; the guard must stop it
    after the loop has filled one window, i.e. a few percent of that.
    """
    looped = unit * (200_000 // len(unit))
    trip = _feed(_PREAMBLE + looped)
    assert trip is not None
    assert trip.channel == "reasoning"
    assert trip.period == len(unit)
    assert trip.repeats >= MIN_REPEATS
    assert trip.chars <= len(_PREAMBLE) + WINDOW_CHARS + 2 * CHECK_STRIDE_CHARS
    assert trip.chars < 225_000 // 10


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_markdown_table(2_000), id="markdown-table"),
        pytest.param(_numbered_near_duplicates(4_000), id="numbered-near-duplicates"),
        pytest.param(_code(600), id="code"),
        pytest.param(_varied_reasoning(200_000), id="varied-reasoning"),
        pytest.param(
            "".join(f"- item {i:05d}: ok\n" for i in range(12_000)), id="short-numbered-lines"
        ),
        pytest.param("x" * (WINDOW_CHARS - 1), id="pure-repeat-below-window"),
    ],
)
def test_legitimate_long_output_never_trips(text: str) -> None:
    """Repeated STRUCTURE with varying content is not a loop.

    Each case is far longer than the window, so the guard checks it dozens of
    times; one differing character inside the window is all it takes to stay
    quiet, and every row/line/function here differs by at least its number.
    The last case is the size floor: pure repetition shorter than a window
    never trips, because nothing below the window is ever checked.
    """
    assert len(text) >= WINDOW_CHARS - 1
    assert _feed(text) is None
    # Delta size must not matter: one-character and large fragments too.
    assert _feed(text, size=1) is None
    assert _feed(text, size=4_000) is None


def test_the_never_trip_corpus_builds_in_linear_time() -> None:
    """The corpus above is built at IMPORT, so what it costs is a collection cost.

    ``parametrize`` arguments are evaluated when the module loads, which means in
    every xdist worker of whichever CI shard holds this file, under coverage
    tracing. The quadratic generator this replaced cost 15 s per import on a
    laptop and kept a CI shard collecting for 402 s (fleet median 168 s) until
    the job timeout cancelled it (PR #2993 CI investigation, round 3).

    COUNTED, NOT TIMED: a wall-clock bound flakes on a loaded runner, while the
    number of profiled calls is deterministic. Ten times the text must cost
    about ten times the calls. The quadratic form costs about a hundred times,
    so a ceiling of 20 separates the two with room on both sides.
    """
    import sys

    def calls(chars: int) -> int:
        count = 0

        def profile(_frame: Any, _event: str, _arg: Any) -> None:
            nonlocal count
            count += 1

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            _varied_reasoning(chars)
        finally:
            sys.setprofile(previous)
        return count

    small, large = calls(1_000), calls(10_000)
    assert large < 20 * small, (small, large)


def test_a_repeat_longer_than_the_period_ceiling_does_not_trip() -> None:
    """A block re-emitted verbatim but longer than MAX_PERIOD_CHARS is out of scope."""
    block = _varied_reasoning(MAX_PERIOD_CHARS + 200)
    assert _feed(block * 60) is None


def test_the_guard_fires_once() -> None:
    guard = StreamDegenerationGuard("reasoning")
    trips = [guard.feed(delta) for delta in _deltas("ab" * 40_000)]
    assert sum(trip is not None for trip in trips) == 1


def test_the_payload_carries_no_model_text() -> None:
    trip = _feed("loop! " * 10_000)
    assert trip is not None
    # The sample is the window's last period, i.e. SOME rotation of the unit.
    assert trip.sample in ("loop! " * 2)
    assert set(trip.payload()) == {"channel", "period", "repeats", "chars"}
    json.dumps(trip.payload())


# ---------------------------------------------------------------------------
# The real loop
# ---------------------------------------------------------------------------


def _laddered() -> ModelSpec:
    # A ladder with cheaper rungs, so a guard that fell through to the
    # ordinary length arm would show up as an effort retreat.
    return ModelSpec(
        provider="test",
        model_id="m",
        reasoning_efforts=("low", "medium", "high"),
        reasoning_effort="high",
    )


class LoopingProvider:
    """A provider whose looping turns never end on their own.

    Each scripted turn is either a list of events (replayed, then the stream
    ends) or the string ``"loop"``: the measured preamble, then the trace's
    period-36 unit in 2-character deltas FOREVER, the way a degenerate model
    streams until its budget is spent. Only cancellation ends it, so a test
    that finishes at all proves the loop cut the stream; ``streamed`` records
    how far each looping turn got before the cut.
    """

    #: The budget the measured turns spent, in characters of reasoning.
    BUDGET_CHARS = 225_423

    def __init__(self, turns: list[list[StreamEvent] | str]):
        self.turns = turns
        self.requests: list[ChatRequest] = []
        self.streamed: list[int] = []
        self.cancelled: list[bool] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def scripted():
            for event in turn:
                yield event

        async def looping():
            index = len(self.streamed)
            self.streamed.append(0)
            self.cancelled.append(False)
            try:
                for delta in _deltas(_PREAMBLE):
                    self.streamed[index] += len(delta)
                    yield StreamReasoningDelta(delta=delta)
                unit = _TRACE_UNITS[0]
                while True:
                    for delta in _deltas(unit):
                        self.streamed[index] += len(delta)
                        if self.streamed[index] > self.BUDGET_CHARS:
                            # The provider's own limit: reaching it means the
                            # guard did NOT cut the stream.
                            yield StreamEndEvent(stop_reason="length")
                            return
                        yield StreamReasoningDelta(delta=delta)
                        # A real socket read yields to the event loop; without
                        # this the pump would drain the generator in one step.
                        await asyncio.sleep(0)
            except asyncio.CancelledError:
                self.cancelled[index] = True
                raise

        return looping() if turn == "loop" else scripted()


async def _run(provider, *, tools=None, model=None):
    context = LoopContext(tools=tools or [])
    events = [
        event
        async for event in AgentLoop().run(
            [Message.user("finish the slide")],
            context,
            make_config(provider, model=model or _laddered()),
            None,
        )
    ]
    end = next(event for event in reversed(events) if isinstance(event, AgentEndEvent))
    notices = [event for event in events if isinstance(event, NoticeEvent)]
    return context, events, end, notices


def _assistant_ends(events: list[Any]) -> list[Message]:
    """The assistant messages the run settled, in order (``message_end`` events)."""
    return [
        e.message
        for e in events
        if isinstance(e, MessageEndEvent)
        and isinstance(e.message, Message)
        and e.message.role == "assistant"
    ]


def _degenerate_record(message: Message) -> dict[str, Any] | None:
    return (message.provider_payload or {}).get("degenerate_stream")


def _budget_notices(notices: list[NoticeEvent]) -> list[NoticeEvent]:
    return [n for n in notices if (n.code or "").startswith("tui.notices.budget_exhausted")]


def _steers(context: LoopContext) -> list[Message]:
    return [
        m
        for m in context.messages
        if isinstance(m, Message) and m.text == DEGENERATE_STREAM_RECOVERY_PROMPT
    ]


@pytest.mark.asyncio
async def test_the_measured_shape_is_cut_early_and_recovers_with_a_steer() -> None:
    provider = LoopingProvider(
        [
            "loop",
            [
                StreamTextDelta(delta="Done: the bubble is moved."),
                StreamEndEvent(stop_reason="stop"),
            ],
        ]
    )
    context, events, end, notices = await _run(provider)

    # Cut, and cut early: a small fraction of the budget the trace spent.
    assert provider.cancelled == [True]
    assert provider.streamed[0] < LoopingProvider.BUDGET_CHARS // 8
    # Recovered: one more request, at the SAME effort (no retreat), carrying
    # the steer and not the looping turn.
    assert len(provider.requests) == 2
    assert [r.model.reasoning_effort for r in provider.requests] == ["high", "high"]
    retry = provider.requests[1].messages
    assert retry[-1].text == DEGENERATE_STREAM_RECOVERY_PROMPT
    assert not any(isinstance(m, Message) and m.role == "assistant" and not m.text for m in retry)
    assert end.error is None and not end.aborted and end.cut_off_cause == ""
    last = context.messages[-1]
    assert isinstance(last, Message) and last.text == "Done: the bubble is moved."
    # Diagnosable: the cut message's end event names the degeneration.
    cut = next(m for m in _assistant_ends(events) if _degenerate_record(m))
    record = _degenerate_record(cut)
    assert record is not None
    assert record["channel"] == "reasoning" and record["period"] == len(_TRACE_UNITS[0])
    assert cut.stop_reason == "length"
    # And the false copy is never said.
    assert _budget_notices(notices) == []
    # The reasoning really streamed to the surfaces before the cut.
    assert any(isinstance(e, ReasoningDeltaEvent) for e in events)


#: Legitimate VISIBLE answers that are exactly periodic for far longer than the
#: guard's window -- every shape agent review round 1 on PR #2993 measured tripping
#: the (since removed) text-channel guard. Each is something a user can ask for.
_PERIODIC_ANSWERS = {
    "identical-table-rows": "| a | b |\n|---|---|\n" + "| 0 | 0 |\n" * 3_000,
    "csv-empty-fields": "a,b,c\n" + ",,\n" * 6_000,
    "csv-identical-rows": "flag,value\n" + "1,0\n" * 5_000,
    "base64-zero-bytes": base64.b64encode(bytes(24_000)).decode(),
    "base64-zero-bytes-mime": base64.encodebytes(bytes(24_000)).decode(),
    "hex-zero-bytes": "00" * 12_000,
    "json-zero-array": json.dumps([0] * 8_000),
    "json-null-objects": json.dumps([{"a": None}] * 2_000),
    "box-drawing-rule": "\u2500" * 20_000,
    "ascii-frame-blank-rows": "+" + "-" * 78 + "+\n" + ("|" + " " * 78 + "|\n") * 300,
    "spaces": " " * 20_000,
    "newlines": "\n" * 20_000,
    "dots": "." * 20_000,
    "svg-path": '<path d="M0 0 ' + "l1 0 " * 5_000 + '"/>',
    "repeated-log-line": "WARN retrying connection\n" * 1_000,
    "ok-lines": "ok\n" * 5_000,
    "cjk-repeated": "\u6211\u4e0d\u77e5\u9053\u3002" * 3_000,
    "emoji": "\U0001f600" * 12_000,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", list(_PERIODIC_ANSWERS.values()), ids=list(_PERIODIC_ANSWERS))
async def test_a_periodic_visible_answer_is_delivered_intact(answer: str) -> None:
    """Visible text is NOT guarded: a long verbatim-periodic answer completes.

    Through the real loop, at the 7-character delta granularity the review used:
    one request, no cut, no steer, no error, and the final assistant message is
    the answer byte for byte. With a text-channel guard each of these was thrown
    away three times and the run ended "Degenerate output" (review M1).
    """
    deltas = [StreamTextDelta(delta=answer[i : i + 7]) for i in range(0, len(answer), 7)]
    provider = LoopingProvider([[*deltas, StreamEndEvent(stop_reason="stop")]])
    context, events, end, notices = await _run(provider)
    assert len(provider.requests) == 1
    assert end.error is None and not end.aborted
    assert _steers(context) == []
    assert not any(_degenerate_record(m) for m in _assistant_ends(events))
    last = context.messages[-1]
    assert isinstance(last, Message) and last.role == "assistant"
    assert last.text == answer and last.stop_reason == "stop"


@pytest.mark.asyncio
async def test_the_recovery_cap_ends_the_run_named_for_the_degeneration() -> None:
    provider = LoopingProvider(["loop"] * (MAX_DEGENERATE_STREAM_RECOVERIES + 1))
    context, events, end, notices = await _run(provider)

    assert len(provider.requests) == MAX_DEGENERATE_STREAM_RECOVERIES + 1
    assert provider.cancelled == [True] * (MAX_DEGENERATE_STREAM_RECOVERIES + 1)
    assert all(n < LoopingProvider.BUDGET_CHARS // 8 for n in provider.streamed)
    # ONE steer, never a run of them (review n2): every re-ask ends with exactly
    # one, and the run leaves exactly one behind for the next user turn to read.
    for retry in provider.requests[1:]:
        assert retry.messages[-1].text == DEGENERATE_STREAM_RECOVERY_PROMPT
        assert sum(m.text == DEGENERATE_STREAM_RECOVERY_PROMPT for m in retry.messages) == 1
    assert len(_steers(context)) == 1
    last = context.messages[-1]
    assert isinstance(last, Message) and last.text == DEGENERATE_STREAM_RECOVERY_PROMPT
    # Named for what happened -- never the output-limit sentence or cause.
    assert end.error is not None and end.error.startswith("Degenerate output:")
    assert "verbatim loop" in end.error
    assert "output limit" not in end.error and "budget" not in end.error
    assert end.cut_off_cause == ""
    assert _budget_notices(notices) == []


@pytest.mark.asyncio
async def test_the_genuine_empty_length_path_still_speaks_and_retreats() -> None:
    """Control: a turn the PROVIDER cut at its limit is not the guard's.

    Same ordering, same copy, same retreat as before this change -- which is
    also the positive control for the "no budget notice" assertions above.
    """
    provider = ScriptedStream(
        [
            [StreamReasoningDelta(delta="thinking " * 50), StreamEndEvent(stop_reason="length")],
            [StreamTextDelta(delta="answer"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context, events, end, notices = await _run(provider)
    assert [r.model.reasoning_effort for r in provider.requests] == ["high", "medium"]
    budget = _budget_notices(notices)
    assert [n.code for n in budget] == ["tui.notices.budget_exhausted_retry"]
    assert "spent its whole output budget" in budget[0].text
    assert _steers(context) == []
    assert not any(_degenerate_record(m) for m in _assistant_ends(events))


@pytest.mark.asyncio
async def test_a_user_abort_still_wins_over_the_guard() -> None:
    """Abort and trip on the SAME delta: the turn is the user's stop, not a re-ask.

    The trip point is computed from the provider's deterministic delta sequence,
    and the user's signal is aborted while the loop is suspended on exactly the
    ``reasoning_delta`` whose feed trips the guard -- so both stops are live when
    the stream ends, which is the race the ordering in ``_model_turn`` decides.
    """
    trip = _feed(_PREAMBLE + _TRACE_UNITS[0] * 2_000)
    assert trip is not None
    signal = AbortSignal()
    provider = LoopingProvider(["loop", "loop"])
    context = LoopContext()
    events = []
    seen = 0
    async for event in AgentLoop().run(
        [Message.user("go")], context, make_config(provider, model=_laddered()), signal
    ):
        events.append(event)
        if isinstance(event, ReasoningDeltaEvent):
            seen += len(event.delta)
            if seen >= trip.chars and not signal.aborted:
                signal.abort("interrupted")
    end = events[-1]
    assert isinstance(end, AgentEndEvent) and end.aborted
    assert len(provider.requests) == 1
    assert _steers(context) == []
    cut = _assistant_ends(events)[0]
    assert cut.stop_reason == "aborted"
    # The guard DID trip on that delta: its record is still on the message.
    assert _degenerate_record(cut)


@pytest.mark.asyncio
async def test_the_steer_and_the_no_progress_guard_do_not_double_fire() -> None:
    """The two repetition guards watch different things and stay independent.

    The run first repeats one tool call past ``NO_PROGRESS_WARNING`` (the
    no-progress steer fires once), then its next turn's reasoning loops (the
    degeneration steer fires once). Each guard must fire exactly once, on its
    own evidence: the degeneration arm must not count as a tool batch, and the
    no-progress guard must not re-fire because a turn was dropped.
    """
    executed: list[str] = []

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append(tool_call_id)
        return ToolResult(tool_call_id=tool_call_id, content=[TextContent(text="unchanged")])

    tool = AgentTool(name="wait", parameters={"type": "object"}, execute=execute)
    repeats = NO_PROGRESS_WARNING + 1
    turns: list[list[StreamEvent] | str] = [
        [
            tool_call_delta(0, id=f"call-{i}", name="wait", args='{"ms": 100}'),
            StreamEndEvent(stop_reason="tool_calls"),
        ]
        for i in range(repeats)
    ]
    turns.append("loop")
    turns.append([StreamTextDelta(delta="Finished."), StreamEndEvent(stop_reason="stop")])
    provider = LoopingProvider(turns)
    context, events, end, notices = await _run(provider, tools=[tool])

    texts = [m.text for m in context.messages if isinstance(m, Message) and m.role == "user"]
    assert texts.count(NO_PROGRESS_RECOVERY_PROMPT) == 1
    assert texts.count(DEGENERATE_STREAM_RECOVERY_PROMPT) == 1
    assert texts.count(LENGTH_STOP_RECOVERY_PROMPT) == 0
    assert len(executed) == repeats
    assert provider.cancelled == [True]
    assert end.error is None
    assert _budget_notices(notices) == []


def test_the_steer_is_harness_chrome() -> None:
    """Persisted as a user row, so every surface must recognise it as chrome."""
    assert DEGENERATE_STREAM_RECOVERY_PROMPT in harness_chrome_prompts()
    assert is_harness_chrome(DEGENERATE_STREAM_RECOVERY_PROMPT)
    assert is_harness_chrome("  " + DEGENERATE_STREAM_RECOVERY_PROMPT + "\n")
    assert not is_harness_chrome("why did the model fall into a loop?")


# ---------------------------------------------------------------------------
# 3. What the cut leaves behind: honest labels on every fold, and its spend.
# ---------------------------------------------------------------------------

#: The output-limit / budget wording every ordinary ``length`` arm uses, and
#: which is FALSE for a guard cut (the stream was stopped ~10 KB in precisely so
#: the budget was not spent).
_LIMIT_WORDS = ("output limit", "output budget", "whole output budget")


def _guard_cut(text: str = "") -> Message:
    """A cut message exactly as the loop settles it (stop reason and payload)."""
    message = Message(
        role="assistant",
        content=[TextContent(text=text)] if text else [],
        stop_reason="length",
        provider_payload={
            "degenerate_stream": {
                "channel": "reasoning",
                "period": 36,
                "repeats": 284,
                "chars": 14_336,
            }
        },
    )
    return message


@pytest.mark.parametrize("text", ["", "I will now click the button."])
def test_a_guard_cut_is_never_labelled_an_output_limit_stop(text: str) -> None:
    """Review m2 at the shared helper: neutral copy, warning tier, both arms."""
    notice = assistant_stop_notice(
        text=text,
        has_tool_calls=False,
        stop_reason="length",
        provider_payload=_guard_cut(text).provider_payload,
    )
    assert notice == ("interrupted", "warning")
    # Control: the same message WITHOUT the guard's record keeps today's copy.
    plain = assistant_stop_notice(
        text=text, has_tool_calls=False, stop_reason="length", provider_payload={}
    )
    assert plain is not None and any(w in plain[0] for w in _LIMIT_WORDS)


@pytest.mark.parametrize("text", ["", "I will now click the button."])
def test_the_phone_folds_say_nothing_false_about_a_guard_cut(text: str) -> None:
    """Review m2 on the phone: the history page AND the attach seed.

    The cut message is never made durable, so these folds reach it as a live
    row (an attached viewer's ``_live_history`` re-prepared mid-run). Both fold
    paths run the real projection code over the real message shape.
    """
    history = [Message.user("move the bubble"), _guard_cut(text)]
    page = fold_messages_to_entries(list(history))
    fold = ProjectionFold(SessionProjection(session_id="s", pid=1))
    fold.fold_history(list(history))
    for rows in (page, list(fold.projection.transcript)):
        notices = [row for row in rows if row.kind == "notice"]
        assert [row.text for row in notices] == ["interrupted"]
        assert not any(w in (row.text or "") for row in rows for w in _LIMIT_WORDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "I will now click the button."])
async def test_the_tui_replay_says_nothing_false_about_a_guard_cut(text: str) -> None:
    """Review m2 in the TUI: the real app's settled-row replay of a cut message."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.conftest import await_session_boot
    from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text

    session = FakeSession()
    session._history = [Message.user("move the bubble"), _guard_cut(text)]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await await_session_boot(app)
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)
    assert "move the bubble" in shown
    assert "interrupted" in shown
    assert not any(w in shown for w in _LIMIT_WORDS)


def test_a_cut_calls_estimate_is_labelled_and_never_a_receipt() -> None:
    """Review m3 at the estimator: the documented arithmetic and its labels."""
    from local_operator.model.configure import (
        _CUT_CHARS_PER_OUTPUT_TOKEN,
        _estimate_consumer_cut_usage,
    )
    from local_operator.providers.context import measure_request

    request = ChatRequest(
        model=ModelSpec(provider="anthropic", model_id="claude-opus-5"),
        system_blocks=["You are terse."],
        messages=[Message.user("move the bubble " * 200)],
    )
    estimate = _estimate_consumer_cut_usage(request, None, cut_chars=14_342)
    assert estimate.billing_basis == "estimated"
    assert estimate.usd_cost is None and estimate.estimated_usd_cost is None
    assert estimate.input_tokens == measure_request(request).tokens > 0
    assert estimate.output_tokens == 14_342 // _CUT_CHARS_PER_OUTPUT_TOKEN
    assert estimate.reasoning_tokens == estimate.output_tokens
    # A partial provider frame that arrived before the cut wins where it has a
    # count, and still never counts as a receipt.
    partial = Usage(input_tokens=52_000, output_tokens=10, usd_cost=0.5)
    merged = _estimate_consumer_cut_usage(request, partial, cut_chars=14_342)
    assert merged.input_tokens == 52_000
    assert merged.output_tokens == 14_342 // _CUT_CHARS_PER_OUTPUT_TOKEN
    assert merged.usd_cost is None and merged.billing_basis == "estimated"


@pytest.mark.asyncio
async def test_a_cut_runs_ledger_is_non_zero_and_marked_estimated(tmp_path) -> None:
    """Review m3 end to end: the REAL recorder under the REAL loop.

    ``SessionStreamFn._record_stream`` wraps a provider that loops forever; the
    guard cuts it, the recorder books the cut call, and the loop puts the same
    figure on the cut message. The analytics ledger must hold a NON-ZERO,
    priced row whose basis says ``estimated`` and whose ``usage_reported`` says
    no receipt arrived; the request-final observation (benchmark cost
    accounting) must say ``estimated`` coverage rather than ``missing``.
    """
    import sqlite3

    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore
    from local_operator.model.configure import SessionStreamFn

    store = AnalyticsStore(tmp_path / "a.db")
    recorder = reset_recorder_for_test(store)
    stream_fn = object.__new__(SessionStreamFn)
    stream_fn._session_id = "sess-cut"
    stream_fn._counts_as_child_request = False
    observed: list[dict[str, Any]] = []

    def notify(target: Any, **kwargs: Any) -> None:
        observed.append(kwargs)

    stream_fn._request_observation_target = lambda: object()  # type: ignore[method-assign]
    stream_fn._notify_request_final = notify  # type: ignore[method-assign]
    # The answer turn reports a real usage frame: the control row that must
    # stay a provider report (basis not ``estimated``, ``usage_reported`` 1).
    answer_usage = Usage(input_tokens=900, output_tokens=12)
    provider = LoopingProvider(
        [
            "loop",
            [
                StreamTextDelta(delta="Done."),
                StreamEndEvent(stop_reason="stop", usage=answer_usage),
            ],
        ]
    )

    def recorded(request: ChatRequest, signal: AbortSignal | None) -> Any:
        return stream_fn._record_stream(request, cast(Any, provider(request, signal)))

    model = ModelSpec(provider="anthropic", model_id="claude-opus-5")
    context, events, end, _notices = await _run(recorded, model=model)
    assert end.error is None and provider.cancelled == [True]

    cut = next(m for m in _assistant_ends(events) if _degenerate_record(m))
    assert cut.usage is not None and cut.usage.billing_basis == "estimated"
    assert cut.usage.output_tokens > 0 and cut.usage.input_tokens > 0
    assert cut.usage.usd_cost is None

    recorder.flush_for_test()
    with sqlite3.connect(tmp_path / "a.db") as db:
        rows = db.execute(
            "SELECT output_tokens, cost_micro, cost_known, billing_basis, usage_reported "
            "FROM calls ORDER BY ts_ms"
        ).fetchall()
    assert len(rows) == 2
    output_tokens, cost_micro, cost_known, basis, reported = rows[0]
    assert output_tokens == cut.usage.output_tokens
    assert cost_known == 1 and cost_micro > 0
    assert basis == "estimated" and reported == 0
    assert rows[1][3] != "estimated" and rows[1][4] == 1

    assert [o["usage_estimated"] for o in observed] == [True, False]
    assert observed[0]["usage"].billing_basis == "estimated"
