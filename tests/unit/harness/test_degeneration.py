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
   and the genuine empty-length path is untouched.

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
import json
from typing import Any

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
from local_operator.harness.rows import harness_chrome_prompts, is_harness_chrome
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
)
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


def _feed(text: str, *, channel: str = "reasoning", size: int = 2):
    guard = StreamDegenerationGuard(channel)  # type: ignore[arg-type]
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
    out: list[str] = []
    i = 0
    while sum(len(w) + 1 for w in out) < chars:
        out.append(words[(i * 7 + i // len(words)) % len(words)])
        if i % 13 == 12:
            out.append(f"(step {i}).")
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


def test_degenerate_visible_text_trips_too() -> None:
    trip = _feed("Here is the plan.\n" + "I will now click the button. " * 2_000, channel="text")
    assert trip is not None
    assert trip.channel == "text"
    assert trip.period == len("I will now click the button. ")


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
    for channel in ("reasoning", "text"):
        assert _feed(text, channel=channel) is None  # type: ignore[arg-type]
        # Delta size must not matter: one-character and large fragments too.
        assert _feed(text, channel=channel, size=1) is None  # type: ignore[arg-type]
        assert _feed(text, channel=channel, size=4_000) is None  # type: ignore[arg-type]


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

    def __init__(self, turns: list[list[StreamEvent] | str], *, channel: str = "reasoning"):
        self.turns = turns
        self.channel = channel
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
            make = StreamReasoningDelta if self.channel == "reasoning" else StreamTextDelta
            try:
                for delta in _deltas(_PREAMBLE):
                    self.streamed[index] += len(delta)
                    yield make(delta=delta)
                unit = _TRACE_UNITS[0]
                while True:
                    for delta in _deltas(unit):
                        self.streamed[index] += len(delta)
                        if self.streamed[index] > self.BUDGET_CHARS:
                            # The provider's own limit: reaching it means the
                            # guard did NOT cut the stream.
                            yield StreamEndEvent(stop_reason="length")
                            return
                        yield make(delta=delta)
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


@pytest.mark.asyncio
async def test_degenerate_visible_text_is_cut_and_recovered() -> None:
    provider = LoopingProvider(
        ["loop", [StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]],
        channel="text",
    )
    context, events, end, notices = await _run(provider)
    assert provider.cancelled == [True]
    assert provider.streamed[0] < LoopingProvider.BUDGET_CHARS // 8
    assert len(_steers(context)) == 1
    # The looping prose is not kept: neither in the context nor replayed.
    assert not any("Hmm! Hmm, hmm" in m.text for m in provider.requests[1].messages)
    assert end.error is None
    assert _budget_notices(notices) == []


@pytest.mark.asyncio
async def test_the_recovery_cap_ends_the_run_named_for_the_degeneration() -> None:
    provider = LoopingProvider(["loop"] * (MAX_DEGENERATE_STREAM_RECOVERIES + 1))
    context, events, end, notices = await _run(provider)

    assert len(provider.requests) == MAX_DEGENERATE_STREAM_RECOVERIES + 1
    assert provider.cancelled == [True] * (MAX_DEGENERATE_STREAM_RECOVERIES + 1)
    assert all(n < LoopingProvider.BUDGET_CHARS // 8 for n in provider.streamed)
    assert len(_steers(context)) == MAX_DEGENERATE_STREAM_RECOVERIES
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
