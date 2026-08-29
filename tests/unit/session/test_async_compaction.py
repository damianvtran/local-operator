"""The advisor-triggered compaction pass runs OFF the turn (BETA).

The advisor CALL was already off the critical path; the pass it authorises was
not. ``_run_compaction`` makes its own summarization provider call, and it was
awaited inline at every gate — invisible while a pass only ever fired at the
600k ceiling (the turn had to stop there anyway), and a visible mid-conversation
stall once the advisor started firing passes early and often.

So an advisor-triggered pass now summarizes in the background and COMMITS at
the next safe boundary. What is pinned here is the part that makes that safe:

- the turn is not blocked by the summarization call;
- the pass lands on a LATER boundary, with the conversation having continued;
- a pass whose prefix moved underneath it is DISCARDED, never misapplied;
- history added while the pass ran survives it;
- the ceiling path and manual ``/compact`` stay SYNCHRONOUS — the safety net
  must not become async-only;
- a failed background pass leaves the session exactly as if none had run.

Real compaction throughout, as in ``test_compaction_advisor.py``: only the
ruler and the provider stream are substituted, never the gate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.compaction.advisor import CompactionHint
from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    CompactionEndEvent,
    CompactionStartEvent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

BIG_MODEL = ModelSpec(provider="test", model_id="opus-like", context_window=1_000_000)
KEEP_RECENT = 40

#: How long the substituted summarization call blocks. Large enough that an
#: INLINE pass is unmistakable in a latency measurement (the assertions use a
#: fraction of it), small enough to keep the suite fast.
SUMMARY_DELAY_S = 0.5


class SlowSummaryStream:
    """A working turn, an advisor answer, and a SLOW summarization call.

    The delay is the whole point: it stands in for the 20-50 s a real
    summarization of a large context takes, so "did the turn wait for it?" is
    answerable by the clock rather than by inspecting call order.
    """

    def __init__(self, delay: float = SUMMARY_DELAY_S) -> None:
        self.delay = delay
        #: When set, a summarization call BLOCKS on it instead of sleeping.
        #: A gate makes "did the turn wait for the summary?" a question about
        #: control flow rather than about the clock: with the call pinned open
        #: the boundary either returns or it does not, and a machine-speed
        #: threshold cannot make the answer flaky. Real wall-clock latency is
        #: measured on the real path in
        #: ``scripts/measure_async_compaction_latency.py``.
        self.gate: asyncio.Event | None = None
        self.summary_calls = 0
        #: Peak number of summarization calls in flight at the same moment.
        #: The latch's real invariant is about CONCURRENCY, not a lifetime
        #: count: a pass that finished and applied may legitimately be
        #: followed by another one later in the same run.
        self.peak_concurrent_summaries = 0
        self._live_summaries = 0
        self.fail_summary = False

    @staticmethod
    def _is_summary(request: Any) -> bool:
        # Recognised by the summarizer's own system prompt, which is what
        # actually distinguishes the call on the wire. Shape alone does not:
        # these sessions are built with ``tools=[]``, so "a system block and
        # no tools" matches an ordinary turn too, and every call looked slow.
        blocks = getattr(request, "system_blocks", None) or []
        return any("context compaction summarizer" in str(block) for block in blocks)

    def __call__(self, request, signal):
        summary = self._is_summary(request)
        if summary:
            self.summary_calls += 1
        delay = self.delay if summary else 0.0
        fail = summary and self.fail_summary

        async def gen():
            if summary:
                self._live_summaries += 1
                self.peak_concurrent_summaries = max(
                    self.peak_concurrent_summaries, self._live_summaries
                )
            try:
                if summary and self.gate is not None:
                    await self.gate.wait()
                elif delay:
                    await asyncio.sleep(delay)
                if fail:
                    raise RuntimeError("summarization failed")
                yield StreamTextDelta(delta="SUMMARY" if summary else "reply")
                yield StreamEndEvent(stop_reason="stop")
            finally:
                if summary:
                    self._live_summaries -= 1

        return gen()


def make_session(tmp_path, stream=None, **kwargs) -> Session:
    settings = kwargs.pop("compaction_settings", advisor_settings())
    return Session(
        model=BIG_MODEL,
        stream_fn=stream or SlowSummaryStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings,
        **kwargs,
    )


def advisor_settings(**overrides) -> CompactionSettings:
    base = {
        "keep_recent_tokens": KEEP_RECENT,
        # context-full pins the strategy to the one that MAKES a summarization
        # provider call. ``auto`` on this image-capable model resolves to
        # snapcompact, which rasterizes locally and makes no call at all, so
        # there would be no latency to measure and nothing to run off the turn.
        "strategy": "context-full",
        "advisor_enabled": True,
        "advisor_floor_tokens": 200_000,
        "advisor_trigger_tokens": 300_000,
        "advisor_every_n_turns": 1,
    }
    base.update(overrides)
    return CompactionSettings(**base)


def pin_measured_context(monkeypatch, tokens: int) -> None:
    monkeypatch.setattr(compaction_api, "messages_tokens_upper_bound", lambda messages: tokens)
    monkeypatch.setattr(compaction_api, "estimate_messages_tokens", lambda messages: tokens)


def usage_at(context_tokens: int) -> Usage:
    return Usage(input_tokens=context_tokens, output_tokens=10, context_tokens=context_tokens)


def seed_hint(session: Session, *, preserve: int = KEEP_RECENT) -> None:
    """Park a validated hint the way a completed advisor call would."""
    session._advisor_hint = CompactionHint(
        preserve_from_id=session._context.messages[-1].id,
        preserve_tokens=preserve,
        compact_now=True,
        confidence=0.9,
        reason="task boundary reached",
        turn_index=session._generation,
    )


async def talk(session: Session, turns: int = 6) -> None:
    for index in range(turns):
        await session.prompt(f"question {index} " + "detail " * 30)


async def drive_boundary(session: Session, context_tokens: int) -> Any:
    """Run the REAL mid-turn hook at a given provider-reported context size."""
    assistant = Message.assistant("mid-run reply")
    assistant.usage = usage_at(context_tokens)
    return await session._on_turn_end([*session._context.messages, assistant])


async def _until(predicate, message: str, tries: int = 200) -> None:
    """Yield the loop until ``predicate`` holds, then assert it did.

    Detached tasks are SCHEDULED synchronously and run later, so a test that
    asserts on their effect immediately after the spawn is asserting on the
    scheduler rather than on the code.
    """
    for _ in range(tries):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(message)


async def settle_background(session: Session, tries: int = 200) -> None:
    """Let the detached pass finish, the way a real turn's next steps do."""
    for _ in range(tries):
        await asyncio.sleep(0.01)
        if not session._compaction_pass_in_flight:
            return


# --- THE GAP THIS CLOSES --------------------------------------------------


@pytest.mark.asyncio
async def test_the_boundary_that_triggers_a_pass_is_not_blocked_by_it(tmp_path, monkeypatch):
    """THE regression. A boundary that fires an advisory pass must return
    without waiting for the summarization call.

    Against the inline code this FAILS: ``_on_turn_end`` awaited
    ``_run_compaction``, so the boundary took at least ``SUMMARY_DELAY_S``.
    The assertion is deliberately generous (half the delay) so it measures the
    inline/async distinction rather than machine speed.
    """
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    # The summarization call is pinned OPEN for the whole boundary. Inline,
    # the boundary can only return by completing it, so it cannot return at
    # all; asynchronously it returns immediately with the call still running.
    stream.gate = asyncio.Event()

    await asyncio.wait_for(drive_boundary(session, 400_000), timeout=5.0)

    # It really is the pass that is outstanding, not a pass that never ran.
    assert session._compaction_pass_in_flight, "no background pass was started"
    # The detached task is scheduled, not yet run: yield until it reaches the
    # provider call. Nothing in production waits for this either.
    await _until(lambda: stream.summary_calls == 1, "the summarization call never started")
    assert session._pending_compaction is None, "the pass committed while still summarizing"

    stream.gate.set()
    await settle_background(session)
    assert session._pending_compaction is not None
    await session.dispose()


@pytest.mark.asyncio
async def test_the_pass_applies_on_a_later_boundary_after_the_talk_continued(tmp_path, monkeypatch):
    """The pass lands on a LATER step, with real conversation in between.

    Against the inline code this fails at the first assertion: the pass
    committed during the first boundary, so there was no window in which the
    conversation continued with the pass still in flight.
    """
    session = make_session(tmp_path)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    await drive_boundary(session, 400_000)
    # NOTHING has been committed yet: no events, context untouched.
    before_ids = [message.id for message in session._context.messages]
    assert not [event for event in events if isinstance(event, CompactionStartEvent)]

    # The conversation CONTINUES while the pass is in flight — this is the
    # concurrency the feature exists for.
    session._context.messages.append(Message.assistant("work done while compacting"))
    await settle_background(session)
    assert session._pending_compaction is not None, "the pass never produced a result"

    replacement = await drive_boundary(session, 400_000)

    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert ends and ends[-1].success, "the finished pass never applied at a later boundary"
    assert ends[-1].detail == "advisor: task boundary reached"
    assert [message.id for message in session._context.messages] != before_ids
    # The loop is handed the rebuilt context, or its run accumulator would
    # re-send the history the pass just removed.
    assert replacement is not None
    assert [message.id for message in replacement] == [
        message.id for message in session._context.messages
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_work_added_while_the_pass_ran_survives_it(tmp_path, monkeypatch):
    """A message appended DURING the pass must still be in context after it.

    This is the failure that would be worst: a background pass applying a cut
    computed against an older, shorter history and silently dropping whatever
    arrived in the meantime.
    """
    session = make_session(tmp_path)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    await drive_boundary(session, 400_000)
    marker = Message.assistant("DO NOT LOSE ME")
    session._context.messages.append(marker)
    await settle_background(session)
    await drive_boundary(session, 400_000)

    texts = [getattr(message, "text", "") or "" for message in session._context.messages]
    assert any(
        "DO NOT LOSE ME" in text for text in texts
    ), "a message appended while the background pass ran was dropped by it"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_pass_whose_prefix_moved_is_discarded(tmp_path, monkeypatch):
    """A stale pass is DISCARDED, not misapplied.

    The prefix is rewritten underneath the in-flight pass (what a competing
    pass or a rebuild does). The summary describes a conversation that no
    longer exists, so applying it would splice it over different history.
    """
    session = make_session(tmp_path)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    await drive_boundary(session, 400_000)
    await settle_background(session)
    assert session._pending_compaction is not None

    # Move the prefix: drop the oldest message, which is inside the span the
    # pending summary replaces. Ids no longer match, so the pass is stale.
    survivors = list(session._context.messages[1:])
    session._context.messages = survivors
    await drive_boundary(session, 400_000)

    assert session._pending_compaction is None, "a stale pass was left pending"
    assert not [
        event for event in events if isinstance(event, CompactionEndEvent) and event.success
    ], "a stale pass was APPLIED to a conversation that had moved past it"
    # Fail open: the session is exactly as if no pass had run.
    assert [message.id for message in session._context.messages] == [
        message.id for message in survivors
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_failed_background_pass_leaves_the_session_untouched(tmp_path, monkeypatch):
    """Fail open: a summarization error is a missing pass, never a broken one."""
    stream = SlowSummaryStream()
    stream.fail_summary = True
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    before_ids = [message.id for message in session._context.messages]
    await drive_boundary(session, 400_000)
    await settle_background(session)

    assert session._pending_compaction is None
    assert not session._compaction_pass_in_flight, "the in-flight latch leaked on failure"
    await drive_boundary(session, 400_000)
    assert [message.id for message in session._context.messages] == before_ids
    assert not [
        event for event in events if isinstance(event, CompactionEndEvent) and event.success
    ]
    await session.dispose()


@pytest.mark.asyncio
async def test_only_one_background_pass_is_outstanding(tmp_path, monkeypatch):
    """The boundary fires per tool batch; concurrent passes must not.

    Without the latch a long tool run would spawn one summarization call per
    batch, each against a snapshot the next invalidates, and bill for all of
    them. The invariant is CONCURRENCY: a pass that has finished and applied
    may legitimately be followed by another one, so a lifetime count would
    reject correct sequential behaviour.
    """
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)
    # Held open for the whole loop, so every later boundary meets a pass that
    # is genuinely still running — the state the latch exists to cover.
    stream.gate = asyncio.Event()

    for _ in range(5):
        seed_hint(session)
        await asyncio.wait_for(drive_boundary(session, 400_000), timeout=5.0)
        await asyncio.sleep(0)  # let any spawned pass reach its provider call

    assert stream.peak_concurrent_summaries <= 1, (
        f"{stream.peak_concurrent_summaries} summarization calls ran at once — the "
        "in-flight latch is not holding, so a long tool run bills a pass per batch"
    )
    assert stream.summary_calls == 1
    assert session._compaction_pass_in_flight

    stream.gate.set()
    await settle_background(session)
    await session.dispose()


# --- WHAT MUST STAY SYNCHRONOUS -------------------------------------------


@pytest.mark.asyncio
async def test_the_ceiling_pass_is_still_synchronous(tmp_path, monkeypatch):
    """The safety net must NOT become async-only.

    With no advisory in play and the context genuinely at the ceiling, the
    turn cannot safely continue, so the pass that relieves it has to complete
    before the boundary returns — blocking is the correct behaviour here.
    """
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 700_000)  # above the 600k trigger
    assert session._advisor_hint is None

    # Pin the summarization open. A SYNCHRONOUS pass cannot get past it, so
    # the boundary must not complete; if it does, the ceiling has been made
    # asynchronous and the safety net is gone.
    stream.gate = asyncio.Event()
    boundary = asyncio.ensure_future(drive_boundary(session, 700_000))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(boundary), timeout=0.5)
    assert not [
        event for event in events if isinstance(event, CompactionEndEvent)
    ], "the ceiling pass reported an outcome before it had a summary"

    stream.gate.set()
    await asyncio.wait_for(boundary, timeout=5.0)

    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert ends and ends[-1].success, "the ceiling pass did not run inline"
    # "mid-turn" is the reason the mid-turn hook stamps; what matters here is
    # that the pass ran inline, which the gate above already proved.
    assert ends[-1].reason == "mid-turn"
    assert session._pending_compaction is None, "the ceiling pass went through the async path"
    await session.dispose()


@pytest.mark.asyncio
async def test_manual_compact_is_still_synchronous(tmp_path, monkeypatch):
    """``/compact`` is a request the user is WAITING for."""
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)

    # The user is WAITING for this one, so it must not return before the
    # summary exists.
    stream.gate = asyncio.Event()
    manual = asyncio.ensure_future(session.compact_now())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(manual), timeout=0.5)

    stream.gate.set()
    outcome = await asyncio.wait_for(manual, timeout=5.0)

    assert outcome.ran, f"manual compaction refused: {outcome.reason} {outcome.detail}"
    assert session._pending_compaction is None, "/compact went through the async path"
    await session.dispose()


@pytest.mark.asyncio
async def test_dispose_does_not_leak_an_in_flight_pass(tmp_path, monkeypatch):
    """A pass still running at teardown is cancelled with the other background
    work, and applies nothing to a disposed session."""
    session = make_session(tmp_path)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    await drive_boundary(session, 400_000)
    assert session._compaction_pass_in_flight
    await session.dispose()

    assert session._pending_compaction is None
    assert not [task for task in session._background_tasks if not task.done()]


# --- A HINT MUST NOT DIVERT A GENUINE CEILING BREACH ----------------------
#
# Agent review round 4, MAJOR-1. The routing used to test "is a hint in hand",
# and justified deferring on the premise that an advisory pass is "by
# definition below the configured threshold". Nothing enforces that:
# `_maybe_spawn_advisor` gates on a LOWER bound only, so a usable hint can sit
# in the slot while the context is genuinely over the ceiling. `should_compact`
# then fires on size alone, the plan carries the hint, and the one pass that is
# not supposed to be deferrable was deferred.


@pytest.mark.asyncio
async def test_a_ceiling_breach_with_a_hint_pending_still_runs_synchronously(tmp_path, monkeypatch):
    """THE round-4 regression, at the reviewer's reproduction size.

    700k against a 600k trigger, with one usable hint seeded. The hint is real
    and usable; the context is over the ceiling anyway, so the pass must
    relieve the turn NOW rather than being handed to the background.
    """
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 700_000)
    seed_hint(session)

    # Pinned open, so a SYNCHRONOUS pass cannot complete the boundary. If the
    # boundary returns anyway, the ceiling was routed to the background.
    stream.gate = asyncio.Event()
    boundary = asyncio.ensure_future(drive_boundary(session, 700_000))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(boundary), timeout=0.5)

    assert stream.summary_calls == 1, "no summarization was started at the ceiling"
    assert not session._compaction_pass_in_flight, (
        "a genuine ceiling breach was deferred to the background because a hint "
        "happened to be pending — the safety net became asynchronous"
    )

    stream.gate.set()
    await asyncio.wait_for(boundary, timeout=5.0)
    ends = [event for event in events if isinstance(event, CompactionEndEvent)]
    assert ends and ends[-1].success, "the ceiling pass never committed"
    assert session._pending_compaction is None
    await session.dispose()


@pytest.mark.asyncio
async def test_a_small_window_at_its_ceiling_is_relieved_on_the_turn(tmp_path, monkeypatch):
    """The reviewer's second shape: 195k of a 200k window, hint pending.

    Here the synchronous pass is the only thing between the turn and an
    overflow, and the absolute 600k knob is irrelevant — the resolved trigger
    is the PERCENTAGE of a small window. Routing must read the same resolved
    number the gate does, not a hard-coded ceiling.
    """
    small = ModelSpec(provider="test", model_id="small", context_window=200_000)
    stream = SlowSummaryStream()
    session = Session(
        model=small,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=advisor_settings(
            # Inside the advisor's own band for this window, so a hint is
            # legitimately in hand at the same time the ceiling is breached.
            advisor_trigger_tokens=100_000,
            advisor_floor_tokens=120_000,
        ),
    )
    await talk(session)
    pin_measured_context(monkeypatch, 195_000)
    seed_hint(session)

    stream.gate = asyncio.Event()
    boundary = asyncio.ensure_future(drive_boundary(session, 195_000))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(boundary), timeout=0.5)

    assert (
        not session._compaction_pass_in_flight
    ), "a 200k-window session at 195k deferred its relief and continued unrelieved"
    stream.gate.set()
    await asyncio.wait_for(boundary, timeout=5.0)
    await session.dispose()


@pytest.mark.asyncio
async def test_repeated_ceiling_boundaries_are_never_starved(tmp_path, monkeypatch):
    """The reviewer's third shape: the deferral must not SUSTAIN.

    With the provider slow and hints arriving, the old routing deferred every
    ceiling boundary and committed nothing across five of them, because the
    spawn refuses to queue and the caller deliberately does not fall through.
    Each ceiling boundary must instead do the pass itself.
    """
    stream = SlowSummaryStream(delay=0.05)
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    events: list[Any] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 700_000)

    committed = 0
    for _ in range(5):
        seed_hint(session)
        await asyncio.wait_for(drive_boundary(session, 700_000), timeout=10.0)
        committed = len([e for e in events if isinstance(e, CompactionEndEvent) and e.success])
        if committed:
            break

    assert committed, (
        "five consecutive ceiling boundaries at 700k committed no pass at all — "
        "the ceiling is being starved by the asynchronous path"
    )
    await settle_background(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_sub_ceiling_advisory_pass_still_goes_to_the_background(tmp_path, monkeypatch):
    """The fix must not close the async path it exists to open.

    Same session, same hint, context genuinely BELOW the trigger: this is the
    case the feature is for, and it must still defer.
    """
    stream = SlowSummaryStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    stream.gate = asyncio.Event()
    await asyncio.wait_for(drive_boundary(session, 400_000), timeout=5.0)

    assert session._compaction_pass_in_flight, (
        "a sub-ceiling advisory pass was made synchronous — the fix for MAJOR-1 "
        "has closed the path the feature exists to open"
    )
    stream.gate.set()
    await settle_background(session)
    await session.dispose()
