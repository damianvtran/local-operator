"""What the status band knows about tokens and money BEFORE a turn ends.

Four reports from the field, all the same shape — the band's two numeric
segments are fed only by turns that complete while the app is running, so every
moment before that they described a session that was not the one on screen:

1. "on resume the initial readout on token usage is inaccurate" — a resumed
   conversation opened claiming an almost-empty context. Measured on a session
   whose last provider reading was 402k of a 1M window: the band said 0.2%/1M.
2. The same resume showed no cost at all, for a conversation with real spend
   already on it.
3. "the cost doesn't start tracking until after the second message" — a new
   session's first turn showed nothing in the cost segment for the whole time it
   ran, which is exactly when a long agentic turn is spending the most.
4. The baseline a NEW session opens with has to be the tokens the first request
   would actually carry, not a figure that ignores half of what is sent.

These are driven through the REAL app and the REAL session wherever the fact
under test lives there: the wiring (`_adopt_session` ordering, the per-call
accrual and its reconciliation at turn end) is what was broken, and a test
calling the accessors directly would pass while the band stayed wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from local_operator.harness.types import Message, ModelSpec, TextContent, Usage
from local_operator.model.registry import ModelInfo
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import ContextUsageReported, TurnEnded

#: $10 in / $100 out per MTok, so a million tokens is a round dollar figure and
#: a mispriced figure is unmistakable rather than a rounding argument.
_MODEL = ModelInfo(id="sonnet", name="sonnet", description="", input_price=10.0, output_price=100.0)


def _resolving():
    return patch(
        "local_operator.model.configure.resolve_model_info",
        side_effect=lambda provider, model_id: _MODEL,
    )


def _spec() -> ModelSpec:
    return ModelSpec(provider="anthropic", model_id="sonnet", context_window=1_000_000)


def _assistant(*, output: int, context: int) -> Message:
    """One settled assistant turn carrying the provider's own usage."""
    return Message(
        role="assistant",
        content=[TextContent(text="answer")],
        stop_reason="stop",
        usage=Usage(
            input_tokens=20,
            output_tokens=output,
            cache_read_tokens=max(context - 20, 0),
            cache_write_tokens=0,
            context_tokens=context,
        ),
    )


async def _session_over(directory: Path, messages: list[Message]) -> Session:
    """A real Session whose transcript already holds ``messages`` — a resume."""
    transcript = Transcript(directory)
    for message in messages:
        await transcript.append_message(message)

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover - never called
        if False:
            yield None

    # A SECOND Transcript over the same directory, so the session replays the
    # rows from disk exactly as `--resume` does rather than inheriting an
    # in-memory list the writer happened to still hold.
    return Session(
        model=_spec(),
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["system"],
    )


async def _settled(app: OperatorApp, pilot: Any, ticks: int = 80) -> None:
    """Wait for the session to be ADOPTED, then let its measurement land.

    The app paints before its session exists — the factory is awaited in a boot
    worker — and everything these tests assert is installed by `_adopt_session`.
    Waiting on that condition rather than on a fixed tick count is what keeps
    them from racing the boot on a loaded machine: the same number of frames is
    plenty in isolation and not enough under a full suite, which is a test that
    fails for a reason unrelated to what it is about.
    """
    for _ in range(ticks):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if app._session is not None:
            break
    assert app._session is not None, "the session never booted"
    # Then wait for the READING, not for a number of frames. Adoption is only
    # the first half: `_measure_preloaded_context` starts its own worker
    # afterwards and these tests assert on its result, so a fixed pause here is
    # the same guess one worker later — it passes on an idle machine and fails
    # when the measurement takes longer than the frames allowed for it.
    status = app._status
    assert status is not None
    for _ in range(ticks):
        if status.context_tokens:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_resumed_session_opens_on_the_provider_s_own_context_reading(
    tmp_path: Path,
) -> None:
    """The band reports what the conversation ACTUALLY holds, exactly.

    The reading is the provider's own ``context_tokens`` from the last call of
    the last turn, so it is installed as exact rather than as an estimate — it
    is the number the band would still be showing had the process never
    stopped. Pinned as exact because that flag is also what stops the local
    preload estimate from overwriting it a moment later, in a worker.
    """
    session = await _session_over(
        tmp_path / "sess",
        [
            Message.user("q1"),
            _assistant(output=900, context=120_000),
            Message.user("q2"),
            _assistant(output=1_100, context=402_000),
        ],
    )

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await _settled(app, pilot)

            assert app._status is not None
            assert app._status.context_tokens == 402_000
            assert app._status.context_is_estimate is False


@pytest.mark.asyncio
async def test_a_resumed_session_opens_with_the_spend_it_already_has(tmp_path: Path) -> None:
    """Cost is seeded rather than left empty, because empty reads as free.

    It is a FLOOR, not the lifetime total: only the last reported turn's usage
    survives in a form the app can price. Asserted as "the last turn's price,
    and not zero" rather than as a total over the conversation, because
    claiming the latter would be the same kind of lie in the other direction.
    """
    session = await _session_over(
        tmp_path / "sess",
        [Message.user("q"), _assistant(output=1_000, context=200_000)],
    )

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await _settled(app, pilot)

            assert app._status is not None
            # Anthropic reports `input_tokens` EXCLUDING its cache buckets, so
            # all three are billed: 20 plain + 199_980 cache-read at $10/MTok
            # (this model publishes no separate cache rate, so cache reads fall
            # back to the input price), plus 1000 output at $100/MTok.
            assert app._total_cost == pytest.approx(0.0002 + 1.9998 + 0.1)
            assert app._status._cost, "a resumed conversation with spend must not read as free"


@pytest.mark.asyncio
async def test_a_new_session_restores_nothing_and_keeps_its_estimate(tmp_path: Path) -> None:
    """The restore is silent on a conversation that never reported usage.

    The complement of the two above, and the reason ``restored_usage`` returns
    the object rather than an int: "nothing was ever reported" and "zero was
    reported" are different facts, and only the first may fall through to the
    local estimate. A new session must keep its estimated baseline and must NOT
    gain a cost segment out of nowhere.
    """
    session = await _session_over(tmp_path / "sess", [])

    async def factory() -> Session:
        return session

    app = OperatorApp(factory)
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await _settled(app, pilot)

            assert app._status is not None
            assert session.restored_usage() is None
            assert app._total_cost == 0.0
            assert app._status._cost == ""
            # The preload estimate still ran, and is still flagged as one.
            assert app._status.context_tokens > 0
            assert app._status.context_is_estimate is True


@pytest.mark.asyncio
async def test_a_compacted_resume_reports_nothing_rather_than_the_old_context(
    tmp_path: Path,
) -> None:
    """A compacted history's readings describe a context that no longer exists.

    Replay puts the summary marker at the head and the KEPT WINDOW after it, and
    those kept messages still carry their pre-compaction ``usage``. Nothing
    supersedes them, because only a completed turn rewrites the reading and a
    session that compacted and then exited never ran one.

    Seeding from one is worse than the bug the seeding fixed: measured at 900k
    against a real ~1.7k, it would install a 527x overstatement as EXACT — which
    also suppresses the correct local estimate — and hand it to
    ``should_compact``, rewriting the user's history on the first turn after the
    resume. ``None`` is the right answer: fall through to the estimate.
    """
    transcript = Transcript(tmp_path / "sess")
    stale = _assistant(output=100, context=900_000)
    await transcript.append_message(Message.user("q"))
    await transcript.append_message(stale)
    await transcript.append_compaction(
        summary="what happened earlier",
        first_kept_entry_id=stale.id,
        tokens_before=900_000,
    )

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover - never called
        if False:
            yield None

    session = Session(
        model=_spec(),
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["system"],
    )

    assert session.restored_usage() is None
    # And the fallback is sane: the local estimate describes the KEPT window,
    # nowhere near the pre-compaction figure it refused.
    assert await session.measure_preloaded_context() < 100_000


@pytest.mark.asyncio
async def test_turns_taken_after_a_compaction_are_still_restored(tmp_path: Path) -> None:
    """The refusal is scoped to readings the pass invalidated, not to the file.

    The complement of the test above, and the reason the boundary is taken from
    the TRANSCRIPT's append order rather than from the replayed list: a session
    that compacted and then ran more turns has a perfectly good newest reading.
    Refusing it because the file contains a marker anywhere would send every
    such resume back to the local estimate — the original bug, reintroduced for
    the majority of long conversations, which are exactly the ones that compact.
    """
    transcript = Transcript(tmp_path / "sess")
    stale = _assistant(output=100, context=900_000)
    await transcript.append_message(Message.user("old"))
    await transcript.append_message(stale)
    await transcript.append_compaction(
        summary="what happened earlier",
        first_kept_entry_id=stale.id,
        tokens_before=900_000,
    )
    # Three ordinary turns since the pass; the last one is the live reading.
    for context in (120_000, 140_000, 160_000):
        await transcript.append_message(Message.user("q"))
        await transcript.append_message(_assistant(output=50, context=context))

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover - never called
        if False:
            yield None

    session = Session(
        model=_spec(),
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["system"],
    )

    restored = session.restored_usage()
    assert restored is not None, "a turn taken after the pass is a valid reading"
    assert restored.context_tokens == 160_000
    assert restored.context_tokens != 900_000, "and it is not the pre-compaction one"


@pytest.mark.asyncio
async def test_a_pruned_reading_is_refused_like_a_compacted_one(tmp_path: Path) -> None:
    """Pruning shrinks the context too, and leaves no marker to notice it by.

    Blanking a tool result is the other pass that makes a reading describe a
    context that no longer exists — and unlike compaction it writes no boundary:
    the journal entry is folded away by ``compact_file`` and the message row
    survives with its original ``usage`` attached. Measured on a real prune, a
    restored reading of 640_000 against a true context of ~1_700, installed as
    exact.

    Wider than the compaction case it mirrors, too: it needs nothing more than
    the agent reading the same large file twice. So the rule is not "after the
    newest compaction" but "not describing a context something has since
    shrunk", and both shrinking passes have to be covered.
    """
    transcript = Transcript(tmp_path / "sess")
    pruned = _assistant(output=100, context=640_000)
    await transcript.append_message(Message.user("read that file"))
    await transcript.append_message(pruned)
    await transcript.append_prune(pruned.id, "[pruned]")

    async def _stream(request: Any, signal: Any = None):  # pragma: no cover - never called
        if False:
            yield None

    def _session_over_dir() -> Session:
        return Session(
            model=_spec(),
            stream_fn=_stream,
            tools=[],
            transcript=Transcript(tmp_path / "sess"),
            system_blocks_provider=lambda: ["system"],
        )

    assert _session_over_dir().restored_usage() is None

    # And after the journal has been FOLDED INTO the rows, where the prune entry
    # no longer exists and only the flag on the message remains.
    reclaimed = await transcript.compact_file(min_reclaim_bytes=1)
    assert reclaimed >= 0
    assert _session_over_dir().restored_usage() is None

    # A healthy turn taken afterwards is still restored: the refusal is scoped
    # to readings the pass invalidated, not to the conversation.
    await Transcript(tmp_path / "sess").append_message(_assistant(output=20, context=15_000))
    restored = _session_over_dir().restored_usage()
    assert restored is not None and restored.context_tokens == 15_000


@pytest.mark.asyncio
async def test_the_baseline_never_loads_the_tokenizer(tmp_path: Path) -> None:
    """The boot-path measurement keeps its documented chars/4 contract.

    ``measure_preloaded_context``'s own docstring refuses two costs, and the
    tokenizer is the first of them: cl100k_base is ~43.6 MB RSS and, on a cold
    cache, a NETWORK fetch of the ranks — during boot, before the user has
    typed. Counting the history with the sharper ruler would have spent exactly
    that (measured: 58 ms and +41 MB against 0.5 ms and +0.1 MB), and the
    memoization does not amortize it because a session that never approaches the
    compaction threshold never tokenizes at all.
    """
    import local_operator.compaction.tokens as tokens_module

    session = await _session_over(
        tmp_path / "sess",
        [Message.user("q" * 5_000), _assistant(output=50, context=10_000)],
    )

    with patch.object(tokens_module, "_get_encoding", side_effect=AssertionError("loaded")) as gate:
        assert await session.measure_preloaded_context() > 0
        assert gate.call_count == 0


@pytest.mark.asyncio
async def test_the_new_session_baseline_counts_what_the_first_request_carries(
    tmp_path: Path,
) -> None:
    """The opening reading is the system blocks AND the schemas AND the history.

    A new session's history is empty, so this is where the baseline's third
    term is proved rather than assumed: the same measurement over a session
    holding a conversation has to grow by that conversation. Before, it counted
    blocks and schemas only, which is why every RESUMED session opened at the
    size of an empty one.
    """
    empty = await _session_over(tmp_path / "empty", [])
    loaded = await _session_over(
        tmp_path / "loaded",
        [Message.user("q" * 4_000), _assistant(output=10, context=5_000)],
    )

    baseline = await empty.measure_preloaded_context()
    with_history = await loaded.measure_preloaded_context()

    assert baseline > 0, "a session with a system prompt is never at zero"
    assert with_history > baseline, "history the model will be sent has to be counted"


@pytest.mark.asyncio
async def test_cost_appears_during_the_first_turn_not_after_it() -> None:
    """Money moves on each model call, so the FIRST turn shows its own spend.

    Reported as "the cost doesn't start tracking until after the second
    message". The per-call event is the only signal available while a turn is
    still running, which is exactly the stretch a long agentic turn is spending
    the most in.
    """
    from tests.unit.tui.test_band_panels import FakeSession, _async_factory

    app = OperatorApp(_async_factory(FakeSession()))
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await pilot.pause()
            assert app._status is not None
            assert app._status._cost == "", "nothing has been spent yet"

            # One model call inside a turn that has NOT ended.
            app.post_message(
                ContextUsageReported(
                    50_000,
                    Usage(input_tokens=1_000, output_tokens=500, context_tokens=50_000),
                )
            )
            await pilot.pause()

            assert app._status._cost, "the segment must move before the turn ends"
            assert app._total_cost == pytest.approx(0.01 + 0.05)
            assert app._status.context_tokens == 50_000


@pytest.mark.asyncio
async def test_a_turn_is_not_billed_twice_for_the_calls_it_already_paid_for() -> None:
    """The turn total supersedes the running one instead of adding to it.

    The hazard the per-call accrual introduces: ``agent_end`` prices the WHOLE
    turn (summing every call), so adding it whole on top of the calls already
    billed would report roughly double. The end adds only the remainder, and
    the accrual resets so the next turn starts clean.
    """
    from tests.unit.tui.test_band_panels import FakeSession, _async_factory

    app = OperatorApp(_async_factory(FakeSession()))
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await pilot.pause()

            # Two calls inside one turn, billed as they land.
            for _ in range(2):
                app.post_message(
                    ContextUsageReported(
                        10_000,
                        Usage(input_tokens=1_000, output_tokens=500, context_tokens=10_000),
                    )
                )
                await pilot.pause()
            assert app._total_cost == pytest.approx(2 * (0.01 + 0.05))

            # The turn ends, reporting the SUM of those same two calls.
            app.post_message(
                TurnEnded(
                    False,
                    None,
                    context_tokens=10_000,
                    usage=Usage(input_tokens=2_000, output_tokens=1_000, context_tokens=10_000),
                )
            )
            await pilot.pause()

            # The total, not the double.
            assert app._total_cost == pytest.approx(0.02 + 0.10)
            assert app._turn_accrued_cost == 0.0, "the next turn must start from zero"


@pytest.mark.asyncio
async def test_a_turn_end_that_prices_more_than_was_accrued_adds_the_remainder() -> None:
    """A call the per-call path never saw is still paid for at the end.

    The turn total is authoritative precisely because it sums calls this app may
    not have received an event for (a provider that reports usage only at the
    end, an event dropped across a reload). The remainder is what keeps the
    figure right in that case rather than under-reporting it.
    """
    from tests.unit.tui.test_band_panels import FakeSession, _async_factory

    app = OperatorApp(_async_factory(FakeSession()))
    with _resolving():
        async with app.run_test(size=(100, 18)) as pilot:
            await pilot.pause()

            app.post_message(
                ContextUsageReported(
                    10_000,
                    Usage(input_tokens=1_000, output_tokens=500, context_tokens=10_000),
                )
            )
            await pilot.pause()

            # The turn totals THREE calls' worth; only one was seen live.
            app.post_message(
                TurnEnded(
                    False,
                    None,
                    context_tokens=10_000,
                    usage=Usage(input_tokens=3_000, output_tokens=1_500, context_tokens=10_000),
                )
            )
            await pilot.pause()

            assert app._total_cost == pytest.approx(0.03 + 0.15)
