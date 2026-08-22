"""The aggregate working line: where it sits, and what it says.

Reported from the field: "the 'Working' text stays under the message I sent but
doesn't move with the conversation... it should update to show what the agent is
currently doing". Two defects in one sentence, and this file pins both.

POSITION. The line used to be appended once at ``agent_start`` and left there,
so every block the turn produced mounted *below* it. Three tool calls in, the
only live thing on screen was somewhere in the scrollback while the foot of the
transcript — the row the user is actually looking at, one line above the
composer — showed a settled receipt. It is now PINNED to the bottom
(``TranscriptView.pin_tail``) and travels with the conversation.

CONTENT. It used to say "working…" and nothing else, through gaps that on a real
provider run to minutes. It now names the current activity, and every label it
shows is derived from an event the app received: a running tool's own
description, the call the model is still dictating, prose arriving, a
compaction, a retry, an approval the turn is parked on. There is no invented
activity and no randomised flavour text — a line that guessed would be worse
than the one that said nothing.

LIFETIME. The line is owned by one turn, and ``/reload``, ``/resume`` and
``/new`` end that turn from outside the event stream — no ``agent_end``
arrives to lift it. Both halves of that are pinned below: the widget the user
can see, and the ``_working_block`` reference behind it, whose survival
silenced the working line for every turn of every session that followed.

The assertions are about the observable frame and the block order, never about
how the app routes messages internally.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from rich.cells import cell_len
from rich.console import RenderableType
from rich.text import Text

from local_operator.harness.types import (
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
    CompactionStarted,
    ToolComposing,
    ToolEnded,
    ToolStarted,
    TurnBoundaryEnd,
    TurnBoundaryStart,
    TurnEnded,
    TurnStarted,
)
from local_operator.tui.widgets.tool_card import format_duration
from local_operator.tui.widgets.transcript import (
    DEFAULT_ACTIVITY,
    NoticeBlock,
    TranscriptView,
    WorkingBlock,
)

from .test_app_pilot import FakeSession, _factory


def _working(app: OperatorApp) -> WorkingBlock | None:
    """The mounted working line, or None when the turn has settled."""
    blocks = app.query_one(TranscriptView).blocks()
    lines = [block for block in blocks if isinstance(block, WorkingBlock)]
    assert len(lines) <= 1, "D25: there is exactly ONE aggregate working line"
    return lines[0] if lines else None


def _activity(app: OperatorApp) -> str:
    line = _working(app)
    assert line is not None, "no working line is mounted"
    return line.activity


def _is_last(app: OperatorApp) -> bool:
    blocks = app.query_one(TranscriptView).blocks()
    return bool(blocks) and isinstance(blocks[-1], WorkingBlock)


def _started(tool_call_id: str, tool_name: str, **args: Any) -> ToolStarted:
    return ToolStarted(
        ToolExecutionStartEvent(tool_call_id=tool_call_id, tool_name=tool_name, args=args)
    )


def _ended(tool_call_id: str, tool_name: str) -> ToolEnded:
    return ToolEnded(
        ToolExecutionEndEvent(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=ToolResult(tool_call_id=tool_call_id, tool_name=tool_name, content=[]),
        )
    )


@pytest.mark.asyncio
async def test_the_working_line_stays_at_the_foot_as_the_turn_grows() -> None:
    """The whole reported defect: it must travel with the conversation.

    Every kind of block a turn can append is thrown at it — a tool row, prose,
    a notice — because the pin lives in ``append_block`` and one uncovered
    caller is all it takes for the line to end up buried again.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        await pilot.pause()
        assert _is_last(app)

        app.post_message(_started("c0", "read", path="src/foo.py"))
        await pilot.pause()
        assert _is_last(app)

        app.post_message(AssistantDelta("thinking out loud"))
        await pilot.pause()
        assert _is_last(app)

        app.post_message(_ended("c0", "read"))
        app.post_message(CompactionStarted("size"))
        await pilot.pause()
        assert _is_last(app)
        # And the blocks it stepped over are still in the order they arrived —
        # a pin that reordered the transcript would be a worse bug than the one
        # it fixes.
        kinds = [type(block).__name__ for block in app.query_one(TranscriptView).blocks()]
        assert kinds[-1] == "WorkingBlock"
        assert kinds.count("WorkingBlock") == 1


@pytest.mark.asyncio
async def test_it_reports_the_running_tool_and_settles_back_to_the_model() -> None:
    """The line reports the call, then stops reporting it when the call ends.

    The second half is the one a latch gets wrong: a working line still naming
    a tool whose row settled is the same lie, one event later.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        await pilot.pause()
        # The GAP before anything has been dictated: a model call is in flight
        # and that is all the app honestly knows.
        assert _activity(app) == DEFAULT_ACTIVITY

        app.post_message(_started("c0", "read", path="src/foo.py"))
        await pilot.pause()
        assert _activity(app) == "running read"
        assert any(
            "running read" in strip.text for strip in app.screen._compositor.render_strips()
        ), "the label reaches the frame, not just the widget"

        app.post_message(_ended("c0", "read"))
        await pilot.pause()
        assert _activity(app) == DEFAULT_ACTIVITY


@pytest.mark.asyncio
async def test_the_model_s_intent_is_what_the_line_says() -> None:
    """The whole point of the second half of the report.

    The card and this line show different KINDS of thing: the card records what
    was run, this line narrates why. So they never carry the same string — a
    line built from the arguments painted the card's own sentence one row lower
    and read as a rendering fault — and the intent, which is the useful half,
    appears exactly once.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c0",
                    tool_name="bash",
                    args={"command": "pytest -q"},
                    intent="Running the spacing suite",
                )
            )
        )
        await pilot.pause()
        assert _activity(app) == "running the spacing suite"
        painted = [strip.text for strip in app.screen._compositor.render_strips()]
        assert len([row for row in painted if "running the spacing suite" in row]) == 1
        assert len([row for row in painted if "pytest -q" in row]) == 1


@pytest.mark.asyncio
async def test_without_an_intent_the_line_falls_back_to_the_tool_name() -> None:
    """Nothing populated ``intent`` before this round, so the degraded path is
    the shipping path until every provider and tool supplies one. It must still
    not restate the arguments the card is showing."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "read", path="src/foo.py"))
        await pilot.pause()
        assert _activity(app) == "running read"


@pytest.mark.asyncio
async def test_a_parallel_batch_is_reported_as_a_plain_count() -> None:
    """A batch drops the intent rather than suffixing a number to it.

    ``Auditing the spacing rule +2 more`` presents ONE call's stated purpose as
    the whole turn's activity, which the three rows above it contradict, and it
    buries the count behind arithmetic. The count is the one fact this row has
    that appears nowhere else in the frame, so it is what a batch says.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c0",
                    tool_name="read",
                    args={"path": "a.py"},
                    intent="Auditing the spacing rule",
                )
            )
        )
        app.post_message(_started("c1", "read", path="b.py"))
        app.post_message(_started("c2", "grep", pattern="needs_gap"))
        await pilot.pause()
        # No intent, no `+N`: three calls are three calls.
        assert _activity(app) == "running 3 tools"

        app.post_message(_ended("c1", "read"))
        await pilot.pause()
        assert _activity(app) == "running 2 tools"

        app.post_message(_ended("c0", "read"))
        await pilot.pause()
        # Down to one, and that one gave no intent, so the fallback shows.
        assert _activity(app) == "running grep"


@pytest.mark.asyncio
async def test_the_clock_survives_a_label_change_within_one_phase() -> None:
    """A batch shedding a call has not changed what the agent is doing.

    Keyed to the rendered string, the clock reset every time the count moved:
    a settling call showed ``✓ 4.0s`` on its receipt while the line two rows
    below restarted at ``0s``, and a batch losing one call every twenty seconds
    could never show a clock past twenty — which is the "has this been stuck"
    question the clock exists to answer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "read", path="a.py"))
        app.post_message(_started("c1", "read", path="b.py"))
        await pilot.pause()
        line = _working(app)
        assert line is not None
        started = line._phase_started

        app.post_message(_ended("c1", "read"))
        await pilot.pause()
        assert _activity(app) == "running read"  # the label moved
        assert line._phase_started == started  # the clock did not

        # A real phase change DOES reset it: that is the case the reset is for.
        app.post_message(_ended("c0", "read"))
        await pilot.pause()
        assert _activity(app) == DEFAULT_ACTIVITY
        assert line._phase_started > started


@pytest.mark.asyncio
async def test_the_line_holds_one_row_whatever_the_clock_says() -> None:
    """The row reserves cells for the clock instead of measuring it, so the
    reservation has to hold for every clock the formatter can return.

    It reserved five — the width of every duration anyone had watched — and
    ``format_duration`` returns six from ``10m10s`` up to ``59m59s``. The
    composed line then came out one cell over the terminal, and because the
    label is model-supplied prose the wrap broke at the word boundary AHEAD of
    the number: the row spent an extra row to show everything EXCEPT the clock,
    for fifty of every sixty seconds past ten minutes. The one fact the row's
    own docstring says nothing else on screen carries.

    Latent, too. The overflow only showed at widths where the truncated label
    did not happen to end on a space — ``rstrip()`` inside ``truncate_cells``
    was silently absorbing the extra cell at 60, 80 and 200 columns while 120
    wrapped. Swept across widths here for exactly that reason.

    ``100h``+ is in the sweep because the hours branch used to be ``{h}h{m}m``
    with an UNBOUNDED hours field, so six cells was the widest only below a
    hundred hours (review round 15). ``format_duration`` now carries a days
    branch and a ``99d+`` cap, so it is bounded by construction; this sweeps
    past both walls to say so.

    Driven by winding ``_phase_started`` back, because that is what ``_paint``
    reads. Three earlier versions of this test set ``_clock`` directly and were
    all VACUOUS — ``_paint`` recomputes it from the phase start on every call,
    so every mutation passed. Reading the painted strips is vacuous too:
    Textual clips each strip to the widget's box before it lands in the render
    cache, so a line composed one cell too wide arrives already trimmed. What
    is asserted here is the text ``_paint`` HANDS the widget, against the
    widget's own box — narrower than the terminal by the transcript's gutters,
    so asserting against the app width would pass trivially.
    """
    for width in (24, 60, 80, 120, 200):
        # A fresh app per width: an `OperatorApp` runs once, and `run_test` on a
        # second context for the same instance hangs waiting for a screen that
        # is never mounted again.
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 20)) as pilot:
            await pilot.pause()
            app.post_message(TurnStarted())
            app.post_message(_started("c0", "read", path="a.py"))
            await pilot.pause()
            line = _working(app)
            assert line is not None, width
            line.set_activity("running a tool with a long model-supplied label " * 3)
            box = line.size.width
            assert box > 0, (width, box)

            composed: list[tuple[float, str]] = []
            original = line.set_content

            def spy(content: RenderableType, **kw: Any) -> None:
                # `_paint` always composes a `Text`; asserted rather than cast,
                # so a change to a plain `str` fails here instead of quietly
                # skipping the width check this test exists for.
                assert isinstance(content, Text), type(content)
                composed.append((elapsed, content.plain))
                original(content, **kw)

            line.set_content = spy  # type: ignore[method-assign]
            # Past every wall in the formatter: the minutes/hours boundary, the
            # old unbounded-hours overflow at 100h, the days branch, and the
            # `100d+` cap — which only becomes load-bearing past 999 days, where
            # `{d}d{h}h` would itself reach seven cells.
            for elapsed in (5, 65, 610, 3599, 7200, 86_399, 362_400, 9_000_000, 100_000_000):
                line._phase_started = time.monotonic() - elapsed
                line._paint()
            line.set_content = original  # type: ignore[method-assign]

            assert len(composed) == 9, (width, composed)
            for elapsed, text in composed:
                assert "\n" not in text, (width, elapsed, text)
                assert cell_len(text) <= box, (width, box, elapsed, text)
                # The clock is PRESENT, not merely fitting: the defect was the
                # number being dropped to make the row fit, which an
                # width-only assertion cannot see.
                assert text.rstrip().endswith(format_duration(elapsed)), (
                    width,
                    elapsed,
                    text,
                )


@pytest.mark.asyncio
async def test_a_dictated_call_is_reported_before_it_runs() -> None:
    """The longest silence in a turn is a large call streaming its arguments."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(ToolComposing(ToolCallComposeEvent(tool_call_id="c0", tool_name="write")))
        await pilot.pause()
        assert _activity(app) == "composing a call"

        # Adoption: the execution replaces the announcement rather than adding
        # to it, so the line must not read "composing 2 tools".
        app.post_message(_started("c0", "write", path="out.txt"))
        await pilot.pause()
        assert _activity(app) == "running write"


@pytest.mark.asyncio
async def test_a_prose_turn_reports_prose_then_the_next_model_call() -> None:
    """A turn with no tools at all must still say something true."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        # A message OPENING is not text arriving — nothing is mounted yet and
        # the model call is still the honest description.
        assert _activity(app) == DEFAULT_ACTIVITY

        app.post_message(AssistantDelta("here is the plan"))
        await pilot.pause()
        assert _activity(app) == "responding"

        app.post_message(AssistantMessageEnd("here is the plan"))
        await pilot.pause()
        assert _activity(app) == DEFAULT_ACTIVITY


@pytest.mark.asyncio
async def test_a_new_model_call_takes_the_line_back_from_the_last_tool() -> None:
    """The gap the line exists for: a batch settles, then the model is quiet."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "read", path="a.py"))
        await pilot.pause()
        assert _activity(app) == "running read"

        app.post_message(TurnBoundaryEnd())
        app.post_message(TurnBoundaryStart())
        await pilot.pause()
        assert _activity(app) == DEFAULT_ACTIVITY


@pytest.mark.asyncio
async def test_a_slow_whole_turn_state_says_what_it_is() -> None:
    """Compaction has no card of its own and runs for a long time."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(CompactionStarted("size"))
        await pilot.pause()
        assert _activity(app) == "compacting context"


@pytest.mark.asyncio
async def test_the_line_is_lifted_when_the_turn_ends() -> None:
    """No settled frame, no summary row: it is transient and it goes."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "read", path="a.py"))
        await pilot.pause()
        assert _working(app) is not None

        app.post_message(_ended("c0", "read"))
        app.post_message(TurnEnded(aborted=False, error=None))
        await pilot.pause()
        assert _working(app) is None
        assert not any("thinking" in strip.text for strip in app.screen._compositor.render_strips())


@pytest.mark.asyncio
async def test_the_line_does_not_survive_an_aborted_turn() -> None:
    """Esc leaves the tool rows marked interrupted; the live line must go.

    A working line left spinning over a stopped turn is the frame that makes a
    user press Ctrl+C again.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert _working(app) is not None

        app.post_message(TurnBoundaryEnd())  # reconciles the orphaned card
        app.post_message(TurnEnded(aborted=True, error=None))
        await pilot.pause()
        assert _working(app) is None


@pytest.mark.asyncio
async def test_a_failed_turn_lifts_the_line_too() -> None:
    """The error notice takes the foot of the transcript, not a dead spinner."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        await pilot.pause()
        app.post_message(TurnEnded(aborted=False, error="provider refused the request"))
        await pilot.pause()
        assert _working(app) is None
        blocks = app.query_one(TranscriptView).blocks()
        assert isinstance(blocks[-1], NoticeBlock)


@pytest.mark.asyncio
async def test_clearing_mid_turn_brings_the_line_back() -> None:
    """``/clear`` empties the screen; it does not stop the turn.

    Without this the agent kept working into a transcript that looked idle, and
    the next tool row arrived out of an apparently finished session.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert _working(app) is not None

        app.action_clear_transcript()
        await pilot.pause()
        assert _working(app) is not None, "a live turn still has to say it is live"
        assert _is_last(app)
        # The cards went with the transcript, so the line falls back to the
        # model call rather than naming a row that is no longer on screen.
        assert _activity(app) == DEFAULT_ACTIVITY

        app.post_message(TurnEnded(aborted=True, error=None))
        await pilot.pause()
        assert _working(app) is None


@pytest.mark.asyncio
async def test_clearing_outside_a_turn_leaves_no_line_behind() -> None:
    """The re-mount is conditional on a turn being live, not on /clear."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_clear_transcript()
        await pilot.pause()
        assert _working(app) is None


@pytest.mark.asyncio
async def test_reloading_mid_turn_lifts_the_line_off_the_dead_turn() -> None:
    """``/reload`` throws the turn away, so its working line goes with it.

    A plain reload keeps the visible ledger, so nothing else unmounts the
    widget: it was left animating a turn that no longer existed, over a session
    that had already been disposed.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert _working(app) is not None

        app._session_factory = lambda: _factory(FakeSession())  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()

        assert _working(app) is None, "the line outlived the turn it describes"
        assert app._working_block is None


@pytest.mark.asyncio
async def test_reloading_mid_turn_leaves_the_next_turn_able_to_say_it_is_working() -> None:
    """The half of the defect that was permanent, and outlives the reload.

    ``_start_working_block`` is idempotent by returning early on a non-None
    ``_working_block``. A reload that unmounted the widget without clearing the
    reference — which is every reload, because ``clear_blocks`` cannot clear a
    reference it does not own — left that guard latched: the replacement
    session, and every session after it, ran its turns with no working line at
    all. Asserting only that the line is gone after the reload misses this
    entirely; the assertion that catches it is the NEXT turn.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert _working(app) is not None

        app._session_factory = lambda: _factory(FakeSession())  # type: ignore[assignment]
        await app._reload_session()
        await pilot.pause()
        assert _working(app) is None

        app.post_message(TurnStarted())
        await pilot.pause()
        assert _working(app) is not None, "the replacement session got no working line"
        assert _is_last(app)
        # What it SAYS is still derived from the preserved ledger — the dead
        # turn's tool card is part of that ledger and a plain /reload keeps it
        # by design — so this pins the mount, not the label.


@pytest.mark.asyncio
async def test_a_session_switch_mid_turn_lifts_the_line_too() -> None:
    """``/resume`` and ``/new`` take the same exit, with the ledger replaced.

    Worse here than on a plain reload, and measured with the guard removed:
    ``clear_blocks`` fires ``_on_transcript_cleared``, which reads a non-None
    ``_working_block`` as "a turn is still live" and mounts a FRESH line into
    the brand-new empty transcript — so the resumed conversation opened with a
    working line animating for a turn that had never started, and the stale
    reference was replaced rather than cleared.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.post_message(TurnStarted())
        app.post_message(_started("c0", "bash", command="sleep 600"))
        await pilot.pause()
        assert _working(app) is not None

        app._session_factory = lambda: _factory(FakeSession())  # type: ignore[assignment]
        # A session SWITCH. `replace_transcript=True` was this call's old
        # spelling; the ledger is now always rebuilt from the new session's
        # own history, so the plain call is the switch.
        await app._reload_session()
        await pilot.pause()

        assert _working(app) is None, "a line was mounted into the new transcript"
        assert app._working_block is None, "the reference survived the transcript"

        app.post_message(TurnStarted())
        await pilot.pause()
        assert _working(app) is not None, "the resumed session got no working line"
        assert _is_last(app)
