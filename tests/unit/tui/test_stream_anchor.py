"""Sticky-bottom scroll anchoring while a message STREAMS.

The reported bug: at the bottom of a long answer, the tail ran off the
viewport as it arrived and could not be read. The cause was that the pin was
applied when a block was APPENDED, and a streaming message appends once and
then grows in place — measured at 80x24, ``scroll_offset.y`` stayed at 0 while
``max_scroll_y`` climbed to 58.

The spec is three states, and each of them is a section below:

* **following** — at the end, every growth keeps it there;
* **released** — the reader scrolled up, and nothing may move them, however
  many deltas arrive;
* **re-acquired** — the reader came back to the end, and following resumes.

Everything below the unit section is driven through a mounted app, because the
question is what the VIEWPORT does, and a widget-level assertion about a flag
would pass happily while the text still scrolled off the screen.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual import events

from local_operator.tui.app import OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.subagent_view import SubagentView
from local_operator.tui.widgets.transcript import (
    GAP_CLASS,
    TAIL_TOLERANCE_ROWS,
    TailAnchor,
    TranscriptView,
    UserBlock,
)

from .conftest import StyledTranscriptApp
from .test_app_pilot import FakeSession, _factory
from .test_band_panels import FakeSession as BandFakeSession
from .test_band_panels import _async_factory, _fake_jobs, _Job
from .test_subagent_view import _call, _text

#: One delta's worth of prose. Long enough that a handful of them overflow an
#: 80x24 screen by a wide margin, which is the only size at which any of this
#: is observable.
SENTENCE = (
    "Scroll anchoring during a stream is a three state rule, and the hard part "
    "is the transition into released, because following itself moves the scroll "
    "offset, so the machine has to be driven by intent. "
)


def _wheel(widget: Any, *, up: bool) -> events.MouseEvent:
    """One notch of wheel, as the driver delivers it.

    Constructed rather than driven through ``Pilot``, which has no scroll
    gesture. Kept in one place so a Textual bump that changes ``MouseEvent``
    breaks here and not in six tests.
    """
    kind = events.MouseScrollUp if up else events.MouseScrollDown
    return kind(
        widget=widget,
        x=1,
        y=1,
        delta_x=0,
        delta_y=-1 if up else 1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


async def _settle(pilot: Any) -> None:
    """Let a gesture finish travelling before asking where it landed.

    A pause FIRST: the keypress has only been posted, so the action that starts
    the animation has not run yet and there would be nothing to wait for.
    """
    await pilot.pause()
    await pilot.wait_for_animation()
    await pilot.pause()


async def _stream(pilot: Any, app: OperatorApp, deltas: int, text: str = "") -> str:
    """Grow ``text`` by ``deltas`` deltas, letting each settle. Returns the text.

    The accumulated text is carried in and out because a delta is the WHOLE
    message so far. Restarting from an empty string would shrink the block,
    which forces the reader to the new bottom and re-acquires the anchor — a
    correct response to content vanishing, and not what any of these tests mean
    to exercise.
    """
    step = text.count(SENTENCE) + 1
    for _ in range(deltas):
        text += f"{step}. {SENTENCE}"
        step += 1
        app.post_message(AssistantDelta(text))
        await pilot.pause()
        await pilot.pause()
    return text


def _gap(view: TranscriptView) -> float:
    """Rows of the tail currently BELOW the bottom of the viewport."""
    return float(view.max_scroll_y) - float(view.scroll_offset.y)


# -- the state machine -------------------------------------------------------


def test_a_fresh_anchor_follows() -> None:
    """An empty surface is at its own end, so the first thing streamed sticks."""
    assert TailAnchor().following is True


def test_a_user_scroll_releases_immediately_not_after_the_frame_settles() -> None:
    """The release cannot wait for a resync.

    Between the wheel event and the frame that applies it, a delta can arrive.
    An anchor still armed in that window scrolls the reader back to the end
    before anyone has measured where they went — which is the burst case, and
    the whole reason the state is set eagerly and only CONFIRMED later.
    """
    anchor = TailAnchor()
    anchor.note_user_scroll()
    assert anchor.following is False


def test_landing_back_at_the_end_re_acquires_and_landing_short_does_not() -> None:
    anchor = TailAnchor()
    anchor.note_user_scroll()

    anchor.resync(at_end=False)
    assert anchor.following is False

    anchor.resync(at_end=True)
    assert anchor.following is True


def test_a_programmatic_scroll_is_not_a_user_scroll() -> None:
    """The distinction is DECLARED, never inferred from the offset.

    Following moves the offset itself, so "the number changed" would release
    the anchor the instant it engaged. Inside the guard the same call that a
    human's wheel makes is ignored.
    """
    anchor = TailAnchor()
    with anchor.programmatic_scroll():
        assert anchor.programmatic is True
        anchor.note_user_scroll()
        assert anchor.following is True
    assert anchor.programmatic is False

    anchor.note_user_scroll()  # same call, outside the guard
    assert anchor.following is False


def test_the_programmatic_guard_nests() -> None:
    """A follow-scroll can settle a layout that scrolls again; the inner exit
    must not re-arm the outer guard and let a stray event through."""
    anchor = TailAnchor()
    with anchor.programmatic_scroll():
        with anchor.programmatic_scroll():
            pass
        assert anchor.programmatic is True
        anchor.note_user_scroll()
    assert anchor.following is True


# -- "at the bottom" has a tolerance -----------------------------------------


@pytest.mark.asyncio
async def test_a_partly_visible_last_line_still_counts_as_the_bottom() -> None:
    """Exact equality is the wrong predicate twice over.

    A last line half off the viewport is the bottom to a human, and the offsets
    are FLOATS — a fractional resting position from a wheel or a scrollbar drag
    makes ``offset == max_scroll_y`` false at a place the reader cannot tell
    apart from the end.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        await _stream(pilot, app, 12)
        cap = float(view.max_scroll_y)
        assert cap > TAIL_TOLERANCE_ROWS, "the fixture must actually overflow"

        view.scroll_to(y=cap - TAIL_TOLERANCE_ROWS, animate=False, immediate=True)
        assert view.is_near_bottom() is True
        assert view.scroll_offset.y != cap, "and not because it landed exactly on it"

        view.scroll_to(y=cap - TAIL_TOLERANCE_ROWS - 1, animate=False, immediate=True)
        assert view.is_near_bottom() is False


@pytest.mark.asyncio
async def test_a_transcript_shorter_than_its_viewport_is_always_at_the_bottom() -> None:
    """Nothing to scroll is not "scrolled away from the end"; an anchor that
    said otherwise would refuse to follow the first screenful of every turn."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        app._append_block(UserBlock("hello"))
        await pilot.pause()
        assert view.max_scroll_y == 0
        assert view.is_near_bottom() is True


# -- following ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_message_growing_in_place_carries_the_viewport_with_it() -> None:
    """THE regression. One block is mounted and then grows; before the anchor
    the offset stayed at 0 while the message ran 58 rows past the bottom."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)

        gaps = []
        text = ""
        for step in range(1, 25):
            text += f"{step}. {SENTENCE}"
            app.post_message(AssistantDelta(text))
            await pilot.pause()
            await pilot.pause()
            gaps.append(_gap(view))

        assert max(gaps) <= TAIL_TOLERANCE_ROWS, f"tail ran off the screen: {gaps}"
        assert float(view.max_scroll_y) > 20, "the fixture must overflow by a lot"
        # And it is ONE block that grew, not twenty-four that were appended —
        # otherwise this passes on the append-time pin it exists to replace.
        assert len(view.blocks()) == 1


@pytest.mark.asyncio
async def test_the_final_authoritative_text_lands_at_the_end_too() -> None:
    """``message_end`` re-renders the whole message in one go, which is the
    single largest height change of the turn and the frame the user is left
    looking at."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        text = await _stream(pilot, app, 10)

        app.post_message(AssistantMessageEnd(text + "\n\nAnd that is the summary."))
        await pilot.pause()
        await pilot.pause()
        assert _gap(view) <= TAIL_TOLERANCE_ROWS


@pytest.mark.asyncio
async def test_a_burst_of_deltas_between_frames_still_ends_at_the_tail() -> None:
    """Deltas arrive at 30 Hz, faster than the screen settles. A scroll issued
    from the delta handler measures the PREVIOUS frame's extent and lands
    permanently short — eight rows short, every burst, when measured."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)

        text = ""
        step = 0
        for _ in range(8):
            for _ in range(4):  # four deltas, no frame in between
                step += 1
                text += f"{step}. {SENTENCE}"
                app.post_message(AssistantDelta(text))
            await pilot.pause()
        await pilot.pause()
        assert _gap(view) <= TAIL_TOLERANCE_ROWS


# -- released ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrolling_up_mid_stream_pins_the_reader_where_they_stopped() -> None:
    """And it stays pinned across many more deltas: releasing on the gesture
    and confirming afterwards is what survives a burst arriving in the window
    between the wheel and the frame that applies it."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        text = await _stream(pilot, app, 14)
        assert float(view.max_scroll_y) > 10, "must overflow before it can be scrolled"

        for _ in range(6):
            view.post_message(_wheel(view, up=True))
        await _settle(pilot)
        assert view.is_following_tail is False
        parked = float(view.scroll_offset.y)
        cap_then = float(view.max_scroll_y)

        await _stream(pilot, app, 12, text)
        assert float(view.scroll_offset.y) == parked, "the deltas moved the reader"
        assert float(view.max_scroll_y) > cap_then, "the fixture must have grown"
        assert view.is_following_tail is False


@pytest.mark.asyncio
async def test_the_keyboard_releases_the_anchor_as_the_wheel_does() -> None:
    """``up`` and ``pageup`` are the transcript's own bindings, and a reader
    using them is reading just as much as one using a mouse."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        text = await _stream(pilot, app, 14)

        view.focus()
        await pilot.press("pageup")
        await _settle(pilot)
        assert view.is_following_tail is False
        parked = float(view.scroll_offset.y)

        await _stream(pilot, app, 8, text)
        assert float(view.scroll_offset.y) == parked


@pytest.mark.asyncio
async def test_following_does_not_release_itself() -> None:
    """The trap the state machine exists for: a follow-scroll changes the
    offset every delta, so any rule that infers "the user scrolled" from the
    offset moving would disarm the anchor the instant it engaged."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)

        offsets = []
        text = ""
        for step in range(1, 16):
            text += f"{step}. {SENTENCE}"
            app.post_message(AssistantDelta(text))
            await pilot.pause()
            await pilot.pause()
            offsets.append(float(view.scroll_offset.y))

        assert view.is_following_tail is True
        assert offsets[-1] > offsets[0], "the offset must really have been moving"


# -- re-acquired -------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrolling_back_to_the_end_resumes_following() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        text = await _stream(pilot, app, 14)

        view.focus()
        await pilot.press("pageup")
        await _settle(pilot)
        assert view.is_following_tail is False

        await pilot.press("end")
        await _settle(pilot)
        assert view.is_following_tail is True

        await _stream(pilot, app, 10, text)
        assert _gap(view) <= TAIL_TOLERANCE_ROWS


@pytest.mark.asyncio
async def test_wheeling_back_down_to_the_tail_re_acquires_without_a_keypress() -> None:
    """The user's own words: "if I scroll to the bottom the stream anchor
    should be acquired". Landing there by wheel is landing there."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app.post_message(AssistantMessageStart())
        await pilot.pause()
        view = app.query_one(TranscriptView)
        text = await _stream(pilot, app, 14)

        for _ in range(4):
            view.post_message(_wheel(view, up=True))
        await _settle(pilot)
        assert view.is_following_tail is False
        parked = float(view.scroll_offset.y)

        # Wheel back DOWN, notch by notch, exactly as a reader would. The last
        # ones are no-ops at the tail, which is itself the case the deferred
        # resync exists for: a gesture that moves nothing must still hand the
        # anchor back.
        for _ in range(8):
            view.post_message(_wheel(view, up=False))
        await _settle(pilot)
        assert float(view.scroll_offset.y) > parked
        assert view.is_following_tail is True

        await _stream(pilot, app, 8, text)
        assert _gap(view) <= TAIL_TOLERANCE_ROWS


# -- the surfaces that share the rule ----------------------------------------


@pytest.mark.asyncio
async def test_the_nested_subagent_transcript_follows_its_own_tail() -> None:
    """The full-page subagent view streams a CHILD's output into a second
    ``TranscriptView``, under the same requirement and — because the rule lives
    on the container — through the same code."""
    job = _Job("sub-1", "audit the ingest path", status="running")
    job.trajectory = [*_text("m1", "Reading the ingest path.")]
    session = BandFakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view("sub-1")
        await pilot.pause()
        view = app.query_one(SubagentView)
        body = view._body

        for step in range(2, 22):
            job.trajectory.append(_call(f"c{step}", "bash", command=f"pytest -q # {step}"))
            app._refresh_band()
            await pilot.pause()
            await pilot.pause()

        assert float(body.max_scroll_y) > 10, "the child's page must overflow"
        assert _gap(body) <= TAIL_TOLERANCE_ROWS
        assert body.is_following_tail is True


@pytest.mark.asyncio
async def test_paging_the_subagent_body_by_its_hint_arrow_releases_the_anchor() -> None:
    """The ↑ affordance is a click on the page's FOOTER, so no input handler of
    the body's own sees it. Unannounced, it would be indistinguishable from the
    anchor moving itself, and the child's output would drag the reader back
    down the moment they tried to read back through it."""
    job = _Job("sub-1", "audit the ingest path", status="running")
    job.trajectory = [*_text("m1", "Reading the ingest path.")]
    session = BandFakeSession()
    session.jobs = _fake_jobs(job)
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._refresh_band()
        await pilot.pause()
        app._open_subagent_view("sub-1")
        await pilot.pause()
        view = app.query_one(SubagentView)
        body = view._body

        for step in range(2, 22):
            job.trajectory.append(_call(f"c{step}", "bash", command=f"pytest -q # {step}"))
            app._refresh_band()
            await pilot.pause()
            await pilot.pause()

        view._scroll_body(down=False)
        await pilot.pause()
        await pilot.pause()
        assert body.is_following_tail is False
        parked = float(body.scroll_offset.y)

        for step in range(22, 30):
            job.trajectory.append(_call(f"c{step}", "bash", command=f"pytest -q # {step}"))
            app._refresh_band()
            await pilot.pause()
            await pilot.pause()
        assert float(body.scroll_offset.y) == parked


# --- history prepend: the anchor survives gap settlement --------------------


def _prose(text: str) -> AssistantBlock:
    """An assistant block whose source text is set directly.

    ``spans_multiple_rows`` is answered from ``_full_text`` rather than by
    rendering, so this is the whole of what adaptive spacing reads.
    """
    block = AssistantBlock()
    block._full_text = text
    return block


@pytest.mark.asyncio
async def test_a_history_prepend_holds_the_readers_row_through_gap_settlement() -> None:
    """The reader's row must not move when older history is mounted above it.

    This pins the mechanism that made history paging drift, not merely that
    *some* restoration happens. ``prepend_blocks`` mounts its batch while the
    blocks are still unmounted, so ``spans_multiple_rows()`` has no width and
    falls back to 80 columns; ``_settle_gaps`` then re-decides those gaps
    against the real width on a LATER layout pass. Re-deciding removes margin
    rows ABOVE the anchor, so any offset derived from ``virtual_size`` growth
    sampled before that pass is stale by exactly the rows the settle reclaimed
    — the reader came to rest up to 2 rows off their line.

    The provisional gap classes are set here explicitly rather than raced for.
    Waiting for the layout passes to interleave the wrong way reproduces the
    drift only intermittently (measured 2/10 through the subagent view), which
    is the timing bet this suite exists to remove. Setting the class states the
    same precondition the 80-column fallback produces, so the settle-time
    correction — and therefore the regression — is deterministic.

    Two-sided by construction: with the fix the gap is held exactly, and with
    the pre-fix growth formula restored it drifts by the reclaimed 2 rows.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 20)) as pilot:
        view = app.query_one(TranscriptView)
        for index in range(60):
            view.append_block(_prose(f"tail {index}"))
        for _ in range(20):
            await pilot.pause()

        # Mid-history, so the anchor is NOT flush with the top of the viewport:
        # a gap of zero cannot show a drift of two rows.
        view.scroll_to(y=25, animate=False)
        for _ in range(20):
            await pilot.pause()

        anchor = view._blocks[0]
        before_gap = anchor.virtual_region.y - view.scroll_y

        older = [_prose(f"older {index}") for index in range(40)]
        # The provisional decision an unmounted batch makes for itself, which
        # `_settle_gaps` then reclaims once real widths exist.
        for block in older[:2]:
            block.add_class(GAP_CLASS)

        view.prepend_blocks(older)
        for _ in range(60):
            await pilot.pause()

        # The settle really did reclaim the provisional rows: without this the
        # assertion below could pass on a batch that never moved.
        assert not any(block.has_class(GAP_CLASS) for block in older)
        assert anchor.virtual_region.y - view.scroll_y == before_gap
