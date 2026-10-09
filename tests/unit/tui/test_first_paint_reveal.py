"""Conversation first paint: one painted state per reveal, and the tail kept.

Three seams, one theme — an open that paints an intermediate layout before the
settled one. Each is measured here as the reader meets it, on the real app in a
pilot, and each test is written to FAIL on a tree without its fix:

* ``_prepare_sidebar_session`` authors the incoming transcript offscreen. A
  prepared replay that folded its blocks at the 80-column fallback re-authored
  every one of them when the revealed view was laid out, so a switch painted
  narrow heights, then real ones (S1 at 160x45: 2/3/5-row blocks in the first
  frame, 1/2/3 in the second, three painted states). The fix is the destination
  width, named by the prepare's caller.
* ``_hold_tail_for_reveal`` places a follower at the tail through the layout
  that reveals or fills the transcript, rather than letting the tail scroll run
  a frame later (measured on a resume: the first frame showed the rows 15 lines
  low).
* the prompt-host seam: a card mounting into the dock takes its rows out of the
  transcript's own height, so the frame that carries the card must already
  carry a follower at the tail (``_hold_tail_across_dock_change``).

The instrument is ``App.post_display_hook`` — Textual calls it once per
compositor display, headless included — so "painted states" here means frames
the compositor actually displayed, not awaits a test counted.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: Frame the switch tests run at: a real lane (the transcript's content box is
#: 142 cells against the 80 the fallback folds at), and tall enough to scroll.
FRAME = (160, 45)

#: Enough prose that the transcript is taller than the frame, so a follower has
#: a tail to be held at and a non-follower has somewhere to sit.
LONG_PROSE = " ".join(f"ZEBRA word{index:02} alpha beta gamma delta epsilon" for index in range(40))


def _app() -> OperatorApp:
    return OperatorApp(lambda: _factory(FakeSession()))


def _frame_samples(app: OperatorApp) -> list[dict[str, float]]:
    """Install a per-display sampler: the state of every painted frame.

    ``scroll``/``extent`` are read from the view at paint time, which is the
    only moment at which "was the reader at the tail in THIS frame" is a fact
    rather than an inference.
    """
    samples: list[dict[str, float]] = []
    real_hook = app.post_display_hook

    def hook() -> None:
        try:
            view = app._transcript_view()
            samples.append(
                {
                    "scroll": float(view.scroll_y),
                    "extent": float(view.max_scroll_y),
                    "blocks": float(len(view.blocks())),
                    "height": float(view.outer_size.height),
                }
            )
        except Exception:  # noqa: BLE001 — a frame before the transcript exists is not a state
            pass
        real_hook()

    app.post_display_hook = hook  # type: ignore[method-assign]
    return samples


def _off_tail(samples: list[dict[str, float]], *, after: int = 0) -> list[dict[str, float]]:
    """The sampled frames that left a follower short of the tail."""
    return [
        sample
        for sample in samples[after:]
        if sample["extent"] > 0 and sample["scroll"] < sample["extent"] - 0.5
    ]


async def _seed(pilot, view: TranscriptView, *, rows: int = 6) -> None:
    for index in range(rows):
        block = AssistantBlock()
        block.set_fold_hint(view.scrollable_content_region.width)
        block.update_text(f"{index:02} " + LONG_PROSE)
        view.append_block(block)
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_prepared_replay_authors_its_blocks_at_the_given_width() -> None:
    """The seam the switch's one-state reveal rests on.

    ``PreparedReplay.prepare`` is asked for the destination width and must
    author every block it builds at it. Zero (no destination) still falls back
    to 80, which is why the caller passing the width is the fix rather than a
    constant here.
    """
    from local_operator.tui.session_presentation import PreparedReplay

    prose = " ".join(f"ZEBRA word{index:02} alpha beta gamma delta epsilon" for index in range(40))
    replay = PreparedReplay()
    replay.prepare([_assistant_message(prose)], bound=5, fold_width=126)
    built = [block for block in replay.blocks if isinstance(block, (UserBlock, AssistantBlock))]
    assert built, "the projection built no authored blocks"
    widths = {block.fold_width(0) for block in built}
    assert widths == {126}, f"blocks were authored at {widths} instead of the destination lane"

    fallback = PreparedReplay()
    fallback.prepare([_assistant_message(prose)], bound=5)
    assert {block.fold_width(80) for block in fallback.blocks} == {
        80
    }, "an unnamed destination must keep the 80-column fallback"


def _assistant_message(text: str):
    from local_operator.harness.types import Message, TextContent

    return Message(
        id="switch-row-0001",
        role="assistant",
        content=[TextContent(text=text)],
    )


@pytest.mark.asyncio
async def test_a_following_reveal_paints_its_first_frame_at_the_tail() -> None:
    """The reveal frame carries the reader at the tail, not one frame after it."""
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        samples = _frame_samples(app)
        view = app._transcript_view()
        view.follow_tail()
        await pilot.pause()

        # Counted from here: every frame the app paints from the fill's first
        # layout on must be at the tail.
        start = len(samples)
        app._hold_tail_for_reveal(view)
        await _seed(pilot, view)
        view.follow_tail()
        await _seed(pilot, view)
        assert not _off_tail(
            samples, after=start
        ), f"a painted frame left the tail: {_off_tail(samples, after=start)}"


@pytest.mark.asyncio
async def test_the_hold_does_not_touch_a_reader_who_scrolled_away() -> None:
    """A parked reader is not yanked to the bottom by a reveal's hold."""
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app._transcript_view()
        view.follow_tail()
        await _seed(pilot, view)
        assert view.is_following_tail, "the seed should end at the tail"
        view.scroll_to(y=0, animate=False, immediate=True, force=True)
        view.note_user_scroll(upward=True)
        await pilot.pause()
        assert not view.is_following_tail
        app._hold_tail_for_reveal(view)
        await _seed(pilot, view)
        assert not view.is_following_tail, "the hold re-acquired the tail for a parked reader"
        assert float(view.scroll_y) < float(view.max_scroll_y) - 0.5


@pytest.mark.asyncio
async def test_the_reveal_hold_is_taken_and_released_around_one_layout() -> None:
    """The hold is DRIVEN by the reveal and dropped again after it.

    The painted-frame measurement for this lives in the first-paint bench (a
    reveal that reaches the tail in ``_size_updated`` paints one extra state —
    S1 warm: 2 states on every switch without the hold, 1 with it), because a
    pilot's frame ordering cannot reproduce it: the compositor has already
    settled by the time a test can look. What a test CAN pin is the one thing a
    future edit is most likely to break — that the reveal asks for the hold and
    that the hold does not outlive the frame it was asked for, which would
    re-land the tail on layouts the reader caused by scrolling away.

    Modelled on ``test_parked_source_seam``: mutation testing there showed that
    a suite which calls the setter directly cannot see the app stop calling it.
    """
    app = _app()
    async with app.run_test(size=FRAME) as pilot:
        await pilot.pause()
        view = app._transcript_view()
        await _seed(pilot, view)
        view.follow_tail()
        await pilot.pause()

        from local_operator.tui.widgets.transcript import TranscriptView as _View

        calls: list[bool] = []
        real = _View.hold_tail_through_layout

        def spy(self: _View, hold: bool) -> None:
            calls.append(hold)
            real(self, hold)

        _View.hold_tail_through_layout = spy  # type: ignore[method-assign]
        try:
            app._hold_tail_for_reveal(view)
            assert calls == [True], f"the reveal did not hold before the layout: {calls}"
            held = view._hold_tail_placement  # type: ignore[attr-defined]
            assert held is True, "the hold was released before the frame it belongs to"
            await pilot.pause()
            assert calls == [True, False], f"the hold outlived its frame: {calls}"
        finally:
            _View.hold_tail_through_layout = real  # type: ignore[method-assign]
