"""Conversation first paint: one painted state per reveal, and the tail kept.

Two seams, one theme — a reveal that paints an intermediate layout before the
settled one. Each is measured here as the reader meets it, on the real app in a
pilot, and each test is written to FAIL on a tree without its fix:

* ``_prepare_sidebar_session`` authors the incoming transcript offscreen. A
  prepared replay that folded its blocks at the 80-column fallback re-authored
  every one of them when the revealed view was laid out, so a switch painted
  narrow heights, then real ones (S1 at 160x45: 2/3/5-row blocks in the first
  frame, 1/2/3 in the second, three painted states). The fix is the destination
  width, named by the prepare's caller.
* ``_hold_tail_for_reveal`` places a FOLLOWER at the tail through the layout
  that reveals or fills the transcript, rather than letting the tail scroll run
  a frame later (measured on a switch: the first frame showed the rows 15 lines
  low, and on a resume the same). The saved-position cell below is its
  complement: a reveal that is going back to a reader's own anchor must NOT be
  dragged to the tail.

The instrument is ``App.post_display_hook`` — Textual calls it once per
compositor display, headless included — so "painted states" here means frames
the compositor actually displayed, not awaits a test counted.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.transcript import TranscriptView, UserBlock
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over
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


@asynccontextmanager
async def _viewer(tmp_path, name: str):
    """A real owner runtime plus a real ``AttachedSession`` viewer over it.

    The saved-position cell needs the commit seam to run for real: a mocked
    session cannot reach it (``_commit_sidebar_session`` refuses to switch away
    from anything that is not a viewer), and the behaviour under test —
    "which frame does the reveal paint" — is about geometry only a laid-out
    view has. Mirrors ``test_parked_source_seam._remote``.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    rows = [
        Message(
            id=f"switch-row-{index:04}", role="assistant", content=[TextContent(text=LONG_PROSE)]
        )
        for index in range(6)
    ]
    await seed_transcript(directory, rows)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await AttachedSession.connect(
        server._record,
        directory.name,
        config_dir=config,
        takeover_factory=_never_take_over,
        display_window=True,
    )
    try:
        yield remote
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


async def _switch_to(app, pilot, remote, *, saved_anchor: bool) -> list[object]:
    """Prepare and commit ``remote`` as a saved view; return the hold calls."""
    source = SessionInteraction(remote)
    app._sidebar_sources[remote.session_id] = source
    app._interactions[id(remote)] = source

    async def lease(session_id, *, speculative=False):
        source.preparations += 1
        return source

    app._lease_sidebar_source = lease  # type: ignore[method-assign]
    source.display_only = True
    source.draft.following_tail = not saved_anchor
    source.draft.scroll_anchor_id = ""
    source.draft.scroll_anchor_part = 0
    source.draft.scroll_offset = 0

    prepared = await app._prepare_sidebar_session(remote.session_id)
    view = prepared[1].replay.view
    if saved_anchor:
        anchor = next(
            (block for block in view.blocks() if block.navigation_anchor_id),
            None,
        )
        assert anchor is not None, "the prepared view has no anchor to save a position on"
        source.draft.scroll_anchor_id = anchor.navigation_anchor_id

    calls: list[object] = []
    real = OperatorApp._hold_tail_for_reveal

    def spy(target):  # noqa: ANN001
        calls.append(target)
        return real(target)

    OperatorApp._hold_tail_for_reveal = staticmethod(spy)  # type: ignore[method-assign]
    try:
        app._commit_sidebar_session(remote.session_id, prepared, app._sidebar_navigation.generation)
        for _ in range(8):
            await pilot.pause()
    finally:
        # `staticmethod` (both here and at the other restore): the helper IS a
        # staticmethod, and a plain reassignment would rebind it as an instance
        # method, so the next production call would pass ``self`` as the view.
        OperatorApp._hold_tail_for_reveal = staticmethod(real)  # type: ignore[method-assign]
        if source.controller is not None:
            source.controller.set_parked(True)
        app._interactions.pop(id(remote), None)
    return calls


@pytest.mark.asyncio
async def test_a_saved_position_is_not_dragged_to_the_tail_by_the_reveal(tmp_path) -> None:
    """The hold is for a FOLLOWER; a saved position is the reveal's own target.

    The two disagree on this path and the reviewer found the gap in a code
    trace: ``_prepare_sidebar_session`` calls ``follow_tail()`` unconditionally
    (a parked view has no saved geometry to hold), so a ``display_only`` source
    with a saved anchor arrives at the reveal with ``following`` armed. Landing
    the tail there paints the END of a conversation whose reader is in the
    middle, and ``restore_revealed_anchor`` walks it back on the next frame —
    the two painted states the operator reported, on the one switch shape the
    bench cannot open (every bench shape opens a live source).

    Without the guard this test fails on the first assertion: the hold is called
    for the saved-position cell.
    """
    async with _viewer(tmp_path, "home") as home, _viewer(tmp_path, "saved") as saved:

        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break

            saved_calls = await _switch_to(app, pilot, saved, saved_anchor=True)
            assert saved_calls == [], (
                "a saved-position reveal was held to the TAIL: the frame the user "
                "sees is the end of a conversation they left mid-way"
            )

            tail_calls = await _switch_to(app, pilot, home, saved_anchor=False)
            # At least once, not exactly once: a switch has TWO commits (the
            # navigation's and the connect leg's, ``_connect_sidebar_source``),
            # and each reveal is entitled to its own hold.
            assert tail_calls, "a follower's reveal was never held to the tail"


@pytest.mark.asyncio
async def test_a_short_resume_holds_the_tail_for_its_first_frame(tmp_path) -> None:
    """The LAUNCH call site, pinned by a spy.

    The reveal helper has two callers and each needs its own pin: the switch
    commit is covered by the saved-position test above, and this is the resume's
    short branch — a conversation whose whole history fits the render window, so
    the viewport-first split (which holds through its backfill page) does not
    apply and the first frame would otherwise be placed at scroll 0 and moved to
    the tail a frame later. Without the call this fails on the assertion.
    """
    async with _viewer(tmp_path, "short") as viewer:
        session = viewer

        async def factory():
            return session

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is not None:
                    break

            calls: list[object] = []
            real = OperatorApp._hold_tail_for_reveal

            def spy(target):  # noqa: ANN001
                calls.append(target)
                return real(target)

            OperatorApp._hold_tail_for_reveal = staticmethod(spy)  # type: ignore[method-assign]
            try:
                app._render_resumed_history(session)
                for _ in range(4):
                    await pilot.pause()
            finally:
                # `staticmethod`: the helper IS a staticmethod, and a plain
                # reassignment would rebind it as an instance method, so the
                # next production call would pass ``self`` as the view.
                OperatorApp._hold_tail_for_reveal = staticmethod(  # type: ignore[method-assign]
                    real
                )

            assert calls, "the resume's first frame was not held to the tail"


def test_the_tail_history_notice_takes_the_width_it_is_given() -> None:
    """F5: the twins are built the same way.

    ``OlderHistoryNotice`` (head) has taken a ``fold_width`` since the prepared
    replay learned to name one; its twin at the other end hardcoded its own
    construction, so the pair disagreed about how a block learns its width. No
    height consequence at today's widths — the copy is 30 cells, one row at 80
    and at 142 — but the seam is the point: a block that folds at a width its
    caller did not name is exactly the defect the argument exists to close.
    """
    from local_operator.tui.session_presentation import HistoryPageNotice

    assert HistoryPageNotice(fold_width=142).fold_width(0) == 142
    assert HistoryPageNotice().fold_width(80) == 80
