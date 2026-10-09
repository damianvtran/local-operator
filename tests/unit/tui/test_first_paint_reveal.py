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


def _frame_samples(app: OperatorApp, session=None) -> list[dict[str, float]]:
    """Install a per-display sampler: the state of every painted frame.

    ``scroll``/``extent`` are read from the view at paint time, which is the
    only moment at which "was the reader at the tail in THIS frame" is a fact
    rather than an inference. Pass ``session`` to also record whether that
    conversation was in front of the reader and whether its view was revealed.
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
                    # "target": painted with THAT conversation in front of
                    # the reader (a switch's samples begin on the outgoing one,
                    # whose reader is wherever they were); "parked": the incoming
                    # view is still offscreen, so this is a composition frame
                    # rather than the reader's own.
                    "target": float(session is not None and app._session is session),
                    "parked": float(bool(view.disabled)),
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


async def _switch_to(
    app, pilot, remote, *, saved_anchor: bool, anchor_index: int = 0, anchor_id: str = ""
) -> list[object]:
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
    # A PRODUCTION DRAFT IS ALREADY SAVED WHEN THE LEASE IS PREPARED: the sidebar
    # captures it on the way out and the interaction carries it back in. A caller
    # that knows the anchor id up front passes it here, because whether the
    # PREPARE knows the reader's position decides whether it arms a position or
    # calls `follow_tail()` — and only the armed path reaches the arrival test's
    # subject.
    source.draft.scroll_anchor_id = anchor_id
    source.draft.scroll_anchor_part = 0
    source.draft.scroll_offset = 0

    prepared = await app._prepare_sidebar_session(remote.session_id)
    view = prepared[1].replay.view
    if saved_anchor and not anchor_id:
        anchors = [block for block in view.blocks() if block.navigation_anchor_id]
        anchor = anchors[anchor_index] if len(anchors) > anchor_index else None
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


def test_the_tail_history_notice_takes_the_width_the_prepare_named() -> None:
    """F5, pinned on what ``prepare`` BUILDS rather than on the class (R2-F2).

    The twins disagreeing about how a block learns its width is the seam this
    exists to close, and the first version of this pin could not see it: it
    constructed ``HistoryPageNotice`` itself, so it passed with
    ``session_presentation.py``'s OWN construction reverted — a pin that survives
    the revert of the line it names is not a pin. This one seeds the pending tail
    the projection is handed and measures the notice THAT call built.

    No height consequence at today's widths (the copy is 30 cells: one row at 80
    and at 142) — the point is the seam, exactly as F5 was answered in round 1.
    """
    from local_operator.tui.session_presentation import (
        HistoryPageNotice,
        PreparedReplay,
    )

    prose = " ".join(f"ZEBRA word{index:02} alpha beta gamma delta epsilon" for index in range(40))
    replay = PreparedReplay()
    replay._resume_pending_tail = [_assistant_message(prose)]
    replay.prepare([_assistant_message(prose)], bound=1, fold_width=126)
    notices = [block for block in replay.blocks if isinstance(block, HistoryPageNotice)]
    assert notices, "the prepare built no tail notice to measure"
    named = notices[0].fold_width(0)
    assert named == 126, (
        "the tail notice was built at its own default, not at the width the caller named",
        named,
    )


@pytest.mark.asyncio
async def test_a_preview_reveal_lands_on_its_saved_anchor_in_one_state(tmp_path) -> None:
    """Q8/R2-F1: a saved mid-conversation position, taken in ONE painted state.

    The shape is a ``display_only`` source — what a first visit, or a return
    after the idle sweep released the lease, commits as. ``_prepare_sidebar_session``
    skips the layout wait for it, so at commit time every block's region is still
    zero: the commit-time restore has no geometry to place the anchor with, and
    the two writers that ran instead both aimed at the END of the conversation
    (the prepare's unconditional ``follow_tail()`` and the mount batch's
    ``_land_on_tail``, which reads an unmeasured view as "at the tail"). QA round
    2 measured the result on the real path: state-104ms at the TOP, state-146ms
    at the tail, settled at the saved position, and the anchor reached in 5 of 7
    runs.

    So the assertions are the accept criterion, over SEVEN fresh targets in one
    pilot (a fresh saved-preview facade each time, so every one of them commits
    through the unmeasured path):

    * the first painted frame that carries content is already the settled one;
    * no frame sits anywhere but the saved position — not the top, not the tail;
    * all seven land, which is the 7/7 that the single-run version could not say.

    Fails on 9c4d6b96c5 with `[41, 0, 39, 41, 41, 41]` (the top frame) and on the
    pre-Q8-fix tree with the tail frame instead.
    """
    async with _viewer(tmp_path, "preview-home") as home:

        async def factory():
            return home

        app = OperatorApp(factory)
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is home:
                    break
            landed = 0
            for index in range(7):
                async with _saved_preview(tmp_path, f"preview-target-{index}") as saved:
                    samples = _frame_samples(app, session=saved)
                    before = len(samples)
                    # The fixture's own message ids are the anchor ids
                    # (`navigation_anchor_id` is the projected message's), and
                    # the draft has to carry the position BEFORE the prepare.
                    await _switch_to(
                        app,
                        pilot,
                        saved,
                        saved_anchor=True,
                        anchor_index=3,
                        anchor_id=f"saved-row-{3:04}",
                    )

                    content = [
                        sample
                        for sample in samples[before:]
                        if sample["target"] == 1.0
                        and sample["parked"] == 0.0
                        and sample["blocks"] > 0
                        and sample["extent"] > 0
                    ]
                    assert content, f"open {index}: no painted frame carried content"
                    settled = content[-1]["scroll"]
                    assert settled > 0, (
                        f"open {index}: the fixture is too short to need a scroll",
                        settled,
                        content[-1]["extent"],
                    )
                    assert min(sample["scroll"] for sample in content) > 0, (
                        f"open {index}: a painted frame sat at the TOP of a conversation whose "
                        "saved anchor is rows further down",
                        [round(sample["scroll"]) for sample in content[:6]],
                    )
                    assert content[0]["scroll"] == settled, (
                        f"open {index}: the first content frame was not the settled position — "
                        "the viewport was walked onto the anchor (or off it) over extra frames",
                        [round(sample["scroll"]) for sample in content[:6]],
                        round(settled),
                    )
                    view = app._transcript_view()
                    assert settled < view.max_scroll_y - 0.5, (
                        f"open {index}: the landed frame is the TAIL, not the saved position",
                        settled,
                        view.max_scroll_y,
                    )
                    source = app._sidebar_sources[saved.session_id]
                    anchor = next(
                        (
                            block
                            for block in view.blocks()
                            if block.navigation_anchor_id == source.draft.scroll_anchor_id
                        ),
                        None,
                    )
                    assert anchor is not None, f"open {index}: the saved anchor is not in the view"
                    # "Landed on the anchor" is the anchor's OWN top row at the top
                    # of the visible content: `region.y` is a container coordinate,
                    # so the comparison is against `content_region.y` and not
                    # against `scroll_y` (the same measurement QA's own matrix
                    # reports as `anchor_top`).
                    off_by = abs(anchor.region.y - view.content_region.y)
                    assert off_by <= 1.5, (
                        f"open {index}: the view did not land on its saved anchor",
                        round(off_by, 1),
                        settled,
                        view.max_scroll_y,
                    )
                    landed += 1
            assert landed == 7, f"the saved position was reached in {landed} of 7 opens"


@asynccontextmanager
async def _saved_preview(tmp_path, name: str, *, rows: int = 6):
    """A viewer over a session with NO runtime: the ``display_only`` shape.

    ``saved_preview`` is the facade ``_lease_sidebar_source``'s saved branch
    builds, so this is what a first visit — or a return after the idle sweep
    released the lease — commits, and the one whose prepare path skips the
    layout wait.
    """
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    seed = [
        Message(
            id=f"saved-row-{index:04}", role="assistant", content=[TextContent(text=LONG_PROSE)]
        )
        for index in range(rows)
    ]
    await seed_transcript(directory, seed)
    remote = await AttachedSession.saved_preview(
        directory.name,
        config_dir=config,
        cwd=str(tmp_path),
        takeover_factory=_never_take_over,
    )
    try:
        yield remote
    finally:
        await remote.dispose()
