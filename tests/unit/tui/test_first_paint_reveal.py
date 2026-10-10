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
from local_operator.tui.app import RESUME_RENDER_MESSAGES, OperatorApp
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


def _prompt_host_sample(app: OperatorApp) -> dict[str, float]:
    """Where the dock's prompt host is, in the frame being sampled.

    F2/Q2/D4 are all about the host's rows: the card is a dock child, so its
    height comes out of the transcript's, and a frame drawn before the host has
    authored that height puts the card's own rows off-screen. `outer_size` and
    `region` are both recorded because the finding was about the difference.
    """
    try:
        host = app.query_one("#prompt-host")
        return {
            "host_y": float(host.region.y),
            "host_h": float(host.region.height),
            "screen_h": float(app.size.height),
        }
    except Exception:  # noqa: BLE001 — no host yet is not a state
        return {"host_y": -1.0, "host_h": -1.0, "screen_h": float(app.size.height)}


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
                    # The transcript's OWN region, which is what the reader's rows
                    # are clipped to: `outer_size` is the widget's size and can
                    # differ from its region for a frame, which is exactly the
                    # distinction F2 (round 1) asked this sampler to keep.
                    "region": float(view.region.height),
                    **_prompt_host_sample(app),
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
async def _viewer(tmp_path, name: str, *, rows: int = 6):
    """``_viewer_session``, minus the handle: what most cells here need."""
    async with _viewer_session(tmp_path, name, rows=rows) as (remote, _handle):
        yield remote


@asynccontextmanager
async def _viewer_session(tmp_path, name: str, *, rows: int = 6):
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
    seed = [
        Message(
            id=f"switch-row-{index:04}", role="assistant", content=[TextContent(text=LONG_PROSE)]
        )
        for index in range(rows)
    ]
    await seed_transcript(directory, seed)
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
        yield remote, handle
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
async def test_a_short_resume_holds_the_tail_for_its_whole_render(tmp_path) -> None:
    """The LAUNCH call site, pinned on the HOLD rather than on a call.

    The reveal helpers have two callers and each needs its own pin: the switch
    commit is covered by the saved-position test above, and this is the resume —
    a conversation whose history fits the render window, where the first frame
    would otherwise be placed at scroll 0 and moved to the tail a frame later.

    BEHAVIOUR, not the call: review round 1's F3 showed the resume's frames
    leaving the tail because the hold was released after one refresh while the
    fill still had mounts to make, and a spy on the helper cannot see that. So
    this asserts what the reader gets — every painted frame on the tail from the
    render onward — plus the two ends of the hold itself: armed when the render
    puts its rows up, released once the fill has nothing left to add.
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

            samples = _frame_samples(app)
            start = len(samples)
            app._render_resumed_history(session)
            view = app._transcript_view()
            assert view._hold_tail_placement, (
                "the resume put its rows up without holding the tail: the first frame is "
                "placed at scroll 0 and moved a frame later"
            )
            for _ in range(8):
                await pilot.pause()
            assert not _off_tail(samples, after=start), (
                f"a painted frame left the tail during the render: "
                f"{_off_tail(samples, after=start)}"
            )
            assert not view._hold_tail_placement, (
                "the render's hold outlived the render: a later extent change would drag a "
                "reader who has scrolled away"
            )


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


@pytest.mark.asyncio
async def test_the_launch_projects_the_whole_window_in_one_pass(tmp_path) -> None:
    """(b) as a MECHANISM: one projection for the whole window, no backfill page.

    The user-visible fact — "the launch paints ONE state" — is the bench's, and
    its A/B (2 states -> 1 on S2/S3/S5/S6) lives in the bench run, because a
    compositor-race unit test cannot reliably catch a frame that a fast boot
    coalesces. What a unit test CAN pin exactly is the shape of the render: the
    projection is called ONCE, for the whole render window, and no older page is
    mounted during the launch. On a tree that splits again, the call carries the
    screenful bound and the backfill page runs — this fails on both.
    """
    async with _viewer(tmp_path, "launch", rows=40) as viewer:

        async def factory():
            return viewer

        app = OperatorApp(factory)
        calls: list[dict[str, object]] = []
        pages: list[dict[str, object]] = []
        real_project = OperatorApp._project_settled_rows
        real_page = OperatorApp._mount_older_resume_page

        def project(_self, history, **kwargs):  # noqa: ANN001
            calls.append(kwargs)
            return real_project(_self, history, **kwargs)

        def page(_self, *args, **kwargs):  # noqa: ANN001
            pages.append(kwargs)
            return real_page(_self, *args, **kwargs)

        OperatorApp._project_settled_rows = project  # type: ignore[method-assign]
        OperatorApp._mount_older_resume_page = page  # type: ignore[method-assign]
        try:
            async with app.run_test(size=FRAME) as pilot:
                for _ in range(200):
                    await pilot.pause()
                    view = app._transcript_view()
                    if view is not None and view.blocks():
                        break
                for _ in range(6):
                    await pilot.pause()
        finally:
            OperatorApp._project_settled_rows = real_project  # type: ignore[method-assign]
            OperatorApp._mount_older_resume_page = real_page  # type: ignore[method-assign]

    assert calls, "the resume never projected a window"
    assert [call.get("bound") for call in calls] == [RESUME_RENDER_MESSAGES], calls
    assert pages == [], f"a page was mounted during the launch: {pages}"


@pytest.mark.asyncio
async def test_a_gate_the_app_already_holds_is_in_the_reveal_frame(tmp_path) -> None:
    """(a): a known gate's card is part of the frame that reveals its rows.

    The card is a dock child, so its rows come out of the transcript's own
    height. Built a turn AFTER the reveal it takes those rows away then, and
    every visible row moves up by them — measured on S4 at 160x45, the
    transcript goes 38 -> 23 rows and the card's own frame is a second painted
    state. For a source whose snapshot the app ALREADY holds — which is what a
    switch back to a live conversation has, and what the paint-first resume's
    attach-behind produces — the gate is known inside the commit's own
    synchronous section, so the card is built and mounted there
    (``_prearm_known_gate``) and the session's ladder adopts the same card when
    it runs.

    Without the pre-arm this fails on the first content frame: no card, and the
    taller transcript — the frame the reader sees jump.
    """
    async with (
        _viewer_session(tmp_path, "alpha") as (alpha, alpha_handle),
        _viewer_session(tmp_path, "beta") as (beta, _beta_handle),
    ):

        async def factory():
            return alpha

        app = OperatorApp(factory)
        gate_task: asyncio.Task[object] | None = None
        async with app.run_test(size=FRAME) as pilot:
            for _ in range(200):
                await pilot.pause()
                if app._session is alpha:
                    break
            beta_source = SessionInteraction(beta)
            app._sidebar_sources[alpha.session_id] = app._interaction
            app._sidebar_sources[beta.session_id] = beta_source
            app._interactions[id(beta)] = beta_source

            async def lease(session_id, *, speculative=False):
                source = app._sidebar_sources[session_id]
                source.preparations += 1
                return source

            app._lease_sidebar_source = lease  # type: ignore[method-assign]
            # Without this an approval auto-answers and never mounts a card.
            app._set_approve_all(False)
            alpha_handle._auto_approve = False

            async def visit(session_id: str) -> None:
                prepared = await app._prepare_sidebar_session(session_id)
                app._commit_sidebar_session(
                    session_id, prepared, app._sidebar_navigation.generation
                )
                for _ in range(12):
                    await pilot.pause()

            gate_task = asyncio.create_task(alpha_handle._approval_gate("write", "Save one record"))
            for _ in range(200):
                await pilot.pause()
                if app._approval is not None:
                    break
            assert app._approval is not None, "the gate never mounted on alpha"

            await visit(beta.session_id)
            assert app._approval is None, "the outgoing session's card is still on screen"

            samples: list[dict[str, float]] = []
            real_hook = app.post_display_hook

            def hook() -> None:
                try:
                    view = app._transcript_view()
                    host = app.query_one("#prompt-host")
                    target = getattr(getattr(app, "_session", None), "session_id", "")
                    samples.append(
                        {
                            "target": 1.0 if target == alpha.session_id else 0.0,
                            "blocks": float(len(view.blocks())),
                            "height": float(view.outer_size.height),
                            # The transcript's OWN region and the dock's host: a
                            # card is a dock child, so its rows come out of the
                            # transcript's region, and a frame drawn before the host
                            # has authored that height pushes the card's own rows off
                            # the screen (F2/Q2/D4).
                            "region": float(view.region.height),
                            "host_y": float(host.region.y),
                            "host_h": float(host.region.height),
                            "screen_h": float(app.size.height),
                            "prompt": float(bool(host.display and host.children)),
                        }
                    )
                except Exception:  # noqa: BLE001 — a frame before the transcript exists
                    pass
                real_hook()

            app.post_display_hook = hook  # type: ignore[method-assign]
            try:
                prepared = await app._prepare_sidebar_session(alpha.session_id)
                app._commit_sidebar_session(
                    alpha.session_id, prepared, app._sidebar_navigation.generation
                )
                # THE DISCRIMINATING MOMENT. The commit is one synchronous
                # section, and the ladder's task cannot run inside it, so at this
                # instant the card is either mounted BY the commit (the pre-arm)
                # or not mounted at all — a later turn would be the frame the
                # reader sees the transcript move on. Without the pre-arm this
                # fails here with `_approval is None`.
                assert app._approval is not None, (
                    "the reveal's own commit did not mount the known gate's card: it "
                    "arrives on a later turn, taking its rows out of the transcript then"
                )
                host = app.query_one("#prompt-host")
                assert host.display and host.children, "the card is registered but not mounted"
                for _ in range(12):
                    await pilot.pause()
            finally:
                app.post_display_hook = real_hook  # type: ignore[method-assign]

            assert gate_task is not None
            gate_task.cancel()

    # Only the INCOMING conversation's frames: the outgoing one paints until the
    # swap, and its card-less frame is not what this test is about.
    content = [
        sample
        for sample in samples
        if sample["target"] == 1.0 and sample["blocks"] > 0 and sample["height"] > 0
    ]
    assert content, f"the return leg painted no rows: {samples}"
    first = content[0]
    assert first["prompt"] == 1, (
        "the frame that revealed the rows carried no card, so its rows come out of "
        "the transcript's height on a LATER frame",
        content[:4],
    )
    # THE CARD'S ROWS ARE RESERVED IN THAT SAME FRAME (`arrange`-time settle, see
    # `OperatorApp._settle_dock_rows_before_reveal`), so the transcript's own
    # region is the settled one immediately and the host sits inside the screen.
    # Before the reservation, measured at this size: region 38, host y=39 with 15
    # rows on a 45-row screen (clipped), settling to 23 at +147 ms with the scroll
    # moving by exactly the card's height — review round 1's F2, QA's Q2, design's
    # D4. `region`, not `outer_size`: the reviewer's point, and it is the region the
    # reader's rows are clipped to.
    settled = content[-1]
    # THE ACCEPTANCE CRITERION FOR THIS FRAME, asserted as an equality and
    # deliberately NOT one-sided. The frame that first carries the card must
    # already have taken the CARD'S OWN ROWS out of the transcript: the
    # transcript's region and the dock's own height both equal to their settled
    # values. An earlier revision asserted `region >= settled`, which passes on
    # both of the defects this exists to catch, because both of them give the
    # transcript MORE rows than it settles with:
    #
    # * the round-1 shape — region `h=38` against a settled `23`, the dock at its
    #   3 rows of chrome instead of 15 (the card added BELOW the transcript, then
    #   a second state at +147 ms taking the rows back);
    # * the partial reserve — region `35`, the card in the dock with only part of
    #   its height authored.
    #
    # Both are measured shapes (review rounds 2 and 3), so this pin must be RED on
    # either, on the head without the reservation, and on a head whose reservation
    # only half-lands.
    assert first["region"] == settled["region"], (
        "the frame that carries the card did not take the card's rows out of the "
        "transcript: the region is not the settled one (round-1 defect 38 vs 23; "
        "partial reserve 35 vs 23)",
        first["region"],
        settled["region"],
        content[:4],
    )
    assert first["host_h"] == settled["host_h"], (
        "the dock's own height is not the settled one in the frame that carries "
        "the card: the card's rows are not reserved there",
        first["host_h"],
        settled["host_h"],
        content[:4],
    )
    assert first["host_y"] + first["host_h"] <= first["screen_h"] + 0.5, (
        "the card itself was pushed off the screen by its own late reservation",
        first["host_y"],
        first["host_h"],
        first["screen_h"],
    )
    # WHAT THIS PINS: the card's rows are in the frame that carries the card, as
    # one equality over the geometry a reader sees — not a claim about the code
    # path. `_settle_dock_rows_before_reveal` runs its pass while the card is still
    # composing (instrumented: `card_mounted=False` in every run the reviewer and
    # this lane took), so the pass cannot be credited with the reservation; what
    # decides the frame is the card's own composition landing before the reveal is
    # painted. The rate therefore belongs to the pin, not to a comment: at matched
    # load the reviewer measured 16 of 16 and 21 of 23 in the two arms, with no
    # measurable gain from the in-turn mount helper, which is why that helper is
    # gone from this branch — a mechanism that does not move the number is not kept
    # for its name.
