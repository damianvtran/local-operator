"""Rendered history contracts, including real owner RPC and every painted frame.

These tests use synthetic sessions only. The suite launcher isolates HOME/config
and removes all CMUX_* before imports, as required for any OperatorApp pilot.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from textual.events import MouseScrollUp

from local_operator.harness.types import Message, TextContent
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import OperatorApp, _PagingLease
from local_operator.tui.session_interaction import SessionInteraction
from local_operator.tui.session_presentation import OlderHistoryNotice
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.transcript import GAP_CLASS, NoticeBlock
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def history(count: int = 600, hidden: int = 0) -> list[Message]:
    return [
        Message(
            id=f"synthetic-row-{i:04}",
            role="assistant",
            content=(
                [TextContent(text=f"Saved row {i:04}")] if i < count or i == count + hidden else []
            ),
            stop_reason="stop",
        )
        for i in range(count + hidden + 1)
    ]


async def settled(app, pilot) -> None:
    for _ in range(200):
        await pilot.pause()
        if (
            app._session is not None
            and app._transcript_view().blocks()
            and not app._resume_paging
            and not app._resume_fill_active
            and not app._resume_check_pending
        ):
            await pilot.pause()
            return
    raise AssertionError("history presentation never settled")


@asynccontextmanager
async def remote_session(tmp_path: Path, rows: list[Message], name: str = "paging"):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / f"synthetic-{name}"
    await seed_transcript(directory, rows)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await RemoteSession.connect(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "height,hidden", [(40, 0), (50, 0), (100, 0), (40, 500), (100, 500), (100, 1400)]
)
async def test_resume_reserves_rendered_viewport_not_raw_messages(height, hidden) -> None:
    session = FakeSession()
    session._history = history(hidden=hidden)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, height)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        notice = app._resume_head_notice
        assert notice is not None
        assert view.scroll_y - notice.virtual_region.bottom >= view.container_size.height
        assert view.virtual_size.height >= 2 * view.container_size.height
        assert app._resume_pending_head  # reserve is not eager whole-history replay
        assert not app._resume_in_zone  # initial geometry does not invent input
        before = len(app._resume_pending_head)
        for _ in range(8):
            app._transcript_scrolled(None)
            await pilot.pause()
        assert len(app._resume_pending_head) == before


@pytest.mark.asyncio
async def test_top_prefix_does_not_anchor_insert_and_every_paint_stays_still() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        retained = view.blocks()[1]
        gap = retained.region.y - view.content_region.y
        frames = []
        refresh = app.screen._compositor_refresh

        def capture() -> None:
            refresh()
            frames.append(retained.region.y - view.content_region.y)

        app.screen._compositor_refresh = capture
        before = len(app._resume_pending_head)
        try:
            view.post_message(MouseScrollUp(view, 1, 1, 0, -1, 0, False, False, False))
            await settled(app, pilot)
        finally:
            app.screen._compositor_refresh = refresh
        assert frames and set(frames) == {gap}
        assert len(app._resume_pending_head) < before
        ids = [block.navigation_anchor_id for block in view.blocks() if block.navigation_anchor_id]
        assert ids == sorted(set(ids))


@pytest.mark.asyncio
async def test_clamped_input_rearms_but_observation_and_downward_input_do_not() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        for _ in range(3):
            view.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            before = len(app._resume_pending_head)
            view.note_user_scroll(continuous=True, upward=False)
            for _ in range(3):
                app._transcript_scrolled(None)
                await pilot.pause()
            assert len(app._resume_pending_head) == before
            view.post_message(MouseScrollUp(view, 1, 1, 0, -1, 0, False, False, False))
            await settled(app, pilot)
            assert len(app._resume_pending_head) < before, (
                view.scroll_y,
                view.scroll_target_y,
                app._resume_in_zone,
                app._resume_check_pending,
                app._resume_paging,
            )
            rested = len(app._resume_pending_head)
            for _ in range(5):
                app._transcript_scrolled(None)
                await pilot.pause()
            assert len(app._resume_pending_head) == rested


@pytest.mark.asyncio
async def test_newer_user_travel_survives_final_anchor_callback() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=4, animate=False, immediate=True)
        await pilot.pause()
        moved = []
        original = view._settle_gaps

        def move_during_settle(blocks) -> None:
            original(blocks)
            # An actual downward request between gap settlement and the final
            # callback must change the mutable anchor, not be undone by it.
            if view._insert_anchor is not None:
                view.note_user_scroll(upward=False)
                view.scroll_to(y=view.scroll_y + 3, animate=False, immediate=True)
                moved.append(view.scroll_y)

        view._settle_gaps = move_during_settle
        app._mount_older_resume_page()
        await settled(app, pilot)
        assert moved and view.scroll_y == moved[-1]


@pytest.mark.asyncio
async def test_remote_fetch_lease_extends_through_painted_settlement(tmp_path) -> None:
    async with remote_session(tmp_path, history(hidden=500)) as remote:

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            view = app._transcript_view()
            assert view.scroll_y >= view.container_size.height
            # Consume only already-local rows so the next real control action
            # must cross the production RemoteSession / runtime RPC boundary.
            while app._resume_pending_head:
                app._mount_older_resume_page()
                await settled(app, pilot)
            assert remote.history_before_token
            active = 0
            maximum = 0
            calls = 0
            real_fetch = remote.load_older_display_page
            cursor = remote.history_before_token

            async def measured_fetch():
                nonlocal active, maximum, calls
                calls += 1
                active += 1
                maximum = max(maximum, active)
                try:
                    return await real_fetch()
                finally:
                    active -= 1

            remote.load_older_display_page = measured_fetch
            refresh = app.screen._compositor_refresh
            leases = []

            def capture() -> None:
                refresh()
                if view._insert_anchor is not None:
                    leases.append(app._resume_paging)

            app.screen._compositor_refresh = capture
            view.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            assert isinstance(app._resume_head_notice, OlderHistoryNotice)
            app._resume_head_notice.action_older()
            await settled(app, pilot)
            app.screen._compositor_refresh = refresh
            assert calls == maximum == 1
            assert leases and all(leases)
            assert remote.history_before_token != cursor
            assert not app._resume_paging
            assert app._resume_head_notice.text() != "start of conversation"


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden", [0, 500])
async def test_real_sidebar_projection_runs_rendered_fill(tmp_path, hidden) -> None:
    async with (
        remote_session(tmp_path, history(hidden=hidden)) as remote,
        remote_session(tmp_path, history(count=1), name="initial") as initial,
    ):

        async def factory():
            return initial

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 100)) as pilot:
            for _ in range(100):
                await pilot.pause()
                if app._session is initial:
                    break
            source = SessionInteraction(remote)
            app._sidebar_sources[remote.session_id] = source

            async def lease(session_id, *, speculative=False):
                assert session_id == remote.session_id
                source.preparations += 1
                return source

            # Only external discovery is redirected. Prepare, commit, remote
            # display paging, projection and layout are all production paths.
            app._lease_sidebar_source = lease
            prepared = await app._prepare_sidebar_session(remote.session_id)
            ready = app._commit_sidebar_session(
                remote.session_id, prepared, app._sidebar_navigation.generation
            )
            await settled(app, pilot)
            view = app._transcript_view()
            assert app._session is remote
            assert view.scroll_y >= view.container_size.height
            assert view.virtual_size.height >= 2 * view.container_size.height
            if ready is not None and not ready.done():
                ready.cancel()  # readiness scheduling is a separate peer's scope


@pytest.mark.asyncio
async def test_remote_no_progress_stops_fill_and_preserves_retry(tmp_path) -> None:
    async with remote_session(tmp_path, history()) as remote:

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            while app._resume_pending_head:
                app._mount_older_resume_page()
                await settled(app, pilot)
            view = app._transcript_view()
            view.scroll_to(y=0, animate=False, immediate=True)
            calls = 0

            async def no_progress():
                nonlocal calls
                calls += 1
                return []

            remote.load_older_display_page = no_progress
            app._start_resume_fill()
            await settled(app, pilot)
            for _ in range(5):
                await pilot.pause()
            assert calls == 1
            assert not app._resume_paging and not app._resume_fill_active
            assert isinstance(app._resume_head_notice, OlderHistoryNotice)
            assert app._resume_head_notice.can_focus
            app._resume_head_notice.action_older()
            await settled(app, pilot)
            assert calls == 2


@pytest.mark.asyncio
async def test_cancel_resistant_remote_completion_cannot_append_after_switch(tmp_path) -> None:
    async with (
        remote_session(tmp_path, history()) as remote,
        remote_session(tmp_path, history(count=1), name="target") as target,
    ):

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            while app._resume_pending_head:
                app._mount_older_resume_page()
                await settled(app, pilot)
            fetched = asyncio.Event()
            release = asyncio.Event()
            returned = asyncio.Event()
            real_fetch = remote.load_older_display_page

            async def delayed_completion():
                rows = await real_fetch()
                fetched.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # Socket/cleanup completion is allowed to resist viewer
                    # cancellation; publication still belongs to its old view.
                    await release.wait()
                returned.set()
                return rows

            remote.load_older_display_page = delayed_completion
            assert isinstance(app._resume_head_notice, OlderHistoryNotice)
            app._resume_head_notice.action_older()
            await asyncio.wait_for(fetched.wait(), 5)
            source = SessionInteraction(target)
            app._sidebar_sources[target.session_id] = source

            async def lease(session_id, *, speculative=False):
                assert session_id == target.session_id
                source.preparations += 1
                return source

            app._lease_sidebar_source = lease
            prepared = await app._prepare_sidebar_session(target.session_id)
            ready = app._commit_sidebar_session(
                target.session_id, prepared, app._sidebar_navigation.generation
            )
            await settled(app, pilot)
            before = list(app._transcript_view().blocks())
            release.set()
            await asyncio.wait_for(returned.wait(), 5)
            for _ in range(4):
                await pilot.pause()
            assert app._session is target
            assert app._transcript_view().blocks() == before
            assert not app._resume_paging
            if ready is not None and not ready.done():
                ready.cancel()


@pytest.mark.asyncio
async def test_projection_failure_releases_gate_without_losing_pending_page(monkeypatch) -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        pending = list(app._resume_pending_head)
        mounted = set(app._resume_mounted_ids)

        def fail(page):
            app._resume_mounted_ids.add("synthetic-partial-projection")
            raise ValueError("synthetic malformed row")

        monkeypatch.setattr(app, "_collect_resume_page_blocks", fail)
        with pytest.raises(ValueError, match="malformed row"):
            app._mount_older_resume_page()
        assert app._resume_pending_head == pending
        assert app._resume_mounted_ids == mounted
        assert not app._resume_paging


@pytest.mark.asyncio
async def test_live_append_during_insert_does_not_enter_older_page_or_move_anchor() -> None:
    from local_operator.tui.widgets.transcript import NoticeBlock

    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        retained = view.blocks()[1]
        gap = retained.region.y - view.content_region.y
        app._mount_older_resume_page()
        tail = NoticeBlock("Synthetic live tail event", "info")
        view.append_block(tail)
        await settled(app, pilot)
        assert view.blocks()[-1] is tail
        assert retained.region.y - view.content_region.y == gap
        view.action_scroll_end()
        await pilot.pause()
        assert view.is_near_bottom()


@pytest.mark.asyncio
async def test_navigation_generation_change_retires_same_view_callbacks() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=0, animate=False, immediate=True)
        app._transcript_scrolled(True)
        assert app._resume_check_pending
        # A navigation that is cancelled before commit leaves this exact
        # transcript on screen. Its stale check must still retire its lease.
        app._sidebar_navigation.generation += 1
        await pilot.pause()
        assert not app._resume_check_pending
        app._mount_older_resume_page()
        assert app._resume_paging
        app._sidebar_navigation.generation += 1
        await settled(app, pilot)
        assert not app._resume_paging
        before = len(app._resume_pending_head)
        view.scroll_to(y=0, animate=False, immediate=True)
        view.note_user_scroll()
        await settled(app, pilot)
        assert len(app._resume_pending_head) < before


@pytest.mark.asyncio
async def test_busy_upward_burst_earns_only_one_pending_page(monkeypatch) -> None:
    original_start = OperatorApp._start_resume_fill
    monkeypatch.setattr(
        OperatorApp,
        "_start_resume_fill",
        lambda self, *, target=None: (
            original_start(self, target=target) if target is not None else None
        ),
    )
    session = FakeSession()
    session._history = history(hidden=500)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        before = len(app._resume_pending_head)
        app._mount_older_resume_page()
        assert app._resume_paging
        for _ in range(20):
            # One event-loop burst, before the active hidden page can settle.
            view._on_mouse_scroll_up(MouseScrollUp(view, 1, 1, 0, -1, 0, False, False, False))
        await settled(app, pilot)
        # The one pending request is a rendered buffer: hidden-only slices
        # must keep yielding until actual rows exist, not stop after 24 events.
        assert before - len(app._resume_pending_head) > 120
        assert view.scroll_y >= view.container_size.height
        rest = len(app._resume_pending_head)
        for _ in range(8):
            app._transcript_scrolled(None)
            await pilot.pause()
        assert len(app._resume_pending_head) == rest


@pytest.mark.asyncio
async def test_one_physical_wheel_notch_moves_once_and_horizontal_is_not_history() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=20, animate=False, immediate=True)
        await pilot.pause()
        before = len(app._resume_pending_head)
        view.post_message(MouseScrollUp(view, 1, 1, 0, -1, 0, False, False, False))
        await settled(app, pilot)
        assert view.scroll_y == 20 - app.scroll_sensitivity_y
        assert len(app._resume_pending_head) == before
        assert not app._resume_in_zone
        view.scroll_to(y=0, animate=False, immediate=True)
        view.post_message(MouseScrollUp(view, 1, 1, 0, -1, 0, True, False, False))
        await settled(app, pilot)
        assert len(app._resume_pending_head) == before
        assert not app._resume_in_zone


@pytest.mark.asyncio
async def test_early_pageup_during_insert_keeps_native_animation_target() -> None:
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=10, animate=False, immediate=True)
        await pilot.pause()
        app._mount_older_resume_page()
        # Earlier than the late-settle movement case: the native animation
        # starts while insert anchoring still participates in fresh arrange.
        # A no-op arrangement must not replace target_y with its interim y.
        view.action_page_up()
        await settled(app, pilot)
        assert not app.animator.is_being_animated(view, "scroll_y")
        assert view.scroll_y == view.scroll_target_y
        assert not app._resume_paging and not app._resume_check_pending


@pytest.mark.asyncio
async def test_revisiting_a_cached_view_never_duplicates_or_retires_its_paging(tmp_path) -> None:
    """A → B → cached A while a real RPC is held: one request, one owner.

    The reviewer's F1: re-entry used to reset an app-global paging flag and
    start a second fetch against the same cursor, and the FIRST completion
    then cleared the newer transaction's gate and surfaced `history changed
    while paging` to the reader.
    """
    async with (
        remote_session(tmp_path, history()) as remote,
        remote_session(tmp_path, history(count=1), name="target") as target,
    ):

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            while app._resume_pending_head:
                app._mount_older_resume_page()
                await settled(app, pilot)

            requests = 0
            fetched = asyncio.Event()
            release = asyncio.Event()
            returned = asyncio.Event()
            real_fetch = remote.load_older_display_page

            async def delayed_completion():
                nonlocal requests
                requests += 1
                rows = await real_fetch()
                fetched.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # Owner-side completion may resist viewer cancellation.
                    await release.wait()
                returned.set()
                return rows

            remote.load_older_display_page = delayed_completion
            notices: list[str] = []
            app._notice = lambda text, kind="info": notices.append(str(text))  # type: ignore

            home = app._interaction
            assert isinstance(app._resume_head_notice, OlderHistoryNotice)
            app._resume_head_notice.action_older()
            await asyncio.wait_for(fetched.wait(), 5)
            assert app._resume_paging

            # Away to B, then BACK to the cached A presentation.
            away = SessionInteraction(target)
            app._sidebar_sources[target.session_id] = away
            sources = {target.session_id: away, remote.session_id: home}

            async def lease(session_id, *, speculative=False):
                chosen = sources[session_id]
                chosen.preparations += 1
                return chosen

            app._lease_sidebar_source = lease
            for session_id in (target.session_id, remote.session_id):
                prepared = await app._prepare_sidebar_session(session_id)
                ready = app._commit_sidebar_session(
                    session_id, prepared, app._sidebar_navigation.generation
                )
                # Not `settled()`: that waits for the paging gate to clear, and
                # this test is deliberately holding it across the round trip.
                for _ in range(12):
                    await pilot.pause()
                if ready is not None and not ready.done():
                    ready.cancel()

            assert app._interaction is home
            # The transaction survived the round trip, so the revisit neither
            # started a second request nor found an open gate to spend.
            assert requests == 1
            assert app._resume_paging
            view = app._transcript_view()
            view.scroll_to(y=0, animate=False, immediate=True)
            app._transcript_scrolled(True)
            for _ in range(6):
                await pilot.pause()
            assert requests == 1

            # The held completion now lands: it owns the gate, so it releases
            # it exactly once, with no error surfaced to the reader.
            before = len(app._resume_pending_head)
            release.set()
            await asyncio.wait_for(returned.wait(), 5)
            await settled(app, pilot)
            for _ in range(6):
                await pilot.pause()
            assert requests == 1
            assert not app._resume_paging
            assert len(app._resume_pending_head) >= before
            assert not any("history changed while paging" in text for text in notices)
            assert not notices, notices

            # And a stale completion from that retired transaction cannot
            # retire whatever the reader started afterwards.
            stale = _PagingLease(source_token=home.token)
            fresh = app._acquire_paging_lease(home)
            assert fresh is not None
            assert not app._release_paging_lease(stale)
            assert app._resume_paging
            assert app._release_paging_lease(fresh)
            assert not app._resume_paging


@pytest.mark.asyncio
async def test_inserted_rows_never_paint_with_provisional_spacing() -> None:
    """U1: newcomers carry their final gap on the frame they first appear.

    Spacing used to be decided in `_settle_gaps`, one painted refresh after
    the mount, so a reader whose upward scroll reached the new rows saw them
    paint flush and then spread apart under an upward-moving finger.

    Asserted on the newcomers themselves, not only on the retained anchor: the
    y offset of every inserted block is compared across each compositor frame
    of the insertion, and the gap class each one carries on its first visible
    frame must equal the one it settles with.
    """
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()

        known = set(view.blocks())
        frames: list[dict[int, tuple[int, bool]]] = []
        refresh = app.screen._compositor_refresh

        def capture() -> None:
            refresh()
            # Only rows actually inside the viewport: a block parked off screen
            # has no painted position for a reader to see move.
            frames.append(
                {
                    id(block): (
                        block.region.y - view.content_region.y,
                        block.has_class(GAP_CLASS),
                    )
                    for block in view.blocks()
                    if block not in known
                    and block.region.bottom > view.content_region.y
                    and block.region.y < view.content_region.bottom
                }
            )

        app.screen._compositor_refresh = capture
        try:
            app._mount_older_resume_page()
            await settled(app, pilot)
        finally:
            app.screen._compositor_refresh = refresh

        newcomers = [block for block in view.blocks() if block not in known]
        assert newcomers, "no rows were inserted"
        final = {id(block): block.has_class(GAP_CLASS) for block in newcomers}

        seen: dict[int, tuple[int, bool]] = {}
        for frame in frames:
            for key, (y, gap) in frame.items():
                assert gap == final[key], (
                    "a newly inserted row painted with provisional spacing "
                    "before its gap settled (U1)"
                )
                if key in seen:
                    assert seen[key][0] == y, (
                        f"visible inserted row moved {seen[key][0]} -> {y} "
                        "between painted frames"
                    )
                seen[key] = (y, gap)
        assert seen, "no inserted row was ever visible during the insertion"


@pytest.mark.asyncio
async def test_end_holds_the_tail_and_still_retires_upward_demand() -> None:
    """`end` must acquire the tail AND cancel older upward demand (U2).

    Reporting the gesture through the full `note_user_scroll` scheduled a
    resync that ran between `follow_tail`'s acquire and its deferred scroll,
    measured the not-yet-moved offset, and released the anchor the keypress
    had just taken — so a reply arriving afterwards landed off screen.
    """
    session = FakeSession()
    session._history = history()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()
        view.focus()

        # Read upward first, so there is genuine upward demand to retire and
        # the reader is demonstrably away from the tail.
        view.scroll_to(y=0, animate=False, immediate=True)
        app._transcript_scrolled(True)
        await settled(app, pilot)
        assert view.is_following_tail is False

        pending_before = len(app._resume_pending_head)
        await pilot.press("end")
        await settled(app, pilot)

        # The anchor the keypress asked for is still held.
        assert view.is_following_tail is True
        # And the demand it retired stays retired: no page is drawn by
        # travelling to the newest content.
        assert not app._resume_in_zone
        assert len(app._resume_pending_head) == pending_before

        # The user-visible half: a reply arriving after `end` is on screen.
        app._append_block(NoticeBlock("a reply that arrives after end", "info"))
        await settled(app, pilot)
        for _ in range(6):
            await pilot.pause()
        arrived = view.blocks()[-1]
        assert view.is_following_tail is True
        assert arrived.region.bottom <= view.content_region.bottom
        assert arrived.region.y >= view.content_region.y


@pytest.mark.asyncio
async def test_explicit_anchor_offset_outranks_a_tail_following_view() -> None:
    """An `anchor_offset` the caller supplied wins over the view's tail state.

    The subagent's Home path loads a page and jumps to the top, and those are
    two different frames: the page can arrive while the jump is still pending,
    so the view is STILL FOLLOWING THE TAIL at the moment `insert_blocks` arms
    its anchor. Nothing is armed, and a restore that consults only the armed
    transaction falls through to the tail branch — throwing the reader to the
    end of the transcript instead of holding the offset the caller measured.

    Driven at the seam rather than through the loader: reproducing it through
    the real subagent worker means racing that pending jump, which is what
    made `test_history_home_key_lands_on_newly_loaded_page` an intermittent
    net (it caught this regression only 4 runs in 5, and the round-2 commit
    that introduced the defect shipped past it). Calling `insert_blocks`
    directly puts the view in the exact state the race produces — following
    the tail, with an explicit anchor supplied — every single run.
    """
    session = FakeSession()
    session._history = history(count=120)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(90, 28)) as pilot:
        await settled(app, pilot)
        view = app._transcript_view()

        # The state the race leaves behind: the caller's scroll has not landed,
        # so the view still holds the tail while the page is already arriving.
        view.follow_tail()
        for _ in range(4):
            await pilot.pause()
        assert view.is_following_tail is True

        retained = view.blocks()
        assert retained, "no retained content to hold"
        # What the reader is looking at, named by identity: the assertion below
        # is about THIS row still being where it was, not about a row index.
        held = next(block for block in retained if block.virtual_region.bottom > 0)

        page = []
        for index in range(40):
            block = AssistantBlock()
            block.update_text(f"older page row {index:03}")
            block.finalize_text()
            page.append(block)

        inserted_at = float(view.max_scroll_y)
        view.insert_blocks(0, page, anchor_offset=0.0)
        for _ in range(14):
            await pilot.pause()

        # The reader is NOT at the end: pre-fix this restored to the tail,
        # deterministically, because the tail branch owned the decision.
        assert view.scroll_y < view.max_scroll_y - 1, (
            "an explicit anchor_offset was overridden by the view's tail state; "
            "the reader was thrown to the end of the transcript"
        )
        # And the anchor did its actual job: the row the reader was on is
        # still at the top of the viewport, with the page mounted above it.
        assert abs(view.scroll_y - held.virtual_region.y) < 1.0
        assert view.max_scroll_y > inserted_at, "the page did not mount above"
        assert view.blocks()[:40] == page, "the page is not above the retained rows"


# --- Audit history, rendered -------------------------------------------------


async def compacted_history(directory: Path, *, compactions: int, rows_each: int):
    """Seed a journal with real compaction cuts, as a resumed session has."""
    from local_operator.session.transcript import Transcript

    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    written: list[Message] = []
    for cut in range(compactions):
        batch = [
            Message(
                id=f"cut{cut}-row-{index:04}",
                role="user" if index % 2 else "assistant",
                content=[TextContent(text=f"cut {cut} row {index:04}")],
                stop_reason="stop",
            )
            for index in range(rows_each)
        ]
        for message in batch:
            await transcript.append_message(message)
        written.extend(batch)
        await transcript.append_compaction(f"summary {cut}", batch[-1].id, 500)
    tail = [
        Message(
            id=f"tail-row-{index:04}",
            role="assistant",
            content=[TextContent(text=f"tail row {index:04}")],
            stop_reason="stop",
        )
        for index in range(rows_each)
    ]
    for message in tail:
        await transcript.append_message(message)
    written.extend(tail)
    transcript.flush()
    return written


@asynccontextmanager
async def compacted_session(tmp_path: Path, *, compactions: int = 3, rows_each: int = 40):
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    directory = config / "sessions" / "synthetic-compacted"
    await compacted_history(directory, compactions=compactions, rows_each=rows_each)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await RemoteSession.connect(
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


@pytest.mark.asyncio
async def test_a_compacted_session_does_not_claim_the_conversation_starts_at_the_cut(
    tmp_path,
) -> None:
    """The operator's report, as a rendered assertion.

    He resumed a long conversation, scrolled up, and the top row said "start of
    conversation" above real messages. It was the honest rendering of a history
    layer that had already discarded them — so the fix is that the layer now
    reaches them, and this asserts the rendered consequence.
    """
    from local_operator.tui.app import (
        RESUME_AUDIT_NOTICE,
        RESUME_AUDIT_UNREACHABLE_NOTICE,
    )

    # Large enough that the reader does NOT reach the head in one fill, which
    # is the situation the operator was in: a long conversation whose top he
    # had to scroll toward.
    async with compacted_session(tmp_path, compactions=6, rows_each=400) as remote:

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            # Drain everything the model can still see, and STOP at the
            # boundary: this asserts the state the operator was in — context
            # exhausted, pre-compaction rows still above — not the state after
            # the whole journal has been read back.
            for _ in range(40):
                if not remote.history_before_token or remote.history_is_audit:
                    break
                await remote.load_older_display_page()
                await pilot.pause()
            assert remote.history_is_audit, "the chain never offered the audit phase"
            app._reconcile_head_notice()
            await pilot.pause()
            notice = app._resume_head_notice
            assert notice is not None, "the audit phase left no head notice to read"
            # The defect, precisely: this row used to say "start of
            # conversation" while pre-compaction rows sat unread on disk.
            assert notice.text() != "start of conversation"
            assert notice.text() in (RESUME_AUDIT_NOTICE, RESUME_AUDIT_UNREACHABLE_NOTICE)
            assert notice.can_focus, "the audit head notice stays actionable"


@pytest.mark.asyncio
async def test_start_of_conversation_appears_only_once_the_audit_chain_drains(tmp_path) -> None:
    """``RESUME_START_NOTICE`` is now a TRUE statement, and only then."""
    from local_operator.tui.app import RESUME_START_NOTICE

    async with compacted_session(tmp_path) as remote:

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            for _ in range(200):
                if not remote.history_before_token:
                    break
                await remote.load_older_display_page()
                await pilot.pause()
            assert not remote.history_before_token
            app._reconcile_head_notice()
            await pilot.pause()
            notice = app._resume_head_notice
            # Head fully drained: the notice may be removed entirely or state
            # the (now true) start of the conversation.
            if notice is not None and not app._resume_pending_head:
                assert notice.text() == RESUME_START_NOTICE
                assert not notice.can_focus, "true exhaustion drops interactivity"


@pytest.mark.asyncio
async def test_the_compaction_marker_renders_as_a_notice_mid_transcript(tmp_path) -> None:
    """The marker had NO renderer and painted as nothing.

    Harmless while it only ever sat at the very top of the model's replay;
    not harmless now that audit paging puts one at each compaction inside the
    transcript, where it is the only thing telling the reader they have crossed
    out of what the agent can still see.
    """
    from local_operator.tui.session_presentation import COMPACTION_MARKER_NOTICE

    async with compacted_session(tmp_path) as remote:

        async def factory():
            return remote

        app = OperatorApp(factory)
        async with app.run_test(size=(100, 40)) as pilot:
            await settled(app, pilot)
            for _ in range(200):
                if not remote.history_before_token:
                    break
                await remote.load_older_display_page()
                await pilot.pause()
            for _ in range(60):
                if not app._resume_pending_head:
                    break
                app._mount_older_resume_page()
                await pilot.pause()
            texts = [
                block.text()
                for block in app._transcript_view().blocks()
                if isinstance(block, NoticeBlock) and hasattr(block, "text")
            ]
            assert COMPACTION_MARKER_NOTICE in texts
