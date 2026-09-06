"""Sidebar identity, urgency, geometry and existing composer interactions."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import replace
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Tooltip

from local_operator.resume import SessionRow
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import (
    CatalogEntry,
    SidebarSettings,
    rank_entries,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.session_sidebar import (
    SIDEBAR_GUTTER,
    SIDEBAR_MAIN_MIN_WIDTH,
    SIDEBAR_SPINNER_INTERVAL_S,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolated_sidebar(tmp_path, monkeypatch):
    # Headless apps must never rename the caller's real multiplexer workspace.
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)


def test_urgency_ranking_keeps_gates_independent_of_acknowledgement():
    rows = [
        CatalogEntry(SessionRow("recent", 100, "Recent")),
        CatalogEntry(SessionRow("active", 50, "Working", live_state="busy")),
        CatalogEntry(SessionRow("done", 10, "Completed"), unseen=True),
        CatalogEntry(SessionRow("ask", 1, "Question", pending="ask"), unseen=True),
    ]
    assert [row.id for row in rank_entries(rows)] == ["ask", "done", "active", "recent"]
    acknowledged = [replace(row, unseen=False) for row in rows]
    assert [row.id for row in rank_entries(acknowledged)] == ["ask", "active", "recent", "done"]
    assert [
        row.id for row in rank_entries([CatalogEntry(SessionRow(id, 0, id)) for id in ("b", "a")])
    ] == ["a", "b"]


@pytest.mark.parametrize(
    "values",
    [{}, {"tui": None}, {"tui": {"sidebar_visible": "false", "sidebar_position": "bottom"}}],
)
def test_sidebar_settings_fail_to_safe_defaults(values):
    assert SidebarSettings.from_values(values) == SidebarSettings()


def test_sidebar_settings_use_nested_registry_paths():
    assert SidebarSettings.from_values(
        {"tui": {"sidebar_visible": True, "sidebar_position": "right"}}
    ) == SidebarSettings(True, "right")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (150, 40)])
@pytest.mark.parametrize("position", ["left", "right"])
async def test_toggle_preserves_draft_and_full_width_when_hidden(size, position):
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.text = "Keep my unsubmitted draft"
        before = editor.region.width
        with patch.object(app, "_overlay_live_state", return_value=[]) as read:
            # A hidden sidebar has no catalog I/O, including manual poll calls.
            app._refresh_sidebar()
            read.assert_not_called()
            app._sidebar_settings = SidebarSettings(False, position)
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._session_sidebar.display
            assert editor.text == "Keep my unsubmitted draft"
            if size[0] == 80:
                assert editor.region.width == before
                assert app._session_sidebar.region.bottom <= app.query_one("#input-dock").region.y
            else:
                assert app.query_one("#session-conversation").size.width >= 60
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert not app._session_sidebar.display
            assert editor.region.width == before
            assert app.screen.virtual_size == app.screen.size
            assert not app.screen.show_vertical_scrollbar


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (150, 40)])
@pytest.mark.parametrize("position", ["left", "right"])
async def test_gutter_faces_the_conversation_at_either_position(size, position):
    """The gap belongs on the edge facing the transcript, and must swap sides.

    The list's age column previously sat one cell from the transcript's first
    character, which read as one crowded block rather than two regions. The
    separation is only worth anything on the side the conversation is on, so a
    hardcoded edge would fix the left dock and leave the right one tight.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._sidebar_settings = SidebarSettings(False, position)
        await pilot.press("ctrl+b")
        await pilot.pause()
        sidebar = app._session_sidebar
        outer, content = sidebar.region, sidebar.content_region
        leading = content.x - outer.x
        trailing = outer.right - content.right
        gutter, edge = (trailing, leading) if position == "left" else (leading, trailing)
        assert gutter == SIDEBAR_GUTTER, f"{position}: conversation-facing gap is {gutter}"
        assert edge == 1, f"{position}: outer edge should keep the 1-cell inset, got {edge}"
        # Whitespace has to come from the width, not from the title column: at
        # 28 cells titles already ellipsize, so paying for the gap out of the
        # content would spend the one thing the list exists to show.
        assert content.width >= 28
        if size[0] > 80:
            assert app.query_one("#session-conversation").size.width >= SIDEBAR_MAIN_MIN_WIDTH


def _hover_entries(ids: tuple[str, ...] = ("alpha", "sess", "gamma")) -> list[CatalogEntry]:
    now = time.time()
    return [
        CatalogEntry(SessionRow(sid, now - 60 * (index + 1), f"Session {sid}"))
        for index, sid in enumerate(ids)
    ]


@pytest.mark.asyncio
async def test_tooltip_survives_in_row_movement_and_the_catalog_poll():
    """The description stays up while the pointer is on the row, observed on
    the screen's real ``Tooltip.display`` — not on the widget attribute.

    Round 4 (M1) found the previous guard vacuous: it asserted
    ``sidebar.tooltip``, which the parent also kept stable, so it passed on
    the very commit it claimed to test. Two distinct mechanisms hid the
    description and both are covered here:

    - ``Screen._handle_mouse_move`` hides a showing tooltip on ANY move over
      the owning widget, before the widget sees the event, and re-arms
      nothing. One cell of jitter dropped it and resting did not bring it
      back. The widget now restores it on an in-row move.
    - The 2 s catalog poll's ``set_entries`` blanked the widget attribute
      under a perfectly still pointer. It now re-resolves instead.

    The hovered row is the LAST one so the tooltip's own footprint never
    covers a row the pointer moves to (it did, and made the previous probe
    read the tooltip widget instead of the list).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30), tooltips=True) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_hover_entries(tuple(f"s{i}" for i in range(1, 7))))
        await pilot.pause()
        tooltip = app.screen.get_child_by_type(Tooltip)
        # By identity, not a literal y: the Active/Previous headers occupy
        # lines that hold no session, so a hardcoded row number would land on
        # a header and never change the hovered identity at all.
        entry_rows = [y for y in range(1, sidebar.size.height) if sidebar._entry_at(y) is not None]
        last_row, first_row = entry_rows[-1], entry_rows[0]

        await pilot.hover("#session-sidebar", offset=(8, last_row))
        await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
        await pilot.pause()
        assert tooltip.display, "description never appeared after the delay"
        shown_for = str(tooltip.render())

        for x in (9, 10, 11):
            await pilot.hover("#session-sidebar", offset=(x, last_row))
            await pilot.pause()
            assert tooltip.display, f"in-row move to x={x} dropped the description"
        await asyncio.sleep(0.3)
        await pilot.pause()
        assert tooltip.display, "description did not stay up while resting after a move"

        sidebar.set_entries(list(sidebar.entries))
        await pilot.pause()
        assert tooltip.display, "the catalog poll blanked a resting description"
        assert str(tooltip.render()) == shown_for

        # A different row is a fresh arrival: hidden until the delay passes,
        # then THAT row's description — never the previous row's.
        await pilot.hover("#session-sidebar", offset=(8, first_row))
        await pilot.pause()
        assert not tooltip.display, "a row change must not pop the old description"
        await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
        await pilot.pause()
        assert tooltip.display and str(tooltip.render()) != shown_for


@pytest.mark.asyncio
async def test_hover_reresolves_when_a_refresh_reorders_under_the_pointer():
    """A reorder must relabel the row the pointer is actually over.

    The ranking moves rows beneath a stationary pointer, so an identity
    remembered from the last mouse event would light and describe whichever
    session slid into that slot.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30), tooltips=True) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_hover_entries(("a", "b", "c")))
        first_row = next(
            y for y in range(1, sidebar.size.height) if sidebar._entry_at(y) is not None
        )
        await pilot.hover("#session-sidebar", offset=(8, first_row))
        sidebar.set_entries(_hover_entries(("c", "b", "a")))
        await pilot.pause()
        under_pointer = sidebar._entry_at(first_row)
        assert under_pointer is not None
        assert sidebar._hover_id == under_pointer.id
        assert under_pointer.row.name in str(sidebar.tooltip or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["left", "right"])
async def test_switch_session_attaches_from_the_composer_and_wraps(position):
    """The shortcut ATTACHES, keeps the caret, and wraps at both ends.

    Distinct from F9's focus-then-arrow flow: this is the one-press form for
    moving between live conversations. Wrapping follows the convention that a
    discrete deliberate press wraps while wheel and page movement clamp.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        attached: list[str] = []
        app._sidebar_settings = SidebarSettings(False, position)
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_hover_entries())
        editor = app.query_one(Editor)
        editor.text = "keep my draft"
        editor.focus()
        await pilot.pause()
        order = [entry.id for entry in sidebar.entries]

        def settle(session_id: str) -> None:
            # Stands in for a whole navigation: the real ``select`` prepares,
            # commits, and then clears the in-flight intent. Each case below
            # then starts from a settled app, which is what moving
            # ``session_id`` by hand is pretending happened.
            attached.append(session_id)
            app._sidebar_navigation.intent_id = ""

        with patch.object(app._sidebar_navigation, "select", side_effect=settle):
            for start, key, expected in (
                (order[1], "ctrl+shift+down", order[2]),
                (order[1], "ctrl+shift+up", order[0]),
                (order[0], "ctrl+shift+up", order[-1]),
                (order[-1], "ctrl+shift+down", order[0]),
            ):
                # The action reads the ATTACHED session, which a real attach
                # would have moved; the list's own cursor is not the source of
                # truth for "where am I".
                with patch.object(
                    type(app._session), "session_id", property(lambda _s, v=start: v)
                ):
                    attached.clear()
                    await pilot.press(key)
                    await pilot.pause()
                    assert attached == [expected], f"{start} + {key} attached {attached}"
        # The composer must keep both the focus and the unsent draft.
        assert app.focused is editor
        assert editor.text == "keep my draft"


@pytest.mark.asyncio
async def test_closed_list_switch_reads_a_fresh_ranking():
    """A switch with the drawer closed must not step over a stale snapshot.

    The catalog is polled only while the list is open, and closing it keeps
    the last entries. Round 4 (M2/Q8/U6) created four sessions with the list
    closed and the shortcut never called the loader — it branched on
    "entries empty", and they were not. The branch is now on visibility.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        app._session_sidebar.set_entries(_hover_entries(("sess", "old")))
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert not app._session_sidebar.display
        attached: list[str] = []
        fresh = _hover_entries(("sess", "new1", "new2", "old"))
        with (
            patch.object(app._sidebar_navigation, "select", side_effect=attached.append),
            patch("local_operator.tui.session_catalog.load_catalog", return_value=fresh) as load,
        ):
            await pilot.press("ctrl+shift+down")
            await asyncio.sleep(0.15)
            await pilot.pause()
        assert load.called, "a closed list must read the catalog before stepping"
        assert attached == ["new1"], f"stepped over the stale ranking: {attached}"
        assert [entry.id for entry in app._session_sidebar.entries][:2] == ["sess", "new1"]


@pytest.mark.asyncio
async def test_switch_burst_in_one_event_batch_steps_once_per_press():
    """A HELD key is many Key events in one batch, and each must hop.

    The shape matters and the earlier regression test had the wrong one.
    ``pilot.press(k, k)`` drains the event loop between keys, so the posted
    ``Selected`` message dispatches and ``select`` sets ``requested_id``
    before the next press reads it — that shape passed while the user's did
    not (round 5, U7). Auto-repeat delivers the events back-to-back with NO
    drain, so the origin has to be published synchronously by the action
    itself. These events are posted straight at the driver, without the
    ``wait_for_idle`` ``press`` inserts, to reproduce that exactly.
    """
    from textual import events

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        app._session_sidebar.set_entries(_hover_entries(("a", "sess", "c", "d")))
        hops: list[str] = []

        def select(session_id: str) -> None:
            hops.append(session_id)
            app._sidebar_navigation.requested_id = session_id

        driver = app._driver
        assert driver is not None
        with patch.object(app._sidebar_navigation, "select", side_effect=select):
            for _ in range(3):
                event = events.Key("ctrl+shift+down", None)
                event.set_sender(app)
                driver.send_message(event)
            await pilot.pause()
            await pilot.pause()
        app._sidebar_navigation.requested_id = ""
        app._sidebar_navigation.intent_id = ""
        assert hops == ["c", "d", "a"], hops


@pytest.mark.asyncio
async def test_switch_burst_with_a_drain_between_presses_still_steps_once_each():
    """The separated shape keeps working — intent is set on both paths."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        app._session_sidebar.set_entries(_hover_entries(("a", "sess", "c", "d")))
        hops: list[str] = []

        def select(session_id: str) -> None:
            hops.append(session_id)
            app._sidebar_navigation.requested_id = session_id

        with patch.object(app._sidebar_navigation, "select", side_effect=select):
            for _ in range(3):
                await pilot.press("ctrl+shift+down")
            await pilot.pause()
        app._sidebar_navigation.requested_id = ""
        app._sidebar_navigation.intent_id = ""
        assert hops == ["c", "d", "a"], hops


@pytest.mark.asyncio
async def test_switch_does_not_fire_under_a_pushed_modal():
    """A priority binding must not switch the conversation beneath a picker.

    ``priority=True`` is what lets the shortcut work while the composer holds
    focus; the cost is that it also fires through a pushed screen unless the
    action itself checks (round 4, MINOR-2).
    """
    from local_operator.tui.widgets.session_picker import SessionPickerScreen

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        app._session_sidebar.set_entries(_hover_entries())
        attached: list[str] = []
        with patch.object(app._sidebar_navigation, "select", side_effect=attached.append):
            app.push_screen(SessionPickerScreen([], time.time()))
            await pilot.pause()
            await pilot.press("ctrl+shift+down")
            await pilot.pause()
        assert attached == []


@pytest.mark.asyncio
async def test_list_window_current_cursor_and_footer_are_independent():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.current_id = "current"
        entries = [CatalogEntry(SessionRow("current", 1, "Current conversation"))]
        entries.extend(
            CatalogEntry(SessionRow(str(i), i + 2, "語言 long name " * 8)) for i in range(100)
        )
        sidebar.set_open(True)
        sidebar.set_entries(entries)
        await pilot.pause()
        sidebar.focus()
        sidebar.action_edge(False)
        assert sidebar.cursor_id != "current"
        assert sidebar.current_id == "current"
        assert len(sidebar.visible_entries) <= sidebar.page_size
        lines = sidebar.render().plain.splitlines()
        assert len(lines) <= sidebar.size.height
        assert lines[-1] == f"1–{sidebar.page_size}/101 · ctrl+b hide"
        assert all(cell_len(line) <= sidebar.size.width for line in lines)
        sidebar.show_error("read failed")
        assert sidebar.entries
        assert sidebar.render().plain.splitlines()[-1] == "Refresh failed"


@pytest.mark.asyncio
async def test_focus_shortcut_preserves_draft_and_returns_to_editor():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app._editor()
        editor.load_text("draft")
        await pilot.press("f9")
        assert app._session_sidebar.display
        assert app._session_sidebar.has_focus
        assert editor.text == "draft"
        await pilot.press("f9", "x")
        assert app.focused is editor
        assert "x" in editor.text
        assert app._session_sidebar.display


@pytest.mark.asyncio
async def test_loading_keeps_sidebar_chrome_without_textual_loading_overlay():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        sidebar = app._session_sidebar
        sidebar.set_open(True)
        sidebar.entries = ()
        sidebar._catalog_loading = True
        await pilot.pause()
        assert sidebar.loading is False
        text = sidebar.render().plain
        assert "Sessions" in text
        assert "Loading conversations" in text
        assert "f9 focus" in text


@pytest.mark.asyncio
async def test_sidebar_escape_restores_settings_and_current_narrow_selection_closes(monkeypatch):
    app = OperatorApp(lambda: _factory(FakeSession()))
    monkeypatch.setattr(app, "_refresh_sidebar", lambda: None)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.action_toggle_sidebar()
        app._cmd_settings(app._notice)
        await pilot.pause()
        settings_focus = app.focused
        assert settings_focus is not None and settings_focus is not app._editor()
        await pilot.click("#session-sidebar", offset=(2, 0))
        await pilot.press("escape")
        assert app.focused is settings_focus
        app._close_settings_view()
        await pilot.resize_terminal(80, 24)
        app.action_toggle_sidebar()
        app._session_sidebar.set_entries([CatalogEntry(SessionRow("sess", 1, "Current"))])
        await pilot.pause()
        sidebar = app._session_sidebar
        row_y = next(y for y in range(1, sidebar.size.height) if sidebar._entry_at(y) is not None)
        await pilot.click("#session-sidebar", offset=(4, row_y))
        assert not app._session_sidebar.display


@pytest.mark.asyncio
async def test_speculative_prepare_does_not_ensure_bound_on_a_cold_owner():
    """P1-d: prewarm must not engage a runtime. A cold speculative prepare
    raises instead of calling ``_ensure_bound`` — that is how speculation
    stays look-only.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from local_operator.session.remote import RemoteSession
    from local_operator.tui.session_interaction import SessionInteraction

    bound = AsyncMock()
    remote = MagicMock(spec=RemoteSession)
    remote.session_id = "other"
    remote.is_cold = True
    remote._ensure_bound = bound
    remote.frontend_state = SimpleNamespace(pending_gate=None)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        source = SessionInteraction(remote)
        app._sidebar_sources["other"] = source

        async def lease(session_id, *, speculative=False):
            source.preparations += 1
            return source

        app._lease_sidebar_source = lease  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="no longer ready"):
            await app._prepare_sidebar_session("other", speculative=True)
        bound.assert_not_awaited()


@pytest.mark.asyncio
async def test_requested_row_is_distinguishable_from_the_keyboard_cursor():
    """The frame must say WHICH row is opening when the two are not the same.

    Requested and cursor shared both signals — `tint-select` and `›` — so
    rendering the mirror case (cursor on one row, requested on another) and
    its swap produced identical frames: the display could not answer "which
    one is opening" (round 5, D6). Reachable from real bindings: focus the
    list, then ctrl+shift+down, or press down before the switch commits.
    """
    from textual.geometry import Region

    async def grid(cursor: str, requested: str) -> list[str]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._sidebar_timer is not None
            app._sidebar_timer.pause()
            sidebar = app._session_sidebar
            sidebar.set_entries(_hover_entries(("alpha", "sess", "gamma")))
            sidebar.focus()
            await pilot.pause()
            sidebar.cursor_id = cursor
            sidebar.requested_id = requested
            sidebar.refresh()
            await pilot.pause()
            lines = sidebar.render_lines(Region(0, 0, sidebar.size.width, sidebar.size.height))
            return ["".join(segment.text for segment in line) for line in lines]

    straight = await grid("alpha", "gamma")
    mirrored = await grid("gamma", "alpha")
    assert straight != mirrored, "requested and cursor render identically"

    # And the distinction is the caret, in the same two columns as before, so
    # nothing reflows: the title still starts where it always did.
    def row_line(lines: list[str], title: str) -> str:
        return next(line for line in lines if title in line)

    assert row_line(straight, "Session alpha").startswith(" ›   ")
    assert row_line(straight, "Session gamma").startswith(" »   ")
    assert row_line(mirrored, "Session alpha").startswith(" »   ")


@pytest.mark.asyncio
async def test_closing_the_sidebar_drains_leased_sources_but_keeps_local_work():
    """Open/close cycles must not accumulate live viewers (the freeze).

    `_sidebar_presentations` is LRU-bounded and drained on close, but
    `_sidebar_sources` was neither: the close path walked the presentations it
    had just emptied, so a prewarmed viewer that never became visible was
    never visited by any path. Measured at 25 cycles: 50 leaked sources, 0
    dispose calls. Each one keeps a socket and a frontend subscription alive,
    and every owner delta then costs a deep state copy per leak (~1.2ms), so
    the cost grows with time-open until the UI stops responding.

    Both directions are asserted. Draining owner-turn retention is safe — the
    viewer is a read-only projection and cannot stop the owner's turn — but
    draining LOCAL retention would kill a live worker or lose an unsent gate
    answer, so a source with local work must survive the close. That second
    assertion is the guard against over-fixing.
    """
    from unittest.mock import MagicMock

    from local_operator.session.remote import RemoteSession
    from local_operator.tui.session_interaction import SessionInteraction

    disposed: list[str] = []
    unsubscribed: list[str] = []

    def lease(app, session_id: str, *, kind: str) -> SessionInteraction:
        remote = MagicMock(spec=RemoteSession)
        remote.session_id = session_id
        remote.is_cold = False
        remote.has_pending_gate_reply = False
        remote.is_streaming = kind == "owner-busy"

        async def _dispose() -> None:
            disposed.append(session_id)

        remote.dispose = MagicMock(side_effect=_dispose)
        source = SessionInteraction(remote)
        # approve_all + owner streaming is the `auto_work` clause: what a
        # prewarmed viewer of a busy background agent looks like.
        source.draft.approve_all = True
        if kind == "local-work":
            # A turn WE submitted over that socket and are still awaiting.
            source.active_workers = 1
        source.unsubscribe_frontend = lambda: unsubscribed.append(session_id)
        source.controller = MagicMock()
        app._sidebar_sources[session_id] = source
        app._interactions[id(remote)] = source
        app._event_sources[source.controller] = source
        return source

    cycles = 8
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        current = str(getattr(app._session, "session_id", ""))
        keep = lease(app, "local-work", kind="local-work")

        for cycle in range(cycles):
            app.action_toggle_sidebar()
            await pilot.pause()
            lease(app, f"ownerbusy{cycle}", kind="owner-busy")
            lease(app, f"idle{cycle}", kind="idle")
            await pilot.pause()
            app.action_toggle_sidebar()
            await pilot.pause()
            await pilot.pause()

        leftover = set(app._sidebar_sources) - {current}
        assert leftover == {"local-work"}, f"leaked {sorted(leftover - {'local-work'})}"
        # Every leased viewer released its socket AND its subscription; the
        # count does not climb with cycles.
        assert len(disposed) == cycles * 2, disposed
        assert len(unsubscribed) == cycles * 2, unsubscribed
        # The over-fix guard: local in-flight work was never touched.
        assert "local-work" not in disposed
        assert not keep.retired


def _mixed_entries() -> list[CatalogEntry]:
    """Two ACTIVE rows (rank tier <=2) and two PREVIOUS (tier 3)."""
    now = 1_700_000_000.0
    return [
        CatalogEntry(SessionRow("act1", now - 60, "Live one", live_state="busy")),
        CatalogEntry(SessionRow("act2", now - 120, "Live two", live_state="busy")),
        CatalogEntry(SessionRow("old1", now - 9000, "Old one")),
        CatalogEntry(SessionRow("old2", now - 99000, "Old two")),
    ]


@pytest.mark.asyncio
async def test_hover_repaints_the_widget_at_most_once_per_move():
    """Textual re-renders a widget inline on every pointer move to hunt links.

    `Screen._forward_event` calls `get_style_at` BEFORE dispatch, which forces
    `_render_content` whenever the widget is dirty — 0.039ms clean against
    1.975ms dirty. Two callers dirtied it per event: our own hover refresh and
    Textual's `watch_hover_style`, which fires for every widget under the
    pointer anywhere in the app and admits in its own comment that it repaints
    "even when there are no links". With `auto_links = False` only ours
    remains.
    """
    from textual import events

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_hover_entries(("a", "b", "c")))
        await pilot.pause()

        renders = 0
        original = sidebar._render_content

        def counted() -> None:
            nonlocal renders
            renders += 1
            original()

        sidebar._render_content = counted  # type: ignore[method-assign]
        moves = 6
        for step in range(moves):
            event = events.MouseMove(
                sidebar,
                x=5,
                y=1 + (step % 3),
                delta_x=0,
                delta_y=0,
                button=0,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=sidebar.region.x + 5,
                screen_y=sidebar.region.y + 1 + (step % 3),
                style=None,
            )
            app.screen._forward_event(event)
            await pilot.pause()
        assert renders <= moves, f"{renders} repaints for {moves} moves"


@pytest.mark.asyncio
async def test_disabling_auto_links_leaves_the_painted_frame_identical():
    """The one real risk of `auto_links = False`, closed by measurement.

    It would silently kill hover highlighting IF this list ever rendered a
    link. It renders none — rows are `Text` spans carrying colour and bold
    only — so the painted output must be identical either way, in text AND in
    style. If a link is ever added here this test is what fails.
    """
    from textual.geometry import Region

    async def frame(auto_links: bool) -> list[tuple[str, str]]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._sidebar_timer is not None
            app._sidebar_timer.pause()
            sidebar = app._session_sidebar
            sidebar.auto_links = auto_links
            sidebar.set_entries(_mixed_entries())
            sidebar.focus()
            await pilot.pause()
            painted: list[tuple[str, str]] = []
            for line in sidebar.render_lines(Region(0, 0, sidebar.size.width, sidebar.size.height)):
                for segment in line:
                    painted.append((segment.text, str(segment.style)))
            return painted

    with_links = await frame(True)
    without = await frame(False)
    assert without == with_links
    # And the premise holds: nothing in the frame carries a link.
    assert not any("link" in style for _text, style in without)


@pytest.mark.asyncio
async def test_the_spinner_does_not_tick_while_blurred_or_closed():
    """Every other animated surface rates through `animation_focused`; this
    list was the only one that did not, so a blurred window kept repainting
    every visible row for a terminal nobody was looking at."""
    from local_operator.tui.animation import (
        BLURRED_SPINNER_INTERVAL_S,
        reset_animation_focus,
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            sidebar = app._session_sidebar
            # Closed: no ticks regardless of busy rows.
            sidebar.set_entries(_mixed_entries())
            await pilot.pause()
            assert sidebar._timer is not None
            assert sidebar._timer._active.is_set() is False

            await pilot.press("ctrl+b")
            await pilot.pause()
            sidebar.set_entries(_mixed_entries())
            await pilot.pause()
            # Open with a busy row: running, at the list's own slower cadence.
            assert sidebar._timer is not None
            assert sidebar._timer._active.is_set() is True
            assert sidebar._spinner_rate == SIDEBAR_SPINNER_INTERVAL_S

            app._set_animation_focused(False)
            await pilot.pause()
            assert sidebar._spinner_rate == BLURRED_SPINNER_INTERVAL_S
    finally:
        reset_animation_focus()


@pytest.mark.asyncio
async def test_navigation_crosses_a_section_header_without_stalling():
    """Headers are rendered rows, never entries — the keyboard never sees one.

    `action_move` indexes `entries` and `_switch_session_from` traverses it
    while the list is CLOSED, so a header in that tuple would let
    ctrl+shift+down "switch" to a header and desync open-from-closed
    navigation. Keeping the split presentational is what makes crossing a
    boundary an ordinary step.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_mixed_entries())
        sidebar.focus()
        await pilot.pause()
        # The frame really does carry both headers.
        assert sidebar._header_lines() == 3  # two headers plus one blank
        sidebar.cursor_id = "act1"

        walked: list[str] = []
        for _ in range(3):
            await pilot.press("down")
            await pilot.pause()
            walked.append(sidebar.cursor_id)
        # act2 is the last ACTIVE row and old1 the first PREVIOUS one: the
        # boundary costs no extra press and never parks on a header.
        assert walked == ["act2", "old1", "old2"], walked
        assert all(step in {entry.id for entry in sidebar.entries} for step in walked)


@pytest.mark.asyncio
async def test_an_empty_section_draws_no_header():
    """All-idle shows only "Previous"; a store of live rows only "Active"."""
    from textual.geometry import Region

    async def headers(entries: list[CatalogEntry]) -> list[str]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._sidebar_timer is not None
            app._sidebar_timer.pause()
            sidebar = app._session_sidebar
            sidebar.set_entries(entries)
            await pilot.pause()
            lines = [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]
            return [
                line.strip()
                for line in lines
                if line.strip() in {"Active Sessions", "Previous Sessions"}
            ]

    now = 1_700_000_000.0
    only_previous = [
        CatalogEntry(SessionRow(f"o{i}", now - 9000 * (i + 1), f"Old {i}")) for i in range(3)
    ]
    only_active = [
        CatalogEntry(SessionRow(f"a{i}", now - 60 * (i + 1), f"Live {i}", live_state="busy"))
        for i in range(3)
    ]
    assert await headers(only_previous) == ["Previous Sessions"]
    assert await headers(only_active) == ["Active Sessions"]
    assert await headers(_mixed_entries()) == ["Active Sessions", "Previous Sessions"]


@pytest.mark.asyncio
async def test_rows_are_independent_so_a_row_scoped_repaint_is_sound():
    """Row-scoped hover refresh assumes one row's paint never affects another.

    If that ever stops holding, repainting the two rows whose ground changed
    would leave a third stale — so the assumption is asserted, not trusted.
    """
    from textual.geometry import Region

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app._sidebar_timer is not None
        app._sidebar_timer.pause()
        sidebar = app._session_sidebar
        sidebar.set_entries(_mixed_entries())
        await pilot.pause()

        def painted() -> list[str]:
            return [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]

        before = painted()
        sidebar._set_hover(2)
        after = painted()
        differing = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
        # Hovering one row changes that row's line and nothing else.
        assert len(differing) <= 1, differing
