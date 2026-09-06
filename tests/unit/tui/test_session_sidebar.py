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

        await pilot.hover("#session-sidebar", offset=(8, 6))
        await asyncio.sleep(float(app.TOOLTIP_DELAY) + 0.2)
        await pilot.pause()
        assert tooltip.display, "description never appeared after the delay"
        shown_for = str(tooltip.render())

        for x in (9, 10, 11):
            await pilot.hover("#session-sidebar", offset=(x, 6))
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
        await pilot.hover("#session-sidebar", offset=(8, 1))
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
        await pilot.hover("#session-sidebar", offset=(8, 2))
        sidebar.set_entries(_hover_entries(("c", "b", "a")))
        await pilot.pause()
        under_pointer = sidebar._entry_at(2)
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
        with patch.object(app._sidebar_navigation, "select", side_effect=attached.append):
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
async def test_switch_burst_steps_from_the_requested_target():
    """Three presses before the first commit are three hops, not one.

    The origin used to be ``_session`` — still the old session until commit
    — so every press in a burst recomputed the same target (round 4, U6 /
    MINOR-1). While a switch is in flight the requested id is the origin.
    """
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
        await pilot.click("#session-sidebar", offset=(4, 1))
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
