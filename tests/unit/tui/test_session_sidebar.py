"""Sidebar identity, urgency, geometry and existing composer interactions."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import pytest
from rich.cells import cell_len

from local_operator.resume import SessionRow
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import (
    CatalogEntry,
    SidebarSettings,
    rank_entries,
)
from local_operator.tui.widgets.editor import Editor
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
