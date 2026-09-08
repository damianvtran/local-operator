"""Steady-state paints are bounded by changed cells, not catalog poll count."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.geometry import Region

from local_operator.resume import SessionRow
from local_operator.tui.app import OperatorApp
from local_operator.tui.session_catalog import CatalogEntry
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def isolated_sidebar(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)
    monkeypatch.setattr(OperatorApp, "_prewarm_sidebar", lambda self, entries: None)


async def prepare(app, pilot, entries):
    await pilot.pause()
    app._set_sidebar_open(True)
    # Retire both the immediate read and future timer reads before installing
    # synthetic rows; a real empty catalog must not overwrite this fixture.
    app._sidebar_timer.pause()
    app._sidebar_refresh_generation += 1
    sidebar = app._session_sidebar
    sidebar.set_entries(entries)
    sidebar._timer.stop()
    sidebar._timer = None
    await pilot.pause()
    return sidebar


@pytest.mark.asyncio
async def test_identical_catalog_does_not_invalidate_rendered_frame():
    entries = [CatalogEntry(SessionRow("aaaaaaaaaaa1", 940, "One", live_state="idle"))]
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.tui.widgets.session_sidebar.time.time", return_value=1000):
        async with app.run_test(size=(150, 40)) as pilot:
            sidebar = await prepare(app, pilot, entries)
            with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                for _ in range(50):
                    # Fresh objects, not object identity, determine no-op polls.
                    sidebar.set_entries(
                        [replace(entry, row=entry.row._replace()) for entry in entries]
                    )
                refresh.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_changes_and_age_still_invalidate_immediately():
    entry = CatalogEntry(SessionRow("aaaaaaaaaaa1", 940, "One", live_state="idle"))
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.tui.widgets.session_sidebar.time.time", return_value=1000) as clock:
        async with app.run_test(size=(150, 40)) as pilot:
            sidebar = await prepare(app, pilot, [entry])
            # Include tooltip-only metadata; a new field must not silently
            # bypass invalidation because somebody listed only painted fields.
            changed = [
                replace(entry, row=entry.row._replace(name="Renamed")),
                replace(entry, row=entry.row._replace(pending="ask")),
                replace(entry, row=entry.row._replace(live_state="busy")),
                replace(entry, row=entry.row._replace(wakes=2)),
                replace(entry, unseen=True),
                replace(entry, completion_kind="error"),
                replace(entry, completion_token="new-token"),
            ]
            for fresh in changed:
                sidebar.set_entries([entry])
                await pilot.pause()
                with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                    sidebar.set_entries([fresh])
                    refresh.assert_called()
                await pilot.pause()
            sidebar.set_entries([entry])
            await pilot.pause()
            for attribute, value in (("current_id", entry.id), ("cursor_id", entry.id)):
                setattr(sidebar, attribute, "")
                sidebar.refresh()
                await pilot.pause()
                setattr(sidebar, attribute, value)
                with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                    sidebar.set_entries([entry])
                    refresh.assert_called()
                await pilot.pause()
            sidebar.show_error("Read failed")
            await pilot.pause()
            with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                sidebar.set_entries([entry])
                refresh.assert_called()
            await pilot.pause()
            clock.return_value = 1060
            with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                sidebar.set_entries([entry])
                refresh.assert_called()
            await pilot.pause()
            assert "2m" in sidebar.render().plain
            sidebar.set_entries([])
            await pilot.pause()
            assert "No conversations yet" in sidebar.render().plain


@pytest.mark.asyncio
async def test_spinner_repaints_only_visible_animated_cells_and_age_boundary():
    entries = [
        CatalogEntry(SessionRow("aaaaaaaaaaa1", 940, "Working", live_state="busy")),
        CatalogEntry(SessionRow("aaaaaaaaaaa2", 939, "Idle", live_state="idle")),
        CatalogEntry(SessionRow("aaaaaaaaaaa3", 938, "Question", live_state="busy", pending="ask")),
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.tui.widgets.session_sidebar.time.time", return_value=1000) as clock:
        async with app.run_test(size=(150, 40)) as pilot:
            sidebar = await prepare(app, pilot, entries)
            rows = sidebar._display_rows()
            title = 0 if sidebar._draws_section_headers(rows) else 1
            busy_y = next(
                y for y, (_, row) in enumerate(rows, title) if row and row.id == entries[0].id
            )
            before = sidebar._painted_lines[busy_y].text
            with (
                patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh,
                patch.object(sidebar, "render", wraps=sidebar.render) as render,
            ):
                sidebar._advance_spinner()
                refresh.assert_called_once_with(Region(2, busy_y, 1, 1))
                await pilot.pause()
                render.assert_not_called()
            after = sidebar._painted_lines[busy_y].text
            assert before != after
            assert before[:2] == after[:2] and before[3:] == after[3:]
            assert "Working" in after
            clock.return_value = 1060
            with patch.object(sidebar, "refresh", wraps=sidebar.refresh) as refresh:
                sidebar._advance_spinner()
                refresh.assert_called_once_with()
            await pilot.pause()
            assert "2m" in sidebar.render().plain


@pytest.mark.asyncio
async def test_cached_strips_match_full_render_across_navigation_and_styles():
    entries = [
        CatalogEntry(SessionRow(f"{i + 1:012x}", 940 - i, f"Working {i}", live_state="busy"))
        for i in range(50)
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    with patch("local_operator.tui.widgets.session_sidebar.time.time", return_value=1000):
        async with app.run_test(size=(150, 40)) as pilot:
            sidebar = await prepare(app, pilot, entries)

            async def assert_equivalent():
                sidebar._advance_spinner()
                await pilot.pause()
                cached = {y: tuple(strip) for y, strip in sidebar._painted_lines.items()}
                sidebar.refresh()
                await pilot.pause()
                full = {y: tuple(strip) for y, strip in sidebar._painted_lines.items()}

                # Strip segmentation may differ after joining a replaced cell;
                # compare each cell's glyph/style, not incidental segment runs.
                def cells(lines):
                    return {
                        y: [(char, segment.style) for segment in segments for char in segment.text]
                        for y, segments in lines.items()
                    }

                assert cells(cached) == cells(full)

            await assert_equivalent()
            sidebar.focus()
            sidebar.current_id = entries[0].id
            sidebar._set_hover(2)
            sidebar.refresh()
            await pilot.pause()
            await assert_equivalent()
            sidebar._scroll(1)
            await pilot.pause()
            await assert_equivalent()
            idle = CatalogEntry(SessionRow("bbbbbbbbbbbb", 1000, "Idle request", live_state="idle"))
            sidebar.set_entries([idle, *entries])
            sidebar._offset = 0
            sidebar.requested_id = idle.id
            sidebar._requested_at = time.monotonic() + 1000
            sidebar.refresh()
            await pilot.pause()
            # The delay expiring changes the glyph AND its ink. Reusing the
            # previous cell's dim style would produce an incorrectly styled
            # opening spinner even though its character advances correctly.
            sidebar._requested_at = -1000
            with patch.object(sidebar, "render", wraps=sidebar.render) as render:
                sidebar._advance_spinner()
                await pilot.pause()
                assert render.call_count > 0
            await assert_equivalent()
            # A style invalidation queued immediately before a tick must win:
            # equal catalog data is not proof that the old strips are reusable.
            sidebar.styles.color = "red"
            with patch.object(sidebar, "render", wraps=sidebar.render) as render:
                sidebar._advance_spinner()
                await pilot.pause()
                assert render.call_count > 0
            await assert_equivalent()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            await assert_equivalent()
