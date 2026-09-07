"""Sidebar identity, urgency, geometry and existing composer interactions."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
    APP_SCREEN_INSET,
    SIDEBAR_GUTTER,
    SIDEBAR_MAIN_COMFORT_WIDTH,
    SIDEBAR_MAIN_MIN_WIDTH,
    SIDEBAR_MAX_WIDTH,
    SIDEBAR_SPINNER_INTERVAL_S,
    SIDEBAR_WIDTH,
    sidebar_content_width,
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


def _quiesce_sidebar_refresh(app: OperatorApp) -> None:
    """Stop both catalog-refresh paths that opening the sidebar starts.

    Every pilot test that hands the list its OWN rows needs this, or the real
    catalog lands on top of them mid-test. `ctrl+b` starts two refreshes, and
    closing one and missing the other is a mistake this file has now made twice
    (code review rounds 2 and 3):

    * ``app._sidebar_timer`` — the 2 s poll, resumed by ``_set_sidebar_open``.
    * the ONE-SHOT worker ``_set_sidebar_open`` launches on the very next line.
      Its ``set_entries`` is gated only on ``_sidebar_refresh_generation``, so
      bumping the counter is what retires a read already in flight; there is no
      worker handle to cancel.

    The observed failure is a test that looks a row up by name raising when the
    empty catalog replaces its rows. The two geometry tests here call it as
    well, but as PROPHYLAXIS rather than against a demonstrated failure: what
    they assert (the docked width, the conversation lane, ``virtual_size``) is a
    pure function of terminal width and does not move with the row count — tested
    directly, by landing 300 rows at this exact point with the quiesce disabled,
    which changed neither assertion in 15 samples (code review round 5).

    Kept at those sites anyway, and stated rather than quietly dropped: a
    catalog arriving mid-assertion is a variable the test does not control, and
    the next person to add a row-reading assertion there should not have to
    rediscover why the other two tests needed it.

    **The bump retires the CURRENT read, not future ones** — ``_refresh_sidebar``
    bumps the counter itself on entry, so anything that starts a refresh AFTER
    this call (a second ``ctrl+b``, a re-open, an explicit ``_refresh_sidebar()``)
    wins the gate and repaints. Call this after the last thing that could start
    one.
    """
    assert app._sidebar_timer is not None
    app._sidebar_timer.pause()
    app._sidebar_refresh_generation += 1


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


async def _focus_settled(pilot, sidebar) -> None:
    """Focus the list and WAIT until it has actually landed.

    ``Widget.focus()`` defers the real work through ``App.call_later``, so a
    single ``pilot.pause()`` after it is a coin toss. That matters here
    because ``render`` gates the cursor mark and its ``tint-select`` ground on
    ``has_focus``: a frame captured before focus arrives is missing a ``›``
    and a background colour that the next frame has. Any test comparing two
    painted frames from two app instances will then fail intermittently, and
    it CLUSTERS, so a handful of consecutive passes proves nothing.
    """
    sidebar.focus()
    for _ in range(20):
        await pilot.pause()
        if sidebar.has_focus:
            return
    raise AssertionError("sidebar never took focus")


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
            await _focus_settled(pilot, sidebar)
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
    """Two ACTIVE rows (rank tier <=2) and two PREVIOUS (tier 3).

    Deliberately reaches Active WITHOUT ``live_state="busy"``: a busy row
    animates, and a test that compares frames from two app instances then
    compares two spinner PHASES. That made the `auto_links` frame-equality
    guard flaky under ``-n0`` (fail/fail/pass, the diff a single glyph) while
    passing under xdist — a broken instrument in front of the one test that
    closes the `auto_links=False` risk. `pending` and `unseen` are tiers 0 and
    1, so the section split is still exercised with a still frame.
    """
    now = 1_700_000_000.0
    return [
        CatalogEntry(SessionRow("act1", now - 60, "Needs you", pending="approval")),
        CatalogEntry(SessionRow("act2", now - 120, "Has news"), True, "completed"),
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
            # Deliberately NOT focused. `render` gates the cursor mark and its
            # tinted ground on `has_focus`, and `focus()` lands through
            # `call_later`, so a focused frame here is a race that fails ~half
            # the time on the identical diff (`'  '` vs `'› '`, `#14110c` vs
            # `#16221a`). Focus has nothing to do with whether the list
            # renders a LINK, which is the only question this guard asks, so
            # the stable frame is the honest instrument rather than a weaker
            # one. The frame IS captured under hover, which is the state a
            # link would be highlighted in at all — that is what keeps this
            # able to fail if a link is ever rendered here.
            await pilot.pause()
            first_row = next(
                y for y in range(1, sidebar.size.height) if sidebar._entry_at(y) is not None
            )
            sidebar._set_hover(first_row)
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
            # Closed: no ticks regardless of busy rows. A BUSY fixture is
            # required here — `_mixed_entries` is deliberately still, because
            # the frame-equality guard cannot compare two spinner phases.
            busy = [
                CatalogEntry(SessionRow("b1", 1_700_000_000.0, "Working", live_state="busy")),
                CatalogEntry(SessionRow("o1", 1_699_000_000.0, "Old one")),
            ]
            sidebar.set_entries(busy)
            await pilot.pause()
            assert sidebar._timer is not None
            assert sidebar._timer._active.is_set() is False

            await pilot.press("ctrl+b")
            await pilot.pause()
            sidebar.set_entries(busy)
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
        await _focus_settled(pilot, sidebar)
        # The frame really does carry both headers.
        # Two headings, each owning the blank beneath it, plus the blank that
        # separates the second heading from the group above.
        assert sidebar._header_lines() == 5
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


@pytest.mark.asyncio
async def test_the_outer_title_yields_to_section_headers_but_survives_quiet_states():
    """One title, not two — and never a frame that opens on bare body copy.

    The panel title and the group headings painted identically (same `muted`,
    same weight, same indent, adjacent lines), so the list opened by saying
    "Sessions" twice with no rendered cue which was which — 2 of 13 usable
    lines at 80x24. The headings now do the title's job when they are there.

    But the quiet states have nothing to head: without the title, loading,
    empty and error would open straight into body copy. Conditioning on "any
    header row exists" gets that for free, since an empty list emits none.
    """
    from textual.geometry import Region

    async def first_lines(setup) -> list[str]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._sidebar_timer is not None
            app._sidebar_timer.pause()
            sidebar = app._session_sidebar
            setup(sidebar)
            if sidebar._timer is not None:
                sidebar._timer.pause()
            await pilot.pause()
            return [
                "".join(segment.text for segment in line).strip()
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]

    populated = await first_lines(lambda sidebar: sidebar.set_entries(_mixed_entries()))
    assert populated[0] == "Active Sessions"
    assert "Sessions" not in populated[:1] or populated[0] != "Sessions"
    # The word appears only inside the two headings, never on its own line.
    assert not any(line == "Sessions" for line in populated)

    # Each heading owns the padding BENEATH it, and the second is separated
    # from the group above: "header, gap, rows" reads as a group starting.
    assert populated[1] == ""
    previous = populated.index("Previous Sessions")
    assert populated[previous - 1] == "", "the second heading collides with the group above"
    assert populated[previous + 1] == "", "the heading does not own its padding"

    for setup in (
        lambda sidebar: sidebar.set_entries([]),
        lambda sidebar: sidebar.show_error("Could not load conversations"),
        lambda sidebar: None,  # loading: never received a catalog
    ):
        quiet = await first_lines(setup)
        assert quiet[0] == "Sessions", quiet[:2]


# ---------------------------------------------------------------------------
# Responsive width
#
# The list is the one surface whose whole job is "recognise your own
# conversation", and at the fixed base width it could not do it: the reported
# frame ellipsized eleven of twelve titles, several of them before the word
# that told them apart from the row above. These pin the growth rule and, more
# importantly, the two ways widening a docked panel usually goes wrong — eating
# the conversation, and regressing narrow terminals.
# ---------------------------------------------------------------------------


def test_a_narrow_terminal_is_left_exactly_as_it_was():
    """Growth must be unable to regress the sizes that already worked.

    Everything at or below the comfort threshold resolves to the base width, so
    the layout, the overlay switch and every existing geometry test describe the
    same app they did before.
    """
    threshold = APP_SCREEN_INSET + SIDEBAR_WIDTH + SIDEBAR_GUTTER + SIDEBAR_MAIN_COMFORT_WIDTH
    for width in (60, 80, 100, threshold - 1, threshold):
        assert sidebar_content_width(width) == SIDEBAR_WIDTH, width


def test_growth_spends_only_surplus_and_stops_at_the_cap():
    """One column of terminal buys at most one column of list, up to the cap."""
    threshold = APP_SCREEN_INSET + SIDEBAR_WIDTH + SIDEBAR_GUTTER + SIDEBAR_MAIN_COMFORT_WIDTH
    assert sidebar_content_width(threshold + 1) == SIDEBAR_WIDTH + 1
    assert sidebar_content_width(threshold + 5) == SIDEBAR_WIDTH + 5
    assert sidebar_content_width(threshold + 500) == SIDEBAR_MAX_WIDTH
    # Monotonic: dragging a window wider never narrows the list.
    widths = [sidebar_content_width(w) for w in range(40, 400)]
    assert widths == sorted(widths)


def test_the_conversation_never_pays_for_the_list_getting_wider():
    """The structural invariant, as arithmetic over the whole width range.

    At every terminal width, whatever the sidebar and its gutter take must
    leave the main lane at least its comfort width. Checked across the range
    rather than at the two sizes a screenshot happens to use.

    The app's own outer inset is part of the sum. Leaving it out is what made
    the documented guarantee wrong by two cells (code review round 1, M1) while
    this test still passed — arithmetic that models the layout inaccurately
    proves only that the arithmetic is self-consistent, which is why the
    companion test below measures the REAL widget instead.
    """
    for width in range(40, 400):
        content = sidebar_content_width(width)
        if content == SIDEBAR_WIDTH:
            continue  # not growing: the pre-existing floors govern
        lane = width - APP_SCREEN_INSET - (content + SIDEBAR_GUTTER)
        assert lane >= SIDEBAR_MAIN_COMFORT_WIDTH, (width, lane)


@pytest.mark.asyncio
async def test_the_measured_conversation_lane_matches_the_documented_guarantee():
    """The same invariant, MEASURED, across the sizes where growth is active.

    The arithmetic test above cannot catch a wrong model of the layout, and did
    not: it agreed with a helper that ignored the app's inset, so the lane sat
    at 78 cells while the constant promised 80 (M1). This drives the real app
    and reads the real widget at every width across the growth band and past
    the cap, which is the only assertion that can fail when the model drifts
    from the layout again.

    Sampled every third column rather than every column: each size boots a
    Textual app, and the property is continuous in width, so the sampling
    catches a systematic error without making the file's runtime unreasonable
    on a loaded machine.
    """
    threshold = APP_SCREEN_INSET + SIDEBAR_WIDTH + SIDEBAR_GUTTER + SIDEBAR_MAIN_COMFORT_WIDTH
    for width in range(threshold, threshold + 40, 3):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            _quiesce_sidebar_refresh(app)
            lane = app.query_one("#session-conversation").size.width
            assert lane >= SIDEBAR_MAIN_COMFORT_WIDTH, (width, lane)
            assert app.screen.virtual_size == app.screen.size, width
            assert not app.screen.show_vertical_scrollbar, width


@pytest.mark.asyncio
async def test_a_wide_terminal_actually_shows_more_of_the_title():
    """The user-visible payoff, measured on the RENDERED rows.

    The arithmetic above proves the budget; this proves the budget reaches the
    text. A long title is rendered at a base-width terminal and a wide one, and
    the wide frame must carry strictly more of it — otherwise the extra cells
    went to padding somewhere and the report would be unfixed.
    """
    name = "Article-search-svc schema review and rollout plan"
    entries = [
        CatalogEntry(SessionRow(id="aaaaaaaaaaa1", mtime=time.time(), name=name, live_state="idle"))
    ]

    async def rendered(size) -> str:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+b")
            await pilot.pause()
            _quiesce_sidebar_refresh(app)
            sidebar = app._session_sidebar
            sidebar.set_entries(entries)
            if sidebar._timer is not None:
                sidebar._timer.pause()
            await pilot.pause()
            from textual.geometry import Region

            lines = [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]
            return next(line for line in lines if "Article" in line)

    narrow = await rendered((100, 30))
    wide = await rendered((160, 40))

    def visible_title(line: str) -> str:
        # Everything up to the ellipsis is what the user can actually read,
        # minus the leading chrome (cursor cells and the state glyph) which is
        # fixed-width and not part of the title budget under test.
        text = line.split("…")[0]
        return text[text.index("Article") :].strip() if "Article" in text else text.strip()

    narrow_title, wide_title = visible_title(narrow), visible_title(wide)
    assert len(wide_title) > len(narrow_title), (narrow, wide)
    # Numbers, not just "more": the base width shows about 20 cells of title
    # and the cap about 38, so the gain is worth the change rather than
    # cosmetic. Asserted as a floor so a future tweak to the chrome does not
    # fail the test for being one cell off.
    assert len(narrow_title) < 25, narrow_title
    assert len(wide_title) >= 35, wide_title
    # A title that FITS the grown budget is shown outright, with no ellipsis:
    # the previous frame ellipsized even short names.
    short = "Article-search-svc schema review"
    assert len(short) <= len(wide_title)
    assert wide_title.startswith(short), wide_title


@pytest.mark.asyncio
async def test_the_grown_list_still_leaves_the_conversation_its_lane():
    """The same invariant, through the real layout rather than the helper."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        assert sidebar.content_region.width > SIDEBAR_WIDTH - SIDEBAR_GUTTER
        assert app.query_one("#session-conversation").size.width >= SIDEBAR_MAIN_COMFORT_WIDTH
        # A wider list must not introduce a scrollbar or a reflow.
        assert app.screen.virtual_size == app.screen.size
        assert not app.screen.show_vertical_scrollbar


def test_the_tooltip_never_names_a_different_state_from_the_glyph():
    """Description and marker are two renderings of one precedence.

    The glyph is a single character, so the description is where a user finds
    out what it meant; the two disagreeing is worse than either being terse.
    When wakes were promoted above presence in ``row_state_mark``, a status map
    keyed only on ``live_state`` would have gone on saying "Ready" beside a
    wake glyph.
    """
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        IDLE_MARKER,
        WAKE_MARKER,
        row_state_mark,
    )

    now = time.time()
    armed = CatalogEntry(SessionRow("wake00000001", now, "armed", live_state="idle", wakes=2))
    assert row_state_mark(armed.row, 0)[0] == WAKE_MARKER
    assert armed.status == "Scheduled (2 wakes)"

    single = CatalogEntry(SessionRow("wake00000002", now, "one", live_state="idle", wakes=1))
    assert single.status == "Scheduled (1 wake)", "the count must not be pluralised at 1"

    # A dormant wake does not win the glyph on a LIVE row, so it must not win
    # the words there either.
    dormant = CatalogEntry(
        SessionRow("dorm00000001", now, "stopped", live_state="idle", wakes=1, wakes_dormant=True)
    )
    assert row_state_mark(dormant.row, 0)[0] == IDLE_MARKER
    assert dormant.status == "Ready"

    # ...but on a COLD row it IS the glyph, so it must be the words (D1).
    cold_dormant = CatalogEntry(
        SessionRow("dorm00000002", now, "cold+stopped", wakes=1, wakes_dormant=True)
    )
    assert row_state_mark(cold_dormant.row, 0)[0] == WAKE_MARKER
    assert cold_dormant.status == "Stopped (1 wake dormant)"

    # Presence outranks an armed wake: "a terminal is watching this" answers
    # "where am I?", which bare residency does not (D2).
    watched = CatalogEntry(
        SessionRow("att000000001", now, "watched", live_state="attached", wakes=2)
    )
    assert row_state_mark(watched.row, 0)[0] == ATTACHED_MARKER
    assert watched.status == "Open"

    # And the states above wakes are unchanged.
    busy = CatalogEntry(SessionRow("busy00000001", now, "working", live_state="busy", wakes=4))
    assert busy.status == "Working"
    gate = CatalogEntry(
        SessionRow("gate00000001", now, "waiting", live_state="idle", pending="approval", wakes=4)
    )
    assert gate.status == "Approval needed"


def test_no_reachable_row_state_pairs_a_glyph_with_the_wrong_words():
    """The invariant itself, over EVERY reachable combination.

    The two round-1 design findings were both a glyph and its description
    disagreeing in a state no frame happened to cover: a cold row with a
    stopped schedule drew the wake mark while the tooltip said "Recent" (D1),
    and the individual glyph and status assertions above each passed. Testing
    the pairing one hand-picked row at a time is what let that through, so this
    enumerates the product of every live state, wake count and dormancy and
    asserts the two renderings agree by construction.

    ``unseen`` is a fourth dimension and is covered SEPARATELY, below, rather
    than folded in here (code review round 1, M2). It short-circuits ``status``
    ahead of every branch this function enumerates, and the sidebar likewise
    overrides the glyph for an unseen row — so the pair that actually reaches a
    user is neither the one ``row_state_mark`` returns nor the one this table
    describes. Enumerating it here would assert a pairing no surface renders;
    the companion test drives the real render path instead.
    """
    from local_operator.tui.terminal_title import SPINNER_FRAMES
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        IDLE_MARKER,
        NEEDS_YOU_MARKER,
        WAKE_MARKER,
        WEDGED_MARKER,
        row_state_mark,
    )

    #: Which descriptions may accompany each glyph. A status not listed for the
    #: glyph a row drew is a contradiction the user would have to resolve.
    ALLOWED = {
        NEEDS_YOU_MARKER: {"Approval needed", "Answer needed"},
        WEDGED_MARKER: {"Not responding"},
        ATTACHED_MARKER: {"Open"},
        IDLE_MARKER: {"Ready"},
        WAKE_MARKER: {"Scheduled", "Stopped"},
        "": {"Recent"},
    }
    now = time.time()
    checked = 0
    for state in ("", "idle", "attached", "busy", "wedged"):
        for pending in (None, "approval", "ask"):
            for wakes, dormant in ((0, False), (1, False), (3, False), (1, True), (2, True)):
                row = SessionRow(
                    "x" * 12,
                    now,
                    "a conversation",
                    live_state=state,
                    pending=pending,
                    wakes=wakes,
                    wakes_dormant=dormant,
                )
                glyph = row_state_mark(row, 0)[0]
                status = CatalogEntry(row).status
                if glyph in SPINNER_FRAMES:
                    assert status == "Working", (state, pending, wakes, dormant, status)
                else:
                    allowed = ALLOWED[glyph]
                    assert any(status.startswith(prefix) for prefix in allowed), (
                        f"glyph {glyph!r} paired with {status!r} "
                        f"(state={state!r} pending={pending!r} wakes={wakes} dormant={dormant})"
                    )
                checked += 1
    assert checked == 75, checked


@pytest.mark.asyncio
async def test_an_unseen_row_pairs_its_completion_mark_with_completion_words():
    """The ``unseen`` dimension, asserted on the RENDERED row.

    Raised as M2 in code review round 1: `CatalogEntry.status` returns "Unseen
    completion" ahead of every state branch, so pairing it with
    ``row_state_mark``'s answer would show `◷` beside those words. That pairing
    never reaches a user — the sidebar substitutes its own mark for an
    unseen row — but the only thing establishing that is the render, so the
    render is what this test reads.

    Written as a pilot test rather than as a call to ``row_state_mark`` for
    exactly that reason: the helper's answer is not what is painted here, and a
    unit-level assertion would describe a surface that does not exist.

    The row here is ``idle`` with a wake armed, which is the case M2 named AND
    is now also the case where the mark still WINS. The states where live
    activity outranks it (``busy``, ``wedged``) are enumerated by
    ``test_no_reachable_unseen_row_pairs_a_glyph_with_the_wrong_words`` below,
    which walks the whole product rather than these three hand-picked rows.
    """
    from textual.geometry import Region

    now = time.time()
    cases = [
        ("complete", "✓", "Unseen completion"),
        ("error", "✗", "Unseen error"),
        # `⊘`, not `✗`: an interruption is unfinished work, not a failure.
        ("interrupted", "⊘", "Unseen interruption"),
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        for kind, glyph, words in cases:
            # An ARMED WAKE on the row is the case M2 names: without the
            # override this row would draw the wake glyph beside "Unseen …".
            entry = CatalogEntry(
                SessionRow("u" * 12, now, f"unseen {kind}", live_state="idle", wakes=2),
                unseen=True,
                completion_kind=kind,
            )
            sidebar.set_entries([entry])
            if sidebar._timer is not None:
                sidebar._timer.pause()
            await pilot.pause()
            painted = [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]
            row = next(line for line in painted if f"unseen {kind}" in line)
            assert glyph in row, (kind, row)
            assert entry.status == words, (kind, entry.status)
            # The wake glyph must NOT be what a user sees on an unseen row.
            assert "◷" not in row, (kind, row)


@pytest.mark.asyncio
async def test_a_busy_row_keeps_its_spinner_while_a_completion_is_unacknowledged():
    """The operator's reported bug, as a test: a resumed session drew ``✗``.

    ``unseen`` is a LEVEL, not an edge — it stays true from the moment a turn
    completes until somebody READS that session, and resuming a session does
    not acknowledge it (``AttentionStore`` clears it only when
    ``receipts.acknowledged`` catches up to ``completions.sequence``). The
    sidebar's unseen override sat unconditionally above the live-state mark,
    so a session the operator had already resumed — actively streaming, its
    runtime publishing ``busy`` — kept painting the mark from its PREVIOUS
    turn over its own spinner. Seven such rows read as a screen of failures
    while every one of them was working.

    The row is asserted as still-unseen, because the fix is a precedence
    change and not an acknowledgement: the completion is still unread, and
    the moment the turn finishes the mark must come back.
    """
    from textual.geometry import Region

    from local_operator.tui.terminal_title import SPINNER_FRAMES

    now = time.time()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        for kind in ("complete", "error", "interrupted"):
            entry = CatalogEntry(
                SessionRow("b" * 12, now, f"resumed {kind}", live_state="busy"),
                unseen=True,
                completion_kind=kind,
            )
            sidebar.set_entries([entry])
            if sidebar._timer is not None:
                sidebar._timer.stop()
                sidebar._timer = None
            sidebar._frame = 2
            sidebar.refresh()
            await pilot.pause()
            painted = [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]
            row = next(line for line in painted if f"resumed {kind}" in line)
            assert SPINNER_FRAMES[2] in row, (kind, row)
            # None of the completion marks may survive on a working row.
            for stale in ("✓", "✗", "⊘"):
                assert stale not in row, (kind, stale, row)
            assert entry.status == "Working", (kind, entry.status)
            # The completion is still UNREAD — this is precedence, not a receipt.
            assert entry.unseen, kind


@pytest.mark.asyncio
async def test_an_interrupted_row_draws_its_own_glyph_and_not_the_error_mark():
    """``✗`` is errors only; an interruption is not a failure.

    The operator's store held 526 ``complete`` and 41 ``interrupted`` rows and
    ZERO ``error`` rows, so before this every ``✗`` ever rendered in the
    sidebar was an interruption wearing the failure glyph.
    """
    from textual.geometry import Region

    now = time.time()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        painted_by_kind = {}
        for kind in ("complete", "error", "interrupted"):
            entry = CatalogEntry(
                SessionRow("c" * 12, now, f"idle {kind}", live_state="idle"),
                unseen=True,
                completion_kind=kind,
            )
            sidebar.set_entries([entry])
            if sidebar._timer is not None:
                sidebar._timer.pause()
            await pilot.pause()
            painted = [
                "".join(segment.text for segment in line)
                for line in sidebar.render_lines(
                    Region(0, 0, sidebar.size.width, sidebar.size.height)
                )
            ]
            painted_by_kind[kind] = next(line for line in painted if f"idle {kind}" in line)
        assert "⊘" in painted_by_kind["interrupted"], painted_by_kind["interrupted"]
        assert "✗" not in painted_by_kind["interrupted"], painted_by_kind["interrupted"]
        assert "✗" in painted_by_kind["error"], painted_by_kind["error"]
        assert "⊘" not in painted_by_kind["error"], painted_by_kind["error"]
        assert "✓" in painted_by_kind["complete"], painted_by_kind["complete"]


@pytest.mark.asyncio
async def test_no_reachable_unseen_row_pairs_a_glyph_with_the_wrong_words():
    """The ``unseen`` product, exhaustively, on the RENDERED row.

    The sibling ``test_no_reachable_row_state_pairs_a_glyph_with_the_wrong_words``
    excludes ``unseen`` on the grounds that it short-circuits ``status`` ahead
    of every branch that function enumerates. That reasoning was sound and the
    exclusion still cost us the reported bug: with the fourth dimension tested
    only through three hand-picked rows, "unseen AND busy" — the combination
    the operator actually hit, on seven rows at once — was covered by nothing.

    So this is the same exhaustive treatment applied to the dimension that was
    left out, driven through the real render because the sidebar's override is
    the only place the pairing becomes real.
    """
    from textual.geometry import Region

    from local_operator.tui.terminal_title import SPINNER_FRAMES

    #: Which words may accompany each glyph on an unseen row.
    ALLOWED = {
        "✓": {"Unseen completion"},
        "✗": {"Unseen error", "Not responding"},
        "⊘": {"Unseen interruption"},
        "!": {"Approval needed", "Answer needed"},
    }
    now = time.time()
    app = OperatorApp(lambda: _factory(FakeSession()))
    checked = 0
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+b")
        await pilot.pause()
        _quiesce_sidebar_refresh(app)
        sidebar = app._session_sidebar
        for kind in ("complete", "error", "interrupted"):
            for state in ("", "idle", "attached", "busy", "wedged"):
                for pending in (None, "approval", "ask"):
                    for wakes, dormant in ((0, False), (2, False), (1, True)):
                        entry = CatalogEntry(
                            SessionRow(
                                "d" * 12,
                                now,
                                "pairing probe",
                                live_state=state,
                                pending=pending,
                                wakes=wakes,
                                wakes_dormant=dormant,
                            ),
                            unseen=True,
                            completion_kind=kind,
                        )
                        sidebar.set_entries([entry])
                        if sidebar._timer is not None:
                            sidebar._timer.stop()
                            sidebar._timer = None
                        sidebar._frame = 2
                        sidebar.refresh()
                        await pilot.pause()
                        painted = [
                            "".join(segment.text for segment in line)
                            for line in sidebar.render_lines(
                                Region(0, 0, sidebar.size.width, sidebar.size.height)
                            )
                        ]
                        row = next(line for line in painted if "pairing probe" in line)
                        status = entry.status
                        case = (kind, state, pending, wakes, dormant, row, status)
                        glyph = next((g for g in ALLOWED if g in row), "")
                        if SPINNER_FRAMES[2] in row:
                            # Working outranks the mark, and says so.
                            assert status == "Working", case
                            assert not glyph, case
                        else:
                            assert glyph, case
                            assert status in ALLOWED[glyph], case
                        checked += 1
    # 3 kinds x 5 states x 3 gates x 3 wake shapes.
    assert checked == 135, checked


# --- notification-click routing (PR #750, round 1 remediation) ---------------
#
# These drive `OperatorApp.viewer_resume_session` — the method a notification
# click reaches over the viewer endpoint — against the REAL app rather than a
# double, because the findings they guard are both about the app-side path the
# double cannot have: which switch route runs (D1) and what "displayed" means
# when the target cannot open (D2).


def _call_from_viewer_thread(app: OperatorApp, session_id: str) -> "Future[str]":
    """Invoke the click's entry point the way the endpoint does: off the app's
    thread, on its own PERSISTENT loop.

    Calling it from Textual's thread is refused by contract
    (``call_from_thread`` raises), so faking that would exercise a path that
    does not exist.

    THE LOOP IS NOT ``asyncio.run``, and that is load-bearing rather than
    stylistic. ``asyncio.run`` calls ``shutdown_default_executor`` on the way
    out, which JOINS any worker thread still blocked in ``to_thread`` — so a
    timeout that fired correctly at 0.5 s did not return to the caller until the
    stranded worker finished 30 s later, and the wedge test below read that as
    an unbounded hop. Measured in isolation: ``wait_for`` fired at 0.50 s,
    ``asyncio.run`` returned at 30.02 s. ``ViewerServer`` runs its loop with
    ``run_until_complete`` on a long-lived thread and never tears it down per
    call, so the harness matches production and the test measures the bound
    rather than the teardown.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.new_event_loop()

    def run() -> str:
        try:
            return loop.run_until_complete(app.viewer_resume_session(session_id))
        finally:
            # Deliberately NOT `shutdown_default_executor`; see above.
            loop.close()

    future = pool.submit(run)
    pool.shutdown(wait=False)
    return future


@pytest.mark.asyncio
async def test_a_notification_click_switches_by_the_sidebars_own_route():
    """D1: the click must not answer a gate the user parked in the session it
    leaves.

    ``/resume`` reloads through ``_deny_queued_approvals``, so routing the click
    that way silently answered "no" to a pending approval in the OUTGOING
    session and rendered no receipt — a click about session B spending session
    A's decision. ``docs/SESSION_SIDEBAR.md`` forbids it of a switch in as many
    words, and ``viewer_resume_session``'s own docstring claimed to inherit that
    invariant while taking the other path.

    Asserted structurally, on WHICH route runs: the sidebar navigation is the
    one that suspends gates (``_suspend_sidebar_gates``) rather than denying
    them, so "the click starts a sidebar navigation and runs no /resume" is the
    property, and it cannot drift the way a re-derived gate assertion could.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        selected: list[str] = []
        commands: list[str] = []

        def settle(session_id: str):
            selected.append(session_id)
            done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            done.set_result(None)
            task = asyncio.ensure_future(asyncio.sleep(0))
            # Pretend the navigation landed: the ack is resolved against what
            # is on screen, so the identity has to move for a success.
            app._adopted_session_id = session_id
            return task

        with (
            patch.object(app, "_select_sidebar_session", side_effect=settle),
            patch.object(app, "_run_slash_command", side_effect=commands.append),
            patch.object(type(app._session), "session_id", property(lambda _s: "target-session")),
        ):
            future = _call_from_viewer_thread(app, "target-session")
            for _ in range(200):
                if future.done():
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            detail = future.result(timeout=5)

        assert detail == "displayed target-session"
        assert selected == ["target-session"], "the click must use the sidebar's navigation"
        assert commands == [], (
            "no slash command may run: /resume denies the outgoing session's "
            "parked approval, which a switch must never do"
        )


@pytest.mark.asyncio
async def test_a_click_on_a_session_that_cannot_open_does_not_claim_success():
    """D2: the ack means DISPLAYED, not dispatched.

    ``open_session`` skips its spawn fallback on this ack, so reporting success
    for a session that failed to open cost the user both outcomes — no new
    window AND the conversation they were reading. ``SessionNavigation`` reports
    failure through its ``failed`` callback and completes its task NORMALLY, so
    the task's own outcome is not the answer; what is on screen is.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()

        def settle(session_id: str):
            # The navigation runs and finishes; the binding never moves, which
            # is exactly what a failed boot looks like from here.
            return asyncio.ensure_future(asyncio.sleep(0))

        with (
            patch.object(app, "_select_sidebar_session", side_effect=settle),
            patch.object(
                type(app._session), "session_id", property(lambda _s: "still-the-old-one")
            ),
        ):
            future = _call_from_viewer_thread(app, "cannot-open")
            for _ in range(200):
                if future.done():
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            with pytest.raises(Exception) as excinfo:
                future.result(timeout=5)

        assert "cannot-open" in str(excinfo.value), (
            "a failed switch must raise so the ack is an error frame and the "
            "click falls through to its spawn fallback"
        )


@pytest.mark.asyncio
async def test_a_blocking_app_hop_does_not_outlast_the_click_bound():
    """MAJOR-2: the hop into Textual is bounded, and now really is.

    ``App.call_from_thread`` ends in a bare ``future.result()`` — it blocks its
    caller with no timeout of its own — so a ``wait_for`` placed AFTER it never
    applied to the hazard the docstring named, and an app wedged in a nested
    pump ran 30 s against a 12 s outer bound. The blocking enqueue is inside the
    bound now, via ``to_thread``.

    **THE HAZARD IS SIMULATED IN ``call_from_thread`` ITSELF, not by wedging the
    app's loop, and that is not a shortcut.** A first version of this test
    blocked Textual's loop with ``call_later`` — but the pilot's own coroutine
    runs ON that loop, so the test body could not advance until the block
    expired either. It measured its own suspension (32 s per run) rather than
    the bound, and would have reported success for the wrong reason. Patching
    the blocking call reproduces exactly the property under test — an enqueue
    that never returns — while leaving the loop able to run the assertions.
    """
    from local_operator.session.runtime.viewers import VIEWER_RESUME_TIMEOUT_S

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        entered = threading.Event()
        released = threading.Event()

        def never_returns(callback, *args, **kwargs):
            # What a wedged app does to its caller: the callback is enqueued and
            # nothing ever services it.
            entered.set()
            released.wait(timeout=30)

        try:
            with (
                patch.object(type(app), "call_from_thread", never_returns),
                patch("local_operator.session.runtime.viewers.VIEWER_RESUME_TIMEOUT_S", 0.5),
            ):
                started = time.monotonic()
                future = _call_from_viewer_thread(app, "anything")
                with pytest.raises(asyncio.TimeoutError):
                    # Generous relative to the 0.5 s bound and far below the
                    # 30 s block, so the two outcomes are unambiguous.
                    await asyncio.to_thread(future.result, 10)
                elapsed = time.monotonic() - started
        finally:
            released.set()

        assert entered.is_set(), "precondition: the blocking hop must have been entered"
        assert elapsed < 5.0, (
            f"the bound did not apply to the blocking hop: gave up after "
            f"{elapsed:.1f}s against a 0.5s bound (the real one is "
            f"{VIEWER_RESUME_TIMEOUT_S}s)"
        )


@pytest.mark.asyncio
async def test_the_real_hosts_focus_call_runs_off_the_serving_loop():
    """MAJOR-1/Q2: the property asserted against the METHOD THAT SHIPS.

    ``tests/unit/session/test_viewer_routing.py`` guards this shape too, but
    through a double — so it proves the contract is expressible, not that
    ``OperatorApp.viewer_focus_window`` honours it. Deleting the ``to_thread``
    hop in ``app.py`` leaves that test green (verified). This one is the guard
    that goes red for it, which is what both review streams asked for: this is
    the method whose blocking ``subprocess.run`` would freeze the TUI, and the
    frozen-loop class is invisible to a Python-level probe once it happens.

    ``activate_window`` is patched to report its thread rather than to touch the
    window server: the question is WHERE it ran, and a real OS call would make
    this test depend on a macOS window server that CI does not have.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        ran_on: list[int] = []

        def fake_activate() -> bool:
            ran_on.append(threading.get_ident())
            return True

        with patch("local_operator.tui.window_focus.activate_window", fake_activate):
            handler_thread: list[int] = []

            async def call() -> str:
                handler_thread.append(threading.get_ident())
                return await app.viewer_focus_window()

            loop = asyncio.new_event_loop()
            try:
                detail = await asyncio.to_thread(loop.run_until_complete, call())
            finally:
                loop.close()

        assert detail == "activated"
        assert ran_on, "precondition: the activation must actually have been attempted"
        assert handler_thread, "precondition: the host coroutine must have run"
        # PRECONDITION: the coroutine ran on its own loop thread, not Textual's
        # — otherwise "different from the handler" says nothing about the loop
        # this endpoint has to keep serving.
        assert handler_thread[0] != threading.get_ident()
        assert ran_on[0] != handler_thread[0], (
            "the OS call ran on the loop serving the click; a slow window "
            "server would stall the endpoint and the click would fall back to "
            "spawning the very window this feature removes"
        )


async def _pump_until(pilot, predicate, *, what: str, turns: int = 2000) -> None:
    """Pump Textual until ``predicate`` holds, then return immediately.

    Bounded by TURNS rather than by a wall-clock budget: per-test durations here
    vary by orders of magnitude under xdist contention, and a fixed drain both
    flakes and taxes every run with time the work did not need. The deadline is
    a backstop against a wedge, never the assertion — see AGENTS.md, "Wait on
    the event, never on the clock".
    """
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
async def test_a_switch_that_overruns_the_bound_cannot_also_commit():
    """Q3: never both — a click causes a switch or a spawn, never one of each.

    The derived bounds fixed the CLIENT giving up first; they did not stop the
    APP from committing behind the endpoint's back. ``wait_for`` cancels the
    waiter, not the navigation, so a switch slower than
    ``VIEWER_RESUME_TIMEOUT_S`` still landed on screen after the endpoint had
    answered failure — ``deliver_click`` reported ``switched=False``,
    ``resume_click.open_session`` spawned a window, and the session switched
    underneath it. QA measured the threshold move rather than close: clean at
    9.8 s, both outcomes at 10.2 s and every value above.

    THE ASSERTION IS THE CONJUNCTION, not the timeout. A test that only checked
    "the endpoint raised" passes with the defect fully present — that is the
    whole shape of this bug. So this checks what the CALLER would do with the
    answer against what is actually on screen: a raise means the caller spawns,
    so the binding must not have moved; a success means it does not spawn, so
    the binding must have. Either is correct; only the pair is the defect.

    The navigation is slowed inside ``prepare`` — where a cold transcript's real
    seconds go — so the commit is genuinely still ahead of it when the bound
    expires, rather than being suppressed before it starts.
    """
    from local_operator.session.runtime import viewers

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        committed: list[str] = []

        async def slow_prepare(session_id: str, *args, **kwargs):
            # Comfortably longer than the patched bound, so the endpoint gives
            # up while the switch is genuinely still in flight. Both are scaled
            # down together: what is under test is the ORDER of the bound, the
            # fence and the commit, which is scale-free.
            await asyncio.sleep(1.0)
            return session_id

        def commit(session_id: str, prepared, generation: int):
            # What a real commit does that matters here: move the binding.
            committed.append(session_id)
            app._adopted_session_id = session_id
            return None

        navigation = app._sidebar_navigation
        with (
            patch.object(navigation, "_prepare", slow_prepare),
            patch.object(navigation, "_commit", commit),
            patch.object(navigation, "_release", lambda prepared: asyncio.sleep(0)),
            patch.object(viewers, "VIEWER_RESUME_TIMEOUT_S", 0.3),
            patch.object(viewers, "VIEWER_ABANDON_SETTLE_S", 2.5),
            patch.object(
                type(app._session),
                "session_id",
                property(lambda _s: committed[-1] if committed else "origin"),
            ),
        ):
            future = _call_from_viewer_thread(app, "target-session")
            await _pump_until(pilot, future.done, what="the endpoint to answer")
            try:
                detail = future.result(timeout=5)
                switched = True
            except Exception:
                detail = ""
                switched = False

            # The defect commits AFTER the endpoint has answered, so answering
            # is not the end of it: settle the navigation itself before reading
            # what landed, or a late commit is simply missed.
            task = navigation._task
            await _pump_until(
                pilot,
                lambda: task is None or task.done(),
                what="the navigation to settle",
            )

        landed = "target-session" in committed
        spawn_would_run = not switched
        assert not (landed and spawn_would_run), (
            "DOUBLE ACTION: the switch committed at "
            f"{committed!r} while the endpoint answered failure, so the click "
            "delivers a session switch AND a duplicate window — the exact "
            "outcome this feature exists to prevent"
        )
        assert switched == landed, (
            f"the ack must describe what happened: switched={switched} "
            f"detail={detail!r} committed={committed!r}"
        )


@pytest.mark.asyncio
async def test_abandoning_an_overrun_click_leaves_the_outgoing_session_whole():
    """Q3, the other half: the cure must not be worse than the disease.

    Cancelling mid-switch was the stated reason for NOT adding cancellation in
    round 2 — a half-swapped app is worse than a duplicate window. This asserts
    the outcome rather than the reasoning.

    THE PREPARATION HERE REFUSES TO BE CANCELLED, deliberately. That models the
    case the endpoint's own docstring concedes — enqueued work that cannot be
    recalled — and it is what makes this a test of the FENCE rather than of
    ``task.cancel()``. If the guarantee rested on cancellation being delivered
    in time it would be a race; it rests instead on the generation bump, which
    is synchronous, and on ``SessionNavigation`` running no await between its
    final generation check and ``_commit``. So the un-cancellable navigation
    below runs to completion and STILL cannot commit, and its preparation is
    released through the same path a user's own cancel uses.
    """
    from local_operator.session.runtime import viewers

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._editor().load_text("half-typed thought")
        await pilot.pause()
        released: list[str] = []
        committed: list[str] = []
        prepared_fully: list[str] = []

        async def stubborn_prepare(session_id: str, *args, **kwargs):
            # Work that outlives its cancellation, as a blocking enqueue does.
            inner = asyncio.ensure_future(asyncio.sleep(1.0))
            while not inner.done():
                try:
                    await asyncio.shield(inner)
                except asyncio.CancelledError:
                    pass
            prepared_fully.append(session_id)
            return session_id

        async def release(prepared) -> None:
            released.append(prepared)

        navigation = app._sidebar_navigation
        with (
            patch.object(navigation, "_prepare", stubborn_prepare),
            patch.object(navigation, "_commit", lambda *a: committed.append(a[0])),
            patch.object(navigation, "_release", release),
            patch.object(viewers, "VIEWER_RESUME_TIMEOUT_S", 0.3),
            patch.object(viewers, "VIEWER_ABANDON_SETTLE_S", 2.5),
        ):
            future = _call_from_viewer_thread(app, "target-session")
            await _pump_until(pilot, future.done, what="the endpoint to answer")
            with pytest.raises(Exception):
                future.result(timeout=5)
            # The un-cancellable preparation must be allowed to FINISH: the
            # property is that it cannot commit even then.
            task = navigation._task
            await _pump_until(
                pilot,
                lambda: bool(prepared_fully) and (task is None or task.done()),
                what="the stubborn preparation to run to completion",
            )

        assert prepared_fully == ["target-session"], (
            "precondition: the preparation must have survived cancellation and "
            "run to completion, or this proves nothing about the fence"
        )
        assert committed == [], (
            "the fence must stop a completed preparation from committing after "
            "the endpoint answered failure"
        )
        assert released == [
            "target-session"
        ], f"the fenced preparation must be released, not leaked: {released!r}"
        # The outgoing conversation is untouched: still bound, still typed in.
        assert app._session is not None
        assert (
            app._editor().text == "half-typed thought"
        ), "the draft the user was holding must survive an abandoned click"
        assert navigation.requested_id == "", "the navigation boundary must be released"
        assert not [t for t in navigation._tasks if not t.done()], "no navigation task may leak"
