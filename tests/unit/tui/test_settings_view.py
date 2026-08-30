"""Pilot tests for the ``/settings`` page.

Driven through the REAL :class:`OperatorApp`, which is the only host that loads
``local_operator.tcss`` — the lightweight hosts elsewhere in this directory
declare no ``CSS_PATH``, so a layout assertion made against one of them is
asserting about a stylesheet that was never applied.

Every test that writes points ``LOCAL_OPERATOR_CONFIG_DIR`` at a tmp_path: the
page writes on Enter, and a test that used the real config dir would edit the
developer's own settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.cells import cell_len

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.settings_io import Kind
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.settings_view import SettingsView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    # The display-flag cache is per PROCESS, so a value left by an earlier test
    # would leak into this one's first read.
    from local_operator.tui.settings import settings_reload

    settings_reload()
    return tmp_path


def _values(tmp_path: Path) -> dict[str, Any]:
    config = tmp_path / "config.yml"
    if not config.exists():
        return {}
    return yaml.safe_load(config.read_text()).get("values", {})


def _select(view: SettingsView, key: str) -> int:
    for index, row in enumerate(view._rows):
        if row.kind == "setting" and row.setting is not None and row.setting.key == key:
            view._selected = index
            view._repaint()
            return index
    raise AssertionError(f"no row for {key}")


@pytest.mark.asyncio
async def test_settings_opens_as_a_mode_and_esc_restores_the_conversation() -> None:
    """A MODE, not a modal: the transcript is HIDDEN and comes back untouched,
    with its blocks, its scroll position and any half-typed prompt intact."""
    from local_operator.tui.widgets.transcript import UserBlock

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("a turn worth keeping"))
        editor = app._editor()
        editor.focus()
        editor.load_text("half-typed prompt")
        await pilot.pause()

        app._run_slash_command("/settings")
        await pilot.pause()
        assert app.screen.has_class("settings")
        assert not app._transcript_view().display
        view = app.query_one(SettingsView)
        assert view.has_focus

        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.has_class("settings")
        assert app._transcript_view().display
        assert app._editor().text == "half-typed prompt"


@pytest.mark.asyncio
async def test_reopening_does_not_raise_duplicate_ids() -> None:
    """``remove()`` only POSTS a prune, so a reopen inside that window would
    mount a second same-id widget and raise out of a handler. Class-identified
    all the way down — this is the assertion behind that decision."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        for _ in range(3):
            app._open_settings_view()
            app._close_settings_view()
        app._open_settings_view()
        await pilot.pause()
        assert len(app.query(SettingsView)) == 1


@pytest.mark.asyncio
async def test_bool_toggle_writes_and_survives_reopen(tmp_path: Path) -> None:
    """The BOOL kind, end to end: a toggle reaches config.yml, and the reopened
    page reads it back — a page that only changed its own memory would pass any
    assertion made against the frame alone."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "display.shimmer")
        view.action_activate()
        await pilot.pause()
        assert _values(tmp_path)["display.shimmer"] is False

        app._close_settings_view()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        index = _select(view, "display.shimmer")
        assert " off" in view.render_lines_for_test()[index + 2]


@pytest.mark.asyncio
async def test_enum_expansion_selects_a_choice(tmp_path: Path) -> None:
    """The ENUM kind: Enter expands, Enter on a choice commits and collapses."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        index = _select(view, "tool_approval_mode")
        view.action_activate()
        await pilot.pause()
        assert view._expanded == "tool_approval_mode"
        # The choices are rows now, indented under the row that owns them.
        assert any(row.kind == "choice" for row in view._rows)

        # Move onto "auto" and take it.
        for offset, row in enumerate(view._rows[index + 1 :], start=index + 1):
            if row.kind == "choice" and row.choice.value == "auto":
                view._selected = offset
                break
        view.action_activate()
        await pilot.pause()
        assert _values(tmp_path)["tool_approval_mode"] == "auto"
        assert view._expanded is None


@pytest.mark.asyncio
async def test_text_edit_saves_on_enter(tmp_path: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "retry.maxRetries")
        view.action_activate()
        await pilot.pause()
        assert view.editing_key == "retry.maxRetries"
        view._buffer = "6"
        view._commit_edit()
        await pilot.pause()
        assert _values(tmp_path)["retry"]["maxRetries"] == 6
        assert view.editing_key is None


@pytest.mark.asyncio
async def test_rejected_value_keeps_the_editor_open_and_writes_nothing(tmp_path: Path) -> None:
    """The behaviour that makes a validating form usable rather than hostile:
    the rejected text stays on screen with the reason beside it, so the user
    can fix one character instead of retyping from memory."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "retry.maxRetries")
        view.action_activate()
        view._buffer = "9999"
        view._commit_edit()
        await pilot.pause()

        assert view.editing_key == "retry.maxRetries", "the editor closed on a rejection"
        assert "at most 100" in view.error_text
        assert view._buffer == "9999", "the rejected text was thrown away"
        # Nothing reached the file.
        assert _values(tmp_path).get("retry", {}).get("maxRetries") != 9999
        # And the reason is ON SCREEN, not merely on the object.
        assert any("at most 100" in line for line in view.render_lines_for_test())


@pytest.mark.asyncio
async def test_esc_is_a_ladder(tmp_path: Path) -> None:
    """One press per rung: editor, then expansion, then the page itself. An Esc
    that closed the whole page from inside an editor would throw away an edit
    the user was only trying to abandon."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.maxRetries")
        view.action_activate()
        await pilot.pause()
        assert view.editing_key is not None
        await pilot.press("escape")
        await pilot.pause()
        assert view.editing_key is None
        assert app.screen.has_class("settings"), "the page closed from the first rung"

        _select(view, "tool_approval_mode")
        view.action_activate()
        await pilot.pause()
        assert view._expanded is not None
        await pilot.press("escape")
        await pilot.pause()
        assert view._expanded is None
        assert app.screen.has_class("settings"), "the page closed from the second rung"

        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.has_class("settings")


@pytest.mark.asyncio
async def test_reset_restores_the_default(tmp_path: Path) -> None:
    """Immediate-write's one real cost is undo; this is the mitigation."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "display.terminal_title")
        view.action_activate()
        await pilot.pause()
        assert _values(tmp_path)["display.terminal_title"] is False
        view.action_reset()
        await pilot.pause()
        assert "display.terminal_title" not in _values(tmp_path)


@pytest.mark.asyncio
async def test_every_movement_on_the_page_clamps_at_the_ends() -> None:
    """The ends HOLD, under every movement this page has.

    Regression for the v0.43.0 report: ``action_move`` wrapped
    (``indices[(position + delta) % len(indices)]``), so holding ``down`` at the
    bottom of a 60-row scrolled list threw the reader back to the top with the
    viewport following. This page is AGENTS.md's documented exception to the
    arrows-wrap convention — see ``SettingsView.action_move`` — and the point of
    the exception is that the whole page agrees: a page where ``down`` clamps
    and ``pagedown`` wraps is worse than either rule applied uniformly.

    Pressed REPEATEDLY at each end rather than once, because a single press
    cannot tell a clamp from an off-by-one that still moves.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        view.action_jump(1)
        last = view._selected
        assert last == view._selectable()[-1]
        for _ in range(10):
            view.action_move(1)
        assert view._selected == last, "down at the bottom did not hold"

        view.action_jump(0)
        first = view._selected
        assert first == view._selectable()[0]
        for _ in range(10):
            view.action_move(-1)
        assert view._selected == first, "up at the top did not hold"

        # Paging past the end CLAMPS — and lands on the SAME end the other
        # gestures reach, not on the last section's first row (UX round 1, U3).
        view.action_jump(0)
        for _ in range(20):
            view.action_section(1)
        assert view._selected == last, "pagedown stopped short of the last row"
        for _ in range(20):
            view.action_section(-1)
        assert view._selected == first, "pageup stopped short of the first row"

        # The wheel clamps too, at both ends.
        view.action_jump(0)
        for _ in range(10):
            view._scroll_rows(-1)
        assert view._selected == first
        view.action_jump(1)
        for _ in range(10):
            view._scroll_rows(1)
        assert view._selected == last


@pytest.mark.asyncio
async def test_arriving_at_the_top_shows_the_section_header_that_owns_the_row() -> None:
    """The top end reads as arrival, the way the bottom already does.

    Travelling up to the first row settled at ``scroll_y=1``: the row was on
    screen but the ``Model`` header that names its section was one line off the
    top edge, and the scrollbar thumb was not quite at the start of its track,
    so the top gave weaker "you have arrived" feedback than the bottom. The
    clamp is what made users dwell there long enough for it to matter (UX round
    1, U1).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        await pilot.pause()

        # Premise: row 0 is an unselectable header owning the first row, so
        # scrolling to the SELECTION alone leaves the title off screen.
        assert view._rows[0].kind == "header"

        view.action_jump(1)
        await pilot.pause()
        for _ in range(80):
            view.action_move(-1)
        await pilot.pause()
        assert view._selected == view._selectable()[0]
        assert view._body.scroll_offset.y == 0, "the owning section header is scrolled off"

        # `home` agrees with holding `up`.
        view.action_jump(1)
        await pilot.pause()
        view.action_jump(0)
        await pilot.pause()
        assert view._body.scroll_offset.y == 0


@pytest.mark.asyncio
async def test_a_retired_row_stops_advertising_keys_that_cannot_act() -> None:
    """The footer names what the keys do on THIS row.

    The last six rows of the page are ``Kind.READONLY``: ``enter`` only reports
    that the setting is retired and ``r`` returns without resetting. The clamp
    turned the bottom of the list from a waypoint into a place users park, under
    a footer still promising ``enter change · r default`` (UX round 1, U2).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        view.action_jump(1)
        await pilot.pause()
        row = view._current()
        assert row is not None and row.setting is not None
        assert row.setting.kind is Kind.READONLY, "premise: the last row is retired"
        hints = view.rendered_hints()
        assert "change" not in hints, hints
        assert "default" not in hints, hints
        # The way OUT is never shed, and moving is still offered.
        assert "esc" in hints and "move" in hints, hints

        # An ordinary row still advertises both.
        view.action_jump(0)
        await pilot.pause()
        hints = view.rendered_hints()
        assert "change" in hints and "default" in hints, hints


@pytest.mark.asyncio
async def test_the_side_panes_still_cycle() -> None:
    """←→ between the read-only panes CYCLES, and is meant to.

    The clamp above is about losing your place in a long scrolled list. Two
    tabs that are both on screen have no ends and nothing that scrolls, so
    clamping them would make the second press of a two-tab toggle silently
    dead — the stuck key the wrap convention exists to prevent.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        start = view._pane
        view.action_pane(1)
        assert view._pane != start
        view.action_pane(1)
        assert view._pane == start, "the panes stopped cycling"


@pytest.mark.asyncio
async def test_cascade_add_reorder_and_remove(tmp_path: Path) -> None:
    """Order IS the setting — a cascade is tried top to bottom — so reordering
    has to be a first-class action, not "delete and retype in the right place"."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/a", "openrouter/b"]})

    def chains() -> dict[str, Any]:
        # Re-read from DISK each time. A ConfigManager held across the page's
        # writes would answer from its own stale in-memory copy — the exact
        # trap `ConfigManager.reload` was added for.
        return _values(tmp_path)["retry"]["fallbackChains"]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        # Open the chain, then add a hop.
        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "default":
                view._selected = index
                break
        view.action_activate()
        await pilot.pause()
        for index, row in enumerate(view._rows):
            if row.kind == "hop_add" and row.chain == "default":
                view._selected = index
                break
        view.action_activate()
        view._buffer = "deepseek/c"
        view._commit_edit()
        await pilot.pause()
        assert chains()["default"] == ["anthropic/a", "openrouter/b", "deepseek/c"]

        # Reorder the first hop down.
        for index, row in enumerate(view._rows):
            if row.kind == "hop" and row.hop_index == 0:
                view._selected = index
                break
        view._move_hop(1)
        await pilot.pause()
        assert chains()["default"][:2] == ["openrouter/b", "anthropic/a"]

        # And delete the highlighted hop.
        view._delete_hop()
        await pilot.pause()
        assert len(chains()["default"]) == 2


@pytest.mark.asyncio
async def test_cascade_rejects_a_hop_that_is_not_a_selector(tmp_path: Path) -> None:
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/a"]})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "default":
                view._selected = index
                break
        view.action_activate()
        await pilot.pause()
        for index, row in enumerate(view._rows):
            if row.kind == "hop_add":
                view._selected = index
                break
        view.action_activate()
        view._buffer = "not-a-selector"
        view._commit_edit()
        await pilot.pause()
        assert view.editing_key is not None
        assert "provider/model" in view.error_text
        assert _values(tmp_path)["retry"]["fallbackChains"]["default"] == ["anthropic/a"]


@pytest.mark.asyncio
async def test_teams_and_agents_panes_render_including_empty_state() -> None:
    """The panes exist to make those features DISCOVERABLE, so the empty state
    has to name the command that creates one — "no teams" alone leaves a user
    unable to tell an empty registry from a broken page."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        view.load(teams=[], agents=[], providers=[])
        await pilot.pause()
        pane = view.rendered_pane()
        assert "no teams configured" in pane
        assert "/team" in pane
        assert "read-only" in pane

        view.action_pane(1)
        await pilot.pause()
        pane = view.rendered_pane()
        assert "no agents configured" in pane

        view.load(
            teams=[("lopdev", "manager · 6 roles", "ships changes")],
            agents=[("coder", "role", "implements a slice")],
            providers=[("anthropic", "signed in")],
        )
        await pilot.pause()
        pane = view.rendered_pane()
        assert "coder" in pane
        assert "anthropic" in pane


@pytest.mark.asyncio
async def test_pane_sheds_on_a_narrow_terminal() -> None:
    """Two columns in 78 cells leaves neither readable, so the pane is HIDDEN
    and the list takes the body. The ←→ hint goes with it: a lit hint whose key
    does nothing is the "nothing happens when I click" bug one step earlier."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        assert not view._pane_view.display
        assert "panes" not in view.rendered_hints()
        # esc always survives — it is the only way out.
        assert "esc" in view.rendered_hints()


@pytest.mark.asyncio
async def test_footer_sheds_but_always_names_esc() -> None:
    """The footer concatenates with no per-clause shedding downstream, so the
    ladder is what keeps a 50-column frame from clipping mid-word."""
    for width in (50, 60, 80, 120):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(width, 24)) as pilot:
            await pilot.pause()
            app._open_settings_view()
            view = app.query_one(SettingsView)
            await pilot.pause()
            hints = view.rendered_hints()
            assert "esc" in hints, (width, hints)
            from rich.cells import cell_len

            assert cell_len(hints) <= max(view.size.width - 2, 1), (width, hints)


@pytest.mark.asyncio
async def test_click_selects_then_activates(tmp_path: Path) -> None:
    """Select-then-activate, cf. session_picker: these rows write config, so a
    stray first click must move the cursor rather than change a setting."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        # A VISIBLE row: `_index_at` refuses a click outside the body's
        # region, which is the same constraint a real pointer has.
        index = _select(view, "retry.maxRetries")
        view._scroll_to_selection()
        await pilot.pause()
        assert index - view._body.scroll_offset.y < view._body.size.height
        # Move the cursor elsewhere so the click has to select first.
        view._selected = 0
        view._repaint()

        class _Event:
            button = 1
            screen_x = view._body.region.x + 4
            screen_y = view._body.region.y + index - view._body.scroll_offset.y
            stopped = False

            def stop(self) -> None:
                self.stopped = True

        event = _Event()
        view.on_click(event)
        await pilot.pause()
        assert view._selected == index
        assert "retry" not in _values(tmp_path), "the first click wrote"

        # The second click on the SAME row activates it — here that opens the
        # editor rather than writing, which is the point: a number needs typing.
        view.on_click(_Event())
        await pilot.pause()
        assert view.editing_key == "retry.maxRetries"


@pytest.mark.asyncio
async def test_non_primary_click_is_inert(tmp_path: Path) -> None:
    """A right-click asking for a context menu must not write a setting."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        index = _select(view, "display.images")
        view._scroll_to_selection()
        await pilot.pause()

        class _Event:
            button = 2
            screen_x = view._body.region.x + 4
            screen_y = view._body.region.y + index - view._body.scroll_offset.y

            def stop(self) -> None:
                raise AssertionError("a button-2 click was handled")

        view.on_click(_Event())
        await pilot.pause()
        assert "display.images" not in _values(tmp_path)


@pytest.mark.asyncio
async def test_hover_tracks_the_row_under_the_pointer() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        class _Move:
            screen_x = view._body.region.x + 4
            screen_y = view._body.region.y + 2

        view.on_mouse_move(_Move())
        await pilot.pause()
        assert view._hovered is not None
        view.on_leave(None)
        await pilot.pause()
        assert view._hovered is None


@pytest.mark.asyncio
async def test_opening_frame_is_the_settled_frame() -> None:
    """A two-pane layout inside a mode is exactly where a post-paint reflow
    shows. If the first painted frame differs from the settled one, the user
    sees motion on open whether or not anyone intended an animation."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        first = view.render_lines_for_test()
        await pilot.pause()
        await pilot.pause()
        assert view.render_lines_for_test() == first


@pytest.mark.asyncio
async def test_the_page_never_makes_the_screen_scrollable() -> None:
    """On this app a scrollable SCREEN is always a bug: the body scrolls and
    the dock is docked, and a screen scrollbar silently costs two cells."""
    for size in ((80, 24), (100, 30), (140, 40)):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app._open_settings_view()
            await pilot.pause()
            assert app.screen.virtual_size.height <= app.screen.size.height, size
            assert not app.screen.show_vertical_scrollbar, size


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
@pytest.mark.asyncio
async def test_the_page_takes_the_whole_view_when_opened_from_the_splash(
    size: tuple[int, int],
) -> None:
    """Opened over the BOOT screen, the page gets the same geometry it gets over
    a conversation — in BOTH dimensions.

    Regression for the v0.43.0 report. ``Screen.boot`` is a whole second layout
    (docked centred card, bottom-aligned transcript, and rows reserved BELOW the
    card in the dock's own padding by ``_sync_boot_composition``). Opening the
    mode only added ``Screen.settings``, so both layouts applied at once and the
    page got the leftovers around a card still holding its clamp.

    THE COLLISION IS NOT THE SAME SHAPE AT EVERY SIZE, which is why both
    dimensions are asserted and why the sizes are parameterised rather than
    looped (a loop reports the first failing size and hides the rest):

    - at 140x40 it is VERTICAL — the reserve costs the page rows, 26 of 38.
    - at 100x30, the size the report was filed from, it is entirely
      HORIZONTAL. The boot composition reserves ZERO rows there, so the page
      gets its full 21 either way and every row-based assertion passes on the
      broken tree; what is wrong is the card, still clamped to 73 cells and
      centred at column 12 instead of spanning 96 from column 1. That is the
      half the operator photographed, and an earlier version of this test could
      not see it (review round 1, F1).

    So the horizontal assertions below are load-bearing, not decoration: the
    ``boot-card`` class IS the clamp, and ``#input-shell``'s width and x are
    what it does. Likewise the dock's OUTER size is what carries the vertical
    reserve, so comparing outer to inner is what catches a reserve left behind —
    an assertion on the dock's content height alone passes against the broken
    tree at every size.
    """
    from local_operator.tui.widgets.transcript import UserBlock

    async def measure(app: OperatorApp, seed_conversation: bool) -> tuple[int, int, int, int]:
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            if seed_conversation:
                app._append_block(UserBlock("hello"))
                await pilot.pause()
                # Content retires the splash, so this path never had the bug.
                assert not app.screen.has_class("boot"), size
            else:
                # The splash is up and nothing has been typed: this IS the boot
                # layout, which is the state the report was filed from.
                assert app.screen.has_class("boot"), size
            app._open_settings_view()
            await pilot.pause()
            await pilot.pause()
            view = app.query_one(SettingsView)
            dock = app.query_one("#input-dock")
            shell = app.query_one("#input-shell")
            screen = app.screen

            # -- horizontal: the card's clamp must be gone --------------------
            assert not screen.has_class("boot-card"), (
                f"{size}: the boot card's width clamp survived into the mode "
                f"(#input-shell is {shell.size.width} cells at x={shell.region.x})"
            )
            # The dock spans the page rather than floating inset over it.
            assert shell.size.width == view.size.width, size
            assert shell.region.x == view.region.x, size

            # -- vertical: the composition's reserve must be gone -------------
            assert dock.outer_size.height == dock.size.height, (
                f"{size}: the boot composition reserve survived into the mode "
                f"({dock.outer_size.height} outer vs {dock.size.height} inner)"
            )
            assert dock.styles.padding.bottom == 0, size
            # Every row of the screen is spoken for by the page and the dock.
            assert view.outer_size.height + dock.outer_size.height == screen.size.height, size
            assert screen.virtual_size.height <= screen.size.height, size
            return (view.size.height, view.size.width, shell.size.width, shell.region.x)

    from_boot = await measure(OperatorApp(lambda: _factory(FakeSession())), False)
    # The same page over a CONVERSATION is the reference: the mode is one
    # layout, so where it was opened from may not change ANY of its geometry.
    reference = await measure(OperatorApp(lambda: _factory(FakeSession())), True)
    assert from_boot == reference, (
        f"{size}: /settings from the splash got (view.h, view.w, shell.w, shell.x)="
        f"{from_boot}, but {reference} from a conversation"
    )


@pytest.mark.asyncio
async def test_leaving_the_page_restores_the_boot_layout() -> None:
    """Esc out of ``/settings`` puts the splash composition back, intact.

    The boot layout is suppressed while the mode is up rather than remembered,
    so this pins the other half of that decision: the class, the card clamp and
    the rows ``_sync_boot_composition`` reserves below the card all have to come
    back — and come back as the app RE-DERIVES them, since a ``/clear`` or a
    session swap can move the splash's own state while the page is open.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        dock = app.query_one("#input-dock")
        before = (
            app.screen.has_class("boot"),
            app.screen.has_class("boot-card"),
            dock.outer_size.height,
            dock.styles.padding.bottom,
            app._welcome.size.height if app._welcome is not None else None,
        )

        app._open_settings_view()
        await pilot.pause()
        await pilot.pause()
        assert not app.screen.has_class("boot")

        app._close_settings_view()
        await pilot.pause()
        await pilot.pause()
        after = (
            app.screen.has_class("boot"),
            app.screen.has_class("boot-card"),
            dock.outer_size.height,
            dock.styles.padding.bottom,
            app._welcome.size.height if app._welcome is not None else None,
        )
        assert after == before, f"the boot composition came back changed: {before} -> {after}"
        assert app._welcome is not None and app._welcome.display


@pytest.mark.asyncio
async def test_the_splash_survives_a_settings_round_trip_over_a_conversation() -> None:
    """The suppression is keyed on the LIVE state, not on a snapshot.

    A transcript that has content must not gain a splash because the page was
    opened and closed over it — the failure a remembered-and-restored boot flag
    would produce.
    """
    from local_operator.tui.widgets.transcript import UserBlock

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("hello"))
        await pilot.pause()
        app._open_settings_view()
        await pilot.pause()
        app._close_settings_view()
        await pilot.pause()
        assert not app.screen.has_class("boot")
        assert app._welcome is not None and not app._welcome.display


@pytest.mark.asyncio
async def test_rows_never_exceed_the_list_width() -> None:
    """A wrapped row breaks the one-row-per-setting contract the cursor and
    ``_index_at`` both depend on: a row on two lines makes every click below it
    land on the wrong setting."""
    from rich.cells import cell_len

    for size in ((80, 24), (100, 30), (140, 40)):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app._open_settings_view()
            view = app.query_one(SettingsView)
            view.load(
                teams=[("lopdev", "manager · 6 roles", "ships local-operator changes end to end")],
                agents=[("architect", "role · effort hi", "structural decisions")],
                providers=[("anthropic", "signed in")],
            )
            await pilot.pause()
            room = view._list_width()
            for line in view._list_text.plain.split("\n"):
                assert cell_len(line) <= room, (size, room, line)


@pytest.mark.asyncio
async def test_enter_twice_on_an_unset_row_writes_nothing(tmp_path: Path) -> None:
    """UX round 1, U1 — the BLOCKER. The editor is seeded from the STORED value,
    never from the rendered one.

    ``_render_value`` speaks DISPLAY vocabulary: an unset value reads ``—`` and
    a ``None`` reads ``auto``. Seeding the buffer from it made the placeholder
    a real, committable value, so opening ``/settings`` and pressing enter twice
    on the first row wrote ``hosting: '—'`` — the provider the next launch boots
    on — with the row still reading ``—`` afterwards, so nothing on screen said
    a write had happened.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        config = tmp_path / "config.yml"
        before = config.read_bytes() if config.exists() else b""

        # The cursor lands here on open; assert that rather than seeking, since
        # "the first row of the page" is what makes this a blocker.
        assert view.selected_key == "hosting"
        assert "—" in view.render_lines_for_test()[view._selected + 2]

        await pilot.press("enter")
        await pilot.pause()
        assert view.editing_key == "hosting"
        assert view._buffer == "", "the display placeholder was seeded into the buffer"

        await pilot.press("enter")
        await pilot.pause()
        assert _values(tmp_path).get("hosting") != "—"
        after = config.read_bytes() if config.exists() else b""
        assert after == before, "enter-enter on an unset row touched the config file"

        # And a typed character starts from empty rather than after the glyph.
        _select(view, "web_search.searxng_endpoint")
        await pilot.press("enter")
        await pilot.press("x")
        await pilot.pause()
        assert view._buffer == "x"


@pytest.mark.asyncio
async def test_an_open_editor_owns_its_navigation_keys(tmp_path: Path) -> None:
    """UX round 1, U2 — left/right/home/end belong to the BUFFER while an editor
    is open. They used to fall through to the page bindings and switch the
    read-only side pane mid-typing, which is a keypress doing something the user
    was not looking at."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "web_search.searxng_endpoint")
        await pilot.press("enter")
        for char in "https://searx.example.com/serch":
            await pilot.press(char)
        await pilot.pause()
        assert view._buffer == "https://searx.example.com/serch"
        pane = view._pane

        await pilot.press("left")
        await pilot.pause()
        assert view._pane == pane, "left switched the side pane during an edit"
        assert view.editing_key == "web_search.searxng_endpoint"
        assert view._caret == len(view._buffer) - 1

        # A caret, not merely a swallowed key: the typo is fixable in place
        # rather than by backspacing the whole tail and retyping it.
        for _ in range(2):
            await pilot.press("left")
        await pilot.press("a")
        await pilot.pause()
        assert view._buffer == "https://searx.example.com/search"

        await pilot.press("home")
        await pilot.pause()
        assert view._caret == 0
        assert view.editing_key is not None, "home discarded the edit"
        assert view.selected_key == "web_search.searxng_endpoint", "home jumped the page"
        await pilot.press("end")
        await pilot.pause()
        assert view._caret == len(view._buffer)

        await pilot.press("enter")
        await pilot.pause()
        assert _values(tmp_path)["web_search"]["searxng_endpoint"] == (
            "https://searx.example.com/search"
        )


@pytest.mark.asyncio
async def test_an_arrow_during_an_edit_does_not_silently_discard_it(tmp_path: Path) -> None:
    """UX round 1, U3 — a valid buffer is not lost to an unrelated navigation
    key. Up/down used to call ``_cancel_edit`` unconditionally, so a user who
    typed a value and pressed down to look elsewhere lost it with no message,
    while the footer promised only ``enter saves · esc cancels``."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.baseDelayMs")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "1500":
            await pilot.press(char)
        await pilot.pause()
        assert view._buffer == "1500"

        await pilot.press("down")
        await pilot.pause()
        # A valid buffer COMMITS on the way out rather than evaporating.
        assert _values(tmp_path)["retry"]["baseDelayMs"] == 1500
        assert view.editing_key is None

        # An INVALID buffer keeps the editor open instead, so the move does not
        # throw away text the user still has to correct.
        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "9999":
            await pilot.press(char)
        await pilot.press("down")
        await pilot.pause()
        assert view.editing_key == "retry.maxRetries", "an invalid buffer was discarded by a move"
        assert view._buffer == "9999"
        assert "at most 100" in view.error_text


@pytest.mark.asyncio
async def test_esc_closes_an_open_chain_before_the_page(tmp_path: Path) -> None:
    """UX round 1, U4 — the missing rung. The enum expansion directly above the
    cascade consumes esc, so a ladder with no rung for an open chain teaches a
    rule and then breaks it at the one place a user is two levels down."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_chains(ConfigManager(tmp_path), {"cheap": ["anthropic/a", "openrouter/b"]})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "cheap":
                view._selected = index
                break
        view.action_activate()
        await pilot.pause()
        assert view._chain == "cheap"
        for index, row in enumerate(view._rows):
            if row.kind == "hop":
                view._selected = index
                break

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.has_class("settings"), "esc from a hop row exited the whole page"
        assert view._chain is None
        # And it leaves the cursor on the chain row it just closed.
        current = view._rows[view._selected]
        assert current.kind == "chain" and current.chain == "cheap"

        await pilot.press("escape")
        await pilot.pause()
        assert not app.screen.has_class("settings")


@pytest.mark.asyncio
async def test_deleting_a_chain_asks_first_and_r_says_what_it_does(tmp_path: Path) -> None:
    """UX round 1, U5 — ``d`` on a CHAIN row is a magnitude above ``d`` on a
    hop: it destroys every hop in it, immediate-write has no undo, and ``r``
    could not bring it back. A hop still deletes outright (one line, cheap to
    retype); a chain asks."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_chains(ConfigManager(tmp_path), {"cheap": ["anthropic/a", "openrouter/b"]})

    def chains() -> dict[str, Any]:
        return _values(tmp_path)["retry"]["fallbackChains"]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        def _chain_row() -> int:
            for index, row in enumerate(view._rows):
                if row.kind == "chain" and row.chain == "cheap":
                    return index
            raise AssertionError("no chain row")

        view._selected = _chain_row()
        view._repaint()
        await pilot.press("d")
        await pilot.pause()
        assert chains()["cheap"] == ["anthropic/a", "openrouter/b"], "one d deleted the chain"
        # The ask is ON SCREEN, naming the chain and what it costs.
        detail = view.render_lines_for_test()[-1]
        assert "cheap" in detail and "2 hops" in detail
        assert "d again" in detail

        # Esc backs out of the confirmation without touching the chain.
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.has_class("settings")
        assert chains()["cheap"] == ["anthropic/a", "openrouter/b"]
        assert "d again" not in view.render_lines_for_test()[-1]

        # A second d confirms.
        view._selected = _chain_row()
        view._repaint()
        await pilot.press("d")
        await pilot.press("d")
        await pilot.pause()
        assert chains() == {}

        # And `r` on the cascade row SAYS what it does rather than swallowing
        # the press while the footer advertises the key. As a NOTICE, not an
        # error: the press was reasonable and the page is answering it, so
        # painting it in the danger ink reported a fault that did not happen
        # (UX round 2, U16).
        _select(view, "retry.fallbackChains")
        await pilot.press("r")
        await pilot.pause()
        assert "r" in view.notice_text and "chain" in view.notice_text
        assert view.error_text == ""


#: Every section the pane paints, as (label, is-the-section-heading).
#: The pane has TWO sections and one honesty rule between them, so the test
#: below iterates this rather than naming `providers` in an assertion — see its
#: docstring for why the provider-specific form of that assertion shipped a
#: silent roster.
_PANE_SECTIONS = ("providers", "roster")


@pytest.mark.asyncio
@pytest.mark.parametrize("height", [18, 20, 22, 24, 26, 28, 29, 30, 34, 40])
@pytest.mark.parametrize("roster_size", [0, 1, 3, 8, 40])
@pytest.mark.parametrize("provider_count", [0, 1, 3])
@pytest.mark.parametrize("pane", ["teams", "agents"])
async def test_the_pane_never_paints_more_lines_than_it_has(
    height: int, roster_size: int, provider_count: int, pane: str
) -> None:
    """The pane's height invariant AND its content honesty, across a size band.

    Parameterised because the round-1 version of this test asserted at 100x30
    only and passed while the failure sat one size band away: below 29 rows the
    view's height drops in one step, ``_budget_pane_rows`` took its ``room <= 0``
    early return, and the caller painted the spill line AND the caption anyway
    — eight lines into a seven-row pane, with ``read-only`` the row that fell
    off (design round 2, D6). That is D2's exact symptom reached by a different
    door, so the invariant is pinned rather than the one size.

    The CONTENT assertions are the honesty half, and they are written as a
    GENERAL property over every section rather than as a rule about one of them.
    That generality is the point of this test, not a stylistic preference.
    Round 3 fixed the provider section (D11: the shedding loop deleted from
    index 1, which is the first provider row rather than the separator, so
    between 20 and 26 rows it ate signed-in providers and painted a bold
    ``providers`` header over nothing) and strengthened this test — but the
    strengthened form iterated ``providers`` and looked for the literal line
    ``providers``, so it encoded the PROVIDER-SPECIFIC rule. The very same
    commit inverted the shedding priority on the sibling section, and this test
    passed on a pane that painted ``teams  agents  (←→)`` with nothing under it
    while three agents were configured (design round 4, D15). Three rounds have
    now shipped a pane that fits its box while saying something false, each time
    with a green height assertion.

    So the invariant asserted here is the one that has to hold for the pane to
    be honest at all, for EVERY section:

    1. A non-empty collection is REPRESENTED — as rows, or folded into a
       ``… N more`` count. Silence about content that exists is the defect.
    2. A section HEADING (the ``providers`` header, the ``teams agents``
       tab row) is never the last thing above nothing, because a heading over
       nothing is indistinguishable from the honest empty state the page paints
       for a genuinely empty registry.

    Parameterised over heights AND over which sections are populated, because
    both defects lived in a specific combination: D11 at three providers between
    20 and 26 rows, D15 at three agents at exactly the height where the third
    provider row fit. An empty section is a real case too — it must paint its
    statement or nothing, never a count that invents an entry.
    """
    providers = [("anthropic", "signed in"), ("openrouter", "api key"), ("openai", "api key")][
        :provider_count
    ]
    roster = [
        (f"{pane[:-1]}-{n}", "role · effort hi", "a summary long enough to take its own row")
        for n in range(roster_size)
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, height)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view.load(
            teams=roster if pane == "teams" else [],
            agents=roster if pane == "agents" else [],
            providers=providers,
        )
        # `action_pane` is a no-op when the pane does not fit, so loop to the
        # wanted tab rather than assuming one press lands on it.
        for _ in range(len(_PANE_SECTIONS) + 1):
            if view._pane == pane:
                break
            view.action_pane(1)
        await pilot.pause()
        rendered = view.rendered_pane()
        lines = rendered.split("\n")
        context = (
            f"at {height} rows, {provider_count} provider(s), " f"{roster_size} {pane}:\n{rendered}"
        )
        painted = len(lines)
        assert painted <= view._pane_view.size.height, (
            f"{painted} lines into a {view._pane_view.size.height}-row pane " f"{context}"
        )
        assert any(
            line.strip() == "read-only" for line in lines
        ), f"the boundary caption was shed {context}"

        # The heading of each section, and the collection it is heading. The
        # tab row is the roster's heading: it is the line that says WHICH
        # roster the lines beneath it are, so a tab row over nothing makes the
        # same false statement a bare `providers` header does.
        def _heading(predicate: Any) -> int | None:
            return next((i for i, line in enumerate(lines) if predicate(line.strip())), None)

        sections: dict[str, tuple[int | None, list[str]]] = {
            "providers": (
                _heading(lambda line: line == "providers" or line.startswith("providers  …")),
                [name for name, _state in providers],
            ),
            "roster": (
                _heading(lambda line: "(←→)" in line),
                [name for name, _facts, _summary in roster],
            ),
        }
        assert set(sections) == set(_PANE_SECTIONS)

        for section, (heading, members) in sections.items():
            if heading is None:
                # A section that is not painted at all says nothing false. The
                # pane is allowed to drop a whole section when it has no room
                # for even one line of it (step 8) — what it may not do is paint
                # a heading and then contradict it.
                continue

            # 1. Representation. Every member either appears by name, or is
            #    inside a count that this section owns.
            named = [name for name in members if any(name in line for line in lines)]
            missing = [name for name in members if name not in named]
            owned = (
                lines[heading:]
                if section == "roster"
                else lines[heading : heading + 1]
                + [line for line in lines[heading + 1 :] if "(←→)" not in line]
            )
            counted = any("…" in line and "more" in line for line in owned)
            plural = "y" if len(missing) == 1 else "ies"
            assert not missing or counted, (
                f"section {section!r}: {len(missing)} configured entr{plural} "
                f"{missing} vanished with nothing saying so, so the pane reads as though "
                f"they are not configured, {context}"
            )

            # 2. A heading is never the last thing above nothing. Either it
            #    carries its own count inline (`providers  … 3 more`, or the tab
            #    row's `… N more`), or a non-blank line follows it.
            inline_count = "…" in lines[heading] and "more" in lines[heading]
            following = [line.strip() for line in lines[heading + 1 :] if line.strip()]
            assert inline_count or following, (
                f"section {section!r}: its heading {lines[heading].strip()!r} was painted with "
                f"nothing under it, which is the frame this pane paints for an EMPTY registry, "
                f"{context}"
            )

            # 3. And when the collection is non-empty, that heading must be
            #    followed by something belonging to THIS section rather than by
            #    the next section's heading — the shape D15 produced, where the
            #    tab row's only follower was the `read-only` caption.
            if members and not inline_count:
                assert any(
                    name in line for name in members for line in lines[heading + 1 :]
                ) or any("…" in line and "more" in line for line in lines[heading + 1 :]), (
                    f"section {section!r}: its heading is followed only by other sections' "
                    f"lines, so nothing in the pane says the {len(members)} configured "
                    f"entr{'y' if len(members) == 1 else 'ies'} exist, {context}"
                )


@pytest.mark.asyncio
async def test_the_read_only_caption_survives_a_long_roster() -> None:
    """Design round 1, D2 — ``read-only`` is the one word carrying the
    editable/not boundary, and it was the word that fell off the bottom of the
    pane once three agents overran the row budget. The roster is what gives
    way, spilling into a ``+N more`` line."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view.load(
            agents=[
                (f"agent-{n}", "role · effort hi", "a summary long enough to take its own row")
                for n in range(8)
            ],
            providers=[("anthropic", "signed in"), ("openrouter", "api key")],
        )
        view.action_pane(1)
        await pilot.pause()
        assert view._pane == "agents"
        pane = view.rendered_pane()
        assert "read-only" in pane
        # It is the LAST line, and it is inside the pane's painted height.
        lines = pane.split("\n")
        assert lines[-1].strip() == "read-only"
        assert len(lines) <= view._pane_view.size.height, pane
        assert "more" in pane, "the roster was not truncated, so nothing was reserved"
        # The spill line is a FACT, not a button: `+` is what `+ add a hop` and
        # `+ add a chain` use in the same frame, so a count that led with it
        # read as an affordance (design round 2, D9).
        spill = next(line for line in lines if "more" in line)
        assert not spill.strip().startswith("+"), spill


@pytest.mark.asyncio
async def test_scope_tags_share_one_column() -> None:
    """Design round 1, D3 — two headers of the same rank must put their
    ``takes effect:`` tag in the same column. Right-aligning each against its
    own title made the position depend on the title's length, which the eye
    reads as an accident rather than as a grid."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        columns = {
            line.index("takes effect:")
            for line in view.render_lines_for_test()
            if "takes effect:" in line
        }
        assert len(columns) == 1, columns


@pytest.mark.asyncio
async def test_the_longest_label_is_not_clipped_mid_parenthetical() -> None:
    """Design round 1, D4 — ``Connectivity backoff cap (ms)`` fitted in the
    column and was still cut to ``(m…``, an opened parenthetical that never
    closes, because the label budget spent two cells that exist."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        index = _select(view, "retry.connectivityBackoffCapMs")
        line = view.render_lines_for_test()[index + 2]
        assert "Connectivity backoff cap (ms)" in line, line


def test_pane_width_matches_the_stylesheet() -> None:
    """``_PANE_WIDTH`` is mirrored from ``.settings-view-pane`` because the
    width has to be known before layout runs. This is the guard that keeps the
    duplicate in step — the drift the analytics panel wanted between its
    ``max-width`` and ``_card_width``."""
    import re
    from pathlib import Path as _Path

    from local_operator.tui.widgets.settings_view import _PANE_WIDTH

    tcss = (
        _Path(__file__).resolve().parents[3] / "local_operator/tui/local_operator.tcss"
    ).read_text()
    block = tcss.split(".settings-view-pane {")[1].split("}")[0]
    match = re.search(r"width:\s*(\d+)", block)
    assert match is not None, "the .settings-view-pane rule must declare a width"
    declared = int(match.group(1))
    assert declared == _PANE_WIDTH


def test_the_body_scrollbar_is_this_app_s_palette() -> None:
    """Design round 1, D1 — ``overflow-y: auto`` with no ``scrollbar-*`` rules
    inherits Textual's stock blue (``#003054`` thumb on ``#000000``, 2 cells),
    which appeared in no other frame in the product. Both sibling scrolling
    surfaces pin theirs at 1 cell against ``$lo-*``; this asserts this one does
    too, since a stylesheet regression is invisible to every other test here."""
    import re
    from pathlib import Path as _Path

    tcss = (
        _Path(__file__).resolve().parents[3] / "local_operator/tui/local_operator.tcss"
    ).read_text()
    block = tcss.split(".settings-view-body {")[1].split("}")[0]
    for rule, expected in (
        ("scrollbar-size-vertical", "1"),
        ("scrollbar-gutter", "stable"),
        ("scrollbar-background", r"\$lo-bg"),
        ("scrollbar-color", r"\$lo-edge"),
        ("scrollbar-color-hover", r"\$lo-dim"),
        ("scrollbar-color-active", r"\$lo-muted"),
    ):
        assert re.search(rf"^\s*{rule}:\s*{expected};", block, re.MULTILINE), (rule, block)


def test_persist_hint_prefix_matches_the_app() -> None:
    """The picker recognises the protected clause by prefix and cannot import
    the app (the app imports it), so the constant is duplicated. Keep them in
    step here rather than discovering the drift as a clipped footer."""
    from local_operator.tui.app import PERSIST_HINT
    from local_operator.tui.widgets.model_picker import PERSIST_HINT_PREFIX

    assert PERSIST_HINT.startswith(PERSIST_HINT_PREFIX)


def test_settings_is_a_frontend_local_slash() -> None:
    """The page writes THIS machine's config; routed to a session owner it
    would persist onto the wrong machine — the rule ``/model default`` states
    explicitly when it refuses to run on a follower."""
    from local_operator.session.frontend_state import _FRONTEND_LOCAL_SLASHES

    assert "settings" in _FRONTEND_LOCAL_SLASHES


def _row_id(view: SettingsView, index: int) -> str:
    row = view._rows[index]
    key = row.setting.key if row.setting is not None else ""
    return f"{row.kind}:{key or row.chain or ''}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "wanted"),
    [
        # `down` and `ctrl+n` step off the add row onto the next SETTING; at
        # 4344dbda both landed back on `chain_add`, which is the stuck key.
        ("down", "setting:tui.theme"),
        ("ctrl+n", "setting:tui.theme"),
        # `up` steps into the chain the commit just created — its trailing
        # `+ add a hop` row is the new neighbour above. At 4344dbda it skipped
        # the inserted rows entirely and landed on the cascade's setting row,
        # so it moved, but not to the row that is actually adjacent.
        ("up", "hop_add:cheap"),
    ],
)
async def test_a_commit_that_inserts_rows_does_not_strand_the_cursor(
    tmp_path: Path, key: str, wanted: str
) -> None:
    """Movement resolves its target AFTER the commit, not before it.

    UX round 2, U13. ``action_move`` snapshotted the selectable list and the
    cursor's position in it, then called ``_leave_row()`` \u2014 which commits, and
    a commit on ``+ add a chain`` inserts the chain row, its hops and an
    ``+ add a hop`` row ABOVE the cursor. Stepping from the stale index landed
    back on ``+ add a chain``: the user pressed an arrow and did not move,
    which is the stuck-key failure the identity-anchored rebuild in
    ``action_move`` exists to prevent. Independent of the end-clamping the
    test above pins — this is about a press in the MIDDLE of the list landing
    on the right row, not about what the ends do.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        for index, row in enumerate(view._rows):
            if row.kind == "chain_add":
                view._selected = index
                break
        view._repaint()
        await pilot.pause()
        origin = _row_id(view, view._selected)

        await pilot.press("enter")
        for char in "cheap openrouter/qwen3-coder":
            await pilot.press(char if char != " " else "space")
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()

        # The chain was committed by the move, which is U3's contract...
        assert "cheap" in settings_io.read_chains(ConfigManager(tmp_path))
        # ...and the cursor is on the row that is genuinely adjacent in the
        # REBUILT list, not on one resolved from the list as it was before.
        assert (
            _row_id(view, view._selected) != origin
        ), f"{key} left the cursor on {origin} after a commit that inserted rows"
        assert _row_id(view, view._selected) == wanted


@pytest.mark.asyncio
async def test_a_click_lands_on_the_row_that_was_clicked(tmp_path: Path) -> None:
    """The worst form of U13: a click carries an explicit target.

    ``on_click`` resolved y-to-row before ``_leave_row()``, so a commit that
    inserted rows moved the row out from under the index it had resolved. The
    user clicked the label they could see and the cursor landed elsewhere.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        for index, row in enumerate(view._rows):
            if row.kind == "chain_add":
                view._selected = index
                break
        view._repaint()
        await pilot.pause()

        await pilot.press("enter")
        for char in "cheap openrouter/qwen3-coder":
            await pilot.press(char if char != " " else "space")
        await pilot.pause()

        # A setting row BELOW the cursor: the commit inserts rows above these,
        # so a stale index is off by exactly the number inserted.
        target = next(
            index
            for index in range(view._selected + 1, len(view._rows))
            if view._rows[index].kind == "setting"
        )
        wanted = _row_id(view, target)
        view._body.scroll_to(y=max(target - 3, 0), animate=False)
        await pilot.pause()
        offset = view._body.scroll_offset.y

        class _Click:
            button = 1
            screen_x = view._body.region.x + 1
            screen_y = view._body.region.y + target - offset

            def stop(self) -> None:
                pass

        view.on_click(_Click())
        await pilot.pause()
        assert _row_id(view, view._selected) == wanted


@pytest.mark.asyncio
async def test_the_delete_ask_does_not_look_like_a_validation_error(tmp_path: Path) -> None:
    """Design round 2, D7 \u2014 an ASK must be distinguishable from a REPORT.

    Both occupy the detail row in the same danger ink. A question holding a
    destructive action that looks exactly like "the thing you typed was
    rejected" gets read as a report and dismissed, which leaves the chain
    undeleted and the user believing something happened. The separation is a
    marker plus a bold question clause, and the footer stops advertising the
    wrong meaning for ``esc``.
    """
    settings_io.write_chains(ConfigManager(tmp_path), {"cheap": ["anthropic/a", "openrouter/b"]})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "cheap":
                view._selected = index
                break
        view._repaint()
        await pilot.pause()

        # Footer before the ask: `esc` leaves the page.
        assert "back to conversation" in view.rendered_hints()

        await pilot.press("d")
        await pilot.pause()
        detail = view.detail_spans()

        # The ask leads with a marker no validation error has, and its question
        # clause is BOLD where an error is not.
        assert detail[0][0].startswith("\u25b8")
        assert detail[0][1].bold, detail
        # The key contract rides the same row, unbolded, so the frame separates
        # "what I am about to destroy" from "how to answer".
        assert any("esc cancels" in text and not style.bold for text, style in detail), detail

        # And the footer now says what `esc` actually does here.
        hints = view.rendered_hints()
        assert "cancel" in hints, hints
        assert "back to conversation" not in hints, hints

        # A validation error is the contrast case: no marker, not bold.
        await pilot.press("escape")
        await pilot.pause()
        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for char in "999999":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        error = view.detail_spans()
        assert error, "expected a rejection on the detail row"
        assert not error[0][0].startswith("\u25b8"), error
        assert not any(style.bold for _text, style in error), error


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
async def test_the_delete_ask_keeps_its_key_contract(size: tuple[int, int]) -> None:
    """Design round 2 D8 and round 3 D12 \u2014 the keys always survive, and the
    name is clipped only when the ROW it is painted into is genuinely too
    narrow to hold it.

    At 80 columns the line ran past the frame with the ellipsis landing outside
    it, so ``to confirm \u00b7 esc cancels`` vanished with no visible mark that
    anything had been cut \u2014 on the page's only destructive prompt. Clipping
    the least load-bearing segment is what ``_paint_pane`` already does.

    The round-3 half is the width the budget is taken AGAINST. D8's fix measured
    the ask with ``_list_width()``, which subtracts the read-only pane \u2014 but
    the detail row is a full-width ``Static`` that never loses those cells. At
    100 columns the ask got 60 of its 96 cells and clipped
    ``openrouter-budget-fallback`` to ``openrou\u2026`` with 29 cells of the row
    empty, while the chain row directly above showed the same name in full
    (design round 3, D12). This test ran at 80x24 ONLY \u2014 the one width where
    the pane is hidden and the two figures agree within two cells \u2014 so the
    defect was invisible to it. Hence the sizes: 100 and 140 are where the pane
    is visible, which is where the defect lived.
    """
    chain = "openrouter-budget-fallback"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        view._confirm_delete = chain
        view._repaint()
        await pilot.pause()
        question, contract = view._confirm_parts()
        # Both keys survive at every width. Which RUNG is chosen is a width
        # decision \u2014 the terse form is the point of the ladder \u2014 so the
        # assertion is that the answer is still statable, not that a
        # particular wording was used.
        assert "d" in contract and "esc cancels" in contract, contract
        line = view.render_lines_for_test()[-1]
        assert "esc cancels" in line, line
        if view._pane_fits():
            # Wide enough for the whole name, so it must not be clipped by a
            # budget taken from a narrower widget than the one painting it.
            # Asserted BEFORE the width helpers below, so that reverting the fix
            # fails on the user-visible symptom rather than on a missing method.
            assert chain in question, (
                f"the chain name was clipped to {question!r} in a "
                f"{view._detail.size.width}-cell row with room for it"
            )
        else:
            # 80 columns genuinely cannot hold the full name plus both keys, so
            # the name is the segment that gives way \u2014 D8's original case.
            assert "\u2026" in question, "the chain name was not the segment that gave way"
        # And the line fits the row it is painted into, which is D8's invariant.
        assert cell_len(line) <= view._detail.size.width, (cell_len(line), line)
        assert cell_len(line) <= view._detail_width(), (cell_len(line), view._detail_width(), line)


@pytest.mark.asyncio
async def test_the_editor_says_that_moving_saves() -> None:
    """UX round 2, U14 \u2014 commit-on-move is right, but it has to be TAUGHT.

    The contract enumerated exactly two exits, ``enter`` and ``esc``, and a user
    reading it concludes anything else is neither. That is the opposite of most
    editors, so a value gets stored by an arrow key the page never mentioned.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select(view, "retry.baseDelayMs")
        await pilot.press("enter")
        await pilot.press("1")
        await pilot.pause()
        row = next(line for line in view.render_lines_for_test() if "Base delay" in line)
        assert "\u2191\u2193 saves" in row, row
        # The footer says it too, on the key it applies to.
        assert "saves" in view.rendered_hints(), view.rendered_hints()


@pytest.mark.asyncio
async def test_the_caret_stays_visible_in_a_long_value() -> None:
    """UX round 2, U15 \u2014 four keys' entire feedback is the caret's position.

    Past ~26 cells at 100 columns the caret fell off the right edge with the
    characters just typed, so left/right/home/end moved a position nothing on
    screen reported \u2014 the exact failure the caret was added to prevent. The
    buffer is painted as a window around the caret instead of from index 0.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select(view, "web_search.searxng_endpoint")
        await pilot.press("enter")
        await pilot.pause()
        for char in "https://searx.example.com/search":
            await pilot.press(char if char != " " else "space")
            await pilot.pause()
            row = next(line for line in view.render_lines_for_test() if "SearXNG" in line)
            assert "\u258f" in row, f"caret lost at {len(view._buffer)} chars: {row!r}"
        # And while navigating back through it.
        await pilot.press("home")
        await pilot.pause()
        for _ in range(len(view._buffer)):
            await pilot.press("right")
            await pilot.pause()
            row = next(line for line in view.render_lines_for_test() if "SearXNG" in line)
            assert "\u258f" in row, f"caret lost at index {view._caret}: {row!r}"


@pytest.mark.asyncio
async def test_choosing_a_theme_leaves_the_cursor_on_the_theme_row() -> None:
    """UX round 2, U17 \u2014 a choice row stops existing when the expansion closes.

    ``action_activate`` left ``_selected`` at the index the choice occupied, so
    picking the 34th theme put the cursor 34 rows away on an unrelated setting
    two sections down with ``r default`` lit on it. Invisible on a 2-choice
    enum, which is why it surfaced only once ``tui.theme`` grew 35 members.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select(view, "tui.theme")
        await pilot.press("enter")
        await pilot.pause()
        choices = [index for index, row in enumerate(view._rows) if row.kind == "choice"]
        assert len(choices) > 10, "expected the theme enum to be registry-sourced"
        # Pick one a long way down the list.
        view._selected = choices[10]
        view._repaint()
        await pilot.press("enter")
        await pilot.pause()
        landed = view._rows[view._selected]
        assert landed.kind == "setting" and landed.setting is not None
        assert landed.setting.key == "tui.theme", _row_id(view, view._selected)


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 5, 10, 20])
async def test_backing_out_of_a_theme_expansion_leaves_the_cursor_on_the_theme_row(
    depth: int,
) -> None:
    """UX round 3, U20 \u2014 ``esc`` out of an expansion must land where ``enter``
    does.

    U17 gave the PICK its re-resolution and left the ABANDON with none: the
    ``escape``/``_expanded`` branch cleared the expansion and repainted, leaving
    ``_selected`` at the index the choice row had occupied. Backing out of
    ``tui.theme`` from its 20th member therefore put the cursor 20 rows away, on
    an unrelated setting two sections down with ``r default`` lit on it \u2014 the
    exact frame U17 argued against, reached by the other key. Backing out is the
    MORE conservative of the two gestures, so it must not be the one that moves
    you.

    Parameterised over depth because the drift is proportional to it: a
    2-choice enum drifts one row and nobody notices, which is how this survived
    a round.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        _select(view, "tui.theme")
        await pilot.press("enter")
        await pilot.pause()
        choices = [index for index, row in enumerate(view._rows) if row.kind == "choice"]
        assert len(choices) > depth, "expected the theme enum to be registry-sourced"
        view._selected = choices[depth]
        view._repaint()
        await pilot.press("escape")
        await pilot.pause()
        # The expansion closed rather than the page, and nothing was written.
        assert view._expanded is None
        assert app.query(SettingsView), "esc closed the whole page, not the expansion"
        landed = view._rows[view._selected]
        assert landed.kind == "setting" and landed.setting is not None
        assert landed.setting.key == "tui.theme", (
            f"esc from depth {depth} drifted the cursor to " f"{_row_id(view, view._selected)}"
        )


@pytest.mark.asyncio
async def test_a_structurally_wrong_config_names_the_file_instead_of_leaking_a_typeerror(
    tmp_path: Path,
) -> None:
    """Review round 3, n3 \u2014 the generic branch must not print raw Python.

    ``values: not-a-mapping`` passes the pre-parse on purpose: the top level IS
    a mapping, which is exactly what ``_load_config`` accepts, and widening the
    check would put the two out of step. It then fails deeper with a
    ``TypeError`` that reached the page verbatim as "could not save: 'str'
    object does not support item assignment" \u2014 which names nothing the user
    can act on and does not say their file is intact.

    The data was already safe here (``origin/main`` destroys this file), so this
    pins the WORDING: the message names the config file and says it may need
    repairing, while keeping the original text for anyone reporting it.
    """
    manager = ConfigManager(tmp_path)
    config_file = tmp_path / "config.yml"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        view._manager = manager
        await pilot.pause()
        # Corrupted with the page ALREADY OPEN, holding a good snapshot \u2014 the
        # real sequence, a config another process rewrites badly underneath it.
        config_file.write_text("values: not-a-mapping\n")
        before = config_file.read_bytes()
        _select(view, "display.shimmer")
        await pilot.press("enter")
        await pilot.pause()
        message = view.error_text
        assert "config.yml" in message, message
        assert "repair" in message, message
        # And the user's bytes are still there to repair.
        assert config_file.read_bytes() == before
        assert not list(tmp_path.glob("config.yml.bad*"))


@pytest.mark.asyncio
async def test_tab_cannot_park_focus_on_the_invisible_scroll_container() -> None:
    """UX round 3, U19 \u2014 an unfocusable body, so the cursor and the keys agree.

    The page's focus chain held two members, ``SettingsView`` and the unlabelled
    ``ScrollableContainer`` behind the row list, so one ``tab`` moved focus to
    the container. The container owns the scroll keys, so the arrows then moved
    the VIEWPORT while the cursor stayed put, with no focus ring or any other
    cue \u2014 and ``enter`` still bubbled to the page, so a user looking at rows
    24-37 with the cursor stranded on row 1 pressed ``enter`` on the row they
    could see and opened an editor on a row off screen.

    Asserted on both halves: focus must not leave the page, and after a tab the
    arrows must still move the CURSOR rather than scroll underneath it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        assert view._body.can_focus is False
        before = view._selected
        for _ in range(3):
            await pilot.press("tab")
            await pilot.pause()
            focused = app.focused
            assert isinstance(focused, SettingsView), (
                "tab parked focus on "
                f"{type(focused).__name__}, which owns the scroll keys and paints no cursor"
            )
        await pilot.press("down")
        await pilot.pause()
        assert view._selected != before, "the arrows scrolled the viewport instead of the cursor"


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["enter", "space", "r", "right"])
async def test_any_other_key_disarms_a_pending_delete(tmp_path: Path, key: str) -> None:
    """UX round 2, U18 \u2014 the ask is answered by ``d`` or ``esc``, and anything
    else cancels it. It was cleared by a cursor move and by ``esc`` but not by
    a key acting on the SAME row: ``enter`` toggled the chain open underneath a
    question about deleting it, and a later ``d`` still deleted it."""
    settings_io.write_chains(ConfigManager(tmp_path), {"cheap": ["anthropic/a", "openrouter/b"]})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "cheap":
                view._selected = index
                break
        view._repaint()
        await pilot.press("d")
        await pilot.pause()
        assert view._confirm_delete == "cheap"

        await pilot.press(key)
        await pilot.pause()
        assert view._confirm_delete is None, f"{key} left the ask armed"

        # And the next `d` therefore re-ASKS rather than deleting.
        await pilot.press("d")
        await pilot.pause()
        assert "cheap" in settings_io.read_chains(ConfigManager(tmp_path))


def _painted_frame(app: OperatorApp) -> str:
    """The frame as EXPORTED, which is what a user actually sees.

    Deliberately not :meth:`SettingsView.rendered_hints`. That accessor returns
    the hints' MODEL strings, which were correct on every run of the defect this
    helper exists to catch: the clipping in D16/U21 lived strictly between the
    logical label and the width the widget was painted at, so three rounds of
    text assertions passed on frames that were missing a clause. Reading the
    exported SVG goes through the same compositor ``save_screenshot`` does, so
    what this returns is what the committed evidence frames would show.
    """
    import html
    import re

    svg = app.export_screenshot()
    return " ".join(
        html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
        for match in re.finditer(r"<text[^>]*>(.*?)</text>", svg, re.S)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (80, 24)])
async def test_a_hint_whose_label_grows_is_painted_at_its_new_width(
    size: tuple[int, int],
) -> None:
    """Design round 4 D16 / UX round 4 U21 — the ``move · saves`` clause reached
    the model but not the screen.

    ``HintButton`` is ``width: auto`` and its ordinary repaint passes
    ``layout=False``, which is right for hover (ink changes, the plain text does
    not) and wrong for a label that GREW. Opening an editor turns the move
    hint's label from ``move`` (8 cells) into ``move · saves`` (16), and through
    the no-layout path the widget stayed 8 cells wide and the new clause was
    clipped off — on the one frame where the rule it states ("an arrow key
    commits this value") first applies. Below 72 columns the row contract has
    already shed, so both carriers of that rule failed together and a plain
    cursor move committed a value with nothing on screen having said it would.

    Asserted against the PAINTED frame rather than ``rendered_hints()``. The
    whole finding is that the model string was right while the paint was wrong,
    so a model-string assertion cannot see it — and did not, for three rounds.

    Deterministic by construction rather than by retrying. In the wild this
    presented at about 2 runs in 8, because it is a race against whether some
    unrelated event forced a layout pass in the same frame and incidentally
    remeasured the widget. Settling the app fully BEFORE the label grows removes
    that other work, which turns the coin flip into the defect every time: the
    pre-fix code clips 6/6 under this setup and the fixed code 0/6.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        view.load(providers=[("anthropic", "signed in")], agents=[("coder", "role", "x")])
        # Settle everything, so the only layout work left is the one the label
        # change itself has to request. This is the deterministic trigger.
        for _ in range(4):
            await pilot.pause()
        hint = view._move_hint
        assert hint.size.width == cell_len(hint.rendered()), "the resting hint was already clipped"

        _select(view, "retry.maxRetries")
        view.action_activate()
        await pilot.pause()

        assert view.editing_key == "retry.maxRetries"
        logical = hint.rendered()
        assert "saves" in logical, "the model no longer states the rule at all"
        assert hint.size.width >= cell_len(logical), (
            f"the move hint is painted {hint.size.width} cells wide for a "
            f"{cell_len(logical)}-cell label {logical!r}, so the clause naming the rule "
            f"that an arrow key commits the value is clipped off the frame"
        )
        frame = _painted_frame(app)
        assert "saves" in frame, (
            "the painted frame does not contain `saves` anywhere, so nothing on screen "
            f"tells the user that moving off this row will commit it:\n{frame}"
        )


@pytest.mark.asyncio
async def test_r_on_an_armed_chain_row_repaints_the_disarmed_ask(tmp_path: Path) -> None:
    """UX round 4 follow-up — ``r`` disarmed the ask without repainting.

    ``action_reset`` cleared ``_confirm_delete`` and then returned early on a
    chain row, because a chain is not a ``setting``. The flag was gone and the
    frame was not redrawn, so the detail row kept asking ``press d again to
    confirm`` and the footer kept offering ``esc cancel`` for an ask that no
    longer existed — after which ``esc`` left the page instead of cancelling it
    and ``d`` re-armed instead of confirming. It fails safe (nothing is deleted)
    but it is the same "model changed, paint did not" class as D16, so it is
    asserted on the frame rather than on the flag: the flag was already correct.
    """
    settings_io.write_chains(ConfigManager(tmp_path), {"cheap": ["anthropic/a", "openrouter/b"]})
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/settings")
        await pilot.pause()
        view = app.query_one(SettingsView)
        for index, row in enumerate(view._rows):
            if row.kind == "chain" and row.chain == "cheap":
                view._selected = index
                break
        view._repaint()
        await pilot.press("d")
        await pilot.pause()
        armed = _painted_frame(app)
        assert "confirm" in armed, "the ask never reached the frame"

        await pilot.press("r")
        await pilot.pause()
        assert view._confirm_delete is None, "`r` left the ask armed"
        disarmed = _painted_frame(app)
        assert disarmed != armed, (
            "the frame after `r` is byte-identical to the armed frame, so the screen is "
            "still asking a question the page has already stopped listening for"
        )
        assert "press d again to confirm" not in disarmed, (
            "the detail row still asks `press d again to confirm` after the ask was "
            f"disarmed:\n{disarmed}"
        )
        assert "cancel" not in disarmed, (
            "the footer still offers `esc cancel` for an ask that no longer exists, so "
            f"`esc` will leave the page instead:\n{disarmed}"
        )
