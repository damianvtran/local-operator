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

import pytest
import yaml

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


def _values(tmp_path: Path) -> dict:
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
async def test_arrows_wrap_and_page_clamps() -> None:
    """The repo's convention: arrows WRAP (a deliberate press), page and wheel
    CLAMP (a gesture that teleported would read as the list resetting itself)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        view.action_jump(0)
        first = view._selected
        view.action_move(-1)
        assert view._selected > first, "up from the first row did not wrap"

        view.action_jump(1)
        last = view._selected
        # Paging past the end CLAMPS.
        for _ in range(20):
            view.action_section(1)
        assert view._selected <= last

        # The wheel clamps too.
        view.action_jump(0)
        top = view._selected
        for _ in range(10):
            view._scroll_rows(-1)
        assert view._selected == top


@pytest.mark.asyncio
async def test_cascade_add_reorder_and_remove(tmp_path: Path) -> None:
    """Order IS the setting — a cascade is tried top to bottom — so reordering
    has to be a first-class action, not "delete and retype in the right place"."""
    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/a", "openrouter/b"]})

    def chains() -> dict:
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
    declared = int(re.search(r"width:\s*(\d+)", block).group(1))
    assert declared == _PANE_WIDTH


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
