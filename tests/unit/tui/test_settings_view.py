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
from textual import events

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
    assertion made against the frame alone.

    Driven through ``action_toggle`` (`space`) since #440: `enter` opens the
    two-choice expansion instead of writing, and `space` is the in-place
    accelerator that kept the one-keystroke flip. The end-to-end claim is
    unchanged — this is still "a bool reaches config.yml and survives a
    reopen", just through the key that now carries it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "display.shimmer")
        view.action_toggle()
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
    """Immediate-write's one real cost is undo; this is the mitigation.

    The flip goes through ``action_toggle`` since #440 (see
    ``test_bool_toggle_writes_and_survives_reopen``), which also puts the row
    OFF-default — the state `r` is now the only one offered in, so this test
    exercises the gating rather than being blocked by it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        _select(view, "display.terminal_title")
        view.action_toggle()
        await pilot.pause()
        assert _values(tmp_path)["display.terminal_title"] is False
        view.action_reset()
        await pilot.pause()
        assert "display.terminal_title" not in _values(tmp_path)


@pytest.mark.asyncio
async def test_reset_on_a_default_row_leaves_config_byte_identical(tmp_path: Path) -> None:
    """#440: `r` on a row already at its default must not WRITE.

    `action_reset` had no default-state guard, so the key ran `reset_setting`
    unconditionally — and `_delete` writes config.yml back whether or not the
    key was there. Pressing `r` on an untouched row rewrote the file, and on a
    machine with no config.yml at all it CREATED one out of a setting the user
    had never chosen: the page's own undo gesture was the only thing in the
    session that changed anything.

    Asserted on config.yml's BYTES, the unit the issue's audit used, rather
    than on the stored value. "The value is still the default" is true of the
    broken behaviour too — `reset_setting` deletes a key that was already
    absent and the value reads the same afterwards — so a value assertion
    passes against the very bug this pins. Only the bytes distinguish "did
    nothing" from "rewrote the file to the same meaning".

    Both halves of the boundary are covered here because they are two states,
    not one: NO FILE AT ALL (the issue's repro) and a file that exists with the
    row at its default. A guard that only handled the first would leave the
    second live.
    """
    config = tmp_path / "config.yml"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        # ---- state 1: no config.yml at all -----------------------------
        # Opening the app materialises one, so it is removed to reproduce the
        # audit's starting state exactly: a machine that has never been
        # configured, where `r` conjured 1005 bytes from nothing.
        config.unlink(missing_ok=True)
        _select(view, "retry.enabled")
        setting = settings_io.resolve_key("retry.enabled")
        assert setting is not None
        assert settings_io.is_default(view._manager, setting)

        await pilot.press("r")
        await pilot.pause()
        assert not config.exists(), (
            "`r` on a default row CREATED config.yml out of a setting the user "
            f"never chose: {config.read_bytes()!r}"
        )

        # ---- state 2: the file exists and the row is at its default -----
        # Written through a DIFFERENT setting, so the file on disk carries a
        # real non-default key while the highlighted row is still untouched —
        # the state where a rewrite would be invisible in the values but
        # visible in the bytes (and in the file's mtime).
        other = settings_io.resolve_key("retry.maxRetries")
        assert other is not None
        settings_io.write_setting(view._manager, other, 7)
        view._manager.reload()
        view._repaint()
        await pilot.pause()

        _select(view, "retry.enabled")
        assert settings_io.is_default(view._manager, setting)
        before = config.read_bytes()

        await pilot.press("r")
        await pilot.pause()

        assert config.read_bytes() == before, "`r` rewrote config.yml for a row already at default"
        # The sibling is what proves the no-op is a no-op rather than a
        # differently-shaped write that happened to round-trip.
        view._manager.reload()
        assert settings_io.read_setting(view._manager, other) == 7
        # It REPORTS rather than swallowing the press: the footer advertises
        # `r default` on this row, and a lit hint whose key does nothing
        # silently is the bug one step earlier (UX round 1, U5).
        assert "default" in view.notice_text, view.notice_text
        assert view.error_text == "", "a row at its default is not an ERROR state"


@pytest.mark.asyncio
async def test_reset_still_writes_on_an_off_default_row(tmp_path: Path) -> None:
    """The guard's other boundary: `r` MUST still reset a row that is off-default.

    Paired with the test above deliberately. A guard that refused every reset
    would pass that one perfectly while destroying the feature — the page's
    only undo — so the two are meaningful only together: one pins that `r`
    does not write when there is nothing to undo, this pins that it does when
    there is.
    """
    config = tmp_path / "config.yml"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        setting = settings_io.resolve_key("retry.enabled")
        assert setting is not None
        settings_io.write_setting(view._manager, setting, False)
        view._manager.reload()
        view._repaint()
        await pilot.pause()

        _select(view, "retry.enabled")
        assert not settings_io.is_default(view._manager, setting)
        before = config.read_bytes()

        await pilot.press("r")
        await pilot.pause()

        assert config.read_bytes() != before, "`r` did not write on an off-default row"
        view._manager.reload()
        assert "enabled" not in _values(tmp_path).get("retry", {}), (
            "`r` left the stored key in place: " f"{_values(tmp_path).get('retry')!r}"
        )
        assert settings_io.is_default(view._manager, setting)
        # A successful reset says nothing: the value column already shows the
        # change landing, so a notice here would be noise on the common path.
        assert view.notice_text == "", view.notice_text


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

        # An ordinary row still advertises `enter`. `r` is a separate question
        # since #440: it is offered only where it would act, so the row has to
        # be taken OFF its default before `r default` is expected \u2014 which is
        # `test_r_is_not_offered_and_writes_nothing_on_a_default_row`'s subject.
        # U2's own finding is unaffected: the retired row sheds both keys, and
        # a row that can act advertises what can act on it.
        view.action_jump(0)
        await pilot.pause()
        hints = view.rendered_hints()
        assert "change" in hints, hints
        assert "default" not in hints, f"`r` is advertised on a defaulted row: {hints}"

        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "4":
            await pilot.press(char)
        await pilot.press("enter")
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


# `test_an_arrow_during_an_edit_does_not_silently_discard_it` (UX round 1, U3)
# stood here. It is REPLACED, not deleted, by
# `test_an_arrow_during_an_edit_discards_and_writes_nothing` at the bottom of
# this file, whose docstring records why the premise it rested on changed
# (#440 §2.4). The pointer is left here on purpose: a reader who greps for U3
# in this file should find the reversal rather than an absence, because an
# apparently-vanished regression test is indistinguishable from a fix someone
# quietly undid.


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

    UPDATED FOR #440, in the one way the contract change requires: the commit
    that inserts the rows is now the explicit ``enter`` rather than the arrow
    itself, because movement no longer writes (see ``_settle_row``). U13's
    subject is untouched — the rows are still inserted above the cursor and the
    arrow that follows must still resolve its target from the REBUILT list. The
    spec notes (§5) that removing movement-triggered commits removes the U13
    hazard CLASS outright; this assertion keeps guarding the mechanism that
    made it safe.
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
        # The explicit accept, which is now the only gesture that commits.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()

        # The chain was committed by the accept...
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
async def test_the_editor_says_what_moving_does() -> None:
    """UX round 2's U14, INVERTED by #440 rather than dropped.

    U14's finding was that a contract enumerating exactly two exits \u2014 ``enter``
    and ``esc`` \u2014 makes a reader conclude anything else is neither, so a rule
    where movement SAVED had to be taught explicitly or values would be stored
    by a key the page never mentioned.

    Under the new contract movement CANCELS (see ``_settle_row``), which is
    what a reader of a two-exit contract already assumes and what every other
    editor does. The teaching burden moves rather than disappearing: the editor
    row states the pair plainly again, and the footer \u2014 U14's own placement,
    and the surface a user reads mid-edit \u2014 is where the arrow keys' behaviour
    is named, because someone with 0.43.x muscle memory is exactly who needs to
    see it there.
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
        assert "enter saves \u00b7 esc cancels" in row, row
        assert "\u2191\u2193 saves" not in row, f"the row still promises that moving saves: {row}"
        # The footer says what the arrows do, on the key it applies to.
        assert "cancels" in view.rendered_hints(), view.rendered_hints()


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
        # `space`, the in-place bool toggle: since #440 `enter` on a bool opens
        # the choice list and writes nothing, so it can no longer reach the
        # writer whose wording this test pins. `space` is the gesture that
        # still writes on one keystroke.
        await pilot.press("space")
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
    text = " ".join(
        html.unescape(re.sub(r"<[^>]+>", "", match.group(1)))
        for match in re.finditer(r"<text[^>]*>(.*?)</text>", svg, re.S)
    )
    # NON-BREAKING spaces, folded to ordinary ones. The exporter emits U+00A0
    # between words so the SVG's text runs keep their spacing, which means every
    # multi-word phrase a caller searches for is absent from the raw string —
    # `"no cascade configured" not in frame` was TRUE of a frame containing that
    # exact line. Folding here rather than at each call site, because the trap is
    # the helper's and a caller cannot see it: the docstring promises what the
    # user sees, and a user does not see the encoding.
    return text.replace("\xa0", " ")


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
        # The clause is `move \u00b7 cancels` since #440 rather than `move \u00b7 saves`
        # \u2014 the rule reversed, the widths did not, and U21's finding is about
        # what happens to a hint whose label GROWS, whatever it grows into.
        assert "cancels" in logical, "the model no longer states the rule at all"
        assert hint.size.width >= cell_len(logical), (
            f"the move hint is painted {hint.size.width} cells wide for a "
            f"{cell_len(logical)}-cell label {logical!r}, so the clause naming what an "
            f"arrow key does to this value is clipped off the frame"
        )
        frame = _painted_frame(app)
        assert "cancels" in frame, (
            "the painted frame does not contain `cancels` anywhere, so nothing on screen "
            f"tells the user that moving off this row will discard it:\n{frame}"
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


@pytest.mark.asyncio
async def test_enter_on_the_cascade_row_does_not_destroy_the_cascade(tmp_path: Path) -> None:
    """#440: `enter` on the failover cascade SETTING row used to open a
    free-text editor seeded with the mapping's Python repr, and committing that
    repr stored it as a STRING — `read_chains` then returned `{}` and the whole
    cascade was gone, unrecoverably, because `r` cannot restore a value that is
    no longer a mapping.

    Asserted on the STORED VALUE rather than on whether an editor opened: the
    editor is the mechanism, the destroyed config is the bug, and a test that
    only checked `view._editing` would pass against any future fall-through
    that still wrote through some other door.
    """
    chains = {"default": ["anthropic/claude-opus-5", "openrouter/deepseek"]}
    settings_io.write_chains(ConfigManager(tmp_path), dict(chains))

    def stored() -> Any:
        # The RAW value on disk, not what `read_chains` makes of it: the repr
        # write is only visible before `read_chains` discards it.
        return _values(tmp_path)["retry"]["fallbackChains"]

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.fallbackChains")
        # A real key press through the app's own binding, not a direct call:
        # the fall-through was on the activation path a user actually travels.
        await pilot.press("enter")
        await pilot.pause()
        # Type a character and accept it, which is what turned the seeded repr
        # into a committed value. Harmless once the row no longer opens an
        # editor, and the only way to reach the destructive commit if it does.
        await pilot.press("x")
        await pilot.press("enter")
        await pilot.pause()

        view._manager.reload()
        assert isinstance(stored(), dict), (
            "retry.fallbackChains is no longer a mapping after `enter` on the cascade "
            f"row — the cascade has been destroyed: {stored()!r}"
        )
        assert settings_io.read_chains(view._manager) == chains, (
            "the cascade did not survive `enter` on its row: "
            f"{settings_io.read_chains(view._manager)!r}"
        )


@pytest.mark.asyncio
async def test_enter_on_the_cascade_row_opens_its_own_editor(tmp_path: Path) -> None:
    """The other half of #440: `enter` must not merely be inert. The footer
    offers `enter change` on this row, so the key travels INTO the cascade's
    own two-level editor — a lit hint whose key does nothing is the same
    complaint one step earlier."""
    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/a"]})

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.fallbackChains")
        await pilot.press("enter")
        await pilot.pause()
        assert view._editing is None, "a text editor opened on a row with no scalar value"
        current = view._current()
        assert current is not None and current.kind == "chain" and current.chain == "default"

        # And it works from there: the chain opens into its hops.
        await pilot.press("enter")
        await pilot.pause()
        assert view._chain == "default"


@pytest.mark.asyncio
async def test_begin_edit_refuses_a_kind_that_is_not_edited_as_text(tmp_path: Path) -> None:
    """The belt-and-braces guard behind #440.

    Every non-text kind has its own `action_activate` branch, so `_begin_edit`
    is only reachable on one by a MISSING branch — and the cost of that
    omission was a destroyed config, not a dead key. Driven directly at
    `_begin_edit`, because the whole point is what happens when the branch that
    should have caught the row is not there.
    """
    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/a"]})

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        for kind in (Kind.CASCADE, Kind.BOOL, Kind.ENUM, Kind.READONLY):
            index = next(
                (
                    index
                    for index, row in enumerate(view._rows)
                    if row.kind == "setting"
                    and row.setting is not None
                    and row.setting.kind is kind
                ),
                None,
            )
            assert index is not None, f"no shipped row of kind {kind}"
            view._begin_edit(view._rows[index])
            assert view._editing is None, f"_begin_edit opened a text editor on {kind}"
            assert view._error, f"_begin_edit refused {kind} silently"

        # A kind that IS edited as text still opens, so the guard did not turn
        # the editor off wholesale.
        index = _select(view, "retry.maxRetries")
        view._begin_edit(view._rows[index])
        assert view._editing == "retry.maxRetries"


#: The exact bytes v0.43.10 left in a victim's `config.yml`. The pre-#440 page
#: seeded its free-text editor with `str(mapping)`, so what got stored is the
#: mapping's Python repr with whatever the user typed appended — reproduced
#: here through the page's own writer rather than hand-written into YAML, so
#: this pins the state the shipped bug actually produced.
_CORRUPT_CASCADE = "{'default': ['anthropic/claude-opus-5', 'openrouter/deepseek']}x"


def _corrupt_the_cascade(tmp_path: Path) -> None:
    """Store the #440 wreckage the way the shipped bug stored it.

    Goes through `settings_io.coerce` + `write_setting` — the exact pair
    `_commit_edit` calls — because a test that wrote the string straight into
    `config.yml` would prove nothing about whether that state is reachable.
    `coerce` returns the text unchanged for a kind it has no parser for, and
    `validate` has no `Kind.CASCADE` arm, which together are why a repr could be
    committed over a mapping in the first place.
    """
    setting = settings_io.resolve_key("retry.fallbackChains")
    assert setting is not None
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, setting, settings_io.coerce(setting, _CORRUPT_CASCADE))
    assert _values(tmp_path)["retry"]["fallbackChains"] == _CORRUPT_CASCADE


@pytest.mark.asyncio
async def test_r_clears_a_malformed_cascade_and_the_frame_does_not_contradict_itself(
    tmp_path: Path,
) -> None:
    """A user already hit by #440 can recover, and the page stops lying to them.

    Two halves of one defect (UX round 1, U1). The page showed the corrupt
    Python repr in the VALUE column while the group line directly underneath
    read `no cascade configured` — two contradictory statements one row apart,
    from which a user cannot tell whether their chains exist. And `r`, which
    the footer advertises on this row and which is the page's documented
    mitigation for immediate-write having no undo, returned before
    `reset_setting`: the corrupt string survived and the notice told the user
    to delete a chain with `d` on a row that is not painted, because
    `read_chains` returns `{}` for an unreadable value.
    """
    _corrupt_the_cascade(tmp_path)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.fallbackChains")
        view._scroll_to_selection()
        await pilot.pause()

        # Half one: the frame. Read as EXPORTED, so this asserts about what the
        # compositor actually painted rather than about a model string.
        corrupt_frame = _painted_frame(app)
        assert "{'default'" not in corrupt_frame, (
            "the value column still paints the corrupt Python repr as if it were the "
            f"setting's value:\n{corrupt_frame}"
        )
        assert "no cascade configured" not in corrupt_frame, (
            "the page says `no cascade configured` under a row whose value column is "
            f"showing something — the two halves of the row disagree:\n{corrupt_frame}"
        )
        assert "malformed cascade" in corrupt_frame, (
            "nothing on the page says the stored value is malformed, so a victim of "
            f"#440 has no way to know why their chains vanished:\n{corrupt_frame}"
        )

        # Half two: `r` — the key the footer lights on this row — repairs it.
        await pilot.press("r")
        await pilot.pause()

        view._manager.reload()
        stored = _values(tmp_path).get("retry", {})
        assert "fallbackChains" not in stored, (
            "`r` left the corrupt value in place, so the page's own documented "
            f"mitigation cannot fix the config it is describing: {stored!r}"
        )
        assert settings_io.read_chains(view._manager) == {}
        assert (
            "malformed" in view.notice_text
        ), f"`r` cleared the value without saying so: {view.notice_text!r}"
        assert view.error_text == "", "clearing a malformed value is not an error"

        # And the recovered page reads as an ordinary empty cascade, not as a
        # third state with its own vocabulary. The detail row still REPORTS
        # what `r` just did — that is the answer to the keypress and it dies
        # with the cursor — but the row and its group line are back to the
        # unset wording.
        healed_frame = _painted_frame(app)
        assert (
            "no cascade configured" in healed_frame
        ), f"the cleared cascade does not read as an empty one:\n{healed_frame}"
        assert "press r to clear it" not in healed_frame, (
            "the row still warns about a malformed value it no longer holds, and offers "
            f"a key that would now do nothing:\n{healed_frame}"
        )


@pytest.mark.asyncio
async def test_r_on_a_healthy_cascade_still_explains_itself_without_writing(
    tmp_path: Path,
) -> None:
    """The malformed carve-out must not turn `r` destructive on a real cascade.

    A healthy cascade has no shipped default to restore — the chains are
    entirely the user's own — so `r` explains rather than deletes (UX round 1,
    U5). Asserted on the config BYTES, because "the chains are still there"
    would also pass if `r` had rewritten the file with the same content.
    """
    chains = {"default": ["anthropic/claude-opus-5", "openrouter/deepseek"]}
    settings_io.write_chains(ConfigManager(tmp_path), dict(chains))
    before = (tmp_path / "config.yml").read_bytes()

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.fallbackChains")
        await pilot.press("r")
        await pilot.pause()

        assert (tmp_path / "config.yml").read_bytes() == before, "`r` wrote to a healthy cascade"
        view._manager.reload()
        assert settings_io.read_chains(view._manager) == chains
        assert "delete a chain with d" in view.notice_text


@pytest.mark.asyncio
async def test_enter_into_the_cascade_does_not_carry_the_previous_row_notice(
    tmp_path: Path,
) -> None:
    """`_enter_cascade` settles the row it leaves (review round 1, B2).

    `r` then `enter` is a natural sequence on exactly this row: `r` answers
    with a notice pointing at `d` on a chain row, and `enter` is how the user
    gets to that chain row. Moving the cursor by assigning `_selected` skipped
    `_leave_row`, so they arrived with the previous row's instruction still on
    screen and no `d deletes it` hint for the row they were now on — the same
    "model changed, paint did not" class as the armed-delete bug. Reaching the
    same row with `down` always painted correctly, which is the comparison this
    test makes.
    """
    settings_io.write_chains(ConfigManager(tmp_path), {"default": ["anthropic/claude-opus-5"]})

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "retry.fallbackChains")
        await pilot.press("r")
        await pilot.pause()
        assert view.notice_text, "the setup press produced no notice to go stale"

        await pilot.press("enter")
        await pilot.pause()

        current = view._current()
        assert current is not None and current.kind == "chain"
        assert view.notice_text == "", (
            "the notice from `r` on the cascade row followed the cursor onto the chain "
            f"row: {view.notice_text!r}"
        )
        detail = view.render_lines_for_test()[-1]
        assert (
            "enter opens the chain" in detail
        ), f"the chain row's own contract is not on screen after `enter`: {detail!r}"


@pytest.mark.asyncio
async def test_the_text_editable_kinds_allow_list_is_pinned(tmp_path: Path) -> None:
    """The guard's carve-out has to be asserted, not merely commented.

    `_TEXT_EDITABLE_KINDS` is the belt-and-braces half of the #440 fix, and its
    membership is a judgement about each kind's INTERACTION: `Kind.LIST` is in
    it because `web_search.providers` genuinely edits as comma-separated text,
    while `Kind.CASCADE` is out because it has no scalar at all. Deleting
    `Kind.LIST` from the set left the whole settings suite green while the
    ordered-providers row became uneditable (review round 1, M1), so the set is
    pinned by NAME here and each member is exercised through the real editor
    below — a membership assertion alone would pass on a kind that no longer
    opens.
    """
    from local_operator.tui.widgets.settings_view import _TEXT_EDITABLE_KINDS

    assert _TEXT_EDITABLE_KINDS == frozenset({Kind.INT, Kind.FLOAT, Kind.TEXT, Kind.LIST}), (
        "the allow-list changed. Adding a kind here hands it to a free-text editor "
        "seeded with `str(stored_value)`, which is what destroyed the cascade in #440; "
        "removing one makes its row silently uneditable. Change this assertion only "
        "with the reason written down."
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        # One real row per allowed kind, driven through the app's own `enter`
        # binding. `web_search.providers` is named explicitly because it is the
        # ONLY `Kind.LIST` row and therefore the entire justification for that
        # member: it is the row that went uneditable with no test objecting.
        for key in ("retry.maxRetries", "retry.usageReservePercent", "hosting"):
            _select(view, key)
            await pilot.press("enter")
            await pilot.pause()
            assert view.editing_key == key, f"{key} no longer opens a text editor"
            await pilot.press("escape")
            await pilot.pause()

        _select(view, "web_search.providers")
        await pilot.press("enter")
        await pilot.pause()
        assert view.editing_key == "web_search.providers", (
            "the ordered-providers row is uneditable — `Kind.LIST` has been dropped from "
            "the allow-list, which no other test in this file notices"
        )
        # It edits as the comma-separated text the LIST comment claims, not as
        # a repr: the seed is what the guard's reasoning rests on.
        assert (
            "," in view._buffer and "[" not in view._buffer
        ), f"the providers editor is not seeded with comma-separated text: {view._buffer!r}"
        await pilot.press("escape")
        await pilot.pause()


def _cursor_on_screen(view: SettingsView) -> bool:
    """Is the highlighted row inside the body's viewport, by GEOMETRY?

    Computed here from the container's own offset and height rather than
    through ``SettingsView._cursor_visible``, so these tests assert the
    behaviour and not the existence of a helper: run against a tree without the
    fix they must fail on where the cursor and the viewport actually are, not
    on an ``AttributeError``.
    """
    offset = view._body.scroll_offset.y
    return offset <= view._selected < offset + view._body.size.height


def _wheel(widget, *, down: bool):
    """A real ``MouseScrollDown``/``Up`` aimed at ``widget``.

    Posted rather than calling the handler directly, the way
    ``test_session_picker`` does it: the wiring under test is Textual's own
    dispatch, and the defect these tests cover lives in that dispatch — the
    body's ``ScrollableContainer`` consumes the wheel and stops the event while
    it can still scroll, so a direct call to ``_scroll_rows`` exercises a path a
    real pointer only reaches at the very end of the list.
    """
    kind = events.MouseScrollDown if down else events.MouseScrollUp
    return kind(
        widget=widget,
        x=1,
        y=1,
        delta_x=0,
        delta_y=1 if down else -1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


@pytest.mark.asyncio
async def test_the_wheel_at_the_bottom_of_the_list_does_not_bounce_back() -> None:
    """The viewport HOLDS at the bottom under one more wheel notch.

    The v0.43.10 report: "you can scroll down to the bottom but then your
    selected option row is still somewhere above the top of the screen which
    ends up scrolling back up". The page held two positions for one view — the
    body container's scroll offset, which the wheel and the scrollbar drive, and
    the cursor, from which ``_scroll_to_selection`` re-derived the viewport.

    The bounce needs the container AT its limit, because that is the only state
    in which the wheel reaches the view at all: Textual stops the event on the
    container while it still has somewhere to go. At 100x30 (viewport 14,
    virtual 60, ``max_scroll_y=46``) the 24th notch moved the cursor 1 -> 2 and
    yanked the viewport from 46 back to 2.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        body = view._body

        # Premise: the list is several times its viewport, or there is no
        # bottom to arrive at and the test proves nothing.
        assert body.max_scroll_y > 0, "premise: the body scrolls"
        parked = view._selected

        # Scroll the CONTAINER to the bottom, the way a scrollbar drag does.
        body.scroll_to(y=body.max_scroll_y, animate=False, immediate=True)
        await pilot.pause()
        assert body.scroll_offset.y == body.max_scroll_y

        # One more notch, delivered as a real event over the list.
        view._list.post_message(_wheel(view._list, down=True))
        await pilot.pause()

        assert body.scroll_offset.y == body.max_scroll_y, (
            "the viewport bounced away from the bottom of the list: "
            f"scroll_y={body.scroll_offset.y} of {body.max_scroll_y}"
        )
        assert view._selected == parked, "the wheel moved the cursor instead of the view"


@pytest.mark.asyncio
async def test_the_wheel_scrolls_the_same_way_wherever_the_pointer_sits() -> None:
    """One page, one scroll model.

    The wheel used to scroll the VIEWPORT over the list (the container consumed
    it) and move the CURSOR over the title, the detail row and the side pane
    (nothing consumed it, so it bubbled to the view). Same gesture, two
    behaviours, chosen by where the pointer happened to be.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        body = view._body

        travelled: dict[str, int] = {}
        for name, target in (
            ("list", view._list),
            ("body", body),
            ("pane", view._pane_view),
            ("detail", view._detail),
            ("title", view._title),
            ("view", view),
        ):
            body.scroll_to(y=0, animate=False, immediate=True)
            view._selected = view._selectable()[0]
            view._repaint()
            await pilot.pause()
            for _ in range(3):
                target.post_message(_wheel(target, down=True))
            await pilot.pause()
            travelled[name] = body.scroll_offset.y
            assert (
                view._selected == view._selectable()[0]
            ), f"the wheel over the {name} moved the cursor"

        assert (
            len(set(travelled.values())) == 1
        ), f"the wheel travels different distances by pointer position: {travelled}"
        assert travelled["list"] > 0, "the wheel did not scroll at all"


@pytest.mark.asyncio
async def test_the_wheel_leaves_the_cursor_where_the_user_put_it(tmp_path: Path) -> None:
    """The wheel moves the VIEW only: it never moves, and never acts on, a row.

    The deliberate consequence of the model is that the cursor and the viewport
    can diverge — the user scrolls away and the highlighted row goes off screen,
    exactly as it does in an editor. What must NOT follow is a write: a wheel
    gesture is a look, so it may not toggle a bool, open an editor, or reset
    anything, and the config file is the assertion that proves it.

    KNOWN GAP, deliberately not closed here. Once the cursor is off screen, a
    subsequent `enter` or `r` still acts on it rather than on the row the user
    is looking at — the U19 hazard, reachable now by scrolling instead of by
    the focus slip that motivated `_body.can_focus = False`. The obvious guard
    ("the first press re-centres, the second acts") lands squarely on the
    activate/commit paths that the incoming edit-mode redesign rewrites, so it
    belongs there rather than as a second, competing rule this PR would leave
    behind. Pinned as a gap so the redesign inherits a statement of it instead
    of a surprise.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        parked = view._selected
        for _ in range(15):
            view._list.post_message(_wheel(view._list, down=True))
        await pilot.pause()

        assert view._selected == parked, "the wheel moved the cursor"
        assert not _cursor_on_screen(
            view
        ), "premise: enough wheel travel to carry the cursor off screen"
        assert view.editing_key is None, "the wheel opened an editor"
        assert _values(tmp_path) == {}, "a wheel gesture wrote to the config"

        # And back up: the view returns, the cursor never having moved.
        for _ in range(15):
            view._list.post_message(_wheel(view._list, down=False))
        await pilot.pause()
        assert view._body.scroll_offset.y == 0
        assert view._selected == parked
        assert _values(tmp_path) == {}


@pytest.mark.asyncio
async def test_the_arrows_still_scroll_the_cursor_into_view() -> None:
    """The cursor keeps its own claim on the viewport, for the keys that move it.

    The wheel handing the viewport to the container must not cost the arrows
    their cursor-following: a ``down`` that moved the cursor below the fold
    without scrolling would hide the row it just selected.

    A PRESERVATION guard, not a regression test — it passes at base too, which
    is the point: this is the behaviour the scroll change had to leave intact.
    Each press is settled with a ``pause`` before the assertion, because the
    scroll is applied on the next refresh and asserting mid-flight measures a
    frame no user is ever shown (it reports a false off-screen at row 37).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        # Held `down` the length of the list, checking the cursor stays visible
        # the WHOLE way rather than only at the end — a viewport that followed
        # in jumps would pass an end-state assertion.
        for _ in range(len(view._selectable()) + 5):
            await pilot.press("down")
            await pilot.pause()
            assert _cursor_on_screen(
                view
            ), f"`down` walked the cursor off screen at row {view._selected}"
        assert view._selected == view._selectable()[-1]

        await pilot.press("home")
        await pilot.pause()
        assert _cursor_on_screen(view)
        assert view._body.scroll_offset.y == 0

        await pilot.press("end")
        await pilot.pause()
        assert _cursor_on_screen(view), "`end` did not scroll the cursor into view"


@pytest.mark.asyncio
async def test_paging_lands_on_the_true_end_of_the_list() -> None:
    """`pagedown` reaches the SAME last row `end` and held `down` reach.

    ``action_section`` is section-scoped, so paging clamped on the last HEADER
    and settled on that section's first row with five selectable rows still
    below it — the page having two answers for where it ends (#425, UX round 1,
    U3). Pinned here across sizes and with an enum expanded, because the row
    count changes with both and a fix that only holds at one size is not one.
    """
    for size in ((100, 30), (140, 40), (80, 24)):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            app._open_settings_view()
            view = app.query_one(SettingsView)
            await pilot.pause()

            await pilot.press("home")
            await pilot.pause()
            for _ in range(25):
                await pilot.press("pagedown")
            await pilot.pause()

            last = view._selectable()[-1]
            assert view._selected == last, (
                f"at {size} pagedown settled on row {view._selected}, "
                f"but the last selectable row is {last}"
            )
            assert (
                view._body.scroll_offset.y == view._body.max_scroll_y
            ), "paging reached the last row without reaching the bottom of the view"

            # And back: `pageup` reaches the first row the same way.
            for _ in range(25):
                await pilot.press("pageup")
            await pilot.pause()
            assert view._selected == view._selectable()[0]


@pytest.mark.asyncio
async def test_a_key_that_writes_reveals_its_row_before_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`space` on an OFF-SCREEN bool scrolls the row into view, then toggles it.

    The wheel moves the viewport and leaves the cursor behind, so the cursor can
    sit off screen while the user reads elsewhere. This page writes immediately:
    measured at 100x30 with the cursor parked on `retry.enabled` and the view
    wheeled 15 notches away, `enter` wrote config.yml and left all 14 painted
    rows byte-identical — a write with no feedback anywhere on the frame (UX
    round 1, U1).

    ORDERING, not end state. The assertion is made from INSIDE the write: the
    real `settings_io.write_setting` is wrapped so the viewport is sampled at
    the moment the value is stored, and the config file is asserted still
    untouched until the acted-on row is visible. An end-state check would pass
    on an implementation that wrote first and scrolled afterwards in the same
    frame, which is exactly the "nothing visibly changed" complaint.
    """
    seen: dict[str, Any] = {}
    real_write = settings_io.write_setting

    def _spy(manager: Any, setting: Any, value: Any) -> Any:
        # Sampled BEFORE the value reaches disk, so `bytes_at_write` is what the
        # file held while the row was still being revealed.
        config = tmp_path / "config.yml"
        seen["bytes_at_write"] = config.read_bytes() if config.exists() else b""
        offset = view._body.scroll_offset.y
        seen["visible_at_write"] = offset <= view._selected < offset + view._body.size.height
        return real_write(manager, setting, value)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        index = _select(view, "retry.enabled")
        view._scroll_to_selection()
        await pilot.pause()
        assert _cursor_on_screen(view), "premise: the row starts on screen"

        before = (
            (tmp_path / "config.yml").read_bytes() if (tmp_path / "config.yml").exists() else b""
        )
        for _ in range(15):
            view._list.post_message(_wheel(view._list, down=True))
        await pilot.pause()
        assert view._selected == index, "premise: the wheel left the cursor alone"
        assert not _cursor_on_screen(view), "premise: the wheel carried the cursor off screen"

        monkeypatch.setattr(settings_io, "write_setting", _spy)
        # `space`, not `enter`. Since #440 `enter` on a bool OPENS the two-choice
        # expansion and writes nothing, so it can no longer carry this
        # assertion; `space` is the retained in-place toggle and is therefore
        # the page's only remaining one-keystroke write \u2014 which makes it exactly
        # the gesture this interlock has to cover (issue #440, second comment:
        # the accelerator must not reintroduce the hazard the contract closes).
        await pilot.press("space")
        await pilot.pause()

        assert seen, "`space` on the off-screen row did not reach the write at all"
        assert seen["bytes_at_write"] == before, (
            "config was written before the acted-on row was revealed: "
            f"{seen['bytes_at_write']!r} != {before!r}"
        )
        assert seen["visible_at_write"], (
            "the write landed while the cursor was still off screen — the user "
            "saw no frame change for a setting that changed"
        )
        # And the press still DID the thing, first try: revealing must not cost
        # the key its action (the interlock this deliberately is not).
        # Stored at its nested `path`, `("retry", "enabled")`, not under the
        # flat key — the same shape the maxRetries tests above assert.
        assert _values(tmp_path)["retry"]["enabled"] is False
        assert _cursor_on_screen(view), "the revealed row did not stay on screen"


@pytest.mark.asyncio
async def test_an_expansion_opened_from_off_screen_is_revealed(tmp_path: Path) -> None:
    """An enum expanded while its row is off screen brings its CHOICES into view.

    `_scroll_to_expansion` exists so a `▾` marker never appears with nothing
    under it. Its guard only caught a group hanging off the BOTTOM edge
    (`last >= offset + height`), so with the cursor wheeled off the TOP the enum
    opened — `_expanded` set, `max_scroll_y` grown 46 -> 48 — while the viewport
    never moved and neither choice row was on screen (UX round 1, U3).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()

        _select(view, "providers.openai.api")
        view._scroll_to_selection()
        await pilot.pause()

        for _ in range(15):
            view._list.post_message(_wheel(view._list, down=True))
        await pilot.pause()
        assert not _cursor_on_screen(view), "premise: the cursor is off screen"

        await pilot.press("enter")
        await pilot.pause()

        assert view._expanded == "providers.openai.api", "premise: the enum opened"
        offset = view._body.scroll_offset.y
        height = view._body.size.height
        choices = [index for index, row in enumerate(view._rows) if row.kind == "choice"]
        assert choices, "premise: the expansion produced choice rows"
        visible = [index for index in choices if offset <= index < offset + height]
        assert visible, (
            "the enum opened with every choice off screen, so the frame did not "
            f"change and the press reads as having failed: choices={choices} "
            f"viewport={offset}..{offset + height - 1}"
        )
        # The owning row is on screen too: choices without the label that says
        # what is being chosen are meaningless.
        assert _cursor_on_screen(view), "the choices were revealed without their owning row"


@pytest.mark.asyncio
async def test_the_wheel_step_follows_the_apps_scroll_sensitivity() -> None:
    """One gesture, one speed — at whatever sensitivity the app is set to.

    The body's container handles the wheel over the list at
    `App.scroll_sensitivity_y`; this view handles it everywhere else. That is a
    per-INSTANCE attribute set in `App.__init__`, not a class constant, so a
    hardcoded step here desynchronises silently. Measured at 4.0 before the fix:
    `{list: 12, pane: 6, title: 6, detail: 6}` — the exact position-dependence
    the scroll model exists to remove, back again (review round 1, S1).
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Deliberately NOT the 2.0 default, which is the value a copied constant
        # happens to match — the drift is only observable off the default.
        app.scroll_sensitivity_y = 4.0
        app._open_settings_view()
        view = app.query_one(SettingsView)
        await pilot.pause()
        body = view._body

        travelled: dict[str, int] = {}
        for name, target in (
            ("list", view._list),
            ("pane", view._pane_view),
            ("detail", view._detail),
            ("title", view._title),
        ):
            body.scroll_to(y=0, animate=False, immediate=True)
            await pilot.pause()
            for _ in range(3):
                target.post_message(_wheel(target, down=True))
            await pilot.pause()
            travelled[name] = body.scroll_offset.y

        assert len(set(travelled.values())) == 1, (
            "at scroll_sensitivity_y=4.0 the wheel travels different distances "
            f"by pointer position: {travelled}"
        )
        assert travelled["list"] == 12, (
            "the container applies 4.0 rows per notch over the list; this view "
            f"must match it rather than a constant: {travelled}"
        )


# ---------------------------------------------------------------------------
# The editing model (#440): enter opens, enter accepts, esc cancels.
#
# These tests are the executable form of the contract in
# `~/local-operator-worktrees/settings-edit-model.md`:
#
#     Nothing on this page changes your configuration until you press `enter`
#     on the thing you want. Everything else — moving, opening, looking,
#     backing out — is free.
#
# They assert on config.yml's BYTES rather than on parsed values, the standard
# of proof #387's round-1 U1 established for this page: a rewrite that changes
# only `last_modified` is still a write, and it is still the page touching a
# file the user did not ask it to touch.
# ---------------------------------------------------------------------------


def _click_row(view: SettingsView, index: int) -> None:
    """Click the row at ``index``, the way `on_click` actually receives one.

    A hand-built ``events.Click`` does not work here: ``_index_at`` maps a
    click through the body's SCREEN coordinates, so the event has to carry
    `screen_x`/`screen_y` that land inside the painted list. This is the same
    stand-in `test_a_click_lands_on_the_row_that_was_clicked` uses, factored
    out because the #440 tests need it for a second reason \u2014 a click elsewhere
    is one of the routes that must cancel an edit and revert a preview.
    """
    offset = view._body.scroll_offset.y

    class _Click:
        button = 1
        screen_x = view._body.region.x + 1
        screen_y = view._body.region.y + index - offset

        def stop(self) -> None:
            pass

    view.on_click(_Click())


def _config_bytes(tmp_path: Path) -> bytes | None:
    """``config.yml``'s exact bytes, or None when the file does not exist.

    None is a distinct outcome from empty on purpose: the §1.3 finding is that
    `r` on an untouched machine CREATED a 1005-byte config file, so "no file"
    has to be distinguishable from "a file that happens to match".
    """
    config = tmp_path / "config.yml"
    return config.read_bytes() if config.exists() else None


async def _open_page(pilot: Any, app: OperatorApp) -> SettingsView:
    """Open ``/settings`` through the app and hand back the mounted view."""
    app._open_settings_view()
    view = app.query_one(SettingsView)
    await pilot.pause()
    return view


def _select_kind(view: SettingsView, kind: str) -> int:
    """Put the cursor on the first row of ``kind`` and return its index."""
    for index, row in enumerate(view._rows):
        if row.kind == kind and row.selectable:
            view._selected = index
            view._repaint()
            return index
    raise AssertionError(f"no selectable row of kind {kind}")


#: Every key the page can receive that is NOT the accept key, including the two
#: the spec adds to the movement set (`pageup`/`pagedown`, which are caret keys
#: today). `enter` is deliberately absent: it is the one gesture allowed to
#: write, and the whole point of the anchor test is that it is the only one.
_NON_ENTER_KEYS = (
    "up",
    "down",
    "ctrl+n",
    "ctrl+p",
    "pageup",
    "pagedown",
    "home",
    "end",
    "tab",
    "left",
    "right",
    "escape",
)

#: The rows the anchor test drives, one per contract shape: a bool (the kind
#: that wrote on a bare `enter`), an enum (the kind that was already right), an
#: int and a text row (the kinds `_leave_row` committed on a move), and the
#: cascade (the kind with no branch at all). `web_search.providers` covers LIST,
#: which coerces through a different path than the scalars.
_ANCHOR_ROWS = (
    "retry.enabled",
    "display.shimmer",
    "tool_approval_mode",
    "tui.theme",
    "retry.maxRetries",
    "web_search.searxng_endpoint",
    "web_search.providers",
    "retry.fallbackChains",
)


@pytest.mark.parametrize("key", _ANCHOR_ROWS)
@pytest.mark.parametrize("gesture", _NON_ENTER_KEYS)
@pytest.mark.asyncio
async def test_no_gesture_but_enter_ever_writes(
    tmp_path: Path, key: str, gesture: str
) -> None:
    """THE ANCHOR TEST for #440. One fresh app per gesture, and a gesture that
    is not `enter` must leave config.yml's bytes exactly as it found them.

    Driven in the state that matters rather than from rest: the row is opened
    first (`enter`, which for every kind is now a non-writing "open"), and the
    gesture is pressed into that open state — an editor with a modified buffer,
    or an expanded choice list with the cursor moved off the stored member.
    That is the state the six no-accept writes lived in: `down` on an open
    editor committed the buffer, `space` on a bool flipped it, and none of it
    was reachable from a resting row.

    A fresh app per gesture (the parametrisation, not a loop) so no case
    inherits another's state — the audit's own method, and the reason it could
    attribute each write to one keystroke.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, key)
        await pilot.pause()

        # Open the row. Under the contract this writes nothing for ANY kind,
        # which the assertion immediately below states so a failure here is
        # attributed to the open rather than to the gesture under test.
        await pilot.press("enter")
        await pilot.pause()
        opened = _config_bytes(tmp_path)
        assert opened is None, f"`enter` opening {key} wrote config.yml"

        # Put the open state somewhere a commit would be VISIBLE: a buffer that
        # differs from the stored value, or a choice cursor off the stored
        # member. Without this an accidental commit would store the value that
        # was already there and the byte comparison could not see it.
        if view.editing_key is not None:
            view._buffer = "7" if key == "retry.maxRetries" else "x"
            view._caret = len(view._buffer)
            view._repaint()
        else:
            view.action_move(1)
        await pilot.pause()
        before = _config_bytes(tmp_path)

        await pilot.press(gesture)
        await pilot.pause()
        after = _config_bytes(tmp_path)
        assert after == before, (
            f"`{gesture}` on an open {key} changed config.yml: "
            f"{before!r} -> {after!r}"
        )


@pytest.mark.parametrize("key", ("retry.enabled", "retry.maxRetries", "tui.theme"))
@pytest.mark.asyncio
async def test_the_wheel_and_a_click_elsewhere_never_write(tmp_path: Path, key: str) -> None:
    """The anchor test's mouse half. `on_click` and `_scroll_rows` both routed
    through `_leave_row`, which committed — so a click on another row stored a
    buffer the user had not accepted, and the wheel discarded the same buffer.
    Under `_settle_row` both cancel, which is what makes the page stop
    contradicting itself about what leaving a row means."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, key)
        await pilot.press("enter")
        await pilot.pause()
        if view.editing_key is not None:
            view._buffer = "7"
            view._caret = 1
            view._repaint()
        else:
            view.action_move(1)
        await pilot.pause()
        before = _config_bytes(tmp_path)

        for _ in range(3):
            view._list.post_message(_wheel(view._list, down=True))
        await pilot.pause()
        assert _config_bytes(tmp_path) == before, "the wheel wrote config.yml"

        # A click on a DIFFERENT row: the gesture `_leave_row` committed on.
        other = next(
            index
            for index in range(view._selected + 1, len(view._rows))
            if view._rows[index].kind == "setting"
        )
        view._body.scroll_to(y=max(other - 3, 0), animate=False)
        await pilot.pause()
        _click_row(view, other)
        await pilot.pause()
        assert _config_bytes(tmp_path) == before, "a click elsewhere wrote config.yml"


@pytest.mark.asyncio
async def test_an_arrow_during_an_edit_discards_and_writes_nothing(tmp_path: Path) -> None:
    """The DELIBERATE REVERSAL of `test_an_arrow_during_an_edit_does_not_silently
    _discard_it` (UX round 1, U3), which this test replaces.

    U3 was correct under its own premise and is not being undone as a mistake.
    It found that `down` on an open editor discarded a valid buffer silently and
    chose commit-on-move over discard, reasoning that silently SAVING is the
    less-bad silent outcome because `r` can undo a save and nothing can undo a
    discard. Round 2's U14 then taught the new rule in the footer.

    The operator changed the premise (#440, and
    `~/local-operator-worktrees/settings-edit-model.md` §2.4): writes are no
    longer immediate, so "which silent outcome is less bad" is no longer the
    question. Under an explicit-accept contract a discard is not a lost save,
    it is the ABSENCE of an action the user never took — and commit-on-move is
    the single rule that makes "explore a setting without changing it"
    impossible, since opening an editor to see the stored value is itself a
    gesture you have to leave.

    So: moving off an open editor cancels, and the invalid-buffer case that
    used to TRAP the cursor no longer does — `_settle_row` cannot fail, so the
    page has no state a movement key cannot leave.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

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
        # DISCARDED, not committed — and the file was never created at all.
        assert _config_bytes(tmp_path) is None, "an arrow committed a typed buffer"
        assert view.editing_key is None
        assert "retry" not in _values(tmp_path)

        # An INVALID buffer no longer holds the cursor. That behaviour existed
        # only because leaving was a WRITE and a write can be refused; a cancel
        # cannot fail, so trapping the user on the row would be friction with
        # nothing behind it.
        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "9999":
            await pilot.press(char)
        moved_from = view._selected
        await pilot.press("down")
        await pilot.pause()
        assert view.editing_key is None, "an invalid buffer trapped the cursor"
        assert view._selected != moved_from, "the cursor could not leave an invalid buffer"
        assert _config_bytes(tmp_path) is None


@pytest.mark.asyncio
async def test_wheel_and_arrow_agree_about_leaving_an_edit(tmp_path: Path) -> None:
    """The contradiction test (spec §1.2), REBASED onto the scroll model.

    The spec measured a page where `_leave_row` COMMITTED a valid buffer for
    arrows, clicks and ctrl+n/p while `_scroll_rows` DISCARDED it for the
    wheel — two rules for one gesture ("the user moved off the row"), with
    nothing on screen distinguishing them.

    The scroll fix (v0.43.17, #447) resolved half of it from the other side and
    landed first, as §7.5 said it would: the wheel now moves the VIEWPORT and
    leaves the cursor on its row, so it stopped being a "leave the row" gesture
    at all and stopped cancelling. The remaining half is this change — the
    arrows stop committing.

    So the assertion is on the property both gestures must share rather than on
    identical state, which they no longer have and should not: neither WRITES,
    which is the contract, and the arrow discards because it genuinely leaves
    the row while the wheel keeps the editor because it does not."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    outcomes: dict[str, Any] = {}
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        for name in ("arrow", "wheel"):
            _select(view, "retry.baseDelayMs")
            await pilot.press("enter")
            for _ in range(len(view._buffer)):
                await pilot.press("backspace")
            for char in "1500":
                await pilot.press(char)
            await pilot.pause()
            if name == "arrow":
                await pilot.press("down")
            else:
                for _ in range(3):
                    view._list.post_message(_wheel(view._list, down=True))
            await pilot.pause()
            outcomes[name] = _config_bytes(tmp_path)
            view._cancel_edit()
            view._repaint()

    assert outcomes["arrow"] == outcomes["wheel"], (
        f"the arrows and the wheel still disagree about an open edit: {outcomes}"
    )
    assert outcomes["arrow"] is None, "leaving an edit wrote config.yml"


@pytest.mark.asyncio
async def test_a_bool_row_expands_rather_than_toggling_on_enter(tmp_path: Path) -> None:
    """Spec §2.5. `enter` on a bool used to toggle and store on one keystroke —
    the gesture a user presses to find out WHAT a row does. It now opens the
    same two-choice expansion an enum gets, marking the stored value `●` and
    the shipped default `(default)`, and writes nothing until a choice is
    accepted."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        index = _select(view, "retry.enabled")

        await pilot.press("enter")
        await pilot.pause()
        assert _config_bytes(tmp_path) is None, "`enter` on a bool wrote config.yml"
        assert view.editing_key is None, "a bool opened a TEXT editor"

        # Two choice rows, directly under the owner, showing both states and
        # naming the shipped one — information the collapsed row does not carry.
        choices = [row for row in view._rows if row.kind == "choice"]
        assert len(choices) == 2, f"a bool expanded into {len(choices)} choices"
        painted = view.render_lines_for_test()
        expansion = "\n".join(painted[index + 3 : index + 5])
        assert "on" in expansion and "off" in expansion
        assert "(default)" in expansion, "the expansion does not name the shipped default"

        # `enter` on a choice is the accept, and it is the FIRST write.
        await pilot.press("down")
        await pilot.pause()
        assert _config_bytes(tmp_path) is None, "browsing a bool's choices wrote config.yml"
        await pilot.press("enter")
        await pilot.pause()
        assert _values(tmp_path)["retry"]["enabled"] is False
        # The cursor returns to the SETTING row (U17's fix, preserved).
        assert view.selected_key == "retry.enabled"


@pytest.mark.asyncio
async def test_space_still_toggles_a_bool_in_place(tmp_path: Path) -> None:
    """The accelerator the operator kept (spec §2.5). `enter` is the
    exploratory key and opens the safe expansion; `space` is the deliberate
    flip for a user who already knows what the row is. The footer never
    advertises `space`, so discovery goes through the safe path."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, "retry.enabled")

        await pilot.press("space")
        await pilot.pause()
        assert _values(tmp_path)["retry"]["enabled"] is False
        assert view.expanded_key is None, "`space` opened the expansion as well as toggling"

        await pilot.press("space")
        await pilot.pause()
        assert _values(tmp_path)["retry"]["enabled"] is True

        # `space` is the bool's accelerator ONLY. On a typed row it must do what
        # `enter` does — open the editor — rather than committing anything,
        # or the page gains a second contract through the back door.
        _select(view, "retry.maxRetries")
        await pilot.press("space")
        await pilot.pause()
        assert view.editing_key == "retry.maxRetries"


@pytest.mark.asyncio
async def test_activation_reveals_an_offscreen_cursor_before_acting(tmp_path: Path) -> None:
    """Issue #440's second comment, and the hazard #447 made reachable.

    The wheel moves the viewport and leaves the cursor behind, so the cursor can
    sit off screen while the user reads elsewhere. Measured on the scroll model:
    `enter` on an off-screen bool wrote config.yml with all fourteen painted
    rows byte-identical — a write the user could not see happen.

    The contract removes most of it for free (a bool `enter` now OPENS rather
    than writing), but `space` is the retained in-place toggle and would
    reintroduce the exact hazard through the accelerator. Both keys therefore
    reveal the cursor first: the config may not change until something visible
    has happened."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        for gesture in ("enter", "space"):
            _select(view, "retry.enabled")
            await pilot.pause()
            # Wheel the cursor off screen — the state only the scroll model can
            # produce, and the one no keyboard gesture can reach.
            for _ in range(40):
                view._list.post_message(_wheel(view._list, down=True))
            await pilot.pause()
            assert not _cursor_on_screen(view), "the wheel did not leave the cursor off screen"
            before = _config_bytes(tmp_path)

            await pilot.press(gesture)
            await pilot.pause()
            # Whatever the key did, the row it did it to is on screen for it.
            assert _cursor_on_screen(view), (
                f"`{gesture}` acted on a row that was never brought into view"
            )
            if gesture == "enter":
                assert _config_bytes(tmp_path) == before, "`enter` wrote off screen"
                await pilot.press("escape")
                await pilot.pause()


@pytest.mark.asyncio
async def test_the_cascade_row_never_opens_a_text_editor(tmp_path: Path) -> None:
    """Spec §1.2's BLOCKER, held closed by the new contract. The cascade row had
    no `action_activate` branch, so it fell through to `_begin_edit`, seeded a
    free-text editor with `str(mapping)`, and one arrow key committed that repr
    over the user's whole failover cascade.

    Kept as a regression net beyond the fix in #449: under this model `down`
    cannot commit anything, so the same missing branch would be cosmetic rather
    than destructive — this asserts BOTH halves, because the second is what
    stops the class of defect rather than the instance."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        manager = ConfigManager(tmp_path)
        manager.set_config_value("retry", {"fallbackChains": {"cheap": ["openrouter/qwen3"]}})
        view = await _open_page(pilot, app)
        view._manager.reload()
        view._repaint()
        _select(view, "retry.fallbackChains")

        await pilot.press("enter")
        await pilot.pause()
        assert view.editing_key is None, "the cascade row opened a text editor"

        for gesture in _NON_ENTER_KEYS:
            await pilot.press(gesture)
            await pilot.pause()
            chains = settings_io.read_chains(view._manager)
            assert chains == {"cheap": ["openrouter/qwen3"]}, (
                f"`{gesture}` on the cascade row destroyed the chains: {chains}"
            )


@pytest.mark.asyncio
async def test_r_is_not_offered_and_writes_nothing_on_a_default_row(tmp_path: Path) -> None:
    """Spec §1.3 and §4.4. `action_reset` had no default-state guard: on a
    machine with no config.yml, landing on any row and pressing the key the
    footer advertises CREATED a 1005-byte config file, with nothing on screen
    saying anything had happened (the row showed its default before and after).

    `r` is now offered only where it would do something — the same rule
    `_paint_hints` already applies to the pane hint and to read-only rows — and
    pressing it anyway is inert. Asserted on the PAINTED footer, not on
    `rendered_hints()`: the shedding ladder decides what actually reaches the
    row, and a hint can be in the string while the width sheds it (spec §7.6)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, "retry.maxRetries")
        await pilot.pause()

        assert settings_io.is_default(view._manager, settings_io.resolve_key("retry.maxRetries"))
        assert not view._reset_hint.display, "`r` is advertised on a row it cannot act on"

        await pilot.press("r")
        await pilot.pause()
        assert _config_bytes(tmp_path) is None, "`r` on a default row created config.yml"

        # Off-default, the key comes back AND the detail line names what it
        # would restore — the answer to "what does r give me?" the user needs
        # before pressing a key with no confirm.
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "4":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        assert _values(tmp_path)["retry"]["maxRetries"] == 4
        assert view._reset_hint.display, "`r` is hidden on a row it can act on"
        assert "default: 10" in view.render_lines_for_test()[-1]

        await pilot.press("r")
        await pilot.pause()
        assert "maxRetries" not in _values(tmp_path).get("retry", {})


@pytest.mark.asyncio
async def test_previewing_a_theme_writes_nothing_and_esc_restores_it(tmp_path: Path) -> None:
    """Spec §3. Preview is ADDED here, not made safe: measured on the pre-change
    tree, browsing `tui.theme` left `current_theme()` and the theme epoch
    exactly where they were, and the app repainted only on the pick.

    It is safe to add now because the model finally has a well-defined cancel.
    The load-bearing half is the RESTORE — omp's `onPreviewCancel` restoring
    `activeThemeBeforePreview` is the pattern — and it has to hold on every exit
    route, because a preview that leaks leaves the app wearing a theme its
    config file disagrees with."""
    from local_operator.tui import theme as theme_mod

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        opened_on = theme_mod.current_theme()
        _select(view, "tui.theme")

        await pilot.press("enter")
        await pilot.pause()
        # Browse until the highlighted choice is NOT the stored theme.
        for _ in range(3):
            await pilot.press("down")
            await pilot.pause()
            if theme_mod.current_theme() != opened_on:
                break
        assert theme_mod.current_theme() != opened_on, "browsing themes previewed nothing"
        assert _config_bytes(tmp_path) is None, "a preview wrote config.yml"

        await pilot.press("escape")
        await pilot.pause()
        assert theme_mod.current_theme() == opened_on, "`esc` left the preview applied"
        assert _config_bytes(tmp_path) is None


@pytest.mark.parametrize("exit_route", ("move-off", "leave-page", "click-elsewhere"))
@pytest.mark.asyncio
async def test_a_theme_preview_reverts_on_every_exit_route(
    tmp_path: Path, exit_route: str
) -> None:
    """The risk table's High row: "preview leaves the app in a theme the file
    disagrees with". Every way out of the expansion — not just `esc`, which
    `test_previewing_a_theme_writes_nothing_and_esc_restores_it` covers — has
    to put the captured theme back, so each route is driven separately rather
    than trusting one of them to stand for the rest.

    The WHEEL is deliberately not among them. Under the scroll model (#447) it
    moves the viewport and leaves the cursor on its choice, so it is not an
    exit from the group at all — see `_scroll_rows`, which records why
    reverting on a glance would be the worse surprise."""
    from local_operator.tui import theme as theme_mod

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        opened_on = theme_mod.current_theme()
        _select(view, "tui.theme")
        await pilot.press("enter")
        for _ in range(3):
            await pilot.press("down")
            await pilot.pause()
            if theme_mod.current_theme() != opened_on:
                break
        assert theme_mod.current_theme() != opened_on

        if exit_route == "move-off":
            for _ in range(60):
                await pilot.press("up")
        elif exit_route == "leave-page":
            # Torn down by the APP, the route the page's own `esc` never sees
            # (a session swap, a `/clear`) and the one where a leaked preview
            # would be least recoverable.
            app._close_settings_view()
        else:
            # A click on a row OUTSIDE the group, which leaves it exactly as an
            # arrow past its end does. A DIFFERENT setting, not the owner: the
            # owner still belongs to the expansion, so clicking it is a click
            # inside the group and correctly keeps it open.
            outside = next(
                index
                for index, row in enumerate(view._rows)
                if row.kind == "setting"
                and row.setting is not None
                and row.setting.key != "tui.theme"
            )
            view._body.scroll_to(y=max(outside - 3, 0), animate=False)
            await pilot.pause()
            _click_row(view, outside)
        await pilot.pause()

        assert theme_mod.current_theme() == opened_on, (
            f"leaving the expansion by {exit_route} kept the previewed theme"
        )
        assert _config_bytes(tmp_path) is None


@pytest.mark.asyncio
async def test_an_arrow_out_of_a_choice_group_leaves_it_rather_than_trapping(
    tmp_path: Path,
) -> None:
    """Invariant 8 (spec §6), specified to be implemented AFTER the scroll fix
    and therefore implemented here. An arrow at the end of an expansion must not
    wrap inside the group and must not stop dead: a choice group that traps the
    cursor is a dead end, and #425's list-wide wrapping rule has to survive.
    It COLLAPSES the expansion and continues, which is also the cancel."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, "retry.enabled")
        await pilot.press("enter")
        await pilot.pause()
        assert view.expanded_key == "retry.enabled"

        # Past the last choice: two `down` presses clear a two-member group.
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        assert view.expanded_key is None, "the cursor was trapped inside the choice group"
        assert view.selected_key != "retry.enabled", "the cursor did not leave the group"
        assert _config_bytes(tmp_path) is None, "leaving a choice group wrote config.yml"


@pytest.mark.asyncio
async def test_display_keys_stay_flat_through_the_choice_path(tmp_path: Path) -> None:
    """Invariant 4 by the NEW route. `display.shimmer` is a literal top-level
    dotted key, not a nesting level, and bools now reach the writer through the
    CHOICE commit rather than through the toggle — so the flat-key trap has to
    be re-proved on the path that actually carries them."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)
        _select(view, "display.shimmer")
        await pilot.press("enter")
        await pilot.pause()
        # Land on the choice that is not the stored one and accept it.
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()

        values = _values(tmp_path)
        assert values["display.shimmer"] is False, "the flat dotted key was nested"
        assert "display" not in values or not isinstance(values.get("display"), dict), (
            f"the choice path nested a flat dotted key: {values}"
        )


@pytest.mark.asyncio
async def test_layout_stability_across_choosing_and_preview() -> None:
    """Invariant 1: the page's CHROME sits at the same y in every state.

    Design round 5 pinned all six existing states at byte-identical y, and the
    model adds three more to the detail line: CHOOSING ("nothing is saved until
    you press enter"), an off-default row's `default: …` clause, and the
    previewing state — which repaints the whole app's ink and therefore makes
    the repaint happen far more often than a pick ever did.

    Measured on the WIDGET GEOMETRY rather than on the row count. The list is
    scrolled inside a fixed viewport, so expanding a group legitimately makes
    the virtual list longer; what must not move is the detail row, the footer
    and the body's own height, because those are what the reader's eye is
    anchored to and a shift in any of them is the whole page reflowing under
    them (spec §6, invariant 1; AGENTS.md "Visual validation" step 4)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        def _chrome() -> tuple[int, int, int, int]:
            return (
                view._detail.region.y,
                view._hints.region.y,
                view._body.size.height,
                view._body.size.width,
            )

        geometry: dict[str, tuple[int, int, int, int]] = {}

        _select(view, "retry.enabled")
        await pilot.pause()
        geometry["browsing"] = _chrome()

        await pilot.press("enter")
        await pilot.pause()
        geometry["choosing"] = _chrome()

        await pilot.press("down")
        await pilot.pause()
        geometry["browsed"] = _chrome()

        await pilot.press("escape")
        await pilot.pause()
        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "4":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        assert view.editing_key is None, f"the editor did not close: {view.error_text}"
        geometry["off-default"] = _chrome()

        _select(view, "tui.theme")
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("down")
        await pilot.pause()
        geometry["previewing"] = _chrome()
        await pilot.press("escape")
        await pilot.pause()
        geometry["reverted"] = _chrome()

        assert len(set(geometry.values())) == 1, (
            f"the page's chrome moves between states, reflowing under the reader: {geometry}"
        )


@pytest.mark.asyncio
async def test_the_footer_teaches_the_state_it_is_in(tmp_path: Path) -> None:
    """Spec §4.5. The footer states what the keys do RIGHT NOW, and the model
    gives it a third state. While CHOOSING, `esc` cancels the expansion rather
    than leaving the page, so advertising `back to conversation` is the same
    footer-vs-detail disagreement design round 2's D7 found; while EDITING, the
    move hint must no longer promise `saves`, which is U14's clause inverted
    rather than removed.

    Asserted on the PAINTED labels at a width that carries them, never on
    `rendered_hints()` alone — the shedding ladder is what decides which
    clauses reach the row, and asserting on the unshed string is the mistake
    review round 3 made (spec §7.6)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        await pilot.pause()
        painted = view.rendered_hints()
        assert "saves" not in painted, "the footer still promises that moving saves"
        assert "cancel" in painted, f"the editing footer does not name the cancel: {painted!r}"

        await pilot.press("escape")
        _select(view, "retry.enabled")
        await pilot.press("enter")
        await pilot.pause()
        painted = view.rendered_hints()
        assert "choose" in painted, f"the choosing footer does not name enter: {painted!r}"
        assert "back to conversation" not in painted, (
            f"`esc` is advertised as leaving the page while it cancels: {painted!r}"
        )


@pytest.mark.asyncio
async def test_the_first_discarded_edit_of_a_session_says_so_once(tmp_path: Path) -> None:
    """Spec §7.4. Commit-on-move shipped recently and, by U21's measurement, was
    never reliably announced — but a silent REVERSAL is exactly what keeps
    catching this page. The first time a session discards a modified buffer to a
    movement key, the detail line says so, in the informational ink rather than
    the danger ink (U16's distinction).

    Once per session, not per occurrence: a message that fires every time
    becomes noise on the row that also carries the user's validation errors."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for char in "7":
            await pilot.press(char)
        await pilot.press("down")
        await pilot.pause()
        assert "discarded" in view.notice_text, (
            f"the first discarded edit said nothing: {view.notice_text!r}"
        )
        assert view.error_text == "", "the migration notice was painted as an error"

        # The SECOND one is silent. The rule has been stated; repeating it on
        # every move would crowd out the errors this row exists to carry.
        _select(view, "retry.baseDelayMs")
        await pilot.press("enter")
        for char in "9":
            await pilot.press(char)
        await pilot.press("down")
        await pilot.pause()
        assert "discarded" not in view.notice_text, (
            f"the migration notice fired twice: {view.notice_text!r}"
        )


@pytest.mark.parametrize("size", [(140, 40), (100, 30), (80, 24)])
@pytest.mark.asyncio
async def test_the_detail_clause_sheds_whole_rather_than_clipping(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """#440 §4.4's shedding rule, at three widths.

    The model adds two clauses to the detail row — `· default: <value>` and,
    while choosing, `· nothing is saved until you press enter`. Both are
    appended to a help string that already competes for the row, so on a narrow
    terminal something has to give.

    It must be the WHOLE clause. Half of "nothing is saved until you press
    enter" is worse than none of it: a sentence cut mid-clause reads as a
    rendering fault rather than as an abbreviation, which is the reasoning
    design round 1's D4 used for the label budget and D8 for the delete ask.
    The clause also sheds BEFORE the help, which answers "what is this" and is
    the more load-bearing half for a user who is lost.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = await _open_page(pilot, app)

        _select(view, "tui.theme")
        await pilot.press("enter")
        await pilot.pause()
        detail = view.render_lines_for_test()[-1]
        clause = "nothing is saved until you press enter"
        # Either the clause is there whole, or it is absent — never a prefix of
        # it, which is what a clip would leave behind.
        if "nothing is saved" in detail:
            assert clause in detail, f"the choosing clause was clipped mid-sentence: {detail!r}"
            assert cell_len(detail) <= view._detail_width(), (cell_len(detail), detail)
        await pilot.press("escape")
        await pilot.pause()

        _select(view, "retry.maxRetries")
        await pilot.press("enter")
        for _ in range(len(view._buffer)):
            await pilot.press("backspace")
        for char in "4":
            await pilot.press(char)
        await pilot.press("enter")
        await pilot.pause()
        detail = view.render_lines_for_test()[-1]
        if "default:" in detail:
            assert "default: 10" in detail, f"the default clause was clipped: {detail!r}"
            assert cell_len(detail) <= view._detail_width(), (cell_len(detail), detail)
