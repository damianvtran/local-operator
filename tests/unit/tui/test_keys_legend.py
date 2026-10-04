"""The `?`/`/keys` keyboard legend (issue #1944): reachability, the gate, the card.

The characters-first contract is the spine of this file: a typist keeps first
refusal on `?` (the composer's TextArea stops printables before any
non-priority binding resolves; the sidebar's typing-home forwards them to the
composer), so the `?` route opens the legend only where no typist can claim the
character — read-only/full-page states and gated modals — while the typed
`/keys` command covers every other state by construction.

Every test drives the real ``OperatorApp`` with its production stylesheet; a
card painted from a CSS-less host would not show what a user sees (AGENTS.md,
"Visual validation").
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.keys_legend import (
    KEY_GAP,
    LEGEND_SECTIONS,
    KeysLegend,
    build_legend_rows,
    key_column,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _app(session: FakeSession | None = None) -> OperatorApp:
    return OperatorApp(lambda: _factory(session or FakeSession()))


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _settle_focus(pilot, widget) -> None:
    for _ in range(20):
        await pilot.pause()
        if widget.has_focus:
            return
    raise AssertionError("focus never landed")


async def _open_list(pilot, app: OperatorApp) -> None:
    """F9's keyboard mode, driven by the real key."""
    await pilot.press("f9")
    sidebar = app._session_sidebar
    await _settle_focus(pilot, sidebar)


async def _type_and_submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter (test_slash_echo's path).

    The picker is dismissed first when it is showing: Enter on an open command
    picker COMPLETES the highlighted row rather than falling through, and esc
    is the key a user presses to keep what they typed.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _legend(app: OperatorApp) -> KeysLegend:
    return app.query_one(KeysLegend)


# -- the characters-first contract: `?` types where a typist can claim it ------


@pytest.mark.asyncio
async def test_the_composer_types_the_character_and_never_opens_the_legend() -> None:
    """The T14 family for `?`: a focused composer keeps first refusal.

    The TextArea stops every printable before any non-priority binding
    resolves, so this is the same refusal that protects the org chart's glyph
    legend and the sidebar's typing-home; a priority binding here would make
    `?` untypeable everywhere, which is why the chord is deliberately not one.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await pilot.press("question_mark")
        await pilot.press("k")
        await pilot.pause()
        assert editor.text == "?k", "the character stopped typing"
        assert not _legend(app).is_open, "the legend opened over a typist"


@pytest.mark.asyncio
async def test_the_list_forwards_the_character_and_never_opens_the_legend() -> None:
    """F9 mode is a typist too: the first `?` is forwarded, not eaten.

    This is the rejection evidence for a sidebar-scoped `?` (the design note's
    D1b pilot): bound there it fired on every press and made the character
    untypeable from list mode entirely, breaking the T12/T14 contract the
    sidebar shipped with #1357.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_list(pilot, app)
        assert app._session_sidebar.has_focus, "premise: f9 focused the list"
        editor = app.query_one(Editor)
        editor.load_text("keep")
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert editor.text == "?keep", "the forwarded character was eaten"
        assert editor.has_focus, "the composer did not take the keyboard back"
        assert not _legend(app).is_open


@pytest.mark.asyncio
async def test_the_org_chart_keeps_its_own_question_mark_legend() -> None:
    """The focused widget's binding wins the chain: the chart toggles its glyph
    legend on `?` and the app-level card must not open with it."""
    from local_operator.teams import TeamEditFields
    from tests.unit.tui.test_team_chart import _registry

    session = FakeSession()
    session.team_registry = _registry(TeamEditFields(name="org"))
    app = _app(session)
    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.pause()
        app._open_org_chart_view("org")
        for _ in range(6):
            await pilot.pause()
        view = app._org_chart_view
        assert view is not None, "premise: the chart opened"
        before = view._legend.display
        await pilot.press("question_mark")
        for _ in range(2):
            await pilot.pause()
        assert view._legend.display is not before, "the chart's own legend did not toggle"
        assert not _legend(app).is_open, "the chart's press leaked to the app binding"


# -- the gate: a live prompt owns the keyboard --------------------------------


@pytest.mark.asyncio
async def test_a_live_ask_picker_refuses_the_legend() -> None:
    """The measured failure the gate exists for: without it `?` opened the card
    over a live ask (the design note's D1 row), and the picker is not a pushed
    screen — `app.screen` stays the transcript — so the gate must consult the
    prompt itself (`_live_prompt`), not only the screen stack."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        question = AskQuestion(
            id="rows",
            question="Which rows should be dropped?",
            options=[
                AskOption(label="Stale", description="nothing reads them"),
                AskOption(label="Orphaned", description="no parent row"),
            ],
        )
        app.run_worker(app.request_user_choice([question]), thread=False)
        for _ in range(8):
            await pilot.pause()
        assert app._ask_screen is not None, "premise: the picker is up"

        await pilot.press("question_mark")
        await pilot.pause()
        assert not _legend(app).is_open, "the legend opened over a live question"

        # The same refusal for the command action called directly — the typed
        # route can never reach this state (the composer cannot take focus),
        # but the ACTION is one call away from the picker's own dispatch.
        app.action_toggle_keys_legend()
        await pilot.pause()
        assert not _legend(app).is_open, "the action bypassed the gate"

        await pilot.press("escape")
        for _ in range(8):
            await pilot.pause()
        assert app._ask_screen is None, "premise: the picker closed"


# -- where `?` is reachable: read-only states ----------------------------------


@pytest.mark.asyncio
async def test_a_read_only_composer_opens_the_legend_and_esc_returns() -> None:
    """A read-only composer cannot stop the character (it answers no key at
    all), so the key reaches the app binding and the card opens AND holds
    focus; esc closes it and hands focus back."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        app._set_composer_read_only(True)
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        await pilot.pause()
        legend = _legend(app)
        assert legend.is_open, "the read-only state did not reach the legend"
        assert app.focused is legend, "the card did not hold the keyboard"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not legend.is_open
        assert not editor.has_focus, "a read-only composer must not be focused back"
        app._set_composer_read_only(False)


@pytest.mark.asyncio
async def test_f9_read_only_opens_the_legend_and_the_draft_survives() -> None:
    """The list refuses to forward to a composer that cannot take focus, so the
    key reaches the app; the draft behind the list is untouched throughout."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_list(pilot, app)
        editor = app.query_one(Editor)
        editor.load_text("draft-keep")
        await pilot.pause()
        app._set_composer_read_only(True)
        await pilot.pause()
        assert app._session_sidebar.has_focus, "premise: read-only left the list focused"

        await pilot.press("question_mark")
        await pilot.pause()
        await pilot.pause()
        legend = _legend(app)
        assert legend.is_open
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not legend.is_open
        assert editor.text == "draft-keep"
        app._set_composer_read_only(False)


# -- closing: `?` toggles, esc closes, focus comes back ------------------------


@pytest.mark.asyncio
async def test_the_question_mark_toggles_the_card_closed_and_restores_focus() -> None:
    """`?` is both routes: the same key that opens the card (via the widget's
    own binding, which wins while the card holds focus) closes it — and the
    surface that had the keyboard before it opened gets it back."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        legend = _legend(app)
        assert legend.is_open
        assert app.focused is legend
        await pilot.press("question_mark")
        await pilot.pause()
        await pilot.pause()
        assert not legend.is_open, "`?` did not toggle the card closed"
        assert app.focused is editor, "the stashed focus did not come back"


# -- the typed route: /keys ----------------------------------------------------


@pytest.mark.asyncio
async def test_the_keys_command_opens_from_the_composer() -> None:
    """The char-safe route: typed text cannot be claimed by any typing surface,
    so this is how the legend is reached from every state `?` cannot serve."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        legend = _legend(app)
        assert legend.is_open
        assert app.focused is legend, "the card did not take the keyboard"
        assert editor.text == "", "the command line stayed in the composer"


@pytest.mark.asyncio
async def test_the_keys_command_opens_from_the_list() -> None:
    """From F9 the `/` is forwarded like any printable and the command runs from
    the composer it lands in — the route the design note picked for exactly this
    state."""
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _open_list(pilot, app)
        assert app._session_sidebar.has_focus, "premise: f9 focused the list"
        for key in ("slash", "k", "e", "y", "s"):
            await pilot.press(key)
            await pilot.pause()
        editor = app.query_one(Editor)
        assert editor.text == "/keys", f"the forwarded typing produced {editor.text!r}"
        if editor._picker.is_open():
            await pilot.press("escape")
            await pilot.pause()
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        assert _legend(app).is_open, "the command did not run from the list"


@pytest.mark.asyncio
async def test_the_keys_command_refuses_a_trailing_argument() -> None:
    """A refused argument must NOT open the card and must leave a notice —
    `_system_notice`'s lane, so the boot composition survives (`/usage`'s
    rule), and no user row is written above a card that never opened."""
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys now")
        await pilot.pause()
        assert not _legend(app).is_open, "an argument opened the card"
        notices = [
            block._text
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        ]
        assert "Use /keys" in notices, notices


# -- the card itself: focus, content, and the narrow-width contract -------------


@pytest.mark.asyncio
async def test_the_card_holds_focus_while_open() -> None:
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        assert _legend(app).can_focus
        assert app.focused is _legend(app)


def test_the_spec_leads_with_the_sessions_list_and_names_the_new_chord() -> None:
    """The single-source spec, pinned where it is data rather than glyphs: the
    list section leads (its chords are the ones no other surface teaches), and
    `ctrl+k` sits beside its compatibility route."""
    assert LEGEND_SECTIONS[0].heading == "the sessions list (f9)"
    keys = [key for section in LEGEND_SECTIONS for key, _ in section.rows]
    for expected in ("ctrl+a", "ctrl+o", "ctrl+k", "f10"):
        assert expected in keys, f"{expected} left the legend"


def test_rows_wrap_to_the_description_column_with_a_hanging_indent() -> None:
    """The pure builder: every row wraps at the description column and every
    continuation line is indented to it, so a long row reads as one entry."""
    width = 24
    rows = [row.plain for row in build_legend_rows(width)]
    assert all(cell_len(line) <= width for line in rows), rows
    indent = " " * (key_column() + KEY_GAP)
    continuations = [line for line in rows if line.startswith(indent)]
    assert continuations, "no continuation rows at a narrow width"
    assert all(line.startswith(indent) and line.strip() for line in continuations)


@pytest.mark.asyncio
async def test_the_narrow_card_never_exceeds_the_screen_and_scrolls_internally() -> None:
    """The spec §1.5 contract, measured on the painted card at two extremes:
    width ≤ screen−2, every painted line inside the content box, no screen
    scrollbar, and at 30×10 the content is reachable by scrolling with the
    hint row pinned as chrome."""
    # At the narrow band the card is (screen − 2) wide and scrolls.
    app = _app()
    async with app.run_test(size=(30, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        legend = _legend(app)
        assert legend.is_open
        assert legend.region.width <= app.screen.size.width - 2, (
            legend.region.width,
            app.screen.size.width,
        )
        for line in legend.render_lines_for_test():
            assert cell_len(line) <= legend.size.width, repr(line)
        assert not app.screen.show_vertical_scrollbar, "the card raised a screen scrollbar"
        assert app.screen.virtual_size == app.screen.size

    # At 30×10 the card squeezes its gutter and scrolls to the end under its
    # pinned hint; the first section scrolls out of view and the hint stays.
    app = _app()
    async with app.run_test(size=(30, 10)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        legend = _legend(app)
        assert legend.is_open
        assert legend.has_class("-squeezed"), "the gutter survived an 8-row ground"
        before = legend.render_lines_for_test()
        assert any("sessions list" in line for line in before), before
        assert before[-1] == "esc close · ? or /keys", before[-1]
        await pilot.press("end")
        await pilot.pause()
        after = legend.render_lines_for_test()
        assert not any("sessions list" in line for line in after), after
        assert after[-1] == "esc close · ? or /keys", after[-1]
        assert not app.screen.show_vertical_scrollbar
