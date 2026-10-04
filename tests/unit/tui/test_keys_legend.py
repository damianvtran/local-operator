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

import asyncio
import time

import pytest
from rich.cells import cell_len
from rich.color import Color
from rich.style import Style

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.keymap import KEY_ACTIONS
from local_operator.tui import theme
from local_operator.tui.app import RESIZE_REFIT_DELAY_S, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.keys_legend import (
    _REMAPPABLE_ROWS,
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
    # Design D7 / UX U4: a card answering "what can I press" that cannot stop
    # a runaway turn is missing its most load-bearing chord.
    for expected in ("esc", "ctrl+c"):
        assert expected in keys, f"{expected} left the legend"


def test_rows_wrap_to_the_description_column_with_a_hanging_indent() -> None:
    """The pure builder, columnar band: every row wraps at the description
    column and every continuation line is indented to it, so a long row reads
    as one entry rather than rows of a second list."""
    width = 52
    rows = [row.plain for row in build_legend_rows(width)]
    assert all(cell_len(line) <= width for line in rows), rows
    indent = " " * (key_column() + KEY_GAP)
    continuations = [line for line in rows if line.startswith(indent)]
    assert continuations, "no continuation rows at a narrow width"
    assert all(line.startswith(indent) and line.strip() for line in continuations)


def test_narrow_widths_stack_the_key_above_the_description() -> None:
    """Design D1: below the longest-word threshold the columns STACK.

    Measured at 30 columns pre-fix: a 13-cell key column inside a 22-cell box
    left 7 cells for copy whose words run to 16 (`pageup/pagedown,`), so
    `composer` came out as `compose` / `r` and the card read as broken. The
    stacked shape gives the description the full content width instead, and
    every word survives whole; the threshold itself is the rule that a
    description column is never narrower than the longest word.
    """
    rows = [row.plain for row in build_legend_rows(22)]
    assert all(cell_len(line) <= 22 for line in rows), rows
    assert "f9" in rows, "the key did not stand alone on its own row"
    assert "pageup/pagedown," in rows, "a word longer than the old column was split"
    for broken in ("compose", "r", "pagedow", "n,"):
        assert broken not in rows, f"force-broken fragment {broken!r} came back"
    # The row count is the other half of D1 (79 rows pre-fix at 30 columns).
    assert len(rows) == 51, f"the stacked row count moved: {len(rows)}"
    # The boundary: at the first width whose column holds the longest word,
    # the layout is columnar again (the key shares its line with copy).
    boundary = [row.plain for row in build_legend_rows(31)]
    assert "f9" not in boundary, "stacking outlived its threshold"
    assert any(line.strip() == "pageup/pagedown," for line in boundary)


def test_the_card_paints_copy_above_the_decoration_rung() -> None:
    """Design D3, pinned as span styles: keys `fg`, descriptions `muted`,
    headings `label` + bold — never `dim`, the 3.43:1 decoration rung the
    copy was demoted to. A future re-demotion fails here, not in a frame."""
    rows = build_legend_rows(52)
    dim = Color.parse(theme.semantic_color("dim"))
    seen: set[Color] = set()
    for row in rows:
        # Full-row styles (headings, stacked key/description rows) live on
        # `row.style`; appended cells live in spans. `Text("")` air rows
        # carry neither, and their `style` is the empty string, not a Style —
        # `getattr` keeps this a style walk rather than a type dance.
        color = getattr(row.style, "color", None)
        if color is not None:
            seen.add(color)
        for span in row.spans:
            if isinstance(span.style, Style) and span.style.color is not None:
                seen.add(span.style.color)
    assert dim not in seen, "the card's copy slipped back to the decoration rung"
    for token in ("fg", "muted", "label"):
        assert Color.parse(theme.semantic_color(token)) in seen, token
    assert isinstance(rows[0].style, Style) and rows[0].style.bold, "the heading lost its weight"


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
        # D2: at the widths where the card overflows worst, the cue is the
        # one in-card sign there is more — it must be the segment that stays.
        assert (
            legend.render_lines_for_test()[-1] == "↑↓ scroll · esc close"
        ), legend.render_lines_for_test()[-1]

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
        assert before[-1] == "↑↓ scroll · esc close", before[-1]
        await pilot.press("end")
        await pilot.pause()
        after = legend.render_lines_for_test()
        assert not any("sessions list" in line for line in after), after
        assert after[-1] == "↑↓ scroll · esc close", after[-1]
        assert not app.screen.show_vertical_scrollbar


@pytest.mark.asyncio
async def test_the_hint_keeps_the_scroll_cue_and_qualifies_the_reopen_routes() -> None:
    """Design D2 / UX U2 on the painted hint row.

    Pre-fix, measured: the cue painted only from content 34 (terminal ≥ 42
    cols) — i.e. never on the 30–41-col card, exactly where the overflow is
    worst — while the reopen route painted at every size ≥ 30 cols; and the
    hint named `?` unqualified although a focused composer types it. Now the
    cue outlives the reopen route, and the reopen route leads with `/keys`
    with `?` marked as its conditional sibling.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        assert _legend(app).render_lines_for_test()[-1] == (
            "esc close · /keys reopens · ? when not typing"
        )

    app = _app()
    async with app.run_test(size=(40, 30)) as pilot:
        await _boot(pilot, app)
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        last = _legend(app).render_lines_for_test()[-1]
        assert last == "↑↓ scroll · esc close", last


@pytest.mark.asyncio
async def test_the_open_card_refits_across_a_live_resize() -> None:
    """U1 / F1: an open card re-measures when the terminal resizes.

    Textual delivers no resize to a `width: auto` host, so the app's resize
    timer drives `_sync_overlay_layout` — the card was simply missing from
    that tuple until review round 1, and stayed 66 cells wide on a 60-col
    screen. POLLED to a deadline (test_aside's shape): the claim is that the
    fit is restored, not that it lands within some exact number of ms.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
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
        await pilot.resize_terminal(60, 24)

        deadline = time.monotonic() + 5.0
        while True:
            await pilot.pause()
            if (
                legend.region.width <= app.screen.size.width - 2
                and legend.region.right <= app.screen.region.right
                and app.screen.virtual_size == app.screen.size
            ):
                break
            assert time.monotonic() < deadline, (
                f"the card never re-measured: card={legend.region} "
                f"screen={app.screen.size} right={app.screen.region.right}"
            )
            await asyncio.sleep(RESIZE_REFIT_DELAY_S)


@pytest.mark.asyncio
async def test_the_open_card_refits_when_the_dock_grows() -> None:
    """F1's second trigger: the dock band moving under an open card.

    Growing the composer re-arranges the dock and emits nothing the card
    hears; the 1 Hz band tick's `_sync_overlay_layout` is the backstop. The
    defect it covers, measured: with the composer grown to ten lines the
    dock top moved to y17 while the card stayed at bottom 23 — six rows
    painted over the docked composer.
    """
    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
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
        editor.text = "\n".join(f"line {index}" for index in range(10))

        deadline = time.monotonic() + 6.0
        while True:
            await pilot.pause()
            shell = app.query_one("#input-shell").region
            if legend.region.bottom <= shell.y:
                break
            assert (
                time.monotonic() < deadline
            ), f"the card never followed the dock: card={legend.region} dock={shell}"
            await asyncio.sleep(RESIZE_REFIT_DELAY_S)


@pytest.mark.asyncio
async def test_the_live_keymap_resolves_the_remappable_rows() -> None:
    """Design D6: the two remappable rows are composed from the live keymap.

    `ctrl+n`/`ctrl+s` are the card's only rows that can go stale after a
    remap; hard-coding their defaults is how the card would start lying. The
    ids are resolved through `App.set_keymap`'s `id -> key` store, so a remap
    repaints the row — and the id strings themselves are pinned against
    `KEY_ACTIONS` so a rename cannot silently un-resolve them.
    """
    ids = {action.id for action in KEY_ACTIONS}
    for identifiers in _REMAPPABLE_ROWS.values():
        assert set(identifiers) <= ids, "a remap id left KEY_ACTIONS"

    app = _app()
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.set_keymap({"keymap.new_session": "ctrl+shift+n", "keymap.resume": "ctrl+shift+s"})
        editor = app.query_one(Editor)
        editor.focus()
        await _settle_focus(pilot, editor)
        await _type_and_submit(pilot, app, "/keys")
        for _ in range(20):
            await pilot.pause()
            if _legend(app).is_open:
                break
        painted = "\n".join(_legend(app).render_lines_for_test())
        assert "ctrl+shift+n/ctrl+shift+s" in painted, painted
        assert "ctrl+n/ctrl+s" not in painted, painted
