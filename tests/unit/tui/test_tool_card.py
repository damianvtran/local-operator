"""Tool card behaviour — the one-line guarantee, icons, expansion, keyboard.

Five contracts are defended here, each of which is a visible promise:

- a COLLAPSED card is exactly one row, at every width, in every state, with
  or without the extra segments the richer states add
- the row leads with a per-TOOL icon that is exactly one cell wide, degrades
  to a plain-unicode set when the terminal is not trusted with Nerd glyphs,
  and never displaces the name or the summary
- diff counters appear only when the tool actually reported them, tinted
  success/danger, and never as a misleading ``+0 -0``
- outcome survives a still, COLOURLESS frame: ``✓``/``✗``/``⊘`` and their
  absence separate the four states with no colour channel at all
- expansion is reachable by MOUSE and by KEYBOARD and answers either way —
  a row with nothing to reveal says so instead of ignoring the activation,
  which is how it was reported ("when I click to expand these lines,
  nothing happens")

Colours are asserted through ``theme.semantic_color`` rather than literal
hexes so a ramp change moves one file, not this suite.
"""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.color import Color, ColorTriplet
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult

from local_operator.tui import glyphs as glyph_mod
from local_operator.tui import theme as theme_mod
from local_operator.tui.glyphs import (
    NERD_TOOL_ICONS,
    PLAIN_ICON_DEFAULT,
    PLAIN_ICON_MCP,
    PLAIN_TOOL_ICONS,
    display_name,
    nerd_icons_enabled,
    tool_icon,
)
from local_operator.tui.widgets import tool_card as card_mod
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.tool_card import (
    COLLAPSE_HINT,
    EXPAND_HINT,
    EXPAND_MAX_LINES,
    ICON_ERROR,
    ICON_INTERRUPTED,
    ICON_SUCCESS,
    LIVE_HEADER_PENDING,
    LIVE_HEADER_RUNNING,
    LIVE_MAX_LINES,
    NO_OUTPUT_NOTICE,
    RUNNING_NOTICE,
    TERSE_NO_OUTPUT_NOTICE,
    ToolCard,
    compact_path,
)
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from tests.unit.tui.conftest import TCSS_PATH, StyledTranscriptApp

#: Every width the one-line guarantee is checked at: pathological narrow,
#: narrow, typical split-pane, standard, and ultrawide.
WIDTHS = (16, 20, 40, 80, 200)


def _style_at(text: Text, needle: str) -> Style:
    """The style covering the first cell of ``needle`` in a built row."""
    index = text.plain.index(needle)
    for span in text.spans:
        if span.start <= index < span.end:
            return cast(Style, span.style)
    return Style()


def _triplet(color: Color | None) -> ColorTriplet:
    """A styled span always carries a concrete color; assert the contract."""
    assert color is not None
    assert color.triplet is not None
    return color.triplet


def _card_text(card: ToolCard) -> Text:
    """The card's painted row; a ToolCard always renders a Text."""
    renderable = card.renderable
    assert isinstance(renderable, Text)
    return renderable


def _assert_fits(card: ToolCard) -> None:
    """The collapsed row fits — and stays a row — at every width."""
    for width in WIDTHS:
        row = card._build_row(width)
        assert "\n" not in row.plain
        assert cell_len(row.plain) <= width, (width, row.plain)


class _ComposerApp(App[None]):
    """A transcript AND the real composer, in the shipped DOM order.

    The order matters to what is being checked: the transcript is above the
    input, so the screen's tab ring runs cards → composer, and Shift+Tab out
    of the composer lands on the LAST action — the one that just ran, which
    is the one a user reaching backwards wants.

    It is also the only harness in which the typing passthrough is REAL: with
    no composer to escape to, a key the row should have forwarded merely falls
    through to the bindings and every assertion still passes.
    """

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        yield TranscriptView()
        yield Editor()


# --- diff counters ---------------------------------------------------------


def test_write_diff_counts_render_in_success_and_danger_tints() -> None:
    card = ToolCard("t", "write", {"path": "notes.md", "content": "x"})
    card.mark_done("Created notes.md (3 chars).", {"path": "notes.md", "added": 12, "removed": 3})
    row = card._build_row(80)

    assert "+12" in row.plain and "-3" in row.plain
    assert _triplet(_style_at(row, "+12").color) == _triplet(
        Style(color=theme_mod.semantic_color("success")).color
    )
    assert _triplet(_style_at(row, "-3").color) == _triplet(
        Style(color=theme_mod.semantic_color("danger")).color
    )
    _assert_fits(card)


def test_a_single_sided_diff_renders_only_that_side() -> None:
    """A pure insertion says ``+N`` and nothing else — no phantom ``-0``."""
    created = ToolCard("t", "write", {"path": "new.py"})
    created.mark_done("Created new.py.", {"added": 40, "removed": 0})
    assert "+40" in created._build_row(80).plain
    assert "-0" not in created._build_row(80).plain

    gutted = ToolCard("t", "edit", {"path": "old.py"})
    gutted.mark_done("Edited old.py.", {"added": 0, "removed": 9})
    assert "-9" in gutted._build_row(80).plain
    assert "+0" not in gutted._build_row(80).plain


@pytest.mark.parametrize(
    "details",
    [
        None,
        {},
        {"path": "notes.md"},
        {"added": 0, "removed": 0},
        {"added": -4, "removed": -1},
        {"added": "12", "removed": "3"},
        {"added": True, "removed": True},
        {"added": None, "removed": None},
        "not-a-mapping",
    ],
    ids=[
        "none",
        "empty",
        "path-only",
        "zero",
        "negative",
        "strings",
        "bools",
        "nulls",
        "not-a-mapping",
    ],
)
def test_unknown_counts_render_nothing(details: object) -> None:
    """Silence beats a wrong number: an unreported count shows no counter.

    ``bool`` is called out explicitly because it is an ``int`` subclass —
    ``{"added": True}`` must not quietly print ``+1``.
    """
    card = ToolCard("t", "write", {"path": "notes.md"})
    card.mark_done("Created notes.md.", details)  # type: ignore[arg-type]
    plain = card._build_row(80).plain
    assert "+" not in plain and "-" not in plain


def test_diff_counters_never_break_the_single_row() -> None:
    """Counters are meta: the cap drops them before the row can overflow."""
    card = ToolCard("t", "edit", {"path": "some/deep/path/module.py"})
    card.mark_done("Edited module.py.", {"added": 12345, "removed": 67890})
    _assert_fits(card)


# --- the write/edit diff expansion ----------------------------------------


def _diff_card() -> ToolCard:
    """A settled write card carrying a rendered unified diff in details."""
    card = ToolCard("t", "edit", {"path": "notes.md"})
    card.mark_done(
        "Edited notes.md: replaced 1 occurrence(s) of old_text.",
        {
            "path": "notes.md",
            "added": 1,
            "removed": 1,
            # diff: `--- `, `+++ `, an @@ hunk, one removed, one added.
            "diff": [
                "--- ",
                "+++ ",
                "@@ -1,4 +1,4 @@",
                " a",
                "-old line",
                "+new line",
                " d",
            ],
        },
    )
    return card


def test_diff_powers_expansion_when_the_summary_is_the_only_output() -> None:
    """A write/edit card expands to its DIFF, not to nothing.

    The tool's result text is a single sentence (the summary), so without the
    diff payload the card would advertise an expansion that reveals the same
    line. The diff is what makes ``can_expand`` true and the expanded body
    non-empty.
    """
    card = _diff_card()
    assert card.can_expand() is True
    assert card._diff is not None
    assert card.toggle_expanded() is True
    content = card._build_content(80).plain
    assert "+new line" in content
    assert "-old line" in content


def test_diff_lines_are_tinted_by_hunk_role() -> None:
    """Added lines are success, removed danger, headers/hunks muted, context dim.

    The expanded body must tell the same story as the summary pill: only the
    leading marker cell is tinted (a coloured line never reads as a wall).
    """
    card = _diff_card()
    card.toggle_expanded()
    content = card._build_content(80)
    plus_style = _style_at(content, "+new line")
    minus_style = _style_at(content, "-old line")
    ctx_style = _style_at(content, " a")
    hunk_style = _style_at(content, "@@ -1,4 +1,4 @@")
    header_style = _style_at(content, "+++")

    # Build the expected triplet from the semantic hex (``#rrggbb``).
    def _expect(semantic: str) -> ColorTriplet:
        hexv = theme_mod.semantic_color(semantic)
        return ColorTriplet(*[int(hexv[i : i + 2], 16) for i in (1, 3, 5)])

    # The added line carries the success green; the removed the danger red —
    # the same two tints the summary pill uses for +N/-N.
    assert _triplet(plus_style.color) == _expect("success")
    assert _triplet(minus_style.color) == _expect("danger")
    # Context rides dim; headers/hunks ride muted — neither success nor danger.
    assert ctx_style != plus_style and ctx_style != minus_style
    assert hunk_style != plus_style and hunk_style != minus_style
    assert header_style != plus_style and header_style != minus_style


def test_a_tool_without_a_diff_expands_to_its_raw_output() -> None:
    """Non-write tools keep the plain-output expansion (no diff in details)."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo\nthree")
    assert card.can_expand() is True
    assert card._diff is None
    card.toggle_expanded()
    content = card._build_content(80).plain
    assert "one" in content and "three" in content


# --- the one-line guarantee ------------------------------------------------


def test_every_settled_state_stays_one_row_at_every_width() -> None:
    running = ToolCard("t", "bash", {"command": "pytest tests -q"})
    _assert_fits(running)

    done = ToolCard("t", "bash", {"command": "pytest tests -q"})
    done.mark_done("42 passed in 3.10s")
    _assert_fits(done)

    failed = ToolCard("t", "grep", {"pattern": "needle"})
    failed.mark_failed("permission denied while reading the file")
    _assert_fits(failed)

    # The longest status label in the vocabulary: the state that used to
    # push a narrow row past its own card.
    interrupted = ToolCard("t", "browser", {"url": "https://example.com/a/b/c"})
    interrupted.mark_interrupted()
    _assert_fits(interrupted)


def test_collapsed_card_settles_at_one_row() -> None:
    card = ToolCard("t", "bash", {"command": "ls -la"})
    card.mark_done("a\nb\nc\nd")
    assert card.settled_rows() == 1
    assert card.spans_multiple_rows() is False


# --- expansion -------------------------------------------------------------


def test_click_expands_then_collapses_back_to_one_row() -> None:
    card = ToolCard("t", "bash", {"command": "ls -la"})
    card.mark_done("total 8\ndrwxr-xr-x  a\n-rw-r--r--  b")

    assert card.can_expand() is True
    assert card.expanded is False
    assert card._row_count == 1

    assert card.toggle_expanded() is True
    assert card.expanded is True
    # One summary row, then the CALL, then one row per output line — no reflow,
    # no wrapping. The call comes first: it is the question the card was opened
    # to answer, and an expansion that showed only the output left no state of
    # the card able to say what had run.
    assert card._row_count == 5
    assert card.spans_multiple_rows() is True
    assert card.settled_rows() == 5
    body = card._build_content(80).plain.splitlines()
    assert body[1].strip() == "command: ls -la"
    assert body[2].strip() == "total 8"
    assert body[4].strip() == "-rw-r--r--  b"

    assert card.toggle_expanded() is False
    assert card.expanded is False
    assert card._row_count == 1
    assert card.spans_multiple_rows() is False
    assert "\n" not in card._build_content(80).plain


def test_on_click_drives_the_toggle() -> None:
    """The mouse path, not just the method: a click is the whole affordance."""

    class _Click:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo")
    event = _Click()
    card.on_click(event)
    assert card.expanded is True
    assert event.stopped is True

    card.on_click(_Click())
    assert card.expanded is False


def test_activating_an_inert_card_answers_instead_of_ignoring_the_click() -> None:
    """No output means no expansion — but never silence, and never a swallow.

    Silence is what the field report was: "when I click to expand these
    lines, nothing happens". A row that offers itself as a target and then
    absorbs the click is indistinguishable from a frozen app, so the row
    answers in the hint slot. The event still bubbles, because the row did
    not consume the click for a toggle and the transcript's own click
    handling must not be starved by a row that had nothing to do.
    """

    class _Click:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("")  # a tool that returned nothing
    assert card.can_expand() is False

    event = _Click()
    card.on_click(event)
    assert card.expanded is False
    assert event.stopped is False
    assert NO_OUTPUT_NOTICE in card._build_row(80).plain
    assert card.toggle_expanded() is False


def test_an_unfinished_card_says_it_is_still_running_not_that_it_is_empty() -> None:
    """Nothing to show and nothing YET are different answers to one click.

    The state that answers rather than opens is now COMPOSING: nothing has run,
    so there is no command to show and no output to stream, and the honest
    reply is "not yet". A RUNNING card opens instead — it has both the command
    and its live output, which is strictly more than the row can hold — so it
    is checked here too, because the two live states must not give the same
    answer to the same click.
    """
    composing = ToolCard("t", "bash")
    composing._state = "composing"
    composing._render_composing()
    assert composing.can_expand() is False
    assert composing.activate() is False
    row = composing._build_row(80).plain
    assert RUNNING_NOTICE in row and NO_OUTPUT_NOTICE not in row

    running = ToolCard("t", "bash", {"command": "sleep 30"})
    assert running.can_expand() is True
    assert running.activate() is True
    assert running.expanded is True


def test_the_notice_is_one_shot_and_leaves_with_the_focus() -> None:
    """Feedback for a keystroke, not a state the row is now in."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("")
    card._set_focused(True)
    card.activate()
    assert NO_OUTPUT_NOTICE in card._build_row(80).plain
    card._set_focused(False)
    assert NO_OUTPUT_NOTICE not in card._build_row(80).plain


def test_hint_appears_only_when_expandable_and_pointed_at_or_focused() -> None:
    """Two conditions, both required: something to reveal AND the row being
    addressed — by the pointer or by the keyboard. At rest the icon and the
    card's fill are the whole affordance; printing the hint on every settled
    row is ~9 cells of permanent chrome on an 80-column terminal."""
    inert = ToolCard("t", "bash", {"command": "ls"})
    inert.mark_done("")
    inert._set_hovered(True)
    assert EXPAND_HINT not in inert._build_row(80).plain  # nothing to expand

    # A RUNNING row is expandable — it holds the command and its live output —
    # so it makes the same offer under the pointer that a settled row does.
    # A COMPOSING row is the one with nothing behind the affordance.
    running = ToolCard("t", "bash", {"command": "ls"})
    running._set_hovered(True)
    assert EXPAND_HINT in running._build_row(80).plain

    composing = ToolCard("t", "bash")
    composing._state = "composing"
    composing._render_composing()
    composing._set_hovered(True)
    assert EXPAND_HINT not in composing._build_row(80).plain  # nothing yet

    expandable = ToolCard("t", "bash", {"command": "ls"})
    expandable.mark_done("one\ntwo")
    assert EXPAND_HINT not in expandable._build_row(80).plain  # at rest: silent

    expandable._set_hovered(True)
    assert EXPAND_HINT in expandable._build_row(80).plain  # hovered: offered

    expandable._set_hovered(False)
    expandable._set_focused(True)
    assert EXPAND_HINT in expandable._build_row(80).plain  # focused: offered

    expandable.toggle_expanded()
    row = expandable._build_row(80).plain
    assert COLLAPSE_HINT in row and EXPAND_HINT not in row


def test_the_pointer_leaving_does_not_put_out_a_focused_rows_hint() -> None:
    """Two pointers, one slot. The mouse wanders; the keyboard does not, and
    the row the keyboard is on has to keep saying what Enter would do."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo")
    card._set_focused(True)
    card._set_hovered(True)
    card._set_hovered(False)
    assert EXPAND_HINT in card._build_row(80).plain


def test_hovered_hint_uses_the_dim_ramp_step() -> None:
    """When it does show, it sits at `dim` — below the summary, above the
    separators, so it reads as an offer rather than as content."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo")
    card._set_hovered(True)
    lit = _style_at(card._build_row(80), EXPAND_HINT)
    assert _triplet(lit.color) == _triplet(Style(color=theme_mod.semantic_color("dim")).color)


def test_the_trailing_slot_is_a_column_not_a_suffix() -> None:
    """Both things that can occupy the slot end at the same cell.

    Appended straight after the summary, the offer landed at a different
    column on every row and slid as the summary truncated — jogging left and
    right under the eye while the outcome beside it was pinned precisely so
    that would not happen. Measured on its RIGHT edge, which is the edge it
    shares with the status column.
    """
    right_edges = set()
    for command in ("ls", "pytest tests/unit/tui -q -x --lf", "make"):
        card = ToolCard("t", "bash", {"command": command})
        card.mark_done("one\ntwo")
        card._set_hovered(True)
        plain = card._build_row(80).plain
        right_edges.add(plain.index(EXPAND_HINT) + cell_len(EXPAND_HINT))

    # A notice is the same slot, so it lands on the same edge despite being
    # a different width.
    inert = ToolCard("t", "todo", {"x": "a"})
    inert.mark_done("")
    inert.activate()
    plain = inert._build_row(80).plain
    right_edges.add(plain.index(NO_OUTPUT_NOTICE) + cell_len(NO_OUTPUT_NOTICE))

    assert len(right_edges) == 1, right_edges


def test_the_notice_wears_the_apps_bracket_idiom() -> None:
    """Bare, the feedback reads as summary text: ``todo     a no output`` is
    one space and one colour step from the argument beside it, and with no
    colour at all it is just ``a no output``. This slot is the direct remedy
    for "nothing happens when I click", so it has to be the least ambiguous
    thing on the row — and the app already owns a bracket for chrome."""
    for notice in (NO_OUTPUT_NOTICE, RUNNING_NOTICE):
        assert notice.startswith("⟨") and notice.endswith("⟩"), notice
    # Same idiom as the affordance they stand in for, so the slot reads as
    # one slot rather than as two unrelated things sharing a cell range.
    assert EXPAND_HINT.startswith("⟨") and COLLAPSE_HINT.startswith("⟨")


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [110, 80, 60, 48, 46, 40, 30])
async def test_activating_an_inert_row_changes_the_painted_frame(width: int) -> None:
    """The reported bug, asserted the way the user experienced it.

    Not "the notice string is present" — "the frame is different afterwards".
    The first fix for this shed the notice at the D8 summary floor along with
    the expand offer, so at 46 columns pressing Enter on an inert row
    repainted the identical bytes: "when I click to expand these lines,
    nothing happens", reproduced by the very code meant to answer it.

    The offer is chrome and may go. The ANSWER to a keystroke may not, so it
    outranks the summary and shortens to its glyph rather than vanishing.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(width, 10)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "todo", {"x": "plan the sprites"})
        view.append_block(card)
        card.mark_done("")
        await pilot.pause()
        await pilot.pause()
        before = [strip.text for strip in app.screen._compositor.render_strips()]

        card.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        after = [strip.text for strip in app.screen._compositor.render_strips()]

        assert after != before, (width, after)
        row = next(line for line in after if line.strip())
        assert NO_OUTPUT_NOTICE in row or TERSE_NO_OUTPUT_NOTICE in row, (width, row)


def test_the_answer_outranks_the_summary_and_the_offer_does_not() -> None:
    """The two occupants of the slot have different priorities, and that
    difference is the fix: chrome yields to content, an answer does not."""
    narrow = 46  # under the D8 floor's pressure, above the glyph rung

    offered = ToolCard("a", "todo", {"x": "plan the sprites and the aliens"})
    offered.mark_done("one\ntwo")
    offered._set_focused(True)
    assert EXPAND_HINT not in offered._build_row(narrow).plain

    answered = ToolCard("b", "todo", {"x": "plan the sprites and the aliens"})
    answered.mark_done("")
    answered.activate()
    row = answered._build_row(narrow).plain
    assert NO_OUTPUT_NOTICE in row
    # Paid for out of the summary, which is unchanged, still on screen, and
    # the least interesting thing on a row that produced no output.
    assert "plan" in row


def test_every_rung_of_the_notice_ladder_still_fits_the_card() -> None:
    """The answer may take the summary's cells; it may not take the row's."""
    for settle in (True, False):
        for width in WIDTHS:
            card = ToolCard("t", "list_variables", {"scope": "environment"})
            if settle:
                card.mark_done("")
            card.activate()
            row = card._build_row(width)
            assert "\n" not in row.plain
            assert cell_len(row.plain) <= width, (width, settle, row.plain)


def test_a_stop_and_a_failure_never_show_the_same_text_column() -> None:
    """``interrupted`` is a constant restating ⊘, so truncating it buys
    nothing and costs discrimination: at 46 a real failure and a user stop
    both painted ``inte…`` — identical in the leftmost and longest column,
    the one the eye lands on first. Below the width that holds the word
    whole it is dropped and the glyph column carries the state alone, which
    it can, being distinguishable from ✓ and ✗ with no colour at all."""
    for width in (110, 80, 60, 48, 46, 40):
        bad = ToolCard("a", "bash", {"command": "pytest"})
        stopped = ToolCard("b", "bash", {"command": "pytest"})
        bad.mark_failed("internal error: worker died")
        stopped.mark_interrupted()
        bad_row = bad._build_row(width).plain
        stop_row = stopped._build_row(width).plain
        assert bad_row != stop_row, (width, bad_row)
        # Whatever rides in front of the glyph must not be the same string.
        assert bad_row.split("✗")[0].strip() != stop_row.split("⊘")[0].strip(), (
            width,
            bad_row,
            stop_row,
        )
        # And the word only ever appears whole.
        assert "interrupted" in stop_row or "inte" not in stop_row, (width, stop_row)


# --- icons -----------------------------------------------------------------
#
# The icon is the one part of the row whose whole value is being recognisable
# at a glance, and the one part that can render as a replacement box on a
# terminal without a patched font. Both halves are pinned.

#: The Nerd Fonts private use area. Codepoints outside it are not glyphs any
#: patched font agreed to supply, whatever they happen to look like locally.
_PUA = range(0xE000, 0xF900)


def test_every_nerd_glyph_is_one_cell_and_lives_in_the_private_use_area() -> None:
    """Two invariants the row's arithmetic and the terminal both depend on.

    Width: the row budgets the icon at exactly one cell, so a two-cell glyph
    would shift the summary budget and push the right-aligned status column
    off the card. Range: outside the PUA a "Nerd glyph" is just some ordinary
    codepoint the local font happened to have, which will be a box on the
    next machine.
    """
    for name, glyph in NERD_TOOL_ICONS.items():
        assert len(glyph) == 1, (name, glyph)
        assert ord(glyph) in _PUA, (name, hex(ord(glyph)))
        assert unicodedata.category(glyph) == "Co", (name, glyph)
        assert cell_len(glyph) == 1, (name, cell_len(glyph))
    for name, glyph in PLAIN_TOOL_ICONS.items():
        assert cell_len(glyph) == 1, (name, cell_len(glyph))
        # The fallback set is the one that has to render WITHOUT a patched
        # font, so it may not itself reach into the private use area.
        assert ord(glyph) not in _PUA, (name, hex(ord(glyph)))


def test_every_builtin_tool_has_a_glyph_in_both_sets() -> None:
    """A tool the map has not heard of falls back correctly, but a BUILTIN
    falling back is a gap in the table, not a graceful degradation."""
    builtins = {
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "todo",
        "wake",
        "list_variables",
        "read_variable",
        "browser",
    }
    assert builtins <= set(NERD_TOOL_ICONS)
    assert builtins <= set(PLAIN_TOOL_ICONS)


def test_the_gate_switches_the_whole_table_not_just_some_of_it(monkeypatch) -> None:
    """One switch, both directions, no half-Nerd row."""
    monkeypatch.delenv(glyph_mod._ENV_DISABLE, raising=False)
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: True)
    assert nerd_icons_enabled() is True
    assert tool_icon("bash") == NERD_TOOL_ICONS["bash"]

    monkeypatch.setenv(glyph_mod._ENV_DISABLE, "1")
    assert nerd_icons_enabled() is False
    assert tool_icon("bash") == PLAIN_TOOL_ICONS["bash"]

    # The settings flag gates it identically with no env var in play.
    monkeypatch.delenv(glyph_mod._ENV_DISABLE, raising=False)
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: False)
    assert nerd_icons_enabled() is False
    assert tool_icon("grep") == PLAIN_TOOL_ICONS["grep"]


def test_unknown_and_mcp_tools_resolve_to_their_own_fallbacks(monkeypatch) -> None:
    """An MCP tool is not a wrench: the row is reporting that the action came
    from a plugged-in server, which is the only thing knowable about a tool
    whose name was minted from a config file."""
    monkeypatch.setenv(glyph_mod._ENV_DISABLE, "1")
    assert tool_icon("mcp__slack_send_message") == PLAIN_ICON_MCP
    assert tool_icon("something_invented") == PLAIN_ICON_DEFAULT
    # tool_name is MODEL-controlled: a provider echoing a different case back
    # must not silently drop every row to the generic glyph.
    assert tool_icon("BASH") == PLAIN_TOOL_ICONS["bash"]
    assert tool_icon("  read  ") == PLAIN_TOOL_ICONS["read"]


def test_no_tool_icon_collides_with_the_outcome_vocabulary() -> None:
    """The head of the row and the tail of the row must not speak the same
    word. ``todo`` used to carry a check-square, the same mark the status
    column prints for "succeeded", so a todo row opened and closed with a
    check and a colourless frame could not tell the two apart. Every glyph in
    the set is a noun; a verdict belongs only at the right edge."""
    verdicts = {ICON_SUCCESS, ICON_ERROR, ICON_INTERRUPTED}
    for table in (NERD_TOOL_ICONS, PLAIN_TOOL_ICONS):
        for name, glyph in table.items():
            assert glyph not in verdicts, (name, glyph)
    assert PLAIN_ICON_MCP not in verdicts
    assert PLAIN_ICON_DEFAULT not in verdicts


def test_an_mcp_row_is_named_for_the_call_not_for_the_prefix() -> None:
    """``mcp__`` is a constant, and a constant in an 8-cell column is five
    wasted cells. Three tools from one server all rendered ``mcp__lin``,
    which is scan-by-shape failing for the tool class a user is most likely
    to have a dozen of. The plug icon already says "from a server"."""
    assert display_name("mcp__linear_create_issue") == "create_issue"
    assert display_name("mcp__linear_list_issues") == "list_issues"
    assert display_name("bash") == "bash"  # builtins pass straight through

    names = [
        # [1], not [0]: the icon is the row's first token.
        ToolCard("t", raw, {})._build_row(100).plain.split()[1]
        for raw in (
            "mcp__linear_create_issue",
            "mcp__linear_list_issues",
            "mcp__linear_get_issue",
        )
    ]
    assert len(set(names)) == 3, names
    assert not any(name.startswith("mcp") for name in names), names


def test_a_degenerate_mcp_name_keeps_whatever_it_has() -> None:
    """A blank name column is worse than a repetitive one, so the stripping
    never runs the string out. The server/tool boundary is genuinely not
    recoverable when the server's own name contains an underscore, so only
    the first segment is taken — the remainder is still the call's own
    identifier rather than the constant prefix."""
    assert display_name("mcp__") == "mcp__"
    assert display_name("mcp__slack") == "slack"
    assert display_name("mcp__my_server_do_thing") == "server_do_thing"


def test_the_icon_leads_the_row_and_displaces_neither_name_nor_summary() -> None:
    """The icon is added TO the row, not instead of part of it."""
    card = ToolCard("t", "grep", {"pattern": "needle"})
    row = card._build_row(80).plain
    assert row.startswith(tool_icon("grep") + " ")
    assert "grep" in row and "needle" in row
    _assert_fits(card)


def test_two_different_tools_do_not_share_a_row_prefix() -> None:
    """The whole point of the icon: a run of rows is told apart by shape."""
    prefixes = {
        name: ToolCard("t", name, {})._build_row(80).plain[0]
        for name in ("bash", "read", "write", "grep", "browser")
    }
    assert len(set(prefixes.values())) == len(prefixes), prefixes


def test_the_running_icon_is_the_accent_and_settles_to_dim() -> None:
    """One of the five places the accent green is spent: a still frame has to
    read "live" without the shimmer (D26)."""
    card = ToolCard("t", "bash", {"command": "sleep 5"})
    icon = tool_icon("bash")
    live = _style_at(card._build_row(80), icon)
    assert _triplet(live.color) == _triplet(Style(color=theme_mod.semantic_color("accent")).color)

    card.mark_done("ok")
    settled = _style_at(card._build_row(80), icon)
    assert _triplet(settled.color) == _triplet(Style(color=theme_mod.semantic_color("dim")).color)


# --- the still, colourless frame -------------------------------------------


def test_the_four_states_are_told_apart_with_no_colour_at_all() -> None:
    """A screenshot, a colour-blind reader, a NO_COLOR terminal, a copied
    transcript: none of them get the tint, so none of them may need it.

    Failure and "still running" are the pair that must never collapse — the
    reported freeze was two rows the user could not tell from finished ones.
    """
    running = ToolCard("a", "bash", {"command": "pytest -q"})
    ok = ToolCard("b", "bash", {"command": "pytest -q"})
    bad = ToolCard("c", "bash", {"command": "pytest -q"})
    stopped = ToolCard("d", "bash", {"command": "pytest -q"})
    ok.mark_done("66 passed")
    bad.mark_failed("1 failed")
    stopped.mark_interrupted()

    plains = {
        "running": running._build_row(80).plain,
        "success": ok._build_row(80).plain,
        "error": bad._build_row(80).plain,
        "interrupted": stopped._build_row(80).plain,
    }
    assert len(set(plains.values())) == 4, plains
    assert "✓" in plains["success"]
    assert "✗" in plains["error"] and "1 failed" in plains["error"]
    assert "⊘" in plains["interrupted"] and "interrupted" in plains["interrupted"]
    # D28: the running row's status column is EMPTY, and that absence is the
    # signal. It must not accidentally carry another state's glyph.
    assert not any(mark in plains["running"] for mark in ("✓", "✗", "⊘"))


def test_the_outcome_glyph_holds_one_column_whatever_the_duration() -> None:
    """A pass/fail column that wobbles by a cell per row is a column the eye
    reads instead of scans, which defeats right-aligning it in the first
    place. Measured from the RIGHT edge, where the column actually lives."""
    offsets = []
    for elapsed in (0.4, 9.9, 12.3, 125.0):
        card = ToolCard("t", "bash", {"command": "pytest -q"})
        card.mark_done("66 passed")
        card._duration = elapsed
        plain = card._build_row(80).plain.rstrip()
        offsets.append(len(plain) - plain.rindex("✓"))
    assert len(set(offsets)) == 1, offsets


def test_all_three_settled_outcomes_share_the_glyph_column() -> None:
    """The column has to hold across STATES, not just across durations.

    Interrupted was the exception: it rendered ``⊘ interrupted`` with no
    duration and sat six cells left of its neighbours. That is the worst row
    to leave out of the column — one Esc marks every tool still in flight, so
    the hole opened across a whole run of rows exactly where an operator
    scans to find where work stopped.
    """
    ok = ToolCard("a", "bash", {"command": "pytest -q"})
    bad = ToolCard("b", "bash", {"command": "pytest -q"})
    stopped = ToolCard("c", "bash", {"command": "pytest -q"})
    ok.mark_done("66 passed")
    bad.mark_failed("1 failed")
    stopped.mark_interrupted()

    columns = set()
    for card, mark in ((ok, "✓"), (bad, "✗"), (stopped, "⊘")):
        plain = card._build_row(80).plain
        assert mark in plain, (card._state, plain)
        columns.add(plain.index(mark))
    assert len(columns) == 1, columns

    # The reason still rides in front of the glyph, so the row says WHY
    # without moving the answer.
    assert "interrupted ⊘" in stopped._build_row(80).plain


def test_a_narrow_row_sheds_its_message_before_its_identity() -> None:
    """Which tool failed outranks what it said about failing.

    The failing row used to be the only row in a narrow ledger to lose its
    name: neighbours kept three cells of identity while it rendered
    ``<icon>  … ✗ 0.0s`` — a cell spent on an ellipsis that says nothing, and
    the one fact worth keeping thrown away to pay for it.
    """
    for width in (20, 24, 30):
        ok = ToolCard("a", "bash", {"command": "pytest"})
        bad = ToolCard("b", "bash", {"command": "pytest"})
        ok.mark_done("passed")
        bad.mark_failed("ModuleNotFoundError: no module named pygame")
        ok_row = ok._build_row(width).plain
        bad_row = bad._build_row(width).plain
        assert "bas" in bad_row, (width, bad_row)
        # Both rows give the name the same room: the failure is not penalised
        # for having something extra to say.
        assert ok_row.index("bas") == bad_row.index("bas"), (width, ok_row, bad_row)
        # And no bare ellipsis survives as a "message".
        assert " … " not in bad_row, (width, bad_row)


def test_a_message_reduced_to_an_ellipsis_is_dropped_not_printed() -> None:
    """One cell that says "there were words here" is one cell wasted.

    A message TRUNCATED to an ellipsis (``Mod…``) is fine and wanted — it
    still names the failure. What must never ship is a message that is
    nothing BUT the ellipsis, standing alone in front of the glyph.
    """
    card = ToolCard("t", "bash", {"command": "pytest"})
    card.mark_failed("ModuleNotFoundError: no module named pygame")
    for width in WIDTHS:
        row = card._build_row(width).plain
        assert " … ✗" not in row, (width, row)
        assert not row.strip().startswith("…"), (width, row)


def test_a_result_that_only_repeats_the_summary_is_not_expandable() -> None:
    """Never advertise an expansion that reveals what is already on the row."""
    card = ToolCard("t", "bash", {"command": "echo hi"})
    card.mark_done("echo hi")
    assert card.can_expand() is False


def test_expanded_output_is_capped_and_says_how_much_is_hidden() -> None:
    total = EXPAND_MAX_LINES + 20
    card = ToolCard("t", "read", {"path": "big.txt"})
    card.mark_done("\n".join(f"line {i}" for i in range(total)))
    card.toggle_expanded()

    rows = card._build_content(80).plain.splitlines()
    # summary + the one-row call + window + marker
    assert len(rows) == 1 + 1 + EXPAND_MAX_LINES + 1
    assert rows[1].strip() == "path: big.txt"
    assert rows[-1].strip() == "… 20 more lines"
    assert card._row_count == len(rows)


def test_expanded_output_never_widens_the_card() -> None:
    """Long output lines truncate; one output line is always exactly one row."""
    card = ToolCard("t", "bash", {"command": "cat wide.txt"})
    card.mark_done("x" * 500 + "\n" + "y" * 500)
    card.toggle_expanded()
    for width in WIDTHS:
        for line in card._build_content(width).plain.splitlines():
            assert cell_len(line) <= width, (width, len(line))


def test_failed_output_renders_in_the_danger_tint() -> None:
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed("exit status 1", "Traceback:\n  boom")
    card.toggle_expanded()
    content = card._build_content(80)
    assert _triplet(_style_at(content, "Traceback:").color) == _triplet(
        Style(color=theme_mod.semantic_color("danger")).color
    )


def test_failed_card_falls_back_to_the_error_as_its_output() -> None:
    """A one-line error is already on the row; a multi-line one is expandable."""
    terse = ToolCard("t", "bash", {"command": "false"})
    terse.mark_failed("exit status 1")
    assert terse.can_expand() is True  # the error differs from the summary

    detailed = ToolCard("t", "bash", {"command": "false"})
    detailed.mark_failed("exit status 1", "line one\nline two")
    detailed.toggle_expanded()
    assert "line two" in detailed._build_content(80).plain


# --- summaries -------------------------------------------------------------


def test_summary_prefers_identity_arguments_over_payload() -> None:
    """A write's row is about the FILE, not the first 60 bytes of its body."""
    card = ToolCard(
        "t",
        "write",
        {"path": "notes.md", "content": "# Heading\n\nA long body that would bury the path."},
    )
    row = card._build_row(80).plain
    assert "notes.md" in row
    assert "Heading" not in row


def test_summary_falls_back_to_scalars_for_unrecognised_tools() -> None:
    card = ToolCard("t", "mcp_thing", {"alpha": "one", "beta": "two", "gamma": "three"})
    row = card._build_row(80).plain
    assert "one two" in row
    assert "three" not in row  # at most two parts, as before


def test_summary_falls_back_to_the_tool_name_when_no_scalars_exist() -> None:
    card = ToolCard("t", "todo", {"items": [{"text": "a"}]})
    assert "todo" in card._build_row(80).plain


def test_the_row_keeps_the_arguments_when_the_model_supplied_an_intent() -> None:
    """The receipt records the FACT; the claim goes on the transient working line.

    This used to be the other way round. An intent is the model's assertion
    about what it is doing and the arguments are what it actually ran, and a
    user reads the ledger back precisely when those two might disagree — a row
    captioned with the claim hides the disagreement. It also put the same
    sentence on the card and on the working line pinned directly beneath it.
    """
    card = ToolCard("t", "write", {"path": "notes.md"}, intent="Recording the decision")
    row = card._build_row(80).plain
    assert "notes.md" in row
    assert "Recording the decision" not in row
    # Carried, though — the working line reads it from here.
    assert card.intent == "recording the decision"


def test_compact_path_shrinks_against_cwd_then_home(monkeypatch) -> None:
    monkeypatch.setattr(card_mod.os, "getcwd", lambda: "/work/project")
    monkeypatch.setenv("HOME", "/home/dev")

    assert compact_path("/work/project/src/main.py") == "src/main.py"
    assert compact_path("/home/dev/notes.md") == "~/notes.md"
    assert compact_path("/etc/hosts") == "/etc/hosts"  # nothing to shrink
    assert compact_path("relative/path.py") == "relative/path.py"
    # Prose that merely contains a slash is left exactly alone.
    assert compact_path("/work/project has two files") == "/work/project has two files"


def test_compact_path_survives_a_deleted_cwd(monkeypatch) -> None:
    """A path is still rendered when the process has no working directory."""

    def _boom() -> str:
        raise OSError("cwd gone")

    monkeypatch.setattr(card_mod.os, "getcwd", _boom)
    monkeypatch.setenv("HOME", "/home/dev")
    assert compact_path("/home/dev/notes.md") == "~/notes.md"


# --- the affordance under the real stylesheet ------------------------------


@pytest.mark.asyncio
async def test_pointing_at_a_row_lifts_its_ground_and_its_hint() -> None:
    """The click affordance is two coordinated signals, and both are live.

    At rest a row shows NO hint at all, so the background step alone would say
    nothing about what clicking does. The test moves a real pointer between
    two rows and checks the hint appears with the lifted ground and that the
    row the pointer LEFT gives both signals back.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 14)) as pilot:
        view = app.query_one(TranscriptView)
        first = ToolCard("a", "read", {"path": "one.py"})
        second = ToolCard("b", "read", {"path": "two.py"})
        view.append_block(first)
        view.append_block(second)
        first.mark_done("line one\nline two")
        second.mark_done("line one\nline two")
        await pilot.pause()

        surface = _triplet(Style(bgcolor=theme_mod.semantic_color("surface")).bgcolor)
        overlay = _triplet(Style(bgcolor=theme_mod.semantic_color("overlay")).bgcolor)
        dim = _triplet(Style(color=theme_mod.semantic_color("dim")).color)

        def ground(card: ToolCard):
            return card.styles.background.rgb

        def has_hint(card: ToolCard) -> bool:
            return EXPAND_HINT in _card_text(card).plain

        def hint_color(card: ToolCard) -> ColorTriplet:
            return _triplet(_style_at(_card_text(card), EXPAND_HINT).color)

        assert ground(first) == surface
        assert not has_hint(first)

        await pilot.hover(first)
        await pilot.pause()
        assert ground(first) == overlay
        assert has_hint(first) and hint_color(first) == dim

        await pilot.hover(second)
        await pilot.pause()
        assert ground(first) == surface
        assert not has_hint(first)  # the row the pointer left goes quiet again
        assert ground(second) == overlay
        assert has_hint(second) and hint_color(second) == dim


@pytest.mark.asyncio
async def test_outcome_reaches_the_ground_not_just_the_glyph() -> None:
    """A failed row stops being neutral; a live row sits one step proud."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 14)) as pilot:
        view = app.query_one(TranscriptView)
        running = ToolCard("a", "bash", {"command": "sleep 5"})
        ok = ToolCard("b", "read", {"path": "one.py"})
        bad = ToolCard("c", "edit", {"path": "two.py"})
        for card in (running, ok, bad):
            view.append_block(card)
        await pilot.pause()
        ok.mark_done("done")
        bad.mark_failed("boom")
        await pilot.pause()

        def expected(token: str) -> ColorTriplet:
            return _triplet(Style(bgcolor=theme_mod.semantic_color(token)).bgcolor)

        assert running.styles.background.rgb == expected("raised")
        assert ok.styles.background.rgb == expected("surface")
        assert bad.styles.background.rgb == expected("tint-danger")


# --- the keyboard path -----------------------------------------------------
#
# Expansion used to be mouse-only. That made it invisible to anyone driving
# the app from the keyboard and unreachable in a terminal with mouse
# reporting off, and it is half of what "nothing happens when I click these"
# turned out to mean. These run through a real Pilot with real keystrokes,
# because bindings only resolve against a focused widget in a live screen.


@pytest.mark.asyncio
async def test_a_focused_row_expands_and_collapses_on_enter_and_space() -> None:
    """Run against the composer harness on purpose: Space is printable, and
    without a text input in the DOM the typing passthrough cannot misroute it
    — which is exactly the configuration that hid the bug once already."""
    app = _ComposerApp()
    async with app.run_test(size=(90, 16)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "bash", {"command": "ls -la"})
        view.append_block(card)
        card.mark_done("total 8\nfile-a\nfile-b")
        await pilot.pause()

        card.focus()
        await pilot.pause()
        assert app.focused is card
        # Focus alone states the offer — the row says what Enter will do
        # before Enter is pressed.
        assert EXPAND_HINT in _card_text(card).plain

        await pilot.press("enter")
        await pilot.pause()
        assert card.expanded is True
        assert card.size.height == 5  # summary + the call + three output rows
        assert COLLAPSE_HINT in _card_text(card).plain

        await pilot.press("space")
        await pilot.pause()
        assert card.expanded is False
        assert card.size.height == 1


@pytest.mark.asyncio
async def test_up_and_down_walk_the_ledger_and_step_out_at_its_ends() -> None:
    """The arrows address ACTIONS, and the ledger is passable, not a trap.

    Non-focusable blocks between two cards are stepped OVER rather than
    stopped at: what the keys traverse is the list of things Enter can act
    on, and a stop on an inert paragraph reads as the key having failed.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 20)) as pilot:
        view = app.query_one(TranscriptView)
        first = ToolCard("a", "read", {"path": "one.py"})
        prose = NoticeBlock("a line of prose between two actions", "info")
        second = ToolCard("b", "read", {"path": "two.py"})
        for block in (first, prose, second):
            view.append_block(block)
        first.mark_done("alpha\nbeta")
        second.mark_done("gamma\ndelta")
        await pilot.pause()

        first.focus()
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.focused is second  # the notice was stepped over

        await pilot.press("up")
        await pilot.pause()
        assert app.focused is first

        # Off the top there is no earlier action: focus leaves the ledger
        # rather than sticking, so the scroll keys become reachable again.
        await pilot.press("up")
        await pilot.pause()
        assert app.focused is not first


@pytest.mark.asyncio
async def test_enter_on_an_inert_focused_row_answers_on_the_row() -> None:
    """The keyboard gets the same answer the mouse gets, from the same path."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 12)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "bash", {"command": "true"})
        view.append_block(card)
        card.mark_done("")  # returned nothing at all
        await pilot.pause()

        card.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert card.expanded is False
        assert NO_OUTPUT_NOTICE in _card_text(card).plain
        # And it is on the painted frame, not merely in the widget's content.
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        assert NO_OUTPUT_NOTICE in painted


@pytest.mark.asyncio
async def test_a_focused_row_is_marked_on_the_ground_distinctly_from_hover() -> None:
    """Two pointers need two marks: the mouse is where the hand is, focus is
    where the keyboard is, and only one of them survives the hand leaving."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 12)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "read", {"path": "one.py"})
        other = ToolCard("b", "read", {"path": "two.py"})
        view.append_block(card)
        view.append_block(other)
        card.mark_done("alpha\nbeta")
        other.mark_done("gamma")
        await pilot.pause()

        def ground(widget: ToolCard) -> tuple[int, int, int]:
            # `Styles.background.rgb` is a plain 3-tuple, not rich's ColorTriplet
            # (a NamedTuple of the same shape); compared by value against it below.
            return widget.styles.background.rgb

        def expected(token: str) -> ColorTriplet:
            return _triplet(Style(bgcolor=theme_mod.semantic_color(token)).bgcolor)

        assert ground(card) == expected("surface")
        card.focus()
        await pilot.pause()
        assert ground(card) == expected("tint-select")

        # The pointer visiting the OTHER row must not take the focus mark off
        # this one; they are different questions with different answers.
        await pilot.hover(other)
        await pilot.pause()
        assert ground(card) == expected("tint-select")
        assert ground(other) == expected("overlay")


@pytest.mark.asyncio
async def test_shift_tab_out_of_the_composer_lands_on_the_last_action() -> None:
    """The keyboard's way IN. Tab is spoken for inside the composer (it
    indents, TUI-013), so Shift+Tab is the door, and it opens onto the most
    recent action rather than the oldest."""
    app = _ComposerApp()
    async with app.run_test(size=(90, 16)) as pilot:
        view = app.query_one(TranscriptView)
        first = ToolCard("a", "read", {"path": "one.py"})
        last = ToolCard("b", "read", {"path": "two.py"})
        view.append_block(first)
        view.append_block(last)
        first.mark_done("alpha")
        last.mark_done("beta\ngamma")
        app.query_one(Editor).focus()
        await pilot.pause()

        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.focused is last

        await pilot.press("enter")
        await pilot.pause()
        assert last.expanded is True


@pytest.mark.asyncio
async def test_typing_on_a_focused_row_reaches_the_composer_intact() -> None:
    """Making rows focusable must not create a place where typing vanishes.

    The app has exactly one text input, so a printable key on a row is never
    ambiguous. Every character is checked, not just the focus move: dropping
    the first one to "wake" the composer is the bug this forecloses.
    """
    app = _ComposerApp()
    async with app.run_test(size=(90, 12)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "read", {"path": "one.py"})
        view.append_block(card)
        card.mark_done("alpha")
        card.focus()
        await pilot.pause()
        assert app.focused is card

        await pilot.press(*"hello")
        await pilot.pause()
        editor = app.query_one(Editor)
        assert editor.text == "hello"
        assert app.focused is editor


@pytest.mark.asyncio
async def test_the_rows_own_keys_win_over_the_passthrough() -> None:
    """The passthrough must not eat the affordance it sits beside.

    Space is both a printable character and this row's toggle, and Textual
    dispatches ``on_key`` BEFORE it resolves the focused widget's bindings —
    so the passthrough saw Space first and typed it into the composer. The
    row now excludes its own keys explicitly, and this pins that: the two
    features live in the same class and the bug is invisible in any harness
    that has no composer for the key to escape to.
    """
    app = _ComposerApp()
    async with app.run_test(size=(90, 12)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("a", "read", {"path": "one.py"})
        view.append_block(card)
        card.mark_done("alpha\nbeta")
        card.focus()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()
        assert card.expanded is True
        assert app.focused is card
        assert app.query_one(Editor).text == ""


# --- control-sequence sanitisation ------------------------------------------
#
# Tool output is arbitrary bytes from arbitrary programs. `ls --color=always`,
# `git diff --color` and pytest/npm/cargo under FORCE_COLOR all emit CSI
# sequences, and an erase-display from a build tool would clear the user's
# screen from inside our own frame.


def test_erase_display_never_reaches_the_frame() -> None:
    """The worst case: a bare \\x1b[2J\\x1b[H would wipe the terminal."""
    card = ToolCard("t", "bash", {"command": "npm run build"})
    card.mark_done("\x1b[2J\x1b[HCLEARED-SCREEN")
    card.toggle_expanded()
    rendered = card._build_content(80).plain
    assert "\x1b" not in rendered
    assert "CLEARED-SCREEN" in rendered  # the text survives, the control does not


def test_colour_codes_are_stripped_from_the_collapsed_row() -> None:
    """No click needed to be exposed: a failure summary lands on the row, and
    cell-aware truncation could cut a CSI in half and emit a corrupt one."""
    card = ToolCard("t", "bash", {"command": "npm run build"})
    card.mark_failed("\x1b[31merror\x1b[0m: build failed", "\x1b[31merror\x1b[0m: build failed")
    row = card._build_row(80)
    assert "\x1b" not in row.plain
    assert "error" in row.plain


def test_control_sequences_in_args_and_partials_are_stripped() -> None:
    """Args and streaming partials are raw text too, not just results."""
    card = ToolCard("t", "bash", {"command": "echo \x1b[1mbold\x1b[0m"})
    assert "\x1b" not in card._build_row(80).plain
    card.set_partial_detail("progress \x1b[32m50%\x1b[0m")
    card._expanded = True
    assert "\x1b" not in card._build_content(80).plain


def test_a_streaming_partial_never_destroys_the_row_s_identity() -> None:
    """Reported from the field: four settled bash rows reading

        >_bash    --- stdout --- (empty) --- stderr --- (empty)    ✓ 1.0s

    and not one of them able to say which command it had run.
    ``set_partial_detail`` was writing over ``_summary`` and nothing restored
    it, so the row's own progress permanently replaced the only record of what
    it did.

    The contract is now stronger than "restored afterwards": progress never
    touches the row at all. It goes to the EXPANSION, where the user asks for
    it, and the collapsed row names the command in every state — which is the
    same guarantee stated as an invariant rather than as a repair.
    """
    card = ToolCard("t", "bash", {"command": "pytest -q tests/unit/tui"})
    card.set_partial_detail("--- stdout ---\nrunning 40 tests")

    running = card._build_row(120).plain
    assert "pytest -q tests/unit/tui" in running
    assert "stdout" not in running

    # …and the same fragment IS reachable, one keystroke away.
    card._expanded = True
    assert "running 40 tests" in card._build_content(120).plain

    card.set_partial_detail("--- stderr ---\nboom")
    card.mark_done("")
    settled = card._build_row(120).plain
    assert "pytest -q tests/unit/tui" in settled
    assert "stderr" not in settled


# -- the LIVE card: a running row names its command, moves, and can be opened -
#
# Reported as three symptoms of one defect. A settled bash row read
#
#     >_bash    199 B · 1s                    ⟨expand⟩ ✓  0.3s
#
# where `199 B · 1s` is the COMPOSING summary — the argument byte count and the
# dictation clock, frozen at the moment dictation ended — sitting where the
# command belongs, beside a second and disagreeing duration. `_compose_facts`
# was never cleared when the row started running, so the label-shed ladder it
# gates stayed armed for the whole life of the card; on a compose row that
# ladder drops the word `composing…` and keeps the numbers, and on every other
# row it drops the COMMAND and keeps them.


def test_a_running_row_names_its_command_not_its_argument_bytes() -> None:
    """The reported frame, at the width that produced it.

    The ladder fires when ``width < _label_min_width()``, a threshold computed
    from the summary's own length — so it is not a bash defect but a
    LONG-SUMMARY defect, and bash is merely the tool whose summaries are long.
    Asserted across the lifecycle because the row was wrong in all of it.
    """
    command = 'cd ~/local-operator && git remote -v; echo "--- gitconfig"; cat .git/config'
    card = ToolCard("t", "bash")
    card.set_composing(199, "bash")
    assert "199 B" in card._build_row(80).plain  # composing: the bytes ARE the news

    card.begin_running("bash", {"command": command}, None)
    running = card._build_row(80).plain
    assert "199 B" not in running
    assert running.startswith(f"{tool_icon('bash')} bash")
    assert "cd ~/local-operator && git remote" in running

    card.mark_done("exit code: 0\nok")
    settled = card._build_row(80).plain
    assert "199 B" not in settled
    assert "cd ~/local-operator && git remote" in settled


def test_a_short_identity_summary_was_never_broken_and_must_stay_that_way() -> None:
    """The CONTROL case, asserted beside the broken one.

    A settled MCP row rendered correctly throughout::

        🛡 workspace_get_gmail_mes…  damian@gominerva.com full        ✓  0.5s

    — real identity arguments, correctly truncated, one duration on the right
    and no `· Ns` artefact in the headline. So the shared path through
    ``_summary_from_args`` was never the problem, and the fix must not have
    moved this row by a cell.

    The two differ in ONE property, which is the whole diagnosis: summary
    LENGTH. ``_label_min_width`` is computed from the summary, so a
    twenty-five-cell identity clears the threshold at every realistic width
    and a seventy-five-cell command clears none of them — which is why the
    ladder fired for shell-shaped calls and only for them. Asserted here so a
    future change to either row has to keep both true at once.
    """
    identity = ToolCard("t", "workspace_get_gmail_messages_content_batch")
    identity.set_composing(240, "workspace_get_gmail_messages_content_batch")
    identity.begin_running(
        "workspace_get_gmail_messages_content_batch",
        {"user_google_email": "damian@gominerva.com", "format": "full"},
        None,
    )
    identity.mark_done("ok\nmore")

    shell = ToolCard("t", "bash")
    shell.set_composing(199, "bash")
    shell.begin_running(
        "bash",
        {"command": 'cd ~/local-operator && git remote -v; echo "--- gitconfig"; cat .git/config'},
        None,
    )
    shell.mark_done("exit code: 0\nok")

    for width in (80, 100, 120):
        identity_row = identity._build_row(width).plain
        shell_row = shell._build_row(width).plain
        # The control: unchanged, and never carrying a compose artefact.
        assert "damian@gominerva.com full" in identity_row
        assert "240 B" not in identity_row
        assert width >= identity._label_min_width()  # never on the ladder
        # The row that was broken, now telling the same kind of truth.
        assert "cd ~/local-operator && git remote" in shell_row
        assert "199 B" not in shell_row
        # Both settle into exactly ONE duration, in the same column. ` · ` is
        # the compose facts' own separator (`199 B · 1s`), so its absence is
        # what says no second clock reached the headline — which was the
        # difference between the two rows in the reported frames.
        for row in (identity_row, shell_row):
            assert row.rstrip().endswith("s")
            assert " · " not in row
        assert identity_row.index(ICON_SUCCESS) == shell_row.index(ICON_SUCCESS)


def test_the_command_is_abbreviated_to_the_row_by_one_rule_in_every_state() -> None:
    """"Abbreviated within the available horizontal line space" — and the SAME
    abbreviation running as settled, so a call cannot read as one thing while
    it works and another once it is done. A row that composed first must be
    indistinguishable from one that did not.
    """
    command = "python -m pytest tests/unit/tui -q -p no:cacheprovider -k expansion"
    for width in (48, 60, 80, 120, 200):
        composed = ToolCard("t", "bash")
        composed.set_composing(203, "bash")
        composed.begin_running("bash", {"command": command}, None)
        direct = ToolCard("t", "bash", {"command": command})
        assert composed._build_row(width).plain == direct._build_row(width).plain
        # A PREFIX of the command survives the cut — never a different command
        # and never a byte count. Measured on the longest prefix the row holds
        # rather than by carving the status off the tail, because the pad
        # between summary and status shrinks to one cell at tight widths and
        # is then indistinguishable from a space inside the command.
        row = direct._build_row(width).plain
        assert "203 B" not in row
        kept = max((n for n in range(1, len(command) + 1) if command[:n] in row), default=0)
        assert kept >= 20, f"width {width} kept only {kept!r} cells of the command"
        assert cell_len(row) <= width


def test_a_running_rows_duration_advances_and_holds_the_settled_column() -> None:
    """A timer that never moves is indistinguishable from a hung command.

    The card reported ``0s`` against a working line reporting 34s because the
    only clock on the row was the composing one, stopped. The running row now
    counts its OWN execution — and lands the number in the column the ✓ will
    use, so settling does not make the row jump.
    """
    card = ToolCard("t", "bash", {"command": "sleep 30"})
    assert card._build_row(80).plain.rstrip().endswith("0s")

    # A card built here started its own clock; ``_started`` is only ``None`` for
    # a card adopted through ``restore()``, which refuses to time a row it did
    # not watch run. Winding the clock back is only meaningful for the former.
    assert card._started is not None, "a card that begins running times itself"
    card._started -= 34.0
    assert card._build_row(80).plain.rstrip().endswith("34s")

    # The duration occupies exactly the cells it will occupy once settled: the
    # two blanks the running row reserves are where the ✓ arrives, so the
    # number does not shift under the eye at the moment the row settles.
    running = card._build_row(80).plain
    card.mark_done("out")
    settled = card._build_row(80).plain
    assert running.index("34s") == settled.index("34s")
    # `34s` is right-justified into DURATION_COL, so the glyph run sits four
    # cells back from the digits, where the running row was painting blanks.
    assert settled[settled.index("34s") - 4 : settled.index("34s") - 2] == f"{ICON_SUCCESS} "
    assert running[running.index("34s") - 4 : running.index("34s") - 2] == "  "


def test_a_composing_row_reports_no_execution_time() -> None:
    """Nothing has RUN, so there is no execution to time. The dictation clock
    rides in the summary beside the byte count it belongs with; a duration in
    the outcome column would claim a tool was executing."""
    card = ToolCard("t", "bash")
    card.set_composing(199, "bash")
    assert card._status_runs() == []


def test_expanding_a_running_call_says_so_and_shows_the_command() -> None:
    """Expanded, a running call and a finished call that printed nothing are
    the same frame — a command and then nothing — and they mean opposite
    things. The header is what separates them, and it says which in words."""
    card = ToolCard("t", "bash", {"command": "sleep 30"})
    assert card.toggle_expanded() is True
    body = card._build_content(80).plain
    assert "command: sleep 30" in body
    assert LIVE_HEADER_RUNNING in body
    assert LIVE_HEADER_PENDING in body  # nothing yet, said as "not yet"

    card.set_partial_detail("--- stdout ---\nline 1")
    body = card._build_content(80).plain
    assert LIVE_HEADER_RUNNING in body
    assert LIVE_HEADER_PENDING not in body  # output arrived: caveat gone
    assert "line 1" in body


def test_the_in_progress_header_is_the_accent_the_running_icon_spends() -> None:
    """Asserted through the semantic ramp, not read off a render.

    The header is the card's answer to "is this alive", and it has to carry in
    a STILL, and possibly colourless, frame — which is why the words say
    "running" rather than leaving it to the tint. But the tint is a real claim
    and a greyscale capture cannot check it: two colours that differ render
    identically once the terminal drops colour, so the assertion belongs here
    where the ramp is addressable.
    """
    card = ToolCard("t", "bash", {"command": "sleep 30"})
    card.toggle_expanded()
    content = card._build_content(80)
    lit = _style_at(content, LIVE_HEADER_RUNNING)
    assert _triplet(lit.color) == _triplet(Style(color=theme_mod.semantic_color("accent")).color)


def test_an_expanded_running_card_can_still_be_closed_once_it_settles_empty()  -> None:
    """`can_expand` goes false under an OPEN card when a running call finishes
    having printed nothing — reachable only now that running cards open. A card
    stuck open is the same trap as one that will not open."""
    card = ToolCard("t", "bash", {"command": "true"})
    card.toggle_expanded()
    card.mark_done("")
    assert card.can_expand() is False
    assert card.toggle_expanded() is False
    assert card.expanded is False


def test_the_live_view_keeps_the_tail_and_bounds_the_card() -> None:
    """A `seq 100000` must not grow the widget without bound, and the live view
    keeps the END: a live view frozen on the first forty lines of a command
    still running is indistinguishable from a hung one, which is the anxiety
    the whole feature exists to relieve."""
    card = ToolCard("t", "bash", {"command": "seq 100000"})
    card._expanded = True
    card.set_partial_detail("\n".join(str(n) for n in range(1, 501)))

    assert len(card._live) == LIVE_MAX_LINES
    assert card._live[-1] == "500"  # the TAIL, not the head
    assert card._live_dropped == 500 - LIVE_MAX_LINES
    body = card._build_content(80).plain
    assert f"… {500 - LIVE_MAX_LINES} earlier lines" in body
    # Bounded height: command row + header + marker + the capped tail.
    assert len(body.splitlines()) <= LIVE_MAX_LINES + 8


def test_a_payload_past_the_ingest_slice_refuses_to_quote_a_line_count() -> None:
    """The slice caps per-update work at a constant — bash re-sends its WHOLE
    accumulated output every 500 ms, so an unsliced parse of a 22 MB snapshot
    measured 1026 ms against 2.95 ms sliced, which at a 2 Hz emit cannot keep
    up at all. What the card has not parsed it must not count: quoting the
    lines dropped from the SLICE reported `… 10903 earlier lines` on a payload
    missing 99981 of them, which reads as a measurement."""
    card = ToolCard("t", "bash", {"command": "seq 1000000"})
    card._expanded = True
    card.set_partial_detail("\n".join(str(n) for n in range(1, 1_000_001)))

    assert card._live_elided is True
    assert card._live[-1] == "1000000"  # still the true tail
    body = card._build_content(80).plain
    assert "… earlier output not shown" in body
    assert "earlier lines" not in body  # no invented number


def test_an_arriving_update_does_not_repaint_and_the_tick_does() -> None:
    """The coalescing, as a contract. A chatty command must cost one repaint
    per tick, not one per update — the same bargain the subagent panel strikes
    with its own single timer."""
    card = ToolCard("t", "bash", {"command": "yes"})
    painted: list[int] = []
    card._refresh_row = lambda: painted.append(1)  # type: ignore[method-assign]

    for n in range(50):
        card.set_partial_detail(f"line {n}")
    assert painted == []  # fifty updates, no repaints
    assert card._live_dirty is True

    card._tick_clock()
    assert painted == [1]  # one tick, one repaint
    assert card._live_dirty is False


def test_the_result_replaces_the_streamed_tail_rather_than_joining_it() -> None:
    """Two accounts of one output, and the streamed one is the truncated,
    out-of-date one. It goes when the real result lands — along with the clock,
    on every settle path including the failing one, which was the only path
    that never stopped it.

    The clock is SEEDED rather than started. These cards are never mounted, so
    ``_start_clock``'s message-pump guard means ``_clock_timer`` is ``None`` for
    their whole life and an ``is None`` assertion after the settle passes no
    matter what the settle does — deleting the ``_settle_live()`` call this
    defends left the suite green. A stub that records its own ``stop`` is the
    difference between asserting the timer was retired and asserting one never
    existed.
    """
    for settle in (
        lambda c: c.mark_done("exit code: 0\nreal output"),
        lambda c: c.mark_failed("boom", "exit code: 1\nreal output"),
        lambda c: c.mark_interrupted(),
    ):
        card = ToolCard("t", "bash", {"command": "sleep 5"})
        card._expanded = True
        card.set_partial_detail("--- stdout ---\npartial output")
        stopped: list[int] = []
        card._clock_timer = cast(Any, SimpleNamespace(stop=lambda: stopped.append(1)))
        assert card._live
        settle(card)
        assert stopped == [1]
        assert card._clock_timer is None
        assert card._live == []
        assert "partial output" not in card._build_content(80).plain


def test_width_accounting_is_correct_once_escapes_are_gone() -> None:
    """cell_len counts '[31m' as 4 visible cells while ESC is 0, so unstripped
    escapes made the fill and the right-aligned status column go ragged."""
    plain = ToolCard("t", "bash", {"command": "run tests"})
    plain.mark_done("ok")
    coloured = ToolCard("t", "bash", {"command": "run tests"})
    coloured.mark_done("\x1b[32mok\x1b[0m")
    for width in (20, 40, 80, 200):
        assert cell_len(coloured._build_row(width).plain) <= width
        assert cell_len(coloured._build_row(width).plain) == cell_len(plain._build_row(width).plain)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # 7-bit CSI
        ("\x1b[2J", ""),  # erase display — would clear the terminal
        ("\x1b[H", ""),  # cursor home
        ("\x1b[38;5;196mred\x1b[0m", "red"),  # 256-colour SGR
        ("\x1b[?25lhidden", "hidden"),  # private-mode CSI with intermediate
        # 8-bit C1 forms. Easy to miss because they do not look like escapes in
        # a decoded str, but \x9b IS a CSI to a terminal honouring C1.
        ("\x9b31mred\x9b0m", "red"),
        ("\x9d0;title\x9cafter", "after"),
        # String controls are removed WITH their payload: device data is not
        # display text, so leaving "tmux;xyz" behind turns a control into
        # wrong content.
        ("\x1b]0;window title\x07x", "x"),
        ("\x1b]8;;http://x\x1b\\link", "link"),
        ("\x1bPtmux;xyz\x1b\\after", "after"),
        ("\x1b_G a=T\x1b\\after", "after"),
        ("\x1b^private\x1b\\after", "after"),
        ("\x1bXsomething\x1b\\after", "after"),
        ("\x1b]0;unterminated", ""),
        # Truncation boundaries: a fragment must not be left for the terminal
        # to complete using the real content that follows.
        ("text\x1b[3", "text"),
        ("text\x9b38;5", "text"),
        ("text\x1b", "text"),
        ("\x1bM", ""),  # two-char escape (reverse index)
        # C0 and other C1 controls
        ("a\x00b\x07c", "abc"),
        ("keep \x7f me", "keep  me"),
        ("a\x85b\x9ac", "abc"),
        # Printable text is preserved EXACTLY — none of it lives in the
        # control ranges, and over-stripping would corrupt real output.
        ("plain text", "plain text"),
        ("emoji 👨‍👩‍👧 and 中文", "emoji 👨‍👩‍👧 and 中文"),
        ("┌─┐│└┘├", "┌─┐│└┘├"),
        ("mixed مرحبا שלום rtl", "mixed مرحبا שלום rtl"),
        ("«»—… ∑∫≈", "«»—… ∑∫≈"),
    ],
)
def test_strip_control_sequences_cases(raw: str, expected: str) -> None:
    from local_operator.tui.widgets.tool_card import _strip_control_sequences

    assert _strip_control_sequences(raw) == expected


def test_no_control_codepoint_ever_survives() -> None:
    """Property check over the whole control space, so a form nobody thought to
    enumerate cannot slip through."""
    from local_operator.tui.widgets.tool_card import _strip_control_sequences

    for code in list(range(0x00, 0x20)) + [0x7F] + list(range(0x80, 0xA0)):
        if code in (0x09, 0x0A):  # tab/newline are handled before this runs
            continue
        out = _strip_control_sequences(f"a{chr(code)}b")
        assert chr(code) not in out, f"U+{code:04X} survived: {out!r}"
