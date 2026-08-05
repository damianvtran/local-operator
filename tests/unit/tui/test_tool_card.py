"""Tool card behaviour — the one-line guarantee, diff counters, expansion.

Three contracts are defended here, each of which is a visible promise:

- a COLLAPSED card is exactly one row, at every width, in every state, with
  or without the extra segments the richer states add
- diff counters appear only when the tool actually reported them, tinted
  success/danger, and never as a misleading ``+0 -0``
- clicking a card reveals its full output and clicking again puts it back to
  one row

Colours are asserted through ``theme.semantic_color`` rather than literal
hexes so a ramp change moves one file, not this suite.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import tool_card as card_mod
from local_operator.tui.widgets.tool_card import (
    COLLAPSE_HINT,
    EXPAND_HINT,
    EXPAND_MAX_LINES,
    ToolCard,
    compact_path,
)
from local_operator.tui.widgets.transcript import TranscriptView
from tests.unit.tui.conftest import StyledTranscriptApp

#: Every width the one-line guarantee is checked at: pathological narrow,
#: narrow, typical split-pane, standard, and ultrawide.
WIDTHS = (16, 20, 40, 80, 200)


def _style_at(text: Text, needle: str) -> Style:
    """The style covering the first cell of ``needle`` in a built row."""
    index = text.plain.index(needle)
    for span in text.spans:
        if span.start <= index < span.end:
            return span.style
    return Style()


def _assert_fits(card: ToolCard) -> None:
    """The collapsed row fits — and stays a row — at every width."""
    for width in WIDTHS:
        row = card._build_row(width)
        assert "\n" not in row.plain
        assert cell_len(row.plain) <= width, (width, row.plain)


# --- diff counters ---------------------------------------------------------


def test_write_diff_counts_render_in_success_and_danger_tints() -> None:
    card = ToolCard("t", "write", {"path": "notes.md", "content": "x"})
    card.mark_done("Created notes.md (3 chars).", {"path": "notes.md", "added": 12, "removed": 3})
    row = card._build_row(80)

    assert "+12" in row.plain and "-3" in row.plain
    assert (
        _style_at(row, "+12").color.triplet
        == Style(color=theme_mod.semantic_color("success")).color.triplet
    )
    assert (
        _style_at(row, "-3").color.triplet
        == Style(color=theme_mod.semantic_color("danger")).color.triplet
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
    # One summary row plus one row per output line — no reflow, no wrapping.
    assert card._row_count == 4
    assert card.spans_multiple_rows() is True
    assert card.settled_rows() == 4
    body = card._build_content(80).plain.splitlines()
    assert body[1].strip() == "total 8"
    assert body[3].strip() == "-rw-r--r--  b"

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


def test_click_on_an_inert_card_does_nothing_and_stays_clickable_elsewhere() -> None:
    """No output means no expansion — and the click is NOT swallowed."""

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
    assert card.toggle_expanded() is False


def test_hint_appears_only_when_hovered_and_expandable() -> None:
    """Two conditions, both required: something to reveal AND the pointer on
    the row. At rest the ▸ chevron is the whole affordance — printing the hint
    on every settled row is ~9 cells of permanent chrome on an 80-column
    terminal restating what the chevron already says."""
    inert = ToolCard("t", "bash", {"command": "ls"})
    inert.mark_done("")
    inert._set_hovered(True)
    assert EXPAND_HINT not in inert._build_row(80).plain  # nothing to expand

    running = ToolCard("t", "bash", {"command": "ls"})
    running._set_hovered(True)
    assert EXPAND_HINT not in running._build_row(80).plain  # not finished

    expandable = ToolCard("t", "bash", {"command": "ls"})
    expandable.mark_done("one\ntwo")
    assert EXPAND_HINT not in expandable._build_row(80).plain  # at rest: silent

    expandable._set_hovered(True)
    assert EXPAND_HINT in expandable._build_row(80).plain  # hovered: offered

    expandable.toggle_expanded()
    row = expandable._build_row(80).plain
    assert COLLAPSE_HINT in row and EXPAND_HINT not in row


def test_hovered_hint_uses_the_dim_ramp_step() -> None:
    """When it does show, it sits at `dim` — below the summary, above the
    separators, so it reads as an offer rather than as content."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo")
    card._set_hovered(True)
    lit = _style_at(card._build_row(80), EXPAND_HINT)
    assert lit.color.triplet == Style(color=theme_mod.semantic_color("dim")).color.triplet


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
    assert len(rows) == 1 + EXPAND_MAX_LINES + 1  # summary + window + marker
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
    assert (
        _style_at(content, "Traceback:").color.triplet
        == Style(color=theme_mod.semantic_color("danger")).color.triplet
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


def test_explicit_intent_wins_over_argument_derived_summaries() -> None:
    card = ToolCard("t", "write", {"path": "notes.md"}, intent="Recording the decision")
    assert "Recording the decision" in card._build_row(80).plain


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

        surface = Style(bgcolor=theme_mod.semantic_color("surface")).bgcolor
        overlay = Style(bgcolor=theme_mod.semantic_color("overlay")).bgcolor
        dim = Style(color=theme_mod.semantic_color("dim")).color

        def ground(card: ToolCard):
            return card.styles.background.rgb

        def has_hint(card: ToolCard) -> bool:
            return EXPAND_HINT in card.renderable.plain

        def hint_color(card: ToolCard):
            return _style_at(card.renderable, EXPAND_HINT).color.triplet

        assert ground(first) == surface.triplet
        assert not has_hint(first)

        await pilot.hover(first)
        await pilot.pause()
        assert ground(first) == overlay.triplet
        assert has_hint(first) and hint_color(first) == dim.triplet

        await pilot.hover(second)
        await pilot.pause()
        assert ground(first) == surface.triplet
        assert not has_hint(first)  # the row the pointer left goes quiet again
        assert ground(second) == overlay.triplet
        assert has_hint(second) and hint_color(second) == dim.triplet


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

        def expected(token: str):
            return Style(bgcolor=theme_mod.semantic_color(token)).bgcolor.triplet

        assert running.styles.background.rgb == expected("raised")
        assert ok.styles.background.rgb == expected("surface")
        assert bad.styles.background.rgb == expected("tint-danger")


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
    assert "\x1b" not in card._build_row(80).plain


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
