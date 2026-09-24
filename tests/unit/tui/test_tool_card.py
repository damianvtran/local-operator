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

import time
import unicodedata
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.color import Color, ColorTriplet
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult

from local_operator.tui import bindings
from local_operator.tui import glyphs as glyph_mod
from local_operator.tui import theme as theme_mod
from local_operator.tui.app import _first_line
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
    DURATION_COL,
    EXPAND_HINT,
    EXPAND_MAX_LINES,
    ICON_ERROR,
    ICON_INTERRUPTED,
    ICON_PARTIAL,
    ICON_SUCCESS,
    LIVE_ADVISORY_GLYPH,
    LIVE_HEADER_PENDING,
    LIVE_HEADER_RUNNING,
    LIVE_MAX_LINES,
    NO_OUTPUT_NOTICE,
    OUTPUT_INDENT,
    QUEUED_NOTICE,
    REASON_MAX_CELLS,
    REASON_MAX_ROWS,
    ROW_INDENT,
    ROW_INDENT_MIN_WIDTH,
    RUNNING_NOTICE,
    START_UNKNOWN,
    TERSE_NO_OUTPUT_NOTICE,
    TERSE_QUEUED_NOTICE,
    ToolCard,
    _category_element,
    compact_path,
)
from local_operator.tui.widgets.transcript import (
    NoticeBlock,
    TranscriptView,
    wrap_cells,
)
from local_operator.web_fetch.failure import FetchFailure, describe, retry_after_note
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
    """A settled edit card carrying a rendered unified diff in details.

    The args carry ``old_text``/``new_text`` deliberately: the expansion
    contract is that these do NOT paint — the diff is the whole body.
    """
    card = ToolCard(
        "t", "edit", {"path": "notes.md", "old_text": "old line", "new_text": "new line"}
    )
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

    # Build the expected triplet from the semantic hex (``#rrggbb``).
    def _expect(semantic: str) -> ColorTriplet:
        hexv = theme_mod.semantic_color(semantic)
        return ColorTriplet(*[int(hexv[i : i + 2], 16) for i in (1, 3, 5)])

    # The added line carries the success green; the removed the danger red —
    # the same two tints the summary pill uses for +N/-N.
    assert _triplet(plus_style.color) == _expect("success")
    assert _triplet(minus_style.color) == _expect("danger")
    # Context rides dim; hunk markers ride muted — neither success nor danger.
    assert ctx_style != plus_style and ctx_style != minus_style
    assert hunk_style != plus_style and hunk_style != minus_style


def test_a_settled_edit_expansion_is_the_diff_alone() -> None:
    """No argument echo, no ``---/+++`` chrome — just the coloured diff.

    Reported from the field: an expanded edit card led with ``path:`` and a
    flattened ``edits: [{"old_text": "…\\n…"}]`` wall — the same change the
    diff below it already showed readably — so the record was buried under
    its own escape sequences. The settled diff branch now paints ONLY the
    diff, and drops difflib's nameless file headers (the path is on the
    summary row).
    """
    card = _diff_card()
    card.toggle_expanded()
    content = card._build_content(80).plain
    # The diff itself is intact…
    assert "+new line" in content
    assert "-old line" in content
    assert "@@ -1,4 +1,4 @@" in content
    # …with no argument block above it…
    assert "old_text:" not in content
    assert "new_text:" not in content
    assert "path:" not in content
    # …and no blank-label file header ROWS. Asserted per-line rather than by
    # substring: a body line may legitimately contain or even start with
    # `---`/`+++` (see the regression test below), so only the leading pair
    # difflib emits is what must be gone.
    lines = content.splitlines()
    assert not any(line.strip() in ("---", "+++") for line in lines)


def test_removed_lines_that_start_with_dashes_survive_the_header_strip() -> None:
    """The header strip is positional, never a pattern over the diff body.

    Regression for review round 1 F1/D1: a removed content line that itself
    begins `--` (a SQL/Lua/Haskell comment) renders inside the body as
    `--- old comment`, and a removed bare `--` renders as exactly `---`. A
    pattern filter silently deleted both — the summary pill said `-2` while
    the expansion showed no red line at all, and with the argument echo gone
    the diff is the SOLE record of the removal.
    """
    card = ToolCard("t", "edit", {"path": "query.sql"})
    card.mark_done(
        "Edited query.sql: replaced 1 occurrence(s) of old_text.",
        {
            "path": "query.sql",
            "added": 0,
            "removed": 2,
            "diff": [
                "--- ",
                "+++ ",
                "@@ -1,4 +1,2 @@",
                " SELECT 1;",
                "--- old comment",
                "---",
                " SELECT 2;",
            ],
        },
    )
    card.toggle_expanded()
    content = card._build_content(80).plain
    # Both removed lines are painted; only the two leading headers are gone.
    assert "--- old comment" in content
    body_lines = [line.strip() for line in content.splitlines()]
    assert "---" in body_lines  # the removed bare `--` line survives
    # The nameless header pair itself (rows before the @@ hunk) is absent:
    # the first diff row painted is the hunk marker.
    at = next(i for i, line in enumerate(body_lines) if line.startswith("@@"))
    assert not any(line in ("---", "+++") for line in body_lines[:at])


def test_a_running_edit_still_shows_its_arguments() -> None:
    """Diff-only is a SETTLED-state rule; live cards keep the argument block.

    While the call runs there is no diff yet, so the arguments are the only
    account of what the tool was asked to do — exactly the frame a user
    opens to answer "what is it editing?".
    """
    card = ToolCard("t", "edit", {"path": "notes.md", "old_text": "a", "new_text": "b"})
    assert card.can_expand() is True
    card.toggle_expanded()
    content = card._build_content(80).plain
    assert "path: notes.md" in content
    assert "old_text:" in content


def test_a_failed_edit_keeps_its_argument_block() -> None:
    """A failed edit's args are the only record of intent; diff-only must not apply.

    The error ("old_text not found") only makes sense next to the old_text
    that was searched for. The diff-only branch therefore gates on the
    SUCCESS state, not on diff presence (review round 1, F2): even if a
    failing tool ever shipped a diff in its details, the failed card must
    still paint its arguments and error, not a diff with no account of what
    was attempted.
    """
    card = ToolCard("t", "edit", {"path": "notes.md", "old_text": "missing", "new_text": "x"})
    card.mark_failed(
        "old_text not found",
        "old_text not found (exact and whitespace-tolerant)",
        # A diff in a FAILED result is unreachable from today's tools, but the
        # branch must be gated by state, not by this payload's presence.
        {"path": "notes.md", "added": 0, "removed": 0, "diff": ["@@ -1 +1 @@", "-a", "+b"]},
    )
    card.toggle_expanded()
    content = card._build_content(80).plain
    assert "old_text: missing" in content
    assert "old_text not found" in content


def test_a_tool_without_a_diff_expands_to_its_raw_output() -> None:
    """Non-write tools keep the plain-output expansion (no diff in details)."""
    card = ToolCard("t", "bash", {"command": "ls"})
    card.mark_done("one\ntwo\nthree")
    assert card.can_expand() is True
    assert card._diff is None
    card.toggle_expanded()
    content = card._build_content(80).plain
    assert "one" in content and "three" in content


def test_web_search_expansion_uses_structured_page_metadata() -> None:
    """The model's bounded text must not impoverish the human expansion."""
    card = ToolCard("t", "web_search", {"query": "python release"})
    card.mark_done(
        "Provider: duckduckgo (credential-free)",
        {
            "provider": "duckduckgo",
            "auth_mode": "credential-free",
            "sources": [
                {
                    "title": "Python 3.13 release",
                    "url": "https://python.org/downloads/release/python-3130/",
                    "snippet": "Release notes and downloads for Python 3.13.",
                }
            ],
        },
    )

    # Search sources are primary content, so their disclosure stays visible
    # without requiring hover or prior knowledge of generic card controls.
    assert EXPAND_HINT in card._build_row(120).plain
    card.toggle_expanded()
    content = card._build_content(120)

    assert "Python 3.13 release" in content.plain
    assert "https://python.org/downloads/release/python-3130/" in content.plain
    assert "Release notes and downloads for Python 3.13." in content.plain
    assert "Ask Operator to web_fetch result N" in content.plain
    assert _style_at(content, "Python 3.13 release").bold is True
    assert (
        _style_at(content, "https://python.org").color != _style_at(content, "Release notes").color
    )


def test_web_fetch_card_shows_fetch_metadata_and_preview() -> None:
    """A web_fetch card renders the structured header rows plus the preview,
    and keeps its disclosure visible at rest like a search card."""
    card = ToolCard("t", "web_fetch", {"url": "https://example.com/docs"})
    card.mark_done(
        "[200] https://example.com/docs\n\n# Docs\n\nBody content here.",
        {
            "url": "https://example.com/docs",
            "final_url": "https://example.com/docs/",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 1234,
            "lines": 40,
        },
    )
    # Disclosure visible without hover: a fetch is a primary result.
    assert EXPAND_HINT in card._build_row(120).plain
    card.toggle_expanded()
    content = card._build_content(120).plain
    assert "Fetched: https://example.com/docs" in content
    assert "final: https://example.com/docs/" in content
    assert "markdownify" in content
    assert "# Docs" in content  # the preview body is shown too
    # D1: the model-facing header block must NOT be duplicated in the card body.
    # The structured rows own the metadata; the "[200] url" lead line and its
    # "method · ctype · cache" line are stripped from the preview.
    assert "[200] https://example.com/docs" not in content
    # And the status/ctype/cache fields appear once (in the structured rows).
    assert content.count("cache miss") == 1


def test_web_fetch_card_humanizes_bytes_and_strips_duplicate_header() -> None:
    """D1 + D2: a large fetch shows humanised bytes in the structured row and does
    not repeat the model-facing header block in the body."""
    body = "[200] https://ex.example\nmarkdownify · text/html · cache miss\n\n" + (
        "\n".join(f"line {i}" for i in range(60))
    )
    card = ToolCard("t", "web_fetch", {"url": "https://ex.example"})
    card.mark_done(
        body,
        {
            "url": "https://ex.example",
            "final_url": "https://ex.example",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 2517000,
            "lines": 60,
        },
    )
    card.toggle_expanded()
    content = card._build_content(120).plain
    # D2: raw 7-digit bytes are humanised.
    assert "2.4 MB" in content
    assert "2517000 B" not in content
    # D1: no duplicated header line in the body.
    assert "[200] https://ex.example" not in content
    assert content.count("cache miss") == 1


def test_read_url_card_uses_fetch_presentation() -> None:
    """A ``read <url>`` records tool_name 'read' but carries fetch details, so it
    must select the fetch card, not the plain file-read presentation."""
    card = ToolCard("t", "read", {"path": "https://example.com"})
    card.mark_done(
        "[200] https://example.com\n\nExample body.",
        {
            "url": "https://example.com",
            "final_url": "https://example.com",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "hit",
            "bytes": 500,
            "lines": 7,
        },
    )
    card.toggle_expanded()
    content = card._build_content(120).plain
    assert "Fetched: https://example.com" in content


def test_read_file_card_not_treated_as_fetch() -> None:
    """A plain file read (no render_method/final_url) keeps the ordinary output
    presentation — the fetch branch must not capture it."""
    card = ToolCard("t", "read", {"path": "/tmp/note.txt"})
    card.mark_done("file line one\nfile line two", {"path": "/tmp/note.txt"})
    card.toggle_expanded()
    content = card._build_content(120).plain
    assert "Fetched:" not in content
    assert "file line one" in content


def test_web_fetch_non_2xx_renders_error_treatment() -> None:
    """F1: a non-2xx fetch card leads with a prominent ⚠ error row (danger ink)
    and does not present the block-page body as normal content. The duplicated
    model-facing header (lead + meta + note) is stripped, leaving one error row."""
    card = ToolCard("t", "web_fetch", {"url": "https://walled.example/x"})
    card.mark_done(
        "⚠ HTTP 403 Forbidden — this is an error/block page, not page content. "
        "https://walled.example/x\n"
        "markdownify · text/html · cache miss\n"
        "(The body below is the error response, not the requested page.)\n\n"
        "Please enable JS and disable any ad blocker.",
        {
            "url": "https://walled.example/x",
            "final_url": "https://walled.example/x",
            "status": 403,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 200,
            "lines": 1,
            "ok": False,
            "http_error": True,
        },
    )
    card.toggle_expanded()
    content = card._build_content(120).plain
    # The prominent error row is present, exactly once.
    assert content.count("⚠ HTTP 403 Forbidden") == 1
    assert "error/block page, not page content" in content
    # The duplicated model-facing header block is stripped from the body.
    assert "(The body below is the error response" not in content
    # The block-page body still shows, but under the error row, not as the lead.
    assert "enable JS" in content
    # The error row is painted in the danger colour (strongest treatment).
    danger = theme_mod.semantic_color("danger")
    assert _style_at(card._build_content(120), "⚠ HTTP 403").color == Color.parse(danger)


def test_a_blocked_card_names_the_vendor_and_the_attempt_count() -> None:
    """Design review round 1, D1 + D2 + D4 + DN2, on the operator's own case.

    Before this round the card printed the vendor-less generic copy in its danger
    row while the row above it said "blocked by Akamai" — two wordings of one
    fact, with the *name* only on the line the card throws away — and the
    successfully-escalated fetch was character-for-character a one-attempt
    success. It also reported the discarded challenge page's byte count on a row
    whose line count described the replacement.
    """
    url = "https://www.shoppersdrugmart.ca/?lang=en&query=power+bar"
    preview = (
        f"⚠ HTTP 403 Forbidden — blocked by Akamai bot protection, not page content. {url}\n"
        "text · text/html · cache miss · 2 attempts (default, browser-profile)\n\n"
        "The origin's bot protection refused this request. A browser-shaped retry was "
        "also refused, so no headless fetch of this URL will succeed.\n"
        "Origin reference: 18.4b182117.1789250183.345ea6ea\n"
        "Next step: use the `browser` tool on this URL.\n"
        f"{url}"
    )
    card = ToolCard("t", "web_fetch", {"url": url})
    card.mark_done(
        preview,
        {
            "url": url,
            "final_url": url,
            "status": 403,
            "content_type": "text/html",
            "render_method": "text",
            "cache": "miss",
            "bytes": 382,
            "lines": 4,
            "ok": False,
            "http_error": True,
            "attempts": 2,
            "profiles": ["default", "browser"],
            "failure_kind": "blocked",
            "block_vendor": "akamai",
            "suggested_tool": "browser",
        },
    )
    card.toggle_expanded()
    content = card._build_content(120).plain

    # D2: ONE danger row, and it is the one that names the vendor.
    assert content.count("⚠ HTTP 403 Forbidden") == 1
    assert "blocked by Akamai bot protection" in content
    assert "error/block page, not page content" not in content
    # D1: the count and the identity change §3.6 promises are on the card.
    assert "2 attempts (default, browser-profile)" in content
    # DN2: no byte count for a replaced body — it describes the discarded page.
    assert "382 B" not in content
    # D7: the summary leads the counts, so the clip zone can only take a count.
    assert "text · 2 attempts (default, browser-profile) · 4 lines" in content

    # D4: the actionable sentence is the anchor, the opaque reference recedes,
    # and the prose itself is NOT dim (3.32:1 on the light error-tinted panel).
    built = card._build_content(120)
    assert _triplet(_style_at(built, "Next step:").color) == _triplet(
        Style(color=theme_mod.semantic_color("signal")).color
    )
    assert _triplet(_style_at(built, "Origin reference:").color) == _triplet(
        Style(color=theme_mod.semantic_color("dim")).color
    )
    assert _triplet(_style_at(built, "The origin's bot protection").color) == _triplet(
        Style(color=theme_mod.semantic_color("muted")).color
    )


def test_a_terminal_failure_card_is_the_same_visual_family() -> None:
    """Design review round 1, D5 + D3: a stall is a FETCH card, not a bash row.

    A terminal failure carries no ``render_method``/``final_url``, so this card
    used to fall through to the generic output body: no ⚠ lead, transcript ink,
    and a ``Fetched:`` row with no status — two failure shapes of one tool reading
    as two different tools. ``failure_kind`` is the key both carry.

    D3 rides along here: the statement is fitted to the card's own width at paint
    time, so it is one row where there is room and never clipped mid-word where
    there is not.
    """
    url = "https://stalls.example/x"
    explanation = (
        "A browser-shaped retry was tried as well, since a stalled request is "
        "frequently a silent block."
    )
    text = (
        "Read timed out after 20.0s — the origin accepted the connection but never "
        "sent a response (2 attempts: default, browser-profile)\n"
        f"{explanation}\n"
        "Next step: use the `browser` tool on this URL. It drives the real browser "
        "(a write-tier, approval-gated action), which is the only path that clears "
        "an interactive challenge.\n"
        f"{url}"
    )
    card = ToolCard("t", "web_fetch", {"url": url})
    card.restore(
        state="error",
        result_text=text,
        error=text.splitlines()[0],
        details={
            "url": url,
            "cache": "miss",
            "failure_kind": "stall",
            "attempts": 2,
            "profiles": ["default", "browser"],
            "suggested_tool": "browser",
        },
    )
    card.toggle_expanded()
    content = card._build_content(100).plain

    # The fetch family: a structured Fetched row and a ⚠ lead…
    assert "Fetched: " + url in content
    assert "⚠ Read timed out after 20.0s" in content
    # …and the lead is not repeated in the body (it was promoted, like the
    # response family's lead is stripped).
    assert content.count("Read timed out after 20.0s") == 1

    # D3 at 80 columns: every row of the sentence is painted in full — no
    # mid-word ellipsis, which is what the 76-cell hard wrap produced.
    narrow = card._build_content(80).plain.splitlines()
    explained = [row for row in narrow if "bot " in row or "silent block" in row]
    assert explained
    assert all("…" not in row for row in explained)
    # D3 at 150: the sentence fits ONE row, because the wrap follows the card's
    # width instead of stopping at a constant 76 cells.
    wide = card._build_content(150).plain.splitlines()
    assert any(explanation in row for row in wide)


def test_the_reflow_permission_belongs_to_our_prose_only() -> None:
    """Design review round 2, D6 — the cause, asserted head-on.

    The permission was keyed on ``isinstance(failure_kind, str)``, which is true
    for EVERY classified non-2xx, so a 404/500's ORIGIN body — the bytes
    ``service.py`` keeps verbatim on purpose — began being re-wrapped where
    round 1 had proved the frame byte-identical. A body may only be reflowed when
    it is OUR prose: the replaced challenge statement, or a failure that got no
    response at all. The predicate is exactly the promotion test beside it.
    """

    def permission(details: dict[str, Any]) -> bool:
        card = ToolCard("t", "web_fetch", {"url": "https://walled.example/x"})
        card.mark_done("body", details)
        return card._fetch_statement

    def shape(**overrides: Any) -> dict[str, Any]:
        # ``_fetch_result_output`` needs a URL to build any header row at all, so
        # every fixture carries one — the flag is only ever set on a card whose
        # fetch branch ran.
        base: dict[str, Any] = {
            "url": "https://walled.example/x",
            "final_url": "https://walled.example/x",
            "cache": "miss",
        }
        base.update(overrides)
        return base

    # The origin's own response, whatever its class: never reflowed.
    for kind, status in (("client", 404), ("server", 500), ("ratelimit", 429)):
        assert (
            permission(
                shape(
                    status=status,
                    failure_kind=kind,
                    render_method="markdownify",
                    http_error=True,
                )
            )
            is False
        ), kind
    # Our prose: the replaced challenge body and a no-response diagnosis.
    assert permission(shape(failure_kind="blocked", render_method="text")) is True
    assert permission(shape(failure_kind="stall")) is True


def test_a_404_body_is_painted_exactly_as_it_arrived() -> None:
    """D6, on the pixels: the wrap permission cannot touch the 404 body.

    Asserted as a DIFF against the same card with the permission forced off, so
    this test fails on any future change that lets the flag reach an origin body
    — not just on the specific re-wrap the designer caught.
    """
    preview = (
        "⚠ HTTP 404 Not Found — this is an error/block page, not page content.\n"
        "markdownify · text/html · cache miss\n\n"
        "| /guide/old-page | /guide/new-page | the 3.0 tree; the redirect table keeps 90 days |\n"
        "    GET /guide/old-page -> 301 https://docs.example.com/guide/new-page\n"
    )
    details = {
        "url": "https://docs.example.com/x",
        "final_url": "https://docs.example.com/x",
        "status": 404,
        "content_type": "text/html",
        "render_method": "markdownify",
        "cache": "miss",
        "bytes": 512,
        "lines": 2,
        "ok": False,
        "http_error": True,
        "failure_kind": "client",
    }
    card = ToolCard("t", "web_fetch", {"url": details["url"]})
    card.mark_done(preview, details)
    card.toggle_expanded()
    shipped = card._build_content(60).plain
    rows = shipped.splitlines()

    # One row per origin line, clipped like every other fixed-width row: the
    # route line keeps its indent and the table stays ONE row (the re-wrap split
    # it into three, one of them a bare ``|``).
    assert sum(1 for row in rows if row.strip().startswith("|")) == 1
    route = [row for row in rows if row.strip().startswith("GET /guide/old-page")]
    assert len(route) == 1
    # ONE row: the URL the re-wrap pushed onto a second row is still on the
    # route's own row, and the indent is intact in front of it.
    assert "-> 301" in route[0] and "https://docs" in route[0]
    assert route[0].startswith("      GET /guide/old-page")

    # And the flag is what gates it: forcing it ON is the D6 defect, so the
    # shipped frame is provably the un-reflowed one rather than a coincidence of
    # the fixture.
    card._fetch_statement = True
    assert card._build_content(60).plain != shipped
    card._fetch_statement = False
    assert card._build_content(60).plain == shipped


def test_the_escalation_survives_the_narrow_rendered_row() -> None:
    """Design review round 2, D7 — the live medium.com numbers at 80 columns.

    The summary was appended last, so a large escalated page clipped it first:
    ``Rendered: markdownify · 800 lines · 51.8 KB · 2 attempts (default,
    brow…``. The identity change is the fact D1 exists for, so it leads the row
    after the method and only a count can be dropped.
    """
    card = ToolCard("t", "web_fetch", {"url": "https://medium.com/"})
    card.mark_done(
        "[200] https://medium.com/\nmarkdownify · text/html · cache miss · "
        "2 attempts (default, browser-profile)\n\npage",
        {
            "url": "https://medium.com/",
            "final_url": "https://medium.com/",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 53067,
            "lines": 800,
            "ok": True,
            "attempts": 2,
            "profiles": ["default", "browser"],
        },
    )
    card.toggle_expanded()
    row = next(
        line.strip()
        for line in card._build_content(80).plain.splitlines()
        if line.strip().startswith("Rendered:")
    )
    assert "2 attempts (default, browser-profile)" in row
    # Order: the summary is not the trailing field, so the clip can only eat a
    # count. At the standard width the byte count is the field that goes.
    assert row.index("2 attempts") < len(row)
    assert "Rendered: markdownify · 2 attempts (default, browser-profile) · 800 lines" in row


def test_a_no_response_card_paints_no_rendered_row_and_d7_ordering_holds() -> None:
    """Design review round 3, D12, and the D7 fact it must not break.

    Moving the attempt summary so it could LEAD a narrowed success row (D7) also
    let it stand alone on a card with no field of its own: a terminal no-response
    failure has no ``render_method``/``lines``/``bytes``, so the row claimed
    ``Rendered: 2 attempts (default, browser-profile)`` on a fetch that rendered
    nothing AND repeated the ``(2 attempts: …)`` clause the ``⚠`` lead already
    carried one row up. Both cards are asserted here because they are one
    decision — where the summary may ride — and a guard that fixed one by
    breaking the other has to fail this test.
    """
    url = "https://stalls.example/x"
    # The preview is composed by the engine's OWN ``describe`` (the shipped path
    # for a terminal failure), not re-typed here: the promoted ⚠ lead is what
    # carries the attempt clause, and a hand-written body without it would test a
    # card the app never paints.
    stall = ToolCard("t", "web_fetch", {"url": url})
    stall.restore(
        state="error",
        result_text=describe(
            FetchFailure(
                kind="stall",
                retryable=False,
                detail=(
                    "read timed out after 9.6s — the origin accepted the connection "
                    "but never sent a response"
                ),
            ),
            attempts=2,
            profiles=("default", "browser"),
            url=url,
        ),
        error="Read timed out after 9.6s",
        details={
            "url": url,
            "cache": "miss",
            "failure_kind": "stall",
            "attempts": 2,
            "profiles": ["default", "browser"],
            "suggested_tool": "browser",
        },
    )
    stall.toggle_expanded()
    stalled = stall._build_content(100).plain
    # The false row is gone…
    assert "Rendered:" not in stalled
    # …and the escalation is still stated, exactly once, by the promoted lead.
    # Normalised first: the lead reflows at 100 columns, so the clause straddles
    # a row boundary as ``(2\n  attempts: …)``.
    flat = " ".join(stalled.split())
    assert flat.count("(2 attempts: default, browser-profile)") == 1

    # D7, re-pinned on a RESPONSE-bearing card: the summary leads the row's
    # counts, so a narrow clip can only ever take a count.
    ok = ToolCard("t", "web_fetch", {"url": "https://medium.com/"})
    ok.mark_done(
        "[200] https://medium.com/\nmarkdownify · text/html · cache miss\n\npage",
        {
            "url": "https://medium.com/",
            "final_url": "https://medium.com/",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 53067,
            "lines": 800,
            "ok": True,
            "attempts": 2,
            "profiles": ["default", "browser"],
        },
    )
    ok.toggle_expanded()
    rendered = next(
        row.strip()
        for row in ok._build_content(80).plain.splitlines()
        if row.strip().startswith("Rendered:")
    )
    assert rendered.startswith("Rendered: markdownify · 2 attempts (default, browser-profile)")


def test_the_retry_after_note_reflows_while_origin_bytes_still_clip() -> None:
    """Design review round 3, D13, and the D6 boundary it must not move.

    On a 503 carrying ``Retry-After`` the §3.4 sentence is OUR prose, but it
    arrives in the BODY beside the origin's own bytes, where the painter may not
    reflow anything (D6). Painted with the body's clip rule it read
    ``…the wait was not spen…`` at 80 columns and lost exactly the clause it
    exists for. It takes the wrap path per line, recognised by the lead
    ``failure.py`` exports, and keeps the body ink because the terminal shape of
    the same sentence rides the statement path in ``muted`` — the origin body on
    the next row still clips, which is the boundary.
    """
    url = "https://api.example.com/v1/orders"
    note = retry_after_note(600.0)
    origin_body = (
        "<html><body><h1>503 Service Unavailable</h1><p>The orders service is "
        "draining and cannot take new work until the pool refills.</p></body></html>"
    )
    card = ToolCard("t", "web_fetch", {"url": url})
    card.mark_failed(
        "⚠ HTTP 503 Service Unavailable — error/block page, not page content.",
        result_text=(
            "⚠ HTTP 503 Service Unavailable — error/block page, not page content. "
            f"{url}\ntext · text/html · cache miss · 1 attempts\n"
            "(The body below is the error response, not the requested page.)\n"
            f"{note}\n\n{origin_body}"
        ),
        details={
            "url": url,
            "final_url": url,
            "status": 503,
            "content_type": "text/html",
            "render_method": "text",
            "cache": "miss",
            "bytes": 612,
            "lines": 1,
            "ok": False,
            "http_error": True,
            "attempts": 1,
            "failure_kind": "server",
            "retry_after_s": 600.0,
        },
        measured_s=0.4,
    )
    card.toggle_expanded()

    narrow = card._build_content(80).plain.splitlines()
    start = next(
        index for index, row in enumerate(narrow) if row.strip().startswith("The origin asked")
    )
    # The sentence reflows instead of clipping: the lead opens it and the last
    # row carries the clause the note exists for.
    note_rows: list[str] = []
    for row in narrow[start:]:
        if not row.strip():
            break
        note_rows.append(row.strip())
    assert len(note_rows) >= 2
    joined = " ".join(note_rows)
    assert "the wait was not spent inside this call." in joined
    assert "…" not in joined

    # The origin body's own row is untouched by the permission: still clipped.
    body_rows = [row.strip() for row in narrow if row.strip().startswith("<html>")]
    assert body_rows
    assert body_rows[0].endswith("…")
    assert "draining and cannot take" not in body_rows[0]

    # Ink: the same sentence in its terminal shape rides ``muted``, so the
    # response-bearing shape must not disagree about it.
    built = card._build_content(80)
    assert _triplet(_style_at(built, "The origin asked us to wait").color) == _triplet(
        Style(color=theme_mod.semantic_color("muted")).color
    )

    # At 100 columns the sentence fits on its own row whole.
    wide = card._build_content(100).plain.splitlines()
    wide_note = [row.strip() for row in wide if row.strip().startswith("The origin asked")]
    assert len(wide_note) == 1
    assert wide_note[0].endswith("the wait was not spent inside this call.")

    # The permission is structural, not a bare prefix match: an ORIGIN body that
    # happens to begin with the same sentence, on a failure that reported NO
    # interval, is still clipped. Without the ``retry_after_s`` precondition a
    # line of origin bytes could be reflowed, which is the D6 defect.
    echo = ToolCard("t", "web_fetch", {"url": url})
    echo.mark_failed(
        "⚠ HTTP 503 Service Unavailable — error/block page, not page content.",
        result_text=(
            "⚠ HTTP 503 Service Unavailable — error/block page, not page content. "
            f"{url}\ntext · text/html · cache miss\n"
            "(The body below is the error response, not the requested page.)\n\n"
            f"{note} The origin quoted our own sentence back at us, at length.\n"
            "and then some more of its own body, long enough to clip as well."
        ),
        details={
            "url": url,
            "final_url": url,
            "status": 503,
            "content_type": "text/html",
            "render_method": "text",
            "cache": "miss",
            "ok": False,
            "http_error": True,
            "attempts": 1,
            "failure_kind": "server",
        },
        measured_s=0.4,
    )
    echo.toggle_expanded()
    echoed = echo._build_content(80).plain.splitlines()
    echoed_note = [row.strip() for row in echoed if row.strip().startswith("The origin asked")]
    assert len(echoed_note) == 1
    assert echoed_note[0].endswith("…")
    assert "quoted our own sentence back" not in echoed_note[0]


def test_the_classified_danger_row_wraps_at_seventy_columns() -> None:
    """Design review round 2, D8.

    ``⚠ HTTP 403 Forbidden — blocked by Akamai bot protection, not page content.``
    is 74 cells against the 58-cell generic row it replaced, so at 70 columns it
    used to clip mid-sentence where the old row fitted whole. A card headline is
    a sentence, not a fixed-width ledger cell, so it reflows like the promoted
    terminal lead.
    """
    url = "https://www.shoppersdrugmart.ca/?lang=en&query=power+bar"
    card = ToolCard("t", "web_fetch", {"url": url})
    card.mark_done(
        f"⚠ HTTP 403 Forbidden — blocked by Akamai bot protection, not page content. {url}\n"
        "text · text/html · cache miss · 2 attempts (default, browser-profile)\n\n"
        "The origin's bot protection refused this request.\n"
        "Next step: use the `browser` tool on this URL.\n",
        {
            "url": url,
            "final_url": url,
            "status": 403,
            "content_type": "text/html",
            "render_method": "text",
            "cache": "miss",
            "lines": 2,
            "ok": False,
            "http_error": True,
            "attempts": 2,
            "profiles": ["default", "browser"],
            "failure_kind": "blocked",
            "block_vendor": "akamai",
            "suggested_tool": "browser",
        },
    )
    card.toggle_expanded()
    rows = [row.strip() for row in card._build_content(70).plain.splitlines()]
    start = next(i for i, row in enumerate(rows) if row.startswith("⚠ HTTP 403"))
    # The headline is a sentence, so it takes the next row: the whole clause is
    # painted and nothing is elided, where the clipped row ended
    # ``…blocked by Akamai bot protection, not page…``.
    danger = rows[start : start + 2]
    assert danger[0].startswith("⚠ HTTP 403 Forbidden — blocked by Akamai")
    assert "bot protection, not page content." in " ".join(danger)
    assert all("…" not in row for row in danger)


def test_the_attempt_summary_wears_signal_ink() -> None:
    """Design review round 2, D9, reconciled with D4's one rule.

    The row's metadata stays ``dim``; the attempt summary is the actionable
    escalation — the reason this fetch succeeded — and the panel's ``dim``
    measures 3.32:1 on the light error tint, below the body floor. One rule: the
    sentence a reader may act on is ``signal``, the metadata recedes.
    """
    card = ToolCard("t", "web_fetch", {"url": "https://medium.com/"})
    card.mark_done(
        "[200] https://medium.com/\nmarkdownify · text/html · cache miss\n\npage",
        {
            "url": "https://medium.com/",
            "final_url": "https://medium.com/",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 1024,
            "lines": 3,
            "ok": True,
            "attempts": 2,
            "profiles": ["default", "browser"],
        },
    )
    card.toggle_expanded()
    built = card._build_content(100)
    assert _triplet(_style_at(built, "· 2 attempts").color) == _triplet(
        Style(color=theme_mod.semantic_color("signal")).color
    )
    assert _triplet(_style_at(built, "Rendered: markdownify").color) == _triplet(
        Style(color=theme_mod.semantic_color("dim")).color
    )


def test_a_same_identity_retry_prints_only_the_count() -> None:
    """Design review round 2, D10 on the card: ``2 attempts (default)`` → ``2 attempts``."""
    card = ToolCard("t", "web_fetch", {"url": "https://flaky.example/x"})
    card.mark_done(
        "[200] https://flaky.example/x\nmarkdownify · text/html · cache miss\n\npage",
        {
            "url": "https://flaky.example/x",
            "final_url": "https://flaky.example/x",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 96,
            "lines": 1,
            "ok": True,
            "attempts": 2,
            "profiles": ["default", "default"],
        },
    )
    card.toggle_expanded()
    content = card._build_content(100).plain
    assert "2 attempts" in content
    assert "(default)" not in content


def test_web_fetch_low_quality_note_surfaced_once() -> None:
    """The low-quality advisory appears, and (D4) only ONCE: the model-facing
    header carried a second `· sparse/JS-gated` copy that D1's header strip
    removes, leaving just the structured advisory row."""
    card = ToolCard("t", "web_fetch", {"url": "https://spa.example"})
    card.mark_done(
        "[200] https://spa.example\nmarkdownify · text/html · cache miss · "
        "sparse/JS-gated (try `browser`)\n\nenable javascript",
        {
            "url": "https://spa.example",
            "final_url": "https://spa.example",
            "status": 200,
            "content_type": "text/html",
            "render_method": "markdownify",
            "cache": "miss",
            "bytes": 100,
            "lines": 1,
            "low_quality": True,
        },
    )
    card.toggle_expanded()
    content = card._build_content(120).plain
    assert "browser" in content
    # D4: exactly one advisory, not the structured row plus the body-header copy.
    assert content.count("sparse/JS-gated") == 1


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
        # `x`/`y` because a real `Click` always carries the cell it landed on,
        # and `on_click` reads them to ask whether a URL was under the pointer
        # (`TranscriptBlock.link_at`) before it treats the click as a toggle.
        # A double without them is not a stand-in for the event, it is a
        # stand-in for an event that cannot exist.
        def __init__(self, x: int = 0, y: int = 0) -> None:
            self.stopped = False
            self.x = x
            self.y = y

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
        # See the sibling double above: a real `Click` carries the cell it
        # landed on, and `on_click` reads it before deciding what the click
        # meant.
        def __init__(self, x: int = 0, y: int = 0) -> None:
            self.stopped = False
            self.x = x
            self.y = y

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
    for notice in (NO_OUTPUT_NOTICE, RUNNING_NOTICE, QUEUED_NOTICE):
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


def test_the_queued_notice_does_not_repeat_the_state_word() -> None:
    """The slot answers "why did nothing expand?"; the row already says which
    wait it is in.

    ``⟨queued⟩`` landed two cells from the status column's own ``queued``, so an
    activation printed the same word twice — which reads as a repaint fault, not
    as an answer (the same class the working line's docstring cites for
    ``· compacting context…`` sitting above ``· compacting context``). The
    family's word for this wait is the sibling of ``still running``: ``waiting``.
    """
    for width in WIDTHS:
        card = ToolCard("q", "wake", {"text": "30m"})
        card.set_composing(14, "wake")
        card.mark_queued()
        card.activate()
        row = card._build_row(width).plain
        # The state word is on the row once (or not at all, narrower than the
        # status column can be drawn) — never twice. `⟨queued⟩` printed it
        # twice on the rows wide enough to hold both runs.
        assert row.count("queued") <= 1, (width, row)

    # Where the slot has room for an answer at all, it IS the wait...
    for width in (40, 80, 200):
        card = ToolCard("q", "wake", {"text": "30m"})
        card.set_composing(14, "wake")
        card.mark_queued()
        card.activate()
        row = card._build_row(width).plain
        assert QUEUED_NOTICE in row or TERSE_QUEUED_NOTICE in row, (width, row)

    # ...and at the width the frames are read at, the answer and the state word
    # are two different words: the wait, and the state.
    card = ToolCard("q", "wake", {"text": "30m"})
    card.set_composing(14, "wake")
    card.mark_queued()
    card.activate()
    row = card._build_row(110).plain
    assert QUEUED_NOTICE in row and "queued" in row, row


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
        "console",
        "send",
    }
    assert builtins <= set(NERD_TOOL_ICONS)
    assert builtins <= set(PLAIN_TOOL_ICONS)


def test_the_gate_switches_the_whole_table_not_just_some_of_it(monkeypatch) -> None:
    """One switch, both directions, no half-Nerd row. Explicit-override cases."""
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


#: Env marker sets that must autodetect as Nerd-capable, one per bundling
#: emulator, keyed by a readable id for the parametrize table.
_NERD_CAPABLE_ENVS = {
    "ghostty_resources": {"GHOSTTY_RESOURCES_DIR": "/opt/ghostty"},
    "ghostty_bin": {"GHOSTTY_BIN": "/opt/ghostty/bin"},
    "ghostty_term_program": {"TERM_PROGRAM": "ghostty"},
    # cmux embeds ghostty and sets TERM=dumb, yet still draws Nerd glyphs: a
    # positive emulator marker must win over a dumb TERM.
    "cmux_ghostty_dumb_term": {"GHOSTTY_BIN": "/opt/ghostty/bin", "TERM": "dumb"},
    "kitty_window_id": {"KITTY_WINDOW_ID": "1"},
    "kitty_term": {"TERM": "xterm-kitty"},
    "wezterm_pane": {"WEZTERM_PANE": "0"},
    "wezterm_executable": {"WEZTERM_EXECUTABLE": "/usr/bin/wezterm"},
    "wezterm_term_program": {"TERM_PROGRAM": "WezTerm"},
    # The Local Operator console: not an emulator marker at all, but the app IS
    # the emulator and it bundles a Nerd-patched face, so the same conclusion
    # follows from the surface marker it injects (design ui-console-tab §6.6).
    "local_operator_console": {"LOCAL_OPERATOR_CONSOLE_SURFACE": "con:1:9f2a"},
}

#: Env marker sets with NO bundled Nerd fallback — these must degrade to plain
#: so the user never sees a tofu box.
_PLAIN_ENVS = {
    "apple_terminal": {"TERM_PROGRAM": "Apple_Terminal", "TERM": "xterm-256color"},
    "plain_xterm": {"TERM": "xterm-256color"},
    "empty": {},
    "dumb_no_marker": {"TERM": "dumb"},
}


@pytest.mark.parametrize("env", list(_NERD_CAPABLE_ENVS.values()), ids=list(_NERD_CAPABLE_ENVS))
def test_autodetect_enables_nerd_icons_for_bundling_terminals(env) -> None:
    """A ghostty/cmux/kitty/wezterm marker means a bundled symbol font, so the
    expanded glyphs render without any user setup — detection is by env marker,
    injectable so the test never touches the real environment."""
    assert glyph_mod._nerd_capable_terminal(env) is True


@pytest.mark.parametrize("env", list(_PLAIN_ENVS.values()), ids=list(_PLAIN_ENVS))
def test_autodetect_falls_back_to_plain_for_unknown_terminals(env) -> None:
    """Apple_Terminal and every unrecognised terminal ship no Nerd fallback,
    so autodetect must say plain: a tofu box is worse than an ASCII icon."""
    assert glyph_mod._nerd_capable_terminal(env) is False


def test_the_console_icon_is_a_different_noun_from_the_shell() -> None:
    """`console` and `bash` must not share a glyph in either table.

    This is the one distinction the icon table exists to keep: `bash` is the
    shell this process runs, the console is a terminal running inside the app.
    `write`/`edit` share a pencil because they share a meaning; these two do
    not, and a reader scanning a ledger of both would otherwise have to read
    the names to tell which row was which.
    """
    assert NERD_TOOL_ICONS["console"] != NERD_TOOL_ICONS["bash"]
    assert PLAIN_TOOL_ICONS["console"] != PLAIN_TOOL_ICONS["bash"]


def test_the_console_marker_is_a_fact_about_the_process_not_a_forgery() -> None:
    """The console marker must not be reachable by impersonating ghostty.

    A design that had set `GHOSTTY_RESOURCES_DIR` in the surface to win this
    predicate would be lying about what is running AND would flip the
    notification protocol for every process in the surface, which is a second
    behaviour bought with the same forgery. The marker is therefore checked by
    its own name, and a `TERM=dumb` console surface still gets glyphs (the app
    renders them, exactly as cmux/ghostty does).
    """
    from local_operator.terminals import CONSOLE_SESSION_ENV, CONSOLE_SURFACE_ENV

    assert glyph_mod._nerd_capable_terminal({CONSOLE_SURFACE_ENV: "con:1:9f2a"}) is True
    assert (
        glyph_mod._nerd_capable_terminal({CONSOLE_SURFACE_ENV: "con:1:9f2a", "TERM": "dumb"})
        is True
    )
    # The session variable alone is not the marker: the surface is the handle
    # both actors name, and it is what the app sets on every surface.
    assert glyph_mod._nerd_capable_terminal({CONSOLE_SESSION_ENV: "s-1"}) is False
    assert glyph_mod._nerd_capable_terminal({}) is False


def test_gate_autodetects_when_config_is_unset(monkeypatch) -> None:
    """Config unset (None = auto) hands the decision to marker detection: the
    bundling terminals get real glyphs, Apple_Terminal and unknowns get plain,
    all with zero config from the user."""
    monkeypatch.delenv(glyph_mod._ENV_DISABLE, raising=False)
    # None models an absent config key — the tri-state "auto" state.
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: None)

    for env in _NERD_CAPABLE_ENVS.values():
        monkeypatch.setattr(glyph_mod, "_nerd_capable_terminal", lambda e=env: True)
        assert nerd_icons_enabled() is True
        assert tool_icon("bash") == NERD_TOOL_ICONS["bash"]

    monkeypatch.setattr(glyph_mod, "_nerd_capable_terminal", lambda: False)
    assert nerd_icons_enabled() is False
    assert tool_icon("bash") == PLAIN_TOOL_ICONS["bash"]


def test_gate_autodetects_ghostty_and_apple_terminal_end_to_end(monkeypatch) -> None:
    """Full path through the real ``os.environ`` read: no kill switch, config
    unset, and the process's own env carrying a ghostty marker vs an
    Apple_Terminal marker."""
    monkeypatch.delenv(glyph_mod._ENV_DISABLE, raising=False)
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: None)
    for var in (
        "GHOSTTY_RESOURCES_DIR",
        "GHOSTTY_BIN",
        "KITTY_WINDOW_ID",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("GHOSTTY_BIN", "/opt/ghostty/bin")
    monkeypatch.setenv("TERM", "dumb")  # cmux/ghostty case: marker beats dumb TERM
    assert nerd_icons_enabled() is True

    monkeypatch.delenv("GHOSTTY_BIN", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    monkeypatch.setenv("TERM", "xterm-256color")
    assert nerd_icons_enabled() is False


def test_explicit_config_overrides_autodetection_both_ways(monkeypatch) -> None:
    """An explicit bool is the user's decision and wins over the terminal: True
    forces glyphs on Apple_Terminal (patched font installed by hand), False
    forces them off on ghostty."""
    monkeypatch.delenv(glyph_mod._ENV_DISABLE, raising=False)

    # Explicit True on an Apple_Terminal env — detection would say plain.
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: True)
    monkeypatch.setattr(glyph_mod, "_nerd_capable_terminal", lambda env=None: False)
    assert nerd_icons_enabled() is True
    assert tool_icon("bash") == NERD_TOOL_ICONS["bash"]

    # Explicit False on a ghostty env — detection would say Nerd.
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: False)
    monkeypatch.setattr(glyph_mod, "_nerd_capable_terminal", lambda env=None: True)
    assert nerd_icons_enabled() is False
    assert tool_icon("bash") == PLAIN_TOOL_ICONS["bash"]


def test_env_kill_switch_beats_config_and_autodetection(monkeypatch) -> None:
    """The kill switch is checked first: it wins over an explicit True config
    and over a Nerd-capable terminal alike (CI, snapshot harnesses)."""
    monkeypatch.setenv(glyph_mod._ENV_DISABLE, "1")
    monkeypatch.setattr(glyph_mod, "settings_get", lambda key, default=None: True)
    monkeypatch.setattr(glyph_mod, "_nerd_capable_terminal", lambda env=None: True)
    assert nerd_icons_enabled() is False
    assert tool_icon("bash") == PLAIN_TOOL_ICONS["bash"]


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
    """The icon is added TO the row, not instead of part of it.

    Led after the row's left inset (``ROW_INDENT``), not from column 0: the
    summary is drawn on the card's own fill, and the icon sitting flush
    against that fill's left wall is what the inset exists to stop.
    """
    card = ToolCard("t", "grep", {"pattern": "needle"})
    row = card._build_row(80).plain
    assert row.startswith(" " * ROW_INDENT + tool_icon("grep") + " ")
    assert "grep" in row and "needle" in row
    _assert_fits(card)


def test_the_row_indent_is_given_up_before_the_answer_slot() -> None:
    """The inset is breathing room, and breathing room goes first.

    At the narrow end the builder is already shedding the tool name a
    character at a time; one more cell spent on aesthetics pushes the row past
    the rung where the inert-row answer survives. So the indent is present at
    the threshold and gone below it — asserted at the boundary rather than at
    a comfortable width, because the boundary is the part that can regress.
    """
    icon = tool_icon("grep")
    at_threshold = ToolCard("t", "grep", {"pattern": "needle"})
    assert at_threshold._build_row(ROW_INDENT_MIN_WIDTH).plain.startswith(" " * ROW_INDENT + icon)
    below = ToolCard("t", "grep", {"pattern": "needle"})
    assert below._build_row(ROW_INDENT_MIN_WIDTH - 1).plain.startswith(icon)


def test_two_different_tools_do_not_share_a_row_prefix() -> None:
    """The whole point of the icon: a run of rows is told apart by shape."""
    prefixes = {
        name: ToolCard("t", name, {})._build_row(80).plain[ROW_INDENT]
        for name in ("bash", "read", "write", "grep", "browser")
    }
    assert len(set(prefixes.values())) == len(prefixes), prefixes


def test_the_running_icon_is_the_accent_and_settles_to_its_category() -> None:
    """One of the places the accent green is spent: a still frame has to read
    "live" without the shimmer (D26).

    On settling, the icon takes its tool's CATEGORY ink rather than one flat
    grey. The name was already category-coded, so the glyph beside it was the
    only mark carrying identity by shape while carrying no colour — which is
    what made a settled ledger a wall of grey. Running still wins outright:
    liveness outranks identity.
    """
    card = ToolCard("t", "bash", {"command": "sleep 5"})
    icon = tool_icon("bash")
    live = _style_at(card._build_row(80), icon)
    assert _triplet(live.color) == _triplet(Style(color=theme_mod.semantic_color("accent")).color)

    card.mark_done("ok")
    settled = _style_at(card._build_row(80), icon)
    expected = bindings.style(_category_element("bash"))
    assert _triplet(settled.color) == _triplet(expected.color)
    # and it is no longer the flat grey every tool used to share
    assert _triplet(settled.color) != _triplet(Style(color=theme_mod.semantic_color("dim")).color)


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


def test_a_sub_50ms_tool_never_renders_the_fabricated_0_0s() -> None:
    """A real sub-50 ms call must not reprint the bug's own string.

    ``f"{0.04:.1f}s"`` is ``0.0s`` — byte-identical to what the fabricated-
    duration defect printed on a row whose duration was simply missing, which
    is why that report was re-filed against a tool that had returned at once.
    ``<0.1s`` says "too fast to measure", and it is exactly DURATION_COL wide,
    so the pass/fail column does not move.
    """
    fast = ToolCard("a", "bash", {"command": "pytest"})
    fast.mark_done("passed")
    fast._duration = 0.04
    row = fast._build_row(80).plain
    assert "<0.1s" in row
    assert "0.0s" not in row

    assert len("<0.1s") == DURATION_COL
    offsets = []
    for elapsed in (0.04, 0.4):
        card = ToolCard("t", "bash", {"command": "pytest"})
        card.mark_done("passed")
        card._duration = elapsed
        plain = card._build_row(80).plain.rstrip()
        offsets.append(len(plain) - plain.rindex("✓"))
    assert len(set(offsets)) == 1, offsets


def test_an_unknown_duration_stays_blank_rather_than_claiming_fast() -> None:
    """``<0.1s`` is a CLAIM; a lost duration must keep saying nothing."""
    card = ToolCard("t", "bash", {"command": "pytest"})
    card.mark_done("passed")
    card._duration = None
    row = card._build_row(80).plain
    assert "<0.1s" not in row
    assert "0.0s" not in row


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


#: The lead every row of a wrapped failure REASON carries after OUTPUT_INDENT:
#: the card's own error glyph on the first row, two blanks on the continuations
#: (design round 1, D1). It is a GLYPH rather than an ink step because the plain
#: body paints its reason in `tool.output.error`, which IS the captured rows' ink
#: on an error card (both resolve to `tint-danger`) — so ink separates nothing,
#: and on a colourless terminal it would say nothing either.
REASON_LEAD = f"{ICON_ERROR} "
#: Every lead a promoted block can open with: an error's cause, and — design
#: review round 1, D1 — a partial result's disclosure. Same mechanism, so the
#: helpers below locate the block by either rather than by one card's state.
_REASON_LEADS = (REASON_LEAD, f"{ICON_PARTIAL} ")


def _collapsed(text: str) -> str:
    """The one-space form the collapsed row's status is stored in."""
    return " ".join(text.split())


def _reason_rows(body: list[str]) -> list[str]:
    """The reason block's text: the body indent and the D1 lead removed.

    Also asserts the lead SHAPE, because it is the one thing telling these rows
    from the tool's captured bytes on a monochrome terminal. Either state glyph
    is accepted (:data:`_REASON_LEADS`): the block is the same mechanism for an
    error's cause and for a partial result's disclosure, and the continuations
    carry blanks of the same width whichever opened it.
    """
    rows: list[str] = []
    lead = ""
    for index, line in enumerate(body):
        assert line.startswith(" " * OUTPUT_INDENT), (index, line)
        text = line[OUTPUT_INDENT:]
        if index == 0:
            lead = next((c for c in _REASON_LEADS if text.startswith(c)), "")
            assert lead, (index, line)
        else:
            assert text.startswith(" " * len(lead)), (index, line)
        rows.append(text[len(lead) :])
    return rows


def _reason_block(rows: list[str]) -> list[str]:
    """The reason block's painted rows, located by its lead rather than by index.

    The rows above it are the summary and the ARGUMENT block, and the argument
    block's own row count moves with the width (it wraps the command, and drops
    its label on a narrow frame), so the block is found by the lead it is the
    only carrier of. The overflow marker carries the same two blanks, so it is
    part of the block too.
    """
    start = next(
        index
        for index, line in enumerate(rows)
        if any(line.startswith(" " * OUTPUT_INDENT + lead) for lead in _REASON_LEADS)
    )
    end = start + 1
    while end < len(rows) and rows[end].startswith(" " * OUTPUT_INDENT + " " * len(REASON_LEAD)):
        end += 1
    return rows[start:end]


def test_a_long_failure_reason_wraps_with_the_body_indent() -> None:
    """The expansion is the ONLY state that can carry a long failure's cause.

    The collapsed row's status cap is a fraction of the row that the reason
    shares with the glyph and the clock, so at 80 columns the row paints
    ``Web search fail…`` and the cause is unreachable there at every width.
    Measured on the real card before this fix: the expanded body cropped the
    sentence at its 72-cell measure, leaving ``refused this search`` and
    ``only provider tried`` ABSENT from the frame entirely (#1066).
    """
    message = (
        "Web search failed: Fetch a page directly, or set PERPLEXITY_API_KEY for "
        "keyed Sonar: the anonymous tier refused this search (wall "
        "fraud_authwall_upsell/LOGIN) ('perplexity' was the only provider tried)"
    )
    card = ToolCard("t", "web_search", {"query": "openai rate limits"})
    card.mark_failed(message)
    card.toggle_expanded()

    body = card._build_content(80).plain.splitlines()[2:]
    # Every continuation keeps the body indent, so a wrapped fragment does not
    # read as a stray transcript row — the defect `session_panel._Body.note`
    # was fixed for — and every row carries the block's lead.
    rows = _reason_rows(body)
    assert len(rows) == 3
    # The sentence is whole: only the wrap's own line breaks were added.
    assert _collapsed(" ".join(rows)) == _collapsed(message)
    assert all(cell_len(line) <= 80 for line in body)


def test_the_body_wrap_is_budgeted_to_the_reason_not_captured_output() -> None:
    """A tool's own bytes still clip: one output line is one row.

    This is the half of the budget the merge is judged on. Measured at 80
    columns on the real card: a 40-line stdout of 397-cell lines occupies 40
    body rows before AND after the wrap — zero rows added. Wrapping every body
    line instead would have spent six rows on each of those lines (240 rows
    against 40), turning one tool call into a scroll trap in the transcript.
    """
    long_line = "payload=" + "x" * 400
    card = ToolCard("t", "bash", {"command": "curl"})
    card.mark_done("\n".join([long_line] * 5))
    card.toggle_expanded()

    body = card._build_content(80).plain.splitlines()[2:]
    assert len(body) == 5
    assert all(line.rstrip().endswith("…") for line in body)
    assert all(cell_len(line) <= 80 - 2 for line in body)


def test_a_failure_wraps_its_reason_and_still_clips_its_captured_output() -> None:
    """One card, both halves: the reason wraps, the receipt beside it crops."""
    reason = "ModelProviderError: " + "rate limited; " * 12
    card = ToolCard("t", "bash", {"command": "pytest"})
    card.mark_failed(reason, reason + "\n" + "x" * 400)
    card.toggle_expanded()

    body = card._build_content(80).plain.splitlines()[2:]
    assert len(body) == 4
    rows = _reason_rows(body[:-1])
    assert sum(1 for row in rows if "rate limited" in row) == 3
    assert rows[0].startswith("ModelProviderError:")
    # The captured line after the reason is still exactly one cropped row.
    assert body[-1].rstrip().endswith("…")


def test_a_result_whose_head_line_is_not_the_reason_keeps_the_plain_crop() -> None:
    """The expansion is the tool's OUTPUT — no synthetic row restating the row.

    ``mark_failed(error, result_text)`` with a ``result_text`` that does not
    begin with the error means the body's head line is captured output, so it
    keeps the crop and the card does not grow a duplicate of the reason the
    collapsed row already carries in full.
    """
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed("exit status 1", "Traceback:\n  boom")
    card.toggle_expanded()

    body = card._build_content(80).plain.splitlines()[2:]
    assert body == ["  Traceback:", "    boom"]


def test_a_tool_authored_head_line_is_claimed_when_the_status_leads_with_it() -> None:
    """R2: the guard is a LEADS test, not a provenance test.

    On the bash surfaces the reason IS the tool's own first output line
    (``app.py`` settles a failed call with
    ``mark_failed(_first_line(result.text), result.text, …)``), so an invariant
    phrased around who composed the line would be false on the surface this card
    most often shows. What the guard tests is whether the collapsed row's status
    leads with the body's head line — and the other half of that same test: a
    body whose head line is NOT the head of the status stays captured.
    """
    raw = "exit status 1: no such file or directory"
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed(_first_line(raw), raw + "\nretry 1")
    assert card._failure_reason() == _first_line(raw)

    other = ToolCard("t", "bash", {"command": "false"})
    other.mark_failed("exit status 2: oops", raw + "\nboom")
    assert other._failure_reason() == ""


#: The three shapes R1 reproduced, as the ``(error, result_text)`` pairs the app
#: itself builds for them. The first two go through the bash surface's settle
#: call (``app.py:22491``/``:38272``: ``mark_failed(_first_line(result.text),
#: result.text, details)``) and are what the whitespace COLLAPSE hides — the body
#: keeps the caller's spacing while ``_error`` is stored ``" ".join(error.split())``,
#: so the raw head line is never equal to it. The third is the executor throw's
#: ``mark_failed(str(error), str(error))`` (``app.py:22464``), whose collapsed
#: status is the head line PLUS the traceback lines: equality can never match it.
_WS_RUN = (
    "ERROR  at 2026-09-13T00:00Z  could not connect to "
    "https://api.example.com/v1/hook after 3 tries"
)
_TAB = "ERROR:\tcould not connect to https://api.example.com/v1/hook after 3 tries, giving up"
_MULTILINE = (
    "ERROR: could not connect to https://api.example.com/v1/hook after 3 "
    'tries, giving up\n  File "/app/hook.py", line 12, in post\n'
    "    raise TimeoutError"
)
APP_SHAPED_FAILURES = (
    pytest.param(_first_line(_WS_RUN), _WS_RUN, id="whitespace-run-in-the-head-line"),
    pytest.param(_first_line(_TAB), _TAB, id="tab-in-the-head-line"),
    pytest.param(_MULTILINE, _MULTILINE, id="multi-line-error"),
)


@pytest.mark.parametrize(("error", "result_text"), APP_SHAPED_FAILURES)
def test_an_app_shaped_failure_wraps_the_head_line_its_status_leads_with(
    error: str, result_text: str
) -> None:
    """R1: the wrap must fire for the shapes the app actually builds.

    On the pre-fix tree every shape below answered "" from ``_failure_reason()``
    — the whitespace-exact comparison against the collapsed ``_error`` refused
    them — so the cause stayed cropped, silently, with no test covering the
    class.
    """
    head = _first_line(result_text)
    assert cell_len(head) > 80 - 2 - OUTPUT_INDENT, head
    card = ToolCard("t", "bash", {"command": "curl"})
    card.mark_failed(error, result_text, None, measured_s=0.4)
    card.toggle_expanded()

    reason = card._failure_reason()
    assert reason, "the collapsed row's status leads with this line"
    assert _collapsed(reason) == _collapsed(head)

    body = card._build_content(80).plain.splitlines()[2:]
    raw_traceback = result_text.splitlines()[1:]
    if raw_traceback:
        # Only the HEAD line is claimed. The rest of a multi-line error stays in
        # the body, cropped like every other captured line, and is never painted
        # twice.
        assert [line[OUTPUT_INDENT:] for line in body[-len(raw_traceback) :]] == raw_traceback
        body = body[: -len(raw_traceback)]
    rows = _reason_rows(body)
    assert _collapsed(" ".join(rows)) == _collapsed(head)
    assert not rows[-1].rstrip().endswith("…")


def test_the_reason_block_is_bounded_by_cells_and_rows_at_every_width() -> None:
    """Two budgets: CELLS bound the content, ROWS bound the shape.

    A cell budget alone would let a 16-column frame (an 8-cell measure) turn 432
    cells into 54 rows; a row budget alone could not carry the shipped 201-cell
    sentence at 40 columns (8 rows at that frame's 32-cell measure). Measured
    over every width the one-line guarantee is checked at, rather than trusted
    from the comment.
    """
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed(" ".join(["boom"] * 2000))
    card.toggle_expanded()

    for width in WIDTHS:
        all_rows = card._build_content(width).plain.splitlines()
        body = _reason_block(all_rows)
        marker = bool(body) and body[-1].strip().startswith("…")
        rows = _reason_rows(body[:-1] if marker else body)
        assert len(rows) <= REASON_MAX_ROWS, (width, len(rows))
        # One row is exempt from the cell budget on purpose: a reason the reader
        # cannot see at all is the bug this exists to fix.
        if len(rows) > 1:
            assert sum(cell_len(row) for row in rows) <= REASON_MAX_CELLS, width
        assert all(cell_len(line) <= width for line in all_rows), width


def test_the_reason_overflow_marker_names_the_rows_it_dropped() -> None:
    """D2/Q1: the cut carries the surface's own ``… N more lines``.

    A bare ``…`` reads as an in-sentence elision; the block says how many rows
    went, the way the captured crop and the live body already do.
    """
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed(" ".join(["boom"] * 2000))
    card.toggle_expanded()

    body = card._build_content(80).plain.splitlines()[2:]
    marker = body[-1].strip()
    assert marker.startswith("… ")
    assert marker.endswith(" more lines")
    # And it counts exactly the rows this card dropped: the wrapped sentence
    # minus the rows the body still holds (the marker is not one of them).
    wrapped = wrap_cells(card._failure_reason(), 80 - 2 - OUTPUT_INDENT)
    assert int(marker.split()[1]) == len(wrapped) - (len(body) - 1)
    # No sentence row carries a bare ellipsis any more.
    assert not any(line.rstrip().endswith("…") for line in body[:-1])


def test_the_reason_survives_the_narrow_frame_the_flat_row_cap_cut() -> None:
    """D2/Q1: the row bound is a SHAPE backstop, not a cut at the sentence's scale.

    Measured: the shipped 201-cell sentence needs 7 rows at the 45-column frame's
    37-cell measure and 8 at the 40-column frame's 32-cell one — so the flat
    six-row cap dropped ``only provider tried`` off the frame at both widths: the
    #1066 symptom returning below ~48 columns, where ``REASON_MAX_CELLS`` alone
    could not bound the shape.
    """
    message = (
        "Web search failed: Fetch a page directly, or set PERPLEXITY_API_KEY for "
        "keyed Sonar: the anonymous tier refused this search (wall "
        "fraud_authwall_upsell/LOGIN) ('perplexity' was the only provider tried)"
    )
    # The card's own lane at a 40- and a 45-column frame (the transcript hands
    # the card 4 cells fewer than the terminal, as the evidence frames show).
    for card_width in (36, 41):
        card = ToolCard("t", "web_search", {"query": "openai rate limits"})
        card.mark_failed(message)
        card.toggle_expanded()

        body = card._build_content(card_width).plain.splitlines()[2:]
        joined = " ".join(line.strip() for line in body)
        # The CAUSE is on the frame. This is the assertion the flat six-row cap
        # failed at both widths — it stopped at ``… fraud_authwall_upsell/LO``
        # (36) and ``… ('perplexity' was the on`` (41), so the tail clause the
        # issue is about went missing again below ~48 columns.
        assert "only provider tried" in joined, card_width
        assert not any(line.strip().startswith("…") for line in body), card_width
        # And it is still the leaded block the rest of this round pins.
        rows = _reason_rows(body)
        assert _collapsed(" ".join(row.strip() for row in rows)) == _collapsed(message), card_width


def test_the_wrapped_reason_carries_a_monochrome_lead() -> None:
    """D1: the block says “this is the card's own sentence” without ink.

    The reason rides ``tool.output.error``, which is the captured rows' ink on an
    error card (both resolve to ``tint-danger``), so before this round a 42-row
    failure card had no visual answer to “which of these rows is our sentence?” —
    the wrap is what made the question askable. The lead echoes the collapsed
    row's own glyph: the card's one piece of monochrome-safe state vocabulary.
    """
    card = ToolCard("t", "bash", {"command": "false"})
    card.mark_failed("ModelProviderError: " + "rate limited; " * 12)
    card.toggle_expanded()

    content = card._build_content(80)
    body = content.plain.splitlines()[2:]
    assert len(body) == 3
    # Asserts the lead's SHAPE on every row: the glyph on the first, two blanks
    # on the continuations, both after OUTPUT_INDENT.
    rows = _reason_rows(body)
    assert rows[0].startswith("ModelProviderError:")
    assert all(cell_len(line) <= 80 for line in body)
    # The glyph rides the outcome ink the collapsed row paints it in, and the
    # needle is body-only because the summary row puts the reason BEFORE its
    # glyph.
    assert _triplet(_style_at(content, f"{REASON_LEAD}ModelProviderError").color) == _triplet(
        Style(color=theme_mod.semantic_color("danger")).color
    )


def test_the_reason_is_painted_once_and_the_hidden_count_follows_it() -> None:
    """The reason is not printed twice, and the marker counts what is left.

    ``mark_failed`` defaults ``result_text`` to the error, so the sentence is
    both the reason and the body's first line: the wrap claims that line, and
    the hidden-line marker must then count the lines the body still holds
    rather than the raw result.
    """
    total = EXPAND_MAX_LINES + 5
    card = ToolCard("t", "read", {"path": "big.txt"})
    card.mark_failed("boom", "boom\n" + "\n".join(f"line {i}" for i in range(total)))
    card.toggle_expanded()

    rows = card._build_content(80).plain.splitlines()
    assert rows.count(f"  {REASON_LEAD}boom") == 1
    assert rows[-1].strip() == f"… {total - EXPAND_MAX_LINES} more lines"


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


def test_send_summary_names_mode_target_and_message() -> None:
    """The send row leads with HOW it lands, then WHO, then the body.

    The mode marker is the discriminator and the row builder sheds from the
    right, so the marker leads: behind the target it died first, and three
    different delivery promises painted identical rows."""
    card = ToolCard(
        "t",
        "send",
        {"target": "release cutter", "message": "gates are green, ready for review"},
    )
    row = card._build_row(100).plain
    assert "release cutter" in row
    assert "wake" in row  # the default delivery mode is visible
    assert "gates are green" in row
    # Order: mode, then target, then the body.
    assert row.index("wake") < row.index("release cutter") < row.index("gates are green")


def test_send_summary_marks_the_quiet_drop_and_now() -> None:
    quiet = ToolCard("t", "send", {"target": "peer", "message": "later", "wake": False})
    assert "quiet" in quiet._build_row(80).plain
    now = ToolCard("t", "send", {"pid": 48213, "message": "stop", "now": True})
    row = now._build_row(80).plain
    assert "pid 48213" in row
    assert "now" in row


def test_send_modes_never_collide_at_narrow_widths() -> None:
    """The defect the leading marker fixes: with the mode to the RIGHT of a long
    target it was truncated away, so a wake, a quiet drop and a mid-turn steer
    painted byte-identical rows at ordinary widths (measured: a 37-cell name
    collided at 62 columns). One of those rows woke a peer and one did not."""
    target = "minerva-user-dashboard-release-cutter"
    message = "gates are green, ready for review"
    for width in (40, 48, 52, 60, 62, 80):
        rows = {
            ToolCard("t", "send", {"target": target, "message": message, **extra})
            ._build_row(width)
            .plain
            for extra in ({}, {"wake": False}, {"now": True})
        }
        assert len(rows) == 3, f"delivery modes collide at {width} columns: {rows}"


def test_send_summary_uses_the_rows_own_budget_for_a_long_message() -> None:
    """No private preview cap: the row's own truncation does the shedding, and
    an extra bound only left a third of the line empty at normal widths. The
    mode and target survive because the mode leads."""
    card = ToolCard("t", "send", {"target": "peer", "message": "x" * 500})
    row = card._build_row(60).plain
    assert "peer" in row
    assert "wake" in row
    # The body fills the line rather than stopping early at a private cap.
    assert row.count("x") > 20


def test_send_summary_stands_in_for_a_missing_target() -> None:
    """The card is painted before the call fails, and a blank slot made the row
    read as though the peer were called "wake"."""
    row = ToolCard("t", "send", {"message": "hello there"})._build_row(80).plain
    assert "wake · ?" in row


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
    """Pins the HARNESS's reverse-tab order, not a route the product has.

    Read this before citing it. In `_ComposerApp` — a stripped harness with a
    transcript and an editor and nothing else — Shift+Tab out of the composer
    does reach the most recent action rather than the oldest, and that ordering
    is what this test still guards.

    In the shipped `OperatorApp` it does not happen at all: `shift+tab` is
    bound to `cycle_effort` with `priority=True` (`app.py`), and Textual
    matches priority bindings before the focused widget ever sees the key.
    Verified against the real app — Shift+Tab from the composer leaves focus on
    the Editor and cycles the effort tier.

    So there is currently NO keyboard route into the ledger, which leaves
    `ToolCard.can_focus` (justified below by exactly this route) reachable only
    by mouse. Closing that gap needs a new key chosen against an already
    crowded keymap, so it is deliberately out of scope here and tracked
    separately; the docstring is corrected rather than the test moved, because
    a test asserting a route the product does not have is worse than no test.
    """
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
    assert running.startswith(f"{' ' * ROW_INDENT}{tool_icon('bash')} bash")
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
    """ "Abbreviated within the available horizontal line space" — and the SAME
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


def test_an_expanded_running_card_can_still_be_closed_once_it_settles_empty() -> None:
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


def test_the_live_advisory_is_a_persistent_state_line_not_output() -> None:
    """The soft memory advisory survives a chatty command (design review D1).

    Prepended to the output it rode the HEAD of a block that keeps the TAIL, so
    on a command printing more than :data:`LIVE_MAX_LINES` — precisely the
    memory-pressure case it exists for — it scrolled off. As a state line under
    the header it is present no matter how much the command prints.
    """
    card = ToolCard("t", "bash", {"command": "seq 100000"})
    card._expanded = True
    card.set_live_advisory("memory 2.7/3.2 GB — approaching the command budget")
    # A chatty payload that would evict any prepended line.
    card.set_partial_detail("\n".join(str(n) for n in range(1, 101)))

    body = card._build_content(80).plain
    assert f"{LIVE_ADVISORY_GLYPH} memory 2.7/3.2 GB" in body
    # It sits with the HEADER (the state block), ABOVE the drop marker and the
    # bounded output tail — not inside output, where it would scroll off.
    header_row = body.index(LIVE_HEADER_RUNNING)
    advisory_row = body.index("approaching the command budget")
    assert advisory_row > header_row
    assert advisory_row < body.index("… ")


def test_the_live_advisory_is_cleared_on_settle_and_on_an_explicit_clear() -> None:
    """The state line does not outlive the condition."""
    card = ToolCard("t", "bash", {"command": "echo hi"})
    card._expanded = True
    card.set_live_advisory("memory 2.7/3.2 GB — approaching the command budget")
    assert "approaching the command budget" in card._build_content(80).plain
    card.set_live_advisory(None)
    assert "approaching the command budget" not in card._build_content(80).plain
    card.set_live_advisory("memory 2.7/3.2 GB — approaching the command budget")
    card._settle_live()
    assert card._live_advisory is None


def test_the_live_advisory_uses_its_own_binding_not_the_output_ink() -> None:
    """D2: the advisory is the HARNESS speaking, in a colour distinct from the
    command's own stdout on the same card."""
    advisory_style = bindings.style("tool.live.advisory")
    output_style = bindings.style("tool.live.dim")
    assert advisory_style != output_style
    # The binding resolves to the theme's amber (warning), not `dim`.
    assert advisory_style.color is not None
    assert (
        advisory_style.color.get_truecolor().hex.lower()
        == theme_mod.semantic_color("warning").lower()
    )


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


# --- repaint vs reflow -----------------------------------------------------
#
# A collapsed card is pinned to `height: 1` by the sheet and an expanded one is
# exactly its row count, so a repaint that did not move the row count cannot
# move anything the container places. Textual's `Static.update` reflows by
# default, and a reflow re-arranges the WHOLE transcript — 7.8 ms across 173
# widgets on a 161-block screen. That was being paid by the 1 Hz clock on every
# running card and by every pointer crossing a row's edge.


def _layout_flags(card: ToolCard) -> list[bool]:
    """Record the ``layout`` argument of every content update on ``card``."""
    seen: list[bool] = []
    original = type(card).set_content

    def recording(self, renderable, *, layout: bool = True) -> None:
        seen.append(layout)
        original(self, renderable, layout=layout)

    card.set_content = recording.__get__(card)  # type: ignore[assignment]
    return seen


@pytest.mark.asyncio
async def test_a_repaint_that_kept_the_row_count_asks_for_no_layout_pass() -> None:
    """The running card's clock repaints once a second to move the duration.
    The row is still one row, so nothing needs re-placing.

    Mounted, because a detached card measures its content and applies none of
    it — the branch under test is the one that only runs with a real width.
    """
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 14)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("t", "bash", {"command": "sleep 30"})
        view.append_block(card)
        await pilot.pause()
        flags = _layout_flags(card)

        card._refresh_row()

        assert card._row_count == 1, "precondition: a collapsed card is one row"
        assert flags == [False]


@pytest.mark.asyncio
async def test_expanding_a_card_does_ask_for_the_layout_pass() -> None:
    """The other half: expansion is the case where the footprint really moves,
    and skipping the reflow there would paint the body into reserved-for-one."""
    app = StyledTranscriptApp()
    async with app.run_test(size=(90, 14)) as pilot:
        view = app.query_one(TranscriptView)
        card = ToolCard("t", "bash", {"command": "ls"})
        view.append_block(card)
        card.mark_done("line one\nline two\nline three")
        await pilot.pause()
        flags = _layout_flags(card)

        assert card.toggle_expanded() is True

        assert card._row_count > 1, "precondition: the expansion added rows"
        assert flags and flags[-1] is True


def test_a_resize_that_did_not_change_the_width_rebuilds_nothing() -> None:
    """An expanded card's own content sets its height, so every expansion
    raises a Resize straight back into ``on_resize``. Unguarded, that rebuilt
    the row to reproduce it byte for byte — measured at half of all rebuilds."""
    card = ToolCard("t", "bash", {"command": "ls"})
    width = card._built_width
    builds = {"n": 0}
    original = ToolCard._refresh_row
    card._refresh_row = (  # type: ignore[assignment]
        lambda: (builds.__setitem__("n", builds["n"] + 1), original(card))[1]
    )

    card.on_resize(SimpleNamespace(size=SimpleNamespace(width=width)))

    assert builds["n"] == 0


def test_a_resize_to_a_new_width_does_rebuild() -> None:
    """The guard must still let a real width change through: the row is folded
    to a width, and a stale fold either clips or leaves a hole."""
    card = ToolCard("t", "bash", {"command": "ls"})
    builds = {"n": 0}
    original = ToolCard._refresh_row
    card._refresh_row = (  # type: ignore[assignment]
        lambda: (builds.__setitem__("n", builds["n"] + 1), original(card))[1]
    )

    card.on_resize(SimpleNamespace(size=SimpleNamespace(width=card._built_width - 20)))

    assert builds["n"] == 1


# ---------------------------------------------------------------------------
# The seeded clock: a row adopted for a call that is ALREADY running.
#
# Reported from the field: switch away from a session whose tool is executing
# and switch back, and the row's elapsed reading restarts at zero and counts up
# from the switch — a `bash` row reading `27s` (and the band above it saying the
# same) while the call is half an hour old. The row was not wrong to refuse a
# number: it had NO start instant, so any number it printed was a claim about
# when the VIEWER arrived. What changed is that the start is now knowable, so
# the refusal is precise rather than total.
#
# These drive the card directly with an INJECTED clock, so "the number is the
# call's true age" is arithmetic rather than a sleep: a test that slept 27
# seconds to prove a count-up would be a test nobody runs. The wall-clock part
# only converts the epoch once, at the seed; see `monotonic_from_epoch`.
# ---------------------------------------------------------------------------


def test_a_restored_running_row_arms_from_the_call_start_epoch() -> None:
    """The epoch is the call's own start, so the number IS its age.

    And it keeps ticking from that anchor — the second read is three seconds
    later on the injected monotonic clock, not a re-conversion of the epoch, so
    a wall-clock adjustment after the seed cannot move a running counter.
    """
    now = [1_000.0]
    card = ToolCard("t", "bash", {"command": "sleep 30"}, clock=lambda: now[0])
    card.restore(state="running", started_at=time.time() - 27.0)

    assert card.started_at is not None, "a call whose start is known dates itself"
    assert card.started_at == pytest.approx(now[0] - 27.0, abs=0.05)
    assert card._build_row(80).plain.rstrip().endswith("27s")

    now[0] += 3.0
    assert card._build_row(80).plain.rstrip().endswith("30s")


def test_a_restored_running_row_without_a_start_epoch_still_withholds() -> None:
    """The other half of the same arm, and not a legacy curiosity.

    A `subagent_view` child row and a call whose producer predates the field
    both reach this with nothing to seed from, and the honest rendering for
    them is still the blank column: their ``_started`` is when this surface
    painted the row, so any number here would be about the viewer.
    """
    now = [1_000.0]
    card = ToolCard("t", "bash", {"command": "sleep 30"}, clock=lambda: now[0])
    card.restore(state="running")

    assert card.started_at is None
    assert card._elapsed() is None
    assert not card._build_row(80).plain.rstrip().endswith("s")


def test_a_settled_row_ignores_a_start_epoch() -> None:
    """Settled states are rendered from the executor's measured interval.

    Passing an epoch must not change that: the receipt for a finished call is
    ``duration_s``, and a start instant has no part in it.
    """
    now = [1_000.0]
    card = ToolCard("t", "bash", {"command": "sleep 1"}, clock=lambda: now[0])
    card.restore(state="success", result_text="done", duration_s=6.0, started_at=time.time() - 27.0)

    assert card.started_at is None, "a settled row holds no running start"
    assert card._duration == 6.0, "the executor's measured interval is the receipt"
    row = card._build_row(80).plain
    assert "6.0s" in row
    assert "27s" not in row, "the epoch has no part in a settled receipt"


def test_begin_running_arms_the_clock_from_the_epoch_when_it_has_one() -> None:
    """The re-entry arm, which the switch actually takes.

    A re-delivered ``ToolStarted`` — `/resume` onto the running turn, or a
    switch away and back replaying the owner's live seed — reaches the SAME
    card the projection restored, through ``begin_running``. Seeding only
    ``restore`` would leave this call resetting the row to zero, which is
    exactly the reported frame.
    """
    now = [2_000.0]
    card = ToolCard("t", "bash", clock=lambda: now[0])
    card.set_composing(12, "bash")
    card.begin_running("bash", {"command": "sleep 30"}, None, started_at=time.time() - 41.0)

    assert card.started_at == pytest.approx(now[0] - 41.0, abs=0.05)
    assert card._build_row(80).plain.rstrip().endswith("41s")

    # Without an epoch the re-entry keeps the withheld clock, which is what a
    # caller with nothing to pass must not be able to lose by accident.
    other = ToolCard("t2", "bash", clock=lambda: now[0])
    other.restore(state="running")
    other.begin_running("bash", {"command": "sleep 30"}, None)
    assert other.started_at is None


def test_an_unknown_start_mounts_clockless_where_none_still_means_begins_now() -> None:
    """QA round 1, Q1: the constructor had one word for two different cases.

    ``None`` on ``started_at`` means "this call begins now", and that is right
    for the ordinary mount — the row appears with the event that started the
    call, so the two instants coincide. It is WRONG for the adopted mount: a
    start event that reaches a view with no row for its call (a re-delivered
    live seed, a rebuilt transcript) is mounting a row for work already in
    flight, and stamping the arrival instant there prints an age about the
    VIEWER — measured at `0s` on the mount and `3s`/`5s` end to end for a call
    that was by then ~13s old.

    So the second case gets its own value. ``START_UNKNOWN`` withholds the
    clock, which is the same reading ``restore`` and ``begin_running`` already
    give a missing epoch, and leaves the ordinary path exactly as it was —
    asserted together, because a fix that withheld both would have broken every
    native row to fix one seam.
    """
    now = [1_500.0]
    unknown = ToolCard(
        "u", "bash", {"command": "sleep 30"}, clock=lambda: now[0], started_at=START_UNKNOWN
    )
    assert unknown.started_at is None
    assert unknown._elapsed() is None
    assert not unknown._build_row(80).plain.rstrip().endswith("s")
    # Not even after time passes and the clock is asked again: this row can
    # never date itself, because it never learned when the call began.
    now[0] += 5.0
    assert unknown._elapsed() is None
    assert not unknown._build_row(80).plain.rstrip().endswith("s")

    ordinary = ToolCard("n", "bash", {"command": "sleep 30"}, clock=lambda: now[0])
    assert ordinary.started_at == 1_505.0, "the ordinary mount still takes the mount instant"
    assert ordinary._build_row(80).plain.rstrip().endswith("0s")


def test_a_row_that_watched_its_own_start_keeps_its_own_zero() -> None:
    """The seeded arm must not displace the ordinary one.

    A call this process watched begin dates itself from the event that started
    it. Pushing a session epoch through this path could only be a re-statement
    of the same instant, so the assertion is that a fresh card's own clock is
    untouched by the arrival of the feature.
    """
    now = [3_000.0]
    card = ToolCard("t", "bash", {"command": "ls"}, clock=lambda: now[0])
    assert card.started_at == 3_000.0
    now[0] += 2.0
    assert card._build_row(80).plain.rstrip().endswith("2s")


def test_a_cut_off_turn_names_its_cards_cut_off() -> None:
    """D2: the word changes, the glyph and the tier do not.

    ``interrupted`` is now reserved for a positively recorded stop, so a row
    stranded by a cut-off that still said ``interrupted ⊘`` re-stated the
    ambiguity the taxonomy removed — on the same screen as the ``✗ turn cut
    off`` notice that had just resolved it (design review round 1, D2).
    """
    card = ToolCard("t", "bash", {"command": "pytest tests/unit -q"})
    card.mark_interrupted(cut_off=True)
    row = card._build_row(100).plain
    assert "cut off" in row
    assert "interrupted" not in row
    # Same mark, same ink: only the word was the ambiguity, and a new glyph or
    # a colour would be new design surface for a finding this small.
    assert "⊘" in row

    deliberate = ToolCard("t", "bash", {"command": "pytest tests/unit -q"})
    deliberate.mark_interrupted()
    assert "interrupted" in deliberate._build_row(100).plain, "a stop still says so"


# --- partial results (a search budget cut the answer short) -----------------
#
# Design review round 1: D1 (the collapsed row could not show the state) and D2
# (the disclosure rode a line the card crops at every width). The texts below are
# the shipped shapes; `test_builtin_tools.py` asserts them against the real tools,
# so these tests are about where they LAND rather than about their wording.
#
# The disclosure the tools compose, in the form D2 measured: on a ripgrep walk
# cut, the header was 329 cells and the first words about truncation landed at
# cell 99 — past the 94-cell crop at 100 columns.

_PARTIAL_MATCHES = (
    "Partial results: 1 match(es) for 'needle_marker' of 1386 "
    "(use skip=200 for the next page) — the walk stopped at 30 s after 12 files; "
    "narrow path=<subdirectory> or include=<glob> to search further (ripgrep):\n"
    "src/a.py:1:needle_marker = 1\n"
    "src/b.py:2:needle_marker = 2"
)


def test_a_partial_result_is_marked_in_the_collapsed_row() -> None:
    """The row an operator scans must answer "is this answer whole?" (D1).

    Design review round 1 measured the alternative on the real card: nine cards
    in one frame, six of them truncated searches, every one painting `✓ 0.4s`
    beside its arguments — row-for-row identical to the complete search for the
    same pattern. The partial state now carries its own glyph, the theme's
    warning ink, and the result's lead line promoted into the row exactly as an
    error's cause is.
    """
    card = ToolCard("t", "grep", {"pattern": "needle_marker"})
    card.mark_done(_PARTIAL_MATCHES, {"partial": True, "partial_stops": []})
    row = card._build_row(100)

    assert ICON_PARTIAL in row.plain
    assert ICON_SUCCESS not in row.plain, "a partial answer must not paint as a complete one"
    # The claim itself survives the crop at every width: that is the half of the
    # state a glance has to be able to catch.
    assert "Partial" in row.plain
    assert _triplet(_style_at(row, ICON_PARTIAL).color) == _triplet(
        Style(color=theme_mod.semantic_color("warning")).color
    )
    _assert_fits(card)

    for width in WIDTHS:
        row = card._build_row(width)
        assert ICON_PARTIAL in row.plain, (width, row.plain)
        assert ICON_SUCCESS not in row.plain, (width, row.plain)
        _assert_fits(card)

    # The control: the same result WITHOUT the flag keeps the quiet ✓ row, so
    # the marker tracks the state and not the text.
    complete = ToolCard("t2", "grep", {"pattern": "needle_marker"})
    complete.mark_done(_PARTIAL_MATCHES.split(":")[0] + ":\nsrc/a.py:1:needle_marker = 1")
    assert ICON_SUCCESS in complete._build_row(100).plain


def test_a_partial_result_wraps_its_disclosure_in_the_body() -> None:
    """The claim and the remedy must both survive the expanded card (D2).

    The card paints a captured output line CROPPED to its measure — 94 cells at
    100 columns, 73 at 80, 54 at 60 — so the disclosure used to show its
    bookkeeping and lose what happened and what to do next. It is the promoted
    lead line instead, which wraps, and it paints in the state's ink rather than
    the captured rows' `dim` (D9: both sampled `#837C6D` on `#301E1A` before, so
    the frame carried no signal that the line was the state).
    """
    card = ToolCard("t", "grep", {"pattern": "needle_marker"})
    card.mark_done(_PARTIAL_MATCHES, {"partial": True})
    card.toggle_expanded()
    lines = card._build_content(80).plain.splitlines()
    body = _collapsed("\n".join(lines))

    assert "narrow path=<subdirectory> or include=<glob> to search further" in body
    assert "src/b.py:2:needle_marker = 2" in body, "the answer itself still paints"
    # Located by its lead rather than by index, and only the BLOCK is handed to
    # the shape helper: the captured rows below it keep the ordinary crop.
    rows = _reason_rows(_reason_block(lines))
    assert rows and rows[0].startswith("Partial results: 1 match(es)"), rows
    # `_reason_rows` asserted the lead itself (either state glyph, blanks after),
    # so the block is the partial card's promoted disclosure and not a captured
    # row that happens to start with a glyph.

    # The captured rows keep the one-line-per-row crop: the exemption is the
    # claimed lead, not the body.
    wide = ToolCard("t2", "grep", {"pattern": "n"})
    wide.mark_done(
        "Partial results: 1 match(es) for 'n' — the walk stopped at 30 s after 1 file;"
        " narrow path=<subdirectory> or include=<glob> to search further:\n" + "x" * 400,
        {"partial": True},
    )
    wide.toggle_expanded()
    lines = wide._build_content(80).plain.splitlines()
    assert all(cell_len(line) <= 80 for line in lines)
    assert not any(len(line.strip()) > 80 for line in lines)


def test_a_partial_flag_without_a_lead_line_promotes_nothing() -> None:
    """A flag alone must not promote a line that is not the disclosure.

    Both halves are required: ``details["partial"]`` says the RESULT is partial,
    the lead prefix says THIS LINE is the claim. A card whose payload predates
    the prefixes — or whose text leads with something else — keeps the ordinary
    crop rather than growing a synthetic row that restates nothing.
    """
    card = ToolCard("t", "grep", {"pattern": "needle"})
    card.mark_done("1 match(es) for 'needle':\nsrc/a.py:1:needle", {"partial": True})
    assert ICON_PARTIAL in card._build_row(100).plain, "the state mark is the flag's"
    card.toggle_expanded()
    body = card._build_content(80).plain
    # No promoted row: the summary is not claimed, so no `◐` lead and no wrap.
    assert ICON_PARTIAL not in body.splitlines()[2:][0]
