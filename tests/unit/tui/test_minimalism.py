"""Minimalism regression — the density contract stays machine-checked.

- The tcss sheet carries NO literal hex: every color resolves through the
  brand-token CSS variables injected from ``tui/theme.py`` (single source).
- A COLLAPSED tool execution is ONE row (the filled card); it only grows
  when the user opens it, by click or by Enter on the focused row.
- The status line is ONE full-width band row.
- Separation is adaptive and opt-in: no block selector carries a uniform
  margin, and the only spacing declaration in the sheet is the class the
  transcript container applies deliberately.
"""

from __future__ import annotations

import re
from pathlib import Path

from rich.cells import cell_len
from textual.widgets import Static

from local_operator.tui.widgets.status_line import StatusLine
from local_operator.tui.widgets.tool_card import ToolCard

TCSS = Path(__file__).parent.parent.parent.parent / "local_operator" / "tui" / "local_operator.tcss"
_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def test_tcss_has_no_literal_hex() -> None:
    """Every color in the sheet rides a ``$lo-*`` token, never a raw hex."""
    text = TCSS.read_text()
    assert not _HEX_RE.search(text), f"literal hex in tcss: {_HEX_RE.findall(text)}"


def test_tcss_pins_card_and_band_heights_rather_than_leaving_them_auto() -> None:
    """The collapsed card and the status band are both pinned, not ``auto``.

    Pinning is not decoration here. A card builds its first row before it is
    laid out, against a guessed width; under ``height: auto`` that first
    over-wide measurement resolves to TWO rows and never comes back down,
    turning a ledger of one-line traces into a double-spaced list. The pin
    makes the worst case a clipped cell for one frame instead.

    The band is TWO rows for one row of content: its top padding row is the gap
    that separates it from the input line above, and because the band and the
    input panel share one fill, a padded row is indistinguishable from a gap.

    Exactly one selector may opt out: an EXPANDED card, whose whole purpose is
    to be taller than one row.
    """
    text = TCSS.read_text()
    # ToolCard and WakeBlock share the pin: a wake receipt is a ledger row,
    # and leaving it on `auto` would be the same first-measurement-sticks-at-
    # two-rows failure the ToolCard pin exists to stop. Combined selectors
    # so the two cannot drift.
    tool_block = re.search(r"^ToolCard,\s*WakeBlock\s*\{([^}]*)\}", text, re.MULTILINE)
    expanded = re.search(
        r"^ToolCard\.tool-expanded,\s*WakeBlock\.wake-expanded\s*\{([^}]*)\}",
        text,
        re.MULTILINE,
    )
    band_block = re.search(r"^#status-band\s*\{([^}]*)\}", text, re.MULTILINE)
    assert tool_block is not None and "height: 1;" in tool_block.group(1)
    assert expanded is not None and "height: auto;" in expanded.group(1)
    assert band_block is not None and "height: 2;" in band_block.group(1)
    # One row of that height is the gap; the band itself still renders one row.
    assert "padding: 1 1 0 0;" in band_block.group(1)


def test_block_selectors_declare_no_margin_or_padding() -> None:
    """The tcss comment claims the density contract — the blocks own no outer
    margin/padding. A margin HERE would insert a blank filler row between
    every pair of blocks, the single largest violation of the minimalist
    mandate; adaptive spacing deliberately rides a separate opt-in class
    (``.gap-above``) that the container applies only where the rhythm needs
    it. Pin the base rule so the claim cannot silently rot."""
    text = TCSS.read_text()
    match = re.search(
        r"^TranscriptBlock,\s*UserBlock,\s*NoticeBlock,\s*RichBlock,\s*"
        r"AssistantBlock,\s*ToolCard,\s*WakeBlock\s*\{([^}]*)\}",
        text,
        re.MULTILINE,
    )
    assert match is not None, "block selectors rule not found in tcss"
    body = match.group(1)
    assert not re.search(r"\b(margin|padding)\s*:", body)


def test_gap_class_is_the_only_block_spacing_declaration() -> None:
    """Exactly one rule in the sheet may open a blank row between blocks.

    A second source of vertical spacing is how "adaptive" quietly decays
    back into "a blank row everywhere", so the count is pinned at one.

    It is also how a gap DOUBLES. Tool rows now take ``.gap-above``
    unconditionally (every action gets its own blank row), so any margin on
    a `ToolCard` selector would stack on top of it and put two blank rows
    between every pair of actions — the "too much spacing" complaint, moved
    from above the group to inside it.
    """
    text = TCSS.read_text()
    gap = re.search(r"^\.gap-above\s*\{([^}]*)\}", text, re.MULTILINE)
    assert gap is not None, ".gap-above rule not found in tcss"
    assert "margin-top: 1;" in gap.group(1)
    # No other rule anywhere in the sheet declares a vertical margin.
    margins = re.findall(r"margin(?:-top|-bottom)?\s*:[^;]*;", text)
    assert margins == ["margin-top: 1;"], margins
    # And no ToolCard rule declares spacing of ANY kind, margin or padding:
    # the card's own 1-cell inner padding is drawn by the row builder, not by
    # the sheet, precisely so it cannot become a vertical row.
    for block in re.findall(r"^(?:ToolCard|WakeBlock)[^{]*\{([^}]*)\}", text, re.MULTILINE):
        assert not re.search(r"\b(margin|padding)\s*:", block), block


def test_tool_card_renders_a_single_row() -> None:
    """Built rows never contain a newline; a finished card settles at 1 row."""
    card = ToolCard("t1", "bash", {"command": "pytest tests -q"})
    card.mark_done()
    row = card._build_row(80)
    assert "\n" not in row.plain
    assert card.settled_rows() == 1
    # A single-row card that OVERFLOWS and gets clipped is not "one line":
    # assert the row actually fits its width across the usable range.
    for w in (16, 20, 40, 80, 200):
        assert cell_len(card._build_row(w).plain) <= w

    card = ToolCard("t2", "grep", {"pattern": "needle"})
    card.mark_failed("permission denied while reading the file")
    row = card._build_row(80)
    assert "\n" not in row.plain
    assert card.settled_rows() == 1
    for w in (16, 20, 40, 80, 200):
        assert cell_len(card._build_row(w).plain) <= w


def test_status_band_renders_a_single_row() -> None:
    """Segments join on one line with the right column right-aligned."""
    status = StatusLine(Static())
    status._model_label = "anthropic/claude-opus-4-5"
    status._cwd = "/opt/local-operator"
    status._context_tokens = 12400
    status._cost = "$0.0021"
    row = status.render_text(80)
    assert "\n" not in row.plain
    # No context window was set, so the spend reports against an explicit
    # unknown denominator rather than inventing a percentage.
    assert "12.4k/—" in row.plain
    assert "$0.0021" in row.plain
    assert cell_len(row.plain) <= 80


def test_a_wrapped_row_never_overhangs_its_frame_even_on_a_cluster() -> None:
    """Per-character widths do not add up for a grapheme CLUSTER.

    `cell_len` counts `1️⃣` (digit + VS16 + keycap) as 2 when handed the whole
    string and as 1+0+0 per character, so a row built from a running per-character
    sum overhung its frame by one cell. Every row is measured whole here, which is
    the only measurement that agrees with what the terminal paints.

    The single-character exception is deliberate and separately documented: a
    frame narrower than one ideograph cannot hold it, and taking nothing looped
    forever.
    """
    from local_operator.tui.widgets.transcript import wrap_cells

    keycap = "qqqqqq1\ufe0f\u20e3www"
    samples = [
        keycap,
        "x" * 40,
        "\u65e5\u672c\u8a9e" * 6,
        "a\u200db" * 8,
        "ctrl+c again to exit - resume with: local-operator --resume fd5a66ef8ce2",
    ]
    for text in samples:
        for width in (1, 2, 5, 10, 20, 34):
            for row in wrap_cells(text, width):
                if len(row) <= 1:
                    continue  # the documented one-character overhang
                assert cell_len(row) <= width, (text, width, row, cell_len(row))

    # And the cluster is not silently dropped: it moves to the next row whole.
    assert "".join(wrap_cells(keycap, 10)) == keycap
