"""Minimalism regression — the density contract stays machine-checked.

- The tcss sheet carries NO literal hex: every color resolves through the
  brand-token CSS variables injected from ``tui/theme.py`` (single source).
- A COLLAPSED tool execution is ONE row (the filled card); it only grows
  when the user clicks it open.
- The status line is ONE full-width band row.
- Separation is adaptive and opt-in: no block selector carries a uniform
  margin, and the only spacing declaration in the sheet is the class the
  transcript container applies deliberately.
"""

from __future__ import annotations

import re

from rich.cells import cell_len
from pathlib import Path

from local_operator.tui.widgets.status_line import StatusLine
from local_operator.tui.widgets.tool_card import ToolCard
from textual.widgets import Static

TCSS = Path(__file__).parent.parent.parent.parent / "local_operator" / "tui" / "local_operator.tcss"
_HEX_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def test_tcss_has_no_literal_hex() -> None:
    """Every color in the sheet rides a ``$lo-*`` token, never a raw hex."""
    text = TCSS.read_text()
    assert not _HEX_RE.search(text), f"literal hex in tcss: {_HEX_RE.findall(text)}"


def test_tcss_pins_one_row_for_tool_cards_and_the_status_band() -> None:
    """The collapsed card and the status band are both pinned to one row.

    Pinning is not decoration here. A card builds its first row before it is
    laid out, against a guessed width; under ``height: auto`` that first
    over-wide measurement resolves to TWO rows and never comes back down,
    turning a ledger of one-line traces into a double-spaced list. The pin
    makes the worst case a clipped cell for one frame instead.

    Exactly one selector may opt out: an EXPANDED card, whose whole purpose
    is to be taller than one row.
    """
    text = TCSS.read_text()
    tool_block = re.search(r"^ToolCard\s*\{([^}]*)\}", text, re.MULTILINE)
    expanded = re.search(r"^ToolCard\.tool-expanded\s*\{([^}]*)\}", text, re.MULTILINE)
    band_block = re.search(r"^#status-band\s*\{([^}]*)\}", text, re.MULTILINE)
    assert tool_block is not None and "height: 1;" in tool_block.group(1)
    assert expanded is not None and "height: auto;" in expanded.group(1)
    assert band_block is not None and "height: 1;" in band_block.group(1)


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
        r"AssistantBlock,\s*ToolCard\s*\{([^}]*)\}",
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
    """
    text = TCSS.read_text()
    gap = re.search(r"^\.gap-above\s*\{([^}]*)\}", text, re.MULTILINE)
    assert gap is not None, ".gap-above rule not found in tcss"
    assert "margin-top: 1;" in gap.group(1)
    # No other rule anywhere in the sheet declares a vertical margin.
    margins = re.findall(r"margin(?:-top|-bottom)?\s*:[^;]*;", text)
    assert margins == ["margin-top: 1;"], margins


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
