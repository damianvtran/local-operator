"""Minimalism regression — the density contract stays machine-checked.

- The tcss sheet carries NO literal hex: every color resolves through the
  brand-token CSS variables injected from ``tui/theme.py`` (single source).
- A tool execution is ONE row (the filled card); a finished card settles at
  exactly one row.
- The status line is ONE full-width band row.
"""

from __future__ import annotations

import re
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


def test_tcss_tool_card_and_status_band_are_single_rows() -> None:
    text = TCSS.read_text()
    tool_block = re.search(r"^ToolCard\s*\{([^}]*)\}", text, re.MULTILINE)
    band_block = re.search(r"^#status-band\s*\{([^}]*)\}", text, re.MULTILINE)
    assert tool_block is not None and "height: 1;" in tool_block.group(1)
    assert band_block is not None and "height: 1;" in band_block.group(1)


def test_tool_card_renders_a_single_row() -> None:
    """Built rows never contain a newline; a finished card settles at 1 row."""
    card = ToolCard("t1", "bash", {"command": "pytest tests -q"})
    card.mark_done()
    row = card._build_row(80)
    assert "\n" not in row.plain
    assert card.settled_rows() == 1

    card = ToolCard("t2", "grep", {"pattern": "needle"})
    card.mark_failed("permission denied while reading the file")
    row = card._build_row(80)
    assert "\n" not in row.plain
    assert card.settled_rows() == 1


def test_status_band_renders_a_single_row() -> None:
    """Segments join on one line with the right column right-aligned."""
    status = StatusLine(Static())
    status._model_label = "anthropic/claude-opus-4-5"
    status._cwd = "/opt/local-operator"
    status._context_tokens = 12400
    status._cost = "$0.0021"
    row = status.render_text(80)
    assert "\n" not in row.plain
    assert "12.4k tok" in row.plain
    assert "$0.0021" in row.plain
