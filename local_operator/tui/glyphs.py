"""Per-tool icons and the terminal capability gate that guards them.

Two tables, ONE lookup. A tool row leads with an icon so a ledger of a dozen
actions can be scanned by shape before it is read by name — the eye finds the
one `edit` in a run of `read`s without parsing ten words. That only works if
the glyph is actually drawn, so the table exists twice:

- :data:`NERD_TOOL_ICONS` — Nerd Font glyphs, taken exclusively from the
  Font Awesome block (U+F000-U+F2E0) of the Nerd Fonts private use area.
  That block is the one region present in BOTH Nerd Fonts v2 and v3 at the
  same codepoints, and the one every patched font in the wild carries; the
  v3-only Material Design range lives on plane 1 (U+F0001+) where terminals
  disagree about width, and the Devicons/Codicons blocks moved between
  releases. Ghostty draws this block at a single cell.
- :data:`PLAIN_TOOL_ICONS` — the fallback drawn when Nerd glyphs are gated
  off. Restricted to Latin-1 and WGL4 characters, which is the repertoire a
  bare xterm, a TTY console and an SSH session into a minimal image can all
  be relied on to have. The set is deliberately SMALLER than the tool list:
  it distinguishes *categories* (shell, read, mutate, search, meta) rather
  than individual tools, because the safe single-cell repertoire runs out
  long before the tool list does and the tool NAME is one column to the
  right. Sharing a fallback glyph costs nothing; a tofu box costs the row.

Every glyph is measured through :func:`rich.cells.cell_len` at import, and
any entry that does not measure exactly one cell is replaced by its plain
counterpart. The tool row's width arithmetic budgets the icon at one cell;
a two-cell glyph would push the right-aligned status column off the card,
so the measurement is a hard gate rather than a warning.

The gate itself (:func:`nerd_icons_enabled`) mirrors ``shimmer_enabled``:
an environment kill switch for CI and for terminals without a patched font,
and a ``display.nerd_icons`` config flag for a persistent preference.
"""

from __future__ import annotations

import os

from rich.cells import cell_len

from local_operator.tui.settings import settings_get

#: Environment kill switch — a terminal without a patched font, or a
#: snapshot harness that wants a stable ASCII-ish frame.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_NERD_ICONS"

#: Prefix every MCP tool name carries (``mcp__<server>_<tool>``, minted by
#: ``local_operator.mcp.tool_bridge.create_mcp_tool_name``). Matched by
#: prefix rather than enumerated: the tool set is whatever servers the user
#: configured, so it cannot be known here.
MCP_NAME_PREFIX = "mcp__"

#: Nerd Font glyph per builtin tool. Keys are the harness-visible tool names
#: (``local_operator.tools.builtin``), lowercased.
NERD_TOOL_ICONS: dict[str, str] = {
    "bash": "\uf120",  # nf-fa-terminal
    "read": "\uf15c",  # nf-fa-file_text
    "write": "\uf040",  # nf-fa-pencil
    "edit": "\uf044",  # nf-fa-pencil_square_o
    "glob": "\uf115",  # nf-fa-folder_open_o
    "grep": "\uf002",  # nf-fa-search
    "todo": "\uf046",  # nf-fa-check_square_o
    "wake": "\uf017",  # nf-fa-clock_o
    "list_variables": "\uf0ca",  # nf-fa-list_ul
    "read_variable": "\uf02b",  # nf-fa-tag
    "browser": "\uf0ac",  # nf-fa-globe
    "task": "\uf0c0",  # nf-fa-users — work handed to another agent
    "agent": "\uf0c0",
}

#: Nerd Font glyph for any ``mcp__*`` tool: a plug, because what the row is
#: reporting is not the tool's own semantics but that it came from a server.
NERD_ICON_MCP = "\uf1e6"  # nf-fa-plug
#: Nerd Font glyph for a tool this table has never heard of.
NERD_ICON_DEFAULT = "\uf0ad"  # nf-fa-wrench

#: Plain-unicode fallbacks. See the module docstring for why several tools
#: share one: `write`/`edit` are both "a pencil", and both variable tools are
#: both "the algebraic unknown".
PLAIN_TOOL_ICONS: dict[str, str] = {
    "bash": "$",  # the shell prompt sigil
    "read": "≡",  # lines of text
    "write": "+",  # content that did not exist before
    "edit": "±",  # content changed in both directions (cf. the +N/-N counters)
    "glob": "*",  # the wildcard itself
    "grep": "/",  # the /pattern/ sigil
    "todo": "▪",  # one item in a list
    "wake": "○",  # a clock face
    "list_variables": "x",  # the algebraic unknown
    "read_variable": "x",
    "browser": "@",  # the URL sigil
    "task": "»",  # work passed onward
    "agent": "»",
}

#: Plain fallback for ``mcp__*`` — a discrete module docked onto the harness.
PLAIN_ICON_MCP = "◆"
#: Plain fallback for an unknown tool. This is the marker every tool row
#: carried before the icon table existed, so the degraded frame is exactly
#: the frame this app has always shipped.
PLAIN_ICON_DEFAULT = "▸"


def _single_cell(glyph: str, fallback: str) -> str:
    """``glyph`` when it measures one cell, otherwise ``fallback``.

    The row builder reserves exactly one cell for the icon. A glyph that
    measures two would not merely look wrong — it would shift the summary
    budget and the right-aligned status column by a cell, which is the one
    class of bug the "one width model" rule exists to prevent. So width is
    checked here, once, against the same ``cell_len`` the row math uses.
    """
    return glyph if cell_len(glyph) == 1 else fallback


#: The Nerd table after the width gate. Built once: the codepoints are
#: constants and ``cell_len`` is a pure function of them, so re-measuring on
#: every row repaint would buy nothing.
_SAFE_NERD_ICONS: dict[str, str] = {
    name: _single_cell(glyph, PLAIN_TOOL_ICONS.get(name, PLAIN_ICON_DEFAULT))
    for name, glyph in NERD_TOOL_ICONS.items()
}
_SAFE_NERD_MCP = _single_cell(NERD_ICON_MCP, PLAIN_ICON_MCP)
_SAFE_NERD_DEFAULT = _single_cell(NERD_ICON_DEFAULT, PLAIN_ICON_DEFAULT)


def nerd_icons_enabled() -> bool:
    """Whether Nerd Font glyphs may be drawn (env kill switch + settings flag).

    False means the terminal cannot be trusted with the private use area —
    an unpatched font renders those codepoints as a replacement box, which is
    strictly worse than the plain glyph it would have shown instead.
    """
    if os.environ.get(_ENV_DISABLE):
        return False
    return bool(settings_get("display.nerd_icons", True))


def tool_icon(tool_name: str) -> str:
    """The one-cell icon leading ``tool_name``'s row.

    Resolution order: exact builtin name, then the ``mcp__`` prefix, then the
    generic fallback. Lookup is case-insensitive because ``tool_name`` is
    MODEL-controlled — a provider that echoes ``Bash`` back must not silently
    drop to the wrench.
    """
    name = tool_name.strip().lower()
    if nerd_icons_enabled():
        icon = _SAFE_NERD_ICONS.get(name)
        if icon is not None:
            return icon
        return _SAFE_NERD_MCP if name.startswith(MCP_NAME_PREFIX) else _SAFE_NERD_DEFAULT
    icon = PLAIN_TOOL_ICONS.get(name)
    if icon is not None:
        return icon
    return PLAIN_ICON_MCP if name.startswith(MCP_NAME_PREFIX) else PLAIN_ICON_DEFAULT
