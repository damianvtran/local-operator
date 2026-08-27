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

The gate itself (:func:`nerd_icons_enabled`) is TRI-STATE. A renderer cannot
interactively probe a terminal for glyph COVERAGE — a DA/XTGETTCAP/OSC query
needs its reply read off stdin, and stdin belongs to Textual's input loop
while the app runs (see :func:`notify.detect_protocol` for the same
constraint). But it CAN read the environment markers an emulator injects and
name the emulator, exactly as :func:`notify.detect_protocol` and
:func:`images.detect_mode` do, and the emulators that bundle a Nerd symbol
fallback font are a known, enumerable set. So the gate resolves in order:

1. env kill switch ``LOCAL_OPERATOR_NO_NERD_ICONS`` -> off (CI, snapshots);
2. an EXPLICIT ``display.nerd_icons`` bool in config -> honoured both ways
   (True lets a user who installed a patched font force glyphs on even in an
   unrecognised terminal; False forces them off);
3. unset (the default, stored as ``None`` = auto) -> :func:`_nerd_capable_terminal`
   decides from the env markers.

The default FLIPPED from unconditional-on to auto because unconditional-on
handed every macOS Terminal.app user a row of tofu boxes with no recourse
short of editing config: Apple_Terminal ships no Nerd symbol fallback, so the
Font Awesome PUA codepoints below render as replacement squares. Autodetect
returns True only for terminals CONFIRMED to carry a symbol fallback font;
everything else gets the plain table, because a plain ASCII icon is strictly
better than a tofu box.
"""

from __future__ import annotations

import os
from typing import Mapping

from rich.cells import cell_len

from local_operator.tui.settings import settings_get

#: The type the detection helper accepts, mirroring ``notify.EnvMap``. A plain
#: dict satisfies it, which is what lets tests inject an environment without
#: mutating the process's real ``os.environ``.
EnvMap = Mapping[str, str]

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
    # nf-fa-tasks, NOT nf-fa-check_square_o. Every other glyph in this table
    # is a noun — terminal, file, pencil, folder, magnifier, clock, globe —
    # and a check mark is a verdict: it is the exact mark the row prints at
    # its right edge for "succeeded", so a todo row opened and closed with a
    # check and a colourless frame could not tell the two apart.
    "todo": "\uf0ae",
    "wake": "\uf017",  # nf-fa-clock_o
    "list_variables": "\uf0ca",  # nf-fa-list_ul
    "read_variable": "\uf02b",  # nf-fa-tag
    "browser": "\uf0ac",  # nf-fa-globe
    "web_search": "\uf0ac",  # nf-fa-globe
    "web_fetch": "\uf019",  # nf-fa-download — a page pulled down over the wire
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
    # `=` (algebraic assignment), NOT `x`: the earlier `x` shared its SHAPE
    # with the `✗` failure verdict the status column prints, the one plain
    # glyph echoing an outcome mark. `=` says "a value bound to a name" with
    # no verdict collision, and stays a single Latin-1 cell.
    "list_variables": "=",
    "read_variable": "=",
    "browser": "@",  # the URL sigil
    "web_search": "?",  # a search query, in the verified ASCII fallback repertoire
    "web_fetch": "\u2193",  # a downward arrow: content pulled down from a URL
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

    The row builder reserves exactly one cell for the icon. A glyph that measures
    two would not merely look wrong — it would shift the summary budget and the
    right-aligned status column by a cell, which is the one class of bug the "one
    width model" rule exists to prevent. So width is checked here, once, against
    the same ``cell_len`` the row math uses.

    What this canNOT do is detect a MISSING glyph. Every Nerd Font codepoint in
    this module is East-Asian-Width Ambiguous, so `cell_len` reports 1 for all of
    them whatever the terminal actually has installed, and the fallback branch is
    unreachable in practice. It is kept because it is the correct guard for the
    thing it does check — a future icon from a Wide block would be caught — and
    because the honest alternative, probing the terminal for glyph coverage, is
    not something a renderer can do. The plain table remains the answer for a
    host without the font, selected by configuration rather than by measurement.
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


def _nerd_capable_terminal(env: EnvMap | None = None) -> bool:
    """Whether ``env``'s terminal is CONFIRMED to render Nerd Font PUA glyphs.

    True only for emulators that ship a Nerd symbol fallback font, so the
    Font Awesome private-use codepoints resolve to real glyphs instead of
    tofu. Detection is from the marker each emulator injects, the same
    technique as :func:`notify.detect_protocol` — reused here rather than
    imported to keep the modules decoupled (glyphs is a leaf; a cross-import
    would couple it to the notification stack).

    Note on ``TERM``: this deliberately does NOT gate on ``TERM`` being
    non-``dumb``. cmux embeds ghostty and sets ``TERM=dumb`` while still
    injecting ``GHOSTTY_*`` and drawing Nerd glyphs fine, so a positive
    emulator marker wins over a ``dumb`` TERM. Anything without a positive
    marker (Apple_Terminal, plain xterm, an ssh into a minimal image, an
    unknown emulator) is False: when in doubt, plain beats tofu.
    """
    source = os.environ if env is None else env
    # ghostty (and cmux, which embeds it) — bundles JetBrainsMono Nerd Font.
    if source.get("GHOSTTY_RESOURCES_DIR") or source.get("GHOSTTY_BIN"):
        return True
    # kitty — bundles Symbols Nerd Font as its symbol_map fallback. ``startswith``
    # rather than equality so TERM variants (``xterm-kitty-direct``) still match,
    # matching notify.detect_protocol's kitty check exactly.
    if source.get("KITTY_WINDOW_ID") or source.get("TERM", "").startswith("xterm-kitty"):
        return True
    # wezterm — ships Nerd Font Symbols as a default fallback font.
    if source.get("WEZTERM_PANE") or source.get("WEZTERM_EXECUTABLE"):
        return True
    # iTerm2 (iterm.app) and Warp (warpterminal) are deliberately OMITTED from
    # this set even though notify.detect_protocol lists them: neither bundles a
    # Nerd symbol fallback font, so the PUA codepoints would tofu there. A
    # patched-font iTerm user opts in via the explicit ``display.nerd_icons``
    # config flag rather than being auto-enabled into replacement boxes.
    term_program = source.get("TERM_PROGRAM", "").lower()
    if term_program in ("ghostty", "wezterm"):
        return True
    return False


def nerd_icons_enabled() -> bool:
    """Whether Nerd Font glyphs may be drawn — tri-state, see module docstring.

    Order: env kill switch, then an explicit ``display.nerd_icons`` bool
    (honoured both ways), then marker-based autodetection when the flag is
    unset (stored as ``None`` = auto). False means the terminal cannot be
    trusted with the private use area: an unpatched font renders those
    codepoints as a replacement box, strictly worse than the plain glyph.
    """
    if os.environ.get(_ENV_DISABLE):
        return False
    # None => the key is absent from config: fall through to autodetection.
    # An explicit True/False is the user's override and wins both ways.
    flag = settings_get("display.nerd_icons", None)
    if flag is not None:
        return bool(flag)
    return _nerd_capable_terminal()


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


def display_name(tool_name: str) -> str:
    """What the row's NAME column should say for ``tool_name``.

    Builtins are already their own best name and pass straight through. MCP
    tools do not: ``create_mcp_tool_name`` mints ``mcp__<server>_<tool>``,
    and in an 8-cell column that constant ``mcp__`` eats five cells before a
    single informative character — three tools from one Linear server all
    rendered ``mcp__lin``, which is the ledger's whole scan-by-shape premise
    failing for the tool class a user is most likely to have a dozen of. The
    plug icon already says "this came from a server"; the name column
    repeating that is the waste.

    So the prefix goes and the server segment goes with it, leaving the CALL:
    ``mcp__linear_create_issue`` -> ``create_issue``, which the column then
    truncates to ``create_i`` — distinct from ``list_iss`` and ``get_issu``,
    which is the discrimination that was missing.

    Two accepted imprecisions, both preferable to what they replace:

    - A server whose own name contains an underscore cannot be split back out
      of the minted string (``mcp__my_server_do_thing`` is genuinely
      ambiguous), so only the FIRST segment is treated as the server. The
      remainder is still the call's own identifier, never the constant.
    - Two servers exposing the same tool name still collide
      (``mcp__github_create_issue`` and ``mcp__gitlab_create_issue`` both read
      ``create_i``). Eight cells cannot hold both halves, and between the two
      the verb is what changes down a run of rows; the summary carries the
      arguments that say which repository.

    Never returns empty: a name that is nothing but the prefix and a server
    keeps whatever it had, because a blank name column is worse than a
    repetitive one.
    """
    name = tool_name.strip()
    if not name.lower().startswith(MCP_NAME_PREFIX):
        return name
    remainder = name[len(MCP_NAME_PREFIX) :]
    _server, separator, call = remainder.partition("_")
    if separator and call:
        return call
    return remainder or name
