"""New-session welcome view — the transcript's EMPTY STATE, not a banner.

A fresh session (and every ``/clear``) shows one horizontally centered,
BORDERLESS block resting on the input card:

    ██      ▄████▄
    ██      ██  ██
    ██      ██  ██
    ██████  ▀████▀

    l o c a l   o p e r a t o r

           v0.15.10
    openrouter/deepseek/deepseek-chat
           ~/local-operator
      ! not logged in — /login openrouter

    /         command picker
    /help     all commands
    ctrl+d    quit

Three things make this a design decision rather than a splash screen:

- **No box.** The mandate forbids bordered chrome; the omp reference draws a
  two-column bordered box appended to the transcript, and this deliberately
  deviates because the product owner asked for a *centered* view and a border
  would violate the density contract. Structure comes from the logo lockup,
  the tint ramp, and one blank row between sections.
- **No accent.** The one green is reserved for the running indicator, links,
  the focused chevron, and the command picker's selected row. The wordmark is
  the brightest thing here (``fg``) precisely because it is a single row; the
  four-row mark sits a step back at ``muted`` so a boot frame is not a wall
  of bright blocks.
- **It degrades in a fixed order.** Terminals are not a fixed canvas, so the
  view sheds decoration before information: the logo goes first, the hints
  second, the status rows last, and the credential warning never. See
  :func:`build_welcome_lines`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.geometry import Size
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import format_model_label
from local_operator.tui.widgets.transcript import NOTICE_GLYPHS

#: The real local-operator mark, downsampled from the shipped raster asset
#: (``static/local-operator-icon-2-light-clear.png``, also the app icon in
#: local-operator-ui) rather than hand-drawn. It depicts a FIGURE WITH A RAISED
#: HAND that doubles as a node graph: three open rings — a large head and two
#: smaller nodes — an arm sweeping from the shoulder up to the top-right node,
#: and two stems of deliberately unequal length, the left one ending in a node
#: and the right running unbroken to the baseline. That asymmetry is what makes
#: it read as a person rather than an abstract cluster, so it must survive any
#: future resizing.
#:
#: Only the three universal block-element glyphs are used (█ ▀ ▄, all in
#: U+258x): half blocks give each cell two square subcells, which is what lets
#: a 2:1 character cell carry round shapes at all, and they need neither
#: box-drawing nor braille coverage — so the mark survives a bare xterm font.
#:
#: TEN rows. Eight was the smallest size at which the shapes were technically
#: distinguishable, and it looked it: the head quantised to a blobby rectangle
#: with a one-subcell counter and both small rings closed into squares. Twelve
#: is marginally crisper still but does not fit the splash region of a 28-row
#: terminal, and a mark that is usually absent is worse than one that is
#: slightly softer. At ten every ring reads as a ring, the arm is a single
#: sweeping stroke rather than a staircase, and the strokes hold an even weight.
LOGO_MARK: tuple[str, ...] = (
    "     ▄█████▄    ▄▄██▄▄",
    "    ██     ██   ██  ██",
    "    ██    ▄█▀    ▀██▀ ",
    "    ▄██████▄     ██   ",
    "  ▄█▀▀     ▀█████▀    ",
    "  ██        ██        ",
    "  ██        ██        ",
    "▄▄██▄       ██        ",
    "██ ▄█       ██        ",
    " ▀▀▀        ██        ",
)

#: The wordmark, in the product's own lowercase voice.
WORDMARK = "local operator"

#: Letterspaced wordmark for the wide lockup. ``" ".join`` over the characters
#: turns the word gap into three cells, which is exactly the spacing a
#: letterspaced wordmark wants — no separate word-gap constant to keep in sync.
WORDMARK_SPACED = " ".join(WORDMARK)

#: Cell width of the mark. Deliberately NOT equal to the plain wordmark's width
#: any more: the old hand-drawn monogram happened to be 14 cells like
#: ``WORDMARK``, and a test asserted that coincidence as though it were the
#: lockup's contract. The real mark's aspect ratio fixes its width at 15 for
#: eight rows, and distorting it to match a string length would turn every ring
#: into an ellipse.
MARK_WIDTH = max(cell_len(row) for row in LOGO_MARK)

#: Width at or above which the full lockup (mark over the letterspaced
#: wordmark) is drawn.
LOGO_FULL_MIN_WIDTH = cell_len(WORDMARK_SPACED)

#: Placeholder while the session factory is still being awaited. The band uses
#: the same word for the same state, so the two never disagree on screen.
MODEL_PENDING = "connecting…"

#: The few affordances a first-time user actually needs. ``/`` and ``/help``
#: are kept separate on purpose: one is the inline picker, the other prints the
#: full two-column list — a user who has met neither cannot infer the other.
HINTS: tuple[tuple[str, str], ...] = (
    ("/", "command picker"),
    ("/help", "all commands"),
    ("ctrl+d", "quit"),
)

#: Key column width for the hint rows: the roomy default, and the squeezed
#: fallback of "longest key plus one space". A narrow terminal drops to the
#: tight column, and then to keys only, rather than letting the DESCRIPTIONS
#: truncate — `command pi…` teaches nothing, while a bare `/` at least still
#: names the affordance. See :func:`_hint_lines`.
HINT_KEY_WIDTH = 10
HINT_KEY_WIDTH_TIGHT = max(cell_len(key) for key, _ in HINTS) + 1

#: Warning body without its remedy, for widths that cannot hold the full
#: `— /login <provider>` tail. A half-printed command is worse than none: the
#: FACT is what must survive, the remedy is what may be dropped.
WARNING_SHORT = "not logged in"

#: Drop priorities for the status rows. Rows are shed lowest-first when the
#: terminal is too short for all of them, so the warning — the only row that
#: changes what the user must DO next — is the last one standing.
_PRIORITY_VERSION = 0
_PRIORITY_CWD = 1
_PRIORITY_MODEL = 2
_PRIORITY_WARNING = 3


@dataclass(frozen=True)
class WelcomeInfo:
    """The session facts the view reports. Frozen so the widget can compare
    two snapshots with ``==`` and repaint only on a real change."""

    version: str = ""
    model_label: str = ""
    cwd: str = ""
    #: Provider id with no stored credential, or ``None``. Populated only when
    #: the answer is *known* to be "no credential" — see
    #: :func:`session_welcome_info`.
    missing_credential: str | None = None


def app_version() -> str:
    """Installed distribution version, or ``""`` when it cannot be read.

    Running from a source tree without an installed distribution is normal
    during development, and a missing version is not worth a traceback on the
    first frame — the row is simply omitted.
    """
    try:
        return package_version("local-operator")
    except PackageNotFoundError:
        return ""


def session_welcome_info(session: Any | None, providers: Any | None) -> WelcomeInfo:
    """Snapshot the facts the welcome view shows.

    Takes the session and provider facade rather than the app so the whole
    gathering step lives here instead of leaking into ``app.py``.

    The credential question is ``is_usable``, not ``has_any_credential``. A key in
    the ENVIRONMENT is what the stream-time cascade resolves, so a session started
    that way runs perfectly — and the narrower check told those users "not logged in
    — /login openrouter" on the first screen, pointing them at a login they do not
    need and cannot usefully perform.

    Both reads are defended. ``model_label`` touches a session that may be
    mid-teardown, and the credential check touches the store on disk; either raising
    here would take down the app's very first render. A failed read degrades to *no
    warning* rather than to a false alarm: telling a correctly configured user they
    are logged out is worse than staying quiet.
    """
    label = ""
    if session is not None:
        try:
            label = session.model_label or ""
        except Exception:
            label = ""
    # The provider is the first segment of the model label, the same convention
    # /model uses to detect a provider switch.
    provider = label.partition("/")[0]
    missing: str | None = None
    if provider and providers is not None:
        try:
            if not providers.is_usable(provider):
                missing = provider
        except Exception:
            missing = None
    return WelcomeInfo(
        version=app_version(),
        model_label=label,
        cwd=os.getcwd(),
        missing_credential=missing,
    )


def _shorten_home(path: str) -> str:
    """Collapse ``$HOME`` to ``~`` (the shell's own shorthand for the same)."""
    home = str(Path.home())
    if path == home:
        return "~"
    prefix = home + os.sep
    if path.startswith(prefix):
        return "~" + os.sep + path[len(prefix) :]
    return path


def _fit_tail(text: str, width: int) -> str:
    """Truncate from the LEFT, keeping the tail.

    Only used for the working directory: the leaf directory is the part that
    identifies "where am I", so a path too long for the terminal must lose its
    root, not its name.
    """
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    kept = text
    while kept and cell_len(kept) > width - 1:
        kept = kept[1:]
    return "…" + kept


def _center(line: Text, width: int) -> Text:
    """Left-pad ``line`` so it sits centered in ``width``."""
    pad = (width - cell_len(line.plain)) // 2
    if pad <= 0:
        return line
    out = Text(" " * pad, no_wrap=True)
    out.append_text(line)
    return out


def _center_blocks(groups: list[list[Text]], width: int) -> list[list[Text]]:
    """Center SEVERAL groups on one shared left edge, block by block.

    Each group keeps its own internal alignment; what is shared is the pad, taken
    from the widest line across all of them. Centring each line on its own width
    produced a diamond of four ragged edges — un-designed, on the first frame
    every user sees — and centring each BLOCK on its own width merely moved the
    problem up a level: the status stack and the hint stack landed on different
    columns, and the offset even changed sign with the length of the model label.

    It also removes a visible twitch: ``model_label`` starts as ``connecting…``
    (12 cells) and resolves to as many as 38, so the row used to re-centre and
    re-widen a second into every boot. The pad is set by the widest line, so the
    stack holds still.
    """
    lines = [line for group in groups for line in group]
    if not lines:
        return groups
    block = max(cell_len(line.plain) for line in lines)
    pad = max(0, (width - block) // 2)
    if pad == 0:
        return groups
    out: list[list[Text]] = []
    for group in groups:
        padded_group: list[Text] = []
        for line in group:
            padded = Text(" " * pad, no_wrap=True)
            padded.append_text(line)
            padded_group.append(padded)
        out.append(padded_group)
    return out


def _logo_lines(width: int, *, flush: bool = False) -> list[Text]:
    """The logo lockup at whichever width tier ``width`` allows.

    - ``>= LOGO_FULL_MIN_WIDTH`` (27): the mark, a blank row, then the
      letterspaced wordmark — a small dense mark over a wide open name.
    - ``>= MARK_WIDTH``: the mark directly over the plain wordmark, flush,
      because a lockup without the width to be loose should not pretend.
    - narrower: the plain wordmark alone, which the caller then truncates.

    ``flush`` drops the breathing row in the wide tier. It is a HEIGHT
    concession, not a width one: on a short terminal the choice is between a
    tighter lockup and no mark at all, and one row of air is not worth the
    product's own identity. The caller escalates to it before shedding sections.
    """
    # `dim`, not `muted`. The intended hierarchy is a compact mark UNDER a wide
    # open name, but the mark is ten rows of solid block glyphs against a
    # wordmark of one row, so one ramp step could not overcome the area
    # difference and the eye landed on the blocks first. Two steps down makes
    # them read as a watermark behind the name.
    mark_style = Style(color=theme_mod.semantic_color("dim"))
    word_style = Style(color=theme_mod.semantic_color("fg"))
    mark = [_center(Text(row, style=mark_style, no_wrap=True), width) for row in LOGO_MARK]
    if width >= LOGO_FULL_MIN_WIDTH:
        spaced = _center(Text(WORDMARK_SPACED, style=word_style, no_wrap=True), width)
        return [*mark, spaced] if flush else [*mark, Text(""), spaced]
    plain = _center(Text(WORDMARK, style=word_style, no_wrap=True), width)
    if width >= MARK_WIDTH:
        return [*mark, plain]
    return [plain]


def _status_rows(info: WelcomeInfo, width: int) -> list[tuple[int, Text]]:
    """Status rows as ``(drop priority, line)``, in render order.

    Values carry no labels: a leading ``v``, a ``/``-separated spec and a
    leading ``~/`` each say what they are, and a label column here would
    duplicate the hints' two-column shape one blank row below.
    """
    dim = Style(color=theme_mod.semantic_color("dim"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    warn = Style(color=theme_mod.semantic_color("warning"))

    rows: list[tuple[int, Text]] = []
    if info.version:
        rows.append((_PRIORITY_VERSION, Text(f"v{info.version}", style=dim, no_wrap=True)))
    # Always drawn, placeholder and all: a row that appears the instant the
    # session boots would shift the whole block a line while the user reads it.
    #
    # When the full label does not fit, reduce it the way the STATUS BAND does
    # (keep the bare model id, drop the provider) rather than letting the final
    # truncation pass keep the head. The two disagreed: the splash printed
    # `openrouter/deepseek/deepseek-…` while the band six rows below printed
    # `deepseek-chat-v3.1`, so one app answered "which model" with opposite
    # halves of the same string (D10).
    label = info.model_label or MODEL_PENDING
    if cell_len(label) > width:
        label = format_model_label(label, short=True)
    rows.append((_PRIORITY_MODEL, Text(label, style=muted, no_wrap=True)))
    if info.cwd:
        shown = _fit_tail(_shorten_home(info.cwd), width)
        rows.append((_PRIORITY_CWD, Text(shown, style=dim, no_wrap=True)))
    if info.missing_credential:
        # The single most common first-run failure, so it is spelled as the
        # command that fixes it. `!` is the app's warning glyph (D14). When the
        # remedy does not fit, the fact is kept and the command dropped whole —
        # a half-printed `/logi…` is an instruction the user cannot follow.
        glyph = NOTICE_GLYPHS["warning"]
        body = f"{glyph} {WARNING_SHORT} — /login {info.missing_credential}"
        if cell_len(body) > width:
            body = f"{glyph} {WARNING_SHORT}"
        rows.append((_PRIORITY_WARNING, Text(body, style=warn, no_wrap=True)))
    return rows


def _hint_lines(width: int) -> list[Text]:
    """Hint rows, left-aligned to a shared key column, block-centered.

    Centering each row independently would ragged the key column; the rows are
    a table, so the TABLE is what gets centered.

    Three width tiers, because the alternative — letting the final truncation
    pass eat the descriptions — turns "command picker" into "command pi…",
    which costs a row and teaches nothing:

    1. the roomy key column,
    2. the tight key column (longest key plus one space),
    3. keys only, which still names every affordance the user can try.
    """
    # The same tint pair the PICKER uses for name/description (fg over muted),
    # not a step quieter. These rows are a preview of the picker — one of them
    # literally says "/  command picker" — and rendering the identical
    # key-then-description shape a full ramp step apart three rows away read as
    # two products' help text pasted together (D13).
    key_style = Style(color=theme_mod.semantic_color("fg"))
    desc_style = Style(color=theme_mod.semantic_color("muted"))

    key_column = 0
    for candidate in (HINT_KEY_WIDTH, HINT_KEY_WIDTH_TIGHT):
        block = max(candidate + cell_len(desc) for _, desc in HINTS)
        if block <= width:
            key_column = candidate
            break
    # UNPADDED, on purpose. The caller pads this block and the status block from
    # one shared width, so the splash has a single left edge below the wordmark.
    # Each block centring on its own widest member put the two on different
    # columns, and the offset changed sign with the length of the model label —
    # the same ragged-edge effect that was removed from inside the status stack,
    # surviving one level up.
    if not key_column:
        return [Text(key, style=key_style, no_wrap=True) for key, _ in HINTS]

    lines: list[Text] = []
    for key, desc in HINTS:
        line = Text(no_wrap=True)
        line.append(key.ljust(key_column), style=key_style)
        line.append(desc, style=desc_style)
        lines.append(line)
    return lines


def build_welcome_lines(info: WelcomeInfo, width: int, height: int) -> list[Text]:
    """Render the welcome block as exactly the lines it occupies.

    Pure, so the geometry is testable without a running app. Returns at most
    ``height`` lines, none wider than ``width``, and NO padding rows: the block
    starts on its first line and ends on its last.

    ``height`` is a row BUDGET, not a canvas to fill. VERTICAL PLACEMENT IS NOT
    THIS FUNCTION'S — the boot layout rests this block on the input card
    (``Screen.boot TranscriptView { align-vertical: bottom }`` in the tcss), and
    an alignment can only place a block that reports its true size. The top pad
    this used to compute was a second opinion about the same rows: it centred the
    block in the region while the alignment put it against the card, and the two
    disagreed by half the region's spare rows.

    Height degradation sheds whole sections in a fixed order — decoration,
    then teaching, then information:

    1. the logo (with the blank row under it),
    2. the hints (with the blank row above them),
    3. status rows, lowest priority first (version, then cwd, then model),
       which stops at one row so the credential warning always survives.
    """
    if width <= 0 or height <= 0:
        return []

    logo = _logo_lines(width)
    status_full = _status_rows(info, width)
    status = list(status_full)
    hints = _hint_lines(width)
    show_logo = True
    show_hints = True

    def total(rows: int) -> int:
        # One blank row joins each visible section to the status rows.
        return rows + (len(logo) + 1 if show_logo else 0) + (len(hints) + 1 if show_hints else 0)

    # The block may fill its budget to the last row. It used to hold one row
    # back so it never touched the input panel below it, and that reserve is now
    # the panel's own top padding row — an always-present blank row inside the
    # panel's fill, which is a gap the block cannot spend on content and cannot
    # accidentally lose. Asking for `height - 1` here would just buy a second
    # copy of that gap at the price of the version row.
    #
    # Escalate by what each step COSTS THE USER, which is not the same as
    # decoration-before-information:
    #
    # 1. tighten the lockup — costs one row of air.
    # 2. drop the weakest status row — the version number, which is the least
    #    actionable thing on the screen. Spending it to keep the mark is a
    #    better trade than losing the product's identity on the one screen that
    #    exists to show it, and at a 28-row terminal this single row is exactly
    #    the difference.
    # 3. drop the mark.
    # 4. drop the hints — a first-time user's way in, so it goes last.
    if total(len(status)) > height:
        logo = _logo_lines(width, flush=True)
    if total(len(status)) > height and len(status) > 1:
        status.pop(min(range(len(status)), key=lambda index: status[index][0]))
    if total(len(status)) > height:
        show_logo = False
    if total(len(status)) > height:
        show_hints = False
    while len(status) > 1 and total(len(status)) > height:
        weakest = min(range(len(status)), key=lambda index: status[index][0])
        status.pop(weakest)

    # One shared pad across the status stack and the hint stack, so the splash has
    # a single left edge below the wordmark whatever the model label turns out to
    # be. The logo is centred separately because it is centred as a LOCKUP —
    # sharing the pad would left-align the mark against the text blocks.
    status_lines, hints = _center_blocks([[line for _, line in status], hints], width)
    lines: list[Text] = []
    if show_logo:
        lines.extend(logo)
        lines.append(Text(""))
    lines.extend(status_lines)
    if show_hints:
        lines.append(Text(""))
        lines.extend(hints)

    for line in lines:
        line.truncate(width, overflow="ellipsis")
    return lines[:height]


class WelcomeView(Static):
    """The transcript's empty state: visible while it holds no blocks.

    Mounted INSIDE ``TranscriptView`` and CONTENT-SIZED (``height: auto`` in the
    tcss): the widget reports exactly the rows its block occupies, which is what
    lets the boot layout rest the splash on the input card. It owns its block's
    horizontal centring and its own height; where that block SITS is the
    stylesheet's, and only the stylesheet's.

    The region it is measured against already excludes the input panel, because
    the panel is docked and the layout engine reserves a docked child's rows
    before offering the rest to the flow — so the height budget here is simply
    the region, with no arithmetic to keep in step with the panel.

    It reads its facts through a callable rather than being pushed updates,
    because the one fact that changes under a visible welcome — the model
    label, which lands when the session factory resolves — is set from four
    places in the app. Polling once every :data:`POLL_INTERVAL_S` from here
    covers all four with no wiring, and the timer RETIRES as soon as a label
    arrives, so an idle splash is not re-reading the credential store forever.
    """

    #: Poll cadence while the model label is still unknown.
    POLL_INTERVAL_S = 0.25

    def __init__(self, info_source: Callable[[], WelcomeInfo]) -> None:
        super().__init__(id="welcome")
        self._info_source = info_source
        self._info = WelcomeInfo()
        self._timer: Any | None = None

    def on_mount(self) -> None:
        self._poll()

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Rows this block needs, out of the rows the region has LEFT.

        Textual asks this before the widget is laid out, so the region comes from
        ``container`` rather than from ``self.size``, which is still the previous
        frame's at this point.

        The region is SHARED. A system notice — one row per MCP server that failed
        to start — and the ``/clear`` receipt are siblings in this same scrollable
        column, and they are placed whatever this returns. Budgeting the whole
        region therefore overdraws it by exactly their rows: the transcript's
        virtual height passes its viewport, and because the boot layout
        bottom-aligns the column, what scrolls out of sight is the TOP OF THE
        LOGO — with a scrollbar thumb appearing beside it. Measured at 96x28 one
        failing server cost the mark's first row, two cost two.

        Subtracting a fixed row would only cover the one-notice case; the count is
        unbounded (a server per notice), so the siblings' own heights are what the
        budget has to come from. Margins count too: the ``.gap-above`` row under a
        visible splash is a row of this region like any other.
        """
        return len(
            build_welcome_lines(self._info, width, max(0, container.height - self._rows_taken()))
        )

    def _rows_taken(self) -> int:
        """Rows the sibling blocks already spend out of the shared region.

        ``outer_size`` is the placed size and excludes margin, so the gap class is
        added back explicitly — a block with a blank row above it occupies two
        rows of the region, not one.
        """
        total = 0
        for sibling in self.siblings:
            if not sibling.display:
                continue
            margin = sibling.styles.margin
            total += sibling.outer_size.height + margin.top + margin.bottom
        return total

    def render(self) -> RenderableType:
        # `self.size.height` is what `get_content_height` returned, so the block
        # rebuilt here is the one that was measured: degradation is idempotent
        # once the budget equals the block's own height.
        lines = build_welcome_lines(self._info, self.size.width, self.size.height)
        # A Group of one Text per row: the lines are already padded and
        # truncated to the widget, so nothing here may re-wrap them.
        return Group(*lines) if lines else Text("")

    def on_resize(self, event: Any) -> None:
        # Content is a function of the widget's size, so a resize is a repaint.
        self.refresh()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the view. Hidden means ``display: none`` — zero rows,
        not an empty block still holding a share of the region."""
        if visible == bool(self.display):
            return
        self.display = visible
        if visible:
            self._poll()
        else:
            self._sync_timer()

    def _poll(self) -> None:
        info = self._info_source()
        if info != self._info:
            self._info = info
            # `layout=True`: new facts can change the block's HEIGHT (the
            # credential warning appears, the model label resolves), and a
            # measured height is cached per container size — a repaint alone
            # would draw the new block into the old row count.
            self.refresh(layout=True)
        self._sync_timer()

    def _sync_timer(self) -> None:
        """Run the poll timer only while it can still learn something."""
        wanted = bool(self.display) and not self._info.model_label
        if wanted and self._timer is None:
            self._timer = self.set_interval(self.POLL_INTERVAL_S, self._poll)
        elif not wanted and self._timer is not None:
            self._timer.stop()
            self._timer = None
