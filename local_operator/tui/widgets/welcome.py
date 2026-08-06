"""New-session welcome view — the transcript's EMPTY STATE, not a banner.

A fresh session (and every ``/clear``) shows one centered, BORDERLESS block in
the space above the input dock:

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
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.status_line import format_model_label
from local_operator.tui.widgets.transcript import NOTICE_GLYPHS

#: The monogram: an angular ``L`` beside a rounded ``O``. Only the three
#: universal block-element glyphs are used (█ ▀ ▄, all in U+258x) — the half
#: blocks round the ``O``'s corners without depending on box-drawing or
#: braille coverage, so the mark survives a bare xterm font.
#:
#: Every row is exactly :data:`MARK_WIDTH` cells wide, and so is
#: :data:`WORDMARK` — that equality is the whole lockup, and it is asserted by
#: the tests rather than left as a coincidence for someone to break.
LOGO_MARK: tuple[str, ...] = (
    "██      ▄████▄",
    "██      ██  ██",
    "██      ██  ██",
    "██████  ▀████▀",
)

#: The wordmark, in the product's own lowercase voice.
WORDMARK = "local operator"

#: Letterspaced wordmark for the wide lockup. ``" ".join`` over the characters
#: turns the word gap into three cells, which is exactly the spacing a
#: letterspaced wordmark wants — no separate word-gap constant to keep in sync.
WORDMARK_SPACED = " ".join(WORDMARK)

#: Cell width of the mark, and of the plain wordmark under it.
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

    Both reads are defended. ``model_label`` touches a session that may be
    mid-teardown, and ``has_any_credential`` touches the credential store on
    disk; either raising here would take down the app's very first render.
    A failed credential read degrades to *no warning* rather than to a false
    alarm: telling a correctly configured user they are logged out is worse
    than staying quiet.
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
            if not providers.has_any_credential(provider):
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


def _center_block(lines: list[Text], width: int) -> list[Text]:
    """Center a GROUP of lines on their widest member, keeping one left edge.

    Centering each line independently ragged the status stack into a diamond —
    with a real OpenRouter label the version, model, cwd and warning rows had
    four different left edges, which is the one composition move that reads as
    un-designed, on the first frame every user sees (D9).

    It also removes a visible twitch: ``model_label`` starts as ``connecting…``
    (12 cells) and resolves to as many as 38, so the row used to re-centre and
    re-widen a second into every boot. Block width is set by the widest row, so
    the stack now holds still.

    ``_hint_lines`` already worked this way and documents why; the status rows
    are the same kind of thing — several facts about one session.
    """
    if not lines:
        return []
    block = max(cell_len(line.plain) for line in lines)
    pad = max(0, (width - block) // 2)
    if pad == 0:
        return lines
    out: list[Text] = []
    for line in lines:
        padded = Text(" " * pad, no_wrap=True)
        padded.append_text(line)
        out.append(padded)
    return out


def _logo_lines(width: int) -> list[Text]:
    """The logo lockup at whichever of three width tiers ``width`` allows.

    - ``>= LOGO_FULL_MIN_WIDTH`` (27): the mark, a blank row, then the
      letterspaced wordmark — a small dense mark over a wide open name.
    - ``>= MARK_WIDTH`` (14): the mark directly over the plain wordmark. Both
      are exactly 14 cells, so they lock flush and the separating row would
      only loosen a lockup that no longer has the width to be loose.
    - narrower: the plain wordmark alone, which the caller then truncates.
    """
    # `dim`, not `muted`. The intended hierarchy is a small dense mark UNDER a
    # wide open name, but the mark is four rows of solid block glyphs (~40 filled
    # cells) against a wordmark of one row and ~11 cells of ink, so one ramp step
    # could not overcome a 4x area difference and the eye landed on the blocks
    # first. Two steps down makes the blocks read as a watermark behind the name
    # — which is also what stops the rounded `O` reading as a zero (D12).
    mark_style = Style(color=theme_mod.semantic_color("dim"))
    word_style = Style(color=theme_mod.semantic_color("fg"))
    mark = [_center(Text(row, style=mark_style, no_wrap=True), width) for row in LOGO_MARK]
    if width >= LOGO_FULL_MIN_WIDTH:
        spaced = _center(Text(WORDMARK_SPACED, style=word_style, no_wrap=True), width)
        return [*mark, Text(""), spaced]
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
    if not key_column:
        return [_center(Text(key, style=key_style, no_wrap=True), width) for key, _ in HINTS]

    block = max(key_column + cell_len(desc) for _, desc in HINTS)
    pad = max(0, (width - block) // 2)
    lines: list[Text] = []
    for key, desc in HINTS:
        line = Text(" " * pad, no_wrap=True)
        line.append(key.ljust(key_column), style=key_style)
        line.append(desc, style=desc_style)
        lines.append(line)
    return lines


def build_welcome_lines(info: WelcomeInfo, width: int, height: int) -> list[Text]:
    """Render the welcome block as exactly the lines it occupies.

    Pure, so the geometry is testable without a running app. Returns at most
    ``height`` lines, none wider than ``width``.

    Height degradation sheds whole sections in a fixed order — decoration,
    then teaching, then information:

    1. the logo (with the blank row under it),
    2. the hints (with the blank row above them),
    3. status rows, lowest priority first (version, then cwd, then model),
       which stops at one row so the credential warning always survives.

    Vertical placement floors the top pad, which lands the block on the upper
    of the two centre positions when the free rows are odd. That is deliberate:
    the input dock sits below this region, so a block nudged up reads as
    centered while a block nudged down reads as crowding the prompt.
    """
    if width <= 0 or height <= 0:
        return []

    logo = _logo_lines(width)
    status = _status_rows(info, width)
    hints = _hint_lines(width)
    show_logo = True
    show_hints = True

    def total(rows: int) -> int:
        # One blank row joins each visible section to the status rows.
        return rows + (len(logo) + 1 if show_logo else 0) + (len(hints) + 1 if show_hints else 0)

    # A row is held back so the block never touches the input dock's rule.
    # Opening the picker shrinks this region to exactly the block's height, so
    # the old arithmetic centered at pad 0 and rendered edge-to-edge: the mark
    # looked like it was sliding off the top and `ctrl+d  quit` sat directly on
    # the rule. That is reachable in ONE keystroke from the state the splash
    # itself teaches, since a hint row says "/  command picker" (D11).
    #
    # The margin is a nicety and yields to content: it is worth shedding the
    # logo or the hints for, but not a status row. At three rows the credential
    # warning and the model matter more than breathing room, so the first pass
    # asks for the margin and the second gives it up rather than shed a fact.
    for usable in (max(1, height - 1), height):
        show_logo = True
        show_hints = True
        if total(len(status)) > usable:
            show_logo = False
        if total(len(status)) > usable:
            show_hints = False
        if total(len(status)) <= usable:
            break

    while len(status) > 1 and total(len(status)) > usable:
        weakest = min(range(len(status)), key=lambda index: status[index][0])
        status.pop(weakest)

    body: list[Text] = []
    if show_logo:
        body.extend(logo)
        body.append(Text(""))
    body.extend(_center_block([line for _, line in status], width))
    if show_hints:
        body.append(Text(""))
        body.extend(hints)

    top = max(0, (usable - len(body)) // 2)
    lines = [Text("") for _ in range(top)]
    lines.extend(body)
    for line in lines:
        line.truncate(width, overflow="ellipsis")
    return lines[:height]


class WelcomeView(Static):
    """The transcript's empty state: visible while it holds no blocks.

    Mounted INSIDE ``TranscriptView`` at ``height: 1fr``, which is what makes
    the region arithmetic disappear — the widget is handed exactly the rows
    above the input dock, and it yields rows to any block mounted under it
    (the ``/clear`` notice) instead of overflowing the scroll area.

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

    def render(self) -> RenderableType:
        lines = build_welcome_lines(self._info, self.size.width, self.size.height)
        # A Group of one Text per row: the lines are already padded and
        # truncated to the widget, so nothing here may re-wrap them.
        return Group(*lines) if lines else Text("")

    def on_resize(self, event: Any) -> None:
        # Content is a function of the widget's size, so a resize is a repaint.
        self.refresh()

    def set_visible(self, visible: bool) -> None:
        """Show or hide the view. Hidden means ``display: none`` — zero rows,
        not an empty block still holding a ``1fr`` share of the region."""
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
            self.refresh()
        self._sync_timer()

    def _sync_timer(self) -> None:
        """Run the poll timer only while it can still learn something."""
        wanted = bool(self.display) and not self._info.model_label
        if wanted and self._timer is None:
            self._timer = self.set_interval(self.POLL_INTERVAL_S, self._poll)
        elif not wanted and self._timer is not None:
            self._timer.stop()
            self._timer = None
