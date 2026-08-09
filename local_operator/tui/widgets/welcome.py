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

    · /resume picks up a recent session where you left off

Five things make this a design decision rather than a splash screen:

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
- **One thing glows, and it is the identity.** A slow swell lifts the mark's
  rows a third of a ramp step above their resting ``dim`` and lets them back
  down, once every 4.8 s, with the two seconds between swells held at exactly
  rest; everything else — wordmark, status, hints — is fixed. Light is only
  ever ADDED, so the mark never appears to gutter, and a still frame at any
  phase reads as the same colour scheme. A boot screen with no motion at all
  reads as a hung app, and the motion this replaces was an accident: the
  input's blinking caret inverting the first letter of its placeholder twice a
  second (now off, see
  :class:`~local_operator.tui.widgets.editor.Editor`). See
  :data:`MARK_PULSE_PERIOD_S`.
- **The last row teaches something new.** One rotating tip sits under the hint
  table, a ramp step quieter: the hints are the three affordances a first-time
  user cannot infer, while the tip is the twelfth thing they would otherwise
  only meet by reading ``/help`` top to bottom. It turns over on a 12 s clock —
  a text change on a slow timer, not an animation — and it is the FIRST section
  the height ladder sheds, because a tip a short terminal has no room for comes
  round again on the next launch. See :data:`TIPS`.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from textual.color import Color
from textual.geometry import Size
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.shimmer import shimmer_enabled
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

#: Seconds between one glow over the mark and the next.
#:
#: The mark is at rest for :data:`MARK_PULSE_PERIOD_S` minus
#: :data:`MARK_PULSE_SWELL_S` of every cycle — two thirds of the time — which is
#: the difference between a strobe and a breath. A continuous sinusoid is ALWAYS
#: moving, and idle motion that never stops is what makes a boot screen nag: the
#: eye keeps returning to it because it never finishes. A glow that arrives,
#: passes and then leaves the frame alone for twice as long as it lasted finishes
#: every 4.8 s, and the still frame it returns to is the mark's own resting
#: ``dim`` — so most frames a user ever looks at are the static frame.
MARK_PULSE_PERIOD_S = 4.8

#: Seconds the glow itself takes, rise and fall together.
#:
#: 0.8 s up and 0.8 s back. Fast enough to read as one gesture rather than as the
#: theme drifting, slow enough that no edge is visible: at 12.5 fps the swell is
#: twenty frames, so nothing ever jumps a step. Below roughly a second the mark
#: starts reading as a pulse-rate monitor, which is the same complaint the
#: blinking caret earned.
MARK_PULSE_SWELL_S = 1.6

#: Timer cadence for the glow: 12.5 fps.
#:
#: This is idle background motion behind whatever the user is typing, so it is
#: budgeted like it. 30 fps would more than double the wakeups for no visible
#: gain: the whole excursion spans only ten distinct hex values (a third of a
#: ramp step, quantised to 8 bits per channel), so most of the extra frames
#: would repaint byte-identical output. At 12.5 fps each colour step is held for
#: a frame or two, which is exactly smooth — and through the two thirds of the
#: cycle that are rest, every tick returns without repainting at all.
MARK_PULSE_INTERVAL_S = 0.08

#: How far along the ramp the glow's peak travels, as a fraction of the step from
#: ``dim`` to ``muted``.
#:
#: MEASURED, not guessed. Contrast on the dark ramp: rest is ``#837C6D`` at
#: 4.55:1 against the ground, and a third of a step lands the peak at ``#928B7C``
#: and 5.57:1 — 1.22:1 peak to rest. That is a mark someone notices only if they
#: are looking at it, and a still frame captured at any phase reads as the same
#: colour scheme. The full ``dim``→``muted`` step is 1.90:1 and buries the
#: wordmark, which :func:`_logo_lines` already rejected as a resting tint.
#:
#: One direction only, and that is the point of a glow: the old breath spent half
#: its cycle BELOW rest, towards ``faint``, which reads as the logo guttering
#: rather than as light passing over it. Light is added here and never taken
#: away, so the mark's floor is the value the lockup's hierarchy was set at.
MARK_PULSE_DEPTH = 0.3


def mark_pulse_phase(elapsed_s: float) -> float:
    """Glow level at ``elapsed_s`` seconds: ``0`` at rest, ``1`` at the peak.

    A raised cosine over the swell and a flat zero through the rest of the cycle.
    Raised cosine rather than a sine or a triangle because it leaves and returns
    to rest with zero slope: the glow has no onset edge and no cut-off, so it
    fades up out of the still frame and back into it. Never negative — see
    :data:`MARK_PULSE_DEPTH`.

    Phase zero IS rest, so the first frame of a boot is identical whether the
    animation is on or off and the motion grows out of the still frame instead of
    the splash appearing pre-brightened and settling.
    """
    into_cycle = elapsed_s % MARK_PULSE_PERIOD_S
    if into_cycle >= MARK_PULSE_SWELL_S:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * (into_cycle / MARK_PULSE_SWELL_S)))


def mark_pulse_color(phase: float) -> str:
    """The mark's hex at ``phase``: ``dim``, lifted towards ``muted``.

    Both ends are ramp tokens read through :mod:`local_operator.tui.theme`, so the
    glow follows a theme switch like everything else and no hex is minted here.
    ``muted`` is the ramp's next step UP from the mark's resting ``dim``, so the
    excursion stays inside one step of the value the lockup's hierarchy was set at
    (see :func:`_logo_lines`) and rest is exactly rest.
    """
    dim = Color.parse(theme_mod.semantic_color("dim"))
    muted = Color.parse(theme_mod.semantic_color("muted"))
    return dim.blend(muted, max(0.0, min(1.0, phase)) * MARK_PULSE_DEPTH).hex


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

#: The rotating tip pool, one line each.
#:
#: Every entry names something THIS BUILD ANSWERS — a command in
#: ``local_operator.tui.app.SLASH_COMMANDS``, a key in
#: ``OperatorApp.BINDINGS``, or a tool the agent is actually handed — because a
#: splash advertising a command the app rejects is worse than a blank row, and
#: this is the one screen a first-run user reads word for word.
#:
#: Each one leads with the command or the verb so that the width tiers can
#: truncate the tail and still leave something actionable behind; each fits
#: inside a 60-cell terminal untruncated (pinned in the tests).
#:
#: Twelve, and deliberately not more: the pool is what a user meets a couple of
#: entries at a time across many launches, so every addition dilutes the odds of
#: seeing the ones that change how the app is used. The first entry is the one
#: every still frame shows (see :meth:`WelcomeView._sync_tip_timer`), which is
#: why it is resumption — the single question a returning user arrives with.
TIPS: tuple[str, ...] = (
    "/resume picks up a recent session where you left off",
    "/model switches the model for this session",
    "/model default <provider>/<id> sets the boot default",
    "/usage shows how much provider quota is left",
    "/approvals auto runs tools unprompted, ask confirms",
    "Type while the agent works — it sends at the next step",
    "esc stops the agent without ending the session",
    "Ask for parallel work and the agent fans out subagents",
    "Compaction runs itself when the context window fills",
    "/mcp lists the MCP servers found in your mcp.json",
    "/skills lists the skills loaded for this session",
    "/goal sets a standing objective, /loop iterates to it",
)

#: The tip's prefix: the app's own `info` glyph, the mark every quiet one-line
#: receipt in the transcript already carries (D14). A word — `tip:` — would cost
#: five cells of the sentence to label a line whose tone already says what it is,
#: and the mandate takes structure from symbols rather than prefixes.
TIP_GLYPH = NOTICE_GLYPHS["info"]

#: Seconds one tip is held before the next takes its place.
#:
#: The splash is glanced at, not read: the row has to hold long enough that it
#: is never mid-sentence when the eye lands on it, and turn over often enough
#: that a user composing a first prompt meets more than one. Under about 8 s the
#: line changes while it is still being read, and — worse — a text change inside
#: the peripheral field pulls focus off the input the user is typing into. Over
#: about 15 s a short session only ever sees the first entry, which makes the
#: pool pointless. 12 s splits that band, and is not a whole multiple of the
#: mark's 4.8 s glow cycle (12 / 4.8 = 2.5), so the two motions never fall into
#: step and start reading as one animation.
TIP_ROTATE_INTERVAL_S = 12.0

#: Narrowest width that gets a tip at all.
#:
#: Below this the row holds a command and a word or two of its reason, which is
#: a fragment rather than a tip — the same judgement :func:`_hint_lines` makes
#: when it drops to keys only. The threshold is a WIDTH and never the current
#: tip's own length: presence has to be the same answer for every entry in the
#: pool, or the block would gain and lose a row as it rotated. See
#: :func:`_tip_lines`.
TIP_MIN_WIDTH = 32

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


def _logo_lines(width: int, *, flush: bool = False, mark_color: str | None = None) -> list[Text]:
    """The logo lockup at whichever width tier ``width`` allows.

    - ``>= LOGO_FULL_MIN_WIDTH`` (27): the mark, a blank row, then the
      letterspaced wordmark — a small dense mark over a wide open name.
    - ``>= MARK_WIDTH``: the mark directly over the plain wordmark, flush,
      because a lockup without the width to be loose should not pretend.
    - narrower: the plain wordmark alone, which the caller then truncates.

    ``flush`` drops the row of air in the wide tier. It is a HEIGHT
    concession, not a width one: on a short terminal the choice is between a
    tighter lockup and no mark at all, and one row of air is not worth the
    product's own identity. The caller escalates to it before shedding sections.

    ``mark_color`` overrides the mark's resting tint with one frame of the glow
    (:func:`mark_pulse_color`). It is a colour and not a phase
    so this stays a pure function of what it is handed — the clock lives in
    :class:`WelcomeView`, and geometry never depends on it either way.
    """
    # `dim`, not `muted`. The intended hierarchy is a compact mark UNDER a wide
    # open name, but the mark is ten rows of solid block glyphs against a
    # wordmark of one row, so one ramp step could not overcome the area
    # difference and the eye landed on the blocks first. Two steps down makes
    # them read as a watermark behind the name. The glow lifts a third of a step
    # UP from this value and always returns to it, so the hierarchy holds at
    # every phase — the mark is never brighter than the wordmark it sits behind.
    mark_style = Style(color=mark_color or theme_mod.semantic_color("dim"))
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


def _tip_lines(width: int, index: int) -> list[Text]:
    """The tip at ``index``, as ONE row — or no row at all when ``width`` is tight.

    The row count is a function of ``width`` alone. That is the contract the
    whole rotation rests on: this view is content-sized and the boot layout rests
    it ON the input card, so a tip that could be one row for one entry and two
    (or none) for the next would shove the entire splash up and down the screen
    every :data:`TIP_ROTATE_INTERVAL_S`. Hence ``no_wrap`` plus the shared
    truncation pass in :func:`build_welcome_lines` rather than a wrap, and hence a
    width threshold rather than a per-tip length test.

    ``index`` is taken modulo the pool so callers can keep a monotonic counter.
    """
    if width < TIP_MIN_WIDTH:
        return []
    # `faint` for the glyph, `dim` for the sentence. `dim` is the quietest ink the
    # app will set a whole sentence in (it is what the version and cwd rows use,
    # at 4.55:1 on the ground); `faint` is a step below that and reserved for
    # marks and separators — legible as a bullet, not as prose. The pair puts the
    # tip under the hints' fg/muted without making it a thing the eye has to work
    # at, which is the one failure mode that would make a rotating row annoying.
    glyph_style = Style(color=theme_mod.semantic_color("faint"))
    body_style = Style(color=theme_mod.semantic_color("dim"))
    line = Text(no_wrap=True)
    line.append(f"{TIP_GLYPH} ", style=glyph_style)
    line.append(TIPS[index % len(TIPS)], style=body_style)
    return [line]


def build_welcome_lines(
    info: WelcomeInfo,
    width: int,
    height: int,
    *,
    mark_color: str | None = None,
    tip_index: int = 0,
) -> list[Text]:
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

    1. the tip (with the blank row above it),
    2. the logo (with the blank row under it),
    3. the hints (with the blank row above them),
    4. status rows, lowest priority first (version, then cwd, then model),
       which stops at one row so the credential warning always survives.

    ``mark_color`` tints the mark for one frame of the glow and is
    the ONLY argument that cannot change the result's shape: it reaches
    :func:`_logo_lines` and stops at a ``Style``. That is what lets
    :class:`WelcomeView` repaint a glow frame without re-measuring — see
    :meth:`WelcomeView._pulse_tick`.

    ``tip_index`` selects the rotating tip. It cannot change the shape either:
    the tip is one row at every width that draws it at all, so a rotation is a
    repaint and never a re-measure — see :func:`_tip_lines` and
    :meth:`WelcomeView._tip_tick`.
    """
    if width <= 0 or height <= 0:
        return []

    logo = _logo_lines(width, mark_color=mark_color)
    status_full = _status_rows(info, width)
    status = list(status_full)
    hints = _hint_lines(width)
    tip = _tip_lines(width, tip_index)
    show_logo = True
    show_hints = True
    show_tip = bool(tip)

    def total(rows: int) -> int:
        # One blank row joins each visible section to the status rows.
        return (
            rows
            + (len(logo) + 1 if show_logo else 0)
            + (len(hints) + 1 if show_hints else 0)
            + (len(tip) + 1 if show_tip else 0)
        )

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
    # 2. drop the tip, and hand that row of air straight back. It is the only row
    #    here the user loses NOTHING permanent with — the same entry comes round on
    #    the next launch, while every other row is either a fact or the way in —
    #    and it is worth two, so what is left is EXACTLY the splash as it stood
    #    before tips existed. From here down a 28-row terminal draws the frame it
    #    always drew, with the tip as the thing that appears when there is room.
    # 3. tighten the lockup again. The blank row is still the cheapest concession
    #    on the screen, so it is spent twice rather than once.
    # 4. drop the weakest status row — the version number, which is the least
    #    actionable thing on the screen. Spending it to keep the mark is a
    #    better trade than losing the product's identity on the one screen that
    #    exists to show it, and at a 28-row terminal this single row is exactly
    #    the difference.
    # 5. drop the mark.
    # 6. drop the hints — a first-time user's way in, so it goes last.
    if total(len(status)) > height:
        logo = _logo_lines(width, flush=True, mark_color=mark_color)
    if total(len(status)) > height and show_tip:
        show_tip = False
        logo = _logo_lines(width, mark_color=mark_color)
    if total(len(status)) > height:
        logo = _logo_lines(width, flush=True, mark_color=mark_color)
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
    if show_tip:
        # Centred on its own, not on the shared pad: it is a sentence, not a row
        # of a table, and it is usually the widest line on the screen — folding it
        # into `_center_blocks` would drag the status and hint stacks left by half
        # its length, and drag them a different distance for every tip.
        lines.append(Text(""))
        lines.extend(_center(line, width) for line in tip)

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

    It also owns the mark's glow and the tip's rotation. Three timers rather than
    one shared tick, because they answer to different things and at cadences
    three orders of magnitude apart: the poll retires when the model label lands,
    the glow runs at 12.5 fps for as long as the splash is on screen, and the tip
    turns over once every twelve seconds. Folding them together would either
    re-read the credential store twelve times a second or glow at 4 fps.
    """

    class BlockResized(Message):
        """The block's row count changed, so anything composed around it has moved.

        Posted only by :meth:`_poll`, which is the only thing that can change the
        height of a splash that is already on screen — the model label lands and a
        credential warning appears with it. The boot composition is centred on that
        height, and the app has no other way to learn it moved: a widget's own
        ``Resize`` never reaches the app, and the composition is deliberately no
        longer a measurement that re-runs itself until it settles.
        """

    #: Poll cadence while the model label is still unknown.
    POLL_INTERVAL_S = 0.25

    def __init__(self, info_source: Callable[[], WelcomeInfo]) -> None:
        super().__init__(id="welcome")
        self._info_source = info_source
        self._info = WelcomeInfo()
        self._timer: Any | None = None
        self._pulse_timer: Any | None = None
        self._pulse_origin = 0.0
        # The mark's tint for the CURRENT frame, or None for its resting `dim`.
        # None is not "unknown": it is the value the splash has always drawn and
        # the value it holds whenever the pulse is not running, so a still frame
        # is the old still frame rather than an arbitrary sample of a new
        # animation. Caching the colour (not the phase) is also what lets a tick
        # decide whether it has anything to repaint.
        self._mark_color: str | None = None
        self._tip_timer: Any | None = None
        # Which tip is on screen. Zero is not a placeholder: it is what a still
        # frame shows (the rotation is gated on the animation switch, see
        # `_sync_tip_timer`), so it is the entry every reproducible frame in the
        # suite and every snapshot carries.
        self._tip_index = 0

    def on_mount(self) -> None:
        self._poll()
        self._sync_pulse_timer()
        self._sync_tip_timer()

    def on_unmount(self) -> None:
        """All three timers die with the widget.

        A Textual interval outlives the widget that made it: it keeps firing at
        a callback whose screen is gone, which this suite has already paid for
        once as a shutdown warning and an intermittent teardown failure.
        :meth:`set_visible` covers the ordinary retirement (the first transcript
        block), but it is not the only exit — every ``run_test`` that never
        sends a prompt tears the app down with the splash still up, and so does
        ``ctrl+d`` on the boot screen.
        """
        self._stop_timer()
        self._stop_pulse_timer()
        self._stop_tip_timer()

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
        return self._block_rows(container.height, width)[0]

    def spare_rows(self, region_height: int, width: int) -> int:
        """Rows a region of ``region_height`` would have LEFT over this block.

        The boot composition is centred on this number, and it has to be answerable
        BEFORE the frame exists: the app resolves the composition inside the resize
        that precedes the first arrange, so the splash lands centred in the first
        frame the terminal is shown instead of walking into place over the next two
        dozen. Sharing :meth:`_block_rows` with ``get_content_height`` is what keeps
        the two from disagreeing — the app is asking exactly the question the layout
        engine is about to ask, one step ahead of it.
        """
        rows, taken = self._block_rows(region_height, width)
        return max(0, region_height - taken - rows)

    def _block_rows(self, region_height: int, width: int) -> tuple[int, int]:
        """``(rows this block occupies, rows its siblings already spend)``."""
        taken = self._rows_taken(region_height, width)
        lines = build_welcome_lines(
            self._info,
            width,
            max(0, region_height - taken),
            tip_index=self._tip_index,
        )
        return len(lines), taken

    def _rows_taken(self, region_height: int, width: int) -> int:
        """Rows the sibling blocks already spend out of the shared region.

        Each sibling is asked what it NEEDS at this width rather than read off its
        placed size. A block mounted a line ago has no placed size yet — an MCP
        failure notice is appended and this runs in the same call — and counting
        it as zero rows budgets the splash for a region it no longer has to
        itself. That is what put the composition one notice behind: three failing
        servers centred the block as though one had failed.

        Margins count too, and they are added back explicitly because
        ``get_content_height`` answers for the content box alone: a block with a
        blank row above it occupies two rows of the region, not one. The gap class
        is already on the block by the time it is mounted (``TranscriptView``
        applies it before the mount and only RE-decides it once the width is
        real), so the margin read here is the one the layout will honour.
        """
        viewport = self.app.size
        total = 0
        for sibling in self.siblings:
            if not sibling.display:
                continue
            margin, gutter = sibling.styles.margin, sibling.styles.gutter
            inner = max(0, width - gutter.width)
            total += (
                sibling.get_content_height(Size(inner, region_height), viewport, inner)
                + gutter.height
                + margin.height
            )
        return total

    def render(self) -> RenderableType:
        # `self.size.height` is what `get_content_height` returned, so the block
        # rebuilt here is the one that was measured: degradation is idempotent
        # once the budget equals the block's own height.
        lines = build_welcome_lines(
            self._info,
            self.size.width,
            self.size.height,
            mark_color=self._mark_color,
            tip_index=self._tip_index,
        )
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
        self._sync_pulse_timer()
        self._sync_tip_timer()
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
            # And tell whatever is composed AROUND the block that it moved. The
            # boot composition reserves rows either side of this splash and is
            # resolved in one pass, so nothing else would ever notice.
            self.post_message(self.BlockResized())
        self._sync_timer()

    def _sync_timer(self) -> None:
        """Run the poll timer only while it can still learn something."""
        wanted = bool(self.display) and not self._info.model_label
        if wanted and self._timer is None:
            self._timer = self.set_interval(self.POLL_INTERVAL_S, self._poll)
        elif not wanted and self._timer is not None:
            self._stop_timer()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _sync_pulse_timer(self) -> None:
        """Glow only while the splash is on screen and animation is allowed.

        Gated on the SAME switch as the shimmer (``LOCAL_OPERATOR_NO_SHIMMER``,
        the ``display.shimmer`` setting) because "hold still" is one decision,
        not one per surface: CI and the SVG snapshot harness turn animation off
        once and expect every frame to be reproducible. With the gate closed no
        timer is created at all — the glow is not merely paused — and the mark
        keeps its resting ``dim``.

        The clock restarts from the moment the splash appears rather than from
        app start, so a ``/clear`` an arbitrary number of seconds in gets the
        same first frame as a boot: at rest, then swelling.
        """
        wanted = bool(self.display) and shimmer_enabled()
        if wanted and self._pulse_timer is None:
            self._pulse_origin = time.monotonic()
            self._pulse_timer = self.set_interval(MARK_PULSE_INTERVAL_S, self._pulse_tick)
        elif not wanted and self._pulse_timer is not None:
            self._stop_pulse_timer()

    def _stop_pulse_timer(self) -> None:
        """Stop glowing and return the mark to rest.

        The colour is cleared with the timer so the next frame drawn after a
        stop is the resting one — a hidden view that comes back on ``/clear``
        must not flash the phase it happened to be paused at.
        """
        if self._pulse_timer is not None:
            self._pulse_timer.stop()
            self._pulse_timer = None
        self._mark_color = None

    def _pulse_tick(self) -> None:
        """One glow frame: a colour, and a repaint only when it MOVED.

        ``refresh()``, never ``refresh(layout=True)``. The glow changes one
        ``Style`` and no geometry, and a re-measure here would re-run the height
        degradation ladder twelve times a second — on a boot frame sitting one
        row from the threshold that drops the mark, that is a block that
        twitches while the user reads it.

        The colour is compared before repainting because most ticks have nothing
        to say: two thirds of the cycle is held at rest, and the swell itself
        quantises to ten hexes across twenty ticks, so a tick that repainted
        unconditionally would send the terminal byte-identical output four times
        out of five.
        """
        color = mark_pulse_color(mark_pulse_phase(time.monotonic() - self._pulse_origin))
        if color == self._mark_color:
            return
        self._mark_color = color
        self.refresh()

    def _sync_tip_timer(self) -> None:
        """Rotate only while the splash is on screen and animation is allowed.

        The SAME gate as the pulse, for the same reason and then one more: a row
        of text that changes on a clock makes every still frame a sample of an
        animation, so a snapshot would capture whichever tip the wall clock
        happened to be holding. With the gate closed no timer exists — the
        rotation is not merely paused — and the row holds at ``TIPS[0]``.
        """
        wanted = bool(self.display) and shimmer_enabled()
        if wanted and self._tip_timer is None:
            # Start somewhere in the ring, then advance by one. A user who types
            # their first prompt straight away sees exactly ONE tip per launch, so
            # a fixed start would make everything after `TIPS[0]` unreachable for
            # them — the pool would exist for nobody. Choosing per tick instead
            # would let the same tip come up twice in a row, which reads as a
            # broken rotation rather than a coincidence.
            #
            # The offset is drawn only behind the animation gate, so the frames
            # that must be reproducible are exactly the frames that hold at
            # `TIPS[0]`, and re-drawn per appearance rather than per app, so a
            # `/clear` an hour in is not the boot frame again.
            self._tip_index = random.randrange(len(TIPS))
            self._tip_timer = self.set_interval(TIP_ROTATE_INTERVAL_S, self._tip_tick)
        elif not wanted and self._tip_timer is not None:
            self._stop_tip_timer()

    def _stop_tip_timer(self) -> None:
        """Stop rotating and return the row to the first tip.

        The index is cleared with the timer for the reason the pulse clears its
        colour: a still frame must be the DEFINED still frame, not the entry the
        rotation happened to be paused on.
        """
        if self._tip_timer is not None:
            self._tip_timer.stop()
            self._tip_timer = None
        self._tip_index = 0

    def _tip_tick(self) -> None:
        """The next tip in the ring, and a repaint that cannot move a row.

        ``refresh()`` and never ``refresh(layout=True)``, and here that is
        load-bearing rather than a saving: the tip occupies exactly one row at
        every width that draws it at all (:func:`_tip_lines`), so the measured
        height cannot have changed. A re-measure would re-run the degradation
        ladder against a block the layout rests on the input card, and this view
        is the one place in the app where a row appearing or vanishing moves the
        whole splash.
        """
        self._tip_index = (self._tip_index + 1) % len(TIPS)
        self.refresh()
