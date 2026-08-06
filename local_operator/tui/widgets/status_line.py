"""Status line — the full-width BAND on the kit's ``sunken`` ground (D3/D17).

The character refinement supersedes the thin border-row trick: the status
line is a full-width band at the bottom of the input dock, painted on the
``sunken`` ground with segments separated by ``·``:

    π openrouter/kimi-k2 · high · ~/local-operator
                    2 agents · 49.6%/1M · $0.12 · 41m1s · Parser rewrite

(one row on a wide terminal; wrapped here only to fit this docstring)

Left group: brand glyph · provider/model · effort · working dir (+ the
shimmering working indicator while a turn streams — faithful: the working
text rides the shimmer sweep; when shimmer is off a static dim spinner keeps
it legible, D26). Right group, right-aligned: subagents · context usage ·
cost · active duration · conversation name.

**Nine segments do not fit an 80-column terminal, and the band is exactly one
row at every width.** Both facts are load-bearing, so overflow is not left to
truncation — truncation would clip whichever segment happened to be last and
leave a half-written number on screen. Instead :data:`_DROP_LADDER` names an
explicit reduction order, applied step by step until the row fits. The
ordering protects what the operator steers by (provider/model, working dir,
context usage) and spends the transient counters (background jobs, elapsed
time) first. All widths measured with ``rich.cells.cell_len`` (one width
model).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from local_operator.tui import theme as theme_mod

#: Spinner frames shown while the session is streaming (~12.5 fps glyph
#: cadence when shimmer is disabled).
_SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
_SPINNER_INTERVAL_S = 0.08

#: The brand glyph leading the band (π — the operator's own mark).
BRAND_GLYPH = "π"

_SEPARATOR = " · "

#: Reduction order, cheapest loss first. Each entry is applied in turn until
#: the row fits the available width.
#:
#: Rationale, because this is the part most likely to look wrong rather than
#: be wrong: ``subagents`` and ``duration`` are transient counters the
#: operator can re-derive from the transcript, so they go first. ``name`` is
#: a label, not a number.
#:
#: The two SHORTEN steps come next and outrank every remaining drop, because
#: they recover width while keeping the segment: a basename still answers
#: "where am I", and a bare model id still answers "who is replying".
#: Together they free ~35 cells on a realistic label, which is more than the
#: cost and context segments cost combined — dropping numbers to preserve a
#: fully-qualified path would be the wrong trade.
#:
#: Cost and context usage are the numbers that change decisions mid-task, so
#: they outlast everything except the left group, and the model is the very
#: last thing to go: a band that cannot say which model is answering is worse
#: than no band. The brand glyph and the streaming spinner are never
#: droppable — the glyph is one cell and the spinner is the liveness signal.
_DROP_LADDER: tuple[str, ...] = (
    "subagents",
    "duration",
    "name",
    "shorten-cwd",
    "shorten-model",
    "cost",
    "context",
    "effort",
    "cwd",
    "model",
)


def format_context_tokens(tokens: int) -> str:
    """Compact context estimate: ``12.4k`` / ``1.2m`` style, plain under 1k."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_window(window: int) -> str:
    """Abbreviate a context window for the denominator: ``1M``, ``200k``.

    Capital ``M`` and lower-case ``k`` are the conventional units for model
    windows, and a whole window renders without a decimal (``1M``, not
    ``1.0M``) — the denominator is a label, not a measurement.
    """
    if window >= 1_000_000:
        scaled = window / 1_000_000
        return f"{scaled:.0f}M" if scaled == int(scaled) else f"{scaled:.1f}M"
    if window >= 1_000:
        scaled = window / 1_000
        return f"{scaled:.0f}k" if scaled == int(scaled) else f"{scaled:.1f}k"
    return str(window)


def format_context_usage(tokens: int, window: int) -> str:
    """Context usage as ``49.6%/1M``; ``12.4k/—`` when the window is unknown.

    A percentage needs a denominator. When the registry has no window for the
    model, showing a bare percentage would invent one, so the spend is reported
    against an explicit unknown instead.

    The em dash rather than ``?`` because it is already this UI's glyph for an
    absent value — unknown cost renders ``$—`` and a provider with no stored
    credential renders ``—``. Two spellings of "unknown" in one row would read
    as two different states.
    """
    if tokens <= 0:
        return ""
    if window <= 0:
        return f"{format_context_tokens(tokens)}/—"
    return f"{tokens / window * 100:.1f}%/{format_window(window)}"


def format_cost(cost: float) -> str:
    """Compact dollar cost: ``$0.0021`` under a cent, ``$0.12`` above."""
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


def format_duration(seconds: float) -> str:
    """Active processing time: ``9s``, ``41m1s``, ``1h2m``.

    Units are dropped once they stop carrying information: past an hour the
    seconds are noise, and a whole minute renders as ``5m`` rather than
    ``5m0s``. Sub-second work renders as ``0s`` rather than vanishing, so a
    finished turn always leaves a mark.
    """
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        minutes, secs = divmod(total, 60)
        return f"{minutes}m{secs}s" if secs else f"{minutes}m"
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


def format_agents(count: int) -> str:
    """``1 agent`` / ``3 agents``; empty at zero (the segment disappears).

    Zero renders as nothing rather than "0 agents": a band that permanently
    reports the absence of background work is nine characters of noise, and
    the segment appearing is itself the signal.
    """
    if count <= 0:
        return ""
    return f"{count} agent" if count == 1 else f"{count} agents"


def format_cwd(cwd: str, *, short: bool) -> str:
    """The working dir as ``~/rel/path``, or just its basename when ``short``.

    Home-relative beats absolute (the ``/Users/<name>`` prefix is the same on
    every row of every session) and beats the bare basename, which is
    ambiguous the moment two checkouts share a directory name — the basename
    is what the overflow ladder falls back to, not what it starts from.
    """
    if not cwd:
        return ""
    path = Path(cwd)
    if short:
        return path.name or cwd
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        # Outside the home tree (/tmp, /opt, a mounted volume): the absolute
        # path is already the shortest honest rendering.
        return str(path)


def format_model_label(label: str, *, short: bool) -> str:
    """``provider/model``, or just the model id when ``short``.

    The provider is the droppable half: an operator who has switched
    providers knows which one they are on, but two providers' model ids are
    rarely confusable, so ``openrouter/moonshotai/kimi-k2`` degrades to
    ``kimi-k2`` rather than disappearing. Only the LAST path segment is kept
    — vendor-scoped ids carry two prefixes, and dropping one of them would
    leave a label that is still too long to help.
    """
    if not label:
        return ""
    if not short:
        return label
    return label.rpartition("/")[2] or label


class StatusLine:
    """Draws status segments into the bottom status band widget.

    Owns no layout row: the band widget (a ``Static`` docked at the bottom of
    the input dock) is repainted through :meth:`refresh`. Call :meth:`update`
    whenever a segment changes; call :meth:`refresh` after resizes so the
    overflow ladder follows the new width (one width model:
    ``rich.cells.cell_len``).

    Every :meth:`update` parameter is optional and ``None`` means "leave this
    segment unchanged" — callers assemble partial updates from whatever the
    event they are handling actually knows.
    """

    def __init__(self, dock: Static, *, clock: Callable[[], float] = time.monotonic) -> None:
        # A `Static`, not a bare `Widget`: the band is repainted by handing
        # it a rich renderable, which only content widgets accept.
        self._dock = dock
        # Injected so duration is testable without sleeping. Monotonic, not
        # wall clock: a duration that jumps when the system clock is adjusted
        # is worse than no duration.
        self._clock = clock
        self._model_label: str = ""
        self._effort: str = ""
        self._cwd: str = ""
        self._context_tokens: int = 0
        self._context_window: int = 0
        self._subagents: int = 0
        self._streaming: bool = False
        self._cost: str = ""
        self._conversation_name: str = ""
        # Cumulative ACTIVE processing time: the sum of turn durations, not
        # wall clock since launch. A session left open over lunch has not
        # been working for two hours, and reporting that it has makes the
        # number useless for judging what a task actually cost.
        self._active_seconds: float = 0.0
        self._turn_started_at: float | None = None
        self._spinner_index: int = 0
        self._spinner_timer = None

    # -- segment setters ----------------------------------------------------
    def update(
        self,
        *,
        model_label: str | None = None,
        effort: str | None = None,
        cwd: str | None = None,
        context_tokens: int | None = None,
        context_window: int | None = None,
        subagents: int | None = None,
        streaming: bool | None = None,
        cost: str | None = None,
        conversation_name: str | None = None,
    ) -> None:
        """Update any subset of segments and repaint the band."""
        if model_label is not None:
            self._model_label = model_label
        if effort is not None:
            self._effort = effort
        if cwd is not None:
            self._cwd = cwd
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if context_window is not None:
            self._context_window = context_window
        if subagents is not None:
            self._subagents = subagents
        if cost is not None:
            self._cost = cost
        if conversation_name is not None:
            self._conversation_name = conversation_name
        if streaming is not None and streaming != self._streaming:
            self._streaming = streaming
            self._mark_turn_boundary(streaming)
            self._sync_spinner_timer()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the band content, truncated to the dock's inner width."""
        width = max(self._dock.size.width, 10)
        self._dock.update(self._render(width))

    def dispose(self) -> None:
        """Stop the spinner timer (idempotent)."""
        self._stop_spinner()

    # -- duration ------------------------------------------------------------
    def _mark_turn_boundary(self, streaming: bool) -> None:
        """Start or bank the active-time clock on a streaming transition.

        Only called on an actual change (``update`` guards it), so a
        redundant ``streaming=False`` from the prompt worker's ``finally``
        cannot bank the same interval twice.
        """
        if streaming:
            self._turn_started_at = self._clock()
        elif self._turn_started_at is not None:
            self._active_seconds += self._clock() - self._turn_started_at
            self._turn_started_at = None

    def _elapsed(self) -> float:
        """Banked active time plus the turn in flight, so the segment ticks
        live off the spinner's repaint rather than needing its own timer."""
        if self._turn_started_at is None:
            return self._active_seconds
        return self._active_seconds + (self._clock() - self._turn_started_at)

    # -- rendering ----------------------------------------------------------
    def _render(self, width: int) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Separators must sit BELOW the things they separate or they read as
        # content: `faint` is the ramp step under `dim` and exists for
        # exactly this — the dots recede and the segments group themselves.
        seam = Style(color=theme_mod.semantic_color("faint"))
        accent = Style(color=theme_mod.semantic_color("accent"))

        dropped: set[str] = set()
        short: set[str] = set()
        # Bound before the loop only so the post-loop clip below is provably
        # reachable with a value. The ladder's first step is the literal
        # ``None`` (render everything), so this is always overwritten.
        row = Text()
        # Walk the ladder until the row fits. Building the row is a handful
        # of string joins and the ladder is ten steps, so the worst case is
        # eleven cheap rebuilds on a resize — far cheaper than the
        # alternative of measuring segments independently and getting the
        # separator arithmetic subtly wrong.
        for step in (None, *_DROP_LADDER):
            if step is not None:
                target = step.partition("shorten-")[2]
                (short if target else dropped).add(target or step)
            left = self._left_text(dropped, short, dim, muted, seam, accent)
            right = self._right_text(dropped, dim, seam)
            row = self._compose(left, right, width, dim)
            if cell_len(row.plain) <= width:
                return row
        # Even the irreducible band (glyph + spinner) overflows: clip it.
        # A terminal this narrow has bigger problems than a status line.
        row.truncate(width, overflow="ellipsis")
        return row

    def _left_text(
        self,
        dropped: set[str],
        short: set[str],
        dim: Style,
        muted: Style,
        seam: Style,
        accent: Style,
    ) -> Text:
        """Brand glyph · provider/model · effort · cwd (+ working indicator)."""
        left = Text()
        left.append(BRAND_GLYPH + " ", style=accent)
        parts: list[tuple[str, Style]] = []
        if self._model_label and "model" not in dropped:
            parts.append((format_model_label(self._model_label, short="model" in short), muted))
        if self._effort and "effort" not in dropped:
            parts.append((self._effort, dim))
        if self._cwd and "cwd" not in dropped:
            rendered = format_cwd(self._cwd, short="cwd" in short)
            if rendered:
                parts.append((rendered, dim))
        for index, (text, style) in enumerate(parts):
            if index:
                left.append(_SEPARATOR, style=seam)
            left.append(text, style=style)
        if self._streaming:
            # The aggregate working LINE (WorkingBlock) carries the shimmer;
            # the band keeps a quiet activity glyph so a still frame still
            # reads "live" (D26). With shimmer off, that line is static too,
            # so the band spells the state out rather than relying on a
            # glyph the eye may read as decoration.
            from local_operator.tui.shimmer import shimmer_enabled

            if parts:
                left.append(_SEPARATOR, style=seam)
            left.append(_SPINNER_FRAMES[self._spinner_index], style=dim)
            if not shimmer_enabled():
                left.append(" working", style=dim)
        return left

    def _right_text(self, dropped: set[str], dim: Style, seam: Style) -> Text:
        """Subagents · context usage · cost · duration · conversation name."""
        parts: list[str] = []
        if "subagents" not in dropped:
            agents = format_agents(self._subagents)
            if agents:
                parts.append(agents)
        if "context" not in dropped:
            usage = format_context_usage(self._context_tokens, self._context_window)
            if usage:
                parts.append(usage)
        if self._cost and "cost" not in dropped:
            parts.append(self._cost)
        if "duration" not in dropped:
            elapsed = self._elapsed()
            # Zero means "nothing has run yet" — an idle band should not
            # claim a 0s task. Any real turn banks at least a few ms.
            if elapsed > 0:
                parts.append(format_duration(elapsed))
        if self._conversation_name and "name" not in dropped:
            parts.append(self._conversation_name)

        right = Text()
        for index, text in enumerate(parts):
            if index:
                right.append(_SEPARATOR, style=seam)
            right.append(text, style=dim)
        return right

    def _compose(self, left: Text, right: Text, width: int, dim: Style) -> Text:
        """Left group, filler, right group — right-aligned to the band edge."""
        gap = max(1, width - cell_len(left.plain) - cell_len(right.plain))
        row = Text()
        row.append_text(left)
        row.append(" " * gap, style=dim)
        row.append_text(right)
        return row

    def render_text(self, width: int) -> Text:
        """Public render entry (tests): segments joined, fitted to width."""
        return self._render(width)

    # -- spinner ------------------------------------------------------------
    def _sync_spinner_timer(self) -> None:
        if self._streaming and self._spinner_timer is None:
            self._spinner_timer = self._dock.set_interval(
                _SPINNER_INTERVAL_S, self._advance_spinner
            )
        elif not self._streaming and self._spinner_timer is not None:
            self._stop_spinner()

    def _advance_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self.refresh()

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None
