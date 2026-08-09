"""Tool execution card — ONE LINE PER ACTION, rendered as a FILLED card.

The character refinement (docs/REWRITE.md §D): tool calls are subtle
background-filled cards — one elevation step brighter than the ground
(kit ``surface`` on ``bg``), full-width single rows with 1-cell inner
padding, NO border. Elevation is a background step, never a shadow or line.

Row anatomy — one COLUMN per field, so a ledger is scanned down, not read
across::

     bash     pytest -q                                     ✓  0.4s
     edit     tui/theme.py                           +12 -3 ✓  0.1s
     bash     false                          exit status 1 ✗  0.2s
     grep     needle                           interrupted ⊘  5.0s

- a per-TOOL icon (``local_operator.tui.glyphs``) leads the row: shape
  before name, so the one ``edit`` in a run of ``read``s is found without
  parsing ten words. It doubles as the QUIET STATIC running marker (D25 —
  the shimmer rides the aggregate working line, not individual rows),
  painted in the accent while running so a still frame reads "live" (D26
  fallback). Exactly one cell wide, enforced in ``glyphs`` against the same
  ``cell_len`` this row's arithmetic uses.
- tool NAME ljust'd into an 8-cell column spine (D7). The column paid for
  the icon: the glyph now carries identity at a glance, so the name needs
  less room than it did when it was the only thing telling one row from the
  next. The name and the summary each carry a two-step tint ramp: while the
  tool is LIVE the name is the string green and the summary is ``muted``;
  once it settles both drop one step (``muted``/``dim``) so the running row
  is always the brightest thing in the transcript.
- command/summary ellipsized to the remaining budget, with absolute paths
  compacted against the cwd/home so a deep path does not eat the row
- diff counters for file-mutating tools: ``+N`` in the success tint and
  ``-N`` in the danger tint, rendered ONLY when the tool result actually
  reported them (an unknown count renders nothing — never ``+0 -0``)
- status right-aligned (D6): EMPTY while running — no trailing glyph, the
  column stays clear until the duration lands (D28); ``✓ duration`` all
  dim on success (D12: only failure gets color); ``✗ error`` danger with
  the duration dim as a second run (D13); ``⊘ interrupted`` dim when the
  turn ended before completion (TUI-019). The duration is right-justified
  into a fixed :data:`DURATION_COL` so the outcome glyph sits in a STABLE
  column — otherwise ``✓ 0.4s`` and ``✓ 12.3s`` put the glyph one cell
  apart, and the pass/fail scan that right-alignment exists to serve has to
  hunt for the answer.
- the reason in front of the glyph shortens differently per state. An error
  message is CONTENT and truncates — ``internal error: worker di…`` still
  names the failure. ``interrupted`` is a constant restating ``⊘``, so it is
  all-or-nothing: truncated to ``inte…`` it was byte-identical to a real
  failure truncated to ``inte…``, in the leftmost and longest column, which
  is the one the eye reaches first. Below the width that holds it whole the
  word goes and the glyph carries the state alone.
- the outcome glyph is what a still, COLOURLESS frame reads: ``✓``/``✗``/
  ``⊘`` separate success, failure and interruption without a single colour,
  and their absence is what says "still running". Tint is a second channel
  on top of the glyph, never the only one.
- an ``⟨expand⟩`` hint trailing the summary — the whole ROW is the click
  target AND a focus stop, and activating it (mouse click, or Enter/Space
  with the row focused) reveals the tool's full output indented beneath the
  summary; activating again collapses back to exactly one row. The hint
  appears while the pointer is on the row or the row holds focus, and flips
  to ``⟨collapse⟩`` when open, so a settled transcript reads as content
  rather than as a wall of controls.
- a row with NOTHING to reveal answers anyway: activating it flashes
  :data:`NO_OUTPUT_NOTICE` (or :data:`RUNNING_NOTICE`) in the same slot for
  :data:`NOTICE_SECONDS`. Silence was read in the field as the app being
  broken — "when I click to expand these lines, nothing happens" — and an
  affordance that sometimes does nothing has to say which time this is.
- the two share a slot but NOT a priority. The hint is an offer: chrome,
  dropped at the D8 summary floor. The notice is the answer to a keystroke
  the user just pressed, so it outranks the summary and shortens along
  :data:`NOTICE_LADDER` — phrase, then ``⟨∅⟩``/``⟨⋯⟩`` — rather than
  vanishing. Shedding them together is how the reported bug came back at 46
  columns, where activating an inert row repainted identical bytes.

State also reaches the ground: the card's background is ``raised`` while
running, ``surface`` once it settles, and the warm ``tint-danger`` ground
when it failed. Outcome is legible from the row's fill alone, at a glance,
without reading a single glyph.

Widths measured through ``rich.cells.cell_len`` only (one width model).
"""

from __future__ import annotations

import os
import time
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.events import Key
from textual.timer import Timer

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.glyphs import display_name, tool_icon
from local_operator.tui.widgets.transcript import (
    TOOL_NAME_COL,
    TOOL_NAME_COL_MAX,
    TranscriptBlock,
    TranscriptView,
)

#: Control-sequence stripping lives in `local_operator.ansi` because the
#: headless renderer needs the identical behaviour and must not import a
#: Textual widget module to get it. Aliased here so the call sites read local.
_strip_control_sequences = strip_control_sequences


#: Outcome glyphs. Plain unicode, NOT Nerd Font: these three are the only
#: thing distinguishing success from failure from interruption in a still,
#: colourless frame, so they may not depend on the user having a patched
#: font. The per-tool ICON at the head of the row may (see `tui.glyphs`) —
#: a wrench where a magnifier belongs loses a scanning aid; a missing ✗
#: loses the answer.
ICON_SUCCESS = "✓"
ICON_ERROR = "✗"
ICON_INTERRUPTED = "⊘"
#: Expansion affordance trailing the summary. Both spellings are the same
#: click target; only the label flips so the row always says what a click does.
EXPAND_HINT = "⟨expand⟩"
COLLAPSE_HINT = "⟨collapse⟩"

#: Answers given in the hint slot when the affordance has nothing to open.
#: An expander that silently ignores half its activations is indistinguishable
#: from a frozen app, which is precisely how it was reported.
#:
#: Bracketed in the SAME idiom as the expand affordance they stand in for.
#: Bare, they read as summary text — `todo     a no output` is one space and
#: one colour step from the argument beside it, and with no colour at all it
#: is just `a no output`. This slot is the direct remedy for "nothing
#: happens when I click", so it has to be the least ambiguous thing on the
#: row, and the app already owns a bracket for "this is chrome, not content".
NO_OUTPUT_NOTICE = "⟨no output⟩"
RUNNING_NOTICE = "⟨still running⟩"

#: The same two answers at three cells, for a row too narrow to spell them.
#: The feedback has to survive FURTHER DOWN the width ladder than the expand
#: affordance does, and the reason is the whole point of the slot: the offer
#: is decoration a tight row can drop, but an activation that leaves a
#: byte-identical frame IS the reported bug ("when I click to expand these
#: lines, nothing happens") reappearing at 46 columns. So the phrase shortens
#: to its glyph rather than vanishing: ∅ for "there is nothing here", ⋯ for
#: "not yet". Brackets are kept at every rung so the slot never reads as
#: content, and both inner glyphs are measured single-width below.
TERSE_NO_OUTPUT_NOTICE = "⟨∅⟩"
TERSE_RUNNING_NOTICE = "⟨⋯⟩"
#: Full phrase -> glyph -> nothing, per notice. The row walks this in order
#: and takes the first rung that fits.
NOTICE_LADDER: dict[str, tuple[str, ...]] = {
    NO_OUTPUT_NOTICE: (NO_OUTPUT_NOTICE, TERSE_NO_OUTPUT_NOTICE),
    RUNNING_NOTICE: (RUNNING_NOTICE, TERSE_RUNNING_NOTICE),
}
#: How long that answer stays on the row. Long enough to read at a glance,
#: short enough that it is gone before the eye returns — it is feedback for
#: a keystroke, not a state the row is in.
NOTICE_SECONDS = 2.0

#: Tool-name column FLOOR (D7: a spine for the eye to scan names). Eight, not
#: ten: the icon column now carries identity, and the two cells it costs come
#: out of the name rather than out of the summary. The transcript owns the
#: shared value and may widen it — see :attr:`TranscriptView.tool_name_col`.
NAME_COL = TOOL_NAME_COL
#: The ceiling that widening respects. Past roughly this width the eye stops
#: scanning a column of names and starts reading a list of them.
NAME_COL_MAX = TOOL_NAME_COL_MAX
#: Below this ROW width the column never grows: at 60 columns the summary needs
#: those cells more than the name does. Measured against the card's own inner
#: width — the value `_build_row` works in, already reduced by the transcript's
#: padding — not the terminal's, so the growth engages at a frame around six
#: cells wider. Named for what it is compared against, because a threshold that
#: does not mean what it says is what makes the next measurement disagree.
NAME_GROWTH_MIN_ROW = 70
#: Right-justification width for the duration, so the outcome glyph in front
#: of it lands in the same column whether the tool took 0.4s or 12.3s. Five
#: cells covers every duration the format produces up to ``9999s``.
DURATION_COL = 5
#: Minimum summary budget before we drop the expand hint (D8 floor).
_SUMMARY_FLOOR = 16
#: Indent of the expanded output block, aligned under the tool name column.
OUTPUT_INDENT = 2
#: Expanded output is capped: a 20k-line bash dump would otherwise turn the
#: transcript into a scroll trap. The head is kept (it carries the command's
#: framing) and the remainder is announced on a dim marker row.
EXPAND_MAX_LINES = 40
#: Search details are persisted separately from the token-bounded model text.
#: The expansion gets enough of each provider snippet to identify the page,
#: while the source URL remains the path to the complete document.
SEARCH_EXPANDED_SNIPPET_CHARS = 360
#: Last-resort width for a card built before it has been laid out. Content
#: rendered at this width is corrected by the first resize.
FALLBACK_WIDTH = 80

#: Argument names that identify WHAT a tool is acting on, as opposed to the
#: payload it is acting WITH. A summary built from these stays about the
#: subject; one built from ``content`` or ``new_text`` is a preview of a
#: file body squeezed into forty cells, which tells the reader nothing.
IDENTITY_ARGS = frozenset(
    {
        "command",
        "path",
        "file_path",
        "url",
        "pattern",
        "query",
        "name",
        "target",
        "message",
    }
)


def _format_bytes(count: int) -> str:
    """A byte count at a glance: `812 B`, `12.4 KB`, `1.2 MB`.

    One decimal above a kilobyte, because the point of the number is that it
    MOVES — a counter that ticks tells the user the model is still dictating,
    which is the whole reason the composing row exists.
    """
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


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


def truncate_cells(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncate ``text`` to at most ``width`` cells (one width model).

    Measurement and slicing both go through ``rich.cells.cell_len``; mixing
    ``len()`` with cell width is a crash class the history documents.
    """
    if width <= 0:
        return ""
    if cell_len(text) <= width:
        return text
    if width <= cell_len(ellipsis):
        return ellipsis[:width]
    out: list[str] = []
    used = 0
    target = width - cell_len(ellipsis)
    for char in text:
        w = cell_len(char)
        if used + w > target:
            break
        out.append(char)
        used += w
    # `rstrip()` so the cut never lands as "word …". Without it the same message
    # truncates in two different typographic styles depending on the terminal
    # width — tight at one width, with a stray space at the next — which is most
    # visible on the error row, whose whole job is to read cleanly beside the
    # outcome glyph. Every caller benefits (band model label, picker
    # descriptions, tool arguments), and the result is never WIDER than before.
    return "".join(out).rstrip() + ellipsis


def compact_path(text: str) -> str:
    """Shrink an absolute path against the cwd, then the home directory.

    Only whole-token absolute paths are rewritten: a sentence that merely
    mentions a slash keeps its exact wording. Deep absolute paths otherwise
    consume the entire summary budget and the informative tail — the file
    actually being touched — is the part that gets ellipsized away.
    """
    if not text.startswith("/") or " " in text:
        return text
    try:
        cwd = os.getcwd()
    except OSError:  # cwd deleted underneath us: fall back to the raw path
        cwd = ""
    if cwd and cwd != "/" and text.startswith(cwd + "/"):
        return text[len(cwd) + 1 :]
    home = os.path.expanduser("~")
    if home and home != "/" and text.startswith(home + "/"):
        return "~/" + text[len(home) + 1 :]
    return text


def _scalar_text(value: object) -> str:
    """One argument value flattened to a single compact line ("" if unusable)."""
    if isinstance(value, str):
        return compact_path(value.strip()).replace("\n", " ").strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _summary_from_args(tool_name: str, args: dict[str, object]) -> str:
    """One-line summary of WHAT the tool is acting on.

    Identity arguments win over payload arguments. A ``write`` call carries
    both ``path`` and ``content``; joining the first two scalars in argument
    order buries the filename under the first sixty characters of the file
    being written, which is the one thing the row is for. When no argument
    is recognisably an identity — an unknown or MCP-provided tool — the
    generic first-two-scalars scan still applies, in argument order.
    """
    parts = [
        text
        for key, value in args.items()
        if key in IDENTITY_ARGS and (text := _scalar_text(value))
    ]
    if not parts:
        parts = [text for value in args.values() if (text := _scalar_text(value))]
    return " ".join(parts[:2]) or tool_name


def _diff_counts(details: dict[str, Any] | None) -> tuple[int, int]:
    """``(added, removed)`` line counts from a tool result's ``details``.

    Unknown, malformed, or negative counts collapse to zero so the renderer
    can stay honest: a card only ever shows a counter it was actually told.
    ``bool`` is excluded explicitly — it is an ``int`` subclass in Python and
    ``details={"added": True}`` must not print ``+1``.
    """
    if not isinstance(details, dict):
        return (0, 0)

    def _count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return value if value > 0 else 0

    return (_count(details.get("added")), _count(details.get("removed")))


def _search_result_output(details: dict[str, Any] | None) -> list[str]:
    """Structured web-search rows: provider, page name, URL, and short snippet."""
    if not isinstance(details, dict):
        return []
    sources = details.get("sources")
    if not isinstance(sources, list) or not sources:
        return []

    provider = _strip_control_sequences(str(details.get("provider") or "search"))
    auth_mode = _strip_control_sequences(str(details.get("auth_mode") or ""))
    lines = [f"Provider: {provider}" + (f" ({auth_mode})" if auth_mode else ""), "Sources:"]
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        title = _strip_control_sequences(
            " ".join(str(source.get("title") or "Untitled result").split())
        )
        url = _strip_control_sequences(" ".join(str(source.get("url") or "").split()))
        snippet = _strip_control_sequences(" ".join(str(source.get("snippet") or "").split()))
        lines.append(f"{index}. {title}")
        if url:
            lines.append(f"   {url}")
        if snippet:
            if len(snippet) > SEARCH_EXPANDED_SNIPPET_CHARS:
                snippet = snippet[: SEARCH_EXPANDED_SNIPPET_CHARS - 1].rstrip() + "…"
            lines.append(f"   {snippet}")
    lines.append("Ask Operator to open result N with browser for the full page.")
    return lines


def _clamp_runs(runs: list[tuple[str, Style]], limit: int) -> list[tuple[str, Style]]:
    """Trim styled runs from the tail so their total is at most ``limit``.

    The last line of defence for the ONE-LINE guarantee: whatever a status
    segment wants to say, the row it lives on is finite.
    """
    if limit <= 0:
        return []
    out: list[tuple[str, Style]] = []
    used = 0
    for text, style in runs:
        room = limit - used
        if room <= 0:
            break
        if cell_len(text) > room:
            text = truncate_cells(text, room, ellipsis="")
            if not text:
                break
        out.append((text, style))
        used += cell_len(text)
    return out


def _row_text() -> Text:
    """A ``Text`` carrying wrap-suppression flags for the Rich-only paths.

    IMPORTANT, because the previous docstring claimed a guarantee this does
    not provide: Textual's ``Content.from_rich_text`` DISCARDS ``no_wrap`` and
    ``overflow`` when a Rich ``Text`` crosses into a widget, so these flags do
    nothing for the composed card. They are kept because they DO apply when
    the same row is measured or exported through Rich directly (tests, SVG
    export), and dropping them would change those paths.

    What actually holds the one-row rule, in order:

    1. ``ToolCard { height: 1 }`` in the stylesheet — the real enforcement. A
       collapsed card is one cell tall, so wrapped content is clipped, never
       reflowed. ``ToolCard.tool-expanded`` relaxes it to ``height: auto``
       precisely because expansion is the one case that may be taller.
    2. The status segment is hard-trimmed to ``width - 3`` in ``_build_row``,
       so no state's label can outgrow its column.
    3. Diff counters are dropped first when the cap bites, so the outcome
       glyph and duration always survive.

    Every segment is measured with ``rich.cells.cell_len``, never ``len``, so
    CJK, emoji and ZWJ sequences account correctly.
    """
    return Text(no_wrap=True, overflow="ellipsis")


class ToolCard(TranscriptBlock):
    """A tool execution: ONE row, on a state-tinted elevation step.

    Lifecycle: construct with ``tool_call_id``/``tool_name`` (running),
    :meth:`mark_done` on success, :meth:`mark_failed` on error,
    :meth:`mark_interrupted` when the turn ends first. Both terminal calls
    accept the tool's result text and ``details`` payload; that is what
    powers the diff counters and the click-to-expand output. Passing them is
    optional so a host that has not wired the result through still gets a
    correct — merely quieter — card.

    The card is a FOCUS STOP. Expansion used to be mouse-only, which made it
    invisible to anyone driving the app from the keyboard and unreachable in
    a terminal without mouse reporting; ``can_focus`` puts every row in the
    screen's tab order, so Shift+Tab out of the composer lands on the last
    action and Up/Down walk the ledger from there.
    """

    #: Adaptive spacing: every tool row takes a blank row above it, because
    #: each row is a separate action (see `transcript.needs_gap_above`).
    SPACING_KIND = "tool"
    LEDGER_ROW = True
    SPACING_AIRY = True

    #: The keyboard half of the expand affordance. Enter and Space both
    #: activate because both are what a user reaches for on a focused row and
    #: neither means anything else here. Up/Down move between ACTIONS rather
    #: than scrolling by a line: once focus is on a card the ledger is what
    #: the arrows are addressing, and the transcript keeps the scroll keys
    #: for when nothing in it is focused.
    #:
    #: Escape is deliberately absent — the app owns it as the "stop the turn"
    #: binding, and a second meaning for the key that aborts work is the last
    #: thing this row should introduce. With no binding here the key simply
    #: bubbles from a focused row to the app, which is the wanted precedence.
    BINDINGS = [
        Binding("enter", "activate", "Expand/collapse", show=False),
        Binding("space", "activate", "Expand/collapse", show=False),
        Binding("up", "focus_previous_action", "Previous action", show=False),
        Binding("down", "focus_next_action", "Next action", show=False),
    ]

    #: The keys above, flattened. ``on_key`` runs BEFORE Textual resolves a
    #: focused widget's bindings, so the typing passthrough has to step aside
    #: for the row's own keys explicitly: Space is both this row's toggle and
    #: a printable character, and without this it typed a space into the
    #: composer instead of expanding the row it was standing on. Derived from
    #: BINDINGS rather than restated, so a fifth key cannot drift out of sync.
    #: BINDINGS is typed as accepting bare tuples as well as Binding objects, so
    #: the key name is read through a narrowing step rather than assumed.
    _BOUND_KEYS = frozenset(
        key.strip()
        for binding in BINDINGS
        for key in (binding.key if isinstance(binding, Binding) else binding[0]).split(",")
    )

    can_focus = True

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, object] | None = None,
        intent: str | None = None,
    ) -> None:
        super().__init__()
        self.tool_call_id = tool_call_id
        # tool_name is MODEL-CONTROLLED: the loop takes it from the tool call
        # itself, so a hallucinated or injected name reaches the frame. It was
        # the one raw-text entry point the sanitisation pass missed, and it is
        # the worst one to miss — the name is rendered on every row, so an
        # erase-display in it clears the terminal without the tool even running.
        self.tool_name = _strip_control_sequences(tool_name)
        self.add_class("tool-card", "tool-running")
        # Sanitised at the boundary: args and intent can carry escapes too
        # (a bash command containing a colour code, an MCP tool's intent).
        self._summary = _strip_control_sequences(
            intent or _summary_from_args(tool_name, args or {})
        )
        self._state: str = "running"
        # Composing-row bookkeeping: bytes seen so far, when dictation started,
        # and the clock that keeps the row visibly alive through a provider's
        # silence. All three are None/0 for a row that never composed.
        self._compose_bytes: int = 0
        self._compose_facts: str = ""
        self._compose_started: float | None = None
        self._compose_timer: Timer | None = None
        self._duration: float | None = None
        self._error: str = ""
        self._started = time.monotonic()
        self._expanded = False
        self._hovered = False
        #: True while the row holds keyboard focus. Tracked separately from
        #: hover: both light the hint, but a pointer leaving the row must not
        #: put out a hint the keyboard is still standing on.
        self._focused = False
        #: One-shot answer occupying the hint slot ("" = nothing to say).
        self._notice = ""
        self._notice_timer: Any = None
        self._added = 0
        self._removed = 0
        #: Cleaned result lines, populated once the tool finishes.
        self._output: list[str] = []
        #: The write/edit tool's rendered unified diff (``details["diff"]``),
        #: colourised at render time. ``None`` (not []) when the tool reported
        #: no diff — a read or bash card expands to its raw output instead.
        self._diff: list[str] | None = None
        #: Rows the card currently occupies (1 collapsed, N expanded).
        self._row_count = 1
        self._refresh_row()

    # -- lifecycle ----------------------------------------------------------
    def mark_done(self, result_text: str = "", details: dict[str, Any] | None = None) -> None:
        """Record success with elapsed duration; the row goes quiet."""
        self._stop_composing()
        self._duration = time.monotonic() - self._started
        self._state = "success"
        self._absorb_result(result_text, details)
        self.remove_class("tool-running")
        self.add_class("tool-success")
        self._refresh_row()
        self.finalize()

    def mark_failed(
        self, error: str, result_text: str = "", details: dict[str, Any] | None = None
    ) -> None:
        """Record failure with a ONE-line error message.

        ``result_text`` defaults to the error itself: a failed tool's full
        message is frequently a stack trace or a multi-line diagnostic, and
        that is exactly what the expansion exists to show.
        """
        self._duration = time.monotonic() - self._started
        self._state = "error"
        self._error = _strip_control_sequences(" ".join(error.split())) or "error"
        self._absorb_result(result_text or error, details)
        self.remove_class("tool-running")
        self.add_class("tool-error")
        self._refresh_row()
        self.finalize()

    def mark_interrupted(self) -> None:
        """Turn ended before this tool completed: dim 'interrupted' state."""
        was_composing = self._state == "composing"
        self._stop_composing()
        if was_composing:
            # The call was never sent, so the row must stop saying it is being
            # written. It keeps the size as a record of how far the model got.
            size = _format_bytes(self._compose_bytes) if self._compose_bytes else "nothing"
            # Facts first, and `_compose_facts` set so the label-shed ladder can
            # reach this row too: gated on the composing STATE, the ladder built
            # for the live row could not touch the record it turns into, and
            # three materially different interrupted rows painted identically
            # from 40 columns down.
            self._compose_facts = f"{size} composed"
            self._summary = f"never sent · {self._compose_facts}"
        self._duration = time.monotonic() - self._started
        self._state = "interrupted"
        self.remove_class("tool-running")
        self.add_class("tool-interrupted")
        self._refresh_row()
        self.finalize()

    def set_composing(self, argument_bytes: int, tool_name: str = "") -> None:
        """Show that the model is still WRITING this call's arguments.

        A separate state from `running`, because nothing has run: the row must
        not claim a tool is executing while the request is still being dictated.

        It shows BOTH what has arrived and how long it has been going, and the
        second half is not decoration. Providers pause mid-call — measured on a
        real Anthropic stream, a `write` block opened and then sent nothing for
        eighty seconds before delivering fourteen kilobytes in under a second.
        A byte count alone is static text through the whole of that pause, which
        is precisely the frame a user reads as a hung agent. A clock that ticks
        says "still going" during exactly the silence that needs it said.
        """
        if self._state not in ("composing", "running"):
            return
        entering = self._state != "composing"
        self._state = "composing"
        # The name can arrive in FRAGMENTS: the first announcement fires on the
        # first piece so the row appears at once, so a provider that splits
        # `write` sends `wr` and then `write`. Following it keeps the visible row
        # and its icon honest for the whole dictation, not just once the call
        # starts running.
        renamed = bool(tool_name and tool_name != self.tool_name)
        if renamed:
            self.tool_name = _strip_control_sequences(tool_name)
        # Two reasons the shared column may have moved: this row's NAME changed,
        # or the row just stopped contributing a name at all by entering
        # `composing`. The second matters because the card is mounted as
        # `running` and flipped here — with the same name it was constructed
        # with, so a rename-only check never fired and a dictated name kept the
        # column it had already widened.
        if renamed or entering:
            parent = self.parent
            if isinstance(parent, TranscriptView):
                parent.invalidate_name_col()
        self._compose_bytes = argument_bytes
        if self._compose_started is None:
            self._compose_started = time.monotonic()
            # One second: fast enough to read as alive, slow enough that the
            # repaint cost is nothing next to the stream it is reporting on.
            self._compose_timer = self.set_interval(1.0, self._tick_composing)
        self._render_composing()

    def _tick_composing(self) -> None:
        """Repaint the elapsed half of a composing row (no new bytes needed)."""
        if self._state == "composing":
            self._render_composing()

    def _render_composing(self) -> None:
        started = self._compose_started or time.monotonic()
        elapsed = max(0, int(time.monotonic() - started))
        clock = format_duration(elapsed)
        # The FACTS lead and the label sheds whole, the same shape this file
        # already uses for a tool summary. Boilerplate-first, `composing…` was
        # protected by the truncation while the two things that move were cut:
        # below 39 columns the row stopped changing at all, and two calls
        # dictating 12.4 KB and 61 B painted identically.
        #
        # The size joins only once there IS one. Providers commonly open the
        # call and deliver its arguments in one late burst, so a leading `0 B`
        # sat on screen for two minutes on a measured run — a number that never
        # moves reads as a stuck counter, which is what this row exists to fix.
        #
        # `composing`, not `writing`: `write` is a TOOL in this app, so
        # `write  writing the call…` read as a stutter and `read  writing the
        # call…` read as wrong.
        facts = f"{_format_bytes(self._compose_bytes)} · {clock}" if self._compose_bytes else clock
        self._compose_facts = facts
        self._summary = f"composing… {facts}"
        self._refresh_row()

    def _stop_composing(self) -> None:
        """Retire the clock; the row is about to become something else."""
        if self._compose_timer is not None:
            self._compose_timer.stop()
            self._compose_timer = None

    def begin_running(
        self, tool_name: str, args: dict[str, object] | None, intent: str | None
    ) -> None:
        """Adopt a composing row as the real execution of the call it announced.

        The same widget rather than a fresh one: the composing row already sits
        in the transcript in the right place, and replacing it would make the
        ledger flicker a row out and an identical row back in at the moment the
        call finally starts.
        """
        self._stop_composing()
        # The name comes from the EXECUTION, not from the announcement. The first
        # compose event fires on the first name fragment — deliberately, so the
        # row appears immediately — and a provider that splits `write` into `wr`
        # and `ite` would otherwise leave `wr` on the settled row forever, in the
        # ledger, on the icon, and in the summary built from it.
        self.tool_name = _strip_control_sequences(tool_name)
        # The duration clock RESTARTS here. `_started` was set when the row was
        # mounted, which for an adopted row is when the model began dictating —
        # so a `write` that executed in 0.1s settled as `✓ 2.4s`, and on the
        # reported 1m41s case would have read `✓ 101s`. Two receipts on one
        # ledger would then be measuring different things with no way to tell
        # which from the row.
        self._started = time.monotonic()
        self._state = "running"
        # The same construction the constructor uses, so an adopted row is
        # byte-identical to one that had never been a composing row.
        self._summary = _strip_control_sequences(
            intent or _summary_from_args(self.tool_name, args or {})
        )
        self._refresh_row()

    def set_partial_detail(self, detail: str) -> None:
        """Replace the running summary with a streaming partial result line."""
        if self._state != "running":
            return
        cleaned = _strip_control_sequences(" ".join(detail.split()))
        if cleaned:
            self._summary = cleaned
            self._refresh_row()

    def _absorb_result(self, result_text: str, details: dict[str, Any] | None) -> None:
        """Capture the expansion payload and diff counters.

        Web search is rendered from structured details, not the model-facing
        text: the latter is deliberately token-capped, while the expansion
        must reliably retain every candidate's page name, URL, and snippet.
        Write/edit tools prefer their rendered diff; all other tools expand to
        cleaned result text.
        """
        self._added, self._removed = _diff_counts(details)
        search_output = (
            _search_result_output(details) if self.tool_name.lower() == "web_search" else []
        )
        self._output = search_output or self._clean_output(result_text)
        diff = details.get("diff") if isinstance(details, dict) else None
        if isinstance(diff, list) and diff:
            self._diff = [str(line) for line in diff]
        else:
            self._diff = None

    def _clean_output(self, result_text: str) -> list[str]:
        """Normalise the result into displayable rows (empty = nothing to show).

        A single-line result that merely repeats the summary carries no new
        information, so the card stays inert rather than advertising an
        expansion that reveals what is already on screen.
        """
        if not result_text:
            return []
        lines = [
            _strip_control_sequences(line.rstrip())
            for line in result_text.expandtabs(4).splitlines()
        ]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if not lines:
            return []
        if len(lines) == 1 and lines[0].strip() == self._summary.strip():
            return []
        return lines

    # -- expansion ----------------------------------------------------------
    def can_expand(self) -> bool:
        """True when the card holds output the one-line summary cannot show.

        Either the plain result output, or a write/edit diff (which can be
        present on its own — a new-file write's summary line is one sentence
        while its diff is the whole file).
        """
        return bool(self._output) or bool(self._diff)

    @property
    def expanded(self) -> bool:
        """True while the full output is revealed beneath the summary row."""
        return self._expanded

    def toggle_expanded(self) -> bool:
        """Flip the expansion (no-op with nothing to show); returns the state.

        Also nudges the transcript to re-decide the gap below this card: a
        card that just grew from one row to twenty may change what the block
        under it needs, and only the container can answer that.
        """
        if not self.can_expand():
            return self._expanded
        self._expanded = not self._expanded
        self.set_class(self._expanded, "tool-expanded")
        self._clear_notice(repaint=False)
        self._refresh_row()
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_after(self)
        return self._expanded

    def activate(self) -> bool:
        """Run the row's one action; returns True when it expanded/collapsed.

        The SINGLE entry point behind both the mouse and the keyboard, so the
        two can never drift into answering a click and ignoring a keystroke.
        With nothing to reveal it flashes an answer instead of returning
        silently: an affordance the user has already been told about (the row
        lights, the hint slot exists) must never absorb an activation without
        a visible consequence.
        """
        if not self.can_expand():
            self._flash_notice()
            return False
        self.toggle_expanded()
        return True

    def action_activate(self) -> None:
        """Enter/Space on a focused row."""
        self.activate()

    def action_focus_next_action(self) -> None:
        """Down: the next focusable row, or out of the ledger entirely."""
        self._move_focus(1)

    def action_focus_previous_action(self) -> None:
        """Up: the previous focusable row, or out of the ledger entirely."""
        self._move_focus(-1)

    def _move_focus(self, delta: int) -> None:
        """Hand focus to a neighbouring card, else to the screen's tab order.

        Falling through at the ends is what makes the ledger something a
        keyboard can pass THROUGH rather than get stuck in: Down off the last
        action reaches the composer, Up off the first reaches the transcript
        itself, where the scroll keys mean what they always meant.
        """
        parent = self.parent
        if isinstance(parent, TranscriptView) and parent.focus_neighbour(self, delta):
            return
        screen = self.screen
        if delta > 0:
            screen.focus_next()
        else:
            screen.focus_previous()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Mouse affordance: the whole row is the target.

        The event is only stopped when the click actually toggled something.
        A click on an inert row still gets its answer, but it keeps bubbling
        so the transcript's own click handling (selection, scroll anchoring)
        is not swallowed by a row that had nothing to do.
        """
        if self.activate():
            event.stop()

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Typing on a focused row goes to the COMPOSER, not into the void.

        The row is a focus stop so the keyboard can reach the expander — but
        it is not somewhere to type, and a transcript that silently swallows
        a sentence is a worse trap than one that could never be focused. The
        app has exactly one text input, so any printable key is unambiguous:
        hand it the focus and re-post the keystroke there, and the user never
        has to discover that a row had focus at all.

        This runs BEFORE :attr:`BINDINGS` — Textual dispatches the focused
        widget's message handlers first and only then resolves its bindings —
        so the row's own keys are excluded here by hand. That is not a
        formality: Space is a printable character, and without the exclusion
        it typed a space into the composer instead of expanding the row the
        user was standing on.

        A FRESH ``Key`` is posted rather than the original: the event that
        reached this handler is already part-way through Textual's dispatch
        (bubbling, default-handling flags) and re-delivering it would be
        re-entering a lifecycle it has half finished.
        """
        if event.key in self._BOUND_KEYS or not event.is_printable:
            return
        composer = self._composer()
        if composer is None:
            return
        composer.focus()
        composer.post_message(Key(event.key, event.character))
        event.stop()
        event.prevent_default()

    def _composer(self):  # type: ignore[no-untyped-def]
        """The app's one text input, or None when there is not one.

        Imported lazily and queried defensively: the card is mounted in
        harnesses that host a transcript and nothing else, and a missing
        composer there must degrade to "the key does nothing" rather than
        raise out of a key handler.
        """
        from local_operator.tui.widgets.editor import Editor

        try:
            return self.app.query_one(Editor)
        except Exception:
            return None

    def on_enter(self, event) -> None:  # type: ignore[no-untyped-def]
        """Pointer over an expandable row: light the hint up to `dim`."""
        self._set_hovered(True)

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        """Pointer gone: the hint goes out again, unless focus is holding it."""
        self._set_hovered(False)

    def on_focus(self, event) -> None:  # type: ignore[no-untyped-def]
        """Keyboard on the row: show what Enter would do."""
        self._set_focused(True)

    def on_blur(self, event) -> None:  # type: ignore[no-untyped-def]
        """Focus elsewhere: the offer goes quiet, and so does any notice."""
        self._set_focused(False)

    def _set_hovered(self, hovered: bool) -> None:
        """Repaint only when the hover state actually changes something.

        A row with nothing to expand shows no hint, so hovering it costs
        nothing — the transcript is a long list and the pointer crosses a
        lot of rows on the way anywhere. The focus check is what keeps the
        pointer from erasing a hint the keyboard put there.
        """
        if hovered == self._hovered or not self.can_expand():
            self._hovered = hovered
            return
        self._hovered = hovered
        if not self._focused:
            self._refresh_row()

    def _set_focused(self, focused: bool) -> None:
        """Track keyboard focus; repaint only when it changes what the row says."""
        if focused == self._focused:
            return
        self._focused = focused
        had_notice = bool(self._notice)
        if not focused:
            # A notice is the answer to an activation on THIS row. Carrying it
            # past the row losing focus would leave "no output" sitting on a
            # card the user has moved away from, reading as a permanent state.
            self._clear_notice(repaint=False)
        if self.can_expand() or had_notice:
            self._refresh_row()

    def _flash_notice(self) -> None:
        """Put the inert-row answer in the hint slot for a couple of seconds.

        A RUNNING tool has no output yet but will; a settled one never will.
        Saying which is the difference between "wait" and "there is nothing
        here", and the row is the only place the user is looking.
        """
        live = self._state in ("running", "composing")
        self._notice = RUNNING_NOTICE if live else NO_OUTPUT_NOTICE
        self._refresh_row()
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if self.is_running:
            self._notice_timer = self.set_timer(NOTICE_SECONDS, self._clear_notice)
        # A card with no running message pump (built but not yet mounted, or a
        # unit test holding one directly) has no clock to schedule against, and
        # `set_timer` would leave an unawaited coroutine behind. The notice then
        # simply persists until the next repaint — the right degradation for a
        # purely cosmetic timer, and never a crash.

    def _clear_notice(self, repaint: bool = True) -> None:
        """Retire the one-shot answer (no-op when there is none)."""
        if self._notice_timer is not None:
            self._notice_timer.stop()
            self._notice_timer = None
        if not self._notice:
            return
        self._notice = ""
        if repaint:
            self._refresh_row()

    # -- resize (TUI-017: rebuild the row when the width changes) -----------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._refresh_row()

    # -- rendering ----------------------------------------------------------
    def refresh_row(self) -> None:
        """Repaint at the current width — the ledger's shared column moved.

        Public because the transcript owns the name column and has to be able to
        say "re-render, the spine changed"; everything else about the row is the
        card's own business.
        """
        self._refresh_row()

    def _refresh_row(self) -> None:
        """Rebuild the card at its OWN width (D3).

        Width resolution walks from the most authoritative source down: the
        widget's laid-out size, its container, the app console, and only
        then :data:`FALLBACK_WIDTH`. Reaching the last step means there is
        no app to paint into yet, so the content is measured but not
        applied: ``_row_count`` — which the spacing and scroll accounting
        both read — stays truthful, and ``on_resize`` paints the real thing
        the moment there is a real width.

        Finalization is bypassed deliberately: a resize or an expand must be
        able to re-fit a settled card, and the content it produces is a pure
        function of the card's state, never new history.
        """
        container = getattr(self, "container_size", None)
        width = self.size.width or (container.width if container else 0)
        detached = False
        if width <= 0:
            try:
                width = self.app.console.width
            except Exception:
                width = FALLBACK_WIDTH
                detached = True
        content = self._build_content(width)
        self._row_count = max(1, len(content.plain.splitlines()))
        if detached:
            return
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(content)
        finally:
            self._finalized = was_finalized

    def _build_content(self, width: int) -> Text:
        """The card: the one-row summary, plus the output when expanded."""
        row = self._build_row(width)
        if not self._expanded:
            return row
        if self._diff:
            self._append_diff_body(row, width)
        elif self.tool_name.lower() == "web_search" and self._output:
            self._append_search_body(row, width)
        elif self._output:
            self._append_output_body(row, width)
        return row

    def _append_output_body(self, row: Text, width: int) -> None:
        """The plain-result expansion (bash/read/etc.): one line per row.

        The output block reuses the card's own inner padding budget and
        truncates per line: one output line is one row, so the expanded
        height is exactly what the marker promises and never reflows.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        body = Style(color=theme_mod.semantic_color("danger")) if self._state == "error" else dim
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        shown = self._output[:EXPAND_MAX_LINES]
        for line in shown:
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=body)
        hidden = len(self._output) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _append_search_body(self, row: Text, width: int) -> None:
        """Search expansion hierarchy: titles lead, URLs signal, snippets recede."""
        fg = Style(color=theme_mod.semantic_color("fg"), bold=True)
        signal = Style(color=theme_mod.semantic_color("signal"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        shown = self._output[:EXPAND_MAX_LINES]
        for line in shown:
            stripped = line.strip()
            if stripped.startswith(("http://", "https://")):
                ink = signal
            elif stripped[:1].isdigit() and ". " in stripped:
                ink = fg
            elif stripped.startswith(("Provider:", "Sources:", "Ask Operator")):
                ink = muted
            else:
                ink = dim
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=ink)
        hidden = len(self._output) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more search line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _append_diff_body(self, row: Text, width: int) -> None:
        """The write/edit expansion: the unified diff, coloured by hunk line.

        ``+`` added in the success green, ``-`` removed in danger, ``@@``
        hunk markers and ``---/+++`` headers muted, context lines dim — the
        same ink law as the counters in the summary row, so the pill on the
        one-line summary and the expanded body tell the same story. Only the
        leading marker character is coloured here; the text rides the card's
        default so a coloured line never reads as a wall of tint.
        """
        success = Style(color=theme_mod.semantic_color("success"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        line_width = max(1, width - 2 - OUTPUT_INDENT)
        indent = " " * OUTPUT_INDENT
        diff = self._diff or []
        shown = diff[:EXPAND_MAX_LINES]
        for raw in shown:
            line = raw.rstrip()
            prefix = line[:1] if line else ""
            if line.startswith("---") or line.startswith("+++") or prefix == "@":
                ink = muted
            elif prefix == "+":
                ink = success
            elif prefix == "-":
                ink = danger
            else:
                ink = dim
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(line, line_width), style=ink)
        hidden = len(diff) - len(shown)
        if hidden > 0:
            marker = f"… {hidden} more diff line{'s' if hidden != 1 else ''}"
            row.append("\n" + indent, style=dim)
            row.append(truncate_cells(marker, line_width), style=dim)

    def _name_col(self, width: int) -> int:
        """The ledger's shared name column, in cells.

        Read from the transcript rather than fixed here, because the column is a
        SPINE: every card has to agree on it or the ledger stops being a column.
        The transcript widens it to fit the longest name on screen when the frame
        can afford it, so two MCP tools sharing a seven-character prefix stay
        distinguishable at the widths where there is obviously room.
        """
        parent = self.parent
        if isinstance(parent, TranscriptView) and width >= NAME_GROWTH_MIN_ROW:
            return parent.tool_name_col
        return NAME_COL

    def _build_row(self, width: int) -> Text:
        """The single summary row — the ONE-LINE guarantee lives here."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Two-step fade on settle: the live row keeps the string green on the
        # name and readable `muted` body text; a settled row drops both one
        # step so the running row is the brightest thing on screen.
        # Composing counts as live: the model is actively producing this call,
        # and a row that dimmed while its arguments streamed would read as a
        # finished action rather than as the one thing currently happening.
        running = self._state in ("running", "composing")
        name_style = Style(color=theme_mod.semantic_color("string")) if running else muted
        summary_style = muted if running else dim
        width = max(width - 2, 10)  # 1-cell inner padding each side (kit rule)

        # Status segment (right-aligned), capped at width // 3 (D8) and then
        # hard-clamped so no state can ever push the row past its card.
        status_cap = max(8, width // 3)
        status_runs = _clamp_runs(self._status_runs(status_cap), max(0, width - 3))
        status_cells = sum(cell_len(text) for text, _style in status_runs)

        # Prefix: icon + space + name column + space. The status segment and
        # both separator cells are part of the summary budget (TUI-018). The
        # name column is cell-width bound and ADAPTIVE: below the full column
        # width the name shrinks (truncated by CELL width — len() on a wide
        # CJK/emoji name would break the spine) before the row overflows its
        # card and clips the status off-screen.
        icon = tool_icon(self.tool_name)
        label = display_name(self.tool_name)
        name_budget = width - (2 + status_cells + 2)
        if name_budget < 2 and self._state in ("error", "interrupted"):
            # The status is the only segment carrying free text, so it is the
            # one that gives way first. A failing row used to be the ONLY row
            # in a narrow ledger to lose its name — its neighbours kept three
            # cells of identity while it rendered `<icon>  … ✗ 0.0s`, spending
            # a cell on an ellipsis to say nothing and dropping the one fact
            # that makes the row worth reading: WHICH tool failed.
            status_runs = _clamp_runs(self._status_runs(status_cap, terse=True), max(0, width - 3))
            status_cells = sum(cell_len(text) for text, _style in status_runs)
            name_budget = width - (2 + status_cells + 2)
        if name_budget < 2:
            # Too narrow for even a shrunken name: degrade to icon + status
            # so the outcome column survives. This is the last rung.
            row = _row_text()
            row.append(icon + " ", style=dim)
            if status_runs:
                used = cell_len(row.plain)
                pad = max(1, width - used - status_cells)
                row.append(" " * pad, style=dim)
                for text, style in status_runs:
                    row.append(text, style=style)
            return row

        name_col = min(self._name_col(width), name_budget)
        name = truncate_cells(label, name_col)
        name = name + " " * max(0, name_col - cell_len(name))
        prefix_cells = 2 + name_col + 1

        # The trailing slot holds ONE of two things, never both: the expand
        # affordance when there is output to reveal, or the one-shot answer
        # when the user activated a row that has none. They are mutually
        # exclusive by construction — a notice is only ever set on a row that
        # cannot expand — so they share a budget and a column.
        #
        # They do NOT share a priority. The affordance is an OFFER: chrome,
        # and the D8 floor drops it before it may eat into the summary. The
        # notice is an ANSWER to a keystroke the user just pressed, and it
        # outranks the summary — which on a no-output row is the least
        # interesting thing present, has not changed, and is still readable
        # underneath. Treating the two the same is how the ORIGINAL bug came
        # back: at 46 columns the floor dropped the notice too, so activating
        # an inert row left a byte-identical frame.
        slot = ""
        slot_token = "dim"
        remaining = max(0, width - prefix_cells - status_cells - 2)
        if self.can_expand():
            # Generic tool output stays quiet until the row is targeted. Search
            # sources are the primary result, not diagnostics, so their
            # disclosure remains visible at rest and in colorless terminals.
            if self.tool_name.lower() == "web_search" or self._hovered or self._focused:
                offer = COLLAPSE_HINT if self._expanded else EXPAND_HINT
                if remaining - (cell_len(offer) + 1) >= _SUMMARY_FLOOR:
                    slot = offer
        elif self._notice:
            # `muted`, one step brighter than the offer's `dim`: this is not
            # something the eye may skip. Walk the ladder — full phrase, then
            # the three-cell glyph — and take the first rung the row can hold
            # with nothing reserved for the summary. Only when even ⟨∅⟩ will
            # not fit does the answer go unsaid, and by then the row is down
            # to its icon and its outcome anyway.
            slot_token = "muted"
            for rung in NOTICE_LADDER.get(self._notice, (self._notice,)):
                if cell_len(rung) + 1 <= remaining:
                    slot = rung
                    break
        slot_cells = cell_len(slot) + 1 if slot else 0
        budget = max(0, remaining - slot_cells)
        if self._compose_facts:
            # The label is shed WHOLE before the facts are touched, the same
            # ladder this file uses for the key hints and the approval clause.
            # Truncating instead protected `composing…` — seventeen cells that
            # say nothing new next to a row that is already visibly live — while
            # cutting the byte count and the clock, which are the only two
            # things on the row that move.
            #
            # The trigger is the CARD's width, not the residual budget, because
            # the budget is not monotone in the width: the status segment's cap
            # (`max(8, width // 3)`) crosses the reason's length on the way down
            # and hands cells BACK, so a budget test shed the label at 66,
            # restored it at 62 and shed it again at 54. A width the full form
            # needs is a constant, so comparing against it can only flip once —
            # the same argument `_verbose_min_width` makes for the approval row.
            summary = self._summary
            if width < self._label_min_width() or cell_len(summary) > budget:
                summary = truncate_cells(self._compose_facts, budget)
        else:
            summary = truncate_cells(self._summary, budget)

        row = _row_text()
        # The icon carries the running state: accent while live (D26 — a still
        # frame must read "live" without the shimmer), dim once settled. This
        # is one of the five places in the app the accent green is spent.
        icon_style = Style(color=theme_mod.semantic_color("accent")) if running else dim
        row.append(icon + " ", style=icon_style)
        row.append(name, style=name_style)
        row.append(" ", style=dim)
        row.append(summary, style=summary_style)

        # ONE right-aligned tail: the slot and the status share a single pad,
        # so the slot is a COLUMN instead of trailing the summary wherever
        # that happens to end. Appended after the summary it landed at a
        # different cell on every row and slid as the summary truncated —
        # jogging left and right under the eye while the outcome beside it
        # was pinned precisely so that would not happen (D27).
        tail_cells = slot_cells + status_cells
        if tail_cells:
            used = cell_len(row.plain)
            row.append(" " * max(1, width - used - tail_cells), style=dim)
            if slot:
                row.append(slot, style=Style(color=theme_mod.semantic_color(slot_token)))
                row.append(" ", style=dim)
            for text, style in status_runs:
                row.append(text, style=style)
        return row

    @property
    def contributes_name(self) -> bool:
        """Whether this row's name may widen the ledger's shared column.

        A call still being DICTATED must not: the name is model-controlled and
        arrives in fragments, so one announced 200-character name took the column
        to its cap, shifted every settled receipt beside it, and kept the width
        after the row settled as `never sent`. The same argument the column already
        makes for a pending approval — a name earns the column when the call it
        names has actually started.
        """
        return self._state != "composing" and self._summary[:10] != "never sent"

    def _ledger_name_col(self) -> int:
        """The shared column's current width, or the fixed floor off-ledger."""
        parent = self.parent
        return parent.tool_name_col if isinstance(parent, TranscriptView) else NAME_COL

    def _label_min_width(self) -> int:
        """Card width at or above which the summary keeps its whole label.

        Counted, not discovered by comparing against a live budget: the budget
        moves non-monotonically as the frame narrows, because the status segment's
        cap sheds its reason on the way down and returns those cells to the
        summary. A width the full row needs is a constant, so the label can only
        flip once — shed as the terminal narrows, never restored.

        Everything the row spends before the summary at its WIDEST: the inner
        padding, icon, name column and their separators, the widest status the
        card can show, and the label itself.
        """
        widest_status = sum(cell_len(text) for text, _style in self._status_runs())
        return (
            2  # inner padding, one cell each side (kit rule)
            + 2  # icon and its separator
            # The WIDEST column the ledger can hand this row: the threshold has
            # to be a constant, so it cannot depend on the frame that is being
            # tested against it.
            + max(NAME_COL, self._ledger_name_col())
            + 1  # separator after the name column
            + cell_len(self._summary)
            + (widest_status + 2 if widest_status else 0)
        )

    def _status_runs(self, cap: int = 0, *, terse: bool = False) -> list[tuple[str, Style]]:
        """The right-aligned status as (text, style) runs (D12/D13/D28).

        Diff counters ride in FRONT of the outcome glyph and are the first
        thing dropped when the cap bites: how a write went is core, how much
        it wrote is meta. ``terse`` goes one step further and drops the
        failure reason too; see :meth:`_outcome_runs`.
        """
        if self._state in ("running", "composing"):
            return []  # D28: no trailing glyph until the duration lands
        core = self._outcome_runs(cap, terse=terse)
        diff = self._diff_runs()
        if not diff:
            return core
        core_cells = sum(cell_len(text) for text, _style in core)
        diff_cells = sum(cell_len(text) for text, _style in diff)
        if cap and core_cells + diff_cells > cap:
            return core
        return diff + core

    def _diff_runs(self) -> list[tuple[str, Style]]:
        """``+N`` / ``-N`` counters, tinted success/danger. Empty when unknown."""
        runs: list[tuple[str, Style]] = []
        if self._added > 0:
            runs.append((f"+{self._added} ", Style(color=theme_mod.semantic_color("success"))))
        if self._removed > 0:
            runs.append((f"-{self._removed} ", Style(color=theme_mod.semantic_color("danger"))))
        return runs

    def _outcome_runs(self, cap: int = 0, *, terse: bool = False) -> list[tuple[str, Style]]:
        """The settled outcome as runs: ``[reason] <glyph> <duration>``.

        ONE shape for all three settled states, which is what makes the
        column a column. The duration is right-justified into
        :data:`DURATION_COL` and the glyph sits immediately in front of it,
        so ``✓``, ``✗`` and ``⊘`` land on the same cell whatever preceded
        them and whether the tool took 0.4s or 12.3s. A column of pass/fail
        marks that wobbles by a cell per row is a column the eye has to read
        instead of scan — which defeats the reason the status segment is
        right-aligned in the first place.

        Interrupted used to be the exception: it returned ``⊘ interrupted``
        with no duration at all and sat six cells left of its neighbours.
        That is the worst row to leave out of the column, because one Esc
        marks EVERY tool still in flight, so the hole opened across a whole
        run of rows exactly where an operator scans to find where work
        stopped.

        ``terse`` drops the reason text. The caller reaches for it when the
        row is too narrow to keep both a tool NAME and a message: identity
        outranks explanation, because a row that cannot say which tool failed
        has stopped being a ledger entry.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        elapsed = self._duration or 0.0
        # Sub-second precision where it distinguishes tools, and the SAME
        # grammar as everything else past a minute: the composing row above this
        # one says `1m57s`, and `117s` two seconds later on the same row is the
        # app disagreeing with itself about how it writes a duration.
        if elapsed < 10:
            duration = f"{elapsed:.1f}s"
        elif elapsed < 60:
            duration = f"{elapsed:.0f}s"
        else:
            duration = format_duration(elapsed)
        duration = duration.rjust(DURATION_COL)
        if self._state == "success":
            # D12: success is quiet — check + duration both dim, no reason.
            return [(f"{ICON_SUCCESS} ", dim), (duration, dim)]

        # The GLYPH sits in the status column with the duration, not at the head
        # of the reason (D20). The right edge is where an operator scans a run
        # of tool rows for pass/fail, and putting `✗` before a right-aligned
        # message moved the failed row's glyph ~25 cells left of the `✓` on every
        # neighbouring row — so the scan found a hole exactly where the answer
        # should be. The reason keeps the space to the glyph's left.
        #
        # `abbreviates` is what separates the two failing states. An error
        # message is CONTENT: `internal error: worker di…` still carries the
        # failure even cut short, so it truncates. "interrupted" is not — it
        # is a constant restating what ⊘ already means, so truncating it buys
        # nothing and costs discrimination: at 46 columns a real failure and
        # a user stop both painted `inte…`, identical in the leftmost and
        # longest column, which is the one the eye lands on first. Below the
        # width that holds the word whole it is dropped, and the glyph column
        # carries the state alone — which it can, being distinguishable from
        # ✓ and ✗ with no colour at all.
        if self._state == "interrupted":
            glyph, reason, tint = ICON_INTERRUPTED, "interrupted", dim
            abbreviates = False
        else:
            danger = Style(color=theme_mod.semantic_color("danger"))
            glyph, reason, tint = ICON_ERROR, self._error, danger
            abbreviates = True

        runs: list[tuple[str, Style]] = []
        if not terse and reason:
            if cap:
                # Uncapped, the caller clamps downstream and the reason rides
                # whole; capped, it has to fit in front of the glyph.
                room = max(0, cap - cell_len(f"{glyph}  ") - cell_len(duration))
                if abbreviates:
                    reason = truncate_cells(reason, room)
                elif cell_len(reason) > room:
                    reason = ""
            # A reason cut down to a bare ellipsis is a cell spent saying
            # "there were words here". Drop it and give the cell back to the
            # columns that still mean something.
            if reason and reason != "…":
                runs.append((f"{reason} ", tint))
        runs.append((f"{glyph} ", tint))
        runs.append((duration, dim))
        return runs

    # -- FINALIZED-BLOCK protocol -------------------------------------------
    def settled_rows(self) -> int:
        """Rows settled now: one collapsed, the whole card when expanded."""
        return self._row_count if self._finalized else 0

    def spans_multiple_rows(self) -> bool:
        """Exact: the card already tracks its own height, collapsed or not."""
        return self._row_count > 1
