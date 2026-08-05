"""Tool execution card — ONE LINE PER ACTION, rendered as a FILLED card.

The character refinement (docs/REWRITE.md §D): tool calls are subtle
background-filled cards — one elevation step brighter than the ground
(kit ``surface`` on ``bg``), full-width single rows with 1-cell inner
padding, NO border. Elevation is a background step, never a shadow or line.

Row anatomy::

    ▸ bash       pytest -q ⟨expand⟩                       ✓ 0.4s
    ▸ edit       tui/theme.py ⟨expand⟩             +12 -3 ✓ 0.1s

- per-tool glyph ``▸``: the QUIET STATIC running marker (D25 — the shimmer
  rides the aggregate working line, not individual rows), painted in the
  accent while running so a still frame reads "live" (D26 fallback)
- tool NAME ljust'd into a 10-cell column spine (D7). The name and the
  summary each carry a two-step tint ramp: while the tool is LIVE the name
  is the string green and the summary is ``muted``; once it settles both
  drop one step (``muted``/``dim``) so the running row is always the
  brightest thing in the transcript.
- command/summary ellipsized to the remaining budget, with absolute paths
  compacted against the cwd/home so a deep path does not eat the row
- diff counters for file-mutating tools: ``+N`` in the success tint and
  ``-N`` in the danger tint, rendered ONLY when the tool result actually
  reported them (an unknown count renders nothing — never ``+0 -0``)
- status right-aligned (D6): EMPTY while running — no trailing glyph, the
  column stays clear until the duration lands (D28); ``✓ duration`` all
  dim on success (D12: only failure gets color); ``✗ error`` danger with
  the duration dim as a second run (D13); ``⊘ interrupted`` dim when the
  turn ended before completion (TUI-019)
- an ``⟨expand⟩`` hint trailing the summary — the whole ROW is the click
  target, and clicking it reveals the tool's full output indented beneath
  the summary; clicking again collapses back to exactly one row. The hint
  appears only when there is output worth revealing and flips to
  ``⟨collapse⟩`` when open. It rests at ``faint`` (the ramp's inert-hint
  step, all but invisible) and steps up to ``dim`` while the pointer is on
  the row, so a settled transcript reads as content, not as controls.

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

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import TranscriptBlock, TranscriptView

#: Status glyphs (no Nerd fonts; plain unicode).
ICON_GLYPH = "▸"  # row prefix: one tool-action marker
ICON_SUCCESS = "✓"
ICON_ERROR = "✗"
ICON_INTERRUPTED = "⊘"
#: Expansion affordance trailing the summary. Both spellings are the same
#: click target; only the label flips so the row always says what a click does.
EXPAND_HINT = "⟨expand⟩"
COLLAPSE_HINT = "⟨collapse⟩"

#: Spinner frames — kept for the aggregate working line; running cards use
#: the quiet static marker (D25), but legacy callers/tests may still read.
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

#: Tool-name column width (D7: a spine for the eye to scan names).
NAME_COL = 10
#: Minimum summary budget before we drop the expand hint (D8 floor).
_SUMMARY_FLOOR = 16
#: Indent of the expanded output block, aligned under the tool name column.
OUTPUT_INDENT = 2
#: Expanded output is capped: a 20k-line bash dump would otherwise turn the
#: transcript into a scroll trap. The head is kept (it carries the command's
#: framing) and the remainder is announced on a dim marker row.
EXPAND_MAX_LINES = 40
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
    return "".join(out) + ellipsis


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
    """A ``Text`` that can never grow a card by wrapping.

    Every segment this module emits is already truncated by cell width, so
    wrapping should be unreachable — but "should be" is not a guarantee. A
    card whose row is built one cell wider than the widget it lands in (the
    unavoidable gap between guessing a width before layout and being told
    the real one afterwards) would silently become TWO rows, and the
    one-line rule would be an arithmetic accident rather than a property.
    ``no_wrap`` makes the worst case a clipped cell instead of a lost line.
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
    """

    #: Adaptive spacing: consecutive tool rows stack tight (see transcript).
    SPACING_KIND = "tool"

    def __init__(
        self,
        tool_call_id: str,
        tool_name: str,
        args: dict[str, object] | None = None,
        intent: str | None = None,
    ) -> None:
        super().__init__()
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name
        self.add_class("tool-card", "tool-running")
        self._summary = intent or _summary_from_args(tool_name, args or {})
        self._state: str = "running"
        self._duration: float | None = None
        self._error: str = ""
        self._started = time.monotonic()
        self._expanded = False
        self._hovered = False
        self._added = 0
        self._removed = 0
        #: Cleaned result lines, populated once the tool finishes.
        self._output: list[str] = []
        #: Rows the card currently occupies (1 collapsed, N expanded).
        self._row_count = 1
        self._refresh_row()

    # -- lifecycle ----------------------------------------------------------
    def mark_done(self, result_text: str = "", details: dict[str, Any] | None = None) -> None:
        """Record success with elapsed duration; the row goes quiet."""
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
        self._error = " ".join(error.split()) or "error"
        self._absorb_result(result_text or error, details)
        self.remove_class("tool-running")
        self.add_class("tool-error")
        self._refresh_row()
        self.finalize()

    def mark_interrupted(self) -> None:
        """Turn ended before this tool completed: dim 'interrupted' state."""
        self._duration = time.monotonic() - self._started
        self._state = "interrupted"
        self.remove_class("tool-running")
        self.add_class("tool-interrupted")
        self._refresh_row()
        self.finalize()

    def set_partial_detail(self, detail: str) -> None:
        """Replace the running summary with a streaming partial result line."""
        if self._state != "running":
            return
        cleaned = " ".join(detail.split())
        if cleaned:
            self._summary = cleaned
            self._refresh_row()

    def _absorb_result(self, result_text: str, details: dict[str, Any] | None) -> None:
        """Capture the payload the expansion and the diff counters read."""
        self._added, self._removed = _diff_counts(details)
        self._output = self._clean_output(result_text)

    def _clean_output(self, result_text: str) -> list[str]:
        """Normalise the result into displayable rows (empty = nothing to show).

        A single-line result that merely repeats the summary carries no new
        information, so the card stays inert rather than advertising an
        expansion that reveals what is already on screen.
        """
        if not result_text:
            return []
        lines = [line.rstrip() for line in result_text.expandtabs(4).splitlines()]
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
        """True when the card holds output the one-line summary cannot show."""
        return bool(self._output)

    @property
    def expanded(self) -> bool:
        """True while the full output is revealed beneath the summary row."""
        return self._expanded

    def toggle_expanded(self) -> bool:
        """Flip the expansion (no-op with nothing to show); returns the state.

        Also nudges the transcript to re-decide the gap below this card: a
        card that just grew from one row to twenty has earned the breathing
        room its collapsed self did not need.
        """
        if not self.can_expand():
            return self._expanded
        self._expanded = not self._expanded
        self.set_class(self._expanded, "tool-expanded")
        self._refresh_row()
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_after(self)
        return self._expanded

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Mouse affordance: the whole row toggles the output view."""
        if not self.can_expand():
            return
        self.toggle_expanded()
        event.stop()

    def on_enter(self, event) -> None:  # type: ignore[no-untyped-def]
        """Pointer over an expandable row: light the hint up to `dim`."""
        self._set_hovered(True)

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        """Pointer gone: the hint recedes to `faint` again."""
        self._set_hovered(False)

    def _set_hovered(self, hovered: bool) -> None:
        """Repaint only when the hover state actually changes something.

        A row with nothing to expand shows no hint, so hovering it costs
        nothing — the transcript is a long list and the pointer crosses a
        lot of rows on the way anywhere.
        """
        if hovered == self._hovered or not self.can_expand():
            self._hovered = hovered
            return
        self._hovered = hovered
        self._refresh_row()

    # -- resize (TUI-017: rebuild the row when the width changes) -----------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._refresh_row()

    # -- rendering ----------------------------------------------------------
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
        if not self._expanded or not self._output:
            return row
        dim = Style(color=theme_mod.semantic_color("dim"))
        body = Style(color=theme_mod.semantic_color("danger")) if self._state == "error" else dim
        # The output block reuses the card's own inner padding budget and
        # truncates per line: one output line is one row, so the expanded
        # height is exactly what the marker promises and never reflows.
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
        return row

    def _build_row(self, width: int) -> Text:
        """The single summary row — the ONE-LINE guarantee lives here."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Two-step fade on settle: the live row keeps the string green on the
        # name and readable `muted` body text; a settled row drops both one
        # step so the running row is the brightest thing on screen.
        running = self._state == "running"
        name_style = Style(color=theme_mod.semantic_color("string")) if running else muted
        summary_style = muted if running else dim
        width = max(width - 2, 10)  # 1-cell inner padding each side (kit rule)

        # Status segment (right-aligned), capped at width // 3 (D8) and then
        # hard-clamped so no state can ever push the row past its card.
        status_cap = max(8, width // 3)
        status_runs = _clamp_runs(self._status_runs(status_cap), max(0, width - 3))
        status_cells = sum(cell_len(text) for text, _style in status_runs)

        # Prefix: glyph + space + name column + space. The status segment and
        # both separator cells are part of the summary budget (TUI-018). The
        # name column is cell-width bound and ADAPTIVE: below the full column
        # width the name shrinks (truncated by CELL width — len() on a wide
        # CJK/emoji name would break the spine) before the row overflows its
        # card and clips the status off-screen.
        name_budget = width - (2 + status_cells + 2)
        if name_budget < 2:
            # Too narrow for even a shrunken name: degrade to glyph + status
            # so the status column survives.
            row = _row_text()
            row.append(ICON_GLYPH + " ", style=dim)
            if status_runs:
                used = cell_len(row.plain)
                pad = max(1, width - used - status_cells)
                row.append(" " * pad, style=dim)
                for text, style in status_runs:
                    row.append(text, style=style)
            return row

        name_col = min(NAME_COL, name_budget)
        name = truncate_cells(self.tool_name, name_col, ellipsis="")
        name = name + " " * max(0, name_col - cell_len(name))
        prefix_cells = 2 + name_col + 1

        # Expand hint: only when there IS something to expand and the summary
        # floor survives it (D8). The label states the action a click performs.
        hint = ""
        if self.can_expand():
            hint = COLLAPSE_HINT if self._expanded else EXPAND_HINT
        hint_cells = cell_len(hint) + 1 if hint else 0
        budget = max(0, width - prefix_cells - status_cells - hint_cells - 2)
        if hint and budget < _SUMMARY_FLOOR:
            hint = ""
            budget = max(0, width - prefix_cells - status_cells - 2)
        summary = truncate_cells(self._summary, budget)

        row = _row_text()
        glyph_style = Style(color=theme_mod.semantic_color("accent")) if running else dim
        row.append(ICON_GLYPH + " ", style=glyph_style)
        row.append(name, style=name_style)
        row.append(" ", style=dim)
        row.append(summary, style=summary_style)
        # The hint appears ONLY under the pointer. A settled transcript is
        # content, not a wall of controls: at rest the ▸ chevron is the whole
        # affordance (and the card's ground lifts on :hover), so printing
        # ⟨expand⟩ on every row costs ~9 cells of permanent chrome on the
        # common 80-column terminal to say what the chevron already says.
        # Keeping it hover-only is what holds the borderless/minimal contract
        # while still giving the mouse a visible target.
        if hint and self._hovered:
            row.append(" ", style=dim)
            row.append(hint, style=Style(color=theme_mod.semantic_color("dim")))

        # Right-align the status column against the content width (D27).
        if status_runs:
            used = cell_len(row.plain)
            pad = max(1, width - used - status_cells)
            row.append(" " * pad, style=dim)
            for text, style in status_runs:
                row.append(text, style=style)
        return row

    def _status_runs(self, cap: int = 0) -> list[tuple[str, Style]]:
        """The right-aligned status as (text, style) runs (D12/D13/D28).

        Diff counters ride in FRONT of the outcome glyph and are the first
        thing dropped when the cap bites: how a write went is core, how much
        it wrote is meta.
        """
        if self._state == "running":
            return []  # D28: no trailing glyph until the duration lands
        core = self._outcome_runs(cap)
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

    def _outcome_runs(self, cap: int = 0) -> list[tuple[str, Style]]:
        """Glyph + duration (or error text) for the settled states."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        elapsed = self._duration or 0.0
        duration = f"{elapsed:.1f}s" if elapsed < 10 else f"{elapsed:.0f}s"
        if self._state == "success":
            # D12: success is quiet — check + duration both dim.
            return [(f"{ICON_SUCCESS} ", dim), (duration, dim)]
        if self._state == "interrupted":
            label = f"{ICON_INTERRUPTED} interrupted"
            if cap:
                label = truncate_cells(label, cap, ellipsis="")
            return [(label, dim)]
        # D13: error message danger, duration dim — two runs.
        error = self._error
        if cap:
            error = truncate_cells(error, max(1, cap - len(f"{ICON_ERROR}  ") - len(duration)))
        return [
            (f"{ICON_ERROR} {error} ", Style(color=theme_mod.semantic_color("danger"))),
            (duration, dim),
        ]

    # -- FINALIZED-BLOCK protocol -------------------------------------------
    def settled_rows(self) -> int:
        """Rows settled now: one collapsed, the whole card when expanded."""
        return self._row_count if self._finalized else 0

    def spans_multiple_rows(self) -> bool:
        """Exact: the card already tracks its own height, collapsed or not."""
        return self._row_count > 1
