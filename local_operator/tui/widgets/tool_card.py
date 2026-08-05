"""Tool execution card — ONE LINE PER ACTION, rendered as a FILLED card.

The character refinement (docs/REWRITE.md §D): tool calls are subtle
background-filled cards — one elevation step brighter than the ground
(kit ``surface`` on ``bg``), full-width single rows with 1-cell inner
padding, NO border. Elevation is a background step, never a shadow or line.

Row anatomy::

    ▸ bash       ls -la …                    ✓ 0.4s

- per-tool glyph ``▸``: the QUIET STATIC running marker (D25 — the shimmer
  rides the aggregate working line, not individual rows), painted in the
  accent while running so a still frame reads "live" (D26 fallback)
- tool NAME in the string tint (violet labels become our green),
  ljust'd into a 10-cell column spine (D7)
- command/summary dim, ellipsized to the remaining budget
- status right-aligned (D6): EMPTY while running — no trailing glyph, the
  column stays clear until the duration lands (D28); ``✓ duration`` all
  dim on success (D12: only failure gets color); ``✗ error`` danger with
  the duration dim as a second run (D13); ``interrupted`` dim when the
  turn ended before completion (TUI-019)
- right-aligned dim ``⟨expand⟩`` hint reserved as the future expansion
  surface — shown only when the summary floor survives it (D8)

Widths measured through ``rich.cells.cell_len`` only (one width model).
"""

from __future__ import annotations

import time

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import TranscriptBlock

#: Status glyphs (no Nerd fonts; plain unicode).
ICON_GLYPH = "▸"  # row prefix: one tool-action marker
ICON_SUCCESS = "✓"
ICON_ERROR = "✗"
#: Future expansion surface hint (right-aligned, dim).
EXPAND_HINT = "⟨expand⟩"

#: Spinner frames — kept for the aggregate working line; running cards use
#: the quiet static marker (D25), but legacy callers/tests may still read.
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

#: Tool-name column width (D7: a spine for the eye to scan names).
NAME_COL = 10
#: Minimum summary budget before we drop the expand hint (D8 floor).
_SUMMARY_FLOOR = 16


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


def _summary_from_args(tool_name: str, args: dict[str, object]) -> str:
    """One-line summary: the intent-like first scalar arg, compactly joined."""
    parts: list[str] = []
    for key, value in args.items():
        if isinstance(value, str):
            text = value.replace("\n", " ").strip()
        elif isinstance(value, (int, float, bool)):
            text = str(value)
        else:
            continue
        if text:
            parts.append(text)
        if len(parts) >= 2:
            break
    summary = " ".join(parts)
    return summary or tool_name


class ToolCard(TranscriptBlock):
    """A single-row tool execution card on the ``surface`` elevation step.

    Lifecycle: construct with ``tool_call_id``/``tool_name`` (running),
    :meth:`mark_done` on success, :meth:`mark_failed` on error,
    :meth:`mark_interrupted` when the turn ends first. The row is the whole
    card; the ``⟨expand⟩`` hint is reserved for a future expansion surface
    and never grows the card.
    """

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
        self._refresh_row()

    # -- lifecycle ----------------------------------------------------------
    def mark_done(self) -> None:
        """Record success with elapsed duration; the row goes quiet."""
        self._duration = time.monotonic() - self._started
        self._state = "success"
        self.remove_class("tool-running")
        self.add_class("tool-success")
        self._refresh_row()
        self.finalize()

    def mark_failed(self, error: str) -> None:
        """Record failure with a ONE-line error message."""
        self._duration = time.monotonic() - self._started
        self._state = "error"
        self._error = " ".join(error.split()) or "error"
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

    # -- resize (TUI-017: rebuild the row when the width changes) -----------
    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._finalized:
            # Temporarily allow a re-render: a resize must re-fit the row.
            self._finalized = False
            self._refresh_row()
            self._finalized = True
        else:
            self._refresh_row()

    # -- rendering ----------------------------------------------------------
    def _refresh_row(self) -> None:
        """Rebuild the single row at the card's OWN width (D3).

        The width is never guessed from a hardcoded constant: before layout
        the row falls back to the app's console width, and ``on_resize``
        rebuilds it at the real width afterwards.
        """
        container = getattr(self, "container_size", None)
        width = self.size.width or (container.width if container else 0)
        if width <= 0:
            try:
                width = self.app.console.width
            except Exception:
                return  # unmounted: the first resize renders the row
        self.set_content(self._build_row(width))

    def _build_row(self, width: int) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        # The name stays the green only while the tool is live; a settled
        # row fades to dim so the running row is the brightest thing on
        # screen (the single-green discipline applies to the name too).
        name_style = (
            Style(color=theme_mod.semantic_color("string")) if self._state == "running" else dim
        )
        width = max(width - 2, 10)  # 1-cell inner padding each side (kit rule)

        # Status segment (right-aligned), capped at width // 3 (D8).
        status_cap = max(8, width // 3)
        status_runs = self._status_runs(status_cap)
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
            row = Text()
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

        # Expand hint: only when the summary floor survives it (D8).
        hint = EXPAND_HINT
        hint_cells = cell_len(hint) + 1  # + separator space
        budget = max(0, width - prefix_cells - status_cells - hint_cells - 2)
        if budget < _SUMMARY_FLOOR:
            hint = ""
            budget = max(0, width - prefix_cells - status_cells - 2)
        summary = truncate_cells(self._summary, budget)

        row = Text()
        glyph_style = (
            Style(color=theme_mod.semantic_color("accent")) if self._state == "running" else dim
        )
        row.append(ICON_GLYPH + " ", style=glyph_style)
        row.append(name, style=name_style)
        row.append(" ", style=dim)
        row.append(summary, style=dim)
        if hint:
            row.append(" ", style=dim)
            row.append(hint, style=dim)

        # Right-align the status column against the content width (D27).
        if status_runs:
            used = cell_len(row.plain)
            pad = max(1, width - used - status_cells)
            row.append(" " * pad, style=dim)
            for text, style in status_runs:
                row.append(text, style=style)
        return row

    def _status_runs(self, cap: int = 0) -> list[tuple[str, Style]]:
        """The right-aligned status as (text, style) runs (D12/D13/D28)."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        if self._state == "running":
            return []  # D28: no trailing glyph until the duration lands
        elapsed = self._duration or 0.0
        duration = f"{elapsed:.1f}s" if elapsed < 10 else f"{elapsed:.0f}s"
        if self._state == "success":
            # D12: success is quiet — check + duration both dim.
            return [(f"{ICON_SUCCESS} ", dim), (duration, dim)]
        if self._state == "interrupted":
            return [("interrupted", dim)]
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
        """One row, settled only once the tool finished."""
        return 1 if self._finalized else 0
