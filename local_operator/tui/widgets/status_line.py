"""Status line — the full-width BAND on the kit's ``sunken`` ground (D3/D17).

The character refinement supersedes the thin border-row trick: the status
line is a full-width band sitting at the very bottom of the screen, painted
on the ``sunken`` ground with icon-led segments separated by ``·``:

    π model · cwd          12.4k tok · $0.0021

Left: brand glyph · model · cwd (+ the shimmering working indicator while a
turn streams — faithful: the working text rides the shimmer sweep; when
shimmer is off a static dim spinner keeps it legible, D26). Right: tokens ·
cost, right-aligned. The input panel sits ABOVE the band. One space of
breathing room around every segment (D18). All widths measured with
``rich.cells.cell_len`` (one width model).
"""

from __future__ import annotations

from pathlib import Path

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


def format_context_tokens(tokens: int) -> str:
    """Compact context estimate: ``12.4k`` / ``1.2m`` style, plain under 1k."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}m"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    return str(tokens)


def format_cost(cost: float) -> str:
    """Compact dollar cost: ``$0.0021`` under a cent, ``$0.12`` above."""
    if cost < 0.01:
        return f"${cost:.4f}"
    if cost < 1.0:
        return f"${cost:.3f}"
    return f"${cost:.2f}"


class StatusLine:
    """Draws status segments into the bottom status band widget.

    Owns no layout row: the band widget (a ``Static`` docked at the bottom)
    is repainted through :meth:`refresh`. Call :meth:`update` whenever a
    segment changes; call :meth:`refresh` after resizes so the truncation
    follows the new width (one width model: ``rich.cells.cell_len``).
    """

    def __init__(self, dock: Static) -> None:
        # A `Static`, not a bare `Widget`: the band is repainted by handing
        # it a rich renderable, which only content widgets accept.
        self._dock = dock
        self._model_label: str = ""
        self._cwd: str = ""
        self._context_tokens: int = 0
        self._streaming: bool = False
        self._cost: str = ""
        self._spinner_index: int = 0
        self._spinner_timer = None

    # -- segment setters ----------------------------------------------------
    def update(
        self,
        *,
        model_label: str | None = None,
        cwd: str | None = None,
        context_tokens: int | None = None,
        streaming: bool | None = None,
        cost: str | None = None,
    ) -> None:
        """Update any subset of segments and repaint the band."""
        if model_label is not None:
            self._model_label = model_label
        if cwd is not None:
            self._cwd = cwd
        if context_tokens is not None:
            self._context_tokens = context_tokens
        if cost is not None:
            self._cost = cost
        if streaming is not None and streaming != self._streaming:
            self._streaming = streaming
            self._sync_spinner_timer()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the band content, truncated to the dock's inner width."""
        width = max(self._dock.size.width, 10)
        self._dock.update(self._render(width))

    def dispose(self) -> None:
        """Stop the spinner timer (idempotent)."""
        self._stop_spinner()

    # -- rendering ----------------------------------------------------------
    def _render(self, width: int) -> Text:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        # Separators must sit BELOW the things they separate or they read as
        # content: `faint` is the ramp step under `dim` and exists for
        # exactly this — the dots recede and the segments group themselves.
        seam = Style(color=theme_mod.semantic_color("faint"))

        # Left: brand glyph · model · cwd
        left = Text()
        left.append(BRAND_GLYPH + " ", style=Style(color=theme_mod.semantic_color("accent")))
        if self._model_label:
            left.append(self._model_label, style=muted)
        if self._cwd:
            if self._model_label:
                left.append(_SEPARATOR, style=seam)
            left.append(Path(self._cwd).name or self._cwd, style=dim)
        if self._streaming:
            # The aggregate working LINE (WorkingBlock) carries the shimmer;
            # the band keeps a quiet activity glyph so a still frame still
            # reads "live" (D26). With shimmer off, that line is static too,
            # so the band spells the state out rather than relying on a
            # glyph the eye may read as decoration.
            from local_operator.tui.shimmer import shimmer_enabled

            left.append(_SEPARATOR, style=seam)
            left.append(_SPINNER_FRAMES[self._spinner_index], style=dim)
            if not shimmer_enabled():
                left.append(" working", style=dim)

        # Right: tokens · cost
        right = Text()
        if self._context_tokens > 0:
            right.append(f"{format_context_tokens(self._context_tokens)} tok", style=dim)
        if self._cost:
            if self._context_tokens > 0:
                right.append(_SEPARATOR, style=seam)
            right.append(self._cost, style=dim)

        left_cells = cell_len(left.plain)
        right_cells = cell_len(right.plain)
        gap = max(1, width - left_cells - right_cells)
        row = Text()
        row.append_text(left)
        row.append(" " * gap, style=dim)
        row.append_text(right)
        row.truncate(width, overflow="ellipsis")
        return row

    def render_text(self, width: int) -> Text:
        """Public render entry (tests): segments joined, truncated to width."""
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
