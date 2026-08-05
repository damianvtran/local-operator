"""Transcript container and the FINALIZED-BLOCK protocol.

Ported from omp's ``TranscriptContainer``: blocks appended to the transcript
declare when they are done mutating. A block exposing ``is_finalized()`` is
treated as immutable by the container — its content is never updated again —
and ``settled_rows()`` reports how many of its rows are provably stable now
(used later for scroll accounting).

Minimalism rule (the brand): blocks own NO outer margin and the container
never adds blank filler rows. Structure comes from symbols, tint, and
spacing — never rules or blank rows. The tcss pins zero margin/padding on
every block.

Layout rhythm (D20): user blocks sit at the gutter (``❯`` at column 0);
everything else indents two cells so the turn spine reads at a glance.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import Static

from local_operator.tui import theme as theme_mod

#: The turn spine (D20): user prompts sit at the gutter; everything else
#: indents two cells so the ``❯`` column reads at a glance.
SPINE_INDENT = 2


class TranscriptBlock(Static):
    """Base class for one transcript entry (assistant, tool, user, notice).

    Content is applied through :meth:`set_content`; once :meth:`finalize` is
    called the block is immutable — further :meth:`set_content` calls are
    ignored, which is the container's guarantee that committed rows never
    change under scroll.

    Row accounting (TUI-011): ``settled_rows`` is LAZY. ``set_content`` does
    not measure the renderable; the count is estimated from the renderable
    only when :meth:`settled_rows` is actually read (and memoized). The hot
    streaming path never pays for measurement.
    """

    DEFAULT_CSS = ""  # all styling lives in local_operator.tcss

    #: Set False once the block will never mutate again.
    _finalized: bool = False
    #: Last applied content, kept for lazy settled_rows measurement.
    _content: object = None
    #: Memoized settled row count (None = not measured yet).
    _settled_rows_cache: int | None = None

    def set_content(self, renderable: object) -> None:
        """Apply ``renderable`` as the block content (no-op once finalized)."""
        if self._finalized:
            return
        self._content = renderable
        self._settled_rows_cache = None  # invalidate the lazy count
        self.update(renderable)

    @property
    def renderable(self) -> object:
        """The current content renderable (rich) — inspection/test hook.

        Textual 8's ``Static`` no longer exposes a public ``renderable``;
        blocks keep their own reference so tests and exporters can read the
        exact rich object last applied via :meth:`set_content`.
        """
        return self._content

    def finalize(self) -> None:
        """Freeze the block; the container never re-renders it afterwards."""
        self._finalized = True

    def is_finalized(self) -> bool:
        """True when the block is immutable (FINALIZED-BLOCK protocol)."""
        return self._finalized

    def settled_rows(self) -> int:
        """Leading rows provably byte-stable now (all rows once finalized).

        Lazy: measured on first read after the last content change, at the
        block's OWN width when mounted (D3: no hardcoded reference width).
        """
        if not self._finalized:
            return 0
        if self._settled_rows_cache is None:
            self._settled_rows_cache = _count_rows(self._content, self.size.width or 80)
        return self._settled_rows_cache


class UserBlock(TranscriptBlock):
    """One user prompt at the gutter: a dim ``❯`` chevron at column 0."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.add_class("user-block")
        line = Text()
        line.append("❯ ", style=Style(color=theme_mod.semantic_color("dim")))
        line.append(text, style=Style(color=theme_mod.semantic_color("fg")))
        self.set_content(line)
        self.finalize()


#: Notice kind glyphs (D14): structure from symbols, not prefixes.
NOTICE_GLYPHS: dict[str, str] = {
    "info": "·",
    "warning": "!",
    "error": "✗",
}


class NoticeBlock(TranscriptBlock):
    """One notice line: glyph + text, tinted by kind (D14), on the spine."""

    _KIND_TOKENS: ClassVar[dict[str, str]] = {
        "info": "dim",
        "warning": "warning",
        "error": "danger",
    }

    def __init__(self, text: str, kind: str = "info") -> None:
        super().__init__()
        self.add_class("notice-block")
        token = self._KIND_TOKENS.get(kind, "dim")
        glyph = NOTICE_GLYPHS.get(kind, "·")
        style = Style(color=theme_mod.semantic_color(token))
        line = Text()
        line.append(" " * SPINE_INDENT, style=style)
        line.append(f"{glyph} ", style=style)
        line.append(text, style=style)
        self.set_content(line)
        self.finalize()


class RichBlock(TranscriptBlock):
    """A finalized block wrapping one pre-built rich renderable.

    Used where the app needs multi-style content (``/help`` columns,
    structured listings) that the single-tint NoticeBlock cannot express.
    Content rides the spine indent (D20).
    """

    def __init__(self, renderable: object) -> None:
        super().__init__()
        self.add_class("rich-block")
        from rich.padding import Padding

        self.set_content(Padding(renderable, (0, 0, 0, SPINE_INDENT)))
        self.finalize()


class WorkingBlock(TranscriptBlock):
    """The ONE aggregate working line (D25): shimmer sweeps it at 30 fps.

    omp rides a single working message, never per-row animation. When
    shimmer is disabled (settings/env), the line falls back to a static dim
    marker so the running state stays legible in a still frame (D26).
    """

    #: Repaint cadence — omp repaints animated loader text at 30 fps.
    _FRAME_MS = 33

    def __init__(self) -> None:
        super().__init__()
        self.add_class("working-block")
        self._frame_ms: float = 0.0
        self._timer = None
        self._paint()

    def on_mount(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled

        if shimmer_enabled():
            self._timer = self.set_interval(self._FRAME_MS / 1000, self._tick)

    def _tick(self) -> None:
        self._frame_ms += self._FRAME_MS
        self._paint()

    def _paint(self) -> None:
        from local_operator.tui.shimmer import shimmer_enabled, shimmer_text

        line = Text(" " * SPINE_INDENT)
        if shimmer_enabled():
            line.append_text(shimmer_text("working…", self._frame_ms))
        else:
            dim = Style(color=theme_mod.semantic_color("dim"))
            line.append("· ", style=dim)
            line.append("working…", style=dim)
        self.set_content(line)

    def stop(self) -> None:
        """Stop the repaint timer and settle on the static frame."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


class TranscriptView(ScrollableContainer):
    """The scrolling column every block appends into.

    Owns exactly one separator behavior: none. Blocks carry no outer margin
    and the container adds no blank filler rows (density is the brand); the
    tcss pins ``padding: 0; margin: 0`` on every block. Appends scroll to
    the bottom unless the user has scrolled up to read.

    ``clear_blocks`` notifies an optional ``on_clear`` hook (TUI-009) so the
    app can reset its streaming/tool-card bookkeeping.
    """

    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__()
        self._blocks: list[TranscriptBlock] = []
        self._on_clear: Callable[[], None] | None = None

    def set_on_clear(self, hook: Callable[[], None] | None) -> None:
        """Install the hook fired after every :meth:`clear_blocks`."""
        self._on_clear = hook

    def append_block(self, block: TranscriptBlock) -> None:
        """Mount ``block`` at the bottom and keep the tail in view.

        Scrolling is deferred through ``call_after_refresh`` so the freshly
        mounted block's layout settles BEFORE ``scroll_end`` measures the
        virtual size (TUI-022) — an immediate scroll would target the stale
        pre-mount extent and land short.
        """
        self._blocks.append(block)
        stick_to_bottom = self._is_near_bottom()
        self.mount(block)
        if stick_to_bottom:
            self.call_after_refresh(self.scroll_end, animate=False)

    def remove_block(self, block: TranscriptBlock) -> None:
        """Remove one block (used to lift the boot hint, D9)."""
        if block in self._blocks:
            self._blocks.remove(block)
            block.remove()

    def blocks(self) -> list[TranscriptBlock]:
        """Blocks in append order (live and finalized)."""
        return list(self._blocks)

    def clear_blocks(self) -> None:
        """Remove every block (the ``/clear`` command)."""
        for block in self._blocks:
            block.remove()
        self._blocks.clear()
        self.scroll_home(animate=False)
        if self._on_clear is not None:
            self._on_clear()

    def _is_near_bottom(self) -> bool:
        """True when the viewport sits at (or within 2 rows of) the bottom."""
        max_offset = self.virtual_size.height - self.size.height
        if max_offset <= 0:
            return True
        return self.scroll_offset.y >= max_offset - 2


def _count_rows(renderable: object, width: int = 80) -> int:
    """Row count a renderable occupies, measured through rich (one model).

    Only called lazily from ``settled_rows`` — never on the streaming path.
    """
    if isinstance(renderable, str):
        return max(1, len(renderable.splitlines()))
    if isinstance(renderable, Text):
        return max(1, len(renderable.plain.splitlines()))
    if renderable is None:
        return 0
    console = Console(width=max(width, 10))
    try:
        segments = console.render(renderable, console.options)  # type: ignore[arg-type]
    except Exception:
        return 1
    rows = 1
    for segment in segments:
        rows += segment.text.count("\n")
    return rows
