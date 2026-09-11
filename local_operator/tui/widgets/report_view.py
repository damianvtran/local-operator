"""The shared report body: a line-API widget that strips ONE line at a time.

Extracted from ``/analytics`` (``analytics_panel``) because the shape it
replaces — a ``Static`` whose height is the whole report, inside a
``VerticalScroll`` — has a cost that is invisible until the report is long and
the widget is repainted often.

Why the ``Static`` shape cannot be afforded on a screen that repaints per
POINTER event: ``Widget._render_content`` converts the widget's OWN height to
strips on every dirty repaint (``textual/widget.py``: ``width, height =
self.size`` then ``Visual.to_strips(..., width, height, ...)``). The height of
the body is the height of the report, not of the viewport, so one hover change
— a pointer crossing a row — re-stripped all 846 body lines / 85,446 cells to
change 2 of them. Measured on a copy of the operator's 156 MB ledger at 120x45
through the real app: **243 ms mean / 320 ms max of loop-thread CPU per row
crossed**, against a 6.6 ms mean / 13.6 ms max floor for the same pointer moves
with no analytics screen open. Scrolling with the pointer resting on the table
is the same defect from the other side — the rows slide, the hovered id
changes, the report is re-stripped — at 174 ms per wheel line.
``Visual.to_strips`` is the entire cost, and the viewport it is converted for is
3.4% of it.

So this widget throws away the "one big renderable" model and renders the way
``RichLog`` and ``OptionList`` do: it owns its ``virtual_size`` and answers
``render_line(y)`` for the lines the compositor actually asks for, caching one
``Strip`` per line. Render becomes O(viewport) instead of O(report), and a
caller that knows which lines changed can invalidate exactly those with
``set_line`` (``refresh_line``) instead of rebuilding the body. Measured on the
same ledger, that takes a row-to-row pointer crossing from 243 ms to 8 ms, of
which the body's own part is the two strip conversions for the two rows whose
tint changed.

The strip is built through ``Visual.to_strips`` and the widget's own
``visual_style``, exactly as ``Widget._render_content`` would: styles,
selection and the CSS-derived foreground stay Textual's, not this widget's, so
the painted frame is byte-identical to the one the ``Static`` body produced
(asserted by the frame comparison the PR carries).

Two rules matter for a caller:

* **``set_lines`` is a full repaint, ``set_line`` is a single row.** Lines are
  addressed by BODY LINE index — the same index the report's own layout uses.
* **A line index is only meaningful at the width it was composed for.** The
  strip cache is dropped when the widget's own width changes, because a
  ``Strip`` is cut to the width it was built at.

Selection is NOT free here, and that is easy to miss when swapping a ``Static``
for a line-API widget: ``Widget.get_selection`` extracts text only from a
``Text``/``Content`` returned by ``_render()``, and this widget returns neither,
so without the overrides below a drag over the report selected nothing, painted
nothing and handed ``ctrl+c`` an empty string — a silent loss of an affordance
that exists on the ``Static`` body it replaces. ``Log`` carries the same two
overrides for the same reason (``textual/widgets/_log.py``: ``get_selection`` and
``selection_updated``): the text, and the per-line highlight. It is ``Log``, not
``RichLog``, that is the model for all three of the selection hooks here —
``Log`` is also the only widget in the library that stamps its strips with
``apply_offsets``, which is what makes the drag resolvable at all.
"""

from __future__ import annotations

from typing import Any

from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.selection import Selection
from textual.strip import Strip
from textual.visual import Visual, visualize


class ReportView(ScrollView):
    """A report body rendered line by line, with per-line strip caching."""

    #: ``overflow-x`` is pinned to ``hidden`` deliberately. ``ScrollView``'s own
    #: default is ``auto``, and this body takes the place of a ``Static`` inside
    #: a ``VerticalScroll`` — which is ``overflow-y: auto; overflow-x: hidden``.
    #: A report can be a few cells wider than its box on a narrow frame (the
    #: tables' numeric columns are sized to the data), so the inherited default
    #: put a horizontal scrollbar on the screen and cost the body a row of
    #: height that the replaced widget never cost it (measured at 90x40: content
    #: height 25 -> 24). The bar is not an affordance anyone asked for — the
    #: report reads down, and a line that overruns is cropped exactly as it was
    #: before.
    DEFAULT_CSS = """
    ReportView {
        overflow-x: hidden;
        overflow-y: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lines: list[Text] = []
        self._strips: dict[int, Strip] = {}
        #: Width the caller COMPOSED the lines for — the one input the resize
        #: handler cannot recover from the widget, since it survives a resize.
        self._composed_width: int = 0
        #: Width the cached strips were built at. A ``Strip`` is cut to the
        #: width it was built at, so a width change invalidates the whole cache
        #: — see ``render_line``. Zero until the first render, which is why the
        #: first comparison is against the live ``self.size.width``.
        self._strip_width: int = 0

    # -- selection -----------------------------------------------------------

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """The text under a drag selection — the affordance a line API loses.

        This is `Log.get_selection`'s shape (`textual/widgets/_log.py`, which is
        also where the offset stamp below comes from), joined over the same model
        the strips are built from, so what is copied is exactly what is painted.
        """
        return selection.extract("\n".join(line.plain for line in self._lines)), "\n"

    def selection_updated(self, selection: Selection | None) -> None:
        """Repaint: the cached strips were built without the new selection."""
        self._strips.clear()
        self.refresh()

    # -- lifecycle -----------------------------------------------------------

    def notify_style_update(self) -> None:
        """Drop cached strips, which hold the OLD resolved styles.

        A strip bakes in the component style it was rendered with, so a theme or
        stylesheet change leaves every cached line pinning the previous ramp
        (`RichLog.notify_style_update` clears its line cache on this hook for the
        same reason; `OptionList` refreshes here and clears its caches on
        `_on_resize` instead, which is the other half of the same idea).
        """
        super().notify_style_update()
        self._strips.clear()
        self.refresh()

    # -- model ---------------------------------------------------------------

    def set_lines(self, lines: list[Text], width: int) -> None:
        """Replace the whole body — one ``Text`` per line — and repaint.

        ``lines`` are already split — the screen flattens ``build_report``'s
        multi-line blocks into one ``Text`` per line before handing them over
        (see ``analytics_panel._flatten_blocks``) — so this is a straight
        hand-off, not a re-render. The body's line numbering and the caller's
        must be the same numbering, or the caller's "patch line N" would
        address a different row than the one it composed.

        ``width`` is the width the lines were COMPOSED for. The content size is
        the wider of that and the longest line, which reproduces what the
        widget this replaces did as ``width: auto``: a report that overruns the
        box it was composed into (the tables' numeric columns are sized to the
        data, so a narrow frame can overrun by a few cells) kept its own wider
        content size, so it was croppable rather than wrapped. Sizing the
        content to the box instead would silently drop the tail of any such
        line — see ``_make_strip``.
        """
        # Keep the strips of lines that came back byte-identical (a metric
        # toggle or an expand rewrites one section and leaves the rest alone),
        # but only at an unchanged width: the lines of a report recomposed at a
        # new width are cut differently even when their PLAIN text happens to
        # match, and a stale strip would paint the old width's row.
        if self.size.width != self._strip_width:
            self._strips.clear()
        else:
            self._strips = {
                index: strip
                for index, strip in self._strips.items()
                if index < len(lines)
                and index < len(self._lines)
                and lines[index] == self._lines[index]
            }
        self._lines = lines
        self._composed_width = width
        self._publish_virtual_size()
        self.refresh()

    def _publish_virtual_size(self) -> None:
        """Publish the content size from the lines, the caller's width and the box.

        At least the box, at least the width the lines were composed for, at
        least the widest line: the ``width: auto`` body this replaces sized
        itself to the content it was given and stretched to its container, and
        the geometry a reader can see (the scroll extent) should not move
        because the widget changed shape underneath it.

        Called again from ``on_resize`` because the box term is only correct
        once layout has settled: a resize-triggered ``set_lines`` reads the
        region BEFORE layout, so it republishes the pre-resize width and the
        content stays that wide until something else repaints it (measured: 101
        kept after a live 120x45 -> 90x40, where a fresh open at 90x40 reports
        86). No visible effect was found for it — the box clips and horizontal
        scrolling is off — but a stale published width is a wrong claim about
        the widget, and it is fixed at the one place that knows better.
        """
        widest = max((cell_len(line.plain) for line in self._lines), default=0)
        self._resize_virtual(
            max(self._composed_width, widest, self.scrollable_content_region.width),
            len(self._lines),
        )

    def on_resize(self, event: events.Resize) -> None:
        self._publish_virtual_size()

    def set_line(self, index: int, line: Text) -> None:
        """Replace ONE line and repaint only it.

        The index is a body line, not a viewport row: ``ScrollView.refresh_line``
        subtracts the scroll offset itself, so a caller never has to know where
        the viewport happens to be sitting.
        """
        if not 0 <= index < len(self._lines):
            return
        self._lines[index] = line
        self._strips.pop(index, None)
        self.refresh_line(index)

    def lines_for_test(self) -> list[Text]:
        """The composed lines, in body order — what ``render_line`` will paint.

        The same idea as ``AnalyticsScreen.render_lines_for_test``: a test can
        assert on the body without going through the compositor, and it reads
        the model the strips are built from rather than a second copy of it.
        """
        return self._lines

    def _resize_virtual(self, width: int, height: int) -> None:
        """Publish the content size, so the scrollbar maths follows the report.

        ``height`` is the line count, and it is what makes the view scroll at
        all: without it ``max_scroll_y`` stays 0 and the body would be cropped
        rather than scrolled. The ``OptionList`` idiom (``virtual_size`` then
        ``_scroll_update``), because setting ``virtual_size`` alone leaves the
        scrollbars describing the old content.
        """
        size = Size(max(1, width), height)
        if size != self.virtual_size:
            self.virtual_size = size
            self._scroll_update(size)

    # -- render --------------------------------------------------------------

    def render_line(self, y: int) -> Strip:
        """The strip for viewport row ``y``, built once and cached per line.

        ``y`` is viewport-relative — Textual asks for the rows it is about to
        paint — so the body line is the scroll offset plus ``y``. This is the
        whole point of the widget: the report's other 800-odd lines are never
        converted at all.

        ``apply_offsets`` is not decoration, and it is the one thing a line-API
        widget must remember for a drag selection to work: the compositor
        recovers the CONTENT offset under the pointer from a ``"offset"`` style
        meta on the segments it renders (``_compositor.get_widget_and_offset_at``),
        so a strip returned without it resolves the pointer's line to nothing —
        the drag then has a start and no end, which textually means "select to
        the end of everything". ``Log`` stamps its strips for exactly this reason
        (it is the library's only ``apply_offsets`` caller); the base ``Static``
        body got the same meta from Textual's own render path.

        The stamp's x is ``scroll_offset.x``, and that is only honest because the
        strip is served UNCROPPED and the axis is pinned: ``Log`` crops by
        ``scroll_x`` before stamping, while this widget strips at the line's own
        width and lets the compositor crop, so enabling horizontal scrolling
        here would offset every meta by ``scroll_x`` without any test noticing.
        If that axis is ever turned on, crop-then-stamp first.
        """
        width = self.size.width
        if width != self._strip_width:
            self._strips.clear()
            self._strip_width = width
        index = int(self.scroll_offset.y) + y
        strip = self._strips.get(index)
        if strip is None:
            strip = self._make_strip(index, width)
            self._strips[index] = strip
        return strip.apply_offsets(int(self.scroll_offset.x), index)

    def _make_strip(self, index: int, width: int) -> Strip:
        if not 0 <= index < len(self._lines):
            # Past the end of the report: a blank of the same width and style,
            # not a short strip, or the region below the last line would be
            # painted with segments missing their background.
            return Strip.blank(width, self.visual_style.rich_style)
        line = self._lines[index]
        span = self._selection_span(index)
        if span is not None:
            # COPY before styling: ``self._lines`` are the screen's own Text
            # objects, reused by the next ``set_lines`` equality check and by
            # ``build_report`` recomposes, so a paint concern must not mutate
            # them.
            line = line.copy()
            line.stylize(self.screen.get_component_rich_style("screen--selection"), *span)
        # A line wider than the widget is stripped at its OWN width and left for
        # the compositor to crop. Stripping it at the widget width instead would
        # wrap it and return line 1 — the tail of the row silently missing from
        # the frame (measured on the operator's ledger at 90 columns: the
        # ``Context read`` row is 77 cells inside a 74-cell box and lost its last
        # five, which the ``width: auto`` body it replaces cropped rather than
        # wrapped). Cropping loses the cells the viewport hides; wrapping loses
        # cells that would have been painted there, which is a different and
        # visible answer.
        return Visual.to_strips(
            self,
            visualize(self, line),
            max(width, cell_len(line.plain)),
            1,
            self.visual_style,
            # The generic path resolves a selection against the visual it is
            # handed, and this one is a SINGLE LINE, so every strip would read
            # the span of body line 0: a selection starting on line 0 (which is
            # where `/analytics` opens — the `Totals` line) then banded EVERY
            # line of the report, 1394 cells over 28 rows where 4 rows are
            # selected (measured, 120x45). The span above is applied by the only
            # code that knows this line's index, so the visual must not apply one.
            apply_selection=False,
        )[0]

    def _selection_span(self, index: int) -> tuple[int, int] | None:
        """Character span of the live selection on body line ``index``.

        ``Selection`` offsets are CHARACTER offsets, not cell columns, and they
        are read raw here on purpose: ``Strip.apply_offsets`` advances ``x`` by
        ``len(segment.text)`` per segment, ``Compositor.get_widget_and_offset_at``
        counts characters into the segment the pointer landed on, and
        ``Selection.extract`` (``get_selection`` above) slices the plain string
        with the same numbers. Converting them a second time from cells — which
        an earlier revision did — is the identity only while every glyph is one
        cell wide, and drifts the highlight by a character per wide glyph while
        the copy stays right, i.e. the frame ends up describing different
        characters than ctrl+c hands over. This is `Log._render_line_strip`'s
        arithmetic.

        ``-1`` means "to the end of the line", which is resolved here because
        only the line knows its length.
        """
        selection = self.text_selection
        if selection is None:
            return None
        span = selection.get_span(index)
        if span is None:
            return None
        start, end = span
        return start, len(self._lines[index].plain) if end == -1 else end
