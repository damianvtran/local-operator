"""One inline image in the transcript, as a first-class block.

The transcript already carries the model's READING of an image — the tool
card's caption line, the prompt's ``[Image #N]`` marker — but not the image
itself, so a session that navigates by screenshots (UI work, chart reviews,
"look at this error") was a ledger of descriptions of pictures nobody could
see. This block puts the picture in the flow, sized consistently, in the
original aspect ratio, live for the session and equally on ``--resume``
(images ride the transcript as base64, so a resumed conversation replays
them from the same bytes the model saw).

Rendering strategy lives in :mod:`local_operator.tui.images` (protocol
detection, kitty escapes, fit arithmetic); this widget owns the lifecycle:

- decode once (Pillow), remember pixels, drop the base64;
- on mount, pick the best mode the terminal supports and paint;
- kitty mode transmits the pixels to the terminal ONCE, downscaled to the
  largest grid the caps allow, then paints plain placeholder text — resizes
  re-place (one short escape) and repaint text, never retransmit;
- a live-image budget caps terminal-side store use; evicted blocks demote
  to half-cells in place, so scrollback keeps the picture;
- unmount deletes the terminal-side image, leaving the terminal clean.

The UNAVAILABLE state is deliberate UX, not an error path: an image whose
bytes are gone (pruned from a resumed transcript, dropped after a provider
rejection) or undecodable (corrupt data, HEIC without the codec) renders a
one-row receipt naming what is missing and why, in the notice ink — the
reader learns an image WAS here, which is the fact the empty space would
have hidden.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, ClassVar, cast

from rich.style import Style
from rich.text import Text

from local_operator.tui import images as images_mod
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import SPINE_INDENT, TranscriptBlock

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PILImage

__all__ = ["ImageBlock"]

#: The half block char half-cell mode paints: fg = top pixel, bg = bottom.
_HALF = "▀"

#: Glyph leading the unavailable receipt. WGL4-safe (no Nerd Font needed).
_UNAVAILABLE_GLYPH = "▨"

#: Glyph leading the HEALTHY text-mode receipt — the prompt's own attachment
#: row already uses ``↑`` for "attached", so the receipt speaks the same
#: grammar and a glyph scan can tell fine-but-not-shown from broken (design
#: round 1, D3). ``▨`` stays exclusively the unavailable mark.
_ATTACHED_GLYPH = "↑"


def _decode(data_b64: str, mime_type: str) -> "PILImage | None":
    """Decode base64 image bytes to a PIL image, or ``None``.

    HEIC/HEIF goes through pillow-heif's opener when the ``images`` extra is
    installed; its absence is one of the honest "unavailable" reasons rather
    than a crash. Every failure maps to ``None`` — the caller renders the
    unavailable receipt, which is this feature's error UX.
    """
    try:
        import io

        from PIL import Image as PIL_Image

        if mime_type in ("image/heic", "image/heif"):
            try:
                from pillow_heif import register_heif_opener

                register_heif_opener()
            except Exception:
                return None
        raw = base64.b64decode(data_b64, validate=False)
        image = PIL_Image.open(io.BytesIO(raw))
        image.load()
        return image
    except Exception:
        return None


def _shrink(image: "PILImage") -> "PILImage":
    """The retained working copy: capped at the largest renderable grid.

    The cap grid (``MAX_COLS`` x ``MAX_ROWS`` cells at the real cell pixel
    size) is the most pixels ANY mode can ever paint from this block, so
    keeping more than that resident buys nothing. Downscaling once here also
    makes the kitty transmit frame free (it IS this copy) and bounds every
    half-cell resize to a small source. Converted to RGBA up front so the
    per-mode paths never re-convert.
    """
    cell = images_mod.cell_size()
    cols, rows = images_mod.fit_cells(
        image.width, image.height, images_mod.MAX_COLS, images_mod.MAX_ROWS
    )
    target = (max(1, cols * cell.width), max(1, rows * cell.height))
    frame = image if image.mode in ("RGB", "RGBA") else image.convert("RGBA")
    if frame.width > target[0] or frame.height > target[1]:
        frame = frame.copy() if frame is image else frame
        frame.thumbnail(target)
    return frame


def _rgb(pixel: tuple[int, ...]) -> str:
    """A rich ``rgb(...)`` color string for one decoded pixel."""
    return f"rgb({pixel[0]},{pixel[1]},{pixel[2]})"


class ImageBlock(TranscriptBlock):
    """One image, indented onto the transcript's text column.

    Sizing is the consistent-ledger rule from :mod:`~local_operator.tui.images`:
    every image fits a shared height ceiling (:data:`~local_operator.tui.images.MAX_ROWS`)
    with width following its own aspect ratio, so a column of mixed
    screenshots scans like rows of a ledger. Small images stay small.

    The block is finalized after every paint (the FINALIZED-BLOCK protocol:
    committed rows never change under scroll) and does the same temporary
    unfreeze dance ``NoticeBlock.restate`` documents for the three legitimate
    repaints: a width change, a kitty→half-cell demotion, and the first paint
    after mount replaces the pre-mount estimate.
    """

    #: Images are their own spacing kind: a change of subject against both the
    #: prompt above and the prose below, so the adaptive rule brackets them.
    SPACING_KIND: ClassVar[str] = "image"

    def __init__(
        self, data_b64: str | None, mime_type: str = "image/png", *, label: str = ""
    ) -> None:
        super().__init__()
        self.add_class("image-block")
        #: Short human name for the unavailable receipt (marker text or file
        #: name). Never rendered while the image itself is on screen — the
        #: caption already lives on the tool card or in the prompt text.
        self._label = label
        decoded = _decode(data_b64, mime_type) if data_b64 else None
        #: Original pixel dimensions, for aspect-ratio fits and the receipt.
        #: Kept as numbers because the pixels themselves are NOT kept at full
        #: size: a session that reads a dozen screenshots would otherwise hold
        #: a dozen multi-megabyte RGBA buffers in the widget tree for its
        #: whole life (the same hazard `UserBlock` documents for base64).
        #: `_shrink` caps the retained copy at the largest grid any render
        #: mode can use, so every later paint has full resolution FOR ITS
        #: GRID while the resident cost stays a few hundred KB per image.
        self._px_width = decoded.width if decoded is not None else 0
        self._px_height = decoded.height if decoded is not None else 0
        self._pil: "PILImage | None" = _shrink(decoded) if decoded is not None else None
        #: Why there is no picture, shown by the receipt. ``None`` = healthy.
        self._unavailable: str | None = None
        if self._pil is None:
            self._unavailable = (
                "no longer in the transcript" if not data_b64 else "could not be decoded"
            )
        #: Kitty state: the terminal-side image id once transmitted, and the
        #: grid the current placement was made for (re-place only on change).
        self._kitty_id: int | None = None
        self._placed: tuple[int, int] | None = None
        #: Pixel aspect of the letterboxed frame actually transmitted — what
        #: :meth:`_build_kitty` compares a new grid against to decide between
        #: a free re-place and a retransmit (review round 2, F8).
        self._transmit_aspect: float | None = None
        #: The mode this block actually painted with. Resolved at first paint
        #: (not construction) so the escape writer installed at app mount is
        #: visible; pinned afterwards except for demotion.
        self._mode: str | None = None
        #: Memoized half-cell frame, keyed by grid, so scroll repaints and
        #: no-op resizes never re-run the pixel walk.
        self._halfcell_cache: tuple[tuple[int, int], Text] | None = None
        self.set_content(self._build())
        self.finalize()

    # -- rendering ---------------------------------------------------------

    def _grid(self) -> tuple[int, int]:
        """The ``(cols, rows)`` this image gets at the current width.

        Fitted against the ORIGINAL pixel dimensions, not the retained
        (possibly downscaled) copy: the fit's job is the true aspect ratio
        and the no-upscale rule, both properties of the source image.
        """
        width = self.size.width or 80
        avail = max(8, width - SPINE_INDENT)
        return images_mod.fit_cells(
            self._px_width,
            self._px_height,
            min(avail, images_mod.MAX_COLS),
            images_mod.MAX_ROWS,
        )

    def _build(self) -> Text:
        if self._pil is None:
            return self._build_unavailable()
        cols, rows = self._grid()
        mode = self._resolve_mode()
        if mode == "kitty":
            frame = self._build_kitty(cols, rows)
            if frame is not None:
                self.styles.height = rows
                return frame
            # The transmit could not be written (no driver yet, or the write
            # failed): paint half-cells now rather than an empty reservation.
            self._mode = "halfcell"
        if self._mode == "halfcell":
            self.styles.height = rows
            return self._build_halfcell(cols, rows)
        return self._build_receipt()

    def _resolve_mode(self) -> str:
        if self._mode is None:
            self._mode = images_mod.detect_mode()
        return self._mode

    def _build_kitty(self, cols: int, rows: int) -> Text | None:
        """Transmit/place as needed, then the placeholder grid as text.

        Returns ``None`` when the escapes cannot reach the terminal, which
        tells :meth:`_build` to demote. The escapes go through the app's
        driver sink (see :func:`~local_operator.tui.images.set_escape_writer`)
        so they cannot interleave with a frame Textual is writing.
        """
        assert self._pil is not None
        cell = images_mod.cell_size()
        box_aspect = (cols * cell.width) / max(1, rows * cell.height)
        if self._kitty_id is not None and self._placed != (cols, rows):
            # The transmitted frame was letterboxed for the aspect of the grid
            # it was transmitted for. A resize that lands on a grid of the
            # SAME aspect (the common case: width changes rarely move the
            # fit's shape) just re-places; one whose aspect materially moved
            # would stretch the old bars into the picture (review round 2,
            # F8), so the image is retransmitted padded for the new grid.
            # Bounded: only an actual grid-aspect change pays it, and the
            # pixels come from the retained capped copy — tens of KB.
            if abs(box_aspect - (self._transmit_aspect or box_aspect)) > 0.01:
                images_mod.release_live(self._kitty_id)
                self._kitty_id = None
                self._placed = None
        if self._kitty_id is None:
            image_id = images_mod.next_image_id()
            frame = self._transmit_frame(cols, rows)
            payload = images_mod.encode_png_base64(images_mod.to_png(frame))
            if not images_mod.write_escape(images_mod.encode_transmit(image_id, payload)):
                return None
            self._kitty_id = image_id
            self._transmit_aspect = frame.width / max(1, frame.height)
            images_mod.register_live(image_id, self._demote_to_halfcell)
        if self._placed != (cols, rows):
            if not images_mod.write_escape(images_mod.encode_placement(self._kitty_id, cols, rows)):
                # The transmit landed but the placement could not be written:
                # without this release the terminal would keep an image that
                # will never be placed, parked in one of the 8 budget slots
                # until unmount (review round 1, F1). Reclaim it and let the
                # caller demote.
                images_mod.release_live(self._kitty_id)
                self._kitty_id = None
                self._placed = None
                self._transmit_aspect = None
                return None
            self._placed = (cols, rows)
        grid = images_mod.placeholder_grid(rows, cols)
        image_id = self._kitty_id
        style = Style(
            color=f"rgb({(image_id >> 16) & 255},{(image_id >> 8) & 255},{image_id & 255})"
        )
        indent = " " * SPINE_INDENT
        text = Text(no_wrap=True)
        for index, row in enumerate(grid):
            if index:
                text.append("\n")
            text.append(indent)
            text.append(row, style=style)
        return text

    def _transmit_frame(self, cols: int, rows: int) -> "PILImage":
        """The pixels sent to the terminal, letterboxed to the PLACEMENT grid.

        The letterbox exists because a kitty placement STRETCHES the image to
        fill its ``c=`` x ``r=`` cell rectangle, and the rectangle's aspect
        only approximates the image's (cells are integers). For a 12-row
        screenshot the rounding error is invisible; for a 2-row icon it was a
        measured 25% vertical stretch (design round 1, D2). Padding the frame
        to exactly the grid rectangle's pixel aspect with TRANSPARENT bars
        makes the stretch-to-fill aspect-true — the bars render as the
        terminal's own background.

        Padded for the grid the placement will actually use, NOT the cap
        grid: padding for the cap while placing into the current-width grid
        compounds the two rectangles' aspect errors — measured 22% off for a
        wide image in a 44-column terminal (review round 2, F8). The pixels
        INSIDE the bars still come from the retained cap-sized copy, scaled
        down to fit the grid box at their true aspect, so nothing upscales.
        A later resize whose grid keeps this aspect re-places for free;
        :meth:`_build_kitty` retransmits when the aspect itself moves.
        """
        assert self._pil is not None
        from PIL import Image as PIL_Image

        cell = images_mod.cell_size()
        box = (max(1, cols * cell.width), max(1, rows * cell.height))
        frame = self._pil if self._pil.mode == "RGBA" else self._pil.convert("RGBA")
        # Scale the retained pixels down to fit inside the box at their true
        # aspect (never up — the retained copy is already cap-bounded).
        scale = min(box[0] / frame.width, box[1] / frame.height, 1.0)
        if scale < 1.0:
            frame = frame.resize(
                (max(1, round(frame.width * scale)), max(1, round(frame.height * scale))),
                PIL_Image.Resampling.LANCZOS,
            )
        if (frame.width, frame.height) == box:
            return frame
        canvas = PIL_Image.new("RGBA", box, (0, 0, 0, 0))
        canvas.paste(frame, ((box[0] - frame.width) // 2, (box[1] - frame.height) // 2))
        return canvas

    def _build_halfcell(self, cols: int, rows: int) -> Text:
        """The image as ``▀`` cells: top pixel in fg ink, bottom in bg.

        Transparent pixels composite over the theme background so a logo on
        alpha does not paint black squares on paper themes. Consecutive
        same-colour cells share one styled span — a screenshot is mostly runs,
        so the segment count stays far under cols x rows.
        """
        assert self._pil is not None
        if self._halfcell_cache is not None and self._halfcell_cache[0] == (cols, rows):
            return self._halfcell_cache[1]
        from PIL import Image as PIL_Image

        # LETTERBOX, never stretch-to-fill (design round 1, D2): the grid's
        # display aspect only approximates the image's, and at small grids the
        # rounding error is gross. The image is scaled to the largest
        # half-cell-pixel rectangle that fits INSIDE the grid at its true
        # display aspect (a half-cell pixel measures cell.width x
        # cell.height/2 real pixels), then centered; the bars are theme
        # background, i.e. invisible.
        cell = images_mod.cell_size()
        grid_w, grid_h = cols, rows * 2
        # Clamped at 1.0: grid quantization can hand a tiny image a grid a
        # shade larger than its pixels, and filling it would upscale.
        scale = min(
            (cols * cell.width) / max(1, self._px_width),
            (rows * cell.height) / max(1, self._px_height),
            1.0,
        )
        inner_w = min(grid_w, max(1, round(self._px_width * scale / cell.width)))
        inner_h = min(grid_h, max(1, round(2 * self._px_height * scale / cell.height)))
        frame = self._pil.convert("RGBA").resize((inner_w, inner_h), PIL_Image.Resampling.LANCZOS)
        background = theme_mod.semantic_color("bg").lstrip("#")
        bg_rgb = tuple(int(background[i : i + 2], 16) for i in (0, 2, 4))
        canvas = PIL_Image.new("RGBA", (grid_w, grid_h), (*bg_rgb, 255))
        canvas.alpha_composite(frame, ((grid_w - inner_w) // 2, (grid_h - inner_h) // 2))
        pixels = canvas.load()
        assert pixels is not None
        indent = " " * SPINE_INDENT
        text = Text(no_wrap=True)
        for row in range(rows):
            if row:
                text.append("\n")
            text.append(indent)
            run_start = 0
            run_style: str | None = None
            line: list[tuple[str, str]] = []
            for col in range(cols):
                # RGBA canvas => getpixel returns a 4-int tuple; the cast
                # narrows PIL's float|tuple union for the type checker.
                top = cast("tuple[int, ...]", pixels[col, row * 2])
                bottom = cast("tuple[int, ...]", pixels[col, row * 2 + 1])
                style = f"{_rgb(top)} on {_rgb(bottom)}"
                if style != run_style:
                    if run_style is not None:
                        line.append((_HALF * (col - run_start), run_style))
                    run_start = col
                    run_style = style
            if run_style is not None:
                line.append((_HALF * (cols - run_start), run_style))
            for glyphs, style in line:
                text.append(glyphs, style=style)
        self._halfcell_cache = ((cols, rows), text)
        return text

    def _build_receipt(self) -> Text:
        """Text mode: one dim row acknowledging the image without pixels."""
        self.styles.height = 1
        detail = f"{self._px_width}x{self._px_height}"
        if self._label:
            detail = f"'{self._label}', {detail}"
        return self._receipt_row(f"image attached ({detail})", glyph=_ATTACHED_GLYPH)

    def _build_unavailable(self) -> Text:
        """The missing-image receipt: what was here, and why it is not."""
        self.styles.height = 1
        reason = self._unavailable or "unavailable"
        # Quoted: an unquoted filename spliced into the sentence reads as a
        # typo (`image dashboard.png unavailable`) — design round 1, D4.
        name = f" '{self._label}'" if self._label else ""
        return self._receipt_row(f"image{name} unavailable — {reason}")

    def _receipt_row(self, message: str, glyph: str = _UNAVAILABLE_GLYPH) -> Text:
        """One receipt row, truncated to the block's width with a real ``…``.

        Truncated HERE, in the string, not via ``Text(no_wrap=True,
        overflow="ellipsis")``: ``set_content`` promotes a ``Text`` through
        ``Content.from_rich_text``, which drops both flags — so the flagged
        version WRAPPED in measurement (2 rows at 44 columns) while
        ``styles.height = 1`` clipped the second row invisibly, losing the
        reason mid-phrase and opening a phantom spacing gap from the 2-row
        measurement (design round 1, D1). A pre-truncated string measures one
        row at every width, so the frame and the measurement cannot disagree.
        """
        from rich.cells import cell_len

        style = Style(color=theme_mod.semantic_color("muted"))
        lead = f"{glyph} "
        room = max(8, (self.size.width or 80) - SPINE_INDENT - cell_len(lead))
        if cell_len(message) > room:
            from rich.cells import set_cell_size

            message = set_cell_size(message, room - 1) + "…"
        text = Text(no_wrap=True)
        text.append(" " * SPINE_INDENT, style=style)
        text.append(lead, style=style)
        text.append(message, style=style)
        return text

    # -- lifecycle ---------------------------------------------------------

    def _repaint(self) -> None:
        """Rebuild through the finalize dance (see class docstring)."""
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build())
        finally:
            self._finalized = was_finalized

    def on_mount(self) -> None:
        """First paint at the real width (construction guessed 80 cols)."""
        self._repaint()

    def on_resize(self, event: object) -> None:
        """Re-fit at the new width; a changed grid is a height change.

        Receipts repaint too: their truncation point is a function of the
        width (see :meth:`_receipt_row`), so a receipt built at 100 columns
        and left alone would clip rather than ellipsize at 44.
        """
        self._repaint()
        parent = self.parent
        refresh = getattr(parent, "refresh_gap_around", None)
        if callable(refresh):
            refresh(self)

    def on_unmount(self) -> None:
        """Return the terminal-side image store entry (``/clear``, exit)."""
        if self._kitty_id is not None:
            images_mod.release_live(self._kitty_id)
            self._kitty_id = None
            self._placed = None
            self._transmit_aspect = None

    def _demote_to_halfcell(self) -> None:
        """Budget eviction: keep the picture, drop the terminal store entry.

        Called by :func:`~local_operator.tui.images.register_live` when this
        image is the oldest past the cap. The terminal-side delete already
        happened; this swaps the placeholder text for half-cell pixels so the
        block stays a picture in scrollback.
        """
        self._kitty_id = None
        self._placed = None
        self._transmit_aspect = None
        self._mode = "halfcell"
        self._repaint()

    # -- selection ---------------------------------------------------------

    def copy_row_is_chrome(self, index: int) -> bool:
        """Every row is furniture: a drag-copy must never paste placeholder
        codepoints or half-block soup into a document. The image is not text,
        so the clipboard gets nothing from it."""
        return True
