"""Inline image rendering for the transcript: protocol detection, fit
arithmetic, kitty graphics encoding, and the half-cell fallback.

The TUI shows images three ways, best available first:

- **Kitty graphics via Unicode placeholders** (``U=1`` + U+10EEEE cells).
  The image bytes are transmitted to the terminal ONCE (APC ``a=t``), a
  virtual placement names a cell grid for them (``a=p,U=1``), and the grid
  is then drawn as ordinary text: each cell is the placeholder character
  plus row/column combining diacritics, with the image id carried in the
  foreground colour. Because the cells are text, Textual's compositor can
  scroll, clip and repaint them like any other content — which is the whole
  reason this mode works inside a compositing TUI at all. A direct
  cursor-positioned placement (``a=p`` without ``U=1``) or a raw Sixel/iTerm2
  blob would be painted once and then shredded by the next compositor frame,
  so those protocols are deliberately not offered.
- **Half-cell pixels**: the image is downscaled with Pillow and painted as
  ``▀`` characters, foreground = top pixel, background = bottom pixel. Pure
  text and colours, so it works in every terminal, over SSH, inside tmux,
  and in the SVG screenshots the test harness exports.
- **A one-line text receipt** naming the format and dimensions, for
  ``display.images = false`` or a terminal where even half-cells would be
  noise (``TERM=dumb``).

Protocol references: kitty ``docs/graphics-protocol.rst`` ("Unicode
placeholders"), and omp's ``packages/tui/src/kitty-graphics.ts``, whose
placeholder strategy this module mirrors. The performance posture is
deliberately stricter than omp's: pixels cross the wire exactly once per
image (downscaled to the placement grid, so a 4 MB screenshot transmits a
few tens of KB), a resize re-places the existing terminal-side image without
retransmitting, and everything after the transmit is plain text repaints.
"""

from __future__ import annotations

import base64
import io
import struct
import sys
from collections import OrderedDict
from typing import Callable

from local_operator.tui.settings import settings_get

__all__ = [
    "CellSize",
    "ImageMode",
    "MAX_LIVE_KITTY_IMAGES",
    "cell_size",
    "detect_mode",
    "encode_delete",
    "encode_placement",
    "encode_transmit",
    "fit_cells",
    "next_image_id",
    "placeholder_grid",
    "register_live",
    "release_live",
    "reset_for_tests",
    "set_escape_writer",
    "write_escape",
]

#: Environment kill switch / override. ``kitty``/``halfcell``/``text`` force a
#: mode; ``off`` is accepted as an alias for ``text`` (the receipt line is the
#: floor — silently showing nothing would hide that an image arrived at all).
_ENV_MODE = "LOCAL_OPERATOR_IMAGES"

#: How many kitty images stay live in the terminal's own store at once.
#:
#: Kitty-protocol terminals keep every transmitted image in a per-terminal
#: store; text clears (``CSI 2J``) do not remove them, so an unbounded session
#: that reads many screenshots piles up store memory for images long scrolled
#: away. omp bounds this at 8 and demotes older images to their text fallback;
#: the budget here does the same but demotes to HALF-CELLS, so an evicted
#: image stays a picture in scrollback — it just stops costing terminal store.
MAX_LIVE_KITTY_IMAGES = 8

#: Grid caps. ``MAX_ROWS`` is the consistent-size rule: every image lands at
#: the same height ceiling so a transcript of mixed screenshots reads as a
#: ledger rather than a scrapbook, with width following each image's own
#: aspect ratio under ``MAX_COLS``. Small images stay small (no upscaling).
MAX_ROWS = 12
MAX_COLS = 72

#: Kitty Unicode placeholder base character (U+10EEEE, Plane 16 PUA).
PLACEHOLDER = "\U0010eeee"

#: Row/column combining diacritics naming a placeholder cell's coordinates.
#: Index ``i`` -> codepoint, from kitty ``gen/rowcolumn-diacritics.txt``
#: (Unicode 6.0.0 NSM set, combining class 230, no decomposition). 297
#: entries, so one image can address a 297x297 cell grid — far past the caps
#: above. The same table omp and textual-image carry.
ROWCOLUMN_DIACRITICS: tuple[int, ...] = (
    0x0305,
    0x030D,
    0x030E,
    0x0310,
    0x0312,
    0x033D,
    0x033E,
    0x033F,
    0x0346,
    0x034A,
    0x034B,
    0x034C,
    0x0350,
    0x0351,
    0x0352,
    0x0357,
    0x035B,
    0x0363,
    0x0364,
    0x0365,
    0x0366,
    0x0367,
    0x0368,
    0x0369,
    0x036A,
    0x036B,
    0x036C,
    0x036D,
    0x036E,
    0x036F,
    0x0483,
    0x0484,
    0x0485,
    0x0486,
    0x0487,
    0x0592,
    0x0593,
    0x0594,
    0x0595,
    0x0597,
    0x0598,
    0x0599,
    0x059C,
    0x059D,
    0x059E,
    0x059F,
    0x05A0,
    0x05A1,
    0x05A8,
    0x05A9,
    0x05AB,
    0x05AC,
    0x05AF,
    0x05C4,
    0x0610,
    0x0611,
    0x0612,
    0x0613,
    0x0614,
    0x0615,
    0x0616,
    0x0617,
    0x0657,
    0x0658,
    0x0659,
    0x065A,
    0x065B,
    0x065D,
    0x065E,
    0x06D6,
    0x06D7,
    0x06D8,
    0x06D9,
    0x06DA,
    0x06DB,
    0x06DC,
    0x06DF,
    0x06E0,
    0x06E1,
    0x06E2,
    0x06E4,
    0x06E7,
    0x06E8,
    0x06EB,
    0x06EC,
    0x0730,
    0x0732,
    0x0733,
    0x0735,
    0x0736,
    0x073A,
    0x073D,
    0x073F,
    0x0740,
    0x0741,
    0x0743,
    0x0745,
    0x0747,
    0x0749,
    0x074A,
    0x07EB,
    0x07EC,
    0x07ED,
    0x07EE,
    0x07EF,
    0x07F0,
    0x07F1,
    0x07F3,
    0x0816,
    0x0817,
    0x0818,
    0x0819,
    0x081B,
    0x081C,
    0x081D,
    0x081E,
    0x081F,
    0x0820,
    0x0821,
    0x0822,
    0x0823,
    0x0825,
    0x0826,
    0x0827,
    0x0829,
    0x082A,
    0x082B,
    0x082C,
    0x082D,
    0x0951,
    0x0953,
    0x0954,
    0x0F82,
    0x0F83,
    0x0F86,
    0x0F87,
    0x135D,
    0x135E,
    0x135F,
    0x17DD,
    0x193A,
    0x1A17,
    0x1A75,
    0x1A76,
    0x1A77,
    0x1A78,
    0x1A79,
    0x1A7A,
    0x1A7B,
    0x1A7C,
    0x1B6B,
    0x1B6D,
    0x1B6E,
    0x1B6F,
    0x1B70,
    0x1B71,
    0x1B72,
    0x1B73,
    0x1CD0,
    0x1CD1,
    0x1CD2,
    0x1CDA,
    0x1CDB,
    0x1CE0,
    0x1DC0,
    0x1DC1,
    0x1DC3,
    0x1DC4,
    0x1DC5,
    0x1DC6,
    0x1DC7,
    0x1DC8,
    0x1DC9,
    0x1DCB,
    0x1DCC,
    0x1DD1,
    0x1DD2,
    0x1DD3,
    0x1DD4,
    0x1DD5,
    0x1DD6,
    0x1DD7,
    0x1DD8,
    0x1DD9,
    0x1DDA,
    0x1DDB,
    0x1DDC,
    0x1DDD,
    0x1DDE,
    0x1DDF,
    0x1DE0,
    0x1DE1,
    0x1DE2,
    0x1DE3,
    0x1DE4,
    0x1DE5,
    0x1DE6,
    0x1DFE,
    0x20D0,
    0x20D1,
    0x20D4,
    0x20D5,
    0x20D6,
    0x20D7,
    0x20DB,
    0x20DC,
    0x20E1,
    0x20E7,
    0x20E9,
    0x20F0,
    0x2CEF,
    0x2CF0,
    0x2CF1,
    0x2DE0,
    0x2DE1,
    0x2DE2,
    0x2DE3,
    0x2DE4,
    0x2DE5,
    0x2DE6,
    0x2DE7,
    0x2DE8,
    0x2DE9,
    0x2DEA,
    0x2DEB,
    0x2DEC,
    0x2DED,
    0x2DEE,
    0x2DEF,
    0x2DF0,
    0x2DF1,
    0x2DF2,
    0x2DF3,
    0x2DF4,
    0x2DF5,
    0x2DF6,
    0x2DF7,
    0x2DF8,
    0x2DF9,
    0x2DFA,
    0x2DFB,
    0x2DFC,
    0x2DFD,
    0x2DFE,
    0x2DFF,
    0xA66F,
    0xA67C,
    0xA67D,
    0xA6F0,
    0xA6F1,
    0xA8E0,
    0xA8E1,
    0xA8E2,
    0xA8E3,
    0xA8E4,
    0xA8E5,
    0xA8E6,
    0xA8E7,
    0xA8E8,
    0xA8E9,
    0xA8EA,
    0xA8EB,
    0xA8EC,
    0xA8ED,
    0xA8EE,
    0xA8EF,
    0xA8F0,
    0xA8F1,
    0xAAB0,
    0xAAB2,
    0xAAB3,
    0xAAB7,
    0xAAB8,
    0xAABE,
    0xAABF,
    0xAAC1,
    0xFE20,
    0xFE21,
    0xFE22,
    0xFE23,
    0xFE24,
    0xFE25,
    0xFE26,
    0x10A0F,
    0x10A38,
    0x1D185,
    0x1D186,
    0x1D187,
    0x1D188,
    0x1D189,
    0x1D1AA,
    0x1D1AB,
    0x1D1AC,
    0x1D1AD,
    0x1D242,
    0x1D243,
    0x1D244,
)

_APC_START = "\x1b_G"
_APC_END = "\x1b\\"

#: Base64 payload chunk size for transmits, per the kitty spec's 4096 limit.
_CHUNK = 4096


# ---------------------------------------------------------------------------
# Mode detection and cell geometry
# ---------------------------------------------------------------------------

ImageMode = str  # "kitty" | "halfcell" | "text"

#: Memoized answers. Detection reads the environment and (for cell size) an
#: ioctl; both are stable for the life of the process, and re-running them
#: per block would put an ioctl on every image append.
_mode_cache: ImageMode | None = None
_cell_cache: "CellSize | None" = None


class CellSize:
    """One terminal cell's pixel footprint, from ``TIOCGWINSZ``.

    The window-size ioctl carries the window's pixel dimensions alongside its
    cell dimensions on terminals that report them (ghostty and kitty do), and
    it is a pure kernel read: unlike the escape-based cell-size queries omp
    and textual-image use, it needs no terminal response, so it stays safe to
    call while Textual owns stdin. Terminals that report zero pixels fall
    back to 8x16 — the classic bitmap-font cell, and the same default
    textual-image lands on.
    """

    __slots__ = ("width", "height")

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


def cell_size() -> CellSize:
    """The terminal's cell pixel size, queried once and memoized."""
    global _cell_cache
    if _cell_cache is None:
        _cell_cache = _query_cell_size()
    return _cell_cache


def _query_cell_size() -> CellSize:
    try:
        import fcntl
        import termios

        stdout = sys.__stdout__
        if stdout is None:
            raise OSError("no stdout")
        packed = fcntl.ioctl(stdout.fileno(), termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        rows, cols, xpixel, ypixel = struct.unpack("HHHH", packed)
        if rows > 0 and cols > 0 and xpixel > 0 and ypixel > 0:
            return CellSize(max(1, xpixel // cols), max(1, ypixel // rows))
    except Exception:
        pass
    return CellSize(8, 16)


def detect_mode() -> ImageMode:
    """Which rendering mode this terminal gets, best first, memoized.

    Kitty graphics are claimed only where Unicode placeholders are KNOWN to
    render: kitty itself (``TERM=xterm-kitty`` / ``KITTY_WINDOW_ID``) and
    ghostty (which is also what cmux embeds). WezTerm and Konsole advertise
    the kitty protocol but do not render placeholder cells (the same finding
    textual-image documents), so they take half-cells. A multiplexer between
    us and the terminal (tmux/screen) would need APC passthrough wrapping and
    pane-scroll bookkeeping this deliberately does not attempt — omp gates
    the same way — so it also takes half-cells, which are plain text and
    survive anything. ``TERM=dumb`` gets the receipt line.
    """
    global _mode_cache
    if _mode_cache is None:
        _mode_cache = _detect_mode()
    return _mode_cache


def _detect_mode() -> ImageMode:
    import os

    forced = (os.environ.get(_ENV_MODE) or "").strip().lower()
    if forced in ("kitty", "halfcell", "text"):
        return forced
    if forced == "off":
        return "text"
    if not settings_get("display.images", True):
        return "text"
    term = os.environ.get("TERM", "")
    if term == "dumb" or not term:
        return "text"
    if os.environ.get("TMUX") or term.startswith(("screen", "tmux")):
        return "halfcell"
    term_program = (os.environ.get("TERM_PROGRAM") or "").lower()
    if term == "xterm-kitty" or os.environ.get("KITTY_WINDOW_ID") or term_program == "ghostty":
        return "kitty"
    return "halfcell"


def reset_for_tests() -> None:
    """Drop every module-level memo and registry (test isolation hook)."""
    global _mode_cache, _cell_cache, _writer
    _mode_cache = None
    _cell_cache = None
    _writer = None
    _live.clear()


# ---------------------------------------------------------------------------
# Fit arithmetic
# ---------------------------------------------------------------------------


def fit_cells(
    px_width: int,
    px_height: int,
    max_cols: int,
    max_rows: int,
    cell: CellSize | None = None,
) -> tuple[int, int]:
    """The ``(cols, rows)`` cell grid ``px_width`` x ``px_height`` fits into.

    Aspect ratio is preserved in PIXELS, not cells: a cell is roughly twice
    as tall as it is wide, so the ratio must be computed against the cell's
    real pixel footprint or every image renders half its true height. The
    scale never exceeds 1.0 — an icon smaller than the grid stays icon-sized
    rather than being blown up into mush — and both axes are clamped to at
    least one cell so a 1px-tall tracking pixel still occupies a row instead
    of producing a zero-height widget.
    """
    cell = cell or cell_size()
    if px_width <= 0 or px_height <= 0:
        return (1, 1)
    max_cols = max(1, max_cols)
    max_rows = max(1, max_rows)
    scale = min(
        (max_cols * cell.width) / px_width,
        (max_rows * cell.height) / px_height,
        1.0,
    )
    cols = max(1, round((px_width * scale) / cell.width))
    rows = max(1, round((px_height * scale) / cell.height))
    return (min(cols, max_cols), min(rows, max_rows))


# ---------------------------------------------------------------------------
# Kitty escape encoding (pure string builders — writing is the caller's job)
# ---------------------------------------------------------------------------

#: Process-global image id allocator. Kitty image ids live in the terminal's
#: OWN registry, which outlives this process and is shared with every other
#: program that ever transmitted an image to the same terminal — so ids start
#: from a random seed (omp does the same) to avoid stomping a neighbour's
#: placements, and stay within 24 bits so the id round-trips through the
#: placeholder cell's RGB foreground colour without needing the fourth-byte
#: diacritic.
_next_id: int | None = None


def next_image_id() -> int:
    """A fresh 24-bit kitty image id, randomly seeded per process."""
    global _next_id
    if _next_id is None:
        import random

        _next_id = random.randint(1, 0xFFFFFF)
    _next_id = (_next_id % 0xFFFFFF) + 1
    return _next_id


def encode_transmit(image_id: int, png_base64: str) -> str:
    """The chunked APC transmit sequence for ``png_base64`` as ``image_id``.

    ``f=100`` declares PNG data (the one compressed format the protocol
    takes; callers convert other formats first), ``q=2`` suppresses the
    terminal's acknowledgement — Textual owns stdin, so a response would land
    in its key parser as garbage input.
    """
    parts: list[str] = []
    offset = 0
    first = True
    total = len(png_base64)
    while offset < total or first:
        chunk = png_base64[offset : offset + _CHUNK]
        offset += _CHUNK
        more = 1 if offset < total else 0
        if first:
            parts.append(f"{_APC_START}a=t,i={image_id},f=100,q=2,m={more};{chunk}{_APC_END}")
            first = False
        else:
            parts.append(f"{_APC_START}m={more};{chunk}{_APC_END}")
    return "".join(parts)


def encode_placement(image_id: int, cols: int, rows: int) -> str:
    """The virtual-placement APC: ``image_id`` scaled onto a cols x rows grid.

    ``U=1`` makes it a Unicode-placeholder placement (displayed wherever the
    placeholder cells land, not at the cursor). A fixed ``p=1`` placement id
    means re-sending this for the same image REPLACES the placement instead
    of stacking a second one — which is what makes a resize re-place without
    retransmitting or leaking placements.
    """
    return f"{_APC_START}a=p,i={image_id},p=1,U=1,c={cols},r={rows},q=2{_APC_END}"


def encode_delete(image_id: int) -> str:
    """Delete ``image_id`` and its placements from the terminal's store."""
    return f"{_APC_START}a=d,d=I,i={image_id},q=2{_APC_END}"


def placeholder_grid(rows: int, cols: int) -> list[str]:
    """The placeholder text rows for a ``rows`` x ``cols`` placement.

    Each cell is U+10EEEE plus a row diacritic plus a column diacritic; the
    terminal maps the cell to the placement's sub-rectangle from those two
    marks and the image id in the cell's foreground colour. Grids are capped
    far below the diacritic table's 297 entries, so indexing cannot overrun.
    """
    return [
        "".join(
            PLACEHOLDER + chr(ROWCOLUMN_DIACRITICS[row]) + chr(ROWCOLUMN_DIACRITICS[col])
            for col in range(cols)
        )
        for row in range(rows)
    ]


def encode_png_base64(data: bytes) -> str:
    """``data`` (already PNG bytes) as the base64 payload string."""
    return base64.b64encode(data).decode("ascii")


def to_png(pil_image: "object") -> bytes:
    """Encode a PIL image to PNG bytes (the transmit wire format)."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")  # type: ignore[attr-defined]
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Escape writer and the live-image budget
# ---------------------------------------------------------------------------

#: Where APC escapes go. Textual serialises every byte it paints through a
#: writer thread, so a second writer (``sys.stdout``) would interleave an APC
#: into the middle of a frame and corrupt it. The app installs
#: ``driver.write`` here — the same door it hands ``TerminalTitle`` and the
#: notifier, for the same reason. ``None`` (headless, tests, exec mode) makes
#: every write a no-op: the placeholder cells still render, they just have no
#: image behind them, which is invisible because headless frames go nowhere.
_writer: Callable[[str], None] | None = None

#: Live kitty images, oldest first: image id -> demote callback. The values
#: are callbacks rather than widget references so this module never holds a
#: widget alive past its unmount.
_live: "OrderedDict[int, Callable[[], None]]" = OrderedDict()


def set_escape_writer(writer: Callable[[str], None] | None) -> None:
    """Install (or clear) the sink APC escapes are written through."""
    global _writer
    _writer = writer


def write_escape(sequence: str) -> bool:
    """Write ``sequence`` to the terminal; ``False`` when there is no sink."""
    if _writer is None:
        return False
    try:
        _writer(sequence)
        return True
    except Exception:
        return False


def register_live(image_id: int, demote: Callable[[], None]) -> None:
    """Admit ``image_id`` to the live-image budget, evicting past the cap.

    Eviction deletes the OLDEST image from the terminal store and calls its
    ``demote`` callback, which repaints that block as half-cells. The evicted
    image therefore stays visible in scrollback as pixels-in-text; only the
    terminal-side store entry is reclaimed. Callbacks run on the UI thread
    (registration only ever happens from widget mount/render paths).
    """
    _live[image_id] = demote
    _live.move_to_end(image_id)
    while len(_live) > MAX_LIVE_KITTY_IMAGES:
        old_id, old_demote = _live.popitem(last=False)
        write_escape(encode_delete(old_id))
        try:
            old_demote()
        except Exception:
            pass  # a demotion repaint must never take the app down


def release_live(image_id: int) -> None:
    """Remove ``image_id`` from the budget and the terminal store.

    Called on widget unmount — ``/clear``, transcript teardown, app exit —
    so the terminal's image registry is left exactly as this process found
    it. Safe to call for ids that were never registered or already evicted.
    """
    if _live.pop(image_id, None) is not None:
        write_escape(encode_delete(image_id))
