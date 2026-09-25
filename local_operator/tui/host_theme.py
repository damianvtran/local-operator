"""The ``terminal`` theme: a ramp derived from the host terminal's own colors.

Every other theme paints fixed hexes, so a user's terminal theme (a blue
Ghostty background, say) never shows through. This asks the terminal what its
colors ARE — OSC 10/11 for the default foreground/background, OSC 4 for ANSI
0-15 — and builds an ordinary hex :class:`ThemeSpec` from the answer. Staying
hex is the point: the elevation ladder (``bg`` -> ``surface`` -> ``raised``)
is a background step, and ANSI-default colors (herdr's and btop's approach)
cannot express one, so they flatten every card. Deriving keeps depth.

The probe must run BEFORE Textual owns stdin (the replies arrive as input
bytes), so :func:`install` is called from ``run_tui`` ahead of the app. A DA1
query (``ESC [ c``) is appended as a sentinel: every terminal answers it, and
answers in order, so its reply means every color reply that is coming has
arrived and there is no fixed wait on a terminal that answers promptly.
"""

from __future__ import annotations

import logging
import os
import re
import select
import sys
import time

from local_operator.tui import theme as theme_mod

logger = logging.getLogger(__name__)

NAME = "terminal"

RGB = tuple[int, int, int]

_REPLY_RE = re.compile(
    rb"\x1b\](10|11|4;(\d+));"
    rb"(?:rgba?:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)(?:/[0-9a-fA-F]+)?|#([0-9a-fA-F]{6}))"
    rb"(?:\x07|\x1b\\)"
)
_DA1_RE = re.compile(rb"\x1b\[\?[\d;]*c")


def _component(text: bytes) -> int:
    # xterm replies scale to the digit count: "ff" and "ffff" are both full.
    return round(int(text, 16) * 255 / (16 ** len(text) - 1))


def parse_replies(data: bytes) -> tuple[RGB | None, RGB | None, dict[int, RGB]]:
    """``(fg, bg, ansi)`` from raw terminal reply bytes; absent replies are None."""
    fg = bg = None
    ansi: dict[int, RGB] = {}
    for match in _REPLY_RE.finditer(data):
        kind, index, r, g, b, hexed = match.groups()
        if hexed:
            rgb = (int(hexed[0:2], 16), int(hexed[2:4], 16), int(hexed[4:6], 16))
        else:
            rgb = (_component(r), _component(g), _component(b))
        if kind == b"10":
            fg = rgb
        elif kind == b"11":
            bg = rgb
        else:
            ansi[int(index)] = rgb
    return fg, bg, ansi


def probe(timeout: float = 0.3) -> bytes:
    """Query the controlling terminal; return its raw replies (b"" if none)."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return b""
    try:
        import termios
        import tty
    except ImportError:  # Windows: no termios, no probe, no terminal theme
        return b""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY)
    except OSError:
        return b""
    try:
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            query = "\x1b]10;?\x1b\\\x1b]11;?\x1b\\"
            query += "".join(f"\x1b]4;{i};?\x1b\\" for i in range(16))
            os.write(fd, (query + "\x1b[c").encode())
            data = b""
            deadline = time.monotonic() + timeout
            while not _DA1_RE.search(data):
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([fd], [], [], left)[0]:
                    break
                data += os.read(fd, 4096)
            return data
        finally:
            # TCSAFLUSH drops any reply that lands after we stop reading, so a
            # slow terminal's late bytes never reach Textual as keystrokes.
            termios.tcsetattr(fd, termios.TCSAFLUSH, saved)
    except (OSError, termios.error):
        return b""
    finally:
        os.close(fd)


# -- color math ----------------------------------------------------------------


def _hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def _luminance(rgb: RGB) -> float:
    def lin(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a: RGB, b: RGB) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def _ensure(color: RGB, grounds: list[RGB], floor: float, toward: RGB) -> RGB:
    """Nudge ``color`` toward ``toward`` until it reads ``floor``:1 on every ground."""
    for step in range(21):
        candidate = _mix(color, toward, step / 20)
        if all(_contrast(candidate, g) >= floor for g in grounds):
            return candidate
    return toward


# Brand dark hues, used for any ANSI slot the terminal did not report.
_FALLBACK: dict[int, RGB] = {
    1: (0xEF, 0x80, 0x78),
    2: (0x57, 0xC7, 0x85),
    3: (0xE0, 0xB0, 0x4B),
    4: (0x6E, 0xA8, 0xD8),
    5: (0xB4, 0x8C, 0xD6),
    10: (0x38, 0xC9, 0x6A),
}


def derive(fg: RGB, bg: RGB, ansi: dict[int, RGB]) -> theme_mod.ThemeSpec:
    """A full semantic ramp from the terminal's fg/bg and ANSI palette.

    Grounds step from ``bg`` toward ``fg`` (so they lighten on a dark terminal
    and darken on a light one); every ink is clamped to the same contrast
    floors the curated palettes are tested against.
    """
    dark = _luminance(bg) < 0.18
    extreme: RGB = (255, 255, 255) if dark else (0, 0, 0)

    surface = _mix(bg, fg, 0.06)
    raised = _mix(bg, fg, 0.10)
    overlay = _mix(bg, fg, 0.14)
    sunken = _mix(bg, (0, 0, 0), 0.3) if dark else _mix(bg, fg, 0.04)
    grounds = [bg, surface]

    ink = _ensure(fg, grounds, 7.0, extreme)
    muted = _ensure(_mix(ink, bg, 0.3), grounds, 4.5, ink)
    dim = _ensure(_mix(ink, bg, 0.5), grounds, 3.4, ink)
    faint = _mix(ink, bg, 0.72)
    edge = _mix(bg, ink, 0.18)
    edge_hi = _mix(bg, ink, 0.26)
    if _contrast(ink, edge) < 4.5:  # selected text sits on `edge`
        edge = _mix(bg, ink, 0.1)

    def hue(*slots: int) -> RGB:
        color = next((ansi[s] for s in slots if s in ansi), _FALLBACK[slots[-1]])
        return _ensure(color, grounds, 4.0, ink)

    green = hue(2)
    accent = hue(10, 2) if 10 in ansi else green
    danger = hue(9, 1) if 9 in ansi else hue(1)
    warning = hue(3)
    signal = hue(4)
    label = hue(5)

    tint_danger = _mix(surface, danger, 0.12)
    danger = _ensure(danger, [*grounds, tint_danger], 4.0, ink)
    tokens = {
        "bg": bg,
        "surface": surface,
        "raised": raised,
        "overlay": overlay,
        "sunken": sunken,
        "fg": ink,
        "muted": muted,
        "dim": dim,
        "faint": faint,
        "edge": edge,
        "edge-hi": edge_hi,
        "accent": accent,
        "success": green,
        "warning": warning,
        "danger": danger,
        "string": green,
        "signal": signal,
        "label": label,
        "tint-danger": tint_danger,
        "tint-select": _mix(surface, accent, 0.10),
        "tint-select-hi": _mix(surface, accent, 0.16),
        "tint-attach": _mix(surface, signal, 0.20),
        "tint-attach-hi": _mix(surface, signal, 0.35),
    }
    return theme_mod.ThemeSpec(
        name=NAME,
        label="Terminal",
        description="Your terminal's own colors, read from it at startup",
        dark=dark,
        tokens={k: _hex(v) for k, v in tokens.items()},
    )


def install() -> bool:
    """Probe the terminal and register the ``terminal`` theme; False if it did not answer."""
    fg, bg, ansi = parse_replies(probe())
    if fg is None or bg is None:
        logger.debug("terminal theme: no OSC 10/11 reply, theme not registered")
        return False
    theme_mod.unregister_theme(NAME)  # a re-entrant boot re-derives it
    theme_mod.register_theme(derive(fg, bg, ansi))
    return True
