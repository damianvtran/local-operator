"""Shimmer — a dim -> muted -> accent band sweeping text.

A cosine band advances at a fixed velocity (30 cells/s) and
recolors the characters under it. Characters outside the band sit in the LOW
tier; the crest reads bold accent. One aggregate working line (D25) and the
streaming assistant indicator ride this; individual running rows keep a quiet
static marker, and a settings-off gate falls back to a still marker (D26) so
running state stays legible in a still frame.

Everything here is allocation-light: per-character styles are built once and
shared (no per-char ``Style`` objects), and the band math is pure integer +
one cosine per character.
"""

from __future__ import annotations

import math
import os
import time

from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.settings import settings_get

#: Band speed — sweeps 30 cells per second.
SHIMMER_SPEED_CELLS_PER_S = 30.0
#: Padding on each side of the text the band travels through.
CLASSIC_PADDING = 10
#: Half-width of the cosine band in cells.
CLASSIC_BAND_HALF_WIDTH = 6
#: Intensity tier thresholds.
TIER_HIGH = 0.65
TIER_MID = 0.22

#: Environment kill switch — CI and snapshot harnesses pin still frames.
_ENV_DISABLE = "LOCAL_OPERATOR_NO_SHIMMER"


def _style(token: str, bold: bool = False) -> Style:
    return Style(color=theme_mod.semantic_color(token), bold=bold)


def shimmer_enabled() -> bool:
    """Whether shimmer animation is active (D26: settings flag + env gate)."""
    if os.environ.get(_ENV_DISABLE):
        return False
    return bool(settings_get("display.shimmer", True))


def classic_intensity(time_ms: float, index: int, length: int) -> float:
    """Band intensity 0..1 for the character at ``index`` at ``time_ms``."""
    period = length + CLASSIC_PADDING * 2
    pos = ((time_ms / 1000.0) * SHIMMER_SPEED_CELLS_PER_S) % period
    dist = abs(index + CLASSIC_PADDING - pos)
    if dist >= CLASSIC_BAND_HALF_WIDTH:
        return 0.0
    return 0.5 * (1 + math.cos((math.pi * dist) / CLASSIC_BAND_HALF_WIDTH))


def shimmer_text(text: str, time_ms: float | None = None) -> Text:
    """Sweep a dim -> muted -> accent crest across ``text`` (classic shimmer).

    ``time_ms`` defaults to the monotonic clock; pass an explicit value in
    tests for deterministic frames. Disabled shimmer (flag or env) returns
    the text styled flat dim — the D26 static fallback.
    """
    if not shimmer_enabled():
        return Text(text, style=_style("dim"))
    if time_ms is None:
        time_ms = time.monotonic() * 1000.0
    low = _style("dim")
    mid = _style("muted")
    high = _style("accent", bold=True)
    out = Text(text)
    length = len(text)
    for i in range(length):
        intensity = classic_intensity(time_ms, i, length)
        style = high if intensity >= TIER_HIGH else (mid if intensity >= TIER_MID else low)
        out.stylize(style, i, i + 1)
    return out
