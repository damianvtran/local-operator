"""Light themes beyond the brand paper ramp.

Light ramps invert every decision the dark ones make: elevation lifts
TOWARD white instead of away from black, state hues need to be darker
rather than brighter to hold contrast, and the tint grounds (``tint-*``)
sit just below the paper rather than just above the night. The brand
``light`` ramp in ``theme.py`` is the reference solve — mirror its
relationships, not its hexes.

Every palette must clear ``tests/unit/tui/test_palette_contrast.py``
(``dark=False`` flips the polarity checks) and be inspected as rendered
frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = []
