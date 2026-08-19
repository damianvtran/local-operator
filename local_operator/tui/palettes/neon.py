"""Neon / retro-futurist themes: synthwave, matrix, tron, and kin.

The design constraint that separates this family from the classics: the
VIBE is carried by the accent, signal and label hues plus the tinted
grounds — never by pushing the body ink to a saturated hue. ``fg`` stays a
near-neutral (a cold white, a pale green-white) so hours of prose reading
do not fatigue; the matrix theme in particular keeps its phosphor green for
the accent and states while the text sits at a desaturated green-white,
because a full screen of #00ff00 body text is the definition of "hard on
the eyes".

Every palette must clear ``tests/unit/tui/test_palette_contrast.py`` and be
inspected as rendered frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = []
