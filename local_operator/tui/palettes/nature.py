"""Nature-inspired themes: sage (Zelda beige/sage-green), forest, ocean…

The family's shared idea is a GROUND with an identity — warm beige, deep
sea, pine shadow — with states kept legible against it. Warm grounds
(sage, desert, autumn) have to re-solve ``danger`` and ``warning`` rather
than copy a cool theme's values: a red that clears 4:1 on a blue-black can
drop under the floor on beige.

Every palette must clear ``tests/unit/tui/test_palette_contrast.py`` and be
inspected as rendered frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = []
