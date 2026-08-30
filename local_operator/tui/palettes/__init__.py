"""Curated color themes for the Local Operator TUI.

Each submodule holds a family of :class:`~local_operator.tui.theme.ThemeSpec`
palettes (classic editor schemes, neon/retro, nature-inspired, light ramps).
They are DATA, deliberately kept out of ``theme.py``: the theme module owns
the semantic vocabulary and the registry mechanics, and this package owns the
thirty-odd ramps that speak it. ``theme._registry`` folds these in lazily on
first lookup, so importing :mod:`local_operator.tui.theme` never pays for the
palette tables until something actually lists or switches themes.

Every palette must be TOTAL over ``theme.SEMANTIC_TOKENS`` — registration
rejects a missing or invented token — and must clear the contrast floors
pinned in ``tests/unit/tui/test_palette_contrast.py``. The floors are derived
from the brand ramps' own measurements (``fg`` ≥ 7:1 on the ground, ``dim``
≥ 3:1, state hues ≥ 4:1…), so "readable" is a checked property of every
theme, not a hope. Authors: run that test, then render the screenshot grid
(``scripts/theme_preview.py``) and LOOK at the frames before submitting.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec


def all_palettes() -> list[ThemeSpec]:
    """Every curated palette, in presentation order (after the brand ramps).

    Order is the picker's row order: classics first because they are the
    names users arrive knowing, then the neon/retro set, then nature, then
    the light ramps at the end where a dark-terminal user scrolls past them.
    The Radient brand ramp goes immediately after the neon FAMILY: it is
    the same blue-black architecture as neon's tron wearing the Radient
    kit, so a user comparing the two finds them one family scroll apart.

    Rosé Pine sits with the classics for the same reason they lead: it is a
    name users arrive knowing. Its three variants stay together — including
    the light one, which breaks the "lights at the end" rule on purpose,
    because a user picking Dawn is choosing between Rosé Pine variants, not
    between light ramps.
    """
    from local_operator.tui.palettes import (  # local: cycle-free at call time
        classics,
        lights,
        nature,
        neon,
        radient,
        rose_pine,
    )

    return [
        *classics.PALETTES,
        *rose_pine.PALETTES,
        *neon.PALETTES,
        *radient.PALETTES,
        *nature.PALETTES,
        *lights.PALETTES,
    ]
