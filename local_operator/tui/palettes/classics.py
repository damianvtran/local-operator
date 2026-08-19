"""Classic editor color schemes, mapped onto the Local Operator token set.

These are the names users arrive already knowing — Monokai, Dracula,
Catppuccin, Tokyo Night, Gruvbox, Nord, Solarized, One Dark — so fidelity to
the published palettes matters more here than in the invented families: the
hues people recognise (Dracula's purple, Monokai's pink) must land on the
tokens where they are most visible (``danger``, ``label``, ``accent``), while
the GROUND ramp (bg → surface → raised → overlay, with ``sunken`` below) is
solved per theme because none of the source palettes define five elevation
steps. Elevation steps follow the brand ramp's own proportions: each step is
a just-visible lift (~1.1–1.3:1 against the last), never a slab.

Every palette must clear the contrast floors in
``tests/unit/tui/test_palette_contrast.py``; run that file after any tweak,
then render ``scripts/theme_preview.py`` and look at the frames.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    ThemeSpec(
        name="monokai",
        label="Monokai",
        description="The classic warm-charcoal editor scheme",
        dark=True,
        tokens={
            "bg": "#272822",
            "surface": "#30312a",
            "raised": "#383931",
            "overlay": "#40413a",
            "sunken": "#1d1e19",
            "fg": "#f8f8f2",
            "muted": "#c8c8c0",
            "dim": "#90918b",
            "faint": "#5a5b55",
            "edge": "#48493f",
            "edge-hi": "#5c5d50",
            "accent": "#a6e22e",
            "success": "#a6e22e",
            "warning": "#fd971f",
            # Monokai Pro's red rather than classic #f92672: the classic pink
            # measures 3.47:1 on this surface — under the 4:1 state floor —
            # and the Pro revision solved exactly this legibility problem
            # while keeping the hue recognisably "Monokai pink".
            "danger": "#ff6188",
            "signal": "#66d9ef",
            "label": "#ae81ff",
            "string": "#a6e22e",
            "tint-danger": "#3a2429",
            "tint-select": "#2c3624",
            "tint-select-hi": "#333f2a",
            "tint-attach": "#263644",
            "tint-attach-hi": "#33506b",
        },
    ),
]
