"""Light themes beyond the brand paper ramp.

Light ramps invert every decision the dark ones make: elevation lifts
TOWARD white instead of away from black, state hues need to be darker
rather than brighter to hold contrast, and the tint grounds (``tint-*``)
sit just below the paper rather than just above the night. The brand
``light`` ramp in ``theme.py`` is the reference solve — mirror its
relationships, not its hexes.

Family-wide solves this file repeats per theme:

- The elevation ladder darkens upward (bg > surface > raised > overlay in
  luminance) in the brand light ramp's proportions — each step a
  just-visible ~1.06–1.14:1 lift, never a slab — and ``sunken`` stays at
  or just below the paper (light polarity has no deep well to sink into;
  the brand ramp's own sunken is only 1.20:1 off the paper).
- State hues must be DARK: a hue that clears 4:1 on a near-black ground
  arrives at ~2–3:1 on paper, so every state here is re-solved downward
  until it measures >= 4:1 on BOTH grounds while staying recognisably
  itself (red, amber, green, blue, violet stay five distinct hues).
- ``fg`` is a near-neutral ink tinted toward the paper's temperature
  (warm ink on cream, cool ink on linen); saturation lives in the state
  hues and the tints, never in body text.
- Tints sit just below the paper's luminance (< 2.2:1 vs bg) so a state
  ground reads as a wash on the page, not a highlighter stripe.

Every palette must clear ``tests/unit/tui/test_palette_contrast.py``
(``dark=False`` flips the polarity checks) and be inspected as rendered
frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Solarized Light — Ethan Schoonover's published values wherever they
    # clear the floors. The grounds ARE canonical: bg=base3, raised=base2,
    # fg=base02, muted=base01, dim=base00, faint=base1; surface/overlay are
    # interpolated between base3 and base2 (Solarized defines only two
    # ground steps, this UI needs five). The accent CONTENT hues are where
    # Solarized famously trades contrast for calm, so several miss the 4:1
    # state floor on base3 and are darkened minimally along their own hue:
    #   - blue    #268bd2 measures 3.41:1 -> #1c6ea4 (same hue, darker)
    #   - green   #859900 measures 2.97:1 -> #657600
    #   - yellow  #b58900 measures 2.98:1 -> #8a6800
    #   - cyan    #2aa198 measures 2.93:1 -> #0e8074
    #   - violet  #6c71c4 measures 3.57:1 on surface -> #5f64bb
    # red #dc322f is canonical (4.29:1/4.06:1 — the only content hue
    # Schoonover pushed dark enough), which is exactly why danger keeps it.
    ThemeSpec(
        name="solarized-light",
        label="Solarized Light",
        description="Schoonover's sunlit base3 paper, the classic 16-color science",
        dark=False,
        tokens={
            "bg": "#fdf6e3",  # base3, canonical
            "surface": "#f6f0dd",  # base3->base2 midpoint (Solarized has no step here)
            "raised": "#eee8d5",  # base2, canonical
            "overlay": "#e2dbc1",  # one just-visible step past base2
            "sunken": "#f5eed9",  # a hair below base3 — light polarity has no well
            "fg": "#073642",  # base02, canonical (12.05:1)
            "muted": "#586e75",  # base01, canonical
            "dim": "#657b83",  # base00, canonical (4.13:1 — Solarized's own body color)
            "faint": "#93a1a1",  # base1, canonical
            "edge": "#e6dfc8",
            "edge-hi": "#d5cdae",
            "accent": "#1c6ea4",  # blue, darkened from canonical #268bd2 (3.41:1)
            "success": "#657600",  # green, darkened from canonical #859900 (2.97:1)
            "string": "#657600",
            "warning": "#8a6800",  # yellow, darkened from canonical #b58900 (2.98:1)
            "danger": "#dc322f",  # red, CANONICAL — it already clears 4:1
            "signal": "#0e8074",  # cyan, darkened from canonical #2aa198 (2.93:1)
            "label": "#5f64bb",  # violet, darkened from canonical #6c71c4 (3.57:1 surf)
            # Lighter than the first solve (#f6e0d2, where canonical red read
            # 3.64:1 — under the 4.0 state floor on the one band that always
            # carries it; review round 1, D1). Pulled toward base3 so the wash
            # stays warm while the red clears 4.1:1.
            "tint-danger": "#fcefe6",
            "tint-select": "#eaefd2",  # green-leaning cast (selection = the green family)
            "tint-select-hi": "#e0e7c2",
            "tint-attach": "#e4ecec",  # cool cyan cast — signal is the file hue
            "tint-attach-hi": "#cfe0e6",
        },
    ),
    # GitHub Light — Primer's published light scale, used verbatim: every
    # hex below is a Primer token (canvas, canvas-subtle, fg-default,
    # fg-muted, border-default, the fg-role state colors, and the
    # attention/danger/accent subtle backgrounds as tints). GitHub solved
    # 4.5:1 on white for all of these, so nothing needed darkening — the
    # only editorial choice is signal=#0550ae (Primer blue-7, the syntax
    # constant blue) so links/files stay GitHub-blue without collapsing
    # into the #0969da accent that carries focus.
    ThemeSpec(
        name="github-light",
        label="GitHub Light",
        description="Clean Primer white with GitHub's own state colors",
        dark=False,
        tokens={
            "bg": "#ffffff",  # canvas.default
            "surface": "#f6f8fa",  # canvas.subtle
            "raised": "#eaeef2",  # neutral.subtle step
            "overlay": "#dde2e8",
            "sunken": "#f0f3f7",
            "fg": "#1f2328",  # fg.default
            "muted": "#59626c",  # fg.muted, nudged for the 4.5 floor on surface
            "dim": "#6e7781",  # fg.subtle
            "faint": "#a8b1bb",
            "edge": "#d8dee4",  # border.default territory
            "edge-hi": "#c2cad3",
            "accent": "#0969da",  # accent.fg — THE GitHub blue
            "success": "#1a7f37",  # success.fg (open-PR green)
            "string": "#0a3069",  # Primer syntax string dark-blue
            "warning": "#9a6700",  # attention.fg
            "danger": "#cf222e",  # danger.fg
            "signal": "#0550ae",  # blue-7: links/files stay blue, distinct from accent
            "label": "#8250df",  # done.fg (the merged-PR purple)
            "tint-danger": "#ffebe9",  # danger.subtle
            "tint-select": "#ddf4ff",  # accent.subtle — GitHub selects in blue
            "tint-select-hi": "#c6e6f8",
            "tint-attach": "#eef2f8",
            "tint-attach-hi": "#d8e4f5",
        },
    ),
    # Paper — warm cream, book-page feel. Deliberately WARMER and softer
    # than the built-in brand "light" (#f7f4ee): the ground is a true cream
    # (#f9f3e6, yellower and a touch brighter), the ink is a warm
    # brown-black rather than the brand's cool-neutral, and the states are
    # bookish — sepia accent, olive string, brick danger — like a printed
    # page with editorial ink annotations rather than a UI.
    ThemeSpec(
        name="paper",
        label="Paper",
        description="Warm cream and brown ink, an open book in good light",
        dark=False,
        tokens={
            "bg": "#f9f3e6",
            "surface": "#f1ead9",
            "raised": "#e7dfc9",
            "overlay": "#dbd1b8",
            "sunken": "#efe8d6",
            "fg": "#332b20",  # warm brown-black ink, 12.6:1
            "muted": "#5f574a",
            "dim": "#7f7666",
            "faint": "#b2a78f",
            "edge": "#e2d9c2",
            "edge-hi": "#cdc1a2",
            # Sepia/rust accent: the one signature is the color of an
            # editor's pen on a manuscript, not a screen blue.
            "accent": "#8a4b2a",
            "success": "#4a7a2e",  # leaf green, dark enough for cream
            "string": "#5d7030",  # olive — reads "quoted text" without neon
            "warning": "#8f6410",
            "danger": "#b03d2e",  # brick red, clearly apart from the sepia accent
            "signal": "#2e6d9e",  # the page's one cool hue: links/files
            "label": "#7d5799",
            "tint-danger": "#f3ddcd",
            "tint-select": "#ece9ca",  # dry-grass cast toward the olive family
            "tint-select-hi": "#e2dfba",
            "tint-attach": "#e7ebe2",  # cool-leaning wash for the signal hue
            "tint-attach-hi": "#d3decf",
        },
    ),
    # Linen — cool off-white with muted pastel-derived states. Where paper
    # is warm and bookish, linen is gray-green fabric: every state hue is
    # deliberately DESATURATED (dusty teal, moss, clay, slate, dusty
    # violet) so the whole surface reads soft even when it is reporting an
    # error. The floor still binds — "muted" here means low chroma, not
    # low contrast — so each hue is darkened until it clears 4:1.
    ThemeSpec(
        name="linen",
        label="Linen",
        description="Cool off-white weave, states in soft dusty pastels",
        dark=False,
        tokens={
            "bg": "#f4f4f1",
            "surface": "#eaebe7",
            "raised": "#dfe1dc",
            "overlay": "#d2d5cf",
            "sunken": "#eceded",
            "fg": "#2b2e2c",  # cool near-neutral ink
            "muted": "#575d59",
            "dim": "#767d78",
            "faint": "#a9b0aa",
            "edge": "#dcdeda",
            "edge-hi": "#c5c9c3",
            "accent": "#377568",  # dusty teal — quiet, but unmistakably the signature
            "success": "#4d7a55",  # moss
            "string": "#5c7561",  # gray-green, one step off success
            "warning": "#8a6a2f",  # clay amber
            "danger": "#a2544e",  # dusty brick — soft, still clearly "red"
            "signal": "#54708f",  # slate blue, kept apart from the teal accent
            "label": "#7a648f",  # dusty violet
            "tint-danger": "#f0e1de",
            "tint-select": "#e4e9e2",
            "tint-select-hi": "#d9e1d8",
            "tint-attach": "#e5e9ec",
            "tint-attach-hi": "#d3dce4",
        },
    ),
    # High Contrast Light — the accessibility ramp. Near-white ground,
    # true ink-black body text (19.3:1), and states saturated AND dark so
    # every one of them clears 6:1 on both grounds (well past the 4:1
    # floor — this theme's contract is headroom, not minimum compliance).
    # The grounds are pure neutrals: on a hue-less page the state colors
    # carry ALL the meaning, so the five hues are pushed maximally apart
    # (pure blue accent, pure green, pure red, brown-amber, deep violet).
    # Tints are the loudest in the file (up to ~1.3:1) because faint
    # washes are exactly what low-vision users lose first.
    ThemeSpec(
        name="high-contrast-light",
        label="High Contrast Light",
        description="Ink on near-white, every state 6:1+, built for legibility",
        dark=False,
        tokens={
            "bg": "#fcfcfc",
            "surface": "#f1f1f1",
            "raised": "#e5e5e5",
            "overlay": "#d8d8d8",
            "sunken": "#f4f4f4",
            "fg": "#0a0a0a",  # ink black, 19.3:1
            "muted": "#3d3d3d",  # 10.6:1 — "secondary" still reads like text
            "dim": "#595959",  # 6.8:1, double the dim floor
            "faint": "#8f8f8f",
            "edge": "#c8c8c8",  # borders visibly darker than any elevation step
            "edge-hi": "#a8a8a8",
            "accent": "#0b47c2",  # saturated royal blue, 7.6:1
            "success": "#0a6b26",  # deep pure green, 6.5:1
            "string": "#123f83",  # navy — code strings stay prose-dark
            "warning": "#7a4d00",  # brown-amber: yellow can't reach 6:1, brown can
            "danger": "#c40016",  # saturated pure red, 6.1:1
            "signal": "#005a9e",  # distinct darker azure for links/files
            "label": "#6a1f9e",  # deep violet, 8.9:1
            "tint-danger": "#fcd9d9",
            "tint-select": "#d6e6fb",  # blue selection — matches the accent's meaning
            "tint-select-hi": "#bcd7f7",
            "tint-attach": "#e3edf9",
            "tint-attach-hi": "#c9def5",
        },
    ),
    # Mint Light — white with a fresh green signature. The trap in an
    # all-green identity is accent/success/string collapsing into one hue,
    # so the greens are split by darkness and temperature: accent is the
    # brightest, freshest mint (#0c824e), success a clearly darker pine
    # (#1d6a3e, 1.36:1 apart from accent), string a warmer grass green —
    # "live", "succeeded", and "quoted" stay three readable things.
    # Grounds carry a barely-there green cast so the page itself feels
    # mint without tinting the text.
    ThemeSpec(
        name="mint-light",
        label="Mint Light",
        description="Crisp white with a cool fresh-mint green accent",
        dark=False,
        tokens={
            "bg": "#fbfdfb",
            "surface": "#eef6ef",
            "raised": "#e0eee2",
            "overlay": "#d0e4d4",
            "sunken": "#f1f7f2",
            "fg": "#1c2b21",  # near-neutral ink with a green whisper
            "muted": "#4c5f52",
            "dim": "#6b7f71",
            "faint": "#a3b5a8",
            "edge": "#d9e7dc",
            "edge-hi": "#bcd4c2",
            "accent": "#0c824e",  # fresh mint — the signature
            "success": "#1d6a3e",  # pine, darker so ✓ never impersonates focus
            "string": "#2e6e2b",  # grass, warmer than both
            "warning": "#8c6410",
            "danger": "#c2413b",
            "signal": "#2470a8",  # cool blue counterweight for links/files
            "label": "#77579e",
            "tint-danger": "#f9e4e0",
            "tint-select": "#dcf2e2",  # mint wash — selection wears the accent hue
            "tint-select-hi": "#c8e9d2",
            "tint-attach": "#e6f0f7",
            "tint-attach-hi": "#d0e2f2",
        },
    ),
]
