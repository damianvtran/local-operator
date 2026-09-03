"""Rosé Pine — all three published variants (main, moon, dawn).

Upstream: github.com/rose-pine/palette, the canonical hex list in
``dist/css/rose-pine.css``. Rosé Pine ships fifteen roles per variant — nine
neutral (``base surface overlay muted subtle text highlight-low/med/high``)
and six accent (``love gold rose pine foam iris``) — against this UI's
twenty-three semantic tokens, so the port is a MAPPING, and the mapping is
the interesting part of this file.

The neutral roles land almost verbatim. Rosé Pine's ``base`` is ``bg``,
``surface`` is ``surface``, ``overlay`` is ``raised``; the fourth elevation
step this UI needs has no upstream value, so ``overlay`` is interpolated one
just-visible step past the canonical ``overlay``, in the ladder's own
proportions. ``highlight-low`` is the hairline ``edge`` and ``highlight-med``
the active ``edge-hi`` — which is exactly what upstream says they are for
(cursor line, borders) — and the text ladder is ``text`` → ``subtle`` →
``muted``, with ``dim`` interpolated between the last two because this UI
has one more text rung than Rosé Pine does.

The accent mapping is the one editorial decision, and it follows upstream's
own usage table rather than hue:

- ``love`` is the documented terminal red and error colour → ``danger``.
- ``gold`` is the documented strings-and-warnings colour → ``warning`` and
  ``string``, both at once, which is precisely what upstream assigns it.
- ``foam`` is the documented "object keys, info, git add" hue → ``signal``,
  this UI's link/file/reference role.
- ``iris`` is the documented links-and-hints magenta → ``label``.
- ``pine`` is the documented terminal green (functions, git rename) →
  ``success``. It is a BLUE-green, not a leaf green, which is Rosé Pine's
  signature inversion and is kept.
- ``rose`` is left for ``accent`` — the theme's own name, and the one hue
  upstream spends on "git change, git dirty": the colour of something
  happening. It carries the caret, the live indicator and focus here.

Contrast is where the variants diverge, and each deviation below carries the
canonical value and its measured ratio. Main and moon needed two lifts each;
dawn is a near-total re-solve, for the reason ``lights.py`` documents — a hue
tuned to glow on a near-black ground arrives at 2–3:1 on paper.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Rosé Pine (main) — the original night. Every neutral is canonical
    # except the two this UI adds: `overlay` (a fourth elevation step past
    # canonical overlay #26233a, in the ladder's own ~1.1x proportion) and
    # `sunken` (one step below base, where the status band sits).
    #
    # Two accents needed work against the 4:1 state floor:
    #   - pine #31748f measures 3.38:1 on base and 3.16:1 on surface — the
    #     ONLY canonical accent under the floor on this ground. It lifts to
    #     #3a8aaa: same blue-green, 4.54:1 / 4.24:1.
    #   - muted #6e6a86 is canonical `faint` here (3.42:1) rather than a
    #     text rung, so `dim` is the subtle/muted midpoint #7f7b98 (4.36:1)
    #     — the extra rung this UI has and Rosé Pine does not.
    ThemeSpec(
        name="rose-pine",
        label="Rosé Pine",
        description="All natural pine, faux fur and a bit of soho vibes",
        dark=True,
        tokens={
            "bg": "#191724",  # canonical base
            "surface": "#1f1d2e",  # canonical surface
            "raised": "#26233a",  # canonical overlay
            "overlay": "#322f45",  # one step past canonical overlay (no upstream value)
            "sunken": "#13111c",  # one step below base (no upstream value)
            "fg": "#e0def4",  # canonical text
            "muted": "#908caa",  # canonical subtle
            "dim": "#7f7b98",  # subtle/muted midpoint — this UI's extra rung
            "faint": "#6e6a86",  # canonical muted (3.42:1, the inert rung)
            "edge": "#21202e",  # canonical highlight-low
            "edge-hi": "#403d52",  # canonical highlight-med
            "accent": "#ebbcba",  # rose — the theme's name and its "something changed" hue
            "success": "#3a8aaa",  # pine #31748f: 3.38:1 on base (< 4)
            "warning": "#f6c177",  # gold — upstream's warnings colour
            "danger": "#eb6f92",  # love — upstream's error/terminal-red
            "string": "#f6c177",  # gold — upstream assigns strings to gold too
            "signal": "#9ccfd8",  # foam — upstream's info/object-key hue
            "label": "#c4a7e7",  # iris — upstream's links/hints magenta
            "tint-danger": "#2e1c28",  # love's cast at base luminance (love reads 5.50:1)
            "tint-select": "#2a2033",  # rose cast — selection wears the accent
            "tint-select-hi": "#342839",
            "tint-attach": "#1e2b39",  # foam-leaning, so `signal` reads on the chip
            "tint-attach-hi": "#2d4457",
        },
    ),
    # Rosé Pine Moon — the same palette one shade warmer and lighter, and
    # the variant most people mean by "Rosé Pine". Grounds are canonical
    # (base #232136, surface #2a273f, overlay #393552 as edge-hi's partner);
    # `raised` is interpolated between surface and canonical overlay because
    # moon's own overlay is spent on the fourth step.
    #
    # The lighter ground costs contrast on exactly three tokens:
    #   - subtle #908caa drops to 4.46:1 on the surface step (< 4.5), so
    #     `muted` is #938fad — the same violet-gray, one step up.
    #   - pine #3e8fb0 measures 3.94:1 on surface (< 4) and lifts to #4195b7.
    #   - rose #ea9a97 is only 43 RGB units from love #eb6f92 on this
    #     variant (main's are 87 apart), close enough that the caret and an
    #     error message read as one hue. It desaturates to #eaaca9 — still
    #     moon's rose, 65 units off love.
    ThemeSpec(
        name="rose-pine-moon",
        label="Rosé Pine Moon",
        description="Rosé Pine's warmer moonlit ground, the soho night one shade up",
        dark=True,
        tokens={
            "bg": "#232136",  # canonical base
            "surface": "#2a273f",  # canonical surface
            "raised": "#312e4a",  # surface->overlay midpoint (no upstream value)
            "overlay": "#393552",  # canonical overlay
            "sunken": "#1c1a2c",  # one step below base (no upstream value)
            "fg": "#e0def4",  # canonical text
            "muted": "#938fad",  # canonical subtle #908caa: 4.46:1 on surface (< 4.5)
            "dim": "#7f7b98",  # subtle/muted midpoint — this UI's extra rung
            "faint": "#6e6a86",  # canonical muted
            "edge": "#2a283e",  # canonical highlight-low
            "edge-hi": "#44415a",  # canonical highlight-med
            "accent": "#eaaca9",  # rose #ea9a97: 43 RGB units from love — too close
            "success": "#4195b7",  # pine #3e8fb0: 3.94:1 on surface (< 4)
            "warning": "#f6c177",  # gold
            "danger": "#eb6f92",  # love
            "string": "#f6c177",  # gold
            "signal": "#9ccfd8",  # foam
            "label": "#c4a7e7",  # iris
            "tint-danger": "#392334",
            "tint-select": "#332a44",
            "tint-select-hi": "#3d3350",
            "tint-attach": "#25334a",
            "tint-attach-hi": "#374c6b",
        },
    ),
    # Rosé Pine Dawn — the light variant, and a near-total re-solve of the
    # accent row for the reason `lights.py` states: a hue that glows on a
    # near-black ground lands at 2–3:1 on paper. Every canonical dawn accent
    # except pine misses the 4:1 state floor (gold 2.05:1, rose 2.60:1,
    # foam 3.14:1, iris 3.47:1, love 3.84:1), so each is darkened along its
    # own hue by the SMALLEST step that clears 4:1 on both grounds — the
    # deviations are 0–25 ΔE, listed per token below: pine needs none, gold
    # (the furthest) needs 25.2.
    #
    # The ground ladder also has to invert. Canonical dawn `surface`
    # #fffaf3 is LIGHTER than `base` #faf4ed, so it cannot be this UI's
    # first elevation step (light polarity darkens upward); it becomes
    # `sunken`, the band below the paper, which is what a lighter-than-page
    # value is actually good for. The ladder then steps down through
    # highlight-low #f4ede8 and canonical overlay #f2e9e1.
    #
    # One pairing stays deliberately close: love and rose both collapse
    # toward the same dusty red under the floor (best achievable separation
    # is ~12 ΔE at full fidelity). `accent` takes rose at 16.8 ΔE from
    # canonical and `danger` takes love at 5.0 ΔE, buying 29 RGB units of
    # separation — tighter than the dark variants, in line with the closest
    # pairings the other light palettes here already ship.
    ThemeSpec(
        name="rose-pine-dawn",
        label="Rosé Pine Dawn",
        description="Rosé Pine at sunrise, warm paper under a soho morning",
        dark=False,
        tokens={
            "bg": "#faf4ed",  # canonical base
            "surface": "#f4ede8",  # canonical highlight-low
            "raised": "#f2e9e1",  # canonical overlay
            "overlay": "#e8ddd4",  # one step past overlay (no upstream value)
            "sunken": "#fffaf3",  # canonical surface — lighter than base, so it sinks
            "fg": "#4d496b",  # canonical text #575279: 6.66:1 on base (< 7)
            "muted": "#6a6681",  # canonical subtle #797593: 4.02:1 (< 4.5)
            "dim": "#7e7a89",  # canonical muted #9893a5: 2.73:1 (< 3.4)
            "faint": "#a5a0af",  # near canonical muted — the inert rung
            "edge": "#eee5dc",  # between highlight-low and highlight-med
            "edge-hi": "#dfdad9",  # canonical highlight-med
            "accent": "#a8594f",  # rose #d7827e: 2.60:1 (< 4) — 16.8 ΔE, held off love
            "success": "#286983",  # pine — CANONICAL (5.59:1/5.27:1), the only one that clears
            "warning": "#9e6200",  # gold #ea9d34: 2.05:1 (< 4) — the largest deviation
            "danger": "#a8566c",  # love #b4637a: 3.84:1 (< 4) — 5.0 ΔE
            "string": "#9e6200",  # gold, same solve as warning
            "signal": "#41787a",  # foam #56949f: 3.14:1 (< 4), held off pine
            "label": "#7d6694",  # iris #907aa9: 3.47:1 (< 4) — 7.8 ΔE
            "tint-danger": "#f6e2e3",
            # Selection/focus, AUTHORED rather than left on the rose cast.
            # `#f0e8e3` measured ΔE 1.71 from `surface` — a focused tool row
            # was indistinguishable from an unfocused one, and dawn was the
            # only theme with the defect (dark 8.54, github-light 8.74,
            # solarized-light 6.65). The paper has no well to sink into, so
            # the step is bought in CHROMA at near-constant luminance.
            #
            # IRIS: dawn's canonical meta accent, already what `label`
            # spends, which makes it the right hue for a selection that is
            # chrome rather than content. Measured: ΔE 10.05 from `surface`,
            # at 1.157:1 against `bg` — well inside the 2.2 tint ceiling.
            "tint-select": "#e9e2f2",
            "tint-select-hi": "#ded2ee",
            "tint-attach": "#e8eef0",  # foam-leaning wash
            "tint-attach-hi": "#d2e0e2",
            # The user's own prompt ground. The per-theme derivation solves
            # for maximum separation and lands on green-teal (159°) here,
            # which is correct by the metric and wrong for the eye: it fights
            # the warm paper. Blue is the one hue family dawn's palette does
            # not already spend — love/gold/rose are warm, pine/foam sit at
            # 180-200°, iris is violet — so a cool cast separates from the
            # ground instead of competing with it. Measured: ΔE 10.0 from
            # `bg` and 9.8 from its worst rival ground, both above the §9
            # floors, at 245° — the best of the blues tested (230/245/260).
            # Tool categories. The derivation leaves mutate/exec on the
            # neutral ramp, which is the safe default for 54 themes but keeps
            # dawn's ledger a single grey. Dawn has six solved accents and was
            # spending two, so these are authored from the canonical set it
            # already carries: foam for reading, gold for changing a file,
            # iris for running something, rose for coordination.
            #
            # Every value is a token dawn has already solved for WCAG on this
            # paper (all four measure >= 4.31:1 on `surface`), the four
            # separate from each other by >= 22.1 ΔE00, and each stays >= 11.6
            # ΔE00 from BOTH outcome hues — a category must never be mistaken
            # for a verdict. `success` is deliberately not reused here: it is
            # the ✓ one column to the right on the same row.
            "tool-read": "#41787a",  # foam — same hue as file paths, deliberately
            "tool-mutate": "#9e6200",  # gold
            "tool-exec": "#7d6694",  # iris
            "tool-meta": "#a8594f",  # rose
        },
    ),
]
