"""Nature-inspired themes: sage (Zelda beige/sage-green), forest, ocean…

The family's shared idea is a GROUND with an identity — warm beige, deep
sea, pine shadow — with states kept legible against it. Warm grounds
(sage, desert, autumn, rosewood) have to re-solve ``danger`` and
``warning`` rather than copy a cool theme's values: a red that clears 4:1
on a blue-black can drop under the floor on beige, because the warm ground
already carries much of the red/yellow luminance a state hue relies on for
separation. Every warm theme here therefore lifts its reds toward salmon
and its ambers toward gold until they measure >= 4:1 on BOTH grounds,
while staying recognisably "red" and "amber" next to each other.

Design constraints the whole family obeys:

- The ground ramp (sunken < bg < surface < raised < overlay) is solved per
  theme in the brand dark ramp's proportions — each lift a just-visible
  ~1.1x step, never a slab — because no nature reference defines five
  elevation steps.
- ``fg`` is always a near-neutral tinted toward the ground's hue (warm
  bone on beige, sea-mist on blue): the vibe lives in accent/signal/label/
  string and the tints, never in body text.
- ``tint-select`` leans toward the theme's accent hue at roughly the
  ground's luminance; ``tint-attach`` stays a cool cast everywhere because
  ``signal`` is the family-wide file/reference hue and must read on it.

Every palette must clear ``tests/unit/tui/test_palette_contrast.py`` and be
inspected as rendered frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Sage — the requested Zelda theme: parchment-dark ground (a warm dark
    # beige-brown, deliberately NOT gray), sage-green accent, warm korok
    # tones in the tints. The selection tint is a mossy olive cast rather
    # than the brand's cool green so the highlight feels grown, not lit.
    ThemeSpec(
        name="sage",
        label="Sage",
        description="Parchment dark and sage green, a quiet Hyrule field at dusk",
        dark=True,
        tokens={
            "bg": "#211b10",
            "surface": "#2b2418",
            "raised": "#342c1f",
            "overlay": "#3d3426",
            "sunken": "#18130b",
            "fg": "#ece5d2",  # warm bone, near-neutral
            "muted": "#bcb39d",
            "dim": "#8b8471",
            "faint": "#544d3c",
            "edge": "#43392a",
            "edge-hi": "#554936",
            "accent": "#a3c47f",  # sage green — soft, gray-leaning, the one signature
            # A step greener and deeper than the accent (review round 1, D2:
            # the first solve sat 21 RGB units away, close enough that a ✓ and
            # the caret read as one hue). 7.4:1 on bg, 70 units off the accent.
            "success": "#7fb968",
            "string": "#94c479",
            # Warm-ground re-solve: a cool theme's #e06c75-class red measures
            # ~3.5:1 on this beige; lifted to salmon to clear 4:1 both grounds.
            "warning": "#e0ac4e",
            "danger": "#f08c78",
            "signal": "#7db8d8",
            "label": "#c39ede",
            "tint-danger": "#33201a",
            "tint-select": "#252b18",  # korok-moss cast
            "tint-select-hi": "#2c331e",
            "tint-attach": "#25343e",
            "tint-attach-hi": "#365064",
        },
    ),
    # Forest — deep pine shadow with a moss accent. The greens are split by
    # temperature so they never collapse: accent is yellow-moss, success is
    # cooler leaf-green, string sits between; signal is a stream-blue kept
    # clearly apart from all three.
    ThemeSpec(
        name="forest",
        label="Forest",
        description="Deep pine shadow, moss-green light through the canopy",
        dark=True,
        tokens={
            "bg": "#0f1a13",
            "surface": "#17231b",
            "raised": "#1f2c23",
            "overlay": "#27352b",
            "sunken": "#0a120d",
            "fg": "#dde8dd",
            "muted": "#a8bba9",
            "dim": "#788d7b",
            "faint": "#465547",
            "edge": "#2c3d31",
            "edge-hi": "#3a4e3f",
            "accent": "#8fbf68",  # moss
            "success": "#7cc487",
            "string": "#9dc47c",
            "warning": "#d8ae52",
            "danger": "#e58579",
            "signal": "#6fb3c9",
            "label": "#b195d6",
            "tint-danger": "#28211c",
            "tint-select": "#1a2a1a",
            "tint-select-hi": "#213321",
            "tint-attach": "#1c2e38",
            "tint-attach-hi": "#2c4a58",
        },
    ),
    # Ocean — deep-sea blue-green ground, pale foam accent. Accent is the
    # brightest, whitest green in the family (sea foam), while success stays
    # a kelp green so "live/focused" and "succeeded" read as two things.
    ThemeSpec(
        name="ocean",
        label="Ocean",
        description="Deep-sea blue-green with a pale sea-foam accent",
        dark=True,
        tokens={
            "bg": "#0c1a20",
            "surface": "#132630",
            "raised": "#1b2e39",
            "overlay": "#233742",
            "sunken": "#081218",
            "fg": "#dcebee",
            "muted": "#a4bec5",
            "dim": "#71919a",
            "faint": "#3e565f",
            "edge": "#254049",
            "edge-hi": "#31525d",
            "accent": "#84e0cf",  # foam
            "success": "#6cc99b",  # kelp
            "string": "#7fcbb0",
            "warning": "#d9b45c",
            "danger": "#ef8b85",
            "signal": "#72b6e4",
            "label": "#ab9ce0",
            "tint-danger": "#26212a",
            "tint-select": "#12302c",  # teal cast, not the brand's leaf green
            "tint-select-hi": "#173a35",
            "tint-attach": "#1a2c42",
            "tint-attach-hi": "#294663",
        },
    ),
    # Desert — warm dark sand with terracotta and cactus. The ground sits a
    # step LIGHTER and yellower than sage's parchment (bg #271e12 vs sage's
    # #211b10) so the two warm-beige themes stay distinguishable side by
    # side: sage is dusk parchment, desert is sunlit sand after dark.
    # Accent (terracotta) and warning (amber) share warmth, so warning is
    # pushed yellow and danger toward a clear coral-red to keep the three
    # warm states distinct at a glance. Warm-ground rule: danger/warning
    # solved here, not copied.
    ThemeSpec(
        name="desert",
        label="Desert",
        description="Warm dark sand, terracotta glow, a stripe of cactus green",
        dark=True,
        tokens={
            "bg": "#271e12",
            "surface": "#31271a",
            "raised": "#3a2f21",
            "overlay": "#433728",
            "sunken": "#1d160c",
            "fg": "#f0e6d5",
            "muted": "#c4b5a0",
            "dim": "#93866f",
            "faint": "#5b503e",
            "edge": "#4a3c2a",
            "edge-hi": "#5d4c36",
            "accent": "#e59d6a",  # terracotta
            "success": "#9ac275",  # cactus
            "string": "#adc275",
            "warning": "#e2b148",
            "danger": "#f58384",  # coral-red, lifted for the warm ground
            "signal": "#84bad8",
            "label": "#cb9edc",
            "tint-danger": "#3b2419",
            "tint-select": "#2f2e18",  # dry-grass cast
            "tint-select-hi": "#37371e",
            "tint-attach": "#2c3843",
            "tint-attach-hi": "#405466",
        },
    ),
    # Autumn — dark oak ground, maple red and amber leaves. Three warm
    # states again: accent is maple-orange, warning a yellower harvest
    # amber, danger a lifted maple red (a cool theme's crimson would sit
    # near 3.5:1 on oak). Success is a drying leaf-green, not spring green.
    ThemeSpec(
        name="autumn",
        label="Autumn",
        description="Dark oak under maple red and harvest amber",
        dark=True,
        tokens={
            "bg": "#1d1510",
            "surface": "#271e17",
            "raised": "#30261e",
            "overlay": "#392e25",
            "sunken": "#140e09",
            "fg": "#eddfd0",
            "muted": "#c0ac97",
            "dim": "#8e7d6a",
            "faint": "#55483a",
            "edge": "#42332a",
            "edge-hi": "#544236",
            "accent": "#e08d4f",  # maple orange
            "success": "#a2b96a",  # drying leaf
            "string": "#b3b96a",
            "warning": "#ddab35",
            "danger": "#f37f6f",  # maple red, warm-ground lifted
            "signal": "#7fb0d3",
            "label": "#c599d6",
            "tint-danger": "#331b14",
            "tint-select": "#292312",  # fallen-leaf cast
            "tint-select-hi": "#312b17",
            "tint-attach": "#233140",
            "tint-attach-hi": "#354c61",
        },
    ),
    # Lavender — dusk purple-gray ground, lavender accent. Label (violet
    # meta) lives one hue step pinker than the accent so the two purples
    # never merge; danger leans rose to stay off the label's lane too.
    ThemeSpec(
        name="lavender",
        label="Lavender",
        description="Purple-gray dusk with a soft lavender glow",
        dark=True,
        tokens={
            "bg": "#191623",
            "surface": "#211e2e",
            "raised": "#2a2638",
            "overlay": "#332e42",
            "sunken": "#110f19",
            "fg": "#e6e2f0",
            "muted": "#b3adc6",
            "dim": "#837c9a",
            "faint": "#4c4762",
            "edge": "#363050",
            "edge-hi": "#453e64",
            "accent": "#b9a3e8",  # lavender
            "success": "#7fc98f",
            "string": "#98c887",
            "warning": "#dcae54",
            "danger": "#ef8595",  # rose-red, apart from both purples
            "signal": "#7db2e2",
            "label": "#cf94d8",  # pinker violet than accent
            "tint-danger": "#2b1a26",
            "tint-select": "#1f2333",  # deeper dusk-blue cast
            "tint-select-hi": "#262b40",
            "tint-attach": "#1c2740",
            "tint-attach-hi": "#2c405f",
        },
    ),
    # Arctic — blue-white on slate with an aurora-green accent. The accent
    # is the one saturated thing in an otherwise icy ramp; the selection
    # tint carries a faint aurora cast so highlighting feels like the sky.
    ThemeSpec(
        name="arctic",
        label="Arctic",
        description="Blue-white on slate, lit by an aurora green",
        dark=True,
        # Re-grounded in review round 1 (D3): the first solve's near-black slate
        # sat 7 RGB units from ocean's ground, so two themes sold as different
        # vibes were pixel-duplicates in a 34-row picker. The slate now sits a
        # step lighter and bluer (bg 41 units from ocean's), which is also what
        # the description promises — blue-white ON slate, not on deep sea. The
        # aurora stays the accent alone (D2): success takes the kelp green so a
        # ✓ and the caret stop sharing one hue.
        tokens={
            "bg": "#1a2431",
            "surface": "#232e3d",
            "raised": "#2c3949",
            "overlay": "#354455",
            "sunken": "#121a25",
            "fg": "#e3ecf4",
            "muted": "#a9bcca",
            "dim": "#77909f",
            "faint": "#425666",
            "edge": "#31435a",
            "edge-hi": "#405671",
            "accent": "#68e0a3",  # aurora — the one saturated thing in the ice
            "success": "#6cc99b",
            "string": "#87ceab",
            "warning": "#dcb45e",
            "danger": "#ee8b8e",
            "signal": "#7fbde8",
            "label": "#b0a3e6",
            "tint-danger": "#32262e",
            "tint-select": "#1c332f",  # aurora cast
            "tint-select-hi": "#223d38",
            "tint-attach": "#22344c",
            "tint-attach-hi": "#334e6c",
        },
    ),
    # Rosewood — dark rosewood ground, dried-rose DANGER, brass warning (as
    # briefed: the rose hue is the failure color here, so accent takes a
    # fresh rose-pink clearly apart from danger's dusty salmon, and warning
    # is brass rather than gold to stay off the wood's own warmth. Another
    # warm ground: both re-solved to >= 4:1 on bg and surface.
    ThemeSpec(
        name="rosewood",
        label="Rosewood",
        description="Dark rosewood, dried-rose reds, a glint of brass",
        dark=True,
        tokens={
            "bg": "#201314",
            "surface": "#2a1c1d",
            "raised": "#332425",
            "overlay": "#3c2c2d",
            "sunken": "#170c0d",
            "fg": "#eee0dc",
            "muted": "#c2aba6",
            "dim": "#907c78",
            "faint": "#584745",
            "edge": "#453130",
            "edge-hi": "#57403e",
            "accent": "#e895b5",  # fresh rose blossom
            "success": "#94bd80",
            "string": "#a8bd80",
            "warning": "#d8ab58",  # brass
            "danger": "#ea8378",  # dried rose — dusty salmon-red
            "signal": "#82b1d4",
            "label": "#c99bd2",
            "tint-danger": "#371c1c",
            "tint-select": "#2b2318",  # warm heartwood cast
            "tint-select-hi": "#332b1e",
            "tint-attach": "#263140",
            "tint-attach-hi": "#394c61",
        },
    ),
]
