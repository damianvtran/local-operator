"""Neon / retro-futurist themes: synthwave, matrix, tron, and kin.

The design constraint that separates this family from the classics: the
VIBE is carried by the accent, signal and label hues plus the tinted
grounds — never by pushing the body ink to a saturated hue. ``fg`` stays a
near-neutral (a cold white, a pale green-white) so hours of prose reading
do not fatigue; the matrix theme in particular keeps its phosphor green for
the accent and states while the text sits at a desaturated green-white,
because a full screen of #00ff00 body text is the definition of "hard on
the eyes".

Grounds do the other half of the work. Each theme's bg carries a CAST
(purple-navy, green-black, blue-black, violet-black...) so the vibe is
present even in an empty pane, but the five elevation steps mirror the
brand dark ramp's proportions — each lift ~1.05–1.13:1 against the last,
a just-visible rise, never a slab. None of the source aesthetics define
five ground steps, so every ramp here is solved per theme by scaling the
signature ground's luminance along the brand ramp's curve.

Every palette must clear ``tests/unit/tui/test_palette_contrast.py`` and be
inspected as rendered frames via ``scripts/theme_preview.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Synthwave '84 (Robb Owen's VS Code theme). Unusually for a neon
    # palette, every canonical hue clears the 4:1 state floor on the
    # canonical ground (#262335) — even the hot red #fe4450 (4.84:1) — so
    # this theme is pure fidelity: pink, cyan, yellow, red, green and the
    # comment-violet are all published values. `dim` is the canonical
    # comment color #848bbd (4.66:1 on bg), which is exactly the job
    # comments do in the source theme.
    ThemeSpec(
        name="synthwave",
        label="Synthwave '84",
        description="Hot pink and cyan on a deep purple-navy night drive",
        dark=True,
        tokens={
            "bg": "#262335",
            "surface": "#2d2a41",
            "raised": "#35314c",
            "overlay": "#3d3958",
            "sunken": "#1e1b2a",
            "fg": "#f2eff8",
            "muted": "#bcb3d4",
            "dim": "#848bbd",
            "faint": "#575071",
            "edge": "#443f5e",
            "edge-hi": "#544e72",
            "accent": "#ff7edb",
            "success": "#72f1b8",
            "warning": "#fede5d",
            "danger": "#fe4450",
            "signal": "#36f9f6",
            "label": "#b893ce",
            "string": "#72f1b8",
            "tint-danger": "#3f2138",
            "tint-select": "#352a55",
            "tint-select-hi": "#3e3264",
            "tint-attach": "#22334f",
            "tint-attach-hi": "#2f476b",
        },
    ),
    # Digital rain. The ground is a near-black with a green cast (never
    # neutral black — the cast IS the CRT), and the whole ramp stays in
    # hue. Raw phosphor #00ff00 clears every floor but is deliberately
    # NOT used anywhere: accent is a tamed phosphor #22e06a and body text
    # a desaturated green-white, per the family constraint above. The
    # off-hue states (amber warning, red danger, cyan signal, violet
    # label) are desaturation-matched so they read as glitches in the
    # rain rather than visitors from another theme.
    ThemeSpec(
        name="matrix",
        label="Matrix",
        description="Phosphor green rain on a near-black terminal screen",
        dark=True,
        tokens={
            "bg": "#050d07",
            "surface": "#0b160e",
            "raised": "#122016",
            "overlay": "#1a2b1e",
            "sunken": "#020703",
            "fg": "#d4e6d6",
            "muted": "#99bd9f",
            "dim": "#639c70",
            "faint": "#2f5238",
            "edge": "#1d3524",
            "edge-hi": "#2a4a33",
            "accent": "#22e06a",
            "success": "#3ecf74",
            "warning": "#d8c24a",
            "danger": "#ff6b5e",
            "signal": "#3fd0c9",
            "label": "#a08fe0",
            "string": "#57d98a",
            "tint-danger": "#231311",
            "tint-select": "#0e2417",
            "tint-select-hi": "#132e1e",
            "tint-attach": "#0d2426",
            "tint-attach-hi": "#133638",
        },
    ),
    # The Grid. Blue-black ground, electric cyan for the light-cycle
    # lines (accent + edges lean cyan-blue), and danger is ORANGE, not
    # red — the Rinzler suit is the franchise's own error color. Warning
    # therefore shifts to a yellow-gold so the two warm states cannot
    # collapse; success is a sea-glass teal-green that stays cool enough
    # to belong on the Grid.
    ThemeSpec(
        name="tron",
        label="Tron",
        description="Electric cyan circuitry on the Grid's blue-black",
        dark=True,
        tokens={
            "bg": "#060b12",
            "surface": "#0c1420",
            "raised": "#131e2d",
            "overlay": "#1a283a",
            "sunken": "#03060b",
            "fg": "#d8e6f2",
            "muted": "#9fb8cc",
            "dim": "#6a89a3",
            "faint": "#33485c",
            "edge": "#1e3245",
            "edge-hi": "#2a4660",
            "accent": "#00d8ff",
            "success": "#3fe0b0",
            "warning": "#e8c14a",
            "danger": "#ff7a2f",
            "signal": "#7ab8ff",
            "label": "#a496ff",
            "string": "#3fe0b0",
            "tint-danger": "#2a1a0d",
            "tint-select": "#0d2430",
            "tint-select-hi": "#122f3e",
            "tint-attach": "#132339",
            "tint-attach-hi": "#1c3453",
        },
    ),
    # Night City. The 2077 marketing trio lands intact: construction
    # yellow #fcee0a as THE accent, glare cyan #00f0f0-family as signal,
    # and the logo red #ff003c as danger (4.95:1 on this violet-black —
    # legal, and its scarlet punch is the point). Selection tints are
    # YELLOW-cast rather than the usual green/blue, because in this theme
    # yellow is the brand's "pay attention" color.
    ThemeSpec(
        name="cyberpunk",
        label="Cyberpunk",
        description="Construction yellow and glare cyan over violet-black Night City",
        dark=True,
        tokens={
            "bg": "#0e0a16",
            "surface": "#16101f",
            "raised": "#1e172a",
            "overlay": "#271e36",
            "sunken": "#080510",
            "fg": "#eae5f2",
            "muted": "#b3a8c6",
            "dim": "#82749c",
            "faint": "#4a4060",
            "edge": "#2e2340",
            "edge-hi": "#3d2f54",
            "accent": "#fcee0a",
            "success": "#3fe07a",
            "warning": "#ff9e3d",
            "danger": "#ff003c",
            "signal": "#00f0ff",
            "label": "#b98aff",
            "string": "#3fe07a",
            "tint-danger": "#2c1420",
            "tint-select": "#28230e",
            "tint-select-hi": "#332c12",
            "tint-attach": "#122733",
            "tint-attach-hi": "#1a3a4c",
        },
    ),
    # Mall-at-closing-time pastels. Same purple ground family as
    # synthwave but every state hue is pulled toward chalk — dusty pink
    # accent, sun-bleached teal signal, sherbet amber — so the theme
    # reads soft-focus where synthwave reads laser-sharp. The pastels
    # start bright enough that no floor forces a deviation; the restraint
    # is the aesthetic.
    ThemeSpec(
        name="vaporwave",
        label="Vaporwave",
        description="Dusty pink and faded teal pastels on deep mall-purple",
        dark=True,
        tokens={
            "bg": "#1f1730",
            "surface": "#271e3b",
            "raised": "#2f2547",
            "overlay": "#382d54",
            "sunken": "#181128",
            "fg": "#ede8f2",
            "muted": "#c0b2d4",
            "dim": "#9184ae",
            "faint": "#5a4e74",
            "edge": "#3c3158",
            "edge-hi": "#4c3f6c",
            "accent": "#f7a8d8",
            "success": "#8fe6c0",
            "warning": "#f0cd8a",
            "danger": "#f2808a",
            "signal": "#7fd8d4",
            "label": "#c5a3f0",
            "string": "#8fe6c0",
            "tint-danger": "#3a2138",
            "tint-select": "#232a48",
            "tint-select-hi": "#2a3358",
            "tint-attach": "#1c2f44",
            "tint-attach-hi": "#28425e",
        },
    ),
    # Sunset-grid racing. The canonical Outrun quad lands intact on the
    # midnight-blue ground: magenta #ff2975 as accent (5.19:1 — clears
    # the floor with room), grid cyan #2de2e6 as signal, sun yellow
    # #f9c80e as warning, and the sunset orange #ff6c11 as danger — this
    # palette has no red, and orange-as-danger keeps the sunset story
    # while staying far from the magenta accent. Selection tints are
    # indigo (the grid at dusk), not green.
    ThemeSpec(
        name="outrun",
        label="Outrun",
        description="Sunset magenta and grid cyan racing over midnight blue",
        dark=True,
        tokens={
            "bg": "#0d1029",
            "surface": "#141834",
            "raised": "#1b2040",
            "overlay": "#23294e",
            "sunken": "#080a1e",
            "fg": "#e6e6f2",
            "muted": "#adafd0",
            "dim": "#7c7fa8",
            "faint": "#454870",
            "edge": "#262c56",
            "edge-hi": "#343b6e",
            "accent": "#ff2975",
            "success": "#3fe0a0",
            "warning": "#f9c80e",
            "danger": "#ff6c11",
            "signal": "#2de2e6",
            "label": "#a48fff",
            "string": "#3fe0a0",
            "tint-danger": "#301526",
            "tint-select": "#1c1444",
            "tint-select-hi": "#241a54",
            "tint-attach": "#102a3e",
            "tint-attach-hi": "#183d58",
        },
    ),
    # Neon through rain. The one theme in this family where the neon is
    # REFLECTED, not direct: a slightly cool charcoal ground and every
    # sign hue knocked back to wet-pavement saturation — dim cyan accent,
    # rose danger, sodium amber. Deliberately the quietest member; it
    # should feel like the city at 3am, two blocks from the signs.
    ThemeSpec(
        name="neon-noir",
        label="Neon Noir",
        description="Rain-dimmed cyan and magenta signs over 3am charcoal",
        dark=True,
        tokens={
            "bg": "#15171c",
            "surface": "#1c1f26",
            "raised": "#242830",
            "overlay": "#2c313b",
            "sunken": "#101216",
            "fg": "#dcdfe4",
            "muted": "#a6acb8",
            "dim": "#767e8c",
            "faint": "#454b56",
            "edge": "#2b303a",
            "edge-hi": "#3a414e",
            "accent": "#5fc4d4",
            "success": "#6cc49a",
            "warning": "#cfae62",
            "danger": "#e07a8a",
            "signal": "#7aa8d8",
            "label": "#b48ec6",
            "string": "#6cc49a",
            "tint-danger": "#2a1c22",
            "tint-select": "#1a262c",
            "tint-select-hi": "#203038",
            "tint-attach": "#1a2634",
            "tint-attach-hi": "#243648",
        },
    ),
    # CRT cabinet. Near-neutral black glass (the one ground here with no
    # strong cast — arcade CRTs were glass-grey when off) and candy
    # primaries for the states: marquee yellow accent, 1-up green,
    # bonus-round orange, hit-flash red, ice-level blue, power-up violet.
    # Each state is its own primary so nothing can collapse; the fg is a
    # warm bone white like a lit dot-matrix score.
    ThemeSpec(
        name="arcade",
        label="Arcade",
        description="Candy RGB game states glowing on black CRT glass",
        dark=True,
        tokens={
            "bg": "#0a0a0c",
            "surface": "#141417",
            "raised": "#1d1d21",
            "overlay": "#26262c",
            "sunken": "#050506",
            "fg": "#e8e8e4",
            "muted": "#b0b0ac",
            "dim": "#7c7c7a",
            "faint": "#44444a",
            "edge": "#2a2a30",
            "edge-hi": "#3a3a42",
            "accent": "#ffd23f",
            "success": "#45e055",
            "warning": "#ff9430",
            "danger": "#ff5252",
            "signal": "#52b4ff",
            "label": "#c792ff",
            "string": "#45e055",
            "tint-danger": "#26100f",
            "tint-select": "#0e2213",
            "tint-select-hi": "#132d1a",
            "tint-attach": "#0e2030",
            "tint-attach-hi": "#153048",
        },
    ),
]
