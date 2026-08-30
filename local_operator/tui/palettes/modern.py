"""The current generation of editor themes, mapped onto the token set.

Where ``classics`` collects the names that were already standards when this
UI was written, these are the schemes people arrive from Neovim and VS Code
already using — Everforest, Kanagawa, Ayu and the Nightfox family, all of
which postdate the classics/neon/nature families and none of which occupy a
space those families already cover: a warm green-tinted dark, an ink-wash
palette after Hokusai, and two blue-blacks that have to be kept apart from
each other.

Three of these ship as multi-variant families upstream (Everforest
dark/light, Kanagawa wave/lotus, Ayu dark/mirage/light). The rule this file
holds to is that within a family the ACCENT ROW stays recognisably the same
palette across variants — that is what makes them variants rather than
unrelated themes — while grounds and any floor re-solve belong to the
individual variant. Across families they are held apart: ``ayu-dark`` and
``nightfox`` are both blue-black, and their accent rows measure a mean
ΔE00 21.9 apart, so the grounds being close (ΔE00 5.4) does not make them
the same theme.

The light variants are the expensive ones, for the reason ``lights.py``
documents family-wide: a hue that clears 4:1 on near-black arrives at
2–3:1 on paper. Everforest's light accents measure 2.12–3.13:1 on their own
canonical ``bg0`` and Ayu's light accents 1.96–3.29:1, so every state hue in
those two is re-solved DOWNWARD along its own hue until it clears 4:1 on
both grounds. Kanagawa's lotus needs it less: Kanagawa published a genuinely
dark light-variant ink set, and four of its accents pass unmodified.

Every deviation from a published value carries the canonical hex, the ratio
that failed and the substitute, because a comment asserting a number is the
evidence the next author will trust. Ratios are the gate's own formula from
``tests/unit/tui/test_palette_contrast.py``; run that file after any tweak,
then render ``scripts/theme_preview.py`` and look at the frames.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Everforest — sainnhe/everforest, `palette.md`, dark/medium. The ground
    # ladder is canonical and needs no interpolation: everforest publishes
    # bg_dim/bg0/bg1/bg2/bg3/bg4, which is exactly the five steps this UI
    # wants plus a hairline. The accent row is canonical too, with one
    # constraint driving the assignment: everforest's `green` and `aqua`
    # measure only ΔE00 9.4 apart — under the loosest shipped precedent — so
    # they cannot both sit in the five-state set. Aqua takes `string`, which
    # is where everforest's own highlighting puts it (strings/characters),
    # leaving accent/success/warning/danger/signal at a minimum separation of
    # ΔE00 14.6.
    ThemeSpec(
        name="everforest",
        label="Everforest",
        description="Warm green-tinted dark, built for long low-strain reading",
        dark=True,
        tokens={
            "bg": "#2D353B",  # canonical bg0
            "surface": "#343F44",  # canonical bg1
            "raised": "#3D484D",  # canonical bg2
            "overlay": "#475258",  # canonical bg3
            "sunken": "#232A2E",  # canonical bg_dim
            # Canonical fg #D3C6AA is 6.40:1 on bg1 (< 7); lifted on-hue.
            "fg": "#DACFB7",
            # Canonical grey2 #9DA9A0 is 4.44:1 on bg1 (< 4.5); lifted on-hue.
            "muted": "#A0ABA3",
            # Canonical grey1 #859289 is 3.33:1 on bg1 (< 3.4); lifted on-hue.
            "dim": "#88948C",
            "faint": "#7A8478",  # canonical grey0 (3.21:1) — the inert rung
            "edge": "#475258",  # canonical bg3
            "edge-hi": "#4F585E",  # canonical bg4
            # Design review D4: the first solve took upstream's SYNTAX role
            # names literally — accent=blue, signal=purple — which left the
            # theme's signature green #A7C080 spent only on a success glyph.
            # The rendered frame read as pink-on-slate: nothing in it said
            # "everforest". This repo's `accent` is the theme's IDENTITY hue
            # (caret, live indicator, focus), so it takes the green, and
            # `signal` — the file/reference ink that paints most of the code
            # in a reply — takes canonical blue instead of the purple.
            "accent": "#A7C080",  # canonical green — the signature
            "success": "#83C092",  # canonical aqua, one step cooler than accent
            "warning": "#DBBC7F",  # canonical yellow
            # Canonical red #E67E80 is 3.95:1 on bg1 (< 4); lifted on-hue by
            # the minimum that clears both grounds (4.62/4.01).
            "danger": "#E68082",
            "string": "#DBBC7F",  # canonical yellow, upstream's string hue
            "signal": "#7FBBB3",  # canonical blue — the cool file/reference ink
            "label": "#D699B6",  # canonical purple
            # Canonical bg_red #514045 puts the lifted red at 3.58:1 — under
            # the 4.0 floor on the one band that always carries it. Sunk
            # toward bg0 with the maroon lead kept, which clears 4.27:1.
            "tint-danger": "#453539",
            "tint-select": "#425047",  # canonical bg_green
            "tint-select-hi": "#4B5B51",
            "tint-attach": "#3A515D",  # canonical bg_blue
            "tint-attach-hi": "#445E6B",
        },
    ),
    # Everforest Light — the official light counterpart, medium contrast.
    # Grounds are canonical (bg0/bg1/bg2/bg3/bg4). The accent row is where
    # the light solve lands hardest: on canonical bg0 the published accents
    # measure red 3.04, orange 2.48, yellow 2.12, green 2.69, aqua 2.79,
    # blue 3.13, purple 2.83 — every one of them under the 4.0 state floor,
    # which is the family-wide light problem `lights.py` describes rather
    # than anything wrong with everforest. Each is darkened along its own
    # hue by the minimum that clears 4:1 on BOTH grounds. Hue drift against
    # the dark variant is upstream's own (everforest's light "blue" #3A94C5
    # is a true blue where its dark "blue" #7FBBB3 is a teal); this file
    # preserves that rather than inventing a shared hue neither ships.
    ThemeSpec(
        name="everforest-light",
        label="Everforest Light",
        description="The forest scheme on warm paper, soft and low-glare",
        dark=False,
        tokens={
            "bg": "#FDF6E3",  # canonical bg0
            "surface": "#F4F0D9",  # canonical bg1
            "raised": "#E6E2CC",  # canonical bg3 (bg2 sits ΔE00 1.06 from surface)
            # Canonical bg2 #EFEBD4 sits only ΔE00 1.06 from bg1, and on paper that
            # middle step reads as no step at all (design review D2: elevation
            # rungs were indistinguishable in the frame). `raised` takes bg3
            # and `overlay` the SOFT variant's bg3 — both upstream everforest
            # grounds — for an even 2.72 / 3.02 / 2.68 ΔE00 ladder.
            "overlay": "#DDD8BE",  # canonical bg3 (soft variant)
            "sunken": "#F8F2DC",  # a hair below bg0 — light polarity has no well
            # Canonical fg #5C6A72 is 5.18:1 on bg0 (< 7); darkened on-hue.
            "fg": "#465157",
            # Canonical grey2 #829181 is 3.08:1 on bg0 (< 4.5); darkened on-hue.
            "muted": "#637062",
            # Canonical grey1 #939F91 is 2.56:1 on bg0 (< 3.4); darkened on-hue.
            "dim": "#768474",
            "faint": "#A6B0A0",  # canonical grey0 (2.08:1) — the inert rung
            "edge": "#E6E2CC",  # canonical bg3
            "edge-hi": "#E0DCC7",  # canonical bg4
            # Same role re-map as the dark variant (design review D2): the
            # signature green carries `accent`, not `success`, and the cool
            # blue carries `signal`. The first solve put a darkened PURPLE on
            # `signal`, so the inline code and file paths — the most-painted
            # ink in a reply — came out hot magenta on cream, and the frame
            # read as anything but a forest.
            "accent": "#6C7B01",  # green, darkened from canonical #8DA101 (2.69:1)
            "success": "#18804F",  # aqua, darkened from canonical #35A77C (2.72:1)
            "warning": "#986D00",  # yellow, darkened from canonical #DFA000 (2.12:1)
            "danger": "#D63A37",  # red, darkened from canonical #F85552 (3.04:1)
            # Aqua, darkened from canonical #35A77C (2.79:1). Taken further
            # down than the floor alone requires: the minimum solve (#2A8462,
            # 4.26:1) landed ΔE00 2.6 from ayu-light's own string green, and
            # two themes in the same batch should not converge on one hex.
            # This sits at 7.1 from it and 19.8 from this theme's own success.
            "string": "#986D00",  # yellow, upstream's string hue (same solve as warning)
            "signal": "#317CA5",  # blue, darkened from canonical #3A94C5 (3.13:1)
            "label": "#A8317E",  # purple, darkened from canonical #DF69BA (2.83:1)
            # Canonical bg_red #FDE3DA puts the darkened red at 3.80:1 —
            # under the 4.0 floor. Lifted toward bg0, which clears 4.11:1.
            "tint-danger": "#FDEEE9",
            "tint-select": "#F0F1D2",  # canonical bg_green
            # DARKER than tint-select, not lighter: on light polarity an
            # additive hover has to deepen, or hover on the selected row
            # reads as the row going away.
            "tint-select-hi": "#E8EAC2",
            "tint-attach": "#E9F0E9",  # canonical bg_blue
            "tint-attach-hi": "#DDE8DD",
        },
    ),
    # Kanagawa (wave) — rebelot/kanagawa.nvim, the default variant, mapped
    # through `themes.lua` rather than guessed from the palette names: bg is
    # sumiInk3 (the theme's `ui.bg`), with sumiInk4/sumiInk5 above it and
    # sumiInk0 below. This is the one theme in the file that ships entirely
    # canonical — Kanagawa's ink-wash ground is dark enough that every
    # published accent clears the state floor with room (lowest is oniViolet
    # at 4.04:1 on the surface step), so nothing here is re-solved.
    #
    # `overlay` is waveBlue2 rather than waveBlue1: waveBlue1 is the popup
    # ground upstream, but it measures LOWER in luminance than sumiInk5, so
    # using it would paint the elevation ladder upside down. waveBlue2 is
    # kanagawa's own search/selection blue and is the lightest ground the
    # scheme names; waveBlue1 keeps its identity as the attach tint.
    ThemeSpec(
        name="kanagawa-wave",
        label="Kanagawa Wave",
        description="Hokusai ink-wash night, muted paper tones on deep sumi",
        dark=True,
        tokens={
            "bg": "#1F1F28",  # canonical sumiInk3 (ui.bg)
            "surface": "#2A2A37",  # canonical sumiInk4
            "raised": "#363646",  # canonical sumiInk5
            "overlay": "#2D4F67",  # canonical waveBlue2 (see the ladder note)
            "sunken": "#16161D",  # canonical sumiInk0
            "fg": "#DCD7BA",  # canonical fujiWhite (11.26:1)
            "muted": "#C8C093",  # canonical oldWhite
            "dim": "#938AA9",  # canonical springViolet1
            "faint": "#727169",  # canonical fujiGray (3.33:1) — the inert rung
            "edge": "#363646",  # canonical sumiInk5
            "edge-hi": "#54546D",  # canonical sumiInk6
            "accent": "#7E9CD8",  # canonical crystalBlue
            "success": "#98BB6C",  # canonical springGreen
            "warning": "#E6C384",  # canonical carpYellow
            "danger": "#FF5D62",  # canonical peachRed
            "string": "#98BB6C",  # springGreen — kanagawa's own string colour
            "signal": "#7AA89F",  # canonical waveAqua2
            "label": "#957FB8",  # canonical oniViolet
            "tint-danger": "#43242B",  # canonical winterRed
            "tint-select": "#2B3328",  # canonical winterGreen
            "tint-select-hi": "#354030",
            "tint-attach": "#252535",  # canonical winterBlue
            "tint-attach-hi": "#223249",  # canonical waveBlue1
        },
    ),
    # Kanagawa Lotus — the official light variant, mapped through the same
    # `themes.lua`. Kanagawa is unusual among light schemes in publishing a
    # genuinely dark ink set, so this needs less re-solving than the other
    # two light themes here: lotusBlue4, lotusViolet4, lotusInk1 and
    # lotusGray2 all pass unmodified. The four that miss are the ones
    # Kanagawa keeps deliberately soft.
    ThemeSpec(
        name="kanagawa-lotus",
        label="Kanagawa Lotus",
        description="The ink-wash palette on aged paper, warm and low-contrast",
        dark=False,
        tokens={
            "bg": "#F2ECBC",  # canonical lotusWhite3 (ui.bg)
            "surface": "#E5DDB0",  # canonical lotusWhite2
            "raised": "#DCD5AC",  # canonical lotusWhite1
            "overlay": "#D5CEA3",  # canonical lotusWhite0
            "sunken": "#E7DBA0",  # canonical lotusWhite4
            # Canonical lotusInk2 #43436C is 6.78:1 on lotusWhite2 (< 7);
            # darkened on-hue by one step (8.01/7.00).
            "fg": "#414169",
            "muted": "#545464",  # canonical lotusInk1 (6.19/5.41)
            "dim": "#716E61",  # canonical lotusGray2 (4.26/3.73)
            "faint": "#8A8980",  # canonical lotusGray3 (2.93:1) — the inert rung
            "edge": "#E4D794",  # canonical lotusWhite5
            "edge-hi": "#A09CAC",  # canonical lotusViolet1
            "accent": "#4D699B",  # canonical lotusBlue4 (4.59/4.02)
            "success": "#5A6F3F",  # green, darkened from canonical #6F894E (3.26:1)
            # Yellow, darkened from canonical #77713F (4.15/3.63). Taken
            # warmer as well as darker: the straight solve (#6F693B) sat only
            # ΔE00 9.2 from this theme's success olive — under the loosest
            # shipped precedent — because lotus's yellow and green are
            # neighbours upstream (ΔE00 11.9). This reads 19.7 from success
            # and 12.0 from the orange `label`.
            "warning": "#7A5A17",
            "danger": "#BD3649",  # red, darkened from canonical #C84053 (4.06/3.55)
            "string": "#506F6A",  # aqua, darkened from canonical #597B75 (3.88/3.39)
            "signal": "#624C83",  # canonical lotusViolet4 (6.07/5.31)
            "label": "#9E5400",  # orange, darkened from canonical #CC6D00 (3.04:1)
            # Kanagawa's own diff-delete ground is lotusRed4 #D9A594, a
            # salmon slab that reads as a fill rather than a cast (2.02:1 vs
            # bg) and drops the darkened red to 2.4:1. This is the paper
            # tinted warm instead: the red clears 4.25:1 on it.
            "tint-danger": "#F0DFC8",
            "tint-select": "#B7D0AE",  # canonical lotusGreen3
            "tint-select-hi": "#A9C4A0",  # deepened, not lightened (light polarity)
            "tint-attach": "#B5CBD2",  # canonical lotusBlue2
            "tint-attach-hi": "#9FB5C9",  # canonical lotusBlue3
        },
    ),
    # Ayu (dark) — ayu-theme/ayu-colors. The repo's `themes/*.yaml` is a
    # generator whose syntax entries are `$palette.<hue>.l<n>` references
    # rather than literal hexes, so these are taken from the published
    # package build (ayu@9.0.0, `dist/dark.js`), which resolves them. Ayu's
    # dark ground is the deepest in this file, and everything on it clears
    # the state floor canonically — lowest is markup red at 6.21:1 on the
    # surface step. `bg` is editor.bg with ui.bg below it as `sunken`, which
    # is the relationship ayu itself ships (ui.bg is the darker chrome).
    ThemeSpec(
        name="ayu-dark",
        label="Ayu Dark",
        description="Near-black blue with high-key amber and lime accents",
        dark=True,
        tokens={
            "bg": "#10141C",  # canonical editor.bg
            "surface": "#141821",  # canonical ui.panel.bg
            "raised": "#161A24",  # canonical editor.line
            "overlay": "#1B1F29",  # canonical ui.line
            "sunken": "#0D1017",  # canonical ui.bg
            "fg": "#BFBDB6",  # canonical editor.fg (9.81:1)
            # Ayu's comment is #ACB6BF at 55% alpha over the editor ground;
            # the flattened value is what this token needs, and it is what
            # the user actually sees upstream.
            "muted": "#ACB6BF",
            "dim": "#8A929E",
            "faint": "#565B66",  # canonical ui.fg (2.71:1) — the inert rung
            "edge": "#1B1F29",  # canonical ui.line
            "edge-hi": "#2B3038",
            "accent": "#59C2FF",  # canonical entity
            "success": "#AAD94C",  # canonical string
            "warning": "#FFB454",  # canonical func
            "danger": "#F07178",  # canonical markup
            "string": "#95E6CB",  # canonical regexp
            "signal": "#D2A6FF",  # canonical constant
            "label": "#FF8F40",  # canonical keyword
            "tint-danger": "#2A1A1E",
            "tint-select": "#152232",
            "tint-select-hi": "#1B2C40",
            "tint-attach": "#141F2A",
            "tint-attach-hi": "#1B2B3A",
        },
    ),
    # Ayu Mirage — the same generator's `mirage` variant, same package build.
    # Mirage is ayu's mid-dark ground, and its accent row is the dark row
    # lightened by the scheme itself, so the family relationship holds
    # without intervention here: every accent is canonical. Only the ground
    # ladder is solved, because ayu publishes three ground values for mirage
    # (ui.bg, editor.bg, panel.bg) where this UI needs five.
    ThemeSpec(
        name="ayu-mirage",
        label="Ayu Mirage",
        description="Ayu's slate-blue middle ground, softer than the dark",
        dark=True,
        tokens={
            "bg": "#242936",  # canonical editor.bg
            "surface": "#282E3B",  # canonical ui.panel.bg
            "raised": "#2E3544",
            "overlay": "#37404F",
            "sunken": "#1F2430",  # canonical ui.bg
            "fg": "#CCCAC2",  # canonical editor.fg (8.85:1)
            "muted": "#B8CFE6",  # canonical comment hue, flattened from 50% alpha
            "dim": "#8FA3B8",
            "faint": "#707A8C",  # canonical ui.fg (3.36:1) — the inert rung
            "edge": "#333B4A",
            "edge-hi": "#3F4859",
            "accent": "#73D0FF",  # canonical entity
            "success": "#D5FF80",  # canonical string
            "warning": "#FFD173",  # canonical func
            "danger": "#F28779",  # canonical markup
            "string": "#95E6CB",  # canonical regexp
            "signal": "#DFBFFF",  # canonical constant
            "label": "#FFAD66",  # canonical keyword
            "tint-danger": "#37292C",
            "tint-select": "#26303F",
            "tint-select-hi": "#2D3A4C",
            "tint-attach": "#25313D",
            "tint-attach-hi": "#2B3C4C",
        },
    ),
    # Ayu Light — same package build, `light` variant. Ayu light is the
    # brightest paper in this file (editor.bg is #FCFCFC, essentially white),
    # which makes it the hardest state-hue solve of the three: on that ground
    # the published accents measure entity 2.84, string 2.42, func 2.04,
    # markup 2.80, constant 3.29, keyword 2.47, regexp 2.22 — all under 4.0.
    # Every one is darkened along its own hue. The ground ladder is
    # canonical (editor.bg / panel.bg / ui.bg / surface.sunk).
    ThemeSpec(
        name="ayu-light",
        label="Ayu Light",
        description="Ayu on near-white paper, crisp with deepened accents",
        dark=False,
        tokens={
            # Upstream ayu light defines three near-identical whites for its
            # panel chrome: editor.bg #FCFCFC, ui.panel.bg #FAFAFA and ui.bg
            # #F8F9FA sit within ΔE00 0.73 of each other. In an EDITOR that is
            # fine — the panels are separated by borders and the code area is
            # the only large field. This UI has no borders on a tool row: the
            # ledger's rows are told apart by their GROUND alone, so those
            # three collapsed into one and the whole ledger dissolved into the
            # page (design review D1 — only the danger row was locatable).
            #
            # So bg keeps the canonical editor white and the rungs above it
            # are re-solved on ayu's own cool grey axis at ΔE00 1.77 / 1.81 per
            # step — the same separation catppuccin-latte ships (1.77) and
            # wider than solarized-light's (1.37). Deviating here is the only way
            # this theme can show a tool row at all.
            "bg": "#FCFCFC",  # canonical editor.bg
            "surface": "#F3F4F5",  # canonical ui.panel.bg #FAFAFA: ΔE00 0.40 (invisible)
            "raised": "#EAECEE",  # canonical ui.bg #F8F9FA: ΔE00 0.73 from surface
            "overlay": "#DDE1E4",  # deepened from surface.sunk #EBEEF0 for the last step
            "sunken": "#F7F8F9",
            # Canonical editor.fg #5C6166 is 6.10:1 on the paper (< 7);
            # darkened on-hue.
            # Every ink here is one step deeper than a straight port would
            # need, and the elevation ladder above is why: a `surface` far
            # enough from `bg` to be SEEN costs contrast on everything
            # measured against it. At canonical #FAFAFA the inks cleared
            # comfortably and no tool row was visible; at a surface the eye
            # can find, `fg` fell to 6.74:1 ON SURFACE (the ground the gate
            # checks) and six state hues to ~3.85:1.
            # Deepening the ink is what buys the ladder — the alternative was
            # a theme that passes the gate and shows no ledger.
            "fg": "#4D5256",
            # Canonical comment #787B80 is 4.14:1 (< 4.5); darkened on-hue.
            "muted": "#6C6E73",
            # Canonical ui.fg #828E9F is 3.24:1 (< 3.4); darkened on-hue.
            "dim": "#768293",
            "faint": "#9AA3AF",  # 2.49:1 — sits below dim, the inert rung
            "edge": "#E3E7EA",
            "edge-hi": "#CBD2D8",
            "accent": "#187BC1",  # entity, darkened from canonical #399EE6 (2.84:1)
            "success": "#618100",  # string, darkened from canonical #86B300 (2.42:1)
            "warning": "#A06C00",  # func, darkened from canonical #F2A300 (2.04:1)
            # Markup red, darkened from canonical #F07171 (2.80:1). The
            # straight lightness walk lands on #E93030, which clears the
            # floor but reads as a pure signal red where ayu's is muted;
            # this keeps more of the canonical's softer chroma at 4.34:1.
            "danger": "#C24A4A",
            "string": "#2F8468",  # regexp, darkened from canonical #4CBF99 (2.22:1)
            "signal": "#8F62BC",  # constant, darkened from canonical #A37ACC (3.29:1)
            "label": "#CD4C00",  # keyword, darkened from canonical #FF7E33 (2.47:1)
            # The darkened red reads 3.91:1 on a straight red wash at this
            # paper's luminance — under the 4.0 floor on the band that always
            # carries it. Lifted toward the paper, which clears 4.09:1.
            "tint-danger": "#FDF3F3",
            "tint-select": "#E8F0F8",
            "tint-select-hi": "#DAE7F3",  # deepened, not lightened (light polarity)
            "tint-attach": "#EAF2EE",
            "tint-attach-hi": "#DBEAE3",
        },
    ),
    # Nightfox — EdenEast/nightfox.nvim. The palette is not in the README;
    # it lives in `lua/nightfox/palette/nightfox.lua`, which is the source
    # this is taken from. The ground ladder and the whole text ramp are
    # canonical, including the ordering that matters most here: upstream's
    # fg3 #71839B (4.09:1) and comment #738091 (3.95:1) already sit in the
    # right relationship for `dim` above `faint`, so neither is invented.
    ThemeSpec(
        name="nightfox",
        label="Nightfox",
        description="Cool blue-slate night, even-toned and unhurried",
        dark=True,
        tokens={
            "bg": "#192330",  # canonical bg1 (default bg)
            "surface": "#212E3F",  # canonical bg2
            "raised": "#29394F",  # canonical bg3
            "overlay": "#39506D",  # canonical bg4
            "sunken": "#131A24",  # canonical bg0
            "fg": "#CDCECF",  # canonical fg1 (10.06:1)
            "muted": "#AEAFB0",  # canonical fg2
            "dim": "#71839B",  # canonical fg3
            "faint": "#738091",  # canonical comment — below fg3, the inert rung
            "edge": "#2B3B51",  # canonical sel0
            "edge-hi": "#3C5372",  # canonical sel1
            "accent": "#719CD6",  # canonical blue
            "success": "#81B29A",  # canonical green
            "warning": "#DBC074",  # canonical yellow
            # Canonical red #C94F6D is 3.64:1 on bg1 and 3.16:1 on bg2 (< 4);
            # lifted on-hue by the minimum that clears both (4.70/4.08).
            "danger": "#D26C85",
            "string": "#81B29A",  # green — nightfox's own string colour
            "signal": "#63CDCF",  # canonical cyan
            "label": "#9D79D6",  # canonical magenta
            "tint-danger": "#301F28",
            "tint-select": "#2B3B51",  # canonical sel0
            "tint-select-hi": "#33455E",
            "tint-attach": "#1D2B3A",
            "tint-attach-hi": "#26384C",
        },
    ),
    # Duskfox — the same plugin's `duskfox` variant, from
    # `lua/nightfox/palette/duskfox.lua`. Duskfox is the violet-ground member
    # of the family, which is what keeps it clear of nightfox's blue-slate
    # despite the shared structure. It ships entirely canonical: the violet
    # ground is dark enough that the lowest accent (blue, 4.62:1 on the
    # surface step) clears the state floor without help, and upstream's
    # comment/fg3 pair again orders correctly for dim above faint.
    ThemeSpec(
        name="duskfox",
        label="Duskfox",
        description="Violet-ground dusk, the rose-tinted end of the fox family",
        dark=True,
        tokens={
            "bg": "#232136",  # canonical bg1 (default bg)
            "surface": "#2D2A45",  # canonical bg2
            "raised": "#373354",  # canonical bg3
            "overlay": "#4B4673",  # canonical bg4
            "sunken": "#191726",  # canonical bg0
            "fg": "#E0DEF4",  # canonical fg1 (11.86:1)
            "muted": "#CDCBE0",  # canonical fg2
            "dim": "#817C9C",  # canonical comment (3.95/3.46)
            "faint": "#6E6A86",  # canonical fg3 (3.03:1) — the inert rung
            "edge": "#433C59",  # canonical sel0
            "edge-hi": "#63577D",  # canonical sel1
            "accent": "#569FBA",  # canonical blue
            "success": "#A3BE8C",  # canonical green
            "warning": "#F6C177",  # canonical yellow
            "danger": "#EB6F92",  # canonical red
            "string": "#A3BE8C",  # green — duskfox's own string colour
            "signal": "#9CCFD8",  # canonical cyan
            "label": "#C4A7E7",  # canonical magenta
            "tint-danger": "#3A2739",
            "tint-select": "#433C59",  # canonical sel0
            "tint-select-hi": "#514A6B",
            "tint-attach": "#26304A",
            "tint-attach-hi": "#31405F",
        },
    ),
]
