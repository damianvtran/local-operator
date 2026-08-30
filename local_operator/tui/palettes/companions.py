"""Companions to themes this product already ships.

Every ramp here is the missing half of a pair a user has already met. Someone
who picked Dracula and then opens their laptop on a train platform wants
Alucard — Dracula's own daylight scheme, not "a light theme"; someone running
Gruvbox dark wants morhetz's light mode, not a beige approximation of it. The
same argument adds the two Catppuccin flavors between the ``mocha`` and
``latte`` already registered, and Tokyo Night's ``day`` beside its ``night``
and ``storm``. The promise is identity, not similarity: the daylight version
has to wear the same hues in the same roles, so a user switching grounds does
not also have to relearn which color means "failed".

That promise is what constrains the solving here. Elsewhere a palette may
re-map hues freely to fit the token set; in this file a companion keeps its
sibling's mapping wherever the floors allow, because the pair is the point.
Where the two disagree it is recorded — ``one-light`` puts ``string`` on green
exactly as the registered ``one-dark`` does, and Alucard spends purple on
``accent`` for the same reason ``dracula`` does.

Fidelity rules, applied to every hex below:

- Published values ship UNCHANGED wherever they clear the gate. A companion
  whose accent is not the upstream accent is not a companion.
- Every deviation carries the canonical hex, the ratio that actually failed,
  and the substitute — measured with the gate's own formula, not estimated.
  A comment asserting a number is the evidence the next author trusts, so a
  wrong one is worse than no comment.
- Light companions re-solve DOWNWARD and dark ones upward, always along the
  source hue: a red that stops being red to buy contrast has lost the thing
  it was fetched for.

The light ramps here follow ``lights.py``'s family solves (elevation darkens
upward, ``sunken`` at or just below the paper, tints as casts rather than
slabs). Every palette clears ``tests/unit/tui/test_palette_contrast.py``.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    # Alucard — Dracula's OFFICIAL light counterpart (draculatheme.com/spec),
    # the companion to the `dracula` this repo already registers. Every
    # content hex is canonical and UNCHANGED: the spec's authors solved this
    # palette for a paper ground, and all seven state hues clear the 4:1
    # floor on both grounds unaided (the tightest is `string`, the spec's
    # yellow #846E15, at 4.80:1/4.59:1). `overlay` is the spec's own opaque
    # line-highlight fallback #E2DECA; the intermediate ground steps are
    # interpolated between it and the background, which the spec does not
    # enumerate.
    #
    # The hue mapping mirrors `dracula`'s deliberately, so the pair reads as
    # one theme in two lights: purple is Dracula's identity color and takes
    # `accent` in both, which leaves pink for `label` in both. Alucard's pink
    # #A3144D is a deep rose rather than the night variant's hot pink —
    # that is the spec's own light-mode answer to the same hue, not a
    # substitution.
    ThemeSpec(
        name="alucard",
        label="Alucard",
        description="Dracula's official daylight half, warm cream and deep jewel ink",
        dark=False,
        tokens={
            "bg": "#fffbeb",  # canonical Alucard background
            "surface": "#fbf6e2",
            "raised": "#f5efd6",
            "overlay": "#e2deca",  # the spec's opaque line-highlight fallback
            "sunken": "#f8f3dd",  # a hair below the paper — light polarity has no well
            "fg": "#1f1f1f",  # canonical foreground (15.89:1)
            # Canonical comment #6C664B is 5.56:1 — comfortably a `dim`, not a
            # `muted`, so it takes the rung it actually reads as and `muted`
            # is solved one step deeper along the same olive-khaki hue.
            "muted": "#4f4a37",
            "dim": "#6c664b",  # canonical comment/current-line
            "faint": "#9b937a",  # same hue, lifted: 2.96:1, below dim's 5.56:1
            "edge": "#efe9d0",
            "edge-hi": "#d9d4bd",
            "accent": "#644ac9",  # canonical purple — Dracula's identity hue
            "success": "#14710a",  # canonical green
            "warning": "#a34d14",  # canonical orange
            "danger": "#cb3a2a",  # canonical red
            "string": "#846e15",  # canonical yellow (the spec assigns it to strings)
            "signal": "#036a96",  # canonical cyan
            "label": "#a3144d",  # canonical pink; purple is spent on accent
            "tint-danger": "#fbeae2",  # canonical red reads 4.30:1 here
            "tint-select": "#eeecf6",  # violet cast, pulled from the canonical selection
            "tint-select-hi": "#e4e1f0",
            "tint-attach": "#e6eff3",  # cool cyan cast — signal is the file hue
            "tint-attach-hi": "#d3e3ea",
        },
    ),
    # Gruvbox Light (medium) — morhetz's own light mode, the companion to the
    # registered `gruvbox` dark. Grounds and text ladder are canonical:
    # bg0 #fbf1c7, light0_soft #f2e5bc as `raised`, fg1 #3c3836 as the ink,
    # dark3 #665c54 as `muted`, light4 #a89984 as `faint`, light2 #d5c4a1 as
    # `edge-hi`. The accent row is gruvbox's own FADED set, which is what the
    # light mode actually paints with — the bright set belongs to the dark
    # ramp and would be unreadable on cream.
    #
    # Yellow is the one hue that cannot stay canonical. On the dark ramp
    # yellow is gruvbox's face and takes `accent`; here faded_yellow is too
    # light to carry a state, so the mapping rotates rather than compromising:
    # faded_blue takes `accent` (the strongest light-mode hue), faded_orange
    # takes `signal`, and the darkened yellow keeps `warning`, the role its
    # meaning already fits.
    ThemeSpec(
        name="gruvbox-light",
        label="Gruvbox Light",
        description="Retro groove by daylight, warm cream and faded ink",
        dark=False,
        tokens={
            "bg": "#fbf1c7",  # canonical light0
            # Canonical light0_soft #f2e5bc is the natural first step, but it
            # measures 3.87:1 against canonical faded_green — under the 4:1
            # state floor on the ground that carries it. `surface` is lifted
            # to #f7ebbd so the whole canonical accent row survives, and
            # light0_soft keeps the `raised` rung one step up.
            "surface": "#f7ebbd",
            "raised": "#f2e5bc",  # canonical light0_soft
            "overlay": "#e6d9ab",
            "sunken": "#f9eec2",
            "fg": "#3c3836",  # canonical fg1/dark1 (10.22:1)
            "muted": "#665c54",  # canonical dark3
            # Canonical gray #928374 is 3.24:1 on bg and 3.07:1 on surface,
            # under the 3.4 `dim` floor. Darkened along its own warm gray to
            # the smallest value that clears both (3.61:1/3.43:1) — the floor
            # for dim is 3.4, not 4.0, so this stays a small nudge.
            "dim": "#8a7b6c",
            "faint": "#a89984",  # canonical light4 — 2.45:1, below dim's 3.61:1
            "edge": "#eee1b3",
            "edge-hi": "#d5c4a1",  # canonical light2
            "accent": "#076678",  # canonical faded_blue
            "success": "#79740e",  # canonical faded_green
            # Canonical faded_yellow #b57614 is 3.33:1 on bg (< 4); darkened
            # along the same hue to 4.26:1/4.05:1.
            "warning": "#9d6611",
            "danger": "#9d0006",  # canonical faded_red
            "string": "#427b58",  # canonical faded_aqua
            # faded_aqua, not faded_orange. The first solve gave `signal` the
            # orange #af3a03, which sits only ΔE00 10.79 from danger #9d0006 —
            # both dark red-orange on cream, and in the rendered frame the
            # failed row's message and the inline code were the same colour to
            # the eye, so "error" stopped being a distinguishable state
            # (design review D5). Aqua is equally canonical, clears the floor
            # on both grounds (4.40:1 / 4.11:1), and lands ΔE00 55.15 off danger.
            "signal": "#427b58",  # canonical faded_aqua
            "label": "#8f3f71",  # canonical faded_purple
            "tint-danger": "#f6e2c2",
            "tint-select": "#eeecb4",  # olive cast — gruvbox green is yellow-green
            "tint-select-hi": "#e6e3a6",
            "tint-attach": "#e9e8c6",
            "tint-attach-hi": "#d9dcb8",
        },
    ),
    # Tokyo Night Day — folke's `day` variant, completing the night/storm/day
    # set this repo already carries two thirds of.
    #
    # This variant is DERIVED, not published: upstream generates it by calling
    # `Util.invert` on the night palette (lua/tokyonight/colors/day.lua), so
    # there is no hand-authored hex table to be faithful to. The values below
    # are taken from upstream's own GENERATED artifacts — extras/kitty/
    # tokyonight_day.conf and extras/wezterm/tokyonight_day.toml, which agree
    # hex for hex — because those are the closest thing to a canonical day
    # palette that exists. Treat them as upstream's output rather than as a
    # spec: a future upstream retune of `night` moves all of these.
    #
    # Inversion optimises for a text editor's syntax, not for one-word UI
    # states, and it shows: five of the six generated accents land in the
    # 3.0–3.4:1 band on the generated background, so each is darkened along
    # its own hue by the minimum that clears both grounds. The hue ASSIGNMENT
    # is kept identical to the registered `tokyo-night` (blue accent, cyan
    # signal, magenta label) so the three variants stay one family.
    ThemeSpec(
        name="tokyo-night-day",
        label="Tokyo Night Day",
        description="Tokyo Night at noon, cool paper under the same neon hues",
        dark=False,
        tokens={
            "bg": "#e1e2e7",  # generated day background
            "surface": "#d9dae1",
            "raised": "#d0d2db",
            "overlay": "#c4c8da",  # generated scrollbar/inactive-tab ground
            "sunken": "#dcdde3",
            # Generated fg #3760bf is 4.52:1 — a fine syntax blue, but far
            # under the 7:1 body floor. Deepened along the same indigo.
            "fg": "#254181",
            # Generated comment #6172b0 is 3.57:1 (< 4.5); deepened on-hue.
            "muted": "#4d5d99",
            # Generated dark5 #8990b3 is 2.42:1 (< 3.4); deepened on-hue.
            "dim": "#67709d",
            "faint": "#8990b3",  # generated dark5 — 2.42:1, below dim's 3.71:1
            "edge": "#d3d5de",
            "edge-hi": "#b7c1e3",  # generated selection background
            "accent": "#1664cf",  # generated blue #2e7de9 is 3.11:1 (< 4)
            "success": "#547036",  # generated green #587539 is 3.75:1 on surface
            "warning": "#806239",  # generated yellow #8c6c3e is 3.75:1 (< 4)
            "danger": "#ce0a43",  # generated red #f52a65 is 3.01:1 (< 4)
            "string": "#387068",  # generated teal (url_color) — canonical, 4.40:1
            "signal": "#007096",  # generated cyan #007197 is 3.96:1 on surface
            "label": "#8635ee",  # generated purple #9854f1 is 3.33:1 (< 4)
            "tint-danger": "#e9d9dd",  # danger reads 4.12:1 here
            "tint-select": "#d8dced",  # blue cast — selection wears the accent
            "tint-select-hi": "#cdd3e8",
            "tint-attach": "#d6dfe4",
            "tint-attach-hi": "#c5d4dc",
        },
    ),
    # One Light — Atom's official light syntax theme (atom/one-light-syntax),
    # the companion to the registered `one-dark`. Upstream defines its colors
    # in HSL rather than hex (styles/colors.less), so the values below are
    # that source converted: mono-1 hsl(230,8%,24%) = #383a42,
    # mono-2 = #696c77, mono-3 = #a0a1a7, and the hue-N accent row.
    #
    # One Light is tuned for syntax on white and most of its accents are too
    # light for a one-word state: red, green, yellow and blue all land in the
    # 3.0–3.9:1 band and are darkened along their own hues. Purple #a626a4 and
    # mono-2 are canonical and untouched. The role mapping mirrors `one-dark`
    # exactly — blue is Atom's identity and takes `accent`, cyan `signal`,
    # purple `label`, and `string` sits on green in both.
    ThemeSpec(
        name="one-light",
        label="One Light",
        description="Atom's daylight standard, cool white with the One accent row",
        dark=False,
        tokens={
            "bg": "#fafafa",  # canonical syntax-bg hsl(230,1%,98%)
            "surface": "#f0f0f2",
            "raised": "#e6e6e9",
            "overlay": "#d8d8dd",
            "sunken": "#f4f4f6",
            "fg": "#383a42",  # canonical mono-1 (10.86:1)
            "muted": "#696c77",  # canonical mono-2 — clears 4.5 unaided
            # Canonical mono-3 #a0a1a7 is 2.47:1 (< 3.4); darkened on-hue.
            "dim": "#808189",
            "faint": "#a0a1a7",  # canonical mono-3 — 2.47:1, below dim's 3.71:1
            "edge": "#e4e4e7",
            "edge-hi": "#c9cace",
            "accent": "#306df1",  # canonical hue-2 blue #4078f2 is 3.56:1 on surface
            "success": "#428441",  # canonical hue-4 green #50a14f is 3.07:1 (< 4)
            "warning": "#9d6c01",  # canonical hue-6-2 #c18401 is 3.06:1 (< 4)
            "danger": "#de3323",  # canonical hue-5 red #e45649 is 3.51:1 (< 4)
            "string": "#428441",  # green, as `one-dark` also assigns it
            "signal": "#017db2",  # canonical hue-1 cyan #0184bc is 3.67:1 on surface
            "label": "#a626a4",  # canonical hue-3 purple — 5.86:1, untouched
            "tint-danger": "#faeeec",  # danger reads 4.02:1 here
            "tint-select": "#e2e9fb",  # blue cast — selection wears the accent
            "tint-select-hi": "#d3def8",
            "tint-attach": "#e3eef3",
            "tint-attach-hi": "#cfe2ec",
        },
    ),
    # Catppuccin Frappé — the third flavor, filling the gap between the
    # registered `catppuccin-mocha` and `catppuccin-latte`. Values are
    # catppuccin/palette v1.8.0. Frappé is generous enough that the entire
    # accent row ships canonical, and the palette supplies its own ground
    # ladder: mantle as `sunken`, surface0 as `raised`, surface1 as
    # `overlay`/`edge-hi`, and text/subtext0 as the top two ink rungs.
    #
    # Role assignment follows `catppuccin-mocha` exactly — mauve is the
    # brand's signature and takes `accent`, lavender `label`, blue `signal` —
    # so the three flavors differ only in ground temperature, which is the
    # whole point of a flavor.
    ThemeSpec(
        name="catppuccin-frappe",
        label="Catppuccin Frappé",
        description="The middle flavor, warm slate under the Catppuccin pastels",
        dark=True,
        tokens={
            "bg": "#303446",  # canonical base
            # Sits between base and surface0 rather than on surface0 itself:
            # canonical overlay1 (`dim` below) needs a ground no lighter than
            # this to clear 3.4, and this keeps a visible elevation step.
            "surface": "#383c52",
            "raised": "#414559",  # canonical surface0
            "overlay": "#51576d",  # canonical surface1
            "sunken": "#292c3c",  # canonical mantle
            "fg": "#c6d0f5",  # canonical text (8.06:1)
            "muted": "#a5adce",  # canonical subtext0
            # Canonical overlay1 #838ba7 is 3.21:1 on the surface step
            # (< 3.4); lifted minimally along the same blue-gray.
            "dim": "#8890ab",
            "faint": "#737994",  # canonical overlay0 — 2.87:1, below dim's 3.88:1
            "edge": "#3b3f56",
            "edge-hi": "#51576d",  # canonical surface1
            "accent": "#ca9ee6",  # canonical mauve — Catppuccin's signature
            "success": "#a6d189",  # canonical green
            "warning": "#ef9f76",  # canonical peach
            "danger": "#e78284",  # canonical red
            "string": "#a6d189",
            "signal": "#8caaee",  # canonical blue
            "label": "#babbf1",  # canonical lavender
            "tint-danger": "#3f3446",  # danger reads 4.42:1 here
            "tint-select": "#3a374f",  # mauve cast — selection wears the accent
            "tint-select-hi": "#433f5b",
            "tint-attach": "#333c53",
            "tint-attach-hi": "#3d4a68",
        },
    ),
    # Catppuccin Macchiato — the fourth flavor, same palette version. Like
    # frappé the whole accent row is canonical and the ground ladder is the
    # palette's own (mantle `sunken`, surface0 `raised`, surface1 `overlay`).
    # Macchiato's darker base gives every hue more room: the tightest state
    # here is red at 5.96:1/5.26:1, against frappé's 4.65:1/4.09:1.
    #
    # Frappé and macchiato are close cousins and the risk is shipping one
    # theme twice. Upstream separates them at the GROUND — base #303446 vs
    # #24273a, ΔE00 4.22 — and this keeps that separation rather than
    # splitting the difference: each ramp is built from its own flavor's
    # mantle/surface steps, so the two ladders measure 3.9–4.2 apart at every
    # rung instead of converging on a shared mid-slate. The accent rows are
    # genuinely different hexes upstream too, and both ship unmodified.
    ThemeSpec(
        name="catppuccin-macchiato",
        label="Catppuccin Macchiato",
        description="The deep flavor, cool navy-slate under the same pastels",
        dark=True,
        tokens={
            "bg": "#24273a",  # canonical base
            "surface": "#2c3045",
            "raised": "#363a4f",  # canonical surface0
            "overlay": "#494d64",  # canonical surface1
            "sunken": "#1e2030",  # canonical mantle
            "fg": "#cad3f5",  # canonical text (9.92:1)
            "muted": "#a5adcb",  # canonical subtext0
            "dim": "#8087a2",  # canonical overlay1 — clears 3.65:1 on surface
            "faint": "#6e738d",  # canonical overlay0 — 3.15:1, below dim's 4.14:1
            "edge": "#2f3348",
            "edge-hi": "#494d64",  # canonical surface1
            "accent": "#c6a0f6",  # canonical mauve
            "success": "#a6da95",  # canonical green
            "warning": "#f5a97f",  # canonical peach
            "danger": "#ed8796",  # canonical red
            "string": "#a6da95",
            "signal": "#8aadf4",  # canonical blue
            "label": "#b7bdf8",  # canonical lavender
            "tint-danger": "#33283c",  # danger reads 5.64:1 here
            "tint-select": "#2d2a44",  # mauve cast
            "tint-select-hi": "#36324f",
            "tint-attach": "#263048",
            "tint-attach-hi": "#2f3d5c",
        },
    ),
    # Palenight — Material Palenight, the indigo cousin of the Material
    # family, from the Material Theme UI palette reference
    # (material-theme.com/docs/reference/color-palette). Background #292D3E,
    # comment/gray #676E95, contrast #202331 as `sunken` and highlight
    # #444267 as `overlay` are all upstream values.
    #
    # The accent row is canonical and untouched — Palenight's whole appeal is
    # that saturated row on indigo, and every one of the six clears 4:1 on
    # both grounds unaided (red #f07178 is tightest at 4.77:1/4.18:1). The
    # text ladder is where the work is: upstream's `foreground` #A6ACCD is
    # 6.11:1, under the 7:1 body floor, so `fg` takes upstream's OWN brighter
    # variables color #eeffff (13.24:1) and #A6ACCD drops to `muted`, the
    # rung it actually measures as.
    #
    # Purple takes `accent` rather than upstream's accent #ab47bc, which is
    # 2.83:1 on its own background — an underline/badge color, not text. That
    # is a role reassignment inside the palette, not a new hue.
    ThemeSpec(
        name="palenight",
        label="Palenight",
        description="Material's indigo evening, saturated hues on soft violet-slate",
        dark=True,
        tokens={
            "bg": "#292d3e",  # canonical background
            "surface": "#31364a",
            "raised": "#383d54",
            "overlay": "#444267",  # canonical highlight
            "sunken": "#202331",  # canonical contrast
            "fg": "#eeffff",  # canonical variables/white (13.24:1)
            "muted": "#a6accd",  # canonical foreground — 6.11:1, a muted not a body ink
            # Canonical comment/gray #676E95 is 2.76:1 (< 3.4); lifted on-hue.
            "dim": "#8287a8",
            "faint": "#676e95",  # canonical comment — 2.76:1, below dim's 3.89:1
            "edge": "#343850",
            "edge-hi": "#4b5171",
            "accent": "#c792ea",  # canonical purple (upstream's keywords hue)
            "success": "#c3e88d",  # canonical green
            "warning": "#f78c6c",  # canonical orange
            "danger": "#f07178",  # canonical red
            "string": "#c3e88d",
            "signal": "#89ddff",  # canonical cyan
            "label": "#82aaff",  # canonical blue
            "tint-danger": "#3b2c3d",  # danger reads 4.55:1 here
            "tint-select": "#332f4a",  # violet cast — selection wears the accent
            "tint-select-hi": "#3b3757",
            "tint-attach": "#25384a",
            "tint-attach-hi": "#31506b",
        },
    ),
]
