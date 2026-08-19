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
    # Dracula — draculatheme.com/spec. The published accents are all loud
    # enough for the state tokens (green 10.4:1, cyan 10.3:1, orange 8.4:1),
    # so fidelity is easy up top; the work is the TEXT ramp: canonical
    # comment #6272a4 is only 3.03:1 on the bg, far under the 4.5 muted
    # floor, so it lands on `faint` (its real role — inert hints) and
    # muted/dim are solved along the same blue-slate hue. Purple is the
    # signature (accent), pink takes the violet-ish `label` slot since
    # purple is spent, and `overlay` is the canonical current-line #44475a.
    ThemeSpec(
        name="dracula",
        label="Dracula",
        description="Purple-on-slate night with the famous pink and cyan",
        dark=True,
        tokens={
            "bg": "#282a36",
            "surface": "#2f323f",
            "raised": "#363948",
            "overlay": "#44475a",  # canonical "current line"
            "sunken": "#1e2029",
            "fg": "#f8f8f2",  # canonical foreground
            "muted": "#b6bdda",  # solved: comment #6272a4 is 3.03:1 (< 4.5)
            "dim": "#7b89bd",  # comment hue, lifted to clear 3.4:1
            "faint": "#6272a4",  # canonical comment — the inert-hint rung
            "edge": "#3d4152",
            "edge-hi": "#4d5268",
            "accent": "#bd93f9",  # canonical purple — THE Dracula color
            "success": "#50fa7b",
            "warning": "#ffb86c",
            "danger": "#ff5555",
            "string": "#50fa7b",
            "signal": "#8be9fd",  # canonical cyan
            "label": "#ff79c6",  # canonical pink; purple is spent on accent
            "tint-danger": "#3c2632",
            "tint-select": "#2f2a44",  # purple cast — selection wears the accent hue
            "tint-select-hi": "#373254",
            "tint-attach": "#263a4a",
            "tint-attach-hi": "#345670",
        },
    ),
    # Catppuccin Mocha — catppuccin.com/palette. The pastels are generous
    # (every state hue clears 7:1 on base), and the palette even supplies
    # the ground ramp: sunken = canonical mantle, overlay = surface0,
    # edge-hi = surface1, and the text ladder is canonical text/subtext0/
    # overlay1/overlay0 verbatim. Mauve is the brand's own signature color;
    # lavender takes `label`, blue takes `signal`.
    ThemeSpec(
        name="catppuccin-mocha",
        label="Catppuccin Mocha",
        description="Soothing pastels on a soft charcoal-blue base",
        dark=True,
        tokens={
            "bg": "#1e1e2e",  # canonical base
            "surface": "#262637",
            "raised": "#2d2d40",
            "overlay": "#313244",  # canonical surface0
            "sunken": "#181825",  # canonical mantle
            "fg": "#cdd6f4",  # canonical text
            "muted": "#a6adc8",  # canonical subtext0
            "dim": "#7f849c",  # canonical overlay1
            "faint": "#6c7086",  # canonical overlay0
            "edge": "#363653",
            "edge-hi": "#45475a",  # canonical surface1
            "accent": "#cba6f7",  # canonical mauve — Catppuccin's signature
            "success": "#a6e3a1",  # canonical green
            "warning": "#fab387",  # canonical peach
            "danger": "#f38ba8",  # canonical red
            "string": "#a6e3a1",
            "signal": "#89b4fa",  # canonical blue
            "label": "#b4befe",  # canonical lavender
            "tint-danger": "#332434",
            "tint-select": "#28283e",
            "tint-select-hi": "#2e2e49",
            "tint-attach": "#213048",
            "tint-attach-hi": "#2d4568",
        },
    ),
    # Catppuccin Latte — the light flavor. Elevation DARKENS upward (paper →
    # deeper cards), mirroring the brand light ramp's solve; `sunken` is the
    # canonical mantle, which sits just below the paper. Latte's published
    # accents are tuned for syntax at large sizes, not one-word UI states,
    # and several fail the 4:1 state floor on this paper: green #40a02b is
    # 2.96:1, yellow #df8e1d 2.31:1, lavender #7287fd 2.81:1. Each is
    # darkened along its own hue until it clears both grounds (green →
    # #2e7a1e, yellow → #8f6203, lavender → #5265d8). Canonical text
    # #4c4f69 also thins to 6.64:1 on the first elevation step, under the
    # 7:1 body floor, so fg deepens slightly to #42455f.
    ThemeSpec(
        name="catppuccin-latte",
        label="Catppuccin Latte",
        description="The Catppuccin pastels poured over warm morning paper",
        dark=False,
        tokens={
            "bg": "#eff1f5",  # canonical base
            "surface": "#e8eaf0",
            "raised": "#dfe2ea",
            "overlay": "#d3d7e3",
            "sunken": "#e6e9ef",  # canonical mantle — the band below the paper
            "fg": "#42455f",  # canonical text #4c4f69: 6.64:1 on surface (< 7)
            "muted": "#5c5f77",  # canonical subtext1
            "dim": "#787c92",
            "faint": "#9ca0b0",  # canonical overlay0
            "edge": "#dcdfe8",
            "edge-hi": "#bcc0cc",  # canonical surface1
            "accent": "#8839ef",  # canonical mauve
            "success": "#2e7a1e",  # canonical green #40a02b: 2.96:1 (< 4)
            "warning": "#8f6203",  # canonical yellow #df8e1d: 2.31:1 (< 4)
            "danger": "#d20f39",  # canonical red — clears at 4.80:1
            "string": "#2e7a1e",
            "signal": "#1e66f5",  # canonical blue
            "label": "#5265d8",  # canonical lavender #7287fd: 2.81:1 (< 4)
            "tint-danger": "#f6dfe1",
            "tint-select": "#e4ecf9",
            "tint-select-hi": "#d5e2f6",
            "tint-attach": "#e3ecfb",
            "tint-attach-hi": "#c8daf6",
        },
    ),
    # Tokyo Night — github.com/tokyo-night. The published night variant
    # supplies most of the ramp verbatim: bg_dark #16161e is `sunken`,
    # fg_gutter #3b4261 is `edge-hi`, comment #565f89 (2.76:1) is `faint`,
    # and the state row is the canonical red/yellow/green/blue/magenta.
    # Blue is the theme's identity, so it takes accent; cyan #7dcfff is the
    # link/file `signal`. `dim` is canonical dark5 #737aa2, which clears
    # the floor here (3.78:1 on surface) though not on storm's ground.
    ThemeSpec(
        name="tokyo-night",
        label="Tokyo Night",
        description="Neon-lit indigo night, the VS Code favourite",
        dark=True,
        tokens={
            "bg": "#1a1b26",  # canonical bg
            "surface": "#20222f",
            "raised": "#262939",
            "overlay": "#2f334d",
            "sunken": "#16161e",  # canonical bg_dark
            "fg": "#c0caf5",  # canonical fg
            "muted": "#a9b1d6",  # canonical fg_dark
            "dim": "#737aa2",  # canonical dark5
            "faint": "#565f89",  # canonical comment
            "edge": "#2c3045",
            "edge-hi": "#3b4261",  # canonical fg_gutter
            "accent": "#7aa2f7",  # canonical blue — the theme's identity
            "success": "#9ece6a",  # canonical green
            "warning": "#e0af68",  # canonical yellow
            "danger": "#f7768e",  # canonical red
            "string": "#9ece6a",
            "signal": "#7dcfff",  # canonical cyan
            "label": "#bb9af7",  # canonical magenta
            "tint-danger": "#2f222f",
            "tint-select": "#1f2837",  # blue cast — selection wears the accent
            "tint-select-hi": "#253044",
            "tint-attach": "#1d2f42",
            "tint-attach-hi": "#2c4a67",
        },
    ),
    # Tokyo Night Storm — the same city, one shade before dark: canonical
    # storm bg #24283b with the identical accent row (the variants share
    # their hues by design). The lighter ground costs contrast everywhere,
    # so the text ladder re-solves: dark5 #737aa2 drops to 3.34:1 on the
    # surface step (< 3.4) and lifts to #7e86b0; comment #565f89 still
    # reads as `faint`. Tints are brighter than night's to stay visible
    # casts against the lighter ground.
    ThemeSpec(
        name="tokyo-night-storm",
        label="Tokyo Night Storm",
        description="Tokyo Night's softer storm-blue ground",
        dark=True,
        tokens={
            "bg": "#24283b",  # canonical storm bg
            "surface": "#2b3048",
            "raised": "#333955",
            "overlay": "#3d4466",
            "sunken": "#1d2032",
            "fg": "#c0caf5",  # canonical fg
            "muted": "#a9b1d6",  # canonical fg_dark
            "dim": "#7e86b0",  # canonical dark5 #737aa2: 3.34:1 on surface (< 3.4)
            "faint": "#565f89",  # canonical comment
            "edge": "#363c58",
            "edge-hi": "#454c70",
            "accent": "#7aa2f7",
            "success": "#9ece6a",
            "warning": "#e0af68",
            "danger": "#f7768e",
            "string": "#9ece6a",
            "signal": "#7dcfff",
            "label": "#bb9af7",
            "tint-danger": "#3a2c3d",
            "tint-select": "#283349",
            "tint-select-hi": "#2e3c55",
            "tint-attach": "#263a52",
            "tint-attach-hi": "#365478",
        },
    ),
    # Gruvbox (dark, medium) — github.com/morhetz/gruvbox. Ground and text
    # ladders are canonical throughout: bg0 #282828, bg0_h #1d2021 as
    # `sunken`, bg1/bg2 as edge/edge-hi, fg1/fg3/gray/bg3+ as the text
    # rungs. Yellow is gruvbox's face (the logo, the cursor line numbers),
    # so it takes accent and bright orange takes warning — they sit far
    # enough apart (8.7:1 vs 5.8:1, yellow vs orange hue) to never merge.
    # Bright red #fb4934 measures 3.97:1 on the surface step — a hair under
    # the 4:1 state floor — so danger is #fb4f3e, the same red warmed one
    # step (4.42:1 bg, 4.09:1 surface), still unmistakably gruvbox red.
    ThemeSpec(
        name="gruvbox",
        label="Gruvbox",
        description="Retro groove — warm bread-crust browns and hard yellows",
        dark=True,
        tokens={
            "bg": "#282828",  # canonical bg0
            "surface": "#302d2c",
            "raised": "#3c3836",  # canonical bg1
            "overlay": "#46403d",
            "sunken": "#1d2021",  # canonical bg0_h (hard contrast bg)
            "fg": "#ebdbb2",  # canonical fg1
            "muted": "#bdae93",  # canonical fg3
            "dim": "#928374",  # canonical gray
            "faint": "#665c54",  # canonical bg3
            "edge": "#3c3836",  # canonical bg1
            "edge-hi": "#504945",  # canonical bg2
            "accent": "#fabd2f",  # canonical bright yellow — the gruvbox face
            "success": "#b8bb26",  # canonical bright green
            "warning": "#fe8019",  # canonical bright orange
            # Canonical bright red #fb4934 is 3.97:1 on the surface step,
            # just under the 4:1 floor; warmed one step to keep the hue.
            "danger": "#fb4f3e",
            "string": "#b8bb26",
            "signal": "#83a598",  # canonical bright blue
            "label": "#d3869b",  # canonical bright purple
            "tint-danger": "#3b2723",
            "tint-select": "#32321c",  # olive cast — gruvbox green is yellow-green
            "tint-select-hi": "#3b3b21",
            "tint-attach": "#28353a",
            "tint-attach-hi": "#3a5158",
        },
    ),
    # Nord — nordtheme.com. Polar Night supplies the ground (nord0 base,
    # nord1/nord2 as raised/overlay, nord3 as edge-hi) and Frost supplies
    # the cool accents (nord8 cyan-ice as THE accent, nord9 as signal).
    # Nord's one real casualty is Aurora red: nord11 #bf616a is 3.05:1 on
    # nord0 — Nord was designed for syntax, not one-word states — so danger
    # lifts along the same desaturated rose to #e0838e (4.64:1/4.30:1).
    # `muted` is a half-step below Snow Storm's nord4 (9.25:1) so the
    # secondary rung stays clearly secondary to fg (10.84:1).
    ThemeSpec(
        name="nord",
        label="Nord",
        description="Arctic bluish calm — frost accents on polar night",
        dark=True,
        tokens={
            "bg": "#2e3440",  # canonical nord0
            "surface": "#333947",
            "raised": "#3b4252",  # canonical nord1
            "overlay": "#434c5e",  # canonical nord2
            "sunken": "#272c36",
            "fg": "#eceff4",  # canonical nord6
            "muted": "#c8cfdd",  # nord4 #d8dee9 is 9.25:1 — too close to fg's rung
            "dim": "#8b96ab",
            "faint": "#616e88",
            "edge": "#3b4252",  # canonical nord1
            "edge-hi": "#4c566a",  # canonical nord3
            "accent": "#88c0d0",  # canonical nord8 — Nord's primary frost
            "success": "#a3be8c",  # canonical nord14
            "warning": "#ebcb8b",  # canonical nord13
            # Canonical nord11 #bf616a measures 3.05:1 on nord0 (< 4);
            # lifted along the same muted rose.
            "danger": "#e0838e",
            "string": "#a3be8c",
            "signal": "#81a1c1",  # canonical nord9
            "label": "#b48ead",  # canonical nord15
            "tint-danger": "#3d323c",
            "tint-select": "#2e3d40",  # frost-teal cast
            "tint-select-hi": "#354a4e",
            "tint-attach": "#2b3b4e",
            "tint-attach-hi": "#3d5673",
        },
    ),
    # One Dark — Atom's default (atom/one-dark-syntax). The accent row is
    # canonical: blue #61afef (Atom's identity hue) as accent, cyan as
    # signal, purple as label, and the red/green/yellow triple verbatim.
    # The text ladder needs the solve: canonical fg #abb2bf is 6.57:1 on
    # the bg — under the 7:1 body floor — so fg lifts along the same cool
    # gray to #bec4d0, and muted/dim step down from it; canonical comment
    # #5c6370 (2.32:1) is exactly the `faint` rung.
    ThemeSpec(
        name="one-dark",
        label="One Dark",
        description="Atom's cool gray-blue standard, steady and unflashy",
        dark=True,
        tokens={
            "bg": "#282c34",  # canonical bg
            "surface": "#2d3139",
            "raised": "#333842",
            "overlay": "#3b414d",
            "sunken": "#21252b",
            # Canonical fg #abb2bf is 6.57:1 on bg (< 7); lifted on-hue.
            "fg": "#bec4d0",
            "muted": "#9aa2b0",
            "dim": "#7d8695",
            "faint": "#5c6370",  # canonical comment
            "edge": "#3a3f4b",
            "edge-hi": "#4b5263",  # canonical visual/gutter gray
            "accent": "#61afef",  # canonical blue — Atom's identity
            "success": "#98c379",  # canonical green
            "warning": "#e5c07b",  # canonical yellow
            "danger": "#e06c75",  # canonical red
            "string": "#98c379",
            "signal": "#56b6c2",  # canonical cyan
            "label": "#c678dd",  # canonical purple
            "tint-danger": "#362a31",
            "tint-select": "#28333c",  # blue cast
            "tint-select-hi": "#2e3c47",
            "tint-attach": "#263647",
            "tint-attach-hi": "#375069",
        },
    ),
    # Solarized Dark — ethanschoonover.com/solarized. Solarized's precise
    # CIELAB symmetry predates UI-state contrast floors, and it shows: on
    # base03 the canonical red #dc322f is 3.25:1, orange #cb4b16 3.26:1,
    # magenta 3.30:1, violet 3.43:1, and blue #268bd2 drops to 3.79:1 on
    # the surface step. The passing hues stay canonical (yellow, green,
    # cyan — cyan is THE solarized accent); the failing ones lift along
    # their own hue by the minimum that clears both grounds: red →
    # #ea5b52, blue → #3a9ae0, violet → #8489d4. The text ladder keeps
    # base0 #839496 as `muted` a hair brightened (#87999b, base0 is
    # 4.41:1 on surface) and base01 as `faint`; body fg is base1 pushed
    # one step (canonical #93a1a1 is 5.61:1, under the 7:1 floor).
    ThemeSpec(
        name="solarized-dark",
        label="Solarized Dark",
        description="The deep teal lab classic, sixteen colors of discipline",
        dark=True,
        tokens={
            "bg": "#002b36",  # canonical base03
            "surface": "#00313d",
            "raised": "#073642",  # canonical base02
            "overlay": "#0e3d4a",
            "sunken": "#00252e",
            # Canonical base1 #93a1a1 is 5.61:1 on base03 (< 7); lifted on-hue.
            "fg": "#b1bcbc",
            # Canonical base0 #839496 is 4.41:1 on the surface step (< 4.5).
            "muted": "#87999b",
            "dim": "#69838c",
            "faint": "#586e75",  # canonical base01
            "edge": "#0b3844",
            "edge-hi": "#16454f",
            "accent": "#2aa198",  # canonical cyan — solarized's signature
            "success": "#859900",  # canonical green
            "warning": "#b58900",  # canonical yellow
            # Canonical red #dc322f is 3.25:1 on base03 (< 4); lifted on-hue.
            "danger": "#ea5b52",
            "string": "#859900",
            # Canonical blue #268bd2 is 3.79:1 on the surface step (< 4).
            "signal": "#3a9ae0",
            # Canonical violet #6c71c4 is 3.43:1 on base03 (< 4).
            "label": "#8489d4",
            # A maroon-leaning cast: on solarized's cool teal ground, a
            # neutral-red mix reads as plain gray, so the red channel leads.
            "tint-danger": "#3a2c32",
            "tint-select": "#0a3538",  # cyan cast
            "tint-select-hi": "#0f4044",
            "tint-attach": "#0b3547",
            "tint-attach-hi": "#134e64",
        },
    ),
]
