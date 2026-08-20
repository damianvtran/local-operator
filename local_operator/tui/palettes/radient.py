"""The Radient brand ramp: the radient.com design language as a TUI theme.

Where the neon family SOLVES its vibe (source aesthetics supply a hue or
two and the ramp is scaled to fit), this one is a direct port: every token
is a published value from the Radient design system in the radient-site
repository (``src/index.css`` — the ``@theme`` block is the single source
of colour truth; ``docs/design-language.md`` there is the rationale), with
the mapping decisions called out per token below. If the site's palette
shifts, shift these to match rather than re-solving — fidelity to the
brand kit is the point of the theme.

The architecture mirrors the brand dark ramp's: a blue-tinted near-black
ground (OKLCH hue 255, the measured hue of the brand blue, so neutrals and
the accent read as one family), elevation as a background step (never a
shadow), hairline borders as the primary structural device, and exactly one
saturated signal blue spent sparingly — primary action, active state, one
emphasised element.

Two deliberate departures from the tron neighbour this theme sits beside:

- Danger is the brand's warm RED, not tron's Rinzler orange. The Radient
  kit ships a real danger hue with its own wash, so the failed-row band is
  the kit's ``danger-wash`` ground, not an invented tint.
- There is no second accent hue. The kit spends one blue, so ``signal``
  (links, file paths) is the accent's hover step — brighter, same hue —
  and ``label`` (tips, skill labels) is the kit's muted text step. A
  violet or cyan here would be a visitor from another theme.
"""

from __future__ import annotations

from local_operator.tui.theme import ThemeSpec

PALETTES: list[ThemeSpec] = [
    ThemeSpec(
        name="radient",
        label="Radient",
        description="One signal blue on a blue-tinted near-black (the Radient kit)",
        dark=True,
        tokens={
            # The site's ground ramp is four steps (void/canvas/surface/
            # elevated); the TUI speaks five. Mapping: void -> sunken (the
            # status band), canvas -> bg, surface -> surface, elevated ->
            # raised. `overlay` (dialogs, active selection) is the ONE
            # interpolated value in this port: the midpoint of the kit's
            # hairline/hairline-strong pair (#252b34/#343c46). The kit runs
            # out of ramp at four ground steps, and the review caught that
            # taking hairline verbatim made overlay byte-identical to
            # `edge` — a dialog ground whose outline would be invisible.
            # The midpoint keeps the ladder monotonic (1.26:1 over raised)
            # and the border visible (1.12:1 against the hairline) while
            # staying inside the kit's own structural tones.
            "bg": "#090d13",  # canvas
            "surface": "#12171d",  # surface
            "raised": "#1c2229",  # elevated
            "overlay": "#2c333d",  # hairline/hairline-strong midpoint (interpolated)
            "sunken": "#030509",  # void
            # Text ramp is verbatim: fg, fg-muted, fg-dim. `faint` (inert
            # hints) is fg-disabled — the kit's quietest ink, exempt from AA
            # on the site for the same reason `faint` is sub-floor here.
            "fg": "#ebf0f5",
            "muted": "#b7bec8",  # fg-muted
            "dim": "#868f9a",  # fg-dim
            "faint": "#5c646e",  # fg-disabled
            "edge": "#252b34",  # hairline
            "edge-hi": "#343c46",  # hairline-strong
            # The one blue. Accent-hover is the same hue pushed brighter,
            # so it carries `signal` (links, paths) — a second hue would
            # break the kit's one-signal discipline.
            "accent": "#51a2f8",
            "signal": "#75baff",  # accent-hover
            # Meta labels (tips, skills) are quiet by design in the kit —
            # eyebrow/label text renders in fg-muted — so `label` follows
            # rather than importing a hue the site never uses.
            "label": "#b7bec8",  # fg-muted
            # States are the kit's semantic trio, verbatim; string rides
            # success as in the brand ramps.
            "success": "#4bc680",
            "warning": "#eebc4a",
            "danger": "#ef6661",
            "string": "#4bc680",
            # Tints are the kit's own washes. Failed row: danger-wash —
            # danger ink holds 5.35:1 on it. Selection: accent-wash at
            # rest, accent-muted on hover, so the selected row reads as the
            # brand's blue ground and hover is still a visible step up
            # (1.2:1 between the two).
            "tint-danger": "#321614",  # danger-wash
            "tint-select": "#101e2d",  # accent-wash
            "tint-select-hi": "#152d46",  # accent-muted
            # The attachment chip is a blue card by family convention; the
            # kit's accent wash/muted pair supplies both steps, keeping the
            # chip and the picker selection in the same blue family the
            # site uses for active grounds.
            "tint-attach": "#101e2d",  # accent-wash
            "tint-attach-hi": "#152d46",  # accent-muted
        },
    ),
]
