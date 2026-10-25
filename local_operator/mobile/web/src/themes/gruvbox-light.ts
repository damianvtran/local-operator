import type { ThemeDefinition } from "./palette-contract";

/**
 * Gruvbox Light (medium contrast) — morhetz's own light mode.
 *
 * Values are from colors/gruvbox.vim: light0 FBF1C7 as the page, fg1/dark1
 * 3C3836 as the ink, dark3 665C54 and dark4 7C6F64 as the lower ink weights,
 * light4 A89984 as the inactive tone, and the FADED accent set — faded_blue
 * 076678, faded_green 79740E, faded_yellow B57614, faded_red 9D0006,
 * faded_orange AF3A03. The faded set is what gruvbox actually paints light
 * mode with; the bright set belongs to the dark ramp and is unreadable here.
 *
 * Gruvbox's light mode has no `accent` of its own — yellow is the scheme's
 * face on the dark ramp but is far too light to carry a state on cream — so
 * faded_blue takes the primary role, faded_orange takes `info`, and the
 * darkened yellow keeps `warning`, the role its meaning already fits.
 */
export const gruvboxLight: ThemeDefinition = {
	id: "gruvboxLight",
	name: "Gruvbox Light",
	description: "Retro groove by daylight, warm cream and faded ink.",
	palette: {
		mode: "light",

		// GROUNDS PORTED FROM THE DESKTOP (gruvbox-light.ts, local-operator-ui
		// src/renderer/src/shared/themes/palettes). The relay's borderless chrome stands
		// every field, the composer and the search pill on `elevated` over `canvas`
		// with no edge, so the step between them IS the boundary; this palette's
		// relay grounds measured under the contract's 4.0 L* elevated−canvas floor
		// (scripts/contrast-contract.mjs), the desktop's re-seat clears it. The
		// desktop moved the inks, hairline and tones on the SAME commit (a new
		// ground changes what every ink is measured against), and those values are
		// ported with it; a ground ported alone failed eight text/edge pairs.
		canvas: "#F8EEC4",
		// Upstream defines light0_soft F2E5BC and light1 EBDBB2 below the page,
		// not above it, so the two raised grounds are derived: they shed the
		// cream's chroma as they rise AND take a lightness step (the desktop's
		// re-seat), because chroma alone left no visible step for a borderless
		// control.
		surface: "#FDF6D8",
		elevated: "#FFFEF2",
		sunken: "#EFE2B9",

		ink: "#3C3836",
		inkMuted: "#4F4640",
		// Canonical dark4 7C6F64 measures 4.21:1 on `sunken`, under the 4.5
		// floor for a tertiary weight — and 4.33–4.34:1 on the two semantic
		// washes, which paint dim ink for real (the failed tool row's gauge,
		// the selected quick-pick's trailing text; D3). Darkened along its own
		// warm gray until all of them clear; dark4 itself still serves as the
		// structural border below, where the floor is 3:1 and it clears with
		// room.
		inkDim: "#685C53",
		inkDisabled: "#A89984",

		hairline: "#DDD2B0",
		borderControl: "#7C6F64",

		accent: "#076678",
		accentHover: "#04414D",
		accentActive: "#022026",
		accentWash: "#F0EBC3",
		onAccent: "#FBF1C7",

		// Canonical faded_green 79740E is 4.21:1 on `sunken`; this is the
		// smallest darkening along the same olive that clears 4.5:1 on all four.
		success: "#6D6701",
		// A wash cannot be built by tinting this cream toward the olive: the two
		// sit at nearly the same luminance, so every mix of them stays under
		// 4.5:1 for the ink. This steps away in lightness while holding the
		// olive hue at low chroma, which is what buys the ratio.
		successWash: "#EFEFE4",
		successBorder: "#847E0F",

		// Canonical faded_yellow B57614 is 3.27:1 on `sunken` — the largest
		// miss in this palette. Darkened along the same amber.
		warning: "#8D5A02",
		warningWash: "#F1EDE7",
		warningBorder: "#AA6E12",

		danger: "#9D0006",
		dangerWash: "#F8E9C0",
		dangerBorder: "#BC0007",

		// faded_orange. Kept distinct from `warning`'s darkened yellow so a
		// faded_aqua, not faded_orange. The orange sat only ΔE00 10.79 from
		// `danger`'s faded_red — both dark red-orange on cream, so an error
		// message and an informational path read as one colour (design
		// review D5). Aqua is equally canonical; darkened one step from
		// #427B58 (4.32:1) to clear this contract's four-ground floor.
		info: "#3B6F50",
		infoWash: "#F8EFD1",
		infoBorder: "#578C6C",

		overlayShadow: "0 12px 32px -12px rgb(60 56 54 / 0.22)",
		scrim: "rgb(60 56 54 / 0.35)",
	},
};
