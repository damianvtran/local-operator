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

		canvas: "#FBF1C7",
		// Upstream defines light0_soft F2E5BC and light1 EBDBB2 below the page,
		// not above it, so the two raised grounds are derived. They separate by
		// shedding the cream's chroma as they rise (ΔE00 2.26 and 2.51) because
		// a ramp this close to white has no lightness left to spend.
		surface: "#FBF3D1",
		elevated: "#FCF5DB",
		sunken: "#FBEFBD",

		ink: "#3C3836",
		inkMuted: "#665C54",
		// Canonical dark4 7C6F64 measures 4.21:1 on `sunken`, under the 4.5
		// floor for a tertiary weight. Darkened minimally along its own warm
		// gray; dark4 itself still serves as the structural border below, where
		// the floor is 3:1 and it clears with room.
		inkDim: "#776A60",
		inkDisabled: "#A89984",

		hairline: "#DFD4B1",
		borderControl: "#7C6F64",

		accent: "#076678",
		accentHover: "#04414D",
		accentActive: "#022026",
		accentWash: "#F0EBC3",
		onAccent: "#FBF1C7",

		// Canonical faded_green 79740E is 4.21:1 on `sunken`; this is the
		// smallest darkening along the same olive that clears 4.5:1 on all four.
		success: "#746F0D",
		// A wash cannot be built by tinting this cream toward the olive: the two
		// sit at nearly the same luminance, so every mix of them stays under
		// 4.5:1 for the ink. This steps away in lightness while holding the
		// olive hue at low chroma, which is what buys the ratio.
		successWash: "#EFEFE4",
		successBorder: "#847E0F",

		// Canonical faded_yellow B57614 is 3.27:1 on `sunken` — the largest
		// miss in this palette. Darkened along the same amber.
		warning: "#956110",
		warningWash: "#F1EDE7",
		warningBorder: "#AA6E12",

		danger: "#9D0006",
		dangerWash: "#F8E9C0",
		dangerBorder: "#BC0007",

		// faded_orange. Kept distinct from `warning`'s darkened yellow so a
		// notice and a caution do not read as the same amber.
		info: "#AF3A03",
		infoWash: "#F8E9BE",
		infoBorder: "#C94303",

		overlayShadow: "0 12px 32px -12px rgb(60 56 54 / 0.22)",
		scrim: "rgb(60 56 54 / 0.35)",
	},
};
