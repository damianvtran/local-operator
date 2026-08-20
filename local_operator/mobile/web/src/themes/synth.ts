import type { ThemeDefinition } from "./palette-contract";

/**
 * Synth.
 *
 * Retro synthwave. Carried: the two purple grounds, the darker purple the old
 * file used for tooltips, the pink trio, the neon blue and the bright orange.
 * The pink is the accent, the neon blue is the informational hue, and the
 * orange is the caution — three of the four semantic roles come straight from
 * the theme.
 */
export const synth: ThemeDefinition = {
	id: "synth",
	name: "Synth",
	description: "Synthwave: deep purple night with hot pink and neon blue.",
	palette: {
		mode: "dark",

		canvas: "#120720",
		surface: "#1B0A2F",
		elevated: "#2C1D43",
		sunken: "#06020D",

		// The old file borrowed the generic dark theme's F9FAFB and 9CA3AF for its
		// text, which is a cool grey ramp sitting on a purple one. The inks here are
		// tinted purple to match the grounds; same weights, same legibility.
		ink: "#F7F2FB",
		inkMuted: "#B9AACA",
		inkDim: "#9C8CAE",
		inkDisabled: "#6B5C7D",

		// The pink at a low tint over surface. It has to stay lighter than elevated,
		// or a divider inside a menu disappears — which is what the old sidebar
		// border did once it was flattened.
		hairline: "#5B1946",
		// Derived. The old theme bounded inputs with neon blue at 25 percent alpha,
		// about 1.5:1.
		borderControl: "#8A7BA0",

		accent: "#FF4081",
		accentHover: "#FF79B0",
		accentActive: "#F50057",
		accentWash: "#2E0E2C",
		// The old theme paired white with this pink, which measures 3.2:1. Ink on
		// the accent fill is the page ground instead, at 4.8:1.
		onAccent: "#120720",

		// Synth has no green. A mint, placed between the theme's pink and its neon
		// blue so it belongs to the same neon family.
		success: "#4DE8A8",
		successWash: "#192230",
		successBorder: "#32836B",

		// The bright orange the old file used for its ask highlights.
		warning: "#FFA500",
		warningWash: "#2E1A1C",
		warningBorder: "#945E0E",

		// The accent is already pink, so danger moves to red: a destructive action
		// must not read as the primary one. This is the smallest hue rotation that
		// separates them.
		danger: "#FF6B6B",
		dangerWash: "#2E1329",
		dangerBorder: "#A44551",

		// The theme's neon blue, an authentic informational hue.
		info: "#00E5FF",
		infoWash: "#10223B",
		infoBorder: "#08819B",

		overlayShadow: "0 12px 32px -12px rgb(6 2 13 / 0.75)",
		scrim: "rgb(6 2 13 / 0.65)",
	},
};
