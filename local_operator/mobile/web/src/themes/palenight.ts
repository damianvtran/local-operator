import type { ThemeDefinition } from "./palette-contract";

/**
 * Palenight — Material Theme's indigo variant.
 *
 * Values are from the Material Theme UI palette reference
 * (material-theme.com/docs/reference/color-palette, Palenight): background
 * 292D3E, contrast 202331, variables EEFFFF, foreground A6ACCD, comment/gray
 * 676E95, and the green/yellow/blue/red/purple/orange/cyan row that the
 * Material family shares.
 *
 * The accent row ships unmodified — that saturated row on indigo is the whole
 * appeal — with one exception noted at `danger`. The ink ladder is where the
 * work is: upstream's `foreground` A6ACCD is a secondary weight by measurement
 * (4.98:1), not a body ink, so `ink` takes upstream's own brighter variables
 * colour EEFFFF and A6ACCD drops to `inkMuted`, the rung it actually reads as.
 *
 * Upstream's named accent is AB47BC, which measures 2.83:1 on its own
 * background — a badge and underline colour, not text. Purple C792EA (the
 * keywords hue) carries `accent` instead. That is a reassignment inside the
 * palette, not a new hue.
 */
export const palenight: ThemeDefinition = {
	id: "palenight",
	name: "Palenight",
	description: "Material's indigo evening, saturated hues on violet-slate.",
	palette: {
		mode: "dark",

		canvas: "#292D3E",
		// Upstream's own raised grounds (buttons 303348, second background
		// 34324A) are not a consistent ladder — they cross each other in
		// lightness — so these two are derived on the canvas hue at ΔE00 2.26
		// and 2.29.
		surface: "#2F3446",
		elevated: "#353B4E",
		sunken: "#232736",

		ink: "#EEFFFF",
		inkMuted: "#A6ACCD",
		// Upstream comment 676E95 is 2.25:1 on `elevated` — far under the 4.5
		// tertiary floor, and it is the inactive tone anyway, so it lands on
		// inkDisabled. inkDim is solved between it and the foreground.
		inkDim: "#A0A4BD",
		inkDisabled: "#676E95",

		hairline: "#3D4155",
		borderControl: "#8287A8",

		accent: "#C792EA",
		accentHover: "#D9B4F1",
		accentActive: "#B874E4",
		accentWash: "#2F3145",
		onAccent: "#292D3E",

		success: "#C3E88D",
		successWash: "#2D3240",
		successBorder: "#AADF5E",

		warning: "#F78C6C",
		warningWash: "#303040",
		warningBorder: "#F5724B",

		// Canonical red F07178 is 3.89:1 on `elevated`, under the 4.5 floor —
		// the one accent in this row that misses. Lifted along the same coral
		// by the minimum that clears (4.51:1 across all four grounds).
		danger: "#F2858B",
		dangerWash: "#303041",
		dangerBorder: "#EF6A72",

		// Upstream's cyan, so this palette needs no invented informational hue.
		info: "#89DDFF",
		infoWash: "#2C3345",
		infoBorder: "#50CCFF",

		overlayShadow: "0 12px 32px -12px rgb(32 35 49 / 0.65)",
		scrim: "rgb(32 35 49 / 0.6)",
	},
};
