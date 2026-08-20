import type { ThemeDefinition } from "./palette-contract";

/**
 * Iceberg, light.
 *
 * The upstream Iceberg light scheme is the source: background E8E9EC, text
 * 33374C, blue 2D539E, green 668E3D, orange C57339, red CC517A and cyan
 * 3F83A6. The app's old file had already darkened several of these by hand
 * for contrast; now that the floors are measured, each one goes back to its
 * upstream value and is darkened only as far as the arithmetic requires.
 */
export const iceberg: ThemeDefinition = {
	id: "iceberg",
	name: "Iceberg",
	description: "Cool blue-grey paper with the Iceberg navy.",
	palette: {
		mode: "light",

		canvas: "#E8E9EC",
		// Upstream has one background and the old file invented F3F4F7 for paper.
		// Two more grounds are derived on the same cool ramp.
		surface: "#F2F3F6",
		// Canvas is upstream's own background, so the canvas/surface step is fixed
		// at ΔE00 2.11 and surface cannot come down without collapsing it. The only
		// room for a visible surface/elevated step is therefore above surface:
		// FAFAFC read 1.58 apart from it, FDFDFF reads 2.15 on the same +2 blue
		// tint. That is the whole headroom a light ramp has under white.
		elevated: "#FDFDFF",
		// A light ramp has very little room below its page ground: canvas is already
		// a mid-light grey, and every step down costs tertiary-ink headroom. This
		// sits 1.07:1 under canvas, which is what caps inkDim below.
		sunken: "#E1E2E7",

		// Iceberg's own text colour. The old file darkened it to 262A3F for
		// contrast, which is no longer necessary — this measures 9:1 on the darkest
		// ground and 11.2:1 on the lightest.
		ink: "#33374C",
		inkMuted: "#4A4E64",
		// Upstream sub colour, carried verbatim; it clears 4.87:1 on sunken, which
		// is the binding ground for a dark ink in a light theme.
		inkDim: "#5B5F74",
		inkDisabled: "#A0A4B8",

		hairline: "#CBCCD2",
		// Derived. The old theme bounded inputs at 15 percent black, about 1.4:1.
		borderControl: "#787D97",

		accent: "#2D539E",
		accentHover: "#1E3A7D",
		accentActive: "#162E63",
		accentWash: "#D5DAE4",
		onAccent: "#F2F5F9",

		// Upstream green 668E3D, darkened to clear 4.5:1 on the light grounds.
		success: "#4C692D",
		successWash: "#DBE0DB",
		successBorder: "#718B57",

		// Upstream orange C57339, darkened for the same reason.
		warning: "#8D5229",
		warningWash: "#E5DDDA",
		warningBorder: "#B87142",

		// Upstream red CC517A, darkened for the same reason. The old file's own
		// error.dark B32D5E was heading the same way by hand.
		danger: "#AB325B",
		dangerWash: "#E5DAE1",
		dangerBorder: "#C76286",

		// Upstream cyan, darkened. Kept as a cyan rather than a second navy so an
		// info callout does not read as a primary one.
		info: "#316682",
		infoWash: "#D7DFE5",
		infoBorder: "#5289A8",

		overlayShadow: "0 12px 32px -12px rgb(38 42 63 / 0.22)",
		scrim: "rgb(38 42 63 / 0.35)",
	},
};
