import type { ThemeDefinition } from "./palette-contract";

/**
 * Duskfox.
 *
 * The `duskfox` variant of the same plugin, from
 * `lua/nightfox/palette/duskfox.lua`.
 *
 * Duskfox is the violet-ground member of the family, which is what keeps it
 * clear of nightfox's blue-slate despite sharing the palette's structure. Its
 * grounds are already four separated steps (ΔE00 4.14 / 3.80 / 3.66) and its
 * ink ramp is the strongest in this batch — fg1 reads 9.03:1 on the lightest
 * ground — so only the three roles noted below move.
 */
export const duskfox: ThemeDefinition = {
	id: "duskfox",
	name: "Duskfox",
	description: "Violet-ground dusk, the rose-tinted end of the fox family.",
	palette: {
		mode: "dark",

		canvas: "#232136",
		surface: "#2D2A45",
		elevated: "#373354",
		sunken: "#191726",

		// Upstream fg1 (9.03:1 on the lightest ground).
		ink: "#E0DEF4",
		// Upstream fg2.
		inkMuted: "#CDCBE0",
		// Upstream comment #817C9C is 3.00:1 on `elevated`, under the 4.5
		// floor; lifted on-hue.
		inkDim: "#A9A4C4",
		// Upstream fg3 — 2.31:1 on `elevated`, the scheme's own inert ink.
		inkDisabled: "#6E6A86",

		hairline: "#433C59",
		// Upstream bg4 #4B4673 is 1.37:1 on `elevated`; it is a ground colour
		// upstream, not a boundary. Lifted along the same violet.
		borderControl: "#887BA3",

		// Upstream blue #569FBA is 4.00:1 on `elevated`, under the 4.5 floor;
		// lifted on-hue. How far to lift is a trade-off rather than a floor:
		// duskfox's blue and cyan are only ΔE00 15.3 apart upstream, and every
		// step of lift walks this role toward `info`. Lifting to 5.23:1 (which
		// buys a comfortable `accentActive`) closed that gap to 8.5. This sits
		// at 4.91:1 — still clear of the floor, `accentActive` still clears at
		// 4.56:1 — and holds 10.1 from `info`.
		accent: "#72AFC6",
		accentHover: "#8BBDD0",
		accentActive: "#68A9C2",
		accentWash: "#2E354A",
		onAccent: "#191726",

		// Upstream green.
		success: "#A3BE8C",
		successWash: "#31323F",
		successBorder: "#A3BE8C",

		// Upstream yellow.
		warning: "#F6C177",
		warningWash: "#38313C",
		warningBorder: "#F6C177",

		// Upstream red #EB6F92 is 4.09:1 on `elevated`; lifted on-hue.
		danger: "#ED7C9C",
		dangerWash: "#3C2C43",
		dangerBorder: "#ED7C9C",

		// Upstream cyan.
		info: "#9CCFD8",
		infoWash: "#323649",
		infoBorder: "#9CCFD8",

		overlayShadow: "0 12px 32px -12px rgb(14 12 22 / 0.75)",
		scrim: "rgb(14 12 22 / 0.62)",
	},
};
