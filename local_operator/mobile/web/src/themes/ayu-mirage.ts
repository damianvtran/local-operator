import type { ThemeDefinition } from "./palette-contract";

/**
 * Ayu Mirage.
 *
 * The `mirage` variant, from the same published build (ayu@9.0.0,
 * `dist/mirage.js`). Mirage is ayu's mid-dark ground, and upstream derives its
 * accent row from the dark one by lightening, so the family relationship holds
 * with no intervention here: string, func and constant all ship canonical and
 * are recognisably the dark variant's hues.
 *
 * Mirage's ground is lighter than ayu-dark's, which costs contrast on the two
 * warm accents. `ink` and `danger` are the roles that pay for it.
 */
export const ayuMirage: ThemeDefinition = {
	id: "ayuMirage",
	name: "Ayu Mirage",
	description: "Ayu's slate-blue middle ground, softer than the dark.",
	palette: {
		mode: "dark",

		canvas: "#242936",
		surface: "#2E3544",
		elevated: "#39404F",
		sunken: "#191D27",

		// Upstream editor.fg #CCCAC2 is 6.33:1 on `elevated`, under the 7:1
		// floor; lifted on-hue.
		ink: "#D6D5CE",
		// Ayu's comment #B8CFE6 at 50% alpha upstream; the flattened value.
		inkMuted: "#B8CFE6",
		inkDim: "#9DAEC2",
		// Upstream ui.fg — 2.40:1 on `elevated`, hence the disabled role.
		inkDisabled: "#707A8C",

		hairline: "#333B4A",
		// A divider at this luminance measures about 1.2:1 on `elevated`;
		// lifted along the same slate-blue to clear the 3:1 structural floor.
		borderControl: "#7E8BA4",

		// Upstream entity.
		accent: "#73D0FF",
		accentHover: "#97DCFF",
		accentActive: "#23B5FF",
		accentWash: "#2A3949",
		onAccent: "#1F2430",

		// Upstream string.
		success: "#D5FF80",
		successWash: "#2F363A",
		successBorder: "#D5FF80",

		// Upstream func.
		warning: "#FFD173",
		warningWash: "#38383B",
		warningBorder: "#FFD173",

		// Upstream markup #F28779 is 4.22:1 on `elevated`, under the 4.5
		// floor; lifted on-hue.
		danger: "#F39185",
		dangerWash: "#39333E",
		dangerBorder: "#F39185",

		// Upstream constant.
		info: "#DFBFFF",
		infoWash: "#3A3A4D",
		infoBorder: "#DFBFFF",

		overlayShadow: "0 12px 32px -12px rgb(16 19 26 / 0.72)",
		scrim: "rgb(16 19 26 / 0.6)",
	},
};
