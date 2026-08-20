import type { ThemeDefinition } from "./palette-contract";

/**
 * Tokyo Night.
 *
 * Carried from the upstream scheme: background 1A1B26, the raised blue-black,
 * foreground C0CAF5, comment 565F89, the blue and purple accents, and the
 * green, yellow, red and cyan that the scheme already names. The comment
 * colour does triple duty here as inkDisabled, hairline and — lightened —
 * the structural border.
 */
export const tokyoNight: ThemeDefinition = {
	id: "tokyoNight",
	name: "Tokyo Night",
	description: "Deep blue-purple night with a bright blue accent.",
	palette: {
		mode: "dark",

		canvas: "#1A1B26",
		surface: "#24283B",
		elevated: "#2F334D",
		// Upstream bg_dark is 16161E, which measures 1.05:1 against canvas. Two
		// levels deeper for a little more separation.
		sunken: "#14141B",

		ink: "#C0CAF5",
		inkMuted: "#A9B1D6",
		// The comment blue lightened to clear 4.5:1 on all four grounds; the
		// comment colour itself is inkDisabled.
		inkDim: "#979EBE",
		inkDisabled: "#565F89",

		hairline: "#3D4462",
		// The comment blue lightened again. The old theme used it at 30 percent
		// alpha for both the decorative rule and the input boundary, which put
		// every input at about 1.3:1.
		borderControl: "#757EA9",

		accent: "#7AA2F7",
		accentHover: "#9EBCFF",
		accentActive: "#6D8FDA",
		accentWash: "#262B3F",
		onAccent: "#1A1B26",

		// Upstream Tokyo Night green.
		success: "#9ECE6A",
		successWash: "#2A302E",
		successBorder: "#637D4B",

		// Upstream yellow.
		warning: "#E0AF68",
		warningWash: "#322D2E",
		warningBorder: "#8C704D",

		// Upstream red.
		danger: "#F7768E",
		dangerWash: "#352632",
		dangerBorder: "#AB5D70",

		// Upstream cyan, so this theme needs no invented informational hue.
		info: "#7DCFFF",
		infoWash: "#263140",
		infoBorder: "#507E9D",

		overlayShadow: "0 12px 32px -12px rgb(9 9 14 / 0.7)",
		scrim: "rgb(9 9 14 / 0.6)",
	},
};
