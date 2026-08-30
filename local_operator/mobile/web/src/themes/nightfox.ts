import type { ThemeDefinition } from "./palette-contract";

/**
 * Nightfox.
 *
 * From EdenEast/nightfox.nvim. The palette is not in the repo README; it
 * lives in `lua/nightfox/palette/nightfox.lua`, which is the source these
 * values come from.
 *
 * The grounds are nightfox's own bg0–bg3 and need no solving — they are
 * already four separated steps (ΔE00 3.13 / 3.81 / 3.97), which is rare among
 * the schemes in this batch. The lifts are all on the ink and accent side,
 * where the lightest ground binds.
 *
 * This theme and `ayuDark` are both blue-blacks; their accent rows average
 * ΔE00 21.9 apart, which is what keeps them from converging.
 */
export const nightfox: ThemeDefinition = {
	id: "nightfox",
	name: "Nightfox",
	description: "Cool blue-slate night, even-toned and unhurried.",
	palette: {
		mode: "dark",

		canvas: "#192330",
		surface: "#212E3F",
		elevated: "#29394F",
		sunken: "#131A24",

		// Upstream fg1 (7.43:1 on the lightest ground).
		ink: "#CDCECF",
		// Upstream fg2.
		inkMuted: "#AEAFB0",
		// Upstream fg3 #71839B is 3.02:1 on `elevated`, under the 4.5 floor;
		// lifted on-hue.
		inkDim: "#93A2B5",
		// Upstream comment — 2.92:1 on `elevated`, the scheme's own inert ink.
		inkDisabled: "#738091",

		hairline: "#2B3B51",
		// Upstream bg4 #39506D is 1.42:1 on `elevated`: upstream uses it as a
		// conceal/border foreground against a darker ground than this one.
		borderControl: "#6483AD",

		// Upstream blue #719CD6 is 4.14:1 on `elevated`, under the 4.5 floor;
		// lifted on-hue and taken to 5.31:1 so `accentActive` still clears.
		accent: "#8FB1DF",
		accentHover: "#ABC4E7",
		accentActive: "#7AA3D9",
		accentWash: "#20304A",
		onAccent: "#131A24",

		// Upstream green.
		success: "#81B29A",
		successWash: "#26353D",
		successBorder: "#81B29A",

		// Upstream yellow.
		warning: "#DBC074",
		warningWash: "#2B3236",
		warningBorder: "#DBC074",

		// Upstream red #C94F6D is 2.69:1 on `elevated`; lifted on-hue.
		danger: "#DB899D",
		dangerWash: "#302F3D",
		dangerBorder: "#DB899D",

		// Upstream cyan.
		info: "#63CDCF",
		infoWash: "#213541",
		infoBorder: "#63CDCF",

		overlayShadow: "0 12px 32px -12px rgb(10 14 20 / 0.75)",
		scrim: "rgb(10 14 20 / 0.62)",
	},
};
