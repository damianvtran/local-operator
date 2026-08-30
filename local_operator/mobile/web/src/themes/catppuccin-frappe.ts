import type { ThemeDefinition } from "./palette-contract";

/**
 * Catppuccin Frappé — the third flavour, from catppuccin/palette v1.8.0.
 *
 * The palette supplies its own ground ramp, so the four grounds are upstream's
 * where they fit: base 303446 is the page and mantle 292C3C sits below it.
 * Text C6D0F5, subtext1 B5BFE2, subtext0 A5ADCE, overlay0 737994 and overlay2
 * 949CBB carry the ink weights, and the accent row is mauve/green/peach/red/
 * blue — the same role assignment `catppuccinMacchiato` uses, so the flavours
 * differ in ground temperature rather than in meaning.
 *
 * Frappé is the lightest of the three dark flavours, which is what makes it
 * the tightest: the binding ground for every ink is `elevated`, and four
 * upstream values land just under their floor against it. Each is lifted along
 * its own hue by the minimum that clears, and the miss is recorded per role.
 */
export const catppuccinFrappe: ThemeDefinition = {
	id: "catppuccinFrappe",
	name: "Catppuccin Frappé",
	description: "The middle flavour — warm slate under the Catppuccin pastels.",
	palette: {
		mode: "dark",

		canvas: "#303446",
		// Upstream surface0 414559 is the natural next step but measures ΔE00
		// 5.65 from base, which would put `elevated` past surface1 and cost
		// every pastel its headroom. These two are seated between base and
		// surface0 instead, at ΔE00 2.29 and 2.31.
		surface: "#363B4E",
		elevated: "#3D4255",
		sunken: "#2A2D3E",

		// Canonical text C6D0F5 is 6.52:1 on `elevated`, under the 7:1 body
		// floor. Lifted along the same lavender-white.
		ink: "#D0D8F7",
		inkMuted: "#B5BFE2",
		// Canonical subtext0 A5ADCE is 4.49:1 on `elevated` — short of 4.5 by
		// a hundredth, and the floor is the floor.
		inkDim: "#A6AECE",
		inkDisabled: "#737994",

		hairline: "#464B60",
		// Upstream overlay2, which clears the 3:1 structural floor with room
		// (3.67:1) where overlay0 — the inactive tone — could not.
		borderControl: "#949CBB",

		accent: "#CA9EE6",
		accentHover: "#DCBEEE",
		accentActive: "#BA82DF",
		accentWash: "#37394D",
		onAccent: "#303446",

		success: "#A6D189",
		successWash: "#343948",
		successBorder: "#8BC365",

		warning: "#EF9F76",
		warningWash: "#383848",
		warningBorder: "#EB8855",

		// Canonical red E78284 is 3.76:1 on `elevated` (< 4.5) — the widest
		// miss in this flavour, and the reason frappé needs a solve where
		// macchiato does not. Lifted along the same rose.
		danger: "#EB989A",
		dangerWash: "#38384A",
		dangerBorder: "#E67F82",

		// Canonical blue 8CAAEE is 4.31:1 on `elevated` (< 4.5).
		info: "#91AEEF",
		infoWash: "#353B4F",
		infoBorder: "#769AEB",

		overlayShadow: "0 12px 32px -12px rgb(35 38 52 / 0.65)",
		scrim: "rgb(35 38 52 / 0.6)",
	},
};
