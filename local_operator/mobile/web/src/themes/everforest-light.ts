import type { ThemeDefinition } from "./palette-contract";

/**
 * Everforest Light.
 *
 * The official light counterpart, medium contrast. The accent row is the same
 * palette as the dark variant re-solved for paper, which is what makes these
 * two variants of one scheme rather than two themes.
 *
 * Two separate problems stack here. First, every published light accent is
 * built for soft contrast and lands at 2.1–3.1:1 on its own bg0, so each is
 * darkened along its own hue — this is the family-wide light solve, not a
 * fault in everforest. Second, the ground ladder: everforest's published
 * light backgrounds are close enough together that bg_dim → bg0 measured
 * ΔE00 1.97 and bg0 → bg1 only 1.06, both under the ~2 that the ramp comment
 * in `local-operator.ts` records as the threshold where a step stops being
 * visible. The ladder below is widened and run warmer as it runs deeper, the
 * same chroma lever that file describes, and now steps 3.00 / 2.38 / 2.96.
 */
export const everforestLight: ThemeDefinition = {
	id: "everforestLight",
	name: "Everforest Light",
	description: "The forest scheme on warm paper, soft and low-glare.",
	palette: {
		mode: "light",

		canvas: "#EAE4C8",
		surface: "#F4EDD4",
		elevated: "#FDF6E3",
		sunken: "#DCD7BC",

		// Upstream fg #5C6A72 is 5.18:1 on bg0, under the 7:1 floor.
		ink: "#394347",
		// Upstream grey2 #829181 is 3.08:1 on bg0.
		inkMuted: "#545F5A",
		// Upstream grey1 #939F91 is 2.56:1 on bg0.
		inkDim: "#546058",
		// Upstream grey0, the scheme's own inert-hint grey.
		inkDisabled: "#A6B0A0",

		hairline: "#DFDAC0",
		// Upstream grey2 #829181 is 2.29:1 on `sunken`, the binding ground for
		// a light theme's boundaries — a mid grey has its easiest time against
		// the lightest ground, so the darkest one is the case that has to clear
		// the 3:1 structural floor.
		borderControl: "#6E7D6D",

		// Upstream blue #3A94C5 is 2.33:1 on `sunken`; darkened on-hue and
		// taken to 5.25:1 so `accentActive` still clears 4.5.
		accent: "#566201",
		accentHover: "#1B445A",
		accentActive: "#276283",
		accentWash: "#D9DDCE",
		onAccent: "#FDF6E3",

		// Upstream green #8DA101 is 2.00:1 on `sunken`.
		success: "#566201",
		successWash: "#DAD6B2",
		successBorder: "#566201",

		// Upstream yellow #DFA000 is 1.58:1 on `sunken`.
		warning: "#795700",
		warningWash: "#DFD6B4",
		warningBorder: "#795700",

		// Upstream red #F85552 is 2.26:1 on `sunken`.
		danger: "#B12724",
		dangerWash: "#E5D3B9",
		dangerBorder: "#B12724",

		// Upstream purple #DF69BA is 2.11:1 on `sunken`.
		info: "#235976",
		infoWash: "#E4D3C1",
		infoBorder: "#A7247E",

		overlayShadow: "0 12px 32px -12px rgb(92 106 114 / 0.28)",
		scrim: "rgb(60 68 66 / 0.45)",
	},
};
