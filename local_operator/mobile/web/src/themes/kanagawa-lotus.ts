import type { ThemeDefinition } from "./palette-contract";

/**
 * Kanagawa Lotus.
 *
 * The official light variant, resolved through the same `themes.lua`. Lotus
 * is unusual among light schemes in publishing a genuinely dark ink set, so
 * it needs less re-solving than the other light themes here: lotusInk1 and
 * lotusGray2 ship canonical, and lotusBlue4 and lotusViolet4 needed only the
 * headroom noted below rather than a hue re-solve.
 *
 * The ground ladder is the work. Lotus publishes lotusWhite0–5, but those
 * six values span a narrow band: taken straight, sunken → canvas measured
 * ΔE00 1.80, under the ~2 threshold the ramp comment in `local-operator.ts`
 * records as where a step stops being visible. The ladder below spreads the
 * published range and runs warmer as it runs deeper, and now steps
 * 2.87 / 2.65 / 2.32.
 */
export const kanagawaLotus: ThemeDefinition = {
	id: "kanagawaLotus",
	name: "Kanagawa Lotus",
	description: "The ink-wash palette on aged paper, warm and low-contrast.",
	palette: {
		mode: "light",

		canvas: "#DFD7A8",
		surface: "#E9E2B6",
		elevated: "#F2ECBC",
		sunken: "#D3CB9C",

		// Upstream lotusInk2 #43436C is 5.67:1 on `sunken`, under the 7:1
		// floor; darkened on-hue.
		ink: "#363658",
		// Upstream lotusInk1, canonical.
		inkMuted: "#545464",
		// Upstream lotusGray2 #716E61 is 3.12:1 on `sunken`, under the 4.5
		// floor; darkened on-hue.
		inkDim: "#57544F",
		// Upstream lotusGray3, the scheme's own comment colour.
		inkDisabled: "#8A8980",

		hairline: "#D3CB9E",
		// Upstream lotusGray2, canonical: 3.12:1 on `sunken`, which is the
		// binding ground for a light theme's boundaries.
		borderControl: "#716E61",

		// Upstream lotusBlue4 #4D699B is 3.36:1 on `sunken`; darkened on-hue
		// and taken to 5.25:1 so `accentActive` still clears 4.5.
		accent: "#384C70",
		accentHover: "#2C3C58",
		accentActive: "#3F557E",
		accentWash: "#CFCEA9",
		onAccent: "#F2ECBC",

		// Upstream lotusGreen #6F894E is 2.38:1 on `sunken`.
		success: "#4A5B34",
		successWash: "#D2CC9E",
		successBorder: "#4A5B34",

		// Upstream lotusYellow #77713F is 3.03:1 on `sunken`. Darkened AND
		// warmed: the straight on-hue solve sat only ΔE00 8.6 from this theme's
		// success olive, because lotus's yellow and green are neighbours
		// upstream (ΔE00 11.9) and darkening both pulls them together. This
		// reads 18.0 from success.
		warning: "#6D5115",
		warningWash: "#D5CB9B",
		warningBorder: "#6D5115",

		// Upstream lotusRed #C84053 is 2.97:1 on `sunken`.
		danger: "#9B2C3C",
		dangerWash: "#D9C89E",
		dangerBorder: "#9B2C3C",

		// Upstream lotusViolet4 #624C83 is 4.44:1 on `sunken`, a hair under the
		// 4.5 floor; darkened one step on-hue to 4.52:1. Lotus publishes no
		// aqua, so upstream's own `info` is this violet rather than the wave
		// variant's teal — the one place the two variants' accent rows part.
		info: "#614B81",
		infoWash: "#D3CAA4",
		infoBorder: "#614B81",

		overlayShadow: "0 12px 32px -12px rgb(84 84 100 / 0.3)",
		scrim: "rgb(67 67 108 / 0.45)",
	},
};
