import type { ThemeDefinition } from "./palette-contract";

/**
 * Ayu Light.
 *
 * The `light` variant, same published build (ayu@9.0.0, `dist/light.js`).
 *
 * This is the hardest solve in the batch. Ayu light is a near-white paper
 * (editor.bg is #FCFCFC) carrying accents tuned for that paper, so measured
 * against the DARKEST ground — which is the binding one for a light theme —
 * every published accent lands between 1.63:1 and 2.62:1. All seven are
 * darkened along their own hue.
 *
 * The ground ladder is the second problem, and near-white is where the lever
 * described in `local-operator.ts` runs out: that ramp buys separation with
 * warmth at fixed L*, but ayu's ramp is a cool neutral with very little
 * chroma to spend. Taken from ayu's own four values the steps measured ΔE00
 * 1.80 / 1.33 / 0.40 — the top step invisible. The ladder below is deepened
 * at the bottom instead, and now steps 2.42 / 2.51 / 3.39. The cost is real:
 * every ink is measured against a darker `sunken` than ayu ships, which is
 * what drives the accent lifts above being as large as they are.
 */
export const ayuLight: ThemeDefinition = {
	id: "ayuLight",
	name: "Ayu Light",
	description: "Ayu on near-white paper, crisp with deepened accents.",
	palette: {
		mode: "light",

		canvas: "#E7EEF5",
		surface: "#F2F8FC",
		elevated: "#FFFFFF",
		sunken: "#DBE4EB",

		// Upstream editor.fg #5C6166 is 4.86:1 on `sunken`, under the 7:1
		// floor; darkened on-hue.
		ink: "#45494D",
		// Upstream comment #787B80 is 3.30:1 on `sunken`.
		inkMuted: "#62656A",
		// Upstream ui.fg #828E9F is 2.58:1 on `sunken`.
		inkDim: "#5B6676",
		inkDisabled: "#9AA3AF",

		hairline: "#DCE3E8",
		// Upstream ui.fg #828E9F is 2.58:1 on `sunken`; a mid grey has its
		// easiest time against the lightest ground, so the darkest one is the
		// case that has to clear the 3:1 structural floor.
		borderControl: "#758295",

		// Upstream entity #399EE6 is 2.26:1 on `sunken`; darkened on-hue and
		// taken to 5.28:1 so `accentActive` still clears 4.5.
		accent: "#125F94",
		accentHover: "#0E4B74",
		accentActive: "#1468A3",
		accentWash: "#D5E2EC",
		onAccent: "#FFFFFF",

		// Upstream string #86B300 is 1.93:1 on `sunken`.
		success: "#536E00",
		successWash: "#DBE4E1",
		successBorder: "#536E00",

		// Upstream func #F2A300 is 1.63:1 on `sunken`.
		warning: "#895C00",
		warningWash: "#E0E3E3",
		warningBorder: "#895C00",

		// Upstream markup #F07171 is 2.23:1 on `sunken`.
		danger: "#B53838",
		dangerWash: "#E4E1E8",
		dangerBorder: "#B53838",

		// Upstream constant #A37ACC is 2.62:1 on `sunken`.
		info: "#8149B9",
		infoWash: "#DFE1F0",
		infoBorder: "#8149B9",

		overlayShadow: "0 12px 32px -12px rgb(92 97 102 / 0.25)",
		scrim: "rgb(69 73 77 / 0.45)",
	},
};
