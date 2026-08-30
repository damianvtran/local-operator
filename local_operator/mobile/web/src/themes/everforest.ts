import type { ThemeDefinition } from "./palette-contract";

/**
 * Everforest.
 *
 * Carried from sainnhe/everforest (`palette.md`, dark/medium): the bg_dim →
 * bg2 ground ladder, the green, yellow and aqua accents, and the grey ramp.
 *
 * The lift this theme needs is the fourth ground. Everforest is designed for
 * soft contrast, and the TUI port only measures ink against two grounds where
 * this contract measures against all four — so `elevated` (#3D484D, the
 * lightest) is what binds here. Canonical fg #D3C6AA measures 5.57:1 on it
 * and the greys land under 4.5:1, so the whole ink ramp is lifted on-hue;
 * `inkMuted` is taken past its 4.5 floor to 5.44:1 so that it stays visibly
 * a rung above `inkDim`, which sits at the floor.
 */
export const everforest: ThemeDefinition = {
	id: "everforest",
	name: "Everforest",
	description: "Warm green-tinted dark, built for long low-strain reading.",
	palette: {
		mode: "dark",

		canvas: "#2D353B",
		surface: "#343F44",
		elevated: "#3D484D",
		sunken: "#232A2E",

		// Upstream fg #D3C6AA is 5.57:1 on `elevated`, under the 7:1 floor.
		ink: "#E5DECD",
		// Upstream grey2 #9DA9A0 is 3.86:1 on `elevated`. Lifted past the 4.5
		// floor to 5.44:1 to keep a visible step above `inkDim`.
		inkMuted: "#BFC7C0",
		// Upstream grey1 #859289 is 2.90:1 on `elevated`.
		inkDim: "#ACB6AF",
		// Upstream grey0, the scheme's own inert-hint grey.
		inkDisabled: "#7A8478",

		hairline: "#475258",
		// Upstream bg4 #4F585E is 1.29:1 on `elevated` — a ground colour doing
		// a boundary's job. Lifted along the same cool grey to clear 3:1.
		borderControl: "#89949C",

		// Upstream blue #7FBBB3 is 4.33:1 on `elevated`, and a value solved
		// Design review D4: the first port took upstream's SYNTAX role names
		// literally — accent=blue, info=purple — which spent everforest's
		// signature green on a success glyph and painted the app's most
		// common ink pink. `accent` is the identity hue here, so it takes
		// the canonical green (4.70:1 worst ground) and `info` takes the
		// canonical blue-teal (5.20:1), matching the TUI palette.
		accent: "#A7C080",
		accentHover: "#B1D6D1",
		accentActive: "#85BFB6",
		accentWash: "#3A4448",
		onAccent: "#2D353B",

		// Upstream green.
		success: "#A7C080",
		successWash: "#3A4442",
		successBorder: "#A7C080",

		// Upstream yellow.
		warning: "#DBBC7F",
		warningWash: "#454842",
		warningBorder: "#DBBC7F",

		// Upstream red #E67E80 is 3.43:1 on `elevated`; lifted on-hue.
		danger: "#ECA0A1",
		dangerWash: "#434147",
		dangerBorder: "#ECA0A1",

		// Upstream purple #D699B6 is 4.07:1 on `elevated`; lifted on-hue.
		// Everforest's aqua is only ΔE00 9.4 from its green, so the purple
		// carries `info` rather than doubling the green family. Lifted toward
		// its magenta end rather than straight up: the plain solve (#DBA6BF)
		// landed ΔE00 10.5 from the lifted red, since both roles brighten into
		// the same pink. This holds 12.6.
		info: "#9ACAC3",
		infoWash: "#42424C",
		infoBorder: "#DEA3C6",

		overlayShadow: "0 12px 32px -12px rgb(20 25 27 / 0.7)",
		scrim: "rgb(20 25 27 / 0.6)",
	},
};
