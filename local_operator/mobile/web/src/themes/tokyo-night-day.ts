import type { ThemeDefinition } from "./palette-contract";

/**
 * Tokyo Night Day — the light variant, completing the set beside the
 * `tokyoNight` this app already ships.
 *
 * Unlike every other palette in this directory there is no published hex table
 * to be faithful to: upstream GENERATES `day` by inverting the `night` palette
 * programmatically (lua/tokyonight/colors/day.lua calls `Util.invert`). The
 * values below are upstream's own generated artifacts — extras/kitty/
 * tokyonight_day.conf and extras/wezterm/tokyonight_day.toml, which agree hex
 * for hex: background E1E2E7, foreground 3760BF, blue 2E7DE9, red F52A65,
 * green 587539, yellow 8C6C3E, purple 9854F1, cyan 007197. Treat them as
 * upstream's output rather than as a spec — a retune of `night` moves all of
 * them.
 *
 * Inversion preserves hue, not contrast, and this is the one palette here
 * where nearly every generated value misses: the generated foreground is
 * 4.19:1 on the darkest ground where the body floor is 7, and the six accents
 * land between 2.78:1 and 3.95:1 against the 4.5 floor. Each is re-solved
 * along its own hue by the minimum that clears all four grounds, and the
 * measured miss is recorded per role below.
 */
export const tokyoNightDay: ThemeDefinition = {
	id: "tokyoNightDay",
	name: "Tokyo Night Day",
	description: "Tokyo Night at noon — cool paper under the same neon hues.",
	palette: {
		mode: "light",

		canvas: "#E1E2E7",
		surface: "#EAEBEE",
		elevated: "#F3F4F5",
		// Generated inactive-tab C4C8DA sits far below the page and would cost
		// every ink two thirds of its headroom on the binding ground. This is
		// one just-visible step under canvas instead (ΔE00 2.06).
		sunken: "#D8DAE1",

		// Generated foreground 3760BF is 4.19:1 on `sunken` — a syntax blue,
		// not a body ink, which is what the 7:1 floor is for. Deepened along
		// the same indigo to 7.00:1.
		ink: "#254180",
		// Generated comment 6172B0 is 3.31:1 on `sunken` (< 4.5).
		inkMuted: "#4C5D99",
		// Generated dark5 8990B3 is 2.24:1 on `sunken` (< 4.5). Deepened along
		// its own hue, and kept a distinct rung from inkMuted above rather than
		// collapsing the two.
		inkDim: "#565E86",
		inkDisabled: "#8990B3",

		hairline: "#C5C9DB",
		// The generated comment blue, which clears the 3:1 structural floor
		// (3.31:1 on the binding ground) where it could not clear 4.5 as text.
		borderControl: "#6172B0",

		// Generated blue 2E7DE9 is 2.88:1 on `sunken` (< 4.5).
		accent: "#145CBF",
		accentHover: "#104895",
		accentActive: "#0C3671",
		accentWash: "#DADDE6",
		onAccent: "#E1E2E7",

		// Generated green 587539 is 3.74:1 on `sunken` (< 4.5).
		success: "#4E6732",
		successWash: "#DADCDE",
		successBorder: "#5B783A",

		// Generated yellow 8C6C3E is 3.47:1 on `sunken` (< 4.5).
		warning: "#765B34",
		warningWash: "#DBDADC",
		warningBorder: "#896A3D",

		// Generated red F52A65 is 2.78:1 on `sunken` — the largest miss here.
		danger: "#BF093E",
		dangerWash: "#E0DDE3",
		dangerBorder: "#DC0A47",

		// Generated cyan 007197 is 3.95:1 on `sunken`, the narrowest miss of
		// the six; darkened one step along the same cyan.
		info: "#00678A",
		infoWash: "#D8DDE3",
		infoBorder: "#0078A1",

		overlayShadow: "0 12px 32px -12px rgb(37 65 128 / 0.22)",
		scrim: "rgb(37 65 128 / 0.35)",
	},
};
