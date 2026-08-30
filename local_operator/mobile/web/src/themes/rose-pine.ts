import type { ThemeDefinition } from "./palette-contract";

/**
 * Rosé Pine, all three published variants.
 *
 * Upstream: github.com/rose-pine/palette, the canonical hex list in
 * `dist/css/rose-pine.css`. Each variant ships nine neutral roles
 * (`base surface overlay muted subtle text highlight-low/med/high`) and six
 * accents (`love gold rose pine foam iris`), against this contract's
 * twenty-nine. The neutrals map almost verbatim; the accents map by
 * upstream's own usage table rather than by hue:
 *
 * - `love` is the documented terminal red and error colour -> `danger`.
 * - `gold` is the documented warnings colour -> `warning`.
 * - `foam` is the documented "object keys, info, git add" hue -> `info`.
 * - `pine` is the documented terminal green (functions) -> `success`. It is
 *   a blue-green rather than a leaf green, which is Rosé Pine's signature
 *   inversion and is kept.
 * - `rose` takes `accent`: the theme's own name, and upstream's "git change,
 *   git dirty" hue - the colour of something happening.
 *
 * `iris` has no role in this contract (the TUI spends it on metadata labels,
 * which this app has no equivalent of), so it appears only in the hover and
 * active steps where the ramp needs a neighbour.
 *
 * ## What the four grounds cost
 *
 * Rosé Pine defines three grounds and this contract needs four. Each variant
 * therefore derives one step, and - importantly - the contract measures every
 * ink and semantic against ALL FOUR, including the lightest. That is a
 * strictly harder test than the ratios Rosé Pine's own docs quote, which is
 * why several accents that look canonical elsewhere are lifted here: the
 * binding ground is `elevated`, not `canvas`. Every deviation below carries
 * the canonical value and the measured reason.
 *
 * @see docs/branding.md § 3 - the floors these values satisfy
 */

/** Rosé Pine (main): the original night. */
export const rosePine: ThemeDefinition = {
	id: "rosePine",
	name: "Rosé Pine",
	description: "All natural pine, faux fur and a bit of soho vibes.",
	palette: {
		mode: "dark",

		canvas: "#191724",
		surface: "#1f1d2e",
		elevated: "#26233a",
		// Upstream has no ground below base. One step down, far enough to stay
		// visible at this darkness (a one-level step measures under the
		// separation floor down here).
		sunken: "#12101b",

		ink: "#e0def4",
		// Upstream's `subtle` #908caa clears the 4.5 floor on all four grounds
		// (5.48 / 5.12 / 4.71 / 5.84), so this pair is a HEADROOM choice, not a
		// floor rescue: on `elevated` it clears by 0.21, and this contract has
		// two mid inks where Rose Pine has one, so `inkDim` must sit BELOW
		// `inkMuted` and would inherit that thin margin. Both are lifted along
		// the same violet-gray toward `text` to buy the second rung room
		// (`inkDim` #a9a5c4 lands at 6.41 on `elevated`). Canonical `muted`
		// #6e6a86 is `inkDisabled`, which is what upstream uses it for
		// (comments, git-ignored) and the one role exempt from the floors.
		inkMuted: "#c5c2dd",
		inkDim: "#a9a5c4",
		inkDisabled: "#6e6a86",

		// Canonical highlight-low: upstream's cursor-line colour, which is exactly
		// a decorative rule's job.
		hairline: "#21202e",
		// Canonical highlight-high #524f67 is 2.25:1 on canvas - it is a highlight,
		// not a boundary, and every input in the app would have been bounded under
		// the 3:1 structural floor. Lifted along the same hue to clear that floor
		// on all four grounds (5.12:1 / 4.40:1 at the extremes).
		borderControl: "#8b87a3",

		accent: "#ebbcba",
		accentHover: "#f2cfcd",
		accentActive: "#d9a5a3",
		accentWash: "#2e2430",
		// Rose is a pale pink at 10.45:1 on canvas, so ink on the accent fill is
		// the page ground rather than a light value.
		onAccent: "#191724",

		// Canonical pine #31748f measures 3.38:1 on canvas and 2.91:1 on
		// `elevated` - the only canonical accent under the floor here, and under
		// it on every ground. Lifted along its own blue-green to 5.35:1 / 4.60:1.
		success: "#5995b2",
		successWash: "#1a2730",
		successBorder: "#417d99",

		warning: "#f6c177",
		warningWash: "#2b2419",
		warningBorder: "#8a7040",

		danger: "#eb6f92",
		dangerWash: "#2e1c26",
		dangerBorder: "#9e5468",

		info: "#9ccfd8",
		infoWash: "#1a2a2e",
		infoBorder: "#4f8189",

		overlayShadow: "0 12px 32px -12px rgb(14 12 20 / 0.7)",
		scrim: "rgb(14 12 20 / 0.6)",
	},
};

/** Rosé Pine Moon: the same palette one shade warmer and lighter. */
export const rosePineMoon: ThemeDefinition = {
	id: "rosePineMoon",
	name: "Rosé Pine Moon",
	description: "Rosé Pine's warmer moonlit ground, one shade up.",
	palette: {
		mode: "dark",

		canvas: "#232136",
		surface: "#2a273f",
		elevated: "#393552",
		sunken: "#1b192a",

		ink: "#e0def4",
		inkMuted: "#c9c6e0",
		inkDim: "#b0acc9",
		inkDisabled: "#6e6a86",

		hairline: "#2a283e",
		borderControl: "#8f8ba9",

		// Canonical moon rose #ea9a97 sits 43 RGB units from love #eb6f92 - on
		// main they are 87 apart - close enough that the primary action and an
		// error message read as the same hue. Desaturated one step to #eaaca9,
		// still moon's rose, now 32 ΔE from danger.
		accent: "#eaaca9",
		accentHover: "#f2c2bf",
		accentActive: "#d69793",
		accentWash: "#382b38",
		onAccent: "#232136",

		// Canonical moon pine #3e8fb0 is 3.19:1 on `elevated`. Lifted, and kept
		// 24.5 ΔE off foam so success and info stay two signals.
		success: "#4eadd9",
		successWash: "#22303c",
		successBorder: "#4a8aa8",

		warning: "#f6c177",
		warningWash: "#37301f",
		warningBorder: "#8f7546",

		// Canonical love #eb6f92 is 4.00:1 on moon's lighter `elevated` - the same
		// red that clears comfortably on main. Lifted 4.4 ΔE to 4.60:1.
		danger: "#f97a9e",
		dangerWash: "#38222e",
		dangerBorder: "#a35a72",

		info: "#9ccfd8",
		infoWash: "#23343a",
		infoBorder: "#57868f",

		overlayShadow: "0 12px 32px -12px rgb(20 18 32 / 0.7)",
		scrim: "rgb(20 18 32 / 0.6)",
	},
};

/** Rosé Pine Dawn: the light variant. */
export const rosePineDawn: ThemeDefinition = {
	id: "rosePineDawn",
	name: "Rosé Pine Dawn",
	description: "Rosé Pine at sunrise, warm paper under a soho morning.",
	palette: {
		mode: "light",

		/*
		 * The ground ladder inverts, and Rosé Pine hands it over almost intact:
		 * canonical dawn `surface` #fffaf3 is LIGHTER than `base` #faf4ed, so
		 * `base` is the page ground, canonical `surface` is the raised one, and
		 * canonical `highlight-low` #f4ede8 - which upstream paints the cursor
		 * line with - becomes the recessed ground. Only `elevated` is derived,
		 * warmed rather than merely lightened for the reason the brand light
		 * ramp documents: a near-white ladder has no lightness headroom left, so
		 * the steps separate on the chroma axis. Walking the ladder from `sunken`
		 * up, adjacent steps measure ΔE 2.06 / 2.04 / 2.55 - just past the ~2 the
		 * eye needs.
		 */
		canvas: "#faf4ed",
		surface: "#fffaf3",
		elevated: "#fffdf9",
		sunken: "#f4ede8",

		// Canonical text #575279 is 6.66:1 on `base` and 6.29:1 on the recessed
		// ground, under the 7:1 primary floor on both. Deepened 3.8 ΔE.
		ink: "#4e4970",
		inkMuted: "#6d6785",
		// Canonical muted #9893a5 is 2.73:1 - a comment colour, not a caption
		// colour. It moves to `inkDisabled` (the floor-exempt role it belongs in)
		// and the tertiary ink is a darker neutral on the same violet-gray.
		inkDim: "#6c697a",
		inkDisabled: "#9893a5",

		hairline: "#e6ddd6",
		// Canonical highlight-high #cecacd is 1.48:1: a highlight, not a boundary.
		borderControl: "#91838c",

		/*
		 * Every canonical dawn accent except pine misses the 4.5:1 floor on the
		 * four grounds - gold 2.05:1, rose 2.60:1, foam 3.14:1, iris 3.47:1,
		 * love 3.84:1 - so each is darkened along its own hue by the smallest
		 * step that clears it.
		 *
		 * `accent` (rose) and `danger` (love) both converge toward the same dusty
		 * red as they darken; at 15.3 ΔE apart they are the tightest pair here,
		 * which is deliberate - holding them further apart meant abandoning one
		 * of the two canonical hues entirely.
		 */
		accent: "#a15552", // rose #d7827e: 2.60:1
		accentHover: "#8b4644",
		accentActive: "#743a38",
		accentWash: "#fbece8",
		onAccent: "#fffaf3",

		success: "#286983", // pine - CANONICAL, the only accent that clears unaided
		successWash: "#e4edf1",
		successBorder: "#5c8ba0",

		warning: "#995d00", // gold #ea9d34: 2.05:1 - the largest deviation
		warningWash: "#f6ead6",
		warningBorder: "#b08b4c",

		danger: "#a1526a", // love #b4637a: 3.84:1
		dangerWash: "#fcebef",
		dangerBorder: "#b57e90",

		// foam #56949f: 3.14:1, and darkening it along its own hue walks it
		// straight into pine. Pushed to the teal side so info and success stay
		// 20.1 ΔE apart, which is the separation the other palettes here hold.
		info: "#107574",
		infoWash: "#e2eeef",
		infoBorder: "#5b979a",

		overlayShadow: "0 12px 32px -12px rgb(87 82 121 / 0.24)",
		scrim: "rgb(87 82 121 / 0.4)",
	},
};
