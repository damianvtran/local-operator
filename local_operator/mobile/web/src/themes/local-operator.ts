import type { ThemeDefinition } from "./palette-contract";

/**
 * The two brand palettes, and the reference implementation of the role
 * contract.
 *
 * These carry the Local Operator design kit's own values — the same warm
 * ramp, accent and semantic triples the marketing site ships, so a user
 * moving from local-operator.com into the app lands on the same surface.
 * Every other palette in this directory is a community or third-party identity
 * and only has to *satisfy* the contract; these two *are* the brand, so when
 * the two disagree, these win.
 *
 * ## Where these deviate from the kit, and why that is allowed
 *
 * `docs/branding.md` settles the precedence: on the *brand* the kit wins, on
 * *how a desktop app should behave* this repo wins, "because a tool read for
 * hours at arm's length is not a page read once". Exactly two things invoke
 * that second clause. No hue the kit owns has been retuned to buy contrast
 * headroom — a brand colour changes in the kit and is followed here, never
 * the other way round.
 *
 * 1. **The light ground ramp** is wider than the site's. A page renders one
 *    surface; this app stacks four, and the kit's near-white ladder made a
 *    panel on canvas and a popover on that panel the same pixel to the eye.
 *    See the ramp comment in `localOperatorLight`.
 * 2. **`info` is not a kit role at all.** The site has no fourth semantic, so
 *    there is nothing here to be faithful to; see the `info` comment in
 *    `localOperatorDark`.
 *
 * ## The warm neutral rule
 *
 * The ramp holds `R > G > B` in light and `R >= G > B` in dark. That is what
 * makes it read as paper and ink rather than as grey, and it is the one
 * structural property a palette cannot fake with a hue rotation. A theme does
 * not have to be warm, but its neutrals must be *consistently* tinted in one
 * direction — mixed-temperature neutrals are what make a palette look
 * accidental.
 *
 * @see docs/branding.md § 2 — the ramp and why it is tinted
 */

/** Verified 2026-08-04 against docs/design-kit/tokens.json in the site repo. */
export const localOperatorDark: ThemeDefinition = {
	id: "localOperatorDark",
	name: "Local Operator Dark",
	description: "The default. Warm near-black with the brand green.",
	palette: {
		mode: "dark",

		canvas: "#16130e",
		surface: "#1e1a14",
		elevated: "#282318",
		sunken: "#0f0c08",

		ink: "#f1eee6",
		inkMuted: "#b5afa2",
		inkDim: "#918b7d",
		inkDisabled: "#5f5a4e",

		hairline: "#353022",
		borderControl: "#837c6d",

		accent: "#38c96a",
		accentHover: "#5ad584",
		accentActive: "#2bb25c",
		accentWash: "#16281d",
		onAccent: "#16130e",

		success: "#57c785",
		successWash: "#16281d",
		successBorder: "#417557",
		warning: "#e0b04b",
		warningWash: "#2a2213",
		warningBorder: "#857036",
		danger: "#ef8078",
		dangerWash: "#2e1b18",
		dangerBorder: "#9e5a51",
		/*
		 * The kit has no `info` role: on the site, "informational" is carried by
		 * the accent, because a fourth semantic hue is a hue nobody can name.
		 * The app needs one anyway — `palette.info` is read by application code
		 * and by MUI's own Alert — and it used to be defined as the accent's
		 * own triple, to keep "one accent, spent about three times per screen"
		 * true.
		 *
		 * That was the wrong thing to economise on. It made `success` and
		 * `info` the same signal: ΔE00 5.1 apart in this palette, 2.2 in the
		 * light one, with washes that were byte-identical. A user cannot be
		 * asked to tell "it worked" from "here is a fact" by colour when the
		 * two colours are the same colour, and the accent budget is about how
		 * often accent is *spent*, not about refusing to name a fourth state.
		 * Every other palette in this directory separates them by 26 to 47.
		 *
		 * So `info` is a blue, which is the reading users already expect, and
		 * it is a warm one: L*72 C*36 at Lab hue 272, which leans to the red
		 * side of blue rather than the cyan side. A cyan would have sat next
		 * to the brand green in hue and read as a fifth tint of it; a cold
		 * azure on a warm near-black reads as pasted in. The wash keeps the
		 * canvas's own red floor (R 25 against canvas R 22, the same trick
		 * `successWash` uses) so it tints the ground rather than replacing it.
		 *
		 * Measured: 8.60:1 on canvas, 8.03:1 on surface, 7.34:1 on its own
		 * wash; ΔE00 41.0 from `success`, 46.1 from `accent`.
		 */
		info: "#86b3f2",
		infoWash: "#192332",
		infoBorder: "#5475a2",

		overlayShadow: "0 12px 32px -12px rgb(0 0 0 / 0.6)",
		scrim: "rgb(0 0 0 / 0.6)",
	},
};

/*
 * Kit-verified 2026-08-04 against docs/design-kit/tokens.json in the site
 * repo, except the four grounds and the `hairline` that separates them, which
 * this app widens for the reason recorded below, and `info`, which the kit
 * does not define.
 */
export const localOperatorLight: ThemeDefinition = {
	id: "localOperatorLight",
	name: "Local Operator Light",
	description: "Warm paper and ink, with the brand green.",
	palette: {
		mode: "light",

		/*
		 * The four grounds are a lightness ladder, and on a light theme the top
		 * of it is cramped: `elevated` is already a shade off white, so there
		 * is almost no headroom above `surface`. The gate's floor here is a
		 * 1.03 luminance ratio, and clearing it is not the same test as being
		 * able to see the step. This ramp cleared the gate and failed the eye:
		 * `canvas`/`surface` measured ΔE00 1.53 and `surface`/`elevated` 1.14,
		 * and in the captured theme frames a plain card on canvas simply did
		 * not read as a separate surface. The observed threshold, from the same
		 * frames, is around ΔE00 2: sage renders at 1.9 and iceberg at 2.1.
		 *
		 * Lightness alone cannot buy that. Every ink in the palette is measured
		 * against `canvas`, so each step down costs contrast, and `elevated` is
		 * pinned near white with nothing above it. The lever that is free is
		 * chroma: ΔE00 has a chroma axis and a contrast ratio does not, so at a
		 * fixed L* the ramp can separate itself by warmth at zero cost to any
		 * assertion. So the ladder now runs warmer as it runs deeper — b* 1.55,
		 * 3.59, 5.42, 7.52 from `elevated` down to `sunken` — which is how a
		 * stack of real paper behaves, and it is the same move that makes
		 * sage's near-white ramp legible where this one was not.
		 *
		 * Adjacent steps now measure ΔE00 2.25 / 2.31 / 2.25 (ratios 1.0536,
		 * 1.0688, 1.0657), so the gate's floor is cleared with room rather than
		 * scraped: `surface` used to be #fcfbf7 at 1.0267 against `elevated`,
		 * under the floor and passing only because the gate rounded before it
		 * compared. The cost is 0.15 of contrast on the semantics measured
		 * against `canvas` — `success` goes 4.78:1 to 4.62:1 — which is spent,
		 * not free, and is why no further depth is taken here.
		 *
		 * Change one of these four and re-run `pnpm check-themes`; they are a
		 * ladder and only make sense relative to each other.
		 */
		canvas: "#f5f0e6",
		surface: "#faf8f1",
		elevated: "#fffefb",
		sunken: "#efe9db",

		ink: "#211e18",
		inkMuted: "#565147",
		inkDim: "#6c675c",
		inkDisabled: "#9a9488",

		// Deepened from #e5e0d5 alongside the ramp above. A hairline is the one
		// role that has to move when its grounds do: left where it was, it fell
		// to ΔE00 3.50 against the new `canvas` and 2.31 against `sunken`,
		// undoing on the dividers exactly what the ramp bought on the panels.
		hairline: "#ddd8ce",
		// #857f70 measures 3.29:1 on `sunken` and 3.95:1 on `elevated`. The
		// binding ground is the darkest one, not the lightest — a mid grey has
		// its easiest time against white — so `sunken` is the case that has to
		// clear the 3:1 structural floor, and it does. The previous light theme
		// used `rgba(0,0,0,0.1)` here, which measured 1.25:1 and was the sole
		// boundary of every input in the app.
		borderControl: "#857f70",

		accent: "#147842",
		accentHover: "#116036",
		accentActive: "#0c4b2a",
		accentWash: "#e7f1e8",
		onAccent: "#F6FAF8",

		success: "#1A774A",
		successWash: "#e6f1ea",
		successBorder: "#3e6b4e",
		warning: "#8a5800",
		warningWash: "#f5ecd9",
		warningBorder: "#7a5a1e",
		danger: "#b23a31",
		dangerWash: "#f7e7e4",
		dangerBorder: "#96544c",
		// A blue, for the reasons written out at `info` in `localOperatorDark`;
		// same Lab hue 272, at L*43 C*40 to sit in this palette's ink family.
		// Measured: 5.11:1 on `canvas`, 5.46:1 on `surface`, 4.86:1 on its own
		// wash, with the border at 5.03:1 against `canvas`. ΔE00 38.9 from
		// `success` and 41.4 from `accent`, against 2.2 and 0.0 before.
		//
		// The wash is seated by loudness, not by eye. Every other wash in this
		// palette sits ΔE00 3.69-7.19 from `canvas`; the first blue tried here
		// sat at 10.29, so the one cool callout in the set arrived 1.4x louder
		// than the four warm ones it joined. Cool is expensive on warm paper —
		// hue distance, not lightness, is what costs — so this holds b* at -2.2
		// instead of -6.0: still visibly the cool one beside `warningWash`, now
		// ΔE00 7.06, just inside `dangerWash`'s 7.19 at the top of the family.
		info: "#2368a8",
		infoWash: "#e9ebef",
		infoBorder: "#486893",

		overlayShadow: "0 12px 32px -12px rgb(20 17 12 / 0.25)",
		scrim: "rgb(20 17 12 / 0.35)",
	},
};
