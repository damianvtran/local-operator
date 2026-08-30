import type { ThemeDefinition } from "./palette-contract";

/**
 * Ayu Dark.
 *
 * The repo's `themes/*.yaml` is a generator whose syntax entries are
 * `$palette.<hue>.l<n>` references rather than literal hexes, so these values
 * come from the published package build (ayu@9.0.0, `dist/dark.js`), which
 * resolves them.
 *
 * Ayu's dark ground is the deepest in this batch, and the whole accent row
 * ships canonical as a result — the lowest is markup red at 5.16:1 on the
 * lightest ground. Only the grounds themselves are solved: ayu publishes
 * ui.bg, editor.bg, editor.line and panel.bg, which do not form four
 * separated steps (sunken → canvas measured ΔE00 1.54 taken straight), so the
 * ladder below is spread to step 3.78 / 2.96 / 3.55.
 *
 * This theme and `nightfox` are both blue-blacks. Their grounds sit ΔE00 5.4
 * apart, but their accent rows average ΔE00 21.9 apart, which is what keeps
 * them distinct themes rather than two names for one palette.
 */
export const ayuDark: ThemeDefinition = {
	id: "ayuDark",
	name: "Ayu Dark",
	description: "Near-black blue with high-key amber and lime accents.",
	palette: {
		mode: "dark",

		canvas: "#10141C",
		surface: "#181D27",
		elevated: "#222834",
		sunken: "#080A0F",

		// Upstream editor.fg (7.86:1 on the lightest ground).
		ink: "#BFBDB6",
		// Ayu's comment is #ACB6BF at 55% alpha over the editor ground; the
		// flattened value is what a solid role needs, and it is what the user
		// actually sees upstream.
		inkMuted: "#ACB6BF",
		inkDim: "#96A0AB",
		// Upstream ui.fg — 2.17:1 on `elevated`, which is why it is the
		// disabled role and not a text one.
		inkDisabled: "#565B66",

		hairline: "#1B1F29",
		// Upstream ui.line #1B1F29 is 1.11:1 on `elevated`: a divider colour
		// doing an input boundary's job. Lifted along the same blue-grey.
		borderControl: "#667284",

		// Upstream entity.
		accent: "#59C2FF",
		accentHover: "#7DCFFF",
		accentActive: "#09A5FF",
		accentWash: "#152431",
		onAccent: "#0D1017",

		// Upstream string.
		success: "#AAD94C",
		successWash: "#181F1F",
		successBorder: "#AAD94C",

		// Upstream func.
		warning: "#FFB454",
		warningWash: "#232120",
		warningBorder: "#FFB454",

		// Upstream markup.
		danger: "#F07178",
		dangerWash: "#221B23",
		dangerBorder: "#F07178",

		// Upstream constant, so this theme needs no invented informational hue.
		info: "#D2A6FF",
		infoWash: "#222232",
		infoBorder: "#D2A6FF",

		overlayShadow: "0 12px 32px -12px rgb(5 6 9 / 0.8)",
		scrim: "rgb(5 6 9 / 0.65)",
	},
};
