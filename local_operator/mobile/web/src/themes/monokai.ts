import type { ThemeDefinition } from "./palette-contract";

/**
 * Monokai.
 *
 * Upstream Monokai is unusually complete for this contract: background
 * 272822, line highlight 3E3D32, foreground F8F8F2, comment 75715E, plus a
 * green, an orange, a yellow, a pink and a blue. Only the recessed ground,
 * the structural border and the wash tints are derived.
 */
export const monokai: ThemeDefinition = {
	id: "monokai",
	name: "Monokai",
	description: "The Monokai scheme: olive-black ground with the acid green.",
	palette: {
		mode: "dark",

		canvas: "#272822",
		// Canvas and elevated are upstream's background and line highlight; surface
		// is the derived step between them, so it is the one that moves. 2D2E27 sat
		// ΔE00 1.95 from canvas — a card edge that is not there. One unit up the
		// same olive ramp reads 2.26, and still leaves 5.05 up to elevated.
		surface: "#2E2F28",
		elevated: "#3E3D32",
		sunken: "#1E1F1A",

		ink: "#F8F8F2",
		inkMuted: "#BFBFBF",
		// Monokai's comment 75715E, lightened until it clears 4.5:1 on the line
		// highlight ground. The comment colour itself stays as inkDisabled.
		inkDim: "#ACA998",
		inkDisabled: "#75715E",

		hairline: "#464740",
		// The comment colour lightened further, so a structural boundary is legally
		// distinct from the decorative hairline above it.
		borderControl: "#8E8A73",

		accent: "#A6E22E",
		accentHover: "#B6E94E",
		accentActive: "#8BC220",
		accentWash: "#363E23",
		onAccent: "#272822",

		// The same green as the accent. Monokai has exactly one green, and a second
		// one invented to fill this role would be a colour no Monokai user
		// recognises. A green success next to a green primary button is the honest
		// reading of this scheme.
		success: "#A6E22E",
		successWash: "#363E23",
		successBorder: "#6D8E29",

		// Monokai's orange.
		warning: "#FD971F",
		warningWash: "#413522",
		warningBorder: "#A76C22",

		// Monokai's pink is F92672, which measures 4.19:1 on canvas. This is the
		// smallest lift that clears 4.5:1 on both grounds and on its own wash.
		danger: "#FB6097",
		dangerWash: "#40282C",
		dangerBorder: "#D04573",

		// Monokai's blue.
		info: "#66D9EF",
		infoWash: "#2F3D3B",
		infoBorder: "#4A8993",

		overlayShadow: "0 12px 32px -12px rgb(15 15 12 / 0.65)",
		scrim: "rgb(15 15 12 / 0.6)",
	},
};
