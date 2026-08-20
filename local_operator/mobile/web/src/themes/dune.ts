import type { ThemeDefinition } from "./palette-contract";

/**
 * Dune.
 *
 * This one is the app's own theme rather than a community spec, so the
 * identity to preserve is narrower: the orange trio, the two warm near-black
 * grounds and the warm secondary text. Everything else is derived on the
 * theme's own ramp.
 */
export const dune: ThemeDefinition = {
	id: "dune",
	name: "Dune",
	description: "Desert night: warm near-black with a vivid orange.",
	palette: {
		mode: "dark",

		canvas: "#0F0D0B",
		surface: "#1A1714",
		elevated: "#261E17",
		// The old theme had no recessed ground. 050403 sits far enough below canvas
		// to stay distinguishable — at this darkness a step of one or two levels
		// measures under the 1.03:1 separation floor.
		sunken: "#050403",

		// The old primary text was F9FAFB, a blue-white. Every other neutral here
		// holds R greater than G greater than B, and one cool value in an otherwise
		// warm ramp is what makes a palette look accidental, so the primary ink is
		// warmed to match. Same lightness, same legibility.
		ink: "#F8F3EC",
		inkMuted: "#BFB3A7",
		inkDim: "#9E9285",
		inkDisabled: "#6B6055",

		hairline: "#3A2B11",
		// Derived. The old theme bounded inputs with orange at 10 percent alpha,
		// which measured about 1.1:1 on the page ground.
		borderControl: "#8A7D6E",

		accent: "#FF8C38",
		accentHover: "#FFA75C",
		accentActive: "#E67016",
		accentWash: "#2C1C10",
		// The old theme paired white with this orange, which measures 2.32:1 — a
		// primary button nobody could read. Ink on the accent fill is the page
		// ground instead, at 8.4:1.
		onAccent: "#0F0D0B",

		// Dune has no semantic hues at all. These three are derived inside the
		// theme's own desert range — olive for success, amber for warning, burnt
		// red for danger — so none of them reads as a foreign import.
		success: "#A3C46B",
		successWash: "#212317",
		successBorder: "#607240",

		warning: "#E8C15A",
		warningWash: "#292314",
		warningBorder: "#867036",

		danger: "#E5604C",
		dangerWash: "#291713",
		dangerBorder: "#A74A3B",

		// Dune has no informational hue, and a blue would be the only cool colour
		// in the theme. Info is the accent triple, as in the brand palettes.
		info: "#FF8C38",
		infoWash: "#2C1C10",
		infoBorder: "#985625",

		overlayShadow: "0 12px 32px -12px rgb(5 4 3 / 0.7)",
		scrim: "rgb(5 4 3 / 0.6)",
	},
};
