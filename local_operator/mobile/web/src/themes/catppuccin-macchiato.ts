import type { ThemeDefinition } from "./palette-contract";

/**
 * Catppuccin Macchiato — the fourth flavour, from catppuccin/palette v1.8.0.
 *
 * Every content value is canonical and unmodified: base 24273A, mantle 1E2030,
 * text CAD3F5, subtext1 B8C0E0, subtext0 A5ADCB, overlay0 6E738D, overlay2
 * 939AB7, and the mauve/green/peach/red/blue accent row. Macchiato's darker
 * base gives every hue more room than frappé's does — the weakest value in the
 * palette is the red at 5.01:1 across all four grounds, against frappé's 3.76
 * for the same role — so nothing here needed a solve.
 *
 * Frappé and macchiato are close cousins, and the risk in shipping both is
 * shipping one theme twice. Upstream separates them at the GROUND (base 303446
 * vs 24273A, ΔE00 4.22) and this keeps that separation rather than splitting
 * the difference: each ramp is built from its own flavour's base and mantle,
 * so the two ladders stay 3.5-4.2 apart at every rung instead of converging on
 * a shared mid-slate. The accent rows are genuinely different hexes upstream
 * too, and both ship unmodified.
 */
export const catppuccinMacchiato: ThemeDefinition = {
	id: "catppuccinMacchiato",
	name: "Catppuccin Macchiato",
	description: "The deep flavour — cool navy-slate under the same pastels.",
	palette: {
		mode: "dark",

		canvas: "#24273A",
		// Seated between base and upstream surface0 363A4F, which is ΔE00 6.10
		// from base on its own — one step where this ramp needs two.
		surface: "#2A2D42",
		elevated: "#30334A",
		sunken: "#1E2132",

		ink: "#CAD3F5",
		inkMuted: "#B8C0E0",
		inkDim: "#A5ADCB",
		inkDisabled: "#6E738D",

		hairline: "#3C4056",
		// Upstream overlay2. overlay0 is the inactive tone and sits below the
		// 3:1 structural floor on the binding ground.
		borderControl: "#939AB7",

		accent: "#C6A0F6",
		accentHover: "#DCC5FA",
		accentActive: "#B27FF3",
		accentWash: "#2A2C42",
		onAccent: "#24273A",

		success: "#A6DA95",
		successWash: "#292D3D",
		successBorder: "#88CD71",

		warning: "#F5A97F",
		warningWash: "#2B2C3C",
		warningBorder: "#F2925D",

		danger: "#ED8796",
		dangerWash: "#2B2A3D",
		dangerBorder: "#E96C7F",

		info: "#8AADF4",
		infoWash: "#292E43",
		infoBorder: "#6E99F1",

		overlayShadow: "0 12px 32px -12px rgb(24 25 38 / 0.65)",
		scrim: "rgb(24 25 38 / 0.6)",
	},
};
