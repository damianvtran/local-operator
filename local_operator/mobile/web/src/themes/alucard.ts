import type { ThemeDefinition } from "./palette-contract";

/**
 * Alucard — Dracula's official light counterpart, and the daylight half of the
 * `dracula` palette this app already ships.
 *
 * Every content value is canonical, from draculatheme.com/spec's Alucard
 * Classic table: background FFFBEB, foreground 1F1F1F, comment 6C664B, and the
 * red/orange/yellow/green/cyan/purple/pink row. Nothing here is darkened.
 * That is unusual for a light palette against this contract — most community
 * light schemes are tuned for syntax and arrive 2-3:1 — and it is not luck:
 * the spec's authors solved Alucard against its own paper, so the weakest
 * value in the whole palette is the red at 4.77:1 across all four grounds.
 *
 * Purple carries `accent` here for the same reason it does in `dracula`: it is
 * the identity hue of both variants, so the pair reads as one theme in two
 * lights rather than two themes that share a name.
 */
export const alucard: ThemeDefinition = {
	id: "alucard",
	name: "Alucard",
	description: "Dracula by daylight — warm cream with the signature purple.",
	palette: {
		mode: "light",

		canvas: "#FFFBEB",
		// A light ramp has almost no room above its page ground, and Alucard's
		// is already near white. These two steps clear the ΔE00 2 the ramp
		// comment in local-operator.ts sets as the visible threshold (2.46 and
		// 2.24) by shedding the cream's own chroma as they rise, rather than by
		// lightness alone, which runs out here.
		surface: "#FFFDF3",
		elevated: "#FFFEF9",
		// The spec has no recessed ground; this is one just-visible step under
		// the paper (ΔE00 2.23), which is all a light ramp has to spend.
		sunken: "#FFF9E3",

		ink: "#1F1F1F",
		// Canonical comment 6C664B measures 5.47:1 on the darkest ground — a
		// tertiary weight, not a secondary one — so it lands on inkDim and
		// inkMuted is solved one step deeper along the same olive-khaki.
		inkMuted: "#4F4A37",
		inkDim: "#6C664B",
		inkDisabled: "#9B937A",

		hairline: "#DFDAC8",
		// The comment olive, which already clears the 3:1 structural floor with
		// room (5.47:1 on the binding ground). A boundary is allowed to be the
		// same value as inkDim when the palette has only one neutral to spend.
		borderControl: "#6C664B",

		accent: "#644AC9",
		accentHover: "#4E35B0",
		accentActive: "#402B91",
		accentWash: "#FAF5EA",
		onAccent: "#FFFBEB",

		success: "#14710A",
		successWash: "#F8F7E4",
		successBorder: "#17830C",

		warning: "#A34D14",
		warningWash: "#FBF3E1",
		warningBorder: "#BB5817",

		danger: "#CB3A2A",
		dangerWash: "#FDF4E4",
		dangerBorder: "#D85041",

		// The spec's cyan, so this palette needs no invented informational hue.
		info: "#036A96",
		infoWash: "#F5F5E8",
		infoBorder: "#037BAE",

		overlayShadow: "0 12px 32px -12px rgb(31 31 31 / 0.22)",
		scrim: "rgb(31 31 31 / 0.35)",
	},
};
