import type { ThemeDefinition } from "./palette-contract";

/**
 * Alucard — Dracula's official light counterpart, and the daylight half of the
 * `dracula` palette this app already ships.
 *
 * Every content value is canonical, from draculatheme.com/spec's Alucard
 * Classic table: background FFFBEB, foreground 1F1F1F, comment 6C664B, and the
 * red/orange/yellow/green/cyan/purple/pink row. The spec's authors solved
 * Alucard against its own paper, so the canonical values held the contract
 * as first shipped. The grounds, dim inks, warning and danger are now the
 * desktop's re-seat (see the ground note below): the page darkened to give a
 * borderless control a visible step, and the inks that sit on it moved with it.
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

		// GROUNDS PORTED FROM THE DESKTOP (alucard.ts, local-operator-ui
		// src/renderer/src/shared/themes/palettes). The relay's borderless chrome stands
		// every field, the composer and the search pill on `elevated` over `canvas`
		// with no edge, so the step between them IS the boundary; this palette's
		// relay grounds measured under the contract's 4.0 L* elevated−canvas floor
		// (scripts/contrast-contract.mjs), the desktop's re-seat clears it. The
		// desktop moved the inks, hairline and tones on the SAME commit (a new
		// ground changes what every ink is measured against), and those values are
		// ported with it; a ground ported alone failed eight text/edge pairs.
		canvas: "#F1EEDE",
		// The desktop moved lightness only, each value keeping its own a/b, so the
		// cream's hue is unchanged: the page darkened to make room above it.
		surface: "#F8F5EA",
		elevated: "#FEFDF8",
		// The spec has no recessed ground; the desktop's is one step under the page.
		sunken: "#EAE5D2",

		ink: "#1F1F1F",
		// The comment olive is a tertiary weight, not a secondary one, so it lands
		// on inkDim and inkMuted is solved one step deeper along the same
		// olive-khaki; both are the desktop's values for the darker page.
		inkMuted: "#4E4937",
		inkDim: "#655F45",
		inkDisabled: "#9B937A",

		hairline: "#D8D3C2",
		// The canonical comment olive, which clears the 3:1 structural floor on
		// every ground (the contract asserts it). A boundary may share a neutral
		// with the dim ink when the palette has only one to spend.
		borderControl: "#6C664B",

		accent: "#644AC9",
		accentHover: "#4E35B0",
		accentActive: "#402B91",
		accentWash: "#F1ECE2",
		onAccent: "#FFFBEB",

		success: "#14710A",
		successWash: "#F8F7E4",
		successBorder: "#17830C",

		warning: "#813100",
		warningWash: "#FBF3E1",
		warningBorder: "#954100",

		danger: "#C12F21",
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
