import type { ThemeDefinition } from "./palette-contract";

/**
 * One Light — Atom's official light syntax theme.
 *
 * Upstream (atom/one-light-syntax, styles/colors.less) defines this palette in
 * HSL rather than hex, so the values here are that source converted: mono-1
 * hsl(230,8%,24%) = 383A42, mono-2 = 696C77, mono-3 = A0A1A7, hue-1 cyan
 * 0184BC, hue-2 blue 4078F2, hue-3 purple A626A4, hue-4 green 50A14F, hue-5
 * red E45649, hue-6-2 orange C18401.
 *
 * One Light is tuned for syntax on white, where a hue only has to be
 * distinguishable from its neighbours; as UI state colours the accents land
 * between 2.43:1 and 3.17:1 against the 4.5 floor. Each is darkened along its
 * own hue, with the measured miss recorded per role.
 */
export const oneLight: ThemeDefinition = {
	id: "oneLight",
	name: "One Light",
	description: "Atom's daylight standard, cool white with the One accent row.",
	palette: {
		mode: "light",

		/*
		 * Upstream's page is FAFAFA — hsl(230,1%,98%), two percent off white.
		 * A four-ground ladder cannot be built upward from there: FAFAFA to
		 * pure white is ΔE00 1.00 in total, against the ~2 per step the ramp
		 * comment in local-operator.ts sets as the visible threshold, so a
		 * card on the page and a popover on the card would be one pixel value
		 * to the eye.
		 *
		 * So the ladder is seated lower and upstream's own white sits at the
		 * TOP of it, which is the same move localOperatorLight makes (its
		 * canvas F5F0E6 sits below its elevated FFFEFB). The theme still reads
		 * as One Light because the ink and accent row carry the identity; the
		 * page ground is the one value that has to move for elevation to be
		 * visible at all. Steps measure ΔE00 2.20 / 2.10 / 2.22.
		 */
		canvas: "#EAEAEA",
		surface: "#F4F4F4",
		elevated: "#FFFFFF",
		sunken: "#E1E0E0",

		ink: "#383A42",
		// Canonical mono-2 696C77 is 3.97:1 on `sunken` (< 4.5); darkened
		// minimally along the same neutral. mono-2 itself becomes the
		// structural border below, where the floor is 3:1.
		inkMuted: "#60636D",
		// Canonical mono-3 A0A1A7 is 1.96:1 on `sunken` (< 4.5).
		inkDim: "#62636A",
		inkDisabled: "#A0A1A7",

		hairline: "#CECED1",
		borderControl: "#696C77",

		// Canonical hue-2 blue 4078F2 is 3.07:1 on `sunken` (< 4.5).
		accent: "#1055EC",
		accentHover: "#0D46C1",
		accentActive: "#0A389B",
		accentWash: "#E5E6EA",
		onAccent: "#EAEAEA",

		// Canonical hue-4 green 50A14F is 2.43:1 on `sunken` (< 4.5).
		success: "#387037",
		successWash: "#E2E5E2",
		successBorder: "#41813F",

		// Canonical hue-6-2 orange C18401 is 2.43:1 on `sunken` (< 4.5).
		warning: "#855B01",
		warningWash: "#E5E4E0",
		warningBorder: "#996901",

		// Canonical hue-5 red E45649 is 2.78:1 on `sunken` (< 4.5).
		danger: "#BE2A1C",
		dangerWash: "#E9E4E4",
		dangerBorder: "#DA3020",

		// Canonical hue-1 cyan 0184BC is 3.17:1 on `sunken` (< 4.5). Kept as
		// the cyan rather than a second blue so an info callout does not read
		// as a primary one.
		info: "#016A97",
		infoWash: "#E1E5E7",
		infoBorder: "#017AAE",

		overlayShadow: "0 12px 32px -12px rgb(56 58 66 / 0.22)",
		scrim: "rgb(56 58 66 / 0.35)",
	},
};
