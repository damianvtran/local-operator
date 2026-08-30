import type { ThemeDefinition } from "./palette-contract";

/**
 * Kanagawa Wave.
 *
 * Carried from rebelot/kanagawa.nvim, the default `wave` variant, resolved
 * through `lua/kanagawa/themes.lua` rather than read off the palette names:
 * the scheme's own `ui.bg` is sumiInk3, with sumiInk4/sumiInk5 above it and
 * sumiInk0 below, which is exactly the four grounds this contract wants.
 *
 * Kanagawa's ink-wash ground is dark enough that most of the palette clears
 * the floors unmodified — fujiWhite, oldWhite, springGreen, carpYellow and
 * the aqua all ship canonical. Only the three roles noted below move, and
 * each moves because the lightest ground (sumiInk5) binds where the TUI
 * port's two grounds did not.
 */
export const kanagawaWave: ThemeDefinition = {
	id: "kanagawaWave",
	name: "Kanagawa Wave",
	description: "Hokusai ink-wash night, muted paper tones on deep sumi.",
	palette: {
		mode: "dark",

		canvas: "#1F1F28",
		surface: "#2A2A37",
		elevated: "#363646",
		sunken: "#16161D",

		// Upstream fujiWhite (8.16:1 on the lightest ground).
		ink: "#DCD7BA",
		// Upstream oldWhite.
		inkMuted: "#C8C093",
		// Upstream springViolet1 #938AA9 is 3.64:1 on `elevated`, under the
		// 4.5 floor; lifted on-hue.
		inkDim: "#A9A3B8",
		// Upstream fujiGray, the scheme's own comment colour.
		inkDisabled: "#727169",

		hairline: "#363646",
		// Upstream sumiInk6 #54546D is 1.62:1 on `elevated` — it is a ground
		// colour upstream, not a boundary. Lifted along the same violet-grey.
		borderControl: "#7E7E9C",

		// Upstream crystalBlue #7E9CD8 is 4.31:1 on `elevated`, and solving at
		// the 4.5 floor leaves no room for a darker `accentActive`. Lifted to
		// 5.25:1 so all three accent states separate.
		accent: "#94ADDF",
		accentHover: "#AFC2E7",
		accentActive: "#83A0DA",
		accentWash: "#2A2D3A",
		onAccent: "#1F1F28",

		// Upstream springGreen.
		success: "#98BB6C",
		successWash: "#2A2D2E",
		successBorder: "#98BB6C",

		// Upstream carpYellow.
		warning: "#E6C384",
		warningWash: "#343032",
		warningBorder: "#E6C384",

		// Upstream peachRed #FF5D62 is 3.94:1 on `elevated`; lifted on-hue.
		danger: "#FF7478",
		dangerWash: "#32262F",
		dangerBorder: "#FF7478",

		// Upstream waveAqua2 #7AA89F clears the floor at 4.47:1 on `elevated`,
		// a hair under 4.5; nudged on-hue to 4.53:1.
		info: "#7CA9A0",
		infoWash: "#2C3238",
		infoBorder: "#7CA9A0",

		overlayShadow: "0 12px 32px -12px rgb(12 12 17 / 0.72)",
		scrim: "rgb(12 12 17 / 0.62)",
	},
};
