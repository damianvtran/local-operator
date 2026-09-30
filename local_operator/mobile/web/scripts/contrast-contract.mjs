#!/usr/bin/env node
/**
 * The palette contrast contract, enforced over the shipped palettes.
 *
 *     node scripts/contrast-contract.mjs
 *
 * `pnpm check-themes` runs this after the generated-CSS freshness check, and
 * CI runs `pnpm check-themes`, so a palette that cannot be read cannot ship.
 * The floors are the ported ones from local-operator-ui's
 * `scripts/contrast-contract.mjs` (branding § 3): primary ink at the strong
 * text floor, the two lower ink weights and every tone at the text floor,
 * control borders and `onAccent` at their respective non-text floors. What is
 * NOT ported is the UI app's component inventory — that file asserts hundreds
 * of specific call sites in an Electron shell; this app's surfaces are the
 * phone screens, so the classes below are the ones the mobile call sites
 * actually paint.
 *
 * ## Why the washes are in the class list at all (D3/D4, mobile UX batch 2)
 *
 * An earlier sweep measured the designer's failing pairs: `inkDim` on
 * `dangerWash` and `accentWash` in gruvboxLight / ayuLight / everforestLight
 * (4.33-4.50 against the 4.5 floor) — dim ink is painted ON those warm fills
 * by real rows (the failed tool row's gauge line, the selected quick-pick's
 * dim trailing text, the ask card's option descriptions), so the pair is a
 * text pair, not decoration. The batch-2 audit also caught `accent` on
 * dracula's `elevated` at 4.22 (the pin sheet's ★, the roster's status dots
 * and the theme picker's ✓ all paint accent on elevated sheets); that one is
 * resolved by moving the token rather than exempting the pair, so this file
 * carries no exceptions list — every assertion is a measurement.
 *
 * ## Why the tones are asserted on `elevated` too (D1, mobile UX batch 2 round 1)
 *
 * The pin-refusal and resume-refusal strips paint `text-danger` on
 * `bg-elevated`, and this file asserted the tones only on canvas, surface and
 * their own wash — so the pair the batch's own strips introduced was exactly
 * the one the gate could not see. Measured dark: monokai 3.76:1, dracula
 * 4.11:1, neon 4.43:1 — three of 31 palettes under the text floor, with
 * monokai's pink-on-olive legible-but-wrong. The pair is now asserted for ALL
 * four tones (a gate that lists one tone is one palette change away from the
 * same hole); the three failing palettes were nudged along their own danger
 * lightness — the same "smallest lift that clears" move the batch made for
 * dracula's elevated — until danger-on-elevated clears 4.5 with margin
 * (monokai 3.76 -> 4.63, dracula 4.11 -> 4.62, neon 4.43 -> 4.60; the
 * remaining 28 palettes measured >= 4.5 already and are unchanged).
 *
 * ## What this can and cannot see
 *
 * It parses the palette SOURCE (via `palette-source.mjs`, the same reader the
 * generator uses — a gate that reads the source differently from the
 * generator it gates is not a gate) and asserts over the roles it finds. It
 * cannot see a component that invents its own fill outside the palette, or a
 * pair that is composited at runtime (an opacity, a blend). When a new
 * component paints a role on a ground this file does not know about, the
 * component's review is expected to add the class here — the same rule the UI
 * repo's file states.
 */

import { loadPalettes } from "./palette-source.mjs";

/* WCAG 2.x relative luminance and contrast ratio, over sRGB hex. */
const channel = (c) => {
	const v = c / 255;
	return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};

const luminance = (hex) => {
	const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
	if (!m) throw new Error(`not a six-digit hex colour: "${hex}"`);
	const n = parseInt(m[1], 16);
	const [r, g, b] = [(n >> 16) & 255, (n >> 8) & 255, n & 255];
	return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
};

const contrast = (fg, bg) => {
	const [a, b] = [luminance(fg), luminance(bg)].sort((x, y) => y - x);
	return (a + 0.05) / (b + 0.05);
};

/** The floors, named once: the number a class is asserted against. */
const FLOOR = {
	/** Primary body ink on the page's four grounds. */
	strong: 7,
	/** Every other readable weight, and every tone used as text. */
	text: 4.5,
	/** A control's only edge, and the label on a filled control. */
	nonText: 3,
};

const GROUNDS = ["canvas", "surface", "elevated", "sunken"];

/**
 * The tones and the grounds each is asserted on: the four page grounds for
 * canvas/surface use, plus its own wash — the fill a tone is painted on when
 * it carries state (a failed row, a danger alert, a success chip).
 */
const TONES = ["success", "warning", "danger", "info"];

/**
 * The wash pairs the mobile call sites paint (D3). Each entry is
 * [foreground, wash] with the surfaces named in the comment above.
 */
const WASH_PAIRS = [
	["accent", "accentWash"],
	["ink", "accentWash"],
	["inkDim", "accentWash"],
	["inkMuted", "accentWash"],
	["inkDim", "dangerWash"],
	["inkMuted", "dangerWash"],
];

let assertions = 0;
const failures = [];

const assertPair = (theme, palette, fg, bg, floor, kind) => {
	assertions += 1;
	const got = contrast(palette[fg], palette[bg]);
	if (got < floor) {
		failures.push(
			`${theme}: ${fg} on ${bg} = ${got.toFixed(3)}:1, under the ` +
				`${floor}:1 ${kind} floor`,
		);
	}
};

const palettes = loadPalettes();
if (palettes.length === 0) {
	console.error("No palettes found. Is src/themes/ present?");
	process.exit(1);
}

for (const { id, palette } of palettes) {
	// The four grounds, then the crosses. A missing role is a hard error: the
	// palette contract makes every role literal, so absence means the class
	// below has drifted from the palettes, not that the pair is exempt.
	const missing = [];
	for (const role of [
		...GROUNDS,
		"ink",
		"inkMuted",
		"inkDim",
		"borderControl",
		"accent",
		"accentWash",
		"onAccent",
		"dangerWash",
		...TONES,
		...TONES.map((t) => `${t}Wash`),
	]) {
		if (!(role in palette)) missing.push(role);
	}
	if (missing.length > 0) {
		failures.push(`${id}: palette is missing roles ${missing.join(", ")}`);
		continue;
	}

	for (const ground of GROUNDS) {
		assertPair(id, palette, "ink", ground, FLOOR.strong, "strong text");
		assertPair(id, palette, "inkMuted", ground, FLOOR.text, "text");
		assertPair(id, palette, "inkDim", ground, FLOOR.text, "text");
		assertPair(id, palette, "borderControl", ground, FLOOR.nonText, "control edge");
		assertPair(id, palette, "accent", ground, FLOOR.text, "text");
	}
	assertPair(id, palette, "onAccent", "accent", FLOOR.text, "label-on-fill");
	for (const tone of TONES) {
		assertPair(id, palette, tone, "canvas", FLOOR.text, "text");
		assertPair(id, palette, tone, "surface", FLOOR.text, "text");
		assertPair(id, palette, tone, "elevated", FLOOR.text, "text");
		assertPair(id, palette, tone, `${tone}Wash`, FLOOR.text, "text");
	}
	for (const [fg, bg] of WASH_PAIRS) {
		assertPair(id, palette, fg, bg, FLOOR.text, "text-on-wash");
	}
}

if (failures.length > 0) {
	console.error(
		`Contrast contract violated — ${failures.length} of ${assertions} ` +
			`assertions failed:\n  ${failures.join("\n  ")}\n\n` +
			"Fix the palette (nudge the ink or the ground along its own ramp) " +
			"or move the call site off the failing pair; this file carries no " +
			"exceptions list on purpose (see its header).",
	);
	process.exit(1);
}
console.log(
	`Contrast contract holds: ${assertions} assertions across ${palettes.length} themes.`,
);
