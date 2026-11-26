/**
 * Derived categorical chart colours: `--lo-chart-1 … --lo-chart-7`.
 *
 * VENDORED. The same pure function ships in three repositories — the desktop
 * theme engine (`src/shared/theme-engine/chart-palette.mjs` in
 * local-operator-ui), this relay web client, and the React Native app — and
 * each pins the same golden vectors in a unit test
 * (`src/lib/chart-palette.test.ts` here). Change it in all three or not at
 * all: a chart that is blue/orange on the desktop and blue/teal on the phone
 * for the same answer reads as two different datasets.
 *
 * WHY DERIVED, NOT AUTHORED. Before this, a pie's slices were the accent plus
 * the semantic roles (info/success/warning) or shades of the accent, which on
 * most themes are neighbouring hues — the operator could not tell slices
 * apart. Authoring seven extra colours per palette would put 7 × 27 hand-picked
 * values beside the role contract with nothing enforcing them. Instead every
 * palette gets them from one algorithm (inline-chart spec §4):
 *
 *   1. Base OKLCH hues in priority order: 255, 60, 185, 345, 95, 300, 145, 25.
 *   2. If the accent is chromatic (C ≥ 0.04), drop the base hue nearest the
 *      accent's hue when it is within 40° — slot 1 is the accent already.
 *   3. Dark theme ⇔ OKLCH L(canvas) < 0.5. Derived slot i (0-based) starts at
 *      L = dark [0.80, 0.70][i % 2] / light [0.50, 0.60][i % 2], C = 0.15
 *      reduced in 0.005 steps until in sRGB gamut, then L steps by 0.01 away
 *      from the ground until contrast ≥ 3.05:1 against EVERY ground.
 *   4. Greedy: slot 2…7 takes the remaining hue maximising its minimum
 *      distance to already-chosen slots, distance = min(ΔE00 normal, ΔE00
 *      under deuteranopia, ΔE00 under protanopia) with the Machado 2009
 *      severity-1 matrices on linear sRGB. Ties keep base order.
 *
 * Pure and dependency-free so the build-time theme generator
 * (`generate-theme-css.mjs`) and a vitest can both import it.
 */

const BASE_HUES = [255, 60, 185, 345, 95, 300, 145, 25];
const SLOTS = 7;
const MIN_CONTRAST = 3.05;

// ---- sRGB <-> linear ------------------------------------------------------

const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const fromLinear = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/** `#rrggbb` → linear-light [r, g, b] in 0..1. */
function hexToLinear(hex) {
	const n = Number.parseInt(hex.replace(/^#/, ""), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => toLinear(v / 255));
}

/** Linear-light [r, g, b] → `#RRGGBB` (clamped, 8-bit rounded). */
function linearToHex(rgb) {
	return `#${rgb
		.map((c) => Math.round(Math.min(1, Math.max(0, fromLinear(c))) * 255))
		.map((v) => v.toString(16).padStart(2, "0"))
		.join("")
		.toUpperCase()}`;
}

// ---- OKLab / OKLCH (Ottosson 2020) ---------------------------------------

function linearToOklab([r, g, b]) {
	const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
	const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
	const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
	return [
		0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
		1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
		0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
	];
}

function oklchToLinear(L, C, h) {
	const a = C * Math.cos((h * Math.PI) / 180);
	const b = C * Math.sin((h * Math.PI) / 180);
	const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
	const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
	const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
	return [
		4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
		-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
		-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
	];
}

function hexToOklch(hex) {
	const [L, a, b] = linearToOklab(hexToLinear(hex));
	return { L, C: Math.hypot(a, b), h: ((Math.atan2(b, a) * 180) / Math.PI + 360) % 360 };
}

const inGamut = (rgb) => rgb.every((c) => c >= -1e-9 && c <= 1 + 1e-9);

// ---- WCAG contrast --------------------------------------------------------

function luminance(hex) {
	const [r, g, b] = hexToLinear(hex);
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrast(a, b) {
	const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
	return (hi + 0.05) / (lo + 0.05);
}

// ---- CIELAB (D65) and ΔE00 -----------------------------------------------

function linearToLab([r, g, b]) {
	const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
	const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
	const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
	const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
	const [fx, fy, fz] = [f(x), f(y), f(z)];
	return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** CIEDE2000 colour difference (Sharma, Wu & Dalal 2005). */
export function deltaE00([L1, a1, b1], [L2, a2, b2]) {
	const rad = Math.PI / 180;
	const C1 = Math.hypot(a1, b1);
	const C2 = Math.hypot(a2, b2);
	const Cbar = (C1 + C2) / 2;
	const G = 0.5 * (1 - Math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)));
	const a1p = (1 + G) * a1;
	const a2p = (1 + G) * a2;
	const C1p = Math.hypot(a1p, b1);
	const C2p = Math.hypot(a2p, b2);
	const hp = (b, a) => (a === 0 && b === 0 ? 0 : ((Math.atan2(b, a) / rad) + 360) % 360);
	const h1p = hp(b1, a1p);
	const h2p = hp(b2, a2p);
	const dLp = L2 - L1;
	const dCp = C2p - C1p;
	let dhp = 0;
	if (C1p * C2p !== 0) {
		dhp = h2p - h1p;
		if (dhp > 180) dhp -= 360;
		else if (dhp < -180) dhp += 360;
	}
	const dHp = 2 * Math.sqrt(C1p * C2p) * Math.sin((dhp / 2) * rad);
	const Lbp = (L1 + L2) / 2;
	const Cbp = (C1p + C2p) / 2;
	let hbp = h1p + h2p;
	if (C1p * C2p !== 0) {
		if (Math.abs(h1p - h2p) > 180) hbp += h1p + h2p < 360 ? 360 : -360;
		hbp /= 2;
	}
	const T = 1 - 0.17 * Math.cos((hbp - 30) * rad) + 0.24 * Math.cos(2 * hbp * rad)
		+ 0.32 * Math.cos((3 * hbp + 6) * rad) - 0.2 * Math.cos((4 * hbp - 63) * rad);
	const dTheta = 30 * Math.exp(-(((hbp - 275) / 25) ** 2));
	const Rc = 2 * Math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7));
	const Sl = 1 + (0.015 * (Lbp - 50) ** 2) / Math.sqrt(20 + (Lbp - 50) ** 2);
	const Sc = 1 + 0.045 * Cbp;
	const Sh = 1 + 0.015 * Cbp * T;
	const Rt = -Math.sin(2 * dTheta * rad) * Rc;
	return Math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh));
}

// ---- Colour-vision deficiency simulation (Machado, Oliveira & Fernandes 2009, severity 1.0)

const PROTANOPIA = [
	[0.152286, 1.052583, -0.204868],
	[0.114503, 0.786281, 0.099216],
	[-0.003882, -0.048116, 1.051998],
];
const DEUTERANOPIA = [
	[0.367322, 0.860646, -0.227968],
	[0.280085, 0.672501, 0.047413],
	[-0.01182, 0.04294, 0.968881],
];

const simulate = (matrix, rgb) =>
	matrix.map((row) => Math.min(1, Math.max(0, row[0] * rgb[0] + row[1] * rgb[1] + row[2] * rgb[2])));

/** Perceptual distance that survives the two common red-green deficiencies. */
export function cvdDistance(hexA, hexB) {
	const a = hexToLinear(hexA);
	const b = hexToLinear(hexB);
	return Math.min(
		deltaE00(linearToLab(a), linearToLab(b)),
		deltaE00(linearToLab(simulate(DEUTERANOPIA, a)), linearToLab(simulate(DEUTERANOPIA, b))),
		deltaE00(linearToLab(simulate(PROTANOPIA, a)), linearToLab(simulate(PROTANOPIA, b))),
	);
}

// ---- The derivation -------------------------------------------------------

/** One derived slot colour for `hue` at 0-based derived index `i`. */
function deriveSlot(hue, i, dark, grounds) {
	let L = dark ? [0.8, 0.7][i % 2] : [0.5, 0.6][i % 2];
	const step = dark ? 0.01 : -0.01;
	for (let guard = 0; guard < 100; guard++) {
		let C = 0.15;
		while (C > 0 && !inGamut(oklchToLinear(L, C, hue))) C = Math.round((C - 0.005) * 1000) / 1000;
		const hex = linearToHex(oklchToLinear(L, Math.max(0, C), hue));
		if (grounds.every((g) => contrast(hex, g) >= MIN_CONTRAST)) return hex;
		L = Math.round((L + step) * 100) / 100;
		if (L <= 0 || L >= 1) return hex;
	}
	throw new Error(`chart palette: no slot for hue ${hue}`);
}

/**
 * @param {{ grounds: string[]; accent: string }} input `#rrggbb` grounds
 *   (canvas first) the marks sit on, and the palette's accent.
 * @returns {string[]} seven `#RRGGBB` colours; [0] is the accent itself.
 */
export function chartPalette({ grounds, accent }) {
	const dark = hexToOklch(grounds[0]).L < 0.5;
	const accentLch = hexToOklch(accent);
	let hues = [...BASE_HUES];
	if (accentLch.C >= 0.04) {
		const gap = (h) => Math.min(Math.abs(h - accentLch.h), 360 - Math.abs(h - accentLch.h));
		const nearest = hues.reduce((best, h) => (gap(h) < gap(best) ? h : best));
		if (gap(nearest) <= 40) hues = hues.filter((h) => h !== nearest);
	}
	const chosen = [accent.toUpperCase()];
	while (chosen.length < SLOTS && hues.length) {
		const i = chosen.length - 1;
		let best = null;
		for (const hue of hues) {
			const hex = deriveSlot(hue, i, dark, grounds);
			const distance = Math.min(...chosen.map((c) => cvdDistance(hex, c)));
			if (!best || distance > best.distance) best = { hue, hex, distance };
		}
		chosen.push(best.hex);
		hues = hues.filter((h) => h !== best.hue);
	}
	return chosen;
}
