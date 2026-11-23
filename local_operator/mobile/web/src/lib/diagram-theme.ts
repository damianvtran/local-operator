/**
 * Pele's colours and fonts, taken from the relay's own theme variables.
 *
 * Pele paints every diagram with `var(--pele-*, fallback)`, so setting those
 * variables on the element that holds the SVG is the whole theme: a light/dark
 * flip or a custom palette recolours drawn diagrams in place, with no redraw.
 * The right-hand sides are `--lo-*`, the per-theme variables
 * `themes.generated.css` defines for every shipped palette, not Tailwind's
 * `--color-*` aliases: those are only emitted when a utility class uses them.
 *
 * SERIES COLOURS are solid theme roles in the order info, success, warning,
 * danger, then accent, then the first three again. The relay palette has no
 * second accent, and the accent is NOT first because it is not distinct from
 * the other roles in six shipped themes (accent equals info in dune, neon,
 * obsidian and radient, and equals success in everforest and monokai). The
 * first four roles used here differ by at least CIE76 dE 19.9 in all 31
 * palettes (`diagram-theme.test.ts` re-measures it), so a pie or branch chart
 * of up to four categories is always readable; a fifth is the accent, which
 * may repeat one of them. Sections of a `journey` stay one neutral colour (Pele
 * has no section-fill variable) and `click` callbacks are ignored.
 */
import type { CSSProperties } from "react";

export const PELE_COLOUR_ROLES = {
	"--pele-bg": "--lo-surface",
	"--pele-fg": "--lo-ink",
	"--pele-muted": "--lo-ink-muted",
	"--pele-line": "--lo-ink-muted",
	"--pele-surface": "--lo-elevated",
	"--pele-surface-alt": "--lo-sunken",
	"--pele-border": "--lo-border-control",
	"--pele-accent": "--lo-accent",
	"--pele-critical": "--lo-danger",
	"--pele-error": "--lo-danger",
	"--pele-series-1": "--lo-info",
	"--pele-series-2": "--lo-success",
	"--pele-series-3": "--lo-warning",
	"--pele-series-4": "--lo-danger",
	"--pele-series-5": "--lo-accent",
	"--pele-series-6": "--lo-info",
	"--pele-series-7": "--lo-success",
	"--pele-series-8": "--lo-warning",
} as const satisfies Record<string, string>;

/** Fonts come from Tailwind's `@theme` (`--font-sans`, `--font-mono`), which the base layer always emits. */
export const PELE_FONT_ROLES = {
	"--pele-font": "--font-sans",
	"--pele-font-mono": "--font-mono",
} as const satisfies Record<string, string>;

/** Base size of diagram labels, px: the transcript's `text-body` (0.875rem). */
export const PELE_FONT_SIZE = 14;

const FALLBACK_SANS = "system-ui, sans-serif";
const FALLBACK_MONO = "ui-monospace, monospace";

/** The style that themes a diagram: every `--pele-*` points at the app's variable. */
export const PELE_THEME_STYLE: CSSProperties = Object.fromEntries(
	Object.entries({ ...PELE_COLOUR_ROLES, ...PELE_FONT_ROLES }).map(([name, role]) => [name, `var(${role})`]),
) as CSSProperties;

export type RoleResolver = (name: string) => string;

/** Reads one CSS variable off the page root. */
export const resolveRootVariable: RoleResolver = (name) =>
	getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function diagramFonts(resolve: RoleResolver = resolveRootVariable): { sans: string; mono: string } {
	return {
		sans: resolve("--font-sans") || FALLBACK_SANS,
		mono: resolve("--font-mono") || FALLBACK_MONO,
	};
}

/**
 * The `--pele-*` declarations with every variable resolved to its current
 * value, for a saved SVG that must keep its colours outside the app. A value
 * that carries a quote or a semicolon would break out of the `style` attribute,
 * so the resolved text is cleaned and an empty one is skipped.
 */
export function resolvedThemeDeclarations(resolve: RoleResolver = resolveRootVariable): Array<[string, string]> {
	const out: Array<[string, string]> = [];
	for (const [name, role] of Object.entries({ ...PELE_COLOUR_ROLES, ...PELE_FONT_ROLES })) {
		const value = resolve(role).replace(/["';{}<>]/g, "").trim();
		if (value !== "") out.push([name, value]);
	}
	return out;
}
