/**
 * Types for `chart-palette.mjs`.
 *
 * The module is plain ESM so the theme generator (`node
 * scripts/generate-theme-css.mjs`) runs it with no build step; the golden
 * vector test (`src/lib/chart-palette.test.ts`) imports the SAME file rather
 * than a copy, and this declaration is what makes that import typed.
 */

/** WCAG 2.x contrast ratio of two `#rrggbb` colours. */
export declare function contrast(a: string, b: string): number;
/** CIEDE2000 between two CIELAB triples. */
export declare function deltaE00(a: readonly number[], b: readonly number[]): number;
/** min(ΔE00 normal, deuteranopia, protanopia) between two `#rrggbb` colours. */
export declare function cvdDistance(a: string, b: string): number;
/** Seven `#RRGGBB` chart colours; [0] is the accent (inline-chart spec §4). */
export declare function chartPalette(input: { grounds: string[]; accent: string }): string[];
