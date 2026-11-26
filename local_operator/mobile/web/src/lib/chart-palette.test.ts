/**
 * Golden vectors for the vendored chart palette (inline-chart spec §4).
 *
 * The desktop app, this relay page and the React Native app each carry a copy
 * of the same pure function and pin these EXACT outputs, so the three surfaces
 * draw one answer's chart in the same colours. A change here that is not made
 * in all three is a divergence, not a tweak.
 */
import { describe, expect, it } from "vitest";
import { chartPalette, contrast, cvdDistance } from "../../scripts/chart-palette.mjs";
import { THEMES } from "../theme";

const GOLDEN: [string, string[], string, string][] = [
	["localOperatorDark", ["#1d1b17", "#26221c", "#2a261d"], "#38c96a", "#38C96A #92C1FD #17B6A8 #DCBC33 #AD87ED #FF96D3 #E18528"],
	["localOperatorLight", ["#f2ede3", "#f7f5ee", "#fcfbf7"], "#137742", "#137742 #1762B6 #059488 #9B3876 #977E0D #915006 #8F68CB"],
	["highContrastLight", ["#E6E6E6", "#F2F2F2", "#fafafa"], "#0846C1", "#0846C1 #756208 #159085 #9B3876 #CA5551 #107823 #8F68CB"],
	["dracula", ["#1D1F2B", "#232539", "#232539"], "#C096FC", "#C096FC #DCBC33 #17B6A8 #7BD77F #E18528 #FF96D3 #ED756E"],
];

describe("chartPalette", () => {
	it.each(GOLDEN)("reproduces the shared golden vector for %s", (_, grounds, accent, expected) => {
		expect(chartPalette({ grounds, accent }).join(" ")).toBe(expected);
	});

	// The relay's own palettes have their own grounds (canvas, surface, elevated
	// — the tooltip sits on elevated), so assert the contract, not hexes.
	it.each(THEMES.map(({ id, palette }) => [id, palette] as const))("%s: seven distinct slots at ≥ 3:1 on every ground", (_, palette) => {
		const grounds = [palette.canvas, palette.surface, palette.elevated];
		const slots = chartPalette({ grounds, accent: palette.accent });
		expect(slots).toHaveLength(7);
		expect(slots[0]).toBe(palette.accent.toUpperCase());
		expect(new Set(slots).size).toBe(7);
		for (const slot of slots.slice(1)) for (const ground of grounds) expect(contrast(slot, ground)).toBeGreaterThanOrEqual(3.05);
		// Series use slots 1-4: they must stay apart under red-green deficiency.
		for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) expect(cvdDistance(slots[i], slots[j])).toBeGreaterThan(6);
	});
});
