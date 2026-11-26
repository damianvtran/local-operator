import { describe, expect, it } from "vitest";
import {
	PERCENT_TICKS, alignedTicks, axisLabel, donutDiameter, estimateWidth, exactValue, foldSlices, niceTicks,
	plotHeight, relaxLabels, stepIndex, thinLabels, tickFormatter, tickTarget, valueDomain, withUnit,
} from "./chart-geometry";

describe("sizing (spec §1)", () => {
	it("derives the plot height from the measured width", () => {
		expect(plotHeight(760)).toBe(240);
		expect(plotHeight(360)).toBe(180);
		expect(plotHeight(600)).toBe(204);
		expect(plotHeight(2000)).toBe(240);
	});
	it("draws the donut at 200 wide and min(180, W − 32) narrow", () => {
		expect(donutDiameter(760)).toBe(200);
		expect(donutDiameter(390)).toBe(180);
		expect(donutDiameter(200)).toBe(168);
	});
});

describe("nice ticks (spec §2)", () => {
	it("fits the operator's 84.04–153.66 USD series to 80…160 without a 0", () => {
		const ticks = niceTicks(...valueDomain([84.04, 101.2, 140.61, 153.66], "line"), tickTarget(240));
		expect(ticks.values).toEqual([80, 100, 120, 140, 160]);
	});
	it("never targets fewer than 4 ticks, so the 180px phone plot also gets 80…160 (X1)", () => {
		expect(tickTarget(180)).toBe(4);
		expect(tickTarget(240)).toBe(5);
		expect(tickTarget(120)).toBe(4);
		expect(tickTarget(400)).toBe(6);
		// The operator's series at both plot heights: 360px phone and 760px column.
		const domain = valueDomain([84.04, 101.2, 140.61, 153.66], "line");
		for (const width of [360, 760]) {
			expect(niceTicks(...domain, tickTarget(plotHeight(width))).values).toEqual([80, 100, 120, 140, 160]);
		}
	});
	it("prints tiny-value ticks in scientific notation with unique labels (R7)", () => {
		const ticks = niceTicks(1e-21, 3e-21, 4);
		const labels = ticks.values.map(tickFormatter("en", ticks));
		expect(new Set(labels).size).toBe(labels.length);
		expect(labels.every((label) => /E-2\d$/.test(label))).toBe(true);
		expect(labels.join(" ")).not.toContain("0.0000");
	});
	it("keeps 0 in every bar domain and on a line that reaches near it", () => {
		expect(niceTicks(...valueDomain([12, 40], "bar"), 5).values[0]).toBe(0);
		expect(niceTicks(...valueDomain([-3, 9], "bar"), 3).values).toEqual([-5, 0, 5, 10]);
		expect(valueDomain([5, 30], "line")).toEqual([0, 30]);
	});
	it("pads flat data instead of collapsing the scale", () => {
		expect(valueDomain([50, 50], "line")).toEqual([45, 55]);
		expect(niceTicks(0, 0, 5).values).toContain(0);
	});
	it("prints tick decimals from the step, never float noise", () => {
		const ticks = niceTicks(0.1, 0.7, 5);
		expect(ticks.values.every((v) => String(v).length <= 4)).toBe(true);
		expect(tickFormatter("en", ticks)(ticks.values[1])).toMatch(/^0\.\d+$/);
	});
	it("pins percent mode to 0–100% at 25% steps", () => {
		expect(PERCENT_TICKS.values).toEqual([0, 25, 50, 75, 100]);
	});
	it("gives the right axis the left axis's tick count so gridlines align", () => {
		const left = niceTicks(0, 153, 5);
		const right = alignedTicks(2.5, 7.8, left.values.length);
		expect(right.values).toHaveLength(left.values.length);
		expect(right.max).toBeGreaterThanOrEqual(7.8);
		expect(right.min).toBeLessThanOrEqual(2.5);
	});
});

describe("number format (spec §2)", () => {
	it("compacts ticks from 10 000 but keeps tooltips exact and grouped", () => {
		const ticks = niceTicks(0, 12_000, 5);
		expect(tickFormatter("en", ticks)(12_000)).toBe("12K");
		expect(exactValue("en", 12_345.67)).toBe("12,345.67");
		expect(exactValue("fr", 1234.5).replace(/\s/g, " ")).toBe("1 234,5");
	});
	it("picks the fewest decimals that keep every compact tick distinct (X3)", () => {
		const around = niceTicks(10_050, 10_300, 5);
		const a = around.values.map(tickFormatter("en", around));
		expect(new Set(a).size).toBe(a.length);
		expect(a).toEqual(["10.05K", "10.10K", "10.15K", "10.20K", "10.25K", "10.30K"]);
		const million = niceTicks(1_000_000, 1_004_000, 4);
		const b = million.values.map(tickFormatter("en", million));
		expect(b).toEqual(["1.000M", "1.001M", "1.002M", "1.003M", "1.004M"]);
		// Round thousands still read 0K, 50K, 100K: no decimals when none are needed.
		const round = niceTicks(0, 100_000, 3);
		expect(round.values.map(tickFormatter("en", round))).toEqual(["0", "50K", "100K"]);
		// Past 3 decimals, compact cannot tell them apart: full grouped numbers.
		const fine = niceTicks(1_000_000, 1_000_400, 4);
		const c = fine.values.map(tickFormatter("en", fine));
		expect(new Set(c).size).toBe(c.length);
		expect(c[0]).toBe("1,000,000");
	});
	it("joins units with no space for %, ‰, °, °C and °F", () => {
		expect(withUnit("12", "USD")).toBe("12 USD");
		expect(withUnit("12", "%")).toBe("12%");
		expect(withUnit("12", "°C")).toBe("12°C");
		expect(withUnit("12")).toBe("12");
	});
});

describe("x labels (spec §2)", () => {
	it("shows the part before the annotation on the axis", () => {
		expect(axisLabel("Nov 24 · rollout")).toBe("Nov 24");
		expect(axisLabel("Q1")).toBe("Q1");
	});
	it("thins 32 categories at 360px so no two labels overlap and both ends show", () => {
		const labels = Array.from({ length: 32 }, (_, i) => `Nov ${i + 1}`);
		const plot = 360 - 40 - 12;
		const slot = plot / 31;
		const xs = labels.map((_, i) => 40 + i * slot);
		const shown = thinLabels(labels, xs, slot, estimateWidth, true);
		expect(shown[0]).toMatchObject({ index: 0, anchor: "start" });
		expect(shown[shown.length - 1]).toMatchObject({ index: 31, anchor: "end" });
		const boxes = shown.map(({ index, text, anchor }) => {
			const w = estimateWidth(text);
			const left = anchor === "start" ? xs[index] : anchor === "end" ? xs[index] - w : xs[index] - w / 2;
			return [left, left + w];
		});
		for (let i = 1; i < boxes.length; i++) expect(boxes[i][0]).toBeGreaterThan(boxes[i - 1][1]);
	});
	// X2 / QA Q1: every check below rebuilds each box from what the renderer
	// draws (x, text-anchor, measured text), not from thinLabels' own numbers.
	const drawnBoxes = (shown: ReturnType<typeof thinLabels>, measure: (s: string) => number) =>
		shown.map(({ x, anchor, text }) => {
			const w = measure(text);
			const left = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
			return [left, left + w] as const;
		});
	const lineLayout = (n: number, width: number, gutterLeft = 40, right = 12) => {
		const slot = (width - gutterLeft - right) / Math.max(1, n - 1);
		return { slot, xs: Array.from({ length: n }, (_, i) => gutterLeft + i * slot) };
	};
	const expectClean = (labels: string[], width: number, measure: (s: string) => number, edge = true) => {
		const { slot, xs } = edge ? lineLayout(labels.length, width) : (() => {
			const s = (width - 40 - 12) / labels.length;
			return { slot: s, xs: labels.map((_, i) => 40 + s * (i + 0.5)) };
		})();
		const shown = thinLabels(labels, xs, slot, measure, edge, [0, width]);
		const boxes = drawnBoxes(shown, measure);
		expect(shown[0].index).toBe(0);
		expect(shown[shown.length - 1].index).toBe(labels.length - 1);
		for (const [left, right] of boxes) {
			expect(left).toBeGreaterThanOrEqual(-0.5);
			expect(right).toBeLessThanOrEqual(width + 0.5);
		}
		for (let i = 1; i < boxes.length; i++) expect(boxes[i][0] - boxes[i - 1][1]).toBeGreaterThanOrEqual(4 - 1e-6);
		return shown;
	};
	const days = (from: Date, n: number) => Array.from({ length: n }, (_, i) =>
		new Date(from.getTime() + i * 86_400_000).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" }));

	it("keeps QA's Q1 cases disjoint: 32 real days and a 7-day dual at 360/366", () => {
		const october = days(new Date(Date.UTC(2025, 9, 25)), 32);
		expect(october[0]).toBe("Oct 25");
		expect(october[31]).toBe("Nov 25");
		for (const width of [360, 366]) expectClean(october, width, estimateWidth);
		const week = days(new Date(Date.UTC(2025, 10, 16)), 7);
		// Dual: a right-axis gutter narrows the plot on both sides.
		const { slot, xs } = lineLayout(7, 366, 40, 44);
		const shown = thinLabels(week, xs, slot, estimateWidth, true, [0, 366]);
		const boxes = drawnBoxes(shown, estimateWidth);
		for (let i = 1; i < boxes.length; i++) expect(boxes[i][0] - boxes[i - 1][1]).toBeGreaterThanOrEqual(4 - 1e-6);
	});

	it("never overlaps long labels at 360 or 760, truncating the last to the room it has", () => {
		const quarters = Array.from({ length: 32 }, (_, i) => `Week ${i + 1} of the quarterly reporting period`);
		const stamps = Array.from({ length: 32 }, (_, i) => `2025-11-${String(16 + Math.floor(i / 4)).padStart(2, "0")} ${String((i % 4) * 6).padStart(2, "0")}:00 UTC`);
		// A wider face than the estimate too (zh-CN glyphs measure ~1.6×).
		const wide = (s: string) => Array.from(s).length * 10.5;
		for (const measure of [estimateWidth, wide]) {
			for (const width of [360, 760]) {
				for (const labels of [quarters, stamps, quarters.slice(0, 2), stamps.slice(0, 3)]) {
					for (const edge of [true, false]) expectClean(labels, width, measure, edge);
				}
			}
		}
		const shown = expectClean(quarters, 360, estimateWidth);
		expect(shown).toHaveLength(2);
		expect(shown[0].text).toBe("Week 1 of the quarterly reporting period");
		expect(shown[1].text.endsWith("…")).toBe(true);
	});

	it("truncates labels with an ellipsis rather than overlapping or hiding an end", () => {
		const shown = thinLabels(["A very long category name indeed", "Another long one here"], [40, 120], 80, estimateWidth, false);
		expect(shown.map(({ index }) => index)).toEqual([0, 1]);
		expect(shown.every(({ text }) => text.endsWith("…"))).toBe(true);
		const [a, b] = shown.map(({ text }) => estimateWidth(text));
		expect(40 + a / 2).toBeLessThan(120 - b / 2);
	});
});

describe("donut fold (spec §3)", () => {
	it("keeps the largest 6 of 12 and folds 6 into Other, last", () => {
		const data = Array.from({ length: 12 }, (_, i) => [`S${i}`, i + 1] as [string, number]);
		const slices = foldSlices(data);
		expect(slices).toHaveLength(7);
		expect(slices.slice(0, 6).map(({ label }) => label)).toEqual(["S11", "S10", "S9", "S8", "S7", "S6"]);
		expect(slices[6].source).toBe(-1);
		expect(slices[6].folded).toHaveLength(6);
		expect(slices[6].value).toBe(1 + 2 + 3 + 4 + 5 + 6);
	});
	it("does not fold 7 slices, only sorts them", () => {
		const slices = foldSlices([["a", 1], ["b", 3], ["c", 2], ["d", 1], ["e", 1], ["f", 1], ["g", 1]]);
		expect(slices.map(({ label }) => label)).toEqual(["b", "c", "a", "d", "e", "f", "g"]);
	});
	it("relaxes outside labels to at least 18px apart inside the box", () => {
		const ys = relaxLabels([50, 52, 53, 200], 18, 8, 216);
		const sorted = [...ys].sort((a, b) => a - b);
		for (let i = 1; i < sorted.length; i++) expect(sorted[i] - sorted[i - 1]).toBeGreaterThanOrEqual(18);
		expect(Math.max(...ys)).toBeLessThanOrEqual(216);
	});
});

describe("keyboard steps (spec §5)", () => {
	it("moves and clamps with arrows, Home and End", () => {
		expect(stepIndex("ArrowRight", 0, 3)).toBe(1);
		expect(stepIndex("ArrowRight", 2, 3)).toBe(2);
		expect(stepIndex("ArrowLeft", 0, 3)).toBe(0);
		expect(stepIndex("End", 0, 3)).toBe(2);
		expect(stepIndex("Home", 2, 3)).toBe(0);
		expect(stepIndex("Enter", 1, 3)).toBeNull();
	});
});
