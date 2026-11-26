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
	it("targets 3 ticks at the 180px phone plot and 5 at 240", () => {
		expect(tickTarget(180)).toBe(3);
		expect(tickTarget(240)).toBe(5);
		expect(niceTicks(84.04, 153.66, 3).values.length).toBeLessThanOrEqual(4);
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
