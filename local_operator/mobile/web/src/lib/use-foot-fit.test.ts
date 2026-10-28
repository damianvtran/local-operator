// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { footFitLevel } from "./use-foot-fit";

describe("the composer's footer fit priority", () => {
	const base = { gap: 4, fixed: [44], queued: 0, model: 44, full: 96, short: 44 };
	it("keeps the full reading when the whole line fits", () => {
		expect(footFitLevel({ ...base, available: 192 })).toBe(0);
	});
	it("compacts the reading before sacrificing the model", () => {
		expect(footFitLevel({ ...base, available: 140 })).toBe(1);
	});
	it("yields the reading before the model", () => {
		expect(footFitLevel({ ...base, available: 100 })).toBe(2);
	});
	it("pins the model at its floor only when even that cannot fit", () => {
		expect(footFitLevel({ ...base, available: 80 })).toBe(3);
	});
	it("counts a queued count at its whole width, so the reading yields to it", () => {
		/* 156 px: folder 44 + count 60 + model 44 + 2 gaps = 156 — the count stays
		   whole and the reading goes, never the other way round (design D8). */
		expect(footFitLevel({ ...base, queued: 60, available: 156 })).toBe(2);
	});
	it("moves the count out whole only when the last level cannot seat it", () => {
		expect(footFitLevel({ ...base, queued: 53, available: 149 })).toBe(2);
		expect(footFitLevel({ ...base, queued: 53, available: 148 })).toBe(4);
		/* 320 + mic (the row is 110 px): folder + `2 queued` + the model's floor
		   need 149 px, so the count moves to the ⋮ menu rather than in pieces. */
		expect(footFitLevel({ ...base, queued: 53, available: 110 })).toBe(4);
	});
	it("never reaches level 4 without a count", () => {
		for (let available = 40; available <= 400; available += 0.5) {
			expect(footFitLevel({ ...base, available })).toBeLessThan(4);
		}
	});
	it("tries an unmeasured count before ruling anything out", () => {
		expect(footFitLevel({ ...base, queued: null, available: 110 })).toBe(0);
	});
});
