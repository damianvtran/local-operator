// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { footFitLevel } from "./use-foot-fit";

describe("the composer's footer fit priority", () => {
	const base = { gap: 4, fixed: [44], model: 44, full: 96, short: 44 };
	it("keeps the full reading when the whole line fits", () => {
		expect(footFitLevel({ ...base, available: 192 })).toBe(0);
	});
	it("compacts the reading before sacrificing the model", () => {
		expect(footFitLevel({ ...base, available: 140 })).toBe(1);
	});
	it("yields the reading before the model", () => {
		expect(footFitLevel({ ...base, available: 100 })).toBe(2);
	});
	it("yields the model only when its floor cannot fit", () => {
		expect(footFitLevel({ ...base, available: 80 })).toBe(3);
	});
	it("accounts for a queued count as fixed content", () => {
		expect(footFitLevel({ ...base, fixed: [44, 60], available: 156 })).toBe(2);
	});
});
