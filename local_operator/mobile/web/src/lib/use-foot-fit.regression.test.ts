// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { footFitLevel } from "./use-foot-fit";

describe("crowded short-footer geometry", () => {
	it("never leaves a reading wider than the remaining row", () => {
		for (const available of [110, 150, 190]) {
			for (const queued of [0, 60]) {
				const level = footFitLevel({ available, gap: 4, fixed: [44, ...(queued ? [queued] : [])], model: 44, full: 96, short: 44 });
				const used = [44, ...(queued ? [queued] : []), ...(level >= 3 ? [] : [44]), ...(level === 0 ? [96] : level === 1 ? [44] : [])];
				const width = used.reduce((a, b) => a + b, 0) + 4 * (used.length - 1);
				expect(width).toBeLessThanOrEqual(available + 0.5);
			}
		}
	});
});
