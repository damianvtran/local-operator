// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { footFitLevel } from "./use-foot-fit";

describe("crowded short-footer geometry", () => {
	it("never leaves a reading wider than the remaining row", () => {
		for (const available of [110, 150, 190]) {
			for (const queued of [0, 60]) {
				const level = footFitLevel({ available, gap: 4, fixed: [44], queued, model: 44, full: 96, short: 44 });
				const used = [44, ...(queued && level < 4 ? [queued] : []), 44, ...(level === 0 ? [96] : level === 1 ? [44] : [])];
				const width = used.reduce((a, b) => a + b, 0) + 4 * (used.length - 1);
				expect(width).toBeLessThanOrEqual(available + 0.5);
			}
		}
	});
	it("seats the whole count or moves it, at every width (design D8)", () => {
		/* The count is either whole in the row or whole in the ⋮ menu; there is
		   no level at which it is in the row with less than its intrinsic width. */
		for (let available = 92; available <= 360; available += 0.5) {
			for (const queued of [53, 58.6, 70]) {
				const level = footFitLevel({ available, gap: 4, fixed: [44], queued, model: 44, full: 88.5, short: 44.3 });
				if (level === 4) {
					expect(44 + queued + 44 + 8).toBeGreaterThan(available + 0.5);
					continue;
				}
				const used = [44, queued, 44, ...(level === 0 ? [88.5] : level === 1 ? [44.3] : [])];
				expect(used.reduce((a, b) => a + b, 0) + 4 * (used.length - 1)).toBeLessThanOrEqual(available + 0.5);
			}
		}
	});
});
