/**
 * The priority-fit rule (operator finding B): everything that fits on one line
 * shows, the rest collapses behind ONE overflow control, and the selected item
 * is never the one hidden.
 */
import { describe, expect, it } from "vitest";
import { fitPriority } from "./use-priority-fit";

// A 360 px phone's strip: 336 px of content box, three fixed chips of 44/116/96
// px with 8 px gaps (=272), and a 44 px `+N` control.
const BASE = { available: 336, fixed: 272, gap: 8, overflow: 44, must: -1 };

describe("fitPriority", () => {
	it("shows everything when everything fits, with no overflow control", () => {
		expect(fitPriority([40], { ...BASE })).toEqual([0]);
	});

	it("collapses the rest behind one control once the line is full", () => {
		const shown = fitPriority([60, 70, 80], { ...BASE, available: 460 });
		// 460 − 272 − 8 (lead gap) − 44 (+N) − 8 = 128: 60+8 fits, 70+8 does not.
		expect(shown).toEqual([0]);
	});

	it("never hides the selected item, giving up the last visible one instead", () => {
		const shown = fitPriority([60, 70, 50], { ...BASE, available: 460, must: 2 });
		expect(shown).toContain(2);
		expect(shown).not.toContain(1);
	});

	it("shows nothing optional when the fixed chips already fill the line", () => {
		expect(fitPriority([60, 70], { ...BASE, available: 300 })).toEqual([]);
	});

	it("keeps a selected item even when the line has no optional room", () => {
		expect(fitPriority([60, 70], { ...BASE, available: 300, must: 1 })).toEqual([1]);
	});
});
