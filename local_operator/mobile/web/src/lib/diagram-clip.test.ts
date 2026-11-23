import { describe, expect, it } from "vitest";
import { inlineMaxHeight, mustClamp } from "./diagram-clip";

describe("inline diagram clamp rule", () => {
	it("shows the 6-node phone flowchart (542 px with padding) whole", () => {
		// Measured with Pele 1.13.0 at <= 600 px width: 232 x 526 drawing + 16 px padding.
		expect(mustClamp(542, 16)).toBe(false);
		expect(inlineMaxHeight(false)).toBe("46rem");
	});

	it("tolerates a modest overflow of the cap, then clamps a runaway drawing", () => {
		expect(mustClamp(46 * 16, 16)).toBe(false);
		expect(mustClamp(46 * 16 + 1, 16)).toBe(true);
		expect(mustClamp(3000, 16)).toBe(true);
		expect(inlineMaxHeight(true)).toBe("40rem");
	});

	it("scales with the root font size", () => {
		expect(mustClamp(46 * 20, 20)).toBe(false);
		expect(mustClamp(46 * 20 + 1, 20)).toBe(true);
	});
});
