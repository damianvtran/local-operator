// The follow decision itself (rounds 3-4). The rendered geometry is the
// headless scenes' lane; what is pinned HERE is the arithmetic those scenes
// depend on: the delta applies, both ends stay inside the scroller, and the
// two positions that are NOT "hold this row" — the tail (the browser's clamp
// already followed the shrink) and the top with the reserve's edge (the
// reveal) — take no write. The tail rule is the round-4 regression
// (reviewer MAJOR 1 = UX U29: clearing the rung at the tail subtracted the
// reserve's height a second time and left the newest row under the fold).
import { describe, expect, it } from "vitest";
import { followScrollTop } from "./scroll-follow";

const base = { domDelta: 0, scrollTop: 500, max: 1000, reserveEdge: false };

describe("the transcript's one follow write", () => {
	it("holds the reader's row by the DOM delta", () => {
		// A rung appears (+61 above) or a row is evicted (-92): both mid-history.
		expect(followScrollTop({ ...base, domDelta: 61 })).toBe(561);
		expect(followScrollTop({ ...base, domDelta: -92 })).toBe(408);
	});

	it("stays inside the scroller", () => {
		expect(followScrollTop({ ...base, domDelta: -900 })).toBe(0);
		expect(followScrollTop({ ...base, scrollTop: 990, max: 1000, domDelta: 40 })).toBe(1000);
	});

	it("writes nothing when the DOM did not move the reader's row", () => {
		// Appends below the reader, a pure scroll: nothing to hold.
		expect(followScrollTop({ ...base, domDelta: 0 })).toBeNull();
		// A transcript that does not scroll: the reveal is content motion.
		expect(followScrollTop({ ...base, max: 0, domDelta: 61 })).toBeNull();
	});

	it("leaves the tail alone: the browser's clamp already followed", () => {
		// The rung clears while the reader sits at the bottom: clearing shrinks
		// max by 61 and the browser clamps scrollTop to it, so the delta that
		// arrives here is the clamp's own movement — subtracting it again would
		// leave the newest row 61px under the fold.
		expect(followScrollTop({ ...base, scrollTop: 939, max: 939, domDelta: -61 })).toBeNull();
		// Same rule within a pixel (fractional heights).
		expect(followScrollTop({ ...base, scrollTop: 938.4, max: 939, domDelta: -61 })).toBeNull();
		// And it does NOT swallow a genuine hold just above the tail: 2px up,
		// the reader's row is what the write is for.
		expect(followScrollTop({ ...base, scrollTop: 937, max: 939, domDelta: -61 })).toBe(876);
	});

	it("reads the reserve's edge at the top as the reveal", () => {
		// At scrollTop 0 the rung appearing must not be compensated: the first
		// row has to slide out from under the strip.
		expect(followScrollTop({ ...base, scrollTop: 0, domDelta: 61, reserveEdge: true })).toBeNull();
		// Content that is NOT the reserve still holds its row at the top (the
		// older-rows prepend), and the write has room to move down.
		expect(followScrollTop({ ...base, scrollTop: 0, domDelta: 30 })).toBe(30);
	});
});
