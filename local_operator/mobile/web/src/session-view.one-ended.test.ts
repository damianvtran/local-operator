/**
 * ONE "ENDED" PER SCREEN (review round 3, R3-5).
 *
 * The header (its dot, sheet and ⋮ Resume) and the composer (its placeholder,
 * its ended send-failure copy and its aborted-turn pill) must agree on whether
 * the session has ended. They read the same `projection`/`connected` from one
 * `useProjection`, so two `sessionHealth` calls could not disagree today, but
 * the round-2 fix was described as "derived once" while it was derived twice,
 * and a second derivation is the seam a future edit splits (one of them gains
 * a condition the other does not). So the screen derives it ONCE and passes
 * that one value to both, and this pins it in source: happy-dom renders cannot
 * tell one call from two when both return the same answer.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const SOURCE = readFileSync(join(__dirname, "screens", "session-view.tsx"), "utf8");
/** Calls only — not the import line or prose in comments. */
const CALLS = SOURCE.split("\n").filter((line) => !/^\s*(\/\/|\/?\*|import\b)/.test(line) && /\bsessionHealth\(/.test(line));

describe("the session screen derives `ended` once (R3-5)", () => {
	it("calls sessionHealth exactly once", () => {
		expect(CALLS).toHaveLength(1);
	});

	it("hands that one value to the header and the composer", () => {
		expect(SOURCE).toMatch(/<Header\b[^>]*\bended=\{ended\}/);
		expect(SOURCE).toMatch(/<Composer[\s\S]*?\bended=\{ended\}/);
	});
});
