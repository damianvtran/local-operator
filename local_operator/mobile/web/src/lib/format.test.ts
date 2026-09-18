// The phone's elapsed-duration wording, pinned to the TUI's.
//
// WHY THIS FILE EXISTS
// --------------------
// `formatElapsed` is a port of `local_operator/tui/widgets/tool_card.py
// ::format_duration`, because both surfaces print the same elapsed time for the
// same work: the phone's band, its tool rows and its subagent rows all read it,
// and the TUI's working block and tool rows read the ported-from function. Two
// spellings of one number is two answers to one question — `2m 15s` beside a
// TUI's `2m15s` for the same phase, which is design round 1's D5 — and a
// formatter that is WIDER than the space reserved for it is how the band's label
// got re-clipped mid-tick (D2) and a row's clock got cut to a shorter, valid
// looking duration (`1000h 40m` → `1000h`, D1).
//
// So the table below is not this suite's opinion: it is the SHARED fixture
// `format.parity.json`, regenerated from the Python by
// `scripts/generate_clock_format_parity.py` and asserted against
// `format_duration` by `tests/unit/mobile/test_tui_bridge.py`. One artifact, two
// suites, one per formatter — so a change to either side fails in its own tree
// rather than diverging with both green (review round 2, MINOR 4). The sweep
// after it is the property the callers depend on: BOUNDED AT SIX CELLS over the
// whole domain. Nothing here asserts pixels — jsdom has no layout; the reserved
// slot is pinned structurally in `components/working-line.test.tsx` and the
// geometry is in the browser frames recorded on the PR.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { formatElapsed } from "./format";

interface ParityFixture {
	cases: [number, string][];
}

/** The generated fixture, read from disk rather than imported: the vitest node
 * environment resolves it the same way whatever the bundler's JSON settings do,
 * and a moved or deleted fixture fails here loudly instead of silently
 * shrinking the table. */
const parity: ParityFixture = JSON.parse(
	readFileSync(join(__dirname, "format.parity.json"), "utf8"),
) as ParityFixture;

describe("formatElapsed", () => {
	it("matches the TUI's format_duration on the shared parity fixture", () => {
		// Every branch, every crossing the form changes at, and the ±3s steps
		// around each crossing, where a rounding difference shows up first.
		expect(parity.cases.length).toBeGreaterThan(40);
		for (const [seconds, expected] of parity.cases) {
			expect(formatElapsed(seconds), `${seconds}s`).toBe(expected);
		}
	});

	it("is bounded at six cells over the whole domain", () => {
		// The property the band's reserved slot and the row's clip depend on: a
		// sweep far past the days cap, including every one-second step across the
		// minute, hour and day boundaries, where the form can change.
		const overlong: string[] = [];
		const check = (seconds: number) => {
			const text = formatElapsed(seconds);
			if (text.length > 6) overlong.push(`${seconds}s -> ${text}`);
		};
		const boundaries = [59, 119, 3599, 3601, 86399, 86401, 8640000];
		for (const boundary of boundaries) {
			for (let seconds = Math.max(0, boundary - 3); seconds <= boundary + 3; seconds++) {
				check(seconds);
			}
		}
		for (let seconds = 0; seconds <= 200 * 86400; seconds += 361) check(seconds);
		expect(overlong).toEqual([]);
	});

	it("withholds rather than printing a duration it cannot trust", () => {
		expect(formatElapsed(Number.NaN)).toBe("");
		expect(formatElapsed(Number.POSITIVE_INFINITY)).toBe("");
		expect(formatElapsed(-1)).toBe("");
	});
});
