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
// So the table below is the TUI's own table, and the sweep after it is the
// property the callers depend on: BOUNDED AT SIX CELLS over the whole domain.
// Nothing here asserts pixels — jsdom has no layout; the reserved slot is
// pinned structurally in `components/working-line.test.tsx` and the geometry is
// in the browser frames recorded on the PR.
import { describe, expect, it } from "vitest";
import { formatElapsed } from "./format";

describe("formatElapsed", () => {
	it("matches the TUI's format_duration at every branch", () => {
		// Values verified against the Python by running it directly:
		// .venv/bin/python -c "from local_operator.tui.widgets.tool_card import format_duration as f; ..."
		const cases: [number, string][] = [
			[0, "0s"],
			[0.4, "0s"], // sub-second work still leaves a mark
			[0.9, "0s"],
			[2.9, "2s"],
			[45, "45s"],
			[59, "59s"],
			[60, "1m"], // a whole minute drops the seconds
			[61, "1m1s"],
			[135, "2m15s"],
			[3599, "59m59s"],
			[3600, "1h"],
			[3661, "1h1m"],
			[86399, "23h59m"],
			[86400, "1d"],
			[362400, "4d4h"],
			[3602400, "41d16h"],
			[100 * 86400, "100d+"], // the cap names the bound it fired at
			[1000 * 86400, "100d+"],
		];
		for (const [seconds, expected] of cases) {
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
