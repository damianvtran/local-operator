// @vitest-environment happy-dom
//
// The reserve that keeps a rung from hiding history (round 2, U23 = D7). The
// session view's state ladder is an overlay, so the column never jumps — and
// the price used to be that the first rows were painted over, unreachably so
// on a transcript that does not scroll (measured headless: 28 of the first
// row's 35px at 390, 38 of 56px at 320). The screen measures the rung and
// hands its height to the transcript as `topInset`; what is pinned HERE is the
// transcript's half of that contract: the reserve is the scroller's FIRST
// child (above the load indicator, the "show N more" control and every row),
// it carries the height it was given, and NOTHING is rendered when nothing
// overlays the scroller. Round 3 added the other half of the same contract:
// the scroller keeps native scroll anchoring OFF, so the follow effect is the
// only hand on `scrollTop` and the platform cannot double-compensate the
// reserve's insertion (measured then: a 53px slide per rung with anchoring on,
// 0.4px with the opt-out). Round 4 is why that hand has to cover MORE than the
// reserve: with anchoring off, the window's own row removal at the cap and the
// older-page prepend are the same hand's job too — the follow tracks every row
// and holds the reader's (its arithmetic is pinned in `lib/scroll-follow.test`,
// the rendered geometry in the headless scenes). The rendered geometry itself
// is the headless captures' lane, and the round-3 mid-history scene measures exactly that.
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(cleanup);

function entry(over: Partial<TranscriptEntry>): TranscriptEntry {
	return {
		id: "e1",
		kind: "notice",
		text: "",
		tool_call_id: "",
		tool_name: "",
		tool_state: "done",
		summary: "",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: true,
		...over,
	};
}

describe("the rung's height reserved inside the scroller", () => {
	it("renders the reserve above every reachable row", () => {
		const { container } = render(
			<Transcript
				pid="1"
				entries={[entry({ id: "a", kind: "user", text: "prompt", final: false })]}
				topInset={52}
			/>,
		);
		const scroller = container.querySelector("div.lo-scroll");
		const spacer = container.querySelector<HTMLElement>("[data-scroll-top-inset]");
		const row = container.querySelector("[data-completion-anchor]");
		expect(spacer).toBeTruthy();
		expect(spacer?.style.height).toBe("52px");
		expect(row).toBeTruthy();
		// First child of the scroller, and before the first row.
		expect(scroller?.firstElementChild).toBe(spacer);
		const children = Array.from(scroller?.children ?? []);
		expect(children.indexOf(spacer as Element)).toBeLessThan(
			children.indexOf(row as Element),
		);
	});

	it("keeps native scroll anchoring off, so the follow is the only hand on scrollTop", () => {
		/* Round 3: with anchoring left on, the browser compensates the reserve's
		   insertion for the same mid-history reader the follow effect holds —
		   measured as a 53px slide per rung, 0.4px with this opt-out. The class
		   IS the mechanism (no JS), so pinning it here pins the single-hand
		   contract; the rendered before/after is the round's headless scene. */
		const { container } = render(
			<Transcript
				pid="1"
				entries={[entry({ id: "a", kind: "user", text: "prompt", final: false })]}
				topInset={52}
			/>,
		);
		const scroller = container.querySelector("div.lo-scroll");
		expect(scroller?.className).toContain("[overflow-anchor:none]");
	});

	it("renders no reserve when nothing overlays the scroller", () => {
		const { container } = render(
			<Transcript
				pid="1"
				entries={[entry({ id: "a", kind: "user", text: "prompt", final: false })]}
			/>,
		);
		expect(container.querySelector("[data-scroll-top-inset]")).toBeNull();
	});
});
