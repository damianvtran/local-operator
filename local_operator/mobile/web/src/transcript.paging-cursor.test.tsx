// @vitest-environment happy-dom
//
// THE OLDER-PAGE CURSOR (first-paint lane T2, the reach fix's second half).
//
// The transcript pages history with the id of its oldest row. It used
// `visible[0].id` — the oldest row inside the MOUNTED window — and that window
// only grows when the reader taps `show N more loaded`. So once fetch pace
// outran taps the cursor stopped moving and every request returned the SAME
// page: measured on the S6 fixture against the real daemon, 40 taps produced
// 139 requests and 4,680 mounted rows holding 240 distinct ones (twenty copies
// of one page, which the reader then scrolls through as a conversation that
// repeats itself). It could not happen before deep history was reachable — a
// request past the last compaction answered `has_more: false` and the walk
// stopped after one page.
//
// What is pinned here: the SECOND fetch asks for what is older than the page
// the first fetch delivered, not for the same page again.
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { getHistory } from "./api";
import type { TranscriptEntry } from "./types";

const calls: string[] = [];

/** A gate a test can lower AFTER the request has started, so the in-flight
    indicator is observable without racing the fetch (the mock resolves on the
    next microtask otherwise). */
let inFlight: Promise<void> | null = null;

vi.mock("./api", () => ({
	getHistory: vi.fn(async (_pid: string, before: string | null) => {
		calls.push(String(before));
		if (inFlight) await inFlight;
		if (before === "p0") {
			return { entries: [row("older-1"), row("older-2")], has_more: true };
		}
		return { entries: [row("older-3")], has_more: false };
	}),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

function row(id: string): TranscriptEntry {
	return {
		id,
		kind: "assistant",
		text: id,
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
	};
}

/** A full window (`PAGE` rows), so the mounted slice is exactly the tail. */
function window(): TranscriptEntry[] {
	return Array.from({ length: 120 }, (_, i) => row(`p${i}`));
}

afterEach(() => {
	cleanup();
	calls.length = 0;
});

describe("the older-history cursor", () => {
	it("asks for the page after the oldest one held, not the oldest one rendered", async () => {
		const { container } = render(<Transcript pid="s1" entries={window()} />);
		const scroller = container.querySelector(".lo-scroll");
		expect(scroller).not.toBeNull();

		// The reader scrolls to the top: the auto-load fetches one page.
		Object.defineProperty(scroller, "scrollTop", { value: 0, writable: true });
		fireEvent.scroll(scroller as Element);
		await waitFor(() => expect(calls.length).toBe(1));
		expect(calls[0]).toBe("p0");

		// The prepend does NOT move the mounted window, so the rows just fetched
		// sit hidden above it. The reader's next scroll upward REVEALS them
		// (grows the window) instead of fetching another page nobody can see —
		// one mechanism at a time, which is what stopped the cascade that pulled
		// 13 pages (2.24 MB) into a window the reader could not reach.
		fireEvent.scroll(scroller as Element);
		await waitFor(() =>
			expect(container.querySelectorAll("[data-completion-anchor]").length).toBe(122),
		);
		expect(calls.length).toBe(1);

		// Once nothing is held above the reader, the next scroll fetches the page
		// below the rows already held — and the cursor is the oldest row HELD
		// (``older-1``), not the oldest row rendered (``p0``, which the mount
		// already dropped from the window).
		fireEvent.scroll(scroller as Element);
		await waitFor(() => expect(calls.length).toBeGreaterThan(1));
		expect(calls[1]).toBe("older-1");
		expect(getHistory).toHaveBeenCalled();
	});

	it("anchors past a pinned opener when the projection is at its cap", async () => {
		/* The daemon pins the conversation's first user row at the head of a
		   capped projection (``_cap_tail``), and that row is NOT the tail's
		   chronological neighbour: paging from it asks for rows older than an
		   opening turn, so everything between the opener and the tail is never
		   requested. The guard against that compared the transcript's length to
		   ``PAGE`` (120) — a number the daemon never sends — so it never fired on
		   a real projection (measured on S6: 22 user turns and 272 rows missing;
		   S3: 28 and 839). The cap the daemon actually sends is 80. */
		const capped = [row("opener"), ...Array.from({ length: 79 }, (_, i) => row(`tail-${i}`))];
		capped[0] = { ...capped[0], kind: "user" };
		const { container } = render(<Transcript pid="s1" entries={capped} />);
		const scroller = container.querySelector(".lo-scroll");
		Object.defineProperty(scroller, "scrollTop", { value: 0, writable: true });
		fireEvent.scroll(scroller as Element);
		await waitFor(() => expect(calls.length).toBe(1));
		expect(calls[0]).toBe("tail-0");
	});

	it("keeps the idle hairline and the in-flight bar the same height", async () => {
		/* The two top-indicator states differ by their HEIGHT (2px bar, 1px
		   hairline), and that pixel is not cosmetic: every swap moved the rows
		   below it, the anchor-hold wrote `scrollTop` to compensate, and the
		   write re-entered `onScroll` and fetched again. On the S6 deep cell 23
		   of 27 transitions in one open were that oscillation against a rig
		   scrolling once every 300 ms. */
		let release!: () => void;
		inFlight = new Promise<void>((resolve) => {
			release = resolve;
		});
		const { container } = render(<Transcript pid="s1" entries={window()} />);
		const scroller = container.querySelector(".lo-scroll") as HTMLElement;
		Object.defineProperty(scroller, "scrollTop", { value: 0, writable: true });
		fireEvent.scroll(scroller);

		const box = () =>
			Array.from(container.querySelectorAll('[aria-hidden="true"]')).find((el) =>
				el.className.includes("h-0.5"),
			) ?? null;
		await waitFor(() => expect(box()).not.toBeNull());
		const loadingBox = box() as Element;

		release();
		inFlight = null;
		await waitFor(() => expect(calls.length).toBe(1));
		await waitFor(() => expect(box()).not.toBeNull());
		const idleBox = box() as Element;

		// Same box, same height — the hairline is 1px INSIDE a 2px row.
		expect(loadingBox.className).toContain("h-0.5");
		expect(idleBox.className).toContain("h-0.5");
	});
});
