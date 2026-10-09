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

vi.mock("./api", () => ({
	getHistory: vi.fn(async (_pid: string, before: string | null) => {
		calls.push(String(before));
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

		// The prepend does NOT move the mounted window, so a reader who scrolls
		// again before tapping `show N more loaded` must still get the page
		// below the rows already held.
		fireEvent.scroll(scroller as Element);
		await waitFor(() => expect(calls.length).toBeGreaterThan(1));
		expect(calls[1]).not.toBe(calls[0]);
		expect(calls[1]).toBe("older-1");
		expect(getHistory).toHaveBeenCalled();
	});
});
