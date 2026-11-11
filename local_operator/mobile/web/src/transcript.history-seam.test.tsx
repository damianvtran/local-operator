// @vitest-environment happy-dom
//
// The daemon's long-session projection is the opening user row plus the newest
// ~79 rows (`transcript_head_pinned`), so the middle of the conversation is
// NOT in the frame. These tests pin what the reader sees around that hole: the
// fetch that fills it must start without a gesture, and no row between the
// opener and the window may be dropped without a visible seam.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript, TranscriptSkeleton } from "./components/transcript";
import { getHistory } from "./api";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

function msg(id: string, kind: "user" | "assistant", text: string): TranscriptEntry {
	return {
		id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true, text_complete: true,
	};
}

/** `count` alternating user/assistant rows `<prefix>-0..`; text is the row id. */
const rows = (prefix: string, count: number, firstKind: "user" | "assistant" = "user") =>
	Array.from({ length: count }, (_, i) =>
		msg(`${prefix}-${i}`, (i % 2 === 0) === (firstKind === "user") ? "user" : "assistant", `${prefix}-${i}`));

/** The wire shape of a long session: the opener pinned ahead of the newest 79. */
const pinnedFrame = () => [msg("opener", "user", "opener"), ...rows("tail", 79, "assistant")];

const scrollerOf = (container: HTMLElement) =>
	container.querySelector<HTMLElement>("[data-transcript-scroller], .overflow-y-auto") as HTMLElement;

describe("pinned-opener history seam", () => {
	it("F2: fetches the missing middle on mount, with no scroll gesture", async () => {
		vi.mocked(getHistory).mockResolvedValue({ entries: rows("mid", 10), has_more: true });
		render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
		// The cursor is the oldest TAIL row, never the pinned opener.
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("tail-0");
	});

	it("F1: a page that lands while the reader is mid-list never leaves rows between the opener and the window unmarked", async () => {
		let resolve!: (page: { entries: TranscriptEntry[]; has_more: boolean }) => void;
		vi.mocked(getHistory).mockImplementation(() => new Promise((r) => { resolve = r; }));
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		const scroller = scrollerOf(container);
		// A fling: the fetch starts near the top (< 120), the reader is far away by the time it lands.
		scroller.scrollTop = 100;
		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		scroller.scrollTop = 500;
		await act(async () => resolve({ entries: rows("old", 120), has_more: true }));
		await waitFor(() => expect(container.querySelector("[data-detached-opener]")).toBeTruthy());

		const held = 120 + 79;
		const mounted = container.querySelectorAll("[data-completion-anchor]").length;
		const dropped = held - mounted;
		// 199 held rows, window of 120: 79 rows sit between the opener and the first mounted row.
		// Either every held row is mounted, or a seam stands where the rest are missing.
		if (dropped > 0) expect(container.querySelector("[data-history-seam]")).toBeTruthy();
		expect(screen.getByText("opener")).toBeTruthy();
	});
});

// A failed /history must stay failed until the reader taps "Try again". The
// error row used to push scrollTop from 0 to ~53 (< 120), `onScroll` read that as
// "near the top" and called `loadOlder`, which cleared the error and fetched
// again — measured at 203 requests in 4 s against a dead server, with the retry
// button never tappable. The invariant: ONLY the explicit retry re-requests
// after a failure, so every automatic path (scroll, the mount fetch, auto-fill)
// is silent while `olderError` stands.
describe("a failed history page", () => {
	const failFirst = () => {
		vi.mocked(getHistory).mockRejectedValue(new Error("offline"));
	};

	it("U1: scrolling near the top does not re-request after a failure", async () => {
		failFirst();
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(container.querySelector("[data-history-slot=failed]")).toBeTruthy());
		expect(getHistory).toHaveBeenCalledTimes(1);
		const scroller = scrollerOf(container);
		for (const top of [0, 53, 0, 53, 100]) {
			scroller.scrollTop = top;
			fireEvent.scroll(scroller);
		}
		await act(async () => { await Promise.resolve(); });
		expect(getHistory).toHaveBeenCalledTimes(1);
		expect(container.querySelector("[data-history-slot=failed]")).toBeTruthy();
	});

	it("U1: Try again retries exactly once per tap", async () => {
		failFirst();
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(container.querySelector("[data-history-slot=failed]")).toBeTruthy());
		fireEvent.click(screen.getByRole("button", { name: "Try again" }));
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(container.querySelector("[data-history-slot=failed]")).toBeTruthy());
		expect(getHistory).toHaveBeenCalledTimes(2);
	});

	it("U1: a non-capped session that failed a scroll-triggered page stays quiet too", async () => {
		failFirst();
		const { container } = render(<Transcript pid="s" entries={rows("only", 6)} />);
		const scroller = scrollerOf(container);
		scroller.scrollTop = 10;
		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
		await act(async () => { await Promise.resolve(); });
		for (const top of [0, 53, 0]) {
			scroller.scrollTop = top;
			fireEvent.scroll(scroller);
		}
		await act(async () => { await Promise.resolve(); });
		expect(getHistory).toHaveBeenCalledTimes(1);
	});
});

describe("the top slot", () => {
	const slot = (container: HTMLElement) => container.querySelector("[data-history-slot]");

	it("is absent, with no seam, on a short session", async () => {
		const { container } = render(<Transcript pid="s" entries={rows("short", 6)} />);
		await act(async () => { await Promise.resolve(); });
		expect(slot(container)).toBeNull();
		expect(container.querySelector("[data-history-seam]")).toBeNull();
		expect(getHistory).not.toHaveBeenCalled();
	});

	it("shows loading with a status label and the seam line under the opener while the page is in flight", async () => {
		vi.mocked(getHistory).mockImplementation(() => new Promise(() => {}));
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(slot(container)?.getAttribute("data-history-slot")).toBe("loading"));
		expect(slot(container)?.getAttribute("role")).toBe("status");
		expect(screen.getByText("Loading earlier messages")).toBeTruthy();
		const opener = container.querySelector("[data-detached-opener]")!;
		expect(opener.nextElementSibling?.hasAttribute("data-history-seam")).toBe(true);
	});

	it("is a tappable idle button, not a fetch, while a turn is live", async () => {
		vi.mocked(getHistory).mockResolvedValue({ entries: rows("mid", 10), has_more: false });
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned streaming />);
		await act(async () => { await Promise.resolve(); });
		expect(getHistory).not.toHaveBeenCalled();
		expect(slot(container)?.getAttribute("data-history-slot")).toBe("idle");
		fireEvent.click(screen.getByRole("button", { name: "Load earlier messages" }));
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
	});

	it("says start of conversation, and drops the seam, once every page is held", async () => {
		vi.mocked(getHistory).mockResolvedValue({ entries: [msg("opener", "user", "opener"), ...rows("mid", 9, "assistant")], has_more: false });
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(slot(container)?.getAttribute("data-history-slot")).toBe("start"));
		expect(container.querySelector("[data-history-seam]")).toBeNull();
		expect(getHistory).toHaveBeenCalledTimes(1);
	});

	it("F1: a tap reveals the rows the window dropped, and the seam goes with them", async () => {
		let resolve!: (page: { entries: TranscriptEntry[]; has_more: boolean }) => void;
		vi.mocked(getHistory).mockImplementation(() => new Promise((r) => { resolve = r; }));
		const { container } = render(<Transcript pid="s" entries={pinnedFrame()} headPinned />);
		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		scrollerOf(container).scrollTop = 500;
		await act(async () => resolve({ entries: rows("old", 120), has_more: false }));
		await waitFor(() => expect(slot(container)?.getAttribute("data-history-slot")).toBe("idle"));
		expect(container.querySelector("[data-history-seam]")).toBeTruthy();
		expect(slot(container)?.getAttribute("data-history-slot")).toBe("idle");
		fireEvent.click(screen.getByRole("button", { name: "Load earlier messages" }));
		await waitFor(() => expect(container.querySelectorAll("[data-completion-anchor]").length).toBe(199));
		expect(container.querySelector("[data-history-seam]")).toBeNull();
		expect(getHistory).toHaveBeenCalledTimes(1);
	});
});

describe("the empty-but-streaming column", () => {
	it("paints the skeleton, not just a hairline, when a turn runs with no rows yet", () => {
		const { container } = render(<Transcript pid="s" entries={[]} streaming emptyContent={<TranscriptSkeleton />} />);
		expect(container.querySelector("[data-transcript-skeleton]")).toBeTruthy();
	});
});
