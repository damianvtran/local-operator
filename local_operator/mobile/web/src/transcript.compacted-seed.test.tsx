// @vitest-environment happy-dom
//
// PR #4521 QA Q3: a compacted session's render opens at the compaction marker
// and pins no opener, so a short kept window reaches the client as a handful of
// rows with `transcript_head_pinned` absent. Read as uncapped, the relay SPA
// painted no load-earlier slot, made no mount fetch, and had nothing to scroll,
// so every turn the compaction summarised was unreachable. The daemon now says
// `transcript_has_older`; these tests pin that the client pages on it WITHOUT
// treating `entries[0]` as a pinned opener (QA Q1's false opener).
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
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

function row(id: string, kind: TranscriptEntry["kind"], text: string): TranscriptEntry {
	return {
		id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true, text_complete: true,
	};
}

/** The QA shape: the newest compaction's marker, then a 3-turn kept window. */
const compactedSeed = (): TranscriptEntry[] => [
	{ ...row("cmp-2", "notice", "summary c2"), custom_type: "compaction" },
	...[120, 121, 122].flatMap((i) => [row(`req-${i}`, "user", `req ${i}`), row(`final-${i}`, "assistant", `final ${i}`)]),
];

describe("short compacted seed (transcript_has_older)", () => {
	it("Q3: fetches older history on mount and offers load-earlier, with no pin", async () => {
		vi.mocked(getHistory).mockResolvedValue({
			entries: [row("req-119", "user", "req 119"), row("final-119", "assistant", "final 119")],
			has_more: true,
		});
		const { container } = render(<Transcript pid="s" entries={compactedSeed()} hasOlder />);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
		// Paged from the render's own oldest row: nothing is skipped as an opener.
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("cmp-2");
		await waitFor(() => expect(screen.getByText("final 119")).toBeTruthy());
		// More remains, so the reader still has the load-earlier control.
		await waitFor(() => expect(container.querySelector("[data-history-slot=idle]")).toBeTruthy());
	});

	it("Q1 guard: a user row at the head is paged from, never detached as the opener", async () => {
		const seed = compactedSeed().slice(1); // the marker trimmed: req-120 heads it
		render(<Transcript pid="s" entries={seed} hasOlder />);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("req-120");
	});

	it("control: the same seed from an older daemon (no field) stays uncapped", async () => {
		const { container } = render(<Transcript pid="s" entries={compactedSeed()} />);
		await new Promise((r) => setTimeout(r, 50));
		expect(getHistory).not.toHaveBeenCalled();
		expect(container.querySelector("[data-history-slot]")).toBeNull();
	});
});
