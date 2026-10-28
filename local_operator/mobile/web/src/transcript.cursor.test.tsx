// @vitest-environment happy-dom
import { cleanup, fireEvent, render, waitFor } from "@testing-library/react";
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

function entry(
	id: string,
	kind: TranscriptEntry["kind"],
	text = "",
	extra: Partial<TranscriptEntry> = {},
): TranscriptEntry {
	return {
		id,
		kind,
		text,
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
		text_complete: true,
		...extra,
	};
}

/* A live projection whose oldest tail rows are the fold's OWN rows: a reasoning
   row (`rz-<message>`) and a tool row (`tc-<call>`). Neither id names a journal
   entry — reasoning is never persisted, and a served page names a tool row
   `<message>:call_<call>` — so the daemon answers either as end-of-history. */
function liveOnlyHead(): TranscriptEntry[] {
	return [
		entry("opening", "user", "question"),
		entry("rz-answer-1", "reasoning", "thinking"),
		entry("answer-1", "assistant", "first answer"),
		entry("tc-call-1", "tool", "", { tool_name: "bash", tool_call_id: "call-1" }),
		...Array.from({ length: 75 }, (_, i) => entry(`row-${i}`, "assistant", `body ${i}`)),
		entry("answer-last", "assistant", "latest"),
	];
}

describe("the first history cursor", () => {
	it("names a journal row, not the reasoning/tool rows the live fold mints", async () => {
		vi.mocked(getHistory).mockResolvedValue({ entries: [], has_more: false });
		const view = render(<Transcript pid="cursor-live-only" entries={liveOnlyHead()} />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		const cursor = vi.mocked(getHistory).mock.calls[0][1];
		expect(cursor).toBe("answer-1");
		expect(cursor).not.toBe("rz-answer-1");
		expect(cursor).not.toBe("tc-call-1");
	});

	it("still never uses the pinned opener, even when the frame cap shrank the projection", async () => {
		/* The frame cap's floor is 16 transcript rows (`FRAME_CAP_TRANSCRIPT_FLOOR`)
		   and it pins the opening user message ahead of a disjoint tail. Below the
		   client's `PROJECTION_ROWS` (80) the old rule fell through to
		   `merged[0].id` — the opener — so every page asked for history older than
		   the first message of the conversation and the middle never loaded. */
		vi.mocked(getHistory).mockResolvedValue({ entries: [], has_more: false });
		const rows = liveOnlyHead().slice(0, 16);
		const view = render(<Transcript pid="cursor-capped" entries={rows} />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("answer-1");
	});

	it("keeps the served pages' own ids as the cursor once history arrives", async () => {
		vi.mocked(getHistory).mockResolvedValue({
			entries: [entry("served-oldest", "assistant", "older"), entry("served-newest", "assistant", "newer")],
			has_more: true,
		});
		const view = render(<Transcript pid="cursor-served" entries={liveOnlyHead()} />);
		const scroller = view.container.querySelector(".lo-scroll")!;

		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));

		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(2));
		expect(vi.mocked(getHistory).mock.calls[1][1]).toBe("served-oldest");
	});
});
