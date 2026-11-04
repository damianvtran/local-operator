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
		final: true,
		text_complete: true,
		details: {},
		...extra,
	} as TranscriptEntry;
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
		const view = render(<Transcript pid="cursor-live-only" entries={liveOnlyHead()} headPinned />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		const cursor = vi.mocked(getHistory).mock.calls[0][1];
		expect(cursor).toBe("answer-1");
		expect(cursor).not.toBe("rz-answer-1");
		expect(cursor).not.toBe("tc-call-1");
	});

	it("never uses the pinned opener when the frame cap shrank the projection", async () => {
		/* The frame cap's floor is 16 transcript rows (`FRAME_CAP_TRANSCRIPT_FLOOR`)
		   and it pins the opening user message ahead of a disjoint tail. The daemon
		   says so on the wire (`transcript_head_pinned`); length cannot, because a
		   16-row pinned tail and a 16-row uncapped one are indistinguishable to the
		   client. */
		vi.mocked(getHistory).mockResolvedValue({
			entries: [
				entry("served-oldest", "assistant", "older history"),
				entry("served-newest", "assistant", "newer history"),
			],
			has_more: true,
		});
		const rows = liveOnlyHead().slice(0, 16);
		const view = render(<Transcript pid="cursor-capped" entries={rows} headPinned />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("answer-1");

		/* AND THE PAGE LANDS UNDER THE OPENER. The opener is pinned ahead of a
		   disjoint tail; if the cap path falls back to the uncapped branch, the
		   prepended page sorts ABOVE it and the opening message ends up in the
		   middle of the conversation (measured: MID0 > MID4 > OPENER > TAIL0). */
		await waitFor(() => expect(view.container.textContent).toContain("older history"));
		const text = view.container.textContent ?? "";
		expect(text.indexOf("question")).toBeGreaterThanOrEqual(0);
		expect(text.indexOf("question")).toBeLessThan(text.indexOf("older history"));
	});

	it("skips every id the live fold mints, not just reasoning and tool rows", async () => {
		/* Notices (`nt-`, `rt-`, `err-`, `rej-`) and compactions (`cx-`) are
		   minted from timestamps and exist only in the projection; the child sheet
		   also synthesises `prompt:<job>`. A cursor on any of them is answered as
		   end-of-history. */
		vi.mocked(getHistory).mockResolvedValue({ entries: [], has_more: false });
		const rows = [
			entry("opening", "user", "question"),
			entry("nt-1793214168142736", "notice", "wake receipt"),
			entry("rt-1793214168142737", "notice", "runtime note"),
			entry("cx-1793214168142738", "compaction", "compacted"),
			entry("prompt:job-7f3", "parent_message", "the launch task"),
			entry("err-1793214168142739", "notice", "turn failed"),
			entry("rej-1793214168142740", "notice", "prompt rejected"),
			entry("answer-1", "assistant", "the first addressable row"),
			...Array.from({ length: 8 }, (_, i) => entry(`row-${i}`, "assistant", `body ${i}`)),
		];
		const view = render(<Transcript pid="cursor-prefixes" entries={rows} headPinned />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("answer-1");
	});

	it("keeps the served pages' own ids as the cursor once history arrives", async () => {
		vi.mocked(getHistory).mockResolvedValue({
			entries: [
				entry("served-oldest", "assistant", "older"),
				entry("served-newest", "assistant", "newer"),
			],
			has_more: true,
		});
		const view = render(<Transcript pid="cursor-served" entries={liveOnlyHead()} headPinned />);
		const scroller = view.container.querySelector(".lo-scroll")!;

		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));

		fireEvent.scroll(scroller);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(2));
		expect(vi.mocked(getHistory).mock.calls[1][1]).toBe("served-oldest");
	});

	it("trusts a daemon that says the head is not pinned", async () => {
		/* `false` is a STATEMENT, not an absence: an uncapped projection whose head
		   really is the oldest row. The cursor is then the head itself. Collapsing
		   `undefined` into `false` at the call site would also collapse this case and
		   the fallback above, which is why the call sites pass the tri-state. */
		vi.mocked(getHistory).mockResolvedValue({ entries: [], has_more: false });
		const view = render(
			<Transcript pid="cursor-unpinned" entries={liveOnlyHead()} headPinned={false} />,
		);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("opening");
	});

	it("skips live-only ids even for a daemon that does not state the pin", async () => {
		/* Rolling upgrades: an older daemon omits `transcript_head_pinned`. The
		   length/kind fallback still skips the live-only ids, which is the part it can
		   do without help. It cannot tell a pinned 16-row tail from an uncapped one —
		   that is exactly what the wire fact exists for — so that case is not asserted
		   here as if it were fixed. */
		vi.mocked(getHistory).mockResolvedValue({ entries: [], has_more: false });
		const view = render(<Transcript pid="cursor-old-daemon" entries={liveOnlyHead()} />);

		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);

		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls[0][1]).toBe("answer-1");
	});
});
