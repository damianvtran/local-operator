// @vitest-environment happy-dom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

function entry(id: string, kind: TranscriptEntry["kind"], text = "", extra: Partial<TranscriptEntry> = {}): TranscriptEntry {
	return {
		id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true, text_complete: true, ...extra,
	};
}
const tool = (id: string) => entry(id, "tool", "", {
	tool_name: "bash", tool_call_id: id, summary: `run ${id}`, elapsed_s: 2,
});

describe("condensed history and recovery", () => {
	it("advances all 22 pages beyond live retention and lets the reader reach their rows", async () => {
		vi.mocked(getHistory).mockImplementation(async (_pid, before, limit = 120) => {
			const end = Number(String(before).replace("row-", ""));
			return { entries: Array.from({ length: limit }, (_, i) => tool(`row-${end - limit + i}`)), has_more: true };
		});
		const rows = [entry("opening", "user", "question"),
			...Array.from({ length: 78 }, (_, i) => tool(`row-${3000 + i}`)), entry("answer", "assistant", "answer")];
		const view = render(<Transcript pid="history-regression" entries={rows} />);
		const scroller = view.container.querySelector(".lo-scroll")!;
		for (let i = 0; i < 22; i++) {
			fireEvent.scroll(scroller);
			await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(i + 1));
			await new Promise((resolve) => setTimeout(resolve, 0));
		}
		const cursors = vi.mocked(getHistory).mock.calls.map((call) => call[1]);
		expect(cursors).toEqual(Array.from({ length: 22 }, (_, i) => `row-${3000 - 120 * i}`));
		expect(vi.mocked(getHistory).mock.calls.every((call) => call[2] === 120)).toBe(true);
		// Advancing a cursor alone is not recovery if the requested rows were
		// still discarded. Open the fold and verify both ends of the loaded span.
		fireEvent.click(view.container.querySelector("[data-condensed-turn] > button")!);
		fireEvent.click(view.container.querySelector("[data-condense-fold] button")!);
		expect(screen.getByText("run row-360")).toBeTruthy();
		expect(screen.getByText("run row-3077")).toBeTruthy();
		expect(view.container.querySelectorAll("[data-completion-anchor]").length).toBeGreaterThan(2000);
		console.log("history: 22 unique 120-row cursors, row-3000 through row-480; loaded row-360 and row-3077 visible");
	});

	it("exposes a failed history page and retries the same cursor successfully", async () => {
		vi.mocked(getHistory).mockRejectedValueOnce(new Error("offline"))
			.mockResolvedValue({ entries: [tool("row-2880")], has_more: false });
		const rows = [tool("row-3000"), entry("answer", "assistant", "answer")];
		const view = render(<Transcript pid="history-error-regression" entries={rows} />);
		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);
		await waitFor(() => expect(screen.getByRole("status")).toBeTruthy());
		fireEvent.click(screen.getByRole("button", { name: "Try again" }));
		await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
		expect(getHistory).toHaveBeenCalledTimes(2);
		expect(vi.mocked(getHistory).mock.calls.map((call) => call[1])).toEqual(["row-3000", "row-3000"]);
		fireEvent.click(view.container.querySelector("[data-condensed-turn] > button")!);
		expect(screen.getByText("run row-2880")).toBeTruthy();
	});
});

describe("quiet-group ownership", () => {
	it("keeps canonical tools in their sole quiet group and restores exactly its hidden rows", () => {
		const rows = [entry("peer1", "peer_message", "first incoming"), tool("tc-a"), tool("tc-b"),
			tool("tc-c"), entry("peer2", "peer_message", "second incoming")];
		const view = render(<Transcript pid="quiet-regression" entries={rows} streaming />);
		expect(view.container.querySelectorAll("button")).toHaveLength(1);
		expect(view.container.querySelectorAll("[data-completion-anchor]")).toHaveLength(0);
		const header = view.container.querySelector("button")!;
		fireEvent.click(header);
		expect(view.container.querySelector("[data-condense-fold]")).toBeNull();
		expect(Array.from(view.container.querySelectorAll("[data-completion-anchor]"),
			(row) => row.getAttribute("data-completion-anchor"))).toEqual(rows.map((row) => row.id));
		for (const id of ["tc-a", "tc-b", "tc-c"]) expect(screen.getAllByText(`run ${id}`)).toHaveLength(1);
		fireEvent.click(header);
		expect(view.container.querySelectorAll("[data-completion-anchor]")).toHaveLength(0);
	});
});

describe("active turn presentation", () => {
	it("keeps the full prompt when the live guard removes its disclosure", () => {
		const rows = [entry("u", "user", "First question line\nThe rest of the prompt"),
			tool("a"), tool("b"), tool("c"), entry("answer", "assistant", "The answer")];
		const view = render(<Transcript pid="live-final-regression" entries={rows} streaming />);
		expect(view.container.querySelector("[data-condensed-turn]")).toBeNull();
		expect(screen.getByText(/The rest of the prompt/)).toBeTruthy();
	});
});
