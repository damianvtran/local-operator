// @vitest-environment happy-dom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { toCondenseRecord } from "./lib/condense-records";
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
const tool = (id: string, extra: Partial<TranscriptEntry> = {}) => entry(id, "tool", "", {
	tool_name: "bash", tool_call_id: id, summary: `run ${id}`, elapsed_s: 2, ...extra,
});

function settledTurn(prefix = "t") {
	return [
		entry(`${prefix}-u`, "user", "First question line\nThe rest of the prompt"),
		tool(`${prefix}-1`), tool(`${prefix}-2`), tool(`${prefix}-3`),
		entry(`${prefix}-a`, "assistant", "The answer"),
	];
}

describe("condensed transcript rendering", () => {
	it("condenses a settled turn on first paint and restores every row on expand", () => {
		const rows = settledTurn();
		const view = render(<Transcript pid="s" entries={rows} />);
		expect(screen.getByText("First question line")).toBeTruthy();
		expect(screen.queryByText(/The rest of the prompt/)).toBeNull();
		expect(screen.queryByText("run t-1")).toBeNull();
		expect(screen.getByText("The answer")).toBeTruthy();
		expect(view.container.querySelectorAll("[data-condense-header]").length).toBe(1);
		expect(getHistory).not.toHaveBeenCalled();

		fireEvent.click(view.container.querySelector("[data-condense-header]")!);
		expect(screen.getByText(/The rest of the prompt/)).toBeTruthy();
		fireEvent.click(view.container.querySelector("[data-condense-fold] button")!);
		expect(screen.getByText("run t-1")).toBeTruthy();
		expect(screen.getByText("run t-2")).toBeTruthy();
		expect(screen.getByText("run t-3")).toBeTruthy();
		expect(getHistory).not.toHaveBeenCalled();
	});

	it("keeps the active turn uncondensed and does not un-condense the watched turn on a live dip", () => {
		const rows = settledTurn();
		const view = render(<Transcript pid="s" entries={rows} streaming />);
		expect(view.container.querySelector("[data-condensed-turn]")).toBeNull();
		fireEvent.click(view.container.querySelector("[data-condense-fold] button")!);
		expect(screen.getByText("run t-1")).toBeTruthy();
		view.rerender(<Transcript pid="s" entries={rows} />);
		expect(view.container.querySelector("[data-condensed-turn]")).not.toBeNull();
		view.rerender(<Transcript pid="s" entries={rows} streaming />);
		expect(view.container.querySelector("[data-condensed-turn]")).not.toBeNull();
	});

	it("names a failed call in a collapsed tool group's accessible state", () => {
		render(<Transcript pid="s" entries={[
			tool("a"), tool("b", { tool_state: "failed", error: "exit 1" }), tool("c"),
		]} streaming />);
		const group = document.querySelector("[data-condense-fold]");
		expect(group).not.toBeNull();
		expect(group?.textContent).toContain("Includes a failed action");
		expect(group?.querySelector("[data-failed-count]")?.textContent).toBe("1");
		expect(group?.querySelector(".text-danger")?.textContent).toBe("✗");
		fireEvent.click(group!.querySelector("button")!);
		fireEvent.click(screen.getByText("run b").closest("button")!);
		expect(screen.getByText("exit 1")).toBeTruthy();
	});

	it("anchors a toggled bar in the scroller's coordinate frame", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn()} />);
		const scroller = container.querySelector<HTMLElement>(".lo-scroll")!;
		const header = container.querySelector<HTMLElement>("[data-condense-header]")!;
		Object.defineProperty(scroller, "scrollHeight", { value: 2000, configurable: true });
		Object.defineProperty(scroller, "clientHeight", { value: 500, configurable: true });
		scroller.scrollTop = 200;
		vi.spyOn(scroller, "getBoundingClientRect").mockReturnValue({ top: 100 } as DOMRect);
		/* The tap reads 400; expansion moves the bar to 500. Relative to the
		   scroller that is 300 → 400, so follow must add 100. Comparing the
		   second viewport-relative top to the first one would add nothing. */
		vi.spyOn(header, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 400 } as DOMRect)
			.mockReturnValue({ top: 500 } as DOMRect);
		fireEvent.click(header);
		expect(scroller.scrollTop).toBe(300);
	});

	it("does not paint a blank notice with an empty custom type", () => {
		const blank = entry("blank", "notice", "", { custom_type: "" });
		expect(toCondenseRecord(blank)).toBeNull();
		const { container } = render(<Transcript pid="s" entries={[blank]} />);
		expect(container.querySelectorAll("[data-completion-anchor]").length).toBe(0);
	});

	it("completes a head-cut settled run through bounded history pages, not a full read", async () => {
		const older = Array.from({ length: 120 }, (_, i) => tool(`old-${i}`));
		vi.mocked(getHistory).mockResolvedValue({ entries: older, has_more: false });
		const tail = [...Array.from({ length: 79 }, (_, i) => tool(`tail-${i}`)), entry("answer", "assistant", "settled")];
		render(<Transcript pid="s" entries={tail} />);
		await waitFor(() => expect(getHistory).toHaveBeenCalled());
		expect(vi.mocked(getHistory).mock.calls.length).toBeLessThanOrEqual(3);
		expect(vi.mocked(getHistory).mock.calls[0][2]).toBe(120);
	});
});
