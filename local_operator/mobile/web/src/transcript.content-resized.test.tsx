// @vitest-environment happy-dom
//
// A row that changes height AFTER the transcript settled (a mermaid fence
// replaced by its drawing, `components/diagram-block.tsx`) announces it with
// CONTENT_RESIZED_EVENT. The scroller keeps native scroll anchoring off
// (`[overflow-anchor:none]`), so the transcript's own anchor-follow is the
// only thing that can hold a reader who has scrolled up into history. Before
// PR #4149 round-3 remediation (F1), the event was handled only for a PINNED
// reader: an unpinned reader's row slid down by the drawing's growth, and the
// next unrelated commit then "corrected" the stale delta and jumped them back.
// happy-dom has no layout, so geometry is stubbed: every row sits at a fixed
// content offset, rows below the diagram move down by `growth`, and the
// viewport position is `offset - scrollTop`.
import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { CONTENT_RESIZED_EVENT } from "./lib/content-resized";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

function entry(id: string, kind: TranscriptEntry["kind"], text: string): TranscriptEntry {
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
	};
}

function stubGeometry(growth: { px: number }) {
	/* Content offsets by anchor id; "a" is the row holding the diagram, every
	   row after it is pushed down by the drawing's growth. */
	const offsets = new Map<string, number>();
	return vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
		const id = this.getAttribute("data-completion-anchor");
		const scroller = this.closest<HTMLElement>(".lo-scroll");
		if (id === null || !scroller) return { top: 0, bottom: 0, height: 0 } as DOMRect;
		if (!offsets.has(id)) offsets.set(id, 400 * (offsets.size + 1));
		const below = id === "a" ? 0 : growth.px;
		const top = (offsets.get(id) ?? 0) + below - scroller.scrollTop;
		return { top, bottom: top, height: 0 } as DOMRect;
	});
}

describe("a row that grows after it painted", () => {
	const rows = [
		entry("a", "assistant", "diagram here"),
		entry("b", "user", "next question"),
		entry("c", "assistant", "next answer"),
	];

	it("holds a reader who scrolled up: the row they read stays put", () => {
		const growth = { px: 0 };
		stubGeometry(growth);
		const view = render(<Transcript pid="content-resized" entries={rows} />);
		const el = view.container.querySelector<HTMLElement>(".lo-scroll")!;
		Object.defineProperty(el, "scrollHeight", { value: 3000, configurable: true });
		Object.defineProperty(el, "clientHeight", { value: 500, configurable: true });
		el.scrollTop = 1000;
		fireEvent.scroll(el); // 3000 - 1000 - 500 = 1500 px above the tail: unpinned
		// Record the anchors in this frame, as any ordinary commit would.
		view.rerender(<Transcript pid="content-resized" entries={[...rows]} />);
		expect(el.scrollTop).toBe(1000);

		growth.px = 240; // the drawing landed, 240 px taller than the code block
		act(() => {
			el.querySelector("[data-completion-anchor]")!.dispatchEvent(
				new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }),
			);
		});
		expect(el.scrollTop).toBe(1240);

		// The next unrelated commit finds nothing stale to undo.
		view.rerender(<Transcript pid="content-resized" entries={[...rows]} />);
		expect(el.scrollTop).toBe(1240);
	});

	it("keeps a pinned reader at the tail", () => {
		const view = render(<Transcript pid="content-resized-pinned" entries={rows} />);
		const el = view.container.querySelector<HTMLElement>(".lo-scroll")!;
		Object.defineProperty(el, "scrollHeight", { value: 3000, configurable: true });
		act(() => {
			el.dispatchEvent(new CustomEvent(CONTENT_RESIZED_EVENT, { bubbles: true }));
		});
		expect(el.scrollTop).toBe(3000);
	});
});
