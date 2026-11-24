// @vitest-environment happy-dom
//
// THE WAY BACK TO THE TAIL (design request, shared by the three surfaces).
//
// The transcript never moves a reader who scrolled up to read, so the return to
// the newest row has to be a control they press. This file pins the DECISIONS:
// the chip appears only when the reader is away from the tail AND content
// arrived after they left it; pressing it puts the scroller at the bottom and
// re-engages the follow; being back at the bottom by any other means hides it;
// and a message THE READER SENT snaps them to the bottom (typing does not).
//
// WHAT THIS LAYER CANNOT PROVE. happy-dom does no layout, so the chip's drawn
// size, its 44 px hit area, the narrow-label swap and the absence of a layout
// shift are shown by the phone-width captures on the PR, not here. The geometry
// this file does use is stubbed: `scrollHeight` 3000, `clientHeight` 500, so a
// reader at 1000 is 1500 px above the tail — the same shape
// `transcript.content-resized.test.tsx` uses.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import type { PendingEcho } from "./pending-echo";
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

const SCROLL_HEIGHT = 3000;
const CLIENT_HEIGHT = 500;
/** 3000 - 1000 - 500 = 1500 px above the tail: well past the 48 px pin rule. */
const MID_HISTORY = 1000;

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

const rows = [
	entry("a", "user", "first question"),
	entry("b", "assistant", "first answer"),
	entry("c", "user", "second question"),
];

function echo(commandId: string): PendingEcho {
	return { commandId, text: "a new question", imageCount: 0, op: "prompt", accepted: false };
}

function scroller(container: HTMLElement): HTMLElement {
	return container.querySelector<HTMLElement>(".lo-scroll")!;
}

/** Stands in for the browser's row geometry: every anchor sits at a fixed
 *  content offset and the viewport position is `offset - scrollTop`. Installed
 *  BEFORE the first render so the anchor hold records the same model the later
 *  commits measure against — a pure scroll then cancels to a zero delta, which
 *  is exactly what a scroll gesture is. */
function stubRects() {
	const offsets = new Map<string, number>();
	vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
		const id = this.getAttribute("data-completion-anchor");
		const el = this.closest<HTMLElement>(".lo-scroll");
		if (id === null || !el) return { top: 0, bottom: 0, height: 0 } as DOMRect;
		if (!offsets.has(id)) offsets.set(id, 400 * (offsets.size + 1));
		const top = (offsets.get(id) ?? 0) - el.scrollTop;
		return { top, bottom: top, height: 0 } as DOMRect;
	});
}

/** Stands in for the scroller's box and puts the reader in history. */
function stubScroller(el: HTMLElement, scrollTop = MID_HISTORY) {
	Object.defineProperty(el, "scrollHeight", { value: SCROLL_HEIGHT, configurable: true });
	Object.defineProperty(el, "clientHeight", { value: CLIENT_HEIGHT, configurable: true });
	el.scrollTop = scrollTop;
	act(() => {
		fireEvent.scroll(el);
	});
}

const chip = () => screen.queryByRole("button", { name: "Jump to latest" });

describe("the jump-to-latest chip", () => {
	it("appears only once new content arrives, and jumps back to the tail", () => {
		stubRects();
		const view = render(<Transcript pid="jump" entries={rows} />);
		const el = scroller(view.container);
		stubScroller(el);
		// Leaving the tail alone offers nothing: there is no new content to go to.
		expect(chip()).toBeNull();

		view.rerender(<Transcript pid="jump" entries={[...rows, entry("d", "assistant", "new answer")]} />);
		expect(chip()).not.toBeNull();

		act(() => {
			fireEvent.click(chip()!);
		});
		expect(el.scrollTop).toBe(SCROLL_HEIGHT);
		expect(chip()).toBeNull();
	});

	it("stays hidden on a same-content repaint and never moves the reader", () => {
		stubRects();
		const view = render(<Transcript pid="repaint" entries={rows} />);
		const el = scroller(view.container);
		stubScroller(el);
		// A fresh array identity, same content: what every projection frame is.
		view.rerender(<Transcript pid="repaint" entries={[...rows]} />);
		expect(chip()).toBeNull();
		expect(el.scrollTop).toBe(MID_HISTORY);
	});

	it("hides as soon as the reader is back at the bottom by their own scrolling", () => {
		stubRects();
		const view = render(<Transcript pid="back" entries={rows} />);
		const el = scroller(view.container);
		stubScroller(el);
		view.rerender(<Transcript pid="back" entries={[...rows, entry("d", "assistant", "new answer")]} />);
		expect(chip()).not.toBeNull();

		el.scrollTop = SCROLL_HEIGHT;
		act(() => {
			fireEvent.scroll(el);
		});
		expect(chip()).toBeNull();
	});

	it("snaps to the bottom when the reader sends, not when they type", () => {
		stubRects();
		const view = render(<Transcript pid="send" entries={rows} />);
		const el = scroller(view.container);
		stubScroller(el);

		// Typing changes nothing this component renders.
		view.rerender(<Transcript pid="send" entries={[...rows]} />);
		expect(el.scrollTop).toBe(MID_HISTORY);

		// The send's own echo appearing IS the request to see the message.
		view.rerender(<Transcript pid="send" entries={[...rows]} pending={[echo("cmd-1")]} />);
		expect(el.scrollTop).toBe(SCROLL_HEIGHT);
		expect(chip()).toBeNull();

		// The echo settling into the real row is not a new send.
		el.scrollTop = MID_HISTORY;
		act(() => {
			fireEvent.scroll(el);
		});
		view.rerender(
			<Transcript pid="send" entries={[...rows, entry("cmd-1", "user", "a new question")]} pending={[]} />,
		);
		expect(el.scrollTop).toBe(MID_HISTORY);
		expect(chip()).not.toBeNull();
	});

	it("floats outside the scroller, so showing it shifts no row", () => {
		stubRects();
		const view = render(<Transcript pid="float" entries={rows} />);
		const el = scroller(view.container);
		stubScroller(el);
		view.rerender(<Transcript pid="float" entries={[...rows, entry("d", "assistant", "new answer")]} />);

		const button = chip()!;
		// The POSITIONING box is a wrapper outside the scroller; the button itself
		// stays in flow inside it, because `.lo-hit`'s `position: relative` would
		// beat Tailwind's `absolute` on the button (see the component's note).
		const box = button.parentElement!;
		expect(box).not.toBe(el);
		expect(box.parentElement).toBe(el.parentElement);
		expect(box.className).toContain("absolute");
		expect(box.className).toContain("pointer-events-none");
		expect(button.className).toContain("pointer-events-auto");
	});
});
