// @vitest-environment happy-dom
//
// A user's own message inside a CONDENSED turn used to be cut to ONE line by CSS
// `truncate` (a bare "…"), and the only way to the rest was the separate
// "Took 8s · 7 actions" bar beneath it. The operator's report: the message "gets
// cut off … you can't expand or collapse it".
//
// The contract here:
//   * a condensed user message keeps its whole text in the DOM and shows the first
//     three lines; when the text is VERY long (8+ wrapped lines) an ICON-ONLY
//     control appears: a chevron (down to open, up to close) with NO visible
//     text, whose accessible name is still "Show more of your message" /
//     "Show less of your message", with aria-expanded and aria-controls;
//   * the control appears ONLY for very long text (short and medium have none);
//   * an uncondensed message (open turn, live turn) is full text with no control;
//   * the reader's explicit "Show more" survives opening and re-condensing the turn;
//   * pressing the control holds ITS place on screen (the scroll anchor).
//
// WHAT THIS LAYER CANNOT PROVE. happy-dom does no layout, so `scrollHeight` is 0
// and "how many lines is this text" is unanswerable here. `stubLayout` stands in
// for the browser's own answers with a deterministic model (21 px a line, 30
// characters a line; `scrollHeight` and the computed `lineHeight` both come from
// it) so the component's DECISIONS are tested against both outcomes. 21, not the
// component's 20 px fallback, so the real `parseFloat` branch is the one running.
// Whether real text at 320/360/390 reaches 8 lines, and the chip's drawn size and
// 56 x 44 hit area, are shown by the phone-size captures on the PR, not by this
// file (no layout here). The clamp is an inline style on purpose:
// an inline style RESOLVES in happy-dom, so dropping the clamp fails the test
// where a class-name echo would pass for a clamp that is present but inert.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

const DEFAULT_LINE_PX = 21;
const CHARS_PER_LINE = 30;
const CLAMP_LINES = 3;

/* happy-dom keeps the prefixed property under React's own spelling and drops the
   `-webkit-box` display, so the resolved line count is the observable. */
const clampOf = (el: HTMLElement): number => {
	const style = el.style as unknown as Record<string, string>;
	return Number(style.WebkitLineClamp || el.style.getPropertyValue("-webkit-line-clamp") || 0);
};

/** The browser's answers to "how tall is this text" and "what is its line height",
 *  in a layout-less DOM. `linePx` may be fractional (text zoom): the browser
 *  hands `scrollHeight` back as an integer, so it is floored here too. */
function stubLayout(linePx: number) {
	const isText = (el: Element) => el.hasAttribute("data-user-text");
	const lines = (el: Element) => Math.max(1, Math.ceil((el.textContent ?? "").length / CHARS_PER_LINE));
	const scrollHeight = vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(function (this: HTMLElement) {
		return isText(this) ? Math.floor(lines(this) * linePx) : 0;
	});
	const real = globalThis.getComputedStyle.bind(globalThis);
	const computed = vi.spyOn(globalThis, "getComputedStyle").mockImplementation((el: Element, pseudo?: string | null) => {
		const style = real(el, pseudo);
		return isText(el) ? new Proxy(style, { get: (target, key) => (key === "lineHeight" ? `${linePx}px` : Reflect.get(target, key)) }) : style;
	});
	return () => {
		scrollHeight.mockRestore();
		computed.mockRestore();
	};
}

let restoreLayout: () => void;
beforeEach(() => {
	restoreLayout = stubLayout(DEFAULT_LINE_PX);
});
afterEach(() => {
	restoreLayout();
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
const tool = (id: string) => entry(id, "tool", "", { tool_name: "bash", tool_call_id: id, summary: `run ${id}`, elapsed_s: 2 });

/** Long enough to reach the fold threshold (8 wrapped lines of 30 characters). */
const LONG =
	"These two issues are on local operator mobile relay and I would like both of them looked at before you change anything at all, please. " +
	"The first is a message that gets cut off and cannot be opened; the second is that my own messages blend into the page. " +
	"Take a look at the pill and at the message I sent, and you will see exactly the two problems I mean.";
/** Past three lines but short of the threshold: a paragraph, read in place. */
const MEDIUM =
	"These two issues are on local operator mobile relay and I would like both of them looked at before you change anything at all, please.";
const SHORT = "Quick one: what time is it?";

function settledTurn(userText: string) {
	return [
		entry("u", "user", userText),
		tool("t-1"), tool("t-2"), tool("t-3"),
		entry("a", "assistant", "The answer"),
	];
}

const userText = (root: ParentNode) => root.querySelector<HTMLElement>("[data-user-text]")!;
const showMore = () => screen.queryByRole("button", { name: "Show more of your message" });
const showLess = () => screen.queryByRole("button", { name: "Show less of your message" });

describe("a user message inside a condensed turn", () => {
	it("keeps the whole text, shows three lines, and offers Show more", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		expect(container.querySelector("[data-condensed-turn]")).not.toBeNull();
		// The full text is in the DOM (a screen reader reads all of it) ...
		expect(userText(container).textContent).toBe(LONG);
		// ... and the box is clamped to three lines (resolved inline style).
		expect(clampOf(userText(container))).toBe(CLAMP_LINES);
		const button = showMore()!;
		expect(button).toBeTruthy();
		expect(button.getAttribute("aria-expanded")).toBe("false");
		// Icon only: no word on screen, but the name a screen reader gets is unchanged.
		expect(button.textContent).toBe("");
		expect(button.getAttribute("aria-controls")).toBe(userText(container).id);
		expect(userText(container).id).not.toBe("");
	});

	it("opens to the full text and closes again, naming the state each way", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		fireEvent.click(showMore()!);
		expect(clampOf(userText(container))).toBe(0);
		expect(userText(container).textContent).toBe(LONG);
		expect(showMore()).toBeNull();
		const less = showLess()!;
		expect(less.getAttribute("aria-expanded")).toBe("true");
		expect(less.textContent).toBe("");
		fireEvent.click(less);
		expect(clampOf(userText(container))).toBe(CLAMP_LINES);
		expect(showMore()).toBeTruthy();
	});

	it("shows a medium message in full, with no clamp and no control", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(MEDIUM)} />);
		expect(container.querySelector("[data-condensed-turn]")).not.toBeNull();
		expect(userText(container).textContent).toBe(MEDIUM);
		// Over three lines, under the threshold: nothing is hidden, nothing to press.
		expect(clampOf(userText(container))).toBe(0);
		expect(showMore()).toBeNull();
		expect(showLess()).toBeNull();
	});

	it("starts folding at exactly the threshold, not a line before", () => {
		const atThreshold = "x".repeat(CHARS_PER_LINE * 8);
		const justUnder = "x".repeat(CHARS_PER_LINE * 7);
		const first = render(<Transcript pid="s" entries={settledTurn(justUnder)} />);
		expect(showMore()).toBeNull();
		first.unmount();
		render(<Transcript pid="s" entries={settledTurn(atThreshold)} />);
		expect(showMore()).toBeTruthy();
	});

	it("points the chevron down to open and up to close, and draws it that way", () => {
		render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		const chevron = () => (showMore() ?? showLess())!.querySelector("svg")!;
		// `data-direction` is the label; the drawn path is what the reader sees.
		const DOWN = "M2.5 4.5 6 8l3.5-3.5";
		const UP = "M2.5 7.5 6 4l3.5 3.5";
		expect(chevron().getAttribute("data-direction")).toBe("down");
		expect(chevron().querySelector("path")!.getAttribute("d")).toBe(DOWN);
		expect(chevron().getAttribute("aria-hidden")).toBe("true");
		fireEvent.click(showMore()!);
		expect(chevron().getAttribute("data-direction")).toBe("up");
		expect(chevron().querySelector("path")!.getAttribute("d")).toBe(UP);
		fireEvent.click(showLess()!);
		expect(chevron().getAttribute("data-direction")).toBe("down");
		expect(chevron().querySelector("path")!.getAttribute("d")).toBe(DOWN);
	});

	it("measures against the line height the browser reports, not a fixed 20 px", () => {
		restoreLayout();
		restoreLayout = stubLayout(24);
		// 7 lines at 24 px is 168 px: over a 20 px guess (160) but under the real 8 lines (192).
		render(<Transcript pid="s" entries={settledTurn("x".repeat(CHARS_PER_LINE * 7))} />);
		expect(showMore()).toBeNull();
	});

	it("still folds an exact 8-line message when text zoom makes the line height fractional", () => {
		restoreLayout();
		restoreLayout = stubLayout(20.4);
		// 8 x 20.4 = 163.2 but `scrollHeight` comes back as the integer 163.
		const first = render(<Transcript pid="s" entries={settledTurn("x".repeat(CHARS_PER_LINE * 8))} />);
		expect(showMore()).toBeTruthy();
		first.unmount();
		// The slack is one pixel, not a line: 7 lines (142 px) still does not fold.
		render(<Transcript pid="s" entries={settledTurn("x".repeat(CHARS_PER_LINE * 7))} />);
		expect(showMore()).toBeNull();
	});

	it("offers no control when the text fits the three lines", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(SHORT)} />);
		expect(container.querySelector("[data-condensed-turn]")).not.toBeNull();
		expect(userText(container).textContent).toBe(SHORT);
		expect(showMore()).toBeNull();
		expect(showLess()).toBeNull();
	});

	it("keeps every line of a multi-line message, not just the first", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn("First question line\nThe rest of the prompt")} />);
		expect(userText(container).textContent).toBe("First question line\nThe rest of the prompt");
	});
});

describe("a user message that is not condensed", () => {
	it("is full text with no clamp and no control while its turn is live", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} streaming />);
		expect(container.querySelector("[data-condensed-turn]")).toBeNull();
		expect(clampOf(userText(container))).toBe(0);
		expect(showMore()).toBeNull();
		expect(showLess()).toBeNull();
	});

	it("loses the control, and the clamp, when the reader opens the turn", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		expect(showMore()).toBeTruthy();
		fireEvent.click(container.querySelector("[data-condense-header]")!);
		expect(clampOf(userText(container))).toBe(0);
		expect(showMore()).toBeNull();
		expect(showLess()).toBeNull();
	});
});

describe("the reader's choice survives the turn bar", () => {
	it("keeps an explicit Show more through open-turn then re-condense", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		fireEvent.click(showMore()!);
		fireEvent.click(container.querySelector("[data-condense-header]")!); // open the turn
		expect(showLess()).toBeNull();
		fireEvent.click(container.querySelector("[data-condense-header]")!); // condense again
		// Still opened by the reader, and the way back is still on screen.
		expect(clampOf(userText(container))).toBe(0);
		expect(showLess()).toBeTruthy();
	});

	it("keeps a folded message folded through open-turn then re-condense", () => {
		const { container } = render(<Transcript pid="s" entries={settledTurn(LONG)} />);
		fireEvent.click(container.querySelector("[data-condense-header]")!);
		fireEvent.click(container.querySelector("[data-condense-header]")!);
		expect(clampOf(userText(container))).toBe(CLAMP_LINES);
		expect(showMore()).toBeTruthy();
	});
});

describe("the scroll anchor", () => {
	function scrollerAt(container: HTMLElement, top: number, scrollHeight = 2000) {
		const scroller = container.querySelector<HTMLElement>(".lo-scroll")!;
		Object.defineProperty(scroller, "scrollHeight", { value: scrollHeight, configurable: true });
		Object.defineProperty(scroller, "clientHeight", { value: 500, configurable: true });
		scroller.scrollTop = top;
		vi.spyOn(scroller, "getBoundingClientRect").mockReturnValue({ top: 100 } as DOMRect);
		return scroller;
	}

	it("OPENING holds the message's own top, so the new lines fall below what the reader sees", () => {
		const { container } = render(<Transcript pid="anchor-open" entries={settledTurn(LONG)} />);
		const scroller = scrollerAt(container, 200);
		const text = userText(container);
		const button = showMore()!;
		/* The control sits at the foot of the text, so opening pushes IT down by 60
		   while the text's top stays at 300. Holding the control would move the
		   scroller 60 and slide the first lines of the message under the header
		   (design review D1); holding the text's top moves nothing. */
		vi.spyOn(text, "getBoundingClientRect").mockReturnValue({ top: 300 } as DOMRect);
		vi.spyOn(button, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 400 } as DOMRect)
			.mockReturnValue({ top: 460 } as DOMRect);
		fireEvent.click(button);
		expect(clampOf(userText(container))).toBe(0);
		expect(scroller.scrollTop).toBe(200);
	});

	it("OPENING still follows the text if something above it moved", () => {
		const { container } = render(<Transcript pid="anchor-open-shift" entries={settledTurn(LONG)} />);
		const scroller = scrollerAt(container, 200);
		const text = userText(container);
		vi.spyOn(text, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 300 } as DOMRect)
			.mockReturnValue({ top: 340 } as DOMRect);
		fireEvent.click(showMore()!);
		expect(scroller.scrollTop).toBe(240);
	});

	it("CLOSING holds the control under the thumb as the text shrinks", () => {
		const { container } = render(<Transcript pid="anchor-close" entries={settledTurn(LONG)} />);
		const scroller = scrollerAt(container, 260);
		fireEvent.click(showMore()!);
		const less = showLess()!;
		/* Closing pulls the control UP by 60 (460 -> 400); the scroller follows it
		   by 60 so the finger is still on it. */
		vi.spyOn(less, "getBoundingClientRect")
			.mockReturnValueOnce({ top: 460 } as DOMRect)
			.mockReturnValue({ top: 400 } as DOMRect);
		scroller.scrollTop = 260;
		fireEvent.click(less);
		expect(clampOf(userText(container))).toBe(CLAMP_LINES);
		expect(scroller.scrollTop).toBe(200);
	});

	it("does not re-pin a tail reader, or move anything, when the geometry is unchanged", () => {
		const { container } = render(<Transcript pid="tail" entries={settledTurn(LONG)} />);
		const scroller = scrollerAt(container, 300, 1000);
		vi.spyOn(userText(container), "getBoundingClientRect").mockReturnValue({ top: 300 } as DOMRect);
		fireEvent.click(showMore()!);
		// A pinned reader would have been written to scrollHeight (1000).
		expect(scroller.scrollTop).toBe(300);
	});
});

describe("every bubble the user said carries the same edge", () => {
	/* The edge is the `border-control` token, which `pnpm check-themes`
	   (scripts/contrast-contract.mjs) holds at >= 3:1 against every ground in all
	   palettes. happy-dom has no stylesheet, so this can only say the call sites
	   agree on the token; that the edge PAINTS is shown by computed styles on the
	   rendered bundle (see the PR). */
	it("gives the settled user bubble, the steer row and the parent bubble the control edge", () => {
		const { container } = render(<Transcript pid="s" entries={[
			entry("u1", "user", SHORT),
			entry("s1", "steer", "go left"),
			entry("p1", "parent_message", "from the parent"),
		]} streaming />);
		for (const text of [SHORT, "go left", "from the parent"]) {
			const bubble = screen.getByText(text).closest("div.rounded-md") as HTMLElement;
			expect(bubble, text).toBeTruthy();
			expect(bubble.classList.contains("border"), text).toBe(true);
			expect(bubble.classList.contains("border-control"), text).toBe(true);
			expect(bubble.classList.contains("border-hairline"), text).toBe(false);
		}
		expect(container.querySelectorAll(".border-hairline").length).toBe(0);
	});
});
