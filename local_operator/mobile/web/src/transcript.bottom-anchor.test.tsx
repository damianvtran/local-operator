// @vitest-environment happy-dom
//
// DESIGN ROUND 1, D1 (PR #4521): a short compacted session jumped ~231 px down
// when its automatic earlier-page load landed, because a transcript shorter
// than the viewport stacked from the TOP. The scroller now carries a
// bottom-anchor spacer that absorbs the free space above the rows, so a
// prepend grows upward and the newest row keeps its distance from the bottom.
//
// happy-dom has no layout engine, so the geometry cell lays the scroller's
// REAL children out with the flex-column rules the stylesheet gives them
// (fixed row heights, the scroller's `gap-2`, each child's inline margin, and
// `flex-grow` on the inline `flex` shorthand): the DOM under test is the
// component's own, and only the box arithmetic is modelled — a rendered
// browser frame is the real-path check this cell cannot replace.
//
// Measured RED on a9ecde89 (origin/main), all four cells: no spacer, so the
// newest row sits 324 px above the bottom and moves to 36 px after the
// prepend (the D1 jump, in this model's units); the structural and overflow
// cells find no spacer; the stylesheet has no `scrollbar-gutter`.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { act, cleanup, render, waitFor } from "@testing-library/react";
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

/** QA's short compacted seed: the compaction marker, then a 3-turn kept window. */
const seed = (): TranscriptEntry[] => [
	{ ...row("cmp-2", "notice", "summary c2"), custom_type: "compaction" },
	...[120, 121, 122].flatMap((i) => [row(`req-${i}`, "user", `req ${i}`), row(`final-${i}`, "assistant", `final ${i}`)]),
];

const VIEWPORT = 700; // the scroller's content box, px
const ROW = 40; // every non-spacer child, px
const GAP = 8; // the scroller's `gap-2`

/** Lay out the scroller's direct children as a flex column would. */
function layout(scroller: HTMLElement): Map<Element, { top: number; bottom: number }> {
	const children = Array.from(scroller.children) as HTMLElement[];
	const grows = (el: HTMLElement) => Number.parseFloat(el.style.flex || "0") > 0;
	const px = (value: string) => (value.endsWith("rem") ? Number.parseFloat(value) * 16 : Number.parseFloat(value || "0"));
	const fixed = children.reduce(
		(sum, el, i) => sum + (grows(el) ? 0 : ROW) + (i > 0 ? GAP : 0) + px(el.style.marginTop) + px(el.style.marginBottom),
		0,
	);
	const free = Math.max(0, VIEWPORT - fixed);
	const boxes = new Map<Element, { top: number; bottom: number }>();
	let y = 0;
	children.forEach((el, i) => {
		if (i > 0) y += GAP;
		y += px(el.style.marginTop);
		const height = grows(el) ? free : ROW;
		boxes.set(el, { top: y, bottom: y + height });
		y += height + px(el.style.marginBottom);
	});
	return boxes;
}

/** The newest row's distance from the scroller's bottom edge. */
function newestOffsetFromBottom(container: HTMLElement): number {
	const scroller = container.querySelector<HTMLElement>(".lo-scroll")!;
	const rows = scroller.querySelectorAll("[data-completion-anchor]");
	const newest = rows[rows.length - 1]!;
	const child = Array.from(scroller.children).find((c) => c.contains(newest))!;
	return VIEWPORT - layout(scroller).get(child)!.bottom;
}

describe("short transcripts anchor to the bottom (D1)", () => {
	it("keeps the newest row's offset from the bottom across the auto-fill prepend", async () => {
		let land!: (page: { entries: TranscriptEntry[]; has_more: boolean }) => void;
		vi.mocked(getHistory).mockReturnValue(new Promise((resolve) => { land = resolve; }));
		const { container } = render(<Transcript pid="s" entries={seed()} hasOlder />);
		await waitFor(() => expect(getHistory).toHaveBeenCalledTimes(1));
		const before = newestOffsetFromBottom(container);

		await act(async () => {
			land({
				entries: [117, 118, 119].flatMap((i) => [row(`req-${i}`, "user", `req ${i}`), row(`final-${i}`, "assistant", `final ${i}`)]),
				has_more: true,
			});
		});
		await waitFor(() => expect(container.textContent).toContain("final 119"));
		const after = newestOffsetFromBottom(container);

		expect(after).toBe(before);
		/* And it sits ON the bottom edge, not merely at a stable offset. */
		expect(before).toBe(0);
	});

	it("renders the anchor as a grow-only spacer after the reserve and before every row", () => {
		const { container } = render(<Transcript pid="s" entries={seed()} topInset={52} />);
		const scroller = container.querySelector<HTMLElement>(".lo-scroll")!;
		const spacer = scroller.querySelector<HTMLElement>(":scope > [data-bottom-anchor]");
		expect(spacer).toBeTruthy();
		const children = Array.from(scroller.children);
		/* The top-inset reserve stays first (transcript.top-inset.test.tsx). */
		expect(children[0]?.hasAttribute("data-scroll-top-inset")).toBe(true);
		expect(children[1]).toBe(spacer);
		/* Grows into free space only, never shrinks a row, and cancels the gap
		   after it so a long transcript's geometry is unchanged. */
		expect(spacer!.style.flex).toMatch(/^1 0 0(px)?$/);
		expect(spacer!.style.marginBottom).toBe("-0.5rem");
		expect(scroller.className).toContain("gap-2");
		expect(spacer!.getAttribute("aria-hidden")).toBe("true");
	});

	/* Design round 1, D2 (PR #4521): a classic scrollbar appearing took ~11 px
	   from the content width and shifted the bubbles sideways. vitest loads no
	   stylesheet, so the rule is pinned at its source. */
	it("reserves the scrollbar gutter on .lo-scroll", () => {
		const css = readFileSync(join(__dirname, "styles", "index.css"), "utf8");
		const block = /\n\.lo-scroll \{([^}]*)\}/.exec(css)?.[1] ?? "";
		expect(block).toMatch(/scrollbar-gutter:\s*stable;/);
	});

	it("contributes no height once the content overflows", () => {
		const long = Array.from({ length: 40 }, (_, i) => row(`r-${i}`, "assistant", `answer ${i}`));
		const { container } = render(<Transcript pid="s" entries={long} />);
		const scroller = container.querySelector<HTMLElement>(".lo-scroll")!;
		const spacer = scroller.querySelector(":scope > [data-bottom-anchor]")!;
		const box = layout(scroller).get(spacer)!;
		expect(box.bottom - box.top).toBe(0);
	});
});
