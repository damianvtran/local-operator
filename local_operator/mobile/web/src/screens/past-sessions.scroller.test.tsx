// @vitest-environment happy-dom
//
// `#/past` SCROLLS ITS LIST, NOT THE WINDOW.
//
// The shell pins the document (`src/styles/index.css`, "the document is not a
// scroller"), so a screen that relies on the window scrolling has no scroller at
// all. This one did: its root declared a min-HEIGHT (a floor, not a bound), so the
// column grew with
// its 40 rows — with no inner scroller, and the rows past the fold were simply
// unreachable (measured on the real bundle at 390x844: the 22nd row's box sat at
// y 1250-1302 in an 844px viewport, 458px below the last pixel a thumb can
// reach).
//
// WHAT THIS LAYER PROVES, and what it cannot. happy-dom does no layout (every box
// is 0x0) and carries no Tailwind stylesheet, so `getComputedStyle` cannot tell a
// bounded column from a tall one — that is the lesson
// `session-view.overflow.test.tsx` records. What IS answerable here is the
// STRUCTURE that makes the bound work: one element that scrolls, that contains
// every row, that the header and search field sit OUTSIDE of, and that is a child
// of the height-bounded column rather than of the document. Those assertions fail
// on the old shape (no scroller ancestor exists at all) and on the plausible half
// fix (an `overflow-y-auto` root, which would put the header and the search field
// inside the scrolling box and scroll them away with the list).
//
// The GEOMETRIC half — that a real touch drag reaches the last row — is measured
// one layer up on the built bundle in headless Chrome, with real
// `Input.dispatchTouchEvent` drags, by `scripts/mobile_overflow_capture.py`
// (`#/past` at 390x844 and 360x780, before and after). Do not read a green run of
// this file as proof of reachability.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PastSession } from "../types";

const searchSessions = vi.fn(async (_q: string, _limit?: number) => ({
	sessions: [] as PastSession[],
	query: "",
}));

vi.mock("../api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../api")>();
	return {
		...actual,
		searchSessions: (q: string, limit?: number) => searchSessions(q, limit),
		resumeSession: vi.fn(),
	};
});

import { PastSessionsScreen } from "./past-sessions";

/** The daemon asks for up to 40 (`api.ts`, `searchSessions`); 22 is the count
    the review and QA rounds measured the unreachable tail with. */
function past(index: number): PastSession {
	return {
		id: `6714def8${String(index).padStart(4, "0")}`,
		name: `conversation ${index}`,
		mtime: 1_792_420_990 + index,
		// Every third row forks, so the row height is not uniform and the list is
		// the shape a real history has.
		forked: index % 3 === 0,
	};
}

function rows(count: number): PastSession[] {
	return Array.from({ length: count }, (_, index) => past(index + 1));
}

/** The refused height floor, assembled rather than written out: `scripts/check-bundle.mjs`
    treats a class-shaped literal anywhere under `src/` as one the app renders with, so
    spelling it here would ship a selector the phone never uses. */
const FLOOR_CLASS = ["min", "h", "full"].join("-");

/** The class list of an element, as tokens. Tokens rather than substrings: a
    a substring test also matches a longer arbitrary value, and a bare `h-dvh`
    check matches the FLOOR spelling `min-` + `h-dvh` — the regression itself. */
function tokens(element: Element): string[] {
	return (element.getAttribute("class") ?? "").split(/\s+/).filter(Boolean);
}

function screenRoot(container: HTMLElement): HTMLElement {
	const root = container.firstElementChild;
	if (!(root instanceof HTMLElement)) throw new Error("the screen rendered no root");
	return root;
}

/** The one element the rows live in. Selected by the app's own scroller marker
    (`.lo-scroll`), which is what `TOUCH_TARGET_JS` in the capture rig drives —
    so a scroller this test cannot find is one the rig cannot drive either. */
function scrollerAround(element: Element): HTMLElement {
	const found = element.closest<HTMLElement>(".lo-scroll");
	if (!found) throw new Error("the row is not inside any .lo-scroll element");
	return found;
}

/** The screen's scroller, found from the outside: used by the state tests, where
    the assertion is "this line of text lives in THE scroller", not "in some
    scroller". */
function screenScroller(container: HTMLElement): HTMLElement {
	const found = container.querySelector<HTMLElement>(".lo-scroll");
	if (!found) throw new Error("the screen rendered no scroller");
	return found;
}

afterEach(() => {
	cleanup();
	searchSessions.mockReset();
});

describe("the past sessions column", () => {
	it("puts 22 rows in the column's own scroller, with the header and search outside it", async () => {
		searchSessions.mockResolvedValue({ sessions: rows(22), query: "" });
		const { container } = render(<PastSessionsScreen />);

		const root = screenRoot(container);
		const last = await screen.findByText("conversation 22");
		const scroller = scrollerAround(last);

		// EVERY row is in it, not merely the first or the last: a scroller that
		// wrapped only part of the list would still leave the rest clipped.
		for (let index = 1; index <= 22; index += 1) {
			const row = screen.getByText(`conversation ${index}`);
			expect(scroller.contains(row), `row ${index} is outside the scroller`).toBe(true);
		}

		// The header and the search field are PINNED: inside the scroller they
		// would scroll away with the list, which is the half fix this catches.
		const heading = screen.getByRole("heading");
		const search = screen.getByRole("textbox");
		expect(scroller.contains(heading)).toBe(false);
		expect(scroller.contains(search)).toBe(false);

		// The scroller is the column's OWN flex child, so its height comes from
		// the bounded column rather than from its content.
		expect(scroller.parentElement).toBe(root);

		// The bound and the clip, as declared tokens — see the module note for why
		// this layer cannot resolve them to pixels.
		expect(tokens(root)).toEqual(expect.arrayContaining(["h-dvh", "overflow-hidden"]));
		expect(tokens(root)).not.toContain(FLOOR_CLASS);
		// `min-h-0` is what lets a flex child shrink below its content; without it
		// a flex child defaults to `min-height: auto` and grows the column instead
		// of scrolling.
		for (const token of ["lo-scroll", "flex-1", "min-h-0", "overflow-y-auto"]) {
			expect(tokens(scroller), `scroller is missing ${token}`).toContain(token);
		}
	});

	it("keeps the loading, empty and error states inside that same scroller", async () => {
		// The three states a list of rows is not in. Each must paint inside the
		// scroller the fix installs: outside it they would sit in a clipped column
		// with nothing to scroll, which is how a tall error message gets its tail
		// cut off.
		let settle: (value: { sessions: PastSession[]; query: string }) => void = () => {};
		searchSessions.mockReturnValue(
			new Promise((resolve) => {
				settle = resolve;
			}),
		);
		const loading = render(<PastSessionsScreen />);
		const loadingText = await screen.findByText(/loading/u);
		expect(scrollerAround(loadingText)).toBe(screenScroller(loading.container));
		cleanup();
		// Settle the unmounted screen's promise so nothing is left in flight.
		settle({ sessions: [], query: "" });

		searchSessions.mockResolvedValue({ sessions: [], query: "" });
		const empty = render(<PastSessionsScreen />);
		const emptyText = await screen.findByText("no past sessions yet");
		expect(scrollerAround(emptyText)).toBe(screenScroller(empty.container));
		cleanup();

		searchSessions.mockRejectedValue(new Error("store unavailable"));
		const failed = render(<PastSessionsScreen />);
		const errorText = await screen.findByText("store unavailable");
		expect(scrollerAround(errorText)).toBe(screenScroller(failed.container));
		// The header is still outside the scroller in the error state, where the
		// list is replaced by a single line.
		const heading = screen.getByRole("heading");
		expect(screenRoot(failed.container).contains(heading)).toBe(true);
		expect(scrollerAround(errorText).contains(heading)).toBe(false);
	});
});
