// @vitest-environment happy-dom
//
// ★ Pinned, and the pin gesture. Two things are asserted against the REAL
// SessionListScreen rather than a copy:
//
//   * the SECTION: a pinned row appears under `★ Pinned` and ONLY there (the
//     sidebar's own rule — a pin lifts a row out of the section it ranked into),
//     and an unpinned row does not;
//   * the GESTURE: a long-press opens the pin action sheet, and a scroll
//     (movement past the threshold) cancels it, so a flick through the list
//     cannot open a sheet on whatever row the finger passed over.
//
// The pin POST is mocked; the store's optimistic half is exercised for real so
// the row is seen to move for the reason it will in production.
import {
	act,
	cleanup,
	fireEvent,
	render,
	screen,
	waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
const setSessionPin = vi.fn(async (_id: string, _pinned: boolean) => ({ ok: true, pinned: true }));
/* The store's optimistic write is spied so the assertion is about the SCREEN's
   call, not a re-implementation of the store's internals. */
const applySessionPin = vi.fn();
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true }),
		retainSessionListStream: () => () => {},
		applySessionPin: (id: string, pinned: boolean) => applySessionPin(id, pinned),
	};
});
vi.mock("./api", () => ({
	getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
	setSessionPin: (id: string, pinned: boolean) => setSessionPin(id, pinned),
}));

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "",
		model_label: "",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		subagents_queued: 0,
		todos_open: 0,
		mtime: 0,
		pinned: false,
		...over,
	};
}

function cardByName(name: string): HTMLButtonElement {
	return screen.getByRole("button", { name: new RegExp(name) });
}

/* A long-press: pointerdown, hold past the threshold (fake timers), release.
   The timer callback sets React state, so the advance is wrapped in `act` —
   without it the sheet's mount is not flushed and a `queryByRole` after it
   would return null for BOTH a cancelled press and a fired one, which is how a
   scroll-cancel test passes vacuously. */
function longPress(card: HTMLElement) {
	vi.useFakeTimers();
	fireEvent.pointerDown(card, { clientX: 10, clientY: 10 });
	act(() => {
		vi.advanceTimersByTime(600);
	});
	vi.useRealTimers();
}

/* The same, but the finger travels before the hold elapses — a scroll. */
function scrollDuringPress(card: HTMLElement) {
	vi.useFakeTimers();
	fireEvent.pointerDown(card, { clientX: 10, clientY: 10 });
	fireEvent.pointerMove(card, { clientX: 10, clientY: 50 });
	act(() => {
		vi.advanceTimersByTime(600);
	});
	vi.useRealTimers();
}

afterEach(() => {
	cleanup();
	sessionList = [];
	setSessionPin.mockClear();
	applySessionPin.mockClear();
});

describe("★ Pinned section", () => {
	it("shows a pinned row under ★ Pinned and not in Active or Previous", () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Pinned", pinned: true }),
			summary({ session_id: "a1", conversation_name: "Unpinned", section: "active" }),
		];
		render(<SessionListScreen />);
		expect(screen.getByText("★ Pinned")).toBeTruthy();
		/* The row is rendered exactly once — the pinned copy — not also in Active. */
		expect(screen.getAllByRole("button", { name: /Pinned/ })).toHaveLength(1);
		expect(screen.getByRole("button", { name: /Unpinned/ })).toBeTruthy();
	});

	it("omits the ★ Pinned heading entirely when nothing is pinned", () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		expect(screen.queryByText("★ Pinned")).toBeNull();
	});

	it("marks a pinned card with an accessible ★", () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Pinned", pinned: true }),
		];
		render(<SessionListScreen />);
		const card = cardByName("Pinned");
		const star = card.querySelector("[aria-label=\"pinned\"]");
		expect(star).toBeTruthy();
		expect(star?.textContent).toContain("★");
	});

	it("carries no ★ on an unpinned card", () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		expect(cardByName("Alpha").querySelector("[aria-label=\"pinned\"]")).toBeNull();
	});
});

describe("the pin gesture", () => {
	it("opens the pin action sheet on a long-press and pins through the shared API", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));

		const action = await screen.findByRole("button", { name: "Pin to the top" });
		fireEvent.click(action);
		/* The optimistic store write and the POST both go to the same row. */
		expect(applySessionPin).toHaveBeenCalledWith("a1", true);
		await waitFor(() => expect(setSessionPin).toHaveBeenCalledWith("a1", true));
	});

	it("offers Unpin for an already-pinned row", async () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Pinned", pinned: true }),
		];
		render(<SessionListScreen />);
		longPress(cardByName("Pinned"));
		expect(await screen.findByRole("button", { name: "Unpin from the top" })).toBeTruthy();
	});

	it("cancels the long-press when the finger moves, so a scroll opens nothing", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		scrollDuringPress(cardByName("Alpha"));
		/* The press must NOT have opened the sheet. Proven against the positive
		   case in the next test, which asserts the SAME query finds it — so this
		   cannot pass merely because the flush is wrong. */
		expect(screen.queryByRole("button", { name: "Pin to the top" })).toBeNull();
	});

	it("DOES open the sheet on an unmoved press (the control for the cancel above)", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));
		expect(await screen.findByRole("button", { name: "Pin to the top" })).toBeTruthy();
	});

	it("does not navigate when a long-press opens the sheet", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		const card = cardByName("Alpha");
		longPress(card);
		await screen.findByRole("button", { name: "Pin to the top" });
		/* The click that follows the release must be swallowed. */
		fireEvent.click(card);
		expect(window.location.hash).not.toBe("#/s/a1");
	});
});

describe("empty sections cost no heading (design round 1, D1)", () => {
	it("omits the Active heading when every active row is pinned away", () => {
		/* A pin LIFTS a row out of its ranked section, so pinning the only
		   active row leaves Active empty — newly reachable because of pinning.
		   The sidebar's own rule is that an empty section contributes no
		   heading, and the ★ Pinned section already followed it. */
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Pinned", pinned: true }),
		];
		render(<SessionListScreen />);
		expect(screen.getByText("★ Pinned")).toBeTruthy();
		expect(screen.queryByText("Active Sessions")).toBeNull();
		expect(screen.queryByText("Previous Sessions")).toBeNull();
	});

	it("omits the Previous heading when there are no previous rows", () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		expect(screen.getByText("Active Sessions")).toBeTruthy();
		expect(screen.queryByText("Previous Sessions")).toBeNull();
	});

	it("says so when a search matches nothing at all", () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		const input = screen.getByPlaceholderText("Search conversations…");
		fireEvent.change(input, { target: { value: "zzz-no-match" } });
		expect(screen.getByText("no matching conversations")).toBeTruthy();
		expect(screen.queryByText("Active Sessions")).toBeNull();
	});
});

describe("the list teaches its own pin gesture (design round 1, D2)", () => {
	/* The caption is ALWAYS MOUNTED and collapses via grid rows (design round 2, D8
	   — an abrupt unmount snapped the list ~23px; round 3 replaced a fixed
	   max-height cap, which clipped a WRAPPED caption, with the 0fr/1fr track
	   trick). So the tests read the wrapper's row class and its `aria-hidden`,
	   which is the same signal a sighted reader and assistive tech get. */
	function hintBox(): HTMLElement {
		/* The grid wrapper: <div grid> > <div overflow-hidden> > <p>. The inner
		   overflow-hidden child is asserted, not just walked past: without it a
		   `1fr` track paints at full height and the collapse silently stops
		   reaching zero (review round 4, MINOR 1). */
		const wrapper = screen.getByText("touch and hold a row to pin it")
			.parentElement as HTMLElement;
		expect(wrapper.className).toContain("overflow-hidden");
		return wrapper.parentElement as HTMLElement;
	}

	it("shows the hint while nothing is pinned", () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		expect(screen.getByText("touch and hold a row to pin it")).toBeTruthy();
		expect(hintBox().className).toContain("grid-rows-[1fr]");
		expect(hintBox().getAttribute("aria-hidden")).toBeNull();
	});

	it("collapses the hint once a pin exists, whose ★ Pinned section is the discoverer", () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Pinned", pinned: true }),
			summary({ session_id: "a1", conversation_name: "Alpha" }),
		];
		render(<SessionListScreen />);
		expect(hintBox().className).toContain("grid-rows-[0fr]");
		expect(hintBox().getAttribute("aria-hidden")).toBe("true");
		expect(screen.getByText("★ Pinned")).toBeTruthy();
	});

	it("collapses the hint when a query matches no row at all (D5)", () => {
		/* The caption names a row to hold, so it must never sit above
		   "no matching conversations". */
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		fireEvent.change(screen.getByPlaceholderText("Search conversations…"), {
			target: { value: "zzz-no-match" },
		});
		expect(hintBox().className).toContain("grid-rows-[0fr]");
		expect(screen.getByText("no matching conversations")).toBeTruthy();
	});

	it("stays collapsed when a search merely HIDES the pinned rows (D6)", () => {
		/* A pin exists in the store but the query hides it, so `pinned` (the
		   VISIBLE pinned list) is empty — the gate must read the store, not the
		   filtered list, or the hint would return here. */
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Zebra pinned", pinned: true }),
			summary({ session_id: "a1", conversation_name: "Alpha" }),
		];
		render(<SessionListScreen />);
		fireEvent.change(screen.getByPlaceholderText("Search conversations…"), {
			target: { value: "alpha" },
		});
		expect(hintBox().className).toContain("grid-rows-[0fr]");
	});
});

describe("a refused pin reports where the reader is looking (D12/D14, R8-2, Q4)", () => {
	/* A list LONGER THAN THE SCREEN, which is the case that exposed both earlier
	   placements: after the last row (D12) and above the first row inside the
	   scroller (D14). With a handful of rows both were on screen, so a short list
	   would pass either way. */
	const ROWS = 32;

	function longList(): SessionSummary[] {
		return Array.from({ length: ROWS }, (_, index) =>
			summary({ session_id: `r${index}`, conversation_name: `Row ${index}` }),
		);
	}

	/* The band: <div absolute> > <p role=alert>. Its positioning is asserted on
	   the way in rather than walked past, because every test below leans on the
	   band being an overlay, and a band that slid back into the flow would have
	   them asserting on the wrong element's classes. */
	function errorBox(): HTMLElement {
		const band = screen.getByRole("alert").parentElement as HTMLElement;
		expect(band.className).toContain("absolute");
		return band;
	}

	/* The band's two states are spelled by these classes alone (hidden:
	   slid up, faded, click-through; shown: in place, opaque). */
	const HIDDEN = ["-translate-y-full", "opacity-0", "pointer-events-none"];
	const SHOWN = ["translate-y-0", "opacity-100"];

	function expectBand(state: string[], not: string[]) {
		const classes = errorBox().className.split(/\s+/);
		for (const name of state) expect(classes).toContain(name);
		for (const name of not) expect(classes).not.toContain(name);
	}

	/* The list's scroll container, found by what makes it one (its overflow
	   class), from a row — not by tag, so a refactor that moves the scroll onto a
	   different element is still what these tests ask about. */
	function scroller(): HTMLElement {
		const found = cardByName("Row 0").closest(".overflow-y-auto");
		expect(found).not.toBeNull();
		return found as HTMLElement;
	}

	/* THE PROPERTY, structurally, because happy-dom has no layout: no
	   `getBoundingClientRect` or `scrollTop` here moves anything, so a pixel
	   assertion would pass on every placement. What keeps the band on screen at
	   any scroll position is that it is NOT scroll content, and what keeps it
	   out of the scroller's layout is that it is taken out of the flow: an
	   `absolute` sibling of the scroller, pinned to the top of a `relative`
	   parent they share. An absolutely positioned child of a flex column is not
	   a flex item, so its height can never come out of the scroller's. The
	   rendered frames at 100% and 200% root font on the PR are the pixel half of
	   this claim. */
	function expectOutsideTheScroller(alert: HTMLElement) {
		const list = scroller();
		expect(list.contains(alert)).toBe(false);
		expect(alert.closest(".overflow-y-auto")).toBeNull();
		const band = errorBox();
		expect(band.parentElement).toBe(list.parentElement);
		const classes = band.className.split(/\s+/);
		for (const name of ["absolute", "inset-x-0", "top-0"]) expect(classes).toContain(name);
		/* The overlay is positioned against the list's own wrapper, not the page,
		   so it sits over the list's top edge rather than over the header. */
		expect((band.parentElement as HTMLElement).className.split(/\s+/)).toContain("relative");
	}

	async function refuseAPinOn(name: string) {
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		longPress(cardByName(name));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		return waitFor(() => {
			const alert = screen.getByRole("alert");
			expect(alert.textContent).not.toBe("");
			return alert;
		});
	}

	it("renders the refusal outside the scroll container, above it", async () => {
		sessionList = longList();
		render(<SessionListScreen />);
		const alert = await refuseAPinOn("Row 0");
		expect(alert.textContent).toContain("Could not save the pin: no saved messages yet");
		expectOutsideTheScroller(alert);
	});

	it("reports a refusal on a row reached by scrolling down the list", async () => {
		/* The reader's natural position (Q4, D14): scrolled to a row far below the
		   first screen. The scroll is simulated — happy-dom records `scrollTop` but
		   lays nothing out — so what is asserted is the structure that makes the
		   band independent of it: the scroll happened on the element the band is
		   not inside, and the scroll position is left where the reader put it (the
		   refusal does not yank the list back to the top to show itself). */
		sessionList = longList();
		render(<SessionListScreen />);
		const list = scroller();
		list.scrollTop = 900;
		fireEvent.scroll(list);
		const alert = await refuseAPinOn(`Row ${ROWS - 2}`);
		expect(alert.textContent).toContain("Could not save the pin: no saved messages yet");
		expectOutsideTheScroller(alert);
		expect(list.scrollTop).toBe(900);
		expectBand(SHOWN, HIDDEN);
	});

	/* happy-dom lays nothing out and never clamps `scrollTop`, so the scroller
	   gets a small model of the column it sits in: a fixed viewport over a
	   fixed amount of content, with `scrollTop` rounded to whole pixels, clamped
	   into the scroll range, and every write the SCREEN makes to it counted. The
	   band is an overlay, so nothing about it is an input to this model — which
	   is the claim: if a later change put the band back in the flow or brought a
	   `scrollTop` compensation back, the writes counted here stop being zero. */
	const COLUMN = 740;
	const CONTENT = 3000;
	const BOTTOM = CONTENT - COLUMN;

	function modelTheColumn(list: HTMLElement) {
		let offset = 0;
		let writes = 0;
		const clamp = (value: number) => Math.min(Math.max(Math.round(value), 0), BOTTOM);
		Object.defineProperty(list, "scrollTop", {
			configurable: true,
			get: () => offset,
			set: (value: number) => {
				writes += 1;
				offset = clamp(value);
			},
		});
		Object.defineProperty(list, "scrollHeight", { configurable: true, get: () => CONTENT });
		Object.defineProperty(list, "clientHeight", { configurable: true, get: () => COLUMN });
		return {
			/* The reader scrolls: the offset changes and the browser says so, and
			   the reader's own scroll is not counted as the screen's write. */
			scrollTo(value: number) {
				offset = clamp(value);
				fireEvent.scroll(list);
			},
			writes: () => writes,
		};
	}

	/* Any ResizeObserver the screen builds is recorded with what it watches, so
	   the band's own resize frames can be delivered to it: the old
	   compensation was driven from exactly this callback. */
	type Resized = (entries: Array<{ contentRect: { height: number } }>) => void;
	function observeEverything() {
		const observed: Array<Resized> = [];
		vi.stubGlobal(
			"ResizeObserver",
			class {
				constructor(private readonly callback: Resized) {}
				observe() {
					observed.push(this.callback);
				}
				unobserve() {}
				disconnect() {}
			},
		);
		return observed;
	}

	/* Opens the band with a refusal on a row the band does NOT cover (so the one
	   deliberate move stays out of it), plays the band's frames, fractions
	   included, to anything watching, then closes it with a successful press on
	   another row. Returns the scroll offset after each direction and how many
	   times the screen wrote it. */
	async function openAndClose(at: number) {
		const observed = observeEverything();
		sessionList = longList();
		render(<SessionListScreen />);
		const list = scroller();
		/* Browser scroll anchoring must stay on: opting out moved a scrolled
		   reader's rows on every unrelated list change (Q6, +76px measured). */
		expect(list.className).not.toContain("overflow-anchor");
		const column = modelTheColumn(list);
		column.scrollTo(at);
		const frames = (heights: number[]) =>
			act(() => {
				for (const height of heights) {
					for (const callback of observed) callback([{ contentRect: { height } }]);
				}
			});
		await refuseAPinOn("Row 20");
		frames([0.9, 1.8, 2.7, 3.6, 12.4, 34.8]);
		const opened = list.scrollTop;
		longPress(cardByName("Row 21"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toBe(""));
		frames([22.1, 9.3, 2.7, 0.9, 0]);
		return { opened, closed: list.scrollTop, writes: column.writes() };
	}

	it("holds the rows still while the band opens and closes, whatever the scroll", async () => {
		/* At the top, mid-list, 60px and 20px from the bottom (D14 / Q4 / Q9 /
		   Q10). The rows hold because the screen never touches the scroll: the
		   band paints over the list instead of taking height from it. The old
		   layout sibling needed a compensation here, and QA measured it leaving
		   every row one row higher after the rollback (−51.2px at 100% text). */
		try {
			for (const at of [0, 1200, BOTTOM - 60, BOTTOM - 20]) {
				const { opened, closed, writes } = await openAndClose(at);
				expect(opened).toBe(at);
				expect(closed).toBe(at);
				expect(writes).toBe(0);
				cleanup();
			}
		} finally {
			vi.unstubAllGlobals();
		}
	});

	it("does not pay a collapse twice when the browser already clamped it at the bottom (Q5)", async () => {
		/* THE REGRESSION. At the very bottom, the old layout sibling's collapse
		   grew `<main>`, the browser clamped `scrollTop` by the band's height, and
		   the compensation subtracted it again: QA measured the rows ending 34.2px
		   lower. With the band out of the flow the scroll range never changes, so
		   there is no clamp to pay and nothing paying it twice — the reader is
		   still exactly at the end of the list. */
		try {
			const { opened, closed, writes } = await openAndClose(BOTTOM);
			expect(opened).toBe(BOTTOM);
			expect(closed).toBe(BOTTOM);
			expect(writes).toBe(0);
		} finally {
			vi.unstubAllGlobals();
		}
	});

	it("shows and hides the band without reflowing the list", async () => {
		/* THE PROPERTY THAT RETIRES THE WHOLE CLASS OF DEFECT (Q9–Q11, D18–D21):
		   if showing the band cannot change the scroller's client height, nothing
		   has to be compensated and the browser's anchoring is left alone. happy-dom
		   has no layout engine, so the client height cannot be measured here; what
		   is asserted is the structural guarantee instead. (1) The band is out of
		   the flow: `absolute`, positioned inside the `relative` wrapper it shares
		   with the scroller, so it is not a flex item of that column. (2) Between
		   its two states only transform, opacity and pointer-events change, none of
		   which reflow; no height, grid-track, padding or display class toggles.
		   (3) It transitions only transform and opacity. (4) The scroller itself
		   is identical in both states. */
		sessionList = longList();
		render(<SessionListScreen />);
		const band = errorBox();
		const list = scroller();
		expectOutsideTheScroller(screen.getByRole("alert"));
		expect(band.className).toContain("transition-[transform,opacity]");
		const hidden = new Set(band.className.split(/\s+/));
		const listBefore = list.className;
		await refuseAPinOn("Row 0");
		const shown = new Set(errorBox().className.split(/\s+/));
		const toggled = [...hidden, ...shown].filter((name) => hidden.has(name) !== shown.has(name));
		expect(toggled.length).toBeGreaterThan(0);
		for (const name of toggled) {
			expect(name).toMatch(/^(-?translate-y-|opacity-|pointer-events-)/);
		}
		expect(list.className).toBe(listBefore);
		expect(list.getAttribute("style")).toBeNull();
	});

	it("keeps the band collapsed and empty until a refusal arrives", () => {
		sessionList = longList();
		render(<SessionListScreen />);
		expectBand(HIDDEN, SHOWN);
		expect(screen.getByRole("alert").textContent).toBe("");
		/* Mounted in the fixed region from the first render, so the live region a
		   screen reader registered is the one the refusal later arrives in. */
		expectOutsideTheScroller(screen.getByRole("alert"));
	});

	it("reads the refusal out while the caption is still expanded", async () => {
		/* The band's predicate is its own, NOT the caption's: with nothing pinned
		   the caption is expanded (its own condition), and a refusal can arrive on
		   exactly that list. A shared predicate would have to collapse one of the
		   two here — and the band is the one that must never be hidden while its
		   text is on screen (D12's a11y half). */
		sessionList = longList();
		render(<SessionListScreen />);
		const alert = await refuseAPinOn("Row 0");
		expect(alert.getAttribute("aria-hidden")).toBeNull();
		expect(alert.closest("[aria-hidden]")).toBeNull();
		expectBand(SHOWN, HIDDEN);
		const caption = screen.getByText("touch and hold a row to pin it");
		const captionBox = caption.parentElement?.parentElement as HTMLElement;
		expect(captionBox.className).toContain("grid-rows-[1fr]");
	});
});
