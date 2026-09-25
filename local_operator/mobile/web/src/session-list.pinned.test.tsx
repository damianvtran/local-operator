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

/* QA Q18: the heaviest tests in this file are SECONDS of real work under load —
   a module re-import per case, four scroll positions in a loop, waits on
   animation frames — and the default 5s per-test budget is what they trip
   (4 of 220 in the full suite at load, 220/220 with `--testTimeout=30000`).
   Raised HERE, on the tests that do that work, and not suite-wide: the budget
   is a property of what a test does, not of the runner, and a global raise
   would hide a genuinely hung test everywhere else. */
const SLOW = 30_000;

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
	}, SLOW);

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
	}, SLOW);

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
	}, SLOW);

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

describe("a pin press reorders nothing until the daemon confirms (Q13/Q14/Q15/D18)", () => {
	/* THE REAL STORE, FED THE FRAMES A DAEMON SENDS. The blocks above stub the
	   store's optimistic half away, which is exactly the half under test here:
	   with `useSessions` pinned to the test's own array, no press can reorder
	   anything and a test written up there would pass on the defect as it
	   stands. So this block re-imports the module with only the list STREAM
	   replaced, and everything above it — the frame handler, the mark
	   settlement, the screen's own `applySessionPin` — is the production one.
	   `vi.doMock` here beats the file-level `vi.mock` for the dynamic import
	   below, which is what makes the re-import possible at all. */
	const ROWS = 12;
	const NAMES = Array.from({ length: ROWS }, (_, index) => `Row ${index}`);

	/* One card at 100% root font, and the band's box at the two scales Q13 and
	   Q15 were measured at. happy-dom lays nothing out, so the geometry the
	   pressed-row rule reads is supplied below. */
	const ROW_H = 56;
	const ROW_H_200 = 112;
	const BAND_H = 44;
	const BAND_H_200 = 100;
	const COLUMN = 740;

	function rowsOf(count: number, pinned: string[] = []): SessionSummary[] {
		return Array.from({ length: count }, (_, index) =>
			summary({
				session_id: `r${index}`,
				conversation_name: `Row ${index}`,
				pinned: pinned.includes(`r${index}`),
			}),
		);
	}

	/* THE DAEMON'S TRANSPORT, faked at the one place the browser owns it. The
	   store's own plumbing — the frame handler, the mark settlement, the
	   refcount — stays real, which is the point: the defect lives in that code,
	   not in EventSource. */
	let opened: FakeEventSource[] = [];
	class FakeEventSource {
		readonly listeners = new Map<string, Array<(event: { data: string }) => void>>();
		onopen: (() => void) | null = null;
		onerror: (() => void) | null = null;
		constructor(readonly url: string) {
			opened.push(this);
		}
		addEventListener(event: string, listener: (event: { data: string }) => void) {
			const list = this.listeners.get(event) ?? [];
			list.push(listener);
			this.listeners.set(event, list);
		}
		removeEventListener() {}
		close() {}
	}

	/* The real store, and a stream with no state of its own: a fresh module per
	   test, so a source left over from the previous one is never the one that
	   answers. Renders, because every case below starts from the daemon's first
	   frame and that is all it starts from. */
	async function realStoreHarness() {
		vi.resetModules();
		vi.doUnmock("./store");
		opened = [];
		vi.stubGlobal("EventSource", FakeEventSource);
		const { SessionListScreen: Screen } = await import(
			"./screens/session-list"
		);
		render(<Screen />);
		expect(opened).toHaveLength(1);
	}

	/* One list frame, exactly as the daemon writes it. Awaited `act`, because the
	   store hands the frame to React from outside React's own event handling and a
	   synchronous `act` scope does not flush that update before the next assertion
	   (measured: the DOM still showed the previous frame's ★ after the push). */
	async function pushFrame(sessions: SessionSummary[]) {
		await act(async () => {
			for (const source of opened) {
				for (const listener of source.listeners.get("sessions") ?? []) {
					listener({ data: JSON.stringify({ sessions }) });
				}
			}
		});
	}

	function mainScroller(): HTMLElement {
		const found = cardByName("Row 0").closest(".overflow-y-auto");
		expect(found).not.toBeNull();
		return found as HTMLElement;
	}

	/* THE ROWS IN THE ORDER THE BROWSER LAYS THEM OUT. This is the instrument
	   that matters for both defects: the scroll the reader saw was the browser
	   answering a REORDER, so the thing to assert about is the rendered order
	   of the rows, not any computed position. Names are matched longest-first
	   because `Row 1` is a prefix of `Row 10`. */
	function rowOrder(list: HTMLElement): string[] {
		const longestFirst = [...NAMES].sort((a, b) => b.length - a.length);
		return Array.from(list.querySelectorAll("button")).map(
			(card) =>
				longestFirst.find((name) => (card.textContent ?? "").includes(name)) ??
				"?",
		);
	}

	function pinnedSection(): boolean {
		return screen.queryByText("★ Pinned") !== null;
	}

	function starOn(name: string): boolean {
		return cardByName(name).querySelector("[aria-label=\"pinned\"]") !== null;
	}

	/* The column, modelled: one `rowHeight`-tall card per rendered row, in DOM
	   order, offset by the scroller's own `scrollTop`. What it buys is the one
	   thing these tests are about — a row's position is a function of the ORDER
	   it is rendered in, so a reorder moves every row below it exactly as it
	   does in a browser, and the screen's own `getBoundingClientRect` reads
	   answer about that model. `writes` counts only the SCREEN's writes. */
	function modelColumn(
		list: HTMLElement,
		geometry: { rowHeight: number; bandHeight: number },
	) {
		let offset = 0;
		let writes = 0;
		const cards = () =>
			Array.from(list.querySelectorAll("button")) as HTMLElement[];
		const box = (top: number, height: number): DOMRect =>
			({
				top,
				bottom: top + height,
				height,
				left: 0,
				right: 0,
				width: 0,
				x: 0,
				y: top,
				toJSON: () => ({}),
			}) as DOMRect;
		Object.defineProperty(list, "scrollTop", {
			configurable: true,
			get: () => offset,
			set: (value: number) => {
				writes += 1;
				offset = Math.max(0, Math.round(value));
			},
		});
		Object.defineProperty(list, "clientHeight", {
			configurable: true,
			get: () => COLUMN,
		});
		Object.defineProperty(list, "scrollHeight", {
			configurable: true,
			get: () => cards().length * geometry.rowHeight,
		});
		const band = screen.getByRole("alert").parentElement as HTMLElement;
		Object.defineProperty(band, "offsetHeight", {
			configurable: true,
			get: () => geometry.bandHeight,
		});
		const realRect = HTMLElement.prototype.getBoundingClientRect;
		HTMLElement.prototype.getBoundingClientRect = function (this: HTMLElement) {
			if (this === list) return box(0, COLUMN);
			const index = cards().indexOf(this);
			if (index < 0) return realRect.call(this);
			return box(index * geometry.rowHeight - offset, geometry.rowHeight);
		};
		return {
			bandBottom: () => geometry.bandHeight,
			offset: () => offset,
			writes: () => writes,
			scrollTo(value: number) {
				offset = Math.max(0, Math.round(value));
			},
			restore() {
				HTMLElement.prototype.getBoundingClientRect = realRect;
			},
		};
	}

	/* The two frames the pressed-row rule waits for, waited for: the screen's
	   pair was scheduled at the commit, so a pair scheduled after it runs after
	   it — the move has been made by the time this resolves. */
	async function settled() {
		await act(async () => {
			await new Promise((resolve) =>
				requestAnimationFrame(() => requestAnimationFrame(resolve)),
			);
			await Promise.resolve();
		});
	}

	async function refusedPinOn(name: string) {
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		longPress(cardByName(name));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).not.toBe(""),
		);
	}

	const restores: Array<() => void> = [];
	afterEach(() => {
		while (restores.length > 0) restores.pop()?.();
		/* The faked transport goes with the test that faked it. */
		vi.unstubAllGlobals();
		vi.doUnmock("./store");
	});

	it("shows the ★ on the press, and leaves the list alone until the daemon answers (Q13/D18)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(4));
		const list = mainScroller();
		const before = rowOrder(list);

		longPress(cardByName("Row 2"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));

		/* Instant feedback: the ★ is drawn in the commit that handles the tap. */
		expect(starOn("Row 2")).toBe(true);
		/* And the list has not moved: no section, same order. THE DEFECT WAS
		   HERE — the row lifted into ★ Pinned on this commit, and the browser
		   answering that reorder is what scrolled the reader. */
		expect(pinnedSection()).toBe(false);
		expect(rowOrder(list)).toEqual(before);

		/* The POST resolving is not a confirmation either: the daemon's list
		   frame is, and only it moves the row. */
		await waitFor(() => expect(setSessionPin).toHaveBeenCalledWith("r2", true));
		await settled();
		expect(pinnedSection()).toBe(false);
		expect(rowOrder(list)).toEqual(before);

		await pushFrame(rowsOf(4, ["r2"]));
		expect(pinnedSection()).toBe(true);
		expect(rowOrder(list)[0]).toBe("Row 2");
	}, SLOW);

	it("reorders nothing at all when the pin is refused, at 100% and at 200% (Q13/D18)", async () => {
		for (const [scale, rowHeight, bandHeight] of [
			["100%", ROW_H, BAND_H],
			["200%", ROW_H_200, BAND_H_200],
		] as Array<[string, number, number]>) {
			await realStoreHarness();
			await pushFrame(rowsOf(6));
			const list = mainScroller();
			const model = modelColumn(list, { rowHeight, bandHeight });
			restores.push(() => model.restore());
			const before = rowOrder(list);

			await refusedPinOn("Row 3");
			await settled();

			/* Both halves of the property, at both scales: the refusal is on
			   screen, the ★ it refuted has fallen back, no ★ Pinned section ever
			   existed, and the rows are in the order they were — nothing moved,
			   so there is nothing to move back. The pixel response of the old
			   optimistic lift (−51.0px at 100%, −101.5px at 200%) came from this
			   reorder happening; it cannot happen from here. */
			expect(screen.getByRole("alert").textContent).toContain(
				"no saved messages yet",
			);
			expect(starOn("Row 3"), `the ★ at ${scale}`).toBe(false);
			expect(pinnedSection(), `a ★ Pinned section at ${scale}`).toBe(false);
			expect(rowOrder(list), `rows moved at ${scale}`).toEqual(before);

			cleanup();
		}
	}, SLOW);

	it("moves rows for a pin the daemon confirms, and only by that one row (Q14, accepted)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(6));
		const list = mainScroller();
		const before = rowOrder(list);

		/* Another client pins Row 1. The row leaves Active for ★ Pinned and the
		   rows below it move by at most its own height: a CONFIRMED pin reorders
		   the list, which the maintainer accepted and the PR documents. */
		await pushFrame(rowsOf(6, ["r1"]));
		expect(pinnedSection()).toBe(true);
		expect(rowOrder(list)).toEqual([
			"Row 1",
			...before.filter((name) => name !== "Row 1"),
		]);

		/* And the refusal that follows moves nothing further: the same gesture,
		   one answered and one not, and only the answered one may reorder. */
		const confirmed = rowOrder(list);
		await refusedPinOn("Row 4");
		await settled();
		expect(rowOrder(list)).toEqual(confirmed);
	}, SLOW);

	it("keeps the pressed row clear of the band once the list has settled, at 200% (Q15/D20)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(ROWS));
		const list = mainScroller();
		const model = modelColumn(list, {
			rowHeight: ROW_H_200,
			bandHeight: BAND_H_200,
		});
		restores.push(() => model.restore());
		/* The reader is at the top of the row they press: at 200% root font the
		   band covers that row, so the reason would be unreadable beside it. */
		model.scrollTo(3 * ROW_H_200);

		await refusedPinOn("Row 3");
		await settled();

		expect(cardByName("Row 3").getBoundingClientRect().top).toBeGreaterThanOrEqual(
			model.bandBottom(),
		);
		/* The minimum that clears the band, written once. Before the fix this
		   effect ran in the commit that carried the error text, against a
		   layout the reader never saw — the row's pre-lift position — so its
		   own condition was false and it never ran again (writes stayed 0). */
		expect(model.writes()).toBe(1);
		expect(model.offset()).toBe(3 * ROW_H_200 - BAND_H_200);
	}, SLOW);

	it("leaves the scroll untouched when the pressed row is already clear of the band", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(ROWS));
		const list = mainScroller();
		const model = modelColumn(list, {
			rowHeight: ROW_H_200,
			bandHeight: BAND_H_200,
		});
		restores.push(() => model.restore());
		/* Scrolled PAST the row that was pressed: the reason is behind the
		   reader, and the rule must not drag them back to a row they left. */
		model.scrollTo(6 * ROW_H_200);

		await refusedPinOn("Row 3");
		await settled();

		expect(model.writes()).toBe(0);
		expect(model.offset()).toBe(6 * ROW_H_200);
	}, SLOW);

	it("never shows the ★ Pinned section across a refused pin — nothing ever reordered (D18)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(6));
		const list = mainScroller();
		const before = rowOrder(list);

		/* Sampled at each commit the refusal passes through. The ★ on the row is
		   the control that proves the samples can see a change at all: if the
		   section is absent in every one of them, it is because no commit
		   contained it, not because the sampler was looking at the wrong DOM. */
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		longPress(cardByName("Row 3"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		const pressed = { star: starOn("Row 3"), section: pinnedSection() };
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).not.toBe(""),
		);
		await settled();
		const refused = { star: starOn("Row 3"), section: pinnedSection() };
		/* The daemon's own frame, carrying the truth, closes the sequence out. */
		await pushFrame(rowsOf(6));
		const reported = pinnedSection();

		expect(pressed.star).toBe(true);
		expect([pressed.section, refused.section, reported]).toEqual([
			false,
			false,
			false,
		]);
		expect(refused.star).toBe(false);
		expect(rowOrder(list)).toEqual(before);
	}, SLOW);
});
