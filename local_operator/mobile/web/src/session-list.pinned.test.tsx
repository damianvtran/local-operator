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
import { HttpError } from "./api";
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
/* The REAL module is spread in, so the test's refusal fixture is the class
   `api.request` actually throws (`HttpError`) rather than a lookalike. */
vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
		setSessionPin: (id: string, pinned: boolean) => setSessionPin(id, pinned),
	};
});

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

/* `hidden` reaches a card the sheet has hidden from assistive tech. The pin
   sheet now STAYS OPEN until the daemon answers (design round 11, D25), and
   `Sheet` marks the list behind it inert + aria-hidden for exactly that window —
   which is the window every ★ assertion below is about. The default stays the
   strict query, so nothing else stops noticing a genuinely hidden card. */
function cardByName(name: string, hidden = false): HTMLButtonElement {
	return screen.getByRole("button", { name: new RegExp(name), hidden });
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
		return cardByName(name, true).querySelector("[aria-label=\"pinned\"]") !== null;
	}

	/* Two animation frames, so a repaint that any part of the mark path
	   scheduled has landed before the assertion that follows. The assertion is
	   about the ORDER, and this keeps it from being answered about a frame
	   nothing has drawn in yet. */
	async function settled() {
		await act(async () => {
			await new Promise((resolve) =>
				requestAnimationFrame(() => requestAnimationFrame(resolve)),
			);
			await Promise.resolve();
		});
	}

	/* A refusal is observed on the mark it takes back: the press paints the ★ and
	   the refused POST clears it again. That is the whole of the response this
	   web UI shows — the band that named the daemon's reason is being rebuilt on
	   its own PR, so the rising and falling ★ is what there is to wait on. The ★
	   is asserted UP first: without that, a wait for it to be gone would pass
	   before the press had been painted at all. */
	async function refusedPinOn(name: string) {
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		longPress(cardByName(name));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		expect(starOn(name)).toBe(true);
		await waitFor(() => expect(starOn(name)).toBe(false));
	}

	afterEach(() => {
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

	it("clears a mark the daemon answered for but did not keep (round 10, MINOR 1)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(4));
		const list = mainScroller();
		const before = rowOrder(list);

		/* A 200 whose body reports the state the route READ BACK — the old one.
		   That is the daemon saying the row is not pinned, and it is the answer
		   the sweep cannot retire: `settlePinMarks` drops a mark only when a
		   frame AGREES with it, so a disagreeing mark would never settle and the
		   ★ would stay on a row the daemon never pinned, for as long as the
		   module lives. No frame is pushed here, deliberately — the guard has to
		   answer the POST's own read-back. */
		setSessionPin.mockResolvedValueOnce({ ok: true, pinned: false });
		longPress(cardByName("Row 1"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		expect(starOn("Row 1")).toBe(true);

		await waitFor(() => expect(starOn("Row 1")).toBe(false));
		expect(pinnedSection()).toBe(false);
		expect(rowOrder(list)).toEqual(before);
	}, SLOW);

	it("reorders nothing at all when the pin is refused (Q13/D18)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(6));
		const list = mainScroller();
		const before = rowOrder(list);

		await refusedPinOn("Row 3");
		await settled();

		/* The property, whole: the ★ the refusal refuted has fallen back, no
		   ★ Pinned section ever existed, and the rows are in the order they were
		   — nothing moved, so there is nothing to move back. The pixel response
		   of the old optimistic lift (−51.0px at 100%, −101.5px at 200% root font)
		   came from that reorder happening; it cannot happen from here.

		   Sampled at ONE root font scale, where it used to be sampled at two: the
		   scale was an input only through the refusal band's own box, which this
		   change moved out to its own PR. What is asserted here is DOM order, and
		   no font size changes it. */
		expect(starOn("Row 3")).toBe(false);
		expect(pinnedSection()).toBe(false);
		expect(rowOrder(list)).toEqual(before);
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

	it("never shows the ★ Pinned section across a refused pin — nothing ever reordered (D18)", async () => {
		await realStoreHarness();
		await pushFrame(rowsOf(6));
		const list = mainScroller();
		const before = rowOrder(list);

		/* Sampled at each commit the refusal passes through. The ★ on the row is
		   the control that proves the samples can see a change at all: if the
		   section is absent in every one of them, it is because no commit
		   contained it, not because the sampler was looking at the wrong DOM.
		   The refusal itself is read off the ★ going back down — the band that
		   named the daemon's reason moved out of this change. */
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		longPress(cardByName("Row 3"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		const pressed = { star: starOn("Row 3"), section: pinnedSection() };
		await waitFor(() => expect(starOn("Row 3")).toBe(false));
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

	it("re-opens a row whose request is still in flight as a dead, explained action (round 12, MAJOR 1)", async () => {
		/* THE ACCEPTANCE SEQUENCE: press a row, dismiss the sheet, re-open THE SAME
		   row while its POST is still outstanding. The pre-fix gate was one
		   screen-wide slot holding "the row last pressed", which the dismissal had
		   already cleared, so the re-opened sheet was LIVE — and its verb and its
		   intent came from the ★, the reader's own unanswered mark, so it offered a
		   live "Unpin from the top" and the press sent `false` for a row the reader
		   had just asked to pin. */
		await realStoreHarness();
		await pushFrame(rowsOf(4));

		/* The POST is held open, so the sheet is re-opened INSIDE the window this
		   defect lives in: a request the daemon has not answered for. */
		let answer: (value: { ok: boolean; pinned: boolean }) => void = () => {};
		setSessionPin.mockImplementationOnce(
			() =>
				new Promise<{ ok: boolean; pinned: boolean }>((resolve) => {
					answer = resolve;
				}),
		);

		longPress(cardByName("Row 1"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		expect(starOn("Row 1")).toBe(true);
		await waitFor(() => expect(setSessionPin).toHaveBeenCalledWith("r1", true));

		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		expect(screen.queryByRole("dialog")).toBeNull();
		longPress(cardByName("Row 1"));

		/* The ★ is still up — the reader's own mark — and the sheet must not read
		   it as this row's state: nothing has been answered, so the verb claims no
		   target state, and the action is dead. */
		const action = (await screen.findByRole("button", {
			name: "Saving…",
		})) as HTMLButtonElement;
		expect(action.disabled).toBe(true);
		expect(action.getAttribute("aria-busy")).toBe("true");
		/* The ★ stays exactly where it was: the reader's own mark is not retracted.
		   It simply stops deciding what the sheet says or what a press sends. */
		expect(starOn("Row 1")).toBe(true);

		/* NOT SILENTLY DEAD: the sheet's own live region names the wait. */
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toContain("Saving"),
		);

		/* And a press here sends NOTHING — least of all the opposite intent. */
		fireEvent.click(action);
		expect(setSessionPin.mock.calls).toEqual([["r1", true]]);

		/* THE SECOND FORM, AND THE ONE THAT COULD SEND THE OPPOSITE: a press on a
		   DIFFERENT row. The pre-fix gate was one screen-wide slot holding "the row
		   last pressed", so Row 2's press evicted Row 1's — Row 1's sheet came back
		   LIVE, read its ★ as this row's state, and offered "Unpin from the top",
		   whose press sent `false` for the request still on its way. */
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		longPress(cardByName("Row 2"));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		await waitFor(() => expect(setSessionPin).toHaveBeenCalledWith("r2", true));
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

		longPress(cardByName("Row 1"));
		const reopened = (await screen.findByRole("button", {
			name: "Saving…",
		})) as HTMLButtonElement;
		expect(reopened.disabled).toBe(true);
		fireEvent.click(reopened);

		/* Two presses, two intents, not one inversion of them. */
		expect(setSessionPin.mock.calls).toEqual([
			["r1", true],
			["r2", true],
		]);

		/* The daemon's answer, which closes the sheet and the sequence out. */
		await act(async () => {
			answer({ ok: true, pinned: true });
		});
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
	}, SLOW);
});

describe("a refused pin says why, in the sheet that asked (design round 11, D25)", () => {
	/* THE DEFECT THIS REPLACES. Closing the sheet on the press left a refusal
	   with nowhere to be said: the ★ appeared at +2.2…5.0ms and cleared at
	   +5.3…20.5ms — a ~3ms existence window, a fifth of a 60Hz frame, so the
	   reader often saw nothing — and on a slow answer it stayed up for the whole
	   wait (306ms, 1516ms, 46.9s measured) and was then withdrawn with no
	   explanation. So the surface the reader acted on stays open until the
	   daemon's answer lands, and the daemon's own reason is rendered inside it. */
	it("keeps the sheet open and shows the daemon's reason in flow inside it", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));

		const action = await screen.findByRole("button", { name: "Pin to the top" });
		/* THE DAEMON'S OWN 409, body and all: `api.request` puts the error body's
		   `error` on the `HttpError`'s message, so this is the sentence a reader
		   actually gets, not a fixture invented here. */
		setSessionPin.mockRejectedValueOnce(
			new HttpError(409, "no saved messages yet — pin it after you send one"),
		);
		fireEvent.click(action);

		const reason = await screen.findByText(
			"Could not save the pin: no saved messages yet — pin it after you send one",
		);
		/* Focus is back ON the action once the wait ends. In a browser `disabled`
		   has already blurred it (the reader is dropped out of the sheet's column
		   mid-wait, and the Sheet's trap only holds its first and last element), so
		   the control the reader pressed is where they come back to. */
		await waitFor(() => expect(document.activeElement).toBe(action));
		/* The sheet is STILL THERE, with the action still on it. */
		const sheet = screen.getByRole("dialog");
		expect(screen.getByRole("button", { name: "Pin to the top" })).toBeTruthy();
		/* The reason is inside it, announced to assistive tech... */
		expect(sheet.contains(reason)).toBe(true);
		expect(reason.getAttribute("role")).toBe("alert");
		/* ...and IN FLOW: a paragraph that is the action's own SIBLING in the
		   sheet's column, so it has to take layout space and cannot be drawn over
		   the control above it. Anything floating (an overlay above the list, a
		   positioned box) fails both of these. */
		expect(reason.parentElement).toBe(action.parentElement);
		expect(reason.className).not.toContain("absolute");
		expect(reason.className).not.toContain("fixed");
		expect(reason.className).not.toContain("z-");
		expect(reason.className).toContain("text-danger");
	});

	it("falls back to a plain line when the daemon's answer carries no reason", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));

		const action = await screen.findByRole("button", { name: "Pin to the top" });
		/* A 409 whose message is only the STATUS — the shape `api.request` produces
		   when a failing response's body is not JSON, and the real spelling of "the
		   daemon gave no reason". `HttpError` is the class the route actually
		   throws, so this exercises the branch the fallback exists for rather than a
		   lookalike `Error` the transport cannot produce. */
		setSessionPin.mockRejectedValueOnce(new HttpError(409, "409"));
		fireEvent.click(action);

		expect(
			await screen.findByText("Could not save the pin: the daemon did not say why"),
		).toBeTruthy();
	});

	it("still closes on a pin the daemon confirms, and the row reaches ★ Pinned", async () => {
		sessionList = [summary({ session_id: "a1", conversation_name: "Alpha" })];
		const view = render(<SessionListScreen />);
		longPress(cardByName("Alpha"));

		const action = await screen.findByRole("button", { name: "Pin to the top" });
		setSessionPin.mockResolvedValueOnce({ ok: true, pinned: true });
		fireEvent.click(action);
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

		/* The daemon's confirming frame is the only thing that moves a row, and
		   the mocked hook reads the list live, so re-rendering is that frame. */
		sessionList = [
			summary({ session_id: "a1", conversation_name: "Alpha", pinned: true }),
		];
		view.rerender(<SessionListScreen />);
		expect(screen.getByText("★ Pinned")).toBeTruthy();
		expect(cardByName("Alpha").querySelector("[aria-label=\"pinned\"]")).toBeTruthy();
	});
});
