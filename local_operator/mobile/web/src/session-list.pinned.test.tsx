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

describe("a refused pin reports where the press happened (design round 5, D12)", () => {
	/* A list LONGER THAN THE SCREEN, which is the case that exposed the placement:
	   with a handful of rows the old bottom-of-list band was on screen too, so a
	   short list would pass either way. */
	const ROWS = 32;

	function longList(): SessionSummary[] {
		return Array.from({ length: ROWS }, (_, index) =>
			summary({ session_id: `r${index}`, conversation_name: `Row ${index}` }),
		);
	}

	/* The band's grid wrapper: <div grid> > <div overflow-hidden> > <p role=alert>,
	   the same shape the caption uses — and the inner child is asserted rather
	   than walked past, because without it a `1fr` track paints at full height and
	   the collapse silently stops reaching zero. */
	function errorBox(): HTMLElement {
		const alert = screen.getByRole("alert");
		const inner = alert.parentElement as HTMLElement;
		expect(inner.className).toContain("overflow-hidden");
		return inner.parentElement as HTMLElement;
	}

	async function refuseAPinOn(name: string) {
		setSessionPin.mockRejectedValueOnce(
			new Error("no saved messages yet — pin it after you send one"),
		);
		render(<SessionListScreen />);
		longPress(cardByName(name));
		fireEvent.click(await screen.findByRole("button", { name: "Pin to the top" }));
		return screen.findByRole("alert");
	}

	it("renders the refusal above the first row, not after the last", async () => {
		sessionList = longList();
		const alert = await refuseAPinOn("Row 0");
		expect(alert.textContent).toContain("Could not save the pin: no saved messages yet");

		/* STRUCTURAL, not pixel: the band PRECEDES both the first and the last row
		   in document order, so it sits at the top of the list surface instead of
		   being appended after the list. The last-row comparison is the one that
		   fails on the old placement; the first-row one is what makes "top" mean
		   top rather than "somewhere above the end". */
		const first = cardByName("Row 0");
		const last = cardByName(`Row ${ROWS - 1}`);
		expect(first.compareDocumentPosition(alert) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
		expect(last.compareDocumentPosition(alert) & Node.DOCUMENT_POSITION_PRECEDING).toBeTruthy();
		/* And it is inside the SAME scroll container as the rows, so it is on screen
		   with the row it is about rather than in some other pane. */
		expect(alert.closest("main")).toBe(first.closest("main"));
	});

	it("keeps the band collapsed and empty until a refusal arrives", () => {
		sessionList = longList();
		render(<SessionListScreen />);
		expect(errorBox().className).toContain("grid-rows-[0fr]");
		expect(screen.getByRole("alert").textContent).toBe("");
	});

	it("reads the refusal out while the caption beside it is still expanded", async () => {
		/* The band's predicate is its own, NOT the caption's: with nothing pinned
		   the caption is expanded (its own condition), and a refusal can arrive on
		   exactly that list. A shared predicate would have to collapse one of the
		   two here — and the band is the one that must never be hidden while its
		   text is on screen (D12's a11y half). */
		sessionList = longList();
		const alert = await refuseAPinOn("Row 0");
		expect(alert.getAttribute("aria-hidden")).toBeNull();
		expect(errorBox().className).toContain("grid-rows-[1fr]");
		const caption = screen.getByText("touch and hold a row to pin it");
		const captionBox = caption.parentElement?.parentElement as HTMLElement;
		expect(captionBox.className).toContain("grid-rows-[1fr]");
	});
});
