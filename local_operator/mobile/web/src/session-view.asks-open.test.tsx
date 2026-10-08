// @vitest-environment happy-dom
//
// THE PHONE'S ASKS SHEET OPENS BY ITSELF — the relay half of the shared open-policy
// contract (six clauses, numbered in `lib/ask-open-policy.ts`; the TUI's cells are in
// `tests/unit/tui/test_ask_open_default.py`).
//
// The render site is the REAL `SessionScreen`, with the REAL `Composer`, `AskDock` and
// `AsksSheet`; only the network (`./api`) and the two streams are stood in for, exactly
// as `session-view.asks.test.tsx` does. That matters because the failures this feature
// can have are all in the WIRING — which frame the screen decides on, what it reads as
// "the user is typing", which gestures it records as a refusal — and a test of the pure
// policy alone passes while the screen never calls it.
//
// The four states the operator named, each with a cell that fails on the tree before
// this change (the dock is there, the sheet is not):
//
//   1. no asks on open                  -> closed
//   2. pending asks on open             -> open
//   3. all asks already addressed       -> closed, and never reopens
//   4. the user closed it while pending -> stays closed, across a re-render, a queue
//                                          refresh, a new ask, and away-and-back
import { StrictMode } from "react";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { OPEN_WINDOW_MS, resetAskOpenPolicy } from "./lib/ask-open-policy";
import { SessionScreen } from "./screens/session-view";
import type { PendingAsk, PendingRequest, SessionProjection } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
	getAsks: vi.fn(async () => ({ asks: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "accepted" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
	};
});

const { getAsks } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "ask-1",
		created_at: 1,
		expires_at: Date.now() + 900_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{
				id: "q1",
				question: "which rollout?",
				options: [{ label: "yes", description: "" }],
				multi: false,
				secret: false,
				persist: false,
			},
		],
		...patch,
	};
}

function approval(): PendingRequest {
	return {
		request_id: "req-1",
		kind: "approval",
		title: "run: make",
		detail: "",
		options: [],
		secret: false,
		question_index: 0,
		question_total: 1,
	};
}

function projection(patch: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Asks",
		cwd: "",
		model_label: "model-x",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...patch,
	} satisfies SessionProjection;
}

/** The wire's two fields for a set of rows, in the shapes the relay publishes: the rows
 *  and the tally of the ones still outstanding (the #864 lesson — absence is the
 *  capability proxy, and only an explicit zero is an answer). */
function frame(rows: PendingAsk[] | null, extra: Partial<SessionProjection> = {}) {
	if (rows === null) return projection(extra);
	const outstanding = rows.filter((r) => r.status === "open" || r.status === "timed_out").length;
	return projection({
		asks: rows.length > 0 ? rows : undefined,
		asks_open: outstanding,
		...extra,
	});
}

/** Install a frame and re-render the SAME screen, the way a new SSE snapshot does. */
function publish(view: ReturnType<typeof render>, next: SessionProjection, sessionId = "s1") {
	slot = { projection: next, connected: true };
	view.rerender(<SessionScreen sessionId={sessionId} />);
}

function mount(next: SessionProjection | null, sessionId = "s1") {
	slot = { projection: next, connected: true };
	return render(<SessionScreen sessionId={sessionId} />);
}

const sheetOpen = () => screen.queryByRole("dialog") !== null;

/** Where the composer keeps this conversation's draft (`store.ts` DRAFT_PREFIX + id). */
const DRAFT_KEY = "lo-mobile-draft:s1";

beforeEach(() => {
	resetAskOpenPolicy();
	(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });
	/* REPLACE, not assign to `location.hash`: the happy-dom window is shared by every cell
	   in this file, and an earlier cell that opened the sheet leaves its `askSheet` flag on
	   the current entry (nothing pops it when the screen unmounts). The policy's own open
	   REUSES an entry that already carries the flag (see the reload cell below), so a leaked
	   flag would turn "claims one entry" into "claims none" and the failure would depend on
	   which cell ran before it. A replacement clears the state without growing the stack. */
	window.history.replaceState(null, "", "#/s/s1");
});

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	slot = { projection: null, connected: true };
});

describe("state 1 — no asks on open: closed", () => {
	it("stays closed on a live queue with nothing in it", async () => {
		const view = mount(frame([]));
		expect(sheetOpen()).toBe(false);
		expect(screen.queryByTestId("ask-dock")).toBeNull();
		/* The contrast arm: the SAME screen, reached the same way, over a queue with an ask
		   waiting, opens. Without it this cell also passes on a tree where nothing ever
		   opens, which would make "closed" an absence of the feature rather than its answer. */
		view.unmount();
		mount(frame([ask()]), "s2");
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("stays closed on a runtime that does not publish asks at all", () => {
		mount(frame(null));
		expect(sheetOpen()).toBe(false);
	});
});

describe("state 2 — pending asks on open: open, once", () => {
	it("opens the asks sheet when a conversation is opened with an ask waiting", async () => {
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		// The dock is still there behind it: it is the minimized form of the same thing.
		expect(screen.getByTestId("ask-dock")).toBeTruthy();
	});

	it("opens on the first frame that RESOLVES the queue, not on the one before it", async () => {
		const view = mount(frame(null));
		expect(sheetOpen()).toBe(false);
		// A frame that carries only the tally (rows dropped to fit the frame) is not "pending
		// asks": there is nothing to draw, and the view must keep waiting.
		publish(view, projection({ asks_open: 2 }));
		expect(sheetOpen()).toBe(false);
		publish(view, frame([ask(), ask({ ask_id: "ask-2" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("waits while the link is down instead of deciding on a stale frame", async () => {
		slot = { projection: frame([ask()]), connected: false };
		const view = render(<SessionScreen sessionId="s1" />);
		expect(sheetOpen()).toBe(false);
		slot = { projection: frame([ask()]), connected: true };
		view.rerender(<SessionScreen sessionId="s1" />);
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("opens once per view: closing it and re-rendering does not bring it back", async () => {
		const view = mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		await waitFor(() => expect(sheetOpen()).toBe(false));
		publish(view, frame([ask()], { version: 2 }));
		expect(sheetOpen()).toBe(false);
	});

	it("claims ONE history entry, so a single Back closes it", async () => {
		const before = window.history.length;
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		expect(window.history.length).toBe(before + 1);
		expect(window.history.state?.askSheet).toBe(true);
	});

	it("reuses an entry that already carries the sheet's flag instead of stacking a second", async () => {
		/* Two real ways to arrive here: a RELOAD with the sheet open (the browser restores
		   `history.state`, flag included) and React's StrictMode re-running the mount
		   effects in the dev server. A second entry would make one Back close the sheet and
		   a second Back "close" it again, leaving the user one step from where they were. */
		window.history.replaceState({ askSheet: true }, "", "#/s/s1");
		const before = window.history.length;
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		expect(window.history.length).toBe(before);
		expect(window.history.state?.askSheet).toBe(true);
	});

	it("claims one entry under StrictMode too (the dev server re-runs mount effects)", async () => {
		/* `beginView` re-arms the view when the effect re-runs, so the decision is taken
		   twice for one mount and the SECOND "open" is only harmless because the entry is
		   claimed once. Production builds do not double-run effects, which is why this has
		   to be asserted here: nothing else would ever exercise it. */
		const before = window.history.length;
		slot = { projection: frame([ask()]), connected: true };
		render(
			<StrictMode>
				<SessionScreen sessionId="s1" />
			</StrictMode>,
		);
		await waitFor(() => expect(sheetOpen()).toBe(true));
		expect(window.history.length).toBe(before + 1);
	});
});

describe("state 3 — everything already addressed on open: closed, and never reopens", () => {
	it("stays closed over a queue of settled rows", async () => {
		const view = mount(
			frame([ask({ status: "answered" }), ask({ ask_id: "ask-2", status: "declined" })]),
		);
		expect(sheetOpen()).toBe(false);
		expect(screen.queryByTestId("ask-dock")).toBeNull();
		// Contrast arm (see state 1): another conversation with an ask waiting does open.
		view.unmount();
		mount(frame([ask()]), "s2");
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("does not open for an ask that arrives afterwards; the dock announces it", () => {
		const view = mount(frame([]));
		publish(view, frame([ask()]));
		expect(sheetOpen()).toBe(false);
		expect(screen.getByTestId("ask-dock")).toBeTruthy();
	});

	it("is not claimed over settled rows that sit beside a positive tally", async () => {
		// The wire's drop order keeps newer settled rows over an older timed-out one, so a
		// frame can name only settled rows while `asks_open` says one ask is outstanding.
		const view = mount(
			projection({ asks: [ask({ status: "answered" })], asks_open: 1 }),
		);
		expect(sheetOpen()).toBe(false);
		publish(view, frame([ask({ ask_id: "late-named" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});
});

describe("state 4 — the user closed it while asks were pending: it stays closed", () => {
	/* A BROWSER DELIVERS THE POP OF `history.back()` A TASK LATER; happy-dom delivers it
	   inside the call. That difference is not cosmetic here: `closeAsks` hands its entry back
	   with `back()`, the sheet's own popstate listener ALSO records a dismissal, and React has
	   detached that listener by the time a browser's pop arrives. With the synchronous pop the
	   listener is still attached, so a `closeAsks` that forgot to record the close itself would
	   still be saved by the pop it caused, and this block would pass over a broken ✕. Deferring
	   the pop restores the browser's order, so the close path has to carry its own record.
	   (The mutation run found exactly that: the "close is not recorded" mutant survived until
	   this was added.) */
	let restoreBack: (() => void) | null = null;
	beforeEach(() => {
		const realBack = window.history.back.bind(window.history);
		const spy = vi.spyOn(window.history, "back").mockImplementation(() => {
			setTimeout(realBack, 0);
		});
		restoreBack = () => spy.mockRestore();
	});
	afterEach(() => {
		restoreBack?.();
		restoreBack = null;
	});

	async function openThenCloseBy(how: "x" | "scrim" | "back") {
		const view = mount(frame([ask(), ask({ ask_id: "ask-2" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		if (how === "x") fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		else if (how === "scrim") fireEvent.click(screen.getByTestId("sheet-scrim"));
		else act(() => void window.dispatchEvent(new PopStateEvent("popstate")));
		await waitFor(() => expect(sheetOpen()).toBe(false));
		return view;
	}

	it.each(["x", "scrim", "back"] as const)(
		"holds across a re-render, a queue refresh, a new ask and away-and-back (closed by %s)",
		async (how) => {
			const view = await openThenCloseBy(how);
			// A re-render with no queue change.
			publish(view, frame([ask(), ask({ ask_id: "ask-2" })], { version: 2 }));
			expect(sheetOpen()).toBe(false);
			// A queue refresh: an ask changed state and is still answerable.
			publish(view, frame([ask({ status: "timed_out" }), ask({ ask_id: "ask-2" })]));
			expect(sheetOpen()).toBe(false);
			// A genuinely new ask: the dock announces it; the sheet does not reopen.
			publish(
				view,
				frame([ask({ status: "timed_out" }), ask({ ask_id: "ask-2" }), ask({ ask_id: "ask-3" })]),
			);
			expect(sheetOpen()).toBe(false);
			expect(screen.getByTestId("ask-dock")).toBeTruthy();
			// Away and back, within the same page lifetime: the screen REMOUNTS (app.tsx keys it
			// by session), so only the module-level record can carry the refusal across.
			view.unmount();
			mount(frame([ask(), ask({ ask_id: "ask-2" })]));
			await new Promise((resolve) => setTimeout(resolve, 30));
			expect(sheetOpen()).toBe(false);
		},
	);

	it("forgets the refusal once every ask it waved off is gone, and opens the next batch", async () => {
		const view = await openThenCloseBy("x");
		view.unmount();
		// While the user was away both asks were answered and the agent asked two new ones.
		mount(frame([ask({ ask_id: "fresh-1" }), ask({ ask_id: "fresh-2" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("holds while ANY waved-off ask is still outstanding", async () => {
		const view = await openThenCloseBy("x");
		view.unmount();
		// ask-1 was answered; ask-2 is still waiting; ask-9 is new.
		mount(frame([ask({ ask_id: "ask-2" }), ask({ ask_id: "ask-9" })]));
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
	});

	it("holds when the returning frame cannot name every outstanding ask (rows dropped to fit)", async () => {
		const view = await openThenCloseBy("x");
		view.unmount();
		/* ask-1 and ask-2 are still waiting, but the wire's size bound dropped their rows: the
		   frame names only a new ask-9 while its tally says two are outstanding. A frame that
		   cannot name them all proves nothing about the ones the user waved off (a dropped
		   row may be exactly one of them), so the refusal HOLDS rather than being forgotten
		   on a guess. */
		mount(frame([ask({ ask_id: "ask-9" })], { asks_open: 2 }));
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
		/* The contrast arm: the SAME rows with an honest tally (nothing was dropped) name every
		   outstanding ask, none of them waved off, so the refusal is released and the new batch
		   opens. Without it this cell also passes on a screen that never opens at all. */
		cleanup();
		mount(frame([ask({ ask_id: "ask-9" })], { asks_open: 1 }));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it.each([
		["another conversation", "#/s/elsewhere"],
		["this conversation's own agent route", "#/s/s1/a/job-1"],
	])(
		"is not recorded when the page is navigated to %s from behind the open sheet",
		async (_label, target) => {
			/* A fragment navigation (a notification tap, a deep link) fires `popstate` as well as
			   `hashchange`, so the sheet's own pop listener sees an external route change exactly
			   as it sees the phone's Back gesture. The two differ in where they LAND: Back lands
			   on this conversation's own root route (the sheet's entry carries no URL change), a
			   navigation lands on another route. Only the first is the user turning the sheet
			   away; recording the second would switch the policy off for a conversation whose
			   sheet nobody closed. */
			const view = mount(frame([ask(), ask({ ask_id: "ask-2" })]));
			await waitFor(() => expect(sheetOpen()).toBe(true));
			window.history.replaceState(null, "", target);
			act(() => void window.dispatchEvent(new PopStateEvent("popstate")));
			await waitFor(() => expect(sheetOpen()).toBe(false));
			view.unmount();
			window.history.replaceState(null, "", "#/s/s1");
			mount(frame([ask(), ask({ ask_id: "ask-2" })]));
			await waitFor(() => expect(sheetOpen()).toBe(true));
		},
	);

	it("is per conversation", async () => {
		const view = await openThenCloseBy("x");
		view.unmount();
		mount(frame([ask({ ask_id: "other-1" })], { session_id: "s2" }), "s2");
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("is not recorded when the user LEAVES for another conversation from the sheet", async () => {
		// Navigating to a foreign row's conversation is not waving this one's asks off, and
		// the router dispatches a synthetic popstate while the sheet's listener is still up.
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "other-1", session_id: "other" })],
		});
		const view = mount(frame([ask(), ask({ ask_id: "ask-2" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		fireEvent.click(await screen.findByRole("button", { name: "open" }));
		await waitFor(() => expect(window.location.hash).toBe("#/s/other"));
		view.unmount();
		window.location.hash = "#/s/s1";
		mount(frame([ask(), ask({ ask_id: "ask-2" })]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});
});

describe("clause 5 — never steal the keyboard, never trap", () => {
	it("does not open over a composer the user is typing into", async () => {
		const view = mount(frame([]));
		const field = screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
		field.focus();
		expect(document.activeElement).toBe(field);
		publish(view, frame([ask()]));
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
		expect(document.activeElement).toBe(field);
	});

	it("does not open over a focused composer, and does not come back when it blurs", async () => {
		// Cold open with the composer holding focus (the one-shot a new conversation uses).
		const view = mount(frame(null));
		const field = screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
		field.focus();
		publish(view, frame([ask()]));
		expect(sheetOpen()).toBe(false);
		field.blur();
		publish(view, frame([ask()], { version: 3 }));
		expect(sheetOpen()).toBe(false);
	});

	it("opens over an UNFOCUSED composer even when it holds a stored draft", async () => {
		// A draft restored from a previous visit is not being typed into; vetoing on it would
		// hide the sheet from every returning user who left a half-written message.
		localStorage.setItem(DRAFT_KEY, "half a sentence");
		mount(frame([ask()]));
		/* THE DRAFT HAS TO BE REAL BEFORE THE CELL MEANS ANYTHING. An earlier version of this
		   cell wrote the draft under a key the composer never reads, so no code path could have
		   seen it and the cell passed over a policy that DID veto on drafts (found by deleting
		   the cell's subject in the mutation run). The composer showing the text proves the
		   key is the one it reads. */
		const field = screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
		expect(field.value).toBe("half a sentence");
		expect(document.activeElement).not.toBe(field);
		await waitFor(() => expect(sheetOpen()).toBe(true));
		// ...and nothing the user wrote was touched, by the open or by the sheet's own focus.
		expect(localStorage.getItem(DRAFT_KEY)).toBe("half a sentence");
	});

	it("does not open over a blocking approval card", async () => {
		mount(frame([ask()], { pending: approval(), pending_count: 1 }));
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
		expect(screen.getByTestId("pending-card")).toBeTruthy();
	});

	it("does not open over another sheet that is already up", async () => {
		/* The first frame must leave the view UNDECIDED (no rows, no tally: the runtime has not
		   said yet). An empty queue would spend the view's one decision as "closed", and the
		   veto would then never be consulted: the mutation run showed this cell passing with
		   the veto deleted when it started from `frame([])`. */
		const view = mount(frame(null));
		fireEvent.click(screen.getByText("model-x"));
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
		publish(view, frame([ask()]));
		await new Promise((resolve) => setTimeout(resolve, 30));
		/* Counted from the DOM, not the accessibility tree: an open sheet hides its siblings
		   from the tree, so a second sheet stacked on top would leave ONE accessible dialog. */
		expect(document.querySelectorAll('[role="dialog"]')).toHaveLength(1);
		expect(screen.queryByText(/^asks/)).toBeNull();
		// ...and the veto is FOR GOOD: closing the model sheet does not bring the asks sheet up.
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		await waitFor(() => expect(sheetOpen()).toBe(false));
		publish(view, frame([ask()], { version: 4 }));
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
	});

	it("can always be closed, by the control the sheet already had", async () => {
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		const close = screen.getByRole("button", { name: "close sheet" });
		fireEvent.click(close);
		await waitFor(() => expect(sheetOpen()).toBe(false));
	});

	it("never opens on the agent route, which has no asks sheet", async () => {
		// The contrast arm: the same frame on the conversation root opens.
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		cleanup();
		resetAskOpenPolicy();
		window.history.replaceState(null, "", "#/s/s1/a/job-1");
		const entries = window.history.length;
		slot = { projection: frame([ask()]), connected: true };
		render(<SessionScreen sessionId="s1" jobId="job-1" />);
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(sheetOpen()).toBe(false);
		/* No sheet is rendered on this route, so `sheetOpen()` alone would stay false even if
		   the policy DID fire here. What a stray open leaves behind is the history entry it
		   claimed, and a later Back that pops a route the reader is still standing on. */
		expect(window.history.length).toBe(entries);
		expect(window.history.state?.askSheet).not.toBe(true);
	});
});

describe("clause 5 — the agent route and the window", () => {
	it("a view armed on the root is not decided from the agent route", async () => {
		/* The root mounts on a frame that cannot say yet (view armed, undecided); the reader
		   taps into a subagent; the runtime then starts publishing asks. The agent route has no
		   asks sheet, so a policy that decided here would claim a history entry and set an open
		   flag nothing renders, and a later Back would pop a route the reader stands on.
		   (A fresh mount of the agent route, as the cell above does, cannot see this: nothing
		   has armed a view yet, so the guard is never what stops it.) */
		const root = mount(frame(null));
		expect(sheetOpen()).toBe(false);
		root.unmount();
		window.history.replaceState(null, "", "#/s/s1/a/job-1");
		const entries = window.history.length;
		slot = { projection: frame([ask()]), connected: true };
		render(<SessionScreen sessionId="s1" jobId="job-1" />);
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(window.history.length).toBe(entries);
		expect(window.history.state?.askSheet).not.toBe(true);
		// Coming back to the root is a NEW view, and it opens as ever.
		cleanup();
		window.history.replaceState(null, "", "#/s/s1");
		mount(frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});

	it("a frame that resolves only after the window is not 'on open'", async () => {
		const view = mount(frame(null));
		vi.useFakeTimers({ toFake: ["Date"] });
		try {
			vi.setSystemTime(Date.now() + OPEN_WINDOW_MS + 1_000);
			publish(view, frame([ask()]));
			await new Promise((resolve) => setTimeout(resolve, 30));
			expect(sheetOpen()).toBe(false);
		} finally {
			vi.useRealTimers();
		}
	});

	it("does not take a focused BUTTON for typing", async () => {
		// Only a field the keyboard types into vetoes; a control that merely holds focus does not.
		const view = mount(frame(null));
		const chip = screen.getByText("model-x").closest("button") as HTMLButtonElement;
		chip.focus();
		expect(document.activeElement).toBe(chip);
		publish(view, frame([ask()]));
		await waitFor(() => expect(sheetOpen()).toBe(true));
	});
});

describe("clause 6 — auto-open is not the door", () => {
	it("a door pressed while the link is down settles the view: the returning frame opens no second sheet", async () => {
		/* The one way the door can be pressed BEFORE the policy has decided: the screen holds a
		   retained frame (rows, so the dock draws) while the link is down, and the policy is
		   waiting for the link rather than deciding on a stale frame. */
		slot = { projection: frame([ask()]), connected: false };
		const view = render(<SessionScreen sessionId="s1" />);
		expect(sheetOpen()).toBe(false);
		fireEvent.click(screen.getByTestId("ask-dock"));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		const entries = window.history.length;
		slot = { projection: frame([ask(), ask({ ask_id: "ask-2" })]), connected: true };
		view.rerender(<SessionScreen sessionId="s1" />);
		await new Promise((resolve) => setTimeout(resolve, 30));
		expect(window.history.length).toBe(entries);
		expect(screen.getAllByRole("dialog")).toHaveLength(1);
		// Closing the door's sheet by hand is a refusal like any other: nothing reopens it.
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		await waitFor(() => expect(sheetOpen()).toBe(false));
		publish(view, frame([ask(), ask({ ask_id: "ask-2" })], { version: 5 }));
		expect(sheetOpen()).toBe(false);
	});

	it("after the view decided to stay closed, the dock still opens the sheet and a later frame stacks nothing on it", async () => {
		// An empty queue spends the view's decision (clause 1); an ask that arrives afterwards
		// is announced by the dock (clause 3), and pressing the dock is the ordinary door.
		const view = mount(frame([]));
		publish(view, frame([ask()]));
		expect(sheetOpen()).toBe(false);
		fireEvent.click(screen.getByTestId("ask-dock"));
		await waitFor(() => expect(sheetOpen()).toBe(true));
		const entries = window.history.length;
		publish(view, frame([ask(), ask({ ask_id: "ask-2" })]));
		expect(window.history.length).toBe(entries);
		expect(screen.getAllByRole("dialog")).toHaveLength(1);
	});
});
