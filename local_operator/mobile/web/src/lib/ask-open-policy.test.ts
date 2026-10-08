// The phone's open-by-default policy, as a pure decision — no screen, no DOM.
//
// The six clauses of the shared open-policy contract are numbered in
// `ask-open-policy.ts` and are the SAME six as the TUI's
// (`tests/unit/tui/test_ask_open_policy.py`), so the cells below are the TUI's cells
// restated over the phone's facts. What the phone does NOT have (an "arrived"
// reading, a settling switch, `asks_truncated`) is asserted ABSENT where its absence
// is a decision someone could undo by porting the TUI literally.
//
// Every cell is written to fail on a tree that lacks the behaviour for a behavioural
// reason; the mutation table in the PR names the mutant each one kills.
import { describe, expect, it } from "vitest";
import {
	AskOpenPolicy,
	OPEN_WINDOW_MS,
	namesEveryOutstanding,
	readQueue,
	type OpenDecision,
	type QueueReading,
} from "./ask-open-policy";

const T0 = 1_000_000;

function policy(): AskOpenPolicy {
	return new AskOpenPolicy();
}

function decide(
	p: AskOpenPolicy,
	conversation: string,
	reading: QueueReading,
	options: {
		at?: number;
		occupied?: boolean;
		surfaceOpen?: boolean;
		ids?: readonly string[] | null;
	} = {},
): OpenDecision {
	return p.decide(conversation, reading, {
		nowMs: options.at ?? T0 + 1,
		occupied: options.occupied ?? false,
		surfaceOpen: options.surfaceOpen ?? false,
		// Default: the frame names every outstanding ask and they are all new ones.
		outstandingIds: options.ids === undefined ? ["x1"] : options.ids,
	});
}

const row = (ask_id: string, status = "open") => ({ ask_id, status });

describe("readQueue — what ONE frame says about the queue", () => {
	it("calls a named answerable ask PENDING (clause 2)", () => {
		expect(readQueue([row("a1")], 1)).toBe("pending");
		// A timed-out ask is still answerable, so it is still pending.
		expect(readQueue([row("a1", "timed_out")], 1)).toBe("pending");
	});

	it("calls an explicit zero tally EMPTY, and only an explicit one (clause 1)", () => {
		expect(readQueue(undefined, 0)).toBe("empty");
		expect(readQueue([], 0)).toBe("empty");
	});

	it("calls a frame that cannot say UNRESOLVED, never empty and never pending (clause 5)", () => {
		// An old daemon: neither field. A not-yet-loaded projection: same.
		expect(readQueue(undefined, undefined)).toBe("unresolved");
		// The tally rode but the rows were dropped to fit the frame: the asks exist and
		// nothing can be drawn, so opening on the tally alone would mount an empty sheet.
		expect(readQueue(undefined, 3)).toBe("unresolved");
		expect(readQueue([], 2)).toBe("unresolved");
	});

	it("calls rows that are all settled SETTLED (clause 3)", () => {
		expect(readQueue([row("a1", "answered"), row("a2", "declined")], 0)).toBe("settled");
		// A runtime that publishes no tally cannot contradict its own rows.
		expect(readQueue([row("a1", "answered")], undefined)).toBe("settled");
	});

	it("does not call settled rows SETTLED beside a positive tally", () => {
		// The wire's drop order keeps NEWER settled rows over an older timed-out one, so a
		// frame can carry three answered rows and `asks_open: 1`: the pending ask is the one
		// the bound dropped. Spending the decision on "settled" would hide it for good.
		expect(
			readQueue([row("s1", "answered"), row("s2", "answered"), row("s3", "declined")], 1),
		).toBe("unresolved");
	});

	it("treats a row with no status as open (the wire's default)", () => {
		expect(readQueue([{ ask_id: "a1" }], undefined)).toBe("pending");
	});
});

describe("namesEveryOutstanding — what a dismissal is judged on", () => {
	it("is true exactly when the rows cover the tally", () => {
		expect(namesEveryOutstanding(2, 2)).toBe(true);
		expect(namesEveryOutstanding(0, 0)).toBe(true);
		expect(namesEveryOutstanding(3, 2)).toBe(false);
	});

	it("fails closed when the runtime publishes no tally", () => {
		expect(namesEveryOutstanding(undefined, 5)).toBe(false);
		expect(namesEveryOutstanding(null, 0)).toBe(false);
	});
});

describe("clause 2 — pending on open opens ONCE", () => {
	it("opens on the first resolved frame, then never again for the view", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending")).toBe("open");
		expect(p.awaiting("conv-a")).toBe(false);
		// A later frame — a re-render, a queue refresh — is a no-op for a decided view.
		expect(decide(p, "conv-a", "pending", { at: T0 + 5 })).toBe("skip");
	});

	it("waits through unresolved frames without spending the decision", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "unresolved", { at: T0 + 1 })).toBe("wait");
		expect(decide(p, "conv-a", "unresolved", { at: T0 + 2 })).toBe("wait");
		expect(p.awaiting("conv-a")).toBe(true);
		expect(decide(p, "conv-a", "pending", { at: T0 + 3 })).toBe("open");
	});

	it("opens again for a NEW view of the same conversation (switching away and back)", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending")).toBe("open");
		p.beginView("conv-b", T0 + 10);
		p.beginView("conv-a", T0 + 20);
		expect(p.awaiting("conv-a")).toBe(true);
		expect(decide(p, "conv-a", "pending", { at: T0 + 21 })).toBe("open");
	});

	it("opens again when the SAME conversation is entered twice with no other view in between", () => {
		/* The real flow: conversation -> session list -> conversation. The list screen never
		   begins a view, so the policy's current view is still this conversation when it is
		   entered again, and a policy that treated "same id as the current view" as "same
		   view" would never open it a second time. */
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending")).toBe("open");
		p.beginView("conv-a", T0 + 60_000);
		expect(p.awaiting("conv-a")).toBe(true);
		expect(decide(p, "conv-a", "pending", { at: T0 + 60_001 })).toBe("open");
	});

	it("is never armed for a view with no conversation id", () => {
		const p = policy();
		p.beginView("", T0);
		expect(p.awaiting("")).toBe(false);
		expect(decide(p, "", "pending")).toBe("skip");
	});

	it("ignores a decision for a conversation that is no longer the view", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		p.beginView("conv-b", T0 + 1);
		expect(decide(p, "conv-a", "pending", { at: T0 + 2 })).toBe("skip");
		expect(p.awaiting("conv-b")).toBe(true);
	});
});

describe("clauses 1 and 3 — nothing to show, or everything already addressed", () => {
	it("stays closed on an empty queue and spends the decision", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "empty")).toBe("skip");
		expect(p.awaiting("conv-a")).toBe(false);
	});

	it("stays closed on settled rows, and an ask arriving later does not reopen the view", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "settled")).toBe("skip");
		// The new ask is announced by the dock; it does not force the sheet open.
		expect(decide(p, "conv-a", "pending", { at: T0 + 30 })).toBe("skip");
	});
});

describe("clause 4 — a deliberate close is respected, keyed by ask id", () => {
	it("holds across a switch away and back while the waved-off ask is still outstanding", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: ["a1", "a2"] })).toBe("open");
		p.noteUserClosed("conv-a", ["a1", "a2"]);
		p.beginView("conv-b", T0 + 10);
		p.beginView("conv-a", T0 + 20);
		expect(decide(p, "conv-a", "pending", { at: T0 + 21, ids: ["a1", "a2"] })).toBe("skip");
	});

	it("holds while ANY waved-off ask is still outstanding (a partly resolved queue)", () => {
		const p = policy();
		p.noteUserClosed("conv-a", ["a1", "a2"]);
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: ["a2"] })).toBe("skip");
	});

	it("forgets the dismissal once NONE of the waved-off asks is outstanding, and opens the new batch", () => {
		const p = policy();
		p.noteUserClosed("conv-a", ["a1", "a2"]);
		// While the user was away both were answered and the agent asked two more.
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: ["b1", "b2"] })).toBe("open");
		expect(p.isDismissed("conv-a")).toBe(false);
	});

	it("tells an emptied-and-refilled queue from an unchanged one with no observation in between", () => {
		// The case that cannot be observed: while the user is away no frame for the
		// conversation is ever seen, so "I saw it empty" can never be witnessed. The
		// ids make the rule a function of the first frame on return.
		const unchanged = policy();
		unchanged.noteUserClosed("conv-a", ["a1"]);
		unchanged.beginView("conv-a", T0);
		expect(decide(unchanged, "conv-a", "pending", { ids: ["a1"] })).toBe("skip");

		const refilled = policy();
		refilled.noteUserClosed("conv-a", ["a1"]);
		refilled.beginView("conv-a", T0);
		expect(decide(refilled, "conv-a", "pending", { ids: ["a9"] })).toBe("open");
	});

	it("a NEW ask beside a still-outstanding waved-off one does not force the sheet open", () => {
		const p = policy();
		p.noteUserClosed("conv-a", ["a1"]);
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: ["a1", "a3"] })).toBe("skip");
	});

	it("HOLDS when the frame cannot name every outstanding ask (fail closed)", () => {
		// A dropped row may be exactly the one still outstanding; releasing on that guess
		// would open a sheet the user refused.
		const p = policy();
		p.noteUserClosed("conv-a", ["a1"]);
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: null })).toBe("skip");
		expect(p.isDismissed("conv-a")).toBe(true);
	});

	it("is per conversation", () => {
		const p = policy();
		p.noteUserClosed("conv-a", ["a1"]);
		p.beginView("conv-b", T0);
		expect(decide(p, "conv-b", "pending", { ids: ["b1"] })).toBe("open");
	});

	it("a second close ADDS to the record rather than replacing it", () => {
		const p = policy();
		p.noteUserClosed("conv-a", ["a1"]);
		p.noteUserClosed("conv-a", ["a2"]);
		p.beginView("conv-a", T0);
		// a1 is the older refusal and is still outstanding: a replacing record would
		// have forgotten it and opened.
		expect(decide(p, "conv-a", "pending", { ids: ["a1"] })).toBe("skip");
	});

	it("records nothing for a close with nothing pending (a glance at history)", () => {
		const p = policy();
		p.noteUserClosed("conv-a", []);
		expect(p.isDismissed("conv-a")).toBe(false);
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { ids: ["a1"] })).toBe("open");
	});

	it("closing settles a view that was still waiting, so a late frame cannot reopen it", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "unresolved")).toBe("wait");
		p.noteUserClosed("conv-a", ["a1"]);
		expect(p.awaiting("conv-a")).toBe(false);
	});

	it("is not cleared when the user later opens the sheet by hand", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		p.noteUserClosed("conv-a", ["a1"]);
		p.noteUserOpened("conv-a");
		expect(p.isDismissed("conv-a")).toBe(true);
	});
});

describe("clause 5 — never steal the keyboard, never open on a guess", () => {
	it("yields to an occupied screen, and for GOOD (the view does not come back for it)", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { occupied: true })).toBe("skip");
		// The user stops typing a minute later: a sheet that appeared then would be the
		// interruption in slow motion.
		expect(decide(p, "conv-a", "pending", { at: T0 + 30, occupied: false })).toBe("skip");
	});

	it("does not open past the window", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { at: T0 + OPEN_WINDOW_MS + 1 })).toBe("skip");
		expect(p.awaiting("conv-a")).toBe(false);
	});

	it("still opens at the edge of the window", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { at: T0 + OPEN_WINDOW_MS })).toBe("open");
	});

	it("is inert when disabled", () => {
		const p = new AskOpenPolicy({ enabled: false });
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending")).toBe("skip");
	});

	it("reset forgets every dismissal and the view", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		p.noteUserClosed("conv-a", ["a1"]);
		p.reset();
		expect(p.isDismissed("conv-a")).toBe(false);
		expect(p.awaiting("conv-a")).toBe(false);
	});
});

describe("clause 6 — auto-open is not the door", () => {
	it("leaves a sheet the user already opened alone", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "pending", { surfaceOpen: true })).toBe("skip");
		expect(p.awaiting("conv-a")).toBe(false);
	});

	it("pressing the door settles a view still waiting on a frame", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		expect(decide(p, "conv-a", "unresolved")).toBe("wait");
		p.noteUserOpened("conv-a");
		expect(decide(p, "conv-a", "pending", { at: T0 + 2 })).toBe("skip");
	});

	it("pressing the door for another conversation settles nothing", () => {
		const p = policy();
		p.beginView("conv-a", T0);
		p.noteUserOpened("conv-b");
		expect(p.awaiting("conv-a")).toBe(true);
	});
});
