// @vitest-environment happy-dom
//
// The asks sheet — the EXPANDED surface (design §5.3). It reads the aggregate
// from `GET /api/asks` (index-backed, so it answers for a conversation whose
// runtime is gone), shows only the rows of the conversation it was opened from,
// and re-reads when the outstanding population moves rather than waiting for a
// manual refresh.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AsksSheet } from "./components/asks-sheet";
import type { PendingAsk, SessionProjection, SessionSummary } from "./types";

vi.mock("./api", () => ({
	getAsks: vi.fn(async () => ({ asks: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })),
}));

let rows: SessionSummary[] = [];
let revision = 0;
/* The opening conversation's own projection slot (its `asks` rows), as the
   session screen's stream would deliver it. */
let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
/* The list frame's revision is the REAL signature rule, folded here the way the
   store folds it: the counter moves only when the signature of `rows` does. A
   test can then say "asks_open stayed 0" and mean it, instead of choosing the
   revision by hand. A test that wants to force a bump still assigns `revision`. */
let lastSignature = "";
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: rows, connected: true, revision: 0 }),
		useProjection: () => slot,
		useAsksRevision: () => {
			const next = actual.asksPopulationSignature(rows);
			if (next !== lastSignature) {
				lastSignature = next;
				revision += 1;
			}
			return revision;
		},
	};
});

const { getAsks } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a1",
		session_id: "mine",
		cwd: "/tmp/w",
		created_at: 1,
		expires_at: Date.now() + 600_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{ id: "q1", question: "ship the fix?", options: [], multi: false, secret: false, persist: false },
		],
		...patch,
	};
}

function summary(patch: Partial<SessionSummary> = {}): SessionSummary {
	return {
		session_id: "mine",
		section: "active",
		conversation_name: "woken-by-monitor",
		cwd: "/tmp/w",
		model_label: "",
		streaming: false,
		needs_attention: false,
		pending_kind: "",
		subagents_running: 0,
		todos_open: 0,
		mtime: 0,
		...patch,
	};
}

beforeEach(() => {
	rows = [];
	revision = 0;
	lastSignature = "";
	slot = { projection: null, connected: true };
	(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

/* THE DRAWER'S QUESTIONS DESTINATION IS MACHINE-WIDE (UX review round 1, U1).
   It opened this sheet with `currentSessionId=""`, which the per-conversation
   filter could not match against anything — so the success path painted
   "nothing waiting" over three outstanding asks. These two cases pin the mode
   and the empty state that must still exist when the queue is genuinely empty. */
describe("the drawer's Questions destination", () => {
	it("lists outstanding asks from every conversation, each named", async () => {
		rows = [
			summary({ session_id: "mine" }),
			summary({ session_id: "other", conversation_name: "tax receipts" }),
		];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-other", session_id: "other" })],
		});

		render(<AsksSheet open onClose={() => {}} currentSessionId="" />);

		await waitFor(() => expect(screen.getByText("ship the fix?")).toBeTruthy());
		// WHICH conversation is asking, so an answer is never attributed wrongly.
		expect(screen.getByText("tax receipts")).toBeTruthy();
		// The success path must not also claim the queue is empty.
		expect(screen.queryByTestId("ask-sheet-empty")).toBeNull();
	});

	it("still states an empty queue when nothing is waiting anywhere", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });

		render(<AsksSheet open onClose={() => {}} currentSessionId="" />);

		await waitFor(() => expect(screen.getByTestId("ask-sheet-empty")).toBeTruthy());
	});
});

describe("AsksSheet", () => {
	it("renders nothing until it is opened, and reads the aggregate when it is", async () => {
		const { container } = render(<AsksSheet open={false} onClose={() => {}} currentSessionId="mine" />);
		expect(container.firstChild).toBeNull();
		expect(getAsks).not.toHaveBeenCalled();
	});

	it("lists and counts only the opening conversation's asks", async () => {
		/* Ben's report (2026-10-15): the sheet opened from one conversation listed
		   every conversation's asks. The aggregate still carries them all; the sheet
		   filters to its own conversation BEFORE it orders or counts, so the title
		   states the same set the list shows. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				ask({ ask_id: "mine-1" }),
				ask({
					ask_id: "other-1",
					session_id: "other",
					questions: [
						{ id: "q1", question: "a", options: [], multi: false, secret: false, persist: false },
						{ id: "q2", question: "b", options: [], multi: false, secret: false, persist: false },
					],
				}),
			],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("1 question waiting")).toBeTruthy());
		const cards = screen.getAllByTestId("ask-card");
		expect(cards.length).toBe(1);
		expect(cards[0].getAttribute("data-ask-id")).toBe("mine-1");
	});

	it("says nothing is waiting when only other conversations have asks", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "other-1", session_id: "other" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/Nothing waiting/)).toBeTruthy());
		expect(screen.queryByTestId("ask-card")).toBeNull();
	});

	it("answers against the opening conversation's session route", async () => {
		const { sendCommand } = await import("./api");
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		fireEvent.change(screen.getByPlaceholderText("Your answer"), {
			target: { value: "yes — dark behind the flag" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("mine", {
				op: "ask_respond",
				ask_id: "a1",
				answers: { q1: ["yes — dark behind the flag"] },
			}),
		);
	});

	it("says nothing is waiting when the aggregate is empty", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/Nothing waiting/)).toBeTruthy());
	});

	it("acknowledges a dismissal while another ask is still waiting", async () => {
		/* UX round 2, U9 = agent review round 2, M1(a): the receipt used to render
		   only inside the `rows.length === 0` branch, so dismissing one of two asks
		   — the ordinary queue — said nothing at all. It now sits above whatever
		   remains, where the card left. */
		const { sendCommand } = await import("./api");
		let current = [ask({ ask_id: "a1" }), ask({ ask_id: "a2" })];
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			current = current.filter((row) => row.ask_id !== "a1");
			return { ok: true, detail: "dismissed" };
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		/* ONE open card and the other ask as a one-line row. */
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() =>
			expect(screen.getByTestId("ask-receipt").textContent).toBe(
				"Dismissed — no longer needed",
			),
		);
		/* AND THE OTHER ASK IS STILL THERE, now the open card: the receipt is a line,
		   not a replacement for the list. */
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a2");
		expect(screen.queryByTestId("ask-sheet-empty")).toBeNull();
	});

	it("does not repeat a dismissal once its own row arrives", async () => {
		/* SA-D8, design D1 = UX U2. Before this PR the row LEFT the aggregate, so
		   the receipt was the only trace. Now a pre-deadline dismissal keeps its
		   row, and the card states the same catalogue sentence — so the sheet
		   said one fact twice, once unattributed above a different card and once
		   on the row it belonged to. The row is the statement; the receipt covers
		   only the gap until the read that carries it lands. */
		const { sendCommand } = await import("./api");
		let current = [ask({ ask_id: "a1" }), ask({ ask_id: "a2" })];
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			current = current.map((row) =>
				row.ask_id === "a1"
					? {
							...row,
							status: "dismissed",
							dismissed_at: Date.now(),
							dismissed_reason: "moot",
						}
					: row,
			);
			return { ok: true, detail: "dismissed" };
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		/* The settled row is back under "Settled", saying why … */
		await waitFor(() => expect(screen.getByText("Dismissed — no longer needed")).toBeTruthy());
		expect(screen.getByRole("heading", { name: "Settled" })).toBeTruthy();
		/* … and the receipt is gone, so it says it once. */
		expect(screen.queryByTestId("ask-receipt")).toBeNull();
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
	});

	it("acknowledges the tap while the read that carries the row is still in flight", async () => {
		/* Review round 2, M2. The read that follows a tap is a round trip; until
		   it lands the list still carries the ask as `open`. Keying the receipt on
		   the id's PRESENCE therefore hid the acknowledgement exactly during the
		   gap it exists to cover — a live-looking card after a successful
		   dismissal, with nothing saying the tap had landed. The test is the
		   ROW's status: the receipt stays until the read returns `dismissed`. */
		const { sendCommand } = await import("./api");
		let current = [ask({ ask_id: "a1" }), ask({ ask_id: "a2" })];
		let calls = 0;
		let release: (() => void) | null = null;
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			calls += 1;
			if (calls === 2) {
				/* The stale read: held open past the tap, carrying the OLD list. */
				await new Promise<void>((resolve) => {
					release = resolve;
				});
			}
			return { asks: current };
		});
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			/* The reply lands BEFORE the read does — the real order. */
			return { ok: true, detail: "dismissed" };
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));

		/* THE GAP: both asks still `open` in the stale list, and the receipt IS shown. */
		await waitFor(() => expect(screen.getByTestId("ask-receipt")).toBeTruthy());
		expect(screen.getByTestId("ask-receipt").textContent).toBe("Dismissed — no longer needed");
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		expect(screen.getAllByTestId("ask-row").length).toBe(1);

		/* The read lands: the row is `dismissed`, and the card takes over. */
		current = [
			ask({ ask_id: "a2" }),
			ask({ ask_id: "a1", status: "dismissed", dismissed_at: Date.now(), dismissed_reason: "moot" }),
		];
		await waitFor(() => expect(release).not.toBeNull(), { timeout: 2000 });
		release!();
		await waitFor(() => expect(screen.queryByTestId("ask-receipt")).toBeNull());
		expect(screen.getByText("Dismissed — no longer needed")).toBeTruthy();
	});

	it("keeps a lapsed dismissal's receipt while its id is still listed", async () => {
		/* The LAPSED door is the exception: a dismissal at or after the deadline
		   still removes the row (case 2 of §17), so no row ever arrives and the
		   receipt is that door's only acknowledgement. It therefore stays even if
		   a read still carries the id — the race this rule must not turn into a
		   silent dismissal. */
		const { sendCommand } = await import("./api");
		let current = [ask({ ask_id: "a1", status: "timed_out" })];
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => ({
			ok: true,
			detail: "dismissed",
		}));
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		/* The only ask is a lapsed one, so it is the open card and its one door
		   (Dismiss, no reason) is in the pinned bar. */
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		await waitFor(() => expect(screen.getByTestId("ask-receipt")).toBeTruthy());
		expect(screen.getByTestId("ask-receipt").textContent).toBe("Dismissed — no reply sent");
		/* The read still lists the id: the receipt must survive it. */
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		expect(screen.getByTestId("ask-receipt")).toBeTruthy();
	});

	it("retires the receipt when the operator's next act is an answer", async () => {
		/* M1(b): with the receipt kept only by "not empty", answering a second ask
		   left "Dismissed — no longer needed" over an answer — the receipt outliving
		   the act it described and naming the wrong one. A non-dismissal settlement
		   clears it. */
		const { sendCommand } = await import("./api");
		let current = [ask({ ask_id: "a1" }), ask({ ask_id: "a2" })];
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			current = current.filter((row) => row.ask_id !== "a1");
			return { ok: true, detail: "dismissed" };
		});
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() => expect(screen.getByTestId("ask-receipt")).toBeTruthy());

		/* The operator answers the remaining ask. */
		fireEvent.change(screen.getByPlaceholderText("Your answer"), {
			target: { value: "yes — dark behind the flag" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(screen.queryByTestId("ask-receipt")).toBeNull());
	});

	it("shows a receipt for the dismissal that emptied it, not the empty state", async () => {
		/* UX round 1, U2: the daemon answers `{"ok": true, "detail": "dismissed"}`,
		   the row leaves the aggregate, and "nothing waiting — questions the agent
		   asks will appear here" was the whole trace of an act the operator had
		   just completed. The receipt names the reason the agent was given. */
		const { sendCommand } = await import("./api");
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in the conversation/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("mine", {
				op: "ask_dismiss",
				ask_id: "a1",
				reason: "answered_in_chat",
			}),
		);
		await waitFor(() => expect(screen.getByText("Dismissed — answered in chat")).toBeTruthy());
		expect(screen.queryByText(/Nothing waiting/)).toBeNull();
	});

	it("re-reads when the outstanding population moves, without a manual refresh", async () => {
		const { rerender } = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(1));
		revision = 1;
		rerender(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(2));
	});

	it("re-reads when a lapsed ask is dismissed elsewhere, although asks_open stays 0", async () => {
		/* #2388: a timed-out ask no longer counts in `asks_open`, so a session
		   holding only a lapsed ask publishes 0 before AND after another surface
		   dismisses it (or answers it late) — the list frame's signature cannot move.
		   The sheet must still drop the gone row promptly, not on the 20 s backstop;
		   the signal is the opening conversation's own projection rows. */
		const lapsed = ask({ ask_id: "t", status: "timed_out" });
		const withAsks = (asks: PendingAsk[]) =>
			({ asks, asks_open: 0 }) as unknown as SessionProjection;
		rows = [summary({ asks_open: 0 })];
		slot = { projection: withAsks([lapsed]), connected: true };
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [lapsed] });
		const { rerender } = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("1 question timed out")).toBeTruthy());
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		expect(getAsks).toHaveBeenCalledTimes(1);

		/* Another surface dismisses it: the index no longer lists it, the session's
		   projection drops the row, and the list frame still says asks_open 0. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [] });
		rows = [summary({ asks_open: 0 })];
		slot = { projection: withAsks([]), connected: true };
		rerender(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(screen.queryAllByTestId("ask-card").length).toBe(0));
		expect(screen.getByText("Questions")).toBeTruthy();
		expect(screen.getByText(/Nothing waiting/)).toBeTruthy();
	});

	it("states the daemon's refusal reason when the read fails", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("the ask index could not be read"),
		);
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() =>
			expect(screen.getByText("the ask index could not be read")).toBeTruthy(),
		);
	});

	it("leads with the head ask — the order the minimized chip names", async () => {
		/* U8 (UX round 1): the chip named the oldest open ask while the sheet led
		   with the newest, so a tap landed on a question the reader had not just
		   read. The sheet now shares `orderedForDisplay` with the chip. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "a-new", created_at: 900 }), ask({ ask_id: "a-old", created_at: 1 })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		/* The head ask is the open card; the other is its one-line row. */
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a-old");
		expect(screen.getByTestId("ask-row").getAttribute("data-ask-id")).toBe("a-new");
	});

	it("offers a retry when the read fails, instead of closing being the only move", async () => {
		/* U6 (UX round 1): a hung or failed aggregate read left the sheet with no
		   way forward but closing it. The hung case is bounded in the component
		   (`AbortSignal.timeout`); the failed case gets this control. */
		(getAsks as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("could not reach the daemon"),
		);
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("could not reach the daemon")).toBeTruthy());
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask()] });
		fireEvent.click(screen.getByRole("button", { name: /Try again/ }));
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
	});

	it("warns before the tap when the owning conversation has ended", async () => {
		rows = [summary({ ended: true })];
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/This conversation has ended/)).toBeTruthy());
		/* THE COST IS ON THE LINE (design round 1, D4 = UX U1/U4): the relay now
		   reopens it itself, so "may need it reopened" described a manual remedy
		   that no longer exists — while the ~30 s the cold arm costs went unsaid. */
		expect(screen.getByText(/Answering will reopen it, which can take up to 30 s/)).toBeTruthy();
	});

	it("warns before the tap when the conversation is simply not running", async () => {
		/* UX round 1, U4: for a durable conversation this daemon never watched die,
		   the list reports `ended:false`, so the strip did not render at all —
		   exactly the operator's case. The aggregate's own `runtime_live` is what
		   makes the cue appear, and the copy says "not running" because "has
		   ended" would be false about it. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ runtime_live: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/This conversation is not running/)).toBeTruthy());
		expect(screen.getByText(/Answering will start it, which can take up to 30 s/)).toBeTruthy();
	});

	it("says nothing about a wait when the relay can deliver now", async () => {
		/* The other half of the rule: a live conversation answers in ~1 s, so a
		   ~30 s promise there would be its own lie. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask({ runtime_live: true })] });
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		expect(screen.queryByText(/30 s/)).toBeNull();
	});

	it("says nothing about a wait above a settled receipt", async () => {
		/* UX round 2, U6: `runtime_live === false` is true of a cold conversation
		   whatever the ask's status, so the strip rendered above a receipt that
		   offers nothing to answer — a wait promised for a tap that does not exist. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ status: "answered", delivered: true, runtime_live: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		/* A settled ask is one line under "Settled"; opening it expands its receipt. */
		await waitFor(() => expect(screen.getByTestId("ask-row")).toBeTruthy());
		fireEvent.click(screen.getByTestId("ask-row"));
		expect(screen.getByTestId("ask-card")).toBeTruthy();
		expect(screen.queryByText(/This conversation is not running/)).toBeNull();
	});

	it("stays silent on a DEAD row, and the card states it", async () => {
		/* Design round 2, D6: on this row `runtime_live` is false for the OTHER
		   reason — nothing can ever read the answer — so a strip promising a ~30 s
		   bring-up would contradict the card's own sentence one element below it. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ runtime_live: false, durable: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() =>
			expect(screen.getByText(/so this question can't be read/)).toBeTruthy(),
		);
		expect(screen.queryByText(/This conversation is not running/)).toBeNull();
	});

	it("keeps the affordances on a LIVE conversation with no transcript", async () => {
		/* Design round 3, D7: the strip and the card both key on
		   `isDeadConversation` — the CONJUNCTION — so a live row with no transcript
		   answers as normal. Gating on durability alone silenced the sheet one
		   element above a card that then refused a working answer. */
		rows = [];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ runtime_live: true, durable: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		expect(screen.queryByText(/This conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
	});

	it("does not count questions a dead row can never have answered", async () => {
		/* Agent review round 2, MINOR-3: a dead row's status is still `open`, so it
		   entered `outstandingAsks` and its questions were counted in the title — a
		   promise of answers the backend refuses, one element above the card that
		   says the ask can never be read. It is the sheet's own "no contradiction
		   one element apart" rule, applied to the title. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				ask({ ask_id: "live", runtime_live: true, durable: false }),
				ask({ ask_id: "gone", runtime_live: false, durable: false }),
			],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("1 question waiting")).toBeTruthy());
	});

	it("carries no count at all when every row is dead", async () => {
		/* The other half: nothing is waiting on the user, so the title states the
		   surface rather than a number nobody can act on. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "gone", runtime_live: false, durable: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("Questions")).toBeTruthy());
	});

	it("titles a lapsed-only queue by what it holds, and counts only offered questions as waiting", async () => {
		/* Design round 1, D1: "1 question waiting" above a card reading Lapsed
		   stated a live decision one element above the card that says there is
		   none. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "t", status: "timed_out" })],
		});
		const { unmount } = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("1 question timed out")).toBeTruthy());
		/* The only ask that can still be acted on is the lapsed one, so it is the
		   open card, read-only, with Dismiss in the pinned bar. */
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-status")).toBe("timed_out");
		expect(screen.queryByRole("radio")).toBeNull();
		unmount();

		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "o" }), ask({ ask_id: "t", status: "timed_out" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("1 question waiting")).toBeTruthy());
	});

	it("promises no bring-up above a lapsed ask, which offers nothing to answer", async () => {
		/* A lapsed (`timed_out`) card shows its question read-only, so
		   "answering will bring it up" would promise a tap the card one element
		   below does not have. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "t", status: "timed_out", runtime_live: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(1));
		expect(screen.getByText("Timed out — the agent goes ahead without your answer")).toBeTruthy();
		expect(screen.queryByText(/This conversation is not running/)).toBeNull();
	});

	it("keeps a timed-out ask as a one-line row beside an open one, which stays fully offered", async () => {
		/* D11, in the new frame: only the open ask is a card; the lapsed one is a
		   single line (its state sentence, the start of its question, a caret) and
		   opening it is an explicit tap. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "o" }), ask({ ask_id: "t", status: "timed_out" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		const row = screen.getByTestId("ask-row");
		expect(row.getAttribute("aria-expanded")).toBe("false");
		expect(row.textContent).toContain("Timed out");
		expect(row.textContent).toContain("ship the fix?");
		expect(screen.getByPlaceholderText("Your answer")).toBeTruthy();
		expect(screen.queryByRole("button", { name: /^Dismiss$/ })).toBeNull();
	});
});

/* THE PANEL'S SHAPE (ask-panel parity with the native app): one open card, the
   other waiting asks as one-line rows, settled asks under "Settled", and the
   controls that answer the open card PINNED outside the scrolling body. */
describe("the ask panel", () => {
	beforeEach(() => localStorage.clear());
	const choice = (id: string, question: string) =>
		ask({
			ask_id: id,
			questions: [
				{
					id: "q1",
					question,
					options: [
						{ label: "yes", description: "" },
						{ label: "no", description: "" },
					],
					multi: false,
					secret: false,
					persist: false,
				},
			],
		});
	const open = () => render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);

	it("pins send, decline and dismiss outside the scroller, with the question inside it", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [choice("a1", "ship the fix?")] });
		open();
		const card = await screen.findByTestId("ask-card");
		const bar = screen.getByTestId("asks-bar");
		const scroller = card.closest(".overflow-y-auto") as HTMLElement;
		expect(scroller).toBeTruthy();
		for (const name of [/Send answer/, /^Decline$/, /^Dismiss…$/]) {
			const control = screen.getByRole("button", { name });
			expect(bar.contains(control)).toBe(true);
			expect(scroller.contains(control)).toBe(false);
		}
		// The bar is a sibling AFTER the scroller inside the dialog's panel.
		expect(scroller.parentElement).toBe(bar.parentElement?.parentElement);
		expect(scroller.compareDocumentPosition(bar) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(scroller.contains(within(card).getByText("ship the fix?"))).toBe(true);
	});

	it("leaves no empty stripe when the open ask has nothing to answer with", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ status: "answered", delivered: true })],
		});
		open();
		await screen.findByTestId("ask-row");
		const bar = screen.getByTestId("asks-bar");
		expect(bar.childElementCount).toBe(0);
		// `empty:hidden` collapses it; the class is the contract happy-dom can see.
		expect(bar.className).toContain("empty:hidden");
	});

	it("opens another waiting ask from its row, and closes the current one", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [choice("a1", "first question?"), choice("a2", "second question?")],
		});
		open();
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a1");
		fireEvent.click(screen.getByTestId("ask-row"));
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a2");
		// The first is now the one-line row, its question started but not drawn in full.
		const row = screen.getByTestId("ask-row");
		expect(row.getAttribute("data-ask-id")).toBe("a1");
		expect(row.textContent).toContain("first question?");
		expect(screen.getAllByRole("button", { name: /Send answer/ })).toHaveLength(1);
	});

	it("keeps the open card where it is when a later read re-ranks the queue", async () => {
		/* The default open ask is LATCHED: a re-rank must not move the card from
		   under a half-typed answer. */
		const two = [choice("a1", "first?"), choice("a2", "second?")];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: two });
		const { rerender } = open();
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a1");
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [{ ...two[1], created_at: 0 }, two[0]],
		});
		revision = 1;
		rerender(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(getAsks).toHaveBeenCalledTimes(2));
		expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a1");
	});

	it("walks on to the next ask after an answer and moves focus to its question", async () => {
		const { sendCommand } = await import("./api");
		let current = [choice("a1", "first question?"), choice("a2", "second question?")];
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			current = [{ ...current[0], status: "answered", delivered: true, answers: { q1: ["yes"] } }, current[1]];
			return { ok: true, detail: "answered" };
		});
		open();
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(1));
		fireEvent.click(screen.getByRole("radio", { name: /^yes$/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() =>
			expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a2"),
		);
		await waitFor(() =>
			expect(document.activeElement).toBe(screen.getByTestId("ask-question-text")),
		);
		expect(screen.getByTestId("ask-question-text").textContent).toBe("second question?");
		// The answered one is now under "Settled", one line.
		await waitFor(() => expect(screen.getByRole("heading", { name: "Settled" })).toBeTruthy());
	});

	it("puts answered asks under Settled, expanding in place, never between options and buttons", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				ask({ ask_id: "done", status: "answered", delivered: true, answers: { q1: ["blue"] } }),
				choice("live", "which colour?"),
			],
		});
		open();
		const heading = await screen.findByRole("heading", { name: "Settled" });
		const rows = screen.getAllByTestId("ask-row");
		expect(rows).toHaveLength(1);
		expect(rows[0].getAttribute("data-ask-id")).toBe("done");
		expect(rows[0].textContent).toContain("Answered — the agent was told");
		// Below everything still waiting.
		const card = screen.getByTestId("ask-card");
		expect(card.compareDocumentPosition(heading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		// Expanding it is the receipt, in place under its own line …
		fireEvent.click(rows[0]);
		expect(screen.getByText("blue")).toBeTruthy();
		expect(screen.getAllByText("Answered — the agent was told")).toHaveLength(1);
		// … and it never sits between the open card's options and the pinned buttons.
		const bar = screen.getByTestId("asks-bar");
		expect(bar.querySelector("dl")).toBeNull();
	});

	it("says why Send is grey, and stops saying it when the form is complete", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				ask({
					questions: [
						{ id: "q1", question: "one?", options: [{ label: "yes", description: "" }], multi: false, secret: false, persist: false },
						{ id: "q2", question: "two?", options: [{ label: "yes", description: "" }], multi: false, secret: false, persist: false },
					],
				}),
			],
		});
		open();
		const bar = await screen.findByTestId("asks-bar");
		await waitFor(() => expect(bar.textContent).toContain("0 of 2 questions answered or skipped"));
		fireEvent.click(screen.getAllByRole("radio", { name: /^yes$/ })[0]);
		expect(bar.textContent).toContain("1 of 2 questions answered or skipped");
		fireEvent.click(screen.getAllByRole("radio", { name: /^yes$/ })[1]);
		expect(bar.textContent).not.toContain("answered");
		expect((within(bar).getByRole("button", { name: /Send answers/ }) as HTMLButtonElement).disabled).toBe(false);
	});

	it("scrolls the open card to the top of the sheet's body when it opens and when it moves on", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [choice("a1", "first?"), choice("a2", "second?")],
		});
		/* happy-dom lays nothing out, so the geometry is stated: the region's top is
		   0 and every card sits 200 px below it, inside a 12 px body padding. */
		const rect = (top: number) => ({ top, bottom: top, left: 0, right: 0, width: 0, height: 0, x: 0, y: top, toJSON: () => ({}) });
		const original = Element.prototype.getBoundingClientRect;
		Element.prototype.getBoundingClientRect = function (this: Element) {
			return this.getAttribute("data-testid") === "ask-card" ? (rect(200) as DOMRect) : (rect(0) as DOMRect);
		};
		try {
			open();
			const card = await screen.findByTestId("ask-card");
			const region = card.closest(".overflow-y-auto") as HTMLElement;
			await waitFor(() => expect(region.scrollTop).toBe(188));
			region.scrollTop = 0;
			fireEvent.click(screen.getByTestId("ask-row"));
			await waitFor(() => expect(region.scrollTop).toBe(188));
		} finally {
			Element.prototype.getBoundingClientRect = original;
		}
	});

	it("says the sheet is reading while the first read is in flight", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(() => new Promise(() => {}));
		open();
		expect(screen.getByText("Reading questions…")).toBeTruthy();
	});

	it("lays the panel out as card, waiting rows, Settled, then the pinned bar", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				ask({ ask_id: "gone", status: "answered", delivered: true, answers: { q1: ["blue"] } }),
				choice("open", "first?"),
				choice("wait", "second?"),
			],
		});
		open();
		const card = await screen.findByTestId("ask-card");
		await waitFor(() => expect(screen.getAllByTestId("ask-row").length).toBe(2));
		const rows = screen.getAllByTestId("ask-row");
		const waiting = rows.find((row) => row.getAttribute("data-ask-id") === "wait") as HTMLElement;
		const settledRow = rows.find((row) => row.getAttribute("data-ask-id") === "gone") as HTMLElement;
		const heading = screen.getByRole("heading", { name: "Settled" });
		const bar = screen.getByTestId("asks-bar");
		const before = (a: Element, b: Element) =>
			Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
		expect(before(card, waiting)).toBe(true);
		expect(before(waiting, heading)).toBe(true);
		expect(before(heading, settledRow)).toBe(true);
		expect(before(settledRow, bar)).toBe(true);
		expect(screen.queryByText("Earlier")).toBeNull();
	});

	it("draws a waiting row with the question first and the whole state sentence under it", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				choice("open", "first?"),
				{ ...choice("wait", "Which cookie should the agent bake for the long weekend party?"), urgent: true },
			],
		});
		open();
		const row = await screen.findByTestId("ask-row");
		const [question, state] = Array.from(row.querySelectorAll(":scope > span:first-child > span")) as HTMLElement[];
		expect(question.textContent).toBe("Which cookie should the agent bake for the long weekend party?");
		expect(question.className).toContain("line-clamp-2");
		expect(question.className).toContain("font-medium");
		expect(question.className).toContain("text-ink");
		// The state sentence is never cut, and its urgent mark is warning ink.
		expect(state.className).not.toContain("truncate");
		expect(state.textContent).toMatch(/^Queued — the agent is continuing; expires in .* · Urgent$/);
		expect(state.querySelector(".text-warning")?.textContent).toContain("· Urgent");
		// The caret mirrors in a right-to-left layout.
		expect(row.querySelector(".lo-directional")).toBeTruthy();
	});

	it("names the conversation on the first line inside a row when the sheet holds several", async () => {
		rows = [
			summary({ session_id: "mine" }),
			summary({ session_id: "other", conversation_name: "tax receipts" }),
		];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				{ ...choice("a", "first?"), session_id: "mine" },
				{ ...choice("b", "second?"), session_id: "other" },
			],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="" />);
		const row = await screen.findByTestId("ask-row");
		const lines = Array.from(row.querySelector("span")!.children) as HTMLElement[];
		expect(lines[0].textContent).toBe("tax receipts");
		expect(lines[1].textContent).toBe("second?");
	});

	it("draws Try again as an outline button, left-aligned under the error line", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("could not reach the daemon"));
		open();
		const line = await screen.findByText("could not reach the daemon");
		const retry = screen.getByRole("button", { name: "Try again" });
		expect(retry.className).toContain("border");
		expect(retry.className).toContain("border-control");
		expect(line.parentElement).toBe(retry.parentElement);
		expect(line.parentElement?.className).toContain("items-start");
	});

	it("reopens on the first answerable ask, not on the last visit's expanded row", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [
				choice("a", "first question?"),
				choice("b", "second question?"),
				ask({ ask_id: "gone", status: "answered", delivered: true, answers: { q1: ["blue"] } }),
			],
		});
		const view = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await screen.findByTestId("ask-card");
		// Visit 1: open a settled row, replacing the card.
		fireEvent.click(await screen.findByRole("button", { name: /blue|answered/i }));
		await waitFor(() => expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("gone"));
		view.rerender(<AsksSheet open={false} onClose={() => {}} currentSessionId="mine" />);
		view.rerender(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card").getAttribute("data-ask-id")).toBe("a"));
	});

	it("puts focus on the sheet's close button once the last ask is answered", async () => {
		const { sendCommand } = await import("./api");
		let current = [choice("a1", "only question?")];
		(getAsks as ReturnType<typeof vi.fn>).mockImplementation(async () => ({ asks: current }));
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementation(async () => {
			current = [{ ...current[0], status: "answered", delivered: true, answers: { q1: ["yes"] } }];
			return { ok: true, detail: "answered" };
		});
		open();
		await screen.findByTestId("ask-card");
		fireEvent.click(screen.getAllByRole("radio")[0]);
		const send = await screen.findByRole("button", { name: /Send answer/ });
		// The reader's place is on Send when they press it (the sheet's own
		// opening focus is on the ✕, which would make this vacuous).
		send.focus();
		expect(document.activeElement).toBe(send);
		fireEvent.click(send);
		await waitFor(() =>
			expect(document.activeElement).toBe(screen.getByRole("button", { name: "close sheet" })),
		);
	});

	it("pins a footer that scrolls inside a cap, so the bar and the reasons cannot clip", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [choice("a", "question?")] });
		open();
		await screen.findByTestId("ask-card");
		const bar = screen.getByTestId("asks-bar");
		const footer = bar.closest("[data-sheet-footer]") as HTMLElement;
		expect(footer.className).toContain("overflow-y-auto");
		expect(footer.className).toContain("min-h-0");
	});
});

