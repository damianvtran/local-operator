// @vitest-environment happy-dom
//
// The asks sheet — the EXPANDED surface (design §5.3). It reads the aggregate
// from `GET /api/asks` (index-backed, so it answers for a conversation whose
// runtime is gone), shows only the rows of the conversation it was opened from,
// and re-reads when the outstanding population moves rather than waiting for a
// manual refresh.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
		await waitFor(() => expect(screen.getByText("asks · 1 question")).toBeTruthy());
		const cards = screen.getAllByTestId("ask-card");
		expect(cards.length).toBe(1);
		expect(cards[0].getAttribute("data-ask-id")).toBe("mine-1");
	});

	it("says nothing is waiting when only other conversations have asks", async () => {
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "other-1", session_id: "other" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/nothing waiting/)).toBeTruthy());
		expect(screen.queryByTestId("ask-card")).toBeNull();
	});

	it("answers against the opening conversation's session route", async () => {
		const { sendCommand } = await import("./api");
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		fireEvent.change(screen.getByPlaceholderText("your answer"), {
			target: { value: "yes — dark behind the flag" },
		});
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
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
		await waitFor(() => expect(screen.getByText(/nothing waiting/)).toBeTruthy());
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
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(2));
		fireEvent.click(screen.getAllByRole("button", { name: /^dismiss…$/ })[0]);
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() =>
			expect(screen.getByTestId("ask-receipt").textContent).toBe(
				"Dismissed — no longer needed",
			),
		);
		/* AND THE OTHER ASK IS STILL THERE: the receipt is a line, not a replacement
		   for the list. */
		expect(screen.getAllByTestId("ask-card").length).toBe(1);
		expect(screen.queryByTestId("ask-sheet-empty")).toBeNull();
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
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(2));
		fireEvent.click(screen.getAllByRole("button", { name: /^dismiss…$/ })[0]);
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() => expect(screen.getByTestId("ask-receipt")).toBeTruthy());

		/* The operator answers the remaining ask. */
		fireEvent.change(screen.getByPlaceholderText("your answer"), {
			target: { value: "yes — dark behind the flag" },
		});
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in chat/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("mine", {
				op: "ask_dismiss",
				ask_id: "a1",
				reason: "answered_in_chat",
			}),
		);
		await waitFor(() => expect(screen.getByText("Dismissed — answered in chat")).toBeTruthy());
		expect(screen.queryByText(/nothing waiting/)).toBeNull();
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
		await waitFor(() => expect(screen.getByText("asks · 1 timed out")).toBeTruthy());
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
		expect(screen.getByText("asks")).toBeTruthy();
		expect(screen.getByText(/nothing waiting/)).toBeTruthy();
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
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(2));
		const cards = screen.getAllByTestId("ask-card");
		expect(cards[0].getAttribute("data-ask-id")).toBe("a-old");
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
		fireEvent.click(screen.getByRole("button", { name: /try again/ }));
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
	});

	it("warns before the tap when the owning conversation has ended", async () => {
		rows = [summary({ ended: true })];
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText(/this conversation has ended/)).toBeTruthy());
		/* THE COST IS ON THE LINE (design round 1, D4 = UX U1/U4): the relay now
		   reopens it itself, so "may need it reopened" described a manual remedy
		   that no longer exists — while the ~30 s the cold arm costs went unsaid. */
		expect(screen.getByText(/answering will reopen it \(this can take up to ~30 s\)/)).toBeTruthy();
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
		await waitFor(() => expect(screen.getByText(/this conversation is not running/)).toBeTruthy());
		expect(screen.getByText(/answering will bring it up \(this can take up to ~30 s\)/)).toBeTruthy();
	});

	it("says nothing about a wait when the relay can deliver now", async () => {
		/* The other half of the rule: a live conversation answers in ~1 s, so a
		   ~30 s promise there would be its own lie. */
		rows = [summary({ ended: false })];
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({ asks: [ask({ runtime_live: true })] });
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		expect(screen.queryByText(/this can take up to ~30 s/)).toBeNull();
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
		await waitFor(() => expect(screen.getByTestId("ask-card")).toBeTruthy());
		expect(screen.queryByText(/this conversation is not running/)).toBeNull();
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
			expect(screen.getByText(/the ask can never be read/)).toBeTruthy(),
		);
		expect(screen.queryByText(/this conversation is not running/)).toBeNull();
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
		expect(screen.queryByText(/this conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
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
		await waitFor(() => expect(screen.getByText("asks · 1 question")).toBeTruthy());
	});

	it("carries no count at all when every row is dead", async () => {
		/* The other half: nothing is waiting on the user, so the title states the
		   surface rather than a number nobody can act on. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "gone", runtime_live: false, durable: false })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("asks")).toBeTruthy());
	});

	it("titles a lapsed-only queue by what it holds, and counts only offered questions as waiting", async () => {
		/* Design round 1, D1: "asks · 1 question" above a card reading Lapsed
		   stated a live decision one element above the card that says there is
		   none. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "t", status: "timed_out" })],
		});
		const { unmount } = render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("asks · 1 timed out")).toBeTruthy());
		/* Collapsed behind its badge (D11): the sheet lists it, the user opens it. */
		expect(screen.getByRole("button", { name: /Timed out/ }).getAttribute("aria-expanded")).toBe(
			"false",
		);
		unmount();

		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "o" }), ask({ ask_id: "t", status: "timed_out" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getByText("asks · 1 question")).toBeTruthy());
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
		fireEvent.click(screen.getByRole("button", { name: /Timed out/ }));
		expect(screen.getByText("the agent goes ahead without your answer")).toBeTruthy();
		expect(screen.queryByText(/this conversation is not running/)).toBeNull();
	});

	it("keeps a timed-out ask collapsed beside an open one, which stays fully offered", async () => {
		/* D11: only the lapsed row collapses; the open card keeps its controls. */
		(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
			asks: [ask({ ask_id: "o" }), ask({ ask_id: "t", status: "timed_out" })],
		});
		render(<AsksSheet open onClose={() => {}} currentSessionId="mine" />);
		await waitFor(() => expect(screen.getAllByTestId("ask-card").length).toBe(2));
		const badges = screen.getAllByRole("button", { name: /Timed out/ });
		expect(badges.length).toBe(1);
		expect(badges[0].getAttribute("aria-expanded")).toBe("false");
		expect(screen.getByPlaceholderText("your answer")).toBeTruthy();
		expect(screen.queryByRole("button", { name: /dismiss — send no reply/ })).toBeNull();
	});
});
