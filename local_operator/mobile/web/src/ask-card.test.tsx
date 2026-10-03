// @vitest-environment happy-dom
//
// The queued-ask card: the ATOMIC submit (one complete map, every question id
// present — the queue refuses a partial one), the deliberate skip (sent as an
// empty list, which is the contract's own spelling for "no answer"), the
// answerable-after-deadline state, and the rule that a refusal is rendered in
// the DAEMON's own words rather than humanised here.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AskCard } from "./components/ask-card";
import type { AskQuestion, PendingAsk } from "./types";

vi.mock("./api", async (importOriginal) => {
	/* PARTIAL, because the card now imports `HttpError` to tell a TERMINAL refusal
	   (a gone conversation, `ask_session_gone`) from one the user can act on. The
	   bare `{ sendCommand }` mock left that import undefined, so the card threw
	   the moment a send was refused. */
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })),
	};
});

const { sendCommand, HttpError } = await import("./api");

function question(patch: Partial<AskQuestion> = {}): AskQuestion {
	return {
		id: "q1",
		question: "ship the fix?",
		options: [
			{ label: "yes", description: "" },
			{ label: "no", description: "" },
		],
		multi: false,
		secret: false,
		persist: false,
		...patch,
	};
}

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "ask-1",
		created_at: 1,
		expires_at: Date.now() + 900_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [question()],
		...patch,
	};
}

function renderCard(row: PendingAsk, onSettled?: () => void, runtimeLive?: boolean) {
	return render(
		<AskCard
			row={row}
			sessionId="s1"
			nowMs={Date.now()}
			runtimeLive={runtimeLive}
			onSettled={onSettled}
		/>,
	);
}

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("AskCard", () => {
	it("sends the whole ask in one atomic body", async () => {
		const settled = vi.fn();
		renderCard(ask(), settled);
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_respond",
				ask_id: "ask-1",
				answers: { q1: ["yes"] },
			}),
		);
		await waitFor(() => expect(settled).toHaveBeenCalled());
	});

	it("refuses to send a partial map, and sends an explicit skip as an empty list", () => {
		renderCard(
			ask({
				questions: [
					question(),
					question({ id: "q2", question: "and the date?" }),
				],
			}),
		);
		const send = screen.getByRole("button", { name: /send answers/ });
		expect((send as HTMLButtonElement).disabled).toBe(true);

		fireEvent.click(screen.getAllByRole("button", { name: /^yes$/ })[0]);
		expect((send as HTMLButtonElement).disabled).toBe(true);

		/* Skipping is not "leaving it blank": it is the explicit empty list the
		   queue's contract defines, so it is what unblocks the submit. */
		fireEvent.click(screen.getAllByRole("button", { name: /skip — send no answer/ })[1]);
		expect((send as HTMLButtonElement).disabled).toBe(false);
		fireEvent.click(send);
		return waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_respond",
				ask_id: "ask-1",
				answers: { q1: ["yes"], q2: [] },
			}),
		);
	});

	it("takes a free-text answer as the typed string, and states it is collecting for all questions", async () => {
		renderCard(ask({ questions: [question({ options: [] })] }));
		fireEvent.change(screen.getByPlaceholderText("your answer"), {
			target: { value: "some prose" },
		});
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_respond",
				ask_id: "ask-1",
				answers: { q1: ["some prose"] },
			}),
		);
	});

	it("never renders a secret question's answer back, and says it is not stored in the transcript", () => {
		renderCard(ask({ questions: [question({ options: [], secret: true, persist: true })] }));
		expect(screen.getByPlaceholderText("paste secret").getAttribute("type")).toBe("password");
		expect(screen.getByText(/not shown in the transcript/)).toBeTruthy();
	});

	it("keeps a timed-out ask answerable, with the deadline stated", () => {
		renderCard(ask({ status: "timed_out" }));
		expect(screen.getByText(/you can still answer/)).toBeTruthy();
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
	});

	it("offers no controls on a settled ask, and no error register on an expired one", () => {
		const { unmount } = renderCard(
			ask({ status: "answered", delivered: true, answers: { q1: ["yes"] } }),
		);
		expect(screen.queryByRole("button", { name: /send answer/ })).toBeNull();
		expect(screen.getByText("yes")).toBeTruthy();
		unmount();

		renderCard(ask({ status: "expired" }));
		expect(screen.queryByRole("button", { name: /send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /dismiss/ })).toBeNull();
	});

	it("declines and dismisses through their own ops", async () => {
		const { unmount } = renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^decline$/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_decline", ask_id: "ask-1" }),
		);
		unmount();
		vi.clearAllMocks();
		/* DISMISS IS A `timed_out` ACTION (agent review round 1, R1): the queue
		   accepts no other status, so the control is offered only where it can
		   land. The open-ask assertion below is the regression guard — the button
		   used to be rendered on an open ask, where every tap collected a
		   refusal sentence that was false about the row under it. */
		renderCard(ask({ status: "timed_out" }));
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_dismiss", ask_id: "ask-1" }),
		);
	});

	it("offers dismiss only on a timed-out ask", () => {
		const { unmount } = renderCard(ask());
		expect(screen.queryByRole("button", { name: /dismiss/ })).toBeNull();
		unmount();
		renderCard(ask({ status: "timed_out" }));
		expect(screen.getByRole("button", { name: /dismiss — send no reply/ })).toBeTruthy();
	});

	it("keeps the answer draft across an unmount — collapse and return keeps it", async () => {
		/* Q-1 (QA) = U1 (UX): §5.0-R7 requires BOTH drafts to survive a collapse.
		   The draft used to be component state and the sheet unmounts when it
		   closes, so collapsing to read the transcript and coming back re-answered
		   every question from scratch. This asserts the survival at the level the
		   failure happened — same ask id, a fresh mount. */
		const first = renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		expect(screen.getByRole("button", { name: /yes/ }).getAttribute("aria-pressed")).toBe("true");
		first.unmount();

		renderCard(ask());
		expect(screen.getByRole("button", { name: /yes/ }).getAttribute("aria-pressed")).toBe("true");
		expect(
			(screen.getByRole("button", { name: /send answer/ }) as HTMLButtonElement).disabled,
		).toBe(false);
	});

	it("composes two picks made in the same tick, instead of losing the first", async () => {
		/* Found by the capture rig, not by a unit test: the rig clicks both of the
		   head ask's options inside ONE evaluate, so both handlers closed over the
		   same pre-tick draft and the second overwrote the first — the form stayed
		   incomplete and the send control stayed disabled while the frame showed an
		   option pressed. One `act` is what reproduces that batching. */
		const two = question({ id: "q1", question: "first?" });
		const other = question({ id: "q2", question: "second?" });
		renderCard(ask({ questions: [two, other] }));
		const picks = screen.getAllByRole("button", { name: /^yes$/ });
		await act(async () => {
			fireEvent.click(picks[0]);
			fireEvent.click(picks[1]);
		});
		expect(
			(screen.getByRole("button", { name: /send answers/ }) as HTMLButtonElement).disabled,
		).toBe(false);
	});

	it("marks the recommended option from the wire's own index", () => {
		renderCard(ask({ questions: [question({ recommended: 0 })] }));
		expect(screen.getByText(/· recommended/)).toBeTruthy();
	});

	it("says why an ask with nothing left to answer cannot be sent", () => {
		/* R6 (agent review round 1): `draft_question_ids` had every question, so
		   the card rendered no fields and a disabled send with no explanation. */
		renderCard(ask({ draft_question_ids: ["q1"] }));
		expect(screen.getByText(/nothing left to answer here/)).toBeTruthy();
		expect(
			(screen.getByRole("button", { name: /send answer/ }) as HTMLButtonElement).disabled,
		).toBe(true);
	});

	it("degrades a settled row with no questions to a plain line instead of throwing", () => {
		/* R5 (agent review round 1): `row.questions[0]` on a row that arrived
		   without the list threw during render, which unmounts the sheet. */
		renderCard({ ...ask({ status: "answered" }), questions: undefined } as unknown as PendingAsk);
		expect(screen.getByTestId("ask-card")).toBeTruthy();
	});

	it("shows the queue's own refusal sentence verbatim, and restores the controls", async () => {
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("already answered by desktop."),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() => expect(screen.getByText("already answered by desktop.")).toBeTruthy());
		/* The refusal is a state the user can act on, not a dead card. */
		expect(
			(screen.getByRole("button", { name: /send answer/ }) as HTMLButtonElement).disabled,
		).toBe(false);
	});

	it("states the wait on a COLD conversation instead of a bare '…'", async () => {
		/* D1 = UX U1: a cold answer is an engage + dial + ack in ONE call (~30 s
		   measured against a warm 1–3 s), and the card's entire in-flight
		   affordance used to be the single glyph `…` — indistinguishable from a
		   hang on the phone's primary answer path. */
		let release: (() => void) | undefined;
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() =>
				new Promise((resolve) => {
					release = () => resolve({ ok: true, detail: "answered" });
				}),
		);
		renderCard(ask(), undefined, false);
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		expect(await screen.findByText(/this can take up to ~30 s/)).toBeTruthy();
		release?.();
	});

	it("promises no wait it cannot substantiate (warm, or an older daemon)", async () => {
		/* ABSENCE IS NOT `false`: `runtime_live` is omitted by an older daemon and
		   by a row read from a projection, and the card must not assert a ~30 s
		   envelope it has no way to know about. */
		let release: (() => void) | undefined;
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() =>
				new Promise((resolve) => {
					release = () => resolve({ ok: true, detail: "answered" });
				}),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() => expect(screen.getByRole("button", { name: "…" })).toBeTruthy());
		expect(screen.queryByText(/this can take up to ~30 s/)).toBeNull();
		release?.();
	});

	it("withholds the controls on a TERMINAL refusal, and keeps the sentence", async () => {
		/* D3: a gone conversation can never be read, so every further tap is a
		   guaranteed 409 — the surface must stop inviting one. The sentence is the
		   daemon's own, written for THIS op (D2). */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new HttpError(
				409,
				"this conversation no longer exists — nothing was sent; the ask can never be read.",
				"ask_session_gone",
			),
		);
		renderCard(ask({ status: "timed_out" }), undefined, false);
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		await waitFor(() => expect(screen.getByText(/nothing was sent/)).toBeTruthy());
		expect(screen.queryByRole("button", { name: /send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^decline$/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /dismiss/ })).toBeNull();
		/* AND THE STATE LINE STANDS DOWN: "Queued — the agent is continuing; expires
		   in 11 m" beside "this conversation no longer exists" is the contradiction
		   the refusal is supposed to end (seen in the capture frame). */
		expect(screen.queryByRole("status")).toBeNull();
	});

	it("keeps the controls for a refusal the user can act on", async () => {
		/* The D3 rule is scoped to the TERMINAL code: a lost race is still a state
		   with a move (read who beat you to it), which is round 1's own design. */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new HttpError(422, "already answered by desktop."),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() => expect(screen.getByText("already answered by desktop.")).toBeTruthy());
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
	});

	it("places a dismiss refusal beside the dismiss control", async () => {
		/* D5: the sentence used to render above the send/decline row, which put it
		   60 px from the button that produced it with the other pair in between. */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("only a timed-out ask can be dismissed; it is still open."),
		);
		renderCard(ask({ status: "timed_out" }));
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		const sentence = await screen.findByText(/only a timed-out ask/);
		const dismiss = screen.getByRole("button", { name: /dismiss — send no reply/ });
		expect(
			dismiss.compareDocumentPosition(sentence) & Node.DOCUMENT_POSITION_FOLLOWING,
		).toBeTruthy();
	});

	it("says a COLD dismissal brings the session up, before the tap", () => {
		/* U3: the label promises "send no reply" and that stays true, but on a cold
		   conversation the relay also brings it up to RECORD the dismissal — the
		   conversation surfaces as active, which the copy owed the reader. */
		const cold = renderCard(ask({ status: "timed_out" }), undefined, false);
		expect(screen.getByText(/bringing the session up to record that/)).toBeTruthy();
		cold.unmount();
		renderCard(ask({ status: "timed_out" }));
		expect(screen.queryByText(/bringing the session up to record that/)).toBeNull();
	});
});
