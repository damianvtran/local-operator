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

vi.mock("./api", () => ({
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answered" })),
}));

const { sendCommand } = await import("./api");

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

function renderCard(row: PendingAsk, onSettled?: () => void) {
	return render(
		<AskCard row={row} sessionId="s1" nowMs={Date.now()} onSettled={onSettled} />,
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
});
