// @vitest-environment happy-dom
//
// The queued-ask card: the ATOMIC submit (one complete map, every question id
// present — the queue refuses a partial one), the deliberate skip (sent as an
// empty list, which is the contract's own spelling for "no answer"), the
// read-only lapsed state after a deadline, and the rule that a refusal is rendered in
// the DAEMON's own words rather than humanised here.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

function renderCard(
	row: PendingAsk,
	onSettled?: () => void,
	onDismissed?: (askId: string, reason: "moot" | "answered_in_chat" | undefined) => void,
) {
	return render(
		<AskCard
			row={row}
			sessionId="s1"
			nowMs={Date.now()}
			onSettled={onSettled}
			onDismissed={onDismissed}
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
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
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
		const send = screen.getByRole("button", { name: /Send answers/ });
		expect((send as HTMLButtonElement).disabled).toBe(true);

		fireEvent.click(screen.getAllByRole("radio", { name: /^yes$/ })[0]);
		expect((send as HTMLButtonElement).disabled).toBe(true);

		/* Skipping is not "leaving it blank": it is the explicit empty list the
		   queue's contract defines, so it is what unblocks the submit. */
		fireEvent.click(screen.getAllByRole("button", { name: /Skip \(send no answer\)/ })[1]);
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
		fireEvent.change(screen.getByPlaceholderText("Your answer"), {
			target: { value: "some prose" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
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
		expect(screen.getByPlaceholderText("Secret value").getAttribute("type")).toBe("password");
		expect(screen.getByText(/not shown in the transcript/)).toBeTruthy();
	});

	it("draws a timed-out ask read-only: the question, no answer controls, Dismiss kept", () => {
		/* Ben's report (2026-10-15): the phone offered the options of an ask whose
		   deadline had passed, as if the agent were still waiting on them. The
		   agent has gone ahead without the answer, so the card reads that and offers
		   only the one op the queue accepts on a timeout — dismiss.

		   The collapse behind a `Timed out` row is the SHEET's now (ask-panel
		   parity: every waiting or lapsed ask but the open one is one line), so this
		   card is always the opened, read-only one and states its whole line. */
		renderCard(ask({ status: "timed_out" }));
		expect(
			screen.getByText("Timed out — the agent goes ahead without your answer"),
		).toBeTruthy();
		expect(screen.queryByRole("radio")).toBeNull();
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^Decline$/ })).toBeNull();
		expect(screen.getByRole("button", { name: /^Dismiss$/ })).toBeTruthy();
		/* The question is stated once. */
		expect(screen.getAllByText("ship the fix?").length).toBe(1);
	});

	it("dismisses a timed-out ask with no reason", async () => {
		const settled = vi.fn();
		renderCard(ask({ status: "timed_out" }), settled);
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_dismiss", ask_id: "ask-1" }),
		);
		await waitFor(() => expect(settled).toHaveBeenCalled());
	});

	it("keeps an OPEN ask answerable when only the phone's clock says its deadline passed", () => {
		/* Review round 1, R3: the phone's clock may run ahead of the runtime's,
		   and a stale `open` row from a stopped runtime arrives already re-derived
		   to `timed_out` by the index readers on the runtime's host. So an `open`
		   row is still live to the agent: the card keeps its controls and says the
		   deadline has passed locally, without calling it lapsed. */
		renderCard(ask({ status: "open", expires_at: Date.now() - 60_000 }));
		expect(screen.getByRole("radio", { name: /^yes$/ })).toBeTruthy();
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
		expect(screen.getByText("Queued — the agent is continuing; deadline passed")).toBeTruthy();
		expect(screen.queryByText(/^Timed out —/)).toBeNull();
		/* R5: an OPEN ask has a door of its own — the quiet `close…`, which asks
		   WHY before anything is sent. What it must NOT offer is the lapsed row's
		   one-tap dismissal, which says "send no reply" over an ask the agent is
		   still owed an answer to. */
		expect(screen.getByRole("button", { name: /^Dismiss…$/ })).toBeTruthy();
		expect(screen.queryByRole("button", { name: /^Dismiss$/ })).toBeNull();
	});

	it("offers no controls on a settled ask, and no error register on an expired one", () => {
		const { unmount } = renderCard(
			ask({ status: "answered", delivered: true, answers: { q1: ["yes"] } }),
		);
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.getByText("yes")).toBeTruthy();
		unmount();

		renderCard(ask({ status: "expired" }));
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /dismiss/ })).toBeNull();
	});

	it("declines and dismisses through their own ops", async () => {
		const { unmount } = renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Decline$/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_decline", ask_id: "ask-1" }),
		);
		unmount();
		vi.clearAllMocks();
		/* THE ONE-TAP DISMISS IS THE LAPSED ROW'S (agent review round 1, R1, kept
		   by R5): the button says "send no reply", which is true only once the
		   agent has walked past the deadline. An OPEN ask's door is `close…` plus
		   a reason, exercised in its own cases below. */
		renderCard(ask({ status: "timed_out" }));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_dismiss", ask_id: "ask-1" }),
		);
	});

	it("offers the one-tap dismissal on a timed-out ask and the reason row on an open one", () => {
		const { unmount } = renderCard(ask());
		expect(screen.queryByRole("button", { name: /^Dismiss$/ })).toBeNull();
		expect(screen.getByRole("button", { name: /^Dismiss…$/ })).toBeTruthy();
		unmount();
		renderCard(ask({ status: "timed_out" }));
		expect(screen.getByRole("button", { name: /^Dismiss$/ })).toBeTruthy();
		expect(screen.queryByRole("button", { name: /^Dismiss…$/ })).toBeNull();
	});

	it("closes an open ask with a reason, and sends nothing until one is chosen", async () => {
		/* R5 (design §15's amended D6): an open ask can be put away from the phone.
		   The tap that REVEALS the reasons is not the dismissal — a reason the
		   operator did not choose is not one they gave — and the two reasons are
		   what the agent acts on (`moot` closes the question, `answered_in_chat`
		   tells it the answer is already in the conversation). */
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(sendCommand).not.toHaveBeenCalled();
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_dismiss",
				ask_id: "ask-1",
				reason: "moot",
			}),
		);
	});

	it("sends answered_in_chat when the operator says they replied in chat", async () => {
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in the conversation/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_dismiss",
				ask_id: "ask-1",
				reason: "answered_in_chat",
			}),
		);
	});

	it("puts the reasons away again on cancel, sending nothing", () => {
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Cancel$/ }));
		expect(screen.queryByRole("button", { name: /^No longer needed/ })).toBeNull();
		expect(screen.getByRole("button", { name: /^Dismiss…$/ })).toBeTruthy();
		expect(sendCommand).not.toHaveBeenCalled();
	});

	it("shows a relay or bridge refusal beside the reasons it answered", async () => {
		/* A relay's open ask is dismissed at its origin, and the queue says so in
		   its own sentence. That sentence belongs beside the reason the operator
		   pressed (D5's rule), and the controls come back so the card is not a
		   dead end. */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("this ask was asked by a relayed conversation — dismiss it there."),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		const sentence = await screen.findByText(/dismiss it there/);
		const reason = screen.getByRole("button", { name: /^No longer needed/ });
		expect(
			reason.compareDocumentPosition(sentence) & Node.DOCUMENT_POSITION_FOLLOWING,
		).toBeTruthy();
		/* INERT NOW (UX round 1, U8): the sentence is shown beside the control that
		   produced it, and the control stops inviting the same refusal again. */
		expect((reason as HTMLButtonElement).disabled).toBe(true);
	});

	it("does not promise a ~30 s bring-up for a dismissal on a cold conversation", async () => {
		/* Agent review round 1, M1: the dismissal takes the no-boot door on every
		   surface (`dismiss_cold`, §14/fork A), so the cold engage clock belongs to
		   an ANSWER. The gate used to be `busy !== "" && cold`, and dismissing an
		   open ask on a cold row printed "bringing the session up — ~30 s" beside
		   two dead `…` buttons: the card asserted the opposite of the contract. */
		let release: (value: unknown) => void = () => {};
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise((resolve) => { release = resolve; }),
		);
		renderCard(ask({ runtime_live: false }));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		expect(screen.queryByText(/Bringing the conversation up/)).toBeNull();
		await act(async () => { release({ ok: true, detail: "dismissed" }); });
	});

	it("clears a refusal's sentence when the reasons are cancelled", async () => {
		/* UX round 2, U10 = agent review round 2, m1: the sentence renders outside
		   the `closing` ternary, so `cancel` left an old refusal sitting under a
		   fresh-looking `dismiss…` until the next op happened to clear it. */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("you already dismissed this."),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await screen.findByText("you already dismissed this.");
		fireEvent.click(screen.getByRole("button", { name: /^Cancel$/ }));
		expect(screen.queryByText("you already dismissed this.")).toBeNull();
		expect(screen.getByRole("button", { name: /^Dismiss…$/ })).toBeTruthy();
	});

	it("names the act, and states what each reason tells the agent", () => {
		/* Design round 1, D1 = UX round 1, U6: the reveal used to be three controls
		   on one line with no heading and no explanation, so "Answered in the conversation"
		   could be read as "send my answer" while sitting 8.5 px under `decline`.
		   The heading says "Dismiss", and each row carries its note. */
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(screen.getByText("Dismiss this question")).toBeTruthy();
		expect(screen.getByRole("button", { name: /^No longer needed/ }).textContent).toContain(
			"The agent stops waiting for an answer",
		);
		expect(
			screen.getByRole("button", { name: /^Answered in the conversation/ }).textContent,
		).toContain("You already replied in the conversation");
	});

	it("keeps the pressed reason's label while the dismissal is in flight", async () => {
		/* UX round 1, U3 (design N3): both buttons used to become `…`, which erased
		   the only on-screen statement of which reason had been sent. */
		let release: (value: unknown) => void = () => {};
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise((resolve) => { release = resolve; }),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in the conversation/ }));
		await waitFor(() =>
			expect(screen.getByRole("button", { name: /^Answered in the conversation/ }).textContent).toContain(
				"Answered in the conversation",
			),
		);
		expect(
			screen.getByRole("button", { name: /^Answered in the conversation/ }).getAttribute("aria-busy"),
		).toBe("true");
		await act(async () => { release({ ok: true, detail: "dismissed" }); });
	});

	it("hands the reason to the sheet on an accepted dismissal", async () => {
		/* UX round 1, U2: the card leaves the aggregate, so the sheet needs the
		   operator's own word to show a receipt instead of its empty state. */
		const dismissed = vi.fn();
		renderCard(ask(), undefined, dismissed);
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		/* THE ASK ID RIDES WITH THE REASON (agent review round 2, M1): the sheet
		   keys its receipt to the ask it was about, so a later dismissal cannot be
		   confused with this one. */
		await waitFor(() => expect(dismissed).toHaveBeenCalledWith("ask-1", "moot"));
	});

	it("draws the context line under the question it refers to", () => {
		/* Design round 1, D3: state → context → question made "this" point at a
		   line the reader had not reached yet; the terminal paints it under. */
		const at = new Date(2026, 9, 23, 19, 36, 0).getTime();
		renderCard(ask({ user_messages_since: 3, last_user_message_at: at }));
		const context = screen.getByTestId("ask-context");
		const questionText = screen.getByText("ship the fix?");
		expect(
			questionText.compareDocumentPosition(context) & Node.DOCUMENT_POSITION_FOLLOWING,
		).toBeTruthy();
	});

	it("draws the stale-ask context line from the wire pair, and only when it can", () => {
		/* SA-D7: "3 messages from you since this was asked (last 19:36)". ABSENCE
		   IS THE CONTRACT — an older runtime sends no field, and a zero says the ask
		   is newer than every message — so the card draws NOTHING in either case
		   rather than a zero the reader would have to disbelieve. */
		const at = new Date(2026, 9, 23, 19, 36, 0).getTime();
		const { unmount } = renderCard(
			ask({ user_messages_since: 3, last_user_message_at: at }),
		);
		const line = screen.getByTestId("ask-context");
		expect(line.textContent).toBe("3 messages from you since this was asked (last 19:36)");
		unmount();

		renderCard(ask({ user_messages_since: 1, last_user_message_at: at }));
		expect(screen.getByTestId("ask-context").textContent).toBe(
			"1 message from you since this was asked (last 19:36)",
		);
	});

	it("draws no context line for an absent count, a zero, or a settled ask", () => {
		const at = new Date(2026, 9, 23, 19, 36, 0).getTime();
		const cases = [
			/* an older runtime: neither field */
			ask(),
			/* the ask is newer than every message */
			ask({ user_messages_since: 0, last_user_message_at: at }),
			/* half a pair: no clock to state */
			ask({ user_messages_since: 3 }),
			/* settled: no wait left to be stale */
			ask({ status: "answered", delivered: true, user_messages_since: 3, last_user_message_at: at }),
		];
		for (const row of cases) {
			const { unmount } = renderCard(row);
			expect(screen.queryByTestId("ask-context")).toBeNull();
			unmount();
		}
	});

	it("keeps the answer draft across an unmount — collapse and return keeps it", async () => {
		/* Q-1 (QA) = U1 (UX): §5.0-R7 requires BOTH drafts to survive a collapse.
		   The draft used to be component state and the sheet unmounts when it
		   closes, so collapsing to read the transcript and coming back re-answered
		   every question from scratch. This asserts the survival at the level the
		   failure happened — same ask id, a fresh mount. */
		const first = renderCard(ask());
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		expect(screen.getByRole("radio", { name: /yes/ }).getAttribute("aria-checked")).toBe("true");
		first.unmount();

		renderCard(ask());
		expect(screen.getByRole("radio", { name: /yes/ }).getAttribute("aria-checked")).toBe("true");
		expect(
			(screen.getByRole("button", { name: /Send answer/ }) as HTMLButtonElement).disabled,
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
		const picks = screen.getAllByRole("radio", { name: /^yes$/ });
		await act(async () => {
			fireEvent.click(picks[0]);
			fireEvent.click(picks[1]);
		});
		expect(
			(screen.getByRole("button", { name: /Send answers/ }) as HTMLButtonElement).disabled,
		).toBe(false);
	});

	describe("the Other row", () => {
		/* The answer draft is persisted in localStorage under the ask id and every
		   case here reuses `ask-1`; without this, one case's open `Other` seeds the
		   next. */
		beforeEach(() => localStorage.clear());
		const otherRow = (multi = false) =>
			screen.getByRole(multi ? "checkbox" : "radio", { name: /^Other/ });
		const typeOther = (text: string) =>
			fireEvent.change(screen.getByRole("textbox", { name: "ship the fix?" }), {
				target: { value: text },
			});
		const send = () => screen.getByRole("button", { name: /Send answer/ }) as HTMLButtonElement;
		const sentAnswers = (answers: Record<string, string[]>) =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_respond",
				ask_id: "ask-1",
				answers,
			});

		it("renders after the options on an option question, with no field until pressed", () => {
			renderCard(ask());
			const buttons = screen.getAllByRole("radio").map((b) => b.textContent);
			expect(buttons.indexOf("no")).toBeLessThan(buttons.findIndex((t) => /^Other/.test(t ?? "")));
			expect(otherRow().getAttribute("aria-checked")).toBe("false");
			expect(screen.queryByRole("textbox")).toBeNull();
		});

		it("sends the typed text, not an index or a label", async () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			expect(otherRow().getAttribute("aria-checked")).toBe("true");
			typeOther("  typed answer ");
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["typed answer"] }));
		});

		it("takes focus only from the press, and re-focuses on a second press instead of deselecting", () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			const field = screen.getByRole("textbox", { name: "ship the fix?" });
			expect(document.activeElement).toBe(field);
			(field as HTMLInputElement).blur();
			fireEvent.click(otherRow());
			expect(otherRow().getAttribute("aria-checked")).toBe("true");
			expect(document.activeElement).toBe(field);
		});

		it("caps the field at the wire's 32768 characters", () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			expect(screen.getByRole("textbox", { name: "ship the fix?" }).getAttribute("maxlength")).toBe(
				"32768",
			);
		});

		it("is exclusive with the options in single-select", async () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			typeOther("typed answer");
			fireEvent.click(screen.getByRole("radio", { name: /^yes$/ }));
			expect(otherRow().getAttribute("aria-checked")).toBe("false");
			expect(screen.queryByRole("textbox")).toBeNull();
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["yes"] }));
		});

		it("is additive and last in multi-select, and unticks when pressed again", async () => {
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(otherRow(true));
			typeOther("something else");
			fireEvent.click(screen.getByRole("checkbox", { name: /^yes$/ }));
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["yes", "something else"] }));

			cleanup();
			vi.clearAllMocks();
			localStorage.clear();
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(screen.getByRole("checkbox", { name: /^yes$/ }));
			fireEvent.click(otherRow(true));
			typeOther("something else");
			fireEvent.click(otherRow(true));
			expect(otherRow(true).getAttribute("aria-checked")).toBe("false");
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["yes"] }));
		});

		it("does not read a label typed into the field as a tick", async () => {
			/* Typing `no, thanks` passes through `no`, an option's label: the cell then
			   holds that string, which must not be read back as a tick of the option. */
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(otherRow(true));
			typeOther("no");
			typeOther("no, thanks");
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["no, thanks"] }));
		});

		it("keeps the question incomplete while Other is open and empty", () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			expect(send().disabled).toBe(true);
			typeOther("   ");
			expect(send().disabled).toBe(true);
			typeOther("now it is an answer");
			expect(send().disabled).toBe(false);
		});

		it("does not grey send on a multi-select that already has a tick, and drops the blank marker", async () => {
			/* Deliberately not the desktop's rule: here a tick IS a complete answer. */
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(screen.getByRole("checkbox", { name: /^yes$/ }));
			fireEvent.click(otherRow(true));
			expect(otherRow(true).getAttribute("aria-checked")).toBe("true");
			expect(send().disabled).toBe(false);
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["yes"] }));
		});

		it("still holds a multi-select open when the open Other is its only entry", () => {
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(otherRow(true));
			expect(send().disabled).toBe(true);
		});

		it("puts the 'choose any' hint above the option rows, not after the Other field", () => {
			renderCard(ask({ questions: [question({ multi: true })] }));
			const hint = screen.getByText("Choose any that apply");
			const first = screen.getByRole("checkbox", { name: /^yes$/ });
			expect(
				hint.compareDocumentPosition(first) & Node.DOCUMENT_POSITION_FOLLOWING,
			).toBeTruthy();
		});

		it("dismisses the keyboard on Enter without submitting, and leaves an IME's Enter alone", () => {
			renderCard(ask());
			fireEvent.click(otherRow());
			const field = screen.getByRole("textbox", { name: "ship the fix?" }) as HTMLInputElement;
			typeOther("half a thought");
			expect(document.activeElement).toBe(field);

			fireEvent.keyDown(field, { key: "Enter", isComposing: true });
			expect(document.activeElement).toBe(field);

			/* Safari confirms a composition with `isComposing` already false and the
			   legacy `keyCode` 229 as the only signal. */
			fireEvent.keyDown(field, { key: "Enter", keyCode: 229 });
			expect(document.activeElement).toBe(field);

			fireEvent.keyDown(field, { key: "Enter" });
			expect(document.activeElement).not.toBe(field);
			expect(sendCommand).not.toHaveBeenCalled();
			expect(field.value).toBe("half a thought");
		});

		it("survives an unmount open and empty, without taking focus, and still sends no blank", async () => {
			const first = renderCard(ask());
			fireEvent.click(otherRow());
			first.unmount();

			renderCard(ask());
			expect(otherRow().getAttribute("aria-checked")).toBe("true");
			expect(document.activeElement).not.toBe(screen.getByRole("textbox"));
			expect(send().disabled).toBe(true);
			typeOther("back again");
			cleanup();
			renderCard(ask());
			expect((screen.getByRole("textbox") as HTMLInputElement).value).toBe("back again");
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["back again"] }));
		});

		it("never lets a blank or duplicate value reach the wire", async () => {
			/* A draft written by something else (or an older build): the submit still
			   trims, drops blanks and keeps the first of a duplicate. */
			localStorage.setItem(
				"lo-mobile-draft:ask:ask-1",
				JSON.stringify({ answers: { q1: ["yes", "  ", "yes", " typed "] }, skipped: [] }),
			);
			renderCard(ask({ questions: [question({ multi: true })] }));
			fireEvent.click(send());
			await waitFor(() => sentAnswers({ q1: ["yes", "typed"] }));
			for (const [, body] of vi.mocked(sendCommand).mock.calls) {
				const answers = (body as { answers: Record<string, string[]> }).answers;
				expect(Object.values(answers).flat()).not.toContain("");
			}
		});

		it("is withheld on a secret question and on a free-text one", () => {
			renderCard(ask({ questions: [question({ options: [], secret: true })] }));
			expect(screen.getByPlaceholderText("Secret value").getAttribute("type")).toBe("password");
			expect(screen.queryByRole("button", { name: /^Other/ })).toBeNull();
			cleanup();
			renderCard(ask({ questions: [question({ options: [] })] }));
			expect(screen.queryByRole("button", { name: /^Other/ })).toBeNull();
			expect(screen.getByPlaceholderText("Your answer")).toBeTruthy();
		});

		it("is withheld on a secret question that has options", () => {
			renderCard(ask({ questions: [question({ secret: true })] }));
			expect(screen.getByRole("radio", { name: /^yes$/ })).toBeTruthy();
			expect(screen.queryByRole("button", { name: /^Other/ })).toBeNull();
		});

		it("receipts a mixed multi answer with ' · ' between the option and the typed text", () => {
			renderCard(
				ask({
					status: "answered",
					delivered: true,
					questions: [
						question({
							multi: true,
							options: [
								{ label: "yes", description: "ship it" },
								{ label: "no", description: "" },
							],
						}),
					],
					answers: { q1: ["yes", "typed, with a comma"] },
				}),
			);
			expect(screen.getByText("yes — ship it · typed, with a comma")).toBeTruthy();
		});

		it("is disabled while the answer is in flight", async () => {
			vi.mocked(sendCommand).mockImplementationOnce(() => new Promise(() => {}));
			renderCard(ask());
			fireEvent.click(otherRow());
			typeOther("slow");
			fireEvent.click(send());
			await waitFor(() => expect((otherRow() as HTMLButtonElement).disabled).toBe(true));
		});
	});

	it("marks the recommended option from the wire's own index", () => {
		renderCard(ask({ questions: [question({ recommended: 0 })] }));
		expect(screen.getByText(/· Recommended/)).toBeTruthy();
	});

	it("badges every row of a multi-select's recommendation set, and ticks none", () => {
		renderCard(
			ask({
				questions: [
					question({
						multi: true,
						options: [
							{ label: "yes", description: "" },
							{ label: "maybe", description: "" },
							{ label: "no", description: "" },
						],
						recommended: 0,
						recommended_all: [0, 1],
					}),
				],
			}),
		);
		expect(screen.getAllByText(/· Recommended/)).toHaveLength(2);
		for (const name of [/yes/, /maybe/, /no/]) {
			expect(screen.getByRole("checkbox", { name }).getAttribute("aria-checked")).toBe("false");
		}
	});

	it("badges one row when a multi-select payload carries only `recommended: 0`", () => {
		/* An older runtime, or a stored row from before the set existed. */
		renderCard(ask({ questions: [question({ multi: true, recommended: 0 })] }));
		expect(screen.getAllByText(/· Recommended/)).toHaveLength(1);
	});

	it("badges nothing for a single-select with no recommendation", () => {
		renderCard(ask({ questions: [question({ recommended: null })] }));
		expect(screen.queryByText(/· Recommended/)).toBeNull();
	});

	it("says why an ask with nothing left to answer cannot be sent", () => {
		/* R6 (agent review round 1): `draft_question_ids` had every question, so
		   the card rendered no fields and a disabled send with no explanation. */
		renderCard(ask({ draft_question_ids: ["q1"] }));
		expect(screen.getByText(/Nothing left to answer here/)).toBeTruthy();
		expect(
			(screen.getByRole("button", { name: /Send answer/ }) as HTMLButtonElement).disabled,
		).toBe(true);
	});

	it("degrades a settled row with no questions to a plain line instead of throwing", () => {
		/* R5 (agent review round 1): `row.questions[0]` on a row that arrived
		   without the list threw during render, which unmounts the sheet. */
		renderCard({ ...ask({ status: "answered" }), questions: undefined } as unknown as PendingAsk);
		expect(screen.getByTestId("ask-card")).toBeTruthy();
	});

	it("states a dismissed row's questions with no answer slot and no description list", () => {
		/* UX round 2, U5 on round 1's U3: the reason line is the state, the
		   question is the receipt, and there is nothing to pair it with — so no
		   `—` (U3) and no `<dl>` whose `<dt>`s describe nothing (U5). */
		renderCard(
			ask({
				status: "dismissed",
				delivered: true,
				dismissed_at: Date.now(),
				dismissed_reason: "moot",
			}),
		);
		expect(screen.getByText("Dismissed — no longer needed")).toBeTruthy();
		expect(screen.getByText("ship the fix?")).toBeTruthy();
		expect(screen.queryByText("—")).toBeNull();
		expect(document.querySelector("dl")).toBeNull();
		expect(document.querySelector("dt")).toBeNull();
	});

	it("shows the queue's own refusal sentence verbatim, and restores the controls", async () => {
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("already answered by desktop."),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(screen.getByText("already answered by desktop.")).toBeTruthy());
		/* The refusal is a state the user can act on, not a dead card. */
		expect(
			(screen.getByRole("button", { name: /Send answer/ }) as HTMLButtonElement).disabled,
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
		renderCard(ask({ runtime_live: false }));
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		expect(await screen.findByText(/30 s/)).toBeTruthy();
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
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		/* The label STAYS while the answer is in flight (the button dims and says it
		   is busy), where it used to collapse to a bare `…` (ask-panel parity). */
		await waitFor(() =>
			expect(
				screen.getByRole("button", { name: /Send answer/ }).getAttribute("aria-busy"),
			).toBe("true"),
		);
		expect((screen.getByRole("button", { name: /Send answer/ }) as HTMLButtonElement).disabled).toBe(true);
		expect(screen.queryByText(/30 s/)).toBeNull();
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
		renderCard(ask({ status: "timed_out", runtime_live: false }));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		await waitFor(() => expect(screen.getByText(/nothing was sent/)).toBeTruthy());
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^Decline$/ })).toBeNull();
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
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(screen.getByText("already answered by desktop.")).toBeTruthy());
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
	});

	it("places a dismiss refusal beside the dismiss control", async () => {
		/* D5: the sentence used to render above the send/decline row, which put it
		   60 px from the button that produced it with the other pair in between. */
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error("only a timed-out ask can be dismissed; it is still open."),
		);
		renderCard(ask({ status: "timed_out" }));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		const sentence = await screen.findByText(/only a timed-out ask/);
		const dismiss = screen.getByRole("button", { name: /^Dismiss$/ });
		expect(
			dismiss.compareDocumentPosition(sentence) & Node.DOCUMENT_POSITION_FOLLOWING,
		).toBeTruthy();
	});

	it("keeps the dismiss control's width while the dismissal is in flight", async () => {
		/* D1 (design round 1): the in-flight mark used to be the bare glyph `…`,
		   which collapsed the control from 208 px to a 44 px square under the
		   thumb. The label IS the width, so keeping it — dimmed and disabled by
		   the `busy` state — is the whole fix, and the assertion is that the
		   in-flight name is the resting one. Send and Decline keep their labels too. */
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise<never>(() => {}),
		);
		renderCard(ask({ status: "timed_out", runtime_live: false }));
		const resting = screen.getByRole("button", { name: "Dismiss" });
		fireEvent.click(resting);
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		const inFlight = screen.getByRole("button", { name: "Dismiss" });
		// The same node, so the same box: nothing remounted and no label changed.
		expect(inFlight).toBe(resting);
		expect((inFlight as HTMLButtonElement).disabled).toBe(true);
		expect(inFlight.getAttribute("aria-busy")).toBe("true");
	});

	it("promises no engage for a COLD dismissal, before or during the tap", async () => {
		/* FORK A (2026-10-16): the relay used to ENGAGE the conversation to RECORD the
		   dismissal, so the card said so in advance (U3) and named the ~30 s while it
		   ran. A dismissal now takes the no-boot door on every surface, so "dismiss —
		   send no reply" is the WHOLE truth warm or cold, and the card must claim no
		   bring-up at either moment. */
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise<never>(() => {}),
		);
		renderCard(ask({ status: "timed_out", runtime_live: false }));
		expect(
			screen.getByRole("button", { name: /^Dismiss$/ }),
		).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss$/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		expect(screen.queryByText(/brought up to record it/)).toBeNull();
		expect(screen.queryByText(/Bringing the conversation up/)).toBeNull();
		expect(screen.queryByText(/~30 s/)).toBeNull();
	});

	it("stops claiming an engage the moment the outcome is known", async () => {
		/* UX round 2, U5: `setBusy("")` lived only in the catch, so a SUCCESSFUL
		   answer left the card asserting `bringing the session up — this can take up
		   to ~30 s` over a dead `…` until the sheet's re-read landed — and if that
		   read failed or hung, indefinitely. */
		renderCard(ask({ runtime_live: false }));
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		await waitFor(() => expect(screen.queryByText(/30 s/)).toBeNull());
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
	});

	it("states a dead conversation before the tap, and withholds every control", () => {
		/* Design round 2, D6 = UX U7: a dead conversation — no durable transcript
		   AND nothing live to deliver to — is the state the route refuses on, so
		   the sentence a 409 would have returned is known in advance. Showing it
		   pre-emptively keeps the guarantee without buying the refused request, and
		   no control invites a tap that cannot work. */
		renderCard(ask({ runtime_live: false, durable: false }));
		expect(
			screen.getByText(/This conversation no longer exists, so this question can't be read\. Start a new conversation and ask again\./),
		).toBeTruthy();
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^Decline$/ })).toBeNull();
		expect(screen.queryByRole("status")).toBeNull();
	});

	it("keeps the controls on a LIVE conversation that has no transcript", () => {
		/* Design round 3, D7: `durable: false` alone is NOT a dead conversation.
		   The route's cold arm is scoped `not _session_is_live(...)`, so a live row
		   never reaches the refusal — and gating the card on durability alone made
		   it refuse an answer the server returned `200 {"ok": true}` for. The
		   mirror of the bug this lane was opened to fix, pinned before it can
		   come back. */
		renderCard(ask({ runtime_live: true, durable: false }));
		expect(screen.queryByText(/This conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
		expect(screen.getByRole("button", { name: /^Decline$/ })).toBeTruthy();
		/* And no wait is promised either: the relay can deliver right now. */
		expect(screen.queryByText(/30 s/)).toBeNull();
	});

	it("keeps a settled receipt in a dead conversation", () => {
		/* The other side of the same gate: `durable: false` must not turn a receipt
		   into a dead end — a row already answered in a conversation that has since
		   gone still has its own record to show. */
		renderCard(ask({ status: "answered", delivered: true, runtime_live: false, durable: false }));
		expect(screen.queryByText(/This conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("status")).toBeTruthy();
	});
});

describe("a long question stays bounded", () => {
	/* The operator's screenshot: a proposal ask carrying a whole team brief
	   filled the card and pushed its options off screen. The question text is
	   its own capped scroller, and the answer controls are NOT inside it, so
	   they stay directly under a visible excerpt whatever the length. */
	it("caps the question text in column units and keeps options outside the scroller", () => {
		const long = Array.from({ length: 400 }, (_, i) => `line ${i}: brief text`).join("\n");
		renderCard(ask({ questions: [question({ question: long })] }));
		const text = screen.getByTestId("ask-question-text");
		expect(text.style.maxHeight).toBe("calc(var(--lo-vvh, 100dvh) * 0.3)");
		expect(text.className).toContain("overflow-y-auto");
		expect(text.className).toContain("whitespace-pre-line");
		const yes = screen.getByRole("radio", { name: /yes/ });
		expect(text.contains(yes)).toBe(false);
	});
});

describe("a multi-question card keeps its controls out of the scrolling body", () => {
	/* PR #3777 code review M2 capped the question stack in a scroller of its own so
	   the shared send/decline row did not fall off screen. The sheet now PINS those
	   controls (ask-panel parity), which is the stronger form of the same rule:
	   content may scroll, the control that answers the card may not. */
	const long = (n: number) =>
		Array.from({ length: 300 }, (_, i) => `q${n} line ${i}: brief text`).join("\n");
	const three = () =>
		ask({
			questions: [1, 2, 3].map((n) => question({ id: `q${n}`, question: long(n) })),
		});

	it("draws send, decline and dismiss in the footer slot it is given, not in the card", () => {
		const bar = document.createElement("div");
		document.body.appendChild(bar);
		try {
			render(
				<AskCard row={three()} sessionId="s1" nowMs={Date.now()} footerSlot={bar} />,
			);
			const card = screen.getByTestId("ask-card");
			for (const name of [/Send answers/, /^Decline$/, /^Dismiss…$/]) {
				const control = screen.getByRole("button", { name });
				expect(bar.contains(control)).toBe(true);
				expect(card.contains(control)).toBe(false);
			}
			// Every question is in the card, each capped by its own scroller, with no
			// second scroller around the stack.
			expect(within(card).getAllByTestId("ask-question-text")).toHaveLength(3);
			expect(screen.queryByTestId("ask-question-list")).toBeNull();
		} finally {
			cleanup();
			bar.remove();
		}
	});

	it("draws nothing for a bar that has not mounted yet, rather than a duplicate", () => {
		renderCard(ask());
		cleanup();
		render(<AskCard row={ask()} sessionId="s1" nowMs={Date.now()} footerSlot={null} />);
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.getByRole("radio", { name: /yes/ })).toBeTruthy();
	});

	it("draws them at the card's own foot when it has no slot", () => {
		renderCard(three());
		const card = screen.getByTestId("ask-card");
		expect(card.contains(screen.getByRole("button", { name: /Send answers/ }))).toBe(true);
	});

	it("leaves a single question unwrapped so its options sit under the excerpt", () => {
		renderCard(ask({ questions: [question({ question: "short" })] }));
		expect(screen.queryByTestId("ask-question-list")).toBeNull();
	});
});

describe("the option rows", () => {
	beforeEach(() => localStorage.clear());

	it("are a radio group for one choice and a checkbox group for several, named by the question", () => {
		const { unmount } = renderCard(ask());
		expect(screen.getByRole("radiogroup", { name: "ship the fix?" })).toBeTruthy();
		expect(screen.getAllByRole("radio").map((node) => node.getAttribute("aria-checked"))).toEqual([
			"false",
			"false",
			"false", // the Other row is the group's last radio
		]);
		unmount();
		renderCard(ask({ questions: [question({ multi: true })] }));
		expect(screen.getByRole("group", { name: "ship the fix?" })).toBeTruthy();
		expect(screen.getAllByRole("checkbox")).toHaveLength(3);
	});

	it("say a row is chosen with a tick and a thicker edge, not colour alone", () => {
		renderCard(ask());
		const yes = screen.getByRole("radio", { name: /^yes$/ });
		const no = screen.getByRole("radio", { name: /^no$/ });
		// Rest: no tick, the 2 px edge on a tinted fill.
		expect(yes.textContent).not.toContain("✓");
		expect(yes.className).toContain("border-s-2");
		expect(yes.className).toContain("border-s-accent");
		expect(yes.className).toContain("bg-accent-wash");
		fireEvent.click(yes);
		// Selected: a tick and the 4 px edge on the same tint; the sibling is unchanged.
		const picked = screen.getByRole("radio", { name: /yes/ });
		expect(picked.getAttribute("aria-checked")).toBe("true");
		expect(picked.textContent).toContain("✓");
		expect(picked.className).toContain("border-s-4");
		expect(picked.className).toContain("bg-accent-wash");
		expect(no.textContent).not.toContain("✓");
		expect(no.className).toContain("border-s-2");
		// The tick is decorative: `aria-checked` already carries the state.
		expect(picked.querySelector("[aria-hidden]")?.textContent).toBe("✓");
	});

	it("keep a chosen row's look while the answer is in flight and dim the others", async () => {
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise<never>(() => {}),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("radio", { name: /^yes$/ }));
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		const picked = screen.getByRole("radio", { name: /yes/ });
		expect((picked as HTMLButtonElement).disabled).toBe(true);
		expect(picked.className).toContain("bg-accent-wash");
		expect(picked.className).not.toContain("bg-sunken");
		const dimmed = screen.getByRole("radio", { name: /^no$/ });
		expect(dimmed.className).toContain("bg-sunken");
		expect(dimmed.className).toContain("border-s-hairline");
		expect(dimmed.querySelector(".text-ink-disabled")).toBeTruthy();
	});

	it("keep every target at least 44 px tall", () => {
		renderCard(ask());
		for (const row of screen.getAllByRole("radio")) expect(row.className).toContain("min-h-11");
	});

	it("show the Other row as `Other   Your answer` on one line until it is opened", () => {
		renderCard(ask());
		const other = screen.getByRole("radio", { name: /^Other/ });
		expect(other.textContent).toBe("OtherYour answer");
		fireEvent.click(other);
		expect(screen.getByRole("radio", { name: /^Other/ }).textContent).toBe("Other✓");
	});
});

describe("the answering bar", () => {
	beforeEach(() => localStorage.clear());
	const two = () =>
		ask({ questions: [question(), question({ id: "q2", question: "and the date?" })] });

	it("says how far along a several-question form is, and what a single question takes", () => {
		const { unmount } = renderCard(two());
		expect(screen.getByText("0 of 2 questions answered or skipped")).toBeTruthy();
		fireEvent.click(screen.getAllByRole("radio", { name: /^yes$/ })[0]);
		expect(screen.getByText("1 of 2 questions answered or skipped")).toBeTruthy();
		unmount();
		localStorage.clear(); // the pick above is kept under this ask's id

		const withOptions = renderCard(ask());
		expect(screen.getByText("Pick an option or type your own answer to send.")).toBeTruthy();
		withOptions.unmount();

		renderCard(ask({ questions: [question({ options: [] })] }));
		expect(screen.getByText("Type an answer or skip the question to send.")).toBeTruthy();
	});

	it("counts a skipped question as answered, and drops the line once send is live", () => {
		renderCard(two());
		fireEvent.click(screen.getAllByRole("radio", { name: /^yes$/ })[0]);
		fireEvent.click(screen.getAllByRole("button", { name: /Skip \(send no answer\)/ })[1]);
		expect(screen.queryByText(/answered$/)).toBeNull();
		expect((screen.getByRole("button", { name: /Send answers/ }) as HTMLButtonElement).disabled).toBe(false);
	});

	it("says nothing about progress while an answer is in flight", async () => {
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise<never>(() => {}),
		);
		renderCard(ask({ questions: [question({ options: [] })] }));
		fireEvent.change(screen.getByRole("textbox"), { target: { value: "ok" } });
		expect(screen.queryByText("Type an answer or skip the question to send.")).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		expect(screen.queryByText(/answered$/)).toBeNull();
	});

	it("puts Send and Decline side by side at half width each, with a quiet Dismiss… under them", () => {
		renderCard(ask());
		const send = screen.getByRole("button", { name: /Send answer/ });
		const decline = screen.getByRole("button", { name: /^Decline$/ });
		expect(send.className).toContain("flex-1");
		expect(decline.className).toContain("flex-1");
		expect(send.parentElement).toBe(decline.parentElement);
		const dismiss = screen.getByRole("button", { name: /^Dismiss…$/ });
		expect(dismiss.parentElement).not.toBe(send.parentElement);
		expect(send.parentElement?.compareDocumentPosition(dismiss)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
	});

	it("swaps the pair and the progress line for the two reasons, and back on Cancel", () => {
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(screen.getByRole("button", { name: /^No longer needed/ })).toBeTruthy();
		expect(screen.getByRole("button", { name: /^Answered in the conversation/ })).toBeTruthy();
		// Nothing a hurried thumb could mistake for "send my answer" stays up.
		expect(screen.queryByRole("button", { name: /Send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^Decline$/ })).toBeNull();
		expect(screen.queryByText(/Pick an option/)).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: /^Cancel$/ }));
		expect(screen.getByRole("button", { name: /Send answer/ })).toBeTruthy();
		expect(sendCommand).not.toHaveBeenCalled();
	});

	it("keeps the refusal line in the bar, above the pair", async () => {
		(sendCommand as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new Error("already answered by desktop."));
		const bar = document.createElement("div");
		document.body.appendChild(bar);
		try {
			render(<AskCard row={ask()} sessionId="s1" nowMs={Date.now()} footerSlot={bar} />);
			fireEvent.click(screen.getByRole("radio", { name: /^yes$/ }));
			fireEvent.click(screen.getByRole("button", { name: /Send answer/ }));
			const line = await screen.findByText("already answered by desktop.");
			expect(bar.contains(line)).toBe(true);
			expect(
				line.compareDocumentPosition(screen.getByRole("button", { name: /Send answer/ })) &
					Node.DOCUMENT_POSITION_FOLLOWING,
			).toBeTruthy();
		} finally {
			cleanup();
			bar.remove();
		}
	});
});

describe("the card's own lines", () => {
	it("draws the lead under the state line, before the question", () => {
		renderCard(ask());
		cleanup();
		render(
			<AskCard
				row={ask()}
				sessionId="s1"
				nowMs={Date.now()}
				lead={<p data-testid="lead">tax receipts</p>}
			/>,
		);
		const status = screen.getByRole("status");
		const lead = screen.getByTestId("lead");
		const text = screen.getByTestId("ask-question-text");
		expect(status.compareDocumentPosition(lead) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(lead.compareDocumentPosition(text) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it("states its state line first, and leaves it out under a row that already says it", () => {
		const { unmount } = renderCard(ask());
		expect(screen.getByRole("status").textContent).toMatch(/^Queued — the agent is continuing; expires in /);
		unmount();
		render(
			<AskCard
				row={ask({ status: "answered", delivered: true, answers: { q1: ["yes"] } })}
				sessionId="s1"
				nowMs={Date.now()}
				hideState
			/>,
		);
		expect(screen.queryByRole("status")).toBeNull();
		expect(screen.getByText("yes")).toBeTruthy();
	});

	it("moves focus to the first question once, only when asked to", () => {
		const focused = vi.fn();
		render(<AskCard row={ask()} sessionId="s1" nowMs={Date.now()} onFocused={focused} />);
		expect(document.activeElement).not.toBe(screen.getByTestId("ask-question-text"));
		expect(focused).not.toHaveBeenCalled();
		cleanup();
		const next = render(
			<AskCard row={ask()} sessionId="s1" nowMs={Date.now()} focusQuestion onFocused={focused} />,
		);
		expect(document.activeElement).toBe(screen.getByTestId("ask-question-text"));
		expect(focused).toHaveBeenCalledTimes(1);
		// A re-render does not take focus again.
		(document.activeElement as HTMLElement).blur();
		next.rerender(
			<AskCard row={ask()} sessionId="s1" nowMs={Date.now() + 1} focusQuestion onFocused={focused} />,
		);
		expect(document.activeElement).not.toBe(screen.getByTestId("ask-question-text"));
	});
});

describe("a secret question's answer is never persisted", () => {
	/* The draft cell lives in localStorage, in plaintext, across reloads. A secret
	   is a credential the agent asked for by name, so what the reader types is kept
	   in the card's memory only, as on the native app. */
	const SECRET = "SENTINEL-secret-123";
	const KEY = "lo-mobile-draft:ask:ask-1";
	const stored = () => localStorage.getItem(KEY) ?? "";
	const mixed = () =>
		ask({
			questions: [
				question({ id: "q1", question: "token?", options: [], secret: true }),
				question({ id: "q2", question: "ship it?" }),
			],
		});
	beforeEach(() => localStorage.clear());

	it("keeps a typed secret out of storage while the card still uses it, and keeps other answers", async () => {
		renderCard(mixed());
		fireEvent.change(screen.getByPlaceholderText("Secret value"), { target: { value: SECRET } });
		// Alone, a secret leaves no cell at all.
		expect(localStorage.getItem(KEY)).toBeNull();
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		expect(stored()).not.toContain(SECRET);
		expect(JSON.parse(stored()).answers).toEqual({ q2: ["yes"] });
		// The card itself still sends what was typed.
		fireEvent.click(screen.getByRole("button", { name: /Send answers/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_respond",
				ask_id: "ask-1",
				answers: { q1: [SECRET], q2: ["yes"] },
			}),
		);
	});

	it("does not bring a secret back after the card is closed and reopened", () => {
		const first = renderCard(mixed());
		fireEvent.change(screen.getByPlaceholderText("Secret value"), { target: { value: SECRET } });
		fireEvent.click(screen.getByRole("radio", { name: /yes/ }));
		first.unmount();
		renderCard(mixed());
		expect((screen.getByPlaceholderText("Secret value") as HTMLInputElement).value).toBe("");
		expect(screen.getByRole("radio", { name: /yes/ }).getAttribute("aria-checked")).toBe("true");
	});

	it("strips a secret an older build already stored, and rewrites the cell", () => {
		localStorage.setItem(
			KEY,
			JSON.stringify({ answers: { q1: [SECRET], q2: ["yes"] }, skipped: ["q3"] }),
		);
		renderCard(mixed());
		expect((screen.getByPlaceholderText("Secret value") as HTMLInputElement).value).toBe("");
		expect(stored()).not.toContain(SECRET);
		expect(JSON.parse(stored())).toEqual({ answers: { q2: ["yes"] }, skipped: ["q3"] });
	});

	it("removes the cell outright when the stored draft held nothing but a secret", () => {
		localStorage.setItem(KEY, JSON.stringify({ answers: { q1: [SECRET] }, skipped: [] }));
		renderCard(mixed());
		expect(localStorage.getItem(KEY)).toBeNull();
	});

	it("still persists a non-secret free-text answer", () => {
		renderCard(ask({ questions: [question({ options: [] })] }));
		fireEvent.change(screen.getByRole("textbox"), { target: { value: "plain text" } });
		expect(JSON.parse(stored()).answers).toEqual({ q1: ["plain text"] });
	});
});

describe("the panel's final look (native parity)", () => {
	beforeEach(() => localStorage.clear());

	it("draws an empty ring on a resting single-choice row and fills it to a disc with a check when chosen", () => {
		renderCard(ask());
		const yes = screen.getByRole("radio", { name: /^yes$/ });
		const ring = yes.querySelector("[data-ask-indicator]") as HTMLElement;
		expect(ring.getAttribute("data-ask-indicator")).toBe("off");
		expect(ring.textContent).toBe("");
		expect(ring.className).toContain("rounded-full");
		expect(ring.className).toContain("border-ink-dim");
		expect(ring.getAttribute("aria-hidden")).toBe("true");
		expect(yes.querySelector(".font-semibold")).toBeNull();
		fireEvent.click(yes);
		const disc = screen.getByRole("radio", { name: /yes/ }).querySelector("[data-ask-indicator]") as HTMLElement;
		expect(disc.getAttribute("data-ask-indicator")).toBe("on");
		expect(disc.textContent).toBe("✓");
		expect(disc.className).toContain("bg-accent");
		expect(disc.className).toContain("text-on-accent");
		// The chosen label is semibold: weight, as well as shape and colour.
		expect(screen.getByRole("radio", { name: /yes/ }).querySelector(".font-semibold")).toBeTruthy();
	});

	it("draws a square, not a ring, on a multi-select row", () => {
		renderCard(ask({ questions: [question({ multi: true })] }));
		const box = screen.getByRole("checkbox", { name: /^yes$/ }).querySelector("[data-ask-indicator]") as HTMLElement;
		expect(box.className).toContain("rounded-xs");
		expect(box.className).not.toContain("rounded-full");
	});

	it("says a closed Other row still holds the reader's text, without sending it", () => {
		renderCard(ask());
		fireEvent.click(screen.getByRole("radio", { name: /^Other/ }));
		fireEvent.change(screen.getByRole("textbox"), { target: { value: "my own words" } });
		fireEvent.click(screen.getByRole("radio", { name: /^yes$/ }));
		const other = screen.getByRole("radio", { name: /^Other/ });
		expect(other.textContent).toContain("Your text is kept, not sent");
		expect(other.textContent).not.toContain("Your answer");
	});

	it("names the dismissal for one question or several", () => {
		const { unmount } = renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(screen.getByText("Dismiss this question")).toBeTruthy();
		unmount();
		renderCard(ask({ questions: [question(), question({ id: "q2", question: "and the date?" })] }));
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(screen.getByText("Dismiss these questions")).toBeTruthy();
	});

	it("gives Send and Decline one height side by side, wrapping to a row each only when their labels cannot fit", () => {
		renderCard(ask());
		const send = screen.getByRole("button", { name: /Send answer/ });
		const decline = screen.getByRole("button", { name: /^Decline$/ });
		for (const button of [send, decline]) {
			for (const cls of ["flex-1", "basis-0", "min-w-fit", "whitespace-nowrap"]) {
				expect(button.className).toContain(cls);
			}
		}
		expect(send.parentElement?.className).toContain("flex-wrap");
	});

	it("shares one row between the progress line and a quiet, regular-weight Dismiss…", () => {
		renderCard(ask());
		const dismiss = screen.getByRole("button", { name: /^Dismiss…$/ });
		const progress = screen.getByText("Pick an option or type your own answer to send.");
		expect(progress.parentElement).toBe(dismiss.parentElement);
		// Wrapping lets Dismiss… drop to its own row when the sentence fills this one.
		expect(progress.parentElement?.className).toContain("flex-wrap");
		expect(progress.className).toContain("flex-auto");
		expect(dismiss.className).toContain("text-ink-muted");
		expect(dismiss.className).not.toContain("font-medium");
		expect(dismiss.className.split(" ")).not.toContain("bg-accent");
		expect(dismiss.className).not.toContain("border");
	});

	it("keeps Dismiss… out of the way while the on-screen keyboard is up, and the progress line in", () => {
		const real = Object.getOwnPropertyDescriptor(window, "visualViewport");
		const listeners: Array<() => void> = [];
		const vv = {
			height: window.innerHeight - 300,
			addEventListener: (_: string, fn: () => void) => listeners.push(fn),
			removeEventListener: () => {},
		};
		Object.defineProperty(window, "visualViewport", { value: vv, configurable: true });
		try {
			renderCard(ask());
			expect(screen.queryByRole("button", { name: /^Dismiss…$/ })).toBeNull();
			expect(screen.getByText("Pick an option or type your own answer to send.")).toBeTruthy();
			// The keyboard goes away: the link comes back.
			vv.height = window.innerHeight;
			act(() => listeners.forEach((fn) => fn()));
			expect(screen.getByRole("button", { name: /^Dismiss…$/ })).toBeTruthy();
		} finally {
			if (real) Object.defineProperty(window, "visualViewport", real);
			else delete (window as { visualViewport?: unknown }).visualViewport;
		}
	});

	it("keeps a lapsed ask's Dismiss a button", () => {
		renderCard(ask({ status: "timed_out" }));
		const dismiss = screen.getByRole("button", { name: /^Dismiss$/ });
		expect(dismiss.className).toContain("min-h-11");
		expect(screen.queryByRole("button", { name: /^Dismiss…$/ })).toBeNull();
	});

	it("draws the card with a hairline border and an urgent mark in warning ink", () => {
		renderCard(ask({ urgent: true }));
		expect(screen.getByTestId("ask-card").className).toContain("border-hairline");
		const urgent = screen.getByText("· Urgent");
		expect(urgent.className).toContain("text-warning");
		expect(urgent.className).toContain("font-medium");
		expect(screen.getByTestId("ask-question-text").className).toContain("text-start");
	});
});

describe("the bar after the remediation batch", () => {
	beforeEach(() => localStorage.clear());

	it("greys a disabled Send with a sunken fill and muted ink, not a faded accent", () => {
		renderCard(ask());
		const send = screen.getByRole("button", { name: /Send answer/ });
		expect((send as HTMLButtonElement).disabled).toBe(true);
		expect(send.className).toContain("disabled:bg-sunken");
		expect(send.className).toContain("disabled:text-ink-muted");
		expect(send.className).not.toContain("disabled:opacity-50");
	});

	it("gives Decline a danger-tinted edge", () => {
		renderCard(ask());
		const decline = screen.getByRole("button", { name: /^Decline$/ });
		expect(decline.className).toContain("border");
		expect(decline.className).toContain("border-danger/40");
	});

	it("moves focus to the first reason on Dismiss…, and back to Dismiss… on Cancel", () => {
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^Dismiss…$/ }));
		expect(document.activeElement).toBe(screen.getAllByRole("button").find((b) => /^(Answered in the conversation|No longer needed)/.test(b.textContent ?? "")));
		fireEvent.click(screen.getByRole("button", { name: /^Cancel$/ }));
		expect(document.activeElement).toBe(screen.getByRole("button", { name: /^Dismiss…$/ }));
	});

	it("centres Cancel in the reasons panel and drops Dismiss…'s start inset", () => {
		renderCard(ask());
		const dismiss = screen.getByRole("button", { name: /^Dismiss…$/ });
		expect(dismiss.className).toContain("ps-0");
		fireEvent.click(dismiss);
		const cancel = screen.getByRole("button", { name: /^Cancel$/ });
		expect(cancel.className).toContain("self-center");
		expect(cancel.className).not.toContain("self-start");
	});

	it("scrolls the Other field into view when it takes focus", () => {
		const scrolled = vi.fn();
		const proto = HTMLElement.prototype as { scrollIntoView?: unknown };
		const had = proto.scrollIntoView;
		proto.scrollIntoView = scrolled;
		try {
			renderCard(ask());
			fireEvent.click(screen.getByRole("radio", { name: /^Other/ }));
			fireEvent.focus(screen.getByRole("textbox"));
			expect(scrolled).toHaveBeenCalledWith({ block: "nearest" });
		} finally {
			if (had) proto.scrollIntoView = had;
			else delete proto.scrollIntoView;
		}
	});
});

