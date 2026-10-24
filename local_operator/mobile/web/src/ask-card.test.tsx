// @vitest-environment happy-dom
//
// The queued-ask card: the ATOMIC submit (one complete map, every question id
// present — the queue refuses a partial one), the deliberate skip (sent as an
// empty list, which is the contract's own spelling for "no answer"), the
// read-only lapsed state after a deadline, and the rule that a refusal is rendered in
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

/** Open a lapsed card's collapsed header — the explicit tap D11 requires before
 *  anything inside a timed-out ask (its Dismiss included) is reachable. */
function expandLapsed() {
	fireEvent.click(screen.getByRole("button", { name: /Timed out/ }));
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

	it("collapses a timed-out ask behind a Timed out badge until the user expands it", () => {
		/* Decision D11 (Ben, 2026-10-15): a lapsed ask reads the way a declined one
		   does on the desktop — one collapsed row with a status badge. Seeing it is
		   an explicit tap; nothing about it is answerable, so the collapsed header
		   offers no control but the disclosure itself. */
		renderCard(ask({ status: "timed_out" }));
		const header = screen.getByRole("button", { name: /Timed out/ });
		expect(header.getAttribute("aria-expanded")).toBe("false");
		expect(header.textContent).toContain("ship the fix?");
		expect(screen.queryByText(/the agent goes ahead without your answer/)).toBeNull();
		expect(screen.queryByRole("button", { name: /dismiss — send no reply/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^yes$/ })).toBeNull();
		/* Mounting it takes no focus: a lapsed ask never claims the user (D12). */
		expect(document.activeElement).not.toBe(header);
	});

	it("expands a timed-out ask to the read-only card: the question, no answer controls, Dismiss kept", () => {
		/* Ben's report (2026-10-15): the phone offered the options of an ask whose
		   deadline had passed, as if the agent were still waiting on them. The
		   agent has gone ahead without the answer, so the expanded card reads that
		   and offers only the one op the queue accepts on a timeout — dismiss. */
		renderCard(ask({ status: "timed_out" }));
		fireEvent.click(screen.getByRole("button", { name: /Timed out/ }));
		expect(
			screen.getByRole("button", { name: /Timed out/ }).getAttribute("aria-expanded"),
		).toBe("true");
		/* UX round 2, U13: the badge already says "Timed out" one line above, so
		   the body starts at what that means rather than repeating the word. */
		expect(screen.getByText("the agent goes ahead without your answer")).toBeTruthy();
		expect(screen.queryByText(/^Timed out —/)).toBeNull();
		expect(screen.queryByRole("button", { name: /^yes$/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^no$/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^decline$/ })).toBeNull();
		expect(screen.getByRole("button", { name: /dismiss — send no reply/ })).toBeTruthy();
		/* UX U7: the question is stated once. Expanded, the header carries only
		   the badge and the body states the question in full. */
		expect(screen.getAllByText("ship the fix?").length).toBe(1);
	});

	it("dismisses a timed-out ask from inside its expanded body", async () => {
		const settled = vi.fn();
		renderCard(ask({ status: "timed_out" }), settled);
		expandLapsed();
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
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
		expect(screen.getByRole("button", { name: /^yes$/ })).toBeTruthy();
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
		expect(screen.getByText("Queued — the agent is continuing; deadline passed")).toBeTruthy();
		expect(screen.queryByText(/^Timed out —/)).toBeNull();
		/* R5: an OPEN ask has a door of its own — the quiet `close…`, which asks
		   WHY before anything is sent. What it must NOT offer is the lapsed row's
		   one-tap dismissal, which says "send no reply" over an ask the agent is
		   still owed an answer to. */
		expect(screen.getByRole("button", { name: /^dismiss…$/ })).toBeTruthy();
		expect(screen.queryByRole("button", { name: /dismiss — send no reply/ })).toBeNull();
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
		/* THE ONE-TAP DISMISS IS THE LAPSED ROW'S (agent review round 1, R1, kept
		   by R5): the button says "send no reply", which is true only once the
		   agent has walked past the deadline. An OPEN ask's door is `close…` plus
		   a reason, exercised in its own cases below. */
		renderCard(ask({ status: "timed_out" }));
		expandLapsed();
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", { op: "ask_dismiss", ask_id: "ask-1" }),
		);
	});

	it("offers the one-tap dismissal on a timed-out ask and the reason row on an open one", () => {
		const { unmount } = renderCard(ask());
		expect(screen.queryByRole("button", { name: /dismiss — send no reply/ })).toBeNull();
		expect(screen.getByRole("button", { name: /^dismiss…$/ })).toBeTruthy();
		unmount();
		renderCard(ask({ status: "timed_out" }));
		expandLapsed();
		expect(screen.getByRole("button", { name: /dismiss — send no reply/ })).toBeTruthy();
		expect(screen.queryByRole("button", { name: /^dismiss…$/ })).toBeNull();
	});

	it("closes an open ask with a reason, and sends nothing until one is chosen", async () => {
		/* R5 (design §15's amended D6): an open ask can be put away from the phone.
		   The tap that REVEALS the reasons is not the dismissal — a reason the
		   operator did not choose is not one they gave — and the two reasons are
		   what the agent acts on (`moot` closes the question, `answered_in_chat`
		   tells it the answer is already in the conversation). */
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in chat/ }));
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^cancel$/ }));
		expect(screen.queryByRole("button", { name: /^No longer needed/ })).toBeNull();
		expect(screen.getByRole("button", { name: /^dismiss…$/ })).toBeTruthy();
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		expect(screen.queryByText(/bringing the session up/)).toBeNull();
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
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^No longer needed/ }));
		await screen.findByText("you already dismissed this.");
		fireEvent.click(screen.getByRole("button", { name: /^cancel$/ }));
		expect(screen.queryByText("you already dismissed this.")).toBeNull();
		expect(screen.getByRole("button", { name: /^dismiss…$/ })).toBeTruthy();
	});

	it("names the act, and states what each reason tells the agent", () => {
		/* Design round 1, D1 = UX round 1, U6: the reveal used to be three controls
		   on one line with no heading and no explanation, so "Answered in chat"
		   could be read as "send my answer" while sitting 8.5 px under `decline`.
		   The heading says "Dismiss", and each row carries its note. */
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		expect(screen.getByText("Dismiss this ask")).toBeTruthy();
		expect(screen.getByRole("button", { name: /^No longer needed/ }).textContent).toContain(
			"stop waiting on it",
		);
		expect(
			screen.getByRole("button", { name: /^Answered in chat/ }).textContent,
		).toContain("you already replied in the conversation");
	});

	it("keeps the pressed reason's label while the dismissal is in flight", async () => {
		/* UX round 1, U3 (design N3): both buttons used to become `…`, which erased
		   the only on-screen statement of which reason had been sent. */
		let release: (value: unknown) => void = () => {};
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise((resolve) => { release = resolve; }),
		);
		renderCard(ask());
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
		fireEvent.click(screen.getByRole("button", { name: /^Answered in chat/ }));
		await waitFor(() =>
			expect(screen.getByRole("button", { name: /^Answered in chat/ }).textContent).toContain(
				"Answered in chat",
			),
		);
		expect(
			screen.getByRole("button", { name: /^Answered in chat/ }).getAttribute("aria-busy"),
		).toBe("true");
		await act(async () => { release({ ok: true, detail: "dismissed" }); });
	});

	it("hands the reason to the sheet on an accepted dismissal", async () => {
		/* UX round 1, U2: the card leaves the aggregate, so the sheet needs the
		   operator's own word to show a receipt instead of its empty state. */
		const dismissed = vi.fn();
		renderCard(ask(), undefined, dismissed);
		fireEvent.click(screen.getByRole("button", { name: /^dismiss…$/ }));
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
		renderCard(ask({ runtime_live: false }));
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
		renderCard(ask({ status: "timed_out", runtime_live: false }));
		expandLapsed();
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
		expandLapsed();
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		const sentence = await screen.findByText(/only a timed-out ask/);
		const dismiss = screen.getByRole("button", { name: /dismiss — send no reply/ });
		expect(
			dismiss.compareDocumentPosition(sentence) & Node.DOCUMENT_POSITION_FOLLOWING,
		).toBeTruthy();
	});

	it("keeps the dismiss control's width while the dismissal is in flight", async () => {
		/* D1 (design round 1): the in-flight mark used to be the bare glyph `…`,
		   which collapsed the control from 208 px to a 44 px square under the
		   thumb. The label IS the width, so keeping it — dimmed and disabled by
		   the `busy` state — is the whole fix, and the assertion is that the
		   in-flight name is the resting one. The glyph stays on answer/decline,
		   where D1 called it optional. */
		(sendCommand as ReturnType<typeof vi.fn>).mockImplementationOnce(
			() => new Promise<never>(() => {}),
		);
		renderCard(ask({ status: "timed_out", runtime_live: false }));
		expandLapsed();
		const resting = screen.getByRole("button", { name: "dismiss — send no reply" });
		fireEvent.click(resting);
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		const inFlight = screen.getByRole("button", { name: "dismiss — send no reply" });
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
		expandLapsed();
		expect(
			screen.getByRole("button", { name: /dismiss — send no reply/ }),
		).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: /dismiss — send no reply/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		expect(screen.queryByText(/brought up to record it/)).toBeNull();
		expect(screen.queryByText(/bringing the session up/)).toBeNull();
		expect(screen.queryByText(/~30 s/)).toBeNull();
	});

	it("stops claiming an engage the moment the outcome is known", async () => {
		/* UX round 2, U5: `setBusy("")` lived only in the catch, so a SUCCESSFUL
		   answer left the card asserting `bringing the session up — this can take up
		   to ~30 s` over a dead `…` until the sheet's re-read landed — and if that
		   read failed or hung, indefinitely. */
		renderCard(ask({ runtime_live: false }));
		fireEvent.click(screen.getByRole("button", { name: /yes/ }));
		fireEvent.click(screen.getByRole("button", { name: /send answer/ }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		await waitFor(() => expect(screen.queryByText(/this can take up to ~30 s/)).toBeNull());
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
	});

	it("states a dead conversation before the tap, and withholds every control", () => {
		/* Design round 2, D6 = UX U7: a dead conversation — no durable transcript
		   AND nothing live to deliver to — is the state the route refuses on, so
		   the sentence a 409 would have returned is known in advance. Showing it
		   pre-emptively keeps the guarantee without buying the refused request, and
		   no control invites a tap that cannot work. */
		renderCard(ask({ runtime_live: false, durable: false }));
		expect(
			screen.getByText(/this conversation no longer exists — the ask can never be read; start a new conversation and ask again\./),
		).toBeTruthy();
		expect(screen.queryByRole("button", { name: /send answer/ })).toBeNull();
		expect(screen.queryByRole("button", { name: /^decline$/ })).toBeNull();
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
		expect(screen.queryByText(/this conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("button", { name: /send answer/ })).toBeTruthy();
		expect(screen.getByRole("button", { name: /^decline$/ })).toBeTruthy();
		/* And no wait is promised either: the relay can deliver right now. */
		expect(screen.queryByText(/this can take up to ~30 s/)).toBeNull();
	});

	it("keeps a settled receipt in a dead conversation", () => {
		/* The other side of the same gate: `durable: false` must not turn a receipt
		   into a dead end — a row already answered in a conversation that has since
		   gone still has its own record to show. */
		renderCard(ask({ status: "answered", delivered: true, runtime_live: false, durable: false }));
		expect(screen.queryByText(/this conversation no longer exists/)).toBeNull();
		expect(screen.getByRole("status")).toBeTruthy();
	});
});
