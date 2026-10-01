/**
 * Ask card — ONE queued ask, with the controls that answer it.
 *
 * WHAT THIS IS. The phone's answer surface for a queued ask (design
 * `docs/design/ask-nonblocking.md` §5.3). It is the queued twin of
 * `pending-card.tsx`, and the differences between the two are exactly the
 * differences the queue introduced:
 *
 *  * The queued wire carries the ask's WHOLE question list at once (§4), so
 *    this card is a FORM — every question visible, one submit — where the
 *    blocking card was a wizard advanced one question at a time. That is not a
 *    style choice: `AskQueue.respond` is atomic per ask and refuses an
 *    incomplete map, so a wizard would have to hold the answers anyway, and a
 *    form is what makes "you have answered 2 of 3" visible.
 *  * A queued ask has a DEADLINE, and the card says when it is (`expires_at`,
 *    rendered on the client clock per §5) rather than leaving the user to guess
 *    whether anybody is still waiting.
 *  * A queued ask can be DISMISSED. Dismissal injects nothing — it buys no
 *    turn — which is why it is offered as a quiet third action rather than a
 *    second button beside Decline, and why its own state line says "no reply
 *    was sent".
 *
 * THE SEND BUTTON IS ENABLED ONLY BY A COMPLETE MAP. Every question id must
 * carry a value; a question the user deliberately skipped is sent as an EMPTY
 * LIST, which is the queue's own spelling for "no answer" (§2.4). Letting a
 * partial map reach the wire would be refused anyway (`_partial_answer_error`),
 * so the card states the rule instead of collecting a refusal for it.
 *
 * REFUSALS KEEP THE DAEMON'S OWN SENTENCE. The queue's copy is the shared one
 * every surface shows — "already answered by desktop", "this ask expired 7 days
 * ago — ask again if it is still needed" — and the whole point of `render.
 * refusal_copy` living in core is that the phone must not author a fourth
 * wording of one rule. The card renders `error.message` verbatim, and only
 * falls back to a plain line when the daemon said nothing usable.
 */
import { useMemo, useState } from "react";
import { sendCommand } from "../api";
import { cn } from "../lib/cn";
import { answeredPairs, askStateLine, isAnswerable, unansweredQuestions } from "../lib/asks";
import { clearAskDraft, useAskDraft } from "../store";
import type { AskDraft } from "../store";
import type { AskQuestion, PendingAsk } from "../types";

/** The daemon's sentence, or the plainest honest thing when it gave none.
 *
 *  `api.request` falls back to the bare status when an error body is not JSON
 *  ("422" under a button explains nothing), and a fetch-level failure arrives as
 *  the browser's own TypeError. Neither is the queue's refusal copy, so neither
 *  may be rendered as though it were. */
function refusalText(error: unknown): string {
	if (error instanceof TypeError) return "could not reach the daemon";
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return "the daemon did not say why";
	return message;
}

/** One question's control, plus whether it currently holds a complete answer. */
function QuestionField({
	question,
	value,
	onChange,
	disabled,
}: {
	question: AskQuestion;
	/** The draft cell: chosen labels, or the typed text for a text/secret
	    question. An empty array means "not answered yet" — which is why a
	    deliberate skip is a separate state (`skipped`), not an empty value. */
	value: string[];
	onChange: (next: string[]) => void;
	disabled: boolean;
}) {
	const options = Array.isArray(question.options) ? question.options : [];
	const chosen = new Set(value);
	if (options.length === 0) {
		/* A free-text or secret question: the answer IS the typed string. */
		return (
			<div className="flex flex-col gap-1">
				{question.secret ? (
					<p className="text-meta text-ink-dim">
						secret — sent directly, not shown in the transcript
						{question.persist ? ", and saved to your credential store" : ""}
					</p>
				) : null}
				<input
					value={value[0] ?? ""}
					onChange={(event) => onChange([event.target.value])}
					disabled={disabled}
					type={question.secret ? "password" : "text"}
					autoComplete={question.secret ? "off" : undefined}
					autoCapitalize={question.secret ? "none" : undefined}
					autoCorrect={question.secret ? "off" : undefined}
					spellCheck={question.secret ? false : undefined}
					placeholder={question.secret ? "paste secret" : "your answer"}
					className="min-h-11 w-full rounded-sm border border-control bg-surface px-3 text-body text-ink outline-none placeholder:text-ink-dim disabled:opacity-50"
				/>
			</div>
		);
	}

	return (
		<div className="flex flex-col gap-2">
			{options.map((option, index) => {
				const on = chosen.has(option.label);
				/* THE RECOMMENDATION IS AN INDEX, not a flag (see `AskQuestion`): the
				   runtime hoists the recommended option to 0 and STATES the position,
				   because the harness option model has no boolean to read. Any index
				   is honoured — the wire, the type and the TUI all treat it as a
				   position, and pinning it to 0 would badge nothing the day the
				   runtime stops hoisting (agent review round 2, N1). */
				const recommended = question.recommended === index;
				return (
					<button
						key={option.label}
						type="button"
						disabled={disabled}
						onClick={() => {
							if (!question.multi) {
								onChange([option.label]);
								return;
							}
							/* Multi-select toggles; the answer map holds a list either
							   way, so the wire shape does not depend on `multi`. */
							const next = new Set(chosen);
							if (next.has(option.label)) next.delete(option.label);
							else next.add(option.label);
							onChange(Array.from(next));
						}}
						aria-pressed={on}
						className={cn(
							"flex min-h-11 flex-col justify-center rounded-sm border border-l-2 px-3 py-2 text-left",
							on
								? "border-accent border-l-accent bg-accent-wash"
								: "border-control border-l-accent bg-elevated active:bg-accent-wash",
							"disabled:opacity-50",
						)}
					>
						<span className="text-body-sm font-medium text-ink">
							{option.label}
							{recommended ? (
								/* The model's recommendation, as a word. §5's contract keeps
								   position as the only channel for older clients, but this
								   client is told outright, and a badge that says so is
								   cheaper to read than "the first one means it". */
								<span className="text-ink-dim"> · recommended</span>
							) : null}
						</span>
						{option.description ? (
							<span className="text-body-sm text-ink-muted">{option.description}</span>
						) : null}
					</button>
				);
			})}
			{question.multi ? (
				<span className="text-meta text-ink-dim">choose any that apply</span>
			) : null}
		</div>
	);
}

export function AskCard({
	row,
	sessionId,
	nowMs,
	onSettled,
}: {
	row: PendingAsk;
	sessionId: string;
	/** The client's clock, for the deadline line (§5: the countdown is rendered
	    from `expires_at` locally, never from a server-computed at-push string). */
	nowMs: number;
	/** Called after this surface settles, declines or dismisses the ask, so an
	    aggregate list can re-read the index rather than guess the new state. */
	onSettled?: () => void;
}) {
	/* The draft map: question id → the labels (or the typed string) the user has
	   chosen but not yet sent. IT LIVES IN THE STORE, not in this component
	   (QA round 1 Q-1 = UX round 1 U1): a card-local `useState` died with the
	   sheet's unmount, so collapsing the sheet discarded the draft that §5.0-R7
	   requires be kept. The chat buffer has lived in the store all along; this is
	   its twin, keyed by ask id so one collapse cannot mix two asks' answers. */
	const [draft, setDraft] = useAskDraft(row.ask_id);
	const [busy, setBusy] = useState<"" | "respond" | "decline" | "dismiss">("");
	const [error, setError] = useState("");

	const state = askStateLine(row, nowMs);
	const status = String(row.status || "open");
	const answerable = isAnswerable(status);
	/* DISMISS IS `timed_out`-ONLY, and the queue is the authority: `AskQueue.dismiss`
	   accepts no other status (`asks/queue.py`), and the design states the rule twice
	   (§2.2's table and `:115`). Offering it on an open or answered ask collected a
	   refusal whose sentence — "only a timed-out ask can be dismissed; it is still
	   open." — is FALSE about the row it sat under (agent review round 1, R1). */
	const dismissible = status === "timed_out";
	const questions = useMemo(() => (Array.isArray(row.questions) ? row.questions : []), [row]);
	const open_questions = useMemo(
		() => (answerable ? unansweredQuestions(row) : []),
		[answerable, row],
	);

	const skipped = useMemo(() => new Set(draft.skipped), [draft.skipped]);
	/* FUNCTIONAL, so two picks in one tick compose instead of overwriting (see
	   `useAskDraft`); the options of two different questions are the normal case. */
	const patchDraft = (patch: (current: AskDraft) => Partial<AskDraft>) =>
		setDraft((current) => {
			/* The patch runs ONCE per update (agent review round 2, N3): calling it
			   in both fields was harmless only because every caller passes a pure
			   function, which is not a property this signature can promise. */
			const next = patch(current);
			return {
				answers: next.answers ?? current.answers,
				skipped: next.skipped ?? current.skipped,
			};
		});

	/** Whether every still-open question carries a usable cell.
	 *
	 *  A question counts as answered when it has a non-empty draft, or the user
	 *  explicitly skipped it (which is sent as the empty list the queue's own
	 *  contract defines). A text answer that is only whitespace is not an
	 *  answer — the runtime would store a blank string, which reads as a
	 *  deliberate empty answer to the model. */
	const filled = (id: string): boolean => {
		if (skipped.has(id)) return true;
		const cell = draft.answers[id] ?? [];
		return cell.some((value) => value.trim() !== "");
	};
	const complete = open_questions.length > 0 && open_questions.every((q) => filled(String(q.id)));

	async function run(kind: "respond" | "decline" | "dismiss", send: () => Promise<unknown>) {
		if (busy) return;
		setBusy(kind);
		setError("");
		try {
			await send();
			/* THE DRAFT IS SPENT ONCE THE ASK SETTLES. Cleared here rather than left
			   for the next mount: the sheet re-reads the aggregate, so a settled card
			   can come back in the list, and a re-filled card the user already
			   answered reads as if their answer never landed. */
			clearAskDraft(row.ask_id);
			onSettled?.();
		} catch (failure) {
			setError(refusalText(failure));
			/* The controls come back: a refusal is a state the user can act on
			   (answer again, or read who beat them to it), and a card left inert
			   by its own error is the greyed-out-buttons defect the blocking card
			   already paid for. */
			setBusy("");
		}
	}

	const respond = () =>
		run("respond", () => {
			/* One body for every question of the ask — the atomic op. A skipped
			   question rides as an empty list, which is the contract's own
			   spelling for "no answer"; omitting the key is what the queue
			   refuses. */
			const answers: Record<string, string[]> = {};
			for (const question of questions) {
				const id = String(question.id);
				if (skipped.has(id)) answers[id] = [];
				else answers[id] = (draft.answers[id] ?? []).map((value) => value.trim());
			}
			return sendCommand(sessionId, { op: "ask_respond", ask_id: row.ask_id, answers });
		});

	const decline = () =>
		run("decline", () =>
			sendCommand(sessionId, { op: "ask_decline", ask_id: row.ask_id }),
		);

	const dismiss = () =>
		run("dismiss", () =>
			sendCommand(sessionId, { op: "ask_dismiss", ask_id: row.ask_id }),
		);

	const settledPairs = answeredPairs(row.questions, row.answers);
	/* INK FOLLOWS THE PROMISE, not the mood (design round 1, D2). An open ask's
	   line was `text-accent` and a settled one `text-success`, two greens that are
	   ΔE 17 apart here and IDENTICAL in three of the 28 shipped themes — so the
	   one distinction this card exists to draw (still waiting vs already settled)
	   could vanish. `accent` is no longer spent on the state line at all: "the
	   agent is continuing" is information, not an achievement, and success green
	   is reserved for the states that are receipts. */
	const toneClass =
		state.tone === "attention"
			? "text-warning"
			: state.tone === "settled"
				? "text-success"
				: state.tone === "gone"
					? "text-ink-dim"
					: "text-ink-muted";

	return (
		<div
			data-testid="ask-card"
			data-ask-id={row.ask_id}
			data-ask-status={String(row.status || "open")}
			className="flex flex-col gap-2 rounded-md border border-hairline bg-surface p-2.5"
		>
			<span className={cn("flex flex-wrap items-center gap-x-2 text-meta", toneClass)}>
				<span role="status">{state.text}</span>
				{row.urgent && answerable ? (
					<span className="text-ink-dim">· urgent</span>
				) : null}
			</span>

			{answerable ? (
				<>
					{open_questions.map((question, index) => {
						const id = String(question.id);
						return (
							<div key={id} className="flex flex-col gap-1.5">
								<span className="flex items-baseline gap-2 text-body font-medium">
									{questions.length > 1 ? (
										<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
											{index + 1}/{questions.length}
										</span>
									) : null}
									<span className="min-w-0">{question.question}</span>
								</span>
								<QuestionField
									question={question}
									value={draft.answers[id] ?? []}
									disabled={busy !== "" || skipped.has(id)}
									onChange={(next) =>
										patchDraft((current) => ({
											answers: { ...current.answers, [id]: next },
										}))
									}
								/>
								{/* A 44 px TARGET, NOT A 17 px LINK (UX round 1, U5 = design round
								    1, D5). This was `text-meta underline`: measured 136x17 against
								    44 px for every other control in the card, under WCAG 2.5.8's
								    24 px floor and with no spacing exception to claim (its centre
								    sits 8 px under the send button). It is the ONLY way to say
								    "no answer to this question", so it is a control, and it now
								    wears the card family's quiet control shape. */}
								{questions.length > 1 ||
								(Array.isArray(question.options) && question.options.length === 0) ? (
									<button
										type="button"
										disabled={busy !== ""}
										onClick={() => {
											const next = new Set(skipped);
											if (next.has(id)) next.delete(id);
											else next.add(id);
											patchDraft(() => ({ skipped: Array.from(next) }));
										}}
										className={cn(
											"flex min-h-11 self-start items-center rounded-sm border border-control px-3 text-body-sm active:bg-elevated disabled:opacity-50",
											skipped.has(id) ? "text-ink" : "text-ink-muted",
										)}
									>
										{skipped.has(id) ? "answer this after all" : "skip — send no answer"}
									</button>
								) : null}
							</div>
						);
					})}

					{/* R6 (agent review round 1): an answerable ask whose questions are all
					    already taken elsewhere used to render no fields and a permanently
					    disabled send, with nothing saying why. The queue is right (a draft
					    elsewhere is not this card's to submit); the card just has to say
					    so instead of inviting a tap that cannot land. */}
					{open_questions.length === 0 ? (
						<p className="text-body-sm text-ink-muted">
							nothing left to answer here — another surface has already taken these
							questions.
						</p>
					) : null}

					{/* THE REFUSAL SITS WITH THE CONTROLS IT ANSWERS (round-1 design, D4):
					    it used to render after the control row, which put it below the
					    fold in the frame captured to show it — the same failure
					    `pending-card.tsx` pinned its own error for. Above the row, the
					    sentence and the control that produced it are on screen together. */}
					{error ? <p className="text-body-sm text-danger">{error}</p> : null}

					<div className="flex flex-wrap gap-2">
						<button
							type="button"
							disabled={busy !== "" || !complete}
							onClick={respond}
							className="flex min-h-11 flex-1 items-center justify-center rounded-sm bg-accent px-4 text-body-sm font-medium text-on-accent active:bg-accent-active disabled:opacity-50"
						>
							{busy === "respond"
								? "…"
								: open_questions.length > 1
									? "send answers"
									: "send answer"}
						</button>
						<button
							type="button"
							disabled={busy !== ""}
							onClick={decline}
							className="flex min-h-11 items-center justify-center rounded-sm border border-danger-border bg-danger-wash px-3 text-body-sm text-danger active:bg-danger-wash disabled:opacity-50"
						>
							{busy === "decline" ? "…" : "decline"}
						</button>
					</div>
					{/* DISMISS IS A THIRD, QUIETER ACTION and it is deliberately not
					    beside Decline: declining ANSWERS the agent ("decide yourself"),
					    dismissing answers nobody and buys no turn. A pair of equal
					    buttons would make them look like two ways of saying no. Its own
					    state line says what happened. */}
					{dismissible ? (
						<button
							type="button"
							disabled={busy !== ""}
							onClick={dismiss}
							className="flex min-h-11 self-start items-center rounded-sm border border-control px-3 text-body-sm text-ink-muted active:bg-elevated disabled:opacity-50"
						>
							{busy === "dismiss" ? "…" : "dismiss — send no reply"}
						</button>
					) : null}
				</>
			) : (
				<>
					{/* A settled ask: the questions and the answers chosen, so the
					    card is a receipt rather than a sentence. `row.answers` holds a
					    secret answer as the KEY the runtime stored, never a value. */}
					{/* PAIRING CARRIED BY SPACE, not only by ink (design round 1, N1):
					    the receipt's inside-pair gap and its between-pair gap measured
					    ~10 px vs ~12 px, so a two-question receipt read as one
					    four-line block. The gap BETWEEN pairs is now the container's;
					    the gap inside a pair stays tight. */}
					{settledPairs.length > 0 ? (
						<dl className="flex flex-col gap-3">
							{settledPairs.map((pair) => (
								<div key={pair.question} className="flex flex-col">
									<dt className="text-body-sm text-ink-muted">{pair.question}</dt>
									<dd className="text-body-sm text-ink">{pair.answer || "—"}</dd>
								</div>
							))}
						</dl>
					) : (
						/* GUARDED like every other read of a wire list (agent review round
						   1, R5): a settled row that arrives without `questions` used to throw
						   during render, which unmounts the whole sheet rather than
						   degrading to a line. */
						<span className="text-body-sm text-ink-muted">
							{questions[0]?.question ?? ""}
						</span>
					)}
					{/* NO DISMISS HERE, deliberately (agent review round 1, R1): dismiss is
					    a `timed_out` action and a timed-out ask renders in the ANSWERABLE
					    branch above, so every status that reaches this branch (answered,
					    late, declined, dismissed, expired, unknown) can only be refused.
					    An EXPIRED ask additionally keeps no error register at all (§5) —
					    the remedy is in the state line. */}
				</>
			)}
		</div>
	);
}
