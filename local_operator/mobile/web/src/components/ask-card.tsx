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
 *  * A queued ask can be DISMISSED. Dismissal asks the model nothing — it buys
 *    no turn — which is why it is offered as a quiet third action rather than a
 *    second button beside Decline, and why its own state line says "no reply
 *    was sent". What it can still OWE is the deadline notice that ask never got
 *    (design `ask-nonblocking.md` §14), delivered on the conversation's next
 *    run: the agent's fact, not this card's action.
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
import { HttpError, sendCommand } from "../api";
import { cn } from "../lib/cn";
import { Disclosure } from "./ui/disclosure";
import {
	answeredPairs,
	askStateLine,
	LAPSED_BODY,
	isAnswerable,
	isDeadConversation,
	isLapsed,
	unansweredQuestions,
} from "../lib/asks";
import { clearAskDraft, messageRuntime, useAskDraft } from "../store";
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

/** The stale-ask context line (SA-D7 / R5), or `null` when there is none.
 *
 *  "3 messages from you since this was asked (last 19:36)": operator messages
 *  admitted since the ask was raised, and when the latest one was. It is the one
 *  line on this card that says a question may ALREADY have been answered in
 *  chat — the reason the card can be closed with `answered_in_chat`.
 *
 *  ABSENCE IS THE CONTRACT, exactly as `asks/store.pending_row` writes it: an
 *  older runtime sends no field, and that draws NOTHING rather than a zero the
 *  reader would have to disbelieve. A zero is a fact (the ask is newer than every
 *  message) and draws nothing either — there is no staleness to report. An OPEN
 *  ask only: a settled one has no wait left to be stale.
 *
 *  The clock is formatted HERE rather than through ICU: it is a wall-clock time,
 *  not prose, and every locale's catalogue carries the finished string inside the
 *  sentence it belongs to. Latin digits in 24-hour form, matching the terminal's
 *  own rendering of the same pair (`tui.widgets.ask_queue.context_text`). */
function contextLine(row: PendingAsk): string | null {
	if (String(row.status || "open") !== "open") return null;
	const count = row.user_messages_since;
	if (typeof count !== "number" || count <= 0) return null;
	const last = row.last_user_message_at;
	if (typeof last !== "number" || last <= 0) return null;
	const at = new Date(last);
	const time = `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
	return messageRuntime.t("relay.ask.context.messages", { count, time });
}

/** The sentence a row shows when the aggregate ALREADY reports its conversation
 *  as DEAD (`isDeadConversation`: no durable transcript AND nothing live to
 *  deliver to) — the state the daemon refuses every op on with
 *  `ask_session_gone` (design round 2, D6).
 *
 *  IT MIRRORS `daemon._ASK_GONE_RESPONSE` rather than waiting for the refusal,
 *  and that duplication is the point of a pre-emptive state: fetching the
 *  sentence would mean buying the guaranteed 409 the state exists to avoid. It
 *  is the ANSWER's wording, remedy included (design round 3, D7's NIT: one
 *  voice), because the card offers no control at all — so the sentence that fits
 *  is the one the primary control would have produced, and `start a new
 *  conversation and ask again` is the user's actual next move. The ops that
 *  carry their own sentence (a dismissal, a handed-over decision) are only told
 *  apart when the refusal comes FROM THE WIRE, which this branch never does. */
const CONVERSATION_GONE_COPY =
	"this conversation no longer exists — the ask can never be read; start a new conversation and ask again.";

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
	onDismissed,
}: {
	row: PendingAsk;
	sessionId: string;
	/** The client's clock, for the deadline line (§5: the countdown is rendered
	    from `expires_at` locally, never from a server-computed at-push string). */
	nowMs: number;
	/** Called after this surface settles, declines or dismisses the ask, so an
	    aggregate list can re-read the index rather than guess the new state. It
	    names the OP: a sheet holding a dismissal receipt must retire it when the
	    operator's next act was an answer, not keep describing the old one
	    (agent review round 2, M1 = UX round 2, U9). */
	onSettled?: (kind: "respond" | "decline" | "dismiss") => void;
	/** Called after a DISMISSAL is accepted, with the ask it was about and the
	    reason that was sent. The sheet needs both: the reason for the receipt's
	    words, and the id so a later dismissal replaces this one instead of being
	    confused with it (M1). */
	onDismissed?: (askId: string, reason: "moot" | "answered_in_chat" | undefined) => void;
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
	/** Which control produced the refusal, so the sentence can sit beside IT
	    (design round 1, D5): a dismiss refusal rendered above the send/decline row
	    landed 60 px from the button that produced it, with the other pair between
	    them. */
	const [errorOp, setErrorOp] = useState<"" | "respond" | "decline" | "dismiss">("");
	/** The conversation is GONE (`ask_session_gone`): terminal, so no control may
	    stay live (design round 1, D3). */
	const [gone, setGone] = useState(false);
	/** The two reason buttons are showing (`close…` pressed on an open ask).
	    A separate state from `busy`: revealing the choice sends NOTHING, so a
	    card that showed the in-flight mark here would claim a dismissal the
	    operator has not made. */
	const [closing, setClosing] = useState(false);
	/** Which reason was SENT, so the button that carries it keeps its label while
	    the dismissal is in flight (UX round 1, U3). Both buttons used to collapse
	    to `…`, which erased the only on-screen statement of what the operator
	    chose. */
	const [sentReason, setSentReason] = useState<"moot" | "answered_in_chat" | undefined>(
		undefined,
	);
	/** A dismiss refusal has been read: the reasons stay visible, inert (U8). */
	const [dismissRefused, setDismissRefused] = useState(false);

	/* THE COLD CASE, and only the known one (design round 1, D1/D4 = UX U1/U3/U4):
	   `runtime_live === false` means the relay will have to bring the conversation
	   up before the op can land — FOR ANSWER AND DECLINE, which is a ~30 s envelope
	   against a warm answer's 1–3 s. A dismissal is the exception that takes the
	   no-boot door on every surface (design §14, fork A), so `cold` buys it nothing
	   and nothing here mentions it. `undefined` (an older daemon, or a row from a
	   projection) promises nothing and keeps the bare `…`.

	   BOTH FACTS ARE READ OFF THE ROW, not passed as props: they are the
	   aggregate's own fields (`PendingAsk.runtime_live` / `.durable`), and a
	   second copy of them on the component was exactly the drift D7 was made of. */
	const cold = row.runtime_live === false;

	const state = askStateLine(row, nowMs);
	const status = String(row.status || "open");
	const answerable = isAnswerable(status);
	/* A LAPSED ASK IS SHOWN, NOT OFFERED (Ben's report, 2026-10-15). Its deadline
	   has passed and the agent has gone ahead without it, so options and a send under
	   it read as though the agent were still waiting on them. The store would still
	   take a late answer on `timed_out` (`isAnswerable` keeps saying so); the phone
	   deliberately stops inviting one. `lapsed` gates the controls only — the dead
	   and dismiss rules below are unchanged. */
	const lapsed = isLapsed(row);
	const offered = answerable && !lapsed;

	/* A ROW THE AGGREGATE ALREADY KNOWS IS DEAD IS TERMINAL BEFORE THE TAP
	   (design round 2, D6 + UX U7), and DEAD IS A CONJUNCTION (design round 3,
	   D7). `isDeadConversation` is the route's own two facts: no durable
	   transcript to read AND nothing live to deliver to. Gating on `durable`
	   alone made the client MORE PESSIMISTIC THAN THE SERVER — a live row with no
	   transcript was shown as a dead end while `ask_respond` on it returned
	   `200 {"ok": true, "detail": "answered"}` — the same class of lie in the
	   other direction, and the reason the predicate lives in `lib/asks.ts` and is
	   shared by this card, the sheet's strip and the sheet's title count. Gated
	   further on `answerable`: a SETTLED row in a dead conversation is a receipt,
	   not a dead end, and must keep its receipt. */
	const goneFromWire = isDeadConversation(row) && answerable;
	const terminal = gone || goneFromWire;
	/* A LAPSED ASK KEEPS ITS ONE-TAP DISMISS (D11, §14), and an OPEN one now has a
	   door of its own (R5 / SA-D4): `AskQueue.dismiss` accepts `open` on every live
	   session (the live refusal was retired — see the design's amended §15 D6), so
	   the refusal this comment used to record ("only a timed-out ask can be
	   dismissed; it is still open.") no longer exists to collect. The two doors
	   differ in what they ASK the operator, not in whether they are allowed: a
	   lapsed ask was already walked past by the agent, so it goes at once; an open
	   one is still owed an answer, so the card asks WHY first and sends nothing
	   until a reason is chosen. */
	const dismissible = status === "timed_out";
	const reason_needed = status === "open";
	const questions = useMemo(() => (Array.isArray(row.questions) ? row.questions : []), [row]);
	const open_questions = useMemo(
		() => (offered ? unansweredQuestions(row) : []),
		[offered, row],
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

	async function run(
		kind: "respond" | "decline" | "dismiss",
		send: () => Promise<unknown>,
		onAccepted?: () => void,
	) {
		if (busy) return;
		setBusy(kind);
		setError("");
		setErrorOp("");
		try {
			await send();
			onAccepted?.();
			/* THE DRAFT IS SPENT ONCE THE ASK SETTLES. Cleared here rather than left
			   for the next mount: the sheet re-reads the aggregate, so a settled card
			   can come back in the list, and a re-filled card the user already
			   answered reads as if their answer never landed. */
			clearAskDraft(row.ask_id);
			onSettled?.(kind);
		} catch (failure) {
			setError(refusalText(failure));
			setErrorOp(kind);
			/* A GONE CONVERSATION IS TERMINAL (design round 1, D3). The ask can
			   never be read, so every further tap is a guaranteed 409 — the
			   sentence, already written for THIS op by the daemon (D2), is the
			   remedy, and the controls must stop inviting a tap that cannot
			   work. */
			if (failure instanceof HttpError && failure.code === "ask_session_gone") {
				setGone(true);
			}
			/* A REFUSED DISMISSAL STOPS INVITING ANOTHER (UX round 1, U8). The
			   daemon's most common sentence here is "you already dismissed this":
			   the ask is settled and every further tap is a guaranteed repeat. The
			   reasons stay on screen — the sentence belongs beside the control that
			   produced it (design D5) — but they go inert, and `cancel` is the way
			   back to a fresh attempt if the refusal was momentary (a dead
			   connection rather than a settled ask). */
			if (kind === "dismiss") setDismissRefused(true);
			/* The controls come back: a refusal is a state the user can act on
			   (answer again, or read who beat them to it), and a card left inert
			   by its own error is the greyed-out-buttons defect the blocking card
			   already paid for. */
		} finally {
			/* THE OUTCOME ENDS THE IN-FLIGHT STATE, WHICHEVER OUTCOME IT IS (UX round
			   2, U5). `setBusy("")` lived only in the catch, so a SUCCESSFUL answer
			   left the card asserting `bringing the session up — this can take up to
			   ~30 s` over a dead `…` until — and only if — the sheet's aggregate
			   re-read replaced the row. A refused read (the sheet's own 8 s bound, or
			   a hung backstop) left that lie on screen indefinitely. The promise has
			   settled, so the claim that it is still settling must stop.

			   The controls coming back after a refusal is deliberate: a refusal is a
			   state the user can act on (answer again, or read who beat them to
			   it), and a card left inert by its own error is the
			   greyed-out-buttons defect the blocking card already paid for. */
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

	const dismiss = (reason?: "moot" | "answered_in_chat") => {
		setSentReason(reason);
		return run(
			"dismiss",
			() =>
			/* The reason rides only when there is one: a lapsed ask's door is the
			   one-key dismissal the queue has always taken, whose body must stay the
			   shape every older daemon accepts. An OPEN ask's reason is the whole
			   point of the two-choice row — `moot` closes the question, and
			   `answered_in_chat` tells the agent the answer is already in the
			   conversation, so it stops waiting and reads what was typed. */
				sendCommand(
					sessionId,
					reason
						? { op: "ask_dismiss", ask_id: row.ask_id, reason }
						: { op: "ask_dismiss", ask_id: row.ask_id },
				),
			() => onDismissed?.(row.ask_id, reason),
		);
	};

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

	const body = (
		<>
			{/* THE STATE LINE STANDS DOWN WHEN THE CARD IS TERMINAL. "Queued — the
			    agent is continuing; expires in 11 m" beside "this conversation no
			    longer exists" is the same contradiction §5's preamble forbids — the
			    timer and the agent both belong to a conversation that is not there.
			    The refusal sentence is the whole state. */}
			{terminal ? null : (
				<span className={cn("flex flex-wrap items-center gap-x-2 text-meta", toneClass)}>
					{/* Under the `Timed out` badge the line starts at what that means:
					    the badge says the word one line above (#2481 UX round 2, U13). A
					    lapsed, non-terminal card is always inside that Disclosure. */}
					<span role="status">{lapsed ? LAPSED_BODY : state.text}</span>
					{row.urgent && offered ? (
						<span className="text-ink-dim">· urgent</span>
					) : null}
				</span>
			)}

			{terminal ? (
				/* TERMINAL, AND IT IS THE SENTENCE THAT SAYS SO (design round 1, D3 for
				   the refusal that arrived; design round 2, D6 for the row the aggregate
				   already reported dead, where the same sentence is shown BEFORE the tap
				   that could only ever be refused). The controls are withheld rather
				   than handed back enabled, because every further tap is a guaranteed
				   409 and the surface must not invite one. The remedy is a NEW message
				   in a new conversation, which is what the daemon's per-op sentence
				   names (D2) when the refusal came from the wire — a dismissal is told
				   nothing was sent, an answer is told to ask again, a decision is told
				   the ask can never be read (U9). */
				<p className="text-body-sm text-danger">
					{gone ? error : CONVERSATION_GONE_COPY}
				</p>
			) : lapsed ? (
				<>
					{/* READ-ONLY: the question the agent asked, so the reader knows what
					    lapsed, and nothing to answer it with. Dismiss is the only op. A
					    lapsed row is always `timed_out` (`isLapsed`; the index readers
					    re-derive a stale `open` past its deadline), so the Dismiss gate and
					    the queue's own rule are the same statement. It stays apart from any
					    Decline: declining ANSWERS the agent, dismissing answers nobody and
					    buys no turn (the agent is told on its next run — §14). */}
					{questions.map((question, index) => (
						<span
							key={String(question.id ?? index)}
							className="flex items-baseline gap-2 text-body-sm text-ink-muted"
						>
							{questions.length > 1 ? (
								<span className="shrink-0 font-mono text-mono-sm text-ink-dim">
									{index + 1}/{questions.length}
								</span>
							) : null}
							<span className="min-w-0">{question.question}</span>
						</span>
					))}
					{dismissible ? (
						<>
							<button
								type="button"
								disabled={busy !== ""}
								aria-busy={busy === "dismiss" || undefined}
								/* No reason: a lapsed ask's door is the one-key dismissal, and
								   the queue records its own default (the agent already walked past
								   that deadline, so nothing turns on the operator naming one). */
								onClick={() => dismiss()}
								className="flex min-h-11 self-start items-center rounded-sm border border-control px-3 text-body-sm text-ink-muted active:bg-elevated disabled:opacity-50"
							>
								{/* D1 (design round 1): the label IS the width, so it stays put while
								    the dismissal is in flight. The in-flight mark used to be the bare
								    glyph "…", which collapsed the control from 208 px to a 44 px square
								    under the thumb — a glitch rather than feedback, and now the whole
								    in-flight life of a dismissal is one append and a flash. The only
								    change is that it dims (`disabled`) and says it is busy. Kept off the
								    answer/decline pair, where D1 called the same glyph optional. */}
								dismiss — send no reply
							</button>
							{/* NO COLD LINE, AND NO COLD CLOCK (fork A, 2026-10-16). A dismissal
							    on a conversation with no runtime used to ENGAGE it to record the
							    dismissal, so the card owed the reader that fact before the tap (UX
							    round 1, U3) and stated its ~30 s while it ran (design round 1, D1 =
							    UX U1). A dismissal now takes the no-boot door on both surfaces — one
							    append against the log, no boot, no turn — so "dismiss — send no
							    reply" above is the WHOLE truth, warm or cold, and there is nothing
							    left to promise or to wait out. Left as a comment rather than an
							    absence, because the next reader came here looking for the engage
							    copy the other ops still genuinely carry: the cold ANSWER's clock at
							    the bottom of this card, and the sheet's strip above it. */}
							{error && errorOp === "dismiss" ? (
								<p className="text-body-sm text-danger">{error}</p>
							) : null}
						</>
					) : null}
				</>
			) : offered ? (
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

					{/* THE STALE-ASK CONTEXT LINE (SA-D7 / R5), UNDER the question it
					    refers to (design round 1, D3): it used to sit above the question,
					    so "this was asked" pointed at a sentence the reader had not read
					    yet — and the terminal paints the same line under its row. It is
					    the card's one statement that the question may already be settled
					    in chat, which is exactly the case the `answered_in_chat` reason
					    exists for. No line at all when the runtime sent no count (an older
					    runtime), when the count is zero, or on a settled card — see
					    `contextLine`. */}
					{(() => {
						const context = terminal ? null : contextLine(row);
						return context ? (
							<p className="text-meta text-ink-dim" data-testid="ask-context">
								{context}
							</p>
						) : null;
					})()}

					{/* THE COLD ANSWER'S CLOCK, STATED (design round 1, D1 = UX U1). A cold
					    answer is an engage plus a dial plus an ack in ONE call — ~30 s
					    measured, against the 1–3 s a warm answer pays — and until this line
					    existed the whole in-flight affordance was a bare `…` with the state
					    line still reading "Queued — the agent is continuing". Saying nothing
					    is also a claim, and it was the wrong one. */}
					{busy !== "" && busy !== "dismiss" && cold ? (
						/* NOT FOR A DISMISSAL (agent review round 1, M1): the engine takes
						   the no-boot door on every surface (`dismiss_cold`, §14/fork A),
						   so the dismissal this card offers is one append whether the
						   conversation is live or cold — the ~30 s bring-up belongs to the
						   ANSWER and DECLINE below, which really do engage. The gate is
						   the op, not the row: `cold` says what an answer would cost. */
						<p className="text-body-sm text-ink-muted">
							bringing the session up — this can take up to ~30 s
						</p>
					) : null}

					{/* THE REFUSAL SITS WITH THE CONTROL IT ANSWERS (design round 1, D5 for
					    the dismiss arm; round 1's D4 fixed the send/decline pair). A dismiss
					    refusal rendered here would land 60 px above the button that produced
					    it, with the send/decline pair between them, so it renders beside
					    dismiss instead — see the dismiss control below. */}
					{error && errorOp !== "dismiss" ? (
						<p className="text-body-sm text-danger">{error}</p>
					) : null}

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

					{/* THE OPEN ASK'S OWN DOOR (R5, §15's amended D6): `close…` puts the
					    QUESTION away without answering anybody, which is the act a stale
					    ask needs and the one the card never offered on an open row. It
					    asks WHY first — the two reasons are what the agent acts on — and
					    a tap on `close…` sends nothing: the reason row is a choice, not a
					    dismissal.

					    IT IS NOT A SECOND `decline`. Declining ANSWERS the agent ("no"),
					    buys nothing and is irreversible; closing says the question stopped
					    mattering and asks the agent for nothing at all, and on a live
					    conversation it need not even start a turn (a dismissal is held
					    durably until the agent's next turn). */}
					{reason_needed ? (
						<div className="flex flex-col gap-2">
							{closing ? (
								/* THE REASONS ARE FULL-WIDTH ROWS WITH THEIR OWN NOTES (design
								   round 1, D1 = UX U6). They used to be three controls on one
								   flex-wrap line — no heading, no explanation, `cancel` orphaned
								   onto a second line in Urdu — and "Answered in chat" sat 8.5 px
								   under a live `decline`, so a reader with a drafted answer could
								   read it as "send my answer". The heading names the act (the
								   same words the terminal uses), and each row states what the
								   agent will be told, in the same shape the answer options above
								   already use. */
								<div className="flex flex-col gap-2">
									{/* THE QUIET CONTROLS SIT ON THE CARD'S TEXT AXIS (design round
									    2, D10): `cancel` and the `dismiss…` reveal are borderless, and
									    their `px-2` box put their words 7.5 px inside the heading and
									    the question, which the eye reads as a misalignment. `px-0`
									    aligns the words to the box's own edge; `min-h-11` keeps the
									    44 px target and the label's width keeps it above the 24 px
									    floor. */}
									<span className="text-meta text-ink-dim">
										{messageRuntime.t("relay.ask.dismiss.title")}
									</span>
									{(["moot", "answered_in_chat"] as const).map((reason) => (
										<button
											key={reason}
											type="button"
											disabled={busy !== "" || dismissRefused}
											aria-busy={
												(busy === "dismiss" && sentReason === reason) || undefined
											}
											onClick={() => dismiss(reason)}
											className={cn(
												"flex w-full flex-col items-start gap-0.5 rounded-sm border border-control px-3 py-2 text-left active:bg-elevated disabled:opacity-50",
												sentReason === reason && busy === "dismiss"
													? "text-ink"
													: "text-ink-muted",
											)}
										>
											{/* THE PRESSED LABEL STAYS PUT (UX round 1, U3). Both buttons
											    used to collapse to `…`, erasing the only statement of
											    which reason had been sent; the pressed row keeps its
											    words and gains the in-flight mark, the sibling dims. */}
											<span className="text-body-sm">
												{messageRuntime.t(`relay.ask.dismiss.${reason}`)}
												{busy === "dismiss" && sentReason === reason ? " …" : ""}
											</span>
											<span className="text-meta text-ink-dim">
												{messageRuntime.t(`relay.ask.dismiss.${reason}_note`)}
											</span>
										</button>
									))}
									<button
										type="button"
										disabled={busy !== ""}
										onClick={() => {
											/* THE REFUSAL GOES WITH THE BLOCK IT EXPLAINED (UX round 2,
											   U10 = agent review round 2, m1): the sentence renders
											   outside the `closing` ternary, so leaving it behind put
											   an old refusal under a fresh-looking `dismiss…`. */
											setClosing(false);
											setDismissRefused(false);
											setError("");
											setErrorOp("");
										}}
										className="flex min-h-11 self-start items-center rounded-sm px-0 text-body-sm text-ink-dim active:bg-elevated disabled:opacity-50"
									>
										{messageRuntime.t("relay.ask.dismiss.cancel")}
									</button>
								</div>
							) : (
								<button
									type="button"
									disabled={busy !== "" || dismissRefused}
									onClick={() => setClosing(true)}
										className="flex min-h-11 self-start items-center rounded-sm px-0 text-body-sm text-ink-dim active:bg-elevated disabled:opacity-50"
								>
									{messageRuntime.t("relay.ask.dismiss.open")}
								</button>
							)}
							{/* THE REFUSAL SITS WITH THE CONTROL THAT PRODUCED IT (design
							    round 1, D5): a relay or bridge ask refuses the dismissal in
							    the queue's own words, and that sentence belongs beside the
							    reason the operator pressed, not 60 px above it. */}
							{error && errorOp === "dismiss" ? (
								<p className="text-body-sm text-danger">{error}</p>
							) : null}
						</div>
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
					    a `timed_out` action and a timed-out ask renders in the LAPSED
					    branch above, so every status that reaches this branch (answered,
					    late, declined, dismissed, withdrawn, expired, unknown) can only
					    be refused. An EXPIRED ask additionally keeps no error register at
					    all (§5) — the remedy is in the state line. */}
				</>
			)}
		</>
	);

	return (
		<div
			data-testid="ask-card"
			data-ask-id={row.ask_id}
			data-ask-status={String(row.status || "open")}
			className="flex flex-col gap-2 rounded-md border border-hairline bg-surface p-2.5"
		>
			{lapsed && !terminal ? (
				/* A LAPSED ASK IS COLLAPSED BEHIND ITS BADGE (decision D11, Ben's
				   2026-10-15 request): it reads the way the desktop's declined row does,
				   one line with a status word, and seeing it is an explicit tap. The
				   app-wide `Disclosure` idiom, never a second collapse pattern, closed by
				   default and taking no focus on mount (D12: a lapsed ask never claims
				   the user). The body is the read-only card above — the lapsed line, the
				   question, and Dismiss, which stays inside so nothing on the collapsed
				   header can be tapped by accident. A TERMINAL row is not wrapped: its
				   refusal sentence is the whole state and must stay readable. */
				<Disclosure
					headerClassName="gap-2"
					header={(open: boolean) => (
						<>
							<span className="shrink-0 rounded-sm border border-hairline px-1.5 text-meta text-ink-muted">
								Timed out
							</span>
							{/* The header names the question only while CLOSED (#2481 UX U7):
							    open, the body states it in full, and a truncated copy one line
							    above it said the same thing twice. */}
							{open ? null : (
								<span className="min-w-0 truncate text-body-sm text-ink-muted">
									{questions[0]?.question ?? ""}
								</span>
							)}
						</>
					)}
				>
					<div className="flex flex-col gap-2 pt-1">{body}</div>
				</Disclosure>
			) : (
				body
			)}
		</div>
	);
}
