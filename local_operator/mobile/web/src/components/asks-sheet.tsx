/**
 * Asks sheet — the queued asks of the conversation it was opened from (design
 * §5.3, R7's EXPANDED state).
 *
 * ONE CONVERSATION, ON PURPOSE. The sheet used to list every conversation's
 * asks, so opening one session's questions showed every session's (Ben's
 * report, 2026-10-15): the entry sits on a conversation's own screen and reads
 * as that conversation's questions, and other conversations' rows under it
 * were noise at best and a wrong answer at worst. It now shows only the rows whose `session_id` is the opening conversation's.
 * Other conversations' asks stay visible where they belong: on the session
 * list, as each row's own ask chip.
 *
 * IT STILL READS THE AGGREGATE (`GET /api/asks`) and filters client-side,
 * because the aggregate is index-backed and needs no runtime: it answers the
 * same way whether the owning runtime is alive or long gone, which a read of
 * the live projection cannot.
 *
 * THE ANSWER SURFACE IS HERE. While this sheet is open, the phone's ask
 * composer is the sheet's own controls — each ask's form, with its own send —
 * and the conversation composer underneath is unreachable (the sheet is modal
 * over the phone column; see `ui/sheet.tsx`). That is §5.0's routing invariant
 * kept structurally rather than by a flag: with the surface EXPANDED the only
 * field that can receive a keystroke is an answer field, so a chat draft can
 * never become an answer, and with it MINIMIZED (this sheet closed) the
 * composer underneath is an ordinary conversation composer, so an answer can
 * never be sent as chat. The two drafts live in different components and
 * neither is ever copied into the other.
 *
 * LIVE WITHOUT A MANUAL REFRESH, and bounded. The sheet re-reads the aggregate
 * when it opens, whenever the daemon's own list frames change the outstanding
 * ask population (the store's `asksRevision` — a new ask or a settlement moves
 * it) or this conversation's own ask rows change (how a lapsed ask's dismissal
 * or late answer arrives: the list frame's count is 0 either side, #2388), and
 * on a 20 s backstop while it is open. The backstop is there for the
 * one event the list cannot announce: a DEADLINE passing in a conversation
 * whose runtime is gone, where nothing publishes a frame at all. It is a read
 * from an open sheet rather than a poll feeding a push, which is the bound
 * `asks/store.index_asks` asks a caller to keep in mind.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { getAsks } from "../api";
import { messageRuntime } from "../store";
import { AskCard } from "./ask-card";
import { Sheet } from "./ui/sheet";
import {
	isAnswerable,
	askRowsSignature,
	isDeadConversation,
	isLapsed,
	orderedForDisplay,
	questionTally,
} from "../lib/asks";
import { useAsksRevision, useProjection, useSessions } from "../store";
import type { PendingAsk } from "../types";

/** How often the open sheet re-reads the aggregate as a deadline backstop. */
const BACKSTOP_MS = 20000;
/** How long a single aggregate read may hang before the sheet says so.
 *
 *  WHY A CLOCK IS THE RIGHT ANSWER FOR A HUNG READ (UX round 1, U6). A read that
 *  FAILS was already handled (the daemon's sentence, or "could not reach the
 *  daemon"); a read that never settles — a connection that is merely flaky,
 *  never reset — left `reading asks…` on screen forever, with the 20 s backstop
 *  re-issuing the same stalled request and no way forward but closing the sheet.
 *  The wait is the failure here, so the wait is what must be bounded. */
const READ_TIMEOUT_MS = 8000;
/** How often the countdowns re-render while the sheet is open.
 *
 *  Deadlines are stated in minutes, so a per-second tick would buy nothing; a
 *  coarse one keeps the crossing within half a minute of the truth (agent review
 *  round 1, N2 — before this the countdown was sampled only when a re-read
 *  landed, so a crossing could be stated up to 20 s late). */
const TICK_MS = 15000;

function failureText(error: unknown): string {
	if (error instanceof TypeError) return "could not reach the daemon";
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return "the daemon did not say why";
	return message;
}

export function AsksSheet({
	open,
	onClose,
	currentSessionId,
}: {
	open: boolean;
	onClose: () => void;
	/** The conversation this sheet was opened from: the only one whose asks it
	    lists (see the module note). */
	currentSessionId: string;
}) {
	const [allRows, setRows] = useState<PendingAsk[]>([]);
	const [loaded, setLoaded] = useState(false);
	/** The receipt for a dismissal this sheet has just seen, KEYED TO THE ASK it
	    was about (agent review round 2, M1 = UX round 2, U9).

	    The daemon answers `{"ok": true, "detail": "dismissed"}`, and the card
	    leaves the aggregate — so the receipt is the only trace of the act. Two
	    things the first cut got wrong, both fixed here: it rendered only when the
	    sheet was EMPTY (so the ordinary multi-ask queue acknowledged nothing), and
	    it survived the operator's NEXT act (answering a second ask left "Dismissed
	    — no longer needed" over an answer, which is the class of claim this lane
	    exists to remove). The ask id is kept so a later dismissal replaces this
	    one rather than being confused with it. */
	const [receipt, setReceipt] = useState<
		{ askId: string; reason: "moot" | "answered_in_chat" | "lapsed" } | null
	>(null);
	const [error, setError] = useState("");
	const [nowMs, setNowMs] = useState(() => Date.now());
	const revision = useAsksRevision();
	const { sessions } = useSessions();
	/* THIS CONVERSATION'S OWN ROWS MOVE WHEN A LAPSED ASK LEAVES. The list
	   frame's `asks_open` does not (a timed-out ask is not counted, #2388), so
	   a dismissal or a late answer on another surface would otherwise wait for
	   the backstop. */
	const rowsSignature = askRowsSignature(useProjection(currentSessionId).projection?.asks);

	const load = useCallback(async () => {
		/* THE READ IS BOUNDED (U6). `AbortSignal.timeout` is the platform's own
		   answer and needs no bookkeeping of its own; where the engine lacks it
		   (the happy-dom test environment does), the read is simply unbounded
		   rather than unavailable — a missing abort must not break the sheet. */
		const signal =
			typeof AbortSignal !== "undefined" && "timeout" in AbortSignal
				? AbortSignal.timeout(READ_TIMEOUT_MS)
				: undefined;
		try {
			const answer = await getAsks(signal);
			setRows(Array.isArray(answer.asks) ? answer.asks : []);
			setError("");
		} catch (failure) {
			if (failure instanceof DOMException && failure.name === "TimeoutError") {
				setError("the read timed out — the daemon is not answering");
				return;
			}
			/* The daemon's sentence, or the plainest honest line when it gave
			   none — the projects sheet's rule, which exists because `request`
			   falls back to the bare status ("503" explains nothing) and a
			   fetch-level failure arrives as the browser's own TypeError. */
			setError(failureText(failure));
		} finally {
			setLoaded(true);
		}
	}, []);

	/* A NEW OPENING RE-READS, and so does a changed population (a new ask, or
	   one settled on another surface, lapsed ones included). ONE effect for all
	   triggers: two effects keyed on `open` would each fire on the opening
	   transition and fetch the aggregate twice for one tap. */
	useEffect(() => {
		if (!open) {
			/* A RECEIPT BELONGS TO ONE OPENING: reopening later must not present
			   a dismissal from the last visit as though it just happened. */
			setReceipt(null);
			return;
		}
		void load();
	}, [open, revision, rowsSignature, load]);

	/* The deadline backstop, while the sheet is open only. */
	useEffect(() => {
		if (!open) return;
		const timer = setInterval(() => void load(), BACKSTOP_MS);
		return () => clearInterval(timer);
	}, [open, load]);

	/* The countdown's own clock (N2): the deadline lines are rendered from
	   `expires_at` on the CLIENT's clock (§5), and this is what makes them
	   advance while the sheet sits open rather than only when a read lands. */
	useEffect(() => {
		if (!open) return;
		setNowMs(Date.now());
		const timer = setInterval(() => setNowMs(Date.now()), TICK_MS);
		return () => clearInterval(timer);
	}, [open]);

	/* Only the ended/not-running strip reads the session list now: whether the
	   opening conversation has ended changes the strip's wording. */
	const ended = useMemo(
		() => sessions.some((session) => session.session_id === currentSessionId && session.ended === true),
		[sessions, currentSessionId],
	);

	/* THIS CONVERSATION'S ROWS ONLY, before anything orders or counts them, so
	   the title, the empty state and the list all describe the same set. */
	const rows = useMemo(
		() => allRows.filter((row) => String(row.session_id) === currentSessionId),
		[allRows, currentSessionId],
	);

	/* THE HEAD FIRST, then the wire's own order (`orderedForDisplay`) — the
	   order the minimized chip names (UX round 1, U8). */
	const listed = useMemo(() => orderedForDisplay(rows), [rows]);
	/* THE RECEIPT IS THE INSTANT ECHO, AND ONLY WHILE ITS OWN ROW IS MISSING
	   (SA-D8, design D1 = UX U2). Since §17 a dismissal taken before the ask's
	   deadline KEEPS the row as a settled receipt, so from the first read after
	   the tap the sheet would state one fact twice: this unattributed line above
	   the list, and the same catalogue sentence on the card below it — two type
	   sizes, two ink tokens, one `gap-3` apart, and the upper one reads as a
	   status of whatever sits directly under it.

	   The ROW is the statement; the receipt covers the gap until it arrives, and
	   is matched on the ask id it already keeps. THE LAPSED DOOR IS THE
	   EXCEPTION and stays unconditional: a dismissal at or after the deadline
	   still removes the row (case 2 of §17), so no row ever arrives to replace
	   the receipt — it is that door's only acknowledgement. */
	const shownReceipt =
		receipt &&
		(receipt.reason === "lapsed" || !listed.some((row) => row.ask_id === receipt.askId))
			? receipt
			: null;
	/* QUESTIONS, the same unit the bar and the header entry state (agent review
	   round 1, R3): three surfaces answering "how much is waiting" in two units is
	   the defect. All three now count this one conversation's rows.

	   ONLY OFFERED QUESTIONS ARE WAITING (design round 1, D1), the dock's and the
	   header's rule: a lapsed question is shown read-only, so "asks · 1 question"
	   above a card reading Timed out stated a live decision one element above the
	   card that says there is none. A lapsed-only sheet is titled for what it
	   holds.

	   DEAD ROWS' QUESTIONS ARE NOT WAITING ON ANYONE (agent review round 2,
	   MINOR-3). A dead conversation's ask still carries `open`, so it enters
	   `outstandingAsks` — but every op on it is refused with `ask_session_gone`,
	   and a count that promises answers the backend will refuse is the same class
	   of claim this whole lane exists to remove. It is the sheet's own rule
	   ("no contradiction one element apart") applied to the title, one element
	   above the strip that already honours it. */
	const { waiting, lapsed } = questionTally(rows.filter((row) => !isDeadConversation(row)));
	const title =
		waiting > 0
			? `asks · ${waiting} question${waiting === 1 ? "" : "s"}`
			: lapsed > 0
				? `asks · ${lapsed} timed out`
				: "asks";

	return (
		<Sheet open={open} onClose={onClose} title={title}>
			<div className="flex flex-col gap-3 p-3">
				{error ? (
					<div className="flex flex-wrap items-center gap-2">
						<p className="text-body-sm text-danger">{error}</p>
						{/* A RETRY, because a hung or failed read must not leave the sheet
						   with closing as its only move (U6). */}
						<button
							type="button"
							onClick={() => void load()}
							className="flex min-h-11 items-center rounded-sm border border-control px-3 text-body-sm text-ink-muted active:bg-elevated"
						>
							try again
						</button>
					</div>
				) : null}
				{/* THE LOADING LINE IS NOT DECORATION: without it the sheet paints its
				    empty state while the read is still in flight, so "nothing waiting"
				    is shown about an answer that has not arrived yet. It is one line and
				    it disappears the moment the read settles. */}
				{!loaded && !error ? (
					<p className="text-body-sm text-ink-muted">reading asks…</p>
				) : null}
				{/* THE RECEIPT IS NOT THE EMPTY STATE (agent review round 2, M1 = UX
				    round 2, U9). It used to render only when the list was empty, so a
				    dismissal from a queue of two acknowledged nothing at all; it now
				    sits above whatever remains — the place the card left — with the
				    empty state below it saying only that nothing is waiting.

				    AND IT STANDS DOWN ONCE ITS OWN ROW ARRIVES (SA-D8): the settled
				    card below states the same sentence, so keeping both would say one
				    thing twice. See `shownReceipt`. */}
				{shownReceipt ? (
					<p className="text-body-sm text-ink-muted" data-testid="ask-receipt">
						{messageRuntime.t(
							shownReceipt.reason === "lapsed"
								? "relay.ask.dismiss.receipt"
								: `relay.ask.dismiss.receipt_${shownReceipt.reason}`,
						)}
					</p>
				) : null}
				{loaded && rows.length === 0 && !error && !shownReceipt ? (
					<p className="text-body-sm text-ink-muted" data-testid="ask-sheet-empty">
						nothing waiting — questions the agent asks will appear here.
					</p>
				) : null}
				{listed.map((row) => {
					const sessionId = currentSessionId;
					return (
						<div key={`${sessionId}:${row.ask_id}`} className="flex flex-col gap-1">
							{/* THE WAIT IS STATED BEFORE THE TAP, AND ONLY WHERE THERE IS ONE
							    (design round 1, D4 = UX U4; the two gates are the round-2 corrections).

							    ANSWERABLE, because a settled receipt offers nothing to answer, so
							    a wait stated above it is noise about a tap that does not exist
							    (UX round 2, U6). This strip used to render only for an `ended` row
							    with copy naming a manual remedy the relay now performs itself;
							    `runtime_live` widened it to the merely-not-running conversation
							    the operator actually hits.

							    DURABILITY, because `runtime_live === false` is ALSO true of a
							    conversation with no transcript at all — where the op is a
							    terminal refusal and the promise of a ~30 s bring-up is a claim
							    about a backend that does not exist (design round 2, D6). On
							    those rows the strip goes silent and the card states the truth
							    it already knows, so the sheet and the card cannot contradict
							    each other one element apart. It is `isDeadConversation`, the
							    CONJUNCTION of the two fields — not `durable` alone, which would
							    silence the strip on a live conversation that answers fine
							    (design round 3, D7).

							    NOT LAPSED, because a lapsed card offers no answer (`isLapsed`),
							    so "answering will bring it up" would promise a tap the card one
							    element below does not have. */}
							{isAnswerable(String(row.status || "open")) &&
							!isLapsed(row) &&
							!isDeadConversation(row) &&
							(ended || row.runtime_live === false) ? (
								<p className="text-meta text-ink-dim">
									{ended
										? "this conversation has ended — answering will reopen it (this can take up to ~30 s)"
										: "this conversation is not running — answering will bring it up (this can take up to ~30 s)"}
								</p>
							) : null}
							<AskCard
								row={row}
								sessionId={sessionId}
								nowMs={nowMs}
								onSettled={(kind) => {
									/* A NON-DISMISSAL SETTLEMENT RETIRES THE RECEIPT (M1): after
									   answering, an unretired "Dismissed — …" would describe the
									   operator's previous act as though it were their last. */
									if (kind !== "dismiss") setReceipt(null);
									void load();
								}}
								onDismissed={(askId, reason) =>
									setReceipt({ askId, reason: reason ?? "lapsed" })
								}
							/>
						</div>
					);
				})}
			</div>
		</Sheet>
	);
}
