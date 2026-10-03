/**
 * Asks sheet — every queued ask, across conversations (design §5.3, R7's
 * EXPANDED state).
 *
 * WHY A SHEET AND NOT A LIST ON THE SESSION SCREEN. An ask is durable: it
 * outlives the runtime that queued it, and it belongs to a CONVERSATION rather
 * than to the screen you happen to be on. The phone's session screen can only
 * show the asks of the session it is addressing, which is exactly the case that
 * matters least — the ask you have not seen is the one in the conversation you
 * are not looking at (a session woken by a wake, a monitor, or a peer's
 * message). This sheet reads the aggregate (`GET /api/asks`), which is
 * index-backed, needs no runtime, and therefore answers the same way whether
 * the owning runtime is alive or long gone.
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
 * it), and on a 20 s backstop while it is open. The backstop is there for the
 * one event the list cannot announce: a DEADLINE passing in a conversation
 * whose runtime is gone, where nothing publishes a frame at all. It is a read
 * from an open sheet rather than a poll feeding a push, which is the bound
 * `asks/store.index_asks` asks a caller to keep in mind.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { getAsks } from "../api";
import { AskCard } from "./ask-card";
import { Sheet } from "./ui/sheet";
import { isAnswerable, orderedForDisplay, outstandingAsks } from "../lib/asks";
import { useAsksRevision, useSessions } from "../store";
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
	onOpenConversation,
}: {
	open: boolean;
	onClose: () => void;
	/** The conversation this sheet was opened from, if any. Rows belonging to it
	    are not labelled with a conversation name — the reader is already there. */
	currentSessionId?: string;
	/** Navigate to a row's conversation (and close the sheet). Absent on screens
	    that cannot navigate — the row then simply carries no such control. */
	onOpenConversation?: (sessionId: string) => void;
}) {
	const [rows, setRows] = useState<PendingAsk[]>([]);
	const [loaded, setLoaded] = useState(false);
	const [error, setError] = useState("");
	const [nowMs, setNowMs] = useState(() => Date.now());
	const revision = useAsksRevision();
	const { sessions } = useSessions();

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
	   one settled on another surface). ONE effect for both triggers: two
	   effects keyed on `open` would each fire on the opening transition and
	   fetch the aggregate twice for one tap. */
	useEffect(() => {
		if (!open) return;
		void load();
	}, [open, revision, load]);

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

	const names = useMemo(() => {
		const map = new Map<string, { name: string; ended: boolean }>();
		for (const session of sessions) {
			map.set(session.session_id, {
				name: session.conversation_name || session.session_id,
				ended: session.ended === true,
			});
		}
		return map;
	}, [sessions]);

	/* THE HEAD FIRST, then the wire's own order (`orderedForDisplay`) — the
	   order the minimized chip names (UX round 1, U8). */
	const listed = useMemo(() => orderedForDisplay(rows), [rows]);
	/* QUESTIONS, the same unit the bar and the header entry state (agent review
	   round 1, R3): three surfaces answering "how much is waiting" in two units is
	   the defect, even where their populations legitimately differ (this one
	   spans every conversation, the bar is this session's). */
	const outstandingQuestions = outstandingAsks(rows).reduce(
		(total, row) => total + (Array.isArray(row.questions) ? row.questions.length : 0),
		0,
	);
	const title =
		outstandingQuestions > 0
			? `asks · ${outstandingQuestions} question${outstandingQuestions === 1 ? "" : "s"}`
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
				{loaded && rows.length === 0 && !error ? (
					<p className="text-body-sm text-ink-muted">
						nothing waiting — questions the agent asks will appear here.
					</p>
				) : null}
				{listed.map((row) => {
					const sessionId = String(row.session_id || currentSessionId || "");
					const foreign = Boolean(sessionId) && sessionId !== currentSessionId;
					const named = names.get(sessionId);
					return (
						<div key={`${sessionId}:${row.ask_id}`} className="flex flex-col gap-1">
							{foreign ? (
								<span className="flex items-baseline gap-2 text-meta text-ink-dim">
									<span className="min-w-0 truncate">
										{named?.name ?? sessionId}
									</span>
									{onOpenConversation && row.durable !== false ? (
										/* THE PARENT OWNS THE WHOLE TRANSITION (agent review round 2,
										   M1). This control used to navigate and THEN close itself, and
										   closing gives the sheet's history entry back with
										   `history.back()` — which, on an entry the navigation had
										   just pushed, popped the target and left the hash back at
										   `#/s/asks`. `onOpenConversation` now clears the sheet AND
										   replaces its entry, in that order, in one place. */
										<button
											type="button"
											onClick={() => onOpenConversation(sessionId)}
											/* 44 px, like every other control on this sheet (design
											   D7 = UX U9): it was a bare `underline` link at 28x17 css
											   — the smallest target here and the only route from the
											   sheet to a foreign ask's conversation. */
											className="flex min-h-11 shrink-0 items-center rounded-sm border border-control px-3 text-body-sm text-ink-muted active:bg-elevated"
										>
											open
										</button>
									) : null}
								</span>
							) : null}
							{/* THE WAIT IS STATED BEFORE THE TAP, AND ONLY WHERE THERE IS ONE
							    (design round 1, D4 = UX U4; the two gates are the round-2 corrections).

							    ANSWERABLE, because a settled receipt offers nothing to answer, so
							    a wait stated above it is noise about a tap that does not exist
							    (UX round 2, U6). This strip used to render only for an `ended` row
							    with copy naming a manual remedy the relay now performs itself;
							    `runtime_live` widened it to the merely-not-running conversation
							    the operator actually hits.

							    DURABLE, because `runtime_live === false` is ALSO true of a
							    conversation with no transcript at all — where every op is a
							    terminal refusal, and the promise of a ~30 s bring-up is a claim
							    about a backend that does not exist (design round 2, D6). On
							    those rows the strip goes silent and the card states the truth
							    it already knows, so the sheet and the card cannot contradict
							    each other one element apart. */}
							{isAnswerable(String(row.status || "open")) &&
							row.durable !== false &&
							(named?.ended || row.runtime_live === false) ? (
								<p className="text-meta text-ink-dim">
									{named?.ended
										? "this conversation has ended — answering will reopen it (this can take up to ~30 s)"
										: "this conversation is not running — answering will bring it up (this can take up to ~30 s)"}
								</p>
							) : null}
							<AskCard
								row={row}
								sessionId={sessionId}
								nowMs={nowMs}
								runtimeLive={row.runtime_live}
								durable={row.durable}
								onSettled={() => void load()}
							/>
						</div>
					);
				})}
			</div>
		</Sheet>
	);
}
