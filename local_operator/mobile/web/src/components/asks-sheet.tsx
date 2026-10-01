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
import { useAsksRevision, useSessions } from "../store";
import type { PendingAsk } from "../types";

/** How often the open sheet re-reads the aggregate as a deadline backstop. */
const BACKSTOP_MS = 20000;

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
	const revision = useAsksRevision();
	const { sessions } = useSessions();

	const load = useCallback(async () => {
		try {
			const answer = await getAsks();
			setRows(Array.isArray(answer.asks) ? answer.asks : []);
			setError("");
		} catch (failure) {
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

	return (
		<Sheet open={open} onClose={onClose} title="asks">
			<div className="flex flex-col gap-3 p-3">
				{error ? <p className="text-body-sm text-danger">{error}</p> : null}
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
				{rows.map((row) => {
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
									{onOpenConversation ? (
										<button
											type="button"
											onClick={() => {
												onOpenConversation(sessionId);
												onClose();
											}}
											className="shrink-0 underline"
										>
											open
										</button>
									) : null}
								</span>
							) : null}
							{/* A conversation whose process is GONE is stated before
							    the controls, not discovered by a refused tap: the ask
							    outlives its runtime, so answering may need it reopened,
							    and the reader deserves to know that before pressing. */}
							{named?.ended ? (
								<p className="text-meta text-ink-dim">
									this conversation has ended — answering may need it reopened
								</p>
							) : null}
							<AskCard
								row={row}
								sessionId={sessionId}
								nowMs={Date.now()}
								onSettled={() => void load()}
							/>
						</div>
					);
				})}
			</div>
		</Sheet>
	);
}
