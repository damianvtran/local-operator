/**
 * Pending echoes — the client-local row for an instruction that is in flight.
 *
 * The problem this exists for: `submitContinuation` resolves only when the
 * daemon answers, and until it did the composer's own textarea was the only
 * place a sent message appeared — so on a phone link a send looked like nothing
 * had happened until the receipt landed and the message popped into the
 * conversation. The row is therefore painted the instant the user sends, and
 * withdrawn or replaced the instant the truth about it arrives.
 *
 * ## Reconciliation is by IDENTITY, never by words
 *
 * The key is the retry envelope's own `command_id`, and it is exact because
 * that id is ALSO the id of the user message the session will write: the phone
 * route hands `message_id=command.command_id` to `Session.prompt`
 * (`serving.py`, `tui_handle.py`), `Session.prompt` builds the user `Message`
 * with it, the durable writer passes `message.id` as the transcript entry id,
 * and the mobile fold emits the user row with `id=message.id`. So the row this
 * module paints and the row the projection later carries share one id, live and
 * on replay alike.
 *
 * Matching on the TEXT instead — or on "is the tail a user row now" — is the
 * `test_user_echo_dedup.py` defect one surface over: a distinct message whose
 * words collide with a pending echo (a repeated "yes" sent twice) would be taken
 * for the one already on screen, and the second message would never be painted.
 *
 * ## The exactly-one-row property is structural, not a timing hope
 *
 * A held echo is never rendered while the projection already carries its id
 * (see `usePendingEchoes`), so the pending row and its real row cannot both be
 * on screen in any frame — including the frame between the projection arriving
 * and the store entry being dropped. The entry is then dropped for real, so a
 * transcript that later loses the row (a re-cap, a `/clear`) cannot resurrect a
 * phantom from this module.
 *
 * ## Nothing is persisted, and nothing needs to be
 *
 * A reload mid-flight is already honest without help from here: the immutable
 * retry envelope in `localStorage` survives it and the composer's mount shows
 * the retained-retry affordance. A second, transient copy of the same fact in
 * storage would need its own TTL, its own eviction bound and its own place in
 * the logout contract for no user-visible gain, so the echo lives in memory for
 * exactly as long as the in-flight request does. It is still PRIVATE content —
 * it holds what the user typed — so `clearPrivateSessionStorage` drops it with
 * every other content-bearing store on an identity change.
 *
 * ## Everything that ends an echo
 *
 *   - the projection carries the id: the real row owns the message now (drop);
 *   - the send threw: the command was refused, or its delivery is ambiguous,
 *     and the composer's own failure alert — with its retry under the SAME
 *     envelope id — is the honest state from there, never a row that looks sent
 *     (withdraw).
 *
 * There is deliberately NO timeout. A `prompt admitted` that the transcript has
 * not caught up with yet, and a steer queued for a later turn boundary, are both
 * rows the session genuinely still owes — the truthful reading is that the row
 * is not there yet, which is precisely what the pending row says.
 */
import { useEffect, useMemo, useSyncExternalStore } from "react";
import type { TranscriptEntry } from "./types";

/** One in-flight command, as the row that stands in for it needs it. */
export interface PendingEcho {
	/** The envelope's id, and the id the session will write its row under. */
	commandId: string;
	/** Exactly the body the envelope carries — never the live draft. */
	text: string;
	/** Attachment count. The BYTES are never held here: the composer keeps its
	    own previews and the receipt path revokes them, so a second copy of an
	    object URL would be a revoke this module could not see coming. */
	imageCount: number;
}

/** Per route, because two conversations can have an instruction in flight at
    once and each owns its own row. */
const echoes = new Map<string, PendingEcho[]>();

const listeners = new Set<() => void>();

/** One stable empty list, so a route with no echo keeps a referentially stable
    snapshot across emissions (`useSyncExternalStore` compares with Object.is). */
const NONE: PendingEcho[] = [];

function emit(): void {
	for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
	listeners.add(listener);
	return () => listeners.delete(listener);
}

export function registerPendingEcho(sessionId: string, echo: PendingEcho): void {
	const held = echoes.get(sessionId);
	if (held) {
		if (held.some((item) => item.commandId === echo.commandId)) return;
		echoes.set(sessionId, [...held, echo]);
	} else {
		echoes.set(sessionId, [echo]);
	}
	emit();
}

export function withdrawPendingEcho(sessionId: string, commandId: string): void {
	const held = echoes.get(sessionId);
	if (!held) return;
	const next = held.filter((item) => item.commandId !== commandId);
	if (next.length === held.length) return;
	if (next.length === 0) echoes.delete(sessionId);
	else echoes.set(sessionId, next);
	emit();
}

/** Drop every echo this device holds, whatever the route.
 *
 * For `clearPrivateSessionStorage`: a signed-out user's echo shows text the
 * next person at this device must not see, on the one surface that renders it. */
export function clearPendingEchoes(): void {
	if (echoes.size === 0) return;
	echoes.clear();
	emit();
}

/** Whether the projection already carries the row for this command.
 *
 * Asked by the composer before it paints an echo — a retry of an envelope the
 * session already wrote must not paint a second row under the same id — and by
 * the hook to decide which echoes are still owed one. One rule, one place. */
export function projectionCarriesCommand(
	transcript: TranscriptEntry[] | undefined,
	commandId: string,
): boolean {
	return projectedUserIds(transcript ?? []).has(commandId);
}

/** The ids of the user rows the projection currently carries. Only `user`
    rows: a notice or a parent message can never stand in for a sent prompt,
    and neither of those carries a command id at all. */
function projectedUserIds(transcript: TranscriptEntry[]): Set<string> {
	const ids = new Set<string>();
	for (const entry of transcript) {
		if (entry.kind === "user") ids.add(entry.id);
	}
	return ids;
}

/** The echoes this route owes a row for — excluding any the projection has
    already answered, so exactly one row per command can ever be on screen. */
export function usePendingEchoes(
	sessionId: string,
	transcript: TranscriptEntry[] | undefined,
): PendingEcho[] {
	const held = useSyncExternalStore(subscribe, () => echoes.get(sessionId) ?? NONE);
	const projected = useMemo(
		() => projectedUserIds(transcript ?? []),
		[transcript],
	);
	const live = held.filter((echo) => !projected.has(echo.commandId));
	/* Drop the answered entries from module state. Separate from the filter
	   above on purpose: the filter is what makes the frame correct, this is what
	   keeps a long-lived tab from accumulating one dead entry per send. */
	useEffect(() => {
		for (const echo of held) {
			if (projected.has(echo.commandId)) {
				withdrawPendingEcho(sessionId, echo.commandId);
			}
		}
	}, [held, projected, sessionId]);
	return live;
}
