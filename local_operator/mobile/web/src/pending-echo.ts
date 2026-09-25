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
 * and the mobile fold emits the row with `id=message.id`. So the row this module
 * paints and the row the projection later carries share one id, live and on
 * replay alike.
 *
 * The KIND is not part of the identity and must not be used as one: the fold
 * writes a phone row as `user`, as `steer`, or — for the rare hub-envelope
 * rewrite — as `parent_message`, all under that one id. `COMMAND_ROW_KINDS`
 * carries the set and the evidence for it; matching on `user` alone shipped a
 * permanent duplicate on the steer path, which is the one flow that exists only
 * while a turn is running (review B-1 / design D1 / QA Q-1).
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
 * ## What the reconciliation LOOKS like, stated honestly
 *
 * The docstring above used to claim the receipt is "text losing its `sending…`
 * caption", which overstates it, and the design round said so (D2). What
 * actually changes is what the pending row's own styling makes it:
 *
 *   - a **steer** takes the transcript's steer box (hairline, `body-sm`, `py-1`)
 *     and keeps it, so for a steer the receipt really is the caption leaving;
 *   - a **prompt** takes the user bubble's box and settles into it, so the
 *     receipt also brings the accent leading edge and brighter ink — that pair
 *     being the pending signal the design round approved as reading "sent, not
 *     yet accepted";
 *   - either way the caption line's own height is given back, which moves a
 *     bottom-pinned conversation by ~21px at receipt — reviewed and ACCEPTED as
 *     the better trade (design D3), not a defect;
 *   - and the rare hub-envelope rewrite settles into a `parent_message` card,
 *     a register the pending row cannot anticipate from the op alone. Recorded
 *     rather than guessed at.
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
import type { ContinuationOp } from "./continuation-command";
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
	/** Which command this was, because the row it settles into depends on it: a
	    steer settles into the transcript's steer row and a prompt into the user
	    bubble, and the pending row takes the box of whichever is coming so that
	    reconciliation is a caption leaving rather than a row changing register
	    (design round 1, D2). It is also what the caption's tense is read off. */
	op: ContinuationOp;
	/** Whether the daemon has ACKNOWLEDGED this command — set by
	    `markPendingEchoAccepted` when the receipt lands, i.e. when the session
	    has admitted the command durably. It is the only thing that separates
	    "we do not know yet" from "the session has it and has not written the row
	    yet", and the two need different words (UX round 1, U2). */
	accepted: boolean;
}

/** The kinds a row the USER authored can be written under, carrying the command
 *  id the front end minted for it.
 *
 * A set of kinds rather than an id-only test, and a set rather than
 * `kind === "user"`, and both halves of that were measured rather than assumed:
 *
 *  - a phone **steer** is written as `kind="steer"` under the same id —
 *    `note_user_message(text, steer=True, message_id=command_id)` in
 *    `serving.py` and `tui_handle.py`, folded by `projection.py`'s
 *    `kind="steer" if steer else "user"`. The composer picks `op="steer"` by
 *    itself whenever the turn is streaming, so matching only `user` left the
 *    one flow that exists ONLY while a turn runs unreconciled: the message
 *    rendered twice and the phantom claimed `sending…` for the life of the tab
 *    (review B-1 / design D1 / QA Q-1, measured at 2224/2224 DOM samples).
 *  - a phone row whose text turned out to be a hub envelope is REWRITTEN to
 *    `kind="parent_message"` by `absorb_user_event` — still under that id, so
 *    it has to reconcile too or the same phantom returns by a rarer door.
 *  - the same row is repainted as `kind="user"` when the session is resumed and
 *    replayed from disk, so matching only `steer` would trade one half of the
 *    bug for the other.
 *
 * An id-only test is deliberately not used: `notice`, `tool` and `assistant`
 * rows have their ids minted by the session, the provider or the fold, and a
 * collision there would retire a pending row for a message the session never
 * wrote.
 */
const COMMAND_ROW_KINDS: ReadonlySet<TranscriptEntry["kind"]> = new Set([
	"user",
	"steer",
	"parent_message",
]);

/** What the pending row says, for the state the echo is in.
 *
 * Every word here is one this product already uses for the same fact, and none
 * of them invites the user to send again:
 *
 *  - `sending…` — no receipt yet: nothing is known about what happened.
 *  - `queued …` — the receipt named a STEER. The message was delivered and is
 *    waiting for a turn boundary, which is the state the composer's own footer
 *    prints as `N queued`; saying `sending…` there for the rest of a turn that
 *    runs for minutes was the reading that invited a second send, and it
 *    contradicted a line 600px below it (UX round 1, U2). The two tenses are
 *    the TUI's own two states for a queued steer — waiting for the step running
 *    now, or for the next turn once this one has ended.
 *  - `sent` — accepted, and not a steer: the session has it and its own row is
 *    imminent.
 */
export function pendingEchoCaption(echo: PendingEcho, streaming: boolean): string {
	if (!echo.accepted) return "sending…";
	if (echo.op !== "steer") return "sent";
	return streaming ? "queued — sends when this step finishes" : "queued — sends with your next message";
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

/** Record that the daemon acknowledged this command, so the row can stop saying
 *  it is still going out (UX round 1, U2).
 *
 * Called by the composer on the receipt — a 2xx is the session's admission, so
 * from there the message is the session's and the only open question is when its
 * own row lands. Silently a no-op when the echo is already gone: the projection
 * may have answered it first, and a steer's receipt can land after the row.
 */
export function markPendingEchoAccepted(sessionId: string, commandId: string): void {
	const held = echoes.get(sessionId);
	if (!held) return;
	const index = held.findIndex((item) => item.commandId === commandId);
	if (index === -1 || held[index].accepted) return;
	const next = [...held];
	next[index] = { ...next[index], accepted: true };
	echoes.set(sessionId, next);
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

/** The ids the projection currently carries a row for, over the kinds a
    user-authored row can be written under — see ``COMMAND_ROW_KINDS`` for why
    the set is those three and not just ``user``. */
function projectedUserIds(transcript: TranscriptEntry[]): Set<string> {
	const ids = new Set<string>();
	for (const entry of transcript) {
		if (COMMAND_ROW_KINDS.has(entry.kind)) ids.add(entry.id);
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
	/* Memoised because this screen repaints on every SSE frame: the filter is what
	   every render would otherwise allocate a fresh array for, and any future
	   effect keyed on `pending` would then re-run per frame (review round 1, N-1). */
	const live = useMemo(
		() => held.filter((echo) => !projected.has(echo.commandId)),
		[held, projected],
	);
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
