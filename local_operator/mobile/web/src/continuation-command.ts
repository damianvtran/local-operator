import { sendCommand } from "./api";
import type { PromptImage } from "./types";

/*
 * ============================================================================
 * Pending-retry envelope lifecycle — the one authoritative contract
 * ============================================================================
 *
 * The persisted immutable retry envelope (`lo-mobile-command:<sessionId>`) is
 * the exact bytes of an instruction whose delivery outcome is UNKNOWN. Its UUID
 * is the identity of that body: a retry replays these bytes under the SAME UUID
 * so an already-admitted instruction is de-duplicated by the daemon instead of
 * being sent twice. Every keep/clear decision in this module and its callers
 * (composer.tsx, api.ts, private-storage.ts, main.tsx) resolves to this table.
 * A second, ad-hoc rule beside it is a defect.
 *
 * KEEP the envelope across — anything that leaves the outcome genuinely
 * ambiguous, because the original UUID may already be admitted on the host:
 *   - transient transport failure: fetch/network error, timeout, AND HTTP
 *     502/504/408. These statuses are ACKNOWLEDGEMENT loss, not rejection: the
 *     daemon writes and drains the owner frame before awaiting its ack
 *     (daemon.py `request`), so a 504 ack-timeout / 502 transport error can
 *     occur after the command was durably admitted.
 *   - page reload and SSE reconnect (persistence + TTL revalidation on read).
 *   - switching to another of MY OWN conversations and back. The envelope is
 *     scoped per session/route, so navigating away must NOT delete it —
 *     returning restores the recovery affordance. (This is U1: the prior build
 *     purged every other route's envelope on composer mount, silently losing
 *     the affordance on the most ordinary navigation there is.)
 *
 * CLEAR the envelope only on — a definitive end of THIS UUID's ambiguity, or an
 * end of the privacy scope it lives in:
 *   - definitive acknowledgement of the UUID (the send resolved).
 *   - definitive rejection: an HTTP status that proves PRE-admission refusal
 *     (400/401/403/409/422/…), never an ambiguous 408/502/504.
 *   - TTL expiry (24h) — read-time revalidation drops an aged envelope.
 *   - explicit user discard.
 *   - logout / identity change / a 401 that ends the authenticated session's
 *     privacy scope: clear ALL scoped storage (drafts + every envelope), so a
 *     later user on the device never inherits private content. This is distinct
 *     from "navigating between my own sessions", which keeps per-session state.
 *
 * The keep/clear split for a caught HTTP error is centralised in
 * `isAmbiguousDeliveryStatus` so transport, correlation, and tests share one
 * definition instead of re-deriving it.
 */

export type ContinuationOp = "prompt" | "steer";

export interface ContinuationEnvelope {
	command_id: string;
	op: ContinuationOp;
	text: string;
	images?: PromptImage[];
}

export interface ContinuationReceipt {
	ok: boolean;
	detail: string;
	envelope: ContinuationEnvelope;
}

interface StoredContinuation {
	version: 1;
	saved_at: number;
	envelope: ContinuationEnvelope;
}

const COMMAND_PREFIX = "lo-mobile-command:";
const COMMAND_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_STORED_COMMAND_CHARS = 4 * 1024 * 1024;
const MAX_COMMAND_TEXT_CHARS = 100_000;
const MAX_COMMAND_IMAGES = 8;
/* Envelopes are scoped per session so navigating between my own conversations
   never drops one (U1). Storage is instead bounded by COUNT: only a handful of
   distinct conversations can hold an unresolved uncertain instruction at once,
   and a new one beyond the bound evicts the OLDEST by save time — never the
   route the user is on. This replaces the old single-envelope, purge-on-mount
   design that failed silently on ordinary navigation. */
const MAX_PENDING_ENVELOPES = 8;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function commandStorageKey(sessionId: string): string {
	return `${COMMAND_PREFIX}${sessionId}`;
}

/** HTTP statuses that leave delivery AMBIGUOUS rather than proving pre-admission
 * rejection. 408 (request timeout), 502 (bad gateway), 504 (gateway/ack
 * timeout) can all occur AFTER the daemon has admitted the UUID, so the
 * envelope must survive them for a same-UUID retry. Any other 4xx/5xx is a
 * definitive server verdict on an un-run command and retires the envelope. */
function isAmbiguousDeliveryStatus(status: number): boolean {
	return status === 408 || status === 502 || status === 504;
}

function validEnvelope(value: Partial<ContinuationEnvelope>): value is ContinuationEnvelope {
	return (
		typeof value.command_id === "string" &&
		UUID_RE.test(value.command_id) &&
		(value.op === "prompt" || value.op === "steer") &&
		typeof value.text === "string" &&
		value.text.length <= MAX_COMMAND_TEXT_CHARS &&
		(value.images === undefined ||
			(Array.isArray(value.images) &&
				value.images.length <= MAX_COMMAND_IMAGES &&
				value.images.every(
					(image) =>
						typeof image === "object" &&
						image !== null &&
						typeof image.data_b64 === "string" &&
						typeof image.mime_type === "string" &&
						image.mime_type.startsWith("image/"),
				)))
	);
}

/** Every stored envelope key with its save time, oldest first. Corrupt or
 * legacy-shaped entries sort as oldest (time 0) so they are the first evicted. */
function storedEnvelopeAges(): { key: string; savedAt: number }[] {
	const rows: { key: string; savedAt: number }[] = [];
	for (let index = 0; index < localStorage.length; index++) {
		const key = localStorage.key(index);
		if (!key?.startsWith(COMMAND_PREFIX)) continue;
		let savedAt = 0;
		try {
			const value = JSON.parse(localStorage.getItem(key) ?? "") as Partial<StoredContinuation>;
			if (typeof value.saved_at === "number") savedAt = value.saved_at;
		} catch {
			/* Unparseable state sorts oldest; it is evicted first. */
		}
		rows.push({ key, savedAt });
	}
	return rows.sort((a, b) => a.savedAt - b.savedAt);
}

/** Hold total pending envelopes at the count bound by evicting the OLDEST,
 * never the route being written. ``keepKey`` is always retained even if it is
 * the oldest, so writing the active route can never evict itself. */
function enforceEnvelopeBound(keepKey: string): void {
	const rows = storedEnvelopeAges().filter((row) => row.key !== keepKey);
	while (rows.length + 1 > MAX_PENDING_ENVELOPES) {
		const victim = rows.shift();
		if (!victim) break;
		localStorage.removeItem(victim.key);
	}
}

function saveEnvelope(key: string, envelope: ContinuationEnvelope, savedAt: number): void {
	const raw = JSON.stringify({ version: 1, saved_at: savedAt, envelope } satisfies StoredContinuation);
	if (raw.length > MAX_STORED_COMMAND_CHARS) {
		throw new Error("This instruction is too large to retain safely for retry.");
	}
	/* Bound BEFORE the write so a fresh envelope for a new route evicts an older
	   route's, never the one we are about to persist (U1: per-session scope). */
	enforceEnvelopeBound(key);
	try {
		localStorage.setItem(key, raw);
	} catch {
		throw new Error("This instruction could not be retained safely for retry.");
	}
}

export function getPendingContinuation(sessionId: string): ContinuationEnvelope | null {
	return storedEnvelope(commandStorageKey(sessionId));
}

export function hasPendingContinuation(sessionId: string): boolean {
	return getPendingContinuation(sessionId) !== null;
}

export function clearPendingContinuation(sessionId: string): void {
	localStorage.removeItem(commandStorageKey(sessionId));
}

function storedEnvelope(key: string): ContinuationEnvelope | null {
	const raw = localStorage.getItem(key);
	if (!raw || raw.length > MAX_STORED_COMMAND_CHARS) {
		localStorage.removeItem(key);
		return null;
	}
	try {
		const value = JSON.parse(raw) as Partial<StoredContinuation> & Partial<ContinuationEnvelope>;
		if (value.version === 1) {
			if (
				typeof value.saved_at === "number" &&
				value.saved_at <= Date.now() &&
				Date.now() - value.saved_at <= COMMAND_TTL_MS &&
				value.envelope &&
				validEnvelope(value.envelope)
			) {
				return value.envelope;
			}
			localStorage.removeItem(key);
			return null;
		}
		if (validEnvelope(value)) {
			/* Upgrade the short-lived raw envelope shipped by the previous build
			   without granting it unbounded storage lifetime. */
			saveEnvelope(key, value, Date.now());
			return value;
		}
	} catch {
		/* Corrupt or quota-truncated state has no trustworthy UUID/body pairing. */
	}
	localStorage.removeItem(key);
	return null;
}

/** Retain the exact producer envelope until acknowledgement. The UUID is the
 * identity of this body, not a mutable composer slot: after uncertain delivery,
 * retries replay these bytes while later edits remain a separate draft. */
export async function submitContinuation(
	sessionId: string,
	op: ContinuationOp,
	text: string,
	images?: PromptImage[],
): Promise<ContinuationReceipt> {
	const key = commandStorageKey(sessionId);
	const stored = storedEnvelope(key);
	const envelope = stored ?? {
		op,
		command_id: crypto.randomUUID(),
		text,
		images: images?.map((image) => ({ ...image })),
	};
	if (!validEnvelope(envelope)) {
		clearPendingContinuation(sessionId);
		throw new Error("This instruction cannot be retained safely for retry.");
	}
	if (!stored) saveEnvelope(key, envelope, Date.now());
	try {
		const reply = await sendCommand(sessionId, envelope);
		clearPendingContinuation(sessionId);
		return { ...reply, envelope };
	} catch (error) {
		/* Keep the UUID/body pair retryable whenever the outcome is ambiguous —
		   every transport failure, plus the acknowledgement-loss statuses
		   (408/502/504) that can follow a durable admission. Retire it only for a
		   status that proves the daemon definitively refused an un-run command;
		   replaying a rejected envelope is never useful and re-sending after an
		   ambiguous failure under a NEW UUID is exactly the duplicate hazard this
		   envelope exists to prevent. See the lifecycle contract above. */
		if (
			typeof error === "object" &&
			error !== null &&
			"status" in error &&
			typeof error.status === "number" &&
			!isAmbiguousDeliveryStatus(error.status)
		) {
			clearPendingContinuation(sessionId);
		}
		throw error;
	}
}
