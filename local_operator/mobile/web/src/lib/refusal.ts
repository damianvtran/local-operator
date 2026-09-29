/**
 * The refusal hygiene every refusal sentence shares.
 *
 * `Could not <act>: <the daemon's sentence>` is this product's one refusal
 * voice: the lead names the act the reader attempted (so the sentence cannot be
 * read as some other failure), and the rest is the daemon's own words, written
 * for this reader, so it passes through rather than being re-worded here — a
 * second copy of one refusal is how two surfaces end up describing the same
 * rule differently (the same rule `humanizeGateError` follows).
 *
 * The pin's spelling arrived first (mobile UX batch 2, U2) and lived alone in
 * `pin-refusal.ts`; the resume's arrived in round 1 of the same program's
 * remediation (D5), where the strip and the past-sessions screen both render
 * it. Two sentences, one hygiene — so the hygiene moved here and each act keeps
 * its own lead, which is the whole difference between them.
 *
 * THE BARE-STATUS CASE IS NOT A REASON. `request` falls back to the status when
 * a failing response's body is not JSON, and `409` under a button the reader
 * just pressed explains nothing, so that spelling gets the plain line instead —
 * as does a failure that carried no message at all.
 *
 * NOTHING BOUNDS A DAEMON SENTENCE. It is an error body, so a stack trace or a
 * multi-line dump would run a strip or a sheet out of its column; the clamp
 * keeps the opening, which is the part that names the rule, and marks the cut
 * rather than pretending the message ended there.
 */

/* The API's refusal type, imported for the one mapping in
   `resumeRefusalText`. */
import { HttpError } from "../api";

export const REFUSAL_REASON_MAX = 240;

export function clampRefusalReason(reason: string): string {
	return reason.length > REFUSAL_REASON_MAX
		? `${reason.slice(0, REFUSAL_REASON_MAX)}…`
		: reason;
}

export function refusalReason(error: unknown): string {
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return "the daemon did not say why";
	return message;
}

/** The one sentence shape: the act, then the daemon's own reason. */
export function refusalText(lead: string, error: unknown): string {
	return `${lead}: ${clampRefusalReason(refusalReason(error))}`;
}

/** The resume refusal — the session strip and the past-sessions list both paint
    it, so the lead lives here rather than at either call site.

    THE DAEMON'S 404 SENTENCE CANNOT BE SHOWN (UX round 2, U26). `no such past
    session: <id>` ends in the session's raw id, which reads as a second clause
    of machine notation — and the id is already on screen in the row this
    sentence sits under. The one thing the reader needs is that the transcript
    folder is gone, so the 404 gets one sentence in the reader's words instead
    of the daemon's. Every other status passes through untouched, as the shared
    hygiene says: the daemon writes those for this reader. */
export function resumeRefusalText(error: unknown): string {
	if (error instanceof HttpError && error.status === 404) {
		return "Could not resume: this session is no longer saved";
	}
	return refusalText("Could not resume", error);
}
