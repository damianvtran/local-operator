/**
 * The pin refusal's wording, in ONE place because two screens show it.
 *
 * The list shows it in the pin sheet that asked; the session view shows it in
 * a strip under its header (mobile UX batch 2, U2 — the sheet control there
 * used to flip back silently, telling the reader nothing). The sentence IS the
 * daemon's own (`no saved messages yet — pin it after you send one`), written
 * for this reader, so it passes through rather than being re-worded here — a
 * second copy of one refusal is how two surfaces end up describing the same
 * rule differently (the same rule `humanizeGateError` follows).
 *
 * The hygiene itself (the clamp, the bare-status rule, the sentence shape) is
 * shared with the resume refusal in `lib/refusal` — this module keeps the pin's
 * public names so its two call sites read as the pin's rule, and delegates.
 *
 * THE BARE-STATUS CASE IS NOT A REASON. `request` falls back to the status when
 * a failing response's body is not JSON, and `409` under a button the reader
 * just pressed explains nothing, so that spelling gets the plain line instead —
 * as does a failure that carried no message at all.
 *
 * A REFUSAL IS THE DAEMON'S SENTENCE AND NOTHING BOUNDS IT. It is an error
 * body, so a stack trace or a multi-line dump would run a sheet (or the session
 * view's strip) out of its column; the clamp keeps the opening, which is the
 * part that names the rule, and marks the cut rather than pretending the
 * message ended there.
 */

import {
	clampRefusalReason,
	refusalReason,
	refusalText,
	REFUSAL_REASON_MAX,
} from "./refusal";

export const PIN_REASON_MAX = REFUSAL_REASON_MAX;

export const clampPinReason = clampRefusalReason;

export const pinRefusalReason = refusalReason;

/** The one sentence every surface shows for a refused pin. */
export function pinRefusalText(error: unknown): string {
	return refusalText("Could not save the pin", error);
}
