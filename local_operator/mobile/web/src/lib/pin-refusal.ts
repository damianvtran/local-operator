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

export const PIN_REASON_MAX = 240;

export function clampPinReason(reason: string): string {
	return reason.length > PIN_REASON_MAX
		? `${reason.slice(0, PIN_REASON_MAX)}…`
		: reason;
}

export function pinRefusalReason(error: unknown): string {
	const message = error instanceof Error ? error.message : String(error);
	if (message === "" || /^\d{3}$/.test(message)) return "the daemon did not say why";
	return message;
}

/** The one sentence every surface shows for a refused pin. */
export function pinRefusalText(error: unknown): string {
	return `Could not save the pin: ${clampPinReason(pinRefusalReason(error))}`;
}
