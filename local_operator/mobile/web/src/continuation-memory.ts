/*
 * In-memory home for a retry envelope that localStorage could not hold.
 *
 * WHY IT EXISTS. The retry envelope (`continuation-command.ts`) is persisted so a
 * reload keeps the exact bytes of an instruction whose delivery is unknown. But
 * localStorage is ~5 M characters PER ORIGIN on iOS Safari (measured: a 5.0 M
 * value stores, 5.3 M throws `QuotaExceededError`), and one message with two
 * ordinary screenshots is ~2.9 M characters. A second image-bearing envelope —
 * another conversation's still-pending send — therefore cannot be stored. That
 * used to abort the send BEFORE any request: the composer printed its catch-all
 * "Couldn't continue this conversation. Try again." with no Retry control, and
 * every retry failed the same way until the other envelope cleared.
 *
 * WHAT IT IS. A module-level map holding the same `{ version, saved_at, envelope }`
 * record, consulted when localStorage has no copy. The send goes ahead and an
 * in-page retry keeps its UUID. What is lost is only persistence across a reload,
 * which is the cost of the browser's quota and strictly better than refusing to
 * send. It is content-bearing, so `clearPrivateSessionStorage` empties it too.
 *
 * It lives in its own module because `private-storage` (imported by `api`) has to
 * reach it and `continuation-command` imports `api`: a shared leaf avoids the cycle.
 */

export interface HeldContinuation {
	version: 1;
	saved_at: number;
	envelope: unknown;
}

const held = new Map<string, HeldContinuation>();

export function holdContinuation(key: string, record: HeldContinuation, limit: number): void {
	held.delete(key);
	held.set(key, record);
	/* Map iteration is insertion order, so the first key is the oldest hold. */
	while (held.size > limit) {
		const oldest = held.keys().next().value;
		if (oldest === undefined) break;
		held.delete(oldest);
	}
}

export function heldContinuation(key: string): HeldContinuation | undefined {
	return held.get(key);
}

export function releaseContinuation(key: string): void {
	held.delete(key);
}

export function clearHeldContinuations(): void {
	held.clear();
}
