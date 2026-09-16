import { useEffect, type RefObject } from "react";
import { markSessionSeen } from "./api";
import type { SessionProjection } from "./types";

/** Poll cadence while the completion has not been acknowledged. */
const CHECK_MS = 500;
/** Consecutive refusals before the retry cadence starts backing off. */
const FAILURES_BEFORE_BACKOFF = 3;
/** Ceiling on the backed-off cadence: ~1 attempt/minute, not ~7,200/hour. */
const MAX_BACKOFF_MS = 60_000;

/** Mounting, subscribing and rendering offscreen history are not evidence of a
 * read. Sample the committed result while it is uncovered in the focused tab.
 * The effect captures identity and token together; navigation cancels its work.
 * An acknowledgement is believed only when its ANSWER says the conversation is
 * read (`unseen: false`) -- see the verification in `check` below.
 */
export function useCompletionView(
	sessionId: string,
	projection: SessionProjection | null,
	root: RefObject<HTMLDivElement | null>,
	blocked: boolean,
) {
	const attention = projection?.attention;
	useEffect(() => {
		if (blocked || projection?.streaming || !attention?.unseen ||
			!attention.completion_token || !attention.anchor_id ||
			projection?.session_id !== sessionId ||
			attention.conversation_id !== `session/${sessionId}`) return;
		const token = attention.completion_token;
		const anchor = attention.anchor_id;
		let cancelled = false;
		let pending = false;
		let acknowledged = false;
		/** Consecutive refusals, and the earliest time the next attempt may run. */
		let refusals = 0;
		let nextAttempt = 0;
		// Count refusals AND unresolved 2xx together; version skew or alternating
		// outcomes must not buy a fresh budget for the same rendered token.
		const unresolved = (reason: unknown) => {
			refusals += 1;
			if (refusals === FAILURES_BEFORE_BACKOFF) console.warn(
				`[attention] could not mark ${sessionId} read after ${refusals} attempts; backing off`, reason,
			);
			if (refusals >= FAILURES_BEFORE_BACKOFF) nextAttempt = Date.now() +
				Math.min(MAX_BACKOFF_MS, CHECK_MS * 2 ** (refusals - FAILURES_BEFORE_BACKOFF + 1));
		};
		const check = () => {
			if (cancelled || pending || acknowledged ||
				document.visibilityState !== "visible" || !document.hasFocus() ||
				Date.now() < nextAttempt) return;
			const element = root.current?.querySelector<HTMLElement>(
				`[data-completion-anchor="${CSS.escape(anchor)}"][data-completion-complete="true"]`,
			);
			if (!element) return;
			const rect = element.getBoundingClientRect();
			// The end of a long result must be visible, not just its first line.
			const x = rect.left + rect.width / 2;
			const y = rect.bottom - 2;
			if (rect.height <= 0 || x < 0 || x >= innerWidth || y < 0 || y >= innerHeight) return;
			const top = document.elementFromPoint(x, y);
			if (!top || !element.contains(top)) return;
			pending = true;
			void markSessionSeen(sessionId, token)
				.then((answer) => {
					// VERIFY, never assume. A resolved call is not a read: `unseen` is
					// the whole verdict, and an older daemon answered a superseded
					// token with a 200 whose state still said `unseen` -- latching here
					// on the resolution is what left the "new" mark on for good.
					// Anything else keeps polling, so a fresh token re-runs this
					// effect and is acknowledged on its own. Identity is part of the
					// verdict rather than assumed, exactly as in the desktop twin: an
					// answer about another conversation settles nothing here.
					const settled = answer?.attention;
					if (settled?.unseen === false &&
						settled.conversation_id === `session/${sessionId}` &&
						settled.completion_token === token) acknowledged = true;
					else unresolved("answer did not settle the rendered completion");
				})
				.catch((error: unknown) => unresolved(error))
				.finally(() => { pending = false; });
		};
		const timer = window.setInterval(check, CHECK_MS);
		const frame = requestAnimationFrame(check);
		return () => {
			cancelled = true;
			clearInterval(timer);
			cancelAnimationFrame(frame);
		};
	}, [sessionId, projection?.session_id, projection?.streaming, attention?.completion_token,
		attention?.anchor_id, attention?.unseen, attention?.conversation_id, blocked, root]);
}
