import { useEffect, type RefObject } from "react";
import { markSessionSeen } from "./api";
import type { SessionProjection } from "./types";

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
		const check = () => {
			if (cancelled || pending || acknowledged ||
				document.visibilityState !== "visible" || !document.hasFocus()) return;
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
					// effect and is acknowledged on its own.
					if (answer?.attention?.unseen === false) acknowledged = true;
				})
				.catch(() => {
					// No optimistic clear, and no latch: a refused receipt (a token the
					// backend has superseded, which it now answers with 409) leaves the
					// authoritative list state intact and the poll running.
				})
				.finally(() => { pending = false; });
		};
		const timer = window.setInterval(check, 500);
		const frame = requestAnimationFrame(check);
		return () => {
			cancelled = true;
			clearInterval(timer);
			cancelAnimationFrame(frame);
		};
	}, [sessionId, projection?.session_id, projection?.streaming, attention?.completion_token,
		attention?.anchor_id, attention?.unseen, attention?.conversation_id, blocked, root]);
}
