/**
 * One-shot "focus this session's composer when it opens" flags.
 *
 * WHY A MODULE-LEVEL ONE-SHOT, and not a route parameter or a context.
 *
 * The list's one-tap start navigates the moment the daemon answers, and the
 * composer that must receive focus belongs to the SESSION SCREEN — one route
 * later, and only mounted once its projection arrives. Nothing in the React
 * tree connects the two, and the intent is not part of the route: `#/s/<id>`
 * means the same thing whether it was reached by the new-session tap or by
 * opening an existing conversation, and only the first may steal focus (a
 * programmatic focus on iOS pops the keyboard, and doing that over a
 * conversation the reader deliberately reopened is hostile).
 *
 * The flag must therefore be consumed EXACTLY ONCE, by the session view, and
 * must NOT survive a page reload: after a reload the flag is gone from memory,
 * which is precisely the wanted behaviour — a reload after the flag was
 * consumed is an ordinary navigation and the composer stays quiet.
 */
const pending = new Set<string>();

/** Remember that ``sessionId``'s composer should take focus when it mounts. */
export function markPendingFocus(sessionId: string): void {
	if (sessionId) pending.add(sessionId);
}

/**
 * Take the flag for ``sessionId`` if it is still there, consuming it.
 *
 * Returns true at most once per :func:`markPendingFocus` call, whatever
 * happens to the component in between — that is what makes the focus a
 * ONE-SHOT rather than "focus on every mount of this session".
 */
export function consumePendingFocus(sessionId: string): boolean {
	return pending.delete(sessionId);
}
