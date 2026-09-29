/**
 * The transcript's single follow decision (round 3; extended in round 4).
 *
 * The scroller carries `overflow-anchor: none`, so ONE hand owns `scrollTop`:
 * with native scroll anchoring left on, Chrome and the component's own write
 * each compensated the reserve's insertion and a mid-history reader moved by
 * the rung's height (round 3, U27 = reviewer MAJOR 1 = QA Q1). That hand has
 * to answer for EVERY change above the reader, not just the reserve — the
 * live window drops its oldest row from the top on each append at the cap
 * (round 4, U28: the platform used to cover that removal silently), a page of
 * older rows can prepend, `show N more loaded` can expand the window, and the
 * reserve itself can grow or clear. The caller measures what the DOM did
 * above the reader, with the user's own scrolling divided out, and hands the
 * numbers here.
 *
 * Most of the time the rule is "hold this row": return `scrollTop + domDelta`.
 * Two positions have a definition that is not "hold this row", and neither
 * takes a write:
 *
 * - AT THE TAIL, the reader is defined by the bottom. When content above
 *   shrinks below the scroll position the browser clamps `scrollTop` to the
 *   new max before this runs (round 4: the reserve CLEARING at the tail), so
 *   the viewport has already followed and applying the delta again would
 *   subtract the same height twice — the reader would be left precisely
 *   `domDelta` px above the fold with the newest row cut off. The clamp
 *   followed; leave it.
 * - AT THE TOP with the reserve's own edge (`reserveEdge`), the slide is the
 *   POINT: the first row comes out from under the strip with no gesture
 *   (round 2, U23 = D7), and `scrollTop` has no room to move up anyway. A
 *   content change that is NOT the reserve's edge still holds its row there —
 *   that is the older-rows prepend's contract.
 *
 * Returns the new `scrollTop`, or null to leave it alone.
 */
export function followScrollTop(input: {
	/** What the DOM did above the reader, in px, positive = the row moved down
	    (content above it grew). 0 means the reader's row is exactly where the
	    last measurement left it, and nothing is written. */
	domDelta: number;
	/** The scroller's `scrollTop` now (after the browser's own clamp). */
	scrollTop: number;
	/** The scroller's `scrollHeight - clientHeight` now. */
	max: number;
	/** Whether this change carried the reserve's edge (`topInset` changed). */
	reserveEdge: boolean;
}): number | null {
	if (input.domDelta === 0) return null;
	/* Nothing scrolls: the reveal happens by the content moving down, and the
	   write would be a no-op clamped to 0. */
	if (input.max <= 0) return null;
	/* The tail: the browser's clamp has already followed the shrink. */
	if (input.max - input.scrollTop <= 1) return null;
	/* The top with the reserve's edge: the slide is the reveal. */
	if (input.reserveEdge && input.scrollTop <= 0) return null;
	return Math.max(0, Math.min(input.scrollTop + input.domDelta, input.max));
}
