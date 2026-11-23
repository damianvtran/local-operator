/**
 * How tall the inline drawing may grow before it is clamped.
 *
 * Why a cap AND a tolerance: Pele reflows a wide left-to-right chart top to bottom
 * under 640 px, so a six-node `flowchart LR` is a 232 x 526 drawing on a phone. The
 * old flat 32 rem (512 px) cap sliced the last node off a chart that small, with
 * only a faint fade to hint at it. A cap is for runaway drawings, so a modest
 * overflow is shown whole: content up to CAP + TOLERANCE is never clipped, and only
 * a drawing taller than that is held at CAP (still behind the fade, now with a
 * labelled hint). Clipping by less than the tolerance would hide a node or two to
 * save a few lines of scroll, which is the wrong trade.
 */
export const INLINE_CAP_REM = 40;
export const INLINE_TOLERANCE_REM = 6;

/** Whether a drawing whose natural height (padding included) is `contentPx` must be clamped. */
export function mustClamp(contentPx: number, remPx: number): boolean {
	return contentPx > (INLINE_CAP_REM + INLINE_TOLERANCE_REM) * remPx;
}

/** The `max-height` to apply: the cap when clamped, else the cap plus tolerance (which fits the content). */
export function inlineMaxHeight(clamped: boolean): string {
	return `${clamped ? INLINE_CAP_REM : INLINE_CAP_REM + INLINE_TOLERANCE_REM}rem`;
}
