/**
 * The session column's height budget — the one place the phone's bounded
 * regions agree on what they are bounded BY.
 *
 * Every capped region in the session column (the pending card, the todos and
 * subagent panels) has to be measured against that column, and `dvh` is not
 * that column. `screens/session-view.tsx` pins the column to
 * `visualViewport.height` in px so the composer stays above the iOS keyboard,
 * while `dvh` follows the DYNAMIC viewport — and a virtual keyboard is an
 * overlay, so it shrinks the first and leaves the second alone (`index.html`
 * sets no `interactive-widget`, so the spec default `resizes-visual` applies).
 * The two agree only while no keyboard is open.
 *
 * That divergence is not theoretical. A `60dvh` card stayed 468px tall inside a
 * column that had fallen to 480px on a 360x780 phone with a 300px keyboard, and
 * `send` — the only way to submit a free-text or secret answer — went under the
 * column's clipped foot (the column is `overflow-hidden`) with no gesture that
 * recovered it. Same arithmetic at 390x844 with a 260px keyboard.
 *
 * So the caps are written against `--lo-vvh`, which the session view publishes
 * from the SAME `visualViewport` handler that pins the column — one number, one
 * writer, so a cap cannot drift from the box it bounds. `100dvh` is the
 * fallback for a browser without `visualViewport` and for a surface outside the
 * column, which is the pre-pin behaviour rather than a new one.
 *
 * These are inline styles rather than Tailwind arbitrary values on purpose.
 * The fraction is a layout contract shared by three components, so it belongs
 * in one named constant instead of being retyped as a class string in each —
 * and a class string is also unassertable: the vitest layer runs under
 * happy-dom with no stylesheet, so `getComputedStyle` on a Tailwind class
 * returns "". A round-1 review shipped a broken cap past four assertions that
 * only compared class names. An inline style resolves, so a cap that stops
 * tracking the column fails a test.
 */

/** The custom property the session view publishes its pinned height as. */
export const COLUMN_HEIGHT_VAR = "--lo-vvh";

/**
 * A question awaiting an answer outranks a task list (branding §7), so the
 * card may claim more of the column than the panels beside it. Fractions
 * rather than pixels because the bound has to hold on every phone, not on the
 * one it was measured on.
 */
export const PENDING_CARD_FRACTION = 0.6;
export const PANEL_FRACTION = 0.4;

/** `max-height` in column units — see the module docstring for why not `dvh`. */
export function columnCap(fraction: number): { maxHeight: string } {
	return {
		maxHeight: `calc(var(${COLUMN_HEIGHT_VAR}, 100dvh) * ${fraction})`,
	};
}
