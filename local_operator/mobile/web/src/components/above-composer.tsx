/**
 * THE ABOVE-COMPOSER STACK: the one slot for everything stacked between the
 * transcript and the composer card (the pending approval/ask card and the
 * status tray today).
 *
 * WHY ONE SLOT. Each of those used to choose its own padding — the card
 * `mx-2`, the tray `px-3 py-1.5`, the ended and degraded bands full-bleed with
 * `px-2`/`px-3` — so nothing shared a vertical rhythm and the ended band sat
 * flush on the composer's rounded top edge (operator report, phone Safari).
 * docs/design/mobile-relay-guidelines.md ("Composer and startup") forbids that
 * kind of stacked independent padding; this slot is the rule it names.
 *
 * THE NUMBERS, and what each is aligned to:
 *  - `px-2` (8 px): the composer card's own outer inset (`COMPOSER_SHELL`'s
 *    `mx-2`), so a card or tray pill's edge lines up with the composer's edge.
 *  - `gap-2` (8 px) between items: the transcript's own row gap (`gap-2`).
 *  - `pb-2` (8 px) to the composer's top edge: the same gap again.
 *  - NO top padding: the transcript scroller's `py-2` already leaves 8 px under
 *    its last line, so the rhythm above the first item is that same 8 px rather
 *    than a doubled 16.
 *  - `min-h-0` keeps the pending card able to yield on a short column (its own
 *    `min-h-0`, "THE CARD YIELDS"); a child that must not shrink, like the
 *    tray, says `shrink-0` itself.
 *
 * Children own their insides, never their outside: no margins and no side
 * padding on an item placed here. Renders nothing when it has nothing, so an
 * empty stack adds no 8 px of its own.
 */
import type { ReactNode } from "react";

export const ABOVE_COMPOSER_STACK = "flex min-h-0 flex-col gap-2 px-2 pb-2";

export function AboveComposerStack({ children }: { children: ReactNode }) {
	const items = (Array.isArray(children) ? children : [children]).filter(
		(child) => child !== null && child !== undefined && child !== false,
	);
	if (items.length === 0) return null;
	return (
		<div data-above-composer="" className={ABOVE_COMPOSER_STACK}>
			{children}
		</div>
	);
}
