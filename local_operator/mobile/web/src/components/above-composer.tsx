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
 *  - THE SLOT SCROLLS; ITS ITEMS DO NOT SHRINK (QA round 1, Q2). On a short
 *    column the deficit has to come from somewhere. The transcript (`flex-1`,
 *    basis 0) is already at 0 by then, and the composer keeps its content
 *    height, so it lands on this slot. With only `min-h-0` here the slot shrank
 *    below its children and they kept painting past its box: [redacted] at
 *    320x440 with the card, the tray and the composer's retry alert, the card
 *    was squeezed to its 20 px padding and the composer covered the tray and
 *    the card's Approve by 43 px. So the slot is the scroller (`min-h-0` +
 *    `overflow-y-auto`, clipped at its own edge, never sideways) and every
 *    item keeps its own height (`*:shrink-0`; the card's column cap still
 *    bounds it, and its body still scrolls inside that cap). Nothing above the
 *    composer can then overlap it: the worst case is a slot showing the top of
 *    its first item, scrollable to the rest.
 *  - NO MASK (no `lo-fade-*` cue). Sheets render IN PLACE, never portalled
 *    (`ui/sheet.tsx`), and the card mounts its pairing sheet inside this slot;
 *    `mask-image` on an ancestor clips even a fixed-position descendant to the
 *    ancestor's box, where `overflow` does not.
 *
 * Children own their insides, never their outside: no margins and no side
 * padding on an item placed here. Renders nothing when it has nothing, so an
 * empty stack adds no 8 px of its own.
 */
import type { ReactNode } from "react";

export const ABOVE_COMPOSER_STACK =
	"lo-scroll flex min-h-0 flex-col gap-2 overflow-y-auto overflow-x-hidden px-2 pb-2 *:shrink-0";

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
