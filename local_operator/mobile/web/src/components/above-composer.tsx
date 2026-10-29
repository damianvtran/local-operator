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
 *  - `mb-2` (8 px) to the composer's top edge: the same gap again, as a
 *    MARGIN, outside the scrollport (design round 2, D6 = review R2-1). As
 *    `pb-2` it sat at the end of the scrolled content, so an overflowing slot
 *    rested flush on the composer (slot bottom 144.6 = composer top 144.6 at
 *    320x440) and the card's cut edge fused into it, both on `elevated`.
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
 *  - AN OVERFLOWING SLOT RESTS ON THE CARD'S DECISION (same round). With the
 *    card, the tray and the retry alert at 320x440 the slot's 65 px viewport
 *    showed only the card's head; its Approve sat below a fold with no cue.
 *    Resting at the slot's raw END is not the answer: [redacted], that shows
 *    the tray, whose "Approve" chip opens the approvals-MODE sheet and does
 *    not answer this request, while the card's own Approve/Deny row stays
 *    above the fold. So the slot brings the card's `data-slot-rest` row (the
 *    row that answers) to its bottom edge, as the card mounts (`restOn`, the
 *    card's identity) and whenever the slot's own box is resized (the keyboard
 *    opening is what makes it overflow). A user's scroll never resizes the
 *    box, so it is never fought; a slot that fits is never moved.
 *
 * Children own their insides, never their outside: no margins and no side
 * padding on an item placed here. Renders nothing when it has nothing, so an
 * empty stack adds no 8 px of its own.
 */
import { useLayoutEffect, useRef, type ReactNode } from "react";

export const ABOVE_COMPOSER_STACK =
	"lo-scroll flex min-h-0 flex-col gap-2 overflow-y-auto overflow-x-hidden px-2 mb-2 *:shrink-0";

export function AboveComposerStack({
	children,
	restOn = null,
}: {
	children: ReactNode;
	/** The identity of an action card in the slot (null when none): while set,
	    an overflowing slot rests with that card's decision row in view. */
	restOn?: string | null;
}) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		const node = ref.current;
		if (!node || !restOn) return;
		const rest = () => {
			const row = node.querySelector<HTMLElement>("[data-slot-rest]");
			if (!row || node.scrollHeight <= node.clientHeight) return;
			const slot = node.getBoundingClientRect();
			/* A long ask's first option can be below the CARD'S inner scroller,
			   not merely below the slot. Reveal it there first; otherwise the
			   slot measures an unclipped row and rests at its end while the
			   decision remains hidden. Only a scroller whose content actually
			   overflows is moved. */
			for (let ancestor = row.parentElement; ancestor && ancestor !== node; ancestor = ancestor.parentElement) {
				if (ancestor.scrollHeight <= ancestor.clientHeight + 1) continue;
				const box = ancestor.getBoundingClientRect();
				const inside = row.getBoundingClientRect();
				if (inside.top < box.top) ancestor.scrollTop += inside.top - box.top;
				else if (inside.bottom > box.bottom) ancestor.scrollTop += inside.bottom - box.bottom;
			}
			const visible = row.getBoundingClientRect();
			// Already whole in view: leave the reader's position alone.
			if (visible.top >= slot.top && visible.bottom <= slot.bottom) return;
			// The row's bottom on the slot's bottom edge, so the card's head
			// stays as close above it as the slot allows.
			node.scrollTop += visible.bottom - slot.bottom;
		};
		rest();
		if (typeof ResizeObserver === "undefined") return;
		const resize = new ResizeObserver(rest);
		resize.observe(node);
		return () => resize.disconnect();
	}, [restOn]);
	const items = (Array.isArray(children) ? children : [children]).filter(
		(child) => child !== null && child !== undefined && child !== false,
	);
	if (items.length === 0) return null;
	return (
		<div ref={ref} data-above-composer="" className={ABOVE_COMPOSER_STACK}>
			{children}
		</div>
	);
}
