// The transport owns cadence. These are one-shot, paint-only effects on nodes
// React has already committed, not another flush, timer or source of text.
const REVEAL_ID = "lop-live-transcript";
const MOTION_QUERY = "(prefers-reduced-motion: no-preference)";
const revealed = new WeakSet<HTMLElement>();

export function allowsReveal(): boolean {
	return typeof window !== "undefined" && !!window.matchMedia?.(MOTION_QUERY).matches;
}

function reveal(node: HTMLElement | null, settled?: () => void): void {
	if (!node || revealed.has(node)) return;
	revealed.add(node);
	if (!node.animate || !allowsReveal()) { settled?.(); return; }
	// Keep the first paint legible while still distinguishing a fresh suffix.
	const animation = node.animate([{ opacity: 0.65 }, { opacity: 1 }], {
		// The relay's existing fast motion token is 120 ms. No fill mode:
		// complete content is always the element's natural resting style.
		duration: 120,
		easing: "ease-out",
	});
	animation.id = REVEAL_ID;
	// Cancellation (final, correction or reduced motion) retires the same
	// wrapper as a finished effect, so neither leaves a second text tree.
	if (settled) void animation.finished.then(settled, settled);
}

export const revealText = (node: HTMLElement | null, settled?: () => void) => reveal(node, settled);
export const revealRow = (node: HTMLElement | null) => reveal(node);

export function settleReveals(root: Element | null, subtree = true): void {
	for (const animation of root?.getAnimations?.({ subtree }) ?? []) {
		if (animation.id === REVEAL_ID) animation.cancel();
	}
}

export function listenForReducedMotion(root: () => Element | null): (() => void) | undefined {
	const query = window.matchMedia?.(MOTION_QUERY);
	if (!query?.addEventListener) return;
	const change = () => { if (!query.matches) settleReveals(root()); };
	query.addEventListener("change", change);
	return () => query.removeEventListener("change", change);
}
