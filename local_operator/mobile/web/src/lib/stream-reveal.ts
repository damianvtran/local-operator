// The transport owns cadence. These are one-shot, paint-only effects on nodes
// React has already committed, not another flush, timer or source of text.
const REVEAL_ID = "lop-live-transcript";
const MOTION_QUERY = "(prefers-reduced-motion: no-preference)";
const revealed = new WeakSet<HTMLElement>();

export function allowsReveal(): boolean {
	return typeof window !== "undefined" && !!window.matchMedia?.(MOTION_QUERY).matches;
}

function reveal(node: HTMLElement | null, opacity: number): void {
	if (!node || revealed.has(node)) return;
	revealed.add(node);
	if (!node.animate || !allowsReveal()) return;
	const animation = node.animate([{ opacity }, { opacity: 1 }], {
		// The relay's existing fast motion token is 120 ms. No fill mode:
		// complete content is always the element's natural resting style.
		duration: 120,
		easing: "ease-out",
	});
	animation.id = REVEAL_ID;
}

export const revealText = (node: HTMLElement | null) => reveal(node, 0.48);
export const revealRow = (node: HTMLElement | null) => reveal(node, 0.65);

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
