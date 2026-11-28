// The transport owns cadence. These are one-shot, paint-only effects on nodes
// React has already committed, not another flush, timer or source of text.
const REVEAL_ID = "lop-live-transcript";
const MOTION_QUERY = "(prefers-reduced-motion: no-preference)";
const revealed = new WeakSet<HTMLElement>();

export function allowsReveal(): boolean {
	return typeof window !== "undefined" && !!window.matchMedia?.(MOTION_QUERY).matches;
}

/** A whole arriving row (tool card, user echo): a light, quick settle. */
const ROW_FADE = { from: 0.65, duration: 120 } as const;
/**
 * Appended STREAMING TEXT: a deep, longer ramp. Words are released a few at a
 * time by `paced-reveal.ts`, so each suffix is small and several overlap; with
 * the old 0.65 -> 1 over 120 ms the ramp spanned ~10 frames on the 120 Hz bench display, at a
 * barely visible contrast, which read as "chunks just appear". Text
 * starting nearly transparent and settling over ~240 ms is clearly a fade, and
 * the overlap of consecutive suffixes is what makes it read as a flow. No fill
 * mode: complete content is always the element's natural resting style.
 */
const TEXT_FADE = { from: 0.15, duration: 240 } as const;

function reveal(node: HTMLElement | null, fade: { from: number; duration: number }, settled?: () => void): void {
	if (!node || revealed.has(node)) return;
	revealed.add(node);
	if (!node.animate || !allowsReveal()) { settled?.(); return; }
	const animation = node.animate([{ opacity: fade.from }, { opacity: 1 }], {
		duration: fade.duration,
		easing: "ease-out",
	});
	animation.id = REVEAL_ID;
	// Cancellation (final, correction or reduced motion) retires the same
	// wrapper as a finished effect, so neither leaves a second text tree.
	if (settled) void animation.finished.then(settled, settled);
}

export const revealText = (node: HTMLElement | null, settled?: () => void) => reveal(node, TEXT_FADE, settled);
export const revealRow = (node: HTMLElement | null) => reveal(node, ROW_FADE);

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
