/** Safari without scroll timelines still gets an honest bottom-edge cue.
 * Chromium's scroll-linked CSS needs no observer or per-scroll React render;
 * unsupported engines measure only the boolean 'content remains below'. A
 * mutation observer covers streamed text and arriving rows, not just resizing
 * the viewport. Callback-ref cleanup scopes listeners to the actual scroller
 * lifetime, including sheets that mount long after their owner component. */
import { useCallback } from "react";

export function useScrollCues(inline = false) {
	return useCallback((node: HTMLElement | null) => {
		if (!node || (typeof CSS !== "undefined" && CSS.supports?.("animation-timeline", "scroll()"))) return;
		const measure = () => {
			const remaining = inline
				? node.scrollWidth - node.clientWidth - Math.abs(node.scrollLeft)
				: node.scrollHeight - node.clientHeight - node.scrollTop;
			node.dataset.moreBelow = String(remaining > 1);
		};
		measure();
		node.addEventListener("scroll", measure, { passive: true });
		const resize = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
		resize?.observe(node);
		const mutation = new MutationObserver(measure);
		mutation.observe(node, { childList: true, subtree: true, characterData: true });
		return () => {
			node.removeEventListener("scroll", measure);
			resize?.disconnect();
			mutation.disconnect();
		};
	}, [inline]);
}
