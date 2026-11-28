import { type RefObject, useEffect, useReducer, useRef } from "react";
import { PacedReveal, TICK_MS } from "./paced-reveal";
import { allowsReveal } from "./stream-reveal";

const clock = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** True while the reader has a range selected inside `root`. Text that grows
 *  under a live selection moves it; pacing steps aside rather than fight it
 *  (the existing `SelectionGuard` only protects the offsets of an append). */
function selectsWithin(root: Element | null): boolean {
	const selection = root?.ownerDocument.getSelection();
	if (!root || !selection || selection.isCollapsed || selection.rangeCount === 0) return false;
	for (let i = 0; i < selection.rangeCount; i++) {
		if (selection.getRangeAt(i).intersectsNode(root)) return true;
	}
	return false;
}

/**
 * Paced reveal of one streamed string, local to the ONE component that draws it.
 *
 * SCOPE IS THE POINT. The pacing state lives in the streaming message's own
 * `Markdown`, not in the transcript or the screen: a tick re-renders that
 * message (and, through `memo`, only its unsettled last block), never the
 * settled rows or the composer. There is ONE timer per streaming message and
 * only while something is still held back: a settled or caught-up answer holds
 * no timer.
 *
 * `paced` says whether this message is in a continuous live stream
 * (`Transcript`'s `motion`). It is the only switch: when it is false (a final
 * row, a reconnect seed, a correction, reduced motion) the received text is
 * returned unchanged and any held backlog is released, so the settled text is
 * byte-identical to what arrived.
 *
 * `hold` initial text only on the FIRST render of an instance. A row that
 * appears with its stream is paced from its first word; a cell that mounts
 * mid-stream, or a `paced` that flips on later, shows what exists and paces only
 * what arrives afterwards (replaying text the reader has seen would be worse
 * than a chunk).
 *
 * SAFE TO LOSE THE TIMER: the interval is cleared on unmount and whenever the
 * backlog is empty; a hidden tab (timers throttle to ~1 Hz there) releases the
 * backlog at once on the next tick, and `MAX_LAG_MS` bounds any held text
 * regardless of how late a tick runs.
 */
export function usePacedText(text: string, paced: boolean, root: RefObject<Element | null>): string {
	const [, render] = useReducer((version: number) => version + 1, 0);
	const held = useRef<PacedReveal | null>(null);
	const mounted = useRef(false);
	// RENDER COALESCING (<= 1 render per tick). A snapshot re-renders this
	// component with new text on its own schedule (the store's rAF flush), and a
	// tick then renders again ~34 ms later: ~75 + 30 renders/s stacked. `drawn` is
	// true when a render the PARENT caused has happened since the last tick
	// looked; the tick then advances the cursor but leaves the paint to the next
	// tick (`owed`) instead of stacking a second render in the same period.
	// `asked` marks the render the tick itself requested so it is not mistaken
	// for a parent's (which would make every other tick skip). Both are written
	// during render on purpose and are idempotent flags, so a discarded or
	// replayed render can at worst delay one paint by one tick.
	const drawn = useRef(false);
	const asked = useRef(false);
	const owed = useRef(false);
	if (asked.current) asked.current = false;
	else drawn.current = true;
	owed.current = false;
	const enabled = paced && allowsReveal();
	if (enabled && held.current === null) {
		held.current = new PacedReveal(text, clock(), !mounted.current);
	}
	mounted.current = true;
	const reveal = held.current;
	if (reveal !== null) {
		// Idempotent render-time writes: `push` is a no-op for equal text and a
		// PacedReveal only moves forward, so a discarded or replayed render
		// (StrictMode, concurrent) leaves it correct.
		reveal.push(text, clock());
		if (!enabled) reveal.flush();
	}
	if (!enabled) held.current = null;
	const running = enabled && reveal !== null && !reveal.idle();

	useEffect(() => {
		if (!running || reveal === null) return;
		// The render that started this timer is a whole tick old by the first tick.
		drawn.current = false;
		const timer = setInterval(() => {
			// A flush is a correctness event (hidden tab, selection): paint it now.
			if (document.hidden || selectsWithin(root.current)) {
				reveal.flush();
				asked.current = true;
				render();
				return;
			}
			const moved = reveal.tick(clock());
			if (moved || owed.current) {
				if (drawn.current) owed.current = true;
				else { owed.current = false; asked.current = true; render(); }
			}
			drawn.current = false;
		}, TICK_MS);
		return () => clearInterval(timer);
	}, [running, reveal, root]);

	return reveal === null ? text : reveal.shown();
}
