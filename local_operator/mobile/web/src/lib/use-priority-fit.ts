/**
 * Priority fit: show as many optional chips as fit on ONE line, and collapse
 * the rest into a single overflow control.
 *
 * WHY THIS EXISTS (operator finding B, review round 1). The list's filter strip
 * used to be an `overflow-x: auto` rail: tags past the screen edge were reached
 * by scrolling the strip sideways, a gesture most phone users never try and the
 * operator called out as bad practice. A phone view must not scroll sideways at
 * all, so the strip now keeps its always-present filters (All, Needs you,
 * Unread) and as many tags as genuinely fit, and the remainder sits behind ONE
 * `+N` control that opens a sheet listing every tag — still one tap away.
 *
 * CONSTRAINTS this shape honours:
 * - First frame = final layout. Widths are measured in a layout effect, so the
 *   one "render everything" measuring pass happens before paint and is never
 *   seen; later passes reuse cached widths.
 * - The SELECTED item is never hidden: a filter that is in force but invisible
 *   would make the list look broken. It takes the last visible slot instead.
 * - No layout engine (the happy-dom test host) measures 0 px, which caches
 *   nothing, so every item renders — the behaviour the tests have always seen.
 *
 * `fitPriority` is the pure rule (unit-tested); `usePriorityFit` only measures.
 */
import { useCallback, useEffect, useLayoutEffect, useState } from "react";

export type FitInput = {
	/** Inline space the row offers its children (content box, CSS px). */
	available: number;
	/** Width already spent by the always-shown items, including their gaps. */
	fixed: number;
	/** The row's column gap. */
	gap: number;
	/** Width of the overflow control that replaces the hidden items. */
	overflow: number;
	/** Index of an item that must stay visible (the selected one), or -1. */
	must: number;
};

/** Indices of the optional items to show, in their original order. */
export function fitPriority(widths: readonly number[], input: FitInput): number[] {
	const { available, fixed, gap, overflow, must } = input;
	const lead = fixed > 0 ? gap : 0;
	const everything = widths.reduce((sum, width) => sum + gap + width, fixed) - (fixed > 0 ? 0 : gap);
	if (widths.length === 0 || everything <= available) return widths.map((_, index) => index);
	// Room left for optional items once the overflow control has its slot.
	const budget = available - fixed - lead - overflow - gap;
	const shown: number[] = [];
	let spent = 0;
	for (const [index, width] of widths.entries()) {
		if (spent + width + gap > budget) break;
		shown.push(index);
		spent += width + gap;
	}
	if (must >= 0 && !shown.includes(must)) {
		// Make room for the selected item from the END of the visible run, so the
		// leading items (the user's own order) are the last thing given up.
		while (shown.length && spent + widths[must] + gap > budget) {
			spent -= widths[shown.pop() as number] + gap;
		}
		shown.push(must);
	}
	return shown;
}

const OVERFLOW_KEY = "\u0000overflow";
/** Measured before the overflow control first renders; replaced once it has. */
const OVERFLOW_ESTIMATE = 44;

export function usePriorityFit(keys: readonly string[], mustKey: string | null) {
	const [row, setRow] = useState<HTMLElement | null>(null);
	const [widths] = useState(() => new Map<string, number>());
	const [shown, setShown] = useState<readonly string[] | null>(null);
	const [, setTick] = useState(0);
	const rowRef = useCallback((node: HTMLElement | null) => setRow(node), []);

	useLayoutEffect(() => {
		if (!row) return;
		for (const element of row.querySelectorAll<HTMLElement>("[data-fit-key]")) {
			if (element.offsetWidth > 0) widths.set(element.dataset.fitKey as string, element.offsetWidth);
		}
		const overflowElement = row.querySelector<HTMLElement>("[data-fit-overflow]");
		if (overflowElement && overflowElement.offsetWidth > 0) widths.set(OVERFLOW_KEY, overflowElement.offsetWidth);
		// An item never measured forces one render-everything pass (pre-paint).
		if (keys.some((key) => !widths.has(key))) {
			setShown((previous) => (previous === null ? previous : null));
			return;
		}
		const style = getComputedStyle(row);
		const gap = Number.parseFloat(style.columnGap) || 0;
		const available =
			row.clientWidth -
			(Number.parseFloat(style.paddingInlineStart) || 0) -
			(Number.parseFloat(style.paddingInlineEnd) || 0);
		const fixedElements = [...row.querySelectorAll<HTMLElement>("[data-fit-fixed]")];
		const fixed =
			fixedElements.reduce((sum, element) => sum + element.offsetWidth, 0) +
			gap * Math.max(0, fixedElements.length - 1);
		const next = fitPriority(
			keys.map((key) => widths.get(key) as number),
			{
				available,
				fixed,
				gap,
				overflow: widths.get(OVERFLOW_KEY) ?? OVERFLOW_ESTIMATE,
				must: mustKey === null ? -1 : keys.indexOf(mustKey),
			},
		).map((index) => keys[index]);
		setShown((previous) =>
			previous && previous.length === next.length && previous.every((key, index) => key === next[index])
				? previous
				: next,
		);
	});

	// A rotation or a resized column changes the room; re-run the fit.
	useEffect(() => {
		if (!row || typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(() => setTick((tick) => tick + 1));
		observer.observe(row);
		return () => observer.disconnect();
	}, [row]);

	const visible = shown ?? keys;
	return { rowRef, visible, hidden: keys.length - visible.length };
}
