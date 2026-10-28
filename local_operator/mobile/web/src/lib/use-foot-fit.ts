/**
 * Foot fit: the composer's facts row (queued count, working folder, model,
 * usage reading) gives way in a fixed PRIORITY ORDER instead of overflowing.
 *
 * WHY THIS EXISTS (review round 1 on #3030, R1-2). In the short layout the
 * facts share one line with attach, the mic and the stop/send pair, and take
 * only the width those leave (`basis-0 flex-1 min-w-0`). [redacted] at 320 px
 * while streaming, that leftover is 156 px, or 110 px with the mic: less than
 * folder 44 + model floor 44 + the 96 px reading. With nothing giving way the
 * reading painted UNDER the stop button, two controls in one hit band. The row
 * now clips at its own edge (`overflow-x-clip`), so it can never paint under
 * the pair whatever its content does. This hook makes sure that clip never
 * has to cut anything, by degrading in this order:
 *
 *   0. everything at full width; the model label is the elastic element and
 *      truncates first (down to its 44 px floor), as it always has;
 *   1. the usage reading compacts to ONE whole figure (context %, else spend):
 *      never a clipped amount (design D1 of the composer redesign);
 *   2. the reading yields entirely: it is a glance, and its full pair returns
 *      the moment the keyboard closes and the facts get their own line. The
 *      ⋮ menu then carries a `Spend & context` row, so the spend sheet keeps a
 *      door (session-view's `Header` gates and latches that row);
 *   3. nothing else can yield and there is no count to move: the model chip is
 *      pinned at its 44 px floor showing the label's first characters (design
 *      D10 — a bare glyph did not say "model"). Only rows narrower than folder
 *      + chip (below ~300 px) get here, and the row's clip is the backstop;
 *   4. the QUEUED COUNT leaves the row, whole, for a plain line in the ⋮ menu
 *      (and stays in the row as screen-reader text). Reached ONLY when the
 *      whole count cannot fit beside the folder and the model's floor.
 *
 * THE COUNT NEVER SHRINKS (design D8 = review R3-1 = QA Q3; manager decision,
 * option A). It used to be `shrink min-w-0 truncate`, and the rule read its
 * box AFTER flexbox had squeezed it: `12 queued` painted as `1…` or a bare `1`
 * (a false figure), its cost was invisible to the rule, and the level depended
 * on history (a cold load at 390 settled at L1 with `2 qu…`; the same width
 * reached through 300 px settled at L2 with the count whole). It is now
 * `shrink-0` and [redacted] at its intrinsic width, so it is pressure: the
 * reading yields before it, and when even that is not enough it moves out
 * whole rather than in pieces. No ellipsized form reads truthfully (`2…`
 * drops "queued"), and a numeric badge would be new copy.
 *
 * CONSTRAINTS, the same ones `use-priority-fit.ts` honours:
 * - First frame = final layout. Measured in a layout effect, so a level change
 *   re-renders before paint.
 * - No layout engine (the happy-dom test host) measures 0 px, which means
 *   "unknown": everything renders at level 0, the behaviour tests always saw.
 * - THE LEVEL IS A FUNCTION OF THE CURRENT INPUTS AND WIDTH, never of history
 *   (review R3-1/R3-5): same content + same width gives the same level, cold
 *   or after any keyboard/viewport path. Every input is therefore [redacted] at
 *   its intrinsic width, never at a width an earlier level left it.
 * - A width that is hidden cannot be [redacted], so the reading's forms and the
 *   moved-out count are cached from the last pass that showed them. A cached
 *   width is valid only for the exact content AND layout it was [redacted] in:
 *   the reading is `w-24` (96 px) in the tall layout and intrinsic in the short
 *   one (R3-5 — one shared slot let a tall 96 px compact figure veto L1 on the
 *   next open), so the cache is keyed by layout + form + label, and it resets
 *   when the label changes. A form never [redacted] is TRIED (shown) before it
 *   is ruled out, all before paint.
 *
 * `footFitLevel` is the pure rule (unit-tested); `useFootFit` only measures.
 */
import { flushSync } from "react-dom";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/** 0 full · 1 compact reading · 2 no reading · 3 model chip at its floor ·
    4 queued count moved to the ⋮ menu. */
export type FootFit = 0 | 1 | 2 | 3 | 4;

export type FootFitInput = {
	/** The facts row's own inline size (it takes only the leftover width). */
	available: number;
	/** The row's column gap. */
	gap: number;
	/** Items that never yield or move (the folder glyph), in px. */
	fixed: readonly number[];
	/** The queued count's INTRINSIC width: 0 = no count, null = not yet
	    [redacted] (it is shown, and so [redacted], before anything is ruled out). */
	queued: number | null;
	/** The model control's floor (its min-width; it grows into what is left). */
	model: number;
	/** The reading's width in its full and compact forms; null = never [redacted]. */
	full: number | null;
	short: number | null;
};

/** Sub-pixel layouts (a 110.4 px row) must not flap on rounding. */
const EPSILON = 0.5;

export function footFitLevel(input: FootFitInput): FootFit {
	const { available, gap, fixed, queued, model, full, short } = input;
	const fits = (widths: readonly number[]) =>
		widths.reduce((sum, width) => sum + width, 0) + gap * Math.max(0, widths.length - 1) <= available + EPSILON;
	// An unmeasured input is tried before anything after it is considered.
	if (queued === null) return 0;
	const items = queued > 0 ? [...fixed, queued] : fixed;
	if (full === null || fits([...items, model, full])) return 0;
	if (short === null || fits([...items, model, short])) return 1;
	if (fits([...items, model])) return 2;
	/* The L3 chip is the same 44 px floor the L2 label already has, so it frees
	   nothing: with a count present the count is the only thing left that can
	   move, and it moves whole (L4). Without one, L3 is the clip backstop. */
	return queued > 0 ? 4 : 3;
}

/** A model floor for a host without stylesheets: `min-w-11` is 44 px. */
const MODEL_FLOOR_FALLBACK = 44;

type WidthCache = { label: string; widths: Map<string, number> };

/** Return the cache for `label`, emptied if it held another label's widths. */
function cacheFor(cache: WidthCache, label: string): Map<string, number> {
	if (cache.label !== label) {
		cache.label = label;
		cache.widths = new Map();
	}
	return cache.widths;
}

/**
 * `short`: the keyboard-open layout. `inputKey`: every input that can change
 * the row's geometry without resizing the row itself (labels, the mic, the
 * reading's text) — draft keystrokes and stream frames must not re-measure.
 * `queued`: the count's whole label, "" when nothing is queued; the hook needs
 * it because at L4 the inline count is not in the row to be found.
 */
export function useFootFit(short: boolean, inputKey: string, queued: string) {
	const [row, setRow] = useState<HTMLElement | null>(null);
	const [level, setLevel] = useState<FootFit>(0);
	const readings = useRef<WidthCache>({ label: "", widths: new Map() });
	const counts = useRef<WidthCache>({ label: "", widths: new Map() });
	const [tick, setTick] = useState(0);
	const rowRef = useCallback((node: HTMLElement | null) => setRow(node), []);

	useLayoutEffect(() => {
		if (!row) return;
		const available = row.getBoundingClientRect().width;
		// No layout engine: nothing is known, so nothing yields.
		if (available <= 0) return;
		const width = (key: string) => row.querySelector<HTMLElement>(`[data-foot="${key}"]`)?.getBoundingClientRect().width ?? 0;
		const reading = row.querySelector<HTMLElement>('[data-foot="reading"]');
		const layout = short ? "short" : "tall";
		const readingWidths = cacheFor(readings.current, reading?.getAttribute("aria-label") ?? "");
		const shown = reading?.dataset.footForm;
		/* `max-w-full` can cap the reading's box at a very narrow row; a capped
		   width cached as "the full form" would later read as fitting. */
		const readingWidth = reading ? Math.max(reading.getBoundingClientRect().width, reading.scrollWidth) : 0;
		if (readingWidth > 0 && (shown === "full" || shown === "short")) readingWidths.set(`${layout}|${shown}`, readingWidth);
		/* The count is `shrink-0 whitespace-nowrap`, so its box IS its intrinsic
		   width; `scrollWidth` is the backstop should anything ever squeeze it. */
		const countWidths = cacheFor(counts.current, queued);
		const count = row.querySelector<HTMLElement>('[data-foot="queued"]');
		if (count) countWidths.set("w", Math.max(count.getBoundingClientRect().width, count.scrollWidth));
		const model = row.querySelector<HTMLElement>('[data-foot="model"]');
		const floor = model ? Number.parseFloat(getComputedStyle(model).minWidth) : Number.NaN;
		const next = footFitLevel({
			available,
			gap: Number.parseFloat(getComputedStyle(row).columnGap) || 0,
			fixed: [width("cwd")].filter((w) => w > 0),
			queued: queued ? countWidths.get("w") ?? null : 0,
			model: Number.isFinite(floor) && floor > 0 ? floor : MODEL_FLOOR_FALLBACK,
			// No reading at all (an owner that publishes none) never yields.
			full: reading ? readingWidths.get(`${layout}|full`) ?? null : 0,
			short: reading ? readingWidths.get(`${layout}|short`) ?? null : 0,
		});
		setLevel((previous) => (previous === next ? previous : next));
	/* Measure only inputs that can change geometry. Draft keystrokes and stream
	   frames re-render the composer without invalidating this row's dimensions. */
	}, [row, short, inputKey, queued, level, tick]);

	// The keyboard opening, a rotation, or the mic appearing changes the room.
	useEffect(() => {
		if (!row || typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(() => flushSync(() => setTick((tick) => tick + 1)));
		observer.observe(row);
		return () => observer.disconnect();
	}, [row]);

	return { rowRef, level };
}
