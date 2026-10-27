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
 *      the moment the keyboard closes and the facts get their own line;
 *   3. the model control yields too, leaving the folder glyph.
 *
 * The folder glyph and a queued count are never hidden: the folder is already
 * icon-only, and a queued count is state the user acted on.
 *
 * CONSTRAINTS, the same ones `use-priority-fit.ts` honours:
 * - First frame = final layout. Measured in a layout effect, so a level change
 *   re-renders before paint.
 * - No layout engine (the happy-dom test host) measures 0 px, which means
 *   "unknown": everything renders at level 0, the behaviour tests always saw.
 * - A width that is hidden cannot be measured, so each reading form's width
 *   is cached from the last pass that showed it; a form never measured is
 *   TRIED (shown) before it is ruled out.
 *
 * `footFitLevel` is the pure rule (unit-tested); `useFootFit` only measures.
 */
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/** 0 full · 1 compact reading · 2 no reading · 3 no reading and no model. */
export type FootFit = 0 | 1 | 2 | 3;

export type FootFitInput = {
	/** The facts row's own inline size (it takes only the leftover width). */
	available: number;
	/** The row's column gap. */
	gap: number;
	/** Items that never yield (queued count, folder glyph), in px. */
	fixed: readonly number[];
	/** The model control's floor (its min-width; it grows into what is left). */
	model: number;
	/** The reading's width in its full and compact forms; null = never measured. */
	full: number | null;
	short: number | null;
};

/** Sub-pixel layouts (a 110.4 px row) must not flap on rounding. */
const EPSILON = 0.5;

export function footFitLevel(input: FootFitInput): FootFit {
	const { available, gap, fixed, model, full, short } = input;
	const fits = (widths: readonly number[]) =>
		widths.reduce((sum, width) => sum + width, 0) + gap * Math.max(0, widths.length - 1) <= available + EPSILON;
	// An unmeasured form is tried before anything after it is considered.
	if (full === null || fits([...fixed, model, full])) return 0;
	if (short === null || fits([...fixed, model, short])) return 1;
	if (fits([...fixed, model])) return 2;
	return 3;
}

/** A model floor for a host without stylesheets: `min-w-11` is 44 px. */
const MODEL_FLOOR_FALLBACK = 44;

export function useFootFit() {
	const [row, setRow] = useState<HTMLElement | null>(null);
	const [level, setLevel] = useState<FootFit>(0);
	const readings = useRef<{ full: number | null; short: number | null }>({ full: null, short: null });
	const [, setTick] = useState(0);
	const rowRef = useCallback((node: HTMLElement | null) => setRow(node), []);

	useLayoutEffect(() => {
		if (!row) return;
		const available = row.getBoundingClientRect().width;
		// No layout engine: nothing is known, so nothing yields.
		if (available <= 0) return;
		const width = (key: string) => row.querySelector<HTMLElement>(`[data-foot="${key}"]`)?.getBoundingClientRect().width ?? 0;
		const reading = row.querySelector<HTMLElement>('[data-foot="reading"]');
		const shown = reading?.dataset.footForm;
		const readingWidth = reading?.getBoundingClientRect().width ?? 0;
		if (readingWidth > 0) {
			const form = shown === "full" || shown === "short" ? shown : (row.closest("[data-short]") ? "short" : "full");
			readings.current[form] = readingWidth;
		}
		/* The full reading is the 96 px slot used outside the compact form.
		   Keep that bound even when the current pass renders the compact
		   figure; using the compact width for both forms makes the fit
		   believe the full reading fits when it does not. */
		if (reading && readings.current.full === null && (shown === "short" || row.closest("[data-short]"))) readings.current.full = 96;
		const model = row.querySelector<HTMLElement>('[data-foot="model"]');
		const floor = model ? Number.parseFloat(getComputedStyle(model).minWidth) : Number.NaN;
		const next = footFitLevel({
			available,
			gap: Number.parseFloat(getComputedStyle(row).columnGap) || 0,
			fixed: [width("queued"), width("cwd")].filter((w) => w > 0),
			model: Number.isFinite(floor) && floor > 0 ? floor : MODEL_FLOOR_FALLBACK,
			// No reading at all (an owner that publishes none) never yields.
			full: reading ? readings.current.full : 0,
			short: reading ? readings.current.short : 0,
		});
		setLevel((previous) => (previous === next ? previous : next));
	});

	// The keyboard opening, a rotation, or the mic appearing changes the room.
	useEffect(() => {
		if (!row || typeof ResizeObserver === "undefined") return;
		const observer = new ResizeObserver(() => setTick((tick) => tick + 1));
		observer.observe(row);
		return () => observer.disconnect();
	}, [row]);

	return { rowRef, level };
}
