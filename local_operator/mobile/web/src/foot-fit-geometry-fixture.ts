/**
 * A flexbox model of the composer's facts row for happy-dom tests.
 *
 * WHY IT EXISTS (review R3-4). happy-dom has no layout engine, and the earlier
 * wiring tests stubbed every footer item at a FIXED width. That stub could not
 * see the defect design D8 found: the queued count was `shrink min-w-0`, so a
 * real browser squeezed it and the fit read the squeezed width. A test whose
 * geometry never shrinks anything passes against that bug. This model derives
 * each item's box from the classes the component actually renders, the way the
 * row's CSS would:
 *
 * - the row is `available` px wide while the column is short (`[data-short]`),
 *   `tallAvailable` otherwise (the keyboard-closed own line);
 * - the folder is 44; a `w-11` model chip is 44; otherwise the model is
 *   `flex-1` with a 44 px floor and takes what is left;
 * - the reading is hidden (0), or its form's intrinsic width in the short
 *   layout, or `w-24` (96) in the tall one;
 * - the queued count is its intrinsic text width when `shrink-0`; a shrinkable
 *   count gets only what is left after everything else's floor (the bug).
 *
 * `scrollWidth` reports the intrinsic width, as a browser does for clipped or
 * squeezed content. Only `[data-foot]` items and the row are measured; every
 * other element reads 0 ("unknown"), so unrelated measuring hooks stay inert.
 * The numbers are the designer's and QA's own measurements at 320-390 px.
 */
import { vi } from "vitest";

export const FOOT = { gap: 4, folder: 44, modelFloor: 44, short: 44.3 } as const;

/** Intrinsic width of a `# queued` label: `2 queued` 53.3, `12 queued` 59.9. */
export function queuedWidth(text: string): number {
	return Math.round(text.length * 6.66 * 10) / 10;
}

export const geometry = { available: 110, tallAvailable: 304 };

function rowOf(element: Element): HTMLElement | null {
	return element.closest<HTMLElement>("[data-foot-level]");
}

function rowWidth(row: HTMLElement): number {
	return row.closest("[data-short]") ? geometry.available : geometry.tallAvailable;
}

function intrinsic(element: HTMLElement): number {
	const foot = element.dataset.foot;
	if (foot === "cwd") return FOOT.folder;
	if (foot === "queued") return queuedWidth(element.textContent ?? "");
	if (foot === "reading") {
		if (element.classList.contains("hidden")) return 0;
		if (!element.closest("[data-short]")) return 96;
		if (element.dataset.footForm === "short") return FOOT.short;
		/* The full pair's width follows its text: `30.9% · $2.47` is 88.5 px,
		   `99.5% · $123.47` 102 px (review R3-6's own figures). */
		const text = element.getAttribute("aria-label")?.replace(/^[^:]*:\s*/, "") ?? "";
		return Math.max(FOOT.short, Math.round(text.length * 6.81 * 10) / 10);
	}
	return 0;
}

/** Every item's box in one row, the way flexbox would lay it out. */
export function layoutRow(row: HTMLElement): Map<Element, number> {
	const items = [...row.querySelectorAll<HTMLElement>(":scope > [data-foot]")].filter(
		(item) => !(item.dataset.foot === "reading" && item.classList.contains("hidden")),
	);
	const boxes = new Map<Element, number>();
	const available = rowWidth(row);
	const gaps = FOOT.gap * Math.max(0, items.length - 1);
	const model = items.find((item) => item.dataset.foot === "model");
	const modelPinned = model?.classList.contains("w-11") ?? false;
	const queued = items.find((item) => item.dataset.foot === "queued");
	const queuedShrinks = queued ? !queued.classList.contains("shrink-0") : false;
	let rigid = gaps;
	for (const item of items) {
		if (item === model) rigid += FOOT.modelFloor;
		else if (item === queued && queuedShrinks) continue;
		else rigid += intrinsic(item);
	}
	for (const item of items) {
		if (item === model) continue;
		if (item === queued && queuedShrinks) {
			boxes.set(item, Math.max(0, Math.min(intrinsic(item), available - rigid)));
		} else boxes.set(item, intrinsic(item));
	}
	if (model) {
		const used = [...boxes.values()].reduce((sum, width) => sum + width, 0) + gaps;
		boxes.set(model, modelPinned ? FOOT.modelFloor : Math.max(FOOT.modelFloor, available - used));
	}
	return boxes;
}

function widthOf(element: HTMLElement): number {
	if (element.hasAttribute("data-foot-level")) return rowWidth(element);
	const row = element.dataset.foot ? rowOf(element) : null;
	if (!row) return 0;
	return layoutRow(row).get(element) ?? 0;
}

/** Install the model; `vi.restoreAllMocks()` removes it. */
export function installFootGeometry() {
	vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
		const width = widthOf(this);
		return { width, height: 44, x: 0, y: 0, top: 0, bottom: 44, left: 0, right: width, toJSON: () => ({}) } as DOMRect;
	});
	vi.spyOn(HTMLElement.prototype, "scrollWidth", "get").mockImplementation(function (this: HTMLElement) {
		return this.dataset.foot ? Math.max(intrinsic(this), widthOf(this)) : 0;
	});
	/* The two properties the hook reads; the row's gap and the chip's floor. */
	const real = window.getComputedStyle.bind(window);
	vi.spyOn(window, "getComputedStyle").mockImplementation((element: Element, pseudo?: string | null) => {
		const style = real(element, pseudo);
		const override = element.hasAttribute("data-foot-level") ? { columnGap: `${FOOT.gap}px` }
			: element.getAttribute("data-foot") === "model" ? { minWidth: `${FOOT.modelFloor}px` } : null;
		/* Bind the real declaration's methods to it (the accessibility queries
		   call `getPropertyValue`), overriding only what layout decides. */
		return override ? new Proxy(style, {
			get: (target, key) => {
				if (key in override) return override[key as keyof typeof override];
				const value = Reflect.get(target, key, target);
				return typeof value === "function" ? value.bind(target) : value;
			},
		}) : style;
	});
}
