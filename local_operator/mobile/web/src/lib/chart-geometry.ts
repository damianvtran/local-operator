/**
 * Pure geometry, scale and formatting rules for the inline `lop-ui` charts.
 *
 * Everything here implements the shared inline-chart design spec, which the
 * desktop app, this relay web client and the React Native app follow from the
 * same text so a chart reads the same on every surface:
 *   §1 sizing (plot height from the measured width, never a stretched viewBox),
 *   §2 axes (nice ticks, the zero rule, label thinning, number format, units),
 *   §3 the donut's fold into "Other (n)".
 * It is kept free of React and the DOM so each rule is unit-tested directly
 * (`chart-geometry.test.ts`) instead of through rendered SVG.
 */

/** Narrow/wide breakpoint on the chart's own content width (spec §1). */
export const NARROW_BELOW = 520;

/** Cartesian plot height: clamp(180, round(W × 0.34), 240). */
export function plotHeight(width: number): number {
	return Math.min(240, Math.max(180, Math.round(width * 0.34)));
}

/** Donut outer diameter: 200 wide; min(180, W − 32) narrow (spec §1/§3). */
export function donutDiameter(width: number): number {
	return width >= NARROW_BELOW ? 200 : Math.max(96, Math.min(180, width - 32));
}

/** Margins around the plot (spec §1). `left` is decided by the tick labels. */
export const MARGIN = { top: 12, right: 12, bottom: 24 } as const;

/** Decimals a number is written with, counting exponent notation. */
export function decimalsOf(value: number): number {
	if (!Number.isFinite(value) || Number.isInteger(value)) return 0;
	const [mantissa, exponent = "0"] = String(value).toLowerCase().split("e");
	const fraction = mantissa.split(".")[1]?.length ?? 0;
	return Math.max(0, Math.min(20, fraction - Number(exponent)));
}

export type Ticks = { min: number; max: number; step: number; values: number[]; decimals: number };

/** Target tick count from the plot height: clamp(3, floor(h / 48), 6). */
export function tickTarget(height: number): number {
	return Math.min(6, Math.max(3, Math.floor(height / 48)));
}

const tickCount = (lo: number, hi: number, step: number) =>
	Math.ceil(hi / step - 1e-9) - Math.floor(lo / step + 1e-9) + 1;

/**
 * Nice ticks (spec §2): the smallest step of {1, 2, 2.5, 5} × 10^k giving at
 * most `target + 1` ticks, with the domain extended outward to whole steps.
 * Steps are rebuilt from integers so ticks never print as 0.30000000000000004.
 */
export function niceTicks(lo: number, hi: number, target: number): Ticks {
	if (!(hi > lo)) [lo, hi] = flatDomain(lo);
	const k0 = Math.floor(Math.log10((hi - lo) / (target + 1))) - 1;
	for (let k = k0; k < k0 + 6; k++) {
		for (const m of [1, 2, 2.5, 5]) {
			const step = Number((m * 10 ** k).toPrecision(12));
			if (tickCount(lo, hi, step) > target + 1) continue;
			const first = Math.floor(lo / step + 1e-9);
			const last = Math.ceil(hi / step - 1e-9);
			const decimals = decimalsOf(step);
			const values = [];
			for (let i = first; i <= last; i++) values.push(Number((i * step).toFixed(decimals)));
			return { min: values[0], max: values[values.length - 1], step, values, decimals };
		}
	}
	// Unreachable for finite input; a two-tick fallback keeps the render alive.
	return { min: lo, max: hi, step: hi - lo, values: [lo, hi], decimals: 0 };
}

/** Single-value / flat data: pad ±10% of |value|, or ±1 at zero. */
function flatDomain(value: number): [number, number] {
	const pad = value === 0 ? 1 : Math.abs(value) * 0.1;
	return [value - pad, value + pad];
}

/**
 * The value domain before nicening (spec §2 "Zero"). Bar-like marks always
 * include 0 (a bar's length IS its value); lines and scatter include 0 only
 * when it lies within a third of the range beyond the data, otherwise the
 * domain fits the data so a 84–154 series is not squashed against a 0 floor.
 */
export function valueDomain(values: readonly number[], kind: "bar" | "line"): [number, number] {
	let lo = Math.min(...values);
	let hi = Math.max(...values);
	if (kind === "bar") return lo === hi && lo === 0 ? [0, 1] : [Math.min(0, lo), Math.max(0, hi)];
	if (lo === hi) return flatDomain(lo);
	const reach = (hi - lo) / 3;
	if (lo - reach <= 0 && 0 <= hi + reach) {
		lo = Math.min(0, lo);
		hi = Math.max(0, hi);
	}
	return [lo, hi];
}

/** Percent mode: exactly 0–100% at 25% steps (spec §2). */
export const PERCENT_TICKS: Ticks = { min: 0, max: 100, step: 25, values: [0, 25, 50, 75, 100], decimals: 0 };

/**
 * Right-axis ticks for a dual chart: the same COUNT as the left axis so the
 * left axis's gridlines also land on every right tick (spec §2 "Dual axis").
 */
export function alignedTicks(lo: number, hi: number, count: number): Ticks {
	if (!(hi > lo)) [lo, hi] = flatDomain(lo);
	const intervals = Math.max(1, count - 1);
	const k0 = Math.floor(Math.log10((hi - lo) / intervals)) - 1;
	for (let k = k0; k < k0 + 8; k++) {
		for (const m of [1, 2, 2.5, 5]) {
			const step = Number((m * 10 ** k).toPrecision(12));
			const first = Math.floor(lo / step + 1e-9);
			if ((first + intervals) * step < hi - 1e-9) continue;
			const decimals = decimalsOf(step);
			const values = Array.from({ length: count }, (_, i) => Number(((first + i) * step).toFixed(decimals)));
			return { min: values[0], max: values[count - 1], step, values, decimals };
		}
	}
	return niceTicks(lo, hi, count - 1);
}

/** Linear scale from a tick domain onto a pixel range (inverted for y). */
export function scale(ticks: Pick<Ticks, "min" | "max">, from: number, to: number) {
	const span = ticks.max - ticks.min || 1;
	return (value: number) => from + ((value - ticks.min) / span) * (to - from);
}

/** Crisp 1px lines: y = round(y) + 0.5 (spec §2 "Gridlines"). */
export const snap = (value: number) => Math.round(value) + 0.5;

// ---- number and unit format ------------------------------------------------

/** Tick formatter: compact (12K, 1.5M) when the domain reaches |10 000|. */
export function tickFormatter(locale: string, ticks: Ticks, suffix = "") {
	const compact = Math.max(Math.abs(ticks.min), Math.abs(ticks.max)) >= 10_000;
	const format = new Intl.NumberFormat(locale, compact
		? { notation: "compact", maximumFractionDigits: 1 }
		: { minimumFractionDigits: ticks.decimals, maximumFractionDigits: ticks.decimals });
	return (value: number) => `${format.format(value)}${suffix}`;
}

/** Tooltip/table value: full authored precision, grouping on, never compact. */
export function exactValue(locale: string, value: number): string {
	const digits = decimalsOf(value);
	return new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

/**
 * Exact text of a SUM of authored values: rounded to the most decimals any
 * addend was written with, so 412.5 + 220.1 + … reads 910.8, not the binary
 * float 910.8000000000001 that decimalsOf would faithfully print.
 */
export function exactSum(locale: string, values: readonly number[]): string {
	const digits = Math.max(0, ...values.map(decimalsOf));
	const sum = Number(values.reduce((a, b) => a + b, 0).toFixed(digits));
	return new Intl.NumberFormat(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(sum);
}

const TIGHT_UNITS = new Set(["%", "‰", "°", "°C", "°F"]);

/** Value + unit: no space before %, ‰, °, °C, °F; one space otherwise. */
export function withUnit(text: string, unit?: string): string {
	if (!unit) return text;
	return TIGHT_UNITS.has(unit) ? `${text}${unit}` : `${text} ${unit}`;
}

/** Share of a whole, one decimal (`38.2%`). */
export function shareText(locale: string, fraction: number): string {
	return new Intl.NumberFormat(locale, { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(fraction);
}

// ---- x labels ----------------------------------------------------------------

/** Axis text is the part before the first ` · ` (`Nov 24 · rollout` → `Nov 24`). */
export function axisLabel(label: string): string {
	const cut = label.indexOf(" · ");
	return cut === -1 ? label : label.slice(0, cut);
}

export type ShownLabel = { index: number; text: string; anchor: "start" | "middle" | "end" };

/**
 * Which x labels to draw (spec §2 "X labels"): stride
 * k = ceil((maxLabelW + 12) / slotW), indices 0, k, 2k… plus always the last;
 * when the last collides with the previous shown one, the previous is dropped.
 * `edgeAnchored` (line/scatter) anchors the first label at its start and the
 * last at its end, so neither hangs off the plot. Labels never rotate; one
 * longer than k·slotW − 8 is truncated with an ellipsis.
 *
 * @param positions the x of each category's mark, in px
 * @param measure   text width in px at the label font
 */
export function thinLabels(
	labels: readonly string[],
	positions: readonly number[],
	slot: number,
	measure: (text: string) => number,
	edgeAnchored: boolean,
): ShownLabel[] {
	const n = labels.length;
	if (n === 0) return [];
	const texts = labels.map(axisLabel);
	const widths = texts.map(measure);
	const stride = Math.max(1, Math.ceil((Math.max(...widths) + 12) / Math.max(slot, 1)));
	const indices: number[] = [];
	for (let i = 0; i < n; i += stride) indices.push(i);
	const anchorOf = (i: number): ShownLabel["anchor"] =>
		!edgeAnchored || n === 1 ? "middle" : i === 0 ? "start" : i === n - 1 ? "end" : "middle";
	let budget = stride * slot - 8;
	const fit = (i: number) => truncate(texts[i], widths[i], budget, measure);
	const box = (i: number) => {
		const w = Math.min(widths[i], Math.max(budget, 0));
		const anchor = anchorOf(i);
		const left = anchor === "start" ? positions[i] : anchor === "end" ? positions[i] - w : positions[i] - w / 2;
		return [left, left + w] as const;
	};
	if (indices[indices.length - 1] !== n - 1) {
		const previous = indices[indices.length - 1];
		if (box(previous)[1] + 4 > box(n - 1)[0]) {
			if (indices.length > 1) indices.pop();
			// Only the first label is left to drop, and the first must stay: share
			// the span between them instead, so both ends show and never touch.
			else budget = Math.max(0, (positions[n - 1] - positions[0]) / (edgeAnchored ? 2 : 1) - 8);
		}
		indices.push(n - 1);
	}
	return indices.map((index) => ({ index, text: fit(index), anchor: anchorOf(index) }));
}

/** Truncate to `budget` px with a trailing ellipsis (binary search on length). */
export function truncate(text: string, width: number, budget: number, measure: (text: string) => number): string {
	if (width <= budget || budget <= 0) return text;
	const chars = Array.from(text);
	let lo = 0;
	let hi = chars.length;
	while (lo < hi) {
		const mid = Math.ceil((lo + hi) / 2);
		if (measure(`${chars.slice(0, mid).join("")}…`) <= budget) lo = mid;
		else hi = mid - 1;
	}
	return `${chars.slice(0, lo).join("")}…`;
}

/** Fallback text width when no canvas is available: 6.6 px per 12px char. */
export const estimateWidth = (text: string) => Array.from(text).length * 6.6;

// ---- donut -------------------------------------------------------------------

export type Slice = {
	label: string;
	value: number;
	/** Position in the authored data, -1 for the folded "Other" slice. */
	source: number;
	/** Authored rows folded into "Other", largest first. */
	folded?: { label: string; value: number }[];
};

/**
 * Sort slices by value descending and, past 7, keep the largest 6 and fold
 * the rest into one "Other" slice, always last (spec §3 "Fold"). The data
 * table keeps every authored row; only the drawing folds.
 */
export function foldSlices(data: readonly (readonly [string, number])[]): Slice[] {
	const sorted = data
		.map(([label, value], source) => ({ label, value, source }))
		.sort((a, b) => b.value - a.value || a.source - b.source);
	if (sorted.length <= 7) return sorted;
	const kept = sorted.slice(0, 6);
	const rest = sorted.slice(6);
	return [...kept, {
		label: "",
		value: rest.reduce((sum, { value }) => sum + value, 0),
		source: -1,
		folded: rest.map(({ label, value }) => ({ label, value })),
	}];
}

/** Labels on one side of the donut, relaxed apart to ≥ `gap` px (spec §3). */
export function relaxLabels(ys: readonly number[], gap: number, top: number, bottom: number): number[] {
	const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0]);
	const placed = order.map(([y]) => y);
	for (let i = 1; i < placed.length; i++) placed[i] = Math.max(placed[i], placed[i - 1] + gap);
	const overflow = placed.length ? placed[placed.length - 1] - bottom : 0;
	if (overflow > 0) {
		placed[placed.length - 1] -= overflow;
		for (let i = placed.length - 2; i >= 0; i--) placed[i] = Math.min(placed[i], placed[i + 1] - gap);
	}
	for (let i = 0; i < placed.length; i++) placed[i] = Math.max(placed[i], top + i * gap);
	const out = new Array<number>(ys.length);
	order.forEach(([, index], i) => { out[index] = placed[i]; });
	return out;
}

/** WCAG contrast of black or white text on `#rrggbb`, whichever is higher. */
export function bestTextOn(hex: string): { color: "#000" | "#FFF"; ratio: number } | null {
	const match = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
	if (!match) return null;
	const n = Number.parseInt(match[1], 16);
	const lum = [(n >> 16) & 255, (n >> 8) & 255, n & 255]
		.map((v) => v / 255)
		.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
	const y = 0.2126 * lum[0] + 0.7152 * lum[1] + 0.0722 * lum[2];
	const black = (y + 0.05) / 0.05;
	const white = 1.05 / (y + 0.05);
	return black >= white ? { color: "#000", ratio: black } : { color: "#FFF", ratio: white };
}

/** Next index for ←/→/↑/↓/Home/End over `count` items, clamped (spec §5). */
export function stepIndex(key: string, index: number, count: number): number | null {
	switch (key) {
		case "ArrowRight": case "ArrowDown": return Math.min(count - 1, index + 1);
		case "ArrowLeft": case "ArrowUp": return Math.max(0, index - 1);
		case "Home": return 0;
		case "End": return count - 1;
		default: return null;
	}
}
