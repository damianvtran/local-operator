/**
 * Per-type geometry for the inline `lop-ui` charts on the relay page.
 *
 * Each chart computes its layout at the MEASURED width (inline-chart spec §1:
 * the SVG is drawn at its real pixel size, never a fixed viewBox stretched
 * with preserveAspectRatio) and hands four answers to `ChartSurface`: what is
 * under a point, what a mark says, where it sits and how it is drawn. The
 * shared interaction, tooltip and "Chart data" disclosure live in
 * `inline-chart-frame.tsx`; the scale/tick/label rules in
 * `lib/chart-geometry.ts`.
 *
 * Why this replaced the previous renderer: a 320×100 viewBox stretched to the
 * column made axes wrong and the plot squat, the pie was 96px with its values
 * in rows beside it, and slice colours were the accent plus semantic roles
 * that are near-identical hues on most themes. All three are what the
 * operator reported.
 */
import { useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CategoryChart, Heatmap, Scatter } from "../inline-ui-parts";
import type { InlineSeries } from "../inline-series";
import {
	MARGIN, NARROW_BELOW, PERCENT_TICKS, type Ticks, alignedTicks, axisLabel, bestTextOn, donutDiameter,
	exactSum, exactValue, foldSlices, niceTicks, plotHeight, relaxLabels, scale, shareText, snap, thinLabels,
	tickFormatter, tickTarget, truncate, valueDomain, withUnit,
} from "../lib/chart-geometry";
import { messageRuntime, useLocaleSnapshot } from "../store";
import {
	type Anchor, ChartBlock, ChartSurface, type DataTable, FILL, HeaderRow, LegendItem, type Mark,
	OTHER_FILL, OTHER_SWATCH, STROKE, SWATCH, type Tip, type TipRow, dim, useMeasure,
} from "./inline-chart-frame";

const t = messageRuntime.t;
const position = (index: number, count: number, locale: string) => {
	const f = new Intl.NumberFormat(locale);
	return t("relay.chart.position", { index: f.format(index + 1), count: f.format(count) });
};
const join = (...parts: (string | undefined)[]) => parts.filter(Boolean).join(", ");

// ---- shared SVG pieces --------------------------------------------------------

/** A bar rect with radius-3 corners on the end AWAY from zero (spec §1). */
function barPath(x: number, w: number, from: number, to: number, horizontal = false, radius = 3): string {
	const length = Math.abs(to - from);
	const r = Math.min(radius, length, w / 2);
	if (horizontal) {
		// x axis is the value; `x`/`w` are the row's y and thickness.
		const [y, h] = [x, w];
		if (to >= from) return `M${from},${y}H${to - r}Q${to},${y} ${to},${y + r}V${y + h - r}Q${to},${y + h} ${to - r},${y + h}H${from}Z`;
		return `M${from},${y}H${to + r}Q${to},${y} ${to},${y + r}V${y + h - r}Q${to},${y + h} ${to + r},${y + h}H${from}Z`;
	}
	if (to <= from) return `M${x},${from}V${to + r}Q${x},${to} ${x + r},${to}H${x + w - r}Q${x + w},${to} ${x + w},${to + r}V${from}Z`;
	return `M${x},${from}V${to - r}Q${x},${to} ${x + r},${to}H${x + w - r}Q${x + w},${to} ${x + w},${to - r}V${from}Z`;
}

/** Horizontal gridlines and the y tick column (spec §2: no spine, no tick marks). */
function YAxis({ ticks, y, left, right, format, side = "left", grid = true }: {
	ticks: Ticks; y: (v: number) => number; left: number; right: number;
	format: (v: number) => string; side?: "left" | "right"; grid?: boolean;
}) {
	return <g>
		{ticks.values.map((v) => (
			<g key={v}>
				{grid && <line x1={left} x2={right} y1={snap(y(v))} y2={snap(y(v))} strokeWidth={1}
					className={v === 0 && ticks.min < 0 ? "stroke-hairline-strong" : "stroke-hairline"} />}
				<text x={side === "left" ? left - 8 : right + 8} y={y(v)} dominantBaseline="central"
					textAnchor={side === "left" ? "end" : "start"} className="fill-ink-muted text-meta tabular-nums">{format(v)}</text>
			</g>
		))}
	</g>;
}

/** Thinned category labels under the plot (spec §2 "X labels"). */
function XLabels({ labels, xs, slot, measure, edge, top, width }: {
	labels: readonly string[]; xs: readonly number[]; slot: number; measure: (s: string) => number;
	edge: boolean; top: number; width: number;
}) {
	return <g>
		{thinLabels(labels, xs, slot, measure, edge).map(({ index, text, anchor }) => {
			// Centred labels are kept inside the SVG so the first/last never clip.
			const half = measure(text) / 2;
			const x = anchor === "middle" ? Math.max(half, Math.min(width - half, xs[index])) : xs[index];
			return <text key={index} x={x} y={top + 18} textAnchor={anchor} className="fill-ink-muted text-meta">{text}</text>;
		})}
	</g>;
}

/** Left margin = widest tick label + 8, minimum 28 (spec §1). */
const gutter = (labels: string[], measure: (s: string) => number) =>
	Math.max(28, Math.ceil(Math.max(0, ...labels.map(measure))) + 8);

// ---- cartesian: line, bar, histogram, series --------------------------------

type Series = { label: string; values: number[]; right?: boolean };
type Cartesian = {
	kind: string;
	title: string;
	unit?: string;
	rightUnit?: string;
	labels: string[];
	series: Series[];
	mode: "line" | "bar" | "grouped" | "stacked" | "percent" | "dual";
	tableId: string;
	width: number;
};

function CartesianChart(props: Cartesian) {
	const { kind, title, unit, rightUnit, labels, series, mode, tableId, width } = props;
	const { uiLocale: locale } = useLocaleSnapshot();
	const probe = useRef<HTMLDivElement>(null);
	const measure = useMeasure(probe, width);
	const n = labels.length;
	const lineLike = mode === "line" || mode === "dual";
	// Mode flags hoisted out of JSX: bar stacks share the gap/rounding rules.
	const stacks = mode === "stacked" || mode === "percent";
	const percent = mode === "percent";
	const plotH = plotHeight(width);
	const target = tickTarget(plotH);

	const [legendActive, setLegend] = useState<number | null>(null);
	const model = useMemo(() => {
		// Stack offsets (stacked/percent) or plain values (others), per category.
		const segments = labels.map((_, c) => {
			const total = series.reduce((sum, s) => sum + Math.abs(s.values[c]), 0) || 1;
			let pos = 0;
			let neg = 0;
			return series.map((s) => {
				const raw = s.values[c];
				const v = mode === "percent" ? (raw / total) * 100 : raw;
				if (mode !== "stacked" && mode !== "percent") return { from: 0, to: v };
				const from = v < 0 ? neg : pos;
				if (v < 0) neg += v; else pos += v;
				return { from, to: from + v };
			});
		});
		const left = series.filter((s) => !s.right);
		const right = series.filter((s) => s.right);
		const leftValues = mode === "stacked"
			? segments.flatMap((segs) => segs.flatMap(({ from, to }) => [from, to]))
			: left.flatMap((s) => s.values);
		const yTicks = mode === "percent" ? PERCENT_TICKS
			: niceTicks(...valueDomain(leftValues, lineLike ? "line" : "bar"), target);
		const rTicks = right.length
			? alignedTicks(...valueDomain(right.flatMap((s) => s.values), "line"), yTicks.values.length)
			: null;
		return { segments, yTicks, rTicks };
	}, [labels, series, mode, lineLike, target]);

	const { segments, yTicks, rTicks } = model;
	const yFormat = tickFormatter(locale, yTicks, mode === "percent" ? "%" : "");
	const rFormat = rTicks ? tickFormatter(locale, rTicks) : null;
	const left = gutter(yTicks.values.map(yFormat), measure);
	const right = rTicks && rFormat ? gutter(rTicks.values.map(rFormat), measure) : MARGIN.right;
	const plotW = Math.max(40, width - left - right);
	const top = MARGIN.top;
	const bottom = top + plotH;
	const height = bottom + MARGIN.bottom;
	const y = scale(yTicks, bottom, top);
	const yR = rTicks ? scale(rTicks, bottom, top) : y;
	const yOf = (s: number) => (series[s].right ? yR : y);

	// Lines span the plot edge to edge; bars sit in equal slots (spec §5 hit model).
	const slot = lineLike ? (n > 1 ? plotW / (n - 1) : plotW) : plotW / n;
	const xs = labels.map((_, c) => (lineLike ? (n > 1 ? left + c * slot : left + plotW / 2) : left + slot * (c + 0.5)));
	const band = slot * 0.7;
	const grouped = mode === "grouped";
	const m = series.length;
	const barW = grouped ? Math.min(48, (band - 2 * (m - 1)) / m) : Math.min(48, band);
	const groupW = grouped ? barW * m + 2 * (m - 1) : barW;
	const barX = (c: number, s: number) => xs[c] - groupW / 2 + (grouped ? s * (barW + 2) : 0);

	const column = (x: number) => {
		if (n === 1) return 0;
		const c = lineLike ? Math.round((x - left) / slot) : Math.floor((x - left) / slot);
		return Math.max(0, Math.min(n - 1, c));
	};
	const hit = (x: number, py: number): Mark | null => {
		const c = column(x);
		if (lineLike) return { c, s: 0 };
		if (grouped) {
			let best = 0;
			for (let s = 1; s < m; s++) if (Math.abs(barX(c, s) + barW / 2 - x) < Math.abs(barX(c, best) + barW / 2 - x)) best = s;
			return { c, s: best };
		}
		if (m === 1) return { c, s: 0 };
		// Stacked: the segment under the pointer, else the nearest one.
		let best = 0;
		let distance = Infinity;
		segments[c].forEach(({ from, to }, s) => {
			const [a, b] = [y(from), y(to)].sort((p, q) => p - q);
			const d = py < a ? a - py : py > b ? py - b : 0;
			if (d < distance) { distance = d; best = s; }
		});
		return { c, s: best };
	};

	const valueText = (s: number, c: number) => {
		const u = series[s].right ? rightUnit : unit;
		return withUnit(exactValue(locale, series[s].values[c]), u);
	};
	const tip = (mark: Mark): { tip: Tip; anchor: Anchor } => {
		const { c, s } = mark;
		const single = m === 1;
		const rows: TipRow[] = series.map((ser, i) => ({
			swatch: SWATCH[i], label: single ? undefined : ser.right ? t("relay.chart.right_axis", { label: ser.label }) : ser.label,
			value: valueText(i, c),
		}));
		const notes: string[] = [];
		if (mode === "percent" || mode === "stacked") {
			const sum = series.reduce((acc, ser) => acc + Math.abs(ser.values[c]), 0) || 1;
			notes.push(t("relay.chart.share", { share: shareText(locale, Math.abs(series[s].values[c]) / sum) }));
		}
		if (mode === "stacked") rows.push({ label: t("relay.chart.total"), value: withUnit(exactSum(locale, series.map((ser) => ser.values[c])), unit) });
		let anchor: Anchor;
		if (lineLike) {
			const ys = series.map((_, i) => yOf(i)(series[i].values[c]));
			anchor = { x: xs[c], y: Math.min(...ys), below: Math.max(...ys) };
		} else if (mode === "stacked" || mode === "percent") {
			const tops = segments[c].flatMap(({ from, to }) => [y(from), y(to)]);
			anchor = { x: xs[c], y: Math.min(...tops), below: Math.max(...tops) };
		} else {
			const { to } = segments[c][s];
			anchor = { x: barX(c, s) + barW / 2, y: Math.min(y(0), y(to)), below: Math.max(y(0), y(to)) };
		}
		return { tip: { heading: labels[c], rows, notes }, anchor };
	};
	const announce = ({ c, s }: Mark) => join(labels[c], m > 1 ? series[s].label : undefined, valueText(s, c), position(c, n, locale));

	const legend = m > 1 && <>{series.map((ser, i) => (
		<LegendItem key={i} swatch={SWATCH[i]} dashed={ser.right}
			label={ser.right ? t("relay.chart.right_axis", { label: ser.label }) : ser.label}
			onEnter={() => setLegend(i)} onLeave={() => setLegend(null)} active={legendActive === null ? undefined : legendActive === i} />
	))}</>;
	return (
		<div ref={probe} className="text-meta">
			<ChartSurface title={title} tableId={tableId} width={width} height={height} kind={kind}
				columns={n} rows={lineLike ? 1 : m} scrub={lineLike}
				hit={hit} tip={tip} announce={announce}
				header={<HeaderRow start={percent ? undefined : unit} end={rightUnit}>{legend}</HeaderRow>}>
				{(active) => {
					const focusSeries = legendActive ?? (active && !lineLike ? active.s : null);
					return <>
						<YAxis ticks={yTicks} y={y} left={left} right={left + plotW} format={yFormat} />
						{rTicks && rFormat && <YAxis ticks={rTicks} y={yR} left={left} right={left + plotW} format={rFormat} side="right" grid={false} />}
						{lineLike && active && <line x1={snap(xs[active.c])} x2={snap(xs[active.c])} y1={top} y2={bottom} strokeWidth={1} className="stroke-hairline-strong" />}
						{lineLike ? series.map((ser, s) => {
							const yy = yOf(s);
							const points = ser.values.map((v, c) => `${xs[c]},${yy(v)}`).join(" ");
							const faded = legendActive !== null && legendActive !== s;
							return <g key={s} className={dim(faded)}>
								{n > 1 && <polyline points={points} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
									strokeDasharray={ser.right ? "6 4" : undefined} className={`fill-none ${STROKE[s]}`} />}
								{(n <= 12 || n === 1) && ser.values.map((v, c) => <circle key={c} cx={xs[c]} cy={yy(v)} r={n === 1 ? 4 : 2.5} className={FILL[s]} />)}
								{active && <circle cx={xs[active.c]} cy={yy(ser.values[active.c])} r={4} strokeWidth={2} className={`${FILL[s]} stroke-canvas`} />}
							</g>;
						}) : segments.map((segs, c) => segs.map(({ from, to }, s) => {
							if (from === to) return null;
							// Stacked segments: a real 1px gap in geometry, never a ground-coloured
							// stroke (spec §1 — a canvas stroke would show as a line on glass).
							let a = y(from);
							const b = y(to);
							if (stacks && from !== 0) a += b < a ? -1 : 1;
							const faded = focusSeries !== null && (legendActive !== null ? focusSeries !== s : !(active && active.c === c && active.s === s));
							// Only a stack's outermost segment carries the rounded end; inner
							// joins stay square so the stack reads as one bar.
							const outer = !stacks
								|| segs.slice(s + 1).every((next) => next.from === next.to || Math.sign(next.to - next.from) !== Math.sign(to - from));
							return <path key={`${c}-${s}`} d={barPath(barX(c, s), barW, a, b, false, outer ? 3 : 0)} className={`${FILL[s]} ${dim(faded)}`} />;
						}))}
						<XLabels labels={labels} xs={xs} slot={slot} measure={measure} edge={lineLike} top={bottom} width={width} />
					</>;
				}}
			</ChartSurface>
		</div>
	);
}

// ---- horizontal bars ------------------------------------------------------------

/**
 * Spec §1: horizontal when n > 12, any label > 12 chars, or on a narrow column
 * with n > 6 — long category names stay readable instead of thinned away.
 */
export function useHorizontal(data: readonly (readonly [string, number])[], width: number) {
	const n = data.length;
	return n > 12 || data.some(([label]) => Array.from(axisLabel(label)).length > 12) || (width < NARROW_BELOW && n > 6);
}

function HorizontalBars({ value, tableId, width }: { value: CategoryChart; tableId: string; width: number }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const probe = useRef<HTMLDivElement>(null);
	const measure = useMeasure(probe, width);
	const { data, unit, title } = value;
	const n = data.length;
	const labelW = Math.min(Math.round(width * 0.4), Math.max(28, ...data.map(([l]) => Math.ceil(measure(axisLabel(l))))) + 8);
	const plotW = Math.max(40, width - labelW - MARGIN.right);
	// Value ticks: the §2 rule with the target taken from the plot WIDTH (one
	// tick per ~64px) — the spec gives no horizontal target; this keeps labels
	// from touching on a 360px phone.
	const ticks = niceTicks(...valueDomain(data.map(([, v]) => v), "bar"), Math.min(6, Math.max(3, Math.floor(plotW / 64))));
	const format = tickFormatter(locale, ticks);
	const x = scale(ticks, labelW, labelW + plotW);
	const rowsH = n * 28;
	const height = rowsH + MARGIN.bottom;
	const rowY = (c: number) => c * 28 + 6;
	const hit = (_: number, py: number): Mark | null => ({ c: Math.max(0, Math.min(n - 1, Math.floor(py / 28))), s: 0 });
	const valueText = (c: number) => withUnit(exactValue(locale, data[c][1]), unit);
	const tip = ({ c }: Mark) => ({
		tip: { heading: data[c][0], rows: [{ value: valueText(c) }] },
		anchor: { x: Math.max(x(0), x(data[c][1])), y: rowY(c), below: rowY(c) + 16 },
	});
	return (
		<div ref={probe} className="text-meta">
			<ChartSurface title={title} tableId={tableId} width={width} height={height} kind="bar" columns={n} rows={1}
				hit={hit} tip={tip} announce={({ c }) => join(data[c][0], valueText(c), position(c, n, locale))}
				header={<HeaderRow end={unit} />}>
				{(active) => <>
					{ticks.values.map((v) => <g key={v}>
						<line y1={0} y2={rowsH} x1={snap(x(v))} x2={snap(x(v))} strokeWidth={1} className={v === 0 && ticks.min < 0 ? "stroke-hairline-strong" : "stroke-hairline"} />
						<text x={x(v)} y={rowsH + 18} textAnchor="middle" className="fill-ink-muted text-meta tabular-nums">{format(v)}</text>
					</g>)}
					{data.map(([label, v], c) => <g key={c}>
						<text x={labelW - 8} y={rowY(c) + 8} dominantBaseline="central" textAnchor="end" className="fill-ink-muted text-meta">
							{truncate(axisLabel(label), measure(axisLabel(label)), labelW - 8, measure)}
						</text>
						{v !== 0 && <path d={barPath(rowY(c), 16, x(0), x(v), true)} className={`fill-chart-1 ${dim(Boolean(active) && active?.c !== c)}`} />}
					</g>)}
				</>}
			</ChartSurface>
		</div>
	);
}

// ---- scatter --------------------------------------------------------------------

function ScatterChart({ value, tableId, width }: { value: Scatter; tableId: string; width: number }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const probe = useRef<HTMLDivElement>(null);
	const measure = useMeasure(probe, width);
	const { data, xLabel, yLabel, title } = value;
	// Keyboard order is left-to-right; the table keeps the authored order.
	const order = useMemo(() => data.map((_, i) => i).sort((a, b) => data[a][0] - data[b][0] || data[a][1] - data[b][1]), [data]);
	const plotH = plotHeight(width);
	const yTicks = niceTicks(...valueDomain(data.map(([, v]) => v), "line"), tickTarget(plotH));
	const yFormat = tickFormatter(locale, yTicks);
	const left = gutter(yTicks.values.map(yFormat), measure);
	const plotW = Math.max(40, width - left - MARGIN.right);
	const xTicks = niceTicks(...valueDomain(data.map(([v]) => v), "line"), tickTarget(plotW));
	const xFormat = tickFormatter(locale, xTicks);
	const top = MARGIN.top;
	const bottom = top + plotH;
	const x = scale(xTicks, left, left + plotW);
	const y = scale(yTicks, bottom, top);
	const hit = (px: number, py: number): Mark | null => {
		let best: Mark | null = null;
		let distance = 24; // spec §5: nearest point within 24px
		order.forEach((i, c) => {
			const d = Math.hypot(x(data[i][0]) - px, y(data[i][1]) - py);
			if (d <= distance) { distance = d; best = { c, s: 0 }; }
		});
		return best;
	};
	const lines = (i: number) => [
		{ value: t("relay.chart.axis_value", { label: xLabel, value: exactValue(locale, data[i][0]) }) },
		{ value: t("relay.chart.axis_value", { label: yLabel, value: exactValue(locale, data[i][1]) }) },
	];
	const tip = ({ c }: Mark) => {
		const i = order[c];
		return { tip: { rows: lines(i) }, anchor: { x: x(data[i][0]), y: y(data[i][1]) - 6, below: y(data[i][1]) + 6 } };
	};
	const xs = xTicks.values.map(x);
	return (
		<div ref={probe} className="text-meta">
			<ChartSurface title={title} tableId={tableId} width={width} height={bottom + MARGIN.bottom} kind="scatter"
				columns={data.length} rows={1} hit={hit} tip={tip}
				announce={({ c }) => join(...lines(order[c]).map((l) => l.value), position(c, data.length, locale))}
				header={<HeaderRow start={yLabel} end={xLabel} />}>
				{(active) => <>
					<YAxis ticks={yTicks} y={y} left={left} right={left + plotW} format={yFormat} />
					{xs.map((px, i) => <line key={i} x1={snap(px)} x2={snap(px)} y1={top} y2={bottom} strokeWidth={1} className="stroke-hairline" />)}
					<XLabels labels={xTicks.values.map(xFormat)} xs={xs} slot={plotW / Math.max(1, xs.length - 1)} measure={measure} edge top={bottom} width={width} />
					{order.map((i, c) => {
						const on = active?.c === c;
						return <circle key={i} cx={x(data[i][0])} cy={y(data[i][1])} r={on ? 6 : 4} fillOpacity={0.85}
							className={`fill-chart-1 ${dim(Boolean(active) && !on)}`} />;
					})}
				</>}
			</ChartSurface>
		</div>
	);
}

// ---- donut ----------------------------------------------------------------------

/** Resolved slot colours, re-read when the theme attribute changes. */
function useSlotColors(ref: React.RefObject<HTMLElement | null>) {
	const [colors, setColors] = useState<string[]>([]);
	useLayoutEffect(() => {
		const read = () => {
			const node = ref.current;
			if (!node) return;
			const style = getComputedStyle(node);
			const next = FILL.map((_, i) => style.getPropertyValue(`--lo-chart-${i + 1}`).trim());
			setColors((prev) => (prev.join() === next.join() ? prev : next));
		};
		read();
		if (typeof MutationObserver === "undefined") return;
		const observer = new MutationObserver(read);
		observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
		return () => observer.disconnect();
	}, [ref]);
	return colors;
}

/** A ring sector with a constant 1.5px angular gap (geometry, not a stroke). */
function sector(cx: number, cy: number, R: number, r: number, a0: number, a1: number, whole: boolean): string {
	const point = (radius: number, angle: number) => `${cx + radius * Math.sin(angle)},${cy - radius * Math.cos(angle)}`;
	if (whole) {
		return `M${point(R, 0)}A${R},${R} 0 1 1 ${point(R, Math.PI)}A${R},${R} 0 1 1 ${point(R, 0)}Z`
			+ `M${point(r, 0)}A${r},${r} 0 1 0 ${point(r, Math.PI)}A${r},${r} 0 1 0 ${point(r, 0)}Z`;
	}
	const gapR = 0.75 / R;
	const gapr = 0.75 / r;
	const [o0, o1] = [a0 + gapR, Math.max(a0 + gapR, a1 - gapR)];
	const [i0, i1] = [a0 + gapr, Math.max(a0 + gapr, a1 - gapr)];
	const large = (end: number, start: number) => (end - start > Math.PI ? 1 : 0);
	return `M${point(R, o0)}A${R},${R} 0 ${large(o1, o0)} 1 ${point(R, o1)}L${point(r, i1)}A${r},${r} 0 ${large(i1, i0)} 0 ${point(r, i0)}Z`;
}

function DonutChart({ value, tableId, width }: { value: CategoryChart; tableId: string; width: number }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const probe = useRef<HTMLDivElement>(null);
	const measure = useMeasure(probe, width);
	const colors = useSlotColors(probe);
	const { data, unit, title } = value;
	const slices = useMemo(() => foldSlices(data), [data]);
	const total = data.reduce((sum, [, v]) => sum + v, 0);
	const wide = width >= NARROW_BELOW;
	const D = donutDiameter(width);
	const R = D / 2;
	const r = R * 0.6;
	const height = D + 24;
	const cx = width / 2;
	const cy = height / 2;
	const labelOf = (i: number) => (slices[i].source === -1 ? t("relay.chart.other", { count: String(slices[i].folded!.length) }) : slices[i].label);
	const fill = (i: number) => (slices[i].source === -1 ? OTHER_FILL : FILL[i]);
	const swatch = (i: number) => (slices[i].source === -1 ? OTHER_SWATCH : SWATCH[i]);
	const fraction = (i: number) => slices[i].value / total;
	const pct = (i: number) => shareText(locale, fraction(i));
	// "Other" is a sum of authored values: print it at their precision.
	const ringPct = (f: number) => new Intl.NumberFormat(locale, { style: "percent", maximumFractionDigits: 0 }).format(f);
	const valueText = (i: number) => withUnit(slices[i].folded
		? exactSum(locale, slices[i].folded!.map(({ value: v }) => v))
		: exactValue(locale, slices[i].value), unit);

	const angles = useMemo(() => {
		let at = 0;
		return slices.map((s) => {
			const from = at;
			at += (s.value / total) * 2 * Math.PI;
			return [from, at] as const;
		});
	}, [slices, total]);
	const mid = (i: number) => (angles[i][0] + angles[i][1]) / 2;

	const hit = (px: number, py: number): Mark | null => {
		const dx = px - cx;
		const dy = py - cy;
		const dist = Math.hypot(dx, dy);
		if (dist < r - 4 || dist > R + 8) return null;
		const angle = (Math.atan2(dx, -dy) + 2 * Math.PI) % (2 * Math.PI);
		const c = angles.findIndex(([a, b]) => angle >= a && angle < b);
		return c === -1 ? null : { c, s: 0 };
	};
	const tipFor = (i: number): Tip => {
		const folded = slices[i].folded;
		const notes = [t("relay.chart.share", { share: pct(i) })];
		const rows: TipRow[] = [{ swatch: swatch(i), value: valueText(i) }];
		if (folded) {
			// "Other" lists what it folded (≤ 8, then "+n more"; spec §3 "Fold").
			for (const item of folded.slice(0, 8)) rows.push({ label: item.label, value: withUnit(exactValue(locale, item.value), unit) });
			if (folded.length > 8) notes.push(t("relay.chart.more", { count: new Intl.NumberFormat(locale).format(folded.length - 8) }));
		}
		return { heading: labelOf(i), rows, notes };
	};
	// Narrow/touch: no floating tooltip — the centre readout IS the tooltip,
	// so a finger never covers it (spec §3/§5).
	const tip = ({ c }: Mark) => {
		if (!wide) return null;
		const a = mid(c);
		const px = cx + (R + 8) * Math.sin(a);
		const py = cy - (R + 8) * Math.cos(a);
		return { tip: tipFor(c), anchor: { x: px, y: Math.min(py, cy - R), below: Math.max(py, cy - R) } };
	};

	// Outside labels on wide columns for slices ≥ 3% (spec §3), relaxed apart.
	const outside = useMemo(() => {
		if (!wide) return [];
		const items = slices.map((_, i) => i).filter((i) => fraction(i) >= 0.03);
		const placed: { i: number; y: number; right: boolean }[] = [];
		for (const right of [false, true]) {
			const list = items.filter((i) => (Math.sin(mid(i)) >= 0) === right);
			const ys = relaxLabels(list.map((i) => cy - (R + 16) * Math.cos(mid(i))), 18, 8, height - 8);
			list.forEach((i, k) => placed.push({ i, y: ys[k], right }));
		}
		return placed;
	}, [wide, slices, angles, R, cy, height]);
	const small = wide ? slices.map((_, i) => i).filter((i) => fraction(i) < 0.03) : [];
	// "Also: {items}" with each item hoverable: split the translated sentence
	// at a sentinel so the locale decides where the list sits in it.
	const alsoParts = t("relay.chart.also", { items: "\u0000" }).split("\u0000");

	const readout = (active: Mark | null) => {
		const lines = active
			? [labelOf(active.c), valueText(active.c), pct(active.c)]
			: [t("relay.chart.total"), withUnit(exactSum(locale, data.map(([, v]) => v)), unit)];
		const maxW = r * 1.6;
		const start = cy - ((lines.length - 1) * 16) / 2;
		return lines.map((line, k) => (
			<text key={k} x={cx} y={start + k * 16} textAnchor="middle" dominantBaseline="central"
				className={`text-meta tabular-nums ${k === 1 ? "fill-ink font-semibold" : "fill-ink-muted"}`}>
				{truncate(line, measure(line), maxW, measure)}
			</text>
		));
	};
	const announce = ({ c }: Mark) => join(labelOf(c), valueText(c), pct(c), position(c, slices.length, locale));

	return (
		<div ref={probe} className="text-meta">
			<ChartSurface title={title} tableId={tableId} width={width} height={height} kind="pie" columns={slices.length} rows={1}
				hit={hit} tip={tip} announce={announce}
				after={(active, setActive) => <>
					{!wide && (
						// Narrow legend: 2-column grid, rows interactive (spec §3 "Narrow").
						<div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-2">
							{slices.map((_, i) => (
								<div key={i} className={`flex min-w-0 items-center gap-1.5 ${dim(Boolean(active) && active?.c !== i)}`}
									onPointerEnter={() => setActive({ c: i, s: 0 })} onPointerLeave={() => setActive(null)}>
									<span aria-hidden="true" className={`size-2.5 shrink-0 rounded-[2px] ${swatch(i)}`} />
									<span className="min-w-0 truncate text-ink"><bdi>{labelOf(i)}</bdi></span>
									<span className="ms-auto text-ink-muted tabular-nums">{pct(i)}</span>
								</div>
							))}
						</div>
					)}
					{small.length > 0 && (
						<div className="mt-2 text-ink-muted">
							{alsoParts[0]}
							{small.map((i, k) => (
								<span key={i} onPointerEnter={() => setActive({ c: i, s: 0 })} onPointerLeave={() => setActive(null)}>
									<bdi>{labelOf(i)}</bdi> {pct(i)}{k < small.length - 1 ? ", " : ""}
								</span>
							))}
							{alsoParts[1]}
						</div>
					)}
				</>}>
				{(active, setActive) => <>
					{slices.map((_, i) => (
						<path key={i} d={sector(cx, cy, R, r, angles[i][0], angles[i][1], slices.length === 1)} fillRule="evenodd"
							className={`${fill(i)} ${dim(Boolean(active) && active?.c !== i)}`} />
					))}
					{!wide && slices.map((_, i) => {
						// In-ring % only where black or white text reaches 4.5:1 on the slot.
						if (fraction(i) < 0.08 || slices[i].source === -1) return null;
						const best = bestTextOn(colors[i] ?? "");
						if (!best || best.ratio < 4.5) return null;
						const a = mid(i);
						const rr = (R + r) / 2;
						return <text key={`p${i}`} x={cx + rr * Math.sin(a)} y={cy - rr * Math.cos(a)} textAnchor="middle" dominantBaseline="central"
							fill={best.color} className={`text-meta font-semibold tabular-nums ${dim(Boolean(active) && active?.c !== i)}`}>
							{ringPct(fraction(i))}
						</text>;
					})}
					{outside.map(({ i, y: ly, right }) => {
						const a = mid(i);
						const sx = cx + (R + 4) * Math.sin(a);
						const sy = cy - (R + 4) * Math.cos(a);
						const ex = cx + (R + 16) * Math.sin(a);
						const tx = ex + (right ? 12 : -12);
						const room = right ? width - tx - 4 : tx - 4;
						const pctText = pct(i);
						const label = truncate(labelOf(i), measure(labelOf(i)), room - measure(pctText) - 8, measure);
						return <g key={`l${i}`} className={dim(Boolean(active) && active?.c !== i)}
							onPointerEnter={() => setActive({ c: i, s: 0 })}>
							<polyline points={`${sx},${sy} ${ex},${ly} ${tx},${ly}`} strokeWidth={1} strokeOpacity={0.6} className="fill-none stroke-ink-muted" />
							<text x={tx + (right ? 4 : -4)} y={ly} dominantBaseline="central" textAnchor={right ? "start" : "end"} className="text-meta">
								<tspan className="fill-ink">{label}</tspan><tspan className="fill-ink-muted" dx={6}>{pctText}</tspan>
							</text>
						</g>;
					})}
					{readout(active)}
				</>}
			</ChartSurface>
		</div>
	);
}

// ---- heatmap ----------------------------------------------------------------------

function HeatmapChart({ value, tableId, width }: { value: Heatmap; tableId: string; width: number }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const probe = useRef<HTMLDivElement>(null);
	const measure = useMeasure(probe, width);
	const hatch = `lo-hatch-${useId().replace(/[^a-zA-Z0-9]/g, "")}`;
	const { columns, rowLabels, rows, unit, title } = value;
	const numbers = rows.flat().filter((v): v is number => v !== null);
	const lo = numbers.length ? Math.min(...numbers) : 0;
	const hi = numbers.length ? Math.max(...numbers) : 0;
	const rowH = width >= NARROW_BELOW ? 28 : 24;
	const left = Math.min(Math.round(width * 0.4), gutter(rowLabels.map(axisLabel), measure));
	const cellW = Math.max(4, (width - left) / columns.length);
	const gridH = rows.length * rowH;
	const height = gridH + MARGIN.bottom;
	// Sequential fill (spec §4): color-mix(in oklab, accent t%, surface), t = 12 + 88 × fraction.
	const mix = (tt: number) => `color-mix(in oklab, var(--lo-accent) ${tt}%, var(--lo-surface))`;
	const fraction = (v: number) => (hi === lo ? 1 : (v - lo) / (hi - lo));
	const hit = (px: number, py: number): Mark | null => {
		if (px < left) return null;
		const c = Math.min(columns.length - 1, Math.floor((px - left) / cellW));
		const s = Math.min(rows.length - 1, Math.floor(py / rowH));
		return s < 0 ? null : { c, s };
	};
	const cellText = (c: number, s: number) => {
		const v = rows[s][c];
		return v === null ? t("relay.chart.no_data") : withUnit(exactValue(locale, v), unit);
	};
	const tip = ({ c, s }: Mark) => ({
		tip: { heading: columns[c], rows: [{ label: rowLabels[s], value: cellText(c, s) }] },
		anchor: { x: left + cellW * (c + 0.5), y: s * rowH, below: (s + 1) * rowH },
	});
	const legend = numbers.length > 0 && (
		<span className="ms-auto inline-flex items-center gap-2 tabular-nums">
			<bdi>{withUnit(exactValue(locale, lo), unit)}</bdi>
			<span aria-hidden="true" className="inline-block h-2 w-30 rounded-full" style={{ backgroundImage: `linear-gradient(to right, ${mix(12)}, ${mix(100)})` }} />
			<bdi>{withUnit(exactValue(locale, hi), unit)}</bdi>
		</span>
	);
	const xs = columns.map((_, c) => left + cellW * (c + 0.5));
	return (
		<div ref={probe} className="text-meta">
			<ChartSurface title={title} tableId={tableId} width={width} height={height} kind="heatmap" columns={columns.length} rows={rows.length}
				hit={hit} tip={tip}
				announce={({ c, s }) => join(rowLabels[s], columns[c], cellText(c, s), position(s * columns.length + c, rows.length * columns.length, locale))}
				header={<HeaderRow>{legend}</HeaderRow>}>
				{(active) => <>
					<defs>
						<pattern id={hatch} width={4} height={4} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
							<line x1={0} y1={0} x2={0} y2={4} strokeWidth={1} className="stroke-hairline-strong" />
						</pattern>
					</defs>
					{rowLabels.map((label, s) => (
						<text key={s} x={left - 8} y={s * rowH + rowH / 2} textAnchor="end" dominantBaseline="central" className="fill-ink-muted text-meta">
							{truncate(axisLabel(label), measure(axisLabel(label)), left - 8, measure)}
						</text>
					))}
					{rows.map((row, s) => row.map((v, c) => (
						<rect key={`${s}-${c}`} x={left + c * cellW + 1} y={s * rowH + 1} width={cellW - 2} height={rowH - 2} rx={3}
							fill={v === null ? `url(#${hatch})` : undefined} style={v === null ? undefined : { fill: mix(12 + 88 * fraction(v)) }}
							className={dim(Boolean(active) && !(active?.c === c && active?.s === s))} />
					)))}
					<XLabels labels={columns} xs={xs} slot={cellW} measure={measure} edge={false} top={gridH - 0} width={width} />
				</>}
			</ChartSurface>
		</div>
	);
}

// ---- entry points -------------------------------------------------------------------

const header = (unit?: string) => unit ?? "";

/** Category charts: line, bar (vertical or horizontal), histogram, pie. */
export function CategoryPlot({ value }: { value: CategoryChart }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const table: DataTable = {
		columns: [header(value.unit)],
		rows: value.data.map(([label, v]) => ({ label, cells: [exactValue(locale, v)] })),
	};
	return <ChartBlock title={value.title} table={table}>
		{({ width, tableId }) => <CategoryBody value={value} width={width} tableId={tableId} />}
	</ChartBlock>;
}

function CategoryBody({ value, width, tableId }: { value: CategoryChart; width: number; tableId: string }) {
	const horizontal = useHorizontal(value.data, width);
	const labels = useMemo(() => value.data.map(([label]) => label), [value.data]);
	const series = useMemo(() => [{ label: value.title, values: value.data.map(([, v]) => v) }], [value]);
	if (value.type === "pie") return <DonutChart value={value} tableId={tableId} width={width} />;
	if (value.type === "bar" && horizontal) return <HorizontalBars value={value} tableId={tableId} width={width} />;
	return <CartesianChart kind={value.type} title={value.title} unit={value.unit} labels={labels} series={series}
		mode={value.type === "line" ? "line" : "bar"} tableId={tableId} width={width} />;
}

export function ScatterPlot({ value }: { value: Scatter }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const table: DataTable = {
		columns: [value.xLabel, value.yLabel],
		rows: value.data.map(([a, b]) => ({ cells: [exactValue(locale, a), exactValue(locale, b)] })),
	};
	return <ChartBlock title={value.title} table={table}>
		{({ width, tableId }) => <ScatterChart value={value} tableId={tableId} width={width} />}
	</ChartBlock>;
}

export function HeatmapPlot({ value }: { value: Heatmap }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const table: DataTable = {
		corner: value.unit,
		columns: value.columns,
		// An empty cell is "no value", not zero: left blank in the table.
		rows: value.rows.map((row, i) => ({ label: value.rowLabels[i], cells: row.map((v) => (v === null ? "" : exactValue(locale, v))) })),
	};
	return <ChartBlock title={value.title} table={table}>
		{({ width, tableId }) => <HeatmapChart value={value} tableId={tableId} width={width} />}
	</ChartBlock>;
}

export function SeriesPlot({ value }: { value: InlineSeries }) {
	const { uiLocale: locale } = useLocaleSnapshot();
	const dual = value.mode === "dual";
	const table: DataTable = {
		columns: value.series.map(({ label, axis }) => {
			const u = dual ? (axis === "right" ? value.rightUnit : value.unit) : value.unit;
			return u ? `${label} (${u})` : label;
		}),
		rows: value.labels.map((label, c) => ({ label, cells: value.series.map(({ values }) => exactValue(locale, values[c])) })),
	};
	const series = useMemo(() => value.series.map(({ label, values, axis }) => ({ label, values, right: axis === "right" })), [value.series]);
	return <ChartBlock title={value.title} table={table}>
		{({ width, tableId }) => <CartesianChart kind={value.mode} title={value.title} unit={value.unit}
			rightUnit={dual ? value.rightUnit : undefined} labels={value.labels} series={series} mode={value.mode}
			tableId={tableId} width={width} />}
	</ChartBlock>;
}

