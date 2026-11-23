import type { ReactNode } from "react";
import type { CategoryChart, Scatter } from "../inline-ui-parts";
import type { InlineSeries } from "../inline-series";

const WIDTH = 320;
const HEIGHT = 100;
const PAD = 7;
const PLOT_WIDTH = WIDTH - 2 * PAD;
const PLOT_HEIGHT = HEIGHT - 2 * PAD;
const INKS = ["fill-accent", "fill-info", "fill-success", "fill-warning"];

// Normalize finite input before range arithmetic, including opposite extremes.
// Use a common scale for every series sharing an axis, never per-series autoscale.
function axis(values: readonly number[]) {
	const scale = Math.max(...values.map(Math.abs)) || 1;
	const low = Math.min(...values) / scale;
	const high = Math.max(...values) / scale;
	const span = high - low || 1;
	return (n: number) => PAD + PLOT_HEIGHT * (1 - (high === low ? 0.5 : (n / scale - low) / span));
}

function plotRange(values: readonly number[], height: number, lower: number, upper: number) {
	const min = Math.min(...values);
	const max = Math.max(...values);
	return (n: number) => min === max ? lower + height / 2 : lower + (n - min) / (max - min) * (upper - lower);
}

function Plot({ children, type }: { children: ReactNode; type: string }) {
	// The chart is a visual summary; the immediately adjacent named table is
	// the same data in reading order with its units, labels and exact values.
	return <svg aria-hidden="true" data-inline-plot={type} viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
		className="mx-auto block max-h-28 w-full">
		{children}
	</svg>;
}

export function PointPlot({ value }: { value: CategoryChart | Scatter }) {
	const y = axis(value.data.map(([, n]) => n));
	const scales = value.type === "scatter" ? value.data.map(([n]) => n) : value.data.map((_, i) => i);
	const maxAbs = Math.max(...scales.map(Math.abs)) || 1;
	const normalized = scales.map((n) => n / maxAbs);
	const x = plotRange(normalized, PLOT_WIDTH, PAD, WIDTH - PAD);
	const points = value.data.map(([, n], i) => [x(normalized[i]), y(n)] as const);
	return <Plot type={value.type}>
		<path d={`M ${PAD} ${PAD} V ${HEIGHT - PAD} H ${WIDTH - PAD}`} className="fill-none stroke-control" />
		{value.type === "line" && <polyline points={points.map(([a, b]) => `${a},${b}`).join(" ")}
			className="fill-none stroke-accent" strokeWidth={2} />}
		{points.map(([a, b], i) => <circle key={i} cx={a} cy={b} r={3} className="fill-accent" />)}
	</Plot>;
}

export function ProportionPlot({ value }: { value: CategoryChart }) {
	const values = value.data.map(([, n]) => n);
	const max = Math.max(...values);
	const total = values.reduce((sum, n) => sum + n / max, 0);
	let offset = 0;
	// Draw in the table's row order, starting at twelve o'clock. The exact
	// values remain in the keys rather than crowding the phone plot perimeter.
	return <svg aria-hidden="true" data-inline-plot="pie" viewBox="0 0 100 100"
		className="mx-auto block size-24">
		{values.map((n, i) => {
			const fraction = n / max / total;
			const sweep = fraction * 2 * Math.PI;
			const from = offset - Math.PI / 2;
			const to = from + sweep;
			offset += sweep;
			const ink = INKS[i % INKS.length];
			if (fraction >= 1) return <circle key={i} cx={50} cy={50} r={46} className={ink} />;
			const polar = (theta: number) => `${50 + 46 * Math.cos(theta)} ${50 + 46 * Math.sin(theta)}`;
			return <path key={i} d={`M 50 50 L ${polar(from)} A 46 46 0 ${fraction > 0.5 ? 1 : 0} 1 ${polar(to)} Z`}
				className={`${ink} stroke-surface`} />;
		})}
	</svg>;
}

export function SeriesPlot({ value }: { value: InlineSeries }) {
	const { series, mode, labels } = value;
	// Two independently normalized scales cannot share an unlabeled plot. The
	// enclosing preview shows both series as an exact, unit-labeled table.
	if (mode === "dual") return null;
	const grouped = mode === "grouped";
	const magnitude = (values: number[]) => Math.max(...values.map(Math.abs)) || 1;
	const scale = magnitude(series.flatMap(({ values }) => values));
	const offsets = labels.map((_, i) => {
		const base = mode === "percent" ? magnitude(series.map(({ values }) => values[i])) : scale;
		const points = series.map(({ values }) => values[i] / base);
		const total = mode === "percent" ? points.reduce((sum, n) => sum + n, 0) : 1;
		let positive = 0;
		let negative = 0;
		return points.map((point) => {
			const end = point / total;
			const start = grouped ? 0 : end < 0 ? negative : positive;
			if (!grouped && end < 0) negative += end;
			if (!grouped && end >= 0) positive += end;
			return { start, end: start + end };
		});
	});
	const y = axis([0, ...offsets.flatMap((points) => points.map(({ end }) => end))]);
	const step = PLOT_WIDTH / labels.length;
	const barWidth = step * 0.78 / (grouped ? series.length : 1);
	return <Plot type={mode}>
		<path d={`M ${PAD} ${y(0)} H ${WIDTH - PAD}`} className="fill-none stroke-control" />
		{offsets.flatMap((points, i) => points.map(({ start, end }, j) => {
			if (start === end) return null;
			const top = Math.min(y(start), y(end));
			return <rect key={`${i}-${j}`} x={PAD + step * (i + 0.11) + (grouped ? j * barWidth : 0)}
				y={top} width={barWidth} height={Math.abs(y(end) - y(start))}
				className={grouped ? INKS[j] : `${INKS[j]} stroke-surface`}
				vectorEffect={grouped ? undefined : "non-scaling-stroke"} />;
		}))}
	</Plot>;
}

export function SeriesKey({ index }: { index: number }) {
	return <svg aria-hidden="true" viewBox="0 0 8 8" className="me-1 inline-block size-2">
		<rect width={8} height={8} rx={1} className={INKS[index % INKS.length]} />
	</svg>;
}
