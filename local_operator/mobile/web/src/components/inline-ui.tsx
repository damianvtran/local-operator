import { useId, type ReactNode } from "react";
import type { AssistantPart, InlineUi } from "../inline-ui-parts";
import { PointPlot, ProportionPlot, SeriesKey, SeriesPlot } from "./inline-ui-charts";
import { Markdown } from "./markdown";

type Cell = string | number | null;

function Preview({ title, unit, columns, rows, rowLabels, decorate, plot, keyBy }: {
	title: string;
	unit?: string;
	columns: string[];
	rows: Cell[][];
	rowLabels?: string[];
	decorate?: (cell: number) => ReactNode;
	plot?: ReactNode;
	keyBy?: "row" | "column";
}) {
	const captionId = useId();
	const columnCount = columns.length + (rowLabels ? 1 : 0);
	const wide = columnCount > 3;
	const cellClass = "px-2 py-1 text-start align-top wrap-anywhere";
	return (
		<div className="min-w-0 rounded-md bg-surface text-body-sm text-ink">
			<div id={captionId} className="px-2 pt-2 pb-1 text-start font-medium wrap-anywhere">
				<bdi>{title}</bdi>{unit && <> <bdi>({unit})</bdi></>}
			</div>
			{plot && <div className="px-2 pb-1">{plot}</div>}
			{/* Wide data grids scroll inside the transcript, including by keyboard. */}
			<div className={wide ? "overflow-x-auto rounded-md focus-visible:outline-accent" : undefined} {...(wide ? {
				role: "region", "aria-labelledby": captionId, tabIndex: 0,
			} : {})}>
				<table aria-labelledby={captionId} className="w-full table-fixed border-collapse" style={wide ? { minWidth: `${columnCount * 5}rem` } : undefined}>
					{columns.some(Boolean) && (
						<thead className="text-ink-muted">
							<tr>
								{rowLabels && <td />}
								{columns.map((label, i) => <th key={i} scope="col" className={`${cellClass} font-medium`}>{keyBy === "column" && <SeriesKey index={i} />}<bdi>{label}</bdi></th>)}
							</tr>
						</thead>
					)}
					<tbody>
						{rows.map((row, i) => (
							<tr key={i} className="border-t border-hairline">
								{rowLabels && <th scope="row" className={`${cellClass} font-normal text-ink-muted`}>{keyBy === "row" && <SeriesKey index={i} />}<bdi>{rowLabels[i]}</bdi></th>}
								{row.map((cell, j) => (
									<td key={j} className={`${cellClass} tabular-nums`}>
										{typeof cell === "number" && decorate?.(cell)}
										<bdi>{cell}</bdi>
									</td>
								))}
							</tr>
						))}
					</tbody>
				</table>
			</div>
		</div>
	);
}

function bars(values: number[]) {
	const scale = Math.max(...values.map(Math.abs)) || 1;
	const lo = Math.min(0, ...values) / scale;
	const hi = Math.max(0, ...values) / scale;
	const span = hi - lo || 1;
	return (cell: number) => {
		const n = cell / scale;
		return (
			<div aria-hidden="true" dir="ltr" className="relative mt-1 h-1.5 rounded-sm bg-sunken">
				<span className="absolute h-full rounded-sm bg-accent" style={{
					left: `${(Math.min(0, n) - lo) / span * 100}%`,
					width: `${Math.abs(n) / span * 100}%`,
				}} />
			</div>
		);
	};
}

function heat(values: number[]) {
	// Normalize before taking the range so opposite finite extremes don't overflow.
	const scale = Math.max(...values.map(Math.abs)) || 1;
	const lo = Math.min(...values) / scale;
	const hi = Math.max(...values) / scale;
	return (cell: number) => {
		const ratio = hi === lo ? 1 : (cell / scale - lo) / (hi - lo);
		return (
			<span aria-hidden="true" className="me-1 inline-block size-2.5 rounded-sm border border-control align-baseline" style={{
				backgroundColor: `color-mix(in srgb, var(--lo-accent) ${ratio * 100}%, var(--lo-surface))`,
			}} />
		);
	};
}

function InlineUiPreview({ value }: { value: InlineUi }) {
	const { title } = value;
	switch (value.type) {
		case "bar":
		case "line":
		case "pie":
		case "histogram":
			return <Preview title={title} unit={value.unit} columns={[""]}
				rowLabels={value.data.map(([label]) => label)} rows={value.data.map(([, n]) => [n])}
				decorate={value.type === "bar" || value.type === "histogram" ? bars(value.data.map(([, n]) => n)) : undefined}
				plot={value.type === "line" ? <PointPlot value={value} /> : value.type === "pie" ? <ProportionPlot value={value} /> : undefined}
				keyBy={value.type === "pie" ? "row" : undefined} />;
		case "scatter":
			return <Preview title={title} columns={[value.xLabel, value.yLabel]} rows={value.data} plot={<PointPlot value={value} />} />;
		case "heatmap":
			return <Preview title={title} unit={value.unit} columns={value.columns} rowLabels={value.rowLabels}
				rows={value.rows} decorate={heat(value.rows.flat().filter((cell): cell is number => cell !== null))} />;
		case "series":
			return <Preview title={title} unit={value.mode === "dual" ? undefined : value.unit} rowLabels={value.labels}
				columns={value.series.map(({ label, axis }) => value.mode === "dual"
					? `${label} (${axis === "right" ? value.rightUnit : value.unit})` : label)}
				rows={value.labels.map((_, i) => value.series.map(({ values }) => values[i]))}
				plot={value.mode === "dual" ? undefined : <SeriesPlot value={value} />}
				keyBy={value.mode === "dual" ? undefined : "column"} />;
		case "table":
		case "spreadsheet":
			return <Preview title={title} columns={value.columns} rows={value.rows} />;
	}
}

export function AssistantContent({ text, overlay }: {
	text: string;
	overlay?: readonly AssistantPart[];
}) {
	// Only the authenticated renderer contract can create widgets. In its
	// absence the row is prose, including literal fences in older journals.
	if (!overlay?.length) return <Markdown text={text} />;
	return (
		<div className="flex min-w-0 flex-col gap-2">
			{overlay.map((part, i) => part.kind === "markdown"
				? <Markdown key={`prose-${i}`} text={part.text} />
				: <InlineUiPreview key={`preview-${i}`} value={part.value} />)}
		</div>
	);
}
