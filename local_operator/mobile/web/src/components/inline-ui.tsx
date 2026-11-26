import { useId } from "react";
import type { AssistantPart, InlineUi } from "../inline-ui-parts";
import { CategoryPlot, HeatmapPlot, ScatterPlot, SeriesPlot } from "./inline-ui-charts";
import { Markdown } from "./markdown";

type Cell = string | number | null;

/** Plain data grids (`table`, `spreadsheet`); charts have their own block. */
function Preview({ title, columns, rows }: {
	title: string;
	columns: string[];
	rows: Cell[][];
}) {
	const captionId = useId();
	const wide = columns.length > 3;
	const cellClass = "px-2 py-1 text-start align-top wrap-anywhere";
	return (
		<div className="min-w-0 rounded-md bg-surface text-body-sm text-ink">
			<div id={captionId} className="px-2 pt-2 pb-1 text-start font-medium wrap-anywhere">
				<bdi>{title}</bdi>
			</div>
			{/* Wide data grids scroll inside the transcript, including by keyboard. */}
			<div className={wide ? "overflow-x-auto rounded-md focus-visible:outline-accent" : undefined} {...(wide ? {
				role: "region", "aria-labelledby": captionId, tabIndex: 0,
			} : {})}>
				<table aria-labelledby={captionId} className="w-full table-fixed border-collapse" style={wide ? { minWidth: `${columns.length * 5}rem` } : undefined}>
					{columns.some(Boolean) && (
						<thead className="text-ink-muted">
							<tr>
								{columns.map((label, i) => <th key={i} scope="col" className={`${cellClass} font-medium`}><bdi>{label}</bdi></th>)}
							</tr>
						</thead>
					)}
					<tbody>
						{rows.map((row, i) => (
							<tr key={i} className="border-t border-hairline">
								{row.map((cell, j) => (
									<td key={j} className={`${cellClass} tabular-nums`}>
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

function InlineUiPreview({ value }: { value: InlineUi }) {
	const { title } = value;
	switch (value.type) {
		// Charts render through the interactive chart block (inline-chart spec):
		// hover/tap/keyboard give each mark's exact value and the full value
		// grid sits behind the collapsed "Chart data" disclosure.
		case "bar":
		case "line":
		case "pie":
		case "histogram":
			return <CategoryPlot value={value} />;
		case "scatter":
			return <ScatterPlot value={value} />;
		case "heatmap":
			return <HeatmapPlot value={value} />;
		case "series":
			return <SeriesPlot value={value} />;
		case "table":
		case "spreadsheet":
			return <Preview title={title} columns={value.columns} rows={value.rows} />;
	}
}

export function AssistantContent({ text, overlay, live = false, askChips = false }: {
	text: string;
	overlay?: readonly AssistantPart[];
	live?: boolean;
	/** Chip ask ids in prose (`Markdown askChips`). Passed to EVERY prose
	 *  segment, so a reply that carries widgets chips its ask ids exactly like
	 *  a prose-only reply; widgets themselves never chip. */
	askChips?: boolean;
}) {
	// Only the authenticated renderer contract can create widgets. In its
	// absence the row is prose, including literal fences in older journals.
	if (!overlay?.length) return <Markdown text={text} live={live} askChips={askChips} />;
	return (
		<div className="flex min-w-0 flex-col gap-2">
			{/* Keep each prose segment's own append boundary; a widget is never a Markdown fade span. */}
			{overlay.map((part, i) => part.kind === "markdown"
				? <Markdown key={`prose-${i}`} text={part.text} live={live} askChips={askChips} />
				: <InlineUiPreview key={`preview-${i}`} value={part.value} />)}
		</div>
	);
}
