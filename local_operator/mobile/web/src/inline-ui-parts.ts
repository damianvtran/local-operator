import { type InlineSeries, decodeInlineSeries } from "./inline-series";

export type CategoryChart = {
	type: "bar" | "line" | "pie" | "histogram";
	title: string;
	unit?: string;
	data: [string, number][];
};
type Table = {
	type: "table";
	title: string;
	columns: string[];
	rows: (string | number)[][];
};
export type Scatter = {
	type: "scatter";
	title: string;
	xLabel: string;
	yLabel: string;
	data: [number, number][];
};
export type Heatmap = {
	type: "heatmap";
	title: string;
	unit?: string;
	columns: string[];
	rowLabels: string[];
	rows: (number | null)[][];
};
type Spreadsheet = {
	type: "spreadsheet";
	title: string;
	columns: string[];
	rows: (string | number | null)[][];
};
export type InlineUi =
	| CategoryChart
	| Scatter
	| Table
	| Spreadsheet
	| Heatmap
	| InlineSeries;
export type AssistantPart =
	| { kind: "markdown"; text: string }
	| { kind: "ui"; value: InlineUi };

const MAX_ROWS = 32;
const MAX_COLUMNS = 8;
const isText = (value: unknown): value is string =>
	typeof value === "string" && value.length <= 256;
const isLabel = (value: unknown): value is string =>
	isText(value) && value.trim().length > 0;
const isNumber = (value: unknown): value is number =>
	typeof value === "number" && Number.isFinite(value);

function decodePayload(value: unknown): InlineUi | null {
	if (!value || typeof value !== "object" || Array.isArray(value)) return null;
	const record = value as Record<string, unknown>;
	const { type, title, unit, data, columns, rows, rowLabels, xLabel, yLabel } =
		record;
	if (type === "series") return decodeInlineSeries(record);
	if (!isLabel(title)) return null;
	if (
		type === "bar" ||
		type === "line" ||
		type === "pie" ||
		type === "histogram"
	) {
		if (
			(unit !== undefined && !isText(unit)) ||
			!Array.isArray(data) ||
			data.length === 0 ||
			data.length > MAX_ROWS ||
			!data.every(
				(point): point is [string, number] =>
					Array.isArray(point) &&
					point.length === 2 &&
					isLabel(point[0]) &&
					isNumber(point[1]) &&
					(type !== "pie" || point[1] > 0) &&
					(type !== "histogram" || point[1] >= 0),
			)
		)
			return null;
		return unit === undefined
			? { type, title, data }
			: { type, title, unit, data };
	}
	if (type === "scatter") {
		if (
			!isLabel(xLabel) ||
			!isLabel(yLabel) ||
			!Array.isArray(data) ||
			data.length === 0 ||
			data.length > MAX_ROWS ||
			!data.every(
				(point): point is [number, number] =>
					Array.isArray(point) && point.length === 2 && point.every(isNumber),
			)
		)
			return null;
		return { type, title, xLabel, yLabel, data };
	}
	if (type === "table" || type === "spreadsheet" || type === "heatmap") {
		if (
			!Array.isArray(columns) ||
			columns.length === 0 ||
			columns.length > MAX_COLUMNS ||
			!columns.every(isLabel) ||
			!Array.isArray(rows) ||
			rows.length > MAX_ROWS ||
			!rows.every((row) => Array.isArray(row) && row.length === columns.length)
		)
			return null;
		if (type === "heatmap") {
			if (
				(unit !== undefined && !isText(unit)) ||
				!Array.isArray(rowLabels) ||
				rowLabels.length !== rows.length ||
				!rowLabels.every(isLabel) ||
				!rows.every((row): row is (number | null)[] =>
					row.every((cell: unknown) => cell === null || isNumber(cell)),
				)
			)
				return null;
			return unit === undefined
				? { type, title, columns, rowLabels, rows }
				: { type, title, unit, columns, rowLabels, rows };
		}
		const isValue = (cell: unknown): cell is string | number =>
			isText(cell) || isNumber(cell);
		if (type === "spreadsheet") {
			if (
				!rows.every((row): row is (string | number | null)[] =>
					row.every((cell: unknown) => cell === null || isValue(cell)),
				)
			)
				return null;
			return { type, title, columns, rows };
		}
		if (!rows.every((row): row is (string | number)[] => row.every(isValue)))
			return null;
		return { type, title, columns, rows };
	}
	return null;
}

export type InlineUiMessages = ReadonlyMap<string, readonly AssistantPart[]>;

function record(value: unknown): Record<string, unknown> | null {
	return value && typeof value === "object" && !Array.isArray(value)
		? value as Record<string, unknown> : null;
}

function splitsSurrogate(text: string, offset: number): boolean {
	const left = text.charCodeAt(offset - 1);
	const right = text.charCodeAt(offset);
	return left >= 0xd800 && left <= 0xdbff && right >= 0xdc00 && right <= 0xdfff;
}

function readOverlay(value: unknown): AssistantPart[] | undefined {
	const envelope = record(value);
	if (envelope?.version !== 1 || typeof envelope.text !== "string" ||
		!Array.isArray(envelope.widgets) || !envelope.widgets.length) return;
	const { text, widgets } = envelope;
	const parts: AssistantPart[] = [];
	let cursor = 0;
	for (const candidate of widgets) {
		const widget = record(candidate);
		const offset = widget?.offset;
		// Slice the envelope's exact UTF-16 text, not the legacy row. A broken
		// placement invalidates this row's overlay, never the projection.
		if (typeof offset !== "number" || !Number.isInteger(offset) ||
			offset < cursor || offset > text.length || splitsSurrogate(text, offset)) return;
		const payload = decodePayload(widget?.payload);
		if (!payload) return;
		if (offset > cursor) parts.push({ kind: "markdown", text: text.slice(cursor, offset) });
		parts.push({ kind: "ui", value: payload });
		cursor = offset;
	}
	if (cursor < text.length) parts.push({ kind: "markdown", text: text.slice(cursor) });
	return parts;
}

export function readInlineUiMessages(
	value: unknown,
	rows: readonly { id: string; kind: string }[],
): InlineUiMessages | undefined {
	const envelope = record(value);
	if (envelope?.version !== 1) return;
	const messages = record(envelope.messages);
	if (!messages) return;
	const decoded = new Map<string, readonly AssistantPart[]>();
	for (const row of rows) {
		// Only live assistant rows may consume a sidecar, never an older page
		// or inherited object keys such as "constructor".
		if (row?.kind !== "assistant" || typeof row.id !== "string" || !row.id ||
			!Object.hasOwn(messages, row.id)) continue;
		const parts = readOverlay(messages[row.id]);
		if (parts) decoded.set(row.id, parts);
	}
	return decoded.size ? decoded : undefined;
}
