type SeriesAxis = "left" | "right";
type Series = { label: string; values: number[]; axis?: SeriesAxis };

export type InlineSeries = {
	type: "series";
	title: string;
	mode: "grouped" | "stacked" | "percent" | "dual";
	labels: string[];
	series: Series[];
	unit?: string;
	rightUnit?: string;
};

const MAX_CATEGORIES = 32;
const MAX_SERIES = 4;
const isText = (value: unknown): value is string =>
	typeof value === "string" && value.length <= 256;
const isLabel = (value: unknown): value is string =>
	isText(value) && value.trim().length > 0;
const isNumber = (value: unknown): value is number =>
	typeof value === "number" && Number.isFinite(value);

export function decodeInlineSeries(
	value: Record<string, unknown>,
): InlineSeries | null {
	const { type, title, mode, labels, unit, rightUnit } = value;
	if (
		type !== "series" ||
		!isLabel(title) ||
		(mode !== "grouped" &&
			mode !== "stacked" &&
			mode !== "percent" &&
			mode !== "dual") ||
		(unit !== undefined && !isText(unit)) ||
		(rightUnit !== undefined && !isText(rightUnit)) ||
		!Array.isArray(labels) ||
		labels.length === 0 ||
		labels.length > MAX_CATEGORIES ||
		!Array.from(labels).every(isLabel) ||
		!Array.isArray(value.series) ||
		value.series.length === 0 ||
		value.series.length > MAX_SERIES
	) {
		return null;
	}

	const series: Series[] = [];
	for (const input of value.series) {
		if (!input || typeof input !== "object" || Array.isArray(input))
			return null;
		const { label, values, axis } = input as Record<string, unknown>;
		if (
			!isLabel(label) ||
			!Array.isArray(values) ||
			values.length !== labels.length ||
			!Array.from(values).every(isNumber) ||
			(axis !== undefined && axis !== "left" && axis !== "right") ||
			(mode === "dual" && axis === undefined)
		) {
			return null;
		}
		series.push(
			axis === undefined ? { label, values } : { label, values, axis },
		);
	}

	if (
		mode === "dual" &&
		(!isLabel(unit) ||
			!isLabel(rightUnit) ||
			!series.some(({ axis }) => axis === "left") ||
			!series.some(({ axis }) => axis === "right"))
	)
		return null;

	// With no negatives, a positive member proves a positive total without
	// adding finite values that might overflow Number.MAX_VALUE.
	if (
		mode === "percent" &&
		(series.some(({ values }) => values.some((n) => n < 0)) ||
			labels.some(
				(_, index) => !series.some(({ values }) => values[index] > 0),
			))
	)
		return null;

	return {
		type,
		title,
		mode,
		labels,
		series,
		...(unit === undefined ? {} : { unit }),
		...(rightUnit === undefined ? {} : { rightUnit }),
	};
}
