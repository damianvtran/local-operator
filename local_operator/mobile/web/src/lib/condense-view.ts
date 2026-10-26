/**
 * Render composition for the relay's condensed transcript.
 *
 * The C1 modules own every condensation decision. This module only joins their
 * two partitions (turn segments and tool folds) into paintable entries, and
 * keeps identity stable across live/replay row ids. Keeping this join pure is
 * important: the first frame and every streamed frame must use exactly the same
 * path, with no effect that first paints flat rows and folds them afterwards.
 */
import type { TranscriptEntry } from "../types";
import { rowKey, toCondenseRecords } from "./condense-records";
import {
	buildRows,
	ledgerName,
	type Row,
} from "./condense-rows";
import {
	foldRuns,
	latchFoldedRuns,
	receiptFoldUnits,
	workedSecondsOf,
	type FoldGroup,
} from "./condense-fold";
import { isFailedResult, isInternalReceipt } from "./condense-predicates";
import { isQuietTurnCall } from "./condense-segments";
import {
	collapsePlan,
	runIsCondensed,
	type RunCollapsePlan,
	type SegmentPlan,
} from "./condense-plan";

export type CondenseEntry =
	| { kind: "row"; key: string; entry: TranscriptEntry; condensed: boolean }
	| { kind: "fold"; key: string; group: Extract<FoldGroup, { kind: "run" }>; children: CondenseEntry[] }
	| {
			kind: "turn";
			key: string;
			run: RunCollapsePlan;
			segment: SegmentPlan;
			children: CondenseEntry[];
	  };

export type CondenseView = {
	entries: CondenseEntry[];
	rows: Row[];
	runs: RunCollapsePlan[];
	foldIds: Set<string>;
	watchedRunId: string | null;
};

/** Production inputs, matching the desktop's fold call site and C1's replay. */
export function relayFoldOptions(liveTurn: boolean, latchedFoldIds: ReadonlySet<string>) {
	return {
		nameOf: (row: Row) => ledgerName(row.record),
		durationOf: workedSecondsOf,
		summaryOf: (row: Row) => row.record.kind === "tool" ? (row.record.summary ?? "") : "",
		// The relay wire can omit args under frame pressure. An absent op is the
		// model's documented generic-label fallback, never an invented operation.
		opOf: () => "",
		failedOf: (row: Row) =>
			row.record.kind === "tool" && isFailedResult(row.record.isError, undefined),
		runningOf: (row: Row) => row.record.kind === "tool" && row.record.phase !== "done",
		executingOf: (row: Row) => row.record.kind === "tool" && row.record.phase === "running",
		startedAtOf: () => null,
		endedAtOf: () => null,
		isFoldable: (row: Row) => row.record.kind === "tool" && !isQuietTurnCall(row.record),
		isReceipt: (row: Row) => isInternalReceipt(row.record),
		receiptSummaryOf: (rows: Row[]) => receiptFoldUnits(rows),
		liveTurn,
		latchedFoldIds,
	};
}

/**
 * Join runs and folds without allowing a row to be claimed twice. A mixed fold
 * straddling a segment boundary renders in place, preserving visible rows; this
 * is the desktop's R1-5 guard rather than a relay-specific shortcut.
 */
export function composeCondensedEntries(
	entries: readonly TranscriptEntry[],
	options: {
		live: boolean;
		openRuns?: ReadonlySet<string>;
		focusHold?: string | null;
		watchedRunId?: string | null;
		olderRowsExist?: boolean;
		previousRows?: Row[];
		latchedFoldIds?: ReadonlySet<string>;
		/** Legacy quiet-group ids: these rows keep the existing Q renderer. */
		quietRowIds?: ReadonlySet<string>;
		/** Runs this pane has already painted condensed; a live newest run may
		   retain those bars, but must not condense for the first time. */
		condensedRunIds?: ReadonlySet<string>;
	},
): CondenseView {
	const records = toCondenseRecords(entries);
	const rows = buildRows(records, options.previousRows ?? []);
	const byId = new Map(entries.map((entry) => [rowKey(entry), entry]));
	const openRuns = new Set(options.openRuns ?? []);
	const quietRowIds = options.quietRowIds ?? new Set<string>();
	const folds = foldRuns(rows, {
		...relayFoldOptions(options.live, options.latchedFoldIds ?? new Set()),
		isFoldable: (row: Row) => !quietRowIds.has(row.record.id) && row.record.kind === "tool" && !isQuietTurnCall(row.record),
		isReceipt: (row: Row) => !quietRowIds.has(row.record.id) && isInternalReceipt(row.record),
	});
	const foldIds = latchFoldedRuns(
		options.latchedFoldIds ?? new Set(),
		{ has: (id: string) => byId.has(id) },
		folds,
	);
	const planOptions = {
		live: options.live,
		focusHold: options.focusHold ?? null,
		watchedRunId: options.watchedRunId ?? null,
		olderRowsExist: options.olderRowsExist === true,
	};
	const plan = collapsePlan(rows, { ...planOptions, openRuns });

	const rowFor = (row: Row, condensed = false): CondenseEntry => ({
		kind: "row",
		key: `row:${row.record.id}`,
		entry: byId.get(row.record.id) as TranscriptEntry,
		condensed,
	});
	const foldFor = (group: Extract<FoldGroup, { kind: "run" }>, condensed: boolean): CondenseEntry => ({
		kind: "fold",
		key: `fold:${group.id}`,
		group,
		children: group.rows.map((row) => rowFor(row, condensed)),
	});
	const output: CondenseEntry[] = [];
	let nextFold = 0;

	for (const [runIndex, run] of plan.runs.entries()) {
		const groups: FoldGroup[] = [];
		while (nextFold < folds.length) {
			const group = folds[nextFold];
			const firstRow = group.kind === "run" ? group.rows[0] : group.row;
			const index = rows.indexOf(firstRow);
			if (index < 0 || index > run.run.endIndex) break;
			groups.push(group);
			nextFold += 1;
		}
		if (!run.collapses) {
			for (const group of groups) {
				output.push(group.kind === "run" ? foldFor(group, false) : rowFor(group.row));
			}
			continue;
		}

		const segmentOf = new Map<string, number>();
		run.segments.forEach((segment, index) => {
			for (const id of segment.segmentIds) segmentOf.set(id, index);
		});
		/* A legacy quiet group owns its whole span. Exempt that span here,
		   not through openRuns: `collapsed` is reader-independent by design,
		   so an open key alone would still paint a T1 around the Q bar. */
		const holdNewestOpen = runIndex === plan.runs.length - 1 && options.live &&
			!(options.condensedRunIds?.has(run.key) ?? false);
		const renderRun = quietRowIds.size === 0 && !holdNewestOpen ? run : {
			...run,
			segments: run.segments.map((segment) =>
				holdNewestOpen || (segment.rows.length > 0 && segment.rows.every((row) => quietRowIds.has(row.record.id)))
					? { ...segment, collapsed: false }
					: segment),
		};
		const bars = new Map<number, Extract<CondenseEntry, { kind: "turn" }>>();
		const activeRun = renderRun;
		/* The prompt may shrink only when a visible bar can restore it. Live
		   and quiet-ownership overrides remove bars after the base plan, so
		   derive prompt presentation from that effective plan too. */
		const condensed = runIsCondensed(activeRun, openRuns);
		for (const group of groups) {
			const members = group.kind === "run" ? group.rows : [group.row];
			const first = segmentOf.get(members[0].record.id);
			const segmentIndex = members.every((row) => segmentOf.get(row.record.id) === first)
				? first
				: undefined;
			const segment = segmentIndex === undefined ? null : activeRun.segments[segmentIndex];
			if (segment && segmentIndex !== undefined && segment.collapsed) {
				let bar = bars.get(segmentIndex);
				if (!bar) {
					bar = {
						kind: "turn",
						key: `turn:${segment.key}`,
						run,
						segment,
						children: [],
					};
					bars.set(segmentIndex, bar);
					output.push(bar);
				}
				bar.children.push(group.kind === "run" ? foldFor(group, true) : rowFor(group.row, true));
				continue;
			}
			output.push(group.kind === "run"
				? foldFor(group, false)
				: rowFor(group.row, condensed && group.row.record.kind === "user"));
		}
	}

	const newest = plan.runs.at(-1);
	const watchedRunId = options.live && newest
		? rows[newest.run.openingIndex]?.record.id ?? null
		: options.watchedRunId ?? null;
	return { entries: output, rows, runs: plan.runs, foldIds, watchedRunId };
}
