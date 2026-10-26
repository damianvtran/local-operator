/** One-line chrome for condensed turns and tool folds (design PR C §2). */
import type { ReactNode } from "react";
import { cn } from "../lib/cn";
import { messageRuntime, useLocaleSnapshot } from "../store";
import { formatElapsed } from "../lib/format";
import { actionCount, countLabel, foldUnits, receiptUnits, segmentLabel } from "../lib/condense-copy";
import type { CondenseEntry } from "../lib/condense-view";
import { Chevron } from "./ui/disclosure";

function Units({ units, className }: { units: string[]; className?: string }) {
	return (
		<span className={cn("flex min-w-0 flex-1 flex-wrap items-center gap-x-1", className)}>
			{units.map((unit, index) => (
				<span key={`${index}:${unit}`} className="contents">
					{index > 0 ? <span aria-hidden className="shrink-0">·</span> : null}
					<span className="min-w-0 whitespace-nowrap">{unit}</span>
				</span>
			))}
		</span>
	);
}

function Header({
	id,
	open,
	onToggle,
	failed,
	failedCount = 0,
	live,
	children,
	label,
}: {
	id: string;
	open: boolean;
	onToggle: () => void;
	failed: boolean;
	failedCount?: number;
	live?: boolean;
	children: ReactNode;
	label?: string;
}) {
	const failureLabel = failed ? messageRuntime.t("relay.transcript.fold.failed_label") : "";
	return (
		<button
			type="button"
			data-condense-header={id}
			aria-expanded={open}
			aria-controls={`${id}:body`}
			aria-label={label}
			onClick={onToggle}
			className="flex h-11 min-h-11 w-full items-center gap-1.5 overflow-hidden rounded-sm px-1.5 text-start select-none active:bg-surface"
		>
			<Chevron open={open} className="shrink-0" />
			<span
				aria-hidden
				className={cn("w-4 shrink-0 text-center font-mono text-mono-sm", failed ? "text-danger" : live ? "lo-pulse text-accent" : "text-success")}
			>
				{failed ? "✗" : live ? "⟳" : ""}
			</span>
			{children}
			{failedCount > 0 ? <span data-failed-count={failedCount} className="shrink-0 font-mono text-mono-sm text-danger">{failedCount}</span> : null}
			{failureLabel ? <span className="sr-only">{failureLabel}</span> : null}
		</button>
	);
}

export function TurnBar({
	entry,
	open,
	onToggle,
	children,
}: {
	entry: Extract<CondenseEntry, { kind: "turn" }>;
	open: boolean;
	onToggle: () => void;
	children: ReactNode;
}) {
	const { segment } = entry;
	const group = segment.facts.group;
	const { uiLocale } = useLocaleSnapshot();
	const formatCount = (count: number, partial = false) => countLabel(count, partial, uiLocale);
	const units: string[] = [];
	const label = segmentLabel(segment.labelKind);
	if (label) units.push(label);
	if (group) {
		units.push(String(group.count));
		if (group.actions > 0) units.push(actionCount(group.actions, segment.facts.partial, formatCount(group.actions, segment.facts.partial)));
	} else {
		if (segment.facts.durationS !== null) {
			units.push(messageRuntime.t("relay.transcript.turn.took", { duration: formatElapsed(segment.facts.durationS) }));
		}
		if (segment.facts.actions > 0) units.push(actionCount(segment.facts.actions, segment.facts.partial, formatCount(segment.facts.actions, segment.facts.partial)));
		if (units.length === 0) units.push(...foldUnits(segment.facts.countUnits));
		if (units.length === 0) units.push(messageRuntime.t("relay.transcript.turn.condensed"));
	}
	const partialTitle = segment.facts.partial
		? messageRuntime.t("relay.transcript.fold.partial_label", { count: segment.facts.actions })
		: undefined;
	return (
		<div data-condensed-turn={entry.run.key} data-condense-segment={segment.key}>
			<Header id={entry.key} open={open} onToggle={onToggle} failed={segment.facts.failed > 0} failedCount={segment.facts.failed} label={[...units, partialTitle].filter(Boolean).join(" · ")}>
				<span title={partialTitle} className="contents">
					<Units units={units} className="text-meta text-ink-dim" />
				</span>
				{segment.completed ? <span className="sr-only">{messageRuntime.t("relay.transcript.turn.completed")}</span> : null}
			</Header>
			{open ? <div id={`${entry.key}:body`} className="flex flex-col gap-2">{children}</div> : null}
		</div>
	);
}

export function FoldBar({
	entry,
	open,
	onToggle,
	children,
}: {
	entry: Extract<CondenseEntry, { kind: "fold" }>;
	open: boolean;
	onToggle: () => void;
	children: ReactNode;
}) {
	const { group } = entry;
	const { uiLocale } = useLocaleSnapshot();
	const totalActions = actionCount(group.actions.length, false, countLabel(group.actions.length, false, uiLocale));
	const liveRow = group.live === null ? null : group.rows.find((row) => row.record.kind === "tool" && row.record.phase === "running");
	const units = [...foldUnits(group.summary.countUnits), ...receiptUnits(group.rows.filter((row) => row.record.kind !== "tool"))];
	if (liveRow && !open) {
		return (
			<div data-condense-fold={group.id}>
				<Header id={entry.key} open={open} onToggle={onToggle} failed={group.failedCount > 0} failedCount={group.failedCount} live>
					<span className="flex min-w-0 flex-1 flex-col justify-center">
						<span className="flex min-w-0 items-center gap-1.5 text-body-sm">
							<span className="shrink truncate text-ink-muted">{liveRow.record.kind === "tool" ? liveRow.record.toolName : ""}</span>
							<span className="min-w-0 flex-1 truncate font-mono text-ink-dim">{liveRow.record.kind === "tool" ? liveRow.record.summary : ""}</span>
						</span>
						<span className="truncate text-meta text-ink-dim">{totalActions}</span>
					</span>
				</Header>
			</div>
		);
	}
	return (
		<div data-condense-fold={group.id}>
			<Header id={entry.key} open={open} onToggle={onToggle} failed={group.failedCount > 0} failedCount={group.failedCount}>
				<Units units={units.length ? units : [actionCount(group.actions.length)]} className="text-body-sm text-ink-muted" />
			</Header>
			{open ? <div id={`${entry.key}:body`} className="flex flex-col ps-[22px]">{children}</div> : null}
		</div>
	);
}
