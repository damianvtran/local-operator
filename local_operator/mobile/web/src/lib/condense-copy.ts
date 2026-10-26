/** Localized words for structured condensation facts; no decision logic here. */
import { messageRuntime } from "../store";
import type { FoldCountSegment } from "./condense-fold";
import type { SegmentLabelKind } from "./condense-segments";
import { quietGroupLabel } from "./quiet-groups";

export function countLabel(count: number, partial: boolean, locale: string): string {
	return `${new Intl.NumberFormat(locale).format(count)}${partial ? "+" : ""}`;
}

export function actionCount(count: number, partial = false, label?: string): string {
	return messageRuntime.t("relay.transcript.turn.actions", {
		count,
		label: label ?? countLabel(count, partial, "en"),
	});
}

const CLASS_KIND: Record<string, string> = {
	"class:files": "files",
	"class:searches": "searches",
	"class:web": "web",
	"class:edits": "edits",
	"class:delegated": "tasks",
	"kind:shell": "shell",
	"kind:python": "python",
};

export function foldUnit(segment: FoldCountSegment): string {
	if (segment.key === "kind:shell") return messageRuntime.t("relay.transcript.fold.shell", { count: segment.count });
	if (segment.key === "kind:python") return messageRuntime.t("relay.transcript.fold.python", { count: segment.count });
	/* Known catalogue classes carry count-free structured nouns. Other model
	   units (waits, job checks, named kinds) already include their count; do
	   not feed that whole phrase into a second counted catalogue template.
	   Technical model labels keep their existing wording until localized as
	   their own structured kinds, rather than guessing nouns by parsing prose. */
	if (!(segment.key in CLASS_KIND) && !segment.key.startsWith("name:")) return segment.label;
	const kind = CLASS_KIND[segment.key] ?? "other";
	return messageRuntime.t("relay.transcript.fold.unit", {
		kind,
		count: segment.count,
		name: segment.key.startsWith("name:") ? segment.key.slice(5) : segment.label,
	});
}

/** Apply the desktop's five-unit display cap without comparing its English. */
export function foldUnits(segments: readonly FoldCountSegment[]): string[] {
	if (segments.length <= 5) return segments.map(foldUnit);
	const kept = segments.slice(0, 5).map(foldUnit);
	const other = segments.slice(5).reduce((sum, segment) => sum + segment.count, 0);
	return [...kept, messageRuntime.t("relay.transcript.fold.more", { count: other })];
}

export function receiptUnits(rows: readonly { record: { kind: string; customType?: string; receipt?: { expectsReply?: boolean; steer?: boolean } } }[]): string[] {
	let incoming = 0;
	let wakes = 0;
	let other = 0;
	for (const row of rows) {
		const record = row.record;
		if (record.kind === "peer" || (record.kind === "custom" && record.customType === "hub_message")) incoming += 1;
		else if (record.kind === "wake") wakes += 1;
		else other += 1;
	}
	const units: string[] = [];
	if (incoming) units.push(messageRuntime.t("relay.transcript.fold.unit", { kind: "incoming", count: incoming, name: "" }));
	if (wakes) units.push(messageRuntime.t("relay.transcript.fold.unit", { kind: "wakes", count: wakes, name: "" }));
	if (other) units.push(messageRuntime.t("relay.transcript.fold.unit", { kind: "other", count: other, name: "receipts" }));
	return units;
}

export function segmentLabel(kind: SegmentLabelKind | null): string | null {
	switch (kind?.kind) {
		case "steered": return messageRuntime.t("relay.transcript.label.steered");
		case "wake": return messageRuntime.t("relay.transcript.label.wake");
		case "peer": return messageRuntime.t("relay.transcript.label.peer");
		case "job": return messageRuntime.t("relay.transcript.label.job");
		case "ask": return messageRuntime.t("relay.transcript.label.ask");
		case "group": return quietGroupLabel(kind.family);
		default: return null;
	}
}
