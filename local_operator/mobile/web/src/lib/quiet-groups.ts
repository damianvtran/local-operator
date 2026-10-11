/**
 * The quiet-group definition (quiet-turn design §5 + §8 S4, rev 2), derived
 * for the relay-web record vocabulary.
 *
 * WHAT A GROUP IS. A quiet group is a client-derived fold over a run's
 * delivery receipts: a maximal run of >= 2 receipt rows with nothing the
 * reader can see between them. A `user` row, a compaction statement, any
 * other notice, a visible card or visible assistant text splits it; tool
 * rows (the quiet `no_reply` call included) sit inside it. ONE receipt is
 * not a group at all — its row keeps its ordinary card.
 *
 * There is no wire kind and no capability flag: the group is a PURE function
 * of the rows on hand, and each surface derives it for itself. The single
 * definition is kept from drifting across surfaces by the shared parity
 * fixture (`quiet-groups.parity.json`, copied byte-identical from
 * local-operator-ui PR #945's branch — the same cross-suite pattern
 * `format.parity.json` and `spend-context.parity.json` use), which this
 * module's test replays case by case.
 *
 * THE RELAY-WEB MAPPING, stated once (what this wire can and cannot express):
 *
 * - `peer_message` rows are the `peer` family; wake receipts (`notice` rows
 *   carrying `details.notice_kind === "wake"`) are the `wake` family. Both
 *   are real, production-firing triggers.
 * - The relay fold renders a monitor prompt and a delivered job result as a
 *   GENERIC notice — no field on this wire distinguishes them — so the
 *   `monitor` / `job` families cannot be derived here, and the fixture's two
 *   `custom` cases are not replayed on this surface (the test names them and
 *   pins the relay reality: they split, which is the safe direction — an
 *   unclassifiable receipt stays visible rather than hiding inside a group).
 * - No per-entry timestamp travels on this wire, so the records fed in on
 *   this surface carry no `ts`: `firstTs`/`lastTs` come back null and no span
 *   is stated (the same clauses a head-cut span nulls, design §5). The field
 *   stays in the shape because it is part of the shared cross-client
 *   contract the fixture pins — a surface whose records carry times (the
 *   desktop) states them from this same derivation.
 *
 * KEY STABILITY. `key` is `qg:<first row id>`: rows only append at the tail,
 * so the key never moves while the group grows, and a press on the bar
 * survives the append. A closed group's facts cannot move either — later
 * appends land outside its boundary — so no latch state is needed here: the
 * derivation is pure per render, which is the latch.
 */

import type { PeerSender, TranscriptEntry } from "../types";

/** The family word's set (design §5): the trigger kinds that can compose a
 * group, plus `mixed` for more than one. `monitor`/`job` are kept because
 * this is the shared contract's own union — see the module doc for why this
 * surface cannot reach them. */
export type QuietGroupFamily = "peer" | "wake" | "monitor" | "job" | "mixed";

/** ONE QUIET GROUP'S sender entry: the identity line and how many receipts
 * this sender contributed. */
export type QuietGroupSender = {
	/** The identity ladder's own spelling — quoted conversation name,
	 * `basename/`, a short session id, or `pid N`. */
	label: string;
	count: number;
};

/** ONE QUIET GROUP (design §5). Every field is derived from the rows on hand;
 * nothing here is ever sent. */
export type QuietGroup = {
	/** `qg:<first row id>` — stable across appends (see the module doc). */
	key: string;
	/** The family the trigger rows compose; `mixed` when more than one. */
	family: QuietGroupFamily;
	/** Trigger rows in the group (>= 2 by construction). */
	count: number;
	/** First trigger's instant; null when the span's head is cut OR the
	 * records carry no times (this wire's state — see the module doc). */
	firstTs: number | null;
	/** Last trigger's instant; null for the same reasons as `firstTs`. */
	lastTs: number | null;
	/** Peer-family senders, by count: the top 2, then one aggregate entry
	 * whose label is `<N> more` for the remaining distinct senders and whose
	 * count is their receipts. Empty for every other family (the shape is
	 * peer-only). */
	senders: QuietGroupSender[];
	/** Non-quiet tool rows in the group. */
	actions: number;
	/** Of `actions`, the rows whose outcome is a genuine error. */
	failed: number;
	/** It is the tail and no later visible row exists (the group may still grow). */
	open: boolean;
	/** Every row of the group, in order. */
	rowIds: string[];
};

/**
 * The row shape this derivation reads. `TranscriptEntry` satisfies it; `ts`
 * is the one field this wire does not carry (see the module doc), and it is
 * optional so a client whose records DO carry times maps them onto the same
 * derivation — the fixture's own contract ("Timestamps are ms epoch, copied
 * verbatim into `firstTs`/`lastTs`").
 */
export type QuietGroupRecord = Pick<
	TranscriptEntry,
	"kind" | "id" | "text" | "details" | "tool_name" | "tool_state"
> & {
	/** Epoch milliseconds; absent when the surface carries no per-entry time. */
	ts?: number;
};

/** A stretch of record indexes, `to` inclusive. */
export type SegmentSpan = { from: number; to: number };

/**
 * The family this row contributes when it is a trigger, or null when the row
 * starts nothing. On this wire only two kinds are triggers: an inbound peer
 * message and a wake receipt.
 */
function quietFamilyOf(record: QuietGroupRecord): QuietGroupFamily | null {
	if (record.kind === "peer_message") return "peer";
	if (record.kind === "notice" && record.details?.notice_kind === "wake") return "wake";
	return null;
}

/** A trigger row: the group's >= 2 unit (the definition's countable rows). */
function isQuietGroupTrigger(record: QuietGroupRecord): boolean {
	return quietFamilyOf(record) !== null;
}

/**
 * Does this row SPLIT a quiet group? The boundary vocabulary is this wire's
 * own visibility list (the collapse's pin list, `staysVisibleWhileCollapsed`,
 * on the desktop): every row a reader must be able to see is a boundary, so a
 * group can never hide one. A `user` row splits, a compaction statement
 * splits, visible assistant prose splits, and a NOTICE splits unless it is a
 * wake receipt — the relay folds stop receipts, gate timeouts, monitor
 * prompts and job results all into `notice` rows, and an unclassifiable
 * receipt must stay visible. Tool rows sit inside, and so does a reasoning
 * row (transient by construction; this surface's unknown-kind path paints
 * nothing for it).
 */
function groupSplitterOf(record: QuietGroupRecord): boolean {
	switch (record.kind) {
		case "tool":
		case "reasoning":
			return false;
		case "peer_message":
			return false;
		case "notice":
			return record.details?.notice_kind !== "wake";
		case "assistant":
			// Visible prose is the reader's answer and splits; an empty or
			// still-streaming row paints nothing and sits inside.
			return record.text.trim().length > 0;
		default:
			// `user`, `steer`, `compaction`, `parent_message`, `subagent_message`,
			// `ask_response`, `ask_timeout`: each paints something the reader
			// must be able to read, so each is a boundary.
			return true;
	}
}

/* The sender ladder's separators. Split on BOTH rather than one: the desktop
 * runs on Windows too, where a bare `/` split keeps the whole path as a name
 * (local-operator-ui `receipt-row-model.ts` makes the same call). */
const TRAILING_SEPARATORS = /[\\/]+$/;
const PATH_SEPARATOR = /[\\/]/;

/**
 * The identity a peer receipt contributes to a group's sender summary.
 *
 * The spelling is `peerIdentity`'s (local-operator-ui, `receipt-row-model`):
 * a name the peer CHOSE (the conversation name) is quoted, the ladder's
 * guesses (a cwd basename, marked with its trailing slash, then a short
 * session id) are not, and a senderless receipt lands on `another session` —
 * the same vocabulary `harness/comms.py` uses, kept identical so a reader
 * meeting one in the summary and one in a card does not think they are two
 * different states. Total over partial senders: every missing field falls to
 * the ladder's next rung rather than throwing.
 */
function quietSenderLabel(sender: PeerSender): string {
	const name = sender.conversation_name ?? "";
	if (name) return `"${name}"`;
	const cwd = (sender.cwd ?? "").replace(TRAILING_SEPARATORS, "");
	if (cwd) {
		const base = cwd.split(PATH_SEPARATOR).pop() ?? "";
		/* A trailing slash says "this is a directory", which is the only
		 * thing that distinguishes a guessed name from a chosen one when
		 * both are unquoted. */
		if (base) return `${base}/`;
	}
	if (sender.session_id) return sender.session_id.slice(0, 8);
	return sender.pid ? `pid ${sender.pid}` : "another session";
}

/**
 * The quiet-turn tool's name (design §4). Core owns the literal
 * (`harness/rows.py` `QUIET_TURN_TOOL`) and the relay fold already hides the
 * pair (S1), so a matching row should not arrive on this wire at all — the
 * exclusion is kept because the shared fixture replays the pairs, and because
 * a mixed-build runtime that predates the fold would otherwise have its quiet
 * call counted as an action.
 */
const QUIET_TURN_TOOL = "no_reply";

function isQuietTurnCall(record: QuietGroupRecord): boolean {
	return record.kind === "tool" && record.tool_name === QUIET_TURN_TOOL;
}

/**
 * Is this tool row a genuine FAILURE? Only `tool_state === "failed"`.
 *
 * The relay fold settles {skipped, aborted} faults into their own
 * `interrupted` state (the same exclusion the desktop's `isFailedCall` makes
 * from the wire's `not_run_kind`), so an interrupted call is not a failure
 * and a call the user stopped is not counted. `error` is a message, not a
 * verdict — the fold sets it exactly when the state is `failed`.
 */
function isFailedCall(record: QuietGroupRecord): boolean {
	return record.kind === "tool" && record.tool_state === "failed";
}

/**
 * The quiet group a single span IS, or null when the span is not one.
 *
 * THE SPAN MUST BE THE WHOLE GROUP (design §5): its neighbours must be
 * splitters or the list's edges (`quietGroupsOf` builds spans that way), and
 * no splitter may sit inside. A span that merely OVERLAPS a group — a caller
 * that sliced inside one — refuses here instead of stating a count over part
 * of something: the caller degrades to the rows' ordinary cards, which is the
 * safe direction. The shared fixture pins the refusal (its sub-span case).
 *
 * `spanHeadLoaded` (default true) is the caller's own head-cut verdict for
 * this span: false nulls the times and leaves the count a minimum — the
 * "at least N" rule, which the caller states in its own vocabulary. `open`
 * (default: the span reaches the list's end) is the tail fact the
 * growing-group rule reads.
 */
export function quietGroupOfSegment(
	records: readonly QuietGroupRecord[],
	span: SegmentSpan,
	options: { spanHeadLoaded?: boolean; open?: boolean } = {},
): QuietGroup | null {
	const headLoaded = options.spanHeadLoaded ?? true;
	for (let i = span.from; i <= span.to; i += 1) {
		if (groupSplitterOf(records[i])) return null;
	}
	if (span.from > 0 && !groupSplitterOf(records[span.from - 1])) return null;
	if (span.to + 1 < records.length && !groupSplitterOf(records[span.to + 1])) {
		return null;
	}
	const triggers: QuietGroupRecord[] = [];
	for (let i = span.from; i <= span.to; i += 1) {
		if (isQuietGroupTrigger(records[i])) triggers.push(records[i]);
	}
	if (triggers.length < 2) return null;
	const first = triggers[0];
	const last = triggers[triggers.length - 1];
	let actions = 0;
	let failed = 0;
	/* Peer senders keep their first-appearance order for the tie-break below. */
	const senderCounts = new Map<string, number>();
	for (let i = span.from; i <= span.to; i += 1) {
		const record = records[i];
		if (record.kind === "tool" && !isQuietTurnCall(record)) {
			actions += 1;
			if (isFailedCall(record)) failed += 1;
			continue;
		}
		if (record.kind === "peer_message") {
			const label = quietSenderLabel(record.details?.sender ?? {});
			senderCounts.set(label, (senderCounts.get(label) ?? 0) + 1);
		}
	}
	let family: QuietGroupFamily | null = null;
	let mixed = false;
	for (const trigger of triggers) {
		const triggerFamily = quietFamilyOf(trigger);
		if (family === null) family = triggerFamily;
		else if (family !== triggerFamily) mixed = true;
	}
	const familyFinal: QuietGroupFamily = mixed ? "mixed" : (family ?? "mixed");
	const senders: QuietGroupSender[] = [];
	if (familyFinal === "peer") {
		const entries = [...senderCounts.entries()].map(([label, count]) => ({
			label,
			count,
		}));
		const order = new Map(entries.map((entry, index) => [entry.label, index]));
		entries.sort(
			(a, b) =>
				b.count - a.count ||
				(order.get(a.label) ?? 0) - (order.get(b.label) ?? 0),
		);
		const top = entries.slice(0, 2);
		if (entries.length > 2) {
			let rest = 0;
			for (const entry of entries.slice(2)) rest += entry.count;
			top.push({ label: `${entries.length - 2} more`, count: rest });
		}
		senders.push(...top);
	}
	const rowIds: string[] = [];
	for (let i = span.from; i <= span.to; i += 1) rowIds.push(records[i].id);
	return {
		key: `qg:${records[span.from].id}`,
		family: familyFinal,
		count: triggers.length,
		firstTs: headLoaded ? (first.ts ?? null) : null,
		lastTs: headLoaded ? (last.ts ?? null) : null,
		senders,
		actions,
		failed,
		open: options.open ?? (span.to === records.length - 1),
		rowIds,
	};
}

/**
 * Every quiet group in a records list, in order: the stretches between
 * splitters, each scored by `quietGroupOfSegment` (so a stretch with fewer
 * than two triggers yields none). The list's own edges are legitimate bounds
 * — that is how a head-cut leading group forms — and the caller states the
 * minimum with its own vocabulary (see `quietGroupOfSegment`).
 */
export function quietGroupsOf(records: readonly QuietGroupRecord[]): QuietGroup[] {
	const groups: QuietGroup[] = [];
	let start = 0;
	for (let i = 0; i <= records.length; i += 1) {
		if (i < records.length && !groupSplitterOf(records[i])) continue;
		if (i > start) {
			const group = quietGroupOfSegment(records, { from: start, to: i - 1 });
			if (group !== null) groups.push(group);
		}
		start = i + 1;
	}
	return groups;
}

/** The family's word for a group bar: family plural, `Messages` for a mix
 * (design §5's copy table; sole author of these words). */
export function quietGroupLabel(family: QuietGroupFamily): string {
	switch (family) {
		case "peer":
			return "Peer messages";
		case "wake":
			return "Wake messages";
		case "monitor":
			return "Monitor messages";
		case "job":
			return "Job results";
		default:
			return "Messages";
	}
}
