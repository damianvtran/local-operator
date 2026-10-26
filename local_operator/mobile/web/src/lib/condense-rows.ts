/**
 * THE ROW LIST A PLAN IS COMPUTED OVER, ported from the desktop.
 * Ported from `damianvtran/local-operator-ui` at sourceHash f9f4611dd3dbcc06…:src/renderer/src/features/chat/canonical/transcript-rows.ts:37-753
 *
 * WHAT IT DECIDES, and why the phone needs it rather than a flat render: which
 * records PAINT at all (`paintsSomething`), which of them are statement rows or
 * trace rows, what the gap tier between two rows is, where a run of turns begins
 * and ends (`walkTurns`/`runsOf`), and which rows close a turn
 * (`closingAnswerIds`). The desktop's answers, unchanged - the fixture replay
 * compares them case by case.
 *
 * HOW IT DIFFERS: `TranscriptRecord` is `CondenseRecord`; `displayName` comes
 * from `condense-predicates.ts` (lifted verbatim). Nothing else.
 */

import type { CondenseRecord } from "./condense-records";
import {
	cyclesOf,
	electAnswer,
	isQuietTurnCall,
	isQuietTurnClose,
	isTerminalMarker,
} from "./condense-segments";
import { displayName } from "./condense-predicates";
export const splitFirstLine = (
	text: string,
): { headline: string; rest: string | null } => {
	const lines = text.split("\n");
	const at = lines.findIndex((line) => line.trim().length > 0);
	if (at < 0) return { headline: text.trim(), rest: null };
	const rest = lines
		.slice(at + 1)
		.join("\n")
		.trim();
	return { headline: lines[at].trim(), rest: rest || null };
};

export type Row = {
	record: CondenseRecord;
	/** Vertical tier before this row. */
	gap: "turn" | "item" | "trace" | "first";
	/**
	 * Is this row the answer its turn was working towards? See
	 * `closingAnswerIds` for what qualifies and why the caption is gated on it
	 * rather than on `!record.streaming` alone.
	 */
	closesTurn: boolean;
};

/**
 * The one row per turn that carries the answer caption.
 *
 * The caption used to be gated on `!record.streaming` alone, which is a fact
 * about a RECORD and not about a turn — so a turn that narrates between calls
 * ("Checking the ledger first.", a call, "Four were late.", a call, "Writing
 * the summary.") painted one clock per paragraph, four identical stamps in a
 * single turn, the worst shape being two clocks 56px apart with one sentence
 * between them (round 1's D1/Q-2, measured on the frames). The operator's
 * wording is narrower than that and is the rule here: "just the final
 * responses, not the in-progress tool intent/response".
 *
 * So the caption belongs to the row the reader is being HANDED, and there are
 * two conditions for it, both about the turn rather than the record:
 *
 * 1. the row is the turn's LAST row that paints anything the turn itself did — a
 *    turn that ends on a ledger row has not handed the reader its answer yet, and
 *    a turn ending on prose that is still streaming has not either, so neither is
 *    captioned. A STATEMENT row at the end is not the turn's work at all, so the
 *    answer before it still closes the turn and keeps its caption
 *    (`isStatementRow`, design round 2's D2-1);
 * 2. that row is a settled assistant record with text in it, which is
 *    `paintsSomething` plus the liveness bit — an unfinished answer closes a turn
 *    without being an answer.
 *
 * A turn is the records between two user records. The user turn itself is never
 * a candidate: its bubble carries the caption on the other rail.
 *
 * It lives here rather than in the view for the reason this module exists — it
 * is a rule with a right answer, asserted directly
 * (`scripts/turn-timestamp.test.mjs`) instead of eyeballed in a frame.
 */
export function closingAnswerIds(
	records: CondenseRecord[],
): ReadonlySet<string> {
	const closing = new Set<string>();
	for (const span of walkTurns(records, (record) => record)) {
		if (span.closingAnswerId !== null) closing.add(span.closingAnswerId);
	}
	return closing;
}

/**
 * One turn's span over an item list, in THAT list's own indices.
 *
 * The unit `closingAnswerIds` has always used, expressed once so the caption
 * rule, the foot line and the turn collapse cannot disagree about what a turn
 * is. A turn opens at a user row and runs until the next user row, EXCEPT that
 * a user row which arrives while the turn is still open is a STEER and belongs
 * to the open turn (see `runsOf` for the rule and its reason).
 */
type TurnSpan = {
	openingIndex: number;
	endIndex: number;
	/** The opening user item's index, or null for a run whose head is cut off. */
	openingUserIndex: number | null;
	/**
	 * Why the run ended, for diagnostics and tests: `answer`/`quiet`/`marker`
	 * name the closure the NEXT user row saw (`quiet` is a settled `no_reply`
	 * close, which ends the turn without handing anything over), `end` is the
	 * list running out.
	 */
	boundary: "answer" | "quiet" | "marker" | "end";
	/** The settled assistant that closes this run, when its tail is one. */
	closingAnswerId: string | null;
	/**
	 * The run's LAST settled quiet close (`no_reply`), when it has one: the
	 * closure anchor that is not an answer - the collapsible gate carries it
	 * (`turn-collapse-model.ts`) while `closingAnswerId` stays null, so no
	 * caption, stamp or foot claims a message the reader was never handed.
	 */
	quietCloseId: string | null;
};

/**
 * The partition itself, shared by `closingAnswerIds` and `runsOf`.
 *
 * A user item opens a new run iff the open run is CLOSED, and the closure test
 * has THREE arms, all stated from the record list alone: the last
 * paint-non-statement record is a settled assistant (the run handed its answer
 * over); a TERMINAL marker has been seen since the run opened (`isTerminalMarker`
 * — the shared boundary vocabulary, NOT a local copy of it); or nothing has run
 * yet. Otherwise the user item is a steer and stays inside the run — steering is
 * NOT marked on the durable wire (the harness persists a steer as an ordinary user
 * message and the live `SteeringDeliveredEvent` is consumed by the stream, not
 * the transcript), so the partition can only be structural, and these facts are
 * the ones the record list actually states.
 *
 * WHY THE BOUNDARY VOCABULARY AND NOT A LOCAL COPY (the operator's own report,
 * 2026-09-30). This test used to read `isCompletionMarker` — `notice &&
 * complete === true` — while the pin list, the classifier and the rows' own paint
 * all read `boundaryKindOf`. The two agreed by copy, and they diverged at the
 * error-level `custom` a `session_incident` is: a terminal marker the rest of the
 * transcript honours, and one this test ignored. So `[session_incident][user]`
 * left `closed()` false, the reader's own FIRST message was absorbed as a steer,
 * and the bar over it was labelled "Steered" while the message sat inside the
 * hidden span. That is the last surviving copy of the four-way list the segments
 * module says it ended, and this is that copy fixed.
 *
 * WHY "NOTHING HAS RUN YET", which the vocabulary arm alone cannot cover. A
 * conversation OPENS with harness prefix rows — `system_prefix`, `selected_model`,
 * `session_mcp_unavailable` — that paint but are not `user` rows and are NOT
 * boundaries (an info-level `custom` is not terminal). The first of them opened
 * the run with `openingUserIndex: null`, so EVERY fresh conversation's first user
 * message steered into a head-cut run that held nothing. A steer is a message to
 * an agent that is WORKING; a run that opens off a non-user row and holds no work
 * of its own — no tool row, no assistant row — is not a turn in flight, it is an
 * empty preamble, so a following user message OPENS its own run. The test needs
 * both halves: it fires only while the run opens off a non-user row AND nothing
 * in it has painted work, so a `[user][user]` pair keeps today's steer semantics
 * (a turn the reader opened is in flight even before its first call).
 *
 * `recordOf` lets the same walk serve records and `Row`s: `Row`s only wrap a
 * subset of records, and each caller needs spans in its own index space.
 */
function walkTurns<T>(
	items: T[],
	recordOf: (item: T) => CondenseRecord,
): TurnSpan[] {
	const spans: TurnSpan[] = [];
	let open: TurnSpan | null = null;
	/** The last record in the open run that paints and is not a statement. */
	let last: CondenseRecord | null = null;
	/** A terminal marker (the shared boundary vocabulary) since the run opened. */
	let sawTerminal = false;
	/** A settled quiet close (see `isQuietTurnClose`) since the run opened. */
	let sawQuietClose = false;
	/** Has the open run done any WORK of its own (a tool or assistant row)? */
	let sawWork = false;

	const closed = (): boolean =>
		sawTerminal || sawQuietClose || settledTail() || nothingHasRunYet();
	/**
	 * Whether the open run's tail is a settled answer - the closure test, now
	 * reading the handover fact as well as the row's kind.
	 *
	 * A LEAD-IN IS NOT A HANDOVER. An assistant row that CARRIES TOOL CALLS is
	 * narration ("Reading the ledger first.") followed by the call it announces:
	 * its calls are still owed, so a user row arriving before those calls report
	 * is a STEER into this turn, not the opener of the next one. Closing here
	 * flushed the run on the lead-in, which made the steer a new turn, withdrew
	 * liveness from every row before it and condensed them behind a bar - and the
	 * bar went away again a pass later when the call's tool row landed and the
	 * tail stopped reading as an answer. That is the operator's reported steer
	 * alternation (2026-10-21: "condensing the messages before it and expanding
	 * them").
	 *
	 * The rule is the transcript's own handover fact, not a second one:
	 * `turn-segments.ts::isFinalResponse` states it from the same field
	 * (`callsTools`, set from the journal payload and from `message_end`),
	 * together with the paint test (`text.length > 0`). This closure deliberately
	 * reads the SAME TWO FACTS MINUS THE PAINT TEST - a settled assistant that
	 * says nothing is still a close, because the run is over whether or not it
	 * had prose to hand over (agent review round 1, F3). It is `!== true` rather
	 * than `=== false` so a record written before the field existed keeps the old
	 * behaviour.
	 *
	 * THE RESIDUE (agent review round 1, F2). An assistant row that DECLARES calls
	 * which never land - a turn that died without its tool rows and without a
	 * terminal marker - makes the next user message a steer into that dead run
	 * rather than the next turn's opener, because from the rows alone it is
	 * indistinguishable from the steer this rule exists for. A marker settles the
	 * question (any terminal marker still closes the run), so the merge only bites
	 * when the death is invisible; the cost is a bar over two turns' work, never
	 * lost data.
	 */
	const settledTail = (): boolean =>
		last !== null &&
		last.kind === "assistant" &&
		!last.streaming &&
		last.callsTools !== true;
	/**
	 * An EMPTY PREAMBLE: a run that opens off a non-user row (head-cut) and has
	 * done no work of its own. Nothing can be steered into it - see the module
	 * comment - so a user row after one OPENS rather than steers.
	 *
	 * TWO HALVES, both load-bearing. `openingUserIndex === null` is what keeps
	 * `[user][user]` a steer: a turn the reader opened is in flight even before
	 * its first call. `sawWork` counts a `tool` row OR an `assistant` row - the
	 * assistant arm includes a streaming and an empty one, because typing while
	 * the agent is writing its first paragraph, or while its first call is being
	 * set up, is the classic steer and nothing has been CALLED yet either way.
	 */
	const nothingHasRunYet = (): boolean =>
		open !== null && open.openingUserIndex === null && !sawWork;
	/*
	 * THE RUN'S ANSWER IS ELECTED, NOT "THE LAST SETTLED ASSISTANT" (issue #665).
	 *
	 * The old rule handed the turn's closing to whatever assistant row came last,
	 * so a short reply to a peer note written after the session was disposed took
	 * the answer's place and the bar swallowed the real one. `electAnswer` (the
	 * turn-segments module) picks the last RESPONSE cycle's close instead, and is
	 * null under the same gate the old rule had - a run that ends on a tool row or
	 * a streaming answer has not handed anything over.
	 *
	 * The RUN BOUNDARY (`closed()`, the user-row test) now reads the SAME terminal
	 * vocabulary this election does (`isTerminalMarker`), plus the empty-preamble
	 * clause — see the module comment on `walkTurns`. The reasoning above is about
	 * WHICH row of a run is its answer, not about which rows END a run; only the
	 * latter changed here, and the gate stays exactly as it was.
	 */
	const closing = (
		from: number,
		to: number,
	): { answerId: string | null; quietCloseId: string | null } => {
		const records: CondenseRecord[] = [];
		for (let i = from; i <= to; i += 1) records.push(recordOf(items[i]));
		const cycles = cyclesOf(records, paintsSomething);
		const answer = electAnswer(records, cycles, {
			paints: paintsSomething,
			isStatement: isStatementRow,
		});
		/*
		 * The quiet close is read off the SAME cycle chain the election reads: a
		 * cycle whose close is a settled quiet call. Last one wins - the run's
		 * LATEST quiet close is the closure a later span's bar states.
		 */
		let quietCloseId: string | null = null;
		for (const cycle of cycles) {
			const close = records[cycle.closeIndex];
			if (isQuietTurnClose(close)) quietCloseId = close.id;
		}
		return {
			answerId: answer === null ? null : records[answer.closeIndex].id,
			quietCloseId,
		};
	};
	const flush = (boundary: TurnSpan["boundary"], endIndex: number) => {
		if (open === null) return;
		open.endIndex = endIndex;
		open.boundary = boundary;
		const closingIds = closing(open.openingIndex, endIndex);
		open.closingAnswerId = closingIds.answerId;
		open.quietCloseId = closingIds.quietCloseId;
		spans.push(open);
		open = null;
	};
	/*
	 * BOTH openings go through here, so the three facts the closure test reads are
	 * reset identically however a run starts (review round 1, R8). The head-cut
	 * branch used to reset none of them while the user-row branch reset all three:
	 * equivalent TODAY, because `open` is nulled only inside `flush` and `flush`
	 * runs at a user row immediately before `open` is reassigned, so the head-cut
	 * branch can only fire at index 0 where all three are already at their initial
	 * values - and a silent trap the day that reachability moves.
	 */
	const openRun = (
		index: number,
		openingUserIndex: number | null,
	): TurnSpan => {
		last = null;
		sawTerminal = false;
		sawQuietClose = false;
		sawWork = false;
		return {
			openingIndex: index,
			endIndex: index,
			openingUserIndex,
			boundary: "end",
			closingAnswerId: null,
			quietCloseId: null,
		};
	};

	items.forEach((item, index) => {
		const record = recordOf(item);
		if (record.kind === "user") {
			if (open === null || closed()) {
				/*
				 * The closure the NEXT user item saw is this run's boundary: a settled
				 * answer first, because that is the row the caption and the collapse
				 * hand the reader; a completion marker only when there is no answer.
				 */
				if (open !== null) {
					flush(
						settledTail() ? "answer" : sawQuietClose ? "quiet" : "marker",
						index - 1,
					);
				}
				open = openRun(index, index);
			}
			/*
			 * Either way the user item itself is never the run's "last content": a
			 * message the reader typed is not work the turn did, which is also why
			 * `closingAnswerIds` excluded user records before this partition existed.
			 */
			return;
		}
		if (open === null) {
			/*
			 * A run whose head is CUT OFF: the visible slice (or the first fetched
			 * page, in `closingAnswerIds`' record-space) starts mid-turn, so there is
			 * no opening user row to key or anchor it. `openingUserIndex: null` is
			 * how `runsOf` reports it and how the collapse refuses to build a bar
			 * over a turn it cannot show the whole of.
			 */
			open = openRun(index, null);
		}
		if (paintsSomething(record) && !isStatementRow(record)) last = record;
		if (isTerminalMarker(record)) sawTerminal = true;
		if (isQuietTurnClose(record)) sawQuietClose = true;
		if (record.kind === "tool" || record.kind === "assistant") sawWork = true;
		open.endIndex = index;
	});
	flush("end", items.length - 1);
	return spans;
}

/**
 * One run of the visible transcript: the unit the turn collapse plans over.
 *
 * Distinct from a fold run (`foldRuns`), which is a stretch of consecutive
 * tool rows INSIDE a turn: this is the turn itself, and a bar over it is the
 * collapsed view of the rows between its opening user message and the answer
 * it worked towards.
 */
export type TurnRun = {
	/**
	 * Stable identity ACROSS THE HEAD ARRIVING LATER: the closing answer's
	 * record id when the run has one (the row the bar and the caption already
	 * speak through), else the run's LAST row's id.
	 *
	 * WHY NOT THE OPENING USER ROW (the pre-fix identity): a run whose head the
	 * fetched rows cut off keys off whatever it does have — that used to be its
	 * first row — so the day a page landed the head, the key changed and every
	 * consumer keyed by it (the reader's expansion on the bar, the bar's own
	 * React key) silently started over. Rows only ever arrive ABOVE a run's
	 * head, so its tail is the stable half and the two candidates above are the
	 * same row in both the head-cut and the head-loaded list.
	 */
	key: string;
	/**
	 * Whether the run's opening user row is in this list. False means the
	 * window's leading edge or the fetched set has walked past it: the span is
	 * PARTIAL. The collapse decides what that costs — a run with its closing
	 * answer on hand still condenses from the loaded span (end-loaded
	 * eligibility, `turn-collapse-model.ts`), stating no duration it cannot
	 * honestly compute.
	 */
	opensWithUserRow: boolean;
	/** Index of the run's first row in the list handed to `runsOf`. */
	openingIndex: number;
	/** Index of the run's last row in that list. */
	endIndex: number;
	boundary: TurnSpan["boundary"];
	/** The settled assistant row that closes this run, when it has one. */
	closingAnswerId: string | null;
	/**
	 * The run's last settled quiet close (`no_reply`), when it has one: the
	 * closure anchor that is NOT an answer. The collapsible gate reads it
	 * (having its closing anchor in the run is what lets a quiet run condense),
	 * while `closingAnswerId` stays null so no caption or stamp claims a
	 * message the reader was never handed.
	 */
	quietCloseId: string | null;
};

/**
 * The rows partitioned into turns, in order, in the ROW index space.
 *
 * This is `walkTurns` over the row list: the same partition `closingAnswerIds`
 * uses, with the indices the render pipeline needs (the collapse model's spans,
 * the foot line's reset points, the `key` a bar is remembered by).
 */
export function runsOf(rows: Row[]): TurnRun[] {
	return walkTurns(rows, (row) => row.record).map((span) => ({
		key: span.closingAnswerId ?? rows[span.endIndex]?.record.id ?? "",
		opensWithUserRow: span.openingUserIndex !== null,
		openingIndex: span.openingIndex,
		endIndex: span.endIndex,
		boundary: span.boundary,
		closingAnswerId: span.closingAnswerId,
		quietCloseId: span.quietCloseId,
	}));
}

/**
 * Is this row a STATEMENT rather than a row of the turn's own work?
 *
 * These are the rows that report something ABOUT the conversation — a session
 * incident, a model switch, a peer message's receipt, a wake delivery — and the
 * closing answer is found past them rather than through them (design round 2,
 * D2-1). The distinction matters because of the three ways a turn can end:
 *
 * - on a LEDGER row the agent is still working, and stripping the caption is
 *   right: the answer it will end on has not been written yet;
 * - on a STREAMING answer the answer has not settled, and stripping it is right
 *   for the same reason;
 * - on a STATEMENT the answer HAS been handed over, and the statement is not a
 *   reason to take its time away — worse, the time becomes unrecoverable, because
 *   a notice and a receipt paint no `<time>` of their own and have no disclosure
 *   to open. Measured on the frames: `[user][answer][notice]` painted nothing at
 *   all on screen.
 *
 * `tool` is deliberately NOT in this set, for the first reason above. `compaction` IS: a compaction
 * receipt renders exactly like a notice - receipt line, no `<time>`, no disclosure to open - so
 * `[user][answer][compaction]` had the same unrecoverable loss (design round 3, D3-1, measured by the
 * designer: 0 captions before, 1 after, at left 102, with the four controls unchanged).
 */
export function isStatementRow(record: CondenseRecord): boolean {
	return (
		record.kind === "notice" ||
		record.kind === "compaction" ||
		record.kind === "custom" ||
		record.kind === "peer" ||
		record.kind === "wake"
	);
	/*
	 * The desktop also names `supplement` here (turn-supplements memo §2.11). This
	 * wire has no such kind: all five kinds the desktop treats as statements are
	 * listed above, and a kind this surface cannot receive would be a branch no
	 * producer can reach.
	 */
}

/**
 * Does this record paint anything the reader can see?
 *
 * For an assistant record the answer is its TEXT, and nothing else. An empty
 * one paints nothing whether it is finished or still streaming. Both halves of
 * that are deliberate: the streaming case is the "Writing" row discussed below,
 * and the SETTLED case — a finished assistant record that never carried prose,
 * because its tool rows were the whole turn — is the invisible-row trap
 * immediately below. One predicate covers both because they are the same fact:
 * a record with nothing in it.
 *
 * THE INVISIBLE-ROW TRAP, because this is exactly the kind of thing that gets
 * reintroduced. The reducer deliberately KEEPS an assistant record that carried
 * only tool calls: it has no prose to show — its tool rows carry the turn — but
 * it still owns its backend id so a live echo of the same turn coalesces onto
 * it instead of duplicating it (see the reducer's durable branch, "An assistant
 * row that only carried tool calls has no prose to show… It still occupies its
 * id"). Deleting it there is not an option.
 *
 * So it must exist in the RECORD list and must never reach the ROW list. When
 * it did, it cost the transcript twice over: it minted a wrapper carrying a top
 * margin for a box of zero height, AND — because the gap tier is decided from
 * the PREVIOUS record — it broke the trace-adjacency chain, so the next tool
 * row fell back to the wider `item` tier. Measured on the shipped build: two
 * adjacent tool rows 48px apart where a tight pair was 28px, at exactly the
 * points where the model happened to emit a prose-free tool turn. That is the
 * "extra space randomly inserted which doesn't look very uniform" in the
 * operator's report — the gap had no visible cause because its cause was
 * invisible.
 *
 * ## Why `streaming` is NOT a reason to paint
 *
 * It used to be. A streaming record with no text yet is the gap between
 * `message_start` and the first token, and the transcript minted a row for it
 * that read "Writing" — which put a second liveness element directly above the
 * working line, where that line was already saying `thinking`. Two elements for
 * one fact, and the redundant one sat in the ANSWER's register rather than on
 * the ledger, so it read as the turn having started when nothing had been
 * written.
 *
 * The TUI has one aggregate liveness element and no per-record equivalent
 * (`WorkingBlock`, `tui/widgets/transcript.py`), and the working line here is
 * the port of it: its `thinking` → `composing N calls` → `waiting to run N
 * calls` → `running` → `responding` ladder already covers every state this row
 * could have described, including composing tool calls, which paint as
 * `composing · N B` on their own tool rows, a call whose dictation has finished
 * and which nothing has started (`queued · N B`), and a call the harness will
 * never run (`never sent · N composed`) — by the harness's own verdict, or
 * because the turn ended while the call was still being dictated or waiting to
 * run, which the TUI settles with the same record. So liveness has ONE channel,
 * and a record with nothing in it paints nothing.
 *
 * ### The safety net that went with it, and why it is not needed
 *
 * The old `|| record.streaming` branch also covered a DESYNC: if a record said
 * it was streaming while the working line was suppressed, the "Writing" row was
 * the only thing left moving. The two are not derived from the same thing by
 * accident — the working line's visibility keys on the SESSION-level
 * `frontend.streaming` flag (`chat-page.tsx` reads
 * `canonical.frontend?.streaming` into `busy`, which arrives here as `waiting`
 * via `chat-content.tsx`), not on
 * any record in the list, so it is live for the whole provider call regardless
 * of what the record list currently holds — and `agent_end` settles the record
 * and that flag together. What USED to be offered as the second half of that
 * argument no longer holds, and is restated here rather than left standing
 * (code review round 1, R1-4): the claim was that `dropLiveRecords` cleared live
 * records on every gap, so nothing could leave one true and the other false.
 * A gap now KEEPS its streaming rows and marks them uncertain
 * (`markLiveRecordsTruncated`), so "a row says it is streaming while the
 * session-level flag says it is not" IS reachable by the normal event path: the
 * gap drops `frontend` to `null` — which is what the pane's reconnecting state
 * reads — while the row it was writing stays on screen, and it is the intended
 * state rather than a desync (the row keeps the text this viewer received; the
 * pane stops claiming a session state it cannot prove). The row is therefore NOT
 * painted from a per-record liveness rule: it is a real row with real text, and
 * the working line is suppressed independently of it. If a future change derives
 * `waiting` from the record list instead, this net has to come back with it.
 *
 * ### The 46.4px step at the first token, accepted deliberately
 *
 * Removing the row unmasked a real motion: before the first token the working
 * line is alone in the column, and at the first token the avatar and the first
 * line of prose appear ABOVE it, so the working line drops from top 96.4 to
 * 142.8 at 1024 — 46.4px in one frame (design review round 1, D5, measured
 * across the two consecutive states). This is accepted, not overlooked.
 *
 * It is legible rather than glitchy: what pushes the status line down is the
 * answer the reader is waiting for, and a status line yielding to content is an
 * event with a cause on screen. The alternative — showing the avatar before the
 * first token so the gutter is already occupied — removes only the horizontal
 * half of the step, adds an avatar that then has to move again when the prose
 * row mounts under it, and the version that removes the step entirely is the
 * avatar taking a real flex slot, which is the row restructure
 * `message-container.tsx` documents as deliberately deferred with a measured
 * collision. Paying that for 46.4px is the wrong trade; if the step is ever
 * reported, fix it there rather than by reinstating a row.
 *
 * Adjacency and spacing are therefore computed over what the reader can SEE,
 * never over what the record list contains. `AssistantRow` returns `null` on
 * this same predicate rather than on a copy of the condition, because two
 * copies of it are how the row comes back.
 */
export function paintsSomething(record: CondenseRecord): boolean {
	if (record.kind !== "assistant") return true;
	return Boolean(record.text);
}

/**
 * Rows that share the machine-voice ledger tier, and so sit tight against each
 * other. Prose and user turns are a different register and take real air.
 */
export function isTraceLike(record: CondenseRecord): boolean {
	return (
		record.kind === "tool" ||
		record.kind === "notice" ||
		record.kind === "compaction" ||
		record.kind === "custom" ||
		// A peer message and a wake delivery are receipts on the same ledger as
		// the calls around them (the TUI draws both as ledger rows —
		// `PeerMessageBlock` and `WakeBlock` in `tui/widgets/transcript.py`), so
		// they take the ledger's gap tier rather than prose's air. A note that
		// arrived mid-run belongs to the run.
		record.kind === "peer" ||
		record.kind === "wake"
	);
}

/**
 * The name this record paints in the ledger's shared name column, or `""` for a
 * record that has no ledger row at all.
 *
 * The column is sized to the longest name ON SCREEN, so a `peer` or `wake` row
 * that kept its name out of that set would break the column for every row around
 * it. Its own name is its kind — the record carries no `toolName`, because these
 * are not tool calls and a field that only ever repeats the kind would be a
 * second way of saying one thing.
 */
export function ledgerName(record: CondenseRecord): string {
	if (record.kind === "tool") return displayName(record.toolName);
	if (record.kind === "peer" || record.kind === "wake") {
		return displayName(record.kind);
	}
	return "";
}

/**
 * Row wrappers are reused across builds when the record AND its layout facts
 * (avatar, gap) are unchanged. The reducer already hands back the same record
 * object for untouched rows; without this pass every commit would still mint
 * a fresh wrapper per row and defeat the memo on `TranscriptRow`, which is
 * exactly what the render counter showed (rows x commits, not 1 per delta).
 */
export function buildRows(
	records: CondenseRecord[],
	previousRows: Row[],
): Row[] {
	const reusable = new Map(previousRows.map((row) => [row.record.id, row]));
	const rows: Row[] = [];
	const closingAnswers = closingAnswerIds(records);
	// The last record that PAINTED, not the last record. An invisible record
	// never becomes `previous`, so it can neither contribute a margin of its own
	// nor downgrade the gap tier of the row after it.
	let previous: CondenseRecord | null = null;
	for (const record of records) {
		if (!paintsSomething(record)) continue;
		const traceLike = isTraceLike(record);
		const previousTrace = previous !== null && isTraceLike(previous);
		let gap: Row["gap"] = "item";
		if (!previous) gap = "first";
		else if (record.kind === "user" || previous.kind === "user") gap = "turn";
		else if (traceLike && previousTrace) gap = "trace";
		/*
		 * A caption INSIDE a row makes the gap above that row the thing the eye
		 * measures first, and design round 1 raised it to a `mark` tier for exactly
		 * that reason: 8px above the row against the caption's own 4px to its chunk
		 * left the line reading as a note on the paragraph above it, so the marked
		 * row took the ramp's 12px between-components step instead.
		 *
		 * THAT TIER IS GONE, because it no longer differs from anything. §D1 sets
		 * every gap inside a turn to 12px, so `item` IS 12px now - the same value,
		 * for the same reason, decided once. A distinct tier whose two entries are
		 * the same class string is a second name for one decision, and it is the kind
		 * of second name that later drifts: the assertion that used to police the
		 * raise ("mark >= item * 1.5") is unreachable at these values.
		 */
		const closesTurn = closingAnswers.has(record.id);
		const prior = reusable.get(record.id);
		rows.push(
			prior &&
				prior.record === record &&
				prior.gap === gap &&
				prior.closesTurn === closesTurn
				? prior
				: { record, gap, closesTurn },
		);
		/*
		 * A quiet call NEVER becomes `previous`: the row paints nothing
		 * (`TranscriptRow` drops it), so the row after it must take its gap from
		 * the last row the reader can actually SEE - exactly the invisible-row
		 * trap the assistant arm above documents (a minted wrapper for an
		 * invisible row broke the trace-adjacency chain and spaced two adjacent
		 * tool rows 48px apart).
		 */
		if (!isQuietTurnCall(record)) previous = record;
	}
	return rows;
}

/**
 * The vertical ladder, `[comfortable, small view]`.
 *
 * Five tiers, and the DISTANCE BETWEEN TIERS is the information: a reader tells
 * "still the same run" from "a new turn started" by the size of the gap alone,
 * because nothing else on the surface marks a boundary (§ 2: remove a border
 * before you tighten the spacing — there are no borders left here to remove).
 *
 * - `trace` is a 2px HAIRLINE. Adjacent ledger rows read as one block the eye
 *   runs down rather than as a list of separated items — the operator's "much
 *   too wide" and the TUI reference's ~20.4px per line — but they are not
 *   fused: at zero the run was one unbroken column in which a reader looking
 *   for where one call ended and the next began had only the text to go on.
 *   Two pixels is the smallest step that reinstates that boundary, and it is
 *   deliberately NOT four: the designer rendered the same run at both, and at
 *   4px the rows float as separate items again, which is the spacing this
 *   whole tier was tightened away from. So a run of N rows measures
 *   `N × 20px + (N-1) × 2px` — the tool row's dense trigger is 20px — and
 *   uniformity is still structural rather than lucky, because every adjacent
 *   like pair gets the same tier and therefore the same distance.
 * - `item` separates the two REGISTERS (prose and ledger) inside one agent
 *   turn. Small, but four times the hairline: proportional prose sitting flush
 *   on a monospace row reads as one wrapped paragraph.
 * - `turn` is deliberately left wide. Tightening a run is only correct if the
 *   turn boundary survives it, and the contrast is what carries the hierarchy —
 *   24px against 2px inside a run in the comfortable view, where it used to be
 *   24px against 4px. In the SMALL view that contrast is 16px against the same
 *   2px (`turn` is `mt-4` there), which is a smaller ratio but still an order of
 *   magnitude, and it is the narrower column's own doing rather than this tier's.
 * - `mark` attaches a truncated row's own caption TO that row rather than to the
 *   paragraph above it (design round 1, D1). Without it the caption's only
 *   separation from the row above was the 8px `item` gap while its own margin to
 *   the chunk is 4px — a 2:1 ratio between two values that both sit inside a
 *   component's tier, so neither side read as a boundary and the line parsed as a
 *   note on the paragraph above it, which is a COMPLETE answer under it. It takes
 *   the ramp's between-components step, 12px, against the caption's 4px: 3:1, and
 *   a boundary larger than anything inside a paragraph (the prose's own line
 *   pitch is 24px, so the ratio is what has to carry it, not the absolute value).
 *   It is the FIRST tier pinned across both views — 12px is the smallest honest
 *   value and the small view's own `item` is 6px, so shrinking it would put the
 *   caption back below its own floor.
 *
 * The hairline does NOT shrink in the small view, unlike every other tier — it
 * is the only TIER whose two values are equal in the table below (`first` also
 * carries two identical entries, but its pair is empty: it is the absence of a
 * margin, not a width held across both views). The tiers above
 * shrink because they are made of several pixels to spend; 2px is already the
 * floor at which a gap is still a gap, and 1px on a hairline reads as an
 * antialiasing artifact rather than as a boundary. Note that this is therefore
 * not photographable: with one value in both views no frame can distinguish
 * them, and the property is pinned by the test over `GAP.trace`'s two indices
 * instead (`scripts/tool-row.test.mjs`).
 */
export const GAP: Record<Row["gap"], [string, string]> = {
	first: ["", ""],
	// 32px - §B1's largest tier, and §D1's stated turn boundary. It no longer
	// shrinks in the small view: this tier is what carries the hierarchy, and the
	// 1024 view is not the narrow column the old 16px was compensating for.
	turn: ["mt-8", "mt-8"],
	// 12px - §D1's "inside a turn everything sits on 12px", which is both the step
	// between the two registers (prose and ledger) and between a turn's own blocks.
	item: ["mt-3", "mt-3"],
	// 2px on the 4px ramp, the same step `TraceGroup` composes its lines with.
	trace: ["mt-0.5", "mt-0.5"],
};
