/**
 * THE TURN SEGMENTS AND THE VISIBLE ROW SET, ported from the desktop.
 * Ported from `damianvtran/local-operator-ui` at sourceHash f9f4611dd3dbcc06…:src/renderer/src/features/chat/canonical/turn-segments.ts:71-1145
 *
 * WHAT THIS PORT IS FOR. The phone must condense a settled turn the way the
 * desktop does: which row is the turn's ANSWER (the last RESPONSE cycle's close,
 * not the last message emitted), which of the remaining rows must stay on screen,
 * and how the rest partitions into ordered disjoint SEGMENTS a bar can stand for.
 * The decisions are checked against the desktop's own generated fixture
 * (`fixtures/condense.parity.json`, `condense.test.ts` replays it); the WORDS in
 * here are only what the model needs to identify a label (`SegmentLabelKind`).
 *
 * HOW IT DIFFERS FROM THE DESKTOP FILE, exhaustively:
 * - `TranscriptRecord` is this surface's `CondenseRecord` (see
 *   `condense-records.ts` for the mapping and every wire gap).
 * - the display-mode module (`../transcript-display-mode`) is folded in here: the
 *   mode is two tokens and a default, both surfaces read the same two, and a
 *   third file for a union this file is the only model consumer of buys nothing.
 * - nothing else. No predicate, no ordering and no arm was re-decided.
 */

import type { CondenseRecord } from "./condense-records";

/** How a settled turn's transcript rows are drawn. */
export type TranscriptDisplayMode = "by-turn" | "by-response";

/** The shipped mode: the desktop's condensation, and the fallback for anything
 * else (an unknown token from a newer build lands here, never an error). */
export const DEFAULT_TRANSCRIPT_DISPLAY_MODE: TranscriptDisplayMode = "by-turn";
export type TriggerKind =
	| "user"
	| "peer"
	| "wake"
	| "job_result"
	| "hub_message"
	| "monitor_prompt";

export type CycleClass = "response" | "commentary";

/**
 * Why a cycle has the class it has. Diagnostic (it names fixtures and failure
 * messages); `class` is what renders.
 */
export type CycleWhy = "user" | "continuation" | "post-terminal" | "head-cut";

export type TurnCycle = {
	/** Index (in the run's record list) of the cycle's first record. */
	start: number;
	/** Index of the cycle's close row: the message that ends this response. */
	closeIndex: number;
	class: CycleClass;
	why: CycleWhy;
	/** What set the cycle going, in order, as far as the journal shows it. */
	initiators: TriggerKind[];
};

/**
 * The custom row types that start a cycle of work the agent had already begun.
 * A job's result, a hub relay and a monitor's prompt are the agent finishing
 * something it owns, so the cycle they open INHERITS the previous cycle's class
 * rather than being ambient chatter.
 */
const CONTINUATION_CUSTOMS: ReadonlySet<string> = new Set([
	"job_result",
	"hub_message",
	"monitor_prompt",
]);

/**
 * The quiet-turn tool, and the two readings of its row (design §5, rev 2).
 *
 * A turn the agent ended with `no_reply` is a SILENT turn: core persists the
 * call and its result as ordinary tool rows (no wire field, no capability
 * flag), and this module is where the transcript turns the pair's structural
 * fact - a settled quiet call CLOSES its cycle - into the closure every
 * consumer reads. An old client that does not know the name simply renders a
 * tool row, which is the accepted degradation.
 *
 * WHY THE NAME IS THE SIGNAL AND NOT THE RESULT TEXT: the UI explicitly avoids
 * text heuristics (see this file's header), and the result's own string is not
 * a contract. `toolName` is.
 */
export const QUIET_TURN_TOOL = "no_reply";

/** The row IS the quiet call, whatever its phase. */
export function isQuietTurnCall(
	record: CondenseRecord,
): record is Extract<CondenseRecord, { kind: "tool" }> {
	return record.kind === "tool" && record.toolName === QUIET_TURN_TOOL;
}

/**
 * The row CLOSES its cycle: the call is settled (`phase === "done"`) and it
 * genuinely ran - a refused, never-sent or aborted call did not end the turn.
 *
 * THE REFUSAL IS THE LIVE CASE (design §5): when a person asked this turn, core
 * answers `no_reply` with an `is_error` result and the model writes text
 * instead - so an errored quiet call must NOT settle the tail, or the bar would
 * announce a quiet end and swallow the answer that is about to arrive. The
 * same exclusion set serves the other non-ends: a never-sent call (the turn
 * died while it was being dictated) and an aborted one (the reader stopped the
 * turn). It is `isFailedCall`'s own never-run/interrupt reading, inverted, so
 * "closed" and "failed" can never both be true of one row.
 *
 * An in-flight call (`phase !== "done"`) is a call still running: reading it as
 * a close would retire the tail before the turn has ended.
 */
export function isQuietTurnClose(record: CondenseRecord): boolean {
	if (!isQuietTurnCall(record)) return false;
	if (record.phase !== "done") return false;
	if (record.isError === true) return false;
	if (record.stopped === true) return false;
	if (record.neverSent === true) return false;
	if (record.notRunReason !== null && record.notRunReason !== undefined) {
		return false;
	}
	return true;
}

/** The trigger a record is, or null when it starts nothing. */
export function triggerOf(record: CondenseRecord): TriggerKind | null {
	switch (record.kind) {
		case "user":
			return "user";
		case "peer":
			return "peer";
		case "wake":
			return "wake";
		case "custom":
			return CONTINUATION_CUSTOMS.has(record.customType)
				? (record.customType as TriggerKind)
				: null;
		default:
			return null;
	}
}

/**
 * The rows that are a BOUNDARY of a turn rather than a step of it, and which
 * kind - the ONE place the transcript decides it.
 *
 * WHY ONE FUNCTION. Four things used to keep their own list of "the rows that
 * mark where something ended": the collapse's pin list (what stays on screen),
 * the classifier's terminal test (what makes a later wake ambient), the run
 * partition's `sawMarker`, and the rows' own paint. They agreed by copy, which is
 * how a `closed` receipt could be pinned and yet not count as the end of a turn.
 * Now both consumers below are one-line reads of this, so a new marker (a
 * `retired` receipt was the last one added) joins every rule the day it joins
 * this switch.
 *
 * - `compaction` - the conversation's memory statement. It changes what the agent
 *   remembers but does not end anything, so it pins (stays where it happened)
 *   without making a later message ambient;
 * - `terminal` - the transcript's own "this turn is over": a completion `notice`
 *   (`Stopped with an error`, `Interrupted`, and the neutral `closed` / `retired`
 *   receipts, all of which carry `complete === true`) and an error-level `custom`
 *   (a `session_incident`, the reason a turn died). It is the operator's dispose,
 *   verbatim.
 */
export type BoundaryKind = "compaction" | "terminal";

/**
 * A settled assistant row whose OWN PROVIDER DECLARATION says it had finished:
 * `stop_reason: "stop"` (Codex ships the same idea as a phase field,
 * `MessagePhase::Commentary | FinalAnswer` - and its own warning, "providers do
 * not emit this consistently, so callers must treat `None` as phase unknown").
 * The durable reducer keeps that field on every text-bearing assistant row
 * (`transcript-reducer.ts:2264`), so this reads a fact the record already states.
 *
 * WHY A DECLARATION AND NOT A LENGTH. The rows this rescues are the ones the
 * harness continued past: a `stop`-then-more-work pair is what the todo guardrail
 * manufactures by re-entering the loop after a no-tool-call yield. A corpus read
 * (the design note behind this change: 10,308 journals, 13,932 runs) found those
 * rows at 0.23 per run and a median 2,540 characters, while the narration that
 * must KEEP hiding (text carried in the same frame as its own `tool_calls`,
 * `stop_reason: "toolUse"`) numbered 203,412 at a median 122 - so "is it long"
 * would happen to separate them, and is still the wrong instrument: a phase says
 * what the model MEANT, and it keeps working for a terse report (the 261-character
 * wake reply in the operator's own journal is the whole complaint). Measured on
 * this machine's own journals (the frozen 2026-10-01 recount recorded on PR #737:
 * 10,367 journals, 15,130 runs, 254,999 prose rows), the rows this rule adds stay
 * at 3,841 = 1.51 % of prose rows and 0.254 per run.
 *
 * THE PHASE AND THE SHAPE BARELY DISAGREE, WHICH IS THE POINT. On the same frozen
 * corpus 20,933 rows declare `stop` and 233,783 declare `toolUse`, and the two
 * populations are almost disjoint in shape: only **4** of the 20,933 `stop` rows
 * carry their OWN `tool_calls` (the lead-in frame), and exactly **1** of the
 * 233,783 `toolUse` rows is not followed by a call. So a length test would happen
 * to separate them too - and would still be the wrong instrument, because it reads
 * the symptom while the phase reads what the model MEANT. The price of reading the
 * declaration is those 4 leading-in `stop` rows staying visible; the price of
 * reading the shape would be every terse report hiding again.
 *
 * UNKNOWN IS ABSENT. `null`/`undefined` never fires, so a record built without the
 * field behaves exactly as it did before this predicate existed. Where it bites: on
 * this machine's journals NO text-bearing assistant row lacks the field at all
 * (`null`/`undefined`: 0 rows; other reasons such as `aborted` / `length`: 285 =
 * 0.11 %), so this clause is exercised by fixtures and by callers that build
 * records by hand - it is compatibility, not a live path.
 */
export function reportsCompletedThought(record: CondenseRecord): boolean {
	return (
		record.kind === "assistant" &&
		!record.streaming &&
		record.stopReason === "stop"
	);
}

/**
 * Is this row a FINAL RESPONSE to the reader - the agent handing something over
 * rather than narrating its own work?
 *
 * THE RULE IS `tool_calls`, and only `tool_calls`. An assistant message that
 * carries its own calls is mid-turn narration/status commentary ("Reading the
 * ledger first." followed by the call it announces); one that carries none is a
 * handover, whether it is the turn's last row or not. The core states the same
 * rule in `tui/narration.py::is_intermediate_narration`, and the reducer keeps
 * the fact on the record (`CondenseRecord.callsTools`, set from the journal
 * payload and from `message_end`).
 *
 * WHY NOT POSITION, `closesTurn`, LENGTH OR A TEXT TEST. Every one of them is
 * wrong on real journals: a wake reply mid-turn is short, mid-turn AND final; a
 * 2,500-character "checking the ledger" paragraph is long AND commentary; and
 * `closesTurn` elects ONE row per turn while the operator asked for a bar on
 * EVERY final response, including the ones between tool calls. `stopReason`
 * does not stand in for the call fact either: a `toolUse` with empty
 * `tool_calls` is reachable and is a final response (the corpus behind PR #737
 * records exactly one such row).
 *
 * SETTLED AND PAINTING TEXT, because a bar exists to copy/hand over prose: a
 * streaming prefix is not settled, and a row with no text has nothing to offer
 * (a tool-call-only row paints nothing at all - `paintsSomething` drops it).
 * `stopReason` is deliberately NOT consulted, which is what makes this predicate
 * WIDER than `reportsCompletedThought`: it also covers the `null` and
 * `aborted`/`length` rows the phase clause cannot see.
 *
 * THE DIRECT TEXT CHECK rather than `paintsSomething`: that predicate lives in
 * `transcript-rows.ts`, which imports this module for `closingAnswerIds`, and a
 * second import edge would be a cycle. For an assistant record `paintsSomething`
 * IS `Boolean(record.text)`, so this is the same fact stated once for the one
 * kind it can be asked of - with the same predicate shape pinned beside it.
 */
export function isFinalResponse(record: CondenseRecord): boolean {
	return (
		record.kind === "assistant" &&
		!record.streaming &&
		record.callsTools !== true &&
		record.text.length > 0
	);
}

/**
 * A durable COMPLETION MARKER: the notice the transcript paints for a completion
 * attention record (`Stopped with an error`, `Interrupted`, and the neutral
 * `closed` / `retired` receipts).
 *
 * WHO READS IT (moved by the run-closure predicate change, 2026-09-30, review
 * round 1's R5). This used to be the shared test three consumers read - the run
 * partition's "a marker closes this run" clause among them - which is exactly how
 * the narrow copy drifted from `boundaryKindOf`: the partition ignored the
 * error-level `custom` a `session_incident` is, so a killed turn's retry was
 * absorbed as a steer (`walkTurns` in `transcript-rows.ts` states the defect).
 * The partition now reads `isTerminalMarker`. What is left here is one consumer,
 * `working-line-model.ts`'s `turnStopped` - the live working line's "the turn
 * stopped" reading, where a completion notice genuinely is the whole question and
 * an error-level custom is not - and `boundaryKindOf`'s own notice arm. That
 * asymmetry is deliberate: the boundary vocabulary is what the transcript uses to
 * END things, and only the two readers above ask the narrower question.
 */
export function isCompletionMarker(record: CondenseRecord): boolean {
	return record.kind === "notice" && record.complete === true;
}

export function boundaryKindOf(record: CondenseRecord): BoundaryKind | null {
	switch (record.kind) {
		case "compaction":
			return "compaction";
		case "notice":
			return isCompletionMarker(record) ? "terminal" : null;
		case "custom":
			return record.level === "error" ? "terminal" : null;
		default:
			return null;
	}
}

/** Does this boundary declare the turn over? See `boundaryKindOf`. */
export function isTerminalMarker(record: CondenseRecord): boolean {
	return boundaryKindOf(record) === "terminal";
}

/**
 * A run's cycles, in order, each classified.
 *
 * `records` is ONE RUN's records (the caller owns the run partition). `paints`
 * is `paintsSomething`, injected rather than imported: `transcript-rows.ts`
 * imports this module for `closingAnswerIds`, and a second copy of the predicate
 * is how the invisible-row trap comes back.
 *
 * The decision table, one forward pass because the continuation rule depends on
 * the previous cycle:
 *
 * | what set the cycle going                                   | class          | why           |
 * |------------------------------------------------------------|----------------|---------------|
 * | a `user` row (the request, or a steer)                     | response       | user          |
 * | only job_result / hub_message / monitor_prompt             | previous cycle | continuation  |
 * | wake / peer, and a terminal marker lies since the last close | commentary     | post-terminal |
 * | wake / peer, no terminal marker                            | previous cycle | continuation  |
 * | nothing visible                                            | previous cycle | continuation  |
 * | no previous cycle exists                                   | response       | head-cut      |
 *
 * The "wake carried the answer" shape is real and common (1265 wake-triggered
 * cycles in the journals sampled ARE the turn's answer), so inheritance is the
 * default and `commentary` needs positive evidence: a terminal marker between
 * the previous close and the ambient trigger. That is a wake or a peer note
 * arriving AFTER the turn was declared over.
 */
export function cyclesOf(
	records: readonly CondenseRecord[],
	paints: (record: CondenseRecord) => boolean,
): TurnCycle[] {
	const n = records.length;
	const isClose: boolean[] = new Array(n).fill(false);
	/*
	 * Backward pass: an assistant row is a close when NO step row follows it
	 * before the next trigger. Walking from the end keeps it linear.
	 */
	let stepAhead = false;
	for (let i = n - 1; i >= 0; i -= 1) {
		const record = records[i];
		if (triggerOf(record) !== null) stepAhead = false;
		else if (record.kind === "tool") {
			/*
			 * A SETTLED quiet call is a structural close (design §5(a)): the turn
			 * ended there on purpose, so the cycle ends there too - which is what
			 * settles the tail (`settledCloseOf` in `turn-collapse-model.ts` reads
			 * these cycles) without anything claiming an answer was handed over.
			 * It is still a STEP for the row before it: narration written ahead of
			 * a quiet call stays narration, exactly as ahead of any other call.
			 */
			if (isQuietTurnClose(record)) isClose[i] = true;
			stepAhead = true;
		} else if (
			record.kind === "assistant" &&
			!record.streaming &&
			paints(record) &&
			!stepAhead
		) {
			isClose[i] = true;
		}
	}

	const cycles: TurnCycle[] = [];
	let start = 0;
	let terminalSinceClose = false;
	for (let i = 0; i < n; i += 1) {
		const record = records[i];
		if (isTerminalMarker(record)) terminalSinceClose = true;
		if (!isClose[i]) continue;

		let firstStepIndex = -1;
		const initiators: TriggerKind[] = [];
		for (let j = start; j < i; j += 1) {
			const trigger = triggerOf(records[j]);
			if (firstStepIndex === -1 && records[j].kind === "tool") {
				firstStepIndex = j;
			}
			// Initiators are what preceded the work, except that a `user` row
			// anywhere in the cycle is a request (a steer arrives mid-work).
			if (trigger !== null && (firstStepIndex === -1 || trigger === "user")) {
				initiators.push(trigger);
			}
		}

		const previous = cycles.at(-1) ?? null;
		let cls: CycleClass;
		let why: CycleWhy;
		if (previous === null) {
			// The run's first cycle is a response whatever set it going: it is
			// either the user's own (the request opens the run) or the head of a
			// run whose opener is not loaded, which must still elect an answer.
			cls = "response";
			why = initiators.includes("user") ? "user" : "head-cut";
		} else if (initiators.includes("user")) {
			cls = "response";
			why = "user";
		} else if (
			initiators.some((kind) => kind === "wake" || kind === "peer") &&
			terminalSinceClose
		) {
			cls = "commentary";
			why = "post-terminal";
		} else {
			cls = previous.class;
			why = "continuation";
		}

		cycles.push({
			start,
			closeIndex: i,
			class: cls,
			why,
			initiators,
		});
		start = i + 1;
		terminalSinceClose = false;
	}
	return cycles;
}

/**
 * The predicates the classifier borrows from `transcript-rows.ts`, INJECTED.
 *
 * `transcript-rows.ts` imports this module (for `closingAnswerIds`), so importing
 * `paintsSomething` / `isStatementRow` back would be a cycle; and a second copy of
 * either predicate is how the invisible-row trap comes back. The caller passes the
 * one definition each.
 */
export type RowPredicates = {
	/** `paintsSomething`: a record with nothing to show is not a close. */
	paints: (record: CondenseRecord) => boolean;
	/** `isStatementRow`: a record ABOUT the conversation, not the turn's own work. */
	isStatement: (record: CondenseRecord) => boolean;
};

/**
 * The elected answer of a run: its LAST response cycle, or null when the run has
 * not handed anything over.
 *
 * NULL IS THE OLD RULE'S GATE, KEPT ON PURPOSE. A run whose last piece of work is
 * a tool row, or an answer that is still streaming, has not handed the reader its
 * answer yet - stripping the caption there is right (the answer it will end on has
 * not been written), and electing an earlier close would put the foot on a row the
 * next token is about to demote. So the run's last painting, non-statement record
 * must be a settled assistant; only then is there an answer to elect, and it is
 * the last RESPONSE cycle's close rather than that final row itself. Statements
 * after it (a receipt, a notice) do not count as work, exactly as
 * `isStatementRow` has always said.
 */
export function electAnswer(
	records: readonly CondenseRecord[],
	cycles: readonly TurnCycle[],
	predicates: RowPredicates,
): TurnCycle | null {
	const lastClose =
		cycles.length > 0 ? cycles[cycles.length - 1].closeIndex : -1;
	let terminalAfterClose = false;
	for (let i = records.length - 1; i > lastClose; i -= 1) {
		if (isTerminalMarker(records[i])) terminalAfterClose = true;
	}
	for (let i = records.length - 1; i >= 0; i -= 1) {
		const record = records[i];
		// A user row is never the run's "last content" (a steer is not work the
		// turn did), and a record that paints nothing is invisible to the reader.
		if (record.kind === "user" || !predicates.paints(record)) continue;
		if (predicates.isStatement(record)) continue;
		/*
		 * The one refinement of the old gate: work that trails the last close AND
		 * is followed by a terminal marker is a follow-up that DIED (a wake's work
		 * cut by the disposal), not an answer still being written. The answer that
		 * was handed over before it is still the answer; refusing to elect it would
		 * hide the reader's answer behind a bar for the sake of a turn that is over.
		 */
		if (record.kind === "assistant" && !record.streaming) break;
		if (record.kind !== "assistant" && terminalAfterClose) break;
		return null;
	}
	for (let i = cycles.length - 1; i >= 0; i -= 1) {
		if (cycles[i].class !== "response") continue;
		if (isQuietTurnClose(records[cycles[i].closeIndex])) {
			/*
			 * A quiet close is excluded AS A CANDIDATE ONLY (design §5(c)): the run
			 * handed the reader nothing, so nothing may carry the caption, the foot
			 * or the stamp. `cyclesOf` still records it as the cycle's close, and
			 * that half is load-bearing - dropping it from the scan would leave the
			 * tail with no close at all, the unsettled state this rule exists to
			 * fix.
			 *
			 * AND NO FALLBACK PAST IT: an earlier cycle's close was already handed
			 * over, and re-electing it would label the run's end with an earlier
			 * answer. The run is in the same state a textless tail has always had -
			 * nothing to elect - so this returns null rather than reaching back.
			 */
			return null;
		}
		return cycles[i];
	}
	return null;
}

/** A maximal contiguous span of hidden records, inclusive, in run indices. */
export type SegmentSpan = { from: number; to: number };

export type RunPartition = {
	cycles: TurnCycle[];
	answer: TurnCycle | null;
	/** Ordered, disjoint hidden spans (see the invariants in `violationsOf`). */
	segments: SegmentSpan[];
	/** Run indices that stay mounted while the run is collapsed. */
	visible: ReadonlySet<number>;
};

/**
 * Partition one run into the rows that stay and the SEGMENTS that hide.
 *
 * WHAT STAYS VISIBLE, in one sentence: the rows the reader is owed - every pinned
 * row (compaction, a completion marker, an error incident - `pinned`), the ANSWER
 * row itself, the run's trailing statements (a receipt or notice after the last
 * close), every close of a RESPONSE cycle, the run's LAST close whatever its class,
 * every `user` row in the span, and any settled row whose own provider declaration
 * is `stop_reason: "stop"`. EVERYTHING ELSE in the span hides - the turn's work,
 * its narration, and every commentary cycle - and hides as SEGMENTS: the maximal
 * contiguous hidden spans between the visible rows.
 *
 * WHY MORE THAN THE ANSWER (the operator's report, "a fulsome response with
 * completion details that we'd want to see, but I have to click to expand"). The
 * old rule kept exactly one close, so a turn that answered, was continued past, and
 * answered again hid the FIRST answer - the one carrying the substance - inside the
 * bar, and the reader had to expand it to read what the agent had already handed
 * over. The five clauses below are the invariant that replaces it, in row-list
 * terms:
 *
 * - V1, a RESPONSE cycle's close: the agent answered something (the request, a job,
 *   a hub relay, a monitor prompt), so the reader is owed it;
 * - V2, the run's LAST close: the reader is owed the last word even when that cycle
 *   is commentary;
 * - V3, a `user` row: the one row in the span the READER wrote. A mid-turn steer
 *   used to be hidden with the work it steered and the bar had to say `Steered` to
 *   admit it (`labelOfSegment`); the surveyed harnesses all keep the reader's own
 *   message in place, and the cost is measured (11.6 % of runs gain one bar);
 * - V4, a settled row the provider declared finished (`reportsCompletedThought`);
 * - V5, a FINAL RESPONSE (`isFinalResponse`) INSIDE A RESPONSE CYCLE: settled,
 *   paints text, and carries no `tool_calls` of its own. This is the operator's
 *   rule rather than the phase's (2026-10-20): a row that hands something to the
 *   reader must keep its bar even when the harness continued past it, because
 *   that is exactly the shape the todo guardrail manufactures -
 *   "stop-then-more-work" - and the reader was being shown the later tool run
 *   with the earlier handover folded away. THE CYCLE CLASS IS THE SECOND HALF
 *   (round 3): a handover in a cycle that is ANSWERING something is the bar case,
 *   while a COMMENTARY cycle's close is post-terminal chatter - pinning it made
 *   a narration-only tail unfold. V2 still keeps the run's LAST close, so the one
 *   commentary close that ends the run stays on screen.
 *
 * WHAT THE FIVE CLAUSES DO NOT COVER (the falsifier agent review round 1 on PR
 * #737 produced, and it was real): a COMMENTARY close that is neither the run's
 * last close (V2) nor `stop`-declared (V4), with no call after it, still hid -
 * e.g. `[N0][W][K][N3][N4]`, where `n3` is commentary and the row after it is
 * another assistant row, not a call. Incidence on this machine's journals: exactly
 * 1 row in the frozen 15,130-run corpus. V5 CLOSES THAT RESIDUAL in the shape it
 * was measured in, because that row carries no `tool_calls`, is therefore a final
 * response by the wire's own reading, and sits in a cycle the wake/receipt
 * inherited as a RESPONSE cycle - so the clause that would cover it is not "any
 * commentary close, visible" (which would un-hide the ambient chatter V1/V2 were
 * shaped to fold) but "any handover in a response cycle, visible".
 *
 * WHAT V5 ADDS TO THE VISIBLE SET, measured on the same frozen corpus rather than
 * assumed: rows that are settled, paint text, carry no calls, and declare
 * something other than `stop` - exactly 1 `toolUse` row (the residual above),
 * 0 rows with no stop reason, and at most 285 `aborted`/`length` rows - so at most
 * 286 of 254,999 prose rows (0.11 %), and V5's cycle scope narrows that further -
 * a commentary-cycle close is no longer pinned, so the frozen figure is an UPPER
 * BOUND for the scoped clause rather than a re-measured count. Every row the
 * clause still adds is a final response by the rule. The rows the condensation exists to fold are untouched: 233,783 `toolUse`
 * rows carry their own `tool_calls`, and the 4 `stop` rows that carry calls were
 * ALREADY visible under V4 - V5 neither adds nor removes them.
 *
 * A V4/V5 ROW IS STILL NOT A CLOSE. It is visible while the run has more work after
 * it, so it splits its segment without splitting its cycle - `isClose`, `cyclesOf`
 * and `electAnswer` are untouched, and the turn keeps exactly ONE answer to carry
 * `closesTurn` (the foot and the one stamp). The segment machinery already renders
 * a visible group between two bars, so nothing else has to learn about it.
 *
 * WHY SEGMENTS AND NOT ONE BAR. A pinned row between two hidden spans (40% of the
 * runs in the journals sampled) used to render AFTER a bar that stood where the
 * first hidden row was, and jumped when the bar opened. Each span is its own bar
 * in its own slot, so a pinned row can never be reordered by expanding anything.
 *
 * THE DISPLAY MODE WIDENS, AND ONLY WIDENS (issue #756). `mode` is the reader's
 * `by-turn` / `by-response` choice (`transcript-display-mode.ts`), and it is an
 * OPTIONAL FIELD whose absence means the shipped default: `from` and the five
 * clauses above are computed identically in both modes, so V1-V5 keep their
 * meaning and `answer` is elected once, before the mode is consulted. `by-turn`
 * is therefore the partition this function shipped, with ONE declared exception:
 * V5 (`isFinalResponse`) keeps a handover the phase rule could not see visible
 * in both modes. In `by-response` EVERY settled text-bearing row joins the
 * visible set on top of the clauses, so a turn that answered, was continued past
 * and answered again
 * shows both answers in place while its tool work still condenses. The widening
 * is one-directional by construction: nothing is ever REMOVED from `visible` for
 * a mode, so a row the invariant keeps stays kept, and the elected answer keeps
 * carrying the caption and the foot (`closesTurn`) rather than becoming one of
 * several undifferentiated rows.
 *
 * `from` is the first index the span may start at: just after the opening user
 * row, or 0 for a run whose head is cut off.
 */
export function partitionRun(
	records: readonly CondenseRecord[],
	options: RowPredicates & {
		from: number;
		pinned: (record: CondenseRecord) => boolean;
		/**
		 * How the transcript draws this run. Omitted (or `by-turn`) is the shipped
		 * condensation; `by-response` adds every settled text-bearing row to the
		 * visible set. See this function's doc.
		 */
		mode?: TranscriptDisplayMode;
	},
): RunPartition {
	const cycles = cyclesOf(records, options.paints);
	const answer = electAnswer(records, cycles, options);
	const visible = new Set<number>();
	records.forEach((record, index) => {
		if (index >= options.from && options.pinned(record)) visible.add(index);
	});
	if (answer !== null) {
		visible.add(answer.closeIndex);
		/*
		 * The trailing statements: whatever follows the LAST close and does no work
		 * (no tool, no assistant text). The old rule kept everything from the
		 * answer onward, so `[user][answer][notice]` and a receipt after the
		 * answer stayed on screen; the walk stops at the first row that is work.
		 */
		const lastClose = cycles[cycles.length - 1].closeIndex;
		for (let i = records.length - 1; i > lastClose; i -= 1) {
			const record = records[i];
			if (!options.isStatement(record)) break;
			visible.add(i);
		}
	}

	/*
	 * THE VISIBILITY INVARIANT (see the doc above): a response close (V1), the
	 * run's last close (V2), the reader's own row (V3), and a row the provider
	 * declared finished (V4). V1 and V2 must not be folded into the answer check:
	 * a run can have several response closes and all of them stay.
	 */
	for (const cycle of cycles) {
		if (
			cycle.class === "response" &&
			!isQuietTurnClose(records[cycle.closeIndex])
		)
			visible.add(cycle.closeIndex);
	}
	if (cycles.length > 0) {
		const last = cycles[cycles.length - 1];
		/*
		 * A QUIET CLOSE IS NOT FORCED VISIBLE (design §10.10: "the row is hidden at
		 * paint"). It paints nothing, so forcing it on screen would only split
		 * every quiet cycle into its own bar around an invisible row - exactly the
		 * one-line-per-receipt shape the group bar exists to fold away. It stays a
		 * close for every closure consumer (above and in `electAnswer`), and the
		 * collapsible gate carries it via `quietCloseId` instead.
		 */
		if (!isQuietTurnClose(records[last.closeIndex])) {
			visible.add(last.closeIndex);
		}
	}
	const responseCycleRows = new Set<number>();
	/*
	 * V5 IS SCOPED TO A RESPONSE CYCLE (review round 3, finding 2). A handover in a
	 * cycle that is ANSWERING something (the request, a job, a hub relay, a monitor
	 * prompt) is the operator's bar case. A COMMENTARY cycle's close is a different
	 * fact - post-terminal chatter - and pinning it made a narration-only tail
	 * unfold (measured: `turn-collapse-model.test.mjs`'s post-terminal wake shape).
	 * V2 still keeps the run's LAST close, so a commentary close that ends the run
	 * stays on screen; only one that would otherwise fold does.
	 *
	 * A ROW IS A MEMBER OF THE CYCLE THAT CONTAINS IT, on its index range: every
	 * emitted response cycle contributes `start..closeIndex`, and the OPEN TAIL
	 * (rows after the last close, plus a run with no close at all) is covered
	 * separately below. An anchor on `closeIndex` alone is not enough - a row that
	 * is still being written has no close yet, and F5's monotonicity invariant
	 * caught it (`U F T T F T A`: the row left the set when a step demoted it from
	 * close to narration, before the cycle that owns it had a close).
	 */
	for (const cycle of cycles) {
		if (cycle.class !== "response") continue;
		for (let i = cycle.start; i <= cycle.closeIndex; i += 1) {
			responseCycleRows.add(i);
		}
	}
	/*
	 * THE OPEN TAIL belongs to the cycle that will own it, and an unfinished cycle
	 * has stated no close yet. `cyclesOf` only emits a cycle at a close, so two
	 * shapes would otherwise lose their pin mid-stream: a run with no close at all
	 * (its FIRST cycle is a response by `cyclesOf`'s head rule) and the rows after
	 * the last response cycle's close, which are the cycle currently being written.
	 * Both are covered here so a live handover does not vanish when the next call
	 * arrives and reappear when its turn closes.
	 *
	 * THE TAIL STARTS AT 0, not 1: a head-cut run has no opening user row, so its
	 * first row is a record whose own bar depends on this clause (round 4, C2 -
	 * `partitionRun([F, T], { from: 0 })` lost `F0`).
	 */
	const last = cycles.at(-1);
	const openTailFrom =
		last === undefined
			? 0
			: last.class === "response"
				? last.closeIndex + 1
				: records.length;
	for (let i = openTailFrom; i < records.length; i += 1) {
		responseCycleRows.add(i);
	}
	records.forEach((record, index) => {
		if (index < options.from) return;
		if (record.kind === "user") visible.add(index);
		else if (
			options.paints(record) &&
			/*
			 * BOTH HALVES: V4 keeps the stop-declared lead-in rows the shape test
			 * cannot see (they carry calls), and V5 keeps every handover the phase
			 * declaration is missing - but only inside a RESPONSE cycle (see above).
			 * Written as an OR rather than folded into one predicate so neither
			 * clause can lose the population the other exists for.
			 */
			(reportsCompletedThought(record) ||
				(isFinalResponse(record) && responseCycleRows.has(index)))
		) {
			visible.add(index);
		}
	});

	/*
	 * BY-RESPONSE WIDENING (issue #756). The reader asked to see every settled
	 * response rather than only the rows the invariant is forced to keep, so every
	 * settled text-bearing row joins the visible set - `streaming` rows are
	 * excluded because the in-flight cycle is drawn in place by the liveness rule
	 * anyway, and `paints` excludes an assistant row that carries no text. This
	 * runs AFTER the clauses above and only ever ADDS, which is what keeps
	 * `by-turn` untouched and keeps the elected answer's marking intact: the
	 * answer is still `electAnswer`'s, and no clause is recomputed.
	 */
	if ((options.mode ?? DEFAULT_TRANSCRIPT_DISPLAY_MODE) === "by-response") {
		records.forEach((record, index) => {
			if (index < options.from) return;
			if (record.kind !== "assistant" || record.streaming) return;
			if (options.paints(record)) visible.add(index);
		});
	}

	const segments: SegmentSpan[] = [];
	let open: number | null = null;
	for (let i = options.from; i < records.length; i += 1) {
		if (visible.has(i)) {
			if (open !== null) segments.push({ from: open, to: i - 1 });
			open = null;
		} else if (open === null) {
			open = i;
		}
	}
	if (open !== null) segments.push({ from: open, to: records.length - 1 });
	return { cycles, answer, segments, visible };
}

/**
 * What a segment's bar says about itself ahead of its clauses, or null for the
 * ordinary bar (whose copy - `Took ...`, `N actions` - is unchanged and needs no
 * word).
 *
 * ONE WORD PER CONCEPT, IN THE APP'S OWN NOUNS (design review round 1 on #708,
 * D2/D3). The first cut split the same trigger by position (`Woken` before the
 * answer, `Followed up` after; `Noted` / `Peer note`) and mixed three grammatical
 * registers. Position is already visible - the bar is above or below the answer -
 * so the word names only WHAT OPENED the cycle, as the noun the rest of the app
 * uses: the TUI's own labels for these rows are `wake` and `peer`, and the receipt
 * row prints the sender rather than a sentence about it.
 *
 * THE WORD COMES FROM THE CYCLE THE SEGMENT ENDS IN, not from any trigger row
 * that happens to lie inside it. A long user-opened turn is full of `job_result`
 * receipts between its steps; reading "the last trigger in the span" labelled that
 * whole bar `Job result`, which is not what it is. A cycle's INITIATORS are only
 * what preceded its first step, so a mid-work receipt cannot name the bar.
 *
 * | what the bar hides                         | label          |
 * |--------------------------------------------|----------------|
 * | a user row of the reader's own (a steer)   | Steered        |
 * | a cycle opened by a wake / monitor prompt  | Wake           |
 * | a cycle opened by a peer / hub message     | Peer message   |
 * | a cycle opened by a job result             | Job result     |
 * | anything else (the reader's own request)   | none           |
 *
 * `Steered` IS NOW A BACKSTOP, AND KEPT RATHER THAN DELETED. It shipped (agent
 * review round 1 on #708, R1-2) as the answer to "a reader's OWN message must not
 * disappear behind an unlabelled bar": the partition then hid a steer with the
 * work it steered, because pinning it would have split every steered turn into two
 * bars. The visibility invariant (V3, see `partitionRun`) took that trade back -
 * a `user` row is now never inside a span at all - so the arm above cannot fire
 * for a partition this module built, and its comment is the record of why.
 *
 * It stays because the FUNCTION is what promises the message is never swallowed:
 * `labelOfSegment` is exported and called on caller-supplied spans (and the
 * head-cut/first-page spans the loader can hand it), so removing the arm would
 * trade a one-word redundancy for the exact failure it was written to prevent.
 * `scripts/turn-segments.test.mjs` asserts both halves: no bar over a steer is
 * produced, and the word is what a span holding a user row would still be called.
 *
 * A QUIET GROUP OVERRIDES THE CYCLE OPENER (design §5): a span that IS a quiet
 * group (>= 2 trigger rows with nothing between them the reader can see) states
 * the family's plural word and its own count/span instead of one cycle's opener -
 * `Peer messages`, never `Peer message`, over twelve receipts. The group is
 * computed by the collapse model (which is where `isFailedCall` and the paint
 * predicates live) and handed IN, so this function keeps no second opinion about
 * what a group is; its only rule is the word.
 */
export type QuietGroupFamily = "peer" | "wake" | "monitor" | "job" | "mixed";

/** The family's word for a group bar: family plural, `Messages` for a mix. */
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

/**
 * WHAT A BAR IS, structurally, before any surface words it.
 *
 * WHY THE DECISION IS SEPARATE FROM THE WORD. The word (`Wake`, `Peer message`,
 * `Steered`, `Job result`, `Ask`, the quiet group's family plural) is copy: this
 * app's own choice, and another client says the same thing in its own catalogue
 * and its own locale. What must not drift between clients is WHICH label a span
 * earns - a span the desktop calls `Wake` and the phone calls `Peer message` is a
 * partition disagreement dressed as a translation, and no fixture can catch it
 * while the only exported answer is an English string. `labelOfSegment` is the
 * one place that turns `SegmentLabelKind` into words, and the parity fixture
 * (`scripts/generate-condense-parity.mjs`) compares the kind alone.
 */
export type SegmentLabelKind =
	/** The span holds a `user` row: the reader steered mid-run. */
	| { kind: "steered" }
	/** The span opens a wake or a monitor prompt. */
	| { kind: "wake" }
	/** The span opens a peer row or a hub message. */
	| { kind: "peer" }
	/** The span opens a job result. */
	| { kind: "job" }
	/** Otherwise nameless, and holds the agent's own questions/answers. */
	| { kind: "ask" }
	/** The span IS a quiet group: the family is the bar's word. */
	| { kind: "group"; family: QuietGroupFamily };

export function segmentLabelKind(
	records: readonly CondenseRecord[],
	cycles: readonly TurnCycle[],
	span: SegmentSpan,
	/*
	 * The quiet group this span IS, when it is one: the caller computes it (one
	 * spelling of the definition, in `turn-collapse-model.ts`) and the word here
	 * follows. Omitted for every caller with no group answer to hand - the arm
	 * simply does not fire, and the cycle opener's own label is used.
	 */
	group?: { family: QuietGroupFamily } | null,
): SegmentLabelKind | null {
	for (let i = span.from; i <= span.to; i += 1) {
		if (records[i].kind === "user") return { kind: "steered" };
	}
	/* The group's word wins over any cycle opener: the span IS the group. */
	if (group !== undefined && group !== null) {
		return { kind: "group", family: group.family };
	}
	const cycle = cycles.find(
		(candidate: TurnCycle) =>
			candidate.start <= span.to && span.to <= candidate.closeIndex,
	);
	/*
	 * A CYCLE THAT NEVER CLOSED (a follow-up that died mid-work, or is still
	 * running) has no entry in `cycles` - a cycle is defined by its close - so its
	 * opener is read the same way a cycle's initiators are: the triggers between
	 * the previous close and the first step row.
	 */
	let opener: TriggerKind | null = cycle?.initiators.at(-1) ?? null;
	if (cycle === undefined) {
		const previous = cycles.filter(
			(candidate: TurnCycle) => candidate.closeIndex < span.from,
		);
		for (
			let i = (previous.at(-1)?.closeIndex ?? -1) + 1;
			i <= span.to && records[i].kind !== "tool";
			i += 1
		) {
			opener = triggerOf(records[i]) ?? opener;
		}
	}
	switch (opener) {
		case "wake":
		case "monitor_prompt":
			return { kind: "wake" };
		case "peer":
		case "hub_message":
			return { kind: "peer" };
		case "job_result":
			return { kind: "job" };
		default:
			/*
			 * THE ASK RECEIPTS GET A NAME OF THEIR OWN.
			 *
			 * A segment made of `ask_response`/`ask_timeout` rows has no opener among
			 * the trigger kinds above, so the bar came back `null` and drew a bare rule
			 * with a chevron: the reader was shown a control with nothing to invite them
			 * in and the accessibility tree carried a button with NO accessible name,
			 * while the two receipts it held - the questions the agent asked and the
			 * answers it was given - sat behind it (design round 1, D2, measured on the
			 * rendered story).
			 *
			 * The arm is deliberately the LAST one: a segment that already reads as a
			 * wake, a peer message or a job result keeps that name, and only the
			 * otherwise-nameless case borrows this one.
			 */
			for (let i = span.from; i <= span.to; i += 1) {
				const kind = records[i].kind;
				if (kind === "ask_response" || kind === "ask_timeout")
					return { kind: "ask" };
			}
			return null;
	}
}

/** The bar's word for a label kind: this surface's own copy (see
 * `SegmentLabelKind` for why the decision is not spelled here). */
export function labelWordOf(kind: SegmentLabelKind | null): string | null {
	switch (kind?.kind) {
		case "steered":
			return "Steered";
		case "wake":
			return "Wake";
		case "peer":
			return "Peer message";
		case "job":
			return "Job result";
		case "ask":
			return "Ask";
		case "group":
			return quietGroupLabel(kind.family);
		default:
			return null;
	}
}

/** The decided label's word, in one call: the shape every renderer wants. */
export function labelOfSegment(
	records: readonly CondenseRecord[],
	cycles: readonly TurnCycle[],
	span: SegmentSpan,
	group?: { family: QuietGroupFamily } | null,
): string | null {
	return labelWordOf(segmentLabelKind(records, cycles, span, group));
}

/**
 * Did this segment RUN TO A REAL END: does its bar earn the completion mark?
 *
 * WHAT THE MARK SAYS, AND WHERE IT APPEARS (design review round 1 on #708, D4;
 * agent review R1-3). The mark says "this section finished" - the closed-disposal
 * receipt without a card. It is carried by every bar that is a SECTION of the turn
 * rather than the ordinary work under the answer: a bar after the answer, and a
 * labelled bar (`Wake`, `Peer message`, `Job result`, `Steered`) before it. The
 * ordinary unlabelled bar sits directly above the answer, and the answer IS the
 * statement that it finished, so a mark there would be constant on every condensed
 * turn in the transcript (the default state the design was approved without one)
 * and carry no information. The rule is therefore the same on both sides of the
 * answer; only the constant case is exempt.
 *
 * COMPLETION NEEDS A REAL CLOSER, AND A CYCLE THAT CLOSED. The segment's last
 * row must be a settled assistant message or a finished call -- and the segment
 * must end inside a cycle at all. `cyclesOf` builds a cycle only around a close
 * row, so a span whose last row is a bare trigger or an unfinished stretch of
 * work belongs to NO cycle: nothing in it ever handed over, and its bar wears no
 * mark whatever its last row happens to be (review round 2, MINOR-2: a wake that
 * never closed but ended on a finished tool call wore the check -- `U T A M W T`
 * -> `Wake ✓` over a section that only ever started). A span the next terminal
 * marker CUT OFF is in the same class for the same reason: it ended in that
 * state, not in success, so no green mark sits over `Stopped with an error`.
 */
export function segmentIsCompleted(
	records: readonly CondenseRecord[],
	cycles: readonly TurnCycle[],
	span: SegmentSpan,
	answerCloseIndex: number | null,
	labelled: boolean,
	/*
	 * The run's last settled quiet close, when it has one (design §5(a)): a
	 * closer that hands the reader nothing, so its bar is the only place the
	 * turn's completion can be stated. See the mark's rule below.
	 */
	quietCloseIndex: number | null = null,
): boolean {
	const closeAt = answerCloseIndex ?? quietCloseIndex;
	if (closeAt === null) return false;
	const afterClose = span.from > closeAt;
	/*
	 * THE PRE-ANSWER EXEMPTION IS THE ANSWER'S OWN: the ordinary bar directly
	 * above a VISIBLE answer needs no mark because the answer is the statement
	 * that the work finished. A QUIET close is the opposite case - its row is
	 * never painted - so a bar governed by one always states the completion it
	 * stands for, labelled or not.
	 */
	if (!afterClose && !labelled && answerCloseIndex !== null) return false;
	if (
		!cycles.some(
			(cycle) => cycle.start <= span.to && span.to <= cycle.closeIndex,
		)
	)
		return false;
	const last = records[span.to];
	const settled =
		last.kind === "assistant"
			? !last.streaming
			: last.kind === "tool"
				? last.phase === "done"
				: false;
	if (!settled) return false;
	const next = span.to + 1 < records.length ? records[span.to + 1] : null;
	return next === null || boundaryKindOf(next) !== "terminal";
}

/**
 * Everything wrong with a partition, as sentences; empty means it is sound.
 *
 * AN INVARIANT THAT AN EMPTY PLAN CANNOT SATISFY. The check is a COVER, not a
 * bounds test: every index in the span must be accounted for exactly once,
 * either visible or inside one segment. A plan that hid nothing while a tool row
 * sat in the span would leave that index uncovered and fail here, so "no
 * segments" is only ever valid for a span that has nothing to hide - the property
 * a bare "every segment is inside the span" assertion lacks, since an empty list
 * satisfies it vacuously. Tests pair this with an independent expectation of how
 * many rows the fixture hides.
 */
export function violationsOf(
	records: readonly CondenseRecord[],
	partition: RunPartition,
	options: { from: number; pinned: (record: CondenseRecord) => boolean },
): string[] {
	const problems: string[] = [];
	const owner = new Map<number, number>();
	partition.segments.forEach((segment, s) => {
		if (segment.from > segment.to) problems.push(`segment ${s} is empty`);
		/*
		 * A SEGMENT ABOVE THE SPAN (agent review round 1 on #708, R1-1). `from` is
		 * the first row a span may hold - just below the run's opening user row, or
		 * the loaded edge - and the cover loop below starts there, so a segment that
		 * swallowed the opening row was invisible to it.
		 */
		if (segment.from < options.from) {
			problems.push(
				`segment ${s} starts at row ${segment.from}, above the span's first row ${options.from}`,
			);
		}
		/*
		 * MAXIMALITY: one bar per CONTIGUOUS hidden span. A hidden row right after a
		 * segment that is not part of it means one span was split into two adjacent
		 * bars (two stacked bars with nothing between them, and a press that opens
		 * half a span). The ownership check above catches a row in TWO segments and
		 * the cover check a row in none; only this catches a row split ACROSS them.
		 */
		if (
			segment.to + 1 < records.length &&
			!partition.visible.has(segment.to + 1)
		) {
			problems.push(
				`segment ${s} is not maximal: row ${segment.to + 1} is hidden too`,
			);
		}
		if (s > 0 && segment.from <= partition.segments[s - 1].to) {
			problems.push(`segment ${s} overlaps or precedes segment ${s - 1}`);
		}
		for (let i = segment.from; i <= segment.to; i += 1) {
			if (owner.has(i)) problems.push(`row ${i} is in two segments`);
			owner.set(i, s);
			if (partition.visible.has(i)) {
				problems.push(`row ${i} is both visible and hidden`);
			}
			if (options.pinned(records[i])) {
				problems.push(`row ${i} is pinned but sits in segment ${s}`);
			}
		}
	});
	for (let i = options.from; i < records.length; i += 1) {
		if (!partition.visible.has(i) && !owner.has(i)) {
			problems.push(`row ${i} is neither visible nor in a segment`);
		}
	}
	if (partition.answer !== null) {
		const at = partition.answer.closeIndex;
		if (owner.has(at)) problems.push("the answer row is hidden");
		if (partition.answer.class !== "response") {
			problems.push("the answer cycle is not a response");
		}
	}
	if (partition.cycles.length > 0 && partition.cycles[0].class !== "response") {
		problems.push("a run's first cycle must be a response");
	}
	return problems;
}
