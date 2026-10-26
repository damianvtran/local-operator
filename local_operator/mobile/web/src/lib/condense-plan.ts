/**
 * THE COLLAPSE PLAN (Tier B), ported from the desktop.
 * Ported from `damianvtran/local-operator-ui` at sourceHash f9f4611dd3dbcc06…:src/renderer/src/features/chat/canonical/turn-collapse-model.ts:128-3335
 *
 * WHAT IT DECIDES: which rows of a settled turn stay on screen and which hide
 * behind ONE bar per contiguous span, what each bar states (duration, action
 * count and its `+` minimum, the quiet group's receipt family, the label kind),
 * which run may be live and how far `live` reaches inside it (only the in-flight
 * cycle), the settle latch across a liveness dip, the focus hold, and the head-cut
 * rule an end-loaded window inherits from the desktop.
 *
 * WHAT IT DELIBERATELY DOES NOT CARRY, and why nothing above changes as a result:
 * - the ALIGN WALK (`snapWindowToRunBoundary`, `windowTopRun*`, `widenTarget`,
 *   `alignWalk*`, `paintedRows`): those exist for the desktop's 60-row DOM window
 *   and its page-by-page walk to a run's head. This surface windows over plan
 *   ENTRIES, so a window edge never cuts a run, and head completion is a later
 *   slice's problem.
 * - the signature memo (`recordSignature`, `collapseRowsKey`,
 *   `collapsePlan*Key`, the `dbg*` counters): performance instrumentation of the
 *   desktop's render path, with no decision in it. The phone's memo key is its
 *   own concern.
 * - the media pins' image stamping (`imageStamp`, `IMAGE_STAMPS`): the relay's
 *   tool rows carry no image refs.
 * - `freezeRecordDeep`: the desktop's development-only record-immutability guard.
 * The server-fact inputs (`runFacts`/`factPage`/`hideCrossSession`) ARE ported and
 * stay dormant: no relay surface supplies them, and keeping the branch means the
 * file can be diffed against its source without a reviewer wondering which arms
 * were removed.
 */

import type { CondenseRecord } from "./condense-records";
import type { TranscriptDisplayMode } from "./condense-segments";
import { type Row, type TurnRun, isStatementRow, ledgerName, paintsSomething, runsOf } from "./condense-rows";
import {
	type QuietGroupFamily,
	type SegmentLabelKind,
	type SegmentSpan,
	type TriggerKind,
	type TurnCycle,
	boundaryKindOf,
	isQuietTurnCall,
	isQuietTurnClose,
	isTerminalMarker,
	labelWordOf,
	partitionRun,
	reportsCompletedThought,
	segmentIsCompleted,
	segmentLabelKind,
	triggerOf,
} from "./condense-segments";
import {
	type CondensePeerSender,
	type HubReceipt,
	EMPTY_SENDER,
	isImageGenTool,
	peerIdentity,
} from "./condense-predicates";
import type { FoldableAction, FoldCountSegment } from "./condense-fold";
import { foldCountSegments, foldSummary, workedSeconds, workedSecondsOf } from "./condense-fold";
export function staysVisibleWhileCollapsed(record: CondenseRecord): boolean {
	/*
	 * THE PIN LIST IS THE BOUNDARY VOCABULARY (`boundaryKindOf`): the same
	 * predicate the classifier reads to decide a turn is over, so a marker that
	 * ends a turn is always also a marker that stays on screen - the two rules
	 * used to be copies, and the `closed`/`retired` receipts were the case that
	 * could have drifted between them.
	 */
	return boundaryKindOf(record) !== null;
}

/**
 * Whether a run currently renders CONDENSED: at least one of its bars is closed.
 *
 * The supplement block condenses with its turn (memo §2.8/§2.11: "one line, no
 * frames mounted while condensed - and restores on expand"), and this is the
 * one reading of the collapse plan that fact is allowed to come from. The
 * block does NOT re-derive it: the caller passes this answer into the row
 * (design round 1, D1.1).
 *
 * `segment.collapsed` is the plan's statement that the span renders as a bar,
 * and it is deliberately independent of the reader's own expansion (an opened
 * bar still renders - its children mount when `open` is set). The reader's
 * side is the same set the bar's own `open` prop reads (`openRuns`), so a bar
 * the reader just opened is not "closed" here and the block restores in the
 * same commit the rows return: one state, two readers, no second owner.
 */
export function runIsCondensed(
	run: RunCollapsePlan,
	openRuns: ReadonlySet<string> | undefined,
	options?: { artifactOwner: string },
): boolean {
	if (options !== undefined) {
		// An ambient internal bar is not the expansion owner of an answer's
		// files/frames. Only that answer's stamped work bar can condense them;
		// an unrelated closed segment (or an uncertain head) is not permission.
		if (
			run.answerId !== options.artifactOwner ||
			run.segments.some((s) => s.internal != null)
		)
			return false;
		return run.segments.some(
			(s) =>
				s.collapsed &&
				!s.facts.partial &&
				s.stampTs !== null &&
				s.stampTs === run.stampTs &&
				!openRuns?.has(s.key),
		);
	}
	return run.segments.some(
		(segment) => segment.collapsed && !openRuns?.has(segment.key),
	);
}

/**
 * Is this tool row a genuine FAILURE?
 *
 * THE SIBLING CONTRACT (session 3d5d52c85ba2 owns interrupted-vs-failed): the
 * bar counts ONLY `error`, and aborted (`stopped`), never-sent
 * (`neverSent`/`notRunReason`) and skipped calls are excluded — a call the
 * user stopped did not fail, and a call no tool ever received has no outcome.
 * The bar shows no "interrupted" wording in v1; the aborted answer's own
 * `Stopped before finishing` caption carries that fact where it already does.
 *
 * ONE LOCAL HELPER on purpose: this is the single place the bar's count is
 * read, so the switch to a shared exported predicate (if one is ever cut) is
 * mechanical. It consumes the sibling's landed facts rather than re-deriving
 * them (their interrupted-vs-failed fix, PR #613): `isInterruptedFault` is
 * their exported class binding for the wire's `not_run_kind`, and the reducer
 * sets `stopped` / clears `isError` from `__fault` in {skipped, aborted} on
 * both the end and the durable arms - so an interrupted call is excluded by
 * the wire's own state and this helper never re-sniffs a reason string. The
 * never-sent exclusions stay even though the fold chip and foot count
 * `isError` alone: the frozen contract keeps every never-sent call out of the
 * SUMMARY counts (v1), and their rows keep their own failure treatment. The
 * planning-fault kinds (`denied`, `gate_failed`, ...) therefore stay excluded
 * here by the same never-sent contract when they arrive with a verdict.
 */
export function isFailedCall(record: CondenseRecord): boolean {
	if (record.kind !== "tool") return false;
	if (record.isError !== true) return false;
	// The `send` delivery pair: `mailbox` and `unconfirmed` are settled non-failures
	// that draw the amber partial row, so they never count here whatever a producer
	// put on `is_error`. Listed beside the never-sent exclusions below for the same
	// reason they are - this helper's answer is a COUNT's answer.
	if (record.neverSent === true) return false;
	if (record.notRunReason !== null && record.notRunReason !== undefined) {
		return false;
	}
	if (record.stopped === true) return false;
	return true;
}

/** ONE QUIET GROUP'S sender entry (design §5, rev 2): the identity line and
 * how many receipts this sender contributed. */
export type QuietGroupSender = {
	/** `peerIdentity`'s own spelling - quoted name, basename, id prefix or pid. */
	label: string;
	count: number;
};

/**
 * ONE QUIET GROUP (design §5, rev 2): a client-derived fold over a run's
 * delivery receipts. No wire kind, no capability flag, no server field - the
 * group is a PURE function of the rows on hand, and the definition is pinned
 * across surfaces by the shared parity fixture
 * (`scripts/fixtures/quiet-groups.parity.json`, copied by the native / relay /
 * TUI slices the same way `format.parity.json` is).
 *
 * WHAT IT IS. A maximal run of >= 2 TRIGGER rows (`peer`, `hub_message`,
 * `wake`, `monitor_prompt`, `job_result`) with nothing the reader can see
 * between them: a `user` row, a row that stays on screen in a collapsed run
 * (`staysVisibleWhileCollapsed` - the compaction statement, completion markers,
 * incidents) or visible assistant text splits it (`groupSplitterOf`). Tool rows
 * - the quiet call included - and rows that paint nothing sit inside it. ONE
 * receipt is not a group at all: its bar keeps the ordinary facts (design §5,
 * sub-case i).
 *
 * WHERE THE NAME LIVES: `labelOfSegment` is handed the group and states the
 * family's plural word; `TurnSummaryFacts.group` carries it to the bar. The
 * `key` is the group's stable identity (`qg:<first row id>`): rows append at
 * the tail and the key never moves, so a growing group's bar is never remounted
 * - the reader's expansion survives the append (the jitter `SegmentPlan.key`'s
 * rule exists to prevent). The newest run's tail span carries the same key
 * BEFORE it is a group (design round F3), so the transition to group moves
 * nothing either.
 *
 * FACTS FREEZE WHEN THE GROUP CLOSES (the latch, design §5): a closed group's
 * boundary is fixed - later appends land outside it - so its count, span and
 * senders cannot move again. The OPEN tail group is the one that grows: its
 * count increases as receipts arrive while the bar stays a single line.
 *
 * HEAD-CUT (the end-loaded rule): a group whose span's head is cut states NO
 * times (null - the true span is unknowable from the loaded rows) and its count
 * is a minimum, which the bar states in its own `partial` vocabulary. Facts
 * come only from rows on hand; no paging, no server field.
 */
export type QuietGroup = {
	/** `qg:<first row id>` - stable across appends (see the doc above). */
	key: string;
	/** The family the trigger rows compose; `mixed` when more than one. */
	family: QuietGroupFamily;
	/** Trigger rows in the group (>= 2 by construction). */
	count: number;
	/** First trigger's instant; null when the span's head is cut (no span to state). */
	firstTs: number | null;
	/** Last trigger's instant; null when the span's head is cut. */
	lastTs: number | null;
	/**
	 * Peer-family senders, by count: the top 2, then one aggregate entry whose
	 * label is `<N> more` for the remaining distinct senders and whose count is
	 * their receipts. Empty for every other family (the shape is peer-only).
	 */
	senders: QuietGroupSender[];
	/** Non-quiet tool rows in the group. */
	actions: number;
	/** Of `actions`, the rows whose outcome is a genuine error (`isFailedCall`). */
	failed: number;
	/** It is the tail and no later visible row exists (the group may still grow). */
	open: boolean;
	/**
	 * Hub receipts in this group that ASK for a reply, and how many: state it only
	 * when non-zero, and as a count because the bar states the number.
	 *
	 * WHY IT LIVES ON THE GROUP AT ALL. It is the phone's clause too: both surfaces
	 * derive their bars from one definition (`quietGroupClauseFacts`), and a fact
	 * only one of them can compute is a parity gap the shared fixture cannot see.
	 * Read from the record's own durable hub shape (`receipt`), never re-derived
	 * from the wire.
	 */
	replyRequested?: number;
	/** Hub receipts in this group delivered as a STEER (same shape as above). */
	steered?: number;
	/** Every row of the group, in order. */
	rowIds: string[];
};

/** The family a trigger row belongs to; null for `user`, which splits groups. */
function quietFamilyOf(trigger: TriggerKind): QuietGroupFamily | null {
	switch (trigger) {
		case "peer":
		case "hub_message":
			return "peer";
		case "wake":
			return "wake";
		case "monitor_prompt":
			return "monitor";
		case "job_result":
			return "job";
		default:
			return null;
	}
}

/**
 * Does this row SPLIT a quiet group? The design's boundary vocabulary (§5), and
 * it is the COLLAPSE's own visibility list: `staysVisibleWhileCollapsed` is
 * literally the set of rows a collapsed run still paints, so a span the plan
 * builds can never contain one - aligning the two entry points by construction
 * rather than by a second copy of the list. That is what keeps a compaction
 * statement (the memory receipt, pinned in v1) out of a group on both sides:
 * `quietGroupsOf` used to scan across one while `planRun` could never render
 * it, the disagreement the design round's F1 finding named.
 *
 * A `user` row splits even though it paints through its own clause (V3), and
 * visible assistant text splits (the response closes and V4 rows - assistant
 * rows that paint); the assistant arm states that rule directly rather than
 * leaning on the pin list, which carries no assistant kinds.
 */
function groupSplitterOf(record: CondenseRecord): boolean {
	if (record.kind === "user") return true;
	if (staysVisibleWhileCollapsed(record)) return true;
	return record.kind === "assistant" && paintsSomething(record);
}

/** A trigger row the group COUNTS: every trigger kind but `user`. */
function isQuietGroupTrigger(record: CondenseRecord): boolean {
	const trigger = triggerOf(record);
	return trigger !== null && trigger !== "user";
}

/**
 * Presentation eligibility is positive RECEIPT provenance, not a quiet outcome.
 * In particular, eval code/stdout and completion-attention ineligibility say
 * nothing about silence. An unstamped old-core hub user remains a user boundary.
 */
function isInternalEpisodeOrigin(record: CondenseRecord): boolean {
	if (record.kind === "peer" || record.kind === "wake") return true;
	if (record.kind === "notice") {
		// applyEvent(subagent_end) owns this reserved canonical-ID namespace
		// (transcript-reducer.ts); generic INFO/local notices are not job status.
		return (
			record.level === "info" &&
			record.complete !== true &&
			record.id.startsWith("subagent:") &&
			record.id.length > "subagent:".length
		);
	}
	if (record.kind !== "custom" || record.level === "error") return false;
	const trigger = triggerOf(record);
	if (trigger === null || trigger === "user") return false;
	if (trigger === "hub_message") {
		// Reply/steer intent is already normalized once by the receipt producer.
		// Unknown historical intent stays visible; never recover it from prose.
		return (
			record.receipt !== undefined &&
			(record.receipt.direction === "to_parent" ||
				record.receipt.direction === "to_child") &&
			record.receipt.expectsReply === false &&
			record.receipt.steer === false
		);
	}
	return true;
}

/** Both turn and action folds must retain actual material owners inline. */
export function ownsMaterialArtifact(record: CondenseRecord): boolean {
	/*
	 * Two desktop arms are absent here: a `supplement` row and record-level
	 * `images` are kinds this wire never carries (relay tool rows have no image
	 * refs; only user and hub rows do, and those are not material owners). The
	 * generating-card arm below IS kept - a phone renders those cards too.
	 */
	// Before output bytes exist, the generating card already owns loading/error
	// and cancellation UI; keep ALL phases, not only successful picture rows.
	return record.kind === "tool" && isImageGenTool(record.toolName);
}

function internalEpisodeBoundary(record: CondenseRecord): boolean {
	// Empty/streaming assistants are future user text. Pin before the first
	// token, not only once prose happens to paint; otherwise the first token
	// would briefly mount inside a closed disclosure (the 0.27 regression).
	if (
		record.kind === "user" ||
		record.kind === "ask_response" ||
		record.kind === "ask_timeout"
	)
		return true;
	/*
	 * A SETTLED assistant row is prose the existing visibility rule already owns:
	 * a commentary close folds with the receipt that prompted it (the wake's own
	 * span, `narration-only in-betweens`). Only an EMPTY or still-streaming one is
	 * FUTURE text - pin it BEFORE the first token, or that token would briefly
	 * mount inside a closed disclosure (the 0.27 regression). Pinning every
	 * assistant row instead tore settled prose out of condensing spans.
	 */
	if (record.kind === "assistant")
		return record.text === "" || record.streaming === true;
	// FoldMedia is picture-only. Retain the actual artifact row/player inline,
	// including videos, rather than substitute a thumbnail or hide its payload.
	if (ownsMaterialArtifact(record)) return true;
	if (record.kind === "notice" || record.kind === "custom")
		return !isInternalEpisodeOrigin(record);
	if (record.kind === "tool")
		return (
			record.toolName === "ask" ||
			isFailedCall(record) ||
			record.stopped === true ||
			record.neverSent === true ||
			(record.notRunReason !== undefined && record.notRunReason !== null)
		);
	return staysVisibleWhileCollapsed(record);
}

/** Protect only receipt-context rows; ordinary human work keeps its old policy. */
function internalEpisodePins(records: readonly CondenseRecord[]): {
	pins: ReadonlySet<CondenseRecord>;
} {
	const pins = new Set<CondenseRecord>();
	let receiptContext = false;
	for (let i = 0; i < records.length; i += 1) {
		const record = records[i];
		if (isInternalEpisodeOrigin(record)) receiptContext = true;
		if (receiptContext && internalEpisodeBoundary(record)) {
			pins.add(record);
			receiptContext = false;
		}
		if (record.kind === "user") receiptContext = false;
	}
	/*
	 * A MATERIAL OWNER IN A RUN THAT CARRIES AN EPISODE stays inline, wherever it
	 * sits relative to the receipt: pictures, video and owned files are not the
	 * episode's private work to hide, and the bar's strip substitutes a thumbnail
	 * for a row that is unmounted. A run with NO episode keeps the base folding -
	 * its pictures ride its own bar and its strip - which is what six ruled
	 * design/UX tests pin.
	 */
	/*
	 * A MATERIAL OWNER IN A RUN THAT CARRIES AN EPISODE stays inline - but only
	 * from the span the episode lives in, not from the whole run.
	 *
	 * WHY THE LOWER BOUND IS THE LAST SETTLED ANSWER BEFORE THE FIRST ORIGIN.
	 * Rows before that answer belong to a bar that SETTLED on its own: pinning
	 * them reached backwards and tore the picture out of its own bar (base holds
	 * `seg:a1[media1] collapsed=true`; the run-wide pin dropped that bar and
	 * painted the media loose, losing its strip and re-keying `seg:a1` to
	 * `seg:i1`). Rows AFTER that answer and before the first receipt are the
	 * episode's own span - a picture there is exactly what the ruled inline
	 * behaviour protects ("produced image and video rows stay inline outside
	 * internal folds", `where === "before"`), so the bound is the answer and not
	 * the first receipt. With no settled answer before the origin there is no
	 * settled bar to protect and the bound is the run's start.
	 */
	const firstOrigin = records.findIndex((record) =>
		isInternalEpisodeOrigin(record),
	);
	if (firstOrigin >= 0) {
		let lowerBound = 0;
		for (let i = firstOrigin - 1; i >= 0; i -= 1) {
			const before = records[i];
			if (before.kind === "assistant" && before.streaming !== true) {
				lowerBound = i + 1;
				break;
			}
		}
		for (let i = lowerBound; i < records.length; i += 1) {
			const record = records[i];
			if (ownsMaterialArtifact(record)) pins.add(record);
		}
	}
	return { pins };
}

export type InternalEpisodeFacts = {
	/** Counts physical canonical receipts, never labels or guessed job IDs. */
	receipts: number;
	/** Null at a loaded-head cut: no invented off-page start or elapsed time. */
	firstTs: number | null;
	/**
	 * Ongoing presentation, NOT successful/completed lifecycle. For the newest
	 * WATCHED run `planRun` reads the pane's flag through the settle latch's
	 * `anchored` (`live || watched`), so a liveness dip cannot flip it.
	 */
	live: boolean;
	/**
	 * The episode sits in the cycle STILL BEING WRITTEN: the pane reads live
	 * (for the newest watched run, through the settle latch's `anchored`) and the
	 * span reaches `liveFrom`. Unlike `live` it carries no tail-position
	 * condition, so it stays true when a streaming answer appends after the
	 * episode. The settle announcement reads this, not `live` and not mere
	 * membership: `live` flips false on that append (a wake re-opening a settled
	 * turn would announce mid-cycle), while membership alone also silences every
	 * SETTLED history episode during a live pass, so the first settle re-announced
	 * old bars (reviewer F2: `['qg:w1','seg:a5']`).
	 */
	unsettled: boolean;
};

function internalEpisodeOf(
	records: readonly CondenseRecord[],
	span: SegmentSpan,
	headLoaded: boolean,
	live: boolean,
	unsettled: boolean,
	humanPending: boolean,
	olderRowsExist: boolean,
): InternalEpisodeFacts | null {
	if (!isInternalEpisodeOrigin(records[span.from])) return null;

	/*
	 * AN INTERNAL EPISODE MAY COLLAPSE ONLY ON WORK THAT HAS ALREADY SETTLED.
	 *
	 * THE RULE, in two arms - either is enough:
	 *   (a) an ANSWER precedes the episode in this run: the episode is a
	 *       continuation of finished work, so folding it does not take content
	 *       away from the reader, it tidies what follows an answer they can
	 *       still read; or
	 *   (b) the episode holds >=2 receipt rows: it is a genuine run of
	 *       exchanges, and a run is what the condensing was built to batch.
	 *
	 * WHY IT IS NEEDED. Eligibility alone classified the origin row and then
	 * condensed it wherever it sat. That over-reaches in both directions QA
	 * round 1 measured: a receipt sitting BESIDE a sibling but with no settled
	 * work before it (`[job_result, tool]` - Q1), and a LONE receipt of any kind
	 * (peer, wake, `subagent:` notice - Q2). In those shapes the episode IS the
	 * run's opening content, so the bar replaces the reader's only content: a
	 * failed background job's verdict, a peer's request, a wake's prompt.
	 * `job_result` failures carry `level: "info"` (see
	 * `canonical-notice.stories.tsx`), so the error-level exemption does not
	 * cover them and the bar label does not say which way the job went.
	 *
	 * WHY IT IS NOT A CUSTOMTYPE EXEMPTION. `isInternalEpisodeOrigin` keeps
	 * `job_result` and `monitor_prompt` as origins ON PURPOSE - they are the
	 * agent finishing something it owns, which is why `turn-segments.ts` carries
	 * both in `CONTINUATION_CUSTOMS`, and the delivery-family suite asserts each
	 * does open a condensed live episode. An episode with >=2 receipts, or one
	 * that follows an answer, must still fold. The two row families are
	 * otherwise field-identical (`kind`, `customType`, `level`, `attribution`,
	 * both carrying a headline), so any type- or field-based rule would either
	 * miss these shapes or fold a real job receipt.
	 *
	 * THE CHECK IS POSITIVE AND STRUCTURAL. It reads settled answers and counted
	 * receipts - never the row's text, and never a "which producer was this"
	 * inference, so an unstamped old-core row is not guessed about.
	 */
	/*
	 * SEEDED FROM THE PAGING STATE AT THE LOADED EDGE, and only there.
	 *
	 * At a head cut the answer this episode continues is off-page, so the loop
	 * below cannot see it. When older rows are known to exist (`olderRowsExist`)
	 * the settled work is real but unloaded: assuming it keeps a head-cut episode
	 * condensed instead of flipping it to loose rows the moment a page cut drops
	 * the answer. `!headLoaded` is what stops this widening past the edge - a
	 * span inside the page has its whole run behind it on hand and is decided by
	 * the loop alone. With nothing older to arrive the receipt genuinely opens the
	 * run and still paints (QA Q1/Q2, reviewer F1).
	 *
	 * `&& unsettled` (agent review round 2, R2-2). The seed exists for a LIVE
	 * span whose answer is off-page. A SETTLED cut span - `[peer, 4 tools,
	 * answer]` - condenses as an ordinary bar whether or not older rows exist, so
	 * seeding it only changed what it IS: an episode keyed `qg:p` while the head
	 * is missing, an ordinary bar keyed `seg:a` once the head lands. The key then
	 * changed under the reader: their open disclosure closed and the new bar was
	 * announced, where base stays open and silent. Scoping to the unsettled span
	 * leaves the settled shape identical before and after the head arrives.
	 *
	 * RESIDUAL, stated plainly: at a true loaded edge with older rows and a lone
	 * receipt the reader sees a bar until the head lands. The completion walk
	 * normally turns it back into rows, but a blocked walk (page budget, off-tail
	 * reader) leaves it behind a bar until opened, and the bar's label cannot say
	 * whether a failed `job_result` is inside.
	 */
	/*
	 * R3-2 policy clarification: older rows prove a cut, not a preceding settled
	 * answer. Only the unsettled edge gets this temporary presentation allowance.
	 * A settled, answerless `[peer, tools]` fragment intentionally paints loose;
	 * once its preceding answer actually loads, the loop below can establish the
	 * continuation and fold it. That is a narrower policy than the original seed,
	 * not evidence that an unseen answer exists. Do not seed settled spans merely
	 * to retain their old fold: that also re-keys settled answer-bearing spans on
	 * head landing (R2-2), closing open disclosures and re-announcing history.
	 */
	let settledBefore = !headLoaded && olderRowsExist && unsettled;
	for (let i = 0; i < span.from; i += 1) {
		const before = records[i];
		if (before.kind === "assistant" && before.streaming !== true) {
			settledBefore = true;
			break;
		}
	}
	// V4/typed quiet closure was derived once for the loaded run. Neither a
	// toolUse lead-in nor eval text can make unanswered human work internal.
	let receipts = 0;
	let quietClose = false;
	for (let i = span.from; i <= span.to; i += 1) {
		if (internalEpisodeBoundary(records[i])) return null;
		if (isInternalEpisodeOrigin(records[i])) receipts += 1;
		if (isQuietTurnClose(records[i])) quietClose = true;
	}
	if (humanPending && !quietClose) return null;
	/*
	 * THE RULE'S VERDICT. An episode may condense only on settled work: an
	 * answer already precedes it, or it is a run of >=2 receipts. Without either,
	 * this episode opens the run's content and must paint - whatever its
	 * customType, and whatever kind of receipt opened it. This subsumes the
	 * lone-row case (a transcript with no preceding answer has none), so the
	 * earlier `records.length === 1 && kind === "custom"` guard is gone: it was
	 * two narrower tests where one general one was needed.
	 */
	if (!settledBefore && receipts < 2) return null;
	return {
		receipts,
		firstTs: headLoaded ? (records[span.from].ts ?? null) : null,
		live,
		unsettled,
	};
}

/**
 * The identity a peer receipt contributes to a group's sender summary.
 *
 * `peerIdentity` reads a FULL `CondensePeerSender` (the reducer's `peerFields` shape,
 * every field a string). Hand-built rows - the model suites' fixtures, the
 * per-pass sweep's mutated records - can carry a partial sender or none at all,
 * and the derivation must stay TOTAL over the record space rather than take the
 * plan down: missing fields fill with the ladder's own empties, which lands a
 * senderless receipt on `another session` exactly as its row would.
 */
function quietSenderOf(record: { sender?: CondensePeerSender }): string {
	return peerIdentity({ ...EMPTY_SENDER, ...(record.sender ?? {}) });
}

/**
 * The quiet group a single span IS, or null when the span is not one.
 *
 * THE SPAN MUST BE THE WHOLE GROUP (design §5): its neighbours must be
 * splitters or the list's edges (`quietGroupsOf` builds spans that way), and no
 * splitter may sit inside. A span that merely OVERLAPS a group - a caller that
 * sliced inside one, e.g. a window starting mid-run - refuses here instead of
 * stating a count over part of something: the bars degrade to their per-cycle
 * labels, which is the safe direction. Pinned statement rows (the compaction
 * receipt, completion markers, incidents) are SPLITTERS now, so a span holding
 * one is refused on both entry points rather than by the scanner alone.
 *
 * `spanHeadLoaded` (default true) is the caller's own head-cut verdict for this
 * span (see `planRun`): false nulls the times and leaves the count a minimum.
 * `open` (default: the span reaches the list's end) is the tail fact the
 * growing-group rule reads; `planRun` refines it for a run that is not the
 * conversation's last.
 */
export function quietGroupOfSegment(
	records: readonly CondenseRecord[],
	span: SegmentSpan,
	options: {
		spanHeadLoaded?: boolean;
		open?: boolean;
	} = {},
): QuietGroup | null {
	const headLoaded = options.spanHeadLoaded ?? true;
	for (let i = span.from; i <= span.to; i += 1) {
		if (groupSplitterOf(records[i])) return null;
	}
	if (span.from > 0 && !groupSplitterOf(records[span.from - 1])) return null;
	if (span.to + 1 < records.length && !groupSplitterOf(records[span.to + 1])) {
		return null;
	}
	const triggers: CondenseRecord[] = [];
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
		if (record.kind === "peer") {
			const label = quietSenderOf(record);
			senderCounts.set(label, (senderCounts.get(label) ?? 0) + 1);
		}
	}
	let family: QuietGroupFamily | null = null;
	let mixed = false;
	for (const trigger of triggers) {
		const triggerFamily = quietFamilyOf(triggerOf(trigger) as TriggerKind);
		if (triggerFamily === null) continue;
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
	/*
	 * THE RECEIPT FAMILIES' TWO DECIDED FACTS, from the trigger rows' own durable
	 * hub shape. SPARSE, like the wire: a group with none carries no field, so a
	 * surface that has none states no clause and the two cannot drift by presence.
	 */
	const receipts = triggers.filter(
		(trigger): trigger is CondenseRecord & { receipt?: HubReceipt } =>
			"receipt" in trigger && trigger.receipt !== undefined,
	);
	const replyRequested = receipts.filter(
		(trigger) => trigger.receipt?.expectsReply === true,
	).length;
	const steered = receipts.filter(
		(trigger) => trigger.receipt?.steer === true,
	).length;
	return {
		key: `qg:${records[span.from].id}`,
		family: familyFinal,
		count: triggers.length,
		firstTs: headLoaded ? (first.ts ?? null) : null,
		lastTs: headLoaded ? (last.ts ?? null) : null,
		senders,
		actions,
		failed,
		open: options.open ?? span.to === records.length - 1,
		rowIds,
		...(replyRequested > 0 ? { replyRequested } : {}),
		...(steered > 0 ? { steered } : {}),
	};
}

/**
 * THE SUMMARY CLAUSE FACTS a bar states, in order, as the shared contract.
 *
 * WHY THIS IS A SHARED DERIVATION, AND WHY IT LIVES HERE NOW. The clause set is the
 * only part of a group a reader actually reads, and it is derived (never sent).
 * While each surface decided its own conditions, the SAME group read
 * `messages · 2 · 3 actions` on one client and bare `messages · 2` on another, and
 * nothing caught it: the parity fixture pinned counts but no clause. The relay and
 * the TUI ported this rule on 2026-10-16 while this app kept its own - so the
 * fixture copies drifted apart and both suites stayed green for nine days. This is
 * the definition the fixture now pins on both sides.
 *
 * The facts are clause IDENTIFIERS with their counts, not rendered text: a surface
 * still owns its words and formatting, exactly as it owns the family word
 * (`quietGroupLabel`). Count semantics match the group fields: `count` is TRIGGER
 * rows, `actions` is non-quiet work rows.
 */
export function quietGroupClauseFacts(
	group: QuietGroup,
	options: { partial?: boolean } = {},
): string[] {
	const facts = [
		`count:${options.partial ? `${group.count}+` : `${group.count}`}`,
	];
	/*
	 * NONZERO, NOT A THRESHOLD: a group with ANY work states it, and a workless
	 * group states no actions clause at all (never `0 actions`).
	 */
	if (group.actions) facts.push(`actions:${group.actions}`);
	if (group.replyRequested)
		facts.push(`reply_requested:${group.replyRequested}`);
	if (group.steered) facts.push(`steered:${group.steered}`);
	return facts;
}

/**
 * Every quiet group in a records list, in order: the stretches between
 * splitters, each scored by `quietGroupOfSegment` (so a stretch with fewer than
 * two triggers yields none). This is the DEFINITION-level entry the shared
 * parity fixture drives; the UI's own planning asks per segment, which is where
 * the head-cut and tail facts are known.
 */
export function quietGroupsOf(
	records: readonly CondenseRecord[],
): QuietGroup[] {
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

/**
 * The bar's facts, in the order a reader needs them.
 *
 * `durationS` is the WORKED time (`workedSeconds`, `trace-fold-model.ts`): the
 * seconds the span's own calls reported, summed - the SAME quantity the turn's
 * foot states as `Worked for ...`. It used to be a wall span (opening row to the
 * latest end instant), and a ladder of bars beside the foot showed two different
 * quantities 16.7x apart on the operator-shaped journal, with action counts that
 * reconciled exactly inviting the reader to reconcile the durations too (design
 * review round 1 on #708, D1). `Took` (bar) and `Worked for` (foot) are one
 * quantity in two places: the pre-answer bars sum to the foot within rounding.
 * A span whose calls reported nothing, or whose head is not loaded, states none.
 */
export type TurnSummaryFacts = {
	durationS: number | null;
	/** Tool rows in the run: the same unit as the fold's summary and the foot. */
	actions: number;
	/**
	 * Whether `actions` is a MINIMUM rather than the run's count — its opening row
	 * is not in the store, so the rows above the loaded span are still unknown.
	 *
	 * WHY A FACT AND NOT A DERIVATION AT THE BAR (design round 1, D1). The bar is
	 * the turn's only size statement, and a head-cut run's count read exactly like
	 * a complete one: the operator's own screenshot said `97 actions` for a turn of
	 * 423 and nothing on the line distinguished it from a settled total. A run can
	 * stay head-cut for good (one taller than the walk's allowance, or a backend
	 * with the pages gone), so the honest marker has to be part of the statement
	 * the model makes — `planRun` is where `opensWithUserRow` is known, and the bar
	 * renders `N+` and states the same in words.
	 */
	partial: boolean;
	/** Tool rows whose outcome is a genuine error (see `isFailedCall`). */
	failed: number;
	/** The first failed row, for the failure control's jump. */
	firstFailedId: string | null;
	/** The fold-style class sentence (`foldSummary`), or null with no actions. */
	title: string | null;
	/**
	 * The same summary as STRUCTURED count segments (`FoldCountSegment`), beside
	 * the words - the cross-surface fact a parity fixture compares (see
	 * `FoldSummarySpec.countUnits` for why English cannot be the contract).
	 */
	countUnits: FoldCountSegment[];
	/**
	 * The quiet group this span IS, when it is one (design §5, rev 2): the bar over
	 * a run of delivery receipts states the family's plural word, the count and the
	 * receipt span instead of the ordinary duration/action clauses — see
	 * `QuietGroup` for the facts and `quietGroupOfSegment` for when a span is one.
	 * Null for every ordinary span, and for a quiet span below the group minimum
	 * (one receipt is not a group; its bar keeps the ordinary facts).
	 *
	 * THE RUN-LEVEL `facts` (the turn's own totals on `RunCollapsePlan`) keep this
	 * null: a group is a statement about ONE SPAN's rows, and only the span knows
	 * its own boundary. The suites that read the run-level facts read the turn's
	 * arithmetic; the group's facts live on the segment the bar stands for.
	 */
	group: QuietGroup | null;
};

/**
 * One run's facts AS THE SERVER STATES THEM, in this model's own vocabulary.
 *
 * THE MODEL'S INPUT SHAPE, deliberately not the wire's (`DesktopOpenFrameRun`),
 * and the adapter that reads the wire is `open-frame.ts`: this module must not
 * grow a second opinion about what `runs_state` means or about which runs carry
 * counts, so a fact arrives here already filtered to a SETTLED run with a whole
 * count, and a caller with no facts passes none at all.
 *
 * WHY IT EXISTS. Condensation is a pure function of the LOADED rows, so a run
 * whose head lies above the page condenses from the loaded span: its bar states
 * a minimum count and no duration, and the pane spends the next several hundred
 * milliseconds (up to `ALIGN_WALK_MAX_PAGES` serial `/history` reads) growing the
 * bar one page at a time. The operator's own case, recorded in PR #702's body:
 * `30 actions` at open, `Took 2h23m · 423 actions` thirteen pages later. A fact
 * is that same turn's total measured from the whole journal, so the bar can be
 * right on the frame the reader first sees it.
 */
export type RunFact = {
	/**
	 * Tool rows in the run: the same unit as `TurnSummaryFacts.actions`.
	 *
	 * EXACT when `complete`, a LOWER BOUND when not - the run held a row body the
	 * index had to drop, and a pathological row must never be reported as an
	 * exact count. The bar's `+` is `TurnSummaryFacts.partial`'s job either way.
	 */
	actions: number;
	/** The run's reported work (`duration_s`, summed), or null for none. */
	workedSeconds: number | null;
	/** Tool rows whose outcome is a genuine error, by the client's own predicate. */
	failed: number;
	/** False: `actions` is a lower bound (see above). */
	complete: boolean;
	/**
	 * The run's opening USER row, as the wire names it, or null for a run that
	 * opens off a non-user row.
	 *
	 * THE MODEL'S ONLY WAY TO TELL "the span in hand IS the run" FROM "the span
	 * in hand is a fragment", and it is load-bearing for a case the row list
	 * cannot express on its own: a page can begin at a STEER. A steer is an
	 * ordinary user row on the wire, the client's own partition folds it into the
	 * run it interrupted (`walkTurns`), and a page whose oldest kept row is that
	 * steer therefore looks to this model exactly like a run that opens with its
	 * own user row - while the run's real head, and every row of work above the
	 * steer, are off-page. `docs/DESKTOP_API.md` states the consequence directly:
	 * "`head_cut: false` does not mean 'every run on the page is whole' either".
	 * This field is what the model compares against the row its span opens at.
	 */
	openingUserId: string | null;
	/**
	 * The run's elected answer, or null while it has none - the run's OTHER
	 * stable identity, and the one a re-keyed run is still recognisable by
	 * (agent review round 1, F3). See `factForRun`.
	 */
	closingAnswerId: string | null;
	/**
	 * Tool rows inside this run that the reader HIDES with `hide_cross_session`
	 * (a `send`, a peer receipt), or null when the backend did not state the
	 * split.
	 *
	 * WHY THE SERVER HAS TO STATE IT (agent review round 1, F2): `action_count`
	 * counts every tool row in the run, while the plan runs over
	 * `visibleRecords(records, hide)` - so with the setting on, the fact's total
	 * and the rows the bar hides are different populations, and subtracting one
	 * from the other put the hidden `send` calls back into the bar (measured: `3
	 * actions` where the visible span holds `2`). The subtrahend is the server's
	 * to state because only it counts the rows the client never receives.
	 */
	crossSessionActions: number | null;
	/**
	 * The hidden rows' worked seconds, when the wire states them. Used only to
	 * keep the duration honest under `hide_cross_session`; when it is absent and
	 * `crossSessionActions` is non-zero the duration cannot be split, and a
	 * fragment's `Took` would silently include work the bar does not show.
	 */
	crossSessionWorkedSeconds: number | null;
};

/** The facts an open-frame page carries, by `run_key` (`TurnRun.key`). */
export type RunFactLookup = ReadonlyMap<string, RunFact>;

/**
 * The page the facts came with, by IDENTITY: the row ids it carried and the row
 * it ended on (agent review round 3, B2).
 *
 * WHY THE PLAN NEEDS IT. A fact is a SNAPSHOT of a run as the page saw it: the rows
 * the page carried, and rows OLDER than its head, are inside its figure - but a row
 * that arrived AFTER the page was published is not, and the client keeps such rows
 * while the fact stays where it was (a live wake, a job result, a peer receipt on an
 * open pane). Left alone, that states an old total as final: measured on a
 * real-core page, a run whose own figure moved `30 / 60 s` → `33 / 63 s` still
 * painted `30 / 60 s` with `partial: false` and the walk retired.
 *
 * IDENTITY, NOT A CLOCK: a wake that landed BEFORE the read is already inside the
 * wire's figure, and a clock-based rule double-counted exactly that case (round 2's
 * B1). See `factAppendedRows` for the test this feeds.
 */
export type FactPageIdentity = {
	/**
	 * Every ROW KEY the page carried, as `entryRecordKey` spells them - not the
	 * page's raw entry ids (agent review round 4, B3): a tool entry becomes
	 * `tool:<tool_call_id>` and a completion marker becomes its anchor, so a set of
	 * entry ids matches no tool row at all and every page-carried tool row reads as
	 * one that arrived later.
	 */
	ids: ReadonlySet<string>;
	/** The newest carried row's key, or null when the page names none. */
	newestKey: string | null;
};

/**
 * One SEGMENT of a run: a maximal contiguous span of hidden rows and the one bar
 * that stands in for it (issue #665).
 *
 * A run's hidden rows used to be ONE bar taking the first hidden row's slot, so a
 * pinned row (a compaction, a stop marker, an incident) between two hidden spans
 * rendered AFTER a bar that preceded it and jumped when the bar opened. Each span
 * is now its own bar in its own slot, in document order, and the pinned rows keep
 * the slots they had - expanding any bar reorders nothing.
 */
export type SegmentPlan = {
	/**
	 * The reader's expansion key for THIS bar (the value `openRuns` holds).
	 *
	 * A KEY NAMES ONE SPAN FOR THE LIFE OF THE CONVERSATION, so it is derived from
	 * the span's OWN anchor row and from nothing else - not from the run's key, and
	 * not from the span's position among its neighbours.
	 *
	 * Two things used to break that, and the second is the operator's jitter:
	 *
	 * - the span nearest the answer inherited the bare run key. A wake after the
	 *   answer makes a NEW span the nearest one, which took the bare key, so a
	 *   reader's stored expansion of the first bar named the second;
	 * - every key was prefixed with `TurnRun.key`, which is the closing answer's id
	 *   (or, while a turn is being written and has none, its LAST row's id). A wake
	 *   reply moves the answer and a streamed row moves the last row, so every bar
	 *   of the run was renamed - its React element remounted and its stored
	 *   expansion dropped - at exactly the moment the agent started thinking again.
	 *
	 * WHICH ROW ANCHORS A SPAN depends on the side of the answer, because rows
	 * only ever arrive at two places: ABOVE the loaded head (a page landing) and
	 * at the TAIL (a live turn). The anchor must be a row that neither of those
	 * moves, and that survives rows INSIDE the span appearing or disappearing (the
	 * cross-session filter hides a receipt and a send row from the span the
	 * moment the setting flips, which moves the span's own last row):
	 *
	 * - a span BEFORE the answer can grow upward when a page lands - the head-cut
	 *   span is exactly that - so its FIRST row is not stable. What ends it is a
	 *   VISIBLE row (a span is a maximal hidden stretch, so a visible row follows
	 *   it), and that row is anchored in place: `seg:<the visible row after it>`.
	 *   A span with nothing after it (an answerless run's tail, which is either
	 *   still in flight - so not a bar - or idle) falls back to its own last row;
	 * - a span AFTER the answer grows at the tail while the follow-up is being
	 *   written and never upward, so its FIRST row is the stable one: `seg:<its
	 *   first row>`;
	 * - the ONLY span that can receive APPENDED rows is the one ending at the
	 *   newest run's end (live rows land at the bottom), and that span is named
	 *   for its FIRST row (`qg:<that row>`: the quiet-group key, design round F3)
	 *   whether or not it is a group yet. Rows append below it, so the first row
	 *   never moves, and the same key serves the below-minimum span and the group
	 *   it grows into - the reader's expansion survives the transition, its React
	 *   element is not remounted, and the settle announcement does not fire a
	 *   second time for the same bar. `quietGroupOfSegment` names its group's key
	 *   from that same row, so the two spellings cannot drift apart. A span that
	 *   can grow UPWARD (a page landing above the loaded head) keeps the
	 *   visible-anchor rule, because its first row is exactly what moves - and a
	 *   mid-run GROUP keeps `seg:<visible anchor>` as its SEGMENT key for the same
	 *   reason, even though its group key (the cross-client identity in
	 *   `QuietGroup.key`) names its first row.
	 *
	 * A row is in exactly one span, so two spans can never share a key. A key of an
	 * older shape (the run key, or `<run key>#<row>`) never starts `seg:`, so it
	 * names nothing: the bar renders collapsed, which is the safe failure for a
	 * stored expansion whose bar is gone. Expansion state is memory-only
	 * (`turn-collapse-open.ts`), so a new shape simply renders every bar collapsed
	 * once - there is never a key to migrate.
	 *
	 * Ordinary human work still waits for settlement. Positive internal receipt
	 * spans can already be bars while live; their qg identity survives closure.
	 * Loaded-head key transfer is bounded and requires proved canonical overlap.
	 */
	key: string;
	/** The hidden rows, in order. */
	rows: Row[];
	/** `rows[0]`'s id: the bar's slot and its `data-record-id`. */
	firstId: string;
	/** This segment's ids (the bar's `data-run-ids`, so the reveal opens the right bar). */
	segmentIds: string[];
	/** The bar's margin tier. See `segmentGap`. */
	gap: Row["gap"];
	/** After the answer: a follow-up section (commentary), not work towards it. */
	afterAnswer: boolean;
	/** The word ahead of the clauses (`Wake`, `Peer message`, `Steered`, ...), or null. */
	label: string | null;
	/**
	 * The label's DECISION, before this surface's word for it. Shared with the
	 * other clients' parity fixture, which must not compare English
	 * (`SegmentLabelKind`): the phone says `Wake` in its own catalogue, and a span
	 * one client labels and the other does not is a partition disagreement that no
	 * string comparison can see.
	 */
	labelKind: SegmentLabelKind | null;
	/** A settled follow-up: the bar carries the completion mark. */
	completed: boolean;
	/** Receipt-origin presentation facts; separate from shared QuietGroup >=2. */
	internal?: InternalEpisodeFacts | null;
	/**
	 * Whether this bar renders (the run condenses AND the focus hold did not stand
	 * this span open). False means its rows are simply drawn in place.
	 */
	collapsed: boolean;
	/**
	 * The turn's ONE stamp, when this bar is the one that carries it: the answer's
	 * instant, on the run's SOLE pre-answer bar (a bar that states the turn's whole
	 * work - the shipped shape, where the bar replaces the answer's foot). With
	 * several pre-answer bars (a pinned row splits the work) no bar states the
	 * turn's totals, so none stamps and the answer's foot keeps the totals and the
	 * stamp. A follow-up bar never stamps - the answer is dated once.
	 */
	stampTs: number | null;
	facts: TurnSummaryFacts;
};

/**
 * One run's plan: what its bars say, and what they hide.
 */
export type RunCollapsePlan = {
	/** The run's stable identity: its closing answer's record id (else its last row's). */
	key: string;
	/** The partition span, in `visible`-row indices. */
	run: TurnRun;
	/** Whether ANY bar renders for this run. */
	collapses: boolean;
	/** Every row of the run, in order. */
	recordIds: string[];
	/**
	 * The rows the bars hide (all segments), in order. Empty when nothing is hidden.
	 *
	 * KEPT, NOT CONSUMED BY `src/` (agent review round 1 on #708, R1-4): the renderer
	 * reads `segments[*]`, `recordIds` and `segment.stampTs`. `hidden`, `gap`,
	 * `answerId` and `stampTs` are the run-level restatement the TWO MODEL SUITES
	 * (`turn-collapse-model.test.mjs`, `turn-segments.test.mjs`) read as the run's
	 * whole-turn contract - 32 property reads written against the pre-segments plan
	 * that still state the turn, not a bar (agent review round 2, NIT-1: no
	 * evidence rig reads a `collapsePlan` result). They are derived from `segments`
	 * in this one function and never independently, so they cannot drift; deleting
	 * them would move those assertions from "the turn" to "flatten the bars" for no
	 * consumer's benefit. Do not add a `src/` reader without asking whether it
	 * wants a bar's value instead.
	 */
	hidden: Row[];
	/** The first bar's margin tier (kept for the same reason as `hidden`). */
	gap: Row["gap"];
	/**
	 * The TURN's facts: what the run did up to and including its ELECTED ANSWER.
	 * Commentary after the answer is not the turn's work, so a follow-up's tool
	 * calls are counted on their own bar and never in this figure - which is what
	 * made the operator's `97 actions` state a total that mixed the answer's work
	 * with the post-dispose chatter.
	 */
	facts: TurnSummaryFacts;
	/** The elected answer's record id, or null when the run handed none over (kept for the suites, see `hidden`). */
	answerId: string | null;
	/** The answer's instant, for the turn's one stamp; null when none (kept for the suites, see `hidden`). */
	stampTs: number | null;
	/** Ordered, disjoint bars (see `SegmentPlan`). */
	segments: SegmentPlan[];
	/**
	 * Whether a bar of this run took its figures from the server's fact.
	 *
	 * THE WALK'S OWN GATE, and the reason it lives on the plan (agent review
	 * round 1, F5): the walk exists to make a fragment bar whole, so it must stand
	 * down exactly when the bar IS whole - and "the map has a fact for this run"
	 * is a different question from "this plan could use it". A reader with
	 * `hide_cross_session` on and no split from the backend, a fact the model
	 * refuses, a run whose loaded edge is a pinned row rather than a fragment:
	 * each leaves the fact UNUSED, and a walk retired on the map alone would leave
	 * that bar stating `N+` with nothing left to complete it. One predicate, read
	 * by both, is the only way they cannot disagree.
	 */
	factApplied: boolean;
};

export type CollapsePlan = {
	runs: RunCollapsePlan[];
};

/**
 * Plan the collapse over a visible row list.
 *
 * PENDING QUESTION IS NOT A SETTLED TURN (design review round 1, D3). `live`
 * is the newest run's UNSETTLEDNESS, and the caller composes it from both
 * halves of that fact: the working line's predicate (a turn being written) and
 * the reader gate (a turn PARKED on a question). The gate half exists because
 * the working line stands down while a question is pending - that stand-down is
 * deliberate (`WorkingLine` yields to the dock) - so a rule that read only the
 * working line condensed a parked turn, then un-condensed it when the call
 * resumed: a bar appearing and vanishing with no reader action, stating `Took`
 * and counts of a turn that had handed over no answer (the live rig's
 * `parked on the approval` note).
 *
 * WHAT STAYS PUT is decided by the segments module (`partitionRun`): the pinned
 * rows, the ELECTED ANSWER and the run's trailing statements. Everything else in
 * the span hides, as one bar per contiguous span. For a head-cut run the first
 * span starts at the run's first LOADED row, which is exactly what makes an
 * end-loaded bar describe only rows on hand.
 *
 * LIVENESS IS THE IN-FLIGHT CYCLE'S, NOT THE RUN'S (operator report, 2026-10-01:
 * "messages are condensed, and then if a peer message or job completes and the
 * agent goes into thinking, the last condensed sequence suddenly un-condenses").
 * A wake / peer / job result does not open a run - only a `user` row does - so it
 * RE-OPENS the run that had just settled, which is then the newest run and
 * `live`. Spending `live` on the whole run expanded a bar the reader had just
 * read while the agent thought, and condensed it again when the agent stopped.
 * `planRun` now spends it only on the rows after the last SETTLED close (see
 * `settledCloseOf`): everything the reader has already seen settle keeps its bar,
 * and the one thing drawn in place is the cycle still being written. When that
 * cycle settles its span becomes a bar at the TAIL, below everything that was
 * already condensed, so nothing above it reflows.
 *
 * THE FOCUS HOLD (`options.focusHold` / `options.openRuns`). A collapse is a
 * transition the reader did not initiate, and it UNMOUNTS rows - a reader
 * whose keyboard focus sits inside a row this pass would hide loses focus to
 * the body. The caller passes the record id holding focus inside the
 * transcript (or null) and the keys of the bars the reader has opened
 * (`openRuns` holds SEGMENT keys, each derived from the span's own anchor row;
 * a key of an older shape names nothing and simply renders collapsed); a segment
 * that would hide the focused row and is NOT open simply does not collapse this
 * pass. The plan still states what it WOULD hide, and the next pass - focus
 * moved on - folds it. Deliberately narrower than "no collapse while focused":
 * a bar over a run the reader is not in cannot disturb them, and a reader
 * already looking at the rows (the bar is open) is not mid-transition.
 */
/**
 * DEV INSTRUMENTATION (UI perf audit A3): how many times `collapsePlan` runs.
 *
 * The walk's question is asked of a plan, and a transcript update used to ask it
 * by building a SECOND plan over the whole store; this counter is what the
 * before/after bench reads to show the per-update cost. Same shape as C1's
 * `messageInputRenderCount`: a module-level counter the bench reads and nothing
 * else touches at runtime.
 */
export const dbgCollapsePlanCalls = { count: 0 };

/**
 * SIGNATURES COMPUTED - a memo MISS, the instrument that pins the per-record
 * memo (agent review round 2, m1r2). Exported beside `dbgCollapsePlanCalls` in
 * the same shape: a module-level counter a test reads and nothing else touches
 * at runtime.
 *
 * WHY IT HAS TO EXIST. Removing the `WeakMap` in `recordSignature` left this
 * suite **7/7 green**, because every shape the bench drives
 * (`transcript-perflush-perf.test.mjs` A3 and A7) hands back a FRESH object for
 * every row every flush - the one shape the memo cannot help - so the
 * optimisation could rot back to a window-per-flush signature with every test
 * passing. The assertion that reads this counter
 * (`scripts/collapse-plan-per-pass.test.mjs`, "the signature is computed once per
 * record object") fails the moment the memo is gone: a one-record flush over an
 * otherwise identity-preserving window misses exactly once with it, and once per
 * row without it.
 */
export const dbgRecordSignatureCalls = { count: 0 };

export function collapsePlan(
	rows: Row[],
	options: {
		live: boolean;
		focusHold?: string | null;
		openRuns?: ReadonlySet<string>;
		/**
		 * THE SETTLE LATCH (operator report, 2026-10-16): the run this pane has
		 * WATCHED live, named by its opening row id.
		 *
		 * THE REPORT: "when I send a steering message, it alternates between
		 * condensing the messages before it and expanding them". The pane's
		 * `live` flag (`canonical-transcript.tsx`'s `working !== null || gate !== null`)
		 * dips at tool-segment boundaries - the runtime's `frontend.streaming`
		 * reads false for a beat mid-turn (sampled through a real steer send in
		 * `scripts/stt-dictation-proof.mjs`: "the snapshot's `streaming` flag was
		 * dipping at a tool-segment boundary"). A dip lands here as `live: false`,
		 * which reads as "the turn is over": every span condenses (`liveFrom = -1`),
		 * and the return pass expands the in-flight spans again - a fold/unfold
		 * pair per flip while the reader does nothing. The 2026-10-01 fix
		 * (`settledCloseOf`) holds while `live` is steady; a dip bypasses it, and a
		 * steer's delivery lands ON a boundary, so the reported alternation is
		 * this pair repeating.
		 *
		 * WHY A LATCH RATHER THAN A SMARTER ARM. "Is this dip a transient?" is not
		 * a question the rows can answer: a first-cycle dip and a genuinely dead
		 * run are the same rows (no close, no marker), so any stateless rule that
		 * refuses to condense one also refuses the other - and the "dead run"
		 * residual is pinned behaviour (`turn-collapse-model.test.mjs`'s R2 case).
		 * The pane therefore remembers WHICH RUN it has watched being written, and
		 * a watched run's plan stops reading `live` at all: `liveFrom` is the
		 * settled close + 1 in every pass, so any number of liveness transitions
		 * reach the same plan. Condensation for a watched run is settledness-driven
		 * alone - it moves when the rows move (a close lands, a marker lands), never
		 * on the flag.
		 *
		 * WHAT THE LATCH DELIBERATELY TRADES. A watched run whose tail never
		 * settles while the pane stays open keeps that tail drawn in place: an
		 * invisible close (a tool-call-only answer) or a marker-less death defers
		 * its bar until the next settled close claims it. That is the conservative
		 * direction - the reader keeps reading what they watched instead of losing
		 * it to a transient - and every real end (a visible answer, a completion
		 * notice, a session incident) settles and folds as before. A FRESH pane
		 * (reload, first open) holds no latch, so a finished transcript arrives in
		 * today's collapsed shape; the accounting is the store's
		 * (`shared/store/turn-collapse-watched.ts`) and the tests'.
		 *
		 * The anchor is the id the transcript wrote for the run while it watched -
		 * its leading row as the pane saw it, which is the run's OPENING ROW when
		 * the window covered it and the CUT EDGE when the window cut the run's
		 * head. The check below is CONTAINMENT over the run's own rows rather than
		 * an equality at its opening index, and that is load-bearing, not an
		 * implementation detail (agent review round 1, R1-1): a head-cut run opens
		 * at its first VISIBLE row (`openingIndex` is slice-relative), and the snap
		 * grows the window into the run's true head on exactly the dip frame (the
		 * completed-run allowance opens when the pane reads not-live), so an
		 * equality there reads a different row precisely when the latch is needed
		 * and the reported alternation survives for tall newest runs - the
		 * reviewer's band: `300 < extra <= 720`, the shape the operator's journal
		 * runs are. Containment is invariant to both moves. The id also survives
		 * the run's live re-keying (a run keys on its last row while being written
		 * and on its answer once settled - the key moves, the registered row does
		 * not). Only the NEWEST run is ever watched: it is the only run liveness is
		 * spent on.
		 */
		watchedRunId?: string | null;
		/**
		 * The reader's transcript display mode (issue #756). Threaded VERBATIM to
		 * `partitionRun` and consulted there; this function has no rule of its own
		 * over it, so the plan cannot grow a second opinion about what a mode means.
		 * Omitted is the shipped `by-turn` condensation.
		 */
		mode?: TranscriptDisplayMode;
		/**
		 * The server's per-run facts when the page carries them (`RunFact`), keyed by
		 * `run_key` - which IS this module's `TurnRun.key`, so the join is a field
		 * and never a position. Omitted or empty is every old backend, every
		 * `building` answer and every peer's conversation, and changes nothing.
		 */
		/**
		 * Whether the reader hides cross-session rows (`hide_cross_session`).
		 *
		 * IN THE PLAN because the facts state the SERVER's totals, which count the
		 * `send` rows and peer receipts this reader filters out before the plan ever
		 * sees them; `factTotalsFor` has to subtract them, and refuses to guess when
		 * it cannot (agent review round 1, F2).
		 */
		hideCrossSession?: boolean;
		/*
		 * The facts carry a run one the page does not paint (`runs[0]` is a run
		 * early on purpose, `open_frame.py`), a `building` answer and every peer's
		 * conversation, and changes nothing.
		 */
		runFacts?: RunFactLookup;
		/**
		 * The page the facts came with, by identity (`FactPageIdentity`). Consulted
		 * beside them, and for the same reason: it is what tells a row the wire's
		 * figure counted from one that arrived after it was published (B2). Omitted
		 * keeps every existing behaviour; a page identity that cannot be compared
		 * refuses the fact rather than stating a figure that may have moved.
		 */
		factPage?: FactPageIdentity | null;
		/**
		 * Whether rows OLDER than the loaded head exist (the caller's paging state:
		 * `hasMore`, or rows the render window hides). Defaults to false, so every
		 * caller that does not know keeps today's behaviour.
		 *
		 * WHY THE PLAN NEEDS IT. An internal episode may collapse only on settled
		 * work, and the settled answer it continues is read from the LOADED records.
		 * At a head cut that answer is simply off-page: a plan that cannot tell "the
		 * run really starts with this receipt" from "an older page has not landed"
		 * treats both as the first and paints the second loose until the head
		 * arrives. This is the one fact that separates them. It is consulted only at
		 * the loaded edge (`!headLoaded`), never for a span inside the page.
		 */
		olderRowsExist?: boolean;
	},
): CollapsePlan {
	dbgCollapsePlanCalls.count += 1;
	/*
	 * Only the NEWEST run can be the one in flight, and only while the pane says
	 * the turn is unsettled (`live`): every earlier run is a finished turn and
	 * behaves like one even when a later turn is streaming above the reader's
	 * place (§4.5's live rule, which `TraceFold`'s `sectionLive` states for
	 * folds as "a run in an OLDER turn must not open itself because a LATER turn
	 * happens to be running"). Within that run `planRun` spends `live` on the
	 * in-flight CYCLE only, not on the whole run.
	 */
	const runs = runsOf(rows);
	const focusHold = options.focusHold ?? null;
	const openRuns = options.openRuns ?? NO_OPEN_RUNS;
	const watchedRunId = options.watchedRunId ?? null;
	return {
		runs: runs.map((run, index) =>
			planRun(
				rows,
				run,
				index === runs.length - 1 && options.live,
				focusHold,
				openRuns,
				options.mode,
				options.runFacts,
				options.hideCrossSession === true,
				options.factPage,
				options.olderRowsExist === true,
				/*
				 * THE SETTLE LATCH, spent the way `live` is: on the NEWEST run only.
				 * `watchedRunId` names the run the pane has watched being written; the
				 * check is containment over the run's rows, never an equality at its
				 * opening index - see `runHoldsRowId` for the window snap's move that
				 * makes the difference (agent review round 1, R1-1).
				 */
				index === runs.length - 1 &&
					watchedRunId !== null &&
					runHoldsRowId(rows, run, watchedRunId),
			),
		),
	};
}

/** The empty set `collapsePlan` reads when the caller hands it no `openRuns`. */
const NO_OPEN_RUNS: ReadonlySet<string> = new Set();

/**
 * Does this run hold the row `id`, anywhere in its span?
 *
 * WHY CONTAINMENT AND NOT THE OPENING INDEX (agent review round 1, R1-1). A run
 * whose head the window cuts opens at its first VISIBLE row - `openingIndex` is
 * slice-relative (`transcript-rows.ts`) - so the id the transcript wrote while
 * live names the slice's leading row, and the snap grows the window to the
 * run's true head on exactly the dip frame (the completed-run allowance in
 * `snapWindowToRunBoundary` opens when the pane reads not-live). An equality
 * test at `openingIndex` therefore reads a DIFFERENT row on the one frame the
 * latch exists for: the dip folds the tail and the re-entry unfolds it - the
 * reported alternation, surviving for tall newest runs. Containment is
 * invariant to both moves: the written row is always one of the run's own rows,
 * head arrived or not.
 */
function runHoldsRowId(rows: Row[], run: TurnRun, id: string): boolean {
	for (let index = run.openingIndex; index <= run.endIndex; index += 1) {
		if (rows[index]?.record.id === id) return true;
	}
	return false;
}

/** The span's gap tier: see the comment at its one call site. */
function segmentGap(first: Row, precededByVisibleRow: boolean): Row["gap"] {
	/*
	 * A bar that follows a VISIBLE row (a pinned marker, a previous bar's answer)
	 * is a block boundary of its own, so it takes the block step rather than the
	 * ledger's hairline: built against its old neighbour - often a tool row that
	 * has collapsed - its first hidden row arrives at the trace tier, which would
	 * leave the bar hugging the marker above it by 2px. The bar that follows the
	 * opening user row keeps that row's turn gap (the shipped rule).
	 */
	return precededByVisibleRow && first.gap === "trace" ? "item" : first.gap;
}

function factsOf(
	rows: readonly Row[],
	options: {
		partial: boolean;
		/** Whether the span's own head is loaded (see the duration's gate below). */
		headLoaded: boolean;
		/** The quiet group this span IS, when it is one (see `TurnSummaryFacts.group`). */
		group?: QuietGroup | null;
	},
): TurnSummaryFacts {
	/*
	 * THE QUIET CALL IS NOT WORK (design §5(e)): it closes the turn, it is not an
	 * action inside it, so it is subtracted from both the count and the worked
	 * sum - the same exclusion core's `without_ask_gate_divert` makes for a divert
	 * (`harness/rows.py`), and the reason `[peer][tool][no_reply]` states `1
	 * action` and not two. Filtering the rows ONCE keeps the sum a single spelling
	 * of `workedSeconds`.
	 */
	const workRows = rows.filter(
		(row) => !(row.record.kind === "tool" && isQuietTurnCall(row.record)),
	);
	const actions: FoldableAction[] = [];
	let failed = 0;
	let firstFailedId: string | null = null;
	for (const row of workRows) {
		if (row.record.kind !== "tool") continue;
		const failedHere = isFailedCall(row.record);
		actions.push({ name: ledgerName(row.record), failed: failedHere });
		if (failedHere) {
			failed += 1;
			firstFailedId ??= row.record.id;
		}
	}
	const worked = workedSeconds(workRows);
	return {
		/*
		 * Shown iff the span's calls reported at least a second - never a `0s`
		 * claim (§4.4) - AND the span's head is REAL: a head-cut span holds only the
		 * rows that happen to be loaded, so a sum over them is a fragment of the
		 * turn's work stated as if it were the turn's (the honesty half of the
		 * end-loaded rule, the same reason its count carries `+`).
		 */
		durationS:
			options.headLoaded && worked !== null && worked >= 1 ? worked : null,
		actions: actions.length,
		/*
		 * The count is a MINIMUM for exactly the reason the duration is absent: the
		 * rows above the loaded span are unknown, so `actions.length` is what has
		 * been LOADED of this turn. The bar says so (`N+ actions`).
		 */
		partial: options.partial,
		failed,
		firstFailedId,
		title: actions.length > 0 ? foldSummary(actions) : null,
		countUnits: actions.length > 0 ? foldCountSegments(actions) : [],
		group: options.group ?? null,
	};
}

/**
 * The index (in the run's record list) of the last row the reader has seen
 * SETTLE - everything up to and including it is finished work - or -1 when the
 * run has none (a first turn that has not answered yet: all of it is in flight).
 *
 * THREE KINDS OF ROW SETTLE, and the largest index wins:
 *
 * - a CLOSE of a cycle (`cyclesOf`: an assistant row with no step after it before
 *   the next trigger), except one the provider declared `stopReason: "toolUse"`.
 *   That is the lead-in frame: its prose settles a moment BEFORE its own tool row
 *   paints, and for that moment it is the run's last row, which `cyclesOf` reads
 *   as a close. Treating it as one would condense the cycle's work, then
 *   un-condense it the instant the call arrives - the flip this function exists
 *   to prevent. Absent or any other declaration counts as settled (the
 *   unknown-is-absent rule the segments module states for
 *   `reportsCompletedThought`);
 * - a TERMINAL marker (`isTerminalMarker`: `Stopped with an error`, `Interrupted`,
 *   the `closed` / `retired` receipts, a session incident). A turn that ended in
 *   one has no assistant close at all, so reading only closes left it "all in
 *   flight" the moment a wake or peer message re-opened the run - the operator's
 *   complaint, in the shape of an interrupted or failed turn. It is the segments
 *   module's own boundary vocabulary, not a second list;
 * - an assistant row the provider declared FINISHED (`reportsCompletedThought`).
 *   `partitionRun` already keeps such a row on screen as a settled thing (V4), and
 *   it is not always a close: the todo guardrail re-enters the loop after a
 *   `stop` yield, so `[U T T A(stop) T]` has tool work after it and no new trigger,
 *   which `cyclesOf` reads as narration. Without this clause the idle plan
 *   (`U T T A`) condensed the first span and the live plan (`U T T A T`) drew it
 *   in place again - the same flip, with no wake in it;
 * - a QUIET close (a settled `no_reply`; design §5). It arrives through the cycle
 *   loop above like any other close - its row is not an assistant, so the
 *   `toolUse` skip does not apply - and that is the whole mechanism: the tail
 *   settles (`liveFrom` moves past the quiet row) and the working line retires,
 *   with nothing claiming an answer was handed over.
 */
function settledCloseOf(
	records: readonly CondenseRecord[],
	cycles: readonly TurnCycle[],
): number {
	let settled = -1;
	for (let i = cycles.length - 1; i >= 0; i -= 1) {
		const close = records[cycles[i].closeIndex];
		if (close.kind === "assistant" && close.stopReason === "toolUse") continue;
		settled = cycles[i].closeIndex;
		break;
	}
	for (let i = records.length - 1; i > settled; i -= 1) {
		const record = records[i];
		if (
			isTerminalMarker(record) ||
			(paintsSomething(record) && reportsCompletedThought(record))
		) {
			return i;
		}
	}
	return settled;
}

/**
 * Does this run's OWN data say it is still being written?
 *
 * WHY THIS EXISTS BESIDE THE SETTLE LATCH (operator report, 2026-10-21). The
 * latch was built on "the rows cannot state whether a turn will resume", which
 * is true of exactly one window: the INTER-CYCLE gap, where a finished call and
 * a dead run are the same rows. It is false for the rest of a live turn, and
 * reading the pane's `live` flag alone there is what made a one-frame flicker
 * out of every liveness dip:
 *
 * - an assistant row still `streaming` cannot be a finished turn's last word;
 * - a tool row whose `phase` is not `done` has not reported its outcome, so the
 *   turn that called it is still open.
 *
 * Both are facts about the run's own records, so anchoring on them is
 * idempotent under `live`: the plan can only change when the rows change, which
 * is the requirement this whole tier exists to hold. A dip (a tool-segment
 * boundary) and a blank (the reconnect arm sets `frontend: null`, which drops
 * `working` and even the latch's store key) both stop mattering here.
 *
 * WHEN IT RELEASES. The moment the last row stops claiming to be live: the
 * streaming flag clears and the pending tool reports `done`. That is the pass
 * on which a completed turn condenses - a real change in the data under the
 * rows, not a sample of a flag.
 *
 * WHAT IT DELIBERATELY DOES NOT COVER. A run whose rows are all settled and
 * whose turn has not resumed yet (the inter-cycle gap). Nothing in the rows can
 * separate that from a run that has genuinely finished, so the pane's latch
 * (`watchedRunId`) still owns it - see `collapsePlan`'s option.
 *
 * THE RELEASE IS THE ROW, AND THERE IS NO TIMEOUT (agent review round 1, F1).
 * The normal operator is `agent_end`, which settles every streaming row and
 * every non-`done` call (`transcript-reducer.ts`'s sweep), and a durable page
 * always carries settled rows. The residue is a LOST TERMINAL FRAME - a process
 * killed mid-turn with no `agent_end` and no `session_incident` - where the row
 * keeps claiming to be live and this run therefore never condenses. That
 * direction is chosen: a stale card that still says it is running is a wrong
 * row the reader can see and act on, while folding rows on a wall clock would
 * hide work under a bar the moment a slow call outlived the guess. The escape
 * already exists in the data - any terminal marker (an error-level `custom`, a
 * completion notice) condenses the run again - and a re-read of the session
 * replaces the stale rows outright.
 */
function runStatesItsOwnLiveness(
	records: readonly CondenseRecord[],
): boolean {
	for (const record of records) {
		if (record.kind === "assistant" && record.streaming === true) return true;
		if (
			record.kind === "tool" &&
			(record.phase === "running" ||
				record.phase === "composing" ||
				record.phase === "queued")
		) {
			return true;
		}
	}
	return false;
}

/**
 * The fact for this run, found by its STABLE identity rather than by the run's
 * live key (agent review round 1, F3).
 *
 * WHY `runFacts.get(run.key)` IS NOT ENOUGH. `TurnRun.key` is derived from the
 * rows in hand, and a run keeps GROWING after it has answered: a wake, a job
 * result or a peer follow-up is not a user row, so it continues the run, and
 * the client re-keys the run to whatever its last row (or last answer) now is.
 * The fact was read once, from the page the frame carried, and is keyed by the
 * run's identity AT THAT MOMENT - so a direct lookup misses exactly the runs
 * that are most likely to be head-cut in the first place (a huge tail run that
 * is still being woken) and the bar drops back to a fragment, arming the walk
 * after the paint. On the base, where there were no facts to lose, the open's
 * own walk had already completed that bar.
 *
 * SO THE RUN'S ROWS ARE ASKED INSTEAD: every id the map answers to (`run_key`,
 * `closing_answer_id`, `opening_user_id` - `open-frame.ts` indexes all three) is
 * a row of THIS run, and row ids are unique, so a hit is the run's own fact
 * however the run has been re-keyed since. The direct lookup stays first because
 * it is the common case and costs nothing.
 */
function factForRun(
	rows: readonly Row[],
	run: TurnRun,
	facts?: RunFactLookup,
): RunFact | null {
	if (!facts || facts.size === 0) return null;
	const direct = facts.get(run.key);
	if (direct !== undefined) return direct;
	for (let i = run.openingIndex; i <= run.endIndex; i += 1) {
		const hit = facts.get(rows[i]?.record.id ?? "");
		if (hit !== undefined) return hit;
	}
	return null;
}

/** A run's own totals, as `factTotalsFor` states them for one reader. */
type FactTotals = {
	actions: number;
	workedSeconds: number | null;
	failed: number;
	complete: boolean;
};

/**
 * The run's totals AS THE WIRE PUBLISHED THEM, or null when the fact cannot be
 * applied to this reader's rows at all.
 *
 * ONE CORRECTION, and it is the difference between the server's population and
 * the client's:
 *
 * - THE HIDDEN CROSS-SESSION WORK IS SUBTRACTED (agent review round 1, F2).
 *   With `hide_cross_session` on, the plan runs over
 *   `visibleRecords(records, true)`, which drops the `send` tool rows and the
 *   peer receipts; the server's `action_count` counts them. Subtracting the
 *   filtered siblings from the unfiltered total put them back into the bar
 *   (measured: `3 actions` where the visible span holds `2`).
 *
 * AND NOTHING IS ADDED, which is what round 2's B1 is about. An earlier revision
 * added every row after `fact.closingAnswerId` on the premise that the server's
 * run ends at its elected answer. IT DOES NOT: the core keeps accumulating every
 * tool row it touches while a run is open - a wake, a job result or a peer
 * follow-up continues the run (`transcript_index.py::_track_run`'s non-user arm
 * never consults `_run_closed()`, which is the very reason a run re-keys at all,
 * F3) - and `_emit_runs` publishes `action_count` untrimmed. #2102's own F4 note
 * measures it: a woken run reports `settled: true, actions: 1` and later
 * `actions: 3` under the SAME key. So `closing_answer_id` names where the run's
 * ANSWER was elected, not where its counting stopped, and the wire's figures are
 * already the run's totals as published. Adding the tail again painted
 * `36 actions / 66 s` for a run whose own figure was `33 / 63 s`, with
 * `partial: false` claiming exactness and the walk retired: the wrong number was
 * FINAL, which is the one outcome this whole lane exists to prevent.
 *
 * A ROW THAT ARRIVED AFTER THE PAGE WAS PUBLISHED is therefore the only case
 * where the wire can be SHORT, and it cannot be told from the rows the fact did
 * count by any clock this client may compare (the F7 lesson: a live row's
 * instant is a receipt stamp, not the journal's). It is caught where it is
 * visible instead - if a bar's own hidden tool rows outnumber the figure it
 * states, the figure is a floor and keeps the `+` (see the floor check in
 * `planRun`).
 *
 * AND WHEN IT CANNOT SPLIT THE HIDDEN WORK IT REFUSES, rather than guessing:
 * with the setting on and no `crossSessionActions` from the backend, the fact's
 * count may include rows this reader never sees, so there is no honest figure to
 * state - the loaded fold stands, the bar keeps its `+`, and the walk stays
 * armed to complete it (`alignWalkRunKeyConfirmed` reads the same refusal). The
 * duration follows the same rule one step further: it is kept when the hidden
 * rows are provably absent (`crossSessionActions === 0`) or when the backend
 * states their seconds, and dropped otherwise, because a `Took` that includes
 * work the bar does not show is the same class of lie as a count that does.
 */
/**
 * The run's tool rows the fact cannot have counted, or `null` when that cannot be
 * established (agent review round 3, B2).
 *
 * THREE ANSWERS, and the difference matters:
 * - no page identity was SUPPLIED (an old rig, a model test, a pane with no facts)
 *   -> `[]`: there is no page to compare against, and today's behaviour stands;
 * - an identity was supplied but CANNOT be used (the page named no newest entry, or
 *   none of its rows are among the ones in hand) -> `null`: the caller refuses the
 *   fact rather than trusting a figure it cannot check;
 * - an identity in hand -> the rows below.
 *
 * THE TEST IS IDENTITY, AND THE IDENTITY IS ROW KEYS, NOT ENTRY IDS (agent review
 * round 4, B3). The page's carried set is mapped through `entryRecordKey` before it
 * reaches this function, because the page's raw entry ids do not match the client's
 * rows for the two classes that matter - a tool entry keys `tool:<tool_call_id>`,
 * and a `completion_attention` entry keys its anchor - so an id-set matched no tool
 * row at all and a SERVED page's own tool rows were added as if they had arrived
 * after it (`12 / 33 s` stated where the wire published `9 / 24 s`).
 *
 * With that mapping the page is a contiguous tail window, so its LAST carried row
 * still in hand is the watermark: a tool row of this run that sits after it and is
 * not one of the page's own keys arrived after the page was published. The rows a
 * wake added BEFORE the read stay out of the sum (the round-2 double count), and a
 * row this build cannot attribute refuses the fact (see the loop).
 */
function factAppendedRows(
	rows: Row[],
	run: TurnRun,
	factPage: FactPageIdentity | null | undefined,
): Row[] | null {
	if (factPage === undefined) return [];
	if (factPage === null) return null;
	let watermark = -1;
	for (let i = 0; i < rows.length; i += 1) {
		if (factPage.ids.has(rows[i].record.id)) watermark = i;
	}
	if (watermark < 0) return null;
	const appended: Row[] = [];
	for (
		let i = Math.max(watermark + 1, run.openingIndex);
		i <= run.endIndex && i < rows.length;
		i += 1
	) {
		const row = rows[i];
		if (row.record.kind !== "tool") continue;
		if (factPage.ids.has(row.record.id)) continue;
		/*
		 * A ROW THIS BUILD CANNOT ATTRIBUTE TO A CALL refuses the fact rather than
		 * being corrected by it: a tool row with no call key, or one the renderer knows
		 * was never sent (`neverSent`/`notRunReason`), is not a row the wire's
		 * `action_count` counted either, so adding it would state a figure no read can
		 * confirm.
		 *
		 * ABSENT IS NULL HERE (agent review round 5, n1): the sibling readers of these two
		 * fields (`buildRows`, the trace fold) answer the same question with `?? null`, and
		 * a `!== null` on the raw value would read a record that simply omits the key as
		 * "not run" - refusing a fact the refuser was never about.
		 */
		if (!row.record.id.startsWith("tool:")) return null;
		if (row.record.neverSent === true) return null;
		if ((row.record.notRunReason ?? null) !== null) return null;
		appended.push(row);
	}
	return appended;
}

function factTotalsFor(
	fact: RunFact | null,
	hideCrossSession: boolean,
): FactTotals | null {
	if (fact === null) return null;
	let actions = fact.actions;
	let worked = fact.workedSeconds;
	if (hideCrossSession) {
		const hidden = fact.crossSessionActions;
		if (hidden === null) return null;
		actions = Math.max(0, actions - hidden);
		if (worked !== null) {
			const hiddenWorked = fact.crossSessionWorkedSeconds;
			if (hiddenWorked !== null) worked = Math.max(0, worked - hiddenWorked);
			else if (hidden > 0) worked = null;
		}
	}
	return {
		actions,
		workedSeconds: worked,
		failed: fact.failed,
		complete: fact.complete,
	};
}

function planRun(
	rows: Row[],
	run: TurnRun,
	live: boolean,
	focusHold: string | null,
	openRuns: ReadonlySet<string>,
	/*
	 * The reader's display mode, threaded straight through to `partitionRun` and
	 * nowhere else: this function owns the key rule, the live split and the stamp,
	 * none of which a mode may move, so it neither reads nor reinterprets the
	 * value. See `collapsePlan`'s option for why the plan keeps no rule of its own.
	 */
	mode?: TranscriptDisplayMode,
	/**
	 * The server's facts for the runs this page carries (`RunFact`), or nothing.
	 * Consulted for the HEAD-CUT span only, and only when the run the page names
	 * is one of these keys; see the block after the segments below for why a
	 * fully-loaded run is deliberately left to the rows it already has.
	 */
	runFacts?: RunFactLookup,
	/**
	 * Whether the reader hides cross-session rows, threaded to `factTotalsFor`
	 * and nowhere else: the plan's rows are already filtered by the caller, so the
	 * only thing this flag may move is the arithmetic on the SERVER's totals
	 * (agent review round 1, F2). Defaults to today's answer - visible - so a rig
	 * that predates the setting keeps its figures.
	 */
	hideCrossSession = false,
	/**
	 * The page the facts came with, by identity (agent review round 3, B2): tool
	 * rows that arrived after it was published are added to the fact, or make it
	 * refuse, so a wake on an open pane cannot leave a stale total stated as final.
	 * Omitted is every caller with no page identity to give, and keeps the fact
	 * exactly as it was.
	 */
	factPage: FactPageIdentity | null | undefined = undefined,
	/** See `collapsePlan`'s `olderRowsExist`. Default false = every old caller. */
	olderRowsExist = false,
	/**
	 * Whether this run is the one the pane has WATCHED live (the settle latch,
	 * see `collapsePlan`'s option). Threaded for the newest run only, the way
	 * `live` is: a watched run's settledness reading is kept in every pass, so a
	 * liveness dip or re-entry cannot move its hide set.
	 */
	watched = false,
): RunCollapsePlan {
	const runRows = rows.slice(run.openingIndex, run.endIndex + 1);
	const records = runRows.map((row) => row.record);
	/*
	 * The span starts just after the opening USER row - or at the list's own
	 * beginning for a run whose head is cut off, where the first row is simply
	 * the oldest one loaded and nothing before it is knowable. A head-cut run
	 * CAN collapse (the end-loaded rule), which is exactly why the duration gate
	 * refuses to read `start` there: this position is the loaded span's edge, not
	 * the turn's own beginning.
	 */
	const from = run.opensWithUserRow ? 1 : 0;
	/*
	 * THE RUN'S FACT, CONSULTED BEFORE THE SPANS ARE BUILT, because the facts no
	 * longer only patch a span's NUMBERS (the block below the map): they also say
	 * whether the span at the loaded edge is a fragment at all. See
	 * `headLoadedIsTheRunsOwn` for the case that made this necessary, and
	 * `open-frame.ts` for why the lookup answers to three ids.
	 */
	const fact = factForRun(rows, run, runFacts);
	/*
	 * WHAT THE FACTS STATE FOR THIS RUN, as this reader counts rows - and null
	 * when they cannot be applied at all, which the head rule below and the walk's
	 * gate both read. See `factTotalsFor` for the two corrections and why a
	 * refusal is the honest answer rather than a guess.
	 */
	/*
	 * ROWS THAT ARRIVED AFTER THE PAGE WAS PUBLISHED ARE NOT IN THE WIRE'S FIGURE
	 * (agent review round 3, B2), and the client holds them the moment a wake, a job
	 * result or a peer receipt lands on an open pane: `factAppendedRows` finds them
	 * by identity, the block below adds exactly them - count and worked seconds
	 * together, and only when every one of them reports a duration, so the sum
	 * cannot silently understate - and REFUSES the fact when the question cannot be
	 * answered at all. The alternative was measured: `30 / 60 s` stated as final for
	 * a run whose own figure had already moved to `33 / 63 s`.
	 */
	const factAppended =
		fact === null ? null : factAppendedRows(rows, run, factPage);
	let factTotals =
		factAppended === null ? null : factTotalsFor(fact, hideCrossSession);
	if (factTotals !== null && factAppended !== null && factAppended.length > 0) {
		const measurable = factAppended.every(
			(row) => typeof workedSecondsOf(row) === "number",
		);
		const addWorked = measurable ? workedSeconds(factAppended) : null;
		if (
			!measurable ||
			(factTotals.workedSeconds !== null && addWorked === null)
		) {
			factTotals = null;
		} else {
			const limit = factTotals.workedSeconds;
			factTotals = {
				...factTotals,
				actions: factTotals.actions + factAppended.length,
				workedSeconds:
					limit === null || addWorked === null ? null : limit + addWorked,
			};
		}
	}
	/*
	 * WHETHER THE PLAN'S OPENING ROW IS THE RUN'S OWN HEAD, which is the only
	 * question a fragment's figures hang on.
	 *
	 * WITH NO USABLE FACT the client's own partition is the only opinion available
	 * and it is the one today's behaviour already rests on: `opensWithUserRow` true
	 * means the span opens at a user row (a whole head), false means the loaded
	 * list begins inside the run.
	 *
	 * WITH ONE the wire's `opening_user_id` outranks it, and that is the STEER case
	 * `docs/DESKTOP_API.md` names: a page may begin at a steer row, so the model
	 * sees a user row at the top and would otherwise call the span whole while
	 * every row above the steer - the run's actual head included - is off-page. The
	 * fact says where the run actually starts; a span whose opening row is anything
	 * else is a fragment, and takes the fact's figures below.
	 *
	 * `null` from the wire is NOT "unknown": it states that the run opens off a
	 * non-user row, so a user row at the top of the span is not its opener either.
	 * The direction is deliberate - the facts win every disagreement with the
	 * local partition, and where they are merely the same answer the subtraction
	 * below reproduces the loaded fold exactly.
	 */
	const headLoadedIsTheRunsOwn =
		factTotals === null
			? run.opensWithUserRow
			: run.opensWithUserRow &&
				fact?.openingUserId != null &&
				rows[run.openingIndex]?.record.id === fact.openingUserId;
	const internalContext = internalEpisodePins(records);
	const partition = partitionRun(records, {
		from,
		paints: paintsSomething,
		isStatement: isStatementRow,
		/*
		 * MATERIAL OWNERS ARE PINNED INSIDE A RECEIPT EPISODE ONLY
		 * (`internalEpisodeBoundary`), never for every run: an ordinary human turn
		 * still folds its pictures under its own bar with the strip, which is the
		 * base behaviour six ruled design/UX tests pin. Pinning them globally
		 * stopped those runs collapsing at all.
		 */
		pinned: (record) =>
			staysVisibleWhileCollapsed(record) || internalContext.pins.has(record),
		mode,
	});
	// Reuse authoritative RESPONSE closes for old-core rows without a V4 stop
	// marker. Explicit toolUse/refusal/error endings are not a completed human
	// response; no new answer election or body interpretation is introduced.
	const responseCloses = new Set(
		partition.cycles
			.filter((cycle) => {
				const record = records[cycle.closeIndex];
				return (
					cycle.class === "response" &&
					record.kind === "assistant" &&
					!record.streaming &&
					(record.stopReason == null || reportsCompletedThought(record))
				);
			})
			.map((cycle) => records[cycle.closeIndex]),
	);
	const humanPendingBefore: boolean[] = [];
	let humanPending = false;
	for (const record of records) {
		humanPendingBefore.push(humanPending);
		if (
			record.kind === "user" ||
			(record.kind === "custom" &&
				(record.receipt?.expectsReply === true ||
					record.receipt?.steer === true))
		)
			humanPending = true;
		else if (
			responseCloses.has(record) ||
			reportsCompletedThought(record) ||
			isQuietTurnClose(record)
		)
			humanPending = false;
	}
	const answerAt = partition.answer?.closeIndex ?? null;
	const answerId = answerAt === null ? null : records[answerAt].id;
	/*
	 * The run's last settled quiet close, as an index (design §5): the completion
	 * anchor a bar can lean on where no answer exists, AND the closer position the
	 * completion mark reads. `-1` from the lookup means the id was not found in
	 * this run's own records, which cannot happen for an id `runsOf` minted over
	 * the same rows - the guard is here so a future caller cannot pass a stranger
	 * in and get an index of `-1` read as "a close before every row".
	 */
	const quietCloseAt = (() => {
		if (run.quietCloseId === null) return null;
		const at = records.findIndex((record) => record.id === run.quietCloseId);
		return at === -1 ? null : at;
	})();

	/* The pre-answer span nearest the answer carries the turn's one stamp. */
	let nearest = -1;
	partition.segments.forEach((span, i) => {
		if (answerAt === null || span.to < answerAt) nearest = i;
	});

	const preAnswerCount = partition.segments.filter(
		(span) => answerAt === null || span.to < answerAt,
	).length;
	const collapsible =
		partition.segments.length > 0 &&
		(run.opensWithUserRow ||
			run.closingAnswerId !== null ||
			/*
			 * The quiet close is the third closing anchor (design §5(b)): a run that
			 * handed nothing over still ended, and the gate exists precisely so a run
			 * with its closing anchor in the list may condense from the loaded span.
			 * It rides its OWN field, never `closingAnswerId` - see `TurnRun`.
			 */
			run.quietCloseId !== null);
	/*
	 * Rows at or after this index belong to the cycle still being written; -1
	 * means nothing is in flight (the pane is not live), so every span may condense.
	 * `settledAt` is the same reading BEFORE the +1 (the run's last settled row,
	 * -1 when there is none), kept beside it because the group exemption below
	 * reads it directly.
	 *
	 * THE SETTLE LATCH ANCHORS THE READING FOR A WATCHED RUN. `live` here is the
	 * pane's instantaneous flag, and it dips mid-turn at tool-segment boundaries
	 * (see `collapsePlan`'s `watchedRunId` option): a dip read as "the turn is
	 * over" condensed every span, and the return expanded the in-flight ones
	 * again - the fold/unfold alternation the operator reported when steering.
	 * A run this pane has watched being written keeps its settledness reading in
	 * BOTH directions of the flag, so `live: true` and `live: false` produce the
	 * same plan for it; the flag still decides for every run the pane has NOT
	 * watched (a fresh open, a finished transcript), which is what keeps the
	 * idle-residual shape (R2) exactly as it was.
	 */
	const anchored = live || watched || runStatesItsOwnLiveness(records);
	const settledAt = anchored ? settledCloseOf(records, partition.cycles) : -1;
	const liveFrom = anchored ? settledAt + 1 : -1;

	/*
	 * THE ONE SPAN THE FACTS CAN COMPLETE. Only a span that OPENS at the loaded
	 * edge is a fragment: its rows above are off-page, while every other span
	 * begins and ends inside the loaded page and is therefore exact already.
	 * Recorded during the map rather than searched for afterwards, so the flag the
	 * segment was built with and the span this block patches cannot drift apart.
	 */

	// Ordinary presentation slots survive a focus hold: otherwise the reader's
	// focus makes its bar temporarily loose and incorrectly grants the live
	// episode a second slot. Record pre-focus eligibility, not every loose tail.
	const ordinarySlots = new Set<number>();
	const segments: SegmentPlan[] = partition.segments.map(
		(span: SegmentSpan, i): SegmentPlan => {
			const segRows = runRows.slice(span.from, span.to + 1);
			const afterAnswer = answerAt !== null && span.from > answerAt;
			/*
			 * A span whose head is the LOADED EDGE - the run's first span - states no
			 * duration, and it is the one the facts complete; every other span states the
			 * worked time of ITS OWN rows, so the pre-answer bars add up to the foot's
			 * figure.
			 *
			 * THE EDGE IS `from`, NOT 0. `from` is where the run's hidden rows start - 0
			 * for a run whose head is off-page, 1 for one whose opening user row is on
			 * hand - and the two agree only while `headLoadedIsTheRunsOwn` is true. A
			 * page that begins at a STEER (clarity 2) opens with a user row the model
			 * excludes, so the span that needs the facts starts at 1: testing for 0 there
			 * would leave the fragment looking whole, which is the defect this rule was
			 * added for.
			 */
			const headLoaded = !(span.from === from && !headLoadedIsTheRunsOwn);
			/*
			 * THE GROUP, when this span is one (design §5): >= 2 delivery receipts
			 * with nothing the reader can see between them. `spanHeadLoaded` is the
			 * span's own head-cut verdict just computed, and `open` states whether a
			 * LATER visible row exists anywhere in the load: only the conversation's
			 * last run can grow from its tail, so only there can the group still grow.
			 */
			/*
			 * THE SETTLE LATCH REACHES THE EPISODE FACTS (settle-latch convergence
			 * round, 2026-10-17). These arguments were the pane's raw flag, and a
			 * dip mid-cycle flipped both facts: the one-bar gate below stopped
			 * suppressing (a SECOND bar flashed in and the re-entry took it back)
			 * and `live`/`unsettled` moved on the plan - a painted change for the
			 * run the latch protects. `anchored` is the reading the settledness
			 * lines above already take (`live || watched`): for the watched run
			 * the facts stop reading the flag, so any number of liveness
			 * transitions reach the same plan; for every OTHER run `anchored ===
			 * live` and a dip does what it always did (the R2 residual). The
			 * `span.to >= liveFrom` half stays because settlement moves it: once
			 * a settled close lands beyond the episode it reads false in EVERY
			 * pass, which is what lets the gate stop and the bar arrive.
			 */
			const internal = internalEpisodeOf(
				records,
				span,
				headLoaded,
				anchored &&
					span.to >= liveFrom &&
					span.to === records.length - 1 &&
					run.endIndex === rows.length - 1,
				anchored && span.to >= liveFrom,
				humanPendingBefore[span.from],
				olderRowsExist,
			);
			const group = quietGroupOfSegment(records, span, {
				spanHeadLoaded: headLoaded,
				open:
					span.to === records.length - 1 && run.endIndex === rows.length - 1,
			});
			/*
			 * The key's rule (and the aliasing it prevents) is `SegmentPlan.key`'s. ONE
			 * span can receive APPENDED rows - the one ending at the newest run's end,
			 * because live rows land at the bottom - and it is named for its FIRST row
			 * (`qg:<that row>`), the same row `quietGroupOfSegment` names a group for
			 * (design round F3): the below-minimum span and the group it grows into
			 * then share the key, so the reader's expansion survives the transition
			 * and the settle announcement does not fire a second time for one bar. A
			 * span that can grow UPWARD (a page landing above the loaded head) keeps
			 * the visible-anchor rule: its first row is exactly what moves.
			 */
			const appendable =
				span.to === runRows.length - 1 && run.endIndex === rows.length - 1;
			// Initially bounded spans keep the visible anchor contract. A live episode
			// already opened by the reader keeps its old qg key through the transcript's
			// bounded overlap transfer; don't rename every settled span to solve that.
			const key = appendable
				? `qg:${segRows[0].record.id}`
				: `seg:${
						afterAnswer
							? segRows[0].record.id
							: (runRows[span.to + 1] ?? segRows[segRows.length - 1]).record.id
					}`;

			/*
			 * THE LABEL'S DECISION, THEN ITS WORD (`SegmentLabelKind`). A span the
			 * model leaves nameless but which is a receipt episode opening on a notice
			 * still reads `Job result` - the arm is the episode's own origin, so it
			 * belongs beside the other kinds rather than in the renderer.
			 */
			const labelKind: SegmentLabelKind | null =
				segmentLabelKind(records, partition.cycles, span, group) ??
				(internal !== null && records[span.from].kind === "notice"
					? { kind: "job" }
					: null);
			const label = labelWordOf(labelKind);
			/*
			 * Only positive receipt-origin spans may compact while live. Settlement
			 * is a lifecycle fact, not permission to show private work loosely.
			 * A peer landing in unanswered human work does not change that work's
			 * presentation, and arbitrary tool-only suffixes never earn this rule.
			 */
			const groupKeepsItsBar =
				internal !== null && group !== null && settledAt >= span.from;
			const foldEligible =
				(collapsible || internal !== null) &&
				(liveFrom < 0 ||
					span.to < liveFrom ||
					groupKeepsItsBar ||
					internal !== null);
			const facts = factsOf(segRows, {
				partial: !headLoaded,
				headLoaded,
				group,
			});
			/*
			 * A SPAN WITH NOTHING TO SHOW IS NOT A BAR. `factsOf` excludes the quiet
			 * call from the count, so a span that holds only a settled `no_reply`
			 * (a peer-triggered turn that ended silently, whose opener record has
			 * not - or not yet, on the live path - landed ahead of it) states no
			 * label, no duration and no actions: a bare check and chevron over rows
			 * that paint nothing. It stays loose, where those rows are invisible, and
			 * it does so the same way live and on reload because it reads only the
			 * span's own facts. A labelled bar, a group and an episode are untouched.
			 */
			const hasNothingToShow =
				label === null &&
				internal === null &&
				group === null &&
				facts.actions === 0 &&
				facts.durationS === null;
			if (internal === null && foldEligible && !hasNothingToShow)
				ordinarySlots.add(i);
			const collapsedHere =
				foldEligible &&
				!hasNothingToShow &&
				!(
					focusHold !== null &&
					!openRuns.has(key) &&
					segRows.some((row) => row.record.id === focusHold)
				);
			return {
				key,
				rows: segRows,
				firstId: segRows[0].record.id,
				segmentIds: segRows.map((row) => row.record.id),
				gap: segmentGap(segRows[0], span.from > from),
				afterAnswer,
				label,
				labelKind,
				internal,
				completed: internal?.live
					? false
					: segmentIsCompleted(
							records,
							partition.cycles,
							span,
							answerAt,
							label !== null,
							quietCloseAt,
						),
				collapsed: collapsedHere,
				stampTs:
					i === nearest && answerAt !== null && preAnswerCount === 1
						? (records[answerAt].ts ?? null)
						: null,
				facts,
			};
		},
	);

	/*
	 * ONE BAR WHILE THE TURN THINKS. A LIVE internal episode must not add a
	 * SECOND bar to a run that already has one standing: PR-4 #2 measures
	 * `barsOf() === 1` while the settled turn's bar is on screen, and the
	 * operator's contract is that that bar stays the same element and stays
	 * condensed. An episode arriving when nothing else in the run has condensed
	 * IS that one bar, so it still compacts - the live tests' fixture is exactly
	 * that shape (its run has no other collapsed segment), which is what keeps
	 * its single live group, its appends and its footer clock intact.
	 *
	 * THE BOUND IS THE OTHER BAR, NOT THE EPISODE. The live boundary does not
	 * separate these cases and neither does phase: measured on both fixtures,
	 *   F1 (live tests): settledAt=1 (active-final), liveFrom=2, episode span
	 *     [active-p,active-t] -> span.from 2 >= liveFrom 2, at the boundary;
	 *   F2 (PR-4 #2):     settledAt=2 (answer:1),     liveFrom=3, episode span
	 *     [wake:1,tool:2]      -> span.from 3 >= liveFrom 3, at the boundary;
	 * and both report afterAnswer=false, completed=false, group=null,
	 * receipts=1, live=true - identical in every field describing the episode.
	 * So the predicate has to read what ELSE the run is already hiding: F1's
	 * only collapsed segment is the episode itself, while F2's already has
	 * `tool:1` standing before the wake episode arrives.
	 *
	 * Settled spans, real receipt groups and `groupKeepsItsBar` are untouched by
	 * construction: only a segment that is BOTH unsettled and already collapsed is
	 * suppressed, and only while a sibling ordinary fold slot exists in the SAME
	 * run (including its focus-held/open state). At settlement `internal.unsettled` reads false, the gate stops applying,
	 * and the second bar plus its one announcement appear as PR-4 #2 expects.
	 */
	for (let gateIndex = 0; gateIndex < segments.length; gateIndex += 1) {
		const gateSegment = segments[gateIndex];
		if (!gateSegment.collapsed || gateSegment.internal?.unsettled !== true)
			continue;
		/*
		 * THE READER'S OWN EXPANSION OUTRANKS THE BAR COUNT. Suppressing a live
		 * episode makes it paint loose, which closes the disclosure that reader
		 * opened - and because the key is unchanged the re-collapse patch in the
		 * transcript never runs, so it closes SILENTLY. A segment this reader has
		 * open is therefore left collapsed: `openRuns` is the store's own answer,
		 * and honouring it costs nothing when it is absent.
		 */
		if (openRuns?.has(gateSegment.key)) continue;
		/*
		 * ONLY AN ORDINARY BAR CLAIMS THE ONE-BAR QUOTA. A sibling live episode
		 * is not the pre-existing bar the operator's contract protects - that bar
		 * is the settled turn's own, non-episode fold. Yielding to a fellow
		 * episode left the second receipt episode of an episode-carrying run loose
		 * (the media regression: `asset-j` stayed painted), so `internal == null`
		 * is what distinguishes the bar that must stay the same element.
		 */
		if ([...ordinarySlots].some((i) => i !== gateIndex)) {
			segments[gateIndex] = { ...gateSegment, collapsed: false };
		}
	}

	/*
	 * Disclosure figures describe the rows THIS bar expands, not an indexed
	 * run remainder. Even an exact whole-run fact includes off-page/prior or
	 * follow-up work that is not a child of this disclosure. A very large fact
	 * must never turn [receipt, one tool] into thousands of inspectable actions.
	 * `factsOf(segRows)` is therefore authoritative for EVERY bar; a missing
	 * head keeps its duration unknown. Whole-run figures remain in openFrame's
	 * separately scoped wire facts, not in an unlabeled segment header.
	 */
	/* The turn's own totals: the run through its answer, commentary excluded. */
	const turnRows = answerAt === null ? runRows : runRows.slice(0, answerAt + 1);
	/*
	 * The run-level figures have no reader of their own today, but they are what a
	 * consumer reads to state this loaded turn's work. They stay in that same
	 * loaded population; an off-page wire total cannot be reconstructed by
	 * summing disclosure children. Head uncertainty remains explicit.
	 */
	const turnFacts = factsOf(turnRows, {
		partial: !headLoadedIsTheRunsOwn,
		headLoaded: headLoadedIsTheRunsOwn,
	});
	const hidden = segments.flatMap((segment) => segment.rows);
	return {
		key: run.key,
		run,
		factApplied: false, // No disclosure population was replaced by a wire aggregate.
		/*
		 * A bar renders iff there is something to hide, this is not the run a live
		 * turn is being written in, and the run can be honestly summarised: its
		 * opening user row is on hand, OR its closing answer is (the end-loaded
		 * rule - the bar then describes exactly the loaded span; see the header).
		 * The focus hold is applied per segment above.
		 */
		collapses: segments.some((segment) => segment.collapsed),
		recordIds: runRows.map((row) => row.record.id),
		hidden,
		gap: segments[0]?.gap ?? "item",
		facts: turnFacts,
		answerId,
		stampTs: answerAt === null ? null : (records[answerAt].ts ?? null),
		segments,
	};
}
