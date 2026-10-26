/**
 * THE TOOL-GROUP TIER (Tier A), ported from the desktop.
 * Ported from `damianvtran/local-operator-ui` at sourceHash f9f4611dd3dbcc06… (desktop MODEL SOURCES, see `parity-sources.json`):src/renderer/src/features/chat/canonical/trace-fold-model.ts:64-1378
 *
 * WHAT IT DECIDES, and why the phone needs the decision rather than a flat list:
 * which consecutive runs of a turn's actions fold into ONE line (`FOLD_MIN_ACTIONS`
 * = 3, or a trailing stretch of >= 2 receipts), what that line COUNTS (classes or
 * kinds, capped at 5 with an "and N other actions" tail), what the fold says while
 * a call is still executing, and how the latch keeps a group the reader has
 * already seen folded from dissolving when a pass re-reads the rows.
 *
 * WHAT IT RETURNS: structured units (`FoldCountSegment {key, label, count}`) beside
 * the strings the desktop paints. The phone words the same counts its own way
 * (the designer's minimal budget: count-line summaries only), so the parity
 * fixture compares `{key, count}` and never the English - see
 * `condense.test.ts` and `fixtures/condense.parity.json`.
 *
 * HOW IT DIFFERS FROM THE DESKTOP FILE, exhaustively:
 * 1. `TranscriptRecord` is `CondenseRecord`; `Row` is `condense-rows`'s.
 * 2. The media family is absent (`foldImages`, `foldMediaSlots/Rows`, `foldMediaClause`,
 *    `FOLD_MEDIA_LIMIT`) and so is the foot line (`turnFeet`): relay tool rows
 *    carry no image refs (only user and hub rows do) and the surface has no foot
 *    line, so porting them would be code no producer can reach.
 * 3. `foldLive` returns the EXECUTING ACTION'S INDEX, not a composed `{verb,
 *    object}`: the live clause must say exactly what the row beside it says, and
 *    this surface's row says `tool_name` + `summary`, not the desktop's verb
 *    vocabulary. The rule (the LAST executing call, and only while it is
 *    executing) is unchanged; only the wording is left to the surface.
 * 4. `op` (the operation token from a call's arguments, `todo view` vs `todo
 *    write`) is whatever the caller passes - this wire strips `details` under
 *    frame pressure, so the port never derives it here. Named in the skip-list.
 */

import type { Row } from "./condense-rows";
import {
	type InternalReceiptFamily,
	displayName,
	receiptFamilyOf,
} from "./condense-predicates";
export const FOLD_MIN_ACTIONS = 3;

/**
 * The trailing-receipt threshold: a run whose TRAILING stretch of adjacent
 * receipts is at least this long folds under one header, whatever its action
 * count.
 *
 * WHY A SECOND THRESHOLD RATHER THAN COUNTING RECEIPTS AS ACTIONS. A receipt is
 * not work the turn DID - it is the turn being told something - so it must
 * never reach the action threshold, the action count, failures, duration or the
 * live clause (see `foldRuns`). But the operator's wall of `job result` rows
 * proved the converse too: a stretch of receipts with no calls between them is
 * a wall of identical standalone rows with one chevron each, and folding it is
 * the whole point of this tier. TWO is the smallest run that has something to
 * summarise; a LONE receipt stays an ordinary row, which is the existing
 * "a lone receipt is content" policy (`scripts/canonical-notice.test.mjs`).
 *
 * WHY THE STRETCH MUST BE THE RUN'S TRAILING ONE, never a count of adjacent
 * receipt pairs anywhere in the run. The 129-owner shape in
 * `scripts/turn-collapse-behaviour.test.mjs` is the failure: after its prepend
 * the one-action run is `[peer, peer, tool]`, so two receipts at its HEAD
 * licensed a fold that hid the call the reader's opened turn bar is pinned to
 * mount. A receipt at the tail is what the reader is being told AFTER the work;
 * folding that is what the wall needs, and a head receipt rides along only when
 * the run already folds on §E2's action threshold (which is why
 * `[peer, tool, wake]` and `[peer, tool, peer]` stay ordinary rows too: their
 * trailing stretch is one).
 */
export const FOLD_MIN_RECEIPTS = 2;

export type FoldableAction = {
	/** The ledger's own name for the call (`ledgerName`), already case-folded. */
	name: string;
	failed: boolean;
	/** Seconds, when the row carries one. */
	durationS?: number | null;
	/**
	 * The row's own object text — the words its row prints beside the verb
	 * (`summaryFromArgs`, or the composing/queued status). The fold's live clause
	 * names a running call with them, and they come through an option rather than
	 * from `name` so the fold shows the row's words rather than a second guess at
	 * them.
	 */
	summary?: string;
	/**
	 * The call's operation token (`toolOp`), when its arguments name one.
	 *
	 * The live clause lifts the row's own label (`toolRowLabel`), and that
	 * composition reads the op now: without it a running `agent` op=show call
	 * would be named with the generic verb while the row beside it said `Viewing
	 * agent designer` - the two surfaces disagreeing, which is exactly what
	 * lifting the label exists to prevent. Absent when the arguments are not in
	 * hand, which keeps the generic verb.
	 */
	op?: string;
	/**
	 * The row's own running predicate: `phase !== "done"`.
	 *
	 * The row's word, so the fold cannot disagree with the row about whether a
	 * call is finished: it blocks the condense (a fold must not hide a live call)
	 * and it is what makes the span's end "now".
	 */
	running?: boolean;
	/**
	 * The row's own EXECUTING predicate: `phase === "running"`. Narrower than
	 * `running`, which also covers `composing` (the model is still dictating the
	 * arguments) and `queued` (dictation over, execution not begun). It is the
	 * live clause's predicate rather than the condense guard's: see `foldLive`.
	 */
	executing?: boolean;
	/**
	 * Ms epoch when the call began EXECUTING (`startedAt`), which the live layer
	 * clears once the call settles.
	 */
	startedAtMs?: number | null;
	/**
	 * Ms epoch when the call completed (`endedAt`), which the live layer stamps
	 * only from its own end frame.
	 */
	endedAtMs?: number | null;
};

export type FoldGroup =
	| { kind: "row"; row: Row }
	| {
			kind: "run";
			/**
			 * Stable identity for the fold, derived from its FIRST row: the fold's
			 * place in the transcript is that row's place, so it cannot move when the
			 * run grows.
			 */
			id: string;
			/**
			 * The gap tier of the run's FIRST row, which is the fold's own margin:
			 * when the run opens a turn this is the turn's 32px, and the first row
			 * inside the fold is drawn at the trace tier instead of carrying it.
			 */
			gap: Row["gap"];
			rows: Row[];
			actions: FoldableAction[];
			/**
			 * The header's summary as its units AND its shape: the count line's segments
			 * and its `and N other actions` tail, or the one flowing clause of the sentence
			 * form (see `foldSummarySpec` for why the shape has to be stated rather than
			 * inferred).
			 */
			summary: FoldSummarySpec;
			failedCount: number;
			/** The wall-clock span, when the run can date itself (see `foldSpan`). */
			span: FoldSpan | null;
			/** The in-flight call, when one is inside this run (see `foldLive`). */
			live: FoldLive | null;
	  };

/**
 * The receipt families' units for one run's header, in a stable order.
 *
 * WHY THE HEADER COUNTS BY FAMILY AND NOT ONE `incoming messages` TOTAL. The
 * old line called every internal receipt an incoming message, which was true
 * while `isInternalReceipt` covered only peer/hub traffic. It is FALSE the
 * moment a job result folds: a `job result` is not a message from anyone, and a
 * reader told `4 incoming messages` cannot know which of the rows behind the
 * fold will be a conversation and which a finished call. Each family therefore
 * states its own noun, and the order is fixed (incoming, job results, wakes,
 * monitor prompts) so the same group does not reword itself as rows arrive.
 *
 * A FAMILY WITH ZERO ROWS CONTRIBUTES NO UNIT - never `0 job results`. The two
 * markers are historical intent (`Reply requested`, `Steered`) and ride the
 * existing grammar; they are not counts and are omitted when absent.
 *
 * INVARIANT (asserted): every receipt row in `rows` is counted exactly once
 * across the family units, so the header's numbers are the expanded group's
 * contents. Arbitrary custom notices never reach here: `isReceipt` decides the
 * membership structurally (see `receiptFamilyOf`). A family the table does not
 * know is counted under `other receipts` rather than dropped.
 */
export function receiptFoldUnits(rows: readonly Row[]): string[] {
	/*
	 * A RECORD KEYED BY FAMILY, and a fallback for anything the family table does
	 * not know yet: a future receipt kind that `receiptFamilyOf` learns must not
	 * silently vanish from the header (the counts would stop summing to the rows
	 * behind the fold). Known families bump their own bucket; whatever is left
	 * over is a receipt no family claims, stated as `N other receipts`.
	 */
	const counts: Record<InternalReceiptFamily, number> = {
		incoming: 0,
		job_result: 0,
		wake: 0,
		monitor_prompt: 0,
	};
	let replyRequested = false;
	let steered = false;
	for (const { record } of rows) {
		const family = receiptFamilyOf(record);
		if (family) counts[family] += 1;
		if (record.kind === "custom") {
			if (record.receipt?.expectsReply === true) replyRequested = true;
			if (record.receipt?.steer === true) steered = true;
		}
	}
	const units: string[] = [];
	const count = (n: number, one: string, many: string) => {
		if (n > 0) units.push(`${n} ${n === 1 ? one : many}`);
	};
	count(counts.incoming, "incoming message", "incoming messages");
	count(counts.job_result, "job result", "job results");
	count(counts.wake, "wake receipt", "wake receipts");
	count(counts.monitor_prompt, "monitor prompt", "monitor prompts");
	count(
		rows.length -
			counts.incoming -
			counts.job_result -
			counts.wake -
			counts.monitor_prompt,
		"other receipt",
		"other receipts",
	);
	if (replyRequested) units.push("Reply requested");
	if (steered) units.push("Steered");
	return units;
}

/**
 * The run's wall-clock span: FIRST start to LAST completion.
 *
 * The operator's definition, and the right one: a sum of measured call durations
 * ignores the gaps between calls and reads `0s` for a run of fast ones, while
 * the span is what "how long was spent in this section" means. `endedAtMs` is
 * null while the run is still in flight — the caller computes the end against
 * its own clock then — and `running` says which of the two the caller is
 * looking at. Emitted ONLY when the run can date itself: no stamps, no span, and
 * the header renders nothing rather than a made-up `0s`.
 */
export type FoldSpan = {
	/** Earliest known execution start across the run, ms epoch. */
	startedAtMs: number;
	/** Latest known completion, ms epoch; null until one action has completed. */
	endedAtMs: number | null;
	/** Some call in the run has not settled: the span runs to `now`. */
	running: boolean;
};

/**
 * The call the condensed header names while its run is in flight.
 *
 * The same two columns `ToolRow` paints (`toolRowLabel`), lifted so a collapsed
 * group still answers "what is it doing right now?" — the operator's sharpest
 * point, because the answer is usually a long-blocking command (`wait`) that a
 * bare count hides entirely.
 */
export type FoldLive = {
	/** Index into the run's own `actions` of the call that is EXECUTING. */
	index: number;
};

/**
 * The class a ledger name belongs to, in §E2's own vocabulary.
 *
 * `null` is a REAL answer: a name this table does not know must not be filed
 * into a class, because the summary is a claim about what the turn did and a
 * wrong class is a wrong claim. Unknown names still fold - they just count by
 * KIND instead of toward a class sentence (`2 files · 1 create_issue`), so the
 * summary never files them into somebody else's activity.
 *
 * `agent` is deliberately NOT in the delegated class, and the reason is the
 * operator's report of 2026-09-27: the word claimed a hand-off for calls that
 * delegated nothing - `list`, `show`, `search` and `sync` are registry reads -
 * and one tool spans both directions, so no class sentence is true for all
 * eight of its ops. Only `task` and `delegate` hand work off; the meta tools
 * count by kind under their own nouns (`KIND_NOUNS`).
 */
export const actionClass = (
	name: string,
): "files" | "searches" | "commands" | "edits" | "web" | "delegated" | null => {
	// Lowercased here rather than at every call site: the ledger's names arrive in
	// both cases (`Web_Fetch` is a real wire name), and a case-sensitive table
	// silently classified it as unknown.
	const n = name.toLowerCase();
	if (
		n === "read" ||
		n === "grep" ||
		n === "glob" ||
		n === "ls" ||
		n === "list"
	)
		return "files";
	if (n === "search_the_web" || n === "web_search" || n === "search")
		return "web";
	if (n === "bash" || n === "exec" || n === "shell" || n === "run")
		return "commands";
	if (n === "edit" || n === "write" || n === "multiedit") return "edits";
	if (n === "task" || n === "delegate") return "delegated";
	if (n === "fetch" || n === "web_fetch" || n === "fetch_url")
		return "searches";
	return null;
};

/**
 * §E2's copy, generated from the counts by class.
 *
 * A single class is named ("Ran 4 commands", "Edited 2 files"). Two phraseable
 * classes are joined ("Explored 4 files, 1 search"), which is the spec's own
 * example and the shape a reader can act on. Anything else - three classes, or
 * any action this table cannot classify - falls to `foldCounts`: the counts by
 * KIND, because a bare total (`4 actions`) tells a reader nothing they can act
 * on and the operator asked for the shape of the work instead ("3 shell · 1
 * python").
 */
/**
 * The summary AS THE HEADER MUST PAINT IT: its units, and which SHAPE it is.
 *
 * WHY THE SHAPE IS PART OF THE CONTRACT (agent review round 2, R2-1, and the same
 * defect QA bounded as Q-r2-3): the count line's units have to hold together - a
 * wrap there may only fall at a `·` - so the header paints each one unbreakable.
 * The SENTENCE is the opposite case: `Explored 4 files, delegated 3 tasks` is
 * prose, its own word spaces are the right places to break, and that is how it
 * painted before this branch. Round 1's first cut painted every unit nowrap,
 * which silently made the sentence unbreakable and left the `break-words` on the
 * span inert (it cannot act inside `white-space: nowrap`). The header is told
 * which shape it has rather than inferring it from the unit count: `6 searches`
 * is one unit too, and it must stay whole.
 */
export type FoldSummarySpec = {
	units: string[];
	/**
	 * THE SAME SUMMARY AS STRUCTURED COUNT SEGMENTS, beside the words.
	 *
	 * WHY A SECOND FIELD RATHER THAN PARSING `units`. A cross-client parity
	 * fixture cannot compare English: another surface owns its own vocabulary for
	 * the same counts (the phone says `3 tool calls` where this app says
	 * `Explored 3 files`), and asserting a shared string would either forbid that
	 * or silently drift. The identifier and the number are the shared facts; the
	 * words are per-surface. `class:agent`-style ids come from the same derivation
	 * that builds the sentence above, so the two can never disagree, and
	 * `generateCondenseFixture` (scripts/traces) is the only reader today.
	 *
	 * It is the CLASS/noun line (`foldCountSegments`) in EVERY case, including the
	 * two-class sentence form, because a surface that renders only the count line
	 * must still be comparable to one that renders the sentence.
	 */
	countUnits: FoldCountSegment[];
	/**
	 * True for the SENTENCE form (a single flowing clause): the browser's own
	 * line breaking is correct for it, so the header does NOT hold it whole.
	 */
	prose: boolean;
};

export const foldSummarySpec = (actions: FoldableAction[]): FoldSummarySpec => {
	const counts = new Map<string, number>();
	let unknown = 0;
	for (const action of actions) {
		const cls = actionClass(action.name);
		if (cls === null) unknown += 1;
		else counts.set(cls, (counts.get(cls) ?? 0) + 1);
	}
	const segments = foldCountSegments(actions);
	if (unknown > 0 || counts.size > 2) {
		const units = foldCountUnits(segments);
		// Only reachable for an empty run, which `foldRuns` never emits a summary for.
		return {
			units: units.length === 0 ? [`${actions.length} actions`] : units,
			prose: false,
			countUnits: segments,
		};
	}

	/*
	 * Ordered by §E2's own sentence order, not by the order the calls happened to
	 * arrive: "Explored 4 files, 1 search" reads as one activity, and a summary
	 * whose wording depended on which call came first would change under a reader
	 * who had not changed anything.
	 */
	const order = ["files", "searches", "web", "commands", "edits", "delegated"];
	const parts: string[] = [];
	for (const cls of order) {
		const count = counts.get(cls) ?? 0;
		if (count === 0) continue;
		if (cls === "files")
			parts.push(`Explored ${count} file${count === 1 ? "" : "s"}`);
		if (cls === "searches")
			parts.push(`${count} search${count === 1 ? "" : "es"}`);
		if (cls === "web")
			parts.push(`Searched the web ${count} time${count === 1 ? "" : "s"}`);
		if (cls === "commands")
			parts.push(`Ran ${count} command${count === 1 ? "" : "s"}`);
		if (cls === "edits")
			parts.push(`Edited ${count} file${count === 1 ? "" : "s"}`);
		if (cls === "delegated")
			parts.push(`Delegated ${count} task${count === 1 ? "" : "s"}`);
	}
	if (parts.length === 0)
		return {
			units: [`${actions.length} actions`],
			prose: true,
			countUnits: segments,
		};
	// A two-class sentence reads as "Explored 4 files, 1 search"; the second part
	// keeps the join's own lowercase, which is why the first part is the only one
	// that capitalises.
	return {
		units: [
			parts.length === 1
				? parts[0]
				: `${parts[0]}, ${parts[1].charAt(0).toLowerCase()}${parts[1].slice(1)}`,
		],
		prose: true,
		countUnits: segments,
	};
};

/** The units the header paints, in order. */
export const foldSummaryUnits = (actions: FoldableAction[]): string[] =>
	foldSummarySpec(actions).units;

/** The units joined with the ` · ` the renderer draws between them. */
export function foldSummary(actions: FoldableAction[]): string {
	return foldSummaryUnits(actions).join(" · ");
}

/**
 * The per-KIND count line: `3 shell · 1 python`, `2 files · 1 shell`.
 *
 * The fallback's replacement for a bare total, in the operator's own example -
 * `3 shell · 1 python`. Two vocabularies, both the app's:
 *
 * - the command kinds are named by their RUNTIME (`shell` for `bash`/`exec`,
 *   `python` for `eval`, whose row verb `Ran Python` already carries the word),
 *   because merging them into "commands" would hide exactly the shape the
 *   operator asked to see, and they are kind NAMES, not count nouns, so they
 *   stay singular (`3 shell`, as written);
 * - everything else counts under its class's own noun from the sentence table
 *   above (files, searches, web searches, edits, tasks), pluralized the way
 *   that sentence pluralizes it.
 *
 * An action this app cannot classify counts under its own display name rather
 * than being filed into somebody else's class: the summary is a claim about
 * what the turn did, and `2 files` for a Linear call would be a wrong claim.
 * The builtins this app KNOWS carry a noun and a plural in `KIND_NOUNS` below,
 * so their counts pluralize sensibly (`4 files · 3 agents` - the display name
 * a raw `agent` call once fell to read `3 agent`).
 *
 * Ordered by the SENTENCE's own order, never by arrival: the same three actions
 * in a different order say the same thing (the reason `foldSummary` orders its
 * parts too). Unknown kinds follow, most-frequent first, alphabetically on a
 * tie, so the line is stable instead of depending on which call came first.
 */

/**
 * The count noun for each builtin the class table does not file.
 *
 * `foldCounts` names an unclassed action by KIND, and until this table the kind
 * vocabulary was the ledger's own display name, which cannot pluralize: three
 * `agent` calls (profile READS - the operator's report of 2026-09-27) counted
 * as `3 agent`, one step after the same three calls folded as `delegated 3
 * tasks`. Each entry names what the call TOUCHED, the way `files` and `tasks`
 * do, with the plural the line needs; a name this table has never heard of (an
 * MCP tool, a tool newer than the table) still counts under its own name.
 *
 * Keys are the lowercase wire names, the same vocabulary `actionClass` reads.
 */
const KIND_NOUNS: Record<string, { noun: string; plural: string }> = {
	agent: { noun: "agent", plural: "agents" },
	team: { noun: "team", plural: "teams" },
	team_delete: { noun: "team deletion", plural: "team deletions" },
	hub: { noun: "subagent", plural: "subagents" },
	secret: { noun: "secret", plural: "secrets" },
	project: { noun: "project", plural: "projects" },
	project_delete: { noun: "project deletion", plural: "project deletions" },
	wake: { noun: "wake", plural: "wakes" },
	jobs: { noun: "job check", plural: "job checks" },
	console: { noun: "console", plural: "consoles" },
	network: { noun: "network call", plural: "network calls" },
	browser: { noun: "browser action", plural: "browser actions" },
	todo: { noun: "todo update", plural: "todo updates" },
	send: { noun: "message", plural: "messages" },
	ask: { noun: "question", plural: "questions" },
	wait: { noun: "wait", plural: "waits" },
	lsp: { noun: "code lookup", plural: "code lookups" },
	web_read: { noun: "page", plural: "pages" },
	list_variables: { noun: "variable lookup", plural: "variable lookups" },
	read_variable: { noun: "variable read", plural: "variable reads" },
	sessions: { noun: "session", plural: "sessions" },
};

/**
 * The cap on the unique action-type segments ONE count line shows, and the
 * tail token that stands in for everything past it.
 *
 * WHY IT EXISTS (operator report, 2026-10-01, relayed by Aida): a long run's
 * header read `6 searches · 1 task · 2 browser actions · 1 ai_search · 1
 * get_tool_access · 1 query_data_sources · 1 todo update · 1 wait · 1
 * workspace_get_gmail_thread_content` - nine unique action types, wider than
 * the column at every realistic window. The operator's ask was a cap: "how
 * many unique action types will show up per line", with everything past the
 * majority classes summarised as `and N other actions`.
 *
 * THE TAIL COUNTS CALLS, NOT TYPES, and the same unit every other number on
 * the line counts (the bar's `N actions`, the foot's `N actions`), so a reader
 * who sums the segments still reaches the turn's action count; the number of
 * hidden TYPES is deliberately not stated, because `and N other actions` is a
 * measure of work and the expanded rows are the lossless record. `1`
 * singularises, like every `fact` on the line.
 *
 * WHAT IS KEPT IS THE FIRST `FOLD_COUNT_LIMIT` UNITS OF THE LINE'S OWN ORDER, and
 * that order is the `order` array below and nothing else: the sentence's class
 * slots (files, searches, web, the command kinds `shell`/`python` in the
 * `commands` slot, edits, delegated), then the three named meta kinds the
 * operator's report used (`agent`, `team`, `hub`); every other kind follows
 * those, sorted by count and then by its noun. So the tail is always what the
 * line's own order already ranked least-major - which is also why a run with
 * many `agent` calls can push a rarer named kind into the tail - and at N or
 * fewer units the line is byte-identical to what it has always been. Five is
 * chosen against the report's own run: it keeps the three classes the run was
 * mostly made of plus two named singles, and the capped line fits the standard
 * column one line while still wrapping at the 640px window (the frames under
 * `docs/evidence/chat-trace-fold/` and its `-before` sibling).
 */
export const FOLD_COUNT_LIMIT = 5;

/**
 * The count line's UNITS, capped at `FOLD_COUNT_LIMIT` with the rest folded into
 * `and N other actions` (see the constant above).
 *
 * WHY UNITS RATHER THAN ONE STRING (design round 1, D1; agent review R1-4, the
 * same defect's code site): a joined sentence gives the browser the spaces around
 * every `·` AND the spaces inside every phrase, so at a narrow column it breaks
 * wherever it likes - the 640px frame broke `and 4 other actions` between the
 * numeral and its noun and left `other actions` alone on line 2. The units are
 * what the line is COMPOSED of, so the renderer can paint each one unbreakable
 * and let the breaks fall at the separators, which is the only boundary a reader
 * recognises here. The joined string is still `join(" · ")` of these units, which
 * is what the bar's `title` and every non-DOM consumer read.
 */
export const foldCountUnits = (
	segments: { label: string; count: number }[],
): string[] => {
	if (segments.length <= FOLD_COUNT_LIMIT)
		return segments.map((segment) => segment.label);
	const kept = segments.slice(0, FOLD_COUNT_LIMIT);
	const other = segments
		.slice(FOLD_COUNT_LIMIT)
		.reduce((total, segment) => total + segment.count, 0);
	return [
		...kept.map((segment) => segment.label),
		/*
		 * ONE unit, and it must stay one: the phrase is a single fact about the
		 * work the cap hid, so it is never broken between `and 4` and `other
		 * actions` - that break is the reader-visible defect D1 measured.
		 */
		`and ${other} other action${other === 1 ? "" : "s"}`,
	];
};

/**
 * ONE COUNT-LINE SEGMENT: the shared decision, plus this surface's word for it.
 *
 * `key` and `count` are the cross-client facts (`kind:shell`, `class:files`,
 * `name:<display name>`); `label` is this app's own phrasing of them. A surface
 * with a different vocabulary renders `key` through its own catalogue and
 * compares only `{key, count}` against a parity fixture - see `FoldSummarySpec`.
 */
export type FoldCountSegment = {
	/** The stable identifier: `class:<class>`, `kind:<tool>`, or `name:<label>`. */
	key: string;
	/** This app's words for the key and count (`3 shell`, `2 files`). */
	label: string;
	count: number;
};

/**
 * The per-kind segments of a run's count line, in the line's own order and NOT
 * yet capped - `foldCountUnits` applies the cap.
 */
export const foldCountSegments = (
	actions: FoldableAction[],
): FoldCountSegment[] => {
	type Kind = { noun: string; plural: string | null };
	const counts = new Map<string, { kind: Kind; count: number }>();
	const bump = (key: string, kind: Kind) => {
		const seen = counts.get(key);
		if (seen) seen.count += 1;
		else counts.set(key, { kind, count: 1 });
	};
	for (const action of actions) {
		const n = action.name.toLowerCase();
		if (n === "bash" || n === "exec" || n === "shell" || n === "run") {
			bump("kind:shell", { noun: "shell", plural: null });
			continue;
		}
		if (n === "eval") {
			bump("kind:python", { noun: "python", plural: null });
			continue;
		}
		const cls = actionClass(action.name);
		if (cls === "files") bump("class:files", { noun: "file", plural: "files" });
		else if (cls === "searches")
			bump("class:searches", { noun: "search", plural: "searches" });
		else if (cls === "web")
			bump("class:web", { noun: "web search", plural: "web searches" });
		else if (cls === "edits")
			bump("class:edits", { noun: "edit", plural: "edits" });
		else if (cls === "delegated")
			bump("class:delegated", { noun: "task", plural: "tasks" });
		else if (n === "todo") {
			/*
			 * `todo view` is a READ and the five write ops are not; one noun for
			 * the family counted a view as an update (`2 todo updates` over a
			 * `Read todos` row - review round 1, R1-6), the same false-claim
			 * species the operator's report closed. The op tells them apart and
			 * the split keeps the fold and the row saying the same thing; an
			 * op-less call is the write noun the family already had.
			 */
			if ((action.op ?? "").toLowerCase() === "view")
				bump("kind:todo:view", { noun: "todo read", plural: "todo reads" });
			else bump("kind:todo", KIND_NOUNS.todo);
		} else if (n === "jobs") {
			/*
			 * A `jobs cancel` is a CANCELLATION, not a check (review round 1,
			 * R1-1): the receipt family prints `N job results`, and `1 job check`
			 * for a call that stopped a job claims the run looked at one. `list`,
			 * `peek` and an op-less call keep the check noun.
			 */
			if ((action.op ?? "").toLowerCase() === "cancel")
				bump("kind:jobs:cancel", {
					noun: "job cancellation",
					plural: "job cancellations",
				});
			else bump("kind:jobs", KIND_NOUNS.jobs);
		} else {
			const noun = KIND_NOUNS[n];
			if (noun) bump(`kind:${n}`, noun);
			else {
				const name = displayName(action.name);
				bump(`name:${name}`, { noun: name, plural: null });
			}
		}
	}
	/*
	 * The sentence's order, with the command kinds in the slot `commands` held:
	 * files, searches, web, shell, python, edits, tasks - then the meta kinds
	 * the operator's report named (`4 files · 3 agents`), which sit after the
	 * classes so a run that delegated AND read profiles states the delegation
	 * first.
	 */
	const order = [
		"class:files",
		"class:searches",
		"class:web",
		"kind:shell",
		"kind:python",
		"class:edits",
		"class:delegated",
		"kind:agent",
		"kind:team",
		"kind:hub",
	];
	const segments: FoldCountSegment[] = [];
	const fact = (count: number, kind: Kind) =>
		`${count} ${count === 1 || kind.plural === null ? kind.noun : kind.plural}`;
	for (const key of order) {
		const seen = counts.get(key);
		if (seen)
			segments.push({
				key,
				label: fact(seen.count, seen.kind),
				count: seen.count,
			});
	}
	const rest = [...counts.entries()]
		.filter(([key]) => !order.includes(key))
		.sort(
			([, a], [, b]) =>
				b.count - a.count || a.kind.noun.localeCompare(b.kind.noun),
		);
	for (const [key, seen] of rest)
		segments.push({
			key,
			label: fact(seen.count, seen.kind),
			count: seen.count,
		});
	return segments;
};

/**
 * The count line as the ONE string its non-DOM consumers read (the collapsed
 * bar's `title`): the units joined by the same ` · ` the renderer draws between
 * them, so the words a reader sees and the words a tooltip carries cannot say
 * different things.
 */
export function foldCounts(actions: FoldableAction[]): string {
	const segments = foldCountSegments(actions);
	// Only reachable for an empty run, which `foldRuns` never emits a summary for.
	if (segments.length === 0) return `${actions.length} actions`;
	return foldCountUnits(segments).join(" · ");
}

/**
 * The wall-clock bounds of ONE action, or null when it cannot date itself.
 *
 * A settled action's start is RECONSTRUCTED as `endedAt - durationS`: the
 * record documents those as two readings of the same span ("the execution
 * start, not the compose start, so it measures the same span the settled
 * `durationS` reports"), and the live layer clears `startedAt` when a call
 * settles so the ticking clock stops — this is where the start comes back from.
 * A call with no stamp on either side contributes NOTHING rather than a guess:
 * the span this feeds is a claim about time, and an invented zero is the claim
 * the operator reported as broken.
 */
function actionBounds(
	action: FoldableAction,
): { startMs: number; endMs: number | null } | null {
	if (action.running === true) {
		if (typeof action.startedAtMs !== "number") return null;
		return { startMs: action.startedAtMs, endMs: null };
	}
	if (
		typeof action.endedAtMs !== "number" ||
		typeof action.durationS !== "number"
	)
		return null;
	return {
		startMs: action.endedAtMs - action.durationS * 1000,
		endMs: action.endedAtMs,
	};
}

/**
 * The run's wall-clock span: FIRST start to LAST completion, or null when the
 * run cannot date itself.
 *
 * The operator's definition, and the right one: a SUM of the measured call
 * durations ignores the gaps between calls and reads `0s` for a run of fast
 * ones — while "how long was spent in this action group" is exactly the span
 * those gaps live in. The span is emitted only when at least one action carries
 * a start and either one carries a completion or the run is still in flight;
 * an older row restored from history carries durations but NO stamps (the
 * durable tool shape persists `duration_s` and no times), so its run shows no
 * clock at all rather than a made-up zero — the header's rule is "nothing when
 * we do not know", the same one `DiffCounters` states for `+0`.
 *
 * `running` is true while ANY action has not settled, and it is deliberately
 * the row's own predicate (`phase !== "done"`): the numbers between calls are
 * part of the section too, so the clock keeps running until the run is over and
 * then freezes at its last completion — it never ticks backward.
 */
export function foldSpan(actions: FoldableAction[]): FoldSpan | null {
	let startedAtMs: number | null = null;
	let endedAtMs: number | null = null;
	let running = false;
	for (const action of actions) {
		if (action.running === true) running = true;
		const bounds = actionBounds(action);
		if (!bounds) continue;
		startedAtMs =
			startedAtMs === null
				? bounds.startMs
				: Math.min(startedAtMs, bounds.startMs);
		if (bounds.endMs !== null)
			endedAtMs =
				endedAtMs === null ? bounds.endMs : Math.max(endedAtMs, bounds.endMs);
	}
	if (startedAtMs === null) return null;
	if (endedAtMs === null && !running) return null;
	return { startedAtMs, endedAtMs, running };
}

/**
 * The call the condensed header names while the run is in flight, or null when
 * every call has settled.
 *
 * The LAST unsettled action is the one being watched — calls run one at a time,
 * so it is the newest activity in the run — and its label is assembled through
 * `toolRowLabel`, the same composition `ToolRow` paints, so the collapsed group
 * and the expanded row cannot say two different things. No output fallback is
 * passed: a call that has not settled has no output to stand in with.
 *
 * NOT the fold's settle predicate on its own (review round 2, R7): after U2 the
 * null means "no call EXECUTING", and a composing or queued call is unsettled
 * yet unnamed. The component closes on `!sectionLive && live === null`, so those
 * phases are held by the section's own liveness rather than by this value; what
 * this value alone closes is the call executing as the section ends.
 */
export function foldLive(actions: FoldableAction[]): FoldLive | null {
	/*
	 * NAMED ONLY WHEN IT IS EXECUTING, and the last such call wins.
	 *
	 * `running` is deliberately wider than this predicate, and the first cut of
	 * this function used it: a composing row (the model still dictating arguments)
	 * or a queued one (dictation over, execution not begun) then put its own
	 * status text in the header - `Running composing`, `Running queued · 22 B` -
	 * for the sub-second windows before its name resolves. That reads as a claim
	 * the app cannot back (nothing is running yet) about the wire's byte count,
	 * in the exact window the operator looks at the header for (UX round 1, U2).
	 * The row still states those phases in its own column; the header names a call
	 * when there is a call to name.
	 *
	 * The LAST executing call, not the first: a settled batch can leave one call
	 * executing while a later sibling composes, and the reader's question is
	 * "what is it doing now", which the newest executing call answers.
	 */
	for (let index = actions.length - 1; index >= 0; index -= 1) {
		if (actions[index].executing !== true) continue;
		/*
		 * THE INDEX, NOT THE SENTENCE. The desktop composes `{verb, object}` from
		 * its own verb vocabulary (`Running pnpm test`); this surface's row prints
		 * `tool_name` + `summary` instead, and the fold's live clause must say
		 * exactly what the row beside it says. So the model states WHICH action is
		 * executing and the surface names it.
		 */
		return { index };
	}
	return null;
}

/**
 * How many receipts the run ENDS with, counting back from the last row.
 *
 * This is the licensing stretch for `FOLD_MIN_RECEIPTS`, and it is deliberately
 * TRAILING-only: the constant's comment records the 129-owner failure a
 * whole-run adjacency count caused. Any non-receipt row stops the count, so a
 * receipt at the HEAD of a run licenses nothing by itself.
 */
export function trailingReceiptStretch(
	rows: readonly Row[],
	isReceipt: (row: Row) => boolean,
): number {
	let stretch = 0;
	for (let index = rows.length - 1; index >= 0; index -= 1) {
		if (!isReceipt(rows[index])) break;
		stretch += 1;
	}
	return stretch;
}

/**
 * The rows, split into single rows and folds.
 *
 * A run is consecutive SAME-TURN actions: a row whose `gap` is `"turn"` opens a
 * new turn and therefore a new run, which is how the boundary between two turns
 * is respected without this model knowing anything about turns. Rows the
 * transcript does not treat as trace-like (prose and notices) break a run.
 * Positively classified internal receipts may bridge calls, but are excluded
 * from the action threshold, work duration, failures and live execution. This
 * preserves chronology without claiming that a received message was an action.
 * A run that ENDS with two or more adjacent receipts folds on its own (see
 * `FOLD_MIN_RECEIPTS` and `trailingReceiptStretch`), which is what stops a wall
 * of job results from painting one row per receipt without folding the call a
 * reader's opened turn bar is pinned to mount.
 */
export function foldRuns(
	rows: Row[],
	options: {
		/** `ledgerName` for a row, injected so this module stays pure. */
		nameOf: (row: Row) => string;
		failedOf: (row: Row) => boolean;
		durationOf?: (row: Row) => number | null;
		/** The row's own object text, for the fold's live clause. */
		summaryOf?: (row: Row) => string;
		/**
		 * The row's operation token (`toolOp`), for the live clause: the clause
		 * lifts the row's own label, and an op-aware tool must be named with the
		 * verb its row prints (`Viewing agent designer`, never `Delegating`).
		 */
		opOf?: (row: Row) => string;
		/** The row's own running predicate: `phase !== "done"`. */
		runningOf?: (row: Row) => boolean;
		/**
		 * The row's EXECUTING predicate (`phase === "running"`), which feeds the
		 * live clause and nothing else: a composing or queued call has no name to
		 * paint yet, and the clause waits for one (see `foldLive`).
		 */
		executingOf?: (row: Row) => boolean;
		/** Ms epoch the call began executing, when the row carries one. */
		startedAtOf?: (row: Row) => number | null;
		/** Ms epoch the call completed, when the row carries one. */
		endedAtOf?: (row: Row) => number | null;
		isFoldable: (row: Row) => boolean;
		/** Receipts may bridge calls but never contribute action arithmetic. */
		isReceipt?: (row: Row) => boolean;
		receiptSummaryOf?: (rows: Row[]) => string[];
		/**
		 * Whether the turn this list belongs to is STILL BEING WRITTEN.
		 *
		 * It exists for one rule, `latchedFoldIds`: a live run's fold must not be
		 * dissolved by the next call. See that option's note. Absent (the settled
		 * default, and every pure test) keeps the threshold-only behaviour exactly.
		 *
		 * Named `liveTurn`, not `live`: `FoldGroup.live` is the condensed header's
		 * live clause, and one word for two different facts is how a later reader
		 * wires the wrong one in (agent review round 1, F5).
		 */
		liveTurn?: boolean;
		/**
		 * Ids of rows this pane has ALREADY PAINTED folded, while the turn is live.
		 *
		 * WHY A LATCH AND NOT ANOTHER THRESHOLD (operator report, 2026-10-21: job
		 * messages and responses "alternate between forcing the tool group open and
		 * then closing it back again"). `FOLD_MIN_RECEIPTS` licenses a fold on the
		 * run's TRAILING receipt stretch, which is positional by design (the head-pair
		 * negatives below must stay ordinary rows). A positional rule is not
		 * monotonic under append: `[t1, j1, j2]` folds, `[t1, j1, j2, t2]` does not
		 * (the new call ends the stretch), `[t1, j1, j2, t2, j3, j4]` folds again - so
		 * every job response close to a call made the group open and close in turn,
		 * with no reader action.
		 *
		 * The latch is the one fact the pure threshold cannot carry: this run was
		 * already shown to the reader as a fold. While the turn is live it keeps
		 * folding, so the group can only ever GROW; at settle the latch is no longer
		 * consulted and the final shape comes from the final rows alone, which is why
		 * every settled negative in `trace-fold-model.test.mjs` still holds (the
		 * head-pair shapes are judged on the completed list, where no latch applies).
		 */
		latchedFoldIds?: ReadonlySet<string>;
	},
): FoldGroup[] {
	const groups: FoldGroup[] = [];
	let run: Row[] = [];
	const flush = () => {
		if (run.length === 0) return;
		const receiptRows = run.filter((row) => options.isReceipt?.(row));
		const actions: FoldableAction[] = run
			.filter((row) => options.isFoldable(row))
			.map((row) => {
				const action: FoldableAction = {
					name: options.nameOf(row),
					failed: options.failedOf(row),
					durationS: options.durationOf ? options.durationOf(row) : null,
					startedAtMs: options.startedAtOf ? options.startedAtOf(row) : null,
					endedAtMs: options.endedAtOf ? options.endedAtOf(row) : null,
				};
				if (options.summaryOf) action.summary = options.summaryOf(row);
				if (options.opOf) action.op = options.opOf(row);
				if (options.runningOf) action.running = options.runningOf(row);
				if (options.executingOf) action.executing = options.executingOf(row);
				return action;
			});
		/*
		 * THE HEADER'S RECEIPT UNITS. The option owns the family grammar
		 * (`receiptFoldUnits` at the call site); the fallback keeps a caller that
		 * classifies receipts but supplies no summary from painting an EMPTY header,
		 * which would be a fold with nothing to say.
		 */
		const suppliedReceiptUnits = receiptRows.length
			? (options.receiptSummaryOf?.(receiptRows) ?? [])
			: [];
		const receiptUnits =
			suppliedReceiptUnits.length > 0
				? suppliedReceiptUnits
				: receiptRows.length > 0
					? [
							`${receiptRows.length} receipt${receiptRows.length === 1 ? "" : "s"}`,
						]
					: [];
		/*
		 * A run folds on EITHER threshold: §E2's three actions, or a TRAILING
		 * stretch of two adjacent receipts (see `FOLD_MIN_RECEIPTS` and
		 * `trailingReceiptStretch`). Receipts still contribute nothing to
		 * `actions`, `failedCount`, `span` or `live` - they only license the fold
		 * and add their own units to the header. A receipt at the run's HEAD rides
		 * along only when the action threshold already folded the run.
		 */
		const trailingReceipts = options.isReceipt
			? trailingReceiptStretch(run, options.isReceipt)
			: 0;
		/*
		 * THE LIVE LATCH (see `latchedFoldIds`). A run this pane has already painted
		 * folded keeps folding while the turn is being written, so a call arriving
		 * after a receipt wall cannot dissolve the group it belongs to. Settled
		 * passes never consult it.
		 */
		const latchedHere =
			options.liveTurn === true &&
			options.latchedFoldIds !== undefined &&
			run.some((row) => options.latchedFoldIds?.has(row.record.id) === true);
		if (
			actions.length >= FOLD_MIN_ACTIONS ||
			trailingReceipts >= FOLD_MIN_RECEIPTS ||
			latchedHere
		) {
			/*
			 * THE ACTION SIDE OF A MIXED LINE IS A COUNT, NOT A SENTENCE.
			 * `foldSummarySpec`'s sentence form ("Ran 3 commands") is right when it
			 * is the whole line, but ``Ran 3 commands · 3 job results`` glues prose
			 * to a noun inventory with the ledger's own separator. Once receipts
			 * join the line, the action side takes the count vocabulary the same
			 * module already uses for multi-class runs (`3 shell`), so every unit
			 * is the same shape and the ` · ` boundaries are the only grammar.
			 * A receipt-only run has no action units at all - never `0 actions`.
			 */
			const actionSpec: FoldSummarySpec = (() => {
				if (actions.length === 0)
					return { units: [], prose: false, countUnits: [] };
				/* ONE derivation of the count segments, whichever shape the words take,
				 * so `countUnits` is the same list the line is built from. */
				const countUnits = foldCountSegments(actions);
				return receiptUnits.length > 0
					? {
							units: foldCountUnits(countUnits),
							prose: false,
							countUnits,
						}
					: foldSummarySpec(actions);
			})();
			groups.push({
				kind: "run",
				id: run[0].record.id,
				gap: run[0].gap,
				rows: run,
				actions,
				summary: {
					units: [...actionSpec.units, ...receiptUnits],
					prose: actionSpec.prose,
					countUnits: actionSpec.countUnits,
				},
				failedCount: actions.filter((action) => action.failed).length,
				span: foldSpan(actions),
				live: foldLive(actions),
			});
		} else {
			for (const row of run) groups.push({ kind: "row", row });
		}
		run = [];
	};

	for (const row of rows) {
		/*
		 * A ROW THAT OPENS A TURN STARTS A NEW RUN, and it is IN that run (design
		 * round 1, D8). It used to be excluded - the run began at the turn's SECOND
		 * action - because the opening row carries the turn's 32px gap and a
		 * margin that size inside a fold's content would open a hole under the
		 * summary line. The cost was a standalone `read` row above `7 actions`
		 * while the turn's foot said `8 actions`: two counts for one turn. The gap
		 * now belongs to the GROUP (`gap` below), which the transcript paints on the
		 * fold's own wrapper, and the row inside the fold is drawn at the trace
		 * tier - so the fold holds the whole consecutive run (§E2) and its count is
		 * the foot's.
		 */
		const opensTurn = row.gap === "turn" || row.gap === "first";
		if (opensTurn) flush();
		if (options.isFoldable(row) || options.isReceipt?.(row)) {
			run.push(row);
			continue;
		}
		flush();
		groups.push({ kind: "row", row });
	}
	flush();
	return groups;
}

/**
 * The next live fold latch: what was latched before, pruned to the STORE's live
 * rows, plus every run this pass painted folded.
 *
 * WHY IT IS A UNION AND NOT A REBUILD (agent review round 1, F1). The transcript
 * window is not the conversation: a snap, a widened page or a filter can paint a
 * frame whose slice omits a run the pane has already shown folded. Rebuilding
 * the set from that slice dropped the run's ids, so the rows came back on the
 * next full pass and re-folded - the same open/close pair through another door.
 * Retaining across the shrink and pruning against `held` (the STORE's ids) is
 * what keeps the latch both sticky and bounded.
 *
 * `held` is asked per LATCHED id, never materialised as a set of the whole store
 * (agent review round 2): this runs inside a memo that re-runs on every streamed
 * token, and building a 20k-entry id set per token measured 0.677 ms against
 * 0.013 ms at 500 records - work proportional to the conversation for a fact
 * about a handful of rows. The transcript's own id index is already a `Map`, so
 * it answers `.has` directly.
 *
 * Pure so the trade is arithmetic rather than a mount: the component owns the
 * ref, this owns the rule.
 */
export function latchFoldedRuns(
	previous: ReadonlySet<string>,
	held: { has(id: string): boolean },
	runs: readonly FoldGroup[],
): Set<string> {
	const next = new Set<string>();
	for (const id of previous) if (held.has(id)) next.add(id);
	for (const group of runs)
		if (group.kind === "run")
			for (const row of group.rows) next.add(row.record.id);
	return next;
}

/**
 * How many SLOTS a condensed group's strip draws in its one row: up to four
 * tiles and the `+N` count.
 *
 * THE ROW IS THE BUDGET, and the number is measured rather than chosen. A tile is
 * 117px on an 8px gutter (`gap-2`), and the last slot is the `+N` control rather
 * than a picture: the count takes the slot the FIFTH picture would have had. The
 * narrowest column this strip renders in was measured at 556px of strip width (a
 * 640px window: 576px of column minus the strip's own 20px indent; the live
 * window's column is wider), so the arithmetic is, in strip terms:
 *
 *     4 x 117 + 4 x 8 + control = 500 + control <= 556   =>   control <= 56px
 *
 * and the control is 33.6px (`+4`), 39.3px (`+10`), 41.4px (`+99`): the row is
 * 541.4px at its worst (`+99`), 14.6px inside the column. In WINDOW terms - which
 * is how a reader meets it - the strip is the window minus 84px (the transcript's
 * 2 x `p-8` and the strip's `ml-5`), so the row needs a window of 541.4 + 84 =
 * ~626px at its worst and 533.6 + 84 = ~618px for `+4`. Against the row this
 * replaces - 98px tiles and `+N more images`, 111.1px at `+4` and 118.9px at
 * `+99` - that is 535.1 / 542.9px of strip, ~619 / ~627px of window (the UX round
 * measured the old floor between a 620px and a 600px window), so like for like the
 * floor does not rise: it falls by ~1.5px at either digit count, while the tile
 * gets 42% more area. The narrow label is what pays for it: with the full noun the
 * same 117px tile would need 4 x 117 + 32 + 118.9 = 618.9px of strip, ~703px of
 * window, which is past the 640px window this layout is modelled on.
 *
 * Because a tile is this wide, FIVE tiles (617px) no longer fit one row: the
 * cap is therefore the slot count, and a run of exactly five pictures draws four
 * tiles and a `+1`, the same shape every larger count has. Capping at all is
 * what keeps the height at one row for any count: without it a run of 25-30
 * pictures costs ~391px at the old size, more than the ~338.7px an EXPANDED group
 * costs - the one case where condensing would be the taller choice (design review
 * round 1, D3).
 */
export const FOLD_MEDIA_LIMIT = 5;

/**
 * How many tiles the strip draws, and how many the `+N` slot stands for.
 *
 * When the run produced more than a row of tiles can hold, the LAST slot is the
 * count itself rather than a fifth picture: the row stays one row either way, and
 * the reader is told how many they are not seeing instead of being left to infer
 * it from a clipped row. The count is also in the condensed header
 * (`foldMediaClause`) and in the strip's own accessible name, so no reader - with
 * or without a pointer - has to count tiles to learn it.
 *
 * `uncapped` is the U8 escape: the spanning bar's sole image-bearing group
 * shows its whole set, because the `+N more images` press that opened the bar
 * has already asked for it - see the option's own comment.
 */
export const foldMediaSlots = (
	count: number,
	{ uncapped = false } = {},
): { shown: number; more: number } =>
	uncapped
		? /* The caller asked for every picture (U8): the bar's own children's case
		   when the group IS the span's whole image story - the reader pressed
		   `+N more images` on the bar, and making them press a second time for a
		   group that holds the same set is the dead end the round-2 UX pass
		   found. The cap still bounds every strip the reader has NOT asked to
		   expand. */
			{ shown: Math.max(count, 0), more: 0 }
		: /* `<`, not `<=`: the limit counts SLOTS, and the fifth slot is the count,
		     so a run of exactly FOLD_MEDIA_LIMIT pictures no longer fits as tiles
		     (5 x 117 + 32 = 617px against the 556px column) and takes the count
		     slot as every larger run does. */
			count < FOLD_MEDIA_LIMIT
			? { shown: Math.max(count, 0), more: 0 }
			: {
					shown: FOLD_MEDIA_LIMIT - 1,
					more: count - (FOLD_MEDIA_LIMIT - 1),
				};

/**
 * How the UNCAPPED strip breaks its tiles into rows: the length of each row.
 *
 * The capped row is four tiles wide, so the uncapped strip wraps at the same four
 * and the width of a row is one constant in both. A plain four-per-row wrap strands
 * a lone tile whenever the count leaves a remainder of one (5 -> 4+1, 9 -> 4,4,1,
 * 13 -> 4,4,4,1), and a one-tile second row reads as an accident (design round,
 * D5: the old strip wrapped 7+1 at 1280px). So when the final row would hold one
 * tile, the last TWO rows are rebalanced to 3+2 (5 -> 3,2; 9 -> 4,3,2): the rule
 * is "no row of one, unless the whole strip is one tile". Returned as lengths
 * rather than a column count so the strip can place each tile explicitly: CSS
 * auto-placement cannot produce an uneven last pair.
 */
export const foldMediaRows = (count: number): number[] => {
	const perRow = FOLD_MEDIA_LIMIT - 1;
	if (count <= 0) return [];
	if (count <= perRow) return [count];
	const rows = Array.from({ length: Math.floor(count / perRow) }, () => perRow);
	const remainder = count % perRow;
	if (remainder === 1) {
		// Take one tile from the last full row so the remainder row holds two.
		rows[rows.length - 1] = perRow - 1;
		rows.push(2);
	} else if (remainder > 1) {
		rows.push(remainder);
	}
	return rows;
};

/**
 * The condensed header's clause for the media a run produced, or `null` for a run
 * that produced none.
 *
 * The strip is a PRESENCE CUE rather than a reader of the pictures - a 64px tile
 * cannot carry a label or a chart's axis, and the frames say so - so the count
 * belongs in the text layer, where it is legible at any tile size and reachable
 * without a pointer. It is also what keeps the sighted reader from getting less
 * than the screen-reader user, whose `aria-label` on the strip has carried the
 * count since the first cut (design review round 1, D3). A run with no pictures
 * gets no clause, which is what keeps that header byte-identical.
 */
export const foldMediaClause = (count: number): string | null => {
	if (count <= 0) return null;
	return `${count} image${count === 1 ? "" : "s"}`;
};

/**
 * THE ONE DEFINITION OF "HOW LONG THE WORK TOOK": the seconds a call itself
 * reported, summed (design review round 1 on #708, D1).
 *
 * The turn's foot (`Worked for 8m17s`) and every condensed bar (`Took 4m43s`) state
 * THIS quantity - model time, queueing and waiting for the reader are not in it -
 * because a ladder shows both on one screen and the action counts on them
 * reconcile exactly (236 + 101 + 78 = 415), which invites the reader to add the
 * durations too. The bars used to state a wall span (opening row to the latest end
 * instant) beside a foot stating this sum, and on the operator-shaped journal the
 * two were 16.7x apart with nothing on screen to say why. Both sides now call the
 * two functions below, so bars + the rest of the turn add up to the foot within
 * the rounding of each printed figure (`formatDuration` floors), and no third
 * derivation can reappear beside them. The words differ (`Took` / `Worked for`)
 * and are the design round's to judge; the quantity does not.
 *
 * A call that reported nothing contributes nothing, and a span in which NO call
 * reported a figure states none (null) - never a `0s` claim.
 */
export function workedSecondsOf(row: Row): number | null {
	return row.record.kind === "tool" ? row.record.durationS : null;
}

/** The sum of `workedSecondsOf` over rows; null when no row reported one. */
export function workedSeconds(rows: readonly Row[]): number | null {
	let total: number | null = null;
	for (const row of rows) {
		const seconds = workedSecondsOf(row);
		if (typeof seconds === "number") total = (total ?? 0) + seconds;
	}
	return total;
}

/** What a finished turn's foot line reports (§E3). */
export type TurnFoot = {
	actions: number;
	failed: number;
	/** Summed seconds, when the turn reported any. */
	durationS: number | null;
	/** The first failed row in the turn, for §E3's jump (U14). */
	firstFailedId: string | null;
};

/**
 * The foot line's data, per turn-closing row id.
 *
 * A turn is the rows between one `user` row and the next turn's opening, which
 * is the same fact the transcript already computes as `closesTurn`; this walks
 * the same list and accumulates up to that point, so the count a reader sees at
 * the foot is the count of rows ABOVE it rather than of the whole conversation.
 */
export function turnFeet(
	rows: Row[],
	options: {
		failedOf: (row: Row) => boolean;
		durationOf?: (row: Row) => number | null;
		isAction: (row: Row) => boolean;
		/**
		 * Whether this row OPENS a turn, i.e. starts a new `runsOf` run.
		 *
		 * The caller passes the partition (`transcript-rows.ts:runsOf`) so the
		 * foot counts the SAME unit the caption rule and the turn collapse do.
		 * The default — `gap === "turn" || "first"` — is the historical proxy and
		 * is WRONG for a steered turn: a steer row carries the turn gap as well,
		 * and resetting there is how a steered run's foot reported only the rows
		 * after the steer (`4 actions` under a bar saying `12`, the F5 defect the
		 * unified unit removes). Callers with no partition in hand (tests, a
		 * record-space walk) keep the proxy.
		 */
		opensRun?: (row: Row) => boolean;
	},
): Map<string, TurnFoot> {
	const feet = new Map<string, TurnFoot>();
	const opensRun =
		options.opensRun ??
		((row: Row) => row.gap === "turn" || row.gap === "first");
	let actions = 0;
	let failed = 0;
	let durationS: number | null = null;
	let firstFailedId: string | null = null;

	for (const row of rows) {
		/*
		 * A row that opens a turn RESETS the tally: a foot line reports what the
		 * turn above it did, and carrying the previous turn's counts into the next
		 * one would report the conversation rather than the turn.
		 */
		if (opensRun(row)) {
			actions = 0;
			failed = 0;
			durationS = null;
			firstFailedId = null;
		}
		if (options.isAction(row)) {
			actions += 1;
			if (options.failedOf(row)) {
				failed += 1;
				firstFailedId ??= row.record.id;
			}
			const seconds = options.durationOf ? options.durationOf(row) : null;
			if (typeof seconds === "number") durationS = (durationS ?? 0) + seconds;
		}
		if (row.closesTurn) {
			feet.set(row.record.id, { actions, failed, durationS, firstFailedId });
		}
	}
	return feet;
}
