/**
 * THE DESKTOP PREDICATES THE CONDENSATION MODEL READS, lifted verbatim.
 *
 * SOURCES, at the desktop sourceHash `parity-sources.json` pins
 * (`f9f4611dd3dbcc06…`), line ranges of THAT source (measured, round 2 - the round-1
 * ranges ended at the next function's start, not at its own end):
 * - `components/trace/tool-row-model.ts` — `isPartialDelivery` 553-573,
 *   `isFailedResult` 618-741, `displayName` 803-833.
 * - `components/trace/receipt-row-model.ts` — `receiptFamilyOf` 194-213,
 *   `isInternalReceipt` 214-252, the `PeerSender` type 318-331,
 *   `EMPTY_SENDER` 332-352, `peerName` 353-388, `peerIdentity` 389-409.
 *
 * WHY A SEPARATE FILE. The model ports (`condense-rows.ts`, `condense-segments.ts`,
 * `condense-fold.ts`, `condense-plan.ts`) are copies of desktop modules, and those
 * modules import four small decision helpers from two other desktop modules
 * (`tool-row-model.ts`, `receipt-row-model.ts`). Those helpers are shared
 * DECISIONS - "is this result a failure", "is this row an internal receipt",
 * "what is this tool called" - so the port takes them verbatim in one place
 * instead of re-deciding them per call site. A rewritten predicate is a surface
 * deciding something the other surface already decided, which is the drift class
 * the parity fixture exists to catch and this file exists to avoid.
 *
 * WHAT IS DELIBERATELY NOT HERE:
 * - `toolRowLabel`/`toolVerb` and the verb tables. They compose the row's ENGLISH
 *   (`Ran pnpm test`, `Viewing agent designer`), and this port's contract is that
 *   the model states DECISIONS while each surface owns its words (see
 *   `condense-fold.ts`'s `foldLive`, which returns the executing row's identity
 *   and lets `tool-row.tsx` say its own sentence).
 * - `toolOp`. The desktop derives an operation token from a call's arguments
 *   (`todo view` vs `todo write`, `jobs cancel` vs `jobs list`); this wire strips
 *   `details` under frame pressure, so half the calls have no arguments to read
 *   and the token would be right only sometimes. `condense-fold.ts` takes `op`
 *   from the caller and the surface passes "" - the reported degradation, named
 *   in `condense.test.ts`'s skip-list and measured in the real-journal agreement
 *   run.
 * - `SendDeliveryState` and the mailbox family. The relay's `tool_state` carries
 *   the fold's verdict (`interrupted` vs `failed`) and the wire has no mailbox
 *   state, so `isFailedResult` is called with no delivery (see its note).
 */

/** The prefix MCP-server tools carry, which the row names without. */
const MCP_PREFIX = "mcp__";

/**
 * A tool's display name: the wire name, with the MCP server's `mcp__<server>__`
 * prefix dropped.
 *
 * WHY THIS IS A DECISION AND NOT WORDS. `ledgerName` and the count line's
 * `name:<label>` key are built from it, so the two surfaces must agree on what the
 * call is CALLED even when they phrase the line differently.
 */
export function displayName(toolName: string): string {
	const name = toolName.trim();
	if (!name.toLowerCase().startsWith(MCP_PREFIX)) return name;
	const rest = name.slice(MCP_PREFIX.length);
	/*
	 * ONE UNDERSCORE, NOT TWO: the producer mints `mcp__<server>_<tool>`, so the
	 * server segment ends at the FIRST `_`. A server whose own name contains an
	 * underscore cannot be split back out, which is why only the first segment is
	 * treated as the server and the remainder is still the call's identifier.
	 * (Found by the real-journal agreement run: the two-underscore reading renamed
	 * `mcp__datadog_search_datadog_metrics` to `datadog_search_datadog_metrics`
	 * and every count-line key built from it disagreed.)
	 */
	const split = rest.indexOf("_");
	if (split <= 0) return rest || name;
	return rest.slice(split + 1) || rest;
}

/**
 * A `send` that settled `mailbox`/`unconfirmed` is NOT a failure.
 *
 * The relay wire has no delivery state at all (G8): a partial `send` the relay
 * marks `done` counts as success on both surfaces, and a never-sent call is
 * `interrupted` and excluded from failures.
 */
export function isPartialDelivery(
	state: string | null | undefined,
): state is "mailbox" | "unconfirmed" {
	return state === "mailbox" || state === "unconfirmed";
}

/** The fold's failure predicate, over the fields this wire carries. */
export function isFailedResult(
	isError: boolean | null | undefined,
	delivery: string | null | undefined,
): boolean {
	if (isError !== true) return false;
	return !isPartialDelivery(delivery);
}

/**
 * The kinds of internal traffic the transcript may fold beside the actions that
 * caused it - one classification rather than a set of type strings repeated at
 * each consumer, ported so the fold and the plan cannot disagree about what a
 * receipt is.
 */
export type InternalReceiptFamily = "incoming" | "job_result" | "wake" | "monitor_prompt";

export function receiptFamilyOf(record: {
	kind: string;
	customType?: string;
}): InternalReceiptFamily | null {
	if (record.kind === "peer") return "incoming";
	if (record.kind === "wake") return "wake";
	if (record.kind !== "custom") return null;
	switch (record.customType) {
		case "hub_message":
			return "incoming";
		case "job_result":
			return "job_result";
		case "monitor_prompt":
			return "monitor_prompt";
		default:
			return null;
	}
}

/** Positive structural classification; arbitrary custom notices stay boundaries. */
export function isInternalReceipt(record: {
	kind: string;
	customType?: string;
}): boolean {
	return receiptFamilyOf(record) !== null;
}

/** A hub delivery's typed facts, in the shape the deskop's reducer holds them. */
export type HubReceipt = {
	direction: string;
	expectsReply: boolean | null;
	steer: boolean | null;
	label: string;
	jobId: string;
};

/** A peer sender's identity, as the wire carries it (all fields advisory). */
export type CondensePeerSender = {
	pid: string;
	conversationName: string;
	cwd: string;
	sessionId: string;
	modelLabel: string;
};

/** The tools this surface renders as a generating-media card. */
const IMAGE_GEN_TOOL_NAMES: ReadonlySet<string> = new Set([
	"generate_image",
	"generate_video",
	"generate_voiceover",
	"generate_sound_effect",
]);

export function isImageGenTool(toolName: string): boolean {
	return IMAGE_GEN_TOOL_NAMES.has(toolName);
}

const TRAILING_SEPARATORS = /[\\/]+$/;
const PATH_SEPARATOR = /[\\/]/;

/**
 * `(name, quoted)` for the sending session, or `("", false)`.
 *
 * The ladder - conversation name, then cwd basename, then a short session id -
 * is why a peer receipt never degrades to a bare pid: a row naming `pid 1` names
 * nothing a reader can act on, and the whole point of the row is to say WHICH
 * session reached in. Ported verbatim; the quiet-group sender bucket keys on it,
 * so the two surfaces must bucket a sender identically even though each words the
 * bar itself.
 */
export function peerName(sender: CondensePeerSender): {
	name: string;
	quoted: boolean;
} {
	if (sender.conversationName) {
		return { name: sender.conversationName, quoted: true };
	}
	const cwd = sender.cwd.replace(TRAILING_SEPARATORS, "");
	if (cwd) {
		const base = cwd.split(PATH_SEPARATOR).pop() ?? "";
		if (base) return { name: `${base}/`, quoted: false };
	}
	if (sender.sessionId) return { name: sender.sessionId.slice(0, 8), quoted: false };
	return { name: "", quoted: false };
}

/** The identity a collapsed receipt row leads with (ported verbatim). */
export function peerIdentity(sender: CondensePeerSender): string {
	const { name, quoted } = peerName(sender);
	if (name) return quoted ? `"${name}"` : name;
	return sender.pid ? `pid ${sender.pid}` : "another session";
}

/** An all-absent sender, so the record shape is never a nullable pair. */
export const EMPTY_SENDER: CondensePeerSender = {
	pid: "",
	conversationName: "",
	cwd: "",
	sessionId: "",
	modelLabel: "",
};
