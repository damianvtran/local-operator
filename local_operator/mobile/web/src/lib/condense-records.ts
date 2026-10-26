/**
 * THE RECORD THE CONDENSATION MODEL READS, and the ONE place this surface's wire
 * is mapped onto it.
 *
 * WHY A DESKTOP-SHAPED RECORD AND NOT `TranscriptEntry` ITSELF. The model this
 * port carries (`condense-rows.ts`, `condense-segments.ts`, `condense-plan.ts`,
 * `condense-fold.ts`) is the desktop's, copied so its decisions can be checked
 * against the desktop's own generated fixture
 * (`fixtures/condense.parity.json`, produced by
 * `local-operator-ui/scripts/generate-condense-parity.mjs`). Every predicate in
 * it is written against that record's fields — `kind`, `streaming`,
 * `stopReason`, `callsTools`, `phase`, `isError`, `stopped`, `neverSent`,
 * `notRunReason`, `toolName`, `customType`, `level`, `complete`, `receipt`,
 * `text`, `durationS`. Rewriting them against `TranscriptEntry` would spread
 * relay knowledge across ~2k lines and turn every predicate into a fresh port
 * that can drift; mapping once, here, keeps the ported code diffable against the
 * desktop file it came from and makes the fixture replay run the SAME code the
 * screen runs.
 *
 * WHY THE WIRE CANNOT EXPRESS EVERYTHING (and what happens when it cannot).
 * Each gap below is decided here and nowhere else, and `condense.test.ts` names
 * the fixture cases that cannot replay because of it:
 *
 * - NO PER-ROW TIME (architect G1). `ts`/`startedAt`/`endedAt` are absent, so a
 *   fold's wall-clock span is null and a bar carries no stamp. The durable
 *   desktop rows have null stamps too, so settled parity is unaffected; `Took`
 *   still works because it sums durations, which `elapsed_s` supplies.
 * - NO QUIET `no_reply` ROW (G4). The relay never mints it, so a quiet-closed
 *   run reads as a STEERED run: its earlier exchange only condenses when the
 *   merged run settles and its bar says `Steered`. 2 calls in 1 of 102 journal
 *   copies, so this is recorded rather than fixed.
 * - NO TYPED ATTENTION MARKER (G5). The relay paints no
 *   `completion_attention` row; `terminal` on a notice is the only terminal
 *   fact, and it covers the `:stop` notice, the live error notice and a session
 *   incident.
 * - LIVE AND REPLAY TOOL IDS DIFFER (G9). Live rows are `tc-<call>`, durable
 *   rows `<msg>:<call>`, so identity is derived here (`rowKey`) instead of being
 *   taken from `id`.
 */
import type { TranscriptEntry } from "../types";

/**
 * The record's discriminant, in the desktop's vocabulary.
 *
 * The relay's kinds map onto it like this (and only here):
 * `steer` -> `user` (a mid-run user row is a steer to `walkTurns`),
 * `reasoning` -> a streaming assistant with no text (it paints nothing, but it
 * keeps the run self-live: the model really is thinking),
 * `notice` + a receipt `custom_type` -> `custom`, `notice` + `terminal` ->
 * a complete notice, everything else keeps its own name.
 */
export type CondenseKind =
	| "user"
	| "assistant"
	| "tool"
	| "notice"
	| "custom"
	| "wake"
	| "peer"
	| "compaction"
	| "ask_response"
	| "ask_timeout";

/** A hub delivery's typed facts, as the desktop's reducer holds them. */
export type CondenseReceipt = {
	direction?: string;
	expectsReply?: boolean;
	steer?: boolean;
};

export type CondenseSender = {
	pid: string;
	conversationName: string;
	cwd: string;
	sessionId: string;
	modelLabel: string;
};

type Base = {
	/** Stable identity (see `rowKey` — never the raw `id` for a tool row). */
	id: string;
	/**
	 * Epoch ms, when the surface carries one. The relay does not (G1), so it is
	 * optional and the model's time readers answer null when it is absent.
	 */
	ts?: number;
};

export type CondenseRecord =
	| (Base & { kind: "user"; text: string })
	| (Base & {
			kind: "assistant";
			text: string;
			streaming: boolean;
			/** The provider's phase declaration (G2), absent when unknown. */
			stopReason?: string | null;
			/** This row carried calls of its own (G3). */
			callsTools?: boolean;
	  })
	| (Base & {
			kind: "tool";
			toolName: string;
			phase: "composing" | "queued" | "running" | "done";
			isError: boolean;
			stopped: boolean;
			/** Always false on this wire: a send mailbox is not modelled (G8). */
			neverSent: boolean;
			notRunReason: null;
			durationS: number | null;
			summary?: string;
			args?: Record<string, unknown>;
	  })
	| (Base & { kind: "notice"; text: string; level: string; complete: boolean })
	| (Base & {
			kind: "custom";
			text: string;
			customType: string;
			level: string;
			receipt?: CondenseReceipt;
	  })
	| (Base & { kind: "wake"; text: string })
	| (Base & { kind: "peer"; body: string; sender: CondenseSender })
	| (Base & { kind: "compaction"; text: string; before: number | null })
	| (Base & { kind: "ask_response"; text: string; level: string })
	| (Base & { kind: "ask_timeout"; text: string; level: string });

/** The receipt families the fold groups by (G6). */
const RECEIPT_CUSTOM_TYPES = new Set(["job_result", "monitor_prompt", "hub_message"]);

/**
 * The stable identity of a row, across the live/replay boundary (G9).
 *
 * A live tool row's id is `tc-<call>`, and the same call after a re-seed is
 * `<msg>:<call>`. The fold's latch, the open registry, dedup and the segment keys
 * all key on this, so a reconnect that re-keyed rows would close a group the
 * reader opened and let a merged history page duplicate it.
 */
export function rowKey(entry: Pick<TranscriptEntry, "id" | "kind" | "tool_call_id">): string {
	return entry.kind === "tool" && entry.tool_call_id
		? `call:${entry.tool_call_id}`
		: entry.id;
}

/** The empty sender shape the model's peer arms read fields off. */
const NO_SENDER: CondenseSender = {
	pid: "",
	conversationName: "",
	cwd: "",
	sessionId: "",
	modelLabel: "",
};

/** The tool phases the relay's wire carries, in the desktop's vocabulary. */
function phaseOf(entry: TranscriptEntry): "composing" | "queued" | "running" | "done" {
	switch (entry.tool_state) {
		case "composing":
		case "queued":
		case "running":
			return entry.tool_state;
		default:
			// `done`, `failed` and `interrupted` are all "settled": the desktop's
			// fold reads the outcome from `isError`/`stopped`, never from the phase.
			return "done";
	}
}

/**
 * One wire row as the model's record, or null when the row must not feed the
 * model at all.
 *
 * `steer` becomes a `user` row because that is what the desktop's `walkTurns`
 * reads a mid-run user row as; the surface still paints it as a steer, which is
 * a rendering decision this module deliberately does not make.
 */
export function toCondenseRecord(entry: TranscriptEntry): CondenseRecord | null {
	const id = rowKey(entry);
	/*
	 * NO PER-ROW TIME ON THIS WIRE (gap G1): `ts` stays absent, so every model
	 * reader of a time answers null and the fold states no wall-clock span. The
	 * field is left off rather than set to a placeholder, because "absent" is the
	 * desktop's own durable-row shape too.
	 */
	const ts: number | undefined = undefined;
	switch (entry.kind) {
		case "user":
		case "steer":
			return { kind: "user", id, ts, text: entry.text };
		case "assistant":
			return {
				kind: "assistant",
				id,
				ts,
				text: entry.text,
				streaming: entry.final === false,
				stopReason: entry.stop_reason || null,
				callsTools: entry.calls_tools === true,
			};
		case "reasoning":
			/*
			 * A transient thinking row: it paints nothing (`paintsSomething`), and
			 * it keeps the run self-live, which is the honest reading - the model
			 * really is thinking and the turn must not condense under the reader.
			 */
			return { kind: "assistant", id, ts, text: "", streaming: true };
		case "tool":
			return {
				kind: "tool",
				id,
				ts,
				toolName: entry.tool_name,
				phase: phaseOf(entry),
				isError: entry.tool_state === "failed",
				stopped: entry.tool_state === "interrupted",
				neverSent: false,
				notRunReason: null,
				// `0.0` means UNKNOWN on this wire, never "instant": the desktop
				// never claims `0s` (`factsOf`), so a zero becomes null here.
				durationS: entry.elapsed_s > 0 ? entry.elapsed_s : null,
				summary: entry.summary,
				args: entry.details?.args as Record<string, unknown> | undefined,
			};
		case "notice": {
			/* Bookkeeping customs can project a text-less notice. It has no
			   renderable content, so it must not enter buildRows as a blank row;
			   importantly this guard is text-based, NOT a custom_type blocklist:
			   session_state and MCP warnings with real text stay visible. */
			if (!entry.text.trim()) return null;
			const severity = String(entry.details?.severity ?? "info");
			const customType = entry.custom_type || "";
			if (customType === "wake_prompt" || entry.details?.notice_kind === "wake")
				return { kind: "wake", id, ts, text: entry.text };
			if (customType === "session_incident")
				return { kind: "custom", id, ts, text: entry.text, customType, level: "error" };
			if (customType && RECEIPT_CUSTOM_TYPES.has(customType))
				return { kind: "custom", id, ts, text: entry.text, customType, level: "info" };
			return {
				kind: "notice",
				id,
				ts,
				text: entry.text,
				level: severity,
				// A notice is COMPLETE only when it is terminal: the completion mark
				// is the desktop's `complete`, and the relay's terminal flag is the
				// only fact that says a turn ended here.
				complete: entry.terminal === true,
			};
		}
		case "compaction":
			return { kind: "compaction", id, ts, text: entry.text, before: null };
		case "peer_message":
			return {
				kind: "peer",
				id,
				ts,
				body: entry.text,
				sender: senderOf(entry.details?.sender),
			};
		case "parent_message":
		case "subagent_message": {
			/*
			 * A hub delivery. With a direction it is a typed receipt (the desktop's
			 * `custom` + `receipt`); WITHOUT one, `parent_message` is the desktop's
			 * "unstamped old-core hub user" - a user boundary - while an unstamped
			 * `subagent_message` stays a custom row with no receipt, so unknown
			 * intent stays visible (turn-collapse-model.ts:343-369).
			 */
			const direction =
				typeof entry.details?.direction === "string"
					? String(entry.details.direction)
					: "";
			const receipt: CondenseReceipt = {
				direction,
				expectsReply: entry.details?.expects_reply === true,
				steer: entry.details?.steer === true,
			};
			if (direction)
				return {
					kind: "custom",
					id,
					ts,
					text: entry.text,
					customType: "hub_message",
					level: "info",
					receipt,
				};
			return entry.kind === "parent_message"
				? { kind: "user", id, ts, text: entry.text }
				: { kind: "custom", id, ts, text: entry.text, customType: "hub_message", level: "info" };
		}
		case "ask_response":
			return {
				kind: "ask_response",
				id,
				ts,
				text: entry.text,
				level: String(entry.details?.severity ?? "info"),
			};
		case "ask_timeout":
			return {
				kind: "ask_timeout",
				id,
				ts,
				text: entry.text,
				level: String(entry.details?.severity ?? "warning"),
			};
		default:
			// Pending echoes and any future kind: the plan is not the place to
			// invent a row. The surface paints what the model does not claim.
			return null;
	}
}

function senderOf(value: unknown): CondenseSender {
	if (value === null || typeof value !== "object") return NO_SENDER;
	const raw = value as Record<string, unknown>;
	const read = (key: string): string =>
		typeof raw[key] === "string" ? (raw[key] as string) : "";
	return {
		pid: read("pid"),
		conversationName: read("conversation_name") || read("conversationName"),
		cwd: read("cwd"),
		sessionId: read("session_id") || read("sessionId"),
		modelLabel: read("model_label") || read("modelLabel"),
	};
}

/** Every wire row the model takes, in wire order. */
export function toCondenseRecords(entries: readonly TranscriptEntry[]): CondenseRecord[] {
	const records: CondenseRecord[] = [];
	for (const entry of entries) {
		const record = toCondenseRecord(entry);
		if (record !== null) records.push(record);
	}
	return records;
}
