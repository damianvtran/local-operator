/**
 * The ask-id chip's rules: which tokens in prose ARE ask ids, what state word a
 * chip carries, and which of THIS session's asks an id resolves to.
 *
 * WHY A MODULE OF ITS OWN. The renderer draws what this module decides, and the
 * repository's habit is that anything worth asserting lives outside a component
 * so it can be exercised without a rendered tree. Every rule here is a pure
 * reading of strings and rows — no React, no store.
 *
 * THE MATCH IS AN EXACT TOKEN, not a substring. An agent cites an ask id it was
 * handed (`a-745e`) inside prose, usually in backticks; the same six characters
 * appear inside hashes, commit shas, paths and file names, and a chip drawn
 * there would be a control that opens nothing. So a match must stand alone: the
 * character before and after it may not be a word character, `-`, `_`, a path
 * separator (`/`, `\`) or a leading `.` — and a trailing `.` only ends the id
 * when no name continues after it. `a-745e6` is a longer hex run, `xa-745e` is
 * the tail of a different token, `a-12ab.json` is a file, and none is an ask id.
 *
 * IDS ARE UNIQUE PER SESSION ONLY (`secrets.token_hex(2)` in the runtime's ask
 * store), so the chip NEVER looks across conversations: an id that is not among
 * THIS session's asks is drawn as plain mono, because opening another
 * conversation's ask would be a jump to the wrong question.
 *
 * This is the relay's reading of the native app's `features/session/ask-chip.ts`
 * (`damianvtran/local-operator-mobile` origin/main). Same grammar, same
 * precedence between the live frame, the transcript receipts and the aggregate;
 * the copy lives in this surface's own catalogue because the words are the
 * relay's.
 */
import type { WithoutParams } from "../i18n/keys.gen";
import type { PendingAsk, TranscriptEntry } from "../types";

/** An ask id as an agent writes it: `a-` and four lower-case hex digits. */
const ASK_ID = /a-[0-9a-f]{4}/g;

/** What may not touch an id and still leave it an id (see the module note). */
const TOKEN_EDGE_BEFORE = /[\w\-/\\.]/;
const TOKEN_EDGE_AFTER = /[\w\-/\\]/;
const TOKEN_ENDS_WITH_DOT_NAME = /^[.][\w-]/;

/** One run of prose, or one ask id inside it. */
export type AskChipSegment =
	| { kind: "text"; text: string }
	| { kind: "ask"; id: string };

/**
 * Split prose into plain runs and ask-id tokens.
 *
 * The fast path is the common one: most spans carry no id, and scanning them
 * for one every frame a streamed answer renders is work with no result.
 */
export function askIdSegments(text: string): AskChipSegment[] {
	if (!text.includes("a-")) return text === "" ? [] : [{ kind: "text", text }];
	const segments: AskChipSegment[] = [];
	let cursor = 0;
	for (const match of text.matchAll(ASK_ID)) {
		const start = match.index ?? 0;
		const end = start + match[0].length;
		if (TOKEN_EDGE_BEFORE.test(text[start - 1] ?? "")) continue;
		if (TOKEN_EDGE_AFTER.test(text[end] ?? "")) continue;
		if (TOKEN_ENDS_WITH_DOT_NAME.test(text.slice(end))) continue;
		if (start > cursor) segments.push({ kind: "text", text: text.slice(cursor, start) });
		segments.push({ kind: "ask", id: match[0] });
		cursor = end;
	}
	if (cursor === 0) return [{ kind: "text", text }];
	if (cursor < text.length) segments.push({ kind: "text", text: text.slice(cursor) });
	return segments;
}

/** Whether a whole string is exactly one ask id (an id alone in a code span). */
export function isAskId(text: string): boolean {
	const segments = askIdSegments(text);
	return segments.length === 1 && segments[0]?.kind === "ask";
}

/**
 * The short word a chip carries, per state — the SAME eight states the state
 * lines name (`lib/asks.ts`), in the one word a chip has room for. A state a
 * newer runtime invents falls through to its own word rather than being
 * reported as a known one (`askStateLine`'s rule, kept here too).
 */
const CHIP_WORD: ReadonlyMap<string, WithoutParams> = new Map<string, WithoutParams>([
	["open", "relay.asks.chip.open"],
	["answered", "relay.asks.chip.answered"],
	["declined", "relay.asks.chip.declined"],
	["timed_out", "relay.asks.chip.timed_out"],
	["late", "relay.asks.chip.late"],
	["dismissed", "relay.asks.chip.dismissed"],
	["withdrawn", "relay.asks.chip.withdrawn"],
	["expired", "relay.asks.chip.expired"],
]);

/** The catalogue key for a state's chip word, or the state's own spelling for
 *  one this build does not know. `null` only for an empty status. */
export function askChipWord(
	status: string,
): { kind: "key"; key: WithoutParams } | { kind: "literal"; text: string } | null {
	/* A Map, not a plain record: a wire status of `constructor` or `toString`
	 * must not reach the catalogue as an inherited member. The runtime may
	 * invent a state, so every key is untrusted input. */
	const known = CHIP_WORD.get(status);
	if (known !== undefined) return { kind: "key", key: known };
	const literal = status.replace(/_/g, " ").trim();
	return literal === "" ? null : { kind: "literal", text: literal };
}

/** Whether a chip's dot is drawn in the accent (a question still open) or muted
 *  (settled). Colour is never the only signal — the word beside it is — so this
 *  is a reading aid, not the state. */
export function askChipOpen(status: string): boolean {
	return status === "open";
}

/** What one id resolves to here: the state word's status and the question the
 *  screen reader names. */
export interface AskChipFact {
	status: string;
	question: string;
}

/** The transcript receipt a settled ask leaves (`ask_response` / `ask_timeout`).
 *  Narrow on purpose: the chip needs the id, the status and the head question,
 *  and taking the whole entry would tie this module to the wire's shape. */
export interface AskChipReceipt {
	kind: string;
	details: {
		ask_id?: string | undefined;
		status?: string | undefined;
		questions?: readonly { question?: string | undefined }[] | undefined;
	};
}

function headQuestion(
	questions: readonly { question?: string | undefined }[] | undefined,
): string {
	return String(questions?.[0]?.question ?? "").replace(/\s+/g, " ").trim();
}

export interface AskChipSources {
	/** The live frame's rows for this conversation (`projection.asks`). They
	 *  carry no `session_id`, so they ARE this session's by construction. */
	frame?: readonly PendingAsk[] | null;
	/** The settled receipts the transcript already drew. This is the only source
	 *  that still knows a settled ask's state after it left the frame. */
	transcript?: readonly (TranscriptEntry | AskChipReceipt)[] | null;
	/** The asks sheet's last successful read (`GET /api/asks`, every
	 *  conversation). It is the only source that can name a WITHDRAWN ask — the
	 *  one settled state the transcript has no row for. Rows of other
	 *  conversations are dropped: ids repeat across sessions. Its `open` rows are
	 *  only trusted when `frame` is not an array (see below). */
	settled?: readonly PendingAsk[] | null;
}

/**
 * The ids THIS session can answer for, in precedence order: the live frame
 * first (the freshest reading of an open ask), then a transcript receipt, then
 * the sheet's aggregate.
 *
 * A FRAME ROW WINS OVER A RECEIPT because the frame is live: an ask answered on
 * another device settles there first, and a stale `open` chip is what the reader
 * would press expecting a question.
 */
export function askChipFacts(
	{ frame, transcript, settled }: AskChipSources,
	sessionId: string,
): Map<string, AskChipFact> {
	const facts = new Map<string, AskChipFact>();
	/* `null` is "this runtime publishes no frame"; an array — even an empty one
	 * — is a live reading that knows what it does not hold. */
	const live = Array.isArray(frame) ? frame : null;
	if (live !== null) {
		for (const row of live) {
			const id = String(row?.ask_id ?? "");
			if (id === "") continue;
			facts.set(id, { status: String(row.status || "open"), question: headQuestion(row.questions) });
		}
	}
	if (Array.isArray(transcript)) {
		for (const entry of transcript) {
			const id = String(entry?.details?.ask_id ?? "");
			if (id === "" || facts.has(id)) continue;
			const status = String(
				entry.details.status || (entry.kind === "ask_timeout" ? "timed_out" : "answered"),
			);
			/* `questions` is on `details` for a receipt; the os-level entry shape
			 * keeps them there too, so one read serves both. */
			facts.set(id, { status, question: headQuestion(entry.details.questions) });
		}
	}
	if (Array.isArray(settled)) {
		for (const row of settled) {
			const id = String(row?.ask_id ?? "");
			if (id === "" || facts.has(id)) continue;
			/* Another conversation's row: an id is unique per session only, so
			 * this is a different ask that happens to spell the same. */
			if (String(row.session_id ?? "") !== sessionId) continue;
			const status = String(row.status || "open");
			/* `open` IS NOT THIS SOURCE'S TO SAY while the frame speaks: the sheet
			 * reads only while it is visible and keeps its last read after it
			 * closes, so an ask the agent has since withdrawn still reads `open`
			 * there — the frame has dropped it and there is no receipt, which is
			 * exactly the headline "the old one, a-745e, is withdrawn". A stale
			 * `open` chip is what the reader would press expecting a question.
			 * Every OTHER state still says something the frame cannot: a settled
			 * ask leaves the frame, and `withdrawn` leaves no receipt at all. */
			if (status === "open" && live !== null) continue;
			facts.set(id, { status, question: headQuestion(row.questions) });
		}
	}
	return facts;
}
