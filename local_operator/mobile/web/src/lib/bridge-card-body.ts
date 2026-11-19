/**
 * Inbound Team Bridge card bodies on the phone relay — the pure half.
 *
 * WHY A SEPARATE READ, NOT A ROW FIELD. Core never writes a Bridge body into
 * the session journal: the journal is replayed, forked, exported and
 * compacted, and a body in it could neither be revoked nor kept out of the
 * model's context (core `team_bridge/card_projection.py`). The relay's
 * transcript row therefore only carries the receipt line ("<sender> ·
 * <conversation>", `harness/bridge_copy.py::bridge_delivery_receipt_line`).
 * The ADMITTED body lives in core's display store and is answered, keyed by
 * the transcript row the phone already reads, by
 *
 *   GET /api/sessions/{session_id}/bridge-cards/{entry_id}/body
 *
 * advertised as `capabilities.features.team_bridge_card_body >= 1` on the
 * session-list frame (core `server/features.py`, lifted by the daemon's
 * `_feature_flags`). The answer is pinned by the bridge-body design §4:
 *
 *   {v: 1, items: [{message_id, conversation_id, state, body?, source?,
 *                   hold?: {class, held_for_s}, reason?}], more}
 *
 * This is a port of the desktop's `bridge-receipt-model.ts` (local-operator-ui
 * #1977) so the two surfaces cannot disagree about what a card may say.
 *
 * CONSTRAINTS:
 *  - Every state and reason is the SERVER'S word. Nothing here derives a
 *    moderation outcome (design safety property D): an unknown state, an
 *    admitted item without a body or a malformed frame all read as
 *    `unavailable`, never as a guess in either direction.
 *  - Fail closed on the capability: an older relay that does not advertise it
 *    is never asked, so its card keeps today's receipt line unchanged.
 *  - No caching. A body can be revoked (sign-out, leave, account switch,
 *    hub delete); the read happens on every mount so revocation stays live.
 *  - The body is untrusted markdown authored by someone else: the card renders
 *    it as data, inside the provenance frame, through the transcript's own
 *    escaping markdown renderer (no HTML, http(s) links only).
 */
import type { Capabilities } from "../types";

/** The relay can answer card bodies (`features.team_bridge_card_body >= 1`). */
export function bridgeCardBodyEnabled(capabilities: Capabilities | null | undefined): boolean {
	const flag = capabilities?.features?.team_bridge_card_body;
	return typeof flag === "number" && Number.isFinite(flag) && flag >= 1;
}

/** The closed state vocabulary of the card-body read (design §4). */
export const BRIDGE_CARD_BODY_STATES = [
	"admitted",
	"pending",
	"held",
	"refused",
	"not_admitted",
	"unavailable",
] as const;
export type BridgeCardBodyState = (typeof BRIDGE_CARD_BODY_STATES)[number];

/** Why an item is `unavailable`, when the server said (design §4). */
export const BRIDGE_CARD_BODY_REASONS = [
	"signed_out",
	"other_account",
	"not_member",
	"deleted",
	"hub_unsupported",
	"offline",
] as const;
export type BridgeCardBodyReason = (typeof BRIDGE_CARD_BODY_REASONS)[number];

export type BridgeCardBodyItem = {
	messageId: string;
	state: BridgeCardBodyState;
	/** Present only on an `admitted` item: the admitted markdown, untrusted. */
	body?: string;
	reason?: BridgeCardBodyReason;
	/** Only on a `held` item whose hold says `class: "transient"`: screening has
	    not run yet, so nobody decided anything (core `card_body._hold_view`).
	    Any other or missing class keeps "Held by moderation": an unknown class
	    is never softened into "waiting". */
	transient?: true;
};

export type BridgeCardBody = {
	items: BridgeCardBodyItem[];
	/** Messages beyond the listed ones: the server's count plus our own cap. */
	more: number;
};

/** The most items a coalesced card lists. Core caps the answer at 20 (design
    §4: up to 200 ids coalesce into one card, the rest are counted in `more`);
    the client holds the same cap so a server that sent more cannot grow the
    card past what was reviewed, and the overflow is counted, not dropped. */
export const BRIDGE_CARD_BODY_MAX_ITEMS = 20;

/** Characters past which a body folds behind "Show more". ~600 characters is
    about ten lines at a phone's measure: an ordinary chat message shows whole,
    one long paste does not bury the rest of the transcript. Same number as the
    desktop card so both surfaces fold the same messages. */
export const BRIDGE_CARD_BODY_COLLAPSE_CHARS = 600;

const STATES: ReadonlySet<string> = new Set(BRIDGE_CARD_BODY_STATES);
const REASONS: ReadonlySet<string> = new Set(BRIDGE_CARD_BODY_REASONS);
const ID_MAX = 128;

const object = (value: unknown): Record<string, unknown> =>
	value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

function itemOf(value: unknown): BridgeCardBodyItem | null {
	const raw = object(value);
	const messageId = raw.message_id;
	const conversationId = raw.conversation_id;
	/* An item without its ids cannot be attributed to a message in this card, so
	   it answers nothing: dropped rather than shown as someone's body. */
	if (
		typeof messageId !== "string" || !messageId || messageId.length > ID_MAX ||
		typeof conversationId !== "string" || !conversationId || conversationId.length > ID_MAX
	)
		return null;
	const declared = typeof raw.state === "string" && STATES.has(raw.state)
		? (raw.state as BridgeCardBodyState)
		: "unavailable";
	/* `admitted` MUST carry a non-empty body; without one there is nothing to
	   show, so it reads as unavailable instead of an empty frame. */
	const admitted = declared === "admitted" && typeof raw.body === "string" && raw.body.trim().length > 0;
	const state: BridgeCardBodyState = declared === "admitted" && !admitted ? "unavailable" : declared;
	const reason = state === "unavailable" && typeof raw.reason === "string" && REASONS.has(raw.reason)
		? (raw.reason as BridgeCardBodyReason)
		: undefined;
	const transient = state === "held" && object(raw.hold).class === "transient";
	return {
		messageId,
		state,
		...(admitted ? { body: raw.body as string } : {}),
		...(reason ? { reason } : {}),
		...(transient ? { transient: true as const } : {}),
	};
}

/** Parse one answer. `null` means the frame is not the pinned v1 shape (a
    newer server, a proxy page, a refusal body); the card then says the body is
    unavailable rather than guessing. */
export function bridgeCardBodyOf(payload: unknown): BridgeCardBody | null {
	const frame = object(payload);
	if (frame.v !== 1 || !Array.isArray(frame.items)) return null;
	const items = frame.items.map(itemOf).filter((item): item is BridgeCardBodyItem => item !== null);
	/* An item dropped for unattributable ids is still a message in this card:
	   it is counted with the overflow so the card never shows 2 of 3 with no
	   hint that a third exists (review m3). */
	const dropped = frame.items.length - items.length;
	const declaredMore = typeof frame.more === "number" && Number.isSafeInteger(frame.more) && frame.more > 0
		? frame.more
		: 0;
	const listed = items.slice(0, BRIDGE_CARD_BODY_MAX_ITEMS);
	return { items: listed, more: declaredMore + dropped + (items.length - listed.length) };
}

/** The one line a NON-admitted item shows, keyed. `hub_unsupported` and an
    absent reason share the generic line: there is nothing the reader can do
    about either, and naming the hub's internals would not help them. */
export type BridgeItemCopyKey =
	| "relay.transcript.bridge.pending"
	| "relay.transcript.bridge.held"
	| "relay.transcript.bridge.held_transient"
	| "relay.transcript.bridge.refused"
	| "relay.transcript.bridge.not_admitted"
	| "relay.transcript.bridge.unavailable"
	| "relay.transcript.bridge.signed_out"
	| "relay.transcript.bridge.other_account"
	| "relay.transcript.bridge.not_member"
	| "relay.transcript.bridge.deleted"
	| "relay.transcript.bridge.offline";

export function bridgeItemCopyKey(item: BridgeCardBodyItem): BridgeItemCopyKey {
	switch (item.state) {
		case "admitted":
			/* Only reached for an admitted item the card cannot render, which the
			   parser already turned into `unavailable`; kept total for the type. */
			return "relay.transcript.bridge.unavailable";
		case "pending":
			return "relay.transcript.bridge.pending";
		case "held":
			return item.transient ? "relay.transcript.bridge.held_transient" : "relay.transcript.bridge.held";
		case "refused":
			/* Never a reason: moderation categories are not shown (design §2). */
			return "relay.transcript.bridge.refused";
		case "not_admitted":
			return "relay.transcript.bridge.not_admitted";
		case "unavailable":
			switch (item.reason) {
				case "signed_out":
					return "relay.transcript.bridge.signed_out";
				case "other_account":
					return "relay.transcript.bridge.other_account";
				case "not_member":
					return "relay.transcript.bridge.not_member";
				case "deleted":
					return "relay.transcript.bridge.deleted";
				case "offline":
					return "relay.transcript.bridge.offline";
				default:
					return "relay.transcript.bridge.unavailable";
			}
	}
}

/**
 * The collapsed preview of a long body, cut where a reader recognises a break:
 * the last paragraph break, then line break, then space inside the limit, so
 * the fold does not land mid-word; a hard cut only for one unbroken token. A
 * boundary in the first half of the limit is ignored (folding 600 characters
 * after 40 hides more than it saves), and a cut never splits a UTF-16
 * surrogate pair (an emoji would render as a replacement glyph).
 */
/** The collapsed preview as painted: the cut plus an ellipsis, unless the cut
    already ends a sentence ("item 7.…" reads as a doubled stop; design N1).
    The worded Show more under it is the fold cue either way. */
export function bridgeBodyPreviewText(preview: { text: string; truncated: boolean }): string {
	if (!preview.truncated) return preview.text;
	return /[.!?…。！？]$/u.test(preview.text) ? preview.text : `${preview.text}…`;
}

export function bridgeBodyPreview(
	body: string,
	limit = BRIDGE_CARD_BODY_COLLAPSE_CHARS,
): { text: string; truncated: boolean } {
	if (body.length <= limit) return { text: body, truncated: false };
	const window = body.slice(0, limit);
	const floor = Math.floor(limit / 2);
	let cut = -1;
	for (const boundary of ["\n\n", "\n", " "]) {
		const at = window.lastIndexOf(boundary);
		if (at >= floor) {
			cut = at;
			break;
		}
	}
	if (cut < 0) {
		cut = limit;
		const code = body.charCodeAt(cut - 1);
		if (code >= 0xd800 && code <= 0xdbff) cut -= 1;
	}
	return { text: body.slice(0, cut).trimEnd(), truncated: true };
}
