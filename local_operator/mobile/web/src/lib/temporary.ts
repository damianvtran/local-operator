/**
 * Temporary chats — the relay's pure decisions (contract §2.3, §7.1 item 12).
 *
 * A temporary chat is live, viewable and steerable while in use, is left out
 * of the ordinary list, and is deleted by the machine after a stretch without
 * activity. Every screen asks THIS module the same four questions, so a list
 * row, the header and the menu cannot disagree about them:
 *
 *  - Is the feature here at all? (`temporaryChatsEnabled`)
 *  - Is this row/session temporary? (`isTemporary`, `sessionTemporary`)
 *  - How does the list split? (`partitionTemporary`, `temporaryGroupOpen`)
 *  - What does a row say? (`temporaryRowLabel`)
 *
 * FAIL CLOSED ON THE CAPABILITY. `features.temporary_chats` is the only gate:
 * absent (an older relay), malformed, or zero means no toggle, no subtitle
 * mark, no group, and the client never sends `temporary` or
 * `include_temporary` — the relay's input models are strict, so an ungated
 * send would be a refusal. A row's `temporary` field is never read without
 * the capability: a row is not a capability claim.
 */
import type { Capabilities, SessionSummary, TemporaryHoldCode, TemporaryInfo } from "../types";
import { TEMPORARY_HOLD_CODES } from "../types";
import type { MessageKey } from "../i18n/keys.gen";

/** The relay advertises temporary chats (`features.temporary_chats >= 1`). */
export function temporaryChatsEnabled(capabilities: Capabilities | null | undefined): boolean {
	const flag = capabilities?.features?.temporary_chats;
	return typeof flag === "number" && Number.isFinite(flag) && flag >= 1;
}

/** The wire object, when it is one this client can read; anything else is
    "not temporary". A malformed mark must not paint a countdown it cannot
    honour, and the server, not this client, decides deletion. */
export function readTemporary(value: unknown): TemporaryInfo | null {
	if (value === null || typeof value !== "object") return null;
	const v = value as Record<string, unknown>;
	if (typeof v.marked_at !== "number" || !Number.isFinite(v.marked_at)) return null;
	const idle = typeof v.idle_minutes === "number" && Number.isFinite(v.idle_minutes) ? v.idle_minutes : 0;
	const deletesAt = typeof v.deletes_at === "number" && Number.isFinite(v.deletes_at) ? v.deletes_at : null;
	const hold = typeof v.hold === "string" && (TEMPORARY_HOLD_CODES as readonly string[]).includes(v.hold)
		? (v.hold as TemporaryHoldCode)
		/* An unknown hold code is still a hold: the server stopped the clock for
		   a reason this build cannot name, so the honest copy is "can't verify". */
		: typeof v.hold === "string" ? "cannot_verify" : null;
	return { marked_at: v.marked_at, idle_minutes: idle, deletes_at: hold ? null : deletesAt, hold };
}

/** The row's mark, read only under the capability. */
export function isTemporary(row: Pick<SessionSummary, "temporary"> | undefined, enabled: boolean): boolean {
	return enabled && readTemporary(row?.temporary) !== null;
}

/** The session's mark for the session view: the events frame's own field
    when it carries one (an explicit `null` included), else the list row's. */
export function sessionTemporary(
	enabled: boolean,
	projectionValue: TemporaryInfo | null | undefined,
	row: Pick<SessionSummary, "temporary"> | undefined,
): TemporaryInfo | null {
	if (!enabled) return null;
	return projectionValue !== undefined ? readTemporary(projectionValue) : readTemporary(row?.temporary);
}

/** Split the daemon's rows into the ordinary list and the `Temporary · N`
    group, keeping the daemon's order in both. Without the capability the
    rows pass through untouched and the group is empty. */
export function partitionTemporary<T extends Pick<SessionSummary, "temporary">>(
	rows: readonly T[],
	enabled: boolean,
): { normal: T[]; temporary: T[] } {
	if (!enabled) return { normal: [...rows], temporary: [] };
	const normal: T[] = [];
	const temporary: T[] = [];
	for (const row of rows) (readTemporary(row.temporary) ? temporary : normal).push(row);
	return { normal, temporary };
}

/** The group is collapsed by default and opens on its own while it holds the
    chat the reader is in (the phone's "current chat" is the last one opened). */
export function temporaryGroupOpen(
	currentId: string | null | undefined,
	rows: readonly Pick<SessionSummary, "session_id">[],
): boolean {
	return Boolean(currentId) && rows.some((row) => row.session_id === currentId);
}

/** Whole minutes until deletion, never below one while the clock still runs:
    "deletes in 0 min" would read as already gone. `null` while held.
    Rounded DOWN, the desktop's and the terminal's rule (QA r2 Q8): `deletes_at`
    is stamped a fraction after the mark, so a ceiling read a fresh 30-minute
    chat as 31, and the three surfaces disagreed at the same moment. */
export function minutesUntilDeletion(info: TemporaryInfo, nowMs: number): number | null {
	if (info.hold || info.deletes_at === null) return null;
	return Math.max(1, Math.floor((info.deletes_at * 1000 - nowMs) / 60000));
}

/** The final stretch the countdown is emphasised in (recommendation: last 5 min). */
export const TEMPORARY_WARNING_MINUTES = 5;

export type TemporaryRowLabel =
	| { key: "relay.chrome.temporary_deletes_in"; minutes: number; warning: boolean }
	| { key: Extract<MessageKey, `relay.chrome.temporary_hold_${string}`>; minutes?: undefined; warning: false };

/** What a temporary row says: the countdown, or why the chat is being kept. */
export function temporaryRowLabel(info: TemporaryInfo, nowMs: number): TemporaryRowLabel {
	if (info.hold) return { key: holdLabelKey(info.hold), warning: false };
	const minutes = minutesUntilDeletion(info, nowMs);
	if (minutes === null) return { key: holdLabelKey("cannot_verify"), warning: false };
	return { key: "relay.chrome.temporary_deletes_in", minutes, warning: minutes <= TEMPORARY_WARNING_MINUTES };
}

/** The copy key for a hold code — one key per code, never a composed one. */
export function holdLabelKey(code: TemporaryHoldCode): Extract<MessageKey, `relay.chrome.temporary_hold_${string}`> {
	return `relay.chrome.temporary_hold_${code}`;
}

/** Making an EXISTING chat temporary asks first; an empty one has nothing to
    delete, so the toggle applies directly. Keeping never asks. */
export function makeTemporaryNeedsConfirm(transcriptLength: number): boolean {
	return transcriptLength > 0;
}

/** The refusal code a pin or project link answers for a temporary chat. */
export const SESSION_TEMPORARY_CODE = "session_temporary";
/** The 404 code for a temporary chat the machine has deleted. */
export const SESSION_TEMPORARY_DELETED_CODE = "session_temporary_deleted";
