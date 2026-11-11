/* The deferred-ack contract for settings switches (`set_model`/`set_effort`).

   WHY. A switch used to be answered only once it landed, and landing waits for
   the session's loop, so a pick made mid-tool or mid-eval hung for the whole
   step. With `defer_ok` a current runtime answers EVERY switch at once, idle or
   busy, with `detail: "queued"` — accepted, applied at the next step — BEFORE
   the switch body (and so any refusal) has run. The outcome arrives later on
   the projection as `settings_results[kind]`, keyed by `request_id`
   (`SessionProjection.settings_results` in `mobile/types.py`). An older
   runtime ignores both keys and answers when the switch lands.

   WHY A MODULE-LEVEL STORE (PR #3697 review R1-M2). The sheet that made the
   pick closes on the ack, and the refusal can land seconds later — after the
   sheet, and possibly the whole session view, has unmounted. Component state
   would lose it and the pick would read as a silent success. So pending picks
   and their refusals live here, keyed by session (`pid`), and the composer
   reconciles them against each projection repaint. It survives switching
   between conversations within the page; a page reload drops it (the
   projection repaint then still shows the model the session actually has).

   Kept out of `api.ts` so tests that mock the transport keep these real. */
import { useSyncExternalStore } from "react";

/** The `detail` of an accepted-but-not-yet-applied switch. Success, never an
    error. Mirrors `attach_client.QUEUED_DETAIL`. */
export const QUEUED_DETAIL = "queued";

/** Which setting a pick moves; the projection keys its outcomes the same way. */
export type SettingsKind = "model" | "effort";

/** One identified outcome as the runtime publishes it. */
export interface SettingsResultWire {
	request_id: string;
	ok: boolean;
	code?: string;
	message?: string;
}

export interface PendingPick {
	requestId: string;
	kind: SettingsKind;
	/** What the user picked, for the refusal line: a provider/model or a rung. */
	label: string;
}

export interface SettingsRefusal {
	requestId: string;
	kind: SettingsKind;
	label: string;
	reason: string;
}

interface SessionSettings {
	/** At most ONE pending pick per kind: settings are last-wins, so a newer
	    pick of a kind makes the older one's outcome moot. This is the bound —
	    no tuned cap — and the reason the map is keyed by kind. */
	pending: Partial<Record<SettingsKind, PendingPick>>;
	refusal: SettingsRefusal | null;
}

const EMPTY: SessionSettings = Object.freeze({ pending: {}, refusal: null }) as SessionSettings;
const sessions = new Map<string, SessionSettings>();
const listeners = new Set<() => void>();

function write(pid: string, next: SessionSettings): void {
	if (next.refusal === null && Object.keys(next.pending).length === 0) sessions.delete(pid);
	else sessions.set(pid, next);
	for (const listener of listeners) listener();
}

function read(pid: string): SessionSettings {
	return sessions.get(pid) ?? EMPTY;
}

/** The optional deferred-ack keys. One `request_id` per pick lets the
    runtime's `settings_results` be matched to this request. */
export function deferredSettingsFields(): { defer_ok: true; request_id: string } {
	return { defer_ok: true, request_id: crypto.randomUUID() };
}

/** Record a pick the runtime acked `queued`; its outcome is still to come.
    A fresh pick also clears an earlier refusal of the same kind — the user has
    already chosen again, so the old "was not applied" no longer describes it. */
export function trackSettingsPick(pid: string, pick: PendingPick): void {
	const current = read(pid);
	const refusal = current.refusal?.kind === pick.kind ? null : current.refusal;
	write(pid, {
		pending: { ...current.pending, [pick.kind]: pick },
		refusal,
	});
}

/** Fold one projection's outcomes into the session's pending picks.
    Only ids this page sent match, so another device's switches and replays of
    an outcome already consumed are ignored. Success is silent (the repaint
    shows the new model); a refusal is kept until dismissed or superseded. */
export function reconcileSettingsResults(
	pid: string,
	results: Readonly<Record<string, SettingsResultWire>> | undefined,
): void {
	if (!results) return;
	const current = sessions.get(pid);
	if (!current) return;
	let pending = current.pending;
	let refusal = current.refusal;
	let changed = false;
	for (const kind of Object.keys(pending) as SettingsKind[]) {
		const pick = pending[kind];
		const result = results[kind];
		if (!pick || !result || result.request_id !== pick.requestId) continue;
		const { [kind]: _settled, ...rest } = pending;
		pending = rest;
		changed = true;
		if (!result.ok) {
			refusal = { requestId: pick.requestId, kind, label: pick.label, reason: result.message || result.code || "" };
		}
	}
	if (changed) write(pid, { pending, refusal });
}

/** Forget a pick that an older runtime answered on landing (its reply WAS the
    outcome), or one whose request failed outright. */
export function settleSettingsPick(pid: string, requestId: string): void {
	const current = sessions.get(pid);
	if (!current) return;
	const pending = { ...current.pending };
	let changed = false;
	for (const kind of Object.keys(pending) as SettingsKind[]) {
		if (pending[kind]?.requestId === requestId) {
			delete pending[kind];
			changed = true;
		}
	}
	if (changed) write(pid, { pending, refusal: current.refusal });
}

export function dismissSettingsRefusal(pid: string): void {
	const current = sessions.get(pid);
	if (!current || current.refusal === null) return;
	write(pid, { pending: current.pending, refusal: null });
}

function subscribe(listener: () => void): () => void {
	listeners.add(listener);
	return () => listeners.delete(listener);
}

/** The session's pending picks and latest refusal, re-rendering on change. */
export function useSessionSettings(pid: string): SessionSettings {
	return useSyncExternalStore(subscribe, () => read(pid));
}

/** Test-only: drop every session's state between cases. */
export function resetSettingsDeferForTests(): void {
	sessions.clear();
	for (const listener of listeners) listener();
}
